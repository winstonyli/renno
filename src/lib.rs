pub mod cont;
pub mod env;
pub mod expr;
pub mod lexer;
pub mod machine;
pub mod parser;
pub mod plist;
pub mod span;
pub mod typecheck;
pub mod types;
pub mod util;
pub mod value;

use env::Env;
use value::{Outcome, Value};

// Default OS thread stacks (~1-8 MiB) aren't enough for pathologically
// deep source: parser::parse and typecheck::elaborate are plain recursive
// descent over native Rust stack frames -- machine::run is trampolined
// and has no such limit, but the front end does. A chain of ~1000 nested
// `let`s is enough to overflow the default stack. Rather than trying to
// bound "how deep is too deep" (fragile, and the real limit depends on
// build settings), run the whole pipeline on a dedicated thread with a
// much larger stack -- the standard fix for recursive-descent parsers
// (rustc does the same for deeply nested expressions).
const WORKER_STACK_SIZE: usize = 256 * 1024 * 1024; // 256 MiB

// parse -> typecheck -> run, catching panics (including a stack overflow
// were one to still occur, or a runtime type-check panic) as errors so a
// bad line in the REPL, or a bad program passed on the command line,
// reports cleanly instead of taking the whole process down.
//
// Returns Outcome, not Value: Value is Rc-based throughout (Env/Cont/Expr
// too) and so isn't Send, but the worker thread's result has to cross
// back to the caller's thread. Outcome is the Send-safe summary that
// makes that crossing possible without making the whole interpreter pay
// Arc's atomic-refcount cost just for this one seam.
pub fn run_source(src: &str) -> Result<Outcome, String> {
    let src = src.to_string();
    std::thread::Builder::new()
        .stack_size(WORKER_STACK_SIZE)
        .spawn(move || run_source_on_this_thread(&src).map(|v| Outcome::from(&v)))
        .expect("failed to spawn interpreter worker thread")
        .join()
        .unwrap_or_else(|_| Err("internal error: interpreter worker thread panicked".to_string()))
}

fn run_source_on_this_thread(src: &str) -> Result<Value, String> {
    let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src)?;
    let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).map_err(|e| e.1.format_error(src, &e.0))?;
    std::panic::catch_unwind(|| machine::run(&arena, elaborated, Env::prelude(), &spans)).map_err(|payload| {
        // Recover the panic's own message (downcast_ref covers both a
        // string-literal `panic!("...")` and a `panic!("{}", format!(...))`
        // -- the only two shapes this interpreter ever panics with) instead
        // of discarding it -- previously this just returned a placeholder
        // telling the caller to go look at stderr. machine::current_span,
        // read AFTER the panic unwound, names WHERE (see its own doc
        // comment for the "last expression evaluated, not necessarily the
        // exact culprit" caveat); no span at all (Perform/Apply's own
        // frame processing panicking before any Eval ever ran) just
        // leaves the bare message.
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "runtime error (no panic message available)".to_string());
        match machine::current_span() {
            Some(span) => span.format_error(src, &msg),
            None => msg,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Raw parse+run, no typecheck -- for tests exercising parser/machine
    // semantics (multi-shot, deep/shallow, arithmetic) independent of the
    // typechecker.
    fn run_untyped(src: &str) -> Value {
        let (arena, spans, root) = parser::parse(src).expect("parse failed");
        machine::run(&arena, root, Env::prelude(), &spans)
    }

    // Walks the elaborated tree looking for a runtime boundary check.
    // typecheck::coerce no longer builds a dedicated Check AST node (see
    // build_boundary_check's own doc comment) -- it desugars
    // into `let __check_tmp = ... in if ... then ... else fail(...)`, so
    // detecting one now means detecting THAT Let's own fixed binder name
    // instead of a distinct node kind. Needed because arena-indexed
    // Expr's derived Debug only prints the immediate node (children are
    // plain ExprRef indices now, not Rc<Expr>, so Debug no longer
    // recurses through them the way it used to).
    fn contains_check(arena: &expr::Arena, root: expr::ExprRef) -> bool {
        use expr::Expr;
        match &arena[root] {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Token(_) | Expr::Var(_) => false,
            Expr::ListLit(items) | Expr::Tuple(items) => items.iter().any(|i| contains_check(arena, *i)),
            Expr::Lambda(_, _, body) => contains_check(arena, *body),
            Expr::App(f, a) => contains_check(arena, *f) || contains_check(arena, *a),
            Expr::Let(var, _, val, body) => {
                var == "__check_tmp" || contains_check(arena, *val) || contains_check(arena, *body)
            }
            Expr::LetRec(bindings, body) => {
                bindings.iter().any(|(_, _, val)| contains_check(arena, *val)) || contains_check(arena, *body)
            }
            Expr::BinOp(_, l, r) => contains_check(arena, *l) || contains_check(arena, *r),
            Expr::If(c, t, e) => contains_check(arena, *c) || contains_check(arena, *t) || contains_check(arena, *e),
            Expr::Perform(_, payload) => contains_check(arena, *payload),
            Expr::Handle { body, handler } => contains_check(arena, *body) || contains_check(arena, *handler),
            Expr::MakeHandler { body, .. } => contains_check(arena, *body),
            Expr::Match(scrutinee, arms) => {
                contains_check(arena, *scrutinee)
                    || arms.iter().any(|(_, guard, body)| {
                        guard.is_some_and(|g| contains_check(arena, g)) || contains_check(arena, *body)
                    })
            }
            // Stale as of records getting a real runtime kind: Expr::Record
            // DOES survive elaboration now (see its own doc comment), so
            // this walks its field values like any other compound node.
            Expr::Record(fields) => fields.iter().any(|(_, v)| contains_check(arena, *v)),
            // Never reaches here -- this walks an ELABORATED tree, and
            // Expr::FieldAccess never survives elaboration (rewritten
            // into an ordinary get_field(...) call -- see its own doc
            // comment).
            Expr::FieldAccess(..) => unreachable!("Expr::FieldAccess never survives elaboration"),
        }
    }

    // handle { let x = perform choose(0) in x + 100 } with
    // handler choose(p, resume) -> resume(1) + resume(2)
    //
    // Multi-shot proof: `resume` is called twice from the handler body.
    // Each call replays the captured continuation (`x + 100`) with a
    // different x, independently -- expect (1+100) + (2+100) = 203.
    #[test]
    fn parses_and_runs_multi_shot_demo() {
        let src = r#"
            handle
              let x = perform choose(0) in
              x + 100
            with handler choose(p, resume) -> resume(1) + resume(2)
        "#;
        assert_eq!(run_untyped(src).as_int(), 203);
    }

    // Two *sequential* occurrences of the same effect (not multi-shot --
    // each is resumed once). deep(handler): the reinstalled handler
    // catches the second occurrence too -> x=1, y=1 -> 2.
    #[test]
    fn parses_and_runs_deep_handler() {
        let src = r#"
            handle
              let x = perform choose(0) in
              let y = perform choose(0) in
              x + y
            with deep(handler choose(p, resume) -> resume(1))
        "#;
        assert_eq!(run_untyped(src).as_int(), 2);
    }

    // Same shape, no deep(...) -- shallow (the MakeHandler default) is
    // consumed by the first occurrence; the second escapes unhandled.
    #[test]
    #[should_panic(expected = "unhandled effect: choose")]
    fn parses_and_runs_shallow_handler_panic() {
        let src = r#"
            handle
              let x = perform choose(0) in
              let y = perform choose(0) in
              x + y
            with handler choose(p, resume) -> resume(1)
        "#;
        run_untyped(src);
    }

    #[test]
    fn if_true_takes_then_branch() {
        assert_eq!(run_untyped("if 1 < 2 then 10 else 20").as_int(), 10);
    }

    #[test]
    fn if_false_takes_else_branch() {
        assert_eq!(run_untyped("if 2 < 1 then 10 else 20").as_int(), 20);
    }

    #[test]
    fn eq_and_bool_literals() {
        assert!(run_untyped("if 3 == 3 then true else false").as_bool());
    }

    #[test]
    fn eq_compares_bools_directly() {
        // Regression test: Eq used to force both operands through Int,
        // rejecting this at typecheck time even though it's a valid
        // comparison.
        let (mut arena, spans, root) = parser::parse("true == false").unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!machine::run(&arena, elaborated, Env::prelude(), &spans).as_bool());
    }

    #[test]
    fn binary_minus() {
        assert_eq!(run_untyped("5 - 3").as_int(), 2);
    }

    #[test]
    fn unary_minus_desugars_to_zero_minus_operand() {
        assert_eq!(run_untyped("-5 + 10").as_int(), 5);
    }

    #[test]
    fn minus_enables_decrementing_recursion() {
        let src = "let rec sum_down = fun n -> if n == 0 then 0 else n + sum_down(n - 1) in sum_down(5)";
        assert_eq!(run_untyped(src).as_int(), 15);
    }

    #[test]
    fn minus_rejects_non_int_operand_statically() {
        let (mut arena, spans, root) = parser::parse("true - 1").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int or Float, found Bool"), "unexpected message: {}", err.0);
    }

    #[test]
    fn multiplication_and_division() {
        assert_eq!(run_untyped("6 * 7").as_int(), 42);
        assert_eq!(run_untyped("10 / 3").as_int(), 3);
    }

    #[test]
    fn mul_div_bind_tighter_than_add_sub() {
        assert_eq!(run_untyped("2 + 3 * 4").as_int(), 14);
        assert_eq!(run_untyped("2 * 3 + 4").as_int(), 10);
        assert_eq!(run_untyped("-2 * 3").as_int(), -6);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn division_by_zero_panics() {
        run_untyped("5 / 0");
    }

    #[test]
    fn mul_rejects_non_int_operand_statically() {
        let (mut arena, spans, root) = parser::parse("true * 1").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int or Float, found Bool"), "unexpected message: {}", err.0);
    }

    #[test]
    fn modulo() {
        assert_eq!(run_untyped("7 % 3").as_int(), 1);
        assert_eq!(run_untyped("6 % 3").as_int(), 0);
        assert_eq!(run_untyped("-7 % 2").as_int(), -1);
    }

    #[test]
    fn modulo_binds_as_tightly_as_mul_div() {
        assert_eq!(run_untyped("2 + 7 % 3").as_int(), 3);
    }

    #[test]
    #[should_panic(expected = "modulo by zero")]
    fn modulo_by_zero_panics() {
        run_untyped("5 % 0");
    }

    // --- floats ---

    #[test]
    fn int_int_arithmetic_stays_int() {
        // Regression guard for the overload: two concrete Ints must still
        // produce an Int result (and Div/Mod must still truncate, and
        // Display must still print bare "2" not "2.0"), not silently
        // drift to Float now that the operators are shared.
        assert_eq!(run_untyped("7 / 2").as_int(), 3);
        assert_eq!(run_untyped("1 + 1").to_string(), "2");
    }

    #[test]
    fn float_arithmetic_and_mixed_promotion() {
        assert_eq!(run_untyped("1.5 + 2.5").as_float(), 4.0);
        assert_eq!(run_untyped("3 + 1.5").as_float(), 4.5);
        assert_eq!(run_untyped("1.5 + 3").as_float(), 4.5);
        assert_eq!(run_untyped("7.0 / 2.0").as_float(), 3.5);
    }

    #[test]
    fn float_display_always_shows_a_decimal_point() {
        // 4.0 must print as "4.0", not bare "4" -- otherwise it'd be
        // indistinguishable from Value::Int(4)'s own Display output.
        assert_eq!(run_untyped("1.5 + 2.5").to_string(), "4.0");
        assert_eq!(run_untyped("3.14").to_string(), "3.14");
    }

    #[test]
    fn unary_minus_works_on_floats() {
        assert_eq!(run_untyped("-3.5 + 1.0").as_float(), -2.5);
    }

    #[test]
    fn lt_and_eq_compare_across_int_and_float() {
        assert!(run_untyped("1.5 < 2").as_bool());
        assert!(run_untyped("1 == 1.0").as_bool());
        assert!(!run_untyped("1 == 1.5").as_bool());
    }

    #[test]
    fn is_int_and_is_float_are_mutually_exclusive() {
        assert_eq!(run_untyped("(is_int(1.5), is_float(1.5), is_int(1), is_float(1))").to_string(), "[false, true, true, false]");
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn float_division_by_zero_panics_same_as_int() {
        // Deliberately NOT IEEE754 inf/NaN -- renno's "0 divisor is
        // always a hard error" story stays uniform across both numeric
        // types instead of quietly diverging for Float.
        run_untyped("1.0 / 0.0");
    }

    #[test]
    fn add_still_rejects_non_numeric_operand_statically() {
        let (mut arena, spans, root) = parser::parse("true + 1.5").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int or Float, found Bool"), "unexpected message: {}", err.0);
    }

    #[test]
    fn int_and_float_stay_mutually_rejecting_outside_arithmetic() {
        // The overload is scoped to the arithmetic/comparison operators
        // only (typecheck::coerce_numeric) -- general type consistency
        // must NOT have been broadened alongside it, so an Int-vs-Float
        // mismatch at an ordinary annotation boundary still statically
        // rejects in both directions, same as any other two concrete
        // types would.
        let (mut arena, spans, root) = parser::parse("let f = fun x: Float -> x in f(2)").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Float, found Int"), "unexpected message: {}", err.0);

        let (mut arena, spans, root) = parser::parse("let f = fun x: Int -> x in f(2.5)").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int, found Float"), "unexpected message: {}", err.0);
    }

    #[test]
    fn dyn_sourced_value_can_cross_into_an_arithmetic_position() {
        let src = r#"
            handle
              let y = perform choose(0) in
              y + 1.5
            with handler choose(p, resume) -> resume(1)
        "#;
        assert_eq!(run_source(src).unwrap().to_string(), "2.5");
    }

    #[test]
    fn dyn_sourced_non_numeric_value_is_rejected_at_the_arithmetic_boundary() {
        let src = r#"
            handle
              let y = perform choose(0) in
              y + 1.5
            with handler choose(p, resume) -> resume(true)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("expected Int | Float, found Bool"), "unexpected message: {err}");
    }

    #[test]
    fn modulo_is_provable_in_refinements() {
        let err = parser::parse("let n: Int where n % 2 == 0 = 5 in n").unwrap_err();
        assert!(err.contains("refinement violated"), "unexpected message: {err}");
    }

    // --- juxtaposition application (f a b, alongside f(a)(b)) ---

    #[test]
    fn juxtaposition_applies_like_explicit_parens() {
        assert_eq!(run_untyped("let add = fun a -> fun b -> a + b in add 1 2").as_int(), 3);
    }

    #[test]
    fn juxtaposition_and_explicit_parens_mix_freely() {
        let src = "let add = fun a -> fun b -> a + b in add(1) 2";
        assert_eq!(run_untyped(src).as_int(), 3);
        let src2 = "let add = fun a -> fun b -> a + b in add 1 (2)";
        assert_eq!(run_untyped(src2).as_int(), 3);
    }

    #[test]
    fn juxtaposed_argument_can_be_parenthesized() {
        assert_eq!(run_untyped("let f = fun x -> x + 1 in f (2 + 3)").as_int(), 6);
    }

    #[test]
    fn juxtaposition_is_left_associative() {
        // `f a b` == `(f a) b`, not `f (a b)` -- confirmed by a function
        // that only type-checks under the LEFT grouping (b is discarded,
        // a's own value is what's returned, doubled).
        let src = "let f = fun a -> fun b -> a + a in f 5 999";
        assert_eq!(run_untyped(src).as_int(), 10);
    }

    #[test]
    fn unary_minus_still_binds_looser_than_application() {
        // `-` sits above postfix in the precedence chain (mul -> unary ->
        // postfix), so it's never an atom-starting token inside postfix's
        // juxtaposition loop -- `f -1` still parses as `f - 1` (binary
        // Sub), the same resolution Haskell/OCaml use for this exact
        // ambiguity. Confirmed by a real subtraction (5 - 1 = 4): if `-1`
        // had instead been swallowed as a juxtaposed, negated argument,
        // this would be a type error (an Int applied to nothing) or a
        // different result entirely, not 4.
        assert_eq!(run_untyped("5 - 1").as_int(), 4);
    }

    #[test]
    fn negating_a_juxtaposed_argument_needs_explicit_parens() {
        assert_eq!(run_untyped("let f = fun x -> x in f (0 - 1)").as_int(), -1);
    }

    #[test]
    fn juxtaposition_works_with_curried_builtins() {
        let src = "fold (fun acc -> fun x -> acc + x) 0 (map (fun x -> x + x) [1, 2, 3])";
        assert_eq!(run_untyped(src).as_int(), 12);
    }

    #[test]
    fn juxtaposition_accepts_bool_and_str_literal_arguments() {
        let src = "let f = fun x -> if x then 1 else 2 in f true + f false";
        assert_eq!(run_untyped(src).as_int(), 3);
        assert_eq!(run_untyped(r#"let len_of = fun s -> len(s) in len_of "hey""#).as_int(), 3);
    }

    #[test]
    fn juxtaposition_accepts_a_perform_expression_as_an_argument() {
        // Perform's own payload is "(" expr ")"-delimited, so it's safe
        // to juxtapose (unlike If/Match/Let/Fun/Data/Handle -- see the
        // next few tests): `f perform choose(0)` is one argument, ending
        // exactly at the payload's closing paren.
        let src = r#"
            handle
              let f = fun x -> x + 100 in
              f perform choose(0)
            with handler choose(p, resume) -> resume(5)
        "#;
        assert_eq!(run_untyped(src).as_int(), 105);
    }

    // A bare (unparenthesized) If/Match/Let/Fun/Data/Handle/HandlerKw is
    // NOT eligible as a juxtaposed argument -- each ends in an unbounded
    // self.expr() for its tail (an `if`'s else branch, a `let`'s body, a
    // match arm's body, a handler expression, ...) with no delimiter of
    // its own, so treating it as "just one argument" would silently
    // swallow every FURTHER juxtaposed argument meant for the outer call
    // instead of stopping at one atom (`f let x = 1 in x 2` would
    // otherwise parse as `App(f, let x=1 in (x 2))`, one argument, not
    // two). Excluding them makes that a parse error instead -- explicit
    // parens route around it, see juxtaposed_argument_can_be_parenthesized.

    #[test]
    fn bare_let_is_not_a_juxtaposable_argument() {
        assert!(parser::parse("f let x = 1 in x").is_err());
        assert!(run_untyped("let f = fun x -> x in f (let x = 1 in x)").as_int() == 1);
    }

    #[test]
    fn bare_if_is_not_a_juxtaposable_argument() {
        assert!(parser::parse("f if true then 1 else 2").is_err());
        assert!(run_untyped("let f = fun x -> x in f (if true then 1 else 2)").as_int() == 1);
    }

    #[test]
    fn bare_match_is_not_a_juxtaposable_argument() {
        assert!(parser::parse("f match x | y -> y").is_err());
    }

    #[test]
    fn bare_fun_is_not_a_juxtaposable_argument() {
        assert!(parser::parse("f fun x -> x").is_err());
    }

    #[test]
    fn bare_handle_and_handler_are_not_juxtaposable_arguments() {
        assert!(parser::parse("f handle x with h").is_err());
        assert!(parser::parse("f handler choose(p, resume) -> resume(1)").is_err());
    }

    // --- String/List primitives ---

    #[test]
    fn string_literal_and_concat() {
        assert_eq!(run_untyped(r#""hello" ++ " world""#).as_str(), "hello world");
    }

    #[test]
    fn string_escapes() {
        assert_eq!(run_untyped(r#""a\"b\\c\n""#).as_str(), "a\"b\\c\n");
    }

    #[test]
    fn len_of_string_counts_chars() {
        assert_eq!(run_untyped(r#"len("hello")"#).as_int(), 5);
    }

    #[test]
    fn list_literal_concat_and_len() {
        assert_eq!(run_untyped("len([1,2] ++ [3,4,5])").as_int(), 5);
    }

    #[test]
    fn empty_list_literal() {
        assert_eq!(run_untyped("len([])").as_int(), 0);
    }

    // --- cons (::) as an expression, not just a pattern ---

    #[test]
    fn cons_expression_prepends() {
        assert_eq!(run_untyped("1 :: 2 :: [3]").to_string(), "[1, 2, 3]");
    }

    #[test]
    fn cons_binds_looser_than_add() {
        // `1 + 2 :: [3]` is `(1 + 2) :: [3]`, not `1 + (2 :: [3])` (which
        // wouldn't even typecheck -- Int + List).
        assert_eq!(run_untyped("1 + 2 :: [3]").to_string(), "[3, 3]");
    }

    #[test]
    fn cons_enables_hand_written_map() {
        // The actual motivating case: before cons was an expression (only
        // a pattern), there was no way to build a list incrementally in
        // renno itself -- map/fold had to be native Rust-loop builtins.
        let src = "let rec my_map = fun f -> fun xs -> \
                     match xs | [] -> [] | h :: t -> f(h) :: my_map(f)(t) \
                   in my_map(fun x -> x * 2)([1, 2, 3])";
        assert_eq!(run_untyped(src).to_string(), "[2, 4, 6]");
    }

    #[test]
    fn cons_unifies_an_unconstrained_head_with_an_unconstrained_tail() {
        // Regression: today's Cons typing widens to List(Dyn) the
        // moment the two sides aren't LITERALLY equal types -- an
        // unconstrained head (Type::Var) and an unconstrained tail
        // element type (also Type::Var, likely a DIFFERENT one) are
        // never equal, so this always widened to Dyn before this task,
        // discarding precision my_map's own body depends on. Routed
        // through a match arm (not an annotated list parameter) so the
        // observable signal is coerce_numeric's own check on h2 + 1,
        // not coerce's own boundary-check insertion -- coerce only ever
        // inspects a type's TOP-LEVEL shape (Dyn/Var vs. concrete), so
        // an annotated `xs: [Int]` parameter would never notice an
        // imprecise NESTED element type either way, old widening or new
        // precision, and silently pass both (confirmed by hand: the
        // earlier version of this test, `let g = fun xs: [Int] -> xs in
        // g(build(1)([2, 3]))`, is GREEN even against the OLD Cons arm
        // -- not a real regression guard at all). Matching through
        // bind_pattern_vars's own already-fixed correlation (Task 3)
        // makes the type flow all the way to an arithmetic operator,
        // which DOES distinguish the two: confirmed this version is RED
        // against the old arm and GREEN against the new one.
        let src = "let build = fun h -> fun t -> h :: t in match build(1)([2, 3]) | [] -> 0 | h2 :: t2 -> h2 + 1";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "h2 should already be Int -- build(1)([2, 3])'s own Cons-typed element, no runtime check needed");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 2);
    }

    #[test]
    fn unifying_a_variable_with_a_shape_containing_itself_is_an_infinite_type_error() {
        // The occurs-check, exercised for real for the first time now
        // that Cons's own unify() call (this task) makes it reachable
        // via an ordinary program: x can't simultaneously be a List and
        // its own element type (x = List(x) = List(List(x)) = ...) --
        // this must be a clean static rejection, not a hang, a stack
        // overflow, or a panic somewhere unrelated.
        let src = "fun x -> x :: x";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("infinite type"), "unexpected message: {}", err.0);
    }

    #[test]
    fn cons_onto_non_list_rejected_statically() {
        // Message wording changed with the move to real unification
        // (Cons's own hand-written "expected a list, found {r_ty}" is
        // now unify()'s own generic "type mismatch: expected {t2}, found
        // {t1}", with t2 rendering as "[Dyn]" since a fresh, still-
        // unconstrained Type::Var displays identically to Dyn by design)
        // -- the rejection itself is unchanged, still a clean static
        // error, just no longer pinned to the old exact phrasing.
        let (mut arena, spans, root) = parser::parse("1 :: 2").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("type mismatch") && err.0.contains("Int"), "unexpected message: {}", err.0);
    }

    #[test]
    #[should_panic(expected = ":: expects a list on the right")]
    fn cons_onto_non_list_panics_at_runtime_for_dyn_sourced_values() {
        // `perform choose(0)` is Dyn -- passes static checking (Dyn is
        // consistent with List(Dyn)), but the handler resumes with a
        // bare Int, which fails at apply_binop instead.
        let src = "handle 1 :: perform choose(0) with handler choose(p, resume) -> resume(5)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).expect("Dyn-sourced value should not be statically rejected");
        machine::run(&arena, elaborated, Env::prelude(), &spans);
    }

    #[test]
    fn my_map_shaped_structural_recursion_is_precisely_typed() {
        // THE motivating example from the design spec's own Motivation
        // section, finally working end-to-end: pattern-binding (Task 3)
        // correlates h with xs's element type, App's new case (Task 4)
        // learns f's shape from f(h), Cons (Task 5) unifies f(h)'s
        // result with the recursive call's own result, and (this task)
        // the [] and h :: t arms combine into List(elem) instead of
        // collapsing to Dyn. The callback's own parameter is annotated
        // (`fun x: Int -> ...`, not bare `fun x -> ...`) DELIBERATELY --
        // my_map's own `f` parameter (inside my_map's own definition,
        // still fully unannotated) is what exercises this task's actual
        // mechanism, and doesn't care whether the CONCRETE function
        // passed to it has its own parameter annotated or inferred; what
        // DOES care is arithmetic on an unannotated operand, which the
        // design spec's own Non-goals already, deliberately, explicitly
        // decided NOT to upgrade ("arithmetic itself is explicitly not
        // being upgraded to real unification this round") -- an
        // unannotated `x + 1` keeps its Task-2-era runtime check
        // regardless of anything this whole plan builds, by design, so
        // asserting full check-freedom with a bare `fun x -> x + 1`
        // would test a Non-goal, not this task's own actual scope.
        let src = r#"
            let rec my_map = fun f -> fun xs ->
                match xs
                | [] -> []
                | h :: t -> f(h) :: my_map(f)(t)
            in let g = fun ys: [Int] -> ys
            in g(my_map(fun x: Int -> x + 1)([1, 2, 3]))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "my_map's own result should already be [Int] -- no runtime check needed");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[2, 3, 4]");
    }

    #[test]
    fn filter_shaped_structural_recursion_is_also_precisely_typed() {
        // A second, genuinely different hand-rolled example beyond
        // my_map itself, per the design spec's own Testing strategy --
        // confirms this isn't narrowly special-cased to my_map's exact
        // shape. Exercises If's own branch combination (this task) in
        // addition to everything my_map's own test already covers.
        //
        // Unlike my_map's test, this one does NOT assert !contains_check
        // -- confirmed (by hand-tracing a genuine failure, not assumed)
        // that it can't, for a reason outside this task's own scope. Not
        // `n`'s own arithmetic (that's annotated, same reasoning as
        // my_map's own fix, above): it's `pred`'s own USE inside
        // `if pred(h) then ...`. `pred`'s own RETURN type is genuinely
        // unknowable at the point my_filter's own body is elaborated --
        // my_filter is checked EXACTLY ONCE, monomorphically, at
        // definition time, not re-elaborated per call site, so `pred`'s
        // return type is still a bare, uninstantiated Type::Var when
        // `Expr::If`'s own PRE-EXISTING, UNCHANGED condition coercion
        // (`coerce(c2, &c_ty, &Type::Bool, ...)` -- the CONDITION's own
        // check, not the BRANCH combination this task actually rewrites)
        // runs. That coercion inserts a runtime check because a bare
        // Type::Var reaching a hardcoded-concrete-type (Bool) target is
        // exactly as blind as coerce's Type::Var-as-Dyn treatment
        // anywhere else (Task 4's own App-arm fix addressed the
        // analogous issue for coerce's PARAMETER-type target, not its
        // CONDITION-type target here -- a different call site, out of
        // this task's own "ListLit/If's branch combination/Match's
        // cross-arm" scope). This is a genuine, deeper architectural
        // ceiling of single-pass, non-specializing elaboration applied
        // to a higher-order PREDICATE specifically (a value flowing into
        // a hardcoded-target coercion, not a unification site) -- not
        // something Tasks 1-6 claim to solve. The list-combining
        // structure itself IS precisely typed (confirmed: the runtime
        // result is correct, and my_map's own analogous structure is
        // fully check-free) -- only pred's own condition-use carries
        // this separate, pre-existing limitation.
        let src = r#"
            let rec my_filter = fun pred -> fun xs ->
                match xs
                | [] -> []
                | h :: t -> if pred(h) then h :: my_filter(pred)(t) else my_filter(pred)(t)
            in let g = fun ys: [Int] -> ys
            in g(my_filter(fun n: Int -> n > 2)([1, 2, 3, 4]))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[3, 4]");
    }

    #[test]
    fn mismatched_if_branches_still_widen_to_dyn_not_a_new_rejection() {
        // Global Constraint: unify()'s failure at If/Match/ListLit falls
        // back to today's exact widening, never a new static error.
        let src = "if 1 < 2 then 1 else true";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok(), "a genuinely mismatched If branch pair must still widen to Dyn, not newly reject");
    }

    #[test]
    fn mismatched_match_arms_still_widen_to_dyn_not_a_new_rejection() {
        // Same Global Constraint, Match's own cross-arm combination
        // specifically (a separate code edit from If's, in this same
        // task) -- must not be accidentally missed.
        let src = "match 1 | 1 -> 1 | _ -> true";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok(), "genuinely mismatched Match arms must still widen to Dyn, not newly reject");
    }

    // --- gradual verification (`where` refinements) ---

    #[test]
    fn refinement_proven_at_parse_time_has_no_runtime_check() {
        // If desugar_refinement's proof succeeds, `body` is used
        // COMPLETELY UNCHANGED -- confirmed here by matching the bound
        // value with a pattern that would fail if any wrapping `if`/`fail`
        // node were still present around it.
        let src = r#"let n: Int where 0 < n = 5 in match n | 5 -> "unwrapped" | _ -> "bug""#;
        assert_eq!(run_untyped(src).as_str(), "unwrapped");
    }

    #[test]
    fn refinement_proven_false_is_a_parse_time_error() {
        // -1 is `0 - 1` (unary minus desugars, see parser::unary), not a
        // literal Expr::Int -- try_eval_closed_int has to see through that
        // for this to be proven at parse time rather than falling back to
        // a runtime check.
        let err = parser::parse("let n: Int where 0 < n = -1 in n").unwrap_err();
        assert!(err.contains("refinement violated"), "unexpected message: {err}");
        assert!(err.contains("= -1"), "unexpected message: {err}");
    }

    #[test]
    fn refinement_on_dyn_sourced_value_falls_back_to_a_runtime_check_that_passes() {
        let src = "handle (let n: Int where 0 < n = perform choose(0) in n + 1) \
                    with handler choose(p, resume) -> resume(5)";
        assert_eq!(run_untyped(src).as_int(), 6);
    }

    #[test]
    #[should_panic(expected = "refinement violated")]
    fn refinement_on_dyn_sourced_value_runtime_check_fails() {
        let src = "handle (let n: Int where 0 < n = perform choose(0) in n + 1) \
                    with handler choose(p, resume) -> resume(-5)";
        run_untyped(src);
    }

    #[test]
    fn refinement_on_lambda_parameter_passes_at_runtime() {
        // Never proven statically -- a parameter's value is whatever the
        // caller passes, unknown at definition time.
        let src = "let f = fun n: Int where 0 < n -> n * 2 in f(3)";
        assert_eq!(run_untyped(src).as_int(), 6);
    }

    #[test]
    #[should_panic(expected = "refinement violated")]
    fn refinement_on_lambda_parameter_fails_at_runtime() {
        let src = "let f = fun n: Int where 0 < n -> n * 2 in f(-3)";
        run_untyped(src);
    }

    // --- boolean operators (&&, ||, !) ---

    #[test]
    fn and_or_not_basic() {
        assert!(!run_untyped("true && false").as_bool());
        assert!(run_untyped("true || false").as_bool());
        assert!(!run_untyped("!true").as_bool());
        assert!(run_untyped("!false").as_bool());
    }

    #[test]
    fn and_binds_tighter_than_or() {
        // `false && true || true` is `(false && true) || true` = true,
        // not `false && (true || true)` = false.
        assert!(run_untyped("false && true || true").as_bool());
    }

    #[test]
    fn comparisons_bind_tighter_than_and() {
        assert!(run_untyped("1 < 2 && 3 < 4").as_bool());
    }

    // --- comparison operators (>, <=, >=, !=), sugar over </== ---

    #[test]
    fn greater_than() {
        assert!(run_untyped("5 > 3").as_bool());
        assert!(!run_untyped("3 > 5").as_bool());
    }

    #[test]
    fn less_or_equal() {
        assert!(run_untyped("5 <= 5").as_bool());
        assert!(run_untyped("4 <= 5").as_bool());
        assert!(!run_untyped("6 <= 5").as_bool());
    }

    #[test]
    fn greater_or_equal() {
        assert!(run_untyped("5 >= 5").as_bool());
        assert!(!run_untyped("5 >= 6").as_bool());
    }

    #[test]
    fn not_equal() {
        assert!(run_untyped("5 != 6").as_bool());
        assert!(!run_untyped("5 != 5").as_bool());
    }

    #[test]
    fn arrow_token_unaffected_by_gt_lexing() {
        // `->` (Arrow) is a distinct two-char token from bare `>` (Gt) --
        // regression guard that adding Gt didn't disturb Arrow's own
        // lexing (logos matches the longest token, so "->" always wins
        // over treating '-' and '>' as separate tokens).
        assert_eq!(run_untyped("let f = fun x: Int -> x + 1 in f(5)").as_int(), 6);
    }

    #[test]
    fn comparison_operators_are_provable_in_refinements() {
        // `>=` desugars into the same If shape `&&`/`||`/`!` do
        // (negate(Lt(...))), so try_eval_bool's existing If arm proves it
        // for free -- no extra work needed for these four operators to
        // compose with gradual verification's prover.
        let err = parser::parse("let n: Int where n >= 100 = 5 in n").unwrap_err();
        assert!(err.contains("refinement violated"), "unexpected message: {err}");
    }

    #[test]
    fn and_short_circuits() {
        // No handler for `choose` at all -- if `&&` evaluated the RHS
        // regardless of the LHS, this would panic with an unhandled
        // effect (caught statically here, since `run_untyped` skips
        // typecheck but the same over-approximation applies to Int/Bool
        // typed programs too -- see the doc comment on and_expr/or_expr).
        // With a real short-circuit, the RHS is never reached at runtime.
        let src = "handle (false && perform choose(0)) with handler choose(p, resume) -> fail(\"handler was invoked!\")";
        assert!(!run_untyped(src).as_bool());
    }

    #[test]
    fn or_short_circuits() {
        let src = "handle (true || perform choose(0)) with handler choose(p, resume) -> fail(\"handler was invoked!\")";
        assert!(run_untyped(src).as_bool());
    }

    #[test]
    #[should_panic(expected = "handler was invoked")]
    fn and_does_not_short_circuit_when_lhs_is_true() {
        // Sanity check for the two tests above: confirms the handler
        // WOULD fire (and this test methodology is actually meaningful)
        // when the RHS really is reached.
        let src = "handle (true && perform choose(0)) with handler choose(p, resume) -> fail(\"handler was invoked!\")";
        run_untyped(src);
    }

    #[test]
    fn boolean_operators_compose_with_refinement_predicates() {
        // The `&&` addition directly closes gradual verification's own
        // documented limitation (only `<`/`==` alone before this).
        let src = "let n: Int where 0 < n && n < 100 = 50 in n";
        assert_eq!(run_untyped(src).as_int(), 50);
    }

    #[test]
    fn compound_refinement_predicate_is_provable_at_parse_time() {
        // try_eval_bool's If arm (added specifically because `&&` desugars
        // into one) is what makes this provable at all -- without it,
        // this would only be caught by a runtime check, not a parse error.
        let err = parser::parse("let n: Int where 0 < n && n < 100 = 200 in n").unwrap_err();
        assert!(err.contains("refinement violated"), "unexpected message: {err}");
        assert!(err.contains("= 200"), "unexpected message: {err}");
    }

    // --- map/fold: structural recursion over List without let rec ---

    #[test]
    fn map_transforms_each_element() {
        assert_eq!(run_untyped("map(fun x -> x + 1)([1, 2, 3])").to_string(), "[2, 3, 4]");
    }

    #[test]
    fn map_on_empty_list() {
        assert_eq!(run_untyped("len(map(fun x -> x + 1)([]))").as_int(), 0);
    }

    #[test]
    fn fold_sums_left_to_right() {
        // f is curried (one renno arg at a time): fold applies f(acc)(item)
        // via two native machine::apply calls per element.
        assert_eq!(run_untyped("fold(fun acc -> fun x -> acc + x)(0)([1, 2, 3, 4])").as_int(), 10);
    }

    #[test]
    fn fold_on_empty_list_returns_init_unchanged() {
        assert_eq!(run_untyped("fold(fun acc -> fun x -> acc + x)(42)([])").as_int(), 42);
    }

    #[test]
    #[should_panic(expected = "unhandled effect: log")]
    fn effect_inside_map_callback_cannot_reach_an_outer_handler() {
        // Documented limitation: map/fold call their callback via
        // machine::apply, which seeds a FRESH continuation -- an effect
        // performed inside the callback can never reach a `handle` that
        // lexically wraps the map/fold call itself. Expected shape for a
        // native structural-recursion primitive (conventionally pure),
        // not something this feature attempts to fix.
        let src = "handle map(fun x -> perform log(x))([1, 2, 3]) with handler log(p, resume) -> resume(p)";
        run_untyped(src);
    }

    #[test]
    fn get_indexes_a_list() {
        assert_eq!(run_untyped("get([10, 20, 30])(0)").as_int(), 10);
        assert_eq!(run_untyped("get([10, 20, 30])(2)").as_int(), 30);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_out_of_bounds_panics() {
        run_untyped("get([10, 20, 30])(3)");
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_negative_index_panics() {
        run_untyped("get([10, 20, 30])(0 - 1)");
    }

    #[test]
    fn list_equality_is_structural() {
        assert!(run_untyped("[1, 2, 3] == [1, 2, 3]").as_bool());
        assert!(!run_untyped("[1, 2] == [1, 2, 3]").as_bool());
    }

    #[test]
    fn string_equality_compares_content() {
        assert!(run_untyped(r#""ab" == "ab""#).as_bool());
        assert!(!run_untyped(r#""ab" == "ac""#).as_bool());
    }

    #[test]
    fn concat_rejects_mismatched_types_statically() {
        let (mut arena, spans, root) = parser::parse(r#"1 ++ "a""#).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("cannot concat"), "unexpected message: {}", err.0);
    }

    // --- let rec ---

    #[test]
    fn let_rec_self_reference_recurses() {
        // Sums 1..5 by counting up -- proves `loop` inside its own body
        // resolves to itself, not an unbound Var lookup. (Predates the `-`
        // operator; minus_enables_decrementing_recursion above covers the
        // more natural counting-down shape.)
        let src = "let rec loop = fun i -> if i < 6 then i + loop(i + 1) else 0 in loop(1)";
        assert_eq!(run_untyped(src).as_int(), 15);
    }

    #[test]
    fn plain_let_with_self_reference_does_not_recurse() {
        // Regression guard: without `rec`, a same-named inner reference
        // means the OUTER `f` (unbound here, so Dyn/unknown at lookup
        // time) -- confirms `rec` is what changes resolution, not just
        // presence of the name.
        let src = "let f = 10 in let f = fun x -> f in f(1)";
        assert_eq!(run_untyped(src).as_int(), 10);
    }

    #[test]
    fn let_rec_with_annotated_function_type_typechecks_and_runs() {
        let src = "let rec loop: (Int -> Int) = fun i -> if i < 6 then i + loop(i + 1) else 0 in loop(1)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 15);
    }

    #[test]
    fn let_rec_composes_with_map() {
        // `map`'s callback itself uses `let rec` internally -- confirms
        // RecClosure works fine as a value passed through machine::apply,
        // not just when called directly from the trampoline loop.
        let src = "map(fun n -> let rec loop = fun i -> if i < n then i + loop(i + 1) else 0 in loop(1))([3, 6])";
        assert_eq!(run_untyped(src).to_string(), "[3, 15]");
    }

    // --- mutual recursion (`let rec ... and ...`) ---

    #[test]
    fn mutual_recursion_even_odd() {
        let src = "let rec is_even = fun n -> if n == 0 then true else is_odd(n - 1) \
                    and is_odd = fun n -> if n == 0 then false else is_even(n - 1) \
                    in is_even(10)";
        assert!(run_untyped(src).as_bool());
    }

    #[test]
    fn mutual_recursion_three_way_cycle() {
        // a(9) -> b(8) -> c(7) -> ... -> a(0) = 1
        let src = "let rec a = fun n -> if n == 0 then 1 else b(n - 1) \
                    and b = fun n -> if n == 0 then 2 else c(n - 1) \
                    and c = fun n -> if n == 0 then 3 else a(n - 1) \
                    in a(9)";
        assert_eq!(run_untyped(src).as_int(), 1);
    }

    #[test]
    fn single_binding_let_rec_still_self_recurses() {
        // Regression guard: plain `let rec` (no `and`) is the group.len()==1
        // case of the same machinery -- not a separate code path.
        let src = "let rec fact = fun n -> if n == 0 then 1 else n * fact(n - 1) in fact(5)";
        assert_eq!(run_untyped(src).as_int(), 120);
    }

    #[test]
    fn mutual_recursion_with_annotated_types_typechecks_and_runs() {
        let src = "let rec is_even: (Int -> Bool) = fun n -> if n == 0 then true else is_odd(n - 1) \
                    and is_odd: (Int -> Bool) = fun n -> if n == 0 then false else is_even(n - 1) \
                    in is_even(7)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!machine::run(&arena, elaborated, Env::prelude(), &spans).as_bool());
    }

    // Shared by every hand-rolled tagged-tuple list test below -- the tag
    // is matched by ordinary literal-Str comparison in pattern position,
    // not ctor-pattern sugar (there's no `data` declaration here).
    // Factored into one constant so the convention (tag spelling, curried
    // Cons) stays in exactly one place.
    const NIL_CONS_PRELUDE: &str =
        "let Nil = (\"Nil\",) in let Cons = fun h -> fun t -> (\"Cons\", h, t) in ";

    #[test]
    fn mutual_recursion_composes_with_hand_rolled_tagged_tuples_and_match() {
        let src = format!(
            "{NIL_CONS_PRELUDE} \
             let rec sum = fun l -> match l | (\"Nil\",) -> 0 | (\"Cons\", h, t) -> h + count(t)
             and count = fun l -> match l | (\"Nil\",) -> 0 | (\"Cons\", h, t) -> 1 + sum(t)
             in sum(Cons(1)(Cons(2)(Cons(3)(Nil))))"
        );
        assert_eq!(run_untyped(&src).as_int(), 5);
    }

    // --- pattern matching ---

    #[test]
    fn match_literal_picks_matching_arm() {
        let src = r#"match 2 | 1 -> "one" | 2 -> "two" | _ -> "many""#;
        assert_eq!(run_untyped(src).as_str(), "two");
    }

    #[test]
    fn match_wildcard_arm_is_fallback() {
        let src = r#"match 99 | 1 -> "one" | _ -> "many""#;
        assert_eq!(run_untyped(src).as_str(), "many");
    }

    #[test]
    fn match_nil_and_cons_recurses_over_a_list() {
        let src = "let rec sum = fun xs -> match xs | [] -> 0 | h :: t -> h + sum(t) in sum([1, 2, 3, 4])";
        assert_eq!(run_untyped(src).as_int(), 10);
    }

    #[test]
    fn match_fixed_length_list_pattern_binds_each_element() {
        assert_eq!(run_untyped("match [1, 2] | [a, b] -> a + b | _ -> 0").as_int(), 3);
    }

    #[test]
    fn match_fixed_length_list_pattern_requires_exact_length() {
        // [a, b] must NOT match a 3-element list -- falls through to the
        // wildcard arm instead of binding a/b partially.
        assert_eq!(run_untyped(r#"match [1, 2, 3] | [a, b] -> "two" | _ -> "other""#).as_str(), "other");
    }

    #[test]
    #[should_panic(expected = "match failed: no pattern matched the value")]
    fn match_with_no_matching_arm_panics() {
        run_untyped(r#"match 5 | 1 -> "x""#);
    }

    #[test]
    fn match_result_type_check() {
        // Every arm's body is Int -- confirms the elaborated Match's result
        // type is Int (not Dyn), same widen-only-on-disagreement rule as If.
        let (mut arena, spans, root) = parser::parse("let f = fun x: Int -> x + 1 in f(match 1 | 1 -> 10 | _ -> 20)").unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "Int arms should need no runtime Check at the Int-annotated call");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 11);
    }

    #[test]
    fn passthrough_identity_function_is_precisely_typed_per_call() {
        // id's parameter merely passes through untouched -- this should
        // generalize into a real polymorphic type, so id(1)'s OWN result
        // type is precisely Int (not Dyn), observable exactly the way
        // match_result_type_check observes Match's own result precision:
        // feeding it into an Int-annotated call must need NO runtime
        // boundary check.
        let src = "let id = fun x -> x in let f = fun y: Int -> y + 1 in f(id(1))";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "id(1) should already be Int -- no runtime check needed");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 2);
    }

    #[test]
    fn passthrough_identity_function_still_works_at_other_types() {
        // Same id, called at Str this time -- confirms generalization
        // means EACH call is independently instantiated, not that id got
        // pinned to Int by the previous test's own call.
        let src = r#"let id = fun x -> x in id("hi")"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_str(), "hi");
    }

    #[test]
    fn passthrough_const_generalizes_its_returned_parameter() {
        // const's SECOND parameter (y) is simply unused and stays Dyn in
        // this implementation (see the plan's own "Before you start") --
        // only the FIRST parameter (x), which is the curried chain's own
        // tail return, needs to generalize for this to be precisely typed.
        let src = r#"let const = fun x -> fun y -> x in let f = fun n: Int -> n + 1 in f(const(1)("ignored"))"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "const(1)(_) should already be Int -- no runtime check needed");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 2);
    }

    #[test]
    fn passthrough_alias_shadowing_does_not_disconnect_the_generalized_variable() {
        // Regression for a real bug: extend_generalized used to
        // auto-generalize over ANY free Type::Var it found, including one
        // that was actually the still-open outer parameter's own name --
        // so z's use here got a disconnected fresh name instead of the
        // SAME name x's own type carries, and the whole thing was
        // wrongly, statically rejected as non-numeric.
        let src = r#"let f = fun x -> let z = x in let x = "surprise" in z in f(1) + 2"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 3);
    }

    #[test]
    fn passthrough_alias_boundary_mismatch_is_caught_cleanly() {
        // Same root cause as above, the OTHER symptom. Under the bug, f's
        // wrapped type was INCOHERENT -- Fun(Var(x#0), row, Var(x#0#1)),
        // two disconnected names -- so f("oops")'s result stayed an
        // uninstantiated, unresolved Type::Var by the time it reached g's
        // boundary; coerce's early-return only guards `to` against
        // Type::Var, not `from`, so that stray Type::Var skipped building
        // a runtime check entirely and the wrong-typed Str value crashed
        // raw arithmetic with an unrelated "expected a number" panic
        // (machine.rs) instead of any kind of clean type error.
        //
        // Fixed, f's type is fully COHERENT (Fun(Var(x#0), row, Var(x#0))
        // -- one connected name), so f("oops") resolves PRECISELY to Str
        // at its own call site -- same per-call precision
        // passthrough_identity_function_is_precisely_typed_per_call checks
        // for Int -- and feeding a precisely-known Str into g's
        // Int-annotated parameter is now a genuine, provable STATIC type
        // mismatch, caught before the program ever runs. That's a
        // stronger outcome than a runtime boundary check would have been
        // (which is what a half-connected fix would produce instead), not
        // a regression: the crash is what's fixed here, and this is
        // fixed even more cleanly, one step earlier, than a bare runtime
        // check would be.
        let src = r#"let f = fun x -> let y = x in y in let g = fun n: Int -> n + 1 in g(f("oops"))"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int") && err.0.contains("Str"), "unexpected message: {}", err.0);
    }

    #[test]
    fn passthrough_generalizes_through_a_chain_of_simple_let_aliases() {
        let src = r#"
            let f = fun x -> let y = x in let z = y in z in
            let g = fun n: Int -> n + 1 in
            g(f(1))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "f(1) should already be Int through the alias chain -- no runtime check needed");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 2);
    }

    #[test]
    fn passthrough_does_not_generalize_a_parameter_used_in_arithmetic() {
        // x is used in a type-fixing operation (+), so it must stay Dyn --
        // calling it at a non-numeric type is still a runtime failure
        // (via the ordinary Dyn boundary check), not a static rejection,
        // confirming this is exactly today's pre-existing behavior,
        // untouched.
        let src = "let f = fun x -> x + 1 in f(true)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).expect("still Dyn-typed, statically accepted same as before this feature");
        let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine::run(&arena, elaborated, Env::prelude(), &spans)
        }));
        assert!(panic_result.is_err(), "f(true) should still panic at runtime (Bool isn't Int), same as before this feature");
    }

    #[test]
    fn passthrough_does_not_break_my_map_shaped_structural_recursion() {
        // Explicit non-goal regression (see the design spec's own
        // "Non-goals" and this plan's "Before you start"): a hand-rolled
        // map-shaped function must keep typechecking exactly as it does
        // today (falls back to Dyn throughout), not newly error out just
        // because this feature now exists.
        let src = r#"
            let rec my_map = fun f -> fun xs ->
                match xs
                | [] -> []
                | h :: t -> f(h) :: my_map(f)(t)
            in my_map(fun x -> x + 1)([1, 2, 3])
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[2, 3, 4]");
    }

    #[test]
    fn passthrough_annotated_alias_still_gets_a_real_boundary_check() {
        // Regression for a critical bug: an ANNOTATED alias (`let y: Int
        // = x in y`) was wrongly treated as a safe passthrough, and the
        // resulting Type::Var flowing into the Int annotation skipped
        // coerce's boundary-check-building entirely -- f("oops") used to
        // silently return the string "oops" with zero error, despite y
        // being statically declared Int. Either a static rejection or a
        // clean runtime boundary panic is an acceptable, sound fix; what
        // must NOT happen is silent success.
        let src = r#"let f = fun x -> let y: Int = x in y in f("oops")"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        match typecheck::check(&mut arena, root, &spans) {
            Err(e) => assert!(e.0.contains("Int") && e.0.contains("Str"), "unexpected static error message: {}", e.0),
            Ok(elaborated) => {
                let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    machine::run(&arena, elaborated, Env::prelude(), &spans)
                }));
                assert!(panic_result.is_err(), "f(\"oops\") must fail (statically or at runtime), not silently return a Str where Int was declared");
            }
        }
    }

    #[test]
    fn passthrough_fun_typed_alias_still_gets_a_real_contract() {
        // Same root cause, a second symptom: an annotated Fun-typed
        // alias used to skip wrap_fun_contract entirely (the Type::Var
        // from-side bypassed coerce's check-building for ANY concrete
        // to-side, not just primitives), silently losing the per-call
        // contract a (Int -> Int)-annotated value is supposed to get.
        let src = "let f = fun x -> let y: (Int -> Int) = x in y in f(fun n -> n)(5)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 5);
    }

    #[test]
    fn passthrough_generalizes_through_let_rec_too() {
        let src = "let rec id = fun x -> x in let f = fun n: Int -> n + 1 in f(id(1))";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "id(1) via let rec should already be Int -- no runtime check needed");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 2);
    }

    #[test]
    fn passthrough_two_calls_in_one_program_are_independently_precise() {
        let src = r#"
            let id = fun x -> x in
            let use_int = fun n: Int -> n + 1 in
            let use_str = fun s: Str -> s ++ "!" in
            (use_int(id(1)), use_str(id("hi")))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "both id(1) and id(\"hi\") should already be precisely typed -- no runtime checks needed for either");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[2, hi!]");
    }

    #[test]
    fn my_map_is_callable_at_two_different_element_types_in_one_program() {
        // Both callbacks are annotated (`fun x: Int -> ...`, `fun x: Str
        // -> ...`), same reasoning as Task 6's own my_map test: an
        // unannotated numeric/concat operand always keeps its Task-2-era
        // runtime check (coerce_numeric's/Concat's own Type::Var
        // treatment), a deliberate, already-settled Non-goal unrelated to
        // generalization. my_map's OWN `f` parameter (inside my_map's own
        // definition) stays fully unannotated either way, so this still
        // fully exercises what THIS test is actually for: calling the
        // SAME generalized my_map at two DIFFERENT concrete element
        // types in one program.
        let src = r#"
            let rec my_map = fun f -> fun xs ->
                match xs
                | [] -> []
                | h :: t -> f(h) :: my_map(f)(t)
            in
            let use_ints = fun ys: [Int] -> ys in
            let use_strs = fun zs: [Str] -> zs in
            (use_ints(my_map(fun x: Int -> x + 1)([1, 2, 3])), use_strs(my_map(fun x: Str -> x ++ "!")(["a", "b"])))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "both my_map calls should already be precisely typed -- no runtime checks needed for either");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[[2, 3, 4], [a!, b!]]");
    }

    #[test]
    fn a_still_open_enclosing_type_variable_is_not_wrongly_re_generalized() {
        // The exact bug class the passthrough feature's own final review
        // found and fixed (see docs/superpowers/specs/2026-09-19-
        // passthrough-polymorphism-design.md) and the concurrent
        // row-variable fix found again for EffectRow::Var: a nested
        // let-alias inside a still-open function body must NOT
        // re-generalize the outer, still-open type variable it merely
        // touches.
        let src = r#"let f = fun x -> let y: Int = x in y in f("oops")"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        // Either a clean static rejection or a clean runtime boundary
        // panic is acceptable -- what must NOT happen is silent success
        // with the wrong value (the original critical bug).
        match typecheck::check(&mut arena, root, &spans) {
            Err(e) => assert!(e.0.contains("Int") && e.0.contains("Str"), "unexpected static error: {}", e.0),
            Ok(elaborated) => {
                let panic_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    machine::run(&arena, elaborated, Env::prelude(), &spans)
                }));
                assert!(panic_result.is_err(), "f(\"oops\") must fail, not silently succeed with a Str where Int was declared");
            }
        }
    }

    #[test]
    fn match_rejects_impossible_pattern_statically() {
        let (mut arena, spans, root) = parser::parse("match 5 | true -> 1 | _ -> 2").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("can never match"), "unexpected message: {}", err.0);
    }

    // --- match guards ---

    #[test]
    fn match_guard_true_takes_the_guarded_arm() {
        assert_eq!(run_untyped(r#"match 4 | n if n > 0 -> "pos" | _ -> "other""#).as_str(), "pos");
    }

    #[test]
    fn match_guard_false_falls_through_to_next_arm() {
        assert_eq!(run_untyped(r#"match -4 | n if n > 0 -> "pos" | _ -> "other""#).as_str(), "other");
    }

    #[test]
    fn match_guard_sees_the_patterns_own_bindings() {
        assert_eq!(run_untyped(r#"match [1, 2] | [a, b] if a < b -> "asc" | _ -> "other""#).as_str(), "asc");
    }

    #[test]
    fn match_guard_falls_through_to_a_differently_shaped_later_arm() {
        // Regression shape for the fallthrough machinery itself: the failed
        // guard's arm and the arm it falls through to don't even share a
        // pattern shape (List vs Var) -- confirms retry re-matches from
        // scratch against the ORIGINAL scrutinee, not just re-checks a
        // guard on the same binding.
        assert_eq!(run_untyped(r#"match [1, 2] | [a, b] if a > b -> "desc" | xs -> "fallback""#).as_str(), "fallback");
    }

    #[test]
    fn match_guard_may_not_perform() {
        let err = parser::parse("match 1 | n if perform foo(n) -> 1 | _ -> 2").unwrap_err();
        assert!(err.contains("match guard may not perform"), "unexpected message: {err}");
    }

    #[test]
    fn match_guard_must_be_bool() {
        // The guard itself (a plain Int literal, not the Dyn-bound `n`) is
        // concretely, statically non-Bool -- rejected the same way as any
        // other concrete-type mismatch, same as If's own cond.
        let (mut arena, spans, root) = parser::parse("match 1 | n if 5 -> 1 | _ -> 0").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Bool, found Int"), "unexpected message: {}", err.0);
    }

    #[test]
    fn guarded_earlier_arm_does_not_mask_a_later_arm_as_unreachable() {
        // A bare Var pattern normally dominates everything after it (see
        // width_dominated_record_arm_reported_unreachable and friends) --
        // but a GUARDED Var might reject, so the second arm must still be
        // reachable, both statically (typechecks) and at runtime (the
        // guard rejects 5, falling through to it).
        let src = "match 5 | n if n > 100 -> 1 | n -> n + 1";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).expect("guarded arm must not mask the fallback as dead code");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 6);
    }

    #[test]
    fn unguarded_earlier_arm_still_masks_a_later_guarded_arm() {
        // The mirror case: an earlier arm with NO guard fully covers
        // everything, so a later arm is genuinely dead regardless of
        // whether that later arm itself carries a guard.
        let (mut arena, spans, root) = parser::parse("match 5 | n -> 1 | n if n > 100 -> 2").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unreachable match arm"), "unexpected message: {}", err.0);
    }

    #[test]
    fn all_arms_guarded_is_non_exhaustive() {
        // Every arm's pattern is a bare Var (normally exhaustive by
        // itself), but every one is guarded -- a guard can reject, so
        // none of them can be relied on to cover the remaining case.
        let src = "match true | b if b -> 1 | b if !b -> 0";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    // --- hand-rolled tagged tuples (a `data` declaration's replacement --
    // see typecheck.rs's own module-level history for why: `data`'s
    // pattern sugar was removed, so a same-arity tagged sum is now just an
    // ordinary tuple whose first element is a literal Str tag, matched by
    // ordinary value comparison in pattern position) ---

    #[test]
    fn hand_rolled_tagged_tuples_round_trip_through_match() {
        let src = r#"
            let None = ("None",) in
            let Some = fun x -> ("Some", x) in
            match Some(5) | ("None",) -> 0 | ("Some", x) -> x
        "#;
        assert_eq!(run_untyped(src).as_int(), 5);
    }

    #[test]
    fn hand_rolled_nullary_tag_matches_its_own_arm() {
        let src = r#"
            let None = ("None",) in
            let Some = fun x -> ("Some", x) in
            match None | ("None",) -> 0 | ("Some", x) -> x
        "#;
        assert_eq!(run_untyped(src).as_int(), 0);
    }

    #[test]
    fn self_referential_tagged_tuple_value_supports_recursive_structures() {
        // No recursive TYPE names this (Type::Tuple/Union are finite trees
        // -- see types::consistent's own doc comment), but recursive
        // VALUES need no type at all: an untyped Cons/Nil built from plain
        // tuples nests to any depth, walked here by an ordinary `let rec`.
        // Constructors are curried like every other multi-arg callable in
        // renno (fold, map): Cons(1)(rest), not Cons(1, rest).
        let src = format!(
            "{NIL_CONS_PRELUDE} \
             let rec sum = fun l -> match l | (\"Nil\",) -> 0 | (\"Cons\", h, t) -> h + sum(t) in
             sum(Cons(1)(Cons(2)(Cons(3)(Nil))))"
        );
        assert_eq!(run_untyped(&src).as_int(), 6);
    }

    #[test]
    fn constructor_argument_is_type_checked_statically() {
        // A hand-rolled "constructor" -- an ordinary typed function
        // returning a tagged tuple -- gets the same static argument
        // checking any other typed function does.
        let src = r#"let Some = fun x: Int -> ("Some", x) in Some("x")"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int, found Str"), "unexpected message: {}", err.0);
    }

    #[test]
    fn tuple_literal_and_destructure_round_trip() {
        let src = "match (1, \"a\", true) | (a, b, c) -> a";
        assert_eq!(run_source(src).unwrap().as_int(), 1);
    }

    #[test]
    fn plain_parens_without_a_comma_stay_ordinary_grouping() {
        assert_eq!(run_source("(1 + 2) * 3").unwrap().as_int(), 9);
    }

    #[test]
    fn nested_tuple_pattern_in_one_match_is_recognized_exhaustive() {
        // Regression test: a single fully-nested pattern like `((p, q),
        // x)` covers all of `((Int, Int), Int)` even though neither
        // top-level sub-pattern is a bare Var -- missing_case's
        // covers_tuple_position must recurse into the nested tuple
        // position, not just check "all top-level subpatterns are Var".
        let src = "match ((1, 2), 3) | ((p, q), x) -> p + q + x";
        assert_eq!(run_source(src).unwrap().as_int(), 6);
    }

    #[test]
    fn cons_pattern_binds_head_to_the_scrutinee_own_element_type() {
        // Regression for the FIRST of the two problems the design spec's
        // Motivation names: h used to always be Dyn regardless of the
        // scrutinee's own (here, explicitly annotated) element type --
        // h + 1 needed a runtime is_int check even though the annotation
        // already proves h is Int. After this task, h is precisely Int,
        // no check needed.
        let src = "let f = fun xs: [Int] -> match xs | [] -> 0 | h :: t -> h + 1 in f([1, 2, 3])";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "h should already be Int from xs's own annotation -- no runtime check needed");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 2);
    }

    #[test]
    fn record_pattern_still_rejects_a_genuinely_impossible_scrutinee() {
        // Regression guard: bind_pattern_vars's rewrite must not weaken
        // pattern_could_match's OWN existing static rejection -- a
        // Record pattern against a concretely-known-Int scrutinee is
        // still a compile-time error, unchanged.
        let src = "let f = fun x: Int -> match x | {y: v} -> v in f(5)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("can never match"), "unexpected message: {}", err.0);
    }

    #[test]
    fn tuple_pattern_with_wrong_arity_rejected_statically() {
        // Regression: pattern_type gives every fixed-length Pattern::List
        // the same blind List(Dyn), discarding its actual length -- and
        // consistent()'s own List/Tuple bridge has to allow ANY length
        // through precisely because it can't see it. Without
        // pattern_could_match's own Tuple-arity arm, a 2-element pattern
        // against a 3-tuple scrutinee was silently "possibly matching,"
        // never actually firing at runtime, with no diagnostic saying so.
        let src = r#"
            let t3: (Int, Int, Int) = (1, 2, 3) in
            match t3 | (a, b) -> a + b | _ -> 0
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("can never match"), "unexpected message: {}", err.0);
    }

    #[test]
    fn tuple_pattern_with_matching_arity_still_accepted() {
        let src = "let t3: (Int, Int, Int) = (1, 2, 3) in match t3 | (a, b, c) -> a + b + c";
        assert_eq!(run_source(src).unwrap().as_int(), 6);
    }

    #[test]
    fn list_pattern_of_any_length_still_accepted_against_a_real_list_type() {
        // The arity fix is Tuple-specific -- a genuine List type has no
        // fixed arity to check a pattern's length against, so this stays
        // exactly as permissive as before: falls through to the wildcard
        // at runtime, not rejected statically.
        let src = "let f = fun xs: [Int] -> match xs | [a, b] -> a + b | _ -> 0 in f([1, 2, 3])";
        assert_eq!(run_source(src).unwrap().as_int(), 0);
    }

    #[test]
    fn tuple_containing_different_opaque_tokens_is_not_consistent() {
        // Two `opaque` occurrences at DIFFERENT source positions are
        // different Type::Token singletons -- a Tuple containing one is
        // never consistent with an otherwise-identical Tuple containing
        // the other, the static counterpart of Meters(n)/Seconds(n)'s
        // own tuple values never comparing `==` at runtime either.
        use types::{consistent, Type};
        let a = Type::Tuple(std::rc::Rc::new(vec![Type::Int, Type::Token(1)]));
        let b = Type::Tuple(std::rc::Rc::new(vec![Type::Int, Type::Token(2)]));
        assert!(!consistent(&a, &b));
    }

    #[test]
    fn tuple_containing_the_same_opaque_token_is_consistent() {
        use types::{consistent, Type};
        let a = Type::Tuple(std::rc::Rc::new(vec![Type::Int, Type::Token(7)]));
        let b = Type::Tuple(std::rc::Rc::new(vec![Type::Int, Type::Token(7)]));
        assert!(consistent(&a, &b));
    }

    #[test]
    fn opaque_expression_and_tuples_compose_into_a_hand_rolled_nominal_type() {
        // `Meters` isn't a `data` block at all here -- just an ordinary
        // function returning a tagged-by-token tuple. `opaque` written in
        // its body is a literal (see Expr::Token's own doc comment): the
        // SAME token every call, since it's the same source position
        // evaluated fresh each time, not a per-call generator -- so two
        // separate Meters(_) values still destructure against the same
        // pattern.
        let src = r#"
            let Meters = fun n -> (n, opaque) in
            match Meters(5)
            | (n, t) -> n
        "#;
        let outcome = run_source(src).unwrap();
        assert_eq!(outcome.as_int(), 5);
    }

    #[test]
    fn two_opaque_tuple_values_from_the_same_function_carry_equal_tokens() {
        let src = r#"
            let Meters = fun n -> (n, opaque) in
            let a = Meters(5) in
            let b = Meters(10) in
            match a
            | (an, at) -> match b
              | (bn, bt) -> at == bt
        "#;
        let outcome = run_source(src).unwrap();
        assert!(outcome.as_bool());
    }

    #[test]
    fn two_opaque_tuple_values_from_different_functions_are_rejected_statically() {
        // Renamed and re-asserted as part of full parametric polymorphism's
        // own Task 3: tuple-pattern destructuring now correlates each
        // position with its REAL type (see Pattern::List's own rewrite
        // above), so `at`/`bt` are now precisely Token(id1)/Token(id2)
        // instead of both collapsing to Dyn. This brings tuple-destructured
        // bindings in line with directly-bound ones, which have ALWAYS
        // statically rejected comparing two provably-different opaque
        // tokens (confirmed on the pre-this-task baseline: `let a = opaque
        // in let b = opaque in a == b` already produces the identical
        // "cannot compare Token with Token" error) -- this was never a
        // deliberate runtime-permissive design choice specific to tuples,
        // just a side effect of the old bind_pattern_vars always widening
        // tuple-destructured bindings to Dyn.
        let src = r#"
            let Meters = fun n -> (n, opaque) in
            let Seconds = fun n -> (n, opaque) in
            match Meters(5)
            | (an, at) -> match Seconds(5)
              | (bn, bt) -> at == bt
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("cannot compare"), "unexpected message: {}", err.0);
    }

    #[test]
    fn tuple_type_annotation_accepts_matching_shape() {
        let src = r#"let f = fun p: (Int, Str) -> p in f((5, "hi"))"#;
        assert_eq!(run_source(src).unwrap().to_string(), "[5, hi]");
    }

    #[test]
    fn single_element_tuple_needs_a_trailing_comma_to_close() {
        // Regression: without trailing-comma support, `,` always required
        // a following expression/pattern/type, so a 1-tuple could never
        // be written at all -- `(x,)` used to be a parse error.
        let src = "match (5,) | (n,) -> n";
        assert_eq!(run_source(src).unwrap().as_int(), 5);
    }

    #[test]
    fn type_alias_names_an_arbitrary_type_expression() {
        let src = "type MyInt = Int in let f = fun x: MyInt -> x in f(5)";
        assert_eq!(run_source(src).unwrap().as_int(), 5);
    }

    #[test]
    fn type_alias_stays_in_scope_across_sibling_atoms_not_just_the_first_one() {
        // Regression: `type` pushes nothing to `atom`'s own `pending`
        // (unlike `let`/`fun`/`data`, which always do), so without
        // `saw_type_alias` tracking this separately, `type X = ... in
        // BODY` parsed BODY as a single atom_leaf() atom instead of a
        // full expr() whenever `type` was the chain's only prefix --
        // meaning the alias got restored the moment the FIRST atom
        // finished, before a LATER sibling atom in the same expression
        // (here, the juxtaposed argument) ever saw it. Both lambdas below
        // reference `MyInt`; the second is parsed in a SEPARATE atom()
        // call from the first (as the application's argument) -- if the
        // scope were too narrow, `MyInt` there would silently fall back
        // to an undeclared Data type, and the two (Int -> Int) function
        // types would then fail to unify.
        let src = "type MyInt = Int in (fun f: (MyInt -> Int) -> f(5)) (fun x: MyInt -> x)";
        assert_eq!(run_source(src).unwrap().as_int(), 5);
    }

    #[test]
    fn union_type_annotation_accepts_either_alternative() {
        let src = "type IntOrStr = Int | Str in let f = fun x: IntOrStr -> x in f(5)";
        assert_eq!(run_source(src).unwrap().as_int(), 5);
        let src2 = r#"type IntOrStr = Int | Str in let f = fun x: IntOrStr -> x in f("hi")"#;
        assert_eq!(run_source(src2).unwrap().to_string(), "hi");
    }

    #[test]
    fn union_type_annotation_rejects_neither_alternative_statically() {
        let src = "type IntOrStr = Int | Str in let f = fun x: IntOrStr -> x in f(true)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int | Str, found Bool"), "unexpected message: {}", err.0);
    }

    #[test]
    fn union_type_dyn_boundary_accepts_a_matching_alternative_at_runtime() {
        let src = r#"
            type IntOrStr = Int | Str in
            handle
              let y = perform choose(0) in
              let f = fun x: IntOrStr -> x in
              f(y)
            with handler choose(p, resume) -> resume("passed")
        "#;
        assert_eq!(run_source(src).unwrap().to_string(), "passed");
    }

    #[test]
    fn union_type_dyn_boundary_rejects_no_matching_alternative_at_runtime() {
        let src = r#"
            type IntOrStr = Int | Str in
            handle
              let y = perform choose(0) in
              let f = fun x: IntOrStr -> x in
              f(y)
            with handler choose(p, resume) -> resume(true)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("expected Int | Str, found Bool"), "unexpected message: {err}");
    }

    #[test]
    fn union_of_tuples_exhaustiveness_needs_every_alternative_covered() {
        // Unlike a single Tuple type, a Union's alternatives are
        // independent -- one arm per alternative is exhaustive even
        // though no SINGLE arm covers the whole union. This is also the
        // shape a hand-rolled `Option` replacement takes: a Union of
        // differently-arity tagged tuples, e.g. `(Str,) | (Str, Int)` for
        // `None`/`Some(Int)` -- the tag itself isn't checked statically,
        // only each alternative's arity (see missing_case's own doc
        // comment).
        let src = r#"
            type Pair = (Int,) | (Int, Int) in
            let f = fun p: Pair ->
              match p
              | (n,) -> n
            in f((5,))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    #[test]
    fn union_of_tuples_exhaustive_when_every_alternative_is_covered() {
        let src = r#"
            type Pair = (Int,) | (Int, Int) in
            let f = fun p: Pair ->
              match p
              | (n,) -> n
              | (n, m) -> n + m
            in f((5,))
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 5);
    }

    #[test]
    fn union_type_fun_alternative_gets_a_real_per_call_contract_not_just_is_fun() {
        // Regression: build_union_check used to only run each
        // alternative's bare shape predicate (is_fun for a Fun
        // alternative), never the full build_boundary_check that
        // installs wrap_fun_contract's real per-call argument/return
        // checking. `y` is callable (Int -> Int) but doesn't match F's
        // own (Int -> Bool) alternative's RETURN type -- only reachable
        // via map's native `apply` (a statically Union-typed value can't
        // be called directly -- App's typecheck only knows Fun/Dyn
        // callees), which is exactly why the mismatch used to slip
        // through silently instead of being caught at the boundary.
        let src = r#"
            type F = (Int -> Bool) | Str in
            handle
              let y = perform choose(0) in
              let g = fun h: F -> h in
              let wrapped = g(y) in
              map(wrapped)([1])
            with handler choose(p, resume) -> resume(fun a: Int -> a)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("expected Bool, found Int"), "unexpected message: {err}");
    }

    // --- records: `{x: 1, y: 2}` -- fixed-arity, per-position-typed,
    // NAMED product, a separate type from Tuple (see types::Type::Record's
    // own doc comment for why: real record ergonomics need order-
    // independent construction, which Tuple's strictly-positional
    // comparison doesn't give). A real name-keyed Value::Record at
    // runtime (see its own doc comment), which is what lets width
    // subtyping work with no per-boundary value transformation. Read
    // back either way -- `.field` access, or destructuring
    // (`{x, y}` punning for `{x: x, y: y}`). ---

    #[test]
    fn record_construction_and_destructure_round_trip() {
        let src = "match {x: 1, y: 2} | {x: a, y: b} -> a + b";
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    #[test]
    fn contains_check_recurses_into_a_records_field_values() {
        // Direct coverage for contains_check's own Expr::Record arm
        // (previously had none -- see its own doc comment history): a
        // Dyn-to-Int check nested inside a field's value expression must
        // still be found, proving the arm actually recurses into field
        // VALUES rather than trivially returning false.
        let src = r#"
            handle
              let y = perform choose(0) in
              let f = fun x: Int -> x in
              {a: f(y)}
            with handler choose(p, resume) -> resume(41)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(contains_check(&arena, elaborated));
    }

    #[test]
    fn contains_check_finds_no_check_in_a_plain_record() {
        let (mut arena, spans, root) = parser::parse("{a: 1}").unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated));
    }

    #[test]
    fn record_field_order_does_not_matter_for_construction_or_pattern() {
        // Fields are sorted by name at parse time everywhere records
        // appear -- construction written in one order, pattern written in
        // the OPPOSITE order, still line up correctly.
        let src = "match {y: 2, x: 1} | {x: a, y: b} -> a - b";
        assert_eq!(run_source(src).unwrap().as_int(), -1);
    }

    #[test]
    fn record_field_punning_works_in_construction_and_pattern() {
        // `{x, y}` means `{x: x, y: y}` on both sides.
        let src = "let x = 3 in let y = 4 in match {x, y} | {x, y} -> x * x + y * y";
        assert_eq!(run_source(src).unwrap().as_int(), 25);
    }

    #[test]
    fn record_duplicate_field_rejected_at_parse_time() {
        let err = parser::parse("{x: 1, x: 2}").unwrap_err();
        assert!(err.contains("field `x` given more than once"), "unexpected message: {err}");
    }

    #[test]
    fn record_type_duplicate_field_rejected_at_parse_time() {
        let err = parser::parse("let f = fun p: {x: Int, x: Int} -> p in f").unwrap_err();
        assert!(err.contains("field `x` given more than once"), "unexpected message: {err}");
    }

    #[test]
    fn empty_record_rejected_at_parse_time() {
        // Matches Tuple's own existing constraint, not a new one -- see
        // parse_record_fields' own doc comment: Tuple has no
        // representation for zero elements either.
        let err = parser::parse("{}").unwrap_err();
        assert!(err.contains("at least one field"), "unexpected message: {err}");
    }

    #[test]
    fn record_type_annotation_accepts_matching_fields() {
        let src = "let f = fun p: {x: Int, y: Str} -> p in f({x: 5, y: \"hi\"})";
        assert_eq!(run_source(src).unwrap().to_string(), "{x: 5, y: hi}");
    }

    #[test]
    fn record_type_annotation_rejects_mismatched_field_names_statically() {
        // Same arity, same field TYPES, different NAMES -- unlike two
        // same-shaped Tuples (always interchangeable), a Record's names
        // are part of its identity: consistent() compares them
        // positionally alongside the types (both sides always sorted).
        let src = "let f = fun p: {x: Int, y: Int} -> p in f({a: 1, b: 2})";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(
            err.0.contains("expected {x: Int, y: Int}, found {a: Int, b: Int}"),
            "unexpected message: {}",
            err.0
        );
    }

    #[test]
    fn dyn_boundary_to_record_rejects_wrong_field_names() {
        // Fixed limitation: is_record/has_field give the Dyn-to-Record
        // boundary a real, name-aware check now (build_shape_predicate's
        // own Record arm) -- a same-arity value with entirely different
        // field names is correctly rejected, not silently accepted.
        let src = r#"
            handle
              let y = perform choose(0) in
              let f = fun p: {x: Int, y: Int} -> p in
              f(y)
            with handler choose(p, resume) -> resume({a: 1, b: 2})
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("expected {x: Int, y: Int}, found Record"), "unexpected message: {err}");
    }

    #[test]
    fn dyn_boundary_to_record_accepts_a_wider_value() {
        // The other half of the same fix: width subtyping applies at the
        // Dyn boundary too, not just to statically-known values -- a
        // Dyn-sourced record with an EXTRA field the annotation never
        // asked for still passes, unchanged (no narrowing -- see
        // Pattern::Record's own doc comment for why none is needed).
        let src = r#"
            handle
              let y = perform choose(0) in
              let f = fun p: {x: Int} -> p in
              f(y)
            with handler choose(p, resume) -> resume({x: 1, y: 2})
        "#;
        assert_eq!(run_source(src).unwrap().to_string(), "{x: 1, y: 2}");
    }

    #[test]
    fn width_subtyping_accepts_a_wider_record_statically() {
        let src = r#"
            let f = fun p: {x: Int} -> p in
            match f({x: 1, y: 2}) | {x} -> x
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 1);
    }

    #[test]
    fn width_subtyping_rejects_a_record_missing_a_required_field() {
        let src = r#"
            let f = fun p: {x: Int, y: Int} -> p in
            f({x: 1})
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(
            err.0.contains("expected {x: Int, y: Int}, found {x: Int}"),
            "unexpected message: {}",
            err.0
        );
    }

    #[test]
    fn single_width_tolerant_arm_is_exhaustive_for_narrower_and_wider_records() {
        // The headline consequence of width-tolerant matching: `{x: a}`
        // alone covers EVERY Record type that includes field `x`,
        // regardless of what else that type's own alternatives carry --
        // no second arm needed the way exact-arity Tuple matching would.
        let src = r#"
            type R = {x: Int} | {x: Int, y: Int} in
            let f = fun p: R -> match p | {x: a} -> a
            in f({x: 1, y: 2}) + f({x: 5})
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 6);
    }

    #[test]
    fn record_union_of_genuinely_different_shapes_needs_both_arms() {
        // Unlike the narrower/wider case above, these two alternatives
        // are NOT one a subset of the other -- `radius` isn't in the
        // second, `side` isn't in the first -- so covering the whole
        // Union genuinely needs one arm per alternative, same as any
        // other Union whose alternatives don't overlap.
        let src = r#"
            type Shape = {kind: Str, radius: Int} | {kind: Str, side: Int} in
            let area = fun p: Shape ->
              match p
              | {radius: r} -> r * r * 3
              | {side: s} -> s * s
            in area({kind: "circle", radius: 2}) + area({kind: "square", side: 3})
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 21);
    }

    #[test]
    fn non_exhaustive_record_union_match_rejected_statically() {
        let src = r#"
            type Shape = {kind: Str, radius: Int} | {kind: Str, side: Int} in
            let area = fun p: Shape -> match p | {radius: r} -> r * r * 3
            in area({kind: "circle", radius: 2})
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    #[test]
    fn record_equality_compares_fields_order_independently() {
        assert!(run_source("{x: 1, y: 2} == {y: 2, x: 1}").unwrap().as_bool());
        assert!(!run_source("{x: 1, y: 2} == {x: 1, y: 3}").unwrap().as_bool());
    }

    #[test]
    fn is_record_and_has_field_builtins() {
        let src = r#"(is_record({x: 1}), has_field({x: 1})("x"), has_field({x: 1})("y"), is_record(5))"#;
        assert_eq!(run_untyped(src).to_string(), "[true, true, false, false]");
    }

    // --- stdlib ---

    #[test]
    fn filter_keeps_elements_the_predicate_accepts() {
        assert_eq!(run_untyped("filter(fun x -> x > 2)([1, 2, 3, 4])").to_string(), "[3, 4]");
    }

    #[test]
    fn reverse_reverses_a_list() {
        assert_eq!(run_untyped("reverse([1, 2, 3])").to_string(), "[3, 2, 1]");
    }

    #[test]
    fn zip_pairs_elements_and_stops_at_the_shorter_list() {
        assert_eq!(run_untyped(r#"zip([1, 2, 3])(["a", "b"])"#).to_string(), "[[1, a], [2, b]]");
    }

    #[test]
    fn sort_orders_by_the_given_comparator() {
        assert_eq!(run_untyped("sort(fun a -> fun b -> a <= b)([3, 1, 2])").to_string(), "[1, 2, 3]");
    }

    #[test]
    fn range_is_exclusive_of_its_end() {
        assert_eq!(run_untyped("range(0)(5)").to_string(), "[0, 1, 2, 3, 4]");
    }

    #[test]
    fn split_and_join_round_trip() {
        assert_eq!(run_untyped(r#"split("a,b,c")(",")"#).to_string(), "[a, b, c]");
        assert_eq!(run_untyped(r#"join(["a", "b", "c"])("-")"#).as_str(), "a-b-c");
    }

    #[test]
    fn trim_strips_leading_and_trailing_whitespace() {
        assert_eq!(run_untyped(r#"trim("  hi  ")"#).as_str(), "hi");
    }

    #[test]
    fn to_str_renders_the_same_text_print_would() {
        // "x = " ++ to_str(x) is renno's string interpolation -- no
        // "${...}" syntax, just this plus ordinary ++ -- so it needs to
        // agree with Display (what print/the CLI already show) exactly.
        assert_eq!(run_untyped("to_str(42)").as_str(), "42");
        assert_eq!(run_untyped(r#""n = " ++ to_str(1 + 2)"#).as_str(), "n = 3");
        assert_eq!(run_untyped("to_str([1, 2])").as_str(), "[1, 2]");
    }

    #[test]
    fn print_echoes_its_argument_unchanged() {
        // The printed side effect itself isn't asserted on here (would
        // need stdout capture) -- this locks in the other half of print's
        // contract, that it's transparent to the surrounding expression
        // regardless of the argument's shape, which is what makes
        // `print(x)` usable inline instead of needing its own statement.
        assert_eq!(run_untyped("print(41) + 1").as_int(), 42);
        assert_eq!(run_untyped(r#"print("hi")"#).as_str(), "hi");
        assert_eq!(run_untyped("print([1, 2, 3])").to_string(), "[1, 2, 3]");
    }

    #[test]
    fn dyn_boundary_to_record_reports_a_clean_error_for_a_non_record_value() {
        // Regression: build_shape_predicate's Record arm used to check
        // has_field BEFORE is_record, so a non-Record Dyn value hit
        // has_field's own panic ("expects a record and a string")
        // instead of the ordinary desugared type-error message every
        // other boundary check produces.
        let src = r#"
            handle
              let y = perform choose(0) in
              let f = fun p: {x: Int} -> p in
              f(y)
            with handler choose(p, resume) -> resume(5)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("expected {x: Int}, found Int"), "unexpected message: {err}");
    }

    #[test]
    fn record_pattern_against_an_incompatible_scrutinee_rejected_statically() {
        // Regression: pattern_type's Pattern::Record => Type::Dyn made
        // this trivially "possibly matchable" via a bare consistent()
        // check, losing the impossible-pattern diagnostic every other
        // pattern shape still gets. pattern_could_match restores it for
        // Record specifically without breaking width-tolerant matching
        // against an actual Record-typed scrutinee (covered by
        // width_subtyping_accepts_a_wider_record_statically and friends).
        let src = r#"let f = fun p: Str -> match p | {x: a} -> a | s -> s in f("hi")"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("can never match"), "unexpected message: {}", err.0);
    }

    #[test]
    fn width_dominated_record_arm_reported_unreachable() {
        // Regression: dominates() had no Pattern::Record arm, so a later
        // arm naming a superset of an earlier arm's fields went
        // undiagnosed as dead code even though width-tolerant matching
        // guarantees the earlier arm already covers it.
        let src = r#"
            type R = {x: Int} | {x: Int, y: Int} in
            let f = fun p: R -> match p | {x: a} -> a | {x: a, y: b} -> a + b
            in f({x: 5})
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unreachable match arm"), "unexpected message: {}", err.0);
    }

    #[test]
    fn record_equality_compares_actual_fields_not_the_annotated_type() {
        // A value satisfying a type (width subtyping) is a different
        // question from two values being equal -- see value_eq's own
        // doc comment. `y` is width-coerced to fit {x: Int} but keeps
        // its extra field, so it correctly does NOT equal a genuinely
        // exact-shape {x: 1}.
        let src = r#"
            handle
              let y = perform choose(0) in
              let r = {x: 1} in
              y == r
            with handler choose(p, resume) -> resume({x: 1, y: 2})
        "#;
        assert!(!run_source(src).unwrap().as_bool());
    }

    #[test]
    fn field_access_reads_the_right_field() {
        assert_eq!(run_source("let p = {x: 3, y: 4} in p.x + p.y").unwrap().as_int(), 7);
    }

    #[test]
    fn field_access_chains_and_binds_to_the_whole_application_chain() {
        // `f(5).x` means `(f(5)).x`, the same postfix binding order
        // juxtaposition/application already established -- dot is a peer
        // of application in the same left-to-right loop, not bound to
        // the argument atom alone.
        let src = "let f = fun n -> {x: n} in f(5).x";
        assert_eq!(run_source(src).unwrap().as_int(), 5);
    }

    #[test]
    fn field_access_on_a_width_subtyped_record_still_works() {
        let src = "let f = fun r: {x: Int} -> r.x in f({x: 1, y: 2})";
        assert_eq!(run_source(src).unwrap().as_int(), 1);
    }

    #[test]
    fn unknown_field_access_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("let p = {x: 3} in p.y").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("no field named `y`"), "unexpected message: {}", err.0);
    }

    #[test]
    fn field_access_on_a_non_record_type_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("5.x").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected a record, found Int"), "unexpected message: {}", err.0);
    }

    #[test]
    fn field_access_on_a_dyn_sourced_record_works_at_runtime() {
        let src = r#"
            handle
              let y = perform choose(0) in
              y.x
            with handler choose(p, resume) -> resume({x: 42})
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 42);
    }

    #[test]
    fn field_access_on_a_dyn_sourced_non_record_panics_at_runtime() {
        // Regression: a Dyn-target `.field` used to dispatch straight to
        // get_field with no shape check first, so a non-record value hit
        // get_field's own internal-assertion panic instead of the file's
        // ordinary fail()+type_name() message every other Dyn-boundary
        // Record check produces. Now goes through build_shape_check
        // first, same as any other Dyn-to-Record boundary.
        let src = r#"
            handle
              let y = perform choose(0) in
              y.x
            with handler choose(p, resume) -> resume(5)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("expected {x: Dyn}, found Int"), "unexpected message: {err}");
    }

    #[test]
    fn field_access_on_a_dyn_sourced_record_missing_the_field_panics_cleanly() {
        let src = r#"
            handle
              let y = perform choose(0) in
              y.x
            with handler choose(p, resume) -> resume({z: 1})
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("expected {x: Dyn}, found Record"), "unexpected message: {err}");
    }

    #[test]
    fn field_access_accepted_on_a_union_when_every_alternative_has_the_field() {
        let src = "let f = fun r: {x: Int} | {x: Int, y: Int} -> r.x in f({x: 1, y: 2}) + f({x: 5})";
        assert_eq!(run_source(src).unwrap().as_int(), 6);
    }

    #[test]
    fn field_access_rejected_on_a_union_when_one_alternative_lacks_the_field() {
        let (mut arena, spans, root) = parser::parse("let f = fun r: {x: Int} | {y: Int} -> r.x in f").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("no field named `x`"), "unexpected message: {}", err.0);
    }

    // --- match exhaustiveness ---

    #[test]
    fn exhaustive_bool_match_typechecks() {
        let src = "match 1 < 2 | true -> 1 | false -> 0";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    #[test]
    fn non_exhaustive_bool_match_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("match 1 < 2 | true -> 1").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    #[test]
    fn exhaustive_list_match_typechecks_even_with_dyn_scrutinee() {
        // xs is an unannotated (Dyn) param -- exhaustiveness is judged from
        // the PATTERN shapes present ([] + unconstrained h :: t), not from
        // the scrutinee's own static type, so this still needs no wildcard.
        let src = "let rec f = fun xs -> match xs | [] -> 0 | h :: t -> h in f([1, 2])";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    #[test]
    fn non_exhaustive_list_match_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("match [1, 2] | [] -> 0").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    #[test]
    fn list_match_with_restrictive_cons_head_is_not_exhaustive() {
        // `1 :: t` only covers non-empty lists whose head is 1 -- NOT
        // every non-empty list -- so this must still be rejected even
        // though a Cons pattern is present.
        let (mut arena, spans, root) = parser::parse("match [2, 3] | [] -> 0 | 1 :: t -> 1").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    #[test]
    fn wildcard_arm_always_makes_a_match_exhaustive() {
        let src = r#"match 5 | 1 -> "a" | _ -> "b""#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    // --- match reachability ---

    #[test]
    fn arm_after_a_wildcard_is_unreachable() {
        let (mut arena, spans, root) = parser::parse("match 5 | _ -> 1 | 1 -> 2").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unreachable match arm"), "unexpected message: {}", err.0);
    }

    #[test]
    fn duplicate_literal_arm_is_unreachable() {
        let (mut arena, spans, root) = parser::parse(r#"match 5 | 1 -> "a" | 1 -> "b" | _ -> "c""#).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unreachable match arm"), "unexpected message: {}", err.0);
    }

    #[test]
    fn wildcard_as_the_last_arm_is_fine() {
        let src = r#"match 5 | 1 -> "a" | 2 -> "b" | _ -> "c""#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    #[test]
    fn distinct_literals_are_all_reachable() {
        // Regression guard: dominates() must not over-fire -- different
        // Int literals must never be reported as unreachable.
        let src = r#"match 5 | 1 -> "a" | 2 -> "b" | 3 -> "c" | _ -> "d""#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    // --- source spans ---

    #[test]
    fn parse_error_reports_line_and_column() {
        // "in" missing after the let's value -- error should point at the
        // token actually found in its place. A stray ")" (not a plain
        // identifier) here specifically because juxtaposition application
        // now means an identifier right after "2" would just extend the
        // VALUE expression (`2 y` == `App(2, y)`, a syntactically valid,
        // if nonsensical, application) rather than leave "in" missing.
        let src = "let x = 1 in\nlet y = 2\n)";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 3, column 1:"), "unexpected message: {err}");
    }

    #[test]
    fn lex_error_reports_line_and_column() {
        let src = "let x = 1 in\nx @ 2";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 2, column 3:"), "unexpected message: {err}");
    }

    #[test]
    fn type_error_reports_the_offending_arguments_line_and_column() {
        // The mismatch is `true`, on line 2 -- not the whole call, and not
        // line 1 where the function itself is defined.
        let src = "let f = fun x: Int -> x + 1 in\nf(true)";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 2, column 3:"), "unexpected message: {err}");
        assert!(err.contains("expected Int, found Bool"), "unexpected message: {err}");
    }

    #[test]
    fn non_exhaustive_match_error_reports_the_matchs_own_line() {
        let src = "let f = fun n ->\n  match n < 5\n  | true -> 1\nin f(3)";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 2, column 3:"), "unexpected message: {err}");
    }

    // --- runtime panic locations ---

    #[test]
    fn unbound_variable_panic_reports_its_location() {
        // `m` is unbound -- machine::run panics inside env.rs, with no
        // Span parameter anywhere near that panic site; the location
        // still comes through via machine::current_span (set centrally
        // in run_loop's Eval step, read back after catch_unwind catches
        // the panic in lib.rs).
        let src = "let f = fun n -> n + m in\nf(3)";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 1, column 22:"), "unexpected message: {err}");
        assert!(err.contains("unbound variable: m"), "unexpected message: {err}");
    }

    #[test]
    fn unhandled_effect_panic_reports_its_location() {
        // Dyn-typed callee (via the explicit annotation) hides this from
        // typecheck's static effect-row check -- perform's own runtime
        // panic (machine.rs's `perform`) is what actually fires, still
        // located precisely at the `perform` call itself.
        let src = "let f: (Dyn -> Dyn) = fun y -> perform choose(y) in\nf(5)";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 1, column 47:"), "unexpected message: {err}");
        assert!(err.contains("unhandled effect: choose"), "unexpected message: {err}");
    }

    #[test]
    fn non_function_call_panic_blames_the_callee_not_the_argument() {
        // `y` is Dyn (sourced from a handled effect) and turns out to be an
        // Int, not callable. The naive "last thing Eval'd" fallback would
        // blame the ARGUMENT `1` (evaluated after `y`, right before the
        // panic) -- machine.rs's Frame::AppFunc/AppArg thread the callee's
        // own span through instead, so this blames `y`.
        let src = "handle\n  let y = perform choose(0) in\n  y(1)\nwith handler choose(p, resume) -> resume(5)";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 3, column 3:"), "unexpected message: {err}");
        assert!(err.contains("expected (Dyn -> Dyn), found Int"), "unexpected message: {err}");
    }

    #[test]
    fn concat_type_mismatch_blames_the_wrong_operand_when_only_one_is_bad() {
        // Both `a` and `b` are Dyn -- typecheck.rs's Concat arm only
        // inserts a runtime Check when exactly ONE side is statically Dyn
        // (see its own doc comment), so with both sides Dyn this reaches
        // apply_binop's own runtime check. `a` (Int) is the actual
        // culprit; `b` (Str) is fine -- blame `a` specifically, not `b`
        // (evaluated last) and not the whole `a ++ b`.
        let src = "handle\n  let a = perform choose(0) in\n  let b = perform choose(1) in\n  a ++ b\nwith deep(handler choose(p, resume) -> if p == 0 then resume(1) else resume(\"x\"))";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 4, column 3:"), "unexpected message: {err}");
        assert!(err.contains("++ expects two strings or two lists"), "unexpected message: {err}");
    }

    #[test]
    fn concat_type_mismatch_blames_the_whole_expression_when_both_operands_are_bad() {
        // Same shape as above, but BOTH `a` and `b` are Int -- neither
        // operand alone explains the failure, so this blames the combined
        // `a ++ b` span rather than arbitrarily picking one side.
        let src = "handle\n  let a = perform choose(0) in\n  let b = perform choose(1) in\n  a ++ b\nwith deep(handler choose(p, resume) -> if p == 0 then resume(1) else resume(2))";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 4, column 3:"), "unexpected message: {err}");
        assert!(err.contains("++ expects two strings or two lists"), "unexpected message: {err}");
    }

    #[test]
    fn division_by_zero_blames_the_divisor() {
        // `d` is Dyn but statically Int-consistent, so typecheck inserts a
        // Check(Int, d) -- that passes at runtime (0 IS an Int), so this
        // reaches apply_binop's own zero check. Blames `d` (the divisor),
        // not `10` (the dividend, which is perfectly fine).
        let src = "handle\n  let d = perform choose(0) in\n  10 / d\nwith handler choose(p, resume) -> resume(0)";
        let err = run_source(src).unwrap_err();
        assert!(err.starts_with("line 3, column 8:"), "unexpected message: {err}");
        assert!(err.contains("division by zero"), "unexpected message: {err}");
    }

    // A statically-exhaustive match can no longer fail at runtime with "no
    // pattern matched" at all: every coverage rule missing_case accepts
    // (a Var arm, both Bool literals, []/h::t, a fully-covering Tuple/
    // Union shape) is one match_pattern is ALSO guaranteed to satisfy for
    // any value of that shape -- unlike the old `data`-era ctor_tag check,
    // which proved exhaustiveness from the declared constructor SET,
    // decoupled from the scrutinee's own runtime shape. There is
    // consequently no test here for "typechecks but panics with no arm
    // matched" -- match_with_no_matching_arm_panics (above) covers the
    // panic itself, for a match that (correctly) never claimed to be
    // exhaustive in the first place.

    // --- row polymorphism ---

    // `f`'s row is a variable (`{e}`), not Dyn -- so calling it isn't a
    // Dyn-callee shrug, it's "whatever f's row turns out to be." Without
    // this annotation (see the next test), the exact same program's effect
    // is invisible to the static checker.
    const APPLY_TWICE_ROW_POLY: &str =
        "let apply_twice = fun f: (Dyn ->{e} Dyn) -> fun x: Dyn -> f(f(x)) in ";

    #[test]
    fn row_polymorphism_catches_unhandled_effect_through_higher_order_call() {
        let src = format!("{APPLY_TWICE_ROW_POLY} apply_twice(fun y -> perform choose(y))(5)");
        let (mut arena, spans, root) = parser::parse(&src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unhandled effect") && err.0.contains("choose"), "unexpected message: {}", err.0);
    }

    #[test]
    #[should_panic(expected = "unhandled effect: choose")]
    fn without_row_annotation_same_program_only_fails_at_runtime() {
        // Same shape, `f`'s row bare Dyn instead of `{e}` -- calling it
        // collapses the whole expression's row to Dyn (today's ordinary
        // higher-order fallback), so typecheck can't catch this; it only
        // fails once machine::run actually gets there. Confirms the row
        // annotation in the previous test is doing real work, not just
        // reproducing what already happened.
        let src = "let apply_twice = fun f: (Dyn -> Dyn) -> fun x: Dyn -> f(f(x)) in \
                    apply_twice(fun y -> perform choose(y))(5)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).expect("should typecheck (falls back to Dyn)");
        machine::run(&arena, elaborated, Env::prelude(), &spans);
    }

    #[test]
    fn row_polymorphic_function_runs_correctly_when_handled() {
        // f(f(x)) performs choose TWICE sequentially (not multi-shot), so
        // this needs deep(...) to catch both occurrences -- same deep/
        // shallow rule as everywhere else in this interpreter, unrelated
        // to row polymorphism itself.
        let src = format!(
            "{APPLY_TWICE_ROW_POLY} handle apply_twice(fun y -> perform choose(y))(5) \
             with deep(handler choose(p, resume) -> resume(p + 1))"
        );
        assert_eq!(run_untyped(&src).as_int(), 7); // 5 -> 6 -> 7
    }

    #[test]
    fn row_polymorphic_function_generalizes_across_uses() {
        // The SAME row-polymorphic `run_it` used twice with two DIFFERENT
        // effects, each handled independently -- if generalization were
        // missing (one shared row variable instead of a fresh instance
        // per use), there'd be no principled reason this should typecheck
        // at all, let alone produce the right answer from both branches.
        let src = r#"
            let run_it = fun f: (Dyn ->{e} Dyn) -> f(0) in
            handle
              handle
                run_it(fun x -> perform a(x)) + run_it(fun x -> perform b(x))
              with handler b(p, resume) -> resume(10)
            with handler a(p, resume) -> resume(1)
        "#;
        assert_eq!(run_untyped(src).as_int(), 11);
    }

    #[test]
    fn row_var_not_regeneralized_through_nested_let_alias() {
        // Regression test: a row variable from a still-open enclosing
        // annotation (`cb`'s own `{e}`) used to get wrongly re-generalized
        // by a plain alias-only `let` nested inside that same function's
        // body -- the exact class of bug fixed for Type::Var (see
        // extend_generalized's doc comment), just reachable through an
        // ordinary user-written row annotation instead of only through
        // passthrough inference. Before the fix, this typechecked clean
        // and only panicked at runtime, unlike the alias-free control
        // (next test) which always caught it statically -- see
        // typecheck::generalizable_row_vars, which now excludes any row
        // var still free in the enclosing ctx, so both shapes behave
        // identically.
        let src = "let f = fun cb: (Dyn ->{e} Dyn) -> let g = cb in g in \
                    f(fun y -> perform choose(y))(0)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unhandled effect") && err.0.contains("choose"), "unexpected message: {}", err.0);
    }

    #[test]
    fn row_var_not_regeneralized_through_nested_let_alias_control() {
        // Same callback, no intervening let-alias -- confirms the previous
        // test's static rejection is really row polymorphism doing its
        // job, not a coincidence of this particular program shape.
        let src = "let f = fun cb: (Dyn ->{e} Dyn) -> cb in \
                    f(fun y -> perform choose(y))(0)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unhandled effect") && err.0.contains("choose"), "unexpected message: {}", err.0);
    }

    #[test]
    fn row_var_not_regeneralized_through_nested_let_record_embedding() {
        // Same bug, different carrier: generalizable_row_vars's own doc
        // comment claims a row var reached through ANY structural
        // embedding (not just a bare alias) is excluded the same way --
        // this exercises that via a Record field instead of a plain alias,
        // so a future change to free_row_vars's Record-recursion arm that
        // regresses this doesn't slip through unnoticed.
        let src = "let f = fun cb: (Dyn ->{e} Dyn) -> let g = {cb: cb} in g.cb in \
                    f(fun y -> perform choose(y))(0)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unhandled effect") && err.0.contains("choose"), "unexpected message: {}", err.0);
    }

    #[test]
    fn str_type_annotation() {
        let src = r#"let f = fun x: Str -> x ++ "!" in f("hi")"#;
        assert_eq!(run_untyped(src).as_str(), "hi!");
    }

    #[test]
    fn list_type_annotation() {
        let src = "let g = fun xs: [Int] -> len(xs) in g([1, 2, 3])";
        assert_eq!(run_untyped(src).as_int(), 3);
    }

    #[test]
    fn handler_rejects_duplicate_binder_names() {
        let err = parser::parse("handler choose(x, x) -> x").unwrap_err();
        assert!(err.contains("different names"), "unexpected message: {err}");
    }

    // --- gradual typing ---

    #[test]
    fn fully_annotated_code_has_no_check_nodes() {
        // Both sides concrete and consistent -- coerce() should insert
        // nothing. Confirms fully-typed code pays zero runtime-check cost.
        let (mut arena, spans, root) = parser::parse("let f = fun x: Int -> x + 1 in f(41)").unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated));
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), 42);
    }

    #[test]
    fn static_type_error_rejected_before_running() {
        // Both sides concrete and inconsistent -- rejected by the checker,
        // never reaches machine::run at all.
        let (mut arena, spans, root) = parser::parse("(fun x: Int -> x + 1)(true)").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int"), "unexpected message: {}", err.0);
    }

    #[test]
    fn dyn_argument_passes_runtime_check_when_value_matches() {
        // Effects stay untyped (Dyn) -- perform's result type is unknown
        // statically. Passing it to the Int-annotated `f` inserts a runtime
        // Check(Int, y); passes here since the handler resumes with an Int.
        let src = r#"
            handle
              let y = perform choose(0) in
              let f = fun x: Int -> x + 1 in
              f(y)
            with handler choose(p, resume) -> resume(41)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(contains_check(&arena, elaborated));
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), 42);
    }

    #[test]
    #[should_panic(expected = "type error: expected Int, found Bool")]
    fn dyn_argument_fails_runtime_check_when_value_mismatches() {
        // Same shape, but the handler resumes with a Bool -- the inserted
        // Check catches the mismatch at the effect/typed-code boundary.
        let src = r#"
            handle
              let y = perform choose(0) in
              let f = fun x: Int -> x + 1 in
              f(y)
            with handler choose(p, resume) -> resume(true)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        machine::run(&arena, elaborated, Env::prelude(), &spans);
    }

    #[test]
    fn dyn_to_fun_boundary_rejects_closure_with_wrong_return_type() {
        // Regression test: matches_type used to accept ANY callable for
        // Type::Fun regardless of its actual signature, so a Dyn-origin
        // closure with the wrong return type would silently pass the
        // boundary check and only fail later with an unrelated panic (or
        // not at all). wrap_fun_contract now re-checks the result of every
        // call through the boundary, so this fails right at the call site.
        let src = r#"
            handle
              let f = perform choose(0) in
              let g: (Int -> Int) = f in
              g(5) + 1
            with handler choose(p, resume) -> resume(fun x -> true)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine::run(&arena, elaborated, Env::prelude(), &spans)
        }));
        assert!(result.is_err(), "expected a panic from the return-type contract check");
    }

    #[test]
    fn handle_with_non_handler_value_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("handle 1 with 5").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected a handler value"), "unexpected message: {}", err.0);
    }

    // --- closed effect-row typing ---

    #[test]
    fn truly_unhandled_effect_rejected_statically() {
        // No `handle` anywhere -- previously this would only fail at
        // runtime, inside machine::run, via perform()'s own panic. Now
        // caught by typecheck::check before anything executes.
        let (mut arena, spans, root) = parser::parse("perform choose(0)").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unhandled effect") && err.0.contains("choose"), "unexpected message: {}", err.0);
    }

    #[test]
    fn fully_handled_program_typechecks() {
        let src = r#"
            handle
              let x = perform choose(0) in
              x + 100
            with handler choose(p, resume) -> resume(1) + resume(2)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    #[test]
    fn nested_handlers_discharge_different_effects() {
        let src = r#"
            handle
              let x = perform choose(0) in
              handle
                perform log(x)
              with handler log(p, resume) -> resume(0)
            with handler choose(p, resume) -> resume(1)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).expect("both effects are handled, should typecheck");
        machine::run(&arena, elaborated, Env::prelude(), &spans);
    }

    #[test]
    fn partially_handled_program_flags_remaining_effect() {
        // "choose" is handled, "log" is not -- the error should name the
        // effect that's actually still open, not the one that was handled.
        let src = r#"
            handle
              let x = perform choose(0) in
              perform log(x)
            with handler choose(p, resume) -> resume(1)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("log"), "unexpected message: {}", err.0);
        assert!(!err.0.contains("choose"), "handled effect wrongly reported: {}", err.0);
    }

    #[test]
    fn dyn_sourced_call_falls_back_permissively() {
        // `f` is an unannotated (Dyn) lambda parameter, so calling it can't
        // be proven to perform any particular set of effects -- the whole
        // expression's row degrades to Dyn rather than a known-unhandled
        // set. Static check can't reject it, but the real unhandled effect
        // still panics at runtime exactly as before this feature existed.
        let src = "(fun f -> f(0))(fun x -> perform mystery(x))";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated =
            typecheck::check(&mut arena, root, &spans).expect("Dyn-sourced call should not be statically rejected");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine::run(&arena, elaborated, Env::prelude(), &spans)
        }));
        assert!(result.is_err(), "expected the runtime unhandled-effect panic as a fallback");
    }

    #[test]
    fn deeply_nested_let_chain_does_not_overflow_the_stack() {
        // Regression test: a long enough chain of nested `let`s used to
        // crash the process with STATUS_STACK_OVERFLOW, from three
        // independent native-recursion sources -- all three are now fixed
        // (see the next test for the full breakdown): parser::atom and
        // typecheck::elaborate flatten a Let/Fun chain iteratively instead
        // of recursing, plist.rs's custom iterative Drop fixes the
        // typecheck Ctx (and Env) teardown cost, and Expr moved into a
        // flat arena (expr.rs) so dropping the whole tree is one Vec
        // drop -- O(n), no recursion -- instead of a recursive walk
        // through nested Rc<Expr> fields. run_source's large-stack thread
        // remains in place as defense in depth for deeply nested
        // App/BinOp/If/Handle (which branch rather than chain, so the
        // flattening above doesn't apply to them, though they're far less
        // likely to reach this depth in realistic code).
        let n = 5000;
        let mut src = String::from("let x0 = 0 in ");
        for i in 1..n {
            src.push_str(&format!("let x{i} = x{} + 1 in ", i - 1));
        }
        src.push_str(&format!("x{}", n - 1));
        let result = run_source(&src).expect("should not crash or error");
        assert_eq!(result.as_int(), (n - 1) as i64);
    }

    #[test]
    fn let_chain_is_o1_stack_construction_and_teardown() {
        // Full proof, no forget() needed anymore: parsing, typechecking,
        // AND dropping everything afterward, all on the default stack, at
        // 10x the depth that used to crash a 256 MiB thread. Construction
        // was fixed by chain flattening (parser.rs, typecheck.rs); Ctx/Env
        // teardown by plist.rs's custom Drop; Expr's own teardown by
        // moving the AST into a flat arena (expr.rs) instead of a linked
        // Rc<Expr> tree -- dropping the arena is dropping one Vec.
        let n = 50_000;
        let mut src = String::from("let x0 = 0 in ");
        for i in 1..n {
            src.push_str(&format!("let x{i} = x{} + 1 in ", i - 1));
        }
        src.push_str(&format!("x{}", n - 1));
        let (mut arena, spans, root) = parser::parse(&src).expect("parsing should not overflow the stack");
        let elaborated = typecheck::check(&mut arena, root, &spans).expect("typechecking should not overflow the stack");
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), (n - 1) as i64);
        // arena, and everything in it, drops normally here.
    }

    #[test]
    fn deep_cont_chain_drops_without_overflowing_the_stack() {
        // Cont(Rc<ContNode>) has the exact same shape as plist.rs's
        // PList<T> (a struct wrapping an Rc pointing at "cargo + rest"),
        // so it has the same recursive-Drop risk: a long enough chain's
        // default field-by-field Drop recurses through `rest` and can
        // overflow the stack, independent of how the chain was built.
        // Built directly here (bypassing the parser/machine entirely) to
        // isolate exactly that -- cont.rs's custom iterative Drop
        // (Rc::try_unwrap-based, same technique as PList's) is what makes
        // this not crash.
        let n = 50_000usize;
        let mut chain = cont::Cont::nil();
        for _ in 0..n {
            chain = cont::Cont::cons(cont::Frame::PerformPayload { effect: "e".to_string() }, chain);
        }
        drop(chain);
    }

    #[test]
    fn unannotated_lambda_param_still_works_arithmetically() {
        // Regression: x is now Type::Var by default (not Type::Dyn) --
        // coerce_numeric needs its own Type::Var arm or this newly,
        // wrongly rejects statically the moment the default changes.
        let src = "let f = fun x -> x + 1 in f(5)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 6);
    }

    #[test]
    fn unannotated_lambda_param_still_supports_field_access() {
        // Same root cause, FieldAccess's own target-type match.
        let src = "let f = fun r -> r.x in f({x: 5})";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 5);
    }

    #[test]
    fn pattern_could_match_accepts_record_and_list_patterns_against_an_unconstrained_var() {
        // Regression for a confirmed existing gap: pattern_could_match's
        // explicit Type::Dyn arms for Record/List patterns had no
        // Type::Var counterpart, so a Record/List pattern against an
        // unannotated (now Type::Var) parameter would be wrongly,
        // statically rejected as "can never match".
        let src = r#"let get_x = fun r -> match r | {x: v} -> v | _ -> 0 in get_x({x: 5})"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 5);
    }

    #[test]
    fn calling_an_unannotated_parameter_learns_its_function_shape() {
        // Regression for the SECOND correlation my_map-shaped code
        // needs: f is an unannotated parameter (Type::Var, not Fun-
        // shaped yet) -- calling it as f(x) must learn f's own param/
        // return types from how it's actually used, not just fall back
        // to Dyn the way today's Type::Dyn-callee branch does.
        let src = "let twice = fun f -> fun x -> f(f(x)) in twice(fun n: Int -> n + 1)(5)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 7);
    }

    #[test]
    fn calling_something_that_resolves_to_a_non_function_is_still_a_hard_error() {
        // Regression guard: the new Type::Var-callee case must still
        // reject calling something that turns out concretely
        // non-callable, same as today's existing "cannot call" rejection
        // -- it must not silently succeed just because unify() is now
        // involved.
        let src = "let apply = fun f -> fun x -> f(x) in apply(5)(1)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("cannot call") || err.0.contains("type mismatch"), "unexpected message: {}", err.0);
    }

    #[test]
    fn inferred_higher_order_param_accepts_a_narrower_callback_contravariantly() {
        // This is the SAME test program the design spec's own
        // motivating example uses, and the SAME one an earlier plan
        // (full parametric polymorphism) confirmed was rejected and
        // deliberately documented as a known, pre-existing ceiling
        // (search src/lib.rs for this exact test name before this
        // change -- it used to assert REJECTION). f's own inferred
        // parameter type ends up requiring {x: Int, y: Int} (from the
        // literal constructed inside use_it's own body); the argument
        // only requires {x: Int} -- narrower, and now correctly
        // accepted via contravariant Fun-parameter subtyping.
        let src = r#"
            let use_it = fun f -> f({x: 1, y: 2}) in
            use_it(fun r: {x: Int} -> r.x)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 1);
    }

    #[test]
    fn explicitly_annotated_higher_order_param_accepts_a_narrower_callback_too() {
        // Companion to the test above, with f's OWN parameter type
        // explicitly annotated instead of inferred -- confirms
        // coerce()'s own fits() integration works independent of
        // inference, the same way the original ceiling was confirmed
        // to predate the whole full-parametric-polymorphism feature by
        // reproducing on an explicitly-annotated equivalent.
        let src = r#"
            let use_it = fun f: (({x: Int, y: Int}) -> Int) -> f({x: 1, y: 2}) in
            use_it(fun r: {x: Int} -> r.x)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 1);
    }

    #[test]
    fn unify_fits_still_rejects_a_callback_requiring_a_field_that_does_not_exist() {
        // Regression guard: unify_fits must not become MORE permissive
        // than fits() itself -- a callback requiring a field the
        // required record doesn't have AT ALL must still be a hard
        // rejection, not silently accepted (or silently discarded, the
        // way the OLD best-effort unify() call in this arm used to
        // swallow ANY failure regardless of cause). Careful with the
        // exact shape: {y: Int} would NOT work for this test -- it's a
        // genuinely valid narrower requirement of {x: Int, y: Int} (see
        // fits_rejects_a_callback_requiring_a_field_that_does_not_
        // exist_in_what_is_required's own comment, Task 1) -- this uses
        // {z: Int}, a field {x: Int, y: Int} doesn't have at all.
        let src = r#"
            let use_it = fun f -> f({x: 1, y: 2}) in
            use_it(fun r: {z: Int} -> r.z)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("type mismatch"), "unexpected message: {}", err.0);
    }

    // --- Final whole-branch review regressions (found after all 7 tasks
    // landed -- each is a valid program that worked before this whole
    // feature and broke after, invisible to any single task's own
    // scoped review because each needs a LATER task's own change to
    // become reachable) ---

    #[test]
    fn coercing_a_dyn_sourced_higher_order_value_does_not_panic() {
        // Regression: Task 4's own param_ty_resolved fix (Expr::App's
        // Type::Fun arm) hands coerce a REAL, resolved Fun(Var, _, Var)
        // shape once a callee's own parameter position has been learned
        // via unification -- coerce's own Type::Fun branch of
        // build_boundary_check recurses into wrap_fun_contract, which
        // recurses again for the return type, landing on a bare
        // Type::Var. build_boundary_check's own Type::Var arm was still
        // `unreachable!()` (a leftover from when Type::Var could never
        // reach this deep, before real unification existed) -- so this
        // program made the typechecker itself PANIC, not just reject.
        let src = r#"
            let apply = fun f -> fun x -> f(x) in
            let d: Dyn = fun n: Int -> n + 1 in
            apply(d)(5)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 6);
    }

    #[test]
    fn tuple_destructuring_through_an_unannotated_parameter_still_works() {
        // Regression: Pattern::List is ALSO renno's tuple pattern
        // (`(a, b)` desugars to it) -- bind_pattern_vars's own
        // Type::Var(_) arm used to force an unconstrained scrutinee into
        // a List shape (indistinguishable from "this IS a list
        // pattern"), which is simply the wrong guess whenever the
        // pattern is actually destructuring a tuple, and unrecoverable
        // once made (infer.subst never shrinks). An actual List-shaped
        // scrutinee still gets its real correlation via Cons's own arm.
        let src = r#"let fst = fun p -> match p | (a, b) -> a | _ -> 0 in fst((1, 2))"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 1);
    }

    #[test]
    fn a_failed_tolerant_unify_does_not_leak_partial_bindings_if() {
        // Regression: unify() mutates infer.subst as it recurses with no
        // rollback -- unifying Tuple([Var(x), Int]) against
        // Tuple([Str, Bool]) binds x := Str at position 0, THEN fails at
        // position 1. Expr::If's own fallback-to-Dyn correctly catches
        // the overall Err and widens the IF's own result to Dyn, exactly
        // as the Global Constraint requires -- but x itself was already,
        // permanently bound to Str by the failed attempt, so the
        // rejection just relocates to x's own later use. A compound
        // (non-scalar) mismatch is essential here: a scalar mismatch
        // (e.g. `if _ then 1 else true`) fails on unify's very FIRST
        // comparison, making zero bindings, and can never exercise this.
        let src = r#"let g = fun x -> if 1 < 2 then (x, 1) else ("s", true) in g(5)"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[5, 1]");
    }

    #[test]
    fn a_failed_tolerant_unify_does_not_leak_partial_bindings_match() {
        // Same bug as the If-shaped test above, Match's own cross-arm
        // combination specifically -- a separate code edit from If's, in
        // this same task (Task 6), must not be accidentally missed.
        let src = r#"let g = fun x -> match 1 | 1 -> (x, 1) | _ -> ("s", true) in g(5)"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[5, 1]");
    }

    #[test]
    fn a_failed_tolerant_unify_does_not_leak_partial_bindings_listlit() {
        // Same bug, ListLit's own element combination.
        let src = r#"let x = 5 in [(x, 1), ("s", true)]"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[[5, 1], [s, true]]");
    }

    #[test]
    fn calling_the_same_inferred_callback_twice_at_a_wider_type_still_widens() {
        // Regression: Expr::App dispatched on the UNRESOLVED f_ty read
        // straight out of Ctx -- if an earlier call already bound f's
        // own type variable to a concrete Fun, a later call still saw
        // the raw, syntactically-unresolved Type::Var and took the
        // Type::Var(_) arm (no coerce() at all, so no width-subtyping
        // tolerance) instead of the Type::Fun arm's own subtyping-aware
        // handling.
        let src = r#"
            let g = fun f ->
                let a = f({x: 1}) in
                f({x: 1, y: 2})
            in g(fun r: {x: Int} -> r.x)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 1);
    }

    #[test]
    fn calling_an_inferred_parameter_learned_to_be_non_int_is_a_static_rejection() {
        // Companion to the resolve-before-dispatch fix above: once a
        // callee's own type variable is bound (here, to Int, via the
        // App's own Type::Var(_) arm learning it from f(1)), a LATER,
        // clearly-incompatible call site is now correctly caught
        // STATICALLY (via the Type::Fun arm's own coerce/consistent
        // check) instead of reaching a raw, un-typed runtime panic.
        let src = r#"let g = fun f -> f(1) in g(5)"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("type mismatch"), "unexpected message: {}", err.0);
    }

    #[test]
    fn unify_fits_accepts_a_tuple_argument_for_a_dyn_list_parameter_like_coerce_does() {
        // Final whole-branch review, Critical: unify() has no arm bridging
        // Type::List(Dyn) and Type::Tuple, but consistent() does (see
        // types.rs, the Type::List/Type::Tuple arm with the Dyn check) --
        // so coerce() (via fits() -> consistent()) correctly accepts a
        // tuple argument at a [Dyn] parameter, but unify_fits used to hard-
        // reject the very same call right after, since its old catch-all
        // delegated to plain unify() with no consistent() rescue. This
        // program worked before this whole plan and is also the ONLY test
        // in this file where unify_fits's own rejection logic (rather than
        // coerce()'s) is what's actually being exercised, since coerce()
        // already accepts this call before unify_fits ever runs.
        let src = r#"
            let f = fun xs: [Dyn] -> len(xs) in f((1, 2))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 2);
    }

    #[test]
    fn fits_accepts_a_callback_requiring_a_narrower_record_than_required() {
        // The actual motivating case: a Fun requiring {x: Int, y: Int}
        // as its own param is satisfied by an actual Fun that only
        // requires {x: Int} -- contravariant parameter subtyping.
        // Swapped on purpose: fits(required, actual) checks
        // fits(actual_param, required_param), not the other way
        // around -- get this backwards and the test below (which
        // checks the REJECT direction) would also wrongly pass.
        use types::{fits, EffectRow, Type};
        let required = Type::Fun(
            std::rc::Rc::new(Type::Record(std::rc::Rc::new(vec![
                ("x".to_string(), Type::Int),
                ("y".to_string(), Type::Int),
            ]))),
            EffectRow::Dyn,
            std::rc::Rc::new(Type::Int),
        );
        let actual = Type::Fun(
            std::rc::Rc::new(Type::Record(std::rc::Rc::new(vec![("x".to_string(), Type::Int)]))),
            EffectRow::Dyn,
            std::rc::Rc::new(Type::Int),
        );
        assert!(fits(&required, &actual));
    }

    #[test]
    fn fits_rejects_a_callback_requiring_a_field_that_does_not_exist_in_what_is_required() {
        // The narrower callback must still name a field that's ACTUALLY
        // present in what the wider requirement supplies. Careful with
        // the exact shape here: {y: Int} alone WOULD correctly fit
        // {x: Int, y: Int} (it's a genuinely valid narrower requirement
        // -- any value satisfying {x, y} also has field y) -- that's
        // NOT the case this test is for. This test uses {z: Int}, a
        // field {x: Int, y: Int} doesn't have AT ALL, to test genuine
        // non-substitutability, not just "a different single field."
        use types::{fits, EffectRow, Type};
        let required = Type::Fun(
            std::rc::Rc::new(Type::Record(std::rc::Rc::new(vec![
                ("x".to_string(), Type::Int),
                ("y".to_string(), Type::Int),
            ]))),
            EffectRow::Dyn,
            std::rc::Rc::new(Type::Int),
        );
        let actual = Type::Fun(
            std::rc::Rc::new(Type::Record(std::rc::Rc::new(vec![("z".to_string(), Type::Int)]))),
            EffectRow::Dyn,
            std::rc::Rc::new(Type::Int),
        );
        assert!(!fits(&required, &actual));
    }

    #[test]
    fn fits_accepts_a_callback_whose_return_type_is_narrower_than_required() {
        // Final whole-branch review, Important #2: every other fits()/
        // coerce()/unify_fits() test in this plan uses Int on BOTH sides
        // of the return type, so a covariant-vs-contravariant bug in the
        // return-type check specifically (e.g. fits(req_ret, act_ret)
        // silently swapped to fits(act_ret, req_ret)) would pass the whole
        // suite undetected. Covariant return: a callback promising to
        // return an Int (a MORE PRECISE, narrower promise) can stand in
        // for one that only promised to return Dyn (a WIDER, less precise
        // promise) -- the opposite direction from the param check.
        use types::{fits, EffectRow, Type};
        let required = Type::Fun(std::rc::Rc::new(Type::Int), EffectRow::Dyn, std::rc::Rc::new(Type::Dyn));
        let actual = Type::Fun(std::rc::Rc::new(Type::Int), EffectRow::Dyn, std::rc::Rc::new(Type::Int));
        assert!(fits(&required, &actual));
    }

    #[test]
    fn fits_rejects_a_callback_whose_return_type_is_wrong() {
        // Companion rejection case for the covariance test above.
        use types::{fits, EffectRow, Type};
        let required = Type::Fun(std::rc::Rc::new(Type::Int), EffectRow::Dyn, std::rc::Rc::new(Type::Int));
        let actual = Type::Fun(std::rc::Rc::new(Type::Int), EffectRow::Dyn, std::rc::Rc::new(Type::Str));
        assert!(!fits(&required, &actual));
    }

    #[test]
    fn eq_still_rejects_fun_values_that_fit_but_are_not_consistent() {
        // Regression guard for this whole plan's own Global Constraint:
        // consistent()'s own semantics must not change anywhere else.
        // Two Fun-typed values that ARE fits()-compatible (one accepts
        // a narrower record than the other requires) but are NOT
        // consistent()-equal (different, non-matching field sets) must
        // still be rejected by ==, exactly as they already are today --
        // fits()-style tolerance must never leak into consistent()'s
        // own equivalence relation.
        let src = r#"
            let f: (({x: Int, y: Int}) -> Int) = fun r -> r.x in
            let g: (({x: Int}) -> Int) = fun r -> r.x in
            f == g
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("cannot compare") || err.0.contains("type mismatch"), "unexpected message: {}", err.0);
    }

    #[test]
    fn consistent_treats_two_named_types_with_the_same_id_as_consistent() {
        use types::{consistent, Type};
        let a = Type::Named("List#3".to_string());
        let b = Type::Named("List#3".to_string());
        assert!(consistent(&a, &b));
    }

    #[test]
    fn consistent_rejects_two_named_types_with_different_ids() {
        // Nominal, not structural: two DIFFERENT ids are never
        // consistent, even though this test never gives either one a
        // real registry definition to compare structurally against --
        // that's the whole point, comparison never looks at what a
        // Named type unfolds to.
        use types::{consistent, Type};
        let a = Type::Named("List#3".to_string());
        let b = Type::Named("Tree#7".to_string());
        assert!(!consistent(&a, &b));
    }

    #[test]
    fn consistent_rejects_a_named_type_against_a_structurally_identical_concrete_type() {
        // Pure nominal comparison: a Named type is never consistent
        // with anything other than another Named type sharing its
        // exact id, even a hand-built value that happens to look
        // exactly like what it (hypothetically) unfolds to.
        use types::{consistent, EffectRow, Type};
        let named = Type::Named("List#3".to_string());
        let concrete = Type::Union(std::rc::Rc::new(vec![
            Type::Tuple(std::rc::Rc::new(vec![Type::Int, Type::Dyn])),
            Type::Int,
        ]));
        let _ = EffectRow::Dyn; // silence unused-import if EffectRow isn't otherwise needed here
        assert!(!consistent(&named, &concrete));
    }

    #[test]
    fn named_type_displays_as_its_own_clean_surface_name() {
        use types::Type;
        let ty = Type::Named("List#3".to_string());
        assert_eq!(format!("{ty}"), "List");
    }

    #[test]
    fn a_self_referential_type_alias_parses_successfully() {
        // This used to be a parse error ("unknown type: List") --
        // List wasn't in scope yet while its own RHS was being parsed.
        // (Bool stands in for a Unit-style base case here -- this
        // grammar has no builtin Unit type; any non-recursive
        // alternative demonstrates the same self-reference detection.)
        let src = r#"
            type List = (Int, List) | Bool in
            5
        "#;
        assert!(parser::parse(src).is_ok());
    }

    #[test]
    fn a_non_recursive_alias_is_completely_unaffected() {
        // Same alias mechanism, but this one never references itself
        // -- must behave EXACTLY as before this whole feature: fully
        // expanded, no Type::Named anywhere, no registry entry.
        let src = r#"
            type Pair = (Int, Int) in
            let p: Pair = (1, 2) in
            p
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        match result {
            Value::List(items) => assert_eq!(items[0].as_int(), 1),
            other => panic!("expected a tuple, got {other}"),
        }
    }

    #[test]
    fn parse_with_named_types_registers_a_self_referential_alias() {
        use types::Type;
        let src = r#"
            type List = (Int, List) | Bool in
            5
        "#;
        let (_, _, _, named_types) = parser::parse_with_named_types(src).unwrap();
        assert_eq!(named_types.len(), 1);
        let def = named_types.values().next().unwrap();
        // The registered definition is List's own one-level structure
        // -- a Union of a Tuple (whose second element is the SAME
        // Named id, pointing back at this very entry) and a plain
        // Bool base case. Confirm it's a Union containing a Tuple
        // whose second element is a Type::Named.
        match def {
            Type::Union(alts) => {
                let has_self_ref_tuple = alts.iter().any(|alt| {
                    matches!(alt, Type::Tuple(items) if items.len() == 2 && matches!(items[1], Type::Named(_)))
                });
                assert!(has_self_ref_tuple, "expected a Tuple alternative whose 2nd element is Type::Named, got {def:?}");
            }
            other => panic!("expected a Union, got {other:?}"),
        }
    }

    #[test]
    fn two_recursive_aliases_sharing_a_surface_name_in_different_scopes_do_not_collide() {
        use types::Type;
        // Parenthesized grouping + sequential `let _ = ... in ...`,
        // not a tuple literal -- both are definitely-valid constructs
        // in this grammar, avoiding any risk that a tuple element
        // doesn't accept a full `type ... in ...` prefix chain (this
        // plan never verified that either way, and there's no reason
        // to depend on it here).
        let src = r#"
            let _ = (type List = (Int, List) | Bool in 1) in
            let _ = (type List = (Str, List) | Bool in 2) in
            3
        "#;
        let (_, _, _, named_types) = parser::parse_with_named_types(src).unwrap();
        // Two SEPARATE registrations, each with its own gensym'd id --
        // not one clobbering the other.
        assert_eq!(named_types.len(), 2);
        let mut sees_int_tuple = false;
        let mut sees_str_tuple = false;
        for def in named_types.values() {
            if let Type::Union(alts) = def {
                for alt in alts.iter() {
                    if let Type::Tuple(items) = alt {
                        if items[0] == Type::Int {
                            sees_int_tuple = true;
                        }
                        if items[0] == Type::Str {
                            sees_str_tuple = true;
                        }
                    }
                }
            }
        }
        assert!(sees_int_tuple && sees_str_tuple, "expected both scopes' own definitions preserved independently");
    }

    #[test]
    fn coerce_accepts_a_tuple_literal_against_a_named_type_via_a_one_level_unfold() {
        // xs: List is a CONCRETE annotation, not Dyn -- the tuple
        // literal's own inferred type is a plain Type::Tuple, never
        // itself Type::Named, so this exercises coerce()'s own new
        // unfold-and-retry rescue (Step 3), not a runtime boundary
        // check at all. Confirmed statically accepted with zero
        // wrapping: if this compiles and runs to completion, the
        // rescue worked; a wrong implementation would surface as a
        // static "type mismatch" error from check_with_named_types
        // instead.
        let src = r#"
            type List = (Int, List) | Bool in
            let f = fun xs: List -> xs in
            f((1, (2, true)))
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        // Value has no as_tuple() method -- tuples are Value::List at
        // runtime (see value.rs's own doc comment: "a fixed-arity
        // List"). Match on Value::List directly instead.
        match &result {
            Value::List(items) => assert_eq!(items[0].as_int(), 1),
            other => panic!("expected a tuple (Value::List), got {other}"),
        }
    }

    #[test]
    fn a_genuinely_dyn_sourced_value_is_checked_at_runtime_against_a_named_type_one_level_deep() {
        // g's own parameter is explicitly Dyn -- it stays Dyn all the
        // way to `f(v)` inside g's body, so THAT call genuinely
        // crosses a real Dyn boundary at runtime (build_boundary_check
        // via coerce's own Dyn/Var early-return-turned-real-check
        // path), not coerce's static rescue.
        let src = r#"
            type List = (Int, List) | Bool in
            let f = fun xs: List -> xs in
            let g = fun v: Dyn -> f(v) in
            g((1, (2, true)))
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        // Value has no as_tuple() method -- tuples are Value::List at
        // runtime (see value.rs's own doc comment: "a fixed-arity
        // List"). Match on Value::List directly instead.
        match &result {
            Value::List(items) => assert_eq!(items[0].as_int(), 1),
            other => panic!("expected a tuple (Value::List), got {other}"),
        }
    }

    #[test]
    fn a_genuinely_dyn_sourced_value_with_the_wrong_shape_is_rejected_at_runtime() {
        let src = r#"
            type List = (Int, List) | Bool in
            let f = fun xs: List -> xs in
            let g = fun v: Dyn -> f(v) in
            g("not a list at all")
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine::run(&arena, elaborated, Env::prelude(), &spans)
        }));
        assert!(outcome.is_err(), "expected the runtime boundary check to reject a Str where List is required");
    }

    #[test]
    fn a_bare_union_self_reference_does_not_infinite_loop_at_a_dyn_boundary() {
        // A's own recursive occurrence is a BARE Union alternative -- not
        // wrapped in a Tuple/Record the way List = (Int, List) | Bool is
        // above, whose Tuple shape check never inspects element types and
        // so never revisits the Named leaf. Unfolding Type::Named(id) here
        // reproduces the exact same Union([Int, Named(id)]) again, which
        // used to send build_union_check/build_shape_predicate/
        // build_boundary_check into genuine infinite recursion in THIS
        // PROCESS's own call stack during elaboration (a real stack
        // overflow, not a graceful type error). g's own parameter is
        // explicitly Dyn, so it stays Dyn all the way to f(v) -- forcing a
        // real runtime boundary check against A, not coerce's own static
        // rescue (mirrors a_genuinely_dyn_sourced_value_is_checked_at_runtime_against_a_named_type_one_level_deep
        // above). With the ancestor-tracking fix, `A`'s own self-reference
        // contributes nothing new to the check -- `type A = Int | A`
        // behaves exactly like plain `Int`, the only alternative that can
        // ever actually match -- so this is expected to type-check and RUN
        // TO COMPLETION (not hang or crash) for a genuine Int value. The
        // fact that `cargo test` returns at all with this test included is
        // itself the primary evidence a stack-overflow bug can't be caught
        // by a plain pass/fail assertion.
        let src = r#"
            type A = Int | A in
            let f = fun xs: A -> xs in
            let g = fun v: Dyn -> f(v) in
            g(1)
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), 1);
    }

    #[test]
    fn matching_on_a_named_typed_value_is_not_incorrectly_flagged_as_impossible() {
        // Without this task's own fix, pattern_could_match would reject
        // BOTH arms below as statically impossible (consistent()'s new
        // nominal-only Type::Named arm rejects it against any
        // structural pattern type), making this a static error instead
        // of running to completion.
        let src = r#"
            type List = (Int, List) | Bool in
            let f = fun xs: List ->
                match xs
                | (h, _) -> h
                | _ -> 0
            in
            f((7, opaque))
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), 7);
    }

    #[test]
    fn pattern_could_match_does_not_infinite_loop_on_a_bare_union_self_reference() {
        // A's own recursive occurrence is a BARE Union alternative (not
        // wrapped in a Tuple the way List = (Int, List) | Bool is
        // above), same shape as build_shape_predicate's own regression
        // at a_bare_union_self_reference_does_not_infinite_loop_at_a_dyn_boundary.
        // A List pattern against a Named(A) scrutinee unfolds to
        // Union([Int, Named(A)]), and Pattern::List's own Union arm fans
        // out into EVERY alternative -- including Named(A) again -- which
        // without pattern_could_match's own ancestor-tracking cycle
        // break would send it into genuine infinite recursion (a real
        // stack overflow) in THIS PROCESS's own call stack during
        // elaboration, rather than a graceful type error. With the fix,
        // `A`'s own self-reference contributes nothing new (revisiting
        // it answers `false` immediately), so the List pattern is
        // correctly and STATICALLY rejected as impossible against an
        // Int-or-A scrutinee -- the fact that this test returns an
        // error at all (rather than crashing the test process) is the
        // primary evidence a stack-overflow bug can't be caught by a
        // plain pass/fail assertion.
        let src = r#"
            type A = Int | A in
            let f = fun xs: A ->
                match xs
                | (h, t) -> h
                | _ -> 0
            in
            f(1)
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let result = typecheck::check_with_named_types(&mut arena, root, &spans, named_types);
        assert!(result.is_err(), "a List pattern can never match an Int-or-A scrutinee");
    }
}
