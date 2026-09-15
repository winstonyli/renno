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
    let (mut arena, spans, root) = parser::parse(src)?;
    let elaborated = typecheck::check(&mut arena, root, &spans).map_err(|e| e.1.format_error(src, &e.0))?;
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
            Expr::Int(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Token(_) | Expr::Var(_) => false,
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
                contains_check(arena, *scrutinee) || arms.iter().any(|(_, body)| contains_check(arena, *body))
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
        assert!(err.0.contains("expected Int, found Bool"), "unexpected message: {}", err.0);
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
        assert!(err.0.contains("expected Int, found Bool"), "unexpected message: {}", err.0);
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
        assert!(parser::parse("f match x with | y -> y").is_err());
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
                     match xs with | [] -> [] | h :: t -> f(h) :: my_map(f)(t) \
                   in my_map(fun x -> x * 2)([1, 2, 3])";
        assert_eq!(run_untyped(src).to_string(), "[2, 4, 6]");
    }

    #[test]
    fn cons_onto_non_list_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("1 :: 2").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected a list, found Int"), "unexpected message: {}", err.0);
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

    // --- gradual verification (`where` refinements) ---

    #[test]
    fn refinement_proven_at_parse_time_has_no_runtime_check() {
        // If desugar_refinement's proof succeeds, `body` is used
        // COMPLETELY UNCHANGED -- confirmed here by matching the bound
        // value with a pattern that would fail if any wrapping `if`/`fail`
        // node were still present around it.
        let src = r#"let n: Int where 0 < n = 5 in match n with | 5 -> "unwrapped" | _ -> "bug""#;
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
             let rec sum = fun l -> match l with | (\"Nil\",) -> 0 | (\"Cons\", h, t) -> h + count(t)
             and count = fun l -> match l with | (\"Nil\",) -> 0 | (\"Cons\", h, t) -> 1 + sum(t)
             in sum(Cons(1)(Cons(2)(Cons(3)(Nil))))"
        );
        assert_eq!(run_untyped(&src).as_int(), 5);
    }

    // --- pattern matching ---

    #[test]
    fn match_literal_picks_matching_arm() {
        let src = r#"match 2 with | 1 -> "one" | 2 -> "two" | _ -> "many""#;
        assert_eq!(run_untyped(src).as_str(), "two");
    }

    #[test]
    fn match_wildcard_arm_is_fallback() {
        let src = r#"match 99 with | 1 -> "one" | _ -> "many""#;
        assert_eq!(run_untyped(src).as_str(), "many");
    }

    #[test]
    fn match_nil_and_cons_recurses_over_a_list() {
        let src = "let rec sum = fun xs -> match xs with | [] -> 0 | h :: t -> h + sum(t) in sum([1, 2, 3, 4])";
        assert_eq!(run_untyped(src).as_int(), 10);
    }

    #[test]
    fn match_fixed_length_list_pattern_binds_each_element() {
        assert_eq!(run_untyped("match [1, 2] with | [a, b] -> a + b | _ -> 0").as_int(), 3);
    }

    #[test]
    fn match_fixed_length_list_pattern_requires_exact_length() {
        // [a, b] must NOT match a 3-element list -- falls through to the
        // wildcard arm instead of binding a/b partially.
        assert_eq!(run_untyped(r#"match [1, 2, 3] with | [a, b] -> "two" | _ -> "other""#).as_str(), "other");
    }

    #[test]
    #[should_panic(expected = "match failed: no pattern matched the value")]
    fn match_with_no_matching_arm_panics() {
        run_untyped(r#"match 5 with | 1 -> "x""#);
    }

    #[test]
    fn match_result_type_check() {
        // Every arm's body is Int -- confirms the elaborated Match's result
        // type is Int (not Dyn), same widen-only-on-disagreement rule as If.
        let (mut arena, spans, root) = parser::parse("let f = fun x: Int -> x + 1 in f(match 1 with | 1 -> 10 | _ -> 20)").unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(!contains_check(&arena, elaborated), "Int arms should need no runtime Check at the Int-annotated call");
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 11);
    }

    #[test]
    fn match_rejects_impossible_pattern_statically() {
        let (mut arena, spans, root) = parser::parse("match 5 with | true -> 1 | _ -> 2").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("can never match"), "unexpected message: {}", err.0);
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
            match Some(5) with | ("None",) -> 0 | ("Some", x) -> x
        "#;
        assert_eq!(run_untyped(src).as_int(), 5);
    }

    #[test]
    fn hand_rolled_nullary_tag_matches_its_own_arm() {
        let src = r#"
            let None = ("None",) in
            let Some = fun x -> ("Some", x) in
            match None with | ("None",) -> 0 | ("Some", x) -> x
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
             let rec sum = fun l -> match l with | (\"Nil\",) -> 0 | (\"Cons\", h, t) -> h + sum(t) in
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
        let src = "match (1, \"a\", true) with | (a, b, c) -> a";
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
        let src = "match ((1, 2), 3) with | ((p, q), x) -> p + q + x";
        assert_eq!(run_source(src).unwrap().as_int(), 6);
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
            match Meters(5) with
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
            match a with
            | (an, at) -> match b with
              | (bn, bt) -> at == bt
        "#;
        let outcome = run_source(src).unwrap();
        assert!(outcome.as_bool());
    }

    #[test]
    fn two_opaque_tuple_values_from_different_functions_carry_different_tokens() {
        let src = r#"
            let Meters = fun n -> (n, opaque) in
            let Seconds = fun n -> (n, opaque) in
            match Meters(5) with
            | (an, at) -> match Seconds(5) with
              | (bn, bt) -> at == bt
        "#;
        let outcome = run_source(src).unwrap();
        assert!(!outcome.as_bool());
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
        let src = "match (5,) with | (n,) -> n";
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
              match p with
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
              match p with
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
        let src = "match {x: 1, y: 2} with | {x: a, y: b} -> a + b";
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
        let src = "match {y: 2, x: 1} with | {x: a, y: b} -> a - b";
        assert_eq!(run_source(src).unwrap().as_int(), -1);
    }

    #[test]
    fn record_field_punning_works_in_construction_and_pattern() {
        // `{x, y}` means `{x: x, y: y}` on both sides.
        let src = "let x = 3 in let y = 4 in match {x, y} with | {x, y} -> x * x + y * y";
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
            match f({x: 1, y: 2}) with | {x} -> x
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
            let f = fun p: R -> match p with | {x: a} -> a
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
              match p with
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
            let area = fun p: Shape -> match p with | {radius: r} -> r * r * 3
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
        let src = r#"let f = fun p: Str -> match p with | {x: a} -> a | s -> s in f("hi")"#;
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
            let f = fun p: R -> match p with | {x: a} -> a | {x: a, y: b} -> a + b
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
        let src = "match 1 < 2 with | true -> 1 | false -> 0";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    #[test]
    fn non_exhaustive_bool_match_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("match 1 < 2 with | true -> 1").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    #[test]
    fn exhaustive_list_match_typechecks_even_with_dyn_scrutinee() {
        // xs is an unannotated (Dyn) param -- exhaustiveness is judged from
        // the PATTERN shapes present ([] + unconstrained h :: t), not from
        // the scrutinee's own static type, so this still needs no wildcard.
        let src = "let rec f = fun xs -> match xs with | [] -> 0 | h :: t -> h in f([1, 2])";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    #[test]
    fn non_exhaustive_list_match_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("match [1, 2] with | [] -> 0").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    #[test]
    fn list_match_with_restrictive_cons_head_is_not_exhaustive() {
        // `1 :: t` only covers non-empty lists whose head is 1 -- NOT
        // every non-empty list -- so this must still be rejected even
        // though a Cons pattern is present.
        let (mut arena, spans, root) = parser::parse("match [2, 3] with | [] -> 0 | 1 :: t -> 1").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("non-exhaustive"), "unexpected message: {}", err.0);
    }

    #[test]
    fn wildcard_arm_always_makes_a_match_exhaustive() {
        let src = r#"match 5 with | 1 -> "a" | _ -> "b""#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    // --- match reachability ---

    #[test]
    fn arm_after_a_wildcard_is_unreachable() {
        let (mut arena, spans, root) = parser::parse("match 5 with | _ -> 1 | 1 -> 2").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unreachable match arm"), "unexpected message: {}", err.0);
    }

    #[test]
    fn duplicate_literal_arm_is_unreachable() {
        let (mut arena, spans, root) = parser::parse(r#"match 5 with | 1 -> "a" | 1 -> "b" | _ -> "c""#).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("unreachable match arm"), "unexpected message: {}", err.0);
    }

    #[test]
    fn wildcard_as_the_last_arm_is_fine() {
        let src = r#"match 5 with | 1 -> "a" | 2 -> "b" | _ -> "c""#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    #[test]
    fn distinct_literals_are_all_reachable() {
        // Regression guard: dominates() must not over-fire -- different
        // Int literals must never be reported as unreachable.
        let src = r#"match 5 with | 1 -> "a" | 2 -> "b" | 3 -> "c" | _ -> "d""#;
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
        let src = "let f = fun n ->\n  match n < 5 with\n  | true -> 1\nin f(3)";
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
}
