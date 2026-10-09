pub mod cont;
pub mod env;
pub mod expr;
pub mod frame;
pub mod index_expr;
pub mod lexer;
pub mod machine;
pub mod parser;
pub mod plist;
pub mod resolve;
pub mod span;
pub mod typecheck;
pub mod types;
pub mod util;
pub mod value;

// SPIKE, not yet a settled decision -- profiling a heavy recursive workload
// (samply, 2026-09-23) found ~63% of self-time inside ntdll's own heap
// manager (RtlAllocateHeap/RtlFreeHeap), not in any renno logic -- every
// Frame::cons/Env frame push/closure Env capture is a small Rc::new, and the
// interpreter is allocation-bound, not logic-bound. mimalloc specializes in
// exactly this workload shape (many small, short-lived allocations).
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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

    // Walks the elaborated tree looking for a runtime boundary check: a
    // native Expr::Check, or a Vec(n) index check (still desugared into
    // `let __check_tmp = ... in if ... then ... else ...`, detected by that
    // Let's fixed binder name). Needed because arena-indexed Expr's derived
    // Debug only prints the immediate node (children are plain ExprRef
    // indices, so Debug does not recurse through them).
    fn contains_check(arena: &expr::Arena, root: expr::ExprRef) -> bool {
        tree_any(arena, root, &|n| match n {
            expr::Expr::Check(..) => true,
            expr::Expr::Let(var, ..) => var.starts_with("__check_tmp#"),
            _ => false,
        })
    }

    // Does any node of the (elaborated) tree under `root` satisfy `pred`?
    // The one tree walker the shape-asserting tests share.
    fn tree_any(arena: &expr::Arena, root: expr::ExprRef, pred: &dyn Fn(&expr::Expr) -> bool) -> bool {
        use expr::Expr;
        let go = |r: &expr::ExprRef| tree_any(arena, *r, pred);
        pred(&arena[root])
            || match &arena[root] {
                Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Token(_) | Expr::Var(_) => false,
                Expr::Check(e, _) | Expr::FieldAccess(e, _) | Expr::Perform(_, e) => go(e),
                Expr::ListLit(items) | Expr::Tuple(items) => items.iter().any(go),
                Expr::Lambda(_, _, body) | Expr::MakeHandler { body, .. } => go(body),
                Expr::App(a, b) | Expr::BinOp(_, a, b) | Expr::Let(_, _, a, b) | Expr::Handle { body: a, handler: b } => go(a) || go(b),
                Expr::LetRec(bindings, body) => bindings.iter().any(|(_, _, v)| go(v)) || go(body),
                Expr::If(c, t, e) => go(c) || go(t) || go(e),
                Expr::Match(s, arms) => go(s) || arms.iter().any(|(_, g, b)| g.as_ref().is_some_and(go) || go(b)),
                Expr::Record(fields) => fields.iter().any(|(_, v)| go(v)),
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

    // Phase 3, Task 2: `elaborate_mode`'s chain-flattening loop now threads
    // a running `cur_mode` through curried Lambda layers, peeling one
    // Fun(param, _, ret) off the annotation per Lambda and Check-ing the
    // next layer against `ret` instead of only Synth-ing it. Both Lambda
    // layers here should adopt their param types from the annotation's own
    // Fun layers, and the innermost body (`x`, a bare Var) gets checked
    // against the innermost Int via check_against's generic fallback --
    // this doesn't observably differ from today's Synth-then-reconcile
    // result for a plain Int, but confirms the loop's own cur_mode
    // threading doesn't panic or misroute across two Lambda layers before
    // Tasks 3/4 add anything Match/If-specific.
    #[test]
    fn check_against_threads_expected_return_type_through_a_curried_lambda() {
        let src = "let f: (Int -> Int -> Int) = fun x -> fun y -> x in f(1)(2)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 1);
    }

    // Regression test for a fix-round-1 finding on this same task: the
    // chain-flattening loop's Lambda arm can downgrade `cur_mode` to
    // `Synth` (when the expected type isn't a `Fun`), which is fine for
    // elaborating the Lambda's own inner tail -- but the RECONSTRUCTED
    // value, after `pending`'s Fun-wrapping unwinds back on, must still be
    // validated against the caller's ORIGINAL expected type. Before the
    // fix, nothing re-checked the reconstructed `Fun` type against a
    // non-Fun (or shape-mismatched) annotation, so both of these silently
    // type-checked instead of erroring.
    #[test]
    fn annotated_let_rejects_a_lambda_whose_shape_does_not_match_the_annotation() {
        // Non-Fun annotation, Fun value: `Int` can never fit `Dyn -> Dyn`.
        let (mut arena, spans, root) = parser::parse("let f: Int = fun x -> x in 1").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int") && err.0.contains("found"), "unexpected message: {}", err.0);

        // Arity mismatch: annotation is a 1-argument Fun, value is a
        // 2-argument curried Fun.
        let (mut arena, spans, root) = parser::parse("let f: (Int -> Int) = fun x -> fun y -> y in 1").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected (Int -> Int)") && err.0.contains("found"), "unexpected message: {}", err.0);
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
        // found and fixed (in its design spec's review) and the concurrent
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
        // The Dyn-to-Record boundary check is name-aware (the native
        // Record test) -- a same-arity value with entirely different
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
        // Regression: a non-Record Dyn value at a Record boundary must
        // produce the ordinary type-error message every other boundary
        // check produces, not an internal field-lookup panic.
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
        // ordinary type-error message every other Dyn-boundary
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
        // `m` is unbound -- resolve.rs resolves it to VarRef::Unbound (not
        // a resolve-time error; gradual typing means an unbound name in
        // source is not itself illegal) and machine::run_loop's
        // Expr::Var arm panics on it lazily, with no Span parameter
        // anywhere near that panic site; the location still comes through
        // via machine::current_span (set centrally in run_loop's Eval
        // step, read back after catch_unwind catches the panic in lib.rs).
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
        use types::{consistent, Type};
        let named = Type::Named("List#3".to_string());
        let concrete = Type::Union(std::rc::Rc::new(vec![
            Type::Tuple(std::rc::Rc::new(vec![Type::Int, Type::Dyn])),
            Type::Int,
        ]));
        assert!(!consistent(&named, &concrete));
    }

    #[test]
    fn named_type_displays_as_its_own_clean_surface_name() {
        use types::Type;
        let ty = Type::Named("List#3".to_string());
        assert_eq!(format!("{ty}"), "List");
    }

    #[test]
    fn indexed_type_displays_as_a_wrapped_type_with_its_index() {
        use crate::index_expr::IndexExpr;
        use std::rc::Rc;
        use types::Type;
        let ty = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Var("n".to_string())));
        assert_eq!(ty.to_string(), "[Dyn](n)");
    }

    #[test]
    fn two_indexed_types_with_equal_wrapped_types_and_equal_indices_are_consistent() {
        use crate::index_expr::IndexExpr;
        use std::rc::Rc;
        use types::Type;
        let a = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Var("n".to_string())));
        let b = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Var("n".to_string())));
        assert!(types::consistent(&a, &b));
    }

    #[test]
    fn two_indexed_types_with_different_indices_are_not_consistent() {
        use crate::index_expr::IndexExpr;
        use std::rc::Rc;
        use types::Type;
        let a = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Lit(3)));
        let b = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Lit(4)));
        assert!(!types::consistent(&a, &b));
    }

    #[test]
    fn indexed_type_is_consistent_with_dyn() {
        use crate::index_expr::IndexExpr;
        use std::rc::Rc;
        use types::Type;
        let a = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Lit(3)));
        assert!(types::consistent(&a, &Type::Dyn));
        assert!(types::consistent(&Type::Dyn, &a));
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
            f((7, true))
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

    #[test]
    fn a_non_recursive_alias_still_produces_no_registry_entries() {
        let src = r#"
            type Pair = (Int, Int) in
            type Triple = (Int, Int, Int) in
            5
        "#;
        let (_, _, _, named_types) = parser::parse_with_named_types(src).unwrap();
        assert!(named_types.is_empty(), "non-recursive aliases must never populate the registry, got {named_types:?}");
    }

    #[test]
    fn two_different_named_types_are_never_interchangeable_even_if_isomorphic() {
        // Two SEPARATE recursive aliases with IDENTICAL structure --
        // nominal comparison means they're still not interchangeable.
        let src = r#"
            type IntList = (Int, IntList) | Bool in
            type OtherIntList = (Int, OtherIntList) | Bool in
            let f = fun xs: IntList -> xs in
            let make_other: (Dyn -> OtherIntList) = fun _ -> (1, (2, opaque)) in
            f(make_other(opaque))
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("type mismatch"), "unexpected message: {err}");
    }

    #[test]
    fn wrap_fun_contract_return_position_self_reference_does_not_always_fail() {
        // F's own return type is F again -- wrap_fun_contract's contract
        // closure, checking its own call's return value against F, hits
        // build_boundary_check's "already visiting F" branch on every
        // single call, since there's no predicate gating it here (unlike
        // build_union_check's own use of the same branch, which is always
        // dead code behind a hard-coded-false predicate). Before the fix,
        // this branch unconditionally failed -- every call through such a
        // contract-wrapped closure panicked, even when the real return
        // value (42 here) never needed checking at all.
        let src = r#"
            type F = (Dyn -> F) in
            let make: (Dyn -> Dyn) = fun v -> (fun y -> 42) in
            let x: F = make(opaque) in
            let y: Dyn = x in
            y(opaque)
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), 42);
    }

    #[test]
    fn plain_parse_and_check_do_not_panic_on_a_self_referential_type_alias() {
        // Regression test for the final fix wave's Defect 1:
        // parser::parse (the ORIGINAL, unchanged signature) builds its
        // own named_types registry internally (Parser's own field) but
        // discards it rather than returning it, so pairing plain parse()
        // with plain typecheck::check() (also unchanged, defaults to an
        // EMPTY registry) used to reach one of Type::Named's several
        // consumers' own `.expect("Type::Named with no registry entry --
        // internal bug")` and PANIC, aborting the whole process --
        // reachable by any caller (in this repo or a library consumer of
        // this crate) who uses the preserved parse()/check() pair
        // instead of the _with_named_types pair on a program using a
        // self-referential type alias, not an internal-bug-only
        // condition as the old comments claimed. Every such call site
        // now degrades gracefully instead of panicking (matching each
        // function's own existing Dyn/Var-permissive fallback), so
        // check() below is expected to return a plain Result either way
        // -- proven here via catch_unwind, mirroring this file's own
        // existing "prove no panic occurs" style (see
        // dyn_to_fun_boundary_rejects_closure_with_wrong_return_type and
        // dyn_sourced_call_falls_back_permissively above). Exercises
        // pattern_could_match (the `match xs` arms), build_boundary_check/
        // build_shape_predicate (the Dyn-to-List crossing at `f(v)`), and
        // coerce (the literal tuple argument to `f`) all in one program,
        // to cover as many of the (five, not four) `expect` sites as one
        // source string reasonably can.
        let src = r#"
            type List = (Int, List) | Bool in
            let f = fun xs: List ->
                match xs
                | (h, _) -> h
                | _ -> 0
            in
            let g = fun v: Dyn -> f(v) in
            g((1, true))
        "#;
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (mut arena, spans, root) = parser::parse(src).unwrap();
            typecheck::check(&mut arena, root, &spans)
        }));
        assert!(
            outcome.is_ok(),
            "plain parse()+check() must never panic on a self-referential type alias, just possibly error"
        );
    }

    #[test]
    fn coerce_rescue_rejects_a_bare_union_self_reference_given_the_wrong_type() {
        // Regression test for the final fix wave's Defect 2:
        // replace_named_with_dyn's own Union arm used to map a bare
        // Type::Named(id) alternative to Type::Dyn (e.g. Union([Int,
        // Named(id)]) -> Union([Int, Dyn])) instead of dropping it --
        // and Dyn is consistent with EVERYTHING, so the whole Union
        // became trivially satisfied by any value at all, making an
        // `A`-typed annotation completely inert. With the fix, the
        // self-referential alternative is dropped instead (Union([Int,
        // Dyn]) -> Union([Int])), agreeing with the runtime path
        // (build_shape_predicate's own ancestor-guard correctly reduces
        // `A` to just `Int` for a re-encountered self-reference). A `Str`
        // argument must be statically rejected -- this is coerce()'s own
        // STATIC rescue specifically (both `xs`/the argument are fully
        // concrete, no Dyn boundary involved at all, unlike Task 3's own
        // a_bare_union_self_reference_does_not_infinite_loop_at_a_dyn_boundary
        // test above, or pattern_could_match's own bare-Union coverage in
        // pattern_could_match_does_not_infinite_loop_on_a_bare_union_self_reference).
        let src = r#"
            type A = Int | A in
            let f = fun xs: A -> xs in
            f("wrong")
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let err = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap_err();
        assert!(err.0.contains("type mismatch"), "unexpected message: {}", err.0);
    }

    #[test]
    fn coerce_rescue_accepts_a_bare_union_self_reference_given_the_reachable_alternative() {
        // The accept-side counterpart to the rejection test just above --
        // an Int literal is exactly the one alternative `type A = Int |
        // A` can ever actually reduce to (its own self-reference
        // contributes nothing new), so coerce()'s static rescue must
        // still accept it after the replace_named_with_dyn fix, the same
        // way it did before (this direction was never broken -- the bug
        // was only ever "accepts too much," never "accepts too little").
        let src = r#"
            type A = Int | A in
            let f = fun xs: A -> xs in
            f(5)
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), 5);
    }

    #[test]
    fn plain_dyn_to_int_program_still_runs_after_indexed_unify_changes() {
        // Renamed from indexed_types_with_a_var_position_unify_the_wrapped_type
        // (final review Finding 4): despite its old name and comment, this
        // program never constructs a Type::Indexed at all -- it's a
        // harness smoke test confirming Task 3's changes didn't break
        // trivial end-to-end compilation, not a test of unify()'s own
        // Indexed arm. See
        // unifying_two_sop_equal_indexed_types_through_if_needs_no_runtime_check
        // below for real coverage of that arm.
        let src = r#"
            let f: (Dyn -> Int) = fun v -> 1 in
            f(opaque)
        "#;
        assert_eq!(run_untyped(src).as_int(), 1);
    }

    #[test]
    fn unifying_two_sop_equal_indexed_types_through_if_needs_no_runtime_check() {
        // Final review Finding 4: unify()'s own Type::Indexed arm had
        // zero real test coverage -- deleting it entirely and re-running
        // the suite still passed, because Expr::App's own unify_fits
        // falls back to consistent() on a unify() failure, and
        // consistent() already has its OWN, independent SOP-equality
        // arm. The one call site where that's NOT true is If/Match's own
        // unify_trial: its failure fallback is a silent widen-to-Dyn,
        // with no consistent()-based rescue at all.
        //
        // `a`/`b` are Vec(3)/Vec(2+1)-typed PARAMETERS (never actually
        // called -- this whole expression is elaborated, not run) --
        // Lit(3) and Add(2,1) are SOP-equal but structurally different
        // IndexExpr trees, forcing the equality check inside unify()'s
        // own arm, not just a trivial identical-tree comparison. With
        // that arm intact, `if true then a else b`'s result type stays
        // PRECISELY Indexed(List(Dyn), 3) (unify succeeds, no widening),
        // so annotating it `: Vec(3)` needs no runtime boundary check at
        // all. If that arm is removed, the if's result degrades to Dyn,
        // and the same annotation then HAS to splice one in -- a
        // difference invisible to a plain pass/fail run (the check would
        // still pass at runtime) but visible via contains_check, which
        // is exactly why this test uses it instead of just running the
        // program. Confirmed to actually fail (contains_check finds a
        // spliced-in check) with unify()'s own Indexed arm commented out,
        // then restored.
        // Deliberately returns `c` directly rather than calling
        // `len(c)`: `len` is an unbound builtin (Type::Dyn in the static
        // ctx), so App's own Type::Dyn arm splices in an unconditional
        // is_fun/wrap_fun_contract check on EVERY call to it, regardless
        // of Vec/Indexed at all -- that would contaminate contains_check
        // with a check unrelated to what this test is trying to isolate.
        let src = r#"
            fun a: Vec(3) ->
            fun b: Vec(2 + 1) ->
                let c: Vec(3) = if true then a else b in
                c
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(
            !contains_check(&arena, elaborated),
            "expected unify()'s own Indexed arm to keep the if's result precisely Indexed(3), needing no runtime check for the following Vec(3) annotation"
        );
    }

    #[test]
    fn fits_requires_both_wrapped_type_and_index_to_match() {
        use crate::index_expr::IndexExpr;
        use std::rc::Rc;
        use types::Type;
        let a = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Lit(3)));
        let b = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Lit(3)));
        assert!(types::fits(&a, &b));
        let c = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Lit(4)));
        assert!(!types::fits(&a, &c));
    }

    #[test]
    fn fits_lets_an_indexed_value_satisfy_its_own_plain_wrapped_type() {
        // The "forget the index" widening (final review Finding 1+3): an
        // Indexed-typed value is usable anywhere its wrapped type is
        // required -- a real Vec(3) satisfies a required plain [Int].
        use crate::index_expr::IndexExpr;
        use std::rc::Rc;
        use types::Type;
        let vec3 = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Int))), Rc::new(IndexExpr::Lit(3)));
        assert!(types::fits(&Type::List(Rc::new(Type::Int)), &vec3));
        // The REVERSE must NOT hold -- a plain [Int] does not carry the
        // length guarantee a required Vec(3) position promises, and
        // silently letting it through would defeat the entire point of
        // tracking the index in the first place.
        assert!(!types::fits(&vec3, &Type::List(Rc::new(Type::Int))));
    }

    #[test]
    fn a_vec_typed_value_can_be_pattern_matched_with_nil_and_cons() {
        // Final review Finding 1+3: before the fix, pattern_could_match's
        // Type::Indexed gap AND bind_pattern_vars's Pattern::Cons hard
        // unify() call both rejected this -- []/:: matching against a
        // Vec(n)-typed scrutinee is supposed to be exactly as possible as
        // against the wrapped [T] itself (spec's own §5: exhaustiveness
        // is unaffected by Vec). `v` is a genuinely Vec(3)-typed value
        // (crossed a real Dyn boundary via `x`, not just a raw literal
        // annotation), matching the same discipline every other Indexed
        // Dyn-boundary test in this file already uses.
        let src = r#"
            let x: Dyn = [1, 2, 3] in
            let v: Vec(3) = x in
            match v
            | [] -> 0
            | h :: t -> h
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 1);
    }

    #[test]
    fn a_vec_typed_value_can_be_passed_where_a_plain_dyn_list_is_expected() {
        // Final review Finding 1+3: a Vec(3)-typed value must be usable
        // anywhere its wrapped type ([Dyn]) is expected -- the fits()
        // widening fix, exercised through a real function call rather
        // than a direct fits() unit test.
        let src = r#"
            let x: Dyn = [1, 2, 3] in
            let v: Vec(3) = x in
            let f = fun ys: [Dyn] -> len(ys) in
            f(v)
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    #[test]
    fn a_plain_list_typed_value_does_not_satisfy_a_required_vec_position() {
        // The reverse of the test above, which must still correctly
        // fail: a plain [Int]-typed value carries no length guarantee,
        // so it must NOT statically satisfy a required Vec(3) parameter
        // -- silently accepting it would defeat the entire point of
        // tracking the index. This is a purely STATIC rejection (both
        // sides fully concrete, no Dyn boundary involved at all).
        let src = r#"
            let x: Dyn = [1, 2, 3] in
            let v: [Int] = x in
            let f = fun ys: Vec(3) -> len(ys) in
            f(v)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("type mismatch"), "unexpected message: {err}");
    }

    #[test]
    fn a_union_containing_a_vec_alternative_falls_through_to_the_other_alternative() {
        // Final review Finding 3: before the fix, build_shape_predicate's
        // Type::Indexed arm checked shape only (is_list), not length --
        // so a length-5 list's shape predicate for the Vec(3) alternative
        // wrongly reported "yes, matches," routing it into Vec(3)'s own
        // full check (which then correctly rejects the length) instead
        // of falling through to try the [Int] alternative next. With the
        // fix, the Vec(3) alternative's own shape predicate correctly
        // reports "no" for a length-5 value, and build_union_check falls
        // through to [Int], which accepts it.
        let src = r#"
            type T = Vec(3) | [Int] in
            let x: Dyn = [1, 2, 3, 4, 5] in
            let y: T = x in
            len(y)
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 5);
    }

    #[test]
    fn vec_with_a_literal_index_parses_and_displays() {
        let ty = parser::parse_type_string("Vec(3)").unwrap();
        assert_eq!(ty.to_string(), "[Dyn](3)");
    }

    #[test]
    fn vec_with_a_variable_index_parses_and_displays() {
        let ty = parser::parse_type_string("Vec(n)").unwrap();
        assert_eq!(ty.to_string(), "[Dyn](n)");
    }

    #[test]
    fn vec_with_an_arithmetic_index_parses() {
        let ty = parser::parse_type_string("Vec(n + m)").unwrap();
        assert_eq!(ty.to_string(), "[Dyn](n + m)");
    }

    #[test]
    fn vec_index_supports_multiplication_and_subtraction() {
        let ty = parser::parse_type_string("Vec(n * 2 - 1)").unwrap();
        assert_eq!(ty.to_string(), "[Dyn](n * 2 - 1)");
    }

    #[test]
    fn vec_without_an_index_argument_is_a_parse_error() {
        assert!(parser::parse_type_string("Vec").is_err());
        assert!(parser::parse_type_string("Vec()").is_err());
    }

    #[test]
    fn type_alias_cannot_redefine_vec() {
        // Same guard as `type Int = ... in ...` etc: parse_type's `Vec(`
        // recognition (see the Vec tests above) runs before the
        // type_aliases lookup, so a `type Vec = ... in ...` alias would
        // otherwise parse fine but be silently discarded the moment it's
        // used as `Vec(...)` -- reject it at the binder instead.
        let err = parser::parse("type Vec = Int in 5").unwrap_err();
        assert!(
            err.contains("cannot redefine builtin type Vec"),
            "expected a redefine-builtin error mentioning Vec, got: {err}"
        );
    }

    #[test]
    fn indexed_syntax_generalizes_to_a_user_defined_alias() {
        use expr::Expr;
        use types::Type;
        use index_expr::IndexExpr;
        let src = r#"
            type List = (Int, List) | Bool in
            let _x: List(2) = (1, (2, false)) in
            5
        "#;
        let (arena, _spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        // Scoped to the parser change alone: inspect the parsed annotation
        // directly rather than running typecheck (Task 6 -- not yet
        // implemented -- owns whether a value can actually be CONSTRUCTED
        // against this type).
        match &arena[root] {
            Expr::Let(_, Some(Type::Indexed(wrapped, idx)), ..) => {
                assert!(matches!(wrapped.as_ref(), Type::Named(_)), "expected the wrapped type to be Type::Named, got {wrapped:?}");
                assert_eq!(**idx, IndexExpr::Lit(2));
            }
            other => panic!("expected a Let with an Indexed(Named(_), 2) annotation, got {other:?}"),
        }
        assert_eq!(named_types.len(), 1);
    }

    // NOTE: uses run_source, not run_untyped -- run_untyped is plain
    // parser::parse + machine::run with NO typecheck::check pass at all,
    // so a plain `: T` annotation (unlike a `where` refinement, which the
    // PARSER itself desugars into an embedded runtime check regardless of
    // whether typecheck ever runs -- see desugar_refinement) is completely
    // inert under it: machine.rs's own Expr::Let arm ignores its `_ann`
    // field outright. Confirmed empirically before this fix: under
    // run_untyped, the "matching length" case passed VACUOUSLY (no check
    // ever ran, so [1,2,3] just flows through unchecked), the "wrong
    // length" case FAILED outright (no panic at all -- [1,2] flows through
    // unchecked and len(y) just returns 2), and the "non-list" case
    // panicked for the WRONG reason (machine.rs's own bare `len expects a
    // string or list` panic, not a clean type error) -- exactly the
    // failure modes this task's own brief warned to watch for. Only
    // run_source's full parse -> typecheck::check -> run pipeline (the
    // same helper dyn_sourced_value_can_cross_into_an_arithmetic_position
    // above already uses for this exact reason) actually exercises
    // coerce/build_boundary_check.
    #[test]
    fn a_dyn_sourced_list_matching_its_declared_length_is_accepted_at_runtime() {
        let src = r#"
            let x: Dyn = [1, 2, 3] in
            let y: Vec(3) = x in
            len(y)
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    #[test]
    fn a_dyn_sourced_list_not_matching_its_declared_length_fails_at_runtime() {
        let src = r#"
            let x: Dyn = [1, 2] in
            let y: Vec(3) = x in
            len(y)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("type error: expected"), "unexpected message: {err}");
    }

    #[test]
    fn a_dyn_sourced_non_list_crossing_into_vec_fails_at_runtime_not_via_a_bare_len_panic() {
        // Must go through the SAME is_list-then-len composition Tuple's own
        // arity check already uses (typecheck.rs's build_shape_predicate,
        // Type::Tuple arm) -- never a bare len() call that could panic
        // internally on a non-list Value before the type-error path runs.
        let src = r#"
            let x: Dyn = 5 in
            let y: Vec(3) = x in
            len(y)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("type error: expected"), "unexpected message: {err}");
        assert!(!err.contains("len expects"), "leaked a bare len() panic instead of a clean type error: {err}");
    }

    #[test]
    fn resolve_index_follows_a_bound_index_variable() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::InferCtx;
        use std::collections::HashMap;
        let mut infer = InferCtx::new(HashMap::new());
        infer.index_subst.insert("n".to_string(), IndexExpr::Lit(3));
        let resolved = infer.resolve_index(&IndexExpr::Var("n".to_string()));
        assert_eq!(resolved, IndexExpr::Lit(3));
    }

    #[test]
    fn resolve_index_leaves_an_unbound_variable_alone() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::InferCtx;
        use std::collections::HashMap;
        let infer = InferCtx::new(HashMap::new());
        let resolved = infer.resolve_index(&IndexExpr::Var("n".to_string()));
        assert_eq!(resolved, IndexExpr::Var("n".to_string()));
    }

    #[test]
    fn resolve_index_deep_substitutes_inside_a_compound_expression() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::InferCtx;
        use std::collections::HashMap;
        use std::rc::Rc;
        let mut infer = InferCtx::new(HashMap::new());
        infer.index_subst.insert("n".to_string(), IndexExpr::Lit(3));
        let expr = IndexExpr::Add(Rc::new(IndexExpr::Var("n".to_string())), Rc::new(IndexExpr::Lit(1)));
        let resolved = infer.resolve_index_deep(&expr);
        assert_eq!(resolved, IndexExpr::Add(Rc::new(IndexExpr::Lit(3)), Rc::new(IndexExpr::Lit(1))));
    }

    #[test]
    fn resolve_deep_on_a_type_substitutes_inside_an_indexed_wrapper() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::InferCtx;
        use std::collections::HashMap;
        use std::rc::Rc;
        use types::Type;
        let mut infer = InferCtx::new(HashMap::new());
        infer.index_subst.insert("n".to_string(), IndexExpr::Lit(3));
        let ty = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Var("n".to_string())));
        let resolved = infer.resolve_deep(&ty);
        assert_eq!(resolved, Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Lit(3))));
    }

    #[test]
    fn unify_index_expr_binds_a_bare_unbound_variable() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_index_expr, InferCtx};
        use std::collections::HashMap;
        let mut infer = InferCtx::new(HashMap::new());
        let span = Span { start: 0, end: 0 };
        unify_index_expr(&IndexExpr::Var("n".to_string()), &IndexExpr::Lit(3), &mut infer, span).unwrap();
        assert_eq!(infer.resolve_index(&IndexExpr::Var("n".to_string())), IndexExpr::Lit(3));
    }

    #[test]
    fn unify_index_expr_binds_regardless_of_which_side_is_the_variable() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_index_expr, InferCtx};
        use std::collections::HashMap;
        let mut infer = InferCtx::new(HashMap::new());
        let span = Span { start: 0, end: 0 };
        unify_index_expr(&IndexExpr::Lit(3), &IndexExpr::Var("n".to_string()), &mut infer, span).unwrap();
        assert_eq!(infer.resolve_index(&IndexExpr::Var("n".to_string())), IndexExpr::Lit(3));
    }

    #[test]
    fn unify_index_expr_recurses_through_matching_shapes() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_index_expr, InferCtx};
        use std::collections::HashMap;
        use std::rc::Rc;
        let mut infer = InferCtx::new(HashMap::new());
        let span = Span { start: 0, end: 0 };
        // Add(n, 1) unified against Add(3, 1) should bind n=3 by recursing
        // into both operand pairs, not by requiring the whole expressions
        // to already be SOP-equal (they aren't, structurally, until n is
        // bound).
        let lhs = IndexExpr::Add(Rc::new(IndexExpr::Var("n".to_string())), Rc::new(IndexExpr::Lit(1)));
        let rhs = IndexExpr::Add(Rc::new(IndexExpr::Lit(3)), Rc::new(IndexExpr::Lit(1)));
        unify_index_expr(&lhs, &rhs, &mut infer, span).unwrap();
        assert_eq!(infer.resolve_index(&IndexExpr::Var("n".to_string())), IndexExpr::Lit(3));
    }

    #[test]
    fn unify_index_expr_succeeds_when_already_sop_equal_with_no_variables() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_index_expr, InferCtx};
        use std::collections::HashMap;
        use std::rc::Rc;
        let mut infer = InferCtx::new(HashMap::new());
        let span = Span { start: 0, end: 0 };
        // m*n vs n*m -- no variable to bind, but already equal per SOP
        // normalization (Phase 1's own index_exprs_equal).
        let lhs = IndexExpr::Mul(Rc::new(IndexExpr::Var("m".to_string())), Rc::new(IndexExpr::Var("n".to_string())));
        let rhs = IndexExpr::Mul(Rc::new(IndexExpr::Var("n".to_string())), Rc::new(IndexExpr::Var("m".to_string())));
        unify_index_expr(&lhs, &rhs, &mut infer, span).unwrap();
        // The real property this test proves: no spurious binding. If
        // shape-recursion ran before the SOP-equality check, it would
        // recurse positionally into (m, n) and (n, m) and bind m := n --
        // an unintended aliasing of two otherwise-independent variables,
        // even though the whole expressions were already equal and
        // needed no binding at all.
        assert!(infer.index_subst.is_empty());
    }

    #[test]
    fn unify_index_expr_fails_on_a_genuine_mismatch() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_index_expr, InferCtx};
        use std::collections::HashMap;
        let mut infer = InferCtx::new(HashMap::new());
        let span = Span { start: 0, end: 0 };
        assert!(unify_index_expr(&IndexExpr::Lit(3), &IndexExpr::Lit(4), &mut infer, span).is_err());
    }

    #[test]
    fn unify_index_expr_rejects_an_occurs_check_violation() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_index_expr, InferCtx};
        use std::collections::HashMap;
        use std::rc::Rc;
        let mut infer = InferCtx::new(HashMap::new());
        let span = Span { start: 0, end: 0 };
        // n unified against (n + 1) would be an infinite index expression.
        let rhs = IndexExpr::Add(Rc::new(IndexExpr::Var("n".to_string())), Rc::new(IndexExpr::Lit(1)));
        assert!(unify_index_expr(&IndexExpr::Var("n".to_string()), &rhs, &mut infer, span).is_err());
    }

    // Final review Finding 1: unify_trial only snapshotted/restored
    // infer.subst, not infer.index_subst -- so a failed trial that had
    // ALREADY bound an index variable partway through (e.g. position 0
    // of a Tuple, before a later position fails) left that binding
    // permanently in place, even though the trial as a WHOLE failed and
    // unify_trial's own doc comment promises "genuinely leaves no
    // trace." Tuple([Indexed(n), Int]) against Tuple([Indexed(3), Bool])
    // binds n := 3 at position 0 (via unify_index_expr), then fails at
    // position 1 (Int vs Bool) -- confirmed to fail WITHOUT the fix
    // (infer.index_subst still contains n -> 3 after unify_trial
    // returns Err) and pass WITH it, by temporarily reverting
    // unify_trial's own index_subst snapshot/restore and rerunning.
    #[test]
    fn unify_trial_rolls_back_index_subst_on_a_failed_trial() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_trial, InferCtx};
        use std::collections::HashMap;
        use std::rc::Rc;
        use types::Type;
        let mut infer = InferCtx::new(HashMap::new());
        let span = Span { start: 0, end: 0 };
        let t1 = Type::Tuple(Rc::new(vec![
            Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Var("n".to_string()))),
            Type::Int,
        ]));
        let t2 = Type::Tuple(Rc::new(vec![
            Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Lit(3))),
            Type::Bool,
        ]));
        assert!(unify_trial(&t1, &t2, &mut infer, span).is_err());
        assert!(
            infer.index_subst.is_empty(),
            "unify_trial must leave no trace on failure, but index_subst still has: {:?}",
            infer.index_subst
        );
    }

    // Final review Finding 2: unify_index_expr resolved both sides via
    // resolve_index -- a ONE-LEVEL resolution that only unwraps a bare
    // Var, never substituting inside a compound expression. With `n`
    // already bound to 3, unifying Add(n, 1) against Lit(4) incorrectly
    // failed: resolve_index left Add(n, 1) untouched (n is buried inside
    // the Add, not itself a bare Var), so neither the bind-arm nor the
    // SOP-equality arm ever saw the already-true fact that n+1 == 4.
    // Confirmed to fail without the resolve_index_deep fix and pass with
    // it (temporarily reverted and restored).
    #[test]
    fn unify_index_expr_resolves_deeply_so_an_already_bound_variable_inside_a_compound_expression_is_recognized() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_index_expr, InferCtx};
        use std::collections::HashMap;
        use std::rc::Rc;
        let mut infer = InferCtx::new(HashMap::new());
        infer.index_subst.insert("n".to_string(), IndexExpr::Lit(3));
        let span = Span { start: 0, end: 0 };
        let lhs = IndexExpr::Add(Rc::new(IndexExpr::Var("n".to_string())), Rc::new(IndexExpr::Lit(1)));
        unify_index_expr(&lhs, &IndexExpr::Lit(4), &mut infer, span).unwrap();
    }

    // The related Minor issue Finding 2 also predicted: with deep
    // resolution, unifying a bare `n` against `n + 0` now hits the
    // SOP-equality arm (n == n+0 under SOP normalization) BEFORE the
    // occurs-check-guarded bind arm ever runs, so it no longer
    // incorrectly rejects this as an "infinite index expression." No
    // index_subst binding is left behind either, since the pair was
    // already equal and needed none.
    #[test]
    fn unify_index_expr_does_not_false_positive_occurs_check_when_sop_equal() {
        use crate::index_expr::IndexExpr;
        use crate::span::Span;
        use crate::typecheck::{unify_index_expr, InferCtx};
        use std::collections::HashMap;
        use std::rc::Rc;
        let mut infer = InferCtx::new(HashMap::new());
        let span = Span { start: 0, end: 0 };
        let rhs = IndexExpr::Add(Rc::new(IndexExpr::Var("n".to_string())), Rc::new(IndexExpr::Lit(0)));
        unify_index_expr(&IndexExpr::Var("n".to_string()), &rhs, &mut infer, span).unwrap();
        assert!(infer.index_subst.is_empty());
    }

    // The brief's own literal Step-1 test -- `identity_vec(3)(v)`, `v:
    // Vec(3)` -- was written and run against this task's own change
    // (both with and without it) before landing this test. It turns out
    // to be a DEAD END, not just "uninformative": it fails BOTH before
    // and after this task's fix, for a reason unrelated to whether
    // unify()'s own Type::Indexed arm is wired correctly. Expr::App's
    // own elaboration (typecheck.rs, the `Type::Fun(param_ty, ...)` arm)
    // calls `coerce(arena, a2, &a_ty, &param_ty_resolved, ...)` BEFORE
    // it ever calls `unify_fits`/`unify()` -- and `coerce` has no
    // `InferCtx` of its own, so it can only fall back to `consistent()`/
    // `fits()` (types.rs), both of which require exact
    // `index_exprs_equal` on the two index expressions with NO variable-
    // binding capability at all. Unifying a parameter's bare `Vec(n)`
    // against a caller's concrete `Vec(3)` (or `Vec(1 + 2)` -- the
    // brief's own suggested strengthening changes nothing here, since
    // SOP-normalization already treats those as the same shape of
    // mismatch against an unbound `Var`) is REJECTED by `coerce()`
    // itself, well before `unify_fits` would ever reach the
    // `Type::Indexed` arm this task changed. Confirmed empirically:
    // reverting this task's own diff and rerunning the brief's literal
    // test produces the SAME error, at the SAME call site, from
    // `coerce`, not from `unify`'s own catch-all -- so this task's fix
    // cannot be exercised through an ordinary function call at all,
    // only through a path that skips `coerce`.
    //
    // If/Match's own branch-combination is exactly that other path --
    // confirmed by this file's own `unifying_two_sop_equal_indexed_types_
    // through_if_needs_no_runtime_check` test just above, which already
    // demonstrates `if`'s elaboration calls `unify_trial`/`unify()`
    // DIRECTLY, with no `coerce()` pre-check at all (its own fallback on
    // a unify() failure is a silent widen-to-Dyn, not a coerce-style
    // static rejection). So THIS test reuses that same `if`-based shape,
    // with a genuinely UNBOUND index variable on one branch (`a: Vec(n)`,
    // `n` never bound to anything before this point) against a concrete
    // `Vec(3)` on the other (`b`) -- exactly unify_index_expr's "bind a
    // bare variable" case, not its SOP-equality fallback (Var("n") and
    // Lit(3) are not already SOP-equal).
    #[test]
    fn unifying_an_indexed_type_with_a_bare_index_variable_through_if_binds_it_via_unify_index_expr() {
        // Before this task's fix: unify()'s old guard
        // (index_exprs_equal(Var("n"), Lit(3))) is false, so unify()
        // fails, unify_trial rolls back cleanly, and If's own fallback
        // silently widens the combined branch type to Dyn -- the
        // subsequent `: Vec(3)` annotation then needs a REAL runtime
        // check (coerce() sees `from: Dyn`, which always needs one).
        // After this task's fix: unify_index_expr binds n := 3, the if's
        // result type stays PRECISELY Indexed(List(Dyn), 3) (no
        // widening), so the same annotation needs no check at all --
        // confirmed by temporarily reverting this task's own diff:
        // contains_check then finds a check (assertion fails), and
        // restoring the diff removes it again (assertion passes).
        let src = r#"
            fun a: Vec(n) ->
            fun b: Vec(3) ->
                let c: Vec(3) = if true then a else b in
                c
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert!(
            !contains_check(&arena, elaborated),
            "expected unify()'s own Type::Indexed arm to bind n := 3 via unify_index_expr, keeping the if's result precisely Indexed(3) so the Vec(3) annotation needs no runtime check"
        );
    }

    #[test]
    fn needs_down_decides_when_a_source_value_must_be_re_checked_against_a_target() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::needs_down;
        use std::rc::Rc;
        use types::Type;
        let list = |t: Type| Type::List(Rc::new(t));
        let rec = |names: &[&str]| Type::Record(Rc::new(names.iter().map(|n| (n.to_string(), Type::Int)).collect()));
        let vec_of = |i: IndexExpr| Type::Indexed(Rc::new(list(Type::Int)), Rc::new(i));
        assert!(!needs_down(&Type::Dyn, &Type::Int));
        assert!(!needs_down(&Type::Var("a".to_string()), &Type::Int));
        assert!(!needs_down(&Type::Int, &Type::Int));
        assert!(needs_down(&Type::Int, &Type::Dyn));
        assert!(needs_down(&Type::Int, &Type::Union(Rc::new(vec![Type::Int, Type::Bool]))));
        assert!(needs_down(&list(Type::Int), &list(Type::Dyn)));
        assert!(needs_down(&rec(&["x", "y"]), &rec(&["x"])));
        assert!(needs_down(&vec_of(IndexExpr::Lit(3)), &vec_of(IndexExpr::Var("m".to_string()))));
        assert!(!needs_down(&vec_of(IndexExpr::Lit(3)), &vec_of(IndexExpr::Lit(3))));
    }

    #[test]
    fn needs_wrapper_decides_when_a_function_cast_needs_a_wrapper() {
        use crate::typecheck::needs_wrapper;
        use std::rc::Rc;
        use types::{EffectRow, Type};
        let fun = |a: Type, b: Type| Type::Fun(Rc::new(a), EffectRow::Dyn, Rc::new(b));
        assert!(needs_wrapper(&fun(Type::Int, Type::Int), &Type::Dyn));
        assert!(!needs_wrapper(&fun(Type::Dyn, Type::Dyn), &Type::Dyn));
        assert!(needs_wrapper(&fun(Type::Dyn, fun(Type::Int, Type::Int)), &Type::Dyn));
        assert!(!needs_wrapper(&fun(Type::Dyn, Type::Named("T#1".to_string())), &Type::Dyn));
    }

    #[test]
    fn free_index_vars_resolved_finds_a_bare_index_variable() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::{free_index_vars_resolved, InferCtx};
        use std::collections::{BTreeSet, HashMap};
        use std::rc::Rc;
        use types::Type;
        let infer = InferCtx::new(HashMap::new());
        let ty = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Var("n".to_string())));
        let vars = free_index_vars_resolved(&ty, &infer);
        assert_eq!(vars, BTreeSet::from(["n".to_string()]));
    }

    #[test]
    fn free_index_vars_resolved_finds_variables_inside_a_compound_index() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::{free_index_vars_resolved, InferCtx};
        use std::collections::{BTreeSet, HashMap};
        use std::rc::Rc;
        use types::Type;
        let infer = InferCtx::new(HashMap::new());
        let index = IndexExpr::Add(Rc::new(IndexExpr::Var("n".to_string())), Rc::new(IndexExpr::Var("m".to_string())));
        let ty = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(index));
        let vars = free_index_vars_resolved(&ty, &infer);
        assert_eq!(vars, BTreeSet::from(["n".to_string(), "m".to_string()]));
    }

    #[test]
    fn free_index_vars_resolved_finds_nothing_in_an_ordinary_type() {
        use crate::typecheck::{free_index_vars_resolved, InferCtx};
        use std::collections::HashMap;
        use types::Type;
        let infer = InferCtx::new(HashMap::new());
        let vars = free_index_vars_resolved(&Type::Int, &infer);
        assert!(vars.is_empty());
    }

    #[test]
    fn a_let_bound_vec_producing_function_generalizes_over_its_own_index_variable() {
        // identity_vec's own `n` must be generalized so it can be called
        // at two DIFFERENT lengths from the same let-binding, the same way
        // an ordinary polymorphic identity function already generalizes
        // over its own type variable.
        let src = r#"
            let identity_vec = fun n: Int -> fun v: Vec(n) -> v in
            let a: Dyn = [1, 2, 3] in
            let b: Dyn = [1, 2] in
            let va: Vec(3) = a in
            let vb: Vec(2) = b in
            len(identity_vec(3)(va)) + len(identity_vec(2)(vb))
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 5);
    }

    // Direct unit test on extend_generalized/lookup themselves (per the
    // task's own Step 2 fallback), bypassing the Expr::App/coerce gap
    // above entirely: builds identity_vec's OWN type by hand --
    // `Fun(Int, Fun(Vec(n), Vec(n)))` -- generalizes it into a fresh Ctx
    // via extend_generalized, then calls lookup TWICE. Without real
    // generalization (index_vars left empty), both lookups would return
    // the exact same stored Type::Indexed(_, IndexExpr::Var("n")) --
    // literally the same `n`, unable to independently unify against two
    // different lengths later. With generalization working, each lookup
    // mints its OWN fresh index-variable name (fresh_index_name's
    // `n#<counter>` shape), so the two instantiations disagree on that
    // name -- the same signal generalizable_type_vars's own fresh
    // Type::Var per lookup gives for ordinary polymorphism.
    #[test]
    fn extend_generalized_and_lookup_mint_a_fresh_index_variable_per_use() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::{extend_generalized, lookup, Ctx, InferCtx};
        use std::rc::Rc;
        use types::{EffectRow, Type};

        fn vec_of(index: IndexExpr) -> Type {
            Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(index))
        }

        // fun n: Int -> fun v: Vec(n) -> v
        let n = IndexExpr::Var("n".to_string());
        let identity_vec_ty = Type::Fun(
            Rc::new(Type::Int),
            EffectRow::Dyn,
            Rc::new(Type::Fun(Rc::new(vec_of(n.clone())), EffectRow::Dyn, Rc::new(vec_of(n)))),
        );

        let mut infer = InferCtx::new(std::collections::HashMap::new());
        let ctx = extend_generalized(&Ctx::empty(), "identity_vec", identity_vec_ty, &infer);

        // Extracts the index-variable name inside `Fun(Int, Fun(Vec(idx), Vec(idx)))`.
        fn index_var_name(ty: &Type) -> String {
            match ty {
                Type::Fun(_, _, ret) => match ret.as_ref() {
                    Type::Fun(param, _, _) => match param.as_ref() {
                        Type::Indexed(_, index) => match index.as_ref() {
                            IndexExpr::Var(name) => name.clone(),
                            other => panic!("expected a bare index variable, got {other:?}"),
                        },
                        other => panic!("expected Vec(_) as the inner param, got {other:?}"),
                    },
                    other => panic!("expected a nested Fun as identity_vec's return type, got {other:?}"),
                },
                other => panic!("expected identity_vec's own type to be a Fun, got {other:?}"),
            }
        }

        let use_a = lookup(&ctx, "identity_vec", &mut infer);
        let use_b = lookup(&ctx, "identity_vec", &mut infer);
        let name_a = index_var_name(&use_a);
        let name_b = index_var_name(&use_b);

        assert_ne!(name_a, "n", "lookup should instantiate a FRESH name, not return the generalized `n` verbatim");
        assert_ne!(name_b, "n", "lookup should instantiate a FRESH name, not return the generalized `n` verbatim");
        assert_ne!(name_a, name_b, "two separate lookups of a generalized index variable must get INDEPENDENT fresh names");
    }

    // Task 6: closes the Expr::App/coerce gap the test just above
    // documents (and Expr::Let/LetRec's identical gap) -- coerce() alone
    // can't bind an index variable, so consistent()'s Type::Indexed arm
    // is now permissive for a bare index variable on either side (mirrors
    // Type::Var), and Let/LetRec each gain a unify_fits call after their
    // own coerce succeeds, mirroring the pattern Expr::App already had.
    #[test]
    fn calling_a_vec_producing_function_through_an_ordinary_call_infers_its_index() {
        // The spec's own §4 motivating example.
        let src = r#"
            let f = fun v: Vec(n) -> v in
            let x: Dyn = [1, 2, 3] in
            let arg: Vec(3) = x in
            len(f(arg))
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    #[test]
    fn a_let_bound_vec_annotation_with_a_bare_index_variable_actually_binds_it() {
        // Not just "accepted" (the new consistent() permissiveness alone
        // would allow this silently, with n left UNBOUND and no check at
        // all) -- REALLY bound, provably, by using the SAME n again
        // immediately after in a position that can only succeed if n is
        // genuinely 3.
        //
        // Routed through an intermediate concrete `w: Vec(3)` rather than
        // annotating `x` (a genuine Dyn value) directly as `Vec(n)`: a bare
        // index variable crossing an ACTUAL Dyn boundary hits a separate,
        // pre-existing gap this task does not touch -- coerce()'s runtime
        // boundary check (build_boundary_check/index_expr_to_expr) splices
        // the index variable into the check as an ordinary Expr::Var, which
        // assumes (see index_expr_to_expr's own doc comment) it is always a
        // real, in-scope RUNTIME parameter by this phase -- true for a
        // dependently-typed function parameter (`fun n: Int -> fun v:
        // Vec(n) -> v`), but not for a bare index variable minted by a
        // plain `let` annotation, which has no runtime binder at all. That
        // gap is a manifestation of the still-open rigid-vs-flexible index
        // variable distinction this task's own brief explicitly defers to
        // Phase 3/5, not something Task 6 is scoped to fix -- so this test
        // reaches `Vec(n)` only from an already-concrete `Vec(3)` (`w`),
        // never straight from Dyn, keeping it a pure static-inference check
        // of coerce()/unify_fits, the actual subject of this task.
        // (2026-10-07: that gap is closed. The check's index is resolved when
        // typechecking finishes: a later binding gives a real length check, an
        // unconstrained n checks is_list only, anything else fails cleanly; see
        // a_dyn_crossing_into_an_unconstrained_vec_n_only_checks_that_it_is_a_list
        // and dyn_to_vec_n_boundary_with_no_runtime_value_for_n_is_a_clean_error.)
        let src = r#"
            let x: Dyn = [1, 2, 3] in
            let w: Vec(3) = x in
            let y: Vec(n) = w in
            let z: Vec(n) = y in
            len(z)
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    #[test]
    fn a_let_bound_vec_annotation_still_rejects_a_genuinely_mismatched_reuse() {
        // The soundness check for the fix above: if n really got bound to
        // 3 by the first Vec(n) annotation, a SECOND, incompatible use of
        // the SAME n (via a fresh let-bound name reusing n at a different
        // concrete length) must still fail -- proving unify_fits actually
        // ran and bound n, rather than consistent()'s own permissiveness
        // silently accepting both independently. A static rejection here
        // is a plain Err (this codebase's own convention for a coerce/
        // unify failure -- see e.g. union_type_dyn_boundary_rejects_no_
        // matching_alternative_at_runtime above), not a panic, so
        // unwrap_err() is the right check, not catch_unwind.
        //
        // Same reasoning as the test above for routing through concrete
        // `ca: Vec(3)`/`cb: Vec(2)` rather than annotating `a`/`b` (genuine
        // Dyn values) directly as `Vec(n)` -- keeps this a pure static
        // check of unify_fits/unify_index_expr, not a collision with the
        // separate pre-existing runtime-boundary-check gap described there.
        let src = r#"
            let a: Dyn = [1, 2, 3] in
            let b: Dyn = [1, 2] in
            let ca: Vec(3) = a in
            let cb: Vec(2) = b in
            let ya: Vec(n) = ca in
            let yb: Vec(n) = cb in
            1
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("index 3 does not unify with index 2"), "unexpected message: {err}");
    }

    // Task 6 review finding: the fix above only excludes an Indexed-vs-
    // Indexed pair from unify_fits's consistent()/fits() rescue at the TOP
    // level of the match -- unify_fits has explicit recursive arms for
    // Fun/Record (both correctly re-run unify_fits itself on their
    // sub-parts, so the exclusion re-fires one level down), but had NONE
    // for Tuple/List, so a Vec(n) conflict nested inside a Tuple/List
    // annotation fell straight to the generic catch-all seeing only the
    // OUTER Tuple/List shape -- never the nested Indexed pair -- and was
    // silently rescued by consistent()'s own permissiveness. Same bound-
    // index-conflict shape as the test just above (n bound to 3, then
    // reused incompatibly as Vec(2)), just one level deeper inside a
    // Tuple annotation instead of directly.
    //
    // Routed through intermediate concrete `wa: Vec(3)`/`wb: Vec(2)`
    // bindings rather than splicing the raw `Dyn`-typed `ca`/`cb` directly
    // into the tuple literal (the review's own literal example) -- adapted
    // for the SAME reason as this file's own pre-existing
    // `a_let_bound_vec_annotation_still_rejects_a_genuinely_mismatched_
    // reuse` test (see its comment above): confirmed empirically that the
    // literal all-Dyn version never even reaches the Tuple/List-arm bug at
    // all, let alone exercises this fix. `coerce()`'s top-level Dyn-
    // boundary check only fires when the WHOLE annotated type is Dyn (`if
    // !matches!(from, Type::Dyn | ...)`) -- here `from` is
    // `Tuple([Dyn, Int])`, not itself Dyn, so no check is inserted, and
    // unify_fits's own new Tuple arm then recurses into comparing
    // `Indexed(n)` against a genuinely-Dyn element, which trivially
    // succeeds via unify()'s own unconditional Dyn permissiveness (no
    // `Type::Var`/index binding happens at all against Dyn) -- so `n`
    // never gets bound in the first place and there is nothing for a
    // second annotation to conflict with, regardless of this fix. That is
    // the SAME already-known, already-deferred raw-Dyn-into-bare-Vec(n)
    // gap Task 6's own report documents as "Deviation 3" for the
    // non-nested case, not a new one -- the concrete `Vec(k)` intermediate
    // sidesteps it exactly as those existing tests already do, keeping
    // this test a pure exercise of the Tuple/List-arm fix. Confirmed via
    // `git stash` (isolating just the new Tuple/List arms in
    // typecheck.rs, keeping this test as-is): pre-fix this ran to
    // completion returning 1 instead of being statically rejected;
    // post-fix it correctly fails with the same index-conflict message.
    #[test]
    fn a_vec_conflict_nested_inside_a_tuple_annotation_is_still_rejected() {
        let src = r#"
            let ca: Dyn = [1, 2, 3] in
            let cb: Dyn = [1, 2] in
            let wa: Vec(3) = ca in
            let wb: Vec(2) = cb in
            let pa: (Vec(n), Int) = (wa, 1) in
            let pb: (Vec(n), Int) = (wb, 2) in
            1
        "#;
        let err = run_source(src).unwrap_err();
        // The literal is now checked against the already-bound index, so the
        // conflict surfaces as a mismatch between the two Vec lengths.
        assert!(err.contains("expected [Dyn](3), found [Dyn](2)"), "unexpected message: {err}");
    }

    // Same gap, List-nested rather than Tuple-nested -- unify_fits had no
    // Type::List arm either, so `[Vec(n)]` has the identical false-
    // negative failure mode as `(Vec(n), Int)` above. Same concrete-
    // intermediate adaptation and reasoning as the Tuple test just above.
    #[test]
    fn a_vec_conflict_nested_inside_a_list_annotation_is_still_rejected() {
        let src = r#"
            let ca: Dyn = [1, 2, 3] in
            let cb: Dyn = [1, 2] in
            let wa: Vec(3) = ca in
            let wb: Vec(2) = cb in
            let la: [Vec(n)] = [wa] in
            let lb: [Vec(n)] = [wb] in
            1
        "#;
        let err = run_source(src).unwrap_err();
        // The literal is now checked against the already-bound index, so the
        // conflict surfaces as a mismatch between the two Vec lengths.
        assert!(err.contains("expected [Dyn](3), found [Dyn](2)"), "unexpected message: {err}");
    }

    #[test]
    fn check_against_accepts_a_matching_literal() {
        use crate::typecheck::{check_against, Ctx, InferCtx};
        use crate::span::Span;
        use crate::expr::{Arena, Expr, SpanMap};
        use crate::types::{Type, EffectRow};
        use std::collections::HashMap;
        let mut arena = Arena::new();
        let mut spans = SpanMap::new();
        let e = arena.push(Expr::Int(5));
        spans.push(Span { start: 0, end: 0 });
        let mut infer = InferCtx::new(HashMap::new());
        let (row, _) = check_against(&mut arena, e, &Type::Int, &Ctx::empty(), &spans, &mut infer).unwrap();
        assert_eq!(row, EffectRow::pure());
    }

    #[test]
    fn check_against_rejects_a_mismatched_literal() {
        use crate::typecheck::{check_against, Ctx, InferCtx};
        use crate::span::Span;
        use crate::expr::{Arena, Expr, SpanMap};
        use crate::types::Type;
        use std::collections::HashMap;
        let mut arena = Arena::new();
        let mut spans = SpanMap::new();
        let e = arena.push(Expr::Int(5));
        spans.push(Span { start: 0, end: 0 });
        let mut infer = InferCtx::new(HashMap::new());
        assert!(check_against(&mut arena, e, &Type::Bool, &Ctx::empty(), &spans, &mut infer).is_err());
    }

    // Phase 3, Task 3: `Expr::If` gets a real `Mode::Check` arm -- both
    // branches are now checked directly against the caller's `expected`
    // type via `check_against`, instead of only being inferred
    // independently and reconciled via `unify_trial`. `if true then 1
    // else "a"` checked against Int: the `then` branch obviously fits,
    // but the `else` branch ("a": Str) does NOT -- under the OLD (Synth-
    // only) behavior this pair would have silently widened to Dyn and
    // only failed (if at all) at runtime; Check mode now catches it
    // statically, the same real-rejection semantics every other
    // Check-mode site in this file already has (e.g. Expr::App's
    // argument check).
    #[test]
    fn if_check_mode_rejects_a_branch_that_does_not_fit_the_expected_type() {
        use crate::typecheck::{check_against, Ctx, InferCtx};
        use crate::span::Span;
        use crate::expr::{Arena, Expr, SpanMap};
        use crate::types::Type;
        use std::collections::HashMap;
        let mut arena = Arena::new();
        let mut spans = SpanMap::new();
        // Build If(Bool(true), Int(1), Str("a")) by hand -- spans pushed
        // in lockstep with arena, same append-only-PrimaryMap idiom every
        // other hand-built-Arena test in this file already uses.
        let c = arena.push(Expr::Bool(true));
        spans.push(Span { start: 0, end: 0 });
        let t = arena.push(Expr::Int(1));
        spans.push(Span { start: 0, end: 0 });
        let e = arena.push(Expr::Str("a".to_string()));
        spans.push(Span { start: 0, end: 0 });
        let if_expr = arena.push(Expr::If(c, t, e));
        spans.push(Span { start: 0, end: 0 });
        let mut infer = InferCtx::new(HashMap::new());
        let err = check_against(&mut arena, if_expr, &Type::Int, &Ctx::empty(), &spans, &mut infer).unwrap_err();
        assert!(err.0.contains("expected Int, found Str"), "unexpected message: {}", err.0);
    }

    // Same task, the positive case: two branches whose SYNTHESIZED types
    // would never mutually unify under Synth (a Vec(n)-shaped value with
    // an unbound index variable has no unify() arm pairing it against a
    // plain Type::List -- only Indexed-Indexed and List-List are
    // handled), yet each individually satisfies a plain `[Int]` expected
    // type -- `x` via Type::Indexed's own "forget the index" widening
    // (types::fits's Indexed rescue arm), the list literal trivially.
    // Confirms Check mode's own per-branch dispatch makes this succeed
    // with NO runtime boundary check inserted (the reconstructed
    // Expr::If's children are exactly the ORIGINAL ExprRefs pushed below,
    // not wrapped in a Dyn-boundary check scaffold) -- real static
    // precision, not a lucky runtime pass.
    #[test]
    fn if_check_mode_accepts_branches_that_only_agree_via_the_expected_type() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::{check_against, extend_generalized, Ctx, InferCtx};
        use crate::span::Span;
        use crate::expr::{Arena, Expr, SpanMap};
        use crate::types::Type;
        use std::collections::HashMap;
        use std::rc::Rc;

        let mut infer = InferCtx::new(HashMap::new());
        let vec_ty = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Var("n".to_string())));
        let ctx = extend_generalized(&Ctx::empty(), "x", vec_ty, &infer);

        let mut arena = Arena::new();
        let mut spans = SpanMap::new();
        let c = arena.push(Expr::Bool(true));
        spans.push(Span { start: 0, end: 0 });
        let t = arena.push(Expr::Var("x".to_string()));
        spans.push(Span { start: 0, end: 0 });
        let one = arena.push(Expr::Int(1));
        spans.push(Span { start: 0, end: 0 });
        let two = arena.push(Expr::Int(2));
        spans.push(Span { start: 0, end: 0 });
        let e = arena.push(Expr::ListLit(vec![one, two]));
        spans.push(Span { start: 0, end: 0 });
        let if_expr = arena.push(Expr::If(c, t, e));
        spans.push(Span { start: 0, end: 0 });

        let expected = Type::List(Rc::new(Type::Int));
        let (_, result_expr) = check_against(&mut arena, if_expr, &expected, &ctx, &spans, &mut infer)
            .expect("both branches individually fit [Int] via Check-mode's own per-branch dispatch, even though they'd never mutually unify under Synth");

        match &arena[result_expr] {
            Expr::If(c2, t2, e2) => {
                assert_eq!(*c2, c, "condition should be unchanged -- Bool needs no coercion to Bool");
                assert_eq!(*t2, t, "Vec(n)'s own index-forgetting widening is a zero-overhead static fit, no boundary check wrapping expected");
                // ListLit's own elaboration always rebuilds a fresh node
                // (it re-pushes `refs` unconditionally, even when no
                // element needed a coercion -- see its arm in
                // elaborate_node), so e2 is a NEW ExprRef on principle,
                // unrelated to Check mode. What matters here is that it's
                // still a plain, unwrapped ListLit of the same two
                // elements -- not a Dyn-boundary check scaffold.
                match &arena[*e2] {
                    Expr::ListLit(items) => assert_eq!(items, &vec![one, two]),
                    other => panic!("expected an unwrapped ListLit, got {other:?}"),
                }
            }
            other => panic!("expected an unwrapped Expr::If, got {other:?}"),
        }
    }

    // Phase 3, Task 4: `Expr::Match` gets the same Synth/Check split as
    // `Expr::If` just above. Synth-mode behavior is unchanged -- see
    // `mismatched_match_arms_still_widen_to_dyn_not_a_new_rejection` above
    // (still exercising the exact same widen-to-Dyn fallback, now routed
    // through the `mode`-branching arm instead of the old unconditional
    // one). This test is the negative Check-mode case: `match true | true
    // -> 1 | false -> "a"` checked against Int -- the second arm's body
    // ("a": Str) does NOT fit Int, and Check mode must now catch that
    // statically via check_against's own coerce failure, the same real-
    // rejection semantics `if_check_mode_rejects_a_branch_that_does_not_fit_the_expected_type`
    // already established for If.
    #[test]
    fn match_check_mode_rejects_an_arm_that_does_not_fit_the_expected_type() {
        use crate::typecheck::{check_against, Ctx, InferCtx};
        use crate::span::Span;
        use crate::expr::{Arena, Expr, Pattern, SpanMap};
        use crate::types::Type;
        use std::collections::HashMap;
        use std::rc::Rc;

        let mut arena = Arena::new();
        let mut spans = SpanMap::new();
        // Build Match(Bool(true), [(Bool(true), None, Int(1)), (Bool(false), None, Str("a"))])
        // by hand -- spans pushed in lockstep with arena, same idiom every
        // other hand-built-Arena test in this file already uses.
        let scrutinee = arena.push(Expr::Bool(true));
        spans.push(Span { start: 0, end: 0 });
        let arm1_body = arena.push(Expr::Int(1));
        spans.push(Span { start: 0, end: 0 });
        let arm2_body = arena.push(Expr::Str("a".to_string()));
        spans.push(Span { start: 0, end: 0 });
        let arms = Rc::new(vec![
            (Pattern::Bool(true), None, arm1_body),
            (Pattern::Bool(false), None, arm2_body),
        ]);
        let match_expr = arena.push(Expr::Match(scrutinee, arms));
        spans.push(Span { start: 0, end: 0 });
        let mut infer = InferCtx::new(HashMap::new());
        let err = check_against(&mut arena, match_expr, &Type::Int, &Ctx::empty(), &spans, &mut infer).unwrap_err();
        assert!(err.0.contains("expected Int, found Str"), "unexpected message: {}", err.0);
    }

    // Same task, the actual Phase-3 payoff end to end: `make`'s inner `f`
    // has a declared return type (`[Int]`) that must flow down through
    // Let->Lambda->Match (the exact threading built in Tasks 2-4) to
    // check EACH Match arm against `[Int]` directly. `v` is a genuinely
    // Vec(n)-typed value (crossed a real Dyn boundary, same discipline as
    // every other Indexed Dyn-boundary test in this file), so this is the
    // real end-to-end case, not a hand-built one. Confirms it type-checks
    // and runs to completion with the right answer.
    //
    // (contains_check isn't used here to prove Check-mode precision --
    // `let vx: Vec(3) = x` above needs, and gets, its own perfectly
    // legitimate runtime boundary check completely unrelated to the
    // Match; a whole-tree "no check anywhere" assertion would conflate
    // the two. The dedicated structural precision test just below --
    // mirroring If's own `if_check_mode_accepts_branches_that_only_agree_via_the_expected_type`
    // -- is what actually isolates and proves the Match-specific claim.)
    #[test]
    fn vec_returning_function_with_a_tail_match_checks_against_its_declared_type() {
        let src = r#"
            let make = fun n: Int -> fun v: Vec(n) ->
                let f: (Bool -> [Int]) = fun flag -> match flag | true -> v | false -> [1, 2] in
                f(true)
            in
            let x: Dyn = [1, 2, 3] in
            let vx: Vec(3) = x in
            len(make(3)(vx))
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    // The structural payoff isolated, same technique as
    // `if_check_mode_accepts_branches_that_only_agree_via_the_expected_type`:
    // `v` (Vec(n), n an unbound rigid index variable) and `[1, 2]` (plain
    // List(Int)) have no unify() arm pairing them directly (only
    // Indexed-Indexed and List-List are handled) -- under Synth these two
    // arms would only reconcile via a widen-to-Dyn fallback. Under Check
    // mode, each is checked against the SAME already-known `[Int]`
    // directly: `v` fits via Vec(n)'s own index-forgetting widening
    // (types::fits's Indexed rescue arm), `[1, 2]` fits trivially, and
    // the reconstructed Match's own arm bodies are exactly the ORIGINAL
    // ExprRefs pushed below -- no Dyn-boundary check scaffold wrapping
    // either one. This is the genuine, isolated "static precision, not a
    // lucky runtime pass" evidence for Match's own Check-mode arm.
    #[test]
    fn match_check_mode_accepts_arms_that_only_agree_via_the_expected_type() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::{check_against, extend_generalized, Ctx, InferCtx};
        use crate::span::Span;
        use crate::expr::{Arena, Expr, Pattern, SpanMap};
        use crate::types::Type;
        use std::collections::HashMap;
        use std::rc::Rc;

        let mut infer = InferCtx::new(HashMap::new());
        let vec_ty = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Var("n".to_string())));
        let ctx = extend_generalized(&Ctx::empty(), "v", vec_ty, &infer);

        let mut arena = Arena::new();
        let mut spans = SpanMap::new();
        let scrutinee = arena.push(Expr::Bool(true));
        spans.push(Span { start: 0, end: 0 });
        let arm1_body = arena.push(Expr::Var("v".to_string()));
        spans.push(Span { start: 0, end: 0 });
        let one = arena.push(Expr::Int(1));
        spans.push(Span { start: 0, end: 0 });
        let two = arena.push(Expr::Int(2));
        spans.push(Span { start: 0, end: 0 });
        let arm2_body = arena.push(Expr::ListLit(vec![one, two]));
        spans.push(Span { start: 0, end: 0 });
        let arms = Rc::new(vec![
            (Pattern::Bool(true), None, arm1_body),
            (Pattern::Bool(false), None, arm2_body),
        ]);
        let match_expr = arena.push(Expr::Match(scrutinee, arms));
        spans.push(Span { start: 0, end: 0 });

        let expected = Type::List(Rc::new(Type::Int));
        let (_, result_expr) = check_against(&mut arena, match_expr, &expected, &ctx, &spans, &mut infer)
            .expect("both arms individually fit [Int] via Check-mode's own per-arm dispatch, even though they'd never mutually unify under Synth");

        match &arena[result_expr] {
            Expr::Match(_, new_arms) => {
                assert_eq!(new_arms[0].2, arm1_body, "Vec(n)'s own index-forgetting widening is a zero-overhead static fit, no boundary check wrapping expected");
                match &arena[new_arms[1].2] {
                    Expr::ListLit(items) => assert_eq!(items, &vec![one, two]),
                    other => panic!("expected an unwrapped ListLit, got {other:?}"),
                }
            }
            other => panic!("expected an unwrapped Expr::Match, got {other:?}"),
        }
    }

    // Phase 3, Task 5: the one shape not yet covered by Tasks 3/4's own
    // tests -- a CURRIED function (two Lambda layers, exercising Task 2's
    // own cur_mode descent through both, not just one, the way
    // `check_against_threads_expected_return_type_through_a_curried_lambda`
    // does) whose innermost tail is an `If` (not a `Match`) reached
    // directly, with no intervening `let` re-annotating the tail's own
    // expected type. `make`'s declared type `(Int -> Vec(3) -> [Int])`
    // flows down through both Lambda layers -- `n: Int`, then `v: Vec(3)`
    // -- to the tail `if n == 0 then v else [9, 9, 9]`, whose two branches
    // (`v: Vec(3)` and `[9, 9, 9]: [Int]`) have no unify() arm pairing them
    // directly and would only reconcile via Synth's widen-to-Dyn fallback;
    // under Check mode each is checked directly against the same
    // already-known `[Int]`, exactly like
    // `if_check_mode_accepts_branches_that_only_agree_via_the_expected_type`
    // and `match_check_mode_accepts_arms_that_only_agree_via_the_expected_type`
    // above, but reached through two curried layers instead of a hand-built
    // single node.
    #[test]
    fn curried_vec_returning_function_with_a_tail_if_checks_against_its_declared_type() {
        let src = r#"
            let make: (Int -> Vec(3) -> [Int]) = fun n -> fun v -> if n == 0 then v else [9, 9, 9] in
            let x: Dyn = [1, 2, 3] in
            let vx: Vec(3) = x in
            len(make(0)(vx))
        "#;
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    // Final-review fix wave, Minor 6: proves Check mode's real precision
    // through the FULL Let->Lambda->Match threading (not a hand-built
    // `check_against` call bypassing it, like
    // `match_check_mode_accepts_arms_that_only_agree_via_the_expected_type`
    // above does) actually REJECTS a genuine mismatch, not just accepts a
    // legitimate one. `f`'s declared type is `(Bool -> [Int])`; the `true`
    // arm's body is a bare `Str`, which cannot fit `[Int]` under any
    // widening rule -- a real error.
    //
    // Verified empirically (temporarily reverting Finding 1's own
    // tail-level-check fix and rerunning just this test) that this
    // particular program is rejected EITHER WAY, fix or no fix -- it does
    // NOT isolate Finding 1's own gap. That's because Match already got
    // its own real Check-mode arm back in Task 4
    // (`Expr::Match`'s `Mode::Check(expected) =>
    // check_against(arena, *body, expected, ...)` branch): once the Lambda
    // peel puts `cur_mode` at `Check([Int])` for the Match tail,
    // `elaborate_node` dispatches straight into that per-arm
    // `check_against`, independently of the new tail-level check this fix
    // wave added around the dispatch call itself. Synth mode's own
    // widen-to-Dyn fallback (`unify_trial` failure -> `Type::Dyn`, see
    // Expr::Match's Synth arm) is what WOULD have silently reconciled
    // `Str` and `List(Int)` -- but only if Match were ever reached in Synth
    // mode here, which it isn't. Finding 1's actual gap needs a tail shape
    // with no Check-mode arm of its own (a bare literal, Var, or call) --
    // this test is still worth having as Task 6 asks: proof that real,
    // full-chain threading produces a real rejection, not a synthetic
    // single-node `check_against` call.
    #[test]
    fn match_check_mode_through_full_threading_rejects_a_genuine_arm_mismatch() {
        let src = r#"
            let f: (Bool -> [Int]) = fun flag -> match flag | true -> "a" | false -> [1, 2] in
            f(true)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("expected"), "expected a real type mismatch, got: {err}");
    }

    // Final-review fix wave, Finding 1's own isolating regression test.
    // The test just above (mirroring the earlier reviewer's own attempt)
    // does NOT isolate Finding 1's tail-level check: its tail is a
    // `Match`, which has its own independent Check-mode arm
    // (`check_against` per arm, added back in Task 4) that catches the
    // mismatch regardless of whether the NEW tail-level check exists.
    //
    // `f`'s tail here is a bare `Var` (`x`) instead -- `elaborate_node`'s
    // `Expr::Var` arm ignores `mode` entirely (ctx lookup only), so a
    // bare Var has NO Check-mode arm of its own. `x`'s own static type is
    // `Dyn` (crossed via the explicit `: Dyn` annotation on the `let`
    // immediately above it, same discipline every other Indexed
    // Dyn-boundary test in this file uses), so the only way a runtime
    // `Vec(3)` boundary check can ever get spliced in here is the NEW
    // tail-level check -- the one right after the tail-dispatch
    // `elaborate_node` call in `elaborate_mode` (currently
    // src/typecheck.rs:2300-2304), which runs `coerce`/`unify_fits`
    // while `cur_mode` is still `Check(Vec(3))` and the tail's own type
    // is still the BARE `Dyn`.
    //
    // Confirmed empirically (temporarily commenting out just that
    // 2300-2304 block, leaving the pre-existing trailing check at the
    // very end of `elaborate_mode` untouched, then `cargo test
    // lambda_body_bare_dyn_tail`): with the block removed, BOTH
    // assertions below flip -- `contains_check` becomes `false` and
    // `run_source` returns `Ok` instead of `Err`. Restoring the block
    // makes both pass again. Root cause: without the tail-level check,
    // the reconstructed type after the Lambda unwind is `(Int -> Dyn)`
    // (not `(Int -> Vec(3))`) -- the trailing check (using the call's
    // OWN original `mode`, `Check((Int -> Vec(3)))`) then calls
    // `coerce(lambda_expr, (Int -> Dyn), (Int -> Vec(3)), ...)`, but
    // `coerce`'s Dyn-boundary branch only fires when `from` is LITERALLY
    // `Type::Dyn` (its own `matches!(from, Type::Dyn | Type::Var(_))`
    // guard) -- not when `from` is a concrete `Type::Fun` whose return
    // position merely happens to resolve to `Dyn`. `consistent((Int ->
    // Dyn), (Int -> Vec(3)))` is trivially true (Dyn is consistent with
    // anything), so `coerce` returns the Lambda completely UNCHANGED --
    // no wrapping, no runtime check -- and `unify_fits` is equally
    // permissive on an unresolved Dyn return position. So without the
    // tail-level check, the trailing check is a silent no-op here: the
    // function is accepted as `(Int -> Vec(3))` on paper while its real
    // body can return a list of ANY length, completely unchecked.
    #[test]
    fn lambda_body_bare_dyn_tail_gets_the_new_tail_level_check() {
        let src = r#"
            let f: (Int -> Vec(3)) = fun n -> let x: Dyn = [1, 2] in x in
            f(0)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans)
            .expect("Dyn is always consistent with Vec(3) -- this must type-check, the mismatch is only caught at runtime");
        assert!(
            contains_check(&arena, elaborated),
            "the tail-level check should have spliced a real Vec(3) runtime boundary check around x -- \
             without it, f's body is silently accepted as Vec(3)-typed with no check at all"
        );

        let err = run_source(src).unwrap_err();
        // Type::Indexed's own Display renders as "[Dyn](3)", not "Vec(3)"
        // (see types.rs's own Display impl) -- this is the same runtime
        // message format every failing Expr::Check produces.
        assert!(err.contains("expected [Dyn](3)"), "expected a genuine Vec(3) length mismatch at runtime, got: {err}");
    }

    // Phase 4 Task 1 -- Case A pattern-match index refinement (spec
    // section 5). All four tests below were hand-verified empirically
    // (temporarily disabling first the base-case hypothesis, then the
    // whole snapshot/restore block, and confirming each specific test
    // flips from pass to fail) rather than trusted from the plan's own
    // illustrative snippets -- see this task's own report for the two
    // real architectural facts that forced the source strings below to
    // diverge from the plan's:
    //
    //   1. BinOp::Cons and Expr::ListLit are NOT mode-aware -- they
    //      unconditionally produce a plain Type::List, never a
    //      Type::Indexed, regardless of what Check-mode `expected` is in
    //      force (see their own arms in elaborate_node). Neither
    //      consistent() nor fits() (types.rs) has any arm letting a
    //      plain List satisfy a required Indexed type -- only two
    //      Type::Indexed values can ever be compared that way. So a
    //      Match arm whose body FRESHLY CONSTRUCTS a list (`[]`, or
    //      `h :: rest`) can never itself satisfy an Indexed `expected`,
    //      no matter what hypothesis this task injects -- confirmed by
    //      hand-running the plan's own literal example, which fails with
    //      "type mismatch: expected [Dyn](n), found [Dyn]" even for the
    //      textually-correct `[] -> []` arm. The only arm SHAPE that can
    //      satisfy an Indexed `expected` is one whose body is already a
    //      Var/parameter carrying a real Type::Indexed type (e.g. `t`
    //      after this task's own Case A override, or `v` itself) --
    //      consistent()'s existing Indexed-vs-Indexed arm is permissive
    //      whenever EITHER side is a bare index variable (see its own
    //      doc comment), which is what actually admits these tests.
    //   2. `let rec`'s own body is elaborated with the binding's
    //      annotation entered via a plain (monomorphic) `extend`, not
    //      `extend_generalized` -- see Expr::LetRec's own doc comment
    //      ("every binding needs to resolve to its own... type WHILE
    //      elaborating every value in the group"). So a self-recursive
    //      call made from INSIDE that same body sees its own callee type
    //      as the literal, un-instantiated `Vec(n) -> Vec(n)` -- the
    //      SAME "n" as the enclosing scope, not a fresh per-call
    //      instantiation. Recursing on a tail (whose real length is one
    //      LESS than n) then requires unifying that same rigid "n"
    //      against "n - 1", a genuine, correct contradiction this
    //      checker properly rejects ("infinite index expression: m
    //      occurs in m + 1") -- confirmed by hand-running a `let rec
    //      same_length` version of the plan's own test. This is a
    //      pre-existing `let rec` limitation, orthogonal to Case A and
    //      out of this task's scope (real polymorphic recursion over
    //      index variables would need `let rec` to generalize before
    //      elaborating its own body, a materially bigger change) -- so
    //      the "recursive" test below demonstrates the same spec section
    //      5 payoff (a Vec(n)-returning function that only type-checks
    //      because each arm's own index hypothesis is visible while
    //      checking that arm) via a plain, non-self-recursive `let`
    //      instead of `let rec`.
    //
    // Both facts are pre-existing, unrelated to this task's own change,
    // and unaffected by it either way -- Case A only ever adds entries to
    // infer.index_subst, it doesn't touch BinOp::Cons/ListLit or
    // LetRec's own elaboration at all.

    #[test]
    fn recursive_vec_function_typechecks_via_pattern_refinement() {
        // The real spec section 5 payoff: `f`'s step arm returns `v`
        // unchanged (always trivially Vec(n), needing no refinement) --
        // but it ALSO proves, via a nested `Vec(n - 1)` annotation on the
        // tail `t`, that `t`'s real length is exactly one less than `v`'s.
        // Without a correct hypothesis in scope while checking this arm,
        // `t`'s real (Case A-assigned) type `Vec(m)` could never satisfy
        // `Vec(n - 1)` -- SOP-normalized equality (index_expr.rs's own
        // §3) needs `index_subst` to actually resolve "n" to "m + 1"
        // first (`(m + 1) - 1` normalizes to `m`, matching `t`'s type
        // exactly). Confirmed empirically: with the step-arm hypothesis
        // injection disabled, this exact program fails with "type
        // mismatch: expected [Dyn](m + 1), found [Dyn](3)" instead of
        // type-checking.
        let src = r#"
            let f: (Vec(n) -> Vec(n)) = fun v ->
                match v
                | [] -> v
                | h :: t ->
                    let proof: Vec(n - 1) = t in
                    v
            in
            let x: Dyn = [1, 2, 3] in
            let vx: Vec(3) = x in
            len(f(vx))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), 3);
    }

    #[test]
    fn base_case_arm_alone_is_rejected_when_it_does_not_fit_vec_zero() {
        // Confirms the base-case hypothesis is REAL, not a no-op: `v`'s
        // real type is always the literal, bare `Vec(n)` (consistent()'s
        // own bare-index-variable permissiveness accepts comparing it
        // against ANY other Indexed type at the static coerce step,
        // regardless of what n actually resolves to) -- so the only way
        // to observe the base arm's own n=0 hypothesis taking effect is
        // through a REAL, index_subst-consulting comparison
        // (unify_fits/unify_index_expr), not through coerce's own static
        // permissiveness. `let proof: Vec(1) = v in ...`, checked while
        // v's own hypothesized index is 0, does exactly that: it forces
        // unify_index_expr to compare the literal 1 against n's
        // hypothesized value 0, a genuine, provable conflict. Without the
        // n=0 injection (confirmed empirically by disabling it), n is
        // simply unbound here and freely binds to 1 instead -- this
        // exact program type-checks fine, proving the rejection below
        // really does depend on the hypothesis being injected.
        let src = r#"
            let f: (Vec(n) -> Int) = fun v ->
                match v
                | [] -> let proof: Vec(1) = v in 0
                | h :: t -> 0
            in
            let x: Dyn = [] in
            let vx: Vec(0) = x in
            f(vx)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(
            err.contains("does not unify"),
            "expected a real index conflict from the n=0 hypothesis (1 vs 0), got: {err}"
        );
    }

    #[test]
    fn synth_mode_match_on_an_indexed_scrutinee_gets_no_refinement() {
        // NOTE: this scrutinee's own index (`Vec(3)`) is a LITERAL, so
        // `case_a_refinement_target`'s own eligibility check already
        // returns None for it independent of `mode` -- this test would
        // pass identically even if the `Mode::Synth => None` gate were
        // deleted outright. It's kept as basic Synth-mode-Match
        // regression coverage (the match still elaborates fine with no
        // refinement machinery touching it), but the gate itself -- the
        // thing this task most needs protected -- is isolated by
        // `synth_mode_match_on_a_bare_index_var_scrutinee_gets_no_refinement`
        // just below, whose scrutinee IS eligible (a bare index Var) and
        // so actually exercises the `Mode::Synth => None` branch.
        let src = r#"
            let f = fun v: Vec(3) ->
                let r = match v | [] -> [] | h :: t -> t in
                r
            in
            let x: Dyn = [1, 2, 3] in
            let vx: Vec(3) = x in
            f(vx)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans);
        assert!(elaborated.is_ok(), "an unannotated let's own Match must still type-check with no refinement, exactly as before this task");
    }

    #[test]
    fn synth_mode_match_on_a_bare_index_var_scrutinee_gets_no_refinement() {
        // Isolates the `Mode::Synth => None` gate itself. Unlike the test
        // just above (whose `Vec(3)` scrutinee is a literal index that
        // `case_a_refinement_target` already excludes regardless of
        // mode), `v` here is annotated `Vec(n)` -- a BARE index Var --
        // so `case_a_refinement_target`'s OWN eligibility check would
        // return `Some(("n", elem_ty))` for it if `mode` allowed it.
        //
        // `f`'s own definition is the un-annotated `let f = ...`'s value,
        // so per Expr::Lambda's own peel logic in typecheck.rs (the
        // `_ => (ann.unwrap_or_else(...), Mode::Synth)` fallback --  no
        // enclosing Check(Fun(..)) is in force since nothing annotates
        // `f`), the Lambda's body -- this Match -- is elaborated in
        // Mode::Synth, with no expected type flowing in.
        //
        // With the gate intact, `t` in the step arm keeps its plain,
        // unindexed List type from `bind_pattern_vars`'s untouched Cons
        // arm (no hypothesis is injected, no re-`extend` override), so
        // `let proof: Vec(n - 1) = t in 1` requires coercing a plain List
        // into a required Indexed type -- rejected. If the
        // `Mode::Synth => None` line were ever deleted,
        // `case_a_refinement_target` WOULD fire here (its own eligibility
        // check alone can't stop it, unlike the literal-index test
        // above): the step arm would re-bind `t` to `Vec(m)` with
        // hypothesis `n = m + 1`, and `proof`'s binding would type-check
        // instead (`n - 1` SOP-normalizes to exactly `m`) -- proving the
        // gate itself, not just the eligibility filter, is load-bearing.
        let src = r#"
            let f = fun v: Vec(n) ->
                match v
                | [] -> 0
                | h :: t ->
                    let proof: Vec(n - 1) = t in 1
            in f
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans);
        assert!(
            elaborated.is_err(),
            "Synth mode must not refine `t`'s type from the scrutinee's own bare index var, so `t : Vec(n - 1)` must be rejected"
        );
    }

    #[test]
    fn index_subst_hypothesis_does_not_leak_past_the_whole_match() {
        // Self-review's own most important correctness property, made
        // concrete: two INDEPENDENT calls to the SAME Vec(n)->Vec(n)
        // function, with two DIFFERENT concrete lengths, must both
        // succeed. This is only possible if "n" comes out of `f`'s own
        // body elaboration genuinely free (so the enclosing `let`
        // generalizes it, and each call gets its own fresh instantiation)
        // -- which in turn is only true if BOTH arms' own per-arm
        // hypothesis (n=0 for the base arm, n=m+1 for the step arm) were
        // fully undone (via the snapshot/restore) once the match's own
        // arm loop finished, not left bound to whichever arm happened to
        // run last. Confirmed empirically: temporarily skipping the
        // restore (leaving the loop's final -- step-arm -- hypothesis
        // n=m+1 permanently in infer.index_subst) makes the SECOND call
        // below fail with "type mismatch: expected [Dyn](m + 1), found
        // [Dyn](5)", since "n" is then permanently, incorrectly pinned to
        // "one more than SOME fixed, never-reconstrained m" instead of
        // being free to instantiate per call.
        let src = r#"
            let f: (Vec(n) -> Vec(n)) = fun v ->
                match v
                | [] -> v
                | h :: t ->
                    let proof: Vec(n - 1) = t in
                    v
            in
            let x3: Dyn = [1, 2, 3] in
            let v3: Vec(3) = x3 in
            let x5: Dyn = [1, 2, 3, 4, 5] in
            let v5: Vec(5) = x5 in
            len(f(v3)) + len(f(v5))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        let result = machine::run(&arena, elaborated, Env::prelude(), &spans);
        assert_eq!(result.as_int(), 8);
    }

    // --- Task 2: ambient-context and hypothesis-composition tests ---
    //
    // Both tests below deliberately use Case A's BASE-case hypothesis
    // (`[]` -> index := 0, a CONCRETE LITERAL), not the step-case
    // hypothesis (`x :: xs` -> index := m + 1, a FRESH VARIABLE), as
    // their discriminating mechanism. This was NOT an arbitrary choice:
    // hand-tracing unify_index_expr (and confirming empirically, see
    // below) shows the step-case hypothesis alone is NOT safe to build a
    // "must succeed when correct, must fail when broken" test around,
    // because unify_index_expr is a DELIBERATELY NARROW unifier (its own
    // doc comment: "binds a BARE unbound variable to whatever it's
    // compared against... does NOT solve equations") -- so a check of
    // the shape "v2 : Vec(<fresh-var-derived expression>)" will happily
    // rescue itself by just binding whichever fresh var is still free at
    // that point to whatever's needed, REGARDLESS of whether the
    // intended hypothesis ever fired. Confirmed empirically: surgically
    // disabling ONLY `infer.index_subst.insert(n_name.clone(),
    // hypothesis)` in the step-case arm (src/typecheck.rs, leaving the
    // tail's own `arm_ctx = extend(..., Vec(m))` retyping intact) does
    // NOT make `recursive_vec_function_typechecks_via_pattern_refinement`
    // fail -- it keeps passing, because `m` (t's own fresh index var)
    // simply free-binds to whatever `n - 1` resolves to instead of
    // genuinely deriving it from the (missing) hypothesis. The base-case
    // hypothesis (0, a literal) has no such escape hatch: once "n" (or
    // whatever the outer index var is) is bound to a concrete literal,
    // EVERY later comparison touching it is a real, non-rescuable
    // structural fact -- exactly the property these two tests need.

    #[test]
    fn refining_one_vec_via_match_also_refines_an_ambient_vec_sharing_the_same_index_var() {
        // Property (a) from the spec's own design review: "It must apply
        // everywhere the refined variable occurs in the ambient context,
        // not only to the matched scrutinee." `v1` and `v2` are two
        // SEPARATE parameters of the SAME curried function, both
        // annotated `Vec(n)` -- the SAME index variable, exactly the
        // spec's own `zip(v1: Vec(n), v2: Vec(n))` shape. Only `v1` is
        // ever matched; `v2` is never touched by any pattern at all.
        //
        // Matching v1's `[]` arm injects the CONCRETE hypothesis n := 0
        // into infer.index_subst (Case A's base-case rule). `v2`'s own
        // stored type in `arm_ctx` is untouched -- still literally
        // `Vec(n)` -- so the ONLY way `let proof: Vec(1) = v2 in ...`
        // can be judged is by resolving v2's index ("n") through the
        // SAME shared index_subst map v1's own match just wrote into.
        // With the hypothesis correctly applied, n resolves to the
        // concrete literal 0, so comparing it against the required
        // literal 1 is a real, unrescuable mismatch (0 != 1) --
        // correctly REJECTED. If the ambient hypothesis did NOT reach
        // v2 (n stayed a free, unbound variable at this point instead),
        // unify_index_expr's own default-bind rule would happily bind n
        // := 1 right there and ACCEPT the program instead -- the wrong
        // answer. So `is_err()` here is the demonstration: it can only
        // be true because v2, an entirely separate ambient binding, saw
        // the SAME concrete hypothesis v1's own match arm produced.
        let src = r#"
            let f: (Vec(n) -> Vec(n) -> Int) = fun v1 -> fun v2 ->
                match v1
                | [] -> let proof: Vec(1) = v2 in 0
                | h :: t -> 0
            in
            let x1: Dyn = [] in
            let vx1: Vec(0) = x1 in
            let x2: Dyn = [] in
            let vx2: Vec(0) = x2 in
            f(vx1)(vx2)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(
            err.contains("does not unify") || err.contains("mismatch"),
            "v2 should be refined to n=0 alongside v1 inside the [] arm, making Vec(1) a real (rejected) conflict, not silently accepted: got {err}"
        );
    }

    #[test]
    fn nested_match_hypotheses_compose_through_resolve_index_deep() {
        // Property (b) from the spec's own design review: hypotheses
        // compose to a fixed point across nested matches. Outer match on
        // `v` (Vec(n)): the step arm hypothesizes n := m + 1 (m fresh)
        // and retypes `t` as Vec(m). A SECOND, NESTED match on `t`
        // (itself now Vec(m), eligible for its own Case A refinement):
        // its OWN base-case arm hypothesizes m := 0 (concrete). At the
        // point both `proof1`/`proof2` below are checked,
        // infer.index_subst holds BOTH "n" -> m + 1 AND "m" -> 0
        // simultaneously (neither match's own restore has run yet --
        // both are still mid-arm).
        //
        // `proof1: Vec(n - 1) = t` alone is NOT sufficient to prove real
        // two-hop composition, despite looking like it should be: `t`'s
        // own stored type is ALREADY the literal `Vec(m)` (set by the
        // OUTER match's own retyping alone), so `n - 1` only needs to
        // SOP-normalize to the bare symbol `m` -- which it does as soon
        // as the OUTER hop alone (n -> m+1) is resolved, REGARDLESS of
        // whether `m` itself further resolves to anything. Confirmed
        // empirically: with ONLY the inner match's own base-case
        // hypothesis (m := 0) disabled, `proof1` alone still
        // type-checks (`(m+1)-1` SOP-normalizes to the free variable
        // `m`, which then structurally matches t's own still-unresolved
        // `m` via unify_index_expr's occurs-check-then-SOP-equality
        // rescue -- a fact about symbolic cancellation, not about the
        // inner hop's own value at all).
        //
        // `w` closes that gap: a THIRD parameter, also declared
        // `Vec(n)` (the very same outer index variable, ambient --
        // never itself pattern-matched, exactly like property (a)'s
        // `v2`), checked against the CONCRETE literal `Vec(1)` from
        // inside the innermost arm. Resolving `w`'s own index all the
        // way down to a bare literal genuinely requires BOTH hops:
        // n -> m+1 -> (0)+1, which only SOP-normalizes to the literal 1
        // if `m` itself was ALSO correctly resolved to 0 -- a partial,
        // one-hop resolution leaves a residual `m` term that can NEVER
        // SOP-cancel against a pure literal (no free-variable rescue is
        // available once the other side is a concrete Lit, per
        // unify_index_expr's own arms). `proof1` is kept anyway: it
        // isolates a broken OUTER hop specifically (if n never learns
        // m+1 at all, comparing t, whose real value is the concrete 0
        // via the inner hypothesis alone, against the still-fully-free
        // "n - 1" is ALSO a genuine, non-rescuable mismatch).
        //
        // Empirically confirmed (see task-2-report.md for the exact
        // transcripts): disabling EITHER hop's own hypothesis insert
        // alone flips this test from passing to failing; both hops
        // active together makes it pass, matching the SOP-normalized,
        // fully-composed answer (0 + 1 = 1) instead of any partially-
        // resolved intermediate.
        let src = r#"
            let f: (Vec(n) -> Vec(n) -> Int) = fun v -> fun w ->
                match v
                | [] -> 0
                | h :: t ->
                    match t
                    | [] ->
                        let proof1: Vec(n - 1) = t in
                        let proof2: Vec(1) = w in
                        1
                    | h2 :: t2 -> 2
            in
            let x1: Dyn = [9] in
            let vx: Vec(1) = x1 in
            let x2: Dyn = [9] in
            let vw: Vec(1) = x2 in
            f(vx)(vw)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans);
        assert!(
            elaborated.is_ok(),
            "n should SOP-normalize to the concrete literal 1 by composing BOTH n->m+1 and m->0 through resolve_index_deep: {:?}",
            elaborated.err()
        );
        let result = machine::run(&arena, elaborated.unwrap(), Env::prelude(), &spans);
        assert_eq!(result.as_int(), 1);
    }

    // --- Task 3, Fix 1: per-arm index_subst restore ---

    #[test]
    fn stale_hypothesis_does_not_leak_into_a_later_ineligible_sibling_arm() {
        // Task 2's own review found a real soundness gap in Task 1's
        // mechanism: infer.index_subst is ONE shared map, mutated in
        // place by each Case-A-eligible arm's own `insert` call, and
        // used to be restored to its pre-match snapshot only ONCE, after
        // the WHOLE arm loop -- not per arm. An arm that injects no
        // hypothesis of its own (the compound-tail `h :: (h2 :: t2)`
        // arm below -- Case A's own v1 "bare Var tail only" restriction
        // means a nested Cons tail gets no override, no hypothesis) did
        // nothing to reset whatever a PRECEDING sibling arm left behind
        // in that same shared map. `[]` runs first (source order) and
        // injects n := 0; the compound-tail arm runs second and used to
        // silently inherit it, even though reaching a length->=2 pattern
        // implies nothing at all about n's real value.
        //
        // `let proof: Vec(5) = v in 1` inside the compound-tail arm is
        // the discriminator. `v`'s own real type is always the literal,
        // bare `Vec(n)` (consistent()'s own bare-index-variable
        // permissiveness means only a REAL comparison via
        // unify_fits/unify_index_expr -- not coerce's own static
        // permissiveness -- can ever observe what n resolves to, same
        // technique `base_case_arm_alone_is_rejected_when_it_does_not_fit_vec_zero`
        // above already established). With the FIX (no hypothesis active
        // for this arm), n is genuinely free going in, so this simply
        // unifies n := 5 -- accepted. With the BUG (n stuck at the
        // leaked literal 0 from the `[]` arm just before it), the SAME
        // comparison instead finds n already pinned to 0, conflicting
        // with the required 5 -- a real, provable "5 != 0" rejection.
        // So the fix makes this program type-check; the bug makes it
        // wrongly fail.
        //
        // Verified empirically (comment-out-and-rerun): temporarily
        // reverting to the single, once-only restore after the whole
        // loop (removing this fix's own per-arm restore at the top of
        // the loop) flips this test from passing to failing with
        // exactly "type mismatch"/"does not unify" (5 vs the leaked 0),
        // confirming the leak is real and this test genuinely depends
        // on the fix.
        //
        // A trailing `h :: t -> 0` arm is included purely so
        // missing_case's own coverage check accepts this as exhaustive
        // (has_nil && has_full_cons, per its own doc comment) -- it
        // plays no role in the leak scenario itself, which only needs
        // `[]` to run before the compound-tail arm.
        let src = r#"
            let f = fun v: Vec(n) ->
                match v
                | [] -> 0
                | h :: h2 :: t2 ->
                    let proof: Vec(5) = v in 1
                | h :: t -> 0
            in
            let x: Dyn = [1, 2, 3, 4, 5] in
            let vx: Vec(5) = x in
            f(vx)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans);
        assert!(
            elaborated.is_ok(),
            "the compound-tail arm earns no hypothesis of its own and must not inherit the [] arm's leaked n=0 -- \
             n should freely bind to 5 here, got: {:?}",
            elaborated.err()
        );
        let result = machine::run(&arena, elaborated.unwrap(), Env::prelude(), &spans);
        assert_eq!(result.as_int(), 1);
    }

    // --- Final review fix wave: key-scoped index_subst restore
    // (critical finding) ---
    //
    // The whole-branch final review found the snapshot/restore above
    // (Task 3 Fix 1) was scoped too broadly: it clones and restores the
    // ENTIRE index_subst map, not just the one key the match's own
    // hypothesis mechanism is entitled to touch (the scrutinee's own
    // index variable). An arm's own body-check can legitimately prove a
    // genuinely UNRELATED index fact about some other ambient binding
    // (e.g. a second Vec-typed parameter's own real length) -- a whole-
    // map restore silently erases that fact too, the moment the very
    // next restore fires (per-arm, or the final one), even though it has
    // nothing to do with the injected hypothesis. The two tests below
    // prove the fix (key-scoped restore, touching only the hypothesis's
    // own variable) both let a real fact survive the match AND still
    // catches an actual contradiction -- not just that something stopped
    // vanishing.

    #[test]
    fn unrelated_ambient_index_fact_proven_inside_an_arm_survives_the_match_and_a_real_conflict_is_still_caught() {
        // `v: Vec(n)` is the match's own eligible scrutinee; `w: Vec(k)`
        // is a completely separate ambient parameter, sharing NO index
        // variable with `v`'s own hypothesis (`k` != `n`). Inside the
        // `[]` arm, `let p1: Vec(2) = w in 0` proves a real fact --
        // k := 2 -- that has nothing to do with the match on `v` at all.
        // AFTER the match returns (same function body, same elaboration
        // pass, no let-generalization boundary crossed), `let p2: Vec(1)
        // = w in inner` re-checks `w` against a CONTRADICTING length.
        //
        // With the fix (key-scoped restore, only ever touching `n`): `k`
        // is untouched by either the per-arm or the final restore, so it
        // stays resolved to 2 past the match -- comparing it against the
        // literal 1 is a real, provable "2 != 1" conflict, correctly
        // REJECTED.
        //
        // With the bug (whole-map restore): the final restore reassigns
        // the ENTIRE index_subst back to its pre-match snapshot, wiping
        // `k`'s own k := 2 binding right alongside `n`'s. `k` is then
        // genuinely free again when `p2` is checked, so it freely binds
        // to 1 with no conflict at all -- the whole program WRONGLY
        // type-checks, exactly the false-accept the final review's own
        // critical finding describes (confirmed via the reviewer's own
        // repro shape).
        //
        // Verified empirically (comment-out-and-rerun): temporarily
        // reverting both restores to whole-map assignment (`infer.index_subst
        // = snapshot[.clone()]`) flips this test from passing (is_err())
        // to failing (the program wrongly type-checks instead).
        let src = r#"
            let f = fun v: Vec(n) -> fun w: Vec(k) ->
                let inner: Int =
                    match v
                    | [] -> let p1: Vec(2) = w in 0
                    | h :: t -> 0
                in
                let p2: Vec(1) = w in
                inner
            in
            let x1: Dyn = [] in
            let vx: Vec(0) = x1 in
            let x2: Dyn = [9] in
            let vw: Vec(1) = x2 in
            f(vx)(vw)
        "#;
        let err = run_source(src).unwrap_err();
        assert!(
            err.contains("does not unify") || err.contains("mismatch"),
            "w's own k=2 fact, proven inside the [] arm, must survive past the match -- comparing it against \
             the real Vec(1) afterward should be a genuine, rejected conflict, got: {err}"
        );
    }

    #[test]
    fn contradictory_ambient_index_facts_across_sibling_arms_are_caught_not_both_silently_accepted() {
        // Same ambient `w: Vec(k)` shape as the test above, but now BOTH
        // sibling arms of the SAME match each prove their own fact about
        // `w` -- the `[]` arm proves k := 2, the `h :: t` arm proves
        // k := 3. These are mutually contradictory (k can't be both),
        // and the fix's own key-scoped restore deliberately does NOT
        // reset `k` between arms (only `n`, the match's own hypothesis
        // key, gets reset) -- so entering the second arm, `k` is already
        // resolved to 2 from the first, and its own `Vec(3)` proof is a
        // real, provable "2 != 3" conflict. The whole match -- and so
        // the whole program -- must be REJECTED.
        //
        // With the bug (whole-map restore between arms): the per-arm
        // restore at the top of the SECOND arm's own iteration wipes `k`
        // back to fully unbound (the pre-match snapshot), so the second
        // arm's own `Vec(3)` proof freely succeeds with no conflict at
        // all -- BOTH arms silently succeed independently, each having
        // "proven" a different, mutually-incompatible fact about the
        // exact same ambient binding, with nothing ever catching the
        // inconsistency. The call site below (`vw: Vec(2)`) then also
        // succeeds against a freshly-instantiated `k`, so the whole
        // program WRONGLY type-checks under the bug.
        //
        // Verified empirically (comment-out-and-rerun): temporarily
        // reverting both restores to whole-map assignment flips this
        // test from passing (is_err()) to failing (the program wrongly
        // type-checks instead).
        let src = r#"
            let f: (Vec(n) -> Vec(k) -> Int) = fun v -> fun w ->
                match v
                | [] -> let p1: Vec(2) = w in 0
                | h :: t -> let p2: Vec(3) = w in 0
            in
            let x1: Dyn = [] in
            let vx: Vec(0) = x1 in
            let x2: Dyn = [9, 9] in
            let vw: Vec(2) = x2 in
            f(vx)(vw)
        "#;
        let elaborated_err = run_source(src);
        assert!(
            elaborated_err.is_err(),
            "the [] arm's own k=2 and the h::t arm's own k=3 are mutually contradictory facts about the SAME \
             ambient binding w -- this must be caught as a real conflict, not silently accepted from both arms \
             independently, got: {elaborated_err:?}"
        );
    }

    // --- Task 3, Step 1: bind_pattern_vars's own Union-correlation gap ---

    #[test]
    fn bind_pattern_vars_falls_back_to_an_unconstrained_type_for_a_named_union_scrutinee() {
        // Task 3's own real prerequisite investigation (spec section 5,
        // Case B): does bind_pattern_vars correlate a Type::Union-typed
        // scrutinee's own alternative-specific field types to a
        // Pattern::List (tagged-tuple-style) sub-binding, or does it
        // fall back to some unconstrained fresh type with no real
        // precision? Confirmed empirically BEFORE writing any Case B
        // refinement logic, per this task's own brief: with the
        // Pattern::List match's Type::Named/Type::Union arms removed
        // (this fix reverted), this exact test fails -- `t`'s real type
        // falls into the generic catch-all (a fresh, totally
        // unconstrained Type::Var, permissive with anything exactly
        // like Type::Dyn), so `let proof: Bool = t in 0` WRONGLY
        // type-checks even though `t`'s real position in `(Int, List)`
        // is the recursive `List` alternative, not `Bool` at all. With
        // the fix, `t` is correctly `Type::Named(id)` (a `List`), and
        // `Type::Named` is nominal-only (types::consistent's own
        // Type::Named arm: consistent only when both ids match exactly,
        // never structurally with anything else) -- so comparing it
        // against a required `Bool` is a genuine, provable mismatch,
        // correctly rejected.
        let src = r#"
            type List = (Int, List) | Bool in
            let v: List = (1, false) in
            match v
            | (h, t) -> let proof: Bool = t in 0
            | b -> 1
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let result = typecheck::check_with_named_types(&mut arena, root, &spans, named_types);
        assert!(
            result.is_err(),
            "t's real type, if correctly correlated, is Type::Named(id) (a List), which must reject a Bool \
             annotation -- got Ok, meaning bind_pattern_vars fell back to an unconstrained type instead: {result:?}"
        );
    }

    // --- Task 3, Case B: Named+Union-derived pattern-match index refinement ---
    //
    // No surface syntax builds a `Type::Indexed(Type::Named(id), _)`
    // value directly: `Vec(n)`'s own parser sugar (parser.rs's `parse_type`,
    // the `name == "Vec"` arm) is hard-coded to wrap `Type::List(Type::Dyn)`
    // only, confirmed by reading it -- there is no `List(n)`-style
    // annotation syntax generalizing that sugar to a user-defined named
    // type, and this task's own scope (typecheck.rs/parser.rs only, no
    // new declaration OR annotation syntax, per the design spec's own
    // "Generalizing to Type::Indexed" section) doesn't add one either.
    // So both tests below parse an ordinary match EXPRESSION through the
    // real parser (getting real Pattern nodes, a real Match ExprRef, and
    // a real named_types registry for the recursive alias), then drive
    // typecheck::check_against DIRECTLY against a hand-built Ctx binding
    // the scrutinee to the Case-B shape this task's own eligibility
    // logic is meant to recognize -- exactly mirroring this file's own
    // existing `extend_generalized_and_lookup_mint_a_fresh_index_variable...`
    // test's identical precedent for exercising typecheck's own
    // internals below the parser's reach.

    #[test]
    fn case_b_base_case_hypothesis_is_real_and_enforced() {
        // Case A's own `base_case_arm_alone_is_rejected_when_it_does_not_fit_vec_zero`
        // test, adapted for Case B: the base alternative (`Bool`, the
        // one NOT structurally containing List) implies index := 0.
        // `v`'s own real type is always the literal, bare
        // `Indexed(Named(id), n)` -- so the only way to observe the
        // base arm's own n=0 hypothesis is through a REAL,
        // index_subst-consulting comparison (unify_fits/
        // unify_index_expr), not coerce's own static permissiveness.
        // Requiring the base arm's own body (bare `v`) to fit
        // `Indexed(Named(id), 1)` forces exactly that comparison: a
        // genuine, provable "1 != 0" conflict if (and only if) the
        // n=0 hypothesis is actually active.
        //
        // Base arm uses `true`/`false` literal patterns, not a bare
        // `b`/wildcard Var -- a Var pattern trivially "could match"
        // BOTH alternatives (pattern_could_match's own generic
        // consistent()-with-Dyn fallback), which this task's own
        // eligibility logic correctly treats as ambiguous (no
        // hypothesis at all, the same safe fallback Case A's own
        // catch-all `_ => {}` uses) -- exactly the arm shape needed
        // to observe a REAL hypothesis, not a Var that would earn none.
        // Both together also satisfy missing_case's own has_true &&
        // has_false exhaustiveness heuristic.
        use crate::index_expr::IndexExpr;
        use crate::typecheck::{check_against, extend, Ctx, InferCtx};
        use std::rc::Rc;
        use types::Type;

        let src = r#"
            type List = (Int, List) | Bool in
            match v
            | (h, t) -> let x: Dyn = 0 in x
            | true -> v
            | false -> v
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        // Look up the specific "List" alias's own minted id directly,
        // rather than relying on `HashMap` iteration order (fragile the
        // moment a second `type` declaration exists in the same source)
        // -- `fresh_named_type_id` (parser.rs) mints ids as `"{name}#{n}"`,
        // so the surface name is always the part before the first `#`.
        let id = named_types.keys().find(|k| k.split('#').next() == Some("List")).unwrap().clone();
        let mut infer = InferCtx::new(named_types);
        let scrut_ty = Type::Indexed(Rc::new(Type::Named(id.clone())), Rc::new(IndexExpr::Var("n".to_string())));
        let ctx = extend(&Ctx::empty(), "v", scrut_ty);
        let expected = Type::Indexed(Rc::new(Type::Named(id)), Rc::new(IndexExpr::Lit(1)));
        let result = check_against(&mut arena, root, &expected, &ctx, &spans, &mut infer);
        assert!(
            result.is_err(),
            "expected a real index conflict from the base alternative's own n=0 hypothesis (1 vs 0), got: {result:?}"
        );
    }

    #[test]
    fn case_b_step_case_hypothesis_and_retyping_enable_a_real_recursive_proof() {
        // Case A's own `recursive_vec_function_typechecks_via_pattern_refinement`
        // test, adapted for Case B: the step alternative (`(Int, List)`,
        // the one containing List exactly once) mints a fresh index
        // variable `m`, retypes the self-referential sub-binding `t` as
        // `Indexed(Named(id), m)` (overriding bind_pattern_vars's own
        // plain `Named(id)` binding), and hypothesizes the scrutinee's
        // own index is `m + 1`. Requiring the step arm's own body (bare
        // `t`) to fit `Indexed(Named(id), n - 1)` only type-checks if
        // BOTH halves are real: resolve_index_deep must actually see
        // n -> m + 1 (the hypothesis) for "(m + 1) - 1" to SOP-normalize
        // to exactly "m" (t's own real, retyped index) -- exactly the
        // same composition Case A's own analogous test relies on,
        // independent of whether the wrapped type is List or a
        // qualifying Named+Union.
        use crate::index_expr::IndexExpr;
        use crate::typecheck::{check_against, extend, Ctx, InferCtx};
        use std::rc::Rc;
        use types::Type;

        let src = r#"
            type List = (Int, List) | Bool in
            match v
            | (h, t) -> t
            | true -> let x: Dyn = 0 in x
            | false -> let x: Dyn = 0 in x
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        // Same fragility fix as the base-case test just above: look up
        // "List"'s own minted id directly instead of relying on
        // `HashMap` iteration order.
        let id = named_types.keys().find(|k| k.split('#').next() == Some("List")).unwrap().clone();
        let mut infer = InferCtx::new(named_types);
        let scrut_ty = Type::Indexed(Rc::new(Type::Named(id.clone())), Rc::new(IndexExpr::Var("n".to_string())));
        let ctx = extend(&Ctx::empty(), "v", scrut_ty);
        let expected = Type::Indexed(
            Rc::new(Type::Named(id)),
            Rc::new(IndexExpr::Sub(Rc::new(IndexExpr::Var("n".to_string())), Rc::new(IndexExpr::Lit(1)))),
        );
        let result = check_against(&mut arena, root, &expected, &ctx, &spans, &mut infer);
        assert!(
            result.is_ok(),
            "expected the step alternative's own m+1 hypothesis to make t: Indexed(Named(id), m) satisfy \
             Indexed(Named(id), n - 1) via SOP normalization, got: {:?}",
            result.err()
        );
    }

    // --- Task 3, fix round 1: critical review finding ---

    #[test]
    fn bind_pattern_vars_union_arm_does_not_infinite_loop_on_a_bare_self_referential_alternative() {
        // Critical review finding, confirmed via a traced counterexample
        // (see this task's own fix-round-1 report entry): reachable via
        // ORDINARY source syntax, no check_against-bypass needed.
        // `named_types["List"]` resolves to `Union([Named("List"),
        // Tuple([Int, Named("List")])])` -- the FIRST alternative is a
        // BARE self-reference, not wrapped in anything structural. The
        // pre-fix bind_pattern_vars's own Type::Union arm ran its
        // `pattern_could_match` reachability probe with a fresh, empty
        // `visiting` set every time, with no memory of the enclosing
        // Type::Named arm's own unfolding -- so it would judge
        // `Named("List")` "reachable" (via ITS OWN nested unfold finding
        // the genuine Tuple alternative two levels down, using yet
        // another fresh empty set), recurse into it, unfold to the same
        // Union again, and repeat forever: unbounded recursion / stack
        // overflow, purely from typechecking this one match expression.
        //
        // With the fix (visiting threaded through, mirroring
        // pattern_could_match's own already-correct Type::Named/
        // Type::Union arms exactly), the Union arm's reachability probe
        // sees "List" already in `visiting` and correctly skips the bare
        // self-referential alternative, landing on the genuine
        // Tuple([Int, Named("List")]) alternative instead -- so this
        // must terminate and type-check successfully (the match body
        // just returns an Int literal from each arm; nothing further
        // constrains t's own type).
        //
        // Task 4's own verification found this test as originally
        // committed used `parser::parse` + `typecheck::check` -- which
        // per `parse`'s own doc comment ("pairing this `parse` with plain
        // `typecheck::check`... is exactly the condition every
        // Type::Named consumer's own missing-registry-entry fallback
        // exists to handle gracefully") means `infer.named_types` is
        // ALWAYS EMPTY here, so `Type::Named("List")` never actually
        // unfolds at all -- it hits the graceful "unknown id" fallback
        // immediately, every time, regardless of whether the cycle-guard
        // fix above is present. Confirmed empirically: this test as
        // originally written keeps passing even with the fix fully
        // reverted (the pre-fix code, and even a version with NO cycle
        // guard at all), because the buggy code path is never reached.
        // Switched to `parse_with_named_types` + `check_with_named_types`
        // (the same pairing Task 3's own Case B tests already use)
        // to make this test genuinely exercise the fix -- re-verified
        // below.
        let src = r#"
            type List = List | (Int, List) in
            let f: (List -> Int) = fun v ->
                match v
                | (h, t) -> 0
                | x -> 1
            in 0
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types);
        assert!(
            elaborated.is_ok(),
            "expected the exact repro program to type-check successfully once the Union arm's own visiting set \
             is correctly threaded (no cycle, t correlates against the genuine Tuple alternative), got: {:?}",
            elaborated.err()
        );
        let result = machine::run(&arena, elaborated.unwrap(), Env::prelude(), &spans);
        assert_eq!(result.as_int(), 0);
    }

    // --- Task 4: coverage gap (finding 6) ---

    #[test]
    fn bind_pattern_vars_falls_back_to_fresh_vars_at_a_cycle_detected_nested_position() {
        // Finding 6 (Task 4 brief): the cycle-detected fallback branch in
        // bind_pattern_vars's own Type::Named arm (`if visiting.contains(id)
        // { return bind_fresh_positions(...) }`, added by Task 3's own
        // fix-round-1) is correct by inspection and was re-verified there,
        // but wasn't exercised by any existing test -- every existing
        // Case B test puts a bare Var at the self-referential position
        // (`t` in `(h, t)`), which binds via Pattern::Var's own arm
        // directly and never re-enters bind_pattern_vars's Type::Named
        // match at all. Triggering the cycle-detected branch specifically
        // needs a NESTED structural pattern (not a bare Var) at that
        // position, so bind_pattern_vars recurses back into a
        // Pattern::List/Type::Named combination with "List" already in
        // `visiting` from the outer unfold -- no third level of recursive
        // type nesting needed, just one more level of pattern nesting on
        // top of the existing two-alternative `List | (Int, List)` type.
        //
        // `(h2, t2)` at the self-ref position: bind_pattern_vars unfolds
        // "List" once for the outer `(h, ..)` (visiting: {} -> {List}),
        // lands on the Tuple alternative, then recurses into position 1
        // with pat=(h2, t2) (a nested Pattern::List) against
        // ty=Named("List") -- this time "List" is ALREADY in visiting, so
        // the cycle guard fires and h2/t2 both get fresh, uncorrelated
        // types via bind_fresh_positions, rather than h2:Int/t2:Named(id)
        // the way a genuinely one-more-level unfold would give them. This
        // is a real, deliberate conservative approximation (mirroring
        // pattern_could_match's own identical stance) -- it fires here
        // even though this specific two-level pattern would have
        // terminated fine with one more unfold, sacrificing precision for
        // a termination guarantee that doesn't inspect how deep the
        // pattern itself goes.
        //
        // Observed via h2: if h2 got the fresh, unconstrained type the
        // cycle guard produces, checking it against a required `Bool`
        // trivially succeeds (permissive, like Type::Dyn). If h2 had
        // instead been correlated to its real position-0 type (`Int`,
        // from `(Int, List)`), the same check would be a genuine,
        // provable mismatch.
        //
        // Uses `parse_with_named_types` + `check_with_named_types` (not
        // plain `parse`/`check`) -- Task 4's own verification found that
        // pairing plain `parse` with plain `check` leaves
        // `infer.named_types` empty (see `parse`'s own doc comment), so
        // `Type::Named("List")` would never actually unfold at all here,
        // making the test vacuous regardless of the cycle guard. See the
        // fix applied just above, to the pre-existing
        // `bind_pattern_vars_union_arm_does_not_infinite_loop_on_a_bare_self_referential_alternative`
        // test, for the same finding.
        let src = r#"
            type List = List | (Int, List) in
            let f: (List -> Int) = fun v ->
                match v
                | (h, (h2, t2)) -> let proof: Bool = h2 in 0
                | x -> 1
            in 0
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types);
        assert!(
            elaborated.is_ok(),
            "expected the cycle-detected fallback to give h2 a fresh, unconstrained type (permissive with \
             Bool), got: {:?}",
            elaborated.err()
        );
        let result = machine::run(&arena, elaborated.unwrap(), Env::prelude(), &spans);
        assert_eq!(result.as_int(), 0);
    }

    #[test]
    fn list_lit_checked_against_indexed_synthesizes_precise_length() {
        let src = r#"
            let f: (Int -> Vec(3)) = fun n -> [n, n, n] in
            f(5)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[5, 5, 5]");
    }

    #[test]
    fn list_lit_checked_against_indexed_rejects_wrong_length() {
        let src = r#"
            let f: (Int -> Vec(3)) = fun n -> [n, n] in
            f(5)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("(2)") && err.0.contains("(3)"), "expected an index-mismatch error showing 2 vs 3, got: {}", err.0);
    }

    #[test]
    fn cons_checked_against_indexed_checks_tail_at_decremented_length() {
        let src = r#"
            let result: Vec(3) = 0 :: [1, 2] in
            result
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[0, 1, 2]");
    }

    #[test]
    fn cons_checked_against_indexed_rejects_wrong_tail_length() {
        let src = r#"
            let result: Vec(3) = 0 :: [1, 2, 3] in
            result
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        // Tightened (final-review Finding 5): the tail's own Check-mode
        // arm decrements the annotation's index symbolically (`3 - 1`,
        // never simplified to `2`) and compares it against the tail
        // literal's own real, synthesized length (3) -- checking both
        // stable, predictable values involved, mirroring
        // list_lit_checked_against_indexed_rejects_wrong_length's own
        // style, instead of the universal "type mismatch" substring.
        assert!(
            err.0.contains("3 - 1") && err.0.contains("found [Int](3)"),
            "expected an index-mismatch error showing the tail's expected length (3 - 1) vs its real length (3), got: {}",
            err.0
        );
    }

    #[test]
    fn letrec_annotated_self_reference_recurses_at_a_decremented_index() {
        // The Motivation example's own root-cause repro, isolated from Gap
        // 1: today this fails typecheck with "infinite index expression: n
        // occurs in n - 1" because `len`'s self-reference is bound
        // monomorphically (plain `extend`) before its body -- containing
        // the recursive call `len(t)` where `t: Vec(n - 1)` -- is
        // elaborated. Returns a plain Int, not a Vec, so it exercises Gap
        // 2 without depending on Gap 1's construction fix at all.
        let src = r#"
            let rec len: (Vec(n) -> Int) = fun v ->
                match v
                | [] -> 0
                | h :: t -> 1 + len(t)
            in
            let v0: Vec(3) = [1, 2, 3] in
            len(v0)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 3);
    }

    #[test]
    fn letrec_annotated_self_reference_generalizes_across_two_different_call_sites() {
        // Final-review Finding 6: every existing recursive-function test
        // in this branch calls its function exactly once, so none of them
        // distinguishes "the index variable is genuinely generalized/
        // polymorphic across calls" (Gap 2's own fix, via
        // extend_generalized/lookup's per-use instantiation) from "it
        // happened to get monomorphically bound at one specific length."
        // Reuses `len`'s own shape from
        // letrec_annotated_self_reference_recurses_at_a_decremented_index
        // above, called at two DIFFERENT lengths in the same program --
        // if `n` were monomorphically bound (Gap 2 unfixed, or a
        // regression of its fix), the second call would fail to
        // type-check against whatever length the first call bound `n`
        // to.
        let src = r#"
            let rec len: (Vec(n) -> Int) = fun v ->
                match v
                | [] -> 0
                | h :: t -> 1 + len(t)
            in
            let v3: Vec(3) = [1, 2, 3] in
            let v5: Vec(5) = [1, 2, 3, 4, 5] in
            (len(v3), len(v5))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        match machine::run(&arena, elaborated, Env::prelude(), &spans) {
            Value::List(items) => assert_eq!((items[0].as_int(), items[1].as_int()), (3, 5)),
            other => panic!("expected a tuple, got {other}"),
        }
    }

    #[test]
    fn letrec_mutual_annotated_recursion_composes_across_the_binding_group() {
        // The composition case the design spec's own Testing Strategy
        // flags as "not reasoned through to full certainty during design"
        // -- two Vec(n)-indexed bindings, each recursing into the OTHER at
        // a decremented index, exercising the per-binding extend_generalized
        // loop's iterative composition rather than a single self-reference.
        let src = r#"
            let rec evens: (Vec(n) -> Vec(n)) = fun v ->
                match v
                | [] -> []
                | h :: t -> h :: odds(t)
            and odds: (Vec(n) -> Vec(n)) = fun v ->
                match v
                | [] -> []
                | h :: t -> h :: evens(t)
            in
            let v0: Vec(4) = [1, 2, 3, 4] in
            evens(v0)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[1, 2, 3, 4]");
    }

    #[test]
    fn letrec_unannotated_self_reference_stays_monomorphic() {
        // Confirms the unannotated path is byte-for-byte unchanged: still
        // plain `extend`, still today's exact behavior.
        let src = r#"
            let rec fact = fun n ->
                if n == 0 then 1 else n * fact(n - 1)
            in fact(5)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 120);
    }

    #[test]
    fn same_length_recursive_vec_function_end_to_end() {
        // The design spec's own Motivation example, run in full: a
        // recursive function that both CONSUMES and PRODUCES a
        // length-indexed Vec(n), needing Gap 1 (Cons/ListLit mode-aware
        // construction, Tasks 1-2) AND Gap 2 (LetRec generalized
        // self-reference, Task 3) together.
        let src = r#"
            let rec same_length: (Vec(n) -> Vec(n)) = fun v ->
                match v
                | [] -> []
                | h :: t -> h :: same_length(t)
            in
            let v0: Vec(3) = [1, 2, 3] in
            same_length(v0)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[1, 2, 3]");
    }

    #[test]
    fn same_length_recursive_vec_function_rejects_a_genuinely_wrong_length() {
        // A hand-written arm that returns the wrong length -- the [] base
        // case should produce Vec(0), not the required Vec(n) at n=0's own
        // own call depth here it's a direct top-level mismatch: the
        // function's OWN declared return type is Vec(n), but this arm
        // returns a 1-element list no matter what v's length is.
        let src = r#"
            let rec bad: (Vec(n) -> Vec(n)) = fun v ->
                match v
                | [] -> [0]
                | h :: t -> h :: bad(t)
            in
            let v0: Vec(3) = [1, 2, 3] in
            bad(v0)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        // Tightened (final-review Finding 5): the [] base-case arm's own
        // hypothesis pins its own index to 0 (Case A pattern refinement),
        // then unify_index_expr rejects that literal-0 index against the
        // function's own already-established `n` decremented once (index
        // 1 for the top-level call at length 3, one match-arm level in)
        // -- a stable, predictable "index 0 does not unify with index 1"
        // message. Checking both index literals involved, not just the
        // universal "type mismatch" substring.
        assert!(
            err.0.contains("index 0") && err.0.contains("index 1"),
            "expected an index-mismatch error showing 0 vs 1, got: {}",
            err.0
        );
    }

    #[test]
    fn tuple_checked_against_case_b_indexed_checks_step_case() {
        // Deviation from the task-6 brief's literal test source: the
        // brief wrote only `| false -> 0` for the Bool arm. missing_case
        // (unaffected by Case B, confirmed by this same file's own
        // case_b_base_case_hypothesis_is_real_and_enforced comment) is a
        // purely structural, type-agnostic heuristic that requires BOTH
        // `true` and `false` literals (or a bare Var) once any Bool
        // literal pattern appears -- a lone `false` reports "the missing
        // Bool case" regardless of what the scrutinee's real type is.
        // Adding the dead `true -> 0` arm satisfies that pre-existing,
        // deliberately-unrelated exhaustiveness rule without touching
        // what this test actually exercises (Case B step construction).
        let src = r#"
            type List = (Int, List) | Bool in
            let f: (Int -> List(1)) = fun n -> (n, false) in
            match f(9)
            | (h, t) -> h
            | true -> 0
            | false -> 0
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 9);
    }

    #[test]
    fn tuple_checked_against_case_b_indexed_rejects_wrong_arity() {
        let src = r#"
            type List = (Int, List) | Bool in
            let f: (Int -> List(1)) = fun n -> (n, n, false) in
            f(9)
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let err = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap_err();
        // Tightened (final-review Finding 5): a 3-tuple isn't step-shaped
        // (step_alt is a 2-tuple), so this falls through to the ordinary
        // generic fallback -- a stable, predictable "expected List(1),
        // found (Int, Int, Bool)" message. Checking both the expected
        // indexed type and the actual synthesized tuple type, not just
        // the universal "type mismatch" substring.
        assert!(
            err.0.contains("List(1)") && err.0.contains("(Int, Int, Bool)"),
            "expected a plain type mismatch (3-tuple isn't step-shaped) naming List(1) and (Int, Int, Bool), got: {}",
            err.0
        );
    }

    #[test]
    fn base_case_construction_against_case_b_indexed_unifies_index_to_zero() {
        let src = r#"
            type List = (Int, List) | Bool in
            let f: (Int -> List(0)) = fun n -> false in
            f(9)
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        assert!(!machine::run(&arena, elaborated, Env::prelude(), &spans).as_bool());
    }

    #[test]
    fn plain_tuple_construction_with_no_indexed_expected_type_is_unaffected() {
        let src = "let t: (Int, Int) = (1, 2) in t";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        match machine::run(&arena, elaborated, Env::prelude(), &spans) {
            Value::List(items) => assert_eq!((items[0].as_int(), items[1].as_int()), (1, 2)),
            other => panic!("expected a tuple, got {other}"),
        }
    }

    #[test]
    fn recursive_function_over_case_b_eligible_named_type_end_to_end() {
        // Mirrors same_length_recursive_vec_function_end_to_end (Task 4),
        // but for a user-defined Case B type instead of the built-in Vec:
        // a recursive function that both consumes and produces a
        // length-indexed List(n), needing Task 5 (syntax), Task 6
        // (construction) and Task 3 (LetRec) all together.
        let src = r#"
            type List = (Int, List) | Bool in
            let rec same_length: (List(n) -> List(n)) = fun v ->
                match v
                | (h, t) -> (h, same_length(t))
                | false -> false
                | true -> false
            in
            let v0: List(2) = (1, (2, false)) in
            match same_length(v0)
            | (h, t) -> h
            | false -> 0
            | true -> 0
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 1);
    }

    #[test]
    fn case_b_base_case_hook_excludes_an_already_indexed_actual_type() {
        // Final-review Finding 1: the base-case hook's Dyn/Var guard only
        // excluded Type::Dyn/Type::Var on the `actual_ty` side, not on
        // `base_alt`. `consistent` treats Dyn as universally consistent
        // with anything, so a qualifying named union whose BASE
        // alternative happens to be Dyn (like this `T`) made the hook
        // wrongly treat ANY actual_ty -- including `w`'s real
        // Indexed(Named(T), 2) -- as satisfying the base case, force-
        // asserting its index to 0 (a genuine conflict with the real
        // index 2) and failing typecheck with a spurious error.
        //
        // The fix shipped is NOT a symmetric Dyn/Var exclusion on
        // `base_alt` -- that was tried and found to regress the
        // legitimate `let x: T(0) = 5 in x` case (Int can only ever
        // satisfy an Indexed(Named(_), _) through base_alt's own Dyn
        // permissiveness, since `consistent` has no Indexed-vs-non-
        // Indexed arm). Instead, `actual_ty` itself is excluded when it
        // already resolves to Type::Indexed AND base_alt is Dyn/Var --
        // the real discriminator is whether `actual_ty` already carries
        // its own independently-established index, not what base_alt
        // looks like. With that guard, `w`'s case falls through to the
        // ordinary trailing unify_fits check instead, which correctly
        // confirms `w`'s type is still T(2).
        let src = r#"
            type T = Dyn | (Int, T) in
            let v: T(2) = (1, (2, 5)) in
            let w: T(2) = v in
            0
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 0);
    }

    #[test]
    fn case_b_base_case_hook_still_handles_an_indexed_base_alternative() {
        // Scoped re-review of Finding 1's fix (see the base-case hook's
        // own comment in typecheck.rs): excluding Type::Indexed(..)
        // actual_ty unconditionally would also block the legitimate case
        // where base_alt is ITSELF Indexed (not Dyn/Var) and actual_ty
        // genuinely is that base case. The shipped guard only excludes
        // Indexed actual_ty when base_alt is Dyn/Var, so this must still
        // unify the index to 0 via the hook, not fall through to the
        // fallback (which has no Indexed-vs-Named consistent arm and
        // would reject it).
        let src = r#"
            type T = Vec(3) | (Int, T) in
            let v: Vec(3) = [1, 2, 3] in
            let b: T(0) = v in
            0
        "#;
        let (mut arena, spans, root, named_types) = parser::parse_with_named_types(src).unwrap();
        let elaborated = typecheck::check_with_named_types(&mut arena, root, &spans, named_types).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 0);
    }

    // Typecheck does NOT reject unbound names (gradual typing), so an
    // unbound name in a function that is never called must stay harmless:
    // the panic is lazy (VarRef::Unbound), not raised when the program is
    // resolved.
    #[test]
    fn unbound_variable_in_an_uncalled_function_is_harmless() {
        assert_eq!(run_source("let f = fun n -> n + m in 3").unwrap(), Outcome::Int(3));
    }

    // A local binding shadows a prelude builtin (Local wins over Prelude).
    #[test]
    fn local_binding_shadows_a_prelude_builtin_at_runtime() {
        assert_eq!(run_source("let len = fun x -> 42 in len([1, 2, 3])").unwrap(), Outcome::Int(42));
    }

    // A group function reaches a sibling defined later, through its own
    // rebuilt [names…, param] frame.
    #[test]
    fn let_rec_group_sibling_and_param_share_one_frame() {
        let src = "let rec a = fun n -> if n == 0 then 0 else b(n - 1) + 1 \
                    and b = fun m -> if m == 0 then 0 else a(m - 1) + 1 \
                    in a(9)";
        assert_eq!(run_source(src).unwrap(), Outcome::Int(9));
    }

    // Non-group `let rec` fallback (Frame::LetRecBody, machine.rs's
    // Expr::LetRec else-branch): `x` and `w` are not functions, so they
    // evaluate left to right in the OUTER scope (both see the lambda's `z`,
    // neither sees the other) and are bound plainly for the body only.
    #[test]
    fn let_rec_fallback_binds_non_function_values_for_the_body() {
        let src = "(fun z -> let rec x = z + 1 and w = z * 2 in x + w)(10)";
        assert_eq!(run_source(src).unwrap(), Outcome::Int(31));
    }

    // A duplicate pattern binder (`x` twice in one arm): resolve.rs's
    // `lookup` takes the LAST slot with that name, so the second `x` (bound
    // to 2) is what the arm body sees.
    #[test]
    fn duplicate_pattern_binder_last_one_wins() {
        let src = "match [1, 2] | [x, x] -> x | _ -> -1";
        assert_eq!(run_source(src).unwrap(), Outcome::Int(2));
    }

    // Three-member mutual recursion group (Frame::Many, not One/Two): each
    // function body's frame is [a, b, c, n], modelled on
    // corpus/mutual_three_way.rn. a(9) -> b(8) -> c(7) -> ... -> a(0) = 1.
    #[test]
    fn three_way_mutual_recursion_group_shares_a_many_slot_frame() {
        let src = "let rec a = fun n -> if n == 0 then 1 else b(n - 1) \
                    and b = fun n -> if n == 0 then 2 else c(n - 1) \
                    and c = fun n -> if n == 0 then 3 else a(n - 1) \
                    in a(9)";
        assert_eq!(run_source(src).unwrap(), Outcome::Int(1));
    }

    // Spec §2 decision lock: `f`'s value is an `If`, not a direct Lambda, so
    // is_direct_group is false even though both branches are functions --
    // is_direct_group is syntactic, not semantic. That routes the whole
    // group through the same non-recursive fallback as
    // `let_rec_fallback_binds_non_function_values_for_the_body` above, so
    // `f` is NOT in scope while its own value is elaborated/evaluated.
    // Typecheck accepts this (LetRec's val_ctx binds every name, including
    // `f`, with a fresh type var while elaborating each value -- see
    // typecheck.rs's Expr::LetRec arm), but the runtime resolver does not:
    // calling `f` recurses into the `if`'s `fun n -> ... f(n - 1) ...`
    // branch, where `f` was resolved in the OUTER scope and is nowhere
    // bound, so it panics lazily via VarRef::Unbound the same way
    // `unbound_variable_panic_reports_its_location` does.
    #[test]
    fn non_direct_let_rec_self_reference_is_unbound_at_runtime() {
        let src = "let rec f = if true then fun n -> if n == 0 then 0 else f(n - 1) \
                    else fun n -> n in f(3)";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("unbound variable: f"), "unexpected message: {err}");
    }

    #[test]
    fn indexing_a_non_indexable_alias_reports_it_is_not_indexable() {
        let err = run_source("type Pair = (Int, Int) in let x: Pair(3) = (1, 2) in x").unwrap_err();
        assert!(err.contains("(Int, Int) is not an indexable type"), "unexpected message: {err}");
    }

    #[test]
    fn step_shaped_tuple_through_a_let_keeps_its_index() {
        let t = "type T = Dyn | (Int, T) in\n";
        assert!(run_source(&format!("{t}let p = (1, 5) in let w: T(1) = p in 0")).is_ok());
        assert!(run_source(&format!("{t}let p = (1, (2, 5)) in let w: T(2) = p in 0")).is_ok());
        assert!(run_source(&format!("{t}let p = (1, 5) in let w: T(0) = p in 0")).is_ok());
        assert!(run_source(&format!("{t}let p = (1, 5) in let w: T(2) = p in 0")).is_err());
        // Dyn base alternative: the inner (2, 5) is a valid T(0).
        assert!(run_source(&format!("{t}let p = (1, (2, 5)) in let w: T(1) = p in 0")).is_ok());
        assert!(run_source(&format!("{t}let p = (true, 5) in let w: T(1) = p in 0")).is_err());
    }

    #[test]
    fn literal_argument_checks_against_an_indexed_parameter() {
        let len = "let rec len: (Vec(n) -> Int) = fun v ->\n match v | [] -> 0 | h :: t -> 1 + len(t) in\n";
        let run = |call: &str| run_source(&format!("{len}{call}"));
        assert_eq!(run("len([1, 2, 3])").unwrap().as_int(), 3);
        assert_eq!(run("len([len([1, 2]), 5])").unwrap().as_int(), 2);
        let fixed = "let f = fun v: Vec(2) -> 0 in\n";
        assert!(run_source(&format!("{fixed}f([1, 2])")).is_ok());
        assert!(run_source(&format!("{fixed}f([1, 2, 3])")).is_err());
    }

    #[test]
    fn literal_checks_nested_indexed_element_types_in_tuples_and_lists() {
        let ok = |s: &str| run_source(s).is_ok();
        // via annotated let
        assert!(ok("let q: (Vec(2), Int) = ([1, 2], 3) in 0"));
        assert!(!ok("let q: (Vec(2), Int) = ([1, 2, 3], 3) in 0"));
        assert!(ok("let q: [Vec(2)] = [[1, 2], [3, 4]] in 0"));
        assert!(!ok("let q: [Vec(2)] = [[1, 2], [3]] in 0"));
        // directly as a call argument
        let f = "let f = fun p: (Vec(2), Int) -> 0 in\n";
        assert!(ok(&format!("{f}f(([1, 2], 3))")));
        assert!(!ok(&format!("{f}f(([1, 2, 3], 3))")));
    }

    // Same conflicts as the two `*_annotation_is_still_rejected` tests
    // above, but the Tuple/List is built by an unannotated `let` first, so
    // literal Check-mode never sees it: only unify_fits's own Tuple/List
    // arms can catch the shared-`n` conflict.
    #[test]
    fn a_vec_conflict_in_a_non_literal_tuple_or_list_is_caught_by_unify_fits() {
        let pre = "let ca: Dyn = [1, 2, 3] in let cb: Dyn = [1, 2] in\n\
                   let wa: Vec(3) = ca in let wb: Vec(2) = cb in\n";
        let tuple = format!("{pre}let ta = (wa, 1) in let tb = (wb, 2) in\n\
                             let pa: (Vec(n), Int) = ta in let pb: (Vec(n), Int) = tb in 1");
        let list = format!("{pre}let xa = [wa] in let xb = [wb] in\n\
                            let la: [Vec(n)] = xa in let lb: [Vec(n)] = xb in 1");
        for src in [tuple, list] {
            let err = run_source(&src).unwrap_err();
            assert!(err.contains("index 3 does not unify with index 2"), "unexpected message: {err}");
        }
    }

    // --- rigid index variables (spec 2026-10-07) ---

    fn rigid_infer(names: &[&str]) -> crate::typecheck::InferCtx {
        let mut infer = crate::typecheck::InferCtx::new(std::collections::HashMap::new());
        for n in names {
            infer.rigid_index.insert((*n).to_string());
        }
        infer
    }

    fn var(n: &str) -> crate::index_expr::IndexExpr {
        crate::index_expr::IndexExpr::Var(n.to_string())
    }

    #[test]
    fn rigid_index_variable_refuses_to_bind_to_a_literal() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::unify_index_expr;
        let span = crate::span::Span { start: 0, end: 0 };
        let mut infer = rigid_infer(&["n"]);
        let err = unify_index_expr(&var("n"), &IndexExpr::Lit(3), &mut infer, span).unwrap_err();
        assert!(err.0.contains("index variable n is fixed by the signature"), "unexpected: {}", err.0);
        // Same with the sides swapped, and n stays unbound.
        assert!(unify_index_expr(&IndexExpr::Lit(3), &var("n"), &mut infer, span).is_err());
        assert_eq!(infer.resolve_index(&var("n")), var("n"));
    }

    #[test]
    fn flexible_index_variable_yields_to_a_rigid_one_in_either_order() {
        use crate::typecheck::unify_index_expr;
        let span = crate::span::Span { start: 0, end: 0 };
        let mut infer = rigid_infer(&["n"]);
        unify_index_expr(&var("n"), &var("m"), &mut infer, span).unwrap();
        assert_eq!(infer.resolve_index(&var("m")), var("n"));
        let mut infer = rigid_infer(&["n"]);
        unify_index_expr(&var("k"), &var("n"), &mut infer, span).unwrap();
        assert_eq!(infer.resolve_index(&var("k")), var("n"));
        assert_eq!(infer.resolve_index(&var("n")), var("n"));
    }

    #[test]
    fn two_distinct_rigid_variables_do_not_unify_but_a_rigid_equals_itself_modulo_sop() {
        use crate::index_expr::IndexExpr;
        use crate::typecheck::unify_index_expr;
        use std::rc::Rc;
        let span = crate::span::Span { start: 0, end: 0 };
        let mut infer = rigid_infer(&["n", "m"]);
        assert!(unify_index_expr(&var("n"), &var("m"), &mut infer, span).is_err());
        // n against n + 0 is SOP-equal, so it is accepted without binding.
        let n_plus_0 = IndexExpr::Add(Rc::new(var("n")), Rc::new(IndexExpr::Lit(0)));
        unify_index_expr(&var("n"), &n_plus_0, &mut infer, span).unwrap();
        assert_eq!(infer.resolve_index(&var("n")), var("n"));
    }

    #[test]
    fn rigid_signature_rejects_a_body_that_ignores_its_parameters_length() {
        for src in [
            "let rec f: (Vec(n) -> Vec(n)) = fun v -> [1, 2, 3] in 0",
            "let f: (Vec(n) -> Vec(n)) = fun v -> [1, 2, 3] in 0",
        ] {
            let err = run_source(src).unwrap_err();
            assert!(err.contains("fixed by the signature"), "{src}: unexpected message: {err}");
        }
    }

    #[test]
    fn rigid_signature_rejects_other_unprovable_bodies() {
        let rejected = [
            // two distinct signature variables are not equal
            "let f: (Vec(n) -> Vec(m)) = fun v -> v in 0",
            // wrong base-case length: hypothesis n = 0, body returns length 1
            "let rec f: (Vec(n) -> Vec(n)) = fun v -> match v | [] -> [1] | h :: t -> h :: f(t) in 0",
            // off by one, symbolic
            "let rec f: (Vec(n) -> Vec(n + 1)) = fun v -> 0 :: 0 :: v in 0",
        ];
        for src in rejected {
            assert!(run_source(src).is_err(), "should be rejected: {src}");
        }
    }

    #[test]
    fn rigid_signature_still_accepts_provably_correct_bodies() {
        let lists = [
            ("let rec f: (Vec(n) -> Vec(n)) = fun v -> v in f([1, 2])", "[1, 2]"),
            ("let rec f: (Vec(n) -> Vec(n)) = fun v -> match v | [] -> [] | h :: t -> h :: f(t) in f([1, 2, 3])", "[1, 2, 3]"),
            ("let rec f: (Vec(n) -> Vec(n + 1)) = fun v -> 0 :: v in f([1, 2])", "[0, 1, 2]"),
            (
                "let rec app: (Vec(m) -> Vec(n) -> Vec(m + n)) = fun a -> fun b -> match a | [] -> b | h :: t -> h :: app(t)(b) in app([1, 2])([3])",
                "[1, 2, 3]",
            ),
        ];
        for (src, want) in lists {
            assert_eq!(run_source(src).unwrap().to_string(), want, "{src}");
        }
        // the same signature used at two different lengths in one program
        let two = "let rec f: (Vec(n) -> Vec(n)) = fun v -> v in let a: Vec(2) = f([1, 2]) in let b: Vec(3) = f([1, 2, 3]) in 0";
        assert_eq!(run_source(two).unwrap().as_int(), 0);
        // Dyn is the escape hatch
        let dyn_escape = "let rec f: (Vec(n) -> Vec(n)) = fun v -> let r: Dyn = [1, 2, 3] in r in 0";
        assert_eq!(run_source(dyn_escape).unwrap().as_int(), 0);
    }

    #[test]
    fn rigid_cons_arm_fresh_variable_cannot_be_pinned_to_reach_the_signature_variable() {
        // The cons arm's hypothesis n := m + 1 uses a fresh m; pinning m in the
        // body must not let a length-5 result through a Vec(n) -> Vec(n) signature.
        for src in [
            "let rec f: (Vec(n) -> Vec(n)) = fun v -> match v | [] -> [] | h :: t -> let w: Vec(4) = t in [1, 2, 3, 4, 5] in f([1, 2])",
            "let rec f: (Vec(n) -> Vec(n)) = fun v -> match v | [] -> [] | h :: t -> let w: Vec(4) = f(t) in [1, 2, 3, 4, 5] in f([1, 2])",
        ] {
            assert!(run_source(src).is_err(), "should be rejected: {src}");
        }
    }

    #[test]
    fn rigid_variables_are_released_after_the_binding() {
        // After f is checked, a later, unrelated annotation reusing the name
        // `n` must be free to bind it again.
        let src = "let f: (Vec(n) -> Vec(n)) = fun v -> v in let x: Vec(n) = [1, 2] in 0";
        assert_eq!(run_source(src).unwrap().as_int(), 0);
    }

    #[test]
    fn rigid_mutual_recursion_group_shares_a_variable_name() {
        let src = "let rec even: (Vec(n) -> Int) = fun v -> match v | [] -> 1 | h :: t -> odd(t) \
                   and odd: (Vec(n) -> Int) = fun v -> match v | [] -> 0 | h :: t -> even(t) in even([1, 2])";
        assert_eq!(run_source(src).unwrap().as_int(), 1);
    }

    #[test]
    fn signature_variable_shadows_an_earlier_existential_of_the_same_name() {
        // `let a: Vec(n) = [1, 2]` binds the existential n := 2 for the rest of
        // the program; a later function signature reusing the name `n` must
        // still quantify its own variable, not inherit 2.
        let accepted = [
            ("let a: Vec(n) = [1, 2] in let f: (Vec(n) -> Vec(n)) = fun v -> v in f([1, 2, 3])", "[1, 2, 3]"),
            ("let a: Vec(n) = [1, 2] in let rec f: (Vec(n) -> Vec(n)) = fun v -> v in f([1, 2, 3])", "[1, 2, 3]"),
            // a body annotation reusing `n` denotes the signature's variable
            ("let a: Vec(n) = [1, 2] in let f: (Vec(n) -> Vec(n)) = fun v -> let w: Vec(n) = v in w in f([1, 2, 3])", "[1, 2, 3]"),
            ("let a: Vec(n) = [1, 2] in let f: (Vec(n) -> Vec(n)) = fun v: Vec(n) -> v in f([1, 2, 3])", "[1, 2, 3]"),
        ];
        for (src, want) in accepted {
            assert_eq!(run_source(src).unwrap().to_string(), want, "{src}");
        }
        // ...and stays rigid: a body that only fits the old n = 2 is rejected.
        let src = "let a: Vec(n) = [1, 2] in let f: (Vec(n) -> Vec(n)) = fun v -> [1, 2] in 0";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("fixed by the signature"), "unexpected message: {err}");
        // The existential itself is untouched: a later non-function reuse still conflicts.
        let src = "let a: Vec(n) = [1, 2] in let f: (Vec(n) -> Vec(n)) = fun v -> v in let b: Vec(n) = [1, 2, 3] in 0";
        assert!(run_source(src).is_err(), "existential n = 2 must still conflict with length 3");
        // Deliberate consequence: a signature cannot capture that existential.
        // `Int -> Vec(n)` promises every length, so returning the length-2 `a` is rejected.
        let src = "let a: Vec(n) = [1, 2] in let f: (Int -> Vec(n)) = fun k -> a in 0";
        assert!(run_source(src).unwrap_err().contains("fixed by the signature"));
    }

    // A Dyn value crossing into Vec(n) is checked at runtime against n's
    // value. Inside a function whose parameter is Vec(n), that value is the
    // parameter's length. These used to panic with "unbound variable: n".
    #[test]
    fn dyn_to_vec_n_boundary_checks_against_the_parameters_length_when_called() {
        let mismatched = [
            // rigid signature, Dyn escape hatch rebinding n
            "let rec f: (Vec(n) -> Vec(n)) = fun v -> let r: Dyn = [1, 2, 3] in let q: Vec(n) = r in q in let a: Vec(2) = f([1, 2]) in a",
            // the dyn_escape test's own program, now actually called
            "let rec f: (Vec(n) -> Vec(n)) = fun v -> let r: Dyn = [1, 2, 3] in r in f([1, 2])",
            // flexible, inline parameter annotation
            "let f = fun v: Vec(n) -> let r: Dyn = [1, 2, 3] in let q: Vec(n) = r in q in f([1, 2])",
            // higher-order contract: the wrapped function's result is checked against its argument's length
            "let g: Dyn = fun x -> [1] in let h: (Vec(n) -> Vec(n)) = g in h([1, 2])",
        ];
        for src in mismatched {
            let err = run_source(src).unwrap_err();
            assert!(err.contains("type error: expected [Dyn](n"), "{src}: unexpected message: {err}");
        }
        let matching = [
            ("let rec f: (Vec(n) -> Vec(n)) = fun v -> let r: Dyn = [1, 2, 3] in let q: Vec(n) = r in q in let a: Vec(3) = f([1, 2, 3]) in a", "[1, 2, 3]"),
            ("let rec f: (Vec(n) -> Vec(n)) = fun v -> let r: Dyn = [1, 2, 3] in r in f([4, 5, 6])", "[1, 2, 3]"),
            ("let f = fun v: Vec(n) -> let r: Dyn = [1, 2, 3] in let q: Vec(n) = r in q in f([1, 2, 3])", "[1, 2, 3]"),
            ("let g: Dyn = fun x -> x in let h: (Vec(n) -> Vec(n)) = g in h([1, 2])", "[1, 2]"),
            // inside the cons arm n is still the parameter's length (n = m + 1)
            ("let rec f: (Vec(n) -> Vec(n)) = fun v -> match v | [] -> [] | h :: t -> let r: Dyn = v in let q: Vec(n) = r in q in f([1, 2])", "[1, 2]"),
            // a shadowed parameter name does not change what n means
            ("let rec f: (Vec(n) -> Vec(n)) = fun v -> let v = 0 in let r: Dyn = [1, 2] in let q: Vec(n) = r in q in f([7, 8])", "[1, 2]"),
        ];
        for (src, want) in matching {
            assert_eq!(run_source(src).unwrap().to_string(), want, "{src}");
        }
    }

    #[test]
    fn dyn_to_vec_n_boundary_ignores_a_term_variable_that_shares_the_index_name() {
        // The typechecker never links a term `n` to the index `n`, so the
        // check must use the parameter's length, not the term's value.
        let src = "let rec f: (Vec(n) -> Int -> Vec(n)) = fun v -> fun n -> let r: Dyn = [1, 2, 3] in let q: Vec(n) = r in q in let a: Vec(2) = f([1, 2])(3) in a";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("type error: expected [Dyn](n"), "unexpected message: {err}");
        let src = "let make = fun n: Int -> fun v: Vec(n) -> let r: Dyn = v in let q: Vec(n) = r in q in \
                   let x: Dyn = [1, 2, 3] in let va: Vec(3) = x in len(make(5)(va))";
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    #[test]
    fn a_vec_n_lambda_whose_check_never_runs_is_checked_on_entry_when_called_through_dyn() {
        // Fun-to-Dyn wrapping (spec 2026-10-08) runs DOWN at the wrapper's
        // entry, so an untyped non-list is now rejected there even though the
        // body's own check never runs. The untouched-on-entry property still
        // holds where no wrapper is built (reached through an untyped
        // parameter, see the `..._untyped_parameter_..._known_limitation` test).
        let src = "let f = fun v: Vec(n) -> if true then 0 else (let r: Dyn = [1] in let q: Vec(n) = r in 0) in \
                   let d: Dyn = f in d(5)";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("type error"), "unexpected message: {err}");
    }

    #[test]
    fn a_dyn_argument_into_a_vec_n_polymorphic_function_needs_only_a_list_when_its_length_is_unconstrained() {
        // The call site's instantiated n has no runtime binder and nothing
        // ever binds it (spec 2026-10-07-dyn-crossing-deferred-length-checks),
        // so the crossing checks is_list only. It used to fail cleanly.
        let src = "let f = fun v: Vec(n) -> v in let x: Dyn = [1, 2] in len(f(x))";
        assert_eq!(run_source(src).unwrap().as_int(), 2);
        // A non-list still fails the crossing cleanly.
        let src = "let f = fun v: Vec(n) -> v in let x: Dyn = 5 in len(f(x))";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("type error: expected [Dyn]("), "unexpected message: {err}");
        assert!(!err.contains("unbound variable"), "unexpected message: {err}");
    }

    #[test]
    fn dyn_to_vec_n_boundary_with_no_runtime_value_for_n_is_a_clean_error() {
        // Two crossings share the existential n, which has no runtime binder:
        // the types claim equal lengths with nothing to compare against, so
        // the check fails cleanly instead of panicking or passing.
        let src = "let x: Dyn = [1, 2, 3] in let y: Vec(n) = x in let z: Vec(n) = x in len(y)";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("index variable n has no runtime value"), "unexpected message: {err}");
        // ...but an n already known statically is compared directly.
        let src = "let a: Vec(n) = [1, 2] in let x: Dyn = [1, 2] in let y: Vec(n) = x in len(y)";
        assert_eq!(run_source(src).unwrap().as_int(), 2);
    }

    #[test]
    fn vec_n_witness_check_on_a_non_list_argument_is_a_clean_error_not_a_panic() {
        // An untyped caller passes a non-list for `v: Vec(n)`; the check
        // site reads the witness length and used to panic inside `len`.
        let src = "let f = fun v: Vec(n) -> (let r: Dyn = [1,2,3] in let q: Vec(n) = r in 0) in let d: Dyn = f in d(5)";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("type error: expected [Dyn], found Int"), "unexpected message: {err}");
        assert!(!err.contains("len expects"), "unexpected message: {err}");
        // A real list witness still works.
        let src = "let f = fun v: Vec(n) -> (let r: Dyn = [1,2,3] in let q: Vec(n) = r in 0) in let d: Dyn = f in d([4,5,6])";
        assert_eq!(run_source(src).unwrap().as_int(), 0);
    }

    // --- Dyn crossings into an unbound Vec(n): deferred length checks (spec 2026-10-07) ---

    fn obligations_after_checking(src: &str) -> usize {
        use crate::typecheck::{check_against, Ctx, InferCtx};
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let mut infer = InferCtx::new(std::collections::HashMap::new());
        check_against(&mut arena, root, &crate::types::Type::Int, &Ctx::empty(), &spans, &mut infer).unwrap();
        infer.obligation_count()
    }

    #[test]
    fn a_dyn_crossing_records_an_obligation_only_when_its_index_has_no_runtime_value() {
        let deferred = [
            "let x: Dyn = [1, 2, 3] in let y: Vec(n) = x in 0",
            "let f = fun v: Vec(n) -> v in let x: Dyn = [1, 2] in let y = f(x) in 0",
            "let f = fun v: Vec(n + 1) -> v in let x: Dyn = [1, 2] in let y = f(x) in 0",
        ];
        for src in deferred {
            assert_eq!(obligations_after_checking(src), 1, "{src}");
        }
        let checked_at_the_crossing = [
            // a literal index
            "let x: Dyn = [1, 2] in let y: Vec(2) = x in 0",
            // n already bound statically
            "let a: Vec(n) = [1, 2] in let x: Dyn = [1, 2] in let y: Vec(n) = x in 0",
            // n is the parameter's length (a witness)
            "let f = fun v: Vec(n) -> let r: Dyn = [1] in let q: Vec(n) = r in 0 in 0",
            // a union target is out of scope: unchanged, no obligation
            "let x: Dyn = [1] in let y: Vec(n) | Int = x in 0",
        ];
        for src in checked_at_the_crossing {
            assert_eq!(obligations_after_checking(src), 0, "{src}");
        }
    }

    #[test]
    fn a_variable_with_a_pending_obligation_is_not_generalized() {
        // The caller's annotation decides n inside the function that contains
        // the crossing, so one use pins it and a second use at another length
        // is a static conflict. Generalizing would give each use a fresh n.
        let conflicting = [
            "let f = fun v: Vec(n) -> v in let x: Dyn = [1, 2, 3] in let mk = fun u -> f(x) in \
             let a: Vec(3) = mk(0) in let b: Vec(4) = mk(0) in 0",
            "let g: (Int -> Vec(k)) = fun i -> let r: Dyn = [1, 2] in r in \
             let a: Vec(2) = g(1) in let b: Vec(3) = g(2) in 0",
        ];
        for src in conflicting {
            let (mut arena, spans, root) = parser::parse(src).unwrap();
            let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
            assert!(err.0.contains("type mismatch"), "{src}: unexpected message: {}", err.0);
        }
        // A crossing whose variable ends up equal to a Vec(n) parameter's
        // length is decided by that witness, so f stays polymorphic.
        let src = "let f = fun v: Vec(n) -> let g = fun u: Vec(k) -> u in let x: Dyn = [1, 2] in let z: Vec(n) = g(x) in z in \
                   let a = f([1, 2]) in let b = f([1, 2, 3]) in 0";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok(), "{src}");
    }

    #[test]
    fn a_deferred_length_check_uses_the_bindings_known_when_typechecking_finishes() {
        // Each crossing's index variable is still unbound when the crossing is
        // elaborated; something later decides it. (matching, wrong length, len)
        let cases = [
            // the caller's annotation
            ("let f = fun v: Vec(n) -> v in let x: Dyn = [1, 2, 3] in let y: Vec(3) = f(x) in len(y)",
             "let f = fun v: Vec(n) -> v in let x: Dyn = [1, 2] in let y: Vec(3) = f(x) in len(y)", 3),
            // through a monomorphic wrapper: the check runs inside mk
            ("let f = fun v: Vec(n) -> v in let x: Dyn = [1, 2, 3] in let mk = fun u -> f(x) in let a: Vec(3) = mk(0) in len(a)",
             "let f = fun v: Vec(n) -> v in let x: Dyn = [1, 2] in let mk = fun u -> f(x) in let a: Vec(3) = mk(0) in len(a)", 3),
            // a sibling annotation after the crossing binds the existential n
            ("let x: Dyn = [7, 8] in let y: Vec(n) = x in let z: Vec(n) = [7, 8] in len(y)",
             "let x: Dyn = [7] in let y: Vec(n) = x in let z: Vec(n) = [7, 8] in len(y)", 2),
            // an enclosing signature's result type
            ("let g: (Int -> Vec(2)) = fun i -> let f = fun v: Vec(k) -> v in let r: Dyn = [1, 2] in f(r) in len(g(0))",
             "let g: (Int -> Vec(2)) = fun i -> let f = fun v: Vec(k) -> v in let r: Dyn = [1] in f(r) in len(g(0))", 2),
            // the crossing's variable ends up equal to a Vec(n) parameter's length
            ("let f = fun v: Vec(n) -> let g = fun u: Vec(k) -> u in let x: Dyn = [1, 2] in let z: Vec(n) = g(x) in z in len(f([5, 6]))",
             "let f = fun v: Vec(n) -> let g = fun u: Vec(k) -> u in let x: Dyn = [1, 2] in let z: Vec(n) = g(x) in z in len(f([5, 6, 7]))", 2),
        ];
        for (good, bad, want) in cases {
            assert_eq!(run_source(good).unwrap().as_int(), want, "{good}");
            let err = run_source(bad).unwrap_err();
            assert!(err.contains("type error: expected [Dyn]("), "{bad}: unexpected message: {err}");
        }
    }

    #[test]
    fn a_deferred_check_over_a_variable_with_no_runtime_value_stays_a_clean_error() {
        // In the cons arm n = m + 1, and the crossing's k ends up equal to
        // m + 1. The arm's m has no runtime value (only n, the parameter's
        // length, has a witness), so the check fails cleanly, never skipped.
        let src = "let rec f: (Vec(n) -> Int) = fun v -> match v | [] -> 0 | h :: t -> \
                   let g = fun u: Vec(k) -> u in let x: Dyn = [1] in let z: Vec(n) = g(x) in 0 in f([1, 2])";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("has no runtime value"), "unexpected message: {err}");
    }

    #[test]
    fn a_dyn_crossing_into_an_unconstrained_vec_n_only_checks_that_it_is_a_list() {
        // n is never bound and no other crossing mentions it: any list passes.
        let accepted = [
            ("let x: Dyn = [1, 2, 3] in let y: Vec(n) = x in len(y)", 3),
            ("let x: Dyn = [] in let y: Vec(n) = x in len(y)", 0),
        ];
        for (src, want) in accepted {
            assert_eq!(run_source(src).unwrap().as_int(), want, "{src}");
        }
        // Still a clean runtime error, never skipped:
        let rejected = [
            // not a list at all
            ("let x: Dyn = 5 in let y: Vec(n) = x in 0", "type error: expected [Dyn](n), found"),
            // two crossings sharing n
            ("let x: Dyn = [1, 2] in let y: Vec(n) = x in let z: Vec(n) = x in 0", "index variable n has no runtime value"),
            // a compound index over an unbound variable (no equation solving)
            ("let f = fun v: Vec(n + 1) -> v in let x: Dyn = [1, 2] in len(f(x))", "has no runtime value"),
            // a union alternative is out of scope: unchanged
            ("let x: Dyn = [1] in let y: Vec(n) | Int = x in 0", "has no runtime value"),
        ];
        for (src, want) in rejected {
            let err = run_source(src).unwrap_err();
            assert!(err.contains(want), "{src}: unexpected message: {err}");
        }
        // The union's other alternative still matches a non-list.
        assert_eq!(run_source("let x: Dyn = 5 in let y: Vec(n) | Int = x in 0").unwrap().as_int(), 0);
    }

    #[test]
    fn a_dyn_crossing_inside_a_function_body_keeps_the_clean_error_for_an_unbound_n() {
        // One crossing site, run twice: both results share the one unbound n,
        // so g would trust unequal-length lists. Never accepted.
        let lambda = "let mk = fun d: Dyn -> (let y: Vec(n) = d in y) in \
                      let g = fun a: Vec(k) -> fun b: Vec(k) -> len(b) in \
                      let x1: Dyn = [1, 2] in let x2: Dyn = [1, 2, 3] in g(mk(x1))(mk(x2))";
        let rec_body = "let rec mk = fun d: Dyn -> (let y: Vec(n) = d in y) in \
                        let g = fun a: Vec(k) -> fun b: Vec(k) -> len(b) in \
                        let x1: Dyn = [1, 2] in let x2: Dyn = [1, 2, 3] in g(mk(x1))(mk(x2))";
        for src in [lambda, rec_body] {
            let err = run_source(src).unwrap_err();
            assert!(err.contains("has no runtime value"), "{src}: unexpected message: {err}");
        }
        // A top-level call site is not inside the callee's body: is_list only.
        let src = "let f = fun v: Vec(n) -> v in let x: Dyn = [1, 2, 3] in len(f(x))";
        assert_eq!(run_source(src).unwrap().as_int(), 3);
    }

    #[test]
    fn a_dyn_function_contract_wrapper_keeps_the_clean_error_for_an_unbound_n() {
        // wrap_fun_contract's wrapper runs on every call, so its return check
        // is inside a function body: k(1) and k(2) share the one unbound n.
        let src = "let h: Dyn = fun i -> if i < 2 then [1,2] else [1,2,3] in \
                   let k: (Int -> Vec(n)) = h in \
                   let g = fun a: Vec(m) -> fun b: Vec(m) -> len(b) in \
                   g(k(1))(k(2))";
        let err = run_source(src).unwrap_err();
        assert!(err.contains("has no runtime value"), "unexpected message: {err}");
    }

    #[test]
    fn an_unbound_vec_n_unified_with_a_match_arm_length_keeps_the_clean_error() {
        // Inside a cons arm the tail's length m is hypothesis-bound (n = m+1),
        // but a unification of q with m survives the arm, so q is not isolated.
        let a9 = "let x: Dyn = [1, 2, 3] in let y: Vec(n) = x in let x2: Dyn = [1] in let z: Vec(q) = x2 in \
                  let r: Int = (match y | [] -> 100 | h :: t -> (let s: Vec(q) = t in len(z) - len(s))) in r";
        let a10 = "let x2: Dyn = [1] in let z: Vec(q) = x2 in \
                   let f = fun v: Vec(n) -> (let r: Int = (match v | [] -> 0 | h :: t -> (let s: Vec(q) = t in len(z) - len(s))) in r) in \
                   f([1, 2, 3]) * 100 + f([1, 2, 3, 4, 5])";
        // the other orientation: the arm's variable is aliased to q through g
        let a4 = "let x: Dyn = [1, 2, 3] in let y: Vec(n) = x in let x2: Dyn = [1] in let z: Vec(q) = x2 in \
                  let g = fun a: Vec(k) -> fun b: Vec(k) -> len(a) - len(b) in \
                  let r: Int = (match y | [] -> 100 | h :: t -> g(t)(z)) in r";
        for src in [a9, a10, a4] {
            let err = run_source(src).unwrap_err();
            assert!(err.contains("has no runtime value"), "{src}: unexpected message: {err}");
        }
        // A crossing inside an arm that is unified with nothing stays is_list-only.
        let a8 = "let x: Dyn = [1, 2, 3] in let y: Vec(n) = x in \
                  let r: Int = (match y | [] -> 100 | h :: t -> (let x2: Dyn = [1] in let z: Vec(q) = x2 in len(z) - len(t))) in r";
        assert_eq!(run_source(a8).unwrap().as_int(), -1);
    }

    #[test]
    fn a_cons_arm_body_cannot_pin_the_tail_length_variable() {
        // The arm's fresh m stands for len(tail) on every run of the arm, so
        // pinning it to 0 (which used to outlive the arm) is a static error.
        // v2 has no Dyn at all and was unsound before.
        let h = "let d1: Dyn = [1,2,3] in let z: Vec(q) = d1 in";
        let g = "let g = fun a: Vec(k) -> fun b: Vec(k) -> len(a) - len(b) in";
        let programs = [
            format!("{h} let w: Vec(0) = (match z | h :: t -> t | _ -> []) in len(w)"),
            format!("{g} {h} let r: Int = (match z | h :: t -> g(t)([]) | _ -> 0) in r"),
            format!("{g} {h} let r: Int = (match z | h :: t -> (let s: Vec(0) = t in g(z)([5])) | _ -> 0) in r"),
            "let x: Dyn=[1,2,3] in let y: Vec(n) = x in let x2: Dyn=[1] in \
             let r: Int = (match y | [] -> 100 | h :: t -> (let z: Vec(q) = x2 in let s: Vec(0) = t in len(z))) in r + len(y)"
                .to_string(),
            "let f = fun z: Vec(q) -> (let w: Vec(0) = (match z | h :: t -> t | _ -> []) in len(w)) in f([1,2,3])".to_string(),
        ];
        for src in programs {
            let err = run_source(&src).unwrap_err();
            assert!(err.contains("is fixed by the signature and cannot equal 0"), "{src}: unexpected message: {err}");
        }
    }

    #[test]
    fn a_cons_arm_tail_length_variable_stays_rigid_after_the_arm() {
        // q aliased to the arm's m inside the arm: pinning q afterwards pins m.
        let g = "let g = fun a: Vec(k) -> fun b: Vec(k) -> len(a) - len(b) in";
        let programs = [
            (format!("let x: Dyn = [1,2,3] in let y: Vec(n) = x in let x2: Dyn = [] in let z: Vec(q) = x2 in {g} let r: Int = (match y | [] -> 100 | h :: t -> g(t)(z)) in let w: Vec(0) = z in r"), "cannot equal 0"),
            (format!("let x: Dyn = [1,2,3] in let y: Vec(n) = x in let x2: Dyn = [9] in let z: Vec(q) = x2 in {g} let r: Int = (match y | [] -> 100 | h :: t -> g(t)(z)) in let w: Vec(1) = z in r"), "cannot equal 1"),
            (format!("{g} let f = fun y: Vec(n) -> fun z: Vec(q) -> (let r: Int = (match y | [] -> 100 | h :: t -> g(t)(z)) in let w: Vec(0) = z in r) in f([1,2,3])([])"), "cannot equal 0"),
        ];
        for (src, want) in programs {
            let err = run_source(&src).unwrap_err();
            assert!(err.contains("is fixed by the signature and") && err.contains(want), "{src}: unexpected message: {err}");
        }
    }

    // --- Fun-to-Dyn / Fun-to-Fun wrapping (spec 2026-10-08), closure form ---

    fn has_let_named(arena: &expr::Arena, root: expr::ExprRef, name: &str) -> bool {
        tree_any(arena, root, &|n| match n {
            expr::Expr::Lambda(param, ..) => param.starts_with(name),
            expr::Expr::Let(var, ..) => var.starts_with(name),
            _ => false,
        })
    }

    fn elaborated_has_wrapper(src: &str) -> bool {
        let (mut arena, spans, root, named) = parser::parse_with_named_types(src).unwrap();
        let out = typecheck::check_with_named_types(&mut arena, root, &spans, named).unwrap();
        has_let_named(&arena, out, "__cf#")
    }

    #[test]
    fn a_typed_function_converted_to_dyn_is_wrapped() {
        for src in [
            "let h = fun a: Vec(3) -> len(a) in let hd: Dyn = h in hd([1])",
            "let f = fun a: Int -> a + 1 in let d: Dyn = f in d(true)",
            "(fun k: (Dyn -> Dyn) -> k(true)) (fun a: Int -> a + 1)",
            "let ap = fun k: ((Int|Bool) -> Int) -> k(true) in ap(fun x: Int -> x + 1)",
            "let ap = fun k: (Dyn -> Int) -> k(true) in ap(fun x: (Int, Int) -> 1)",
            "let ap = fun k: (Dyn -> Int) -> k(true) in ap(fun x: [Int] -> 1)",
        ] {
            let err = run_source(src).expect_err(src);
            assert!(err.contains("type error"), "{src}: unexpected message: {err}");
            assert!(!err.contains("expected a number"), "{src}: {err}");
        }
        for (src, want) in [
            ("let f = fun a: Int -> a + 1 in let d: Dyn = f in d(2)", 3),
            ("let ap = fun k: ((Int|Bool) -> Int) -> k(1) in ap(fun x: Int -> x + 1)", 2),
            ("let f = fun a: Int -> a + 1 in let d: Dyn = f in let g: (Int -> Int) = d in g(2)", 3),
            ("let f = fun x: Int -> x in let apply = fun g -> g(1) in apply(f)", 1),
        ] {
            assert_eq!(run_source(src).unwrap().as_int(), want, "{src}");
        }
    }

    #[test]
    fn re_crossed_function_rejects_a_bad_argument_statically() {
        let err = run_source("let f = fun a: Int -> a + 1 in let d: Dyn = f in let g: (Int -> Int) = d in g(true)").expect_err("static mismatch");
        assert!(err.contains("type mismatch"), "{err}");
    }

    #[test]
    fn a_dyn_crossed_curried_vec_n_function_compares_lengths_across_calls() {
        let curried = "let d = fun a: Vec(n) -> fun b: Vec(n) -> len(b) in let x: Dyn = d in ";
        assert!(run_source(&format!("{curried}x([1])([1, 2])")).is_err());
        assert_eq!(run_source(&format!("{curried}x([1, 2])([3, 4])")).unwrap().as_int(), 2);
    }

    #[test]
    fn a_dyn_crossed_vec_n_function_does_not_fail_spuriously() {
        assert_eq!(run_source("let f = fun v: Vec(n) -> len(v) in let d: Dyn = f in d([1, 2, 3])").unwrap().as_int(), 3);
        assert_eq!(run_source("let f = fun v: Vec(n+1) -> len(v) in let d: Dyn = f in d([1, 2])").unwrap().as_int(), 2);
    }

    #[test]
    fn a_contravariant_callback_is_checked_through_the_upcast_recursion() {
        let err = run_source("let ap = fun k: ((Int -> Int) -> Int) -> k(fun x: Int -> x) in let d: Dyn = ap in d(fun cb -> cb(true))").expect_err("bad callback result");
        assert!(err.contains("expected Int, found Bool"), "{err}");
    }

    #[test]
    fn a_returned_function_is_upcast_through_dyn() {
        let mk = "let mk = fun a: Int -> fun b: Int -> 7 in let d: Dyn = mk in ";
        let err = run_source(&format!("{mk}d(1)(true)")).expect_err("bad returned-function argument");
        assert!(err.contains("expected Int, found Bool"), "{err}");
        assert_eq!(run_source(&format!("{mk}d(1)(2)")).unwrap().as_int(), 7);
        assert!(run_source(&format!("{mk}d(true)(2)")).is_err());
        let mk3 = "let mk = fun a: Int -> fun b: Int -> fun c: Int -> 7 in let d: Dyn = mk in ";
        assert!(run_source(&format!("{mk3}d(1)(2)(true)")).is_err());
        assert_eq!(run_source(&format!("{mk3}d(1)(2)(3)")).unwrap().as_int(), 7);
    }

    #[test]
    fn a_witnessed_non_bare_index_in_a_returned_function_keeps_its_length_check() {
        let h = "let h = fun a: Vec(n) -> fun b: Vec(n+1) -> len(b) in let x: Dyn = h in ";
        assert_eq!(run_source(&format!("{h}x([1])([1, 2])")).unwrap().as_int(), 2);
        assert!(run_source(&format!("{h}x([1])([1, 2, 3])")).is_err());
    }

    #[test]
    fn a_named_return_through_dyn_terminates_without_upcast() {
        // Only an annotated binding gives a source Fun a recursive Named return that typechecks today.
        assert_eq!(run_source("type F = Int | (Int, F) in let mk: (Int -> F) = fun a: Int -> a in let d: Dyn = mk in d(1)").unwrap().as_int(), 1);
    }

    #[test]
    // Only the entry call crosses the wrapper; the recursion inside go is direct.
    fn a_deep_tail_loop_entered_through_a_dyn_wrapper_completes() {
        let src = "let rec go: (Int -> Int -> Int) = fun i: Int -> fun acc: Int -> if i == 0 then acc else go(i - 1)(acc + 1) in let d: Dyn = go in d(100000)(0)";
        assert_eq!(run_source(src).unwrap().as_int(), 100000);
    }

    // Every LetRec in an elaborated tree is still a direct Lambda group
    // (resolve::is_direct_group); otherwise the recursive name becomes unbound.
    fn let_rec_groups_direct(arena: &expr::Arena, root: expr::ExprRef, out: &mut Vec<bool>) {
        use expr::Expr;
        let mut go = |r: &expr::ExprRef| let_rec_groups_direct(arena, *r, out);
        match &arena[root] {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Token(_) | Expr::Var(_) => {}
            Expr::Check(operand, _) => go(operand),
            Expr::ListLit(items) | Expr::Tuple(items) => items.iter().for_each(go),
            Expr::Lambda(_, _, body) => go(body),
            Expr::App(f, a) => {
                go(f);
                go(a);
            }
            Expr::Let(_, _, val, body) => {
                go(val);
                go(body);
            }
            Expr::LetRec(bindings, body) => {
                let direct = resolve::is_direct_group(arena, bindings);
                bindings.iter().for_each(|(_, _, v)| go(v));
                go(body);
                out.push(direct);
            }
            Expr::BinOp(_, l, r) => {
                go(l);
                go(r);
            }
            Expr::If(c, t, e) => {
                go(c);
                go(t);
                go(e);
            }
            Expr::Perform(_, p) => go(p),
            Expr::Handle { body, handler } => {
                go(body);
                go(handler);
            }
            Expr::MakeHandler { body, .. } => go(body),
            Expr::Match(s, arms) => {
                go(s);
                for (_, g, b) in arms.iter() {
                    g.iter().for_each(&mut go);
                    go(b);
                }
            }
            Expr::Record(fields) => fields.iter().for_each(|(_, v)| go(v)),
            Expr::FieldAccess(..) => unreachable!(),
        }
    }

    fn elaborated_let_rec_groups(src: &str) -> Vec<bool> {
        let (mut arena, spans, root, named) = parser::parse_with_named_types(src).unwrap();
        let out = typecheck::check_with_named_types(&mut arena, root, &spans, named).unwrap();
        let mut groups = Vec::new();
        let_rec_groups_direct(&arena, out, &mut groups);
        groups
    }

    #[test]
    fn a_curried_literal_at_a_dyn_position_checks_each_stage() {
        let d = "let d: Dyn = fun a: Int -> fun b: Int -> a + b in ";
        let err = run_source(&format!("{d}d(1)(true)")).expect_err("bad second argument");
        assert!(err.contains("type error") && err.contains("expected Int, found Bool"), "{err}");
        assert!(run_source(&format!("{d}d(true)(2)")).is_err());
        assert_eq!(run_source(&format!("{d}d(1)(2)")).unwrap().as_int(), 3);
    }

    // The DOWN check calls builtins (is_int, fail); a parameter
    // named like one must not capture them.
    #[test]
    fn a_literal_parameter_named_like_a_builtin_does_not_capture_the_check() {
        assert_eq!(run_source("let d: Dyn = fun is_int: Int -> is_int + 1 in d(2)").unwrap().as_int(), 3);
        for name in ["fail", "is_int", "is_list"] {
            let src = format!("let d: Dyn = fun {name}: Int -> {name} + 1 in d(true)");
            let err = run_source(&src).expect_err(&src);
            assert!(err.contains("type error: expected Int, found Bool"), "{src}: {err}");
        }
        let rec = "let rec is_int: Dyn = fun is_int: Int -> if is_int < 1 then 0 else 1 in is_int(3)";
        assert_eq!(elaborated_let_rec_groups(rec), vec![true]);
    }

    #[test]
    fn a_literal_is_rebuilt_in_place_not_wrapped_in_a_closure() {
        // Literal form: no `let __cf = ... in fun __ca` outer closure.
        assert!(!elaborated_has_wrapper("let d: Dyn = fun a: Int -> fun b: Int -> a + b in d(1)(2)"));
        assert!(elaborated_has_wrapper("let f = fun a: Int -> a + 1 in let d: Dyn = f in d(2)"));
    }

    #[test]
    fn an_annotated_let_rec_literal_at_dyn_stays_a_direct_recursive_group() {
        let src = "let rec f: Dyn = fun n: Int -> if n < 1 then 0 else f(n - 1) in ";
        assert_eq!(elaborated_let_rec_groups(&format!("{src}f(3)")), vec![true]);
        assert_eq!(run_source(&format!("{src}f(3)")).unwrap().as_int(), 0);
        let err = run_source(&format!("{src}f(true)")).expect_err("bad argument");
        assert!(err.contains("type error") && err.contains("expected Int, found Bool"), "{err}");
    }

    // The error location of a rejected literal-form call: the first line is
    // "line L, column C: ..." (spec 8). `let d: Dyn = fun a: Int -> a + 1 in
    // d(true)` -> "line 1, column 14: type error: expected Int, found Bool",
    // the span of the cast site (the literal crossing into Dyn); it used to
    // be column 39, the argument `true`, whatever ran last.
    #[test]
    fn a_rejected_literal_call_reports_a_location() {
        let err = run_source("let d: Dyn = fun a: Int -> a + 1 in d(true)").expect_err("bad argument");
        assert!(err.starts_with("line 1, column 14: "), "{err}");
        assert!(err.contains("expected Int, found Bool"), "{err}");
    }

    #[test]
    fn a_literal_vec_n_lambda_at_dyn_only_checks_is_list_and_is_a_pinned_known_limitation() {
        let d = "let d: Dyn = fun a: Vec(n) -> fun b: Vec(n) -> len(b) in ";
        // PINNED parked limitation (spec 7): the literal form never compares
        // lengths, so mismatched lengths are accepted.
        assert_eq!(run_source(&format!("{d}d([1])([1, 2])")).unwrap().as_int(), 2);
        assert!(run_source(&format!("{d}d(5)")).is_err());
        assert!(run_source(&format!("{d}d([1])(5)")).is_err());
    }

    #[test]
    fn a_loose_vec_literal_through_a_dyn_hop_is_rejected_when_the_index_is_fixed() {
        let err = run_source("let k: Dyn = fun x: Vec(3) -> 1 in k([1])").expect_err("wrong length");
        assert!(err.contains("type error"), "{err}");
        assert_eq!(run_source("let k: Dyn = fun x: Vec(3) -> 1 in k([1, 2, 3])").unwrap().as_int(), 1);
    }

    #[test]
    fn a_dyn_callee_argument_is_cast_to_dyn() {
        let err = run_source("let f = fun a: Int -> a + 1 in map(f)([true])").expect_err("bad element");
        assert!(err.contains("type error") && !err.contains("expected a number"), "{err}");
        assert_eq!(run_source("let f = fun a: Int -> a + 1 in len(map(f)([1, 2]))").unwrap().as_int(), 2);
    }

    #[test]
    fn a_perform_payload_function_is_cast_to_dyn() {
        let src = "handle perform op(fun a: Int -> a + 1) with handler op(p, resume) -> resume(p(true))";
        let err = run_source(src).expect_err("bad application");
        assert!(err.contains("type error") && !err.contains("expected a number"), "{err}");
        let ok = "handle perform op(fun a: Int -> a + 1) with handler op(p, resume) -> resume(p(1))";
        assert_eq!(run_source(ok).unwrap().as_int(), 2);
    }

    #[test]
    fn a_wrapped_function_resumed_twice_is_checked_each_time() {
        let src = "handle (let f = perform choose(0) in f(10)) with handler choose(p, resume) -> resume(fun a: Int -> a + 1) + resume(fun a: Int -> a * 2)";
        assert_eq!(run_source(src).unwrap().as_int(), 31);
        let bad = "handle (let f = perform choose(0) in f(true)) with handler choose(p, resume) -> resume(fun a: Int -> a + 1) + resume(fun a: Int -> a * 2)";
        assert!(run_source(bad).unwrap_err().contains("type error: expected Int, found Bool"));
        let lit = "let rec f: Dyn = fun n: Int -> if n < 1 then 0 else f(n - 1) in handle (perform choose(0)) + f(2) with handler choose(p, resume) -> resume(1) + resume(2)";
        assert_eq!(elaborated_let_rec_groups(lit), vec![true]);
        assert_eq!(run_source(lit).unwrap().as_int(), 3);
    }

    #[test]
    fn a_literal_loop_that_re_enters_its_wrapper_every_iteration_completes() {
        let src = "let rec f: Dyn = fun n: Int -> if n < 1 then 0 else f(n - 1) in f(100000)";
        assert_eq!(run_source(src).unwrap().as_int(), 0);
    }

    #[test]
    fn a_union_target_is_not_wrapped_is_a_known_limitation() {
        let src = "let h: ((Dyn -> Dyn) | Bool) = fun x: Int -> x in (fun k: (Dyn -> Dyn) -> k(true))(h)";
        // Union targets are not descended: the cast is skipped and the argument is unchecked (spec 7).
        assert_eq!(run_source(src).unwrap(), Outcome::Bool(true));
    }

    #[test]
    fn a_named_target_is_not_wrapped_is_a_known_limitation() {
        let src = "type F = Dyn -> F in let g: F = fun x: Int -> fun y: Int -> y in let d: Dyn = g in d(true)";
        // Named targets are not descended: `d(true)` returns the inner function instead of rejecting `true` (spec 7).
        assert!(matches!(run_source(src).unwrap(), Outcome::Function));
    }

    #[test]
    fn a_var_returning_call_into_a_dyn_callback_is_rejected_today_known_limitation() {
        let src = "let id = fun k -> k in (fun k: (Dyn -> Dyn) -> k(true))(id(fun x: Int -> x + 1))";
        // Current behaviour: the bad call IS rejected, blaming the cast site (the
        // argument `id(..)`, column 57; it used to blame the `true` in `k(true)`, column 50); pinned so a change shows.
        let err = run_source(src).expect_err("rejected today");
        assert!(err.starts_with("line 1, column 57: type error: expected Int, found Bool"), "{err}");
    }

    #[test]
    fn unannotated_code_emits_no_function_wrapper() {
        for src in [
            "let f = fun x: Int -> x in let apply = fun g -> g(1) in apply(f)",
            "map(fun x -> x + 1)([1,2])",
        ] {
            assert!(!elaborated_has_wrapper(src), "{src}");
        }
        assert!(elaborated_has_wrapper("let f = fun a: Int -> a + 1 in let d: Dyn = f in d(2)"));
    }

    #[test]
    fn synthesized_builtin_calls_are_not_captured_by_user_bindings() {
        for (src, want) in [
            ("let is_int = 0 in len(map(fun x: Int -> x + 1)([1, 2]))", 2),
            ("let g = fun is_int: Bool -> len(map(fun x: Int -> x + 1)([1, 2])) in g(true)", 2),
            ("let is_list = 0 in len(map(fun v: [Int] -> len(v))([[1], [2]]))", 2),
            ("let len = 0 in let f = fun a: Vec(n) -> fun b: Vec(n) -> 7 in let d: Dyn = f in d([1])([2])", 7),
            ("let is_fun = 0 in let d: Dyn = fun x: Int -> x in d(1)", 1),
            ("let is_bool = 0 in let d: Dyn = fun x: Bool -> 1 in d(true)", 1),
            ("let is_float = 0 in let d: Dyn = fun x: Float -> 1 in d(1.5)", 1),
        ] {
            assert_eq!(run_source(src).unwrap().as_int(), want, "{src}");
        }
        let s = "let is_str = 0 in let d: Dyn = fun s: Str -> s in d(\"a\")";
        assert_eq!(run_source(s).unwrap(), Outcome::Str("a".to_string()));
        for src in [
            "let fail = fun x -> 0 in let f = fun a: Int -> a + 1 in let d: Dyn = f in d(true)",
            "let is_int = fun x -> true in let x: Dyn = true in let y: Int = x in y",
            "let fail = 0 in let x: Dyn = true in let y: Int = x in y",
        ] {
            let err = run_source(src).unwrap_err();
            assert!(err.contains("type error: expected Int, found Bool"), "{src}: {err}");
        }
    }

    #[test]
    fn synthesized_cast_binders_are_not_captured_by_user_variables() {
        for (src, want) in [
            ("let __ca = 10 in let d: Dyn = fun a: Int -> a + __ca in d(1)", 11),
            ("let __ca = 10 in len(map(fun a: Int -> a + __ca)([1]))", 1),
            ("let __ca = 10 in map(fun a: Int -> a + __ca)([1])", -1),
            ("let __ca2 = 100 in let d: Dyn = fun a: Int -> fun b: Int -> a + b + __ca2 in d(1)(2)", 103),
            ("let d: Dyn = fun __ca2: Int -> fun b: Int -> __ca2 + b in d(1)(2)", 3),
            ("let __cf = 5 in let f = fun a: Int -> a + __cf in let g = fun x -> x in let d: Dyn = f in d(1)", 6),
        ] {
            if want == -1 {
                assert_eq!(run_source(src).unwrap(), Outcome::List(vec![Outcome::Int(11)]), "{src}");
            } else {
                assert_eq!(run_source(src).unwrap().as_int(), want, "{src}");
            }
        }
    }

    #[test]
    fn a_monomorphic_vec_n_witness_checks_list_shape_only_in_the_closure_form_known_limitation() {
        // `n` is bound by an earlier Dyn-to-Vec(n) check; the wrapper gets a
        // per-call witness, so DOWN checks only that the argument is a list.
        // The direct call f([1]) is rejected.
        let src = "let x: Dyn = [1, 2] in let y: Vec(n) = x in let f = fun w: Vec(n) -> len(w) in let d: Dyn = f in d([1])";
        assert_eq!(run_source(src).unwrap().as_int(), 1);
        let direct = "let x: Dyn = [1, 2] in let y: Vec(n) = x in let f = fun w: Vec(n) -> len(w) in f([1])";
        assert!(run_source(direct).is_err());
    }

    #[test]
    fn refinement_runtime_check_is_not_captured_by_a_user_fail() {
        for src in [
            "let fail = fun x -> 0 in handle (let n: Int where 0 < n = perform choose(0) in n + 1) with handler choose(p, resume) -> resume(-5)",
            "let g = fun fail: Int -> handle (let n: Int where 0 < n = perform choose(0) in n + 1) with handler choose(p, resume) -> resume(-5) in g(1)",
            "let fail = fun x -> 0 in let f = fun n: Int where 0 < n -> n * 2 in f(-3)",
        ] {
            let err = run_source(src).unwrap_err();
            assert!(err.contains("refinement violated"), "{src}: {err}");
        }
    }

    #[test]
    fn synthesized_check_temporaries_are_not_captured_by_user_variables() {
        for (src, want) in [
            ("let __check_tmp = 10 in let d: Dyn = 3 in let y: Int = d in y + __check_tmp", 13),
            ("let __check_tmp = 10 in let d: Dyn = 3 in let y: Int | Bool = d in __check_tmp", 10),
            ("let __check_tmp = 10 in let d: Dyn = [1, 2] in let y: (Int, Int) = d in len(y) + __check_tmp", 12),
            ("let __contract_fn = 10 in let d: Dyn = fun x: Int -> x in d(1) + __contract_fn", 11),
            ("let __contract_arg = 10 in let d: Dyn = fun x: Int -> x in d(1) + __contract_arg", 11),
            ("let __contract_arg = 10 in let __contract_fn = 20 in let f = fun a: Vec(n) -> 7 in let d: Dyn = f in d([1]) + __contract_arg + __contract_fn", 37),
            ("let d: Dyn = fun __contract_arg: Int -> fun __contract_fn: Int -> __contract_arg + __contract_fn in d(1)(2)", 3),
            ("let d: Dyn = fun __check_tmp: Int -> __check_tmp in d(4)", 4),
        ] {
            assert_eq!(run_source(src).unwrap().as_int(), want, "{src}");
        }
    }

    // --- Native Expr::Check (spec 2026-10-08-cast-cost-design, phase 1) ---

    fn elaborated_tree(src: &str) -> (expr::Arena, expr::ExprRef) {
        let (mut arena, spans, root, named) = parser::parse_with_named_types(src).unwrap();
        let out = typecheck::check_with_named_types(&mut arena, root, &spans, named).unwrap();
        (arena, out)
    }

    // An Assert Check with exactly this test is somewhere in the tree.
    fn has_assert_check(src: &str, test: expr::Test) -> bool {
        let (arena, root) = elaborated_tree(src);
        tree_any(&arena, root, &|n| matches!(n, expr::Expr::Check(_, s) if s.test == test && s.mode == expr::CheckMode::Assert))
    }

    // A synthesized call to one of the prelude predicates a check used to
    // desugar into (`#is_int`, `#is_list`, ...).
    fn calls_a_check_predicate(src: &str) -> bool {
        let (arena, root) = elaborated_tree(src);
        tree_any(&arena, root, &|n| matches!(n, expr::Expr::Var(v) if v.starts_with("#is_")))
    }

    type ShapeCase = (&'static str, Vec<&'static str>, Vec<(&'static str, &'static str)>, Option<expr::Test>);

    // (annotation, accepted Dyn values, rejected Dyn values with the expected
    // message fragment, the native test of its Assert Check -- None when the
    // crossing stays desugared). Every shape the old builders produced.
    fn native_check_shapes() -> Vec<ShapeCase> {
        use expr::Test;
        let x = || vec!["x".to_string()];
        vec![
            ("Int", vec!["3"], vec![("true", "type error: expected Int, found Bool")], Some(Test::Int)),
            ("Float", vec!["1.5"], vec![("\"a\"", "type error: expected Float, found Str")], Some(Test::Float)),
            ("Bool", vec!["true"], vec![("3", "type error: expected Bool, found Int")], Some(Test::Bool)),
            ("Str", vec!["\"a\""], vec![("3", "type error: expected Str, found Int")], Some(Test::Str)),
            ("[Int]", vec!["[1]", "[]"], vec![("3", "type error: expected [Int], found Int")], Some(Test::List)),
            ("(Int -> Int)", vec!["fun x: Int -> x"], vec![("3", "type error: expected (Dyn -> Dyn), found Int")], Some(Test::Fun)),
            (
                "(Int, Int)",
                vec!["[1, 2]"],
                vec![("[1, 2, 3]", "type error: expected (Int, Int), found List"), ("3", "type error: expected (Int, Int), found Int")],
                Some(Test::Tuple(2)),
            ),
            (
                "{x: Int}",
                vec!["{x: 1}", "{x: 1, y: 2}"],
                vec![("{y: 2}", "type error: expected {x: Int}, found Record"), ("5", "type error: expected {x: Int}, found Int")],
                Some(Test::Record(x())),
            ),
            ("Int | Bool", vec!["3", "true"], vec![("\"a\"", "type error: expected Int | Bool, found Str")], Some(Test::Or(vec![Test::Int, Test::Bool]))),
            (
                "Int | (Int, Int) | {x: Int}",
                vec!["3", "[1, 2]", "{x: 1}"],
                vec![("[1]", "type error: expected Int | (Int, Int) | {x: Int}, found List"), ("{y: 1}", "found Record")],
                Some(Test::Or(vec![Test::Int, Test::Tuple(2), Test::Record(x())])),
            ),
            ("Vec(2)", vec!["[1, 2]"], vec![("[1]", "expected [Dyn](2), found List"), ("3", "found Int")], None),
            ("Int | Vec(2)", vec!["3", "[1, 2]"], vec![("[1]", "found List"), ("true", "found Bool")], None),
            ("Int | (Int -> Int)", vec!["3", "fun x: Int -> x"], vec![("\"a\"", "type error: expected Int | (Int -> Int), found Str")], None),
        ]
    }

    // A bare Fun crossing is a per-call contract: it is only exercised by calling.
    fn crossing(ty: &str, v: &str) -> String {
        let used = if ty == "(Int -> Int)" { "y(1)" } else { "1" };
        format!("let d: Dyn = {v} in let y: {ty} = d in {used}")
    }

    #[test]
    fn every_dyn_boundary_shape_accepts_and_rejects_as_before() {
        for (ty, accepted, rejected, _) in native_check_shapes() {
            for v in accepted {
                let src = crossing(ty, v);
                assert_eq!(run_source(&src).unwrap().as_int(), 1, "{src}");
            }
            for (v, want) in rejected {
                let src = crossing(ty, v);
                let err = run_source(&src).expect_err(&src);
                assert!(err.contains(want), "{src}: {err}");
            }
        }
    }

    #[test]
    fn every_shallow_dyn_boundary_shape_is_one_native_check() {
        for (ty, _, _, native) in native_check_shapes() {
            let src = crossing(ty, "3");
            if let Some(test) = native {
                assert!(has_assert_check(&src, test), "{src}: no native Check");
                let (arena, root) = elaborated_tree(&src);
                assert!(!has_let_named(&arena, root, "__check_tmp#"), "{src}: still desugared");
            }
            assert!(!calls_a_check_predicate(&src), "{src}: still calls a prelude predicate");
        }
    }

    #[test]
    fn dyn_and_unannotated_targets_emit_no_check() {
        for src in ["let d: Dyn = 3 in let y: Dyn = d in y", "let f = fun x -> x in f(3)"] {
            let (arena, root) = elaborated_tree(src);
            assert!(!tree_any(&arena, root, &|n| matches!(n, expr::Expr::Check(..))), "{src}");
        }
    }

    #[test]
    fn a_numeric_operand_of_unknown_type_is_one_native_int_or_float_check() {
        let src = "let f = fun x -> x + 1 in f(2)";
        assert!(has_assert_check(src, expr::Test::Or(vec![expr::Test::Int, expr::Test::Float])));
        let (arena, root) = elaborated_tree(src);
        assert!(!has_let_named(&arena, root, "__check_tmp#"));
        assert_eq!(run_source(src).unwrap().as_int(), 3);
        let err = run_source("let f = fun x -> x + 1 in f(true)").unwrap_err();
        assert!(err.contains("type error: expected Int | Float, found Bool"), "{err}");
    }

    #[test]
    fn an_annotated_callback_check_is_native_and_allocation_free() {
        let src = "map(fun x: Int -> x + 1)([1, 2])";
        assert!(has_assert_check(src, expr::Test::Int));
        let (arena, root) = elaborated_tree(src);
        assert!(!has_let_named(&arena, root, "__check_tmp#"));
        assert!(!calls_a_check_predicate(src));
        assert_clean_rejection("map(fun x: Int -> x + 1)([true])", BAD_INT);
    }

    #[test]
    fn a_closure_form_cast_check_is_native() {
        let src = "let f = fun x: Int -> x + 1 in let d: Dyn = f in d(1)";
        assert!(has_assert_check(src, expr::Test::Int));
        assert!(!calls_a_check_predicate(src));
        assert_clean_rejection("let f = fun x: Int -> x + 1 in let d: Dyn = f in d(true)", BAD_INT);
    }

    #[test]
    fn a_vec_n_index_check_stays_desugared_but_tests_natively() {
        let src = "let d: Dyn = [1, 2] in let y: Vec(2) = d in 1";
        let (arena, root) = elaborated_tree(src);
        assert!(has_let_named(&arena, root, "__check_tmp#"));
        assert!(tree_any(&arena, root, &|n| matches!(n, expr::Expr::Check(_, s) if s.test == expr::Test::List && s.mode == expr::CheckMode::Probe)));
        assert!(!calls_a_check_predicate(src));
    }

    #[test]
    fn a_native_check_inside_a_resumed_continuation_runs_on_every_resume() {
        let ok = "handle (let y: Int = perform choose(0) in y + 100) with handler choose(p, resume) -> resume(1) + resume(2)";
        assert_eq!(run_source(ok).unwrap().as_int(), 203);
        let bad = "handle (let y: Int = perform choose(0) in y + 100) with handler choose(p, resume) -> resume(1) + resume(true)";
        assert_clean_rejection(bad, BAD_INT);
    }

    #[test]
    fn a_failing_native_check_blames_the_cast_site() {
        // The Dyn-to-Int crossing is the annotated binding's value `d`
        // (line 2, column 14), not the later expression that runs last.
        let err = run_source("let d: Dyn = true in\nlet y: Int = d in\ny + 1").unwrap_err();
        assert!(err.starts_with("line 2, column 14: type error: expected Int, found Bool"), "{err}");
        // A numeric operand blames the operand.
        let err = run_source("let f = fun x -> 1 + x in\nf(true)").unwrap_err();
        assert!(err.starts_with("line 1, column 22: type error: expected Int | Float, found Bool"), "{err}");
    }

    // --- Join-to-Dyn arm casts (spec 2026-10-08-join-to-dyn-casts) ---

    fn elaborated_has_cast(src: &str) -> bool {
        let (mut arena, spans, root, named) = parser::parse_with_named_types(src).unwrap();
        let out = typecheck::check_with_named_types(&mut arena, root, &spans, named).unwrap();
        has_let_named(&arena, out, "__ca#")
    }

    fn assert_clean_rejection(src: &str, want: &str) {
        let err = run_source(src).expect_err(src);
        assert!(err.contains(want), "{src}: {err}");
        assert!(!err.contains("expected a number"), "{src}: {err}");
    }

    const BAD_INT: &str = "type error: expected Int, found Bool";

    #[test]
    fn a_list_of_disagreeing_typed_functions_is_cast_to_dyn() {
        let f = "fun x: Int -> x + 1";
        let g = "fun s: Str -> s";
        assert_clean_rejection(&format!("match [{f}, {g}] | [p, q] -> p(true) | _ -> 0"), BAD_INT);
        assert_clean_rejection(&format!("match [{f}, {g}, fun b: Bool -> 1] | [p, q, r] -> r(1) | _ -> 0"), "type error: expected Bool, found Int");
        // A third element that unified with the already-Dyn join is cast too.
        assert_clean_rejection(&format!("match [{f}, {g}, fun y: Int -> y] | [p, q, r] -> r(true) | _ -> 0"), BAD_INT);
        // Returned functions are cast in turn.
        let curried = "[fun x: Int -> fun y: Int -> y, fun s: Str -> s]";
        assert_clean_rejection(&format!("match {curried} | [p, q] -> p(1)(true) | _ -> 0"), BAD_INT);
        assert_eq!(run_source(&format!("match {curried} | [p, q] -> p(1)(2) | _ -> 0")).unwrap().as_int(), 2);
        // Correct calls are unchanged.
        assert_eq!(run_source(&format!("match [{f}, {g}] | [p, q] -> p(1) | _ -> 0")).unwrap().as_int(), 2);
        assert_eq!(run_source(&format!("match [{f}, {g}] | [p, q] -> q(\"a\") | _ -> 0")).unwrap(), Outcome::Str("a".to_string()));
    }

    #[test]
    fn disagreeing_typed_functions_joined_by_if_or_match_are_cast_to_dyn() {
        let f = "(fun x: Int -> x + 1)";
        let g = "(fun s: Str -> s)";
        for src in [
            format!("let g = if 1 < 2 then {f} else {g} in g(true)"),
            format!("let g = if 2 < 1 then {f} else {g} in g(true)"),
            format!("let g = match 1 | 1 -> {f} | _ -> {g} in g(true)"),
            format!("let g = match 2 | 1 -> {f} | _ -> {g} in g(true)"),
        ] {
            let want = if src.contains("2 < 1") || src.contains("match 2") { "type error: expected Str, found Bool" } else { BAD_INT };
            assert_clean_rejection(&src, want);
        }
        // Third arm unified with the already-Dyn join of the first two.
        assert_clean_rejection(&format!("let g = match 3 | 1 -> {f} | 2 -> {g} | _ -> (fun b: Bool -> 1) in g(1)"), "type error: expected Bool, found Int");
        assert_eq!(run_source(&format!("let g = if 1 < 2 then {f} else {g} in g(1)")).unwrap().as_int(), 2);
        assert_eq!(run_source(&format!("let g = if 2 < 1 then {f} else {g} in g(\"a\")")).unwrap(), Outcome::Str("a".to_string()));
        assert_eq!(run_source(&format!("let g = match 2 | 1 -> {f} | _ -> {g} in g(\"a\")")).unwrap(), Outcome::Str("a".to_string()));
    }

    #[test]
    fn a_partly_dyn_sibling_in_a_join_is_cast_to_the_join_type() {
        let a = "let a: (Dyn -> Int) = fun x -> 1 in";
        let y = "(fun y: Int -> y)";
        for src in [
            format!("{a} let g = if 2 < 1 then a else {y} in g(true)"),
            format!("{a} let g = match 2 | 1 -> a | _ -> {y} in g(true)"),
            format!("{a} match [a, {y}] | [p, q] -> q(true) | _ -> 0"),
            // A Dyn-typed sibling makes the join Dyn.
            format!("let a: Dyn = 5 in let g = if 2 < 1 then a else {y} in g(true)"),
            format!("let a: Dyn = 5 in match [a, {y}] | [p, q] -> q(true) | _ -> 0"),
        ] {
            assert_clean_rejection(&src, BAD_INT);
        }
        assert_eq!(run_source(&format!("{a} let g = if 2 < 1 then a else {y} in g(3)")).unwrap().as_int(), 3);
        // Arm order decides the join (pinned): with the precise arm first the
        // join is `Int -> Int` and the bad call stays a STATIC error.
        let err = run_source(&format!("{a} let g = if 2 < 1 then {y} else a in g(true)")).expect_err("static");
        assert!(err.contains("type mismatch"), "{err}");
        let err = run_source(&format!("{a} match [{y}, a] | [p, q] -> p(true) | _ -> 0")).expect_err("static");
        assert!(err.contains("type mismatch"), "{err}");
    }

    #[test]
    fn joins_that_need_no_cast_emit_no_cast() {
        for src in [
            "if 1 < 2 then (fun x: Int -> x) else (fun y: Int -> y)",
            "[fun x -> x, fun y -> y]",
            "if 1 < 2 then (fun x -> x) else (fun y: Int -> y)",
            "[fun x: Int -> x]",
            "if 1 < 2 then 1 else 2",
            "[1, 2, 3]",
            "match 1 | 1 -> (fun x: Int -> x) | _ -> (fun y: Int -> y)",
            "if 1 < 2 then 1 else \"a\"",
            "[1, \"a\"]",
            "if 1 < 2 then (fun x -> x) else (fun s -> s)",
        ] {
            assert!(!elaborated_has_cast(src), "{src}");
        }
        for src in [
            "[fun x: Int -> x, fun s: Str -> s]",
            "if 1 < 2 then (fun x: Int -> x) else (fun s: Str -> s)",
            "match 1 | 1 -> (fun x: Int -> x) | _ -> (fun s: Str -> s)",
        ] {
            assert!(elaborated_has_cast(src), "{src}");
        }
    }

    #[test]
    fn unchanged_join_behaviour_is_preserved() {
        for (src, want) in [
            ("let g = if 1 < 2 then (fun x: Int -> x + 1) else (fun y: Int -> y) in g(1)", 2),
            ("match [fun x: Int -> x + 1] | [p] -> p(1) | _ -> 0", 2),
            ("let g = if 1 < 2 then (fun x -> x) else (fun y: Int -> y) in g(5)", 5),
            ("let d: Dyn = if 1 < 2 then (fun x: Int -> x + 1) else (fun s: Str -> s) in d(1)", 2),
        ] {
            assert_eq!(run_source(src).unwrap().as_int(), want, "{src}");
        }
        // Same-typed arms and a single-element list keep their static errors.
        for src in [
            "let g = if 1 < 2 then (fun x: Int -> x) else (fun y: Int -> y) in g(true)",
            "match [fun x: Int -> x] | [p] -> p(true) | _ -> 0",
        ] {
            let err = run_source(src).expect_err(src);
            assert!(err.contains("type mismatch"), "{src}: {err}");
        }
        // Check mode against Dyn already coerces each arm.
        assert_clean_rejection("let g: Dyn = if 1 < 2 then (fun x: Int -> x) else (fun s: Str -> s) in g(true)", BAD_INT);
    }

    #[test]
    fn a_function_list_checked_against_a_dyn_list_is_unchecked_known_limitation() {
        // Container casts are parked (fun-to-dyn spec 7): the list type
        // `[Int -> Int]` coerces to `[Dyn]` with no wrapper, so the bad call
        // is not rejected with a clean type error (pinned; flips with the fix).
        for src in [
            "let l: [Dyn] = [fun x: Int -> x + 1] in match l | [p] -> p(true) | _ -> 0",
            "let l: [Dyn] = [fun x: Int -> x + 1, fun s: Str -> s] in match l | [p, q] -> p(true) | _ -> 0",
        ] {
            let r = std::panic::catch_unwind(|| run_source(src));
            let clean = matches!(&r, Ok(Err(e)) if e.contains(BAD_INT));
            assert!(!clean, "{src}: now rejected cleanly; remove the known_limitation pin");
        }
    }

    #[test]
    fn a_vec_n_function_in_a_join_compares_lengths_like_the_closure_form() {
        // Probed empirically (spec 2026-10-08-join-to-dyn-casts, open item 4):
        // inside `fun v: Vec(n)` the outer witness `n` is visible at the join
        // site, so a wrong-length call through the join is rejected exactly as
        // in the closure form (`let k = .. in let d: Dyn = k in d`), and the
        // right length is not rejected spuriously.
        let wrap = |body: &str, call: &str| format!("let f = fun v: Vec(n) -> {body} in {call}");
        let closure = "(let k = fun a: Vec(n) -> len(a) in let d: Dyn = k in d)";
        let if_join = "(if 1 < 2 then (fun a: Vec(n) -> len(a)) else (fun s: Str -> s))";
        let list_join = "[fun a: Vec(n) -> len(a), fun s: Str -> s]";
        for (body, call_ok, call_bad) in [
            (closure, "f([1,2])([3,4])", "f([1,2])([3])"),
            (if_join, "f([1,2])([3,4])", "f([1,2])([3])"),
            (list_join, "match f([1,2]) | [p, q] -> p([3,4]) | _ -> 0", "match f([1,2]) | [p, q] -> p([3]) | _ -> 0"),
        ] {
            assert_eq!(run_source(&wrap(body, call_ok)).unwrap().as_int(), 2, "{body}");
            assert_clean_rejection(&wrap(body, call_bad), "type error: expected [Dyn](n");
        }
        // A concrete length and a Vec(n) function value reaching the join.
        let three = "let g = if 1 < 2 then (fun a: Vec(3) -> len(a)) else (fun s: Str -> s) in";
        assert_eq!(run_source(&format!("{three} g([1,2,3])")).unwrap().as_int(), 3);
        assert_clean_rejection(&format!("{three} g([1])"), "type error: expected [Dyn](3)");
        let h = "let h = fun a: Vec(n) -> fun b: Vec(n) -> len(b) in let g = if 1 < 2 then h else (fun s: Str -> s) in";
        assert_eq!(run_source(&format!("{h} g([1,2])([3,4])")).unwrap().as_int(), 2);
        assert_clean_rejection(&format!("{h} g([1])([3,4])"), "type error: expected [Dyn](n");
    }

    #[test]
    fn an_effect_performed_in_a_join_arm_still_resumes_through_the_cast() {
        let ok = "handle (let g = if 1 < 2 then (fun x: Int -> perform choose(x)) else (fun s: Str -> 0) in g(1)) \
                  with handler choose(p, resume) -> resume(1) + resume(2)";
        assert_eq!(run_source(ok).unwrap().as_int(), 3);
        let bad = "handle (let g = if 1 < 2 then (fun x: Int -> perform choose(x)) else (fun s: Str -> 0) in g(true)) \
                   with handler choose(p, resume) -> resume(1) + resume(2)";
        assert_clean_rejection(bad, BAD_INT);
    }

    // --- Var-typed passthrough to Dyn sinks (spec 2026-10-08-var-passthrough-casts) ---

    #[test]
    fn a_typed_function_passed_through_an_untyped_parameter_to_a_dyn_sink_is_cast() {
        let f = "fun x: Int -> x + 1";
        let two = "fun a: Int -> fun b: Int -> a + b";
        for src in [
            format!("let ap = fun g -> map(g)([true]) in ap({f})"),
            format!("let ap = fun g -> filter(g)([true]) in ap(fun x: Int -> x < 2)"),
            format!("let ap = fun g -> fold(g)(0)([true]) in ap({two})"),
            // Hops: each untyped layer is generalised and instantiated afresh.
            format!("let ap = fun g -> let ap2 = fun h -> map(h)([true]) in ap2(g) in ap({f})"),
            format!("let ap = fun g -> let ap2 = fun h -> let ap3 = fun k -> map(k)([true]) in ap3(h) in ap2(g) in ap({f})"),
            // Directly applied lambda (monomorphic parameter).
            format!("(fun g -> map(g)([true]))({f})"),
            // Perform payload.
            format!("let ap = fun g -> perform op(g) in handle ap({f}) with handler op(p, resume) -> resume(p(true))"),
            // Closure-form argument.
            format!("let f = {f} in let ap = fun g -> map(g)([true]) in ap(f)"),
            // Curried: the UP recursion casts the returned function too.
            "let ap = fun g -> map(g)([true]) in ap(fun a: Int -> fun b: Int -> a + b)".to_string(),
            "let ap = fun g -> let r = perform op(g) in r in \
             handle ap(fun a: Int -> fun b: Int -> a + b) with handler op(p, resume) -> resume(p(1)(true))".to_string(),
        ] {
            assert_clean_rejection(&src, BAD_INT);
        }
    }

    #[test]
    fn correct_calls_through_an_untyped_dyn_sink_parameter_are_unchanged() {
        let list = |xs: &[i64]| Outcome::List(xs.iter().map(|n| Outcome::Int(*n)).collect());
        assert_eq!(run_source("let ap = fun g -> map(g)([1]) in ap(fun x: Int -> x + 1)").unwrap(), list(&[2]));
        assert_eq!(run_source("let ap = fun g -> map(g)([1]) in ap(fun x -> x + 1)").unwrap(), list(&[2]));
        assert_eq!(run_source("let ap = fun g -> fold(g)(0)([1, 2]) in ap(fun a: Int -> fun b: Int -> a + b)").unwrap().as_int(), 3);
        assert_eq!(
            run_source("let ap = fun g -> let ap2 = fun h -> map(h)([1]) in ap2(g) in ap(fun x: Int -> x + 1)").unwrap(),
            list(&[2])
        );
        assert_eq!(run_source("(fun g -> map(g)([1]))(fun x: Int -> x + 1)").unwrap(), list(&[2]));
        assert_eq!(
            run_source("let ap = fun g -> perform op(g) in handle ap(fun a: Int -> a + 1) with handler op(p, resume) -> resume(p(1))")
                .unwrap()
                .as_int(),
            2
        );
        // A non-function argument and independent instantiations (the flag
        // lives on the scheme variable, not on an instance).
        assert_eq!(run_source("let ap = fun g -> map(g)([1]) in ap(fun x: Int -> x + 1)").unwrap(), list(&[2]));
        // The second instantiation is checked on its own (previously this
        // silently accepted a Str function applied to an Int).
        let two_calls = "let ap = fun g -> map(g)([1]) in let a = ap(fun x: Int -> x + 1) in ap(fun s: Str -> s)";
        assert_clean_rejection(two_calls, "type error: expected Str, found Int");
        let ok_then_bad = "let ap = fun g -> map(g)([true]) in let a = ap(fun x -> x) in ap(fun x: Int -> x + 1)";
        assert_clean_rejection(ok_then_bad, BAD_INT);
        let bad_then_ok = "let ap = fun g -> map(g)([true]) in let a = ap(fun x: Int -> x + 1) in a";
        assert_clean_rejection(bad_then_ok, BAD_INT);
        // A curried function called correctly.
        let curried = "let ap = fun g -> fold(g)(0)([1, 2]) in ap(fun a: Int -> fun b: Int -> a + b)";
        assert_eq!(run_source(curried).unwrap().as_int(), 3);
    }

    #[test]
    fn a_non_function_reaching_an_untyped_dyn_sink_parameter_is_unchanged() {
        assert_eq!(run_source("let ap = fun g -> let r = perform op(g) in r in handle ap(5) with handler op(p, resume) -> resume(p)").unwrap().as_int(), 5);
        assert_eq!(run_source("let ap = fun g -> len(g) in ap([1, 2])").unwrap().as_int(), 2);
    }

    #[test]
    fn a_multi_shot_handler_over_a_cast_dyn_sink_still_resumes_twice() {
        let ok = "let ap = fun g -> perform op(g) in \
                  handle ap(fun a: Int -> a + 1) with handler op(p, resume) -> resume(p(1)) + resume(p(2))";
        assert_eq!(run_source(ok).unwrap().as_int(), 5);
        let bad = "let ap = fun g -> perform op(g) in \
                   handle ap(fun a: Int -> a + 1) with handler op(p, resume) -> resume(p(1)) + resume(p(true))";
        assert_clean_rejection(bad, BAD_INT);
    }

    #[test]
    fn unannotated_callbacks_through_an_untyped_dyn_sink_parameter_emit_no_wrapper() {
        for src in [
            "let ap = fun g -> map(g)([1]) in ap(fun x -> x + 1)",
            "let ap = fun g -> map(g)([1]) in ap(fun x -> x)",
            "let ap = fun g -> let ap2 = fun h -> map(h)([1]) in ap2(g) in ap(fun x -> x)",
            "let ap = fun g -> perform op(g) in handle ap(fun x -> x) with handler op(p, resume) -> resume(p)",
            "(fun g -> map(g)([1]))(fun x -> x)",
            // Sink flagged, but nothing typed reaches it.
            "let ap = fun g -> map(g)([1]) in ap(map)",
            // A polymorphic use with no sink at all.
            "let apply = fun g -> g(1) in apply(fun x: Int -> x)",
        ] {
            assert!(!elaborated_has_wrapper(src) && !elaborated_has_cast(src), "{src}");
        }
        // The typed counterparts do carry the wrapper.
        for src in [
            "let ap = fun g -> map(g)([1]) in ap(fun x: Int -> x + 1)",
            "(fun g -> map(g)([1]))(fun x: Int -> x + 1)",
            "let ap = fun g -> perform op(g) in handle ap(fun x: Int -> x) with handler op(p, resume) -> resume(p)",
        ] {
            // Literal lambdas are rebuilt in place (`__ca`), no `__cf` closure.
            assert!(elaborated_has_cast(src), "{src}");
        }
    }

    #[test]
    fn marking_a_dyn_sink_variable_loses_no_polymorphic_precision() {
        // Option (c) (bind the Var to Dyn) would turn these into Dyn results.
        let src = "let f = fun xs -> let n = len(xs) in xs in let ys: [Int] = f([1, 2]) in ys";
        assert_eq!(run_source(src).unwrap(), Outcome::List(vec![Outcome::Int(1), Outcome::Int(2)]));
        assert!(!elaborated_has_wrapper(src) && !elaborated_has_cast(src));
        // The argument comes back with its own type: `g` stays a function.
        let id2 = "let id2 = fun g -> let r = map(g)([1]) in g in (id2(fun x: Int -> x + 1))(5)";
        assert_eq!(run_source(id2).unwrap().as_int(), 6);
        let id2_bad = "let id2 = fun g -> let r = map(g)([1]) in g in (id2(fun x: Int -> x + 1))(true)";
        let err = run_source(id2_bad).expect_err("static");
        assert!(err.contains("type mismatch"), "{err}");
    }

    #[test]
    fn a_var_bound_to_a_function_before_the_sink_keeps_its_behaviour() {
        // `g(1)` first fixes g : Int -> Int, so the ordinary cast fires.
        let call_first = "let ap = fun g -> let r = g(1) in map(g)([true]) in ap(fun x: Int -> x + 1)";
        assert_clean_rejection(call_first, BAD_INT);
        // Variables unified with Dyn or passed to typed callees already worked.
        for src in [
            "let ap = fun g -> let d: Dyn = g in d(true) in ap(fun x: Int -> x + 1)",
            "let call = fun h: Dyn -> h(true) in let ap = fun g -> call(g) in ap(fun x: Int -> x + 1)",
            "let call = fun h: Dyn -> h(true) in let ap = fun g -> let ap2 = fun k -> call(k) in ap2(g) in ap(fun x: Int -> x + 1)",
            "let ap = fun g -> let d: Dyn = g in map(d)([true]) in ap(fun x: Int -> x + 1)",
        ] {
            assert_clean_rejection(src, BAD_INT);
        }
        let ret = "let ap = fun g -> g in let h: Dyn = ap(fun a: Int -> a + 1) in h(true)";
        assert_clean_rejection(ret, BAD_INT);
    }

    #[test]
    fn sink_first_then_call_is_still_unchecked_known_limitation() {
        // The Var is bound to a Fun only after the sink was elaborated; the
        // flag dies with the binding (spec 2026-10-08-var-passthrough-casts
        // section 7, item 1). Pinned: flips if a deferred sink patch lands.
        let src = "let ap = fun g -> let r = map(g)([true]) in g(1) in ap(fun x: Int -> x + 1)";
        let r = std::panic::catch_unwind(|| run_source(src));
        let clean = matches!(&r, Ok(Err(e)) if e.contains(BAD_INT));
        assert!(!clean, "now rejected cleanly; remove the known_limitation pin");
    }

    #[test]
    fn a_vec_n_function_through_an_untyped_dyn_sink_parameter_is_checked_for_list_shape() {
        // `map(g)` applies g to each ELEMENT. The wrapper's DOWN is `is_list`
        // only (the witness is the wrapper's own, no length check), so a list
        // element still returns [0]...
        let f = "let f = fun v: Vec(n) -> if true then 0 else (let r: Dyn = [1] in let q: Vec(n) = r in 0) in";
        let ok = format!("{f} let ap = fun g -> map(g)([[5]]) in ap(f)");
        assert_eq!(run_source(&ok).unwrap(), Outcome::List(vec![Outcome::Int(0)]));
        // ...and a non-list element, which the old pin of this test let in
        // unchecked (`map(g)([5])` returned [0]), is now rejected cleanly.
        for el in ["5", "true"] {
            let bad = format!("{f} let ap = fun g -> map(g)([{el}]) in ap(f)");
            assert_clean_rejection(&bad, "type error: expected [Dyn], found");
        }
    }

    #[test]
    fn a_dyn_returning_function_parameter_check_is_the_identity_not_an_ice() {
        // DOWN on a `Dyn -> Dyn` parameter builds a return check against Dyn,
        // which is no check at all (was `unreachable!` in build_boundary_check).
        for src in [
            "let ap = fun g -> map(g)([fun x -> x + 1]) in ap(fun f: (Dyn -> Dyn) -> f(1))",
            "let ap = fun g -> map(g)([fun x -> x + 1]) in ap(fun f: (Dyn ->{e} Dyn) -> f(1))",
            "map(fun f: (Dyn ->{e} Dyn) -> f(1))([fun x -> x + 1])",
            "let ap: Dyn = fun f: (Dyn -> Dyn) -> f(1) in ap(fun x -> x + 1)",
        ] {
            let r = std::panic::catch_unwind(|| run_source(src));
            assert!(matches!(&r, Ok(Ok(_))), "{src}: {r:?}");
        }
        let src = "let ap = fun g -> map(g)([fun x -> x + 1]) in ap(fun f: (Dyn -> Dyn) -> f(1))";
        assert_eq!(run_source(src).unwrap(), Outcome::List(vec![Outcome::Int(2)]));
        // Two typed functions of different types joined by `if`.
        let join = "let g = fun a: Int -> a in let h = fun s: Str -> s in (if true then g else h)(1)";
        let r = std::panic::catch_unwind(|| run_source(join));
        assert!(matches!(&r, Ok(Ok(_))), "{join}: {r:?}");
    }

    #[test]
    fn a_failed_trial_unify_does_not_leak_a_dyn_sink_flag() {
        // `h` never reaches a Dyn sink: the `(g, 1)` / `(h, true)` join fails
        // its trial unify after binding g's and h's variables together, and
        // that alias must not carry g's flag over to h. h stays unwrapped.
        let h = "fun f: (Dyn -> Dyn) -> f(1)";
        for join in ["if true then (g, 1) else (h, true)", "match [(g, 1), (h, true)] | [p, q] -> p | _ -> (g, 1)"] {
            let src = format!(
                "let k = fun g -> fun h -> let r = map(g)([1]) in let j = ({join}) in h in \
                 (k(fun x -> x)({h}))(fun x -> x)"
            );
            assert_eq!(run_source(&src).unwrap().as_int(), 1, "{src}");
            assert!(!elaborated_has_cast(&src), "{src}");
        }
        let list = "let k = fun g -> fun h -> let r = map(g)([1]) in let j = [(g, 1), (h, true)] in h in \
                    (k(fun x -> x)(fun f: (Dyn -> Dyn) -> f(1)))(fun x -> x)";
        assert_eq!(run_source(list).unwrap().as_int(), 1, "{list}");
        assert!(!elaborated_has_cast(list), "{list}");
    }

    #[test]
    fn a_perform_as_a_join_arm_is_handled_and_the_resumed_function_is_still_checked() {
        let prog = |call: &str| format!(
            "handle (let g = if 1 < 2 then (perform pick(0)) else (fun s: Str -> s) in g({call})) \
             with handler pick(p, resume) -> resume(fun x: Int -> x + 1)"
        );
        assert_eq!(run_source(&prog("1")).unwrap().as_int(), 2);
        assert_clean_rejection(&prog("true"), BAD_INT);
    }

    #[test]
    fn a_let_rec_function_arm_joined_to_a_str_function_is_cast() {
        let prog = |call: &str| format!(
            "let rec f = fun n: Int -> if n < 1 then 0 else f(n - 1) in \
             let g = if 1 < 2 then f else (fun s: Str -> s) in g({call})"
        );
        assert_eq!(run_source(&prog("3")).unwrap().as_int(), 0);
        assert_clean_rejection(&prog("true"), BAD_INT);
    }
}
