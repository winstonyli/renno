mod cont;
mod env;
mod expr;
mod lexer;
mod machine;
mod parser;
mod plist;
mod typecheck;
mod types;
mod value;

use env::Env;

fn main() {
    let src = r#"
        handle
          let x = perform choose(0) in
          x + 100
        with handler choose(p, resume) -> resume(1) + resume(2)
    "#;
    let expr = parser::parse(src).expect("parse failed");
    // `x` is Dyn (bound from perform's result -- effects stay untyped for
    // now), but flows into `x + 100`, which expects Int: elaboration
    // inserts a runtime Check(Int, x) there. Passes since the handler
    // resumes with real ints.
    let elaborated = typecheck::check(&expr).unwrap_or_else(|e| panic!("type error: {}", e.0));
    let result = machine::run(elaborated, Env::prelude());
    println!("result = {}", result.as_int());
    assert_eq!(result.as_int(), 203);
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let expr = parser::parse(src).expect("parse failed");
        let result = machine::run(expr, Env::prelude());
        assert_eq!(result.as_int(), 203);
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
        let expr = parser::parse(src).expect("parse failed");
        let result = machine::run(expr, Env::prelude());
        assert_eq!(result.as_int(), 2);
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
        let expr = parser::parse(src).expect("parse failed");
        machine::run(expr, Env::prelude());
    }

    #[test]
    fn if_true_takes_then_branch() {
        let expr = parser::parse("if 1 < 2 then 10 else 20").unwrap();
        assert_eq!(machine::run(expr, Env::prelude()).as_int(), 10);
    }

    #[test]
    fn if_false_takes_else_branch() {
        let expr = parser::parse("if 2 < 1 then 10 else 20").unwrap();
        assert_eq!(machine::run(expr, Env::prelude()).as_int(), 20);
    }

    #[test]
    fn eq_and_bool_literals() {
        let expr = parser::parse("if 3 == 3 then true else false").unwrap();
        assert!(machine::run(expr, Env::prelude()).as_bool());
    }

    #[test]
    fn eq_compares_bools_directly() {
        // Regression test: Eq used to force both operands through Int,
        // rejecting this at typecheck time even though it's a valid
        // comparison.
        let expr = parser::parse("true == false").unwrap();
        let elaborated = typecheck::check(&expr).unwrap();
        assert!(!machine::run(elaborated, Env::prelude()).as_bool());
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
        let expr = parser::parse("let f = fun x: Int -> x + 1 in f(41)").unwrap();
        let elaborated = typecheck::check(&expr).unwrap();
        assert!(!format!("{elaborated:?}").contains("Check"));
        let result = machine::run(elaborated, Env::prelude());
        assert_eq!(result.as_int(), 42);
    }

    #[test]
    fn static_type_error_rejected_before_running() {
        // Both sides concrete and inconsistent -- rejected by the checker,
        // never reaches machine::run at all.
        let expr = parser::parse("(fun x: Int -> x + 1)(true)").unwrap();
        let err = typecheck::check(&expr).unwrap_err();
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
        let expr = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&expr).unwrap();
        assert!(format!("{elaborated:?}").contains("Check"));
        let result = machine::run(elaborated, Env::prelude());
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
        let expr = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&expr).unwrap();
        machine::run(elaborated, Env::prelude());
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
        let expr = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&expr).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine::run(elaborated, Env::prelude())
        }));
        assert!(result.is_err(), "expected a panic from the return-type contract check");
    }

    #[test]
    fn handle_with_non_handler_value_rejected_statically() {
        let expr = parser::parse("handle 1 with 5").unwrap();
        let err = typecheck::check(&expr).unwrap_err();
        assert!(err.0.contains("expected a handler value"), "unexpected message: {}", err.0);
    }
}
