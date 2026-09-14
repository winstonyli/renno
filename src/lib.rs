pub mod cont;
pub mod env;
pub mod expr;
pub mod lexer;
pub mod machine;
pub mod parser;
pub mod plist;
pub mod typecheck;
pub mod types;
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
    let (mut arena, root) = parser::parse(src)?;
    let elaborated = typecheck::check(&mut arena, root).map_err(|e| e.0)?;
    std::panic::catch_unwind(|| machine::run(&arena, elaborated, Env::prelude()))
        .map_err(|_| "runtime error (see panic message above)".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Raw parse+run, no typecheck -- for tests exercising parser/machine
    // semantics (multi-shot, deep/shallow, arithmetic) independent of the
    // typechecker.
    fn run_untyped(src: &str) -> Value {
        let (arena, root) = parser::parse(src).expect("parse failed");
        machine::run(&arena, root, Env::prelude())
    }

    // Walks the elaborated tree looking for a Check node. Needed because
    // arena-indexed Expr's derived Debug only prints the immediate node
    // (children are plain ExprRef indices now, not Rc<Expr>, so Debug no
    // longer recurses through them the way it used to).
    fn contains_check(arena: &expr::Arena, root: expr::ExprRef) -> bool {
        use expr::Expr;
        match &arena[root] {
            Expr::Check(..) => true,
            Expr::Int(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Var(_) => false,
            Expr::ListLit(items) => items.iter().any(|i| contains_check(arena, *i)),
            Expr::Lambda(_, _, body) => contains_check(arena, *body),
            Expr::App(f, a) => contains_check(arena, *f) || contains_check(arena, *a),
            Expr::Let(_, _, val, body, _) => contains_check(arena, *val) || contains_check(arena, *body),
            Expr::BinOp(_, l, r) => contains_check(arena, *l) || contains_check(arena, *r),
            Expr::If(c, t, e) => contains_check(arena, *c) || contains_check(arena, *t) || contains_check(arena, *e),
            Expr::Perform(_, payload) => contains_check(arena, *payload),
            Expr::Handle { body, handler } => contains_check(arena, *body) || contains_check(arena, *handler),
            Expr::MakeHandler { body, .. } => contains_check(arena, *body),
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
        let (mut arena, root) = parser::parse("true == false").unwrap();
        let elaborated = typecheck::check(&mut arena, root).unwrap();
        assert!(!machine::run(&arena, elaborated, Env::prelude()).as_bool());
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
        let (mut arena, root) = parser::parse(r#"1 ++ "a""#).unwrap();
        let err = typecheck::check(&mut arena, root).unwrap_err();
        assert!(err.0.contains("cannot concat"), "unexpected message: {}", err.0);
    }

    // --- let rec ---

    #[test]
    fn let_rec_self_reference_recurses() {
        // Sums 1..5 by counting UP (no `-` operator exists yet) -- proves
        // `loop` inside its own body resolves to itself, not an unbound
        // Var lookup.
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
        let (mut arena, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude()).as_int(), 15);
    }

    #[test]
    fn let_rec_composes_with_map() {
        // `map`'s callback itself uses `let rec` internally -- confirms
        // RecClosure works fine as a value passed through machine::apply,
        // not just when called directly from the trampoline loop.
        let src = "map(fun n -> let rec loop = fun i -> if i < n then i + loop(i + 1) else 0 in loop(1))([3, 6])";
        assert_eq!(run_untyped(src).to_string(), "[3, 15]");
    }

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
        let (mut arena, root) = parser::parse(&src).unwrap();
        let err = typecheck::check(&mut arena, root).unwrap_err();
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
        let (mut arena, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root).expect("should typecheck (falls back to Dyn)");
        machine::run(&arena, elaborated, Env::prelude());
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
        let (mut arena, root) = parser::parse("let f = fun x: Int -> x + 1 in f(41)").unwrap();
        let elaborated = typecheck::check(&mut arena, root).unwrap();
        assert!(!contains_check(&arena, elaborated));
        let result = machine::run(&arena, elaborated, Env::prelude());
        assert_eq!(result.as_int(), 42);
    }

    #[test]
    fn static_type_error_rejected_before_running() {
        // Both sides concrete and inconsistent -- rejected by the checker,
        // never reaches machine::run at all.
        let (mut arena, root) = parser::parse("(fun x: Int -> x + 1)(true)").unwrap();
        let err = typecheck::check(&mut arena, root).unwrap_err();
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
        let (mut arena, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root).unwrap();
        assert!(contains_check(&arena, elaborated));
        let result = machine::run(&arena, elaborated, Env::prelude());
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
        let (mut arena, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root).unwrap();
        machine::run(&arena, elaborated, Env::prelude());
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
        let (mut arena, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine::run(&arena, elaborated, Env::prelude())
        }));
        assert!(result.is_err(), "expected a panic from the return-type contract check");
    }

    #[test]
    fn handle_with_non_handler_value_rejected_statically() {
        let (mut arena, root) = parser::parse("handle 1 with 5").unwrap();
        let err = typecheck::check(&mut arena, root).unwrap_err();
        assert!(err.0.contains("expected a handler value"), "unexpected message: {}", err.0);
    }

    // --- closed effect-row typing ---

    #[test]
    fn truly_unhandled_effect_rejected_statically() {
        // No `handle` anywhere -- previously this would only fail at
        // runtime, inside machine::run, via perform()'s own panic. Now
        // caught by typecheck::check before anything executes.
        let (mut arena, root) = parser::parse("perform choose(0)").unwrap();
        let err = typecheck::check(&mut arena, root).unwrap_err();
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
        let (mut arena, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root).is_ok());
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
        let (mut arena, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root).expect("both effects are handled, should typecheck");
        machine::run(&arena, elaborated, Env::prelude());
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
        let (mut arena, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root).unwrap_err();
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
        let (mut arena, root) = parser::parse(src).unwrap();
        let elaborated =
            typecheck::check(&mut arena, root).expect("Dyn-sourced call should not be statically rejected");
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine::run(&arena, elaborated, Env::prelude())
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
        let (mut arena, root) = parser::parse(&src).expect("parsing should not overflow the stack");
        let elaborated = typecheck::check(&mut arena, root).expect("typechecking should not overflow the stack");
        let result = machine::run(&arena, elaborated, Env::prelude());
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
