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
            Expr::Let(_, _, val, body) => contains_check(arena, *val) || contains_check(arena, *body),
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
            Expr::DataGroup(_, body) => contains_check(arena, *body),
            Expr::FieldAccess(target, _) => contains_check(arena, *target),
            Expr::NamedCall(callee, args) => {
                contains_check(arena, *callee) || args.iter().any(|(_, v)| contains_check(arena, *v))
            }
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

    #[test]
    fn mutual_recursion_composes_with_adt_and_match() {
        let src = r#"
            data List = Nil | Cons(Int, List) in
            let rec sum = fun l -> match l with | Nil -> 0 | Cons(h, t) -> h + count(t)
            and count = fun l -> match l with | Nil -> 0 | Cons(h, t) -> 1 + sum(t)
            in sum(Cons(1)(Cons(2)(Cons(3)(Nil))))
        "#;
        assert_eq!(run_untyped(src).as_int(), 5);
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

    // --- ADTs (sugar over tagged Lists -- see parser::build_ctor_value) ---

    #[test]
    fn adt_nullary_and_unary_constructors_round_trip_through_match() {
        let src = "data Option = None | Some(Int) in match Some(5) with | None -> 0 | Some(x) -> x";
        assert_eq!(run_untyped(src).as_int(), 5);
    }

    #[test]
    fn adt_nullary_constructor_matches_its_own_arm() {
        let src = "data Option = None | Some(Int) in match None with | None -> 0 | Some(x) -> x";
        assert_eq!(run_untyped(src).as_int(), 0);
    }

    #[test]
    fn adt_self_referential_field_supports_recursive_structures() {
        // `List`'s own name used as Cons's second field type -- resolves to
        // Dyn (see parse_type's uppercase-Ident fallback), which is what
        // makes a recursive type nameable at all with no name-resolution
        // pass. Constructors are curried like every other multi-arg
        // callable in renno (fold, map): Cons(1)(rest), not Cons(1, rest).
        let src = r#"
            data List = Nil | Cons(Int, List) in
            let rec sum = fun l -> match l with | Nil -> 0 | Cons(h, t) -> h + sum(t) in
            sum(Cons(1)(Cons(2)(Cons(3)(Nil))))
        "#;
        assert_eq!(run_untyped(src).as_int(), 6);
    }

    #[test]
    fn adt_constructor_argument_is_type_checked_statically() {
        let src = r#"data Option = None | Some(Int) in Some("x")"#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int, found Str"), "unexpected message: {}", err.0);
    }

    #[test]
    fn adt_lowercase_constructor_name_rejected_at_parse_time() {
        let err = parser::parse("data Option = none | Some(Int) in None").unwrap_err();
        assert!(err.contains("uppercase"), "unexpected message: {err}");
    }

    // --- nominal ADT typing ---

    #[test]
    fn distinct_data_types_with_identical_shape_are_not_interchangeable() {
        // Celsius and Fahrenheit both wrap a single Int -- structurally
        // identical, but Type::Data is nominal (name-compared), so passing
        // one where the other is annotated is a static error.
        let src = "data Celsius = MkC(Int) in\ndata Fahrenheit = MkF(Int) in\nlet f = fun x: Celsius -> x in\nf(MkF(100))";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Celsius, found Fahrenheit"), "unexpected message: {}", err.0);
    }

    #[test]
    fn same_data_type_annotation_accepted() {
        let src = "data Celsius = MkC(Int) in let f = fun x: Celsius -> x in f(MkC(100))";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).to_string(), "[MkC, 100]");
    }

    #[test]
    fn self_referential_field_gets_real_nominal_checking() {
        // Cons's second field is typed List (its own enclosing data type,
        // per parser::ctor_type/Type::Data) -- passing a non-List there is
        // now a static error, not silently accepted as Dyn would allow.
        let src = "data List = Nil | Cons(Int, List) in Cons(1)(5)";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected List, found Int"), "unexpected message: {}", err.0);
    }

    #[test]
    fn self_referential_field_still_accepts_correct_recursive_structures() {
        let src = r#"
            data List = Nil | Cons(Int, List) in
            let rec sum = fun l -> match l with | Nil -> 0 | Cons(h, t) -> h + sum(t) in
            sum(Cons(1)(Cons(2)(Cons(3)(Nil))))
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 6);
    }

    #[test]
    fn dyn_sourced_value_flowing_into_a_data_annotation_gets_a_runtime_check() {
        // Runtime check is necessarily shallow (see value::matches_type's
        // Data arm): confirms "some tagged value", not "specifically this
        // data type" -- a bare Int still fails it, which is the case that
        // matters most (catching an obviously wrong value at the boundary).
        let src = r#"
            data Option = None | Some(Int) in
            let f = fun x: Option -> x in
            handle f(perform choose(0)) with handler choose(p, resume) -> resume(42)
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            machine::run(&arena, elaborated, Env::prelude(), &spans)
        }));
        assert!(result.is_err(), "expected a panic: 42 doesn't match Data(\"Option\")'s shallow shape check");
    }

    // --- named-field access ---

    #[test]
    fn named_field_access_reads_the_right_field() {
        let src = "data Point = Point(x: Int, y: Int) in let p = Point(1)(2) in p.x + p.y";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 3);
    }

    #[test]
    fn named_field_access_on_three_field_record() {
        let src = "data Point = Point(x: Int, y: Int, z: Int) in let p = Point(1)(2)(3) in p.z";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 3);
    }

    #[test]
    fn unknown_field_name_rejected_statically() {
        let src = "data Point = Point(x: Int, y: Int) in let p = Point(1)(2) in p.z";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("no field named `z`"), "unexpected message: {}", err.0);
    }

    #[test]
    fn field_access_on_multi_constructor_type_rejected_statically() {
        let src = "data Option = None | Some(x: Int) in let p = Some(5) in p.x";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("exactly one constructor"), "unexpected message: {}", err.0);
    }

    #[test]
    fn field_access_on_unnamed_constructor_rejected_statically() {
        let src = "data Pair = Pair(Int, Int) in let p = Pair(1)(2) in p.x";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("no named fields"), "unexpected message: {}", err.0);
    }

    #[test]
    fn field_access_on_non_data_type_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("5.x").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected a `data` type"), "unexpected message: {}", err.0);
    }

    #[test]
    #[should_panic(expected = "requires typechecking")]
    fn field_access_on_the_untyped_path_panics_clearly() {
        run_untyped("data Point = Point(x: Int, y: Int) in Point(1)(2).x");
    }

    // --- named-field construction and patterns ---

    #[test]
    fn named_construction_and_named_pattern_round_trip() {
        let src = r#"
            data Point = Point(x: Int, y: Int) in
            match Point { x: 3, y: 4 } with
            | Point { x: a, y: b } -> a * a + b * b
        "#;
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 25);
    }

    #[test]
    fn named_construction_field_order_does_not_matter() {
        let src = "data Point = Point(x: Int, y: Int) in \
                    let p = Point { y: 4, x: 3 } in p.x * p.x + p.y * p.y";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), 25);
    }

    #[test]
    fn named_pattern_field_order_does_not_matter() {
        // Positional construction, named pattern in the OPPOSITE order --
        // confirms reordering happens on both sides independently.
        let src = "data Point = Point(x: Int, y: Int) in \
                    match Point(3)(4) with | Point { y: b, x: a } -> a - b";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let elaborated = typecheck::check(&mut arena, root, &spans).unwrap();
        assert_eq!(machine::run(&arena, elaborated, Env::prelude(), &spans).as_int(), -1);
    }

    #[test]
    fn named_construction_missing_field_rejected_statically() {
        let (mut arena, spans, root) = parser::parse("data Point = Point(x: Int, y: Int) in Point { x: 1 }").unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("missing field `y`"), "unexpected message: {}", err.0);
    }

    #[test]
    fn named_construction_unknown_field_rejected_statically() {
        let src = "data Point = Point(x: Int, y: Int) in Point { x: 1, y: 2, z: 3 }";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("no field named `z`"), "unexpected message: {}", err.0);
    }

    #[test]
    fn named_construction_duplicate_field_rejected_statically() {
        let src = "data Point = Point(x: Int, y: Int) in Point { x: 1, x: 2 }";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("given more than once"), "unexpected message: {}", err.0);
    }

    #[test]
    fn named_construction_on_multi_constructor_type_rejected_statically() {
        let src = "data Option = None | Some(x: Int) in Some { x: 1 }";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("not exactly one"), "unexpected message: {}", err.0);
    }

    #[test]
    fn named_construction_field_type_still_checked() {
        let src = "data Point = Point(x: Int, y: Int) in Point { x: true, y: 2 }";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("expected Int, found Bool"), "unexpected message: {}", err.0);
    }

    #[test]
    fn named_pattern_missing_field_rejected_statically() {
        let src = "data Point = Point(x: Int, y: Int) in match Point(1)(2) with | Point { x: a } -> a";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        let err = typecheck::check(&mut arena, root, &spans).unwrap_err();
        assert!(err.0.contains("missing field `y`"), "unexpected message: {}", err.0);
    }

    #[test]
    #[should_panic(expected = "requires typechecking")]
    fn named_construction_on_the_untyped_path_panics_clearly() {
        run_untyped("data Point = Point(x: Int, y: Int) in Point { x: 1, y: 2 }");
    }

    #[test]
    #[should_panic(expected = "requires typechecking")]
    fn named_pattern_on_the_untyped_path_panics_clearly() {
        run_untyped("data Point = Point(x: Int, y: Int) in match Point(1)(2) with | Point { x: a, y: b } -> a");
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
    fn exhaustive_adt_match_typechecks() {
        let src = "data Option = None | Some(Int) in match Some(5) with | None -> 0 | Some(x) -> x";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
        assert!(typecheck::check(&mut arena, root, &spans).is_ok());
    }

    #[test]
    fn non_exhaustive_adt_match_rejected_statically() {
        let src = "data Option = None | Some(Int) in match Some(5) with | None -> 0";
        let (mut arena, spans, root) = parser::parse(src).unwrap();
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
        // token actually found in its place (line 2, where "y" starts).
        let src = "let x = 1 in\nlet y = 2\ny";
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
    fn match_failed_panic_reports_the_matchs_location() {
        // Exhaustiveness passes statically (None/Some cover every
        // constructor `data Option` declared) -- but `x` actually comes
        // from a handler resuming with a bare Int, not a tagged List, so
        // match_pattern finds no arm at runtime despite that. Confirms a
        // panic from machine.rs's OWN code (not env.rs/value.rs) is
        // located the same way.
        let src = r#"
            data Option = None | Some(Int) in
            handle
              let x = perform choose(0) in
              match x with
              | None -> 0
              | Some(y) -> y
            with deep(handler choose(p, resume) -> resume(42))
        "#;
        let err = run_source(src).unwrap_err();
        assert!(err.contains("match failed: no pattern matched the value"), "unexpected message: {err}");
        assert!(err.contains("line 5"), "unexpected message: {err}");
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
