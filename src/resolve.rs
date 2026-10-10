use cranelift_entity::EntityRef;

use crate::env::PRELUDE;
use crate::expr::{Arena, Expr, ExprRef, Pattern};

// Where a variable lives at runtime, decided statically. See the spec's
// scope table (the env-frames design spec, §2): `hops` counts runtime frames outward from the innermost
// one, `slot` indexes within that frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VarRef {
    Local { hops: u32, slot: u32 },
    // Index into env::PRELUDE.
    Prelude(u32),
    // In no scope and not in the prelude. Typecheck does not reject unbound
    // names (gradual typing), so this is NOT a resolve-time error: the machine
    // panics `unbound variable: {name}` lazily if it ever evaluates such a
    // Var (same message, same laziness as before the resolver existed; the
    // span is preserved via machine::current_span).
    Unbound,
}

// Side table keyed by the SAME ExprRef as the Arena (the SpanMap idiom):
// only Expr::Var nodes get an entry (an unbound name gets VarRef::Unbound).
// Dense over the range of Var nodes the walk met, `base..base + vars.len()`,
// not over the whole arena prefix: machine::eval_type resolves each annotation
// on its own, and a table sized to the annotation's position would make
// checking quadratic in the program size.
pub struct Resolved {
    base: usize,
    vars: Vec<Option<VarRef>>,
}

impl Resolved {
    pub fn get(&self, expr: ExprRef) -> Option<VarRef> {
        self.vars.get(expr.index().wrapping_sub(self.base)).copied().flatten()
    }
}

// Binder names a pattern introduces, in the order match_pattern must push
// their values: List left-to-right, Cons head then tail, Record fields in
// stored order. Recursion depth is bounded by the PATTERN's own size (source
// text), same argument as machine::match_pattern.
pub fn pattern_vars(p: &Pattern) -> Vec<String> {
    let mut out = Vec::new();
    collect_pattern_vars(p, &mut out);
    out
}

fn collect_pattern_vars(p: &Pattern, out: &mut Vec<String>) {
    match p {
        Pattern::Var(n) => out.push(n.clone()),
        Pattern::Int(_) | Pattern::Bool(_) | Pattern::Str(_) => {}
        Pattern::List(ps) => ps.iter().for_each(|p| collect_pattern_vars(p, out)),
        Pattern::Cons(h, t) => {
            collect_pattern_vars(h, out);
            collect_pattern_vars(t, out);
        }
        Pattern::Record(fields) => fields.iter().for_each(|(_, p)| collect_pattern_vars(p, out)),
    }
}

enum Work {
    Visit(ExprRef),
    // Enter / leave one runtime frame with these binder names.
    Push(Vec<String>),
    Pop,
}

// Iterative on purpose: parser and typecheck flatten Let/Fun chains to keep
// long programs off the native stack, and a recursive walk here would bring
// the overflow back. The work stack is LIFO, so each node pushes its
// children in REVERSE of the order they must be processed in.
// Never fails on a name: an unresolvable one becomes VarRef::Unbound.
pub fn resolve(arena: &Arena, root: ExprRef) -> Resolved {
    let mut found: Vec<(ExprRef, VarRef)> = Vec::new();
    let mut scopes: Vec<Vec<String>> = Vec::new();
    let mut work = vec![Work::Visit(root)];
    while let Some(item) = work.pop() {
        match item {
            Work::Push(names) => scopes.push(names),
            Work::Pop => {
                scopes.pop();
            }
            Work::Visit(id) => visit(arena, id, &scopes, &mut found, &mut work),
        }
    }
    let base = found.iter().map(|(id, _)| id.index()).min().unwrap_or(0);
    let end = found.iter().map(|(id, _)| id.index() + 1).max().unwrap_or(0);
    let mut vars = vec![None; end.saturating_sub(base)];
    for (id, r) in found {
        let slot = &mut vars[id.index() - base];
        match *slot {
            Some(prev) if prev != r => {
                panic!("resolver: {id} reached under conflicting scopes ({prev:?} vs {r:?})")
            }
            _ => *slot = Some(r),
        }
    }
    Resolved { base, vars }
}

// A name spelled `#builtin` (never lexable) is a synthesized reference to a
// prelude builtin: it ignores the lexical scope, so no user binding can capture
// it (typecheck::prelude_var builds these).
fn lookup(scopes: &[Vec<String>], name: &str) -> VarRef {
    if let Some(builtin) = name.strip_prefix('#') {
        return match PRELUDE.iter().position(|(n, _)| *n == builtin) {
            Some(i) => VarRef::Prelude(i as u32),
            None => VarRef::Unbound,
        };
    }
    for (hops, frame) in scopes.iter().rev().enumerate() {
        // Last slot wins: preserves today's shadowing for a duplicate
        // binder in one frame, e.g. the pattern `(x, x)`.
        if let Some(slot) = frame.iter().rposition(|n| n == name) {
            return VarRef::Local { hops: hops as u32, slot: slot as u32 };
        }
    }
    match PRELUDE.iter().position(|(n, _)| *n == name) {
        Some(i) => VarRef::Prelude(i as u32),
        // Not an error here: see VarRef::Unbound.
        None => VarRef::Unbound,
    }
}

// Queue `child` to be visited inside a frame of `names`: processed as
// Push, Visit, Pop (pushed here in reverse).
fn scoped(work: &mut Vec<Work>, names: Vec<String>, child: ExprRef) {
    work.push(Work::Pop);
    work.push(Work::Visit(child));
    work.push(Work::Push(names));
}

// The ONE definition of "this `let rec` is a recursive function group"
// (spec §2): every binding value is syntactically a direct Lambda. The
// resolver assigns the group layout ([names…, param] per function body,
// [names…] for the body) iff this holds, and machine.rs's Expr::LetRec arm
// builds the RecClosure group iff this holds -- sharing this function is
// what keeps the two from drifting apart.
pub fn is_direct_group(arena: &Arena, bindings: &[(String, Option<ExprRef>, ExprRef)]) -> bool {
    bindings.iter().all(|(_, _, v)| matches!(arena[*v], Expr::Lambda(..)))
}

fn visit(
    arena: &Arena,
    id: ExprRef,
    scopes: &[Vec<String>],
    found: &mut Vec<(ExprRef, VarRef)>,
    work: &mut Vec<Work>,
) {
    match &arena[id] {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Token(_) | Expr::TypeLit(_) => {}
        // A node shared under two scopes is caught when the table is built.
        Expr::Var(name) => found.push((id, lookup(scopes, name))),
        Expr::Tuple(items) | Expr::ListLit(items) => {
            for i in items.iter().rev() {
                work.push(Work::Visit(*i));
            }
        }
        Expr::Record(fields) => {
            for (_, v) in fields.iter().rev() {
                work.push(Work::Visit(*v));
            }
        }
        Expr::FieldAccess(target, _) => work.push(Work::Visit(*target)),
        // A Check's length slots are evaluated in the Check's own scope,
        // on entry, before the target value exists.
        Expr::Check(target, spec) => {
            work.push(Work::Visit(*target));
            for slot in spec.lens.iter().rev() {
                work.push(Work::Visit(*slot));
            }
        }
        Expr::Len(term) => {
            let mut witnesses = Vec::new();
            term.witnesses(&mut witnesses);
            for w in witnesses.into_iter().rev() {
                work.push(Work::Visit(w));
            }
        }
        Expr::Lambda(param, _, body) => scoped(work, vec![param.clone()], *body),
        Expr::App(f, a) => {
            work.push(Work::Visit(*a));
            work.push(Work::Visit(*f));
        }
        // `val` is resolved in the OUTER scope (plain let is non-recursive);
        // only `body` sees the new binder.
        Expr::Let(var, _, val, body) => {
            scoped(work, vec![var.clone()], *body);
            work.push(Work::Visit(*val));
        }
        Expr::LetRec(bindings, body) => {
            let names: Vec<String> = bindings.iter().map(|(n, _, _)| n.clone()).collect();
            // Reverse processing order: the body's frame [names…] comes last…
            scoped(work, names.clone(), *body);
            let group = is_direct_group(arena, bindings);
            for (_, _, v) in bindings.iter().rev() {
                if group {
                    // Direct-Lambda group: each function body runs in one
                    // frame [names…, param] (the Lambda node itself is never
                    // evaluated in the runtime's group form).
                    let Expr::Lambda(param, _, lbody) = &arena[*v] else { unreachable!("group checked above") };
                    let mut frame = names.clone();
                    frame.push(param.clone());
                    scoped(work, frame, *lbody);
                } else {
                    // Fallback: values evaluate in the OUTER scope; the names
                    // are bound only for `body`.
                    work.push(Work::Visit(*v));
                }
            }
        }
        Expr::BinOp(_, l, r) => {
            work.push(Work::Visit(*r));
            work.push(Work::Visit(*l));
        }
        Expr::If(c, t, e) => {
            work.push(Work::Visit(*e));
            work.push(Work::Visit(*t));
            work.push(Work::Visit(*c));
        }
        Expr::Perform(_, payload) => work.push(Work::Visit(*payload)),
        Expr::Handle { body, handler } => {
            work.push(Work::Visit(*handler));
            work.push(Work::Visit(*body));
        }
        Expr::MakeHandler { payload_var, resume_var, body, .. } => {
            scoped(work, vec![payload_var.clone(), resume_var.clone()], *body)
        }
        Expr::Match(scrutinee, arms) => {
            for (pat, guard, body) in arms.iter().rev() {
                let names = pattern_vars(pat);
                let bind = !names.is_empty(); // zero-var arm: no runtime frame
                if bind {
                    work.push(Work::Pop);
                }
                work.push(Work::Visit(*body));
                if let Some(g) = guard {
                    work.push(Work::Visit(*g));
                }
                if bind {
                    work.push(Work::Push(names));
                }
            }
            work.push(Work::Visit(*scrutinee));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::PRELUDE;
    use crate::expr::{Arena, Expr};

    fn let_rec_bindings(src: &str) -> (Arena, Vec<(String, Option<crate::expr::ExprRef>, crate::expr::ExprRef)>) {
        let (arena, _s, root) = crate::parser::parse(src).unwrap();
        let Expr::LetRec(bindings, _) = &arena[root] else { panic!("expected LetRec") };
        let bindings = bindings.as_ref().clone();
        (arena, bindings)
    }

    #[test]
    fn is_direct_group_true_only_when_every_value_is_a_lambda() {
        let (a, b) = let_rec_bindings("let rec f = fun x -> f(x) in f");
        assert!(is_direct_group(&a, &b));
        let (a, b) = let_rec_bindings("let rec f = fun x -> x and g = fun y -> y in f");
        assert!(is_direct_group(&a, &b));
        let (a, b) = let_rec_bindings("let rec f = fun x -> x and y = 3 in f");
        assert!(!is_direct_group(&a, &b));
        let (a, b) = let_rec_bindings("let rec x = 5 in x");
        assert!(!is_direct_group(&a, &b));
    }

    // Every Var node named `name` that the resolver assigned a VarRef to,
    // sorted (order of nodes in the arena is a parser detail).
    fn refs_of(src: &str, name: &str) -> Vec<VarRef> {
        let (arena, _spans, root) = crate::parser::parse(src).expect("parse failed");
        let resolved = resolve(&arena, root);
        let mut out: Vec<VarRef> = arena
            .iter()
            .filter_map(|(id, e)| match e {
                Expr::Var(n) if n == name => resolved.get(id),
                _ => None,
            })
            .collect();
        out.sort();
        out
    }

    fn local(hops: u32, slot: u32) -> VarRef {
        VarRef::Local { hops, slot }
    }

    #[test]
    fn lambda_param_is_slot_zero() {
        assert_eq!(refs_of("fun x -> x", "x"), vec![local(0, 0)]);
    }

    #[test]
    fn outer_lambda_param_is_one_hop_up() {
        assert_eq!(refs_of("fun x -> fun y -> x", "x"), vec![local(1, 0)]);
    }

    #[test]
    fn local_shadows_prelude_and_bare_name_hits_prelude() {
        assert_eq!(refs_of("let len = 1 in len", "len"), vec![local(0, 0)]);
        let idx = PRELUDE.iter().position(|(n, _)| *n == "len").unwrap() as u32;
        assert_eq!(refs_of("len", "len"), vec![VarRef::Prelude(idx)]);
    }

    #[test]
    fn duplicate_pattern_var_last_slot_wins() {
        assert_eq!(refs_of("match [1, 2] | [x, x] -> x", "x"), vec![local(0, 1)]);
    }

    #[test]
    fn zero_var_arm_pushes_no_frame() {
        // scrutinee y: [y] -> (0,0); arm `0` binds nothing so its body's y
        // is also (0,0); arm `z` pushes [z], so its body's y is one hop up.
        assert_eq!(
            refs_of("fun y -> match y | 0 -> y | z -> y", "y"),
            vec![local(0, 0), local(0, 0), local(1, 0)]
        );
    }

    #[test]
    fn guard_sees_arm_vars() {
        assert_eq!(refs_of("match 5 | n if n > 100 -> n", "n"), vec![local(0, 0), local(0, 0)]);
    }

    #[test]
    fn cons_pattern_binds_head_then_tail() {
        let src = "match [1] | h :: t -> h + len(t)";
        assert_eq!(refs_of(src, "h"), vec![local(0, 0)]);
        assert_eq!(refs_of(src, "t"), vec![local(0, 1)]);
    }

    #[test]
    fn pattern_vars_order() {
        let (arena, _s, root) = crate::parser::parse("match [1] | h :: [a, b] -> h").unwrap();
        let Expr::Match(_, arms) = &arena[root] else { panic!("expected Match") };
        assert_eq!(pattern_vars(&arms[0].0), vec!["h", "a", "b"]);
    }

    #[test]
    fn handler_frame_is_payload_then_resume() {
        assert_eq!(refs_of("handle 1 with handler choose(p, k) -> k", "k"), vec![local(0, 1)]);
        assert_eq!(refs_of("handle 1 with handler choose(p, k) -> p", "p"), vec![local(0, 0)]);
    }

    #[test]
    fn let_rec_group_mutual_recursion_layout() {
        // Inside each function body the frame is [ev, od, n]; the `let rec`
        // body's frame is [ev, od].
        let src = "let rec ev = fun n -> od(n) and od = fun n -> ev(n) in ev";
        assert_eq!(refs_of(src, "od"), vec![local(0, 1)]);
        assert_eq!(refs_of(src, "ev"), vec![local(0, 0), local(0, 0)]);
        assert_eq!(refs_of(src, "n"), vec![local(0, 2), local(0, 2)]);
    }

    #[test]
    fn let_rec_non_function_uses_fallback_frame() {
        assert_eq!(refs_of("let rec x = 5 in x", "x"), vec![local(0, 0)]);
    }

    // Documents the spec's confirmed decision: a group with any non-Lambda
    // binding is NOT recursive (same as today's runtime all_closures==false
    // path), so `f` is not in scope inside its own body. Scoping AND timing
    // match today: `f` is unbound there, and that only fails lazily, if the
    // body is ever evaluated.
    #[test]
    fn let_rec_mixed_group_is_not_recursive() {
        assert_eq!(
            refs_of("let rec f = fun x -> f(x) and y = 3 in y", "f"),
            vec![VarRef::Unbound]
        );
    }

    #[test]
    fn unbound_variable_resolves_to_unbound() {
        assert_eq!(refs_of("zzz", "zzz"), vec![VarRef::Unbound]);
    }

    #[test]
    fn pattern_vars_record_fields_in_stored_order() {
        let (arena, _s, root) = crate::parser::parse("match {a: 1, b: 2} | {a: p, b: q} -> q").unwrap();
        let Expr::Match(_, arms) = &arena[root] else { panic!("expected Match") };
        assert_eq!(pattern_vars(&arms[0].0), vec!["p", "q"]);
    }

    // `_` is an ordinary Pattern::Var, so it occupies a slot (faithful).
    #[test]
    fn wildcard_pattern_occupies_a_slot() {
        assert_eq!(refs_of("match [1, 2] | [_, y] -> y", "y"), vec![local(0, 1)]);
    }

    // Fallback group: values evaluate in the OUTER scope (`z` is the lambda
    // param, one frame), and only the body sees `x`.
    #[test]
    fn let_rec_fallback_values_resolve_in_outer_scope() {
        let src = "fun z -> let rec x = z in x";
        assert_eq!(refs_of(src, "z"), vec![local(0, 0)]);
        assert_eq!(refs_of(src, "x"), vec![local(0, 0)]);
    }

    // Built by hand (no parser recursion) and resolved on the DEFAULT test
    // thread stack (~2 MiB): a recursive resolver would overflow at this
    // depth, an iterative one doesn't.
    #[test]
    fn deep_let_chain_does_not_overflow_the_stack() {
        let n = 50_000usize;
        let mut arena = Arena::new();
        let last = arena.push(Expr::Var(format!("x{}", n - 1)));
        let mut body = last;
        for i in (0..n).rev() {
            let val = if i == 0 { arena.push(Expr::Int(0)) } else { arena.push(Expr::Var(format!("x{}", i - 1))) };
            body = arena.push(Expr::Let(format!("x{i}"), None, val, body));
        }
        let resolved = resolve(&arena, body);
        assert_eq!(resolved.get(last), Some(local(0, 0)));
    }

    #[test]
    #[should_panic(expected = "conflicting")]
    fn shared_node_under_different_scopes_is_detected() {
        let mut arena = Arena::new();
        let shared = arena.push(Expr::Var("a".into()));
        let l1 = arena.push(Expr::Lambda("a".into(), None, shared)); // a -> (0,0)
        let inner = arena.push(Expr::Lambda("b".into(), None, shared));
        let l2 = arena.push(Expr::Lambda("a".into(), None, inner)); // a -> (1,0)
        let root = arena.push(Expr::Tuple(vec![l1, l2]));
        resolve(&arena, root);
    }

    // The table covers only the Var nodes the walk met, so resolving a small
    // subtree late in a big arena (an annotation, machine::eval_type) costs
    // the subtree, not the arena prefix.
    #[test]
    fn the_table_spans_only_the_resolved_subtree() {
        let mut arena = Arena::new();
        for i in 0..10_000 {
            arena.push(Expr::Int(i));
        }
        let f = arena.push(Expr::Var("#List".into()));
        let x = arena.push(Expr::Var("#Int".into()));
        let app = arena.push(Expr::App(f, x));
        let resolved = resolve(&arena, app);
        assert_eq!(resolved.vars.len(), 2);
        let list = PRELUDE.iter().position(|(n, _)| *n == "List").unwrap() as u32;
        assert_eq!(resolved.get(f), Some(VarRef::Prelude(list)));
        assert_eq!(resolved.get(ExprRef::new(0)), None);
        assert_eq!(resolved.get(app), None);
    }
}
