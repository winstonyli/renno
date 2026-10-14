use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

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
// Dense over the range of Var nodes the walks met, `base..base + vars.len()`,
// not over the whole arena prefix: a table sized to a late subtree's position
// would make checking quadratic in the program size. It grows as more subtrees
// are resolved into it (`resolve_into`): `eval_pure_in` resolves each
// evaluation's copy of its root, and the checker keeps one table for all
// its check-time evaluations, since a closure built in one run is called in a
// later one, and its body needs an entry whichever run resolved it. A node
// must be resolved under one layout only. Open nodes (ones that read a binder
// outside themselves) are copied for each evaluation, so no two evaluations
// resolve the same one. Closed nodes resolve the same under every layout, so
// `sharing` finds them once, up front, and evaluations share them (`shares`).
#[derive(Default)]
pub struct Resolved {
    base: usize,
    vars: Vec<Option<VarRef>>,
    // Indexed by ExprRef: the node is closed and its entries are already in
    // `vars`. Covers the nodes that existed when `sharing` ran; later nodes
    // are not shared. Empty (nothing shared) for `Resolved::default()`.
    shared: Vec<bool>,
}

impl Resolved {
    pub fn get(&self, expr: ExprRef) -> Option<VarRef> {
        self.vars.get(expr.index().wrapping_sub(self.base)).copied().flatten()
    }

    // True if `expr` is closed: an evaluation may run the original node instead
    // of a copy, and its entries need no resolving.
    pub fn shares(&self, expr: ExprRef) -> bool {
        self.shared.get(expr.index()).copied().unwrap_or(false)
    }

    // Records where `id` lives, growing the range to cover it. Growing at the
    // front doubles, so a run of descending nodes stays linear overall.
    fn insert(&mut self, id: ExprRef, r: VarRef) {
        let i = id.index();
        if self.vars.is_empty() {
            self.base = i;
        } else if i < self.base {
            let grow = (self.base - i).max(self.vars.len()).min(self.base);
            self.vars.splice(0..0, std::iter::repeat_n(None, grow));
            self.base -= grow;
        }
        if i - self.base >= self.vars.len() {
            self.vars.resize(i - self.base + 1, None);
        }
        let slot = &mut self.vars[i - self.base];
        match *slot {
            Some(prev) if prev != r => {
                panic!("resolver: {id} reached under conflicting scopes ({prev:?} vs {r:?})")
            }
            _ => *slot = Some(r),
        }
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
    // The node's subtree is done (`sharing` only).
    Leave(ExprRef),
}

// Iterative on purpose: parser and typecheck flatten Let/Fun chains to keep
// long programs off the native stack, and a recursive walk here would bring
// the overflow back. The work stack is LIFO, so each node pushes its
// children in REVERSE of the order they must be processed in.
// Never fails on a name: an unresolvable one becomes VarRef::Unbound.
pub fn resolve(arena: &Arena, root: ExprRef) -> Resolved {
    let mut resolved = Resolved::default();
    resolve_into(arena, root, &[], &mut resolved);
    resolved
}

// `resolve`, adding to `resolved` and with `outer` as one enclosing frame
// (none if empty) around `root`: the layout of a runtime env that holds those
// names' values. A node `resolved` already holds must come out the same.
//
// A shared (closed) node is not entered: `sharing` already put its entries in.
// A debug build enters it anyway, so every debug run re-checks, through
// `insert`'s conflict panic, that a closed node resolves alike everywhere.
pub fn resolve_into(arena: &Arena, root: ExprRef, outer: &[String], resolved: &mut Resolved) {
    let skip = |id: ExprRef| !cfg!(debug_assertions) && resolved.shares(id);
    for (id, r) in walk(arena, root, outer, skip) {
        resolved.insert(id, r);
    }
}

// The table of the whole program `root` (what `resolve` returns) that also
// finds its closed nodes: ones whose Vars all resolve the same under any
// layout around them, so an evaluation can share them instead of copying them.
// A node is closed when every Var in it is
//   (a) bound by a binder inside the node, or
//   (b) a synthesized `#name` (always the prelude), or
//   (c) neither Local here nor a name any binder of the program binds (`bound`),
//       so no check-time scope can make it Local.
// For (a), hops and slot come only from frames the node itself pushes, and a
// frame's shape is syntactic. Nodes the walk does not visit (annotation
// subtrees, the Lambda node of a function group, nodes added later) are not
// closed. The entries this writes for open nodes are never read: an open
// original is always copied before it is evaluated.
pub fn sharing(arena: &Arena, root: ExprRef, bound: &HashSet<String>) -> Resolved {
    let mut resolved = Resolved { shared: vec![false; arena.len()], ..Resolved::default() };
    let mut scopes = Scopes::default();
    let mut found = Vec::new();
    let mut work = vec![Work::Visit(root)];
    // Per node entered and not yet left: the depth of its scope, and the lowest
    // frame (index into `scopes.frames`) any Var in it binds in. A frame
    // numbered below the node's own depth lies outside it.
    let mut lows: Vec<(isize, isize)> = Vec::new();
    while let Some(item) = work.pop() {
        match item {
            Work::Push(names) => scopes.push(names),
            Work::Pop => scopes.pop(),
            Work::Visit(id) => {
                lows.push((scopes.frames.len() as isize, isize::MAX));
                work.push(Work::Leave(id));
                visit(arena, id, &scopes, &mut found, &mut work);
                for (var, r) in found.drain(..) {
                    resolved.insert(var, r);
                    let Expr::Var(name) = &arena[var] else { unreachable!("only Var nodes are resolved") };
                    let low = match r {
                        VarRef::Local { hops, .. } => scopes.frames.len() as isize - 1 - hops as isize,
                        _ if name.starts_with('#') => isize::MAX,
                        _ if bound.contains(name) => -1,
                        _ => isize::MAX,
                    };
                    lows.last_mut().expect("the Var's own entry").1 = low;
                }
            }
            Work::Leave(id) => {
                let (depth, low) = lows.pop().expect("a node entered");
                resolved.shared[id.index()] = low >= depth;
                if let Some(parent) = lows.last_mut() {
                    parent.1 = parent.1.min(low);
                }
            }
        }
    }
    resolved
}

// The names `root` uses but does not bind (a prelude name counts as free; a
// synthesized `#name` does not), each once, in first-use order. typecheck
// looks these up to find the `let`s an annotation depends on.
pub fn free_vars(arena: &Arena, root: ExprRef) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (id, r) in walk(arena, root, &[], |_| false) {
        let Expr::Var(name) = &arena[id] else { unreachable!("only Var nodes are resolved") };
        if !matches!(r, VarRef::Local { .. }) && !name.starts_with('#') && !out.contains(name) {
            out.push(name.clone());
        }
    }
    out
}

// Every Var node under `root` with where it lives, in visit order, without
// entering a node `skip` names.
fn walk(arena: &Arena, root: ExprRef, outer: &[String], skip: impl Fn(ExprRef) -> bool) -> Vec<(ExprRef, VarRef)> {
    let mut found: Vec<(ExprRef, VarRef)> = Vec::new();
    let mut scopes = Scopes::default();
    if !outer.is_empty() {
        scopes.push(outer.to_vec());
    }
    let mut work = vec![Work::Visit(root)];
    while let Some(item) = work.pop() {
        match item {
            Work::Push(names) => scopes.push(names),
            Work::Pop => scopes.pop(),
            Work::Visit(id) if skip(id) => {}
            Work::Visit(id) => visit(arena, id, &scopes, &mut found, &mut work),
            Work::Leave(_) => unreachable!("only `sharing` leaves nodes"),
        }
    }
    found
}

// A name spelled `#builtin` (never lexable) is a synthesized reference to a
// prelude builtin: it ignores the lexical scope, so no user binding can capture
// it (typecheck::prelude_var builds these).
fn lookup(scopes: &Scopes, name: &str) -> VarRef {
    if let Some(builtin) = name.strip_prefix('#') {
        return prelude_ref(builtin);
    }
    if let Some(&(frame, slot)) = scopes.index.get(name).and_then(|s| s.last()) {
        return VarRef::Local { hops: (scopes.frames.len() - 1 - frame) as u32, slot };
    }
    prelude_ref(name)
}

// First PRELUDE entry of that name, else Unbound (not an error here: see
// VarRef::Unbound).
fn prelude_ref(name: &str) -> VarRef {
    static INDEX: OnceLock<HashMap<&'static str, u32>> = OnceLock::new();
    let index = INDEX.get_or_init(|| {
        let mut m = HashMap::new();
        for (i, (n, _)) in PRELUDE.iter().enumerate() {
            m.entry(*n).or_insert(i as u32);
        }
        m
    });
    match index.get(name) {
        Some(&i) => VarRef::Prelude(i),
        None => VarRef::Unbound,
    }
}

// The enclosing frames' binders: `frames` keeps each frame's names (to undo a
// push), `index` the stack of bindings per name, innermost frame and last slot
// on top, so a lookup is O(1) however deep the scope.
#[derive(Default)]
struct Scopes {
    frames: Vec<Vec<String>>,
    index: HashMap<String, Vec<(usize, u32)>>,
}

impl Scopes {
    fn push(&mut self, names: Vec<String>) {
        let frame = self.frames.len();
        // Later slot pushed last, so the last slot wins: preserves shadowing
        // for a duplicate binder in one frame, e.g. the pattern `(x, x)`.
        for (slot, n) in names.iter().enumerate() {
            self.index.entry(n.clone()).or_default().push((frame, slot as u32));
        }
        self.frames.push(names);
    }

    fn pop(&mut self) {
        let names = self.frames.pop().expect("pop of an empty scope stack");
        for n in names.iter().rev() {
            let stack = self.index.get_mut(n).expect("binder indexed at push");
            stack.pop();
            if stack.is_empty() {
                self.index.remove(n);
            }
        }
    }
}

// Queue `child` to be visited inside a frame of `names`: processed as
// Push, Visit, Pop (pushed here in reverse).
fn scoped(work: &mut Vec<Work>, names: Vec<String>, child: ExprRef) {
    work.push(Work::Pop);
    work.push(Work::Visit(child));
    work.push(Work::Push(names));
}

// What a `let rec` builds. The ONE syntactic classification (spec §2, §15),
// shared by the resolver, the machine and the checker so they cannot drift:
//   Functions: every value is a direct Lambda: a RecClosure group, layout
//              [names…, param] per function body, [names…] for the body.
//   Types:     every value is an unannotated call (rule P): the machine builds a
//              recursive type group; the values run in a frame [names…] of
//              placeholders, the body in [names…] of the finished values.
//   Plain:     anything else: the values run in the OUTER scope.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GroupKind {
    Functions,
    Types,
    Plain,
}

pub fn group_kind(arena: &Arena, bindings: &[(String, Option<ExprRef>, ExprRef)]) -> GroupKind {
    if bindings.iter().all(|(_, _, v)| matches!(arena[*v], Expr::Lambda(..))) {
        GroupKind::Functions
    } else if bindings.iter().all(|(_, ann, v)| ann.is_none() && matches!(arena[*v], Expr::App(..))) {
        GroupKind::Types
    } else {
        GroupKind::Plain
    }
}

fn visit(
    arena: &Arena,
    id: ExprRef,
    scopes: &Scopes,
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
        // A kept annotation runs in the scope of the binder it annotates.
        Expr::FieldAccess(target, _) | Expr::Annot(target, _) => work.push(Work::Visit(*target)),
        // A Check's length slots are evaluated in the Check's own scope,
        // on entry, before the target value exists.
        Expr::Check(target, spec) => {
            work.push(Work::Visit(*target));
            for slot in spec.lens.iter().rev() {
                work.push(Work::Visit(*slot));
            }
        }
        // So are a Conform's, and the variable holding its type.
        Expr::Conform(..) => {
            let mut kids = Vec::new();
            arena[id].children(&mut kids);
            for kid in kids.into_iter().rev() {
                work.push(Work::Visit(kid));
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
            let kind = group_kind(arena, bindings);
            if kind == GroupKind::Types {
                // Type group: the values run in a frame of the group's own
                // placeholders, the same layout [names…] as the body's, at the same
                // depth, so ONE scope serves them all (a scope per value would cost
                // O(names) each, quadratic in the group's size).
                work.push(Work::Pop);
                work.push(Work::Visit(*body));
                for (_, _, v) in bindings.iter().rev() {
                    work.push(Work::Visit(*v));
                }
                work.push(Work::Push(names));
                return;
            }
            // Reverse processing order: the body's frame [names…] comes last…
            scoped(work, names.clone(), *body);
            for (_, _, v) in bindings.iter().rev() {
                if kind == GroupKind::Functions {
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

    // The linear scan `Scopes` replaced, kept as the reference.
    fn naive_lookup(scopes: &[Vec<String>], name: &str) -> VarRef {
        let prelude = |n: &str| match PRELUDE.iter().position(|(p, _)| *p == n) {
            Some(i) => VarRef::Prelude(i as u32),
            None => VarRef::Unbound,
        };
        if let Some(builtin) = name.strip_prefix('#') {
            return prelude(builtin);
        }
        for (hops, frame) in scopes.iter().rev().enumerate() {
            if let Some(slot) = frame.iter().rposition(|n| n == name) {
                return VarRef::Local { hops: hops as u32, slot: slot as u32 };
            }
        }
        prelude(name)
    }

    #[test]
    fn scopes_lookup_matches_the_naive_scan() {
        let mut names: Vec<String> = ["a", "b", "c", "x", "len", "Int", "nope"].iter().map(|s| s.to_string()).collect();
        names.extend(["#len", "#Int", "#nope", "#a"].iter().map(|s| s.to_string()));
        let mut seed = 12345u64;
        let mut rnd = move |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as usize) % n
        };
        let mut scopes = Scopes::default();
        let mut naive: Vec<Vec<String>> = Vec::new();
        // Frame 0 plays the outer frame; duplicates and shadowing come from
        // the small name pool.
        for step in 0..4000 {
            if naive.is_empty() || (rnd(10) < 5 && naive.len() < 40) {
                let n = rnd(4);
                let frame: Vec<String> = (0..n).map(|_| names[rnd(5)].clone()).collect();
                scopes.push(frame.clone());
                naive.push(frame);
            } else {
                scopes.pop();
                naive.pop();
            }
            for name in &names {
                assert_eq!(lookup(&scopes, name), naive_lookup(&naive, name), "step {step}, name {name}");
            }
        }
    }

    #[test]
    fn group_kind_classifies_functions_types_and_plain() {
        use GroupKind::{Functions, Plain, Types};
        for (src, want) in [
            ("let rec f = fun x -> f(x) in f", Functions),
            ("let rec f = fun x -> x and g = fun y -> y in f", Functions),
            ("let rec f = fun x -> x and y = 3 in f", Plain),
            ("let rec x = 5 in x", Plain),
            ("let rec T = Union(Int, T) in T", Types),
            ("let rec A = Union(Int, B) and B = Tuple(Bool, A) in A", Types),
            ("let rec T = Union(Int, T) and n = 3 in T", Plain),
            ("let rec T: Type = Union(Int, T) in T", Plain),
        ] {
            let (a, b) = let_rec_bindings(src);
            assert_eq!(group_kind(&a, &b), want, "{src}");
        }
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
    fn free_vars_lists_the_names_a_tree_does_not_bind() {
        let free = |src: &str| {
            let (arena, _spans, root) = crate::parser::parse(src).expect("parse failed");
            free_vars(&arena, root)
        };
        assert_eq!(free("fun a -> Tuple((a, b, a, Int, b))"), vec!["Tuple", "b", "Int"]);
        assert_eq!(free("let x = y in x"), vec!["y"]);
        assert_eq!(free("let rec f = fun n -> f(g(n)) in f"), vec!["g"]);
        assert_eq!(free("match p | (h, t) -> h"), vec!["p"]);
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

    // An outer frame is one runtime frame around the root: `resolve_into`
    // numbers its names by slot, and what the root binds itself sits inside it.
    #[test]
    fn resolve_into_puts_the_outer_names_in_one_enclosing_frame() {
        let mut arena = Arena::new();
        let (a, b) = (arena.push(Expr::Var("a".into())), arena.push(Expr::Var("b".into())));
        let pair = arena.push(Expr::Tuple(vec![a, b]));
        let (c, a2) = (arena.push(Expr::Var("c".into())), arena.push(Expr::Var("a".into())));
        let inner = arena.push(Expr::Tuple(vec![c, a2]));
        let lam = arena.push(Expr::Lambda("c".into(), None, inner));
        let mut resolved = Resolved::default();
        resolve_into(&arena, pair, &["a".to_string(), "b".to_string()], &mut resolved);
        resolve_into(&arena, lam, &["a".to_string()], &mut resolved);
        assert_eq!((resolved.get(a), resolved.get(b)), (Some(local(0, 0)), Some(local(0, 1))));
        assert_eq!((resolved.get(c), resolved.get(a2)), (Some(local(0, 0)), Some(local(1, 0))));
        // No outer names: no frame, so a free name is the prelude's or unbound.
        let mut bare = Resolved::default();
        resolve_into(&arena, pair, &[], &mut bare);
        assert_eq!(bare.get(a), Some(VarRef::Unbound));
    }

    // Subtrees resolved later, at smaller or larger positions, extend the table.
    #[test]
    fn the_table_grows_in_both_directions() {
        let mut arena = Arena::new();
        let vars: Vec<ExprRef> = (0..40).map(|_| arena.push(Expr::Var("#Int".into()))).collect();
        let int = PRELUDE.iter().position(|(n, _)| *n == "Int").unwrap() as u32;
        let mut resolved = Resolved::default();
        for i in [20usize, 21, 10, 39, 0, 5, 30] {
            resolve_into(&arena, vars[i], &[], &mut resolved);
        }
        for (i, v) in vars.iter().enumerate() {
            let seen = [20usize, 21, 10, 39, 0, 5, 30].contains(&i);
            assert_eq!(resolved.get(*v), seen.then_some(VarRef::Prelude(int)), "node {i}");
        }
        // Descending order stays linear in the range: the front doubles.
        let mut down = Resolved::default();
        for v in vars.iter().rev() {
            resolve_into(&arena, *v, &[], &mut down);
        }
        assert!(down.vars.len() <= 2 * vars.len());
    }

    // The same node resolved again under the same layout is fine; under another
    // layout it is the resolver's conflict panic, through a shared table too.
    #[test]
    #[should_panic(expected = "conflicting")]
    fn a_node_in_a_shared_table_keeps_one_layout() {
        let mut arena = Arena::new();
        let a = arena.push(Expr::Var("a".into()));
        let mut resolved = Resolved::default();
        resolve_into(&arena, a, &["a".to_string()], &mut resolved);
        resolve_into(&arena, a, &["a".to_string()], &mut resolved);
        resolve_into(&arena, a, &["z".to_string(), "a".to_string()], &mut resolved);
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

    // Nodes of `src` (in arena order) that satisfy `pick`, and whether
    // `sharing` found each closed. `bound` stands for the names a binder of
    // the program binds.
    fn closed_of(src: &str, bound: &[&str], pick: impl Fn(&Arena, ExprRef) -> bool) -> Vec<bool> {
        let (arena, _spans, root) = crate::parser::parse(src).expect("parse failed");
        let bound: HashSet<String> = bound.iter().map(|s| s.to_string()).collect();
        let table = sharing(&arena, root, &bound);
        let picked: Vec<bool> = arena.iter().filter(|(id, _)| pick(&arena, *id)).map(|(id, _)| table.shares(id)).collect();
        assert!(!picked.is_empty(), "{src}: nothing picked");
        picked
    }

    fn var_named(name: &'static str) -> impl Fn(&Arena, ExprRef) -> bool {
        move |arena, id| matches!(&arena[id], Expr::Var(n) if n == name)
    }

    fn the_node(arena: &Arena, id: ExprRef) -> bool {
        // The root is the last node the parser pushes.
        id.index() + 1 == arena.len()
    }

    // `Resolved::sharing` finds the nodes that resolve the same under every walk.
    #[test]
    fn closed_nodes_are_the_ones_every_walk_resolves_alike() {
        // A lambda reads only its own parameter: closed; the Var itself is not.
        assert_eq!(closed_of("fun x -> x", &[], the_node), [true]);
        assert_eq!(closed_of("fun x -> x", &[], var_named("x")), [false]);
        // `Int` is closed while nothing binds it, and open once some binder does
        // (a walk might then see it as that binder's Local).
        assert_eq!(closed_of("let f = fun x -> Int in f", &[], var_named("Int")), [true]);
        assert_eq!(closed_of("let f = fun x -> Int in f", &["Int"], var_named("Int")), [false]);
        assert_eq!(closed_of("let Int = Bool in let f = fun x -> Int in f", &["Int"], var_named("Int")), [false]);
        // A synthesized `#Int` is the prelude's whatever binds `Int`.
        let mut arena = Arena::new();
        let hash_int = arena.push(Expr::Var("#Int".into()));
        let bound: HashSet<String> = ["Int".to_string()].into();
        assert!(sharing(&arena, hash_int, &bound).shares(hash_int));
        // A type group is closed when its values read only its own names, and
        // each value, which reads them from outside itself, is not.
        let src = "let rec T = Union(Int, T) in T";
        assert_eq!(closed_of(src, &["T"], the_node), [true]);
        assert_eq!(closed_of(src, &["T"], |a, id| matches!(a[id], Expr::App(..))), [false]);
        assert_eq!(closed_of("fun u -> (let rec T = Union(Int, T) in T)", &["T", "u"], the_node), [true]);
        assert_eq!(closed_of("let rec T = Union(Int, U) in T", &["T"], the_node), [true]);
        assert_eq!(closed_of("fun u -> (let rec T = Union(u, T) in T)", &["T", "u"], |a, id| matches!(a[id], Expr::LetRec(..))), [false]);
        // Match arms: one with no variable pushes no frame, one with a variable does.
        let src = "fun a -> match a | 0 -> a | y -> y";
        assert_eq!(closed_of(src, &["a", "y"], |a, id| matches!(a[id], Expr::Match(..))), [false]);
        assert_eq!(closed_of(src, &["a", "y"], the_node), [true]);
        assert_eq!(closed_of("match 1 | 2 -> 3 | y -> y", &["y"], |a, id| matches!(a[id], Expr::Match(..))), [true]);
        assert_eq!(closed_of("match 1 | 2 -> 3 | y -> y", &["y"], var_named("y")), [false]);
        // A handler clause binds its payload and resume variables.
        let src = "handle perform ping(3) with handler ping(x, k) -> k(x + 1)";
        assert_eq!(closed_of(src, &["x", "k"], |a, id| matches!(a[id], Expr::MakeHandler { .. })), [true]);
        assert_eq!(closed_of(src, &["x", "k"], |a, id| matches!(a[id], Expr::App(..))), [false]);
        // An annotation subtree is never visited: not closed.
        let src = "let t: Int = 3 in t";
        let (arena, _spans, root) = crate::parser::parse(src).expect("parse failed");
        let table = sharing(&arena, root, &HashSet::new());
        let Expr::Let(_, Some(ann), ..) = arena[root] else { panic!("an annotated let") };
        assert!(!table.shares(ann));
        assert!(table.shares(root));
        // A node added after the pass is not.
        let mut arena = arena;
        let added = arena.push(Expr::Int(1));
        assert!(!table.shares(added));
    }

    // 50k nested lets on the default stack: the pass is iterative.
    #[test]
    fn sharing_pass_is_iterative() {
        let mut arena = Arena::new();
        let mut body = arena.push(Expr::Int(0));
        for _ in 0..50_000 {
            let val = arena.push(Expr::Int(1));
            body = arena.push(Expr::Let("x".into(), None, val, body));
        }
        assert!(sharing(&arena, body, &HashSet::new()).shares(body));
    }
}
