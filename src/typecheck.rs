use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::expr::{Arena, BinOp, Expr, ExprRef, Pattern, SpanMap};
use crate::plist::PList;
use crate::span::Span;
use crate::types::{consistent, record_satisfies, EffectRow, Type};
use crate::util::find_field;

// The Span is always an ORIGINAL (pre-elaboration) node's -- the one whose
// type was being checked when the error fired, always already in scope
// (elaborate_node's own `expr` parameter, or a child ExprRef it destructured
// from that same original node) at the point of construction. typecheck
// never needs to invent a span for a node IT synthesizes (a boundary check,
// a wrap_fun_contract chain): every error path returns before any such node
// is built.
#[derive(Debug)]
pub struct TypeError(pub String, pub Span);

// A binding's type, plus the row-variable AND value-type-variable names
// generalized over it -- quantified fresh at every use, the way ML/
// Haskell generalize a `let`-bound type. Row variables only ever come
// from what the user wrote (an explicit `->{e}` annotation) -- renno has
// no unification engine, so there's nothing here that infers one from
// scratch. Type variables (`type_vars`) are the opposite: manufactured
// purely by typecheck::passthrough_generalizable_params's own inference,
// never written by a user -- see Type::Var's own doc comment. Either way,
// this stays simple name substitution (extend_generalized computes both
// sets once at the `let`, lookup renames them to fresh names at each
// reference), not a general unification engine.
//
// A row variable being user-written rather than manufactured does NOT
// make it safe to generalize wherever free_row_vars finds it -- see
// generalizable_row_vars's own doc comment for why it still needs a
// ctx-wide "is this still open in an enclosing scope" check, the same
// class of hazard Type::Var has (just reachable through ordinary
// annotations instead of only through passthrough inference).
//
// The general rule behind BOTH exclusions, stated once here so a future
// THIRD kind of variable needing this treatment has one place to read it
// instead of re-deriving it: generalize(ctx, ty) = freevars(ty) MINUS
// freevars(ctx) -- never quantify over a name that's still free somewhere
// still open in the enclosing scope. `extend_generalized_with_type_vars`'s
// caller-supplied list is just an optimization of that same rule, valid
// only when there's exactly one call site that manufactures the variable
// and so can hand back precisely which names are its own (Type::Var's
// case); with no such single manufacturer (EffectRow::Var's case, born
// from any annotation anywhere), the rule has to be applied directly via a
// ctx-wide scan (generalizable_row_vars/free_row_vars_in_ctx below).
#[derive(Clone)]
struct Scheme {
    row_vars: Vec<String>,
    type_vars: Vec<String>,
    ty: Type,
}

impl Scheme {
    fn mono(ty: Type) -> Scheme {
        Scheme { row_vars: Vec::new(), type_vars: Vec::new(), ty }
    }
}

type Ctx = PList<Scheme>;

// The real, threaded state a Hindley-Milner-style unifier needs, carried
// through every elaborate()/elaborate_node() call as `&mut InferCtx` --
// see the design spec's own "Data model" section for the full rationale
// (functional substitution over union-find). Generalization does NOT
// live here -- no level counter, no birth-level tracking -- it reuses
// the ctx-wide technique a concurrent row-variable fix already proved
// in this codebase (see Task 7); this struct is just a substitution.
//
// #[allow(dead_code)]: nothing calls unify() from a real decision point
// yet (that's Tasks 4-6), so `subst` and most of these methods are
// genuinely unread by this task alone -- lift this once a later task
// wires them in.
#[allow(dead_code)]
struct InferCtx {
    // Grows monotonically as unify() binds variables; never shrinks --
    // no backtracking, matching this checker's existing single-pass
    // character everywhere else.
    subst: HashMap<String, Type>,
}

#[allow(dead_code)]
impl InferCtx {
    fn new() -> InferCtx {
        InferCtx { subst: HashMap::new() }
    }

    // Mints a fresh Type::Var -- every unannotated binding site (Lambda
    // params, Pattern::Var, LetRec self-reference) and every
    // unify()-driven "this must be some List/Fun shape, but we don't
    // know its element/param/return type yet" moment goes through this,
    // never Type::Var(name) built by hand. Reuses the EXISTING
    // fresh_type_name counter (already in this file, used by `lookup`'s
    // own instantiation) rather than a second, separate one -- there's
    // no reason a fresh name minted here and one minted by instantiating
    // a generalized scheme should draw from different namespaces.
    fn fresh_var(&mut self, base: &str) -> Type {
        Type::Var(fresh_type_name(base))
    }

    // Walks `ty` through the current substitution ONE STEP -- following
    // it to whatever it now points to, recursively, until reaching
    // either an unbound variable or a concrete shape, but NOT recursing
    // into a Fun/List/Tuple/Record's own NESTED positions. This is what
    // `unify`'s own structural recursion needs internally at each level
    // (mirroring how little `subst_type`/`resolve_row` already resolve
    // at a time) -- it is NOT enough on its own to hand back a complete,
    // fully-substituted result type to some OTHER caller; see
    // `resolve_deep` for that.
    fn resolve(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(name) => match self.subst.get(name) {
                Some(bound) => self.resolve(bound),
                None => ty.clone(),
            },
            other => other.clone(),
        }
    }

    // Like `resolve`, but recursively rebuilds a FULLY substituted Type,
    // walking every nested position (the same recursive shape
    // `subst_type` already has, just consulting `self.subst` instead of
    // a passed-in map). Used anywhere a caller hands back a *complete*
    // result type rather than continuing to unify against it
    // structurally: Expr::App's own return type (Task 4), and the
    // result types BinOp::Cons/If/Match/ListLit hand back after their
    // own unify() calls succeed (Tasks 5-6), and generalization's own
    // free-variable collection (Task 7).
    fn resolve_deep(&self, ty: &Type) -> Type {
        match self.resolve(ty) {
            Type::Fun(param, row, ret) => {
                Type::Fun(Rc::new(self.resolve_deep(&param)), row, Rc::new(self.resolve_deep(&ret)))
            }
            Type::List(elem) => Type::List(Rc::new(self.resolve_deep(&elem))),
            Type::Tuple(items) => Type::Tuple(Rc::new(items.iter().map(|t| self.resolve_deep(t)).collect())),
            Type::Union(items) => Type::Union(Rc::new(items.iter().map(|t| self.resolve_deep(t)).collect())),
            Type::Record(fields) => {
                Type::Record(Rc::new(fields.iter().map(|(n, t)| (n.clone(), self.resolve_deep(t))).collect()))
            }
            other => other,
        }
    }
}

// Does `name` (a Type::Var's own name, already confirmed unbound) appear
// free anywhere inside `ty`? Resolves through infer's substitution as it
// recurses, so an already-bound variable's own target is checked too --
// this is what an occurs-check needs to actually catch a genuinely
// infinite type (e.g. attempting to unify `a` with `List(a)`), not just
// a syntactically-obvious one.
//
// #[allow(dead_code)]: only called from unify(), which nothing calls yet.
#[allow(dead_code)]
fn occurs_in(name: &str, ty: &Type, infer: &InferCtx) -> bool {
    match infer.resolve(ty) {
        Type::Var(n) => n == name,
        Type::List(elem) => occurs_in(name, &elem, infer),
        Type::Fun(param, _row, ret) => occurs_in(name, &param, infer) || occurs_in(name, &ret, infer),
        Type::Tuple(items) | Type::Union(items) => items.iter().any(|t| occurs_in(name, t, infer)),
        Type::Record(fields) => fields.iter().any(|(_, t)| occurs_in(name, t, infer)),
        Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) => false,
    }
}

// The real unifier. Resolves both sides through infer.subst first, then:
// an unbound Type::Var on either side gets BOUND (after an occurs-check)
// to the other, already-resolved side; Type::Dyn on either side succeeds
// trivially with no new binding (this is what keeps Dyn's own "consistent
// with everything" story intact inside the new engine -- unify never
// forces a Dyn-sourced value's shape); two matching concrete shapes
// recurse structurally; anything else is a real type mismatch.
//
// #[allow(dead_code)]: not yet called from any real decision point --
// Tasks 4-6 wire it into Expr::App/BinOp::Cons/If/Match/ListLit.
#[allow(dead_code)]
fn unify(t1: &Type, t2: &Type, infer: &mut InferCtx, span: Span) -> Result<(), TypeError> {
    let t1 = infer.resolve(t1);
    let t2 = infer.resolve(t2);
    match (&t1, &t2) {
        (Type::Var(n1), Type::Var(n2)) if n1 == n2 => Ok(()),
        (Type::Var(name), other) | (other, Type::Var(name)) => {
            if occurs_in(name, other, infer) {
                return Err(TypeError(format!("infinite type: {name} occurs in {other}"), span));
            }
            infer.subst.insert(name.clone(), other.clone());
            Ok(())
        }
        (Type::Dyn, _) | (_, Type::Dyn) => Ok(()),
        (Type::Int, Type::Int) | (Type::Float, Type::Float) | (Type::Bool, Type::Bool) | (Type::Str, Type::Str) => Ok(()),
        (Type::Token(a), Type::Token(b)) if a == b => Ok(()),
        (Type::List(a), Type::List(b)) => unify(a, b, infer, span),
        (Type::Fun(p1, _r1, ret1), Type::Fun(p2, _r2, ret2)) => {
            unify(p1, p2, infer, span)?;
            unify(ret1, ret2, infer, span)
            // Row unification is out of scope here -- effect rows already
            // have their own, separate, unrelated generalize/instantiate
            // mechanism (bind_row_vars/resolve_row), untouched by this
            // plan; two Fun types unify on their param/return shape only.
        }
        (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => {
            a.iter().zip(b.iter()).try_for_each(|(x, y)| unify(x, y, infer, span))
        }
        (Type::Record(a), Type::Record(b)) if a.len() == b.len() => {
            for (name, ty_a) in a.iter() {
                match find_field(b, name) {
                    Some(ty_b) => unify(ty_a, ty_b, infer, span)?,
                    None => return Err(TypeError(format!("type mismatch: expected {t2}, found {t1}"), span)),
                }
            }
            Ok(())
        }
        _ => Err(TypeError(format!("type mismatch: expected {t2}, found {t1}"), span)),
    }
}

fn lookup(ctx: &Ctx, name: &str) -> Type {
    // Unbound at typecheck time: don't error here, machine::run's own
    // `unbound variable` panic at runtime is the right place for that.
    match ctx.get(name) {
        None => Type::Dyn,
        Some(scheme) if scheme.row_vars.is_empty() && scheme.type_vars.is_empty() => scheme.ty,
        Some(scheme) => {
            let row_subst: HashMap<String, EffectRow> = scheme
                .row_vars
                .iter()
                .map(|v| (v.clone(), EffectRow::Var(fresh_row_name(v))))
                .collect();
            let type_subst: HashMap<String, Type> =
                scheme.type_vars.iter().map(|v| (v.clone(), Type::Var(fresh_type_name(v)))).collect();
            subst_type(&scheme.ty, &row_subst, &type_subst)
        }
    }
}

// Ordinary (non-generalized) binding -- lambda parameters, and anything
// else that isn't a `let`. Row variables in `ty`, if any, stay exactly as
// written: shared verbatim by every use within this one scope, not
// instantiated fresh per use (that's what `extend_generalized` is for).
fn extend(ctx: &Ctx, name: &str, ty: Type) -> Ctx {
    ctx.bind(name, Scheme::mono(ty))
}

// `let`-binding: if `ty` mentions any row-variable names (from an explicit
// `->{e}` annotation somewhere in it), generalize over them so each
// reference gets its own fresh instantiation -- otherwise two calls to the
// same row-polymorphic function with different concrete callbacks would
// wrongly be forced to agree on one row.
//
// Deliberately does NOT auto-derive type_vars from free_type_vars(&ty) --
// see extend_generalized_with_type_vars for why, and why type_vars needs a
// narrower fix (a caller-supplied list) instead of what row_vars gets below
// (a ctx-wide exclusion check). row_vars still auto-derives from
// free_row_vars(&ty), but only picks up a name if generalizable_row_vars
// confirms it ISN'T also free somewhere still open in `ctx` -- see that
// function's own doc comment for why an EffectRow::Var needs this
// (confirmed reachable: `let f = fun cb: (Dyn ->{e} Dyn) -> let g = cb in g
// in f(fun y -> perform choose(y))(0)` used to typecheck clean and only
// panic at runtime -- "e" leaking from cb's still-open annotation into g's
// generalized scheme, then getting freshly (and wrongly) renamed on every
// `g` reference, exactly like the Type::Var bug this mirrors). Delegates to
// extend_generalized_with_type_vars (with an empty type_vars) rather than
// repeating its body: the two are identical apart from that one field, and
// a hand-copied second implementation is exactly the kind of thing that
// silently drifts out of sync with the first the next time either changes.
fn extend_generalized(ctx: &Ctx, name: &str, ty: Type) -> Ctx {
    extend_generalized_with_type_vars(ctx, name, ty, Vec::new())
}

// Identical to extend_generalized except type_vars is supplied by the
// caller instead of auto-derived -- Type::Var is only ever manufactured by
// ONE place (elaborate_generalizing_passthrough, for a lambda parameter
// WHILE that parameter's own function body is still being elaborated), so a
// plain alias-only `let` nested inside that body (e.g. `let y = x in y`)
// auto-harvesting free Type::Vars the way row_vars harvests row vars would
// wrongly re-generalize over the still-open outer parameter's own fresh
// name -- disconnecting `y`'s uses from `x`'s (each `lookup` of `y` would
// mint yet another fresh name instead of sharing the parameter's). Because
// there's exactly one source, the caller can hand back the EXACT fresh
// names it just introduced, which are safe to generalize by construction.
// row_vars has no single source like that (any annotation currently in
// scope can introduce one), so it uses generalizable_row_vars's ctx-wide
// check instead -- see that function's own doc comment. The only caller
// that may supply `type_vars` here is the `Expr::Let`/`Expr::LetRec` wiring
// for a binding whose value came back from elaborate_generalizing_passthrough
// (and only when there's no explicit annotation on the binding).
fn extend_generalized_with_type_vars(ctx: &Ctx, name: &str, ty: Type, type_vars: Vec<String>) -> Ctx {
    let row_vars = generalizable_row_vars(ctx, &ty);
    ctx.bind(name, Scheme { row_vars, type_vars, ty })
}

// Row-variable names free in `ty` that are actually safe to generalize
// over: free_row_vars(ty) MINUS whatever's ALSO free somewhere still open
// in `ctx`. Unlike Type::Var (see extend_generalized_with_type_vars's doc
// comment), an EffectRow::Var has no single manufacturing call site whose
// caller could hand back a safe list -- it comes from ANY `->{e}`
// annotation anywhere currently in scope (a lambda parameter's own
// annotation, bound via plain `extend`, stays open for that parameter's
// entire body). A nested `let` that merely happens to mention that same
// row-var name (most often through a bare alias like `let g = cb in g`,
// but any value whose type structurally embeds it -- a tuple, a record --
// works the same way) must not re-generalize over it, for exactly the
// reason Type::Var's own fix document gives: the name isn't THIS let's own
// to quantify, it belongs to the enclosing, not-yet-closed binding, and
// generalizing it anyway disconnects every future reference from that
// binding's real identity.
//
// Skips the ctx walk entirely when `ty` has no row vars at all -- the
// overwhelmingly common case (most bindings never mention a row-polymorphic
// annotation), so an ordinary long `let` chain (see
// deeply_nested_let_chain_does_not_overflow_the_stack) stays O(1) per let
// instead of paying an O(ctx depth) scan on every single one; only a let
// whose value's type actually mentions a row variable pays that cost.
fn generalizable_row_vars(ctx: &Ctx, ty: &Type) -> Vec<String> {
    let candidates = free_row_vars(ty);
    if candidates.is_empty() {
        return Vec::new();
    }
    let still_open = free_row_vars_in_ctx(ctx);
    candidates.difference(&still_open).cloned().collect()
}

// Every row-variable name free in any type currently bound in `ctx` --
// generalizable_row_vars's own "still open" set. Walks the whole visible
// scope chain (PList::for_each, iterative -- see its own doc comment), not
// just the immediately-enclosing binding: a row var can be several scopes
// out from the `let` that's about to (maybe) re-capture it. Excludes each
// visited scheme's OWN `row_vars`: a name that scheme already generalized
// over is SEALED, not still open -- `lookup` always renames it to a fresh
// name before it's ever read again (see `lookup`'s own fresh_row_name
// call), so the literal name surviving inside that scheme's stored `ty` is
// dead and must not be treated as blocking some unrelated, later `let`
// that happens to reuse the same letter in its own annotation.
//
// ponytail: this is an O(ctx depth) walk, paid once per row-var-carrying
// `let` (generalizable_row_vars's own short-circuit keeps every OTHER let
// at O(1)) -- fine at this codebase's scale, but a chain of N sequential
// row-polymorphic `let`s costs O(N^2) instead of O(N) to typecheck.
// Upgrade path if that ever matters: thread a cumulative "still open row
// vars" set incrementally through Ctx itself (unioned in once per `bind`,
// O(1) amortized) instead of re-walking the whole chain from scratch here.
fn free_row_vars_in_ctx(ctx: &Ctx) -> BTreeSet<String> {
    let mut vars = BTreeSet::new();
    ctx.for_each(|scheme: &Scheme| {
        let sealed: BTreeSet<String> = scheme.row_vars.iter().cloned().collect();
        vars.extend(free_row_vars(&scheme.ty).difference(&sealed).cloned());
    });
    vars
}

fn free_row_vars(ty: &Type) -> BTreeSet<String> {
    match ty {
        Type::Fun(param, row, ret) => {
            let mut vars = free_row_vars(param);
            vars.extend(free_row_vars(ret));
            if let EffectRow::Var(name) = row {
                vars.insert(name.clone());
            }
            vars
        }
        Type::List(elem) => free_row_vars(elem),
        Type::Tuple(items) | Type::Union(items) => items.iter().flat_map(free_row_vars).collect(),
        Type::Record(fields) => fields.iter().flat_map(|(_, t)| free_row_vars(t)).collect(),
        Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) | Type::Var(_) => BTreeSet::new(),
    }
}

// Conservative, purely syntactic: does `param` (an unannotated lambda
// parameter's name) only ever appear, within `body`, in a position that's
// safe to generalize? See the design spec's "The passthrough rule" for
// the full intended rule set -- this implements a NARROWER subset (see
// this plan's own "Before you start"): only (a) the function's own tail
// return expression being exactly a (possibly let-aliased) reference to
// the tracked parameter, or (b) the parameter simply never appearing at
// all. `tracked` starts as `{param}` and grows through simple
// `let name2 = <tracked> in body` aliasing as the walk descends -- this
// is what lets `let f = fun x -> let y = x in y in ...` still generalize.
// Anything else touching a tracked name (an operator, an `if`/`match`,
// being passed as an argument, being bound via anything but a plain
// `let`, ...) disqualifies it. `is_tail` marks whether `e` is currently in
// the function's own return position -- only there does a bare tracked
// reference count as "the value flows out," which is what
// passthrough_generalizable_params below actually needs to know.
fn is_passthrough_safe(arena: &Arena, e: ExprRef, tracked: &BTreeSet<String>, is_tail: bool) -> bool {
    match &arena[e] {
        Expr::Var(name) => !tracked.contains(name) || is_tail,
        Expr::Let(var, ann, val, body) => {
            let val_is_bare_alias = ann.is_none() && matches!(&arena[*val], Expr::Var(n) if tracked.contains(n));
            if val_is_bare_alias {
                let mut widened = tracked.clone();
                widened.insert(var.clone());
                is_passthrough_safe(arena, *body, &widened, is_tail)
            } else {
                !mentions_any(arena, *val, tracked) && is_passthrough_safe(arena, *body, tracked, is_tail)
            }
        }
        _ => !mentions_any(arena, e, tracked),
    }
}

// Does `e`'s whole expression tree reference any name in `tracked`,
// anywhere? Shared by is_passthrough_safe's own conservative default arm
// (anything that isn't a bare Var or a plain Let is disqualified the
// moment it touches a tracked name anywhere within it) -- same "walk the
// whole Expr grammar" shape as parser::contains_perform, for the same
// reason: a flat, exhaustive match over every variant, not a partial one.
fn mentions_any(arena: &Arena, e: ExprRef, tracked: &BTreeSet<String>) -> bool {
    match &arena[e] {
        Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Token(_) => false,
        Expr::Var(name) => tracked.contains(name),
        Expr::Tuple(items) | Expr::ListLit(items) => items.iter().any(|i| mentions_any(arena, *i, tracked)),
        Expr::Record(fields) => fields.iter().any(|(_, v)| mentions_any(arena, *v, tracked)),
        Expr::FieldAccess(target, _) => mentions_any(arena, *target, tracked),
        Expr::Lambda(_, _, body) => mentions_any(arena, *body, tracked),
        Expr::App(f, a) => mentions_any(arena, *f, tracked) || mentions_any(arena, *a, tracked),
        Expr::Let(_, _, val, body) => mentions_any(arena, *val, tracked) || mentions_any(arena, *body, tracked),
        Expr::LetRec(bindings, body) => {
            bindings.iter().any(|(_, _, v)| mentions_any(arena, *v, tracked)) || mentions_any(arena, *body, tracked)
        }
        Expr::BinOp(_, l, r) => mentions_any(arena, *l, tracked) || mentions_any(arena, *r, tracked),
        Expr::If(c, t, e) => mentions_any(arena, *c, tracked) || mentions_any(arena, *t, tracked) || mentions_any(arena, *e, tracked),
        Expr::Perform(_, payload) => mentions_any(arena, *payload, tracked),
        Expr::Handle { body, handler } => mentions_any(arena, *body, tracked) || mentions_any(arena, *handler, tracked),
        Expr::MakeHandler { body, .. } => mentions_any(arena, *body, tracked),
        Expr::Match(scrutinee, arms) => {
            mentions_any(arena, *scrutinee, tracked)
                || arms.iter().any(|(_, guard, body)| {
                    guard.is_some_and(|g| mentions_any(arena, g, tracked)) || mentions_any(arena, *body, tracked)
                })
        }
    }
}

// `val` is known to be (a curried chain of) Expr::Lambda -- peels every
// UNANNOTATED parameter off in order (an already-annotated one is never a
// candidate: this design never overrides an explicit annotation) and
// classifies each independently against the function's own final body via
// is_passthrough_safe, starting fresh from `{that parameter's own name}`
// each time (one parameter's generalizability never depends on another's).
fn passthrough_generalizable_params(arena: &Arena, val: ExprRef) -> Vec<bool> {
    let mut params = Vec::new();
    let mut cur = val;
    while let Expr::Lambda(param, ann, body) = &arena[cur] {
        params.push((param.clone(), ann.is_none()));
        cur = *body;
    }
    let final_body = cur;
    params
        .iter()
        .map(|(name, unannotated)| {
            *unannotated && {
                let tracked: BTreeSet<String> = std::iter::once(name.clone()).collect();
                is_passthrough_safe(arena, final_body, &tracked, true)
            }
        })
        .collect()
}

fn fresh_row_name(base: &str) -> String {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{base}#{n}")
}

// Same idea as fresh_row_name, its own separate counter -- a Type::Var's
// namespace is unrelated to an EffectRow::Var's, so there's no reason to
// share one (and every reason not to, in case they're ever compared or
// logged together during debugging).
fn fresh_type_name(base: &str) -> String {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{base}#{n}")
}

fn resolve_row(row: &EffectRow, subst: &HashMap<String, EffectRow>) -> EffectRow {
    match row {
        EffectRow::Var(name) => subst.get(name).cloned().unwrap_or_else(|| row.clone()),
        other => other.clone(),
    }
}

// Recurses into every Type variant free_row_vars also recurses into
// (Fun/List/Tuple/Union/Record), so a row variable this function finds
// reachable is also one this function can rename -- the two are meant to
// stay in lockstep, since a variable extend_generalized captured (via
// free_row_vars) but subst_type then failed to touch would keep its
// literal original name at every instantiation instead of getting a
// fresh one per reference, silently breaking per-reference row
// polymorphism for whatever's inside that untouched variant.
//
// The Union and Record arms are UNVERIFIED by any test: no renno program
// was found where a shared-vs-fresh row-var name inside a Union or
// Record actually changes typecheck's verdict or runtime behavior --
// row_consistent treats any EffectRow::Var as consistent with anything
// regardless of name, and the only way to get a callable Fun back out of
// a Union/Record is by pattern-matching it, which types the result Dyn
// regardless of what precision subst_type preserved going in. Kept
// anyway, on the same "recurse everywhere free_row_vars does" principle
// the Fun/List/Tuple arms already establish -- correct by construction,
// not by a failing case this fixed.
fn subst_type(ty: &Type, row_subst: &HashMap<String, EffectRow>, type_subst: &HashMap<String, Type>) -> Type {
    match ty {
        Type::Var(name) => type_subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
        Type::Fun(param, row, ret) => Type::Fun(
            Rc::new(subst_type(param, row_subst, type_subst)),
            resolve_row(row, row_subst),
            Rc::new(subst_type(ret, row_subst, type_subst)),
        ),
        Type::List(elem) => Type::List(Rc::new(subst_type(elem, row_subst, type_subst))),
        Type::Tuple(items) => Type::Tuple(Rc::new(items.iter().map(|t| subst_type(t, row_subst, type_subst)).collect())),
        Type::Union(alts) => Type::Union(Rc::new(alts.iter().map(|t| subst_type(t, row_subst, type_subst)).collect())),
        Type::Record(fields) => {
            Type::Record(Rc::new(fields.iter().map(|(n, t)| (n.clone(), subst_type(t, row_subst, type_subst))).collect()))
        }
        other => other.clone(),
    }
}

// Where a row variable actually gets bound to something concrete: `param`
// is the callee's OWN declared parameter type (e.g. `(Dyn ->{e} Dyn)`),
// `arg` is the actual argument's inferred type at this call site (e.g.
// `(Dyn -> Dyn)` with a Closed({choose}) row, from an ordinary unannotated
// closure). Every row variable found in `param`'s structure whose matching
// position in `arg` is concrete gets bound in `subst`; the caller applies
// that substitution to the call's return type (and its own row) so the
// variable's meaning flows out of this one application.
fn bind_row_vars(param: &Type, arg: &Type, subst: &mut HashMap<String, EffectRow>) {
    if let (Type::Fun(p1, r1, p2), Type::Fun(a1, r2, a2)) = (param, arg) {
        if let EffectRow::Var(name) = r1 {
            if !matches!(r2, EffectRow::Var(_)) {
                subst.entry(name.clone()).or_insert_with(|| r2.clone());
            }
        }
        bind_row_vars(p1, a1, subst);
        bind_row_vars(p2, a2, subst);
    }
}

// Same idea as bind_row_vars, for ordinary Type::Var instead of
// EffectRow::Var: `param` is the callee's OWN (already-instantiated,
// possibly Type::Var-containing) declared parameter type; `arg` is the
// caller's actual argument's inferred type. Where `param`'s structure
// names a bare Type::Var and `arg`'s matching position is something more
// concrete, bind it -- the caller substitutes that into the return type
// so a generalized function's result is precisely typed once its
// argument's own type is known, not just Type::Var (which would render
// exactly like Dyn -- see Type::Var's own doc comment). This is what
// turns `id(5)`'s call-site type from Dyn into Int.
//
// #[allow(dead_code)]: its only caller (Expr::App's Fun-callee arm) now
// uses unify()+resolve_deep instead, which supersede it -- the function
// itself is deleted in Task 7, alongside the rest of the passthrough-era
// machinery it belongs to.
#[allow(dead_code)]
fn bind_type_vars(param: &Type, arg: &Type, subst: &mut HashMap<String, Type>) {
    match (param, arg) {
        (Type::Var(name), _) => {
            if !matches!(arg, Type::Var(_)) {
                subst.entry(name.clone()).or_insert_with(|| arg.clone());
            }
        }
        (Type::Fun(p1, _, p2), Type::Fun(a1, _, a2)) => {
            bind_type_vars(p1, a1, subst);
            bind_type_vars(p2, a2, subst);
        }
        _ => {}
    }
}

// "must be callable" -- the shape used wherever we need to check a value is
// applicable at all without knowing its exact signature (an unannotated
// Dyn-typed callee, or the innermost check inside a function contract).
fn any_fun() -> Type {
    Type::Fun(Rc::new(Type::Dyn), EffectRow::Dyn, Rc::new(Type::Dyn))
}

// The only place a runtime boundary check gets built: `from` is Dyn (or
// Type::Var, treated identically -- see its own doc comment) and `to` is
// concrete. If both sides are concrete
// and disagree, that's a real static error -- reject before running at
// all, UNLESS `to` is a Record that `from` (also a concrete Record)
// width-satisfies (see types::record_satisfies' own doc comment) -- the
// one place `from`/`to`'s naming is more than accidental, since that
// relation is genuinely directional, unlike consistent()'s own symmetric
// one. Accepted with NO wrapping at all: a wider record needs no runtime
// projection to be used where a narrower type is expected (see
// Pattern::Record's own doc comment for why). If `from` is already
// exactly consistent and concrete, no check needed either: zero overhead
// for fully-annotated code. The error, if any, points at `span` -- the
// CALLER's job to have already looked up via the ORIGINAL
// (pre-elaboration) ExprRef for the value in question, e.g. `spans[val]`
// where `val` is a field straight off the un-elaborated node, NOT the
// elaborated `e` this function receives to potentially wrap:
// elaborate_node re-pushes almost every compound node it touches (App,
// BinOp, If, ...) regardless of whether anything actually changed, so
// `e` itself is often already a brand new ExprRef past the end of the
// parser-built SpanMap by the time it reaches here -- indexing spans BY
// IT, not by the original, was a latent bug (worked by coincidence
// whenever the mismatched value happened to be a leaf that elaborate_node
// returns unchanged).
fn coerce(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, span: Span) -> Result<ExprRef, TypeError> {
    if !consistent(from, to) {
        if let (Type::Record(actual), Type::Record(required)) = (from, to)
            && record_satisfies(required, actual)
        {
            return Ok(e);
        }
        return Err(TypeError(format!("type mismatch: expected {to}, found {from}"), span));
    }
    if !matches!(from, Type::Dyn | Type::Var(_)) || *to == Type::Dyn || matches!(to, Type::Var(_)) {
        return Ok(e);
    }
    Ok(build_boundary_check(arena, e, to))
}

// Like `coerce`, but the target is "Int or Float" rather than one fixed
// type -- Add/Sub/Mul/Div/Mod/Lt's own operand check. Kept separate from
// the general coerce()/consistent() machinery ON PURPOSE: Int and Float
// stay mutually INCONSISTENT everywhere else in the type system (a
// [Float]-annotated parameter still statically rejects a [Int] argument,
// a Fun's declared Int parameter still rejects a Float argument, etc.) --
// only these specific operators treat the two as interchangeable, so this
// local helper is where that interchangeability lives. Building the
// actual Dyn-boundary runtime check by reaching into build_boundary_check
// with an ad-hoc Union([Int, Float]) is safe reuse despite that: it only
// asks "does this runtime value look like one of these shapes," which
// carries no implication for consistent() or any other static check.
fn coerce_numeric(arena: &mut Arena, e: ExprRef, ty: &Type, span: Span) -> Result<ExprRef, TypeError> {
    match ty {
        Type::Int | Type::Float => Ok(e),
        Type::Dyn | Type::Var(_) => Ok(build_boundary_check(arena, e, &Type::Union(Rc::new(vec![Type::Int, Type::Float])))),
        other => Err(TypeError(format!("type mismatch: expected Int or Float, found {other}"), span)),
    }
}

// Builds the actual runtime check for a definitely-Dyn-origin value
// against concrete target type `to` -- shared between coerce's own
// Dyn-to-concrete crossings and wrap_fun_contract's return-type check (a
// higher-order contract's return value is ALSO always Dyn-origin, calling
// an unknown wrapped function). Infallible: by the time this runs, `to`
// is just "what runtime check to build," no static consistency question
// left open (that's coerce's own job, checked before this is ever
// called).
//
// Desugars into ordinary language machinery -- `if <predicate> then value
// else fail(msg)` -- instead of a dedicated Check AST node: Int/Bool/Str/
// List get a single builtin predicate call (is_int, etc.); Fun gets a
// real per-call contract (wrap_fun_contract, since a bare callability tag
// can't see inside a closure -- "callable" isn't "callable with this
// exact signature"); Tuple/Union get their own shape predicates below.
// Every predicate call (including Fun's nested contract) panics with the
// same "type error: expected X, found Y" text, built at RUNTIME via the
// type_name builtin since the actual mismatched value's type isn't known
// until then.
fn build_boundary_check(arena: &mut Arena, e: ExprRef, to: &Type) -> ExprRef {
    match to {
        Type::Int => build_shallow_check(arena, e, to, "is_int"),
        Type::Float => build_shallow_check(arena, e, to, "is_float"),
        Type::Bool => build_shallow_check(arena, e, to, "is_bool"),
        Type::Str => build_shallow_check(arena, e, to, "is_str"),
        Type::List(_) => build_shallow_check(arena, e, to, "is_list"),
        Type::Fun(param_ty, _row, ret_ty) => wrap_fun_contract(arena, e, param_ty.clone(), ret_ty.clone()),
        Type::Token(id) => build_token_check(arena, e, to, *id),
        Type::Tuple(_) => build_shape_check(arena, e, to),
        Type::Record(_) => build_shape_check(arena, e, to),
        Type::Union(_) => build_union_check(arena, e, to),
        Type::Dyn => unreachable!("coerce only calls this once *to != Type::Dyn is already established"),
        Type::Var(_) => unreachable!("coerce only calls this once *to != Type::Var is already established"),
    }
}

// `let __check_tmp = e in if <alt1-shape> then <alt1's OWN full check>
// else if <alt2-shape> then <alt2's OWN full check> else ... else
// fail(...)` -- tries each alternative's bare shape predicate
// (build_shape_predicate) in turn, and only once ONE of them matches,
// runs that SPECIFIC alternative's full build_boundary_check (not just
// the shape test again) -- so a Fun alternative gets its real per-call
// contract (wrap_fun_contract), the same rigor that type would get as a
// plain (non-union) annotation, not just "shaped like something
// callable." Can't simply call build_boundary_check per alternative and
// fall through on its own failure -- its fail() would abort the whole
// check on the first non-matching alternative instead of trying the next
// one -- so the shape predicate decides WHICH alternative's full check to
// run, and that full check (redundantly, but harmlessly) re-confirms the
// same shape on its way to the real work.
fn build_union_check(arena: &mut Arena, e: ExprRef, to: &Type) -> ExprRef {
    let alts: Rc<Vec<Type>> = match to {
        Type::Union(alts) => alts.clone(),
        _ => unreachable!("build_union_check is only ever called with a Union target"),
    };
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let mut result = build_fail_call(arena, to, tmp_ref);
    for alt in alts.iter().rev() {
        let pred = build_shape_predicate(arena, tmp_ref, alt);
        let checked = build_boundary_check(arena, tmp_ref, alt);
        result = arena.push(Expr::If(pred, checked, result));
    }
    arena.push(Expr::Let(tmp, None, e, result))
}

// The bare boolean half of build_boundary_check's per-type dispatch --
// "does `value_ref` shallowly look like `ty`," with no let-binding and no
// fail() of its own, so build_union_check can OR several of these
// together before deciding anything. Mirrors build_boundary_check's own
// arms exactly (same shallow-check precedent each one sets), just
// stopping short of wrapping the result in Let/If/fail.
fn build_shape_predicate(arena: &mut Arena, value_ref: ExprRef, ty: &Type) -> ExprRef {
    match ty {
        Type::Dyn => arena.push(Expr::Bool(true)),
        Type::Var(_) => arena.push(Expr::Bool(true)),
        Type::Int => build_predicate_call(arena, "is_int", value_ref),
        Type::Float => build_predicate_call(arena, "is_float", value_ref),
        Type::Bool => build_predicate_call(arena, "is_bool", value_ref),
        Type::Str => build_predicate_call(arena, "is_str", value_ref),
        Type::List(_) => build_predicate_call(arena, "is_list", value_ref),
        Type::Fun(..) => build_predicate_call(arena, "is_fun", value_ref),
        Type::Token(id) => {
            let lit = arena.push(Expr::Token(*id));
            arena.push(Expr::BinOp(BinOp::Eq, value_ref, lit))
        }
        Type::Tuple(items) => {
            let is_list = build_predicate_call(arena, "is_list", value_ref);
            let len_var = arena.push(Expr::Var("len".to_string()));
            let len_call = arena.push(Expr::App(len_var, value_ref));
            let arity_lit = arena.push(Expr::Int(items.len() as i64));
            let len_eq = arena.push(Expr::BinOp(BinOp::Eq, len_call, arity_lit));
            let false_lit = arena.push(Expr::Bool(false));
            arena.push(Expr::If(is_list, len_eq, false_lit))
        }
        // Width-tolerant, unlike Tuple's exact arity check above: `v` need
        // only HAVE (at least) every required field -- extra fields are
        // fine, see Pattern::Record's own doc comment for why nothing
        // downstream can ever observe them. `is_record(v) && has_field(v,
        // "x") && has_field(v, "y") && ...`, one clause per required
        // field name, BUT `is_record` has to be the OUTERMOST/first-
        // evaluated check, same as `is_list` is for Tuple just above --
        // `has_field` panics on a non-Record argument (see its own doc
        // comment), so building the has_field chain first and only
        // wrapping it in `is_record` at the very end (rather than
        // starting the fold from `is_record` and nesting has_field
        // AROUND it) would let has_field run on a value that was never
        // confirmed to be a Record at all.
        Type::Record(fields) => {
            let has_all = fold_predicate(arena, fields, FoldOp::And, |arena, (name, _)| {
                build_str_call(arena, "has_field", value_ref, name)
            });
            let is_record = build_predicate_call(arena, "is_record", value_ref);
            let false_lit = arena.push(Expr::Bool(false));
            arena.push(Expr::If(is_record, has_all, false_lit))
        }
        Type::Union(alts) => {
            fold_predicate(arena, alts, FoldOp::Or, |arena, alt| build_shape_predicate(arena, value_ref, alt))
        }
    }
}

// Whether fold_predicate combines its items with AND (every one must
// hold) or OR (any one is enough).
enum FoldOp {
    And,
    Or,
}

// Folds `items` right-to-left into a short-circuiting AND/OR of
// `pred(item)`, built from ordinary If/Bool nodes -- shared by Union's
// own "is it consistent with ANY alternative" fold above and Record's
// own "does it have EVERY required field" fold just above that, which
// differ only in which of AND/OR they each need.
fn fold_predicate<T>(
    arena: &mut Arena,
    items: &[T],
    op: FoldOp,
    mut pred: impl FnMut(&mut Arena, &T) -> ExprRef,
) -> ExprRef {
    let mut result = arena.push(Expr::Bool(matches!(op, FoldOp::And)));
    for item in items.iter().rev() {
        let p = pred(arena, item);
        result = match op {
            FoldOp::And => {
                let false_lit = arena.push(Expr::Bool(false));
                arena.push(Expr::If(p, result, false_lit))
            }
            FoldOp::Or => {
                let true_lit = arena.push(Expr::Bool(true));
                arena.push(Expr::If(p, true_lit, result))
            }
        };
    }
    result
}

fn build_predicate_call(arena: &mut Arena, name: &str, value_ref: ExprRef) -> ExprRef {
    let v = arena.push(Expr::Var(name.to_string()));
    arena.push(Expr::App(v, value_ref))
}

// `builtin(value_ref, str_arg)` -- the two-argument counterpart to
// build_predicate_call, shared by Record's own has_field-based shape
// check above and `.field` access's get_field desugaring below.
fn build_str_call(arena: &mut Arena, builtin: &str, value_ref: ExprRef, str_arg: &str) -> ExprRef {
    let f = arena.push(Expr::Var(builtin.to_string()));
    let applied = arena.push(Expr::App(f, value_ref));
    let arg_lit = arena.push(Expr::Str(str_arg.to_string()));
    arena.push(Expr::App(applied, arg_lit))
}

// `let __check_tmp = e in if <cond(__check_tmp)> then __check_tmp else
// fail("type error: expected {to}, found " ++ type_name(__check_tmp))` --
// binding `e` once (not re-evaluating it for both the condition and the
// passed-through result) is what makes this safe for an `e` with side
// effects (an arbitrary expression, not necessarily a bare variable).
// Shared skeleton behind build_token_check/build_shape_check/
// build_shallow_check below -- each supplies its own `cond`, differing
// only in what "looks like `to`" actually means for that kind of type.
fn build_checked(arena: &mut Arena, e: ExprRef, to: &Type, cond: impl FnOnce(&mut Arena, ExprRef) -> ExprRef) -> ExprRef {
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let pred = cond(arena, tmp_ref);
    let fail_call = build_fail_call(arena, to, tmp_ref);
    let if_expr = arena.push(Expr::If(pred, tmp_ref, fail_call));
    arena.push(Expr::Let(tmp, None, e, if_expr))
}

// A singleton: the only thing a Dyn-sourced value could ever satisfy this
// against is the EXACT same Value::Token, so this reuses ordinary `==`
// (BinOp::Eq, backed by machine::value_eq's own Token arm) against a
// freshly-embedded literal of the same id, rather than a dedicated
// predicate builtin.
fn build_token_check(arena: &mut Arena, e: ExprRef, to: &Type, id: u64) -> ExprRef {
    build_checked(arena, e, to, |arena, v| {
        let literal = arena.push(Expr::Token(id));
        arena.push(Expr::BinOp(BinOp::Eq, v, literal))
    })
}

// Shallow: confirms shape (arity for Tuple; field PRESENCE, width-
// tolerantly, for Record -- see build_shape_predicate's own Record arm),
// not that each position/field's own value matches ITS OWN element type.
// A full recursive check (extract each element, apply
// build_boundary_check to it, rebuild the tuple/record) is possible but
// not built yet; this matches the same "confirm the shape, not deeper"
// precedent every other Dyn boundary check here already sets. Entirely
// generic over `to` -- the actual shape test is build_shape_predicate's
// own job, not re-derived here, so this one function serves both
// Type::Tuple and Type::Record with nothing Tuple/Record-specific of its
// own.
fn build_shape_check(arena: &mut Arena, e: ExprRef, to: &Type) -> ExprRef {
    build_checked(arena, e, to, |arena, v| build_shape_predicate(arena, v, to))
}

// `predicate` is looked up by name through the ordinary prelude Env
// (`is_int`/`is_bool`/`is_str`/`is_list`/`is_fun`), the same way
// `fail`/`type_name` already are -- see their own doc comments on the
// (pre-existing, accepted) shadowing risk that implies.
fn build_shallow_check(arena: &mut Arena, e: ExprRef, to: &Type, predicate: &str) -> ExprRef {
    build_checked(arena, e, to, |arena, v| build_predicate_call(arena, predicate, v))
}

// `fail("type error: expected {to}, found " ++ type_name(value_ref))` --
// shared by every check builder above, so the message format (and the
// exact wording every existing test asserts on) lives in exactly one
// place.
fn build_fail_call(arena: &mut Arena, to: &Type, value_ref: ExprRef) -> ExprRef {
    let prefix = arena.push(Expr::Str(format!("type error: expected {to}, found ")));
    let type_name_var = arena.push(Expr::Var("type_name".to_string()));
    let type_name_call = arena.push(Expr::App(type_name_var, value_ref));
    let msg = arena.push(Expr::BinOp(BinOp::Concat, prefix, type_name_call));
    let fail_var = arena.push(Expr::Var("fail".to_string()));
    arena.push(Expr::App(fail_var, msg))
}

// Wraps a Dyn-origin value in a fresh closure that, on every call: checks
// the argument matches param_ty (via the wrapper's own declared param type
// -- ordinary App-site coercion at the wrapper's call sites handles that),
// confirms the wrapped value is actually callable, applies it, then checks
// the result against ret_ty (via build_boundary_check, recursively -- a
// higher-order function returning another Tuple/Union/Fun value gets that
// same value's own proper check, not just a shallow tag test). This is a
// real higher-order contract (each call re-validated), not a one-time tag
// check -- a value that merely looks like a function can't smuggle a
// wrong return type through it. Built entirely from existing Expr nodes
// (Let/Lambda/If/App/Var/BinOp), no new Value representation needed. No
// span bookkeeping here: these nodes are synthesized, not sourced from
// the program text, and typecheck never looks up a span for them (see
// TypeError's doc comment).
fn wrap_fun_contract(arena: &mut Arena, e: ExprRef, param_ty: Rc<Type>, ret_ty: Rc<Type>) -> ExprRef {
    let fn_var = "__contract_fn".to_string();
    let arg_var = "__contract_arg".to_string();
    let fn_var_ref = arena.push(Expr::Var(fn_var.clone()));
    let arg_var_ref = arena.push(Expr::Var(arg_var.clone()));
    let checked_fn = build_shallow_check(arena, fn_var_ref, &any_fun(), "is_fun");
    let call = arena.push(Expr::App(checked_fn, arg_var_ref));
    let checked_call = build_boundary_check(arena, call, &ret_ty);
    let lambda = arena.push(Expr::Lambda(arg_var, Some((*param_ty).clone()), checked_call));
    arena.push(Expr::Let(fn_var, None, e, lambda))
}

// A `let`/`fun` prefix collected while flattening a chain of them (see
// `elaborate`) -- deferred until the terminal body is elaborated, then
// folded back into nested Let/Lambda nodes (and their types/rows) in
// reverse, in the exact shape their original per-node match arms produced.
enum PendingElab {
    Let { var: String, bound_ty: Type, val_row: EffectRow, val: ExprRef },
    // One `let rec` group's elaborated bindings (each carrying its own
    // inferred/annotated type, row, and elaborated value), folding back
    // into a single Expr::LetRec.
    LetRec { bindings: Vec<(String, Type, EffectRow, ExprRef)> },
    Fun { param: String, param_ty: Type },
}

// Recognizes the handler-expression shapes this codebase actually
// produces (MakeHandler, optionally wrapped in deep(...)/shallow(...))
// well enough to know which effect name a `handle` discharges. Anything
// else (a bare variable, a computed handler) returns None -- Handle then
// conservatively does NOT subtract anything from the body's row, which is
// the sound direction to fail in: at worst it over-reports an effect as
// possibly-unhandled, never hides a real one.
fn discharged_effect(arena: &Arena, handler: ExprRef) -> Option<String> {
    match &arena[handler] {
        Expr::MakeHandler { effect, .. } => Some(effect.clone()),
        Expr::App(f, arg) => match &arena[*f] {
            Expr::Var(name) if name == "deep" || name == "shallow" => discharged_effect(arena, *arg),
            _ => None,
        },
        _ => None,
    }
}

// The shallow type a pattern shape implies, used only to statically reject
// a scrutinee that could never match it (e.g. an Int pattern against a
// Bool scrutinee) -- not full pattern-based type refinement. Var matches
// anything, so it's Dyn; a plain List/Cons pattern doesn't know the
// element type from the pattern alone, so Dyn there too (this is also
// what lets a tuple pattern match a Tuple-typed scrutinee -- List(Dyn) is
// consistent with any Tuple, see types::consistent's own bridging arm).
fn pattern_type(pat: &Pattern) -> Type {
    match pat {
        Pattern::Var(_) => Type::Dyn,
        Pattern::Int(_) => Type::Int,
        Pattern::Bool(_) => Type::Bool,
        Pattern::Str(_) => Type::Str,
        Pattern::List(_) | Pattern::Cons(..) => Type::List(Rc::new(Type::Dyn)),
        // Dyn -- but unlike every other arm above, this is NOT what
        // decides whether a Record pattern can match a given scrutinee
        // (Dyn is trivially consistent with anything, which would make
        // EVERY scrutinee type "possibly matchable," silently losing the
        // same impossible-pattern diagnostic every other arm here
        // provides). This is only the DISPLAY type used in that
        // diagnostic's own message text -- see pattern_could_match, the
        // function that actually decides Record's case, right below.
        Pattern::Record(_) => Type::Dyn,
    }
}

// True for a Type::Record, or a Type::Union whose every alternative is
// itself record_shaped -- the set of static types `.field` access (see
// Expr::FieldAccess's own doc comment) can meaningfully ask a field of.
// Anything else (Int, Tuple, Fun, a Union with a non-record alternative,
// ...) is definitely not, and gets the "expected a record" error instead
// of "no field named" -- the two are worth telling apart even though
// both are just `.field` being rejected, the same way a Fun-annotation
// mismatch and a wrong-argument-type mismatch are worded differently.
fn record_shaped(ty: &Type) -> bool {
    match ty {
        Type::Record(_) => true,
        Type::Union(alts) => alts.iter().all(record_shaped),
        _ => false,
    }
}

// The type `.field` access on a record_shaped `ty` returns, if `ty`
// definitely HAS this field -- for a Union, only when EVERY alternative
// has it (recursing, so a Union of Unions works too), since a value of
// that type could be ANY alternative at runtime. Field types across
// alternatives widen to Dyn on disagreement, the same "no union types,
// pick Dyn instead" rule If's branches/ListLit's elements/Match's arms
// already use elsewhere in this file. None means the field is missing
// from at least one alternative (or, for a bare Record, missing
// entirely) -- record_shaped's own caller already ruled out "not even
// shaped like a record" separately, so this only ever needs to answer
// the narrower "does it have GOT this one" question.
fn record_field_type(ty: &Type, name: &str) -> Option<Type> {
    match ty {
        Type::Record(fields) => find_field(fields, name).cloned(),
        Type::Union(alts) => {
            let mut result: Option<Type> = None;
            for alt in alts.iter() {
                let t = record_field_type(alt, name)?;
                result = Some(match &result {
                    Some(prev) if *prev == t => t,
                    Some(_) => Type::Dyn,
                    None => t,
                });
            }
            result
        }
        _ => None,
    }
}

// Does `pat` have ANY chance of matching a value of type `ty`? For every
// pattern shape except Record this is exactly `consistent(ty,
// &pattern_type(pat))` -- but a plain consistent() check is too coarse
// for Record specifically, now that matching is width-tolerant: a
// pattern naming only SOME of a Record type's fields must still be
// considered a possible match (that's the whole point of width
// subtyping), which consistent()'s own exact-field-set Record arm can't
// express (it would wrongly reject `{x: a}` against a scrutinee typed
// `{x: Int, y: Int}` as impossible, and giving Pattern::Record a plain
// Type::Dyn phantom type -- see pattern_type's own Record arm -- would
// swing too far the OTHER way and never reject anything, even a Record
// pattern against a manifestly non-record scrutinee like Str). So Record
// gets its own real check here: every field the pattern names must
// exist in the scrutinee type (recursing into a Union's alternatives,
// any one of which might supply it), everything else falls back to the
// ordinary consistent()-based question.
fn pattern_could_match(pat: &Pattern, ty: &Type) -> bool {
    match (pat, ty) {
        (Pattern::Record(fields), Type::Record(type_fields)) => {
            fields.iter().all(|(name, _)| find_field(type_fields, name).is_some())
        }
        (Pattern::Record(_), Type::Union(alts)) => alts.iter().any(|alt| pattern_could_match(pat, alt)),
        (Pattern::Record(_), Type::Dyn | Type::Var(_)) => true,
        (Pattern::Record(_), _) => false,
        // Same story as Record just above, for the same reason: a fixed-
        // length list pattern's own arity is real, provable information
        // pattern_type's blind List(Dyn) throws away, and consistent()'s
        // List/Tuple bridge has to allow ANY length through precisely
        // because it can't see it -- so a genuinely mismatched-arity
        // tuple pattern needs to be caught here, before that bridge ever
        // runs. Type::List scrutinees still fall through to the generic
        // consistent()-based catch-all below, correctly: an actual List
        // type has no fixed arity to check a pattern's length against.
        (Pattern::List(subpats), Type::Tuple(items)) => subpats.len() == items.len(),
        (Pattern::List(_), Type::Union(alts)) => alts.iter().any(|alt| pattern_could_match(pat, alt)),
        (Pattern::List(_), Type::Dyn | Type::Var(_)) => true,
        _ => consistent(ty, &pattern_type(pat)),
    }
}

// Extends `ctx` with every Var this pattern binds, correlated with
// `scrutinee_ty` where that's informative: a Cons/List pattern's element
// bindings unify against the scrutinee's own (possibly still-open, via a
// fresh Type::Var) element type, instead of always Dyn -- this is the
// FIRST of the two problems full parametric polymorphism exists to
// solve (see the design spec's own Motivation). Fallible now (it wasn't
// before): a genuinely impossible correlation (e.g. unifying two
// concretely-different, incompatible shapes) is a real static error, the
// same class of mistake `pattern_could_match`'s own existing rejection
// already catches for the cases IT can see -- this rewrite doesn't
// replace that check, it adds a second, complementary one that can see
// INTO a pattern's own sub-bindings, not just its top-level shape.
fn bind_pattern_vars(ctx: &Ctx, pat: &Pattern, scrutinee_ty: &Type, infer: &mut InferCtx, span: Span) -> Result<Ctx, TypeError> {
    match pat {
        // Resolve before storing, not just clone: by the time a Pattern::Var
        // binds, an enclosing Cons/List/Record arm may have ALREADY unified
        // scrutinee_ty's own Type::Var against something concrete (e.g.
        // Cons's own unify(scrutinee_ty, List(elem_ty)) runs before it
        // recurses into `head`) -- storing the raw, unresolved Type::Var
        // reference would hide that from every later consumer that reads
        // straight from ctx without a chance to resolve it first
        // (coerce_numeric and Expr::FieldAccess, per Task 2's own parity
        // fixes, both treat ANY Type::Var as permissively as Type::Dyn,
        // with no InferCtx of their own to tell an already-resolved one
        // apart from a still-open one). resolve_deep is a safe no-op when
        // scrutinee_ty is still genuinely unconstrained (e.g. my_map's own
        // unannotated xs -- nothing to resolve yet) and only ever changes
        // the stored type when something upstream has ALREADY pinned it
        // down.
        Pattern::Var(name) => Ok(extend(ctx, name, infer.resolve_deep(scrutinee_ty))),
        Pattern::Int(_) | Pattern::Bool(_) | Pattern::Str(_) => Ok(ctx.clone()),
        // A fixed-length List pattern is really a Tuple-shaped correlation
        // (each position its own type), not a single shared element type
        // -- so, like Record just below, figure out each position's REAL
        // type from scrutinee_ty FIRST, then recurse using that type
        // directly. Order matters here: an earlier draft minted an
        // independent fresh var per position, recursed into it
        // immediately, and only unified against the real scrutinee shape
        // AFTER the loop -- for a NESTED List/Tuple sub-pattern, that
        // recursive call would see its own scrutinee_ty as a totally
        // unconstrained Type::Var and eagerly bind it to a generic
        // List(fresh) shape (via this same function's own Type::Var arm
        // below) before the outer call ever got a chance to instead
        // correlate that position with, say, a concrete nested Tuple --
        // producing a spurious List-vs-Tuple unification failure for
        // exactly the "num-position, no bare Var" nested-tuple pattern
        // this design is supposed to handle. Determining the real
        // per-position type before recursing avoids the fresh var ever
        // being eagerly bound to the wrong shape.
        Pattern::List(pats) => match scrutinee_ty {
            Type::Tuple(items) if items.len() == pats.len() => {
                let mut c = ctx.clone();
                for (p, item_ty) in pats.iter().zip(items.iter()) {
                    c = bind_pattern_vars(&c, p, item_ty, infer, span)?;
                }
                Ok(c)
            }
            Type::List(shared_elem) => {
                let mut c = ctx.clone();
                for p in pats {
                    c = bind_pattern_vars(&c, p, shared_elem, infer, span)?;
                }
                Ok(c)
            }
            Type::Var(_) => {
                // Still fully unconstrained: force scrutinee_ty into a
                // List(fresh shared element) shape via unify -- same
                // choice Cons makes just below, at the cost of not being
                // able to give each position its own distinct type the
                // way an already-known Tuple scrutinee can (there's
                // nothing here yet to tell List and Tuple apart).
                let elem_ty = infer.fresh_var("elem");
                let list_shape = Type::List(Rc::new(elem_ty.clone()));
                unify(scrutinee_ty, &list_shape, infer, span)?;
                let mut c = ctx.clone();
                for p in pats {
                    c = bind_pattern_vars(&c, p, &elem_ty, infer, span)?;
                }
                Ok(c)
            }
            // A concrete, non-Tuple, non-List scrutinee against a List
            // pattern: pattern_could_match's own existing check (called
            // by Expr::Match before this function ever runs) already
            // rejects this case statically -- bind each position to its
            // own independent fresh var (nothing real to correlate
            // against) so nested sub-bindings still work.
            _ => {
                let mut c = ctx.clone();
                for p in pats {
                    let elem_ty = infer.fresh_var("elem");
                    c = bind_pattern_vars(&c, p, &elem_ty, infer, span)?;
                }
                Ok(c)
            }
        },
        Pattern::Cons(head, tail) => {
            let elem_ty = infer.fresh_var("elem");
            let list_shape = Type::List(Rc::new(elem_ty.clone()));
            unify(scrutinee_ty, &list_shape, infer, span)?;
            let c = bind_pattern_vars(ctx, head, &elem_ty, infer, span)?;
            bind_pattern_vars(&c, tail, &list_shape, infer, span)
        }
        Pattern::Record(fields) => {
            let mut c = ctx.clone();
            for (name, p) in fields {
                let field_ty = match scrutinee_ty {
                    Type::Record(type_fields) => find_field(type_fields, name).cloned().unwrap_or_else(|| infer.fresh_var(name)),
                    _ => infer.fresh_var(name),
                };
                c = bind_pattern_vars(&c, p, &field_ty, infer, span)?;
            }
            Ok(c)
        }
    }
}

// Reachability: does `earlier` already cover every value `later` could
// ever match, making `later` dead code if it comes after `earlier`? Only
// two cheaply-provable shapes are recognized, the same "prove what's
// cheap, stay silent otherwise" stance as missing_case (there's no
// runtime signal for an unreachable arm to fall back on -- it just
// silently never runs, so this is the only place that can ever catch it):
//   - Var (including "_") dominates everything -- it matches any value.
//   - a literal (Int/Bool/Str) dominates an identical literal -- an exact
//     duplicate can never be reached, match_pattern already took the
//     earlier one.
// List/Cons patterns are NOT compared for subsumption here (e.g. an
// earlier unconstrained `h :: t` making a later `1 :: t` unreachable) --
// a real but more nuanced case than this function attempts; a redundant
// arm of that shape is silently allowed, same as any other question this
// checker can't cheaply answer.
//
// Record IS handled, though, unlike List/Cons -- width-tolerant matching
// (see Pattern::Record's own doc comment) makes this cheaply provable in
// a way List/Cons's open-ended lengths aren't: an earlier pattern
// dominates a later one iff every field name the earlier pattern needs
// is also named by the later one (so any value matching `later` already
// has everything `earlier` needs too), and each such shared field's own
// sub-pattern is dominated in turn. `{x: a}` dominates `{x: a, y: b}`
// this way -- the second arm can never run.
fn dominates(earlier: &Pattern, later: &Pattern) -> bool {
    match earlier {
        Pattern::Var(_) => true,
        Pattern::Int(n) => matches!(later, Pattern::Int(m) if m == n),
        Pattern::Bool(b) => matches!(later, Pattern::Bool(c) if c == b),
        Pattern::Str(s) => matches!(later, Pattern::Str(t) if t == s),
        Pattern::Record(efields) => match later {
            Pattern::Record(lfields) => {
                efields.iter().all(|(name, esub)| find_field(lfields, name).is_some_and(|lsub| dominates(esub, lsub)))
            }
            _ => false,
        },
        _ => false,
    }
}

// The index of the first arm some EARLIER arm already fully dominates, if
// any -- reported as unreachable (dead code). `has_guard[j]` gates whether
// arm j can dominate anything: a GUARDED earlier arm might not fire (its
// guard could reject), so it covering arm i's pattern doesn't guarantee
// arm i never runs -- only an unguarded earlier arm's coverage is a
// guarantee. An earlier arm's own guard therefore makes it transparent to
// this check, never a later arm's.
fn first_unreachable(patterns: &[&Pattern], has_guard: &[bool]) -> Option<usize> {
    (0..patterns.len()).find(|&i| (0..i).any(|j| !has_guard[j] && dominates(patterns[j], patterns[i])))
}

// Does `pat` cover every possible value at a position statically known to
// have type `ty`? Only ever called with `ty` a Tuple or Record
// (missing_case's own guard, including per-alternative for a Union), but
// recurses into NESTED tuple/record positions too: `((p, q), x)` covers
// all of `((Int, Int), Int)` even though NEITHER top-level sub-pattern is
// a bare Var, because the first one is itself a fully-covering pattern
// for ITS OWN (also Tuple) position. Does NOT help a scrutinee bound by
// an outer match/lambda pattern first (renno has no pattern-driven type
// refinement -- see bind_pattern_vars's own doc comment -- so that
// binding's own type is just Dyn, not the precise Tuple/Record type this
// needs): a fully-nested pattern in ONE match sidesteps that, a match on
// a separately-destructured intermediate variable does not.
//
// Record's own arm is genuinely simpler than Tuple's, not just a variant
// of it: since Pattern::Record matching is width-tolerant (extra fields
// on the VALUE are never looked at -- see its own doc comment), a
// pattern covers a Record TYPE as soon as every field the pattern names
// exists in the type with a covering sub-pattern -- no arity match
// required at all. `{x: a}` alone is exhaustive for `{x: Int}` AND for
// `{x: Int, y: Int}` AND for any other Record type that happens to
// include field `x`.
fn covers_tuple_position(pat: &Pattern, ty: &Type) -> bool {
    match (pat, ty) {
        (Pattern::Var(_), _) => true,
        (Pattern::List(subpats), Type::Tuple(items)) if subpats.len() == items.len() => {
            subpats.iter().zip(items.iter()).all(|(sp, t)| covers_tuple_position(sp, t))
        }
        (Pattern::Record(pat_fields), Type::Record(type_fields)) => pat_fields
            .iter()
            .all(|(name, sp)| find_field(type_fields, name).is_some_and(|t| covers_tuple_position(sp, t))),
        _ => false,
    }
}

// Exhaustiveness is checked only where it's cheaply provable -- same stance
// as everywhere else in this checker (effect rows, list-length arity):
// stay silent and defer to match_pattern's own runtime panic rather than
// try to enumerate an open domain (Int and Str literals, in particular,
// never close). Four shapes are recognized as covering every possible
// value:
//   - any Var (including "_") pattern present, anywhere -- matches
//     everything by itself.
//   - Tuple: a fixed-length, all-Var/wildcard (or recursively
//     fully-covering) pattern of exactly the scrutinee's own known arity
//     (only meaningful when `scrut_ty` IS a Tuple -- its length is fixed
//     and known, unlike a general List's).
//   - Union: every alternative has SOME pattern (not necessarily the
//     same one) that fully covers it -- independent of Tuple's own
//     single-pattern rule, since a Union's alternatives are independent
//     shapes.
//   - Bool: both `true` and `false` literals present.
//   - List: both `[]` and an unconstrained `h :: t` present (`h`/`t`
//     themselves must be Var -- `1 :: t` only covers SOME non-empty
//     lists, not all of them).
// Anything else is reported as possibly non-exhaustive: Int/Str literals
// with no catch-all, or a List match using only fixed-length patterns.
fn missing_case(patterns: &[&Pattern], scrut_ty: &Type) -> Option<String> {
    if patterns.iter().copied().any(|p| matches!(p, Pattern::Var(_))) {
        return None;
    }

    // A Tuple's arity is fixed and known (unlike a general List, which
    // could be any length) -- a fixed-length pattern of exactly that
    // arity, whose every position either binds a Var (matches anything
    // there) or is ITSELF a fully-covering nested tuple pattern for that
    // position's own Tuple-typed field (covers_tuple_position recurses),
    // covers every possible value -- the same way a bare Var arm does for
    // anything else. Nested-but-not-fully-covering (e.g. a literal at
    // some position) still falls through to "possibly non-exhaustive,"
    // same as everywhere else this checker declines to enumerate.
    // A Record's arity is fixed and known too -- same story as Tuple,
    // just also carrying field names that (per covers_tuple_position's
    // own doc comment) don't matter for this particular check.
    if let Type::Tuple(_) | Type::Record(_) = scrut_ty {
        let covers_every_tuple = patterns.iter().copied().any(|p| covers_tuple_position(p, scrut_ty));
        if covers_every_tuple {
            return None;
        }
    }

    // A Union's alternatives are independent -- unlike Tuple, no SINGLE
    // pattern needs to cover the whole thing; exhaustive means every
    // alternative has SOME pattern (not necessarily the same one) that
    // fully covers it. `type Pair = (Int,) | (Int, Int) in ... | (n,) ->
    // ... | (n, m) -> ...` is exhaustive this way even though neither arm
    // alone would satisfy Tuple's own single-pattern check.
    if let Type::Union(alts) = scrut_ty {
        let covers_every_alt = alts.iter().all(|alt| patterns.iter().copied().any(|p| covers_tuple_position(p, alt)));
        if covers_every_alt {
            return None;
        }
    }

    let has_true = patterns.iter().copied().any(|p| matches!(p, Pattern::Bool(true)));
    let has_false = patterns.iter().copied().any(|p| matches!(p, Pattern::Bool(false)));
    if has_true && has_false {
        return None;
    }

    let has_nil = patterns.iter().copied().any(|p| matches!(p, Pattern::List(items) if items.is_empty()));
    let has_full_cons = patterns.iter().copied().any(
        |p| matches!(p, Pattern::Cons(h, t) if matches!(**h, Pattern::Var(_)) && matches!(**t, Pattern::Var(_))),
    );
    if has_nil && has_full_cons {
        return None;
    }

    Some(if has_true || has_false {
        "the missing Bool case (add `true`/`false`, or a wildcard `_` arm)".to_string()
    } else if has_nil || has_full_cons {
        "the missing List case (add `[]`/`h :: t`, or a wildcard `_` arm)".to_string()
    } else {
        "some value not covered by any arm (add a wildcard `_` arm)".to_string()
    })
}

// If `val` is (a curried chain of) Lambda ending in some body, elaborates
// it with each passthrough-generalizable unannotated parameter bound to a
// fresh Type::Var instead of Type::Dyn. This peels the SAME Lambda chain
// passthrough_generalizable_params already walked, binding each param's
// name to its (possibly generalized) type and recording it, then
// elaborates the final (post-Lambda) body once against that context and
// wraps the result back up in Fun/Lambda layers itself -- the same
// PendingElab::Fun reconstruction `elaborate`'s own peeling loop does.
// This can't simply re-bind ctx and fall through to a plain recursive
// `elaborate(arena, val, ...)` call on the whole chain: that call's own
// peeling loop would re-walk these SAME (still-unannotated in the AST)
// Lambda nodes and re-extend each name to Type::Dyn again, which --
// PList::get's own most-recent-binding-wins semantics -- would shadow the
// fresh Type::Var this function just bound, silently losing it. Returns
// None (falls back to plain `elaborate`) whenever `val` isn't a Lambda at
// all, or none of its parameters qualify -- so this is always a strict
// superset of today's behavior, never a change to it.
//
// Also hands back the exact fresh Type::Var names this call introduced (one
// per generalizable parameter), so the caller can generalize the OUTER
// binding over exactly those names via extend_generalized_with_type_vars --
// never via extend_generalized's own auto-derivation, which would wrongly
// also pick up any OTHER Type::Var merely encountered inside `val`'s body
// (e.g. a still-open enclosing parameter's own name, reached through a
// nested ordinary `let` alias). See extend_generalized's doc comment.
//
// The Vec<String> is the fresh Type::Var names this call introduced --
// named as its own alias (clippy's own type_complexity threshold) purely
// to keep this signature readable, not a semantically meaningful type.
type GeneralizingPassthroughResult = Option<Result<(Type, EffectRow, ExprRef, Vec<String>), TypeError>>;

fn elaborate_generalizing_passthrough(
    arena: &mut Arena,
    val: ExprRef,
    ctx: &Ctx,
    spans: &SpanMap,
    infer: &mut InferCtx,
) -> GeneralizingPassthroughResult {
    if !matches!(arena[val], Expr::Lambda(..)) {
        return None;
    }
    let generalizable = passthrough_generalizable_params(arena, val);
    if !generalizable.iter().any(|g| *g) {
        return None;
    }
    let mut cur_ctx = ctx.clone();
    let mut cur = val;
    let mut frames: Vec<(String, Type)> = Vec::new();
    let mut fresh_type_vars: Vec<String> = Vec::new();
    for is_generalizable in generalizable {
        match &arena[cur] {
            Expr::Lambda(param, ann, body) => {
                let param_ty = if is_generalizable {
                    let fresh_name = fresh_type_name(param);
                    fresh_type_vars.push(fresh_name.clone());
                    Type::Var(fresh_name)
                } else {
                    ann.clone().unwrap_or(Type::Dyn)
                };
                cur_ctx = extend(&cur_ctx, param, param_ty.clone());
                frames.push((param.clone(), param_ty));
                cur = *body;
            }
            _ => unreachable!("generalizable.len() matches the Lambda chain passthrough_generalizable_params walked"),
        }
    }
    Some(elaborate(arena, cur, &cur_ctx, spans, infer).map(|(mut result_ty, mut result_row, mut result_expr)| {
        for (param, param_ty) in frames.into_iter().rev() {
            result_ty = Type::Fun(Rc::new(param_ty.clone()), result_row, Rc::new(result_ty));
            result_row = EffectRow::pure();
            result_expr = arena.push(Expr::Lambda(param, Some(param_ty), result_expr));
        }
        (result_ty, result_row, result_expr, fresh_type_vars)
    }))
}

// Bidirectional-lite synthesis: walks the tree once, producing the
// inferred Type, the inferred EffectRow (closed, no polymorphism -- just
// the union of effect names this expression's evaluation might perform),
// and an elaborated Expr (same shape, with boundary checks -- or, at Fun
// boundaries, full contracts -- spliced into the arena at Dyn-to-concrete
// crossings; unchanged leaves are returned by their original ExprRef,
// nothing new allocated for them).
//
// Row inference is deliberately limited: it doesn't look inside a
// MakeHandler clause body at all (typing what a handler does when it
// resumes is a genuinely subtler question -- Koka/Frank treat it as its
// own effect scope -- out of scope here), and it falls back to
// EffectRow::Dyn wherever a value's own type is Dyn (an unannotated
// function, an unrecognized handler expression). Within those limits it's
// sound: check() below rejects a program only when it can prove an effect
// is never discharged, and stays silent (deferring to today's runtime
// panic) whenever it can't prove either way.
fn elaborate(arena: &mut Arena, expr: ExprRef, ctx: &Ctx, spans: &SpanMap, infer: &mut InferCtx) -> Result<(Type, EffectRow, ExprRef), TypeError> {
    // Peels off a run of leading `let`/`fun` prefixes iteratively -- a
    // match per iteration, not a recursive call -- so a long chain of
    // either (sequential `let`s, or a deeply curried `fun a -> fun b ->
    // ...`) costs O(1) native stack instead of O(chain length). This is
    // the same idea, and fixes the same failure mode, as parser::atom's
    // chain flattening: a long enough chain used to overflow the stack on
    // deeply nested/generated source (see lib.rs's run_source). `val`
    // itself is elaborated via an ordinary recursive call -- it's a
    // separate subexpression, not part of this spine, so a chain nested
    // in a `let`'s VALUE position (rather than its body) isn't flattened
    // by this loop; that's a rarer pattern than the one that was observed
    // to actually overflow.
    let mut pending = Vec::new();
    let mut cur_expr = expr;
    let mut cur_ctx: Ctx = ctx.clone();
    loop {
        // Expr is Clone and, now that its fields are ExprRef (Copy)
        // instead of Rc<Expr>, cheap to clone -- this ends the borrow on
        // `arena` before the arm below needs to mutate it.
        let node = arena[cur_expr].clone();
        match node {
            Expr::Let(var, ann, val, body) => {
                let (val_ty, val_row, val2, extra_type_vars) = match elaborate_generalizing_passthrough(arena, val, &cur_ctx, spans, infer) {
                    Some(result) => result?,
                    None => {
                        let (ty, row, e) = elaborate(arena, val, &cur_ctx, spans, infer)?;
                        (ty, row, e, Vec::new())
                    }
                };
                let ann_is_none = ann.is_none();
                let (bound_ty, val3) = match ann {
                    Some(t) => (t.clone(), coerce(arena, val2, &val_ty, &t, spans[val])?),
                    None => (val_ty, val2),
                };
                let extra_type_vars = if ann_is_none { extra_type_vars } else { Vec::new() };
                cur_ctx = if extra_type_vars.is_empty() {
                    extend_generalized(&cur_ctx, &var, bound_ty.clone())
                } else {
                    extend_generalized_with_type_vars(&cur_ctx, &var, bound_ty.clone(), extra_type_vars)
                };
                pending.push(PendingElab::Let { var, bound_ty, val_row, val: val3 });
                cur_expr = body;
            }
            // Every binding needs to resolve to its own (eventual) type
            // WHILE elaborating every value in the group (not just its
            // own), so that a reference to any sibling -- including
            // itself -- type-checks precisely rather than falling back to
            // Dyn. That's only possible where a type is already known,
            // i.e. annotated -- an unannotated binding still works (that
            // one reference is just Dyn, like any other unbound-at-this-
            // point lookup), it's just not precisely typed. All values
            // are elaborated against this SAME pre-bound context (not a
            // progressively-updated one): they're simultaneous, not
            // sequential, so f seeing g's real inferred type (rather than
            // just g's annotation-or-Dyn) would wrongly depend on
            // which one happens to come first in the `and` chain.
            Expr::LetRec(bindings, body) => {
                let mut val_ctx = cur_ctx.clone();
                for (name, ann, _) in bindings.iter() {
                    val_ctx = extend(&val_ctx, name, ann.clone().unwrap_or_else(|| infer.fresh_var(name)));
                }
                let mut elaborated = Vec::with_capacity(bindings.len());
                let mut extra_type_vars_per_binding = Vec::with_capacity(bindings.len());
                for (name, ann, val) in bindings.iter() {
                    let (val_ty, val_row, val2, extra_type_vars) = match elaborate_generalizing_passthrough(arena, *val, &val_ctx, spans, infer) {
                        Some(result) => result?,
                        None => {
                            let (ty, row, e) = elaborate(arena, *val, &val_ctx, spans, infer)?;
                            (ty, row, e, Vec::new())
                        }
                    };
                    let (bound_ty, val3) = match ann {
                        Some(t) => (t.clone(), coerce(arena, val2, &val_ty, t, spans[*val])?),
                        None => (val_ty, val2),
                    };
                    extra_type_vars_per_binding.push(if ann.is_none() { extra_type_vars } else { Vec::new() });
                    elaborated.push((name.clone(), bound_ty, val_row, val3));
                }
                for ((name, bound_ty, _, _), extra_type_vars) in elaborated.iter().zip(extra_type_vars_per_binding) {
                    cur_ctx = if extra_type_vars.is_empty() {
                        extend_generalized(&cur_ctx, name, bound_ty.clone())
                    } else {
                        extend_generalized_with_type_vars(&cur_ctx, name, bound_ty.clone(), extra_type_vars)
                    };
                }
                pending.push(PendingElab::LetRec { bindings: elaborated });
                cur_expr = body;
            }
            Expr::Lambda(param, ann, body) => {
                let param_ty = ann.unwrap_or_else(|| infer.fresh_var(&param));
                cur_ctx = extend(&cur_ctx, &param, param_ty.clone());
                pending.push(PendingElab::Fun { param, param_ty });
                cur_expr = body;
            }
            _ => break,
        }
    }

    let (mut result_ty, mut result_row, mut result_expr) = elaborate_node(arena, cur_expr, &cur_ctx, spans, infer)?;

    for frame in pending.into_iter().rev() {
        match frame {
            PendingElab::Let { var, bound_ty, val_row, val } => {
                // Matches the original Let arm: body's type propagates
                // through unchanged, row is the union of val's and body's.
                result_row = EffectRow::union(&val_row, &result_row);
                result_expr = arena.push(Expr::Let(var, Some(bound_ty), val, result_expr));
            }
            PendingElab::LetRec { bindings } => {
                // Matches the original LetRec arm: body's type propagates
                // through unchanged, row is the union of every binding's
                // plus body's.
                for (_, _, val_row, _) in &bindings {
                    result_row = EffectRow::union(val_row, &result_row);
                }
                let group: Vec<(String, Option<Type>, ExprRef)> =
                    bindings.into_iter().map(|(name, ty, _, val)| (name, Some(ty), val)).collect();
                result_expr = arena.push(Expr::LetRec(Rc::new(group), result_expr));
            }
            PendingElab::Fun { param, param_ty } => {
                // Matches the original Lambda arm: the body's row is
                // embedded in the Fun type, not propagated -- evaluating
                // the Lambda expression itself is always pure.
                result_ty = Type::Fun(Rc::new(param_ty.clone()), result_row, Rc::new(result_ty));
                result_row = EffectRow::pure();
                result_expr = arena.push(Expr::Lambda(param, Some(param_ty), result_expr));
            }
        }
    }

    Ok((result_ty, result_row, result_expr))
}

// Every Expr variant except Let/Lambda, which `elaborate` peels off
// iteratively above -- reached only once no more chain prefix remains.
// `expr` (this function's own parameter) is always the ORIGINAL,
// pre-elaboration ExprRef for whatever's currently being checked, so
// `spans[expr]` is a valid, always-available "point at this whole
// construct" location for any error an arm below doesn't have a more
// specific sub-expression to blame instead.
fn elaborate_node(arena: &mut Arena, expr: ExprRef, ctx: &Ctx, spans: &SpanMap, infer: &mut InferCtx) -> Result<(Type, EffectRow, ExprRef), TypeError> {
    let node = arena[expr].clone();
    match node {
        Expr::Int(_) => Ok((Type::Int, EffectRow::pure(), expr)),
        Expr::Float(_) => Ok((Type::Float, EffectRow::pure(), expr)),
        Expr::Bool(_) => Ok((Type::Bool, EffectRow::pure(), expr)),
        Expr::Str(_) => Ok((Type::Str, EffectRow::pure(), expr)),
        // A singleton per source position -- see Expr::Token's own doc
        // comment.
        Expr::Token(id) => Ok((Type::Token(id), EffectRow::pure(), expr)),
        Expr::Var(name) => Ok((lookup(ctx, &name), EffectRow::pure(), expr)),

        // Per-position types, no widening -- unlike ListLit just below,
        // which exists for a genuinely variable-length, conceptually
        // homogeneous sequence. `(1, "a")` is `Type::Tuple([Int, Str])`,
        // not widened to `List(Dyn)`.
        Expr::Tuple(items) => {
            let mut row = EffectRow::pure();
            let mut tys = Vec::with_capacity(items.len());
            let mut refs = Vec::with_capacity(items.len());
            for item in items {
                let (item_ty, item_row, item2) = elaborate(arena, item, ctx, spans, infer)?;
                row = EffectRow::union(&row, &item_row);
                tys.push(item_ty);
                refs.push(item2);
            }
            Ok((Type::Tuple(Rc::new(tys)), row, arena.push(Expr::Tuple(refs))))
        }

        // Fields already arrive sorted by name (parser::parse_record_fields),
        // so elaborating them in the order given is elaborating (and
        // evaluating) them in canonical order, no extra reordering needed.
        // Infers the precise types::Type::Record and re-emits Expr::Record
        // with its fields elaborated -- unlike Tuple's own arm just above,
        // this is NOT rewritten into anything else: machine.rs evaluates
        // Expr::Record directly into a Value::Record (see both their own
        // doc comments for why records need this and Tuple doesn't).
        Expr::Record(fields) => {
            let mut row = EffectRow::pure();
            let mut field_tys = Vec::with_capacity(fields.len());
            let mut field_refs = Vec::with_capacity(fields.len());
            for (name, field_expr) in fields.iter() {
                let (field_ty, field_row, field2) = elaborate(arena, *field_expr, ctx, spans, infer)?;
                row = EffectRow::union(&row, &field_row);
                field_tys.push((name.clone(), field_ty));
                field_refs.push((name.clone(), field2));
            }
            Ok((Type::Record(Rc::new(field_tys)), row, arena.push(Expr::Record(Rc::new(field_refs)))))
        }

        // `p.x` -- desugars into an ordinary get_field(target, "x") call
        // (Builtin::GetField, machine.rs) -- never a new machine.rs
        // opcode. Statically checked where `target`'s type says enough
        // to check: a concrete Type::Record (or a Type::Union of them,
        // every alternative required to have the field -- see
        // record_field_type's own doc comment), rejected immediately if
        // it's neither and not Dyn either, the same way calling a
        // non-Fun value is. A Dyn target gets a REAL runtime shape check
        // (build_shape_check, same machinery every other Dyn-to-Record
        // boundary already uses) before the get_field call, not a bare
        // call -- Expr::App's own Type::Dyn arm makes the identical
        // choice (wrapping in build_shallow_check) for the exact same
        // reason its own comment gives: a bare call would leave a
        // shape mismatch to machine.rs's own differently-worded (and
        // differently-styled) panic instead of this file's ordinary
        // fail()+type_name() message.
        Expr::FieldAccess(target, name) => {
            let (target_ty, target_row, target2) = elaborate(arena, target, ctx, spans, infer)?;
            let field_ty = match &target_ty {
                Type::Dyn | Type::Var(_) => Type::Dyn,
                ty if record_shaped(ty) => record_field_type(ty, &name)
                    .ok_or_else(|| TypeError(format!("no field named `{name}`"), spans[expr]))?,
                other => return Err(TypeError(format!("expected a record, found {other}"), spans[target])),
            };
            let checked_target = if matches!(target_ty, Type::Dyn | Type::Var(_)) {
                let required = Type::Record(Rc::new(vec![(name.clone(), Type::Dyn)]));
                build_shape_check(arena, target2, &required)
            } else {
                target2
            };
            let call = build_str_call(arena, "get_field", checked_target, &name);
            Ok((field_ty, target_row, call))
        }

        Expr::ListLit(items) => {
            let mut row = EffectRow::pure();
            let mut elem_ty: Option<Type> = None;
            let mut refs = Vec::with_capacity(items.len());
            for item in items {
                let (item_ty, item_row, item2) = elaborate(arena, item, ctx, spans, infer)?;
                row = EffectRow::union(&row, &item_row);
                refs.push(item2);
                elem_ty = Some(match elem_ty {
                    None => item_ty,
                    Some(t) => match unify(&t, &item_ty, infer, spans[item]) {
                        Ok(()) => infer.resolve_deep(&t),
                        Err(_) => Type::Dyn,
                    },
                });
            }
            // Empty list: a fresh Type::Var, not Type::Dyn -- so a
            // combining site elsewhere (Match's own cross-arm
            // combination, right below) has something real to unify
            // against instead of a Dyn that would otherwise immediately
            // flatten the whole result.
            let list_ty = Type::List(Rc::new(elem_ty.unwrap_or_else(|| infer.fresh_var("elem"))));
            Ok((list_ty, row, arena.push(Expr::ListLit(refs))))
        }

        Expr::Let(..) | Expr::LetRec(..) | Expr::Lambda(..) => {
            unreachable!("Let/LetRec/Lambda are peeled by elaborate's chain-flattening loop")
        }

        Expr::App(f, a) => {
            let (f_ty, f_row, f2) = elaborate(arena, f, ctx, spans, infer)?;
            let (a_ty, a_row, a2) = elaborate(arena, a, ctx, spans, infer)?;
            let (call_row, ret_ty, app2) = match &f_ty {
                Type::Fun(param_ty, call_row, ret_ty) => {
                    // Resolve BEFORE handing to coerce, not just before/
                    // after unify below: coerce has no InferCtx of its
                    // own, and its OWN Type::Var-as-target handling is
                    // blanket-permissive -- ZERO check inserted -- exactly
                    // like Dyn (see coerce's own `matches!(to,
                    // Type::Var(_))` early-return). If param_ty is a bare
                    // Type::Var that unify() has ALREADY bound to
                    // something concrete elsewhere (e.g. `apply = fun f ->
                    // fun x -> f(x)`, where f's own parameter position
                    // gets bound to a concrete Fun shape while apply's
                    // OWN body is elaborated), coerce needs to see that
                    // REAL, resolved shape -- not the stale Var reference
                    // -- or a genuinely invalid call (`apply(5)`, 5 where
                    // a function is required) silently passes with no
                    // check at all. A no-op when param_ty is still
                    // genuinely unconstrained (the ordinary id/const
                    // passthrough-made-real case) or already concrete
                    // (the width/union-subtyping cases below) -- only
                    // changes what coerce sees when something upstream
                    // has ALREADY resolved it.
                    //
                    // ponytail: known ceiling, PRE-EXISTING and not
                    // caused by this line -- exposing a real, resolved
                    // Fun shape to coerce also exposes it to
                    // consistent()'s own Fun arm, which checks BOTH
                    // sides' param types with plain, symmetric
                    // consistent() (no contravariance). A higher-order
                    // parameter inferred to require a WIDER record (e.g.
                    // from a literal constructed inside the callee's own
                    // body) rejects being satisfied by a callback
                    // accepting only a NARROWER one, even though that's
                    // exactly what contravariant function subtyping
                    // would allow. Confirmed via direct comparison this
                    // is NOT new: the identical rejection, byte-for-byte,
                    // already happens today for an EXPLICITLY annotated
                    // parameter of the same shape, on the untouched
                    // pre-this-plan baseline -- this line just gives
                    // inferred positions the SAME behavior explicit ones
                    // always had, which is this whole plan's own theme.
                    // See inferred_higher_order_param_inherits_the_same_
                    // width_subtyping_ceiling_annotated_ones_already_have
                    // (Step 1) for the documented, confirmed-pre-existing
                    // case. Upgrade path if ever needed: give
                    // consistent()'s own Fun arm real contravariant
                    // parameter subtyping -- a substantially larger,
                    // separate feature, well outside this plan's own
                    // scope.
                    let param_ty_resolved = infer.resolve_deep(param_ty);
                    let a3 = coerce(arena, a2, &a_ty, &param_ty_resolved, spans[a])?;
                    let mut row_subst = HashMap::new();
                    // Deliberately the RAW param_ty here, not
                    // param_ty_resolved -- row variables only ever
                    // originate from an explicit `->{e}` annotation
                    // (bind_row_vars's own doc comment), and an
                    // explicitly-annotated param_ty is never a bare
                    // Type::Var in the first place, so resolving it here
                    // would always be a no-op. Keeping this call on the
                    // same raw reference unify() and the row/type
                    // substitution below already use, rather than
                    // introducing a third variant, is simpler with no
                    // behavior difference.
                    bind_row_vars(param_ty, &a_ty, &mut row_subst);
                    let call_row2 = resolve_row(call_row, &row_subst);
                    // Best-effort only -- NOT `?`. `coerce`, just above,
                    // is ALREADY the sole authority on whether this call
                    // is valid: it's subtyping-aware (record width via
                    // record_satisfies, Union alternatives), matching
                    // consistent()'s own established semantics. unify()
                    // is NOT subtyping-aware (its Record/Tuple arms
                    // require exact structural equality, the same
                    // symmetric relation consistent() uses for those
                    // shapes) -- deliberately not generalized to width/
                    // Union tolerance here, since unify() is also used at
                    // several OTHER call sites (Cons, bind_pattern_vars,
                    // and a later task's If/Match/ListLit combination)
                    // where an asymmetric "required vs actual" direction
                    // doesn't apply and exact matching is exactly what's
                    // wanted. unify()'s only job at THIS specific call
                    // site is opportunistic: bind any free Type::Var
                    // param_ty still has (e.g. an instantiated
                    // generalized scheme's own fresh param -- the
                    // passthrough-made-real case) so the return type
                    // below can resolve precisely. Its failure here means
                    // "nothing new to bind, or this pair needs coerce's
                    // own richer width/Union tolerance instead" -- never
                    // a fresh rejection of a call coerce already
                    // approved one line above.
                    //
                    // ponytail: known ceiling -- a genuinely POLYMORPHIC
                    // param combined with a width-subtyped/Union argument
                    // in the SAME call (e.g. a generalized `{x: 'a} -> 'a`
                    // called with `{x: 5, y: 10}`) won't get `'a` bound
                    // here, since unify()'s Record arm rejects on arity
                    // before ever reaching the shared field. Falls back to
                    // reading as Dyn via the still-unresolved Var, same
                    // permissive behavior this whole feature started
                    // from -- not a regression, just not sharpened by
                    // this task. Upgrade path if ever needed: give
                    // unify()'s own Record/Tuple/Union arms the same
                    // one-directional width/alternative tolerance
                    // record_satisfies/consistent() already establish,
                    // choosing a direction (required vs actual) per call
                    // site rather than baking one into unify() globally.
                    let _ = unify(param_ty, &a_ty, infer, spans[a]);
                    // resolve_deep alone only ever consults infer.subst
                    // (ordinary Type::Var bindings) -- it knows nothing
                    // about row_subst, built separately just above.
                    // Apply row_subst first (subst_type with an empty
                    // type_subst leaves every Type::Var untouched, only
                    // touching EffectRow::Var positions), then
                    // resolve_deep to layer in whatever unify() just
                    // bound -- the two substitutions act on disjoint
                    // parts of the type (EffectRow positions vs Type::Var
                    // positions), so composing them in either order gives
                    // the same result.
                    let ret_ty2 = infer.resolve_deep(&subst_type(ret_ty, &row_subst, &HashMap::new()));
                    (call_row2, ret_ty2, arena.push(Expr::App(f2, a3)))
                }
                // NEW: the callee itself isn't known to be a function
                // YET -- an unannotated parameter (Task 2) starts life as
                // a bare Type::Var. Mint a fresh Fun shape and unify the
                // callee against it: this is what actually lets an
                // unannotated callback's parameter type become known
                // from how it's called, the crux of my_map's own
                // correlation (see the design spec's own Motivation and
                // "Where unify is called", site 2).
                Type::Var(_) => {
                    let param_ty = infer.fresh_var("param");
                    let ret_ty = infer.fresh_var("ret");
                    let fun_shape = Type::Fun(Rc::new(param_ty.clone()), EffectRow::Dyn, Rc::new(ret_ty.clone()));
                    unify(&f_ty, &fun_shape, infer, spans[f])?;
                    unify(&param_ty, &a_ty, infer, spans[a])?;
                    let ret_ty2 = infer.resolve_deep(&ret_ty);
                    (EffectRow::Dyn, ret_ty2, arena.push(Expr::App(f2, a2)))
                }
                Type::Dyn => {
                    // Unknown callee: still route "is this even callable"
                    // through the same is_fun/fail desugaring
                    // build_shallow_check uses everywhere else, rather
                    // than leaving it to a differently-worded panic in
                    // machine.rs. Can't know what it might perform, so the
                    // call contributes an unknown (Dyn) row.
                    let f3 = build_shallow_check(arena, f2, &any_fun(), "is_fun");
                    (EffectRow::Dyn, Type::Dyn, arena.push(Expr::App(f3, a2)))
                }
                other => return Err(TypeError(format!("cannot call a value of type {other}"), spans[f])),
            };
            let row = EffectRow::union(&EffectRow::union(&f_row, &a_row), &call_row);
            Ok((ret_ty, row, app2))
        }

        Expr::BinOp(op, l, r) => {
            let (l_ty, l_row, l2) = elaborate(arena, l, ctx, spans, infer)?;
            let (r_ty, r_row, r2) = elaborate(arena, r, ctx, spans, infer)?;
            let row = EffectRow::union(&l_row, &r_row);
            match op {
                // Arithmetic and ordering: both operands must be Int or
                // Float (mixing promotes to Float) -- see coerce_numeric's
                // own doc comment for why this stays a local special case
                // rather than a general Int<->Float consistent() rule.
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Lt => {
                    let l3 = coerce_numeric(arena, l2, &l_ty, spans[l])?;
                    let r3 = coerce_numeric(arena, r2, &r_ty, spans[r])?;
                    // Both concretely Int: stays Int, exactly like before
                    // Float existed (Div/Mod's truncating semantics are
                    // untouched). Either side concretely Float: the OTHER
                    // side promotes to Float at runtime no matter what it
                    // turns out to be (even a Dyn side that's actually an
                    // Int), so Float is knowable here regardless. Anything
                    // else (some Dyn side, no concrete Float forcing
                    // promotion) can't be known until runtime.
                    let numeric_result = match (&l_ty, &r_ty) {
                        (Type::Float, _) | (_, Type::Float) => Type::Float,
                        (Type::Int, Type::Int) => Type::Int,
                        _ => Type::Dyn,
                    };
                    let result_ty = if op == BinOp::Lt { Type::Bool } else { numeric_result };
                    Ok((result_ty, row, arena.push(Expr::BinOp(op, l3, r3))))
                }
                // Equality: operands just need to be consistent with EACH
                // OTHER, not both forced to Int -- `true == false` is a
                // real comparison. Int/Float is the one extra pairing
                // allowed here beyond plain consistent() (numeric_pair) --
                // deliberately narrow to this direct concrete-vs-concrete
                // comparison, NOT extended to a Dyn side: a Dyn value
                // still has to match the OTHER side's actual concrete
                // shape exactly to compare at all (unchanged from before
                // Float existed), same as it already required for Int.
                // If one side is Dyn and the other concrete, coerce the
                // Dyn side to the concrete side's type so the runtime
                // value at least has a known tag; apply_binop compares by
                // matching Value variants.
                BinOp::Eq => {
                    let numeric_pair = matches!(l_ty, Type::Int | Type::Float) && matches!(r_ty, Type::Int | Type::Float);
                    if !numeric_pair && !consistent(&l_ty, &r_ty) {
                        return Err(TypeError(
                            format!("type mismatch: cannot compare {l_ty} with {r_ty}"),
                            spans[expr],
                        ));
                    }
                    let l3 = if l_ty == Type::Dyn && r_ty != Type::Dyn {
                        coerce(arena, l2, &l_ty, &r_ty, spans[l])?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty, spans[r])?
                    } else {
                        r2
                    };
                    Ok((Type::Bool, row, arena.push(Expr::BinOp(op, l3, r3))))
                }
                // Concat: like Eq, operands just need to be consistent with
                // each other -- but ALSO must actually be Str or List, not
                // e.g. Int (unlike Eq, which is happy to compare any two
                // consistent types). Result type: whichever side is
                // concretely known; Dyn if neither is.
                BinOp::Concat => {
                    if !consistent(&l_ty, &r_ty) {
                        return Err(TypeError(
                            format!("type mismatch: cannot concat {l_ty} with {r_ty}"),
                            spans[expr],
                        ));
                    }
                    let result_ty = match (&l_ty, &r_ty) {
                        (Type::Dyn, Type::Dyn) => Type::Dyn,
                        (Type::Dyn, t) | (t, Type::Dyn) => t.clone(),
                        (Type::Str, Type::Str) => Type::Str,
                        (Type::List(_), Type::List(_)) => l_ty.clone(),
                        _ => {
                            return Err(TypeError(
                                format!(
                                    "type mismatch: cannot concat {l_ty} with {r_ty} (expected two strings or two lists)"
                                ),
                                spans[expr],
                            ))
                        }
                    };
                    let l3 = if l_ty == Type::Dyn && r_ty != Type::Dyn {
                        coerce(arena, l2, &l_ty, &r_ty, spans[l])?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty, spans[r])?
                    } else {
                        r2
                    };
                    Ok((result_ty, row, arena.push(Expr::BinOp(op, l3, r3))))
                }
                // `h :: t`: `t` must be List-shaped (or Dyn -- checked no
                // more precisely than that, same shallow "is this a list
                // at all" story build_boundary_check's own is_list arm
                // tells everywhere else; a wrong-shaped Dyn value still
                // fails at apply_binop, just without a location any more
                // precise than machine::current_span already gives every
                // other runtime panic). Result type widens to List(Dyn)
                // unless `h`'s type and `t`'s element type actually agree.
                BinOp::Cons => {
                    // Try real unification first: if the tail side is
                    // (or resolves to) some List(elem) shape -- known
                    // concretely, or still-open via a fresh Type::Var --
                    // unify the head's type into that same element type,
                    // recovering precision unify() alone provides.
                    // Falls back to today's exact "consistent + widen"
                    // check only when the tail side is neither, matching
                    // the SAME hard-error behavior `::` already had --
                    // this is an App/pattern-binding-class site (a
                    // genuine shape requirement), not an If/Match/ListLit
                    // one, so a failure here stays a real error, not a
                    // fallback-to-Dyn.
                    let elem_ty = infer.fresh_var("elem");
                    let list_shape = Type::List(Rc::new(elem_ty.clone()));
                    unify(&r_ty, &list_shape, infer, spans[r])?;
                    unify(&l_ty, &elem_ty, infer, spans[l])?;
                    let result_ty = Type::List(Rc::new(infer.resolve_deep(&elem_ty)));
                    Ok((result_ty, row, arena.push(Expr::BinOp(op, l2, r2))))
                }
            }
        }

        Expr::If(c, t, e) => {
            let (c_ty, c_row, c2) = elaborate(arena, c, ctx, spans, infer)?;
            let c3 = coerce(arena, c2, &c_ty, &Type::Bool, spans[c])?;
            let (t_ty, t_row, t2) = elaborate(arena, t, ctx, spans, infer)?;
            let (e_ty, e_row, e2) = elaborate(arena, e, ctx, spans, infer)?;
            // Try unify first (resolves open type variables when
            // possible); fall back to today's exact widen-to-Dyn
            // behavior on genuine failure, never a new rejection -- see
            // this plan's own Global Constraints.
            let result_ty = match unify(&t_ty, &e_ty, infer, spans[expr]) {
                Ok(()) => infer.resolve_deep(&t_ty),
                Err(_) => Type::Dyn,
            };
            // Only one branch runs, but which one isn't known statically,
            // so the possible effects are the union of both.
            let row = EffectRow::union(&c_row, &EffectRow::union(&t_row, &e_row));
            Ok((result_ty, row, arena.push(Expr::If(c3, t2, e2))))
        }

        // The effect this specific operation performs, plus whatever the
        // payload expression itself might perform.
        Expr::Perform(effect, payload) => {
            let (_, payload_row, payload2) = elaborate(arena, payload, ctx, spans, infer)?;
            let row = EffectRow::union(&payload_row, &EffectRow::single(&effect));
            Ok((Type::Dyn, row, arena.push(Expr::Perform(effect, payload2))))
        }

        Expr::Handle { body, handler } => {
            let (_, body_row, body2) = elaborate(arena, body, ctx, spans, infer)?;
            let (handler_ty, handler_row, handler2) = elaborate(arena, handler, ctx, spans, infer)?;
            // types.rs has no Type::Handler -- but every handler-producing
            // expression (MakeHandler, or deep(...)/shallow(...) applied
            // to one, or a var bound from either) synthesizes Dyn by this
            // same convention, so a concretely-typed handler expression
            // (Int, Bool, Fun) can never legitimately be one. Reject it
            // statically instead of letting it reach machine.rs's panic.
            if handler_ty != Type::Dyn {
                return Err(TypeError(
                    format!("handle: expected a handler value, found expression of type {handler_ty}"),
                    spans[handler],
                ));
            }
            // Discharge the effect this handler catches, if we can
            // statically tell which one that is. If not, conservatively
            // leave body_row untouched (over-approximate, never hide).
            let row = match discharged_effect(arena, handler) {
                Some(effect) => body_row.remove(&effect),
                None => body_row,
            };
            let row = EffectRow::union(&row, &handler_row);
            Ok((Type::Dyn, row, arena.push(Expr::Handle { body: body2, handler: handler2 })))
        }

        // Tried top to bottom at runtime, but statically each arm is just
        // elaborated independently under its own pattern-bound context.
        // Checked before any of that: reachability (first_unreachable) --
        // a dead arm has no runtime signal to fall back on, it just
        // silently never runs, so this is the only chance to catch it, and
        // there's no point elaborating a body that can't run anyway.
        Expr::Match(scrutinee, arms) => {
            let pats: Vec<&Pattern> = arms.iter().map(|(p, _, _)| p).collect();
            let has_guard: Vec<bool> = arms.iter().map(|(_, g, _)| g.is_some()).collect();
            if let Some(i) = first_unreachable(&pats, &has_guard) {
                return Err(TypeError(
                    "unreachable match arm: an earlier arm already covers everything it matches".to_string(),
                    spans[arms[i].2],
                ));
            }

            let (scrut_ty, scrut_row, scrutinee2) = elaborate(arena, scrutinee, ctx, spans, infer)?;
            let mut row = scrut_row;
            let mut result_ty: Option<Type> = None;
            let mut new_arms = Vec::with_capacity(arms.len());
            for (pat, (_, guard, body)) in pats.iter().copied().zip(arms.iter()) {
                if !pattern_could_match(pat, &scrut_ty) {
                    return Err(TypeError(
                        format!(
                            "match: pattern of type {} can never match scrutinee of type {scrut_ty}",
                            pattern_type(pat)
                        ),
                        spans[expr],
                    ));
                }
                let arm_ctx = bind_pattern_vars(ctx, pat, &scrut_ty, infer, spans[expr])?;
                // Same Bool coercion as If's own cond -- a Dyn-typed guard
                // gets a runtime is_bool check inserted, same as
                // everywhere else Dyn meets an expected concrete type.
                let guard2 = match guard {
                    Some(g) => {
                        let (guard_ty, guard_row, g2) = elaborate(arena, *g, &arm_ctx, spans, infer)?;
                        let g3 = coerce(arena, g2, &guard_ty, &Type::Bool, spans[*g])?;
                        row = EffectRow::union(&row, &guard_row);
                        Some(g3)
                    }
                    None => None,
                };
                let (arm_ty, arm_row, body2) = elaborate(arena, *body, &arm_ctx, spans, infer)?;
                row = EffectRow::union(&row, &arm_row);
                result_ty = Some(match result_ty {
                    None => arm_ty,
                    Some(t) => match unify(&t, &arm_ty, infer, spans[expr]) {
                        Ok(()) => infer.resolve_deep(&t),
                        Err(_) => Type::Dyn,
                    },
                });
                new_arms.push((pat.clone(), guard2, body2));
            }
            // A guarded arm's pattern can't be relied on to cover
            // anything for exhaustiveness -- its guard might reject --
            // so only unguarded arms' patterns count here (mirrors
            // first_unreachable's own has_guard gating above).
            let unguarded_pats: Vec<&Pattern> = pats
                .iter()
                .copied()
                .zip(has_guard.iter())
                .filter(|(_, g)| !**g)
                .map(|(p, _)| p)
                .collect();
            if let Some(missing) = missing_case(&unguarded_pats, &scrut_ty) {
                return Err(TypeError(format!("non-exhaustive match: {missing}"), spans[expr]));
            }
            Ok((result_ty.unwrap_or(Type::Dyn), row, arena.push(Expr::Match(scrutinee2, Rc::new(new_arms)))))
        }

        // Constructing the handler value is pure -- the clause body's own
        // effects (including what `resume` re-enters) aren't modeled here;
        // see the doc comment on `elaborate`.
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let inner_ctx = extend(&extend(ctx, &payload_var, Type::Dyn), &resume_var, Type::Dyn);
            let (_, _, body2) = elaborate(arena, body, &inner_ctx, spans, infer)?;
            Ok((
                Type::Dyn,
                EffectRow::pure(),
                arena.push(Expr::MakeHandler { effect, payload_var, resume_var, body: body2 }),
            ))
        }
    }
}

pub fn check(arena: &mut Arena, root: ExprRef, spans: &SpanMap) -> Result<ExprRef, TypeError> {
    let mut infer = InferCtx::new();
    let (_, row, elaborated) = elaborate(arena, root, &Ctx::empty(), spans, &mut infer)?;
    match row {
        EffectRow::Closed(unhandled) if !unhandled.is_empty() => {
            let names: Vec<_> = unhandled.into_iter().collect();
            Err(TypeError(
                format!(
                    "unhandled effect{}: {}",
                    if names.len() > 1 { "s" } else { "" },
                    names.join(", ")
                ),
                spans[root],
            ))
        }
        _ => Ok(elaborated),
    }
}
