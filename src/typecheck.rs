use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::expr::{Arena, BinOp, Expr, ExprRef, Pattern, SpanMap};
use crate::index_expr::IndexExpr;
use crate::plist::PList;
use crate::span::Span;
use crate::types::{consistent, fits, row_consistent, EffectRow, Type};
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

// A binding's type, plus the row-variable, value-type-variable, AND
// index-variable names generalized over it -- quantified fresh at every
// use, the way ML/Haskell generalize a `let`-bound type. Row variables
// only ever come from what the user wrote (an explicit `->{e}`
// annotation); type variables (`type_vars`) are manufactured by real
// unification (`unify`, via `InferCtx::fresh_var`) at many sites, never
// written by a user -- see Type::Var's own doc comment; index variables
// (`index_vars`) are the same idea one level down, for a name appearing
// inside a `Type::Indexed`'s own `IndexExpr` rather than as a `Type`
// itself (manufactured by `unify_index_expr`, via `index_subst`, not
// `fresh_var` -- a separate namespace, see `InferCtx::index_subst`'s own
// doc comment). All three stay simple name substitution
// (extend_generalized computes every set once at the `let`, lookup
// renames them to fresh names at each reference), not a second
// unification engine of its own.
//
// None of the three kinds of variable is safe to generalize wherever its
// own free-variable collector finds it -- see generalizable_row_vars's,
// generalizable_type_vars's, and generalizable_index_vars's own doc
// comments for why each needs a ctx-wide "is this still open in an
// enclosing scope" check.
//
// The general rule behind ALL THREE exclusions, stated once here so a
// future FOURTH kind of variable needing this treatment has one place to
// read it instead of re-deriving it: generalize(ctx, ty) = freevars(ty)
// MINUS freevars(ctx) -- never quantify over a name that's still free
// somewhere still open in the enclosing scope. Applied directly via a
// ctx-wide scan for each kind (generalizable_row_vars/
// free_row_vars_in_ctx, generalizable_type_vars/free_type_vars_in_ctx,
// and generalizable_index_vars/free_index_vars_in_ctx below) -- there is
// no caller-supplied-list shortcut for any of them (Type::Var's own used
// to have one, extend_generalized_with_type_vars, back when it had
// exactly one manufacturing call site; real unification mints it at many,
// so that shortcut no longer applies -- see the design spec's own "Data
// model" section).
#[derive(Clone)]
pub(crate) struct Scheme {
    row_vars: Vec<String>,
    type_vars: Vec<String>,
    // Index-variable names generalized over this binding -- Phase 2's
    // own addition, alongside row_vars/type_vars above. Same
    // generalize-then-instantiate treatment: generalizable_index_vars
    // computes this set at the `let` (a ctx-wide "still open elsewhere"
    // exclusion, exactly like the other two), lookup mints a fresh
    // IndexExpr::Var name per reference via fresh_index_name.
    index_vars: Vec<String>,
    ty: Type,
}

impl Scheme {
    fn mono(ty: Type) -> Scheme {
        Scheme { row_vars: Vec::new(), type_vars: Vec::new(), index_vars: Vec::new(), ty }
    }
}

// pub(crate), not private, only because this project's own test
// convention (see unify_index_expr's neighbors) keeps tests in
// src/lib.rs -- a different module -- rather than because anything
// outside typecheck.rs is meant to name this directly.
pub(crate) type Ctx = PList<Scheme>;

// The real, threaded state a Hindley-Milner-style unifier needs, carried
// through every elaborate()/elaborate_node() call as `&mut InferCtx` --
// see the design spec's own "Data model" section for the full rationale
// (functional substitution over union-find). Generalization does NOT
// live here -- no level counter, no birth-level tracking -- it reuses
// the ctx-wide technique a concurrent row-variable fix already proved
// in this codebase (see Task 7); this struct is just a substitution.
//
// `unify` is now called unconditionally from Expr::App/BinOp::Cons/If/
// Match/ListLit (Tasks 4-6), so `subst` and every method below are live
// on every elaboration, not just exercised by this file's own tests.
pub(crate) struct InferCtx {
    // Grows monotonically as unify() binds variables; never shrinks --
    // no backtracking, matching this checker's existing single-pass
    // character everywhere else.
    subst: HashMap<String, Type>,
    // Grows monotonically exactly like `subst`, for the same reason --
    // no backtracking anywhere in this checker. A SEPARATE map from
    // `subst`: an index variable (IndexExpr::Var) and a type variable
    // (Type::Var) are different namespaces that happen to both be
    // plain Strings -- conflating them would let an ordinary generic
    // function's own type-var name collide with an unrelated Vec's
    // own index-var name.
    pub(crate) index_subst: HashMap<String, IndexExpr>,
    // Every genuinely self-referential type alias this program's own
    // parse registered (parser::Parser's own `named_types`, handed in
    // once at construction) -- consulted on demand by
    // build_boundary_check/build_shape_predicate/pattern_could_match
    // whenever they encounter a Type::Named reference, to unfold it
    // exactly one level. Never mutated after construction -- this
    // registry is a static, parse-time-complete fact about the
    // program, unlike `subst`.
    named_types: HashMap<String, Type>,
}

impl InferCtx {
    pub(crate) fn new(named_types: HashMap<String, Type>) -> InferCtx {
        InferCtx { subst: HashMap::new(), index_subst: HashMap::new(), named_types }
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
    pub(crate) fn resolve_deep(&self, ty: &Type) -> Type {
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
            Type::Indexed(wrapped, index) => {
                Type::Indexed(Rc::new(self.resolve_deep(&wrapped)), Rc::new(self.resolve_index_deep(&index)))
            }
            other => other,
        }
    }

    // IndexExpr's own analog of resolve() -- follows a bound index
    // variable to whatever it's currently bound to, one level (not
    // recursively substituting INSIDE a compound result -- that's
    // resolve_index_deep's job, same split as resolve/resolve_deep).
    pub(crate) fn resolve_index(&self, e: &IndexExpr) -> IndexExpr {
        match e {
            IndexExpr::Var(name) => match self.index_subst.get(name) {
                Some(bound) => self.resolve_index(bound),
                None => e.clone(),
            },
            other => other.clone(),
        }
    }

    // Like resolve_index, but rebuilds a fully-substituted IndexExpr,
    // walking every nested position -- same split as resolve_deep vs.
    // resolve for ordinary Types.
    pub(crate) fn resolve_index_deep(&self, e: &IndexExpr) -> IndexExpr {
        match self.resolve_index(e) {
            IndexExpr::Add(a, b) => IndexExpr::Add(Rc::new(self.resolve_index_deep(&a)), Rc::new(self.resolve_index_deep(&b))),
            IndexExpr::Sub(a, b) => IndexExpr::Sub(Rc::new(self.resolve_index_deep(&a)), Rc::new(self.resolve_index_deep(&b))),
            IndexExpr::Mul(a, b) => IndexExpr::Mul(Rc::new(self.resolve_index_deep(&a)), Rc::new(self.resolve_index_deep(&b))),
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
// Called from unify(), itself called unconditionally from Expr::App/
// BinOp::Cons/If/Match/ListLit (Tasks 4-6) -- live on every elaboration.
fn occurs_in(name: &str, ty: &Type, infer: &InferCtx) -> bool {
    match infer.resolve(ty) {
        Type::Var(n) => n == name,
        Type::List(elem) => occurs_in(name, &elem, infer),
        Type::Fun(param, _row, ret) => occurs_in(name, &param, infer) || occurs_in(name, &ret, infer),
        Type::Tuple(items) | Type::Union(items) => items.iter().any(|t| occurs_in(name, t, infer)),
        Type::Record(fields) => fields.iter().any(|(_, t)| occurs_in(name, t, infer)),
        // A Named reference is a fixed nominal leaf (see its own doc
        // comment) -- it never structurally contains a nested Type, so
        // it can never contain `name` as a free Type::Var either, same
        // as Type::Token right next to it here.
        Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) | Type::Named(_) => false,
        // Wraps one nested Type (the index expression is never itself a
        // Type::Var) -- same recurse-into-the-wrapped-type precedent as
        // List's own arm just above.
        Type::Indexed(wrapped, _) => occurs_in(name, &wrapped, infer),
    }
}

// IndexExpr's own analog of occurs_in -- does `name` appear free
// anywhere inside `e`, resolving through infer's index substitution as
// it recurses? Same rationale as occurs_in's own doc comment: without
// this, unifying `n` against `n + 1` would silently build an infinite
// index expression instead of a clean type error.
fn occurs_in_index(name: &str, e: &IndexExpr, infer: &InferCtx) -> bool {
    match infer.resolve_index(e) {
        IndexExpr::Var(n) => n == name,
        IndexExpr::Lit(_) => false,
        IndexExpr::Add(a, b) | IndexExpr::Sub(a, b) | IndexExpr::Mul(a, b) => {
            occurs_in_index(name, &a, infer) || occurs_in_index(name, &b, infer)
        }
    }
}

// A real, but DELIBERATELY NARROW unifier for IndexExpr -- see the
// design spec's own §4 and this plan's Global Constraints for why:
// this binds a BARE unbound variable to whatever it's compared
// against (mirroring unify()'s own Type::Var arm exactly), and
// recurses when both sides already share the same top-level shape.
// It does NOT solve equations -- unifying `n+1` against `4` does NOT
// infer `n=3` (that would need a real constraint solver, out of scope
// per the spec's own Non-goals). Anything that doesn't hit one of
// those two cases falls back to Phase 1's own index_exprs_equal
// (SOP normalization), which still decides plenty on its own (e.g.
// `m*n` against `n*m` with no unbound variable anywhere).
//
// pub(crate), not private, only because this project's own test
// convention (see occurs_in's neighbors) keeps tests in src/lib.rs --
// a different module -- rather than because anything outside
// typecheck.rs is meant to call this directly.
pub(crate) fn unify_index_expr(a: &IndexExpr, b: &IndexExpr, infer: &mut InferCtx, span: Span) -> Result<(), TypeError> {
    // resolve_index_deep, not resolve_index -- a shallow resolve only
    // unwraps a bare Var, so a bound variable buried inside a compound
    // expression (e.g. `n` in `Add(n, 1)`) stayed unsubstituted and
    // could make an already-true equality (n=3 => n+1 == 4) look like a
    // mismatch, or make the SOP-equality arm below miss cases it should
    // catch (see final review Finding 2).
    let a = infer.resolve_index_deep(a);
    let b = infer.resolve_index_deep(b);
    match (&a, &b) {
        (IndexExpr::Var(n1), IndexExpr::Var(n2)) if n1 == n2 => Ok(()),
        (IndexExpr::Var(name), other) | (other, IndexExpr::Var(name)) => {
            if occurs_in_index(name, other, infer) {
                // A structural occurrence isn't always a genuine
                // infinite-expression violation: `n` against `n + 0` (or
                // any other SOP-equal shape) LOOKS self-referential
                // positionally, but SOP normalization already proves the
                // two sides equal, needing no binding at all -- the same
                // rescue the arm below gives non-Var compound pairs,
                // extended to cover this arm too (final review Finding
                // 2's related Minor issue). Only a GENUINE occurs
                // violation (not SOP-equal either) still errors.
                return if crate::index_expr::index_exprs_equal(&a, &b) {
                    Ok(())
                } else {
                    Err(TypeError(format!("infinite index expression: {name} occurs in {other}"), span))
                };
            }
            infer.index_subst.insert(name.clone(), other.clone());
            Ok(())
        }
        // SOP-equality is checked BEFORE the shape-recursion arms below.
        // Two compound expressions can already be equal under SOP
        // normalization (e.g. `m*n` vs `n*m`) without being pointwise
        // equal operand-by-operand -- if shape-recursion ran first here,
        // it would recurse positionally into (m, n) and (n, m) and bind
        // `m := n`, aliasing two otherwise-independent variables as an
        // unintended side effect. Checking equality first means a pair
        // that's already SOP-equal needs no binding at all; a pair that
        // ISN'T SOP-equal (e.g. `Add(n, 1)` vs `Add(3, 1)`, not equal
        // while `n` is unbound) correctly falls through to recursion.
        _ if crate::index_expr::index_exprs_equal(&a, &b) => Ok(()),
        (IndexExpr::Add(a1, a2), IndexExpr::Add(b1, b2))
        | (IndexExpr::Sub(a1, a2), IndexExpr::Sub(b1, b2))
        | (IndexExpr::Mul(a1, a2), IndexExpr::Mul(b1, b2)) => {
            unify_index_expr(a1, b1, infer, span)?;
            unify_index_expr(a2, b2, infer, span)
        }
        _ => Err(TypeError(format!("type mismatch: index {a} does not unify with index {b}"), span)),
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
// Called unconditionally from Expr::App/BinOp::Cons/If/Match/ListLit
// (Tasks 4-6) -- a real decision point on every elaboration, not a
// not-yet-wired-in mechanism.
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
        // Same nominal-by-id precedent as consistent()'s own new arm
        // (types.rs) -- never unfolds, never binds a Type::Var as a
        // side effect of comparing two Named ids.
        (Type::Named(a), Type::Named(b)) if a == b => Ok(()),
        (Type::List(a), Type::List(b)) => unify(a, b, infer, span),
        // The wrapped type recurses through unify()'s own Var-binding
        // behavior first (matching Type::Fun's own param-then-return
        // order just below); the index halves then go through
        // unify_index_expr -- a real, but deliberately narrow unifier
        // (see its own doc comment) that binds a bare unbound
        // IndexExpr::Var to the other side, rather than the plain
        // SOP-equality check this arm used before Phase 2 Task 3 wired
        // real index-variable inference in.
        (Type::Indexed(wa, ia), Type::Indexed(wb, ib)) => {
            unify(wa, wb, infer, span)?;
            unify_index_expr(ia, ib, infer, span)
        }
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

// The unification-flavored counterpart to fits(): same directional
// walk, but since this IS a unification (not just a static check), it
// also resolves and BINDS any free Type::Var it finds along the way --
// exactly like unify() does, via the same delegation to unify() itself
// for every case that isn't specifically Fun or Record. That delegation
// first tries unify_trial (binds Type::Vars on success, leaves no trace
// on failure); if that trial fails, the pair is still accepted when
// consistent() would accept it, mirroring whatever coerce()/fits() already
// granted -- unify_fits must never be stricter than coerce() itself, since
// it runs right after coerce() as a pure add-on (binding Vars), not a
// second gate. See the catch-all below.
fn unify_fits(required: &Type, actual: &Type, infer: &mut InferCtx, span: Span) -> Result<(), TypeError> {
    let required = infer.resolve(required);
    let actual = infer.resolve(actual);
    match (&required, &actual) {
        (Type::Fun(req_param, req_row, req_ret), Type::Fun(act_param, act_row, act_ret)) => {
            unify_fits(act_param, req_param, infer, span)?;
            unify_fits(req_ret, act_ret, infer, span)?;
            if !row_consistent(req_row, act_row) {
                return Err(TypeError(format!("type mismatch: expected {required}, found {actual}"), span));
            }
            Ok(())
        }
        (Type::Record(req_fields), Type::Record(act_fields)) => {
            for (name, req_ty) in req_fields.iter() {
                match find_field(act_fields, name) {
                    Some(act_ty) => unify_fits(req_ty, act_ty, infer, span)?,
                    None => return Err(TypeError(format!("type mismatch: expected {required}, found {actual}"), span)),
                }
            }
            Ok(())
        }
        // Added alongside Fun/Record above: a Tuple or List annotation can
        // wrap a Vec(n) element (e.g. `(Vec(n), Int)`, `[Vec(n)]`) just
        // like a Record field can, and without an explicit recursive arm
        // here, that pair falls straight to the generic catch-all below --
        // which sees only the OUTER Tuple/List shape, never the nested
        // Indexed pair, so the Type::Indexed-vs-Type::Indexed rescue
        // exclusion just below never fires and a genuine bound-index
        // conflict (n already bound to 3, freshly compared against an
        // incompatible Vec(2)) is wrongly rescued by consistent()'s own
        // permissiveness one level down. Recursing through unify_fits
        // itself (not unify_trial/the bare rescue) re-fires that exclusion
        // at the nesting depth where the Indexed pair actually appears --
        // same fix shape the Record arm above already gets right for
        // `{v: Vec(n)}`. Tuple's length check mirrors unify()'s own Tuple
        // arm just below in this file: a length mismatch is a plain type
        // mismatch, not something to recurse into element-wise.
        (Type::Tuple(req_items), Type::Tuple(act_items)) if req_items.len() == act_items.len() => {
            req_items.iter().zip(act_items.iter()).try_for_each(|(req_ty, act_ty)| unify_fits(req_ty, act_ty, infer, span))
        }
        (Type::List(req_elem), Type::List(act_elem)) => unify_fits(req_elem, act_elem, infer, span),
        // Type::Indexed gets no explicit arm here, deliberately -- this
        // match already has none for Union either, so anything that isn't
        // Fun/Record/Tuple/List falls through to the generic catch-all
        // below: unify_trial tries unify() first (which DOES have its own
        // Indexed arm), with a consistent()/fits() rescue on failure. That
        // path gives Indexed the right answer on match, index mismatch,
        // AND binding a Type::Var reachable only through the wrapped
        // type -- see the design spec's own §3 for the rationale -- so
        // adding a redundant explicit arm here would only duplicate what
        // already works.
        // ponytail: no Type::Var binding via the consistent()/fits() rescue path
        // below -- unify_trial is tried first and binds Vars on success, but a Var
        // reachable ONLY through consistent()/fits() accepting the pair (e.g. only
        // resolvable through one alternative of a Union, or only through the
        // List(Dyn)/Tuple bridge, or only through an Indexed value's own
        // forget-the-index widening) stays unbound (reads as Dyn via the unresolved
        // Var) -- same permissive-fallback ceiling this feature area already
        // documents elsewhere (e.g. If/Match/ListLit branch combination), not a
        // regression, just not newly closed by this fix. Upgrade if real programs
        // need tighter constraint binding through these rescue-only paths.
        _ => match unify_trial(&required, &actual, infer, span) {
            Ok(()) => Ok(()),
            Err(e) => {
                // An Indexed-vs-Indexed pair already has full, authoritative
                // handling via unify()'s own dedicated Indexed arm (see this
                // match's own doc comment above) -- but consistent()'s
                // Type::Indexed arm is now permissive for a bare index
                // variable on either side (Task 6, added so coerce() itself
                // can no-op on a Vec(n) position -- see its own doc
                // comment), and fits()'s own Indexed arm is guarded to
                // exclude an Indexed `required` (see its doc comment), so
                // it falls straight through to that SAME consistent() call
                // for this exact pair shape. Left unguarded, EITHER would
                // wrongly rescue a genuine index conflict unify_index_expr
                // just correctly rejected here (e.g. n already bound to 3
                // by an earlier Vec(n) annotation, freshly compared against
                // a second, incompatible Vec(2)) purely because the
                // required/actual side happens to still be written as a
                // bare `n` -- silently undoing the very binding this
                // function exists to enforce. unify()'s verdict is already
                // authoritative for this one pair shape, so skip the
                // rescue entirely rather than let it re-litigate what
                // unify_index_expr (which DOES resolve through
                // infer.index_subst) already decided.
                if matches!((&required, &actual), (Type::Indexed(..), Type::Indexed(..))) {
                    return Err(e);
                }
                // fits(), not just consistent(): an Indexed-typed `actual`
                // satisfying a plain-typed `required` position (types::fits's
                // own new Indexed arm, "forgetting" the index) is a real
                // fits()/consistent() divergence outside the Fun/Record cases
                // already special-cased above, and unify_fits's own doc
                // comment promises it's never stricter than coerce()/fits()
                // itself -- coerce() already accepts this exact pair via
                // fits() before unify_fits ever runs on it (Expr::App), so
                // this must too.
                if consistent(&required, &actual) || fits(&required, &actual) {
                    Ok(())
                } else if let Type::Named(id) = &required {
                    // Same one-level-unfold rescue as coerce()'s own new
                    // fallback (see its doc comment) -- unify_fits runs
                    // right after coerce() on this exact same (required,
                    // actual) pair (Expr::App's own param_ty/a_ty), so it
                    // must accept anything coerce() itself just accepted,
                    // not re-reject it here a second time. A missing
                    // registry entry IS reachable (not an internal-bug-
                    // only case): a caller pairing plain parse() with
                    // plain check() gets an empty registry -- see coerce()'s
                    // own doc comment on this exact same condition. When
                    // that happens, skip this rescue (coerce() itself will
                    // already have skipped it too, for the same reason) and
                    // fall through to the ordinary Err(e) below.
                    match infer.named_types.get(id) {
                        Some(raw) => {
                            let unfolded = replace_named_with_dyn(&raw.clone(), id);
                            if consistent(&actual, &unfolded) || fits(&unfolded, &actual) {
                                Ok(())
                            } else {
                                Err(e)
                            }
                        }
                        None => Err(e),
                    }
                } else {
                    Err(e)
                }
            }
        },
    }
}

// unify() mutates infer.subst as it recurses and has no rollback -- a
// structural mismatch found partway through (e.g. unifying
// Tuple([Var(x), Int]) against Tuple([Str, Bool]) binds x := Str at
// position 0, THEN fails at position 1) leaves x permanently bound even
// though the unify() call, as a WHOLE, failed. At a hard-error call site
// (bind_pattern_vars/App/Cons) that's harmless -- Err there aborts the
// entire elaboration, so a half-applied binding is never observed. But
// at a fallback-to-Dyn site (If/Match/ListLit's own branch combination),
// the Global Constraint requires a failed unify() to behave EXACTLY like
// today's widen-to-Dyn -- not silently leave a partial binding that
// relocates the rejection to some LATER, unrelated use of that same
// variable. This wrapper snapshots infer.subst AND infer.index_subst
// before trying, and restores both on failure, so a failed trial
// genuinely leaves no trace (see final review Finding 1 -- index_subst
// didn't exist yet when this wrapper's own doc comment was first
// written, and unify() only started writing to it once Task 3 wired
// unify_index_expr in).
//
// pub(crate), not private, only because this project's own test
// convention (see unify_index_expr's neighbors) keeps tests in
// src/lib.rs -- a different module -- rather than because anything
// outside typecheck.rs is meant to call this directly.
pub(crate) fn unify_trial(t1: &Type, t2: &Type, infer: &mut InferCtx, span: Span) -> Result<(), TypeError> {
    let snapshot = infer.subst.clone();
    // unify() (Task 3) now writes to index_subst too, via
    // unify_index_expr's own bare-variable-bind arm -- a failed trial
    // must roll that back as well, or a later, unrelated index
    // comparison would silently see a binding this trial never actually
    // committed to.
    let index_snapshot = infer.index_subst.clone();
    let result = unify(t1, t2, infer, span);
    if result.is_err() {
        infer.subst = snapshot;
        infer.index_subst = index_snapshot;
    }
    result
}

// `Type::Indexed`'s own "forget the index" widening (see types::fits's
// own new Indexed arm), applied locally wherever an operator cares only
// about an operand's wrapped shape, not its statically-tracked length
// (BinOp::Cons/Concat) -- reusing types::fits itself isn't an option at
// these two call sites since they run through unify()/consistent()
// directly, not fits(). One level only: an Indexed value's wrapped type
// is never itself Indexed in this phase (see Type::Indexed's own doc
// comment).
fn forget_index(ty: &Type) -> Type {
    match ty {
        Type::Indexed(wrapped, _) => (**wrapped).clone(),
        _ => ty.clone(),
    }
}

pub(crate) fn lookup(ctx: &Ctx, name: &str, infer: &mut InferCtx) -> Type {
    // Unbound at typecheck time: don't error here, machine::run's own
    // `unbound variable` panic at runtime is the right place for that.
    match ctx.get(name) {
        None => Type::Dyn,
        Some(scheme) if scheme.row_vars.is_empty() && scheme.type_vars.is_empty() && scheme.index_vars.is_empty() => scheme.ty,
        Some(scheme) => {
            let row_subst: HashMap<String, EffectRow> = scheme
                .row_vars
                .iter()
                .map(|v| (v.clone(), EffectRow::Var(fresh_row_name(v))))
                .collect();
            let type_subst: HashMap<String, Type> =
                scheme.type_vars.iter().map(|v| (v.clone(), infer.fresh_var(v))).collect();
            let index_subst: HashMap<String, IndexExpr> =
                scheme.index_vars.iter().map(|v| (v.clone(), IndexExpr::Var(fresh_index_name(v)))).collect();
            subst_type(&scheme.ty, &row_subst, &type_subst, &index_subst)
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

// `let`-binding: generalizes over both row variables (via the existing
// generalizable_row_vars, unchanged) and type variables (via
// generalizable_type_vars) symmetrically -- there is no longer a
// separate caller-supplied-list variant to keep in sync with this one
// (that mechanism, extend_generalized_with_type_vars, only worked when
// Type::Var had exactly one manufacturing call site; real unification
// mints it at many, so it no longer applies -- see the design spec's own
// "Data model" section).
//
// Row variables: if `ty` mentions any row-variable names (from an
// explicit `->{e}` annotation somewhere in it), generalize over them so
// each reference gets its own fresh instantiation -- otherwise two calls
// to the same row-polymorphic function with different concrete callbacks
// would wrongly be forced to agree on one row. row_vars auto-derives
// from free_row_vars(&ty), but only picks up a name if
// generalizable_row_vars confirms it ISN'T also free somewhere still
// open in `ctx` -- see that function's own doc comment for why an
// EffectRow::Var needs this (confirmed reachable: `let f = fun cb: (Dyn
// ->{e} Dyn) -> let g = cb in g in f(fun y -> perform choose(y))(0)`
// used to typecheck clean and only panic at runtime -- "e" leaking from
// cb's still-open annotation into g's generalized scheme, then getting
// freshly (and wrongly) renamed on every `g` reference, exactly like the
// Type::Var bug generalizable_type_vars guards against for the type-var
// side).
//
// Resolve deeply before storing, not just for the free-variable scan:
// unify() can bind one Type::Var to ANOTHER Type::Var (e.g. Cons's own
// unify(&l_ty, &elem_ty, ...) when both sides are still bare vars),
// leaving the stored type's own param positions naming the OLD alias
// rather than the canonical variable generalizable_type_vars actually
// generalizes over. lookup's own instantiation only renames a
// Type::Var node whose LITERAL name is a generalized one -- an
// aliased-but-not-literally-renamed position would silently keep
// resolving to the ORIGINAL (never freshened) variable forever,
// producing an incoherent instantiated type. Resolving once here
// ensures every position that means the same variable also LITERALLY
// names it. Safe to snapshot: infer.subst is write-once per name --
// unify() only ever binds a name that infer.resolve() just showed is
// still unbound (never rebinds one that already has a target), so a
// name resolved here to something concrete can never later resolve to
// something ELSE. A name left as a bare Type::Var here either gets
// generalized (and lookup freshens it before any later use can unify
// against it) or stays free in `ctx` (and downstream consumers that
// read it back out re-resolve it themselves -- see Expr::App's own
// `let f_ty = infer.resolve(&f_ty);`). resolve_deep leaves EffectRow
// untouched (it only walks Type positions), so this doesn't affect
// row-variable generalization at all.
// pub(crate) for the same test-access reason as Ctx/unify_index_expr
// above -- extend_generalized/lookup together are what this task's
// own unit test exercises directly, bypassing the separate Expr::App
// coerce() gap a source-level `identity_vec(3)(va)` call would hit.
pub(crate) fn extend_generalized(ctx: &Ctx, name: &str, ty: Type, infer: &InferCtx) -> Ctx {
    let ty = infer.resolve_deep(&ty);
    let row_vars = generalizable_row_vars(ctx, &ty);
    let type_vars = generalizable_type_vars(ctx, &ty, infer);
    let index_vars = generalizable_index_vars(ctx, &ty, infer);
    ctx.bind(name, Scheme { row_vars, type_vars, index_vars, ty })
}

// Row-variable names free in `ty` that are actually safe to generalize
// over: free_row_vars(ty) MINUS whatever's ALSO free somewhere still open
// in `ctx`. An EffectRow::Var has no single manufacturing call site --
// it comes from ANY `->{e}` annotation anywhere currently in scope (a
// lambda parameter's own annotation, bound via plain `extend`, stays open
// for that parameter's entire body) -- generalizable_type_vars needs the
// identical ctx-wide check for the same reason, just for Type::Var
// instead. A nested `let` that merely happens to mention that same
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
        Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) | Type::Named(_) | Type::Var(_) => BTreeSet::new(),
        // Wraps one nested Type -- same recurse-into-the-wrapped-type
        // precedent as List's own arm just above.
        Type::Indexed(wrapped, _) => free_row_vars(wrapped),
    }
}

// Collects `ty`'s own free Type::Var names, resolved through infer's
// current substitution first -- a variable already bound to something
// concrete isn't "free" any more. Unlike free_row_vars, this needs
// infer: an EffectRow::Var is never bound by anything except generalize/
// instantiate itself, but a Type::Var can be bound at any point during
// elaboration by unify().
fn free_type_vars_resolved(ty: &Type, infer: &InferCtx) -> BTreeSet<String> {
    match infer.resolve(ty) {
        Type::Var(name) => {
            let mut vars = BTreeSet::new();
            vars.insert(name);
            vars
        }
        Type::Fun(param, _row, ret) => {
            let mut vars = free_type_vars_resolved(&param, infer);
            vars.extend(free_type_vars_resolved(&ret, infer));
            vars
        }
        Type::List(elem) => free_type_vars_resolved(&elem, infer),
        Type::Tuple(items) | Type::Union(items) => items.iter().flat_map(|t| free_type_vars_resolved(t, infer)).collect(),
        Type::Record(fields) => fields.iter().flat_map(|(_, t)| free_type_vars_resolved(t, infer)).collect(),
        Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) | Type::Named(_) => BTreeSet::new(),
        // Wraps one nested Type -- same recurse-into-the-wrapped-type
        // precedent as List's own arm just above.
        Type::Indexed(wrapped, _) => free_type_vars_resolved(&wrapped, infer),
    }
}

// Every index-variable name free anywhere inside `e` -- IndexExpr has
// no binding forms of its own (see its own grammar), so every Var is
// free by construction; this is a plain collection, not an occurs- or
// scope-sensitive walk the way free_type_vars_resolved needs to be
// for Type::Var.
fn free_index_vars(e: &IndexExpr) -> BTreeSet<String> {
    match e {
        IndexExpr::Var(name) => BTreeSet::from([name.clone()]),
        IndexExpr::Lit(_) => BTreeSet::new(),
        IndexExpr::Add(a, b) | IndexExpr::Sub(a, b) | IndexExpr::Mul(a, b) => {
            let mut vars = free_index_vars(a);
            vars.extend(free_index_vars(b));
            vars
        }
    }
}

// Type::Indexed's own index-variable analog of free_type_vars_resolved
// -- walks both the wrapped type (for a NESTED Indexed, however deep)
// and, at each Indexed layer, the index expression's own free
// variables (resolved through infer.index_subst first, mirroring how
// free_type_vars_resolved resolves through infer.subst before
// collecting). Every other Type variant contributes nothing -- an
// index variable can only ever appear inside a Type::Indexed's own
// IndexExpr, never anywhere else in a Type.
pub(crate) fn free_index_vars_resolved(ty: &Type, infer: &InferCtx) -> BTreeSet<String> {
    match ty {
        Type::Indexed(wrapped, index) => {
            let mut vars = free_index_vars_resolved(wrapped, infer);
            vars.extend(free_index_vars(&infer.resolve_index_deep(index)));
            vars
        }
        Type::Fun(param, _row, ret) => {
            let mut vars = free_index_vars_resolved(param, infer);
            vars.extend(free_index_vars_resolved(ret, infer));
            vars
        }
        Type::List(elem) => free_index_vars_resolved(elem, infer),
        Type::Tuple(items) | Type::Union(items) => items.iter().flat_map(|t| free_index_vars_resolved(t, infer)).collect(),
        Type::Record(fields) => fields.iter().flat_map(|(_, t)| free_index_vars_resolved(t, infer)).collect(),
        _ => BTreeSet::new(),
    }
}

// Type::Var's own analog of free_row_vars_in_ctx: every type-variable
// name free in any type currently bound in `ctx`, excluding each visited
// scheme's own (sealed) type_vars -- see free_row_vars_in_ctx's own doc
// comment for why the exclusion matters (a name a scheme already
// generalized over is dead, not still open).
//
// ponytail: same O(ctx depth) walk, same O(N^2)-on-a-long-chain ceiling
// free_row_vars_in_ctx already carries a note for -- but paid far more
// often here: unannotated bindings default to a fresh Type::Var broadly
// (Task 2), so most `let`s in an ordinary program have a type variable
// to scan for, unlike a row variable (which only ever comes from an
// explicit `->{e}` annotation). generalizable_type_vars's own
// short-circuit below still keeps a variable-free binding at O(1), but
// that's now the LESS common case. Same upgrade path as
// free_row_vars_in_ctx: thread a cumulative "still open type vars" set
// incrementally through Ctx itself instead of re-walking the chain here.
fn free_type_vars_in_ctx(ctx: &Ctx, infer: &InferCtx) -> BTreeSet<String> {
    let mut vars = BTreeSet::new();
    ctx.for_each(|scheme: &Scheme| {
        let sealed: BTreeSet<String> = scheme.type_vars.iter().cloned().collect();
        vars.extend(free_type_vars_resolved(&scheme.ty, infer).difference(&sealed).cloned());
    });
    vars
}

// Type::Var's own analog of generalizable_row_vars: free_type_vars_resolved(ty)
// MINUS whatever's also free somewhere still open in `ctx`. Skips the ctx
// walk when `ty` has no type vars at all, same short-circuit
// generalizable_row_vars uses for the same reason (keep the overwhelmingly
// common case -- a binding whose type mentions no type variable at all --
// at O(1) instead of paying an O(ctx depth) scan on every single let).
fn generalizable_type_vars(ctx: &Ctx, ty: &Type, infer: &InferCtx) -> Vec<String> {
    let candidates = free_type_vars_resolved(ty, infer);
    if candidates.is_empty() {
        return Vec::new();
    }
    let still_open = free_type_vars_in_ctx(ctx, infer);
    candidates.difference(&still_open).cloned().collect()
}

// Type::Indexed's own analog of free_type_vars_in_ctx/generalizable_type_vars
// -- same ctx-wide "still open somewhere enclosing" exclusion, just over
// index-variable names via free_index_vars_resolved (Task 4) and
// Scheme.index_vars instead of free_type_vars_resolved/Scheme.type_vars.
fn free_index_vars_in_ctx(ctx: &Ctx, infer: &InferCtx) -> BTreeSet<String> {
    let mut vars = BTreeSet::new();
    ctx.for_each(|scheme: &Scheme| {
        let sealed: BTreeSet<String> = scheme.index_vars.iter().cloned().collect();
        vars.extend(free_index_vars_resolved(&scheme.ty, infer).difference(&sealed).cloned());
    });
    vars
}

fn generalizable_index_vars(ctx: &Ctx, ty: &Type, infer: &InferCtx) -> Vec<String> {
    let candidates = free_index_vars_resolved(ty, infer);
    if candidates.is_empty() {
        return Vec::new();
    }
    let still_open = free_index_vars_in_ctx(ctx, infer);
    candidates.difference(&still_open).cloned().collect()
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

// Same idea as fresh_type_name, its own separate counter/namespace --
// an index variable's identity is unrelated to a Type::Var's or an
// EffectRow::Var's, for the same reason index_subst is its own map.
fn fresh_index_name(base: &str) -> String {
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
// Index-expression analog of subst_type's own Type::Var arm -- renames a
// bare IndexExpr::Var per index_subst (built fresh per lookup() call, one
// entry per generalized index variable), leaving Lit and any variable not
// in the map untouched, and recursing into Add/Sub/Mul the same way
// subst_type recurses into Fun/List/Tuple/Union/Record.
fn subst_index_expr(e: &IndexExpr, index_subst: &HashMap<String, IndexExpr>) -> IndexExpr {
    match e {
        IndexExpr::Var(name) => index_subst.get(name).cloned().unwrap_or_else(|| e.clone()),
        IndexExpr::Lit(_) => e.clone(),
        IndexExpr::Add(a, b) => IndexExpr::Add(Rc::new(subst_index_expr(a, index_subst)), Rc::new(subst_index_expr(b, index_subst))),
        IndexExpr::Sub(a, b) => IndexExpr::Sub(Rc::new(subst_index_expr(a, index_subst)), Rc::new(subst_index_expr(b, index_subst))),
        IndexExpr::Mul(a, b) => IndexExpr::Mul(Rc::new(subst_index_expr(a, index_subst)), Rc::new(subst_index_expr(b, index_subst))),
    }
}

fn subst_type(ty: &Type, row_subst: &HashMap<String, EffectRow>, type_subst: &HashMap<String, Type>, index_subst: &HashMap<String, IndexExpr>) -> Type {
    match ty {
        Type::Var(name) => type_subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
        Type::Indexed(wrapped, index) => Type::Indexed(
            Rc::new(subst_type(wrapped, row_subst, type_subst, index_subst)),
            Rc::new(subst_index_expr(index, index_subst)),
        ),
        Type::Fun(param, row, ret) => Type::Fun(
            Rc::new(subst_type(param, row_subst, type_subst, index_subst)),
            resolve_row(row, row_subst),
            Rc::new(subst_type(ret, row_subst, type_subst, index_subst)),
        ),
        Type::List(elem) => Type::List(Rc::new(subst_type(elem, row_subst, type_subst, index_subst))),
        Type::Tuple(items) => Type::Tuple(Rc::new(items.iter().map(|t| subst_type(t, row_subst, type_subst, index_subst)).collect())),
        Type::Union(alts) => Type::Union(Rc::new(alts.iter().map(|t| subst_type(t, row_subst, type_subst, index_subst)).collect())),
        Type::Record(fields) => {
            Type::Record(Rc::new(fields.iter().map(|(n, t)| (n.clone(), subst_type(t, row_subst, type_subst, index_subst))).collect()))
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

// "must be callable" -- the shape used wherever we need to check a value is
// applicable at all without knowing its exact signature (an unannotated
// Dyn-typed callee, or the innermost check inside a function contract).
fn any_fun() -> Type {
    Type::Fun(Rc::new(Type::Dyn), EffectRow::Dyn, Rc::new(Type::Dyn))
}

// Replaces every occurrence of `Type::Named(id)` inside `ty` with
// `Type::Dyn` -- used by coerce()'s own one-level-unfold rescue below,
// and by unify_fits's own matching rescue right above (which runs
// immediately after coerce() on the very same pair at an App site, and
// so must never reject anything coerce() itself just accepted), on the
// SAME id each one just unfolded. See coerce()'s own call site doc
// comment for why: a self-referential alias's one-level definition
// necessarily contains this SAME Type::Named leaf again wherever the
// recursion occurred, and comparing a concrete literal's own
// substructure against that leaf via the ordinary nominal
// consistent()/fits() could never succeed.
//
// The Union arm is special: a BARE Union alternative that is exactly
// Type::Named(id) (e.g. `type A = Int | A`) is DROPPED rather than
// mapped to Dyn, because Dyn is consistent with EVERYTHING -- mapping it
// to Dyn would make the whole Union trivially satisfied by any value at
// all (Union([Int, Dyn]) accepts a Str), silently defeating the
// annotation. Dropping it instead reduces Union([Int, Named(id)]) to
// just Union([Int]), agreeing with the runtime path's own
// build_shape_predicate, whose matching Named arm correctly answers
// `false` for a re-encountered self-reference (see its doc comment) --
// so the self-referential alternative correctly contributes nothing
// beyond what's already reachable through the other alternatives, on
// both the static and runtime path alike. A degenerate fully-self-
// referential alias with no non-recursive alternative at all (`type A =
// A`) reduces to an empty Union, which consistent()/fits() both
// correctly treat as satisfiable by nothing (see their own Union arms) --
// i.e. "always false," not a crash or silent accept.
fn replace_named_with_dyn(ty: &Type, id: &str) -> Type {
    match ty {
        Type::Named(n) if n == id => Type::Dyn,
        Type::List(elem) => Type::List(Rc::new(replace_named_with_dyn(elem, id))),
        Type::Fun(param, row, ret) => Type::Fun(
            Rc::new(replace_named_with_dyn(param, id)),
            row.clone(),
            Rc::new(replace_named_with_dyn(ret, id)),
        ),
        Type::Tuple(items) => Type::Tuple(Rc::new(items.iter().map(|t| replace_named_with_dyn(t, id)).collect())),
        Type::Union(items) => Type::Union(Rc::new(
            items
                .iter()
                .filter(|t| !matches!(t, Type::Named(n) if n == id))
                .map(|t| replace_named_with_dyn(t, id))
                .collect(),
        )),
        Type::Record(fields) => Type::Record(Rc::new(
            fields.iter().map(|(n, t)| (n.clone(), replace_named_with_dyn(t, id))).collect(),
        )),
        other => other.clone(),
    }
}

// The only place a runtime boundary check gets built: `from` is Dyn (or
// Type::Var, treated identically -- see its own doc comment) and `to` is
// concrete. If both sides are concrete
// and disagree, that's a real static error -- reject before running at
// all, UNLESS `fits(to, from)` succeeds: a directional subtyping check that
// accepts wider-field Records (see Pattern::Record's own doc comment for why
// a wider record needs no runtime projection to be used where a narrower
// type is expected) and narrower-parameter Functions (contravariant
// parameters, covariant return type). The `from`/`to` parameter order
// reflects this directionality, unlike consistent()'s own symmetric relation.
// Both cases incur NO wrapping at all: zero overhead for fully-annotated code.
// The error, if any, points at `span` -- the CALLER's job to have already
// looked up via the ORIGINAL (pre-elaboration) ExprRef for the value in
// question, e.g. `spans[val]` where `val` is a field straight off the
// un-elaborated node, NOT the elaborated `e` this function receives to
// potentially wrap: elaborate_node re-pushes almost every compound node it
// touches (App, BinOp, If, ...) regardless of whether anything actually
// changed, so `e` itself is often already a brand new ExprRef past the end
// of the parser-built SpanMap by the time it reaches here -- indexing spans
// BY IT, not by the original, was a latent bug (worked by coincidence
// whenever the mismatched value happened to be a leaf that elaborate_node
// returns unchanged).
fn coerce(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, span: Span, named_types: &HashMap<String, Type>) -> Result<ExprRef, TypeError> {
    if !consistent(from, to) {
        if fits(to, from) {
            return Ok(e);
        }
        // One more rescue, tried only when the exact/width-tolerant
        // checks above both failed: if `to` is a Named reference,
        // unfold it ONE level and retry the SAME two-step check
        // against what it unfolds to. Without this, a literal
        // structural value (a tuple/union literal -- never itself
        // Type::Named, since nothing in this file's own elaboration
        // ever produces that variant for a literal) could NEVER
        // satisfy a Named-typed position without first crossing a
        // real Dyn boundary, which would make constructing a
        // recursive value from a literal effectively impossible.
        // Deliberately asymmetric -- only `to` (never `from`) gets
        // unfolded here, matching every other Type::Named consumer in
        // this file (build_boundary_check/build_shape_predicate/
        // pattern_could_match all unfold the REQUIRED/target side,
        // never the actual/source side). Statically proven when it
        // succeeds, so `e` returns UNCHANGED -- zero overhead, the
        // same "fully-annotated code pays nothing" property every
        // other branch of this function already has.
        // A missing registry entry IS reachable here (not an internal-bug-
        // only case): a caller pairing plain parse() with plain check()
        // gets an empty registry even though parse() itself built one
        // internally (parse()'s own unchanged signature just discards
        // it) -- see build_boundary_check's own doc comment. When that
        // happens, skip this rescue entirely (treat `to` as if it weren't
        // Type::Named for the purposes of this one check) and fall
        // through to the ordinary mismatch error below, rather than
        // panicking.
        // `to`'s registry entry, only when `to` actually is Type::Named --
        // folded into one Option (rather than nesting `if let Type::Named`
        // around `if let Some(raw) = ...`) so there's a single if-let
        // below instead of two, avoiding a clippy::collapsible_if without
        // reaching for a let-chain (this codebase's own established style
        // elsewhere: see bind_row_vars's own nested if, left as-is).
        let named_raw = match to {
            Type::Named(id) => named_types.get(id).map(|raw| (id, raw)),
            _ => None,
        };
        if let Some((id, raw)) = named_raw {
            // A self-referential alias's own one-level definition, by
            // construction, contains this SAME Type::Named leaf again
            // wherever the recursion occurred (parser::Parser's own
            // `contains_named`/insert dance) -- e.g. `List`'s own
            // definition `(Int, List) | Bool` recurses back into
            // Type::Named(id) at the Tuple's own 2nd position. Compared
            // via the ordinary NOMINAL consistent()/fits() (a Type::Named
            // is never structurally consistent with anything but its own
            // exact id -- Task 1's own committed design, unchanged here),
            // a concrete literal's own substructure there could NEVER
            // match -- no literal is ever itself elaborated as
            // Type::Named. Substituting Dyn at that exact leaf instead
            // defers it, matching this whole rescue's own "one level
            // only, nested stays unresolved" character: Dyn is
            // consistent with anything, so only the outer shape actually
            // gets checked here.
            let unfolded = replace_named_with_dyn(raw, id);
            // ponytail: this whole rescue is a purely static, zero-overhead
            // decision -- so for a literal value, nothing is EVER checked at
            // the recursive position past this first level, not deferred to
            // a later shallow check the way a genuine Dyn-boundary crossing
            // works, just never checked, period (e.g. `f((1, (2, 3)))`
            // against `List = (Int, List) | Bool` never confirms `3` is
            // secretly wrapped Bool-shaped anything at that inner position).
            // Upgrade if a real program needs a malformed literal caught at
            // this specific position instead of by whatever consumes it later.
            if consistent(from, &unfolded) || fits(&unfolded, from) {
                return Ok(e);
            }
        }
        return Err(TypeError(format!("type mismatch: expected {to}, found {from}"), span));
    }
    if !matches!(from, Type::Dyn | Type::Var(_)) || *to == Type::Dyn || matches!(to, Type::Var(_)) {
        return Ok(e);
    }
    Ok(build_boundary_check(arena, e, to, named_types, &HashSet::new()))
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
fn coerce_numeric(arena: &mut Arena, e: ExprRef, ty: &Type, span: Span, named_types: &HashMap<String, Type>) -> Result<ExprRef, TypeError> {
    match ty {
        Type::Int | Type::Float => Ok(e),
        Type::Dyn | Type::Var(_) => Ok(build_boundary_check(arena, e, &Type::Union(Rc::new(vec![Type::Int, Type::Float])), named_types, &HashSet::new())),
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
// `visiting` is the set of Named ids currently being unfolded somewhere
// ABOVE this call in the SAME unfold chain (an ancestor set, not a global
// history) -- empty at every genuinely fresh entry point (coerce's own
// Dyn-boundary crossing, coerce_numeric, the FieldAccess call site), and
// extended by exactly one id each time the Type::Named arm below actually
// descends into an unfold. Needed because a self-referential alias whose
// recursive occurrence is a BARE Union alternative (e.g. `type A = Int |
// A`, contrast with `List = (Int, List) | Bool`, where the Tuple's own
// shape check never inspects element types at all and so never revisits
// the Named leaf) would otherwise unfold to the exact same Union([Int,
// Named(id)]) forever -- genuine infinite recursion in THIS function's own
// call stack during elaboration, not a runtime concern.
fn build_boundary_check(arena: &mut Arena, e: ExprRef, to: &Type, named_types: &HashMap<String, Type>, visiting: &HashSet<String>) -> ExprRef {
    match to {
        Type::Int => build_shallow_check(arena, e, to, "is_int"),
        Type::Float => build_shallow_check(arena, e, to, "is_float"),
        Type::Bool => build_shallow_check(arena, e, to, "is_bool"),
        Type::Str => build_shallow_check(arena, e, to, "is_str"),
        Type::List(_) => build_shallow_check(arena, e, to, "is_list"),
        Type::Fun(param_ty, _row, ret_ty) => wrap_fun_contract(arena, e, param_ty.clone(), ret_ty.clone(), named_types, visiting),
        Type::Token(id) => build_token_check(arena, e, to, *id),
        Type::Tuple(_) => build_shape_check(arena, e, to, named_types, visiting),
        Type::Record(_) => build_shape_check(arena, e, to, named_types, visiting),
        Type::Union(_) => build_union_check(arena, e, to, named_types, visiting),
        Type::Dyn => unreachable!("coerce only calls this once *to != Type::Dyn is already established"),
        Type::Var(_) => e,
        // Same is_list-then-len composition as Type::Tuple's own arm
        // above (build_shape_predicate's Tuple case), just checking the
        // length AGAINST the index expression instead of a fixed arity
        // literal -- reusing build_checked (rather than hand-rolling the
        // Let/If/fail wrapper Tuple's own build_shape_check call already
        // provides) keeps `e` evaluated exactly once, the same
        // no-double-evaluation guarantee every other arm here gets.
        // `is_list` has to run BEFORE `len`, same reason as Tuple's own
        // arm: `len` panics internally (see machine.rs) on a non-list
        // argument, which would surface as an unrelated bare panic
        // instead of this function's own clean "type error: expected
        // ..., found ..." message.
        //
        // Only Type::List-wrapped Indexed types are reachable this early
        // -- a Type::Named-wrapped Indexed (derived index-refinement) is
        // a later phase's own concern, nothing before this phase can
        // construct one yet (see index_expr_to_expr's own doc comment
        // for the matching "Phase 2 only" note on the index side).
        Type::Indexed(wrapped, index) => match wrapped.as_ref() {
            Type::List(_) => build_checked(arena, e, to, |arena, v| build_indexed_shape_cond(arena, v, index)),
            _ => e,
        },
        // One level only, matching every other shape check in this
        // file: look up what this id unfolds to and build ITS OWN
        // boundary check, exactly as if `to` had been written directly
        // as that unfolded shape. Any Type::Named NESTED inside that
        // unfolded shape stays unresolved -- checked later, lazily,
        // only if something else separately touches that position.
        // A missing registry entry IS reachable (not an internal-bug-only
        // case): parser::parse (the original, unchanged signature) builds
        // its own named_types registry but discards it rather than
        // returning it, so a caller pairing plain parse() with plain
        // check() (also unchanged, defaults to an empty registry) hits
        // this with nothing registered. Degrade the same way this
        // function's own Type::Var(_) arm above does for "no information
        // here" -- return `e` unchecked -- rather than panicking.
        //
        // If `id` is already an ancestor in THIS unfold chain, stop
        // instead of unfolding again -- but return `e` unchecked
        // (matching this function's own Type::Var(_) arm above) rather
        // than an unconditional fail(). Two different callers reach this
        // branch, and only ONE of them makes fail() safe: build_union_check's
        // own use is always gated behind build_shape_predicate's matching
        // arm, which hard-codes `false` for this exact alternative (see
        // its doc comment) -- so build_union_check's own `If(pred, checked,
        // ...)` can never actually select this branch at runtime, and a
        // fail() there was always dead code no matter what sat in it.
        // wrap_fun_contract's own call (a self-reference in a Fun's
        // RETURN position, e.g. `type F = Dyn -> F`) is NOT gated by any
        // predicate -- it's spliced directly into a contract-wrapped
        // closure's own body, so a fail() there ran unconditionally on
        // EVERY call through that closure, even when the real return
        // value never needed checking at all. `e` unchecked is correct
        // for both: harmless where it's dead code anyway, and the right
        // "checked later, lazily" answer where it actually runs.
        Type::Named(id) => {
            if visiting.contains(id) {
                return e;
            }
            let Some(raw) = named_types.get(id) else { return e };
            let mut visiting = visiting.clone();
            visiting.insert(id.clone());
            let unfolded = raw.clone();
            build_boundary_check(arena, e, &unfolded, named_types, &visiting)
        }
    }
}

// Translates a (currently always closed -- no dependent parameters
// exist until Phase 2) IndexExpr into an ordinary Expr the machine can
// evaluate, for splicing into a synthesized runtime check. A bare
// Var here has no binder yet in this phase; Phase 2 makes this
// meaningful by ensuring any Var appearing in a REACHABLE Vec(n)
// position is always a real, in-scope function parameter by then.
fn index_expr_to_expr(arena: &mut Arena, e: &IndexExpr) -> ExprRef {
    match e {
        IndexExpr::Var(name) => arena.push(Expr::Var(name.clone())),
        IndexExpr::Lit(n) => arena.push(Expr::Int(*n)),
        IndexExpr::Add(a, b) => {
            let a2 = index_expr_to_expr(arena, a);
            let b2 = index_expr_to_expr(arena, b);
            arena.push(Expr::BinOp(BinOp::Add, a2, b2))
        }
        IndexExpr::Sub(a, b) => {
            let a2 = index_expr_to_expr(arena, a);
            let b2 = index_expr_to_expr(arena, b);
            arena.push(Expr::BinOp(BinOp::Sub, a2, b2))
        }
        IndexExpr::Mul(a, b) => {
            let a2 = index_expr_to_expr(arena, a);
            let b2 = index_expr_to_expr(arena, b);
            arena.push(Expr::BinOp(BinOp::Mul, a2, b2))
        }
    }
}

// The raw boolean condition shared by build_boundary_check's and
// build_shape_predicate's own Type::Indexed(List(_), _) arms: "is
// `value_ref` list-shaped AND does its length equal `index`" -- the same
// is_list-then-len composition Type::Tuple's own arity check uses
// elsewhere in this file. `is_list` has to run BEFORE `len`, same reason
// as Tuple's own arm: `len` panics internally (see machine.rs) on a
// non-list argument, which would surface as an unrelated bare panic
// instead of a clean "type error: expected ..., found ..." message.
// Returning the bare condition (not a full Let/If/fail wrapper) lets
// build_boundary_check wrap it via build_checked (evaluating `value_ref`
// exactly once) while build_shape_predicate uses it directly as its own
// bare predicate (no let-binding, callers may OR several of these
// together) -- see each caller's own doc comment for why they need
// different wrapping.
fn build_indexed_shape_cond(arena: &mut Arena, value_ref: ExprRef, index: &IndexExpr) -> ExprRef {
    let is_list = build_predicate_call(arena, "is_list", value_ref);
    let len_var = arena.push(Expr::Var("len".to_string()));
    let len_call = arena.push(Expr::App(len_var, value_ref));
    let index_expr = index_expr_to_expr(arena, index);
    let len_eq = arena.push(Expr::BinOp(BinOp::Eq, len_call, index_expr));
    let false_lit = arena.push(Expr::Bool(false));
    arena.push(Expr::If(is_list, len_eq, false_lit))
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
fn build_union_check(arena: &mut Arena, e: ExprRef, to: &Type, named_types: &HashMap<String, Type>, visiting: &HashSet<String>) -> ExprRef {
    let alts: Rc<Vec<Type>> = match to {
        Type::Union(alts) => alts.clone(),
        _ => unreachable!("build_union_check is only ever called with a Union target"),
    };
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let mut result = build_fail_call(arena, to, tmp_ref);
    for alt in alts.iter().rev() {
        let pred = build_shape_predicate(arena, tmp_ref, alt, named_types, visiting);
        let checked = build_boundary_check(arena, tmp_ref, alt, named_types, visiting);
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
fn build_shape_predicate(arena: &mut Arena, value_ref: ExprRef, ty: &Type, named_types: &HashMap<String, Type>, visiting: &HashSet<String>) -> ExprRef {
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
            fold_predicate(arena, alts, FoldOp::Or, |arena, alt| build_shape_predicate(arena, value_ref, alt, named_types, visiting))
        }
        // Same one-level-only unfold as build_boundary_check's own new
        // arm just above. A missing registry entry IS reachable (not an
        // internal-bug-only case): a caller pairing plain parse() with
        // plain check() gets an empty registry even though parse() itself
        // built one internally (parse()'s own unchanged signature just
        // discards it) -- see build_boundary_check's own doc comment.
        // Degrade the same way this function's own Type::Dyn/Var(_) arms
        // above do for "no information here" -- answer `true` -- rather
        // than panicking.
        //
        // Same ancestor-tracking cycle break as build_boundary_check's
        // own Named arm (see its doc comment for the bare-Union-
        // alternative infinite-recursion case this prevents), but
        // stopping at `false` here instead of build_fail_call: re-trying
        // the same id against the same value gives no new information
        // beyond what the non-recursive alternatives already cover, so
        // this correctly makes e.g. `type A = Int | A` behave equivalently
        // to plain `Int` (the only alternative that can ever actually
        // match) instead of recursing forever.
        Type::Named(id) => {
            if visiting.contains(id) {
                return arena.push(Expr::Bool(false));
            }
            let Some(raw) = named_types.get(id) else { return arena.push(Expr::Bool(true)) };
            let mut visiting = visiting.clone();
            visiting.insert(id.clone());
            let unfolded = raw.clone();
            build_shape_predicate(arena, value_ref, &unfolded, named_types, &visiting)
        }
        // Unlike Named's own one-level unfold just above (pure delegation
        // to what it unfolds to), an Indexed value's LENGTH is real,
        // provable shape information this check must not throw away --
        // this is what lets build_union_check correctly fall through a
        // Vec(3) alternative to try a plain [Int] alternative next
        // instead of routing a length-5 list into Vec(3)'s own full
        // check (which would then reject it) just because it merely
        // "looks like some list." Reuses the SAME is_list-then-len
        // condition build_boundary_check's own Indexed arm uses (see
        // build_indexed_shape_cond's own doc comment) rather than
        // shape-only delegation. A Type::Named-wrapped Indexed (derived
        // index-refinement) is a later phase's own concern -- nothing
        // before this phase can construct one yet -- so it still falls
        // back to shape-only delegation, unchanged.
        Type::Indexed(wrapped, index) => match wrapped.as_ref() {
            Type::List(_) => build_indexed_shape_cond(arena, value_ref, index),
            _ => build_shape_predicate(arena, value_ref, wrapped, named_types, visiting),
        },
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
fn build_shape_check(arena: &mut Arena, e: ExprRef, to: &Type, named_types: &HashMap<String, Type>, visiting: &HashSet<String>) -> ExprRef {
    build_checked(arena, e, to, |arena, v| build_shape_predicate(arena, v, to, named_types, visiting))
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
fn wrap_fun_contract(arena: &mut Arena, e: ExprRef, param_ty: Rc<Type>, ret_ty: Rc<Type>, named_types: &HashMap<String, Type>, visiting: &HashSet<String>) -> ExprRef {
    let fn_var = "__contract_fn".to_string();
    let arg_var = "__contract_arg".to_string();
    let fn_var_ref = arena.push(Expr::Var(fn_var.clone()));
    let arg_var_ref = arena.push(Expr::Var(arg_var.clone()));
    let checked_fn = build_shallow_check(arena, fn_var_ref, &any_fun(), "is_fun");
    let call = arena.push(Expr::App(checked_fn, arg_var_ref));
    let checked_call = build_boundary_check(arena, call, &ret_ty, named_types, visiting);
    let lambda = arena.push(Expr::Lambda(arg_var, Some((*param_ty).clone()), checked_call));
    arena.push(Expr::Let(fn_var, None, e, lambda))
}

// The bidirectional-checking mode a call to `elaborate_mode`/`elaborate_node`
// runs under. `Synth` is today's only mode (infer a type bottom-up, no
// expected type available). `Check(expected)` is new: the caller already
// knows what type this expression must have (typically a declared
// annotation flowing down through a Let/Lambda chain) -- `If` and `Match`
// use this to check each branch/arm against `expected` directly instead of
// inferring each one and reconciling them against EACH OTHER afterward
// (today's only option, via unify_trial). Every other expression shape
// still just synthesizes regardless of mode; `check_against`'s own
// coerce-then-unify_fits fallback (mirroring Expr::App's existing
// argument-check pattern) is what makes Check mode sound for those shapes
// too, without each of them needing its own Check-mode arm.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Mode<'a> {
    Synth,
    Check(&'a Type),
}

// A `let`/`fun` prefix collected while flattening a chain of them (see
// `elaborate_mode`) -- deferred until the terminal body is elaborated, then
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
// `visiting` is the same ancestor-tracking device build_boundary_check/
// build_shape_predicate use (see build_boundary_check's own doc comment):
// the set of Type::Named ids currently being unfolded somewhere ABOVE
// this call in the SAME unfold chain, empty at the one genuine entry
// point (the Expr::Match call site below) and extended by exactly one id
// each time the Type::Named arm actually descends into an unfold. Needed
// for the same reason it's needed there: a self-referential alias whose
// recursive occurrence is a BARE Union alternative (`type A = Int | A`)
// unfolds to the exact same Union([Int, Named(id)]) every time, and a
// Record/List pattern's own Union arm just below fans out into EVERY
// alternative (including that same Named(id) again) -- without this,
// that's genuine infinite recursion in this function's own call stack,
// not just a runtime concern.
fn pattern_could_match(pat: &Pattern, ty: &Type, named_types: &HashMap<String, Type>, visiting: &HashSet<String>) -> bool {
    match (pat, ty) {
        (Pattern::Record(fields), Type::Record(type_fields)) => {
            fields.iter().all(|(name, _)| find_field(type_fields, name).is_some())
        }
        (Pattern::Record(_), Type::Union(alts)) => alts.iter().any(|alt| pattern_could_match(pat, alt, named_types, visiting)),
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
        (Pattern::List(_), Type::Union(alts)) => alts.iter().any(|alt| pattern_could_match(pat, alt, named_types, visiting)),
        (Pattern::List(_), Type::Dyn | Type::Var(_)) => true,
        // One level only, matching every other consumer of Type::Named
        // in this file: unfold what this id stands for and ask the SAME
        // question about that shape instead. A missing registry entry IS
        // reachable (not an internal-bug-only case): a caller pairing
        // plain parse() with plain check() gets an empty registry even
        // though parse() itself built one internally (parse()'s own
        // unchanged signature just discards it) -- see
        // build_boundary_check's own doc comment. Degrade the same way
        // this function's own Type::Dyn/Var(_) arms above do for "no
        // information here" -- answer `true` -- rather than panicking.
        //
        // Ancestor-tracking cycle break, same as build_shape_predicate's
        // own Named arm: if `id` is already being unfolded somewhere
        // above this call, stop and answer `false` instead of unfolding
        // again -- re-trying the same id against the same pattern gives
        // no new information beyond what the non-recursive alternatives
        // already cover (the `any()` fan-out above already tried them),
        // so this correctly makes e.g. `type A = Int | A` behave
        // equivalently to plain `Int` for reachability purposes instead
        // of recursing forever.
        (_, Type::Named(id)) => {
            if visiting.contains(id) {
                return false;
            }
            let Some(unfolded) = named_types.get(id) else { return true };
            let mut visiting = visiting.clone();
            visiting.insert(id.clone());
            pattern_could_match(pat, unfolded, named_types, &visiting)
        }
        // An Indexed-typed scrutinee is, at runtime, just its wrapped
        // type's own value (Vec(n) sugars over this) -- coverage of
        // []/:: is a purely structural question, independent of what the
        // index is (see the design spec's own §5 note that exhaustiveness
        // is unaffected by Vec). Delegate straight to the wrapped type,
        // mirroring build_shape_predicate's own Type::Indexed arm exactly.
        (_, Type::Indexed(wrapped, _)) => pattern_could_match(pat, wrapped, named_types, visiting),
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
// Case A eligibility (spec section 5): is `scrut_ty` (already resolved via
// infer.resolve_deep) a Type::Indexed wrapping a plain Type::List, whose
// own index expression (already resolved via infer.resolve_index_deep)
// is a BARE variable? If so, returns that variable's name and the
// wrapped element type -- everything a Match arm needs to compute its
// own base/step hypothesis. Anything else (a non-Indexed scrutinee, an
// Indexed-but-non-List wrapped type -- that's Case B, a later task's own
// job -- a literal or compound index expression with no single name to
// bind a hypothesis under) returns None: the safe "not applicable"
// fallback used throughout this design.
fn case_a_refinement_target(resolved_scrut_ty: &Type, resolved_index: Option<&IndexExpr>) -> Option<(String, Rc<Type>)> {
    match (resolved_scrut_ty, resolved_index) {
        (Type::Indexed(wrapped, _), Some(IndexExpr::Var(name))) => match &**wrapped {
            Type::List(elem_ty) => Some((name.clone(), elem_ty.clone())),
            _ => None,
        },
        _ => None,
    }
}

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
            // A still-fully-unconstrained scrutinee (Type::Var) is
            // handled the SAME way as a concrete, non-Tuple, non-List
            // scrutinee just below: bind each position to its own
            // independent fresh var, no unify against a forced List
            // shape. Pattern::List is ALSO renno's tuple pattern
            // (`(a, b)` desugars to this same variant) -- the two are
            // indistinguishable at this point, so forcing scrutinee_ty
            // into a List shape here is a guess that's WRONG whenever
            // the pattern is actually destructuring a Tuple, and
            // unrecoverable once made (infer.subst never shrinks). An
            // actual List-shaped scrutinee still gets its real
            // correlation via Cons's own arm, which is where my_map's
            // own precision genuinely comes from -- this arm never
            // forcing List costs nothing there.
            Type::Var(_) => {
                let mut c = ctx.clone();
                for p in pats {
                    let elem_ty = infer.fresh_var("elem");
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
            // "Forget" an Indexed scrutinee's own tracked index before
            // correlating its shape -- same widening as BinOp::Cons's own
            // fix (see forget_index's own doc comment): matching `h :: t`
            // against a Vec(3)-typed scrutinee needs to see its list
            // shape, not hard-fail here just because Vec(3) isn't
            // literally Type::List.
            unify(&forget_index(scrutinee_ty), &list_shape, infer, span)?;
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
// for ITS OWN (also Tuple) position. A scrutinee bound by an outer
// match/lambda pattern first now ALSO benefits: bind_pattern_vars's own
// Pattern::Var arm resolves and stores the real scrutinee type (Task 3's
// own correlation, not always Dyn any more), so `match (1, 2) | p ->
// match p | (a, b) -> a + b` is correctly recognized exhaustive too, not
// just a fully-nested pattern written in one match.
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
fn elaborate_mode(arena: &mut Arena, expr: ExprRef, ctx: &Ctx, spans: &SpanMap, infer: &mut InferCtx, mode: Mode) -> Result<(Type, EffectRow, ExprRef), TypeError> {
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
    let mut cur_mode = mode;
    loop {
        // Expr is Clone and, now that its fields are ExprRef (Copy)
        // instead of Rc<Expr>, cheap to clone -- this ends the borrow on
        // `arena` before the arm below needs to mutate it.
        let node = arena[cur_expr].clone();
        match node {
            Expr::Let(var, ann, val, body) => {
                // `cur_mode` is deliberately NOT read or written here: a
                // `let` binding's own annotation (if any) is a fully
                // self-contained expected type for `val` -- it doesn't
                // come from, or change, whatever mode the ENCLOSING
                // chain is running under, and the body/tail after this
                // binding stays under that same enclosing mode.
                let (bound_ty, val_row, val3) = match ann {
                    // `check_against` folds today's coerce-then-unify_fits
                    // pair into one call, AND -- since val may itself be a
                    // Lambda/If/Match -- lets the declared annotation `t`
                    // actually flow into val's own tail position instead of
                    // only being checked against val's independently
                    // inferred type after the fact. This is Phase 3's real
                    // entry point: `let f: A -> Vec(n) = fn(x) = <body>`
                    // now checks <body> against Vec(n) directly.
                    Some(t) => {
                        let (val_row, val4) = check_against(arena, val, &t, &cur_ctx, spans, infer)?;
                        (t.clone(), val_row, val4)
                    }
                    None => {
                        let (val_ty, val_row, val2) = elaborate(arena, val, &cur_ctx, spans, infer)?;
                        (val_ty, val_row, val2)
                    }
                };
                cur_ctx = extend_generalized(&cur_ctx, &var, bound_ty.clone(), infer);
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
                for (name, ann, val) in bindings.iter() {
                    // Same check_against restructuring as Expr::Let just
                    // above -- see its own comment. `cur_mode` is likewise
                    // untouched: each binding's annotation is its own
                    // self-contained expected type.
                    let (bound_ty, val_row, val3) = match ann {
                        Some(t) => {
                            let (val_row, val4) = check_against(arena, *val, t, &val_ctx, spans, infer)?;
                            (t.clone(), val_row, val4)
                        }
                        None => {
                            let (val_ty, val_row, val2) = elaborate(arena, *val, &val_ctx, spans, infer)?;
                            (val_ty, val_row, val2)
                        }
                    };
                    elaborated.push((name.clone(), bound_ty, val_row, val3));
                }
                for (name, bound_ty, _, _) in elaborated.iter() {
                    cur_ctx = extend_generalized(&cur_ctx, name, bound_ty.clone(), infer);
                }
                pending.push(PendingElab::LetRec { bindings: elaborated });
                cur_expr = body;
            }
            Expr::Lambda(param, ann, body) => {
                // The one place `cur_mode` actually changes: if the
                // caller already knows this Lambda must have type
                // Fun(param_ty, _, ret_ty) -- e.g. it's the value of a
                // `let f: X -> Y = fn(p) = ...` binding, or a nested
                // Lambda inside a curried one -- adopt that expected
                // param type (or this Lambda's own local annotation,
                // unify_fits-checked against it so a mismatched explicit
                // annotation is still a real error, not silently
                // overridden) and switch `cur_mode` to Check(ret_ty) for
                // the rest of the chain, so THIS Lambda's own body (and
                // anything nested inside it: more Lets, another curried
                // Lambda layer, or a tail Match/If) gets checked against
                // ret_ty instead of inferred blind. Any other expected
                // type (Check with a non-Fun type, or Synth) means this
                // position isn't known to be a function -- fall back to
                // today's exact Synth behavior for this Lambda's own
                // param/body.
                let (param_ty, next_mode): (Type, Mode) = match cur_mode {
                    Mode::Check(Type::Fun(expected_param, _eff, expected_ret)) => {
                        let param_ty = match ann {
                            Some(local) => {
                                // Parameter position is contravariant --
                                // see unify_fits's own Fun arm, which
                                // recurses as unify_fits(act_param,
                                // req_param), i.e. (actual, required), not
                                // this function's own top-level (required,
                                // actual) order. `local` is the actual
                                // (the Lambda's own explicit annotation);
                                // `expected_param` is required (the outer
                                // Fun type's declared param type).
                                unify_fits(&local, expected_param, infer, spans[cur_expr])?;
                                local
                            }
                            None => (**expected_param).clone(),
                        };
                        (param_ty, Mode::Check(expected_ret))
                    }
                    _ => (ann.unwrap_or_else(|| infer.fresh_var(&param)), Mode::Synth),
                };
                cur_ctx = extend(&cur_ctx, &param, param_ty.clone());
                pending.push(PendingElab::Fun { param, param_ty });
                cur_mode = next_mode;
                cur_expr = body;
            }
            _ => break,
        }
    }

    // Dispatch the chain's tail under whatever `cur_mode` is in force at
    // this point -- which may have been legitimately downgraded to Synth
    // by a Lambda peel above whose own Fun-shape check didn't fire, or may
    // still genuinely be a `Check(_)` inherited from an outer annotation
    // (e.g. a Lambda peel that DID descend into `Check(ret_ty)` for its own
    // body). Either way, immediately enforce `cur_mode` right here, at the
    // position actually reached -- this is the fallback the spec's own
    // section 6 describes ("propagate the expected type top-down... until a
    // form is reached with no checking rule, then fall back to elaborate +
    // unify/coerce"): most shapes (anything but If/Match in Check mode)
    // have no checking rule of their own, so this is where that fallback
    // actually fires. When the tail IS an If/Match under Check mode, it
    // already returns `expected.clone()` for `result_ty`, so this becomes a
    // proven no-op (coerce/unify_fits against equal types never do
    // anything here) -- safe to apply unconditionally. This is NOT a
    // substitute for the trailing, post-unwind check below: that one
    // catches the case where a Lambda peel's OWN Fun-shape check didn't
    // fire and `cur_mode` was downgraded to Synth, silently dropping the
    // outer obligation -- this tail-level check can't see that, since by
    // then `cur_mode` is already Synth. Both checks are needed.
    let (mut result_ty, mut result_row, mut result_expr) =
        elaborate_node(arena, cur_expr, &cur_ctx, spans, infer, cur_mode)?;
    if let Mode::Check(expected) = cur_mode {
        result_expr = coerce(arena, result_expr, &result_ty, expected, spans[cur_expr], &infer.named_types)?;
        unify_fits(expected, &result_ty, infer, spans[cur_expr])?;
        result_ty = expected.clone();
    }

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

    // The one place the caller's ORIGINAL obligation is actually enforced:
    // after Lambda/Let/LetRec unwinding has reconstructed the TRUE overall
    // type of `expr` (not just the innermost tail's), require it to fit
    // `mode`'s own expected type -- using `mode`, never `cur_mode` (which
    // may have been downgraded to Synth by a Lambda peel whose own
    // Fun-shape check didn't fire; that downgrade only ever meant "freely
    // elaborate the inner tail," never "skip validating the outer,
    // reconstructed value against what the caller actually asked for" --
    // see this file's own regression history for why skipping this step
    // silently accepted `let f: Int = fun x -> x in ...`).
    if let Mode::Check(expected) = mode {
        let checked = coerce(arena, result_expr, &result_ty, expected, spans[expr], &infer.named_types)?;
        unify_fits(expected, &result_ty, infer, spans[expr])?;
        result_ty = expected.clone();
        result_expr = checked;
    }

    Ok((result_ty, result_row, result_expr))
}

// Today's only mode until Phase 3: infer `expr`'s type bottom-up with no
// expected type available. Unchanged signature and behavior from before
// this file had a Mode enum -- every existing call site is untouched.
fn elaborate(arena: &mut Arena, expr: ExprRef, ctx: &Ctx, spans: &SpanMap, infer: &mut InferCtx) -> Result<(Type, EffectRow, ExprRef), TypeError> {
    elaborate_mode(arena, expr, ctx, spans, infer, Mode::Synth)
}

// The bidirectional-checking entry point (spec section 6): verify `expr`
// against an already-known `expected` type. Callers that already know the
// type discard the redundant `Type` `elaborate_mode` would otherwise
// return (by contract, `Mode::Check(expected)` always returns
// `expected.clone()` on success) and get back only the effect row and the
// elaborated expression, the same shape `coerce` alone used to hand back
// to callers like Expr::Let's own annotated-binding case before this
// function existed.
pub(crate) fn check_against(arena: &mut Arena, expr: ExprRef, expected: &Type, ctx: &Ctx, spans: &SpanMap, infer: &mut InferCtx) -> Result<(EffectRow, ExprRef), TypeError> {
    let (_, row, expr2) = elaborate_mode(arena, expr, ctx, spans, infer, Mode::Check(expected))?;
    Ok((row, expr2))
}

// Every Expr variant except Let/Lambda, which `elaborate_mode` peels off
// iteratively above -- reached only once no more chain prefix remains.
// `expr` (this function's own parameter) is always the ORIGINAL,
// pre-elaboration ExprRef for whatever's currently being checked, so
// `spans[expr]` is a valid, always-available "point at this whole
// construct" location for any error an arm below doesn't have a more
// specific sub-expression to blame instead.
fn elaborate_node(arena: &mut Arena, expr: ExprRef, ctx: &Ctx, spans: &SpanMap, infer: &mut InferCtx, mode: Mode) -> Result<(Type, EffectRow, ExprRef), TypeError> {
    let node = arena[expr].clone();
    match node {
        Expr::Int(_) => Ok((Type::Int, EffectRow::pure(), expr)),
        Expr::Float(_) => Ok((Type::Float, EffectRow::pure(), expr)),
        Expr::Bool(_) => Ok((Type::Bool, EffectRow::pure(), expr)),
        Expr::Str(_) => Ok((Type::Str, EffectRow::pure(), expr)),
        // A singleton per source position -- see Expr::Token's own doc
        // comment.
        Expr::Token(id) => Ok((Type::Token(id), EffectRow::pure(), expr)),
        Expr::Var(name) => Ok((lookup(ctx, &name, infer), EffectRow::pure(), expr)),

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
                build_shape_check(arena, target2, &required, &infer.named_types, &HashSet::new())
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
                    Some(t) => match unify_trial(&t, &item_ty, infer, spans[item]) {
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
            // Resolve before dispatching: f_ty is read straight out of
            // Ctx (via lookup), which -- since Task 7's ctx-wide
            // generalization -- can hand back a raw Type::Var that
            // unify() has ALREADY bound to something concrete elsewhere
            // (e.g. a variable reused across two calls in the same
            // scope). Matching on the unresolved form would route a
            // callee that's actually a concrete Fun through the
            // Type::Var(_) arm below -- which has no coerce() at all --
            // instead of the Type::Fun arm's own subtyping-aware
            // handling, or would silently skip the Type::Dyn arm's own
            // is_fun check when the callee resolves to Dyn. Same
            // principle as Task 3's resolve_deep in Pattern::Var and
            // Task 4's param_ty_resolved just below.
            let f_ty = infer.resolve(&f_ty);
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
                    let param_ty_resolved = infer.resolve_deep(param_ty);
                    let a3 = coerce(arena, a2, &a_ty, &param_ty_resolved, spans[a], &infer.named_types)?;
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
                    unify_fits(param_ty, &a_ty, infer, spans[a])?;
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
                    let ret_ty2 = infer.resolve_deep(&subst_type(ret_ty, &row_subst, &HashMap::new(), &HashMap::new()));
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
                    let l3 = coerce_numeric(arena, l2, &l_ty, spans[l], &infer.named_types)?;
                    let r3 = coerce_numeric(arena, r2, &r_ty, spans[r], &infer.named_types)?;
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
                        coerce(arena, l2, &l_ty, &r_ty, spans[l], &infer.named_types)?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty, spans[r], &infer.named_types)?
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
                    // "Forget" either side's own tracked index for the
                    // purposes of this operator -- Concat only cares
                    // whether an operand is list-shaped, not what its
                    // statically-known length is (see types::fits's own
                    // new Type::Indexed arm for the same widening, used
                    // for the App-argument case instead). Phase 1 has no
                    // rule for the RESULT length of concatenating two
                    // Vec(n)s, so this deliberately drops to the wrapped
                    // type entirely rather than half-tracking it.
                    let l_ty = forget_index(&l_ty);
                    let r_ty = forget_index(&r_ty);
                    if !consistent(&l_ty, &r_ty) {
                        return Err(TypeError(
                            format!("type mismatch: cannot concat {l_ty} with {r_ty}"),
                            spans[expr],
                        ));
                    }
                    let result_ty = match (&l_ty, &r_ty) {
                        (Type::Dyn | Type::Var(_), Type::Dyn | Type::Var(_)) => Type::Dyn,
                        (Type::Dyn | Type::Var(_), t) | (t, Type::Dyn | Type::Var(_)) => t.clone(),
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
                        coerce(arena, l2, &l_ty, &r_ty, spans[l], &infer.named_types)?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty, spans[r], &infer.named_types)?
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
                // other runtime panic).
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
                    // "Forget" the tail's own tracked index before
                    // unifying its shape -- same widening as Concat's own
                    // arm just above (see forget_index's own doc
                    // comment); the result is always a plain Type::List
                    // regardless, so there's nothing Indexed left to
                    // preserve past this point anyway.
                    unify(&forget_index(&r_ty), &list_shape, infer, spans[r])?;
                    unify(&l_ty, &elem_ty, infer, spans[l])?;
                    let result_ty = Type::List(Rc::new(infer.resolve_deep(&elem_ty)));
                    Ok((result_ty, row, arena.push(Expr::BinOp(op, l2, r2))))
                }
            }
        }

        Expr::If(c, t, e) => {
            let (c_ty, c_row, c2) = elaborate(arena, c, ctx, spans, infer)?;
            let c3 = coerce(arena, c2, &c_ty, &Type::Bool, spans[c], &infer.named_types)?;
            let (result_ty, t_row, e_row, t2, e2) = match mode {
                // Today's exact behavior, unchanged: infer both branches
                // independently, try unify first (resolves open type
                // variables when possible), and fall back to widen-to-Dyn
                // on genuine failure, never a new rejection -- see this
                // plan's own Global Constraints.
                Mode::Synth => {
                    let (t_ty, t_row, t2) = elaborate(arena, t, ctx, spans, infer)?;
                    let (e_ty, e_row, e2) = elaborate(arena, e, ctx, spans, infer)?;
                    let result_ty = match unify_trial(&t_ty, &e_ty, infer, spans[expr]) {
                        Ok(()) => infer.resolve_deep(&t_ty),
                        Err(_) => Type::Dyn,
                    };
                    (result_ty, t_row, e_row, t2, e2)
                }
                // New: both branches are checked directly against the
                // SAME already-known `expected` type instead of being
                // inferred independently and reconciled against EACH
                // OTHER -- this lets two branches that wouldn't mutually
                // unify under Synth (e.g. one carries an unbound index
                // variable, the other a concrete/differently-shaped type)
                // still both type-check, each against `expected`
                // directly, whenever `expected` itself is something both
                // individually satisfy. A genuine mismatch is now a real
                // error (via check_against's own coerce failure) rather
                // than a silent widen-to-Dyn -- intentional, ordinary
                // Check-mode subsumption, matching every other Check-mode
                // site in this file (e.g. Expr::App's argument check).
                Mode::Check(expected) => {
                    let (t_row, t2) = check_against(arena, t, expected, ctx, spans, infer)?;
                    let (e_row, e2) = check_against(arena, e, expected, ctx, spans, infer)?;
                    (expected.clone(), t_row, e_row, t2, e2)
                }
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
            // Refinement (spec section 5) only ever applies in Check mode
            // -- Synth mode's own escape-variable risk is exactly what
            // Phase 3's bidirectional retrofit was built to avoid (spec
            // section 6's own rationale) -- and only when the scrutinee's
            // own resolved type is genuinely Type::Indexed with a bare
            // index variable (a literal or compound index has no single
            // name to hypothesize about). `index_snapshot` is only ever
            // Some when refinement is eligible at all, so an ordinary,
            // non-Vec match pays no extra cost.
            let resolved_scrut_ty = infer.resolve_deep(&scrut_ty);
            let resolved_index = match &resolved_scrut_ty {
                Type::Indexed(_, idx) => Some(infer.resolve_index_deep(idx)),
                _ => None,
            };
            let refinement_target = match mode {
                Mode::Check(_) => case_a_refinement_target(&resolved_scrut_ty, resolved_index.as_ref()),
                Mode::Synth => None,
            };
            let index_snapshot = refinement_target.as_ref().map(|_| infer.index_subst.clone());
            let mut row = scrut_row;
            let mut result_ty: Option<Type> = None;
            let mut new_arms = Vec::with_capacity(arms.len());
            for (pat, (_, guard, body)) in pats.iter().copied().zip(arms.iter()) {
                if !pattern_could_match(pat, &scrut_ty, &infer.named_types, &HashSet::new()) {
                    return Err(TypeError(
                        format!(
                            "match: pattern of type {} can never match scrutinee of type {scrut_ty}",
                            pattern_type(pat)
                        ),
                        spans[expr],
                    ));
                }
                let mut arm_ctx = bind_pattern_vars(ctx, pat, &scrut_ty, infer, spans[expr])?;
                // Case A hypothesis injection (spec section 5): base case
                // ([]) implies the scrutinee's own index is 0; step case
                // (x :: xs, xs a BARE Var -- see this plan's own v1 scope
                // note) mints a fresh index variable, re-types `xs` as
                // Vec(fresh) (overriding bind_pattern_vars's own plain,
                // unindexed binding via a second `extend` call), and
                // hypothesizes the scrutinee's own index is `fresh + 1`.
                // Injecting into infer.index_subst (not a separate
                // ctx-rewrite pass) is sufficient: every Type::Indexed
                // comparison this arm's own body-check can reach --
                // unify, unify_index_expr, and transitively unify_fits/
                // check_against -- already resolves through index_subst
                // on demand (Phase 1-3), so an ambient binding sharing
                // this same index variable, or `expected` itself if it
                // mentions it, sees the hypothesis too, with no extra
                // machinery.
                if let Some((n_name, elem_ty)) = &refinement_target {
                    match pat {
                        Pattern::List(items) if items.is_empty() => {
                            infer.index_subst.insert(n_name.clone(), IndexExpr::Lit(0));
                        }
                        Pattern::Cons(_, tail) => {
                            if let Pattern::Var(tail_name) = &**tail {
                                let m = fresh_index_name("m");
                                let hypothesis = IndexExpr::Add(Rc::new(IndexExpr::Var(m.clone())), Rc::new(IndexExpr::Lit(1)));
                                infer.index_subst.insert(n_name.clone(), hypothesis);
                                let refined_tail_ty = Type::Indexed(Rc::new(Type::List(elem_ty.clone())), Rc::new(IndexExpr::Var(m)));
                                arm_ctx = extend(&arm_ctx, tail_name, refined_tail_ty);
                            }
                            // A compound/nested tail pattern: no override,
                            // no hypothesis -- v1 scope boundary, see this
                            // plan's own Global Constraints.
                        }
                        _ => {}
                    }
                }
                // Same Bool coercion as If's own cond -- a Dyn-typed guard
                // gets a runtime is_bool check inserted, same as
                // everywhere else Dyn meets an expected concrete type.
                let guard2 = match guard {
                    Some(g) => {
                        let (guard_ty, guard_row, g2) = elaborate(arena, *g, &arm_ctx, spans, infer)?;
                        let g3 = coerce(arena, g2, &guard_ty, &Type::Bool, spans[*g], &infer.named_types)?;
                        row = EffectRow::union(&row, &guard_row);
                        Some(g3)
                    }
                    None => None,
                };
                // Same If-style split as Expr::If's own Check-mode arm:
                // Synth infers each arm independently and reconciles them
                // against EACH OTHER via unify_trial (today's exact,
                // unchanged behavior); Check instead checks each arm's
                // body directly against the SAME already-known `expected`
                // type, so arms that wouldn't mutually unify under Synth
                // can still both type-check as long as each individually
                // satisfies `expected`. No per-arm reconciliation is
                // needed in Check mode -- `expected` is already the
                // answer -- so `result_ty` is simply never touched there.
                let (arm_row, body2) = match mode {
                    Mode::Synth => {
                        let (arm_ty, arm_row, body2) = elaborate(arena, *body, &arm_ctx, spans, infer)?;
                        result_ty = Some(match result_ty {
                            None => arm_ty,
                            Some(t) => match unify_trial(&t, &arm_ty, infer, spans[expr]) {
                                Ok(()) => infer.resolve_deep(&t),
                                Err(_) => Type::Dyn,
                            },
                        });
                        (arm_row, body2)
                    }
                    Mode::Check(expected) => check_against(arena, *body, expected, &arm_ctx, spans, infer)?,
                };
                row = EffectRow::union(&row, &arm_row);
                new_arms.push((pat.clone(), guard2, body2));
            }
            // Restore, once, only after the WHOLE arm loop completes --
            // never per-arm. This is the ONE deliberate exception to
            // infer.index_subst's otherwise strictly monotonic, never-
            // shrinks character (see InferCtx.index_subst's own doc
            // comment and unify_trial's identical snapshot/restore idiom,
            // which this mirrors): a per-arm hypothesis is only ever
            // valid while THAT arm's own body is being checked, and must
            // not leak into a later arm (which may need an incompatible
            // hypothesis for the SAME index variable) or into code after
            // the match entirely.
            if let Some(snapshot) = index_snapshot {
                infer.index_subst = snapshot;
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
            let final_ty = match mode {
                Mode::Synth => result_ty.unwrap_or(Type::Dyn),
                Mode::Check(expected) => expected.clone(),
            };
            Ok((final_ty, row, arena.push(Expr::Match(scrutinee2, Rc::new(new_arms)))))
        }

        // Constructing the handler value is pure -- the clause body's own
        // effects (including what `resume` re-enters) aren't modeled here;
        // see the doc comment on `elaborate_mode`.
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

// Delegates to check_with_named_types with an empty registry -- kept as
// a separate signature (rather than adding a 4th parameter to every
// caller) so the many existing callers that never use self-referential
// type aliases are unaffected. Note this means pairing `check` with
// plain `parser::parse` on a program using a self-referential type
// alias hits every Type::Named consumer with nothing registered -- see
// coerce()'s own doc comment on why that's a real, reachable condition
// each consumer now degrades gracefully for, rather than an internal bug.
pub fn check(arena: &mut Arena, root: ExprRef, spans: &SpanMap) -> Result<ExprRef, TypeError> {
    check_with_named_types(arena, root, spans, HashMap::new())
}

// Same as `check`, but resolves a Type::Named reference against
// `named_types` (parser::parse_with_named_types's own registry)
// instead of an empty one. `check`'s own signature stays untouched --
// every existing caller that never uses self-referential type aliases
// is unaffected; this exists so the real production path
// (lib.rs's own run_source_on_this_thread) can support the feature.
pub fn check_with_named_types(arena: &mut Arena, root: ExprRef, spans: &SpanMap, named_types: HashMap<String, Type>) -> Result<ExprRef, TypeError> {
    let mut infer = InferCtx::new(named_types);
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
