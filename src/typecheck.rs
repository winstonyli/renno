use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::expr::{Arena, BinOp, CheckMode, CheckSpec, Expr, ExprRef, Pattern, SpanMap, Test};
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
    // Index variables that unify_index_expr must NOT bind: the free index
    // variables of an annotated binding's signature while its value is
    // checked (spec 2026-10-07). Hypotheses injected by Match write
    // index_subst directly, so they are unaffected.
    pub(crate) rigid_index: HashSet<String>,
    // Annotation index-variable names renamed while an annotated binding's
    // value is checked (open_signature): written name -> fresh name. Applied
    // to every annotation read inside that value (scoped_annotation), so a
    // body annotation reusing the name denotes the signature's variable.
    index_rename: HashMap<String, String>,
    // Runtime values for index variables, innermost last: while a Lambda
    // whose parameter is Vec(n) is elaborated, `n` is that parameter's
    // length. A Dyn-to-Vec(n) boundary check inside reads `len(hidden)` of a
    // hidden alias (`hidden = param`), which the Lambda inserts at its entry
    // only if some check used it. See index_var_to_expr. Pushed and popped
    // by elaborate_mode's Lambda frames; an elaboration error abandons the
    // whole check, so an Err path never needs to pop.
    index_witness: Vec<IndexWitness>,
    // Dyn-to-Vec(n) crossings whose length check needs an index variable
    // that nothing at runtime holds at the crossing (no witness, no static
    // binding yet): a later annotation may still bind it. See Obligation.
    // A RefCell because the check builders only see a shared &InferCtx
    // (CheckEnv), the same reason IndexWitness.used is a Cell.
    obligations: RefCell<Vec<Obligation>>,
    // How many function bodies (Lambda, handler clause) enclose the point
    // being elaborated: code inside one can run any number of times.
    repeatable_depth: usize,
    // Every genuinely self-referential type alias this program's own
    // parse registered (parser::Parser's own `named_types`, handed in
    // once at construction) -- consulted on demand by
    // build_boundary_check/build_shape_predicate/pattern_could_match
    // whenever they encounter a Type::Named reference, to unfold it
    // exactly one level. Never mutated after construction -- this
    // registry is a static, parse-time-complete fact about the
    // program, unlike `subst`.
    named_types: HashMap<String, Type>,
    // Names of type variables whose value reaches a Dyn sink (the argument of
    // a Dyn callee, a Perform payload) while the variable is still unbound
    // (spec 2026-10-08-var-passthrough-casts). The Fun-callee App arm casts a
    // typed function argument to Dyn when the callee's parameter is such a
    // variable. Grows like `subst` and is snapshotted/restored with it by
    // unify_trial (a failed trial must not leak flags); propagated by
    // `lookup` and `unify`.
    dyn_sunk: HashSet<String>,
}

impl InferCtx {
    pub(crate) fn new(named_types: HashMap<String, Type>) -> InferCtx {
        InferCtx {
            subst: HashMap::new(),
            index_subst: HashMap::new(),
            rigid_index: HashSet::new(),
            index_rename: HashMap::new(),
            index_witness: Vec::new(),
            obligations: RefCell::new(Vec::new()),
            repeatable_depth: 0,
            named_types,
            dyn_sunk: HashSet::new(),
        }
    }

    // Flags `ty` as reaching a Dyn sink if it is (still) a bare type variable.
    fn note_dyn_sink(&mut self, ty: &Type) {
        if let Type::Var(v) = self.resolve(ty) {
            self.dyn_sunk.insert(v);
        }
    }

    fn is_dyn_sunk(&self, ty: &Type) -> bool {
        !self.dyn_sunk.is_empty() && matches!(self.resolve(ty), Type::Var(v) if self.dyn_sunk.contains(&v))
    }

    #[cfg(test)]
    pub(crate) fn obligation_count(&self) -> usize {
        self.obligations.borrow().len()
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

// Outcome of asking whether a rigid index variable's equation holds.
// SOP normalization over index polynomials either proves two sides equal
// or refutes them, so `Unknown` is unreachable today; it exists as the
// marked place for the future gradual runtime check (spec 2026-10-07,
// "Potential future step").
enum RigidEq {
    Equal,
    Unequal,
    #[allow(dead_code)]
    Unknown,
}

fn classify_rigid_eq(a: &IndexExpr, b: &IndexExpr) -> RigidEq {
    if crate::index_expr::index_exprs_equal(a, b) {
        RigidEq::Equal
    } else {
        RigidEq::Unequal
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
        // A flexible variable yields to a rigid one: bind the flexible side,
        // never the rigid one. (The or-pattern arm below would otherwise
        // pick the left variable as the one to bind.)
        (IndexExpr::Var(rigid), IndexExpr::Var(flex)) if infer.rigid_index.contains(rigid) && !infer.rigid_index.contains(flex) => {
            infer.index_subst.insert(flex.clone(), IndexExpr::Var(rigid.clone()));
            Ok(())
        }
        (IndexExpr::Var(name), other) | (other, IndexExpr::Var(name)) => {
            // A rigid variable (still unbound: both sides are already
            // resolved) may only be equated, never bound.
            if infer.rigid_index.contains(name) {
                return match classify_rigid_eq(&a, &b) {
                    RigidEq::Equal => Ok(()),
                    RigidEq::Unequal | RigidEq::Unknown => Err(TypeError(
                        format!("index variable {name} is fixed by the signature and cannot equal {other}"),
                        span,
                    )),
                };
            }
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
            // Two variables being equated share one flag (a hop's instance
            // aliasing the caller's variable carries the sink back to it).
            if let Type::Var(o) = other
                && (infer.dyn_sunk.contains(name) || infer.dyn_sunk.contains(o))
            {
                infer.dyn_sunk.insert(name.clone());
                infer.dyn_sunk.insert(o.clone());
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
    // The Dyn-sink flags travel with variable aliasing in unify(), so a
    // failed trial must drop the flags it spread as well.
    let sunk_snapshot = infer.dyn_sunk.clone();
    let result = unify(t1, t2, infer, span);
    if result.is_err() {
        infer.subst = snapshot;
        infer.index_subst = index_snapshot;
        infer.dyn_sunk = sunk_snapshot;
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
            // An instance of a quantified variable that reaches a Dyn sink
            // reaches it too.
            if !infer.dyn_sunk.is_empty() {
                for (v, fresh) in &type_subst {
                    if infer.dyn_sunk.contains(v) {
                        infer.note_dyn_sink(fresh);
                    }
                }
            }
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
//
// `pub(crate)`, not private: Task 3's own Case B tests need to bind a
// scrutinee variable directly to a hand-built `Type::Indexed(Type::
// Named(id), _)` -- a shape no surface syntax can express (`Vec(n)`'s
// own parser sugar is hard-coded to wrap `Type::List(Dyn)` only) -- and
// drive `check_against` on it directly, mirroring this file's own
// existing `extend_generalized_and_lookup_mint_a_fresh_index_variable...`
// test's identical precedent for exercising typecheck's own internals
// below the parser's reach. Same progressive, only-as-far-as-needed
// visibility widening this project already uses repeatedly.
pub(crate) fn extend(ctx: &Ctx, name: &str, ty: Type) -> Ctx {
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
    collect_index_vars(ty, &|e| infer.resolve_index_deep(e))
}

// The index-variable names `ty` mentions as written, NOT resolved through
// index_subst (open_signature needs to see the already-bound ones).
fn index_vars_as_written(ty: &Type) -> BTreeSet<String> {
    collect_index_vars(ty, &|e| e.clone())
}

fn collect_index_vars(ty: &Type, resolve: &dyn Fn(&IndexExpr) -> IndexExpr) -> BTreeSet<String> {
    match ty {
        Type::Indexed(wrapped, index) => {
            let mut vars = collect_index_vars(wrapped, resolve);
            vars.extend(free_index_vars(&resolve(index)));
            vars
        }
        Type::Fun(param, _row, ret) => {
            let mut vars = collect_index_vars(param, resolve);
            vars.extend(collect_index_vars(ret, resolve));
            vars
        }
        Type::List(elem) => collect_index_vars(elem, resolve),
        Type::Tuple(items) | Type::Union(items) => items.iter().flat_map(|t| collect_index_vars(t, resolve)).collect(),
        Type::Record(fields) => fields.iter().flat_map(|(_, t)| collect_index_vars(t, resolve)).collect(),
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
    let mut still_open = free_index_vars_in_ctx(ctx, infer);
    // A variable a pending obligation still needs is decided by whoever binds
    // it later, inside the function containing the crossing. Generalizing it
    // would give each use a fresh copy and leave the obligation unresolvable
    // (spec 2026-10-07-dyn-crossing-deferred-length-checks, polymorphism rule).
    still_open.extend(pending_obligation_vars(infer));
    candidates.difference(&still_open).cloned().collect()
}

fn pending_obligation_vars(infer: &InferCtx) -> BTreeSet<String> {
    infer.obligations.borrow().iter().flat_map(|ob| ob.unresolved_vars(infer)).collect()
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
fn coerce(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, span: Span, env: &CheckEnv) -> Result<ExprRef, TypeError> {
    let checked = coerce_check(arena, e, from, to, span, env)?;
    Ok(coerce_cast(arena, checked, from, to, &env.with_span(span)))
}

// The static consistency check plus the Dyn-to-concrete boundary check. A
// Fun source is returned unchanged; coerce_cast wraps it separately.
fn coerce_check(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, span: Span, env: &CheckEnv) -> Result<ExprRef, TypeError> {
    let named_types = &env.infer.named_types;
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
        // Indexing syntax parses on any alias (parser.rs, Case B), so a
        // non-eligible one only fails here -- name that, not a bare mismatch.
        if let Type::Indexed(wrapped, _) = to {
            let indexable = match &**wrapped {
                Type::List(_) | Type::Var(_) | Type::Dyn => true,
                Type::Named(id) => qualifying_named_alternatives(id, named_types).is_some(),
                _ => false,
            };
            if !indexable {
                return Err(TypeError(format!("{wrapped} is not an indexable type"), span));
            }
        }
        return Err(TypeError(format!("type mismatch: expected {to}, found {from}"), span));
    }
    if !matches!(from, Type::Dyn | Type::Var(_)) || *to == Type::Dyn || matches!(to, Type::Var(_)) {
        return Ok(e);
    }
    Ok(build_boundary_check(arena, e, to, &env.with_span(span), &HashSet::new()))
}

// `t` with every Indexed whose index mentions a variable `has_value` rejects
// reduced to the type it wraps, so build_boundary_check never needs a witness
// it cannot get. Deliberate extension: a non-bare index (Vec(n+1)) keeps its
// length compare when every variable in it is witnessed in scope; otherwise
// the check degrades to is_list only (see build_cast).
fn erase_open_indexed(t: &Type, has_value: &dyn Fn(&str) -> bool) -> Type {
    let go = |t: &Type| erase_open_indexed(t, has_value);
    match t {
        Type::Indexed(w, i) if free_index_vars(i).iter().all(|v| has_value(v)) => Type::Indexed(Rc::new(go(w)), i.clone()),
        Type::Indexed(w, _) => go(w),
        Type::List(e) => Type::List(Rc::new(go(e))),
        Type::Tuple(ts) => Type::Tuple(Rc::new(ts.iter().map(go).collect())),
        Type::Union(ts) => Type::Union(Rc::new(ts.iter().map(go).collect())),
        Type::Record(fs) => Type::Record(Rc::new(fs.iter().map(|(n, t)| (n.clone(), go(t))).collect())),
        Type::Fun(p, r, b) => Type::Fun(Rc::new(go(p)), r.clone(), Rc::new(go(b))),
        other => other.clone(),
    }
}

// Structural equality that ignores effect rows (no runtime row representation).
// Intentionally Fun/structure-only with an == fallback. coerce_cast uses it as
// a cheap identity guard so an equal-up-to-rows cast skips a redundant DOWN
// (needs_wrapper already answers false for equal concrete types).
fn same_ignoring_rows(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Fun(p, _, r), Type::Fun(p2, _, r2)) => same_ignoring_rows(p, p2) && same_ignoring_rows(r, r2),
        (Type::List(x), Type::List(y)) => same_ignoring_rows(x, y),
        (Type::Tuple(xs), Type::Tuple(ys)) | (Type::Union(xs), Type::Union(ys)) => xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x, y)| same_ignoring_rows(x, y)),
        (Type::Record(xs), Type::Record(ys)) => xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|((n, x), (m, y))| n == m && same_ignoring_rows(x, y)),
        (Type::Indexed(w, i), Type::Indexed(w2, i2)) => i == i2 && same_ignoring_rows(w, w2),
        _ => a == b,
    }
}

// Wraps a Fun-typed `e` so that, seen as `to` (Dyn or a Fun), its argument is
// re-checked against the parameter type it was really declared with and a
// returned function is wrapped in turn. Closure form:
// `let __cf#k = e in fun __ca#k: A' -> UP(__cf#k(DOWN(__ca#k)))`, fresh Var
// nodes and fresh unlexable binder names each time (cast_binder). The body runs once per call (repeatable) and records no obligations.
// Returns `e` unchanged when no wrapper is needed.
fn coerce_cast(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, env: &CheckEnv) -> ExprRef {
    let from = env.infer.resolve_deep(from);
    let to = env.infer.resolve_deep(to);
    // A cast from a type to itself is the identity. (A literal cast keeps the
    // Lambda a Lambda, so an annotated `let rec` binding stays a direct group
    // either way; this guard just avoids a redundant DOWN.)
    if same_ignoring_rows(&from, &to) || !needs_wrapper(&from, &to) {
        return e;
    }
    if matches!(arena[e], Expr::Lambda(..)) {
        return build_literal_cast(arena, e, &from, &to, env);
    }
    build_cast(arena, e, &from, &to, env)
}

// Casts each Synth-joined arm from its own type to the FINAL joined type, so a
// typed function that reaches a Dyn (or partly-Dyn Fun) join keeps its
// parameter checks. Call once, after the whole join: a later arm can still
// change the join, and an arm that unified with an already-Dyn join needs the
// cast too. Infallible; non-function and equal-typed arms come back unchanged.
fn cast_join_arms(arena: &mut Arena, arms: &mut [ExprRef], tys: &[Type], join: &Type, infer: &InferCtx) {
    for (arm, ty) in arms.iter_mut().zip(tys) {
        *arm = coerce_cast(arena, *arm, ty, join, &CheckEnv::new(infer));
    }
}

// Binder names a cast introduces (`__cf`, `__ca`): fresh and unlexable
// (`__ca#7`), so neither a user variable nor a nested cast's own binder can
// capture or shadow them. Shared by build_cast and build_literal_cast.
fn cast_binder(base: &str) -> String {
    fresh_index_name(base)
}

// The (parameter, result) types a cast aims at: those of a Fun target, Dyn for
// both when the target is Dyn.
fn cast_targets(to: &Type) -> (Type, Type) {
    match to {
        Type::Fun(a2, _, b2) => ((**a2).clone(), (**b2).clone()),
        _ => (Type::Dyn, Type::Dyn),
    }
}

// Is index variable `v` already bound by a witness in scope (a local one, or
// one the inference recorded)?
fn index_witnessed<'a>(env: &'a CheckEnv<'a>) -> impl Fn(&str) -> bool + 'a {
    move |v| env.local.iter().any(|w| w.var == v) || env.infer.index_witness.iter().any(|w| w.var == v)
}

// The literal form of build_cast for a Lambda `e` (spec 2026-10-08 sec 5):
// the Lambda is rebuilt in place,
// `fun __ca#k: A' -> let p = DOWN(__ca#k) in UP(body)`,
// so an annotated `let rec` binding stays a direct Lambda (is_direct_group)
// and no closure is allocated. A witness binder (`let hidden = p in ...`) at
// the top of the body is kept, and UP recurses into a Lambda body. Here a
// parameter's index variables are never witnessed: DOWN is is_list only for
// them, unless the variable is already witnessed by an enclosing scope
// (parked, spec sec 7).
fn build_literal_cast(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, env: &CheckEnv) -> ExprRef {
    let Expr::Lambda(param, _, body) = arena[e].clone() else { unreachable!("build_literal_cast takes a Lambda") };
    let Type::Fun(a, _, b) = from else { unreachable!("build_literal_cast takes a Fun source") };
    let (a_target, b_target) = cast_targets(to);
    let witnessed = index_witnessed(env);
    let up_body = if needs_upcast(b, &b_target) { literal_up(arena, body, b, &b_target, env) } else { body };
    if !needs_down(a, &a_target) {
        return arena.push(Expr::Lambda(param, Some(a_target), up_body));
    }
    // The incoming value arrives under a fresh name and is checked outside the
    // user's parameter scope, so a parameter named like a builtin the check
    // calls (is_int, fail, type_name, ...) cannot capture it.
    let ca = cast_binder("__ca");
    let ca_ref = arena.push(Expr::Var(ca.clone()));
    let checked = build_boundary_check(arena, ca_ref, &erase_open_indexed(a, &witnessed), env, &HashSet::new());
    let body = arena.push(Expr::Let(param, None, checked, up_body));
    arena.push(Expr::Lambda(ca, Some(a_target), body))
}

// UP for a literal's body: a Lambda body is rebuilt in turn (through a witness
// binder `let hidden = <var> in ...`; the arm matches any `let y = a in ...`
// too, which is harmless); any other body is wrapped in the
// closure form.
fn literal_up(arena: &mut Arena, body: ExprRef, b: &Type, b_target: &Type, env: &CheckEnv) -> ExprRef {
    match arena[body].clone() {
        Expr::Lambda(..) => build_literal_cast(arena, body, b, b_target, env),
        Expr::Let(hidden, None, val, inner) if matches!(arena[val], Expr::Var(_)) => {
            let inner = literal_up(arena, inner, b, b_target, env);
            arena.push(Expr::Let(hidden, None, val, inner))
        }
        _ => build_cast(arena, body, b, b_target, env),
    }
}

// The closure form of coerce_cast for a Fun `from` (needs_wrapper holds).
// Every binder is fresh (cast_binder), so a returned function's wrapper, nested
// inside this one's lambda, never shadows it. A Vec(n) parameter whose `n` has
// no witness in scope binds one from `__ca#k` (DOWN then needs only is_list:
// the length half would be `len(__ca#k) == n`, a tautology); a variable already
// witnessed in scope is never rebound, so a nested wrapper compares against
// the outer value. Any other open index degrades to is_list.
// Known limitation (spec 2026-10-08 sec 7): a monomorphic `n` that no witness
// in scope covers (e.g. `n` fixed by an earlier Dyn-to-Vec(n) check, then
// `let f = fun w: Vec(n) -> .. in let d: Dyn = f in d([1])`) gets a fresh
// per-call witness, so DOWN checks list shape only, while the direct call
// `f([1])` is rejected. Pinned by a test.
fn build_cast(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, env: &CheckEnv) -> ExprRef {
    let Type::Fun(a, _, b) = from else { unreachable!("build_cast takes a Fun source") };
    let (a_target, b_target) = cast_targets(to);
    let cf = cast_binder("__cf");
    let ca = cast_binder("__ca");
    let witnessed = index_witnessed(env);
    let witness = list_length_var(a, env.infer).filter(|v| !witnessed(v)).map(|var| IndexWitness::new(var, &ca));
    let mut local = env.local.clone();
    local.extend(witness.as_ref());
    let body_env = CheckEnv { infer: env.infer, local, defer: false, repeatable: true, span: env.span };
    let ca_ref = arena.push(Expr::Var(ca.clone()));
    let arg = if needs_down(a, &a_target) {
        build_boundary_check(arena, ca_ref, &erase_open_indexed(a, &witnessed), &body_env, &HashSet::new())
    } else {
        ca_ref
    };
    let cf_ref = arena.push(Expr::Var(cf.clone()));
    let call = arena.push(Expr::App(cf_ref, arg));
    let result = if needs_upcast(b, &b_target) { build_cast(arena, call, b, &b_target, &body_env) } else { call };
    let body = match &witness {
        Some(w) if w.used.get() => w.bind(arena, &ca, result),
        _ => result,
    };
    let lambda = arena.push(Expr::Lambda(ca, Some(a_target), body));
    arena.push(Expr::Let(cf, None, e, lambda))
}
// True iff `t` has no Dyn, Var, Union or free index variable at any depth:
// a type a strict (non-gradual) `fits` check can be trusted on. A Union
// counts as loose because `fits(Int, Int | Bool)` holds. Named and Token
// are tolerated (nominal leaves). All take resolve_deep'd types.
pub(crate) fn loose_free(t: &Type) -> bool {
    match t {
        Type::Dyn | Type::Var(_) | Type::Union(_) => false,
        Type::List(elem) => loose_free(elem),
        Type::Fun(p, _, r) => loose_free(p) && loose_free(r),
        Type::Tuple(items) => items.iter().all(loose_free),
        Type::Record(fields) => fields.iter().all(|(_, t)| loose_free(t)),
        Type::Indexed(wrapped, _) => loose_free(wrapped) && index_vars_as_written(t).is_empty(),
        _ => true,
    }
}

// `a_src` statically fits `a_target` and nothing about the target is loose.
// Argument order mirrors fits' param contravariance: fits(act_param, req_param).
pub(crate) fn strict_fits(a_target: &Type, a_src: &Type) -> bool {
    fits(a_src, a_target) && loose_free(a_target)
}

// Must a value of type `a_src` be re-checked at runtime to be used as `a_target`?
pub(crate) fn needs_down(a_src: &Type, a_target: &Type) -> bool {
    !matches!(a_src, Type::Dyn | Type::Var(_)) && !strict_fits(a_target, a_src)
}

// Does a return value of type `b` need a wrapper to be seen as `b_target`?
// Recurses only into a bare Fun return; never Named, Union, Tuple, Record, List.
fn needs_upcast(b: &Type, b_target: &Type) -> bool {
    matches!(b, Type::Fun(..)) && needs_wrapper(b, b_target)
}

// Does casting `from` (a Fun) to `to` (Dyn, read as Fun(Dyn, _, Dyn), or a
// Fun) need a wrapper around the function value?
pub(crate) fn needs_wrapper(from: &Type, to: &Type) -> bool {
    let Type::Fun(a, _, b) = from else { return false };
    match to {
        Type::Dyn => needs_down(a, &Type::Dyn) || needs_upcast(b, &Type::Dyn),
        Type::Fun(a2, _, b2) => needs_down(a, a2) || needs_upcast(b, b2),
        _ => false,
    }
}

// Like `coerce`, but the target is "Int or Float" rather than one fixed
// type -- Add/Sub/Mul/Div/Mod/Lt's own operand check. Kept separate from
// the general coerce()/consistent() machinery ON PURPOSE: Int and Float
// stay mutually INCONSISTENT everywhere else in the type system (a
// [Float]-annotated parameter still statically rejects a [Int] argument,
// a Fun's declared Int parameter still rejects a Float argument, etc.) --
// only these specific operators treat the two as interchangeable, so this
// local helper is where that interchangeability lives. The Dyn-boundary
// runtime check is one native Int-or-Float Check: it only asks "does this
// runtime value look like one of these shapes," which carries no
// implication for consistent() or any other static check.
fn coerce_numeric(arena: &mut Arena, e: ExprRef, ty: &Type, span: Span, env: &CheckEnv) -> Result<ExprRef, TypeError> {
    match ty {
        Type::Int | Type::Float => Ok(e),
        Type::Dyn | Type::Var(_) => {
            let numeric = Type::Union(Rc::new(vec![Type::Int, Type::Float]));
            Ok(build_check_node(arena, e, Test::Or(vec![Test::Int, Test::Float]), CheckMode::Assert, &numeric, &env.with_span(span)))
        }
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
// Every shallow shape (Int/Float/Bool/Str/List/Token/Tuple arity/Record
// fields/a union of those, a Named one unfolded) is ONE native Expr::Check
// node (build_shape_check): no let, no env frame, no builtin call. Fun gets
// a real per-call contract (wrap_fun_contract, since a bare callability tag
// can't see inside a closure -- "callable" isn't "callable with this
// exact signature"). What a Check can't express stays desugared: a Vec(n)
// length compare (it evaluates an index expression in the environment, so
// it keeps the `let tmp = e in if cond then tmp else <failing Check>`
// skeleton) and a union with a Fun or Vec(n) alternative (build_union_check,
// which picks the alternative with Probe checks). Every failure panics with
// the same "type error: expected X, found Y" text.
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
fn build_boundary_check(arena: &mut Arena, e: ExprRef, to: &Type, env: &CheckEnv, visiting: &HashSet<String>) -> ExprRef {
    match to {
        Type::Int | Type::Float | Type::Bool | Type::Str | Type::List(_) | Type::Token(_) | Type::Tuple(_) | Type::Record(_) => {
            build_shape_check(arena, e, to, env, visiting)
        }
        Type::Fun(param_ty, _row, ret_ty) => wrap_fun_contract(arena, e, param_ty.clone(), ret_ty.clone(), env, visiting),
        Type::Union(_) if shape_test(to, env, visiting).is_none() || needs_contract(to, env, visiting) => {
            build_union_check(arena, e, to, env, visiting)
        }
        Type::Union(_) => build_shape_check(arena, e, to, env, visiting),
        // Dyn accepts everything, so a check against it is the identity. Not
        // unreachable: wrap_fun_contract checks a call's result against the
        // Fun's return type, which is Dyn for `Dyn -> Dyn` (and nested
        // builders reach it through tuple/union alternatives).
        Type::Dyn | Type::Var(_) => e,
        // is-a-list-then-len-equals-the-index, checked against the index
        // EXPRESSION (not a fixed arity literal), so this keeps the let/if
        // skeleton: binding `e` once is what keeps it evaluated exactly
        // once. The list test has to run BEFORE `len`: `len` panics
        // internally (see machine.rs) on a non-list argument, which would
        // surface as an unrelated bare panic instead of this function's own
        // clean "type error: expected ..., found ..." message.
        //
        // Only Type::List-wrapped Indexed types are reachable this early
        // -- a Type::Named-wrapped Indexed (derived index-refinement) is
        // a later phase's own concern, nothing before this phase can
        // construct one yet (see index_expr_to_expr's own doc comment
        // for the matching "Phase 2 only" note on the index side).
        Type::Indexed(wrapped, index) => match wrapped.as_ref() {
            Type::List(_) => {
                let tmp = fresh_index_name("__check_tmp");
                let tmp_ref = arena.push(Expr::Var(tmp.clone()));
                let (cond, len_eq) = build_indexed_shape_cond(arena, tmp_ref, index, env);
                if env.defer {
                    record_obligation(len_eq, index, env);
                }
                let fail = build_type_error(arena, tmp_ref, to, env);
                let if_expr = arena.push(Expr::If(cond, tmp_ref, fail));
                arena.push(Expr::Let(tmp, None, e, if_expr))
            }
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
        // than an unconditional failing check. Two different callers reach this
        // branch, and only ONE of them makes a failure safe: build_union_check's
        // own use is always gated behind build_shape_predicate's matching
        // arm, which hard-codes `false` for this exact alternative (see
        // its doc comment) -- so build_union_check's own `If(pred, checked,
        // ...)` can never actually select this branch at runtime, and a
        // failure there was always dead code no matter what sat in it.
        // wrap_fun_contract's own call (a self-reference in a Fun's
        // RETURN position, e.g. `type F = Dyn -> F`) is NOT gated by any
        // predicate -- it's spliced directly into a contract-wrapped
        // closure's own body, so a failure there ran unconditionally on
        // EVERY call through that closure, even when the real return
        // value never needed checking at all. `e` unchecked is correct
        // for both: harmless where it's dead code anyway, and the right
        // "checked later, lazily" answer where it actually runs.
        Type::Named(id) => {
            if visiting.contains(id) {
                return e;
            }
            let Some(raw) = env.infer.named_types.get(id) else { return e };
            let mut visiting = visiting.clone();
            visiting.insert(id.clone());
            let unfolded = raw.clone();
            build_boundary_check(arena, e, &unfolded, env, &visiting)
        }
    }
}

// A runtime value for an index variable: `hidden` is a variable bound by
// synthesized code to the list itself (a Lambda's parameter, or a
// contract's argument); a check reads `len(hidden)` at the check site, so
// the entry path evaluates nothing (an untyped caller passing a non-list
// is unaffected unless a check actually runs). `used` records whether any
// check read it, so the binder is only emitted when needed.
struct IndexWitness {
    var: String,
    hidden: String,
    used: Cell<bool>,
}

impl IndexWitness {
    fn new(var: String, param: &str) -> IndexWitness {
        IndexWitness { var, hidden: fresh_index_name(&format!("list_{param}")), used: Cell::new(false) }
    }

    // `let hidden = list_var in body`: an alias, immune to later shadowing
    // of the parameter's name.
    fn bind(&self, arena: &mut Arena, list_var: &str, body: ExprRef) -> ExprRef {
        let list_ref = arena.push(Expr::Var(list_var.to_string()));
        arena.push(Expr::Let(self.hidden.clone(), None, list_ref, body))
    }
}

// A Dyn-to-Vec(n) length check whose index has a variable with no runtime
// value at the crossing (no witness, no static binding there); spec
// 2026-10-07-dyn-crossing-deferred-length-checks. `index` is the target
// index with every binding known at the crossing substituted (runtime_index),
// so a Match arm's hypothesis, restored after the arm, stays in force.
// `witnesses` are the (variable, hidden alias) pairs in scope at the
// crossing, outermost first. `len_eq` is the check's `len(value) == <index>`
// node, which resolve_obligations patches in place.
struct Obligation {
    len_eq: ExprRef,
    index: IndexExpr,
    witnesses: Vec<(String, String)>,
    // The crossing sits inside a function body, so one site can produce many
    // values sharing its index variable.
    repeatable: bool,
}

impl Obligation {
    fn witness_vars(&self) -> Vec<&str> {
        self.witnesses.iter().map(|(var, _)| var.as_str()).collect()
    }

    // The index under the bindings known now, as the check would read it.
    fn current_index(&self, infer: &InferCtx) -> IndexExpr {
        runtime_index(&self.index, infer, &self.witness_vars(), true)
    }

    // The variables the check still has no runtime value for.
    fn unresolved_vars(&self, infer: &InferCtx) -> BTreeSet<String> {
        let witnessed = self.witness_vars();
        free_index_vars(&self.current_index(infer)).into_iter().filter(|v| !witnessed.contains(&v.as_str())).collect()
    }
}

// `e` as a runtime check reads it (index_var_to_expr's order): a variable
// with a witness stays, a bound one is replaced by its binding, recursively,
// an unbound one stays. With `by_alias`, an unbound variable that a witness's
// own variable has since been unified with is renamed to that witness's
// variable: the witness holds its value too.
fn runtime_index(e: &IndexExpr, infer: &InferCtx, witness_vars: &[&str], by_alias: bool) -> IndexExpr {
    let go = |x: &IndexExpr| Rc::new(runtime_index(x, infer, witness_vars, by_alias));
    match e {
        IndexExpr::Var(name) => {
            if witness_vars.contains(&name.as_str()) {
                return e.clone();
            }
            if let Some(bound) = infer.index_subst.get(name) {
                return runtime_index(bound, infer, witness_vars, by_alias);
            }
            let alias = if by_alias {
                witness_vars.iter().rev().find(|w| infer.resolve_index_deep(&IndexExpr::Var(w.to_string())) == *e)
            } else {
                None
            };
            match alias {
                Some(w) => IndexExpr::Var(w.to_string()),
                None => e.clone(),
            }
        }
        IndexExpr::Lit(_) => e.clone(),
        IndexExpr::Add(a, b) => IndexExpr::Add(go(a), go(b)),
        IndexExpr::Sub(a, b) => IndexExpr::Sub(go(a), go(b)),
        IndexExpr::Mul(a, b) => IndexExpr::Mul(go(a), go(b)),
    }
}

// Records an obligation for a crossing outside a union when its index has a
// variable index_var_to_expr cannot give a runtime value (the check built
// for it holds that clean failure until resolved). Every witness in scope
// is marked used: the resolved check may read any of them, and their alias
// binders are emitted (or not) before typechecking finishes.
fn record_obligation(len_eq: ExprRef, index: &IndexExpr, env: &CheckEnv) {
    let in_scope: Vec<&IndexWitness> = env.infer.index_witness.iter().chain(env.local.iter().copied()).collect();
    let vars: Vec<&str> = in_scope.iter().map(|w| w.var.as_str()).collect();
    let index = runtime_index(index, env.infer, &vars, false);
    if free_index_vars(&index).iter().all(|v| vars.contains(&v.as_str())) {
        return;
    }
    for w in &in_scope {
        w.used.set(true);
    }
    let witnesses = in_scope.iter().map(|w| (w.var.clone(), w.hidden.clone())).collect();
    env.infer.obligations.borrow_mut().push(Obligation { len_eq, index, witnesses, repeatable: env.infer.repeatable_depth > 0 || env.repeatable });
}

// Patches every obligation's `len(value) == <index>` node once inference has
// finished, so bindings made after the crossing count: the index under the
// final bindings, each variable read from a witness in scope at the
// crossing, otherwise the clean "no runtime value" failure. Runs before
// anything evaluates the tree.
//
// An index that is still a bare unbound variable no other obligation
// mentions, and that sits in no function body, constrains nothing, so the
// check is `is_list` alone (the node becomes `true`). Two crossings sharing
// one such variable would claim equal lengths with nothing to compare
// against, so they keep the clean failure. So does a variable unified with any
// other index variable, in either direction (e.g. inside a match arm whose
// hypothesis was since removed): it is not isolated, and the crossing's own
// index must be unchanged by the bindings.
fn resolve_obligations(arena: &mut Arena, infer: &InferCtx) {
    let obligations = infer.obligations.take();
    let mut mentions: HashMap<String, usize> = HashMap::new();
    for ob in &obligations {
        for var in free_index_vars(&ob.current_index(infer)) {
            *mentions.entry(var).or_insert(0) += 1;
        }
    }
    for ob in obligations {
        let index = ob.current_index(infer);
        let unconstrained = matches!((&index, &ob.index), (IndexExpr::Var(name), IndexExpr::Var(orig))
            if name == orig
                && !ob.repeatable
                && !ob.witness_vars().contains(&name.as_str())
                && mentions.get(name) == Some(&1)
                && !infer.index_subst.values().any(|v| free_index_vars(v).contains(name)));
        if unconstrained {
            arena[ob.len_eq] = Expr::Bool(true);
            continue;
        }
        let len_call = match &arena[ob.len_eq] {
            Expr::BinOp(BinOp::Eq, len_call, _) => *len_call,
            _ => unreachable!("an obligation's len_eq is the BinOp built by build_indexed_shape_cond"),
        };
        let witnesses: Vec<IndexWitness> = ob
            .witnesses
            .iter()
            .map(|(var, hidden)| IndexWitness { var: var.clone(), hidden: hidden.clone(), used: Cell::new(true) })
            .collect();
        let env = CheckEnv { infer, local: witnesses.iter().collect(), defer: false, repeatable: false, span: None };
        let index_ref = index_expr_to_expr(arena, &index, &env);
        arena[ob.len_eq] = Expr::BinOp(BinOp::Eq, len_call, index_ref);
    }
}

// What the runtime-check builders (build_boundary_check and friends) need:
// the type registry and index state (InferCtx, as of where the check is
// spliced); `local` holds witnesses bound inside synthesized code
// (wrap_fun_contract). `defer` is false inside a union target: a Vec(n)
// alternative whose n has no runtime value keeps failing the whole check, so
// it records no obligation.
struct CheckEnv<'a> {
    infer: &'a InferCtx,
    local: Vec<&'a IndexWitness>,
    defer: bool,
    // Building inside a synthesized function body (wrap_fun_contract's
    // wrapper), which runs on every call of the wrapped function.
    repeatable: bool,
    // The cast site a failing native check blames (CheckSpec::span); None
    // where no source span is at hand (a join arm).
    span: Option<Span>,
}

// `n` when `ty` is Vec(n) with `n` (resolved) still a bare variable: the
// variable a list of this type witnesses the value of.
fn list_length_var(ty: &Type, infer: &InferCtx) -> Option<String> {
    match ty {
        Type::Indexed(wrapped, index) if matches!(wrapped.as_ref(), Type::List(_)) => match infer.resolve_index_deep(index) {
            IndexExpr::Var(name) => Some(name),
            _ => None,
        },
        _ => None,
    }
}

impl<'a> CheckEnv<'a> {
    fn new(infer: &'a InferCtx) -> CheckEnv<'a> {
        CheckEnv { infer, local: Vec::new(), defer: true, repeatable: false, span: None }
    }

    // A fresh environment for a cast at source span `span`.
    fn at(infer: &'a InferCtx, span: Span) -> CheckEnv<'a> {
        CheckEnv { span: Some(span), ..CheckEnv::new(infer) }
    }

    // This environment, blaming `span`.
    fn with_span(&self, span: Span) -> CheckEnv<'a> {
        CheckEnv { infer: self.infer, local: self.local.clone(), defer: self.defer, repeatable: self.repeatable, span: Some(span) }
    }
}

// Translates an IndexExpr into an ordinary Expr the machine can evaluate,
// for splicing into a synthesized runtime check. See index_var_to_expr for
// where each variable's runtime value comes from.
fn index_expr_to_expr(arena: &mut Arena, e: &IndexExpr, env: &CheckEnv) -> ExprRef {
    match e {
        IndexExpr::Var(name) => index_var_to_expr(arena, name, env),
        IndexExpr::Lit(n) => arena.push(Expr::Int(*n)),
        IndexExpr::Add(a, b) => {
            let a2 = index_expr_to_expr(arena, a, env);
            let b2 = index_expr_to_expr(arena, b, env);
            arena.push(Expr::BinOp(BinOp::Add, a2, b2))
        }
        IndexExpr::Sub(a, b) => {
            let a2 = index_expr_to_expr(arena, a, env);
            let b2 = index_expr_to_expr(arena, b, env);
            arena.push(Expr::BinOp(BinOp::Sub, a2, b2))
        }
        IndexExpr::Mul(a, b) => {
            let a2 = index_expr_to_expr(arena, a, env);
            let b2 = index_expr_to_expr(arena, b, env);
            arena.push(Expr::BinOp(BinOp::Mul, a2, b2))
        }
    }
}

// An index variable's runtime value, first match wins:
// 1. a witness: `len` of the innermost list parameter (or contract
//    argument) typed Vec(name);
// 2. its static binding in index_subst, translated in turn;
// 3. otherwise nothing at runtime holds it (an existential from a plain
//    `let x: Vec(n) = <Dyn>`, or a call-site instantiation): the check fails
//    with a clean type error rather than reading an unbound variable.
// A same-named TERM variable (`fun n: Int -> fun v: Vec(n)`) is never used:
// the typechecker does not link a term `n` to the index `n`, so reading it
// would let a wrong-length value through (or reject a right one).
fn index_var_to_expr(arena: &mut Arena, name: &str, env: &CheckEnv) -> ExprRef {
    let witness = env.local.iter().rev().copied().chain(env.infer.index_witness.iter().rev()).find(|w| w.var == name);
    if let Some(w) = witness {
        w.used.set(true);
        // An untyped caller may have passed a non-list for the witness
        // parameter; `len` would panic on it, so test is_list first.
        let list_ref = arena.push(Expr::Var(w.hidden.clone()));
        let is_list = build_probe(arena, list_ref, Test::List, &Type::List(Rc::new(Type::Dyn)), env);
        let len_ref = arena.push(Expr::Var(w.hidden.clone()));
        let len_call = build_prelude_call(arena, "len", len_ref);
        let msg = arena.push(Expr::Str(format!("type error: index variable {name}'s witness is not a list")));
        let fail_var = prelude_var(arena, "fail");
        let fail_call = arena.push(Expr::App(fail_var, msg));
        return arena.push(Expr::If(is_list, len_call, fail_call));
    }
    if let Some(bound) = env.infer.index_subst.get(name) {
        return index_expr_to_expr(arena, bound, env);
    }
    let msg = arena.push(Expr::Str(format!("type error: index variable {name} has no runtime value to check a length against")));
    let fail_var = prelude_var(arena, "fail");
    arena.push(Expr::App(fail_var, msg))
}

// The raw boolean condition shared by build_boundary_check's and
// build_shape_predicate's own Type::Indexed(List(_), _) arms: "is
// `value_ref` list-shaped AND does its length equal `index`". The list
// probe has to run BEFORE `len`: `len` panics internally (see machine.rs)
// on a non-list argument, which would surface as an unrelated bare panic
// instead of a clean "type error: expected ..., found ..." message.
// Returning the bare condition (not a full Let/If/fail wrapper) lets
// build_boundary_check wrap it in its let/if skeleton (evaluating
// `value_ref` exactly once) while build_shape_predicate uses it directly as
// its own bare predicate (no let-binding, callers may OR several of these
// together). Also returns the `len(value) == <index>` node, which a
// deferred obligation patches in place (resolve_obligations).
fn build_indexed_shape_cond(arena: &mut Arena, value_ref: ExprRef, index: &IndexExpr, env: &CheckEnv) -> (ExprRef, ExprRef) {
    let is_list = build_probe(arena, value_ref, Test::List, &Type::List(Rc::new(Type::Dyn)), env);
    let len_var = prelude_var(arena, "len");
    let len_call = arena.push(Expr::App(len_var, value_ref));
    let index_expr = index_expr_to_expr(arena, index, env);
    let len_eq = arena.push(Expr::BinOp(BinOp::Eq, len_call, index_expr));
    let false_lit = arena.push(Expr::Bool(false));
    (arena.push(Expr::If(is_list, len_eq, false_lit)), len_eq)
}

// The union check for a union a single Check can't express (shape_test is
// None: a Vec(n) alternative; or needs_contract: a Fun alternative):
// `let __check_tmp = e in if <alt1-shape> then <alt1's OWN full check>
// else if <alt2-shape> then <alt2's OWN full check> else ... else
// <failing Check>` -- tries each alternative's shape predicate
// (build_shape_predicate) in turn, and only once ONE of them matches,
// runs that SPECIFIC alternative's full build_boundary_check -- so a Fun
// alternative gets its real per-call contract (wrap_fun_contract), the same
// rigor that type would get as a plain (non-union) annotation, not just
// "shaped like something callable." Can't simply call build_boundary_check
// per alternative and fall through on its own failure -- it would abort the
// whole check on the first non-matching alternative instead of trying the
// next one -- so the shape predicate decides WHICH alternative's full check
// to run. An alternative with no contract needs no second check: its shape
// predicate IS its full check.
fn build_union_check(arena: &mut Arena, e: ExprRef, to: &Type, env: &CheckEnv, visiting: &HashSet<String>) -> ExprRef {
    let alts: Rc<Vec<Type>> = match to {
        Type::Union(alts) => alts.clone(),
        _ => unreachable!("build_union_check is only ever called with a Union target"),
    };
    let env = &CheckEnv { infer: env.infer, local: env.local.clone(), defer: false, repeatable: env.repeatable, span: env.span };
    let tmp = fresh_index_name("__check_tmp");
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let mut result = build_type_error(arena, tmp_ref, to, env);
    for alt in alts.iter().rev() {
        let pred = build_shape_predicate(arena, tmp_ref, alt, env, visiting);
        let checked = if needs_contract(alt, env, visiting) { build_boundary_check(arena, tmp_ref, alt, env, visiting) } else { tmp_ref };
        result = arena.push(Expr::If(pred, checked, result));
    }
    arena.push(Expr::Let(tmp, None, e, result))
}

// The shallow runtime test for `ty` as one native Test, or None when it
// can't be one: a Vec(n) length compare needs the index expression
// evaluated in the environment (build_indexed_shape_cond), so it -- and any
// union or Named alias reaching one -- stays desugared. `visiting` is the
// same ancestor set as build_boundary_check's: a Named already being
// unfolded is `Never` here (re-trying it adds nothing beyond the
// non-recursive alternatives, so `type A = Int | A` behaves as plain Int),
// an unregistered Named is `Any` ("no information here"). Mirrors what
// the old predicate builder answered, arm for arm.
fn shape_test(ty: &Type, env: &CheckEnv, visiting: &HashSet<String>) -> Option<Test> {
    Some(match ty {
        Type::Dyn | Type::Var(_) => Test::Any,
        Type::Int => Test::Int,
        Type::Float => Test::Float,
        Type::Bool => Test::Bool,
        Type::Str => Test::Str,
        Type::List(_) => Test::List,
        Type::Fun(..) => Test::Fun,
        Type::Token(id) => Test::Token(*id),
        // Exact arity, unlike Record's width tolerance.
        Type::Tuple(items) => Test::Tuple(items.len()),
        // Width-tolerant: `v` need only HAVE (at least) every required
        // field -- extra fields are fine, see Pattern::Record's own doc
        // comment for why nothing downstream can ever observe them.
        Type::Record(fields) => Test::Record(fields.iter().map(|(name, _)| name.clone()).collect()),
        Type::Union(alts) => Test::Or(alts.iter().map(|alt| shape_test(alt, env, visiting)).collect::<Option<Vec<_>>>()?),
        Type::Named(id) if visiting.contains(id) => Test::Never,
        Type::Named(id) => match env.infer.named_types.get(id) {
            Some(raw) => shape_test(raw, env, &extend_visiting(visiting, id))?,
            None => Test::Any,
        },
        Type::Indexed(wrapped, _) => match wrapped.as_ref() {
            Type::List(_) => return None,
            other => shape_test(other, env, visiting)?,
        },
    })
}

fn extend_visiting(visiting: &HashSet<String>, id: &str) -> HashSet<String> {
    let mut visiting = visiting.clone();
    visiting.insert(id.to_string());
    visiting
}

// Does checking `ty` do more than its shape test -- is a Fun reachable, so a
// matching value must also get its per-call contract? Decides whether a
// union check needs build_union_check's pick-then-check order.
fn needs_contract(ty: &Type, env: &CheckEnv, visiting: &HashSet<String>) -> bool {
    match ty {
        Type::Fun(..) => true,
        Type::Union(alts) => alts.iter().any(|alt| needs_contract(alt, env, visiting)),
        Type::Named(id) if !visiting.contains(id) => {
            env.infer.named_types.get(id).is_some_and(|raw| needs_contract(raw, env, &extend_visiting(visiting, id)))
        }
        _ => false,
    }
}

// The bare boolean half of build_boundary_check's per-type dispatch --
// "does `value_ref` shallowly look like `ty`," with no let-binding and no
// failure of its own, so build_union_check can try several of these in turn
// before deciding anything. A native Probe wherever shape_test has a Test;
// only a Vec(n) length compare (directly, or reached through a union or
// alias) is built from ordinary If/Bool nodes.
fn build_shape_predicate(arena: &mut Arena, value_ref: ExprRef, ty: &Type, env: &CheckEnv, visiting: &HashSet<String>) -> ExprRef {
    if let Some(test) = shape_test(ty, env, visiting) {
        return build_probe(arena, value_ref, test, ty, env);
    }
    match ty {
        // An Indexed value's LENGTH is real, provable shape information this
        // check must not throw away -- this is what lets build_union_check
        // correctly fall through a Vec(3) alternative to try a plain [Int]
        // alternative next instead of routing a length-5 list into Vec(3)'s
        // own full check (which would then reject it) just because it merely
        // "looks like some list." The same condition build_boundary_check's
        // own Indexed arm uses. A Type::Named-wrapped Indexed (derived
        // index-refinement) is a later phase's own concern -- nothing before
        // this phase can construct one yet -- so it delegates to the shape.
        Type::Indexed(wrapped, index) => match wrapped.as_ref() {
            Type::List(_) => build_indexed_shape_cond(arena, value_ref, index, env).0,
            other => build_shape_predicate(arena, value_ref, other, env, visiting),
        },
        // `if pred1 then true else if pred2 then true else ... false`,
        // short-circuiting like the Test::Or it stands in for.
        Type::Union(alts) => {
            let mut result = arena.push(Expr::Bool(false));
            for alt in alts.iter().rev() {
                let pred = build_shape_predicate(arena, value_ref, alt, env, visiting);
                let true_lit = arena.push(Expr::Bool(true));
                result = arena.push(Expr::If(pred, true_lit, result));
            }
            result
        }
        // shape_test is None for a Named only past a registered, unvisited
        // unfold (a visited or unregistered one answers Never/Any).
        Type::Named(id) => {
            let raw = env.infer.named_types.get(id).expect("shape_test answered None for an unregistered alias");
            build_shape_predicate(arena, value_ref, raw, env, &extend_visiting(visiting, id))
        }
        _ => unreachable!("every other type has a native shape test"),
    }
}

// An Expr::Check of `value_ref` against `test` in the given mode, blaming
// the cast site `env.span`; `to` is the type whose display text a failure
// reports.
fn build_check_node(arena: &mut Arena, value_ref: ExprRef, test: Test, mode: CheckMode, to: &Type, env: &CheckEnv) -> ExprRef {
    arena.push(Expr::Check(value_ref, Rc::new(CheckSpec { test, mode, to: to.to_string(), span: env.span })))
}

// The Bool outcome of `test`: a union picking its alternative.
fn build_probe(arena: &mut Arena, value_ref: ExprRef, test: Test, to: &Type, env: &CheckEnv) -> ExprRef {
    build_check_node(arena, value_ref, test, CheckMode::Probe, to, env)
}

// A check that always fails with "type error: expected {to}, found
// {type of value_ref}" -- the else-branch of every desugared check.
fn build_type_error(arena: &mut Arena, value_ref: ExprRef, to: &Type, env: &CheckEnv) -> ExprRef {
    build_check_node(arena, value_ref, Test::Never, CheckMode::Assert, to, env)
}

// The one native check: `e` evaluated once and tested against `to`'s whole
// shallow shape (shape_test) -- arity for Tuple, field PRESENCE
// (width-tolerantly) for Record, not that each position/field's own value
// matches ITS OWN element type. A full recursive check is possible but not
// built yet; this matches the same "confirm the shape, not deeper"
// precedent every other Dyn boundary check here already sets. Callers pass
// only a type shape_test answers (Fun callers ask for any_fun()).
fn build_shape_check(arena: &mut Arena, e: ExprRef, to: &Type, env: &CheckEnv, visiting: &HashSet<String>) -> ExprRef {
    let test = shape_test(to, env, visiting).expect("build_shape_check takes a type with a native shape test");
    build_check_node(arena, e, test, CheckMode::Assert, to, env)
}

// A synthesized reference to the prelude builtin `name`. Spelled `#name`,
// which no source text can bind or lex, and which `resolve` maps straight to
// the prelude table: a user binding named like the builtin (`let len = ..`,
// a parameter `fail`) cannot capture a synthesized call. EVERY builtin
// synthesized code still calls (`len`, `fail`, `get_field`) goes through here.
fn prelude_var(arena: &mut Arena, name: &str) -> ExprRef {
    arena.push(Expr::Var(format!("#{name}")))
}

fn build_prelude_call(arena: &mut Arena, name: &str, value_ref: ExprRef) -> ExprRef {
    let v = prelude_var(arena, name);
    arena.push(Expr::App(v, value_ref))
}

// `builtin(value_ref, str_arg)` -- the two-argument counterpart to
// build_prelude_call, used by `.field` access's get_field desugaring.
fn build_str_call(arena: &mut Arena, builtin: &str, value_ref: ExprRef, str_arg: &str) -> ExprRef {
    let f = prelude_var(arena, builtin);
    let applied = arena.push(Expr::App(f, value_ref));
    let arg_lit = arena.push(Expr::Str(str_arg.to_string()));
    arena.push(Expr::App(applied, arg_lit))
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
// A Vec(n) parameter makes `n` the argument's length for the return check
// (a local witness, bound inside the wrapper only if that check reads it).
fn wrap_fun_contract(arena: &mut Arena, e: ExprRef, param_ty: Rc<Type>, ret_ty: Rc<Type>, env: &CheckEnv, visiting: &HashSet<String>) -> ExprRef {
    let fn_var = fresh_index_name("__contract_fn");
    let arg_var = fresh_index_name("__contract_arg");
    let fn_var_ref = arena.push(Expr::Var(fn_var.clone()));
    let arg_var_ref = arena.push(Expr::Var(arg_var.clone()));
    let checked_fn = build_shape_check(arena, fn_var_ref, &any_fun(), env, &HashSet::new());
    let call = arena.push(Expr::App(checked_fn, arg_var_ref));
    let witness = list_length_var(&param_ty, env.infer).map(|var| IndexWitness::new(var, &arg_var));
    let checked_call = match &witness {
        Some(w) => {
            let mut local = env.local.clone();
            local.push(w);
            build_boundary_check(arena, call, &ret_ty, &CheckEnv { infer: env.infer, local, defer: env.defer, repeatable: true, span: env.span }, visiting)
        }
        None => {
            let env = CheckEnv { infer: env.infer, local: env.local.clone(), defer: env.defer, repeatable: true, span: env.span };
            build_boundary_check(arena, call, &ret_ty, &env, visiting)
        }
    };
    let checked_call = match &witness {
        Some(w) if w.used.get() => w.bind(arena, &arg_var, checked_call),
        _ => checked_call,
    };
    let lambda = arena.push(Expr::Lambda(arg_var, Some((*param_ty).clone()), checked_call));
    arena.push(Expr::Let(fn_var, None, e, lambda))
}

// The bidirectional-checking mode a call to `elaborate_mode`/`elaborate_node`
// runs under. `Synth` is today's only mode (infer a type bottom-up, no
// expected type available). `Check(expected)` is new: the caller already
// knows what type this expression must have (typically a declared
// annotation flowing down through a Let/Lambda chain).
//
// The mode-consulting arms in `elaborate_node`/`elaborate_mode` today:
// `Expr::Tuple`, `Expr::ListLit`, `BinOp::Cons`, `Expr::If`, `Expr::Match`.
// `If` and `Match` check each branch/arm against `expected` directly
// instead of inferring each one and reconciling them against EACH OTHER
// afterward (today's only option for everything else, via unify_trial).
// `Expr::ListLit`/`BinOp::Cons` (Case A) and `Expr::Tuple` (Case B) check
// against an `Indexed`-typed `expected` to construct a value with a
// statically-known index, rather than synthesizing a plain, un-indexed
// shape (see the design spec 2026-09-20, sections 1 and 4). Every OTHER
// expression shape still just synthesizes regardless of mode;
// `check_against`'s own coerce-then-unify_fits fallback (mirroring
// Expr::App's existing argument-check pattern) is what makes Check mode
// sound for those shapes too, without each of them needing its own
// Check-mode arm.
//
// Invariant for future maintainers: any FUTURE Check-mode arm added here
// over an `Indexed(Named(_))` expected type must also be added to
// `case_b_base_candidate`'s own exclusion list (this file) -- that
// function's own tail-dispatch hook has a Mode::Synth-only fallback path
// for its own "not a recognized special shape" case, and a new Check-mode
// arm left out of its exclusion list is silently bypassed by that
// fallback exactly the way `Expr::If`/`Expr::Match` themselves were
// before Task 6's own review caught it (see `case_b_base_candidate`'s own
// doc comment for the traced regression).
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
    // `witness`: this Lambda pushed an infer.index_witness entry (its
    // parameter is Vec(n)), popped when the frame unwinds.
    Fun { param: String, param_ty: Type, witness: bool },
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

// Eligibility (spec section 5): is `scrut_ty` (already resolved via
// infer.resolve_deep) a Type::Indexed wrapping a shape Match's own arm
// loop knows how to derive a base/step hypothesis from, whose own index
// expression (already resolved via infer.resolve_index_deep) is a BARE
// variable? Two wrapped shapes qualify -- Case A, a plain Type::List
// (the fixed []/:: rule), and Case B, a qualifying Type::Named+
// Type::Union pair (the rule derived from THAT type's own structure,
// see `qualifying_named_alternatives`) -- anything else (a non-Indexed
// scrutinee, a wrapped shape neither case recognizes, a literal or
// compound index expression with no single name to bind a hypothesis
// under) returns None: the safe "not applicable" fallback used
// throughout this design. Renamed from Task 1's own
// `case_a_refinement_target` now that Task 3 extends it to cover Case B
// too.
enum IndexedRefinementTarget {
    List { index_var: String, elem_ty: Rc<Type> },
    // `base_alt`/`step_alt` are the two alternatives
    // `qualifying_named_alternatives` already sorted out -- the SAME
    // per-alternative shapes Match's own arm loop re-checks a pattern
    // against via `pattern_could_match` to decide which hypothesis (if
    // any) that specific arm earns.
    Named { index_var: String, id: String, base_alt: Rc<Type>, step_alt: Rc<Type> },
}

fn indexed_refinement_target(
    resolved_scrut_ty: &Type,
    resolved_index: Option<&IndexExpr>,
    named_types: &HashMap<String, Type>,
) -> Option<IndexedRefinementTarget> {
    match (resolved_scrut_ty, resolved_index) {
        (Type::Indexed(wrapped, _), Some(IndexExpr::Var(name))) => match &**wrapped {
            Type::List(elem_ty) => Some(IndexedRefinementTarget::List { index_var: name.clone(), elem_ty: elem_ty.clone() }),
            Type::Named(id) => qualifying_named_alternatives(id, named_types).map(|(base_alt, step_alt)| {
                IndexedRefinementTarget::Named {
                    index_var: name.clone(),
                    id: id.clone(),
                    base_alt: Rc::new(base_alt),
                    step_alt: Rc::new(step_alt),
                }
            }),
            _ => None,
        },
        _ => None,
    }
}

// Case B eligibility, spec section 5 points 1-2: does `named_types[id]`
// resolve to a Type::Union of EXACTLY two alternatives, one that
// doesn't structurally contain `Named(id)` at all (the base case, 0
// occurrences) and one that contains it EXACTLY once (the step case --
// not zero, not two-or-more: a step alternative referencing the id
// twice or more, e.g. a binary tree's `Node(Tree, Tree)` shape,
// deliberately doesn't qualify either, per the spec's own Non-goals).
// Anything else -- missing id, wrong alternative count, any other
// occurrence-count pairing -- isn't eligible: the same safe "not
// applicable" fallback used throughout this design. Returns
// (base_alt, step_alt) in that fixed order regardless of which one
// `named_types` happens to store first.
fn qualifying_named_alternatives(id: &str, named_types: &HashMap<String, Type>) -> Option<(Type, Type)> {
    let Type::Union(alts) = named_types.get(id)? else { return None };
    let [a, b] = alts.as_slice() else { return None };
    match (crate::parser::count_named(a, id), crate::parser::count_named(b, id)) {
        (0, 1) => Some((a.clone(), b.clone())),
        (1, 0) => Some((b.clone(), a.clone())),
        _ => None,
    }
}

// Case B's own analog of Case A's `Pattern::Cons(_, tail)` tail-position
// check (spec section 5 point 3): locates the sub-pattern bound at the
// step alternative's own single self-referential position -- guaranteed
// to exist and be unique by `qualifying_named_alternatives`'s own
// occurrence-count check -- so the caller can retype it as
// `Type::Indexed(Named(id), fresh)`, the same override Case A already
// does for a Cons's own tail. Only the spec's own worked shape is
// recognized: the self-reference IS the whole alternative (`type X = Y
// | X`), or the alternative is a Tuple with the self-reference as one
// bare item (the spec's own `(Int, List)`) -- mirroring Case A's own
// "bare Var tail only" v1 restriction, anything with the self-reference
// nested deeper (inside a further Tuple/List/Record) returns None, the
// same safe fallback.
fn self_ref_pattern<'a>(pat: &'a Pattern, step_alt: &Type, id: &str) -> Option<&'a Pattern> {
    match step_alt {
        Type::Named(n) if n == id => Some(pat),
        Type::Tuple(items) => match pat {
            Pattern::List(pats) if pats.len() == items.len() => {
                items.iter().zip(pats.iter()).find_map(|(item_ty, p)| match item_ty {
                    Type::Named(n) if n == id => Some(p),
                    _ => None,
                })
            }
            _ => None,
        },
        _ => None,
    }
}

// Case B construction, step case (design spec 2026-09-20 sec 4):
// mirrors self_ref_pattern's own Type::Tuple sub-case (used on the
// pattern-matching side, Phase 4) but returns a POSITION into the
// Tuple's items instead of a &Pattern, since a Tuple literal's items
// at a construction site are ExprRefs, not Patterns to recurse into.
// Deliberately does NOT also mirror self_ref_pattern's separate
// top-level `Type::Named(n) if n == id` case (for a degenerate `type X
// = Y | X` shape, the self-reference IS the whole alternative) -- not
// an oversight, just unreachable from this function's only call site,
// which has already matched `step_alt` as a `Type::Tuple` (the
// step-shaped-literal check) before ever calling this.
fn self_ref_position(step_alt: &Type, id: &str) -> Option<usize> {
    match step_alt {
        Type::Tuple(items) => items.iter().position(|item_ty| matches!(item_ty, Type::Named(n) if n == id)),
        _ => None,
    }
}

// Case B construction, base-case detection for elaborate_mode's own
// tail dispatch (design spec 2026-09-20 sec 4): `expected` must
// resolve to Indexed(Named(id), idx) with a qualifying 2-alternative
// union (qualifying_named_alternatives), and `cur_expr` must NOT
// already be a step-shaped Tuple literal -- that case is handled
// entirely inside Expr::Tuple's own Mode::Check arm below, dispatched
// normally; treating it as a base-case candidate here too would
// double-elaborate it.
//
// DEVIATION from the task-6 brief's literal text: the brief's own
// version excluded only the step-shaped Tuple case. That missed a real
// regression -- If/Match ALSO already have their own Mode-aware
// Check-mode arms inside elaborate_node (Phase 3's own If/Match
// retrofit, plus Match's Phase 4 Case B refinement-hypothesis
// machinery). Treating an If/Match tail as a base-case candidate here
// routes it through this hook's plain Mode::Synth `elaborate()` call
// instead of `elaborate_node(..., Mode::Check(expected))`, silently
// discarding that machinery -- confirmed by a real regression in
// `case_b_step_case_hypothesis_and_retyping_enable_a_real_recursive_proof`
// (a pre-existing, passing Phase 4 test) when the brief's version was
// tried verbatim: the match's own step-arm hypothesis (`t`'s retyped
// `Indexed(Named(id), m)`) never got computed under Synth, so checking
// its body against `Indexed(Named(id), n - 1)` failed. Excluding
// If/Match here, exactly like Tuple's own step-shape exclusion, lets
// elaborate_node's normal dispatch handle them and fixes it.
fn case_b_base_candidate(arena: &Arena, cur_expr: ExprRef, expected: &Type, infer: &InferCtx) -> Option<(String, Type, Type, Rc<IndexExpr>)> {
    let Type::Indexed(wrapped, idx) = infer.resolve_deep(expected) else { return None };
    let Type::Named(id) = wrapped.as_ref() else { return None };
    let (base_alt, step_alt) = qualifying_named_alternatives(id, &infer.named_types)?;
    match &arena[cur_expr] {
        Expr::If(..) | Expr::Match(..) => return None,
        Expr::Tuple(items) => {
            if let Type::Tuple(step_items) = &step_alt {
                if items.len() == step_items.len() {
                    return None;
                }
            }
        }
        _ => {}
    }
    Some((id.clone(), base_alt, step_alt, idx))
}

// Does `ty` mention an Indexed type anywhere? Gates pushing an expected
// element type down into a Tuple/List literal's items: only Indexed
// element types need it (a plain literal can never satisfy one), and
// checking every item would insert runtime casts for Dyn items that the
// whole-literal coerce deliberately leaves alone.
fn contains_indexed(ty: &Type) -> bool {
    match ty {
        Type::Indexed(..) => true,
        Type::List(t) => contains_indexed(t),
        Type::Tuple(ts) | Type::Union(ts) => ts.iter().any(contains_indexed),
        Type::Record(fs) => fs.iter().any(|(_, t)| contains_indexed(t)),
        Type::Fun(p, _, r) => contains_indexed(p) || contains_indexed(r),
        _ => false,
    }
}

// A closed-form value for an index expression with no free variables,
// else None (an unbound variable, or arithmetic overflow).
fn const_index(e: &IndexExpr) -> Option<i64> {
    match e {
        IndexExpr::Var(_) => None,
        IndexExpr::Lit(n) => Some(*n),
        IndexExpr::Add(a, b) => const_index(a)?.checked_add(const_index(b)?),
        IndexExpr::Sub(a, b) => const_index(a)?.checked_sub(const_index(b)?),
        IndexExpr::Mul(a, b) => const_index(a)?.checked_mul(const_index(b)?),
    }
}

// Type-level mirror of Expr::Tuple's Case B step arm, for a value whose
// synthesized type is already a plain `Type::Tuple` (e.g. read back from
// an unannotated `let`) rather than a literal: is `actual` a legitimate
// `Named(id)(idx)` step value? Only decided when `idx` is a known
// positive constant -- a symbolic or zero index keeps the base-case
// treatment, since a Dyn base alternative would also accept the tuple
// there. Non-recursive positions must be consistent with the step
// alternative's; the recursive position is checked against `idx - 1`,
// recursing while it is itself a tuple and ending at a base value (or
// Dyn/Var, which carries no information to contradict).
fn is_step_shaped_tuple(actual: &Type, base_alt: &Type, step_alt: &Type, id: &str, idx: &IndexExpr, infer: &InferCtx) -> bool {
    let (Type::Tuple(items), Type::Tuple(step_items)) = (actual, step_alt) else { return false };
    let Some(k) = const_index(&infer.resolve_index_deep(idx)).filter(|k| *k > 0) else { return false };
    let Some(rec_pos) = self_ref_position(step_alt, id) else { return false };
    if items.len() != step_items.len() {
        return false;
    }
    if !(0..items.len()).all(|i| i == rec_pos || consistent(&items[i], &step_items[i])) {
        return false;
    }
    let tail = infer.resolve_deep(&items[rec_pos]);
    match &tail {
        Type::Dyn | Type::Var(_) => true,
        Type::Tuple(_) if k > 1 => is_step_shaped_tuple(&tail, base_alt, step_alt, id, &IndexExpr::Lit(k - 1), infer),
        _ => k == 1 && !matches!(tail, Type::Indexed(..)) && consistent(&tail, base_alt),
    }
}

// Shared "nothing real to correlate against" fallback: bind each of a
// Pattern::List's own positions to its own independent fresh var.
// Factored out of what used to be three separate, identically-bodied
// arms in bind_pattern_vars's own Pattern::List match (Type::Var, the
// final catch-all, and now the two new Type::Named/Type::Union arms'
// own "no matching alternative found" cases below) -- same fallback,
// same reasoning, one copy.
fn bind_fresh_positions(
    ctx: &Ctx,
    pats: &[Pattern],
    infer: &mut InferCtx,
    span: Span,
    visiting: &HashSet<String>,
) -> Result<Ctx, TypeError> {
    let mut c = ctx.clone();
    for p in pats {
        let elem_ty = infer.fresh_var("elem");
        c = bind_pattern_vars(&c, p, &elem_ty, infer, span, visiting)?;
    }
    Ok(c)
}

// `bind_fresh_positions`'s own Record analog: bind each of a
// Pattern::Record's own named fields to its own independent fresh var,
// keyed by that field's own name (matching this function's own existing
// per-field fallback). Same "nothing real to correlate against"
// fallback, factored out for the same reason -- Finding 6's new
// Type::Indexed/Named/Union delegation arms below need this same
// fallback for their own "no matching alternative found" cases, not
// just the catch-all.
fn bind_fresh_record_fields(
    ctx: &Ctx,
    fields: &[(String, Pattern)],
    infer: &mut InferCtx,
    span: Span,
    visiting: &HashSet<String>,
) -> Result<Ctx, TypeError> {
    let mut c = ctx.clone();
    for (name, p) in fields {
        let field_ty = infer.fresh_var(name);
        c = bind_pattern_vars(&c, p, &field_ty, infer, span, visiting)?;
    }
    Ok(c)
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
//
// Fix round 1 (critical review finding): `visiting` mirrors
// pattern_could_match's own parameter of the same name/position exactly
// -- an accumulated set of Type::Named ids already being unfolded on
// THIS recursion path, threaded unchanged through every existing arm
// below and only ever grown by the Type::Named arm itself. Without this,
// the Type::Union arm's own `pattern_could_match` probe used a fresh,
// empty set on every call, with no memory of an enclosing Type::Named
// arm's own unfolding -- letting a Union alternative that is itself a
// bare self-reference (e.g. `type List = List | (Int, List)`, where the
// FIRST alternative is `Named("List")` verbatim) look "reachable" via
// its own fresh probe forever, recursing without end. See this task's
// own fix-round-1 report entry for the traced counterexample.
fn bind_pattern_vars(
    ctx: &Ctx,
    pat: &Pattern,
    scrutinee_ty: &Type,
    infer: &mut InferCtx,
    span: Span,
    visiting: &HashSet<String>,
) -> Result<Ctx, TypeError> {
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
                    c = bind_pattern_vars(&c, p, item_ty, infer, span, visiting)?;
                }
                Ok(c)
            }
            Type::List(shared_elem) => {
                let mut c = ctx.clone();
                for p in pats {
                    c = bind_pattern_vars(&c, p, shared_elem, infer, span, visiting)?;
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
            Type::Var(_) => bind_fresh_positions(ctx, pats, infer, span, visiting),
            // A Type::Indexed-wrapped scrutinee (Vec(n)-style, or Case
            // B's own Named-wrapped analog per §2's generalization) --
            // correlate against the WRAPPED type directly, "forgetting"
            // the index the same way Pattern::Cons's own arm below
            // already does via forget_index, and pattern_could_match's
            // own Type::Indexed arm mirrors exactly. Needed for Case B:
            // a Case-B-eligible scrutinee's own real stored type is
            // `Type::Indexed(Type::Named(id), n)`, still wrapped, by the
            // time a match arm's own pattern (e.g. the tagged-tuple-style
            // `(h, t)`) gets here -- without this arm it fell into the
            // generic catch-all below, no correlation at all.
            Type::Indexed(wrapped, _) => bind_pattern_vars(ctx, pat, wrapped, infer, span, visiting),
            // A Type::Named scrutinee (a `type X = ... in` alias
            // reference) -- unfold ONE level via infer.named_types and
            // recurse, mirroring pattern_could_match's own Type::Named
            // arm exactly (including its "unknown id" permissive
            // fallback: nothing registered to unfold into means nothing
            // real to correlate against, same as the generic catch-all).
            // This is Case B's own real prerequisite, found and fixed by
            // this task's own investigation (see its report): without
            // it, a Case-B-style scrutinee's own Type::Named(id) never
            // even reaches the Type::Union arm just below -- it's still
            // nominally Named at this point, not yet unfolded to a
            // Union at all.
            //
            // Fix round 1: a cycle guard IS needed here after all -- the
            // "only ever recurses into an already-proved-reachable
            // shape" argument this comment used to make was wrong
            // because the Type::Union arm just below used to run its own
            // reachability probe with a FRESH, empty visiting set every
            // time, disconnected from whatever this arm had already
            // unfolded -- so a Union alternative that is ITSELF a bare
            // self-reference (`type List = List | (Int, List)`) could
            // look reachable forever, recursing without end (see this
            // task's own fix-round-1 report entry for the traced
            // repro). Mirrors pattern_could_match's own Type::Named arm
            // exactly: if `id` is already being unfolded on this path,
            // stop and fall back to the safe fresh-vars-per-position
            // behavior instead of unfolding again; otherwise recurse
            // with `id` added to the accumulated set.
            Type::Named(id) => {
                if visiting.contains(id) {
                    return bind_fresh_positions(ctx, pats, infer, span, visiting);
                }
                match infer.named_types.get(id).cloned() {
                    Some(unfolded) => {
                        let mut visiting = visiting.clone();
                        visiting.insert(id.clone());
                        bind_pattern_vars(ctx, pat, &unfolded, infer, span, &visiting)
                    }
                    None => bind_fresh_positions(ctx, pats, infer, span, visiting),
                }
            }
            // A Type::Union scrutinee (a resolved alias's own RHS, or an
            // ordinary Union type written directly) -- Case B's own
            // second real prerequisite: find WHICH alternative this
            // pattern's own shape actually corresponds to (reusing
            // pattern_could_match's own per-alternative logic -- the
            // exact same question Expr::Match already asks, via that
            // same function, before ever reaching bind_pattern_vars at
            // all) and correlate against THAT alternative's own real
            // structure, instead of falling through to the fresh,
            // unconstrained-per-position fallback below. Without this,
            // Case B's own step alternative sub-bindings (e.g. `t` in
            // `(h, t)`) would keep NO type precision at all even after
            // this task's own index-hypothesis injection lands --
            // looking tested (the hypothesis fires) without actually
            // working end-to-end (nothing observably depends on it
            // being correct). If more than one alternative's shape
            // admits `pat` (a genuinely ambiguous pattern, e.g. a Union
            // of two same-arity Tuples), the first match wins --
            // deterministic, same "best effort, not exhaustive" stance
            // pattern_could_match's own Union arm already takes.
            //
            // Fix round 1: this arm's own `pattern_could_match` probe
            // must reuse the SAME accumulated `visiting` set passed into
            // this arm -- not a fresh `&HashSet::new()` -- and the
            // subsequent recursive call into whichever alternative
            // `.find()` picks must also pass that same set through
            // unchanged (this arm never adds to it; only the Type::Named
            // arm above does). A fresh set here was the actual bug: it
            // let this arm "forget" that an enclosing Type::Named arm was
            // already unfolding `id`, so a bare self-referential
            // alternative could look reachable again and again forever.
            Type::Union(alts) => match alts.iter().find(|alt| pattern_could_match(pat, alt, &infer.named_types, visiting)) {
                Some(matching_alt) => bind_pattern_vars(ctx, pat, matching_alt, infer, span, visiting),
                None => bind_fresh_positions(ctx, pats, infer, span, visiting),
            },
            // A concrete, non-Tuple, non-List, non-Indexed/Named/Union
            // scrutinee against a List pattern: pattern_could_match's
            // own existing check (called by Expr::Match before this
            // function ever runs) already rejects this case statically
            // -- bind each position to its own independent fresh var
            // (nothing real to correlate against) so nested sub-bindings
            // still work.
            _ => bind_fresh_positions(ctx, pats, infer, span, visiting),
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
            let c = bind_pattern_vars(ctx, head, &elem_ty, infer, span, visiting)?;
            bind_pattern_vars(&c, tail, &list_shape, infer, span, visiting)
        }
        Pattern::Record(fields) => match scrutinee_ty {
            Type::Record(type_fields) => {
                let mut c = ctx.clone();
                for (name, p) in fields {
                    let field_ty = find_field(type_fields, name).cloned().unwrap_or_else(|| infer.fresh_var(name));
                    c = bind_pattern_vars(&c, p, &field_ty, infer, span, visiting)?;
                }
                Ok(c)
            }
            // Finding 6 (final review): the same Type::Indexed/Named/Union
            // delegation Pattern::List already got from Task 3's own Case B
            // prerequisite fix, mirrored here for Record so a Record
            // pattern matched against a Vec(n)-of-records or a Case-B-
            // eligible Named/Union scrutinee gets real per-field
            // correlation too, instead of always falling back to
            // `infer.fresh_var` per field (the same "zero correlation" gap
            // Task 3 closed for List).
            Type::Indexed(wrapped, _) => bind_pattern_vars(ctx, pat, wrapped, infer, span, visiting),
            Type::Named(id) => {
                if visiting.contains(id) {
                    return bind_fresh_record_fields(ctx, fields, infer, span, visiting);
                }
                match infer.named_types.get(id).cloned() {
                    Some(unfolded) => {
                        let mut visiting = visiting.clone();
                        visiting.insert(id.clone());
                        bind_pattern_vars(ctx, pat, &unfolded, infer, span, &visiting)
                    }
                    None => bind_fresh_record_fields(ctx, fields, infer, span, visiting),
                }
            }
            Type::Union(alts) => match alts.iter().find(|alt| pattern_could_match(pat, alt, &infer.named_types, visiting)) {
                Some(matching_alt) => bind_pattern_vars(ctx, pat, matching_alt, infer, span, visiting),
                None => bind_fresh_record_fields(ctx, fields, infer, span, visiting),
            },
            _ => bind_fresh_record_fields(ctx, fields, infer, span, visiting),
        },
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
                        let (t, renames) = open_signature(&t, infer);
                        let (val_row, val4) = check_against_rigid(arena, val, &t, &renames, &cur_ctx, spans, infer)?;
                        (t, val_row, val4)
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
                // Each annotation opened once (open_signature), so the
                // pre-binding below and the value check share its renames.
                let opened: Vec<Option<(Type, Renames)>> =
                    bindings.iter().map(|(_, ann, _)| ann.as_ref().map(|t| open_signature(t, infer))).collect();
                for ((name, _, _), ann) in bindings.iter().zip(&opened) {
                    let ann = ann.as_ref().map(|(t, _)| t);
                    val_ctx = match ann {
                        // Generalized, not plain extend: a fully-given
                        // annotation makes checking mode against it
                        // sound even for a recursive self/sibling
                        // reference (Mycroft's polymorphic-recursion
                        // result; design spec 2026-09-20 sec 2) --
                        // gated on an explicit annotation, not new
                        // inference, so this is deliberately narrower
                        // than general HM polymorphic recursion.
                        Some(ty) => extend_generalized(&val_ctx, name, ty.clone(), infer),
                        None => extend(&val_ctx, name, infer.fresh_var(name)),
                    };
                }
                let mut elaborated = Vec::with_capacity(bindings.len());
                for ((name, _, val), ann) in bindings.iter().zip(opened) {
                    // Same check_against restructuring as Expr::Let just
                    // above -- see its own comment. `cur_mode` is likewise
                    // untouched: each binding's annotation is its own
                    // self-contained expected type.
                    let (bound_ty, val_row, val3) = match ann {
                        Some((t, renames)) => {
                            let (val_row, val4) = check_against_rigid(arena, *val, &t, &renames, &val_ctx, spans, infer)?;
                            (t, val_row, val4)
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
                let ann = ann.map(|t| scoped_annotation(&t, infer));
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
                let witness = list_length_var(&param_ty, infer).map(|var| IndexWitness::new(var, &param));
                let has_witness = witness.is_some();
                infer.index_witness.extend(witness);
                infer.repeatable_depth += 1;
                pending.push(PendingElab::Fun { param, param_ty, witness: has_witness });
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
    // Case B construction, base case (design spec 2026-09-20 sec 4):
    // give the tail one extra dispatch option alongside plain
    // elaborate_node, the same tail-dispatch point Phase 3 built for
    // If/Match's own Check-mode treatment. When it doesn't qualify (or
    // qualifies but the actual type isn't consistent with the base
    // alternative), this falls through to elaborate_node's own Synth
    // result, and the UNCHANGED trailing coerce/unify_fits check right
    // below does today's ordinary generic fallback -- correctly
    // rejecting a value that's neither shape.
    let (mut result_ty, mut result_row, mut result_expr) = match cur_mode {
        Mode::Check(expected) => match case_b_base_candidate(arena, cur_expr, expected, infer) {
            Some((id, base_alt, step_alt, idx)) => {
                let (actual_ty, row, expr2) = elaborate(arena, cur_expr, &cur_ctx, spans, infer)?;
                // DEVIATION from the task-6 brief's literal text: the
                // brief's own condition was bare `consistent(&actual_ty,
                // &base_alt)`. `consistent` treats Dyn/Var as
                // universally consistent with anything (types.rs's own
                // doc comment) -- exactly the "no real information here"
                // signal this codebase treats specially everywhere else
                // (pattern_could_match, build_shape_predicate, etc.), not
                // proof that a value genuinely IS the base alternative.
                // Firing unify_index_expr(idx, 0) on that trivial
                // consistency force-committed the scrutinee's index to 0
                // for values that were never actually shown to be the
                // base case -- confirmed by a real regression in
                // `case_b_step_case_hypothesis_and_retyping_enable_a_real_recursive_proof`
                // (a pre-existing, passing Phase 4 test): its true/false
                // arms deliberately use `let x: Dyn = 0 in x` bodies, and
                // this hook's tail dispatch (reached again for the
                // Let-peeled `x`) wrongly asserted index 0 against the
                // ALREADY-hypothesized `n - 1`, a real conflict. Excluding
                // Dyn/Var here restores the original fallback (falls to
                // the trailing unify_fits, which is permissive with Dyn/
                // Var exactly as before this task) while keeping the
                // genuine case (a real, concrete base-alternative value
                // like a `false` literal) working exactly as the brief
                // intended.
                //
                // Final-review Finding 1: `consistent` is symmetric in
                // what counts as "universally consistent" -- Dyn/Var is
                // permissive on EITHER side, not just `actual_ty`'s. A
                // qualifying named union whose BASE alternative happens to
                // be Dyn/Var (e.g. `type T = Dyn | (Int, T)`) makes
                // `consistent(actual_ty, base_alt)` trivially true for ANY
                // `actual_ty`, including a genuinely step-shaped one (a
                // `w: T(2) = v` reference to an already-Indexed(Named(T),
                // 2) value) -- force-asserting index 0 against an index
                // that's already known to be 2, a genuine conflict.
                //
                // The review's own proposed fix was the literal mirror of
                // the existing actual_ty guard -- also exclude Dyn/Var on
                // `base_alt`. Verified BROKEN by empirical reproduction:
                // it also rejects `let x: T(0) = 5 in x`, the ordinary,
                // legitimate base case for exactly this kind of type (Dyn
                // base_alt matching a plain Int with no pre-existing index
                // at all) -- there `consistent`'s Dyn-side permissiveness
                // is exactly correct, not a false signal, since nothing
                // else in this codebase can ever prove a plain Int equals
                // an Indexed(Named(id), _) type otherwise (consistent()
                // has no Indexed-vs-non-Indexed arm at all -- see
                // types::consistent). Excluding Dyn/Var base_alt
                // unconditionally throws that legitimate case out with the
                // bug.
                //
                // The actual discriminator isn't "is base_alt Dyn/Var" --
                // it's "does actual_ty ALREADY carry its own, independently
                // established index" (i.e. `actual_ty` itself resolves to
                // Type::Indexed). Only then does force-asserting index 0
                // risk overriding real, already-known index information;
                // a plain concrete actual_ty (Int, Bool, Tuple, Fun, ...)
                // has no pre-existing index to conflict with, so treating
                // it as the base case is always sound. This ADDS an
                // Indexed(..) exclusion to the original actual_ty Dyn/Var
                // exclusion (both are still needed -- Dyn/Var alone would
                // miss the bug below, and Indexed alone would reopen the
                // Task 6 regression `case_b_step_case_hypothesis_and_...`
                // that motivated the original Dyn/Var exclusion). The
                // Indexed exclusion only applies when base_alt is itself
                // Dyn/Var -- an Indexed actual_ty against a concrete,
                // non-Dyn/Var base_alt (e.g. `type T = Vec(3) | (Int, T)`)
                // still gets a chance here, since the generic fallback's
                // `consistent` has no Indexed-vs-Named arm to catch that
                // case on its own.
                //
                // Confirmed by direct repro: `w: T(2) = v` (actual_ty =
                // Indexed(Named(T), 2), base_alt = Dyn) now falls through
                // to the trailing unify_fits, which already has full,
                // correct Indexed-vs-Indexed handling via unify() -- while
                // `let x: T(0) = 5 in x` and the deeper `let v: T(2) =
                // (1, (2, 5))` (whose innermost `5` is checked against a
                // still-symbolic `T(2 - 1 - 1)`) both keep working, since
                // Int is never Type::Indexed; and `type T = Vec(3) |
                // (Int, T) in let b: T(0) = [1,2,3] in ...` (actual_ty =
                // Indexed(List,3), base_alt = Indexed(List,3), not
                // Dyn/Var) still unifies index 0 correctly.
                let actual_is_indexed = matches!(actual_ty, Type::Indexed(..));
                let base_is_dyn_or_var = matches!(base_alt, Type::Dyn | Type::Var(_));
                if is_step_shaped_tuple(&actual_ty, &base_alt, &step_alt, &id, &idx, infer) {
                    // Already-synthesized step-shaped Tuple (see the helper).
                    (expected.clone(), row, expr2)
                } else if !matches!(actual_ty, Type::Dyn | Type::Var(_))
                    && !(actual_is_indexed && base_is_dyn_or_var)
                    && consistent(&actual_ty, &base_alt)
                {
                    unify_index_expr(&idx, &IndexExpr::Lit(0), infer, spans[cur_expr])?;
                    (expected.clone(), row, expr2)
                } else {
                    (actual_ty, row, expr2)
                }
            }
            None => elaborate_node(arena, cur_expr, &cur_ctx, spans, infer, cur_mode)?,
        },
        Mode::Synth => elaborate_node(arena, cur_expr, &cur_ctx, spans, infer, cur_mode)?,
    };
    if let Mode::Check(expected) = cur_mode {
        result_expr = coerce(arena, result_expr, &result_ty, expected, spans[cur_expr], &CheckEnv::new(infer))?;
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
            PendingElab::Fun { param, param_ty, witness } => {
                infer.repeatable_depth -= 1;
                // Matches the original Lambda arm: the body's row is
                // embedded in the Fun type, not propagated -- evaluating
                // the Lambda expression itself is always pure.
                result_ty = Type::Fun(Rc::new(param_ty.clone()), result_row, Rc::new(result_ty));
                result_row = EffectRow::pure();
                if witness {
                    let w = infer.index_witness.pop().expect("a Fun frame's witness is the innermost one");
                    if w.used.get() {
                        result_expr = w.bind(arena, &param, result_expr);
                    }
                }
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
        let checked = coerce(arena, result_expr, &result_ty, expected, spans[expr], &CheckEnv::new(infer))?;
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

// check_against for an annotated binding's value: a FUNCTION signature's
// still-unbound index variables are rigid for the duration, so the body
// cannot pin them (e.g. `n := 3`) by unification. Only variables THIS call
// newly marked are released afterwards, so nested annotated bindings
// compose. Non-function annotations (`let x: Vec(n) = [1, 2]`) keep their
// flexible, binding-style variables: there `n` is an existential to be
// inferred from the value, not a signature parameter. See spec 2026-10-07.
// `renames` (from open_signature) are in force for the value only, then
// restored, the same way the rigid set is.
fn check_against_rigid(
    arena: &mut Arena,
    val: ExprRef,
    ann: &Type,
    renames: &[(String, String)],
    ctx: &Ctx,
    spans: &SpanMap,
    infer: &mut InferCtx,
) -> Result<(EffectRow, ExprRef), TypeError> {
    let signature_vars = if matches!(ann, Type::Fun(..)) { free_index_vars_resolved(ann, infer) } else { BTreeSet::new() };
    let added: Vec<String> = signature_vars.into_iter().filter(|v| infer.rigid_index.insert(v.clone())).collect();
    let prior: Vec<(String, Option<String>)> =
        renames.iter().map(|(from, to)| (from.clone(), infer.index_rename.insert(from.clone(), to.clone()))).collect();
    let result = check_against(arena, val, ann, ctx, spans, infer);
    for (from, old) in prior.into_iter().rev() {
        match old {
            Some(old) => infer.index_rename.insert(from, old),
            None => infer.index_rename.remove(&from),
        };
    }
    for v in &added {
        infer.rigid_index.remove(v);
    }
    result
}

// An annotation as read inside an annotated binding's value: names that an
// enclosing open_signature renamed denote the renamed variable.
fn scoped_annotation(ann: &Type, infer: &InferCtx) -> Type {
    if infer.index_rename.is_empty() {
        return ann.clone();
    }
    let map: HashMap<String, IndexExpr> =
        infer.index_rename.iter().map(|(from, to)| (from.clone(), IndexExpr::Var(to.clone()))).collect();
    subst_type(ann, &HashMap::new(), &HashMap::new(), &map)
}

// A function signature's index variables are its own parameters. Index
// variable names are program-global strings, and a non-function annotation
// (`let a: Vec(n) = [1, 2]`) binds its `n` existentially for the rest of the
// program, so a later signature reusing the name would otherwise inherit
// `n = 2` instead of quantifying. Rename such names (bound in index_subst,
// and not an enclosing signature's own rigid variable, which a nested
// signature deliberately shares) to fresh ones. Returns the renamed
// annotation and the renames to scope over the value (check_against_rigid).
type Renames = Vec<(String, String)>;

fn open_signature(ann: &Type, infer: &InferCtx) -> (Type, Renames) {
    let ann = scoped_annotation(ann, infer);
    if !matches!(ann, Type::Fun(..)) {
        return (ann, Vec::new());
    }
    let renames: Renames = index_vars_as_written(&ann)
        .into_iter()
        .filter(|v| infer.index_subst.contains_key(v) && !infer.rigid_index.contains(v))
        .map(|v| {
            let fresh = fresh_index_name(&v);
            (v, fresh)
        })
        .collect();
    if renames.is_empty() {
        return (ann, renames);
    }
    let map: HashMap<String, IndexExpr> = renames.iter().map(|(from, to)| (from.clone(), IndexExpr::Var(to.clone()))).collect();
    (subst_type(&ann, &HashMap::new(), &HashMap::new(), &map), renames)
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
        // Emitted by elaboration itself (Dyn-boundary checks), never parsed.
        Expr::Check(..) => unreachable!("Expr::Check is typecheck output, never elaboration input"),
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
            // Case B construction, step case (design spec 2026-09-20
            // sec 4): checked against an Indexed(Named(id), idx) type
            // whose named union qualifies and whose step alternative
            // is a same-arity Tuple, check each non-recursive item
            // against its corresponding step_alt element and the
            // recursive item (self_ref_position) against
            // Named(id)(idx - 1) -- Case A's Cons arm (Task 2),
            // generalized from a fixed List/Cons shape to an
            // arbitrary 2-alternative named union.
            if let Mode::Check(expected) = mode {
                if let Type::Indexed(wrapped, idx) = infer.resolve_deep(expected) {
                    if let Type::Named(id) = wrapped.as_ref() {
                        if let Some((_, step_alt)) = qualifying_named_alternatives(id, &infer.named_types) {
                            if let Type::Tuple(step_items) = &step_alt {
                                if step_items.len() == items.len() {
                                    if let Some(rec_pos) = self_ref_position(&step_alt, id) {
                                        let mut row = EffectRow::pure();
                                        let mut refs = Vec::with_capacity(items.len());
                                        for (i, item) in items.into_iter().enumerate() {
                                            let (item_row, item2) = if i == rec_pos {
                                                let rec_expected = Type::Indexed(Rc::new(Type::Named(id.clone())), Rc::new(IndexExpr::Sub(idx.clone(), Rc::new(IndexExpr::Lit(1)))));
                                                check_against(arena, item, &rec_expected, ctx, spans, infer)?
                                            } else {
                                                check_against(arena, item, &step_items[i], ctx, spans, infer)?
                                            };
                                            row = EffectRow::union(&row, &item_row);
                                            refs.push(item2);
                                        }
                                        return Ok((expected.clone(), row, arena.push(Expr::Tuple(refs))));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let mut row = EffectRow::pure();
            let mut tys = Vec::with_capacity(items.len());
            let mut refs = Vec::with_capacity(items.len());
            let expected_items = match mode {
                Mode::Check(expected) => match infer.resolve_deep(expected) {
                    Type::Tuple(ts) if ts.len() == items.len() => Some(ts),
                    _ => None,
                },
                Mode::Synth => None,
            };
            for (i, item) in items.into_iter().enumerate() {
                let (item_ty, item_row, item2) = match expected_items.as_ref().map(|ts| &ts[i]).filter(|t| contains_indexed(t)) {
                    Some(want) => {
                        let (r, e) = check_against(arena, item, want, ctx, spans, infer)?;
                        (want.clone(), r, e)
                    }
                    None => elaborate(arena, item, ctx, spans, infer)?,
                };
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
        // choice (a native Check) for the exact same reason its own
        // comment gives: a bare call would leave a shape mismatch to
        // machine.rs's own differently-worded (and differently-styled)
        // panic instead of this file's ordinary "type error: expected X,
        // found Y" message.
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
                build_shape_check(arena, target2, &required, &CheckEnv::at(infer, spans[target]), &HashSet::new())
            } else {
                target2
            };
            let call = build_str_call(arena, "get_field", checked_target, &name);
            Ok((field_ty, target_row, call))
        }

        Expr::ListLit(items) => {
            let item_count = items.len();
            let mut row = EffectRow::pure();
            let mut elem_ty: Option<Type> = None;
            let mut refs = Vec::with_capacity(items.len());
            let mut item_tys = Vec::with_capacity(items.len());
            let want_elem = match mode {
                Mode::Check(expected) => match infer.resolve_deep(expected) {
                    Type::List(t) if contains_indexed(&t) => Some((*t).clone()),
                    _ => None,
                },
                Mode::Synth => None,
            };
            for item in items {
                let (item_ty, item_row, item2) = match &want_elem {
                    Some(want) => {
                        let (r, e) = check_against(arena, item, want, ctx, spans, infer)?;
                        (want.clone(), r, e)
                    }
                    None => elaborate(arena, item, ctx, spans, infer)?,
                };
                row = EffectRow::union(&row, &item_row);
                refs.push(item2);
                item_tys.push(item_ty.clone());
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
            let elem_ty = elem_ty.unwrap_or_else(|| infer.fresh_var("elem"));
            if matches!(mode, Mode::Synth) {
                cast_join_arms(arena, &mut refs, &item_tys, &elem_ty, infer);
            }
            // Case A construction (design spec 2026-09-20 sec 1): checked
            // against an Indexed(List(_), _) expected type, synthesize the
            // PRECISE length-indexed type instead of a plain List --
            // elaborate_mode's own trailing Check-mode enforcement (the
            // unify_fits call right after this function returns) then
            // confirms this literal's actual length against whatever index
            // the expected type carries, so no explicit check is needed here.
            if let Mode::Check(expected) = mode {
                if let Type::Indexed(wrapped, _) = infer.resolve_deep(expected) {
                    if matches!(wrapped.as_ref(), Type::List(_)) {
                        let list_ty = Type::Indexed(Rc::new(Type::List(Rc::new(elem_ty))), Rc::new(IndexExpr::Lit(item_count as i64)));
                        return Ok((list_ty, row, arena.push(Expr::ListLit(refs))));
                    }
                }
            }
            let list_ty = Type::List(Rc::new(elem_ty));
            Ok((list_ty, row, arena.push(Expr::ListLit(refs))))
        }

        Expr::Let(..) | Expr::LetRec(..) | Expr::Lambda(..) => {
            unreachable!("Let/LetRec/Lambda are peeled by elaborate's chain-flattening loop")
        }

        Expr::App(f, a) => {
            let (f_ty, f_row, f2) = elaborate(arena, f, ctx, spans, infer)?;
            let f_ty = infer.resolve(&f_ty);
            // A callee whose parameter type mentions an Indexed type
            // (`Vec(n)`, `T(n)`, bare or nested in a tuple/list) gets its argument CHECKED against it, so a
            // list/tuple literal argument can satisfy the index -- Synth
            // alone gives a plain un-indexed List/Tuple that never can.
            // Deliberately limited to such parameters, and to ones with no
            // free row variables: Check mode would otherwise drop the
            // argument's own synthesized type, which bind_row_vars below
            // needs to bind effect-row variables.
            let (a_ty, a_row, a2) = match &f_ty {
                Type::Fun(param_ty, ..) if {
                    let p = infer.resolve_deep(param_ty);
                    contains_indexed(&p) && free_row_vars(&p).is_empty()
                } => {
                    let param_resolved = infer.resolve_deep(param_ty);
                    let (row, a2) = check_against(arena, a, &param_resolved, ctx, spans, infer)?;
                    (param_resolved, row, a2)
                }
                _ => elaborate(arena, a, ctx, spans, infer)?,
            };
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
                    // Read the flag now: unify_fits below binds the variable
                    // to the argument's type and with it the evidence.
                    let sunk = infer.is_dyn_sunk(&param_ty_resolved);
                    let a3 = coerce_check(arena, a2, &a_ty, &param_ty_resolved, spans[a], &CheckEnv::new(infer))?;
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
                    // param_ty_resolved is stale after unify_fits (it still
                    // shows the Var an unannotated `apply(f)` just bound), so
                    // the cast re-resolves both sides.
                    // A parameter variable that feeds a Dyn sink in the callee's
                    // body is cast like a Dyn parameter.
                    let cast_to = if sunk { &Type::Dyn } else { param_ty };
                    let a3 = coerce_cast(arena, a3, &a_ty, cast_to, &CheckEnv::at(infer, spans[a]));
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
                    // through the same native Check every other boundary
                    // uses, rather than leaving it to a differently-worded
                    // panic in machine.rs. Can't know what it might perform,
                    // so the call contributes an unknown (Dyn) row.
                    let f3 = build_shape_check(arena, f2, &any_fun(), &CheckEnv::at(infer, spans[f]), &HashSet::new());
                    // The argument crosses into an unknown function, so it is
                    // cast to Dyn like any other Dyn position (a typed
                    // function argument gets its wrapper).
                    infer.note_dyn_sink(&a_ty);
                    let a3 = coerce_cast(arena, a2, &a_ty, &Type::Dyn, &CheckEnv::at(infer, spans[a]));
                    (EffectRow::Dyn, Type::Dyn, arena.push(Expr::App(f3, a3)))
                }
                other => return Err(TypeError(format!("cannot call a value of type {other}"), spans[f])),
            };
            let row = EffectRow::union(&EffectRow::union(&f_row, &a_row), &call_row);
            Ok((ret_ty, row, app2))
        }

        Expr::BinOp(op, l, r) => {
            // Case A construction (design spec 2026-09-20 sec 1, Cons half):
            // checked against an Indexed(List(_), _) expected type, `::`
            // checks its tail against a length-decremented Vec and its head
            // against the element type, instead of the unconditional Synth
            // elaboration below (which would throw away `expected` entirely --
            // same reason ListLit needed its own Mode-aware branch, Task 1).
            if op == BinOp::Cons {
                if let Mode::Check(expected) = mode {
                    if let Type::Indexed(wrapped, idx) = infer.resolve_deep(expected) {
                        if let Type::List(elem_ty) = wrapped.as_ref() {
                            let elem_ty = (**elem_ty).clone();
                            let tail_expected = Type::Indexed(wrapped.clone(), Rc::new(IndexExpr::Sub(idx.clone(), Rc::new(IndexExpr::Lit(1)))));
                            let (l_row, l2) = check_against(arena, l, &elem_ty, ctx, spans, infer)?;
                            let (r_row, r2) = check_against(arena, r, &tail_expected, ctx, spans, infer)?;
                            let row = EffectRow::union(&l_row, &r_row);
                            return Ok((expected.clone(), row, arena.push(Expr::BinOp(op, l2, r2))));
                        }
                    }
                }
            }
            let (l_ty, l_row, l2) = elaborate(arena, l, ctx, spans, infer)?;
            let (r_ty, r_row, r2) = elaborate(arena, r, ctx, spans, infer)?;
            let row = EffectRow::union(&l_row, &r_row);
            match op {
                // Arithmetic and ordering: both operands must be Int or
                // Float (mixing promotes to Float) -- see coerce_numeric's
                // own doc comment for why this stays a local special case
                // rather than a general Int<->Float consistent() rule.
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Lt => {
                    let l3 = coerce_numeric(arena, l2, &l_ty, spans[l], &CheckEnv::new(infer))?;
                    let r3 = coerce_numeric(arena, r2, &r_ty, spans[r], &CheckEnv::new(infer))?;
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
                        coerce(arena, l2, &l_ty, &r_ty, spans[l], &CheckEnv::new(infer))?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty, spans[r], &CheckEnv::new(infer))?
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
                        coerce(arena, l2, &l_ty, &r_ty, spans[l], &CheckEnv::new(infer))?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty, spans[r], &CheckEnv::new(infer))?
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
            let c3 = coerce(arena, c2, &c_ty, &Type::Bool, spans[c], &CheckEnv::new(infer))?;
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
                    let mut arms = [t2, e2];
                    cast_join_arms(arena, &mut arms, &[t_ty, e_ty], &result_ty, infer);
                    let [t2, e2] = arms;
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
            let (payload_ty, payload_row, payload2) = elaborate(arena, payload, ctx, spans, infer)?;
            // The handler receives the payload as Dyn.
            infer.note_dyn_sink(&payload_ty);
            let payload2 = coerce_cast(arena, payload2, &payload_ty, &Type::Dyn, &CheckEnv::at(infer, spans[payload]));
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
                Mode::Check(_) => indexed_refinement_target(&resolved_scrut_ty, resolved_index.as_ref(), &infer.named_types),
                Mode::Synth => None,
            };
            // Scope the eventual restore to the ONE key this mechanism is
            // entitled to touch -- the hypothesis's own index variable --
            // not the whole index_subst map. A whole-map snapshot/restore
            // (this file's own unify_trial idiom, meant for a genuinely
            // SPECULATIVE unification that either fully commits or fully
            // rolls back) is the wrong shape here: an arm's own body-check
            // is not speculative, and can legitimately prove unrelated
            // index facts (an ambient Vec(k) binding's own real length,
            // say) that must survive the match, not be silently erased
            // alongside the hypothesis when it's reverted (critical review
            // finding, Phase 4 final review -- confirmed reproducible via
            // ordinary source: an in-arm `let p: Vec(2) = w in ..` proof
            // used to get wiped by the very next restore).
            let index_key: Option<String> = refinement_target.as_ref().map(|t| match t {
                IndexedRefinementTarget::List { index_var, .. } | IndexedRefinementTarget::Named { index_var, .. } => index_var.clone(),
            });
            let prior_index_value: Option<Option<IndexExpr>> = index_key.as_ref().map(|k| infer.index_subst.get(k).cloned());
            // Restores `infer.index_subst`'s hypothesis key to its value
            // from BEFORE this match ever ran -- re-inserting the prior
            // binding if there was one, or removing the key entirely if it
            // was unbound. Shared by the per-arm restore (start of every
            // iteration) and the final restore (after the whole loop) so
            // both do the IDENTICAL key-scoped logic, never the old
            // whole-map assignment.
            fn restore_index_key(infer: &mut InferCtx, index_key: &Option<String>, prior_index_value: &Option<Option<IndexExpr>>) {
                if let (Some(k), Some(prior)) = (index_key, prior_index_value) {
                    match prior {
                        Some(v) => {
                            infer.index_subst.insert(k.clone(), v.clone());
                        }
                        None => {
                            infer.index_subst.remove(k);
                        }
                    }
                }
            }
            // The fresh cons-arm length variable m is made rigid and stays so: it is
            // a unique name, and pins of it can alias through other variables
            // after the arm ends.
            let mut row = scrut_row;
            let mut result_ty: Option<Type> = None;
            let mut new_arms = Vec::with_capacity(arms.len());
            let mut arm_tys = Vec::with_capacity(arms.len());
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
                // Task 3 Fix 1: restore infer.index_subst's hypothesis
                // key to its pre-match value BEFORE deciding THIS arm's
                // own hypothesis (if any) -- not just once, after the
                // whole loop. index_subst is one shared map, mutated in
                // place by each eligible arm's own `insert` call below;
                // without this per-arm restore, an arm that injects no
                // hypothesis of its own (the `_ => {}`/no-override
                // fallthroughs just below) would silently INHERIT
                // whatever a PRECEDING sibling arm left behind in that
                // same shared map, even though reaching THIS arm's own
                // pattern shape implies nothing at all about the real
                // hypothesis. Concretely: `match v | [] -> .. | h :: (h2
                // :: t2) -> <body>` -- the compound-tail arm's own
                // <body> used to wrongly see `n` still resolved to the
                // `[]` arm's own 0, purely because `[]` happened to run
                // first in source order. Restoring here, before every
                // arm's own injection (if any) runs, makes every arm
                // start from the SAME true ambient state regardless of
                // processing order -- confirmed empirically (see this
                // task's own report): reverting to a single restore
                // after the whole loop flips
                // `stale_hypothesis_does_not_leak_into_a_later_ineligible_sibling_arm`
                // from passing to failing. Critical review finding (Phase
                // 4 final review): this restore used to reassign the
                // WHOLE index_subst map back to a pre-match clone, which
                // also erased any UNRELATED index fact a preceding arm's
                // own body legitimately proved (e.g. an ambient Vec(k)
                // binding's real length) -- restore_index_key touches
                // only the hypothesis's own key, leaving everything else
                // an arm proves intact.
                restore_index_key(infer, &index_key, &prior_index_value);
                let mut arm_ctx = bind_pattern_vars(ctx, pat, &scrut_ty, infer, spans[expr], &HashSet::new())?;
                // Hypothesis injection (spec section 5): base case implies
                // the scrutinee's own index is 0; step case mints a fresh
                // index variable, re-types the self-referential
                // sub-binding as Indexed(_, fresh) (overriding
                // bind_pattern_vars's own plain binding via a second
                // `extend` call), and hypothesizes the scrutinee's own
                // index is `fresh + 1`. Case A (List) derives base/step
                // from the fixed []/:: patterns directly; Case B (Named+
                // Union) derives the SAME shape of fact by asking which
                // of the type's own two alternatives this specific arm's
                // pattern corresponds to (reusing pattern_could_match's
                // existing per-alternative logic, spec section 5 Case B
                // point 4 -- no new exhaustiveness logic needed). Either
                // case, injecting into infer.index_subst (not a separate
                // ctx-rewrite pass) is sufficient: every Type::Indexed
                // comparison this arm's own body-check can reach --
                // unify, unify_index_expr, and transitively unify_fits/
                // check_against -- already resolves through index_subst
                // on demand (Phase 1-3), so an ambient binding sharing
                // this same index variable, or `expected` itself if it
                // mentions it, sees the hypothesis too, with no extra
                // machinery.
                if let Some(target) = &refinement_target {
                    match target {
                        IndexedRefinementTarget::List { index_var: n_name, elem_ty } => match pat {
                            Pattern::List(items) if items.is_empty() => {
                                infer.index_subst.insert(n_name.clone(), IndexExpr::Lit(0));
                            }
                            Pattern::Cons(_, tail) => {
                                if let Pattern::Var(tail_name) = &**tail {
                                    let m = fresh_index_name("m");
                                    let hypothesis = IndexExpr::Add(Rc::new(IndexExpr::Var(m.clone())), Rc::new(IndexExpr::Lit(1)));
                                    infer.index_subst.insert(n_name.clone(), hypothesis);
                                    infer.rigid_index.insert(m.clone());
                                    let refined_tail_ty = Type::Indexed(Rc::new(Type::List(elem_ty.clone())), Rc::new(IndexExpr::Var(m)));
                                    arm_ctx = extend(&arm_ctx, tail_name, refined_tail_ty);
                                }
                                // A compound/nested tail pattern: no override,
                                // no hypothesis -- v1 scope boundary, see this
                                // plan's own Global Constraints.
                            }
                            _ => {}
                        },
                        IndexedRefinementTarget::Named { index_var: n_name, id, base_alt, step_alt } => {
                            // `base_alt`/`step_alt` are themselves already-
                            // unfolded alternatives of `id` (see
                            // `qualifying_named_alternatives`), so this
                            // probe is conceptually already mid-unfold of
                            // `id` -- pass `{id}`, not an empty set, for
                            // consistency with the `visiting`-threading
                            // discipline the rest of this phase established
                            // (the exact shape whose absence caused
                            // bind_pattern_vars's own fix-round-1 bug).
                            // Doesn't change behavior today (verified by
                            // trace: neither alternative can itself unfold
                            // back to `id` without violating
                            // `qualifying_named_alternatives`'s own
                            // occurrence-count check), just defense in
                            // depth.
                            let visiting_id = HashSet::from([id.clone()]);
                            let matches_base = pattern_could_match(pat, base_alt, &infer.named_types, &visiting_id);
                            let matches_step = pattern_could_match(pat, step_alt, &infer.named_types, &visiting_id);
                            if matches_base && !matches_step {
                                infer.index_subst.insert(n_name.clone(), IndexExpr::Lit(0));
                            } else if matches_step && !matches_base {
                                if let Some(Pattern::Var(self_ref_name)) = self_ref_pattern(pat, step_alt, id) {
                                    let m = fresh_index_name("m");
                                    let hypothesis = IndexExpr::Add(Rc::new(IndexExpr::Var(m.clone())), Rc::new(IndexExpr::Lit(1)));
                                    infer.index_subst.insert(n_name.clone(), hypothesis);
                                    infer.rigid_index.insert(m.clone());
                                    let refined_ty = Type::Indexed(Rc::new(Type::Named(id.clone())), Rc::new(IndexExpr::Var(m)));
                                    arm_ctx = extend(&arm_ctx, self_ref_name, refined_ty);
                                }
                                // The self-referential position is bound
                                // by something other than a bare Var (or
                                // isn't found at all, e.g. a step
                                // alternative shape self_ref_pattern
                                // doesn't recognize): no override, no
                                // hypothesis -- same v1 scope boundary as
                                // Case A's own compound-tail restriction.
                            }
                            // Ambiguous (this arm's own pattern could
                            // match BOTH alternatives, e.g. a bare `_`/Var
                            // covering the whole scrutinee) or neither:
                            // no hypothesis, the same safe fallback as
                            // Case A's own catch-all `_ => {}` just above.
                        }
                    }
                }
                // Same Bool coercion as If's own cond -- a Dyn-typed guard
                // gets a runtime is_bool check inserted, same as
                // everywhere else Dyn meets an expected concrete type.
                let guard2 = match guard {
                    Some(g) => {
                        let (guard_ty, guard_row, g2) = elaborate(arena, *g, &arm_ctx, spans, infer)?;
                        let g3 = coerce(arena, g2, &guard_ty, &Type::Bool, spans[*g], &CheckEnv::new(infer))?;
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
                        arm_tys.push(arm_ty.clone());
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
            // Final restore, after the WHOLE arm loop completes -- on top
            // of the per-arm restore each iteration now ALSO does at its
            // own start (Task 3 Fix 1, see the comment there for why a
            // once-only restore silently leaked a preceding arm's own
            // hypothesis into a later, unrelated one). This last restore
            // is what makes a per-arm hypothesis fully undone once the
            // LAST arm's own body-check finishes too, and is what keeps
            // the hypothesis's own key from leaking into code after the
            // match entirely -- same key-scoped `restore_index_key`
            // helper as the per-arm restore above (critical review
            // finding: a whole-map restore here erased unrelated index
            // facts an arm's own body legitimately proved, same bug as
            // the per-arm restore's, just triggered once at the end
            // instead of every iteration).
            restore_index_key(infer, &index_key, &prior_index_value);
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
            if matches!(mode, Mode::Synth) {
                let mut bodies: Vec<ExprRef> = new_arms.iter().map(|(_, _, b)| *b).collect();
                cast_join_arms(arena, &mut bodies, &arm_tys, &final_ty, infer);
                for (arm, body) in new_arms.iter_mut().zip(bodies) {
                    arm.2 = body;
                }
            }
            Ok((final_ty, row, arena.push(Expr::Match(scrutinee2, Rc::new(new_arms)))))
        }

        // Constructing the handler value is pure -- the clause body's own
        // effects (including what `resume` re-enters) aren't modeled here;
        // see the doc comment on `elaborate_mode`.
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let inner_ctx = extend(&extend(ctx, &payload_var, Type::Dyn), &resume_var, Type::Dyn);
            infer.repeatable_depth += 1;
            let body_result = elaborate(arena, body, &inner_ctx, spans, infer);
            infer.repeatable_depth -= 1;
            let (_, _, body2) = body_result?;
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
    resolve_obligations(arena, &infer);
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
