use std::cell::{Cell, OnceCell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::expr::{AnnotKind, AnnotSpec, Arena, BinOp, CheckSpec, Expr, ExprRef, Lazy, LenExpr, Lowered, Pattern, ConformSpec, SpanMap, Test, Then, TypeSource};
use crate::index_expr::{index_exprs_compare, IndexCmp, IndexExpr};
use crate::machine::{eval_pure_in, pure_type, Pure, Stuck, ANNOTATION_FUEL};
use crate::resolve::{group_kind, GroupKind, Resolved};
use crate::span::Span;
use crate::value::Value;
use rpds::{HashTrieMap, HashTrieSet};
use crate::types::{consistent, each_child, is_big, row_consistent, unfold, ById, EffectRow, Hc, Interned, NodeMemo, RecGroup, RecRef, RecVar, Rebuild, Summary, Type, Visited, FxMap, FxSet};
use crate::util::find_field;
mod relate;
use self::relate::{relate, relate_cast};
pub(crate) use self::relate::{Cause, Rel, Residual, Why};

// The Span is always an ORIGINAL (pre-elaboration) node's -- the one whose
// type was being checked when the error fired, always already in scope
// (elaborate_node's own `expr` parameter, or a child ExprRef it destructured
// from that same original node) at the point of construction. typecheck
// never needs to invent a span for a node IT synthesizes (a boundary check,
// a static contract): every error path returns before any such node
// is built.
#[derive(Debug)]
pub struct TypeError(pub String, pub Span);

// Why a unification failed, with its text not yet written: most failures are tried and dropped
// (unify_trial, a rescue), and writing the text of a big type costs far more than the
// unification did. `?` into a TypeError writes it (capped, Type::brief).
#[derive(Debug)]
pub(crate) enum Mismatch {
    Expected { expected: Type, found: Type, span: Span },
    Infinite { name: String, ty: Type, span: Span },
    CannotTell { from: Type, to: Type, span: Span },
    Text(TypeError),
}

impl From<TypeError> for Mismatch {
    fn from(e: TypeError) -> Mismatch {
        Mismatch::Text(e)
    }
}

impl From<Mismatch> for TypeError {
    fn from(m: Mismatch) -> TypeError {
        match m {
            Mismatch::Expected { expected, found, span } => TypeError(format!("type mismatch: expected {}, found {}", expected.brief(), found.brief()), span),
            Mismatch::Infinite { name, ty, span } => TypeError(format!("infinite type: {name} occurs in {}", ty.brief()), span),
            Mismatch::CannotTell { from, to, span } => cannot_tell(&from, &to, span),
            Mismatch::Text(e) => e,
        }
    }
}

// A binding's type, plus the variables generalized over it -- quantified
// fresh at every use, the way ML/Haskell generalize a `let`-bound type.
// Three sorts of variable can be quantified. Row variables come from a
// `->{e}` the user wrote, but the name is never kept: the binder mints a
// fresh `e#k` for it (open_annotation, see Ctx::annot), as it does for a
// written index name; type variables
// are manufactured by real unification (`unify`, via `InferCtx::fresh_var`)
// at many sites, never written by a user -- see Type::Var's own doc comment;
// index variables are the same idea one level down, for a name appearing
// inside a `Type::Indexed`'s own `IndexExpr` rather than as a `Type` itself
// (manufactured by `unify_index_expr`, via `index_subst`, not `fresh_var` --
// a separate namespace, see `InferCtx::index_subst`'s own doc comment). All
// three stay simple name substitution (extend_generalized computes the set
// once at the `let`, lookup renames them to fresh names at each reference),
// not a second unification engine of its own.
//
// THE RULE, one place for all three sorts (and any future one):
// generalize(ctx, ty) = freevars(ty) MINUS freevars(ctx) -- never quantify
// over a name that's still free somewhere still open in the enclosing
// scope. The mechanism is the `open` set every Ctx carries (see Ctx):
// `freevars(ctx)` is read from it, not by walking the bindings, so a `let`
// costs the size of its own type plus the number of variables currently
// open, however many bindings precede it. Index variables a pending
// obligation still needs are excluded on top (see `generalizable`).
pub(crate) struct Scheme {
    // The variables generalized over this binding, of all three sorts,
    // sorted by sort then name. lookup mints a fresh name per reference.
    quantified: Vec<TyVar>,
    ty: Type,
    // A `let`/`let rec` binder's definition, which an annotation naming it
    // evaluates (CheckDef). None for a binder whose value exists only at run
    // time: a parameter, a pattern variable, a handler's payload or resume.
    def: Option<Rc<CheckDef>>,
}

impl Scheme {
    fn mono(ty: Type) -> Scheme {
        Scheme { quantified: Vec::new(), ty, def: None }
    }
}

// A type variable of any of the three sorts a Scheme can quantify: an
// effect-row variable, a Type::Var, or an index variable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum TyVar {
    Row(String),
    Type(String),
    Index(String),
}

// Every variable of every sort `ty` mentions, as written: no substitution
// is consulted (callers resolve first, or reopen the names later).
pub(crate) fn vars_in(ty: &Type) -> BTreeSet<TyVar> {
    fn go(ty: &Type, out: &mut BTreeSet<TyVar>, seen: &mut Visited) {
        if !ty.summary().any(Summary::VAR | Summary::ROWVAR | Summary::INDEXED) || !seen.first(ty) {
            return;
        }
        match ty {
            Type::Var(name) => {
                out.insert(TyVar::Type(name.clone()));
            }
            Type::Fun(param, row, ret) => {
                if let EffectRow::Var(name) = row {
                    out.insert(TyVar::Row(name.clone()));
                }
                go(param, out, seen);
                go(ret, out, seen);
            }
            Type::List(elem) => go(elem, out, seen),
            Type::Tuple(items) | Type::Union(items) => items.iter().for_each(|t| go(t, out, seen)),
            Type::Record(fields) => fields.iter().for_each(|(_, t)| go(t, out, seen)),
            Type::Indexed(wrapped, index) => {
                go(wrapped, out, seen);
                out.extend(free_index_vars(index).into_iter().map(TyVar::Index));
            }
            Type::Rec(g, _) => g.0.bodies.iter().for_each(|b| go(b, out, seen)),
            Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) | Type::RecVar(_) => {}
        }
    }
    let mut out = BTreeSet::new();
    go(ty, &mut out, &mut Visited::new());
    out
}

// The index-variable names among `vars`.
fn index_names(vars: BTreeSet<TyVar>) -> BTreeSet<String> {
    vars.into_iter().filter_map(|v| if let TyVar::Index(name) = v { Some(name) } else { None }).collect()
}

// The variables a stored name stands for now: a Type or Index variable that
// unification has since bound stands for the variables of what it was bound
// to; a Row variable is never bound.
fn reopen(var: &TyVar, infer: &InferCtx) -> BTreeSet<TyVar> {
    match var {
        TyVar::Row(_) => BTreeSet::from([var.clone()]),
        TyVar::Type(name) => {
            vars_in(&infer.resolve_deep(&Type::Var(name.clone())))
        }
        TyVar::Index(name) => {
            free_index_vars(&infer.resolve_index_deep(&IndexExpr::Var(name.clone()))).into_iter().map(TyVar::Index).collect()
        }
    }
}

// A scope: the visible bindings, the variables still open in it (i.e. free in
// some binding's type and not quantified by that binding), and the annotation
// variables its binders introduced. All three are persistent, so extending a
// scope is cheap and an enclosing scope is untouched: a `let` may quantify
// only a variable no enclosing scope has open, and an annotation variable is
// visible exactly in the scope that introduced it.
//
// Constructing and dropping a 50k-binding chain on the default stack is
// pinned by let_chain_is_o1_stack_construction_and_teardown. A `CheckDef`
// holds the scope before its binder, so a drop path through defs is the
// thing to re-check if that test ever overflows.
#[derive(Clone)]
pub(crate) struct Ctx {
    // Innermost binding of each name; a shadowed one is no longer reachable.
    binds: HashTrieMap<String, Rc<Scheme>>,
    // As written when each binding was added, union-only: a shadowed
    // binding keeps its variables open (its type may still mention them
    // through values already built from it). A binding's own quantified
    // names are not added: the scheme is sealed, since lookup renames them
    // before any later read. open_vars reads the names
    // through the substitution as it stands, so a rollback in `unify_trial`
    // or a match arm's hypotheses cannot leave it stale.
    open: HashTrieSet<TyVar>,
    // The annotation variables in scope, keyed by the row or index name as the user wrote it
    // (never a `TyVar::Type`): the fresh variable each denotes here (open_annotation). A scoped
    // annotation variable is also in `open`: it is free in the context.
    annot: HashTrieMap<TyVar, AnnotVar>,
}

// What a written annotation variable stands for in a scope. `universal`: a function
// signature's or a lambda parameter's variable, one an inner binder of the same name
// shares; otherwise an existential, one the checked value fixes. Only a signature's
// variables are rigid (`rigid_index`: the body may not pin them); a lambda parameter's
// stays flexible, so a body that pins it is rejected by the ordinary mismatch, not by
// "fixed by the signature". A body annotation reusing the name shares the variable
// either way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AnnotVar {
    pub(crate) var: TyVar,
    pub(crate) universal: bool,
}

impl Ctx {
    pub(crate) fn empty() -> Ctx {
        Ctx { binds: HashTrieMap::new(), open: HashTrieSet::new(), annot: HashTrieMap::new() }
    }

    // This scope with `entries` (as open_annotation introduced them) in scope and open.
    pub(crate) fn with_annot(&self, entries: &[(TyVar, AnnotVar)]) -> Ctx {
        let mut ctx = self.clone();
        for (written, entry) in entries {
            ctx.open = ctx.open.insert(entry.var.clone());
            ctx.annot = ctx.annot.insert(written.clone(), entry.clone());
        }
        ctx
    }

    // The entry a written name has in this scope.
    pub(crate) fn annot_var(&self, written: &TyVar) -> Option<&AnnotVar> {
        self.annot.get(written)
    }

    fn get(&self, name: &str) -> Option<Rc<Scheme>> {
        self.binds.get(name).cloned()
    }

    fn bind(&self, name: &str, scheme: Scheme) -> Ctx {
        let mut open = self.open.clone();
        for var in vars_in(&scheme.ty) {
            if !scheme.quantified.contains(&var) && !open.contains(&var) {
                open = open.insert(var);
            }
        }
        Ctx { binds: self.binds.insert(name.to_string(), Rc::new(scheme)), open, annot: self.annot.clone() }
    }

    pub(crate) fn open_vars(&self, infer: &InferCtx) -> BTreeSet<TyVar> {
        self.open.iter().flat_map(|v| reopen(v, infer)).collect()
    }
}

// A group of direct `fun`s is two definitions (one for its own values, one for
// the code after it), so it is evaluated at most twice; any other `let`, a
// type group included, at most once.
// What a check-time evaluation needs to evaluate a `let`-bound name (spec
// section 1, type lets): the binder's right-hand side as the parser built it,
// and the scope it was checked in. The right-hand side is evaluated once, on
// first demand, in a frame of the values of the definitions it reads
// (def_value); an annotation that names the binder runs in a frame of the
// values of the definitions it reads (annotation_type, eval_annotation).
pub(crate) struct CheckDef {
    // Its place in InferCtx::defs, which is creation order: a definition reads
    // only names bound before it, so every dependency has a smaller index.
    idx: usize,
    node: DefNode,
    // The definitions it reads directly (def_deps).
    deps: OnceCell<Deps>,
    // Its right-hand side's value, computed on first demand (def_value).
    value: OnceCell<DefValue>,
    // The scope before the binder: where the right-hand side's names resolve.
    ctx: Ctx,
    // The elaborated value(s): what the run time evaluates for this binder.
    values: Vec<ExprRef>,
    // Memo of CheckDef::fallible.
    fallible: OnceCell<bool>,
    // Memo: this definition or one it reaches is fallible (trans_fallible).
    trans_fallible: OnceCell<bool>,
    // The index variables written in this definition and the ones it reaches
    // (def_index_names).
    index_names: OnceCell<Rc<BTreeSet<String>>>,
    // Settling erasure: a finished check-time evaluation's closure holds this
    // definition (`reached`), a clean one does (`reached_clean`), a kept
    // annotation's closure does (`kept`). Set on what the evaluation read,
    // carried along the reads by settle_definitions.
    reached: Cell<bool>,
    reached_clean: Cell<bool>,
    kept: Cell<bool>,
    // An annotation's evaluation finished with this definition, and its
    // binding cannot fail at run time (settle_definitions).
    proven: Cell<bool>,
    // Something named the binder at value position (lookup), or a kept
    // annotation reads it.
    used: Cell<bool>,
}

// What an expression reads of the definitions: one name it uses, the definition
// that name denotes (an index into InferCtx::defs), and which of that
// definition's binders it is (0 for a `let`).
#[derive(Clone)]
struct Read {
    name: String,
    def: usize,
    member: usize,
}

// What a definition's right-hand side evaluated to at check time, once
// (def_value): its value, or the failure of the earliest definition (the
// smallest index, `origin`) it reaches that failed. A group's value is the
// tuple of its binders' values.
enum DefValue {
    Done(Value),
    Failed { origin: usize, why: Failure },
}

// Why a check-time evaluation did not produce a value (machine::Pure without
// its value).
#[derive(Clone)]
enum Failure {
    Stuck(Stuck),
    Error(String),
}

impl Failure {
    fn into_pure(self) -> Pure {
        match self {
            Failure::Stuck(why) => Pure::Stuck(why),
            Failure::Error(msg) => Pure::Error(msg),
        }
    }
}

// The definitions one definition reads directly, and whether it is stuck: it
// names a value that exists only at run time (a parameter, a pattern variable),
// or reads a definition that is stuck (spec section 2's stuck rule), so only the
// run time can evaluate it. Indices, never `Rc`s: a chain of `Rc`s would drop
// recursively.
struct Deps {
    reads: Vec<Read>,
    stuck: bool,
}

enum DefNode {
    Let(ExprRef),
    // A `let rec` group: the original bindings.
    Group(Rc<Vec<(String, Option<ExprRef>, ExprRef)>>),
}

impl CheckDef {
    // A binding the run time does not need: proven, and no value-position
    // reference reads it.
    fn erasable(&self) -> bool {
        self.proven.get() && !self.used.get()
    }

    // Evaluating the binding only builds closures or type literals, so it
    // cannot fail: a check inside a function body runs only when something
    // calls it, and a caller looks the binder up (`used`).
    fn inert(&self, arena: &Arena) -> bool {
        self.values.iter().all(|v| matches!(arena[settled(arena, *v)], Expr::Lambda(..) | Expr::TypeLit(_)))
    }

    // The parsed trees the definition is written in: a right-hand side, and for
    // a group the binders' annotations too.
    fn sources(&self) -> Vec<ExprRef> {
        match &self.node {
            DefNode::Let(rhs) => vec![*rhs],
            DefNode::Group(bindings) => bindings.iter().flat_map(|(_, ann, v)| ann.iter().chain([v]).copied()).collect(),
        }
    }

    // What its right-hand side reads, in the scope the binder was checked in.
    // A group of direct `fun`s and a type group bind their own names for their
    // values (closures, placeholders); any other group's values see the outer
    // scope (machine.rs's LetRec fallback).
    fn direct_reads(&self, arena: &Arena, infer: &InferCtx) -> (Vec<Read>, bool) {
        match &self.node {
            DefNode::Let(rhs) => reads_in(arena, &[*rhs], &[], &self.ctx, infer),
            DefNode::Group(bindings) => {
                let names: Vec<String> = if group_kind(arena, bindings) != GroupKind::Plain {
                    bindings.iter().map(|(n, _, _)| n.clone()).collect()
                } else {
                    Vec::new()
                };
                let values: Vec<ExprRef> = bindings.iter().map(|(_, _, v)| *v).collect();
                reads_in(arena, &values, &names, &self.ctx, infer)
            }
        }
    }

    // Some value holds a run-time check that could fail (fallible_check). The
    // answer cannot change once the values are elaborated: a later cast slot
    // lies in later code.
    fn fallible(&self, arena: &Arena, spans: &SpanMap, infer: &InferCtx) -> bool {
        *self.fallible.get_or_init(|| self.values.iter().any(|v| fallible_check(arena, *v, spans, infer)))
    }
}

// `n` without the erased annotation sites settle_erasure has yet to remove.
fn settled(arena: &Arena, mut n: ExprRef) -> ExprRef {
    while let Expr::Let(_, None, annot, inner) = &arena[n]
        && let Expr::Annot(_, spec) = &arena[*annot]
        && spec.erased.get()
    {
        n = *inner;
    }
    n
}

// Can a run-time check in the elaborated tree at `root` fail? A kept
// annotation site can (an erased one is about to be removed), and so can an
// Unknown site's check and a contract (its checks run at later calls). A cast
// slot resolve_pendings has yet to fill can. So can a boundary that needs
// nothing beyond its native test (`Then::Done`), unless its operand is
// closed (it names nothing a program binder binds) and the check passes when
// eval_pure runs it now: evaluation is deterministic, so it passes at run time
// too. That decides the shape guard on a prelude constructor (`List` in
// `List(Int)`) and leaves every check on a parameter or a `let` fallible.
// `bound_names` comes from the parsed tree, so a binder elaboration adds
// (`__ca`, `__cf`) looks closed; evaluated alone it is unbound,
// so such a check counts as fallible, which is the safe answer.
// Each check is decided on its own: one inside a closed operand (in a function
// body, say) did not necessarily run when the outer one passed.
fn fallible_check(arena: &Arena, root: ExprRef, spans: &SpanMap, infer: &InferCtx) -> bool {
    let slots: HashSet<ExprRef> = infer.pendings.borrow().iter().map(|p| p.slot).collect();
    let mut closed = Vec::new();
    let mut work = vec![root];
    let mut kids = Vec::new();
    while let Some(n) = work.pop() {
        if slots.contains(&n) {
            return true;
        }
        match &arena[n] {
            Expr::Annot(_, spec) if spec.erased.get() => continue,
            Expr::Annot(..) => return true,
            _ => {}
        }
        if let Expr::Conform(operand, spec) = &arena[n] {
            // Only a static boundary with nothing beyond its native test can be decided now.
            let TypeSource::Static(lazy) = &spec.source else { return true };
            if !matches!(lazy.lowered.get(), Some(Some(lowered)) if matches!(lowered.then, Then::Done)) {
                return true;
            }
            if !spec.lens.is_empty() || crate::resolve::free_vars(arena, *operand).iter().any(|name| infer.bound_names.contains(name)) {
                return true;
            }
            closed.push(n);
        }
        kids.clear();
        arena[n].children(&mut kids);
        work.extend(&kids);
    }
    closed
        .into_iter()
        .any(|n| !matches!(crate::machine::eval_pure(arena, n, spans, crate::machine::ANNOTATION_FUEL), crate::machine::Pure::Done(_)))
}

// pub(crate), not private, only because this project's own test
// convention (see unify_index_expr's neighbors) keeps tests in
// src/lib.rs -- a different module -- rather than because anything
// outside typecheck.rs is meant to name this directly.
// The real, threaded state a Hindley-Milner-style unifier needs, carried
// through every elaborate()/elaborate_node() call as `&mut InferCtx` --
// see the design spec's own "Data model" section for the full rationale
// (functional substitution over union-find). Generalization does NOT
// live here -- no level counter, no birth-level tracking -- it reads the
// per-scope open-variable set Ctx carries (THE RULE at Scheme); this
// struct is just a substitution.
//
// `unify` is now called unconditionally from Expr::App/BinOp::Cons/If/
// Match/ListLit (Tasks 4-6), so `subst` and every method below are live
// on every elaboration, not just exercised by this file's own tests.
pub(crate) struct InferCtx {
    // Grows monotonically as unify() binds variables; never shrinks --
    // no backtracking, matching this checker's existing single-pass
    // character everywhere else.
    subst: HashTrieMap<String, Type>,
    // Grows monotonically exactly like `subst`, for the same reason --
    // no backtracking anywhere in this checker. A SEPARATE map from
    // `subst`: an index variable (IndexExpr::Var) and a type variable
    // (Type::Var) are different namespaces that happen to both be
    // plain Strings -- conflating them would let an ordinary generic
    // function's own type-var name collide with an unrelated Vec's
    // own index-var name.
    pub(crate) index_subst: HashTrieMap<String, IndexExpr>,
    // Index variables that unify_index_expr must NOT bind: the free index
    // variables of an annotated binding's signature (the fresh ones its
    // binder introduced, not the user's spelling) while its value is
    // checked (spec 2026-10-07). Hypotheses injected by Match write
    // index_subst directly, so they are unaffected.
    pub(crate) rigid_index: HashSet<String>,
    // Runtime values for index variables, innermost last: while a Lambda
    // whose parameter is Vec(n) is elaborated, `n` is that parameter's
    // length. A Dyn-to-Vec(n) boundary check inside reads the length of a
    // hidden alias (`hidden = param`), which the Lambda inserts at its entry
    // only if some check used it. See index_len_term. Pushed and popped
    // by elaborate_mode's Lambda frames; an elaboration error abandons the
    // whole check, so an Err path never needs to pop.
    index_witness: Vec<IndexWitness>,
    // Dyn-to-Vec(n) crossings whose length check needs an index variable
    // that nothing at runtime holds at the crossing (no witness, no static
    // binding yet): a later annotation may still bind it. See Obligation.
    // A RefCell because the check builders only see a shared &InferCtx
    // (CheckEnv), the same reason IndexWitness.used is a Cell.
    obligations: RefCell<Vec<Obligation>>,
    // Casts deferred to the end of inference (Pending).
    pendings: RefCell<Vec<Pending>>,
    // Set by Handle just before it elaborates a syntactic handler: the row the clause'"'"'s `resume`
    // has (the body'"'"'s row without the handled effect). The MakeHandler arm takes it; None (a
    // handler met anywhere else) types `resume` as Dyn and discards the clause'"'"'s row.
    resume_row: Option<EffectRow>,
    // How many function bodies (Lambda, handler clause) enclose the point
    // being elaborated: code inside one can run any number of times.
    repeatable_depth: usize,
    // Names of type variables whose value reaches a Dyn sink (the argument of
    // a Dyn callee, a Perform payload) while the variable is still unbound
    // (spec 2026-10-08-var-passthrough-casts). The Fun-callee App arm casts a
    // typed function argument to Dyn when the callee's parameter is such a
    // variable. Grows like `subst` and is snapshotted/restored with it by
    // unify_trial (a failed trial must not leak flags); propagated by
    // `lookup` and `unify`.
    dyn_sunk: HashTrieSet<String>,
    // Every name some binder in the arena binds (check). A
    // name in an annotation that no binder has can only be a prelude name or
    // unbound, so annotation_type skips the scope walk for it: without this,
    // `Int` in each of n annotations would walk a scope n deep.
    bound_names: HashSet<String>,
    // Every definition, by CheckDef::idx.
    defs: Vec<Rc<CheckDef>>,
    // Where the variables of every check-time evaluation live. One table for
    // them all: a closure one evaluation builds is called by a later one. Each
    // evaluation resolves a copy of its root (eval_pure_in), so a node is never
    // resolved under two layouts.
    resolved: Resolved,
    // The definition whose value each parsed node is (a let's right-hand side, a
    // group member's value), and which member; None when two definitions claim
    // one node, which is never reused. Elaborated once, so two claims only come
    // from a node elaborated twice.
    value_of: FxMap<ExprRef, Option<(usize, usize)>>,
    // An annotation is being elaborated (annotation_type): a name it reads
    // is not a value-position read, so `lookup` leaves CheckDef::used alone.
    in_annotation: bool,
    // The value of a recursive `let rec` group about to be elaborated: its
    // `fun` must stay a direct Lambda (resolve::GroupKind::Functions), so a kept
    // parameter annotation goes inside it (the Fun fold).
    direct_value: Option<ExprRef>,
    // The finished check-time evaluations and the nodes erasure may remove,
    // in the order they were made: settle_erasure decides them all once
    // inference has finished.
    proofs: Vec<Proof>,
    splices: Vec<Splice>,
}

impl InferCtx {
    pub(crate) fn new() -> InferCtx {
        InferCtx {
            subst: HashTrieMap::new(),
            index_subst: HashTrieMap::new(),
            rigid_index: HashSet::new(),
            index_witness: Vec::new(),
            obligations: RefCell::new(Vec::new()),
            pendings: RefCell::new(Vec::new()),
            resume_row: None,
            repeatable_depth: 0,
            dyn_sunk: HashTrieSet::new(),
            bound_names: HashSet::new(),
            defs: Vec::new(),
            resolved: Resolved::default(),
            value_of: FxMap::default(),
            in_annotation: false,
            direct_value: None,
            proofs: Vec::new(),
            splices: Vec::new(),
        }
    }

    fn new_def(&mut self, node: DefNode, values: Vec<ExprRef>, ctx: &Ctx) -> Rc<CheckDef> {
        let def = Rc::new(CheckDef {
            idx: self.defs.len(),
            node,
            deps: OnceCell::new(),
            value: OnceCell::new(),
            ctx: ctx.clone(),
            values,
            fallible: OnceCell::new(),
            trans_fallible: OnceCell::new(),
            index_names: OnceCell::new(),
            reached: Cell::new(false),
            reached_clean: Cell::new(false),
            kept: Cell::new(false),
            proven: Cell::new(false),
            used: Cell::new(false),
        });
        self.defs.push(def.clone());
        def
    }

    // Records that parsed node `node` is member `member` of definition `def`'s
    // value; a second claim on the node withdraws it (never reused). Defensive:
    // no program is known to elaborate one parsed value node twice.
    fn note_value_of(&mut self, node: ExprRef, def: usize, member: usize) {
        self.value_of.entry(node).and_modify(|e| *e = None).or_insert(Some((def, member)));
    }

    // Flags `ty` as reaching a Dyn sink if it is (still) a bare type variable.
    fn note_dyn_sink(&mut self, ty: &Type) {
        if let Type::Var(v) = self.resolve(ty) {
            self.dyn_sunk.insert_mut(v);
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
        // Only a bound variable or a bound index can change anything.
        let relevant = (if self.subst.is_empty() { 0 } else { Summary::VAR }) | (if self.index_subst.is_empty() { 0 } else { Summary::INDEXED });
        ResolveDeep { infer: self, relevant, memo: NodeMemo::default() }.walk(ty).unwrap_or_else(|| ty.clone())
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

// `InferCtx::resolve_deep` as a walk: a bound variable is followed to its solution and an
// index through `resolve_index_deep`; everything else is rebuilt only where something changed.
struct ResolveDeep<'a> {
    infer: &'a InferCtx,
    relevant: u8,
    memo: NodeMemo<Option<Type>>,
}

impl Rebuild for ResolveDeep<'_> {
    fn skip(&self, s: Summary) -> bool {
        !s.any(self.relevant)
    }
    fn memo(&mut self) -> &mut NodeMemo<Option<Type>> {
        &mut self.memo
    }
    fn special(&mut self, t: &Type) -> Option<Option<Type>> {
        match t {
            Type::Var(_) => {
                let solved = self.infer.resolve(t);
                if solved == *t {
                    return Some(None);
                }
                Some(Some(self.walk(&solved).unwrap_or(solved)))
            }
            Type::Indexed(wrapped, index) => {
                let (wrapped2, index2) = (self.walk(wrapped), self.infer.resolve_index_deep(index));
                if wrapped2.is_none() && index2 == **index {
                    return Some(None);
                }
                Some(Some(Type::Indexed(wrapped2.map_or_else(|| wrapped.clone(), Hc::new), Rc::new(index2))))
            }
            _ => None,
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
    // A node with no variable in it has nothing to find; a node already walked (this call, the
    // substitution constant meanwhile) found nothing, else the walk would have stopped there.
    fn go(name: &str, ty: &Type, infer: &InferCtx, seen: &mut Visited) -> bool {
        if !ty.summary().any(Summary::VAR) || !seen.first(ty) {
            return false;
        }
        match infer.resolve(ty) {
            Type::Var(n) => n == name,
            Type::List(elem) => go(name, &elem, infer, seen),
            Type::Fun(param, _row, ret) => go(name, &param, infer, seen) || go(name, &ret, infer, seen),
            Type::Tuple(items) | Type::Union(items) => items.iter().any(|t| go(name, t, infer, seen)),
            Type::Record(fields) => fields.iter().any(|(_, t)| go(name, t, infer, seen)),
            Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) | Type::RecVar(_) => false,
            Type::Rec(g, _) => g.0.bodies.iter().any(|b| go(name, b, infer, seen)),
            // Wraps one nested Type (the index expression is never itself a
            // Type::Var) -- same recurse-into-the-wrapped-type precedent as
            // List's own arm just above.
            Type::Indexed(wrapped, _) => go(name, &wrapped, infer, seen),
        }
    }
    go(name, ty, infer, &mut Visited::new())
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
pub(crate) fn unify_index_expr(a: &IndexExpr, b: &IndexExpr, infer: &mut InferCtx) -> Rel {
    // resolve_index_deep, not resolve_index: a bound variable buried in a compound must be seen.
    let a = infer.resolve_index_deep(a);
    let b = infer.resolve_index_deep(b);
    let refuted = |why: String| Rel::Refuted(Why::Index { msg: why, bare: false });
    match (&a, &b) {
        _ if a == b => Rel::Proven,
        // A flexible variable yields to a rigid one: bind the flexible side, never the rigid one.
        (IndexExpr::Var(rigid), IndexExpr::Var(flex)) if infer.rigid_index.contains(rigid) && !infer.rigid_index.contains(flex) => {
            infer.index_subst.insert_mut(flex.clone(), IndexExpr::Var(rigid.clone()));
            Rel::Proven
        }
        (IndexExpr::Var(name), other) | (other, IndexExpr::Var(name)) => {
            // A rigid variable may only be equated, never bound.
            if infer.rigid_index.contains(name) {
                return match index_exprs_compare(&a, &b) {
                    Some(IndexCmp::Equal) => Rel::Proven,
                    None => undecided(),
                    Some(_) => refuted(format!("index variable {} is fixed by the signature and cannot equal {other}", IndexExpr::Var(name.clone()))),
                };
            }
            if occurs_in_index(name, other, infer) {
                // `n` against `n + 0` is SOP-equal; only a GENUINE occurs violation errors.
                return match index_exprs_compare(&a, &b) {
                    Some(IndexCmp::Equal) => Rel::Proven,
                    None => undecided(),
                    Some(_) => refuted(format!("infinite index expression: {} occurs in {other}", IndexExpr::Var(name.clone()))),
                };
            }
            infer.index_subst.insert_mut(name.clone(), other.clone());
            Rel::Proven
        }
        _ => {
            // SOP-equality is checked BEFORE the shape-recursion arms below: `m*n` vs `n*m` is
            // equal without being pointwise equal, and recursing first would bind `m := n`.
            let cmp = index_exprs_compare(&a, &b);
            if cmp == Some(IndexCmp::Equal) {
                return Rel::Proven;
            }
            let rel = match (&a, &b) {
                (IndexExpr::Add(a1, a2), IndexExpr::Add(b1, b2))
                | (IndexExpr::Sub(a1, a2), IndexExpr::Sub(b1, b2))
                | (IndexExpr::Mul(a1, a2), IndexExpr::Mul(b1, b2)) => match unify_index_expr(a1, b1, infer) {
                    Rel::Proven => unify_index_expr(a2, b2, infer),
                    other => other,
                },
                _ => refuted(format!("type mismatch: index {a} does not unify with index {b}")),
            };
            // Past the SOP cap the positional recursion may still prove the pair equal (it binds
            // nothing unsoundly), but a failure of it proves nothing.
            if cmp.is_none() && rel != Rel::Proven { undecided() } else { rel }
        }
    }
}

/// The checker could neither prove nor refute the equation.
fn undecided() -> Rel {
    Rel::Unknown(Residual::INDEX_EQ, Cause::Incomplete)
}

/// `unify_index_expr` as the checker's `Result`: a refutation is its message, and an Unknown is a
/// static error too (no run-time test decides an index equation; Pending defers only casts).
pub(crate) fn unify_index(a: &IndexExpr, b: &IndexExpr, infer: &mut InferCtx, span: Span) -> Result<(), TypeError> {
    match unify_index_expr(a, b, infer) {
        Rel::Proven => Ok(()),
        Rel::Refuted(why) => Err(TypeError(why.message().into_owned(), span)),
        // Unknown arises only when the SOP normalizer hits its cap.
        Rel::Unknown(..) => Err(TypeError("could not decide (expression too large) whether the two index expressions are equal".to_string(), span)),
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
fn unify(t1: &Type, t2: &Type, infer: &mut InferCtx, span: Span) -> Result<(), Mismatch> {
    unify_in(t1, t2, infer, span, &mut FxSet::default())
}

// `done` holds the big pairs this call has already unified (after resolution). Within one call
// the substitution only grows, so a pair unified once stays unified; a failure ends the call,
// so no failed pair is ever recorded. Each call (and each unify_trial) has its own set.
fn unify_in(t1: &Type, t2: &Type, infer: &mut InferCtx, span: Span, done: &mut FxSet<(ById, ById)>) -> Result<(), Mismatch> {
    let t1 = infer.resolve(t1);
    let t2 = infer.resolve(t2);
    let key = (is_big(&t1) || is_big(&t2)).then(|| (ById(t1.clone()), ById(t2.clone())));
    if key.as_ref().is_some_and(|k| done.contains(k)) {
        return Ok(());
    }
    unify_resolved(&t1, &t2, infer, span, done)?;
    done.extend(key);
    Ok(())
}

fn unify_resolved(t1: &Type, t2: &Type, infer: &mut InferCtx, span: Span, done: &mut FxSet<(ById, ById)>) -> Result<(), Mismatch> {
    let mismatch = || Mismatch::Expected { expected: t2.clone(), found: t1.clone(), span };
    match (t1, t2) {
        (Type::Var(n1), Type::Var(n2)) if n1 == n2 => Ok(()),
        (Type::Var(name), other) | (other, Type::Var(name)) => {
            if occurs_in(name, other, infer) {
                return Err(Mismatch::Infinite { name: name.clone(), ty: other.clone(), span });
            }
            // Two variables being equated share one flag (a hop's instance
            // aliasing the caller's variable carries the sink back to it).
            if let Type::Var(o) = other
                && (infer.dyn_sunk.contains(name) || infer.dyn_sunk.contains(o))
            {
                infer.dyn_sunk.insert_mut(name.clone());
                infer.dyn_sunk.insert_mut(o.clone());
            }
            infer.subst.insert_mut(name.clone(), other.clone());
            Ok(())
        }
        (Type::Dyn, _) | (_, Type::Dyn) => Ok(()),
        (Type::Int, Type::Int) | (Type::Float, Type::Float) | (Type::Bool, Type::Bool) | (Type::Str, Type::Str) => Ok(()),
        (Type::Token(a), Type::Token(b)) if a == b => Ok(()),
        // Equal recursive types (structurally, E1): never unfolds, never binds a
        // Type::Var as a side effect of comparing them.
        (Type::Rec(a, i), Type::Rec(b, j)) if i == j && a == b => Ok(()),
        (Type::List(a), Type::List(b)) => unify_in(a, b, infer, span, done),
        // The wrapped type recurses through unify()'s own Var-binding
        // behavior first (matching Type::Fun's own param-then-return
        // order just below); the index halves then go through
        // unify_index_expr -- a real, but deliberately narrow unifier
        // (see its own doc comment) that binds a bare unbound
        // IndexExpr::Var to the other side, rather than the plain
        // SOP-equality check this arm used before Phase 2 Task 3 wired
        // real index-variable inference in.
        (Type::Indexed(wa, ia), Type::Indexed(wb, ib)) => {
            unify_in(wa, wb, infer, span, done)?;
            Ok(unify_index(ia, ib, infer, span)?)
        }
        (Type::Fun(p1, _r1, ret1), Type::Fun(p2, _r2, ret2)) => {
            unify_in(p1, p2, infer, span, done)?;
            unify_in(ret1, ret2, infer, span, done)
            // Row unification is out of scope here -- effect rows already
            // have their own, separate, unrelated generalize/instantiate
            // mechanism (bind_row_vars/resolve_row), untouched by this
            // plan; two Fun types unify on their param/return shape only.
        }
        (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => {
            a.iter().zip(b.iter()).try_for_each(|(x, y)| unify_in(x, y, infer, span, done))
        }
        (Type::Record(a), Type::Record(b)) if a.len() == b.len() => {
            for (name, ty_a) in a.iter() {
                match find_field(b, name) {
                    Some(ty_b) => unify_in(ty_a, ty_b, infer, span, done)?,
                    None => return Err(mismatch()),
                }
            }
            Ok(())
        }
        _ => Err(mismatch()),
    }
}

// The unification-flavored counterpart to relate(): same directional
// walk, but since this IS a unification (not just a static check), it
// also resolves and BINDS any free Type::Var it finds along the way --
// exactly like unify() does, via the same delegation to unify() itself
// for every case that isn't specifically Fun or Record. That delegation
// first tries unify_trial (binds Type::Vars on success, leaves no trace
// on failure); if that trial fails, the pair is still accepted unless
// relate() refutes it -- unify_fits must never be stricter than
// coerce_check, since it runs right after coerce() as a pure add-on
// (binding Vars), not a second gate. See the catch-all below.
fn unify_fits(required: &Type, actual: &Type, infer: &mut InferCtx, span: Span) -> Result<(), Mismatch> {
    fits_in(required, actual, infer, span, &mut FxSet::default())
}

// `done` holds the big pairs this call has already accepted, relate rescue included. That is
// safe under the contract above (never stricter than coerce_check): a remembered acceptance
// cannot add strictness, and every caller propagates an error with `?`, so a failure ends the
// call.
fn fits_in(required: &Type, actual: &Type, infer: &mut InferCtx, span: Span, done: &mut FxSet<(ById, ById)>) -> Result<(), Mismatch> {
    let required = infer.resolve(required);
    let actual = infer.resolve(actual);
    let key = (is_big(&required) || is_big(&actual)).then(|| (ById(required.clone()), ById(actual.clone())));
    if key.as_ref().is_some_and(|k| done.contains(k)) {
        return Ok(());
    }
    fits_resolved(&required, &actual, infer, span, done)?;
    done.extend(key);
    Ok(())
}

fn fits_resolved(required: &Type, actual: &Type, infer: &mut InferCtx, span: Span, done: &mut FxSet<(ById, ById)>) -> Result<(), Mismatch> {
    let mismatch = || Mismatch::Expected { expected: required.clone(), found: actual.clone(), span };
    match (required, actual) {
        (Type::Fun(req_param, req_row, req_ret), Type::Fun(act_param, act_row, act_ret)) => {
            fits_in(act_param, req_param, infer, span, done)?;
            fits_in(req_ret, act_ret, infer, span, done)?;
            if !row_consistent(req_row, act_row) {
                return Err(mismatch());
            }
            Ok(())
        }
        (Type::Record(req_fields), Type::Record(act_fields)) => {
            for (name, req_ty) in req_fields.iter() {
                match find_field(act_fields, name) {
                    Some(act_ty) => fits_in(req_ty, act_ty, infer, span, done)?,
                    None => return Err(mismatch()),
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
        // incompatible Vec(2)) is wrongly rescued by the relate-based
        // rescue's permissiveness one level down. Recursing through unify_fits
        // itself (not unify_trial/the bare rescue) re-fires that exclusion
        // at the nesting depth where the Indexed pair actually appears --
        // same fix shape the Record arm above already gets right for
        // `{v: Vec(n)}`. Tuple's length check mirrors unify()'s own Tuple
        // arm just below in this file: a length mismatch is a plain type
        // mismatch, not something to recurse into element-wise.
        (Type::Tuple(req_items), Type::Tuple(act_items)) if req_items.len() == act_items.len() => {
            req_items.iter().zip(act_items.iter()).try_for_each(|(req_ty, act_ty)| fits_in(req_ty, act_ty, infer, span, done))
        }
        (Type::List(req_elem), Type::List(act_elem)) => fits_in(req_elem, act_elem, infer, span, done),
        // Type::Indexed gets no explicit arm here, deliberately -- this
        // match already has none for Union either, so anything that isn't
        // Fun/Record/Tuple/List falls through to the generic catch-all
        // below: unify_trial tries unify() first (which DOES have its own
        // Indexed arm), with a relate() rescue on failure. That
        // path gives Indexed the right answer on match, index mismatch,
        // AND binding a Type::Var reachable only through the wrapped
        // type -- see the design spec's own §3 for the rationale -- so
        // adding a redundant explicit arm here would only duplicate what
        // already works.
        // ponytail: no Type::Var binding via the relate() rescue path
        // below -- unify_trial is tried first and binds Vars on success, but a Var
        // reachable ONLY through relate() not refuting the pair (e.g. only
        // resolvable through one alternative of a Union, or only through the
        // List(Dyn)/Tuple bridge, or only through an Indexed value's own
        // forget-the-index widening) stays unbound (reads as Dyn via the unresolved
        // Var) -- same permissive-fallback ceiling this feature area already
        // documents elsewhere (e.g. If/Match/ListLit branch combination), not a
        // regression, just not newly closed by this fix. Upgrade if real programs
        // need tighter constraint binding through these rescue-only paths.
        _ => match unify_trial(required, actual, infer, span) {
            Ok(()) => Ok(()),
            Err(e) => {
                // An Indexed-vs-Indexed pair already has full, authoritative
                // handling via unify()'s own dedicated Indexed arm (see this
                // match's own doc comment above) -- but relate treats a bare
                // flexible index variable on either side as Unknown rather
                // than Refuted (so coerce() itself can no-op on a Vec(n)
                // position and a real binding can happen afterward via
                // unify_fits), so the rescue below would accept this pair
                // too. Left unguarded, it would
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
                if matches!((required, actual), (Type::Indexed(..), Type::Indexed(..))) {
                    return Err(e);
                }
                // Never stricter than coerce_check, which rejects exactly the Refuted and Undecided pairs.
                match relate(actual, required, infer) {
                    Rel::Refuted(_) => Err(e),
                    Rel::Unknown(res, _) if res.undecided() => Err(Mismatch::CannotTell { from: actual.clone(), to: required.clone(), span }),
                    _ => Ok(()),
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
pub(crate) fn unify_trial(t1: &Type, t2: &Type, infer: &mut InferCtx, span: Span) -> Result<(), Mismatch> {
    // The three maps are persistent (rpds), so each snapshot is an O(1) share, not a copy.
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

// `Type::Indexed`'s own "forget the index" widening (the widening
// relate applies to an Indexed source), applied locally wherever an operator cares only
// about an operand's wrapped shape, not its statically-tracked length
// (BinOp::Cons/Concat) -- relate isn't an option at
// these two call sites since they run through unify()/consistent()
// directly, not relate. One level only: an Indexed value's wrapped type
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
    let scheme = ctx.get(name);
    if let Some(def) = scheme.as_ref().and_then(|s| s.def.as_ref())
        && !infer.in_annotation
    {
        def.used.set(true);
    }
    match scheme {
        None => Type::Dyn,
        Some(scheme) if scheme.quantified.is_empty() => scheme.ty.clone(),
        Some(scheme) => {
            let names = |sort: fn(&TyVar) -> Option<&String>| -> Vec<String> { scheme.quantified.iter().filter_map(sort).cloned().collect() };
            let row_vars = names(|v| if let TyVar::Row(n) = v { Some(n) } else { None });
            let type_vars = names(|v| if let TyVar::Type(n) = v { Some(n) } else { None });
            let index_vars = names(|v| if let TyVar::Index(n) = v { Some(n) } else { None });
            let row_subst: HashMap<String, EffectRow> =
                row_vars.iter().map(|v| (v.clone(), EffectRow::Var(fresh_row_name(v)))).collect();
            let type_subst: HashMap<String, Type> =
                type_vars.iter().map(|v| (v.clone(), infer.fresh_var(v))).collect();
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
                index_vars.iter().map(|v| (v.clone(), IndexExpr::Var(fresh_index_name(v)))).collect();
            subst_type(&scheme.ty, &row_subst, &type_subst, &index_subst)
        }
    }
}

// Ordinary (non-generalized) binding -- lambda parameters, and anything
// else that isn't a `let`. Variables in `ty`, if any, are shared verbatim by
// every use within this one scope, not instantiated fresh per use (that's what
// `extend_generalized` is for). Annotation row variables are already fresh
// `e#k` by the time they get here (open_annotation).
//
// `pub(crate)`, not private: Task 3's own Case B tests need to bind a
// scrutinee variable directly to a hand-built `Type::Indexed(Type::
// Rec(..), _)` -- a shape no surface syntax can express (`Vec(n)`'s
// own parser sugar is hard-coded to wrap `Type::List(Dyn)` only) -- and
// drive `check_against` on it directly, mirroring this file's own
// existing `extend_generalized_and_lookup_mint_a_fresh_index_variable...`
// test's identical precedent for exercising typecheck's own internals
// below the parser's reach. Same progressive, only-as-far-as-needed
// visibility widening this project already uses repeatedly.
pub(crate) fn extend(ctx: &Ctx, name: &str, ty: Type) -> Ctx {
    ctx.bind(name, Scheme::mono(ty))
}

// `let`-binding: generalizes over row, type and index variables alike,
// by THE RULE stated at Scheme -- the variables of `ty` that no enclosing
// scope has open (`generalizable`).
//
// Why the exclusion matters, e.g. for row variables (confirmed reachable:
// `let f = fun cb: (Dyn ->{e} Dyn) -> let g = cb in g in f(fun y -> perform
// choose(y))(0)` used to typecheck clean and only panic at runtime -- "e"
// leaking from cb's still-open annotation into g's generalized scheme, then
// getting freshly (and wrongly) renamed on every `g` reference): a
// parameter's annotation stays open for that parameter's entire body, and a
// nested `let` that merely mentions the same name (through an alias, a
// tuple, a record) must not re-generalize it, since the name belongs to the
// enclosing, not-yet-closed binding. A Type::Var or index variable minted
// by unification is the same.
//
// Resolve deeply before storing, not just for the free-variable scan:
// unify() can bind one Type::Var to ANOTHER Type::Var (e.g. Cons's own
// unify(&l_ty, &elem_ty, ...) when both sides are still bare vars),
// leaving the stored type's own param positions naming the OLD alias
// rather than the canonical variable `generalizable` actually
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
// pub(crate): the production binder (let, let rec, lambda) and, for the same test-access reason as Ctx/unify_index_expr
// above -- extend_generalized/lookup together are what this task's
// own unit test exercises directly, bypassing the separate Expr::App
// coerce() gap a source-level `identity_vec(3)(va)` call would hit.
// `def` is the binder's definition, which an annotation naming it evaluates.
pub(crate) fn extend_generalized(ctx: &Ctx, name: &str, ty: Type, infer: &InferCtx, def: Option<Rc<CheckDef>>) -> Ctx {
    let ty = infer.resolve_deep(&ty);
    let quantified = generalizable(ctx, &ty, infer);
    ctx.bind(name, Scheme { quantified, ty, def })
}

// The variables of `ty` (already resolve_deep'ed) that this `let` may
// quantify: those no enclosing scope has open. An index variable a pending
// obligation still needs is decided by whoever binds it later, inside the
// function containing the crossing; generalizing it would give each use a
// fresh copy and leave the obligation unresolvable (spec
// 2026-10-07-dyn-crossing-deferred-length-checks, polymorphism rule).
fn generalizable(ctx: &Ctx, ty: &Type, infer: &InferCtx) -> Vec<TyVar> {
    let candidates = vars_in(ty);
    if candidates.is_empty() {
        return Vec::new();
    }
    let mut open = ctx.open_vars(infer);
    if candidates.iter().any(|v| matches!(v, TyVar::Index(_))) {
        open.extend(pending_obligation_vars(infer).into_iter().map(TyVar::Index));
    }
    candidates.difference(&open).cloned().collect()
}

// Every index-variable name free anywhere inside `e` -- IndexExpr has
// no binding forms of its own (see its own grammar), so every Var is
// free by construction; this is a plain collection, not an occurs- or
// scope-sensitive walk.
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

// Recurses into every Type variant vars_in also recurses into
// (Fun/List/Tuple/Union/Record), so a row variable this function finds
// reachable is also one this function can rename -- the two are meant to
// stay in lockstep, since a variable extend_generalized captured (via
// vars_in) but subst_type then failed to touch would keep its
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
// anyway, on the same "recurse everywhere vars_in does" principle
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

pub(crate) fn subst_type(ty: &Type, row_subst: &HashMap<String, EffectRow>, type_subst: &HashMap<String, Type>, index_subst: &HashMap<String, IndexExpr>) -> Type {
    // Only a variable some map names can change anything.
    let relevant = (if type_subst.is_empty() { 0 } else { Summary::VAR })
        | (if row_subst.is_empty() { 0 } else { Summary::ROWVAR })
        | (if index_subst.is_empty() { 0 } else { Summary::INDEXED });
    SubstType { row_subst, type_subst, index_subst, relevant, memo: NodeMemo::default() }.walk(ty).unwrap_or_else(|| ty.clone())
}

struct SubstType<'a> {
    row_subst: &'a HashMap<String, EffectRow>,
    type_subst: &'a HashMap<String, Type>,
    index_subst: &'a HashMap<String, IndexExpr>,
    relevant: u8,
    memo: NodeMemo<Option<Type>>,
}

impl Rebuild for SubstType<'_> {
    fn skip(&self, s: Summary) -> bool {
        !s.any(self.relevant)
    }
    fn memo(&mut self) -> &mut NodeMemo<Option<Type>> {
        &mut self.memo
    }
    fn special(&mut self, t: &Type) -> Option<Option<Type>> {
        match t {
            Type::Var(name) => Some(self.type_subst.get(name).cloned()),
            Type::Indexed(wrapped, index) => {
                let (wrapped2, index2) = (self.walk(wrapped), subst_index_expr(index, self.index_subst));
                if wrapped2.is_none() && index2 == **index {
                    return Some(None);
                }
                Some(Some(Type::Indexed(wrapped2.map_or_else(|| wrapped.clone(), Hc::new), Rc::new(index2))))
            }
            Type::Fun(param, row, ret) => {
                let (param2, row2, ret2) = (self.walk(param), resolve_row(row, self.row_subst), self.walk(ret));
                if param2.is_none() && ret2.is_none() && row2 == *row {
                    return Some(None);
                }
                Some(Some(Type::Fun(param2.map_or_else(|| param.clone(), Hc::new), row2, ret2.map_or_else(|| ret.clone(), Hc::new))))
            }
            _ => None,
        }
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
    // Binding is first-wins, so a pair already walked adds nothing.
    fn go(param: &Type, arg: &Type, subst: &mut HashMap<String, EffectRow>, seen: &mut FxSet<(ById, ById)>) {
        if !param.summary().any(Summary::ROWVAR) || (is_big(param) && !seen.insert((ById(param.clone()), ById(arg.clone())))) {
            return;
        }
        if let (Type::Fun(p1, r1, p2), Type::Fun(a1, r2, a2)) = (param, arg) {
            if let EffectRow::Var(name) = r1 {
                if !matches!(r2, EffectRow::Var(_)) {
                    subst.entry(name.clone()).or_insert_with(|| r2.clone());
                }
            }
            go(p1, a1, subst, seen);
            go(p2, a2, subst, seen);
        }
    }
    go(param, arg, subst, &mut FxSet::default());
}

// "must be callable" -- the shape used wherever we need to check a value is
// applicable at all without knowing its exact signature (an unannotated
// Dyn-typed callee, or the innermost check inside a function contract).
fn any_fun() -> Type {
    Type::Fun(Hc::new(Type::Dyn), EffectRow::Dyn, Hc::new(Type::Dyn))
}

// The only place a runtime boundary check gets built: `from` is Dyn (or
// Type::Var, treated identically -- see its own doc comment) and `to` is
// concrete. If both sides are concrete
// and relate() refutes the pair, that's a real static error -- reject before
// running at all. relate() is directional (wider-field Records and
// narrower-parameter Functions fit; see Pattern::Record's own doc comment for
// why a wider record needs no runtime projection), and a proven pair incurs NO
// wrapping at all: zero overhead for fully-annotated code.
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
    let env = env.with_span(span);
    // A variable solved since it was read is judged (and reported) by its solution. An unsolved
    // one keeps the check here: binding it later (even to Dyn) proves nothing about the values
    // that reach this point, so a check is never deferred (only casts are, Pending; deferring
    // checks dropped the Int test of `let y: Int = x` once a later `let d: Dyn = f` bound f's
    // type to Dyn).
    let solved = |t: &Type| if matches!(t, Type::Var(_)) { env.infer.resolve_deep(t) } else { t.clone() };
    let (from, to) = (&solved(from), &solved(to));
    if matches!(to, Type::Var(_)) {
        return Ok(e);
    }
    if matches!(from, Type::Var(_)) {
        return Ok(if *to == Type::Dyn { e } else { build_boundary_check(arena, e, to, &env) });
    }
    match relate(from, to, env.infer) {
        Rel::Proven => Ok(e),
        Rel::Refuted(why) => Err(refuted_error(from, to, &why, span)),
        Rel::Unknown(res, _) if res.undecided() => Err(cannot_tell(from, to, span)),
        // Left to coerce_cast (a function wrapper) or unify_fits (an index equation).
        Rel::Unknown(res, _) if !res.needs_test() => Ok(e),
        Rel::Unknown(..) => {
            if matches!(&arena[e], Expr::ListLit(items) if items.is_empty()) {
                // No element to check, but no tuple of positive arity is empty (relate sees `[a]`).
                // Only a list target is trusted without a test: `[]` is not an alias, union or record
                // value. (Returning `Ok(e)` for every other target was a soundness hole found in round
                // 1: `let s: L = []` with `let rec L = Union(Tuple(Int, L), Bool)` ran unchecked.)
                match strip_indexed(&env.infer.resolve_deep(to)) {
                    Type::Tuple(ts) if !ts.is_empty() => return Err(TypeError(format!("type mismatch: expected {}, found {}", to.brief(), from.brief()), span)),
                    Type::List(_) => return Ok(e),
                    _ => {}
                }
            }
            // A raw Dyn source's lengths are all checked (against witnesses, or as
            // obligations). Any other source's only where the index has a runtime value
            // (erase_open_indexed); the rest are the unifier's.
            let target = if *from == Type::Dyn { to.clone() } else { erase_open_indexed(to, &env) };
            Ok(build_boundary_check(arena, e, &target, &env))
        }
    }
}

// `t` as a cast built now would read it: solved type variables substituted, each index
// through runtime_index (a witnessed variable stays, so a Match hypothesis restored later
// cannot change what it means).
fn snapshot_type(t: &Type, infer: &InferCtx, witness_vars: &[&str]) -> Type {
    Snapshot { infer, witness_vars, memo: NodeMemo::default() }.walk(t).unwrap_or_else(|| t.clone())
}

struct Snapshot<'a> {
    infer: &'a InferCtx,
    witness_vars: &'a [&'a str],
    memo: NodeMemo<Option<Type>>,
}

impl Rebuild for Snapshot<'_> {
    fn skip(&self, s: Summary) -> bool {
        !s.any(Summary::VAR | Summary::INDEXED)
    }
    fn memo(&mut self) -> &mut NodeMemo<Option<Type>> {
        &mut self.memo
    }
    fn special(&mut self, t: &Type) -> Option<Option<Type>> {
        match t {
            Type::Var(_) => {
                let solved = self.infer.resolve(t);
                if matches!(solved, Type::Var(_)) {
                    return Some(None);
                }
                Some(Some(self.walk(&solved).unwrap_or(solved)))
            }
            Type::Indexed(w, i) => {
                let (w2, i2) = (self.walk(w), runtime_index(i, self.infer, self.witness_vars, false));
                if w2.is_none() && i2 == **i {
                    return Some(None);
                }
                Some(Some(Type::Indexed(w2.map_or_else(|| w.clone(), Hc::new), Rc::new(i2))))
            }
            _ => None,
        }
    }
}

// A cast whose types still hold an unsolved type variable (design 3.2, casts only): `slot`
// stands in for the cast and resolve_pendings fills it once inference has finished, so a
// variable solved after the cast site (sink-then-call) still gets its function wrapper.
struct Pending {
    slot: ExprRef,
    e: ExprRef,
    from: Type,
    to: Type,
    witnesses: Vec<(String, String)>,
    repeatable: bool,
    span: Option<Span>,
}

// Every witness in scope is marked used: the final cast may read any of them.
fn defer_cast(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, env: &CheckEnv) -> ExprRef {
    let node = arena[e].clone();
    let slot = arena.push(node);
    let in_scope: Vec<&IndexWitness> = env.infer.index_witness.iter().chain(env.local.iter().copied()).collect();
    for w in &in_scope {
        w.used.set(true);
    }
    let vars: Vec<&str> = in_scope.iter().map(|w| w.var.as_str()).collect();
    let (from, to) = (snapshot_type(from, env.infer, &vars), snapshot_type(to, env.infer, &vars));
    let witnesses = in_scope.iter().map(|w| (w.var.clone(), w.hidden.clone())).collect();
    let repeatable = env.infer.repeatable_depth > 0 || env.repeatable;
    env.infer.pendings.borrow_mut().push(Pending { slot, e, from, to, witnesses, repeatable, span: env.span });
    slot
}

// Builds every pending cast under the final substitution, in record order (an inner cast
// first, so an outer one copies its finished node). Runs before resolve_obligations.
fn resolve_pendings(arena: &mut Arena, infer: &InferCtx) {
    for p in infer.pendings.take() {
        let witnesses: Vec<IndexWitness> =
            p.witnesses.iter().map(|(var, hidden)| IndexWitness { var: var.clone(), hidden: hidden.clone(), used: Cell::new(true) }).collect();
        let env = CheckEnv { infer, local: witnesses.iter().collect(), defer: true, repeatable: p.repeatable, span: p.span };
        let vars: Vec<&str> = p.witnesses.iter().map(|(var, _)| var.as_str()).collect();
        let (from, to) = (snapshot_type(&p.from, infer, &vars), snapshot_type(&p.to, infer, &vars));
        let result = cast_now(arena, p.e, &from, &to, &env);
        let node = arena[result].clone();
        arena[p.slot] = node;
    }
}

// The one message for a non-dischargeable Unknown (Residual::UNDECIDED).
fn cannot_tell(from: &Type, to: &Type, span: Span) -> TypeError {
    TypeError(
        format!("cannot tell statically or at run time whether {} fits {}: an alternative differs only in a function type, a nominal type or a length", from.brief(), to.brief()),
        span,
    )
}

// The message for a Refuted pair: an index pair the old gate passed to the unifier keeps the
// unifier's text; an Indexed target on a non-indexable alias says so; otherwise the raw types.
fn refuted_error(from: &Type, to: &Type, why: &Why, span: Span) -> TypeError {
    if let Why::Index { msg, bare: true } = why {
        return TypeError(msg.clone(), span);
    }
    if let Type::Indexed(wrapped, _) = to {
        let indexable = match &**wrapped {
            Type::List(_) | Type::Var(_) | Type::Dyn => true,
            Type::Rec(g, i) => qualifying_rec_alternatives(g, *i).is_some(),
            _ => false,
        };
        if !indexable {
            return TypeError(format!("{} is not an indexable type", wrapped.brief()), span);
        }
    }
    TypeError(format!("type mismatch: expected {}, found {}", to.brief(), from.brief()), span)
}
// `t` without any Vec(n) index, at every depth.
fn strip_indexed(t: &Type) -> Type {
    StripIndexed { memo: NodeMemo::default() }.walk(t).unwrap_or_else(|| t.clone())
}

struct StripIndexed {
    memo: NodeMemo<Option<Type>>,
}

impl Rebuild for StripIndexed {
    fn skip(&self, s: Summary) -> bool {
        !s.any(Summary::INDEXED)
    }
    fn memo(&mut self) -> &mut NodeMemo<Option<Type>> {
        &mut self.memo
    }
    fn special(&mut self, t: &Type) -> Option<Option<Type>> {
        match t {
            Type::Indexed(w, _) => Some(Some(self.walk(w).unwrap_or_else(|| (**w).clone()))),
            _ => None,
        }
    }
}

// `t` with every Indexed whose index has no runtime value at this boundary
// (index_has_value) reduced to the type it wraps, so build_boundary_check never
// needs a witness it cannot get. A non-bare index (Vec(n+1)) keeps its length
// compare when every variable in it has a value; otherwise the check degrades
// to is_list only (see build_cast).
fn erase_open_indexed(t: &Type, env: &CheckEnv) -> Type {
    EraseOpen { env, memo: NodeMemo::default() }.walk(t).unwrap_or_else(|| t.clone())
}

struct EraseOpen<'a, 'b> {
    env: &'a CheckEnv<'b>,
    memo: NodeMemo<Option<Type>>,
}

impl Rebuild for EraseOpen<'_, '_> {
    fn skip(&self, s: Summary) -> bool {
        !s.any(Summary::INDEXED)
    }
    fn memo(&mut self) -> &mut NodeMemo<Option<Type>> {
        &mut self.memo
    }
    fn special(&mut self, t: &Type) -> Option<Option<Type>> {
        match t {
            Type::Indexed(w, i) => {
                let w2 = self.walk(w);
                if !index_has_value(i, self.env) {
                    return Some(Some(w2.unwrap_or_else(|| (**w).clone())));
                }
                Some(w2.map(|w2| Type::Indexed(Hc::new(w2), i.clone())))
            }
            _ => None,
        }
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
    let has_type_var = |ty: &Type| vars_in(ty).iter().any(|v| matches!(v, TyVar::Type(_)));
    if has_type_var(&from) || has_type_var(&to) {
        return defer_cast(arena, e, &from, &to, env);
    }
    cast_now(arena, e, &from, &to, env)
}

fn cast_now(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, env: &CheckEnv) -> ExprRef {
    let (from, to) = (from.clone(), to.clone());
    // Only a function value relate cannot prove is wrapped. Refuted never reaches here from
    // coerce (coerce_check reported it); joins and Dyn sinks cannot refute a Fun.
    if !matches!(from, Type::Fun(..)) || !matches!(relate_cast(&from, &to, env.infer), Rel::Unknown(..)) {
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

// The (parameter, result) types a cast aims at: those of a Fun target (a Rec one
// unfolded a step, a Union with exactly one Fun alternative aimed at it), Dyn for both otherwise.
fn cast_targets(to: &Type) -> (Type, Type) {
    match to {
        Type::Fun(a2, _, b2) => ((**a2).clone(), (**b2).clone()),
        Type::Rec(g, i) => match unfold(g, *i) {
            Type::Rec(..) => (Type::Dyn, Type::Dyn),
            raw => cast_targets(&raw),
        },
        Type::Union(alts) => {
            let mut funs = alts.iter().filter(|t| matches!(t, Type::Fun(..)));
            match (funs.next(), funs.next()) {
                (Some(f), None) => cast_targets(f),
                _ => (Type::Dyn, Type::Dyn),
            }
        }
        _ => (Type::Dyn, Type::Dyn),
    }
}

// Does every variable of index `i` have a runtime value at this boundary: a
// witness in scope, or a static binding whose own variables do (the order
// index_len_term reads them in)?
fn index_has_value(i: &IndexExpr, env: &CheckEnv) -> bool {
    let vars: Vec<&str> = env.local.iter().copied().chain(env.infer.index_witness.iter()).map(|w| w.var.as_str()).collect();
    free_index_vars(&runtime_index(i, env.infer, &vars, false)).iter().all(|v| vars.contains(&v.as_str()))
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
    let up_body = cast_up(arena, body, b, &b_target, env, true);
    if relate_cast(&a_target, a, env.infer) == Rel::Proven {
        let ann = type_lit(arena, a_target);
        return arena.push(Expr::Lambda(param, Some(ann), up_body));
    }
    // The incoming value arrives under a fresh name and is checked outside the
    // user's parameter scope, so a parameter of the same name cannot capture
    // the value.
    let ca = cast_binder("__ca");
    let ca_ref = arena.push(Expr::Var(ca.clone()));
    let checked = build_boundary_check(arena, ca_ref, &erase_open_indexed(a, env), env);
    let body = arena.push(Expr::Let(param, None, checked, up_body));
    let ann = type_lit(arena, a_target);
    arena.push(Expr::Lambda(ca, Some(ann), body))
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

// UP for a result `res` of type `b` seen as `b_target`: nothing when relate proves it; a function
// result is cast in turn; any other result is checked against what the target promises (h03).
// Runs once per call: repeatable, records no obligations.
fn cast_up(arena: &mut Arena, res: ExprRef, b: &Type, b_target: &Type, env: &CheckEnv, literal: bool) -> ExprRef {
    if relate_cast(b, b_target, env.infer) == Rel::Proven {
        return res;
    }
    match b {
        Type::Fun(..) if literal => literal_up(arena, res, b, b_target, env),
        Type::Fun(..) => build_cast(arena, res, b, b_target, env),
        _ => {
            let body_env = CheckEnv { infer: env.infer, local: env.local.clone(), defer: false, repeatable: true, span: env.span };
            let target = erase_open_indexed(b_target, &body_env);
            build_boundary_check(arena, res, &target, &body_env)
        }
    }
}

// The closure form of coerce_cast for a Fun `from` (relate_cast is Unknown).
// Every binder is fresh (cast_binder), so a returned function's wrapper, nested
// inside this one's lambda, never shadows it. A Vec(n) parameter, of the source
// or of the target, whose `n` has no witness in scope binds one from `__ca#k`
// (one hidden alias for both): DOWN then needs only is_list (the length half
// would be `len(__ca#k) == n`, a tautology), and UP compares a `Vec(n)` result
// with the argument. A variable already witnessed in scope is never rebound,
// so a nested wrapper compares against the outer value. Any other open index
// degrades to is_list.
// Known limitation (spec 2026-10-08 sec 7): a monomorphic `n` that no witness
// in scope covers gets a fresh per-call witness, so DOWN checks list shape
// only. The real case is `n` bound through a tuple-typed existential
// (`let y: Vec(n) = x in let p: ((Vec(n) -> Int), Int) = (fun w -> len(w), 0)
// in match p | (f, _) -> let d: Dyn = f in d([1])` returns 1), while the
// direct call `f([1])` fails at run time. Pinned by
// `a_monomorphic_vec_n_witness_checks_list_shape_only_in_the_closure_form_known_limitation`.
fn build_cast(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, env: &CheckEnv) -> ExprRef {
    let Type::Fun(a, _, b) = from else { unreachable!("build_cast takes a Fun source") };
    let (a_target, b_target) = cast_targets(to);
    let cf = cast_binder("__cf");
    let ca = cast_binder("__ca");
    let witnessed = index_witnessed(env);
    let hidden = fresh_index_name(&format!("list_{ca}"));
    let witnesses: Vec<IndexWitness> = [list_length_var(a, env.infer), list_length_var(&a_target, env.infer)]
        .into_iter()
        .flatten()
        .filter(|v| !witnessed(v))
        .map(|var| IndexWitness { var, hidden: hidden.clone(), used: Cell::new(false) })
        .collect();
    let mut local = env.local.clone();
    local.extend(witnesses.iter());
    let body_env = CheckEnv { infer: env.infer, local, defer: false, repeatable: true, span: env.span };
    let ca_ref = arena.push(Expr::Var(ca.clone()));
    let arg = if relate_cast(&a_target, a, env.infer) != Rel::Proven {
        build_boundary_check(arena, ca_ref, &erase_open_indexed(a, env), &body_env)
    } else {
        ca_ref
    };
    let cf_ref = arena.push(Expr::Var(cf.clone()));
    let call = arena.push(Expr::App(cf_ref, arg));
    let result = cast_up(arena, call, b, &b_target, &body_env, false);
    let body = match witnesses.iter().find(|w| w.used.get()) {
        Some(w) => w.bind(arena, &ca, result),
        None => result,
    };
    let ann = type_lit(arena, a_target);
    let lambda = arena.push(Expr::Lambda(ca, Some(ann), body));
    arena.push(Expr::Let(cf, None, e, lambda))
}

// Like `coerce`, but the target is "Int or Float" rather than one fixed
// type -- Add/Sub/Mul/Div/Mod/Lt's own operand check. Kept separate from
// the general coerce()/consistent() machinery ON PURPOSE: Int and Float
// stay mutually INCONSISTENT everywhere else in the type system (a
// [Float]-annotated parameter still statically rejects a [Int] argument,
// a Fun's declared Int parameter still rejects a Float argument, etc.) --
// only these specific operators treat the two as interchangeable, so this
// local helper is where that interchangeability lives. The Dyn-boundary
// runtime check is one native Int-or-Float test: it only asks "does this
// runtime value look like one of these shapes," which carries no
// implication for consistent() or any other static check.
fn coerce_numeric(arena: &mut Arena, e: ExprRef, ty: &Type, span: Span, env: &CheckEnv) -> Result<ExprRef, TypeError> {
    match ty {
        Type::Int | Type::Float => Ok(e),
        Type::Dyn | Type::Var(_) => {
            let numeric = Type::Union(Hc::new(vec![Type::Int, Type::Float]));
            Ok(build_boundary_check(arena, e, &numeric, &env.with_span(span)))
        }
        other => Err(TypeError(format!("type mismatch: expected Int or Float, found {}", other.brief()), span)),
    }
}

// Builds the actual runtime check for a definitely-Dyn-origin value
// against concrete target type `to` -- shared between coerce's own
// Dyn-to-concrete crossings and a cast's synthesized parameter and result
// checks. Infallible: by the time this runs, `to` is just "what runtime
// check to build," no static consistency question left open (that's
// coerce's own job, checked before this is ever called).
//
// Every boundary is ONE native Expr::Conform node, trusted (the binder's type
// already enforces a function's argument), whose type is lowered here
// (`lower`): no let, no env frame, no builtin call. Every function-free shape
// (Int/Float/Bool/Str/Token, Vec(k) and Vec(n), and List/Tuple/Record of
// those checked element by element, a union of those, a recursive type
// through any depth) lowers to a native test the node applies on entry; a
// Vec(n) length is read from a length slot the node evaluates then. A
// function, or a union with a function alternative, also gets a contract:
// callability is checked at once and the value is wrapped in a contract that
// checks each call's result, since a bare callability tag can't see inside a
// closure. Every failure panics with the same "type error: expected X, found
// Y" text, or the "index variable n has no runtime value" text when a length
// slot had no value.
//
// A shape lowers `to` itself under the real environment (so an index bound
// here reads as its binding, a witness as its list's length); a function or a
// union lowers a snapshot of `to` under an empty one, as an Unknown site's
// run-time lowering does. Each index variable a lowered length slot reads
// gets one slot of the node's own (read when it is entered) and one
// obligation if nothing at run time holds it yet.
fn build_boundary_check(arena: &mut Arena, e: ExprRef, to: &Type, env: &CheckEnv) -> ExprRef {
    let shape = match boundary(to, &mut Visiting::default()) {
        Boundary::None => return e,
        Boundary::Shape(_) => true,
        Boundary::Fun(..) | Boundary::Union(_) => false,
    };
    let (blank, blank_env);
    let (ty, lowering_env) = if shape {
        (to.clone(), env)
    } else {
        let vars: Vec<&str> = env.local.iter().copied().chain(env.infer.index_witness.iter()).map(|w| w.var.as_str()).collect();
        blank = InferCtx::new();
        blank_env = CheckEnv { defer: false, span: env.span, ..CheckEnv::new(&blank) };
        (snapshot_type(to, env.infer, &vars), &blank_env)
    };
    let lazy = Lazy::new(Rc::new(ty));
    let Some(lowered) = lazy.lowered.get_or_init(|| lower(&lazy, lowering_env)).clone() else { return e };
    // The index variables the check reads, each with whether it is read once per call
    // (inside a contract) and whether every length slot mentioning it is exactly that
    // variable (a bare one: see Obligation::bare).
    let mut read: BTreeMap<String, (bool, bool)> = BTreeMap::new();
    if matches!(lowered.then, Then::Done) {
        for slot in &lowered.slots {
            for name in free_index_vars(slot) {
                let (_, bare) = read.entry(name).or_insert((false, true));
                *bare &= matches!(slot, IndexExpr::Var(_));
            }
        }
    } else {
        let mut names = BTreeMap::new();
        checked_index_vars(&lazy.ty, env.infer, &mut Seen::default(), true, false, &mut names);
        read.extend(names.into_iter().map(|(name, repeatable)| (name, (repeatable, true))));
    }
    let (mut index_names, mut lens) = (Vec::with_capacity(read.len()), Vec::with_capacity(read.len()));
    for (name, (repeatable, bare)) in read {
        let index = IndexExpr::Var(name.clone());
        let term = index_len_term(arena, &index, env);
        let slot = arena.push(Expr::Len(term));
        if env.defer {
            record_obligation(slot, &index, &CheckEnv { infer: env.infer, local: env.local.clone(), defer: env.defer, repeatable: env.repeatable || repeatable, span: env.span }, bare);
        }
        index_names.push(name);
        lens.push(slot);
    }
    let spec = ConformSpec { source: TypeSource::Static(lazy), index_names, lens, span: env.span, trusted: true };
    arena.push(Expr::Conform(e, Rc::new(spec)))
}

// The kind of check a boundary against a type needs (build_boundary_check's
// dispatch, which `lower` shares).
//
// Owned: a recursive type's unfolding is a temporary, nothing to borrow from.
enum Boundary {
    // Nothing to check.
    None,
    // One native check of this function-free type.
    Shape(Type),
    // A contract on a callable value: the unfolded Fun type, its parameter
    // and its result.
    Fun(Type, Hc<Type>, Hc<Type>),
    // A union with a Fun alternative: the first alternative that holds is
    // picked, and a Fun one gets its contract.
    Union(Type),
}

// The index variables a check against `ty` can read when it runs, each with
// whether it is read inside a contract (so once per call). `spine` is where
// the check follows a Fun into its result: the target itself, a union
// alternative, an alias, a contract's result. A function's own parameter
// variable is bound by each call to its argument's length, not read from the
// site; a parameter itself is not checked (the site is trusted), and a
// function inside a container is only a callability tag.
// The recursive types it already walked: a global memo, so each is walked once. A Rec is keyed
// by group pointer and member; the Rc is kept so the pointer stays unique.
#[derive(Default)]
struct Seen {
    rec: HashSet<(*const RecGroup, usize)>,
    keep: Vec<RecRef>,
    // What each big node added in this mode. A revisit adds the same names to *its* `out`: that
    // map may differ from the first visit's (a Fun's own variable is dropped from its inner one),
    // so skipping the revisit would lose names.
    nodes: FxMap<(ById, bool, bool), BTreeMap<String, bool>>,
}

fn checked_index_vars(ty: &Type, infer: &InferCtx, seen: &mut Seen, spine: bool, in_contract: bool, out: &mut BTreeMap<String, bool>) {
    // Only an Indexed names a variable.
    if !ty.summary().any(Summary::INDEXED) {
        return;
    }
    if !is_big(ty) {
        return checked_index_vars_node(ty, infer, seen, spine, in_contract, out);
    }
    let key = (ById(ty.clone()), spine, in_contract);
    let mine = match seen.nodes.get(&key) {
        Some(done) => done.clone(),
        None => {
            let mut mine = BTreeMap::new();
            checked_index_vars_node(ty, infer, seen, spine, in_contract, &mut mine);
            seen.nodes.insert(key, mine.clone());
            mine
        }
    };
    for (name, repeatable) in mine {
        *out.entry(name).or_insert(false) |= repeatable;
    }
}

fn checked_index_vars_node(ty: &Type, infer: &InferCtx, seen: &mut Seen, spine: bool, in_contract: bool, out: &mut BTreeMap<String, bool>) {
    match ty {
        Type::Fun(param, _, ret) if spine => {
            let mut inner = BTreeMap::new();
            checked_index_vars(ret, infer, seen, true, true, &mut inner);
            if let Some(own) = list_length_var(param, infer) {
                inner.remove(&own);
            }
            for (name, repeatable) in inner {
                *out.entry(name).or_insert(false) |= repeatable;
            }
        }
        Type::Union(alts) => {
            for alt in alts.iter() {
                checked_index_vars(alt, infer, seen, spine, in_contract, out);
            }
        }
        Type::Rec(g, i) => {
            if seen.rec.insert((Rc::as_ptr(&g.0), *i)) {
                seen.keep.push(g.clone());
                checked_index_vars(&unfold(g, *i), infer, seen, spine, in_contract, out);
            }
        }
        Type::Indexed(wrapped, index) => {
            for name in free_index_vars(index) {
                *out.entry(name).or_insert(false) |= in_contract;
            }
            checked_index_vars(wrapped, infer, seen, spine && !matches!(**wrapped, Type::List(_)), in_contract, out);
        }
        Type::List(elem) => checked_index_vars(elem, infer, seen, false, in_contract, out),
        Type::Tuple(items) => {
            for t in items.iter() {
                checked_index_vars(t, infer, seen, false, in_contract, out);
            }
        }
        Type::Record(fields) => {
            for (_, t) in fields.iter() {
                checked_index_vars(t, infer, seen, false, in_contract, out);
            }
        }
        _ => {}
    }
}

// `to` with every alias wrapper and recursive type above its first real shape
// unfolded, each one added to `visiting`: checked exactly as if `to` had been
// written directly as that unfolded shape. A recursive type nested inside a
// container of that shape is checked by the native deep test (shape_of's
// `Ref`s); one reached only through a Fun position stays unresolved -- checked
// later, lazily, only if something else separately touches that position.
fn boundary(to: &Type, visiting: &mut Visiting) -> Boundary {
    match to {
        Type::Int | Type::Float | Type::Bool | Type::Str | Type::List(_) | Type::Token(_) | Type::Tuple(_) | Type::Record(_) => Boundary::Shape(to.clone()),
        Type::Fun(param_ty, _row, ret_ty) => Boundary::Fun(to.clone(), param_ty.clone(), ret_ty.clone()),
        Type::Union(_) if needs_contract(to, visiting) => Boundary::Union(to.clone()),
        Type::Union(_) => Boundary::Shape(to.clone()),
        // Dyn accepts everything, so a check against it is the identity. Not
        // unreachable: a contract checks a call's result against the Fun's
        // return type, which is Dyn for `Dyn -> Dyn`.
        Type::Dyn | Type::Var(_) => Boundary::None,
        // A placeholder never reaches a check (machine::pure_type rejects it).
        Type::RecVar(_) => Boundary::None,
        // A list's length, literal or read at run time from a length slot,
        // is part of its one native test.
        Type::Indexed(w, _) if matches!(**w, Type::List(_)) => Boundary::Shape(to.clone()),
        // An alias (or Dyn/Var) wrapper: no runtime length to compare, but the shape is checked.
        Type::Indexed(wrapped, _) => boundary(wrapped, visiting),
        // If this member is already an ancestor in THIS unfold chain (by group
        // pointer and member), stop instead of unfolding again, and check nothing
        // (matching the Type::Var(_) arm above): `let rec A = Union(A, Fun(Int,
        // Int))` would otherwise unfold forever. A contract's result is a Lazy of
        // its own (shared with every equal type of the same lowering,
        // Lazy::child), so a self-reference there (`let rec F = Fun(Dyn, F)`) is
        // checked at each call.
        Type::Rec(g, i) => {
            if visiting.has_rec(g, *i) {
                return Boundary::None;
            }
            visiting.rec.push((g.clone(), *i));
            boundary(&unfold(g, *i), visiting)
        }
    }
}

// An Unknown site's type value, lowered when the run time reaches the site
// (spec 11.5): `lower` under an empty environment, blaming the site's span.
pub(crate) fn lower_conform(lazy: &Lazy, span: Option<Span>) -> Option<Rc<Lowered>> {
    let infer = InferCtx::new();
    lower(lazy, &CheckEnv { defer: false, span, ..CheckEnv::new(&infer) })
}

// A type as checks, for a static boundary at check time (build_boundary_check)
// and an Unknown site's type value at run time (lower_conform): the checks for
// any type, with the same failure messages. None: nothing to check. A function
// is checked for callability and wrapped in the site's contract (unless the
// contract would check nothing, see `wrap`); a union with a function
// alternative picks the first alternative whose shape holds, and its check is
// the failure when none does. `env` is where an index variable gets its length:
// the real one at a static shape, an empty one otherwise.
fn lower(lazy: &Lazy, env: &CheckEnv) -> Option<Rc<Lowered>> {
    let ty = &*lazy.ty;
    let mut visiting = Visiting::default();
    let mut defs = Defs::default();
    let (test, to, then) = match boundary(ty, &mut visiting) {
        Boundary::None => return None,
        Boundary::Shape(to) => (shape_test(&to, env, &mut defs, &Visiting::default()), to, Then::Done),
        Boundary::Fun(fun_ty, param, ret) => (Test::Fun, fun_ty, wrap(lazy, &param, &ret)),
        Boundary::Union(to) => {
            let then = pick(lazy, &to, env, &mut defs, &visiting);
            (Test::Never, to, then)
        }
    };
    let bodies = defs.bodies.into_iter().map(|body| body.expect("every alias body is built before shape_test returns")).collect();
    let check = CheckSpec { test, defs: bodies, to: to.clone(), span: env.span };
    Some(Rc::new(Lowered { check: Rc::new(check), slots: defs.lens, then }))
}

// What a function boundary with this parameter and result needs: a contract,
// unless it would check nothing (both types are checked by nothing): a bare
// callability test, so a Dyn callee or a `Dyn -> Dyn` annotation builds no
// contract per call.
fn wrap(lazy: &Lazy, param: &Hc<Type>, ret: &Hc<Type>) -> Then {
    let free = |t: &Hc<Type>| matches!(boundary(t, &mut Visiting::default()), Boundary::None);
    if free(param) && free(ret) {
        return Then::Done;
    }
    Then::Wrap(lazy.child(Rc::new((**param).clone())), lazy.child(Rc::new((**ret).clone())))
}

// A union's alternatives as lower's Pick: each one's shape test
// (one already being unfolded above never holds) and what a value passing it
// still needs. Every
// alternative shares one alias table and one set of length slots.
fn pick(lazy: &Lazy, to: &Type, env: &CheckEnv, defs: &mut Defs, visiting: &Visiting) -> Then {
    let Type::Union(alts) = to else { unreachable!("pick is only ever called with a Union target") };
    let pick_one = |alt: &Type, defs: &mut Defs| {
        let test = match alt {
            Type::Rec(g, i) if visiting.has_rec(g, *i) => Test::Never,
            _ => shape_test(alt, env, defs, &Visiting::default()),
        };
        let mut visiting = visiting.clone();
        let then = match boundary(alt, &mut visiting) {
            Boundary::Fun(_, param, ret) => wrap(lazy, &param, &ret),
            Boundary::Union(nested) => pick(lazy, &nested, env, defs, &visiting),
            Boundary::None | Boundary::Shape(_) => Then::Done,
        };
        (test, then)
    };
    Then::Pick(alts.iter().map(|alt| pick_one(alt, defs)).collect())
}

// A runtime value for an index variable: `hidden` is a variable bound by
// synthesized code to the list itself (a Lambda's parameter, or a
// contract's argument); a check reads the length of `hidden` (a Len::Witness slot) when it is entered, so
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
// crossing, outermost first. `slot` is the Conform's `Expr::Len` length slot
// for the variable, which resolve_obligations patches in place (every copy of
// the Conform shares it).
#[derive(Clone)]
struct Obligation {
    slot: ExprRef,
    index: IndexExpr,
    witnesses: Vec<(String, String)>,
    // The crossing sits inside a function body, so one site can produce many
    // values sharing its index variable.
    repeatable: bool,
    // Every length slot of the site that reads the variable is that variable
    // alone, so a lone unconstrained one may be `Any`. A variable inside a
    // compound index (`Vec(n + 1)`) has no length of its own to leave free.
    bare: bool,
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

// `e` as a runtime check reads it (index_len_term's order): a variable
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

// Records an obligation for a length slot (a union alternative's too) when its
// index has a variable index_len_term cannot give a runtime value (the slot
// holds that clean `NoValue` failure until resolved). Every witness in scope
// is marked used: the resolved slot may read any of them, and their alias
// binders are emitted (or not) before typechecking finishes.
fn record_obligation(slot: ExprRef, index: &IndexExpr, env: &CheckEnv, bare: bool) {
    if let Some(ob) = make_obligation(slot, index, env, bare) {
        env.infer.obligations.borrow_mut().push(ob);
    }
}

fn make_obligation(slot: ExprRef, index: &IndexExpr, env: &CheckEnv, bare: bool) -> Option<Obligation> {
    let in_scope: Vec<&IndexWitness> = env.infer.index_witness.iter().chain(env.local.iter().copied()).collect();
    let vars: Vec<&str> = in_scope.iter().map(|w| w.var.as_str()).collect();
    let index = runtime_index(index, env.infer, &vars, false);
    if free_index_vars(&index).iter().all(|v| vars.contains(&v.as_str())) {
        return None;
    }
    for w in &in_scope {
        w.used.set(true);
    }
    let witnesses = in_scope.iter().map(|w| (w.var.clone(), w.hidden.clone())).collect();
    Some(Obligation { slot, index, witnesses, repeatable: env.infer.repeatable_depth > 0 || env.repeatable, bare })
}

// Patches every obligation's length slot once inference has finished, so
// bindings made after the crossing count: the index under the final
// bindings, each variable read from a witness in scope at the crossing,
// otherwise the clean `NoValue` failure. Runs before anything evaluates the
// tree.
//
// An index that is still a bare unbound variable no other obligation
// mentions, that no compound length slot of its site reads either (`bare`), and
// that sits in no function body, constrains nothing, so the
// check is "is a list" alone (the slot becomes `Any`). Two crossings sharing
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
                && ob.bare
                && !ob.repeatable
                && !ob.witness_vars().contains(&name.as_str())
                && mentions.get(name) == Some(&1)
                && !infer.index_subst.values().any(|v| free_index_vars(v).contains(name)));
        if unconstrained {
            arena[ob.slot] = Expr::Len(LenExpr::Any);
            continue;
        }
        let witnesses: Vec<IndexWitness> = ob
            .witnesses
            .iter()
            .map(|(var, hidden)| IndexWitness { var: var.clone(), hidden: hidden.clone(), used: Cell::new(true) })
            .collect();
        let env = CheckEnv { infer, local: witnesses.iter().collect(), defer: false, repeatable: false, span: None };
        let term = index_len_term(arena, &index, &env);
        arena[ob.slot] = Expr::Len(term);
    }
}

// What the runtime-check builders (build_boundary_check and friends) need:
// the index state (InferCtx, as of where the check is
// spliced); `local` holds witnesses bound inside synthesized code
// (a cast's parameter alias). `defer` is false inside a cast's synthesized body
// (build_cast, cast_up) and in resolve_obligations, which record no obligations.
struct CheckEnv<'a> {
    infer: &'a InferCtx,
    local: Vec<&'a IndexWitness>,
    defer: bool,
    // Building inside a synthesized function body (a cast's wrapper), which
    // runs on every call of the wrapped function.
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

// An index as a length slot reads it at run time, per variable, first match
// wins:
// 1. a witness: the length of the innermost list parameter (or contract
//    argument) typed Vec(name), read through its hidden alias;
// 2. its static binding in index_subst, translated in turn;
// 3. otherwise nothing at runtime holds it (an existential from a plain
//    `let x: Vec(n) = <Dyn>`, or a call-site instantiation): `NoValue`, the
//    check's clean failure rather than a read of an unbound variable.
// A same-named TERM variable (`fun n: Int -> fun v: Vec(n)`) is never used:
// the typechecker does not link a term `n` to the index `n`, so reading it
// would let a wrong-length value through (or reject a right one).
fn index_len_term(arena: &mut Arena, e: &IndexExpr, env: &CheckEnv) -> LenExpr {
    let mut go = |x: &IndexExpr| Box::new(index_len_term(arena, x, env));
    match e {
        IndexExpr::Var(name) => {
            let witness = env.local.iter().rev().copied().chain(env.infer.index_witness.iter().rev()).find(|w| w.var == *name);
            if let Some(w) = witness {
                w.used.set(true);
                return LenExpr::Witness(arena.push(Expr::Var(w.hidden.clone())), name.as_str().into());
            }
            match env.infer.index_subst.get(name) {
                Some(bound) => index_len_term(arena, bound, env),
                None => LenExpr::NoValue(name.as_str().into()),
            }
        }
        IndexExpr::Lit(n) => LenExpr::Lit(*n),
        IndexExpr::Add(a, b) => LenExpr::Add(go(a), go(b)),
        IndexExpr::Sub(a, b) => LenExpr::Sub(go(a), go(b)),
        IndexExpr::Mul(a, b) => LenExpr::Mul(go(a), go(b)),
    }
}

// A native Test plus the alias table its `Test::Ref`s index and the length
// slots its `Test::ListLenArg`s index: each an index as the check reads it
// (runtime_index). What the `is` builtin lowers a type value to (lower_test).
struct Shape {
    test: Test,
    defs: Vec<Test>,
    lens: Vec<IndexExpr>,
}

// What shape_test accumulates: each recursive type's body (None while it is being
// built) and the length slots.
#[derive(Default)]
struct Defs {
    // A Rec's body, by exact identity (a global memo: each Rec is unfolded once).
    rec_ids: FxMap<ById, usize>,
    // A big compound node's alias: None after its first visit, Some(i) once a second visit gave
    // it a body (so a shared node is tested once per value, not once per path to it). Keyed like
    // StackMemo, since an unguarded Union alternative depends on the recursive types above it;
    // `keep` holds those groups so their pointers stay unique.
    shared: FxMap<StackKey, Option<usize>>,
    keep: Vec<RecRef>,
    bodies: Vec<Option<Test>>,
    lens: Vec<IndexExpr>,
}

// `ty` as one native Shape.
fn shape_of(ty: &Type, env: &CheckEnv) -> Shape {
    let mut defs = Defs::default();
    let test = shape_test(ty, env, &mut defs, &Visiting::default());
    let bodies = defs.bodies.into_iter().map(|body| body.expect("every alias body is built before shape_test returns")).collect();
    Shape { test, defs: bodies, lens: defs.lens }
}

// The native test for a run-time type value (the `is` builtin): the test and
// its alias table, shape_of under an empty environment. A Vec whose length is
// a variable has no value to compare against, so it is an Err.
pub(crate) fn lower_test(ty: &Type) -> Result<(Test, Vec<Test>), String> {
    let infer = InferCtx::new();
    let shape = shape_of(ty, &CheckEnv::new(&infer));
    match shape.lens.is_empty() {
        true => Ok((shape.test, shape.defs)),
        false => Err("is: type has a free length variable".to_string()),
    }
}

// The deep test for `ty`: what the value must be, element by element
// (function-free types; a Fun position is only a callability tag). A Vec(n)
// whose index is not a literal reads its length from a slot (len_slot).
// `unguarded` holds
// the recursive types unfolded since the last container: meeting one again
// there would loop without consuming the value, so it is `Never` (making
// `let rec A = Union(Int, A)` behave as plain Int). Past a container, a recursive
// type becomes one `Ref` into `defs`, whose body is built once, so it
// terminates because the value does. A placeholder is `Any` ("no information
// here"; machine::pure_type keeps it out of every check).
fn shape_test(ty: &Type, env: &CheckEnv, defs: &mut Defs, unguarded: &Visiting) -> Test {
    // A big node reached again becomes one alias (Test::Ref), so a type that is a DAG yields a
    // test of the DAG's size, and the machine's (alias, value) memo keeps the walk linear.
    let compound = matches!(ty, Type::List(_) | Type::Tuple(_) | Type::Record(_) | Type::Union(_) | Type::Indexed(..));
    let key = if compound { StackMemo::default().key(ty, 0, unguarded) } else { None };
    let Some(key) = key else { return shape_node(ty, env, defs, unguarded) };
    match defs.shared.get(&key).copied() {
        Some(Some(i)) => Test::Ref(i),
        Some(None) => {
            let i = defs.bodies.len();
            defs.bodies.push(None);
            defs.shared.insert(key, Some(i));
            let body = shape_node(ty, env, defs, unguarded);
            defs.bodies[i] = Some(body);
            Test::Ref(i)
        }
        None => {
            defs.keep.extend(unguarded.rec.iter().map(|(g, _)| g.clone()));
            defs.shared.insert(key, None);
            shape_node(ty, env, defs, unguarded)
        }
    }
}

fn shape_node(ty: &Type, env: &CheckEnv, defs: &mut Defs, unguarded: &Visiting) -> Test {
    let element = |t: &Type, defs: &mut Defs| shape_test(t, env, defs, &Visiting::default());
    match ty {
        Type::Dyn | Type::Var(_) | Type::RecVar(_) => Test::Any,
        Type::Int => Test::Int,
        Type::Float => Test::Float,
        Type::Bool => Test::Bool,
        Type::Str => Test::Str,
        Type::List(el) => match element(el, defs) {
            Test::Any => Test::List,
            t => Test::ListOf(Box::new(t)),
        },
        Type::Fun(..) => Test::Fun,
        Type::Token(id) => Test::Token(*id),
        // Exact arity, unlike Record's width tolerance.
        Type::Tuple(items) => Test::TupleOf(items.iter().map(|t| element(t, defs)).collect()),
        // Width-tolerant: `v` need only HAVE (at least) every required
        // field -- extra fields are fine, see Pattern::Record's own doc
        // comment for why nothing downstream can ever observe them.
        Type::Record(fields) => Test::RecordOf(fields.iter().map(|(name, t)| (name.clone(), element(t, defs))).collect()),
        Type::Union(alts) => Test::Or(alts.iter().map(|alt| shape_test(alt, env, defs, unguarded)).collect()),
        Type::Rec(g, k) if unguarded.has_rec(g, *k) => Test::Never,
        Type::Rec(g, k) if !unguarded.is_empty() => shape_test(&unfold(g, *k), env, defs, &unguarded.with_rec(g, *k)),
        Type::Rec(g, k) => match defs.rec_ids.get(&ById(ty.clone())) {
            Some(&i) => Test::Ref(i),
            None => {
                let i = defs.bodies.len();
                defs.bodies.push(None);
                defs.rec_ids.insert(ById(ty.clone()), i);
                let body = shape_test(&unfold(g, *k), env, defs, &unguarded.with_rec(g, *k));
                defs.bodies[i] = Some(body);
                Test::Ref(i)
            }
        },
        // A literal length is tested as is; any other index is read from a slot.
        Type::Indexed(wrapped, index) => match (wrapped.as_ref(), env.infer.resolve_index_deep(index)) {
            (Type::List(el), IndexExpr::Lit(k)) => match usize::try_from(k) {
                Ok(k) => Test::ListLen(Box::new(element(el, defs)), k),
                Err(_) => Test::Never,
            },
            (Type::List(el), _) => {
                let slot = len_slot(index, env, defs);
                Test::ListLenArg(Box::new(element(el, defs)), slot)
            }
            (other, _) => shape_test(other, env, defs, unguarded),
        },
    }
}

// The length slot for a Vec(index): one per distinct index as the check reads
// it (runtime_index), so every position with the same index, at any depth and
// through every level of a recursive alias, compares against one value.
fn len_slot(index: &IndexExpr, env: &CheckEnv, defs: &mut Defs) -> usize {
    let vars: Vec<&str> = env.local.iter().copied().chain(env.infer.index_witness.iter()).map(|w| w.var.as_str()).collect();
    let index = runtime_index(index, env.infer, &vars, false);
    match defs.lens.iter().position(|slot| *slot == index) {
        Some(i) => i,
        None => {
            defs.lens.push(index);
            defs.lens.len() - 1
        }
    }
}

// Does checking `ty` do more than its shape test -- is a Fun reachable, so a
// matching value must also get its per-call contract? Decides whether a
// union check picks its alternative (Boundary::Union).
pub(crate) fn needs_contract(ty: &Type, visiting: &Visiting) -> bool {
    fn go(ty: &Type, visiting: &Visiting, memo: &mut StackMemo) -> bool {
        let key = memo.key(ty, 0, visiting);
        if let Some(done) = key.as_ref().and_then(|k| memo.map.get(k)) {
            return *done;
        }
        let result = match ty {
            Type::Fun(..) => true,
            Type::Union(alts) => alts.iter().any(|alt| go(alt, visiting, memo)),
            Type::Rec(g, i) if !visiting.has_rec(g, *i) => go(&unfold(g, *i), &visiting.with_rec(g, *i), memo),
            _ => false,
        };
        memo.put(key, visiting, result);
        result
    }
    go(ty, visiting, &mut StackMemo::default())
}

// A walk's answers for big nodes, valid for one call: keyed by the node, a caller-chosen
// number (the address of a pattern, else 0) and the recursive types being unfolded above it
// (`Visiting`), since the answer may depend on any of them.
#[derive(Default)]
struct StackMemo {
    map: FxMap<StackKey, bool>,
    // The groups the keys point at, kept so an address is never reused within the call.
    keep: Vec<RecRef>,
}

type StackKey = (ById, usize, Vec<(*const RecGroup, usize)>);

impl StackMemo {
    fn key(&self, ty: &Type, extra: usize, visiting: &Visiting) -> Option<StackKey> {
        is_big(ty).then(|| (ById(ty.clone()), extra, visiting.rec.iter().map(|(g, i)| (Rc::as_ptr(&g.0), *i)).collect()))
    }

    fn put(&mut self, key: Option<StackKey>, visiting: &Visiting, result: bool) {
        if let Some(key) = key {
            self.keep.extend(visiting.rec.iter().map(|(g, _)| g.clone()));
            self.map.insert(key, result);
        }
    }
}

// A synthesized reference to the prelude builtin `name`. Spelled `#name`,
// which no source text can bind or lex, and which `resolve` maps straight to
// the prelude table: a user binding named like the builtin (a parameter
// `get_field`) cannot capture a synthesized call. The one builtin synthesized
// code still calls, `get_field`, goes through here.
fn prelude_var(arena: &mut Arena, name: &str) -> ExprRef {
    arena.push(Expr::Var(format!("#{name}")))
}

// `builtin(value_ref, str_arg)`, used by `.field` access's get_field
// desugaring.
fn build_str_call(arena: &mut Arena, builtin: &str, value_ref: ExprRef, str_arg: &str) -> ExprRef {
    let f = prelude_var(arena, builtin);
    let applied = arena.push(Expr::App(f, value_ref));
    let arg_lit = arena.push(Expr::Str(str_arg.to_string()));
    arena.push(Expr::App(applied, arg_lit))
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
// over an `Indexed(Rec(_))` expected type must also be added to
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
    // `def` decides whether the binding stays (CheckDef::erasable, settle_erasure).
    Let { var: String, bound_ty: Type, val_row: EffectRow, val: ExprRef, def: Rc<CheckDef> },
    // One `let rec` group's elaborated bindings (each carrying its own
    // inferred/annotated type, row, and elaborated value), folding back
    // into a single Expr::LetRec.
    // `sites`: the members' kept annotations, evaluated before the group.
    LetRec { bindings: Vec<(String, Type, EffectRow, ExprRef)>, def: Rc<CheckDef>, sites: Vec<(String, Site)>, kind: GroupKind },
    // `witness`: this Lambda pushed an infer.index_witness entry (its
    // parameter is Vec(n)), popped when the frame unwinds. `site`: its
    // parameter's kept annotation; `direct`: it is a recursive group's value.
    Fun { param: String, param_ty: Type, witness: bool, site: Option<Site>, direct: bool },
}

// Recognizes the handler-expression shapes this codebase actually
// produces (MakeHandler, optionally wrapped in deep(...)/shallow(...))
// well enough to know which effect name a `handle` discharges. Anything
// else (a bare variable, a computed handler) returns None --
// Handle then widens the body's row to Dyn, which is
// the sound direction: an unknown handler may catch anything, so the
// effect check defers to run time instead of rejecting.
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
        Pattern::List(_) | Pattern::Cons(..) => Type::List(Hc::new(Type::Dyn)),
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
    // A node walked before answered true (else the walk would have stopped there).
    fn go(ty: &Type, seen: &mut Visited) -> bool {
        match ty {
            Type::Record(_) => true,
            Type::Union(alts) => !seen.first(ty) || alts.iter().all(|alt| go(alt, seen)),
            _ => false,
        }
    }
    go(ty, &mut Visited::new())
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
    fn go(ty: &Type, name: &str, memo: &mut FxMap<ById, Option<Type>>) -> Option<Type> {
        match ty {
            Type::Record(fields) => find_field(fields, name).cloned(),
            Type::Union(alts) => {
                if is_big(ty)
                    && let Some(done) = memo.get(&ById(ty.clone()))
                {
                    return done.clone();
                }
                let mut result: Option<Type> = None;
                let mut missing = false;
                for alt in alts.iter() {
                    let Some(t) = go(alt, name, memo) else {
                        missing = true;
                        break;
                    };
                    result = Some(match &result {
                        Some(prev) if *prev == t => t,
                        Some(_) => Type::Dyn,
                        None => t,
                    });
                }
                let result = if missing { None } else { result };
                if is_big(ty) {
                    memo.insert(ById(ty.clone()), result.clone());
                }
                result
            }
            _ => None,
        }
    }
    go(ty, name, &mut FxMap::default())
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
// The recursive types being unfolded somewhere above a point in a walk (an ancestor chain, not
// a global memo), by group pointer and member (the Rc is kept so the pointer stays unique).
#[derive(Clone, Default)]
pub(crate) struct Visiting {
    rec: Vec<(RecRef, usize)>,
}

impl Visiting {
    fn of(rec: &Type) -> Visiting {
        match rec {
            Type::Rec(g, i) => Visiting::default().with_rec(g, *i),
            _ => Visiting::default(),
        }
    }

    fn with_rec(&self, g: &RecRef, i: usize) -> Visiting {
        let mut v = self.clone();
        v.rec.push((g.clone(), i));
        v
    }

    fn has_rec(&self, g: &RecRef, i: usize) -> bool {
        self.rec.iter().any(|(h, k)| *k == i && Rc::ptr_eq(&h.0, &g.0))
    }

    fn is_empty(&self) -> bool {
        self.rec.is_empty()
    }
}

// `visiting` is the same ancestor-tracking device `boundary` uses:
// the recursive types currently being unfolded somewhere ABOVE
// this call in the SAME unfold chain, empty at the one genuine entry
// point (the Expr::Match call site below) and extended by exactly one member
// each time the Type::Rec arm actually descends into an unfold. Needed
// for the same reason it's needed there: a self-referential type whose
// recursive occurrence is a BARE Union alternative (`let rec A = Union(Int, A)`)
// unfolds to the exact same Union([Int, A]) every time, and a
// Record/List pattern's own Union arm just below fans out into EVERY
// alternative (including that same A again) -- without this,
// that's genuine infinite recursion in this function's own call stack,
// not just a runtime concern.
pub(crate) fn pattern_could_match(pat: &Pattern, ty: &Type, visiting: &Visiting) -> bool {
    could_match(pat, ty, visiting, &mut StackMemo::default())
}

fn could_match(pat: &Pattern, ty: &Type, visiting: &Visiting, memo: &mut StackMemo) -> bool {
    let key = memo.key(ty, pat as *const Pattern as usize, visiting);
    if let Some(done) = key.as_ref().and_then(|k| memo.map.get(k)) {
        return *done;
    }
    let result = match (pat, ty) {
        (Pattern::Record(fields), Type::Record(type_fields)) => {
            fields.iter().all(|(name, _)| find_field(type_fields, name).is_some())
        }
        (Pattern::Record(_), Type::Union(alts)) => alts.iter().any(|alt| could_match(pat, alt, visiting, memo)),
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
        (Pattern::List(_), Type::Union(alts)) => alts.iter().any(|alt| could_match(pat, alt, visiting, memo)),
        (Pattern::List(_), Type::Dyn | Type::Var(_)) => true,
        // One level only, matching every other consumer of Type::Rec in this
        // file: unfold what this member stands for and ask the SAME question
        // about that shape instead.
        //
        // Ancestor-tracking cycle break (by group pointer and member), same as
        // `boundary`'s own Rec arm: if the member is already being unfolded
        // somewhere above this call, stop and answer `false` instead of
        // unfolding again -- re-trying the same member against the same pattern
        // gives no new information beyond what the non-recursive alternatives
        // already cover (the `any()` fan-out above already tried them), so this
        // correctly makes e.g. `let rec A = Union(Int, A)` behave equivalently
        // to plain `Int` for reachability purposes instead of recursing forever.
        (_, Type::Rec(g, i)) => {
            if visiting.has_rec(g, *i) {
                return false;
            }
            could_match(pat, &unfold(g, *i), &visiting.with_rec(g, *i), memo)
        }
        // An Indexed-typed scrutinee is, at runtime, just its wrapped
        // type's own value (Vec(n) sugars over this) -- coverage of
        // []/:: is a purely structural question, independent of what the
        // index is (see the design spec's own §5 note that exhaustiveness
        // is unaffected by Vec). Delegate straight to the wrapped type,
        // mirroring shape_test's own Type::Indexed arm.
        (_, Type::Indexed(wrapped, _)) => could_match(pat, wrapped, visiting, memo),
        _ => consistent(ty, &pattern_type(pat)),
    };
    memo.put(key, visiting, result);
    result
}

// Eligibility (spec section 5): is `scrut_ty` (already resolved via
// infer.resolve_deep) a Type::Indexed wrapping a shape Match's own arm
// loop knows how to derive a base/step hypothesis from, whose own index
// expression (already resolved via infer.resolve_index_deep) is a BARE
// variable? Two wrapped shapes qualify -- Case A, a plain Type::List
// (the fixed []/:: rule), and Case B, a qualifying recursive Union
// (the rule derived from THAT type's own structure,
// see `qualifying_rec_alternatives`) -- anything else (a non-Indexed
// scrutinee, a wrapped shape neither case recognizes, a literal or
// compound index expression with no single name to bind a hypothesis
// under) returns None: the safe "not applicable" fallback used
// throughout this design. Renamed from Task 1's own
// `case_a_refinement_target` now that Task 3 extends it to cover Case B
// too.
enum IndexedRefinementTarget {
    List { index_var: String, elem_ty: Hc<Type> },
    // `rec` is the recursive type (a Type::Rec) the Indexed wraps;
    // `base_alt`/`step_alt` are the two alternatives
    // `qualifying_alternatives` already sorted out -- the SAME
    // per-alternative shapes Match's own arm loop re-checks a pattern
    // against via `pattern_could_match` to decide which hypothesis (if
    // any) that specific arm earns. `self_ref` says where the step
    // alternative mentions `rec`.
    Rec { index_var: String, rec: Type, base_alt: Rc<Type>, step_alt: Rc<Type>, self_ref: SelfRef },
}

// Where a step alternative mentions the recursive type being refined: the alternative itself,
// one bare Tuple item, or somewhere deeper (not recognised: the safe "not applicable").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelfRef {
    Whole,
    Item(usize),
    Deep,
}

fn indexed_refinement_target(resolved_scrut_ty: &Type, resolved_index: Option<&IndexExpr>) -> Option<IndexedRefinementTarget> {
    match (resolved_scrut_ty, resolved_index) {
        (Type::Indexed(wrapped, _), Some(IndexExpr::Var(name))) => match &**wrapped {
            Type::List(elem_ty) => Some(IndexedRefinementTarget::List { index_var: name.clone(), elem_ty: elem_ty.clone() }),
            rec @ Type::Rec(g, i) => qualifying_rec_alternatives(g, *i).map(|(base_alt, step_alt, self_ref)| IndexedRefinementTarget::Rec {
                index_var: name.clone(),
                rec: rec.clone(),
                base_alt: Rc::new(base_alt),
                step_alt: Rc::new(step_alt),
                self_ref,
            }),
            _ => None,
        },
        _ => None,
    }
}

// Case B eligibility (spec section 5 points 1-2) for member `i` of a group: the raw body
// is a Union of EXACTLY two alternatives, one mentioning the member not at all (the base
// case, 0 occurrences) and the other exactly once (the step case, at any depth, nested groups
// included -- not zero, not two-or-more: a binary tree's `Node(Tree, Tree)` step deliberately
// doesn't qualify, per the spec's own Non-goals). Anything else is not eligible: the same
// safe "not applicable" fallback used throughout this design. Returns
// (base_alt, step_alt, self_ref) in that fixed order regardless of which one the body
// stores first. The alternatives come back unfolded; `self_ref` is read off
// the raw step, where the member is still a Bound.
fn qualifying_rec_alternatives(g: &RecRef, i: usize) -> Option<(Type, Type, SelfRef)> {
    let Type::Union(raw) = &g.0.bodies[i] else { return None };
    let [a, b] = raw.as_slice() else { return None };
    let (base, step) = match (count_bound(a, 0, i), count_bound(b, 0, i)) {
        (0, 1) => (0, 1),
        (1, 0) => (1, 0),
        _ => return None,
    };
    let Type::Union(unfolded) = unfold(g, i) else { return None };
    let is_self = |t: &Type| matches!(t, Type::RecVar(RecVar::Bound { depth: 0, member }) if *member as usize == i);
    let self_ref = match &raw[step] {
        t if is_self(t) => SelfRef::Whole,
        Type::Tuple(items) => items.iter().position(is_self).map_or(SelfRef::Deep, SelfRef::Item),
        _ => SelfRef::Deep,
    };
    Some((unfolded[base].clone(), unfolded[step].clone(), self_ref))
}

// How many references to member `member` of the group whose body is being walked does `t`
// hold? `nest` counts the groups entered since.
// Saturated at 2: callers only ask for none, one, or more.
fn count_bound(t: &Type, nest: u32, member: usize) -> usize {
    fn go(t: &Type, nest: u32, member: usize, memo: &mut NodeMemo<usize>) -> usize {
        // No reference leaves a subtree whose reach is within `nest`.
        if t.summary().reach() <= nest {
            return 0;
        }
        let key = is_big(t).then(|| (ById(t.clone()), nest));
        if let Some(done) = key.as_ref().and_then(|k| memo.get(k)) {
            return *done;
        }
        let result = match t {
            Type::RecVar(RecVar::Bound { depth, member: m }) => usize::from(*depth == nest && *m as usize == member),
            Type::Rec(h, _) => h.0.bodies.iter().map(|b| go(b, nest + 1, member, memo)).sum::<usize>().min(2),
            _ => {
                let mut n = 0;
                each_child(t, &mut |c| n = (n + go(c, nest, member, memo)).min(2));
                n
            }
        };
        if let Some(key) = key {
            memo.insert(key, result);
        }
        result
    }
    go(t, nest, member, &mut NodeMemo::default())
}

// Case B's own analog of Case A's `Pattern::Cons(_, tail)` tail-position
// check (spec section 5 point 3): locates the sub-pattern bound at the
// step alternative's own single self-referential position -- guaranteed
// to exist and be unique by `qualifying_rec_alternatives`'s own
// occurrence-count check -- so the caller can retype it as
// `Type::Indexed(rec, fresh)`, the same override Case A already
// does for a Cons's own tail. Only the spec's own worked shape is
// recognized: the self-reference IS the whole alternative (`let rec X =
// Union(Y, X)`), or the alternative is a Tuple with the self-reference as one
// bare item (the spec's own `(Int, List)`) -- mirroring Case A's own
// "bare Var tail only" v1 restriction, anything with the self-reference
// nested deeper (inside a further Tuple/List/Record) returns None, the
// same safe fallback.
fn self_ref_pattern<'a>(pat: &'a Pattern, step_alt: &Type, self_ref: SelfRef) -> Option<&'a Pattern> {
    match (self_ref, step_alt, pat) {
        (SelfRef::Whole, _, _) => Some(pat),
        (SelfRef::Item(k), Type::Tuple(items), Pattern::List(pats)) if pats.len() == items.len() => pats.get(k),
        _ => None,
    }
}

// Case B construction, step case (design spec 2026-09-20 sec 4):
// mirrors self_ref_pattern's own Type::Tuple sub-case (used on the
// pattern-matching side, Phase 4) but returns a POSITION into the
// Tuple's items instead of a &Pattern, since a Tuple literal's items
// at a construction site are ExprRefs, not Patterns to recurse into.
// Deliberately does NOT also mirror self_ref_pattern's separate
// `SelfRef::Whole` case (for a degenerate `let rec X
// = Union(Y, X)` shape, the self-reference IS the whole alternative) -- not
// an oversight, just unreachable from this function's only call site,
// which has already matched `step_alt` as a `Type::Tuple` (the
// step-shaped-literal check) before ever calling this.
fn self_ref_position(step_alt: &Type, self_ref: SelfRef) -> Option<usize> {
    match (step_alt, self_ref) {
        (Type::Tuple(_), SelfRef::Item(k)) => Some(k),
        _ => None,
    }
}

// Case B construction, base-case detection for elaborate_mode's own
// tail dispatch (design spec 2026-09-20 sec 4): `expected` must
// resolve to Indexed(rec, idx) with a qualifying 2-alternative
// union (qualifying_rec_alternatives), and `cur_expr` must NOT
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
// `Indexed(rec, m)`) never got computed under Synth, so checking
// its body against `Indexed(rec, n - 1)` failed. Excluding
// If/Match here, exactly like Tuple's own step-shape exclusion, lets
// elaborate_node's normal dispatch handle them and fixes it.
fn case_b_base_candidate(arena: &Arena, cur_expr: ExprRef, expected: &Type, infer: &InferCtx) -> Option<(Type, Type, SelfRef, Rc<IndexExpr>)> {
    let Type::Indexed(wrapped, idx) = infer.resolve_deep(expected) else { return None };
    let Type::Rec(g, i) = &*wrapped else { return None };
    let (base_alt, step_alt, self_ref) = qualifying_rec_alternatives(g, *i)?;
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
    Some((base_alt, step_alt, self_ref, idx))
}

// Does `ty` mention an Indexed type anywhere? Gates pushing an expected
// element type down into a Tuple/List literal's items: only Indexed
// element types need it (a plain literal can never satisfy one), and
// checking every item would insert runtime casts for Dyn items that the
// whole-literal coerce deliberately leaves alone.
fn contains_indexed(ty: &Type) -> bool {
    // A node walked before held none (else the walk would have stopped there).
    fn go(ty: &Type, seen: &mut Visited) -> bool {
        if !ty.summary().any(Summary::INDEXED) || !seen.first(ty) {
            return false;
        }
        match ty {
            Type::Indexed(..) => true,
            Type::List(t) => go(t, seen),
            Type::Tuple(ts) | Type::Union(ts) => ts.iter().any(|t| go(t, seen)),
            Type::Record(fs) => fs.iter().any(|(_, t)| go(t, seen)),
            Type::Fun(p, _, r) => go(p, seen) || go(r, seen),
            _ => false,
        }
    }
    go(ty, &mut Visited::new())
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
// `rec(idx)` step value? Only decided when `idx` is a known
// positive constant -- a symbolic or zero index keeps the base-case
// treatment, since a Dyn base alternative would also accept the tuple
// there. Non-recursive positions must be consistent with the step
// alternative's; the recursive position is checked against `idx - 1`,
// recursing while it is itself a tuple and ending at a base value (or
// Dyn/Var, which carries no information to contradict).
fn is_step_shaped_tuple(actual: &Type, base_alt: &Type, step_alt: &Type, self_ref: SelfRef, idx: &IndexExpr, infer: &InferCtx) -> bool {
    let (Type::Tuple(items), Type::Tuple(step_items)) = (actual, step_alt) else { return false };
    let Some(k) = const_index(&infer.resolve_index_deep(idx)).filter(|k| *k > 0) else { return false };
    let Some(rec_pos) = self_ref_position(step_alt, self_ref) else { return false };
    if items.len() != step_items.len() {
        return false;
    }
    if !(0..items.len()).all(|i| i == rec_pos || consistent(&items[i], &step_items[i])) {
        return false;
    }
    let tail = infer.resolve_deep(&items[rec_pos]);
    match &tail {
        Type::Dyn | Type::Var(_) => true,
        Type::Tuple(_) if k > 1 => is_step_shaped_tuple(&tail, base_alt, step_alt, self_ref, &IndexExpr::Lit(k - 1), infer),
        _ => k == 1 && !matches!(tail, Type::Indexed(..)) && consistent(&tail, base_alt),
    }
}

// Shared "nothing real to correlate against" fallback: bind each of a
// Pattern::List's own positions to its own independent fresh var.
// Factored out of what used to be three separate, identically-bodied
// arms in bind_pattern_vars's own Pattern::List match (Type::Var, the
// final catch-all, and now the Type::Rec/Type::Union arms'
// own "no matching alternative found" cases below) -- same fallback,
// same reasoning, one copy.
fn bind_fresh_positions(
    ctx: &Ctx,
    pats: &[Pattern],
    infer: &mut InferCtx,
    span: Span,
    visiting: &Visiting,
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
// Type::Indexed/Rec/Union delegation arms below need this same
// fallback for their own "no matching alternative found" cases, not
// just the catch-all.
fn bind_fresh_record_fields(
    ctx: &Ctx,
    fields: &[(String, Pattern)],
    infer: &mut InferCtx,
    span: Span,
    visiting: &Visiting,
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
// -- an accumulated set of recursive types already being unfolded on
// THIS recursion path, threaded unchanged through every existing arm
// below and only ever grown by the Type::Rec arm itself. Without this,
// the Type::Union arm's own `pattern_could_match` probe used a fresh,
// empty set on every call, with no memory of an enclosing Type::Rec
// arm's own unfolding -- letting a Union alternative that is itself a
// bare self-reference (e.g. `let rec List = Union(List, Tuple(Int, List))`, where the
// FIRST alternative is `List` itself) look "reachable" via
// its own fresh probe forever, recursing without end.
fn bind_pattern_vars(
    ctx: &Ctx,
    pat: &Pattern,
    scrutinee_ty: &Type,
    infer: &mut InferCtx,
    span: Span,
    visiting: &Visiting,
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
            // B's own recursive-type analog per §2's generalization) --
            // correlate against the WRAPPED type directly, "forgetting"
            // the index the same way Pattern::Cons's own arm below
            // already does via forget_index, and pattern_could_match's
            // own Type::Indexed arm mirrors exactly. Needed for Case B:
            // a Case-B-eligible scrutinee's own real stored type is
            // `Type::Indexed(Type::Rec(..), n)`, still wrapped, by the
            // time a match arm's own pattern (e.g. the tagged-tuple-style
            // `(h, t)`) gets here -- without this arm it fell into the
            // generic catch-all below, no correlation at all.
            Type::Indexed(wrapped, _) => bind_pattern_vars(ctx, pat, wrapped, infer, span, visiting),
            // A Type::Rec scrutinee (a recursive `let rec X = ... in` type
            // reference) -- unfold ONE level and recurse, mirroring
            // pattern_could_match's own Type::Rec arm exactly. This is
            // Case B's own real prerequisite: without it, a Case-B-style
            // scrutinee's own Type::Rec never even reaches the Type::Union
            // arm just below -- it's still a Rec at this point, not yet
            // unfolded to a Union at all.
            //
            // A cycle guard is needed: the Type::Union arm just below runs
            // its reachability probe with this same `visiting` set, so a
            // Union alternative that is ITSELF a bare self-reference
            // (`let rec List = Union(List, Tuple(Int, List))`) cannot look
            // reachable forever. If the member is already being unfolded on
            // this path, stop and fall back to the safe
            // fresh-vars-per-position behavior instead of unfolding again;
            // otherwise recurse with it added to the accumulated set.
            Type::Rec(g, i) => {
                if visiting.has_rec(g, *i) {
                    return bind_fresh_positions(ctx, pats, infer, span, visiting);
                }
                bind_pattern_vars(ctx, pat, &unfold(g, *i), infer, span, &visiting.with_rec(g, *i))
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
            // This arm's own `pattern_could_match` probe must reuse the SAME
            // accumulated `visiting` set passed into this arm, and the
            // subsequent recursive call into whichever alternative
            // `.find()` picks must also pass that same set through
            // unchanged (this arm never adds to it; only the Type::Rec
            // arm above does). A fresh set would let this arm "forget"
            // that an enclosing Type::Rec arm was already unfolding the
            // member, so a bare self-referential alternative could look
            // reachable again and again forever.
            Type::Union(alts) => match alts.iter().find(|alt| pattern_could_match(pat, alt, visiting)) {
                Some(matching_alt) => bind_pattern_vars(ctx, pat, matching_alt, infer, span, visiting),
                None => bind_fresh_positions(ctx, pats, infer, span, visiting),
            },
            // A concrete, non-Tuple, non-List, non-Indexed/Rec/Union
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
            let list_shape = Type::List(Hc::new(elem_ty.clone()));
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
            // Finding 6 (final review): the same Type::Indexed/Rec/Union
            // delegation Pattern::List already got from Task 3's own Case B
            // prerequisite fix, mirrored here for Record so a Record
            // pattern matched against a Vec(n)-of-records or a Case-B-
            // eligible Rec/Union scrutinee gets real per-field
            // correlation too, instead of always falling back to
            // `infer.fresh_var` per field (the same "zero correlation" gap
            // Task 3 closed for List).
            Type::Indexed(wrapped, _) => bind_pattern_vars(ctx, pat, wrapped, infer, span, visiting),
            Type::Rec(g, i) => {
                if visiting.has_rec(g, *i) {
                    return bind_fresh_record_fields(ctx, fields, infer, span, visiting);
                }
                bind_pattern_vars(ctx, pat, &unfold(g, *i), infer, span, &visiting.with_rec(g, *i))
            }
            Type::Union(alts) => match alts.iter().find(|alt| pattern_could_match(pat, alt, visiting)) {
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
    // fully covers it. `let Pair = Union(Tuple((Int,)), Tuple(Int, Int)) in ... | (n,) ->
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
// A binder's annotation, evaluated (spec 2026-10-10 types-as-values): the
// type the binder is checked against, and the anonymous let the run time
// evaluates before the binder, when one is kept (spec 11.1).
struct Annotation {
    ty: Type,
    site: Option<Site>,
}

// A kept annotation: `let name = Expr::Annot(..)`, placed in the scope the
// checker evaluated the annotation in. The name is unlexable, so nothing in
// the program can capture it. An Unknown site also checks its binder
// (`conform`).
struct Site {
    name: String,
    annot: ExprRef,
    spec: Rc<AnnotSpec>,
    conform: Option<Rc<ConformSpec>>,
}

impl Site {
    fn wrap(&self, arena: &mut Arena, inner: ExprRef) -> ExprRef {
        arena.push(Expr::Let(self.name.clone(), None, self.annot, inner))
    }

    // `operand` checked against the site's type, at an Unknown site.
    fn check(&self, arena: &mut Arena, operand: ExprRef) -> ExprRef {
        match &self.conform {
            Some(spec) => arena.push(Expr::Conform(operand, spec.clone())),
            None => operand,
        }
    }

    // `let name = Conform(name, ..) in body` at an Unknown site, else `body`.
    fn rebind(&self, arena: &mut Arena, name: &str, body: ExprRef) -> ExprRef {
        if self.conform.is_none() {
            return body;
        }
        let var = arena.push(Expr::Var(name.to_string()));
        let checked = self.check(arena, var);
        arena.push(Expr::Let(name.to_string(), None, checked, body))
    }

    // `outer`, the node holding the site, becomes `erased` if
    // settle_erasure erases the site.
    fn record(&self, infer: &mut InferCtx, outer: ExprRef, erased: ExprRef) {
        infer.splices.push(Splice { outer, inner: erased, what: Erasable::Site(self.spec.clone()) });
    }

    fn bind(&self, arena: &mut Arena, infer: &mut InferCtx, inner: ExprRef) -> ExprRef {
        let outer = self.wrap(arena, inner);
        self.record(infer, outer, inner);
        outer
    }
}

// Wraps a let's value `inner` in the site's anonymous let, if there is one,
// checking it against the site's type at an Unknown site.
fn bind_site(arena: &mut Arena, infer: &mut InferCtx, site: &Option<Site>, inner: ExprRef) -> ExprRef {
    match site {
        Some(site) => {
            let checked = site.check(arena, inner);
            site.bind(arena, infer, checked)
        }
        None => inner,
    }
}

// An annotation is an expression: its Type is what it evaluates to in the
// binder's scope `ctx`. One naming a program binder is the value of an
// anonymous type let before the binder (spec 11.1), elaborated as a let's
// right-hand side would be. A Diverged site (the check-time evaluation ran
// out of fuel) and an Unknown site (the annotation names a value only the
// run time has) are always kept, and type the binder Dyn; a Confirm site is
// kept only if running it can fail (rule 10.2), which settle_erasure decides
// once inference has finished (from the definitions the annotation reads
// directly; settle_definitions follows the reads). An error is reported at the annotation; an
// Unknown site's check blames `blame`.
fn annotation_type(arena: &mut Arena, ann: ExprRef, blame: Span, ctx: &Ctx, spans: &SpanMap, infer: &mut InferCtx) -> Result<Annotation, TypeError> {
    if let Expr::TypeLit(t) = &arena[ann] {
        return Ok(Annotation { ty: (**t).clone(), site: None });
    }
    let err = |msg: String| TypeError(msg, spans[ann]);
    let (reads, direct_stuck) = reads_in(arena, &[ann], &[], ctx, infer);
    let stuck = direct_stuck || reads.iter().any(|r| def_deps(arena, infer, r.def).stuck);
    if reads.is_empty() && !stuck {
        // Only prelude constructors: closed and never fallible.
        let ty = pure_type(eval_annotation(arena, ann, &reads, spans, infer)).map_err(err)?;
        return Ok(Annotation { ty: (*ty).clone(), site: None });
    }
    // A nested annotation (a parameter's, inside this one) restores the flag.
    let outer = std::mem::replace(&mut infer.in_annotation, true);
    let elaborated = elaborate(arena, ann, ctx, spans, infer);
    infer.in_annotation = outer;
    let (_, _, elaborated) = elaborated?;
    let outcome = if stuck { None } else { Some(eval_annotation(arena, ann, &reads, spans, infer)) };
    let (ty, kind) = match outcome {
        None => (Type::Dyn, AnnotKind::Unknown),
        Some(Pure::Stuck(Stuck::Fuel)) => (Type::Dyn, AnnotKind::Diverged),
        Some(outcome) => {
            let ty = pure_type(outcome).map_err(err)?;
            ((*ty).clone(), AnnotKind::Confirm(ty))
        }
    };
    let site_name = fresh_index_name("__rt");
    let conform = matches!(kind, AnnotKind::Unknown).then(|| {
        let names = written_index_names(arena, infer, ann, &reads);
        conform_spec(arena, &site_name, &names, blame, ctx, infer)
    });
    let spec = Rc::new(AnnotSpec::new(kind, spans[ann]));
    let read_defs: Vec<usize> = reads.iter().map(|r| r.def).collect();
    if matches!(spec.kind, AnnotKind::Confirm(_)) {
        infer.proofs.push(Proof::Site(read_defs, elaborated, spec.clone()));
    } else {
        // The run time evaluates the annotation, so it keeps every binding it reads.
        read_defs.iter().for_each(|i| infer.defs[*i].kept.set(true));
    }
    let annot = arena.push(Expr::Annot(elaborated, spec.clone()));
    Ok(Annotation { ty, site: Some(Site { name: site_name, annot, spec, conform }) })
}

// An Unknown site's check (spec 11.4). The index variables its run-time type
// can mention are `names`, those written in the annotation and in the
// definitions it reads (written_index_names): an index term is not a value, so
// it reaches a type value only from a TypeLit. Each gets a length slot read in
// the binder's scope, as a Check's would be; one nothing here binds is the
// annotation's own, and matches any length. Unlike the binder that introduces
// it statically, repeated occurrences do not share it (parked, spec 11.13). The
// site's contract checks read the same table.
fn conform_spec(arena: &mut Arena, site: &str, names: &BTreeSet<String>, blame: Span, ctx: &Ctx, infer: &InferCtx) -> Rc<ConformSpec> {
    let names: Vec<String> = names.iter().cloned().collect();
    // The table is read by the site's contract calls too, so its obligations
    // are repeatable.
    let env = CheckEnv { repeatable: true, ..CheckEnv::at(infer, blame) };
    let mut lens = Vec::with_capacity(names.len());
    for name in &names {
        // A name this scope introduced denotes its variable (if that has a value to read); any
        // other is the annotation's own, and matches any length.
        let bound = match ctx.annot_var(&TyVar::Index(name.clone())) {
            Some(AnnotVar { var: TyVar::Index(var), .. })
                if infer.index_witness.iter().any(|w| w.var == *var) || infer.index_subst.contains_key(var) || infer.rigid_index.contains(var) =>
            {
                Some(IndexExpr::Var(var.clone()))
            }
            _ => None,
        };
        let term = match &bound {
            Some(index) => index_len_term(arena, index, &env),
            None => LenExpr::Any,
        };
        let slot = arena.push(Expr::Len(term));
        if let Some(index) = &bound {
            record_obligation(slot, index, &env, true);
        }
        lens.push(slot);
    }
    let ty = arena.push(Expr::Var(site.to_string()));
    Rc::new(ConformSpec { source: TypeSource::Site(ty, RefCell::new(None)), index_names: names, lens, span: Some(blame), trusted: false })
}

// A check-time evaluation that finished, replayed by settle_erasure. Each holds
// the definitions the evaluation read directly (indices); the ones they read
// are reached by the sweep.
enum Proof {
    // A Confirm site: the definitions it read, its elaborated annotation, and
    // its node.
    Site(Vec<usize>, ExprRef, Rc<AnnotSpec>),
}

// A node erasure may remove: `outer` becomes a copy of `inner` when `what`
// is erased.
struct Splice {
    outer: ExprRef,
    inner: ExprRef,
    what: Erasable,
}

enum Erasable {
    Site(Rc<AnnotSpec>),
    Binder(Rc<CheckDef>),
}

// Decides every erasure once inference has finished: until resolve_pendings
// has run, a cast slot counts as fallible, and nearly every type function's
// body holds one (`a` in `fun a -> List(a)` goes to a Dyn parameter). The
// evaluations replay in order, so the sites inside a definition are decided
// before an annotation that reads it. A definition's flags then follow the
// reads, once, from the newest definition to the oldest (settle_definitions).
// Then each erased node is overwritten by its replacement, in ascending order:
// a node's children come before it in the arena, so a replacement is copied
// only after its own removals.
fn settle_erasure(arena: &mut Arena, spans: &SpanMap, infer: &mut InferCtx) {
    for proof in std::mem::take(&mut infer.proofs) {
        match proof {
            Proof::Site(reads, elaborated, spec) => {
                let clean = note_evaluation(arena, spans, infer, &reads);
                if clean && !fallible_check(arena, elaborated, spans, infer) {
                    spec.erased.set(true);
                } else {
                    // The run time evaluates the annotation, so it keeps every binding it reads.
                    reads.iter().for_each(|i| infer.defs[*i].kept.set(true));
                }
            }
        }
    }
    settle_definitions(arena, infer);
    let mut splices = std::mem::take(&mut infer.splices);
    splices.retain(|s| match &s.what {
        Erasable::Site(spec) => spec.erased.get(),
        Erasable::Binder(def) => def.erasable(),
    });
    splices.sort_by_key(|s| cranelift_entity::EntityRef::index(s.outer));
    for s in splices {
        arena[s.outer] = arena[s.inner].clone();
    }
}

// An evaluation that finished read `reads` directly, and through them every
// definition they reach (its closure). The evaluation ran their parsed values,
// without the checks elaboration added, so a definition is proven only if its
// run-time binding cannot fail: it is inert, or nothing in the closure holds a
// check that could fail (the binding may call any of it). Marks the definitions
// it read as reached, and clean ones as reached clean; settle_definitions
// carries both along the reads. Returns whether the closure is clean.
fn note_evaluation(arena: &Arena, spans: &SpanMap, infer: &InferCtx, reads: &[usize]) -> bool {
    let clean = !reads.iter().any(|i| trans_fallible(arena, spans, infer, *i));
    for i in reads {
        let def = &infer.defs[*i];
        def.reached.set(true);
        if clean {
            def.reached_clean.set(true);
        }
    }
    clean
}

// Some definition in the closure of `i` (its own included) holds a run-time
// check that could fail, memoized. Iterative, oldest definition first, like
// def_deps. Each `fallible` is first computed by a proof whose closure
// contains it, and the sites inside a definition are decided before any proof
// that reads it.
fn trans_fallible(arena: &Arena, spans: &SpanMap, infer: &InferCtx, i: usize) -> bool {
    if infer.defs[i].trans_fallible.get().is_none() {
        let mut todo: Vec<usize> = Vec::new();
        let mut queued: HashSet<usize> = HashSet::new();
        let mut work = vec![i];
        while let Some(j) = work.pop() {
            if infer.defs[j].trans_fallible.get().is_some() || !queued.insert(j) {
                continue;
            }
            work.extend(def_deps(arena, infer, j).reads.iter().map(|r| r.def));
            todo.push(j);
        }
        todo.sort_unstable();
        for j in todo {
            let def = &infer.defs[j];
            let fallible = def.fallible(arena, spans, infer) || def_deps(arena, infer, j).reads.iter().any(|r| infer.defs[r.def].trans_fallible.get() == Some(&true));
            let _ = def.trans_fallible.set(fallible);
        }
    }
    *infer.defs[i].trans_fallible.get().expect("computed above")
}

// Carries the flags note_evaluation and the kept sites set along the reads,
// from the newest definition to the oldest (a definition reads only older
// ones), and settles each: `proven` if some finished evaluation whose closure
// has it was clean, or it is inert and some finished evaluation has it;
// `used` also if a kept annotation's closure has it.
fn settle_definitions(arena: &Arena, infer: &InferCtx) {
    for i in (0..infer.defs.len()).rev() {
        let def = &infer.defs[i];
        if !(def.reached.get() || def.kept.get()) {
            continue;
        }
        if def.reached_clean.get() || (def.reached.get() && def.inert(arena)) {
            def.proven.set(true);
        }
        if def.kept.get() {
            def.used.set(true);
        }
        for read in &def_deps(arena, infer, i).reads {
            let dep = &infer.defs[read.def];
            dep.reached.set(dep.reached.get() || def.reached.get());
            dep.reached_clean.set(dep.reached_clean.get() || def.reached_clean.get());
            dep.kept.set(dep.kept.get() || def.kept.get());
        }
    }
}

// The index variables written in `ann` and in every definition it reaches
// through `reads`: what a TypeLit in any of them mentions.
fn written_index_names(arena: &Arena, infer: &InferCtx, ann: ExprRef, reads: &[Read]) -> Rc<BTreeSet<String>> {
    let mut names = BTreeSet::new();
    collect_index_names(arena, &[ann], &mut names);
    let sets: Vec<Rc<BTreeSet<String>>> = reads.iter().map(|r| def_index_names(arena, infer, r.def)).collect();
    union_names(names, sets)
}

// The names of `own` and every set in `sets`, reusing the largest set's `Rc`
// when the rest add nothing to it: a definition that writes no index variable
// of its own shares its reads'.
fn union_names(own: BTreeSet<String>, mut sets: Vec<Rc<BTreeSet<String>>>) -> Rc<BTreeSet<String>> {
    sets.sort_by_key(|s| std::cmp::Reverse(s.len()));
    let mut sets = sets.into_iter();
    let mut acc = sets.next().unwrap_or_default();
    for set in sets {
        if !set.is_subset(&acc) {
            Rc::make_mut(&mut acc).extend(set.iter().cloned());
        }
    }
    if !own.is_subset(&acc) {
        Rc::make_mut(&mut acc).extend(own);
    }
    acc
}

// What index variables the TypeLits under `roots` mention.
fn collect_index_names(arena: &Arena, roots: &[ExprRef], names: &mut BTreeSet<String>) {
    let (mut work, mut kids) = (roots.to_vec(), Vec::new());
    while let Some(n) = work.pop() {
        if let Expr::TypeLit(t) = &arena[n] {
            names.extend(index_names(vars_in(t)));
        }
        kids.clear();
        arena[n].children(&mut kids);
        work.extend(&kids);
    }
}

// The index variables written in definition `i` and in the ones it reaches,
// memoized. Iterative, oldest first, like def_deps.
fn def_index_names(arena: &Arena, infer: &InferCtx, i: usize) -> Rc<BTreeSet<String>> {
    if infer.defs[i].index_names.get().is_none() {
        let mut todo: Vec<usize> = Vec::new();
        let mut queued: HashSet<usize> = HashSet::new();
        let mut work = vec![i];
        while let Some(j) = work.pop() {
            if infer.defs[j].index_names.get().is_some() || !queued.insert(j) {
                continue;
            }
            work.extend(def_deps(arena, infer, j).reads.iter().map(|r| r.def));
            todo.push(j);
        }
        todo.sort_unstable();
        for j in todo {
            let mut own = BTreeSet::new();
            collect_index_names(arena, &infer.defs[j].sources(), &mut own);
            let sets = def_deps(arena, infer, j).reads.iter().map(|r| infer.defs[r.def].index_names.get().expect("a dependency has a smaller index").clone()).collect();
            let _ = infer.defs[j].index_names.set(union_names(own, sets));
        }
    }
    infer.defs[i].index_names.get().expect("computed above").clone()
}

// The definitions `roots` (resolved in `ctx`, with `skip` bound around them)
// read directly, one `Read` per name in first-use order, and whether one of
// the names stands for a value that exists only at run time.
fn reads_in(arena: &Arena, roots: &[ExprRef], skip: &[String], ctx: &Ctx, infer: &InferCtx) -> (Vec<Read>, bool) {
    let mut reads: Vec<Read> = Vec::new();
    let mut stuck = false;
    let mut seen: HashSet<String> = HashSet::new();
    for root in roots {
        for name in crate::resolve::free_vars(arena, *root) {
            if skip.contains(&name) || !infer.bound_names.contains(&name) || !seen.insert(name.clone()) {
                continue;
            }
            let Some(scheme) = ctx.get(&name) else { continue };
            let Some(def) = &scheme.def else {
                stuck = true;
                continue;
            };
            let member = match &def.node {
                DefNode::Let(..) => 0,
                DefNode::Group(bindings) => bindings.iter().rposition(|(n, _, _)| *n == name).unwrap_or(0),
            };
            reads.push(Read { name, def: def.idx, member });
        }
    }
    (reads, stuck)
}

// What definition `i` reads directly, computed once, along with that of every
// definition it reaches that has not been yet. Iterative: a chain of
// dependencies reaches tens of thousands. A dependency has a smaller index, so
// stuckness (transitive) is settled in ascending order.
fn def_deps<'a>(arena: &Arena, infer: &'a InferCtx, i: usize) -> &'a Deps {
    if infer.defs[i].deps.get().is_none() {
        let mut fresh: Vec<(usize, Vec<Read>, bool)> = Vec::new();
        let mut queued: HashSet<usize> = HashSet::new();
        let mut work = vec![i];
        while let Some(j) = work.pop() {
            if infer.defs[j].deps.get().is_some() || !queued.insert(j) {
                continue;
            }
            let (reads, stuck) = infer.defs[j].direct_reads(arena, infer);
            work.extend(reads.iter().map(|r| r.def));
            fresh.push((j, reads, stuck));
        }
        fresh.sort_unstable_by_key(|(j, _, _)| *j);
        for (j, reads, direct) in fresh {
            let stuck = direct || reads.iter().any(|r| infer.defs[r.def].deps.get().expect("a dependency has a smaller index").stuck);
            let _ = infer.defs[j].deps.set(Deps { reads, stuck });
        }
    }
    infer.defs[i].deps.get().expect("computed above")
}

// Evaluates, once each and in creation order, the definitions `roots` name
// that have no value yet, and every definition they reach. None of them is
// stuck (the caller checked), so each reads only the values of definitions
// with a smaller index. Iterative: a chain of definitions reaches tens of
// thousands.
fn ensure_values(arena: &mut Arena, spans: &SpanMap, infer: &mut InferCtx, roots: &[usize]) {
    let mut todo: Vec<usize> = Vec::new();
    let mut queued: HashSet<usize> = HashSet::new();
    let mut work: Vec<usize> = roots.to_vec();
    while let Some(i) = work.pop() {
        if infer.defs[i].value.get().is_some() || !queued.insert(i) {
            continue;
        }
        work.extend(def_deps(arena, infer, i).reads.iter().map(|r| r.def));
        todo.push(i);
    }
    todo.sort_unstable();
    for i in todo {
        let value = def_value(arena, spans, infer, i);
        let _ = infer.defs[i].value.set(value);
    }
}

// Member `member` of definition `def`'s value: a group's value is the list of
// its members', a let's is the value itself.
fn member_value(def: &CheckDef, value: &Value, member: usize) -> Value {
    match (&def.node, value) {
        (DefNode::Group(_), Value::List(members)) => members[member].clone(),
        _ => value.clone(),
    }
}

// The value running parsed node `n` would give, if some finished definition
// holds it: `n` is that definition's right-hand side (or a group member's
// value) and the definition is Done. Running `n` inline is deterministic and
// a Done definition is not stuck, so it reads only definitions, and every name
// in it means the same binder here as in the checker's scope: the values agree
// (eval_pure_in's debug build checks it). Nothing is evaluated for this; a
// definition that failed or has no value yet is run inline, as before.
fn reusable(defs: &[Rc<CheckDef>], value_of: &FxMap<ExprRef, Option<(usize, usize)>>, n: ExprRef) -> Option<Value> {
    let (d, member) = (*value_of.get(&n)?)?;
    match defs[d].value.get() {
        Some(DefValue::Done(value)) => Some(member_value(&defs[d], value, member)),
        _ => None,
    }
}

// The names `reads` stand for, with the values the definitions they read have,
// as the frame an evaluation runs in; or, if any of those definitions failed,
// the failure of the earliest one.
fn read_frame(infer: &InferCtx, reads: &[Read]) -> Result<(Vec<String>, Vec<Value>), (usize, Failure)> {
    let mut names = Vec::with_capacity(reads.len());
    let mut values = Vec::with_capacity(reads.len());
    let mut failed: Option<(usize, Failure)> = None;
    for read in reads {
        match infer.defs[read.def].value.get().expect("a dependency is evaluated first") {
            DefValue::Done(value) => {
                names.push(read.name.clone());
                values.push(member_value(&infer.defs[read.def], value, read.member));
            }
            DefValue::Failed { origin, why } if failed.as_ref().is_none_or(|(o, _)| origin < o) => failed = Some((*origin, why.clone())),
            DefValue::Failed { .. } => {}
        }
    }
    failed.map_or(Ok((names, values)), Err)
}

// Definition `i`'s value: its right-hand side evaluated in a frame of the
// values of the definitions it reads, with a step budget of its own. A group's
// definition builds its `let rec (bindings) in (names)` root here; it runs once
// per definition, and only a group of direct `fun`s is two definitions.
fn def_value(arena: &mut Arena, spans: &SpanMap, infer: &mut InferCtx, i: usize) -> DefValue {
    #[cfg(test)]
    DEF_EVALS.with(|c| c.set(c.get() + 1));
    let def = infer.defs[i].clone();
    let reads = def_deps(arena, infer, i).reads.clone();
    let (names, values) = match read_frame(infer, &reads) {
        Ok(frame) => frame,
        Err((origin, why)) => return DefValue::Failed { origin, why },
    };
    let root = match &def.node {
        DefNode::Let(rhs) => *rhs,
        DefNode::Group(bindings) => {
            let members: Vec<ExprRef> = bindings.iter().map(|(n, _, _)| arena.push(Expr::Var(n.clone()))).collect();
            let tuple = arena.push(Expr::Tuple(members));
            arena.push(Expr::LetRec(bindings.clone(), tuple))
        }
    };
    match eval_pure_in(arena, root, &names, values, |n| reusable(&infer.defs, &infer.value_of, n), &mut infer.resolved, spans, ANNOTATION_FUEL) {
        Pure::Done(value) => DefValue::Done(value),
        Pure::Stuck(why) => DefValue::Failed { origin: i, why: Failure::Stuck(why) },
        Pure::Error(msg) => DefValue::Failed { origin: i, why: Failure::Error(msg) },
    }
}

#[cfg(test)]
thread_local! {
    // How many definitions def_value has evaluated on this thread.
    pub(crate) static DEF_EVALS: Cell<usize> = const { Cell::new(0) };
}

// What `ann`, which reads `reads` directly, evaluates to at check time: the
// failure of the earliest definition it reaches that failed, else the
// annotation itself, run in a frame of the values it reads.
fn eval_annotation(arena: &mut Arena, ann: ExprRef, reads: &[Read], spans: &SpanMap, infer: &mut InferCtx) -> Pure {
    let roots: Vec<usize> = reads.iter().map(|r| r.def).collect();
    ensure_values(arena, spans, infer, &roots);
    match read_frame(infer, reads) {
        Ok((names, values)) => eval_pure_in(arena, ann, &names, values, |n| reusable(&infer.defs, &infer.value_of, n), &mut infer.resolved, spans, ANNOTATION_FUEL),
        Err((_, why)) => why.into_pure(),
    }
}

// The annotation slot typecheck re-emits for a binder whose type it knows.
fn type_lit(arena: &mut Arena, ty: Type) -> ExprRef {
    arena.push(Expr::TypeLit(Rc::new(ty)))
}

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
                // The variables a non-function annotation introduced: existential, so in scope
                // (and open) for the value and the rest of the scope. A signature's are the
                // value's alone: they are generalized at this let.
                let mut existentials = Vec::new();
                let (bound_ty, val_row, val3) = match ann {
                    // `check_against` folds today's coerce-then-unify_fits
                    // pair into one call, AND -- since val may itself be a
                    // Lambda/If/Match -- lets the declared annotation `t`
                    // actually flow into val's own tail position instead of
                    // only being checked against val's independently
                    // inferred type after the fact. This is Phase 3's real
                    // entry point: `let f: A -> Vec(n) = fn(x) = <body>`
                    // now checks <body> against Vec(n) directly.
                    // A kept annotation is evaluated just before the value, in
                    // the value's scope.
                    Some(ann) => {
                        let annotation = annotation_type(arena, ann, spans[val], &cur_ctx, spans, infer)?;
                        let universal = is_signature(&annotation.ty);
                        let mut introduced = Vec::new();
                        let t = open_annotation(&annotation.ty, &cur_ctx, universal, &mut introduced);
                        let (val_row, val4) = check_against_rigid(arena, val, &t, &cur_ctx.with_annot(&introduced), spans, infer)?;
                        if !universal {
                            existentials = introduced;
                        }
                        (t, val_row, bind_site(arena, infer, &annotation.site, val4))
                    }
                    None => {
                        let (val_ty, val_row, val2) = elaborate(arena, val, &cur_ctx, spans, infer)?;
                        (val_ty, val_row, val2)
                    }
                };
                let def = infer.new_def(DefNode::Let(val), vec![val3], &cur_ctx);
                infer.note_value_of(val, def.idx, 0);
                cur_ctx = extend_generalized(&cur_ctx.with_annot(&existentials), &var, bound_ty.clone(), infer, Some(def.clone()));
                pending.push(PendingElab::Let { var, bound_ty, val_row, val: val3, def });
                cur_expr = body;
            }
            // In a recursive group (a group of direct `fun`s), every
            // binding needs to resolve to its own (eventual) type
            // WHILE elaborating every value in the group (not just its
            // own), so that a reference to any sibling -- including
            // itself -- type-checks precisely rather than falling back to
            // Dyn. That's only possible where a type is already known,
            // i.e. annotated -- an unannotated binding still works (that
            // one reference is just a fresh type variable, like any other unbound-at-this-
            // point lookup), it's just not precisely typed. All values
            // are elaborated against this SAME pre-bound context (not a
            // progressively-updated one): they're simultaneous, not
            // sequential, so f seeing g's real inferred type (rather than
            // just g's annotation-or-Dyn) would wrongly depend on
            // which one happens to come first in the `and` chain.
            Expr::LetRec(bindings, body) => {
                // The values are checked in the scope the run time evaluates
                // them in (machine.rs's LetRec): a group of direct `fun`s sees
                // its own names, bound to its closures (a definition of their
                // own, so an annotation naming one evaluates it); a type group
                // (rule P) sees them as placeholders for the types it is
                // building, which no annotation can name (stuck, then an error
                // at the annotation); any other group's values see the outer
                // scope only.
                let kind = group_kind(arena, &bindings);
                let group = (kind == GroupKind::Functions).then(|| {
                    let def = infer.new_def(DefNode::Group(bindings.clone()), Vec::new(), &cur_ctx);
                    // Its values are not elaborated yet: count them fallible,
                    // so no evaluation that reads it counts as clean
                    // (settle_definitions).
                    let _ = def.fallible.set(true);
                    def
                });
                let mut val_ctx = cur_ctx.clone();
                // Each annotation opened once, against the outer scope and
                // into one `introduced` vector (a name two members write is
                // one variable, unless an outer existential of that name is in
                // scope: a function member then shadows it with a fresh one
                // while a non-function member shares the outer one, so
                // `let rec f: (Vec(n) -> Int) = .. and k: Vec(n) = [1, 2]`
                // differs with and without `let a: Vec(n) = [1, 2] in` in
                // front; written by both a function and a non-function
                // member it is existential), so the pre-binding below and the
                // value check share its variables. A kept one is evaluated
                // before the group, in the scope its annotation was
                // evaluated in.
                let mut introduced = Vec::new();
                let mut opened: Vec<Option<Type>> = Vec::with_capacity(bindings.len());
                let mut sites = Vec::new();
                for (name, ann, val) in bindings.iter() {
                    opened.push(match ann {
                        Some(ann) => {
                            let annotation = annotation_type(arena, *ann, spans[*val], &cur_ctx, spans, infer)?;
                            sites.extend(annotation.site.map(|site| (name.clone(), site)));
                            Some(open_annotation(&annotation.ty, &cur_ctx, is_signature(&annotation.ty), &mut introduced))
                        }
                        None => None,
                    });
                }
                for ((name, _, _), ann) in bindings.iter().zip(&opened) {
                    // A plain group binds nothing for its values.
                    if kind == GroupKind::Plain {
                        break;
                    }
                    // A type group has no annotations: its names are Dyn, and stuck
                    // (no definition) for an annotation inside a value.
                    let Some(group) = &group else {
                        val_ctx = val_ctx.bind(name, Scheme::mono(Type::Dyn));
                        continue;
                    };
                    val_ctx = match ann {
                        // Generalized, not plain extend: a fully-given
                        // annotation makes checking mode against it
                        // sound even for a recursive self/sibling
                        // reference (Mycroft's polymorphic-recursion
                        // result; design spec 2026-09-20 sec 2) --
                        // gated on an explicit annotation, not new
                        // inference, so this is deliberately narrower
                        // than general HM polymorphic recursion.
                        Some(ty) => extend_generalized(&val_ctx, name, ty.clone(), infer, Some(group.clone())),
                        None => val_ctx.bind(name, Scheme { def: Some(group.clone()), ..Scheme::mono(infer.fresh_var(name)) }),
                    };
                }
                // The pre-bindings above are built first: the group's own universals are
                // not yet open there, so they are quantified and a recursive reference
                // instantiates them afresh (polymorphic recursion).
                val_ctx = val_ctx.with_annot(&introduced);
                let mut elaborated = Vec::with_capacity(bindings.len());
                for ((name, _, val), ann) in bindings.iter().zip(opened) {
                    // Same check_against restructuring as Expr::Let just
                    // above -- see its own comment. `cur_mode` is likewise
                    // untouched: each binding's annotation is its own
                    // self-contained expected type.
                    if group.is_some() {
                        infer.direct_value = Some(*val);
                    }
                    let (bound_ty, val_row, val3) = match ann {
                        Some(t) => {
                            let (val_row, val4) = check_against_rigid(arena, *val, &t, &val_ctx, spans, infer)?;
                            (t, val_row, val4)
                        }
                        None => {
                            let (val_ty, val_row, val2) = elaborate(arena, *val, &val_ctx, spans, infer)?;
                            (val_ty, val_row, val2)
                        }
                    };
                    elaborated.push((name.clone(), bound_ty, val_row, val3));
                }
                let values = elaborated.iter().map(|(_, _, _, v)| *v).collect();
                let def = infer.new_def(DefNode::Group(bindings.clone()), values, &cur_ctx);
                // Not the early definition of a direct-`fun` group: it is made
                // before its values exist, and is the group as they see it.
                for (m, (_, _, val)) in bindings.iter().enumerate() {
                    infer.note_value_of(*val, def.idx, m);
                }
                // The group's existential variables stay in scope, and open, after it.
                introduced.retain(|(_, entry)| !entry.universal);
                cur_ctx = cur_ctx.with_annot(&introduced);
                for (name, bound_ty, _, _) in elaborated.iter() {
                    cur_ctx = extend_generalized(&cur_ctx, name, bound_ty.clone(), infer, Some(def.clone()));
                }
                pending.push(PendingElab::LetRec { bindings: elaborated, def, sites, kind });
                cur_expr = body;
            }
            Expr::Lambda(param, ann, body) => {
                // Taken before the annotation is elaborated, which may elaborate lambdas of its own.
                let direct = infer.direct_value.take() == Some(cur_expr);
                let mut site = None;
                // A parameter annotation's variables are universal, and in scope (and open)
                // for the body.
                let mut introduced = Vec::new();
                let ann = match ann {
                    Some(ann) => {
                        let annotation = annotation_type(arena, ann, spans[ann], &cur_ctx, spans, infer)?;
                        site = annotation.site;
                        Some(open_annotation(&annotation.ty, &cur_ctx, true, &mut introduced))
                    }
                    None => None,
                };
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
                cur_ctx = extend(&cur_ctx.with_annot(&introduced), &param, param_ty.clone());
                let witness = list_length_var(&param_ty, infer).map(|var| IndexWitness::new(var, &param));
                let has_witness = witness.is_some();
                infer.index_witness.extend(witness);
                infer.repeatable_depth += 1;
                pending.push(PendingElab::Fun { param, param_ty, witness: has_witness, site, direct });
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
            Some((base_alt, step_alt, self_ref, idx)) => {
                let (actual_ty, row, expr2) = elaborate(arena, cur_expr, &cur_ctx, spans, infer)?;
                // DEVIATION from the task-6 brief's literal text: the
                // brief's own condition was bare `consistent(&actual_ty,
                // &base_alt)`. `consistent` treats Dyn/Var as
                // universally consistent with anything (types.rs's own
                // doc comment) -- exactly the "no real information here"
                // signal this codebase treats specially everywhere else
                // (pattern_could_match, shape_test, etc.), not
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
                // be Dyn/Var (e.g. `let rec T = Union(Dyn, Tuple(Int, T))`) makes
                // `consistent(actual_ty, base_alt)` trivially true for ANY
                // `actual_ty`, including a genuinely step-shaped one (a
                // `w: T(2) = v` reference to an already-Indexed(Rec(T),
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
                // an Indexed(Rec(..), _) type otherwise (consistent()
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
                // non-Dyn/Var base_alt (e.g. `let rec T = Union(Vec(3), Tuple(Int, T))`)
                // still gets a chance here, since the generic fallback's
                // `consistent` has no Indexed-vs-Rec arm to catch that
                // case on its own.
                //
                // Confirmed by direct repro: `w: T(2) = v` (actual_ty =
                // Indexed(Rec(T), 2), base_alt = Dyn) now falls through
                // to the trailing unify_fits, which already has full,
                // correct Indexed-vs-Indexed handling via unify() -- while
                // `let x: T(0) = 5 in x` and the deeper `let v: T(2) =
                // (1, (2, 5))` (whose innermost `5` is checked against a
                // still-symbolic `T(2 - 1 - 1)`) both keep working, since
                // Int is never Type::Indexed; and `let rec T = Union(Vec(3),
                // Tuple(Int, T)) in let b: T(0) = [1,2,3] in ...` (actual_ty =
                // Indexed(List,3), base_alt = Indexed(List,3), not
                // Dyn/Var) still unifies index 0 correctly.
                let actual_is_indexed = matches!(actual_ty, Type::Indexed(..));
                let base_is_dyn_or_var = matches!(base_alt, Type::Dyn | Type::Var(_));
                if is_step_shaped_tuple(&actual_ty, &base_alt, &step_alt, self_ref, &idx, infer) {
                    // Already-synthesized step-shaped Tuple (see the helper).
                    (expected.clone(), row, expr2)
                } else if !matches!(actual_ty, Type::Dyn | Type::Var(_))
                    && !(actual_is_indexed && base_is_dyn_or_var)
                    && consistent(&actual_ty, &base_alt)
                {
                    unify_index(&idx, &IndexExpr::Lit(0), infer, spans[cur_expr])?;
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
            PendingElab::Let { var, bound_ty, val_row, val, def } => {
                // Matches the original Let arm: body's type propagates
                // through unchanged, row is the union of val's and body's.
                result_row = EffectRow::union(&val_row, &result_row);
                // A type let only annotations read is dead at run time
                // (settle_erasure).
                let ann = type_lit(arena, bound_ty);
                let body = result_expr;
                result_expr = arena.push(Expr::Let(var, Some(ann), val, body));
                infer.splices.push(Splice { outer: result_expr, inner: body, what: Erasable::Binder(def) });
            }
            PendingElab::LetRec { bindings, def, sites, kind } => {
                // Matches the original LetRec arm: body's type propagates
                // through unchanged, row is the union of every binding's
                // plus body's.
                for (_, _, val_row, _) in &bindings {
                    result_row = EffectRow::union(val_row, &result_row);
                }
                // A type group stays annotation-free: rule P (resolve::group_kind) reads
                // the elaborated tree too, and a TypeLit annotation would make it Plain.
                let group: Vec<(String, Option<ExprRef>, ExprRef)> = bindings
                    .into_iter()
                    .map(|(name, ty, _, val)| (name, (kind != GroupKind::Types).then(|| type_lit(arena, ty)), val))
                    .collect();
                debug_assert_eq!(group_kind(arena, &group), kind, "elaboration changed what the `let rec` builds");
                // An Unknown member is checked once the group is built, and
                // that check reads it.
                let mut body = result_expr;
                for (name, site) in sites.iter().rev() {
                    body = site.rebind(arena, name, body);
                }
                if sites.iter().any(|(_, site)| site.conform.is_some()) {
                    def.used.set(true);
                }
                result_expr = arena.push(Expr::LetRec(Rc::new(group), body));
                infer.splices.push(Splice { outer: result_expr, inner: body, what: Erasable::Binder(def) });
                for (_, site) in sites.iter().rev() {
                    result_expr = site.bind(arena, infer, result_expr);
                }
            }
            PendingElab::Fun { param, param_ty, witness, site, direct } => {
                infer.repeatable_depth -= 1;
                // Matches the original Lambda arm: the body's row is
                // embedded in the Fun type, not propagated -- evaluating
                // the Lambda expression itself is always pure.
                result_ty = Type::Fun(Hc::new(param_ty.clone()), result_row, Hc::new(result_ty));
                result_row = EffectRow::pure();
                if witness {
                    let w = infer.index_witness.pop().expect("a Fun frame's witness is the innermost one");
                    if w.used.get() {
                        result_expr = w.bind(arena, &param, result_expr);
                    }
                }
                let ann = type_lit(arena, param_ty);
                result_expr = match site {
                    // A kept annotation is evaluated when the closure is
                    // created, where the parameter is not bound yet.
                    // An Unknown one checks the argument on each call.
                    Some(site) if !direct => {
                        let body = site.rebind(arena, &param, result_expr);
                        let lambda = arena.push(Expr::Lambda(param, Some(ann), body));
                        site.bind(arena, infer, lambda)
                    }
                    // A recursive group's value must stay a Lambda: the
                    // annotation is evaluated on each call instead, under
                    // a fresh parameter name, so it still cannot see the
                    // parameter. Erased, it is the plain Lambda.
                    Some(site) => {
                        let plain = arena.push(Expr::Lambda(param.clone(), Some(ann), result_expr));
                        let arg = fresh_index_name("__ra");
                        let arg_ref = arena.push(Expr::Var(arg.clone()));
                        let checked = site.check(arena, arg_ref);
                        let body = arena.push(Expr::Let(param, None, checked, result_expr));
                        let body = site.wrap(arena, body);
                        let lambda = arena.push(Expr::Lambda(arg, Some(ann), body));
                        site.record(infer, lambda, plain);
                        lambda
                    }
                    None => arena.push(Expr::Lambda(param, Some(ann), result_expr)),
                };
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
// index variables (already the fresh ones open_annotation minted at the
// binder) are rigid for the duration, so the body cannot pin them (e.g.
// `n := 3`) by unification. Only variables THIS call newly marked are
// released afterwards, so nested annotated bindings compose. Non-function
// annotations (`let x: Vec(n) = [1, 2]`) keep their flexible, binding-style
// variables: there `n` is an existential to be inferred from the value, not
// a signature parameter. See spec 2026-10-07.
fn check_against_rigid(arena: &mut Arena, val: ExprRef, ann: &Type, ctx: &Ctx, spans: &SpanMap, infer: &mut InferCtx) -> Result<(EffectRow, ExprRef), TypeError> {
    let signature_vars = if is_signature(ann) { index_names(vars_in(&infer.resolve_deep(ann))) } else { BTreeSet::new() };
    let added: Vec<String> = signature_vars.into_iter().filter(|v| infer.rigid_index.insert(v.clone())).collect();
    let result = check_against(arena, val, ann, ctx, spans, infer);
    for v in &added {
        infer.rigid_index.remove(v);
    }
    result
}

// A function annotation is a signature: its variables are its own parameters (universal).
// Any other annotation's variables are existentials the checked value fixes.
pub(crate) fn is_signature(ty: &Type) -> bool {
    matches!(ty, Type::Fun(..))
}

// Introduce the row and index variables a written annotation `ty` names, at its binder. Each
// written variable shares the entry already in scope or mints a fresh `name#k` (recorded in
// `introduced`, for Ctx::with_annot); `ty` is returned with the fresh names substituted.
//   - A universal binder shares an in-scope universal entry and shadows an in-scope existential.
//   - An existential binder shares any in-scope entry.
//   - A name with no entry in scope (or, for a group of annotations opened against one scope,
//     already in `introduced`) is introduced here; sharing an introduced entry through an
//     existential binder makes it existential.
pub(crate) fn open_annotation(ty: &Type, ctx: &Ctx, universal: bool, introduced: &mut Vec<(TyVar, AnnotVar)>) -> Type {
    let mut rows: HashMap<String, EffectRow> = HashMap::new();
    let mut indices: HashMap<String, IndexExpr> = HashMap::new();
    for written in vars_in(ty) {
        if matches!(written, TyVar::Type(_)) {
            continue;
        }
        let var = match ctx.annot_var(&written) {
            Some(entry) if entry.universal || !universal => entry.var.clone(),
            _ => match introduced.iter_mut().find(|(w, _)| *w == written) {
                Some((_, entry)) => {
                    entry.universal &= universal;
                    entry.var.clone()
                }
                None => {
                    let var = match &written {
                        TyVar::Row(name) => TyVar::Row(fresh_row_name(name)),
                        TyVar::Index(name) => TyVar::Index(fresh_index_name(name)),
                        TyVar::Type(_) => unreachable!("type variables are skipped above"),
                    };
                    introduced.push((written.clone(), AnnotVar { var: var.clone(), universal }));
                    var
                }
            },
        };
        match (written, var) {
            (TyVar::Row(w), TyVar::Row(f)) => {
                rows.insert(w, EffectRow::Var(f));
            }
            (TyVar::Index(w), TyVar::Index(f)) => {
                indices.insert(w, IndexExpr::Var(f));
            }
            _ => unreachable!("an annotation variable keeps its sort"),
        }
    }
    subst_type(ty, &rows, &HashMap::new(), &indices)
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
        Expr::Len(_) | Expr::Annot(..) | Expr::Conform(..) => {
            unreachable!("Expr::Len, Expr::Annot and Expr::Conform are typecheck output, never elaboration input")
        }
        Expr::Int(_) => Ok((Type::Int, EffectRow::pure(), expr)),
        Expr::Float(_) => Ok((Type::Float, EffectRow::pure(), expr)),
        Expr::Bool(_) => Ok((Type::Bool, EffectRow::pure(), expr)),
        Expr::Str(_) => Ok((Type::Str, EffectRow::pure(), expr)),
        // A singleton per source position -- see Expr::Token's own doc
        // comment.
        Expr::Token(id) => Ok((Type::Token(id), EffectRow::pure(), expr)),
        // A type value is Dyn to the checker, like the prelude's type names
        // (there is no type of types yet). The parser only puts one in an
        // annotation slot, which is never elaborated as a value.
        Expr::TypeLit(_) => Ok((Type::Dyn, EffectRow::pure(), expr)),
        Expr::Var(name) => Ok((lookup(ctx, &name, infer), EffectRow::pure(), expr)),

        // Per-position types, no widening -- unlike ListLit just below,
        // which exists for a genuinely variable-length, conceptually
        // homogeneous sequence. `(1, "a")` is `Type::Tuple([Int, Str])`,
        // not widened to `List(Dyn)`.
        Expr::Tuple(items) => {
            // Case B construction, step case (design spec 2026-09-20
            // sec 4): checked against an Indexed(Rec(..), idx) type
            // whose union qualifies and whose step alternative
            // is a same-arity Tuple, check each non-recursive item
            // against its corresponding step_alt element and the
            // recursive item (self_ref_position) against
            // the Rec at idx - 1 -- Case A's Cons arm (Task 2),
            // generalized from a fixed List/Cons shape to an
            // arbitrary 2-alternative recursive union.
            if let Mode::Check(expected) = mode {
                if let Type::Indexed(wrapped, idx) = infer.resolve_deep(expected) {
                    {
                        if let Some((_, step_alt, self_ref)) = match &*wrapped {
                            Type::Rec(g, i) => qualifying_rec_alternatives(g, *i),
                            _ => None,
                        } {
                            if let Type::Tuple(step_items) = &step_alt {
                                if step_items.len() == items.len() {
                                    if let Some(rec_pos) = self_ref_position(&step_alt, self_ref) {
                                        let mut row = EffectRow::pure();
                                        let mut refs = Vec::with_capacity(items.len());
                                        for (i, item) in items.into_iter().enumerate() {
                                            let (item_row, item2) = if i == rec_pos {
                                                let rec_expected = Type::Indexed(wrapped.clone(), Rc::new(IndexExpr::Sub(idx.clone(), Rc::new(IndexExpr::Lit(1)))));
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
            Ok((Type::Tuple(Hc::new(tys)), row, arena.push(Expr::Tuple(refs))))
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
            Ok((Type::Record(Hc::new(field_tys)), row, arena.push(Expr::Record(Rc::new(field_refs)))))
        }

        // `p.x` -- desugars into an ordinary get_field(target, "x") call
        // (Builtin::GetField, machine.rs) -- never a new machine.rs
        // opcode. Statically checked where `target`'s type says enough
        // to check: a concrete Type::Record (or a Type::Union of them,
        // every alternative required to have the field -- see
        // record_field_type's own doc comment), rejected immediately if
        // it's neither and not Dyn either, the same way calling a
        // non-Fun value is. A Dyn target gets a REAL runtime shape check
        // (build_boundary_check, same machinery every other Dyn-to-Record
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
                other => return Err(TypeError(format!("expected a record, found {}", other.brief()), spans[target])),
            };
            let checked_target = if matches!(target_ty, Type::Dyn | Type::Var(_)) {
                let required = Type::Record(Hc::new(vec![(name.clone(), Type::Dyn)]));
                build_boundary_check(arena, target2, &required, &CheckEnv::at(infer, spans[target]))
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
                        let list_ty = Type::Indexed(Hc::new(Type::List(Hc::new(elem_ty))), Rc::new(IndexExpr::Lit(item_count as i64)));
                        return Ok((list_ty, row, arena.push(Expr::ListLit(refs))));
                    }
                }
            }
            let list_ty = Type::List(Hc::new(elem_ty));
            Ok((list_ty, row, arena.push(Expr::ListLit(refs))))
        }

        Expr::Let(..) | Expr::LetRec(..) | Expr::Lambda(..) => {
            unreachable!("Let/LetRec/Lambda are peeled by elaborate's chain-flattening loop")
        }

        Expr::App(f, a) => {
            let (f_ty, f_row, f2) = elaborate(arena, f, ctx, spans, infer)?;
            let mut f_ty = infer.resolve(&f_ty);
            // A callee typed by a (recursive) alias is called through its body, one step.
            let unfolded = match &f_ty {
                Type::Rec(g, i) => Some(unfold(g, *i)).filter(|raw| !matches!(raw, Type::Rec(..))),
                _ => None,
            };
            f_ty = unfolded.unwrap_or(f_ty);
            // A callee whose parameter type mentions an Indexed type
            // (`Vec(n)`, `Indexed(T, n)`, bare or nested in a tuple/list) gets its argument CHECKED against it, so a
            // list/tuple literal argument can satisfy the index -- Synth
            // alone gives a plain un-indexed List/Tuple that never can.
            // Deliberately limited to such parameters, and to ones with no
            // free row variables: Check mode would otherwise drop the
            // argument's own synthesized type, which bind_row_vars below
            // needs to bind effect-row variables.
            let (a_ty, a_row, a2) = match &f_ty {
                Type::Fun(param_ty, ..) if {
                    let p = infer.resolve_deep(param_ty);
                    contains_indexed(&p) && !vars_in(&p).iter().any(|v| matches!(v, TyVar::Row(_)))
                } => {
                    let param_resolved = infer.resolve_deep(param_ty);
                    let (row, a2) = check_against(arena, a, &param_resolved, ctx, spans, infer)?;
                    (param_resolved, row, a2)
                }
                _ => elaborate(arena, a, ctx, spans, infer)?,
            };
            // Resolve before dispatching: f_ty is read straight out of
            // Ctx (via lookup), which -- since generalization quantifies
            // only variables not open in the context -- can hand back a raw Type::Var that
            // unify() has ALREADY bound to something concrete elsewhere
            // (e.g. a variable reused across two calls in the same
            // scope). Matching on the unresolved form would route a
            // callee that's actually a concrete Fun through the
            // Type::Var(_) arm below -- which has no coerce() at all --
            // instead of the Type::Fun arm's own subtyping-aware
            // handling, or would silently skip the Type::Dyn arm's own
            // function check when the callee resolves to Dyn. Same
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
                    let fun_shape = Type::Fun(Hc::new(param_ty.clone()), EffectRow::Dyn, Hc::new(ret_ty.clone()));
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
                    let f3 = build_boundary_check(arena, f2, &any_fun(), &CheckEnv::at(infer, spans[f]));
                    // The argument crosses into an unknown function, so it is
                    // cast to Dyn like any other Dyn position (a typed
                    // function argument gets its wrapper).
                    infer.note_dyn_sink(&a_ty);
                    let a3 = coerce_cast(arena, a2, &a_ty, &Type::Dyn, &CheckEnv::at(infer, spans[a]));
                    (EffectRow::Dyn, Type::Dyn, arena.push(Expr::App(f3, a3)))
                }
                other => return Err(TypeError(format!("cannot call a value of type {}", other.brief()), spans[f])),
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
                            format!("type mismatch: cannot compare {} with {}", l_ty.brief(), r_ty.brief()),
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
                    // statically-known length is (relate applies the same
                    // widening to an Indexed source, used for the
                    // App-argument case instead). Phase 1 has no
                    // rule for the RESULT length of concatenating two
                    // Vec(n)s, so this deliberately drops to the wrapped
                    // type entirely rather than half-tracking it.
                    let l_ty = forget_index(&l_ty);
                    let r_ty = forget_index(&r_ty);
                    if !consistent(&l_ty, &r_ty) {
                        return Err(TypeError(
                            format!("type mismatch: cannot concat {} with {}", l_ty.brief(), r_ty.brief()),
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
                                    "type mismatch: cannot concat {} with {} (expected two strings or two lists)",
                                    l_ty.brief(),
                                    r_ty.brief()
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
                    let list_shape = Type::List(Hc::new(elem_ty.clone()));
                    // "Forget" the tail's own tracked index before
                    // unifying its shape -- same widening as Concat's own
                    // arm just above (see forget_index's own doc
                    // comment); the result is always a plain Type::List
                    // regardless, so there's nothing Indexed left to
                    // preserve past this point anyway.
                    unify(&forget_index(&r_ty), &list_shape, infer, spans[r])?;
                    unify(&l_ty, &elem_ty, infer, spans[l])?;
                    let result_ty = Type::List(Hc::new(infer.resolve_deep(&elem_ty)));
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
            let discharged = discharged_effect(arena, handler);
            infer.resume_row = discharged.as_ref().map(|effect| body_row.remove(effect));
            let handler_result = elaborate(arena, handler, ctx, spans, infer);
            infer.resume_row = None;
            let (handler_ty, handler_row, handler2) = handler_result?;
            // types.rs has no Type::Handler -- but every handler-producing
            // expression (MakeHandler, or deep(...)/shallow(...) applied
            // to one, or a var bound from either) synthesizes Dyn by this
            // same convention, so a concretely-typed handler expression
            // (Int, Bool, Fun) can never legitimately be one. Reject it
            // statically instead of letting it reach machine.rs's panic.
            let handler_ty = infer.resolve_deep(&handler_ty);
            if !matches!(handler_ty, Type::Dyn | Type::Var(_)) {
                return Err(TypeError(
                    format!("handle: expected a handler value, found expression of type {}", handler_ty.brief()),
                    spans[handler],
                ));
            }
            // Discharge the effect this handler catches, if we can
            // statically tell which one that is. An unknown handler (a
            // variable, a parameter) may catch anything: the row becomes Dyn.
            let row = match discharged {
                Some(effect) => body_row.remove(&effect),
                None => EffectRow::Dyn,
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
                Mode::Check(_) => indexed_refinement_target(&resolved_scrut_ty, resolved_index.as_ref()),
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
                IndexedRefinementTarget::List { index_var, .. } | IndexedRefinementTarget::Rec { index_var, .. } => index_var.clone(),
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
                            infer.index_subst.insert_mut(k.clone(), v.clone());
                        }
                        None => {
                            infer.index_subst.remove_mut(k);
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
                if !pattern_could_match(pat, &scrut_ty, &Visiting::default()) {
                    return Err(TypeError(
                        format!(
                            "match: pattern of type {} can never match scrutinee of type {}",
                            pattern_type(pat).brief(),
                            scrut_ty.brief()
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
                let mut arm_ctx = bind_pattern_vars(ctx, pat, &scrut_ty, infer, spans[expr], &Visiting::default())?;
                // Hypothesis injection (spec section 5): base case implies
                // the scrutinee's own index is 0; step case mints a fresh
                // index variable, re-types the self-referential
                // sub-binding as Indexed(_, fresh) (overriding
                // bind_pattern_vars's own plain binding via a second
                // `extend` call), and hypothesizes the scrutinee's own
                // index is `fresh + 1`. Case A (List) derives base/step
                // from the fixed []/:: patterns directly; Case B (Rec+
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
                                infer.index_subst.insert_mut(n_name.clone(), IndexExpr::Lit(0));
                            }
                            Pattern::Cons(_, tail) => {
                                if let Pattern::Var(tail_name) = &**tail {
                                    let m = fresh_index_name("m");
                                    let hypothesis = IndexExpr::Add(Rc::new(IndexExpr::Var(m.clone())), Rc::new(IndexExpr::Lit(1)));
                                    infer.index_subst.insert_mut(n_name.clone(), hypothesis);
                                    infer.rigid_index.insert(m.clone());
                                    let refined_tail_ty = Type::Indexed(Hc::new(Type::List(elem_ty.clone())), Rc::new(IndexExpr::Var(m)));
                                    arm_ctx = extend(&arm_ctx, tail_name, refined_tail_ty);
                                }
                                // A compound/nested tail pattern: no override,
                                // no hypothesis -- v1 scope boundary, see this
                                // plan's own Global Constraints.
                            }
                            _ => {}
                        },
                        IndexedRefinementTarget::Rec { index_var: n_name, rec, base_alt, step_alt, self_ref } => {
                            // `base_alt`/`step_alt` are themselves already-
                            // unfolded alternatives of the recursive type (see
                            // `qualifying_rec_alternatives`), so this
                            // probe is conceptually already mid-unfold of
                            // the member -- pass its own visiting entry, not an empty set, for
                            // consistency with the `visiting`-threading
                            // discipline the rest of this phase established
                            // (the exact shape whose absence caused
                            // bind_pattern_vars's own fix-round-1 bug).
                            // Doesn't change behavior today (verified by
                            // trace: neither alternative can itself unfold
                            // back to the member without violating
                            // `qualifying_rec_alternatives`'s own
                            // occurrence-count check), just defense in
                            // depth.
                            let visiting_id = Visiting::of(rec);
                            let matches_base = pattern_could_match(pat, base_alt, &visiting_id);
                            let matches_step = pattern_could_match(pat, step_alt, &visiting_id);
                            if matches_base && !matches_step {
                                infer.index_subst.insert_mut(n_name.clone(), IndexExpr::Lit(0));
                            } else if matches_step && !matches_base {
                                if let Some(Pattern::Var(self_ref_name)) = self_ref_pattern(pat, step_alt, *self_ref) {
                                    let m = fresh_index_name("m");
                                    let hypothesis = IndexExpr::Add(Rc::new(IndexExpr::Var(m.clone())), Rc::new(IndexExpr::Lit(1)));
                                    infer.index_subst.insert_mut(n_name.clone(), hypothesis);
                                    infer.rigid_index.insert(m.clone());
                                    let refined_ty = Type::Indexed(Hc::new(rec.clone()), Rc::new(IndexExpr::Var(m)));
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

        // Constructing the handler value is pure. The clause body is typed as a function of the
        // row its `resume` has: Handle supplies it (resume_row) when it can see the clause, and
        // the clause's own row then joins the handle's row. Anywhere else `resume` is Dyn and the
        // clause's row is discarded.
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let resume_row = infer.resume_row.take();
            let resume_ty = match &resume_row {
                Some(row) => Type::Fun(Hc::new(Type::Dyn), row.clone(), Hc::new(Type::Dyn)),
                None => Type::Dyn,
            };
            let inner_ctx = extend(&extend(ctx, &payload_var, Type::Dyn), &resume_var, resume_ty);
            infer.repeatable_depth += 1;
            let body_result = elaborate(arena, body, &inner_ctx, spans, infer);
            infer.repeatable_depth -= 1;
            let (_, clause_row, body2) = body_result?;
            Ok((
                Type::Dyn,
                if resume_row.is_some() { clause_row } else { EffectRow::pure() },
                arena.push(Expr::MakeHandler { effect, payload_var, resume_var, body: body2 }),
            ))
        }
    }
}

// Every name a binder in `arena` binds (InferCtx::bound_names).
fn binder_names(arena: &Arena) -> HashSet<String> {
    let mut names = HashSet::new();
    for (_, e) in arena.iter() {
        match e {
            Expr::Let(var, ..) | Expr::Lambda(var, ..) => {
                names.insert(var.clone());
            }
            Expr::LetRec(bindings, _) => names.extend(bindings.iter().map(|(n, _, _)| n.clone())),
            Expr::MakeHandler { payload_var, resume_var, .. } => names.extend([payload_var.clone(), resume_var.clone()]),
            Expr::Match(_, arms) => names.extend(arms.iter().flat_map(|(p, _, _)| crate::resolve::pattern_vars(p))),
            _ => {}
        }
    }
    names
}

pub fn check(arena: &mut Arena, root: ExprRef, spans: &SpanMap) -> Result<ExprRef, TypeError> {
    let mut infer = InferCtx::new();
    infer.bound_names = binder_names(arena);
    let (_, row, elaborated) = elaborate(arena, root, &Ctx::empty(), spans, &mut infer)?;
    resolve_pendings(arena, &infer);
    resolve_obligations(arena, &infer);
    settle_erasure(arena, spans, &mut infer);
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
