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

// A binding's type, plus the row-variable names (from explicit `->{e}`
// annotations reachable in it) that are generalized -- quantified fresh at
// every use, the way ML/Haskell generalize a `let`-bound type. Row
// variables never come from inference (renno has none for value types),
// only from what the user wrote, so this is simple name substitution, not
// a unification engine: extend_generalized computes the row_vars once at
// the `let`, lookup renames them to fresh names at each reference.
#[derive(Clone)]
struct Scheme {
    row_vars: Vec<String>,
    ty: Type,
}

impl Scheme {
    fn mono(ty: Type) -> Scheme {
        Scheme { row_vars: Vec::new(), ty }
    }
}

type Ctx = PList<Scheme>;

fn lookup(ctx: &Ctx, name: &str) -> Type {
    // Unbound at typecheck time: don't error here, machine::run's own
    // `unbound variable` panic at runtime is the right place for that.
    match ctx.get(name) {
        None => Type::Dyn,
        Some(scheme) if scheme.row_vars.is_empty() => scheme.ty,
        Some(scheme) => {
            let subst: HashMap<String, EffectRow> = scheme
                .row_vars
                .iter()
                .map(|v| (v.clone(), EffectRow::Var(fresh_row_name(v))))
                .collect();
            subst_type(&scheme.ty, &subst)
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
fn extend_generalized(ctx: &Ctx, name: &str, ty: Type) -> Ctx {
    let row_vars: Vec<String> = free_row_vars(&ty).into_iter().collect();
    ctx.bind(name, Scheme { row_vars, ty })
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
        Type::Dyn | Type::Int | Type::Bool | Type::Str | Type::Token(_) => BTreeSet::new(),
    }
}

fn fresh_row_name(base: &str) -> String {
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
fn subst_type(ty: &Type, subst: &HashMap<String, EffectRow>) -> Type {
    match ty {
        Type::Fun(param, row, ret) => Type::Fun(
            Rc::new(subst_type(param, subst)),
            resolve_row(row, subst),
            Rc::new(subst_type(ret, subst)),
        ),
        Type::List(elem) => Type::List(Rc::new(subst_type(elem, subst))),
        Type::Tuple(items) => Type::Tuple(Rc::new(items.iter().map(|t| subst_type(t, subst)).collect())),
        Type::Union(alts) => Type::Union(Rc::new(alts.iter().map(|t| subst_type(t, subst)).collect())),
        Type::Record(fields) => {
            Type::Record(Rc::new(fields.iter().map(|(n, t)| (n.clone(), subst_type(t, subst))).collect()))
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

// The only place a runtime boundary check gets built: `from` is Dyn
// (unknown statically) and `to` is concrete. If both sides are concrete
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
    if *from != Type::Dyn || *to == Type::Dyn {
        return Ok(e);
    }
    Ok(build_boundary_check(arena, e, to))
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
        Type::Bool => build_shallow_check(arena, e, to, "is_bool"),
        Type::Str => build_shallow_check(arena, e, to, "is_str"),
        Type::List(_) => build_shallow_check(arena, e, to, "is_list"),
        Type::Fun(param_ty, _row, ret_ty) => wrap_fun_contract(arena, e, param_ty.clone(), ret_ty.clone()),
        Type::Token(id) => build_token_check(arena, e, to, *id),
        Type::Tuple(_) => build_shape_check(arena, e, to),
        Type::Record(_) => build_shape_check(arena, e, to),
        Type::Union(_) => build_union_check(arena, e, to),
        Type::Dyn => unreachable!("coerce only calls this once *to != Type::Dyn is already established"),
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
        Type::Int => build_predicate_call(arena, "is_int", value_ref),
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
            let mut result = arena.push(Expr::Bool(true));
            for (name, _) in fields.iter().rev() {
                let has = build_has_field_call(arena, value_ref, name);
                let false_lit = arena.push(Expr::Bool(false));
                result = arena.push(Expr::If(has, result, false_lit));
            }
            let is_record = build_predicate_call(arena, "is_record", value_ref);
            let false_lit = arena.push(Expr::Bool(false));
            arena.push(Expr::If(is_record, result, false_lit))
        }
        Type::Union(alts) => {
            let mut result = arena.push(Expr::Bool(false));
            for alt in alts.iter().rev() {
                let p = build_shape_predicate(arena, value_ref, alt);
                let true_lit = arena.push(Expr::Bool(true));
                result = arena.push(Expr::If(p, true_lit, result));
            }
            result
        }
    }
}

fn build_predicate_call(arena: &mut Arena, name: &str, value_ref: ExprRef) -> ExprRef {
    let v = arena.push(Expr::Var(name.to_string()));
    arena.push(Expr::App(v, value_ref))
}

// `has_field(value_ref, field_name)` -- the two-argument counterpart to
// build_predicate_call, used only by Record's own shape predicate/check.
fn build_has_field_call(arena: &mut Arena, value_ref: ExprRef, field_name: &str) -> ExprRef {
    let f = arena.push(Expr::Var("has_field".to_string()));
    let applied = arena.push(Expr::App(f, value_ref));
    let name_lit = arena.push(Expr::Str(field_name.to_string()));
    arena.push(Expr::App(applied, name_lit))
}

// A singleton: the only thing a Dyn-sourced value could ever satisfy this
// against is the EXACT same Value::Token, so this reuses ordinary `==`
// (BinOp::Eq, backed by machine::value_eq's own Token arm) against a
// freshly-embedded literal of the same id, rather than a dedicated
// predicate builtin.
fn build_token_check(arena: &mut Arena, e: ExprRef, to: &Type, id: u64) -> ExprRef {
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let literal = arena.push(Expr::Token(id));
    let eq = arena.push(Expr::BinOp(BinOp::Eq, tmp_ref, literal));
    let fail_call = build_fail_call(arena, to, tmp_ref);
    let if_expr = arena.push(Expr::If(eq, tmp_ref, fail_call));
    arena.push(Expr::Let(tmp, None, e, if_expr))
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
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let pred = build_shape_predicate(arena, tmp_ref, to);
    let fail_call = build_fail_call(arena, to, tmp_ref);
    let if_expr = arena.push(Expr::If(pred, tmp_ref, fail_call));
    arena.push(Expr::Let(tmp, None, e, if_expr))
}

// `let __check_tmp = e in if <predicate>(__check_tmp) then __check_tmp
// else fail("type error: expected {to}, found " ++ type_name(__check_tmp))`
// -- binding `e` once (not re-evaluating it for both the predicate call
// and the passed-through result) is what makes this safe for an `e` with
// side effects (an arbitrary expression, not necessarily a bare
// variable). `predicate` is looked up by name through the ordinary
// prelude Env (`is_int`/`is_bool`/`is_str`/`is_list`/`is_fun`), the same
// way `fail`/`type_name` already are -- see their own doc comments on the
// (pre-existing, accepted) shadowing risk that implies.
fn build_shallow_check(arena: &mut Arena, e: ExprRef, to: &Type, predicate: &str) -> ExprRef {
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let pred_call = build_predicate_call(arena, predicate, tmp_ref);
    let fail_call = build_fail_call(arena, to, tmp_ref);
    let if_expr = arena.push(Expr::If(pred_call, tmp_ref, fail_call));
    arena.push(Expr::Let(tmp, None, e, if_expr))
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
        (Pattern::Record(_), Type::Dyn) => true,
        (Pattern::Record(_), _) => false,
        _ => consistent(ty, &pattern_type(pat)),
    }
}

// Extends `ctx` with every Var this pattern binds, each as Dyn (renno has
// no pattern-driven type refinement -- e.g. a List pattern's element
// bindings don't learn the list's element type). "_" is just an ordinary
// Var name here, bound like any other.
fn bind_pattern_vars(ctx: &Ctx, pat: &Pattern) -> Ctx {
    match pat {
        Pattern::Var(name) => extend(ctx, name, Type::Dyn),
        Pattern::Int(_) | Pattern::Bool(_) | Pattern::Str(_) => ctx.clone(),
        Pattern::List(pats) => pats.iter().fold(ctx.clone(), |c, p| bind_pattern_vars(&c, p)),
        Pattern::Cons(head, tail) => bind_pattern_vars(&bind_pattern_vars(ctx, head), tail),
        Pattern::Record(fields) => fields.iter().fold(ctx.clone(), |c, (_, p)| bind_pattern_vars(&c, p)),
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
// any -- reported as unreachable (dead code).
fn first_unreachable(patterns: &[&Pattern]) -> Option<usize> {
    (0..patterns.len()).find(|&i| (0..i).any(|j| dominates(patterns[j], patterns[i])))
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
fn elaborate(arena: &mut Arena, expr: ExprRef, ctx: &Ctx, spans: &SpanMap) -> Result<(Type, EffectRow, ExprRef), TypeError> {
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
                let (val_ty, val_row, val2) = elaborate(arena, val, &cur_ctx, spans)?;
                let (bound_ty, val3) = match ann {
                    Some(t) => (t.clone(), coerce(arena, val2, &val_ty, &t, spans[val])?),
                    None => (val_ty, val2),
                };
                cur_ctx = extend_generalized(&cur_ctx, &var, bound_ty.clone());
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
                    val_ctx = extend(&val_ctx, name, ann.clone().unwrap_or(Type::Dyn));
                }
                let mut elaborated = Vec::with_capacity(bindings.len());
                for (name, ann, val) in bindings.iter() {
                    let (val_ty, val_row, val2) = elaborate(arena, *val, &val_ctx, spans)?;
                    let (bound_ty, val3) = match ann {
                        Some(t) => (t.clone(), coerce(arena, val2, &val_ty, t, spans[*val])?),
                        None => (val_ty, val2),
                    };
                    elaborated.push((name.clone(), bound_ty, val_row, val3));
                }
                for (name, bound_ty, _, _) in &elaborated {
                    cur_ctx = extend_generalized(&cur_ctx, name, bound_ty.clone());
                }
                pending.push(PendingElab::LetRec { bindings: elaborated });
                cur_expr = body;
            }
            Expr::Lambda(param, ann, body) => {
                let param_ty = ann.unwrap_or(Type::Dyn);
                cur_ctx = extend(&cur_ctx, &param, param_ty.clone());
                pending.push(PendingElab::Fun { param, param_ty });
                cur_expr = body;
            }
            _ => break,
        }
    }

    let (mut result_ty, mut result_row, mut result_expr) = elaborate_node(arena, cur_expr, &cur_ctx, spans)?;

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
fn elaborate_node(arena: &mut Arena, expr: ExprRef, ctx: &Ctx, spans: &SpanMap) -> Result<(Type, EffectRow, ExprRef), TypeError> {
    let node = arena[expr].clone();
    match node {
        Expr::Int(_) => Ok((Type::Int, EffectRow::pure(), expr)),
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
                let (item_ty, item_row, item2) = elaborate(arena, item, ctx, spans)?;
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
                let (field_ty, field_row, field2) = elaborate(arena, *field_expr, ctx, spans)?;
                row = EffectRow::union(&row, &field_row);
                field_tys.push((name.clone(), field_ty));
                field_refs.push((name.clone(), field2));
            }
            Ok((Type::Record(Rc::new(field_tys)), row, arena.push(Expr::Record(Rc::new(field_refs)))))
        }

        Expr::ListLit(items) => {
            let mut row = EffectRow::pure();
            let mut elem_ty: Option<Type> = None;
            let mut refs = Vec::with_capacity(items.len());
            for item in items {
                let (item_ty, item_row, item2) = elaborate(arena, item, ctx, spans)?;
                row = EffectRow::union(&row, &item_row);
                refs.push(item2);
                // Same rule as If's branches: differing concrete element
                // types aren't an error (no union types) -- widen to Dyn.
                elem_ty = Some(match elem_ty {
                    None => item_ty,
                    Some(t) if t == item_ty => t,
                    Some(_) => Type::Dyn,
                });
            }
            let list_ty = Type::List(Rc::new(elem_ty.unwrap_or(Type::Dyn)));
            Ok((list_ty, row, arena.push(Expr::ListLit(refs))))
        }

        Expr::Let(..) | Expr::LetRec(..) | Expr::Lambda(..) => {
            unreachable!("Let/LetRec/Lambda are peeled by elaborate's chain-flattening loop")
        }

        Expr::App(f, a) => {
            let (f_ty, f_row, f2) = elaborate(arena, f, ctx, spans)?;
            let (a_ty, a_row, a2) = elaborate(arena, a, ctx, spans)?;
            let (call_row, ret_ty, app2) = match &f_ty {
                Type::Fun(param_ty, call_row, ret_ty) => {
                    let a3 = coerce(arena, a2, &a_ty, param_ty, spans[a])?;
                    // If param_ty names a row variable (from an explicit
                    // `->{e}` annotation on the callee) and the argument's
                    // own inferred type reveals a concrete row in the
                    // matching position, bind it -- and carry that binding
                    // into the return type and this call's own row, so a
                    // row-polymorphic function's result is precisely typed
                    // once its callback is known, not just Dyn.
                    let mut subst = HashMap::new();
                    bind_row_vars(param_ty, &a_ty, &mut subst);
                    let ret_ty2 = subst_type(ret_ty, &subst);
                    let call_row2 = resolve_row(call_row, &subst);
                    (call_row2, ret_ty2, arena.push(Expr::App(f2, a3)))
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
            let (l_ty, l_row, l2) = elaborate(arena, l, ctx, spans)?;
            let (r_ty, r_row, r2) = elaborate(arena, r, ctx, spans)?;
            let row = EffectRow::union(&l_row, &r_row);
            match op {
                // Arithmetic and ordering: both operands must be Int.
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Lt => {
                    let l3 = coerce(arena, l2, &l_ty, &Type::Int, spans[l])?;
                    let r3 = coerce(arena, r2, &r_ty, &Type::Int, spans[r])?;
                    let result_ty = if op == BinOp::Lt { Type::Bool } else { Type::Int };
                    Ok((result_ty, row, arena.push(Expr::BinOp(op, l3, r3))))
                }
                // Equality: operands just need to be consistent with EACH
                // OTHER, not both forced to Int -- `true == false` is a
                // real comparison. If one side is Dyn and the other
                // concrete, coerce the Dyn side to the concrete side's
                // type so the runtime value at least has a known tag;
                // apply_binop compares by matching Value variants.
                BinOp::Eq => {
                    if !consistent(&l_ty, &r_ty) {
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
                    let list_of_dyn = Type::List(Rc::new(Type::Dyn));
                    if !consistent(&r_ty, &list_of_dyn) {
                        return Err(TypeError(
                            format!("type mismatch: expected a list, found {r_ty}"),
                            spans[r],
                        ));
                    }
                    let result_ty = match &r_ty {
                        Type::List(elem) if **elem == l_ty => Type::List(Rc::new(l_ty.clone())),
                        _ => Type::List(Rc::new(Type::Dyn)),
                    };
                    Ok((result_ty, row, arena.push(Expr::BinOp(op, l2, r2))))
                }
            }
        }

        Expr::If(c, t, e) => {
            let (c_ty, c_row, c2) = elaborate(arena, c, ctx, spans)?;
            let c3 = coerce(arena, c2, &c_ty, &Type::Bool, spans[c])?;
            let (t_ty, t_row, t2) = elaborate(arena, t, ctx, spans)?;
            let (e_ty, e_row, e2) = elaborate(arena, e, ctx, spans)?;
            // Branches with differing concrete types aren't an error here
            // (no union types) -- just widen to Dyn rather than reject.
            let result_ty = if t_ty == e_ty { t_ty } else { Type::Dyn };
            // Only one branch runs, but which one isn't known statically,
            // so the possible effects are the union of both.
            let row = EffectRow::union(&c_row, &EffectRow::union(&t_row, &e_row));
            Ok((result_ty, row, arena.push(Expr::If(c3, t2, e2))))
        }

        // The effect this specific operation performs, plus whatever the
        // payload expression itself might perform.
        Expr::Perform(effect, payload) => {
            let (_, payload_row, payload2) = elaborate(arena, payload, ctx, spans)?;
            let row = EffectRow::union(&payload_row, &EffectRow::single(&effect));
            Ok((Type::Dyn, row, arena.push(Expr::Perform(effect, payload2))))
        }

        Expr::Handle { body, handler } => {
            let (_, body_row, body2) = elaborate(arena, body, ctx, spans)?;
            let (handler_ty, handler_row, handler2) = elaborate(arena, handler, ctx, spans)?;
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
            let pats: Vec<&Pattern> = arms.iter().map(|(p, _)| p).collect();
            if let Some(i) = first_unreachable(&pats) {
                return Err(TypeError(
                    "unreachable match arm: an earlier arm already covers everything it matches".to_string(),
                    spans[arms[i].1],
                ));
            }

            let (scrut_ty, scrut_row, scrutinee2) = elaborate(arena, scrutinee, ctx, spans)?;
            let mut row = scrut_row;
            let mut result_ty: Option<Type> = None;
            let mut new_arms = Vec::with_capacity(arms.len());
            for (pat, (_, body)) in pats.iter().copied().zip(arms.iter()) {
                if !pattern_could_match(pat, &scrut_ty) {
                    return Err(TypeError(
                        format!(
                            "match: pattern of type {} can never match scrutinee of type {scrut_ty}",
                            pattern_type(pat)
                        ),
                        spans[expr],
                    ));
                }
                let arm_ctx = bind_pattern_vars(ctx, pat);
                let (arm_ty, arm_row, body2) = elaborate(arena, *body, &arm_ctx, spans)?;
                row = EffectRow::union(&row, &arm_row);
                // Same widen-to-Dyn-on-disagreement rule as If's branches
                // and ListLit's elements -- no union types.
                result_ty = Some(match result_ty {
                    None => arm_ty,
                    Some(t) if t == arm_ty => t,
                    Some(_) => Type::Dyn,
                });
                new_arms.push((pat.clone(), body2));
            }
            if let Some(missing) = missing_case(&pats, &scrut_ty) {
                return Err(TypeError(format!("non-exhaustive match: {missing}"), spans[expr]));
            }
            Ok((result_ty.unwrap_or(Type::Dyn), row, arena.push(Expr::Match(scrutinee2, Rc::new(new_arms)))))
        }

        // Constructing the handler value is pure -- the clause body's own
        // effects (including what `resume` re-enters) aren't modeled here;
        // see the doc comment on `elaborate`.
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let inner_ctx = extend(&extend(ctx, &payload_var, Type::Dyn), &resume_var, Type::Dyn);
            let (_, _, body2) = elaborate(arena, body, &inner_ctx, spans)?;
            Ok((
                Type::Dyn,
                EffectRow::pure(),
                arena.push(Expr::MakeHandler { effect, payload_var, resume_var, body: body2 }),
            ))
        }
    }
}

pub fn check(arena: &mut Arena, root: ExprRef, spans: &SpanMap) -> Result<ExprRef, TypeError> {
    let (_, row, elaborated) = elaborate(arena, root, &Ctx::empty(), spans)?;
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
