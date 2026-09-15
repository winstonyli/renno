use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::expr::{Arena, BinOp, DataInfo, Expr, ExprRef, Pattern, SpanMap};
use crate::plist::PList;
use crate::span::Span;
use crate::types::{consistent, EffectRow, Type};

// The Span is always an ORIGINAL (pre-elaboration) node's -- the one whose
// type was being checked when the error fired, always already in scope
// (elaborate_node's own `expr` parameter, or a child ExprRef it destructured
// from that same original node) at the point of construction. typecheck
// never needs to invent a span for a node IT synthesizes (a Check, a
// wrap_fun_contract chain): every error path returns before any such node
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
        Type::Dyn | Type::Int | Type::Bool | Type::Str | Type::Data(_, _) | Type::Token(_) => BTreeSet::new(),
    }
}

// Finds the DataInfo a resolved Type::Data(name, brand) refers to. When
// `brand` is Some, that's exact -- match the declaration with the SAME
// brand, not just the same name (two shadowed opaque `data Foo` blocks
// share a name but never a brand, see types::consistent_inner's own doc
// comment). When `brand` is None (unbranded/structural), name is all
// there is -- `.rev()` picks the lexically-current (most recently
// declared) `data Name` on a shadowing collision, not the first/outer
// one, same reasoning as every other `fields` lookup in this codebase.
fn lookup_data<'a>(fields: &'a [Rc<DataInfo>], name: &str, brand: Option<u64>) -> Option<&'a Rc<DataInfo>> {
    match brand {
        Some(b) => fields.iter().rev().find(|f| f.type_name == name && f.brand == Some(b)),
        None => fields.iter().rev().find(|f| f.type_name == name),
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

fn subst_type(ty: &Type, subst: &HashMap<String, EffectRow>) -> Type {
    match ty {
        Type::Fun(param, row, ret) => Type::Fun(
            Rc::new(subst_type(param, subst)),
            resolve_row(row, subst),
            Rc::new(subst_type(ret, subst)),
        ),
        Type::List(elem) => Type::List(Rc::new(subst_type(elem, subst))),
        Type::Tuple(items) => Type::Tuple(Rc::new(items.iter().map(|t| subst_type(t, subst)).collect())),
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
// all. If `from` is already exactly consistent and concrete, no check
// needed: zero overhead for fully-annotated code. The error, if any,
// points at `span` -- the CALLER's job to have already looked up via the
// ORIGINAL (pre-elaboration) ExprRef for the value in question, e.g.
// `spans[val]` where `val` is a field straight off the un-elaborated
// node, NOT the elaborated `e` this function receives to potentially
// wrap: elaborate_node re-pushes almost every compound node it touches
// (App, BinOp, If, ...) regardless of whether anything actually changed,
// so `e` itself is often already a brand new ExprRef past the end of the
// parser-built SpanMap by the time it reaches here -- indexing spans BY
// IT, not by the original, was a latent bug (worked by coincidence
// whenever the mismatched value happened to be a leaf that elaborate_node
// returns unchanged).
fn coerce(
    arena: &mut Arena,
    e: ExprRef,
    from: &Type,
    to: &Type,
    span: Span,
    fields: &[Rc<DataInfo>],
) -> Result<ExprRef, TypeError> {
    if !consistent(from, to, fields) {
        // Same printed name, different brand: Display never shows a brand
        // (it's meant to stay hidden -- see Type::Data's own doc comment),
        // so "expected Meters, found Meters" alone would be genuinely
        // confusing -- name a likely cause instead of leaving the reader
        // to guess why two identically-printed types disagree.
        let note = match (from, to) {
            (Type::Data(fname, fbrand), Type::Data(tname, tbrand)) if fname == tname && fbrand != tbrand => {
                " (two separately-declared `data` blocks reusing this name -- not the same declaration)"
            }
            _ => "",
        };
        return Err(TypeError(format!("type mismatch: expected {to}, found {from}{note}"), span));
    }
    if *from != Type::Dyn || *to == Type::Dyn {
        return Ok(e);
    }
    Ok(build_boundary_check(arena, e, to, fields))
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
// else fail(msg)` -- instead of a dedicated Check/CheckData AST node:
// Int/Bool/Str/List get a single builtin predicate call (is_int, etc.);
// Fun gets a real per-call contract (wrap_fun_contract, since a bare
// callability tag can't see inside a closure -- "callable" isn't
// "callable with this exact signature"); Data gets check_data_shape with
// a witness built from the declaration's own ctor_types (Value::
// matches_type's old Data case could only confirm "some non-empty tagged
// List," not "specifically shaped like THIS declaration" -- see
// types::Type::Data's doc comment on that gap). Every predicate call
// (including Fun's nested contract) still panics with the exact same
// "type error: expected X, found Y" text the old Check/CheckData Frames
// produced, built at RUNTIME via the type_name builtin since the actual
// mismatched value's type isn't known until then.
fn build_boundary_check(arena: &mut Arena, e: ExprRef, to: &Type, fields: &[Rc<DataInfo>]) -> ExprRef {
    match to {
        Type::Int => build_shallow_check(arena, e, to, "is_int"),
        Type::Bool => build_shallow_check(arena, e, to, "is_bool"),
        Type::Str => build_shallow_check(arena, e, to, "is_str"),
        Type::List(_) => build_shallow_check(arena, e, to, "is_list"),
        Type::Fun(param_ty, _row, ret_ty) => wrap_fun_contract(arena, e, param_ty.clone(), ret_ty.clone(), fields),
        Type::Data(name, brand) => match lookup_data(fields, name, *brand) {
            Some(info) => {
                let shape: Vec<(String, usize)> = info
                    .ctor_types
                    .iter()
                    .map(|(cname, ctys)| (cname.clone(), 1 + ctys.len() + usize::from(info.brand.is_some())))
                    .collect();
                build_data_check(arena, e, to, &shape, info.brand)
            }
            // Shouldn't normally happen once `fields` is the real registry
            // elaborate builds (every Type::Data comes from an actual
            // declaration) -- conservatively falls back to the old
            // shallow "is this even a list" check rather than panicking,
            // same "can't prove it, don't crash" spirit as everywhere
            // else in this checker.
            None => build_shallow_check(arena, e, to, "is_list"),
        },
        Type::Token(id) => build_token_check(arena, e, to, *id),
        Type::Tuple(items) => build_tuple_check(arena, e, to, items.len()),
        Type::Union(_) => build_union_check(arena, e, to, fields),
        Type::Dyn => unreachable!("coerce only calls this once *to != Type::Dyn is already established"),
    }
}

// `let __check_tmp = e in if <alt1-shape> || <alt2-shape> || ... then
// __check_tmp else fail(...)` -- shallow, like every other check here:
// confirms the value matches SOME alternative's own shape (Tuple's own
// arity-only test, Data's own tag+length+brand test, etc, whichever
// alternative it is), not a full recursive per-position verification.
// Unlike the single-type check builders above, this can't just call
// build_boundary_check per alternative and fall through on failure --
// build_boundary_check's own fail() would abort the WHOLE check on the
// first alternative that doesn't match, instead of trying the next one --
// so this builds each alternative's bare boolean predicate (via
// build_shape_predicate) and OR's them together first, deciding only
// once, at the end, whether to pass the value through or fail.
fn build_union_check(arena: &mut Arena, e: ExprRef, to: &Type, fields: &[Rc<DataInfo>]) -> ExprRef {
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let pred = build_shape_predicate(arena, tmp_ref, to, fields);
    let fail_call = build_fail_call(arena, to, tmp_ref);
    let if_expr = arena.push(Expr::If(pred, tmp_ref, fail_call));
    arena.push(Expr::Let(tmp, None, e, if_expr))
}

// The bare boolean half of build_boundary_check's per-type dispatch --
// "does `value_ref` shallowly look like `ty`," with no let-binding and no
// fail() of its own, so build_union_check can OR several of these
// together before deciding anything. Mirrors build_boundary_check's own
// arms exactly (same shallow-check precedent each one sets), just
// stopping short of wrapping the result in Let/If/fail.
fn build_shape_predicate(arena: &mut Arena, value_ref: ExprRef, ty: &Type, fields: &[Rc<DataInfo>]) -> ExprRef {
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
        Type::Data(name, brand) => match lookup_data(fields, name, *brand) {
            Some(info) => {
                let shape: Vec<(String, usize)> = info
                    .ctor_types
                    .iter()
                    .map(|(cname, ctys)| (cname.clone(), 1 + ctys.len() + usize::from(info.brand.is_some())))
                    .collect();
                build_data_predicate(arena, value_ref, &shape, info.brand)
            }
            None => build_predicate_call(arena, "is_list", value_ref),
        },
        Type::Union(alts) => {
            let mut result = arena.push(Expr::Bool(false));
            for alt in alts.iter().rev() {
                let p = build_shape_predicate(arena, value_ref, alt, fields);
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

// Same witness-building as build_data_check, but returns just the
// check_data_shape CALL (a Bool), not the surrounding let/if/fail.
fn build_data_predicate(arena: &mut Arena, value_ref: ExprRef, shape: &[(String, usize)], brand: Option<u64>) -> ExprRef {
    let pairs: Vec<ExprRef> = shape
        .iter()
        .map(|(name, len)| {
            let tag = arena.push(Expr::Str(name.clone()));
            let len_expr = arena.push(Expr::Int(*len as i64));
            arena.push(Expr::ListLit(vec![tag, len_expr]))
        })
        .collect();
    let witness = arena.push(Expr::ListLit(pairs));
    let brand_expr = match brand {
        Some(id) => arena.push(Expr::Token(id)),
        None => arena.push(Expr::Bool(false)),
    };
    let pred_var = arena.push(Expr::Var("check_data_shape".to_string()));
    let call1 = arena.push(Expr::App(pred_var, value_ref));
    let call2 = arena.push(Expr::App(call1, witness));
    arena.push(Expr::App(call2, brand_expr))
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

// Shallow, like List(_) in build_boundary_check: confirms arity, not that
// each position's own value matches ITS OWN element type -- a full
// per-position recursive check (extract each element, apply
// build_boundary_check to it, rebuild the tuple) is possible but not
// built yet; this matches the same "confirm the shape, not deeper"
// precedent every other Dyn boundary check here already sets.
fn build_tuple_check(arena: &mut Arena, e: ExprRef, to: &Type, arity: usize) -> ExprRef {
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));
    let fail_call = build_fail_call(arena, to, tmp_ref);
    let is_list_var = arena.push(Expr::Var("is_list".to_string()));
    let is_list_call = arena.push(Expr::App(is_list_var, tmp_ref));
    let len_var = arena.push(Expr::Var("len".to_string()));
    let len_call = arena.push(Expr::App(len_var, tmp_ref));
    let arity_lit = arena.push(Expr::Int(arity as i64));
    let len_eq = arena.push(Expr::BinOp(BinOp::Eq, len_call, arity_lit));
    let inner_if = arena.push(Expr::If(len_eq, tmp_ref, fail_call));
    let outer_if = arena.push(Expr::If(is_list_call, inner_if, fail_call));
    arena.push(Expr::Let(tmp, None, e, outer_if))
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
    let pred_var = arena.push(Expr::Var(predicate.to_string()));
    let pred_call = arena.push(Expr::App(pred_var, tmp_ref));
    let fail_call = build_fail_call(arena, to, tmp_ref);
    let if_expr = arena.push(Expr::If(pred_call, tmp_ref, fail_call));
    arena.push(Expr::Let(tmp, None, e, if_expr))
}

// Same overall shape as build_shallow_check, but the predicate is
// check_data_shape(value, witness, brand) instead of a single-argument
// is_X -- `witness` (a literal `[[tag, len], ...]`) and `brand` (a
// Value::Token literal, or `false` when unbranded) are built fresh here
// from `shape`/`brand` and re-evaluated on every check (cheap: a handful
// of small literals, only at a Dyn-to-Data boundary crossing, not on any
// hot arithmetic path) rather than cached across calls.
fn build_data_check(arena: &mut Arena, e: ExprRef, to: &Type, shape: &[(String, usize)], brand: Option<u64>) -> ExprRef {
    let tmp = "__check_tmp".to_string();
    let tmp_ref = arena.push(Expr::Var(tmp.clone()));

    let pairs: Vec<ExprRef> = shape
        .iter()
        .map(|(name, len)| {
            let tag = arena.push(Expr::Str(name.clone()));
            let len_expr = arena.push(Expr::Int(*len as i64));
            arena.push(Expr::ListLit(vec![tag, len_expr]))
        })
        .collect();
    let witness = arena.push(Expr::ListLit(pairs));
    let brand_expr = match brand {
        Some(id) => arena.push(Expr::Token(id)),
        None => arena.push(Expr::Bool(false)),
    };

    let pred_var = arena.push(Expr::Var("check_data_shape".to_string()));
    let call1 = arena.push(Expr::App(pred_var, tmp_ref));
    let call2 = arena.push(Expr::App(call1, witness));
    let pred_call = arena.push(Expr::App(call2, brand_expr));

    let fail_call = build_fail_call(arena, to, tmp_ref);
    let if_expr = arena.push(Expr::If(pred_call, tmp_ref, fail_call));
    arena.push(Expr::Let(tmp, None, e, if_expr))
}

// `fail("type error: expected {to}, found " ++ type_name(value_ref))` --
// shared by both check builders above, so the message format (and the
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
// higher-order function returning another Data/Fun value gets that same
// value's own proper check, not just a shallow tag test). This is a real
// higher-order contract (each call re-validated), not a one-time tag
// check -- a value that merely looks like a function can't smuggle a
// wrong return type through it. Built entirely from existing Expr nodes
// (Let/Lambda/If/App/Var/BinOp), no new Value representation needed. No
// span bookkeeping here: these nodes are synthesized, not sourced from
// the program text, and typecheck never looks up a span for them (see
// TypeError's doc comment).
fn wrap_fun_contract(arena: &mut Arena, e: ExprRef, param_ty: Rc<Type>, ret_ty: Rc<Type>, fields: &[Rc<DataInfo>]) -> ExprRef {
    let fn_var = "__contract_fn".to_string();
    let arg_var = "__contract_arg".to_string();
    let fn_var_ref = arena.push(Expr::Var(fn_var.clone()));
    let arg_var_ref = arena.push(Expr::Var(arg_var.clone()));
    let checked_fn = build_shallow_check(arena, fn_var_ref, &any_fun(), "is_fun");
    let call = arena.push(Expr::App(checked_fn, arg_var_ref));
    let checked_call = build_boundary_check(arena, call, &ret_ty, fields);
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
// element type from the pattern alone, so Dyn there too. A constructor
// pattern (ctor_tag) is ALSO Dyn, not List(Dyn) -- it could belong to any
// `data` type (pattern_type has no name-resolution pass to know which),
// so it needs to stay consistent with a nominal Data(name) scrutinee too,
// not just a structural List one.
fn pattern_type(pat: &Pattern) -> Type {
    match pat {
        Pattern::Var(_) => Type::Dyn,
        Pattern::Int(_) => Type::Int,
        Pattern::Bool(_) => Type::Bool,
        Pattern::Str(_) => Type::Str,
        // Never written by a user (see Pattern::Token's own doc comment)
        // -- only ever appears as the trailing element of a List pattern
        // an opaque ctor's own desugaring built, which is typed via that
        // List pattern's own arm below, not this one directly.
        Pattern::Token(_) => Type::Dyn,
        _ if ctor_tag(pat).is_some() => Type::Dyn,
        Pattern::List(_) | Pattern::Cons(..) => Type::List(Rc::new(Type::Dyn)),
        Pattern::NamedCtor(..) => {
            unreachable!("resolve_pattern always rewrites NamedCtor into List before a Match's patterns reach here")
        }
    }
}

// Extends `ctx` with every Var this pattern binds, each as Dyn (renno has
// no pattern-driven type refinement -- e.g. a List pattern's element
// bindings don't learn the list's element type). "_" is just an ordinary
// Var name here, bound like any other.
fn bind_pattern_vars(ctx: &Ctx, pat: &Pattern) -> Ctx {
    match pat {
        Pattern::Var(name) => extend(ctx, name, Type::Dyn),
        Pattern::Int(_) | Pattern::Bool(_) | Pattern::Str(_) | Pattern::Token(_) => ctx.clone(),
        Pattern::List(pats) => pats.iter().fold(ctx.clone(), |c, p| bind_pattern_vars(&c, p)),
        Pattern::Cons(head, tail) => bind_pattern_vars(&bind_pattern_vars(ctx, head), tail),
        Pattern::NamedCtor(..) => {
            unreachable!("resolve_pattern always rewrites NamedCtor into List before a Match's patterns reach here")
        }
    }
}

// The literal tag at the front of an ADT constructor pattern -- a
// Pattern::List whose first element is a Pattern::Str (see parser.rs's
// pattern_atom, which builds exactly this shape for `Some(x)`/`None`).
// Anything else (a raw list pattern with no such tag, a literal, Var,
// Cons) isn't a constructor pattern, so None.
fn ctor_tag(pat: &Pattern) -> Option<&str> {
    match pat {
        Pattern::List(items) => match items.first() {
            Some(Pattern::Str(tag)) => Some(tag.as_str()),
            _ => None,
        },
        _ => None,
    }
}

// Rewrites a Pattern::NamedCtor into the ordinary Pattern::List shape
// every other pattern-consuming function here already understands --
// recursing through List/Cons so a NamedCtor nested anywhere inside a
// larger pattern (`Some(Point { x: a, y: b })`) still gets resolved, not
// just one at the top level. This is the ONLY place `field: name` order
// gets reconciled with the constructor's own DECLARED order (`data
// Point = Point(x: Int, y: Int)`), which is why it needs `fields` --
// something no other Pattern-consuming function in this file does. `span`
// is the enclosing Match's own span (patterns carry no span of their
// own), used for every error this can raise: an unknown constructor tag,
// a field named twice, an unknown field name, or a missing one.
// Appends a branded type's hidden trailing tag to a ctor-shaped pattern's
// `items`, if it has one -- the one piece both resolve_pattern's NamedCtor
// arm and FieldAccess's synthetic pattern need identically (each already
// has its own `info: &DataInfo` in scope), kept in one place so the two
// non-adjacent call sites can't independently drift on how it's encoded.
fn push_brand_tag(items: &mut Vec<Pattern>, brand: Option<u64>) {
    if let Some(id) = brand {
        items.push(Pattern::Token(id));
    }
}

fn resolve_pattern(pat: &Pattern, fields: &[Rc<DataInfo>], span: Span) -> Result<Pattern, TypeError> {
    match pat {
        Pattern::Var(_) | Pattern::Int(_) | Pattern::Bool(_) | Pattern::Str(_) | Pattern::Token(_) => Ok(pat.clone()),
        Pattern::List(items) => {
            Ok(Pattern::List(items.iter().map(|p| resolve_pattern(p, fields, span)).collect::<Result<_, _>>()?))
        }
        Pattern::Cons(head, tail) => Ok(Pattern::Cons(
            Box::new(resolve_pattern(head, fields, span)?),
            Box::new(resolve_pattern(tail, fields, span)?),
        )),
        Pattern::NamedCtor(tag, named) => {
            // `.rev()`: `fields` accumulates outer-to-inner as `data`
            // blocks are peeled (elaborate's DataGroup arm just pushes,
            // never replaces), so a later, shadowing `data` block that
            // reuses `tag` sits AFTER the one it shadows. Searching from
            // the end picks the lexically-current declaration instead of
            // the first (already-shadowed) one.
            let info = fields
                .iter()
                .rev()
                .find(|f| f.ctors.iter().any(|(n, field_names)| n == tag && !field_names.is_empty()))
                .ok_or_else(|| TypeError(format!("no `data` type has a constructor `{tag}` with named fields"), span))?;
            let (_, field_names) = info.ctors.iter().find(|(n, _)| n == tag).unwrap();

            let mut by_name: HashMap<&str, &Pattern> = HashMap::new();
            for (name, p) in named {
                if by_name.insert(name.as_str(), p).is_some() {
                    return Err(TypeError(format!("field `{name}` given more than once"), span));
                }
            }
            for (name, _) in named {
                if !field_names.iter().any(|f| f == name) {
                    return Err(TypeError(format!("{} has no field named `{name}`", info.type_name), span));
                }
            }
            let mut items = vec![Pattern::Str(tag.clone())];
            for fname in field_names {
                let p = by_name
                    .get(fname.as_str())
                    .ok_or_else(|| TypeError(format!("missing field `{fname}` in `{tag}` pattern"), span))?;
                items.push(resolve_pattern(p, fields, span)?);
            }
            // Branded types stamp one hidden trailing element into every
            // constructed value (see DataInfo::brand, build_ctor_value) --
            // match it here too, strictly, so a named-field pattern for
            // this type only ever matches a value this EXACT `data` block
            // produced, the same defense positional patterns get via
            // Parser::branded_ctors.
            push_brand_tag(&mut items, info.brand);
            Ok(Pattern::List(items))
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
// List/Cons/constructor patterns are NOT compared for subsumption here
// (e.g. an earlier unconstrained `h :: t` making a later `1 :: t`
// unreachable) -- a real but more nuanced case than this function
// attempts; a redundant arm of that shape is silently allowed, same as
// any other question this checker can't cheaply answer.
fn dominates(earlier: &Pattern, later: &Pattern) -> bool {
    match earlier {
        Pattern::Var(_) => true,
        Pattern::Int(n) => matches!(later, Pattern::Int(m) if m == n),
        Pattern::Bool(b) => matches!(later, Pattern::Bool(c) if c == b),
        Pattern::Str(s) => matches!(later, Pattern::Str(t) if t == s),
        _ => false,
    }
}

// The index of the first arm some EARLIER arm already fully dominates, if
// any -- reported as unreachable (dead code).
fn first_unreachable(patterns: &[Pattern]) -> Option<usize> {
    (0..patterns.len()).find(|&i| (0..i).any(|j| dominates(&patterns[j], &patterns[i])))
}

// Exhaustiveness is checked only where it's cheaply provable -- same stance
// as everywhere else in this checker (effect rows, list-length arity):
// stay silent and defer to match_pattern's own runtime panic rather than
// try to enumerate an open domain (Int and Str literals, in particular,
// never close). Five shapes are recognized as covering every possible
// value:
//   - any Var (including "_") pattern present, anywhere -- matches
//     everything by itself.
//   - Tuple: a fixed-length, all-Var/wildcard pattern of exactly the
//     scrutinee's own known arity (only meaningful when `scrut_ty` IS a
//     Tuple -- its length is fixed and known, unlike a general List's).
//   - Bool: both `true` and `false` literals present.
//   - List: both `[]` and an unconstrained `h :: t` present (`h`/`t`
//     themselves must be Var -- `1 :: t` only covers SOME non-empty
//     lists, not all of them).
//   - a `data`-declared type: every arm's pattern is an ADT constructor
//     tag (ctor_tag), and that exact set of tags matches one of the
//     constructor sets `data` recorded (see Expr::DataGroup).
// Anything else is reported as possibly non-exhaustive: Int/Str literals
// with no catch-all, a List match using only fixed-length patterns, or an
// ADT match that doesn't cover every constructor its `data` declared.
//
// Does `pat` cover every possible value at a position statically known to
// have type `ty`? Only ever called with `ty` a Tuple (missing_case's own
// guard), but recurses into NESTED tuple positions too: `((p, q), x)`
// covers all of `((Int, Int), Int)` even though NEITHER top-level
// sub-pattern is a bare Var, because the first one is itself a
// fully-covering pattern for ITS OWN (also Tuple) position. Does NOT help
// a scrutinee bound by an outer match/lambda pattern first (renno has no
// pattern-driven type refinement -- see bind_pattern_vars's own doc
// comment -- so that binding's own type is just Dyn, not the precise
// Tuple type this needs): a fully-nested pattern in ONE match sidesteps
// that, a match on a separately-destructured intermediate variable does
// not.
fn covers_tuple_position(pat: &Pattern, ty: &Type) -> bool {
    match (pat, ty) {
        (Pattern::Var(_), _) => true,
        (Pattern::List(subpats), Type::Tuple(items)) if subpats.len() == items.len() => {
            subpats.iter().zip(items.iter()).all(|(sp, t)| covers_tuple_position(sp, t))
        }
        _ => false,
    }
}

fn missing_case(patterns: &[Pattern], fields: &[Rc<DataInfo>], scrut_ty: &Type) -> Option<String> {
    if patterns.iter().any(|p| matches!(p, Pattern::Var(_))) {
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
    if let Type::Tuple(_) = scrut_ty {
        let covers_every_tuple = patterns.iter().any(|p| covers_tuple_position(p, scrut_ty));
        if covers_every_tuple {
            return None;
        }
    }

    // A Union's alternatives are independent -- unlike Tuple, no SINGLE
    // pattern needs to cover the whole thing; exhaustive means every
    // alternative has SOME pattern (not necessarily the same one) that
    // fully covers it. `type Option = (Token,) | (Token, Int) in ... |
    // (t,) -> ... | (t, x) -> ...` is exhaustive this way even though
    // neither arm alone would satisfy Tuple's own single-pattern check.
    if let Type::Union(alts) = scrut_ty {
        let covers_every_alt = alts.iter().all(|alt| patterns.iter().any(|p| covers_tuple_position(p, alt)));
        if covers_every_alt {
            return None;
        }
    }

    let has_true = patterns.iter().any(|p| matches!(p, Pattern::Bool(true)));
    let has_false = patterns.iter().any(|p| matches!(p, Pattern::Bool(false)));
    if has_true && has_false {
        return None;
    }

    let has_nil = patterns.iter().any(|p| matches!(p, Pattern::List(items) if items.is_empty()));
    let has_full_cons = patterns.iter().any(
        |p| matches!(p, Pattern::Cons(h, t) if matches!(**h, Pattern::Var(_)) && matches!(**t, Pattern::Var(_))),
    );
    if has_nil && has_full_cons {
        return None;
    }

    let tags: Option<BTreeSet<String>> = patterns.iter().map(|p| ctor_tag(p).map(str::to_string)).collect();
    if let Some(tags) = &tags {
        let matches_some_data_type = fields.iter().any(|info| {
            let its_tags: BTreeSet<String> = info.ctors.iter().map(|(name, _)| name.clone()).collect();
            &its_tags == tags
        });
        if matches_some_data_type {
            return None;
        }
    }

    Some(if has_true || has_false {
        "the missing Bool case (add `true`/`false`, or a wildcard `_` arm)".to_string()
    } else if has_nil || has_full_cons {
        "the missing List case (add `[]`/`h :: t`, or a wildcard `_` arm)".to_string()
    } else if tags.is_some() {
        "a constructor this `data` declared (add it, or a wildcard `_` arm)".to_string()
    } else {
        "some value not covered by any arm (add a wildcard `_` arm)".to_string()
    })
}

// Bidirectional-lite synthesis: walks the tree once, producing the
// inferred Type, the inferred EffectRow (closed, no polymorphism -- just
// the union of effect names this expression's evaluation might perform),
// and an elaborated Expr (same shape, with Check nodes -- or, at Fun
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
fn elaborate(
    arena: &mut Arena,
    expr: ExprRef,
    ctx: &Ctx,
    fields: &[Rc<DataInfo>],
    spans: &SpanMap,
) -> Result<(Type, EffectRow, ExprRef), TypeError> {
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
    let mut cur_fields: Vec<Rc<DataInfo>> = fields.to_vec();
    loop {
        // Expr is Clone and, now that its fields are ExprRef (Copy)
        // instead of Rc<Expr>, cheap to clone -- this ends the borrow on
        // `arena` before the arm below needs to mutate it.
        let node = arena[cur_expr].clone();
        match node {
            Expr::Let(var, ann, val, body) => {
                let (val_ty, val_row, val2) = elaborate(arena, val, &cur_ctx, &cur_fields, spans)?;
                let (bound_ty, val3) = match ann {
                    Some(t) => (t.clone(), coerce(arena, val2, &val_ty, &t, spans[val], &cur_fields)?),
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
                    let (val_ty, val_row, val2) = elaborate(arena, *val, &val_ctx, &cur_fields, spans)?;
                    let (bound_ty, val3) = match ann {
                        Some(t) => (t.clone(), coerce(arena, val2, &val_ty, t, spans[*val], &cur_fields)?),
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
            // Transparent marker for a `data` declaration (see
            // parser::build_ctor_value) -- peeled here, not left as a
            // plain elaborate_node case, for the same reason Let/Lambda
            // are: a long chain of `data` blocks costs O(1) native stack,
            // not O(chain length). Nothing folds back for it: the
            // elaborated tree drops this node entirely -- machine.rs's own
            // passthrough arm only matters on the un-typechecked path
            // (e.g. tests' run_untyped). Its only job here is extending
            // `cur_fields` so a later Match in `body` can check
            // exhaustiveness against it (missing_case), and a later
            // FieldAccess can resolve a `.field` against it.
            Expr::DataGroup(info, body) => {
                cur_fields.push(info);
                cur_expr = body;
            }
            _ => break,
        }
    }

    let (mut result_ty, mut result_row, mut result_expr) =
        elaborate_node(arena, cur_expr, &cur_ctx, &cur_fields, spans)?;

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

// Every Expr variant except Let/Lambda/DataGroup, which `elaborate` peels
// off iteratively above -- reached only once no more chain prefix remains.
// `expr` (this function's own parameter) is always the ORIGINAL,
// pre-elaboration ExprRef for whatever's currently being checked, so
// `spans[expr]` is a valid, always-available "point at this whole
// construct" location for any error an arm below doesn't have a more
// specific sub-expression to blame instead.
fn elaborate_node(
    arena: &mut Arena,
    expr: ExprRef,
    ctx: &Ctx,
    fields: &[Rc<DataInfo>],
    spans: &SpanMap,
) -> Result<(Type, EffectRow, ExprRef), TypeError> {
    let node = arena[expr].clone();
    match node {
        Expr::Int(_) => Ok((Type::Int, EffectRow::pure(), expr)),
        Expr::Bool(_) => Ok((Type::Bool, EffectRow::pure(), expr)),
        Expr::Str(_) => Ok((Type::Str, EffectRow::pure(), expr)),
        // A singleton per source position -- see Expr::Token's own doc
        // comment. Also reached for the ctor-field-list marker's own
        // auto-generated brand element (inside build_ctor_value's
        // ListLit), where its precise type doesn't actually matter: that
        // ListLit's own elem_ty already widens to Dyn from mixing the Str
        // tag with the constructor's other field types in every case that
        // arises in practice, so consistent_inner's List/Data bridging
        // arm (Str/Dyn only) stays satisfied regardless of what this one
        // trailing element types as.
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
                let (item_ty, item_row, item2) = elaborate(arena, item, ctx, fields, spans)?;
                row = EffectRow::union(&row, &item_row);
                tys.push(item_ty);
                refs.push(item2);
            }
            Ok((Type::Tuple(Rc::new(tys)), row, arena.push(Expr::Tuple(refs))))
        }

        Expr::ListLit(items) => {
            let mut row = EffectRow::pure();
            let mut elem_ty: Option<Type> = None;
            let mut refs = Vec::with_capacity(items.len());
            for item in items {
                let (item_ty, item_row, item2) = elaborate(arena, item, ctx, fields, spans)?;
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

        Expr::Let(..) | Expr::LetRec(..) | Expr::Lambda(..) | Expr::DataGroup(..) => {
            unreachable!("Let/LetRec/Lambda/DataGroup are peeled by elaborate's chain-flattening loop")
        }

        Expr::App(f, a) => {
            let (f_ty, f_row, f2) = elaborate(arena, f, ctx, fields, spans)?;
            let (a_ty, a_row, a2) = elaborate(arena, a, ctx, fields, spans)?;
            let (call_row, ret_ty, app2) = match &f_ty {
                Type::Fun(param_ty, call_row, ret_ty) => {
                    let a3 = coerce(arena, a2, &a_ty, param_ty, spans[a], fields)?;
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
            let (l_ty, l_row, l2) = elaborate(arena, l, ctx, fields, spans)?;
            let (r_ty, r_row, r2) = elaborate(arena, r, ctx, fields, spans)?;
            let row = EffectRow::union(&l_row, &r_row);
            match op {
                // Arithmetic and ordering: both operands must be Int.
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Lt => {
                    let l3 = coerce(arena, l2, &l_ty, &Type::Int, spans[l], fields)?;
                    let r3 = coerce(arena, r2, &r_ty, &Type::Int, spans[r], fields)?;
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
                    if !consistent(&l_ty, &r_ty, fields) {
                        return Err(TypeError(
                            format!("type mismatch: cannot compare {l_ty} with {r_ty}"),
                            spans[expr],
                        ));
                    }
                    let l3 = if l_ty == Type::Dyn && r_ty != Type::Dyn {
                        coerce(arena, l2, &l_ty, &r_ty, spans[l], fields)?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty, spans[r], fields)?
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
                    if !consistent(&l_ty, &r_ty, fields) {
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
                        coerce(arena, l2, &l_ty, &r_ty, spans[l], fields)?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty, spans[r], fields)?
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
                    if !consistent(&r_ty, &list_of_dyn, fields) {
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
            let (c_ty, c_row, c2) = elaborate(arena, c, ctx, fields, spans)?;
            let c3 = coerce(arena, c2, &c_ty, &Type::Bool, spans[c], fields)?;
            let (t_ty, t_row, t2) = elaborate(arena, t, ctx, fields, spans)?;
            let (e_ty, e_row, e2) = elaborate(arena, e, ctx, fields, spans)?;
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
            let (_, payload_row, payload2) = elaborate(arena, payload, ctx, fields, spans)?;
            let row = EffectRow::union(&payload_row, &EffectRow::single(&effect));
            Ok((Type::Dyn, row, arena.push(Expr::Perform(effect, payload2))))
        }

        Expr::Handle { body, handler } => {
            let (_, body_row, body2) = elaborate(arena, body, ctx, fields, spans)?;
            let (handler_ty, handler_row, handler2) = elaborate(arena, handler, ctx, fields, spans)?;
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
            // Named-field patterns (Pattern::NamedCtor) are resolved into
            // ordinary List/Str shapes right here, first -- everything
            // below (reachability, exhaustiveness, per-arm binding, and
            // the pattern finally stored in the elaborated tree) works on
            // the resolved form, the only one machine.rs ever understands.
            let pats: Vec<Pattern> =
                arms.iter().map(|(p, _)| resolve_pattern(p, fields, spans[expr])).collect::<Result<_, _>>()?;
            if let Some(i) = first_unreachable(&pats) {
                return Err(TypeError(
                    "unreachable match arm: an earlier arm already covers everything it matches".to_string(),
                    spans[arms[i].1],
                ));
            }

            let (scrut_ty, scrut_row, scrutinee2) = elaborate(arena, scrutinee, ctx, fields, spans)?;
            let mut row = scrut_row;
            let mut result_ty: Option<Type> = None;
            let mut new_arms = Vec::with_capacity(arms.len());
            for (pat, (_, body)) in pats.iter().zip(arms.iter()) {
                let pat_ty = pattern_type(pat);
                if !consistent(&scrut_ty, &pat_ty, fields) {
                    return Err(TypeError(
                        format!("match: pattern of type {pat_ty} can never match scrutinee of type {scrut_ty}"),
                        spans[expr],
                    ));
                }
                let arm_ctx = bind_pattern_vars(ctx, pat);
                let (arm_ty, arm_row, body2) = elaborate(arena, *body, &arm_ctx, fields, spans)?;
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
            if let Some(missing) = missing_case(&pats, fields, &scrut_ty) {
                return Err(TypeError(format!("non-exhaustive match: {missing}"), spans[expr]));
            }
            Ok((result_ty.unwrap_or(Type::Dyn), row, arena.push(Expr::Match(scrutinee2, Rc::new(new_arms)))))
        }

        // Constructing the handler value is pure -- the clause body's own
        // effects (including what `resume` re-enters) aren't modeled here;
        // see the doc comment on `elaborate`.
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let inner_ctx = extend(&extend(ctx, &payload_var, Type::Dyn), &resume_var, Type::Dyn);
            let (_, _, body2) = elaborate(arena, body, &inner_ctx, fields, spans)?;
            Ok((
                Type::Dyn,
                EffectRow::pure(),
                arena.push(Expr::MakeHandler { effect, payload_var, resume_var, body: body2 }),
            ))
        }

        // `target.field` -- only resolvable when target's type is
        // concretely known as a nominal Data(name) whose `data` block
        // named this field (see DataInfo) and has EXACTLY one
        // constructor: with more than one, which constructor's field
        // would `.field` even mean? Desugars into an ordinary Match
        // against that sole constructor, with a Var at the target
        // field's position and "_" everywhere else -- reuses existing
        // pattern-matching machinery entirely, so machine.rs needs no new
        // opcode and this Match is exhaustive/unambiguous by
        // construction (skipping first_unreachable/missing_case, which
        // are for programs a human wrote, not ones this function builds
        // knowing they're already correct).
        Expr::FieldAccess(target, field) => {
            let (target_ty, target_row, target2) = elaborate(arena, target, ctx, fields, spans)?;
            let (type_name, brand) = match &target_ty {
                Type::Data(name, brand) => (name.clone(), *brand),
                other => {
                    return Err(TypeError(
                        format!("cannot access field `{field}` on a value of type {other} (expected a `data` type)"),
                        spans[target],
                    ))
                }
            };
            let info = lookup_data(fields, &type_name, brand).ok_or_else(|| {
                TypeError(format!("cannot access field `{field}`: no known fields for type {type_name}"), spans[target])
            })?;
            if info.ctors.len() != 1 {
                return Err(TypeError(
                    format!(
                        "cannot access field `{field}` on {type_name}: field access needs a `data` type with exactly one constructor, but {type_name} has {}",
                        info.ctors.len()
                    ),
                    spans[target],
                ));
            }
            let (ctor_name, field_names) = &info.ctors[0];
            if field_names.is_empty() {
                return Err(TypeError(
                    format!("cannot access field `{field}` on {type_name}: its constructor has no named fields"),
                    spans[target],
                ));
            }
            let idx = field_names.iter().position(|f| *f == field).ok_or_else(|| {
                // Unlike the errors above (about the RECEIVER's type),
                // this one's about the access itself, so it points at the
                // whole `target.field`, not just `target`.
                TypeError(format!("{type_name} has no field named `{field}`"), spans[expr])
            })?;

            let mut items = vec![Pattern::Str(ctor_name.clone())];
            for i in 0..field_names.len() {
                items.push(Pattern::Var(if i == idx { "__field".to_string() } else { "_".to_string() }));
            }
            // Strict, not a wildcard, for the same defense-in-depth reason
            // resolve_pattern's NamedCtor arm is strict: `target`'s type
            // is already known statically, but a Dyn-sourced value could
            // still reach here having slipped past a shallow runtime
            // Check -- matching the exact brand catches that instead of
            // silently reading a field off the wrong declaration's value.
            push_brand_tag(&mut items, info.brand);
            let var_ref = arena.push(Expr::Var("__field".to_string()));
            let arms = Rc::new(vec![(Pattern::List(items), var_ref)]);
            let match_expr = arena.push(Expr::Match(target2, arms));
            Ok((Type::Dyn, target_row, match_expr))
        }

        // `Ctor { field: expr, ... }` -- only resolvable, like FieldAccess,
        // once `callee`'s type is concretely known as a nominal
        // Data(name) with EXACTLY one constructor whose fields are all
        // named. Reorders the given fields into the constructor's own
        // DECLARED order and desugars into the same curried Application
        // chain `Ctor(a)(b)` already produces -- reusing App's existing
        // per-argument coercion (coerce) rather than duplicating it.
        // Constructors never have row-polymorphic parameter types
        // (parser::ctor_type always builds EffectRow::pure() arrows), so
        // unlike ordinary App there's no row-variable binding to do here.
        Expr::NamedCall(callee, named_args) => {
            let (callee_ty, callee_row, callee2) = elaborate(arena, callee, ctx, fields, spans)?;
            let mut param_tys = Vec::new();
            let mut cur_ty = &callee_ty;
            let (type_name, brand) = loop {
                match cur_ty {
                    Type::Fun(p, _, ret) => {
                        param_tys.push((**p).clone());
                        cur_ty = ret.as_ref();
                    }
                    Type::Data(name, brand) => break (name.clone(), *brand),
                    other => {
                        return Err(TypeError(
                            format!("cannot use `{{ field: value, ... }}` syntax on a value of type {other} (expected a `data` constructor)"),
                            spans[callee],
                        ))
                    }
                }
            };
            let info = lookup_data(fields, &type_name, brand).ok_or_else(|| {
                TypeError(format!("cannot use `{{ ... }}` syntax: no known fields for type {type_name}"), spans[callee])
            })?;
            if info.ctors.len() != 1 {
                return Err(TypeError(
                    format!(
                        "cannot use `{{ ... }}` syntax on {type_name}: it has {} constructors, not exactly one",
                        info.ctors.len()
                    ),
                    spans[callee],
                ));
            }
            let (_, field_names) = &info.ctors[0];
            if field_names.is_empty() {
                return Err(TypeError(
                    format!("cannot use `{{ ... }}` syntax on {type_name}: its constructor has no named fields"),
                    spans[callee],
                ));
            }

            let mut by_name: HashMap<&str, ExprRef> = HashMap::new();
            for (name, val) in named_args.iter() {
                if by_name.insert(name.as_str(), *val).is_some() {
                    return Err(TypeError(format!("field `{name}` given more than once"), spans[expr]));
                }
            }
            for (name, _) in named_args.iter() {
                if !field_names.iter().any(|f| f == name) {
                    return Err(TypeError(format!("{type_name} has no field named `{name}`"), spans[expr]));
                }
            }

            let mut result = callee2;
            let mut row = callee_row;
            for (fname, param_ty) in field_names.iter().zip(param_tys.iter()) {
                let val = *by_name
                    .get(fname.as_str())
                    .ok_or_else(|| TypeError(format!("missing field `{fname}` in {type_name} construction"), spans[expr]))?;
                let (a_ty, a_row, a2) = elaborate(arena, val, ctx, fields, spans)?;
                let a3 = coerce(arena, a2, &a_ty, param_ty, spans[val], fields)?;
                row = EffectRow::union(&row, &a_row);
                result = arena.push(Expr::App(result, a3));
            }
            Ok((Type::Data(type_name, brand), row, result))
        }
    }
}

pub fn check(arena: &mut Arena, root: ExprRef, spans: &SpanMap) -> Result<ExprRef, TypeError> {
    let (_, row, elaborated) = elaborate(arena, root, &Ctx::empty(), &[], spans)?;
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
