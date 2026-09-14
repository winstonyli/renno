use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::expr::{Arena, BinOp, Expr, ExprRef, Pattern, SpanMap};
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
        Type::Dyn | Type::Int | Type::Bool | Type::Str | Type::Data(_) => BTreeSet::new(),
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

// The only place a runtime Check gets inserted: `from` is Dyn (unknown
// statically) and `to` is concrete. If both sides are concrete and
// disagree, that's a real static error -- reject before running at all.
// If `from` is already exactly consistent and concrete, no check needed:
// zero overhead for fully-annotated code. The error, if any, points at
// `span` -- the CALLER's job to have already looked up via the ORIGINAL
// (pre-elaboration) ExprRef for the value in question, e.g. `spans[val]`
// where `val` is a field straight off the un-elaborated node, NOT the
// elaborated `e` this function receives to potentially wrap: elaborate_node
// re-pushes almost every compound node it touches (App, BinOp, If, ...)
// regardless of whether anything actually changed, so `e` itself is often
// already a brand new ExprRef past the end of the parser-built SpanMap by
// the time it reaches here -- indexing spans BY IT, not by the original,
// was a latent bug (worked by coincidence whenever the mismatched value
// happened to be a leaf that elaborate_node returns unchanged).
//
// Crossing into a Fun type is special: value::matches_type only confirms
// "this is callable," not "callable with this exact signature" (a tag
// check can't see inside a closure). So a Dyn value flowing into an
// annotated Fun position gets wrapped in a real per-call contract instead
// of a bare tag Check -- see wrap_fun_contract.
fn coerce(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type, span: Span) -> Result<ExprRef, TypeError> {
    if !consistent(from, to) {
        return Err(TypeError(format!("type mismatch: expected {to}, found {from}"), span));
    }
    if *from != Type::Dyn || *to == Type::Dyn {
        return Ok(e);
    }
    match to {
        Type::Fun(param_ty, _row, ret_ty) => Ok(wrap_fun_contract(arena, e, param_ty.clone(), ret_ty.clone())),
        _ => Ok(arena.push(Expr::Check(to.clone(), e))),
    }
}

// Wraps a Dyn-origin value in a fresh closure that, on every call: checks
// the argument matches param_ty (via the wrapper's own declared param type
// -- ordinary App-site coercion at the wrapper's call sites handles that),
// confirms the wrapped value is actually callable, applies it, then checks
// the result against ret_ty. This is a real higher-order contract (each
// call re-validated), not a one-time tag check -- a value that merely
// looks like a function can't smuggle a wrong return type through it.
// Built entirely from existing Expr nodes (Let/Lambda/Check/App/Var), no
// new Value representation needed. No span bookkeeping here: these nodes
// are synthesized, not sourced from the program text, and typecheck never
// looks up a span for them (see TypeError's doc comment).
fn wrap_fun_contract(arena: &mut Arena, e: ExprRef, param_ty: Rc<Type>, ret_ty: Rc<Type>) -> ExprRef {
    let fn_var = "__contract_fn".to_string();
    let arg_var = "__contract_arg".to_string();
    let fn_var_ref = arena.push(Expr::Var(fn_var.clone()));
    let arg_var_ref = arena.push(Expr::Var(arg_var.clone()));
    let checked_fn = arena.push(Expr::Check(any_fun(), fn_var_ref));
    let call = arena.push(Expr::App(checked_fn, arg_var_ref));
    let checked_call = arena.push(Expr::Check((*ret_ty).clone(), call));
    let lambda = arena.push(Expr::Lambda(arg_var, Some((*param_ty).clone()), checked_call));
    arena.push(Expr::Let(fn_var, None, e, lambda, false))
}

// A `let`/`fun` prefix collected while flattening a chain of them (see
// `elaborate`) -- deferred until the terminal body is elaborated, then
// folded back into nested Let/Lambda nodes (and their types/rows) in
// reverse, in the exact shape their original per-node match arms produced.
enum PendingElab {
    Let { var: String, bound_ty: Type, val_row: EffectRow, val: ExprRef, rec: bool },
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
        _ if ctor_tag(pat).is_some() => Type::Dyn,
        Pattern::List(_) | Pattern::Cons(..) => Type::List(Rc::new(Type::Dyn)),
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
// never close). Four shapes are recognized as covering every possible
// value:
//   - any Var (including "_") pattern present, anywhere -- matches
//     everything by itself.
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
fn missing_case(patterns: &[Pattern], groups: &[BTreeSet<String>]) -> Option<String> {
    if patterns.iter().any(|p| matches!(p, Pattern::Var(_))) {
        return None;
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
        if groups.iter().any(|g| g == tags) {
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
    groups: &[BTreeSet<String>],
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
    let mut cur_groups: Vec<BTreeSet<String>> = groups.to_vec();
    loop {
        // Expr is Clone and, now that its fields are ExprRef (Copy)
        // instead of Rc<Expr>, cheap to clone -- this ends the borrow on
        // `arena` before the arm below needs to mutate it.
        let node = arena[cur_expr].clone();
        match node {
            Expr::Let(var, ann, val, body, rec) => {
                // `rec`: `var` needs to resolve to its own (eventual) type
                // WHILE elaborating `val`, for a self-reference inside it
                // to type-check precisely rather than falling back to Dyn.
                // That's only possible if the type is already known, i.e.
                // annotated -- an unannotated `let rec` still works (the
                // self-reference is just Dyn, like any other unbound-at-
                // this-point lookup), it's just not precisely typed.
                let val_ctx = match (rec, &ann) {
                    (true, Some(t)) => extend(&cur_ctx, &var, t.clone()),
                    _ => cur_ctx.clone(),
                };
                let (val_ty, val_row, val2) = elaborate(arena, val, &val_ctx, &cur_groups, spans)?;
                let (bound_ty, val3) = match ann {
                    Some(t) => (t.clone(), coerce(arena, val2, &val_ty, &t, spans[val])?),
                    None => (val_ty, val2),
                };
                cur_ctx = extend_generalized(&cur_ctx, &var, bound_ty.clone());
                pending.push(PendingElab::Let { var, bound_ty, val_row, val: val3, rec });
                cur_expr = body;
            }
            Expr::Lambda(param, ann, body) => {
                let param_ty = ann.unwrap_or(Type::Dyn);
                cur_ctx = extend(&cur_ctx, &param, param_ty.clone());
                pending.push(PendingElab::Fun { param, param_ty });
                cur_expr = body;
            }
            // Transparent marker for a `data` declaration's constructor tag
            // set (see parser::build_ctor_value) -- peeled here, not left
            // as a plain elaborate_node case, for the same reason Let/
            // Lambda are: a long chain of `data` blocks costs O(1) native
            // stack, not O(chain length). Nothing folds back for it: the
            // elaborated tree drops this node entirely -- machine.rs's own
            // passthrough arm only matters on the un-typechecked path
            // (e.g. tests' run_untyped). Its only job here is extending
            // `cur_groups` so a later Match in `body` can check
            // exhaustiveness against it (see missing_case).
            Expr::DataGroup(tags, body) => {
                cur_groups.push((*tags).clone());
                cur_expr = body;
            }
            _ => break,
        }
    }

    let (mut result_ty, mut result_row, mut result_expr) =
        elaborate_node(arena, cur_expr, &cur_ctx, &cur_groups, spans)?;

    for frame in pending.into_iter().rev() {
        match frame {
            PendingElab::Let { var, bound_ty, val_row, val, rec } => {
                // Matches the original Let arm: body's type propagates
                // through unchanged, row is the union of val's and body's.
                result_row = EffectRow::union(&val_row, &result_row);
                result_expr = arena.push(Expr::Let(var, Some(bound_ty), val, result_expr, rec));
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
    groups: &[BTreeSet<String>],
    spans: &SpanMap,
) -> Result<(Type, EffectRow, ExprRef), TypeError> {
    let node = arena[expr].clone();
    match node {
        Expr::Int(_) => Ok((Type::Int, EffectRow::pure(), expr)),
        Expr::Bool(_) => Ok((Type::Bool, EffectRow::pure(), expr)),
        Expr::Str(_) => Ok((Type::Str, EffectRow::pure(), expr)),
        Expr::Var(name) => Ok((lookup(ctx, &name), EffectRow::pure(), expr)),

        Expr::ListLit(items) => {
            let mut row = EffectRow::pure();
            let mut elem_ty: Option<Type> = None;
            let mut refs = Vec::with_capacity(items.len());
            for item in items {
                let (item_ty, item_row, item2) = elaborate(arena, item, ctx, groups, spans)?;
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

        Expr::Let(..) | Expr::Lambda(..) | Expr::DataGroup(..) => {
            unreachable!("Let/Lambda/DataGroup are peeled by elaborate's chain-flattening loop")
        }

        Expr::App(f, a) => {
            let (f_ty, f_row, f2) = elaborate(arena, f, ctx, groups, spans)?;
            let (a_ty, a_row, a2) = elaborate(arena, a, ctx, groups, spans)?;
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
                    // through the same Check mechanism everything else
                    // uses, rather than leaving it to a differently-worded
                    // panic in machine.rs. Can't know what it might
                    // perform, so the call contributes an unknown (Dyn) row.
                    let f3 = arena.push(Expr::Check(any_fun(), f2));
                    (EffectRow::Dyn, Type::Dyn, arena.push(Expr::App(f3, a2)))
                }
                other => return Err(TypeError(format!("cannot call a value of type {other}"), spans[f])),
            };
            let row = EffectRow::union(&EffectRow::union(&f_row, &a_row), &call_row);
            Ok((ret_ty, row, app2))
        }

        Expr::BinOp(op, l, r) => {
            let (l_ty, l_row, l2) = elaborate(arena, l, ctx, groups, spans)?;
            let (r_ty, r_row, r2) = elaborate(arena, r, ctx, groups, spans)?;
            let row = EffectRow::union(&l_row, &r_row);
            match op {
                // Arithmetic and ordering: both operands must be Int.
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Lt => {
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
            }
        }

        Expr::If(c, t, e) => {
            let (c_ty, c_row, c2) = elaborate(arena, c, ctx, groups, spans)?;
            let c3 = coerce(arena, c2, &c_ty, &Type::Bool, spans[c])?;
            let (t_ty, t_row, t2) = elaborate(arena, t, ctx, groups, spans)?;
            let (e_ty, e_row, e2) = elaborate(arena, e, ctx, groups, spans)?;
            // Branches with differing concrete types aren't an error here
            // (no union types) -- just widen to Dyn rather than reject.
            let result_ty = if t_ty == e_ty { t_ty } else { Type::Dyn };
            // Only one branch runs, but which one isn't known statically,
            // so the possible effects are the union of both.
            let row = EffectRow::union(&c_row, &EffectRow::union(&t_row, &e_row));
            Ok((result_ty, row, arena.push(Expr::If(c3, t2, e2))))
        }

        Expr::Check(ty, inner) => {
            let (_, row, inner2) = elaborate(arena, inner, ctx, groups, spans)?;
            let ty_ret = ty.clone();
            Ok((ty_ret, row, arena.push(Expr::Check(ty, inner2))))
        }

        // The effect this specific operation performs, plus whatever the
        // payload expression itself might perform.
        Expr::Perform(effect, payload) => {
            let (_, payload_row, payload2) = elaborate(arena, payload, ctx, groups, spans)?;
            let row = EffectRow::union(&payload_row, &EffectRow::single(&effect));
            Ok((Type::Dyn, row, arena.push(Expr::Perform(effect, payload2))))
        }

        Expr::Handle { body, handler } => {
            let (_, body_row, body2) = elaborate(arena, body, ctx, groups, spans)?;
            let (handler_ty, handler_row, handler2) = elaborate(arena, handler, ctx, groups, spans)?;
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
            let pats: Vec<Pattern> = arms.iter().map(|(p, _)| p.clone()).collect();
            if let Some(i) = first_unreachable(&pats) {
                return Err(TypeError(
                    "unreachable match arm: an earlier arm already covers everything it matches".to_string(),
                    spans[arms[i].1],
                ));
            }

            let (scrut_ty, scrut_row, scrutinee2) = elaborate(arena, scrutinee, ctx, groups, spans)?;
            let mut row = scrut_row;
            let mut result_ty: Option<Type> = None;
            let mut new_arms = Vec::with_capacity(arms.len());
            for (pat, body) in arms.iter() {
                let pat_ty = pattern_type(pat);
                if !consistent(&scrut_ty, &pat_ty) {
                    return Err(TypeError(
                        format!("match: pattern of type {pat_ty} can never match scrutinee of type {scrut_ty}"),
                        spans[expr],
                    ));
                }
                let arm_ctx = bind_pattern_vars(ctx, pat);
                let (arm_ty, arm_row, body2) = elaborate(arena, *body, &arm_ctx, groups, spans)?;
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
            if let Some(missing) = missing_case(&pats, groups) {
                return Err(TypeError(format!("non-exhaustive match: {missing}"), spans[expr]));
            }
            Ok((result_ty.unwrap_or(Type::Dyn), row, arena.push(Expr::Match(scrutinee2, Rc::new(new_arms)))))
        }

        // Constructing the handler value is pure -- the clause body's own
        // effects (including what `resume` re-enters) aren't modeled here;
        // see the doc comment on `elaborate`.
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let inner_ctx = extend(&extend(ctx, &payload_var, Type::Dyn), &resume_var, Type::Dyn);
            let (_, _, body2) = elaborate(arena, body, &inner_ctx, groups, spans)?;
            Ok((
                Type::Dyn,
                EffectRow::pure(),
                arena.push(Expr::MakeHandler { effect, payload_var, resume_var, body: body2 }),
            ))
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
