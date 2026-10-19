use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::cont::{Cont, Frame, Meta, Node};
use crate::env::{Env, PRELUDE};
use crate::expr::{AnnotKind, AnnotSpec, Arena, BinOp, CheckSpec, Expr, ExprRef, Lazy, Len, Length, LenExpr, Lowered, Pattern, ConformSpec, Seal, SpanMap, Test, Then, TypeSource};
use crate::frame::Bindings;
use crate::index_expr::IndexExpr;
use crate::iterate::{step_shared, Iter, Next};
use crate::list::{List, SealId};
use crate::resolve::{group_kind, resolve, GroupKind, Resolved, VarRef};
use crate::span::Span;
use crate::text::Text;
use crate::types::{first_free, fresh_rec_id, open_record_tail_under, Hc, RecVar, Row, Tail, Type};
use crate::util::find_field;
use crate::value::{Builtin, Contract, HandlerData, PathStep, SiteCtx, TypeCtor, Value};

enum Control {
    Eval(ExprRef, Env),
    Apply(Value),
    Perform(String, Value),
}

thread_local! {
    // Updated every time the trampoline begins evaluating a new expression
    // (see run_loop's Eval arm) -- read back by lib.rs's catch_unwind,
    // AFTER a panic is caught, to attach a "line L, column C" location to
    // whatever runtime panic just fired (unbound variable, a failed
    // Check, a non-function call, `perform` with no handler, ...), no
    // matter which function actually called panic! -- frame.rs's,
    // value.rs's, and machine.rs's own panics all get this for free, with
    // no Span parameter threaded into any of them.
    //
    // Not perfectly precise: a panic several steps removed from the last
    // Eval still reports the last EXPRESSION evaluated, not necessarily the
    // exact sub-part at fault, wherever nothing more specific was threaded
    // through. Two cases common enough to be worth the extra Span fields
    // are handled precisely instead, by threading the RELEVANT
    // sub-expression's own span (not just "whatever Eval'd last") through
    // the Frame that eventually panics: apply_binop's operand-type/
    // div-by-zero/etc panics (Frame::BinOpL/BinOpR carry both operands'
    // spans plus the whole BinOp's; apply_binop picks whichever operand is
    // actually the wrong type, e.g. `true + 1` blames `true` even though
    // `1` was Eval'd more recently) and "attempt to call a non-function
    // value" (Frame::AppFunc/AppArg carry the CALLEE's own span, so `f(1)`
    // with `f` not callable blames `f`, not the unrelated argument `1` --
    // the naive last-Eval fallback would wrongly blame the ARGUMENT, since
    // it's evaluated after `f` and right before the panic fires).
    // Everywhere else (builtin argument-type panics, match failure, ...)
    // still falls back to last-Eval, always in the neighborhood, never
    // wildly off, without threading a Span through every Frame variant and
    // Value accessor -- the same 80/20 trade this checker already makes
    // elsewhere (see typecheck::TypeError's doc comment, missing_case).
    static CURRENT_SPAN: Cell<Option<Span>> = const { Cell::new(None) };
}

pub fn current_span() -> Option<Span> {
    CURRENT_SPAN.with(|c| c.get())
}

// Trampolined CEK-style step loop -- no native recursion, so no stack
// overflow risk from deep programs or from resuming captured continuations.
// `arena` is only ever read here (all mutation happens during parsing and
// typecheck::elaborate); ExprRef fields are Copy, so threading node
// references through Control/Frame costs nothing beyond a plain integer
// copy -- no Rc bump, unlike when this held Rc<Expr>.
pub fn run(arena: &Arena, expr: ExprRef, env: Env, spans: &SpanMap) -> Value {
    debug_assert!(env.is_root(), "machine::run expects Env::prelude() (the root)");
    // One static pass, then the trampoline only ever indexes: see resolve.rs.
    let resolved = resolve(arena, expr);
    finished(run_loop::<false>(arena, Control::Eval(expr, env), Cont::nil(), spans, &resolved, 0))
}

// The ordinary (non-pure) loop never gets stuck.
fn finished(r: Result<Value, Stuck>) -> Value {
    match r {
        Ok(v) => v,
        Err(why) => unreachable!("only eval_pure gets stuck ({why:?})"),
    }
}

// Why a check-time evaluation (eval_pure) could not finish. Each one leaves a
// value only the run time can compute (spec section 2's "stuck" outcome).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stuck {
    // The step budget ran out.
    Fuel,
    // A `perform`: effects never run at check time.
    Effect,
    // A builtin outside the pure fragment (`print`, `map`, ...).
    Builtin,
    // A record field read: `.field` in a parsed tree (eval_pure), or the
    // get_field call elaboration lowers it to (a run-time annotation).
    Field,
}

// What eval_pure produced.
pub enum Pure {
    Done(Value),
    Stuck(Stuck),
    // A run-time error the evaluation hit (applying a non-function, `handle`,
    // a constructor given the wrong argument...): it would fail at run time too.
    Error(String),
}

// The step budget of one check-time evaluation (spec section 5). Measured
// 2026-10-10: every corpus and example annotation takes at most 9 steps; the
// largest in the lib tests takes 7628 (a union of 120 tuples). A million steps
// take 70-90 ms in a release build: the wait before a divergent type
// function's annotation is reported.
pub const ANNOTATION_FUEL: u64 = 1_000_000;

// The check-mode entry into the machine's step loop (spec section 1): evaluates
// `expr` in the prelude alone, no effects and at most `fuel` steps. The checker
// supplies an annotation's scope as the values of the definitions it names
// (eval_pure_in; typecheck's annotation_type). It never prints, and it leaves
// CURRENT_SPAN as it found it.
pub fn eval_pure(arena: &Arena, expr: ExprRef, spans: &SpanMap, fuel: u64) -> Pure {
    run_pure(arena, expr, Env::prelude(), spans, &resolve(arena, expr), fuel)
}

// `eval_pure` inside one enclosing frame: `names` are bound to `values`. A
// copy of `expr` is resolved into `resolved`, which the caller keeps so that a
// closure this evaluation returns (or stores in a value) can be called by a
// later one that shares the table. The copy is what keeps that table sound: a
// variable's slot is only valid for the layout its node was resolved under,
// and the same tree can be evaluated under several (a `let` nested in another
// definition's right-hand side is reached by both), so every evaluation gets
// one copy of its own, and every walk resolves only nodes its own copy made.
// A subtree `reuse` has a value for (a nested definition already evaluated) is
// not copied or run: a fresh variable stands for it, and the frame binds that
// variable to the value, after `names`.
//
// A debug build runs the same `expr` once more with nothing reused (own copy,
// own table, a clone of the arena, so nothing it makes lingers) and asserts
// that the two results print alike: reusing a value must equal running the
// subtree again. It compares at the root: a nested subtree may read binders
// between it and the root, so it cannot run alone. A run that does not finish
// (stuck, error, out of fuel) in the inline version is not compared; reuse may
// finish where it runs out of fuel, since the reused steps are not recounted.
#[allow(clippy::too_many_arguments)]
pub fn eval_pure_in(
    arena: &mut Arena,
    expr: ExprRef,
    names: &[String],
    values: Vec<Value>,
    reuse: impl Fn(ExprRef) -> Option<Value>,
    resolved: &mut Resolved,
    spans: &SpanMap,
    fuel: u64,
) -> Pure {
    debug_assert_eq!(names.len(), values.len(), "eval_pure_in: one value per name");
    let (mut bound, mut frame_values) = (names.to_vec(), values);
    let original = expr;
    let expr = crate::expr::copy_tree(arena, expr, |arena, n| {
        let v = reuse(n)?;
        bound.push(format!("{}#reused", bound.len()));
        frame_values.push(v);
        Some(arena.push(Expr::Var(bound.last().expect("just pushed").clone())))
    });
    crate::resolve::resolve_into(arena, expr, &bound, resolved);
    let mut frame = Bindings::default();
    #[cfg(debug_assertions)]
    let kept = (names.len() < bound.len()).then(|| frame_values[..names.len()].to_vec());
    frame_values.into_iter().for_each(|v| frame.push(v));
    let env = frame.extend(&Env::prelude());
    let out = run_pure(arena, expr, env, spans, resolved, fuel);
    #[cfg(debug_assertions)]
    if let Some(kept) = kept {
        assert_reuse_matches_inline(arena, original, names, kept, resolved, spans, fuel, &out);
    }
    #[cfg(not(debug_assertions))]
    let _ = original;
    out
}

// The debug cross-check of `eval_pure_in`: `expr` run with nothing reused.
#[cfg(debug_assertions)]
#[allow(clippy::too_many_arguments)]
fn assert_reuse_matches_inline(arena: &Arena, expr: ExprRef, names: &[String], values: Vec<Value>, resolved: &Resolved, spans: &SpanMap, fuel: u64, reused: &Pure) {
    let mut scratch = arena.clone();
    let copy = crate::expr::copy_tree(&mut scratch, expr, |_, _| None);
    // A clone: the closures in `values` run their bodies, which only the real
    // table has entries for, and the scratch nodes must not enter it.
    let mut table = resolved.clone();
    crate::resolve::resolve_into(&scratch, copy, names, &mut table);
    let mut frame = Bindings::default();
    values.into_iter().for_each(|v| frame.push(v));
    let env = frame.extend(&Env::prelude());
    if let Pure::Done(inline) = run_pure(&scratch, copy, env, spans, &table, fuel) {
        let Pure::Done(got) = reused else { panic!("reuse failed where running the subtrees again finished with {inline}") };
        assert_eq!(got.to_string(), inline.to_string(), "a reused value differs from running its subtree again");
    }
}

// Runs `expr` (resolved in `resolved`) in check mode; see eval_pure.
fn run_pure(arena: &Arena, expr: ExprRef, env: Env, spans: &SpanMap, resolved: &Resolved, fuel: u64) -> Pure {
    let saved_span = current_span();
    let run = std::panic::AssertUnwindSafe(|| run_loop::<true>(arena, Control::Eval(expr, env), Cont::nil(), spans, resolved, fuel));
    let result = std::panic::catch_unwind(run);
    CURRENT_SPAN.with(|c| c.set(saved_span));
    match result {
        Ok(Ok(v)) => Pure::Done(v),
        Ok(Err(why)) => Pure::Stuck(why),
        Err(payload) => match payload.downcast::<crate::util::RunError>() {
            Ok(e) => Pure::Error(e.0),
            Err(other) => std::panic::resume_unwind(other),
        },
    }
}

// An annotation's Type: eval_pure, which must finish with a type value.
pub fn eval_type(arena: &Arena, ann: ExprRef, spans: &SpanMap) -> Result<Type, String> {
    if let Expr::TypeLit(t) = &arena[ann] {
        return Ok((**t).clone());
    }
    pure_type(eval_pure(arena, ann, spans, ANNOTATION_FUEL)).map(|t| (*t).clone())
}

// An annotation evaluation's type, or the message for why it has none. Check
// time (eval_type, typecheck's annotation_type) and run time (an
// Expr::Annot) report the same words.
pub fn pure_type(outcome: Pure) -> Result<Rc<Type>, String> {
    match outcome {
        // A placeholder of a recursive type group that has not closed yet can never
        // become an annotation, whether bare or nested (see Iter::TypeGroup).
        Pure::Done(Value::Type(t)) => match first_free(&t) {
            Some(name) => Err(format!("recursive type {name} is used before it is built")),
            None => Ok(t),
        },
        Pure::Done(v) => Err(format!("annotation did not evaluate to a type, found {}", v.type_name())),
        Pure::Stuck(Stuck::Fuel) => Err(format!("annotation did not terminate within {ANNOTATION_FUEL} steps")),
        Pure::Stuck(Stuck::Effect) => Err("an annotation cannot perform an effect".to_string()),
        Pure::Stuck(Stuck::Builtin) => Err("an annotation can only call the type constructors".to_string()),
        Pure::Stuck(Stuck::Field) => Err("an annotation cannot read a record field".to_string()),
        Pure::Error(msg) => Err(msg),
    }
}

// A kept annotation's evaluation has finished (spec 11.3): its type, or the
// run-time error at the annotation. A Confirm site checks the type against
// the check-time one, then skips later evaluations.
fn annotation_value(spec: &AnnotSpec, outcome: Result<Value, Stuck>) -> Rc<Type> {
    let outcome = match outcome {
        Ok(v) => Pure::Done(v),
        Err(why) => Pure::Stuck(why),
    };
    let t = pure_type(outcome).unwrap_or_else(|msg| {
        set_current_span(Some(spec.span));
        crate::run_error!("{msg}")
    });
    if let AnnotKind::Confirm(expected) = &spec.kind {
        if *t != **expected {
            set_current_span(Some(spec.span));
            crate::run_error!("annotation evaluated to {} at run time but {} at check time", t.brief(), expected.brief())
        }
        spec.done.set(true);
    }
    t
}

// Feeds `value`, a callback's (or an element's) result, to the Iterate frame on top of `cont`.
// Its node is copied first (`Cont::unshared_top`) if anything else holds it: after a resume
// while the captured continuation is alive, the live top node IS the captured segment's.
fn iterate(cont: &mut Cont, value: Value, spans: &SpanMap) -> Control {
    let Frame::Iterate { state, pending } = &mut cont.unshared_top().frame else {
        unreachable!("internal: iterate is called on an Iterate frame")
    };
    if let Some(b) = pending.take() {
        // The first half of a two-argument call returned a function: apply it to `b`.
        *cont = Cont::cons(Frame::AppArg { func: value, callee_span: None }, cont.clone());
        return Control::Apply(b);
    }
    // Copy-on-write: copies the state only if a captured continuation still holds it.
    let next = step_shared(state, value, spans);
    follow(cont, next)
}

// Does what the Iterate frame on top of `cont` asked for.
fn follow(cont: &mut Cont, next: Next) -> Control {
    match next {
        Next::Call { f, a, b } => {
            if b.is_some() {
                let Frame::Iterate { pending, .. } = &mut cont.unshared_top().frame else {
                    unreachable!("internal: follow is called on an Iterate frame")
                };
                *pending = b;
            }
            // The same entry an ordinary call takes, so closures, continuations,
            // builtins and contracts all behave as they do anywhere else.
            *cont = Cont::cons(Frame::AppArg { func: f, callee_span: None }, cont.clone());
            Control::Apply(a)
        }
        Next::Eval(expr, env) => Control::Eval(expr, env),
        Next::Tail(expr, env) => {
            pop(cont);
            Control::Eval(expr, env)
        }
        Next::Done(v) => {
            pop(cont);
            Control::Apply(v)
        }
    }
}

// Drops the Iterate frame on top of `cont`.
fn pop(cont: &mut Cont) {
    let Some(Node { rest, .. }) = cont.node() else { unreachable!("internal: follow is called on an Iterate frame") };
    *cont = rest.clone();
}

// What a saturated builtin call does: yields a value, or starts a step-driven computation
// whose first request is `Next` (already `Done` for an empty list: no frame).
enum Dispatch {
    Value(Value),
    Iterate(Iter, Next),
}

// Continues `cont` with a builtin call's outcome, pushing the Iterate frame if one starts.
fn enter(dispatch: Dispatch, cont: &mut Cont) -> Control {
    match dispatch {
        Dispatch::Value(v) | Dispatch::Iterate(_, Next::Done(v)) => Control::Apply(v),
        Dispatch::Iterate(iter, next) => {
            *cont = Cont::cons(Frame::iterate(iter), cont.clone());
            follow(cont, next)
        }
    }
}

// PURE is eval_pure's check mode: a step budget (`fuel`), no effects, only the
// type-constructor builtins. Every PURE test is a constant, so the ordinary
// instance (`run_loop::<false>`) compiles without them: no cost on the hot path.
fn run_loop<const PURE: bool>(arena: &Arena, mut control: Control, mut cont: Cont, spans: &SpanMap, resolved: &Resolved, mut fuel: u64) -> Result<Value, Stuck> {
    // The handlers (and joins) below the segment `cont` is running; each run starts with none.
    // PURE never installs one.
    let mut meta = Meta::default();
    loop {
        // Native code between sub-evaluations (a higher-order builtin's callbacks, a
        // literal's elements, a `let rec` group's values) takes the result first. Its state is
        // changed in place when only this evaluation holds it, so building n elements is
        // O(n); one a captured continuation shares is copied first.
        if matches!(control, Control::Apply(_)) && matches!(cont.top(), Some(Frame::Iterate { .. })) {
            let Control::Apply(value) = control else { unreachable!() };
            control = iterate(&mut cont, value, spans);
            continue;
        }
        match control {
            Control::Eval(expr, env) => {
                if PURE {
                    if fuel == 0 {
                        return Err(Stuck::Fuel);
                    }
                    fuel -= 1;
                }
                // `spans` covers every node the PARSER produced, but the
                // tree actually running here is typecheck's ELABORATED
                // one -- full of re-emitted and brand-new (Check, contract
                // chains) nodes past the end of `spans`, which never
                // tracks those (see typecheck::TypeError's doc comment).
                // Skip the update rather than panic on those; CURRENT_SPAN
                // just keeps whatever the last KNOWN-location Eval set,
                // which is the same "last expression evaluated" fallback
                // already documented above.
                if let Some(span) = spans.get(expr) {
                    CURRENT_SPAN.with(|c| c.set(Some(*span)));
                }
                match &arena[expr] {
                Expr::Annot(ann, spec) => match &spec.kind {
                    AnnotKind::Confirm(t) if spec.done.get() => control = Control::Apply(Value::Type(t.clone())),
                    // The check-time evaluation ran out of fuel, and the run
                    // time's budget is the same (spec 12.1).
                    AnnotKind::Diverged if PURE => return Err(Stuck::Fuel),
                    AnnotKind::Diverged => control = Control::Apply(Value::Type(annotation_value(spec, Err(Stuck::Fuel)))),
                    _ => {
                        #[cfg(test)]
                        spec.evals.set(spec.evals.get() + 1);
                        if PURE {
                            // Already in check mode (a type function's body):
                            // the same loop and budget evaluate it.
                            cont = Cont::cons(Frame::Annot { spec: spec.clone() }, cont);
                            control = Control::Eval(*ann, env);
                        } else {
                            // One nested check-mode loop: no continuation
                            // spans it, since `perform` is stuck before one
                            // is captured.
                            let outcome = run_loop::<true>(arena, Control::Eval(*ann, env), Cont::nil(), spans, resolved, ANNOTATION_FUEL);
                            control = Control::Apply(Value::Type(annotation_value(spec, outcome)));
                        }
                    }
                },
                Expr::Conform(operand, spec) => {
                    // A static boundary's type was lowered at check time; an Unknown site's is the value of its
                    // annotation's binder. Borrowed, so a boundary that needs nothing more than its native test
                    // adds no reference-count traffic to the hot path.
                    let site;
                    let lazy: &Rc<Lazy> = match &spec.source {
                        TypeSource::Static(lazy) => lazy,
                        TypeSource::Site(ty, memo) => {
                            let Value::Type(t) = local(&env, resolved, *ty) else { panic!("internal: a conform node's type variable holds no type") };
                            site = conform_lazy(memo, t);
                            &site
                        }
                    };
                    if let Some(lowered) = lazy_lowered(lazy, spec) {
                        cont = match lowered.then {
                            Then::Done => {
                                let lens = done_lens(spec, lowered, &env, resolved);
                                Cont::cons(Frame::Check { spec: lowered.check.clone(), lens }, cont)
                            }
                            _ => {
                                // The slots read the env the operand is evaluated in.
                                let table = spec.lens.iter().map(|slot| eval_slot(slot, &env, resolved)).collect();
                                Cont::cons(Frame::Conform { lazy: lazy.clone(), ctx: Rc::new(SiteCtx { spec: spec.clone(), table, witness: None, path: None }) }, cont)
                            }
                        };
                    }
                    control = Control::Eval(*operand, env);
                }
                Expr::Int(n) => control = Control::Apply(Value::Int(*n)),
                Expr::Float(x) => control = Control::Apply(Value::Float(*x)),
                Expr::Bool(b) => control = Control::Apply(Value::Bool(*b)),
                Expr::Str(s) => control = Control::Apply(Value::Str(Text::from(s.as_str()))),
                Expr::Token(id) => control = Control::Apply(Value::Token(*id)),
                Expr::TypeLit(t) => control = Control::Apply(Value::Type(t.clone())),
                // Always 2+ items (see Expr::Tuple's own doc comment), so
                // unlike ListLit there's no empty-case to special-case --
                // same left-to-right Iter::Elems machinery either way,
                // building a plain Value::List.
                Expr::Tuple(items) => {
                    let (iter, first) = Iter::elems(None, items.clone(), env.clone());
                    cont = Cont::cons(Frame::iterate(iter), cont);
                    control = Control::Eval(first, env);
                }
                // The same Iter::Elems machinery, with the field names
                // (fixed for the whole record) zipped back in at the end
                // into a Value::Record instead of a bare Value::List (see
                // Value::Record's own doc comment for why records need a
                // name-keyed runtime shape). Always 1+ fields:
                // parser::parse_record_fields rejects `{}` (see its own
                // doc comment), the same "no
                // empty case to special-case" guarantee Tuple's own arm
                // above relies on.
                Expr::Record(fields) => {
                    let names: Rc<Vec<String>> = Rc::new(fields.iter().map(|(n, _)| n.clone()).collect());
                    let (iter, first) = Iter::elems(Some(names), fields.iter().map(|(_, v)| *v).collect(), env.clone());
                    cont = Cont::cons(Frame::iterate(iter), cont);
                    control = Control::Eval(first, env);
                }
                // Reachable only if this tree skipped typecheck::check --
                // elaborate_node's own FieldAccess arm rewrites every
                // Expr::FieldAccess into an ordinary get_field(...) call
                // during elaboration (see its own doc comment). Same
                // caveat as Expr::Record's own arm just above: a
                // properly typechecked tree never has one left by the
                // time it gets here, but machine::run itself has no such
                // guarantee for a tree that reached it directly (e.g.
                // lib.rs's run_untyped) -- `.field` access simply isn't
                // supported on that untyped path. eval_pure runs parsed
                // trees, so in check mode it is stuck, not a bug.
                Expr::FieldAccess(..) => {
                    if PURE {
                        return Err(Stuck::Field);
                    }
                    unreachable!(
                        "Expr::FieldAccess reached machine::run directly -- `.field` needs typecheck::check first, see its own doc comment"
                    )
                }
                Expr::ListLit(items) => {
                    if items.is_empty() {
                        control = Control::Apply(Value::List(List::empty()));
                    } else {
                        let (iter, first) = Iter::elems(None, items.clone(), env.clone());
                        cont = Cont::cons(Frame::iterate(iter), cont);
                        control = Control::Eval(first, env);
                    }
                }
                Expr::Var(name) => {
                    control = Control::Apply(match resolved.get(expr) {
                        Some(VarRef::Local { hops, slot }) => env.get(hops, slot),
                        Some(VarRef::Prelude(i)) => PRELUDE[i as usize].1.value(),
                        // Lazy, at evaluation time -- CURRENT_SPAN was set at
                        // the top of this Eval step, so the message keeps its
                        // source location exactly as before the resolver.
                        Some(VarRef::Unbound) => crate::run_error!("unbound variable: {name}"),
                        None => panic!("internal: variable `{name}` was never resolved"),
                    });
                }
                Expr::Lambda(_, _, body) => {
                    control = Control::Apply(Value::Closure(*body, env));
                }
                Expr::App(f, a) => {
                    let callee_span = spans.get(*f).copied();
                    cont = Cont::cons(Frame::AppFunc { arg: *a, env: env.clone(), callee_span }, cont);
                    control = Control::Eval(*f, env);
                }
                Expr::Let(_, _, val_expr, body) => {
                    cont = Cont::cons(Frame::LetBody { body: *body, env: env.clone() }, cont);
                    control = Control::Eval(*val_expr, env);
                }
                Expr::LetRec(bindings, body) => {
                    let kind = group_kind(arena, bindings);
                    if kind == GroupKind::Types {
                        // Rule P: every value is an unannotated call, so the group
                        // builds recursive types. The values run in a frame [names…]
                        // of placeholders (one fresh id per name); once all have
                        // run, close_group ties the knot (Iter::TypeGroup).
                        let ids: Rc<[u64]> = bindings.iter().map(|_| fresh_rec_id()).collect();
                        let mut frame = Bindings::default();
                        for ((name, _, _), id) in bindings.iter().zip(ids.iter()) {
                            frame.push(Value::Type(Rc::new(Type::RecVar(RecVar::Free { id: *id, name: Rc::from(name.as_str()) }))));
                        }
                        let vals_env = frame.extend(&env);
                        let first = bindings[0].2; // parser never emits an empty group
                        let iter = Iter::TypeGroup { bindings: bindings.clone(), body: *body, ids, done: Vec::new(), env, vals_env: vals_env.clone() };
                        cont = Cont::cons(Frame::iterate(iter), cont);
                        control = Control::Eval(first, vals_env);
                    } else if kind == GroupKind::Functions {
                        // Every value is a direct Lambda: build the group
                        // straight from the lambda bodies -- evaluating a
                        // Lambda is pure, so skipping it is unobservable --
                        // and bind all the RecClosures in ONE frame [names…],
                        // the layout resolve.rs assigns the `let rec` body.
                        let bodies: Rc<[ExprRef]> = bindings
                            .iter()
                            .map(|(_, _, v)| match &arena[*v] {
                                Expr::Lambda(_, _, b) => *b,
                                _ => unreachable!("group_kind checked"),
                            })
                            .collect();
                        let mut frame = Bindings::default();
                        for i in 0..bodies.len() {
                            frame.push(Value::RecClosure(bodies.clone(), i, env.clone()));
                        }
                        control = Control::Eval(*body, frame.extend(&env));
                    } else {
                        // Plain group (spec §2 fallback): values
                        // evaluate in the OUTER scope, left to right, and the
                        // names are bound plainly (non-recursively) for the
                        // body only (Iter::LetRec).
                        let (iter, first) = Iter::let_rec(bindings.iter().map(|(_, _, v)| *v).collect(), *body, env.clone());
                        cont = Cont::cons(Frame::iterate(iter), cont);
                        control = Control::Eval(first, env);
                    }
                }
                Expr::BinOp(op, l, r) => {
                    let (l_span, r_span) = (spans.get(*l).copied(), spans.get(*r).copied());
                    cont = Cont::cons(Frame::BinOpL { op: *op, rhs: *r, env: env.clone(), l_span, r_span }, cont);
                    control = Control::Eval(*l, env);
                }
                Expr::If(c, t, e) => {
                    cont = Cont::cons(Frame::If { then_: *t, else_: *e, env: env.clone() }, cont);
                    control = Control::Eval(*c, env);
                }
                Expr::Perform(effect, payload) => {
                    if PURE {
                        return Err(Stuck::Effect);
                    }
                    cont = Cont::cons(Frame::PerformPayload { effect: effect.clone() }, cont);
                    control = Control::Eval(*payload, env);
                }
                Expr::Handle { body, handler } => {
                    if PURE {
                        // It would capture a continuation at check time (spec section 2).
                        crate::run_error!("an annotation cannot contain `handle`")
                    }
                    cont = Cont::cons(Frame::InstallHandler { body: *body, env: env.clone() }, cont);
                    control = Control::Eval(*handler, env);
                }
                Expr::Match(scrutinee, arms) => {
                    cont = Cont::cons(Frame::MatchArms { arms: arms.clone(), env: env.clone() }, cont);
                    control = Control::Eval(*scrutinee, env);
                }
                Expr::MakeHandler { effect, body, .. } => {
                    control = Control::Apply(Value::Handler(Rc::new(HandlerData {
                        effect: effect.clone(),
                        body: *body,
                        env,
                        deep: false,
                    })));
                }
                }
            }

            // Matches on `frame` by reference: each arm clones only the
            // specific fields it actually moves elsewhere, instead of
            // eagerly cloning the whole frame (most fields on most arms
            // would otherwise be cloned and immediately discarded).
            Control::Apply(value) => match cont.node() {
                // The segment returned: continue after the nearest mark (a handler's identity
                // return clause, or a join), going round the loop again so the Iterate check
                // sees the new top.
                None => match meta.underflow() {
                    None => return Ok(value),
                    Some(rest) => {
                        cont = rest;
                        control = Control::Apply(value);
                    }
                },
                Some(Node { frame, rest }) => {
                    let rest = rest.clone();
                    // Each arm clones the specific fields it needs into
                    // owned locals FIRST (ending frame's borrow there),
                    // then reassigns `cont`/`control` -- reassigning `cont`
                    // while still reading through the old borrow doesn't
                    // typecheck, even though it would be sound.
                    match frame {
                        Frame::AppFunc { arg, env, callee_span } => {
                            let (arg, env, callee_span) = (*arg, env.clone(), *callee_span);
                            cont = Cont::cons(Frame::AppArg { func: value, callee_span }, rest);
                            control = Control::Eval(arg, env);
                        }
                        Frame::AppArg { func, callee_span } => {
                            let callee_span = *callee_span;
                            let mut func = func.clone();
                            let mut value = value;
                            cont = rest;
                            // A contract checks the argument, then the result
                            // once `f` returns, and applies `f`.
                            while let Value::Contract(c) = &func {
                                let c = c.clone();
                                let (arg, ret_ctx) = enter_contract(&c, value);
                                cont = Cont::cons(Frame::Conform { lazy: c.ret.clone(), ctx: ret_ctx }, cont);
                                value = arg;
                                func = c.f.clone();
                            }
                            match func {
                                Value::Closure(body, closure_env) => {
                                    control = Control::Eval(body, closure_env.push1(value));
                                }
                                Value::RecClosure(group, index, closure_env) => {
                                    // Rebuild the whole group in ONE frame,
                                    // argument last -- [names…, param], the
                                    // layout resolve::visit assigns a group
                                    // function body -- on EVERY call, so a
                                    // reference to any sibling (including
                                    // this one) resolves recursively with no
                                    // mutation and no Rc cycle. A length-1
                                    // group is two values: no Vec allocated.
                                    let mut frame = Bindings::default();
                                    for i in 0..group.len() {
                                        frame.push(Value::RecClosure(group.clone(), i, closure_env.clone()));
                                    }
                                    frame.push(value);
                                    control = Control::Eval(group[index], frame.extend(&closure_env));
                                }
                                Value::Continuation(k) => {
                                    // resume(value): push the captured
                                    // continuation back in front of whatever
                                    // comes after this call: its segment by
                                    // pointer, plus its marks. Nothing is
                                    // copied per frame -- multi-shot is
                                    // calling this arm more than once with
                                    // the same k.
                                    meta.resume(&k, &mut cont);
                                    control = Control::Apply(value);
                                }
                                Value::Builtin(b) => {
                                    if PURE && !matches!(b, Builtin::Ty(_)) {
                                        return Err(if matches!(b, Builtin::GetField) { Stuck::Field } else { Stuck::Builtin });
                                    }
                                    set_current_span(callee_span);
                                    control = enter(collect_builtin_arg(b, Vec::new(), value), &mut cont);
                                }
                                // Never Ty: every type constructor takes one argument.
                                Value::PartialBuiltin(b, prev_args) => {
                                    // Take the collected args when this call is their only
                                    // holder; clone (and count it) when the partial is shared.
                                    let args = Rc::try_unwrap(prev_args).unwrap_or_else(|shared| {
                                        #[cfg(test)]
                                        PARTIAL_SHARED.with(|c| c.set(c.get() + 1));
                                        (*shared).clone()
                                    });
                                    set_current_span(callee_span);
                                    control = enter(collect_builtin_arg(b, args, value), &mut cont);
                                }
                                _ => {
                                    set_current_span(callee_span);
                                    crate::run_error!("attempt to call a non-function value")
                                }
                            }
                        }
                        Frame::InstallHandler { body, env } => {
                            let (body, env) = (*body, env.clone());
                            cont = rest;
                            match value {
                                Value::Handler(data) => {
                                    meta.install(data, &mut cont);
                                    control = Control::Eval(body, env);
                                }
                                _ => crate::run_error!("handle: expected a handler value"),
                            }
                        }
                        Frame::LetBody { body, env } => {
                            let (body, env) = (*body, env.clone());
                            cont = rest;
                            control = Control::Eval(body, env.push1(value));
                        }
                        Frame::Iterate { .. } => unreachable!("internal: an Iterate frame is fed at the top of the loop"),
                        Frame::MatchArms { arms, env } => {
                            let (arms, env) = (arms.clone(), env.clone());
                            let outcome = first_match(&arms, 0, &value, &env);
                            let (new_cont, new_control) = dispatch_arm_outcome(outcome, arms, env, value, rest);
                            cont = new_cont;
                            control = new_control;
                        }
                        Frame::MatchGuard { arms, idx, outer_env, guard_env, value: scrutinee } => {
                            let (arms, idx, outer_env, guard_env, scrutinee) =
                                (arms.clone(), *idx, outer_env.clone(), guard_env.clone(), scrutinee.clone());
                            if value.as_bool() {
                                let (_, _, body) = &arms[idx];
                                cont = rest;
                                control = Control::Eval(*body, guard_env);
                            } else {
                                let outcome = first_match(&arms, idx + 1, &scrutinee, &outer_env);
                                let (new_cont, new_control) = dispatch_arm_outcome(outcome, arms, outer_env, scrutinee, rest);
                                cont = new_cont;
                                control = new_control;
                            }
                        }
                        Frame::BinOpL { op, rhs, env, l_span, r_span } => {
                            let (op, rhs, env, l_span, r_span) = (*op, *rhs, env.clone(), *l_span, *r_span);
                            cont = Cont::cons(Frame::BinOpR { op, lhs: value, l_span, r_span }, rest);
                            control = Control::Eval(rhs, env);
                        }
                        Frame::BinOpR { op, lhs, l_span, r_span } => {
                            let (op, lhs, l_span, r_span) = (*op, lhs.clone(), *l_span, *r_span);
                            cont = rest;
                            control = Control::Apply(apply_binop(op, lhs, value, l_span, r_span));
                        }
                        Frame::If { then_, else_, env } => {
                            let (then_, else_, env) = (*then_, *else_, env.clone());
                            cont = rest;
                            control = Control::Eval(if value.as_bool() { then_ } else { else_ }, env);
                        }
                        // Decided while `frame` is still borrowed (nothing
                        // is cloned); `cont` is reassigned after its last use.
                        Frame::Check { spec, lens } => {
                            if !test_holds(&spec.test, &spec.defs, lens, &value) {
                                no_value_error(&spec.test, spec, lens, &value);
                                check_failed(spec, lens, &value, &None)
                            }
                            cont = rest;
                            control = Control::Apply(value);
                        }
                        Frame::Annot { spec } => {
                            let t = annotation_value(spec, Ok(value));
                            cont = rest;
                            control = Control::Apply(Value::Type(t));
                        }
                        Frame::Conform { lazy, ctx } => {
                            let value = conform(lazy, ctx, value);
                            cont = rest;
                            control = Control::Apply(value);
                        }
                        Frame::PerformPayload { effect } => {
                            let effect = effect.clone();
                            cont = rest;
                            control = Control::Perform(effect, value);
                        }
                    }
                }
            },

            Control::Perform(effect, payload) => {
                control = perform(&mut cont, &mut meta, &effect, payload);
            }
        }
    }
}

// ---- The deep check: ONE iterative walker ----
//
// `test_holds` (every Dyn-boundary Expr::Conform, and the `is` builtin) and `explain` (the failure message's "at element 1")
// are two runs of `walk`, so there is a single definition of the test and the
// native stack use is O(1) in the value's nesting depth (a recursive check
// overflowed the worker stack at about 150,000 levels in debug builds). The
// walker keeps a work list of (test, value) steps and, for `Or`, a stack of
// choice points: a failing alternative rewinds the work list to the height
// the `Or` started at and tries the next one. `defs` is the owning
// CheckSpec's alias table (what Test::Ref indexes) and `lens` the values of
// its length slots (what Test::ListLenArg reads); `is` passes `&[]` for the
// slots (lower_test rejects a type that needs one). The value is finite and acyclic, and every step consumes
// structure, so the walk ends.

// A scalar test decided by the value's own tag; None for a compound test.
fn leaf(t: &Test, v: &Value) -> Option<bool> {
    Some(match t {
        Test::Any => true,
        Test::Never => false,
        Test::Int => matches!(v, Value::Int(_)),
        Test::Float => matches!(v, Value::Float(_)),
        Test::Bool => matches!(v, Value::Bool(_)),
        Test::Str => matches!(v, Value::Str(_)),
        Test::List => matches!(v, Value::List(_)),
        // A representation, not a signature: the higher-order contract
        // (Value::Contract) re-checks each call separately.
        Test::Fun => {
            matches!(v, Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..) | Value::Contract(_))
        }
        Test::Token(id) => matches!(v, Value::Token(t) if t == id),
        Test::Elems { .. } | Test::TupleOf(_) | Test::RecordOf(_) | Test::Ref(..) | Test::Or(_) => return None,
    })
}

// A Conform's length slot under `env`: its term, evaluated.
fn eval_slot(slot: &std::cell::RefCell<LenExpr>, env: &Env, resolved: &Resolved) -> Len {
    to_len(eval_len(&slot.borrow(), env, resolved))
}

// A length term's value as a slot holds it.
fn to_len(n: Result<i64, Len>) -> Len {
    match n {
        Ok(n) => usize::try_from(n).map_or(Len::Never, Len::Exact),
        Err(len) => len,
    }
}

// `op` on two length values: the first non-integer operand, left to right,
// wins, and an overflow is Never.
fn len_arith(a: Result<i64, Len>, b: impl FnOnce() -> Result<i64, Len>, op: fn(i64, i64) -> Option<i64>) -> Result<i64, Len> {
    op(a?, b()?).ok_or(Len::Never)
}

// A length term's integer value, or (Err) the slot value it reduces to
// instead. Total: a witness that is not a list is NoValue and an overflow is
// Never, so evaluating a slot never panics whatever an untyped caller passed;
// only the test decides.
fn eval_len(term: &LenExpr, env: &Env, resolved: &Resolved) -> Result<i64, Len> {
    let arith = |a: &LenExpr, b: &LenExpr, op| len_arith(eval_len(a, env, resolved), || eval_len(b, env, resolved), op);
    match term {
        LenExpr::Lit(n) => Ok(*n),
        LenExpr::Witness(var, name) => match resolved.get(*var) {
            Some(VarRef::Local { hops, slot }) => match env.get(hops, slot) {
                Value::List(items) => i64::try_from(items.len()).map_err(|_| Len::Never),
                _ => Err(Len::NoValue(name.clone())),
            },
            other => panic!("internal: witness of {name} resolved to {other:?}"),
        },
        LenExpr::Add(a, b) => arith(a, b, i64::checked_add),
        LenExpr::Sub(a, b) => arith(a, b, i64::checked_sub),
        LenExpr::Mul(a, b) => arith(a, b, i64::checked_mul),
        LenExpr::Any => Err(Len::Any),
        LenExpr::NoValue(name) => Err(Len::NoValue(name.clone())),
    }
}

// An index in a run-time type, valued by `value`: an Unknown site's table
// (each variable the annotation's text writes) and a contract's witness. Any
// other variable came with a type value built elsewhere and matches any
// length (spec 11.4).
fn index_len(index: &IndexExpr, value: &impl Fn(&str) -> Option<Len>) -> Result<i64, Len> {
    let arith = |a: &IndexExpr, b: &IndexExpr, op| len_arith(index_len(a, value), || index_len(b, value), op);
    match index {
        IndexExpr::Var(name) => match value(name) {
            Some(Len::Exact(k)) => i64::try_from(k).map_err(|_| Len::Never),
            Some(other) => Err(other),
            None => Err(Len::Any),
        },
        IndexExpr::Lit(n) => Ok(*n),
        IndexExpr::Add(a, b) => arith(a, b, i64::checked_add),
        IndexExpr::Sub(a, b) => arith(a, b, i64::checked_sub),
        IndexExpr::Mul(a, b) => arith(a, b, i64::checked_mul),
    }
}

// The value of a variable a synthesized node reads (a Conform's type, a
// contract's witness): always a local.
fn local(env: &Env, resolved: &Resolved, var: ExprRef) -> Value {
    match resolved.get(var) {
        Some(VarRef::Local { hops, slot }) => env.get(hops, slot),
        other => panic!("internal: a conform node's variable resolved to {other:?}"),
    }
}

// The length-slot values of a boundary that needs nothing beyond its native
// test, read on entry in `env`: each of the lowered type's slots over the
// values of the names it reads (the Conform's own slots, evaluated on demand,
// so a check with no length slots allocates nothing).
fn done_lens(spec: &ConformSpec, lowered: &Lowered, env: &Env, resolved: &Resolved) -> Vec<Len> {
    lowered
        .slots
        .iter()
        .map(|index| {
            to_len(index_len(index, &|name| {
                let i = spec.index_names.iter().position(|n| n == name)?;
                Some(eval_slot(&spec.lens[i], env, resolved))
            }))
        })
        .collect()
}

// A contract call's parameter type and argument: when the type is Vec(n),
// `n` and the argument's length (a non-list has no length; the argument's own
// check rejects it).
fn param_witness(param: &Type, arg: &Value) -> Option<(String, Len)> {
    let Type::Indexed(list, index) = param else { return None };
    let (Type::List(_), IndexExpr::Var(n)) = (&**list, &**index) else { return None };
    let len = match arg {
        Value::List(items) => Len::Exact(items.len()),
        _ => Len::NoValue(n.as_str().into()),
    };
    Some((n.clone(), len))
}

#[cfg(test)]
thread_local! {
    // How many times a PartialBuiltin call found its collected args shared (and cloned them).
    pub(crate) static PARTIAL_SHARED: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    // How many Lazy families conform_lazy has minted (memo misses) on this thread.
    pub(crate) static LAZY_MINTS: Cell<usize> = const { Cell::new(0) };
}

// An Unknown site's checks against type value `t`, lowered once while the same
// value keeps coming back (spec 11.5).
fn conform_lazy(memo: &std::cell::RefCell<Option<Rc<Lazy>>>, t: Rc<Type>) -> Rc<Lazy> {
    let mut memo = memo.borrow_mut();
    if let Some(seen) = memo.as_ref()
        && (Rc::ptr_eq(&seen.ty, &t) || *seen.ty == *t)
    {
        // `==`: a recursive type rebuilt per call is a new value but the same type;
        // without it every call would mint a fresh Lazy cycle.
        return seen.clone();
    }
    #[cfg(test)]
    LAZY_MINTS.with(|c| c.set(c.get() + 1));
    let lazy = Lazy::new(t);
    *memo = Some(lazy.clone());
    lazy
}

// The checks for a type (None: nothing to check), lowered the first time they
// are asked for (a static boundary's already is): the blame span is the site's.
// Borrowed from the Lazy.
fn lazy_lowered<'a>(lazy: &'a Lazy, spec: &ConformSpec) -> Option<&'a Lowered> {
    lazy.lowered.get_or_init(|| crate::typecheck::lower_conform(lazy, spec.span)).as_deref()
}

// `value` conformed to the type `lazy` holds, at the site `ctx` (spec 11.6): a
// failing check is the site's failure; a function becomes a contract, and a
// union with a function alternative picks the first alternative that holds.
fn conform(lazy: &Lazy, ctx: &Rc<SiteCtx>, value: Value) -> Value {
    let Some(lowered) = lazy_lowered(lazy, &ctx.spec) else { return value };
    let lens: Vec<Len> = lowered.slots.iter().map(|index| to_len(index_len(index, &|name| ctx.lookup(name)))).collect();
    let check = &*lowered.check;
    // A union's own test is the failure when no alternative holds.
    if !matches!(lowered.then, Then::Pick(_)) && !test_holds(&check.test, &check.defs, &lens, &value) {
        no_value_error(&check.test, check, &lens, &value);
        check_failed(check, &lens, &value, &ctx.path)
    }
    conform_then(&lowered.then, check, &lens, value, ctx)
}

// What `value`, which passed its test, still needs.
fn conform_then(then: &Then, check: &CheckSpec, lens: &[Len], value: Value, ctx: &Rc<SiteCtx>) -> Value {
    match then {
        Then::Done => value,
        Then::Wrap(param, ret) => Value::Contract(Rc::new(Contract { f: value, param: param.clone(), ret: ret.clone(), ctx: ctx.clone(), ret_ctx: ctx.clone() })),
        Then::Pick(alternatives) => {
            for (test, then) in alternatives {
                if test_holds(test, &check.defs, lens, &value) {
                    return conform_then(then, check, lens, value, ctx);
                }
                no_value_error(test, check, lens, &value);
            }
            check_failed(check, lens, &value, &ctx.path)
        }
    }
}

// A call of a contract: the argument (checked against the parameter type at
// an untrusted site) and the site the result is checked at. A Vec(n)
// parameter binds `n` to the argument's length for the result, unless the
// site already holds an exact length for `n` (the scope's, as build_cast does).
fn enter_contract(c: &Contract, arg: Value) -> (Value, Rc<SiteCtx>) {
    let ret_ctx = match param_witness(&c.param.ty, &arg) {
        Some((n, len)) if !matches!(c.ret_ctx.lookup(&n), Some(Len::Exact(_))) => Rc::new(c.ret_ctx.with_witness(n, len)),
        _ => c.ret_ctx.clone(),
    };
    let arg = if c.ctx.spec.trusted { arg } else { conform(&c.param, &c.ctx, arg) };
    (arg, ret_ctx)
}

// A failing test that reached a length nothing at run time holds, on a list:
// that is the failure.
fn no_value_error(test: &Test, spec: &CheckSpec, lens: &[Len], value: &Value) {
    if let Some(name) = reached_no_value(test, &spec.defs, lens, value) {
        set_current_span(spec.span);
        let name = crate::types::surface_name(name);
        crate::run_error!("type error: index variable {name} has no runtime value to check a length against")
    }
}

// A failed Assert check: the innermost offender and the way down to it.
fn check_failed(spec: &CheckSpec, lens: &[Len], value: &Value, fn_path: &Option<Rc<PathStep>>) -> ! {
    set_current_span(spec.span);
    let mut path = Vec::new();
    let found = explain(&spec.test, &spec.defs, lens, value, &mut path);
    let at = if path.is_empty() { String::new() } else { format!(" at {}", path_text(&path)) };
    let site = if fn_path.is_some() { format!(" (function at {})", PathStep::text(fn_path)) } else { String::new() };
    crate::run_error!("type error: expected {}, found {}{at}{site}", spec.to.brief(), found.type_name())
}

#[cfg(test)]
thread_local! {
    // Elements tested by the non-tracking walk: one per element of a scalar scan and one per
    // `Step::Test` popped. The complexity tests bound it instead of timing (read on the thread
    // that ran the program, the same pattern as LAZY_MINTS).
    pub(crate) static CHECK_STEPS: Cell<usize> = const { Cell::new(0) };
}

#[allow(unused_variables)]
fn count_check_steps(n: usize) {
    #[cfg(test)]
    CHECK_STEPS.with(|c| c.set(c.get() + n));
}

enum Step<'t, 'a> {
    // Check this value against this test; the u32 is its path node.
    Test(&'t Test, &'a Value, u32),
    // The current `Or` alternative passed: drop its choice point.
    Commit,
    // Fail at this value (a record missing a field, after the fields it does
    // have passed: a present failing field is reported before the record).
    FailAt(&'a Value, u32),
    // Everything above this entered (alias, value) has passed: remember it.
    // A list entered through a `Ref` with an id also gets its whole seal (read only at this
    // exact view).
    Mark(MemoKey, Option<(&'a List, SealId)>),
    // Every element of this list passed the every-element test: seal it (never in a tracked
    // walk). Popped only after the elements pushed above it all passed; an `Or` rewind drops it.
    Seal(&'a List, SealId),
}

// An (alias, value identity) pair of the walk's memo.
type MemoKey = (usize, (usize, u64));

// A compound value's identity (its shared allocation; a list's includes the view's
// interval, as two views of one block are different lists), for the walk's memo of
// decided (alias, value) pairs. Scalars are as cheap to re-test as to look up.
fn identity(v: &Value) -> Option<(usize, u64)> {
    match v {
        Value::List(items) => Some(items.identity()),
        Value::Record(fields) => Some((Rc::as_ptr(fields) as usize, 0)),
        _ => None,
    }
}

// An `Or` being tried: where to rewind the work list to, the alternatives
// left, and the failure that got furthest so far.
struct Choice<'t, 'a> {
    height: usize,
    alternatives: &'t [Test],
    next: usize,
    value: &'a Value,
    node: u32,
    best: Fail<'a>,
}

#[derive(Clone, Copy)]
struct Fail<'a> {
    node: u32,
    depth: u32,
    found: &'a Value,
}

// One step of the way down to a failure.
#[derive(Clone, Copy)]
enum Label<'t> {
    Element(usize),
    Field(&'t str),
}

// Path nodes (parent, depth, label), filled only by the tracked run; node 0
// is the root.
type Nodes<'t> = Vec<(u32, u32, Label<'t>)>;

fn child<'t, const TRACK: bool>(nodes: &mut Nodes<'t>, parent: u32, label: Label<'t>) -> u32 {
    if !TRACK {
        return 0;
    }
    let depth = nodes[parent as usize].1 + 1;
    nodes.push((parent, depth, label));
    (nodes.len() - 1) as u32
}

// Pushes one test per element, reversed so the stack pops them in element order (a list
// is walked forward, so the segment is reversed after the pushes).
fn push_elements<'t, 'a, const TRACK: bool>(
    work: &mut Vec<Step<'t, 'a>>,
    nodes: &mut Nodes<'t>,
    node: u32,
    tests: impl Iterator<Item = (&'t Test, &'a Value)>,
) {
    let start = work.len();
    for (i, (t, x)) in tests.enumerate() {
        let n = child::<TRACK>(nodes, node, Label::Element(i));
        work.push(Step::Test(t, x, n));
    }
    work[start..].reverse();
}

// None when `v` passes `test`, else the innermost failure (with TRACK, its
// path node and depth; an `Or` reports its alternative that got furthest,
// the first on ties). `no_value` gets the first NoValue slot reached on a list.
fn walk<'t, 'a, const TRACK: bool>(
    test: &'t Test,
    defs: &'t [Test],
    lens: &'t [Len],
    v: &'a Value,
    nodes: &mut Nodes<'t>,
    no_value: &mut Option<&'t str>,
) -> Option<Fail<'a>> {
    if TRACK {
        nodes.push((0, 0, Label::Element(0)));
    }
    let depth_of = |nodes: &Nodes<'t>, n: u32| if TRACK { nodes[n as usize].1 } else { 0 };
    let mut work: Vec<Step<'t, 'a>> = vec![Step::Test(test, v, 0)];
    let mut choices: Vec<Choice<'t, 'a>> = Vec::new();
    // Whether an entered (alias, value) passed (true) or failed (false). Both
    // are pure functions of the pair, so an `Or` whose alternatives share a
    // compound prefix (`{k: [T], v: Int} | {k: [T], v: Str}`) re-walks it once,
    // not once per alternative (doubling per level). A skipped failure reports
    // at the alias position itself: the same furthest-failure ranking as the
    // first walk, which already holds the deeper report.
    let mut memo: HashMap<MemoKey, bool> = HashMap::new();
    'main: loop {
        let mut fail: Fail<'a> = 'step: {
            let step = work.pop()?;
            let (t, v, node) = match step {
                Step::Commit => {
                    choices.pop();
                    continue 'main;
                }
                Step::Mark(key, whole) => {
                    memo.insert(key, true);
                    if let Some((items, id)) = whole {
                        items.seal_whole(id);
                    }
                    continue 'main;
                }
                Step::Seal(items, id) => {
                    items.seal_every(id);
                    continue 'main;
                }
                Step::FailAt(v, node) => break 'step Fail { node, depth: depth_of(nodes, node), found: v },
                Step::Test(t, v, node) => {
                    if !TRACK {
                        count_check_steps(1);
                    }
                    (t, v, node)
                }
            };
            let here = Fail { node, depth: depth_of(nodes, node), found: v };
            if let Some(ok) = leaf(t, v) {
                if ok {
                    continue 'main;
                }
                break 'step here;
            }
            match t {
                Test::Ref(i, seal) => {
                    let whole = match (seal.0.filter(|_| !TRACK), v) {
                        (Some(id), Value::List(items)) => Some((items, id)),
                        _ => None,
                    };
                    if whole.is_some_and(|(items, id)| items.has_whole_seal(id)) {
                        continue 'main;
                    }
                    if let Some(p) = identity(v) {
                        match memo.get(&(*i, p)) {
                            Some(true) => continue 'main,
                            Some(false) => break 'step here,
                            None => work.push(Step::Mark((*i, p), whole)),
                        }
                    }
                    work.push(Step::Test(&defs[*i], v, node));
                }
                Test::Elems { elem: element, len, seal } => {
                    let Value::List(items) = v else { break 'step here };
                    match len {
                        Length::Any => {}
                        Length::Exact(len) => {
                            if items.len() != *len {
                                break 'step here;
                            }
                        }
                        Length::Slot(slot) => match &lens[*slot] {
                            Len::Exact(len) if items.len() == *len => {}
                            Len::Any => {}
                            Len::NoValue(name) => {
                                no_value.get_or_insert(&**name);
                                break 'step here;
                            }
                            Len::Exact(_) | Len::Never => break 'step here,
                        },
                    }
                    // A tracked walk (the failure explanation) ignores seals.
                    if let Some(id) = seal.0.filter(|_| !TRACK) {
                        if items.is_sealed_every(id) {
                            continue 'main;
                        }
                        if leaf(element, v).is_some() {
                            let mut tested = 0;
                            let bad = items.unsealed(id).any(|x| {
                                tested += 1;
                                leaf(element, x) == Some(false)
                            });
                            count_check_steps(tested);
                            if bad {
                                break 'step here;
                            }
                            items.seal_every(id);
                        } else {
                            work.push(Step::Seal(items, id));
                            push_elements::<TRACK>(&mut work, nodes, node, items.unsealed(id).map(|x| (&**element, x)));
                        }
                        continue 'main;
                    }
                    if leaf(element, v).is_some() {
                        // A scalar element test (`leaf` answers Some for those
                        // whatever the value): scan inline, nothing pushed.
                        let scanned = items.iter().enumerate().find(|(_, x)| leaf(element, x) == Some(false));
                        if !TRACK {
                            count_check_steps(scanned.map_or(items.len(), |(i, _)| i + 1));
                        }
                        match scanned {
                            None => continue 'main,
                            Some((i, found)) => {
                                let n = child::<TRACK>(nodes, node, Label::Element(i));
                                break 'step Fail { node: n, depth: depth_of(nodes, n), found };
                            }
                        }
                    }
                    push_elements::<TRACK>(&mut work, nodes, node, items.iter().map(|x| (&**element, x)));
                }
                Test::TupleOf(elements) => {
                    let Value::List(items) = v else { break 'step here };
                    if items.len() != elements.len() {
                        break 'step here;
                    }
                    push_elements::<TRACK>(&mut work, nodes, node, elements.iter().zip(items.iter()));
                }
                Test::RecordOf(required) => {
                    let Value::Record(fields) = v else { break 'step here };
                    let present: Vec<_> = required.iter().filter_map(|(name, t)| Some((name, t, find_field(fields, name)?))).collect();
                    if present.len() < required.len() {
                        work.push(Step::FailAt(v, node));
                    }
                    for (name, t, x) in present.into_iter().rev() {
                        let n = child::<TRACK>(nodes, node, Label::Field(name));
                        work.push(Step::Test(t, x, n));
                    }
                }
                Test::Or(alternatives) => {
                    let Some(first) = alternatives.first() else { break 'step here };
                    choices.push(Choice { height: work.len(), alternatives, next: 1, value: v, node, best: here });
                    work.push(Step::Commit);
                    work.push(Step::Test(first, v, node));
                }
                _ => unreachable!("scalar tests are decided by `leaf`"),
            }
            continue 'main;
        };
        // A failure: rewind to the nearest `Or` with an alternative left; an
        // exhausted one fails with its furthest alternative.
        loop {
            let Some(choice) = choices.last_mut() else { return Some(fail) };
            if fail.depth > choice.best.depth {
                choice.best = fail;
            }
            // The aliases entered since this `Or` started contain the failure.
            for step in work.drain(choice.height..) {
                if let Step::Mark(key, _) = step {
                    memo.insert(key, false);
                }
            }
            if choice.next < choice.alternatives.len() {
                let alternative = &choice.alternatives[choice.next];
                choice.next += 1;
                work.push(Step::Commit);
                work.push(Step::Test(alternative, choice.value, choice.node));
                break;
            }
            fail = choices.pop().expect("a choice point was just inspected").best;
        }
    }
}

fn test_holds(test: &Test, defs: &[Test], lens: &[Len], v: &Value) -> bool {
    match leaf(test, v) {
        Some(ok) => ok,
        // `Int | Float`, the check on every Dyn arithmetic operand: decided by
        // the value's tag, with no work list or choice point.
        None => match test {
            Test::Or(alternatives) if alternatives.iter().all(|a| leaf(a, v).is_some()) => alternatives.iter().any(|a| leaf(a, v) == Some(true)),
            // The walk's O(1) sealed query, asked before it allocates a work list (the length
            // is still checked first, as the walk does).
            Test::Elems { len, seal: Seal(Some(id), _), .. }
                if matches!(v, Value::List(items) if items.is_sealed_every(*id) && match len {
                    Length::Any => true,
                    Length::Exact(n) => items.len() == *n,
                    Length::Slot(_) => false,
                }) =>
            {
                true
            }
            _ => walk::<false>(test, defs, lens, v, &mut Vec::new(), &mut None).is_none(),
        },
    }
}

// The index variable a failing check could not compare a length against: the
// first NoValue slot the walk reached on a list (an `Or` alternative that
// reached one and failed counts, so a union reports it only when every
// alternative fails). None for an ordinary mismatch. Cold path: re-walks only
// when a slot is NoValue.
fn reached_no_value<'t>(test: &'t Test, defs: &'t [Test], lens: &'t [Len], v: &Value) -> Option<&'t str> {
    if !lens.iter().any(|len| matches!(len, Len::NoValue(_))) {
        return None;
    }
    let mut no_value = None;
    walk::<false>(test, defs, lens, v, &mut Vec::new(), &mut no_value);
    no_value
}

// The innermost value that makes `test` fail on `v`, with the way down to it
// pushed onto `path` ("element 1", "field b"). Cold path of a failing Assert
// check only: the tracked re-run of `walk`, linear in the value. A union
// cannot say which alternative was meant: it descends into the one that got
// furthest. `v` itself when nothing fails.
fn explain<'a>(test: &Test, defs: &[Test], lens: &[Len], v: &'a Value, path: &mut Vec<String>) -> &'a Value {
    let mut nodes = Vec::new();
    let Some(fail) = walk::<true>(test, defs, lens, v, &mut nodes, &mut None) else { return v };
    let mut labels = Vec::new();
    let mut n = fail.node;
    while n != 0 {
        let (parent, _, label) = nodes[n as usize];
        labels.push(match label {
            Label::Element(i) => format!("element {i}"),
            Label::Field(name) => format!("field {name}"),
        });
        n = parent;
    }
    path.extend(labels.into_iter().rev());
    fail.found
}

// The path as printed: a failure at the bottom of a million-deep value would
// otherwise print megabytes, so the middle of a long one is elided.
fn path_text(path: &[String]) -> String {
    const KEEP: usize = 10;
    if path.len() <= 2 * KEEP {
        return path.join(", ");
    }
    format!("{}, ... {} more ..., {}", path[..KEEP].join(", "), path.len() - 2 * KEEP, path[path.len() - KEEP..].join(", "))
}

pub(crate) fn set_current_span(span: Option<Span>) {
    if let Some(s) = span {
        CURRENT_SPAN.with(|c| c.set(Some(s)));
    }
}

// Widest span covering both -- used only where a panic can't be pinned on
// either operand alone (Concat with BOTH sides wrong-typed). Not the
// original BinOp node's own span: that node no longer exists in `spans`
// once typecheck::elaborate has run (it always rebuilds a BinOp node via
// arena.push, even when neither operand needed a Check -- see elaborate's
// own doc comment on spans only covering parser-produced nodes), so this
// reconstructs the same "whole expression" span from its two operands'
// (still-original, still-tracked) spans instead.
fn combine_spans(a: Option<Span>, b: Option<Span>) -> Option<Span> {
    match (a, b) {
        (Some(a), Some(b)) => Some(Span { start: a.start.min(b.start), end: a.end.max(b.end) }),
        (Some(s), None) | (None, Some(s)) => Some(s),
        (None, None) => None,
    }
}

// Int or Float, for Add/Sub/Mul/Div/Mod/Lt's own runtime promotion --
// typecheck::coerce_numeric only ever lets an Int/Float/checked-Dyn
// operand reach here (see its own doc comment), so the two variants here
// are the only shapes apply_binop's arithmetic arms ever actually see;
// anything else is a genuinely Dyn-sourced value whose runtime shape
// disagreed with what the boundary check demanded, and as_num_at's own
// fallback arm panics for it.
//
// as_num_at only touches CURRENT_SPAN on the FAILING path (never on a
// successful match) -- setting it unconditionally would clobber whatever
// the last real Eval left behind with this operand's own span even when
// this operand is perfectly fine, which is wrong the moment `span` itself
// is None (an elaborated, Check-wrapped Dyn operand has no span of its
// own in `spans` -- see combine_spans's doc comment -- so the CORRECT
// blame for a later failure is exactly "whatever Eval ran last", which
// this must not overwrite on the way there). So e.g. `true + 1` blames
// `true` specifically, not `1` (the last thing Eval'd before this call,
// and not at fault here) -- and `10 / y` with `y` a Dyn-sourced 0 blames
// `y` via that same untouched last-Eval fallback, not `10`.
enum Num {
    Int(i64),
    Float(f64),
}

impl Num {
    fn as_f64(&self) -> f64 {
        match self {
            Num::Int(n) => *n as f64,
            Num::Float(x) => *x,
        }
    }
}

fn as_num_at(v: &Value, span: Option<Span>) -> Num {
    match v {
        Value::Int(n) => Num::Int(*n),
        Value::Float(x) => Num::Float(*x),
        _ => {
            set_current_span(span);
            crate::run_error!("type error: expected Int | Float, found {}", v.type_name())
        }
    }
}

// Add/Sub/Mul/Div/Mod/Lt are Int-or-Float (see typecheck::coerce_numeric)
// -- both Int stays Int, exactly the same as before Float existed
// (including Div/Mod's truncating semantics and their own division/
// modulo-by-zero panics); either operand Float promotes the other to f64
// and produces a Float result, with zero-divisor still an explicit panic
// rather than IEEE754's own inf/NaN, so renno's "0 divisor is always a
// hard error" story stays uniform across both numeric types instead of
// quietly diverging for Float. A wrong operand type panics via
// as_num_at(), which blames whichever operand's own span was passed in
// (left evaluated, thus blamed, before right, matching apply_binop's
// call sites below). Div/Mod-by-zero pass `r_span` -- it's always the
// right operand that's zero, never a type question about either side --
// but like any operand span, that's only the RIGHT OPERAND'S OWN span
// when it's a simple, still-tracked node (a var or literal); a nested
// sub-expression divisor (`10 / (2 - 2)`) gets its own fresh, untracked
// BinOp node from elaborate, so `r_span` is None there too and this falls
// back to whatever Eval ran last inside it -- still somewhere inside the
// divisor, just not the divisor's own span as a whole. Eq is structural: it compares whatever tags
// the two values actually carry (typecheck.rs only requires the two
// operand types to be consistent with each other, not both Int), so it
// dispatches on Value directly instead of projecting through as_num_at(),
// and never panics. Concat (++) and Cons (::) typecheck.rs statically
// rejects when it can tell; a runtime panic here only fires for a
// Dyn-sourced operand of the wrong tag, in which case the fallback arm
// below picks the specific operand's span that's actually responsible
// (both, via combine_spans, only for `++` when NEITHER side was
// statically known -- see typecheck.rs's own doc comment on that gap, and
// on Cons's `t` being the only operand `::` can ever blame).
fn apply_binop(op: BinOp, lhs: Value, rhs: Value, l_span: Option<Span>, r_span: Option<Span>) -> Value {
    match op {
        BinOp::Add => match (as_num_at(&lhs, l_span), as_num_at(&rhs, r_span)) {
            (Num::Int(a), Num::Int(b)) => Value::Int(a + b),
            (a, b) => Value::Float(a.as_f64() + b.as_f64()),
        },
        BinOp::Sub => match (as_num_at(&lhs, l_span), as_num_at(&rhs, r_span)) {
            (Num::Int(a), Num::Int(b)) => Value::Int(a - b),
            (a, b) => Value::Float(a.as_f64() - b.as_f64()),
        },
        BinOp::Mul => match (as_num_at(&lhs, l_span), as_num_at(&rhs, r_span)) {
            (Num::Int(a), Num::Int(b)) => Value::Int(a * b),
            (a, b) => Value::Float(a.as_f64() * b.as_f64()),
        },
        BinOp::Div => match (as_num_at(&lhs, l_span), as_num_at(&rhs, r_span)) {
            (Num::Int(a), Num::Int(b)) => {
                if b == 0 {
                    set_current_span(r_span);
                    crate::run_error!("division by zero");
                }
                Value::Int(a / b)
            }
            (a, b) => {
                let b = b.as_f64();
                if b == 0.0 {
                    set_current_span(r_span);
                    crate::run_error!("division by zero");
                }
                Value::Float(a.as_f64() / b)
            }
        },
        BinOp::Mod => match (as_num_at(&lhs, l_span), as_num_at(&rhs, r_span)) {
            (Num::Int(a), Num::Int(b)) => {
                if b == 0 {
                    set_current_span(r_span);
                    crate::run_error!("modulo by zero");
                }
                Value::Int(a % b)
            }
            (a, b) => {
                let b = b.as_f64();
                if b == 0.0 {
                    set_current_span(r_span);
                    crate::run_error!("modulo by zero");
                }
                Value::Float(a.as_f64() % b)
            }
        },
        BinOp::Lt => match (as_num_at(&lhs, l_span), as_num_at(&rhs, r_span)) {
            (Num::Int(a), Num::Int(b)) => Value::Bool(a < b),
            (a, b) => Value::Bool(a.as_f64() < b.as_f64()),
        },
        BinOp::Eq => Value::Bool(value_eq(&lhs, &rhs)),
        BinOp::Concat => match (&lhs, &rhs) {
            (Value::Str(a), Value::Str(b)) => Value::Str(a.concat(b)),
            (Value::List(a), Value::List(b)) => Value::List(a.append(b)),
            _ => {
                // Whichever operand isn't Str/List is the actual culprit;
                // if both are (e.g. an Int and a Bool), neither operand
                // alone explains it, so blame the whole `l ++ r` instead.
                let l_ok = matches!(lhs, Value::Str(_) | Value::List(_));
                let r_ok = matches!(rhs, Value::Str(_) | Value::List(_));
                set_current_span(match (l_ok, r_ok) {
                    (false, true) => l_span,
                    (true, false) => r_span,
                    _ => combine_spans(l_span, r_span),
                });
                crate::run_error!("++ expects two strings or two lists")
            }
        },
        BinOp::Cons => match &rhs {
            Value::List(items) => Value::List(items.cons(lhs)),
            _ => {
                set_current_span(r_span);
                crate::run_error!(":: expects a list on the right")
            }
        },
    }
}

// Appends one argument to a builtin's collected-so-far list, dispatching
// the real operation once `b`'s declared arity is reached, otherwise
// handing back a PartialBuiltin waiting for the rest.
fn collect_builtin_arg(b: Builtin, mut args: Vec<Value>, arg: Value) -> Dispatch {
    args.push(arg);
    if args.len() != b.arity() {
        return Dispatch::Value(Value::PartialBuiltin(b, Rc::new(args)));
    }
    match b {
        Builtin::Map | Builtin::Filter | Builtin::Fold | Builtin::Sort => start_iteration(b, args),
        _ => Dispatch::Value(dispatch_builtin(b, args)),
    }
}

// The higher-order builtins: argument shapes are checked here, before any callback runs
// (the same messages and callee span as ever); the callbacks run as a Frame::Iterate.
fn start_iteration(b: Builtin, mut args: Vec<Value>) -> Dispatch {
    let (iter, next) = match b {
        Builtin::Map => {
            let (list, f) = (args.pop(), args.pop());
            match (f, list) {
                (Some(f), Some(Value::List(items))) => Iter::map(f, items),
                _ => crate::run_error!("map expects a function and a list"),
            }
        }
        Builtin::Fold => {
            let (list, init, f) = (args.pop(), args.pop(), args.pop());
            match (f, init, list) {
                (Some(f), Some(init), Some(Value::List(items))) => Iter::fold(f, init, items),
                _ => crate::run_error!("fold expects a function, an initial value, and a list"),
            }
        }
        Builtin::Filter => {
            let (list, pred) = (args.pop(), args.pop());
            match (pred, list) {
                (Some(pred), Some(Value::List(items))) => Iter::filter(pred, items),
                _ => crate::run_error!("filter expects a function and a list"),
            }
        }
        Builtin::Sort => {
            let (list, cmp) = (args.pop(), args.pop());
            match (cmp, list) {
                (Some(cmp), Some(Value::List(items))) => Iter::sort(cmp, items.to_vec()),
                _ => crate::run_error!("sort expects a comparator and a list"),
            }
        }
        _ => unreachable!("internal: not a higher-order builtin"),
    };
    Dispatch::Iterate(iter, next)
}

fn dispatch_builtin(b: Builtin, mut args: Vec<Value>) -> Value {
    match b {
        // deep/shallow: clone the handler data, flip the `deep` bit, hand
        // back a new handler value. No AST-level flag.
        Builtin::Deep | Builtin::Shallow => match args.pop() {
            Some(Value::Handler(data)) => {
                Value::Handler(Rc::new(HandlerData { deep: b == Builtin::Deep, ..(*data).clone() }))
            }
            _ => crate::run_error!("deep/shallow expect a handler value"),
        },
        Builtin::Len => match args.pop() {
            Some(Value::Str(s)) => Value::Int(s.chars().count() as i64),
            Some(Value::List(items)) => Value::Int(items.len() as i64),
            _ => crate::run_error!("len expects a string or list"),
        },
        Builtin::Fail => match args.pop() {
            Some(Value::Str(s)) => crate::run_error!("{s}"),
            _ => crate::run_error!("fail expects a string message"),
        },
        Builtin::Get => {
            let (i, list) = (args.pop(), args.pop());
            match (list, i) {
                (Some(Value::List(items)), Some(Value::Int(idx))) => {
                    match usize::try_from(idx).ok().and_then(|idx| items.get(idx)) {
                        Some(v) => v.clone(),
                        None => crate::run_error!("get: index out of bounds"),
                    }
                }
                _ => crate::run_error!("get expects a list and an int"),
            }
        }
        // (type, value) -- args.pop() order matches Get's own convention.
        Builtin::Is => match (args.pop(), args.pop()) {
            (Some(v), Some(Value::Type(t))) => match crate::typecheck::lower_test(&t) {
                Ok((test, defs)) => Value::Bool(test_holds(&test, &defs, &[], &v)),
                Err(msg) => crate::run_error!("{msg}"),
            },
            _ => crate::run_error!("is expects a type and a value"),
        },
        // (record, name) -- args.pop() order matches Get's own (list, i)
        // convention: last-pushed arg (name) pops first.
        // Panics on a missing field the same way Get panics on an
        // out-of-range index -- renno has no Option/Result to return
        // instead. The common case (target statically known to have this
        // field) never reaches here with a missing name at all --
        // typecheck::elaborate_node's own Expr::FieldAccess arm already
        // rejected that statically; this only fires for a genuinely
        // Dyn-sourced value that turns out to be missing what was asked
        // for, or isn't a Record at all.
        Builtin::GetField => {
            let (name, record) = (args.pop(), args.pop());
            match (record, name) {
                (Some(Value::Record(fields)), Some(Value::Str(name))) => match find_field(&fields, &name) {
                    Some(v) => v.clone(),
                    None => crate::run_error!("no field named `{name}`"),
                },
                _ => crate::run_error!("get_field expects a record and a string"),
            }
        }
        // Echoes its argument back unchanged after printing it -- see
        // Builtin::Print's own doc comment.
        Builtin::Print => match args.pop() {
            Some(v) => {
                println!("{v}");
                v
            }
            None => crate::run_error!("print expects one argument"),
        },
        Builtin::ToStr => match args.pop() {
            Some(v) => Value::Str(Text::from(v.to_string())),
            None => crate::run_error!("to_str expects one argument"),
        },
        // Higher-order: collect_builtin_arg sends these to start_iteration.
        Builtin::Map | Builtin::Fold | Builtin::Filter | Builtin::Sort => unreachable!("internal: a higher-order builtin is started, not dispatched"),
        Builtin::Reverse => match args.pop() {
            Some(Value::List(items)) => Value::List(List::from_bottom_up(items.iter().cloned())),
            _ => crate::run_error!("reverse expects a list"),
        },
        Builtin::Range => {
            let (end, start) = (args.pop(), args.pop());
            match (start, end) {
                (Some(Value::Int(start)), Some(Value::Int(end))) => Value::List(List::from_bottom_up((start..end).rev().map(Value::Int))),
                _ => crate::run_error!("range expects two ints"),
            }
        }
        Builtin::Join => {
            let (sep, parts) = (args.pop(), args.pop());
            match (parts, sep) {
                (Some(Value::List(items)), Some(Value::Str(sep))) => {
                    let strs: Vec<&str> = items
                        .iter()
                        .map(|v| match v {
                            Value::Str(s) => &**s,
                            _ => crate::run_error!("join expects a list of strings"),
                        })
                        .collect();
                    Value::Str(Text::from(strs.join(&sep)))
                }
                _ => crate::run_error!("join expects a list and a string"),
            }
        }
        // The constants (Int, ...) never get here: Builtin::value hands them
        // out as Value::Type already.
        Builtin::Ty(ctor) => match args.pop() {
            Some(arg) => Value::Type(Rc::new(construct_type(ctor, arg))),
            None => crate::run_error!("a type constructor expects one argument"),
        },
    }
}

// What a type constructor builds from its argument (spec section 3's call forms).
fn construct_type(ctor: TypeCtor, arg: Value) -> Type {
    match (ctor, arg) {
        (TypeCtor::List, Value::Type(t)) => Type::List(Hc::new((*t).clone())),
        (TypeCtor::Tuple, Value::List(items)) if !items.is_empty() => Type::Tuple(Hc::new(types_of(&items, "Tuple"))),
        (TypeCtor::Union, Value::List(items)) if !items.is_empty() => {
            let alts = types_of(&items, "Union");
            if alts.iter().any(|t| open_record_tail_under(t, true)) {
                crate::run_error!("a union cannot contain an open record type (`{{x: T, ..r}}`)");
            }
            Type::Union(Hc::new(alts))
        }
        (TypeCtor::Record, Value::Record(fields)) => {
            let mut fields: Vec<(String, Type)> = fields
                .iter()
                .map(|(n, v)| match v {
                    Value::Type(t) => (n.clone(), (**t).clone()),
                    other => crate::run_error!("Record expects a record of types, field `{n}` is a {}", other.type_name()),
                })
                .collect();
            fields.sort_by(|a, b| a.0.cmp(&b.0));
            Type::record(fields)
        }
        // ({fields}, "r") is the open record `{fields, ..r}`.
        (TypeCtor::Record, Value::List(items)) => match &items.to_vec()[..] {
            [Value::Record(fields), Value::Str(row)] => {
                let entries = fields
                    .iter()
                    .map(|(n, v)| match v {
                        Value::Type(t) => (n.clone(), (**t).clone()),
                        other => crate::run_error!("Record expects a record of types, field `{n}` is a {}", other.type_name()),
                    })
                    .collect();
                Type::Record(Row::from_entries(entries, Tail::Var(row.to_string())))
            }
            _ => crate::run_error!("Record expects a record of types, or (record of types, \"row\")"),
        },
        // (A, B) is `A -> B`; (A, "e", B) is `A ->{e} B`.
        (TypeCtor::Fun, Value::List(items)) => match &items.to_vec()[..] {
            [Value::Type(a), Value::Type(b)] => Type::Fun(Hc::new((**a).clone()), Row::dyn_(), Hc::new((**b).clone())),
            [Value::Type(a), Value::Str(row), Value::Type(b)] => {
                Type::Fun(Hc::new((**a).clone()), Row::var(row.to_string()), Hc::new((**b).clone()))
            }
            _ => crate::run_error!("Fun expects (param, result) or (param, \"row\", result) types"),
        },
        (TypeCtor::Vec, Value::Int(n)) if n >= 0 => {
            Type::Indexed(Hc::new(Type::List(Hc::new(Type::Dyn))), Rc::new(IndexExpr::Lit(n)))
        }
        // An indexed type gives its index: an annotation passes a symbolic
        // `Vec(n)` as the type Vec(n), since an index variable is not a value
        // (spec 11.8, 12.4). Any other type gets the generic message.
        (TypeCtor::Vec, Value::Type(t)) => match &*t {
            Type::Indexed(_, index) => Type::Indexed(Hc::new(Type::List(Hc::new(Type::Dyn))), index.clone()),
            _ => crate::run_error!("Vec cannot take a Type"),
        },
        // (T, k) is T at index k; (T, Vec(n)) takes the index n, which is
        // how the parser passes a symbolic `Indexed(T, n)`.
        (TypeCtor::Indexed, Value::List(items)) => match &items.to_vec()[..] {
            [Value::Type(t), Value::Int(k)] if *k >= 0 => Type::Indexed(Hc::new((**t).clone()), Rc::new(IndexExpr::Lit(*k))),
            [Value::Type(t), Value::Type(i)] => match &**i {
                Type::Indexed(_, index) => Type::Indexed(Hc::new((**t).clone()), index.clone()),
                _ => crate::run_error!("Indexed expects (type, index)"),
            },
            _ => crate::run_error!("Indexed expects (type, index)"),
        },
        (TypeCtor::TypeOf, Value::Token(id)) => Type::Token(id),
        (ctor, arg) => crate::run_error!("{ctor:?} cannot take a {}", arg.type_name()),
    }
}

fn types_of(items: &List, ctor: &str) -> Vec<Type> {
    items
        .iter()
        .map(|v| match v {
            Value::Type(t) => (**t).clone(),
            other => crate::run_error!("{ctor} expects types, found a {}", other.type_name()),
        })
        .collect()
}

// What trying an arm's pattern (from some starting index) turned up.
enum ArmOutcome {
    // Pattern matched, arm has no guard -- take it, under the pattern's bindings.
    Body(ExprRef, Env),
    // Pattern matched, but `arms[idx]`'s guard still needs evaluating
    // (under `guard_env`, the pattern's bindings) before it's known
    // whether this arm is actually taken.
    Guard { idx: usize, guard: ExprRef, guard_env: Env },
}

// Tries `arms[start..]` in order, returning the first one whose pattern
// matches `value` -- either ready to run (no guard) or needing its guard
// evaluated first. `env` is the match expression's OWN env (unaffected by
// any arm's pattern bindings), extended per-candidate by match_pattern.
fn first_match(arms: &[(Pattern, Option<ExprRef>, ExprRef)], start: usize, value: &Value, env: &Env) -> Option<ArmOutcome> {
    arms[start..].iter().enumerate().find_map(|(i, (pat, guard, body))| {
        let mut bound = Bindings::default();
        if !match_pattern(pat, value, &mut bound) {
            return None;
        }
        // The resolver and this function must bind the same names in the
        // same order (resolve::pattern_vars is the shared contract).
        debug_assert_eq!(
            bound.len(),
            crate::resolve::pattern_vars(pat).len(),
            "match_pattern and resolve::pattern_vars disagree"
        );
        // Zero binders => no frame, exactly like the resolver's scope table.
        let env2 = bound.extend(env);
        Some(match guard {
            None => ArmOutcome::Body(*body, env2),
            Some(g) => ArmOutcome::Guard { idx: start + i, guard: *g, guard_env: env2 },
        })
    })
}

// Shared by Frame::MatchArms and Frame::MatchGuard's own retry-on-failed-
// guard branch -- both need to turn a `first_match` result into the next
// (Cont, Control) step, or panic the same way on total failure.
fn dispatch_arm_outcome(
    outcome: Option<ArmOutcome>,
    arms: Rc<Vec<(Pattern, Option<ExprRef>, ExprRef)>>,
    outer_env: Env,
    value: Value,
    rest: Cont,
) -> (Cont, Control) {
    match outcome {
        Some(ArmOutcome::Body(body, matched_env)) => (rest, Control::Eval(body, matched_env)),
        Some(ArmOutcome::Guard { idx, guard, guard_env }) => (
            Cont::cons(Frame::MatchGuard { arms, idx, outer_env, guard_env: guard_env.clone(), value }, rest),
            Control::Eval(guard, guard_env),
        ),
        None => crate::run_error!("match failed: no pattern matched the value"),
    }
}

// Native recursion here is bounded by the PATTERN's own size (as written in
// source), not by the data it's matched against. Pushes each binder's value
// into `out` in resolve::pattern_vars order; on failure `out` holds a
// partial result the caller discards.
fn match_pattern(pat: &Pattern, value: &Value, out: &mut Bindings) -> bool {
    match pat {
        Pattern::Var(_) => {
            out.push(value.clone());
            true
        }
        Pattern::Int(n) => matches!(value, Value::Int(v) if v == n),
        Pattern::Bool(b) => matches!(value, Value::Bool(v) if v == b),
        Pattern::Str(s) => matches!(value, Value::Str(v) if &**v == s.as_str()),
        Pattern::List(pats) => match value {
            Value::List(items) if items.len() == pats.len() => {
                pats.iter().zip(items.iter()).all(|(p, v)| match_pattern(p, v, out))
            }
            _ => false,
        },
        Pattern::Cons(head, tail) => match value {
            Value::List(items) if !items.is_empty() => {
                let (first, rest) = items.split_first().expect("a non-empty list has a head");
                match_pattern(head, first, out) && match_pattern(tail, &Value::List(rest), out)
            }
            _ => false,
        },
        // Width-tolerant, looked up by NAME -- see Pattern::Record's doc
        // comment (unchanged behavior).
        Pattern::Record(fields) => match value {
            Value::Record(entries) => {
                fields.iter().all(|(name, p)| find_field(entries, name).is_some_and(|v| match_pattern(p, v, out)))
            }
            _ => false,
        },
    }
}

fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        // Cross-type: `1 == 1.0` -- typecheck.rs's own Eq arm allows this
        // ONE extra pairing beyond its usual same-type-or-Dyn rule (see
        // its own numeric_pair comment), so this has to actually agree
        // with a real comparison, not just "false, different tags" the
        // way e.g. Int vs Str would be if it ever reached here.
        (Value::Int(x), Value::Float(y)) | (Value::Float(y), Value::Int(x)) => *x as f64 == *y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::Token(x), Value::Token(y)) => x == y,
        // Structural (spec section 9.2): a recursive type is a finite closed term, so
        // comparing it terminates.
        (Value::Type(x), Value::Type(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| value_eq(a, b))
        }
        // By name, not position -- O(n*m), not the O(n) a positional zip
        // (like List's own arm just above) would give, but NOT safely
        // interchangeable with one: every Value::Record built through
        // typecheck::check does land in the parser's canonical
        // sorted-by-name order, but nothing here can assume that -- a
        // caller that skips typecheck::check entirely (e.g. lib.rs's
        // run_untyped, or any future one) can construct two same-length,
        // different-field-set records that a pure positional zip would
        // wrongly call equal (`{a:1}` vs `{b:1}`, position 0 both `1`).
        // Comparing names at each position is what makes this correct
        // regardless of how the values were built, not just an
        // unnecessarily cautious version of the faster zip.
        //
        // Same field count, and every one of `x`'s fields present in `y`
        // under that name with an equal value. Deliberately EXACT, not
        // width-tolerant: `==`
        // compares the actual data two values carry, which is a
        // different question from typecheck::coerce's own "does this
        // value's type fit where that type is expected" -- width
        // subtyping is purely a static/matching-time notion (see
        // Pattern::Record's own doc comment: no value is ever narrowed
        // at a boundary), so a Dyn-sourced value width-coerced to fit a
        // narrower annotation still carries every field it always had,
        // and `==` still sees all of them. A value that fits a type is
        // not the same claim as two values being equal, the same way a
        // `Seconds` satisfying a `Meters` annotation (structural Tuple
        // typing) doesn't make two DIFFERENT Meters values compare
        // equal just because both satisfy the same annotation.
        (Value::Record(x), Value::Record(y)) => {
            x.len() == y.len() && x.iter().all(|(n, v)| find_field(y, n).is_some_and(|v2| value_eq(v, v2)))
        }
        _ => false,
    }
}

// Search outward through the marks for the nearest handler of `effect`
// (Meta::capture). The current segment and the marks passed become `k`, the
// first-class `resume` value bound in the handler body -- immutable, so the
// handler can apply it zero, one, or many times; the handler's own clause runs
// outside it (deep: `resume` reinstalls it; shallow: it does not). No arena
// needed here -- HandlerData.body is an ExprRef, just handed to Control::Eval
// as-is; the next loop iteration in `run` is what looks it up.
fn perform(cont: &mut Cont, meta: &mut Meta, effect: &str, payload: Value) -> Control {
    let Some((data, k)) = meta.capture(effect, cont) else { crate::run_error!("unhandled effect: {effect}") };
    Control::Eval(data.body, data.env.push2(payload, Value::Continuation(k)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(n: usize) -> Value {
        Value::List(List::from_vec(vec![Value::Int(0); n]))
    }

    // Two evaluations share one table: the closure the first builds (over the
    // frame `[a]`) is called by the second (a frame `[f]`), whose own nodes
    // are resolved after the closure's body.
    #[test]
    fn eval_pure_in_binds_a_frame_and_shares_the_table_across_runs() {
        let mut arena = Arena::new();
        let spans = SpanMap::new();
        let (x, a) = (arena.push(Expr::Var("x".into())), arena.push(Expr::Var("a".into())));
        let body = arena.push(Expr::BinOp(BinOp::Add, x, a));
        let lam = arena.push(Expr::Lambda("x".into(), None, body));
        let (f, five) = (arena.push(Expr::Var("f".into())), arena.push(Expr::Int(5)));
        let call = arena.push(Expr::App(f, five));
        let mut table = Resolved::default();
        let Pure::Done(closure) = eval_pure_in(&mut arena, lam, &["a".to_string()], vec![Value::Int(10)], |_| None, &mut table, &spans, 100) else {
            panic!("the lambda must evaluate")
        };
        let Pure::Done(sum) = eval_pure_in(&mut arena, call, &["f".to_string()], vec![closure], |_| None, &mut table, &spans, 100) else {
            panic!("the call must evaluate")
        };
        assert_eq!(sum.as_int(), 15);
        let Pure::Done(pair) = eval_pure_in(&mut arena, five, &[], Vec::new(), |_| None, &mut table, &spans, 100) else { panic!("a literal") };
        assert_eq!(pair.as_int(), 5);
    }

    // One root evaluated under two different frames: each evaluation runs its
    // own copy, so the closures it builds read their own `a` and the shared
    // table never sees one node under two layouts.
    #[test]
    fn eval_pure_in_evaluates_one_root_under_two_frames() {
        let mut arena = Arena::new();
        let spans = SpanMap::new();
        let (x, a) = (arena.push(Expr::Var("x".into())), arena.push(Expr::Var("a".into())));
        let body = arena.push(Expr::BinOp(BinOp::Add, x, a));
        let lam = arena.push(Expr::Lambda("x".into(), None, body));
        let (f, five) = (arena.push(Expr::Var("f".into())), arena.push(Expr::Int(5)));
        let call = arena.push(Expr::App(f, five));
        let mut table = Resolved::default();
        let build = |arena: &mut Arena, names: &[&str], values: Vec<Value>, table: &mut Resolved| {
            let names: Vec<String> = names.iter().map(|n| n.to_string()).collect();
            let Pure::Done(closure) = eval_pure_in(arena, lam, &names, values, |_| None, table, &spans, 100) else { panic!("the lambda must evaluate") };
            closure
        };
        let c1 = build(&mut arena, &["a"], vec![Value::Int(10)], &mut table);
        let c2 = build(&mut arena, &["z", "a"], vec![Value::Int(0), Value::Int(20)], &mut table);
        for (closure, want) in [(c1, 15), (c2, 25)] {
            let Pure::Done(sum) = eval_pure_in(&mut arena, call, &["f".to_string()], vec![closure], |_| None, &mut table, &spans, 100) else { panic!("the call must evaluate") };
            assert_eq!(sum.as_int(), want);
        }
        build(&mut arena, &["a"], vec![Value::Int(30)], &mut table);
    }

    // A subtree `reuse` has a value for is read from the frame, not copied or
    // run: `zzz` is unbound, so running it would fail, and the copy is the
    // Let, one variable for the reused value, the Tuple and its two Vars.
    #[test]
    fn eval_pure_in_runs_a_reused_subtree_as_its_value() {
        let mut arena = Arena::new();
        let spans = SpanMap::new();
        let zzz = arena.push(Expr::Var("zzz".into()));
        let (b1, b2) = (arena.push(Expr::Var("b".into())), arena.push(Expr::Var("b".into())));
        let pair = arena.push(Expr::Tuple(vec![b1, b2]));
        let root = arena.push(Expr::Let("b".into(), None, zzz, pair));
        let len = arena.len();
        let mut table = Resolved::default();
        let reuse = |n: ExprRef| (n == zzz).then_some(Value::Int(7));
        let Pure::Done(v) = eval_pure_in(&mut arena, root, &[], Vec::new(), reuse, &mut table, &spans, 100) else { panic!("the reused subtree must not run") };
        assert_eq!(v.to_string(), "[7, 7]");
        assert_eq!(arena.len() - len, 5);
    }

    // The debug cross-check: a reused value that differs from what running the
    // subtree gives (here 4 for `1 + 2`) is caught.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "a reused value differs from running its subtree again")]
    fn eval_pure_in_debug_catches_a_reused_value_that_differs_from_running_it() {
        let mut arena = Arena::new();
        let spans = SpanMap::new();
        let (one, two) = (arena.push(Expr::Int(1)), arena.push(Expr::Int(2)));
        let sum = arena.push(Expr::BinOp(BinOp::Add, one, two));
        let (b1, b2) = (arena.push(Expr::Var("b".into())), arena.push(Expr::Var("b".into())));
        let pair = arena.push(Expr::Tuple(vec![b1, b2]));
        let root = arena.push(Expr::Let("b".into(), None, sum, pair));
        let mut table = Resolved::default();
        let reuse = |n: ExprRef| (n == sum).then_some(Value::Int(4));
        eval_pure_in(&mut arena, root, &[], Vec::new(), reuse, &mut table, &spans, 100);
    }

    #[test]
    fn a_long_path_is_elided_in_the_middle() {
        let path: Vec<String> = (0..1000).map(|i| format!("element {i}")).collect();
        let text = path_text(&path);
        assert!(text.starts_with("element 0, element 1,") && text.ends_with("element 998, element 999"), "{text}");
        assert!(text.contains("... 980 more ...") && text.len() < 300, "{text}");
        assert_eq!(path_text(&path[..20]), path[..20].join(", "));
    }

    // The scalar-`Or` fast path must agree with the walker for every value.
    #[test]
    fn a_scalar_or_agrees_with_the_walker() {
        let list = Value::List(List::from_vec(vec![Value::Int(1)]));
        let values = [Value::Int(1), Value::Float(1.0), Value::Bool(true), list];
        let ors = [
            Test::Or(vec![]),
            Test::Or(vec![Test::Int, Test::Float]),
            Test::Or(vec![Test::Never, Test::Bool]),
            Test::Or(vec![Test::Int, Test::list_of(Box::new(Test::Int))]),
        ];
        for t in &ors {
            for v in &values {
                assert_eq!(test_holds(t, &[], &[], v), walk::<false>(t, &[], &[], v, &mut Vec::new(), &mut None).is_none());
            }
        }
    }

    #[test]
    fn test_holds_covers_every_shape() {
        let record = Value::Record(Rc::new(vec![("x".to_string(), Value::Int(1)), ("y".to_string(), Value::Int(2))]));
        let closure = Value::Builtin(Builtin::Len);
        let holds = |t: &Test, v: &Value| test_holds(t, &[], &[], v);
        assert!(holds(&Test::Any, &Value::Bool(true)) && !holds(&Test::Never, &Value::Bool(true)));
        assert!(holds(&Test::Int, &Value::Int(1)) && !holds(&Test::Int, &Value::Float(1.0)));
        assert!(holds(&Test::Float, &Value::Float(1.0)) && !holds(&Test::Float, &Value::Int(1)));
        assert!(holds(&Test::Bool, &Value::Bool(false)) && !holds(&Test::Bool, &Value::Int(0)));
        assert!(holds(&Test::Str, &Value::Str(Text::from("a"))) && !holds(&Test::Str, &Value::Int(0)));
        assert!(holds(&Test::List, &list(0)) && !holds(&Test::List, &record));
        assert!(holds(&Test::Fun, &closure) && !holds(&Test::Fun, &Value::Int(0)));
        assert!(holds(&Test::Token(7), &Value::Token(7)) && !holds(&Test::Token(7), &Value::Token(8)));
        assert!(!holds(&Test::Token(7), &Value::Int(7)));
        let tuple2 = Test::TupleOf(vec![Test::Any, Test::Any]);
        assert!(holds(&tuple2, &list(2)) && !holds(&tuple2, &list(3)) && !holds(&tuple2, &record));
        let fields = |names: &[&str]| Test::RecordOf(names.iter().map(|n| (n.to_string(), Test::Any)).collect());
        assert!(holds(&fields(&["x"]), &record) && holds(&fields(&["x", "y"]), &record));
        assert!(!holds(&fields(&["z"]), &record) && !holds(&fields(&["x"]), &list(1)));
        let int_or_token = Test::Or(vec![Test::Int, Test::Token(7)]);
        assert!(holds(&int_or_token, &Value::Token(7)) && holds(&int_or_token, &Value::Int(1)) && !holds(&int_or_token, &Value::Bool(true)));
        assert!(!holds(&Test::Or(vec![]), &Value::Int(1)));
    }

    // `is` on a type whose Vec length is a variable has no length to compare
    // against: a run-time error, not a silent shape-only test.
    #[test]
    fn is_rejects_a_type_with_a_free_length_variable() {
        let list = Type::List(Hc::new(Type::Int));
        let open = Type::Indexed(Hc::new(list), Rc::new(IndexExpr::Var("n".to_string())));
        let err = std::panic::catch_unwind(|| crate::typecheck::lower_test(&open)).expect("lower_test must not panic");
        assert_eq!(err.unwrap_err(), "is: type has a free length variable");
        assert!(crate::typecheck::lower_test(&Type::List(Hc::new(Type::Int))).is_ok());
    }

    #[test]
    fn deep_tests_follow_the_value_and_explain_names_the_failing_element() {
        let ints = |xs: &[i64]| Value::List(List::from_vec(xs.iter().map(|n| Value::Int(*n)).collect()));
        let list_of_int = Test::list_of(Box::new(Test::Int));
        assert!(test_holds(&list_of_int, &[], &[], &ints(&[1, 2])) && test_holds(&list_of_int, &[], &[], &ints(&[])));
        let mixed = Value::List(List::from_vec(vec![Value::Int(1), Value::Bool(true)]));
        assert!(!test_holds(&list_of_int, &[], &[], &mixed));
        let mut path = Vec::new();
        assert_eq!(explain(&list_of_int, &[], &[], &mixed, &mut path).type_name(), "Bool");
        assert_eq!(path, ["element 1"]);

        let record = Value::Record(Rc::new(vec![("a".to_string(), Value::Int(1)), ("b".to_string(), mixed.clone())]));
        let nested = Test::RecordOf(vec![("a".to_string(), Test::Int), ("b".to_string(), list_of_int.clone())]);
        assert!(!test_holds(&nested, &[], &[], &record));
        let mut path = Vec::new();
        assert_eq!(explain(&nested, &[], &[], &record, &mut path).type_name(), "Bool");
        assert_eq!(path, ["field b", "element 1"]);

        // let rec L = Union(Tuple(Int, L), Bool): the type body is defs[0], reached through Ref(0).
        let defs = [Test::Or(vec![Test::TupleOf(vec![Test::Int, Test::reference(0)]), Test::Bool])];
        let pair = |a: Value, b: Value| Value::List(List::from_vec(vec![a, b]));
        let good = pair(Value::Int(1), pair(Value::Int(2), Value::Bool(true)));
        let bad = pair(Value::Int(1), pair(Value::Int(2), Value::Int(3)));
        assert!(test_holds(&Test::reference(0), &defs, &[], &good) && !test_holds(&Test::reference(0), &defs, &[], &bad));
        let mut path = Vec::new();
        assert_eq!(explain(&Test::reference(0), &defs, &[], &bad, &mut path).type_name(), "Int");
        assert_eq!(path, ["element 1", "element 1"]);
    }

    // The walker uses O(1) native stack: this runs on the 2 MiB default test
    // thread, where a recursive check (or a recursive Drop, hence the forgets)
    // overflows after a few thousand levels.
    #[test]
    fn the_walker_handles_a_very_deep_spine_on_a_small_stack() {
        const DEPTH: usize = 100_000;
        let pair = |a: Value, b: Value| Value::List(List::from_vec(vec![a, b]));
        let defs = [Test::Or(vec![Test::TupleOf(vec![Test::Int, Test::reference(0)]), Test::Bool])];
        let chain = |leaf: Value| (0..DEPTH).fold(leaf, |tail, i| pair(Value::Int(i as i64), tail));
        let good = chain(Value::Bool(true));
        assert!(test_holds(&Test::reference(0), &defs, &[], &good));
        let bad = chain(Value::Str(Text::from("x")));
        assert!(!test_holds(&Test::reference(0), &defs, &[], &bad));
        let mut path = Vec::new();
        assert_eq!(explain(&Test::reference(0), &defs, &[], &bad, &mut path).type_name(), "Str");
        assert_eq!(path.len(), DEPTH);
        // nested single-element lists: [[[...[Int]...]]] against ListOf(Ref) where the alias is [self] | Int
        let list_defs = [Test::Or(vec![Test::Int, Test::list_of(Box::new(Test::reference(0)))])];
        let nest = (0..DEPTH).fold(Value::Int(1), |inner, _| Value::List(List::from_vec(vec![inner])));
        assert!(test_holds(&Test::reference(0), &list_defs, &[], &nest));
        // the nested chains cannot be dropped recursively on this stack
        std::mem::forget((good, bad, nest));
    }

    // The old recursive definitions, kept as the oracle for the walker: same
    // answer, same path, same offending value on random (test, defs, value).
    fn holds_recursive(test: &Test, defs: &[Test], v: &Value) -> bool {
        match test {
            Test::Elems { elem: t, len: Length::Any, .. } => matches!(v, Value::List(items) if items.iter().all(|x| holds_recursive(t, defs, x))),
            Test::Elems { elem: t, len: Length::Exact(n), .. } => {
                matches!(v, Value::List(items) if items.len() == *n && items.iter().all(|x| holds_recursive(t, defs, x)))
            }
            Test::Elems { len: Length::Slot(_), .. } => unreachable!("the oracle's tests have their slots substituted"),
            Test::TupleOf(ts) => matches!(v, Value::List(items) if items.len() == ts.len() && items.iter().zip(ts).all(|(x, t)| holds_recursive(t, defs, x))),
            Test::RecordOf(fs) => matches!(v, Value::Record(fields) if fs.iter().all(|(n, t)| find_field(fields, n).is_some_and(|x| holds_recursive(t, defs, x)))),
            Test::Ref(i, _) => holds_recursive(&defs[*i], defs, v),
            Test::Or(alternatives) => alternatives.iter().any(|t| holds_recursive(t, defs, v)),
            scalar => leaf(scalar, v).expect("scalar"),
        }
    }

    fn explain_recursive<'a>(test: &Test, defs: &[Test], v: &'a Value, path: &mut Vec<String>) -> &'a Value {
        match (test, v) {
            (Test::Ref(i, _), _) => explain_recursive(&defs[*i], defs, v, path),
            (Test::Elems { len: Length::Exact(n), .. }, Value::List(items)) if items.len() != *n => v,
            (Test::Elems { elem: t, .. }, Value::List(items)) => match items.iter().position(|x| !holds_recursive(t, defs, x)) {
                Some(i) => {
                    path.push(format!("element {i}"));
                    explain_recursive(t, defs, items.get(i).expect("a failing element exists"), path)
                }
                None => v,
            },
            (Test::TupleOf(ts), Value::List(items)) if items.len() == ts.len() => {
                match items.iter().zip(ts).position(|(x, t)| !holds_recursive(t, defs, x)) {
                    Some(i) => {
                        path.push(format!("element {i}"));
                        explain_recursive(&ts[i], defs, items.get(i).expect("a failing element exists"), path)
                    }
                    None => v,
                }
            }
            (Test::RecordOf(fs), Value::Record(fields)) => {
                for (name, t) in fs {
                    if let Some(x) = find_field(fields, name).filter(|x| !holds_recursive(t, defs, x)) {
                        path.push(format!("field {name}"));
                        return explain_recursive(t, defs, x, path);
                    }
                }
                v
            }
            (Test::Or(alternatives), _) => {
                let mut best: (Vec<String>, &'a Value) = (Vec::new(), v);
                for alt in alternatives {
                    let mut alt_path = Vec::new();
                    let found = explain_recursive(alt, defs, v, &mut alt_path);
                    if alt_path.len() > best.0.len() {
                        best = (alt_path, found);
                    }
                }
                path.extend(best.0);
                best.1
            }
            _ => v,
        }
    }

    // xorshift64: fixed seed, no dependency.
    struct Rng(u64);
    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }
    }

    const NAMES: [&str; 3] = ["a", "b", "c"];

    // `refs`: a Ref(0|1) may appear (only under a container, like real tests).
    fn random_test(r: &mut Rng, depth: u32, refs: bool) -> Test {
        match if depth == 0 { r.below(5) } else { r.below(11) } {
            0 => Test::Int,
            1 => Test::Bool,
            2 => Test::Any,
            3 => Test::Str,
            4 if refs && r.below(2) == 0 => Test::reference(r.below(2)),
            4 => Test::Float,
            5 => Test::list_of(Box::new(random_test(r, depth - 1, true))),
            6 if r.below(2) == 0 => Test::list_len_arg(Box::new(random_test(r, depth - 1, true)), r.below(4)),
            6 => Test::list_len(Box::new(random_test(r, depth - 1, true)), r.below(4)),
            7 => Test::TupleOf((0..r.below(4)).map(|_| random_test(r, depth - 1, true)).collect()),
            8 => Test::RecordOf((0..r.below(4)).map(|i| (NAMES[i].to_string(), random_test(r, depth - 1, true))).collect()),
            _ => Test::Or((0..r.below(4)).map(|_| random_test(r, depth - 1, refs)).collect()),
        }
    }

    fn random_value(r: &mut Rng, depth: u32) -> Value {
        match if depth == 0 { r.below(4) } else { r.below(8) } {
            0 => Value::Int(1),
            1 => Value::Bool(true),
            2 => Value::Str(Text::from("s")),
            3 => Value::Float(1.0),
            4 | 5 => Value::List(List::from_vec((0..r.below(4)).map(|_| random_value(r, depth - 1)).collect())),
            _ => {
                let mut fields = Vec::new();
                for name in NAMES {
                    if r.below(3) > 0 {
                        fields.push((name.to_string(), random_value(r, depth - 1)));
                    }
                }
                Value::Record(Rc::new(fields))
            }
        }
    }

    // The slot values the random tests' ListLenArg(_, 0..4) read: one of each kind.
    fn test_lens() -> Vec<Len> {
        vec![Len::Exact(2), Len::Any, Len::NoValue(Rc::from("n")), Len::Never]
    }

    // `test` with each ListLenArg replaced by what its slot means, for the
    // recursive oracle: Exact(k) a literal ListLen, Any a plain ListOf, and a
    // slot with no value (NoValue, Never) a test nothing passes.
    fn subst_lens(test: &Test, lens: &[Len]) -> Test {
        let go = |t: &Test| subst_lens(t, lens);
        match test {
            Test::Elems { elem: t, len: Length::Slot(slot), .. } => match &lens[*slot] {
                Len::Exact(k) => Test::list_len(Box::new(go(t)), *k),
                Len::Any => Test::list_of(Box::new(go(t))),
                Len::NoValue(_) | Len::Never => Test::Never,
            },
            Test::Elems { elem: t, len: Length::Any, .. } => Test::list_of(Box::new(go(t))),
            Test::Elems { elem: t, len: Length::Exact(k), .. } => Test::list_len(Box::new(go(t)), *k),
            Test::TupleOf(ts) => Test::TupleOf(ts.iter().map(go).collect()),
            Test::RecordOf(fs) => Test::RecordOf(fs.iter().map(|(name, t)| (name.clone(), go(t))).collect()),
            Test::Or(ts) => Test::Or(ts.iter().map(go).collect()),
            scalar => scalar.clone(),
        }
    }

    #[test]
    fn a_length_slot_is_exact_any_or_no_value() {
        let lens = test_lens();
        let list = |n: i64| Value::List(List::from_vec((0..n).map(Value::Int).collect()));
        let arg = |slot| Test::list_len_arg(Box::new(Test::Int), slot);
        assert!(test_holds(&arg(0), &[], &lens, &list(2)));
        assert!(!test_holds(&arg(0), &[], &lens, &list(3)));
        assert!(test_holds(&arg(1), &[], &lens, &list(5)));
        // Any skips only the length: the elements are still tested.
        assert!(!test_holds(&arg(1), &[], &lens, &Value::List(List::from_vec(vec![Value::Bool(true)]))));
        assert!(!test_holds(&arg(2), &[], &lens, &list(2)));
        assert!(!test_holds(&arg(3), &[], &lens, &list(0)));
        // A NoValue slot is reported only when the walk reaches it on a list,
        // inside a failing union alternative too.
        assert_eq!(reached_no_value(&arg(2), &[], &lens, &list(1)), Some("n"));
        assert_eq!(reached_no_value(&arg(2), &[], &lens, &Value::Int(1)), None);
        assert_eq!(reached_no_value(&arg(0), &[], &lens, &list(1)), None);
        let or = Test::Or(vec![arg(2), Test::Bool]);
        assert_eq!(reached_no_value(&or, &[], &lens, &list(1)), Some("n"));
        assert!(test_holds(&or, &[], &lens, &Value::Bool(true)));
        // One slot shared by every element: [Vec(n)] with n = 2.
        let nested = Test::list_of(Box::new(Test::list_len_arg(Box::new(Test::Any), 0)));
        assert!(test_holds(&nested, &[], &lens, &Value::List(List::from_vec(vec![list(2), list(2)]))));
        let bad = Value::List(List::from_vec(vec![list(2), list(1)]));
        assert!(!test_holds(&nested, &[], &lens, &bad));
        let mut path = Vec::new();
        explain(&nested, &[], &lens, &bad, &mut path);
        assert_eq!(path, ["element 1"]);
    }

    // Two views of one block are two lists: the walk's memo of decided (alias, value) pairs
    // must not equate them (a pass for the short view says nothing about the longer one).
    #[test]
    fn the_walk_memo_tells_two_views_of_one_block_apart() {
        let short = List::empty().cons(Value::Int(1)).cons(Value::Int(2));
        let long = short.cons(Value::Str(Text::from("x")));
        assert_eq!(short.identity().0, long.identity().0, "the cons claimed a slot of the same block");
        let defs = [Test::list_of(Box::new(Test::Int))];
        let pair = Test::TupleOf(vec![Test::reference(0), Test::reference(0)]);
        let value = Value::List(List::from_vec(vec![Value::List(short.clone()), Value::List(long)]));
        assert!(!test_holds(&pair, &defs, &[], &value));
        assert!(test_holds(&pair, &defs, &[], &Value::List(List::from_vec(vec![Value::List(short.clone()), Value::List(short)]))));
    }

    // The same for two views that differ only at the low end (a snoc claim).
    #[test]
    fn the_walk_memo_tells_two_snoc_views_of_one_block_apart() {
        let one = |v: Value| List::from_vec(vec![v]);
        let base = List::from_vec(vec![Value::Int(1), Value::Int(2), Value::Int(3)]).append(&one(Value::Int(4))).append(&one(Value::Int(5)));
        let long = base.append(&one(Value::Str(Text::from("x"))));
        assert_eq!(base.identity().0, long.identity().0, "the snoc claimed a slot of the same block");
        assert_ne!(base.identity(), long.identity());
        let defs = [Test::list_of(Box::new(Test::Int))];
        let pair = Test::TupleOf(vec![Test::reference(0), Test::reference(0)]);
        assert!(!test_holds(&pair, &defs, &[], &Value::List(List::from_vec(vec![Value::List(base.clone()), Value::List(long)]))));
        assert!(test_holds(&pair, &defs, &[], &Value::List(List::from_vec(vec![Value::List(base.clone()), Value::List(base)]))));
    }

    // `[nan] == [nan]` is false, so equality may not shortcut on a shared list.
    #[test]
    fn a_list_holding_nan_is_not_equal_to_itself() {
        let l = Value::List(List::from_vec(vec![Value::Float(f64::NAN)]));
        assert!(!value_eq(&l, &l.clone()));
    }

    #[test]
    fn the_walker_agrees_with_the_recursive_definition_on_random_cases() {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let lens = test_lens();
        let (mut failing, total) = (0, 30_000);
        for _ in 0..total {
            let defs = [random_test(&mut r, 3, false), random_test(&mut r, 3, false)];
            let test = random_test(&mut r, 4, false);
            let v = random_value(&mut r, 4);
            // The oracle sees each slot's meaning spelled out as a plain test.
            let (want_test, want_defs) = (subst_lens(&test, &lens), defs.each_ref().map(|d| subst_lens(d, &lens)));
            let want = holds_recursive(&want_test, &want_defs, &v);
            assert_eq!(want, test_holds(&test, &defs, &lens, &v), "holds differs: {test:?} {defs:?} {v}");
            if !want {
                failing += 1;
                let (mut want_path, mut got_path) = (Vec::new(), Vec::new());
                let want_found = explain_recursive(&want_test, &want_defs, &v, &mut want_path);
                let got_found = explain(&test, &defs, &lens, &v, &mut got_path);
                assert_eq!(want_path, got_path, "path differs: {test:?} {defs:?} {v}");
                assert!(std::ptr::eq(want_found, got_found), "found differs: {test:?} {defs:?} {v}");
            }
        }
        assert!(failing > total / 10 && failing < total, "the generator should produce both outcomes: {failing}/{total} failing");
    }

    // The soundness net for block seals: values that share blocks (made only by the real list
    // operations), checked by random tests in random order with repeats, must get the
    // oracle's verdict, and failures the oracle's path, whatever was checked before. The
    // length slots change between rounds, so a test that reads one must never be sealed.
    fn element(r: &mut Rng, pool: &[Value]) -> Value {
        match r.below(8) {
            0 => Value::Bool(true),
            1 => Value::Str(Text::from("s")),
            2 if !pool.is_empty() => pool[r.below(pool.len())].clone(),
            _ => Value::Int(r.below(3) as i64),
        }
    }

    fn random_pool(r: &mut Rng, size: usize) -> Vec<Value> {
        let mut pool = vec![Value::List(List::empty())];
        while pool.len() < size {
            let pick = |r: &mut Rng, pool: &[Value]| pool[r.below(pool.len())].clone();
            let list = |v: Value| match v {
                Value::List(l) => Some(l),
                _ => None,
            };
            let next = match r.below(10) {
                0 => Some(Value::List(List::from_vec((0..r.below(6)).map(|_| element(r, &pool)).collect()))),
                1..=3 => list(pick(r, &pool)).map(|l| Value::List(l.cons(element(r, &pool)))),
                4 => list(pick(r, &pool)).and_then(|l| l.split_first().map(|(_, t)| Value::List(t))),
                5 => list(pick(r, &pool)).zip(list(pick(r, &pool))).map(|(a, b)| Value::List(a.append(&b))),
                6 => Some(Value::List(List::from_vec(vec![pick(r, &pool), pick(r, &pool)]))),
                7 => Some(Value::List(List::from_vec(vec![pick(r, &pool)]))),
                8 => Some(Value::Record(Rc::new(vec![("a".to_string(), pick(r, &pool)), ("b".to_string(), element(r, &pool))]))),
                _ => list(pick(r, &pool)).map(|l| Value::List(l.cons(Value::Int(1)).cons(Value::Int(2)))),
            };
            pool.extend(next);
        }
        pool
    }

    // What a test needs before it runs: its seal ids.
    fn prepare(test: &mut Test, defs: &mut [Test; 2]) {
        crate::expr::assign_seals(test, defs, &mut Then::Done);
    }

    #[test]
    fn checks_agree_with_the_recursive_definition_on_values_that_share_blocks() {
        let mut r = Rng(0x2545_F491_4F6C_DD1D);
        let pool = random_pool(&mut r, 160);
        let by_hand = [
            Test::list_of(Box::new(Test::Int)),
            Test::list_of(Box::new(Test::Or(vec![Test::Int, Test::Str]))),
            Test::list_of(Box::new(Test::list_of(Box::new(Test::Int)))),
            Test::list_len(Box::new(Test::Int), 2),
            Test::list_of(Box::new(Test::TupleOf(vec![Test::Int, Test::Int]))),
        ];
        let mut tests: Vec<(Test, [Test; 2])> = Vec::new();
        for i in 0..60 {
            let mut defs = [random_test(&mut r, 3, false), random_test(&mut r, 3, false)];
            let mut test = by_hand.get(i).cloned().unwrap_or_else(|| random_test(&mut r, 4, false));
            prepare(&mut test, &mut defs);
            tests.push((test, defs));
        }
        let (mut failing, mut total) = (0, 0);
        for round in 0..4 {
            let mut lens = test_lens();
            lens.rotate_left(round);
            for _ in 0..5_000 {
                let (test, defs) = &tests[r.below(tests.len())];
                let v = &pool[r.below(pool.len())];
                let (want_test, want_defs) = (subst_lens(test, &lens), defs.each_ref().map(|d| subst_lens(d, &lens)));
                let want = holds_recursive(&want_test, &want_defs, v);
                assert_eq!(want, test_holds(test, defs, &lens, v), "holds differs: {test:?} {defs:?} {v}");
                total += 1;
                if !want {
                    failing += 1;
                    let (mut want_path, mut got_path) = (Vec::new(), Vec::new());
                    let want_found = explain_recursive(&want_test, &want_defs, v, &mut want_path);
                    let got_found = explain(test, defs, &lens, v, &mut got_path);
                    assert_eq!(want_path, got_path, "path differs: {test:?} {defs:?} {v}");
                    assert!(std::ptr::eq(want_found, got_found), "found differs: {test:?} {defs:?} {v}");
                }
            }
        }
        assert!(failing > total / 20 && failing < total * 19 / 20, "both outcomes expected: {failing}/{total} failing");
    }
}
