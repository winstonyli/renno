use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::cont::{Cont, ContNode, Frame};
use crate::env::{Env, PRELUDE};
use crate::expr::{Arena, BinOp, CheckMode, Expr, ExprRef, Pattern, SpanMap, Test};
use crate::frame::Bindings;
use crate::resolve::{is_direct_group, resolve, Resolved, VarRef};
use crate::span::Span;
use crate::util::find_field;
use crate::value::{Builtin, HandlerData, Value};

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
    run_loop(arena, Control::Eval(expr, env), Cont::nil(), spans, &resolved)
}

// Calls a renno function VALUE from native Rust code (used by fold/map's
// Builtin dispatch below to invoke the callback once per element) by
// seeding the same trampoline the AppArg frame's own dispatch already
// implements, rather than duplicating that Closure/Continuation/Builtin
// matching logic a second time.
//
// The seeded continuation is fresh (Cont::nil()), not the caller's real
// one -- so an effect performed inside `func` can NEVER reach a `handle`
// that lexically wraps the outer fold/map call; it always panics
// "unhandled effect", regardless of what the calling program looks like.
// Threading the real outer continuation through a native re-entrant call
// like this is a substantially harder problem (effect handlers crossing
// an FFI-like boundary) that this feature doesn't attempt. In practice
// this is the expected shape for a structural-recursion primitive anyway
// -- fold/map are conventionally pure transformations.
pub fn apply(arena: &Arena, func: Value, arg: Value, spans: &SpanMap, resolved: &Resolved) -> Value {
    run_loop(arena, Control::Apply(arg), Cont::cons(Frame::AppArg { func, callee_span: None }, Cont::nil()), spans, resolved)
}

fn run_loop(arena: &Arena, mut control: Control, mut cont: Cont, spans: &SpanMap, resolved: &Resolved) -> Value {
    loop {
        match control {
            Control::Eval(expr, env) => {
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
                Expr::Check(operand, spec) => {
                    cont = Cont::cons(Frame::Check { spec: spec.clone() }, cont);
                    control = Control::Eval(*operand, env);
                }
                Expr::Int(n) => control = Control::Apply(Value::Int(*n)),
                Expr::Float(x) => control = Control::Apply(Value::Float(*x)),
                Expr::Bool(b) => control = Control::Apply(Value::Bool(*b)),
                Expr::Str(s) => control = Control::Apply(Value::Str(Rc::from(s.as_str()))),
                Expr::Token(id) => control = Control::Apply(Value::Token(*id)),
                // Always 2+ items (see Expr::Tuple's own doc comment), so
                // unlike ListLit there's no empty-case to special-case --
                // same left-to-right Frame::ListElems machinery either way,
                // building a plain Value::List.
                Expr::Tuple(items) => {
                    let mut remaining = items.clone();
                    remaining.reverse();
                    let first = remaining.pop().expect("Expr::Tuple always has at least 2 items");
                    cont = Cont::cons(Frame::ListElems { remaining, done: Vec::new(), env: env.clone() }, cont);
                    control = Control::Eval(first, env);
                }
                // Same left-to-right remaining/done accumulation shape as
                // LetRec's own non-group fallback just below (Frame::
                // LetRecBody) -- not ListElems, since the final wrap
                // differs (Value::Record, not Value::List; see
                // Frame::RecordElems's own doc comment, including how it
                // differs from LetRecBody's own shape). Always 1+ fields:
                // parser::parse_record_fields rejects `{}` (see its own
                // doc comment), the same "no
                // empty case to special-case" guarantee Tuple's own arm
                // above relies on.
                Expr::Record(fields) => {
                    let names: Rc<Vec<String>> = Rc::new(fields.iter().map(|(n, _)| n.clone()).collect());
                    let mut remaining: Vec<ExprRef> = fields.iter().map(|(_, v)| *v).collect();
                    remaining.reverse();
                    let first = remaining.pop().expect("parse_record_fields rejects an empty record");
                    cont = Cont::cons(
                        Frame::RecordElems { names, remaining, done: Vec::new(), env: env.clone() },
                        cont,
                    );
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
                // supported on that untyped path.
                Expr::FieldAccess(..) => unreachable!(
                    "Expr::FieldAccess reached machine::run directly -- `.field` needs typecheck::check first, see its own doc comment"
                ),
                Expr::ListLit(items) => {
                    if items.is_empty() {
                        control = Control::Apply(Value::List(Rc::new(Vec::new())));
                    } else {
                        // Reversed so ListElems can pop() (O(1)) instead of
                        // remove(0) as each element finishes.
                        let mut remaining = items.clone();
                        remaining.reverse();
                        let first = remaining.pop().unwrap();
                        cont = Cont::cons(
                            Frame::ListElems { remaining, done: Vec::new(), env: env.clone() },
                            cont,
                        );
                        control = Control::Eval(first, env);
                    }
                }
                Expr::Var(name) => {
                    control = Control::Apply(match resolved.get(expr) {
                        Some(VarRef::Local { hops, slot }) => env.get(hops, slot),
                        Some(VarRef::Prelude(i)) => Value::Builtin(PRELUDE[i as usize].1),
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
                    if is_direct_group(arena, bindings) {
                        // Every value is a direct Lambda: build the group
                        // straight from the lambda bodies -- evaluating a
                        // Lambda is pure, so skipping it is unobservable --
                        // and bind all the RecClosures in ONE frame [names…],
                        // the layout resolve.rs assigns the `let rec` body.
                        let bodies: Rc<[ExprRef]> = bindings
                            .iter()
                            .map(|(_, _, v)| match &arena[*v] {
                                Expr::Lambda(_, _, b) => *b,
                                _ => unreachable!("is_direct_group checked"),
                            })
                            .collect();
                        let mut frame = Bindings::default();
                        for i in 0..bodies.len() {
                            frame.push(Value::RecClosure(bodies.clone(), i, env.clone()));
                        }
                        control = Control::Eval(*body, frame.extend(&env));
                    } else {
                        // Not a function group (spec §2 fallback): values
                        // evaluate in the OUTER scope, left to right, and the
                        // names are bound plainly (non-recursively) for the
                        // body only. Reversed, same reason as ListLit: pop().
                        let mut remaining: Vec<ExprRef> = bindings.iter().map(|(_, _, v)| *v).collect();
                        remaining.reverse();
                        let first = remaining.pop().unwrap(); // parser never emits an empty group
                        cont = Cont::cons(
                            Frame::LetRecBody { remaining, done: Vec::new(), body: *body, env: env.clone() },
                            cont,
                        );
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
                    cont = Cont::cons(Frame::PerformPayload { effect: effect.clone() }, cont);
                    control = Control::Eval(*payload, env);
                }
                Expr::Handle { body, handler } => {
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
            Control::Apply(value) => match &*cont.0 {
                ContNode::Nil => return value,
                ContNode::Frame(frame, rest) => {
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
                            let func = func.clone();
                            cont = rest;
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
                                    // resume(value): splice the captured
                                    // continuation back in front of whatever
                                    // comes after this call. Cloning k here
                                    // (each time resume is invoked) is just
                                    // an Rc clone -- multi-shot is calling
                                    // this arm more than once with the same k.
                                    cont = Cont::append(&k, &cont);
                                    control = Control::Apply(value);
                                }
                                Value::Builtin(b) => {
                                    set_current_span(callee_span);
                                    control = Control::Apply(collect_builtin_arg(arena, b, Vec::new(), value, spans, resolved));
                                }
                                Value::PartialBuiltin(b, prev_args) => {
                                    let args = (*prev_args).clone();
                                    set_current_span(callee_span);
                                    control = Control::Apply(collect_builtin_arg(arena, b, args, value, spans, resolved));
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
                                    cont = Cont::cons(Frame::HandlerMark(data), cont);
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
                        Frame::LetRecBody { remaining, done, body, env } => {
                            let (mut remaining, mut done, body, env) = (remaining.clone(), done.clone(), *body, env.clone());
                            done.push(value);
                            cont = rest;
                            match remaining.pop() {
                                Some(next) => {
                                    cont = Cont::cons(Frame::LetRecBody { remaining, done, body, env: env.clone() }, cont);
                                    control = Control::Eval(next, env);
                                }
                                None => {
                                    // Non-group fallback: bind every value
                                    // (same order as the names) in one frame.
                                    let mut frame = Bindings::default();
                                    for v in done {
                                        frame.push(v);
                                    }
                                    control = Control::Eval(body, frame.extend(&env));
                                }
                            }
                        }
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
                        // Vec clones here are O(remaining/done length) per
                        // element -- fine for typical list-literal sizes;
                        // a large literal would make this O(n^2) overall.
                        // Worth an index-based rewrite if that ever matters.
                        Frame::ListElems { remaining, done, env } => {
                            let (mut remaining, mut done, env) = (remaining.clone(), done.clone(), env.clone());
                            done.push(value);
                            cont = rest;
                            match remaining.pop() {
                                Some(next) => {
                                    cont = Cont::cons(Frame::ListElems { remaining, done, env: env.clone() }, cont);
                                    control = Control::Eval(next, env);
                                }
                                None => control = Control::Apply(Value::List(Rc::new(done))),
                            }
                        }
                        // Same left-to-right accumulation as ListElems just
                        // above (and as Frame::LetRecBody's own fallback
                        // below, which has no `names` of its own to zip
                        // back in), differing only in the final wrap:
                        // `names` (fixed for the whole record) zipped
                        // together with `done` into a Value::Record instead
                        // of a bare Value::List.
                        Frame::RecordElems { names, remaining, done, env } => {
                            let (names, mut remaining, mut done, env) =
                                (names.clone(), remaining.clone(), done.clone(), env.clone());
                            done.push(value);
                            cont = rest;
                            match remaining.pop() {
                                Some(next) => {
                                    cont = Cont::cons(
                                        Frame::RecordElems { names, remaining, done, env: env.clone() },
                                        cont,
                                    );
                                    control = Control::Eval(next, env);
                                }
                                None => {
                                    let fields = names.iter().cloned().zip(done).collect();
                                    control = Control::Apply(Value::Record(Rc::new(fields)));
                                }
                            }
                        }
                        Frame::Check { spec } => {
                            let spec = spec.clone();
                            let holds = test_holds(&spec.test, &spec.defs, &value);
                            cont = rest;
                            control = Control::Apply(match spec.mode {
                                CheckMode::Probe => Value::Bool(holds),
                                CheckMode::Assert if holds => value,
                                CheckMode::Assert => {
                                    set_current_span(spec.span);
                                    let mut path = Vec::new();
                                    let found = explain(&spec.test, &spec.defs, &value, &mut path);
                                    let at = if path.is_empty() { String::new() } else { format!(" at {}", path_text(&path)) };
                                    crate::run_error!("type error: expected {}, found {}{at}", spec.to, found.type_name())
                                }
                            });
                        }
                        Frame::PerformPayload { effect } => {
                            let effect = effect.clone();
                            cont = rest;
                            control = Control::Perform(effect, value);
                        }
                        Frame::HandlerMark(_) => {
                            // Handled body finished normally (no pending
                            // effect reached this mark) -- identity return
                            // clause: just pass the value through. Nothing
                            // in the HandlerData is needed here, so nothing
                            // gets cloned.
                            cont = rest;
                            control = Control::Apply(value);
                        }
                    }
                }
            },

            Control::Perform(effect, payload) => {
                control = perform(&mut cont, &effect, payload);
            }
        }
    }
}

// ---- The deep check: ONE iterative walker ----
//
// `test_holds` (every Dyn-boundary Expr::Check, and the is_int/is_float/...
// prelude predicates) and `explain` (the failure message's "at element 1")
// are two runs of `walk`, so there is a single definition of the test and the
// native stack use is O(1) in the value's nesting depth (a recursive check
// overflowed the worker stack at about 150,000 levels in debug builds). The
// walker keeps a work list of (test, value) steps and, for `Or`, a stack of
// choice points: a failing alternative rewinds the work list to the height
// the `Or` started at and tries the next one. `defs` is the owning
// CheckSpec's alias table (what Test::Ref indexes; the prelude predicates
// pass `&[]`). The value is finite and acyclic, and every step consumes
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
        // (typecheck::wrap_fun_contract) re-checks each call separately.
        Test::Fun => {
            matches!(v, Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..))
        }
        Test::Token(id) => matches!(v, Value::Token(t) if t == id),
        Test::ListOf(_) | Test::ListLen(..) | Test::TupleOf(_) | Test::RecordOf(_) | Test::Ref(_) | Test::Or(_) => return None,
    })
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
    Mark((usize, usize)),
}

// A compound value's identity (its shared allocation), for the walk's memo of
// decided (alias, value) pairs. Scalars are as cheap to re-test as to look up.
fn identity(v: &Value) -> Option<usize> {
    match v {
        Value::List(items) => Some(Rc::as_ptr(items) as usize),
        Value::Record(fields) => Some(Rc::as_ptr(fields) as usize),
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

// None when `v` passes `test`, else the innermost failure (with TRACK, its
// path node and depth; an `Or` reports its alternative that got furthest,
// the first on ties).
fn walk<'t, 'a, const TRACK: bool>(test: &'t Test, defs: &'t [Test], v: &'a Value, nodes: &mut Nodes<'t>) -> Option<Fail<'a>> {
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
    let mut memo: HashMap<(usize, usize), bool> = HashMap::new();
    'main: loop {
        let mut fail: Fail<'a> = 'step: {
            let step = work.pop()?;
            let (t, v, node) = match step {
                Step::Commit => {
                    choices.pop();
                    continue 'main;
                }
                Step::Mark(key) => {
                    memo.insert(key, true);
                    continue 'main;
                }
                Step::FailAt(v, node) => break 'step Fail { node, depth: depth_of(nodes, node), found: v },
                Step::Test(t, v, node) => (t, v, node),
            };
            let here = Fail { node, depth: depth_of(nodes, node), found: v };
            if let Some(ok) = leaf(t, v) {
                if ok {
                    continue 'main;
                }
                break 'step here;
            }
            match t {
                Test::Ref(i) => {
                    if let Some(p) = identity(v) {
                        match memo.get(&(*i, p)) {
                            Some(true) => continue 'main,
                            Some(false) => break 'step here,
                            None => work.push(Step::Mark((*i, p))),
                        }
                    }
                    work.push(Step::Test(&defs[*i], v, node));
                }
                Test::ListOf(element) => {
                    let Value::List(items) = v else { break 'step here };
                    if leaf(element, v).is_some() {
                        // A scalar element test (`leaf` answers Some for those
                        // whatever the value): scan inline, nothing pushed.
                        match items.iter().position(|x| leaf(element, x) == Some(false)) {
                            None => continue 'main,
                            Some(i) => {
                                let n = child::<TRACK>(nodes, node, Label::Element(i));
                                break 'step Fail { node: n, depth: depth_of(nodes, n), found: &items[i] };
                            }
                        }
                    }
                    // Reversed, so the stack pops them in element order.
                    for (i, x) in items.iter().enumerate().rev() {
                        let n = child::<TRACK>(nodes, node, Label::Element(i));
                        work.push(Step::Test(element, x, n));
                    }
                }
                Test::ListLen(element, len) => {
                    let Value::List(items) = v else { break 'step here };
                    if items.len() != *len {
                        break 'step here;
                    }
                    for (i, x) in items.iter().enumerate().rev() {
                        let n = child::<TRACK>(nodes, node, Label::Element(i));
                        work.push(Step::Test(element, x, n));
                    }
                }
                Test::TupleOf(elements) => {
                    let Value::List(items) = v else { break 'step here };
                    if items.len() != elements.len() {
                        break 'step here;
                    }
                    for (i, (x, element)) in items.iter().zip(elements).enumerate().rev() {
                        let n = child::<TRACK>(nodes, node, Label::Element(i));
                        work.push(Step::Test(element, x, n));
                    }
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
                if let Step::Mark(key) = step {
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

fn test_holds(test: &Test, defs: &[Test], v: &Value) -> bool {
    match leaf(test, v) {
        Some(ok) => ok,
        // `Int | Float`, the check on every Dyn arithmetic operand: decided by
        // the value's tag, with no work list or choice point.
        None => match test {
            Test::Or(alternatives) if alternatives.iter().all(|a| leaf(a, v).is_some()) => alternatives.iter().any(|a| leaf(a, v) == Some(true)),
            _ => walk::<false>(test, defs, v, &mut Vec::new()).is_none(),
        },
    }
}

// The innermost value that makes `test` fail on `v`, with the way down to it
// pushed onto `path` ("element 1", "field b"). Cold path of a failing Assert
// check only: the tracked re-run of `walk`, linear in the value. A union
// cannot say which alternative was meant: it descends into the one that got
// furthest. `v` itself when nothing fails.
fn explain<'a>(test: &Test, defs: &[Test], v: &'a Value, path: &mut Vec<String>) -> &'a Value {
    let mut nodes = Vec::new();
    let Some(fail) = walk::<true>(test, defs, v, &mut nodes) else { return v };
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

fn set_current_span(span: Option<Span>) {
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
            (Value::Str(a), Value::Str(b)) => Value::Str(Rc::from(format!("{a}{b}"))),
            (Value::List(a), Value::List(b)) => {
                let mut v = Vec::with_capacity(a.len() + b.len());
                v.extend(a.iter().cloned());
                v.extend(b.iter().cloned());
                Value::List(Rc::new(v))
            }
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
            Value::List(items) => {
                let mut v = Vec::with_capacity(items.len() + 1);
                v.push(lhs);
                v.extend(items.iter().cloned());
                Value::List(Rc::new(v))
            }
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
fn collect_builtin_arg(arena: &Arena, b: Builtin, mut args: Vec<Value>, arg: Value, spans: &SpanMap, resolved: &Resolved) -> Value {
    args.push(arg);
    if args.len() == b.arity() {
        dispatch_builtin(arena, b, args, spans, resolved)
    } else {
        Value::PartialBuiltin(b, Rc::new(args))
    }
}

fn dispatch_builtin(arena: &Arena, b: Builtin, mut args: Vec<Value>, spans: &SpanMap, resolved: &Resolved) -> Value {
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
                    let idx = usize::try_from(idx).ok().filter(|&idx| idx < items.len());
                    match idx {
                        Some(idx) => items[idx].clone(),
                        None => crate::run_error!("get: index out of bounds"),
                    }
                }
                _ => crate::run_error!("get expects a list and an int"),
            }
        }
        Builtin::IsInt => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Int, &[], &v))),
        Builtin::IsFloat => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Float, &[], &v))),
        Builtin::IsBool => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Bool, &[], &v))),
        Builtin::IsStr => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Str, &[], &v))),
        Builtin::IsList => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::List, &[], &v))),
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
        Builtin::IsFun => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Fun, &[], &v))),
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
            Some(v) => Value::Str(Rc::from(v.to_string())),
            None => crate::run_error!("to_str expects one argument"),
        },
        Builtin::Map => {
            let (list, f) = (args.pop(), args.pop());
            match (f, list) {
                (Some(f), Some(Value::List(items))) => {
                    let mapped: Vec<Value> =
                        items.iter().map(|v| apply(arena, f.clone(), v.clone(), spans, resolved)).collect();
                    Value::List(Rc::new(mapped))
                }
                _ => crate::run_error!("map expects a function and a list"),
            }
        }
        Builtin::Fold => {
            let (list, init, f) = (args.pop(), args.pop(), args.pop());
            match (f, init, list) {
                (Some(f), Some(init), Some(Value::List(items))) => {
                    let mut acc = init;
                    for item in items.iter() {
                        // f is curried (one renno-level argument at a
                        // time): f(acc) yields a closure, applied to item.
                        let partial = apply(arena, f.clone(), acc, spans, resolved);
                        acc = apply(arena, partial, item.clone(), spans, resolved);
                    }
                    acc
                }
                _ => crate::run_error!("fold expects a function, an initial value, and a list"),
            }
        }
        Builtin::Filter => {
            let (list, pred) = (args.pop(), args.pop());
            match (pred, list) {
                (Some(pred), Some(Value::List(items))) => {
                    let filtered: Vec<Value> = items
                        .iter()
                        .filter(|v| apply(arena, pred.clone(), (*v).clone(), spans, resolved).as_bool())
                        .cloned()
                        .collect();
                    Value::List(Rc::new(filtered))
                }
                _ => crate::run_error!("filter expects a function and a list"),
            }
        }
        Builtin::Reverse => match args.pop() {
            Some(Value::List(items)) => {
                let mut items = (*items).clone();
                items.reverse();
                Value::List(Rc::new(items))
            }
            _ => crate::run_error!("reverse expects a list"),
        },
        Builtin::Sort => {
            let (list, cmp) = (args.pop(), args.pop());
            match (cmp, list) {
                (Some(cmp), Some(Value::List(items))) => {
                    let mut items = (*items).clone();
                    items.sort_by(|a, b| {
                        let a_first = apply(arena, apply(arena, cmp.clone(), a.clone(), spans, resolved), b.clone(), spans, resolved).as_bool();
                        if a_first {
                            std::cmp::Ordering::Less
                        } else if apply(arena, apply(arena, cmp.clone(), b.clone(), spans, resolved), a.clone(), spans, resolved).as_bool() {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Equal
                        }
                    });
                    Value::List(Rc::new(items))
                }
                _ => crate::run_error!("sort expects a comparator and a list"),
            }
        }
        Builtin::Range => {
            let (end, start) = (args.pop(), args.pop());
            match (start, end) {
                (Some(Value::Int(start)), Some(Value::Int(end))) => Value::List(Rc::new((start..end).map(Value::Int).collect())),
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
                    Value::Str(Rc::from(strs.join(&*sep)))
                }
                _ => crate::run_error!("join expects a list and a string"),
            }
        }
    }
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
                let tail_val = Value::List(Rc::new(items[1..].to_vec()));
                match_pattern(head, &items[0], out) && match_pattern(tail, &tail_val, out)
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

// Search outward through `cont` for a matching HandlerMark, capturing every
// frame passed along the way into `k`. `k` becomes the first-class
// `resume` value bound in the handler body -- a persistent Cont, so the
// handler can apply it zero, one, or many times. No arena needed here --
// HandlerData.body is an ExprRef, just handed to Control::Eval as-is; the
// next loop iteration in `run` is what looks it up.
fn perform(cont: &mut Cont, effect: &str, payload: Value) -> Control {
    let mut captured = Vec::new();
    let mut node = cont.clone();

    loop {
        match &*node.0 {
            ContNode::Nil => crate::run_error!("unhandled effect: {effect}"),
            ContNode::Frame(frame, rest) => {
                if let Frame::HandlerMark(data) = frame {
                    if data.effect == effect {
                        // deep: reinstall this same HandlerMark at the far
                        // end of k, exactly where it originally sat, so an
                        // effect performed while running the resumed
                        // continuation is caught by this handler again.
                        // shallow: k ends bare -- a repeat occurrence
                        // escapes to whatever handler sits further out.
                        let base = if data.deep { Cont::cons(frame.clone(), Cont::nil()) } else { Cont::nil() };
                        let k = Cont::from_frames(captured, base);
                        let handler_env = data.env.push2(payload, Value::Continuation(k));
                        *cont = rest.clone();
                        return Control::Eval(data.body, handler_env);
                    }
                }
                captured.push(frame.clone());
                node = rest.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(n: usize) -> Value {
        Value::List(Rc::new(vec![Value::Int(0); n]))
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
        let list = Value::List(Rc::new(vec![Value::Int(1)]));
        let values = [Value::Int(1), Value::Float(1.0), Value::Bool(true), list];
        let ors = [
            Test::Or(vec![]),
            Test::Or(vec![Test::Int, Test::Float]),
            Test::Or(vec![Test::Never, Test::Bool]),
            Test::Or(vec![Test::Int, Test::ListOf(Box::new(Test::Int))]),
        ];
        for t in &ors {
            for v in &values {
                assert_eq!(test_holds(t, &[], v), walk::<false>(t, &[], v, &mut Vec::new()).is_none());
            }
        }
    }

    #[test]
    fn test_holds_covers_every_shape() {
        let record = Value::Record(Rc::new(vec![("x".to_string(), Value::Int(1)), ("y".to_string(), Value::Int(2))]));
        let closure = Value::Builtin(Builtin::Len);
        let holds = |t: &Test, v: &Value| test_holds(t, &[], v);
        assert!(holds(&Test::Any, &Value::Bool(true)) && !holds(&Test::Never, &Value::Bool(true)));
        assert!(holds(&Test::Int, &Value::Int(1)) && !holds(&Test::Int, &Value::Float(1.0)));
        assert!(holds(&Test::Float, &Value::Float(1.0)) && !holds(&Test::Float, &Value::Int(1)));
        assert!(holds(&Test::Bool, &Value::Bool(false)) && !holds(&Test::Bool, &Value::Int(0)));
        assert!(holds(&Test::Str, &Value::Str(Rc::from("a"))) && !holds(&Test::Str, &Value::Int(0)));
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

    #[test]
    fn deep_tests_follow_the_value_and_explain_names_the_failing_element() {
        let ints = |xs: &[i64]| Value::List(Rc::new(xs.iter().map(|n| Value::Int(*n)).collect()));
        let list_of_int = Test::ListOf(Box::new(Test::Int));
        assert!(test_holds(&list_of_int, &[], &ints(&[1, 2])) && test_holds(&list_of_int, &[], &ints(&[])));
        let mixed = Value::List(Rc::new(vec![Value::Int(1), Value::Bool(true)]));
        assert!(!test_holds(&list_of_int, &[], &mixed));
        let mut path = Vec::new();
        assert_eq!(explain(&list_of_int, &[], &mixed, &mut path).type_name(), "Bool");
        assert_eq!(path, ["element 1"]);

        let record = Value::Record(Rc::new(vec![("a".to_string(), Value::Int(1)), ("b".to_string(), mixed.clone())]));
        let nested = Test::RecordOf(vec![("a".to_string(), Test::Int), ("b".to_string(), list_of_int.clone())]);
        assert!(!test_holds(&nested, &[], &record));
        let mut path = Vec::new();
        assert_eq!(explain(&nested, &[], &record, &mut path).type_name(), "Bool");
        assert_eq!(path, ["field b", "element 1"]);

        // type L = (Int, L) | Bool: the alias body is defs[0], reached through Ref(0).
        let defs = [Test::Or(vec![Test::TupleOf(vec![Test::Int, Test::Ref(0)]), Test::Bool])];
        let pair = |a: Value, b: Value| Value::List(Rc::new(vec![a, b]));
        let good = pair(Value::Int(1), pair(Value::Int(2), Value::Bool(true)));
        let bad = pair(Value::Int(1), pair(Value::Int(2), Value::Int(3)));
        assert!(test_holds(&Test::Ref(0), &defs, &good) && !test_holds(&Test::Ref(0), &defs, &bad));
        let mut path = Vec::new();
        assert_eq!(explain(&Test::Ref(0), &defs, &bad, &mut path).type_name(), "Int");
        assert_eq!(path, ["element 1", "element 1"]);
    }

    // The walker uses O(1) native stack: this runs on the 2 MiB default test
    // thread, where a recursive check (or a recursive Drop, hence the forgets)
    // overflows after a few thousand levels.
    #[test]
    fn the_walker_handles_a_very_deep_spine_on_a_small_stack() {
        const DEPTH: usize = 100_000;
        let pair = |a: Value, b: Value| Value::List(Rc::new(vec![a, b]));
        let defs = [Test::Or(vec![Test::TupleOf(vec![Test::Int, Test::Ref(0)]), Test::Bool])];
        let chain = |leaf: Value| (0..DEPTH).fold(leaf, |tail, i| pair(Value::Int(i as i64), tail));
        let good = chain(Value::Bool(true));
        assert!(test_holds(&Test::Ref(0), &defs, &good));
        let bad = chain(Value::Str(Rc::from("x")));
        assert!(!test_holds(&Test::Ref(0), &defs, &bad));
        let mut path = Vec::new();
        assert_eq!(explain(&Test::Ref(0), &defs, &bad, &mut path).type_name(), "Str");
        assert_eq!(path.len(), DEPTH);
        // nested single-element lists: [[[...[Int]...]]] against ListOf(Ref) where the alias is [self] | Int
        let list_defs = [Test::Or(vec![Test::Int, Test::ListOf(Box::new(Test::Ref(0)))])];
        let nest = (0..DEPTH).fold(Value::Int(1), |inner, _| Value::List(Rc::new(vec![inner])));
        assert!(test_holds(&Test::Ref(0), &list_defs, &nest));
        // the nested chains cannot be dropped recursively on this stack
        std::mem::forget((good, bad, nest));
    }

    // The old recursive definitions, kept as the oracle for the walker: same
    // answer, same path, same offending value on random (test, defs, value).
    fn holds_recursive(test: &Test, defs: &[Test], v: &Value) -> bool {
        match test {
            Test::ListOf(t) => matches!(v, Value::List(items) if items.iter().all(|x| holds_recursive(t, defs, x))),
            Test::ListLen(t, n) => matches!(v, Value::List(items) if items.len() == *n && items.iter().all(|x| holds_recursive(t, defs, x))),
            Test::TupleOf(ts) => matches!(v, Value::List(items) if items.len() == ts.len() && items.iter().zip(ts).all(|(x, t)| holds_recursive(t, defs, x))),
            Test::RecordOf(fs) => matches!(v, Value::Record(fields) if fs.iter().all(|(n, t)| find_field(fields, n).is_some_and(|x| holds_recursive(t, defs, x)))),
            Test::Ref(i) => holds_recursive(&defs[*i], defs, v),
            Test::Or(alternatives) => alternatives.iter().any(|t| holds_recursive(t, defs, v)),
            scalar => leaf(scalar, v).expect("scalar"),
        }
    }

    fn explain_recursive<'a>(test: &Test, defs: &[Test], v: &'a Value, path: &mut Vec<String>) -> &'a Value {
        match (test, v) {
            (Test::Ref(i), _) => explain_recursive(&defs[*i], defs, v, path),
            (Test::ListLen(_, n), Value::List(items)) if items.len() != *n => v,
            (Test::ListOf(t), Value::List(items)) | (Test::ListLen(t, _), Value::List(items)) => match items.iter().position(|x| !holds_recursive(t, defs, x)) {
                Some(i) => {
                    path.push(format!("element {i}"));
                    explain_recursive(t, defs, &items[i], path)
                }
                None => v,
            },
            (Test::TupleOf(ts), Value::List(items)) if items.len() == ts.len() => {
                match items.iter().zip(ts).position(|(x, t)| !holds_recursive(t, defs, x)) {
                    Some(i) => {
                        path.push(format!("element {i}"));
                        explain_recursive(&ts[i], defs, &items[i], path)
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
            4 if refs && r.below(2) == 0 => Test::Ref(r.below(2)),
            4 => Test::Float,
            5 => Test::ListOf(Box::new(random_test(r, depth - 1, true))),
            6 => Test::ListLen(Box::new(random_test(r, depth - 1, true)), r.below(4)),
            7 => Test::TupleOf((0..r.below(4)).map(|_| random_test(r, depth - 1, true)).collect()),
            8 => Test::RecordOf((0..r.below(4)).map(|i| (NAMES[i].to_string(), random_test(r, depth - 1, true))).collect()),
            _ => Test::Or((0..r.below(4)).map(|_| random_test(r, depth - 1, refs)).collect()),
        }
    }

    fn random_value(r: &mut Rng, depth: u32) -> Value {
        match if depth == 0 { r.below(4) } else { r.below(8) } {
            0 => Value::Int(1),
            1 => Value::Bool(true),
            2 => Value::Str(Rc::from("s")),
            3 => Value::Float(1.0),
            4 | 5 => Value::List(Rc::new((0..r.below(4)).map(|_| random_value(r, depth - 1)).collect())),
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

    #[test]
    fn the_walker_agrees_with_the_recursive_definition_on_random_cases() {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let (mut failing, total) = (0, 30_000);
        for _ in 0..total {
            let defs = [random_test(&mut r, 3, false), random_test(&mut r, 3, false)];
            let test = random_test(&mut r, 4, false);
            let v = random_value(&mut r, 4);
            let want = holds_recursive(&test, &defs, &v);
            assert_eq!(want, test_holds(&test, &defs, &v), "holds differs: {test:?} {defs:?} {v}");
            if !want {
                failing += 1;
                let (mut want_path, mut got_path) = (Vec::new(), Vec::new());
                let want_found = explain_recursive(&test, &defs, &v, &mut want_path);
                let got_found = explain(&test, &defs, &v, &mut got_path);
                assert_eq!(want_path, got_path, "path differs: {test:?} {defs:?} {v}");
                assert!(std::ptr::eq(want_found, got_found), "found differs: {test:?} {defs:?} {v}");
            }
        }
        assert!(failing > total / 10 && failing < total, "the generator should produce both outcomes: {failing}/{total} failing");
    }
}
