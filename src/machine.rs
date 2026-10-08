use std::cell::Cell;
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
                        Some(VarRef::Unbound) => panic!("unbound variable: {name}"),
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
                                    control = Control::Apply(collect_builtin_arg(arena, b, Vec::new(), value, spans, resolved));
                                }
                                Value::PartialBuiltin(b, prev_args) => {
                                    let args = (*prev_args).clone();
                                    control = Control::Apply(collect_builtin_arg(arena, b, args, value, spans, resolved));
                                }
                                _ => {
                                    set_current_span(callee_span);
                                    panic!("attempt to call a non-function value")
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
                                _ => panic!("handle: expected a handler value"),
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
                            let holds = test_holds(&spec.test, &value);
                            cont = rest;
                            control = Control::Apply(match spec.mode {
                                CheckMode::Probe => Value::Bool(holds),
                                CheckMode::Assert if holds => value,
                                CheckMode::Assert => {
                                    set_current_span(spec.span);
                                    panic!("type error: expected {}, found {}", spec.to, value.type_name())
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

// The one definition of every shallow Dyn-boundary test (Expr::Check) and of
// the is_int/is_float/... prelude predicates, which share it.
fn test_holds(test: &Test, v: &Value) -> bool {
    match test {
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
        Test::Tuple(arity) => matches!(v, Value::List(items) if items.len() == *arity),
        Test::Record(names) => matches!(v, Value::Record(fields) if names.iter().all(|n| find_field(fields, n).is_some())),
        Test::Or(alternatives) => alternatives.iter().any(|t| test_holds(t, v)),
    }
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
            panic!("expected a number")
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
                    panic!("division by zero");
                }
                Value::Int(a / b)
            }
            (a, b) => {
                let b = b.as_f64();
                if b == 0.0 {
                    set_current_span(r_span);
                    panic!("division by zero");
                }
                Value::Float(a.as_f64() / b)
            }
        },
        BinOp::Mod => match (as_num_at(&lhs, l_span), as_num_at(&rhs, r_span)) {
            (Num::Int(a), Num::Int(b)) => {
                if b == 0 {
                    set_current_span(r_span);
                    panic!("modulo by zero");
                }
                Value::Int(a % b)
            }
            (a, b) => {
                let b = b.as_f64();
                if b == 0.0 {
                    set_current_span(r_span);
                    panic!("modulo by zero");
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
                panic!("++ expects two strings or two lists")
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
                panic!(":: expects a list on the right")
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
            _ => panic!("deep/shallow expect a handler value"),
        },
        Builtin::Len => match args.pop() {
            Some(Value::Str(s)) => Value::Int(s.chars().count() as i64),
            Some(Value::List(items)) => Value::Int(items.len() as i64),
            _ => panic!("len expects a string or list"),
        },
        Builtin::Fail => match args.pop() {
            Some(Value::Str(s)) => panic!("{s}"),
            _ => panic!("fail expects a string message"),
        },
        Builtin::Get => {
            let (i, list) = (args.pop(), args.pop());
            match (list, i) {
                (Some(Value::List(items)), Some(Value::Int(idx))) => {
                    let idx = usize::try_from(idx).ok().filter(|&idx| idx < items.len());
                    match idx {
                        Some(idx) => items[idx].clone(),
                        None => panic!("get: index out of bounds"),
                    }
                }
                _ => panic!("get expects a list and an int"),
            }
        }
        Builtin::IsInt => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Int, &v))),
        Builtin::IsFloat => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Float, &v))),
        Builtin::IsBool => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Bool, &v))),
        Builtin::IsStr => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Str, &v))),
        Builtin::IsList => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::List, &v))),
        Builtin::IsRecord => Value::Bool(matches!(args.pop(), Some(Value::Record(_)))),
        // (record, name) -- args.pop() order matches Get's own (list, i)
        // convention: last-pushed arg (name) pops first.
        Builtin::HasField => {
            let (name, record) = (args.pop(), args.pop());
            match (record, name) {
                (Some(Value::Record(fields)), Some(Value::Str(name))) => {
                    Value::Bool(find_field(&fields, &name).is_some())
                }
                _ => panic!("has_field expects a record and a string"),
            }
        }
        // Same (record, name) argument order as HasField just above.
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
                    None => panic!("no field named `{name}`"),
                },
                _ => panic!("get_field expects a record and a string"),
            }
        }
        Builtin::IsFun => Value::Bool(args.pop().is_some_and(|v| test_holds(&Test::Fun, &v))),
        Builtin::TypeName => match args.pop() {
            Some(v) => Value::Str(Rc::from(v.type_name())),
            None => panic!("type_name expects one argument"),
        },
        // Echoes its argument back unchanged after printing it -- see
        // Builtin::Print's own doc comment.
        Builtin::Print => match args.pop() {
            Some(v) => {
                println!("{v}");
                v
            }
            None => panic!("print expects one argument"),
        },
        Builtin::ToStr => match args.pop() {
            Some(v) => Value::Str(Rc::from(v.to_string())),
            None => panic!("to_str expects one argument"),
        },
        Builtin::Map => {
            let (list, f) = (args.pop(), args.pop());
            match (f, list) {
                (Some(f), Some(Value::List(items))) => {
                    let mapped: Vec<Value> =
                        items.iter().map(|v| apply(arena, f.clone(), v.clone(), spans, resolved)).collect();
                    Value::List(Rc::new(mapped))
                }
                _ => panic!("map expects a function and a list"),
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
                _ => panic!("fold expects a function, an initial value, and a list"),
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
                _ => panic!("filter expects a function and a list"),
            }
        }
        Builtin::Reverse => match args.pop() {
            Some(Value::List(items)) => {
                let mut items = (*items).clone();
                items.reverse();
                Value::List(Rc::new(items))
            }
            _ => panic!("reverse expects a list"),
        },
        Builtin::Zip => {
            let (ys, xs) = (args.pop(), args.pop());
            match (xs, ys) {
                (Some(Value::List(xs)), Some(Value::List(ys))) => {
                    let zipped: Vec<Value> = xs
                        .iter()
                        .zip(ys.iter())
                        .map(|(a, b)| Value::List(Rc::new(vec![a.clone(), b.clone()])))
                        .collect();
                    Value::List(Rc::new(zipped))
                }
                _ => panic!("zip expects two lists"),
            }
        }
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
                _ => panic!("sort expects a comparator and a list"),
            }
        }
        Builtin::Range => {
            let (end, start) = (args.pop(), args.pop());
            match (start, end) {
                (Some(Value::Int(start)), Some(Value::Int(end))) => Value::List(Rc::new((start..end).map(Value::Int).collect())),
                _ => panic!("range expects two ints"),
            }
        }
        Builtin::Split => {
            let (sep, s) = (args.pop(), args.pop());
            match (s, sep) {
                (Some(Value::Str(s)), Some(Value::Str(sep))) => {
                    let parts: Vec<Value> = s.split(&*sep).map(|p| Value::Str(Rc::from(p))).collect();
                    Value::List(Rc::new(parts))
                }
                _ => panic!("split expects two strings"),
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
                            _ => panic!("join expects a list of strings"),
                        })
                        .collect();
                    Value::Str(Rc::from(strs.join(&*sep)))
                }
                _ => panic!("join expects a list and a string"),
            }
        }
        Builtin::Trim => match args.pop() {
            Some(Value::Str(s)) => Value::Str(Rc::from(s.trim())),
            _ => panic!("trim expects a string"),
        },
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
        None => panic!("match failed: no pattern matched the value"),
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
            ContNode::Nil => panic!("unhandled effect: {effect}"),
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
    fn test_holds_covers_every_shape() {
        let record = Value::Record(Rc::new(vec![("x".to_string(), Value::Int(1)), ("y".to_string(), Value::Int(2))]));
        let closure = Value::Builtin(Builtin::Len);
        assert!(test_holds(&Test::Any, &Value::Bool(true)) && !test_holds(&Test::Never, &Value::Bool(true)));
        assert!(test_holds(&Test::Int, &Value::Int(1)) && !test_holds(&Test::Int, &Value::Float(1.0)));
        assert!(test_holds(&Test::Float, &Value::Float(1.0)) && !test_holds(&Test::Float, &Value::Int(1)));
        assert!(test_holds(&Test::Bool, &Value::Bool(false)) && !test_holds(&Test::Bool, &Value::Int(0)));
        assert!(test_holds(&Test::Str, &Value::Str(Rc::from("a"))) && !test_holds(&Test::Str, &Value::Int(0)));
        assert!(test_holds(&Test::List, &list(0)) && !test_holds(&Test::List, &record));
        assert!(test_holds(&Test::Fun, &closure) && !test_holds(&Test::Fun, &Value::Int(0)));
        assert!(test_holds(&Test::Token(7), &Value::Token(7)) && !test_holds(&Test::Token(7), &Value::Token(8)));
        assert!(!test_holds(&Test::Token(7), &Value::Int(7)));
        assert!(test_holds(&Test::Tuple(2), &list(2)) && !test_holds(&Test::Tuple(2), &list(3)) && !test_holds(&Test::Tuple(2), &record));
        let fields = |names: &[&str]| Test::Record(names.iter().map(|n| n.to_string()).collect());
        assert!(test_holds(&fields(&["x"]), &record) && test_holds(&fields(&["x", "y"]), &record));
        assert!(!test_holds(&fields(&["z"]), &record) && !test_holds(&fields(&["x"]), &list(1)));
        let int_or_token = Test::Or(vec![Test::Int, Test::Token(7)]);
        assert!(test_holds(&int_or_token, &Value::Token(7)) && test_holds(&int_or_token, &Value::Int(1)) && !test_holds(&int_or_token, &Value::Bool(true)));
        assert!(!test_holds(&Test::Or(vec![]), &Value::Int(1)));
    }
}
