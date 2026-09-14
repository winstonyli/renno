use std::rc::Rc;

use crate::cont::{Cont, ContNode, Frame};
use crate::env::Env;
use crate::expr::{Arena, BinOp, Expr, ExprRef};
use crate::value::{Builtin, HandlerData, Value};

enum Control {
    Eval(ExprRef, Env),
    Apply(Value),
    Perform(String, Value),
}

// Trampolined CEK-style step loop -- no native recursion, so no stack
// overflow risk from deep programs or from resuming captured continuations.
// `arena` is only ever read here (all mutation happens during parsing and
// typecheck::elaborate); ExprRef fields are Copy, so threading node
// references through Control/Frame costs nothing beyond a plain integer
// copy -- no Rc bump, unlike when this held Rc<Expr>.
pub fn run(arena: &Arena, expr: ExprRef, env: Env) -> Value {
    let mut control = Control::Eval(expr, env);
    let mut cont = Cont::nil();

    loop {
        match control {
            Control::Eval(expr, env) => match &arena[expr] {
                Expr::Int(n) => control = Control::Apply(Value::Int(*n)),
                Expr::Bool(b) => control = Control::Apply(Value::Bool(*b)),
                Expr::Str(s) => control = Control::Apply(Value::Str(Rc::from(s.as_str()))),
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
                Expr::Var(name) => control = Control::Apply(env.lookup(name)),
                Expr::Lambda(param, _ann, body) => {
                    control = Control::Apply(Value::Closure(param.clone(), *body, env));
                }
                Expr::App(f, a) => {
                    cont = Cont::cons(Frame::AppFunc { arg: *a, env: env.clone() }, cont);
                    control = Control::Eval(*f, env);
                }
                Expr::Let(var, _ann, val_expr, body) => {
                    cont = Cont::cons(
                        Frame::LetBody { var: var.clone(), body: *body, env: env.clone() },
                        cont,
                    );
                    control = Control::Eval(*val_expr, env);
                }
                Expr::BinOp(op, l, r) => {
                    cont = Cont::cons(Frame::BinOpL { op: *op, rhs: *r, env: env.clone() }, cont);
                    control = Control::Eval(*l, env);
                }
                Expr::If(c, t, e) => {
                    cont = Cont::cons(Frame::If { then_: *t, else_: *e, env: env.clone() }, cont);
                    control = Control::Eval(*c, env);
                }
                Expr::Check(ty, inner) => {
                    cont = Cont::cons(Frame::CheckFrame { ty: ty.clone() }, cont);
                    control = Control::Eval(*inner, env);
                }
                Expr::Perform(effect, payload) => {
                    cont = Cont::cons(Frame::PerformPayload { effect: effect.clone() }, cont);
                    control = Control::Eval(*payload, env);
                }
                Expr::Handle { body, handler } => {
                    cont = Cont::cons(Frame::InstallHandler { body: *body, env: env.clone() }, cont);
                    control = Control::Eval(*handler, env);
                }
                Expr::MakeHandler { effect, payload_var, resume_var, body } => {
                    control = Control::Apply(Value::Handler(Rc::new(HandlerData {
                        effect: effect.clone(),
                        payload_var: payload_var.clone(),
                        resume_var: resume_var.clone(),
                        body: *body,
                        env,
                        deep: false,
                    })));
                }
            },

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
                        Frame::AppFunc { arg, env } => {
                            let (arg, env) = (*arg, env.clone());
                            cont = Cont::cons(Frame::AppArg { func: value }, rest);
                            control = Control::Eval(arg, env);
                        }
                        Frame::AppArg { func } => {
                            let func = func.clone();
                            cont = rest;
                            match func {
                                Value::Closure(param, body, closure_env) => {
                                    control = Control::Eval(body, closure_env.bind(param, value));
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
                                    // deep/shallow: clone the handler data,
                                    // flip the `deep` bit, hand back a new
                                    // handler value. No AST-level flag.
                                    control = Control::Apply(match (b, value) {
                                        (Builtin::Deep, Value::Handler(data)) => {
                                            Value::Handler(Rc::new(HandlerData { deep: true, ..(*data).clone() }))
                                        }
                                        (Builtin::Shallow, Value::Handler(data)) => {
                                            Value::Handler(Rc::new(HandlerData { deep: false, ..(*data).clone() }))
                                        }
                                        (Builtin::Len, Value::Str(s)) => Value::Int(s.chars().count() as i64),
                                        (Builtin::Len, Value::List(items)) => Value::Int(items.len() as i64),
                                        _ => panic!("invalid builtin application"),
                                    });
                                }
                                _ => panic!("attempt to call a non-function value"),
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
                        Frame::LetBody { var, body, env } => {
                            let (var, body, env) = (var.clone(), *body, env.clone());
                            cont = rest;
                            control = Control::Eval(body, env.bind(var, value));
                        }
                        Frame::BinOpL { op, rhs, env } => {
                            let (op, rhs, env) = (*op, *rhs, env.clone());
                            cont = Cont::cons(Frame::BinOpR { op, lhs: value }, rest);
                            control = Control::Eval(rhs, env);
                        }
                        Frame::BinOpR { op, lhs } => {
                            let (op, lhs) = (*op, lhs.clone());
                            cont = rest;
                            control = Control::Apply(apply_binop(op, lhs, value));
                        }
                        Frame::If { then_, else_, env } => {
                            let (then_, else_, env) = (*then_, *else_, env.clone());
                            cont = rest;
                            control = Control::Eval(if value.as_bool() { then_ } else { else_ }, env);
                        }
                        Frame::CheckFrame { ty } => {
                            let ty = ty.clone();
                            cont = rest;
                            if value.matches_type(&ty) {
                                control = Control::Apply(value);
                            } else {
                                panic!("type error: expected {ty}, found {}", value.type_name());
                            }
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

// Add/Lt are Int-only -- wrong operand type panics via as_int(). Eq is
// structural: it compares whatever tags the two values actually carry
// (typecheck.rs only requires the two operand types to be consistent with
// each other, not both Int), so it dispatches on Value directly instead of
// projecting through as_int(). Concat (++) is Str/Str or List/List only --
// typecheck.rs rejects anything else statically when it can tell, so a
// runtime panic here only fires for a Dyn-sourced value of the wrong tag.
fn apply_binop(op: BinOp, lhs: Value, rhs: Value) -> Value {
    match op {
        BinOp::Add => Value::Int(lhs.as_int() + rhs.as_int()),
        BinOp::Lt => Value::Bool(lhs.as_int() < rhs.as_int()),
        BinOp::Eq => Value::Bool(value_eq(&lhs, &rhs)),
        BinOp::Concat => match (lhs, rhs) {
            (Value::Str(a), Value::Str(b)) => Value::Str(Rc::from(format!("{a}{b}"))),
            (Value::List(a), Value::List(b)) => {
                let mut v = Vec::with_capacity(a.len() + b.len());
                v.extend(a.iter().cloned());
                v.extend(b.iter().cloned());
                Value::List(Rc::new(v))
            }
            _ => panic!("++ expects two strings or two lists"),
        },
    }
}

fn value_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Str(x), Value::Str(y)) => x == y,
        (Value::List(x), Value::List(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(a, b)| value_eq(a, b))
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
                        let handler_env = data
                            .env
                            .bind(data.payload_var.clone(), payload)
                            .bind(data.resume_var.clone(), Value::Continuation(k));
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
