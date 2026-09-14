use std::rc::Rc;

use crate::cont::{Cont, ContNode, Frame};
use crate::env::Env;
use crate::expr::{BinOp, Expr};
use crate::value::{Builtin, HandlerData, Value};

enum Control {
    Eval(Rc<Expr>, Env),
    Apply(Value),
    Perform(String, Value),
}

// Trampolined CEK-style step loop -- no native recursion, so no stack
// overflow risk from deep programs or from resuming captured continuations.
pub fn run(expr: Rc<Expr>, env: Env) -> Value {
    let mut control = Control::Eval(expr, env);
    let mut cont = Cont::nil();

    loop {
        match control {
            Control::Eval(expr, env) => match &*expr {
                Expr::Int(n) => control = Control::Apply(Value::Int(*n)),
                Expr::Bool(b) => control = Control::Apply(Value::Bool(*b)),
                Expr::Var(name) => control = Control::Apply(env.lookup(name)),
                Expr::Lambda(param, body) => {
                    control = Control::Apply(Value::Closure(param.clone(), body.clone(), env));
                }
                Expr::App(f, a) => {
                    cont = Cont::cons(Frame::AppFunc { arg: a.clone(), env: env.clone() }, cont);
                    control = Control::Eval(f.clone(), env);
                }
                Expr::Let(var, val_expr, body) => {
                    cont = Cont::cons(
                        Frame::LetBody { var: var.clone(), body: body.clone(), env: env.clone() },
                        cont,
                    );
                    control = Control::Eval(val_expr.clone(), env);
                }
                Expr::BinOp(op, l, r) => {
                    cont = Cont::cons(Frame::BinOpL { op: *op, rhs: r.clone(), env: env.clone() }, cont);
                    control = Control::Eval(l.clone(), env);
                }
                Expr::If(c, t, e) => {
                    cont = Cont::cons(
                        Frame::If { then_: t.clone(), else_: e.clone(), env: env.clone() },
                        cont,
                    );
                    control = Control::Eval(c.clone(), env);
                }
                Expr::Perform(effect, payload) => {
                    cont = Cont::cons(Frame::PerformPayload { effect: effect.clone() }, cont);
                    control = Control::Eval(payload.clone(), env);
                }
                Expr::Handle { body, handler } => {
                    cont = Cont::cons(
                        Frame::InstallHandler { body: body.clone(), env: env.clone() },
                        cont,
                    );
                    control = Control::Eval(handler.clone(), env);
                }
                Expr::MakeHandler { effect, payload_var, resume_var, body } => {
                    control = Control::Apply(Value::Handler(Rc::new(HandlerData {
                        effect: effect.clone(),
                        payload_var: payload_var.clone(),
                        resume_var: resume_var.clone(),
                        body: body.clone(),
                        env,
                        deep: false,
                    })));
                }
            },

            Control::Apply(value) => match &*cont.0 {
                ContNode::Nil => return value,
                ContNode::Frame(frame, rest) => {
                    let rest = rest.clone();
                    match frame.clone() {
                        Frame::AppFunc { arg, env } => {
                            cont = Cont::cons(Frame::AppArg { func: value }, rest);
                            control = Control::Eval(arg, env);
                        }
                        Frame::AppArg { func } => {
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
                                        _ => panic!("deep/shallow expect a handler value"),
                                    });
                                }
                                _ => panic!("attempt to call a non-function value"),
                            }
                        }
                        Frame::InstallHandler { body, env } => {
                            cont = rest;
                            match value {
                                Value::Handler(data) => {
                                    cont = Cont::cons(
                                        Frame::HandlerMark {
                                            effect: data.effect.clone(),
                                            payload_var: data.payload_var.clone(),
                                            resume_var: data.resume_var.clone(),
                                            handler_body: data.body.clone(),
                                            env: data.env.clone(),
                                            deep: data.deep,
                                        },
                                        cont,
                                    );
                                    control = Control::Eval(body, env);
                                }
                                _ => panic!("handle: expected a handler value"),
                            }
                        }
                        Frame::LetBody { var, body, env } => {
                            cont = rest;
                            control = Control::Eval(body, env.bind(var, value));
                        }
                        Frame::BinOpL { op, rhs, env } => {
                            cont = Cont::cons(Frame::BinOpR { op, lhs: value }, rest);
                            control = Control::Eval(rhs, env);
                        }
                        Frame::BinOpR { op, lhs } => {
                            cont = rest;
                            control = Control::Apply(apply_binop(op, lhs, value));
                        }
                        Frame::If { then_, else_, env } => {
                            cont = rest;
                            control = Control::Eval(
                                if value.as_bool() { then_ } else { else_ },
                                env,
                            );
                        }
                        Frame::PerformPayload { effect } => {
                            cont = rest;
                            control = Control::Perform(effect, value);
                        }
                        Frame::HandlerMark { .. } => {
                            // Handled body finished normally (no pending
                            // effect reached this mark) -- identity return
                            // clause: just pass the value through.
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

// Int-only for now; comparisons producing Bool. No mixed-type coercion --
// wrong operand type panics via as_int(), matching the rest of the runtime.
fn apply_binop(op: BinOp, lhs: Value, rhs: Value) -> Value {
    match op {
        BinOp::Add => Value::Int(lhs.as_int() + rhs.as_int()),
        BinOp::Eq => Value::Bool(lhs.as_int() == rhs.as_int()),
        BinOp::Lt => Value::Bool(lhs.as_int() < rhs.as_int()),
    }
}

// Search outward through `cont` for a matching HandlerMark, capturing every
// frame passed along the way into `k`. `k` becomes the first-class
// `resume` value bound in the handler body -- a persistent Cont, so the
// handler can apply it zero, one, or many times.
fn perform(cont: &mut Cont, effect: &str, payload: Value) -> Control {
    let mut captured = Vec::new();
    let mut node = cont.clone();

    loop {
        match &*node.0 {
            ContNode::Nil => panic!("unhandled effect: {effect}"),
            ContNode::Frame(frame, rest) => {
                if let Frame::HandlerMark { effect: e, payload_var, resume_var, handler_body, env, deep } = frame {
                    if e == effect {
                        // deep: reinstall this same HandlerMark at the far
                        // end of k, exactly where it originally sat, so an
                        // effect performed while running the resumed
                        // continuation is caught by this handler again.
                        // shallow: k ends bare -- a repeat occurrence
                        // escapes to whatever handler sits further out.
                        let mut k = if *deep {
                            Cont::cons(frame.clone(), Cont::nil())
                        } else {
                            Cont::nil()
                        };
                        for f in captured.into_iter().rev() {
                            k = Cont::cons(f, k);
                        }
                        let handler_env = env
                            .bind(payload_var.clone(), payload)
                            .bind(resume_var.clone(), Value::Continuation(k));
                        *cont = rest.clone();
                        return Control::Eval(handler_body.clone(), handler_env);
                    }
                }
                captured.push(frame.clone());
                node = rest.clone();
            }
        }
    }
}
