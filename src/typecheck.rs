use std::rc::Rc;

use crate::expr::{BinOp, Expr};
use crate::plist::PList;
use crate::types::{consistent, EffectRow, Type};

#[derive(Debug)]
pub struct TypeError(pub String);

type Ctx = PList<Type>;

fn lookup(ctx: &Ctx, name: &str) -> Type {
    // Unbound at typecheck time: don't error here, machine::run's own
    // `unbound variable` panic at runtime is the right place for that.
    ctx.get(name).unwrap_or(Type::Dyn)
}

fn extend(ctx: &Ctx, name: &str, ty: Type) -> Ctx {
    ctx.bind(name, ty)
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
// zero overhead for fully-annotated code.
//
// Crossing into a Fun type is special: value::matches_type only confirms
// "this is callable," not "callable with this exact signature" (a tag
// check can't see inside a closure). So a Dyn value flowing into an
// annotated Fun position gets wrapped in a real per-call contract instead
// of a bare tag Check -- see wrap_fun_contract.
fn coerce(e: Rc<Expr>, from: &Type, to: &Type) -> Result<Rc<Expr>, TypeError> {
    if !consistent(from, to) {
        return Err(TypeError(format!("type mismatch: expected {to}, found {from}")));
    }
    if *from != Type::Dyn || *to == Type::Dyn {
        return Ok(e);
    }
    match to {
        Type::Fun(param_ty, _row, ret_ty) => Ok(wrap_fun_contract(e, param_ty.clone(), ret_ty.clone())),
        _ => Ok(Rc::new(Expr::Check(to.clone(), e))),
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
// new Value representation needed.
fn wrap_fun_contract(e: Rc<Expr>, param_ty: Rc<Type>, ret_ty: Rc<Type>) -> Rc<Expr> {
    let fn_var = "__contract_fn".to_string();
    let arg_var = "__contract_arg".to_string();
    Rc::new(Expr::Let(
        fn_var.clone(),
        None,
        e,
        Rc::new(Expr::Lambda(
            arg_var.clone(),
            Some((*param_ty).clone()),
            Rc::new(Expr::Check(
                (*ret_ty).clone(),
                Rc::new(Expr::App(
                    Rc::new(Expr::Check(any_fun(), Rc::new(Expr::Var(fn_var)))),
                    Rc::new(Expr::Var(arg_var)),
                )),
            )),
        )),
    ))
}

// Recognizes the handler-expression shapes this codebase actually
// produces (MakeHandler, optionally wrapped in deep(...)/shallow(...))
// well enough to know which effect name a `handle` discharges. Anything
// else (a bare variable, a computed handler) returns None -- Handle then
// conservatively does NOT subtract anything from the body's row, which is
// the sound direction to fail in: at worst it over-reports an effect as
// possibly-unhandled, never hides a real one.
fn discharged_effect(handler: &Expr) -> Option<&str> {
    match handler {
        Expr::MakeHandler { effect, .. } => Some(effect),
        Expr::App(f, arg) => match &**f {
            Expr::Var(name) if name == "deep" || name == "shallow" => discharged_effect(arg),
            _ => None,
        },
        _ => None,
    }
}

// Bidirectional-lite synthesis: walks the tree once, producing the
// inferred Type, the inferred EffectRow (closed, no polymorphism -- just
// the union of effect names this expression's evaluation might perform),
// and an elaborated Expr (same shape, with Check nodes -- or, at Fun
// boundaries, full contracts -- spliced in at Dyn-to-concrete crossings).
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
pub fn elaborate(expr: &Expr, ctx: &Ctx) -> Result<(Type, EffectRow, Rc<Expr>), TypeError> {
    match expr {
        Expr::Int(n) => Ok((Type::Int, EffectRow::pure(), Rc::new(Expr::Int(*n)))),
        Expr::Bool(b) => Ok((Type::Bool, EffectRow::pure(), Rc::new(Expr::Bool(*b)))),
        Expr::Var(name) => Ok((lookup(ctx, name), EffectRow::pure(), Rc::new(Expr::Var(name.clone())))),

        Expr::Lambda(param, ann, body) => {
            let param_ty = ann.clone().unwrap_or(Type::Dyn);
            let (body_ty, body_row, body2) = elaborate(body, &extend(ctx, param, param_ty.clone()))?;
            // Evaluating the Lambda expression itself is pure -- the
            // body's row only manifests when the resulting function is
            // called, so it's embedded in the Fun type, not returned here.
            Ok((
                Type::Fun(Rc::new(param_ty.clone()), body_row, Rc::new(body_ty)),
                EffectRow::pure(),
                Rc::new(Expr::Lambda(param.clone(), Some(param_ty), body2)),
            ))
        }

        Expr::App(f, a) => {
            let (f_ty, f_row, f2) = elaborate(f, ctx)?;
            let (a_ty, a_row, a2) = elaborate(a, ctx)?;
            let called_row_and_ty_expr = match &f_ty {
                Type::Fun(param_ty, call_row, ret_ty) => {
                    let a3 = coerce(a2, &a_ty, param_ty)?;
                    (call_row.clone(), (**ret_ty).clone(), Rc::new(Expr::App(f2, a3)))
                }
                Type::Dyn => {
                    // Unknown callee: still route "is this even callable"
                    // through the same Check mechanism everything else
                    // uses, rather than leaving it to a differently-worded
                    // panic in machine.rs. Can't know what it might
                    // perform, so the call contributes an unknown (Dyn) row.
                    let f3 = Rc::new(Expr::Check(any_fun(), f2));
                    (EffectRow::Dyn, Type::Dyn, Rc::new(Expr::App(f3, a2)))
                }
                other => return Err(TypeError(format!("cannot call a value of type {other}"))),
            };
            let (call_row, ret_ty, app2) = called_row_and_ty_expr;
            let row = EffectRow::union(&EffectRow::union(&f_row, &a_row), &call_row);
            Ok((ret_ty, row, app2))
        }

        Expr::Let(var, ann, val, body) => {
            let (val_ty, val_row, val2) = elaborate(val, ctx)?;
            let (bound_ty, val3) = match ann {
                Some(t) => (t.clone(), coerce(val2, &val_ty, t)?),
                None => (val_ty, val2),
            };
            let (body_ty, body_row, body2) = elaborate(body, &extend(ctx, var, bound_ty.clone()))?;
            let row = EffectRow::union(&val_row, &body_row);
            Ok((body_ty, row, Rc::new(Expr::Let(var.clone(), Some(bound_ty), val3, body2))))
        }

        Expr::BinOp(op, l, r) => {
            let (l_ty, l_row, l2) = elaborate(l, ctx)?;
            let (r_ty, r_row, r2) = elaborate(r, ctx)?;
            let row = EffectRow::union(&l_row, &r_row);
            match op {
                // Arithmetic and ordering: both operands must be Int.
                BinOp::Add | BinOp::Lt => {
                    let l3 = coerce(l2, &l_ty, &Type::Int)?;
                    let r3 = coerce(r2, &r_ty, &Type::Int)?;
                    let result_ty = if *op == BinOp::Add { Type::Int } else { Type::Bool };
                    Ok((result_ty, row, Rc::new(Expr::BinOp(*op, l3, r3))))
                }
                // Equality: operands just need to be consistent with EACH
                // OTHER, not both forced to Int -- `true == false` is a
                // real comparison. If one side is Dyn and the other
                // concrete, coerce the Dyn side to the concrete side's
                // type so the runtime value at least has a known tag;
                // apply_binop compares by matching Value variants.
                BinOp::Eq => {
                    if !consistent(&l_ty, &r_ty) {
                        return Err(TypeError(format!(
                            "type mismatch: cannot compare {l_ty} with {r_ty}"
                        )));
                    }
                    let l3 = if l_ty == Type::Dyn && r_ty != Type::Dyn {
                        coerce(l2, &l_ty, &r_ty)?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(r2, &r_ty, &l_ty)?
                    } else {
                        r2
                    };
                    Ok((Type::Bool, row, Rc::new(Expr::BinOp(*op, l3, r3))))
                }
            }
        }

        Expr::If(c, t, e) => {
            let (c_ty, c_row, c2) = elaborate(c, ctx)?;
            let c3 = coerce(c2, &c_ty, &Type::Bool)?;
            let (t_ty, t_row, t2) = elaborate(t, ctx)?;
            let (e_ty, e_row, e2) = elaborate(e, ctx)?;
            // Branches with differing concrete types aren't an error here
            // (no union types) -- just widen to Dyn rather than reject.
            let result_ty = if t_ty == e_ty { t_ty } else { Type::Dyn };
            // Only one branch runs, but which one isn't known statically,
            // so the possible effects are the union of both.
            let row = EffectRow::union(&c_row, &EffectRow::union(&t_row, &e_row));
            Ok((result_ty, row, Rc::new(Expr::If(c3, t2, e2))))
        }

        Expr::Check(ty, inner) => {
            let (_, row, inner2) = elaborate(inner, ctx)?;
            Ok((ty.clone(), row, Rc::new(Expr::Check(ty.clone(), inner2))))
        }

        // The effect this specific operation performs, plus whatever the
        // payload expression itself might perform.
        Expr::Perform(effect, payload) => {
            let (_, payload_row, payload2) = elaborate(payload, ctx)?;
            let row = EffectRow::union(&payload_row, &EffectRow::single(effect));
            Ok((Type::Dyn, row, Rc::new(Expr::Perform(effect.clone(), payload2))))
        }

        Expr::Handle { body, handler } => {
            let (_, body_row, body2) = elaborate(body, ctx)?;
            let (handler_ty, handler_row, handler2) = elaborate(handler, ctx)?;
            // types.rs has no Type::Handler -- but every handler-producing
            // expression (MakeHandler, or deep(...)/shallow(...) applied
            // to one, or a var bound from either) synthesizes Dyn by this
            // same convention, so a concretely-typed handler expression
            // (Int, Bool, Fun) can never legitimately be one. Reject it
            // statically instead of letting it reach machine.rs's panic.
            if handler_ty != Type::Dyn {
                return Err(TypeError(format!(
                    "handle: expected a handler value, found expression of type {handler_ty}"
                )));
            }
            // Discharge the effect this handler catches, if we can
            // statically tell which one that is. If not, conservatively
            // leave body_row untouched (over-approximate, never hide).
            let row = match discharged_effect(handler) {
                Some(effect) => body_row.remove(effect),
                None => body_row,
            };
            let row = EffectRow::union(&row, &handler_row);
            Ok((Type::Dyn, row, Rc::new(Expr::Handle { body: body2, handler: handler2 })))
        }

        // Constructing the handler value is pure -- the clause body's own
        // effects (including what `resume` re-enters) aren't modeled here;
        // see the doc comment on `elaborate`.
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let inner_ctx = extend(&extend(ctx, payload_var, Type::Dyn), resume_var, Type::Dyn);
            let (_, _, body2) = elaborate(body, &inner_ctx)?;
            Ok((
                Type::Dyn,
                EffectRow::pure(),
                Rc::new(Expr::MakeHandler {
                    effect: effect.clone(),
                    payload_var: payload_var.clone(),
                    resume_var: resume_var.clone(),
                    body: body2,
                }),
            ))
        }
    }
}

pub fn check(expr: &Rc<Expr>) -> Result<Rc<Expr>, TypeError> {
    let (_, row, elaborated) = elaborate(expr, &Ctx::empty())?;
    match row {
        EffectRow::Closed(unhandled) if !unhandled.is_empty() => {
            let names: Vec<_> = unhandled.into_iter().collect();
            Err(TypeError(format!(
                "unhandled effect{}: {}",
                if names.len() > 1 { "s" } else { "" },
                names.join(", ")
            )))
        }
        _ => Ok(elaborated),
    }
}
