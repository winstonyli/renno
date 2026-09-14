use std::rc::Rc;

use crate::expr::{BinOp, Expr};
use crate::types::{consistent, Type};

#[derive(Debug)]
pub struct TypeError(pub String);

type Ctx = Vec<(String, Type)>;

fn lookup(ctx: &Ctx, name: &str) -> Type {
    for (n, t) in ctx.iter().rev() {
        if n == name {
            return t.clone();
        }
    }
    // Unbound at typecheck time: don't error here, machine::run's own
    // `unbound variable` panic at runtime is the right place for that.
    Type::Dyn
}

fn extend(ctx: &Ctx, name: &str, ty: Type) -> Ctx {
    let mut c = ctx.clone();
    c.push((name.to_string(), ty));
    c
}

// The only place a runtime Check gets inserted: `from` is Dyn (unknown
// statically) and `to` is concrete. If both sides are concrete and
// disagree, that's a real static error -- reject before running at all.
// If `from` is already exactly consistent and concrete, no check needed:
// zero overhead for fully-annotated code.
fn coerce(e: Rc<Expr>, from: &Type, to: &Type) -> Result<Rc<Expr>, TypeError> {
    if !consistent(from, to) {
        return Err(TypeError(format!("type mismatch: expected {to}, found {from}")));
    }
    if *from == Type::Dyn && *to != Type::Dyn {
        Ok(Rc::new(Expr::Check(to.clone(), e)))
    } else {
        Ok(e)
    }
}

// Bidirectional-lite synthesis: walks the tree once, producing both the
// inferred Type and an elaborated Expr (same shape, with Check nodes
// spliced in at Dyn-to-concrete boundaries). Effects stay untyped (Dyn) --
// that's a separate, later phase (effect typing), not this one.
pub fn elaborate(expr: &Expr, ctx: &Ctx) -> Result<(Type, Rc<Expr>), TypeError> {
    match expr {
        Expr::Int(n) => Ok((Type::Int, Rc::new(Expr::Int(*n)))),
        Expr::Bool(b) => Ok((Type::Bool, Rc::new(Expr::Bool(*b)))),
        Expr::Var(name) => Ok((lookup(ctx, name), Rc::new(Expr::Var(name.clone())))),

        Expr::Lambda(param, ann, body) => {
            let param_ty = ann.clone().unwrap_or(Type::Dyn);
            let (body_ty, body2) = elaborate(body, &extend(ctx, param, param_ty.clone()))?;
            Ok((
                Type::Fun(Rc::new(param_ty.clone()), Rc::new(body_ty)),
                Rc::new(Expr::Lambda(param.clone(), Some(param_ty), body2)),
            ))
        }

        Expr::App(f, a) => {
            let (f_ty, f2) = elaborate(f, ctx)?;
            let (a_ty, a2) = elaborate(a, ctx)?;
            match f_ty {
                Type::Fun(param_ty, ret_ty) => {
                    let a3 = coerce(a2, &a_ty, &param_ty)?;
                    Ok(((*ret_ty).clone(), Rc::new(Expr::App(f2, a3))))
                }
                Type::Dyn => Ok((Type::Dyn, Rc::new(Expr::App(f2, a2)))),
                other => Err(TypeError(format!("cannot call a value of type {other}"))),
            }
        }

        Expr::Let(var, ann, val, body) => {
            let (val_ty, val2) = elaborate(val, ctx)?;
            let (bound_ty, val3) = match ann {
                Some(t) => (t.clone(), coerce(val2, &val_ty, t)?),
                None => (val_ty, val2),
            };
            let (body_ty, body2) = elaborate(body, &extend(ctx, var, bound_ty.clone()))?;
            Ok((body_ty, Rc::new(Expr::Let(var.clone(), Some(bound_ty), val3, body2))))
        }

        Expr::BinOp(op, l, r) => {
            let (l_ty, l2) = elaborate(l, ctx)?;
            let (r_ty, r2) = elaborate(r, ctx)?;
            let l3 = coerce(l2, &l_ty, &Type::Int)?;
            let r3 = coerce(r2, &r_ty, &Type::Int)?;
            let result_ty = match op {
                BinOp::Add => Type::Int,
                BinOp::Eq | BinOp::Lt => Type::Bool,
            };
            Ok((result_ty, Rc::new(Expr::BinOp(*op, l3, r3))))
        }

        Expr::If(c, t, e) => {
            let (c_ty, c2) = elaborate(c, ctx)?;
            let c3 = coerce(c2, &c_ty, &Type::Bool)?;
            let (t_ty, t2) = elaborate(t, ctx)?;
            let (e_ty, e2) = elaborate(e, ctx)?;
            // Branches with differing concrete types aren't an error here
            // (no union types) -- just widen to Dyn rather than reject.
            let result_ty = if t_ty == e_ty { t_ty } else { Type::Dyn };
            Ok((result_ty, Rc::new(Expr::If(c3, t2, e2))))
        }

        Expr::Check(ty, inner) => {
            let (_, inner2) = elaborate(inner, ctx)?;
            Ok((ty.clone(), Rc::new(Expr::Check(ty.clone(), inner2))))
        }

        // Effects are untyped for now (later phase: effect typing).
        Expr::Perform(effect, payload) => {
            let (_, payload2) = elaborate(payload, ctx)?;
            Ok((Type::Dyn, Rc::new(Expr::Perform(effect.clone(), payload2))))
        }
        Expr::Handle { body, handler } => {
            let (_, body2) = elaborate(body, ctx)?;
            let (_, handler2) = elaborate(handler, ctx)?;
            Ok((Type::Dyn, Rc::new(Expr::Handle { body: body2, handler: handler2 })))
        }
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let inner_ctx = extend(&extend(ctx, payload_var, Type::Dyn), resume_var, Type::Dyn);
            let (_, body2) = elaborate(body, &inner_ctx)?;
            Ok((
                Type::Dyn,
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
    elaborate(expr, &Vec::new()).map(|(_, e)| e)
}
