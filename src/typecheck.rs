use std::rc::Rc;

use crate::expr::{Arena, BinOp, Expr, ExprRef};
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
fn coerce(arena: &mut Arena, e: ExprRef, from: &Type, to: &Type) -> Result<ExprRef, TypeError> {
    if !consistent(from, to) {
        return Err(TypeError(format!("type mismatch: expected {to}, found {from}")));
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
// new Value representation needed.
fn wrap_fun_contract(arena: &mut Arena, e: ExprRef, param_ty: Rc<Type>, ret_ty: Rc<Type>) -> ExprRef {
    let fn_var = "__contract_fn".to_string();
    let arg_var = "__contract_arg".to_string();
    let fn_var_ref = arena.push(Expr::Var(fn_var.clone()));
    let arg_var_ref = arena.push(Expr::Var(arg_var.clone()));
    let checked_fn = arena.push(Expr::Check(any_fun(), fn_var_ref));
    let call = arena.push(Expr::App(checked_fn, arg_var_ref));
    let checked_call = arena.push(Expr::Check((*ret_ty).clone(), call));
    let lambda = arena.push(Expr::Lambda(arg_var, Some((*param_ty).clone()), checked_call));
    arena.push(Expr::Let(fn_var, None, e, lambda))
}

// A `let`/`fun` prefix collected while flattening a chain of them (see
// `elaborate`) -- deferred until the terminal body is elaborated, then
// folded back into nested Let/Lambda nodes (and their types/rows) in
// reverse, in the exact shape their original per-node match arms produced.
enum PendingElab {
    Let { var: String, bound_ty: Type, val_row: EffectRow, val: ExprRef },
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
pub fn elaborate(arena: &mut Arena, expr: ExprRef, ctx: &Ctx) -> Result<(Type, EffectRow, ExprRef), TypeError> {
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
                let (val_ty, val_row, val2) = elaborate(arena, val, &cur_ctx)?;
                let (bound_ty, val3) = match ann {
                    Some(t) => (t.clone(), coerce(arena, val2, &val_ty, &t)?),
                    None => (val_ty, val2),
                };
                cur_ctx = extend(&cur_ctx, &var, bound_ty.clone());
                pending.push(PendingElab::Let { var, bound_ty, val_row, val: val3 });
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

    let (mut result_ty, mut result_row, mut result_expr) = elaborate_node(arena, cur_expr, &cur_ctx)?;

    for frame in pending.into_iter().rev() {
        match frame {
            PendingElab::Let { var, bound_ty, val_row, val } => {
                // Matches the original Let arm: body's type propagates
                // through unchanged, row is the union of val's and body's.
                result_row = EffectRow::union(&val_row, &result_row);
                result_expr = arena.push(Expr::Let(var, Some(bound_ty), val, result_expr));
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
fn elaborate_node(arena: &mut Arena, expr: ExprRef, ctx: &Ctx) -> Result<(Type, EffectRow, ExprRef), TypeError> {
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
                let (item_ty, item_row, item2) = elaborate(arena, item, ctx)?;
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

        Expr::Let(..) | Expr::Lambda(..) => {
            unreachable!("Let/Lambda are peeled by elaborate's chain-flattening loop")
        }

        Expr::App(f, a) => {
            let (f_ty, f_row, f2) = elaborate(arena, f, ctx)?;
            let (a_ty, a_row, a2) = elaborate(arena, a, ctx)?;
            let (call_row, ret_ty, app2) = match &f_ty {
                Type::Fun(param_ty, call_row, ret_ty) => {
                    let a3 = coerce(arena, a2, &a_ty, param_ty)?;
                    (call_row.clone(), (**ret_ty).clone(), arena.push(Expr::App(f2, a3)))
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
                other => return Err(TypeError(format!("cannot call a value of type {other}"))),
            };
            let row = EffectRow::union(&EffectRow::union(&f_row, &a_row), &call_row);
            Ok((ret_ty, row, app2))
        }

        Expr::BinOp(op, l, r) => {
            let (l_ty, l_row, l2) = elaborate(arena, l, ctx)?;
            let (r_ty, r_row, r2) = elaborate(arena, r, ctx)?;
            let row = EffectRow::union(&l_row, &r_row);
            match op {
                // Arithmetic and ordering: both operands must be Int.
                BinOp::Add | BinOp::Lt => {
                    let l3 = coerce(arena, l2, &l_ty, &Type::Int)?;
                    let r3 = coerce(arena, r2, &r_ty, &Type::Int)?;
                    let result_ty = if op == BinOp::Add { Type::Int } else { Type::Bool };
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
                        return Err(TypeError(format!(
                            "type mismatch: cannot compare {l_ty} with {r_ty}"
                        )));
                    }
                    let l3 = if l_ty == Type::Dyn && r_ty != Type::Dyn {
                        coerce(arena, l2, &l_ty, &r_ty)?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty)?
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
                        return Err(TypeError(format!(
                            "type mismatch: cannot concat {l_ty} with {r_ty}"
                        )));
                    }
                    let result_ty = match (&l_ty, &r_ty) {
                        (Type::Dyn, Type::Dyn) => Type::Dyn,
                        (Type::Dyn, t) | (t, Type::Dyn) => t.clone(),
                        (Type::Str, Type::Str) => Type::Str,
                        (Type::List(_), Type::List(_)) => l_ty.clone(),
                        _ => {
                            return Err(TypeError(format!(
                                "type mismatch: cannot concat {l_ty} with {r_ty} (expected two strings or two lists)"
                            )))
                        }
                    };
                    let l3 = if l_ty == Type::Dyn && r_ty != Type::Dyn {
                        coerce(arena, l2, &l_ty, &r_ty)?
                    } else {
                        l2
                    };
                    let r3 = if r_ty == Type::Dyn && l_ty != Type::Dyn {
                        coerce(arena, r2, &r_ty, &l_ty)?
                    } else {
                        r2
                    };
                    Ok((result_ty, row, arena.push(Expr::BinOp(op, l3, r3))))
                }
            }
        }

        Expr::If(c, t, e) => {
            let (c_ty, c_row, c2) = elaborate(arena, c, ctx)?;
            let c3 = coerce(arena, c2, &c_ty, &Type::Bool)?;
            let (t_ty, t_row, t2) = elaborate(arena, t, ctx)?;
            let (e_ty, e_row, e2) = elaborate(arena, e, ctx)?;
            // Branches with differing concrete types aren't an error here
            // (no union types) -- just widen to Dyn rather than reject.
            let result_ty = if t_ty == e_ty { t_ty } else { Type::Dyn };
            // Only one branch runs, but which one isn't known statically,
            // so the possible effects are the union of both.
            let row = EffectRow::union(&c_row, &EffectRow::union(&t_row, &e_row));
            Ok((result_ty, row, arena.push(Expr::If(c3, t2, e2))))
        }

        Expr::Check(ty, inner) => {
            let (_, row, inner2) = elaborate(arena, inner, ctx)?;
            let ty_ret = ty.clone();
            Ok((ty_ret, row, arena.push(Expr::Check(ty, inner2))))
        }

        // The effect this specific operation performs, plus whatever the
        // payload expression itself might perform.
        Expr::Perform(effect, payload) => {
            let (_, payload_row, payload2) = elaborate(arena, payload, ctx)?;
            let row = EffectRow::union(&payload_row, &EffectRow::single(&effect));
            Ok((Type::Dyn, row, arena.push(Expr::Perform(effect, payload2))))
        }

        Expr::Handle { body, handler } => {
            let (_, body_row, body2) = elaborate(arena, body, ctx)?;
            let (handler_ty, handler_row, handler2) = elaborate(arena, handler, ctx)?;
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
            let row = match discharged_effect(arena, handler) {
                Some(effect) => body_row.remove(&effect),
                None => body_row,
            };
            let row = EffectRow::union(&row, &handler_row);
            Ok((Type::Dyn, row, arena.push(Expr::Handle { body: body2, handler: handler2 })))
        }

        // Constructing the handler value is pure -- the clause body's own
        // effects (including what `resume` re-enters) aren't modeled here;
        // see the doc comment on `elaborate`.
        Expr::MakeHandler { effect, payload_var, resume_var, body } => {
            let inner_ctx = extend(&extend(ctx, &payload_var, Type::Dyn), &resume_var, Type::Dyn);
            let (_, _, body2) = elaborate(arena, body, &inner_ctx)?;
            Ok((
                Type::Dyn,
                EffectRow::pure(),
                arena.push(Expr::MakeHandler { effect, payload_var, resume_var, body: body2 }),
            ))
        }
    }
}

pub fn check(arena: &mut Arena, root: ExprRef) -> Result<ExprRef, TypeError> {
    let (_, row, elaborated) = elaborate(arena, root, &Ctx::empty())?;
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
