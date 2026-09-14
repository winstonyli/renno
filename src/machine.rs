use std::cell::Cell;
use std::rc::Rc;

use crate::cont::{Cont, ContNode, Frame};
use crate::env::Env;
use crate::expr::{Arena, BinOp, Expr, ExprRef, Pattern, SpanMap};
use crate::span::Span;
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
    // matter which function actually called panic! -- env.rs's,
    // value.rs's, and machine.rs's own panics all get this for free, with
    // no Span parameter threaded into any of them.
    //
    // Not perfectly precise: a panic several steps removed from the last
    // Eval (deep inside apply_binop or Value::as_int, say) reports the
    // last EXPRESSION evaluated, not necessarily the exact sub-part at
    // fault -- e.g. `1 + true` reports `true`'s own span (the last thing
    // evaluated before apply_binop's as_int() panics on it), not the
    // whole `1 + true`. Always in the neighborhood, never wildly off,
    // without threading a Span through every Frame variant and Value
    // accessor -- the same 80/20 trade this checker already makes
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
    run_loop(arena, Control::Eval(expr, env), Cont::nil(), spans)
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
pub fn apply(arena: &Arena, func: Value, arg: Value, spans: &SpanMap) -> Value {
    run_loop(arena, Control::Apply(arg), Cont::cons(Frame::AppArg { func }, Cont::nil()), spans)
}

fn run_loop(arena: &Arena, mut control: Control, mut cont: Cont, spans: &SpanMap) -> Value {
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
                    cont = Cont::cons(Frame::LetBody { var: var.clone(), body: *body, env: env.clone() }, cont);
                    control = Control::Eval(*val_expr, env);
                }
                Expr::LetRec(bindings, body) => {
                    let names: Rc<Vec<String>> = Rc::new(bindings.iter().map(|(n, _, _)| n.clone()).collect());
                    // Reversed, same reason as ListLit: pop() (O(1))
                    // instead of remove(0) as each binding finishes.
                    let mut remaining: Vec<ExprRef> = bindings.iter().map(|(_, _, v)| *v).collect();
                    remaining.reverse();
                    let first = remaining.pop().unwrap(); // parser never emits an empty group
                    cont = Cont::cons(
                        Frame::LetRecBody { names, remaining, done: Vec::new(), body: *body, env: env.clone() },
                        cont,
                    );
                    control = Control::Eval(first, env);
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
                Expr::Match(scrutinee, arms) => {
                    cont = Cont::cons(Frame::MatchArms { arms: arms.clone(), env: env.clone() }, cont);
                    control = Control::Eval(*scrutinee, env);
                }
                // Pure compile-time marker for typecheck (see Expr's doc
                // comment on this variant) -- a typechecked program never
                // has one (elaborate unwraps it), but the untyped path
                // (e.g. tests' run_untyped) runs the parser's raw output
                // directly, so this still needs to evaluate straight
                // through to `body`.
                Expr::DataGroup(_, body) => control = Control::Eval(*body, env),
                // Unlike DataGroup, a typechecked program never leaves one
                // of these behind EITHER (typecheck::elaborate_node
                // resolves it into an ordinary Match) -- but resolving it
                // needs static type information (which `data` type does
                // the target belong to, and which field index does `name`
                // mean) that has nowhere to come from at runtime, so
                // there's no sensible fallback the way DataGroup's plain
                // passthrough is. The untyped path (e.g. tests'
                // run_untyped) just can't use field access.
                Expr::FieldAccess(_, field) => {
                    panic!("field access `.{field}` requires typechecking (run through typecheck::check, not raw parser output)")
                }
                // Same story as FieldAccess, for the same reason: needs
                // the constructor's declared field order, which only
                // exists statically.
                Expr::NamedCall(..) => {
                    panic!("named-field construction requires typechecking (run through typecheck::check, not raw parser output)")
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
                                Value::RecClosure(group, index, closure_env) => {
                                    // Rebind EVERY name in the group to (a
                                    // fresh copy of) its own RecClosure
                                    // every call, not just once at
                                    // construction -- that's what makes a
                                    // reference to any sibling (including
                                    // this one's own name) inside `body`
                                    // resolve recursively, with Env's
                                    // ordinary persistent bind/lookup doing
                                    // all the work. No mutation, no AST
                                    // rewriting. A plain single-function
                                    // `let rec` is just the group.len()==1
                                    // case -- same loop, one iteration.
                                    let (_, param, body) = group[index].clone();
                                    let mut env2 = closure_env.clone();
                                    for i in 0..group.len() {
                                        let (name, _, _) = group[i].clone();
                                        env2 = env2.bind(name, Value::RecClosure(group.clone(), i, closure_env.clone()));
                                    }
                                    env2 = env2.bind(param, value);
                                    control = Control::Eval(body, env2);
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
                                    control = Control::Apply(collect_builtin_arg(arena, b, Vec::new(), value, spans));
                                }
                                Value::PartialBuiltin(b, prev_args) => {
                                    let args = (*prev_args).clone();
                                    control = Control::Apply(collect_builtin_arg(arena, b, args, value, spans));
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
                        Frame::LetRecBody { names, remaining, done, body, env } => {
                            let (names, mut remaining, mut done, body, env) =
                                (names.clone(), remaining.clone(), done.clone(), *body, env.clone());
                            done.push(value);
                            cont = rest;
                            match remaining.pop() {
                                Some(next) => {
                                    cont = Cont::cons(
                                        Frame::LetRecBody { names, remaining, done, body, env: env.clone() },
                                        cont,
                                    );
                                    control = Control::Eval(next, env);
                                }
                                None => {
                                    // Every binding's value is in (`done`,
                                    // same order as `names`). If they're
                                    // ALL functions, wrap the whole group
                                    // into mutually-referencing RecClosure
                                    // values (see its own doc comment). A
                                    // group with any non-function value is
                                    // meaningless as `let rec` (same as the
                                    // single-binding case) but not an
                                    // error -- just bind everything as-is,
                                    // no recursion magic.
                                    let all_closures = done.iter().all(|v| matches!(v, Value::Closure(..)));
                                    // Every RecClosure's own stored env is
                                    // this ORIGINAL one -- fixed, never the
                                    // progressively-growing `env2` below --
                                    // so rebuilding the group on a call
                                    // always starts from the same base,
                                    // not one that already has some
                                    // members bound from THIS construction.
                                    let base_env = env;
                                    let mut env2 = base_env.clone();
                                    if all_closures {
                                        let group: Rc<Vec<(String, String, ExprRef)>> = Rc::new(
                                            names
                                                .iter()
                                                .zip(done.iter())
                                                .map(|(name, v)| match v {
                                                    Value::Closure(param, cbody, _) => {
                                                        (name.clone(), param.clone(), *cbody)
                                                    }
                                                    _ => unreachable!("all_closures already checked"),
                                                })
                                                .collect(),
                                        );
                                        for (i, name) in names.iter().enumerate() {
                                            env2 =
                                                env2.bind(name.clone(), Value::RecClosure(group.clone(), i, base_env.clone()));
                                        }
                                    } else {
                                        for (name, v) in names.iter().zip(done) {
                                            env2 = env2.bind(name.clone(), v);
                                        }
                                    }
                                    control = Control::Eval(body, env2);
                                }
                            }
                        }
                        Frame::MatchArms { arms, env } => {
                            let (arms, env) = (arms.clone(), env.clone());
                            cont = rest;
                            match first_match(&arms, &value, &env) {
                                Some((body, matched_env)) => control = Control::Eval(body, matched_env),
                                None => panic!("match failed: no pattern matched the value"),
                            }
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
        BinOp::Sub => Value::Int(lhs.as_int() - rhs.as_int()),
        BinOp::Mul => Value::Int(lhs.as_int() * rhs.as_int()),
        BinOp::Div => {
            let (l, r) = (lhs.as_int(), rhs.as_int());
            if r == 0 {
                panic!("division by zero");
            }
            Value::Int(l / r)
        }
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
        BinOp::Cons => match rhs {
            Value::List(items) => {
                let mut v = Vec::with_capacity(items.len() + 1);
                v.push(lhs);
                v.extend(items.iter().cloned());
                Value::List(Rc::new(v))
            }
            _ => panic!(":: expects a list on the right"),
        },
    }
}

// Appends one argument to a builtin's collected-so-far list, dispatching
// the real operation once `b`'s declared arity is reached, otherwise
// handing back a PartialBuiltin waiting for the rest.
fn collect_builtin_arg(arena: &Arena, b: Builtin, mut args: Vec<Value>, arg: Value, spans: &SpanMap) -> Value {
    args.push(arg);
    if args.len() == b.arity() {
        dispatch_builtin(arena, b, args, spans)
    } else {
        Value::PartialBuiltin(b, Rc::new(args))
    }
}

fn dispatch_builtin(arena: &Arena, b: Builtin, mut args: Vec<Value>, spans: &SpanMap) -> Value {
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
        Builtin::Map => {
            let (list, f) = (args.pop(), args.pop());
            match (f, list) {
                (Some(f), Some(Value::List(items))) => {
                    let mapped: Vec<Value> =
                        items.iter().map(|v| apply(arena, f.clone(), v.clone(), spans)).collect();
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
                        let partial = apply(arena, f.clone(), acc, spans);
                        acc = apply(arena, partial, item.clone(), spans);
                    }
                    acc
                }
                _ => panic!("fold expects a function, an initial value, and a list"),
            }
        }
    }
}

// Tries `arms` in order, returning the first one whose pattern matches
// `value`, together with `env` extended by whatever that pattern bound.
fn first_match(arms: &[(Pattern, ExprRef)], value: &Value, env: &Env) -> Option<(ExprRef, Env)> {
    arms.iter().find_map(|(pat, body)| match_pattern(pat, value, env.clone()).map(|env2| (*body, env2)))
}

// Native recursion here is bounded by the PATTERN's own size (as written
// in source), not by the data it's matched against -- a Cons/List pattern
// can only nest as deep as the program text does, so this can't overflow
// the way recursing over arbitrary runtime data would.
fn match_pattern(pat: &Pattern, value: &Value, env: Env) -> Option<Env> {
    match pat {
        Pattern::Var(name) => Some(env.bind(name.clone(), value.clone())),
        Pattern::Int(n) => matches!(value, Value::Int(v) if v == n).then_some(env),
        Pattern::Bool(b) => matches!(value, Value::Bool(v) if v == b).then_some(env),
        Pattern::Str(s) => matches!(value, Value::Str(v) if &**v == s.as_str()).then_some(env),
        Pattern::List(pats) => match value {
            Value::List(items) if items.len() == pats.len() => {
                let mut env = env;
                for (p, v) in pats.iter().zip(items.iter()) {
                    env = match_pattern(p, v, env)?;
                }
                Some(env)
            }
            _ => None,
        },
        Pattern::Cons(head, tail) => match value {
            Value::List(items) if !items.is_empty() => {
                let tail_val = Value::List(Rc::new(items[1..].to_vec()));
                let env = match_pattern(head, &items[0], env)?;
                match_pattern(tail, &tail_val, env)
            }
            _ => None,
        },
        // typecheck::resolve_pattern always rewrites this into Pattern::List
        // before a Match reaches an elaborated program; only the untyped
        // path (e.g. tests' run_untyped) could ever hand one to
        // match_pattern directly, and it has no field-order information to
        // match against anyway (see Expr::NamedCall's own doc comment).
        Pattern::NamedCtor(tag, _) => {
            panic!("named-field pattern `{tag} {{ ... }}` requires typechecking (run through typecheck::check, not raw parser output)")
        }
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
