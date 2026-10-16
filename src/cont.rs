use std::rc::Rc;

use crate::env::Env;
use crate::expr::{AnnotSpec, BinOp, CheckSpec, ExprRef, Lazy, Len, Pattern};
use crate::iterate::Iter;
use crate::span::Span;
use crate::value::{HandlerData, SiteCtx, Value};

// One step of "what's left to do", defunctionalized so it can live as data
// (not the native Rust call stack). This is what makes multi-shot resume
// possible: a captured Cont is an immutable, cheaply-clonable value.
#[derive(Clone)]
pub enum Frame {
    // `callee_span` is the CALLEE sub-expression's own span (`f` in `f(a)`),
    // not the whole App -- Some when it came from an ordinary Eval of an
    // App expr; None for AppArg frames the Iterate driver builds for a
    // higher-order builtin's callback (map, filter, fold, sort) -- there's no
    // App expr behind that, so CURRENT_SPAN is just left alone, same fallback
    // as before this field existed. Carried from AppFunc into AppArg so a
    // "not a function" panic blames `f` itself, not whatever the
    // argument's own span happened to be (the last thing Eval'd before the
    // panic fires, evaluated strictly after `f` and unrelated to why `f`
    // isn't callable).
    AppFunc { arg: ExprRef, env: Env, callee_span: Option<Span> },
    AppArg { func: Value, callee_span: Option<Span> },
    LetBody { body: ExprRef, env: Env },
    // Scrutinee has just been evaluated to `value` -- try `arms` in order.
    MatchArms { arms: Rc<Vec<(Pattern, Option<ExprRef>, ExprRef)>>, env: Env },
    // A pattern matched (binding `guard_env`) but its arm has a guard,
    // which has just been evaluated -- true takes `arms[idx]`'s body under
    // `guard_env`, false resumes the search from `idx + 1` against the
    // ORIGINAL scrutinee `value` and pre-match `outer_env` (guard_env's
    // pattern bindings must NOT leak into a sibling arm's own match
    // attempt).
    MatchGuard { arms: Rc<Vec<(Pattern, Option<ExprRef>, ExprRef)>>, idx: usize, outer_env: Env, guard_env: Env, value: Value },
    // `l_span`/`r_span` are the two operands' own spans, carried from
    // BinOpL into BinOpR and into apply_binop, which picks whichever one
    // actually explains a given panic (see apply_binop's own doc comment):
    // the operand that's actually the wrong type when that's knowable, or
    // a combination of both when neither operand alone accounts for the
    // failure (e.g. `++` with both sides Dyn and wrong-typed). This is
    // deliberately NOT always `r_span` (the last thing Eval'd before
    // apply_binop runs) -- that fallback would wrongly blame a perfectly
    // fine right operand whenever the LEFT one is actually at fault
    // (`true + 1`, say).
    BinOpL { op: BinOp, rhs: ExprRef, env: Env, l_span: Option<Span>, r_span: Option<Span> },
    BinOpR { op: BinOp, lhs: Value, l_span: Option<Span>, r_span: Option<Span> },
    If { then_: ExprRef, else_: ExprRef, env: Env },
    // The operand of an Expr::Conform whose type needs nothing beyond its
    // native test (`Then::Done`) has just evaluated -- test it against the
    // spec, with the length slots read when the Conform was entered.
    Check { spec: Rc<CheckSpec>, lens: Vec<Len> },
    // A value that must conform to a type needing a contract (a function, or a
    // union with a function alternative) has just evaluated, at a site, or as
    // a contract call's result: machine::conform checks it and wraps what it
    // still needs.
    Conform { lazy: Rc<Lazy>, ctx: Rc<SiteCtx> },
    // A kept annotation reached inside a check-mode evaluation (a type
    // function's body) has just evaluated: machine::annotation_value takes it.
    Annot { spec: Rc<AnnotSpec> },
    PerformPayload { effect: String },
    // `handle body with handler_expr`: handler_expr has just evaluated to a
    // Value::Handler -- next step installs it as a HandlerMark and evals body.
    InstallHandler { body: ExprRef, env: Env },
    // Wraps the same HandlerData a Value::Handler carries -- an Rc clone
    // (one pointer bump) instead of six separately-cloned fields, and one
    // definition instead of two structurally-identical ones to keep in sync.
    HandlerMark(Rc<HandlerData>),
    // Native code between sub-evaluations: a higher-order builtin (map, filter, fold, sort)
    // between callbacks, a list/tuple/record literal between elements, a `let rec` group
    // between values. machine::iterate feeds each result to `state`, which says what to
    // evaluate or apply next, so the progress lives on the continuation, where `perform` can
    // capture it and `resume` can replay it (multi-shot). `state` is shared copy-on-write:
    // cloning this frame (a capture, a splice) is an Rc bump, and the driver copies the
    // state only when it is about to change one a captured continuation still holds.
    // `pending` is the second argument of a curried two-argument call (fold's `x` in
    // f(acc)(x), sort's `b`) whose first half is running.
    Iterate { state: Rc<Iter>, pending: Option<Value> },
}

// Persistent, Nil-terminated linked stack of frames. Cloning a Cont is an
// Rc clone -- O(1) -- which is what lets a captured continuation be resumed
// more than once without the calls interfering with each other.
#[derive(Clone)]
pub struct Cont(pub Rc<ContNode>);

pub enum ContNode {
    Nil,
    Frame(Frame, Cont),
}

impl Frame {
    // A fresh Iterate frame for a state that has no second argument pending.
    pub fn iterate(state: Iter) -> Frame {
        Frame::Iterate { state: Rc::new(state), pending: None }
    }
}

impl Cont {
    pub fn nil() -> Cont {
        Cont(Rc::new(ContNode::Nil))
    }

    pub fn cons(frame: Frame, rest: Cont) -> Cont {
        Cont(Rc::new(ContNode::Frame(frame, rest)))
    }

    // Fold `frames` onto `tail`, frames[0] ending up closest to the top
    // (the next one popped). Shared by `append` below and by
    // `machine::perform`, which captures frames while searching for a
    // handler and needs this exact same rebuild -- one fold, not two.
    pub fn from_frames(frames: Vec<Frame>, tail: Cont) -> Cont {
        let mut acc = tail;
        for f in frames.into_iter().rev() {
            acc = Cont::cons(f, acc);
        }
        acc
    }

    // Splice `k` (a captured continuation) in front of `tail`. O(len(k)):
    // rebuilds k's frames onto tail since a singly-linked persistent list
    // has no O(1) append. Fine at skeleton scale; revisit if profiling
    // shows resume-heavy code paying for it.
    pub fn append(k: &Cont, tail: &Cont) -> Cont {
        let mut frames = Vec::new();
        let mut node = k.clone();
        loop {
            match &*node.0 {
                ContNode::Nil => break,
                ContNode::Frame(f, rest) => {
                    frames.push(f.clone());
                    node = rest.clone();
                }
            }
        }
        Cont::from_frames(frames, tail.clone())
    }
}

// Rust's default Drop for a struct wrapping Rc<ContNode> recurses through `rest` one field-drop at a
// time, which is O(depth) native stack for a long continuation chain (many
// nested lets/binops evaluated in sequence build exactly this shape).
// Unwind iteratively instead via Rc::try_unwrap, stopping the moment some
// other clone of a node is still held elsewhere (a captured/resumable
// continuation, e.g.) -- that node and everything under it stays alive and
// gets cleaned up normally by whoever else holds it.
impl Drop for Cont {
    fn drop(&mut self) {
        let mut node = std::mem::replace(&mut self.0, Rc::new(ContNode::Nil));
        loop {
            match Rc::try_unwrap(node) {
                Ok(ContNode::Frame(_, mut rest)) => node = std::mem::replace(&mut rest.0, Rc::new(ContNode::Nil)),
                _ => break,
            }
        }
    }
}
