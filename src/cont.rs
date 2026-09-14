use std::rc::Rc;

use crate::env::Env;
use crate::expr::{BinOp, ExprRef};
use crate::types::Type;
use crate::value::{HandlerData, Value};

// One step of "what's left to do", defunctionalized so it can live as data
// (not the native Rust call stack). This is what makes multi-shot resume
// possible: a captured Cont is an immutable, cheaply-clonable value.
#[derive(Clone)]
pub enum Frame {
    AppFunc { arg: ExprRef, env: Env },
    AppArg { func: Value },
    LetBody { var: String, body: ExprRef, env: Env },
    BinOpL { op: BinOp, rhs: ExprRef, env: Env },
    BinOpR { op: BinOp, lhs: Value },
    If { then_: ExprRef, else_: ExprRef, env: Env },
    CheckFrame { ty: Type },
    PerformPayload { effect: String },
    // `handle body with handler_expr`: handler_expr has just evaluated to a
    // Value::Handler -- next step installs it as a HandlerMark and evals body.
    InstallHandler { body: ExprRef, env: Env },
    // Wraps the same HandlerData a Value::Handler carries -- an Rc clone
    // (one pointer bump) instead of six separately-cloned fields, and one
    // definition instead of two structurally-identical ones to keep in sync.
    HandlerMark(Rc<HandlerData>),
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

// Same shape, same fix as plist.rs's PList<T>: Rust's default Drop for a
// struct wrapping Rc<ContNode> recurses through `rest` one field-drop at a
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
