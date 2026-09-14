use std::rc::Rc;

use crate::env::Env;
use crate::expr::Expr;
use crate::value::Value;

// One step of "what's left to do", defunctionalized so it can live as data
// (not the native Rust call stack). This is what makes multi-shot resume
// possible: a captured Cont is an immutable, cheaply-clonable value.
#[derive(Clone)]
pub enum Frame {
    AppFunc { arg: Rc<Expr>, env: Env },
    AppArg { func: Value },
    LetBody { var: String, body: Rc<Expr>, env: Env },
    AddL { rhs: Rc<Expr>, env: Env },
    AddR { lhs: Value },
    If0 { then_: Rc<Expr>, else_: Rc<Expr>, env: Env },
    PerformPayload { effect: String },
    // `handle body with handler_expr`: handler_expr has just evaluated to a
    // Value::Handler -- next step installs it as a HandlerMark and evals body.
    InstallHandler { body: Rc<Expr>, env: Env },
    HandlerMark {
        effect: String,
        payload_var: String,
        resume_var: String,
        handler_body: Rc<Expr>,
        env: Env,
        deep: bool,
    },
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
        let mut acc = tail.clone();
        for f in frames.into_iter().rev() {
            acc = Cont::cons(f, acc);
        }
        acc
    }
}
