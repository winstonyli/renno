use std::rc::Rc;

use crate::env::Env;
use crate::expr::{BinOp, CheckSpec, ExprRef, Len, Pattern};
use crate::span::Span;
use crate::value::{HandlerData, Value};

// One step of "what's left to do", defunctionalized so it can live as data
// (not the native Rust call stack). This is what makes multi-shot resume
// possible: a captured Cont is an immutable, cheaply-clonable value.
#[derive(Clone)]
pub enum Frame {
    // `callee_span` is the CALLEE sub-expression's own span (`f` in `f(a)`),
    // not the whole App -- Some when it came from an ordinary Eval of an
    // App expr; None for AppArg frames built directly by machine::apply's
    // native re-entrant call, e.g. fold/map's callback -- there's no App
    // expr behind that, so CURRENT_SPAN is just left alone, same fallback
    // as before this field existed. Carried from AppFunc into AppArg so a
    // "not a function" panic blames `f` itself, not whatever the
    // argument's own span happened to be (the last thing Eval'd before the
    // panic fires, evaluated strictly after `f` and unrelated to why `f`
    // isn't callable).
    AppFunc { arg: ExprRef, env: Env, callee_span: Option<Span> },
    AppArg { func: Value, callee_span: Option<Span> },
    LetBody { body: ExprRef, env: Env },
    // Non-group `let rec` fallback (see machine.rs's Expr::LetRec arm):
    // evaluating the binding values left to right in the OUTER scope;
    // `remaining`/`done` mirror ListElems's accumulation shape.
    LetRecBody { remaining: Vec<ExprRef>, done: Vec<Value>, body: ExprRef, env: Env },
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
    // Evaluates a list literal's elements left to right. `remaining` are
    // not-yet-evaluated; `done` accumulates results in order.
    ListElems { remaining: Vec<ExprRef>, done: Vec<Value>, env: Env },
    // Evaluating a record's field VALUES left to right -- `names` is the
    // fixed, full field-name list for the whole record (needed once every
    // value is in, to zip back together into the final Value::Record;
    // LetRecBody just above accumulates `remaining`/`done` the same way,
    // but has no `names` field of its own -- its fallback just binds the
    // finished values in order via a runtime Bindings frame, since the
    // resolver already fixed each name's slot by position, with no
    // Value::Record-shaped result to zip back together). Not ListElems:
    // that always wraps its `done` as a plain Value::List, which is right
    // for Tuple/ListLit but wrong here -- see Value::Record's own doc
    // comment for why records need a
    // name-keyed runtime shape ListElems has no way to produce.
    RecordElems { names: Rc<Vec<String>>, remaining: Vec<ExprRef>, done: Vec<Value>, env: Env },
    // The operand of an Expr::Check has just evaluated -- test it against
    // the spec, with the length slots read when the Check was entered.
    Check { spec: Rc<CheckSpec>, lens: Vec<Len> },
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
