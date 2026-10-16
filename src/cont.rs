use std::rc::Rc;

use crate::env::Env;
use crate::expr::{AnnotSpec, BinOp, CheckSpec, ExprRef, Lazy, Len, Pattern};
use crate::iterate::Iter;
use crate::span::Span;
use crate::value::{HandlerData, SiteCtx, Value};

// One step of "what's left to do", defunctionalized so it can live as data
// (not the native Rust call stack). This is what makes multi-shot resume
// possible: a captured continuation (`Captured`) is an immutable, cheaply-clonable value.
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
    // Value::Handler -- next step installs it as a Mark (see Meta) and evals body.
    InstallHandler { body: ExprRef, env: Env },
    // Native code between sub-evaluations: a higher-order builtin (map, filter, fold, sort)
    // between callbacks, a list/tuple/record literal between elements, a `let rec` group
    // between values. machine::iterate feeds each result to `state`, which says what to
    // evaluate or apply next, so the progress lives on the continuation, where `perform` can
    // capture it and `resume` can replay it (multi-shot). `state` is shared copy-on-write:
    // cloning this frame (copying a node a captured continuation shares, see
    // `Cont::unshared_top`) is an Rc bump, and the driver copies the state only when it is
    // about to change one a captured continuation still holds.
    // `pending` is the second argument of a curried two-argument call (fold's `x` in
    // f(acc)(x), sort's `b`) whose first half is running.
    Iterate { state: Rc<Iter>, pending: Option<Value> },
}

// Test-only operation counters (the same pattern as list.rs): the U11 complexity tests count
// operations instead of timing them, so they are deterministic and fast in a debug build.
// Thread-locals, read on the thread that ran the program.
#[cfg(test)]
thread_local! {
    // Each `Cont::cons`: one frame node allocated.
    pub(crate) static CONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // The cost of capture and resume: each mark `capture` visits while searching for its
    // handler, plus each mark `resume` pushes.
    pub(crate) static WALKED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_counters() {
    CONS.with(|c| c.set(0));
    WALKED.with(|c| c.set(0));
}

// (cons, walked)
#[cfg(test)]
pub(crate) fn counters() -> (usize, usize) {
    (CONS.with(|c| c.get()), WALKED.with(|c| c.get()))
}

#[allow(unused_variables)]
fn count_walked(n: usize) {
    #[cfg(test)]
    WALKED.with(|c| c.set(c.get() + n));
}

// Persistent linked stack of frames: one SEGMENT, the frames between two handlers (the handlers
// themselves are `Meta`'s marks, below it). `None` is the empty stack, which costs no
// allocation (the same shape as `Env`). Cloning a Cont is an Rc clone -- O(1) -- so capture
// and resume move a whole segment by pointer.
#[derive(Clone, Default)]
pub struct Cont(Option<Rc<Node>>);

pub struct Node {
    pub frame: Frame,
    pub rest: Cont,
}

impl Frame {
    // A fresh Iterate frame for a state that has no second argument pending.
    pub fn iterate(state: Iter) -> Frame {
        Frame::Iterate { state: Rc::new(state), pending: None }
    }
}

impl Cont {
    pub fn nil() -> Cont {
        Cont(None)
    }

    // The one place a frame node is allocated.
    fn new_node(frame: Frame, rest: Cont) -> Rc<Node> {
        #[cfg(test)]
        CONS.with(|c| c.set(c.get() + 1));
        Rc::new(Node { frame, rest })
    }

    pub fn cons(frame: Frame, rest: Cont) -> Cont {
        Cont(Some(Cont::new_node(frame, rest)))
    }

    // The top node, or None for the empty stack.
    pub fn node(&self) -> Option<&Node> {
        self.0.as_deref()
    }

    // The top frame, or None for the empty stack.
    pub fn top(&self) -> Option<&Frame> {
        self.node().map(|n| &n.frame)
    }

    // The top node, uniquely owned: copied (one frame clone, which for an Iterate frame bumps
    // its state's count) if anything else holds it, which is the normal case after a resume
    // while the captured continuation is alive (its segment's top node IS the live top).
    // The one place a node is mutated in place; everything else is immutable once consed.
    // Panics on the empty stack.
    pub fn unshared_top(&mut self) -> &mut Node {
        let rc = self.0.as_mut().expect("internal: the top of an Iterate chain is a frame");
        if Rc::get_mut(rc).is_none() {
            *rc = Cont::new_node(rc.frame.clone(), rc.rest.clone());
        }
        Rc::get_mut(rc).expect("internal: just unshared")
    }

    pub fn is_nil(&self) -> bool {
        self.0.is_none()
    }
}

// What runs after the current segment of frames returns, in two kinds (a `Mark`):
//   - `handler: Some(h)`: the current segment is the body of `handle .. with h`; `rest` is what
//     runs after the `handle` expression.
//   - `handler: None`: a join. A shallow `resume` cannot concatenate the captured segment in
//     front of the resumer's segment without copying, so it leaves the resumer's segment
//     (`rest`) here, to continue into when the captured one returns.
// A perform skips joins when it looks for its handler, but captures them with the frames above
// the handler: their `rest` is pending work.
#[derive(Clone)]
pub struct Mark {
    handler: Option<Rc<HandlerData>>,
    rest: Cont,
}

// The meta-continuation: the marks below the segment `run_loop` is running, nearest last. The
// frames between two handlers are one segment (a plain `Cont`), so capture and resume move a
// segment by pointer instead of copying its frames: O(marks passed), never O(frames).
#[derive(Default)]
pub struct Meta(Vec<Mark>);

// A captured continuation (the value `resume` is bound to): the segment from the perform up to
// the nearest mark, the marks the search passed (bottom first), and the matched handler if it is
// deep (pushed back below the segment on every resume, so a perform inside the resumed
// computation finds it again). Immutable, so it can be resumed any number of times.
pub struct Captured {
    top: Cont,
    marks: Vec<Mark>,
    reinstall: Option<Rc<HandlerData>>,
}

impl Meta {
    // `handle body with h`: the current segment ends here, and the body runs as a fresh one.
    pub fn install(&mut self, h: Rc<HandlerData>, cont: &mut Cont) {
        let rest = std::mem::take(cont);
        self.0.push(Mark { handler: Some(h), rest });
    }

    // The current segment returned: the segment to continue with (a handler's identity return
    // clause, or a join), or None when the run is over.
    pub fn underflow(&mut self) -> Option<Cont> {
        self.0.pop().map(|m| m.rest)
    }

    // `perform effect`: finds the nearest handler for it, makes `cont` and the marks between
    // into the continuation `resume` is bound to, and leaves the matched handler's own
    // surroundings running (its clause is outside it: an effect it performs goes outward).
    // None when no handler handles `effect`; nothing is changed then.
    pub fn capture(&mut self, effect: &str, cont: &mut Cont) -> Option<(Rc<HandlerData>, Rc<Captured>)> {
        let i = self.0.iter().rposition(|m| m.handler.as_ref().is_some_and(|h| h.effect == effect))?;
        count_walked(self.0.len() - i);
        // Bottom first; joins included.
        let marks: Vec<Mark> = self.0.drain(i + 1..).collect();
        let matched = self.0.pop().expect("internal: rposition found the mark");
        let h = matched.handler.expect("internal: the matched mark has a handler");
        let reinstall = h.deep.then(|| h.clone());
        let top = std::mem::replace(cont, matched.rest);
        Some((h, Rc::new(Captured { top, marks, reinstall })))
    }

    // `resume(v)`: `k` runs in front of whatever comes after the call (`cont`).
    pub fn resume(&mut self, k: &Captured, cont: &mut Cont) {
        let below = std::mem::replace(cont, k.top.clone());
        let mut pushed = k.marks.len();
        // An empty segment below a handler-less mark would only pass the value on.
        if k.reinstall.is_some() || !below.is_nil() {
            self.0.push(Mark { handler: k.reinstall.clone(), rest: below });
            pushed += 1;
        }
        self.0.extend(k.marks.iter().cloned());
        count_walked(pushed);
    }
}

// Rust's default Drop for a struct wrapping Rc<Node> recurses through `rest` one field-drop at a
// time, which is O(depth) native stack for a long continuation chain (many
// nested lets/binops evaluated in sequence build exactly this shape).
// Unwind iteratively instead via Rc::try_unwrap, stopping the moment some
// other clone of a node is still held elsewhere (a captured/resumable
// continuation, e.g.) -- that node and everything under it stays alive and
// gets cleaned up normally by whoever else holds it. Allocates nothing: the
// emptied `rest` is `None`, so each node's own Drop finds nothing to unwind.
impl Drop for Cont {
    fn drop(&mut self) {
        let mut cur = self.0.take();
        while let Some(rc) = cur {
            match Rc::try_unwrap(rc) {
                Ok(mut node) => cur = node.rest.0.take(),
                Err(_) => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The default test-thread stack (~2 MiB) would overflow on a recursive Drop at this depth.
    #[test]
    fn a_captured_segment_of_a_million_frames_drops_on_a_small_stack() {
        let mut chain = Cont::nil();
        for _ in 0..1_000_000 {
            chain = Cont::cons(Frame::PerformPayload { effect: "e".to_string() }, chain);
        }
        drop(Captured { top: chain, marks: Vec::new(), reinstall: None });
    }
}
