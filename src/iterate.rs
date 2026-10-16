// Step-driven native computations: the machine's one mechanism for native code that keeps
// partial state across sub-evaluations. The higher-order builtins (map, filter, fold, sort)
// are state machines driven one callback at a time; a literal's elements and a `let rec`
// group's values accumulate the same way, one sub-expression at a time. The state lives in a
// Frame::Iterate on the continuation (machine::iterate), not in a native Rust loop, so an
// effect performed in a callback reaches the handler around the builtin call, multi-shot
// included, and a callback cannot nest the native stack.

use std::rc::Rc;

use crate::env::Env;
use crate::expr::{ExprRef, SpanMap};
use crate::frame::Bindings;
use crate::types::{close_group, Type};
use crate::value::Value;

// What a step-driven computation asks the machine to do next.
#[derive(Clone)]
pub enum Next {
    // Apply `f` to `a`; if `b` is Some, apply that result to `b` (a curried two-argument
    // call: fold's f(acc)(x), sort's cmp(a)(b)). The final result is fed back to `step`.
    // The frame stays.
    Call { f: Value, a: Value, b: Option<Value> },
    // Evaluate the expression in the environment and feed its value back to `step`. The
    // frame stays.
    Eval(ExprRef, Env),
    // Pop the frame, then evaluate (a `let rec` group's body).
    Tail(ExprRef, Env),
    // Finished: pop the frame and continue with this value.
    Done(Value),
}

// The bindings of a `let rec` (Expr::LetRec's first field).
pub type LetRecBindings = Rc<Vec<(String, Option<ExprRef>, ExprRef)>>;

// The state of one running builtin. A Frame::Iterate holds it in an `Rc`, shared copy-on-
// write: a capture by `perform` and a splice by `resume` are reference-count bumps, and the
// driver copies the state (`Rc::make_mut`) only when it is about to change one that some
// captured continuation still holds.
#[derive(Clone)]
pub enum Iter {
    // `i` is the index of the element whose callback is in flight.
    Map { f: Value, items: Rc<Vec<Value>>, i: usize, out: Vec<Value> },
    Filter { p: Value, items: Rc<Vec<Value>>, i: usize, out: Vec<Value> },
    // The accumulator is not stored: it travels as the call's argument and comes back as
    // the step's input, so fold's state is O(1) and a multi-shot branch copies nothing.
    Fold { f: Value, items: Rc<Vec<Value>>, i: usize },
    // Boxed: much larger than the others.
    Sort(Box<Sort>),
    // A list or tuple literal (`record` None) or a record literal's field values, evaluated
    // left to right; `remaining` is reversed so `pop` takes the next, `done` holds the values.
    Elems { record: Option<Rc<Vec<String>>>, remaining: Vec<ExprRef>, done: Vec<Value>, env: Env },
    // A non-group `let rec`: the values run in the outer scope, then `body` runs on a frame
    // of them (same order as the names).
    LetRec { remaining: Vec<ExprRef>, done: Vec<Value>, body: ExprRef, env: Env },
    // A `let rec` of unannotated calls (resolve::GroupKind::Types): its values run in
    // `vals_env`, a frame of one placeholder type per name (`ids`); `done` holds the finished
    // ones. After the last, close_group builds the recursive types and `body` runs in a frame
    // of the finished values on `env`.
    TypeGroup { bindings: LetRecBindings, body: ExprRef, ids: Rc<[u64]>, done: Vec<Value>, env: Env, vals_env: Env },
}

// The driver's one way to advance a state a frame holds: copy-on-write, so a state a
// captured continuation shares is copied first and each resumption continues from the
// state as captured.
pub fn step_shared(state: &mut Rc<Iter>, v: Value, spans: &SpanMap) -> Next {
    Rc::make_mut(state).step(v, spans)
}

fn list(items: Vec<Value>) -> Value {
    if !items.is_empty() {
        crate::list::count_writes(items.len());
        crate::list::count_block();
    }
    Value::List(Rc::new(items))
}

impl Iter {
    // The constructors return the state and its first request (Done at once for an empty
    // list).
    pub fn map(f: Value, items: Rc<Vec<Value>>) -> (Iter, Next) {
        let out = Vec::with_capacity(items.len());
        let next = match items.first() {
            Some(x) => Next::Call { f: f.clone(), a: x.clone(), b: None },
            None => Next::Done(list(Vec::new())),
        };
        (Iter::Map { f, items, i: 0, out }, next)
    }

    pub fn filter(p: Value, items: Rc<Vec<Value>>) -> (Iter, Next) {
        let out = Vec::with_capacity(items.len());
        let next = match items.first() {
            Some(x) => Next::Call { f: p.clone(), a: x.clone(), b: None },
            None => Next::Done(list(Vec::new())),
        };
        (Iter::Filter { p, items, i: 0, out }, next)
    }

    pub fn fold(f: Value, init: Value, items: Rc<Vec<Value>>) -> (Iter, Next) {
        let next = match items.first() {
            Some(x) => Next::Call { f: f.clone(), a: init, b: Some(x.clone()) },
            None => Next::Done(init),
        };
        (Iter::Fold { f, items, i: 0 }, next)
    }

    pub fn sort(cmp: Value, items: &[Value]) -> (Iter, Next) {
        let mut sort = Sort::new(cmp, items);
        let next = match sort.question() {
            Some((a, b)) => Next::Call { f: sort.cmp.clone(), a, b: Some(b) },
            None => Next::Done(list(std::mem::take(&mut sort.src))),
        };
        (Iter::Sort(Box::new(sort)), next)
    }

    // Starts a list/tuple literal (`record` None) or a record literal's values: the state and
    // the first expression to evaluate. `items` is never empty (an empty list literal is a
    // constant, and a tuple or record has at least one element).
    pub fn elems(record: Option<Rc<Vec<String>>>, mut items: Vec<ExprRef>, env: Env) -> (Iter, ExprRef) {
        items.reverse();
        let first = items.pop().expect("an elems literal has at least one element");
        (Iter::Elems { record, remaining: items, done: Vec::new(), env }, first)
    }

    // Starts a non-group `let rec`: the state and the first value to evaluate (the parser
    // never emits an empty group).
    pub fn let_rec(mut values: Vec<ExprRef>, body: ExprRef, env: Env) -> (Iter, ExprRef) {
        values.reverse();
        let first = values.pop().expect("the parser never emits an empty group");
        (Iter::LetRec { remaining: values, done: Vec::new(), body, env }, first)
    }

    // Feeds one result: a callback's, or an evaluated element's. The driver calls this only
    // through `Rc::make_mut`, so the `mem::take`s below never empty a state a captured
    // continuation holds. `spans` is for a type group's error.
    pub fn step(&mut self, v: Value, spans: &SpanMap) -> Next {
        match self {
            Iter::Elems { record, remaining, done, env } => {
                done.push(v);
                match remaining.pop() {
                    Some(next) => Next::Eval(next, env.clone()),
                    None => {
                        let done = std::mem::take(done);
                        Next::Done(match record {
                            None => list(done),
                            Some(names) => Value::Record(Rc::new(names.iter().cloned().zip(done).collect())),
                        })
                    }
                }
            }
            Iter::LetRec { remaining, done, body, env } => {
                done.push(v);
                match remaining.pop() {
                    Some(next) => Next::Eval(next, env.clone()),
                    None => {
                        let mut frame = Bindings::default();
                        for v in std::mem::take(done) {
                            frame.push(v);
                        }
                        Next::Tail(*body, frame.extend(env))
                    }
                }
            }
            Iter::TypeGroup { bindings, body, ids, done, env, vals_env } => {
                done.push(v);
                if done.len() < bindings.len() {
                    return Next::Eval(bindings[done.len()].2, vals_env.clone());
                }
                let done = std::mem::take(done);
                let results: Vec<Option<Type>> = done.iter().map(|v| if let Value::Type(t) = v { Some((**t).clone()) } else { None }).collect();
                let names: Vec<Rc<str>> = bindings.iter().map(|(n, _, _)| Rc::from(n.as_str())).collect();
                let closed = close_group(&names, ids, &results).unwrap_or_else(|(i, _)| {
                    crate::machine::set_current_span(spans.get(bindings[i].2).copied());
                    crate::run_error!("recursive type {} is not built by a type constructor", names[i])
                });
                let mut frame = Bindings::default();
                for (v, c) in done.into_iter().zip(closed) {
                    frame.push(c.map_or(v, |t| Value::Type(Rc::new(t))));
                }
                Next::Tail(*body, frame.extend(env))
            }
            Iter::Map { f, items, i, out } => {
                out.push(v);
                *i += 1;
                match items.get(*i) {
                    Some(x) => Next::Call { f: f.clone(), a: x.clone(), b: None },
                    None => Next::Done(list(std::mem::take(out))),
                }
            }
            Iter::Filter { p, items, i, out } => {
                if v.as_bool() {
                    out.push(items[*i].clone());
                }
                *i += 1;
                match items.get(*i) {
                    Some(x) => Next::Call { f: p.clone(), a: x.clone(), b: None },
                    None => Next::Done(list(std::mem::take(out))),
                }
            }
            Iter::Fold { f, items, i } => {
                *i += 1;
                match items.get(*i) {
                    Some(x) => Next::Call { f: f.clone(), a: v, b: Some(x.clone()) },
                    None => Next::Done(v),
                }
            }
            Iter::Sort(s) => match s.answer(v.as_bool()) {
                Some((a, b)) => Next::Call { f: s.cmp.clone(), a, b: Some(b) },
                None => Next::Done(list(std::mem::take(&mut s.src))),
            },
        }
    }
}

// A natural merge sort driven one comparison at a time. `cmp(x, y)` means "x may stay
// before y": a run is ascending while cmp(prev, next) and strictly descending while not,
// decided by its first pair (a descending run is reversed in place), and a merge takes the
// left element iff cmp(left, right). With a `<=`-style comparator the sort is stable; a
// strict `<` orders ties right-first. Sorted and strictly reversed input take n-1
// comparisons (one run), random input about n*log2(n/2), and the number of questions is
// at most n-1 + n*ceil(log2 n) whatever the answers, so an inconsistent comparator still
// terminates with a permutation.
#[derive(Clone)]
pub struct Sort {
    pub cmp: Value,
    // The runs; the result at the end.
    pub src: Vec<Value>,
    // The merge target of the current pass.
    dst: Vec<Value>,
    // Ends of the runs in `src` (ascending; the last is len once the scan is done).
    runs: Vec<usize>,
    // Ends of the runs `dst` gets in the current pass.
    next_runs: Vec<usize>,
    at: At,
}

#[derive(Clone, Copy)]
enum At {
    // Run scan: the current run is src[start..i], and the pending question is
    // cmp(src[i-1], src[i]). `desc` is None until the run's direction is known.
    Scan { start: usize, i: usize, desc: Option<bool> },
    // Merging runs[k] = src[..mid] with runs[k+1] = src[mid..hi] into dst: the pending
    // question is cmp(src[l], src[r]).
    Merge { k: usize, mid: usize, hi: usize, l: usize, r: usize },
}

impl Sort {
    fn new(cmp: Value, items: &[Value]) -> Sort {
        Sort {
            cmp,
            src: items.to_vec(),
            dst: Vec::with_capacity(items.len()),
            runs: Vec::new(),
            next_runs: Vec::new(),
            at: At::Scan { start: 0, i: 1, desc: None },
        }
    }

    // Applies the answer to the pending question, then runs to the next question
    // (None: sorted). `At` is Copy: matched by value and written back, so close_run can
    // borrow self.
    fn answer(&mut self, before: bool) -> Option<(Value, Value)> {
        match self.at {
            At::Scan { start, i, desc } => match desc {
                None => self.at = At::Scan { start, i: i + 1, desc: Some(!before) },
                Some(d) if before != d => self.at = At::Scan { start, i: i + 1, desc },
                Some(_) => {
                    self.close_run(start, i, desc);
                    self.at = At::Scan { start: i, i: i + 1, desc: None };
                }
            },
            At::Merge { k, mid, hi, l, r } => {
                let (l, r) = if before {
                    self.dst.push(self.src[l].clone());
                    (l + 1, r)
                } else {
                    self.dst.push(self.src[r].clone());
                    (l, r + 1)
                };
                self.at = At::Merge { k, mid, hi, l, r };
            }
        }
        self.question()
    }

    fn close_run(&mut self, start: usize, end: usize, desc: Option<bool>) {
        if desc == Some(true) {
            self.src[start..end].reverse();
        }
        self.runs.push(end);
    }

    // Advances through everything that needs no comparison to the next pending question.
    fn question(&mut self) -> Option<(Value, Value)> {
        let n = self.src.len();
        loop {
            match self.at {
                At::Scan { start, i, desc } => {
                    if i < n {
                        return Some((self.src[i - 1].clone(), self.src[i].clone()));
                    }
                    if start < n {
                        self.close_run(start, n, desc);
                    }
                    if self.runs.len() <= 1 {
                        return None;
                    }
                    self.begin_pass();
                }
                At::Merge { k, mid, hi, l, r } => {
                    if l < mid && r < hi {
                        return Some((self.src[l].clone(), self.src[r].clone()));
                    }
                    // At most one of these two is non-empty.
                    self.dst.extend_from_slice(&self.src[l..mid]);
                    self.dst.extend_from_slice(&self.src[r..hi]);
                    self.next_runs.push(hi);
                    let k = k + 2; // the next pair is runs k and k+1
                    if k + 1 < self.runs.len() {
                        let (mid, end) = (self.runs[k], self.runs[k + 1]);
                        self.at = At::Merge { k, mid, hi: end, l: hi, r: mid };
                        continue;
                    }
                    if k < self.runs.len() {
                        // An odd run left over: copy it.
                        let end = self.runs[k];
                        self.dst.extend_from_slice(&self.src[hi..end]);
                        self.next_runs.push(end);
                    }
                    std::mem::swap(&mut self.src, &mut self.dst);
                    self.dst.clear();
                    self.runs = std::mem::take(&mut self.next_runs);
                    if self.runs.len() <= 1 {
                        return None;
                    }
                    self.begin_pass();
                }
            }
        }
    }

    // Starts a pass at the pair (runs[0], runs[1]); needs runs.len() >= 2.
    fn begin_pass(&mut self) {
        let (mid, hi) = (self.runs[0], self.runs[1]);
        self.at = At::Merge { k: 0, mid, hi, l: 0, r: mid };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Same generator as machine.rs's test Rng.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn int(v: &Value) -> i64 {
        match v {
            Value::Int(n) => *n,
            _ => panic!("not an int"),
        }
    }

    fn ints(v: &[i64]) -> Vec<Value> {
        v.iter().map(|&n| Value::Int(n)).collect()
    }

    // Drives a sort to the end, answering each question with `answer`; returns the result
    // and the number of questions asked.
    fn drive(items: &[i64], mut answer: impl FnMut(i64, i64) -> bool) -> (Vec<i64>, usize) {
        let (mut iter, mut next) = Iter::sort(Value::Bool(true), &ints(items));
        let mut asked = 0;
        loop {
            match next {
                Next::Call { a, b: Some(b), .. } => {
                    asked += 1;
                    next = iter.step(Value::Bool(answer(int(&a), int(&b))), &SpanMap::new());
                }
                Next::Done(Value::List(out)) => return (out.iter().map(int).collect(), asked),
                _ => panic!("a sort asks two-argument questions and finishes with a list"),
            }
        }
    }

    // The bound on questions for any answers: runs scan, then ceil(log2 n) merge passes.
    fn bound(n: usize) -> usize {
        n.saturating_sub(1) + n * (usize::BITS - n.saturating_sub(1).leading_zeros()) as usize
    }

    // A random list whose keys repeat; the value is key * 1000 + index, so equal keys
    // are told apart by index.
    fn random_items(r: &mut Rng, n: usize, keys: u64) -> Vec<i64> {
        (0..n).map(|i| (r.next() % keys) as i64 * 1000 + i as i64).collect()
    }

    #[test]
    fn matches_a_stable_sort_by_key() {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        for n in (0..40).chain([100, 257, 1000]) {
            for keys in [1, 2, 5, 1000] {
                let items = random_items(&mut r, n, keys);
                let (got, _) = drive(&items, |a, b| a / 1000 <= b / 1000);
                let mut want = items.clone();
                want.sort_by_key(|x| x / 1000);
                assert_eq!(got, want, "n={n} keys={keys}");
            }
        }
    }

    #[test]
    fn sorted_and_reversed_input_take_n_minus_1_comparisons() {
        for n in [0usize, 1, 2, 3, 10, 1000] {
            let up: Vec<i64> = (0..n as i64).collect();
            let down: Vec<i64> = up.iter().rev().cloned().collect();
            let (got, asked) = drive(&up, |a, b| a <= b);
            assert_eq!((got, asked), (up.clone(), n.saturating_sub(1)));
            let (got, asked) = drive(&down, |a, b| a <= b);
            assert_eq!((got, asked), (up, n.saturating_sub(1)));
        }
    }

    #[test]
    fn random_input_stays_within_the_comparison_bound() {
        let mut r = Rng(12345);
        for n in [2usize, 3, 7, 64, 1000, 4096] {
            let items = random_items(&mut r, n, 1_000_000);
            let (_, asked) = drive(&items, |a, b| a <= b);
            assert!(asked <= bound(n), "n={n} asked={asked} bound={}", bound(n));
        }
    }

    #[test]
    fn an_inconsistent_comparator_terminates_with_a_permutation() {
        let mut r = Rng(777);
        for n in [2usize, 3, 5, 16, 100, 1000] {
            let items = random_items(&mut r, n, 50);
            let mut coin = Rng(n as u64 * 31 + 1);
            let (got, asked) = drive(&items, |_, _| coin.next() & 1 == 0);
            assert!(asked <= bound(n), "n={n} asked={asked} bound={}", bound(n));
            let (mut got, mut want) = (got, items);
            got.sort();
            want.sort();
            assert_eq!(got, want, "n={n}");
        }
    }

    // A captured state (a clone) and the live one continue independently: each ends where
    // a fresh run fed the same full answer stream ends.
    #[test]
    fn a_cloned_state_continues_independently() {
        let mut r = Rng(4242);
        for n in [5usize, 20, 200] {
            let items = random_items(&mut r, n, 10);
            for prefix in [0usize, 1, 3, n, 2 * n] {
                // Run `prefix` questions from a random stream, then clone mid-way.
                let (iter, mut next) = Iter::sort(Value::Bool(true), &ints(&items));
                let mut iter = Rc::new(iter);
                let mut stream = Vec::new();
                let mut coin = Rng(n as u64 + prefix as u64 + 9);
                while stream.len() < prefix {
                    if !matches!(next, Next::Call { .. }) {
                        break;
                    }
                    let ans = coin.next() & 1 == 0;
                    stream.push(ans);
                    next = step_shared(&mut iter, Value::Bool(ans), &SpanMap::new());
                }
                // The fork shares the state, as a captured continuation does: the first step
                // of each copy must go through the driver's copy-on-write.
                let fork = (Rc::clone(&iter), next.clone());
                // Each copy finishes with its own answer stream.
                for ((mut it, mut nx), bias) in [((iter, next), 0u64), (fork, 1)] {
                    let mut full = stream.clone();
                    let mut c = Rng(1000 + bias * 17 + n as u64);
                    let out = loop {
                        match nx {
                            Next::Call { .. } => {
                                let ans = c.next() % 3 != bias;
                                full.push(ans);
                                nx = step_shared(&mut it, Value::Bool(ans), &SpanMap::new());
                            }
                            Next::Done(Value::List(out)) => break out.iter().map(int).collect::<Vec<_>>(),
                            _ => unreachable!(),
                        }
                    };
                    let mut replay = full.into_iter();
                    let (fresh, _) = drive(&items, |_, _| replay.next().expect("the stream is as long as the questions"));
                    assert_eq!(out, fresh, "n={n} prefix={prefix} bias={bias}");
                }
            }
        }
    }

    #[test]
    fn map_filter_fold_step_through_their_elements() {
        let items = Rc::new(ints(&[1, 2, 3]));
        let (mut it, mut next) = Iter::map(Value::Bool(true), items.clone());
        let mut seen = Vec::new();
        while let Next::Call { a, b: None, .. } = next {
            seen.push(int(&a));
            next = it.step(Value::Int(int(&a) * 10), &SpanMap::new());
        }
        assert_eq!(seen, [1, 2, 3]);
        assert!(matches!(&next, Next::Done(Value::List(out)) if out.iter().map(int).collect::<Vec<_>>() == [10, 20, 30]));

        let (mut it, mut next) = Iter::filter(Value::Bool(true), items.clone());
        while let Next::Call { a, .. } = next {
            next = it.step(Value::Bool(int(&a) != 2), &SpanMap::new());
        }
        assert!(matches!(&next, Next::Done(Value::List(out)) if out.iter().map(int).collect::<Vec<_>>() == [1, 3]));

        let (mut it, mut next) = Iter::fold(Value::Bool(true), Value::Int(100), items);
        let mut seen = Vec::new();
        while let Next::Call { a, b: Some(b), .. } = next {
            seen.push((int(&a), int(&b)));
            next = it.step(Value::Int(int(&a) + int(&b)), &SpanMap::new());
        }
        assert_eq!(seen, [(100, 1), (101, 2), (103, 3)]);
        assert!(matches!(next, Next::Done(Value::Int(106))));
    }
}
