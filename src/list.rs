// The list representation: a claim-list (a Bagwell VList whose growth slots are claimed by
// the first cons that reaches them).
//
// One structure whose two extremes are an array (one block, built in bulk) and a cons list
// (blocks of one slot). A block is a fixed-capacity array of write-once slots, filled
// bottom-up; a view `(block, top)` sees `slots[..top]`, the head being `slots[top - 1]`, and
// the rest of the list is whatever sits `below` the block. Consing onto a view whose next
// slot is empty *claims* that slot, so building by repeated cons is O(1) amortized and
// contiguous; the first cons above a position wins it, and every later cons at the same
// position (a branch: `a = x :: xs`, `b = y :: xs`) starts a fresh block over the same
// tail. A slot is written at most once, so no view's contents ever change and the structure
// is persistent in safe Rust.
//
// Cost: `::`, head and tail are O(1); `++` is O(|a|) whatever |b|; `get` is O(1) on a
// bulk-built list (a literal, `range`, a builtin's result, a tuple) and O(blocks) on a
// cons-built one (O(log n) when built linearly, up to O(n) for a list built by repeated
// branching, as for a cons list).

use std::cell::{Cell, OnceCell};
use std::rc::Rc;

use crate::value::Value;

// Test-only operation counters: the complexity tests count operations instead of timing
// them, so they are deterministic and fast in a debug build. Thread-locals, read on the
// thread that ran the program (the same pattern as machine::LAZY_MINTS).
#[cfg(test)]
thread_local! {
    // Each Value placed into list storage.
    pub(crate) static WRITES: Cell<usize> = const { Cell::new(0) };
    // Each storage allocation.
    pub(crate) static BLOCKS: Cell<usize> = const { Cell::new(0) };
    // Each block visited by `get`.
    pub(crate) static HOPS: Cell<usize> = const { Cell::new(0) };
    // Each empty slot created: room reserved for later conses, or a slot that is filled in a
    // second pass. An exactly-sized block is built in one pass and creates none.
    pub(crate) static EMPTIES: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_counters() {
    WRITES.with(|c| c.set(0));
    BLOCKS.with(|c| c.set(0));
    HOPS.with(|c| c.set(0));
    EMPTIES.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn empties() -> usize {
    EMPTIES.with(|c| c.get())
}

#[cfg(test)]
pub(crate) fn counters() -> (usize, usize, usize) {
    (WRITES.with(|c| c.get()), BLOCKS.with(|c| c.get()), HOPS.with(|c| c.get()))
}

#[allow(unused_variables)]
fn count_writes(n: usize) {
    #[cfg(test)]
    WRITES.with(|c| c.set(c.get() + n));
}

fn count_block() {
    #[cfg(test)]
    BLOCKS.with(|c| c.set(c.get() + 1));
}

#[allow(unused_variables)]
fn count_empties(n: usize) {
    #[cfg(test)]
    EMPTIES.with(|c| c.set(c.get() + n));
}

fn count_hop() {
    #[cfg(test)]
    HOPS.with(|c| c.set(c.get() + 1));
}

// A view of a list: Clone is one Rc bump.
#[derive(Clone, Default)]
pub struct List {
    // None: the empty list.
    block: Option<Rc<Block>>,
    // This view sees `block.slots[..top]`; top >= 1 whenever `block` is Some.
    top: usize,
    // top + below.len, cached.
    len: usize,
}

struct Block {
    // Physical order bottom-up: slots[top - 1] is a view's head.
    slots: Box<[OnceCell<Value>]>,
    // The list under slots[0].
    below: List,
    // The growth claim: set by the first cons that doubles on top of this block, and born
    // set for a bulk block (sealed), so a second cons onto the same full block never
    // reserves a large one.
    grown: Cell<bool>,
}

// Views are made three ways, each preserving "every view (b, top) has b.slots[..top] set":
// a bulk build sets every slot at creation; a successful `OnceCell::set` at slot `top` makes
// the view `top + 1`; a tail makes `top - 1`, and at `top == 1` returns `below`.
impl List {
    pub fn empty() -> List {
        List::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    // The bulk build: one exactly-sized, sealed block.
    pub fn from_vec(items: Vec<Value>) -> List {
        let len = items.len();
        if len == 0 {
            return List::empty();
        }
        count_writes(len);
        count_block();
        let slots: Box<[OnceCell<Value>]> = items.into_iter().rev().map(OnceCell::from).collect();
        List { block: Some(Rc::new(Block { slots, below: List::empty(), grown: Cell::new(true) })), top: len, len }
    }

    pub fn cons(&self, head: Value) -> List {
        self.push_all(std::iter::once(head))
    }

    // The one growth primitive: `items` (bottom-first) go on top of this list.
    //   1. Claim: while the view's next slot exists and is empty, fill it.
    //   2. A new block for what is left (below = the view so far). Its capacity is
    //      `max(left, 2 * capacity)` when the view sits at the top of a full, unsealed block
    //      and this call wins that block's growth claim, otherwise exactly `left`.
    // So a list built by repeated cons grows in blocks of 1, 2, 4, ...; a cons onto a branch
    // point, a bulk list or a tail view gets an exact block.
    pub fn push_all(&self, mut items: impl ExactSizeIterator<Item = Value>) -> List {
        let (mut top, mut len) = (self.top, self.len);
        let mut refused = None;
        if let Some(b) = &self.block {
            while top < b.slots.len() {
                let Some(x) = items.next() else { break };
                match b.slots[top].set(x) {
                    Ok(()) => {
                        count_writes(1);
                        top += 1;
                        len += 1;
                    }
                    Err(x) => {
                        refused = Some(x);
                        break;
                    }
                }
            }
        }
        let claimed = List { block: self.block.clone(), top, len };
        let left = items.len() + usize::from(refused.is_some());
        if left == 0 {
            return claimed;
        }
        let capacity = match &claimed.block {
            Some(b) if top == b.slots.len() && !b.grown.replace(true) => left.max(2 * b.slots.len()),
            _ => left,
        };
        count_writes(left);
        count_block();
        count_empties(capacity);
        let slots: Box<[OnceCell<Value>]> = (0..capacity).map(|_| OnceCell::new()).collect();
        for (slot, x) in slots.iter().zip(refused.into_iter().chain(items)) {
            let _ = slot.set(x);
        }
        List { len: claimed.len + left, block: Some(Rc::new(Block { slots, below: claimed, grown: Cell::new(false) })), top: left }
    }

    // `self ++ b`: O(1) when either side is empty, otherwise O(|self|) and `b` is never
    // copied: `self`'s elements are consed on top of it (a `[x] ++ acc` loop behaves exactly
    // like `x :: acc`).
    pub fn append(&self, b: &List) -> List {
        if self.is_empty() {
            return b.clone();
        }
        if b.is_empty() {
            return self.clone();
        }
        b.push_all(self.to_vec().into_iter().rev())
    }

    pub fn first(&self) -> Option<&Value> {
        let b = self.block.as_ref()?;
        Some(b.slots[self.top - 1].get().expect("a view's slots are set"))
    }

    // The head and the rest, in O(1).
    pub fn split_first(&self) -> Option<(&Value, List)> {
        let head = self.first()?;
        let b = self.block.as_ref()?;
        let tail = if self.top == 1 { b.below.clone() } else { List { block: self.block.clone(), top: self.top - 1, len: self.len - 1 } };
        Some((head, tail))
    }

    // The i-th element: O(1) within the top block, one more step per block below it. There
    // is deliberately no `Index` impl, so the O(blocks) cost stays visible at call sites.
    pub fn get(&self, i: usize) -> Option<&Value> {
        if i >= self.len {
            return None;
        }
        let (mut block, mut top, mut i) = (self.block.as_deref(), self.top, i);
        loop {
            let b = block?;
            count_hop();
            if i < top {
                return b.slots[top - 1 - i].get();
            }
            i -= top;
            top = b.below.top;
            block = b.below.block.as_deref();
        }
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter { block: self.block.as_deref(), pos: self.top, left: self.len }
    }

    // A copy of the elements in order: for the cold paths that want a slice.
    pub fn to_vec(&self) -> Vec<Value> {
        self.iter().cloned().collect()
    }

    // What the walk's memo keys a compound value by: the block and the view's height (two
    // views of one block differ in `top`; clones are equal).
    pub fn identity(&self) -> (usize, usize) {
        match &self.block {
            Some(b) => (Rc::as_ptr(b) as usize, self.top),
            None => (0, 0),
        }
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.block.as_ref().map_or(0, |b| b.slots.len())
    }
}

impl FromIterator<Value> for List {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> List {
        List::from_vec(iter.into_iter().collect())
    }
}

// Logical order: a block's slots from `top - 1` down to 0, then the blocks below.
pub struct Iter<'a> {
    block: Option<&'a Block>,
    // Slots of the current block still to yield (the next is `slots[pos - 1]`).
    pos: usize,
    left: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<&'a Value> {
        loop {
            let b = self.block?;
            if self.pos > 0 {
                self.pos -= 1;
                self.left -= 1;
                return Some(b.slots[self.pos].get().expect("a view's slots are set"));
            }
            self.pos = b.below.top;
            self.block = b.below.block.as_deref();
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl ExactSizeIterator for Iter<'_> {}

// A chain of blocks (a list built by repeated branching) can be a million deep: drop it in a
// loop, the same pattern as cont.rs's and frame.rs's Drop, not by recursion.
impl Drop for Block {
    fn drop(&mut self) {
        let mut next = std::mem::take(&mut self.below).block;
        while let Some(rc) = next {
            match Rc::try_unwrap(rc) {
                Ok(mut b) => next = std::mem::take(&mut b.below).block,
                Err(_) => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn val(n: i64) -> Value {
        Value::Int(n)
    }

    fn ints(l: &List) -> Vec<i64> {
        l.iter().map(Value::as_int).collect()
    }

    fn bulk(xs: &[i64]) -> List {
        List::from_vec(xs.iter().map(|&n| val(n)).collect())
    }

    // 0..n built by repeated cons onto the empty list: blocks of 1, 2, 4, ...
    fn linear(n: i64) -> List {
        (0..n).fold(List::empty(), |l, i| l.cons(val(i)))
    }

    // Checks every observer of `l` against the model `want` (head-first).
    fn check(l: &List, want: &[i64]) {
        assert_eq!(l.len(), want.len());
        assert_eq!(l.is_empty(), want.is_empty());
        assert_eq!(ints(l), want);
        assert_eq!(l.iter().len(), want.len());
        assert_eq!(l.to_vec().iter().map(Value::as_int).collect::<Vec<_>>(), want);
        for (i, w) in want.iter().enumerate() {
            assert_eq!(l.get(i).map(Value::as_int), Some(*w), "get({i})");
        }
        assert!(l.get(want.len()).is_none());
        let (mut rest, mut seen) = (l.clone(), Vec::new());
        while let Some((h, t)) = rest.split_first() {
            seen.push(h.as_int());
            rest = t;
        }
        assert_eq!(seen, want);
        assert!(rest.is_empty());
    }

    #[test]
    fn from_vec_iter_to_vec_round_trip() {
        for n in [0i64, 1, 5, 1000] {
            let want: Vec<i64> = (0..n).collect();
            check(&bulk(&want), &want);
            let collected: List = want.iter().map(|&x| val(x)).collect();
            check(&collected, &want);
        }
    }

    #[test]
    fn get_agrees_with_the_model_on_every_kind_of_list() {
        check(&linear(100), &(0..100).rev().collect::<Vec<_>>());
        // Branch-built: every level conses onto an already-claimed position.
        let mut l = bulk(&[1]);
        let mut want = vec![1];
        for i in 0..50 {
            let _first = l.cons(val(1000 + i)); // wins the position
            l = l.cons(val(i)); // a branch: a fresh block over the same tail
            want.insert(0, i);
        }
        check(&l, &want);
        // Appended.
        let (a, b) = (linear(7), linear(11));
        let want: Vec<i64> = (0..7).rev().chain((0..11).rev()).collect();
        check(&a.append(&b), &want);
        // A tail view, then consed onto.
        let (_, t) = linear(10).split_first().unwrap();
        check(&t.cons(val(99)), &[99, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn claim_then_branch() {
        let base = bulk(&[1]);
        let a = base.cons(val(2));
        let b = base.cons(val(3));
        check(&base, &[1]);
        check(&a, &[2, 1]);
        check(&b, &[3, 1]);
        // A grown block: the first cons above a position claims it, the second branches.
        let xs = linear(3); // [2, 1, 0], top block capacity 1 (blocks of 1, 2: top 2 of capacity 2)
        let first = xs.cons(val(10));
        let second = xs.cons(val(20));
        check(&xs, &[2, 1, 0]);
        check(&first, &[10, 2, 1, 0]);
        check(&second, &[20, 2, 1, 0]);
        assert_ne!(first.identity().0, second.identity().0, "the second cons got its own block");
    }

    #[test]
    fn the_growth_claim_is_taken_once() {
        let full = linear(1); // one exact slot, unsealed
        assert_eq!(full.capacity(), 1);
        let first = full.cons(val(1));
        assert_eq!(first.capacity(), 2, "the first cons doubles");
        let second = full.cons(val(2));
        assert_eq!(second.capacity(), 1, "the second cons onto the same full block does not");
        check(&first, &[1, 0]);
        check(&second, &[2, 0]);
        // A linear build doubles: blocks 1, 2, 4, 8 hold 15.
        let l = linear(15);
        assert_eq!(l.capacity(), 8);
        let mut blocks = 0;
        let mut b = l.block.as_deref();
        while let Some(x) = b {
            blocks += 1;
            b = x.below.block.as_deref();
        }
        assert_eq!(blocks, 4);
    }

    #[test]
    fn bulk_blocks_are_sealed() {
        let l = bulk(&[1, 2, 3, 4]);
        assert_eq!(l.capacity(), 4);
        let c = l.cons(val(0));
        assert_eq!(c.capacity(), 1, "a cons onto a bulk list starts an exact block");
        check(&c, &[0, 1, 2, 3, 4]);
        // And so does a second one: no doubling anywhere.
        assert_eq!(l.cons(val(7)).capacity(), 1);
    }

    #[test]
    fn append_cases() {
        let (e, a, b) = (List::empty(), bulk(&[1, 2]), bulk(&[3, 4, 5]));
        check(&e.append(&e), &[]);
        check(&a.append(&e), &[1, 2]);
        check(&e.append(&b), &[3, 4, 5]);
        check(&a.append(&b), &[1, 2, 3, 4, 5]);
        // Empty sides are free: the very same block.
        assert_eq!(a.append(&e).identity(), a.identity());
        assert_eq!(e.append(&b).identity(), b.identity());
        // Neither operand changes.
        check(&a, &[1, 2]);
        check(&b, &[3, 4, 5]);
        // `[x] ++ acc` behaves like `x :: acc`: the loop makes log-many blocks.
        let mut acc = List::empty();
        for i in 0..1000 {
            acc = bulk(&[i]).append(&acc);
        }
        check(&acc, &(0..1000).rev().collect::<Vec<_>>());
        let mut blocks = 0;
        let mut b = acc.block.as_deref();
        while let Some(x) = b {
            blocks += 1;
            b = x.below.block.as_deref();
        }
        assert!(blocks <= 12, "blocks {blocks}");
        // `acc ++ [x]` is correct (and linear per step).
        let mut snoc = List::empty();
        for i in 0..50 {
            snoc = snoc.append(&bulk(&[i]));
        }
        check(&snoc, &(0..50).collect::<Vec<_>>());
    }

    #[test]
    fn identity_distinguishes_views_of_one_block() {
        let l = linear(2); // blocks of 1 and 2: top block holds [1]; add one more claim
        let longer = l.cons(val(5));
        let (_, shorter) = longer.split_first().unwrap();
        assert_eq!(longer.identity().0, shorter.identity().0, "one block");
        assert_ne!(longer.identity(), shorter.identity(), "two tops");
        assert_eq!(longer.identity(), longer.clone().identity());
    }

    #[test]
    fn sizes() {
        assert!(size_of::<List>() <= 24);
        assert_eq!(size_of::<OnceCell<Value>>(), size_of::<Value>(), "a slot costs one Value");
        assert_eq!(size_of::<Value>(), 40);
    }

    #[test]
    fn dropping_a_million_block_chain_does_not_overflow() {
        let mut l = bulk(&[0]);
        for i in 0..1_000_000 {
            let _winner = l.cons(val(i));
            l = l.cons(val(i)); // a branch each time: a chain of exact one-slot blocks
        }
        assert_eq!(l.len(), 1_000_001);
        drop(l);
    }

    // Random operations on a pool of lists against Vec models: a list's contents never
    // change whatever is consed onto it elsewhere.
    #[test]
    fn random_operations_never_disturb_other_lists() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self, n: usize) -> usize {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                (self.0 % n as u64) as usize
            }
        }
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let mut pool: Vec<(List, Vec<i64>)> = vec![(List::empty(), vec![]), (bulk(&[1, 2, 3]), vec![1, 2, 3])];
        let mut counter = 100;
        for _ in 0..4000 {
            let (l, m) = pool[r.next(pool.len())].clone();
            let new = match r.next(6) {
                0..=2 => {
                    counter += 1;
                    let mut m = m;
                    m.insert(0, counter);
                    (l.cons(val(counter)), m)
                }
                3 => match l.split_first() {
                    Some((_, t)) => (t, m[1..].to_vec()),
                    None => (l, m),
                },
                4 => {
                    let (o, om) = pool[r.next(pool.len())].clone();
                    (l.append(&o), m.into_iter().chain(om).collect())
                }
                _ => {
                    let n = r.next(4);
                    let xs: Vec<i64> = (0..n as i64).map(|i| 1000 + i).collect();
                    (l.append(&bulk(&xs)), m.into_iter().chain(xs).collect())
                }
            };
            if new.1.len() < 400 {
                pool.push(new);
            }
        }
        for (l, m) in &pool {
            check(l, m);
        }
    }
}
