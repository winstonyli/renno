// The list representation: a claim-list (a Bagwell VList whose growth slots are claimed by
// the first cons that reaches them).
//
// One structure whose two extremes are an array (one block, built in bulk) and a cons list
// (blocks of one slot). A block is a fixed-capacity array of write-once slots (the claim
// engine's storage, claimed.rs); a view `(block, lo, hi)` sees `slots[lo..hi]`, the head being
// `slots[hi - 1]` and the last element `slots[lo]`, and the rest of the list is whatever sits
// `below` the block. Consing onto a view whose next slot above is empty *claims* that slot,
// and appending a short list to a single-block view whose next slot below is empty claims
// those, so building by repeated cons or by `acc ++ [x]` is O(1) amortized and contiguous;
// the first extension at a position wins it, and every later one at the same position (a
// branch: `a = x :: xs`, `b = y :: xs`) starts a fresh block over the same tail. A slot is
// written at most once, so no view's contents ever change and the structure is persistent in
// safe Rust.
//
// Cost: `::` is amortized O(1) (one cons can open a block of twice the top block's capacity,
// each block grows at most once), head and tail are O(1); `++` is O(min(|a|, |b|)) when a
// claim succeeds (amortized O(1) for `acc ++ [x]`) and O(|a|) otherwise; `get` is O(1) on a
// bulk-built list (a literal, `range`, a builtin's result, a tuple) and O(blocks) on a
// cons-built one (O(log n) when built linearly, up to O(n) for a list built by repeated
// branching, as for a cons list).
//
// Memory: a view keeps alive every value claimed into a free slot of its block, above or
// below, including by a branch that was since dropped (see the spec, section 20, parked item
// M1).

use std::cell::{Cell, OnceCell};
use std::rc::Rc;

use crate::claimed::{Claimed, Side, Slots};
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
    // This view sees `block.cells.slots[lo..hi]` (head `slots[hi - 1]`, last `slots[lo]`), then
    // `block.below`. lo < hi whenever `block` is Some; lo > 0 only in a block whose `below` is
    // empty.
    lo: u32,
    hi: u32,
    // (hi - lo) + below.len, cached.
    len: usize,
}

struct Block {
    // The slots (one written interval, containing every view) and the growth claim, set by
    // the first copy that reserves room past this block and born set for a bulk block, so a
    // second one never reserves a large block (claimed.rs).
    cells: Claimed<Cells>,
    // The list under the slots.
    below: List,
    // What a check last proved about this block (see `Mark`).
    seal: Cell<Mark>,
}

type Cells = Box<[OnceCell<Value>]>;

// The identity of a test as a seal records it: assigned by expr.rs's interner, never 0.
pub type SealId = std::num::NonZeroU32;

// One fact about immutable data: the view `(block, lo..hi)` (its slots `[lo..hi]`, then
// everything below the block) passes the test `#id` (id 0: nothing recorded). A slot is set
// at most once and `below` never changes, so the fact stays true for the block's whole life
// whoever wrote it: a seal is only ever overwritten by another true fact (last writer wins),
// never invalidated. For an "every element passes" test the fact is closed under sub-views
// inside the interval; any other test's whole seal is read only at exactly this interval
// (`has_whole_seal`).
#[derive(Clone, Copy, Default)]
struct Mark {
    id: u32,
    lo: u32,
    hi: u32,
}

impl Mark {
    fn covers(self, id: u32, lo: u32, hi: u32) -> bool {
        self.id == id && self.lo <= lo && hi <= self.hi
    }
}

// A block index as a view stores it. Unreachable in practice (over 160 GB of Values), but a
// run error rather than a silent wrap.
fn idx(x: usize) -> u32 {
    u32::try_from(x).unwrap_or_else(|_| crate::run_error!("a list block over {} elements", u32::MAX))
}

// The seal a whole seal of `id` at the view `[lo, hi)` writes over `old`, or None: write
// nothing. Nothing when `old` already covers at least this view (an every-element seal written
// by the same walk, which a whole seal would clobber: a tail of the list is then checked at its
// own `Ref` against `old`, O(1), instead of re-scanned); `old`'s id does not matter.
fn whole_mark(old: Mark, id: SealId, lo: u32, hi: u32) -> Option<Mark> {
    (old.id == 0 || !(old.lo <= lo && hi <= old.hi)).then_some(Mark { id: id.get(), lo, hi })
}

// Views are made three ways, each preserving "every view (b, lo, hi) has b.slots[lo..hi] set":
// a bulk build sets every slot at creation; a claim past the written interval's frontier
// (`hi` or `lo`) widens the interval and makes the view that claimed it longer; a tail makes
// `hi - 1`, and at `hi - lo == 1` returns `below`.
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

    // The bulk builds: one exactly-sized, sealed block, made in one pass over the elements
    // (every slot is created already set).
    //
    // From a head-first Vec: the Vec is reversed in place and its allocation becomes the
    // block's storage (a Value and an OnceCell<Value> have the same layout).
    pub fn from_vec(mut items: Vec<Value>) -> List {
        if items.is_empty() {
            return List::empty();
        }
        items.reverse();
        let cells: Vec<OnceCell<Value>> = items.into_iter().map(OnceCell::from).collect();
        let n = cells.len();
        List::block(List::empty(), Claimed::new(cells.into_boxed_slice(), true), 0, n)
    }

    // From elements in bottom-first order, i.e. the last element of the list first: what a
    // producer that can walk backwards (`range`, `reverse`) gives without a reversing pass.
    pub fn from_bottom_up(items: impl Iterator<Item = Value>) -> List {
        let mut cells = Vec::with_capacity(items.size_hint().0);
        items.for_each(|x| cells.push(OnceCell::from(x)));
        let n = cells.len();
        if n == 0 {
            return List::empty();
        }
        List::block(List::empty(), Claimed::new(cells.into_boxed_slice(), true), 0, n)
    }

    // The one place a block is made: `cells`, written in `[lo, hi)`, over `below`.
    fn block(below: List, cells: Claimed<Cells>, lo: usize, hi: usize) -> List {
        debug_assert!(lo == 0 || below.is_empty(), "internal: room below only in a single block");
        let cap = cells.slots.len();
        idx(cap);
        count_writes(hi - lo);
        count_block();
        count_empties(cap - (hi - lo));
        List { len: below.len + (hi - lo), lo: lo as u32, hi: hi as u32, block: Some(Rc::new(Block { cells, below, seal: Cell::default() })) }
    }

    pub fn cons(&self, head: Value) -> List {
        self.push_all(std::iter::once(head))
    }

    // The one growth primitive: `items` (bottom-first) go on top of this list.
    //   1. Claim: fill the free slots above the view, as many as fit.
    //   2. A new block for what is left (below = the view so far). Its capacity is
    //      `max(left, 2 * (hi - lo))` when the view sits at the top of a full block and this
    //      call wins that block's growth claim, otherwise exactly `left`.
    // So a list built by repeated cons grows in blocks of 1, 2, 4, ...; a cons onto a branch
    // point, a bulk list or a tail view gets an exact block.
    pub fn push_all(&self, mut items: impl ExactSizeIterator<Item = Value>) -> List {
        let mut claimed = self.clone();
        if let Some(b) = &self.block {
            let n = b.cells.claim_some(self.hi as usize, &mut items);
            count_writes(n);
            claimed.hi += n as u32;
            claimed.len += n;
        }
        let left = items.len();
        if left == 0 {
            return claimed;
        }
        let capacity = match &claimed.block {
            Some(b) if b.cells.should_grow(Side::High, claimed.hi as usize) => left.max(2 * (claimed.hi - claimed.lo) as usize),
            _ => left,
        };
        let cells = Claimed::new(Cells::with_room(0, left, items, capacity - left), false);
        List::block(claimed, cells, 0, left)
    }

    // `self ++ b`: O(1) when either side is empty. Otherwise the shorter side is written: when
    // `b` is shorter and `self` is a view that can be extended in place or copied with room, `b`
    // is claimed under `self` (`snoc_all`, amortized O(1) for `acc ++ [x]`); in every other
    // case `self`'s elements are consed on top of `b`, which is never copied (a `[x] ++ acc`
    // loop behaves exactly like `x :: acc`, and is O(|self|)).
    pub fn append(&self, b: &List) -> List {
        if self.is_empty() {
            return b.clone();
        }
        if b.is_empty() {
            return self.clone();
        }
        if b.len < self.len {
            if let Some(l) = self.snoc_all(b) {
                return l;
            }
        }
        b.push_all(self.iter_rev().cloned())
    }

    // `self ++ items` by the low end of a block, or None when this view cannot take it:
    //   1. Claim: a single-block view whose low neighbour slots are free (the engine refuses a
    //      branch, whose neighbour is taken) takes `items` in place.
    //   2. Copy-grow: a view at its low frontier that wins the block's growth claim is copied
    //      into a fresh single block with as much room again below (amortized O(1) per later
    //      claim).
    // Otherwise None: a branch point pays the front path's cost and never reserves a block.
    fn snoc_all(&self, items: &List) -> Option<List> {
        let blk = self.block.as_ref()?;
        let (lo, n) = (self.lo as usize, items.len);
        // Only a single-block view snocs in place: an element under `slots[lo]` would land
        // between the block and `below`.
        if blk.below.is_empty() && blk.cells.claim(Side::Low, lo, items.iter().cloned()) {
            count_writes(n);
            return Some(List { block: self.block.clone(), lo: (lo - n) as u32, hi: self.hi, len: self.len + n });
        }
        if !blk.cells.should_grow(Side::Low, lo) {
            return None;
        }
        let total = self.len + n;
        let cells = Claimed::<Cells>::copy(Side::Low, total, items.iter_rev().chain(self.iter_rev()).cloned(), true);
        Some(List::block(List::empty(), cells, total, 2 * total))
    }

    pub fn first(&self) -> Option<&Value> {
        let b = self.block.as_ref()?;
        Some(b.cells.slots[self.hi as usize - 1].get().expect("a view's slots are set"))
    }

    // The head and the rest, in O(1).
    pub fn split_first(&self) -> Option<(&Value, List)> {
        let head = self.first()?;
        let b = self.block.as_ref()?;
        let tail = if self.hi - self.lo == 1 { b.below.clone() } else { List { block: self.block.clone(), lo: self.lo, hi: self.hi - 1, len: self.len - 1 } };
        Some((head, tail))
    }

    // Drops the head in place: the cursor step of an iteration (no reference-count traffic
    // except when the view crosses into the block below).
    pub fn pop_front(&mut self) {
        let Some(b) = &self.block else { return };
        self.len -= 1;
        if self.hi - self.lo == 1 {
            *self = b.below.clone();
        } else {
            self.hi -= 1;
        }
    }

    // The i-th element: O(1) within the top block, one more step per block below it. There
    // is deliberately no `Index` impl, so the O(blocks) cost stays visible at call sites.
    pub fn get(&self, i: usize) -> Option<&Value> {
        if i >= self.len {
            return None;
        }
        let (mut block, mut lo, mut hi, mut i) = (self.block.as_deref(), self.lo as usize, self.hi as usize, i);
        loop {
            let b = block?;
            count_hop();
            if i < hi - lo {
                return b.cells.slots[hi - 1 - i].get();
            }
            i -= hi - lo;
            (lo, hi) = (b.below.lo as usize, b.below.hi as usize);
            block = b.below.block.as_deref();
        }
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter { block: self.block.as_deref(), pos: self.hi as usize, floor: self.lo as usize, left: self.len }
    }

    // The elements last to first, which is a block's slots in ascending order: no copy.
    pub fn iter_rev(&self) -> RevIter<'_> {
        let (mut segs, mut view) = (Vec::new(), self);
        while let Some(b) = &view.block {
            segs.push(&b.cells.slots[view.lo as usize..view.hi as usize]);
            view = &b.below;
        }
        RevIter { cur: segs.pop().unwrap_or_default().iter(), segs, left: self.len }
    }

    // A copy of the elements in order: for the cold paths that want a slice.
    pub fn to_vec(&self) -> Vec<Value> {
        let mut v = Vec::with_capacity(self.len);
        self.iter().for_each(|x| v.push(x.clone()));
        v
    }

    // What the walk's memo keys a compound value by: the block and the view's interval (two
    // views of one block differ in `lo` or `hi`; clones are equal).
    pub fn identity(&self) -> (usize, u64) {
        match &self.block {
            Some(b) => (Rc::as_ptr(b) as usize, (u64::from(self.lo) << 32) | u64::from(self.hi)),
            None => (0, 0),
        }
    }

    // ---- Seals: what a check does not need to test again ----

    // The elements, head first, that an every-element test `id` has not yet been proved for:
    // in a block whose seal is `id` only the slots outside the sealed interval, and nothing
    // below it (the seal covers everything under the block); in any other block every slot,
    // then the view below.
    pub fn unsealed(&self, id: SealId) -> Unsealed<'_> {
        let mut it = Unsealed { id: id.get(), block: None, pos: 0, floor: 0, low: (0, 0), more: false };
        it.enter(self.block.as_deref(), self.lo as usize, self.hi as usize);
        it
    }

    // Does an every-element test `id` already hold for this whole list? The same as
    // `unsealed` being empty, in O(1).
    pub fn is_sealed_every(&self, id: SealId) -> bool {
        match &self.block {
            None => true,
            Some(b) => b.seal.get().covers(id.get(), self.lo, self.hi),
        }
    }

    // Records that every element of this view passes the test `id`, after a check passed
    // them all. Each block of the chain is written top-down, at the interval of the view the
    // chain passes through (this view's, then each block below's), until a block already
    // sealed for `id` is reached: that seal also asserts everything below, so the two
    // intervals are unioned when they overlap or touch (every slot in the union passed), and
    // the view's own interval replaces a disjoint one (the gap is unproved); then stop.
    pub fn seal_every(&self, id: SealId) {
        let (id, mut view) = (id.get(), self);
        while let Some(b) = &view.block {
            let (s, (lo, hi)) = (b.seal.get(), (view.lo, view.hi));
            if s.id == id {
                let m = if s.lo <= hi && lo <= s.hi { Mark { id, lo: s.lo.min(lo), hi: s.hi.max(hi) } } else { Mark { id, lo, hi } };
                b.seal.set(m);
                return;
            }
            b.seal.set(Mark { id, lo, hi });
            view = &b.below;
        }
    }

    // A whole-value seal (a test that is not closed under tails): true only for exactly this
    // view.
    pub fn has_whole_seal(&self, id: SealId) -> bool {
        self.block.as_ref().is_some_and(|b| {
            let s = b.seal.get();
            s.id == id.get() && s.lo == self.lo && s.hi == self.hi
        })
    }

    // Records the whole seal, unless that would lose a fact (`whole_mark`).
    pub fn seal_whole(&self, id: SealId) {
        if let Some(b) = &self.block {
            if let Some(mark) = whole_mark(b.seal.get(), id, self.lo, self.hi) {
                b.seal.set(mark);
            }
        }
    }

    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.block.as_ref().map_or(0, |b| b.cells.slots.len())
    }
}

impl FromIterator<Value> for List {
    fn from_iter<I: IntoIterator<Item = Value>>(iter: I) -> List {
        List::from_vec(iter.into_iter().collect())
    }
}

// Logical order: a block's slots from `hi - 1` down to `lo`, then the blocks below.
pub struct Iter<'a> {
    block: Option<&'a Block>,
    // Slots of the current block still to yield are `slots[floor..pos]`, the next being
    // `slots[pos - 1]`.
    pos: usize,
    floor: usize,
    left: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<&'a Value> {
        loop {
            let b = self.block?;
            if self.pos > self.floor {
                self.pos -= 1;
                self.left -= 1;
                return Some(b.cells.slots[self.pos].get().expect("a view's slots are set"));
            }
            self.pos = b.below.hi as usize;
            self.floor = b.below.lo as usize;
            self.block = b.below.block.as_deref();
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }

    // Internal iteration (what `for_each` and the bulk builders use): one tight loop per
    // block, with none of `next`'s per-element block check.
    fn fold<B, F: FnMut(B, &'a Value) -> B>(mut self, init: B, mut f: F) -> B {
        let mut acc = init;
        while let Some(b) = self.block {
            for cell in b.cells.slots[self.floor..self.pos].iter().rev() {
                acc = f(acc, cell.get().expect("a view's slots are set"));
            }
            self.pos = b.below.hi as usize;
            self.floor = b.below.lo as usize;
            self.block = b.below.block.as_deref();
        }
        acc
    }
}

impl ExactSizeIterator for Iter<'_> {}

// The elements an every-element test has not been proved for, head first (`List::unsealed`).
pub struct Unsealed<'a> {
    id: u32,
    block: Option<&'a Block>,
    // The slots still to yield in the current block are `slots[floor..pos]`, the next being
    // `slots[pos - 1]`; then `slots[low.0..low.1]` (the part under a seal's interval).
    pos: usize,
    floor: usize,
    low: (usize, usize),
    // Whether the view below the current block is still to be visited.
    more: bool,
}

impl<'a> Unsealed<'a> {
    // Positions on the view `(block, lo..hi)`.
    fn enter(&mut self, block: Option<&'a Block>, lo: usize, hi: usize) {
        self.block = block;
        self.pos = hi;
        let Some(b) = block else { return };
        let s = b.seal.get();
        if s.id == self.id {
            let clamp = |x: u32| (x as usize).clamp(lo, hi);
            self.floor = clamp(s.hi);
            self.low = (lo, clamp(s.lo));
            self.more = false;
        } else {
            self.floor = lo;
            self.low = (0, 0);
            self.more = true;
        }
    }
}

impl<'a> Iterator for Unsealed<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<&'a Value> {
        loop {
            let b = self.block?;
            if self.pos > self.floor {
                self.pos -= 1;
                return Some(b.cells.slots[self.pos].get().expect("a view's slots are set"));
            }
            if self.low.1 > self.low.0 {
                (self.floor, self.pos) = self.low;
                self.low = (0, 0);
                continue;
            }
            if !self.more {
                self.block = None;
                return None;
            }
            self.enter(b.below.block.as_deref(), b.below.lo as usize, b.below.hi as usize);
        }
    }
}

// Bottom-first: the block chain's slices, deepest block first.
pub struct RevIter<'a> {
    cur: std::slice::Iter<'a, OnceCell<Value>>,
    // The slices still to visit, the next one last.
    segs: Vec<&'a [OnceCell<Value>]>,
    left: usize,
}

impl<'a> Iterator for RevIter<'a> {
    type Item = &'a Value;

    fn next(&mut self) -> Option<&'a Value> {
        loop {
            if let Some(cell) = self.cur.next() {
                self.left -= 1;
                return Some(cell.get().expect("a view's slots are set"));
            }
            self.cur = self.segs.pop()?.iter();
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }

    fn fold<B, F: FnMut(B, &'a Value) -> B>(mut self, init: B, mut f: F) -> B {
        let mut acc = init;
        for cell in self.cur.by_ref() {
            acc = f(acc, cell.get().expect("a view's slots are set"));
        }
        while let Some(seg) = self.segs.pop() {
            for cell in seg {
                acc = f(acc, cell.get().expect("a view's slots are set"));
            }
        }
        acc
    }
}

impl ExactSizeIterator for RevIter<'_> {}

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
        // pop_front is the same step, in place.
        let (mut cursor, mut seen) = (l.clone(), Vec::new());
        while let Some(h) = cursor.first() {
            seen.push(h.as_int());
            cursor.pop_front();
            assert_eq!(cursor.len(), want.len() - seen.len());
        }
        assert_eq!(seen, want);
        cursor.pop_front();
        assert!(cursor.is_empty());
    }

    // The bulk builds create every slot already set (no empty slot) and one block each.
    #[test]
    fn bulk_builds_are_one_pass_and_agree() {
        for n in [0i64, 1, 2, 7, 1000] {
            let want: Vec<i64> = (0..n).collect();
            let rev: Vec<i64> = want.iter().rev().copied().collect();
            reset_counters();
            let a = List::from_vec(want.iter().map(|&x| val(x)).collect());
            let b = List::from_bottom_up((0..n).rev().map(val));
            let c = a.append(&b);
            let d = List::from_bottom_up(a.iter().cloned());
            assert_eq!(empties(), 0, "n = {n}");
            check(&a, &want);
            check(&b, &want);
            check(&d, &rev);
            let both: Vec<i64> = want.iter().chain(&want).copied().collect();
            check(&c, &both);
            assert_eq!(a.capacity(), want.len());
            assert_eq!(d.capacity(), want.len());
        }
    }

    // iter_rev walks a many-block list (cons-built, branched, appended) bottom-first.
    #[test]
    fn iter_rev_is_the_reverse_of_iter() {
        let l = linear(100);
        let branched = l.cons(val(-1));
        let joined = bulk(&[7, 8, 9]).append(&branched);
        for x in [&List::empty(), &l, &branched, &joined, &l.split_first().unwrap().1] {
            let mut want = ints(x);
            want.reverse();
            assert_eq!(x.iter_rev().len(), want.len());
            assert_eq!(x.iter_rev().map(Value::as_int).collect::<Vec<_>>(), want);
        }
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
        // A block with a free slot (linear(2) is a 2-slot block holding one element): the
        // first cons above that position claims the slot in place, the second branches.
        let xs = linear(2);
        assert_eq!((xs.capacity(), xs.len()), (2, 2));
        let first = xs.cons(val(10));
        let second = xs.cons(val(20));
        check(&xs, &[1, 0]);
        check(&first, &[10, 1, 0]);
        check(&second, &[20, 1, 0]);
        assert_eq!(first.identity().0, xs.identity().0, "the first cons claimed the free slot");
        assert_ne!(second.identity().0, xs.identity().0, "the second cons got its own block");
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
        // The bottom-up build (range, reverse) is sealed the same way.
        let r = List::from_bottom_up((1..=4).rev().map(val));
        assert_eq!(r.cons(val(0)).capacity(), 1);
        assert_eq!(r.cons(val(7)).capacity(), 1);
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
        assert!(size_of::<Block>() <= 64);
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
            let new = match r.next(8) {
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

    // ---- Snoc claims (V2) ----

    // `[1, 2, 3] ++ [4] ++ [5]` as the machine builds it: a front-path chain, then one copy into a
    // 10-slot block with room below: the elements sit in slots [5, 10).
    fn roomy() -> List {
        bulk(&[1, 2, 3]).append(&bulk(&[4])).append(&bulk(&[5]))
    }

    #[test]
    fn a_snoc_claims_the_low_end_then_branches() {
        let base = roomy();
        check(&base, &[1, 2, 3, 4, 5]);
        assert_eq!((base.capacity(), base.lo, base.hi), (10, 5, 10));
        let a = base.append(&bulk(&[6]));
        let b = base.append(&bulk(&[7]));
        check(&base, &[1, 2, 3, 4, 5]);
        check(&a, &[1, 2, 3, 4, 5, 6]);
        check(&b, &[1, 2, 3, 4, 5, 7]);
        assert_eq!(a.identity().0, base.identity().0, "the first snoc claimed a slot");
        assert_ne!(b.identity().0, base.identity().0, "the second snoc at that position branched");
        // A two-element snoc claims both slots.
        let c = a.append(&bulk(&[8, 9]));
        check(&c, &[1, 2, 3, 4, 5, 6, 8, 9]);
        assert_eq!((c.identity().0, c.lo), (base.identity().0, 2));
        // Cons, head and tail on a snoc-built view stay O(1) and correct.
        check(&c.cons(val(0)), &[0, 1, 2, 3, 4, 5, 6, 8, 9]);
        let (h, t) = c.split_first().unwrap();
        assert_eq!(h.as_int(), 1);
        check(&t, &[2, 3, 4, 5, 6, 8, 9]);
        let u = t.append(&bulk(&[10]));
        check(&u, &[2, 3, 4, 5, 6, 8, 9, 10]);
        assert_eq!(u.identity().0, base.identity().0, "a tail view at the frontier claims too");
        check(&c, &[1, 2, 3, 4, 5, 6, 8, 9]);
        // Two views that differ only in `lo` have different identities.
        assert_ne!(a.identity(), base.identity());
        assert_eq!(a.identity(), a.clone().identity());
    }

    // The engine's growth rule (0A.1): only a view at the frontier grows. `b` is a branch, so
    // it takes the exact front path: [7]'s bulk block, then an exact 5-slot block over it.
    #[test]
    fn a_branched_snoc_never_grows() {
        let base = roomy();
        let _a = base.append(&bulk(&[6]));
        let b = base.append(&bulk(&[7]));
        check(&b, &[1, 2, 3, 4, 5, 7]);
        assert_eq!(b.capacity(), 5);
    }

    #[test]
    fn snoc_growth_doubles_once_per_block() {
        reset_counters();
        let mut l = List::empty();
        for i in 0..100 {
            l = l.append(&bulk(&[i]));
        }
        let (writes, blocks, _) = counters();
        // 100 literals, 1 front-path write, copies of 3 + 7 + 15 + 31 + 63 = 119, 93 claims.
        assert_eq!((writes, blocks), (100 + 1 + 119 + 93, 100 + 1 + 5));
        assert_eq!(empties(), 3 + 7 + 15 + 31 + 63);
        check(&l, &(0..100).collect::<Vec<_>>());
    }

    #[test]
    fn snoc_growth_claim_is_taken_once() {
        // Fill roomy's room (5 claims), so the view sits at the block's low edge.
        let mut edge = roomy();
        for i in 6..=10 {
            edge = edge.append(&bulk(&[i]));
        }
        assert_eq!((edge.capacity(), edge.lo), (10, 0));
        let first = edge.append(&bulk(&[11]));
        assert_eq!(first.capacity(), 22, "the first snoc past the edge copies into a doubled block");
        let second = edge.append(&bulk(&[12]));
        assert_eq!(second.capacity(), 10, "the second takes the front path: an exact block");
        check(&first, &(1..=11).collect::<Vec<_>>());
        check(&second, &(1..=10).chain([12]).collect::<Vec<_>>());
        // A bulk list is born grown: its first snoc takes the front path too.
        assert_eq!(bulk(&[1, 2, 3]).append(&bulk(&[4])).capacity(), 3);
        // Equal lengths take the front path (no empty slot, as today).
        reset_counters();
        check(&bulk(&[1, 2]).append(&bulk(&[3, 4])), &[1, 2, 3, 4]);
        assert_eq!(empties(), 0);
    }

    #[test]
    fn a_snoc_claimed_slot_below_a_seal_is_unsealed() {
        let l = roomy();
        l.seal_every(id(1));
        let a = l.append(&bulk(&[6]));
        assert_eq!(a.identity().0, l.identity().0);
        assert_eq!(unsealed(&a, 1), [6]);
        assert!(!a.is_sealed_every(id(1)));
        a.seal_every(id(1));
        for t in tails(&a).iter().chain(tails(&l).iter()) {
            assert!(unsealed(t, 1).is_empty() && t.is_sealed_every(id(1)), "{:?}", ints(t));
        }
        // A snoc and a cons on the same sealed view: each sees only its own new slot.
        let b = a.append(&bulk(&[7]));
        let c = a.cons(val(0));
        assert_eq!(unsealed(&b, 1), [7]);
        assert_eq!(unsealed(&c, 1), [0]);
    }

    // l = [1..5] in slots [5, 10); a = [4, 5, 6] in [4, 7); b = [4, 5, 6, 7] in [3, 7);
    // only6 = [6] in [4, 5); only7 = [7] in [3, 4).
    fn intervals() -> (List, List, List, List, List) {
        let l = roomy();
        let mut short = l.clone();
        for _ in 0..3 {
            short.pop_front();
        }
        let a = short.append(&bulk(&[6]));
        let b = a.append(&bulk(&[7]));
        let last = |v: &List| tails(v).into_iter().find(|t| t.len() == 1).unwrap();
        let (only6, only7) = (last(&a), last(&b));
        assert_eq!((only6.lo, only6.hi, only7.lo, only7.hi), (4, 5, 3, 4));
        (l, a, b, only6, only7)
    }

    #[test]
    fn touching_seal_intervals_union() {
        let (l, a, b, only6, _) = intervals();
        l.seal_every(id(1));
        only6.seal_every(id(1)); // [4, 5) touches [5, 10): one seal [4, 10)
        assert!(l.is_sealed_every(id(1)) && a.is_sealed_every(id(1)) && only6.is_sealed_every(id(1)));
        assert_eq!(unsealed(&b, 1), [7]);
    }

    #[test]
    fn disjoint_seal_intervals_are_replaced_not_merged() {
        let (l, _, b, _, only7) = intervals();
        l.seal_every(id(1));
        only7.seal_every(id(1)); // [3, 4) and [5, 10) leave slot 4 unproved: the view's own interval wins
        assert!(only7.is_sealed_every(id(1)));
        assert_eq!(unsealed(&l, 1), ints(&l), "the replaced interval is simply unknown again");
        assert_eq!(unsealed(&b, 1), [4, 5, 6], "upper range of b above [3, 4)");
        assert!(!b.is_sealed_every(id(1)));
    }

    // ---- Seals ----

    fn id(n: u32) -> SealId {
        SealId::new(n).expect("a seal id is not 0")
    }

    fn unsealed(l: &List, n: u32) -> Vec<i64> {
        l.unsealed(id(n)).map(Value::as_int).collect()
    }

    // The view and every tail of it, as `split_first` walks them.
    fn tails(l: &List) -> Vec<List> {
        let (mut rest, mut out) = (l.clone(), vec![l.clone()]);
        while let Some((_, t)) = rest.split_first() {
            out.push(t.clone());
            rest = t;
        }
        out
    }

    #[test]
    fn an_unsealed_list_has_every_element_unsealed() {
        for l in [bulk(&[1, 2, 3]), linear(10), List::empty()] {
            assert_eq!(unsealed(&l, 1), ints(&l));
            assert_eq!(l.is_sealed_every(id(1)), l.is_empty());
        }
    }

    #[test]
    fn a_seal_covers_the_view_and_every_tail() {
        for l in [bulk(&[1, 2, 3]), linear(1), linear(100), bulk(&[]).append(&linear(40))] {
            l.seal_every(id(1));
            for t in tails(&l) {
                assert!(unsealed(&t, 1).is_empty(), "{:?}", ints(&t));
                assert!(t.is_sealed_every(id(1)));
            }
        }
    }

    // Blocks below the top one are sealed too (a cons-built list is several blocks).
    #[test]
    fn a_seal_reaches_every_block_of_the_chain() {
        let l = linear(100);
        l.seal_every(id(1));
        let (mut view, mut blocks) = (&l, 0);
        while let Some(b) = &view.block {
            assert_eq!(b.seal.get().id, 1, "block {blocks} of the chain");
            view = &b.below;
            blocks += 1;
        }
        assert!(blocks >= 5);
    }

    // A cons that claims a free slot above a sealed view leaves exactly that slot unsealed.
    #[test]
    fn a_claimed_slot_above_a_seal_is_unsealed() {
        let l = linear(5);
        assert!(l.capacity() > l.hi as usize,"the top block has a free slot");
        l.seal_every(id(1));
        let a = l.cons(val(100));
        assert_eq!(a.identity().0, l.identity().0, "the cons claimed a slot");
        assert_eq!(unsealed(&a, 1), [100]);
        // A second cons at the same position opens a block over the sealed view: also one slot.
        let b = l.cons(val(200));
        assert_ne!(b.identity().0, l.identity().0);
        assert_eq!(unsealed(&b, 1), [200]);
        // The first view is untouched by either.
        assert!(unsealed(&l, 1).is_empty());
        // Sealing the claimed view raises the height and covers it.
        a.seal_every(id(1));
        assert!(unsealed(&a, 1).is_empty() && unsealed(&a.cons(val(300)), 1) == [300]);
        assert_eq!(unsealed(&b, 1), [200]);
    }

    #[test]
    fn a_new_block_over_a_sealed_bulk_block_has_its_own_slots_unsealed() {
        let b = bulk(&[7, 7, 7]);
        b.seal_every(id(1));
        // `a ++ b` puts a's elements on top of b: a full sealed block, so a new block opens.
        let l = bulk(&[1, 2]).append(&b);
        assert_eq!(unsealed(&l, 1), [1, 2]);
        assert!(unsealed(&b, 1).is_empty());
    }

    // Two element tests on the same blocks: the seal holds one at a time (a different id
    // overwrites it with its own true fact), and neither sees the other's coverage.
    #[test]
    fn a_seal_for_one_id_says_nothing_about_another() {
        let l = linear(20);
        l.seal_every(id(1));
        assert_eq!(unsealed(&l, 2), ints(&l));
        l.seal_every(id(2));
        assert!(unsealed(&l, 2).is_empty());
        assert_eq!(unsealed(&l, 1), ints(&l), "overwritten: id 1 is simply unknown again");
        l.seal_every(id(1));
        assert!(unsealed(&l, 1).is_empty());
    }

    // A lower recorded height is never raised by a lower view, and a higher view raises it.
    #[test]
    fn sealing_a_shorter_view_never_lowers_the_coverage() {
        let l = linear(3);
        let top = l.clone();
        let mid = l.split_first().map(|(_, t)| t).unwrap();
        top.seal_every(id(1));
        mid.seal_every(id(1));
        assert!(unsealed(&top, 1).is_empty(), "the shorter view's seal must not shrink the longer one's");
        // In the other order the longer view raises the height.
        let l = linear(3);
        let mid = l.split_first().map(|(_, t)| t).unwrap();
        mid.seal_every(id(1));
        assert_eq!(unsealed(&l, 1), [2]);
        l.seal_every(id(1));
        assert!(unsealed(&l, 1).is_empty());
    }

    #[test]
    fn a_whole_seal_holds_only_at_its_exact_height() {
        let l = linear(3);
        let short = l.split_first().map(|(_, t)| t).unwrap();
        assert!(!l.has_whole_seal(id(1)));
        l.seal_whole(id(1));
        assert!(l.has_whole_seal(id(1)) && !short.has_whole_seal(id(1)) && !l.has_whole_seal(id(2)));
        short.seal_whole(id(1));
        // F3: a seal covering at least this view is never overwritten, so the short view's whole
        // seal is not written (the block keeps the taller one).
        assert!(!short.has_whole_seal(id(1)) && l.has_whole_seal(id(1)), "a covering seal is kept");
        // A taller view does replace a shorter one's.
        let tall = bulk(&[1, 2, 3, 4]);
        let low = tall.split_first().map(|(_, t)| t).unwrap();
        low.seal_whole(id(1));
        assert!(low.has_whole_seal(id(1)));
        tall.seal_whole(id(1));
        assert!(tall.has_whole_seal(id(1)) && !low.has_whole_seal(id(1)));
        // A whole seal and an every-element seal never mix (different ids), but they share the cell.
        assert_eq!(unsealed(&l, 2), ints(&l));
    }

    // A whole seal is exact-view only, and never written over a seal covering at least this
    // view (the every-element seal the same walk just wrote).
    #[test]
    fn a_whole_seal_never_clobbers_a_covering_seal() {
        let w = id(7);
        assert_eq!(whole_mark(Mark::default(), w, 0, 5).map(|m| (m.id, m.lo, m.hi)), Some((7, 0, 5)));
        let every = Mark { id: 3, lo: 2, hi: 5 };
        assert!(whole_mark(every, w, 2, 5).is_none() && whole_mark(every, w, 3, 4).is_none());
        assert_eq!(whole_mark(every, w, 1, 5).map(|m| (m.id, m.lo, m.hi)), Some((7, 1, 5)));
        assert_eq!(whole_mark(every, w, 2, 6).map(|m| m.id), Some(7));
        // Through the list: the every-element seal survives a whole seal at the same view.
        let l = bulk(&[1, 2, 3]);
        l.seal_every(id(3));
        l.seal_whole(id(7));
        assert!(l.is_sealed_every(id(3)) && !l.has_whole_seal(id(7)));
        // A whole seal is exact on both ends: a snoc-extended view does not inherit it.
        let r = roomy();
        let a = r.append(&bulk(&[6]));
        r.seal_whole(id(7));
        assert!(r.has_whole_seal(id(7)) && !a.has_whole_seal(id(7)));
    }

    // The model: the element addresses some earlier `seal_every(id)` call covered. `unsealed`
    // may only skip an element the model covers, its answer is a head-first ordered subset
    // of the view, and the call just made leaves its own view fully sealed.
    #[test]
    fn unsealed_never_skips_an_element_no_earlier_seal_covered() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self, n: usize) -> usize {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                (self.0 % n as u64) as usize
            }
        }
        let addr = |x: &Value| x as *const Value as usize;
        let mut r = Rng(0x2545_F491_4F6C_DD1D);
        let mut pool: Vec<List> = vec![List::empty(), bulk(&[1, 2, 3]), linear(9)];
        let mut covered: [std::collections::HashSet<usize>; 4] = Default::default();
        let mut counter = 100;
        let mut graveyard: Vec<List> = Vec::new();
        for _ in 0..3000 {
            let l = pool[r.next(pool.len())].clone();
            match r.next(9) {
                0..=2 => {
                    counter += 1;
                    pool.push(l.cons(val(counter)));
                }
                3 => pool.extend(l.split_first().map(|(_, t)| t)),
                4 => pool.push(l.append(&pool[r.next(pool.len())].clone())),
                5 | 6 => {
                    let n = 1 + r.next(3) as u32;
                    l.seal_every(id(n));
                    covered[n as usize].extend(l.iter().map(addr));
                    // Only the view itself: a tail that enters a lower block directly sees that
                    // block's own seal, which another id may have overwritten since (sound,
                    // and the documented single-entry thrash).
                    assert!(l.unsealed(id(n)).next().is_none() && l.is_sealed_every(id(n)));
                }
                _ => {
                    counter += 1;
                    pool.push(l.append(&bulk(&[counter])));
                }
            }
            if pool.len() > 600 {
                // Kept alive: a freed block's addresses could be reused and a stale address
                // would then count as covered.
                graveyard.push(pool.swap_remove(r.next(pool.len())));
            }
            let (probe, n) = (&pool[r.next(pool.len())], 1 + r.next(3) as u32);
            let got: Vec<usize> = probe.unsealed(id(n)).map(addr).collect();
            let all: Vec<usize> = probe.iter().map(addr).collect();
            let mut at = 0;
            for a in &got {
                at += all[at..].iter().position(|x| x == a).expect("an unsealed element is an element of the view, in order") + 1;
            }
            for a in &all {
                assert!(got.contains(a) || covered[n as usize].contains(a), "an element no seal covered was skipped");
            }
            assert_eq!(probe.is_sealed_every(id(n)), got.is_empty());
        }
    }
}
