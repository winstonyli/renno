// The claim engine shared by lists (list.rs) and strings (text.rs). A storage's slots are
// written at most once, and the written slots always form one interval. A view sees a
// sub-interval and is never changed. Extending a view whose edge is the interval's frontier
// writes the free slots beyond it (a claim); any other view at that position (a branch, a
// replayed continuation) is refused and its caller copies. So no holder ever sees a change,
// with no ownership analysis.
use std::cell::{Cell, OnceCell};

// Which end of the slot range a claim writes. Lists: High is the head end (cons), Low the
// last-element end (snoc). Strings: High is the end (append), Low the start (prepend).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Low,
    High,
}

pub trait Slots {
    type Item;
    // `low` free slots, then the `n` `items` (low to high), then `high` free slots. `n` is
    // explicit because the callers chain two operands, and `Chain` is not ExactSizeIterator.
    fn with_room(low: usize, n: usize, items: impl Iterator<Item = Self::Item>, high: usize) -> Self;
    // Some(free slots beyond `at` on `side`) when `at` is the written interval's frontier on
    // that side; None when the slot past `at` is already written (a branch).
    fn room(&self, side: Side, at: usize) -> Option<usize>;
    // Write `items` outward from `at` (High: at, at + 1, ...; Low: at - 1, at - 2, ...) and
    // move the frontier past them. The caller has checked the room.
    fn write(&self, side: Side, at: usize, items: impl Iterator<Item = Self::Item>);
}

pub struct Claimed<S> {
    pub(crate) slots: S,
    // The growth claim: taken by the first copy that reserves room past this storage, and
    // born taken for a bulk build, so a branch never reserves a second large copy.
    grown: Cell<bool>,
}

impl<S: Slots> Claimed<S> {
    pub fn new(slots: S, grown: bool) -> Self {
        Claimed { slots, grown: Cell::new(grown) }
    }

    // All or nothing: `items` go beyond the view edge `at` on `side`, or nothing is written.
    pub fn claim(&self, side: Side, at: usize, items: impl ExactSizeIterator<Item = S::Item>) -> bool {
        if self.slots.room(side, at).is_some_and(|r| r >= items.len()) {
            self.slots.write(side, at, items);
            true
        } else {
            false
        }
    }

    // As many of `items` as fit above `at` (lists' cons: the rest goes into a chained block).
    pub fn claim_some(&self, at: usize, items: &mut impl ExactSizeIterator<Item = S::Item>) -> usize {
        let n = self.slots.room(Side::High, at).unwrap_or(0).min(items.len());
        self.slots.write(Side::High, at, items.by_ref().take(n));
        n
    }

    // Whether a copy made because the view at `at` could not extend in place gets room: only
    // when that view sits at the frontier on `side` (not a branch) and wins the growth claim.
    pub fn should_grow(&self, side: Side, at: usize) -> bool {
        self.slots.room(side, at).is_some() && !self.grown.replace(true)
    }

    // A fresh storage holding the `n` `items` (low to high), with as much room again on
    // `side` when `grow`, otherwise exact.
    pub fn copy(side: Side, n: usize, items: impl Iterator<Item = S::Item>, grow: bool) -> Self {
        let room = if grow { n } else { 0 };
        let (low, high) = match side {
            Side::Low => (room, 0),
            Side::High => (0, room),
        };
        Claimed::new(S::with_room(low, n, items, high), false)
    }
}

impl<T> Slots for Box<[OnceCell<T>]> {
    type Item = T;

    fn with_room(low: usize, n: usize, items: impl Iterator<Item = T>, high: usize) -> Self {
        let mut v = Vec::with_capacity(low + n + high);
        v.resize_with(low, OnceCell::new);
        v.extend(items.map(OnceCell::from));
        assert_eq!(v.len(), low + n, "internal: with_room got {n} items");
        v.resize_with(low + n + high, OnceCell::new);
        v.into_boxed_slice()
    }

    fn room(&self, side: Side, at: usize) -> Option<usize> {
        // The written slots are one interval containing the view, so the slot past the edge
        // being empty means every slot beyond it is empty.
        match side {
            Side::High if at == self.len() => Some(0),
            Side::High => self[at].get().is_none().then_some(self.len() - at),
            Side::Low if at == 0 => Some(0),
            Side::Low => self[at - 1].get().is_none().then_some(at),
        }
    }

    fn write(&self, side: Side, at: usize, items: impl Iterator<Item = T>) {
        for (k, x) in items.enumerate() {
            let i = match side {
                Side::High => at + k,
                Side::Low => at - 1 - k,
            };
            assert!(self[i].set(x).is_ok(), "internal: a slot beyond the frontier is free");
        }
    }
}

// The view model the generic test drives: what a storage's items are as small numbers.
#[cfg(test)]
pub(crate) trait Model: Slots + Sized {
    fn item(x: u8) -> Self::Item;
    // The written slots [lo, hi), low to high.
    fn read(&self, lo: usize, hi: usize) -> Vec<u8>;
}

// Random claims at both ends, branches, tail views and copies over a pool of views, each
// checked against a Vec model: no view's contents ever change. T-S1 runs it for ByteSlots.
#[cfg(test)]
pub(crate) fn random_claims_match_a_model<S: Model>() {
    use std::rc::Rc;
    struct V<S> {
        c: Rc<Claimed<S>>,
        lo: usize,
        hi: usize,
        model: Vec<u8>, // low to high
    }
    fn view<S: Model>(c: Claimed<S>, lo: usize, model: Vec<u8>) -> V<S> {
        V { c: Rc::new(c), lo, hi: lo + model.len(), model }
    }
    let mut seed: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let exact = |xs: &[u8]| Claimed::<S>::copy(Side::High, xs.len(), xs.iter().map(|&x| S::item(x)), false);
    let mut pool = vec![view(exact(&[1]), 0, vec![1]), view(exact(&[2, 3]), 0, vec![2, 3])];
    for step in 0..3000usize {
        let i = next(pool.len());
        let (c, lo, hi, model) = (pool[i].c.clone(), pool[i].lo, pool[i].hi, pool[i].model.clone());
        if next(4) == 0 && hi - lo > 1 {
            // A tail view at either end: no longer at the frontier unless nothing was claimed past it.
            let v = if next(2) == 0 { V { c, lo: lo + 1, hi, model: model[1..].to_vec() } } else { V { c, lo, hi: hi - 1, model: model[..model.len() - 1].to_vec() } };
            pool.push(v);
            continue;
        }
        let side = if next(2) == 0 { Side::Low } else { Side::High };
        let add: Vec<u8> = (0..1 + next(3)).map(|k| (step * 7 + k) as u8).collect();
        let (at, joined) = match side {
            Side::High => (hi, [model.as_slice(), add.as_slice()].concat()),
            Side::Low => (lo, add.iter().rev().chain(&model).copied().collect::<Vec<u8>>()),
        };
        let v = if c.claim(side, at, add.iter().map(|&x| S::item(x))) {
            match side {
                Side::High => V { c, lo, hi: hi + add.len(), model: joined },
                Side::Low => V { c, lo: lo - add.len(), hi, model: joined },
            }
        } else {
            let grow = c.should_grow(side, at);
            let n = joined.len();
            let room = if grow { n } else { 0 };
            let lo = if side == Side::Low { room } else { 0 };
            view(Claimed::<S>::copy(side, n, joined.iter().map(|&x| S::item(x)), grow), lo, joined)
        };
        if v.model.len() < 48 {
            pool.push(v);
        }
        if step % 100 == 0 || step == 2999 {
            for v in &pool {
                assert_eq!(v.c.slots.read(v.lo, v.hi), v.model, "step {step}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;

    type S = Box<[OnceCell<Value>]>;

    impl Model for S {
        fn item(x: u8) -> Value {
            Value::Int(i64::from(x))
        }
        fn read(&self, lo: usize, hi: usize) -> Vec<u8> {
            self[lo..hi].iter().map(|c| c.get().expect("a view's slots are written").as_int() as u8).collect()
        }
    }

    #[test]
    fn random_claims_match_a_model_for_list_slots() {
        random_claims_match_a_model::<S>();
    }

    #[test]
    fn a_claim_then_a_branch_then_growth_only_at_the_frontier() {
        let c: Claimed<S> = Claimed::copy(Side::Low, 2, [1, 2].map(S::item).into_iter(), true); // [_, _, 1, 2]
        assert_eq!((c.slots.room(Side::Low, 2), c.slots.room(Side::High, 4)), (Some(2), Some(0)));
        assert!(c.claim(Side::Low, 2, [3].map(S::item).into_iter()), "writes slot 1");
        assert_eq!(c.slots.room(Side::Low, 2), None, "the old edge is now a branch");
        assert!(!c.claim(Side::Low, 2, [4].map(S::item).into_iter()));
        assert!(!c.should_grow(Side::Low, 2), "a branch never grows, and leaves the growth claim");
        assert!(c.should_grow(Side::Low, 1), "the frontier wins the growth claim");
        assert!(!c.should_grow(Side::Low, 1), "once");
        assert!(!c.claim(Side::Low, 1, [5, 6].map(S::item).into_iter()), "too little room: all or nothing");
        assert_eq!(c.slots.read(1, 4), [3, 1, 2]);
        let mut more = [7, 8, 9].map(S::item).into_iter();
        let d: Claimed<S> = Claimed::copy(Side::High, 1, [1].map(S::item).into_iter(), true); // [1, _]
        assert_eq!(d.claim_some(1, &mut more), 1, "claim_some fills what fits");
        assert_eq!(more.len(), 2);
        assert_eq!(d.slots.read(0, 2), [1, 7]);
    }
}
