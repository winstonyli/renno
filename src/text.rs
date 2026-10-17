// Strings: a flat `Rc<str>` (literals, builtin results), or a view of a growable byte buffer
// that `++` extends in place through the shared claim engine (claimed.rs): this module adds
// only the byte storage (`ByteSlots`), whose `Deref` to `str` is the crate's one `unsafe`.
use crate::claimed::{Claimed, Side, Slots};
use std::cell::Cell;
use std::fmt;
use std::ops::Deref;
use std::rc::Rc;

// Test-only operation counters (the same pattern as list.rs): bytes placed into string
// storage and string allocations, read on the thread that ran the program.
#[cfg(test)]
thread_local! {
    pub(crate) static BYTES: Cell<usize> = const { Cell::new(0) };
    pub(crate) static BUFS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_counters() {
    BYTES.with(|c| c.set(0));
    BUFS.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn counters() -> (usize, usize) {
    (BYTES.with(|c| c.get()), BUFS.with(|c| c.get()))
}

// A new string allocation holding `bytes` bytes.
#[allow(unused_variables)]
pub(crate) fn count_new(bytes: usize) {
    #[cfg(test)]
    {
        BYTES.with(|c| c.set(c.get() + bytes));
        BUFS.with(|c| c.set(c.get() + 1));
    }
}

// Bytes claimed into an existing buffer.
#[allow(unused_variables)]
fn count_claim(bytes: usize) {
    #[cfg(test)]
    BYTES.with(|c| c.set(c.get() + bytes));
}

// The representation is private: safe code outside this module cannot build a `Grow` with an
// arbitrary range, which is what makes the SAFETY argument in `Deref` hold for all safe code.
// Rc makes Text !Send and !Sync (pinned by a test below).
#[derive(Clone)]
pub struct Text(Repr);

#[derive(Clone)]
enum Repr {
    Flat(Rc<str>),
    Grow { buf: Rc<Claimed<ByteSlots>>, lo: usize, hi: usize },
}

// A `Cell<u8>` cannot tell an unwritten byte from a written 0, so unlike list slots this
// storage keeps its written interval explicitly. Only these methods write `bytes`. The type
// is private, so nothing outside this module can call them.
struct ByteSlots {
    bytes: Box<[Cell<u8>]>,
    lo: Cell<usize>,
    hi: Cell<usize>,
}

impl Slots for ByteSlots {
    type Item = u8;

    fn with_room(low: usize, n: usize, items: impl Iterator<Item = u8>, high: usize) -> Self {
        let bytes: Box<[Cell<u8>]> = (0..low + n + high).map(|_| Cell::new(0)).collect();
        let mut k = low;
        for x in items {
            bytes[k].set(x);
            k += 1;
        }
        assert_eq!(k, low + n, "internal: with_room got {n} bytes");
        ByteSlots { bytes, lo: Cell::new(low), hi: Cell::new(low + n) }
    }

    fn room(&self, side: Side, at: usize) -> Option<usize> {
        match side {
            Side::High => (at == self.hi.get()).then_some(self.bytes.len() - at),
            Side::Low => (at == self.lo.get()).then_some(at),
        }
    }

    // The bytes are written before the frontier moves past them (the SAFETY argument in Deref).
    fn write(&self, side: Side, at: usize, items: impl Iterator<Item = u8>) {
        let mut k = 0;
        for x in items {
            let i = match side {
                Side::High => at + k,
                Side::Low => at - 1 - k,
            };
            self.bytes[i].set(x);
            k += 1;
        }
        match side {
            Side::High => self.hi.set(at + k),
            Side::Low => self.lo.set(at - k),
        }
    }
}

impl Text {
    pub fn concat(&self, b: &Text) -> Text {
        if self.is_empty() {
            return b.clone();
        }
        if b.is_empty() {
            return self.clone();
        }
        // The longer side stays and the shorter is written: append onto a longer left side,
        // prepend onto a longer right side.
        let back = self.len() >= b.len();
        let (side, longer) = if back { (Side::High, self) } else { (Side::Low, b) };
        let mut grow = false;
        if let Repr::Grow { buf, lo, hi } = &longer.0 {
            let (lo, hi) = (*lo, *hi);
            let claimed = if back { buf.claim(side, hi, b.bytes()) } else { buf.claim(side, lo, self.bytes().rev()) };
            if claimed {
                let n = if back { b.len() } else { self.len() };
                count_claim(n);
                let (lo, hi) = if back { (lo, hi + n) } else { (lo - n, hi) };
                return Text(Repr::Grow { buf: buf.clone(), lo, hi });
            }
            grow = buf.should_grow(side, if back { hi } else { lo });
        }
        // Copy: as much room again on the joining side when the longer operand sat at its
        // frontier and won its growth claim; otherwise exact (a branch, a flat string).
        let n = self.len() + b.len();
        count_new(n);
        let buf = Claimed::<ByteSlots>::copy(side, n, self.bytes().chain(b.bytes()), grow);
        let lo = buf.slots.lo.get();
        Text(Repr::Grow { buf: Rc::new(buf), lo, hi: lo + n })
    }
}

impl Deref for Text {
    type Target = str;

    fn deref(&self) -> &str {
        match &self.0 {
            Repr::Flat(s) => s,
            Repr::Grow { buf, lo, hi } => {
                let cells = &buf.slots.bytes[*lo..*hi];
                // SAFETY: (1) Cell<u8> is repr(transparent) over u8. (2) Nothing writes these
                // bytes while the borrow lives: `ByteSlots::write` and `with_room` write a byte
                // only while it lies outside [slots.lo, slots.hi) (the engine checks `room`
                // first), after which it is inside that interval for the buffer's whole life,
                // and every view lies inside it; Rc and Cell make Text !Send and !Sync, so no
                // other thread writes. (3) UTF-8: every view is a concatenation of whole &strs
                // written at its claimed edge, so it starts and ends on char boundaries.
                unsafe {
                    let bytes = &*(cells as *const [Cell<u8>] as *const [u8]);
                    debug_assert!(std::str::from_utf8(bytes).is_ok());
                    std::str::from_utf8_unchecked(bytes)
                }
            }
        }
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self)
    }
}

impl PartialEq for Text {
    fn eq(&self, other: &Text) -> bool {
        **self == **other
    }
}

impl From<&str> for Text {
    fn from(s: &str) -> Text {
        count_new(s.len());
        Text(Repr::Flat(Rc::from(s)))
    }
}

impl From<String> for Text {
    fn from(s: String) -> Text {
        count_new(s.len());
        Text(Repr::Flat(Rc::from(s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str) -> Text {
        Text::from(s)
    }

    fn buf_of(x: &Text) -> Option<*const Claimed<ByteSlots>> {
        match &x.0 {
            Repr::Grow { buf, .. } => Some(Rc::as_ptr(buf)),
            Repr::Flat(_) => None,
        }
    }

    fn cap(x: &Text) -> usize {
        match &x.0 {
            Repr::Grow { buf, .. } => buf.slots.bytes.len(),
            Repr::Flat(s) => s.len(),
        }
    }

    // Compile-time check that Text is !Send: if it were Send, `f` would be ambiguous.
    #[test]
    fn text_is_not_send() {
        trait AmbiguousIfSend<A> {
            fn f() {}
        }
        impl<T: ?Sized> AmbiguousIfSend<()> for T {}
        impl<T: ?Sized + Send> AmbiguousIfSend<u8> for T {}
        <Text as AmbiguousIfSend<_>>::f();
    }

    // The engine's generic model test (T-L1), run for the byte storage: the claim, branch,
    // growth and copy logic is the one list slots already pass.
    impl crate::claimed::Model for ByteSlots {
        fn item(x: u8) -> u8 {
            x
        }
        fn read(&self, lo: usize, hi: usize) -> Vec<u8> {
            self.bytes[lo..hi].iter().map(Cell::get).collect()
        }
    }

    #[test]
    fn random_claims_match_a_model_for_byte_slots() {
        crate::claimed::random_claims_match_a_model::<ByteSlots>();
    }

    // "ab" ++ "c" is exact (a flat operand); ++ "d" doubles: "abcd" in an 8-byte buffer.
    fn roomy() -> Text {
        t("ab").concat(&t("c")).concat(&t("d"))
    }

    #[test]
    fn an_append_claims_then_branches() {
        let base = roomy();
        assert_eq!((&*base, cap(&base)), ("abcd", 8));
        let x = base.concat(&t("e"));
        let y = base.concat(&t("f"));
        assert_eq!((&*x, &*y, &*base), ("abcde", "abcdf", "abcd"));
        assert_eq!(buf_of(&x), buf_of(&base), "the first append claimed");
        assert_ne!(buf_of(&y), buf_of(&base), "the second copied");
        assert_eq!(cap(&y), 5, "a branch copies exactly");
    }

    #[test]
    fn a_prepend_claims_then_branches() {
        let q = t("x").concat(&t("y").concat(&t("zz")));
        assert_eq!((&*q, cap(&q)), ("xyzz", 8));
        let a = t("1").concat(&q);
        let b = t("2").concat(&q);
        assert_eq!((&*a, &*b, &*q), ("1xyzz", "2xyzz", "xyzz"));
        assert_eq!(buf_of(&a), buf_of(&q));
        assert_ne!(buf_of(&b), buf_of(&q));
    }

    #[test]
    fn operands_from_one_buffer() {
        let s = roomy();
        let ss = s.concat(&s);
        assert_eq!((&*ss, &*s), ("abcdabcd", "abcd"));
        assert_eq!(buf_of(&ss), buf_of(&s));
        assert_eq!(&*s.concat(&t("z")), "abcdz");
    }

    #[test]
    fn empty_sides_are_free_and_equality_ignores_the_representation() {
        let s = roomy();
        assert_eq!(buf_of(&t("").concat(&s)), buf_of(&s));
        assert_eq!(buf_of(&s.concat(&t(""))), buf_of(&s));
        assert!(t("abcd") == s && s != t("abc"));
        assert_eq!(s.to_string(), "abcd");
    }

    #[test]
    fn sizes() {
        assert!(size_of::<Text>() <= 32);
        assert_eq!(size_of::<crate::value::Value>(), 40);
    }

    // Random concats over a pool against String models, with multi-byte pieces: no string's
    // contents ever change, and every view is valid UTF-8 (Deref's debug_assert).
    #[test]
    fn random_concats_never_disturb_other_strings() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self, n: usize) -> usize {
                self.0 ^= self.0 << 13;
                self.0 ^= self.0 >> 7;
                self.0 ^= self.0 << 17;
                (self.0 % n as u64) as usize
            }
        }
        let pieces = ["", "a", "é", "xy", "ü€", "hello"];
        let mut r = Rng(0x9E37_79B9_7F4A_7C15);
        let mut pool: Vec<(Text, String)> = pieces.iter().map(|p| (t(p), p.to_string())).collect();
        for _ in 0..4000 {
            let (a, am) = pool[r.next(pool.len())].clone();
            let (b, bm) = match r.next(3) {
                0 => pool[r.next(pool.len())].clone(),
                _ => {
                    let p = pieces[r.next(pieces.len())];
                    (t(p), p.to_string())
                }
            };
            let (c, cm) = if r.next(2) == 0 { (a.concat(&b), am + &bm) } else { (b.concat(&a), bm + &am) };
            if cm.len() < 400 {
                pool.push((c, cm));
            }
        }
        for (x, m) in &pool {
            assert_eq!(&**x, m.as_str());
        }
    }
}
