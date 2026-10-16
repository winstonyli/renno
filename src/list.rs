// Test-only operation counters for the list representation: the complexity tests count
// operations instead of timing them, so they are deterministic and fast in a debug build.
// Thread-locals, read on the thread that ran the program (the same pattern as
// machine::LAZY_MINTS).

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
thread_local! {
    // Each Value placed into list storage.
    pub(crate) static WRITES: Cell<usize> = const { Cell::new(0) };
    // Each storage allocation.
    pub(crate) static BLOCKS: Cell<usize> = const { Cell::new(0) };
    // Each block visited by `get`.
    pub(crate) static HOPS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_counters() {
    WRITES.with(|c| c.set(0));
    BLOCKS.with(|c| c.set(0));
    HOPS.with(|c| c.set(0));
}

#[cfg(test)]
pub(crate) fn counters() -> (usize, usize, usize) {
    (WRITES.with(|c| c.get()), BLOCKS.with(|c| c.get()), HOPS.with(|c| c.get()))
}

#[allow(unused_variables)]
pub(crate) fn count_writes(n: usize) {
    #[cfg(test)]
    WRITES.with(|c| c.set(c.get() + n));
}

pub(crate) fn count_block() {
    #[cfg(test)]
    BLOCKS.with(|c| c.set(c.get() + 1));
}

pub(crate) fn count_hop() {
    #[cfg(test)]
    HOPS.with(|c| c.set(c.get() + 1));
}
