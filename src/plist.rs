use rpds::HashTrieMap;
use std::rc::Rc;

// Generic persistent, prepend-only list keyed by name. O(1) extend, and
// most-recent-binding-wins lookup in O(log n) through a persistent hash map
// beside the list (a chain of thousands of lets must not make every lookup of
// an old binder, such as a prelude name, walk the whole chain). The list
// still holds every binding, shadowed ones included, for `for_each`. Used
// ONLY by typecheck's Ctx (payload = Scheme) -- the runtime Env moved to
// frame::Env (per-scope immutable frames, statically resolved lookups); see
// env.rs and resolve.rs.
pub struct PList<T> {
    head: Rc<PNode<T>>,
    latest: HashTrieMap<String, Rc<T>>,
}

enum PNode<T> {
    Empty,
    Bind(Rc<T>, PList<T>),
}

impl<T> Clone for PList<T> {
    fn clone(&self) -> Self {
        PList { head: self.head.clone(), latest: self.latest.clone() }
    }
}

impl<T: Clone> PList<T> {
    pub fn empty() -> Self {
        PList { head: Rc::new(PNode::Empty), latest: HashTrieMap::new() }
    }

    pub fn bind(&self, name: impl Into<String>, value: T) -> Self {
        let name = name.into();
        let value = Rc::new(value);
        PList { head: Rc::new(PNode::Bind(value.clone(), self.clone())), latest: self.latest.insert(name, value) }
    }

    pub fn get(&self, name: &str) -> Option<T> {
        self.latest.get(name).map(|v| (**v).clone())
    }

    // Visits every value currently bound, innermost (most recent) first,
    // shadowed bindings included -- a borrowing walk of the list, kept
    // iterative for the same reason the custom Drop impl below is: a chain
    // built from a long sequential program must not cost native stack
    // proportional to its depth. Used by typecheck::free_row_vars_in_ctx to
    // scan the WHOLE visible scope chain; there is no early exit, so `f` runs
    // for every node. Harmless for today's only caller (even that is
    // short-circuited one level up by typecheck::generalizable_row_vars before
    // it reaches here), but worth knowing before reusing this against a much
    // deeper PList<T> in a hot path.
    pub fn for_each(&self, mut f: impl FnMut(&T)) {
        let mut node: &PNode<T> = &self.head;
        loop {
            match node {
                PNode::Empty => return,
                PNode::Bind(v, parent) => {
                    f(v);
                    node = &parent.head;
                }
            }
        }
    }
}

// Rust's default Drop for a struct wrapping Rc<PNode<T>> recurses through
// `parent` one field-drop at a time -- for a chain built from a long
// program (many sequential lets, or a deep typecheck Ctx built while
// walking one) that's the same O(depth) native stack cost this whole
// persistent-list design was meant to avoid, just moved from construction
// to teardown (confirmed independently: building and dropping a 50000-
// deep chain with no parser or typechecker involved at all overflows the
// default stack). Unwind iteratively instead: take ownership of each
// node in turn via Rc::try_unwrap, which only succeeds while we hold the
// last reference to it. The moment some other clone of a node still
// exists (an Env captured by a closure or continuation, e.g.), stop --
// that node and everything under it stays alive and gets cleaned up
// normally by whoever else holds it, so this never double-frees or
// unwinds something still in use.
impl<T> Drop for PList<T> {
    fn drop(&mut self) {
        let mut node = std::mem::replace(&mut self.head, Rc::new(PNode::Empty));
        loop {
            match Rc::try_unwrap(node) {
                // `parent` has its own Drop impl (this one, recursively),
                // so it can't be partially moved out of -- swap its inner
                // Rc out first. `parent` then wraps Empty and drops
                // trivially when this match arm ends.
                Ok(PNode::Bind(_, mut parent)) => node = std::mem::replace(&mut parent.head, Rc::new(PNode::Empty)),
                _ => break,
            }
        }
    }
}
