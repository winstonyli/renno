use std::rc::Rc;

// Generic persistent, singly-linked, prepend-only list keyed by name. O(1)
// extend (an Rc clone), O(depth) lookup (most-recent-binding-wins). Used
// ONLY by typecheck's Ctx (payload = Type) -- the runtime Env moved to
// frame::Env (per-scope immutable frames, statically resolved lookups); see
// env.rs and resolve.rs.
#[derive(Clone)]
pub struct PList<T>(Rc<PNode<T>>);

enum PNode<T> {
    Empty,
    Bind(String, T, PList<T>),
}

impl<T: Clone> PList<T> {
    pub fn empty() -> Self {
        PList(Rc::new(PNode::Empty))
    }

    pub fn bind(&self, name: impl Into<String>, value: T) -> Self {
        PList(Rc::new(PNode::Bind(name.into(), value, self.clone())))
    }

    pub fn get(&self, name: &str) -> Option<T> {
        let mut node = self.clone();
        loop {
            match &*node.0 {
                PNode::Empty => return None,
                PNode::Bind(n, v, parent) => {
                    if n == name {
                        return Some(v.clone());
                    }
                    node = parent.clone();
                }
            }
        }
    }

    // Visits every value currently bound, innermost (most recent) first --
    // same iterative Rc-clone walk as `get`, just without stopping at a
    // name match. Used by typecheck::free_row_vars_in_ctx to scan the
    // WHOLE visible scope chain; kept iterative for the same reason `get`
    // and the custom Drop impl below are: a chain built from a long
    // sequential program must not cost native stack proportional to its
    // depth. Unlike `get`, there's no early exit -- `f` runs for every
    // node, unconditionally. Harmless for today's only caller (Ctx, and
    // even that's short-circuited one level up by
    // typecheck::generalizable_row_vars before it ever reaches here), but
    // worth knowing before reusing this against a much deeper PList<Value>
    // Env in a hot path: add a variant that lets `f` signal "stop" first.
    pub fn for_each(&self, mut f: impl FnMut(&T)) {
        let mut node = self.clone();
        loop {
            match &*node.0 {
                PNode::Empty => return,
                PNode::Bind(_, v, parent) => {
                    f(v);
                    node = parent.clone();
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
        let mut node = std::mem::replace(&mut self.0, Rc::new(PNode::Empty));
        loop {
            match Rc::try_unwrap(node) {
                // `parent` has its own Drop impl (this one, recursively),
                // so it can't be partially moved out of -- swap its inner
                // Rc out first. `parent` then wraps Empty and drops
                // trivially when this match arm ends.
                Ok(PNode::Bind(_, _, mut parent)) => node = std::mem::replace(&mut parent.0, Rc::new(PNode::Empty)),
                _ => break,
            }
        }
    }
}
