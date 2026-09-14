use std::rc::Rc;

// Generic persistent, singly-linked, prepend-only list keyed by name. O(1)
// extend (an Rc clone), O(depth) lookup (most-recent-binding-wins). Shared
// by the runtime Env (payload = Value) and the typechecker's Ctx (payload =
// Type) so a fix to the shadowing/lookup semantics applies to both instead
// of living as two independently-maintained copies.
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
}
