use std::rc::Rc;

use crate::value::Value;

// Persistent environment: extending never mutates the parent, so any Env
// captured by a closure or continuation stays valid forever. Lookup is O(n)
// linear scan for now -- fine for a skeleton, swap to resolved indices later.
#[derive(Clone)]
pub struct Env(pub Rc<EnvNode>);

pub enum EnvNode {
    Empty,
    Bind(String, Value, Env),
}

impl Env {
    pub fn empty() -> Env {
        Env(Rc::new(EnvNode::Empty))
    }

    pub fn bind(&self, name: impl Into<String>, value: Value) -> Env {
        Env(Rc::new(EnvNode::Bind(name.into(), value, self.clone())))
    }

    pub fn lookup(&self, name: &str) -> Value {
        let mut node = self.clone();
        loop {
            match &*node.0 {
                EnvNode::Empty => panic!("unbound variable: {name}"),
                EnvNode::Bind(n, v, parent) => {
                    if n == name {
                        return v.clone();
                    }
                    node = parent.clone();
                }
            }
        }
    }
}
