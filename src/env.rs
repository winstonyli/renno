use crate::plist::PList;
use crate::value::{Builtin, Value};

// Persistent environment: extending never mutates the parent, so any Env
// captured by a closure or continuation stays valid forever.
pub type Env = PList<Value>;

impl PList<Value> {
    // `deep`/`shallow` bound as ordinary values -- functions that take a
    // handler value and return a new one with the reinstall bit flipped.
    // Nothing else in the core language knows about deep vs shallow.
    pub fn prelude() -> Env {
        Env::empty()
            .bind("deep", Value::Builtin(Builtin::Deep))
            .bind("shallow", Value::Builtin(Builtin::Shallow))
            .bind("len", Value::Builtin(Builtin::Len))
            .bind("map", Value::Builtin(Builtin::Map))
            .bind("fold", Value::Builtin(Builtin::Fold))
            .bind("fail", Value::Builtin(Builtin::Fail))
            .bind("get", Value::Builtin(Builtin::Get))
    }

    pub fn lookup(&self, name: &str) -> Value {
        self.get(name).unwrap_or_else(|| panic!("unbound variable: {name}"))
    }
}
