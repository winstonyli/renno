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
            // Runtime half of typecheck::coerce's boundary-check
            // desugaring (see its own doc comment) -- ordinary prelude
            // builtins, not hidden, the same way `fail` already isn't.
            .bind("is_int", Value::Builtin(Builtin::IsInt))
            .bind("is_bool", Value::Builtin(Builtin::IsBool))
            .bind("is_str", Value::Builtin(Builtin::IsStr))
            .bind("is_list", Value::Builtin(Builtin::IsList))
            .bind("is_fun", Value::Builtin(Builtin::IsFun))
            .bind("is_record", Value::Builtin(Builtin::IsRecord))
            .bind("has_field", Value::Builtin(Builtin::HasField))
            // Runtime half of `.field` access's own desugaring
            // (typecheck::elaborate_node's Expr::FieldAccess arm) --
            // same "ordinary prelude builtin, not hidden" treatment.
            .bind("get_field", Value::Builtin(Builtin::GetField))
            .bind("type_name", Value::Builtin(Builtin::TypeName))
            .bind("print", Value::Builtin(Builtin::Print))
    }

    pub fn lookup(&self, name: &str) -> Value {
        self.get(name).unwrap_or_else(|| panic!("unbound variable: {name}"))
    }
}
