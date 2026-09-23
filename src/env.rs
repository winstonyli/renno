use crate::plist::PList;
use crate::value::{Builtin, Value};

// Persistent environment: extending never mutates the parent, so any Env
// captured by a closure or continuation stays valid forever.
pub type Env = PList<Value>;

// Single source of truth for the builtin prelude: Env::prelude() folds it
// into a chain today, and resolve::resolve looks names up in it (as
// VarRef::Prelude(index)). ORDER MATTERS in two ways: it is the order the
// bind chain is built in (first entry = deepest), and it fixes the
// VarRef::Prelude(i) indices (harmless: both consumers read this one table).
// Names are unique.
pub static PRELUDE: &[(&str, Builtin)] = &[
    // `deep`/`shallow` bound as ordinary values -- functions that take a
    // handler value and return a new one with the reinstall bit flipped.
    // Nothing else in the core language knows about deep vs shallow.
    ("deep", Builtin::Deep),
    ("shallow", Builtin::Shallow),
    ("len", Builtin::Len),
    ("map", Builtin::Map),
    ("fold", Builtin::Fold),
    ("fail", Builtin::Fail),
    ("get", Builtin::Get),
    // Runtime half of typecheck::coerce's boundary-check
    // desugaring (see its own doc comment) -- ordinary prelude
    // builtins, not hidden, the same way `fail` already isn't.
    ("is_int", Builtin::IsInt),
    ("is_float", Builtin::IsFloat),
    ("is_bool", Builtin::IsBool),
    ("is_str", Builtin::IsStr),
    ("is_list", Builtin::IsList),
    ("is_fun", Builtin::IsFun),
    ("is_record", Builtin::IsRecord),
    ("has_field", Builtin::HasField),
    // Runtime half of `.field` access's own desugaring
    // (typecheck::elaborate_node's Expr::FieldAccess arm) --
    // same "ordinary prelude builtin, not hidden" treatment.
    ("get_field", Builtin::GetField),
    ("type_name", Builtin::TypeName),
    ("print", Builtin::Print),
    ("to_str", Builtin::ToStr),
    ("filter", Builtin::Filter),
    ("reverse", Builtin::Reverse),
    ("zip", Builtin::Zip),
    ("sort", Builtin::Sort),
    ("range", Builtin::Range),
    ("split", Builtin::Split),
    ("join", Builtin::Join),
    ("trim", Builtin::Trim),
];

impl PList<Value> {
    pub fn prelude() -> Env {
        PRELUDE.iter().fold(Env::empty(), |env, (name, b)| env.bind(*name, Value::Builtin(*b)))
    }

    pub fn lookup(&self, name: &str) -> Value {
        self.get(name).unwrap_or_else(|| panic!("unbound variable: {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prelude_table_has_27_unique_names() {
        assert_eq!(PRELUDE.len(), 27);
        let mut names: Vec<&str> = PRELUDE.iter().map(|(n, _)| *n).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 27, "duplicate name in PRELUDE");
    }

    #[test]
    fn prelude_env_matches_table() {
        let env = Env::prelude();
        for (name, b) in PRELUDE {
            match env.get(name) {
                Some(Value::Builtin(got)) => assert!(got == *b, "wrong builtin bound to {name}"),
                _ => panic!("{name} not bound to a Builtin in Env::prelude()"),
            }
        }
    }
}
