use crate::value::Builtin;

pub use crate::frame::Env;

// Single source of truth for the builtin prelude: Env::prelude() is the
// empty root frame chain, and resolve::resolve looks names up in it (as
// VarRef::Prelude(index)). ORDER MATTERS: it fixes the VarRef::Prelude(i)
// indices -- harmless, both consumers read this one table. Names are unique.
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
    // Ordinary prelude predicates, not hidden, the same way `fail` isn't.
    // Dyn-boundary checks no longer call them (typecheck emits a native
    // Expr::Check; machine::test_holds is the shared definition).
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

impl Env {
    // The runtime root. The builtin prelude lives in PRELUDE, not in any
    // frame: resolve::VarRef::Prelude(i) indexes it directly. Kept as a
    // constructor so every `machine::run(.., Env::prelude(), ..)` call site
    // is unchanged.
    pub fn prelude() -> Env {
        Env::root()
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
    fn prelude_env_is_the_empty_root() {
        assert!(Env::prelude().is_root());
    }
}
