use crate::value::{Builtin, TypeCtor};

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
    // `is(T, v)`: v satisfies the type value T. An ordinary prelude builtin,
    // not hidden, the same way `fail` isn't; Dyn-boundary checks do not call
    // it (typecheck emits a native Expr::Check; machine::test_holds is the
    // shared definition). List(T) is deep, Dyn always holds, and a Fun type is
    // callability only (its parameter and result types are ignored).
    ("is", Builtin::Is),
    // Runtime half of `.field` access's own desugaring
    // (typecheck::elaborate_node's Expr::FieldAccess arm) --
    // same "ordinary prelude builtin, not hidden" treatment.
    ("get_field", Builtin::GetField),
    ("print", Builtin::Print),
    ("to_str", Builtin::ToStr),
    ("filter", Builtin::Filter),
    ("reverse", Builtin::Reverse),
    ("sort", Builtin::Sort),
    ("range", Builtin::Range),
    ("join", Builtin::Join),
    // The type constructors: an annotation is desugared into calls to these
    // (as `#List`, `#Fun`, ..., which no user binding can capture); `Vec(`
    // and `Indexed(` call the names in scope (spec 11.8).
    ("Int", Builtin::Ty(TypeCtor::Int)),
    ("Float", Builtin::Ty(TypeCtor::Float)),
    ("Bool", Builtin::Ty(TypeCtor::Bool)),
    ("Str", Builtin::Ty(TypeCtor::Str)),
    ("Dyn", Builtin::Ty(TypeCtor::Dyn)),
    ("List", Builtin::Ty(TypeCtor::List)),
    ("Tuple", Builtin::Ty(TypeCtor::Tuple)),
    ("Record", Builtin::Ty(TypeCtor::Record)),
    ("Union", Builtin::Ty(TypeCtor::Union)),
    ("Fun", Builtin::Ty(TypeCtor::Fun)),
    ("Vec", Builtin::Ty(TypeCtor::Vec)),
    ("Indexed", Builtin::Ty(TypeCtor::Indexed)),
    ("TypeOf", Builtin::Ty(TypeCtor::TypeOf)),
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
    fn prelude_table_has_unique_names() {
        let mut names: Vec<&str> = PRELUDE.iter().map(|(n, _)| *n).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), PRELUDE.len(), "duplicate name in PRELUDE");
    }

    #[test]
    fn prelude_env_is_the_empty_root() {
        assert!(Env::prelude().is_root());
    }
}
