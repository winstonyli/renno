// A dependency-free leaf: no `use crate::...` of its own, so any module
// can share this without taking on an inappropriate coupling. Exists
// specifically because "look up a field by name" was independently
// reimplemented across the static/runtime split renno's record support
// has (types::Type::Record's own (String, Type) pairs, typecheck's
// Pattern-vs-Type field checks, machine::Value::Record's (String, Value)
// pairs at runtime) -- machine.rs in particular has no other reason to
// depend on types.rs (the interpreter doesn't need to know about static
// types to run a program), so this couldn't just live there.

// The first entry in `fields` named `name`, if any -- shared shape behind
// every "does this record have field X, and if so what's there" question,
// whether `T` is a static `Type` or a runtime `Value`.
pub fn find_field<'a, T>(fields: &'a [(String, T)], name: &str) -> Option<&'a T> {
    fields.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

// The payload of every deliberate run-time error (a type error, division by zero,
// an unhandled effect...). run_source tells it apart from any other panic, which
// is an interpreter bug and reported as an internal error.
pub struct RunError(pub String);

#[macro_export]
macro_rules! run_error {
    ($($arg:tt)*) => {
        ::std::panic::panic_any($crate::util::RunError(format!($($arg)*)))
    };
}

// Keeps the default panic message for an interpreter bug and drops it for a RunError,
// which run_source reports on its own. Process-global: the binary installs it, not the library.
pub fn quiet_run_errors() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !info.payload().is::<RunError>() {
            default(info);
        }
    }));
}
