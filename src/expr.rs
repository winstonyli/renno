use cranelift_entity::{entity_impl, PrimaryMap};

use crate::types::Type;

// AST nodes live in a flat arena instead of being individually
// Rc-allocated and linked by pointer. Two things this buys:
//   - Dropping the arena is dropping one Vec -- O(n), no recursion. The
//     old Rc<Expr>-linked tree's default Drop recursed through every
//     nested field and could overflow the stack on deeply nested source
//     (confirmed independently, see lib.rs's earlier fix history); a
//     flat Vec's Drop just iterates.
//   - ExprRef is Copy (a plain u32 index), so cloning a node reference
//     (e.g. into a Frame in machine.rs) is a plain copy, not even an Rc
//     bump.
// Traversal (parsing, typecheck, evaluation) is unaffected by this --
// arena allocation changes how nodes are OWNED, not how they're walked,
// so the Let/Fun chain flattening in parser.rs and typecheck.rs is still
// what keeps deep chains off the native call stack.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ExprRef(u32);
entity_impl!(ExprRef, "expr");

pub type Arena = PrimaryMap<ExprRef, Expr>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Eq,
    Lt,
    // `++`: Str++Str or List++List. Kept separate from Add rather than
    // overloading `+` (Python/JS-style) -- simpler and lower-risk than
    // generalizing Add's existing Int-only typecheck arm; revisit if `+`
    // overloading turns out to read better in practice.
    Concat,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Bool(bool),
    Str(String),
    // Variable-arity, unlike every other node -- elaborate/machine handle
    // it with a loop over the Vec rather than a fixed-shape match.
    ListLit(Vec<ExprRef>),
    Var(String),
    // param annotation is optional -- None means Dyn. The typechecker fills
    // this in (or leaves it) when elaborating; the parser fills it in only
    // when the source has an explicit `: Type` annotation.
    Lambda(String, Option<Type>, ExprRef),
    App(ExprRef, ExprRef),
    // Last field: `let rec` (true) vs plain `let` (false). Meaningful only
    // when `val` evaluates to a function -- machine.rs wraps it into a
    // Value::RecClosure so it can rebind its own name into scope each time
    // it's called (see Value::RecClosure's doc comment for how that avoids
    // needing mutation or an AST-rewriting Y-combinator encoding).
    Let(String, Option<Type>, ExprRef, ExprRef, bool),
    BinOp(BinOp, ExprRef, ExprRef),
    // Runtime type check: produced only by typecheck::elaborate, never by
    // the parser directly. Verifies the inner expr's value matches `Type`
    // before passing it through -- inserted only where a Dyn-typed value
    // flows into an annotated position, so unannotated code pays nothing.
    Check(Type, ExprRef),
    // cond must evaluate to Bool.
    If(ExprRef, ExprRef, ExprRef),
    Perform(String, ExprRef),
    // `handler` evaluates to a Value::Handler. Handle itself carries no
    // deep/shallow flag -- that lives on the handler value, set by the
    // `deep`/`shallow` builtins (see env::prelude), not on this AST node.
    Handle {
        body: ExprRef,
        handler: ExprRef,
    },
    // Builds the base (shallow) handler value for one effect clause.
    // `fun (payload_var) (resume_var) -> body`, tagged with the effect name.
    MakeHandler {
        effect: String,
        payload_var: String,
        resume_var: String,
        body: ExprRef,
    },
}
