use std::collections::BTreeSet;
use std::rc::Rc;

use cranelift_entity::{entity_impl, PrimaryMap};

use crate::span::Span;
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
// The byte-span of each Arena node's ORIGINAL source text, indexed by the
// same ExprRef -- built by the parser in lockstep with Arena (see
// parser::Parser::push_spanned) so the two never desync. A node typecheck
// synthesizes (a Check, a wrap_fun_contract chain, a re-emitted Let with a
// resolved type) gets no entry here; typecheck never needs one, since every
// error it reports uses the span of an ORIGINAL (pre-elaboration) ExprRef
// that was already in scope before any such node was built -- see
// typecheck::TypeError.
pub type SpanMap = PrimaryMap<ExprRef, Span>;

// No ADTs yet, so patterns only destructure the value shapes renno already
// has natively: literals (matched by equality), List's two structural
// forms (fixed-length and cons), and Var, which matches anything and binds
// it -- "_" is just an ordinary (unused) Var name, not a special token.
#[derive(Debug, Clone)]
pub enum Pattern {
    Var(String),
    Int(i64),
    Bool(bool),
    Str(String),
    // [], [p1, p2, ...] -- matches only a list of exactly this length.
    List(Vec<Pattern>),
    // `head :: tail` -- matches a non-empty list of any length.
    Cons(Box<Pattern>, Box<Pattern>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    // Integer division, truncating toward zero (Rust's own `/` on i64) --
    // dividing by zero panics at runtime (apply_binop), same as any other
    // Dyn-sourced-value mismatch this interpreter can't rule out statically.
    Div,
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
    // `match scrutinee with | p1 -> e1 | p2 -> e2 ...`, tried top to bottom,
    // first match wins. Arms in Rc (not a plain Vec, unlike ListLit) since
    // a captured continuation can carry a MatchArms frame (see cont.rs) --
    // resuming it more than once (multi-shot) would otherwise reclone the
    // whole arm list on every resume.
    Match(ExprRef, Rc<Vec<(Pattern, ExprRef)>>),
    // Emitted only for a `data` declaration (see parser::build_ctor_value)
    // -- purely a compile-time marker recording which constructor tags
    // belong to one type, consumed by typecheck's Match exhaustiveness
    // check (missing_case) and otherwise fully transparent: evaluates
    // straight through to `body` (see machine.rs), and typecheck's own
    // elaborate unwraps it -- doesn't re-emit it -- once the tag set has
    // been recorded, so it never reaches an already-typechecked program.
    DataGroup(Rc<BTreeSet<String>>, ExprRef),
}
