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

// Patterns destructure the value shapes renno has natively: literals
// (matched by equality), List's two structural forms (fixed-length and
// cons), Record (name-keyed, width-tolerant), and Var, which matches
// anything and binds it -- "_" is just an ordinary (unused) Var name, not
// a special token. A tuple pattern (`(p, q)`) is sugar over List -- see
// parser::pattern_atom's LParen arm -- so there's no separate variant for
// it either.
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
    // `{x: p, y: p, ...}` -- matches a Value::Record that has AT LEAST
    // these field names (see Value::Record's own doc comment), each
    // looked up by name and matched against its own sub-pattern. Unlike
    // List's own two forms, this is NOT exact -- a field the pattern
    // doesn't name is simply never looked at, which is what makes width
    // subtyping work with no separate value-narrowing step anywhere (see
    // types::record_satisfies' own doc comment): a wider record value
    // just flows through unchanged, and only a pattern that actually
    // names an extra field could ever observe it.
    Record(Vec<(String, Pattern)>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    // Int/Int truncates toward zero (Rust's own `/` on i64); Float/Float
    // (or a mix, promoting the Int side) is real division -- see types::
    // Type::Float's own doc comment. Dividing by zero panics at runtime
    // (apply_binop) for BOTH numeric types, not just Int -- deliberately
    // not IEEE754 inf/NaN for the Float case, so "0 divisor is a hard
    // error" stays one uniform story instead of quietly diverging by type.
    Div,
    // Remainder, truncating toward zero same as Div (Rust's own `%` on
    // both i64 and f64) -- so `-7 % 2` is `-1`, not `1`. Same zero-divisor
    // panic as Div, same Int/Float promotion story.
    Mod,
    Eq,
    Lt,
    // `++`: Str++Str or List++List. Kept separate from Add rather than
    // overloading `+` (Python/JS-style) across Str/List -- Add/Sub/Mul/
    // Div/Mod/Lt DID end up overloaded across Int/Float (see types::
    // Type::Float's own doc comment, and typecheck::coerce_numeric), so
    // this comment's own "revisit if + overloading turns out to read
    // better in practice" already got revisited once, for numerics
    // specifically -- Concat stays its own operator regardless: Str/List
    // concatenation and Int/Float arithmetic aren't the same question,
    // and generalizing Add across Str/List would still mean the exact
    // ambiguity-with-numeric-`+` this operator exists to avoid.
    Concat,
    // `h :: t`: prepend h onto list t, producing a new list. The mirror
    // image of Pattern::Cons, which only DEstructures -- until now there
    // was no way to CONstruct a list incrementally in-language at all
    // (only a full [a, b, c] literal or O(n^2) repeated `++`), which is
    // also why map/fold had to be native Rust-loop builtins rather than
    // ordinary renno functions written with `let rec` + `match`.
    Cons,
}

// The runtime test an `Expr::Check` applies to a value (machine::test_holds),
// derived from the target type. It follows the VALUE (finite, immutable,
// acyclic), so a deep test of a function-free type always terminates. A
// function position is only a callability tag (the per-call contract is
// typecheck::wrap_fun_contract's job, and only for a bare Fun target).
#[derive(Debug, Clone, PartialEq)]
pub enum Test {
    // Dyn/Var: everything passes. Never: nothing does (a failing branch, or a
    // Named alias already being unfolded with no container in between).
    Any,
    Never,
    Int,
    Float,
    Bool,
    Str,
    // A list of anything (`ListOf(Any)`, kept as its own fast variant: the
    // is_list prelude predicate and the Vec(n) length checks use it).
    List,
    Fun,
    // The exact Value::Token with this id.
    Token(u64),
    // A list every element of which passes.
    ListOf(Box<Test>),
    // A list of exactly this many elements, each passing (a Vec(k) with a literal k).
    ListLen(Box<Test>, usize),
    // A list of exactly this many elements, position i passing tests[i].
    TupleOf(Vec<Test>),
    // A record with at least these fields, each field's value passing its
    // test (width-tolerant: other fields are never looked at).
    RecordOf(Vec<(String, Test)>),
    // `CheckSpec::defs[i]`: a Named alias's body, reached only through a
    // container so the recursion consumes value structure.
    Ref(usize),
    // Any alternative passes (a union, first match wins).
    Or(Vec<Test>),
}

// What Expr::Check does with the test's outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CheckMode {
    // Return the value, or panic `type error: expected {to}, found {type}`.
    Assert,
    // Return the Bool outcome (a union picking its alternative).
    Probe,
}

#[derive(Debug)]
pub struct CheckSpec {
    pub test: Test,
    // The bodies `Test::Ref(i)` indexes; empty unless the target reaches a
    // recursive Named alias.
    pub defs: Vec<Test>,
    pub mode: CheckMode,
    // The type's display text, for the Assert message.
    pub to: String,
    // The cast site a failure blames (the node itself has no span in the
    // parser's SpanMap); None keeps whatever span ran last.
    pub span: Option<Span>,
}

#[derive(Debug, Clone)]
pub enum Expr {
    // `e` tested natively against `spec` -- evaluates `e` once, no effects,
    // no env frame. Typecheck-synthesized only (a Dyn-boundary check);
    // never written by the parser.
    Check(ExprRef, Rc<CheckSpec>),
    Int(i64),
    // A second numeric literal kind, distinct from Int -- see types::
    // Type::Float's own doc comment for how far the Int/Float interop
    // goes (arithmetic operators only, not general type consistency).
    Float(f64),
    Bool(bool),
    Str(String),
    // A literal Value::Token(u64), unique per SOURCE POSITION (this u64
    // is that position) -- the same value every time this exact node
    // evaluates, no matter how many times (a plain literal, not a
    // generator). Produced by the surface `opaque` expression (parser::
    // atom_leaf, any time a user writes it) -- typed Type::Token(this id)
    // by elaborate_node, so a tuple/field containing one gets real
    // nominal identity from ordinary structural comparison, no separate
    // brand-comparison mechanism needed.
    Token(u64),
    // `(a, b, c)` -- a fixed-arity product, elaborated with types::
    // Type::Tuple (per-position types, no widening, unlike ListLit just
    // below). Evaluates to a plain Value::List at runtime (machine.rs
    // reuses ListLit's own Frame::ListElems machinery unchanged) -- no
    // new Value kind needed. Always 2+ items: parser::atom_leaf's LParen
    // arm only builds this once it's seen a comma: a single parenthesized
    // expression with no comma stays ordinary grouping.
    Tuple(Vec<ExprRef>),
    // `{y: 2, x: 1}` -- fields ALWAYS sorted by name by the parser (see
    // Type::Record's own doc comment), though order is no longer
    // semantically significant once evaluated: elaborate_node's own
    // Record arm infers types::Type::Record from it (elaborating each
    // field value, unlike Tuple's own sibling arm this is NOT rewritten
    // away -- see Value::Record's own doc comment for why records need
    // their own name-keyed runtime kind), and machine.rs evaluates it
    // directly via Frame::RecordElems into a Value::Record.
    Record(Rc<Vec<(String, ExprRef)>>),
    // `p.x` -- reads field `x` off a record. Parser-emitted and
    // typecheck-consumed only: elaborate_node's own FieldAccess arm
    // desugars it into an ordinary `get_field(target, "x")` call (see
    // Builtin::GetField's own doc comment) after checking what it can
    // statically -- machine.rs never evaluates a FieldAccess node
    // directly, the same "parser/typecheck marker, rewritten away before
    // it can reach the runtime loop" shape Expr::Record itself used to
    // have (back when it rewrote to Expr::Tuple), except FieldAccess
    // never graduates out of that shape the way Record did, since
    // reading one field needs no evaluation semantics beyond an ordinary
    // two-argument call.
    FieldAccess(ExprRef, String),
    // Variable-arity, unlike every other node -- elaborate/machine handle
    // it with a loop over the Vec rather than a fixed-shape match.
    ListLit(Vec<ExprRef>),
    Var(String),
    // param annotation is optional -- None means Dyn. The typechecker fills
    // this in (or leaves it) when elaborating; the parser fills it in only
    // when the source has an explicit `: Type` annotation.
    Lambda(String, Option<Type>, ExprRef),
    App(ExprRef, ExprRef),
    // Plain, non-recursive `let var = val in body` -- `var` is NOT in
    // scope while `val` is being evaluated. See LetRec for `let rec`.
    Let(String, Option<Type>, ExprRef, ExprRef),
    // `let rec f = val_f [and g = val_g ...] in body` -- one or more
    // SIMULTANEOUSLY recursive bindings, each visible to every other
    // one's value (and its own), not just sequentially in scope like Let.
    // Recursive only when the group is "direct": every value is
    // SYNTACTICALLY a Lambda (resolve::is_direct_group -- a purely
    // syntactic test, not "evaluates to a function"; an `if` whose every
    // branch is a Lambda does NOT qualify). Only then does machine.rs wrap
    // the whole group into mutually-referencing Value::RecClosure values so
    // each can call any sibling (including itself) by name (see
    // Value::RecClosure's doc comment for how that avoids needing
    // mutation or an AST-rewriting Y-combinator encoding); otherwise the
    // values evaluate in the outer scope and the names are bound only for
    // `body`, non-recursively (see resolve.rs's LetRec arm and machine.rs's
    // Frame::LetRecBody fallback). A single binding (`let rec f = ... in
    // ...`, no `and`) is just the length-1 case of the direct-group form --
    // no separate representation for plain self-recursion.
    LetRec(Rc<Vec<(String, Option<Type>, ExprRef)>>, ExprRef),
    BinOp(BinOp, ExprRef, ExprRef),
    // cond must evaluate to Bool.
    If(ExprRef, ExprRef, ExprRef),
    Perform(String, ExprRef),
    // `handler` evaluates to a Value::Handler. Handle itself carries no
    // deep/shallow flag -- that lives on the handler value, set by the
    // `deep`/`shallow` builtins (see env::PRELUDE, indexed at runtime via
    // resolve::VarRef::Prelude), not on this AST node.
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
    // `match scrutinee | p1 -> e1 | p2 if g -> e2 ...`, tried top to
    // bottom, first match wins. Arms in Rc (not a plain Vec, unlike
    // ListLit) since a captured continuation can carry a MatchArms frame
    // (see cont.rs) -- resuming it more than once (multi-shot) would
    // otherwise reclone the whole arm list on every resume. The middle
    // `Option<ExprRef>` is an optional `if` guard: when present, the arm
    // is only taken if the pattern matches AND the guard evaluates to
    // true, otherwise matching falls through to the next arm as if this
    // one's pattern hadn't matched at all (see machine.rs's Frame::
    // MatchGuard). A guard may never `perform` -- enforced statically by
    // parser::contains_perform, not by this type -- see the guard
    // check in parser.rs for why (multi-shot resume replaying match-arm
    // selection itself).
    Match(ExprRef, Rc<Vec<(Pattern, Option<ExprRef>, ExprRef)>>),
}
