use std::collections::HashMap;
use std::rc::Rc;

use cranelift_entity::{entity_impl, EntityRef, PrimaryMap};

use crate::list::SealId;
use crate::span::Span;
use crate::index_expr::IndexExpr;
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
// synthesizes (a Conform, a re-emitted Let with a
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

// The runtime test a boundary applies to a value (machine::test_holds),
// derived from the target type. It follows the VALUE (finite, immutable,
// acyclic), so a deep test of a function-free type always terminates. A
// function position is only a callability tag (the per-call contract is
// the Conform's `Then`, and only for a Fun target or a union alternative).
#[derive(Debug, Clone, PartialEq)]
pub enum Test {
    // Dyn/Var: everything passes. Never: nothing does (a failing branch, or a
    // recursive type already being unfolded with no container in between).
    Any,
    Never,
    Int,
    Float,
    Bool,
    Str,
    // A list of anything (`ListOf(Any)`, kept as its own fast variant: the
    // `is(List, v)` and the Vec(n) length checks use it).
    List,
    Fun,
    // The exact Value::Token with this id.
    Token(u64),
    // A list of the given length (`Length`), every element of which passes `elem`.
    // `seal` names the every-element test for the list blocks' seals (None: never sealed).
    Elems { elem: Box<Test>, len: Length, seal: Seal },
    // A list of exactly this many elements, position i passing tests[i].
    TupleOf(Vec<Test>),
    // A record with at least these fields, each field's value passing its
    // test (width-tolerant: other fields are never looked at).
    RecordOf(Vec<(String, Test)>),
    // `CheckSpec::defs[i]`: a recursive type's body, reached only through a
    // container so the recursion consumes value structure. `seal` names the whole
    // test for a list or tuple block's exact-view seal.
    Ref(usize, Seal),
    // Any alternative passes (a union, first match wins).
    Or(Vec<Test>),
}

// How long an `Elems` list must be.
#[derive(Debug, Clone, PartialEq)]
pub enum Length {
    // Any length (a plain `List(T)`).
    Any,
    // Exactly this many elements (a Vec(k) with a literal k).
    Exact(usize),
    // As long as `lens[slot]` says (the values of the owning Lowered's `slots`, read when
    // the check is entered): a Vec(n) with a non-literal n.
    Slot(usize),
}

// The identity of a test for the list blocks' seals (list.rs), or None: this test is
// never sealed. Derived data, a cache key and not meaning, so it is invisible to equality
// (every two Seals are equal) and to Debug: structural assertions on tests and snapshots do
// not depend on the interner's order.
#[derive(Clone, Copy, Default)]
pub struct Seal(pub Option<SealId>);

impl PartialEq for Seal {
    fn eq(&self, _: &Seal) -> bool {
        true
    }
}

impl std::fmt::Debug for Seal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("_")
    }
}

// The constructors of tests that carry no seal yet (assign_seals fills them in).
impl Test {
    pub fn list_of(elem: Box<Test>) -> Test {
        Test::Elems { elem, len: Length::Any, seal: Seal::default() }
    }

    pub fn list_len(elem: Box<Test>, len: usize) -> Test {
        Test::Elems { elem, len: Length::Exact(len), seal: Seal::default() }
    }

    pub fn list_len_arg(elem: Box<Test>, slot: usize) -> Test {
        Test::Elems { elem, len: Length::Slot(slot), seal: Seal::default() }
    }

    pub fn reference(def: usize) -> Test {
        Test::Ref(def, Seal::default())
    }
}

// ---- Seal ids: structural, interned, lens-free (spec 21) ----
//
// A seal on a list block says "this view passes test #id" (list.rs). The id must be equal
// exactly when the two tests mean the same, and must not keep any test alive, so it is a
// structural key interned per thread: the interner holds keys (integers and names) and never
// a Test, CheckSpec or Lazy. Keys are built bottom-up, each node once over its children's
// ids, so assigning ids to a test is linear in its size.
//
// Per thread is sound because every value a seal can sit on is not Send (list blocks hold
// Rc, and so do the CheckSpecs and Lazys that carry ids): a sealed block and a test carrying
// an id always live on the thread that made both, so both drew their ids from that thread's
// interner (check-time evaluations and the run share one thread, lib.rs
// run_source_on_this_thread). If Rc ever became Arc this interner would have to be global.
//
// Equal id means equal meaning:
//   - the id of an every-element test is `Every(rel(element test))`;
//   - `rel` of a node is its tag, its scalar payload (a Token's id, a literal length, record
//     field names in order) and its children's rels in order, and `rel(Ref(i))` is
//     `RefLocal(i)`;
//   - a node that reaches a `Ref` is rooted: its final id is `Rooted(G, rel)`, where G is
//     the interned table of every def body's rel, so two such nodes are equal only under the
//     same defs;
//   - a test whose verdict reads a length slot gets no id (the slot's value changes from
//     call to call): any node with an `Elems { len: Slot }` below it, and, if any def has one,
//     every node that reaches a `Ref` (conservative). The slotted list's own length is
//     checked natively outside the seal, so `Vec(n)` itself is sealed while `List(Vec(n))`
//     is not.

// The most keys one thread interns: past it a new test gets no id and behaves as before
// (bounds the growth from run-time lowerings of ever-new types).
const SEAL_CAP: usize = 65_536;

const TAG_ANY: u8 = 0;
const TAG_NEVER: u8 = 1;
const TAG_INT: u8 = 2;
const TAG_FLOAT: u8 = 3;
const TAG_BOOL: u8 = 4;
const TAG_STR: u8 = 5;
const TAG_LIST: u8 = 6;
const TAG_FUN: u8 = 7;
const TAG_TOKEN: u8 = 8;
const TAG_ELEMS_ANY: u8 = 9;
const TAG_ELEMS_EXACT: u8 = 10;
const TAG_TUPLE: u8 = 11;
const TAG_RECORD: u8 = 12;
const TAG_OR: u8 = 13;
const TAG_REF: u8 = 14;
const TAG_EVERY: u8 = 15;
const TAG_TABLE: u8 = 16;
const TAG_ROOTED: u8 = 17;

#[derive(PartialEq, Eq, Hash)]
struct Key {
    tag: u8,
    num: u64,
    names: Vec<String>,
    kids: Vec<u32>,
}

struct Interner {
    keys: crate::types::FxMap<Key, u32>,
    cap: usize,
}

thread_local! {
    static INTERNER: std::cell::RefCell<Interner> = std::cell::RefCell::new(Interner { keys: Default::default(), cap: SEAL_CAP });
}

// The id of `key`, or None when the interner is full.
fn intern(tag: u8, num: u64, names: Vec<String>, kids: Vec<u32>) -> Option<u32> {
    let key = Key { tag, num, names, kids };
    INTERNER.with(|i| {
        let mut i = i.borrow_mut();
        if let Some(&id) = i.keys.get(&key) {
            return Some(id);
        }
        if i.keys.len() >= i.cap {
            return None;
        }
        let id = u32::try_from(i.keys.len() + 1).ok()?;
        i.keys.insert(key, id);
        Some(id)
    })
}

#[cfg(test)]
pub(crate) fn set_seal_cap(cap: usize) {
    INTERNER.with(|i| i.borrow_mut().cap = cap);
}

// What a subtree contributes to its parent's key.
struct Info {
    // The structural id of the node (None: it has none, see above).
    rel: Option<u32>,
    // Whether the node reaches a `Ref`.
    refs: bool,
    // Whether its verdict reads a length slot.
    lens: bool,
}

// Assigns the seal ids of `test`, of the `defs` it reaches, and of every alternative test of
// `then` (they all share the defs). A hand-built test that never goes through here carries
// no ids and behaves as if seals did not exist.
pub fn assign_seals(test: &mut Test, defs: &mut [Test], then: &mut Then) {
    let mut ctx = SealCtx { top: Default::default(), table: None, def_lens: false };
    // Pass 1: each def's rel and whether any contains a slot. A Ref's rel does not depend on
    // the table, so the rels are complete before it exists.
    let mut rels = Some(Vec::new());
    for def in defs.iter_mut() {
        let info = ctx.go(def, false);
        ctx.def_lens |= info.lens;
        rels = rels.zip(info.rel).map(|(mut v, r)| {
            v.push(r);
            v
        });
    }
    // No defs, no Ref to root: skip the table key (the common `is(List(Int))` lowering).
    ctx.table = if defs.is_empty() { None } else { rels.and_then(|kids| intern(TAG_TABLE, 0, Vec::new(), kids)) };
    // Pass 2 writes the final ids.
    for def in defs.iter_mut() {
        ctx.go(def, true);
    }
    ctx.top.set(true);
    ctx.go(test, true);
    ctx.alternatives(then);
}

struct SealCtx {
    // The next `go` is on a root test, whose own structural key nobody reads (no parent).
    top: std::cell::Cell<bool>,
    // G: the interned table of the def bodies (None: some def has no id, or there are none).
    table: Option<u32>,
    // Some def reads a length slot: every node that reaches a Ref gets no id.
    def_lens: bool,
}

impl SealCtx {
    fn alternatives(&self, then: &mut Then) {
        if let Then::Pick(alts) = then {
            for (test, then) in alts {
                self.top.set(true);
                self.go(test, true);
                self.alternatives(then);
            }
        }
    }

    // A rooted id: nodes that reach a Ref mean something only under the defs.
    fn root(&self, id: u32, refs: bool) -> Option<u32> {
        match (refs, self.def_lens, self.table) {
            (false, _, _) => Some(id),
            (true, false, Some(g)) => intern(TAG_ROOTED, u64::from(g), Vec::new(), vec![id]),
            _ => None,
        }
    }

    fn leaf(&self, tag: u8, num: u64) -> Info {
        Info { rel: intern(tag, num, Vec::new(), Vec::new()), refs: false, lens: false }
    }

    // Combines the children's infos under one tag.
    fn node(&self, tag: u8, names: Vec<String>, kids: Vec<Info>) -> Info {
        let (refs, lens) = (kids.iter().any(|k| k.refs), kids.iter().any(|k| k.lens));
        let rels: Option<Vec<u32>> = kids.iter().map(|k| k.rel).collect();
        let rel = if lens { None } else { rels.and_then(|rels| intern(tag, 0, names, rels)) };
        Info { rel, refs, lens }
    }

    // Bottom-up over `t`; with `write`, stores each list and Ref node's id.
    fn go(&self, t: &mut Test, write: bool) -> Info {
        let top = self.top.replace(false);
        match t {
            Test::Any => self.leaf(TAG_ANY, 0),
            Test::Never => self.leaf(TAG_NEVER, 0),
            Test::Int => self.leaf(TAG_INT, 0),
            Test::Float => self.leaf(TAG_FLOAT, 0),
            Test::Bool => self.leaf(TAG_BOOL, 0),
            Test::Str => self.leaf(TAG_STR, 0),
            Test::List => self.leaf(TAG_LIST, 0),
            Test::Fun => self.leaf(TAG_FUN, 0),
            Test::Token(id) => self.leaf(TAG_TOKEN, *id),
            Test::Elems { elem, len, seal } => {
                let e = self.go(elem, write);
                if write {
                    // The every-element id ignores the length (checked natively each time).
                    let id = if e.lens { None } else { e.rel.and_then(|r| intern(TAG_EVERY, 0, Vec::new(), vec![r])) };
                    seal.0 = id.and_then(|id| self.root(id, e.refs)).and_then(SealId::new);
                }
                let lens = e.lens || matches!(len, Length::Slot(_));
                let rel = match (len, e.rel) {
                    _ if lens || top => None,
                    (Length::Any, Some(r)) => intern(TAG_ELEMS_ANY, 0, Vec::new(), vec![r]),
                    (Length::Exact(k), Some(r)) => intern(TAG_ELEMS_EXACT, *k as u64, Vec::new(), vec![r]),
                    _ => None,
                };
                Info { rel, refs: e.refs, lens }
            }
            Test::TupleOf(ts) => {
                let kids = ts.iter_mut().map(|t| self.go(t, write)).collect();
                self.node(TAG_TUPLE, Vec::new(), kids)
            }
            Test::RecordOf(fields) => {
                let names = fields.iter().map(|(name, _)| name.clone()).collect();
                let kids = fields.iter_mut().map(|(_, t)| self.go(t, write)).collect();
                self.node(TAG_RECORD, names, kids)
            }
            Test::Or(alts) => {
                let kids = alts.iter_mut().map(|t| self.go(t, write)).collect();
                self.node(TAG_OR, Vec::new(), kids)
            }
            Test::Ref(i, seal) => {
                let lens = self.def_lens;
                let rel = if lens { None } else { intern(TAG_REF, *i as u64, Vec::new(), Vec::new()) };
                if write {
                    seal.0 = rel.and_then(|r| self.root(r, true)).and_then(SealId::new);
                }
                Info { rel, refs: true, lens }
            }
        }
    }
}

#[derive(Debug)]
pub struct CheckSpec {
    pub test: Test,
    // The bodies `Test::Ref(i)` indexes; empty unless the target reaches a
    // recursive type.
    pub defs: Vec<Test>,
    // The target type, written (capped, Type::brief) only if a check fails, for the failure
    // message `type error: expected {to}, found {type}`.
    pub to: Type,
    // The cast site a failure blames (the node itself has no span in the
    // parser's SpanMap); None keeps whatever span ran last.
    pub span: Option<Span>,
}

// A length slot's term: an index expression as the run-time check reads it.
// Evaluating one is total (machine::eval_len), so a Conform evaluates its slots
// on entry whatever the value turns out to be.
#[derive(Debug, Clone, PartialEq)]
pub enum LenExpr {
    Lit(i64),
    // The length of the list a hidden alias holds: an Expr::Var node naming
    // the alias, and the index variable it witnesses (for the message).
    Witness(ExprRef, Rc<str>),
    Add(Box<LenExpr>, Box<LenExpr>),
    Sub(Box<LenExpr>, Box<LenExpr>),
    Mul(Box<LenExpr>, Box<LenExpr>),
    // Any length: an isolated unconstrained variable, so only "is a list".
    Any,
    // An index variable nothing at run time holds: a list fails the check.
    NoValue(Rc<str>),
}

// A Conform's length slot: shared with the obligation that patches it.
pub type LenSlot = Rc<std::cell::RefCell<LenExpr>>;

impl LenExpr {
    // Appends the witness Var nodes this term reads to `out`.
    pub fn witnesses(&self, out: &mut Vec<ExprRef>) {
        match self {
            LenExpr::Witness(var, _) => out.push(*var),
            LenExpr::Add(a, b) | LenExpr::Sub(a, b) | LenExpr::Mul(a, b) => {
                a.witnesses(out);
                b.witnesses(out);
            }
            LenExpr::Lit(_) | LenExpr::Any | LenExpr::NoValue(_) => {}
        }
    }
}

// A length slot's value, read when its Conform is entered.
#[derive(Debug, Clone, PartialEq)]
pub enum Len {
    Exact(usize),
    Any,
    // No list is this long: the term is negative or overflows.
    Never,
    NoValue(Rc<str>),
}

// What the check-time evaluation of a kept annotation found (spec 11.2).
#[derive(Debug)]
pub enum AnnotKind {
    // It finished with this type. Unless the checker erases the site, the
    // run time confirms it: a definition it ran, or the elaborated
    // annotation, holds a check that could fail.
    Confirm(Rc<Type>),
    // It ran out of fuel: reaching the site is the fuel error.
    Diverged,
    // It names a value only the run time has: the run time evaluates it
    // every time the site is reached, and an Expr::Conform checks the
    // binder against the result.
    Unknown,
}

// Where a boundary's type comes from (P3).
#[derive(Debug)]
pub enum TypeSource {
    // Known at check time: lowered then (the Lazy's `lowered` cell is already
    // filled). The root Lazy is held, not just its Lowered, so a recursive
    // type's family stays alive and `F = Fun(Dyn, F)` lowers once.
    Static(Rc<Lazy>),
    // An Unknown site (spec 11.2): the type is the value the Var node `ty`
    // names (the site's annotation binder), lowered when reached
    // (typecheck::lower_conform); the cell memoizes the Lazy of the type it
    // last saw.
    Site(ExprRef, std::cell::RefCell<Option<Rc<Lazy>>>),
}

// A boundary's check (spec 11.2, 11.6; P3): Expr::Conform tests its operand
// against the type its `source` gives, with today's run-time check messages.
// One node for every boundary: a function-free shape (a native test, a
// `Then::Done`), a function or a union with a function alternative (a
// contract), and an Unknown site's run-time type.
#[derive(Debug)]
pub struct ConformSpec {
    pub source: TypeSource,
    // The index variable names the type's text writes (a static boundary: the
    // ones its lowered slots read), and for each the length slot holding its
    // run-time value (spec 11.4). A slot is patched in place once inference has
    // finished (typecheck::resolve_obligations), so it is shared and mutable;
    // the witness Var nodes it reads are children of the node.
    pub index_names: Vec<String>,
    pub lens: Vec<LenSlot>,
    // The source span a failure blames (None keeps whatever span ran last).
    pub span: Option<Span>,
    // The site's own binder is typed to the checker (a static boundary), so
    // callers already enforce a function's parameter: its contract checks
    // only callability and the result, not each argument.
    pub trusted: bool,
}

// A type value's lowering (None: nothing to check), made the first time a
// value is checked against it and shared by every use of this Lazy. The memo
// of an Unknown site holds the Lazy of the type it last saw (holding the Rc
// keeps the type's address from being reused by another type), and a Lowered's
// function contract holds one for its parameter and one for its result.
// Those are drawn from the root's `family`: a type already in it is reused,
// so a recursive type (`let rec F = Fun(Dyn, F)`) lowers to a finite cyclic graph
// once, not a fresh chain per call. The family lists its members weakly, so
// only the graph itself keeps them alive.
#[derive(Debug)]
pub struct Lazy {
    pub ty: Rc<Type>,
    pub lowered: std::cell::OnceCell<Option<Rc<Lowered>>>,
    family: Rc<std::cell::RefCell<Vec<std::rc::Weak<Lazy>>>>,
}

impl Lazy {
    pub fn new(ty: Rc<Type>) -> Rc<Lazy> {
        let family = Rc::new(std::cell::RefCell::new(Vec::new()));
        Lazy::join(&family, ty)
    }

    // The Lazy for `ty` in this one's family: the member holding an equal
    // type, else a new member.
    pub fn child(&self, ty: Rc<Type>) -> Rc<Lazy> {
        Lazy::join(&self.family, ty)
    }

    fn join(family: &Rc<std::cell::RefCell<Vec<std::rc::Weak<Lazy>>>>, ty: Rc<Type>) -> Rc<Lazy> {
        let mut members = family.borrow_mut();
        members.retain(|member| member.strong_count() > 0);
        if let Some(found) = members.iter().filter_map(|member| member.upgrade()).find(|member| *member.ty == *ty) {
            return found;
        }
        let lazy = Rc::new(Lazy { ty, lowered: std::cell::OnceCell::new(), family: family.clone() });
        members.push(Rc::downgrade(&lazy));
        lazy
    }
}

// A type as checks: a native Check spec, for
// each length slot its tests read the index whose value fills it, and what a
// value passing `check` still needs.
#[derive(Debug)]
pub struct Lowered {
    pub check: Rc<CheckSpec>,
    pub slots: Vec<IndexExpr>,
    pub then: Then,
}

// What a value needs beyond a Lowered's native test (typecheck::boundary's
// Fun and Union cases).
#[derive(Debug)]
pub enum Then {
    // Nothing more.
    Done,
    // The site's contract, with these parameter and result types.
    Wrap(Rc<Lazy>, Rc<Lazy>),
    // The first alternative whose test holds (against the Lowered's defs and
    // slots), then what that alternative needs; none holds: the Lowered's
    // check fails.
    Pick(Vec<(Test, Then)>),
}

#[derive(Debug)]
pub struct AnnotSpec {
    pub kind: AnnotKind,
    // A Confirm site that has evaluated once: later executions yield its type.
    pub done: std::cell::Cell<bool>,
    // The checker found the site cannot fail and removes it (typecheck's
    // settle_erasure); never read at run time.
    pub erased: std::cell::Cell<bool>,
    // The annotation's source span, which an evaluation failure blames.
    pub span: Span,
    // How many times the annotation was evaluated (the Confirm-once test).
    #[cfg(test)]
    pub evals: std::cell::Cell<u32>,
}

impl AnnotSpec {
    pub fn new(kind: AnnotKind, span: Span) -> AnnotSpec {
        AnnotSpec {
            kind,
            done: std::cell::Cell::new(false),
            erased: std::cell::Cell::new(false),
            span,
            #[cfg(test)]
            evals: std::cell::Cell::new(0),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    // A kept annotation's value (spec 11.2): the elaborated annotation,
    // evaluated in check mode to a Value::Type. Typecheck binds it to an
    // unlexable name just before the annotated binder.
    Annot(ExprRef, Rc<AnnotSpec>),
    // A boundary (spec 11.2, 11.6; P3): the operand conformed to a type,
    // natively. The type is known at check time (a static boundary, lowered
    // then) or is the value an Unknown site's annotation variable holds
    // (lowered when reached). Evaluates the operand once, no effects, no env
    // frame. Typecheck-synthesized only; never written by the parser.
    Conform(ExprRef, Rc<ConformSpec>),
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
    // reuses ListLit's own Iter::Elems machinery unchanged) -- no
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
    // directly via Iter::Elems into a Value::Record.
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
    // A type value as a literal (it evaluates to Value::Type). The parser
    // writes one where an annotation names something that is not an
    // expression yet: a symbolic index (the `n` of
    // `Vec(n)` or `Indexed(T, n + 1)`, passed as the type Vec(n));
    // typecheck writes one for every annotation it re-emits.
    TypeLit(Rc<Type>),
    // The annotation slots (param, `let`, `let rec`) hold an EXPRESSION
    // (machine::eval_pure turns it into a Type at check time); None means
    // Dyn. The parser fills one in only when the source has an explicit
    // `: Type` annotation, desugared into calls to the prelude type
    // constructors; the typechecker re-emits every binder with a TypeLit.
    Lambda(String, Option<ExprRef>, ExprRef),
    App(ExprRef, ExprRef),
    // Plain, non-recursive `let var = val in body` -- `var` is NOT in
    // scope while `val` is being evaluated. See LetRec for `let rec`.
    Let(String, Option<ExprRef>, ExprRef, ExprRef),
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
    // Iter::LetRec fallback). A single binding (`let rec f = ... in
    // ...`, no `and`) is just the length-1 case of the direct-group form --
    // no separate representation for plain self-recursion.
    LetRec(Rc<Vec<(String, Option<ExprRef>, ExprRef)>>, ExprRef),
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

impl Expr {
    // Appends this node's direct sub-expressions to `out`, annotations included.
    pub fn children(&self, out: &mut Vec<ExprRef>) {
        self.children_with(out, true);
    }

    // `children`, annotation slots only when `annotations`.
    pub(crate) fn children_with(&self, out: &mut Vec<ExprRef>, annotations: bool) {
        match self {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Token(_) | Expr::Var(_) | Expr::TypeLit(_) => {}
            Expr::Conform(e, spec) => {
                out.push(*e);
                if let TypeSource::Site(ty, _) = &spec.source {
                    out.push(*ty);
                }
                for slot in &spec.lens {
                    slot.borrow().witnesses(out);
                }
            }
            Expr::FieldAccess(e, _) | Expr::Perform(_, e) | Expr::Annot(e, _) => out.push(*e),
            Expr::Lambda(_, ann, body) => {
                out.extend(ann.filter(|_| annotations));
                out.push(*body);
            }
            Expr::MakeHandler { body, .. } => out.push(*body),
            Expr::Tuple(items) | Expr::ListLit(items) => out.extend(items),
            Expr::Record(fields) => out.extend(fields.iter().map(|(_, v)| *v)),
            Expr::App(a, b) | Expr::BinOp(_, a, b) => out.extend([*a, *b]),
            Expr::Let(_, ann, val, body) => {
                out.extend(ann.filter(|_| annotations));
                out.extend([*val, *body]);
            }
            Expr::LetRec(bindings, body) => {
                for (_, ann, v) in bindings.iter() {
                    out.extend(ann.filter(|_| annotations));
                    out.push(*v);
                }
                out.push(*body);
            }
            Expr::If(c, t, e) => out.extend([*c, *t, *e]),
            Expr::Handle { body, handler } => out.extend([*body, *handler]),
            Expr::Match(scrutinee, arms) => {
                out.push(*scrutinee);
                for (_, guard, body) in arms.iter() {
                    out.extend(*guard);
                    out.push(*body);
                }
            }
        }
    }
}

// A copy of the parsed tree rooted at `root`. Every reachable node is pushed
// again, children mapped to their copies (a node reached twice is copied once,
// so the copy shares as the original does), except a node `replace` answers
// for: it returns the node to put in its place (it may push one), and the
// subtree under the original is neither visited nor copied. Annotation slots
// are not copied either: neither the resolver nor the check-mode machine
// enters them, so the copy's point at the originals. Iterative, so a deep chain
// costs no native stack. Check-time evaluation runs the copy
// (machine::eval_pure_in), so each evaluation has nodes of its own; `replace`
// is where it reads an already-evaluated nested definition's value instead.
pub fn copy_tree(arena: &mut Arena, root: ExprRef, mut replace: impl FnMut(&mut Arena, ExprRef) -> Option<ExprRef>) -> ExprRef {
    let mut copies: HashMap<ExprRef, ExprRef> = HashMap::new();
    let mut stack = vec![(root, false)];
    let mut kids = Vec::new();
    while let Some((n, expanded)) = stack.pop() {
        if copies.contains_key(&n) {
            continue;
        }
        if !expanded && let Some(r) = replace(arena, n) {
            copies.insert(n, r);
            continue;
        }
        if !expanded {
            stack.push((n, true));
            kids.clear();
            arena[n].children_with(&mut kids, false);
            stack.extend(kids.iter().filter(|k| !copies.contains_key(*k)).map(|k| (*k, false)));
            continue;
        }
        let at = |k: &ExprRef| copies[k];
        let node = match &arena[n] {
            Expr::Annot(..) | Expr::Conform(..) => unreachable!("a check-time evaluation runs a parsed tree"),
            Expr::Int(i) => Expr::Int(*i),
            Expr::Float(f) => Expr::Float(*f),
            Expr::Bool(b) => Expr::Bool(*b),
            Expr::Str(s) => Expr::Str(s.clone()),
            Expr::Token(t) => Expr::Token(*t),
            Expr::Var(v) => Expr::Var(v.clone()),
            Expr::TypeLit(t) => Expr::TypeLit(t.clone()),
            Expr::Tuple(items) => Expr::Tuple(items.iter().map(at).collect()),
            Expr::ListLit(items) => Expr::ListLit(items.iter().map(at).collect()),
            Expr::Record(fields) => Expr::Record(Rc::new(fields.iter().map(|(name, v)| (name.clone(), at(v))).collect())),
            Expr::FieldAccess(e, name) => Expr::FieldAccess(at(e), name.clone()),
            Expr::Lambda(param, ann, body) => Expr::Lambda(param.clone(), *ann, at(body)),
            Expr::App(f, a) => Expr::App(at(f), at(a)),
            Expr::Let(name, ann, val, body) => Expr::Let(name.clone(), *ann, at(val), at(body)),
            Expr::LetRec(bindings, body) => {
                Expr::LetRec(Rc::new(bindings.iter().map(|(name, ann, v)| (name.clone(), *ann, at(v))).collect()), at(body))
            }
            Expr::BinOp(op, a, b) => Expr::BinOp(*op, at(a), at(b)),
            Expr::If(c, t, e) => Expr::If(at(c), at(t), at(e)),
            Expr::Perform(effect, e) => Expr::Perform(effect.clone(), at(e)),
            Expr::Handle { body, handler } => Expr::Handle { body: at(body), handler: at(handler) },
            Expr::MakeHandler { effect, payload_var, resume_var, body } => {
                Expr::MakeHandler { effect: effect.clone(), payload_var: payload_var.clone(), resume_var: resume_var.clone(), body: at(body) }
            }
            Expr::Match(scrutinee, arms) => Expr::Match(
                at(scrutinee),
                Rc::new(arms.iter().map(|(pat, guard, body)| (pat.clone(), guard.as_ref().map(at), at(body))).collect()),
            ),
        };
        let copy = arena.push(node);
        copies.insert(n, copy);
    }
    copies[&root]
}

// Gives every node typecheck synthesized (index >= spans.len()) the span of its children's
// hull, so spans.get(n) is Some for every node of the elaborated tree rooted at `root`. A
// synthesized leaf takes its parent's span; a node unreachable from `root` takes the root's.
// Parser-built entries are never changed.
pub fn fill_synthesized_spans(arena: &Arena, root: ExprRef, spans: &mut SpanMap) {
    let (original, total) = (spans.len(), arena.len());
    let mut span: Vec<Option<Span>> = (0..total).map(|i| spans.get(ExprRef::new(i)).copied()).collect();
    let mut visited = vec![false; total];
    let mut order = Vec::new(); // post-order: children before parents
    let mut stack = vec![(root, false)];
    let mut kids = Vec::new();
    while let Some((n, expanded)) = stack.pop() {
        if expanded {
            order.push(n);
            if span[n.index()].is_none() {
                kids.clear();
                arena[n].children(&mut kids);
                span[n.index()] = kids.iter().filter_map(|k| span[k.index()]).reduce(|a, b| Span { start: a.start.min(b.start), end: a.end.max(b.end) });
            }
        } else if !std::mem::replace(&mut visited[n.index()], true) {
            stack.push((n, true));
            kids.clear();
            arena[n].children(&mut kids);
            stack.extend(kids.iter().map(|k| (*k, false)));
        }
    }
    for &n in order.iter().rev() {
        let Some(parent) = span[n.index()] else { continue };
        kids.clear();
        arena[n].children(&mut kids);
        for k in &kids {
            span[k.index()].get_or_insert(parent);
        }
    }
    let fallback = span[root.index()].unwrap_or(Span { start: 0, end: 0 });
    for s in &span[original..] {
        spans.push(s.unwrap_or(fallback));
    }
    debug_assert_eq!(spans.len(), total);
}

#[cfg(test)]
mod seal_tests {
    use super::*;
    use crate::types::Hc;

    // The seal ids of a test in pre-order (None: no id), for the nodes that carry one.
    fn seals(t: &Test, out: &mut Vec<Option<SealId>>) {
        match t {
            Test::Elems { elem, seal, .. } => {
                out.push(seal.0);
                seals(elem, out);
            }
            Test::Ref(_, seal) => out.push(seal.0),
            Test::TupleOf(ts) | Test::Or(ts) => ts.iter().for_each(|t| seals(t, out)),
            Test::RecordOf(fs) => fs.iter().for_each(|(_, t)| seals(t, out)),
            _ => {}
        }
    }

    // The id of the root of `test` after assign_seals (None: it has none).
    fn root_id(mut test: Test, mut defs: Vec<Test>) -> Option<SealId> {
        assign_seals(&mut test, &mut defs, &mut Then::Done);
        let mut out = Vec::new();
        seals(&test, &mut out);
        out[0]
    }

    fn list_of(t: Test) -> Test {
        Test::list_of(Box::new(t))
    }

    #[test]
    fn equal_structures_get_equal_ids_wherever_they_are_lowered() {
        let ty = Type::List(Hc::new(Type::Int));
        let (a, defs) = crate::typecheck::lower_test(&ty).unwrap();
        let (b, _) = crate::typecheck::lower_test(&ty).unwrap();
        // A static boundary and an Unknown site lower through `lower`.
        let lazy = Lazy::new(Rc::new(ty));
        let lowered = crate::typecheck::lower_conform(&lazy, None).expect("a list is checked");
        let (mut ids, mut other) = (Vec::new(), Vec::new());
        seals(&a, &mut ids);
        seals(&lowered.check.test, &mut other);
        assert!(defs.is_empty() && ids[0].is_some());
        assert_eq!(ids, other);
        let mut again = Vec::new();
        seals(&b, &mut again);
        assert_eq!(ids, again);
        // The same list tests built by hand agree with the lowered one, and `Length` is no part
        // of an every-element id: a Vec(2) and a Vec(3) of Int share the element test.
        assert_eq!(root_id(list_of(Test::Int), vec![]), ids[0]);
        assert_eq!(root_id(Test::list_len(Box::new(Test::Int), 2), vec![]), ids[0]);
        assert_eq!(root_id(Test::list_len(Box::new(Test::Int), 3), vec![]), ids[0]);
        assert_eq!(root_id(Test::list_len_arg(Box::new(Test::Int), 0), vec![]), ids[0], "Vec(n) itself is sealed");
    }

    #[test]
    fn different_structures_get_different_ids() {
        let id = |t: Test| root_id(t, vec![]);
        let distinct = [
            list_of(Test::Int),
            list_of(Test::Str),
            list_of(Test::Token(1)),
            list_of(Test::Token(2)),
            list_of(Test::list_len(Box::new(Test::Int), 2)),
            list_of(Test::list_len(Box::new(Test::Int), 3)),
            list_of(Test::list_of(Box::new(Test::Int))),
            list_of(Test::TupleOf(vec![Test::Int, Test::Str])),
            list_of(Test::TupleOf(vec![Test::Str, Test::Int])),
            list_of(Test::TupleOf(vec![Test::Int])),
            list_of(Test::RecordOf(vec![("a".into(), Test::Int)])),
            list_of(Test::RecordOf(vec![("b".into(), Test::Int)])),
            list_of(Test::RecordOf(vec![("a".into(), Test::Int), ("b".into(), Test::Int)])),
            list_of(Test::Or(vec![Test::Int, Test::Str])),
            list_of(Test::Or(vec![Test::Str, Test::Int])),
        ];
        let ids: Vec<_> = distinct.iter().cloned().map(id).collect();
        for (i, a) in ids.iter().enumerate() {
            assert!(a.is_some(), "{:?}", distinct[i]);
            for (j, b) in ids.iter().enumerate() {
                assert_eq!(i == j, a == b, "{:?} vs {:?}", distinct[i], distinct[j]);
            }
        }
    }

    // A node that reaches a Ref is rooted in the def table: the same test over different defs
    // has different ids, and the same defs give the same one.
    #[test]
    fn a_node_reaching_a_ref_is_rooted_in_its_defs() {
        let over = |def: Test| root_id(list_of(Test::reference(0)), vec![def]);
        let int_or_rec = |leaf: Test| Test::Or(vec![leaf, list_of(Test::reference(0))]);
        let (a, b, c) = (over(int_or_rec(Test::Int)), over(int_or_rec(Test::Str)), over(int_or_rec(Test::Int)));
        assert!(a.is_some() && b.is_some());
        assert_ne!(a, b);
        assert_eq!(a, c);
        // The Ref's own whole-seal id is rooted too, and differs from any every-element id.
        let mut test = Test::reference(0);
        let mut defs = vec![int_or_rec(Test::Int)];
        assign_seals(&mut test, &mut defs, &mut Then::Done);
        let Test::Ref(_, whole) = &test else { panic!("a Ref") };
        assert!(whole.0.is_some() && whole.0 != a);
    }

    #[test]
    fn a_test_that_reads_a_length_slot_gets_no_id() {
        let slotted = || Test::list_len_arg(Box::new(Test::Int), 0);
        // The slotted list itself is sealed; a list of slotted lists is not.
        assert!(root_id(slotted(), vec![]).is_some());
        assert_eq!(root_id(list_of(slotted()), vec![]), None);
        assert_eq!(root_id(list_of(Test::TupleOf(vec![Test::Int, slotted()])), vec![]), None);
        assert_eq!(root_id(list_of(Test::Or(vec![Test::Int, list_of(slotted())])), vec![]), None);
        // A def with a slot takes away the ids of everything reaching a Ref, and only those.
        let defs = || vec![Test::Or(vec![Test::Int, slotted()])];
        assert_eq!(root_id(list_of(Test::reference(0)), defs()), None);
        assert_eq!(root_id(Test::reference(0), defs()), None);
        assert!(root_id(list_of(Test::Int), defs()).is_some());
    }

    #[test]
    fn every_alternative_of_a_pick_gets_ids_and_hand_built_tests_get_none() {
        let hand = list_of(Test::Int);
        let mut none = Vec::new();
        seals(&hand, &mut none);
        assert_eq!(none, [None]);
        let mut test = Test::Never;
        let mut then = Then::Pick(vec![
            (list_of(Test::Int), Then::Done),
            (Test::Bool, Then::Pick(vec![(list_of(Test::Str), Then::Done)])),
        ]);
        assign_seals(&mut test, &mut [], &mut then);
        let Then::Pick(alts) = &then else { panic!("a Pick") };
        let mut out = Vec::new();
        seals(&alts[0].0, &mut out);
        let Then::Pick(inner) = &alts[1].1 else { panic!("a nested Pick") };
        seals(&inner[0].0, &mut out);
        assert!(out.len() == 2 && out.iter().all(Option::is_some) && out[0] != out[1]);
    }

    // Past the cap a test gets no id (it behaves as before); under it ids are assigned.
    #[test]
    fn the_interner_stops_at_its_cap() {
        set_seal_cap(0);
        assert_eq!(root_id(list_of(Test::Int), vec![]), None);
        set_seal_cap(2);
        // Int is key 1 and Every(Int) key 2: the inner list is sealed, but its parent's key
        // is the third and does not fit.
        let mut test = list_of(list_of(Test::Int));
        assign_seals(&mut test, &mut [], &mut Then::Done);
        let mut out = Vec::new();
        seals(&test, &mut out);
        assert_eq!(out.len(), 2);
        assert!(out[0].is_none() && out[1].is_some(), "{out:?}");
    }
}
