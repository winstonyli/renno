use std::cell::{Cell, RefCell};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::ops::Deref;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::LocalKey;

use crate::index_expr::{index_exprs_equal, IndexExpr};
use crate::util::find_field;

// ---- Shared nodes ----------------------------------------------------------------------
//
// Every compound type payload is an `Hc` (a hash-consed `Rc`): building the same structure
// twice gives the same node, so a type is a DAG whose size is its number of DISTINCT nodes
// however large its expansion, and equal types are pointer-equal. Each node carries a
// `Summary` of what its subtree holds (see below), computed once when the node is made.
//
// Node identity is exact: a node's key is its variant plus its children's pointers plus its
// inline data (a variable's name, a token's id, an effect row, an index expression, a `Rec`'s
// member names). `==` is the language's equality, which is coarser in one way (E1: recursive
// groups are equal up to member names), so `Hc::eq` is a pointer test, falling back to a
// structural comparison only when a `Rec` is inside, the one place equal types can be
// distinct nodes (they must stay distinct so each prints its own names).

// What a subtree contains. A summary may only PRUNE a walk ("this flag is absent, so the
// walker's 'nothing here' answer holds"); it never replaces a walker's answer, because some
// walkers stop at `Rec` or `Fun`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Summary {
    flags: u8,
    reach: u32,
    // The tree-expanded node count, saturating: tells a walker whether a node is big enough to
    // be worth remembering (below MEMO_MIN a walk costs less than the memo lookup).
    size: u16,
}

impl Summary {
    // A `Type::Var` appears.
    pub const VAR: u8 = 1;
    // A `Fun`'s row has a variable tail.
    pub const ROWVAR: u8 = 2;
    // An `Indexed` appears.
    pub const INDEXED: u8 = 4;
    // A `RecVar::Free` placeholder appears.
    pub const FREE: u8 = 8;
    // A `Rec` appears.
    pub const REC: u8 = 16;

    fn of(flags: u8) -> Summary {
        Summary { flags, reach: 0, size: 0 }
    }

    // The children's summaries joined: sizes add.
    fn join(self, other: Summary) -> Summary {
        Summary { flags: self.flags | other.flags, reach: self.reach.max(other.reach), size: self.size.saturating_add(other.size) }
    }

    // The summary of a node whose children's summaries are joined into `self`.
    fn node(self) -> Summary {
        Summary { size: self.size.saturating_add(1), ..self }
    }

    pub fn size(self) -> u16 {
        self.size
    }

    // Does any flag in `mask` appear?
    pub fn any(self, mask: u8) -> bool {
        self.flags & mask != 0
    }

    // The deepest `RecVar::Bound` reference plus one that points out of the subtree (0: no
    // reference leaves it, so the subtree is closed).
    pub fn reach(self) -> u32 {
        self.reach
    }
}

pub struct HcBox<T: Interned> {
    summary: Summary,
    value: T,
}

// The table of live nodes of one payload type: buckets by exact hash, holding weak
// references so a node dies with its last user. Swept when it has doubled since the last sweep.
pub struct Table<T: Interned> {
    buckets: HashMap<u64, Bucket<T>, BuildHasherDefault<KeyHasher>>,
    entries: usize,
    sweep_at: usize,
}

// The nodes sharing one hash: almost always a single one.
struct Bucket<T: Interned> {
    first: Weak<HcBox<T>>,
    more: Vec<Weak<HcBox<T>>>,
}

impl<T: Interned> Default for Table<T> {
    fn default() -> Table<T> {
        Table { buckets: HashMap::default(), entries: 0, sweep_at: 1024 }
    }
}

// The table's keys are already hashes.
#[derive(Default)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("keys are u64")
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

pub trait Interned: Sized + Default + 'static {
    fn table() -> &'static LocalKey<RefCell<Table<Self>>>;
    fn summary(&self) -> Summary;
    fn exact_eq(&self, other: &Self) -> bool;
    fn exact_hash(&self, h: &mut FxHasher);
}

pub struct Hc<T: Interned>(Rc<HcBox<T>>);

impl<T: Interned> Hc<T> {
    pub fn new(value: T) -> Hc<T> {
        let mut hasher = FxHasher::default();
        value.exact_hash(&mut hasher);
        let key = hasher.finish();
        T::table().with(|cell| {
            let mut table = cell.borrow_mut();
            let node = match table.buckets.entry(key) {
                Entry::Occupied(mut slot) => {
                    let bucket = slot.get();
                    for weak in std::iter::once(&bucket.first).chain(&bucket.more) {
                        if let Some(found) = weak.upgrade()
                            && found.value.exact_eq(&value)
                        {
                            return Hc(found);
                        }
                    }
                    let node = Rc::new(HcBox { summary: value.summary(), value });
                    slot.get_mut().more.push(Rc::downgrade(&node));
                    node
                }
                Entry::Vacant(slot) => {
                    let node = Rc::new(HcBox { summary: value.summary(), value });
                    slot.insert(Bucket { first: Rc::downgrade(&node), more: Vec::new() });
                    node
                }
            };
            table.entries += 1;
            if table.entries > table.sweep_at {
                table.buckets.retain(|_, bucket| {
                    bucket.more.retain(|w| w.strong_count() > 0);
                    if bucket.first.strong_count() == 0 && !bucket.more.is_empty() {
                        bucket.first = bucket.more.swap_remove(0);
                    }
                    bucket.first.strong_count() > 0
                });
                table.entries = table.buckets.values().map(|b| 1 + b.more.len()).sum();
                table.sweep_at = 2 * table.entries + 1024;
            }
            Hc(node)
        })
    }

    pub fn ptr_eq(a: &Hc<T>, b: &Hc<T>) -> bool {
        Rc::ptr_eq(&a.0, &b.0)
    }

    // The node's address, for exact hashing.
    fn addr(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    pub fn summary(&self) -> Summary {
        self.0.summary
    }
}

// A type's nodes own each other (a `Fun` its parameter and result), so a plain drop of a
// deeply nested type recurses over the native stack: `fun a -> fun a -> ... 0` is as deep a
// type as it has links. A node's payload is dropped through `bounded_drop` (the `Default`
// left behind owns nothing).
impl<T: Interned> Drop for HcBox<T> {
    fn drop(&mut self) {
        crate::util::bounded_drop(std::mem::take(&mut self.value));
    }
}

impl<T: Interned> Clone for Hc<T> {
    fn clone(&self) -> Hc<T> {
        Hc(self.0.clone())
    }
}

impl<T: Interned> Deref for Hc<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.value
    }
}

impl<T: Interned> AsRef<T> for Hc<T> {
    fn as_ref(&self) -> &T {
        &self.0.value
    }
}

impl<T: Interned + fmt::Debug> fmt::Debug for Hc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.value.fmt(f)
    }
}

impl<T: Interned + fmt::Display> fmt::Display for Hc<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.value.fmt(f)
    }
}

impl<T: Interned + PartialEq> PartialEq for Hc<T> {
    fn eq(&self, other: &Hc<T>) -> bool {
        Hc::ptr_eq(self, other) || (self.0.summary.any(Summary::REC) && other.0.summary.any(Summary::REC) && self.0.value == other.0.value)
    }
}

// A small multiplicative hasher for node keys (the keys are short and hashed on every
// construction).
#[derive(Default)]
pub struct FxHasher(u64);

impl FxHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }
    fn write_u8(&mut self, n: u8) {
        self.add(n as u64);
    }
    fn write_u32(&mut self, n: u32) {
        self.add(n as u64);
    }
    fn write_u64(&mut self, n: u64) {
        self.add(n);
    }
    fn write_usize(&mut self, n: usize) {
        self.add(n as u64);
    }
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }
}

macro_rules! interned_table {
    ($t:ty) => {
        fn table() -> &'static LocalKey<RefCell<Table<$t>>> {
            thread_local! {
                static TABLE: RefCell<Table<$t>> = RefCell::new(Table::default());
            }
            &TABLE
        }
    };
}

impl Interned for Type {
    interned_table!(Type);

    fn summary(&self) -> Summary {
        match self {
            Type::Var(_) => Summary::of(Summary::VAR).node(),
            Type::List(e) => e.summary().node(),
            Type::Fun(a, row, b) => {
                let rows = if matches!(row.tail, Tail::Var(_)) { Summary::ROWVAR } else { 0 };
                a.summary().join(b.summary()).join(Summary::of(rows)).node()
            }
            Type::Tuple(items) | Type::Union(items) => items.summary().node(),
            Type::Record(fields) => fields.summary().node(),
            Type::Indexed(w, _) => w.summary().join(Summary::of(Summary::INDEXED)).node(),
            Type::RecVar(RecVar::Free { .. }) => Summary::of(Summary::FREE).node(),
            Type::RecVar(RecVar::Bound { depth, .. }) => Summary { reach: depth + 1, ..Summary::default() }.node(),
            Type::Rec(g, _) => {
                let bodies = g.0.bodies.summary();
                Summary { flags: bodies.flags | Summary::REC, reach: g.0.escape, size: bodies.size }.node()
            }
            Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) => Summary::default().node(),
        }
    }

    fn exact_eq(&self, other: &Type) -> bool {
        match (self, other) {
            (Type::Dyn, Type::Dyn) | (Type::Int, Type::Int) | (Type::Float, Type::Float) | (Type::Bool, Type::Bool) | (Type::Str, Type::Str) => true,
            (Type::Var(a), Type::Var(b)) => a == b,
            (Type::List(a), Type::List(b)) => Hc::ptr_eq(a, b),
            (Type::Fun(a1, r1, b1), Type::Fun(a2, r2, b2)) => Hc::ptr_eq(a1, a2) && r1 == r2 && Hc::ptr_eq(b1, b2),
            (Type::Token(a), Type::Token(b)) => a == b,
            (Type::Rec(g, i), Type::Rec(h, j)) => i == j && (Rc::ptr_eq(&g.0, &h.0) || (g.0.names == h.0.names && Hc::ptr_eq(&g.0.bodies, &h.0.bodies))),
            (Type::RecVar(a), Type::RecVar(b)) => a == b,
            (Type::Tuple(a), Type::Tuple(b)) | (Type::Union(a), Type::Union(b)) => Hc::ptr_eq(a, b),
            (Type::Record(a), Type::Record(b)) => Hc::ptr_eq(a, b),
            (Type::Indexed(a, i), Type::Indexed(b, j)) => Hc::ptr_eq(a, b) && i == j,
            _ => false,
        }
    }

    fn exact_hash(&self, h: &mut FxHasher) {
        std::mem::discriminant(self).hash(h);
        match self {
            Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str => {}
            Type::Var(name) => name.hash(h),
            Type::List(e) => e.addr().hash(h),
            Type::Fun(a, row, b) => (a.addr(), row.entries.addr(), &row.tail, b.addr()).hash(h),
            Type::Token(id) => id.hash(h),
            Type::Rec(g, i) => (&g.0.names, g.0.bodies.addr(), i).hash(h),
            Type::RecVar(RecVar::Bound { depth, member }) => (depth, member).hash(h),
            Type::RecVar(RecVar::Free { id, name }) => (id, name).hash(h),
            Type::Tuple(items) | Type::Union(items) => items.addr().hash(h),
            Type::Record(fields) => fields.addr().hash(h),
            Type::Indexed(w, i) => (w.addr(), i).hash(h),
        }
    }
}

impl Interned for Vec<Type> {
    interned_table!(Vec<Type>);

    fn summary(&self) -> Summary {
        self.iter().fold(Summary::default(), |s, t| s.join(t.summary()))
    }

    fn exact_eq(&self, other: &Vec<Type>) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a.exact_eq(b))
    }

    fn exact_hash(&self, h: &mut FxHasher) {
        self.len().hash(h);
        self.iter().for_each(|t| t.exact_hash(h));
    }
}

impl Interned for Vec<(String, Type)> {
    interned_table!(Vec<(String, Type)>);

    fn summary(&self) -> Summary {
        self.iter().fold(Summary::default(), |s, (_, t)| s.join(t.summary()))
    }

    fn exact_eq(&self, other: &Vec<(String, Type)>) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|((n, a), (m, b))| n == m && a.exact_eq(b))
    }

    fn exact_hash(&self, h: &mut FxHasher) {
        self.len().hash(h);
        for (name, t) in self {
            name.hash(h);
            t.exact_hash(h);
        }
    }
}

// `Default` is what a dropped node's payload is replaced with (HcBox::drop); never read.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Type {
    Dyn,
    #[default]
    Int,
    // A second, DISTINCT numeric type from Int -- not generally consistent
    // with it (types::consistent has no Int<->Float arm; a [Float]-
    // annotated parameter still statically rejects a [Int] argument, same
    // as any other two concrete types). Int/Float interop lives ONLY in
    // typecheck::elaborate's own arithmetic BinOp arm (Add/Sub/Mul/Div/
    // Mod/Lt/Eq), via a local coerce_numeric helper that never touches
    // this general relation -- deliberately narrow, so this new type
    // can't quietly widen unrelated Tuple/Record/List/Fun consistency
    // checks elsewhere in the checker.
    Float,
    // A generalized value-type variable -- see typecheck::Scheme's own
    // doc comment for the full generalize-at-`let`/instantiate-at-use
    // story (mirrors a row's `Tail::Var` exactly, just for ordinary types
    // instead of effect rows). Minted at many binding sites via
    // InferCtx::fresh_var, never written by a user -- there is no surface
    // syntax for this. Treated exactly like Type::Dyn by
    // `consistent`/`coerce` everywhere except unify(), which binds it
    // into infer.subst. Precision is recovered via InferCtx::resolve/
    // resolve_deep reading that binding back to substitute a concrete type.
    // IS surfaced to a user: an unresolved Type::Var can appear in a
    // static type-mismatch message (e.g. `id(id) + 1`). Its Display impl
    // renders it identically to Type::Dyn ("Dyn") rather than some
    // internal name like "<generic>" -- renno has no generics syntax a
    // user would recognize such a name as referring to, and since a
    // Type::Var behaves exactly like Dyn everywhere it's observable,
    // showing "Dyn" is the more honest, actionable thing to tell them.
    Var(String),
    Bool,
    Str,
    // Element type. renno's List is a native primitive (Value::List, the
    // claim-list of list.rs).
    List(Hc<Type>),
    // param, effect row (what calling this may perform), return type.
    Fun(Hc<Type>, Row, Hc<Type>),
    // The static type of an `opaque` expression -- a singleton, one per
    // SOURCE POSITION (the u64), never per evaluation: `opaque` written at
    // one spot always has this same type no matter how many times that
    // code runs (Expr::Token is a literal, not a fresh-each-call
    // generator -- see its own doc comment). Two Token types are
    // consistent only when their ids match exactly (consistent_inner);
    // this is what lets a Token-typed tuple field carry real nominal
    // identity using nothing but ordinary structural comparison -- no
    // separate name-registry/brand-comparison mechanism needed.
    Token(u64),
    // Member `usize` of a recursive type group: a closed, finite mu-term (see RecGroup).
    Rec(RecRef, usize),
    // A reference inside a group's bodies (Bound), or the placeholder the machine hands a
    // `let rec` type group's values while it builds them (Free).
    RecVar(RecVar),
    // Fixed-arity, per-position-typed product -- `(a, b, c)` syntax.
    // Anonymous (no declared name, no registry lookup): consistent_inner
    // compares two Tuple types directly, positionally, with nothing to
    // look up. Runtime representation is a plain Value::List (see
    // Expr::Tuple's own doc comment) -- no new Value kind needed.
    Tuple(Hc<Vec<Type>>),
    // Fixed-arity, per-position-typed product, like Tuple -- but each
    // position also carries a field name, and consistency additionally
    // requires the names to match (not just the types). ALWAYS stored
    // sorted by name (parser::parse_record_fields' job, for the type
    // itself, a construction, and a pattern alike) -- that's what makes
    // `{y: 2, x: 1}` and `{x: 1, y: 2}` the same type. Anonymous, no
    // registry: like Tuple, two Record types are compared directly,
    // positionally (name then type), nothing to look up.
    //
    // Unlike Tuple, Record does NOT share Tuple's runtime representation
    // -- it has its own real, name-keyed Value::Record (see its own doc
    // comment for why: width subtyping needs a value whose fields are
    // addressable by name, not position, so an extra field a narrower
    // type never asked for is simply never observed rather than needing
    // to be projected away at some boundary). `coerce` uses a SEPARATE,
    // one-directional relation for that -- types::record_satisfies, not
    // this type's own (exact, symmetric) consistency.
    Record(Hc<Vec<(String, Type)>>),
    // "One of these" -- a self-contained sum, no registry/name lookup.
    // Typically each alternative is a Tuple whose first element is a
    // distinct Token (a hand-rolled sum: `Type::Union([Tuple([
    // Token(id_none)]), Tuple([Token(id_some), Int])])`), but nothing
    // here requires that shape -- any two types can be unioned. Named via
    // `let Name = Union(A, B) in ...`; the union itself
    // has no runtime representation of its own, only whatever VALUE
    // actually flows through ends up being one alternative's own shape.
    Union(Hc<Vec<Type>>),
    // `Vec(n)` sugars over this -- NOT its own bespoke primitive. Wraps
    // either Type::List directly (the primary, worked case) or a
    // qualifying Type::Rec+Type::Union (derived index-refinement, a
    // later phase) with one IndexExpr tracking its own length/size.
    // See the design spec's own section 2 for the full rationale.
    Indexed(Hc<Type>, Rc<IndexExpr>),
}

// An effect row (spec 2026-10-17-one-row-sort-design.md): the effects a call may perform.
// `entries` are the known labels, sorted, each with a payload type reserved for typed effects
// (always `Type::Dyn` for now). `tail` says what else may be performed: nothing (`Closed`), a
// row variable (`Var`, solved through `InferCtx::row_subst`), or anything (`Dyn`, when
// information was lost crossing a `Dyn` boundary). A `Dyn` tail never has entries.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub entries: Hc<Vec<(String, Type)>>,
    pub tail: Tail,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Tail {
    Closed,
    Var(String),
    Dyn,
}

thread_local! {
    static PURE_ROW: Row = Row::new(Vec::new(), Tail::Closed);
    static DYN_ROW: Row = Row::new(Vec::new(), Tail::Dyn);
    // The entries of a row with no labels (every row variable's own row), shared.
    static NO_LABELS: Hc<Vec<(String, Type)>> = Hc::new(Vec::new());
}

impl Row {
    // `labels` in any order, duplicates allowed; a `Dyn` tail drops them.
    pub fn new(mut labels: Vec<String>, tail: Tail) -> Row {
        if tail == Tail::Dyn {
            labels.clear();
        }
        if labels.is_empty() {
            return Row { entries: NO_LABELS.with(Hc::clone), tail };
        }
        labels.sort();
        labels.dedup();
        Row { entries: Hc::new(labels.into_iter().map(|l| (l, Type::Dyn)).collect()), tail }
    }

    pub fn pure() -> Row {
        PURE_ROW.with(Row::clone)
    }

    pub fn single(effect: &str) -> Row {
        Row::new(vec![effect.to_string()], Tail::Closed)
    }

    pub fn var(name: impl Into<String>) -> Row {
        Row::new(Vec::new(), Tail::Var(name.into()))
    }

    pub fn dyn_() -> Row {
        DYN_ROW.with(Row::clone)
    }

    pub fn labels(&self) -> impl Iterator<Item = &str> + '_ {
        self.entries.iter().map(|(l, _)| l.as_str())
    }

    pub fn has(&self, label: &str) -> bool {
        self.entries.binary_search_by(|(l, _)| l.as_str().cmp(label)).is_ok()
    }

    pub fn is_dyn(&self) -> bool {
        self.tail == Tail::Dyn
    }

    // Discharge one effect (what a `handle` does to its body's row). An absent label leaves the
    // row as it is: a tail cannot be subtracted from (no "lacks" constraint; spec 3.1 Handle).
    pub fn remove(&self, effect: &str) -> Row {
        if !self.has(effect) {
            return self.clone();
        }
        Row::new(self.labels().filter(|l| *l != effect).map(String::from).collect(), self.tail.clone())
    }
}

// ---- Recursive types -------------------------------------------------------------------
//
// A recursive type is a closed, FINITE mu-term: `Rec(group, i)` is member i of `group`, whose
// bodies mention the group's members as `RecVar::Bound { depth, member }` (de Bruijn style:
// depth counts the nested groups between the reference and the group it names). Nothing is
// cyclic, so the derived Debug/Drop and `==` terminate. `unfold` is the only way to look inside.
// The machine builds groups where a `let rec` of unannotated calls runs; see close_group.

#[derive(Debug)]
pub struct RecGroup {
    pub names: Vec<Rc<str>>,
    pub bodies: Hc<Vec<Type>>,
    // No body mentions (at any depth, nested groups included) a Type::Var, an Indexed,
    // a `Tail::Var` or a RecVar::Free: a walker that renames variables may return it as is.
    pub plain: bool,
    // How many enclosing groups the bodies reach through a Bound (0: closed). Lets `unfold`
    // skip a nested group that never mentions the group being unfolded.
    pub escape: u32,
}

impl RecGroup {
    pub fn new(names: Vec<Rc<str>>, bodies: Vec<Type>) -> RecGroup {
        let bodies = Hc::new(bodies);
        let summary = bodies.summary();
        let plain = !summary.any(Summary::VAR | Summary::ROWVAR | Summary::INDEXED | Summary::FREE);
        RecGroup { names, bodies, plain, escape: summary.reach().saturating_sub(1) }
    }
}

// Equality is structural, member names ignored (E1): two separately built `A(Int)` are equal.
#[derive(Clone, Debug)]
pub struct RecRef(pub Rc<RecGroup>);

impl PartialEq for RecRef {
    fn eq(&self, other: &RecRef) -> bool {
        Rc::ptr_eq(&self.0, &other.0) || self.0.bodies == other.0.bodies
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RecVar {
    Bound { depth: u32, member: u32 },
    Free { id: u64, name: Rc<str> },
}

static REC_ID: AtomicU64 = AtomicU64::new(0);

pub fn fresh_rec_id() -> u64 {
    REC_ID.fetch_add(1, Ordering::Relaxed)
}

pub fn each_child(t: &Type, f: &mut dyn FnMut(&Type)) {
    match t {
        Type::List(e) => f(e),
        Type::Fun(a, _, b) => {
            f(a);
            f(b);
        }
        Type::Tuple(items) | Type::Union(items) => items.iter().for_each(f),
        Type::Record(fields) => fields.iter().for_each(|(_, t)| f(t)),
        Type::Indexed(w, _) => f(w),
        _ => {}
    }
}

fn map_vec(items: &[Type], f: &mut dyn FnMut(&Type) -> Option<Type>) -> Option<Vec<Type>> {
    let mapped: Vec<Option<Type>> = items.iter().map(&mut *f).collect();
    if mapped.iter().all(Option::is_none) {
        return None;
    }
    Some(mapped.into_iter().zip(items).map(|(m, old)| m.unwrap_or_else(|| old.clone())).collect())
}

// `t` with `f` applied to every direct child type; None when no child changed (so an
// unchanged subtree keeps its Rc). Rec and RecVar have no children here: callers own them.
fn map_children(t: &Type, f: &mut dyn FnMut(&Type) -> Option<Type>) -> Option<Type> {
    match t {
        Type::List(e) => f(e).map(|e| Type::List(Hc::new(e))),
        Type::Fun(a, row, b) => {
            let (na, nb) = (f(a), f(b));
            if na.is_none() && nb.is_none() {
                return None;
            }
            Some(Type::Fun(na.map_or_else(|| a.clone(), Hc::new), row.clone(), nb.map_or_else(|| b.clone(), Hc::new)))
        }
        Type::Tuple(items) => map_vec(items, f).map(|v| Type::Tuple(Hc::new(v))),
        Type::Union(items) => map_vec(items, f).map(|v| Type::Union(Hc::new(v))),
        Type::Record(fields) => {
            let mapped: Vec<Option<Type>> = fields.iter().map(|(_, t)| f(t)).collect();
            if mapped.iter().all(Option::is_none) {
                return None;
            }
            Some(Type::Record(Hc::new(
                mapped.into_iter().zip(fields.iter()).map(|(m, (n, old))| (n.clone(), m.unwrap_or_else(|| old.clone()))).collect(),
            )))
        }
        Type::Indexed(w, i) => f(w).map(|w| Type::Indexed(Hc::new(w), i.clone())),
        _ => None,
    }
}

// The group rebuilt with `f` applied to each body; None when no body changed.
pub fn map_group(h: &RecRef, k: usize, f: &mut dyn FnMut(&Type) -> Option<Type>) -> Option<Type> {
    let bodies = map_vec(&h.0.bodies, f)?;
    Some(Type::Rec(RecRef(Rc::new(RecGroup::new(h.0.names.clone(), bodies))), k))
}

// A node's exact identity as a map key: two keys are equal exactly when they are the same node.
// (`Type` itself has no `Hash`: its `==` ignores a group's member names, a node's identity
// does not.)
pub struct ById(pub Type);

impl PartialEq for ById {
    fn eq(&self, other: &ById) -> bool {
        self.0.exact_eq(&other.0)
    }
}

impl Eq for ById {}

impl Hash for ById {
    fn hash<H: Hasher>(&self, h: &mut H) {
        let mut node = FxHasher::default();
        self.0.exact_hash(&mut node);
        h.write_u64(node.finish());
    }
}

// What a walk remembers: its result per node (and per context, for a walk whose answer depends
// on where the node sits). One memo lives for one walk; the substitution a walk reads may
// change between walks, so nothing is kept across them.
pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FxSet<K> = HashSet<K, BuildHasherDefault<FxHasher>>;

pub type NodeMemo<V> = FxMap<(ById, u32), V>;

// A node smaller than this is walked again rather than looked up.
const MEMO_MIN: u16 = 24;

// The nodes a query walk has been through, so a shared node is walked once. Like a memo it
// lives for one call, and a node smaller than MEMO_MIN is never recorded.
#[derive(Default)]
pub struct Visited(FxSet<ById>);

impl Visited {
    pub fn new() -> Visited {
        Visited::default()
    }

    // True the first time `t` is offered (always, for a small node).
    pub fn first(&mut self, t: &Type) -> bool {
        !is_big(t) || self.0.insert(ById(t.clone()))
    }
}

// Big enough for a walk to remember (a memo or visited-set entry costs more than walking a
// small node again).
pub fn is_big(t: &Type) -> bool {
    t.summary().size >= MEMO_MIN
}

// A walk that rebuilds a type, returning None where nothing changed. Subtrees the walker says
// it would not change are skipped in O(1) (`skip`, from the node's summary), and the result for
// each big enough node is remembered, so the cost is the number of DISTINCT nodes, not the size
// of the expansion. A walker supplies its own treatment of some nodes (`special`); the rest
// descend through `map_children`, and a group through its bodies.
pub trait Rebuild {
    // True when `s` says the walk would change nothing in such a subtree.
    fn skip(&self, s: Summary) -> bool;
    // What else the result depends on besides the node (a nesting depth); part of the memo key.
    fn context(&self) -> u32 {
        0
    }
    fn memo(&mut self) -> &mut NodeMemo<Option<Type>>;
    // The walker's own result for `t` (None: descend as usual).
    fn special(&mut self, _t: &Type) -> Option<Option<Type>> {
        None
    }

    fn walk(&mut self, t: &Type) -> Option<Type> {
        let summary = t.summary();
        if self.skip(summary) {
            return None;
        }
        let key = (summary.size >= MEMO_MIN).then(|| (ById(t.clone()), self.context()));
        if let Some(done) = key.as_ref().and_then(|k| self.memo().get(k)) {
            return done.clone();
        }
        let result = match self.special(t) {
            Some(result) => result,
            None => match t {
                Type::Rec(h, k) => map_group(h, *k, &mut |b| self.walk(b)),
                _ => map_children(t, &mut |c| self.walk(c)),
            },
        };
        if let Some(key) = key {
            self.memo().insert(key, result.clone());
        }
        result
    }
}

// Replaces the references to `g` that sit `depth` groups out by `g` itself.
struct SubstBound<'a> {
    g: &'a RecRef,
    depth: u32,
    memo: NodeMemo<Option<Type>>,
}

impl Rebuild for SubstBound<'_> {
    fn skip(&self, s: Summary) -> bool {
        s.reach() <= self.depth
    }
    fn context(&self) -> u32 {
        self.depth
    }
    fn memo(&mut self) -> &mut NodeMemo<Option<Type>> {
        &mut self.memo
    }
    fn special(&mut self, t: &Type) -> Option<Option<Type>> {
        match t {
            Type::RecVar(RecVar::Bound { depth, member }) if *depth == self.depth => Some(Some(Type::Rec(self.g.clone(), *member as usize))),
            Type::Rec(h, k) => {
                self.depth += 1;
                let result = map_group(h, *k, &mut |b| self.walk(b));
                self.depth -= 1;
                Some(result)
            }
            _ => None,
        }
    }
}

// One step into member `i`: its body with every reference to the group replaced by the group
// itself. Only closed terms are unfolded, so nothing needs shifting. Subtrees that mention no
// reference are returned as the same nodes.
pub fn unfold(g: &RecRef, i: usize) -> Type {
    let mut walk = SubstBound { g, depth: 0, memo: NodeMemo::default() };
    walk.walk(&g.0.bodies[i]).unwrap_or_else(|| g.0.bodies[i].clone())
}

// Replaces each Free placeholder `f(id, depth)` answers for; depth counts the groups entered.
struct ReplaceFree<'a> {
    f: &'a dyn Fn(u64, u32) -> Option<Type>,
    depth: u32,
    memo: NodeMemo<Option<Type>>,
}

impl Rebuild for ReplaceFree<'_> {
    fn skip(&self, s: Summary) -> bool {
        !s.any(Summary::FREE)
    }
    fn context(&self) -> u32 {
        self.depth
    }
    fn memo(&mut self) -> &mut NodeMemo<Option<Type>> {
        &mut self.memo
    }
    fn special(&mut self, t: &Type) -> Option<Option<Type>> {
        match t {
            Type::RecVar(RecVar::Free { id, .. }) => Some((self.f)(*id, self.depth)),
            Type::Rec(h, k) => {
                self.depth += 1;
                let result = map_group(h, *k, &mut |b| self.walk(b));
                self.depth -= 1;
                Some(result)
            }
            _ => None,
        }
    }
}

// `t` with each Free placeholder `f(id, depth)` answers for replaced. None when nothing changed.
fn replace_free(t: &Type, depth: u32, f: &dyn Fn(u64, u32) -> Option<Type>) -> Option<Type> {
    ReplaceFree { f, depth, memo: NodeMemo::default() }.walk(t)
}

// The name of the first placeholder `t` still contains (a type the machine built while a
// group's values were running and that escaped before the group closed).
pub fn first_free(t: &Type) -> Option<Rc<str>> {
    fn go(t: &Type, seen: &mut Visited) -> Option<Rc<str>> {
        // A node already walked found nothing (else the walk would have stopped there).
        if !t.summary().any(Summary::FREE) || !seen.first(t) {
            return None;
        }
        match t {
            Type::RecVar(RecVar::Free { name, .. }) => Some(name.clone()),
            Type::Rec(h, _) => h.0.bodies.iter().find_map(|b| go(b, seen)),
            _ => {
                let mut found = None;
                each_child(t, &mut |c| {
                    if found.is_none() {
                        found = go(c, seen);
                    }
                });
                found
            }
        }
    }
    go(t, &mut Visited::new())
}

#[derive(Debug, PartialEq, Eq)]
pub enum CloseError {
    // The member is a bare reference to a member of its own group, or its unfolding never
    // reaches a type constructor.
    Bare,
    // A member some other member's type mentions is not a type at all.
    NotAType,
}

// Closes a `let rec` type group the machine just evaluated. `results[i]` is the value of
// member i if it is a type (it mentions the group's members as Free placeholders with `ids`),
// else None. Each strongly connected component of members that some type mentions becomes its
// own RecGroup (holding only its own members, in a canonical order); the rest keep their type
// with the placeholders replaced by the closed members. Err is (member, why).
pub fn close_group(names: &[Rc<str>], ids: &[u64], results: &[Option<Type>]) -> Result<Vec<Option<Type>>, (usize, CloseError)> {
    let n = ids.len();
    let index: HashMap<u64, usize> = ids.iter().enumerate().map(|(j, id)| (*id, j)).collect();
    for (i, r) in results.iter().enumerate() {
        if let Some(Type::RecVar(RecVar::Free { id, .. })) = r
            && index.contains_key(id)
        {
            return Err((i, CloseError::Bare));
        }
    }
    // succ[i]: the members whose placeholder member i's type mentions.
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut referenced = vec![false; n];
    for (i, r) in results.iter().enumerate() {
        if let Some(t) = r {
            collect_free(t, &index, &mut succ[i]);
            succ[i].sort_unstable();
            succ[i].dedup();
            for &j in &succ[i] {
                referenced[j] = true;
            }
        }
    }
    if let Some(j) = (0..n).find(|&j| referenced[j] && results[j].is_none()) {
        return Err((j, CloseError::NotAType));
    }
    if !referenced.contains(&true) {
        return Ok(results.to_vec());
    }
    // Each strongly connected component is its own group, built after every component it
    // reaches (Tarjan emits them in that order), so a member of another component is just
    // that component's finished Rec.
    let mut scc_of = vec![0usize; n];
    let components = strongly_connected(&succ);
    for (c, comp) in components.iter().enumerate() {
        for &m in comp {
            scc_of[m] = c;
        }
    }
    let mut slot = vec![0usize; n];
    let mut closed: Vec<Option<Type>> = vec![None; n];
    let mut out: Vec<Option<Type>> = vec![None; n];
    for (c, mut comp) in components.into_iter().enumerate() {
        let resolve = |id: u64, depth: u32, slot: &[usize], closed: &[Option<Type>]| {
            let m = *index.get(&id)?;
            if scc_of[m] == c {
                Some(Type::RecVar(RecVar::Bound { depth, member: slot[m] as u32 }))
            } else {
                closed[m].clone()
            }
        };
        let outside = |id: u64, _: u32, closed: &[Option<Type>]| index.get(&id).and_then(|&m| if scc_of[m] == c { None } else { closed[m].clone() });
        if comp.len() == 1 && !succ[comp[0]].contains(&comp[0]) {
            // A lone member that is not on a cycle (others may mention it): no group, just its
            // type with the finished members it reaches.
            let m = comp[0];
            out[m] = results[m].as_ref().map(|t| replace_free(t, 0, &|id, d| outside(id, d, &closed)).unwrap_or_else(|| t.clone()));
            closed[m] = out[m].clone();
            continue;
        }
        let mut types: Vec<Type> = Vec::with_capacity(comp.len());
        if comp.len() == 1 {
            slot[comp[0]] = 0;
            let t = results[comp[0]].as_ref().expect("referenced members are types");
            types.push(replace_free(t, 0, &|id, d| resolve(id, d, &slot, &closed)).unwrap_or_else(|| t.clone()));
        } else {
            // Members are listed in a canonical order, so equal structures give equal groups
            // however the `let rec` was written (see canonical_order), then bound to it.
            comp.sort_unstable();
            let mut bodies: HashMap<usize, Type> = HashMap::new();
            for &m in &comp {
                let t = results[m].as_ref().expect("referenced members are types");
                bodies.insert(m, replace_free(t, 0, &|id, d| outside(id, d, &closed)).unwrap_or_else(|| t.clone()));
            }
            canonical_order(&mut comp, &bodies, &index, &scc_of, c);
            for (p, &m) in comp.iter().enumerate() {
                slot[m] = p;
            }
            for &m in &comp {
                let t = &bodies[&m];
                types.push(replace_free(t, 0, &|id, d| resolve(id, d, &slot, &closed)).unwrap_or_else(|| t.clone()));
            }
        }
        let g = RecRef(Rc::new(RecGroup::new(comp.iter().map(|&m| names[m].clone()).collect(), types)));
        // Contractive: unfolding a member must reach a type constructor, never loop through members.
        for (p, &m) in comp.iter().enumerate() {
            if !matches!(g.0.bodies[p], Type::RecVar(_) | Type::Rec(..)) {
                continue;
            }
            let mut keep = vec![g.clone()];
            let mut seen = HashSet::from([(Rc::as_ptr(&g.0), p)]);
            let mut cur = (g.clone(), p);
            while let Type::Rec(h, k) = unfold(&cur.0, cur.1) {
                if !seen.insert((Rc::as_ptr(&h.0), k)) {
                    return Err((m, CloseError::Bare));
                }
                keep.push(h.clone());
                cur = (h, k);
            }
        }
        for (p, &m) in comp.iter().enumerate() {
            closed[m] = Some(Type::Rec(g.clone(), p));
            out[m] = closed[m].clone();
        }
    }
    Ok(out)
}

// The indices (into `index`) of the group members whose placeholder `t` mentions, with repeats.
fn collect_free(t: &Type, index: &HashMap<u64, usize>, found: &mut Vec<usize>) {
    fn go(t: &Type, index: &HashMap<u64, usize>, found: &mut Vec<usize>, seen: &mut Visited) {
        if !t.summary().any(Summary::FREE) || !seen.first(t) {
            return;
        }
        match t {
            Type::RecVar(RecVar::Free { id, .. }) => found.extend(index.get(id)),
            Type::Rec(h, _) => h.0.bodies.iter().for_each(|b| go(b, index, found, seen)),
            _ => each_child(t, &mut |c| go(c, index, found, seen)),
        }
    }
    go(t, index, found, &mut Visited::new());
}

// Tarjan's algorithm, iterative: the components of the graph `succ`, each after every component
// it reaches.
fn strongly_connected(succ: &[Vec<usize>]) -> Vec<Vec<usize>> {
    const UNSEEN: usize = usize::MAX;
    let n = succ.len();
    let (mut order, mut low, mut on_stack) = (vec![UNSEEN; n], vec![0usize; n], vec![false; n]);
    let (mut stack, mut out, mut counter) = (Vec::new(), Vec::new(), 0usize);
    for root in 0..n {
        if order[root] != UNSEEN {
            continue;
        }
        let mut work = vec![(root, 0usize)];
        (order[root], low[root], counter) = (counter, counter, counter + 1);
        stack.push(root);
        on_stack[root] = true;
        while let Some(&(v, k)) = work.last() {
            if k < succ[v].len() {
                work.last_mut().expect("nonempty").1 += 1;
                let w = succ[v][k];
                if order[w] == UNSEEN {
                    (order[w], low[w], counter) = (counter, counter, counter + 1);
                    stack.push(w);
                    on_stack[w] = true;
                    work.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(order[w]);
                }
            } else {
                work.pop();
                if let Some(&(parent, _)) = work.last() {
                    low[parent] = low[parent].min(low[v]);
                }
                if low[v] == order[v] {
                    let mut comp = Vec::new();
                    loop {
                        let w = stack.pop().expect("v is on the stack");
                        on_stack[w] = false;
                        comp.push(w);
                        if w == v {
                            break;
                        }
                    }
                    out.push(comp);
                }
            }
        }
    }
    out
}

// Orders the members of one mutually recursive component by what their types look like, not by
// where they were written (colour refinement: a member's colour hashes its type's shape with
// each reference to a member replaced by that member's colour, repeated until no more
// members are told apart). Members that stay alike keep their written order, which is the
// best a cheap pass can do; every order is a correct group, only equality across spellings
// depends on it.
fn canonical_order(comp: &mut [usize], bodies: &HashMap<usize, Type>, index: &HashMap<u64, usize>, scc_of: &[usize], c: usize) {
    use std::hash::{Hash, Hasher};
    // A node's hash is built from its children's, each hashed once per walk (`memo`).
    fn skeleton(t: &Type, colour: &dyn Fn(u64) -> Option<u64>, memo: &mut FxMap<ById, u64>) -> u64 {
        let big = t.summary().size >= MEMO_MIN;
        if big && let Some(done) = memo.get(&ById(t.clone())) {
            return *done;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        match t {
            Type::RecVar(RecVar::Free { id, .. }) => match colour(*id) {
                Some(k) => (1u8, k).hash(&mut h),
                // An enclosing group's placeholder: its id is fresh per evaluation, so only
                // the fact that it is one may enter the hash.
                None => 2u8.hash(&mut h),
            },
            Type::RecVar(RecVar::Bound { depth, member }) => (3u8, depth, member).hash(&mut h),
            Type::Rec(g, k) => {
                (4u8, k, g.0.bodies.len()).hash(&mut h);
                if !g.0.plain {
                    g.0.bodies.iter().for_each(|b| skeleton(b, colour, memo).hash(&mut h));
                }
            }
            Type::Token(id) => (5u8, id).hash(&mut h),
            Type::Record(fields) => (6u8, fields.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()).hash(&mut h),
            _ => std::mem::discriminant(t).hash(&mut h),
        }
        if !matches!(t, Type::Rec(..)) {
            each_child(t, &mut |ch| skeleton(ch, colour, memo).hash(&mut h));
        }
        let hash = h.finish();
        if big {
            memo.insert(ById(t.clone()), hash);
        }
        hash
    }
    let mut colours: HashMap<usize, u64> = comp.iter().map(|&m| (m, 0)).collect();
    let mut distinct = 1;
    for _ in 0..8 {
        let next: HashMap<usize, u64> = comp
            .iter()
            .map(|&m| {
                let of = |id: u64| index.get(&id).filter(|&&j| scc_of[j] == c).map(|j| colours[j]);
                (m, skeleton(&bodies[&m], &of, &mut FxMap::default()))
            })
            .collect();
        let count = next.values().collect::<HashSet<_>>().len();
        colours = next;
        if count <= distinct {
            break;
        }
        distinct = count;
    }
    comp.sort_by_key(|m| colours[m]);
}

// The name the user wrote for an index, row or type variable: the part before the first `#`
// (the rest is checker bookkeeping). Applied at message time only; stored names stay exact.
pub(crate) fn surface_name(name: &str) -> &str {
    name.split('#').next().unwrap_or(name)
}

// The base of an opened arrow-row tail (D10): a lambda's closed row gets a fresh tail `'#k`.
// Not an identifier, so no written `->{e}` can produce it. An unbound one prints as nothing.
pub(crate) const OPENED_ROW: &str = "'";

impl fmt::Display for Row {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let labels = self.labels().collect::<Vec<_>>().join(", ");
        match &self.tail {
            Tail::Dyn => write!(f, "Dyn"),
            Tail::Closed => write!(f, "{{{labels}}}"),
            Tail::Var(name) if surface_name(name) == OPENED_ROW => write!(f, "{{{labels}}}"),
            Tail::Var(name) if labels.is_empty() => write!(f, "{{{}}}", surface_name(name)),
            Tail::Var(name) => write!(f, "{{{labels} | {}}}", surface_name(name)),
        }
    }
}

// Consistency, not equality: Dyn is consistent with everything. This is
// what makes it "gradual" -- an annotated boundary only rejects code where
// BOTH sides are concretely known and disagree. Effect rows follow the
// same rule: Dyn row is consistent with any row, two Closed rows must
// match exactly (no row subtyping/variance modeled yet).
//
// No cycle guard: every Type here is a finite tree (a recursive type is a closed mu-term whose
// back-references are Bound indices, never pointers), so a plain structural recursion always
// terminates.
pub fn consistent(a: &Type, b: &Type) -> bool {
    consistent_in(a, b, &mut FxMap::default())
}

// `memo` holds the answers for big pairs (the relation is pure), so a shared pair is decided once.
fn consistent_in(a: &Type, b: &Type, memo: &mut FxMap<(ById, ById), bool>) -> bool {
    let key = (is_big(a) || is_big(b)).then(|| (ById(a.clone()), ById(b.clone())));
    if let Some(done) = key.as_ref().and_then(|k| memo.get(k)) {
        return *done;
    }
    let result = consistent_pair(a, b, memo);
    if let Some(key) = key {
        memo.insert(key, result);
    }
    result
}

fn consistent_pair(a: &Type, b: &Type, memo: &mut FxMap<(ById, ById), bool>) -> bool {
    match (a, b) {
        (Type::Dyn, _) | (_, Type::Dyn) => true,
        // Treated exactly like Dyn here -- the one place it behaves
        // differently is unify(), which binds it into infer.subst,
        // not a change to this general relation. See Type::Var's
        // own doc comment.
        (Type::Var(_), _) | (_, Type::Var(_)) => true,
        (Type::Int, Type::Int) => true,
        // Deliberately NOT (Type::Int, Type::Float) => true -- see
        // Type::Float's own doc comment on why that interop stays local
        // to the arithmetic operators, not this general relation.
        (Type::Float, Type::Float) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::Str, Type::Str) => true,
        (Type::List(a), Type::List(b)) => consistent_in(a, b, memo),
        (Type::Fun(a1, r1, b1), Type::Fun(a2, r2, b2)) => {
            consistent_in(a1, a2, memo) && consistent_in(b1, b2, memo) && row_consistent(r1, r2)
        }
        // A singleton per source position -- consistent only with itself
        // (and Dyn, already handled above). No registry lookup: the id
        // alone says everything there is to say.
        (Type::Token(a), Type::Token(b)) => a == b,
        // Structural up to member names (RecRef's own equality); never against its unfolding.
        (Type::Rec(a, i), Type::Rec(b, j)) => i == j && a == b,
        // Two Indexed types are consistent when their wrapped types are
        // consistent AND their index expressions are equal (§3's SOP
        // normalization) -- both halves required, mirroring how List's
        // own arm requires its element type to match.
        //
        // A bare, unbound index variable on EITHER side is permissive,
        // mirroring Type::Var's own unconditional permissiveness just
        // above -- historically this let coerce() no-op on a Vec(n)
        // parameter/annotation; coerce now decides on relate (which
        // treats a flexible index variable as Unknown), so this arm
        // matters only for `==`/`++`/Union-membership and pattern checks. The guard is IndexExpr::Var specifically, not
        // "contains a variable anywhere" -- Vec(n+1) against Vec(4)
        // still correctly requires exact SOP-equality, matching
        // unify_index_expr's own identical no-equation-solving limit.
        // NOTE: consistent() is symmetric and also backs `==`/Union-
        // membership -- this permissiveness applies there too, which is
        // why Expr::Let/LetRec also gain a follow-up unify_fits call:
        // without one, a bare index variable accepted here can be left
        // silently UNBOUND with no runtime check at all (rigid vs.
        // flexible index variables is a deeper, still-open concern).
        (Type::Indexed(wa, ia), Type::Indexed(wb, ib)) => {
            consistent_in(wa, wb, memo)
                && (index_exprs_equal(ia, ib) || matches!(**ia, IndexExpr::Var(_)) || matches!(**ib, IndexExpr::Var(_)))
        }
        // Positional, structural, no name/registry involved -- same idea
        // as List's own element comparison, just per-position instead of
        // one shared element type.
        (Type::Tuple(a), Type::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| consistent_in(x, y, memo))
        }
        // Same idea as Tuple, plus each position's NAME must also match --
        // reuses record_satisfies (the one-directional "has at least
        // these fields" relation coerce's width subtyping is built from)
        // for the actual per-field name+type check, adding only the
        // length equality that turns "at least" into "exactly." Sound
        // because field names within one Type::Record are always unique
        // (parse_record_fields rejects a duplicate) -- same length plus
        // every one of `a`'s (name, type) pairs found in `b` forces the
        // two name SETS to be identical, not just one contained in the
        // other.
        (Type::Record(a), Type::Record(b)) => a.len() == b.len() && record_satisfies_in(a, b, memo),
        // Two unions: consistent as SETS -- every alternative on each
        // side has a match on the other (order and duplicates don't
        // matter, only membership). Must come before the single-Union
        // arm below (Rust checks patterns top to bottom; that arm's
        // `other: &Type` would otherwise also match a Union on the other
        // side).
        (Type::Union(a), Type::Union(b)) => {
            a.iter().all(|x| b.iter().any(|y| consistent_in(x, y, memo))) && b.iter().all(|y| a.iter().any(|x| consistent_in(x, y, memo)))
        }
        // One union, one concrete type: consistent if the concrete side
        // matches ANY ONE alternative -- "is this one of the possible
        // shapes," the same question a runtime value crossing a
        // Union-typed boundary would need answered.
        (Type::Union(alts), other) | (other, Type::Union(alts)) => alts.iter().any(|alt| consistent_in(alt, other, memo)),
        // Every Tuple value IS, structurally, a List at runtime (see
        // Type::Tuple's own doc comment) -- needed so a tuple pattern
        // (parser::pattern_atom desugars `(p, q)` straight into
        // Pattern::List, typed List(Dyn) by typecheck::pattern_type, same
        // as any other List pattern) can match a Tuple-typed scrutinee.
        (Type::List(elem), Type::Tuple(_)) | (Type::Tuple(_), Type::List(elem)) => matches!(**elem, Type::Dyn),
        // No equivalent bridge for Record: a `{x, y}` pattern is now its
        // own Pattern::Record (see its own doc comment), not sugar over
        // Pattern::List, so typecheck::pattern_type gives it plain
        // Type::Dyn like every other imprecisely-typed pattern -- already
        // consistent with anything via the Dyn arm at the top of this
        // function, no special-casing needed here.
        _ => false,
    }
}

// One-directional: does `actual` have AT LEAST every field `required`
// names, each with a consistent type? (Extra fields on `actual` are
// fine -- see Pattern::Record's own doc comment for why nothing ever
// needs to know about them.) This is NOT part of `consistent()` --
// `consistent()` stays a symmetric equivalence relation everywhere,
// including for two Record types (exact field-set match, used by `==`
// and Union-alternative membership) -- width subtyping is a one-way
// "does this value fit where that type is expected" question, asked only
// by relate (typecheck/relate.rs), which has its own Record arm; this
// helper survives for consistent()'s width-aware callers.
pub fn record_satisfies(required: &[(String, Type)], actual: &[(String, Type)]) -> bool {
    record_satisfies_in(required, actual, &mut FxMap::default())
}

fn record_satisfies_in(required: &[(String, Type)], actual: &[(String, Type)], memo: &mut FxMap<(ById, ById), bool>) -> bool {
    required.iter().all(|(name, ty)| find_field(actual, name).is_some_and(|aty| consistent_in(ty, aty, memo)))
}

// Name-blind row consistency for callers with no InferCtx (`consistent_in`): a Dyn or any
// variable tail is consistent with anything; two closed rows need equal label sets.
pub fn row_consistent(a: &Row, b: &Row) -> bool {
    match (&a.tail, &b.tail) {
        (Tail::Dyn, _) | (_, Tail::Dyn) | (Tail::Var(_), _) | (_, Tail::Var(_)) => true,
        (Tail::Closed, Tail::Closed) => a.labels().eq(b.labels()),
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write_type(self, f, &Cell::new(usize::MAX))
    }
}

// The most type nodes a message prints (`Type::brief`); a bigger type ends in `...`. A type value
// is a DAG, so its text can be exponentially larger than the type itself.
const BRIEF_NODES: usize = 10_000;

// A type for a message: the same text as its Display, cut off after BRIEF_NODES nodes.
pub struct Brief<'a>(&'a Type);

impl fmt::Display for Brief<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write_type(self.0, f, &Cell::new(BRIEF_NODES))
    }
}

impl Type {
    pub fn brief(&self) -> Brief<'_> {
        Brief(self)
    }
}

// The one rendering of a type: each node printed spends one unit of `budget`, and a node met with
// none left prints `...` (so printing stops growing once it runs out).
fn write_type(t: &Type, f: &mut fmt::Formatter, budget: &Cell<usize>) -> fmt::Result {
    let left = budget.get();
    if left == 0 {
        return write!(f, "...");
    }
    budget.set(left - 1);
    match t {
        Type::Dyn => write!(f, "Dyn"),
        Type::Int => write!(f, "Int"),
        Type::Float => write!(f, "Float"),
        // The name is internal bookkeeping (see Type::Var's own doc
        // comment) that a user has no way to interpret -- render
        // identically to Type::Dyn, which is what a Type::Var
        // actually behaves like everywhere it's observable.
        Type::Var(_) => write!(f, "Dyn"),
        // The member's own name, never its body.
        Type::Rec(g, i) => write!(f, "{}", g.0.names[*i]),
        Type::RecVar(RecVar::Free { name, .. }) => write!(f, "{name}"),
        // Unreachable by construction (a Bound only lives inside a group's bodies).
        Type::RecVar(RecVar::Bound { depth, member }) => write!(f, "?{depth}.{member}"),
        Type::Bool => write!(f, "Bool"),
        Type::Str => write!(f, "Str"),
        Type::List(elem) => {
            write!(f, "[")?;
            write_type(elem, f, budget)?;
            write!(f, "]")
        }
        // Dyn row prints as a plain arrow -- matches every existing
        // (unannotated-row) Fun type exactly as before this existed.
        Type::Fun(a, row, b) => {
            write!(f, "(")?;
            write_type(a, f, budget)?;
            match row.tail {
                Tail::Dyn => write!(f, " -> ")?,
                _ => write!(f, " ->{row} ")?,
            }
            write_type(b, f, budget)?;
            write!(f, ")")
        }
        // The brand (the `opaque`'s source offset), so two token types
        // read differently in a message (spec 12.6); Value::type_name
        // names a token value the same way.
        Type::Token(id) => write!(f, "Token#{id}"),
        Type::Tuple(items) => {
            write!(f, "(")?;
            for (i, t) in items.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write_type(t, f, budget)?;
            }
            write!(f, ")")
        }
        Type::Record(fields) => {
            write!(f, "{{")?;
            for (i, (name, t)) in fields.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{name}: ")?;
                write_type(t, f, budget)?;
            }
            write!(f, "}}")
        }
        Type::Union(alts) => {
            for (i, t) in alts.iter().enumerate() {
                if i > 0 {
                    write!(f, " | ")?;
                }
                write_type(t, f, budget)?;
            }
            Ok(())
        }
        Type::Indexed(wrapped, index) => {
            write_type(wrapped, f, budget)?;
            write!(f, "({index})")
        }
    }
}
