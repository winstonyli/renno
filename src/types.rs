use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::index_expr::{index_exprs_equal, IndexExpr};
use crate::util::find_field;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Dyn,
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
    // story (mirrors EffectRow::Var exactly, just for ordinary types
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
    // Element type. renno's List is a native primitive (Value::List,
    // Rc<Vec<Value>>).
    List(Rc<Type>),
    // param, effect row (what calling this may perform), return type.
    Fun(Rc<Type>, EffectRow, Rc<Type>),
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
    // A reference to a genuinely self-referential type's own
    // definition -- `let rec List = Union(Tuple(Int, List), Bool) in ...` -- rather
    // than the ordinary, fully-expanded structural value every OTHER
    // (non-recursive) type resolves to directly. The string is always
    // a gensym'd unique id in the shape "Base#N" (see typecheck's
    // fresh_named_type_id), never the bare user-typed name --
    // recursive types in different, unrelated scopes can share a surface
    // name, and the registry this leaf is looked up in (InferCtx's own
    // `named_types`, populated by recursive_type_group) outlives those
    // scopes, so it needs a name that can never collide across them.
    //
    // A lightweight, non-expanding leaf, in the same spirit as
    // Type::Var/Type::Token: a tag consulted on demand, never
    // something requiring evaluation or unfolding to construct.
    // Comparison (consistent()/unify()) is purely nominal, by id --
    // see their own new arms -- so nothing here ever needs to unfold
    // what this id actually stands for, and no cycle-detection is
    // needed anywhere in this file's own structural recursion, even
    // though a registry entry for a genuinely recursive type
    // structurally contains this SAME leaf pointing at itself.
    // shape_test/build_boundary_check/pattern_could_match/
    // relate/unify_fits (typecheck.rs) are the consumers that ever look
    // up what a Named id stands for, and only one level at a time -- see
    // their own doc comments.
    Named(String),
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
    Tuple(Rc<Vec<Type>>),
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
    Record(Rc<Vec<(String, Type)>>),
    // "One of these" -- a self-contained sum, no registry/name lookup.
    // Typically each alternative is a Tuple whose first element is a
    // distinct Token (a hand-rolled sum: `Type::Union([Tuple([
    // Token(id_none)]), Tuple([Token(id_some), Int])])`), but nothing
    // here requires that shape -- any two types can be unioned. Named via
    // `let Name = Union(A, B) in ...`; the union itself
    // has no runtime representation of its own, only whatever VALUE
    // actually flows through ends up being one alternative's own shape.
    Union(Rc<Vec<Type>>),
    // `Vec(n)` sugars over this -- NOT its own bespoke primitive. Wraps
    // either Type::List directly (the primary, worked case) or a
    // qualifying Type::Named+Type::Union (derived index-refinement, a
    // later phase) with one IndexExpr tracking its own length/size.
    // See the design spec's own section 2 for the full rationale.
    Indexed(Rc<Type>, Rc<IndexExpr>),
}

// Closed effect row: an exact known set (Closed), "unknown, could be
// anything" (Dyn, when info was lost crossing a Dyn boundary -- an untyped
// callee, an unrecognized handler expression), or a named row VARIABLE
// (Var) written explicitly in a function-type annotation (`(A ->{e} B)`).
// Var is deliberately name-based rather than a fresh-generated id with no
// surface meaning: it only ever originates from something the user typed,
// never from inference, so there's no unification engine here -- just
// generalization (typecheck::extend_generalized, at `let`) and
// instantiation (typecheck::lookup, at each use) by simple substitution,
// plus a small binding step at the App site where a concrete function
// value finally supplies what the variable stands for. See typecheck.rs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectRow {
    Dyn,
    Closed(BTreeSet<String>),
    Var(String),
}

impl EffectRow {
    pub fn pure() -> EffectRow {
        EffectRow::Closed(BTreeSet::new())
    }

    pub fn single(effect: &str) -> EffectRow {
        let mut s = BTreeSet::new();
        s.insert(effect.to_string());
        EffectRow::Closed(s)
    }

    // What effects might happen if EITHER of two sub-expressions' effects
    // happen -- used to combine sibling subexpressions, and to over-
    // approximate an `if`'s two branches (only one runs, but which one
    // isn't known statically, so take the union rather than guess).
    //
    // Var propagates through unioning with itself or with "nothing extra"
    // (pure) unchanged, so a row-polymorphic function's row can flow
    // through the accumulator all the way to the point it's finally bound
    // to something concrete. Mixed with anything else (a different row, a
    // real effect, Dyn) it collapses to Dyn: representing "this variable's
    // eventual value, plus these other effects" precisely would need a
    // genuine open-row representation (known labels + a variable tail),
    // which is more machinery than this feature's scope covers -- Dyn is
    // the same safe "can't prove it, don't hide it" fallback used
    // everywhere else in this checker.
    pub fn union(a: &EffectRow, b: &EffectRow) -> EffectRow {
        match (a, b) {
            (EffectRow::Var(x), EffectRow::Var(y)) if x == y => EffectRow::Var(x.clone()),
            (EffectRow::Var(v), EffectRow::Closed(s)) | (EffectRow::Closed(s), EffectRow::Var(v))
                if s.is_empty() =>
            {
                EffectRow::Var(v.clone())
            }
            (EffectRow::Var(_), _) | (_, EffectRow::Var(_)) => EffectRow::Dyn,
            (EffectRow::Dyn, _) | (_, EffectRow::Dyn) => EffectRow::Dyn,
            (EffectRow::Closed(x), EffectRow::Closed(y)) => {
                EffectRow::Closed(x.union(y).cloned().collect())
            }
        }
    }

    // Discharge one effect name (what a `handle` does to its body's row).
    // Unknown rows (Dyn, or an unresolved Var) can't be subtracted from --
    // stay as they are.
    pub fn remove(&self, effect: &str) -> EffectRow {
        match self {
            EffectRow::Dyn | EffectRow::Var(_) => self.clone(),
            EffectRow::Closed(s) => {
                let mut s = s.clone();
                s.remove(effect);
                EffectRow::Closed(s)
            }
        }
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
    pub bodies: Vec<Type>,
    // No body mentions (at any depth, nested groups included) a Type::Var, an Indexed, an
    // EffectRow::Var or a RecVar::Free: a walker that renames variables may return it as is.
    pub plain: bool,
    // How many enclosing groups the bodies reach through a Bound (0: closed). Lets `unfold`
    // skip a nested group that never mentions the group being unfolded.
    pub escape: u32,
}

impl RecGroup {
    pub fn new(names: Vec<Rc<str>>, bodies: Vec<Type>) -> RecGroup {
        let (mut plain, mut escape) = (true, 0);
        for body in &bodies {
            scan(body, 0, &mut plain, &mut escape);
        }
        RecGroup { names, bodies, plain, escape }
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

fn scan(t: &Type, nest: u32, plain: &mut bool, escape: &mut u32) {
    match t {
        Type::Var(_) | Type::Indexed(..) | Type::Fun(_, EffectRow::Var(_), _) | Type::RecVar(RecVar::Free { .. }) => *plain = false,
        Type::RecVar(RecVar::Bound { depth, .. }) if *depth > nest => *escape = (*escape).max(*depth - nest),
        Type::Rec(h, _) => {
            *plain &= h.0.plain;
            if h.0.escape > nest + 1 {
                *escape = (*escape).max(h.0.escape - (nest + 1));
            }
        }
        _ => {}
    }
    each_child(t, &mut |c| scan(c, nest, plain, escape));
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
        Type::List(e) => f(e).map(|e| Type::List(Rc::new(e))),
        Type::Fun(a, row, b) => {
            let (na, nb) = (f(a), f(b));
            if na.is_none() && nb.is_none() {
                return None;
            }
            Some(Type::Fun(na.map_or_else(|| a.clone(), Rc::new), row.clone(), nb.map_or_else(|| b.clone(), Rc::new)))
        }
        Type::Tuple(items) => map_vec(items, f).map(|v| Type::Tuple(Rc::new(v))),
        Type::Union(items) => map_vec(items, f).map(|v| Type::Union(Rc::new(v))),
        Type::Record(fields) => {
            let mapped: Vec<Option<Type>> = fields.iter().map(|(_, t)| f(t)).collect();
            if mapped.iter().all(Option::is_none) {
                return None;
            }
            Some(Type::Record(Rc::new(
                mapped.into_iter().zip(fields.iter()).map(|(m, (n, old))| (n.clone(), m.unwrap_or_else(|| old.clone()))).collect(),
            )))
        }
        Type::Indexed(w, i) => f(w).map(|w| Type::Indexed(Rc::new(w), i.clone())),
        _ => None,
    }
}

// The group rebuilt with `f` applied to each body; None when no body changed.
pub fn map_group(h: &RecRef, k: usize, f: &mut dyn FnMut(&Type) -> Option<Type>) -> Option<Type> {
    let bodies = map_vec(&h.0.bodies, f)?;
    Some(Type::Rec(RecRef(Rc::new(RecGroup::new(h.0.names.clone(), bodies))), k))
}

fn subst_bound(t: &Type, depth: u32, g: &RecRef) -> Option<Type> {
    match t {
        Type::RecVar(RecVar::Bound { depth: d, member }) if *d == depth => Some(Type::Rec(g.clone(), *member as usize)),
        Type::Rec(h, k) => {
            if h.0.escape <= depth {
                return None;
            }
            map_group(h, *k, &mut |b| subst_bound(b, depth + 1, g))
        }
        _ => map_children(t, &mut |c| subst_bound(c, depth, g)),
    }
}

// One step into member `i`: its body with every reference to the group replaced by the group
// itself. Only closed terms are unfolded, so nothing needs shifting. Subtrees that mention no
// reference are returned as the same Rcs.
pub fn unfold(g: &RecRef, i: usize) -> Type {
    subst_bound(&g.0.bodies[i], 0, g).unwrap_or_else(|| g.0.bodies[i].clone())
}

fn occurs_free(t: &Type, id: u64) -> bool {
    match t {
        Type::RecVar(RecVar::Free { id: i, .. }) => *i == id,
        Type::Rec(h, _) => !h.0.plain && h.0.bodies.iter().any(|b| occurs_free(b, id)),
        _ => {
            let mut found = false;
            each_child(t, &mut |c| found = found || occurs_free(c, id));
            found
        }
    }
}

// `t` with each Free placeholder `f(id, depth)` answers for replaced; depth counts the groups
// entered. None when nothing changed. Plain groups hold no placeholder and are skipped.
fn replace_free(t: &Type, depth: u32, f: &dyn Fn(u64, u32) -> Option<Type>) -> Option<Type> {
    match t {
        Type::RecVar(RecVar::Free { id, .. }) => f(*id, depth),
        Type::Rec(h, k) => {
            if h.0.plain {
                return None;
            }
            map_group(h, *k, &mut |b| replace_free(b, depth + 1, f))
        }
        _ => map_children(t, &mut |c| replace_free(c, depth, f)),
    }
}

// The name of the first placeholder `t` still contains (a type the machine built while a
// group's values were running and that escaped before the group closed).
pub fn first_free(t: &Type) -> Option<Rc<str>> {
    match t {
        Type::RecVar(RecVar::Free { name, .. }) => Some(name.clone()),
        Type::Rec(h, _) if !h.0.plain => h.0.bodies.iter().find_map(first_free),
        Type::Rec(..) => None,
        _ => {
            let mut found = None;
            each_child(t, &mut |c| {
                if found.is_none() {
                    found = first_free(c);
                }
            });
            found
        }
    }
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
// else None. Members some type mentions become a RecGroup, the rest keep their type with the
// placeholders replaced by the closed members. Err is (member, why).
pub fn close_group(names: &[Rc<str>], ids: &[u64], results: &[Option<Type>]) -> Result<Vec<Option<Type>>, (usize, CloseError)> {
    for (i, r) in results.iter().enumerate() {
        if let Some(Type::RecVar(RecVar::Free { id, .. })) = r
            && ids.contains(id)
        {
            return Err((i, CloseError::Bare));
        }
    }
    let referenced: Vec<usize> =
        (0..ids.len()).filter(|&j| results.iter().flatten().any(|t| occurs_free(t, ids[j]))).collect();
    if let Some(&j) = referenced.iter().find(|&&j| results[j].is_none()) {
        return Err((j, CloseError::NotAType));
    }
    if referenced.is_empty() {
        return Ok(results.to_vec());
    }
    let pos = |id: u64| ids.iter().position(|x| *x == id).and_then(|j| referenced.iter().position(|r| *r == j));
    let bodies: Vec<Type> = referenced
        .iter()
        .map(|&j| {
            let t = results[j].as_ref().expect("checked above");
            let bound = |id: u64, depth: u32| pos(id).map(|p| Type::RecVar(RecVar::Bound { depth, member: p as u32 }));
            replace_free(t, 0, &bound).unwrap_or_else(|| t.clone())
        })
        .collect();
    let g = RecRef(Rc::new(RecGroup::new(referenced.iter().map(|&j| names[j].clone()).collect(), bodies)));
    let closed = |id: u64, _: u32| pos(id).map(|p| Type::Rec(g.clone(), p));
    let out: Vec<Option<Type>> = results
        .iter()
        .enumerate()
        .map(|(j, r)| {
            r.as_ref().map(|t| match referenced.iter().position(|m| *m == j) {
                Some(p) => Type::Rec(g.clone(), p),
                None => replace_free(t, 0, &closed).unwrap_or_else(|| t.clone()),
            })
        })
        .collect();
    // Contractive: unfolding a member must reach a type constructor, never loop through members.
    for (p, &j) in referenced.iter().enumerate() {
        let mut keep = vec![g.clone()];
        let mut seen = HashSet::from([(Rc::as_ptr(&g.0), p)]);
        let mut cur = (g.clone(), p);
        while let Type::Rec(h, k) = unfold(&cur.0, cur.1) {
            if !seen.insert((Rc::as_ptr(&h.0), k)) {
                return Err((j, CloseError::Bare));
            }
            keep.push(h.clone());
            cur = (h, k);
        }
    }
    Ok(out)
}

// The name the user wrote for an index, row or type variable: the part before the first `#`
// (the rest is checker bookkeeping). Applied at message time only; stored names stay exact.
pub(crate) fn surface_name(name: &str) -> &str {
    name.split('#').next().unwrap_or(name)
}

impl fmt::Display for EffectRow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EffectRow::Dyn => write!(f, "Dyn"),
            EffectRow::Var(name) => write!(f, "{{{}}}", surface_name(name)),
            EffectRow::Closed(s) => {
                write!(f, "{{{}}}", s.iter().cloned().collect::<Vec<_>>().join(", "))
            }
        }
    }
}

// Consistency, not equality: Dyn is consistent with everything. This is
// what makes it "gradual" -- an annotated boundary only rejects code where
// BOTH sides are concretely known and disagree. Effect rows follow the
// same rule: Dyn row is consistent with any row, two Closed rows must
// match exactly (no row subtyping/variance modeled yet).
//
// No registry, no cycle guard: every Type here is a plain finite tree
// (Tuple/Union hold an Rc<Vec<Type>>, never a back-reference to
// themselves -- nothing in this checker ever builds a genuinely-cyclic
// Type value), so a plain structural recursion always terminates.
pub fn consistent(a: &Type, b: &Type) -> bool {
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
        (Type::List(a), Type::List(b)) => consistent(a, b),
        (Type::Fun(a1, r1, b1), Type::Fun(a2, r2, b2)) => {
            consistent(a1, a2) && consistent(b1, b2) && row_consistent(r1, r2)
        }
        // A singleton per source position -- consistent only with itself
        // (and Dyn, already handled above). No registry lookup: the id
        // alone says everything there is to say.
        (Type::Token(a), Type::Token(b)) => a == b,
        // Nominal, not structural -- same singleton-by-id precedent
        // Token's own arm just above already sets. Two Named types are
        // consistent only when their ids match exactly; a Named type
        // is never consistent with anything structurally identical to
        // what it (or any other Named type) unfolds to. Never unfolds
        // the registry -- see Type::Named's own doc comment.
        (Type::Named(a), Type::Named(b)) => a == b,
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
            consistent(wa, wb)
                && (index_exprs_equal(ia, ib) || matches!(**ia, IndexExpr::Var(_)) || matches!(**ib, IndexExpr::Var(_)))
        }
        // Positional, structural, no name/registry involved -- same idea
        // as List's own element comparison, just per-position instead of
        // one shared element type.
        (Type::Tuple(a), Type::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| consistent(x, y))
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
        (Type::Record(a), Type::Record(b)) => a.len() == b.len() && record_satisfies(a, b),
        // Two unions: consistent as SETS -- every alternative on each
        // side has a match on the other (order and duplicates don't
        // matter, only membership). Must come before the single-Union
        // arm below (Rust checks patterns top to bottom; that arm's
        // `other: &Type` would otherwise also match a Union on the other
        // side).
        (Type::Union(a), Type::Union(b)) => {
            a.iter().all(|x| b.iter().any(|y| consistent(x, y))) && b.iter().all(|y| a.iter().any(|x| consistent(x, y)))
        }
        // One union, one concrete type: consistent if the concrete side
        // matches ANY ONE alternative -- "is this one of the possible
        // shapes," the same question a runtime value crossing a
        // Union-typed boundary would need answered.
        (Type::Union(alts), other) | (other, Type::Union(alts)) => alts.iter().any(|alt| consistent(alt, other)),
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
    required.iter().all(|(name, ty)| find_field(actual, name).is_some_and(|aty| consistent(ty, aty)))
}

pub fn row_consistent(a: &EffectRow, b: &EffectRow) -> bool {
    match (a, b) {
        (EffectRow::Dyn, _) | (_, EffectRow::Dyn) => true,
        (EffectRow::Var(_), _) | (_, EffectRow::Var(_)) => true,
        (EffectRow::Closed(x), EffectRow::Closed(y)) => x == y,
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Dyn => write!(f, "Dyn"),
            Type::Int => write!(f, "Int"),
            Type::Float => write!(f, "Float"),
            // The name is internal bookkeeping (see Type::Var's own doc
            // comment) that a user has no way to interpret -- render
            // identically to Type::Dyn, which is what a Type::Var
            // actually behaves like everywhere it's observable.
            Type::Var(_) => write!(f, "Dyn"),
            // Shows the clean surface name, not the internal gensym'd
            // id -- mirrors Type::Var's own precedent just above
            // (hiding its internal name, always showing "Dyn" instead)
            // of not exposing machinery the user never wrote. The
            // surface name is always the part before the first "#" --
            // see the id's own construction in parser::atom.
            Type::Named(id) => write!(f, "{}", surface_name(id)),
            // The member's own name, never its body.
            Type::Rec(g, i) => write!(f, "{}", g.0.names[*i]),
            Type::RecVar(RecVar::Free { name, .. }) => write!(f, "{name}"),
            // Unreachable by construction (a Bound only lives inside a group's bodies).
            Type::RecVar(RecVar::Bound { depth, member }) => write!(f, "?{depth}.{member}"),
            Type::Bool => write!(f, "Bool"),
            Type::Str => write!(f, "Str"),
            Type::List(elem) => write!(f, "[{elem}]"),
            // Dyn row prints as a plain arrow -- matches every existing
            // (unannotated-row) Fun type exactly as before this existed.
            Type::Fun(a, EffectRow::Dyn, b) => write!(f, "({a} -> {b})"),
            Type::Fun(a, row, b) => write!(f, "({a} ->{row} {b})"),
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
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
            Type::Record(fields) => {
                write!(f, "{{")?;
                for (i, (name, t)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {t}")?;
                }
                write!(f, "}}")
            }
            Type::Union(alts) => {
                for (i, t) in alts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{t}")?;
                }
                Ok(())
            }
            Type::Indexed(wrapped, index) => write!(f, "{wrapped}({index})"),
        }
    }
}
