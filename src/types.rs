use std::collections::{BTreeSet, HashMap, HashSet};
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
    // qualifying Type::Rec+Type::Union (derived index-refinement, a
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
        if !referenced[comp[0]] {
            // A lone member nothing mentions (a cycle would mention itself): no group.
            let m = comp[0];
            out[m] = results[m].as_ref().map(|t| replace_free(t, 0, &|id, d| outside(id, d, &closed)).unwrap_or_else(|| t.clone()));
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
    match t {
        Type::RecVar(RecVar::Free { id, .. }) => found.extend(index.get(id)),
        Type::Rec(h, _) => {
            if !h.0.plain {
                h.0.bodies.iter().for_each(|b| collect_free(b, index, found));
            }
        }
        _ => each_child(t, &mut |c| collect_free(c, index, found)),
    }
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
    fn skeleton(t: &Type, colour: &dyn Fn(u64) -> Option<u64>, h: &mut std::collections::hash_map::DefaultHasher) {
        match t {
            Type::RecVar(RecVar::Free { id, .. }) => match colour(*id) {
                Some(k) => (1u8, k).hash(h),
                None => (2u8, id).hash(h),
            },
            Type::RecVar(RecVar::Bound { depth, member }) => (3u8, depth, member).hash(h),
            Type::Rec(g, k) => {
                (4u8, k, g.0.bodies.len()).hash(h);
                if !g.0.plain {
                    g.0.bodies.iter().for_each(|b| skeleton(b, colour, h));
                }
            }
            Type::Token(id) => (5u8, id).hash(h),
            Type::Record(fields) => (6u8, fields.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>()).hash(h),
            _ => std::mem::discriminant(t).hash(h),
        }
        if !matches!(t, Type::Rec(..)) {
            each_child(t, &mut |ch| skeleton(ch, colour, h));
        }
    }
    let mut colours: HashMap<usize, u64> = comp.iter().map(|&m| (m, 0)).collect();
    let mut distinct = 1;
    for _ in 0..8 {
        let next: HashMap<usize, u64> = comp
            .iter()
            .map(|&m| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                let of = |id: u64| index.get(&id).filter(|&&j| scc_of[j] == c).map(|j| colours[j]);
                skeleton(&bodies[&m], &of, &mut h);
                (m, h.finish())
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
// No cycle guard: every Type here is a finite tree (a recursive type is a closed mu-term whose
// back-references are Bound indices, never pointers), so a plain structural recursion always
// terminates.
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
