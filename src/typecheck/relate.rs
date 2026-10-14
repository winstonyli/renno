//! Directional, infer-aware type relation (design 2026-10-08, section 3.1). Pure and read-only.
//! S2 T1 added `Why`/`Undecided`/`relate_cast`; consumers: coerce_check and the unify_fits rescue (T3), coerce_cast (T4).
use super::{free_index_vars, unify_index_expr, InferCtx};
use crate::index_expr::{index_exprs_compare, IndexCmp, IndexExpr};
use crate::types::{is_big, row_consistent, unfold, ById, RecRef, Type};
use crate::util::find_field;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
#[cfg(test)]
use std::collections::BTreeSet;
/// Native-stack guard (a source type that is itself very deep). Degenerate aliases are cut earlier
/// by the revisit guard in the `Rec` arm.
const MAX_DEPTH: usize = 64;
/// Total `go` calls per `relate`: a breadth backstop for pathological unions.
const FUEL: u32 = 10_000;
/// Imprecise = gradual by construction (Dyn, Var, Union, free index variable); Incomplete = both
/// sides precise but the checker could not decide (cap, solving, fuel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Cause {
    Imprecise,
    Incomplete,
}
/// The runtime work that would decide an Unknown. Unit placeholders: payloads were not built (S3 lowers from `to`). `Fun` is
/// a wrapper around a function value (no value test); a Dyn used as a function is `Test` + `Fun`.
/// A set of the kinds of work an Unknown mentions. Nothing reads how they combine (all of, any
/// of), only which kinds occur, so a union of bits is enough and costs nothing to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Residual(u8);
impl Residual {
    pub(crate) const TEST: Residual = Residual(1);
    pub(crate) const FUN: Residual = Residual(2);
    pub(crate) const LEN: Residual = Residual(4);
    pub(crate) const INDEX_EQ: Residual = Residual(8);
    /// A cap was hit.
    pub(crate) const LAZY: Residual = Residual(16);
    /// No value test, wrapper or unifier step decides it: a static error.
    pub(crate) const UNDECIDED: Residual = Residual(32);
    #[cfg(test)]
    fn collect(self, out: &mut BTreeSet<&'static str>) {
        let names = [(Self::TEST, "Test"), (Self::FUN, "Fun"), (Self::LEN, "Len"), (Self::INDEX_EQ, "IndexEq"), (Self::LAZY, "Lazy"), (Self::UNDECIDED, "Undecided")];
        out.extend(names.iter().filter(|(bit, _)| self.0 & bit.0 != 0).map(|(_, name)| *name));
    }
    /// Some part is decided by a value test (lowered by `build_boundary_check`).
    pub(crate) fn needs_test(self) -> bool {
        self.0 & (Self::TEST.0 | Self::LEN.0 | Self::LAZY.0) != 0
    }
    pub(crate) fn undecided(self) -> bool {
        self.0 & Self::UNDECIDED.0 != 0
    }
}
impl std::ops::BitOr for Residual {
    type Output = Residual;
    fn bitor(self, other: Residual) -> Residual {
        Residual(self.0 | other.0)
    }
}
/// Why a pair is Refuted. Only a `Shape` refutation is seen by the value test that lowers a
/// `Residual::TEST`, which matters when it is one alternative of a Union source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Why {
    /// A value test tells them apart: base type, tuple arity, missing field, list vs tuple.
    Shape(&'static str),
    /// A value test cannot: a function signature or effect row, a nominal id, an index position.
    Opaque(&'static str),
    /// An index pair; `msg` is `unify_index_expr`'s text for it. `bare`: either index as written
    /// is a bare variable (the old gate let those through to the unifier, whose text users see).
    Index { msg: String, bare: bool },
}
impl Why {
    pub(crate) fn message(&self) -> Cow<'static, str> {
        match self {
            Why::Shape(m) | Why::Opaque(m) => Cow::Borrowed(m),
            Why::Index { msg, .. } => Cow::Owned(msg.clone()),
        }
    }
    fn observable(&self) -> bool {
        matches!(self, Why::Shape(_))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Rel {
    Proven,
    Unknown(Residual, Cause),
    Refuted(Why),
}
impl Rel {
    fn refuted(why: &'static str) -> Rel { Rel::Refuted(Why::Shape(why)) }
    fn opaque(why: &'static str) -> Rel { Rel::Refuted(Why::Opaque(why)) }
    /// The distinct residual kinds an Unknown mentions (empty for Proven/Refuted).
    #[cfg(test)]
    pub(crate) fn kinds(&self) -> BTreeSet<&'static str> {
        let mut out = BTreeSet::new();
        if let Rel::Unknown(r, _) = self {
            r.collect(&mut out);
        }
        out
    }
    #[cfg(test)]
    pub(crate) fn cause(&self) -> Option<Cause> {
        if let Rel::Unknown(_, c) = self { Some(*c) } else { None }
    }
}
// Conjunction: Refuted dominates, then Unknown (worst cause), else Proven.
fn and(rels: impl IntoIterator<Item = Rel>) -> Rel {
    let (mut unknown, mut cause) = (None, Cause::Imprecise);
    for r in rels {
        match r {
            Rel::Refuted(why) => return Rel::Refuted(why),
            Rel::Unknown(res, c) => {
                cause = cause.max(c);
                unknown = Some(unknown.map_or(res, |seen| seen | res));
            }
            Rel::Proven => {}
        }
    }
    unknown.map_or(Rel::Proven, |res| Rel::Unknown(res, cause))
}
// Disjunction over target-Union alternatives (the caller already returned on a Proven one):
// all Refuted refutes; otherwise Unknown, as precise as its most precise Unknown alternative.
fn or(rels: Vec<Rel>) -> Rel {
    let (mut unknown, mut cause, mut why) = (None, Cause::Incomplete, Why::Shape("empty union"));
    for r in rels {
        match r {
            Rel::Unknown(res, c) => {
                cause = cause.min(c);
                unknown = Some(unknown.map_or(res, |seen| seen | res));
            }
            Rel::Refuted(w) => if why.observable() { why = w },
            Rel::Proven => return Rel::Proven,
        }
    }
    unknown.map_or(Rel::Refuted(why), |res| Rel::Unknown(res, cause))
}
/// Per-`relate` bookkeeping: remaining `go` calls, and the (source, recursive type) pairs being unfolded.
struct Budget {
    left: Cell<u32>,
    // A Rec target being unfolded: the source, and the group (kept, so its pointer stays
    // unique) and member.
    unfolding_rec: RefCell<Vec<(Type, RecRef, usize)>>,
    ignore_rows: bool,
    // The decided pairs of big nodes, by (source, target, context): a shared node is related
    // once per context, not once per path to it. The context names the stack of recursive
    // unfoldings above a call (0: none; each unfolding gets a fresh number for its extent), since
    // the revisit guard makes an answer depend on that stack. A pair whose computation hit a cap
    // (`caps` counts them) is never stored, so a stored answer never depends on how much budget
    // was left.
    pairs: RefCell<HashMap<(ById, ById, u32), Rel>>,
    context: Cell<u32>,
    contexts: Cell<u32>,
    caps: Cell<u32>,
}
impl Default for Budget {
    fn default() -> Self {
        Budget {
            left: Cell::new(FUEL),
            unfolding_rec: RefCell::new(Vec::new()),
            ignore_rows: false,
            pairs: RefCell::new(HashMap::new()),
            context: Cell::new(0),
            contexts: Cell::new(0),
            caps: Cell::new(0),
        }
    }
}
fn loose(t: &Type) -> bool {
    matches!(t, Type::Dyn | Type::Var(_))
}
/// Can a value of type `from` be used where `to` is expected? Resolves bound type and index variables.
pub(crate) fn relate(from: &Type, to: &Type, infer: &InferCtx) -> Rel {
    go(from, to, infer, &Budget::default(), 0)
}
/// `relate` for a cast: effect rows have no runtime representation, so they never decide a cast.
pub(crate) fn relate_cast(from: &Type, to: &Type, infer: &InferCtx) -> Rel {
    go(from, to, infer, &Budget { ignore_rows: true, ..Budget::default() }, 0)
}
fn go(from: &Type, to: &Type, infer: &InferCtx, fuel: &Budget, depth: usize) -> Rel {
    // Structurally identical types relate whatever their size (every arm below proves a pair of
    // equal components), so a type deeper or wider than the caps is not Unknown against itself.
    if from == to {
        return Rel::Proven;
    }
    let (from, to) = (infer.resolve(from), infer.resolve(to));
    let key = (is_big(&from) || is_big(&to)).then(|| (ById(from.clone()), ById(to.clone()), fuel.context.get()));
    if let Some(done) = key.as_ref().and_then(|k| fuel.pairs.borrow().get(k).cloned()) {
        return done;
    }
    if depth > MAX_DEPTH || fuel.left.get() == 0 {
        fuel.caps.set(fuel.caps.get() + 1);
        return Rel::Unknown(Residual::LAZY, Cause::Incomplete);
    }
    fuel.left.set(fuel.left.get() - 1);
    let caps = fuel.caps.get();
    let result = decide(from, to, infer, fuel, depth + 1);
    // A result computed past a cap (even one a union folded into a plain test) is not stored.
    if let Some(key) = key
        && fuel.caps.get() == caps
    {
        fuel.pairs.borrow_mut().insert(key, result.clone());
    }
    result
}
fn decide(from: Type, to: Type, infer: &InferCtx, fuel: &Budget, d: usize) -> Rel {
    let unknown = |r: Residual| Rel::Unknown(r, Cause::Imprecise);
    match (&from, &to) {
        (Type::Var(a), Type::Var(b)) if a == b => Rel::Proven,
        (Type::Fun(..), Type::Dyn) => fun_rel(&from, &to, infer, fuel, d), // cast as Fun(Dyn, Dyn)
        (_, t) if loose(t) => Rel::Proven,
        (f, Type::Fun(..)) if loose(f) => unknown(Residual::TEST | Residual::FUN),
        (f, Type::Indexed(..)) if loose(f) => unknown(Residual::TEST | Residual::LEN),
        (f, _) if loose(f) => unknown(Residual::TEST),
        (Type::Union(alts), _) => {
            let rels: Vec<Rel> = alts.iter().map(|a| go(a, &to, infer, fuel, d)).collect();
            if rels.iter().all(|r| *r == Rel::Proven) {
                Rel::Proven
            } else if !rels.is_empty() && rels.iter().all(|r| matches!(r, Rel::Refuted(_))) {
                let seen = rels.iter().all(|r| matches!(r, Rel::Refuted(w) if w.observable()));
                Rel::Refuted(if seen { Why::Shape("union: no alternative fits") } else { Why::Opaque("union: no alternative fits") })
            } else if alts.iter().zip(&rels).any(|(a, r)| match r {
                // A value test of `to` cannot see this refutation, nor add a function wrapper.
                Rel::Refuted(w) => !w.observable(),
                // An Undecided nested in the alternative stays Undecided. Otherwise only an Imprecise
                // Unknown Fun alternative is a wrapper to add; Incomplete (a cap) stays a runtime test.
                Rel::Unknown(res, c) => res.undecided() || (*c == Cause::Imprecise && matches!(infer.resolve(a), Type::Fun(..))),
                Rel::Proven => false,
            }) {
                Rel::Unknown(Residual::UNDECIDED, Cause::Incomplete)
            } else {
                unknown(Residual::TEST)
            }
        }
        (_, Type::Union(alts)) => {
            let mut rels = Vec::new();
            for a in alts.iter() {
                let r = go(&from, a, infer, fuel, d);
                if r == Rel::Proven {
                    return r;
                }
                rels.push(r);
            }
            or(rels)
        }
        (Type::Indexed(wf, i), Type::Indexed(wt, j)) => and([go(wf, wt, infer, fuel, d), relate_index(i, j, infer)]),
        (Type::Indexed(wf, _), _) => go(wf, &to, infer, fuel, d), // forgetting an index is sound; the reverse is not
        (_, Type::Indexed(..)) => Rel::opaque("plain type into indexed"),
        // Equal pairs were proven by `from == to` above. No directional body relating: Amber's
        // rule without an assumption set is unsound for a contravariant occurrence.
        (Type::Rec(..), Type::Rec(..)) => Rel::opaque("recursive: different types"),
        (_, Type::Rec(g, i)) => {
            let hit = |(f, h, k): &(Type, RecRef, usize)| k == i && std::rc::Rc::ptr_eq(&h.0, &g.0) && *f == from;
            if fuel.unfolding_rec.borrow().iter().any(hit) {
                return Rel::refuted("recursive: cyclic unfolding");
            }
            fuel.unfolding_rec.borrow_mut().push((from.clone(), g.clone(), *i));
            let outer = fuel.context.replace(fuel.contexts.get() + 1);
            fuel.contexts.set(fuel.contexts.get() + 1);
            let r = go(&from, &unfold(g, *i), infer, fuel, d);
            fuel.context.set(outer);
            fuel.unfolding_rec.borrow_mut().pop();
            r
        }
        (Type::Rec(..), _) => Rel::opaque("recursive source"),
        (Type::Int, Type::Int) | (Type::Float, Type::Float) | (Type::Bool, Type::Bool) | (Type::Str, Type::Str) => Rel::Proven,
        (Type::Token(a), Type::Token(b)) => {
            if a == b { Rel::Proven } else { Rel::refuted("token") }
        }
        (Type::List(a), Type::List(b)) => go(a, b, infer, fuel, d),
        (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => and(a.iter().zip(b.iter()).map(|(x, y)| go(x, y, infer, fuel, d))),
        (Type::Tuple(_), Type::Tuple(_)) => Rel::refuted("tuple arity"),
        // The tuple/list bridge exists only for a Dyn/Var element (types::consistent's has Dyn only).
        (Type::List(e), Type::Tuple(_)) => {
            if loose(&infer.resolve(e)) { unknown(Residual::TEST) } else { Rel::refuted("list into tuple") }
        }
        (Type::Tuple(_), Type::List(e)) => {
            if loose(&infer.resolve(e)) { Rel::Proven } else { Rel::refuted("tuple into list") }
        }
        (Type::Record(a), Type::Record(b)) => and(b.iter().map(|(name, t)| match find_field(a, name) {
            Some(ft) => go(ft, t, infer, fuel, d),
            None => Rel::refuted("missing field"),
        })),
        (Type::Fun(..), Type::Fun(..)) => fun_rel(&from, &to, infer, fuel, d),
        _ => Rel::refuted("shape"),
    }
}
// Params contravariant, returns covariant; any Unknown component is a wrapper (Residual::FUN),
// except an Undecided, which no wrapper can discharge and so propagates.
fn fun_rel(from: &Type, to: &Type, infer: &InferCtx, fuel: &Budget, d: usize) -> Rel {
    let Type::Fun(a, r, b) = from else { return Rel::refuted("not a function") };
    let (a2, b2) = match to {
        Type::Fun(a2, r2, b2) => {
            if !fuel.ignore_rows && !row_consistent(r, r2) {
                return Rel::opaque("effect row");
            }
            ((**a2).clone(), (**b2).clone())
        }
        _ => (Type::Dyn, Type::Dyn),
    };
    match and([go(&a2, a, infer, fuel, d), go(b, &b2, infer, fuel, d)]) {
        Rel::Unknown(res, c) if !res.undecided() => Rel::Unknown(Residual::FUN, c),
        Rel::Refuted(Why::Shape(w)) => Rel::Refuted(Why::Opaque(w)),
        other => other,
    }
}
/// Read-only index half of `unify_index_expr`. Equal -> Proven; cap/overflow -> Unknown(IndexEq,
/// Incomplete); differ by a nonzero constant -> Refuted; a bare flexible variable (bound by
/// unification) -> Unknown(IndexEq, Imprecise); a compound with a flexible variable (needs
/// solving) -> Unknown(IndexEq, Incomplete); only rigid variables/literals left -> Refuted.
fn relate_index(a: &IndexExpr, b: &IndexExpr, infer: &InferCtx) -> Rel {
    let (ra, rb) = (infer.resolve_index_deep(a), infer.resolve_index_deep(b));
    match index_exprs_compare(&ra, &rb) {
        None => return Rel::Unknown(Residual::INDEX_EQ, Cause::Incomplete),
        Some(IndexCmp::Equal) => return Rel::Proven,
        Some(IndexCmp::NonzeroConst) => return index_refuted(a, b, infer),
        Some(IndexCmp::Other) => {}
    }
    let flexible = |v: &String| !infer.rigid_index.contains(v);
    let bare_flexible = |e: &IndexExpr| matches!(e, IndexExpr::Var(v) if flexible(v));
    let mut vars = free_index_vars(&ra);
    vars.extend(free_index_vars(&rb));
    if bare_flexible(&ra) || bare_flexible(&rb) {
        Rel::Unknown(Residual::INDEX_EQ, Cause::Imprecise)
    } else if vars.iter().any(flexible) {
        Rel::Unknown(Residual::INDEX_EQ, Cause::Incomplete)
    } else {
        index_refuted(a, b, infer)
    }
}
/// The unifier's own text for a refuted index pair (run on a scratch copy; nothing is bound).
fn index_refuted(a: &IndexExpr, b: &IndexExpr, infer: &InferCtx) -> Rel {
    let mut scratch = InferCtx::new();
    scratch.index_subst = infer.index_subst.clone();
    scratch.rigid_index = infer.rigid_index.clone();
    // (required, actual) = (target, source), the order unify_fits uses.
    let msg = match unify_index_expr(b, a, &mut scratch) {
        Rel::Refuted(w) => w.message().into_owned(),
        _ => format!("type mismatch: index {b} does not unify with index {a}"), // guard: relate refutes only pairs unify cannot bind
    };
    Rel::Refuted(Why::Index { msg, bare: matches!(a, IndexExpr::Var(_)) || matches!(b, IndexExpr::Var(_)) })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EffectRow, Hc};
    use std::rc::Rc;
    fn list(t: Type) -> Type {
        Type::List(Hc::new(t))
    }
    fn tup(ts: &[Type]) -> Type {
        Type::Tuple(Hc::new(ts.to_vec()))
    }
    fn uni(ts: &[Type]) -> Type {
        Type::Union(Hc::new(ts.to_vec()))
    }
    fn rec(fs: &[(&str, Type)]) -> Type {
        Type::Record(Hc::new(fs.iter().map(|(n, t)| (n.to_string(), t.clone())).collect()))
    }
    fn fun(a: Type, b: Type) -> Type {
        Type::Fun(Hc::new(a), EffectRow::Dyn, Hc::new(b))
    }
    fn var(n: &str) -> Type {
        Type::Var(n.to_string())
    }
    fn rec_group(bodies: Vec<Type>) -> crate::types::RecRef {
        use crate::types::{RecGroup, RecRef};
        RecRef(Rc::new(RecGroup::new(vec![Rc::from("L")], bodies)))
    }
    fn bound0() -> Type {
        Type::RecVar(crate::types::RecVar::Bound { depth: 0, member: 0 })
    }
    // L = (Int, L) | Bool
    fn l_ty() -> Type {
        Type::Rec(rec_group(vec![uni(&[tup(&[Type::Int, bound0()]), Type::Bool])]), 0)
    }
    fn iv(n: &str) -> IndexExpr {
        IndexExpr::Var(n.to_string())
    }
    fn lit(k: i64) -> IndexExpr {
        IndexExpr::Lit(k)
    }
    fn add(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        IndexExpr::Add(Rc::new(a), Rc::new(b))
    }
    fn vec_n(i: IndexExpr) -> Type {
        Type::Indexed(Hc::new(list(Type::Dyn)), Rc::new(i))
    }
    // a := Int, s := Str, u unbound; index n rigid, z := 0.
    fn ctx() -> InferCtx {
        let mut i = InferCtx::new();
        i.subst.insert("a".to_string(), Type::Int);
        i.subst.insert("s".to_string(), Type::Str);
        i.rigid_index.insert("n".to_string());
        i.index_subst.insert("z".to_string(), lit(0));
        i
    }
    enum Exp {
        P,
        R,
        U(&'static [&'static str], Cause),
    }
    use Cause::{Imprecise as Imp, Incomplete as Inc};
    use Exp::{P, R, U};
    #[test]
    fn relate_table() {
        let mut big = add(iv("a"), iv("b"));
        for _ in 0..6 {
            big = IndexExpr::Mul(Rc::new(big.clone()), Rc::new(big));
        }
        let (i, s, d, b) = (Type::Int, Type::Str, Type::Dyn, Type::Bool);
        let is = uni(&[i.clone(), s.clone()]);
        let ii = tup(&[i.clone(), i.clone()]);
        let fx = |p: Type, r: Type, row: EffectRow| Type::Fun(Hc::new(p), row, Hc::new(r));
        let cases: Vec<(&str, Type, Type, Exp)> = vec![
            ("int-int", i.clone(), i.clone(), P),
            ("int-str", i.clone(), s.clone(), R),
            ("int-float policy", i.clone(), Type::Float, R),
            ("float-int policy", Type::Float, i.clone(), R),
            ("dyn-int", d.clone(), i.clone(), U(&["Test"], Imp)),
            ("int-dyn", i.clone(), d.clone(), P),
            ("unbound var-int", var("u"), i.clone(), U(&["Test"], Imp)),
            ("int-unbound var", i.clone(), var("u"), P),
            ("bound var ok", var("a"), i.clone(), P),
            ("bound var bad", var("s"), i.clone(), R),
            ("list int-dyn", list(i.clone()), list(d.clone()), P),
            ("list dyn-int", list(d.clone()), list(i.clone()), U(&["Test"], Imp)),
            ("list int-str", list(i.clone()), list(s.clone()), R),
            ("tuple dyn field", tup(&[d.clone(), i.clone()]), ii.clone(), U(&["Test"], Imp)),
            ("tuple arity", tup(std::slice::from_ref(&i)), ii.clone(), R),
            ("tuple-list dyn", ii.clone(), list(d.clone()), P),
            ("list dyn-tuple", list(d.clone()), ii.clone(), U(&["Test"], Imp)),
            ("list int-tuple", list(i.clone()), ii.clone(), R),
            ("list unbound-var-tuple (deliberate Var bridge)", list(var("u")), ii.clone(), U(&["Test"], Imp)),
            ("tuple-list unbound-var (deliberate Var bridge)", ii.clone(), list(var("u")), P),
            ("int into union", i.clone(), is.clone(), P),
            ("union into int", is.clone(), i.clone(), U(&["Test"], Imp)),
            ("union same", is.clone(), is.clone(), P),
            ("union narrowing is directional", uni(&[i.clone(), b.clone()]), is.clone(), U(&["Test"], Imp)),
            ("bool into union", b.clone(), is.clone(), R),
            ("union into dyn", is.clone(), d.clone(), P),
            ("record width", rec(&[("x", i.clone()), ("y", i.clone())]), rec(&[("x", i.clone())]), P),
            ("record missing", rec(&[("x", i.clone())]), rec(&[("x", i.clone()), ("y", i.clone())]), R),
            ("record dyn field", rec(&[("x", d.clone())]), rec(&[("x", i.clone())]), U(&["Test"], Imp)),
            ("fun same", fun(i.clone(), i.clone()), fun(i.clone(), i.clone()), P),
            ("fun typed into dyn", fun(i.clone(), i.clone()), d.clone(), U(&["Fun"], Imp)),
            ("fun dyn into dyn", fun(d.clone(), d.clone()), d.clone(), P),
            ("dyn into fun", d.clone(), fun(i.clone(), i.clone()), U(&["Fun", "Test"], Imp)),
            ("fun result dyn (h02)", fun(i.clone(), d.clone()), fun(i.clone(), i.clone()), U(&["Fun"], Imp)),
            ("fun wider param ok", fun(is.clone(), i.clone()), fun(i.clone(), i.clone()), P),
            ("fun narrower param", fun(i.clone(), i.clone()), fun(is.clone(), i.clone()), U(&["Fun"], Imp)),
            ("fun rows differ", fx(i.clone(), i.clone(), EffectRow::single("a")), fx(i.clone(), i.clone(), EffectRow::pure()), R),
            ("rec same", l_ty(), l_ty(), P),
            ("rec into int", l_ty(), i.clone(), R),
            ("literal one level into L", tup(&[i.clone(), b.clone()]), l_ty(), P),
            ("literal two levels into L (t10)", tup(&[i.clone(), ii.clone()]), l_ty(), R),
            ("dyn into L", d.clone(), l_ty(), U(&["Test"], Imp)),
            ("vec same", vec_n(lit(3)), vec_n(lit(3)), P),
            ("vec literal mismatch", vec_n(lit(3)), vec_n(lit(4)), R),
            ("vec into flexible var", vec_n(lit(3)), vec_n(iv("m")), U(&["IndexEq"], Imp)),
            ("rigid vs literal", vec_n(iv("n")), vec_n(lit(3)), R),
            ("rigid vs flexible compound (x2)", vec_n(iv("n")), vec_n(add(iv("k"), lit(1))), U(&["IndexEq"], Inc)),
            ("rigid vs n+1", vec_n(iv("n")), vec_n(add(iv("n"), lit(1))), R),
            ("sop equal", vec_n(add(iv("n"), lit(1))), vec_n(add(lit(1), iv("n"))), P),
            ("sop cap (x1)", vec_n(big.clone()), vec_n(add(big, lit(1))), U(&["IndexEq"], Inc)),
            ("index_subst applied (m6)", vec_n(lit(1)), vec_n(add(iv("z"), lit(1))), P),
            ("vec into plain list", vec_n(lit(3)), list(d.clone()), P),
            ("plain list into vec", list(d.clone()), vec_n(lit(3)), R),
            ("dyn into vec", d.clone(), vec_n(lit(3)), U(&["Len", "Test"], Imp)),
            ("fun into unbound var", fun(i.clone(), i.clone()), var("u"), P),
            ("union mix, opaque alt (fun signature)", uni(&[fun(i.clone(), i.clone()), b.clone()]), uni(&[fun(s.clone(), i.clone()), b.clone()]), U(&["Undecided"], Inc)),
            ("union mix, index alt", uni(&[vec_n(lit(3)), b.clone()]), uni(&[vec_n(lit(4)), b.clone()]), U(&["Undecided"], Inc)),
            ("union mix, fun alt needs a wrapper", uni(&[fun(i.clone(), i.clone()), b.clone()]), uni(&[fun(d.clone(), d.clone()), b.clone()]), U(&["Undecided"], Inc)),
            ("undecided in a tuple component", tup(&[uni(&[fun(i.clone(), i.clone()), b.clone()]), i.clone()]), tup(&[uni(&[fun(d.clone(), d.clone()), b.clone()]), i.clone()]), U(&["Undecided"], Inc)),
            ("undecided in a record field", rec(&[("p", uni(&[fun(i.clone(), i.clone()), b.clone()]))]), rec(&[("p", uni(&[fun(d.clone(), d.clone()), b.clone()]))]), U(&["Undecided"], Inc)),
            ("undecided in a non-fun alt of a union source", uni(&[tup(&[uni(&[fun(i.clone(), i.clone()), b.clone()]), i.clone()]), s.clone()]), uni(&[tup(&[uni(&[fun(d.clone(), d.clone()), b.clone()]), i.clone()]), s.clone()]), U(&["Undecided"], Inc)),
            ("undecided under an arrow parameter", fun(uni(&[fun(d.clone(), d.clone()), b.clone()]), i.clone()), fun(uni(&[fun(i.clone(), i.clone()), b.clone()]), i.clone()), U(&["Undecided"], Inc)),
            ("undecided under an arrow result", fun(i.clone(), uni(&[fun(i.clone(), i.clone()), b.clone()])), fun(i.clone(), uni(&[fun(d.clone(), d.clone()), b.clone()])), U(&["Undecided"], Inc)),
            ("same nested union-with-fun is Proven", tup(&[uni(&[fun(i.clone(), i.clone()), b.clone()]), i.clone()]), tup(&[uni(&[fun(i.clone(), i.clone()), b.clone()]), i.clone()]), P),
            ("union mix, shape only stays a test", uni(&[i.clone(), s.clone()]), uni(&[i.clone(), b.clone()]), U(&["Test"], Imp)),
        ];
        let infer = ctx();
        for (name, from, to, exp) in cases {
            let got = relate(&from, &to, &infer);
            let ok = match (&exp, &got) {
                (P, Rel::Proven) | (R, Rel::Refuted(_)) => true,
                (U(kinds, cause), Rel::Unknown(..)) => got.kinds().into_iter().collect::<Vec<_>>().as_slice() == *kinds && got.cause() == Some(*cause),
                _ => false,
            };
            assert!(ok, "{name}: {from} -> {to}: got {got:?}");
        }
    }
    #[test]
    fn refutations_carry_their_kind_and_the_unifier_text() {
        let infer = ctx();
        let (i, s) = (Type::Int, Type::Str);
        assert_eq!(relate(&i, &s, &infer), Rel::Refuted(Why::Shape("shape")));
        let m = Type::Rec(rec_group(vec![list(bound0())]), 0);
        assert!(matches!(relate(&m, &l_ty(), &infer), Rel::Refuted(Why::Opaque(_))));
        assert!(matches!(relate(&fun(i.clone(), i.clone()), &fun(s.clone(), i.clone()), &infer), Rel::Refuted(Why::Opaque(_))));
        let rigid = Why::Index { msg: "index variable n is fixed by the signature and cannot equal 3".into(), bare: true };
        assert_eq!(relate(&vec_n(iv("n")), &vec_n(lit(3)), &infer), Rel::Refuted(rigid));
        let lits = Why::Index { msg: "type mismatch: index 4 does not unify with index 3".into(), bare: false };
        assert_eq!(relate(&vec_n(lit(3)), &vec_n(lit(4)), &infer), Rel::Refuted(lits));
        // Rows have no runtime representation: a cast ignores them.
        let fx = |row: EffectRow| Type::Fun(Hc::new(i.clone()), row, Hc::new(i.clone()));
        assert!(matches!(relate(&fx(EffectRow::single("a")), &fx(EffectRow::pure()), &infer), Rel::Refuted(_)));
        assert_eq!(relate_cast(&fx(EffectRow::single("a")), &fx(EffectRow::pure()), &infer), Rel::Proven);
    }

    // 2^50 leaves, 50 nodes: the pair memo makes each distinct pair cost one decision, so this
    // is far under the fuel cap and decides exactly (a tree walk would run out of fuel first).
    #[test]
    fn relate_decides_shared_nodes_exactly() {
        let dag = |leaf: Type| (0..50).fold(leaf, |t, _| tup(&[t.clone(), t]));
        let infer = InferCtx::new();
        assert_eq!(relate(&dag(Type::Int), &dag(Type::Dyn), &infer), Rel::Proven);
        assert_eq!(relate(&dag(Type::Dyn), &dag(Type::Int), &infer).kinds(), BTreeSet::from(["Test"]));
        assert!(matches!(relate(&dag(Type::Int), &dag(Type::Str), &infer), Rel::Refuted(_)));
        // Past the depth cap the answer is still Lazy, and a capped answer is not remembered.
        let deep = |leaf: Type| (0..70).fold(leaf, |t, _| tup(&[t.clone(), t]));
        assert!(matches!(relate(&deep(Type::Int), &deep(Type::Bool), &infer), Rel::Unknown(Residual::LAZY, Cause::Incomplete)));
    }

    #[test]
    fn relate_depth_guard_backstops_a_deep_source() {
        let infer = InferCtx::new();
        let deep = |leaf: Type| (0..70).fold(leaf, |t, _| list(t));
        assert!(matches!(relate(&deep(Type::Int), &deep(Type::Bool), &infer), Rel::Unknown(Residual::LAZY, Cause::Incomplete)));
    }

    #[test]
    fn relate_unfolds_a_rec_target() {
        let (i, b, d) = (Type::Int, Type::Bool, Type::Dyn);
        let l = l_ty();
        let m = Type::Rec(rec_group(vec![list(bound0())]), 0);
        let infer = ctx();
        assert_eq!(relate(&l, &l, &infer), Rel::Proven);
        assert!(matches!(relate(&m, &l, &infer), Rel::Refuted(_)), "different types are never interchangeable");
        assert!(matches!(relate(&l, &i, &infer), Rel::Refuted(_)));
        assert_eq!(relate(&tup(&[i.clone(), b.clone()]), &l, &infer), Rel::Proven, "one level of the literal fits");
        let two = tup(&[i.clone(), tup(&[i.clone(), i.clone()])]);
        assert!(matches!(relate(&two, &l, &infer), Rel::Refuted(_)));
        let got = relate(&d, &l, &infer);
        assert!(matches!(got, Rel::Unknown(..)) && got.cause() == Some(Cause::Imprecise), "{got:?}");
    }

    #[test]
    fn relate_terminates_on_degenerate_rec_types() {
        // let rec A = Union(A, A, Bool) (2^64 steps by depth alone), B = B, C = Union(Int, C), hand-built
        // (close_group would reject B): unfolding the same type against the same source again is a cycle,
        // so these are Refuted, not Lazy.
        let a = Type::Rec(rec_group(vec![uni(&[bound0(), bound0(), Type::Bool])]), 0);
        let b = Type::Rec(rec_group(vec![bound0()]), 0);
        let c = Type::Rec(rec_group(vec![uni(&[Type::Int, bound0()])]), 0);
        let infer = InferCtx::new();
        assert!(matches!(relate(&Type::Int, &a, &infer), Rel::Refuted(_)));
        assert!(matches!(relate(&Type::Int, &b, &infer), Rel::Refuted(_)));
        assert_eq!(relate(&Type::Bool, &a, &infer), Rel::Proven);
        assert_eq!(relate(&Type::Int, &c, &infer), Rel::Proven);
        assert!(matches!(relate(&Type::Str, &c, &infer), Rel::Refuted(_)));
    }

    #[test]
    fn relate_cast_decides_down_and_wrapper() {
        let infer = ctx();
        let (i, s, d, b) = (Type::Int, Type::Str, Type::Dyn, Type::Bool);
        let rec_ = |ns: &[&str]| rec(&ns.iter().map(|n| (*n, Type::Int)).collect::<Vec<_>>());
        let p = |f: &Type, t: &Type| relate_cast(f, t, &infer) == Rel::Proven;
        // DOWN of a source param `a` against target param `t` iff !p(t, a) (was needs_down(a, t)).
        assert!(p(&i, &d) && p(&i, &var("u")) && p(&i, &i));
        assert!(!p(&d, &i) && !p(&uni(&[i.clone(), b.clone()]), &i) && !p(&list(d.clone()), &list(i.clone())));
        assert!(!p(&rec_(&["x"]), &rec_(&["x", "y"])));
        assert!(!p(&vec_n(iv("m")), &vec_n(lit(3))) && p(&vec_n(lit(3)), &vec_n(lit(3))));
        assert!(p(&uni(&[i.clone(), s.clone()]), &uni(&[i.clone(), s.clone()]))); // cons: no redundant DOWN
        // A wrapper iff the Fun source is Unknown (was needs_wrapper).
        assert!(matches!(relate_cast(&fun(i.clone(), i.clone()), &d, &infer), Rel::Unknown(..)));
        assert!(p(&fun(d.clone(), d.clone()), &d) && p(&fun(d.clone(), l_ty()), &d));
        assert!(matches!(relate_cast(&fun(d.clone(), fun(i.clone(), i.clone())), &d, &infer), Rel::Unknown(..)));
        // Migrated from the fits unit tests.
        assert!(p(&fun(rec_(&["x"]), i.clone()), &fun(rec_(&["x", "y"]), i.clone())));
        assert!(matches!(relate(&fun(rec_(&["z"]), i.clone()), &fun(rec_(&["x", "y"]), i.clone()), &infer), Rel::Refuted(_)));
        assert!(p(&fun(i.clone(), i.clone()), &fun(i.clone(), d.clone())));
        assert!(matches!(relate(&fun(i.clone(), s.clone()), &fun(i.clone(), i.clone()), &infer), Rel::Refuted(_)));
    }

    #[test]
    fn identical_types_relate_beyond_the_caps() {
        let infer = ctx();
        let mut deep = Type::Int;
        for _ in 0..(MAX_DEPTH + 6) {
            deep = fun(Type::Int, deep);
        }
        let wide = uni(&(1..=120).map(|k| tup(&vec![Type::Int; k])).chain([fun(Type::Int, Type::Int)]).collect::<Vec<_>>());
        for t in [deep, wide] {
            assert_eq!(relate(&t, &t.clone(), &infer), Rel::Proven);
            assert_eq!(relate_cast(&t, &t.clone(), &infer), Rel::Proven);
        }
        // A cap on a non-identical pair is a runtime test, never Undecided.
        let (d1, d2) = (0..MAX_DEPTH + 6).fold((Type::Int, Type::Bool), |(a, b), _| (fun(Type::Int, a), fun(Type::Int, b)));
        let (u1, u2) = (uni(&[d1, Type::Bool]), uni(&[d2, Type::Bool, Type::Str]));
        let got = relate(&u1, &u2, &infer);
        assert!(!matches!(got, Rel::Unknown(Residual::UNDECIDED, _)), "{got:?}");
    }
}
