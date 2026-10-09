//! Directional, infer-aware type relation (design 2026-10-08, section 3.1). Pure and read-only.
//! Stage 1 adds `Why`/`Undecided`/`relate_cast`; stage 2 consumers: coerce_check and the unify_fits rescue (T3), coerce_cast (T4).
use super::{free_index_vars, unify_index_expr, InferCtx};
use crate::index_expr::{index_exprs_compare, IndexCmp, IndexExpr};
use crate::types::{row_consistent, Type};
use crate::util::find_field;
use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap};
/// Native-stack guard (a source type that is itself very deep). Degenerate aliases are cut earlier
/// by the revisit guard in the `Named` arm.
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
/// The runtime work that would decide an Unknown. Unit placeholders: S2 adds payloads. `Fun` is
/// a wrapper around a function value (no value test); a Dyn used as a function is `Test` + `Fun`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Residual {
    Test,
    Fun,
    Len,
    IndexEq,
    Lazy,
    /// No value test, wrapper or unifier step decides it: a static error.
    Undecided,
    All(Vec<Residual>),
    Any(Vec<Residual>),
}
impl Residual {
    fn collect(&self, out: &mut BTreeSet<&'static str>) {
        match self {
            Residual::Test => {
                out.insert("Test");
            }
            Residual::Fun => {
                out.insert("Fun");
            }
            Residual::Len => {
                out.insert("Len");
            }
            Residual::IndexEq => {
                out.insert("IndexEq");
            }
            Residual::Lazy => {
                out.insert("Lazy");
            }
            Residual::Undecided => {
                out.insert("Undecided");
            }
            Residual::All(rs) | Residual::Any(rs) => rs.iter().for_each(|r| r.collect(out)),
        }
    }
}
impl Residual {
    /// Some part is decided by a value test (lowered by `build_boundary_check`).
    pub(crate) fn needs_test(&self) -> bool {
        match self {
            Residual::Test | Residual::Len | Residual::Lazy => true,
            Residual::All(rs) | Residual::Any(rs) => rs.iter().any(Residual::needs_test),
            _ => false,
        }
    }
    pub(crate) fn undecided(&self) -> bool {
        match self {
            Residual::Undecided => true,
            Residual::All(rs) | Residual::Any(rs) => rs.iter().any(Residual::undecided),
            _ => false,
        }
    }
}
/// Why a pair is Refuted. Only a `Shape` refutation is seen by the value test that lowers a
/// `Residual::Test`, which matters when it is one alternative of a Union source.
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
    pub(crate) fn refuted(why: &'static str) -> Rel { Rel::Refuted(Why::Shape(why)) }
    fn opaque(why: &'static str) -> Rel { Rel::Refuted(Why::Opaque(why)) }
    /// The distinct residual kinds an Unknown mentions (empty for Proven/Refuted).
    pub(crate) fn kinds(&self) -> BTreeSet<&'static str> {
        let mut out = BTreeSet::new();
        if let Rel::Unknown(r, _) = self {
            r.collect(&mut out);
        }
        out
    }
    pub(crate) fn cause(&self) -> Option<Cause> {
        if let Rel::Unknown(_, c) = self { Some(*c) } else { None }
    }
}
// Conjunction: Refuted dominates, then Unknown (worst cause), else Proven.
fn and(rels: impl IntoIterator<Item = Rel>) -> Rel {
    let (mut unknowns, mut cause) = (Vec::new(), Cause::Imprecise);
    for r in rels {
        match r {
            Rel::Refuted(why) => return Rel::Refuted(why),
            Rel::Unknown(res, c) => {
                cause = cause.max(c);
                unknowns.push(res);
            }
            Rel::Proven => {}
        }
    }
    match unknowns.len() {
        0 => Rel::Proven,
        1 => Rel::Unknown(unknowns.remove(0), cause),
        _ => Rel::Unknown(Residual::All(unknowns), cause),
    }
}
// Disjunction over target-Union alternatives (the caller already returned on a Proven one):
// all Refuted refutes; otherwise Unknown, as precise as its most precise Unknown alternative.
fn or(rels: Vec<Rel>) -> Rel {
    let (mut unknowns, mut cause, mut why) = (Vec::new(), Cause::Incomplete, Why::Shape("empty union"));
    for r in rels {
        match r {
            Rel::Unknown(res, c) => {
                cause = cause.min(c);
                unknowns.push(res);
            }
            Rel::Refuted(w) => if why.observable() { why = w },
            Rel::Proven => return Rel::Proven,
        }
    }
    if unknowns.is_empty() { Rel::Refuted(why) } else { Rel::Unknown(Residual::Any(unknowns), cause) }
}
/// Per-`relate` bookkeeping: remaining `go` calls, and the (source, alias) pairs being unfolded.
struct Budget {
    left: Cell<u32>,
    unfolding: RefCell<Vec<(Type, String)>>,
    ignore_rows: bool,
}
impl Default for Budget {
    fn default() -> Self {
        Budget { left: Cell::new(FUEL), unfolding: RefCell::new(Vec::new()), ignore_rows: false }
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
    if depth > MAX_DEPTH || fuel.left.get() == 0 {
        return Rel::Unknown(Residual::Lazy, Cause::Incomplete);
    }
    fuel.left.set(fuel.left.get() - 1);
    let (from, to, d) = (infer.resolve(from), infer.resolve(to), depth + 1);
    let unknown = |r: Residual| Rel::Unknown(r, Cause::Imprecise);
    match (&from, &to) {
        (Type::Var(a), Type::Var(b)) if a == b => Rel::Proven,
        (Type::Fun(..), Type::Dyn) => fun_rel(&from, &to, infer, fuel, d), // cast as Fun(Dyn, Dyn)
        (_, t) if loose(t) => Rel::Proven,
        (f, Type::Fun(..)) if loose(f) => unknown(Residual::All(vec![Residual::Test, Residual::Fun])),
        (f, Type::Indexed(..)) if loose(f) => unknown(Residual::All(vec![Residual::Test, Residual::Len])),
        (f, _) if loose(f) => unknown(Residual::Test),
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
                Rel::Unknown(..) => matches!(infer.resolve(a), Type::Fun(..)),
                Rel::Proven => false,
            }) {
                Rel::Unknown(Residual::Undecided, Cause::Incomplete)
            } else {
                unknown(Residual::Test)
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
        (Type::Named(a), Type::Named(b)) => {
            if a == b { Rel::Proven } else { Rel::opaque("named: different ids") }
        }
        (_, Type::Named(id)) => match infer.named_types.get(id) {
            // Unfolding the same alias against the same source again is a cycle with no new evidence
            // (`type A = Int | A`): no finite derivation, so Refuted rather than Lazy.
            Some(_) if fuel.unfolding.borrow().iter().any(|(f, i)| i == id && *f == from) => Rel::refuted("named: cyclic unfolding"),
            Some(raw) => {
                fuel.unfolding.borrow_mut().push((from.clone(), id.clone()));
                let r = go(&from, raw, infer, fuel, d);
                fuel.unfolding.borrow_mut().pop();
                r
            }
            None => Rel::opaque("named: not in registry"),
        },
        (Type::Named(_), _) => Rel::opaque("named: nominal source"),
        (Type::Int, Type::Int) | (Type::Float, Type::Float) | (Type::Bool, Type::Bool) | (Type::Str, Type::Str) => Rel::Proven,
        (Type::Token(a), Type::Token(b)) => {
            if a == b { Rel::Proven } else { Rel::refuted("token") }
        }
        (Type::List(a), Type::List(b)) => go(a, b, infer, fuel, d),
        (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => and(a.iter().zip(b.iter()).map(|(x, y)| go(x, y, infer, fuel, d))),
        (Type::Tuple(_), Type::Tuple(_)) => Rel::refuted("tuple arity"),
        // The tuple/list bridge exists only for a Dyn/Var element (types::consistent's has Dyn only).
        (Type::List(e), Type::Tuple(_)) => {
            if loose(&infer.resolve(e)) { unknown(Residual::Test) } else { Rel::refuted("list into tuple") }
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
// Params contravariant, returns covariant; any Unknown component is a wrapper (Residual::Fun).
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
        Rel::Unknown(_, c) => Rel::Unknown(Residual::Fun, c),
        Rel::Refuted(Why::Shape(w)) => Rel::Refuted(Why::Opaque(w)),
        other => other,
    }
}
/// Read-only index half of `unify_index_expr`. Equal -> Proven; cap/overflow -> Unknown(IndexEq,
/// Incomplete); differ by a nonzero constant -> Refuted; a bare flexible variable (bound by
/// unification) -> Unknown(IndexEq, Imprecise); a compound with a flexible variable (needs
/// solving) -> Unknown(IndexEq, Incomplete); only rigid variables/literals left -> Refuted.
pub(crate) fn relate_index(a: &IndexExpr, b: &IndexExpr, infer: &InferCtx) -> Rel {
    let (ra, rb) = (infer.resolve_index_deep(a), infer.resolve_index_deep(b));
    match index_exprs_compare(&ra, &rb) {
        None => return Rel::Unknown(Residual::IndexEq, Cause::Incomplete),
        Some(IndexCmp::Equal) => return Rel::Proven,
        Some(IndexCmp::NonzeroConst) => return index_refuted(a, b, infer),
        Some(IndexCmp::Other) => {}
    }
    let flexible = |v: &String| !infer.rigid_index.contains(v);
    let bare_flexible = |e: &IndexExpr| matches!(e, IndexExpr::Var(v) if flexible(v));
    let mut vars = free_index_vars(&ra);
    vars.extend(free_index_vars(&rb));
    if bare_flexible(&ra) || bare_flexible(&rb) {
        Rel::Unknown(Residual::IndexEq, Cause::Imprecise)
    } else if vars.iter().any(flexible) {
        Rel::Unknown(Residual::IndexEq, Cause::Incomplete)
    } else {
        index_refuted(a, b, infer)
    }
}
/// The unifier's own text for a refuted index pair (run on a scratch copy; nothing is bound).
fn index_refuted(a: &IndexExpr, b: &IndexExpr, infer: &InferCtx) -> Rel {
    let mut scratch = InferCtx::new(HashMap::new());
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
    use crate::types::EffectRow;
    use std::collections::HashMap;
    use std::rc::Rc;
    fn list(t: Type) -> Type {
        Type::List(Rc::new(t))
    }
    fn tup(ts: &[Type]) -> Type {
        Type::Tuple(Rc::new(ts.to_vec()))
    }
    fn uni(ts: &[Type]) -> Type {
        Type::Union(Rc::new(ts.to_vec()))
    }
    fn rec(fs: &[(&str, Type)]) -> Type {
        Type::Record(Rc::new(fs.iter().map(|(n, t)| (n.to_string(), t.clone())).collect()))
    }
    fn fun(a: Type, b: Type) -> Type {
        Type::Fun(Rc::new(a), EffectRow::Dyn, Rc::new(b))
    }
    fn var(n: &str) -> Type {
        Type::Var(n.to_string())
    }
    fn named(id: &str) -> Type {
        Type::Named(id.to_string())
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
        Type::Indexed(Rc::new(list(Type::Dyn)), Rc::new(i))
    }
    // a := Int, s := Str, u unbound; index n rigid, z := 0; L = (Int, L) | Bool.
    fn ctx() -> InferCtx {
        let raw = uni(&[tup(&[Type::Int, named("L#1")]), Type::Bool]);
        let mut i = InferCtx::new(HashMap::from([("L#1".to_string(), raw)]));
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
        let fx = |p: Type, r: Type, row: EffectRow| Type::Fun(Rc::new(p), row, Rc::new(r));
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
            ("named same", named("L#1"), named("L#1"), P),
            ("named different", named("M#2"), named("L#1"), R),
            ("named into int", named("L#1"), i.clone(), R),
            ("literal one level into L", tup(&[i.clone(), b.clone()]), named("L#1"), P),
            ("literal two levels into L (t10)", tup(&[i.clone(), ii.clone()]), named("L#1"), R),
            ("dyn into L", d.clone(), named("L#1"), U(&["Test"], Imp)),
            ("vec same", vec_n(lit(3)), vec_n(lit(3)), P),
            ("vec literal mismatch", vec_n(lit(3)), vec_n(lit(4)), R),
            ("vec into flexible var", vec_n(lit(3)), vec_n(iv("m")), U(&["IndexEq"], Imp)),
            ("rigid vs literal", vec_n(iv("n")), vec_n(lit(3)), R),
            ("rigid vs flexible compound (x2)", vec_n(iv("n")), vec_n(add(iv("k"), lit(1))), U(&["IndexEq"], Inc)),
            ("rigid vs n+1", vec_n(iv("n")), vec_n(add(iv("n"), lit(1))), R),
            ("sop equal", vec_n(add(iv("n"), lit(1))), vec_n(add(lit(1), iv("n"))), P),
            ("sop cap (x1)", vec_n(big.clone()), vec_n(big), U(&["IndexEq"], Inc)),
            ("index_subst applied (m6)", vec_n(lit(1)), vec_n(add(iv("z"), lit(1))), P),
            ("vec into plain list", vec_n(lit(3)), list(d.clone()), P),
            ("plain list into vec", list(d.clone()), vec_n(lit(3)), R),
            ("dyn into vec", d.clone(), vec_n(lit(3)), U(&["Len", "Test"], Imp)),
            ("fun into unbound var", fun(i.clone(), i.clone()), var("u"), P),
            ("union mix, opaque alt (fun signature)", uni(&[fun(i.clone(), i.clone()), b.clone()]), uni(&[fun(s.clone(), i.clone()), b.clone()]), U(&["Undecided"], Inc)),
            ("union mix, index alt", uni(&[vec_n(lit(3)), b.clone()]), uni(&[vec_n(lit(4)), b.clone()]), U(&["Undecided"], Inc)),
            ("union mix, fun alt needs a wrapper", uni(&[fun(i.clone(), i.clone()), b.clone()]), uni(&[fun(d.clone(), d.clone()), b.clone()]), U(&["Undecided"], Inc)),
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
        assert!(matches!(relate(&named("M#2"), &named("L#1"), &infer), Rel::Refuted(Why::Opaque(_))));
        assert!(matches!(relate(&fun(i.clone(), i.clone()), &fun(s.clone(), i.clone()), &infer), Rel::Refuted(Why::Opaque(_))));
        let rigid = Why::Index { msg: "index variable n is fixed by the signature and cannot equal 3".into(), bare: true };
        assert_eq!(relate(&vec_n(iv("n")), &vec_n(lit(3)), &infer), Rel::Refuted(rigid));
        let lits = Why::Index { msg: "type mismatch: index 4 does not unify with index 3".into(), bare: false };
        assert_eq!(relate(&vec_n(lit(3)), &vec_n(lit(4)), &infer), Rel::Refuted(lits));
        // Rows have no runtime representation: a cast ignores them.
        let fx = |row: EffectRow| Type::Fun(Rc::new(i.clone()), row, Rc::new(i.clone()));
        assert!(matches!(relate(&fx(EffectRow::single("a")), &fx(EffectRow::pure()), &infer), Rel::Refuted(_)));
        assert_eq!(relate_cast(&fx(EffectRow::single("a")), &fx(EffectRow::pure()), &infer), Rel::Proven);
    }

    #[test]
    fn relate_terminates_on_degenerate_aliases() {
        // type A = A | A | Bool (2^64 steps by depth alone), type B = B, type C = Int | C: unfolding the
        // same alias against the same source again is a cycle, so these are Refuted, not Lazy.
        let reg = HashMap::from([
            ("A#1".to_string(), uni(&[named("A#1"), named("A#1"), Type::Bool])),
            ("B#2".to_string(), named("B#2")),
            ("C#3".to_string(), uni(&[Type::Int, named("C#3")])),
        ]);
        let infer = InferCtx::new(reg);
        assert!(matches!(relate(&Type::Int, &named("A#1"), &infer), Rel::Refuted(_)));
        assert!(matches!(relate(&Type::Int, &named("B#2"), &infer), Rel::Refuted(_)));
        assert_eq!(relate(&Type::Bool, &named("A#1"), &infer), Rel::Proven);
        assert_eq!(relate(&Type::Int, &named("C#3"), &infer), Rel::Proven);
        assert!(matches!(relate(&Type::Str, &named("C#3"), &infer), Rel::Refuted(_)));
        // The depth guard is the backstop for a source that is itself too deep.
        let deep = (0..70).fold(Type::Int, |t, _| list(t));
        assert!(matches!(relate(&deep, &deep, &infer), Rel::Unknown(Residual::Lazy, Cause::Incomplete)));
    }
}
