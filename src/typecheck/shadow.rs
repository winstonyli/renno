//! Stage-0 shadow of `relate`: records where the old predicates and `relate` disagree. Observation
//! only; deleted in S2. Process-wide `OnceLock<Option<Sink>>` (one atomic load per call when the
//! gate is off) and one `write_all` per line on an append-mode `File` behind a `Mutex`: the
//! parallel test threads serialize on the mutex, and the OS appends each write atomically, so
//! even several test processes sharing one log cannot interleave inside a line. No buffering, so
//! nothing is lost to `process::exit` or a panic.
use super::InferCtx;
use super::relate::{Rel, relate};
use crate::expr::ExprRef;
use crate::span::Span;
use crate::types::Type;
use crate::util::find_field;
use std::fs::{File, OpenOptions};
use std::fmt::Display;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
struct Sink {
    out: Mutex<File>,
    tag: Option<String>,
}
static SINK: OnceLock<Option<Sink>> = OnceLock::new();
fn sink() -> Option<&'static Sink> {
    SINK.get_or_init(|| {
        let out = OpenOptions::new().create(true).append(true).open(std::env::var_os("RENNO_SHADOW")?).ok()?;
        Some(Sink { out: Mutex::new(out), tag: std::env::var("RENNO_SHADOW_TAG").ok() })
    })
    .as_ref()
}
#[inline]
pub(super) fn on() -> bool {
    sink().is_some()
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Site {
    CoerceCheck,
    NeedsWrapper,
    NeedsDown,
    NeedsUpcast,
    Eq,
    Concat,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Old {
    Free,
    Check,
    Reject,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum New {
    Free,
    Check,
    Reject,
}
fn new_of(site: Site, rel: &Rel) -> New {
    match rel {
        Rel::Proven => New::Free,
        // coerce_check never wraps a function: coerce_cast does (the NeedsWrapper site). A residual
        // that is only a wrapper therefore leaves the coerce_check decision Free.
        Rel::Unknown(..) if site == Site::CoerceCheck && rel.kinds().iter().all(|k| *k == "Fun") => New::Free,
        Rel::Unknown(..) => New::Check,
        Rel::Refuted(_) => New::Reject,
    }
}
fn agree(site: Site, old: Old, new: New) -> bool {
    match site {
        Site::Eq | Site::Concat => (old == Old::Reject) == (new == New::Reject),
        _ => matches!((old, new), (Old::Free, New::Free) | (Old::Check, New::Check) | (Old::Reject, New::Reject)),
    }
}
fn mentions(t: &Type, p: &dyn Fn(&Type) -> bool) -> bool {
    p(t) || match t {
        Type::List(e) => mentions(e, p),
        Type::Fun(a, _, b) => mentions(a, p) || mentions(b, p),
        Type::Tuple(ts) | Type::Union(ts) => ts.iter().any(|x| mentions(x, p)),
        Type::Record(fs) => fs.iter().any(|(_, x)| mentions(x, p)),
        Type::Indexed(w, _) => mentions(w, p),
        _ => false,
    }
}
// Corresponding sub-pairs of two types (through List/Tuple/Record/Fun/Indexed): does `p` hold for any?
fn any_pair(a: &Type, b: &Type, p: &dyn Fn(&Type, &Type) -> bool) -> bool {
    p(a, b)
        || match (a, b) {
            (Type::List(x), Type::List(y)) => any_pair(x, y, p),
            (Type::Tuple(xs), Type::Tuple(ys)) => xs.iter().zip(ys.iter()).any(|(x, y)| any_pair(x, y, p)),
            (Type::Record(xs), Type::Record(ys)) => xs.iter().any(|(n, x)| find_field(ys, n).is_some_and(|y| any_pair(x, y, p))),
            (Type::Fun(a1, _, b1), Type::Fun(a2, _, b2)) => any_pair(a1, a2, p) || any_pair(b1, b2, p),
            (Type::Indexed(x, _), Type::Indexed(y, _)) => any_pair(x, y, p),
            (Type::Indexed(x, _), y) => any_pair(x, y, p),
            (x, Type::Indexed(y, _)) => any_pair(x, y, p),
            _ => false,
        }
}
// Every class below is a NARROW predicate (see "Expected classes"): kinds and shape, not mere presence.
// `from`/`to` are resolve_deep'd; `raw_from_var`: the caller's `from` was a bare Var. `needs_check` and
// `same_ignoring_rows` are the old, independent predicates: a Proven that they contradict is not filed.
fn classify(site: Site, old: Old, rel: &Rel, raw_from_var: bool, from: &Type, to: &Type) -> &'static str {
    let kinds = rel.kinds();
    let only = |k: &str| kinds.len() == 1 && kinds.contains(k);
    let either = |p: &dyn Fn(&Type) -> bool| mentions(from, p) || mentions(to, p);
    let indexed_both = mentions(from, &|t| matches!(t, Type::Indexed(..))) && mentions(to, &|t| matches!(t, Type::Indexed(..)));
    let index_kinds = kinds.iter().all(|k| matches!(*k, "IndexEq" | "Len"));
    let top_union = matches!(from, Type::Union(_)) || matches!(to, Type::Union(_));
    let dyn_either = either(&|t| matches!(t, Type::Dyn | Type::Var(_)));
    let inner_dyn = !matches!(from, Type::Dyn | Type::Var(_)) && mentions(from, &|t| matches!(t, Type::Dyn | Type::Var(_)));
    let width_gap = |x: &Type, y: &Type| matches!((x, y), (Type::Record(a), Type::Record(b)) if a.len() != b.len());
    let var_bridge = |x: &Type, y: &Type| matches!((x, y), (Type::List(e), Type::Tuple(_)) | (Type::Tuple(_), Type::List(e)) if matches!(**e, Type::Var(_) | Type::Dyn));
    let ok = !matches!(rel, Rel::Refuted(_));
    // A bare Dyn/Var source into a Named (or Indexed-of-Named) target: build_boundary_check has no check for
    // Indexed(Named) (`_ => e`) or for a Named missing from the registry, so the old gate adds nothing.
    let test_len = kinds.iter().all(|k| matches!(*k, "Test" | "Len"));
    let dyn_into_named = matches!(from, Type::Dyn | Type::Var(_)) && match to {
        Type::Named(_) => true,
        Type::Indexed(w, _) => matches!(**w, Type::Named(_)),
        _ => false,
    };
    match (site, old, rel) {
        (Site::CoerceCheck, Old::Free, Rel::Unknown(..)) if dyn_into_named && test_len => "hn:dyn-into-named-unchecked",
        (Site::CoerceCheck, Old::Free, Rel::Unknown(..)) if only("Test") && matches!(from, Type::Union(_)) => "u:union-narrow",
        (Site::CoerceCheck, Old::Free, Rel::Unknown(..)) if only("IndexEq") => "flex:index-bound-later",
        (Site::CoerceCheck, Old::Free, Rel::Unknown(..)) if only("Test") && inner_dyn => "h:container-dyn-inside",
        (Site::CoerceCheck, Old::Free, Rel::Refuted(_)) if matches!(to, Type::Named(_)) && !matches!(from, Type::Named(_)) => "t:recursive-literal",
        (Site::CoerceCheck | Site::Eq | Site::Concat, Old::Free, Rel::Refuted(w)) if w.starts_with("index") => "rigid:rejected-later-by-unify",
        (Site::CoerceCheck, Old::Reject, _) if ok && indexed_both && index_kinds => "m:ctxfree-index-gate",
        (Site::CoerceCheck | Site::Eq | Site::Concat, Old::Reject, _) if ok && top_union => "u:union-direction",
        (Site::Eq | Site::Concat, Old::Reject, _) if ok && any_pair(from, to, &width_gap) => "eq:record-width",
        (Site::CoerceCheck | Site::Eq | Site::Concat, Old::Reject, _) if ok && any_pair(from, to, &var_bridge) => "br:var-bridge",
        (Site::CoerceCheck, Old::Check, Rel::Proven) if raw_from_var && !super::needs_check(from, to) => "late:resolved-var-redundant-check",
        (Site::NeedsWrapper | Site::NeedsUpcast, Old::Free, Rel::Unknown(..)) if only("Fun") && dyn_either => "hf:fun-result-dyn",
        (Site::NeedsDown, Old::Check, Rel::Proven) if !super::needs_check(from, to) => "cons:over-conservative",
        (Site::NeedsWrapper | Site::NeedsUpcast, Old::Check, Rel::Proven) if super::same_ignoring_rows(from, to) => "cons:over-conservative",
        _ => "unclassified",
    }
}
fn write_line(s: &Sink, line: &str) {
    if let Ok(mut f) = s.out.lock() {
        let _ = f.write_all(line.as_bytes());
    }
}
// Agreements are logged as short `A` lines (counted, never inspected); disagreements in full:
// D, site, old, new, cause|-, kinds joined by +|-, class, tag, pos (byte offset|-), from, to.
struct Row<'a> {
    site: Site,
    old: Old,
    new: New,
    rel: &'a Rel,
    class: &'a str,
    pos: Option<usize>,
}
fn d_line(s: &Sink, r: Row, from: &dyn Display, to: &dyn Display) {
    let cause = r.rel.cause().map_or("-".to_string(), |c| format!("{c:?}"));
    let kinds = r.rel.kinds().into_iter().collect::<Vec<_>>().join("+");
    let kinds = if kinds.is_empty() { "-" } else { &kinds };
    let pos = r.pos.map_or("-".to_string(), |p| p.to_string());
    let current = std::thread::current();
    let tag = s.tag.as_deref().or(current.name()).unwrap_or("?");
    let (site, old, new, class) = (r.site, r.old, r.new, r.class);
    write_line(s, &format!("D\t{site:?}\t{old:?}\t{new:?}\t{cause}\t{kinds}\t{class}\t{tag}\t{pos}\t{from}\t{to}\n"));
}
// `pos`: byte offset of the checked expression where the site knows it (coerce_check only);
// `raw_from_var`: the caller's `from` was a bare Var.
#[derive(Default, Clone, Copy)]
struct Extra {
    raw_from_var: bool,
    pos: Option<usize>,
}
fn emit(site: Site, old: Old, rel: &Rel, from: &Type, to: &Type, extra: Extra, infer: &InferCtx) {
    let Some(s) = sink() else { return };
    let new = new_of(site, rel);
    if agree(site, old, new) {
        write_line(s, &format!("A\t{site:?}\t{old:?}\t{new:?}\n"));
        return;
    }
    let (f, t) = (infer.resolve_deep(from), infer.resolve_deep(to));
    let class = classify(site, old, rel, extra.raw_from_var, &f, &t);
    d_line(s, Row { site, old, new, rel, class, pos: extra.pos }, &f, &t);
}
fn old_of(wrapped: bool) -> Old {
    if wrapped { Old::Check } else { Old::Free }
}
pub(super) fn coerce_check(infer: &InferCtx, from: &Type, to: &Type, e: ExprRef, out: Option<ExprRef>, span: Span) {
    let old = match out {
        None => Old::Reject,
        Some(r) if r == e => Old::Free,
        Some(_) => Old::Check,
    };
    emit(Site::CoerceCheck, old, &relate(from, to, infer), from, to, Extra { raw_from_var: matches!(from, Type::Var(_)), pos: Some(span.start) }, infer);
}
pub(super) fn needs_wrapper(infer: &InferCtx, from: &Type, to: &Type) -> bool {
    let old = super::needs_wrapper(from, to);
    if on() && matches!(infer.resolve(from), Type::Fun(..)) {
        emit(Site::NeedsWrapper, old_of(old), &relate(from, to, infer), from, to, Extra::default(), infer);
    }
    old
}
// Compared as relate(a_target -> a_src): strict_fits(a_target, a_src) is fits(required = a_src, actual = a_target).
pub(super) fn needs_down(infer: &InferCtx, a_src: &Type, a_target: &Type) -> bool {
    let old = super::needs_down(a_src, a_target);
    if on() {
        emit(Site::NeedsDown, old_of(old), &relate(a_target, a_src, infer), a_target, a_src, Extra::default(), infer);
    }
    old
}
pub(super) fn needs_upcast(infer: &InferCtx, b: &Type, b_target: &Type) -> bool {
    let old = super::needs_upcast(b, b_target);
    if on() && matches!(infer.resolve(b), Type::Fun(..)) {
        emit(Site::NeedsUpcast, old_of(old), &relate(b, b_target, infer), b, b_target, Extra::default(), infer);
    }
    old
}
// Symmetric `consistent` sites (== and ++): New is the better direction.
pub(super) fn consistent(infer: &InferCtx, site: Site, a: &Type, b: &Type) -> bool {
    let old = crate::types::consistent(a, b);
    if on() {
        let (ab, ba) = (relate(a, b, infer), relate(b, a, infer));
        let rank = |r: &Rel| match r {
            Rel::Proven => 0,
            Rel::Unknown(..) => 1,
            Rel::Refuted(_) => 2,
        };
        let best = if rank(&ab) <= rank(&ba) { ab } else { ba };
        emit(site, if old { Old::Free } else { Old::Reject }, &best, a, b, Extra::default(), infer);
    }
    old
}
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::relate::{Cause, Residual};
    use crate::index_expr::IndexExpr;
    use std::rc::Rc;
    #[test]
    fn agree_and_classify_follow_the_verdict_mapping() {
        assert!(agree(Site::CoerceCheck, Old::Free, New::Free) && !agree(Site::CoerceCheck, Old::Free, New::Check));
        assert!(agree(Site::Eq, Old::Free, New::Check) && !agree(Site::Eq, Old::Reject, New::Check));
        let (i, d) = (Type::Int, Type::Dyn);
        let un = Type::Union(Rc::new(vec![Type::Int, Type::Str]));
        let vec1 = Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(IndexExpr::Lit(1)));
        let tup1 = Type::Tuple(Rc::new(vec![Type::Int]));
        let named = Type::Named("L#1".into());
        let rec = |n: usize| Type::Record(Rc::new(["x", "y"].iter().take(n).map(|f| (f.to_string(), Type::Int)).collect()));
        let unk = |r| Rel::Unknown(r, Cause::Imprecise);
        // A wrapper-only residual is not a coerce_check decision; a Dyn used as a function is.
        assert_eq!(new_of(Site::CoerceCheck, &unk(Residual::Fun)), New::Free);
        assert_eq!(new_of(Site::NeedsWrapper, &unk(Residual::Fun)), New::Check);
        assert_eq!(new_of(Site::CoerceCheck, &unk(Residual::All(vec![Residual::Test, Residual::Fun]))), New::Check);
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &unk(Residual::Test), false, &un, &i), "u:union-narrow");
        assert_eq!(classify(Site::CoerceCheck, Old::Reject, &Rel::Proven, false, &vec1, &vec1), "m:ctxfree-index-gate");
        assert_eq!(classify(Site::NeedsWrapper, Old::Free, &unk(Residual::Fun), false, &i, &d), "hf:fun-result-dyn");
        assert_eq!(classify(Site::CoerceCheck, Old::Check, &Rel::Proven, true, &i, &i), "late:resolved-var-redundant-check");
        assert_eq!(classify(Site::NeedsDown, Old::Check, &Rel::Proven, false, &i, &i), "cons:over-conservative");
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &Rel::refuted("shape"), false, &tup1, &named), "t:recursive-literal");
        assert_eq!(classify(Site::CoerceCheck, Old::Reject, &unk(Residual::Test), false, &Type::List(Rc::new(Type::Var("a".into()))), &tup1), "br:var-bridge");
        assert_eq!(classify(Site::Eq, Old::Reject, &Rel::Proven, false, &rec(2), &rec(1)), "eq:record-width");
        assert_eq!(classify(Site::Eq, Old::Reject, &Rel::Proven, false, &un, &i), "u:union-direction");
        // Near misses must NOT be absorbed: a class is a narrow, falsifiable predicate.
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &unk(Residual::Len), false, &un, &i), "unclassified"); // Union present, wrong kind
        assert_eq!(classify(Site::NeedsWrapper, Old::Free, &unk(Residual::Fun), false, &i, &i), "unclassified"); // no Dyn anywhere
        assert_eq!(classify(Site::CoerceCheck, Old::Check, &Rel::Proven, true, &d, &i), "unclassified"); // old check was right: needs_check(Dyn, Int)
        assert_eq!(classify(Site::NeedsDown, Old::Check, &Rel::Proven, false, &d, &i), "unclassified"); // same
        assert_eq!(classify(Site::Eq, Old::Reject, &Rel::Proven, false, &rec(2), &rec(2)), "unclassified"); // Record present, no width gap
        assert_eq!(classify(Site::CoerceCheck, Old::Reject, &Rel::Proven, false, &vec1, &i), "unclassified"); // only one side Indexed
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &Rel::refuted("shape"), false, &tup1, &Type::List(Rc::new(named.clone()))), "unclassified"); // Named only nested
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &Rel::Proven, false, &i, &i), "unclassified");
        // Classes that had no positive row, and one near miss each (added in round 3).
        let lst = |t: Type| Type::List(Rc::new(t));
        let ft = |p: Type, r: Type| Type::Fun(Rc::new(p), crate::types::EffectRow::Dyn, Rc::new(r));
        let vecn = Type::Indexed(Rc::new(named.clone()), Rc::new(IndexExpr::Lit(1)));
        let len_test = || unk(Residual::All(vec![Residual::Len, Residual::Test]));
        let idx = |c| Rel::Unknown(Residual::IndexEq, c);
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &unk(Residual::Test), false, &lst(d.clone()), &lst(i.clone())), "h:container-dyn-inside");
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &idx(Cause::Incomplete), false, &vec1, &vec1), "flex:index-bound-later");
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &Rel::refuted("index: rigid or literal mismatch"), false, &vec1, &vec1), "rigid:rejected-later-by-unify");
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &len_test(), false, &d, &vecn), "hn:dyn-into-named-unchecked");
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &unk(Residual::Test), false, &Type::Var("a".into()), &named), "hn:dyn-into-named-unchecked");
        assert_eq!(classify(Site::NeedsWrapper, Old::Check, &Rel::Proven, false, &ft(i.clone(), i.clone()), &ft(i.clone(), i.clone())), "cons:over-conservative");
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &unk(Residual::Test), false, &d, &i), "unclassified"); // Dyn into a plain type: the old gate checks it, so Free is a different bug
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &len_test(), false, &lst(d.clone()), &vecn), "unclassified"); // Dyn only nested: not the bare-Dyn hn: shape
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &unk(Residual::All(vec![Residual::IndexEq, Residual::Test])), false, &i, &i), "unclassified"); // flex: IndexEq alone
        assert_eq!(classify(Site::NeedsDown, Old::Free, &idx(Cause::Imprecise), false, &vec1, &vec1), "unclassified"); // flex: only at CoerceCheck
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &Rel::refuted("shape"), false, &i, &i), "unclassified"); // rigid: only index refutations
        assert_eq!(classify(Site::CoerceCheck, Old::Free, &unk(Residual::Test), false, &lst(un.clone()), &lst(i.clone())), "unclassified"); // Union only nested
        assert_eq!(classify(Site::Eq, Old::Reject, &Rel::Proven, false, &lst(un.clone()), &lst(i.clone())), "unclassified"); // Union only nested
        assert_eq!(classify(Site::CoerceCheck, Old::Reject, &Rel::Proven, false, &lst(i.clone()), &tup1), "unclassified"); // an Int element is no Var bridge
        assert_eq!(classify(Site::CoerceCheck, Old::Check, &Rel::Proven, false, &i, &i), "unclassified"); // late: the source was not a bare Var
        assert_eq!(classify(Site::NeedsWrapper, Old::Check, &Rel::Proven, false, &ft(i.clone(), d.clone()), &ft(i.clone(), i.clone())), "unclassified"); // different functions: the wrapper was needed
        assert_eq!(classify(Site::NeedsWrapper, Old::Free, &unk(Residual::Test), false, &i, &d), "unclassified"); // hf: Fun kind only
    }
}
