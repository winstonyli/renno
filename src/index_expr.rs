use std::collections::BTreeMap;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexExpr {
    Var(String),
    Lit(i64),
    Add(Rc<IndexExpr>, Rc<IndexExpr>),
    Sub(Rc<IndexExpr>, Rc<IndexExpr>),
    Mul(Rc<IndexExpr>, Rc<IndexExpr>),
}

impl std::fmt::Display for IndexExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            IndexExpr::Var(name) => write!(f, "{name}"),
            IndexExpr::Lit(n) => write!(f, "{n}"),
            IndexExpr::Add(a, b) => write!(f, "{a} + {b}"),
            IndexExpr::Sub(a, b) => write!(f, "{a} - {b}"),
            IndexExpr::Mul(a, b) => write!(f, "{a} * {b}"),
        }
    }
}

// A monomial is a multiset of variable names (repetition = exponent),
// represented as a sorted Vec<String> for a canonical, comparable key.
// The polynomial is a map from that canonical monomial key to its
// integer coefficient -- BTreeMap so two structurally-different-but-
// equal expressions normalize to the SAME iteration order, making
// comparison a plain PartialEq on the map.
type Monomial = Vec<String>;
type Polynomial = BTreeMap<Monomial, i64>;

// Deliberately practical, not a claim of unbounded decidability -- see
// the design spec's own section 3. Checked after every merge, so a
// blow-up is caught as soon as it happens, not after fully expanding.
const MAX_MONOMIALS: usize = 64;

fn normalize(e: &IndexExpr) -> Option<Polynomial> {
    match e {
        IndexExpr::Lit(0) => Some(Polynomial::new()),
        IndexExpr::Lit(n) => Some(BTreeMap::from([(Vec::new(), *n)])),
        IndexExpr::Var(name) => Some(BTreeMap::from([(vec![name.clone()], 1)])),
        IndexExpr::Add(a, b) => {
            let mut result = normalize(a)?;
            merge_add(&mut result, &normalize(b)?)?;
            (result.len() <= MAX_MONOMIALS).then_some(result)
        }
        IndexExpr::Sub(a, b) => {
            let mut result = normalize(a)?;
            let mut rhs = normalize(b)?;
            for coeff in rhs.values_mut() {
                *coeff = coeff.checked_neg()?;
            }
            merge_add(&mut result, &rhs)?;
            (result.len() <= MAX_MONOMIALS).then_some(result)
        }
        IndexExpr::Mul(a, b) => {
            let lhs = normalize(a)?;
            let rhs = normalize(b)?;
            let mut result = Polynomial::new();
            for (lm, lc) in lhs.iter() {
                for (rm, rc) in rhs.iter() {
                    let mut monomial: Monomial = lm.iter().chain(rm.iter()).cloned().collect();
                    monomial.sort();
                    let product = lc.checked_mul(*rc)?;
                    let entry = result.entry(monomial).or_insert(0);
                    *entry = entry.checked_add(product)?;
                    if result.len() > MAX_MONOMIALS {
                        return None;
                    }
                }
            }
            Some(result)
        }
    }
}

// Returns `None` (never panics) on coefficient overflow, exactly like the
// existing monomial-count cap -- propagated up through the `?` operator at
// each call site, which index_exprs_equal already turns into `false`.
fn merge_add(into: &mut Polynomial, other: &Polynomial) -> Option<()> {
    for (monomial, coeff) in other.iter() {
        let entry = into.entry(monomial.clone()).or_insert(0);
        *entry = entry.checked_add(*coeff)?;
    }
    Some(())
}

// Zero coefficients are noise (e.g. `n - n` normalizes to a `{[n]: 0}`
// entry, not an empty map) -- strip them before comparing, so `n - n`
// and `0` compare equal.
fn drop_zero_terms(p: &mut Polynomial) {
    p.retain(|_, coeff| *coeff != 0);
}

/// Decides equality of two index expressions via canonical sum-of-
/// products normal form -- see the spec's own §3. Returns `false`
/// (never panics) both when the two expressions are genuinely
/// different AND when either side exceeds this module's own,
/// documented monomial cap -- an honest "cannot prove equal," not a
/// claim that the two are actually unequal.
pub fn index_exprs_equal(a: &IndexExpr, b: &IndexExpr) -> bool {
    index_exprs_compare(a, b) == Some(IndexCmp::Equal)
}

/// How two index expressions relate on their sum-of-products forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexCmp {
    Equal,
    /// `a - b` is a nonzero constant: unequal under every assignment.
    NonzeroConst,
    /// Unequal as polynomials, but some assignment might equate them.
    Other,
}

/// `None` = cannot decide (monomial cap or i64 overflow); never "unequal".
pub fn index_exprs_compare(a: &IndexExpr, b: &IndexExpr) -> Option<IndexCmp> {
    let (mut na, mut nb) = (normalize(a)?, normalize(b)?);
    drop_zero_terms(&mut na);
    drop_zero_terms(&mut nb);
    if na == nb {
        return Some(IndexCmp::Equal);
    }
    let mut diff = na;
    for (monomial, coeff) in nb {
        let entry = diff.entry(monomial).or_insert(0);
        *entry = entry.checked_sub(coeff)?;
    }
    drop_zero_terms(&mut diff);
    Some(if diff.len() == 1 && diff.contains_key(&Vec::<String>::new()) { IndexCmp::NonzeroConst } else { IndexCmp::Other })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> IndexExpr {
        IndexExpr::Var(name.to_string())
    }
    fn lit(n: i64) -> IndexExpr {
        IndexExpr::Lit(n)
    }
    fn add(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        IndexExpr::Add(Rc::new(a), Rc::new(b))
    }
    fn sub(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        IndexExpr::Sub(Rc::new(a), Rc::new(b))
    }
    fn mul(a: IndexExpr, b: IndexExpr) -> IndexExpr {
        IndexExpr::Mul(Rc::new(a), Rc::new(b))
    }

    #[test]
    fn identical_literals_are_equal() {
        assert!(index_exprs_equal(&lit(5), &lit(5)));
    }

    #[test]
    fn different_literals_are_not_equal() {
        assert!(!index_exprs_equal(&lit(5), &lit(6)));
    }

    #[test]
    fn same_variable_is_equal_to_itself() {
        assert!(index_exprs_equal(&var("n"), &var("n")));
    }

    #[test]
    fn different_variables_are_not_equal() {
        assert!(!index_exprs_equal(&var("n"), &var("m")));
    }

    #[test]
    fn multiplication_is_commutative() {
        // m*n == n*m -- the nonlinear equality SOP normalization
        // decides for free, per the spec's own §3.
        assert!(index_exprs_equal(&mul(var("m"), var("n")), &mul(var("n"), var("m"))));
    }

    #[test]
    fn addition_is_commutative_and_associative() {
        // (m+n)+k == k+(n+m)
        let lhs = add(add(var("m"), var("n")), var("k"));
        let rhs = add(var("k"), add(var("n"), var("m")));
        assert!(index_exprs_equal(&lhs, &rhs));
    }

    #[test]
    fn distributes_multiplication_over_addition() {
        // (m+n)*(m-n) == m*m - n*n
        let lhs = mul(add(var("m"), var("n")), sub(var("m"), var("n")));
        let rhs = sub(mul(var("m"), var("m")), mul(var("n"), var("n")));
        assert!(index_exprs_equal(&lhs, &rhs));
    }

    #[test]
    fn structurally_different_expressions_are_not_equal() {
        // n+1 != n+2
        assert!(!index_exprs_equal(&add(var("n"), lit(1)), &add(var("n"), lit(2))));
    }

    #[test]
    fn coefficient_overflow_returns_false_instead_of_panicking() {
        // A large-literal multiplication (i64::MAX * 3) overflows an i64
        // coefficient during normalize()'s Mul arm -- index_exprs_equal's
        // own doc comment promises "false, never panics" even here, not
        // just for the monomial-count cap.
        let big = mul(lit(i64::MAX), lit(3));
        assert!(!index_exprs_equal(&big, &big));
    }

    #[test]
    fn a_deliberately_oversized_expression_is_not_proven_equal() {
        // Repeated squaring blows past the 64-monomial cap fast:
        // (a+b)^64 (built via repeated self-multiplication, since this
        // grammar has no ^) expands to 65 monomials, exceeding the cap.
        // Exceeding the cap must return false (honest "can't prove"),
        // never panic.
        let mut big = add(var("a"), var("b"));
        for _ in 0..6 {
            big = mul(big.clone(), big.clone());
        }
        // Compare it against itself -- structurally IS equal, but the
        // cap must still be honestly enforced rather than silently
        // skipped for a self-comparison.
        assert!(!index_exprs_equal(&big, &big));
    }

    #[test]
    fn compare_distinguishes_equal_nonzero_constant_difference_other_and_cap() {
        assert_eq!(index_exprs_compare(&add(var("n"), lit(1)), &add(lit(1), var("n"))), Some(IndexCmp::Equal));
        assert_eq!(index_exprs_compare(&add(var("n"), lit(1)), &add(var("n"), lit(2))), Some(IndexCmp::NonzeroConst));
        assert_eq!(index_exprs_compare(&lit(3), &lit(4)), Some(IndexCmp::NonzeroConst));
        assert_eq!(index_exprs_compare(&var("n"), &var("m")), Some(IndexCmp::Other));
        let mut big = add(var("a"), var("b"));
        for _ in 0..6 {
            big = mul(big.clone(), big.clone()); }
        assert_eq!(index_exprs_compare(&big, &big), None);
        assert!(!index_exprs_equal(&big, &big)); }
}
