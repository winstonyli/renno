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
// this plan's own Global Constraints. Checked after every merge, so a
// blow-up is caught as soon as it happens, not after fully expanding.
const MAX_MONOMIALS: usize = 64;

fn normalize(e: &IndexExpr) -> Option<Polynomial> {
    match e {
        IndexExpr::Lit(0) => Some(Polynomial::new()),
        IndexExpr::Lit(n) => Some(BTreeMap::from([(Vec::new(), *n)])),
        IndexExpr::Var(name) => Some(BTreeMap::from([(vec![name.clone()], 1)])),
        IndexExpr::Add(a, b) => {
            let mut result = normalize(a)?;
            merge_add(&mut result, &normalize(b)?);
            (result.len() <= MAX_MONOMIALS).then_some(result)
        }
        IndexExpr::Sub(a, b) => {
            let mut result = normalize(a)?;
            let mut rhs = normalize(b)?;
            for coeff in rhs.values_mut() {
                *coeff = -*coeff;
            }
            merge_add(&mut result, &rhs);
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
                    *result.entry(monomial).or_insert(0) += lc * rc;
                    if result.len() > MAX_MONOMIALS {
                        return None;
                    }
                }
            }
            Some(result)
        }
    }
}

fn merge_add(into: &mut Polynomial, other: &Polynomial) {
    for (monomial, coeff) in other.iter() {
        *into.entry(monomial.clone()).or_insert(0) += coeff;
    }
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
    let (Some(mut na), Some(mut nb)) = (normalize(a), normalize(b)) else {
        return false;
    };
    drop_zero_terms(&mut na);
    drop_zero_terms(&mut nb);
    na == nb
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
}
