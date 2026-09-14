use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Dyn,
    Int,
    Bool,
    // param, effect row (what calling this may perform), return type.
    Fun(Rc<Type>, EffectRow, Rc<Type>),
}

// Closed effect row: no row polymorphism, no row variables -- just an
// exact known set (Closed), or "unknown, could be anything" (Dyn) when
// info was lost crossing a Dyn boundary (an untyped callee, an unrecognized
// handler expression). Dyn here plays the same role Type::Dyn does for
// values: the safe, permissive fallback that keeps gradual code running
// exactly as it did before this existed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectRow {
    Dyn,
    Closed(BTreeSet<String>),
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
    pub fn union(a: &EffectRow, b: &EffectRow) -> EffectRow {
        match (a, b) {
            (EffectRow::Dyn, _) | (_, EffectRow::Dyn) => EffectRow::Dyn,
            (EffectRow::Closed(x), EffectRow::Closed(y)) => {
                EffectRow::Closed(x.union(y).cloned().collect())
            }
        }
    }

    // Discharge one effect name (what a `handle` does to its body's row).
    // Unknown (Dyn) rows can't be subtracted from -- stay Dyn.
    pub fn remove(&self, effect: &str) -> EffectRow {
        match self {
            EffectRow::Dyn => EffectRow::Dyn,
            EffectRow::Closed(s) => {
                let mut s = s.clone();
                s.remove(effect);
                EffectRow::Closed(s)
            }
        }
    }
}

impl fmt::Display for EffectRow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EffectRow::Dyn => write!(f, "Dyn"),
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
pub fn consistent(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Dyn, _) | (_, Type::Dyn) => true,
        (Type::Int, Type::Int) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::Fun(a1, r1, b1), Type::Fun(a2, r2, b2)) => {
            consistent(a1, a2) && consistent(b1, b2) && row_consistent(r1, r2)
        }
        _ => false,
    }
}

fn row_consistent(a: &EffectRow, b: &EffectRow) -> bool {
    match (a, b) {
        (EffectRow::Dyn, _) | (_, EffectRow::Dyn) => true,
        (EffectRow::Closed(x), EffectRow::Closed(y)) => x == y,
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Dyn => write!(f, "Dyn"),
            Type::Int => write!(f, "Int"),
            Type::Bool => write!(f, "Bool"),
            // Dyn row prints as a plain arrow -- matches every existing
            // (unannotated-row) Fun type exactly as before this existed.
            Type::Fun(a, EffectRow::Dyn, b) => write!(f, "({a} -> {b})"),
            Type::Fun(a, row, b) => write!(f, "({a} ->{row} {b})"),
        }
    }
}
