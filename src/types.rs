use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Dyn,
    Int,
    Bool,
    Str,
    // Element type. renno's List is a native primitive (Value::List,
    // Rc<Vec<Value>>) rather than something expressed via user-defined
    // algebraic types, since renno has neither ADTs, pattern matching, nor
    // general recursion yet -- all three would be prerequisites for a
    // "std lib" `data List a = Nil | Cons a (List a)` definition. Worth
    // revisiting once those exist: this could become sugar over a
    // user-space definition instead of a builtin, the way it works in
    // languages with real sum types.
    List(Rc<Type>),
    // param, effect row (what calling this may perform), return type.
    Fun(Rc<Type>, EffectRow, Rc<Type>),
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

impl fmt::Display for EffectRow {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EffectRow::Dyn => write!(f, "Dyn"),
            EffectRow::Var(name) => write!(f, "{{{name}}}"),
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
        (Type::Str, Type::Str) => true,
        (Type::List(a), Type::List(b)) => consistent(a, b),
        (Type::Fun(a1, r1, b1), Type::Fun(a2, r2, b2)) => {
            consistent(a1, a2) && consistent(b1, b2) && row_consistent(r1, r2)
        }
        _ => false,
    }
}

fn row_consistent(a: &EffectRow, b: &EffectRow) -> bool {
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
            Type::Bool => write!(f, "Bool"),
            Type::Str => write!(f, "Str"),
            Type::List(elem) => write!(f, "[{elem}]"),
            // Dyn row prints as a plain arrow -- matches every existing
            // (unannotated-row) Fun type exactly as before this existed.
            Type::Fun(a, EffectRow::Dyn, b) => write!(f, "({a} -> {b})"),
            Type::Fun(a, row, b) => write!(f, "({a} ->{row} {b})"),
        }
    }
}
