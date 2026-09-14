use std::fmt;
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Dyn,
    Int,
    Bool,
    Fun(Rc<Type>, Rc<Type>),
}

// Consistency, not equality: Dyn is consistent with everything. This is
// what makes it "gradual" -- an annotated boundary only rejects code where
// BOTH sides are concretely known and disagree.
pub fn consistent(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Dyn, _) | (_, Type::Dyn) => true,
        (Type::Int, Type::Int) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::Fun(a1, b1), Type::Fun(a2, b2)) => consistent(a1, a2) && consistent(b1, b2),
        _ => false,
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::Dyn => write!(f, "Dyn"),
            Type::Int => write!(f, "Int"),
            Type::Bool => write!(f, "Bool"),
            Type::Fun(a, b) => write!(f, "({a} -> {b})"),
        }
    }
}
