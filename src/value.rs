use std::rc::Rc;

use crate::cont::Cont;
use crate::env::Env;
use crate::expr::Expr;

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Closure(String, Rc<Expr>, Env),
    Continuation(Cont),
    Unit,
}

impl Value {
    pub fn as_int(&self) -> i64 {
        match self {
            Value::Int(n) => *n,
            _ => panic!("expected int"),
        }
    }
}
