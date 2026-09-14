use std::rc::Rc;

use crate::cont::Cont;
use crate::env::Env;
use crate::expr::Expr;

// A handler as data: which effect it handles, the clause body plus its two
// binder names, the env it closes over, and whether it reinstalls itself
// around the resumed continuation (deep) or not (shallow, the default from
// MakeHandler). `deep`/`shallow` builtins just flip this bit on a clone --
// no AST-level distinction needed.
#[derive(Clone)]
pub struct HandlerData {
    pub effect: String,
    pub payload_var: String,
    pub resume_var: String,
    pub body: Rc<Expr>,
    pub env: Env,
    pub deep: bool,
}

#[derive(Clone, Copy)]
pub enum Builtin {
    Deep,
    Shallow,
}

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Closure(String, Rc<Expr>, Env),
    Continuation(Cont),
    Handler(Rc<HandlerData>),
    Builtin(Builtin),
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
