use std::rc::Rc;

use crate::cont::Cont;
use crate::env::Env;
use crate::expr::Expr;
use crate::types::Type;

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
    Bool(bool),
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

    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            _ => panic!("expected bool"),
        }
    }

    // Runtime side of gradual typing: does this value's tag match the
    // static Type it's being checked against? Fun matches any callable
    // representation (Closure/Continuation/Builtin) -- Handler isn't
    // callable via App, so it doesn't match Fun.
    pub fn matches_type(&self, t: &Type) -> bool {
        match (self, t) {
            (_, Type::Dyn) => true,
            (Value::Int(_), Type::Int) => true,
            (Value::Bool(_), Type::Bool) => true,
            (Value::Closure(..), Type::Fun(_, _)) => true,
            (Value::Continuation(_), Type::Fun(_, _)) => true,
            (Value::Builtin(_), Type::Fun(_, _)) => true,
            _ => false,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "Int",
            Value::Bool(_) => "Bool",
            Value::Closure(..) | Value::Continuation(_) | Value::Builtin(_) => "Fun",
            Value::Handler(_) => "Handler",
            Value::Unit => "Unit",
        }
    }
}
