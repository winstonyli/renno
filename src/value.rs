use std::fmt;
use std::rc::Rc;

use crate::cont::Cont;
use crate::env::Env;
use crate::expr::ExprRef;
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
    pub body: ExprRef,
    pub env: Env,
    pub deep: bool,
}

#[derive(Clone, Copy)]
pub enum Builtin {
    Deep,
    Shallow,
    // Str -> Int (character count) or List -> Int (element count).
    Len,
}

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Str(Rc<str>),
    // Rc<Vec<Value>>, not a persistent cons-list: most list use in a
    // scripting language is indexing/iteration, which arrays serve better
    // than cons-lists -- and renno can't yet write cons-list-shaped
    // recursive functions anyway (no pattern matching, no general
    // recursion). Revisit if/when those land and list-heavy functional
    // code becomes common.
    List(Rc<Vec<Value>>),
    Closure(String, ExprRef, Env),
    Continuation(Cont),
    Handler(Rc<HandlerData>),
    Builtin(Builtin),
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

    pub fn as_str(&self) -> &str {
        match self {
            Value::Str(s) => s,
            _ => panic!("expected string"),
        }
    }

    // Runtime side of gradual typing: does this value's tag match the
    // static Type it's being checked against? Fun matches any callable
    // representation (Closure/Continuation/Builtin) -- Handler isn't
    // callable via App, so it doesn't match Fun. This is deliberately a
    // shallow "is it callable at all" check, not "does it have this exact
    // signature" -- typecheck::coerce wraps Dyn-to-Fun crossings in a real
    // per-call contract (checking each argument/result) instead of relying
    // on this alone; this stays the innermost callability primitive that
    // contract bottoms out on, the same role Int/Bool tag-checks play.
    pub fn matches_type(&self, t: &Type) -> bool {
        match (self, t) {
            (_, Type::Dyn) => true,
            (Value::Int(_), Type::Int) => true,
            (Value::Bool(_), Type::Bool) => true,
            (Value::Str(_), Type::Str) => true,
            // Shallow, like Fun -- confirms "this is a list," not that its
            // elements match the declared element type.
            (Value::List(_), Type::List(_)) => true,
            (Value::Closure(..), Type::Fun(_, _, _)) => true,
            (Value::Continuation(_), Type::Fun(_, _, _)) => true,
            (Value::Builtin(_), Type::Fun(_, _, _)) => true,
            _ => false,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "Int",
            Value::Bool(_) => "Bool",
            Value::Str(_) => "Str",
            Value::List(_) => "List",
            Value::Closure(..) | Value::Continuation(_) | Value::Builtin(_) => "Fun",
            Value::Handler(_) => "Handler",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{n}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Str(s) => write!(f, "{s}"),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Value::Closure(..) | Value::Continuation(_) | Value::Builtin(_) => write!(f, "<function>"),
            Value::Handler(_) => write!(f, "<handler>"),
        }
    }
}

// A Send-safe summary of a Value. Value itself is built entirely on Rc
// (Env/Cont/Expr all use it) -- deliberately, since the interpreter never
// needs real concurrency and Rc avoids Arc's atomic refcount overhead on
// every clone in the hot evaluation loop. That makes Value itself !Send,
// which matters at exactly one seam: run_source (lib.rs) runs the
// parser/typechecker on a dedicated large-stack thread to avoid a native
// stack overflow on deeply nested source, and the result has to cross
// back over a thread boundary. Outcome is what crosses it.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Int(i64),
    Bool(bool),
    Str(String),
    List(Vec<Outcome>),
    Function,
    Handler,
}

impl Outcome {
    pub fn as_int(&self) -> i64 {
        match self {
            Outcome::Int(n) => *n,
            _ => panic!("expected int"),
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Outcome::Bool(b) => *b,
            _ => panic!("expected bool"),
        }
    }
}

impl From<&Value> for Outcome {
    fn from(v: &Value) -> Outcome {
        match v {
            Value::Int(n) => Outcome::Int(*n),
            Value::Bool(b) => Outcome::Bool(*b),
            Value::Str(s) => Outcome::Str(s.to_string()),
            Value::List(items) => Outcome::List(items.iter().map(Outcome::from).collect()),
            Value::Closure(..) | Value::Continuation(_) | Value::Builtin(_) => Outcome::Function,
            Value::Handler(_) => Outcome::Handler,
        }
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Outcome::Int(n) => write!(f, "{n}"),
            Outcome::Bool(b) => write!(f, "{b}"),
            Outcome::Str(s) => write!(f, "{s}"),
            Outcome::List(items) => {
                write!(f, "[")?;
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                }
                write!(f, "]")
            }
            Outcome::Function => write!(f, "<function>"),
            Outcome::Handler => write!(f, "<handler>"),
        }
    }
}
