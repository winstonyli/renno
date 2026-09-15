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

#[derive(Clone, Copy, PartialEq)]
pub enum Builtin {
    Deep,
    Shallow,
    // Str -> Int (character count) or List -> Int (element count).
    Len,
    // (a -> b, [a]) -> [b]. Applies its callback via machine::apply --
    // see that function's doc comment for the effect-handling caveat
    // (the callback runs in a fresh continuation, so an effect it
    // performs can never reach a `handle` wrapping the outer map call).
    Map,
    // (acc -> a -> acc, acc, [a]) -> acc, left to right. The structural
    // eliminator for List: a native, Rust-loop-driven fold is how renno
    // gets real list-consuming recursion without a general `let rec` --
    // unconditionally terminating (bounded by the list's own length),
    // no user-definable fixpoint needed. Same effect-handling caveat as
    // Map.
    Fold,
    // Str -> Dyn (never actually returns -- always panics with its
    // argument as the message). What a `where` refinement clause
    // (parser::desugar_refinement) desugars into when it CAN'T be proven
    // at parse time: `let n: Int where P = val in body` becomes
    // `let n: Int = val in if P then body else fail("...")`, an ordinary
    // runtime check built entirely from existing If/App nodes.
    Fail,
    // ([a], Int) -> a -- indexing by function rather than new `[]`
    // syntax/BinOp, the same way structural list operations (map/fold)
    // are already builtins, not operators. Panics on a negative or
    // out-of-range index; renno has no Option/Result in the prelude to
    // return instead, and match_pattern/apply_binop already establish
    // "a Dyn-sourced shape mismatch panics at the point of use" as this
    // interpreter's one error-handling story.
    Get,
}

impl Builtin {
    pub fn arity(self) -> usize {
        match self {
            Builtin::Deep | Builtin::Shallow | Builtin::Len | Builtin::Fail => 1,
            Builtin::Map | Builtin::Get => 2,
            Builtin::Fold => 3,
        }
    }
}

#[derive(Clone)]
pub enum Value {
    Int(i64),
    Bool(bool),
    Str(Rc<str>),
    // An `opaque` `data` constructor's hidden trailing brand tag (see
    // parser::build_ctor_value) -- a DISTINCT kind from Int specifically
    // so it can never be confused with (or forged as) a real Int field:
    // Pattern::Int can never match it, and Pattern::Token can never match
    // an ordinary Value::Int. Never produced by any surface syntax, never
    // a legal type annotation target (matches_type always rejects it);
    // purely an implementation-internal runtime tag.
    Token(u64),
    // Rc<Vec<Value>>, not a persistent cons-list: most list use in a
    // scripting language is indexing/iteration, which arrays serve better
    // than cons-lists. Pattern matching (Pattern::List/Cons) and `data`
    // (a tagged List, see types::Type::Data) both destructure this same
    // representation.
    List(Rc<Vec<Value>>),
    Closure(String, ExprRef, Env),
    // `let rec f = fun p -> body [and g = ... ...] in ...`: `group[i]` is
    // (name_i, param_i, body_i) for the i-th binding in one mutually-
    // recursive group (a plain single-function `let rec`, no `and`, is
    // just the length-1 case -- no separate representation for it). This
    // value IS group[index]; calling it rebinds EVERY name in the group
    // into the environment used to evaluate group[index]'s body, every
    // time it's called (not once, at construction) -- each pointing at
    // its own fresh RecClosure over the same group. That's what lets any
    // member's body call ANY sibling (including itself) by name, with no
    // mutation and no AST-rewriting Y-combinator encoding -- Env is
    // already persistent and cheap to extend, so re-deriving "an env with
    // the whole group bound" on each call is just N more `bind`s, same
    // idea as binding the parameter itself.
    RecClosure(Rc<Vec<(String, String, ExprRef)>>, usize, Env),
    Continuation(Cont),
    Handler(Rc<HandlerData>),
    Builtin(Builtin),
    // A multi-arg Builtin (Map, Fold) with some but not all of its
    // arguments collected so far. Applying it adds one more; once the
    // count reaches `Builtin::arity`, the real operation dispatches.
    PartialBuiltin(Builtin, Rc<Vec<Value>>),
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
            // Same shallowness as List above, for the DEFENSIVE fallback
            // path only -- typecheck::coerce normally builds an
            // Expr::CheckData instead of an ordinary Check for a Data
            // target (see its own doc comment), which actually verifies
            // the constructor tag, field count, and (when opaque) brand.
            // This arm only fires if that lookup somehow failed to find
            // the declaration, so it stays the same shallow "some
            // tagged/ADT-shaped value" check CheckData was built to
            // replace.
            (Value::List(items), Type::Data(_, _)) => !items.is_empty(),
            (Value::Closure(..), Type::Fun(_, _, _)) => true,
            (Value::RecClosure(..), Type::Fun(_, _, _)) => true,
            (Value::Continuation(_), Type::Fun(_, _, _)) => true,
            (Value::Builtin(_), Type::Fun(_, _, _)) => true,
            (Value::PartialBuiltin(..), Type::Fun(_, _, _)) => true,
            // Never satisfies any surface annotation -- there's no syntax
            // to even write a type that would mean "a brand tag."
            (Value::Token(_), _) => false,
            _ => false,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "Int",
            Value::Bool(_) => "Bool",
            Value::Str(_) => "Str",
            Value::Token(_) => "Token",
            Value::List(_) => "List",
            Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..) => "Fun",
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
            // Never the numeric id -- that would leak a meaningless
            // implementation detail (and previously did, back when this
            // was just a bare Value::Int). A branded value printed at the
            // top level (Value::List's own arm below loops straight
            // through to here for its trailing element) now shows this
            // marker instead of a raw number.
            Value::Token(_) => write!(f, "<brand>"),
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
            Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..) => {
                write!(f, "<function>")
            }
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
    // Mirrors Value::Token -- see its own doc comment.
    Token,
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
            Value::Token(_) => Outcome::Token,
            Value::List(items) => Outcome::List(items.iter().map(Outcome::from).collect()),
            Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..) => {
                Outcome::Function
            }
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
            Outcome::Token => write!(f, "<brand>"),
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
