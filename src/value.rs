use std::fmt;
use std::rc::Rc;

use crate::cont::Cont;
use crate::env::Env;
use crate::expr::ExprRef;

// A handler as data: which effect it handles, the clause body, the env it
// closes over (the body's two binders, payload and resume, are pushed as a
// frame at call time -- resolve.rs assigns them fixed slots, so their names
// don't need to be carried here), and whether it reinstalls itself around
// the resumed continuation (deep) or not (shallow, the default from
// MakeHandler). `deep`/`shallow` builtins just flip this bit on a clone --
// no AST-level distinction needed.
#[derive(Clone)]
pub struct HandlerData {
    pub effect: String,
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
    // (Dyn, Str) -> Bool -- does this Record have a field with this name?
    // Record's own field-presence analogue to Get: named lookup instead
    // of positional, and a bool instead of panicking, since "does it have
    // this field" (unlike "index N of this list") is a genuinely common
    // question to ask about a value whose exact shape isn't statically
    // known -- width subtyping means a Dyn-sourced record's exact field
    // set is routinely broader than what any one annotation names. What
    // typecheck::build_boundary_check's Record arm desugars a Dyn-to-
    // Record boundary check into: is_record(v) && has_field(v, "x") &&
    // ... , one clause per required field name.
    HasField,
    // (Dyn, Str) -> Dyn -- extracts a named field's value. What `.field`
    // access desugars into (typecheck::elaborate_node's own
    // Expr::FieldAccess arm) rather than a dedicated runtime opcode --
    // same "no new machine.rs code" story build_predicate_call's other
    // callers already get, since this is just an ordinary two-argument
    // call like HasField. Panics, like Get, rather than returning an
    // Option: renno has no Option/Result in the prelude, and every other
    // Dyn-sourced shape mismatch already panics at the point of use.
    GetField,
    // Dyn -> Bool, one per primitive tag. typecheck::coerce desugars a
    // Dyn-to-primitive boundary Check into `if is_X(e) then e else
    // fail(...)` using these, instead of a dedicated Check AST node/Frame
    // -- see coerce's own doc comment. IsFun covers every callable
    // representation (Closure/RecClosure/Continuation/Builtin/
    // PartialBuiltin), the same shallow "is it callable at all" question
    // Value::matches_type's own Fun arm used to answer.
    IsInt,
    // Dyn -> Bool. Float's own tag test, same story as IsInt -- used both
    // directly (a `Float`-annotated Dyn boundary) and as one half of the
    // numeric Union check typecheck::coerce_numeric builds for Add/Sub/
    // Mul/Div/Mod/Lt's own Dyn operands.
    IsFloat,
    IsBool,
    IsStr,
    IsList,
    IsFun,
    // Dyn -> Bool. Record's own tag test, same story as the other IsX
    // builtins above -- see Value::Record's own doc comment for why it's
    // a distinct kind from List (name-keyed, not positional) rather than
    // reusing IsList.
    IsRecord,
    // Dyn -> Str: the same string Value::type_name() computes, exposed so
    // a desugared boundary-check failure can build its own "found {type}"
    // message at runtime (the actual runtime value's type isn't known
    // until then).
    TypeName,
    // Dyn -> Dyn: writes the argument's own Display impl to stdout (same
    // rendering the REPL/CLI already gives a program's final result --
    // see main.rs's run_and_print) followed by a newline, then returns
    // the argument UNCHANGED -- so `print(x)` reads as a transparent
    // side-effecting echo, usable inline (`print(x) + 1`) rather than
    // needing a separate binding just to observe a value. A raw Rust-
    // level side effect, not routed through renno's own `perform`/
    // `handle` -- same precedent Fail already sets (its panic! is
    // likewise invisible to the effect-row tracker), not a new pattern.
    Print,
    // Dyn -> Str: the same rendering Print writes to stdout, captured as
    // a Str instead -- `"x = " ++ to_str(x)` is renno's answer to string
    // interpolation (no `"...${e}..."` syntax; see [[renno_future_string_
    // interpolation]]), so this is the one piece those needed, not a
    // separate feature of its own.
    ToStr,
    // (a -> Bool, [a]) -> [a]. Map's own structural sibling -- same
    // callback-driven shape, same effect-handling caveat (see Map's own
    // doc comment).
    Filter,
    // [a] -> [a]. Single-arg, unlike Filter/Map -- nothing to call back
    // into, just an ordinary structural rebuild.
    Reverse,
    // ([a], [b]) -> [(a, b)]. Stops at the shorter list -- there's no
    // renno-level Option/Result to pad the longer one out with, the same
    // "no partial value to invent" story Get's out-of-range panic and
    // GetField's missing-field panic already tell, just resolved by
    // truncating instead of panicking since running out of pairs isn't a
    // shape error the way indexing past the end is.
    Zip,
    // ((a, a) -> Bool, [a]) -> [a]. `cmp(x, y)` means "x belongs at or
    // before y". Bool, not a three-way Ordering -- renno has no Ordering
    // type, and Bool is the same contract every other renno-level
    // predicate (Filter's own, map's callback) already uses. ponytail:
    // dispatch_builtin's own Sort arm calls `cmp` up to twice per
    // comparison (translating this Bool predicate into Rust's own
    // Ordering for sort_by) -- upgrade to a single 3-way callback only if
    // sort ever shows up in a profile; not worth an Ordering type for one
    // builtin otherwise.
    Sort,
    // (Int, Int) -> [Int], exclusive of the end (`range(0, 3)` is
    // `[0, 1, 2]`) -- same convention as Rust's own `0..3`, not `..=`.
    Range,
    // (Str, Str) -> [Str]: `split(s, sep)` -- subject first, same
    // argument order as Get/HasField/GetField's own (subject, ...)
    // convention. Splitting on "" is left to Rust's own str::split
    // behavior (an empty-string separator, which yields the string cut
    // between every char) rather than special-cased.
    Split,
    // ([Str], Str) -> Str: `join(parts, sep)` -- the inverse of Split,
    // same subject-first argument order.
    Join,
    // Str -> Str: strips leading/trailing whitespace, same definition
    // Rust's own str::trim uses (Unicode `White_Space`, not just ASCII).
    Trim,
}

impl Builtin {
    pub fn arity(self) -> usize {
        match self {
            Builtin::Deep
            | Builtin::Shallow
            | Builtin::Len
            | Builtin::Fail
            | Builtin::IsInt
            | Builtin::IsFloat
            | Builtin::IsBool
            | Builtin::IsStr
            | Builtin::IsList
            | Builtin::IsFun
            | Builtin::IsRecord
            | Builtin::TypeName
            | Builtin::Print
            | Builtin::ToStr
            | Builtin::Reverse
            | Builtin::Trim => 1,
            Builtin::Map
            | Builtin::Get
            | Builtin::HasField
            | Builtin::GetField
            | Builtin::Filter
            | Builtin::Zip
            | Builtin::Sort
            | Builtin::Range
            | Builtin::Split
            | Builtin::Join => 2,
            Builtin::Fold => 3,
        }
    }
}

#[derive(Clone)]
pub enum Value {
    Int(i64),
    // See types::Type::Float's own doc comment -- a distinct numeric kind
    // from Int, not silently unified with it at this representation
    // level either; apply_binop is the one place that reaches across the
    // two (promoting an Int operand to f64 whenever the other operand is
    // Float).
    Float(f64),
    Bool(bool),
    Str(Rc<str>),
    // What an `opaque` expression evaluates to -- a DISTINCT kind from
    // Int specifically so it can never be confused with (or forged as) a
    // real Int value: Pattern::Int can never match it. Never a legal type
    // annotation target (none of is_int/is_bool/is_str/is_list/is_fun
    // ever recognizes one -- see typecheck::build_boundary_check;
    // Type::Token itself has no surface spelling either).
    Token(u64),
    // Rc<Vec<Value>>, not a persistent cons-list: most list use in a
    // scripting language is indexing/iteration, which arrays serve better
    // than cons-lists. Pattern matching (Pattern::List/Cons) and tuples
    // (a fixed-arity List, see types::Type::Tuple) both destructure this
    // same representation.
    List(Rc<Vec<Value>>),
    // A record: name-keyed, unlike Tuple's plain positional List (see
    // types::Type::Record's own doc comment for why records need this --
    // width subtyping needs a value whose fields can be looked up by
    // name, not just position, so an extra field a narrower type never
    // asked for is simply never observed instead of needing to be
    // projected away at some boundary). Order is never significant here
    // (lookup is always by name -- see machine::match_pattern's own
    // Pattern::Record arm), though the parser's existing sort-by-name
    // behavior is harmless and left in place.
    Record(Rc<Vec<(String, Value)>>),
    Closure(ExprRef, Env),
    // `let rec f = fun p -> body [and g = ... ...] in ...` where EVERY value
    // is a direct Lambda (resolve::is_direct_group): `group[i]` is the body
    // of the i-th function in one mutually-recursive group (a plain
    // single-function `let rec`, no `and`, is just the length-1 case). This
    // value IS group[index]; calling it builds ONE frame [every group
    // member's RecClosure..., the argument] on the closure env (the layout
    // resolve.rs assigns) every time it's called -- not once, at
    // construction -- which is what lets any member's body call ANY sibling
    // (including itself) with no mutation and no Rc cycle: each call
    // re-derives its own siblings.
    RecClosure(Rc<[ExprRef]>, usize, Env),
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

    pub fn as_float(&self) -> f64 {
        match self {
            Value::Float(x) => *x,
            _ => panic!("expected float"),
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

    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::Str(_) => "Str",
            Value::Token(_) => "Token",
            Value::List(_) => "List",
            Value::Record(_) => "Record",
            Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..) => "Fun",
            Value::Handler(_) => "Handler",
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{n}"),
            // Always at least one decimal digit -- Rust's own f64 Display
            // prints a whole number like 4.0 as bare "4", which would be
            // indistinguishable from Value::Int(4)'s own output. Only
            // forced for the whole-number case; 3.14 still prints as
            // "3.14" via the ordinary `{n}` path, not truncated to "3.1".
            Value::Float(n) if n.fract() == 0.0 && n.is_finite() => write!(f, "{n:.1}"),
            Value::Float(n) => write!(f, "{n}"),
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
            // Prints every field the value actually carries -- including
            // ones width subtyping let it keep that no annotation ever
            // named (see Outcome::from's own doc comment on this Value).
            Value::Record(fields) => {
                write!(f, "{{")?;
                for (i, (name, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {v}")?;
                }
                write!(f, "}}")
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
    Float(f64),
    Bool(bool),
    Str(String),
    // Mirrors Value::Token -- see its own doc comment.
    Token,
    List(Vec<Outcome>),
    // Mirrors Value::Record -- see its own doc comment.
    Record(Vec<(String, Outcome)>),
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
            Value::Float(n) => Outcome::Float(*n),
            Value::Bool(b) => Outcome::Bool(*b),
            Value::Str(s) => Outcome::Str(s.to_string()),
            Value::Token(_) => Outcome::Token,
            Value::List(items) => Outcome::List(items.iter().map(Outcome::from).collect()),
            // Width subtyping is only invisible to PATTERN matching (see
            // Pattern::Record's own doc comment) -- it does NOT hide a
            // wider value's extra fields from THIS conversion, which
            // walks every field the Value actually carries, not just the
            // ones some earlier annotation happened to name. A record
            // width-coerced to satisfy a narrower type still crosses to
            // Outcome (and Display, below) with every field intact --
            // same kind of leak Token's own Display/Outcome arm already
            // has for its hidden brand id, just for a Record's "extra"
            // fields instead of a Token's numeric id.
            Value::Record(fields) => {
                Outcome::Record(fields.iter().map(|(n, v)| (n.clone(), Outcome::from(v))).collect())
            }
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
            // Same whole-number formatting as Value::Float's own Display arm.
            Outcome::Float(n) if n.fract() == 0.0 && n.is_finite() => write!(f, "{n:.1}"),
            Outcome::Float(n) => write!(f, "{n}"),
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
            Outcome::Record(fields) => {
                write!(f, "{{")?;
                for (i, (name, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {v}")?;
                }
                write!(f, "}}")
            }
            Outcome::Function => write!(f, "<function>"),
            Outcome::Handler => write!(f, "<handler>"),
        }
    }
}
