use std::fmt;
use std::rc::Rc;

use crate::cont::Captured;
use crate::env::Env;
use crate::expr::{ExprRef, Lazy, Len, ConformSpec};
use crate::list::List;
use crate::text::Text;
use crate::types::{first_free, Type};

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
    // (a -> b, [a]) -> [b]. Runs as a Frame::Iterate (iterate.rs): the
    // callback is applied on the real continuation, so an effect it
    // performs reaches a `handle` around the map call, multi-shot included.
    Map,
    // (acc -> a -> acc, acc, [a]) -> acc, left to right. The structural
    // eliminator for List: a native, step-driven fold is how renno
    // gets real list-consuming recursion without a general `let rec` --
    // unconditionally terminating (bounded by the list's own length),
    // no user-definable fixpoint needed. Runs as a Frame::Iterate like Map.
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
    // (Dyn, Str) -> Dyn -- extracts a named field's value. What `.field`
    // access desugars into (typecheck::elaborate_node's own
    // Expr::FieldAccess arm) rather than a dedicated runtime opcode --
    // same "no new machine.rs code" story, since this is just an ordinary
    // two-argument call like Get. Panics, like Get, rather than returning an
    // Option: renno has no Option/Result in the prelude, and every other
    // Dyn-sourced shape mismatch already panics at the point of use.
    GetField,
    // (Dyn, Str, Dyn) -> Dyn -- a record with one named field replaced. What `{a: e, ..r}` desugars into (typecheck's
    // Expr::RecordUpdate arm), one call per label. The original record is untouched (copy on write).
    SetField,
    // (Type, Dyn) -> Bool: does the value satisfy the type, for user
    // programs. A Dyn boundary check is a native Expr::Conform instead; both
    // share machine::test_holds, and the type is lowered by
    // typecheck::lower_test. A List type is tested element by element, Dyn
    // always holds, and a Fun type is callability only (Closure, RecClosure,
    // Continuation, Builtin, PartialBuiltin or Contract), ignoring the
    // parameter and result types.
    Is,
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
    // interpolation (no `"...${e}..."` syntax), so this is the one piece those needed, not a
    // separate feature of its own.
    ToStr,
    // (a -> Bool, [a]) -> [a]. Map's own structural sibling -- same
    // callback-driven shape, also a Frame::Iterate (see Map's own doc
    // comment).
    Filter,
    // [a] -> [a]. Single-arg, unlike Filter/Map -- nothing to call back
    // into, just an ordinary structural rebuild.
    Reverse,
    // ((a, a) -> Bool, [a]) -> [a]. `cmp(x, y)` means "x belongs at or
    // before y". Bool, not a three-way Ordering -- renno has no Ordering
    // type, and Bool is the same contract every other renno-level
    // predicate (Filter's own, map's callback) already uses. Calls `cmp`
    // once per comparison, as a step-driven natural merge sort
    // (iterate.rs): a merge takes the left element iff cmp(left, right),
    // so a `<=` comparator is stable and a strict `<` orders ties
    // right-first. Sorted and strictly reversed input take n-1 calls.
    Sort,
    // (Int, Int) -> [Int], exclusive of the end (`range(0, 3)` is
    // `[0, 1, 2]`) -- same convention as Rust's own `0..3`, not `..=`.
    Range,
    // ([Str], Str) -> Str: `join(parts, sep)` -- subject first, same
    // argument order as Get/GetField's own (subject, ...) convention.
    Join,
    // A prelude type constructor (`Int`, `List`, `Tuple`, ...): a constant
    // reads as its Value::Type, a constructor builds one from its argument.
    Ty(TypeCtor),
}

// The prelude names that build type values. An annotation is desugared into
// calls to these (parser::type_call) and evaluated by machine::eval_pure; in
// value position they are ordinary prelude bindings.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TypeCtor {
    Int,
    Float,
    Bool,
    Str,
    Dyn,
    // T -> [T]
    List,
    // A non-empty tuple or list of types -> (A, B, ...)
    Tuple,
    // A record of types -> {x: A, ...}
    Record,
    // A non-empty tuple or list of types -> A | B | ...
    Union,
    // (A, B) -> (A -> B); (A, "e", B) -> (A ->{e} B), the row variable by name.
    Fun,
    // A non-negative Int k -> Vec(k); an indexed type -> Vec at its index.
    // A symbolic index (`Vec(n)`) is not a value, so an annotation passes it
    // as the type Vec(n), an Expr::TypeLit.
    Vec,
    // (T, k) or (T, an indexed type) -> T at that index. An annotation passes
    // `Indexed(T, n)` as (T, Vec(n)) when n is symbolic, so it may name a variable.
    Indexed,
    // A token (the value of an `opaque`) -> the singleton type of that brand.
    // Only code holding the token can name its type.
    TypeOf,
}

impl TypeCtor {
    // The constants, which a prelude lookup turns straight into a type value.
    pub fn constant(self) -> Option<Type> {
        match self {
            TypeCtor::Int => Some(Type::Int),
            TypeCtor::Float => Some(Type::Float),
            TypeCtor::Bool => Some(Type::Bool),
            TypeCtor::Str => Some(Type::Str),
            TypeCtor::Dyn => Some(Type::Dyn),
            TypeCtor::List
            | TypeCtor::Tuple
            | TypeCtor::Record
            | TypeCtor::Union
            | TypeCtor::Fun
            | TypeCtor::Vec
            | TypeCtor::Indexed
            | TypeCtor::TypeOf => None,
        }
    }
}

impl Builtin {
    pub fn arity(self) -> usize {
        match self {
            Builtin::Deep
            | Builtin::Shallow
            | Builtin::Len
            | Builtin::Fail
            | Builtin::Print
            | Builtin::ToStr
            | Builtin::Reverse
            | Builtin::Ty(_) => 1,
            Builtin::Map
            | Builtin::Is
            | Builtin::Get
            | Builtin::GetField
            | Builtin::Filter
            | Builtin::Sort
            | Builtin::Range
            | Builtin::Join => 2,
            Builtin::Fold | Builtin::SetField => 3,
        }
    }

    // What a prelude reference to this builtin evaluates to.
    pub fn value(self) -> Value {
        match self {
            Builtin::Ty(c) => match c.constant() {
                Some(t) => Value::Type(Rc::new(t)),
                None => Value::Builtin(self),
            },
            _ => Value::Builtin(self),
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
    // A flat string or a view of a growable buffer that `++` extends in place; see text.rs.
    Str(Text),
    // What an `opaque` expression evaluates to -- a DISTINCT kind from
    // Int specifically so it can never be confused with (or forged as) a
    // real Int value: Pattern::Int can never match it. Never a legal type
    // annotation target (no check ever recognizes one -- see
    // typecheck::build_boundary_check;
    // Type::Token itself has no surface spelling either).
    Token(u64),
    // A claim-list (list.rs): contiguous blocks of write-once slots, so `::`, head
    // and tail are O(1) (recursion over a list is linear), `get` is O(1) on a list
    // built in bulk (a literal, `range`, a builtin's result) and O(blocks) on a
    // cons-built one, and iteration stays mostly contiguous. Pattern matching
    // (Pattern::List/Cons) and tuples (a fixed-arity List, see types::Type::Tuple)
    // both destructure this same representation.
    List(List),
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
    Continuation(Rc<Captured>),
    Handler(Rc<HandlerData>),
    Builtin(Builtin),
    // A multi-arg Builtin (Map, Fold) with some but not all of its
    // arguments collected so far. Applying it adds one more; once the
    // count reaches `Builtin::arity`, the real operation dispatches.
    PartialBuiltin(Builtin, Rc<Vec<Value>>),
    // A type as a value: what the prelude type constructors build and an
    // annotation evaluates to (machine::eval_pure).
    Type(Rc<Type>),
    // A function checked against a function type (machine::conform): applying
    // it checks the argument (at an untrusted site), applies `f` and checks the
    // result. Immutable shared data, so a multi-shot resume replays it safely.
    Contract(Rc<Contract>),
}

#[derive(Clone)]
pub struct Contract {
    pub f: Value,
    pub param: Rc<Lazy>,
    pub ret: Rc<Lazy>,
    pub ctx: Rc<SiteCtx>,
    // The site its result is blamed at: the same Rc as `ctx` unless two wrappers collapsed.
    pub ret_ctx: Rc<SiteCtx>,
}

// What a site's checks read the length of an index variable from: the site's
// own table (its annotation's index names, valued where the site was reached)
// and the witnesses contract calls have added since, innermost first.
pub struct SiteCtx {
    pub spec: Rc<ConformSpec>,
    pub table: Vec<Len>,
    pub witness: Option<Rc<Witness>>,
    // Where the function this site wraps sits inside its container, for an error's "function at" text.
    pub path: Option<Rc<PathStep>>,
}

pub struct Witness {
    pub name: String,
    pub len: Len,
    pub parent: Option<Rc<Witness>>,
}

// One step down from a container to a function inside it.
pub enum Step {
    Index(usize),
    Slot(usize),
    Field(String),
}

pub struct PathStep {
    pub step: Step,
    pub parent: Option<Rc<PathStep>>,
}

impl PathStep {
    // The steps from the container down to the function, root first; empty for no path.
    pub fn text(path: &Option<Rc<PathStep>>) -> String {
        let mut steps = Vec::new();
        let mut node = path.as_deref();
        while let Some(p) = node {
            steps.push(&p.step);
            node = p.parent.as_deref();
        }
        steps
            .iter()
            .rev()
            .map(|s| match s {
                Step::Index(i) => format!("[{i}]"),
                Step::Slot(i) => format!(".{i}"),
                Step::Field(f) => format!(".{f}"),
            })
            .collect()
    }
}

impl SiteCtx {
    pub fn lookup(&self, name: &str) -> Option<Len> {
        let mut w = self.witness.as_deref();
        while let Some(node) = w {
            if node.name == name {
                return Some(node.len.clone());
            }
            w = node.parent.as_deref();
        }
        table_lookup(&self.spec.index_names, &self.table, name)
    }

    pub fn with_witness(&self, name: String, len: Len) -> SiteCtx {
        SiteCtx { spec: self.spec.clone(), table: self.table.clone(), witness: Some(Rc::new(Witness { name, len, parent: self.witness.clone() })), path: self.path.clone() }
    }

    pub fn with_path(&self, path: Option<Rc<PathStep>>) -> SiteCtx {
        SiteCtx { spec: self.spec.clone(), table: self.table.clone(), witness: self.witness.clone(), path }
    }
}

// The site-table entry for index variable `name`, if the annotation writes it.
pub fn table_lookup(names: &[String], table: &[Len], name: &str) -> Option<Len> {
    names.iter().position(|n| n == name).map(|i| table[i].clone())
}

impl Value {
    pub fn as_int(&self) -> i64 {
        match self {
            Value::Int(n) => *n,
            _ => crate::run_error!("expected int"),
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            Value::Float(x) => *x,
            _ => crate::run_error!("expected float"),
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            _ => crate::run_error!("expected bool"),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Value::Str(s) => s,
            _ => crate::run_error!("expected string"),
        }
    }

    // A message's name for the value's kind. A token's names its brand, as
    // its type does (Type::Token's Display), so two tokens' names differ.
    pub fn type_name(&self) -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(match self {
            Value::Int(_) => "Int",
            Value::Float(_) => "Float",
            Value::Bool(_) => "Bool",
            Value::Str(_) => "Str",
            Value::Token(id) => return std::borrow::Cow::Owned(format!("Token#{id}")),
            Value::List(_) => "List",
            Value::Record(_) => "Record",
            Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..) | Value::Contract(_) => "Fun",
            Value::Handler(_) => "Handler",
            // A group name read inside its own all-calls `let rec` (rule P) is still the
            // group's placeholder; name it, so a failed check says why it is not a value.
            Value::Type(t) => match first_free(t) {
                Some(name) => return std::borrow::Cow::Owned(format!("recursive type placeholder {name} (`{name}` inside its own `let rec` of calls is the group's own binding, not an outer `{name}`, and is not a value yet)")),
                None => "Type",
            },
        })
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
            Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..) | Value::Contract(_) => {
                write!(f, "<function>")
            }
            Value::Handler(_) => write!(f, "<handler>"),
            Value::Type(t) => write!(f, "{t}"),
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
    // A type value, as its display text.
    Type(String),
}

impl Outcome {
    pub fn as_int(&self) -> i64 {
        match self {
            Outcome::Int(n) => *n,
            _ => crate::run_error!("expected int"),
        }
    }

    pub fn as_bool(&self) -> bool {
        match self {
            Outcome::Bool(b) => *b,
            _ => crate::run_error!("expected bool"),
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
            Value::Closure(..) | Value::RecClosure(..) | Value::Continuation(_) | Value::Builtin(_) | Value::PartialBuiltin(..) | Value::Contract(_) => {
                Outcome::Function
            }
            Value::Handler(_) => Outcome::Handler,
            Value::Type(t) => Outcome::Type(t.to_string()),
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
            Outcome::Type(t) => write!(f, "{t}"),
        }
    }
}
