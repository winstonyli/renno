use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

use crate::expr::DataInfo;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Dyn,
    Int,
    Bool,
    Str,
    // Element type. renno's List is a native primitive (Value::List,
    // Rc<Vec<Value>>), not itself expressed via `data`/Pattern -- but
    // `data` declarations (see parser::build_ctor_value) go the other way
    // around: a user-defined ADT desugars INTO a tagged List, rather than
    // List becoming sugar over an ADT.
    List(Rc<Type>),
    // param, effect row (what calling this may perform), return type.
    Fun(Rc<Type>, EffectRow, Rc<Type>),
    // A `data`-declared type, named. Every constructor parser::parser
    // builds for one `data Name = ...` is annotated to return this (see
    // parser::ctor_type), so `Some(5)` synthesizes Data("Option") instead
    // of falling back to Dyn/[Dyn].
    //
    // Structural by default: consistent() looks up both names' recorded
    // constructor shapes (DataInfo::ctor_types) and accepts two
    // DIFFERENTLY-named Data types as consistent when those shapes match --
    // so `data Celsius = Mk(Int)` and `data Fahrenheit = Mk(Int)` DO unify,
    // the same way two structurally-identical List or Fun types always
    // have. Opt a type OUT of that (real nominal distinctness, e.g. so
    // Celsius and Fahrenheit can never be swapped for each other) by
    // giving one of its constructors an `opaque` field -- see
    // DataInfo::brand.
    //
    // `opaque` is enforced at RUNTIME too, not just statically: every
    // constructor of a branded type stamps one hidden trailing tag into
    // its value (parser::build_ctor_value), and every pattern that can
    // match it -- positional, named, or FieldAccess's own synthetic one --
    // carries the identical tag, so an unrelated value simply fails to
    // pattern-match rather than being silently accepted as this type. A
    // bare Dyn-boundary type annotation checks it too, via the
    // check_data_shape builtin (typecheck::build_boundary_check's Data
    // arm desugars into a call to it): its witness carries the
    // declaration's ctor shapes AND its brand id, so a same-shaped value
    // from a DIFFERENT opaque declaration (Meters's Mk vs Seconds's Mk)
    // is rejected there too, not just at an actual pattern-match/
    // field-access site.
    //
    // The second field is that same brand id, mirrored onto the STATIC
    // type (None for an unbranded/structural type) -- not just carried at
    // runtime in Value. Every place a Type::Data is built (parser::
    // ctor_type, parser::parse_type's bare-identifier case) fills it in
    // from whichever `data Name` declaration is lexically in scope there
    // (Parser::type_brands), so consistent_inner can tell two SEPARATELY-
    // declared, same-named opaque `data` blocks (shadowing) apart by brand
    // instead of by name alone -- see consistent_inner's own doc comment.
    Data(String, Option<u64>),
    // The static type of an `opaque` expression -- a singleton, one per
    // SOURCE POSITION (the u64), never per evaluation: `opaque` written at
    // one spot always has this same type no matter how many times that
    // code runs (Expr::Token is a literal, not a fresh-each-call
    // generator -- see its own doc comment). Two Token types are
    // consistent only when their ids match exactly (consistent_inner);
    // this is what lets a Token-typed tuple/record field carry real
    // nominal identity using nothing but ordinary structural comparison,
    // no separate brand-comparison machinery needed for values built this
    // way (contrast with Data's own brand field above, which predates
    // this and still has its own dedicated comparison arm).
    Token(u64),
    // Fixed-arity, per-position-typed product -- `(a, b, c)` syntax.
    // Anonymous (no declared name, no registry lookup): unlike Data,
    // consistent_inner compares two Tuple types directly, positionally,
    // with nothing to look up. Runtime representation is a plain
    // Value::List (see Expr::Tuple's own doc comment) -- same "no new
    // Value kind" choice Data's own tagged-list encoding already made.
    Tuple(Rc<Vec<Type>>),
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
//
// `fields` is the same registry elaborate/elaborate_node thread everywhere
// (every `data` block seen so far) -- needed only to look up a Data type's
// recorded shape when two DIFFERENT names meet (see the Data arm below);
// every other case ignores it.
pub fn consistent(a: &Type, b: &Type, fields: &[Rc<DataInfo>]) -> bool {
    consistent_inner(a, b, fields, &mut BTreeSet::new())
}

// `seen` tracks the (name, name) pairs currently being compared -- assumed
// consistent, coinductively, if re-encountered before this comparison
// finishes. Without it, two mutually-recursive differently-named types
// (`data ListA = NilA | ConsA(Int, ListA)` vs `data ListB = NilB |
// ConsB(Int, ListB)`) would recurse forever: comparing them structurally
// means comparing ConsA's Int field (fine) and its ListA field against
// ListB -- which is exactly the top-level comparison again. Standard
// technique for recursive-type equivalence (the same idea as the "Amber
// rules"): assume a pair equal while still verifying it, and only that
// assumption lets a genuinely-cyclic proof terminate instead of looping.
fn consistent_inner(a: &Type, b: &Type, fields: &[Rc<DataInfo>], seen: &mut BTreeSet<(String, String)>) -> bool {
    match (a, b) {
        (Type::Dyn, _) | (_, Type::Dyn) => true,
        (Type::Int, Type::Int) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::Str, Type::Str) => true,
        (Type::List(a), Type::List(b)) => consistent_inner(a, b, fields, seen),
        (Type::Fun(a1, r1, b1), Type::Fun(a2, r2, b2)) => {
            consistent_inner(a1, a2, fields, seen) && consistent_inner(b1, b2, fields, seen) && row_consistent(r1, r2)
        }
        // A singleton per source position -- consistent only with itself
        // (and Dyn, already handled above). No registry lookup: the id
        // alone says everything there is to say.
        (Type::Token(a), Type::Token(b)) => a == b,
        // Positional, structural, no name/registry involved -- same idea
        // as List's own element comparison, just per-position instead of
        // one shared element type.
        (Type::Tuple(a), Type::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| consistent_inner(x, y, fields, seen))
        }
        // Either side opaque (Some brand): consistent ONLY when both carry
        // the EXACT SAME brand -- i.e. resolve to the identical `data`
        // declaration. This is what closes the shadowing gap this arm used
        // to have: two SEPARATELY-declared, same-named opaque `data Foo`
        // blocks now carry DIFFERENT brand ids (Parser::type_brands is
        // reset per-declaration, see its own doc comment), so they're
        // never consistent with each other even though their names are
        // textually identical -- matching how they already behaved at
        // runtime (mismatched brand tag fails to pattern-match). A branded
        // type is also never consistent with an unbranded one, no matter
        // how identical the shapes look, same as before this fix (that
        // part didn't change, just how it's decided -- directly from the
        // type's own brand field now, no `fields` lookup needed).
        (Type::Data(_, a_brand), Type::Data(_, b_brand)) if a_brand.is_some() || b_brand.is_some() => {
            a_brand == b_brand
        }
        // Neither side opaque: same name is always consistent (a type is
        // always consistent with itself) -- also what lets an ordinary
        // self-referential `data` type (`Cons(Int, List)` inside `data
        // List` itself) compare cheaply with no lookup at all. Different
        // names: structural by default -- same set of constructor names,
        // each with pairwise-consistent field types. Either name failing
        // to resolve in `fields` (shouldn't happen once `fields` is the
        // real registry elaborate builds) conservatively rejects rather
        // than guesses. `.rev()`: an unbranded name can still be shadowed
        // by an unrelated (differently-shaped) redeclaration reusing the
        // same name -- pick the lexically-current one, same reasoning as
        // every other `fields` lookup in this codebase (see e.g.
        // typecheck::elaborate_node's FieldAccess arm).
        (Type::Data(a_name, _), Type::Data(b_name, _)) => {
            if a_name == b_name {
                return true;
            }
            let key =
                if a_name < b_name { (a_name.clone(), b_name.clone()) } else { (b_name.clone(), a_name.clone()) };
            if !seen.insert(key.clone()) {
                return true;
            }
            let result = (|| {
                let a_info = fields.iter().rev().find(|f| &f.type_name == a_name)?;
                let b_info = fields.iter().rev().find(|f| &f.type_name == b_name)?;
                if a_info.ctor_types.len() != b_info.ctor_types.len() {
                    return Some(false);
                }
                Some(a_info.ctor_types.iter().all(|(cname, ctys)| {
                    b_info.ctor_types.iter().find(|(n, _)| n == cname).is_some_and(|(_, ctys2)| {
                        ctys.len() == ctys2.len()
                            && ctys.iter().zip(ctys2).all(|(t1, t2)| consistent_inner(t1, t2, fields, seen))
                    })
                }))
            })()
            .unwrap_or(false);
            seen.remove(&key);
            result
        }
        // Every Data(name) value IS, structurally, a List at runtime (see
        // Type::Data's doc comment) -- a constructor's own VALUE
        // expression elaborates structurally (e.g. `None`'s `["None"]`
        // synthesizes List(Str), not Data("Option")), so this needs to
        // hold for parser::ctor_type's annotation to coerce cleanly, with
        // no runtime Check inserted (from isn't Dyn on either side).
        //
        // Narrowed to Str/Dyn element types, not "any List": provably
        // exactly the set a ctor body's own inferred list type can ever
        // be (the tag is always Str; any field of a different type widens
        // the WHOLE list to Dyn, which then stays Dyn -- see ListLit's
        // elaboration). A concretely-typed list like `[Int]` was never a
        // legitimate ctor body, so it's no longer accepted here either --
        // closes a real gap where an arbitrary `[Int]` value satisfied any
        // Data-annotated parameter with zero runtime check.
        (Type::List(elem), Type::Data(_, _)) | (Type::Data(_, _), Type::List(elem)) => {
            matches!(**elem, Type::Str | Type::Dyn)
        }
        // Every Tuple value IS, structurally, a List at runtime (see
        // Type::Tuple's own doc comment) -- needed so a tuple pattern
        // (parser::pattern_atom desugars `(p, q)` straight into
        // Pattern::List, typed List(Dyn) by typecheck::pattern_type, same
        // as any other List pattern) can match a Tuple-typed scrutinee.
        // Narrowed to Dyn specifically (not Str too, unlike Data's own
        // bridging arm above): a tuple pattern's desugared element type is
        // always exactly List(Dyn), never List(Str).
        (Type::List(elem), Type::Tuple(_)) | (Type::Tuple(_), Type::List(elem)) => matches!(**elem, Type::Dyn),
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
            // Brand never printed -- same "hidden" philosophy as the
            // runtime tag itself (see Type::Data's own doc comment).
            Type::Data(name, _) => write!(f, "{name}"),
            // Id never printed -- same "hidden" philosophy as Data's own
            // brand.
            Type::Token(_) => write!(f, "Token"),
            Type::Tuple(items) => {
                write!(f, "(")?;
                for (i, t) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
        }
    }
}
