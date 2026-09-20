use std::collections::BTreeSet;
use std::fmt;
use std::rc::Rc;

use crate::index_expr::{index_exprs_equal, IndexExpr};
use crate::util::find_field;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Dyn,
    Int,
    // A second, DISTINCT numeric type from Int -- not generally consistent
    // with it (types::consistent has no Int<->Float arm; a [Float]-
    // annotated parameter still statically rejects a [Int] argument, same
    // as any other two concrete types). Int/Float interop lives ONLY in
    // typecheck::elaborate's own arithmetic BinOp arm (Add/Sub/Mul/Div/
    // Mod/Lt/Eq), via a local coerce_numeric helper that never touches
    // this general relation -- deliberately narrow, so this new type
    // can't quietly widen unrelated Tuple/Record/List/Fun consistency
    // checks elsewhere in the checker.
    Float,
    // A generalized value-type variable -- see typecheck::Scheme's own
    // doc comment for the full generalize-at-`let`/instantiate-at-use
    // story (mirrors EffectRow::Var exactly, just for ordinary types
    // instead of effect rows). Minted at many binding sites via
    // InferCtx::fresh_var, never written by a user -- there is no surface
    // syntax for this. Treated exactly like Type::Dyn by
    // `consistent`/`coerce` everywhere except unify(), which binds it
    // into infer.subst. Precision is recovered via InferCtx::resolve/
    // resolve_deep reading that binding back to substitute a concrete type.
    // IS surfaced to a user: an unresolved Type::Var can appear in a
    // static type-mismatch message (e.g. `id(id) + 1`). Its Display impl
    // renders it identically to Type::Dyn ("Dyn") rather than some
    // internal name like "<generic>" -- renno has no generics syntax a
    // user would recognize such a name as referring to, and since a
    // Type::Var behaves exactly like Dyn everywhere it's observable,
    // showing "Dyn" is the more honest, actionable thing to tell them.
    Var(String),
    Bool,
    Str,
    // Element type. renno's List is a native primitive (Value::List,
    // Rc<Vec<Value>>).
    List(Rc<Type>),
    // param, effect row (what calling this may perform), return type.
    Fun(Rc<Type>, EffectRow, Rc<Type>),
    // The static type of an `opaque` expression -- a singleton, one per
    // SOURCE POSITION (the u64), never per evaluation: `opaque` written at
    // one spot always has this same type no matter how many times that
    // code runs (Expr::Token is a literal, not a fresh-each-call
    // generator -- see its own doc comment). Two Token types are
    // consistent only when their ids match exactly (consistent_inner);
    // this is what lets a Token-typed tuple field carry real nominal
    // identity using nothing but ordinary structural comparison -- no
    // separate name-registry/brand-comparison mechanism needed.
    Token(u64),
    // A reference to a genuinely self-referential `type` alias's own
    // definition -- `type List = (Int, List) | Bool in ...` -- rather
    // than the ordinary, fully-expanded structural value every OTHER
    // (non-recursive) alias resolves to directly. The string is always
    // a gensym'd unique id in the shape "Base#N" (see parser::atom's
    // own `Token::TypeKw` arm), never the bare user-typed name --
    // aliases in different, unrelated scopes can share a surface name,
    // and the registry this leaf is looked up in (InferCtx's own
    // `named_types`, populated once by the parser) has to survive
    // past the parser's OWN scoped/LIFO-restored `type_aliases`, so it
    // needs a name that can never collide across scopes.
    //
    // A lightweight, non-expanding leaf, in the same spirit as
    // Type::Var/Type::Token: a tag consulted on demand, never
    // something requiring evaluation or unfolding to construct.
    // Comparison (consistent()/unify()) is purely nominal, by id --
    // see their own new arms -- so nothing here ever needs to unfold
    // what this id actually stands for, and no cycle-detection is
    // needed anywhere in this file's own structural recursion, even
    // though a registry entry for a genuinely recursive alias
    // structurally contains this SAME leaf pointing at itself.
    // build_shape_predicate/build_boundary_check/pattern_could_match/
    // coerce/unify_fits (typecheck.rs) are the consumers that ever look
    // up what a Named id stands for, and only one level at a time -- see
    // their own doc comments.
    Named(String),
    // Fixed-arity, per-position-typed product -- `(a, b, c)` syntax.
    // Anonymous (no declared name, no registry lookup): consistent_inner
    // compares two Tuple types directly, positionally, with nothing to
    // look up. Runtime representation is a plain Value::List (see
    // Expr::Tuple's own doc comment) -- no new Value kind needed.
    Tuple(Rc<Vec<Type>>),
    // Fixed-arity, per-position-typed product, like Tuple -- but each
    // position also carries a field name, and consistency additionally
    // requires the names to match (not just the types). ALWAYS stored
    // sorted by name (parser::parse_record_fields' job, for the type
    // itself, a construction, and a pattern alike) -- that's what makes
    // `{y: 2, x: 1}` and `{x: 1, y: 2}` the same type. Anonymous, no
    // registry: like Tuple, two Record types are compared directly,
    // positionally (name then type), nothing to look up.
    //
    // Unlike Tuple, Record does NOT share Tuple's runtime representation
    // -- it has its own real, name-keyed Value::Record (see its own doc
    // comment for why: width subtyping needs a value whose fields are
    // addressable by name, not position, so an extra field a narrower
    // type never asked for is simply never observed rather than needing
    // to be projected away at some boundary). `coerce` uses a SEPARATE,
    // one-directional relation for that -- types::record_satisfies, not
    // this type's own (exact, symmetric) consistency.
    Record(Rc<Vec<(String, Type)>>),
    // "One of these" -- a self-contained sum, no registry/name lookup.
    // Typically each alternative is a Tuple whose first element is a
    // distinct Token (a hand-rolled sum: `Type::Union([Tuple([
    // Token(id_none)]), Tuple([Token(id_some), Int])])`), but nothing
    // here requires that shape -- any two types can be unioned. Named via
    // `type Name = A | B in ...` (Parser::type_aliases); the union itself
    // has no runtime representation of its own, only whatever VALUE
    // actually flows through ends up being one alternative's own shape.
    Union(Rc<Vec<Type>>),
    // `Vec(n)` sugars over this -- NOT its own bespoke primitive. Wraps
    // either Type::List directly (the primary, worked case) or a
    // qualifying Type::Named+Type::Union (derived index-refinement, a
    // later phase) with one IndexExpr tracking its own length/size.
    // See the design spec's own section 2 for the full rationale.
    Indexed(Rc<Type>, Rc<IndexExpr>),
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
// No registry, no cycle guard: every Type here is a plain finite tree
// (Tuple/Union hold an Rc<Vec<Type>>, never a back-reference to
// themselves -- nothing in this checker ever builds a genuinely-cyclic
// Type value), so a plain structural recursion always terminates.
pub fn consistent(a: &Type, b: &Type) -> bool {
    match (a, b) {
        (Type::Dyn, _) | (_, Type::Dyn) => true,
        // Treated exactly like Dyn here -- the one place it behaves
        // differently is unify(), which binds it into infer.subst,
        // not a change to this general relation. See Type::Var's
        // own doc comment.
        (Type::Var(_), _) | (_, Type::Var(_)) => true,
        (Type::Int, Type::Int) => true,
        // Deliberately NOT (Type::Int, Type::Float) => true -- see
        // Type::Float's own doc comment on why that interop stays local
        // to the arithmetic operators, not this general relation.
        (Type::Float, Type::Float) => true,
        (Type::Bool, Type::Bool) => true,
        (Type::Str, Type::Str) => true,
        (Type::List(a), Type::List(b)) => consistent(a, b),
        (Type::Fun(a1, r1, b1), Type::Fun(a2, r2, b2)) => {
            consistent(a1, a2) && consistent(b1, b2) && row_consistent(r1, r2)
        }
        // A singleton per source position -- consistent only with itself
        // (and Dyn, already handled above). No registry lookup: the id
        // alone says everything there is to say.
        (Type::Token(a), Type::Token(b)) => a == b,
        // Nominal, not structural -- same singleton-by-id precedent
        // Token's own arm just above already sets. Two Named types are
        // consistent only when their ids match exactly; a Named type
        // is never consistent with anything structurally identical to
        // what it (or any other Named type) unfolds to. Never unfolds
        // the registry -- see Type::Named's own doc comment.
        (Type::Named(a), Type::Named(b)) => a == b,
        // Two Indexed types are consistent when their wrapped types are
        // consistent AND their index expressions are equal (§3's SOP
        // normalization) -- both halves required, mirroring how List's
        // own arm requires its element type to match.
        //
        // A bare, unbound index variable on EITHER side is permissive,
        // mirroring Type::Var's own unconditional permissiveness just
        // above -- this is specifically what lets coerce() no-op on a
        // Vec(n) parameter/annotation the same way it already no-ops
        // on an ordinary Type::Var one, so a REAL binding can happen
        // afterward via unify_fits (see Expr::App/Let/LetRec's own
        // elaboration). The guard is IndexExpr::Var specifically, not
        // "contains a variable anywhere" -- Vec(n+1) against Vec(4)
        // still correctly requires exact SOP-equality, matching
        // unify_index_expr's own identical no-equation-solving limit.
        // NOTE: consistent() is symmetric and also backs `==`/Union-
        // membership -- this permissiveness applies there too, which is
        // why Expr::Let/LetRec also gain a follow-up unify_fits call:
        // without one, a bare index variable accepted here can be left
        // silently UNBOUND with no runtime check at all (rigid vs.
        // flexible index variables is a deeper, still-open concern).
        (Type::Indexed(wa, ia), Type::Indexed(wb, ib)) => {
            consistent(wa, wb)
                && (index_exprs_equal(ia, ib) || matches!(**ia, IndexExpr::Var(_)) || matches!(**ib, IndexExpr::Var(_)))
        }
        // Positional, structural, no name/registry involved -- same idea
        // as List's own element comparison, just per-position instead of
        // one shared element type.
        (Type::Tuple(a), Type::Tuple(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| consistent(x, y))
        }
        // Same idea as Tuple, plus each position's NAME must also match --
        // reuses record_satisfies (the one-directional "has at least
        // these fields" relation coerce's width subtyping is built from)
        // for the actual per-field name+type check, adding only the
        // length equality that turns "at least" into "exactly." Sound
        // because field names within one Type::Record are always unique
        // (parse_record_fields rejects a duplicate) -- same length plus
        // every one of `a`'s (name, type) pairs found in `b` forces the
        // two name SETS to be identical, not just one contained in the
        // other.
        (Type::Record(a), Type::Record(b)) => a.len() == b.len() && record_satisfies(a, b),
        // Two unions: consistent as SETS -- every alternative on each
        // side has a match on the other (order and duplicates don't
        // matter, only membership). Must come before the single-Union
        // arm below (Rust checks patterns top to bottom; that arm's
        // `other: &Type` would otherwise also match a Union on the other
        // side).
        (Type::Union(a), Type::Union(b)) => {
            a.iter().all(|x| b.iter().any(|y| consistent(x, y))) && b.iter().all(|y| a.iter().any(|x| consistent(x, y)))
        }
        // One union, one concrete type: consistent if the concrete side
        // matches ANY ONE alternative -- "is this one of the possible
        // shapes," the same question a runtime value crossing a
        // Union-typed boundary would need answered.
        (Type::Union(alts), other) | (other, Type::Union(alts)) => alts.iter().any(|alt| consistent(alt, other)),
        // Every Tuple value IS, structurally, a List at runtime (see
        // Type::Tuple's own doc comment) -- needed so a tuple pattern
        // (parser::pattern_atom desugars `(p, q)` straight into
        // Pattern::List, typed List(Dyn) by typecheck::pattern_type, same
        // as any other List pattern) can match a Tuple-typed scrutinee.
        (Type::List(elem), Type::Tuple(_)) | (Type::Tuple(_), Type::List(elem)) => matches!(**elem, Type::Dyn),
        // No equivalent bridge for Record: a `{x, y}` pattern is now its
        // own Pattern::Record (see its own doc comment), not sugar over
        // Pattern::List, so typecheck::pattern_type gives it plain
        // Type::Dyn like every other imprecisely-typed pattern -- already
        // consistent with anything via the Dyn arm at the top of this
        // function, no special-casing needed here.
        _ => false,
    }
}

// One-directional: does `actual` have AT LEAST every field `required`
// names, each with a consistent type? (Extra fields on `actual` are
// fine -- see Pattern::Record's own doc comment for why nothing ever
// needs to know about them.) This is NOT part of `consistent()` --
// `consistent()` stays a symmetric equivalence relation everywhere,
// including for two Record types (exact field-set match, used by `==`
// and Union-alternative membership) -- width subtyping is a one-way
// "does this value fit where that type is expected" question, asked only
// by typecheck::coerce, alongside (not instead of) its own consistent()
// check.
pub fn record_satisfies(required: &[(String, Type)], actual: &[(String, Type)]) -> bool {
    required.iter().all(|(name, ty)| find_field(actual, name).is_some_and(|aty| consistent(ty, aty)))
}

// A directional generalization of record_satisfies to also cover Fun:
// is `actual` safely substitutable wherever `required` is expected?
// Fun is contravariant on its param (a function that accepts a WIDER
// range of inputs -- i.e. a NARROWER required param type -- can stand
// in for one declared to need only a narrower range) and covariant on
// its return (a function promising a MORE PRECISE, narrower return can
// stand in for one that only promised something wider). Record reuses
// record_satisfies's own width-tolerant shape (walk `required`'s own
// fields, find_field into `actual`), but recurses through `fits`
// itself rather than `consistent` -- deliberately a SEPARATE function
// from record_satisfies, not a generalization it delegates to:
// consistent()'s own Type::Record arm needs record_satisfies to keep
// recursing through consistent() (exact, symmetric), since Eq/Union-
// membership must NOT silently gain Fun's own contravariant tolerance
// through a Record field. Everything else delegates straight to
// consistent(), unchanged.
pub fn fits(required: &Type, actual: &Type) -> bool {
    match (required, actual) {
        (Type::Fun(req_param, req_row, req_ret), Type::Fun(act_param, act_row, act_ret)) => {
            fits(act_param, req_param) && fits(req_ret, act_ret) && row_consistent(req_row, act_row)
        }
        (Type::Record(required), Type::Record(actual)) => required
            .iter()
            .all(|(name, ty)| find_field(actual, name).is_some_and(|aty| fits(ty, aty))),
        // "Forgetting" an Indexed value's own tracked index is a sound,
        // one-directional widening -- an Indexed-typed value is usable
        // anywhere its wrapped type is expected (e.g. a Vec(3)-typed list
        // passed where a plain [T] is required), the same direction every
        // other relation in this function already models. Guarded so this
        // ONLY fires when `required` itself isn't Indexed: two Indexed
        // types must still go through consistent()'s own exact-index arm
        // above (via the catch-all below), and a plain, non-Indexed
        // `actual` must NOT satisfy a required Indexed position -- that
        // asymmetry is the entire point of tracking the index at all, so
        // there is deliberately no symmetric arm the other way.
        (required, Type::Indexed(actual_wrapped, _)) if !matches!(required, Type::Indexed(..)) => {
            fits(required, actual_wrapped)
        }
        _ => consistent(required, actual),
    }
}

pub fn row_consistent(a: &EffectRow, b: &EffectRow) -> bool {
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
            Type::Float => write!(f, "Float"),
            // The name is internal bookkeeping (see Type::Var's own doc
            // comment) that a user has no way to interpret -- render
            // identically to Type::Dyn, which is what a Type::Var
            // actually behaves like everywhere it's observable.
            Type::Var(_) => write!(f, "Dyn"),
            // Shows the clean surface name, not the internal gensym'd
            // id -- mirrors Type::Var's own precedent just above
            // (hiding its internal name, always showing "Dyn" instead)
            // of not exposing machinery the user never wrote. The
            // surface name is always the part before the first "#" --
            // see the id's own construction in parser::atom.
            Type::Named(id) => write!(f, "{}", id.split('#').next().unwrap_or(id)),
            Type::Bool => write!(f, "Bool"),
            Type::Str => write!(f, "Str"),
            Type::List(elem) => write!(f, "[{elem}]"),
            // Dyn row prints as a plain arrow -- matches every existing
            // (unannotated-row) Fun type exactly as before this existed.
            Type::Fun(a, EffectRow::Dyn, b) => write!(f, "({a} -> {b})"),
            Type::Fun(a, row, b) => write!(f, "({a} ->{row} {b})"),
            // Id never printed -- opaque means opaque.
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
            Type::Record(fields) => {
                write!(f, "{{")?;
                for (i, (name, t)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{name}: {t}")?;
                }
                write!(f, "}}")
            }
            Type::Union(alts) => {
                for (i, t) in alts.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{t}")?;
                }
                Ok(())
            }
            Type::Indexed(wrapped, index) => write!(f, "{wrapped}({index})"),
        }
    }
}
