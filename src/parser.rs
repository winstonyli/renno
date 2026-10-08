use std::collections::HashMap;
use std::rc::Rc;

use crate::expr::{Arena, BinOp, Expr, ExprRef, Pattern, SpanMap};
use crate::index_expr::IndexExpr;
use crate::lexer::{tokenize, Token};
use crate::span::Span;
use crate::types::{EffectRow, Type};
use crate::util::find_field;

// Delegates to parse_with_named_types and drops the registry -- kept as
// a separate signature (rather than making every caller take a 4th
// tuple element) so the 115+ existing callers in this codebase's own
// test suite that never use self-referential type aliases are
// unaffected. Note this means `parse`'s own caller has no way to reach
// the registry parse_with_named_types builds internally -- pairing this
// `parse` with plain `typecheck::check` (which defaults to an empty
// registry) on a program using a self-referential type alias is exactly
// the condition every Type::Named consumer's own missing-registry-entry
// fallback exists to handle gracefully; see coerce()'s own doc comment.
pub fn parse(src: &str) -> Result<(Arena, SpanMap, ExprRef), String> {
    let (arena, spans, root, _named_types) = parse_with_named_types(src)?;
    Ok((arena, spans, root))
}

// Same as `parse`, but ALSO returns the registry of genuinely
// self-referential type aliases this parse registered (Parser's own
// `named_types` field) -- needed by typecheck::check_with_named_types
// to resolve a Type::Named reference's own shape. Every ordinary
// caller that never uses self-referential type aliases keeps calling
// plain `parse` above, completely unaffected; this exists so the one
// real production entry point (lib.rs's own run_source_on_this_thread)
// can support the feature end to end without changing `parse`'s own
// signature and forcing every one of its 115+ existing callers in this
// codebase's own test suite to update for a capability they don't use.
pub fn parse_with_named_types(src: &str) -> Result<(Arena, SpanMap, ExprRef, HashMap<String, Type>), String> {
    let (tokens, tok_spans): (Vec<Token>, Vec<Span>) = tokenize(src)?.into_iter().unzip();
    let mut p = Parser {
        tokens,
        tok_spans,
        pos: 0,
        arena: Arena::new(),
        expr_spans: SpanMap::new(),
        src,
        type_aliases: HashMap::new(),
        named_types: HashMap::new(),
    };
    let root = p.expr()?;
    if p.pos != p.tokens.len() {
        let span = p.span_at();
        return Err(p.err_at(span, format!("trailing tokens after expression: {:?}", &p.tokens[p.pos..])));
    }
    Ok((p.arena, p.expr_spans, root, p.named_types))
}

// Test-only entry point: parses a single, standalone type expression
// (not a full program) -- used by the Vec(...) syntax tests (see the
// design spec's own §1/§2), which need to inspect a Type in isolation
// rather than run a whole program. Not part of the language's own real
// parsing path.
#[cfg(test)]
pub fn parse_type_string(src: &str) -> Result<Type, String> {
    let (tokens, tok_spans): (Vec<Token>, Vec<Span>) = tokenize(src)?.into_iter().unzip();
    let mut p = Parser {
        tokens,
        tok_spans,
        pos: 0,
        arena: Arena::new(),
        expr_spans: SpanMap::new(),
        src,
        type_aliases: HashMap::new(),
        named_types: HashMap::new(),
    };
    p.parse_fun_type()
}

struct Parser<'a> {
    tokens: Vec<Token>,
    tok_spans: Vec<Span>,
    pos: usize,
    arena: Arena,
    expr_spans: SpanMap,
    src: &'a str,
    // Type name -> the fully-resolved Type it stands for -- `type Name =
    // TypeExpr in body` (see atom's own `Some(Token::TypeKw)` arm), a pure
    // compile-time directive with no runtime effect of its own. Consulted
    // by parse_type's bare-identifier case: a name found here is
    // substituted in directly, fully resolved at the point of use, not
    // deferred. Scoped lexically: updated when a `type` binder is parsed,
    // restored (LIFO) once its own `in <body>` ends -- see atom's own
    // `type_alias_restore`.
    type_aliases: HashMap<String, Type>,
    // Every genuinely self-referential `type` alias registered so far
    // (Parser::atom's own `Token::TypeKw` arm) -- keyed by the SAME
    // gensym'd id its own Type::Named leaf carries, so it survives
    // past this parser's own scoped/LIFO-restored `type_aliases`
    // (which only remembers a NAME's CURRENT alias while parsing is
    // still inside that name's own scope). Unlike `type_aliases`, this
    // is never restored/popped -- once an alias is registered here, it
    // stays for the rest of parsing and is handed to whichever of
    // `parse`/`parse_with_named_types` the caller invoked.
    named_types: HashMap<String, Type>,
}

// Mints a globally-unique id for a `type` alias binder occurrence --
// same freshening idiom typecheck.rs's own fresh_row_name/
// fresh_type_name already use (a monotonic counter appended to a base
// name), just its own separate counter and its own separate namespace:
// this mechanism exists purely at parse time, before InferCtx (which
// owns the OTHER two) is even constructed.
fn fresh_named_type_id(base: &str) -> String {
    static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{base}#{n}")
}

// Does `ty` structurally contain a Type::Named leaf carrying exactly
// `id` anywhere inside it? Used right after parsing a `type` alias's
// own RHS to decide whether it actually referenced itself (mirrors the
// shape of typecheck.rs's own free_row_vars/free_type_vars_resolved --
// a plain structural walk, terminating because `ty` here is always a
// FRESH parse result, never something already containing a completed
// recursive reference of its own).
fn contains_named(ty: &Type, id: &str) -> bool {
    match ty {
        Type::Named(n) => n == id,
        Type::List(elem) => contains_named(elem, id),
        Type::Fun(param, _row, ret) => contains_named(param, id) || contains_named(ret, id),
        Type::Tuple(items) | Type::Union(items) => items.iter().any(|t| contains_named(t, id)),
        Type::Record(fields) => fields.iter().any(|(_, t)| contains_named(t, id)),
        Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) | Type::Var(_) => false,
        // Wraps one nested Type (the index is never itself a Type, so
        // has nothing to check here) -- same recurse-into-the-wrapped-
        // type precedent as List's own arm just above.
        Type::Indexed(wrapped, _) => contains_named(wrapped, id),
    }
}

// `contains_named`'s own occurrence-COUNTING analog, same recursive
// shape exactly (bool-returning `||`/`any` swapped for usize-returning
// `+`/`sum`) -- Case B's own eligibility needs to tell "exactly once"
// apart from "zero" or "two or more" (spec section 5, Case B point 2:
// a step alternative referencing the recursive id twice or more, e.g. a
// binary tree's `Node(Tree, Tree)` shape, deliberately doesn't qualify),
// which a plain bool can't distinguish. Terminates for the same
// structural reason `contains_named` does, though NOT because `ty` is
// always a fresh parse result here: Case B's own real caller
// (`qualifying_named_alternatives`, typecheck.rs) runs this on an
// ALREADY-REGISTERED alias's own RHS, which can (and typically does)
// contain completed `Type::Named(id)` leaves. It still terminates
// because neither `contains_named` nor `count_named` ever UNFOLDS a
// `Type::Named` reference -- both only count/check its occurrences as
// an opaque leaf in an already-fully-parsed type structure, so there's
// no recursion to follow through it at all.
//
// `pub(crate)`, not private: Phase 4 Task 3's own Case B eligibility
// check (typecheck.rs's `qualifying_named_alternatives`) needs to ask
// this question about an already-registered alias's own alternatives
// (spec section 5, Case B point 1) -- same progressive, only-as-far-
// as-needed visibility widening this project already uses repeatedly
// for `InferCtx`/`Scheme`/`Ctx`/etc. `contains_named` just above has no
// caller outside this file and stays private.
pub(crate) fn count_named(ty: &Type, id: &str) -> usize {
    match ty {
        Type::Named(n) => usize::from(n == id),
        Type::List(elem) => count_named(elem, id),
        Type::Fun(param, _row, ret) => count_named(param, id) + count_named(ret, id),
        Type::Tuple(items) | Type::Union(items) => items.iter().map(|t| count_named(t, id)).sum(),
        Type::Record(fields) => fields.iter().map(|(_, t)| count_named(t, id)).sum(),
        Type::Dyn | Type::Int | Type::Float | Type::Bool | Type::Str | Type::Token(_) | Type::Var(_) => 0,
        Type::Indexed(wrapped, _) => count_named(wrapped, id),
    }
}

// A `let`/`fun` prefix collected while flattening a chain of them (see
// `atom`) -- deferred until the terminal body is parsed, then folded back
// into nested Let/Lambda nodes in reverse.
enum PendingBinder {
    // `where_pred`, if present, is a gradual-verification refinement on
    // this binding (`let n: Int where n > 0 = ...`) -- see
    // desugar_refinement for how it's resolved at fold-back time (proven
    // outright when `val` is a literal, else an ordinary runtime check).
    // Not supported on `let rec` (see the `rec` check where `where` is
    // parsed) -- proving would need to reason about a recursive value,
    // which is well past what desugar_refinement's tiny evaluator attempts.
    Let { var: String, ann: Option<Type>, val: ExprRef, where_pred: Option<ExprRef> },
    // `let rec f = val_f [and g = val_g ...] in ...` -- one or more
    // simultaneously-recursive bindings folding back into a single
    // Expr::LetRec (never Expr::Let, which is never recursive).
    LetRec { bindings: Vec<(String, Option<Type>, ExprRef)> },
    // `where_pred` here is NEVER proven statically, even when `val` looks
    // like a literal at some call site -- a Lambda parameter's value is
    // whatever the CALLER passes, unknown at definition time, so this
    // always becomes a real runtime check.
    Fun { param: String, ann: Option<Type>, where_pred: Option<ExprRef> },
}

// A tiny, deliberately narrow compile-time evaluator for `where` refinement
// predicates (desugar_refinement): substitutes `subst` for every Var named
// `var_name` (the only variable a refinement can meaningfully reference --
// the value being refined) and tries to reduce to a literal Int, handling
// only Int literals, that one Var, and +, -, *, /, % over two such. None
// the moment anything else appears -- the caller falls back to a real
// runtime check rather than reject the predicate as unsupported syntax,
// the same "prove what's cheap, defer to runtime otherwise" stance as this
// checker's other analyses (typecheck's missing_case, dominates). A
// predicate can use any comparison (`<`/`<=`/`>`/`>=`/`==`/`!=`) and
// `&&`/`||`/`!` freely -- see try_eval_bool, which recognizes the `If`
// shape all of those desugar into.
fn try_eval_int(arena: &Arena, expr: ExprRef, var_name: &str, subst: i64) -> Option<i64> {
    match &arena[expr] {
        Expr::Int(n) => Some(*n),
        Expr::Var(name) if name == var_name => Some(subst),
        Expr::BinOp(op, l, r) => {
            let l = try_eval_int(arena, *l, var_name, subst)?;
            let r = try_eval_int(arena, *r, var_name, subst)?;
            match op {
                BinOp::Add => Some(l + r),
                BinOp::Sub => Some(l - r),
                BinOp::Mul => Some(l * r),
                BinOp::Div if r != 0 => Some(l / r),
                BinOp::Mod if r != 0 => Some(l % r),
                _ => None,
            }
        }
        _ => None,
    }
}

// Same idea, for evaluating a `let`'s OWN bound value expression -- no
// substitution, since nothing is in scope yet at that point (Int literals
// and +/-/*// over them only). Unary minus (`-n`, desugared to `0 - n` --
// see `unary`) falls out of this for free, so `-1` counts as a literal
// for proving purposes just like `1` does -- without this, "n = -1
// violates n > 0" would only be caught by a runtime check, not proven at
// parse time, since `-1` is never actually an Expr::Int node.
fn try_eval_closed_int(arena: &Arena, expr: ExprRef) -> Option<i64> {
    match &arena[expr] {
        Expr::Int(n) => Some(*n),
        Expr::BinOp(op, l, r) => {
            let l = try_eval_closed_int(arena, *l)?;
            let r = try_eval_closed_int(arena, *r)?;
            match op {
                BinOp::Add => Some(l + r),
                BinOp::Sub => Some(l - r),
                BinOp::Mul => Some(l * r),
                BinOp::Div if r != 0 => Some(l / r),
                BinOp::Mod if r != 0 => Some(l % r),
                _ => None,
            }
        }
        _ => None,
    }
}

// Same idea, for the predicate's own top-level Bool result: an Int
// comparison (`<`/`==`, via try_eval_int on both sides) or a bare Bool
// literal.
fn try_eval_bool(arena: &Arena, expr: ExprRef, var_name: &str, subst: i64) -> Option<bool> {
    match &arena[expr] {
        Expr::Bool(b) => Some(*b),
        Expr::BinOp(BinOp::Lt, l, r) => {
            Some(try_eval_int(arena, *l, var_name, subst)? < try_eval_int(arena, *r, var_name, subst)?)
        }
        Expr::BinOp(BinOp::Eq, l, r) => {
            Some(try_eval_int(arena, *l, var_name, subst)? == try_eval_int(arena, *r, var_name, subst)?)
        }
        // `&&`/`||`/`!` all desugar into exactly this If shape
        // (and_expr/or_expr/unary), so evaluating an If by trying its
        // condition first, then whichever branch that picks, handles all
        // three uniformly -- a refinement predicate like `0 < n && n <
        // 100` can still be proven, not just a bare single comparison.
        Expr::If(c, t, e) => {
            if try_eval_bool(arena, *c, var_name, subst)? {
                try_eval_bool(arena, *t, var_name, subst)
            } else {
                try_eval_bool(arena, *e, var_name, subst)
            }
        }
        _ => None,
    }
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    // Span of the token at the current (not yet consumed) position, or an
    // empty span at end-of-input if none remain.
    fn span_at(&self) -> Span {
        self.tok_spans.get(self.pos).copied().unwrap_or(Span { start: self.src.len(), end: self.src.len() })
    }

    // Span of the token bump() most recently consumed -- what an "expected
    // X, found Y" error is actually complaining about, and the natural
    // "end" position when closing off a multi-token construct.
    fn span_before(&self) -> Span {
        self.tok_spans
            .get(self.pos.saturating_sub(1))
            .copied()
            .unwrap_or(Span { start: self.src.len(), end: self.src.len() })
    }

    fn err_at(&self, span: Span, msg: String) -> String {
        span.format_error(self.src, &msg)
    }

    // Walks `e`'s whole expression tree looking for an Expr::Perform node
    // -- the static half of "guards may never perform" (see match's own
    // Token::Match arm). Conservative
    // on purpose, not exhaustively precise: this flags a `perform` even
    // inside an uncalled Lambda literal the guard would never actually
    // invoke, since telling "present in the text" from "actually reached
    // when the guard runs" needs real reachability analysis, not a plain
    // tree walk. ponytail: syntactic over-approximation, not reachability
    // analysis -- upgrade only if a real guard is rejected by this in
    // practice.
    fn contains_perform(arena: &Arena, e: ExprRef) -> bool {
        match &arena[e] {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Str(_) | Expr::Token(_) | Expr::Var(_) => false,
            Expr::Perform(_, _) => true,
            Expr::Tuple(items) | Expr::ListLit(items) => items.iter().any(|i| Self::contains_perform(arena, *i)),
            Expr::Record(fields) => fields.iter().any(|(_, v)| Self::contains_perform(arena, *v)),
            Expr::FieldAccess(target, _) | Expr::Check(target, _) => Self::contains_perform(arena, *target),
            Expr::Lambda(_, _, body) => Self::contains_perform(arena, *body),
            Expr::App(f, a) => Self::contains_perform(arena, *f) || Self::contains_perform(arena, *a),
            Expr::Let(_, _, val, body) => Self::contains_perform(arena, *val) || Self::contains_perform(arena, *body),
            Expr::LetRec(bindings, body) => {
                bindings.iter().any(|(_, _, v)| Self::contains_perform(arena, *v)) || Self::contains_perform(arena, *body)
            }
            Expr::BinOp(_, l, r) => Self::contains_perform(arena, *l) || Self::contains_perform(arena, *r),
            Expr::If(c, t, e) => {
                Self::contains_perform(arena, *c) || Self::contains_perform(arena, *t) || Self::contains_perform(arena, *e)
            }
            Expr::Handle { body, handler } => Self::contains_perform(arena, *body) || Self::contains_perform(arena, *handler),
            Expr::MakeHandler { body, .. } => Self::contains_perform(arena, *body),
            Expr::Match(scrutinee, arms) => {
                Self::contains_perform(arena, *scrutinee)
                    || arms.iter().any(|(_, guard, body)| {
                        guard.is_some_and(|g| Self::contains_perform(arena, g)) || Self::contains_perform(arena, *body)
                    })
            }
        }
    }

    // The only way an Expr node should ever be added to the arena --
    // keeps expr_spans in lockstep with it (same ExprRef, pushed in the
    // same call), which is what lets typecheck later look up any
    // original (pre-elaboration) node's source span with a plain index.
    fn push_spanned(&mut self, e: Expr, span: Span) -> ExprRef {
        let r = self.arena.push(e);
        let r2 = self.expr_spans.push(span);
        debug_assert_eq!(r, r2, "arena and expr_spans desynced -- an Expr was pushed without push_spanned");
        r
    }

    fn expect(&mut self, want: &Token) -> Result<(), String> {
        match self.bump() {
            Some(ref t) if t == want => Ok(()),
            other => Err(self.err_at(self.span_before(), format!("expected {want:?}, found {other:?}"))),
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Some(Token::Ident(s)) => Ok(s),
            other => Err(self.err_at(self.span_before(), format!("expected identifier, found {other:?}"))),
        }
    }

    // Parses "ident (":" parse_value)? ("," ident (":" parse_value)?)*"
    // up to (and consuming) the closing "}" -- the opening "{" is assumed
    // already consumed, same convention as every other bracketed form
    // here (e.g. atom_leaf's own LParen arm). Sorts the result by field
    // name and rejects a duplicate, so records are ALWAYS canonically
    // ordered from the moment they're parsed -- whether this is a record
    // TYPE, a construction, or a pattern, that's what lets `{y: 2, x: 1}`
    // and `{x: 1, y: 2}` mean the exact same thing with zero runtime
    // name-tracking (see types::Type::Record's own doc comment). Shared
    // by all three call sites so this bookkeeping -- and its error
    // message -- lives in exactly one place, instead of being hand-rolled
    // three times.
    //
    // `default`, when present, makes the ":" optional: a bare `x` field
    // puns to `default(self, "x")` (field-punning, for expression/pattern
    // construction: `{x, y}` means `{x: x, y: y}`). `None` makes every
    // field require an explicit ": T" -- used for record TYPES, which
    // have no value of their own to pun with.
    //
    // At least one field is required -- `{}` is a parse error, not a
    // zero-field record. This matches Tuple's own existing constraint,
    // not a new one: a record desugars into Expr::Tuple/Type::Tuple's
    // shape (see Type::Record's own doc comment), and Tuple has no
    // representation for zero elements either -- `()` is ordinary
    // grouping, and even the one-element case needs a trailing comma
    // (`(x,)`) specifically to stay distinct from grouping. `{}` has no
    // such ambiguity to resolve, so there's no reason to invent a
    // zero-arity case Tuple itself doesn't support.
    fn parse_record_fields<T>(
        &mut self,
        parse_value: fn(&mut Self) -> Result<T, String>,
        default: Option<fn(&mut Self, String) -> T>,
    ) -> Result<Vec<(String, T)>, String> {
        if matches!(self.peek(), Some(Token::RBrace)) {
            return Err(self.err_at(self.span_before(), "a record needs at least one field".to_string()));
        }
        let mut fields: Vec<(String, T)> = Vec::new();
        loop {
            let name = self.ident()?;
            // Checked right here, before consuming anything else -- a
            // duplicate is reported at the OFFENDING field's own name
            // token (self.span_before() is still that ident, nothing has
            // bumped since), not at the closing "}" a later post-hoc scan
            // would only reach after the whole record (and its span) had
            // already moved past every field.
            if find_field(&fields, &name).is_some() {
                return Err(self.err_at(self.span_before(), format!("field `{name}` given more than once")));
            }
            let value = if matches!(self.peek(), Some(Token::Colon)) {
                self.bump();
                parse_value(self)?
            } else {
                match default {
                    Some(d) => d(self, name.clone()),
                    None => {
                        return Err(self.err_at(
                            self.span_before(),
                            format!("field `{name}` needs a type: `{name}: T`"),
                        ))
                    }
                }
            };
            fields.push((name, value));
            if matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
                if matches!(self.peek(), Some(Token::RBrace)) {
                    break;
                }
            } else {
                break;
            }
        }
        self.expect(&Token::RBrace)?;
        // No further dedup pass needed -- the loop above already rejects
        // a repeated name the moment it's seen, so `fields` is guaranteed
        // unique here; only the canonical ordering is left to do.
        fields.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(fields)
    }

    // Optional `: Type` annotation, e.g. after a param name or a let binder.
    // parse_type_atom_union, NOT parse_union_type -- same "don't chain ->
    // at this level" reasoning parse_type's own doc comment gives: `fun x:
    // T -> body` must not let `T` swallow that `->` as its own (an
    // unparenthesized function type still needs explicit parens here,
    // `fun x: (A -> B) -> body`), and that restriction has to extend to
    // `|` too, or `fun x: A | B -> body` would ambiguously let `B`'s
    // alternative reach for the arrow the same way a bare function type
    // would.
    fn opt_annotation(&mut self) -> Result<Option<Type>, String> {
        if matches!(self.peek(), Some(Token::Colon)) {
            self.bump();
            Ok(Some(self.parse_type_atom_union()?))
        } else {
            Ok(None)
        }
    }

    // type_atom_union := type ("|" type)*  -- parse_type's own ATOM level
    // (no "->"), unioned. Used where a trailing "->" must NOT be consumed
    // as part of the type (opt_annotation -- see its own doc comment); a
    // function type still needs explicit parens there, same as before
    // union types existed, and now so does a function type as one union
    // alternative (`fun x: (A -> B) | C -> body`).
    fn parse_type_atom_union(&mut self) -> Result<Type, String> {
        let first = self.parse_type()?;
        if !matches!(self.peek(), Some(Token::Pipe)) {
            return Ok(first);
        }
        let mut alts = vec![first];
        while matches!(self.peek(), Some(Token::Pipe)) {
            self.bump();
            alts.push(self.parse_type()?);
        }
        Ok(Type::Union(Rc::new(alts)))
    }

    // union_type := fun_type ("|" fun_type)*  (left-associative in surface
    // syntax, though Type::Union itself is an unordered set -- order in
    // the Vec never matters to types::consistent). `|` binds LOOSER than
    // `->` (parse_fun_type's own right-recursion is greedy, so `A -> B | C`
    // parses as `(A -> B) | C`, not `A -> (B | C)`). Used everywhere a
    // trailing `->` couldn't be ambiguous with something else that follows
    // (unlike opt_annotation -- see parse_type_atom_union): inside `[...]`/
    // `(...)`, and a `type` alias's own RHS (terminated by `in`, not `->`/
    // `=`).
    fn parse_union_type(&mut self) -> Result<Type, String> {
        let first = self.parse_fun_type()?;
        if !matches!(self.peek(), Some(Token::Pipe)) {
            return Ok(first);
        }
        let mut alts = vec![first];
        while matches!(self.peek(), Some(Token::Pipe)) {
            self.bump();
            alts.push(self.parse_fun_type()?);
        }
        Ok(Type::Union(Rc::new(alts)))
    }

    // Bare, atom-only: "Int" | "Bool" | "Str" | "Dyn" | "[" fun_type "]" |
    // "(" fun_type ")". Deliberately does NOT chain "->" at this level --
    // an annotation site (`fun x: T ->`, `let x: T =`) is always
    // immediately followed by its own "->"/"=" token, so a bare trailing
    // arrow here would be ambiguous between "this type continues" and "the
    // annotation just ended". A function type must be parenthesized to
    // disambiguate: `fun f: (Int -> Int) -> ...`. "[" / "]" don't have that
    // ambiguity (nothing else starts with "["), so a list element type can
    // freely be a function type without extra parens: `[Int -> Int]`.
    fn parse_type(&mut self) -> Result<Type, String> {
        match self.bump() {
            Some(Token::LBracket) => {
                let elem = self.parse_fun_type()?;
                self.expect(&Token::RBracket)?;
                Ok(Type::List(Rc::new(elem)))
            }
            // `(T)` stays ordinary grouping; `(T, T, ...)` (a comma
            // present) is a tuple TYPE -- Type::Tuple, matching Expr::
            // Tuple's own comma-means-tuple rule on the expression side.
            // Trailing comma allowed (needed to write a single-element
            // tuple type `(T,)` at all, same reason as expressions).
            Some(Token::LParen) => {
                let first = self.parse_union_type()?;
                if matches!(self.peek(), Some(Token::Comma)) {
                    let mut items = vec![first];
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.bump();
                        if matches!(self.peek(), Some(Token::RParen)) {
                            break;
                        }
                        items.push(self.parse_union_type()?);
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Type::Tuple(Rc::new(items)))
                } else {
                    self.expect(&Token::RParen)?;
                    Ok(first)
                }
            }
            // "{" ident ":" type ("," ident ":" type)* "}"  -- see
            // Type::Record's own doc comment for why every field needs an
            // explicit ": T" here (no punning at the type level, there's
            // no value to pun with) and why the result comes back sorted.
            Some(Token::LBrace) => {
                let fields = self.parse_record_fields(Self::parse_union_type, None)?;
                Ok(Type::Record(Rc::new(fields)))
            }
            // A capitalized name: either one of the four builtin type
            // names (Int/Bool/Str/Dyn -- plain Idents, not reserved
            // keywords, so they stay available as ordinary variable names
            // everywhere outside a type position) or a reference to a
            // `type Name = ... in` alias (Parser::type_aliases) currently
            // in scope -- fully resolved right here, not deferred. Unlike
            // `data`'s own former bare-Data(name) fallback, an unknown
            // name is a static error immediately: there's no "maybe it
            // resolves later" story left once every type must be a
            // builtin, a Token/Tuple/Union, or a named alias.
            Some(Token::Ident(name)) if name.chars().next().is_some_and(char::is_uppercase) => {
                if name == "Vec" && matches!(self.peek(), Some(Token::LParen)) {
                    self.bump(); // consume '('
                    let index = self.parse_index_expr()?;
                    self.expect(&Token::RParen)?;
                    return Ok(Type::Indexed(Rc::new(Type::List(Rc::new(Type::Dyn))), Rc::new(index)));
                }
                // Case B syntax generalization (design spec 2026-09-20
                // sec 3): the SAME indexing syntax works for any other
                // alias currently in scope -- no eligibility check here,
                // that's Phase 4's existing pattern-matching-side
                // qualifying_named_alternatives, unchanged. Zero syntax
                // conflicts: nothing in this grammar expects '(' right
                // after a bare alias reference, since parse_type's own
                // Ident arm otherwise returns immediately once resolved.
                if matches!(self.peek(), Some(Token::LParen)) {
                    if let Some(resolved) = self.type_aliases.get(&name) {
                        let resolved_clone = resolved.clone();
                        self.bump();
                        let index = self.parse_index_expr()?;
                        self.expect(&Token::RParen)?;
                        return Ok(Type::Indexed(Rc::new(resolved_clone), Rc::new(index)));
                    }
                }
                match Self::builtin_type(&name) {
                    Some(ty) => Ok(ty),
                    None => match self.type_aliases.get(&name) {
                        Some(ty) => Ok(ty.clone()),
                        None => Err(self.err_at(self.span_before(), format!("unknown type: {name}"))),
                    },
                }
            }
            other => Err(self.err_at(self.span_before(), format!("expected a type, found {other:?}"))),
        }
    }

    // Index-expression grammar only: Var | Lit | + | - | * -- see the
    // spec's own §1 for why this deliberately doesn't reuse the full
    // expression parser (embedding arbitrary expressions in type
    // position would reopen the divergent-typechecker risk the design
    // explicitly rejected). Standard two-level precedence: `+`/`-` loosest,
    // left-associative; `*` tighter, left-associative; parenthesized
    // sub-expressions and atoms (a variable name or an integer literal)
    // at the bottom.
    fn parse_index_expr(&mut self) -> Result<IndexExpr, String> {
        let mut lhs = self.parse_index_term()?;
        loop {
            match self.peek() {
                Some(Token::Plus) => {
                    self.bump();
                    let rhs = self.parse_index_term()?;
                    lhs = IndexExpr::Add(Rc::new(lhs), Rc::new(rhs));
                }
                Some(Token::Minus) => {
                    self.bump();
                    let rhs = self.parse_index_term()?;
                    lhs = IndexExpr::Sub(Rc::new(lhs), Rc::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_index_term(&mut self) -> Result<IndexExpr, String> {
        let mut lhs = self.parse_index_atom()?;
        while matches!(self.peek(), Some(Token::Star)) {
            self.bump();
            let rhs = self.parse_index_atom()?;
            lhs = IndexExpr::Mul(Rc::new(lhs), Rc::new(rhs));
        }
        Ok(lhs)
    }

    fn parse_index_atom(&mut self) -> Result<IndexExpr, String> {
        match self.bump() {
            Some(Token::Int(n)) => Ok(IndexExpr::Lit(n)),
            Some(Token::Ident(name)) => Ok(IndexExpr::Var(name)),
            Some(Token::LParen) => {
                let inner = self.parse_index_expr()?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            other => Err(self.err_at(self.span_before(), format!("expected an index expression, found {other:?}"))),
        }
    }

    // The five builtin type names. Not reserved lexer keywords (unlike
    // If/Let/Fun/...) since they're only ever meaningful inside a type
    // position -- resolved contextually right here instead, which is what
    // keeps them available as ordinary identifiers (`let Int = 5 in ...`)
    // everywhere else. Shared between parse_type's own lookup and the
    // `type Name = ...` alias binder's collision guard just below, so the
    // two can't drift out of sync.
    fn builtin_type(name: &str) -> Option<Type> {
        match name {
            "Int" => Some(Type::Int),
            "Float" => Some(Type::Float),
            "Bool" => Some(Type::Bool),
            "Str" => Some(Type::Str),
            "Dyn" => Some(Type::Dyn),
            _ => None,
        }
    }

    // True iff the tokens right after the CURRENT position (not yet
    // consumed) are exactly "{" ident "}" -- the row-variable shape
    // (`->{e}`), which parse_fun_type must tell apart from a record TYPE
    // that also happens to start with "{" right where a return type goes
    // (`-> {x: Int}`). The two are distinguishable one token further in:
    // a row variable's "{" is followed by exactly one ident then an
    // IMMEDIATE "}", while a record type's ident is always followed by
    // ":" (record types have no punning -- see parse_record_fields' own
    // doc comment). Two-token lookahead is safe here since `tokens` is a
    // plain pre-lexed Vec, not a stream.
    fn peek_is_row_var(&self) -> bool {
        matches!(self.tokens.get(self.pos + 1), Some(Token::Ident(_)))
            && matches!(self.tokens.get(self.pos + 2), Some(Token::RBrace))
    }

    // fun_type := type ("->" ("{" ident "}")? fun_type)?  (right-assoc) --
    // only reachable from inside parens, where ")" unambiguously ends it.
    // The optional `{name}` after "->" names a row variable for row
    // polymorphism (typecheck::extend_generalized generalizes it at a
    // `let`, typecheck::lookup instantiates a fresh copy at each use). No
    // `{name}` -- the default, and the only option before this existed --
    // means EffectRow::Dyn (unknown effects, gradual default), consistent
    // with every other unannotated position. A "{" that ISN'T a row
    // variable (peek_is_row_var says no) falls through to Dyn here and
    // gets parsed as an ordinary return-type atom instead, by the
    // recursive parse_fun_type -> parse_type call below -- which is
    // exactly how a record type (`-> {x: Int}`) reaches parse_type's own
    // LBrace arm rather than being swallowed as a malformed row variable.
    fn parse_fun_type(&mut self) -> Result<Type, String> {
        let atom = self.parse_type()?;
        if matches!(self.peek(), Some(Token::Arrow)) {
            self.bump();
            let row = if matches!(self.peek(), Some(Token::LBrace)) && self.peek_is_row_var() {
                self.bump();
                let name = self.ident()?;
                self.expect(&Token::RBrace)?;
                EffectRow::Var(name)
            } else {
                EffectRow::Dyn
            };
            let ret = self.parse_fun_type()?;
            Ok(Type::Fun(Rc::new(atom), row, Rc::new(ret)))
        } else {
            Ok(atom)
        }
    }

    // pattern := pattern_atom ("::" pattern)?  (right-assoc, `h :: t`)
    fn pattern(&mut self) -> Result<Pattern, String> {
        let head = self.pattern_atom()?;
        if matches!(self.peek(), Some(Token::ColonColon)) {
            self.bump();
            let tail = self.pattern()?;
            Ok(Pattern::Cons(Box::new(head), Box::new(tail)))
        } else {
            Ok(head)
        }
    }

    // pattern_atom := Int | true | false | Str | ident
    //               | "[" (pattern ("," pattern)*)? "]"
    //               | "(" pattern ("," pattern)* ","? ")"
    fn pattern_atom(&mut self) -> Result<Pattern, String> {
        match self.bump() {
            Some(Token::Int(n)) => Ok(Pattern::Int(n)),
            Some(Token::True) => Ok(Pattern::Bool(true)),
            Some(Token::False) => Ok(Pattern::Bool(false)),
            Some(Token::Str(s)) => Ok(Pattern::Str(s)),
            // `(p)` stays ordinary grouping (unchanged); `(p, p, ...)` (a
            // comma present) destructures a tuple -- tuples are plain
            // Value::List at runtime (see Expr::Tuple's own doc comment),
            // so this is just sugar for the same Pattern::List `[p, p,
            // ...]` already matches lists with.
            Some(Token::LParen) => {
                let first = self.pattern()?;
                if matches!(self.peek(), Some(Token::Comma)) {
                    let mut items = vec![first];
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.bump();
                        // Trailing comma allowed -- without it a single-
                        // element tuple pattern `(p,)` could never close.
                        if matches!(self.peek(), Some(Token::RParen)) {
                            break;
                        }
                        items.push(self.pattern()?);
                    }
                    self.expect(&Token::RParen)?;
                    Ok(Pattern::List(items))
                } else {
                    self.expect(&Token::RParen)?;
                    Ok(first)
                }
            }
            // "{" ident (":" pattern)? ("," ident (":" pattern)?)* "}"  --
            // record destructure, a REAL Pattern::Record (not sugar over
            // Pattern::List -- see its own doc comment for why: width-
            // tolerant, name-based matching needs the names kept, not
            // dropped). A bare `x` field puns to `x: x` (see
            // parse_record_fields' own doc comment).
            Some(Token::LBrace) => {
                // `{x, y}` means `{x: x, y: y}` in pattern position too --
                // no arena/span bookkeeping needed (Pattern is a plain
                // inline enum, not arena-indexed) -- takes `&mut Self`
                // only because that's parse_record_fields' shared
                // `default` parameter shape.
                let fields =
                    self.parse_record_fields(Self::pattern, Some(|_: &mut Self, name| Pattern::Var(name)))?;
                Ok(Pattern::Record(fields))
            }
            // No case distinction: with `data`-declared constructors gone,
            // any identifier here -- upper or lowercase, "_" included --
            // just binds. A hand-rolled tagged value (`(opaque, x)`) is
            // matched positionally instead, via the tuple/list forms
            // above -- see the pin-pattern discussion for the still-open
            // "match against an already-bound value" gap that leaves.
            Some(Token::Ident(name)) => Ok(Pattern::Var(name)),
            Some(Token::LBracket) => {
                let mut items = Vec::new();
                if !matches!(self.peek(), Some(Token::RBracket)) {
                    items.push(self.pattern()?);
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.bump();
                        items.push(self.pattern()?);
                    }
                }
                self.expect(&Token::RBracket)?;
                Ok(Pattern::List(items))
            }
            other => Err(self.err_at(self.span_before(), format!("expected a pattern, found {other:?}"))),
        }
    }

    // expr := or_expr
    fn expr(&mut self) -> Result<ExprRef, String> {
        self.or_expr()
    }

    // or_expr := and_expr ("||" and_expr)*  (left-associative, loosest of
    // the boolean/comparison operators -- standard placement, `&&` binds
    // tighter). Desugars into `if lhs then true else rhs` -- reusing If's
    // existing lazy-branch semantics for SHORT-CIRCUITING (`rhs` isn't
    // evaluated when `lhs` is already true), rather than a new BinOp
    // (which would evaluate both sides eagerly -- wrong wherever `rhs`
    // performs an effect or diverges).
    fn or_expr(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let mut lhs = self.and_expr()?;
        while matches!(self.peek(), Some(Token::PipePipe)) {
            self.bump();
            let rhs = self.and_expr()?;
            let span = Span { start, end: self.span_before().end };
            let true_lit = self.push_spanned(Expr::Bool(true), span);
            lhs = self.push_spanned(Expr::If(lhs, true_lit, rhs), span);
        }
        Ok(lhs)
    }

    // and_expr := cmp ("&&" cmp)*  (left-associative). Same short-
    // circuiting reasoning as or_expr: `if lhs then rhs else false`.
    fn and_expr(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let mut lhs = self.cmp()?;
        while matches!(self.peek(), Some(Token::AmpAmp)) {
            self.bump();
            let rhs = self.cmp()?;
            let span = Span { start, end: self.span_before().end };
            let false_lit = self.push_spanned(Expr::Bool(false), span);
            lhs = self.push_spanned(Expr::If(lhs, rhs, false_lit), span);
        }
        Ok(lhs)
    }

    // cmp := cons (("==" | "<") cons)?  -- non-associative, one comparison
    fn cmp(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let lhs = self.cons()?;
        let tok = match self.peek() {
            Some(Token::EqEq | Token::Lt | Token::Gt | Token::LtEq | Token::GtEq | Token::BangEq) => {
                self.peek().cloned()
            }
            _ => None,
        };
        match tok {
            Some(tok) => {
                self.bump();
                let rhs = self.cons()?;
                let span = Span { start, end: self.span_before().end };
                // Only `==`/`<` are real BinOps -- `>`/`<=`/`>=`/`!=` are
                // sugar over them (flipped operands, or negated), same
                // "reuse what exists" approach as `&&`/`||`/`!`
                // desugaring into If rather than adding new opcodes.
                let node = match tok {
                    Token::EqEq => self.push_spanned(Expr::BinOp(BinOp::Eq, lhs, rhs), span),
                    Token::Lt => self.push_spanned(Expr::BinOp(BinOp::Lt, lhs, rhs), span),
                    // a > b  ==  b < a
                    Token::Gt => self.push_spanned(Expr::BinOp(BinOp::Lt, rhs, lhs), span),
                    // a <= b  ==  !(b < a)
                    Token::LtEq => {
                        let lt = self.push_spanned(Expr::BinOp(BinOp::Lt, rhs, lhs), span);
                        self.negate(lt, span)
                    }
                    // a >= b  ==  !(a < b)
                    Token::GtEq => {
                        let lt = self.push_spanned(Expr::BinOp(BinOp::Lt, lhs, rhs), span);
                        self.negate(lt, span)
                    }
                    // a != b  ==  !(a == b)
                    Token::BangEq => {
                        let eq = self.push_spanned(Expr::BinOp(BinOp::Eq, lhs, rhs), span);
                        self.negate(eq, span)
                    }
                    _ => unreachable!(),
                };
                Ok(node)
            }
            None => Ok(lhs),
        }
    }

    // `if operand then false else true` -- the same desugaring `!` uses in
    // `unary`, shared here so `<=`/`>=`/`!=` (each "not the flipped/direct
    // comparison") don't duplicate it.
    fn negate(&mut self, operand: ExprRef, span: Span) -> ExprRef {
        let f = self.push_spanned(Expr::Bool(false), span);
        let t = self.push_spanned(Expr::Bool(true), span);
        self.push_spanned(Expr::If(operand, f, t), span)
    }

    // cons := add ("::" cons)?  (right-associative, via right recursion
    // rather than a loop, so `1 :: 2 :: [3]` parses as `1 :: (2 :: [3])`
    // -- the same shape a chain of cons PATTERNS already builds. Binds
    // looser than "+"/"-"/"*"/"/"/"++" (so `1 + 2 :: xs` is `(1 + 2) ::
    // xs`) and tighter than "=="/"<", matching the usual OCaml/Haskell
    // placement for list cons.
    fn cons(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let lhs = self.add()?;
        if matches!(self.peek(), Some(Token::ColonColon)) {
            self.bump();
            let rhs = self.cons()?;
            let span = Span { start, end: self.span_before().end };
            Ok(self.push_spanned(Expr::BinOp(BinOp::Cons, lhs, rhs), span))
        } else {
            Ok(lhs)
        }
    }

    // add := mul (("+" | "-" | "++") mul)*  (left-associative)
    fn add(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let mut lhs = self.mul()?;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => BinOp::Add,
                Some(Token::Minus) => BinOp::Sub,
                Some(Token::PlusPlus) => BinOp::Concat,
                _ => break,
            };
            self.bump();
            let rhs = self.mul()?;
            let span = Span { start, end: self.span_before().end };
            lhs = self.push_spanned(Expr::BinOp(op, lhs, rhs), span);
        }
        Ok(lhs)
    }

    // mul := unary (("*" | "/") unary)*  (left-associative, binds tighter
    // than "+"/"-"/"++" -- `1 + 2 * 3` is `1 + (2 * 3)`)
    fn mul(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let mut lhs = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Token::Star) => BinOp::Mul,
                Some(Token::Slash) => BinOp::Div,
                Some(Token::Percent) => BinOp::Mod,
                _ => break,
            };
            self.bump();
            let rhs = self.unary()?;
            let span = Span { start, end: self.span_before().end };
            lhs = self.push_spanned(Expr::BinOp(op, lhs, rhs), span);
        }
        Ok(lhs)
    }

    // unary := ("-" | "!")? postfix -- prefix negation and boolean not,
    // both desugared at parse time rather than new AST nodes: "-x" into
    // `0 - x` (Sub already exists, and this is its only user), "!x" into
    // `if x then false else true` (reuses If, same as and_expr/or_expr --
    // there's no eagerness concern for a UNARY operator the way there is
    // for &&/||, but reusing If still means zero new Expr/Value/machine.rs
    // surface, consistent with how the rest of this parser prefers sugar
    // over new primitives). Right-recursive (`unary` not `postfix` on the
    // operand) so `- -x`/`!!x` parse too, for whatever that's worth.
    fn unary(&mut self) -> Result<ExprRef, String> {
        if matches!(self.peek(), Some(Token::Minus)) {
            let start = self.span_at().start;
            self.bump();
            let operand = self.unary()?;
            let span = Span { start, end: self.span_before().end };
            let zero = self.push_spanned(Expr::Int(0), span);
            Ok(self.push_spanned(Expr::BinOp(BinOp::Sub, zero, operand), span))
        } else if matches!(self.peek(), Some(Token::Bang)) {
            let start = self.span_at().start;
            self.bump();
            let operand = self.unary()?;
            let span = Span { start, end: self.span_before().end };
            Ok(self.negate(operand, span))
        } else {
            self.postfix()
        }
    }

    // postfix := atom (("." ident) | atom)*  -- curried calls, either by
    // juxtaposition (f a b) or explicit parens (f(a)(b) -- unaffected:
    // "(" is itself an atom-starting token, so parsing the parenthesized
    // expression as the argument atom produces the identical App chain
    // either way), and field access p.x. Both are peers in this ONE
    // left-to-right loop over the accumulating `e`: `f(a).x` means
    // `(f(a)).x` (dot binds to whatever the chain has built so far, not
    // to the argument atom alone), and correspondingly `f x.y` means
    // `(f x).y`, not `f(x.y)`.
    //
    // The juxtaposed-argument case parses just ONE atom (self.atom(), not
    // self.expr()) for the same reason application binds tighter than
    // every operator in ML/Haskell: `f a + b` must mean `(f a) + b`, not
    // `f (a + b)`. Since unary `-` sits ABOVE postfix in this precedence
    // chain (mul -> unary -> postfix), a bare `-` is never an atom-
    // starting token here -- `f -1` therefore still parses as `f - 1`
    // (Sub), matching Haskell/OCaml's own resolution of this exact
    // ambiguity, but for free: nothing here special-cases it, the
    // existing precedence already forces it. Negating an argument still
    // needs explicit parens: `f (-1)`.
    fn postfix(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let mut e = self.atom()?;
        loop {
            match self.peek() {
                Some(tok) if Self::starts_juxtaposed_arg(tok) => {
                    let arg = self.atom()?;
                    let span = Span { start, end: self.span_before().end };
                    e = self.push_spanned(Expr::App(e, arg), span);
                }
                Some(Token::Dot) => {
                    self.bump();
                    let name = self.ident()?;
                    let span = Span { start, end: self.span_before().end };
                    e = self.push_spanned(Expr::FieldAccess(e, name), span);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    // NOT the full First(atom) set -- deliberately narrower. `atom`/
    // `atom_leaf` also start on If/Match/Let/Fun/Handle/HandlerKw, but
    // every one of those ends in an UNBOUNDED `self.expr()` for its
    // tail position (an `if`'s `else` branch, a `let`'s body, a `match`
    // arm's body, a `handle`'s handler expression, ...) with no closing
    // delimiter of its own -- so as a bare JUXTAPOSED argument, that tail
    // would silently swallow every subsequent argument meant for the
    // OUTER application instead of stopping at just one atom. Concretely,
    // if these were included, `f let x = 1 in x 2` would parse as
    // `App(f, (let x = 1 in (x 2)))` -- a single argument -- not
    // `App(App(f, let x=1 in x), 2)` as "any atom-starting token is an
    // argument boundary" would suggest; same story for `f if c then a
    // else b 2`, `f match ... 2`, `f handle ... 2`. Excluding them here
    // means that ambiguity is a parse error (trailing tokens) instead of
    // a silent wrong grouping -- passing one of these as an argument
    // still works, just needs explicit parens: `f (let x = 1 in x) 2`.
    // They're still valid as postfix's OWN leading atom (the unrestricted
    // `self.atom()` call before this loop even starts), and still valid
    // as an explicitly-parenthesized argument (LParen stays included; the
    // parens are exactly what makes the tail bounded again).
    fn starts_juxtaposed_arg(tok: &Token) -> bool {
        matches!(
            tok,
            Token::Int(_)
                | Token::Float(_)
                | Token::True
                | Token::False
                | Token::Str(_)
                | Token::Ident(_)
                | Token::LBracket
                | Token::LParen
                | Token::Perform
        )
    }

    // Peels off a run of leading `let ... in` / `fun ... ->` prefixes
    // iteratively -- a token peek per iteration, not a recursive call --
    // so a long chain of either of them costs O(1) native
    // stack instead of O(chain length). That chain shape is exactly what
    // used to overflow the stack on deeply nested/generated source (see
    // lib.rs's run_source). The terminal body/value once the chain ends is
    // parsed with an ordinary self.expr() call, same as the original
    // recursive version -- only the "is there another prefix" bookkeeping
    // moved out of the call stack, not the grammar itself.
    fn atom(&mut self) -> Result<ExprRef, String> {
        let mut pending: Vec<(usize, PendingBinder)> = Vec::new();
        // (name, its OLD value before this `type` binder touched it) --
        // restored, in reverse, at the end of this atom() call, so a
        // `type` binder's effect on `type_aliases` is scoped to its own
        // `in <body>` the same way its Env binding already is, not left
        // dangling for the rest of the file.
        let mut type_alias_restore: Vec<(String, Option<Type>)> = Vec::new();
        // `type` pushes NOTHING to `pending` (no AST node to fold back --
        // see the arm below), so `pending.is_empty()` alone can't tell the
        // terminal-parsing branch below "a prefix WAS seen" the way it can
        // for Let/Fun/Data (which always push one). Without this, `type X
        // = ... in BODY` where BODY starts with anything other than
        // Let/Fun/Data/TypeKw would parse BODY as a single atom_leaf()
        // atom instead of a full self.expr() -- meaning `type_aliases`
        // gets restored (see the end of this function) the moment that
        // ONE atom finishes, before any LATER sibling atom in the same
        // expression (a juxtaposed argument, say) gets a chance to see the
        // alias still in scope.
        let mut saw_type_alias = false;
        loop {
            match self.peek() {
                Some(Token::TypeKw) => {
                    // `type Name = TypeExpr in body` -- a pure compile-time
                    // directive: no wrapped AST node, no PendingBinder
                    // entry (nothing to fold back), just a scoped update to
                    // `type_aliases` before continuing this same peeling
                    // loop (parsing the next prefix, or the terminal body).
                    self.bump();
                    saw_type_alias = true;
                    let name = self.ident()?;
                    if Self::builtin_type(&name).is_some() || name == "Vec" {
                        // Int/Bool/Str/Dyn resolve contextually in
                        // parse_type rather than being reserved lexer
                        // keywords (see builtin_type's own doc comment) --
                        // that frees them as plain variable names, but an
                        // alias silently shadowing one would mean a later
                        // `Int` in type position resolves to the OLD
                        // builtin no matter what this alias says (parse_
                        // type checks builtin_type before type_aliases),
                        // silently discarding the alias. Reject it here
                        // instead of leaving that footgun undiagnosed.
                        // `Vec` gets the same treatment even though it's
                        // not a `builtin_type`: parse_type's `Vec(` check
                        // (above) runs before the type_aliases lookup too,
                        // so a `Vec` alias would be silently discarded the
                        // same way whenever parens follow it.
                        return Err(self.err_at(
                            self.span_before(),
                            format!("cannot redefine builtin type {name} as an alias"),
                        ));
                    }
                    self.expect(&Token::Equals)?;
                    // Insert a placeholder BEFORE parsing the RHS, keyed
                    // to a fresh id -- any self-reference inside the RHS
                    // resolves through the SAME, already-existing
                    // alias-lookup path (parse_type's own
                    // `self.type_aliases.get(&name)` arm) with no new
                    // grammar needed. Captured here (before the insert),
                    // not after parsing the RHS, so an OUTER scope's own
                    // same-named alias (if any) is what gets restored,
                    // exactly as the pre-existing scoping logic already
                    // requires.
                    let previous = self.type_aliases.get(&name).cloned();
                    let fresh_id = fresh_named_type_id(&name);
                    self.type_aliases.insert(name.clone(), Type::Named(fresh_id.clone()));
                    let ty = self.parse_union_type()?;
                    self.expect(&Token::In)?;
                    // Did the RHS actually reference itself? If not,
                    // this is an ordinary, non-recursive alias --
                    // discard the placeholder, store the real parsed
                    // type exactly as this alias mechanism already did
                    // before this whole feature existed. If so,
                    // register the definition (which may itself
                    // structurally contain this SAME Type::Named leaf,
                    // nested wherever the self-reference occurred) and
                    // make every future use of `name` resolve to the
                    // lightweight reference instead of the (impossible
                    // to fully construct) expansion.
                    let final_ty = if contains_named(&ty, &fresh_id) {
                        self.named_types.insert(fresh_id.clone(), ty);
                        Type::Named(fresh_id)
                    } else {
                        ty
                    };
                    type_alias_restore.push((name.clone(), previous));
                    self.type_aliases.insert(name, final_ty);
                }
                Some(Token::Let) => {
                    let start = self.span_at().start;
                    self.bump();
                    let rec = matches!(self.peek(), Some(Token::Rec));
                    if rec {
                        self.bump();
                    }
                    // `and` only continues a `rec` group -- a plain `let`
                    // is always exactly one binding.
                    let mut bindings = Vec::new();
                    let mut where_pred = None;
                    loop {
                        let var = self.ident()?;
                        let ann = self.opt_annotation()?;
                        if matches!(self.peek(), Some(Token::Where)) {
                            if rec {
                                let span = self.span_at();
                                return Err(self.err_at(
                                    span,
                                    "`where` refinements aren't supported on `let rec` bindings".to_string(),
                                ));
                            }
                            self.bump();
                            where_pred = Some(self.expr()?);
                        }
                        self.expect(&Token::Equals)?;
                        let val = self.expr()?;
                        bindings.push((var, ann, val));
                        if rec && matches!(self.peek(), Some(Token::And)) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    self.expect(&Token::In)?;
                    if rec {
                        pending.push((start, PendingBinder::LetRec { bindings }));
                    } else {
                        let (var, ann, val) = bindings.into_iter().next().unwrap();
                        pending.push((start, PendingBinder::Let { var, ann, val, where_pred }));
                    }
                }
                Some(Token::Fun) => {
                    let start = self.span_at().start;
                    self.bump();
                    let param = self.ident()?;
                    let ann = self.opt_annotation()?;
                    let where_pred = if matches!(self.peek(), Some(Token::Where)) {
                        self.bump();
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    self.expect(&Token::Arrow)?;
                    pending.push((start, PendingBinder::Fun { param, ann, where_pred }));
                }
                _ => break,
            }
        }

        let mut result = if pending.is_empty() && !saw_type_alias { self.atom_leaf()? } else { self.expr()? };
        // Every wrapping binder shares this same END position (the
        // terminal body's own end) -- only its START differs (where its
        // own `let`/`fun` keyword began). `let x = 1 in let y = 2 in
        // body`'s outer Let spans `[first "let", end of body]`; the inner
        // one spans `[second "let", end of body]`.
        let end = self.expr_spans[result].end;
        for (start, binder) in pending.into_iter().rev() {
            let span = Span { start, end };
            result = match binder {
                PendingBinder::Let { var, ann, val, where_pred } => {
                    let body = match where_pred {
                        None => result,
                        Some(pred) => self.desugar_refinement(&var, pred, val, result, span)?,
                    };
                    self.push_spanned(Expr::Let(var, ann, val, body), span)
                }
                PendingBinder::LetRec { bindings } => {
                    self.push_spanned(Expr::LetRec(Rc::new(bindings), result), span)
                }
                PendingBinder::Fun { param, ann, where_pred } => {
                    // Never proven statically here -- see PendingBinder::Fun's
                    // own doc comment on why a parameter's value can't be.
                    let body = match where_pred {
                        None => result,
                        Some(pred) => self.wrap_runtime_check(pred, result, span),
                    };
                    self.push_spanned(Expr::Lambda(param, ann, body), span)
                }
            };
        }
        for (name, old) in type_alias_restore.into_iter().rev() {
            match old {
                Some(ty) => {
                    self.type_aliases.insert(name, ty);
                }
                None => {
                    self.type_aliases.remove(&name);
                }
            }
        }
        Ok(result)
    }

    // `let name: T where pred = val in body` -- "gradual verification":
    // a refinement predicate that's PROVEN outright when it cheaply can
    // be (skipping the runtime check entirely -- zero overhead, the
    // actual payoff of doing this gradually rather than as a bare
    // assert), and falls back to an ordinary runtime check otherwise.
    // Proving is only ever attempted when `val` reduces to a closed Int
    // constant (try_eval_closed_int -- a Lambda parameter's ACTUAL value,
    // or any expression that isn't just literals and arithmetic, isn't
    // known until runtime; see PendingBinder::Fun's own doc comment for
    // the parameter case), and only for the narrow predicate shapes
    // try_eval_bool understands.
    // A predicate that evaluates to PROVABLY FALSE is a parse-time error:
    // the program could never have satisfied it, so there is no runtime
    // to defer to.
    fn desugar_refinement(
        &mut self,
        name: &str,
        pred: ExprRef,
        val: ExprRef,
        body: ExprRef,
        span: Span,
    ) -> Result<ExprRef, String> {
        if let Some(n) = try_eval_closed_int(&self.arena, val) {
            if let Some(proven) = try_eval_bool(&self.arena, pred, name, n) {
                return if proven {
                    Ok(body)
                } else {
                    Err(self.err_at(span, format!("refinement violated: `{name}` = {n} does not satisfy the `where` clause")))
                };
            }
        }
        Ok(self.wrap_runtime_check(pred, body, span))
    }

    // `if pred then body else fail("...")`, built entirely from existing
    // Expr nodes -- no new Value representation or machine.rs opcode
    // needed for a refinement that can't be proven at parse time.
    fn wrap_runtime_check(&mut self, pred: ExprRef, body: ExprRef, span: Span) -> ExprRef {
        let msg = self.push_spanned(Expr::Str("refinement violated".to_string()), span);
        let fail_var = self.push_spanned(Expr::Var("#fail".to_string()), span);
        let fail_call = self.push_spanned(Expr::App(fail_var, msg), span);
        self.push_spanned(Expr::If(pred, body, fail_call), span)
    }

    // Every atom form except `let`/`fun`/`type`, which `atom` handles
    // iteratively above. Reached only once no more chain prefix remains.
    fn atom_leaf(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        match self.bump() {
            Some(Token::Int(n)) => Ok(self.push_spanned(Expr::Int(n), Span { start, end: self.span_before().end })),
            Some(Token::Float(x)) => Ok(self.push_spanned(Expr::Float(x), Span { start, end: self.span_before().end })),
            Some(Token::True) => {
                Ok(self.push_spanned(Expr::Bool(true), Span { start, end: self.span_before().end }))
            }
            Some(Token::False) => {
                Ok(self.push_spanned(Expr::Bool(false), Span { start, end: self.span_before().end }))
            }
            Some(Token::Str(s)) => Ok(self.push_spanned(Expr::Str(s), Span { start, end: self.span_before().end })),
            // A standalone `opaque` expression -- a literal Value::Token
            // unique to THIS token's own position (`start`), same id
            // scheme as the ctor-field-list marker's own auto-generated
            // brand (see Expr::Token's own doc comment). First-class: can
            // be bound, passed, compared (`==`, via machine::value_eq),
            // and embedded in an ordinary tuple/list for real nominal
            // identity -- typed Type::Token(start) by elaborate_node, not
            // Dyn, so structural comparison alone tells two such fields
            // apart.
            Some(Token::Opaque) => Ok(self.push_spanned(Expr::Token(start as u64), Span { start, end: self.span_before().end })),
            Some(Token::Ident(name)) => {
                Ok(self.push_spanned(Expr::Var(name), Span { start, end: self.span_before().end }))
            }

            // [e1, e2, ...] -- no trailing comma, no empty-element gaps.
            Some(Token::LBracket) => {
                let mut items = Vec::new();
                if !matches!(self.peek(), Some(Token::RBracket)) {
                    items.push(self.expr()?);
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.bump();
                        items.push(self.expr()?);
                    }
                }
                self.expect(&Token::RBracket)?;
                Ok(self.push_spanned(Expr::ListLit(items), Span { start, end: self.span_before().end }))
            }

            Some(Token::If) => {
                let cond = self.expr()?;
                self.expect(&Token::Then)?;
                let then_ = self.expr()?;
                self.expect(&Token::Else)?;
                let else_ = self.expr()?;
                Ok(self.push_spanned(Expr::If(cond, then_, else_), Span { start, end: self.span_before().end }))
            }

            Some(Token::Perform) => {
                let effect = self.ident()?;
                self.expect(&Token::LParen)?;
                let payload = self.expr()?;
                self.expect(&Token::RParen)?;
                Ok(self.push_spanned(Expr::Perform(effect, payload), Span { start, end: self.span_before().end }))
            }

            // handle <body> with <handler-expr>
            Some(Token::Handle) => {
                let body = self.expr()?;
                self.expect(&Token::With)?;
                let handler = self.expr()?;
                Ok(self.push_spanned(Expr::Handle { body, handler }, Span { start, end: self.span_before().end }))
            }

            // handler <effect>(<payload_var>, <resume_var>) -> <body>
            Some(Token::HandlerKw) => {
                let effect = self.ident()?;
                self.expect(&Token::LParen)?;
                let payload_var = self.ident()?;
                self.expect(&Token::Comma)?;
                let resume_var = self.ident()?;
                if payload_var == resume_var {
                    // Env::bind prepends, so identical names would make
                    // resume_var silently shadow payload_var -- the payload
                    // becomes unreachable with no error. Reject at parse
                    // time instead of leaving that footgun for the handler
                    // author to discover by reading machine.rs.
                    return Err(self.err_at(
                        self.span_before(),
                        format!(
                            "handler {effect}: payload and resume binders must have different names, both named {payload_var:?}"
                        ),
                    ));
                }
                self.expect(&Token::RParen)?;
                self.expect(&Token::Arrow)?;
                let body = self.expr()?;
                Ok(self.push_spanned(
                    Expr::MakeHandler { effect, payload_var, resume_var, body },
                    Span { start, end: self.span_before().end },
                ))
            }

            // `(e)` stays ordinary grouping (unchanged); `(e, e, ...)`
            // (a comma present) is a tuple literal -- see Expr::Tuple's
            // own doc comment.
            Some(Token::LParen) => {
                let first = self.expr()?;
                if matches!(self.peek(), Some(Token::Comma)) {
                    let mut items = vec![first];
                    while matches!(self.peek(), Some(Token::Comma)) {
                        self.bump();
                        // Trailing comma allowed (also what makes a
                        // single-element tuple `(x,)` writable at all --
                        // without it, `,` would always need a following
                        // expression, so a 1-tuple could never close).
                        if matches!(self.peek(), Some(Token::RParen)) {
                            break;
                        }
                        items.push(self.expr()?);
                    }
                    self.expect(&Token::RParen)?;
                    Ok(self.push_spanned(Expr::Tuple(items), Span { start, end: self.span_before().end }))
                } else {
                    self.expect(&Token::RParen)?;
                    Ok(first)
                }
            }

            // "{" ident (":" expr)? ("," ident (":" expr)?)* "}"  -- record
            // construction. A bare `x` field puns to `x: x` (see
            // parse_record_fields' own doc comment). Fields come back
            // sorted by name, though that's cosmetic now -- elaborate_node
            // infers types::Type::Record from this Expr::Record, and
            // machine.rs evaluates it directly into a name-keyed
            // Value::Record (see both their own doc comments).
            Some(Token::LBrace) => {
                // `{x, y}` means `{x: x, y: y}` -- a punned field's
                // default value is a fresh Expr::Var for its own name,
                // spanned at the identifier just consumed
                // (parse_record_fields calls this right after
                // `self.ident()`, before bumping anything else).
                let fields = self.parse_record_fields(
                    Self::expr,
                    Some(|p: &mut Self, name: String| {
                        let span = p.span_before();
                        p.push_spanned(Expr::Var(name), span)
                    }),
                )?;
                Ok(self.push_spanned(Expr::Record(Rc::new(fields)), Span { start, end: self.span_before().end }))
            }

            // match <scrutinee> (| pattern -> expr)+  -- no "with": a literal
            // "|" can never appear inside an expression, so unlike `handle`
            // (see its own arm below -- handler-expr is arbitrary, `with` is
            // its only delimiter), the scrutinee's unbounded tail (same
            // juxtaposition hazard starts_juxtaposed_arg documents above) is
            // already unambiguous without one. The leading "|" is therefore
            // mandatory now, not optional/OCaml-style as before: optional
            // would reopen that exact ambiguity (`match e p -> body` could
            // otherwise be `e` applied to `p`).
            Some(Token::Match) => {
                let scrutinee = self.expr()?;
                self.expect(&Token::Pipe)?;
                let mut arms = Vec::new();
                loop {
                    let pat = self.pattern()?;
                    // Optional `if cond` guard -- reuses the existing `if`
                    // token rather than a new keyword (Rust/Scala/Python's
                    // choice, not OCaml/Erlang/Swift's `when`/`where`), so this costs zero
                    // new lexer tokens. No ambiguity: pattern grammar never
                    // otherwise produces `if` right before `Arrow`.
                    let guard = if matches!(self.peek(), Some(Token::If)) {
                        self.bump();
                        let g = self.expr()?;
                        // A guard may never `perform` -- decided against on
                        // purpose (not an oversight): match fallthrough
                        // means a guard's effects would fire once per
                        // ATTEMPTED arm, not once per taken one, and
                        // renno's `resume` is genuinely multi-shot, so a
                        // guard's handler resuming more than once would
                        // replay arm selection (and the arm body) itself
                        // from one `match`. See contains_perform's own doc comment above.
                        if Self::contains_perform(&self.arena, g) {
                            return Err(self.err_at(
                                self.span_before(),
                                "match guard may not perform an effect".to_string(),
                            ));
                        }
                        Some(g)
                    } else {
                        None
                    };
                    self.expect(&Token::Arrow)?;
                    let body = self.expr()?;
                    arms.push((pat, guard, body));
                    if matches!(self.peek(), Some(Token::Pipe)) {
                        self.bump();
                    } else {
                        break;
                    }
                }
                Ok(self.push_spanned(Expr::Match(scrutinee, Rc::new(arms)), Span { start, end: self.span_before().end }))
            }

            other => Err(self.err_at(self.span_before(), format!("unexpected token: {other:?}"))),
        }
    }
}
