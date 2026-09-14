use std::collections::HashMap;
use std::rc::Rc;

use crate::expr::{Arena, BinOp, DataInfo, Expr, ExprRef, Pattern, SpanMap};
use crate::lexer::{tokenize, Token};
use crate::span::Span;
use crate::types::{EffectRow, Type};

pub fn parse(src: &str) -> Result<(Arena, SpanMap, ExprRef), String> {
    let (tokens, tok_spans): (Vec<Token>, Vec<Span>) = tokenize(src)?.into_iter().unzip();
    let mut p = Parser {
        tokens,
        tok_spans,
        pos: 0,
        arena: Arena::new(),
        expr_spans: SpanMap::new(),
        src,
        branded_ctors: HashMap::new(),
    };
    let root = p.expr()?;
    if p.pos != p.tokens.len() {
        let span = p.span_at();
        return Err(p.err_at(span, format!("trailing tokens after expression: {:?}", &p.tokens[p.pos..])));
    }
    Ok((p.arena, p.expr_spans, root))
}

struct Parser<'a> {
    tokens: Vec<Token>,
    tok_spans: Vec<Span>,
    pos: usize,
    arena: Arena,
    expr_spans: SpanMap,
    src: &'a str,
    // Ctor name -> brand id, for every `data` block parsed SO FAR whose
    // constructors carry a hidden `opaque` runtime tag (see
    // build_ctor_value/pattern_atom). Patterns are built at parse time
    // with no type information available, so this is how a positional
    // constructor pattern (`Mk(x)`) learns it needs the same hidden
    // trailing tag its ctor's VALUE carries. Updated as each `data` block
    // finishes parsing its constructor list (before its body is parsed,
    // so it's visible to every pattern the body can contain) -- a later
    // `data` block reusing a ctor name always overwrites (or clears) the
    // entry, modeling ordinary lexical shadowing.
    branded_ctors: HashMap<String, u64>,
}

// A `let`/`fun`/`data` prefix collected while flattening a chain of them
// (see `atom`) -- deferred until the terminal body is parsed, then folded
// back into nested Let/Lambda/DataGroup nodes in reverse.
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
    // `data Name = Ctor1(T, ...) | Ctor2 | ...` -- one pending item expands
    // to N nested Lets when folded back (one per constructor), not one.
    // See build_ctor_value for what each constructor's bound value is, and
    // ctor_type for the Type::Data(Name) annotation each one gets.
    // Each field is optionally named (parse_ctor_field) -- `field: Int`
    // instead of a bare `Int` -- for FieldAccess's `.field` desugaring;
    // unnamed by default, and not required to be uniform within one `data`
    // block (a constructor whose fields are only PARTLY named still just
    // has no field-name entry built for it -- see the Data fold-back arm).
    // `brand`: Some(id), unique to this `data` block's own source position,
    // if any constructor wrote an `opaque` field -- see DataInfo::brand.
    Data { type_name: String, ctors: Vec<(String, Vec<(Option<String>, Type)>)>, brand: Option<u64> },
}

// The type a `data Name = ... | Ctor(T1, T2) | ...` constructor gets:
// `T1 -> T2 -> ... -> Data(Name)`, curried the same way the constructor's
// own VALUE is (build_ctor_value). Two `data` types with identically-
// shaped constructors (e.g. `data Celsius = Mk(Int)` and `data Fahrenheit
// = Mk(Int)`) are consistent with each other by default (structural, see
// types::consistent) -- add an `opaque` field to either if they should
// NOT be interchangeable. See types::Type::Data's own doc comment.
fn ctor_type(type_name: &str, field_tys: &[(Option<String>, Type)]) -> Type {
    let mut result = Type::Data(type_name.to_string());
    for (_, ty) in field_tys.iter().rev() {
        result = Type::Fun(Rc::new(ty.clone()), EffectRow::pure(), Rc::new(result));
    }
    result
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

    // Optional `: Type` annotation, e.g. after a param name or a let binder.
    fn opt_annotation(&mut self) -> Result<Option<Type>, String> {
        if matches!(self.peek(), Some(Token::Colon)) {
            self.bump();
            Ok(Some(self.parse_type()?))
        } else {
            Ok(None)
        }
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
            Some(Token::TyInt) => Ok(Type::Int),
            Some(Token::TyBool) => Ok(Type::Bool),
            Some(Token::TyStr) => Ok(Type::Str),
            Some(Token::TyDyn) => Ok(Type::Dyn),
            Some(Token::LBracket) => {
                let elem = self.parse_fun_type()?;
                self.expect(&Token::RBracket)?;
                Ok(Type::List(Rc::new(elem)))
            }
            Some(Token::LParen) => {
                let t = self.parse_fun_type()?;
                self.expect(&Token::RParen)?;
                Ok(t)
            }
            // A capitalized name that isn't one of the built-in type
            // keywords: a reference to a `data`-declared type -- Type::Data
            // is purely nominal (see its own doc comment), just a name
            // compared for equality, so this needs no name-resolution pass
            // to know whether "Option"/"List"/whatever was ever actually
            // declared, or where. That's also what makes a self-referential
            // field (`Cons(Int, List)` inside `data List` itself) work with
            // no special-casing: `List` here is just the string "List",
            // nothing to look up.
            Some(Token::Ident(name)) if name.chars().next().is_some_and(char::is_uppercase) => Ok(Type::Data(name)),
            other => Err(self.err_at(self.span_before(), format!("expected a type, found {other:?}"))),
        }
    }

    // fun_type := type ("->" ("{" ident "}")? fun_type)?  (right-assoc) --
    // only reachable from inside parens, where ")" unambiguously ends it.
    // The optional `{name}` after "->" names a row variable for row
    // polymorphism (typecheck::extend_generalized generalizes it at a
    // `let`, typecheck::lookup instantiates a fresh copy at each use). No
    // `{name}` -- the default, and the only option before this existed --
    // means EffectRow::Dyn (unknown effects, gradual default), consistent
    // with every other unannotated position.
    fn parse_fun_type(&mut self) -> Result<Type, String> {
        let atom = self.parse_type()?;
        if matches!(self.peek(), Some(Token::Arrow)) {
            self.bump();
            let row = if matches!(self.peek(), Some(Token::LBrace)) {
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

    // One field inside a `data` constructor's parens: `field: Type`, or a
    // bare `Type` (no name -- positional, no `.field` access possible for
    // it later). No lookahead needed to disambiguate: a bare type can
    // never itself START with a lowercase identifier (parse_type's
    // uppercase-Ident fallback requires an uppercase first letter, and no
    // other type-starting token is a lowercase-starting Ident), so seeing
    // one unambiguously means "field name, then `:`".
    fn parse_ctor_field(&mut self) -> Result<(Option<String>, Type), String> {
        if let Some(Token::Ident(name)) = self.peek() {
            if name.chars().next().is_some_and(char::is_lowercase) {
                let name = name.clone();
                self.bump();
                self.expect(&Token::Colon)?;
                return Ok((Some(name), self.parse_fun_type()?));
            }
        }
        Ok((None, self.parse_fun_type()?))
    }

    // One item in a `data` constructor's field list, which may be an
    // ordinary field (see parse_ctor_field) OR a bare `opaque` marker --
    // consuming no field slot (no param, no name, no type; None here means
    // "not a real field, push nothing"). `saw_opaque` is THIS constructor's
    // own flag (the caller enforces every constructor in a block has one,
    // or none -- see the `data` arm) -- the actual brand id is derived
    // once for the whole block from the block's own `data` keyword
    // position, not from where `opaque` itself was written. See
    // DataInfo::brand and build_ctor_value/pattern_atom for where that id
    // ends up: stamped into (and matched against) every constructor's
    // runtime representation.
    fn parse_ctor_field_or_opaque(&mut self, saw_opaque: &mut bool) -> Result<Option<(Option<String>, Type)>, String> {
        if matches!(self.peek(), Some(Token::Opaque)) {
            self.bump();
            *saw_opaque = true;
            Ok(None)
        } else {
            Ok(Some(self.parse_ctor_field()?))
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

    // pattern_atom := Int | true | false | Str | ident | "[" (pattern ("," pattern)*)? "]"
    // No parens for grouping yet -- every pattern shape renno currently
    // needs (literals, Var, fixed-length list, cons) is expressible without
    // them; add if a real program needs `(h :: t) :: rest`-style nesting.
    fn pattern_atom(&mut self) -> Result<Pattern, String> {
        match self.bump() {
            Some(Token::Int(n)) => Ok(Pattern::Int(n)),
            Some(Token::True) => Ok(Pattern::Bool(true)),
            Some(Token::False) => Ok(Pattern::Bool(false)),
            Some(Token::Str(s)) => Ok(Pattern::Str(s)),
            // Case decides Var vs constructor, same convention as ML/
            // Haskell/OCaml: `x`/`_` bind, `Some`/`None`/`Cons` match a
            // tag. Positional (`Some(p)`, desugars right here to the same
            // List shape build_ctor_value constructs -- Some(p) matches
            // ["Some", p], None matches ["None"]) renders invisible to
            // typecheck.rs/machine.rs entirely. Named (`Point { x: p, ...
            // }`) can't desugar here -- it needs the constructor's
            // declared field ORDER, which only typecheck knows (see
            // resolve_pattern) -- so it keeps its own Pattern::NamedCtor
            // shape until then.
            Some(Token::Ident(name)) if name.chars().next().is_some_and(char::is_uppercase) => {
                if matches!(self.peek(), Some(Token::LBrace)) {
                    self.bump();
                    let mut fields = Vec::new();
                    if !matches!(self.peek(), Some(Token::RBrace)) {
                        fields.push(self.parse_named_field_pattern()?);
                        while matches!(self.peek(), Some(Token::Comma)) {
                            self.bump();
                            fields.push(self.parse_named_field_pattern()?);
                        }
                    }
                    self.expect(&Token::RBrace)?;
                    return Ok(Pattern::NamedCtor(name, fields));
                }
                // Looked up before `name` moves into the tag below -- Some
                // iff this ctor's `data` block is branded (see
                // Parser::branded_ctors), in which case the matching
                // hidden trailing element (appended by build_ctor_value)
                // must be accounted for here too, or this pattern's length
                // would never match its own ctor's real values.
                let brand_id = self.branded_ctors.get(&name).copied();
                let mut items = vec![Pattern::Str(name)];
                if matches!(self.peek(), Some(Token::LParen)) {
                    self.bump();
                    if !matches!(self.peek(), Some(Token::RParen)) {
                        items.push(self.pattern()?);
                        while matches!(self.peek(), Some(Token::Comma)) {
                            self.bump();
                            items.push(self.pattern()?);
                        }
                    }
                    self.expect(&Token::RParen)?;
                }
                if let Some(id) = brand_id {
                    items.push(Pattern::Int(id as i64));
                }
                Ok(Pattern::List(items))
            }
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

    // One field inside a `Ctor { field: pattern, ... }` pattern.
    fn parse_named_field_pattern(&mut self) -> Result<(String, Pattern), String> {
        let name = self.ident()?;
        self.expect(&Token::Colon)?;
        let pat = self.pattern()?;
        Ok((name, pat))
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

    // postfix := atom (("(" expr ")") | ("." ident))*  -- curried calls
    // f(a)(b), and field access p.x (only meaningful once typechecked --
    // see Expr::FieldAccess's own doc comment).
    fn postfix(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let mut e = self.atom()?;
        loop {
            match self.peek() {
                Some(Token::LParen) => {
                    self.bump();
                    let arg = self.expr()?;
                    self.expect(&Token::RParen)?;
                    let span = Span { start, end: self.span_before().end };
                    e = self.push_spanned(Expr::App(e, arg), span);
                }
                Some(Token::Dot) => {
                    self.bump();
                    let field = self.ident()?;
                    let span = Span { start, end: self.span_before().end };
                    e = self.push_spanned(Expr::FieldAccess(e, field), span);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    // Peels off a run of leading `let ... in` / `fun ... ->` / `data ...
    // in` prefixes iteratively -- a token peek per iteration, not a
    // recursive call -- so a long chain of any of them costs O(1) native
    // stack instead of O(chain length). That chain shape is exactly what
    // used to overflow the stack on deeply nested/generated source (see
    // lib.rs's run_source). The terminal body/value once the chain ends is
    // parsed with an ordinary self.expr() call, same as the original
    // recursive version -- only the "is there another prefix" bookkeeping
    // moved out of the call stack, not the grammar itself.
    fn atom(&mut self) -> Result<ExprRef, String> {
        let mut pending: Vec<(usize, PendingBinder)> = Vec::new();
        // (ctor name, what self.branded_ctors mapped it to BEFORE a `data`
        // block in this same chain touched it) -- restored, in reverse, at
        // the end of this atom() call, so a `data` block's effect on
        // branded_ctors is scoped to its own `in <body>` the same way its
        // Env binding already is, not left dangling for the rest of the
        // file. See the `Some(Token::Data)` arm below.
        let mut branded_restore: Vec<(String, Option<u64>)> = Vec::new();
        loop {
            match self.peek() {
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
                Some(Token::Data) => {
                    let start = self.span_at().start;
                    self.bump();
                    let type_name = self.ident()?;
                    self.expect(&Token::Equals)?;
                    let mut ctors = Vec::new();
                    // Whether EACH constructor (same order/length as
                    // `ctors`) wrote an `opaque` field -- checked for
                    // all-or-nothing once the whole block is parsed, below.
                    let mut ctor_has_opaque = Vec::new();
                    loop {
                        let ctor_span = self.span_at();
                        let name = self.ident()?;
                        if !name.chars().next().is_some_and(char::is_uppercase) {
                            // Pattern parsing (see pattern_atom) uses case
                            // alone to tell a constructor pattern from an
                            // ordinary binding -- a lowercase constructor
                            // name would be unmatchable in a pattern
                            // (always parsed as Var, never as this ctor's
                            // tag), so reject it here rather than let that
                            // surprise show up later.
                            return Err(self.err_at(
                                ctor_span,
                                format!(
                                    "data {type_name}: constructor names must start with an uppercase letter, found {name:?}"
                                ),
                            ));
                        }
                        let mut saw_opaque = false;
                        let field_tys = if matches!(self.peek(), Some(Token::LParen)) {
                            self.bump();
                            let mut tys = Vec::new();
                            if !matches!(self.peek(), Some(Token::RParen)) {
                                if let Some(f) = self.parse_ctor_field_or_opaque(&mut saw_opaque)? {
                                    tys.push(f);
                                }
                                while matches!(self.peek(), Some(Token::Comma)) {
                                    self.bump();
                                    if let Some(f) = self.parse_ctor_field_or_opaque(&mut saw_opaque)? {
                                        tys.push(f);
                                    }
                                }
                            }
                            self.expect(&Token::RParen)?;
                            tys
                        } else {
                            Vec::new()
                        };
                        ctors.push((name, field_tys));
                        ctor_has_opaque.push(saw_opaque);
                        if matches!(self.peek(), Some(Token::Pipe)) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    self.expect(&Token::In)?;

                    // `opaque` is a whole-TYPE brand (see DataInfo::brand),
                    // but written per-constructor -- require it on every
                    // constructor, or none, so that scope is never a
                    // silent surprise (unlike named fields, which degrade
                    // quietly when only partially used: brand is safety-
                    // relevant, named fields are cosmetic).
                    let any_opaque = ctor_has_opaque.iter().any(|&b| b);
                    if any_opaque {
                        // Position, not just name, in the message -- two
                        // constructors in one block CAN share a name (e.g.
                        // `Mk(Int) | Mk(Bool, opaque)`), and by-name-only
                        // reporting can't tell those apart.
                        if let Some((missing_idx, missing_name)) = ctors
                            .iter()
                            .zip(&ctor_has_opaque)
                            .enumerate()
                            .find(|&(_, (_, &has))| !has)
                            .map(|(i, ((n, _), _))| (i, n.clone()))
                        {
                            let (branded_idx, branded_name) = ctors
                                .iter()
                                .zip(&ctor_has_opaque)
                                .enumerate()
                                .find(|&(_, (_, &has))| has)
                                .map(|(i, ((n, _), _))| (i, n.clone()))
                                .unwrap();
                            return Err(self.err_at(
                                Span { start, end: self.span_before().end },
                                format!(
                                    "data {type_name}: `opaque` must appear in every constructor or none -- found on constructor #{} ({branded_name}), missing on constructor #{} ({missing_name})",
                                    branded_idx + 1,
                                    missing_idx + 1
                                ),
                            ));
                        }
                    }
                    // One id for the whole block (its own `data` keyword's
                    // position), not the `opaque` token's -- every
                    // constructor shares it. See build_ctor_value and
                    // pattern_atom for where this id is actually stamped
                    // into (and matched against) runtime values.
                    let brand = any_opaque.then_some(start as u64);
                    for (name, _) in &ctors {
                        // Record what this name mapped to BEFORE this block
                        // touches it, so it can be put back once this
                        // block's own `in <body>` scope ends (see
                        // `branded_restore` below) -- otherwise a `data`
                        // block nested inside a larger one's body would
                        // permanently overwrite an outer, still-in-scope
                        // ctor's brand for the rest of the file.
                        branded_restore.push((name.clone(), self.branded_ctors.get(name).copied()));
                        match brand {
                            Some(id) => {
                                self.branded_ctors.insert(name.clone(), id);
                            }
                            None => {
                                self.branded_ctors.remove(name.as_str());
                            }
                        }
                    }

                    pending.push((start, PendingBinder::Data { type_name, ctors, brand }));
                }
                _ => break,
            }
        }

        let mut result = if pending.is_empty() { self.atom_leaf()? } else { self.expr()? };
        // Every wrapping binder shares this same END position (the
        // terminal body's own end) -- only its START differs (where its
        // own `let`/`fun`/`data` keyword began). `let x = 1 in let y = 2
        // in body`'s outer Let spans `[first "let", end of body]`; the
        // inner one spans `[second "let", end of body]`.
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
                PendingBinder::Data { type_name, ctors, brand } => {
                    // A constructor's fields count as "named" only when
                    // EVERY one of them has a name -- a partially-named
                    // constructor (mixing `field: Int` with a bare `Int`)
                    // just gets no entry here, so FieldAccess later
                    // reports "no such field" for it rather than guessing
                    // which position an unnamed field occupies.
                    let info = DataInfo {
                        type_name: type_name.clone(),
                        ctors: ctors
                            .iter()
                            .map(|(name, fields)| {
                                let field_names = if fields.iter().all(|(n, _)| n.is_some()) {
                                    fields.iter().map(|(n, _)| n.clone().unwrap()).collect()
                                } else {
                                    Vec::new()
                                };
                                (name.clone(), field_names)
                            })
                            .collect(),
                        ctor_types: ctors
                            .iter()
                            .map(|(name, fields)| (name.clone(), fields.iter().map(|(_, ty)| ty.clone()).collect()))
                            .collect(),
                        brand,
                    };
                    let mut body = result;
                    for (name, field_tys) in ctors.into_iter().rev() {
                        let val = self.build_ctor_value(&name, &field_tys, brand, span);
                        let ty = Some(ctor_type(&type_name, &field_tys));
                        body = self.push_spanned(Expr::Let(name, ty, val, body), span);
                    }
                    self.push_spanned(Expr::DataGroup(Rc::new(info), body), span)
                }
            };
        }
        // Undo every branded_ctors change this atom() call made, in
        // reverse (LIFO), now that its own body -- the only scope any of
        // those `data` blocks' brands were ever meant to cover -- is fully
        // parsed. Without this, a `data` block nested anywhere inside this
        // body (a let-bound sub-expression, a parenthesized atom, ...)
        // would permanently clobber an outer, still-in-scope declaration's
        // brand for the rest of the file.
        for (name, old) in branded_restore.into_iter().rev() {
            match old {
                Some(id) => {
                    self.branded_ctors.insert(name, id);
                }
                None => {
                    self.branded_ctors.remove(&name);
                }
            }
        }
        Ok(result)
    }

    // ADTs are sugar over renno's existing native List: a constructed value
    // IS a List whose first element is a Str tag (the constructor name)
    // and whose remaining elements are the fields -- e.g. `Some(5)` is
    // `["Some", 5]`, `None` is `["None"]`. That's a plain value with no new
    // Value representation, and it's exactly the shape pattern_atom's
    // constructor-pattern case (above) already expects, so match "just
    // works" with zero changes to typecheck.rs or machine.rs. A 0-arity
    // constructor is that List literal directly; an n-arity one is a chain
    // of n curried Lambdas (annotated with the declared field types, so
    // e.g. `Some("x")` is rejected statically) ending in the List literal.
    // These nodes are entirely synthesized (no distinct source text of
    // their own), so they all just inherit the enclosing `data` block's
    // own span rather than getting a more precise one.
    //
    // `brand`, when Some, appends ONE more trailing element -- the block's
    // hidden runtime tag (see DataInfo::brand) -- after every visible
    // field. Every pattern that can ever match this constructor's value
    // (positional, via pattern_atom's `branded_ctors` lookup; named, via
    // typecheck::resolve_pattern; FieldAccess's own synthetic pattern)
    // gets the identical trailing element appended, so machine.rs's
    // ordinary exact-length List matching (unchanged) naturally rejects a
    // value whose hidden tag doesn't match -- including one from an
    // unrelated `data` block that merely shares this one's name and shape.
    fn build_ctor_value(&mut self, name: &str, field_tys: &[(Option<String>, Type)], brand: Option<u64>, span: Span) -> ExprRef {
        let tag = self.push_spanned(Expr::Str(name.to_string()), span);
        let mut items = vec![tag];
        let params: Vec<String> = (0..field_tys.len()).map(|i| format!("_{i}")).collect();
        for p in &params {
            items.push(self.push_spanned(Expr::Var(p.clone()), span));
        }
        if let Some(id) = brand {
            items.push(self.push_spanned(Expr::Int(id as i64), span));
        }
        let mut value = self.push_spanned(Expr::ListLit(items), span);
        for (p, (_, ty)) in params.iter().zip(field_tys.iter()).rev() {
            value = self.push_spanned(Expr::Lambda(p.clone(), Some(ty.clone()), value), span);
        }
        value
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
        let fail_var = self.push_spanned(Expr::Var("fail".to_string()), span);
        let fail_call = self.push_spanned(Expr::App(fail_var, msg), span);
        self.push_spanned(Expr::If(pred, body, fail_call), span)
    }

    // One field inside a `Ctor { field: expr, ... }` construction.
    fn parse_named_arg(&mut self) -> Result<(String, ExprRef), String> {
        let name = self.ident()?;
        self.expect(&Token::Colon)?;
        let val = self.expr()?;
        Ok((name, val))
    }

    // Every atom form except `let`/`fun`/`data`, which `atom` handles
    // iteratively above. Reached only once no more chain prefix remains.
    fn atom_leaf(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        match self.bump() {
            Some(Token::Int(n)) => Ok(self.push_spanned(Expr::Int(n), Span { start, end: self.span_before().end })),
            Some(Token::True) => {
                Ok(self.push_spanned(Expr::Bool(true), Span { start, end: self.span_before().end }))
            }
            Some(Token::False) => {
                Ok(self.push_spanned(Expr::Bool(false), Span { start, end: self.span_before().end }))
            }
            Some(Token::Str(s)) => Ok(self.push_spanned(Expr::Str(s), Span { start, end: self.span_before().end })),
            // `Ident { field: expr, ... }` -- named-field construction,
            // ONLY meaningful once typechecked (see Expr::NamedCall's own
            // doc comment); a bare `Ident` not followed by `{` is just an
            // ordinary variable reference, unchanged.
            Some(Token::Ident(name)) => {
                let callee_span = Span { start, end: self.span_before().end };
                if matches!(self.peek(), Some(Token::LBrace)) {
                    self.bump();
                    let mut fields = Vec::new();
                    if !matches!(self.peek(), Some(Token::RBrace)) {
                        fields.push(self.parse_named_arg()?);
                        while matches!(self.peek(), Some(Token::Comma)) {
                            self.bump();
                            fields.push(self.parse_named_arg()?);
                        }
                    }
                    self.expect(&Token::RBrace)?;
                    let callee = self.push_spanned(Expr::Var(name), callee_span);
                    let span = Span { start, end: self.span_before().end };
                    Ok(self.push_spanned(Expr::NamedCall(callee, Rc::new(fields)), span))
                } else {
                    Ok(self.push_spanned(Expr::Var(name), callee_span))
                }
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

            Some(Token::LParen) => {
                let e = self.expr()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }

            // match <scrutinee> with (| pattern -> expr)+  -- the first "|"
            // before the first arm is optional (OCaml-style), every one
            // after it is required to separate arms.
            Some(Token::Match) => {
                let scrutinee = self.expr()?;
                self.expect(&Token::With)?;
                if matches!(self.peek(), Some(Token::Pipe)) {
                    self.bump();
                }
                let mut arms = Vec::new();
                loop {
                    let pat = self.pattern()?;
                    self.expect(&Token::Arrow)?;
                    let body = self.expr()?;
                    arms.push((pat, body));
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
