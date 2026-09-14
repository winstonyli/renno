use std::rc::Rc;

use crate::expr::{Arena, BinOp, DataInfo, Expr, ExprRef, Pattern, SpanMap};
use crate::lexer::{tokenize, Token};
use crate::span::Span;
use crate::types::{EffectRow, Type};

pub fn parse(src: &str) -> Result<(Arena, SpanMap, ExprRef), String> {
    let (tokens, tok_spans): (Vec<Token>, Vec<Span>) = tokenize(src)?.into_iter().unzip();
    let mut p = Parser { tokens, tok_spans, pos: 0, arena: Arena::new(), expr_spans: SpanMap::new(), src };
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
}

// A `let`/`fun`/`data` prefix collected while flattening a chain of them
// (see `atom`) -- deferred until the terminal body is parsed, then folded
// back into nested Let/Lambda/DataGroup nodes in reverse.
enum PendingBinder {
    Let { var: String, ann: Option<Type>, val: ExprRef },
    // `let rec f = val_f [and g = val_g ...] in ...` -- one or more
    // simultaneously-recursive bindings folding back into a single
    // Expr::LetRec (never Expr::Let, which is never recursive).
    LetRec { bindings: Vec<(String, Option<Type>, ExprRef)> },
    Fun { param: String, ann: Option<Type> },
    // `data Name = Ctor1(T, ...) | Ctor2 | ...` -- one pending item expands
    // to N nested Lets when folded back (one per constructor), not one.
    // See build_ctor_value for what each constructor's bound value is, and
    // ctor_type for the nominal Type::Data(Name) annotation each one gets.
    // Each field is optionally named (parse_ctor_field) -- `field: Int`
    // instead of a bare `Int` -- for FieldAccess's `.field` desugaring;
    // unnamed by default, and not required to be uniform within one `data`
    // block (a constructor whose fields are only PARTLY named still just
    // has no field-name entry built for it -- see the Data fold-back arm).
    Data { type_name: String, ctors: Vec<(String, Vec<(Option<String>, Type)>)> },
}

// The nominal type a `data Name = ... | Ctor(T1, T2) | ...` constructor
// gets: `T1 -> T2 -> ... -> Data(Name)`, curried the same way the
// constructor's own VALUE is (build_ctor_value) -- Data(Name) is exactly
// what lets two `data` types with identically-shaped constructors (e.g.
// `data Celsius = Mk(Int)` and `data Fahrenheit = Mk(Int)`) stay
// statically distinguishable, since consistent() only accepts two Data
// with the exact same name. See types::Type::Data's own doc comment for
// the runtime side's necessarily shallower story.
fn ctor_type(type_name: &str, field_tys: &[(Option<String>, Type)]) -> Type {
    let mut result = Type::Data(type_name.to_string());
    for (_, ty) in field_tys.iter().rev() {
        result = Type::Fun(Rc::new(ty.clone()), EffectRow::pure(), Rc::new(result));
    }
    result
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

    // expr := cmp
    fn expr(&mut self) -> Result<ExprRef, String> {
        self.cmp()
    }

    // cmp := cons (("==" | "<") cons)?  -- non-associative, one comparison
    fn cmp(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let lhs = self.cons()?;
        let op = match self.peek() {
            Some(Token::EqEq) => Some(BinOp::Eq),
            Some(Token::Lt) => Some(BinOp::Lt),
            _ => None,
        };
        match op {
            Some(op) => {
                self.bump();
                let rhs = self.cons()?;
                let span = Span { start, end: self.span_before().end };
                Ok(self.push_spanned(Expr::BinOp(op, lhs, rhs), span))
            }
            None => Ok(lhs),
        }
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
                _ => break,
            };
            self.bump();
            let rhs = self.unary()?;
            let span = Span { start, end: self.span_before().end };
            lhs = self.push_spanned(Expr::BinOp(op, lhs, rhs), span);
        }
        Ok(lhs)
    }

    // unary := "-"? postfix -- prefix negation, desugared to `0 - x` at
    // parse time rather than a new AST node (Sub already exists, and this
    // is the only user). Right-recursive (`unary` not `postfix` on the
    // operand) so `- -x` parses too, for whatever that's worth.
    fn unary(&mut self) -> Result<ExprRef, String> {
        if matches!(self.peek(), Some(Token::Minus)) {
            let start = self.span_at().start;
            self.bump();
            let operand = self.unary()?;
            let span = Span { start, end: self.span_before().end };
            let zero = self.push_spanned(Expr::Int(0), span);
            Ok(self.push_spanned(Expr::BinOp(BinOp::Sub, zero, operand), span))
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
                    loop {
                        let var = self.ident()?;
                        let ann = self.opt_annotation()?;
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
                        pending.push((start, PendingBinder::Let { var, ann, val }));
                    }
                }
                Some(Token::Fun) => {
                    let start = self.span_at().start;
                    self.bump();
                    let param = self.ident()?;
                    let ann = self.opt_annotation()?;
                    self.expect(&Token::Arrow)?;
                    pending.push((start, PendingBinder::Fun { param, ann }));
                }
                Some(Token::Data) => {
                    let start = self.span_at().start;
                    self.bump();
                    let type_name = self.ident()?;
                    self.expect(&Token::Equals)?;
                    let mut ctors = Vec::new();
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
                        let field_tys = if matches!(self.peek(), Some(Token::LParen)) {
                            self.bump();
                            let mut tys = Vec::new();
                            if !matches!(self.peek(), Some(Token::RParen)) {
                                tys.push(self.parse_ctor_field()?);
                                while matches!(self.peek(), Some(Token::Comma)) {
                                    self.bump();
                                    tys.push(self.parse_ctor_field()?);
                                }
                            }
                            self.expect(&Token::RParen)?;
                            tys
                        } else {
                            Vec::new()
                        };
                        ctors.push((name, field_tys));
                        if matches!(self.peek(), Some(Token::Pipe)) {
                            self.bump();
                        } else {
                            break;
                        }
                    }
                    self.expect(&Token::In)?;
                    pending.push((start, PendingBinder::Data { type_name, ctors }));
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
                PendingBinder::Let { var, ann, val } => self.push_spanned(Expr::Let(var, ann, val, result), span),
                PendingBinder::LetRec { bindings } => {
                    self.push_spanned(Expr::LetRec(Rc::new(bindings), result), span)
                }
                PendingBinder::Fun { param, ann } => self.push_spanned(Expr::Lambda(param, ann, result), span),
                PendingBinder::Data { type_name, ctors } => {
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
                    };
                    let mut body = result;
                    for (name, field_tys) in ctors.into_iter().rev() {
                        let val = self.build_ctor_value(&name, &field_tys, span);
                        let ty = Some(ctor_type(&type_name, &field_tys));
                        body = self.push_spanned(Expr::Let(name, ty, val, body), span);
                    }
                    self.push_spanned(Expr::DataGroup(Rc::new(info), body), span)
                }
            };
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
    fn build_ctor_value(&mut self, name: &str, field_tys: &[(Option<String>, Type)], span: Span) -> ExprRef {
        let tag = self.push_spanned(Expr::Str(name.to_string()), span);
        let mut items = vec![tag];
        let params: Vec<String> = (0..field_tys.len()).map(|i| format!("_{i}")).collect();
        for p in &params {
            items.push(self.push_spanned(Expr::Var(p.clone()), span));
        }
        let mut value = self.push_spanned(Expr::ListLit(items), span);
        for (p, (_, ty)) in params.iter().zip(field_tys.iter()).rev() {
            value = self.push_spanned(Expr::Lambda(p.clone(), Some(ty.clone()), value), span);
        }
        value
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
