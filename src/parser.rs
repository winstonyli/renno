use std::rc::Rc;

use crate::expr::{Arena, BinOp, Expr, ExprRef, Pattern, SpanMap};
use crate::index_expr::IndexExpr;
use crate::lexer::{tokenize, Token};
use crate::span::Span;
use crate::types::{Hc, Type};
use crate::util::find_field;

// The nesting budget: how many levels of native recursion a program may ask of the passes
// that walk it by recursion (the parser, then typecheck, and the Drop of a pattern or a
// type). Every such pass takes a bounded stack frame per level, and a deeper program
// would overflow the stack and abort the process, so a deeper program is an error here,
// once, before any of them runs. A `let`/`fun` chain is not a level (parser, checker and
// resolver all walk it by iteration). Real programs nest well under 100.
//
// Measured at 2,000: the worker's 256 MiB stack holds about 4,700 levels of the most
// expensive pass (typecheck's elaborate) in an unoptimized build, 66,000 in release, so
// the budget leaves the unoptimized build more than 2x headroom (see docs/superpowers/
// plans/u10-report.md; nesting_inside_the_budget_runs pins it for every construct).
pub const MAX_NESTING: usize = 2000;

pub fn parse(src: &str) -> Result<(Arena, SpanMap, ExprRef), String> {
    let (tokens, tok_spans): (Vec<Token>, Vec<Span>) = tokenize(src)?.into_iter().unzip();
    let mut p = Parser { tokens, tok_spans, pos: 0, depth: 0, arena: Arena::new(), expr_spans: SpanMap::new(), src };
    let root = p.expr()?;
    if p.pos != p.tokens.len() {
        let span = p.span_at();
        return Err(p.err_at(span, format!("unexpected `{}` after the expression", span.text(src))));
    }
    p.check_tree_depth(root)?;
    Ok((p.arena, p.expr_spans, root))
}

// Parses a module's text (its own line numbers) into the caller's arena; the caller shifts the new
// nodes' spans to the module's place in the virtual source (imports::expand).
pub fn parse_module(arena: &mut Arena, spans: &mut SpanMap, text: &str) -> Result<ExprRef, String> {
    let (tokens, tok_spans): (Vec<Token>, Vec<Span>) = tokenize(text)?.into_iter().unzip();
    let mut p = Parser { tokens, tok_spans, pos: 0, depth: 0, arena: std::mem::replace(arena, Arena::new()), expr_spans: std::mem::replace(spans, SpanMap::new()), src: text };
    let result = (|| {
        let root = p.expr()?;
        if p.pos != p.tokens.len() {
            let span = p.span_at();
            return Err(p.err_at(span, format!("unexpected `{}` after the expression", span.text(text))));
        }
        p.check_tree_depth(root)?;
        Ok(root)
    })();
    *arena = p.arena;
    *spans = p.expr_spans;
    result
}

// Test-only entry point: parses a single, standalone type expression
// (not a full program) -- used by the Vec(...) syntax tests (see the
// design spec's own §1/§2), which need to inspect a Type in isolation
// rather than run a whole program. Not part of the language's own real
// parsing path.
#[cfg(test)]
pub fn parse_type_string(src: &str) -> Result<Type, String> {
    let (tokens, tok_spans): (Vec<Token>, Vec<Span>) = tokenize(src)?.into_iter().unzip();
    let mut p = Parser { tokens, tok_spans, pos: 0, depth: 0, arena: Arena::new(), expr_spans: SpanMap::new(), src };
    let ty = p.parse_fun_type()?;
    crate::machine::eval_type(&p.arena, ty, &p.expr_spans)
}

// What may follow `, ..` in braces: nothing (a record pattern), a row
// variable name (a record type) or a base expression (an update).
enum RecordRest<'a, T> {
    None,
    Row(&'a mut Option<String>),
    Base(&'a mut Option<T>),
}

struct Parser<'a> {
    tokens: Vec<Token>,
    tok_spans: Vec<Span>,
    pos: usize,
    // Open recursive calls that `nested` counts, against MAX_NESTING.
    depth: usize,
    arena: Arena,
    expr_spans: SpanMap,
    src: &'a str,
}

// A `let`/`fun` prefix collected while flattening a chain of them (see
// `atom`) -- deferred until the terminal body is parsed, then folded back
// into nested Let/Lambda nodes in reverse.
enum PendingBinder {
    Let { var: String, ann: Option<ExprRef>, val: ExprRef },
    // `let rec f = val_f [and g = val_g ...] in ...` -- one or more
    // simultaneously-recursive bindings folding back into a single
    // Expr::LetRec (never Expr::Let, which is never recursive).
    LetRec { bindings: Vec<(String, Option<ExprRef>, ExprRef)> },
    Fun { param: String, ann: Option<ExprRef> },
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

    // How an error names the token bump() just consumed: its source text, or the end of input.
    fn found(&self, tok: &Option<Token>) -> String {
        match tok {
            Some(_) => format!("`{}`", self.span_before().text(self.src)),
            None => "end of input".to_string(),
        }
    }

    fn err_at(&self, span: Span, msg: String) -> String {
        span.format_error(self.src, &msg)
    }

    fn too_deep(&self, span: Span) -> String {
        self.err_at(span, format!("expression nested too deeply (limit {MAX_NESTING})"))
    }

    // One more level of recursion, or the nesting error. `leave(k)` gives back k levels.
    fn enter(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            return Err(self.too_deep(self.span_at()));
        }
        Ok(())
    }

    // `f` one level deeper. Every recursive parse function that is not a loop re-enters
    // itself through this (or is the wrapper around one that does).
    fn nested<T>(&mut self, f: impl FnOnce(&mut Self) -> Result<T, String>) -> Result<T, String> {
        self.enter()?;
        let r = f(self);
        self.depth -= 1;
        r
    }

    // The nesting of the finished tree. The parser also builds depth in loops (`1 + 1 + ..`,
    // `f(a)(b)..`, `a || b || ..`) that `nested` never sees, and a parenthesis adds no node,
    // so this is a second, independent count: what typecheck and the later passes will
    // recurse through. The body of a `let`/`let rec`/`fun` is the next link of a chain they
    // walk by iteration, so reaching it costs no level (a chain is unbounded: what the
    // checker builds along it, scopes and types, drops through util::bounded_drop). Iterative.
    fn check_tree_depth(&self, root: ExprRef) -> Result<(), String> {
        let mut stack = vec![(root, 0usize)];
        let mut kids = Vec::new();
        while let Some((n, depth)) = stack.pop() {
            if depth > MAX_NESTING {
                return Err(self.too_deep(self.expr_spans[n]));
            }
            let chain_body = match &self.arena[n] {
                Expr::Let(_, _, _, body) | Expr::LetRec(_, body) | Expr::Lambda(_, _, body) => Some(*body),
                _ => None,
            };
            kids.clear();
            self.arena[n].children(&mut kids);
            stack.extend(kids.iter().map(|&k| if Some(k) == chain_body { (k, depth) } else { (k, depth + 1) }));
        }
        Ok(())
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
        // Iterative: this runs at parse time, before the depth walk can bound the subtree.
        let mut stack = vec![e];
        while let Some(n) = stack.pop() {
            if matches!(arena[n], Expr::Perform(..)) {
                return true;
            }
            arena[n].children(&mut stack);
        }
        false
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
            other => Err(self.err_at(self.span_before(), format!("expected `{}`, found {}", want.text(), self.found(&other)))),
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Some(Token::Ident(s)) => Ok(s),
            other => Err(self.err_at(self.span_before(), format!("expected an identifier, found {}", self.found(&other)))),
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
        rest: RecordRest<T>,
    ) -> Result<Vec<(String, T)>, String> {
        if matches!(self.peek(), Some(Token::RBrace)) {
            return Err(self.err_at(self.span_before(), "a record needs at least one field".to_string()));
        }
        if matches!(self.peek(), Some(Token::DotDot)) && matches!(rest, RecordRest::Base(_)) {
            return Err(self.err_at(self.span_at(), "an update needs at least one field".to_string()));
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
                if matches!(self.peek(), Some(Token::DotDot)) {
                    // `..r`: the row variable of an open record TYPE, or the base
                    // of an UPDATE expression, last in the braces.
                    match rest {
                        RecordRest::Row(slot) => {
                            self.bump();
                            *slot = Some(self.ident()?);
                        }
                        RecordRest::Base(slot) => {
                            self.bump();
                            *slot = Some(parse_value(self)?);
                        }
                        RecordRest::None => {
                            return Err(self.err_at(
                                self.span_at(),
                                "`..r` is only allowed in a record type or an update".to_string(),
                            ))
                        }
                    }
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
    //
    // Every type form below builds an EXPRESSION (spec 2026-10-10 types-as-
    // values, section 3): sugar desugars into calls to the prelude's type
    // constructors, spelled `#List`, `#Fun`, ... so no user binding can
    // capture them. A name the user writes is a plain variable, resolved in the
    // binder's scope, and `Name(arg)` is an ordinary call (`Vec(e)` and
    // `Indexed(T, e)` too, see parse_type). typecheck evaluates
    // the result with annotation_type.
    fn opt_annotation(&mut self) -> Result<Option<ExprRef>, String> {
        if matches!(self.peek(), Some(Token::Colon)) {
            self.bump();
            Ok(Some(self.parse_type_atom_union()?))
        } else {
            Ok(None)
        }
    }

    // `#name` as an expression, spanning from `start` to the last token read.
    fn type_ctor(&mut self, name: &str, start: usize) -> ExprRef {
        let span = Span { start, end: self.span_before().end };
        self.push_spanned(Expr::Var(format!("#{name}")), span)
    }

    // `#name(arg)`.
    fn type_call(&mut self, name: &str, arg: ExprRef, start: usize) -> ExprRef {
        let ctor = self.type_ctor(name, start);
        let span = Span { start, end: self.span_before().end };
        self.push_spanned(Expr::App(ctor, arg), span)
    }

    // `#name((items...))`: Tuple, Union and Fun take one tuple argument.
    fn type_call_items(&mut self, name: &str, items: Vec<ExprRef>, start: usize) -> ExprRef {
        let span = Span { start, end: self.span_before().end };
        let tuple = self.push_spanned(Expr::Tuple(items), span);
        self.type_call(name, tuple, start)
    }

    // A type that is already known (a symbolic index): no evaluation left to do.
    fn type_lit(&mut self, ty: Type, start: usize) -> ExprRef {
        let span = Span { start, end: self.span_before().end };
        self.push_spanned(Expr::TypeLit(Rc::new(ty)), span)
    }

    // The index of `Vec(e)` or `Indexed(T, e)` as the argument passed: an Int
    // for a literal, else the type Vec(e), a TypeLit, since an index variable
    // has no value (spec ruling 12.4). The prelude Vec and Indexed take the
    // index from either.
    fn index_arg(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let index = self.parse_index_expr()?;
        Ok(match index {
            IndexExpr::Lit(k) => {
                let span = Span { start, end: self.span_before().end };
                self.push_spanned(Expr::Int(k), span)
            }
            index => self.type_lit(Type::Indexed(Hc::new(Type::List(Hc::new(Type::Dyn))), Rc::new(index)), start),
        })
    }

    // type_atom_union := type ("|" type)*  -- parse_type's own ATOM level
    // (no "->"), unioned. Used where a trailing "->" must NOT be consumed
    // as part of the type (opt_annotation -- see its own doc comment); a
    // function type still needs explicit parens there, same as before
    // union types existed, and now so does a function type as one union
    // alternative (`fun x: (A -> B) | C -> body`).
    fn parse_type_atom_union(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let first = self.parse_type()?;
        if !matches!(self.peek(), Some(Token::Pipe)) {
            return Ok(first);
        }
        let mut alts = vec![first];
        while matches!(self.peek(), Some(Token::Pipe)) {
            self.bump();
            alts.push(self.parse_type()?);
        }
        Ok(self.type_call_items("Union", alts, start))
    }

    // union_type := fun_type ("|" fun_type)*  (left-associative in surface
    // syntax, though Type::Union itself is an unordered set -- order in
    // the Vec never matters to types::consistent). `|` binds LOOSER than
    // `->` (parse_fun_type's own right-recursion is greedy, so `A -> B | C`
    // parses as `(A -> B) | C`, not `A -> (B | C)`). Used everywhere a
    // trailing `->` couldn't be ambiguous with something else that follows
    // (unlike opt_annotation -- see parse_type_atom_union): inside `[...]`/
    // `(...)`, and the arguments of a type call (terminated by `,`/`)`).
    fn parse_union_type(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let first = self.parse_fun_type()?;
        if !matches!(self.peek(), Some(Token::Pipe)) {
            return Ok(first);
        }
        let mut alts = vec![first];
        while matches!(self.peek(), Some(Token::Pipe)) {
            self.bump();
            alts.push(self.parse_fun_type()?);
        }
        Ok(self.type_call_items("Union", alts, start))
    }

    // Bare, atom-only: a name or call (`T`, `F(A)`), `Vec(n)`,
    // `Indexed(T, n)`, an Int, "[" fun_type "]", "(" fun_type ")" or a
    // record type. Deliberately does NOT chain "->" at this level --
    // an annotation site (`fun x: T ->`, `let x: T =`) is always
    // immediately followed by its own "->"/"=" token, so a bare trailing
    // arrow here would be ambiguous between "this type continues" and "the
    // annotation just ended". A function type must be parenthesized to
    // disambiguate: `fun f: (Int -> Int) -> ...`. "[" / "]" don't have that
    // ambiguity (nothing else starts with "["), so a list element type can
    // freely be a function type without extra parens: `[Int -> Int]`.
    fn parse_type(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        match self.bump() {
            Some(Token::LBracket) => {
                let elem = self.parse_fun_type()?;
                self.expect(&Token::RBracket)?;
                Ok(self.type_call("List", elem, start))
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
                    Ok(self.type_call_items("Tuple", items, start))
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
                let mut tail = None;
                let fields = self.parse_record_fields(Self::parse_union_type, None, RecordRest::Row(&mut tail))?;
                let span = Span { start, end: self.span_before().end };
                let record = self.push_spanned(Expr::Record(Rc::new(fields)), span);
                // `{x: T, ..r}` is `#Record(({x: T}, "r"))`.
                match tail {
                    Some(name) => {
                        let row = self.push_spanned(Expr::Str(name), span);
                        Ok(self.type_call_items("Record", vec![record, row], start))
                    }
                    None => Ok(self.type_call("Record", record, start)),
                }
            }
            // `Vec(e)` and `Indexed(T, e)` (T at index e, spec section 9.1)
            // call whatever `Vec` and `Indexed` are in scope, like any name
            // (spec 11.8), so a local binding of either changes them. The
            // index language stays built in: the index is parsed here and
            // passed as an argument (index_arg). Matched before the plain
            // name case below.
            Some(Token::Ident(name)) if (name == "Vec" || name == "Indexed") && matches!(self.peek(), Some(Token::LParen)) => {
                let indexed = name == "Indexed";
                let head_span = Span { start, end: self.span_before().end };
                let head = self.push_spanned(Expr::Var(name), head_span);
                let paren_start = self.span_at().start;
                self.bump();
                let arg = if indexed {
                    let wrapped = self.parse_union_type()?;
                    self.expect(&Token::Comma)?;
                    let index = self.index_arg()?;
                    self.expect(&Token::RParen)?;
                    let span = Span { start: paren_start, end: self.span_before().end };
                    self.push_spanned(Expr::Tuple(vec![wrapped, index]), span)
                } else {
                    let index = self.index_arg()?;
                    self.expect(&Token::RParen)?;
                    index
                };
                let span = Span { start, end: self.span_before().end };
                Ok(self.push_spanned(Expr::App(head, arg), span))
            }
            // A name is an ordinary variable, resolved lexically when
            // typecheck evaluates the annotation, so `let Int = 5` changes
            // what `Int` means (spec section 9.3). `F(A)` applies it (spec
            // section 9.1).
            Some(Token::Ident(name)) => {
                let span = Span { start, end: self.span_before().end };
                let mut head = self.push_spanned(Expr::Var(name), span);
                // `M.T`: a type read out of a record (a module); typecheck evaluates it like `.field`.
                while matches!(self.peek(), Some(Token::Dot)) && matches!(self.tokens.get(self.pos + 1), Some(Token::Ident(_))) {
                    self.bump();
                    let field = self.ident()?;
                    let span = Span { start, end: self.span_before().end };
                    head = self.push_spanned(Expr::FieldAccess(head, field), span);
                }
                self.type_apps(head, start)
            }
            // An Int is a value an applied type function may take (`Rep(2)`).
            Some(Token::Int(n)) => {
                let span = Span { start, end: self.span_before().end };
                Ok(self.push_spanned(Expr::Int(n), span))
            }
            other => Err(self.err_at(self.span_before(), format!("expected a type, found {}", self.found(&other)))),
        }
    }

    // `head(args)(args)...` in a type: each call passes one argument, a tuple
    // when there is a comma, like `f(a, b)` in an expression. The items are
    // types, so the sugar works inside: `Pair([Int])`.
    fn type_apps(&mut self, mut head: ExprRef, start: usize) -> Result<ExprRef, String> {
        while matches!(self.peek(), Some(Token::LParen)) {
            self.bump();
            let args_start = self.span_before().start;
            let mut items = vec![self.parse_union_type()?];
            let mut tuple = false;
            while matches!(self.peek(), Some(Token::Comma)) {
                self.bump();
                tuple = true;
                if matches!(self.peek(), Some(Token::RParen)) {
                    break;
                }
                items.push(self.parse_union_type()?);
            }
            self.expect(&Token::RParen)?;
            let arg = if tuple {
                let span = Span { start: args_start, end: self.span_before().end };
                self.push_spanned(Expr::Tuple(items), span)
            } else {
                items.pop().expect("one item")
            };
            let span = Span { start, end: self.span_before().end };
            head = self.push_spanned(Expr::App(head, arg), span);
        }
        Ok(head)
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
        // Each `+`/`-` makes the tree one level deeper (an index is not in the arena, so
        // check_tree_depth never sees it); the levels are given back at the end.
        let mut built = 0;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => IndexExpr::Add,
                Some(Token::Minus) => IndexExpr::Sub,
                _ => break,
            };
            self.bump();
            self.enter()?;
            built += 1;
            let rhs = self.parse_index_term()?;
            lhs = op(Rc::new(lhs), Rc::new(rhs));
        }
        self.depth -= built;
        Ok(lhs)
    }

    fn parse_index_term(&mut self) -> Result<IndexExpr, String> {
        let mut lhs = self.parse_index_atom()?;
        let mut built = 0;
        while matches!(self.peek(), Some(Token::Star)) {
            self.bump();
            self.enter()?;
            built += 1;
            let rhs = self.parse_index_atom()?;
            lhs = IndexExpr::Mul(Rc::new(lhs), Rc::new(rhs));
        }
        self.depth -= built;
        Ok(lhs)
    }

    fn parse_index_atom(&mut self) -> Result<IndexExpr, String> {
        match self.bump() {
            Some(Token::Int(n)) => Ok(IndexExpr::Lit(n)),
            Some(Token::Ident(name)) => Ok(IndexExpr::Var(name)),
            Some(Token::LParen) => {
                let inner = self.nested(Self::parse_index_expr)?;
                self.expect(&Token::RParen)?;
                Ok(inner)
            }
            other => Err(self.err_at(self.span_before(), format!("expected an index expression, found {}", self.found(&other)))),
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
    // means Tail::Dyn (unknown effects, gradual default), consistent
    // with every other unannotated position. A "{" that ISN'T a row
    // variable (peek_is_row_var says no) falls through to Dyn here and
    // gets parsed as an ordinary return-type atom instead, by the
    // recursive parse_fun_type -> parse_type call below -- which is
    // exactly how a record type (`-> {x: Int}`) reaches parse_type's own
    // LBrace arm rather than being swallowed as a malformed row variable.
    // `A -> B` is `#Fun((A, B))`; `A ->{e} B` is `#Fun((A, "e", B))`.
    fn parse_fun_type(&mut self) -> Result<ExprRef, String> {
        let start = self.span_at().start;
        let atom = self.nested(Self::parse_type)?;
        if matches!(self.peek(), Some(Token::Arrow)) {
            self.bump();
            let row = if matches!(self.peek(), Some(Token::LBrace)) && self.peek_is_row_var() {
                self.bump();
                let row_start = self.span_at().start;
                let name = self.ident()?;
                let row = self.push_spanned(Expr::Str(name), Span { start: row_start, end: self.span_before().end });
                self.expect(&Token::RBrace)?;
                Some(row)
            } else {
                None
            };
            let ret = self.nested(Self::parse_fun_type)?;
            let items = match row {
                Some(row) => vec![atom, row, ret],
                None => vec![atom, ret],
            };
            Ok(self.type_call_items("Fun", items, start))
        } else {
            Ok(atom)
        }
    }

    // pattern := pattern_atom ("::" pattern)?  (right-assoc, `h :: t`)
    fn pattern(&mut self) -> Result<Pattern, String> {
        let head = self.nested(Self::pattern_atom)?;
        if matches!(self.peek(), Some(Token::ColonColon)) {
            self.bump();
            let tail = self.nested(Self::pattern)?;
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
            // Value::List at runtime (see Expr::Tuple's own doc comment), so this is a
            // Pattern::Tuple, matched like the list pattern `[p, p, ...]`.
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
                    Ok(Pattern::Tuple(items))
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
                    self.parse_record_fields(Self::pattern, Some(|_: &mut Self, name| Pattern::Var(name)), RecordRest::None)?;
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
            other => Err(self.err_at(self.span_before(), format!("expected a pattern, found {}", self.found(&other)))),
        }
    }

    // expr := or_expr
    fn expr(&mut self) -> Result<ExprRef, String> {
        self.nested(Self::or_expr)
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
                if matches!(self.peek(), Some(Token::EqEq | Token::Lt | Token::Gt | Token::LtEq | Token::GtEq | Token::BangEq)) {
                    let at = self.span_at();
                    return Err(self.err_at(at, format!("comparisons don't chain: parenthesize one side of `{}`", at.text(self.src))));
                }
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
            let rhs = self.nested(Self::cons)?;
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
            let operand = self.nested(Self::unary)?;
            let span = Span { start, end: self.span_before().end };
            let zero = self.push_spanned(Expr::Int(0), span);
            Ok(self.push_spanned(Expr::BinOp(BinOp::Sub, zero, operand), span))
        } else if matches!(self.peek(), Some(Token::Bang)) {
            let start = self.span_at().start;
            self.bump();
            let operand = self.nested(Self::unary)?;
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
                _ => break,
            }
        }

        let mut result = if pending.is_empty() { self.atom_leaf()? } else { self.expr()? };
        // Every wrapping binder shares this same END position (the
        // terminal body's own end) -- only its START differs (where its
        // own `let`/`fun` keyword began). `let x = 1 in let y = 2 in
        // body`'s outer Let spans `[first "let", end of body]`; the inner
        // one spans `[second "let", end of body]`.
        let end = self.expr_spans[result].end;
        for (start, binder) in pending.into_iter().rev() {
            let span = Span { start, end };
            result = match binder {
                PendingBinder::Let { var, ann, val } => {
                    self.push_spanned(Expr::Let(var, ann, val, result), span)
                }
                PendingBinder::LetRec { bindings } => {
                    self.push_spanned(Expr::LetRec(Rc::new(bindings), result), span)
                }
                PendingBinder::Fun { param, ann } => {
                    self.push_spanned(Expr::Lambda(param, ann, result), span)
                }
            };
        }
        Ok(result)
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
                let mut base = None;
                let fields = self.parse_record_fields(
                    Self::expr,
                    Some(|p: &mut Self, name: String| {
                        let span = p.span_before();
                        p.push_spanned(Expr::Var(name), span)
                    }),
                    RecordRest::Base(&mut base),
                )?;
                let fields = Rc::new(fields);
                let span = Span { start, end: self.span_before().end };
                Ok(match base {
                    Some(base) => self.push_spanned(Expr::RecordUpdate(fields, base), span),
                    None => self.push_spanned(Expr::Record(fields), span),
                })
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
