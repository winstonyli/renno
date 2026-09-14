use std::rc::Rc;

use crate::expr::{Arena, BinOp, Expr, ExprRef, Pattern};
use crate::lexer::{tokenize, Token};
use crate::types::{EffectRow, Type};

pub fn parse(src: &str) -> Result<(Arena, ExprRef), String> {
    let tokens = tokenize(src)?;
    let mut p = Parser { tokens, pos: 0, arena: Arena::new() };
    let root = p.expr()?;
    if p.pos != p.tokens.len() {
        return Err(format!("trailing tokens after expression: {:?}", &p.tokens[p.pos..]));
    }
    Ok((p.arena, root))
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    arena: Arena,
}

// A `let`/`fun` prefix collected while flattening a chain of them (see
// `atom`) -- deferred until the terminal body is parsed, then folded back
// into nested Let/Lambda nodes in reverse.
enum PendingBinder {
    Let { var: String, ann: Option<Type>, rec: bool, val: ExprRef },
    Fun { param: String, ann: Option<Type> },
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn bump(&mut self) -> Option<Token> {
        let t = self.tokens.get(self.pos).cloned();
        self.pos += 1;
        t
    }

    fn expect(&mut self, want: &Token) -> Result<(), String> {
        match self.bump() {
            Some(ref t) if t == want => Ok(()),
            other => Err(format!("expected {want:?}, found {other:?}")),
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.bump() {
            Some(Token::Ident(s)) => Ok(s),
            other => Err(format!("expected identifier, found {other:?}")),
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
            other => Err(format!("expected a type, found {other:?}")),
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
            other => Err(format!("expected a pattern, found {other:?}")),
        }
    }

    // expr := cmp
    fn expr(&mut self) -> Result<ExprRef, String> {
        self.cmp()
    }

    // cmp := add (("==" | "<") add)?  -- non-associative, one comparison
    fn cmp(&mut self) -> Result<ExprRef, String> {
        let lhs = self.add()?;
        let op = match self.peek() {
            Some(Token::EqEq) => Some(BinOp::Eq),
            Some(Token::Lt) => Some(BinOp::Lt),
            _ => None,
        };
        match op {
            Some(op) => {
                self.bump();
                let rhs = self.add()?;
                Ok(self.arena.push(Expr::BinOp(op, lhs, rhs)))
            }
            None => Ok(lhs),
        }
    }

    // add := postfix (("+" | "++") postfix)*  (left-associative)
    fn add(&mut self) -> Result<ExprRef, String> {
        let mut lhs = self.postfix()?;
        loop {
            let op = match self.peek() {
                Some(Token::Plus) => BinOp::Add,
                Some(Token::PlusPlus) => BinOp::Concat,
                _ => break,
            };
            self.bump();
            let rhs = self.postfix()?;
            lhs = self.arena.push(Expr::BinOp(op, lhs, rhs));
        }
        Ok(lhs)
    }

    // postfix := atom ("(" expr ")")*  -- supports curried calls f(a)(b)
    fn postfix(&mut self) -> Result<ExprRef, String> {
        let mut e = self.atom()?;
        while matches!(self.peek(), Some(Token::LParen)) {
            self.bump();
            let arg = self.expr()?;
            self.expect(&Token::RParen)?;
            e = self.arena.push(Expr::App(e, arg));
        }
        Ok(e)
    }

    // Peels off a run of leading `let ... in` / `fun ... ->` prefixes
    // iteratively -- a token peek per iteration, not a recursive call --
    // so a long chain of either (`let x1 = .. in let x2 = .. in ...`, or
    // a deeply curried `fun a -> fun b -> fun c -> ...`) costs O(1)
    // native stack instead of O(chain length). That chain shape is
    // exactly what used to overflow the stack on deeply nested/generated
    // source (see lib.rs's run_source). The terminal body/value once the
    // chain ends is parsed with an ordinary self.expr() call, same as the
    // original recursive version -- only the "is there another prefix"
    // bookkeeping moved out of the call stack, not the grammar itself.
    fn atom(&mut self) -> Result<ExprRef, String> {
        let mut pending = Vec::new();
        loop {
            match self.peek() {
                Some(Token::Let) => {
                    self.bump();
                    let rec = matches!(self.peek(), Some(Token::Rec));
                    if rec {
                        self.bump();
                    }
                    let var = self.ident()?;
                    let ann = self.opt_annotation()?;
                    self.expect(&Token::Equals)?;
                    let val = self.expr()?;
                    self.expect(&Token::In)?;
                    pending.push(PendingBinder::Let { var, ann, rec, val });
                }
                Some(Token::Fun) => {
                    self.bump();
                    let param = self.ident()?;
                    let ann = self.opt_annotation()?;
                    self.expect(&Token::Arrow)?;
                    pending.push(PendingBinder::Fun { param, ann });
                }
                _ => break,
            }
        }

        let mut result = if pending.is_empty() { self.atom_leaf()? } else { self.expr()? };
        for binder in pending.into_iter().rev() {
            result = match binder {
                PendingBinder::Let { var, ann, rec, val } => self.arena.push(Expr::Let(var, ann, val, result, rec)),
                PendingBinder::Fun { param, ann } => self.arena.push(Expr::Lambda(param, ann, result)),
            };
        }
        Ok(result)
    }

    // Every atom form except `let`/`fun`, which `atom` handles iteratively
    // above. Reached only once no more chain prefix remains.
    fn atom_leaf(&mut self) -> Result<ExprRef, String> {
        match self.bump() {
            Some(Token::Int(n)) => Ok(self.arena.push(Expr::Int(n))),
            Some(Token::True) => Ok(self.arena.push(Expr::Bool(true))),
            Some(Token::False) => Ok(self.arena.push(Expr::Bool(false))),
            Some(Token::Str(s)) => Ok(self.arena.push(Expr::Str(s))),
            Some(Token::Ident(name)) => Ok(self.arena.push(Expr::Var(name))),

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
                Ok(self.arena.push(Expr::ListLit(items)))
            }

            Some(Token::If) => {
                let cond = self.expr()?;
                self.expect(&Token::Then)?;
                let then_ = self.expr()?;
                self.expect(&Token::Else)?;
                let else_ = self.expr()?;
                Ok(self.arena.push(Expr::If(cond, then_, else_)))
            }

            Some(Token::Perform) => {
                let effect = self.ident()?;
                self.expect(&Token::LParen)?;
                let payload = self.expr()?;
                self.expect(&Token::RParen)?;
                Ok(self.arena.push(Expr::Perform(effect, payload)))
            }

            // handle <body> with <handler-expr>
            Some(Token::Handle) => {
                let body = self.expr()?;
                self.expect(&Token::With)?;
                let handler = self.expr()?;
                Ok(self.arena.push(Expr::Handle { body, handler }))
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
                    return Err(format!(
                        "handler {effect}: payload and resume binders must have different names, both named {payload_var:?}"
                    ));
                }
                self.expect(&Token::RParen)?;
                self.expect(&Token::Arrow)?;
                let body = self.expr()?;
                Ok(self.arena.push(Expr::MakeHandler { effect, payload_var, resume_var, body }))
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
                Ok(self.arena.push(Expr::Match(scrutinee, Rc::new(arms))))
            }

            other => Err(format!("unexpected token: {other:?}")),
        }
    }
}
