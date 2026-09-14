use std::rc::Rc;

use crate::expr::{BinOp, Expr};
use crate::lexer::{tokenize, Token};
use crate::types::Type;

pub fn parse(src: &str) -> Result<Rc<Expr>, String> {
    let tokens = tokenize(src)?;
    let mut p = Parser { tokens, pos: 0 };
    let e = p.expr()?;
    if p.pos != p.tokens.len() {
        return Err(format!("trailing tokens after expression: {:?}", &p.tokens[p.pos..]));
    }
    Ok(Rc::new(e))
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
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

    // Bare, atom-only: "Int" | "Bool" | "(" fun_type ")". Deliberately does
    // NOT chain "->" at this level -- an annotation site (`fun x: T ->`,
    // `let x: T =`) is always immediately followed by its own "->"/"="
    // token, so a bare trailing arrow here would be ambiguous between
    // "this type continues" and "the annotation just ended". A function
    // type must be parenthesized to disambiguate: `fun f: (Int -> Int) -> ...`.
    fn parse_type(&mut self) -> Result<Type, String> {
        match self.bump() {
            Some(Token::TyInt) => Ok(Type::Int),
            Some(Token::TyBool) => Ok(Type::Bool),
            Some(Token::LParen) => {
                let t = self.parse_fun_type()?;
                self.expect(&Token::RParen)?;
                Ok(t)
            }
            other => Err(format!("expected a type, found {other:?}")),
        }
    }

    // fun_type := type ("->" fun_type)?  (right-associative) -- only
    // reachable from inside parens, where ")" unambiguously ends it.
    fn parse_fun_type(&mut self) -> Result<Type, String> {
        let atom = self.parse_type()?;
        if matches!(self.peek(), Some(Token::Arrow)) {
            self.bump();
            let ret = self.parse_fun_type()?;
            Ok(Type::Fun(Rc::new(atom), Rc::new(ret)))
        } else {
            Ok(atom)
        }
    }

    // expr := cmp
    fn expr(&mut self) -> Result<Expr, String> {
        self.cmp()
    }

    // cmp := add (("==" | "<") add)?  -- non-associative, one comparison
    fn cmp(&mut self) -> Result<Expr, String> {
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
                Ok(Expr::BinOp(op, Rc::new(lhs), Rc::new(rhs)))
            }
            None => Ok(lhs),
        }
    }

    // add := postfix ("+" postfix)*  (left-associative)
    fn add(&mut self) -> Result<Expr, String> {
        let mut lhs = self.postfix()?;
        while matches!(self.peek(), Some(Token::Plus)) {
            self.bump();
            let rhs = self.postfix()?;
            lhs = Expr::BinOp(BinOp::Add, Rc::new(lhs), Rc::new(rhs));
        }
        Ok(lhs)
    }

    // postfix := atom ("(" expr ")")*  -- supports curried calls f(a)(b)
    fn postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.atom()?;
        while matches!(self.peek(), Some(Token::LParen)) {
            self.bump();
            let arg = self.expr()?;
            self.expect(&Token::RParen)?;
            e = Expr::App(Rc::new(e), Rc::new(arg));
        }
        Ok(e)
    }

    fn atom(&mut self) -> Result<Expr, String> {
        match self.bump() {
            Some(Token::Int(n)) => Ok(Expr::Int(n)),
            Some(Token::True) => Ok(Expr::Bool(true)),
            Some(Token::False) => Ok(Expr::Bool(false)),
            Some(Token::Ident(name)) => Ok(Expr::Var(name)),

            Some(Token::Fun) => {
                let param = self.ident()?;
                let ann = self.opt_annotation()?;
                self.expect(&Token::Arrow)?;
                let body = self.expr()?;
                Ok(Expr::Lambda(param, ann, Rc::new(body)))
            }

            Some(Token::Let) => {
                let var = self.ident()?;
                let ann = self.opt_annotation()?;
                self.expect(&Token::Equals)?;
                let val = self.expr()?;
                self.expect(&Token::In)?;
                let body = self.expr()?;
                Ok(Expr::Let(var, ann, Rc::new(val), Rc::new(body)))
            }

            Some(Token::If) => {
                let cond = self.expr()?;
                self.expect(&Token::Then)?;
                let then_ = self.expr()?;
                self.expect(&Token::Else)?;
                let else_ = self.expr()?;
                Ok(Expr::If(Rc::new(cond), Rc::new(then_), Rc::new(else_)))
            }

            Some(Token::Perform) => {
                let effect = self.ident()?;
                self.expect(&Token::LParen)?;
                let payload = self.expr()?;
                self.expect(&Token::RParen)?;
                Ok(Expr::Perform(effect, Rc::new(payload)))
            }

            // handle <body> with <handler-expr>
            Some(Token::Handle) => {
                let body = self.expr()?;
                self.expect(&Token::With)?;
                let handler = self.expr()?;
                Ok(Expr::Handle { body: Rc::new(body), handler: Rc::new(handler) })
            }

            // handler <effect>(<payload_var>, <resume_var>) -> <body>
            Some(Token::HandlerKw) => {
                let effect = self.ident()?;
                self.expect(&Token::LParen)?;
                let payload_var = self.ident()?;
                self.expect(&Token::Comma)?;
                let resume_var = self.ident()?;
                self.expect(&Token::RParen)?;
                self.expect(&Token::Arrow)?;
                let body = self.expr()?;
                Ok(Expr::MakeHandler { effect, payload_var, resume_var, body: Rc::new(body) })
            }

            Some(Token::LParen) => {
                let e = self.expr()?;
                self.expect(&Token::RParen)?;
                Ok(e)
            }

            other => Err(format!("unexpected token: {other:?}")),
        }
    }
}
