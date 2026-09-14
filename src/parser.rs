use std::rc::Rc;

use crate::expr::Expr;
use crate::lexer::{tokenize, Token};

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

    // expr := add
    fn expr(&mut self) -> Result<Expr, String> {
        self.add()
    }

    // add := postfix ("+" postfix)*  (left-associative)
    fn add(&mut self) -> Result<Expr, String> {
        let mut lhs = self.postfix()?;
        while matches!(self.peek(), Some(Token::Plus)) {
            self.bump();
            let rhs = self.postfix()?;
            lhs = Expr::Add(Rc::new(lhs), Rc::new(rhs));
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
            Some(Token::Ident(name)) => Ok(Expr::Var(name)),

            Some(Token::Fun) => {
                let param = self.ident()?;
                self.expect(&Token::Arrow)?;
                let body = self.expr()?;
                Ok(Expr::Lambda(param, Rc::new(body)))
            }

            Some(Token::Let) => {
                let var = self.ident()?;
                self.expect(&Token::Equals)?;
                let val = self.expr()?;
                self.expect(&Token::In)?;
                let body = self.expr()?;
                Ok(Expr::Let(var, Rc::new(val), Rc::new(body)))
            }

            Some(Token::If0) => {
                let cond = self.expr()?;
                self.expect(&Token::Then)?;
                let then_ = self.expr()?;
                self.expect(&Token::Else)?;
                let else_ = self.expr()?;
                Ok(Expr::If0(Rc::new(cond), Rc::new(then_), Rc::new(else_)))
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
