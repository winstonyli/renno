use logos::Logos;

use crate::span::Span;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip(r"//[^\n]*", allow_greedy = true))]
pub enum Token {
    #[token("fun")]
    Fun,
    #[token("let")]
    Let,
    #[token("rec")]
    Rec,
    #[token("and")]
    And,
    // A literal, evaluating to a token unique to its own source position
    // (see Expr::Token's own doc comment) -- combined with tuples, this
    // builds a hand-rolled nominal type: two tuples carrying the SAME
    // `opaque`'s token are only ever consistent with each other, never
    // with a same-shaped tuple carrying a DIFFERENT `opaque`'s.
    #[token("opaque")]
    Opaque,
    #[token("in")]
    In,
    #[token("if")]
    If,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("perform")]
    Perform,
    #[token("handle")]
    Handle,
    #[token("with")]
    With,
    #[token("handler")]
    HandlerKw,
    #[token("match")]
    Match,

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token(":")]
    Colon,
    // Cons pattern: `h :: t`, matches a non-empty list. Longer than Colon,
    // so logos's longest-match rule prefers this over `:` `:` on "::".
    #[token("::")]
    ColonColon,
    #[token("|")]
    Pipe,
    // Boolean and/or/not. Longer than a bare `|`, so logos's longest-
    // match rule prefers this over two separate `|` tokens -- harmless in
    // practice anyway, since a lone `|` only ever appears between match
    // arms, never inside an expression, so it can't collide with a valid
    // `||`. `and`/`or` as WORDS are already taken (`let rec f = ... and
    // g = ...`), hence symbolic operators here, matching the rest of
    // renno's operator set (+ - * / == < ++ ::).
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("!")]
    Bang,
    #[token("->")]
    Arrow,
    #[token("=")]
    Equals,
    #[token("++")]
    PlusPlus,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("==")]
    EqEq,
    #[token("!=")]
    BangEq,
    #[token("<")]
    Lt,
    #[token("<=")]
    LtEq,
    #[token(">")]
    Gt,
    #[token(">=")]
    GtEq,

    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Int(i64),

    // Digits required on BOTH sides of the dot -- no `3.`/`.5` -- so this
    // never collides with Dot (`.field` access): `3.field` has no digit
    // after the dot, so this regex simply doesn't match there and the
    // lexer falls through to Int("3") + Dot + Ident("field") exactly as
    // before Float existed. Where a number IS followed by digits after a
    // dot, this regex is strictly longer than Int's own bare-digits match
    // at the same position, so logos's longest-match rule prefers this.
    #[regex(r"[0-9]+\.[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),

    // `"..."` with a small set of escapes (\" \\ \n \t). Slice includes
    // the surrounding quotes; unescape() strips them and processes escapes.
    #[regex(r#""([^"\\]|\\.)*""#, |lex| unescape(lex.slice()))]
    Str(String),

    // Keyword #[token]s above take priority over this regex on exact
    // matches (logos: equal-length match, token literal wins over regex).
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),
}

fn unescape(raw: &str) -> Option<String> {
    let inner = &raw[1..raw.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next()? {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'n' => out.push('\n'),
                't' => out.push('\t'),
                _ => return None,
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

pub fn tokenize(src: &str) -> Result<Vec<(Token, Span)>, String> {
    let mut lexer = Token::lexer(src);
    let mut tokens = Vec::new();
    while let Some(result) = lexer.next() {
        let span = lexer.span();
        let span = Span { start: span.start, end: span.end };
        match result {
            Ok(tok) => tokens.push((tok, span)),
            Err(_) => {
                let bad = src[span.start..].chars().next().unwrap_or(' ');
                let msg = if bad == '"' { "unterminated string literal".to_string() } else { format!("unexpected character `{bad}`") };
                return Err(span.format_error(src, &msg));
            }
        }
    }
    Ok(tokens)
}

impl Token {
    // Source text of the fixed tokens the parser asks for by name (`expect`), for
    // "expected `)`" messages. `fixed_token_text_lexes_back_to_its_token` keeps this
    // table honest against the `#[token]` attributes above.
    pub fn text(&self) -> &'static str {
        match self {
            Token::Arrow => "->",
            Token::Comma => ",",
            Token::Else => "else",
            Token::Equals => "=",
            Token::In => "in",
            Token::LParen => "(",
            Token::Pipe => "|",
            Token::RBrace => "}",
            Token::RBracket => "]",
            Token::RParen => ")",
            Token::Then => "then",
            Token::With => "with",
            _ => "a different token",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_token_text_lexes_back_to_its_token() {
        for t in [
            Token::Arrow, Token::Comma, Token::Else, Token::Equals, Token::In, Token::LParen,
            Token::Pipe, Token::RBrace, Token::RBracket, Token::RParen, Token::Then, Token::With,
        ] {
            let lexed = tokenize(t.text()).unwrap();
            assert_eq!(lexed.len(), 1);
            assert_eq!(lexed[0].0, t);
        }
    }
}
