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
    #[token("data")]
    Data,
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
    #[token("Int")]
    TyInt,
    #[token("Bool")]
    TyBool,
    #[token("Str")]
    TyStr,
    #[token("Dyn")]
    TyDyn,

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
    #[token("==")]
    EqEq,
    #[token("<")]
    Lt,

    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Int(i64),

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
            Err(_) => return Err(span.format_error(src, "lex error")),
        }
    }
    Ok(tokens)
}
