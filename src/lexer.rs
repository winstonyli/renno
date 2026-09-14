use logos::Logos;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n]+")]
#[logos(skip(r"//[^\n]*", allow_greedy = true))]
pub enum Token {
    #[token("fun")]
    Fun,
    #[token("let")]
    Let,
    #[token("in")]
    In,
    #[token("if0")]
    If0,
    #[token("then")]
    Then,
    #[token("else")]
    Else,
    #[token("perform")]
    Perform,
    #[token("handle")]
    Handle,
    #[token("with")]
    With,
    #[token("handler")]
    HandlerKw,

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(",")]
    Comma,
    #[token("->")]
    Arrow,
    #[token("=")]
    Equals,
    #[token("+")]
    Plus,

    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Int(i64),

    // Keyword #[token]s above take priority over this regex on exact
    // matches (logos: equal-length match, token literal wins over regex).
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, String> {
    Token::lexer(src)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "lex error".to_string())
}
