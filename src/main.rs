use renno::{parser, run_source};

fn run_file(path: &str) {
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            std::process::exit(1);
        }
    };
    match run_source(&src) {
        Ok(v) => println!("{v}"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

// A parse error from the parser simply running out of tokens mid-construct
// (an unclosed `let ... in`, an unmatched paren/bracket, a `match` waiting
// on its next arm) always ends its summary line in the bare word "None" --
// Option<Token>::None's own Debug text -- since every "expected X, found
// Y"/"unexpected token: Y" site prints `{other:?}` as the LAST thing in
// its message, on the exact token bump() returned. A real (even a
// misplaced) token always prints as `Some(...)`, ending in `)`, never in
// the bare word "None" -- including one that happens to BE the identifier
// "None" (renno's own ADT convention for a nullary constructor), which
// prints as `Some(Ident("None"))`. That distinction is what lets the REPL
// tell "still typing" apart from "this is just wrong" with a plain string
// check, no second parser mode needed: keep buffering on this, report
// anything else immediately.
//
// Only the FIRST line matters -- format_error appends a source snippet
// after it, so a plain `contains`/`ends_with` on the whole (multi-line)
// error string would miss this or match spuriously inside the snippet.
fn looks_incomplete(parse_err: &str) -> bool {
    parse_err.lines().next().is_some_and(|line| line.ends_with("None"))
}

fn repl() {
    use std::io::{self, BufRead, Write};
    println!("renno REPL -- Ctrl+D to exit");
    let stdin = io::stdin();
    let mut buffer = String::new();
    loop {
        print!("{}", if buffer.is_empty() { "renno> " } else { "...... " });
        if io::stdout().flush().is_err() {
            break;
        }
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF -- drops any unfinished buffered input
            Ok(_) => {}
            Err(_) => break,
        }
        let blank = line.trim().is_empty();

        if buffer.is_empty() {
            if blank {
                continue;
            }
            buffer.push_str(&line);
            match parser::parse(&buffer) {
                Ok(_) => {
                    let src = std::mem::take(&mut buffer);
                    run_and_print(&src);
                }
                Err(e) if looks_incomplete(&e) => {} // wait for more input
                Err(e) => {
                    println!("error: {e}");
                    buffer.clear();
                }
            }
            continue;
        }

        // Already mid multi-line input: a blank line is the explicit "run
        // it now" signal, checked BEFORE trying to parse -- a `match` with
        // only its first arm typed so far, say, already parses as a
        // complete (if probably not what was meant) expression, and there
        // is no syntactic way to tell "done" from "about to add `| ...`"
        // apart. Rather than guess, wait for this the same way a real
        // multi-line REPL session (Python's included) does: buffer keeps
        // growing until either it's obviously broken (a real parse error,
        // reported immediately) or the user says "go" with a blank line.
        if blank {
            let src = std::mem::take(&mut buffer);
            run_and_print(&src);
            continue;
        }
        buffer.push_str(&line);
        if let Err(e) = parser::parse(&buffer) {
            if !looks_incomplete(&e) {
                println!("error: {e}");
                buffer.clear();
            }
        }
    }
}

fn run_and_print(src: &str) {
    match run_source(src) {
        Ok(v) => println!("{v}"),
        Err(e) => println!("error: {e}"),
    }
}

fn main() {
    match std::env::args().nth(1) {
        Some(path) => run_file(&path),
        None => repl(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eof_mid_let_looks_incomplete() {
        let err = parser::parse("let x = 1 in").unwrap_err();
        assert!(looks_incomplete(&err), "unexpected message: {err}");
    }

    #[test]
    fn eof_mid_match_looks_incomplete() {
        let err = parser::parse("match 5 | 1 -> ").unwrap_err();
        assert!(looks_incomplete(&err), "unexpected message: {err}");
    }

    #[test]
    fn a_real_syntax_error_does_not_look_incomplete() {
        let err = parser::parse("let x = in 5").unwrap_err();
        assert!(!looks_incomplete(&err), "unexpected message: {err}");
    }

    #[test]
    fn an_identifier_literally_spelled_none_does_not_look_incomplete() {
        // Regression guard: the heuristic keys off Option<Token>::None's
        // OWN Debug text, not the substring "None" anywhere in the
        // message -- a real (if oddly placed) token must never trip it,
        // including one spelled "None" itself (renno's own ADT convention
        // for a nullary constructor) -- `expect(&Token::Equals)` here
        // finds `Some(Ident("None"))`, not the bare `None` a genuine
        // end-of-input would produce.
        let err = parser::parse("let x None").unwrap_err();
        assert!(err.contains("None"), "test setup didn't hit the case it means to: {err}");
        assert!(!looks_incomplete(&err), "unexpected message: {err}");
    }
}
