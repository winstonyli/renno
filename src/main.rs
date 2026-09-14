use renno::run_source;

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

fn repl() {
    use std::io::{self, BufRead, Write};
    println!("renno REPL -- Ctrl+D to exit");
    let stdin = io::stdin();
    loop {
        print!("renno> ");
        if io::stdout().flush().is_err() {
            break;
        }
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match run_source(line) {
            Ok(v) => println!("{v}"),
            Err(e) => println!("error: {e}"),
        }
    }
}

fn main() {
    match std::env::args().nth(1) {
        Some(path) => run_file(&path),
        None => repl(),
    }
}
