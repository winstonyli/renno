mod cont;
mod env;
mod expr;
mod lexer;
mod machine;
mod parser;
mod value;

use std::rc::Rc;

use env::Env;
use expr::{BinOp, Expr};

// handle {
//   let x = perform choose 0 in
//   x + 100
// } with (fun p resume -> resume(1) + resume(2)) tagged for "choose"
//
// Multi-shot proof: `resume` is called twice from the handler body.
// Each call replays the captured continuation (`x + 100`) with a
// different x, independently -- expect (1+100) + (2+100) = 203.
fn build_multi_shot_demo() -> Rc<Expr> {
    Rc::new(Expr::Handle {
        body: Rc::new(Expr::Let(
            "x".into(),
            Rc::new(Expr::Perform("choose".into(), Rc::new(Expr::Int(0)))),
            Rc::new(Expr::BinOp(BinOp::Add, Rc::new(Expr::Var("x".into())), Rc::new(Expr::Int(100)))),
        )),
        handler: Rc::new(Expr::MakeHandler {
            effect: "choose".into(),
            payload_var: "_ignored".into(),
            resume_var: "resume".into(),
            body: Rc::new(Expr::BinOp(
                BinOp::Add,
                Rc::new(Expr::App(Rc::new(Expr::Var("resume".into())), Rc::new(Expr::Int(1)))),
                Rc::new(Expr::App(Rc::new(Expr::Var("resume".into())), Rc::new(Expr::Int(2)))),
            )),
        }),
    })
}

fn main() {
    let src = r#"
        handle
          let x = perform choose(0) in
          x + 100
        with handler choose(p, resume) -> resume(1) + resume(2)
    "#;
    let expr = parser::parse(src).expect("parse failed");
    let result = machine::run(expr, Env::prelude());
    println!("result = {}", result.as_int());
    assert_eq!(result.as_int(), 203);
}

// handle { let x = perform choose 0 in let y = perform choose 0 in x + y }
// with choose(_, resume) -> resume(1)
//
// Two *sequential* occurrences of the same effect (not multi-shot -- each
// is resumed once). deep(handler): the reinstalled handler catches the
// second occurrence too -> x=1, y=1 -> 2. Plain (shallow default): handler
// is consumed by the first occurrence, the second escapes unhandled ->
// panics.
fn two_sequential_performs(deep: bool) -> Rc<Expr> {
    let base_handler = Rc::new(Expr::MakeHandler {
        effect: "choose".into(),
        payload_var: "_ignored".into(),
        resume_var: "resume".into(),
        body: Rc::new(Expr::App(Rc::new(Expr::Var("resume".into())), Rc::new(Expr::Int(1)))),
    });
    let handler = if deep {
        Rc::new(Expr::App(Rc::new(Expr::Var("deep".into())), base_handler))
    } else {
        base_handler
    };
    Rc::new(Expr::Handle {
        body: Rc::new(Expr::Let(
            "x".into(),
            Rc::new(Expr::Perform("choose".into(), Rc::new(Expr::Int(0)))),
            Rc::new(Expr::Let(
                "y".into(),
                Rc::new(Expr::Perform("choose".into(), Rc::new(Expr::Int(0)))),
                Rc::new(Expr::BinOp(BinOp::Add, Rc::new(Expr::Var("x".into())), Rc::new(Expr::Var("y".into())))),
            )),
        )),
        handler,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_shot_resume_replays_independently() {
        let result = machine::run(build_multi_shot_demo(), Env::prelude());
        assert_eq!(result.as_int(), 203);
    }

    #[test]
    fn deep_handler_catches_repeat_effect() {
        let result = machine::run(two_sequential_performs(true), Env::prelude());
        assert_eq!(result.as_int(), 2);
    }

    #[test]
    #[should_panic(expected = "unhandled effect: choose")]
    fn shallow_handler_does_not_catch_repeat_effect() {
        machine::run(two_sequential_performs(false), Env::prelude());
    }

    #[test]
    fn parses_and_runs_multi_shot_demo() {
        let src = r#"
            handle
              let x = perform choose(0) in
              x + 100
            with handler choose(p, resume) -> resume(1) + resume(2)
        "#;
        let expr = parser::parse(src).expect("parse failed");
        let result = machine::run(expr, Env::prelude());
        assert_eq!(result.as_int(), 203);
    }

    #[test]
    fn parses_and_runs_deep_handler() {
        let src = r#"
            handle
              let x = perform choose(0) in
              let y = perform choose(0) in
              x + y
            with deep(handler choose(p, resume) -> resume(1))
        "#;
        let expr = parser::parse(src).expect("parse failed");
        let result = machine::run(expr, Env::prelude());
        assert_eq!(result.as_int(), 2);
    }

    #[test]
    #[should_panic(expected = "unhandled effect: choose")]
    fn parses_and_runs_shallow_handler_panic() {
        let src = r#"
            handle
              let x = perform choose(0) in
              let y = perform choose(0) in
              x + y
            with handler choose(p, resume) -> resume(1)
        "#;
        let expr = parser::parse(src).expect("parse failed");
        machine::run(expr, Env::prelude());
    }

    #[test]
    fn if_true_takes_then_branch() {
        let expr = parser::parse("if 1 < 2 then 10 else 20").unwrap();
        assert_eq!(machine::run(expr, Env::prelude()).as_int(), 10);
    }

    #[test]
    fn if_false_takes_else_branch() {
        let expr = parser::parse("if 2 < 1 then 10 else 20").unwrap();
        assert_eq!(machine::run(expr, Env::prelude()).as_int(), 20);
    }

    #[test]
    fn eq_and_bool_literals() {
        let expr = parser::parse("if 3 == 3 then true else false").unwrap();
        assert!(machine::run(expr, Env::prelude()).as_bool());
    }
}
