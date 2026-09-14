mod cont;
mod env;
mod expr;
mod machine;
mod value;

use std::rc::Rc;

use env::Env;
use expr::Expr;

// handle {
//   let x = perform choose 0 in
//   x + 100
// } with choose(_, resume) -> resume(1) + resume(2)
//
// Multi-shot proof: `resume` is called twice from the handler body.
// Each call replays the captured continuation (`x + 100`) with a
// different x, independently -- expect (1+100) + (2+100) = 203.
fn build_multi_shot_demo() -> Rc<Expr> {
    Rc::new(Expr::Handle {
        body: Rc::new(Expr::Let(
            "x".into(),
            Rc::new(Expr::Perform("choose".into(), Rc::new(Expr::Int(0)))),
            Rc::new(Expr::Add(Rc::new(Expr::Var("x".into())), Rc::new(Expr::Int(100)))),
        )),
        effect: "choose".into(),
        payload_var: "_ignored".into(),
        resume_var: "resume".into(),
        handler: Rc::new(Expr::Add(
            Rc::new(Expr::App(Rc::new(Expr::Var("resume".into())), Rc::new(Expr::Int(1)))),
            Rc::new(Expr::App(Rc::new(Expr::Var("resume".into())), Rc::new(Expr::Int(2)))),
        )),
        deep: true,
    })
}

fn main() {
    let result = machine::run(build_multi_shot_demo(), Env::empty());
    println!("result = {}", result.as_int());
    assert_eq!(result.as_int(), 203);
}

// handle { let x = perform choose 0 in let y = perform choose 0 in x + y }
// with choose(_, resume) -> resume(1)
//
// Two *sequential* occurrences of the same effect (not multi-shot -- each
// is resumed once). Deep: the reinstalled handler catches the second
// occurrence too -> x=1, y=1 -> 2. Shallow: handler is consumed by the
// first occurrence, the second escapes unhandled -> panics.
fn two_sequential_performs(deep: bool) -> Rc<Expr> {
    Rc::new(Expr::Handle {
        body: Rc::new(Expr::Let(
            "x".into(),
            Rc::new(Expr::Perform("choose".into(), Rc::new(Expr::Int(0)))),
            Rc::new(Expr::Let(
                "y".into(),
                Rc::new(Expr::Perform("choose".into(), Rc::new(Expr::Int(0)))),
                Rc::new(Expr::Add(Rc::new(Expr::Var("x".into())), Rc::new(Expr::Var("y".into())))),
            )),
        )),
        effect: "choose".into(),
        payload_var: "_ignored".into(),
        resume_var: "resume".into(),
        handler: Rc::new(Expr::App(Rc::new(Expr::Var("resume".into())), Rc::new(Expr::Int(1)))),
        deep,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_shot_resume_replays_independently() {
        let result = machine::run(build_multi_shot_demo(), Env::empty());
        assert_eq!(result.as_int(), 203);
    }

    #[test]
    fn deep_handler_catches_repeat_effect() {
        let result = machine::run(two_sequential_performs(true), Env::empty());
        assert_eq!(result.as_int(), 2);
    }

    #[test]
    #[should_panic(expected = "unhandled effect: choose")]
    fn shallow_handler_does_not_catch_repeat_effect() {
        machine::run(two_sequential_performs(false), Env::empty());
    }
}
