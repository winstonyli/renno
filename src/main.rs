mod cont;
mod env;
mod expr;
mod machine;
mod value;

use std::rc::Rc;

use env::Env;
use expr::Expr;

fn main() {
    // handle {
    //   let x = perform choose 0 in
    //   x + 100
    // } with choose(_, resume) -> resume(1) + resume(2)
    //
    // Multi-shot proof: `resume` is called twice from the handler body.
    // Each call replays the captured continuation (`x + 100`) with a
    // different x, independently -- expect (1+100) + (2+100) = 203.
    let program = Expr::Handle {
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
    };

    let result = machine::run(Rc::new(program), Env::empty());
    println!("result = {}", result.as_int());
    assert_eq!(result.as_int(), 203);
}
