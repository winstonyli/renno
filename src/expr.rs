use std::rc::Rc;

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Var(String),
    Lambda(String, Rc<Expr>),
    App(Rc<Expr>, Rc<Expr>),
    Let(String, Rc<Expr>, Rc<Expr>),
    Add(Rc<Expr>, Rc<Expr>),
    If0(Rc<Expr>, Rc<Expr>, Rc<Expr>),
    Perform(String, Rc<Expr>),
    // `handler` evaluates to a Value::Handler. Handle itself carries no
    // deep/shallow flag -- that lives on the handler value, set by the
    // `deep`/`shallow` builtins (see env::prelude), not on this AST node.
    Handle {
        body: Rc<Expr>,
        handler: Rc<Expr>,
    },
    // Builds the base (shallow) handler value for one effect clause.
    // `fun (payload_var) (resume_var) -> body`, tagged with the effect name.
    MakeHandler {
        effect: String,
        payload_var: String,
        resume_var: String,
        body: Rc<Expr>,
    },
}
