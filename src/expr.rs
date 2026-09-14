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
    Handle {
        body: Rc<Expr>,
        effect: String,
        payload_var: String,
        resume_var: String,
        handler: Rc<Expr>,
        // deep: handler reinstalls itself around the resumed continuation,
        // so a later occurrence of the same effect inside it is caught
        // again by this handler. shallow: handles one occurrence only.
        deep: bool,
    },
}
