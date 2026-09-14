use std::rc::Rc;

use crate::types::Type;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Eq,
    Lt,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Bool(bool),
    Var(String),
    // param annotation is optional -- None means Dyn. The typechecker fills
    // this in (or leaves it) when elaborating; the parser fills it in only
    // when the source has an explicit `: Type` annotation.
    Lambda(String, Option<Type>, Rc<Expr>),
    App(Rc<Expr>, Rc<Expr>),
    Let(String, Option<Type>, Rc<Expr>, Rc<Expr>),
    BinOp(BinOp, Rc<Expr>, Rc<Expr>),
    // Runtime type check: produced only by typecheck::elaborate, never by
    // the parser directly. Verifies the inner expr's value matches `Type`
    // before passing it through -- inserted only where a Dyn-typed value
    // flows into an annotated position, so unannotated code pays nothing.
    Check(Type, Rc<Expr>),
    // cond must evaluate to Bool.
    If(Rc<Expr>, Rc<Expr>, Rc<Expr>),
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
