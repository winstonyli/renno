// `import("path")` expansion: see docs/superpowers/specs/2026-10-20-imports-design.md.
use std::path::{Path, PathBuf};

use crate::expr::{Arena, Expr, ExprRef, SpanMap};
use crate::resolve::{self, VarRef};
use crate::span::Span;

// The virtual source: the main text, then each module's text after a newline, so every span is an
// offset into `text`; `files` maps a base offset to the file's display path (main first, base 0).
pub struct Sources {
    pub text: String,
    pub files: Vec<(usize, String)>,
}

// A canonical Windows path starts `\\?\`; messages show it without.
pub fn display_path(p: &Path) -> String {
    let s = p.display().to_string();
    s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s)
}

impl Sources {
    pub fn new(main: &str) -> Self {
        Sources { text: main.to_string(), files: vec![(0, String::new())] }
    }

    pub fn format_error(&self, span: Span, msg: &str) -> String {
        if self.files.len() == 1 {
            return span.format_error(&self.text, msg);
        }
        let i = self.files.iter().rposition(|(b, _)| *b <= span.start).unwrap_or(0);
        let (base, path) = &self.files[i];
        let end = self.files.get(i + 1).map_or(self.text.len(), |(b, _)| b - 1);
        let local = &self.text[*base..end.max(*base)];
        let s = Span { start: span.start - base, end: span.end.saturating_sub(*base) };
        let (line, col) = s.line_col(local);
        let place = if path.is_empty() { String::new() } else { format!("{path}:") };
        format!("{place}line {line}, column {col}: {msg}\n{}", s.snippet(local))
    }
}

// Splices every `import("p")` under `root` (a prelude `import` applied to a string literal) with the
// file's expression. `dir` is the base directory; `main` the main file's canonical path, if any.
pub fn expand(arena: &mut Arena, spans: &mut SpanMap, root: ExprRef, sources: &mut Sources, dir: &Path, main: Option<PathBuf>) -> Result<(), String> {
    // Cheap guard: no `import` name anywhere means nothing to resolve.
    if !arena.values().any(|e| matches!(e, Expr::Var(n) if n == "import")) {
        return Ok(());
    }
    expand_tree(arena, spans, root, sources, dir, &mut main.into_iter().collect())
}

fn reachable(arena: &Arena, root: ExprRef) -> Vec<ExprRef> {
    let (mut out, mut work, mut kids) = (Vec::new(), vec![root], Vec::new());
    while let Some(n) = work.pop() {
        out.push(n);
        kids.clear();
        arena[n].children(&mut kids);
        work.extend(kids.iter().copied());
    }
    out
}

fn expand_tree(arena: &mut Arena, spans: &mut SpanMap, root: ExprRef, sources: &mut Sources, dir: &Path, stack: &mut Vec<PathBuf>) -> Result<(), String> {
    let import_idx = crate::env::PRELUDE.iter().position(|(n, _)| *n == "import").expect("import is in the prelude") as u32;
    let resolved = resolve::resolve(arena, root);
    let mut calls = Vec::new();
    for n in reachable(arena, root) {
        if let Expr::App(f, a) = &arena[n] {
            if let (Expr::Var(_), Expr::Str(p)) = (&arena[*f], &arena[*a]) {
                if resolved.get(*f) == Some(VarRef::Prelude(import_idx)) {
                    calls.push((n, p.clone()));
                }
            }
        }
    }
    for (call, path) in calls {
        let canon = dir.join(&path).canonicalize().map_err(|e| sources.format_error(spans[call], &format!("import \"{path}\": {e}")))?;
        if stack.contains(&canon) {
            let chain: Vec<String> = stack.iter().chain([&canon]).map(|p| display_path(p)).collect();
            return Err(sources.format_error(spans[call], &format!("import cycle: {}", chain.join(" -> "))));
        }
        let text = std::fs::read_to_string(&canon).map_err(|e| sources.format_error(spans[call], &format!("import \"{path}\": {e}")))?;
        let (first, base) = (arena.len(), sources.text.len() + 1);
        let mroot = crate::parser::parse_module(arena, spans, &text).map_err(|e| format!("{}:{e}", display_path(&canon)))?;
        for s in spans.values_mut().skip(first) {
            s.start += base;
            s.end += base;
        }
        sources.text.push('\n');
        sources.text.push_str(&text);
        sources.files.push((base, display_path(&canon)));
        stack.push(canon.clone());
        expand_tree(arena, spans, mroot, sources, canon.parent().unwrap_or(dir), stack)?;
        stack.pop();
        // Hygiene: the module must be closed over the prelude; its prelude names become `#name`.
        let mres = resolve::resolve(arena, mroot);
        for v in reachable(arena, mroot) {
            if let Expr::Var(name) = &arena[v] {
                match mres.get(v) {
                    Some(VarRef::Unbound) => {
                        return Err(sources.format_error(spans[v], &format!("module {} refers to undefined name `{name}`", display_path(&canon))));
                    }
                    Some(VarRef::Prelude(_)) if !name.starts_with('#') => {
                        let renamed = format!("#{name}");
                        arena[v] = Expr::Var(renamed);
                    }
                    _ => {}
                }
            }
        }
        let call_span = spans[call];
        arena[call] = arena[mroot].clone();
        spans[call] = call_span;
    }
    Ok(())
}
