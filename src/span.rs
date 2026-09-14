// A byte-offset range into the original source text -- a small Copy struct
// of our own rather than std::ops::Range<usize> (which isn't Copy, since it
// doubles as an Iterator and consuming it advances `start`; that's a real
// footgun for something meant only as an immutable position marker passed
// around and cloned freely). Converted to a human-readable line:col only at
// the point an error is finally formatted (see line_col) -- carrying one
// around costs nothing until then.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    // 1-indexed (line, column), the way editors/compilers report it. Scans
    // `src` up to `self.start` once -- fine for error reporting's one-shot
    // use, not meant for repeated or hot-path calls.
    pub fn line_col(&self, src: &str) -> (usize, usize) {
        let mut line = 1;
        let mut col = 1;
        for ch in src[..self.start.min(src.len())].chars() {
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    // The source line this span starts on, with a `^^^` caret underneath
    // pointing at its columns -- rustc's classic single-span look, hand-
    // rolled rather than pulling in a crate (ariadne, annotate-snippets)
    // since every error here carries exactly one Span, never several
    // related ones. A span that continues past this line's end (rare --
    // most renno constructs are short) just underlines to the line's own
    // end rather than spanning multiple lines.
    //
    // Revisit with a real diagnostics crate if/when an error wants to show
    // MULTIPLE labeled spans at once -- e.g. "expected Int here (from this
    // annotation) ... found Bool here" pointing at two different places
    // (coerce already has both the annotation's origin and the mismatched
    // value's own span in scope, just doesn't use both today). Hand-
    // rolling that well -- aligning gutters across snippets, connecting
    // multi-line labels -- is exactly what annotate-snippets (rustc's own
    // choice) or ariadne already solve; a second hand-rolled span in one
    // message isn't worth reinventing that.
    pub fn snippet(&self, src: &str) -> String {
        let (line_no, col) = self.line_col(src);
        let line_text = src.lines().nth(line_no - 1).unwrap_or("");
        let remaining_on_line = line_text.len().saturating_sub(col - 1);
        let span_len = self.end.saturating_sub(self.start).max(1);
        let caret_len = span_len.min(remaining_on_line).max(1);

        let gutter_width = line_no.to_string().len();
        let blank_gutter = " ".repeat(gutter_width);
        let caret_line = format!("{}{}", " ".repeat(col - 1), "^".repeat(caret_len));

        format!(
            "{bg} |\n{ln:>gw$} | {lt}\n{bg} | {cl}",
            bg = blank_gutter,
            ln = line_no,
            gw = gutter_width,
            lt = line_text,
            cl = caret_line,
        )
    }

    // The full one-line-summary-plus-snippet message every error path
    // (lex, parse, typecheck) renders through -- one place to keep the
    // three consistent.
    pub fn format_error(&self, src: &str, msg: &str) -> String {
        let (line, col) = self.line_col(src);
        format!("line {line}, column {col}: {msg}\n{}", self.snippet(src))
    }
}
