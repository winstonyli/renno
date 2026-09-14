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
}
