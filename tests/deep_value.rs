// The binary prints and drops a result nested far deeper than its own main thread's
// stack allows (the process used to abort at about 4,000 levels).
use std::process::Command;

#[test]
fn the_binary_prints_a_deeply_nested_result() {
    let depth = 100_000;
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("deep_value.rn");
    let src = format!("let rec nest = fun n -> fun acc -> if n == 0 then acc else nest(n - 1)([acc]) in nest({depth})(1)");
    std::fs::write(&path, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_renno")).arg(&path).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    assert_eq!(text.trim_end(), format!("{}1{}", "[".repeat(depth), "]".repeat(depth)));
}
