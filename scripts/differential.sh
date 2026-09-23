#!/usr/bin/env bash
# Differential check: the pre-change binary vs the current release build over
# every example and corpus program -- stdout, stderr and exit status must match.
# Usage: bash scripts/differential.sh [baseline-binary]
set -u
BASE="${1:-target/baseline/renno-4a58bb3.exe}"
NEW="target/release/renno.exe"
[ -x "$BASE" ] || { echo "missing baseline binary: $BASE"; exit 2; }
[ -x "$NEW" ] || { echo "missing $NEW (run: cargo build --release)"; exit 2; }
tmp="$(mktemp -d)"
fail=0
count=0
for f in examples/*.rn corpus/*.rn; do
  [ -f "$f" ] || continue
  count=$((count + 1))
  "$BASE" "$f" >"$tmp/b.out" 2>"$tmp/b.err"; b=$?
  "$NEW"  "$f" >"$tmp/n.out" 2>"$tmp/n.err"; n=$?
  # main.rs installs no panic hook, so a runtime panic (e.g. an unbound
  # variable, an unhandled effect) goes through Rust's DEFAULT hook, which
  # prints a line like `thread '<unnamed>' (12345) panicked at
  # src\machine.rs:927:22:` to stderr -- both the OS thread id AND the
  # Rust-source file/line/col legitimately differ run to run and binary to
  # binary (confirmed empirically: the SAME binary run twice on the same
  # program prints two different thread ids), so that one line is
  # normalized to a fixed token, identically on both sides, before
  # comparing. Everything else on stderr -- the panic message itself, the
  # `error: ...` line with its renno-source `line N, column M:` span,
  # stdout, and exit status -- is compared byte-for-byte.
  sed -E "s/^thread .* panicked at .*\$/thread panicked at <loc>/" "$tmp/b.err" >"$tmp/b.err.norm"
  sed -E "s/^thread .* panicked at .*\$/thread panicked at <loc>/" "$tmp/n.err" >"$tmp/n.err.norm"
  if [ "$b" != "$n" ] || ! cmp -s "$tmp/b.out" "$tmp/n.out" || ! cmp -s "$tmp/b.err.norm" "$tmp/n.err.norm"; then
    echo "DIFF: $f (exit $b vs $n)"
    diff "$tmp/b.out" "$tmp/n.out" | head -5
    diff "$tmp/b.err.norm" "$tmp/n.err.norm" | head -5
    fail=1
  fi
done
rm -rf "$tmp"
echo "$count programs compared, $([ $fail = 0 ] && echo 'all identical' || echo 'DIFFERENCES FOUND')"
exit $fail
