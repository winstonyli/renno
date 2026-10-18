# Parked work

One list of deferred work. Each entry: what / why parked / trigger / evidence / status. Evidence cites git-ignored working docs under `docs/superpowers/plans/` (backlog `2026-10-16-paring-and-unification-backlog.md`) unless a file is named; dates are when last checked. Seeded 2026-10-17 from the backlog, reports and memory; line numbers in the backlog are stale, re-find before use.

## Language and type system
- **U1 effect rows get real unification** (one row sort, open rows with a tail; replaces bind_row_vars, resolve_row, row_consistent, the union-to-Dyn collapse, variable-handler Dyn). Why: large; first decide whether row-polymorphic records are ever wanted. Trigger: row-polymorphic records wanted, or P5. Evidence: backlog U1, 2026-10-16. Status: parked.
- **P5 perform/handler/handle trio** collapse. Why: no net win until U1. Trigger: U1 done. Status: held for U1.
- **P6 index sub-grammar into ordinary Int expressions.** Why: blocked on the static-Type design (L). Trigger: that design lands. Evidence: backlog P6. Status: parked.
- **P7 literal patterns as guard sugar.** Why: low payoff (Bool exhaustiveness, switch speed). Trigger: a user wants either. Status: parked.
- **U3 If/Match/ListLit joins fall to Dyn** (one `join` producing a Union). Why: may add static rejections in the corpus; check first. Trigger: a program loses precision to Dyn. Evidence: backlog U3. Status: parked.
- **U4 annotation-variable escape check** ("fixed by the signature"). Why: M; known-limitation pins in lib.rs tests. Trigger: a wrong accept/reject report. Evidence: backlog U4. Status: parked.
- **U6 or-patterns and negative literal patterns.** Why: works against P7. Trigger: literal patterns stay and are wanted. Status: parked.
- **U7 containers not cast at Dyn crossings** (one lazy structural wrapper through List/Tuple/Record via Value::Contract). Why: L, soundness. Trigger: an unsound crossing shows up in a real program. Evidence: backlog U7. Status: parked.
- **U9 index solving** (index_solve_unit first, ~170 lines; Fourier-Motzkin later). Why: L. Trigger: programs needing inequality reasoning on `Vec(n)`. Evidence: memory note (no FM code on main, verified 2026-10-08). Status: parked.
- **`let rec n = (fun x -> [x])(n) in n` prints `[n]`** (placeholder escapes into a value). Decided: reject. Why: needs its own branch. Trigger: next typecheck pass over let rec. Status: parked.
- **Smaller shipped-feature leftovers:** build_cast (P3b), L1/L5 and shared `Cont::nil` (P4b), F8/E2/R-vs-R (recursive types), nested variable n / index solving / S5 (gradual unknown), CC2/CC3 and SameLen (length slots), e6 shallow re-perform and variable handlers (clause rows), FieldAccess/Pattern::Record row-polymorphism gap. Why: each small or blocked on U1/row-poly. Trigger: a user case. Evidence: the per-feature memory notes. Status: parked.

## Performance and robustness
- **U8b follow-up: remaining check-time quadratics** (poly_chain2, mutual_group, fnchain, nested_lam; all under half a second at plausible sizes). Why: low value. Trigger: a real program over ~2 s check time. Evidence: u8-superlinear-remeasure-2026-10-16.md, 2026-10-17 idle re-time. Status: parked. (U8b itself, level-based generalization, shipped on main 96e9024; residual cost +7-10% on the largest generalized-let chains, `gone20000`/`gone50000`, from persistent-map allocation and drop; accepted.)
- **D4 strict cons/snoc alternation is still quadratic** (`deque_5000` 1.28x). Why: two-ended claim block cannot serve both ends alternately. Trigger: a deque-shaped workload. Evidence: v2-phase-l-t1-report.md. Status: parked.
- **D9 seals are not transferred on copy-grow; M1 retention** (a discarded append keeps its value alive in a shared block). Why: slight cost, no user. Trigger: memory growth report. Status: parked.
- **U11b persistent map/filter accumulators; U11c shallow handler re-handling O(n^2)** with a non-tail resume. Why: iterate.rs hot path, own bench risk, no user. Trigger: a program with map/filter under a non-tail or multi-shot resume. Evidence: u11-report.md. Status: parked.
- **Update-at-index / record update via refcount `Rc::make_mut`.** Why: the claim engine cannot do it and no such builtin exists. Trigger: such a feature is wanted. Evidence: 2026-10-16-inplace-mutation-design.md. Status: parked.
- **Early-group residual** (933 MB at nesting d=1000). Why: accepted cost. Trigger: a real program that hits it. Evidence: spec section 16. Status: parked.
- **Run-time-built deep types and values still abort around 5M levels; the debug 70k-deep `Dyn` cast aborts.** Why: only reachable by generated programs. Trigger: such a program in practice. Evidence: u10-report.md. Status: parked.

## Decided against
- **P4 write map/filter/fold/... in renno itself:** the renno versions are 2-5x slower. Status: decided against (2026-10-17).
- **P8 `let rec` without `and`:** groups are load-bearing. Status: decided against.
- **Unify Tuple onto the keyed Record representation:** Tuple stays `Value::List`. Status: decided against (reconfirmed twice).
- **Ownership or substructural typing:** conflicts with sharing and multi-shot continuations. Status: decided against.
- **Persistent sequences (imbl RRB) as the list type:** 3-30x slower on cons/head/tail/index. Evidence: persistent-spike-report.md. Status: decided against.
- **Generated chains over 2000 levels (else-if, `+`, `++`, `||`, `::`, parens) are rejected** by `parser::MAX_NESTING`. Documented in the README. Status: decided against lifting it.
