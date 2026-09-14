use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use renno::run_source;

// `let x0 = 0 in let x1 = x0 + 1 in ... in x{n-1}`. Exercises Env/Ctx
// extend+lookup scaling: both are the shared PList (src/plist.rs), an
// Rc-based persistent list with O(1) extend -- this benchmark is the
// direct proof that a chain of n bindings costs O(n), not the O(n^2) the
// old Vec-cloning Ctx used to cost before that fix.
//
// n=1000 used to crash the process here with STATUS_STACK_OVERFLOW:
// unlike machine::run (trampolined, stack-safe at any depth), the parser
// and typecheck::elaborate are plain recursive descent over native Rust
// stack frames. run_source (lib.rs) now runs the whole pipeline on a
// dedicated large-stack thread specifically to survive this -- n=1000 is
// kept here as a direct regression check on that fix, not because it was
// ever close to the real ceiling.
fn let_chain_source(n: usize) -> String {
    let mut src = String::from("let x0 = 0 in ");
    for i in 1..n {
        src.push_str(&format!("let x{i} = x{} + 1 in ", i - 1));
    }
    src.push_str(&format!("x{}", n - 1));
    src
}

// A single `perform`, resumed n times from one handler clause -- multi-
// shot resume repeated n times from the SAME captured continuation.
// Exercises Cont::append's splice-and-replay path.
fn multishot_source(n: usize) -> String {
    let resumes: Vec<String> = (1..=n).map(|i| format!("resume({i})")).collect();
    format!(
        "handle perform choose(0) with handler choose(p, resume) -> {}",
        resumes.join(" + ")
    )
}

// n sequential (not multi-shot) occurrences of the same effect under a
// deep handler -- each one gets its own freshly-reinstalled handler
// instance. Exercises Frame::HandlerMark's reinstall (now an Rc clone,
// not six field clones) repeated n times.
fn deep_chain_source(n: usize) -> String {
    let mut src = String::new();
    for i in 0..n {
        src.push_str(&format!("let x{i} = perform choose(0) in "));
    }
    let sum: Vec<String> = (0..n).map(|i| format!("x{i}")).collect();
    format!("handle {src}{} with deep(handler choose(p, resume) -> resume(1))", sum.join(" + "))
}

// `let rec sum = fun n -> match n with | 0 -> 0 | _ -> n + sum(n - 1) in
// sum(n)` -- a realistic recursive hot loop, using the two features that
// have replaced hand-rolled recursion as renno's idiomatic style this
// session (let rec, match) but had zero benchmark coverage before now.
// Exercises Value::RecClosure's rebind-on-every-call path and
// Frame::MatchArms's dispatch (match_pattern), n times each.
fn recursive_match_source(n: i64) -> String {
    format!("let rec sum = fun n -> match n with | 0 -> 0 | _ -> n + sum(n - 1) in sum({n})")
}

// Same shape, but each step also constructs a `data` value and reads two
// fields back off it (Point(n)(n) -- a curried constructor call, elaborated
// into nested Lambda/ListLit -- then p.x + p.y, each field access
// desugared by typecheck into its own single-arm Match). Exercises
// construction and FieldAccess's desugared-Match path together, since
// ADTs/named fields also had zero benchmark coverage before now.
fn adt_field_access_source(n: i64) -> String {
    format!(
        "data Point = Point(x: Int, y: Int) in \
         let rec sum = fun n -> match n with | 0 -> 0 | _ -> (let p = Point(n)(n) in p.x + p.y) + sum(n - 1) in \
         sum({n})"
    )
}

fn full_pipeline(c: &mut Criterion) {
    let src = std::fs::read_to_string("examples/multi_shot.rn").expect("run from the repo root");
    c.bench_function("full_pipeline/multi_shot.rn", |b| {
        b.iter(|| run_source(black_box(&src)).unwrap())
    });
}

fn let_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("let_chain");
    for n in [10usize, 100, 1000] {
        let src = let_chain_source(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &src, |b, src| {
            b.iter(|| run_source(black_box(src)).unwrap())
        });
    }
    group.finish();
}

fn multishot_resume(c: &mut Criterion) {
    let mut group = c.benchmark_group("multishot_resume");
    for n in [10usize, 50, 100] {
        let src = multishot_source(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &src, |b, src| {
            b.iter(|| run_source(black_box(src)).unwrap())
        });
    }
    group.finish();
}

fn deep_reinstall(c: &mut Criterion) {
    let mut group = c.benchmark_group("deep_reinstall");
    for n in [10usize, 100, 1000] {
        let src = deep_chain_source(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &src, |b, src| {
            b.iter(|| run_source(black_box(src)).unwrap())
        });
    }
    group.finish();
}

fn recursive_match(c: &mut Criterion) {
    let mut group = c.benchmark_group("recursive_match");
    for n in [10i64, 100, 1000] {
        let src = recursive_match_source(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &src, |b, src| {
            b.iter(|| run_source(black_box(src)).unwrap())
        });
    }
    group.finish();
}

fn adt_field_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("adt_field_access");
    for n in [10i64, 100, 1000] {
        let src = adt_field_access_source(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &src, |b, src| {
            b.iter(|| run_source(black_box(src)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    full_pipeline,
    let_chain,
    multishot_resume,
    deep_reinstall,
    recursive_match,
    adt_field_access
);
criterion_main!(benches);
