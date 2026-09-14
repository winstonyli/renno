use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use renno::run_source;

// `let x0 = 0 in let x1 = x0 + 1 in ... in x{n-1}`. Exercises Env/Ctx
// extend+lookup scaling: both are the shared PList (src/plist.rs), an
// Rc-based persistent list with O(1) extend -- this benchmark is the
// direct proof that a chain of n bindings costs O(n), not the O(n^2) the
// old Vec-cloning Ctx used to cost before that fix.
//
// Keep n well under ~1000: unlike machine::run (trampolined, stack-safe
// at any depth), the parser and typecheck::elaborate are plain recursive
// descent over native Rust stack frames -- a long enough chain overflows
// it (confirmed: n=1000 crashes with STATUS_STACK_OVERFLOW). That's a
// real scalability limit of the front end, not something to paper over
// by silently raising the thread stack size here.
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

fn full_pipeline(c: &mut Criterion) {
    let src = std::fs::read_to_string("examples/multi_shot.rn").expect("run from the repo root");
    c.bench_function("full_pipeline/multi_shot.rn", |b| {
        b.iter(|| run_source(black_box(&src)).unwrap())
    });
}

fn let_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("let_chain");
    for n in [10usize, 100, 300] {
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
    for n in [10usize, 100, 300] {
        let src = deep_chain_source(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &src, |b, src| {
            b.iter(|| run_source(black_box(src)).unwrap())
        });
    }
    group.finish();
}

criterion_group!(benches, full_pipeline, let_chain, multishot_resume, deep_reinstall);
criterion_main!(benches);
