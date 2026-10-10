# renno

A small scripting language with algebraic effect handlers and gradual typing, implemented in Rust as a trampolined CEK-style abstract machine.

```
let rec fact = fun n -> if n == 0 then 1 else n * fact(n - 1) in fact(10)
```

## Scope

renno is a language-design and implementation study, not a production language. It is an interpreter only (no compiler or bytecode format), runs single-file programs or a REPL, and has no module system, package manager, FFI, or standard library beyond a small built-in prelude. The language and its error messages may change without notice.

## Highlights

- **Algebraic effects**: `perform`/`handle`, first-class handler values, deep and shallow semantics, and genuine multi-shot resumption (a captured continuation can be resumed zero, one, or many times).
- **Gradual typing**: every position is `Dyn` unless annotated. Annotated code is checked statically and pays no runtime cost; annotated boundaries crossed by an unannotated (`Dyn`) value get a runtime check inserted automatically.
- **Gradual verification**: `let n: Int where 0 < n = 5 in ...` — a refinement predicate proven outright when it cheaply can be (a literal value, zero runtime cost), and backed by a real runtime check otherwise. Same three-way shape as gradual typing's own boundary checks, generalized from type tags to arbitrary decidable predicates.
- **Closed and row-polymorphic effect typing**: `check()` statically rejects a program if it can prove an effect is never handled. Row-polymorphic function types (`(Dyn ->{e} Dyn)`) let effect-safety survive through higher-order calls.
- **`let rec` and mutual recursion**: `let rec f = ... and g = ... in ...` — any function in the group can call any sibling (including itself) by name.
- **Pattern matching**: literals, lists (`[]`, `[a, b]`, `h :: t`), tuples, and records, with static exhaustiveness and reachability checking wherever those are cheaply provable.
- **Tuples**: `(a, b, c)` — a fixed-arity, per-position-typed product, distinct from a variable-length `List`. Desugars to a plain list at runtime (no new value representation), typed `Type::Tuple` for real positional structural comparison. `(p, q)` also works as a pattern, sugar for `[p, q]`.
- **Records**: `{x: 1, y: 2}` — a named counterpart to Tuple, a real separate runtime kind (name-keyed, not Tuple's positional one — needed for width subtyping below to work with no per-boundary value transformation). Read back by `.field` access or by destructuring (`match p | {x, y} -> ...`, `{x, y}` punning for `{x: x, y: y}`). **Width subtyping**: a `{x: Int}`-annotated parameter accepts any record with *at least* field `x` — extras just ride along, unobserved by anything that didn't ask for them, and a `Dyn`-to-`Record` boundary check is genuinely name-aware (not just arity) and every field's value is checked against its field type; `.field` access is backed by the prelude builtin `get_field`.
- **Hand-rolled sum types**: no dedicated ADT syntax — a same-arity tagged sum is an ordinary tuple whose first element is a literal tag (`let None = ("None",) in let Some = fun x -> ("Some", x) in ...`), matched by ordinary literal comparison in pattern position. No new runtime representation, no constructor-pattern sugar to special-case.
- **First-class `opaque`**: `opaque` is an expression — writing it anywhere yields a token unique to that exact source position (the same token every time that code runs, since it's a literal, not a generator), bindable, passable, and comparable with `==`. Combined with tuples, this builds a hand-rolled nominal type: `let Meters = fun n -> (n, opaque) in ...` — two `Meters(_)` values always carry equal tokens, and a same-shaped tuple from anywhere else never does.
- **Union types and type aliases**: `type Name = TypeExpr in ...` names any type expression, and `A | B` unions two or more into one — a self-contained "one of these" with no registry lookup, usable in any annotation (`fun x: Int | Str -> ...`). A `Dyn` value crossing a Union-typed boundary is accepted if it matches *any* alternative, element by element (a union of containers checks inside them); `match` exhaustiveness over a Union-typed scrutinee is proven when every alternative (not necessarily by the same arm) is covered by some pattern — `type Pair = (Int,) | (Int, Int) in ... | (n,) -> n | (n, m) -> n + m` is exhaustive this way, even though neither arm alone covers the whole union.
- **Diagnostics**: every parse error, type error, and runtime panic reports a `line, column` location with a source snippet and a caret, not just a bare message.
- **Multi-line REPL**: `let`/`match` blocks spanning multiple lines can be typed directly at the prompt.

## Requirements

Rust 1.95 or newer (the highest minimum version among the dependencies, via `cranelift-entity`), edition 2024. Tested on Windows with stable and nightly toolchains.

## Quick start

Run a file:

```bash
cargo run --release -- examples/mutual_recursion.rn
```

Start the REPL:

```bash
cargo run --release
```

Run the test suite:

```bash
cargo test
```

Run the benchmarks:

```bash
cargo bench
```

Run the differential check (compares a baseline binary against the current release build over every `examples/` and `corpus/` program). It needs a baseline `renno.exe` that isn't in the repo — build one from an earlier commit and pass its path, or it defaults to `target/baseline/renno-4a58bb3.exe`:

```bash
cargo build --release
bash scripts/differential.sh path/to/baseline/renno.exe
```

## Language tour

### Values and arithmetic

```
1 + 2 * 3        -- 7, usual precedence
7 % 3            -- 1, same precedence as * and /
"a" ++ "b"       -- "ab"
[1, 2] ++ [3]    -- [1, 2, 3]
true == false    -- false
5 > 3   5 <= 5   5 >= 6   5 != 6   -- true, true, false, true
true && false || true    -- true (&& binds tighter than ||), both short-circuiting
!true                     -- false
1 :: 2 :: [3]             -- [1, 2, 3]
```

### Functions, currying, `let`, `let rec`

Functions are curried closures; `fun a -> fun b -> ...` and calling `f(a)(b)` are the normal shape. Application also works by juxtaposition, `f a b` — the two mix freely, and negating a juxtaposed argument needs its own parens (`f (-1)`, not `f -1`, the same resolution Haskell/OCaml use: unary `-` binds looser than application, so `f -1` parses as `f - 1`):

```
let add = fun a -> fun b -> a + b in add(1)(2)   -- 3
let add = fun a -> fun b -> a + b in add 1 2     -- 3, identical
```

`let rec` makes a binding visible inside its own value, for recursion that an ordinary `let` can't express — but only when every value in the group is written as a direct `fun ...` (a syntactic test: `if cond then fun ... else fun ...` does not qualify, even though both branches are functions). When that holds, every name is visible to every value (including its own), simultaneously, not just sequentially in scope like `let`:

```
let rec fact = fun n -> if n == 0 then 1 else n * fact(n - 1) in fact(5)   -- 120
```

Otherwise (any value not written directly as `fun ...`) the group is not recursive: the names are bound only for the body, and each value is evaluated in the outer scope, left to right, the same as a sequence of plain `let`s. A value that references one of the group's own names in that case is not in scope yet and fails at runtime with `unbound variable`, not at typecheck time (typecheck accepts every group member referencing every other, including itself, regardless of whether the group ends up recursive at runtime — see Known limitations).

`and` extends this to a group of mutually recursive functions:

```
let rec is_even = fun n -> if n == 0 then true else is_odd(n - 1)
and is_odd = fun n -> if n == 0 then false else is_even(n - 1)
in is_even(10)
```

### Pattern matching

```
match xs
| [] -> 0
| h :: t -> h + sum(t)
```

A match is rejected statically if it's missing an obviously necessary case (`[]`/`h :: t` both present, `true`/`false` both present, or every alternative of a `Tuple`/`Record`/`Union`-typed scrutinee covered by some arm), and if an earlier arm already covers everything a later one would ever match.

`::` also works as an expression (`h :: t` prepends `h` onto list `t`), not just a pattern, so list-building functions can be written by hand:

```
let rec map = fun f -> fun xs ->
  match xs
  | [] -> []
  | h :: t -> f(h) :: map(f)(t)
in map(fun x -> x * 2)([1, 2, 3])   -- [2, 4, 6]
```

### Tuples and hand-rolled sum types

```
let p = (1, "a", true) in
match p
| (a, b, c) -> a   -- 1
```

`(a, b, ...)` is a fixed-arity product — typed positionally (`Type::Tuple`), not widened to a common element type the way `List` elements are. `(a)` with no comma stays ordinary parenthesized grouping; `(a,)` with a trailing comma is the one-element tuple; a tuple pattern `(p, q)` is sugar over the list pattern `[p, q]`, since a tuple is a plain list at runtime.

There's no dedicated ADT syntax — a same-arity tagged sum is just a tuple whose first element is a literal tag, matched by ordinary value comparison in pattern position:

```
let None = ("None",) in
let Some = fun x -> ("Some", x) in
match Some(5)
| ("None",) -> 0
| ("Some", x) -> x   -- 5
```

Constructors here are ordinary functions, curried like any other multi-argument callable (`Cons(1)(rest)`, not `Cons(1, rest)`).

Tuple types are structural by default: two differently-named `type` aliases for the same shape are interchangeable, same as List or Fun.

```
type Meters = (Int,) in
type Seconds = (Int,) in
let f = fun x: Meters -> x in f((5,))   -- accepted: any (Int,) tuple satisfies either alias
```

### Records

```
let p = {x: 3, y: 4} in
match p
| {x, y} -> x * x + y * y   -- 25
```

`{name: expr, ...}` — like Tuple, but each position also carries a field NAME, and it's a real separate runtime kind (name-keyed `Value::Record`, not a disguised list). Field order is never significant — `{y: 4, x: 3}` and `{x: 3, y: 4}` construct the identical value — though the parser does sort fields by name for its own bookkeeping, which is why field VALUE expressions evaluate in sorted-by-name order rather than as written; only observable if a field expression has a side effect (`perform`).

Read back either way: `.field` access, or destructuring (the same mechanism tuples already use, where a bare field name puns to itself in both construction and pattern position):

```
let x = 3 in let y = 4 in
{x, y}.x + match {x, y} | {x, y} -> y   -- 7, `.x` reads directly, `{x, y}` means `{x: x, y: y}` on both sides
```

`.field` desugars entirely into an ordinary call (no new runtime opcode) and is a peer of application in the same postfix chain: `f(a).x` means `(f(a)).x`, not `f(a.x)`.

**Width subtyping**: a `{x: Int}`-typed parameter accepts any record that has *at least* field `x` — extra fields just ride along, since matching is by name and a field a pattern doesn't name is simply never looked at:

```
let f = fun p: {x: Int} -> p in
match f({x: 1, y: 2}) | {x} -> x   -- 1, `y` is still there on the value, just never asked for
```

That's also why a single arm can be exhaustive across record shapes that would need one arm each as a Tuple: `{x: a}` alone covers every alternative of `type R = {x: Int} | {x: Int, y: Int} in ...`, since every alternative has field `x`. A `Dyn`-sourced record crossing a record-typed boundary gets a real, name-aware check too — not just an arity check — so a same-arity value with the wrong field names is correctly rejected, while a wider one is correctly accepted. Each field's value is checked too, and a failure names the field.

### First-class `opaque`

`opaque` is an expression in its own right, evaluating to a token unique to wherever it's written:

```
let brand = opaque in
brand == opaque   -- false: a DIFFERENT `opaque`, a different token
brand == brand     -- true: same token, same binding
```

Combined with tuples, this builds a nominal type — `opaque`, written once inside a function body, is a literal: the same token every call, since it's the same source position evaluated fresh each time, not a fresh token per call:

```
let Meters = fun n -> (n, opaque) in
let a = Meters(5) in
let b = Meters(10) in
match a
| (an, at) -> match b
  | (bn, bt) -> at == bt   -- true: same function, same embedded `opaque`
```

A same-shaped tuple built by a *different* function never compares equal — its `opaque` is a different token, from a different source position — and typecheck can already tell the two tuples' inferred types are never consistent, so comparing them directly (rather than through something that first erases both sides to `Dyn`) is a *static* type error, not a runtime `false`.

### Effects

```
handle
  let x = perform choose(0) in
  x + 100
with handler choose(p, resume) -> resume(1) + resume(2)   -- 203, multi-shot
```

`deep(handler ...)` reinstalls the handler around a resumed continuation, so it also catches an effect performed again during the resume; `shallow(handler ...)` (the default) does not.

### Gradual typing

```
let f = fun x: Int -> x + 1 in f(41)     -- checked statically, no runtime check inserted
```

```
handle
  let y = perform choose(0) in           -- y: Dyn, unknown until runtime
  let f = fun x: Int -> x + 1 in
  f(y)                                    -- a runtime check is inserted here
with handler choose(p, resume) -> resume(41)
```

### Gradual verification

```
let n: Int where 0 < n = 5 in n + 1     -- proven at parse time; compiles to exactly `let n = 5 in n + 1`
```

```
let f = fun n: Int where 0 < n -> n * 2 in f(-3)   -- a parameter's value is never known this early, so this is a real runtime check -- and it fails
```

Compound predicates (`&&`/`||`/`!`) are just as provable — `let n: Int where 0 < n && n < 100 = 200 in n` is a parse-time error, not a runtime one.

More complete examples for every feature above live in [`examples/`](examples/).

## Architecture

- `lexer.rs` — logos-based tokenizer.
- `parser.rs` — recursive-descent parser into an arena-allocated AST (`expr.rs`); iteratively flattens long `let`/`fun` chains to keep native stack usage O(1) regardless of chain length.
- `typecheck.rs` — bidirectional-lite gradual type checker: infers types and effect rows in one pass, turns a `Dyn`-to-concrete boundary into a native `Expr::Check` node (a real per-call contract for a `Fun` target; only a `Vec(n)` length with a non-literal index and a few unions are still desugared into `if`/`fail` code), and checks match exhaustiveness/reachability.
- `resolve.rs` — static resolver: walks the (already typechecked) AST once and records, for every variable reference, exactly where it lives at runtime (`VarRef::Local { hops, slot }`, `VarRef::Prelude(index)`, or `VarRef::Unbound`) — no name comparison happens during evaluation.
- `machine.rs` — a trampolined CEK-style step loop (`cont.rs` holds the defunctionalized continuation frames), reading `resolve.rs`'s `VarRef`s to look variables up. The machine and the deep container check are iterative (explicit continuation frames, an explicit work list), so evaluation and checking use O(1) native stack in program depth and in value depth, and resuming captured continuations cannot overflow it. Still native-recursive over a very deeply nested *value* (pre-existing, parked): `Display` of a result (runs on the 1 MiB main thread; aborts at roughly 4,000 nested levels), `Outcome::from` (roughly 700,000-1,000,000 levels) and `Drop` of a nested `Value::List`/`Record` (roughly 2-3 million in release); a stack overflow aborts the process. Measurements: `docs/superpowers/specs/2026-10-08-deep-check-stack-safety.md`.
- `frame.rs` — the runtime `Env`: a persistent chain of immutable per-scope frames (`Env::get(hops, slot)`), indexed by resolve.rs's static `VarRef::Local`, not by name.
- `env.rs` — `env::PRELUDE`, the single builtin table both `resolve.rs` (assigning `VarRef::Prelude` indices) and `machine.rs` (looking a builtin up by that index) read.
- `value.rs` — runtime value representation.
- `plist.rs` — the persistent linked list typecheck's context (`Ctx`) is built from; no longer used by the runtime `Env` (see `frame.rs`).
- `span.rs` — source locations and the snippet-with-caret error rendering shared by every error path.

`run_source` (`lib.rs`) runs the whole parse → typecheck → run pipeline on a dedicated large-stack worker thread, since the parser and type checker are ordinary native recursive descent (unlike the machine, which is trampolined).

## Known limitations

- A handler clause's own effects join the `handle` row when the handler is written in place (`handle ... with handler op(p, resume) -> ...`, optionally under `deep`/`shallow`): `resume` counts as a call that performs the body's remaining effects, so a clause performing an unhandled effect is a static error. A handler reached through a variable or parameter gets row `Dyn` (anything may escape; checked at run time). A shallow handler whose continuation performs the handled effect a second time is still only a run-time error, because a row is a set and cannot count occurrences.
- `let rec`'s recursion is a purely syntactic, group-wide test (every value directly `fun ...`), not a semantic one: typecheck itself is more permissive — it binds every name in the group (with a fresh type variable, or a generalized one if annotated) while elaborating every value, so a self/sibling reference type-checks in EITHER case. When the group doesn't qualify as recursive at runtime, that same reference is unbound at the point each value actually runs, and fails lazily with `unbound variable: <name>` the first time (if ever) that code path is reached — not at typecheck time, and not necessarily on every call (e.g. inside a branch that's never taken).
- A branded value's hidden `opaque` tag is not hidden from `Value`'s own `Display`/`Outcome` conversion — a branded tuple printed or returned at the top level shows an extra trailing `<brand>` element, since neither knows a value's static type well enough to leave it out (the id itself is never printed: the brand tag is its own `Value::Token`/`Outcome::Token` kind, not a plain `Int`, so it can't be mistaken for one either).
- Match exhaustiveness and reachability are checked only where cheaply provable (see the doc comments on `missing_case`/`first_unreachable` in `typecheck.rs`); anything past that silently falls back to a runtime panic.
- A runtime panic's reported location generally falls back to the last expression *evaluated*, not necessarily the exact sub-expression at fault a few steps later. Two common cases are threaded through precisely instead of relying on that fallback: `apply_binop`'s operand-type/div-by-zero/etc panics blame whichever operand is actually at fault (or their combined span, when neither alone explains it), and "attempt to call a non-function value" blames the callee, not an unrelated argument evaluated afterward. Everywhere else (builtin argument-type panics, match failure, ...) still uses the last-evaluated fallback.
- `where` refinement predicates are limited to what `<`/`<=`/`>`/`>=`/`==`/`!=`, `&&`/`||`/`!`, and arithmetic can express. Proving is attempted only when the bound value reduces to a closed Int constant at parse time (`try_eval_closed_int`); a Lambda parameter's refinement is never proven statically, since its actual value is unknown until a caller supplies one.
- A `Dyn`-to-container boundary is checked all the way down, decided by the static type alone: `[Int]` means every element is Int, `(Int, Str)` checks both positions, `{a: Int}` checks field `a`, through unions and (recursive) type aliases, and a failure names the element (`type error: expected [Int], found Bool at element 1`). The same check applies to a value whose static type is only partly `Dyn` (`[Dyn]` into `[Int]`, `(Dyn, Int)` into `(Int, Int)`, a union into one of its members), except an empty list literal into a list target (an empty literal into an alias or union target that has a tuple or list alternative is checked at run time; into a record target, or a union without one, it is still a static error). `[Float]` rejects Int elements. Not yet checked inside a container: the per-call contract of a function (only that it is callable) and the length of a `Vec(n)` whose `n` is a variable (only that it is a list). A literal length is checked there, through aliases too: `[Vec(2)]` rejects `[[1, 2], [3]]` at element 1. Parked. Failure messages elide the middle of a path longer than 20 segments. The check walks the value with an explicit work list, so a deeply nested value costs heap, not native stack; a literal tuple annotated with a recursive alias is checked statically for about 21 levels (`let l: L = (1, (2, 3))` is rejected); a literal nested deeper is still caught, but as a run-time error. Match exhaustiveness for nested tuple patterns only recognizes a single, fully-nested pattern in ONE `match` (`((p, q), x)`) — a scrutinee first bound by an outer match/lambda pattern loses its precise Tuple type (renno has no pattern-driven type refinement), so chaining separate `match`es over the same value falls back to "possibly non-exhaustive."
- Width subtyping is invisible only to pattern matching, not to everything — a wider record accepted where a narrower type is annotated keeps every field it always had (no value is ever narrowed at a boundary), so those "extra" fields still show up if the value is printed or crosses to `Outcome` (same kind of leak the `opaque` brand tag above already has, just for a Record's own extra fields instead of a Token's hidden id), and `==` on two records still compares everything either side actually carries, not just what some annotation happened to name — a value satisfying a type is a different question from two values being equal.
- `Type::Token` (`opaque`'s own static type) has no surface spelling, and deliberately won't get one: the only way to write it would be to recognize one specific syntactic pattern (e.g. `let NAME = opaque in ...`) in a new parser-side alias table — exactly the kind of bespoke, narrowly-special-cased machinery this codebase has consistently avoided in favor of general mechanisms (tuples, unions, type aliases) that don't need to know about `opaque` at all. So a hand-rolled `opaque`-tagged sum (`let None = (opaque,) in let Some = fun x -> (opaque, x) in ...`) fully works — construction, matching via ordinary tuple patterns, `==` on the tokens, `Dyn`-boundary shape checks — but never gets a `type` alias naming it, and so never gets static exhaustiveness checking. Same "prove what's cheap, stay silent otherwise" treatment an Int-literal match with no wildcard already gets, not a special exception.
- A `Dyn`-to-`Union` boundary check picks the matching alternative via a native shape test (build_shape_predicate), then runs THAT alternative's own full `build_boundary_check` — so a `Fun`-typed alternative gets the same real per-call contract (`wrap_fun_contract`) a bare (non-union) `Fun` annotation would, not just an "is this callable" test. A statically Union-typed value still can't be CALLED directly (`fun h: (Int -> Int) | Str -> h(5)` is rejected — `App`'s typecheck only knows how to call a `Fun`- or `Dyn`-typed callee); only reachable once the value has been passed through something that erases it to `Dyn` first (an unannotated parameter, `map`'s own native `apply`, ...).
- `cannot tell statically or at run time whether A fits B` is a static error when a union-typed source, at any depth (inside a tuple, list or record component, or under a function parameter or result), has an alternative differing from the target only in a function signature, a nominal type or a length, because neither the type checker nor a run-time value test can decide that alternative (typical case: a union containing a typed function, passed where a different union or a function type needing a wrapper is expected). Annotate the source as `Dyn` to defer the check to run time. A target union is rejected as a whole when any alternative that could apply is undecidable, even if another alternative could have been decided by a value test.

## AI-assisted development

Much of renno was written with AI assistance (Claude Code). Every commit is authored and committed under the repository owner's name; commits made with Claude's help carry a `Co-Authored-By: Claude` trailer. At the time of writing that is 194 of the 216 commits (about 90%). Treat the git history as the record.

## License

MIT; see [LICENSE](LICENSE).
