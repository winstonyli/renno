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
- **Row-polymorphic effect typing**: `check()` statically rejects a program if it can prove an effect is never handled. Row variables are inferred and unified; unannotated higher-order functions are effect-polymorphic; a lambda's row is open; a signature's `->{e}` is rigid.
- **`let rec` and mutual recursion**: `let rec f = ... and g = ... in ...` — any function in the group can call any sibling (including itself) by name.
- **Pattern matching**: literals, lists (`[]`, `[a, b]`, `h :: t`), tuples, and records, with static exhaustiveness and reachability checking wherever those are cheaply provable.
- **Tuples**: `(a, b, c)` — a fixed-arity, per-position-typed product, distinct from a variable-length `List`. Desugars to a plain list at runtime (no new value representation), typed `Type::Tuple` for real positional structural comparison. `(p, q)` also works as a pattern, sugar for `[p, q]`.
- **Records**: `{x: 1, y: 2}` — a named counterpart to Tuple, a real separate runtime kind (name-keyed, not Tuple's positional one — needed for width subtyping below to work with no per-boundary value transformation). Read back by `.field` access or by destructuring (`match p | {x, y} -> ...`, `{x, y}` punning for `{x: x, y: y}`). **Width subtyping**: a `{x: Int}`-annotated parameter accepts any record with *at least* field `x` — extras just ride along, unobserved by anything that didn't ask for them, and a `Dyn`-to-`Record` boundary check is genuinely name-aware (not just arity) and every field's value is checked against its field type; `.field` access is backed by the prelude builtin `get_field`.
- **Hand-rolled sum types**: no dedicated ADT syntax — a same-arity tagged sum is an ordinary tuple whose first element is a literal tag (`let None = ("None",) in let Some = fun x -> ("Some", x) in ...`), matched by ordinary literal comparison in pattern position. No new runtime representation, no constructor-pattern sugar to special-case.
- **First-class `opaque`**: `opaque` is an expression — writing it anywhere yields a token unique to that exact source position (the same token every time that code runs, since it's a literal, not a generator), bindable, passable, and comparable with `==`. Combined with tuples, this builds a hand-rolled nominal type: `let Meters = fun n -> (n, opaque) in ...` — two `Meters(_)` values always carry equal tokens, and a same-shaped tuple from anywhere else never does.
- **Union types**: a `let` names any type expression (`let IntOrStr = Union(Int, Str) in ...`), and `A | B` unions two or more into one — a self-contained "one of these" with no registry lookup, usable in any annotation (`fun x: Int | Str -> ...`). A `Dyn` value crossing a Union-typed boundary is accepted if it matches *any* alternative, element by element (a union of containers checks inside them); `match` exhaustiveness over a Union-typed scrutinee is proven when every alternative (not necessarily by the same arm) is covered by some pattern — `let Pair = Union(Tuple((Int,)), Tuple(Int, Int)) in ... | (n,) -> n | (n, m) -> n + m` is exhaustive this way, even though neither arm alone covers the whole union.
- **Types are values**: the type names are prelude values (`Int`, `Float`, `Bool`, `Str`, `Dyn`) and constructors (`List`, `Tuple`, `Record`, `Union`, `Fun`, `Vec`, `Indexed`, `TypeOf`), so `to_str(List(Int))` is `[Int]`, and every annotation is sugar for calls to them: `[Int]` is `List(Int)`, `(Int, Str)` is `Tuple((Int, Str))`, `{x: Int}` is `Record({x: Int})`, `A | B` is `Union((A, B))`, `A -> B` is `Fun((A, B))` and `A ->{e} B` is `Fun((A, "e", B))`. The sugar names the prelude constructors directly, so a local `let List = 2` does not change what `[Int]` means; a name written in an annotation is an ordinary variable, so after `let Int = 5` the annotation `Int` is an error (`annotation did not evaluate to a type`), and `Vec(n)` and `Indexed(T, n)` call whatever `Vec` and `Indexed` are in scope. Their index is still built in: a literal one is passed as an Int (`Vec(2)` passes `2`), and one that names an index variable, which has no value, as the type `Vec(n)` (`Indexed(T, n)` passes `(T, Vec(n))`); the prelude's `Vec` and `Indexed` take the index from either. A `let` names a type (`let T = List(Int) in let x: T = [1] in ...`), a function from types to types is a parameterized alias (`let Pair = fun a -> Tuple((a, a)) in let p: Pair(Int) = (1, 2) in ...`), and a `let rec` whose values are all unannotated calls builds a recursive type wherever it runs, at check time or run time, so recursive types work inside type functions (`let rec L = Union((Tuple((Int, L)), Bool)) in ...`, `let A = fun v -> (let rec R = Union(Tuple(v, R), Bool) in R)`). Inside an annotation a call's arguments are parsed as types, so `Pair([Int])` works and `(Int, Str)` there is the tuple type: write `Tuple(Int, Str)`, not `Tuple((Int, Str))`. Annotations are evaluated at check time, so the definitions they name must be pure (a `perform` or a non-type builtin is a static error, and a record field read is one at check time; through a run-time value it is a run-time error); one that does not finish within the step budget is an error where the run time reaches it (`annotation did not terminate within 1000000 steps`; each type `let` is evaluated at most once at check time (a group of direct `fun`s at most twice: as its own values see it and as the code after it does; an enclosing definition reuses the value of a nested `let` already evaluated instead of re-running it (one not yet evaluated, or one that failed, runs inline)), with a budget of its own, and every annotation reads the values of the `let`s it names), and a type `let` only annotations read is not run when running it cannot fail. An annotation that names a function's parameter, a pattern variable or a handler's payload is evaluated at run time instead, once per closure created for a lambda parameter and on each call for a parameter of a direct `let rec`; the checker treats the binder as `Dyn`, and the run time checks it against the result with the message a static check gives (`let f = fun T -> fun x: T -> x in f(Int)(true)` fails with `type error: expected Int, found Bool`). A value annotated with a function type, statically or with a run-time type, is checked for callability at once. It is then wrapped in a contract that checks each call's result, and also each argument when the binder is untyped (`let f = fun T -> fun g: T -> g in f(Fun((Int, Int)))(fun y -> true)(3)` fails with `type error: expected Int, found Bool`). One whose value, or a definition the same annotation uses, holds a run-time check not known to pass (a cast from `Dyn`, say) stays and runs like any `let`, so that check still fails at run time. `Indexed(T, n)` is `T` at index `n` (written `T(n)` before), `TypeOf(B)` is the type of the `opaque` token `B`, and `==` compares types structurally, a recursive type up to the names of its members (union order matters). Types are hash-consed, so a type built as a DAG (`let T1 = Tuple((T0, T0)) in let T2 = Tuple((T1, T1)) in ...`, whose tree expansion doubles at each step) is checked, compared, generalized and run-time-checked in time proportional to its distinct nodes, not its expansion; a value shared the same way is checked once per shared node. `is(T)(v)` is true when the value `v` satisfies the type value `T`, by the same native test a `Dyn` crossing runs: `is(Union(Int, Str))(x)`, `is(Float)(1)` is false (no coercion), `is(List(Int))(xs)` checks every element not already known to pass this test (a passed list-element test is remembered on the list's blocks, so a list function reached through a run-time-typed parameter, a `Dyn` crossing or an untrusted contract re-checks only what changed, not the whole list at each call), `is(Dyn)(v)` always holds, a record type tolerates extra fields, and a `Fun` type tests callability only (its parameter and result types are ignored). `is` takes a type, not an annotation, so a `Vec` whose length is a variable is a run-time error (`Vec(3)` works). Two limits remain. `==` on two distinct recursive types compares their member names structurally, which walks the expansion, so for some shapes of recursive-type DAG it can still take exponential time (accepted; the common cases are memoized). And an error message names at most 10,000 type nodes, then `...`; a shared pair `relate` decided near the root also counts as decided at a shared position more than 64 levels deep, where it used to become a run-time test. Likewise a pair of huge shared types the old checker gave up on (its work budget ran out, so it left a run-time test) is now decided in full, so a program passing `(Dyn.., Bool)` where `(.., Int) | Str` is expected is rejected at check time instead of failing, if at all, when run.
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

Otherwise (any value not written directly as `fun ...`) the group is not recursive, with one exception: if every value is a call with no annotation (decided from the syntax alone), the group is a recursive type (see **Types are values**), built when it runs, and inside its values each name refers to the group's own unfinished type, not to an outer binding of the same name: an annotation naming one is an error (`recursive type L is used before it is built`), and a value that is a bare name of the group is `recursive type T is not built by a type constructor`. A group that is not recursive binds the names only for the body, and each value is evaluated in the outer scope, left to right, the same as a sequence of plain `let`s. A value that references one of the group's own names in that case is not in scope yet and fails at runtime with `unbound variable`, not at typecheck time (typecheck checks such a group's values in the outer scope too, where that name is unknown and typed `Dyn`; an annotation inside a value reads names the same way, so there a group member's name is `unbound variable` at check time, or means its outer binding — see Known limitations).

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

List cost: `x :: xs` is amortized O(1) and `h :: t` and the head and tail of a pattern are O(1), so a recursion over a list is linear; `a ++ b` writes the shorter side: O(|b|) when `b` is shorter and a claim succeeds, amortized O(1) for `acc ++ [x]` in a loop, and O(|a|) otherwise (O(1) when either side is empty; a snoc onto a cons-built or branched list copies once into a block with room); `get(xs)(i)` is O(1) on a list built in bulk (a literal, `range`, a builtin's result) and O(log n) on one built by consing (O(n) in the worst case, a list built by repeated branching, as for a cons list). The representation is a claim-list (`src/list.rs`). Memory is not always bounded by the old copy-per-tail behaviour: a list keeps alive every value consed (or snocced, at the block's low end) into a free slot of its block, even by a branch that was then discarded (a 10-million-element list consed onto a shared list and dropped stays reachable while any view of that block lives).

String cost: `s ++ t` writes the shorter side into a growable byte buffer when the longer operand's edge is the buffer's frontier, so a builder loop (`s ++ "a"`, `"a" ++ s`, `acc ++ to_str(x)`, `h ++ sep ++ rest`) is amortized O(1) per step at either end; any other `++` copies once (O(|s| + |t|), with as much room again when it was at the frontier). Literals, `to_str` and `join` results are flat single allocations. A grown string keeps up to 2x of its buffer alive, and a view keeps the whole buffer alive. Strict alternation of both ends (`"[" ++ s ++ "]"` in a loop) copies each step and stays quadratic (parked, as for lists). `len` is still O(n) (it counts chars). The representation is `src/text.rs`.

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

Tuple types are structural by default: two differently-named type `let`s for the same shape are interchangeable, same as List or Fun.

```
let Meters = Tuple((Int,)) in
let Seconds = Tuple((Int,)) in
let f = fun x: Meters -> x in f((5,))   -- accepted: any (Int,) tuple satisfies either name
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

That's also why a single arm can be exhaustive across record shapes that would need one arm each as a Tuple: `{x: a}` alone covers every alternative of `let R = Union(Record({x: Int}), Record({x: Int, y: Int})) in ...`, since every alternative has field `x`. A `Dyn`-sourced record crossing a record-typed boundary gets a real, name-aware check too — not just an arity check — so a same-arity value with the wrong field names is correctly rejected, while a wider one is correctly accepted. Each field's value is checked too, and a failure names the field.

**Open records**: a record type can end in a row variable, `{x: Int, ..r}` ("at least `x`, and whatever else `r` stands for"), so a function can read some fields and hand the rest back unchanged:

```
let g: ({x: Int, ..r} -> {x: Int, ..r}) = fun p -> p in
g({x: 1, y: 2}).y   -- 2, the type remembers `y` rides along
```

An unannotated parameter that is read with `.x` or matched with `{x: v}` is inferred the same way, so `let get = fun r -> r.x in get(5)` and `get({y: 1})` are static errors (the parameter's type is `{x: _, ..e}`); a `Dyn` argument still passes and `.x` keeps its run-time check. `..r` is a type-annotation form only (a record literal or pattern cannot carry one); `Record(({x: Int}, "r"))` builds the same open type as a value. A row variable written both as a record tail and as an effect row (`->{r}`) is an annotation error, and an open tail inside a `Union` or a recursive type is a construction error.

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

A `perform` and a `resume` do not copy the stack: the frames between two handlers are one segment, which a capture takes and a resume pushes back by pointer. Their cost is the number of handlers, plus pending shallow resumes (joins), between the `perform` and its handler, not the recursion depth, so a `perform` in non-tail recursion is linear overall (`let rec go = fun i -> if i == 0 then 0 else (perform e(i)) + go(i - 1)` under a deep handler runs 100,000 levels in well under a second). The exception is a shallow handler that re-handles with a non-tail resume: each such resume leaves a join behind, every later `perform` walks them, and the loop is O(n^2) (as before this change), e.g. `let rec run = fun n -> fun th -> handle th() with handler tick(p, r) -> if n == 0 then 0 else run(n - 1)(fun u -> r(0) + 1) in let rec loop = fun i -> let x = perform tick(i) in loop(i + 1) in run(N)(fun u -> loop(0))`.

Effects performed in a `map`/`filter`/`fold`/`sort` callback reach the enclosing handler, multi-shot included (`handle map(fun x -> perform choose(x))([1, 2]) with deep(handler choose(p, resume) -> resume(p + 100))` is `[101, 102]`): these builtins run as a `Frame::Iterate` on the real continuation. A handler that does not resume in tail position (`resume(x) + 1`, or resuming twice) and fires on every element costs O(n^2) in the list length for `map`/`filter` (the partial result is copied per resumption).

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

### Verification by guard

```
let f = fun n: Int -> if 0 < n then n * 2 else fail("n must be positive") in f(-3)   -- an ordinary `if` with `fail`; stops with that message
```

More complete examples for every feature above live in [`examples/`](examples/).

## Architecture

- `lexer.rs` — logos-based tokenizer.
- `parser.rs` — recursive-descent parser into an arena-allocated AST (`expr.rs`); iteratively flattens long `let`/`fun` chains to keep native stack usage O(1) regardless of chain length. A type annotation is parsed as an expression: the sugar becomes calls to the prelude's type constructors (`#List`, `#Fun`, ..., a spelling no user binding can capture), an index that names an index variable (passed to `Vec` as the type `Vec(n)`, and to `Indexed` as `(T, Vec(n))`), is an `Expr::TypeLit`, and any other name, `Vec` and `Indexed` included, is a plain variable that typecheck resolves in the binder's scope (it evaluates each `let` an annotation names once and runs the annotation in a frame of those `let`s' check-time values, and drops a type `let` only annotations read when running it cannot fail).
- `typecheck.rs` — bidirectional-lite gradual type checker: infers types and effect rows in one pass, turns a `Dyn`-to-concrete boundary into one native `Expr::Conform` node, whose type is lowered at check time for a static boundary and when reached for a run-time annotation (every boundary, including function and union-with-function types, is that one node), and checks match exhaustiveness/reachability. A `let` generalizes over the type, row and index variables no enclosing scope holds: each variable a scope holds carries a level, lowered whenever unification links it to another variable, and a `let` quantifies the variables above its own level, so generalization costs the size of the `let`'s type instead of a scan of the context. Each scope also carries the annotation variables (the row and index names a user wrote) it introduced, alongside its level: a written name stands for a fresh variable minted at its binder (`open_annotation`), so a name means the same thing only within the scope that introduced it. An annotation that names a binder is kept for the run time: an `Expr::Annot` evaluated once where its value is known but its checks may fail, or, where it names a run-time value (a parameter, a pattern variable, a handler's payload), evaluated where it is reached, with an `Expr::Conform` that lowers the resulting type to a check, or to a per-call contract for a function type, giving the static messages.
- `resolve.rs` — static resolver: walks the (already typechecked) AST once and records, for every variable reference, exactly where it lives at runtime (`VarRef::Local { hops, slot }`, `VarRef::Prelude(index)`, or `VarRef::Unbound`) — no name comparison happens during evaluation.
- `machine.rs` — a trampolined CEK-style step loop (`cont.rs` holds the defunctionalized continuation frames, as segments between handlers, and the handler marks that separate them), reading `resolve.rs`'s `VarRef`s to look variables up. The machine and the deep container check are iterative (explicit continuation frames, an explicit work list), so evaluation and checking use O(1) native stack in program depth and in value depth, and resuming captured continuations and nesting `map`/`filter`/`fold`/`sort` callbacks cannot overflow it. Still native-recursive over a very deeply nested *value* (pre-existing, parked): `Display` of a result (now on the big-stack thread, which holds about a million levels; it aborted at roughly 4,000 on the 1 MiB main thread), `Outcome::from` (between 500,000 and 1,000,000 levels, measured in release through the binary) and `Drop` of a nested `Value::List`/`Record` (roughly 2-3 million in release, measured before P4b; a nested list now drops through `Rc<Block>`, so the figure is stale and not re-measured); a stack overflow aborts the process. Also parked, same class (the nesting budget bounds source, not what a program builds at run time): a type built to millions of levels at run time (`let rec t = fun n -> if n == 0 then Int else List(t(n - 1))`) still aborts at about 5,000,000 levels in release, and in the unoptimized build casting a 70,000-deep `Dyn` value to a 70,000-deep type overflows; fixing either needs iterative type walkers (or a value/type depth budget at construction), not done. A chain of list blocks (a list built by repeated branching can be a million deep) drops iteratively, not by recursion. Measurements: `docs/superpowers/specs/2026-10-08-deep-check-stack-safety.md`. The same step loop has a check mode, `machine::eval_pure` and its frame-taking form `eval_pure_in` (a step budget, no effects, and no builtin but the type constructors; `eval_pure_in` evaluates its own copy of the expression, in which a nested definition that already has a value is read from the frame instead of copied; it binds the values of the definitions it reads and resolves the copy into a table the checker keeps across evaluations. Nested definitions therefore cost memory linear in their depth, except a recursive `fun` group evaluated from inside its own body before the definitions nested in it exist), which typecheck uses to turn each annotation into its type, and the run time re-enters for an annotation it keeps.
- `frame.rs` — the runtime `Env`: a persistent chain of immutable per-scope frames (`Env::get(hops, slot)`), indexed by resolve.rs's static `VarRef::Local`, not by name.
- `env.rs` — `env::PRELUDE`, the single builtin table both `resolve.rs` (assigning `VarRef::Prelude` indices) and `machine.rs` (looking a builtin up by that index) read.
- `value.rs` — runtime value representation.
- `list.rs` — the list representation: a claim-list (a VList whose growth slots the first cons claims), with O(1) cons, head and tail, and amortized O(1) snoc, on the shared claim engine, and an O(1) bulk `get`.
- `claimed.rs` — the claim engine: write-once storage whose written slots form one interval; a view at the frontier extends in place, any other view copies. Backs lists and strings.
- `text.rs` — strings: `Text`, a flat `Rc<str>` or a view of a growable byte buffer (the second `Slots` storage, `ByteSlots`) that `++` extends through the claim engine; its `Deref` to `str` holds the crate's one `unsafe`, checked under Miri.
- `span.rs` — source locations and the snippet-with-caret error rendering shared by every error path.

`run_source` (`lib.rs`) runs the whole parse → typecheck → run pipeline on a dedicated large-stack (256 MiB) worker thread, since the parser and type checker are ordinary native recursive descent (unlike the machine, which is trampolined), and the binary runs everything (the REPL's parse, the run, printing and dropping the result) on one such thread. Source may nest at most 2,000 levels (`parser::MAX_NESTING`): a program nested deeper (`[[[...]]]`, a long `else if` chain, `1 + 1 + ...` with thousands of terms, a deep pattern or annotation) is the error `expression nested too deeply (limit 2000)` at the first level past the limit, not a stack overflow. A `let`/`let rec`/`fun` chain is not a nesting level (the parser, checker and resolver walk it by iteration) and has no cap: the scopes and types the checker builds along it (a shadowed scheme is reachable only through the next one's scope; a `fun` chain's type is as deep as the chain) drop through `util::bounded_drop`, which bounds the native stack of nested drops, and chains of 1,000,000 `let` (rebinding one name or not), `let rec` or `fun` links run in both the unoptimized and the release build. Trade-off, stated plainly: generated source that is deeper than 2,000 levels is now rejected where much of it used to run (release handled about 66,000 levels) -- long `else if` chains, `+`/`++`/`||`/`::` chains with more than 2,000 terms, parentheses nested more than 2,000 deep. The limit stays at 2,000 because the unoptimized build overflows at about 4,700 levels. Flat shapes are unaffected (a list literal of 1,000,000 elements, a record of 50,000 fields and a `match` of 3,000 arms all run, and so does a `let`/`fun` chain of 1,000,000 links); only depth is limited, not size. Real programs nest under 100 levels.

## Known limitations

- A handler clause's own effects join the `handle` row when the handler is written in place (`handle ... with handler op(p, resume) -> ...`, optionally under `deep`/`shallow`): `resume` counts as a call that performs the body's remaining effects, so a clause performing an unhandled effect is a static error. A handler reached through a variable or parameter gets row `Dyn` (anything may escape; checked at run time). A shallow handler whose continuation performs the handled effect a second time is still only a run-time error, because a row is a set and cannot count occurrences.
- A function that handles an effect of its parameter (`let apply = fun f -> handle f(1) with handler a(p, resume) -> resume(5) in apply(fun x -> perform a(x))`) is rejected `unhandled effect: a`, because handling removes labels from an open row and rows have no difference form; it ran before effect rows were unified.
- Two calls of callback parameters whose effects are unioned in one result share one effect row, so they cannot carry independent effects.
- A union of two distinct row variables that are both fixed by signatures is rejected (`effect row variables e1 and e2 are fixed by the signature and cannot be combined`).
- `let rec`'s recursion is a purely syntactic, group-wide test (every value directly `fun ...`), not a semantic one, and typecheck follows it — it binds every name in the group (with a fresh type variable, or a generalized one if annotated) only while elaborating the values of a recursive group; any other group's values are checked in the outer scope, where they run. There a self/sibling reference with no outer binding is unknown to typecheck (typed `Dyn`), so it is unbound at the point each value actually runs, and fails lazily with `unbound variable: <name>` the first time (if ever) that code path is reached — not at typecheck time, and not necessarily on every call (e.g. inside a branch that's never taken).
- A branded value's hidden `opaque` tag is not hidden from `Value`'s own `Display`/`Outcome` conversion — a branded tuple printed or returned at the top level shows an extra trailing `<brand>` element, since neither knows a value's static type well enough to leave it out (a value never prints its id, which only a message naming a token type shows, as `Token#8`: the brand tag is its own `Value::Token`/`Outcome::Token` kind, not a plain `Int`, so it can't be mistaken for one either).
- Match exhaustiveness and reachability are checked only where cheaply provable (see the doc comments on `missing_case`/`first_unreachable` in `typecheck.rs`); anything past that silently falls back to a runtime panic.
- A runtime panic's reported location generally falls back to the last expression *evaluated*, not necessarily the exact sub-expression at fault a few steps later. Three common cases are threaded through precisely instead of relying on that fallback: `apply_binop`'s operand-type/div-by-zero/etc panics blame whichever operand is actually at fault (or their combined span, when neither alone explains it), and "attempt to call a non-function value" blames the callee, not an unrelated argument evaluated afterward; a builtin's argument-type panic blames the callee (every node typecheck synthesizes inherits the hull of its children's spans). Everywhere else (match failure, ...) still uses the last-evaluated fallback.
- A `Dyn`-to-container boundary is checked all the way down, decided by the static type alone: `[Int]` means every element is Int, `(Int, Str)` checks both positions, `{a: Int}` checks field `a`, through unions and (recursive) types, and a failure names the element (`type error: expected [Int], found Bool at element 1`). The same check applies to a value whose static type is only partly `Dyn` (`[Dyn]` into `[Int]`, `(Dyn, Int)` into `(Int, Int)`, a union into one of its members), except an empty list literal into a list target (an empty literal into an alias or union target that has a tuple or list alternative is checked at run time; into a record target, or a union without one, it is still a static error). `[Float]` rejects Int elements. A function inside a list, tuple or record crossing from `Dyn` (or an untyped site) into a typed container is wrapped like a bare function crossing, so every call checks its argument and result against the annotation, and a failure names where it sits (`type error: expected Int, found Str (function at [0])`); a container with no function in it is not rebuilt, and repeated crossings of one function keep one wrapper. The other direction is covered too: a typed container holding a function that goes into `Dyn` (a `Dyn` binding, a parameter, a `++` or `::` join with a `Dyn` list) wraps that function, so a `Dyn` caller's wrong argument is rejected (`let d: Dyn = (f, 2) in match d | (g, _) -> g("s")` fails with `expected Int, found Str`). A function whose type is a `Union` with a function alternative, or a recursive type, also keeps its parameter check going into `Dyn` (`let u: Union(Int, (Int -> Int)) = ... in let d: Dyn = u in d("s")` fails with `expected Int, found Str`). The same holds for such a function as an element of a container (`let l: [Union(Int, (Int -> Int))] = ... in let d: Dyn = l in match d | g :: _ -> g("s") | _ -> 0` fails with `expected Int, found Str`). A function leaf whose parameter is `Vec(n)`, with `n` bound by an enclosing function, also compares its argument's length with that `n` (container leaf or `Union` alternative); a leaf whose `n` has no enclosing length accepts any length. A `Vec` length is checked there, literal or not, through aliases too: `[Vec(2)]` rejects `[[1, 2], [3]]` at element 1, and inside `fun xs: Vec(n)` a `[Vec(n)]` compares every element with the length of `xs`. An index variable with no run-time value at the crossing behaves as at the top level: a single crossing outside a function body that mentions it once only requires a list; inside a function body, with two crossings that share it, or with several distinct indices in one annotation (`(Vec(n), Vec(n + 1))`) it fails with `index variable n has no runtime value to check a length against`. Failure messages elide the middle of a path longer than 20 segments. The check walks the value with an explicit work list, so a deeply nested value costs heap, not native stack; a literal tuple annotated with a recursive alias is checked statically for about 21 levels (`let l: L = (1, (2, 3))` is rejected); a literal nested deeper is still caught, but as a run-time error. Match exhaustiveness for nested tuple patterns only recognizes a single, fully-nested pattern in ONE `match` (`((p, q), x)`) — a scrutinee first bound by an outer match/lambda pattern loses its precise Tuple type (renno has no pattern-driven type refinement), so chaining separate `match`es over the same value falls back to "possibly non-exhaustive."
- Width subtyping is invisible only to pattern matching, not to everything — a wider record accepted where a narrower type is annotated keeps every field it always had (no value is ever narrowed at a boundary), so those "extra" fields still show up if the value is printed or crosses to `Outcome` (same kind of leak the `opaque` brand tag above already has, just for a Record's own extra fields instead of a Token's hidden id), and `==` on two records still compares everything either side actually carries, not just what some annotation happened to name — a value satisfying a type is a different question from two values being equal.
- Open-record inference pins a parameter only through `.field` reads and through `match` arms that all name the same fields. A pinned payload follows the arms' binders only when every arm binds it to a plain variable whose type is already known and the types agree; otherwise it stays free (tagged dispatch), so a call can still pass a payload of the wrong type there. A closed record joined with an open one narrows the open tail, which can reject a later call that lacked a field the join added.
- `Type::Token` (`opaque`'s own static type) is spelled `TypeOf(B)`, applied to the token value itself, so only code holding the token can name its type: an ordinary constructor rather than a parser-side alias table keyed on `let NAME = opaque`. A hand-rolled `opaque`-tagged sum (`let None = (opaque,) in let Some = fun x -> (opaque, x) in ...`) fully works — construction, matching via ordinary tuple patterns, `==` on the tokens, `Dyn`-boundary shape checks. A message names a token type by its brand, the source offset of its `opaque` (`expected Token#8, found Token#26`); a token value still prints as `<brand>`.
- Annotation variables (`n` in `Vec(n)`, `e` in `->{e}`) are scoped lexically, not program-wide: a name is introduced at the binder that writes it, and the same spelling in a sibling scope is an unrelated variable, so results do not depend on the order of sibling bindings (one caveat: in a `let rec` group a name written by two members is one variable, unless an outer existential of that name is in scope, which a function member shadows with a fresh variable and a non-function member shares, so `let rec f: (Vec(n) -> Int) = fun v -> len(v) and k: Vec(n) = [1, 2] in f([1, 2, 3])` is rejected while the same program behind `let a: Vec(n) = [1, 2] in` returns 3). A variable is universal or existential. A function signature (a `let` or `let rec` annotation that is a function type) and a lambda parameter's annotation write universal variables, which an inner binder of the same name shares. Only a signature's variables are rigid: its body may not pin them (`n` is fixed by the signature). A lambda parameter's variable stays flexible, so a body that pins it is rejected by the ordinary mismatch (`(fun a: Vec(n) -> let w: Vec(n) = [1, 2, 3] in len(a))([1])` fails because the body's `Vec(n)` shares the parameter's variable and fixes n := 3); any other `let` annotation (`let x: Vec(n) = [1, 2]`) writes existentials, which the value fixes and an inner binder of the same name shares. An inner universal binder shadows an in-scope existential. The extents: a signature's variables cover the value only (they are generalized at the `let`; a body annotation reusing the name denotes the same variable); a parameter's cover the lambda body; an existential's cover the value and the `let` body. A type alias is evaluated where it is used, so the names it mentions resolve at its use, not where it was defined. Annotations evaluated at run time (below) introduce no names.
- An annotation evaluated at run time is weaker than a static one in a few places, all lenient (none rejects a correct program): the checker treats its binder as `Dyn`, so `fun T -> fun x: List(T) -> x + 1` is accepted; an index variable that only such an annotation writes matches any length, so repeated occurrences need not agree (`(fun Vec -> fun x: Vec(n) -> fun y: Vec(n) -> y)(Vec)([1])([1, 2])` returns `[1, 2]`), and so does a length a function type introduces only in its result; a run-time function contract does not check effect rows; and a `let rec` member whose annotation is evaluated at run time is checked once after the group is built, so calls inside the group bypass it. A user `Vec` that annotates its own parameter (`let Vec = fun n: Int -> ...`) fails only at run time for a symbolic index, which it receives as the type `Vec(n)`, and an index variable reaches a recursive type only through an argument (`L(Vec(n))`); `let rec A = Union(Vec(n), A)` is an error (value position).
- A `Dyn`-to-`Union` boundary check is one native node that picks the first alternative whose shape holds, so a `Fun`-typed alternative gets the same contract a bare (non-union) `Fun` annotation would (callability at once, each call's result checked), not just an "is this callable" test; a function that is an alternative of a statically typed union does not have its domain checked at run time. A statically Union-typed value still can't be CALLED directly (`fun h: (Int -> Int) | Str -> h(5)` is rejected — `App`'s typecheck only knows how to call a `Fun`- or `Dyn`-typed callee); only reachable once the value has been passed through something that erases it to `Dyn` first (an unannotated parameter, a prelude higher-order builtin, ...).
- `cannot tell statically or at run time whether A fits B` is a static error when a union-typed source, at any depth (inside a tuple, list or record component, or under a function parameter or result), has an alternative differing from the target only in a function signature, a token type, a different recursive type or a length, because neither the type checker nor a run-time value test can decide that alternative (typical case: a union containing a typed function, passed where a different union or a function type needing a wrapper is expected). Annotate the source as `Dyn` to defer the check to run time. A target union is rejected as a whole when any alternative that could apply is undecidable, even if another alternative could have been decided by a value test.
- `++` requires one operand's type to fit the other's (the directional rule, so `[{x: 1}] ++ [{x: 1, y: 2}]` is a list of `{x: Int}` whose second value keeps `y`), and the result has that type; `x ++ 1` and tuple operands are static errors, and two function elements whose signatures name different effect row variables (`->{e1}`, `->{e2}`) are rejected. A `Dyn` element is checked against the other operand only when the other operand's type is the result: `[1] ++ [d]` checks `d`'s elements, `[d] ++ [1]` is a `[Dyn]` whose elements are checked where they are used. A type variable that meets `Dyn` inside `++` becomes `Dyn` (`fun p -> [p] ++ d` is `Dyn -> [Dyn]`). List literals, `::` and `if` that mix a concrete element with a `Dyn` one are not checked at the join (`[1, d]`).

## AI-assisted development

Much of renno was written with AI assistance (Claude Code). Every commit is authored and committed under the repository owner's name; commits made with Claude's help carry a `Co-Authored-By: Claude` trailer. At the time of writing that is 194 of the 216 commits (about 90%). Treat the git history as the record.

## License

MIT; see [LICENSE](LICENSE).
