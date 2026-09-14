# renno

A small scripting language with algebraic effect handlers and gradual typing, implemented in Rust as a trampolined CEK-style abstract machine.

```
let rec fact = fun n -> if n == 0 then 1 else n * fact(n - 1) in fact(10)
```

## Highlights

- **Algebraic effects**: `perform`/`handle`, first-class handler values, deep and shallow semantics, and genuine multi-shot resumption (a captured continuation can be resumed zero, one, or many times).
- **Gradual typing**: every position is `Dyn` unless annotated. Annotated code is checked statically and pays no runtime cost; annotated boundaries crossed by an unannotated (`Dyn`) value get a runtime check inserted automatically.
- **Gradual verification**: `let n: Int where 0 < n = 5 in ...` — a refinement predicate proven outright when it cheaply can be (a literal value, zero runtime cost), and backed by a real runtime check otherwise. Same three-way shape as gradual typing's own boundary checks, generalized from type tags to arbitrary decidable predicates.
- **Closed and row-polymorphic effect typing**: `check()` statically rejects a program if it can prove an effect is never handled. Row-polymorphic function types (`(Dyn ->{e} Dyn)`) let effect-safety survive through higher-order calls.
- **`let rec` and mutual recursion**: `let rec f = ... and g = ... in ...` — any function in the group can call any sibling (including itself) by name.
- **Pattern matching**: literals, lists (`[]`, `[a, b]`, `h :: t`), and ADT constructors, with static exhaustiveness and reachability checking wherever those are cheaply provable.
- **ADTs**: `data Option = None | Some(Int) in ...`, desugared entirely into tagged lists and ordinary pattern matching — no new runtime representation for the common case. Constructors get real types (`Type::Data`), **structural by default** — two differently-named types with the same constructor names and field types unify — with an opt-in `opaque` field to make a type nominal: a hidden per-declaration tag, checked both statically and at every construct/destructure, so it's never interchangeable with anything but itself.
- **Named fields**: `data Point = Point(x: Int, y: Int) in ...` supports `p.x` access, `Point { x: 1, y: 2 }` construction (any order), and `Point { x: a, y: b }` patterns (any order) — all sugar over ordinary positional construction and pattern matching, for any `data` type with exactly one constructor.
- **Diagnostics**: every parse error, type error, and runtime panic reports a `line, column` location with a source snippet and a caret, not just a bare message.
- **Multi-line REPL**: `let`/`match`/`data` blocks spanning multiple lines can be typed directly at the prompt.

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

Functions are curried closures; `fun a -> fun b -> ...` and calling `f(a)(b)` are the normal shape:

```
let add = fun a -> fun b -> a + b in add(1)(2)   -- 3
```

`let rec` makes a binding visible inside its own value, for recursion that an ordinary `let` can't express:

```
let rec fact = fun n -> if n == 0 then 1 else n * fact(n - 1) in fact(5)   -- 120
```

`and` extends this to a group of mutually recursive functions:

```
let rec is_even = fun n -> if n == 0 then true else is_odd(n - 1)
and is_odd = fun n -> if n == 0 then false else is_even(n - 1)
in is_even(10)
```

### Pattern matching

```
match xs with
| [] -> 0
| h :: t -> h + sum(t)
```

A match is rejected statically if it's missing an obviously necessary case (`[]`/`h :: t` both present, `true`/`false` both present, or every constructor a `data` type declared), and if an earlier arm already covers everything a later one would ever match.

`::` also works as an expression (`h :: t` prepends `h` onto list `t`), not just a pattern, so list-building functions can be written by hand:

```
let rec map = fun f -> fun xs ->
  match xs with
  | [] -> []
  | h :: t -> f(h) :: map(f)(t)
in map(fun x -> x * 2)([1, 2, 3])   -- [2, 4, 6]
```

### ADTs and named fields

```
data Option = None | Some(Int) in
match Some(5) with
| None -> 0
| Some(x) -> x
```

```
data Point = Point(x: Int, y: Int) in
let p = Point { x: 3, y: 4 } in
match p with
| Point { y: b, x: a } -> a * a + b * b   -- 25
```

Positional construction and patterns (`Point(3)(4)`, `Point { x, y }` written `Point(x, y)`) still work and freely mix with the named forms above — named fields are an alternative notation, not a replacement. Constructors are curried like any other multi-argument callable (`Point(3)(4)`, not `Point(3, 4)`).

`data` types are structural: two differently-named types with the same constructor names and field types are interchangeable, same as List or Fun.

```
data Meters = Mk(Int) in
data Seconds = Mk(Int) in
let f = fun x: Meters -> x in f(Mk(5))   -- accepted: Seconds's Mk(5) satisfies a Meters annotation
```

Add an `opaque` field (required on every constructor of the block, or none) to opt a type out of that. It's a real hidden field, not just a static marker: every constructor stamps one invisible tag — unique to that `data` block's own source position — into its value, and every pattern that can match it carries the same tag, so a value only ever destructures against the exact declaration that produced it:

```
data Meters = Mk(Int, opaque) in
data Seconds = Mk(Int, opaque) in
let f = fun x: Meters -> x in f(Mk(5))   -- static error: each `opaque` type is only consistent with itself
```

That static check is only half the story — the hidden tag also means a value that somehow reaches a branded type's pattern from the wrong declaration (e.g. crossing a `Dyn` boundary, or a shadowed same-named `data` redeclaration) fails to *match* at runtime instead of silently behaving like the wrong type.

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
  f(y)                                    -- a runtime Check is inserted here
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
- `parser.rs` — recursive-descent parser into an arena-allocated AST (`expr.rs`); iteratively flattens long `let`/`fun`/`data` chains to keep native stack usage O(1) regardless of chain length.
- `typecheck.rs` — bidirectional-lite gradual type checker: infers types and effect rows in one pass, inserts `Check` nodes (or full higher-order contracts) only at `Dyn`-to-concrete boundaries, and checks match exhaustiveness/reachability.
- `machine.rs` — a trampolined CEK-style step loop (`cont.rs` holds the defunctionalized continuation frames). No native recursion during evaluation, so no stack-overflow risk from deep programs or from resuming captured continuations.
- `value.rs` — runtime value representation.
- `plist.rs` — the persistent linked list `Env` and typecheck's context are both built from.
- `span.rs` — source locations and the snippet-with-caret error rendering shared by every error path.

`run_source` (`lib.rs`) runs the whole parse → typecheck → run pipeline on a dedicated large-stack worker thread, since the parser and type checker are ordinary native recursive descent (unlike the machine, which is trampolined).

## Known limitations

- Effect-row inference doesn't look inside a handler clause's own body — what a handler does when it resumes isn't modeled.
- A runtime type check at a `Dyn`-to-`Data(name)` boundary can only confirm "this is some tagged value," not "specifically this data type" — no type name is stamped into an UNBRANDED value at runtime (an `opaque`-branded one does carry a tag, but it's only ever checked by actually pattern-matching/field-accessing the value, not by a bare type annotation or `Dyn` boundary check, and not hidden from `Value`'s own `Display`/`Outcome` conversion — a branded value printed or returned at the top level shows its raw tag as an extra trailing element, since neither knows a value's static type well enough to leave it out). Two separately-declared `data` blocks that reuse the same type name (shadowing) still statically unify regardless of brand — the runtime tag bounds the damage (a mismatched value fails to destructure) but doesn't close the static gap.
- Match exhaustiveness and reachability are checked only where cheaply provable (see the doc comments on `missing_case`/`first_unreachable` in `typecheck.rs`); anything past that silently falls back to a runtime panic.
- A runtime panic's reported location is the last expression *evaluated*, not necessarily the exact sub-expression at fault a few steps later.
- `where` refinement predicates are limited to what `<`/`<=`/`>`/`>=`/`==`/`!=`, `&&`/`||`/`!`, and arithmetic can express. Proving is attempted only when the bound value reduces to a closed Int constant at parse time (`try_eval_closed_int`); a Lambda parameter's refinement is never proven statically, since its actual value is unknown until a caller supplies one.
