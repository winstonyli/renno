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
- **Tuples**: `(a, b, c)` — a fixed-arity, per-position-typed product, distinct from a variable-length `List`. Desugars to a plain list at runtime (no new value representation), typed `Type::Tuple` for real positional structural comparison. `(p, q)` also works as a pattern, sugar for `[p, q]`.
- **First-class `opaque`**: `opaque` is an expression, not just a `data`-block marker — writing it anywhere yields a token unique to that exact source position (the same token every time that code runs, since it's a literal, not a generator), bindable, passable, and comparable with `==`. Combined with tuples, this builds a hand-rolled nominal type with no `data` block at all: `let Meters = fun n -> (n, opaque) in ...` — two `Meters(_)` values always carry equal tokens, and a same-shaped tuple from anywhere else never does.
- **Union types and type aliases**: `type Name = TypeExpr in ...` names any type expression, and `A | B` unions two or more into one — a self-contained "one of these" with no registry lookup, usable in any annotation (`fun x: Int | Str -> ...`). A `Dyn` value crossing a Union-typed boundary is accepted if it shallowly matches *any* alternative; `match` exhaustiveness over a Union-typed scrutinee is proven when every alternative (not necessarily by the same arm) is covered by some pattern — `type Pair = (Int,) | (Int, Int) in ... | (n,) -> n | (n, m) -> n + m` is exhaustive this way, even though neither arm alone covers the whole union. (`Type::Token`, `opaque`'s own static type, has no surface spelling yet — a hand-rolled `opaque`-tagged sum's *values* work today, but its type can't be named in a `type` alias for exhaustiveness checking until that gap closes.)
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

Functions are curried closures; `fun a -> fun b -> ...` and calling `f(a)(b)` are the normal shape. Application also works by juxtaposition, `f a b` — the two mix freely, and negating a juxtaposed argument needs its own parens (`f (-1)`, not `f -1`, the same resolution Haskell/OCaml use: unary `-` binds looser than application, so `f -1` parses as `f - 1`):

```
let add = fun a -> fun b -> a + b in add(1)(2)   -- 3
let add = fun a -> fun b -> a + b in add 1 2     -- 3, identical
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

### Tuples and first-class `opaque`

```
let p = (1, "a", true) in
match p with
| (a, b, c) -> a   -- 1
```

`(a, b, ...)` is a fixed-arity product — typed positionally (`Type::Tuple`), not widened to a common element type the way `List` elements are. `(a)` with no comma stays ordinary parenthesized grouping; a tuple pattern `(p, q)` is sugar over the list pattern `[p, q]`, since a tuple is a plain list at runtime.

`opaque` isn't only a `data`-block marker — it's an expression in its own right, evaluating to a token unique to wherever it's written:

```
let brand = opaque in
brand == opaque   -- false: a DIFFERENT `opaque`, a different token
brand == brand     -- true: same token, same binding
```

Combined with tuples, this builds a nominal type with no `data` block at all — `opaque`, written once inside a function body, is a literal: the same token every call, since it's the same source position evaluated fresh each time, not a fresh token per call:

```
let Meters = fun n -> (n, opaque) in
let a = Meters(5) in
let b = Meters(10) in
match a with
| (an, at) -> match b with
  | (bn, bt) -> at == bt   -- true: same function, same embedded `opaque`
```

A same-shaped tuple built by a *different* function never compares equal, the same distinction `data`'s own `opaque` field draws — just reached through ordinary tuples and structural comparison instead of a dedicated brand mechanism.

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
- `parser.rs` — recursive-descent parser into an arena-allocated AST (`expr.rs`); iteratively flattens long `let`/`fun`/`data` chains to keep native stack usage O(1) regardless of chain length.
- `typecheck.rs` — bidirectional-lite gradual type checker: infers types and effect rows in one pass, desugars a `Dyn`-to-concrete boundary into an ordinary `if <predicate> then value else fail(...)` (a real per-call contract for a `Fun` target) built from prelude builtins rather than a dedicated AST node, and checks match exhaustiveness/reachability.
- `machine.rs` — a trampolined CEK-style step loop (`cont.rs` holds the defunctionalized continuation frames). No native recursion during evaluation, so no stack-overflow risk from deep programs or from resuming captured continuations.
- `value.rs` — runtime value representation.
- `plist.rs` — the persistent linked list `Env` and typecheck's context are both built from.
- `span.rs` — source locations and the snippet-with-caret error rendering shared by every error path.

`run_source` (`lib.rs`) runs the whole parse → typecheck → run pipeline on a dedicated large-stack worker thread, since the parser and type checker are ordinary native recursive descent (unlike the machine, which is trampolined).

## Known limitations

- Effect-row inference doesn't look inside a handler clause's own body — what a handler does when it resumes isn't modeled.
- A runtime type check at a `Dyn`-to-`Data(name)` boundary (the `check_data_shape` builtin) confirms the value's constructor tag and field count match one of the annotated type's own declared constructors, plus its brand tag when the annotated type is `opaque` — not full field-type recursion, so a value shaped exactly like some OTHER structurally-consistent type (differently named, same ctor names/arities, e.g. `Celsius`/`Fahrenheit`) is still accepted, by design (structural typing). A branded value's hidden tag is still not hidden from `Value`'s own `Display`/`Outcome` conversion — a branded value printed or returned at the top level shows an extra trailing `<brand>` element, since neither knows a value's static type well enough to leave it out (the id itself is never printed: the brand tag is its own `Value::Token`/`Outcome::Token` kind, not a plain `Int`, so it can't be mistaken for one either). Two separately-declared `data` blocks that reuse the same type name (shadowing) are now correctly distinguished when either is `opaque` (`Type::Data`'s brand id, resolved by `Parser::type_brands`/`Parser::branded_ctors`) — only an UNBRANDED same-named shadowing redeclaration still unifies structurally with itself by name, which is the intended default (same as any other structurally-identical pair).
- Match exhaustiveness and reachability are checked only where cheaply provable (see the doc comments on `missing_case`/`first_unreachable` in `typecheck.rs`); anything past that silently falls back to a runtime panic.
- A runtime panic's reported location generally falls back to the last expression *evaluated*, not necessarily the exact sub-expression at fault a few steps later. Two common cases are threaded through precisely instead of relying on that fallback: `apply_binop`'s operand-type/div-by-zero/etc panics blame whichever operand is actually at fault (or their combined span, when neither alone explains it), and "attempt to call a non-function value" blames the callee, not an unrelated argument evaluated afterward. Everywhere else (builtin argument-type panics, match failure, ...) still uses the last-evaluated fallback.
- `where` refinement predicates are limited to what `<`/`<=`/`>`/`>=`/`==`/`!=`, `&&`/`||`/`!`, and arithmetic can express. Proving is attempted only when the bound value reduces to a closed Int constant at parse time (`try_eval_closed_int`); a Lambda parameter's refinement is never proven statically, since its actual value is unknown until a caller supplies one.
- A runtime type check at a `Dyn`-to-`Tuple` boundary confirms arity only (is this a list of the right length), not that each position's own value matches its own element type — a full per-position recursive check is possible but not built yet. There's no surface syntax to write `Type::Token` in a type annotation (only `opaque`-as-expression infers one); static rejection of a mismatched opaque-tuple value at an annotated boundary isn't reachable from source yet, though the underlying `consistent_inner` comparison and runtime `==` both already work. Match exhaustiveness for nested tuple patterns only recognizes a single, fully-nested pattern in ONE `match` (`((p, q), x)`) — a scrutinee first bound by an outer match/lambda pattern loses its precise Tuple type (renno has no pattern-driven type refinement), so chaining separate `match`es over the same value falls back to "possibly non-exhaustive."
- A `Dyn`-to-`Union` boundary check picks the matching alternative via a shallow shape test (build_shape_predicate), then runs THAT alternative's own full `build_boundary_check` — so a `Fun`-typed alternative gets the same real per-call contract (`wrap_fun_contract`) a bare (non-union) `Fun` annotation would, not just an "is this callable" test. A statically Union-typed value still can't be CALLED directly (`fun h: (Int -> Int) | Str -> h(5)` is rejected — `App`'s typecheck only knows how to call a `Fun`- or `Dyn`-typed callee); only reachable once the value has been passed through something that erases it to `Dyn` first (an unannotated parameter, `map`'s own native `apply`, ...).
