# Higher-order operators in expression_layer — design sketch

**Status**: design doc, not yet implemented.

## What this is

A structural extension to `src/lib/expression_layer/core.py` that lets operators
take **sub-expressions as children** and evaluate them per-step under bound
variables. This is the operational form of "infinite sums" in symbolic
regression: instead of trying to express Σ_{k=0}^∞ as a closed-form primitive,
we let SR build accumulators / fixed-point recurrences as expression trees.

```
Current (first-order) operator:
    Expr.evaluate(env) collects child values, calls op.fn(*values).
    Children are leaves or already-evaluated arrays.

Proposed (higher-order) operator:
    Some children are held as Expr objects, not pre-evaluated.
    op.fn is given (sub_exprs, evaluated_args, env) and decides how/when
    to evaluate the sub_exprs — typically inside a loop, with the loop
    variables injected into env.
```

The minimal vocabulary added on top of this mechanism:

| Operator | Signature | Equivalent in math |
|---|---|---|
| `fold(g, x, init)` | g: Expr(acc,x), x: array, init: scalar | out_t = g(out_{t-1}, x_t), emit final |
| `scan(g, x, init)` | same | same, emit all out_t |

Everything else (`map_window`, `reduce`, `unfold`) can be added later via the
same mechanism without further structural change. `fold` + `scan` is what's
needed to validate the design.

## Why this matters

Three things become representable that currently aren't:

1. **Recursive accumulators as discoveries, not built-ins.** `ema(x, h)` is
   currently a primitive whose closed form `α x_t + (1−α) y_{t−1}` is hard-coded.
   With `fold`, SR can *rediscover* this form: `fold((acc, x) → α·x + (1−α)·acc, x, 0)`.
   That's a test the framework can run on itself.

2. **Hawkes-style and AR-family dynamics inside the search space.** Currently SR
   cannot find `y_t = β·y_{t−1} + x_t` because no operator expresses the
   self-reference. With `fold`, it lives natively in the tree.

3. **Fixed-point sums.** Σ_{k=0}^∞ r^k x_{t−k} = fold((acc, x) → r·acc + x, …).
   SR can converge on geometric series, exponential smoothers, IIR filters as
   *emergent* expressions — exactly the "infinite sums" wishlist item.

## What this won't fix

- **Compute cost.** A `fold` over T=1M bars is a Python loop, not a vectorised
  numpy op. Searching over expressions containing `fold` is 10–100× slower than
  current pointwise ops. Mitigation in §Performance.
- **Identifiability.** `fold(g₁,x,c₁) == fold(g₂,x,c₂)` is possible for many
  (g, c) pairs (e.g. any g that ignores acc reduces to a stateless map). SR's
  complexity penalty handles this only crudely.
- **PySR backend compatibility.** PySR's Julia kernel doesn't natively support
  higher-order ops. Two options below; the cleaner one bypasses PySR for fold/scan.

## Design choice — implementation path

| Path | Mechanism | Pros | Cons |
|---|---|---|---|
| **A1: HigherOrderOperator subclass** | New class; Operator/Expr dispatch on it; explicit lazy-child marking | Type-safe, extensible, future ops (`scan`, `map_window`, `reduce`) inherit machinery | New class hierarchy; ~150 LOC |
| **A2: Special-case in Expr.evaluate** | `if op.name in {"fold","scan"}: …` branch | Minimal diff (~30 LOC) | Doesn't generalise; every new HO op needs its own special case |

**Recommendation: A1.** Once the wishlist needs `scan` + `map_window` + `reduce`,
A2 becomes a tangle of special cases. The one-time architectural cost of A1 buys
a clean substrate for every higher-order op the project will want.

## Architecture

### `HigherOrderOperator` class

```python
@dataclass
class HigherOrderOperator(Operator):
    """Operator whose children include sub-expressions evaluated per-step.

    `lazy_children` is a tuple of child indices held as Expr (not pre-evaluated).
    All other children are evaluated normally before fn is called.

    The op's fn receives:
        fn(lazy_exprs: list[Expr], eager_values: list[ndarray|float], env: dict)
    and is responsible for invoking lazy_expr.evaluate(env=...) with the
    appropriate bindings at each step.
    """
    lazy_children: tuple[int, ...] = ()
```

`Expr.evaluate` learns one branch:

```python
def evaluate(self, env=None):
    if isinstance(self.op, HigherOrderOperator):
        lazy_idx = set(self.op.lazy_children)
        lazy_exprs = [self.children[i] for i in sorted(lazy_idx)]
        eager_vals = [c.evaluate(env) for i, c in enumerate(self.children)
                      if i not in lazy_idx]
        return self.op.fn(lazy_exprs, eager_vals, env or {})
    # existing first-order path unchanged
    vals = [c.evaluate(env) for c in self.children]
    return self.op.fn(*vals)
```

### `fold` and `scan` definitions

```python
def _fold_fn(lazy_exprs, eager_vals, env):
    g_expr = lazy_exprs[0]      # binary: takes acc, x
    x, init = eager_vals        # x is (T,) array; init is scalar
    x = np.asarray(x, dtype=float)
    acc = float(init)
    for t in range(len(x)):
        acc = float(g_expr.evaluate({**env, "_acc": acc, "_x": x[t]}))
    return acc

def _scan_fn(lazy_exprs, eager_vals, env):
    g_expr = lazy_exprs[0]
    x, init = eager_vals
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    acc = float(init)
    for t in range(len(x)):
        acc = float(g_expr.evaluate({**env, "_acc": acc, "_x": x[t]}))
        out[t] = acc
    return out

fold = _register(HigherOrderOperator(
    name="fold", arity=3, fn=_fold_fn, lazy_children=(0,)))
scan = _register(HigherOrderOperator(
    name="scan", arity=3, fn=_scan_fn, lazy_children=(0,)))
```

### Binding convention for the sub-expression

`g_expr` is built using two reserved Variable names: `_acc` (the running
accumulator) and `_x` (the current input element). E.g.:

```python
# EMA as a fold (rediscoverable):
acc = Variable("_acc", np.zeros(1))   # placeholder data; rebound per step
xt  = Variable("_x",   np.zeros(1))
alpha = Constant(0.1)
g = add(mul(alpha, xt), mul(sub(Constant(1.0), alpha), acc))
ema_via_fold = scan(g, x_array_variable, Constant(0.0))
```

The reserved names `_acc` / `_x` are part of the contract. SR/PySR searches over
g_expr are constrained to expressions whose leaves are in {`_acc`, `_x`, Constants}.

### PySR-adapter compatibility

PySR's Julia kernel doesn't accept higher-order operators directly. Two layers:

1. **PySR runs first-order only.** Discover g(acc, x) by treating it as a
   standalone two-input regression: build a synthetic dataset where rows are
   (acc_t, x_t, acc_{t+1}) sampled from a reference trajectory; PySR fits
   acc_{t+1} ≈ g(acc_t, x_t). Then wrap the result in `fold` / `scan` for
   evaluation in expression_layer.
2. **Native GP (no PySR) searches over fold-containing trees.** Add a small
   pure-Python GP loop that mutates/crosses-over trees including `fold` /
   `scan`. Used when the unknown is the recursive structure itself, not the
   per-step rule.

Layer 1 handles 90% of the use cases (we know it's a recursion, want to
discover the per-step rule). Layer 2 stays optional — only build it when a
benchmark requires it.

## Performance

Pure-Python per-step evaluation of `g_expr` inside a length-T loop is the bottle-
neck. Mitigations:

- **Compile g_expr once.** Walk the tree, build a callable that takes
  (acc, x_value, *consts) → next_acc using closures or `eval()` of generated
  source. ~10× speedup; cleanly testable.
- **Vectorise when g is affine in acc.** Detect g(acc, x) = a(x)·acc + b(x);
  for affine recurrences, closed-form filtering is O(T) with numpy vector
  ops. Optional fast path.
- **Cap T during search.** PySR-style search uses T ≤ 10⁴; final-fit/eval uses
  full T. Standard SR practice.

## Test plan

`tests/expression_layer/test_higher_order.py`:

1. **EMA equivalence.** Build EMA two ways — via `core.ema(x, h)` and via
   `scan(g, x, init)` with g = α·_x + (1−α)·_acc. Assert array equality on a
   random series.
2. **Cumsum via fold.** scan((acc, x) → acc + x, x, 0) == np.cumsum(x).
3. **Geometric series convergence.** fold((acc, x) → r·acc + x, ones(N), 0)
   approaches 1/(1−r) for r∈(0,1) as N→∞; check |out − 1/(1−r)| < 1e-3 at N=200.
4. **PySR rediscovery (slow test).** Generate y_t from a known g (e.g.,
   AR(1) y = 0.7·y_{t−1} + x). Feed (acc_t, x_t, y_t) rows to PySR with binary
   ops; verify recovered expression matches the AR(1) form within complexity 5.
5. **Backward compat.** All existing `core.py` tests pass unchanged.

## Acceptance criteria

- A1 implemented; HigherOrderOperator + fold + scan in `core.py`
- All existing tests pass
- New tests above pass
- PySR adapter has a `fit_recurrence_step(x, y)` helper that returns the
  per-step rule g as an Expr
- One worked example in `benchmarks/run_fold_rediscover_ema.py`: PySR
  rediscovers EMA from data, output written to `benchmarks/results/`

## Build order

1. `HigherOrderOperator` + dispatch in Expr.evaluate (1 day, 100 LOC)
2. `fold` + `scan` + tests 1–3 above (½ day)
3. `pysr_adapter.fit_recurrence_step` (½ day)
4. EMA-rediscovery benchmark, test 4, results doc (½ day)

Total: ~2.5 dev-days. Independent of any benchmark; can ship standalone.
