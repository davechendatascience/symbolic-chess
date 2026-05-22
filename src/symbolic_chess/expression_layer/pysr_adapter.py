"""PySR adapter for the expression layer.

PySR is a *pointwise* (row-wise) symbolic regressor — it treats each (X[i], y[i])
row as independent. Our expression layer has *temporal* operators (lag, ema, diff)
that need the full series. We bridge the gap by:

  1. Pre-computing temporal expansions of input Variables (lagged, EMA-smoothed,
     diff'd) using our own Expr layer, producing a flat "variable bank".
  2. Feeding the bank to PySR for pointwise composition search.
  3. Parsing PySR's sympy-format output back into our Expr trees for downstream
     evaluation, composition, or auditing.

All `pysr`/`sympy` imports are deferred to function calls so this module imports
cleanly even when PySR is not installed.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from .core import (
    Variable, Constant, Operator, Expr, REGISTRY,
    add, sub, mul, div, neg, sign, abs_, tanh, sqrt_signed,
    lag as lag_op, ema as ema_op, diff as diff_op, roll_mean as roll_mean_op,
)


# ---------------- Variable bank construction ----------------

def expand_temporal_bank(
    variables: list[Variable],
    *,
    lags: tuple[int, ...] = (1, 3, 12, 24),
    ema_halflifes: tuple[int, ...] = (6, 24, 168),
    rolling_means: tuple[int, ...] = (),
    include_diffs: bool = True,
) -> list[Variable]:
    """Pre-compute lagged/EMA/diff/rolling-mean versions of input Variables.

    Returns a flat list of Variables for PySR's pointwise input.
    Names: f'{base}_lag{k}', f'{base}_ema{h}', f'{base}_diff{k}', f'{base}_ma{w}'.

    The original Variables are kept in the bank (lag/ema/diff are *extensions*).
    """
    bank: list[Variable] = list(variables)
    for v in variables:
        for k in lags:
            arr = lag_op(v, Constant(k)).evaluate()
            bank.append(Variable(f"{v.name}_lag{k}", arr))
        if include_diffs:
            for k in lags:
                arr = diff_op(v, Constant(k)).evaluate()
                bank.append(Variable(f"{v.name}_diff{k}", arr))
        for h in ema_halflifes:
            arr = ema_op(v, Constant(h)).evaluate()
            bank.append(Variable(f"{v.name}_ema{h}", arr))
        for w in rolling_means:
            arr = roll_mean_op(v, Constant(w)).evaluate()
            bank.append(Variable(f"{v.name}_ma{w}", arr))
    return bank


# ---------------- Search ----------------

def run_pysr(
    variables: list[Variable],
    target: np.ndarray,
    *,
    binary_operators: Optional[list[str]] = None,
    unary_operators: Optional[list[str]] = None,
    niterations: int = 40,
    populations: int = 20,
    population_size: int = 50,
    maxsize: int = 25,
    complexity_of_constants: int = 1,
    verbosity: int = 1,
    procs: int = 1,
    parsimony: float = 0.0032,
    random_state: int = 0,
    extra_kwargs: Optional[dict] = None,
):
    """Run PySR symbolic regression on the variable bank.

    Parameters
    ----------
    variables : list of Variable — the input features (typically the output of
        expand_temporal_bank to give PySR access to lagged/smoothed versions).
    target : (T,) array — supervised target (e.g. DP-optimal positions, forward
        return, gauss-extremum tag).
    binary_operators, unary_operators : PySR operator names (defaults are a small
        useful set). Names must be ones PySR understands ('+', '-', '*', '/', 'tanh',
        'abs', 'sign', 'sin', 'cos', 'exp', 'log', etc.).
    Other args : pass-through to PySRRegressor.

    Returns
    -------
    model : fitted PySRRegressor (has .equations_ Pareto front)
    best_expr : Expr or None — the best-loss equation as our Expr tree
                (None if conversion failed; check model.get_best() in that case).
    """
    from pysr import PySRRegressor   # deferred

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["abs", "tanh", "sign"]
    extra_kwargs = extra_kwargs or {}

    # Stack variable data
    X = np.column_stack([v.data for v in variables]).astype(np.float64)
    y = np.asarray(target, dtype=np.float64).ravel()
    var_names = [v.name for v in variables]

    # Drop NaN rows (temporal expansions create warm-up NaNs)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[valid]
    y = y[valid]

    model = PySRRegressor(
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        niterations=niterations,
        populations=populations,
        population_size=population_size,
        maxsize=maxsize,
        complexity_of_constants=complexity_of_constants,
        verbosity=verbosity,
        procs=procs,
        parsimony=parsimony,
        random_state=random_state,
        **extra_kwargs,
    )
    model.fit(X, y, variable_names=var_names)

    best_expr = None
    try:
        best_row = model.get_best()
        if best_row is not None:
            sympy_eq = best_row["sympy_format"]
            best_expr = sympy_to_expr(sympy_eq, variables)
    except Exception:
        pass

    return model, best_expr


# ---------------- Sympy → Expr conversion ----------------

def sympy_to_expr(sympy_expr, variables: list[Variable]) -> Expr:
    """Convert a sympy expression (PySR output) to our Expr tree.

    Parameters
    ----------
    sympy_expr : sympy.Expr
    variables : the Variable objects with names matching sympy Symbols in the expression

    Returns
    -------
    Expr (or Variable/Constant if sympy_expr is a leaf)
    """
    var_map = {v.name: v for v in variables}
    return _sympy_to_expr_recursive(sympy_expr, var_map)


def _sympy_to_expr_recursive(node, var_map):
    """Recursive sympy → Expr conversion."""
    import sympy

    # Leaves
    if isinstance(node, sympy.Symbol):
        name = str(node)
        if name in var_map:
            return var_map[name]
        raise ValueError(f"Unknown variable in sympy expression: {name!r}")
    if isinstance(node, (sympy.Integer, sympy.Float, sympy.Rational)):
        return Constant(float(node))
    # NumberSymbol like pi, E
    if hasattr(node, "is_number") and node.is_number:
        return Constant(float(node))

    # n-ary additive
    if isinstance(node, sympy.Add):
        args = list(node.args)
        result = _sympy_to_expr_recursive(args[0], var_map)
        for a in args[1:]:
            result = add(result, _sympy_to_expr_recursive(a, var_map))
        return result
    if isinstance(node, sympy.Mul):
        args = list(node.args)
        result = _sympy_to_expr_recursive(args[0], var_map)
        for a in args[1:]:
            result = mul(result, _sympy_to_expr_recursive(a, var_map))
        return result

    # Power: handle integer exponents via repeated mul; ÷ via Constant(1)/base
    if isinstance(node, sympy.Pow):
        base = _sympy_to_expr_recursive(node.base, var_map)
        exp = node.exp
        if isinstance(exp, sympy.Integer):
            e = int(exp)
            if e == 0:
                return Constant(1.0)
            if e == 1:
                return base
            if e >= 2:
                result = base
                for _ in range(e - 1):
                    result = mul(result, base)
                return result
            if e == -1:
                return div(Constant(1.0), base)
            if e <= -2:
                pos = -e
                pwr = base
                for _ in range(pos - 1):
                    pwr = mul(pwr, base)
                return div(Constant(1.0), pwr)
        # Non-integer exponent (e.g., sqrt) — approximate via sqrt_signed if 1/2
        if isinstance(exp, sympy.Rational) and exp.p == 1 and exp.q == 2:
            return sqrt_signed(base)
        # Unsupported exponent: raise so user knows
        raise NotImplementedError(
            f"Unsupported sympy Pow exponent: {exp} (only integers and 1/2 supported)"
        )

    # Unary functions
    fn = node.func
    if fn == sympy.tanh:
        return tanh(_sympy_to_expr_recursive(node.args[0], var_map))
    if fn == sympy.Abs:
        return abs_(_sympy_to_expr_recursive(node.args[0], var_map))
    if fn == sympy.sign:
        return sign(_sympy_to_expr_recursive(node.args[0], var_map))

    raise NotImplementedError(
        f"Unsupported sympy node {type(node).__name__}: {node} (func={node.func})"
    )


# ---------------- Recurrence-step discovery ----------------

def fit_recurrence_step(
    acc: np.ndarray,
    x: np.ndarray,
    y_next: np.ndarray,
    *,
    binary_operators: Optional[list[str]] = None,
    unary_operators: Optional[list[str]] = None,
    **pysr_kwargs,
):
    """Discover the per-step rule g(_acc, _x) -> next_acc as an Expr.

    Used when a recurrence acc_{t+1} = g(acc_t, x_t) is hypothesised and SR
    should discover g. Once g is found, wrap it with `fold(g, x, init)` or
    `scan(g, x, init)` to evaluate the recurrence on a series — the binding
    convention (_acc, _x) is what HigherOrderOperator's fold/scan use at
    each step.

    PySR sees (acc, x, y_next) as i.i.d. 2-input regression rows; temporal
    structure is *not* exploited here — that's the point. We're discovering
    the per-step rule from samples, then the higher-order op composes it.

    Parameters
    ----------
    acc : (N,) array of accumulator values at step t
    x   : (N,) array of input values at step t
    y_next : (N,) array of accumulator values at step t+1 (target)
    binary_operators / unary_operators : forwarded to run_pysr (defaults are
        a small set sufficient to discover affine recurrences like EMA/AR(1)).
    pysr_kwargs : forwarded to run_pysr (niterations, populations, etc.)

    Returns
    -------
    model : fitted PySRRegressor
    g_expr : Expr | None — best Pareto equation. Leaves reference Variables
             named `_acc` and `_x`; can be passed as the first child of
             `fold` or `scan` directly.

    Example
    -------
    >>> # Generate AR(1) samples: acc_{t+1} = 0.7*acc_t + x_t
    >>> rng = np.random.default_rng(0)
    >>> N = 1000
    >>> acc_t = rng.standard_normal(N)
    >>> x_t   = rng.standard_normal(N)
    >>> y     = 0.7 * acc_t + x_t
    >>> model, g = fit_recurrence_step(acc_t, x_t, y)
    >>> # Wrap g for evaluation on a real series:
    >>> series = Variable("series", rng.standard_normal(500))
    >>> ar1_out = scan(g, series, Constant(0.0)).evaluate()
    """
    acc_var = Variable("_acc", np.asarray(acc, dtype=float))
    x_var = Variable("_x", np.asarray(x, dtype=float))
    return run_pysr(
        [acc_var, x_var],
        np.asarray(y_next, dtype=float),
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        **pysr_kwargs,
    )


# ---------------- Convenience ----------------

def equation_table(model) -> list[dict]:
    """Return a clean list of dicts for the Pareto front equations.

    Each entry: {complexity, loss, score, sympy_format, equation}.
    """
    out = []
    try:
        eqs = model.equations_
        for _, row in eqs.iterrows():
            out.append({
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "score": float(row.get("score", float("nan"))),
                "sympy_format": row["sympy_format"],
                "equation": str(row["equation"]),
            })
    except Exception:
        pass
    return out


__all__ = [
    "expand_temporal_bank",
    "run_pysr",
    "fit_recurrence_step",
    "sympy_to_expr",
    "equation_table",
]
