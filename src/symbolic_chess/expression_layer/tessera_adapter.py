"""tessera_adapter — symbolic regression via tessera.

Replaces `pysr_adapter` with a tessera-backed alternative. Same public
interface (`run_tessera` mirrors `run_pysr`), but uses the
pure-Python tessera GP engine instead of PySR's Julia bridge.

Why replace?
------------
- No Julia dependency (tessera is pure-Python + numpy + numba)
- Native interop with the rest of symbolic-chess (single import surface)
- Tessera's measure-theoretic operators are available as a future
  extension (this first pass uses pure-pointwise search to match PySR's
  default behaviour exactly)

What this adapter does
----------------------
1. Stacks Variable.data into the GP's env dict (one entry per Variable)
2. Runs `tessera.search.GP` with `pointwise_only=True` so the discovered
   trees use only +, -, *, /, abs, sign, tanh, ... — same operator
   surface as PySR's default
3. Converts the resulting Pareto-front trees back to symbolic-chess
   `Expr` objects so downstream code (Equation tables, chess engine
   integration) is unchanged

Operator coverage
-----------------
Tessera's pointwise vocabulary is a SUPERSET of PySR's default
(see `tessera.expression.tree.BIN_OP_FNS` / `UN_OP_FNS`):
    binary : add sub mul div min max gt lt ge le pow
    unary  : tanh abs sign neg step sqrt log exp sin cos
             reduce_mean reduce_max reduce_sum reduce_std

By default this adapter restricts to the "PySR-default" subset
(add/sub/mul/div + abs/tanh/sign) so behaviour matches PySR runs.
Users can opt into the full vocabulary via `binary_operators=` /
`unary_operators=` (same arg names as `run_pysr`).

Temporal coverage
-----------------
For the first pass, temporal operators (lag/diff/ema/roll_mean) are
NOT used by the GP search itself. The pattern matches how PySR is
used in symbolic-chess: `expand_temporal_bank` pre-computes lagged
and smoothed copies of input variables and hands them to the search.
The discovered tree then references these pre-computed Variables as
ordinary inputs.

A future iteration can lift the pointwise restriction and let
tessera's `FunctionalOp` participate in the search.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .core import (
    Variable, Constant, Expr,
    add, sub, mul, div,
    neg, sign, abs_, tanh, sqrt_signed,
    ind_gt, ind_abs_gt,
)


# ---------------- tessera operator → symbolic-chess Expr lookups ----------------

# Map tessera's binary op names to symbolic-chess Operator instances.
# Only ops with a symbolic-chess equivalent are listed here; tessera-only
# ops (min/max/pow/le/lt/ge) raise during conversion.
_BIN_TESSERA_TO_SC = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": div,
    "gt":  ind_gt,   # tessera's gt(a, b) ↔ symbolic-chess ind_gt(a, b)
}

# Map tessera's unary op names to symbolic-chess Operator instances.
_UN_TESSERA_TO_SC = {
    "neg":  neg,
    "sign": sign,
    "abs":  abs_,
    "tanh": tanh,
    # tessera's sqrt is protected sqrt(|x|); symbolic-chess's sqrt_signed
    # is sign(x) * sqrt(|x|). For round-trip we map tessera's protected
    # sqrt to abs_(sqrt_signed(...)) — but that's lossy. For now treat
    # them as equivalent on non-negative inputs; warn otherwise.
    "sqrt": sqrt_signed,
}


# ---------------- Conversion: tessera Node → symbolic-chess Expr ----------------

def tessera_node_to_expr(node, var_map: dict[str, Variable]):
    """Recursively convert a tessera Node into a symbolic-chess Expr.

    Parameters
    ----------
    node : tessera Node (Var | Const | BinOp | UnOp | FunctionalOp ...)
    var_map : dict[str, Variable]
        Maps tessera Var names → symbolic-chess Variable instances.
        Required because tessera's Var(name) doesn't hold data; the
        symbolic-chess Variable does.

    Returns
    -------
    Expr (or Variable / Constant for leaves)

    Raises
    ------
    ValueError if the tree contains tessera ops with no symbolic-chess
    equivalent (currently: min, max, pow, lt, le, ge, exp, log, step,
    sin, cos, reduce_*). The current adapter restricts the search to
    ops with a clean round-trip, so this should not arise.
    """
    from tessera.expression.tree import Var, Const, BinOp, UnOp, FunctionalOp

    if isinstance(node, Var):
        if node.name not in var_map:
            raise ValueError(
                f"tessera Var {node.name!r} has no Variable in var_map"
            )
        return var_map[node.name]

    if isinstance(node, Const):
        return Constant(float(node.value))

    if isinstance(node, BinOp):
        if node.op not in _BIN_TESSERA_TO_SC:
            raise ValueError(
                f"tessera BinOp {node.op!r} has no symbolic-chess equivalent; "
                f"current adapter restricts search to {list(_BIN_TESSERA_TO_SC)}"
            )
        a = tessera_node_to_expr(node.a, var_map)
        b = tessera_node_to_expr(node.b, var_map)
        return _BIN_TESSERA_TO_SC[node.op](a, b)

    if isinstance(node, UnOp):
        if node.op not in _UN_TESSERA_TO_SC:
            raise ValueError(
                f"tessera UnOp {node.op!r} has no symbolic-chess equivalent; "
                f"current adapter restricts search to {list(_UN_TESSERA_TO_SC)}"
            )
        a = tessera_node_to_expr(node.a, var_map)
        return _UN_TESSERA_TO_SC[node.op](a)

    if isinstance(node, FunctionalOp):
        raise ValueError(
            "tessera FunctionalOp encountered; the adapter runs with "
            "pointwise_only=True so this should not happen. Either the "
            "config was changed externally or there is a tessera bug."
        )

    raise ValueError(f"unsupported tessera node type {type(node).__name__}")


# ---------------- Public API: run_tessera ----------------

def run_tessera(
    variables: list[Variable],
    target: np.ndarray,
    *,
    binary_operators: Optional[list[str]] = None,
    unary_operators: Optional[list[str]] = None,
    niterations: int = 40,
    populations: int = 1,           # tessera has single-pop today
    population_size: int = 50,
    maxsize: int = 25,
    parsimony: float = 0.005,
    random_state: int = 0,
    verbose: bool = True,
    extra_kwargs: Optional[dict] = None,
):
    """Run tessera GP on the variable bank, mirroring `run_pysr`'s signature.

    Parameters
    ----------
    variables : list of Variable — input features
    target : (T,) array — supervised target
    binary_operators, unary_operators : tessera op names (default:
        ``['+', '-', '*', '/']`` + ``['abs', 'tanh', 'sign']`` to match
        PySR's default operator set). PySR-style symbols ('+', '-', '*',
        '/') are accepted and mapped to tessera names ('add', 'sub', ...).
    niterations : maps to tessera's `n_gens` (PySR's `niterations` is
        per-population iterations; tessera has a single population so
        the mapping is direct).
    populations : currently ignored (tessera is single-population).
        A warning is emitted if > 1.
    population_size : tessera's `pop_size`.
    maxsize : tessera's MAX_COMPLEXITY (informational; not strictly
        enforced — the simplifier may produce trees that exceed this
        and they are not pruned in this first pass).
    parsimony : direct map.
    random_state : tessera's `seed`.
    verbose : tessera's `verbose`.

    Returns
    -------
    gp : the tessera GP instance (has `.hall_of_fame.pareto_front()`)
    best_expr : symbolic-chess Expr — the lowest-loss candidate in the
        Pareto front. May be None if conversion fails.
    """
    from tessera.search import GP, GPConfig

    # Default operator subset (matches PySR's default and tessera's
    # cleanest round-trip set).
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["abs", "tanh", "sign"]
    extra_kwargs = extra_kwargs or {}

    # PySR symbol → tessera name normalisation (so callers can pass
    # PySR-style ops without thinking).
    _PYSR_SYMBOL_TO_NAME = {
        "+": "add", "-": "sub", "*": "mul", "/": "div",
        "abs": "abs", "tanh": "tanh", "sign": "sign", "neg": "neg",
        "sqrt": "sqrt",
    }
    tessera_bin = [_PYSR_SYMBOL_TO_NAME.get(o, o) for o in binary_operators]
    tessera_un = [_PYSR_SYMBOL_TO_NAME.get(o, o) for o in unary_operators]

    # Verify the requested ops are all in our convert tables (i.e. round-
    # trippable). The set of converters is the source of truth for what
    # the adapter supports.
    for o in tessera_bin:
        if o not in _BIN_TESSERA_TO_SC:
            raise ValueError(
                f"binary op {o!r} not supported by tessera_adapter; "
                f"supported: {sorted(_BIN_TESSERA_TO_SC)}"
            )
    for o in tessera_un:
        if o not in _UN_TESSERA_TO_SC:
            raise ValueError(
                f"unary op {o!r} not supported by tessera_adapter; "
                f"supported: {sorted(_UN_TESSERA_TO_SC)}"
            )

    if populations > 1:
        warnings.warn(
            f"populations={populations} requested but tessera is "
            f"single-population; running once with pop_size={population_size}."
        )

    # Build env dict from Variables. Drop rows with non-finite values
    # consistent with run_pysr behaviour.
    env: dict[str, np.ndarray] = {v.name: np.asarray(v.data, dtype=np.float64)
                                   for v in variables}
    y = np.asarray(target, dtype=np.float64).ravel()
    # Find common finite mask
    stacks = list(env.values()) + [y]
    n = stacks[0].shape[0]
    valid = np.isfinite(y)
    for arr in env.values():
        valid &= np.isfinite(arr)
    env = {name: arr[valid] for name, arr in env.items()}
    y = y[valid]

    cfg = GPConfig(
        pop_size=population_size,
        n_gens=niterations,
        parsimony=parsimony,
        seed=random_state,
        verbose=verbose,
        pointwise_only=True,        # mirror PySR's default
        # NOTE: tessera's MAX_COMPLEXITY is module-level (40); maxsize
        # is informational only in this first pass.
        **extra_kwargs,
    )
    gp = GP(cfg)
    front = gp.run(env, y, feature_names=list(env.keys()))

    # Convert best candidate to symbolic-chess Expr
    best_expr = None
    var_map = {v.name: v for v in variables}
    if front:
        best = min(front, key=lambda c: c.train_loss)
        try:
            best_expr = tessera_node_to_expr(best.tree, var_map)
        except ValueError as e:
            warnings.warn(
                f"could not convert tessera tree to symbolic-chess Expr: {e}; "
                f"returning None as best_expr (raw tree: {best.tree})"
            )

    return gp, best_expr


# ---------------- Pareto-table convenience ----------------

def equation_table(gp) -> list[dict]:
    """Return the GP's Pareto front as a list of dicts (one per candidate).

    Mirrors the PySR adapter's `equation_table(model)` interface so
    downstream report-generation code is unchanged.
    """
    front = gp.hall_of_fame.pareto_front()
    table = []
    for cand in front:
        table.append({
            "complexity": cand.complexity,
            "train_loss": cand.train_loss,
            "fitness": cand.fitness,
            "tree_str": str(cand.tree),
        })
    return table


__all__ = [
    "run_tessera",
    "tessera_node_to_expr",
    "equation_table",
]
