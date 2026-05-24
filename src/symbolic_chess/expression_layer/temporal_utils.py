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
