"""Expression layer: uniform tree representation for symbolic-regression /
genetic-programming search over time-series functions.

For fast per-row scalar evaluation (e.g., chess engine, online prediction),
use `compile_expr(expr, var_names) → callable(features_1d) → float`. This
walks the tree once and compiles a Python lambda using the `math` module
for true scalar speed — ~50-100x faster than recursive Expr.evaluate per call.
Pointwise ops only — temporal/higher-order ops require batch/sequence
context and raise NotImplementedError.


Pure-numpy core (no domain-specific imports).

Core abstractions:
    Variable     — named time-series leaf, evaluates to a (T,) array
    Constant     — scalar leaf, evaluates to a Python float
    Operator     — base operation: name, arity, callable, display symbol
    Expr         — internal node: Operator + list of children (Variable | Constant | Expr)
    REGISTRY     — dict of named operators, the "alphabet" available to search

Every node has .evaluate(env=None) → array/scalar, .complexity() → int,
.leaves() → set[str] of variable names used.

Operators that take parameters (lag, ema, indicator with threshold) accept those
as Constant children, so the tree is purely composable — no special parameter slots.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Union, Optional
import numpy as np


# ---------------- Leaves ----------------

@dataclass
class Variable:
    """A named time-series leaf."""
    name: str
    data: np.ndarray   # shape (T,)

    def evaluate(self, env: Optional[dict] = None) -> np.ndarray:
        if env is not None and self.name in env:
            return env[self.name]
        return self.data

    def __repr__(self) -> str:
        return self.name

    def to_string(self) -> str:
        return self.name

    def complexity(self) -> int:
        return 1

    def leaves(self) -> set[str]:
        return {self.name}


@dataclass
class Constant:
    """A scalar constant. Broadcasts in arithmetic with array operands."""
    value: float

    def evaluate(self, env: Optional[dict] = None) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"{self.value:g}"

    def to_string(self) -> str:
        return f"{self.value:g}"

    def complexity(self) -> int:
        return 1

    def leaves(self) -> set[str]:
        return set()


# ---------------- Operators ----------------

@dataclass
class Operator:
    """Base operator: callable, arity, display symbol.

    Calling an Operator with Expr-like children returns an Expr node:
        add(x, y)  →  Expr(op=add, children=[x, y])

    Parameterized operators (lag, ema, indicator threshold) accept the parameter
    as an additional Constant child, e.g. lag(x, Constant(3)) for lag-by-3.
    """
    name: str
    arity: int
    fn: Callable
    symbol: Optional[str] = None
    is_infix: bool = False   # True for +, -, *, / (display as `a op b`)

    def __post_init__(self):
        if self.symbol is None:
            self.symbol = self.name

    def __call__(self, *args) -> "Expr":
        if len(args) != self.arity:
            raise ValueError(f"operator {self.name} expects {self.arity} args, got {len(args)}")
        return Expr(op=self, children=list(args))


@dataclass
class HigherOrderOperator(Operator):
    """Operator whose children include sub-expressions evaluated per-step.

    `lazy_children` is a tuple of child indices held as Expr (not pre-evaluated).
    All other children are evaluated normally before fn is called.

    The op's fn receives:
        fn(lazy_exprs: list[Expr], eager_values: list, env: dict)
    and is responsible for invoking lazy_expr.evaluate(env=...) with the
    appropriate bindings at each step (e.g. {"_acc": acc, "_x": x_t}).

    Reserved Variable names for the binding convention: `_acc` (running
    accumulator) and `_x` (current input element). Sub-expressions for fold/
    scan should be built referencing Variables with these names.
    """
    lazy_children: tuple = ()


# ---------------- Expression node ----------------

@dataclass
class Expr:
    """Internal expression tree node: Operator + children."""
    op: Operator
    children: list

    def evaluate(self, env: Optional[dict] = None) -> np.ndarray:
        if isinstance(self.op, HigherOrderOperator):
            lazy_idx = set(self.op.lazy_children)
            lazy_exprs = [self.children[i] for i in sorted(lazy_idx)]
            eager_vals = [c.evaluate(env) for i, c in enumerate(self.children)
                          if i not in lazy_idx]
            return self.op.fn(lazy_exprs, eager_vals, env if env is not None else {})
        vals = [c.evaluate(env) for c in self.children]
        return self.op.fn(*vals)

    def __repr__(self) -> str:
        return self.to_string()

    def to_string(self) -> str:
        if self.op.is_infix and len(self.children) == 2:
            return f"({_child_str(self.children[0])} {self.op.symbol} {_child_str(self.children[1])})"
        args = ", ".join(_child_str(c) for c in self.children)
        return f"{self.op.symbol}({args})"

    def complexity(self) -> int:
        return 1 + sum(c.complexity() for c in self.children)

    def leaves(self) -> set[str]:
        out: set[str] = set()
        for c in self.children:
            out |= c.leaves()
        return out


def _child_str(c) -> str:
    if hasattr(c, "to_string"):
        return c.to_string()
    return repr(c)


# ---------------- Helper functions for operators ----------------

def _safe_div(x, y):
    """x / y with 0 returned for |y| < eps."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.true_divide(x, y)
    if np.isscalar(result):
        return float(result) if np.isfinite(result) else 0.0
    result = np.where(np.isfinite(result), result, 0.0)
    return result


def _lag_fn(x, k):
    """Shift array right by k (zero-padded)."""
    k = int(k)
    if k <= 0:
        return np.asarray(x, dtype=float).copy()
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    if k < len(x):
        out[k:] = x[:-k]
    return out


def _diff_fn(x, k):
    """x_t - x_{t-k}."""
    x = np.asarray(x, dtype=float)
    return x - _lag_fn(x, k)


def _ema_fn(x, halflife):
    """Exponential moving average with given halflife."""
    halflife = float(halflife)
    if halflife <= 0:
        return np.asarray(x, dtype=float).copy()
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i-1]
    return out


def _rolling_mean(x, w):
    """Trailing rolling mean over a window of w bars."""
    w = int(w)
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return x.copy()
    out = np.full_like(x, np.nan)
    if len(x) < w:
        return out
    cs = np.concatenate(([0.0], np.cumsum(x)))
    out[w-1:] = (cs[w:] - cs[:-w]) / w
    return out


def _rolling_std(x, w):
    """Trailing rolling std over a window of w bars."""
    w = int(w)
    x = np.asarray(x, dtype=float)
    if w <= 1:
        return np.zeros_like(x)
    mean = _rolling_mean(x, w)
    sq_mean = _rolling_mean(x * x, w)
    var = np.maximum(sq_mean - mean * mean, 0.0)
    return np.sqrt(var)


def _ind_gt(x, tau):
    return (np.asarray(x) > float(tau)).astype(float)


def _ind_abs_gt(x, tau):
    return (np.abs(np.asarray(x)) > float(tau)).astype(float)


def _sign_change(x):
    x = np.asarray(x, dtype=float)
    s = np.sign(x)
    if len(s) == 0:
        return s
    return (np.diff(s, prepend=s[0]) != 0).astype(float)


def _sqrt_signed(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.sqrt(np.abs(x))


# ---------------- Higher-order operators ----------------

def _fold_fn(lazy_exprs, eager_vals, env):
    """fold(g, x, init): out_t = g(out_{t-1}, x_t), returns final out_T.

    Sub-expression g must reference reserved Variable names `_acc` and `_x`.
    """
    g_expr = lazy_exprs[0]
    x, init = eager_vals
    x = np.asarray(x, dtype=float)
    acc = float(init)
    for t in range(len(x)):
        acc = float(np.asarray(g_expr.evaluate({**env, "_acc": acc, "_x": float(x[t])})))
    return acc


def _scan_fn(lazy_exprs, eager_vals, env):
    """scan(g, x, init): out_t = g(out_{t-1}, x_t), emits all out_t as array.

    Sub-expression g must reference reserved Variable names `_acc` and `_x`.
    """
    g_expr = lazy_exprs[0]
    x, init = eager_vals
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    acc = float(init)
    for t in range(len(x)):
        acc = float(np.asarray(g_expr.evaluate({**env, "_acc": acc, "_x": float(x[t])})))
        out[t] = acc
    return out


# ---------------- Registry ----------------

REGISTRY: dict[str, Operator] = {}


def _register(op: Operator) -> Operator:
    REGISTRY[op.name] = op
    return op


# arithmetic (binary, infix)
add = _register(Operator("add", 2, np.add, "+", is_infix=True))
sub = _register(Operator("sub", 2, np.subtract, "-", is_infix=True))
mul = _register(Operator("mul", 2, np.multiply, "*", is_infix=True))
div = _register(Operator("div", 2, _safe_div, "/", is_infix=True))

# pointwise nonlinear (unary)
neg = _register(Operator("neg", 1, np.negative, "neg"))
sign = _register(Operator("sign", 1, np.sign, "sign"))
abs_ = _register(Operator("abs", 1, np.abs, "abs"))
tanh = _register(Operator("tanh", 1, np.tanh, "tanh"))
sqrt_signed = _register(Operator("sqrt_signed", 1, _sqrt_signed, "ssqrt"))

# temporal (binary: input, parameter)
lag = _register(Operator("lag", 2, _lag_fn, "lag"))
diff = _register(Operator("diff", 2, _diff_fn, "diff"))
ema = _register(Operator("ema", 2, _ema_fn, "ema"))
roll_mean = _register(Operator("roll_mean", 2, _rolling_mean, "ma"))
roll_std = _register(Operator("roll_std", 2, _rolling_std, "sd"))

# crossing / regime
ind_gt = _register(Operator("ind_gt", 2, _ind_gt, "I_gt"))
ind_abs_gt = _register(Operator("ind_abs_gt", 2, _ind_abs_gt, "I_absgt"))
sign_change = _register(Operator("sign_change", 1, _sign_change, "dsign"))

# higher-order (sub-expression as first child)
fold = _register(HigherOrderOperator(
    name="fold", arity=3, fn=_fold_fn, symbol="fold", lazy_children=(0,)))
scan = _register(HigherOrderOperator(
    name="scan", arity=3, fn=_scan_fn, symbol="scan", lazy_children=(0,)))


# ---------------- compile_expr: fast per-row scalar callable ----------------

import math as _math


def _safe_div_scalar(a, b):
    return a / b if b != 0 else 0.0


def _sqrt_signed_scalar(x):
    return _math.copysign(_math.sqrt(abs(x)), x)


# Template strings for emit; {a}, {b} are placeholders for compiled children.
# Each is plain Python — no numpy — so the resulting lambda is true scalar speed.
_COMPILE_OP_TEMPLATES = {
    "add": "({a} + {b})",
    "sub": "({a} - {b})",
    "mul": "({a} * {b})",
    "div": "_safe_div({a}, {b})",
    "neg": "(-({a}))",
    "abs": "abs({a})",
    "sign": "(1.0 if ({a}) > 0 else (-1.0 if ({a}) < 0 else 0.0))",
    "tanh": "_math.tanh({a})",
    "sqrt_signed": "_sqrt_signed({a})",
    "ind_gt": "(1.0 if ({a}) > ({b}) else 0.0)",
    "ind_abs_gt": "(1.0 if abs({a}) > ({b}) else 0.0)",
}


def _compile_emit(node, var_idx: dict) -> str:
    """Walk an Expr tree, emit a Python source string for scalar evaluation."""
    if isinstance(node, Variable):
        if node.name not in var_idx:
            raise ValueError(
                f"Variable {node.name!r} not in var_names; "
                f"compile_expr needs all leaves declared."
            )
        return f"x[{var_idx[node.name]}]"
    if isinstance(node, Constant):
        return repr(float(node.value))
    if isinstance(node, Expr):
        if isinstance(node.op, HigherOrderOperator):
            raise NotImplementedError(
                f"compile_expr cannot compile higher-order op {node.op.name!r}. "
                f"Use Expr.evaluate() for fold/scan; compile_expr is for pointwise "
                f"per-row scalar evaluation only."
            )
        template = _COMPILE_OP_TEMPLATES.get(node.op.name)
        if template is None:
            raise NotImplementedError(
                f"compile_expr does not support op {node.op.name!r} "
                f"(temporal or unregistered). Pointwise ops only."
            )
        children_src = [_compile_emit(c, var_idx) for c in node.children]
        if node.op.arity == 1:
            return template.format(a=children_src[0])
        if node.op.arity == 2:
            return template.format(a=children_src[0], b=children_src[1])
        raise NotImplementedError(f"arity {node.op.arity} not supported by compile_expr")
    raise TypeError(f"Unknown node type for compile_expr: {type(node).__name__}")


def compile_expr(expr, var_names: list):
    """Compile an Expr tree to a fast scalar callable.

    Returns a function `f(features: array-like) → float` where `features[i]`
    corresponds to `var_names[i]`.

    The compiled callable is 50-100x faster than `expr.evaluate({name: val, ...})`
    for per-row evaluation (e.g., chess alpha-beta leaves, per-tick prediction).

    Pointwise operators only: arithmetic, neg, abs, sign, tanh, sqrt_signed,
    ind_gt, ind_abs_gt. Temporal ops (lag, ema, ...) and higher-order ops
    (fold, scan) raise NotImplementedError.

    Parameters
    ----------
    expr : Variable | Constant | Expr — the tree to compile
    var_names : list of str — names in the order features will be supplied
    """
    var_idx = {n: i for i, n in enumerate(var_names)}
    src = _compile_emit(expr, var_idx)
    func_src = f"lambda x: {src}"
    func = eval(func_src, {
        "_math": _math, "abs": abs,
        "_safe_div": _safe_div_scalar,
        "_sqrt_signed": _sqrt_signed_scalar,
    })
    func.__doc__ = f"compiled from: {expr.to_string() if hasattr(expr, 'to_string') else repr(expr)}"
    return func


__all__ = [
    "Variable", "Constant", "Operator", "HigherOrderOperator", "Expr", "REGISTRY",
    "add", "sub", "mul", "div",
    "neg", "sign", "abs_", "tanh", "sqrt_signed",
    "lag", "diff", "ema", "roll_mean", "roll_std",
    "ind_gt", "ind_abs_gt", "sign_change",
    "fold", "scan",
    "compile_expr",
]
