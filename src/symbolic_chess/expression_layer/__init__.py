"""expression_layer — uniform tree representation for symbolic-regression /
genetic-programming search over time-series functions.

Core abstractions (all in `core`):
    Variable     — named time-series leaf
    Constant     — scalar leaf
    Operator     — base operation with arity + callable + display symbol
    Expr         — tree node: Operator + children

The REGISTRY dict contains named instances of the base operators
(arithmetic, pointwise nonlinear, temporal, crossing/regime). Search
tools (tessera adapter, genetic programming, exhaustive enumeration)
can pull from REGISTRY without caring about implementation details.

Design property: zero domain-specific infrastructure dependencies. Pure
numpy + python-chess for board features.

Usage:
    from symbolic_chess.expression_layer import Variable, Constant, add, mul, tanh, ema

    x = Variable("x", np.linspace(0, 1, 100))
    expr = mul(tanh(x), ema(x, Constant(10)))
    out = expr.evaluate()        # → np.ndarray of length 100
    print(expr.to_string())      # → "(tanh(x) * ema(x, 10))"
    print(expr.complexity())     # → 6
    print(expr.leaves())         # → {"x"}

SR backend (2026-05-24):
    The symbolic-regression search is performed by `tessera` (pure-Python
    SR library, pinned in pyproject.toml). Use `run_tessera(...)` for
    the search; downstream code uses `equation_table(gp)` and
    `predict_with_tree(tree, X, feature_names)` for the Pareto-front
    inspection + test-set prediction.

    The PySR/Julia backend that previously lived in `pysr_adapter.py`
    was removed in this same date; see git history for the rationale.
"""
from .core import (
    # leaves
    Variable, Constant,
    # base classes
    Operator, HigherOrderOperator, Expr,
    # registry
    REGISTRY,
    # arithmetic (binary, infix)
    add, sub, mul, div,
    # pointwise nonlinear (unary)
    neg, sign, abs_, tanh, sqrt_signed,
    # temporal (binary: input, parameter)
    lag, diff, ema, roll_mean, roll_std,
    # crossing / regime
    ind_gt, ind_abs_gt, sign_change,
    # higher-order
    fold, scan,
    # compilation
    compile_expr,
)
# Temporal-feature pre-computation + sympy parsing (utilities, not
# SR-backend-specific). Was previously inside pysr_adapter.py;
# extracted on 2026-05-24 when PySR was removed.
from .temporal_utils import (
    expand_temporal_bank,
    sympy_to_expr,
)
# Tessera adapter — pure-Python SR backend (replaces the PySR/Julia
# backend that was removed 2026-05-24).
from .tessera_adapter import (
    run_tessera, tessera_node_to_expr, predict_with_tree,
    equation_table,
)

__all__ = [
    "Variable", "Constant", "Operator", "HigherOrderOperator", "Expr", "REGISTRY",
    "add", "sub", "mul", "div",
    "neg", "sign", "abs_", "tanh", "sqrt_signed",
    "lag", "diff", "ema", "roll_mean", "roll_std",
    "ind_gt", "ind_abs_gt", "sign_change",
    "fold", "scan",
    "compile_expr",
    "expand_temporal_bank", "sympy_to_expr",
    "run_tessera", "tessera_node_to_expr", "predict_with_tree",
    "equation_table",
]
