"""expression_layer — uniform tree representation for symbolic-regression /
genetic-programming search over time-series functions.

Core abstractions (all in `core`):
    Variable     — named time-series leaf
    Constant     — scalar leaf
    Operator     — base operation with arity + callable + display symbol
    Expr         — tree node: Operator + children

The REGISTRY dict contains named instances of the base operators (arithmetic,
pointwise nonlinear, temporal, crossing/regime). Search tools (PySR adapter,
genetic programming, exhaustive enumeration) can pull from REGISTRY without
caring about implementation details.

Design property: zero domain-specific infrastructure dependencies. Pure numpy
+ python-chess for board features.

Usage:
    from symbolic_chess.expression_layer import Variable, Constant, add, mul, tanh, ema

    x = Variable("x", np.linspace(0, 1, 100))
    expr = mul(tanh(x), ema(x, Constant(10)))
    out = expr.evaluate()        # → np.ndarray of length 100
    print(expr.to_string())      # → "(tanh(x) * ema(x, 10))"
    print(expr.complexity())     # → 6
    print(expr.leaves())         # → {"x"}
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
# PySR adapter — deferred Julia/pysr imports, safe to import here
from .pysr_adapter import (
    expand_temporal_bank, run_pysr, fit_recurrence_step,
    sympy_to_expr, equation_table,
)

__all__ = [
    "Variable", "Constant", "Operator", "HigherOrderOperator", "Expr", "REGISTRY",
    "add", "sub", "mul", "div",
    "neg", "sign", "abs_", "tanh", "sqrt_signed",
    "lag", "diff", "ema", "roll_mean", "roll_std",
    "ind_gt", "ind_abs_gt", "sign_change",
    "fold", "scan",
    "compile_expr",
    "expand_temporal_bank", "run_pysr", "fit_recurrence_step",
    "sympy_to_expr", "equation_table",
]
