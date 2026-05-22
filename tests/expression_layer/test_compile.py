"""Tests for compile_expr — fast per-row scalar compilation.

compile_expr walks an Expr tree once and emits a Python lambda for ~50-100x
faster per-row evaluation than recursive Expr.evaluate(). Used by the chess
engine to keep alpha-beta search tractable.
"""
import numpy as np
import pytest

from symbolic_chess.expression_layer import (
    Variable, Constant,
    add, sub, mul, div, neg, abs_, sign, tanh, sqrt_signed,
    ind_gt, ind_abs_gt,
    ema, fold, scan,
    compile_expr,
)


# ---------------- Numerical equivalence with Expr.evaluate ----------------

def test_compile_simple_addition():
    x = Variable("x", np.array([0.0]))
    y = Variable("y", np.array([0.0]))
    expr = add(x, y)
    f = compile_expr(expr, ["x", "y"])
    assert f(np.array([1.0, 2.0])) == 3.0
    assert f(np.array([10.0, -5.0])) == 5.0


def test_compile_arithmetic_compound():
    x = Variable("x", np.array([0.0]))
    y = Variable("y", np.array([0.0]))
    # (x + y) * (x - y) = x^2 - y^2
    expr = mul(add(x, y), sub(x, y))
    f = compile_expr(expr, ["x", "y"])
    for xv, yv in [(2.0, 3.0), (-1.0, 4.0), (0.5, 0.5)]:
        assert f(np.array([xv, yv])) == pytest.approx(xv * xv - yv * yv)


def test_compile_with_constants():
    x = Variable("x", np.array([0.0]))
    # 2*x + 3
    expr = add(mul(Constant(2.0), x), Constant(3.0))
    f = compile_expr(expr, ["x"])
    assert f(np.array([5.0])) == 13.0
    assert f(np.array([-1.0])) == 1.0


def test_compile_div_safe_on_zero():
    x = Variable("x", np.array([0.0]))
    y = Variable("y", np.array([0.0]))
    expr = div(x, y)
    f = compile_expr(expr, ["x", "y"])
    # Normal division
    assert f(np.array([10.0, 2.0])) == 5.0
    # Safe div: y=0 returns 0
    assert f(np.array([10.0, 0.0])) == 0.0


def test_compile_unary_ops():
    x = Variable("x", np.array([0.0]))
    f_neg = compile_expr(neg(x), ["x"])
    f_abs = compile_expr(abs_(x), ["x"])
    f_sign = compile_expr(sign(x), ["x"])
    f_tanh = compile_expr(tanh(x), ["x"])
    f_sqrt = compile_expr(sqrt_signed(x), ["x"])
    for v in [-3.0, 0.0, 2.5]:
        arr = np.array([v])
        assert f_neg(arr) == -v
        assert f_abs(arr) == abs(v)
        assert f_sign(arr) == (1.0 if v > 0 else (-1.0 if v < 0 else 0.0))
        assert f_tanh(arr) == pytest.approx(np.tanh(v))
        expected_sqrt = np.sign(v) * np.sqrt(abs(v))
        assert f_sqrt(arr) == pytest.approx(expected_sqrt)


def test_compile_indicator_ops():
    x = Variable("x", np.array([0.0]))
    f_gt = compile_expr(ind_gt(x, Constant(1.0)), ["x"])
    f_absgt = compile_expr(ind_abs_gt(x, Constant(2.0)), ["x"])
    assert f_gt(np.array([0.5])) == 0.0
    assert f_gt(np.array([1.5])) == 1.0
    assert f_absgt(np.array([-3.0])) == 1.0
    assert f_absgt(np.array([1.0])) == 0.0


def test_compile_matches_evaluate_on_random_tree():
    """Compiled output matches Expr.evaluate result on the same inputs."""
    rng = np.random.default_rng(42)
    n = 50
    x_data = rng.standard_normal(n)
    y_data = rng.standard_normal(n)
    x = Variable("x", x_data)
    y = Variable("y", y_data)
    # A non-trivial expression: tanh((x*y) + abs(x - 0.3)) - sign(y)
    expr = sub(tanh(add(mul(x, y), abs_(sub(x, Constant(0.3))))), sign(y))
    expected = expr.evaluate()

    f = compile_expr(expr, ["x", "y"])
    actual = np.array([f(np.array([x_data[i], y_data[i]])) for i in range(n)])
    assert np.allclose(actual, expected, atol=1e-10)


# ---------------- Error cases ----------------

def test_compile_raises_on_unknown_variable():
    x = Variable("x", np.array([0.0]))
    expr = add(x, Variable("z", np.array([0.0])))
    with pytest.raises(ValueError, match="not in var_names"):
        compile_expr(expr, ["x"])   # z missing


def test_compile_raises_on_temporal_op():
    x = Variable("x", np.array([0.0]))
    expr = ema(x, Constant(10))
    with pytest.raises(NotImplementedError, match="does not support op 'ema'"):
        compile_expr(expr, ["x"])


def test_compile_raises_on_higher_order_op():
    x = Variable("x", np.array([0.0]))
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(acc, xt)
    expr = scan(g, x, Constant(0.0))
    with pytest.raises(NotImplementedError, match="higher-order"):
        compile_expr(expr, ["x"])


# ---------------- Performance / signature ----------------

def test_compile_returns_callable():
    x = Variable("x", np.array([0.0]))
    f = compile_expr(add(x, Constant(1.0)), ["x"])
    assert callable(f)
    assert f.__doc__ is not None
    assert "compiled from" in f.__doc__


def test_compile_speed_advantage():
    """Compiled callable should be substantially faster than recursive evaluate.

    This is a regression test — if eval gets >10x slower vs compile, something
    is wrong with the compilation. Not strict (timing can vary), just sanity.
    """
    import time
    x = Variable("x", np.array([0.0]))
    y = Variable("y", np.array([0.0]))
    # 7-deep nested expression
    expr = tanh(add(mul(x, y), sub(abs_(x), mul(y, Constant(0.5)))))
    f = compile_expr(expr, ["x", "y"])

    arr = np.array([1.5, -0.7])
    env = {"x": 1.5, "y": -0.7}

    # Warm up
    for _ in range(100):
        f(arr); expr.evaluate(env)

    t0 = time.perf_counter()
    for _ in range(10000):
        f(arr)
    t_compiled = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(10000):
        expr.evaluate(env)
    t_eval = time.perf_counter() - t0

    print(f"\n  compiled: {t_compiled*1e6/10000:.2f} us/call")
    print(f"  evaluate: {t_eval*1e6/10000:.2f} us/call")
    print(f"  speedup:  {t_eval/t_compiled:.1f}x")
    # Should be at least 5x faster — well below the 50-100x target
    # to leave room for slow CI runners
    assert t_compiled < t_eval / 5, (
        f"compiled ({t_compiled:.4f}s) not >5x faster than evaluate ({t_eval:.4f}s)"
    )
