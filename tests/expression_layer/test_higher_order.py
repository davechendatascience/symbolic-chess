"""Tests for higher-order operators (fold, scan) in expression_layer.

Design doc: docs/expression_layer_higher_order.md

Tests 1-3 from the doc:
    1. EMA equivalence — scan rediscovers core.ema
    2. Cumsum via scan
    3. Geometric series convergence to 1/(1-r)

Test 4 (PySR rediscovery) lives in benchmarks/, not here.
"""
import numpy as np
import pytest

from symbolic_chess.expression_layer import (
    Variable, Constant, Operator, HigherOrderOperator, Expr, REGISTRY,
    add, sub, mul, div, tanh,
    ema, fold, scan,
)


# ---------------- Structural sanity ----------------

def test_fold_and_scan_are_registered():
    assert "fold" in REGISTRY
    assert "scan" in REGISTRY
    assert isinstance(REGISTRY["fold"], HigherOrderOperator)
    assert isinstance(REGISTRY["scan"], HigherOrderOperator)


def test_fold_arity_is_three():
    assert fold.arity == 3
    assert scan.arity == 3


def test_fold_lazy_children_marks_g_lazy():
    # First child (the sub-expression g) is held lazily; x and init are eager.
    assert fold.lazy_children == (0,)
    assert scan.lazy_children == (0,)


def test_fold_arity_mismatch_raises():
    x = Variable("x", np.ones(5))
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(acc, xt)
    with pytest.raises(ValueError):
        fold(g, x)  # missing init


# ---------------- Test 1: EMA equivalence ----------------

def test_scan_rediscovers_ema():
    """scan with g = alpha*_x + (1-alpha)*_acc, init = x[0], matches core.ema."""
    rng = np.random.default_rng(42)
    x_data = rng.standard_normal(500)
    halflife = 10.0
    alpha = 1.0 - 0.5 ** (1.0 / halflife)

    x_var = Variable("x", x_data)
    ema_ref = ema(x_var, Constant(halflife)).evaluate()

    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(mul(Constant(alpha), xt), mul(Constant(1.0 - alpha), acc))
    out = scan(g, x_var, Constant(float(x_data[0]))).evaluate()

    assert np.allclose(out, ema_ref, atol=1e-10)


def test_scan_ema_different_halflives():
    """Sanity: different halflives give different EMAs, but the form holds."""
    rng = np.random.default_rng(7)
    x_data = rng.standard_normal(300)
    x_var = Variable("x", x_data)
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))

    for hl in [3.0, 20.0, 100.0]:
        alpha = 1.0 - 0.5 ** (1.0 / hl)
        g = add(mul(Constant(alpha), xt), mul(Constant(1.0 - alpha), acc))
        out = scan(g, x_var, Constant(float(x_data[0]))).evaluate()
        ref = ema(x_var, Constant(hl)).evaluate()
        assert np.allclose(out, ref, atol=1e-10), f"failed at halflife={hl}"


# ---------------- Test 2: Cumsum via scan ----------------

def test_scan_computes_cumsum():
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x_var = Variable("x", x_data)
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(acc, xt)
    out = scan(g, x_var, Constant(0.0)).evaluate()
    assert np.allclose(out, np.cumsum(x_data))


def test_scan_cumsum_with_nonzero_init():
    x_data = np.array([1.0, 1.0, 1.0])
    x_var = Variable("x", x_data)
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(acc, xt)
    out = scan(g, x_var, Constant(10.0)).evaluate()
    # 10 + [1,1,1] cumsum = [11, 12, 13]
    assert np.allclose(out, [11.0, 12.0, 13.0])


# ---------------- Test 3: Geometric series convergence ----------------

@pytest.mark.parametrize("r", [0.3, 0.5, 0.7, 0.9])
def test_fold_geometric_series_converges(r):
    """fold((acc, x) -> r*acc + x, ones(N), 0) -> 1/(1-r) as N grows."""
    N = 400
    x_data = np.ones(N)
    x_var = Variable("x", x_data)
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(mul(Constant(r), acc), xt)
    final = fold(g, x_var, Constant(0.0)).evaluate()
    expected = 1.0 / (1.0 - r)
    assert abs(final - expected) < 1e-3, f"r={r}: got {final}, expected {expected}"


def test_fold_returns_scalar_not_array():
    """fold emits final accumulator only."""
    x_var = Variable("x", np.array([1.0, 2.0, 3.0]))
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(acc, xt)
    result = fold(g, x_var, Constant(0.0)).evaluate()
    assert np.isscalar(result) or np.asarray(result).ndim == 0


# ---------------- Backward compat sanity ----------------

def test_existing_first_order_op_still_works():
    """Adding HigherOrderOperator must not break the first-order path."""
    x = Variable("x", np.array([1.0, 2.0, 3.0]))
    y = Variable("y", np.array([10.0, 20.0, 30.0]))
    expr = mul(add(x, y), Constant(0.5))
    assert np.allclose(expr.evaluate(), [5.5, 11.0, 16.5])


# ---------------- Composability ----------------

def test_scan_output_can_feed_first_order_op():
    """scan returns an array; downstream first-order ops should accept it."""
    x_data = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    x_var = Variable("x", x_data)
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(acc, xt)
    cumsum_expr = scan(g, x_var, Constant(0.0))
    # Multiply the running sum by 2
    doubled = mul(cumsum_expr, Constant(2.0))
    out = doubled.evaluate()
    assert np.allclose(out, [2.0, 4.0, 6.0, 8.0, 10.0])


def test_nested_scan_in_arithmetic_with_other_variable():
    """scan with env-bound _acc/_x; outer env binds another Variable correctly."""
    x_data = np.array([1.0, 2.0, 3.0])
    y_data = np.array([10.0, 20.0, 30.0])
    x_var = Variable("x", x_data)
    y_var = Variable("y", y_data)
    acc = Variable("_acc", np.zeros(1))
    xt = Variable("_x", np.zeros(1))
    g = add(acc, xt)
    csum = scan(g, x_var, Constant(0.0))
    expr = add(csum, y_var)
    out = expr.evaluate()
    # cumsum(x) + y = [1,3,6] + [10,20,30] = [11,23,36]
    assert np.allclose(out, [11.0, 23.0, 36.0])


# ---------------- Test 4: PySR rediscovery (slow, requires Julia/PySR) ----------------

@pytest.mark.slow
def test_pysr_rediscovers_ar1_recurrence():
    """PySR rediscovers the AR(1)/EMA per-step rule from i.i.d. samples.

    Marked `slow` — depends on PySR + Julia. The benchmark
    `benchmarks/run_fold_rediscover_ema.py` is the canonical artefact;
    this test is the pytest mirror so CI / `pytest -m slow` can gate on it.
    """
    pytest.importorskip("pysr")
    from symbolic_chess.expression_layer import fit_recurrence_step

    rng = np.random.default_rng(0)
    n = 3000
    halflife = 10.0
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    acc_t = rng.standard_normal(n)
    x_t = rng.standard_normal(n)
    y_next = alpha * x_t + (1.0 - alpha) * acc_t

    model, g_expr = fit_recurrence_step(
        acc_t, x_t, y_next,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=30, populations=15, population_size=30, maxsize=10,
        verbosity=0, procs=1, random_state=0,
    )
    assert g_expr is not None, "PySR returned no Expr"
    # Discovered g should evaluate to (approximately) the AR(1) form on the
    # training samples themselves.
    pred = np.array([
        float(g_expr.evaluate({"_acc": acc_t[i], "_x": x_t[i]}))
        for i in range(0, n, 10)   # subsample to keep this fast
    ])
    true = y_next[::10]
    err = np.max(np.abs(pred - true))
    assert err < 1e-3, f"PySR-discovered g deviates by {err}, expected < 1e-3"
