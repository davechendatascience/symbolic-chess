"""Tests for the expression layer.

These tests are pure-numpy and don't require any benchmark data.
"""
import numpy as np
import pytest

from symbolic_chess.expression_layer import (
    Variable, Constant, Operator, Expr, REGISTRY,
    add, sub, mul, div,
    neg, sign, abs_, tanh, sqrt_signed,
    lag, diff, ema, roll_mean, roll_std,
    ind_gt, ind_abs_gt, sign_change,
)


# ----------- Leaves -----------

def test_variable_evaluates_to_data():
    x = Variable("x", np.array([1.0, 2.0, 3.0]))
    assert np.allclose(x.evaluate(), [1.0, 2.0, 3.0])


def test_variable_env_overrides_data():
    x = Variable("x", np.array([1.0, 2.0, 3.0]))
    env = {"x": np.array([10.0, 20.0, 30.0])}
    assert np.allclose(x.evaluate(env), [10.0, 20.0, 30.0])


def test_constant_evaluates_to_scalar():
    c = Constant(5.0)
    assert c.evaluate() == 5.0


# ----------- Arithmetic -----------

def test_add_two_variables():
    x = Variable("x", np.array([1.0, 2.0, 3.0]))
    y = Variable("y", np.array([10.0, 20.0, 30.0]))
    expr = add(x, y)
    assert np.allclose(expr.evaluate(), [11.0, 22.0, 33.0])


def test_sub_with_constant():
    x = Variable("x", np.array([5.0, 10.0]))
    expr = sub(x, Constant(2.0))
    assert np.allclose(expr.evaluate(), [3.0, 8.0])


def test_mul_and_div():
    x = Variable("x", np.array([4.0, 9.0]))
    expr = div(x, Constant(2.0))
    assert np.allclose(expr.evaluate(), [2.0, 4.5])


def test_div_by_zero_returns_zero():
    x = Variable("x", np.array([1.0, 2.0, 3.0]))
    y = Variable("y", np.array([2.0, 0.0, 4.0]))
    expr = div(x, y)
    out = expr.evaluate()
    assert np.allclose(out, [0.5, 0.0, 0.75])


# ----------- Pointwise nonlinear -----------

def test_neg_sign_abs_tanh():
    x = Variable("x", np.array([-1.0, 0.0, 2.0]))
    assert np.allclose(neg(x).evaluate(), [1.0, 0.0, -2.0])
    assert np.allclose(sign(x).evaluate(), [-1.0, 0.0, 1.0])
    assert np.allclose(abs_(x).evaluate(), [1.0, 0.0, 2.0])
    assert np.allclose(tanh(x).evaluate(), np.tanh([-1.0, 0.0, 2.0]))


def test_sqrt_signed():
    x = Variable("x", np.array([-4.0, 0.0, 9.0]))
    out = sqrt_signed(x).evaluate()
    assert np.allclose(out, [-2.0, 0.0, 3.0])


# ----------- Temporal -----------

def test_lag():
    x = Variable("x", np.array([1.0, 2.0, 3.0, 4.0]))
    expr = lag(x, Constant(1))
    assert np.allclose(expr.evaluate(), [0.0, 1.0, 2.0, 3.0])


def test_lag_zero_is_identity():
    x = Variable("x", np.array([1.0, 2.0, 3.0]))
    assert np.allclose(lag(x, Constant(0)).evaluate(), [1.0, 2.0, 3.0])


def test_diff_one():
    x = Variable("x", np.array([1.0, 2.0, 4.0, 7.0]))
    expr = diff(x, Constant(1))
    assert np.allclose(expr.evaluate(), [1.0, 1.0, 2.0, 3.0])


def test_ema_constant_converges():
    x = Variable("x", np.ones(200))
    expr = ema(x, Constant(10))
    out = expr.evaluate()
    assert np.isclose(out[-1], 1.0, atol=1e-6)


def test_rolling_mean():
    x = Variable("x", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    expr = roll_mean(x, Constant(3))
    out = expr.evaluate()
    expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert np.allclose(out[2:], expected[2:])


def test_rolling_std():
    x = Variable("x", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    expr = roll_std(x, Constant(3))
    out = expr.evaluate()
    # std of [1,2,3]=0.8165, of [2,3,4]=0.8165, of [3,4,5]=0.8165
    assert np.isclose(out[2], 0.8164965, atol=1e-5)


# ----------- Crossing / regime -----------

def test_ind_gt():
    x = Variable("x", np.array([-0.5, 0.0, 0.5, 1.5]))
    expr = ind_gt(x, Constant(0.4))
    assert np.allclose(expr.evaluate(), [0, 0, 1, 1])


def test_ind_abs_gt():
    x = Variable("x", np.array([-0.5, 0.0, 0.5, 1.5]))
    expr = ind_abs_gt(x, Constant(0.6))
    assert np.allclose(expr.evaluate(), [0, 0, 0, 1])


def test_sign_change():
    x = Variable("x", np.array([1.0, 1.0, -1.0, -1.0, 1.0]))
    out = sign_change(x).evaluate()
    # prepend=s[0]=1 → diff = [0,0,-2,0,2] → !=0 at indices 2,4
    assert np.allclose(out, [0, 0, 1, 0, 1])


# ----------- Composite trees -----------

def test_complex_tree():
    x = Variable("x", np.array([1.0, 2.0, 3.0]))
    y = Variable("y", np.array([0.5, 1.0, 1.5]))
    expr = mul(add(x, Constant(1.0)), tanh(y))
    expected = (np.array([1.0, 2.0, 3.0]) + 1.0) * np.tanh([0.5, 1.0, 1.5])
    assert np.allclose(expr.evaluate(), expected)


def test_complexity():
    x = Variable("x", np.array([1.0]))
    y = Variable("y", np.array([2.0]))
    assert x.complexity() == 1
    assert add(x, y).complexity() == 3
    assert mul(add(x, y), x).complexity() == 5
    assert mul(add(x, y), tanh(x)).complexity() == 6


def test_leaves():
    x = Variable("x", np.array([1.0]))
    y = Variable("y", np.array([2.0]))
    expr = mul(add(x, y), x)
    assert expr.leaves() == {"x", "y"}
    # Constants don't count as leaves
    expr2 = add(x, Constant(3.0))
    assert expr2.leaves() == {"x"}


# ----------- Registry -----------

def test_registry_has_expected_operators():
    expected = [
        "add", "sub", "mul", "div",
        "neg", "sign", "abs", "tanh", "sqrt_signed",
        "lag", "diff", "ema", "roll_mean", "roll_std",
        "ind_gt", "ind_abs_gt", "sign_change",
    ]
    for name in expected:
        assert name in REGISTRY, f"missing operator: {name}"


def test_registry_lookup_then_call():
    x = Variable("x", np.array([1.0, 2.0, 3.0]))
    add_op = REGISTRY["add"]
    expr = add_op(x, Constant(10.0))
    assert np.allclose(expr.evaluate(), [11.0, 12.0, 13.0])


# ----------- to_string / repr -----------

def test_to_string_infix():
    x = Variable("x", np.array([1.0]))
    y = Variable("y", np.array([2.0]))
    expr = add(x, y)
    s = expr.to_string()
    assert "+" in s and "x" in s and "y" in s


def test_to_string_prefix():
    x = Variable("x", np.array([1.0]))
    expr = tanh(lag(x, Constant(3)))
    s = expr.to_string()
    assert "tanh" in s and "lag" in s and "x" in s


# ----------- Operator arity validation -----------

def test_arity_mismatch_raises():
    x = Variable("x", np.array([1.0]))
    with pytest.raises(ValueError):
        add(x)   # add needs 2 args
    with pytest.raises(ValueError):
        tanh(x, Constant(0.5))   # tanh needs 1 arg


# ----------- Sanity: reproduce RCBS drive as an Expr tree -----------

def test_rcbs_drive_as_expr_tree():
    """Reproduce the RCBS feedback drive (non-recurrent part) as a tree."""
    n = 1000
    rng = np.random.default_rng(0)
    sigma_w24 = rng.standard_normal(n)
    sigma_w168 = rng.standard_normal(n)
    mu3_w168 = 0.5 * rng.standard_normal(n)

    sw24 = Variable("sigma_w24", sigma_w24)
    sw168 = Variable("sigma_w168", sigma_w168)
    mu3 = Variable("mu3_w168", mu3_w168)
    beta = Constant(0.5); gamma = Constant(1.5); delta = Constant(0.3); tau = Constant(0.1)

    # drive = β·(-σ_w24) + γ·(-σ_w168)·I(|σ_w168|>τ) + δ·μ̃3_w168·sign(σ_w24)
    drive = add(
        add(
            mul(beta, neg(sw24)),
            mul(gamma, mul(neg(sw168), ind_abs_gt(sw168, tau))),
        ),
        mul(delta, mul(mu3, sign(sw24))),
    )
    out = drive.evaluate()
    expected = (
        0.5 * (-sigma_w24)
        + 1.5 * (-sigma_w168) * (np.abs(sigma_w168) > 0.1).astype(float)
        + 0.3 * mu3_w168 * np.sign(sigma_w24)
    )
    assert np.allclose(out, expected)
    # Sanity on metadata
    assert drive.leaves() == {"sigma_w24", "sigma_w168", "mu3_w168"}
    assert drive.complexity() > 10
