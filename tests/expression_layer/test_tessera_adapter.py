"""Smoke tests for the tessera adapter.

Verifies that:
1. tessera is installed and importable as a dependency
2. tessera_node_to_expr converts each supported tessera op correctly
3. run_tessera finds a low-loss expression on a synthetic problem
4. equation_table returns the expected schema
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip the whole module if tessera isn't installed (e.g., dev env without it)
tessera = pytest.importorskip("tessera")

from symbolic_chess.expression_layer import (
    Variable, Constant, Expr,
    add, sub, mul, div, neg, sign, abs_, tanh, sqrt_signed, ind_gt,
    run_tessera, tessera_node_to_expr, equation_table,
)


# ---------------- tessera_node_to_expr round-trip ----------------

def test_convert_var():
    """tessera Var → symbolic-chess Variable by name lookup."""
    from tessera.expression.tree import Var
    sc_var = Variable("x", np.array([1.0, 2.0, 3.0]))
    var_map = {"x": sc_var}
    out = tessera_node_to_expr(Var("x"), var_map)
    assert out is sc_var


def test_convert_const():
    from tessera.expression.tree import Const
    out = tessera_node_to_expr(Const(3.14), {})
    assert isinstance(out, Constant)
    assert abs(out.value - 3.14) < 1e-9


def test_convert_binop_add():
    """tessera BinOp('add', Var, Const) → symbolic-chess add(Variable, Constant)."""
    from tessera.expression.tree import Var, Const, BinOp
    sc_x = Variable("x", np.array([1.0, 2.0, 3.0]))
    tess_tree = BinOp("add", Var("x"), Const(1.0))
    sc_expr = tessera_node_to_expr(tess_tree, {"x": sc_x})
    assert isinstance(sc_expr, Expr)
    # Evaluate: should give x + 1 = [2, 3, 4]
    result = sc_expr.evaluate()
    np.testing.assert_allclose(result, [2.0, 3.0, 4.0])


def test_convert_unop_tanh():
    from tessera.expression.tree import Var, UnOp
    sc_x = Variable("x", np.array([0.0, 1.0, -1.0]))
    tess_tree = UnOp("tanh", Var("x"))
    sc_expr = tessera_node_to_expr(tess_tree, {"x": sc_x})
    result = sc_expr.evaluate()
    np.testing.assert_allclose(result, np.tanh([0.0, 1.0, -1.0]))


def test_convert_compound_tree():
    """tanh(a * b) + c — exercise nested conversion."""
    from tessera.expression.tree import Var, Const, BinOp, UnOp
    sc_a = Variable("a", np.array([1.0, 2.0]))
    sc_b = Variable("b", np.array([3.0, 4.0]))
    var_map = {"a": sc_a, "b": sc_b}

    tess_tree = BinOp("add",
                      UnOp("tanh", BinOp("mul", Var("a"), Var("b"))),
                      Const(0.5))
    sc_expr = tessera_node_to_expr(tess_tree, var_map)
    result = sc_expr.evaluate()
    expected = np.tanh([3.0, 8.0]) + 0.5
    np.testing.assert_allclose(result, expected)


def test_unsupported_op_raises():
    """tessera ops without a symbolic-chess equivalent should raise."""
    from tessera.expression.tree import Var, BinOp
    sc_x = Variable("x", np.array([1.0, 2.0]))
    bad = BinOp("min", Var("x"), Var("x"))   # min not in _BIN_TESSERA_TO_SC
    with pytest.raises(ValueError, match="min"):
        tessera_node_to_expr(bad, {"x": sc_x})


# ---------------- run_tessera end-to-end ----------------

def test_run_tessera_finds_signal_on_simple_target():
    """Tiny SR problem: target = x + 2*y. tessera should find this exactly
    or very close in a small GP run."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    target = x + 2.0 * y

    x_var = Variable("x", x)
    y_var = Variable("y", y)

    gp, best_expr = run_tessera(
        [x_var, y_var], target,
        niterations=15, population_size=40,
        random_state=42, verbose=False,
    )

    assert best_expr is not None, "tessera should have found a low-loss expression"
    # Evaluate the discovered expression and check it's close to target
    pred = best_expr.evaluate()
    # MSE should be low — under variance of the residual, ideally near 0
    mse = np.mean((pred - target) ** 2)
    target_var = np.var(target)
    assert mse < target_var * 0.5, (
        f"discovered MSE {mse:.4g} is not significantly below target variance "
        f"{target_var:.4g} — GP didn't find the linear relationship"
    )


def test_run_tessera_returns_gp_with_pareto_front():
    """Smoke check: gp.hall_of_fame.pareto_front() exists and contains
    Candidates."""
    rng = np.random.default_rng(0)
    n = 50
    x = rng.standard_normal(n)
    target = 2.0 * x + 0.5

    x_var = Variable("x", x)
    gp, _ = run_tessera([x_var], target,
                        niterations=5, population_size=20,
                        random_state=0, verbose=False)
    front = gp.hall_of_fame.pareto_front()
    assert len(front) >= 1
    for cand in front:
        assert hasattr(cand, "tree")
        assert hasattr(cand, "complexity")
        assert hasattr(cand, "train_loss")


def test_equation_table_schema():
    """equation_table(gp) returns list of dicts with expected keys."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    target = x * 2.0

    gp, _ = run_tessera([Variable("x", x)], target,
                        niterations=5, population_size=20,
                        random_state=0, verbose=False)
    table = equation_table(gp)
    assert len(table) >= 1
    for row in table:
        assert "complexity" in row
        assert "train_loss" in row
        assert "fitness" in row
        assert "tree_str" in row


# ---------------- Op-restriction validation ----------------

def test_unsupported_binary_op_rejected_up_front():
    """Requesting an unsupported binary op raises before GP starts."""
    x = Variable("x", np.arange(10.0))
    with pytest.raises(ValueError, match="not supported"):
        run_tessera([x], np.arange(10.0),
                    binary_operators=["+", "min"],   # 'min' not in adapter
                    niterations=1, population_size=10, verbose=False)


def test_pysr_symbol_aliases_work():
    """PySR-style symbols ('+', '-', '*', '/') should be accepted and
    mapped to tessera names (add, sub, mul, div)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(50)
    target = x + 0.5
    gp, best = run_tessera(
        [Variable("x", x)], target,
        binary_operators=["+", "-", "*"],
        unary_operators=[],
        niterations=3, population_size=10,
        random_state=0, verbose=False,
    )
    # Just verify no error — schema correctness covered above
    assert best is not None or gp.hall_of_fame.pareto_front()
