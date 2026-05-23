"""Mutation operator tests. Each operator must:
  - Produce a valid tree (passes validate_tree) or signal failure
  - Be deterministic given a seeded RNG
  - Never reference unknown features
"""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from symbolic_chess.strategy.tree import (
    BinOp, Const, Node, StrategyTreeSpec, UnOp, Var,
    complexity, depth, iter_subtrees, used_features,
)
from symbolic_chess.symbolic_az.mutation import (
    MAX_COMPLEXITY, MAX_DEPTH,
    constant_jitter, mutate, op_swap, random_subtree,
    subtree_crossover, subtree_swap, term_delete, term_insert,
    validate_tree,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# A small but realistic feature catalog
FEATURES = [
    "WP_count", "WN_count", "WB_count", "WR_count", "WQ_count",
    "BP_count", "BN_count", "BB_count", "BR_count", "BQ_count",
    "material_net", "W_mobility", "B_mobility",
    "central_WP", "central_BP", "central_BN", "central_BB",
    "B_king_zone_own_pawns",
    "phase",
]


def _load_iter1() -> Node:
    return StrategyTreeSpec.load(REPO_ROOT / "strategies" / "iter1_joint.json").tree


def test_validate_accepts_iter1():
    t = _load_iter1()
    assert validate_tree(t, set(FEATURES)) is None


def test_validate_rejects_overdepth():
    t = Var("WP_count")
    for _ in range(MAX_DEPTH):
        t = BinOp("add", t, Const(1.0))
    err = validate_tree(t, set(FEATURES))
    assert err is not None and "depth" in err


def test_random_subtree_is_valid():
    rng = random.Random(42)
    for _ in range(50):
        t = random_subtree(rng, FEATURES, max_depth=3)
        assert validate_tree(t, set(FEATURES)) is None


def test_subtree_swap_changes_tree():
    rng = random.Random(0)
    parent = _load_iter1()
    child = subtree_swap(parent, rng, FEATURES)
    assert child != parent
    # Sometimes the new subtree might happen to equal what was replaced;
    # but with a fresh rng and a non-trivial parent this should be reliable.


def test_subtree_swap_validates():
    rng = random.Random(7)
    parent = _load_iter1()
    for _ in range(50):
        child = subtree_swap(parent, rng, FEATURES)
        # Allow occasional over-budget — caller is expected to validate
        # but feature names must always be ones we declared
        unknown = used_features(child) - set(FEATURES)
        assert not unknown


def test_subtree_crossover_uses_both_parents():
    rng = random.Random(1)
    a = _load_iter1()
    b = BinOp("mul", Const(2.0), Var("phase"))
    # Cross many times; at some point B's "phase" should appear in offspring
    found = False
    for _ in range(30):
        child = subtree_crossover(a, b, rng)
        if "phase" in used_features(child):
            found = True
            break
    assert found, "crossover never pulled material from parent B"


def test_constant_jitter_keeps_structure():
    rng = random.Random(3)
    parent = _load_iter1()
    child = constant_jitter(parent, rng)
    # Same number of nodes, same op structure — only constant values change
    assert complexity(parent) == complexity(child)
    p_consts = [n.value for n in iter_subtrees(parent) if isinstance(n, Const)]
    c_consts = [n.value for n in iter_subtrees(child)  if isinstance(n, Const)]
    assert len(p_consts) == len(c_consts)
    # At least one should have changed
    assert any(abs(p - c) > 1e-10 for p, c in zip(p_consts, c_consts))


def test_term_insert_increases_complexity():
    rng = random.Random(5)
    parent = _load_iter1()
    child = term_insert(parent, rng, FEATURES)
    assert complexity(child) > complexity(parent)


def test_term_delete_does_not_increase_complexity():
    rng = random.Random(9)
    parent = BinOp("add", Var("WP_count"), BinOp("sub", Var("WN_count"), Const(1.0)))
    for _ in range(20):
        child = term_delete(parent, rng)
        assert complexity(child) <= complexity(parent)


def test_op_swap_preserves_complexity():
    rng = random.Random(11)
    parent = _load_iter1()
    child = op_swap(parent, rng)
    assert complexity(child) == complexity(parent)


def test_mutate_returns_valid_offspring():
    rng = random.Random(2026)
    a = _load_iter1()
    b = StrategyTreeSpec.load(REPO_ROOT / "strategies" / "cx13_iter0.json").tree
    n_valid = 0
    n_attempts = 200
    for _ in range(n_attempts):
        child = mutate([a, b], rng, FEATURES)
        if child is not None:
            assert validate_tree(child, set(FEATURES)) is None
            n_valid += 1
    # Should succeed comfortably more often than not
    assert n_valid > n_attempts * 0.7, f"only {n_valid}/{n_attempts} valid"


def test_mutate_deterministic_given_seed():
    a = _load_iter1()
    rng1 = random.Random(123)
    rng2 = random.Random(123)
    c1 = mutate([a], rng1, FEATURES)
    c2 = mutate([a], rng2, FEATURES)
    assert c1 == c2
