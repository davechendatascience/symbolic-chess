"""Strategy tree dataclass tests — serialization, structure, evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from symbolic_chess.strategy.tree import (
    BinOp, Const, StrategyTreeSpec, UnOp, Var,
    complexity, depth, evaluate, from_dict, iter_subtrees, pretty,
    to_dict, used_features,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_roundtrip_iter0():
    """Round-trip the on-disk cx13_iter0 spec without losing info."""
    j = json.loads((REPO_ROOT / "strategies" / "cx13_iter0.json").read_text())
    tree = from_dict(j["expression_tree"])
    j2 = to_dict(tree)
    # JSON structure equal (ignoring float repr drift)
    tree2 = from_dict(j2)
    assert tree == tree2

def test_roundtrip_iter1():
    j = json.loads((REPO_ROOT / "strategies" / "iter1_joint.json").read_text())
    tree = from_dict(j["expression_tree"])
    assert tree == from_dict(to_dict(tree))

def test_basic_construction():
    t = BinOp("mul", Const(0.5), Var("material_net"))
    assert complexity(t) == 3
    assert depth(t) == 2
    assert used_features(t) == {"material_net"}

def test_iter_subtrees_visits_all():
    t = BinOp("sub", BinOp("add", Var("x"), Const(1.0)), UnOp("tanh", Var("y")))
    nodes = list(iter_subtrees(t))
    assert len(nodes) == 6   # root + 2 binop + 1 const + 1 unop + 1 var (the last var)

def test_evaluate_matches_iter1_on_startpos():
    """iter1: (material_net + (WB_count - central_BP - central_BN)) * 0.17522278.
    At startpos: 0 + (2 - 0 - 0) = 2; * 0.17522 = 0.35044."""
    spec = StrategyTreeSpec.load(REPO_ROOT / "strategies" / "iter1_joint.json")
    features = dict(material_net=0, WB_count=2, central_BP=0, central_BN=0)
    val = evaluate(spec.tree, features)
    assert abs(val - 0.35044556) < 1e-5, f"got {val}"

def test_evaluate_iter0_kq_v_k():
    """iter0 on K+Q vs K: material_net=9, other features near zero.
    term1 = 9/0.0146 = 616.4; term2 small. eval ≈ 616."""
    spec = StrategyTreeSpec.load(REPO_ROOT / "strategies" / "cx13_iter0.json")
    features = dict(
        material_net=9.0, W_mobility=2.0, central_BP=0.0,
        B_king_zone_own_pawns=0.0, central_BB=0.0,
    )
    val = evaluate(spec.tree, features)
    assert 615 < val < 618, f"got {val}"

def test_safe_div():
    t = BinOp("div", Const(5.0), Const(0.0))
    assert evaluate(t, {}) == 0.0

def test_spec_save_load_preserves_meta(tmp_path):
    spec = StrategyTreeSpec(
        id="test_meta",
        tree=BinOp("add", Var("WP_count"), Const(1.0)),
        output_scale=2.5,
        meta={"provenance": {"iter": 99, "method": "manual"}},
    )
    p = tmp_path / "test.json"
    spec.save(p)
    loaded = StrategyTreeSpec.load(p)
    assert loaded.id == "test_meta"
    assert loaded.tree == spec.tree
    assert loaded.output_scale == 2.5
    assert loaded.meta["provenance"]["iter"] == 99

def test_pretty_prints_iter1():
    spec = StrategyTreeSpec.load(REPO_ROOT / "strategies" / "iter1_joint.json")
    s = pretty(spec.tree)
    assert "material_net" in s
    assert "WB_count" in s
    assert "0.17522" in s
