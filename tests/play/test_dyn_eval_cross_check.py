"""Python ↔ C++ cross-check for the dynamic eval.

For each fixture FEN, evaluate the strategy spec
  (a) on the Python side, by walking the expression_tree against a
      chess_feature_bank vector, and
  (b) on the C++ side, by invoking `math_engine.exe --strategy ... --eval-fen ...`.

Both must agree to within a small integer tolerance (rounding differences in
float→int truncation are accepted).

If this drifts in the future, mutation-driven symbolic-AZ would have a silent
Python/C++ divergence — every game played by the C++ engine would be against a
DIFFERENT strategy than the one the Python mutator thinks it produced. So we
test it.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import chess
import numpy as np
import pytest

from symbolic_chess.expression_layer.board import (
    chess_feature_bank, encode_position,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY = REPO_ROOT / "math-engine-cpp" / "build" / "math_engine.exe"
STRATEGIES = REPO_ROOT / "strategies"

FIXTURES = [
    ("startpos",   "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("k_v_k",      "k7/8/8/8/8/8/8/3K4 w - - 0 1"),
    ("kq_v_k",     "k7/8/8/8/8/8/8/QK6 w - - 0 1"),
    ("kiwipete",   "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("midgame_b",  "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4"),
    ("endgame",    "8/5k2/8/8/4P3/4K3/8/8 w - - 0 1"),
]

# Binary operator dispatch for evaluating an expression_tree in Python.
BIN = {
    "add": lambda a, b: a + b,
    "+":   lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "-":   lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "*":   lambda a, b: a * b,
    "div": lambda a, b: 0.0 if b == 0.0 else a / b,
    "/":   lambda a, b: 0.0 if b == 0.0 else a / b,
    "min": min,
    "max": max,
}
UN = {
    "tanh": math.tanh,
    "abs":  abs,
    "sign": lambda x: (1.0 if x > 0 else (-1.0 if x < 0 else 0.0)),
    "neg":  lambda x: -x,
}


def eval_tree_py(node: dict, features: dict[str, float]) -> float:
    if "var" in node:
        name = node["var"]
        if name not in features:
            raise KeyError(f"unknown feature {name}")
        return float(features[name])
    if "const" in node:
        return float(node["const"])
    if "op" in node:
        op = node["op"]
        args = node.get("args", [])
        if op in BIN:
            return BIN[op](eval_tree_py(args[0], features),
                           eval_tree_py(args[1], features))
        if op in UN:
            return UN[op](eval_tree_py(args[0], features))
        raise ValueError(f"unknown op {op}")
    raise ValueError(f"malformed node {node}")


def py_eval_fen(spec: dict, fen: str) -> int:
    """Mirror C++ eval_dyn::evaluate on the Python side."""
    planes = encode_position(fen)[None]
    bank = chess_feature_bank(planes, fens=[fen])
    features = {k: float(v[0]) for k, v in bank.items()}
    val = eval_tree_py(spec["expression_tree"], features)
    val *= spec.get("output_scale", 1.0)
    val = max(-30000.0, min(30000.0, val))
    return int(val)


def cpp_eval_fen(strategy_path: Path, fen: str) -> int:
    """Run math_engine.exe --strategy <path> --eval-fen <fen>, parse stdout."""
    out = subprocess.run(
        [str(BINARY), "--strategy", str(strategy_path), "--eval-fen", fen],
        capture_output=True, text=True, timeout=10,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"engine failed (code={out.returncode}): {out.stderr.strip()}"
        )
    # stdout's first line is the integer eval; stderr has "loaded strategy: ..."
    last = out.stdout.strip().splitlines()[-1]
    return int(last)


@pytest.mark.parametrize("spec_id", ["cx13_iter0", "iter1_joint"])
@pytest.mark.parametrize("fen_id,fen", FIXTURES)
def test_python_matches_cpp(spec_id: str, fen_id: str, fen: str):
    """Same expression_tree must yield the same int eval in Python and C++."""
    spec_path = STRATEGIES / f"{spec_id}.json"
    spec = json.loads(spec_path.read_text())
    py_val = py_eval_fen(spec, fen)
    cpp_val = cpp_eval_fen(spec_path, fen)
    # Allow ±1 cp drift for rounding (mobility uses pseudo vs legal — only iter0)
    assert abs(py_val - cpp_val) <= 1, (
        f"{spec_id} on {fen_id}: py={py_val} cpp={cpp_val}"
    )


def test_iter0_dyn_matches_hardcoded():
    """Dyn-eval iter0 must match the hardcoded cx13 eval at startpos."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    # Hardcoded path: no --strategy flag → routes to evaluate_sr_cx13
    out_hard = subprocess.run(
        [str(BINARY), "--eval-fen", fen],
        capture_output=True, text=True, timeout=10,
    )
    out_dyn = subprocess.run(
        [str(BINARY), "--strategy", str(STRATEGIES / "cx13_iter0.json"), "--eval-fen", fen],
        capture_output=True, text=True, timeout=10,
    )
    hard = int(out_hard.stdout.strip().splitlines()[-1])
    dyn = int(out_dyn.stdout.strip().splitlines()[-1])
    assert hard == dyn, f"hardcoded={hard} dyn={dyn}"
