from pathlib import Path

import pytest
import chess

from symbolic_chess.play.uci_bridge import UciEngine


REPO = Path(__file__).resolve().parents[2]
BINARY_CANDIDATES = [
    REPO / "math-engine-cpp" / "build" / "math_engine.exe",
    REPO / "math-engine-cpp" / "build" / "Release" / "math_engine.exe",
    REPO / "math-engine-cpp" / "build" / "math_engine",
]


def _find_binary() -> Path | None:
    for p in BINARY_CANDIDATES:
        if p.is_file():
            return p
    return None


@pytest.fixture(scope="module")
def binary():
    p = _find_binary()
    if p is None:
        pytest.skip("math-engine-cpp not built")
    return p


def test_startpos_bestmove_is_legal(binary):
    with UciEngine(binary, default_depth=2) as eng:
        eng.new_game()
        mv = eng.bestmove("startpos", depth=2)
        b = chess.Board()
        assert chess.Move.from_uci(mv) in b.legal_moves


def test_followup_after_e4_e5(binary):
    with UciEngine(binary, default_depth=2) as eng:
        eng.new_game()
        mv = eng.bestmove("startpos", moves=["e2e4", "e7e5"], depth=2)
        b = chess.Board()
        b.push_uci("e2e4")
        b.push_uci("e7e5")
        assert chess.Move.from_uci(mv) in b.legal_moves
