"""Tests for chess_engine.self_play — outcome decay + corpus utilities.

The corpus generator itself depends on Stockfish; smoke-test separately.
"""
import numpy as np
import pytest
import chess

from symbolic_chess.chess_engine.self_play import decay_outcome, _outcome_from_board


# ---------------- decay_outcome ----------------

def test_decay_terminal_position_keeps_signal():
    assert decay_outcome(1, 0) == 1.0
    assert decay_outcome(-1, 0) == -1.0
    assert decay_outcome(0, 0) == 0.0
    assert decay_outcome(0, 50) == 0.0


def test_decay_attenuates_with_distance():
    # default decay=0.98
    assert decay_outcome(1, 1) == pytest.approx(0.98)
    assert decay_outcome(1, 10) == pytest.approx(0.98 ** 10)
    assert decay_outcome(-1, 50) == pytest.approx(-(0.98 ** 50))


def test_decay_custom_rate():
    assert decay_outcome(1, 5, decay=0.9) == pytest.approx(0.9 ** 5)
    assert decay_outcome(1, 5, decay=1.0) == 1.0   # no decay


# ---------------- _outcome_from_board ----------------

def test_outcome_white_win():
    # Fool's mate — black mates white; outcome = -1 from white's perspective
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    board.push_san("Qh4#")
    assert board.is_checkmate()
    assert _outcome_from_board(board) == -1


def test_outcome_stalemate_is_zero():
    # Stalemate: white K c6, white Q c7, black K a8, black to move
    board = chess.Board("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    assert _outcome_from_board(board) == 0


def test_outcome_insufficient_material_is_zero():
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert board.is_insufficient_material()
    # board.result(claim_draw=True) returns "1/2-1/2"
    assert _outcome_from_board(board) == 0
