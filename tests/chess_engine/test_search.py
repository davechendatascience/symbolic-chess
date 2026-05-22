"""Tests for chess_engine.search — negamax, alpha-beta, eval wiring."""
import numpy as np
import pytest
import chess

from symbolic_chess.chess_engine.search import (
    negamax, best_move, make_expr_evaluator, material_eval,
    INF, MATE_SCORE,
)
from symbolic_chess.expression_layer import Variable, Constant, add, sub, mul


# ---------------- material_eval ----------------

def test_material_eval_starting_position_is_zero():
    board = chess.Board()
    assert material_eval(board) == 0.0


def test_material_eval_after_capture():
    # White captures black knight: white +320
    board = chess.Board()
    board.set_board_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    # Remove black knight on b8 → white +320
    board.remove_piece_at(chess.B8)
    assert material_eval(board) == 320.0


def test_material_eval_white_up_a_queen():
    # White has extra queen
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert material_eval(board) == 900.0


def test_material_eval_black_up_a_rook():
    board = chess.Board("4k3/8/8/8/8/8/8/r3K3 w - - 0 1")
    assert material_eval(board) == -500.0


# ---------------- Terminal detection ----------------

def test_negamax_returns_mate_score_at_checkmate():
    # Fool's mate position — white is checkmated
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    board.push_san("Qh4#")
    assert board.is_checkmate()
    score = negamax(board, depth=3, alpha=-INF, beta=INF, eval_fn=material_eval)
    # STM (white) is checkmated → -MATE_SCORE
    assert score == -MATE_SCORE


def test_negamax_returns_zero_on_stalemate():
    # Classic stalemate: black king h1, white king f1, white queen g3, black to move
    board = chess.Board("8/8/8/8/8/6Q1/8/5KBk b - - 0 1")
    # Position-specific construction: actually let's use a known stalemate fen
    board = chess.Board("k7/2Q5/2K5/8/8/8/8/8 b - - 0 1")
    # Black king on a8, white queen c7, white king c6, black to move → stalemate
    assert board.is_stalemate()
    score = negamax(board, depth=2, alpha=-INF, beta=INF, eval_fn=material_eval)
    assert score == 0.0


# ---------------- Mate-in-1 search ----------------

def test_best_move_finds_mate_in_one():
    """Corner mate: white queen+king pins black king in the corner.

    Position: black K a1, white K c2, white Q b3.
    White plays Qb1#: queen pinned in front of black king, protected by white K.
    """
    board = chess.Board("8/8/8/8/8/1Q6/2K5/k7 w - - 0 1")
    move = best_move(board, depth=2, eval_fn=material_eval)
    board.push(move)
    assert board.is_checkmate(), f"Expected checkmate after {move}, got fen={board.fen()}"


def test_best_move_prefers_capturing_free_piece():
    """With depth 2, the engine should capture an undefended enemy piece.

    Position is loaded with material so the capture doesn't trigger insufficient-
    material draw (which would mask the preference).
    """
    # White: K e1, N b1, P a2 (so post-capture is still K+N+P vs K, ample material)
    # Black: K e8, P a3 (free pawn, no defender)
    board = chess.Board("4k3/8/8/8/8/p7/P7/1N2K3 w - - 0 1")
    move = best_move(board, depth=2, eval_fn=material_eval)
    assert move == chess.Move.from_uci("b1a3"), f"Expected Nxa3, got {move}"


# ---------------- Alpha-beta consistency (no pruning vs pruning) ----------------

def _negamax_no_prune(board, depth, eval_fn):
    """Reference implementation without alpha-beta — full minimax."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth <= 0:
        cp = eval_fn(board)
        return cp if board.turn == chess.WHITE else -cp
    best = -INF
    for move in board.legal_moves:
        board.push(move)
        s = -_negamax_no_prune(board, depth - 1, eval_fn)
        board.pop()
        if s > best:
            best = s
    return best


def test_alpha_beta_matches_full_minimax():
    """Pruned negamax returns identical score to full minimax on the same position."""
    # Small middlegame position
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    )
    for d in (1, 2, 3):
        pruned = negamax(board, d, -INF, INF, material_eval)
        full = _negamax_no_prune(board, d, material_eval)
        assert pruned == full, f"depth {d}: pruned {pruned} vs full {full}"


# ---------------- make_expr_evaluator wiring ----------------

def test_expr_evaluator_returns_centipawns():
    """A simple material-net Expr should produce a finite eval on starting position."""
    feature_names = ["material_net"]
    material_net = Variable("material_net", np.array([0.0]))
    # eval = 100 * material_net
    expr = mul(Constant(100.0), material_net)
    eval_fn = make_expr_evaluator(expr, feature_names)

    board = chess.Board()
    assert eval_fn(board) == 0.0   # starting position, material_net=0

    # Remove a black knight → material_net = +3 → eval = 300
    board.remove_piece_at(chess.B8)
    assert eval_fn(board) == 300.0


def test_expr_evaluator_drives_best_move():
    """An Expr-based eval can be plugged into best_move successfully."""
    feature_names = ["material_net"]
    material_net = Variable("material_net", np.array([0.0]))
    expr = mul(Constant(100.0), material_net)
    eval_fn = make_expr_evaluator(expr, feature_names)

    # Same as test_best_move_prefers_capturing_free_piece
    board = chess.Board("4k3/8/8/8/8/p7/P7/1N2K3 w - - 0 1")
    move = best_move(board, depth=2, eval_fn=eval_fn)
    assert move == chess.Move.from_uci("b1a3")


# ---------------- best_move robustness ----------------

def test_best_move_returns_none_at_terminal():
    # Position with no legal moves (checkmate)
    board = chess.Board()
    board.push_san("f3")
    board.push_san("e5")
    board.push_san("g4")
    board.push_san("Qh4#")
    assert best_move(board, depth=1, eval_fn=material_eval) is None
