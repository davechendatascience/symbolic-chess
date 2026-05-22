"""Negamax + alpha-beta chess search with a pluggable evaluation function.

Design:
  - The search is generic over the evaluation. `eval_fn(board) -> centipawns`
    returns score from WHITE's perspective; negamax flips sign per ply.
  - The expression layer wires in via `make_expr_evaluator(expr, feature_names)`
    which composes `compile_expr` (fast scalar lambda) with the chess feature
    bank (`encode_position` + `chess_feature_bank`).
  - A handcrafted `material_eval` is provided as a sanity baseline and to
    bootstrap SR training corpora.

Terminal handling:
  - Checkmate: STM loses → -MATE_SCORE
  - Stalemate / 50-move / threefold / insufficient material: 0
"""
from __future__ import annotations
from typing import Callable
import numpy as np
import chess

from symbolic_chess.expression_layer.core import compile_expr
from symbolic_chess.expression_layer.board import encode_position, chess_feature_bank


INF = 1_000_000.0
MATE_SCORE = 100_000.0

# Standard chess centipawn material values
_MATERIAL = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


# ---------------- Evaluation functions ----------------

def material_eval(board: chess.Board) -> float:
    """Material-only centipawn eval from WHITE's perspective.

    Fast baseline (~5 µs/board) — useful both as a benchmark opponent and as
    the seed eval when bootstrapping SR training data.
    """
    score = 0
    for piece in board.piece_map().values():
        v = _MATERIAL.get(piece.piece_type, 0)
        score += v if piece.color == chess.WHITE else -v
    return float(score)


def make_expr_evaluator(expr, feature_names: list) -> Callable[[chess.Board], float]:
    """Compose compile_expr with the chess feature bank → eval_fn(board).

    The returned callable extracts features for the position, packs them into
    a 1D array in the order of `feature_names`, and calls the compiled lambda.
    Pipeline per call: ~50 µs feature-bank + ~1 µs compiled eval — dominated
    by feature extraction. Incremental feature updates are a v1 optimisation.

    The expression is expected to be in WHITE's perspective (positive = white
    better). Negamax does the sign-flipping per ply.

    Parameters
    ----------
    expr : Expr | Variable | Constant — the SR-discovered evaluator
    feature_names : list of str — names from chess_feature_bank in the order
        the compiled lambda expects them. Must match what the SR was trained on.
    """
    compiled = compile_expr(expr, feature_names)

    def eval_fn(board: chess.Board) -> float:
        fen = board.fen()
        planes = encode_position(fen)[np.newaxis]
        bank = chess_feature_bank(planes, fens=[fen])
        x = np.array([bank[n][0] for n in feature_names], dtype=np.float64)
        return float(compiled(x))

    return eval_fn


# ---------------- Move ordering ----------------

def _order_moves(board: chess.Board, moves):
    """Captures first, then everything else. Cheap pruning win."""
    return sorted(moves, key=lambda m: not board.is_capture(m))


# ---------------- Negamax + alpha-beta ----------------

def negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    eval_fn: Callable[[chess.Board], float],
) -> float:
    """Negamax with alpha-beta pruning. Returns score from STM's perspective.

    Terminal detection runs BEFORE depth check so mate at depth-0 still scores
    correctly.
    """
    # Terminal nodes
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() \
       or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0.0

    if depth <= 0:
        cp = eval_fn(board)
        return cp if board.turn == chess.WHITE else -cp

    best = -INF
    for move in _order_moves(board, board.legal_moves):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, eval_fn)
        board.pop()
        if score > best:
            best = score
        if best > alpha:
            alpha = best
        if alpha >= beta:
            break
    return best


def best_move(
    board: chess.Board,
    depth: int,
    eval_fn: Callable[[chess.Board], float],
) -> chess.Move | None:
    """Return the best move at given search depth, or None if no legal moves."""
    best_score = -INF
    best: chess.Move | None = None
    for move in _order_moves(board, board.legal_moves):
        board.push(move)
        score = -negamax(board, depth - 1, -INF, INF, eval_fn)
        board.pop()
        if score > best_score:
            best_score = score
            best = move
    return best
