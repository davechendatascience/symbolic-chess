"""chess_engine — pluggable-eval chess search.

Provides:
  - negamax(board, depth, alpha, beta, eval_fn) — alpha-beta search
  - best_move(board, depth, eval_fn) — top-level move selection
  - make_expr_evaluator(expr, feature_names) — compose compile_expr with
    chess feature bank into a board → centipawn evaluator
  - material_eval — a fast handcrafted eval, useful baseline and for
    bootstrapping SR training data

eval_fn convention: takes a chess.Board, returns centipawns from WHITE's
perspective. The search internally converts via negamax sign-flipping.
"""
from .search import (
    negamax, best_move,
    make_expr_evaluator, material_eval,
    INF,
)
from .self_play import (
    generate_self_play_corpus, decay_outcome,
)

__all__ = [
    "negamax", "best_move",
    "make_expr_evaluator", "material_eval",
    "INF",
    "generate_self_play_corpus", "decay_outcome",
]
