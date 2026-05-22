"""symbolic_chess — symbolic regression as a learner of chess strategy.

Sub-packages:
  expression_layer  Tree representation, operators, compile_expr, PySR adapter,
                    chess board encoding + spatial features.
  chess_engine      Python alpha-beta search + Stockfish self-play helpers.

The C++ engine (math-engine-cpp/) is a sibling at the repo root, not a Python
sub-package.
"""

__version__ = "0.1.0"
