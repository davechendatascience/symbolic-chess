// Evaluation function.
//
// evaluate(board) returns centipawns from WHITE's perspective.
// Positive = white better; negative = black better.
//
// v0: simple material-only eval (sanity baseline).
// v0.1: PySR cx=13 expression embedded (see math_engine_cpp_v0.md).

#pragma once

#include "chess.hpp"

namespace math_engine::eval {

// Default dispatcher — currently routes to evaluate_sr_cx13.
int evaluate(const chess::Board& board);

// Handcrafted centipawn material baseline.
int evaluate_material(const chess::Board& board);

// PySR Pareto cx=13 expression from pysr_chess_stockfish.md.
int evaluate_sr_cx13(const chess::Board& board);

}  // namespace math_engine::eval
