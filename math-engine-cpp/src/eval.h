// Evaluation function.
//
// evaluate(board) returns centipawns from WHITE's perspective.
// Positive = white better; negative = black better.
//
// v0: simple material-only eval (sanity baseline).
// v0.1: PySR cx=13 expression embedded (see math_engine_cpp_v0.md).

#pragma once

#include "chess.hpp"

#include <memory>
#include <string>

namespace math_engine::eval_dyn { struct StrategySpec; }

namespace math_engine::eval {

// Default dispatcher. If a dynamic strategy has been set via
// set_dynamic_strategy(), routes there; otherwise to evaluate_sr_cx13.
int evaluate(const chess::Board& board);

// Handcrafted centipawn material baseline.
int evaluate_material(const chess::Board& board);

// PySR Pareto cx=13 expression from pysr_chess_stockfish.md.
int evaluate_sr_cx13(const chess::Board& board);

// Install a dynamic (JSON-loaded) strategy as the active eval. After this
// call, evaluate() routes to eval_dyn instead of the hardcoded cx13 path.
// Pass nullptr (or call clear_dynamic_strategy) to revert.
void set_dynamic_strategy(std::shared_ptr<const eval_dyn::StrategySpec> spec);
void clear_dynamic_strategy();

// ID of the currently-active strategy, or "cx13_iter0" if none loaded.
std::string active_strategy_id();

}  // namespace math_engine::eval
