// Dynamic (JSON-loaded) eval — loads a strategy spec at startup, evaluates by
// walking the AST against the feature vector.
//
// Strategy JSON schema (the relevant fields):
//   {
//     "id": "...",
//     "expression_tree": { ... },   // see ast.h for accepted node forms
//     "output_scale": 100.0          // (optional) multiply eval by this scalar
//                                    // before returning the integer centipawn
//                                    // score; defaults to 1.0
//   }
//
// If the loaded strategy lacks "expression_tree", the legacy "expression"
// field (sympy-format text) is *ignored* — only tree specs are accepted by
// the C++ engine. The Python side is responsible for converting sympy →
// tree before invoking the engine with --strategy.

#pragma once

#include "ast.h"
#include "chess.hpp"

#include <memory>
#include <string>

namespace math_engine::eval_dyn {

struct StrategySpec {
    std::string id;
    std::unique_ptr<ast::Node> root;
    float output_scale = 1.0f;   // result multiplier (centipawn unit conversion)
};

// Load + validate a strategy from JSON. Throws std::runtime_error on
// malformed JSON, unknown operators/variables, or missing expression_tree.
StrategySpec load_strategy(const std::string& json_path);

// Evaluate a board under a loaded strategy. Returns int centipawns
// (WHITE perspective), matching the contract of math_engine::eval::evaluate.
int evaluate(const StrategySpec& spec, const chess::Board& board);

}  // namespace math_engine::eval_dyn
