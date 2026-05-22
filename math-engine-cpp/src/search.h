// Alpha-beta search with TT, quiescence, and iterative deepening.
//
// Two entry points:
//   - find_best_move(board, depth)      — fixed-depth search, returns best move
//   - find_best_move_id(board, limits)  — iterative deepening with time/depth limits
//
// Negamax convention: every internal call returns score from STM perspective.
// Eval is WHITE-perspective; sign-flipped at leaves.

#pragma once

#include "chess.hpp"

namespace math_engine::search {

struct SearchLimits {
    int max_depth = 64;       // hard depth cap (search may stop earlier)
    int time_ms   = 0;        // 0 = no time limit (depth-bound only)
};

struct SearchResult {
    chess::Move best_move = chess::Move::NO_MOVE;
    int score = 0;
    int depth_reached = 0;
    uint64_t nodes = 0;
};

// Fixed-depth search (legacy interface, used by tests).
chess::Move find_best_move(chess::Board& board, int depth);

// Iterative deepening + time-managed search.
SearchResult find_best_move_id(chess::Board& board, SearchLimits limits);

// Clear transposition table (call on ucinewgame).
void clear_tt();

}  // namespace math_engine::search
