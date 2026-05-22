// Transposition table — Zobrist-hashed cache of search results.
//
// Each entry stores:
//   - key       : Zobrist hash from chess::Board::hash()
//   - score     : negamax score returned at the stored depth
//   - depth     : search depth at which `score` was computed
//   - bound     : EXACT / LOWER (fail-high) / UPPER (fail-low)
//   - best_move : best move from that node (used for move ordering)
//
// Replacement strategy: always-replace. v0 keeps it simple. Tier-2 would
// use depth-preferred + age-aware buckets.

#pragma once

#include "chess.hpp"
#include <cstdint>
#include <vector>

namespace math_engine::tt {

enum Bound : uint8_t { BOUND_EXACT = 0, BOUND_LOWER = 1, BOUND_UPPER = 2 };

struct Entry {
    uint64_t key = 0;
    int32_t  score = 0;
    int16_t  depth = -1;
    uint8_t  bound = BOUND_EXACT;
    chess::Move best_move = chess::Move::NO_MOVE;
};

class Table {
public:
    explicit Table(size_t mb = 16);
    void resize(size_t mb);
    void clear();

    // Probe by Zobrist key. Returns pointer to entry if present, else nullptr.
    // Caller is responsible for verifying key match (we return whichever entry
    // sits at that hash slot).
    const Entry* probe(uint64_t key) const;

    // Store result of a search.
    void store(uint64_t key, int score, int depth, uint8_t bound, chess::Move best_move);

    size_t size_entries() const { return table_.size(); }

private:
    std::vector<Entry> table_;
    uint64_t mask_ = 0;   // size - 1 (size is power-of-two)
};

}  // namespace math_engine::tt
