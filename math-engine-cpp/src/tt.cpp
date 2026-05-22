// Transposition table implementation.

#include "tt.h"

namespace math_engine::tt {

namespace {

constexpr size_t bytes_per_entry = sizeof(Entry);

size_t round_to_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

}  // namespace

Table::Table(size_t mb) { resize(mb); }

void Table::resize(size_t mb) {
    size_t bytes = mb * 1024 * 1024;
    size_t raw_entries = bytes / bytes_per_entry;
    size_t entries = round_to_pow2(raw_entries) / 2;   // round down to power-of-two
    if (entries < 1024) entries = 1024;
    table_.assign(entries, Entry{});
    mask_ = entries - 1;
}

void Table::clear() {
    for (auto& e : table_) e = Entry{};
}

const Entry* Table::probe(uint64_t key) const {
    const Entry& e = table_[key & mask_];
    return e.key == key ? &e : nullptr;
}

void Table::store(uint64_t key, int score, int depth, uint8_t bound, chess::Move best_move) {
    Entry& e = table_[key & mask_];
    // Always-replace policy (v0). Deeper-preferred would help but adds complexity.
    e.key = key;
    e.score = static_cast<int32_t>(score);
    e.depth = static_cast<int16_t>(depth);
    e.bound = bound;
    e.best_move = best_move;
}

}  // namespace math_engine::tt
