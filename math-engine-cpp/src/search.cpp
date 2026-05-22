// Alpha-beta + TT + quiescence + iterative deepening — tier-2 engine.
//
// Negamax convention: every call returns score from side-to-move's perspective.
// Eval is WHITE perspective; sign-flipped at qsearch / depth-0 leaves.
//
// The TT shortens search by reusing scores from previously-visited positions.
// Quiescence search resolves tactical sequences at leaves so static eval
// isn't called in mid-capture. Iterative deepening lets time-limited search
// return a best-so-far when aborted; the TT's deeper-iteration entries
// dramatically improve move ordering in subsequent iterations.

#include "search.h"
#include "eval.h"
#include "tt.h"

#include "chess.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

namespace math_engine::search {

namespace {

constexpr int INF = 1'000'000;
constexpr int MATE_SCORE = 100'000;
constexpr int MATE_THRESHOLD = MATE_SCORE - 1024;   // mate-in-N range
constexpr int QSEARCH_MAX_PLY = 32;                 // safety bound on q-tree depth

tt::Table g_tt(32);   // 32 MB default

// Per-search bookkeeping (set by find_best_move_id, read by search_node).
struct SearchContext {
    std::chrono::steady_clock::time_point deadline;
    bool has_deadline = false;
    bool aborted = false;
    uint64_t nodes = 0;
};

bool out_of_time(SearchContext& ctx) {
    if (!ctx.has_deadline) return false;
    // Only check the clock every 2048 nodes to keep overhead low
    if ((ctx.nodes & 2047) != 0) return false;
    return std::chrono::steady_clock::now() >= ctx.deadline;
}

// MVV-LVA-lite: captures first (sorted by victim value descending).
int move_score(const chess::Board& board, chess::Move mv, chess::Move tt_move) {
    if (mv == tt_move) return 1'000'000;
    if (board.isCapture(mv)) {
        auto victim = board.at(mv.to());
        return 100'000 + static_cast<int>(victim.type());
    }
    return 0;
}

void order_moves(chess::Movelist& moves, const chess::Board& board, chess::Move tt_move) {
    std::sort(moves.begin(), moves.end(),
              [&board, tt_move](chess::Move a, chess::Move b) {
                  return move_score(board, a, tt_move) > move_score(board, b, tt_move);
              });
}

// ---- Quiescence search ----
//
// At depth-0 leaves, instead of returning static eval immediately, we keep
// searching captures (and promotions) until a quiet position is reached.
// "Stand pat" means: the side to move can also choose to NOT capture (i.e.
// the static eval is a lower bound on the achievable score for STM).
int qsearch(chess::Board& board, int alpha, int beta, int ply, SearchContext& ctx) {
    ctx.nodes++;
    if (out_of_time(ctx)) { ctx.aborted = true; return 0; }
    if (ply >= QSEARCH_MAX_PLY) {
        int cp = eval::evaluate(board);
        return (board.sideToMove() == chess::Color::WHITE) ? cp : -cp;
    }
    int stand_pat = eval::evaluate(board);
    if (board.sideToMove() == chess::Color::BLACK) stand_pat = -stand_pat;
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    chess::Movelist captures;
    chess::movegen::legalmoves<chess::movegen::MoveGenType::CAPTURE>(captures, board);
    order_moves(captures, board, chess::Move::NO_MOVE);

    for (auto mv : captures) {
        board.makeMove(mv);
        int score = -qsearch(board, -beta, -alpha, ply + 1, ctx);
        board.unmakeMove(mv);
        if (ctx.aborted) return 0;
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

// ---- Main negamax ----
int negamax(chess::Board& board, int depth, int alpha, int beta, int ply, SearchContext& ctx) {
    ctx.nodes++;
    if (out_of_time(ctx)) { ctx.aborted = true; return 0; }

    // Terminal detection
    auto game_result = board.isGameOver();
    if (game_result.first != chess::GameResultReason::NONE) {
        if (game_result.first == chess::GameResultReason::CHECKMATE) {
            return -MATE_SCORE + ply;   // prefer faster mates
        }
        return 0;
    }

    // TT probe
    uint64_t key = board.hash();
    const tt::Entry* tt_entry = g_tt.probe(key);
    chess::Move tt_move = chess::Move::NO_MOVE;
    if (tt_entry && tt_entry->depth >= depth) {
        int s = tt_entry->score;
        if (tt_entry->bound == tt::BOUND_EXACT) return s;
        if (tt_entry->bound == tt::BOUND_LOWER && s >= beta) return s;
        if (tt_entry->bound == tt::BOUND_UPPER && s <= alpha) return s;
    }
    if (tt_entry) tt_move = tt_entry->best_move;

    if (depth <= 0) return qsearch(board, alpha, beta, ply, ctx);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    order_moves(moves, board, tt_move);

    int best = -INF;
    chess::Move best_move = chess::Move::NO_MOVE;
    int orig_alpha = alpha;
    for (auto mv : moves) {
        board.makeMove(mv);
        int score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, ctx);
        board.unmakeMove(mv);
        if (ctx.aborted) return 0;
        if (score > best) { best = score; best_move = mv; }
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;   // fail-high
    }

    // TT store
    uint8_t bound = tt::BOUND_EXACT;
    if (best <= orig_alpha) bound = tt::BOUND_UPPER;
    else if (best >= beta)  bound = tt::BOUND_LOWER;
    g_tt.store(key, best, depth, bound, best_move);

    return best;
}

}  // namespace

chess::Move find_best_move(chess::Board& board, int depth) {
    SearchLimits limits;
    limits.max_depth = depth;
    return find_best_move_id(board, limits).best_move;
}

SearchResult find_best_move_id(chess::Board& board, SearchLimits limits) {
    SearchContext ctx;
    if (limits.time_ms > 0) {
        ctx.has_deadline = true;
        ctx.deadline = std::chrono::steady_clock::now()
                     + std::chrono::milliseconds(limits.time_ms);
    }

    SearchResult result;
    chess::Movelist root_moves;
    chess::movegen::legalmoves(root_moves, board);
    if (root_moves.empty()) return result;
    result.best_move = root_moves[0];   // fallback

    // Iterative deepening: each iteration warms the TT for the next.
    for (int d = 1; d <= limits.max_depth; ++d) {
        chess::Move iter_best = chess::Move::NO_MOVE;
        int iter_score = -INF;

        uint64_t key = board.hash();
        const tt::Entry* tt_entry = g_tt.probe(key);
        chess::Move tt_move = tt_entry ? tt_entry->best_move : chess::Move::NO_MOVE;
        order_moves(root_moves, board, tt_move);

        int alpha = -INF, beta = INF;
        for (auto mv : root_moves) {
            board.makeMove(mv);
            int score = -negamax(board, d - 1, -beta, -alpha, 1, ctx);
            board.unmakeMove(mv);
            if (ctx.aborted) break;
            if (score > iter_score) { iter_score = score; iter_best = mv; }
            if (iter_score > alpha) alpha = iter_score;
        }

        if (ctx.aborted) break;   // incomplete iteration — discard

        // Store root entry so next iteration probes find this iter's best move
        g_tt.store(key, iter_score, d, tt::BOUND_EXACT, iter_best);

        result.best_move = iter_best;
        result.score = iter_score;
        result.depth_reached = d;
        result.nodes = ctx.nodes;
    }

    return result;
}

void clear_tt() {
    g_tt.clear();
}

}  // namespace math_engine::search
