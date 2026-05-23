// Evaluation function.
//
// Provides two implementations:
//   - evaluate_material(board): handcrafted centipawn material baseline.
//   - evaluate_sr_cx13(board):  PySR Pareto cx=13 expression from
//                               benchmarks/results/pysr_chess_stockfish.md.
//
// evaluate(board) is the default dispatcher — currently routes to the SR eval.
// Both return ints in centipawn-equivalent units, WHITE perspective.

#include "eval.h"
#include "eval_dyn.h"

#include "chess.hpp"

#include <memory>
#include <string>

namespace math_engine::eval {

namespace {
std::shared_ptr<const eval_dyn::StrategySpec> g_dyn_strategy;
}

void set_dynamic_strategy(std::shared_ptr<const eval_dyn::StrategySpec> spec) {
    g_dyn_strategy = std::move(spec);
}

void clear_dynamic_strategy() {
    g_dyn_strategy.reset();
}

std::string active_strategy_id() {
    return g_dyn_strategy ? g_dyn_strategy->id : std::string("cx13_iter0");
}

namespace {

// ---- Unit material weights (matches Python make_fast_eval_cx13) ----
constexpr float UNIT_PAWN   = 1.0f;
constexpr float UNIT_KNIGHT = 3.0f;
constexpr float UNIT_BISHOP = 3.0f;
constexpr float UNIT_ROOK   = 5.0f;
constexpr float UNIT_QUEEN  = 9.0f;

// ---- Centipawn material weights (handcrafted baseline) ----
constexpr int CP_PAWN   = 100;
constexpr int CP_KNIGHT = 320;
constexpr int CP_BISHOP = 330;
constexpr int CP_ROOK   = 500;
constexpr int CP_QUEEN  = 900;

// d4, e4, d5, e5 — file and rank 0-indexed
constexpr int CENTRAL_SQUARES[4] = {
    /*d4=*/ 8 * 3 + 3,
    /*e4=*/ 8 * 3 + 4,
    /*d5=*/ 8 * 4 + 3,
    /*e5=*/ 8 * 4 + 4,
};

// Count WHITE's legal moves regardless of actual side-to-move.
// Implemented by serializing the board to FEN, flipping the stm field, and
// re-parsing. ~10-20us per call; acceptable for v0.
float white_mobility(const chess::Board& board) {
    if (board.sideToMove() == chess::Color::WHITE) {
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        return static_cast<float>(moves.size());
    }
    std::string fen = board.getFen();
    // FEN field 2 is side-to-move ('w' or 'b'). Replace.
    auto sp1 = fen.find(' ');
    if (sp1 != std::string::npos && sp1 + 1 < fen.size()) {
        fen[sp1 + 1] = 'w';
    }
    chess::Board tmp;
    tmp.setFen(fen);
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, tmp);
    return static_cast<float>(moves.size());
}

}  // namespace

int evaluate_material(const chess::Board& board) {
    int score = 0;
    for (int sq = 0; sq < 64; ++sq) {
        auto piece = board.at(chess::Square(sq));
        if (piece == chess::Piece::NONE) continue;
        int v = 0;
        switch (piece.type().internal()) {
            case chess::PieceType::PAWN:   v = CP_PAWN;   break;
            case chess::PieceType::KNIGHT: v = CP_KNIGHT; break;
            case chess::PieceType::BISHOP: v = CP_BISHOP; break;
            case chess::PieceType::ROOK:   v = CP_ROOK;   break;
            case chess::PieceType::QUEEN:  v = CP_QUEEN;  break;
            default: continue;   // KING and NONE
        }
        if (piece.color() == chess::Color::WHITE) score += v;
        else                                      score -= v;
    }
    return score;
}

int evaluate_sr_cx13(const chess::Board& board) {
    float material_net = 0.0f;
    int central_bp = 0;
    int central_bb = 0;
    int b_king_zone_own_pawns = 0;

    // Pass 1: material + central counts via single board scan.
    for (int sq_idx = 0; sq_idx < 64; ++sq_idx) {
        auto piece = board.at(chess::Square(sq_idx));
        if (piece == chess::Piece::NONE) continue;
        float v = 0.0f;
        switch (piece.type().internal()) {
            case chess::PieceType::PAWN:   v = UNIT_PAWN;   break;
            case chess::PieceType::KNIGHT: v = UNIT_KNIGHT; break;
            case chess::PieceType::BISHOP: v = UNIT_BISHOP; break;
            case chess::PieceType::ROOK:   v = UNIT_ROOK;   break;
            case chess::PieceType::QUEEN:  v = UNIT_QUEEN;  break;
            default: continue;   // KING contributes 0 here
        }
        if (piece.color() == chess::Color::WHITE) material_net += v;
        else                                      material_net -= v;

        // central_BP, central_BB tracking
        if (piece.color() == chess::Color::BLACK) {
            for (int cs : CENTRAL_SQUARES) {
                if (sq_idx == cs) {
                    if (piece.type() == chess::PieceType::PAWN) ++central_bp;
                    else if (piece.type() == chess::PieceType::BISHOP) ++central_bb;
                }
            }
        }
    }

    // Pass 2: B_king_zone_own_pawns
    auto bk_sq = board.kingSq(chess::Color::BLACK);
    int bk_idx = bk_sq.index();
    int bk_rank = bk_idx / 8;
    int bk_file = bk_idx % 8;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int df = -1; df <= 1; ++df) {
            int r = bk_rank + dr;
            int f = bk_file + df;
            if (r < 0 || r > 7 || f < 0 || f > 7) continue;
            int sq_idx = r * 8 + f;
            auto piece = board.at(chess::Square(sq_idx));
            if (piece != chess::Piece::NONE
                && piece.color() == chess::Color::BLACK
                && piece.type() == chess::PieceType::PAWN) {
                ++b_king_zone_own_pawns;
            }
        }
    }

    // W_mobility
    float w_mobility = white_mobility(board);

    // The cx=13 expression — exact-copy from
    // benchmarks/results/pysr_chess_stockfish.md:
    //   (material_net / 0.0146)
    //  - ((W_mobility * central_BP) / ((1.3168 - B_king_zone_own_pawns) - central_BB))
    // Uses safe-div (returns 0 on zero denominator) to match Python's _safe_div.
    auto safe_div = [](float a, float b) -> float {
        return (b == 0.0f) ? 0.0f : a / b;
    };
    float term1 = safe_div(material_net, 0.01460058f);
    float denom = (1.3168311f - static_cast<float>(b_king_zone_own_pawns))
                  - static_cast<float>(central_bb);
    float term2 = safe_div(w_mobility * static_cast<float>(central_bp), denom);
    float cp = term1 - term2;
    return static_cast<int>(cp);
}

int evaluate(const chess::Board& board) {
    if (g_dyn_strategy) {
        return eval_dyn::evaluate(*g_dyn_strategy, board);
    }
    return evaluate_sr_cx13(board);
}

}  // namespace math_engine::eval
