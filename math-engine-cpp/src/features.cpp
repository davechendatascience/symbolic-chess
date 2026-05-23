// Chess feature bank — mirrors symbolic_chess.expression_layer.board.
//
// 29 features in canonical order. See features.h for naming + indexing rules.
// Implementations are direct ports of the Python numpy code; comments call
// out non-obvious choices.
//
// Mobility uses legalmoves (matching existing iter-0 C++ eval.cpp). Python
// chess_feature_bank uses pseudo_legal_moves, but the iter-1 expression does
// not reference mobility, so the small numeric drift on mobility features is
// inconsequential for the cross-check.

#include "features.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace math_engine::features {

namespace {

const std::vector<std::string> kFeatureNames = {
    // Piece counts (10)
    "WP_count", "WN_count", "WB_count", "WR_count", "WQ_count",
    "BP_count", "BN_count", "BB_count", "BR_count", "BQ_count",
    // Material net (1)
    "material_net",
    // Pawn structure (4)
    "W_doubled_pawns", "B_doubled_pawns",
    "W_passed_pawns",  "B_passed_pawns",
    // Central control (6)
    "central_WP", "central_WN", "central_WB",
    "central_BP", "central_BN", "central_BB",
    // King zone (4)
    "W_king_zone_enemy", "B_king_zone_enemy",
    "W_king_zone_own_pawns", "B_king_zone_own_pawns",
    // Pawn-file imbalance (1)
    "pawn_file_imbalance",
    // Phase (1)
    "phase",
    // Mobility (2)
    "W_mobility", "B_mobility",
};

// d4=27, e4=28, d5=35, e5=36 (rank*8 + file, 0-indexed from a1)
constexpr int kCentralSquares[4] = { 3*8+3, 3*8+4, 4*8+3, 4*8+4 };

int piece_to_unit_offset(chess::PieceType pt) {
    switch (pt.internal()) {
        case chess::PieceType::PAWN:   return 0;
        case chess::PieceType::KNIGHT: return 1;
        case chess::PieceType::BISHOP: return 2;
        case chess::PieceType::ROOK:   return 3;
        case chess::PieceType::QUEEN:  return 4;
        case chess::PieceType::KING:   return 5;
        default: return -1;
    }
}

// Count legal moves for a given side, regardless of board's stm.
int legal_moves_for(const chess::Board& board, chess::Color side) {
    if (board.sideToMove() == side) {
        chess::Movelist mv;
        chess::movegen::legalmoves(mv, board);
        return static_cast<int>(mv.size());
    }
    std::string fen = board.getFen();
    auto sp = fen.find(' ');
    if (sp != std::string::npos && sp + 1 < fen.size()) {
        fen[sp + 1] = (side == chess::Color::WHITE) ? 'w' : 'b';
    }
    chess::Board tmp(fen);
    chess::Movelist mv;
    chess::movegen::legalmoves(mv, tmp);
    return static_cast<int>(mv.size());
}

}  // namespace

const std::vector<std::string>& feature_names() {
    return kFeatureNames;
}

int feature_index(const std::string& name) {
    for (size_t i = 0; i < kFeatureNames.size(); ++i)
        if (kFeatureNames[i] == name) return static_cast<int>(i);
    return -1;
}

std::vector<float> compute_features(const chess::Board& board) {
    std::vector<float> f(kFeatureNames.size(), 0.0f);

    // counts[color][piece_offset] where color: 0=W, 1=B, offset: P=0..K=5
    int counts[2][6] = {{0}};
    // pawns_per_file[color][file]
    int pawns_per_file[2][8] = {{0}};
    // central counts (P,N,B per side)
    int central[2][3] = {{0}};

    int wk_sq = -1, bk_sq = -1;
    // Track pawn locations as a (color, rank, file) bool grid for passed-pawn calc.
    bool pawn_on[2][8][8] = {{{false}}};

    for (int sq = 0; sq < 64; ++sq) {
        auto piece = board.at(chess::Square(sq));
        if (piece == chess::Piece::NONE) continue;
        int color_idx = (piece.color() == chess::Color::WHITE) ? 0 : 1;
        int off = piece_to_unit_offset(piece.type());
        if (off < 0) continue;
        ++counts[color_idx][off];

        int rank = sq / 8;
        int file = sq % 8;

        if (off == 0) {  // PAWN
            ++pawns_per_file[color_idx][file];
            pawn_on[color_idx][rank][file] = true;
        }
        if (off == 5) {  // KING
            if (color_idx == 0) wk_sq = sq; else bk_sq = sq;
        }
        // Central squares — track P/N/B per side
        if (off <= 2) {  // P, N, B
            for (int cs : kCentralSquares) {
                if (sq == cs) { ++central[color_idx][off]; break; }
            }
        }
    }

    auto set = [&](const char* name, float val) {
        int idx = feature_index(name);
        if (idx >= 0) f[idx] = val;
    };

    // Piece counts
    set("WP_count", counts[0][0]); set("WN_count", counts[0][1]);
    set("WB_count", counts[0][2]); set("WR_count", counts[0][3]);
    set("WQ_count", counts[0][4]);
    set("BP_count", counts[1][0]); set("BN_count", counts[1][1]);
    set("BB_count", counts[1][2]); set("BR_count", counts[1][3]);
    set("BQ_count", counts[1][4]);

    // Material net (P=1, N=3, B=3, R=5, Q=9)
    float mat = (counts[0][0] - counts[1][0]) * 1.0f
              + (counts[0][1] - counts[1][1]) * 3.0f
              + (counts[0][2] - counts[1][2]) * 3.0f
              + (counts[0][3] - counts[1][3]) * 5.0f
              + (counts[0][4] - counts[1][4]) * 9.0f;
    set("material_net", mat);

    // Doubled pawns
    int wd = 0, bd = 0;
    for (int file = 0; file < 8; ++file) {
        if (pawns_per_file[0][file] > 1) wd += pawns_per_file[0][file] - 1;
        if (pawns_per_file[1][file] > 1) bd += pawns_per_file[1][file] - 1;
    }
    set("W_doubled_pawns", wd);
    set("B_doubled_pawns", bd);

    // Passed pawns (per Python passed_pawns_count). A white pawn at (r,f) is
    // passed iff no black pawn sits on file f-1, f, or f+1 at any rank > r.
    int w_passed = 0, b_passed = 0;
    for (int r = 0; r < 8; ++r) {
        for (int file = 0; file < 8; ++file) {
            if (pawn_on[0][r][file]) {
                bool blocked = false;
                for (int rr = r + 1; rr < 8 && !blocked; ++rr) {
                    for (int df = -1; df <= 1 && !blocked; ++df) {
                        int ff = file + df;
                        if (ff < 0 || ff > 7) continue;
                        if (pawn_on[1][rr][ff]) blocked = true;
                    }
                }
                if (!blocked) ++w_passed;
            }
            if (pawn_on[1][r][file]) {
                bool blocked = false;
                for (int rr = r - 1; rr >= 0 && !blocked; --rr) {
                    for (int df = -1; df <= 1 && !blocked; ++df) {
                        int ff = file + df;
                        if (ff < 0 || ff > 7) continue;
                        if (pawn_on[0][rr][ff]) blocked = true;
                    }
                }
                if (!blocked) ++b_passed;
            }
        }
    }
    set("W_passed_pawns", w_passed);
    set("B_passed_pawns", b_passed);

    // Central control
    set("central_WP", central[0][0]); set("central_WN", central[0][1]);
    set("central_WB", central[0][2]);
    set("central_BP", central[1][0]); set("central_BN", central[1][1]);
    set("central_BB", central[1][2]);

    // King zone: 3x3 around king. Count enemy N/B/R/Q in W's zone; own pawns separately.
    auto in_zone = [](int king_sq, int target_sq) {
        if (king_sq < 0) return false;
        int kr = king_sq / 8, kf = king_sq % 8;
        int tr = target_sq / 8, tf = target_sq % 8;
        return std::abs(kr - tr) <= 1 && std::abs(kf - tf) <= 1;
    };
    int w_kz_enemy = 0, b_kz_enemy = 0;
    int w_kz_own_pawns = 0, b_kz_own_pawns = 0;
    for (int sq = 0; sq < 64; ++sq) {
        auto piece = board.at(chess::Square(sq));
        if (piece == chess::Piece::NONE) continue;
        int color_idx = (piece.color() == chess::Color::WHITE) ? 0 : 1;
        int off = piece_to_unit_offset(piece.type());
        if (off < 0) continue;
        // enemy N/B/R/Q (offsets 1..4) in white's king zone
        if (color_idx == 1 && off >= 1 && off <= 4 && in_zone(wk_sq, sq)) ++w_kz_enemy;
        if (color_idx == 0 && off >= 1 && off <= 4 && in_zone(bk_sq, sq)) ++b_kz_enemy;
        // own pawn in own king zone
        if (color_idx == 0 && off == 0 && in_zone(wk_sq, sq)) ++w_kz_own_pawns;
        if (color_idx == 1 && off == 0 && in_zone(bk_sq, sq)) ++b_kz_own_pawns;
    }
    set("W_king_zone_enemy", w_kz_enemy);
    set("B_king_zone_enemy", b_kz_enemy);
    set("W_king_zone_own_pawns", w_kz_own_pawns);
    set("B_king_zone_own_pawns", b_kz_own_pawns);

    // Pawn-file imbalance: sum_files |w_pawns_in_file - b_pawns_in_file|
    int imbalance = 0;
    for (int file = 0; file < 8; ++file) {
        imbalance += std::abs(pawns_per_file[0][file] - pawns_per_file[1][file]);
    }
    set("pawn_file_imbalance", imbalance);

    // Phase: nonpawn-nonking pieces / 14, clipped to [0,1]
    int nonpawn = 0;
    for (int c = 0; c < 2; ++c) {
        for (int o = 1; o <= 4; ++o) nonpawn += counts[c][o];   // N,B,R,Q both sides
    }
    set("phase", std::min(1.0f, std::max(0.0f, nonpawn / 14.0f)));

    // Mobility (legal moves per side)
    set("W_mobility", static_cast<float>(legal_moves_for(board, chess::Color::WHITE)));
    set("B_mobility", static_cast<float>(legal_moves_for(board, chess::Color::BLACK)));

    return f;
}

}  // namespace math_engine::features
