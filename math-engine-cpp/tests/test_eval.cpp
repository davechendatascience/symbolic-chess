// Smoke test for evaluate() — verifies material accounting on a few FENs.
// Compiled when BUILD_TESTS=ON; run via ctest.

#include "eval.h"
#include "chess.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace {

int approx_assert(int got, int want, int tol, const char* label) {
    if (std::abs(got - want) > tol) {
        std::cerr << "FAIL " << label << ": got " << got
                  << ", expected " << want << " (+/-" << tol << ")\n";
        return 1;
    }
    std::cout << "PASS " << label << ": " << got << "\n";
    return 0;
}

}  // namespace

int main() {
    using namespace math_engine::eval;
    int failures = 0;

    // ---- evaluate_material baseline ----
    chess::Board start;
    failures += approx_assert(evaluate_material(start), 0, 0,
                              "material: starting position");
    chess::Board white_queen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    failures += approx_assert(evaluate_material(white_queen), 900, 0,
                              "material: white up queen");
    chess::Board black_rook("4k3/8/8/8/8/8/8/r3K3 w - - 0 1");
    failures += approx_assert(evaluate_material(black_rook), -500, 0,
                              "material: black up rook");

    // ---- evaluate_sr_cx13 cross-check against Python make_fast_eval_cx13.
    // Reference values produced by benchmarks/math_engine_vs_stockfish.py.
    // Tolerance of 2 absorbs float-to-int truncation across the two
    // implementations.
    failures += approx_assert(evaluate_sr_cx13(start), 0, 2,
                              "sr_cx13: starting position");

    chess::Board after_e4("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
    failures += approx_assert(evaluate_sr_cx13(after_e4), 0, 2,
                              "sr_cx13: after 1.e4");

    chess::Board italian("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4");
    failures += approx_assert(evaluate_sr_cx13(italian), 48, 2,
                              "sr_cx13: italian-ish");

    failures += approx_assert(evaluate_sr_cx13(white_queen), 616, 2,
                              "sr_cx13: KQ vs K (material-only term)");

    chess::Board kvk("8/8/8/8/8/8/8/4K2k w - - 0 1");
    failures += approx_assert(evaluate_sr_cx13(kvk), 0, 2,
                              "sr_cx13: K vs K");

    if (failures > 0) {
        std::cerr << failures << " tests failed\n";
        return 1;
    }
    std::cout << "all tests passed\n";
    return 0;
}
