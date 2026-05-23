// math-engine-cpp — entry point.
//
// Two modes:
//   1. UCI engine  (default):
//        math_engine [--strategy <spec.json>]
//      Loads optional JSON strategy spec (overrides hardcoded cx13), then
//      delegates to uci_loop().
//
//   2. Eval probe  (--eval-fen):
//        math_engine [--strategy <spec.json>] --eval-fen "<FEN>"
//      Prints the integer centipawn eval for one position and exits. Used by
//      the Python↔C++ cross-check test.

#include "eval.h"
#include "eval_dyn.h"
#include "uci.h"

#include "chess.hpp"

#include <cstdio>
#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    std::string strategy_path;
    std::string eval_fen;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--strategy" || arg == "-s") && i + 1 < argc) {
            strategy_path = argv[++i];
        } else if (arg == "--eval-fen" && i + 1 < argc) {
            eval_fen = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "math-engine-cpp\n"
                      << "  --strategy <path>   load JSON expression-tree strategy\n"
                      << "  --eval-fen <fen>    evaluate one FEN and exit\n";
            return 0;
        }
    }

    if (!strategy_path.empty()) {
        try {
            auto spec = std::make_shared<math_engine::eval_dyn::StrategySpec>(
                math_engine::eval_dyn::load_strategy(strategy_path));
            math_engine::eval::set_dynamic_strategy(spec);
            std::cerr << "loaded strategy: " << spec->id
                      << "  (from " << strategy_path << ")\n";
        } catch (const std::exception& e) {
            std::cerr << "ERROR loading strategy: " << e.what() << "\n";
            return 1;
        }
    }

    if (!eval_fen.empty()) {
        try {
            chess::Board b(eval_fen);
            std::cout << math_engine::eval::evaluate(b) << "\n";
        } catch (const std::exception& e) {
            std::cerr << "ERROR evaluating FEN: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    math_engine::uci_loop();
    return 0;
}
