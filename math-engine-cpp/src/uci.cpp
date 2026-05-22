// UCI protocol implementation — v0 scope.
//
// Recognised commands: uci, isready, ucinewgame, position, go, stop, quit.
// Unknown commands are silently ignored (per UCI spec).

#include "uci.h"
#include "search.h"

#include "chess.hpp"

#include <iostream>
#include <sstream>
#include <string>

namespace math_engine {

namespace {

constexpr const char* ENGINE_NAME = "math-engine-cpp";
constexpr const char* ENGINE_AUTHOR = "PySR + market-analysis";
constexpr const char* ENGINE_VERSION = "0.1.0";

// Active game state. UCI is single-threaded by design.
chess::Board board;

void handle_uci() {
    std::cout << "id name " << ENGINE_NAME << " " << ENGINE_VERSION << "\n";
    std::cout << "id author " << ENGINE_AUTHOR << "\n";
    // No options exposed in v0
    std::cout << "uciok" << std::endl;
}

void handle_isready() {
    std::cout << "readyok" << std::endl;
}

void handle_ucinewgame() {
    board = chess::Board();
    search::clear_tt();
}

// position startpos [moves m1 m2 ...]
// position fen <FEN parts> [moves ...]
void handle_position(std::istringstream& iss) {
    std::string token;
    iss >> token;
    if (token == "startpos") {
        board = chess::Board();
        if (iss >> token && token == "moves") {
            // fall through to apply moves
        } else {
            return;
        }
    } else if (token == "fen") {
        // FEN has 6 space-separated fields; collect them.
        std::string fen;
        for (int i = 0; i < 6 && iss >> token; ++i) {
            if (i > 0) fen += " ";
            fen += token;
        }
        board = chess::Board(fen);
        if (iss >> token && token != "moves") return;
    } else {
        return;   // malformed
    }
    // Apply move list (UCI long algebraic, e.g., e2e4, g7g8q)
    while (iss >> token) {
        chess::Move mv = chess::uci::uciToMove(board, token);
        if (mv == chess::Move::NO_MOVE) break;
        board.makeMove(mv);
    }
}

// go [depth N] [movetime N] [wtime N btime N winc N binc N]
// Iterative deepening: search until depth or time budget exhausted.
void handle_go(std::istringstream& iss) {
    search::SearchLimits limits;
    limits.max_depth = 64;   // hard cap; ID will stop earlier on time
    int wtime = 0, btime = 0, winc = 0, binc = 0;
    int movetime = -1;
    bool depth_set = false;

    std::string token;
    while (iss >> token) {
        if (token == "depth")          { iss >> limits.max_depth; depth_set = true; }
        else if (token == "movetime")  iss >> movetime;
        else if (token == "wtime")     iss >> wtime;
        else if (token == "btime")     iss >> btime;
        else if (token == "winc")      iss >> winc;
        else if (token == "binc")      iss >> binc;
    }

    if (movetime > 0) {
        limits.time_ms = movetime;
    } else if (wtime > 0 || btime > 0) {
        // Simple time mgmt: allocate ~1/30 of remaining time + 25% of increment.
        int my_time = (board.sideToMove() == chess::Color::WHITE) ? wtime : btime;
        int my_inc  = (board.sideToMove() == chess::Color::WHITE) ? winc  : binc;
        limits.time_ms = std::max(50, my_time / 30 + my_inc / 4);
    } else if (!depth_set) {
        limits.max_depth = 6;   // sensible default for "go" with no args
    }

    auto result = search::find_best_move_id(board, limits);
    if (result.best_move == chess::Move::NO_MOVE) {
        std::cout << "bestmove 0000" << std::endl;
        return;
    }
    // UCI info line — depth, score, nodes
    std::cout << "info depth " << result.depth_reached
              << " score cp " << result.score
              << " nodes " << result.nodes
              << std::endl;
    std::cout << "bestmove " << chess::uci::moveToUci(result.best_move) << std::endl;
}

}  // namespace

void uci_loop() {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string command;
        iss >> command;

        if (command == "uci")              handle_uci();
        else if (command == "isready")     handle_isready();
        else if (command == "ucinewgame")  handle_ucinewgame();
        else if (command == "position")    handle_position(iss);
        else if (command == "go")          handle_go(iss);
        else if (command == "stop")        { /* v0: no async search to stop */ }
        else if (command == "quit")        break;
        // else: silently ignore (per UCI spec)
    }
}

}  // namespace math_engine
