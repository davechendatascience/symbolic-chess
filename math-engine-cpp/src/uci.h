// UCI protocol entry point.
//
// uci_loop() reads commands from stdin, dispatches to engine internals,
// writes responses to stdout. Standard UCI subset for v0:
//   uci, isready, ucinewgame, position, go, stop, quit.
//
// All UCI I/O lives here; engine modules (search, eval) never touch stdin/stdout.

#pragma once

namespace math_engine {

void uci_loop();

}  // namespace math_engine
