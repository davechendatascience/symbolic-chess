# math-engine-cpp

A UCI-compatible chess engine using a symbolic-regression-derived
evaluation function. Iter-0 ships with the cx=13 PySR Pareto equation
distilled from Stockfish; future iterations come from self-play training
(see `docs/math_engine_cpp_v0.md`).

## Build

Requires C++17, CMake 3.16+, and a C++ compiler (MSVC, GCC, or Clang).

```
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

Output: `build/math_engine` (or `build/Release/math_engine.exe` on
Windows MSVC).

## Run

Standalone UCI engine — interact via stdin/stdout or load into any UCI GUI
(CuteChess, Arena, Lichess BotBoard, etc.):

```
$ ./build/math_engine
uci
id name math-engine-cpp 0.1.0
id author PySR + market-analysis
uciok
isready
readyok
position startpos moves e2e4 e7e5
go depth 5
bestmove ...
quit
```

## Test

```
cmake --build build --target test_eval
./build/test_eval
ctest --test-dir build
```

## Architecture

```
src/
├── main.cpp     # entry: delegates to uci_loop()
├── uci.{h,cpp}  # UCI protocol (input parsing, output formatting)
├── search.{h,cpp}  # negamax + alpha-beta + move ordering
└── eval.{h,cpp}    # PySR-derived evaluation
third_party/
└── chess.hpp    # disservin/chess-library (MIT, vendored)
```

See `docs/math_engine_cpp_v0.md` (in repo root) for the full design
sketch and roadmap.

## Status

| Component | v0 status |
|---|---|
| UCI loop                       | ✅ |
| Alpha-beta + capture ordering  | ✅ |
| Material eval                  | ✅ |
| PySR cx=13 eval                | ⏳ |
| Transposition table            | ⏳ |
| Quiescence search              | ⏳ |
| Iterative deepening            | ⏳ |
| CuteChess-CLI Elo measurement  | ⏳ |

## License

MIT. See `LICENSE`. `third_party/chess.hpp` is MIT (Disservin).
