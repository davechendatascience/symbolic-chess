# math-engine-cpp v0 — design sketch

**Status**: design doc, not yet implemented.
**Location**: `engines/math-engine-cpp/` (subdir of market-analysis; designed
for clean extraction to a standalone GitHub project).

## What this is

A standalone C++ chess engine that runs the symbolic regression-discovered
evaluation function inside a competitive alpha-beta search. UCI-compatible
binary, so any chess GUI (CuteChess, Arena, lichess-bot) can drive it and
pit it against Stockfish or other engines.

```
PySR / market-analysis (Python)   →  closed-form eval expression
                                      │
                                      ↓ (manual transcription, ~5 LOC)
engines/math-engine-cpp/ (C++)    →  UCI engine binary
                                      │
                                      ↓
CuteChess-CLI tournament          →  Elo vs Stockfish, vs other engines,
                                      vs prior iterations of this engine
```

The C++ engine itself is **not where SR happens**. SR stays in Python because
PySR is Python+Julia. The C++ side is the *runtime* — fast search + fast
eval — that turns a PySR expression into a competitive chess player.

## Why C++ at all (when Python "works")

Three reasons:

1. **Search speed.** Pure-Python alpha-beta is ~10k nodes/sec. Stockfish is
   ~100M nodes/sec. The Python engine searches depth 3-4 in tournament time;
   a C++ engine searches depth 8-10. Search depth dominates eval quality —
   a mediocre eval at depth 8 routinely beats a strong eval at depth 3.
2. **Research artefact.** A standalone UCI engine is the standard deliverable
   in the chess engine community — submittable to CCRL, playable on Lichess,
   compatible with tools like CuteChess and OpenBench. A Python prototype is
   not.
3. **Self-improvement loop scale.** The self-improvement architecture (search-
   depth bootstrapping or population coevolution) requires playing 1000s of
   games per iteration. At ~30s/game in Python, that's 8+ hours per iter.
   At ~3s/game in C++, it's under an hour.

## Project layout

```
engines/math-engine-cpp/
├── CMakeLists.txt
├── README.md
├── LICENSE                  # MIT (chess-library is MIT; PySR-derived eval is ours)
├── .gitignore
├── docs/
│   └── architecture.md      # implementation notes, perft results, etc.
├── third_party/
│   └── chess-library/       # disservin/chess-library, vendored
├── src/
│   ├── main.cpp             # entry point + UCI loop
│   ├── uci.h, uci.cpp       # UCI protocol
│   ├── search.h, search.cpp # alpha-beta + ID + quiescence + time mgmt
│   ├── tt.h, tt.cpp         # transposition table (Zobrist)
│   ├── eval.h, eval.cpp     # PySR-derived evaluation
│   ├── move_order.h, .cpp   # captures-first, killers, history
│   └── types.h              # common types and constants
├── tests/
│   ├── perft.cpp            # move-gen correctness
│   ├── test_eval.cpp        # eval cross-check vs Python
│   └── test_uci.cpp         # UCI command parsing
└── bench/
    └── bench.cpp            # standard NPS benchmark
```

**Extraction plan.** When ready for its own repo:
```
git filter-branch --subdirectory-filter engines/math-engine-cpp/ -- --all
```
preserves the commit history for just this subdirectory. The vendored
chess-library moves cleanly; everything else is self-contained.

## Dependencies

| Library | Purpose | License | Notes |
|---|---|---|---|
| [disservin/chess-library](https://github.com/Disservin/chess-library) | Bitboard board, move generation, FEN/UCI parsing, Zobrist | MIT | Header-only, C++17, ~5k LOC, used by Halogen and other competitive engines |
| (none else) | — | — | No SIMD/threading deps in v0; single-threaded engine |

**Why chess-library, not Stockfish's code or write-from-scratch:**
- Stockfish is GPL; would force us GPL and require carrying Stockfish's
  conventions. We want clean MIT.
- Write-from-scratch is ~3 weeks just for legal-move generation + perft
  validation. Out of scope for v0.
- chess-library is purpose-built as a competitive-engine foundation. The
  separation of "search/eval" (ours) from "move gen + board rep" (library)
  is exactly the right architectural split.

## Build

CMake + any C++17-capable compiler:

```
cd engines/math-engine-cpp
cmake -B build -S .
cmake --build build --config Release
./build/math-engine    # or .\build\Release\math-engine.exe on Windows
```

Tested compilers (v0): MSVC 2022, MinGW-w64 GCC 13+, GCC 11+ on Linux,
Clang 14+ on macOS.

## UCI scope (v0)

Standard UCI subset:

| Command | Behaviour |
|---|---|
| `uci` | identifies as "math-engine-cpp" + options |
| `isready` | responds `readyok` |
| `ucinewgame` | clears TT |
| `position startpos [moves m1 m2 ...]` | sets position from start + moves |
| `position fen <FEN> [moves ...]` | sets position from FEN + moves |
| `go depth N` | search to depth N, return `bestmove` |
| `go movetime N` | search for N milliseconds |
| `go wtime ... btime ...` | tournament time control |
| `stop` | abort current search, return current best |
| `quit` | exit |

Out of v0: pondering, `go infinite`, MultiPV, options beyond Hash size.

## Search (v0)

| Feature | Why |
|---|---|
| **Negamax + alpha-beta** | Standard foundation |
| **Iterative deepening** | Lets time-controlled search return best-so-far on abort |
| **Transposition table** | Zobrist hashing; 2-bucket replacement (always-replace + depth-preferred). ~10-20% pruning improvement. |
| **Move ordering** | (1) TT move, (2) captures by MVV-LVA, (3) killer moves (2 per ply), (4) history heuristic. Doubles effective search depth on tactical positions. |
| **Quiescence search** | At depth-0 leaves: search captures (and promotions) only until quiet. Eliminates horizon-effect blunders like game 5 in iter-0 tournament. |
| **Repetition / 50-move detection** | Required for correctness in long games |

Out of v0: null-move pruning, late move reductions, futility pruning,
aspiration windows, multi-threading. These are tier 2 optimizations that
buy ~50-200 Elo each; v0 should be plain-vanilla competitive search.

## Evaluation (v0)

The cx=13 PySR Pareto equation from
`benchmarks/results/pysr_chess_stockfish.md`:

```
eval(board) = (material_net / 0.01460058)
            - ((W_mobility * central_BP) /
               ((1.3168311 - B_king_zone_own_pawns) - central_BB))
```

C++ implementation in `eval.cpp`:

```cpp
float evaluate(const chess::Board& b) {
    float material_net = compute_material_net(b);
    float w_mobility   = float(chess::movegen::legalmoves(b, chess::Color::WHITE).size());
    float central_bp   = count_central_pawns(b, chess::Color::BLACK);
    float king_zone_p  = king_zone_own_pawns(b, chess::Color::BLACK);
    float central_bb   = count_central_bishops(b, chess::Color::BLACK);

    return (material_net / 0.01460058f)
         - ((w_mobility * central_bp) /
            ((1.3168311f - king_zone_p) - central_bb));
}
```

**Side-to-move convention.** The expression is in WHITE's perspective.
Negamax in `search.cpp` flips sign at each ply, so the caller need not
worry about STM.

**Versioning.** Each PySR-discovered eval gets a numbered struct
(`EvalCx13_500Pos_D15`) so multiple evals can coexist for tournament
comparison. The active one is selected at compile time via a CMake option
or at runtime via a UCI option.

**Correctness check.** `tests/test_eval.cpp` reads a set of FENs +
expected eval values (computed in Python via `make_fast_eval_cx13`) and
asserts agreement to 1e-3. If C++ and Python disagree, the bug is in the
C++ feature computation, not the expression.

## Testing

1. **Perft** — standard move-gen correctness test. chess-library's perft
   should pass at depth 6 for the standard test suite (Kiwipete, Position 3,
   etc.). If perft passes, our move generation is correct.
2. **Eval cross-check** — Python eval vs C++ eval on N FENs, assert match.
3. **UCI smoke** — pipe `uci\nisready\nposition startpos\ngo depth 4\nquit`
   through the binary, verify `bestmove` is emitted.
4. **Tournament vs Stockfish** — CuteChess-CLI with 100-game match at
   short time control, capture PGN and Elo.

## Performance targets

| Metric | v0 target | Justification |
|---|---|---|
| Nodes per second | ≥ 500k | 50× our Python engine; conservative for C++ alpha-beta |
| Depth at 1 sec/move | ≥ 6 | Tournament-relevant depth |
| Elo vs Stockfish at d=1 | ≥ +200 (beats Stockfish d=1) | The Python engine scores ~15% vs SF d=1; a 50× faster engine with same eval should crush SF d=1 |
| Elo vs Stockfish at d=8 | -200 to -500 | Reasonable for handcrafted-eval engine vs NNUE |
| Compile time (release) | ≤ 30s | Single TU per source file, header-only deps |

## Acceptance criteria for v0

The engine is "v0 complete" when ALL of:

1. Compiles cleanly with MSVC, MinGW, and GCC on Linux
2. Passes the standard perft test suite
3. Plays a 100-game match vs Stockfish (any depth) without crashing,
   timeouts, or illegal-move emissions
4. Beats Stockfish at depth 1 in a 100-game match (target ≥ 60% score)
5. README has clear build + run + test instructions
6. Eval cross-check passes (Python and C++ agree to 1e-3 on 100+ FENs)

## What v0 doesn't do

- **Doesn't train.** SR / training stays in Python. The C++ engine is a
  runtime; new evals come in by editing `eval.cpp` and recompiling.
- **Doesn't tune itself.** Search parameters (TT size, history weights,
  killer counts) are hand-set, not tuned via SPSA or CLOP. Tier 2 work.
- **Doesn't do opening books / endgame tablebases.** Out of scope for v0.

## Self-improvement loop (post-v0)

Once v0 ships, the self-improvement architecture (Architecture C from
`docs/benchmark_stockfish_distillation.md` extensions):

1. v0 plays itself via CuteChess-CLI → games + outcomes
2. SR (Python) trains on outcome-decayed positions → new expression
3. New expression embedded into v0.1 (recompile)
4. v0.1 plays v0 in a gauntlet, also plays Stockfish for ground-truth Elo
5. Iterate

The CuteChess-CLI pipeline does double duty: it's both the validation
harness AND the training-data generator.

## Build order

1. **Skeleton + chess-library + Hello-UCI** (1 day, ~500 LOC). CMake compiles,
   `uci` command identifies the engine, `quit` exits cleanly.
2. **Material eval + negamax + alpha-beta** (1 day, ~300 LOC). Engine plays a
   legal move at fixed depth, passes self-play smoke test.
3. **PySR cx=13 eval + eval cross-check** (½ day, ~150 LOC). Eval values
   match Python to 1e-3.
4. **Transposition table + Zobrist** (1 day, ~200 LOC). Search depth 5 in
   the same time it previously needed for depth 4.
5. **Quiescence search + move ordering** (1 day, ~200 LOC). Tactical
   blunder rate drops; horizon-effect losses (like Python game 5 in 20
   plies) disappear.
6. **Iterative deepening + time management** (½ day, ~150 LOC). Engine
   handles UCI `go wtime/btime` properly.
7. **CuteChess-CLI tournament** (½ day, no new C++). 100-game match vs
   Stockfish at various depths, PGN + Elo reported.

Total: ~5 dev-days for the v0 deliverable.
