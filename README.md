# symbolic-chess

A research project exploring **symbolic regression as a learner of game strategy**.
Distills closed-form chess evaluation functions from data, plays the resulting
expressions inside a real chess engine, and iterates via self-play.

## What's here

```
symbolic-chess/
├── src/symbolic_chess/         Python framework
│   ├── expression_layer/       SR primitives (operators, compile, PySR adapter)
│   └── chess_engine/           Python alpha-beta engine + Stockfish self-play
├── math-engine-cpp/            Standalone UCI chess engine in C++ (the runtime)
├── benchmarks/                 Distillation + tournament scripts
├── docs/                       Design sketches (one per major component)
└── tests/                      pytest suite for the Python framework
```

## The story

1. **Discover an eval** — PySR fits a closed-form expression `f(board features) → centipawns`
   against ~5k Stockfish-labelled positions. Pareto front gives a small, interpretable
   evaluator. Best at cx=13: `(material_net / 0.0146) − ((W_mobility * central_BP) /
   ((1.317 − B_king_zone_own_pawns) − central_BB))`.
2. **Plug into a search engine** — that expression becomes the leaf evaluator inside an
   alpha-beta + TT + quiescence + iterative-deepening engine, compiled in C++ via
   [chess-library](https://github.com/Disservin/chess-library). Exposes UCI; any chess
   GUI can drive it.
3. **Measure** — UCI-vs-UCI tournament against Stockfish at known-depth Elo references.
   Iter-0 C++ + cx=13 eval scores ~52% vs Stockfish d=1 → ~1750 Elo.
4. **Self-improve** — engine plays itself, outcomes feed back into PySR → new eval →
   recompile → measure → iterate. Architecture C of `docs/math_engine_cpp_v0.md`.

## Sub-experiment: KPK equilibrium

The other thrust is *testing chess theory via SR*. KPK (K+P vs K) is solved in
textbooks; SR rediscovers the rule-of-square / king-proximity / opposition structure
at concept level (Pareto cx=6/8/16), plateaus at 88% accuracy because the current
operator set lacks the piecewise/modular structure for full key-squares logic. See
`docs/kpk_equilibrium_distillation.md` and `benchmarks/results/pysr_kpk.md`.

## Build

**Python framework:**
```
pip install -r requirements.txt
pytest tests/
```

**C++ engine:**
```
cd math-engine-cpp
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
./build/math_engine            # or build/Release/math_engine.exe on Windows MSVC
```

**Run a tournament vs Stockfish:**
```
python benchmarks/cpp_engine_vs_stockfish.py --n-games 20 --math-depth 6 \
    --sf-depths 1 2 3 --stockfish <path/to/stockfish>
```

## Status & design docs

| Component | Doc | Status |
|---|---|---|
| Higher-order SR ops (fold/scan) + compile_expr | `docs/expression_layer_higher_order.md` | shipped |
| Stockfish-eval distillation benchmark | `docs/benchmark_stockfish_distillation.md` | iter-0 shipped, larger corpus pending |
| KPK equilibrium test | `docs/kpk_equilibrium_distillation.md` | shipped (concept-level success) |
| math-engine-cpp v0 | `docs/math_engine_cpp_v0.md` | shipped (TT + quiescence + ID + UCI) |

## License

MIT. See `LICENSE`.
