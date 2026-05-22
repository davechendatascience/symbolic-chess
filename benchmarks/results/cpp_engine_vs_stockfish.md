# math-engine-cpp vs Stockfish

Engine: `engines/math-engine-cpp/build/math_engine.exe`
Eval: cx=13 PySR Pareto (compiled C++)
Search features: alpha-beta + TT + quiescence + iterative deepening

**math-engine-cpp depth:** 6
**Games per opponent depth:** 20

## Tournament results

| Stockfish depth | Stockfish Elo (est) | Wins | Draws | Losses | Math win-rate | Elo diff (95% CI) | Math Elo (est) |
|---|---|---|---|---|---|---|---|
| 1 | 1750 | 3 | 15 | 2 | 0.525 | +17 (-131, +165) | 1767 (1619, 1915) |
| 2 | 1950 | 1 | 15 | 4 | 0.425 | -53 (-202, +97) | 1897 (1748, 2047) |
| 3 | 2150 | 2 | 8 | 10 | 0.300 | -147 (-308, +13) | 2003 (1842, 2163) |

## Comparison to iter-0 Python engine

From `benchmarks/results/math_engine_vs_stockfish.md` (Python engine, depth 3, same cx=13 eval, n=20):

| Opponent | Iter-0 (Py d=3) Elo | This run (C++) Elo | Gain |
|---|---|---|---|
| Stockfish d=1 | 1582 | 1767 | +185 |
| Stockfish d=2 | 1759 | 1897 | +138 |
| Stockfish d=3 | 1935 | 2003 | +68 |

_Elapsed: 743.2s_
