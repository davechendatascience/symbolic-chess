# Math engine vs Stockfish — iteration 0

Source eval: cx=13 PySR Pareto equation from `benchmarks/results/pysr_chess_stockfish.md` (500-position Stockfish d=15 distillation).

**Math engine search depth:** 3
**Games per opponent depth:** 20

## Tournament results

| Stockfish depth | Stockfish Elo (est) | Wins | Draws | Losses | Math win-rate | Elo diff (95% CI) | Math Elo (est) |
|---|---|---|---|---|---|---|---|
| 1 | 1750 | 1 | 9 | 10 | 0.275 | -168 (-333, -4) | 1582 (1417, 1746) |
| 2 | 1950 | 0 | 10 | 10 | 0.250 | -191 (-360, -22) | 1759 (1590, 1928) |
| 3 | 2150 | 0 | 9 | 11 | 0.225 | -215 (-390, -40) | 1935 (1760, 2110) |

## Eval expression

```
eval = (material_net / 0.01460058)
     - ((W_mobility * central_BP) /
        ((1.3168311 - B_king_zone_own_pawns) - central_BB))
```

Five features: material_net (centipawn balance), W_mobility (white pseudo-legal moves), central_BP (black pawns on d4/d5/e4/e5), B_king_zone_own_pawns (black king's pawn shield), central_BB (black bishops in center).

## Interpretation

- **Baseline reference.** Stockfish at d=1 is ~1750 Elo (strong amateur). Material-only at depth 3 would score roughly 1100-1300 Elo.
- **Math engine Elo estimate.** Win-rate against Stockfish at known-depth-Elo gives a calibrated point estimate. CI is wide on n=20-30 games — interpret accordingly.
- **Iteration 0 status.** This is the eval distilled from 500 positions only. Self-improvement architecture (search-depth bootstrap or population coevolution) will produce iter 1+ evals; the right framing is whether these beat the iter-0 Elo here.

_Elapsed: 594.4s_
