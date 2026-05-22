# Stockfish distillation benchmark — design sketch

**Status**: design doc, not yet implemented.

## What this is

A benchmark where the framework observes Stockfish play and tries to discover a
closed-form expression for Stockfish's value function. The test asks: can
symbolic regression recover a compact, interpretable evaluator that mimics an
opaque strong evaluator?

```
Observation:    (position p_i, stockfish_eval(p_i, depth=20)) for i = 1..N
                — N ≈ 10⁴–10⁵ positions sampled from games
SR target:      f(features(p)) ≈ stockfish_eval(p), measured in centipawns
Search space:   expression_layer with chess-domain feature variables
                and spatial-aggregation operators
Success:        TEST R² and Kendall-τ vs Stockfish; closed-form expression
                small enough to read
```

This is the chess specialisation of a general distillation methodology:
**aggregate-channel features + symbolic regression on an opaque scalar target.**
The same pattern transfers to Go, image classifiers, or any spatially-structured
opaque evaluator — chess is a concrete first instance with a known upper bound
(pre-NNUE handcrafted eval, ~2400 Elo).

## Why this is meaningful (and what it won't do)

What the benchmark tests:

1. **Can SR distill an opaque scalar function** given the right feature bank?
   This is a clean methodological question; the answer is interesting whether
   yes or no.
2. **What is the complexity floor** for a Stockfish-mimicking expression at
   each R² level? The Pareto front (complexity vs error) is the deliverable.
3. **Does the framework's spatial-aggregation machinery generalise** from
   `spatial.py` (1D PDE fields) to 2D piece-channel grids? If yes, the same
   primitives work for any 2D opaque evaluator.

What it will not do:

- **Beat Stockfish or recover NNUE.** NNUE is a 50M-parameter network;
  any closed-form expression has 10²–10³ effective parameters at most. The
  ceiling is Stockfish's *handcrafted* eval (pre-2020), which scored ~2400 Elo.
- **Capture tactical search.** SR distils the *leaf evaluator*, not the
  alpha-beta search around it. A Stockfish-eval-mimic plugged into shallow
  search is much weaker than full-depth Stockfish.
- **Solve credit assignment from game outcomes.** Inputs are
  (position, Stockfish eval) pairs — a supervised regression. Self-play
  with terminal-only reward is a separate problem (separate doc).

## Design choice — feature representation

| Level | Variables | What SR learns | Tradeoff |
|---|---|---|---|
| **B1: Handcrafted scalars** | ~20 scalars: material, mobility, PST sums, king-safety counts, passed pawns | Weights & interactions of pre-engineered terms | Narrow: SR has little to discover; bank does the work |
| **B2: Aggregated spatial channels** | ~50 vars: per-piece-type file-sums, rank-sums, diagonal-sums, king-zone occupancy + B1 scalars | Both feature combinations *and* spatial weightings | Generalises to any 2D opaque evaluator; finite-dim PySR-tractable |
| **B3: Raw 8×8 × 12-channel grid** | 768 binary vars | Features and weights from scratch | Too many vars for PySR's tree search; needs an embedding step first |

**Recommendation: B2.** Generalises better than B1 (the methodology is
"build spatial aggregates, search over their combinations" — transferable to
Go, image evaluators, board-game variants). Tractable unlike B3. The spatial
aggregation operators are themselves a reusable contribution to
`expression_layer/`.

B1 still serves as a sanity-check baseline: an SR run with only handcrafted
scalars sets the floor. B2 must beat it on TEST R² to justify the spatial
machinery.

## Architecture

### New module: `src/lib/expression_layer/board.py`

Parallel to `spatial.py`. Pure numpy. No dependency on a chess engine — takes
board state as input, produces feature arrays.

```python
# Board encoding: (N_positions, 12, 8, 8) int8 array
#   12 channels = {white, black} × {P, N, B, R, Q, K}
#   8×8 grid, channel value ∈ {0, 1}
def encode_position(fen: str) -> np.ndarray  # → (12, 8, 8)
def encode_corpus(fens: list[str]) -> np.ndarray  # → (N, 12, 8, 8)

# Spatial aggregations — analogues of spatial.py operators
def file_sum(planes, channel) -> ndarray   # (N, 8) — pawns per file, etc.
def rank_sum(planes, channel) -> ndarray   # (N, 8)
def diagonal_sum(planes, channel) -> ndarray  # (N, 15) — two directions
def king_zone(planes, side) -> ndarray  # (N,) — squares around king
def piece_count(planes, channel) -> ndarray  # (N,) — total material

# Bank builder
def chess_feature_bank(planes) -> dict[str, ndarray]:
    """Returns a dict of named (N,) feature vectors for PySR consumption."""
```

### Feature bank (v0 vocabulary)

Per-side (white, black) yields 2× each:

- **Material**: `P, N, B, R, Q` counts (5 × 2 = 10 vars)
- **Mobility proxy**: `pseudo_moves` (legal moves count, computed via python-chess)
- **Pawn structure**: `pawn_islands`, `passed_pawns`, `doubled_pawns`
- **King safety**: `king_zone_own_pieces`, `king_zone_enemy_pieces`,
  `king_file` (0–7), `king_pawn_shield_count`
- **Spatial aggregates**: `pawn_file_imbalance` (Σ|file_sum_white − file_sum_black|),
  `central_control` (occupancy of d4/d5/e4/e5)
- **Game phase**: `phase` ∈ [0, 1] derived from non-pawn material

Total ~30–40 variables. Each is a (N,) array; PySR sees them as scalar features
per position.

### Corpus

| Source | Size | Notes |
|---|---|---|
| Lichess puzzle DB (CC0) | 4M tactics | Already-annotated, varied phases |
| Self-play Stockfish games | Custom | Generate with `--depth 20`, log every 5th ply |
| Computer chess championship PGNs (CCRL) | ~500k games | Diverse strong engines |

**v0 plan:** 50k positions sampled from Lichess puzzles (filtered for
diverse-phase coverage), each scored by `stockfish_eval(position, depth=20)`
in centipawns. Centipawns clipped to [−1000, +1000] (positions with mating
eval are excluded — they're not regression targets, they're terminal states).

### TRAIN / TEST split

By **game date**, not random. Train on positions from games before 2024-01-01,
test on 2024+ positions. Avoids position leakage and tests temporal stability
of the distilled expression. This mirrors the project-wide convention
(walk-forward, no random shuffling on time-correlated data).

### Loss

PySR target: predict Stockfish centipawn eval. Loss = MSE on clipped eval.
Secondary metric: Kendall-τ between predicted and Stockfish eval (rank
agreement is what matters in practice — the absolute scale of centipawns is
arbitrary).

### Baselines

1. **Constant predictor** (eval = 0): floor.
2. **Material-only linear regression** (closed form): standard chess HCE
   floor. Expected R² ≈ 0.60.
3. **Full handcrafted-feature linear regression** on the B2 bank: this is
   "what a smart linear model can do with the same inputs". SR must beat this
   to justify the framework.
4. **Linear regression** ≅ pre-NNUE Stockfish 11 HCE in spirit: published
   weights exist for comparison.

### Pipeline file: `benchmarks/run_pysr_chess_stockfish.py`

Mirrors the structure of the existing `run_pysr_*.py` scripts:

```
1. Load corpus (or generate via stockfish CLI if absent — cached to data/chess/)
2. Encode positions → planes
3. Build feature bank via chess_feature_bank()
4. TRAIN/TEST split by date
5. Fit baselines (constant, material-LR, full-LR) → record R² floor
6. Run PySR with binary ops {+, -, *, /} + unary {tanh, abs}
   - Complexity penalty tuned so Pareto front spans ~5 to ~50 complexity
   - Multiple seeds (3+)
7. For each Pareto equation: report TRAIN/TEST R², Kendall-τ, complexity
8. Plot Pareto front: complexity vs TEST R²
9. Write results to benchmarks/results/pysr_chess_stockfish.md
```

## Acceptance criteria

The benchmark "succeeds" (in the sense that the framework demonstrably works
on this problem) if **any of the following**:

1. SR finds a closed-form expression with TEST R² > 0.85 at complexity < 30
   that beats full-feature linear regression by > 0.03 R².
2. SR finds a complexity ≤ 15 expression with TEST R² > 0.75 (compact and
   beats material-only LR substantially).
3. The Pareto front shows a clear complexity → R² gain curve that flattens
   above the LR baseline.

Failure mode (clean negative result):
- SR's best Pareto expression at complexity ≤ 50 is dominated by linear
  regression on the same features. This would mean: closed-form chess eval
  is essentially a linear combination of these handcrafted features, and SR's
  flexibility doesn't help. Still a publishable finding.

## What it tests about the framework

| Framework capability | Tested by |
|---|---|
| Spatial-aggregation operators on 2D channels | The B2 feature bank using `board.py` |
| Domain-feature-bank pattern (cf. `spatial.py` for PDEs) | Module structure itself |
| PySR adapter with non-time-series inputs | Whole pipeline (positions are i.i.d. samples, not a time series) |
| Walk-forward / temporal splitting discipline | TRAIN/TEST by date |
| Pareto-front reporting convention | Results doc |

Higher-order operators (`fold`, `scan` from `expression_layer_higher_order.md`)
are **not** used in this benchmark — chess positions don't have a time axis to
fold over. Stockfish distillation and higher-order ops are independent
workstreams; this benchmark validates the *spatial* side of the framework, the
other validates the *temporal* side.

## Build order

1. `board.py` with encode + 4–5 aggregation operators + chess_feature_bank
   (~2 days, ~300 LOC + python-chess as new dep)
2. Corpus generation: stockfish CLI wrapper, 50k Lichess positions annotated,
   cached to `data/chess/stockfish_eval_50k.parquet` (~1 day, runs overnight)
3. Baselines: const + LRs, results table (~½ day)
4. PySR run + Pareto reporting (~1 day, several hours of compute per seed)
5. Results doc `benchmarks/results/pysr_chess_stockfish.md` (~½ day)

Total: ~5 dev-days + overnight compute. New external dep: `python-chess` (pip),
`stockfish` (system binary, brew/apt installable).

## Open questions

- **Should the corpus exclude tactical positions?** Tactical positions have
  evals dominated by search depth, not positional features. Including them
  may force SR to model noise; excluding them biases toward strategic
  positions. v0 plan: include all; filter via robust loss (Huber) instead.
- **Self-play vs Lichess corpus.** Self-play guarantees in-distribution
  coverage; Lichess puzzles are filtered for tactical interest. v0 starts
  with Lichess for convenience; add self-play if results suggest distribution
  shift.
- **Phase-conditional expressions.** A single expression for opening / middle-
  game / endgame may be too rigid. v0 keeps it single. If the Pareto front
  caps below the LR baseline, try a `phase` gate (one expression per phase
  via piecewise tanh) as a v1 extension.
