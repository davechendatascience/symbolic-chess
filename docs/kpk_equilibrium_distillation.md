# KPK equilibrium distillation — design sketch

**Status**: design doc, not yet implemented.

## What this is

A minimum-scope test of the thesis: *can symbolic regression discover the
equilibrium value function of a closed game from data?* The chess endgame
K+P vs K (KPK) is the right minimum test:

- **Closed**: complete information, finite state space, no chance
- **Solved**: textbook theory gives an exact closed-form criterion
  ("rule of the square" plus "key-squares" theory) for whether
  white-to-move wins or draws
- **Tractable**: ~30k legal positions, all labellable by Stockfish at
  depth 20+ in minutes — no Lichess scraping, no NNUE leakage
- **Has structure**: the answer is *not* a polynomial in piece
  coordinates. It involves Chebyshev distances, conditional logic, and
  parity (opposition). If SR can recover this, the framework demonstrably
  extracts non-trivial equilibrium structure.

```
Question:    given KPK position (WK_sq, WP_sq, BK_sq, side-to-move),
             can white force a win with best play?
Data:        N legal positions + Stockfish d=20 binary label {win, draw}
SR target:   discover a closed-form classifier f(features) ∈ [0, 1]
Success:     SR's Pareto-front finds an expression of complexity ≤ 15
             with TEST accuracy ≥ 0.90 — comparable to a textbook ruleset
```

## Why KPK and not the full game

The full Stockfish-distillation benchmark (`docs/benchmark_stockfish_distillation.md`)
asks "can SR approximate Stockfish's eval over all positions?" That's a *regression*
problem with a high-dimensional answer (NNUE has ~50M parameters). The fair-game
target there is "match LR baseline within a few R² points."

KPK is different. KPK has a **known correct answer** that *is* expressible in
closed form. If SR can find an expression that matches textbook theory at high
accuracy and low complexity, that's not just regression — it's empirical
verification that the equilibrium structure of a chess endgame lives within
SR's reachable expression space. That's a stronger and more falsifiable claim.

## What success / failure would tell us

| Outcome | Implication for the framework |
|---|---|
| SR rediscovers rule-of-the-square with cx ≤ 15, acc ≥ 0.95 | Strong evidence that SR can extract equilibrium structure for *any* closed-game subproblem. Justifies scaling to KQK, KRK, KBNK, opening-theory, etc. |
| SR fits the data well (acc ≥ 0.95) but at high cx (50+) — opaque expression | SR is "memorising" not "understanding". Suggests current operator set is inadequate for piecewise/conditional structure. Need to extend with explicit `min`, `max`, conditional ops. |
| SR fails to beat full-LR baseline (acc < 0.85) | Closed-form classification of KPK is genuinely hard for SR. Either the feature engineering is insufficient (no Chebyshev distances exposed) or SR's GP cannot find the disjunctive structure. Useful negative result — bounds what SR can/can't do. |

## Theoretical baseline (textbook KPK rules)

The standard chess theory for K+P vs K:

### Rule of the square

If the black king is *outside the square of the pawn*, the pawn promotes
unaided. Formally, with white pawn at file `wp_f`, rank `wp_r` (0-indexed,
rank-7 = 8th rank for white):

```
promo_dist  = 7 − wp_r                       # plies for pawn to promote
file_gap    = |bk_f − wp_f|
rank_gap    = 7 − bk_r                       # plies for BK to reach promotion square (white POV)
inside_sq   = max(file_gap, rank_gap) ≤ promo_dist
```

(With a one-tempo adjustment if it's white's move: `promo_dist + 1` instead.)

If `inside_sq = False`: white wins trivially (BK can't catch the pawn).

### Key squares

If BK is inside the square, the position is won iff WK occupies (or can
reach) one of the **key squares**:

```
For white pawn on rank wp_r, file wp_f (with wp_r ≤ 4, before the 6th rank):
  key squares = { (wp_f − 1, wp_r + 2), (wp_f, wp_r + 2), (wp_f + 1, wp_r + 2) }

For wp_r ≥ 5 (pawn close to promotion):
  key squares = the three squares two ranks ahead, plus the three on rank 7
  (involves rook-pawn special cases for files a and h)
```

Plus the **opposition** rule: with kings facing each other across one square
on the same file/rank, the player *not* to move has the opposition and wins.

### Theoretical baseline as an expression

A pure rule-of-the-square classifier:
```
f(features) = 1.0 if max(|bk_f − wp_f|, 7 − bk_r) > (7 − wp_r) + wtm else 0.0
```
This is cx ~10 and gives ~70-80% accuracy (covers the easy wins, misses
positions where BK is in the square but white still wins via key squares).

A full theoretical classifier (key squares + opposition) is cx ~30-50 by
necessity — it's piecewise, involves modular arithmetic for opposition,
and has a-file/h-file special cases.

**The SR target is to land between these two**: discover the rule-of-the-square
form *and* some key-square extension at complexity ≤ 15-20, accuracy ≥ 0.90.

## Corpus design

### Sampling

- **Sample size**: 5000–10000 unique positions
- **Piece placement**: uniform random over WK ∈ [0, 63], WP ∈ {rank 1–6} ×
  {file 0–3 only}, BK ∈ [0, 63], wtm ∈ {0, 1}
- **File canonicalisation**: WP restricted to files a–d removes mirror
  symmetry without loss of generality (the f, g, h files are identical
  under reflection, reducing legal-position count by ~2×)
- **Reject**: overlapping pieces, kings adjacent, pawn on rank 1 or 8,
  positions where the side *not* to move is in check (illegal)

### Labelling

Stockfish at **depth 20**. Convert score to binary:

```
score_white = info["score"].white()
if score_white.is_mate():
    label = 1 if score_white.mate() > 0 else 0   # negative mate = white loses (rare in KPK)
else:
    cp = score_white.score()
    label = 1 if cp >= 200 else 0                # 200cp threshold
```

For KPK at d=20+, Stockfish gives definitive mate scores for wins and
~0 cp for draws. Borderline positions are rare; the 200cp threshold is
generous.

Caching: parquet at `data/chess/kpk_corpus.parquet`.

### Feature vocabulary

12 scalar features, exposing what theory needs but *not* pre-baking the
answer:

| Name | Definition |
|---|---|
| `wk_rank`, `wk_file` | white king coordinates |
| `wp_rank`, `wp_file` | white pawn coordinates |
| `bk_rank`, `bk_file` | black king coordinates |
| `wtm` | 1.0 if white to move, else 0.0 |
| `promo_dist` | `7 − wp_rank` (plies pawn needs to promote) |
| `d_kp` | Chebyshev distance WK ↔ WP |
| `d_bk_p` | Chebyshev distance BK ↔ WP |
| `d_bk_promo` | Chebyshev distance BK ↔ promotion square `(wp_file, 7)` |
| `d_kk` | Chebyshev distance WK ↔ BK |
| `wp_file_to_edge` | `min(wp_file, 7 − wp_file)` (distance to nearest a/h file) |

Distances are computed up front because SR's operator set lacks `max` —
we expose Chebyshev directly so SR can compose with rather than rediscover
it. Open question: include `max` in the operator set and let SR rediscover
distance metrics? Defer to v1; cx budget too tight for v0.

## Pipeline

`benchmarks/kpk_corpus.py`:
- `sample_kpk_positions(n, seed)` → list of (wk, wp, bk, wtm)
- `kpk_features(wk, wp, bk, wtm)` → dict
- `label_with_stockfish(positions, depth=20)` → labels
- `build_kpk_corpus(out_path, n, depth)` → DataFrame, cached

`benchmarks/run_pysr_kpk.py`:
1. Load corpus
2. Build feature matrix
3. Random TRAIN/TEST split (random is fine — positions are i.i.d., not time-series)
4. Baselines:
   - Constant (predict majority class)
   - Rule-of-the-square classifier (theoretical baseline)
   - Linear regression on all features (overfit-prone but a useful floor)
5. PySR with `binary_operators = {+, -, *, /}`, `unary_operators = {abs, sign, tanh}`,
   target = binary label, MSE loss
6. Pareto front + TEST accuracy per equation
7. Write `benchmarks/results/pysr_kpk.md` with side-by-side comparison of
   discovered Pareto expressions and the theoretical rule-of-the-square form

## Acceptance criteria

| Tier | Condition |
|---|---|
| **Strong pass** | Pareto eq with cx ≤ 15 and TEST accuracy ≥ 0.95 — clear win |
| **Acceptable** | Pareto eq with cx ≤ 20 and TEST accuracy ≥ 0.90 — matches rule-of-the-square |
| **Marginal** | Pareto eq with cx ≤ 30 and TEST accuracy ≥ 0.85 — beats LR but partial |
| **Fail** | No eq exceeds LR baseline at any cx — SR cannot encode the equilibrium |

## What this won't do

- **Doesn't solve KPK** — Stockfish already solves it perfectly. The benchmark
  tests whether SR can *encode* the solution.
- **Doesn't generalise to all chess endings** — KQK, KRK, KBNK each have
  their own structure. KPK is a probe; success motivates trying the others.
- **Doesn't replace a tablebase** — even a perfect SR solution would be
  slower than table lookup. The value is theoretical: a closed-form
  expression is *interpretable* in a way a tablebase is not.

## What comes next if it succeeds

1. **KQK distillation** — simpler than KPK (no pawn complications); should
   yield a clean centralisation rule
2. **KRK distillation** — "drive the lone king to the edge" — known theory
3. **Opening principles** — distill Stockfish's first-10-ply evaluation
   into closed form; tests whether classical opening theory (centre control,
   development, king safety) emerges from SR
4. **Self-improvement loop** — use the KPK methodology as the inner loop
   of search-depth bootstrapping in the broader chess work

## Build order

1. `benchmarks/kpk_corpus.py` — sampling + features + Stockfish labelling
   (~½ day, ~150 LOC; overnight cache run optional)
2. `benchmarks/run_pysr_kpk.py` — full pipeline + theoretical baseline +
   Pareto reporting (~½ day, ~200 LOC; PySR run ~10–30 min)
3. `benchmarks/results/pysr_kpk.md` — generated by the run, hand-edited
   with theory-vs-discovered analysis
