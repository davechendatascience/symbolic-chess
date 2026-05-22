"""C++ math-engine-cpp vs Stockfish tournament — Elo measurement.

Same Elo methodology as benchmarks/math_engine_vs_stockfish.py (iter-0 Python
engine), but plays the C++ binary via UCI subprocess. Directly comparable to
the iter-0 results — quantifies the Elo gain from:
  - C++ vs Python search (~100x NPS speedup)
  - TT + quiescence + iterative deepening
"""
from __future__ import annotations
import argparse
import math
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np
import chess
import chess.engine

from chess_corpus import find_stockfish  # type: ignore


OUT_REPORT = ROOT / "benchmarks" / "results" / "cpp_engine_vs_stockfish.md"

STOCKFISH_ELO_AT_DEPTH = {
    1: 1750, 2: 1950, 3: 2150, 4: 2300, 5: 2400, 6: 2500,
}


# ---------------- Elo math (mirrors math_engine_vs_stockfish.py) ----------------

def score_to_elo_diff(score: float, n: int) -> tuple[float, float, float]:
    def elo(s: float) -> float:
        if s <= 0.0: return -400.0
        if s >= 1.0: return 400.0
        return -400.0 * math.log10(1.0 / s - 1.0)

    z = 1.96
    if n <= 0:
        return elo(score), -400.0, 400.0
    denom = 1.0 + z * z / n
    centre = (score + z * z / (2 * n)) / denom
    margin = z * math.sqrt((score * (1 - score) + z * z / (4 * n)) / n) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return elo(score), elo(lo), elo(hi)


# ---------------- Game play ----------------

def play_game(white_engine, white_limit, black_engine, black_limit,
              random_plies: int, rng: np.random.Generator,
              max_plies: int = 200) -> tuple[str, int]:
    """Play one game. Each engine is a chess.engine.SimpleEngine; limit is its
    chess.engine.Limit. random_plies opening moves break determinism.
    """
    board = chess.Board()
    plies = 0
    while plies < random_plies and not board.is_game_over(claim_draw=True):
        legal = list(board.legal_moves)
        if not legal: break
        board.push(legal[int(rng.integers(0, len(legal)))])
        plies += 1
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        engine = white_engine if board.turn == chess.WHITE else black_engine
        limit = white_limit if board.turn == chess.WHITE else black_limit
        result = engine.play(board, limit)
        mv = result.move
        if mv is None: break
        board.push(mv)
        plies += 1
    return board.result(claim_draw=True), plies


def play_tournament(cpp_engine, cpp_limit, sf_engine, sf_limit,
                    sf_depth: int, n_games: int, seed: int,
                    random_plies: int) -> dict:
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    for g in range(n_games):
        math_white = (g % 2 == 0)
        if math_white:
            white_e, white_l, black_e, black_l = cpp_engine, cpp_limit, sf_engine, sf_limit
        else:
            white_e, white_l, black_e, black_l = sf_engine, sf_limit, cpp_engine, cpp_limit
        t0 = time.time()
        result, n_plies = play_game(
            white_e, white_l, black_e, black_l,
            random_plies=random_plies, rng=rng,
        )
        elapsed = time.time() - t0
        if result == "1-0":
            score = 1.0 if math_white else 0.0
        elif result == "0-1":
            score = 1.0 if not math_white else 0.0
        else:
            score = 0.5
        if score == 1.0: wins += 1
        elif score == 0.5: draws += 1
        else: losses += 1
        print(f"  game {g+1}/{n_games} | math={'W' if math_white else 'B'} | "
              f"result={result:>7} | plies={n_plies:>3} | score={score:.1f} | "
              f"{elapsed:.1f}s", flush=True)
    total = wins + 0.5 * draws
    wr = total / n_games
    elo_pt, elo_lo, elo_hi = score_to_elo_diff(wr, n_games)
    return {
        "sf_depth": sf_depth, "n_games": n_games,
        "wins": wins, "draws": draws, "losses": losses,
        "score": total, "win_rate": wr,
        "elo_diff": elo_pt, "elo_lo": elo_lo, "elo_hi": elo_hi,
        "sf_elo_est": STOCKFISH_ELO_AT_DEPTH.get(sf_depth, None),
    }


# ---------------- Report ----------------

def write_report(args, results, elapsed):
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# math-engine-cpp vs Stockfish\n\n")
        f.write(f"Engine: `{args.engine}`\n")
        f.write(f"Eval: cx=13 PySR Pareto (compiled C++)\n")
        f.write(f"Search features: alpha-beta + TT + quiescence + iterative deepening\n\n")
        f.write(f"**math-engine-cpp depth:** {args.math_depth}\n")
        f.write(f"**Games per opponent depth:** {args.n_games}\n\n")

        f.write("## Tournament results\n\n")
        f.write("| Stockfish depth | Stockfish Elo (est) | Wins | Draws | Losses | "
                "Math win-rate | Elo diff (95% CI) | Math Elo (est) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            sf_elo = r["sf_elo_est"]
            math_elo_pt = (sf_elo + r["elo_diff"]) if sf_elo else None
            math_elo_lo = (sf_elo + r["elo_lo"]) if sf_elo else None
            math_elo_hi = (sf_elo + r["elo_hi"]) if sf_elo else None
            ci_str = f"{r['elo_diff']:+.0f} ({r['elo_lo']:+.0f}, {r['elo_hi']:+.0f})"
            math_elo_str = (f"{math_elo_pt:.0f} ({math_elo_lo:.0f}, {math_elo_hi:.0f})"
                            if math_elo_pt is not None else "n/a")
            f.write(f"| {r['sf_depth']} | {sf_elo} | {r['wins']} | {r['draws']} | "
                    f"{r['losses']} | {r['win_rate']:.3f} | {ci_str} | {math_elo_str} |\n")

        f.write("\n## Comparison to iter-0 Python engine\n\n")
        f.write("From `benchmarks/results/math_engine_vs_stockfish.md` "
                "(Python engine, depth 3, same cx=13 eval, n=20):\n\n")
        f.write("| Opponent | Iter-0 (Py d=3) Elo | This run (C++) Elo | Gain |\n")
        f.write("|---|---|---|---|\n")
        iter0 = {1: 1582, 2: 1759, 3: 1935}
        for r in results:
            iter0_elo = iter0.get(r["sf_depth"])
            this_elo = (r["sf_elo_est"] + r["elo_diff"]) if r["sf_elo_est"] else None
            gain = (this_elo - iter0_elo) if (this_elo is not None and iter0_elo) else None
            iter0_str = str(iter0_elo) if iter0_elo else "n/a"
            this_str = f"{this_elo:.0f}" if this_elo is not None else "n/a"
            gain_str = f"{gain:+.0f}" if gain is not None else "n/a"
            f.write(f"| Stockfish d={r['sf_depth']} | {iter0_str} | {this_str} | {gain_str} |\n")

        f.write(f"\n_Elapsed: {elapsed:.1f}s_\n")
    print(f"\nReport written: {OUT_REPORT}")


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=20)
    parser.add_argument("--math-depth", type=int, default=6)
    parser.add_argument("--sf-depths", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--engine", type=str,
                        default="math-engine-cpp/build/math_engine.exe")
    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-plies", type=int, default=4)
    args = parser.parse_args()

    cpp_path = str(Path(args.engine).resolve())
    sf_path = find_stockfish(args.stockfish)
    print(f"math-engine-cpp: {cpp_path}")
    print(f"stockfish:       {sf_path}\n")

    t0 = time.time()
    all_results = []
    with chess.engine.SimpleEngine.popen_uci(cpp_path) as cpp_engine, \
         chess.engine.SimpleEngine.popen_uci(sf_path) as sf_engine:
        cpp_limit = chess.engine.Limit(depth=args.math_depth)
        for sf_depth in args.sf_depths:
            sf_limit = chess.engine.Limit(depth=sf_depth)
            print(f"\n--- Stockfish depth {sf_depth} "
                  f"(~{STOCKFISH_ELO_AT_DEPTH.get(sf_depth, '?')} Elo) ---")
            r = play_tournament(
                cpp_engine, cpp_limit, sf_engine, sf_limit,
                sf_depth=sf_depth, n_games=args.n_games,
                seed=args.seed + sf_depth, random_plies=args.random_plies,
            )
            all_results.append(r)
            print(f"  Result: W={r['wins']} D={r['draws']} L={r['losses']} | "
                  f"score={r['win_rate']:.3f} | Elo diff {r['elo_diff']:+.0f}")

    elapsed = time.time() - t0
    write_report(args, all_results, elapsed)
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
