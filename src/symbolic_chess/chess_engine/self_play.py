"""Stockfish self-play game generation for SR training data.

A single Stockfish process plays both sides at a fixed depth, producing
(fen, outcome, plies_to_end) records for every position. Each game starts
with a few random plies for opening variety; the remainder is Stockfish vs
Stockfish.

Schema (parquet):
  game_id       int   — index of the game (0..N-1)
  ply_number    int   — 0-indexed ply within the game
  fen           str   — position after the move was played
  outcome       int   — terminal result from WHITE: +1 / 0 / -1
  plies_to_end  int   — number of plies until terminal from this position
  random_phase  bool  — True if move was random opening, False if Stockfish

Use `decay_outcome(outcome, plies_to_end, decay)` to compress the terminal
signal back through the game; the result is the SR regression target.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import chess
import chess.engine

# Reuse stockfish discovery from the benchmark corpus helper
import sys as _sys
_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent.parent.parent / "benchmarks"
if str(_BENCH) not in _sys.path:
    _sys.path.insert(0, str(_BENCH))
from chess_corpus import find_stockfish   # type: ignore


# ---------------- Outcome utilities ----------------

def _outcome_from_board(board: chess.Board) -> int:
    """Terminal outcome from WHITE's perspective: +1 / 0 / -1.

    Assumes board.is_game_over() is True.
    """
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    return 0


def decay_outcome(outcome: int, plies_to_end: int, decay: float = 0.98) -> float:
    """Discount terminal outcome backwards through the game.

    A position N plies before the terminal carries signal = outcome * decay**N.
    SR fits this as the regression target — positions near the end get strong
    labels, early positions get muted labels.
    """
    return float(outcome) * (decay ** int(plies_to_end))


# ---------------- Single-game play ----------------

def _play_one_game(
    engine: chess.engine.SimpleEngine,
    *,
    depth: int,
    random_plies: int,
    max_plies: int,
    rng: np.random.Generator,
) -> list[dict]:
    """Play one game; return a list of per-ply records.

    random_plies opening moves are uniform over legal moves (variety); the
    rest are Stockfish at the given depth.
    """
    board = chess.Board()
    records: list[dict] = []
    moves_played = 0
    while not board.is_game_over(claim_draw=True) and moves_played < max_plies:
        if moves_played < random_plies:
            moves = list(board.legal_moves)
            mv = moves[int(rng.integers(0, len(moves)))]
            random_phase = True
        else:
            result = engine.play(board, chess.engine.Limit(depth=depth))
            mv = result.move
            random_phase = False
        board.push(mv)
        records.append({
            "ply_number": moves_played,
            "fen": board.fen(),
            "random_phase": random_phase,
        })
        moves_played += 1

    outcome = _outcome_from_board(board) if board.is_game_over(claim_draw=True) else 0
    total_plies = len(records)
    for i, rec in enumerate(records):
        rec["outcome"] = outcome
        rec["plies_to_end"] = total_plies - 1 - i
    return records


# ---------------- Top-level corpus builder ----------------

def generate_self_play_corpus(
    out_path: Path | str,
    n_games: int,
    *,
    depth: int = 6,
    random_plies: int = 4,
    max_plies: int = 200,
    engine_path: Optional[str] = None,
    engine_name: str = "engine",
    stockfish_path: Optional[str] = None,    # deprecated alias for back-compat
    seed: int = 0,
    overwrite: bool = False,
    log_every_games: int = 25,
) -> pd.DataFrame:
    """Generate n_games of engine-vs-engine self-play; cache positions to parquet.

    Engine selection:
    - `engine_path` — explicit path to any UCI binary (math-engine-cpp, stockfish, …)
    - `stockfish_path` — back-compat alias; if `engine_path` is None falls back to
      `find_stockfish(stockfish_path)` (resolves env var / PATH).

    `engine_name` is used in log lines only.

    Returns the DataFrame. If out_path exists and overwrite is False, the
    cache is loaded directly.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"Loading cached self-play corpus from {out_path}")
        return pd.read_parquet(out_path)

    if engine_path is not None:
        binary = engine_path
    else:
        binary = find_stockfish(stockfish_path)
    rng = np.random.default_rng(seed)
    print(f"Generating {n_games} games of {engine_name}-vs-{engine_name} "
          f"at depth={depth} ({random_plies} random opening plies)")
    t0 = time.time()
    all_records: list[dict] = []
    with chess.engine.SimpleEngine.popen_uci(binary) as engine:
        for g in range(n_games):
            recs = _play_one_game(
                engine,
                depth=depth,
                random_plies=random_plies,
                max_plies=max_plies,
                rng=rng,
            )
            for r in recs:
                r["game_id"] = g
            all_records.extend(recs)
            if log_every_games and (g + 1) % log_every_games == 0:
                elapsed = time.time() - t0
                rate = (g + 1) / elapsed
                eta = (n_games - g - 1) / rate
                print(f"  game {g+1}/{n_games} | {len(all_records)} positions | "
                      f"{rate:.2f} games/s | ETA {eta:.0f}s")

    df = pd.DataFrame(all_records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} positions ({n_games} games) to {out_path}")
    game_outcomes = df.groupby("game_id")["outcome"].first()
    print(f"  games: white_wins={(game_outcomes == 1).sum()}, "
          f"draws={(game_outcomes == 0).sum()}, "
          f"black_wins={(game_outcomes == -1).sum()}")
    return df


# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stockfish self-play corpus")
    parser.add_argument("--out", type=str,
                        default="data/chess/self_play_games.parquet")
    parser.add_argument("--n", type=int, default=200, help="number of games")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    df = generate_self_play_corpus(
        Path(args.out),
        n_games=args.n,
        depth=args.depth,
        random_plies=args.random_plies,
        max_plies=args.max_plies,
        stockfish_path=args.stockfish,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(df.head())
    print(df.describe(include="all"))
