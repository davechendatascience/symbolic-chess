"""Chess corpus generation: synthetic FENs + Stockfish annotation.

Standalone helper for the chess distillation benchmark. Two responsibilities:

  1. Generate FEN positions covering varied phases. For v0 we use random
     play from the starting position (N plies). Easy to swap for Lichess
     puzzle / CCRL imports later.

  2. Annotate each FEN with `stockfish_eval(depth=D)` in centipawns. Uses
     python-chess's UCI bridge so any UCI-compliant engine binary works.
     Mate scores are clipped to ±1000 (treated as terminal, not regression
     targets — the design doc excludes them but we clip here for safety).

Output: a parquet file with columns (fen, eval_cp, depth, plies_played).
Cached so the benchmark loads instantly on re-runs.

Stockfish binary
----------------
Pass `stockfish_path` explicitly, or set env var STOCKFISH_PATH, or rely on
`shutil.which("stockfish")`. If not found, raises with a clear message.
"""
from __future__ import annotations
import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import chess
import chess.engine


# ---------------- Binary discovery ----------------

def find_stockfish(explicit_path: Optional[str] = None) -> str:
    """Locate a Stockfish-compatible UCI binary. Raises FileNotFoundError if missing.

    Always returns an absolute, resolved path — subprocess launchers (used by
    python-chess) don't always resolve relative paths against cwd correctly
    on Windows.
    """
    if explicit_path:
        p = Path(explicit_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"stockfish_path not a file: {explicit_path}")
        return str(p)
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file():
            return str(p)
    on_path = shutil.which("stockfish") or shutil.which("stockfish.exe")
    if on_path:
        return str(Path(on_path).resolve())
    raise FileNotFoundError(
        "Stockfish binary not found. Options:\n"
        "  1. Pass stockfish_path=<absolute path to stockfish[.exe]>\n"
        "  2. Set STOCKFISH_PATH environment variable\n"
        "  3. Add stockfish to PATH\n"
        "Download from https://stockfishchess.org/download/ or "
        "install via conda: `conda install -c conda-forge stockfish`."
    )


# ---------------- FEN generation ----------------

def generate_random_fens(
    n_positions: int,
    plies_range: tuple[int, int] = (8, 60),
    seed: int = 0,
) -> list[str]:
    """Generate FENs by playing random legal moves from the starting position.

    Each position is sampled at a random ply count in `plies_range`. Reaches
    a fresh starting position when the game ends prematurely (checkmate /
    stalemate / insufficient material).
    """
    rng = np.random.default_rng(seed)
    fens: list[str] = []
    while len(fens) < n_positions:
        board = chess.Board()
        target_plies = int(rng.integers(plies_range[0], plies_range[1] + 1))
        for _ in range(target_plies):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            mv = moves[int(rng.integers(0, len(moves)))]
            board.push(mv)
        # Skip terminal positions — Stockfish gives mate scores there, not eval
        if board.is_game_over():
            continue
        fens.append(board.fen())
    return fens


# ---------------- Stockfish annotation ----------------

def annotate_with_stockfish(
    fens: list[str],
    stockfish_path: Optional[str] = None,
    depth: int = 20,
    eval_clip: int = 1000,
    log_every: int = 500,
) -> np.ndarray:
    """Return per-position Stockfish eval in centipawns, clipped to ±eval_clip.

    Always evaluated from white's perspective. Mate scores are clipped to
    ±eval_clip (treated as bounded approximations of "winning"/"losing").
    """
    binary = find_stockfish(stockfish_path)
    evals = np.zeros(len(fens), dtype=np.float64)
    t0 = time.time()
    with chess.engine.SimpleEngine.popen_uci(binary) as engine:
        for i, fen in enumerate(fens):
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white()
            if score.is_mate():
                cp = float(eval_clip if score.mate() > 0 else -eval_clip)
            else:
                cp = float(score.score(mate_score=eval_clip))
            evals[i] = float(np.clip(cp, -eval_clip, eval_clip))
            if log_every and (i + 1) % log_every == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-9)
                print(f"  annotated {i + 1}/{len(fens)} positions ({rate:.1f} pos/s)")
    return evals


# ---------------- Corpus builder ----------------

def build_corpus(
    out_path: Path,
    n_positions: int,
    *,
    depth: int = 20,
    plies_range: tuple[int, int] = (8, 60),
    stockfish_path: Optional[str] = None,
    seed: int = 0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Generate and annotate n_positions; write to parquet at out_path.

    Returns the DataFrame. If out_path exists and overwrite=False, loads from
    cache and returns immediately.
    """
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"Loading cached corpus from {out_path}")
        return pd.read_parquet(out_path)

    print(f"Generating {n_positions} random positions (seed={seed}) ...")
    fens = generate_random_fens(n_positions, plies_range=plies_range, seed=seed)
    print(f"Annotating with Stockfish depth={depth} ...")
    evals = annotate_with_stockfish(fens, stockfish_path=stockfish_path, depth=depth)

    df = pd.DataFrame({
        "fen": fens,
        "eval_cp": evals,
        "depth": depth,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} positions to {out_path}")
    return df


# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build chess corpus for SR distillation")
    parser.add_argument("--out", type=str, default="data/chess/stockfish_eval.parquet")
    parser.add_argument("--n", type=int, default=1000, help="number of positions")
    parser.add_argument("--depth", type=int, default=20, help="stockfish search depth")
    parser.add_argument("--stockfish", type=str, default=None,
                        help="path to stockfish binary (else uses STOCKFISH_PATH / PATH)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    df = build_corpus(
        Path(args.out),
        n_positions=args.n,
        depth=args.depth,
        stockfish_path=args.stockfish,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(df.head())
    print(df["eval_cp"].describe())
