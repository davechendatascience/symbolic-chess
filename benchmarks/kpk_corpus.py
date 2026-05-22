"""K+P-vs-K (KPK) corpus generation + feature extraction.

Generates legal KPK positions (WK + WP + BK + side-to-move), labels each
with Stockfish at depth 20 as binary "white wins eventually", and caches to
parquet. Feature extractor exposes piece coordinates and Chebyshev distances
sufficient for SR to express rule-of-the-square / key-squares logic.

Design doc: docs/kpk_equilibrium_distillation.md

WP file is canonicalised to a-d (files 0-3); mirror-symmetric positions on
files e-h are equivalent. WP rank ∈ [1, 6] (ranks 2-7 in chess notation).
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import chess
import chess.engine

# Reuse stockfish discovery from the benchmarks helper
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from chess_corpus import find_stockfish   # type: ignore


# ---------------- Position sampling ----------------

def _is_legal_kpk(wk_sq: int, wp_sq: int, bk_sq: int, wtm: bool) -> bool:
    """Check legality: no overlaps, kings not adjacent, pawn rank 2-7, valid for the side to move."""
    squares = {wk_sq, wp_sq, bk_sq}
    if len(squares) != 3:
        return False
    if chess.square_rank(wp_sq) in (0, 7):
        return False
    if max(abs(chess.square_rank(wk_sq) - chess.square_rank(bk_sq)),
           abs(chess.square_file(wk_sq) - chess.square_file(bk_sq))) <= 1:
        return False
    board = chess.Board(None)
    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(wp_sq, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE if wtm else chess.BLACK
    board.castling_rights = chess.BB_EMPTY
    return board.is_valid()


def sample_kpk_positions(n: int, seed: int = 0) -> list[dict]:
    """Sample n unique legal KPK positions.

    WP canonicalised to files a-d (files 0-3) to remove mirror symmetry.
    """
    rng = np.random.default_rng(seed)
    seen: set[tuple] = set()
    out: list[dict] = []
    attempts = 0
    max_attempts = n * 50
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        wp_file = int(rng.integers(0, 4))      # a-d only
        wp_rank = int(rng.integers(1, 7))      # ranks 2-7
        wk_sq = int(rng.integers(0, 64))
        bk_sq = int(rng.integers(0, 64))
        wtm = bool(rng.integers(0, 2))
        wp_sq = chess.square(wp_file, wp_rank)
        key = (wk_sq, wp_sq, bk_sq, wtm)
        if key in seen:
            continue
        if not _is_legal_kpk(wk_sq, wp_sq, bk_sq, wtm):
            continue
        seen.add(key)
        out.append({"wk_sq": wk_sq, "wp_sq": wp_sq, "bk_sq": bk_sq, "wtm": wtm})
    if len(out) < n:
        print(f"WARNING: only {len(out)}/{n} positions sampled after {attempts} attempts")
    return out


def position_to_fen(wk_sq: int, wp_sq: int, bk_sq: int, wtm: bool) -> str:
    board = chess.Board(None)
    board.set_piece_at(wk_sq, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(wp_sq, chess.Piece(chess.PAWN, chess.WHITE))
    board.set_piece_at(bk_sq, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE if wtm else chess.BLACK
    board.castling_rights = chess.BB_EMPTY
    return board.fen()


# ---------------- Feature extraction ----------------

def _cheby(a_rank: int, a_file: int, b_rank: int, b_file: int) -> int:
    return max(abs(a_rank - b_rank), abs(a_file - b_file))


def kpk_features(wk_sq: int, wp_sq: int, bk_sq: int, wtm: bool) -> dict:
    """Return per-position scalar features for SR consumption."""
    wk_r, wk_f = chess.square_rank(wk_sq), chess.square_file(wk_sq)
    wp_r, wp_f = chess.square_rank(wp_sq), chess.square_file(wp_sq)
    bk_r, bk_f = chess.square_rank(bk_sq), chess.square_file(bk_sq)
    return {
        "wk_rank": float(wk_r), "wk_file": float(wk_f),
        "wp_rank": float(wp_r), "wp_file": float(wp_f),
        "bk_rank": float(bk_r), "bk_file": float(bk_f),
        "wtm": 1.0 if wtm else 0.0,
        "promo_dist": float(7 - wp_r),
        "d_kp": float(_cheby(wk_r, wk_f, wp_r, wp_f)),
        "d_bk_p": float(_cheby(bk_r, bk_f, wp_r, wp_f)),
        "d_bk_promo": float(_cheby(bk_r, bk_f, 7, wp_f)),
        "d_kk": float(_cheby(wk_r, wk_f, bk_r, bk_f)),
        "wp_file_to_edge": float(min(wp_f, 7 - wp_f)),
    }


FEATURE_NAMES = [
    "wk_rank", "wk_file", "wp_rank", "wp_file", "bk_rank", "bk_file", "wtm",
    "promo_dist", "d_kp", "d_bk_p", "d_bk_promo", "d_kk", "wp_file_to_edge",
]


# ---------------- Stockfish labelling ----------------

def label_with_stockfish(
    positions: list[dict],
    stockfish_path: Optional[str] = None,
    depth: int = 20,
    time_limit: Optional[float] = None,
    cp_win_threshold: int = 200,
    log_every: int = 50,
) -> np.ndarray:
    """Binary win/draw labels from Stockfish.

    Search budget: depth=`depth` OR time=`time_limit` seconds per position
    (Limit applies whichever ends first). For KPK at d=15-20 most positions
    return mate scores in <100ms; time_limit caps the worst cases.

    label = 1 iff Stockfish reports a mate score for white OR cp >= threshold.
    label = 0 otherwise.
    """
    binary = find_stockfish(stockfish_path)
    limit = chess.engine.Limit(depth=depth, time=time_limit)
    out = np.zeros(len(positions), dtype=np.float64)
    t0 = time.time()
    with chess.engine.SimpleEngine.popen_uci(binary) as engine:
        for i, pos in enumerate(positions):
            fen = position_to_fen(pos["wk_sq"], pos["wp_sq"], pos["bk_sq"], pos["wtm"])
            board = chess.Board(fen)
            info = engine.analyse(board, limit)
            score = info["score"].white()
            if score.is_mate():
                out[i] = 1.0 if score.mate() > 0 else 0.0
            else:
                cp = score.score()
                out[i] = 1.0 if cp is not None and cp >= cp_win_threshold else 0.0
            if log_every and (i + 1) % log_every == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-9)
                eta = (len(positions) - i - 1) / rate
                print(f"  labelled {i + 1}/{len(positions)} ({rate:.1f} pos/s, ETA {eta:.0f}s)",
                      flush=True)
    return out


# ---------------- Corpus builder ----------------

def build_kpk_corpus(
    out_path: Path | str,
    n_positions: int,
    *,
    depth: int = 20,
    time_limit: Optional[float] = None,
    stockfish_path: Optional[str] = None,
    seed: int = 0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Sample, feature-extract, label, cache. Loads from cache if exists."""
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        print(f"Loading cached KPK corpus from {out_path}")
        return pd.read_parquet(out_path)

    print(f"Sampling {n_positions} legal KPK positions (seed={seed}) ...", flush=True)
    positions = sample_kpk_positions(n_positions, seed=seed)
    print(f"  got {len(positions)} unique positions", flush=True)

    print(f"Computing features ...", flush=True)
    feature_rows = [kpk_features(**{k: p[k] for k in ("wk_sq", "wp_sq", "bk_sq", "wtm")})
                    for p in positions]
    feat_df = pd.DataFrame(feature_rows)

    print(f"Labelling with Stockfish (depth={depth}, time_limit={time_limit}) ...", flush=True)
    labels = label_with_stockfish(
        positions, stockfish_path=stockfish_path,
        depth=depth, time_limit=time_limit,
    )

    fens = [position_to_fen(p["wk_sq"], p["wp_sq"], p["bk_sq"], p["wtm"])
            for p in positions]
    pos_df = pd.DataFrame({
        "fen": fens,
        "wk_sq": [p["wk_sq"] for p in positions],
        "wp_sq": [p["wp_sq"] for p in positions],
        "bk_sq": [p["bk_sq"] for p in positions],
        "label": labels,
    })   # wtm omitted here; feat_df carries the float version
    df = pd.concat([pos_df, feat_df], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    n_wins = int((labels == 1.0).sum())
    print(f"Wrote {len(df)} positions to {out_path} | "
          f"white_wins={n_wins} ({100*n_wins/len(df):.1f}%), "
          f"draws={len(df) - n_wins}")
    return df


# ---------------- CLI ----------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="KPK corpus builder")
    parser.add_argument("--out", type=str,
                        default="data/chess/kpk_corpus.parquet")
    parser.add_argument("--n", type=int, default=5000, help="positions to sample")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    df = build_kpk_corpus(
        Path(args.out),
        n_positions=args.n,
        depth=args.depth,
        stockfish_path=args.stockfish,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(df.head())
    print(df["label"].value_counts(normalize=True))
