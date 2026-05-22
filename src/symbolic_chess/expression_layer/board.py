"""Chess board representation + spatial operators for expression_layer.

Parallel to `spatial.py` (1D PDE fields). Encodes chess positions as
(12, 8, 8) plane stacks (12 = {white, black} x {P, N, B, R, Q, K}), then
provides aggregation operators (file/rank/diagonal sums, king-zone counts)
that produce per-position scalar features for symbolic regression.

Design doc: docs/benchmark_stockfish_distillation.md

Workflow
--------
  1. encode_corpus(fens) → planes (N, 12, 8, 8)
  2. chess_feature_bank(planes, fens) → dict[name, (N,) array]
  3. Wrap each entry as a Variable; hand to PySR / expression_layer
  4. PySR sees i.i.d. positions, not a time series

No engine (Stockfish) dependency — only python-chess for FEN parsing and
mobility. Heavy computations (mobility, passed_pawns) loop per position;
acceptable at the ~50k-position scale this benchmark targets.
"""
from __future__ import annotations
import numpy as np
import chess


# ---------------- Channel index conventions ----------------

CHANNEL_NAMES = [
    "WP", "WN", "WB", "WR", "WQ", "WK",
    "BP", "BN", "BB", "BR", "BQ", "BK",
]
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11

PIECE_TYPE_TO_OFFSET = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}


# ---------------- Encoding ----------------

def encode_position(fen: str) -> np.ndarray:
    """Encode a FEN string to (12, 8, 8) int8 planes.

    planes[c, r, f] = 1 iff a piece of channel c sits on rank r, file f.
    Rank/file follow python-chess conventions: rank 0 = '1', file 0 = 'a'.
    """
    board = chess.Board(fen)
    planes = np.zeros((12, 8, 8), dtype=np.int8)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        offset = PIECE_TYPE_TO_OFFSET[piece.piece_type]
        channel = offset if piece.color == chess.WHITE else 6 + offset
        planes[channel, rank, file] = 1
    return planes


def encode_corpus(fens: list[str]) -> np.ndarray:
    """Encode a list of FENs to (N, 12, 8, 8) int8 planes."""
    out = np.zeros((len(fens), 12, 8, 8), dtype=np.int8)
    for i, fen in enumerate(fens):
        out[i] = encode_position(fen)
    return out


# ---------------- Aggregations: per-channel scalar / vector features ----------------

def piece_count(planes: np.ndarray, channel: int) -> np.ndarray:
    """(N, 12, 8, 8) → (N,) — count of pieces in given channel per position."""
    return planes[:, channel].sum(axis=(1, 2)).astype(np.float64)


def file_sum(planes: np.ndarray, channel: int) -> np.ndarray:
    """(N, 12, 8, 8) → (N, 8) — count per file (sum over ranks)."""
    return planes[:, channel].sum(axis=1).astype(np.float64)


def rank_sum(planes: np.ndarray, channel: int) -> np.ndarray:
    """(N, 12, 8, 8) → (N, 8) — count per rank (sum over files)."""
    return planes[:, channel].sum(axis=2).astype(np.float64)


def diagonal_sum_a1h8(planes: np.ndarray, channel: int) -> np.ndarray:
    """Sums along a1–h8-direction diagonals (file - rank constant).

    Returns (N, 15): index d ∈ [0, 14] corresponds to file - rank + 7.
    """
    N = planes.shape[0]
    out = np.zeros((N, 15), dtype=np.float64)
    plane = planes[:, channel]
    for r in range(8):
        for f in range(8):
            out[:, f - r + 7] += plane[:, r, f]
    return out


def diagonal_sum_a8h1(planes: np.ndarray, channel: int) -> np.ndarray:
    """Sums along a8–h1-direction diagonals (file + rank constant).

    Returns (N, 15): index d ∈ [0, 14] corresponds to file + rank.
    """
    N = planes.shape[0]
    out = np.zeros((N, 15), dtype=np.float64)
    plane = planes[:, channel]
    for r in range(8):
        for f in range(8):
            out[:, f + r] += plane[:, r, f]
    return out


CENTRAL_SQUARES = ((3, 3), (3, 4), (4, 3), (4, 4))   # d4, e4, d5, e5


def central_control(planes: np.ndarray, channel: int) -> np.ndarray:
    """Count of channel's pieces on the 4 central squares (d4/e4/d5/e5)."""
    s = np.zeros(planes.shape[0], dtype=np.float64)
    for r, f in CENTRAL_SQUARES:
        s += planes[:, channel, r, f]
    return s


def king_zone_count(planes: np.ndarray, target_channel: int, king_color: int) -> np.ndarray:
    """Count pieces of target_channel within one square of king_color's king.

    king_color: 0 for white (uses WK), 1 for black (uses BK). The "zone" is
    the 3×3 square centred on the king (inclusive), clipped at board edges.

    Vectorized: 3×3 dilation of the king plane (padded sliding sum) gives
    the zone mask; multiply by target and sum.
    """
    king_channel = WK if king_color == 0 else BK
    target = planes[:, target_channel].astype(np.float64)
    king = planes[:, king_channel].astype(np.float64)
    # Pad with zeros, then sum over the nine 3×3 offsets → dilated king mask
    padded = np.pad(king, ((0, 0), (1, 1), (1, 1)), mode="constant")
    zone = (
        padded[:, 0:8, 0:8] + padded[:, 0:8, 1:9] + padded[:, 0:8, 2:10]
        + padded[:, 1:9, 0:8] + padded[:, 1:9, 1:9] + padded[:, 1:9, 2:10]
        + padded[:, 2:10, 0:8] + padded[:, 2:10, 1:9] + padded[:, 2:10, 2:10]
    )
    zone = np.minimum(zone, 1.0)   # collapse to binary mask
    return (target * zone).sum(axis=(1, 2))


# ---------------- Pawn-structure features ----------------

def doubled_pawns_count(planes: np.ndarray, channel: int) -> np.ndarray:
    """Sum of max(0, file_count - 1) across all 8 files. channel should be WP or BP."""
    files = file_sum(planes, channel)
    return np.maximum(files - 1, 0).sum(axis=1)


def passed_pawns_count(planes: np.ndarray, color: int) -> np.ndarray:
    """Number of passed pawns for given color (0=white, 1=black).

    A pawn is passed iff no enemy pawn sits on the same or adjacent file
    on any rank strictly ahead of it.

    Vectorized via:
      file_threat[r, f] = any enemy pawn on rank r, file in {f-1, f, f+1}
      ahead_threat[r, f] = OR of file_threat over ranks strictly ahead
    A square is "passed" iff own pawn occupies it AND no ahead_threat.
    """
    own_ch = WP if color == 0 else BP
    enemy_ch = BP if color == 0 else WP
    own = planes[:, own_ch].astype(bool)
    enemy = planes[:, enemy_ch].astype(bool)

    # file_threat[n, r, f] = enemy on (r, f-1) or (r, f) or (r, f+1)
    enemy_pad = np.pad(enemy, ((0, 0), (0, 0), (1, 1)), mode="constant")
    file_threat = enemy_pad[:, :, 0:8] | enemy_pad[:, :, 1:9] | enemy_pad[:, :, 2:10]

    if color == 0:
        # White: "ahead" = higher ranks. cum_or[r] = OR over ranks ≥ r → flip,
        # accumulate, flip back; ahead_threat[r] = cum_or[r+1].
        cum_or = np.logical_or.accumulate(file_threat[:, ::-1, :], axis=1)[:, ::-1, :]
        ahead_threat = np.empty_like(cum_or)
        ahead_threat[:, :7, :] = cum_or[:, 1:8, :]
        ahead_threat[:, 7, :] = False
    else:
        # Black: "ahead" = lower ranks. cum_or[r] = OR over ranks ≤ r;
        # ahead_threat[r] = cum_or[r-1].
        cum_or = np.logical_or.accumulate(file_threat, axis=1)
        ahead_threat = np.empty_like(cum_or)
        ahead_threat[:, 1:, :] = cum_or[:, :7, :]
        ahead_threat[:, 0, :] = False

    passed_squares = own & ~ahead_threat
    return passed_squares.sum(axis=(1, 2)).astype(np.float64)


# ---------------- Game phase ----------------

def game_phase(planes: np.ndarray) -> np.ndarray:
    """Phase ∈ [0, 1]: 0 = endgame (only kings ± pawns), 1 = full non-pawn material.

    Counts non-pawn pieces (max 14 across both sides: 2N+2B+2R+1Q per side).
    """
    nonpawn = np.zeros(planes.shape[0], dtype=np.float64)
    for ch in (WN, WB, WR, WQ, BN, BB, BR, BQ):
        nonpawn += planes[:, ch].sum(axis=(1, 2))
    return np.clip(nonpawn / 14.0, 0.0, 1.0)


# ---------------- Mobility (python-chess dependent) ----------------

def pseudo_legal_move_count(fens: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (W_mobility, B_mobility), one count per position.

    Computed by swapping side-to-move; both sides' move counts are extracted
    from the same position. Castling rights / en-passant rights ignored
    (cheap-and-cheerful mobility proxy).
    """
    N = len(fens)
    w = np.zeros(N, dtype=np.float64)
    b = np.zeros(N, dtype=np.float64)
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        original_turn = board.turn
        board.turn = chess.WHITE
        w[i] = board.pseudo_legal_moves.count()
        board.turn = chess.BLACK
        b[i] = board.pseudo_legal_moves.count()
        board.turn = original_turn
    return w, b


# ---------------- Feature bank ----------------

def chess_feature_bank(
    planes: np.ndarray,
    fens: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build a per-position feature dict for SR consumption.

    Returns dict[name → (N,) np.float64].

    If `fens` is provided, mobility features (W_mobility, B_mobility) are
    included. Otherwise they are omitted.
    """
    bank: dict[str, np.ndarray] = {}

    # Material counts (10 vars)
    for label, ch in (("WP", WP), ("WN", WN), ("WB", WB), ("WR", WR), ("WQ", WQ)):
        bank[f"{label}_count"] = piece_count(planes, ch)
    for label, ch in (("BP", BP), ("BN", BN), ("BB", BB), ("BR", BR), ("BQ", BQ)):
        bank[f"{label}_count"] = piece_count(planes, ch)

    # Material net (helpful aggregates — lets SR find balance directly)
    bank["material_net"] = (
        1.0 * (bank["WP_count"] - bank["BP_count"])
        + 3.0 * (bank["WN_count"] - bank["BN_count"])
        + 3.0 * (bank["WB_count"] - bank["BB_count"])
        + 5.0 * (bank["WR_count"] - bank["BR_count"])
        + 9.0 * (bank["WQ_count"] - bank["BQ_count"])
    )

    # Pawn structure
    bank["W_doubled_pawns"] = doubled_pawns_count(planes, WP)
    bank["B_doubled_pawns"] = doubled_pawns_count(planes, BP)
    bank["W_passed_pawns"] = passed_pawns_count(planes, 0)
    bank["B_passed_pawns"] = passed_pawns_count(planes, 1)

    # Central control (minor pieces + pawns by side, summed across central squares)
    for label, ch in (("WP", WP), ("WN", WN), ("WB", WB),
                       ("BP", BP), ("BN", BN), ("BB", BB)):
        bank[f"central_{label}"] = central_control(planes, ch)

    # King safety — enemy pieces in king zone
    bank["W_king_zone_enemy"] = (
        king_zone_count(planes, BN, 0) + king_zone_count(planes, BB, 0)
        + king_zone_count(planes, BR, 0) + king_zone_count(planes, BQ, 0)
    )
    bank["B_king_zone_enemy"] = (
        king_zone_count(planes, WN, 1) + king_zone_count(planes, WB, 1)
        + king_zone_count(planes, WR, 1) + king_zone_count(planes, WQ, 1)
    )
    bank["W_king_zone_own_pawns"] = king_zone_count(planes, WP, 0)
    bank["B_king_zone_own_pawns"] = king_zone_count(planes, BP, 1)

    # Pawn-file imbalance (rough proxy for asymmetric pawn structures)
    w_files = file_sum(planes, WP)
    b_files = file_sum(planes, BP)
    bank["pawn_file_imbalance"] = np.abs(w_files - b_files).sum(axis=1)

    # Phase
    bank["phase"] = game_phase(planes)

    # Mobility — only if fens supplied
    if fens is not None:
        w_mob, b_mob = pseudo_legal_move_count(fens)
        bank["W_mobility"] = w_mob
        bank["B_mobility"] = b_mob

    return bank


__all__ = [
    "CHANNEL_NAMES",
    "WP", "WN", "WB", "WR", "WQ", "WK",
    "BP", "BN", "BB", "BR", "BQ", "BK",
    "encode_position", "encode_corpus",
    "piece_count", "file_sum", "rank_sum",
    "diagonal_sum_a1h8", "diagonal_sum_a8h1",
    "central_control", "king_zone_count",
    "doubled_pawns_count", "passed_pawns_count", "game_phase",
    "pseudo_legal_move_count",
    "chess_feature_bank",
]
