"""Tests for the chess board representation + spatial features.

Design doc: docs/benchmark_stockfish_distillation.md
"""
import numpy as np
import pytest

import chess

from symbolic_chess.expression_layer.board import (
    encode_position, encode_corpus,
    piece_count, file_sum, rank_sum,
    diagonal_sum_a1h8, diagonal_sum_a8h1,
    central_control, king_zone_count,
    doubled_pawns_count, passed_pawns_count, game_phase,
    pseudo_legal_move_count, chess_feature_bank,
    WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
    CHANNEL_NAMES,
)


STARTING_FEN = chess.STARTING_FEN
EMPTY_KINGS_ONLY = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
W_PASSED_PAWN_FEN = "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1"      # white d5 pawn, black king e8 — passed
DOUBLED_PAWN_FEN = "4k3/8/8/8/3P4/3P4/8/4K3 w - - 0 1"     # two white pawns on d-file


# ---------------- Encoding sanity ----------------

def test_starting_position_piece_counts():
    planes = encode_position(STARTING_FEN)
    assert planes.shape == (12, 8, 8)
    assert planes[WP].sum() == 8
    assert planes[BP].sum() == 8
    assert planes[WN].sum() == 2
    assert planes[BR].sum() == 2
    assert planes[WK].sum() == 1
    assert planes[BK].sum() == 1


def test_starting_position_white_king_at_e1():
    planes = encode_position(STARTING_FEN)
    # rank 0 (= '1'), file 4 (= 'e')
    assert planes[WK, 0, 4] == 1
    assert planes[WK].sum() == 1


def test_starting_position_black_king_at_e8():
    planes = encode_position(STARTING_FEN)
    assert planes[BK, 7, 4] == 1


def test_empty_position_has_only_kings():
    planes = encode_position(EMPTY_KINGS_ONLY)
    for ch in range(12):
        if ch in (WK, BK):
            assert planes[ch].sum() == 1
        else:
            assert planes[ch].sum() == 0


def test_encode_corpus_shape():
    fens = [STARTING_FEN, EMPTY_KINGS_ONLY, W_PASSED_PAWN_FEN]
    planes = encode_corpus(fens)
    assert planes.shape == (3, 12, 8, 8)
    assert planes[0, WP].sum() == 8
    assert planes[1, WP].sum() == 0
    assert planes[2, WP].sum() == 1


# ---------------- Aggregation operators ----------------

def test_piece_count_returns_per_position_scalar():
    planes = encode_corpus([STARTING_FEN, EMPTY_KINGS_ONLY])
    out = piece_count(planes, WP)
    assert out.shape == (2,)
    assert out[0] == 8.0
    assert out[1] == 0.0


def test_file_sum_starting_pawns_are_one_per_file():
    planes = encode_corpus([STARTING_FEN])
    files = file_sum(planes, WP)
    assert files.shape == (1, 8)
    # Each file has exactly one pawn in starting position
    assert np.allclose(files[0], np.ones(8))


def test_rank_sum_starting_pawns_concentrated_on_rank_1():
    planes = encode_corpus([STARTING_FEN])
    ranks = rank_sum(planes, WP)
    assert ranks.shape == (1, 8)
    # All white pawns on rank 1 (index 1)
    assert ranks[0, 1] == 8.0
    for r in (0, 2, 3, 4, 5, 6, 7):
        assert ranks[0, r] == 0.0


def test_diagonal_sums_shape():
    planes = encode_corpus([STARTING_FEN])
    d1 = diagonal_sum_a1h8(planes, WP)
    d2 = diagonal_sum_a8h1(planes, WP)
    assert d1.shape == (1, 15)
    assert d2.shape == (1, 15)
    # Sum across diagonals must equal total piece count
    assert d1[0].sum() == 8.0
    assert d2[0].sum() == 8.0


def test_central_control_starting_position_is_empty():
    planes = encode_corpus([STARTING_FEN])
    # No central pawns/pieces in starting position
    assert central_control(planes, WP)[0] == 0.0
    assert central_control(planes, WN)[0] == 0.0


def test_central_control_after_e4():
    # White plays e4: pawn on e4 (rank 3, file 4)
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    planes = encode_corpus([fen])
    assert central_control(planes, WP)[0] == 1.0   # e4 is one of the central squares


def test_king_zone_count_white_king_starting():
    planes = encode_corpus([STARTING_FEN])
    # White king on e1 (rank 0, file 4); zone = ranks 0-1, files 3-5
    # White pawns in that zone: d2, e2, f2 — three pawns
    count_own_pawns = king_zone_count(planes, WP, 0)
    assert count_own_pawns[0] == 3.0


def test_king_zone_count_empty_when_no_king_pieces():
    planes = encode_corpus([EMPTY_KINGS_ONLY])
    assert king_zone_count(planes, WP, 0)[0] == 0.0


# ---------------- Pawn-structure features ----------------

def test_doubled_pawns_none_in_starting_position():
    planes = encode_corpus([STARTING_FEN])
    assert doubled_pawns_count(planes, WP)[0] == 0.0


def test_doubled_pawns_detected():
    planes = encode_corpus([DOUBLED_PAWN_FEN])
    # Two white pawns on d-file: max(0, 2 - 1) = 1
    assert doubled_pawns_count(planes, WP)[0] == 1.0


def test_passed_pawns_starting_position_is_zero():
    planes = encode_corpus([STARTING_FEN])
    # No passed pawns at start (full enemy pawn rank ahead)
    assert passed_pawns_count(planes, 0)[0] == 0.0
    assert passed_pawns_count(planes, 1)[0] == 0.0


def test_passed_pawn_detected():
    planes = encode_corpus([W_PASSED_PAWN_FEN])
    # White d5 pawn, no black pawns on c/d/e files ahead → passed
    assert passed_pawns_count(planes, 0)[0] == 1.0
    assert passed_pawns_count(planes, 1)[0] == 0.0


# ---------------- Game phase ----------------

def test_game_phase_starting_is_one():
    planes = encode_corpus([STARTING_FEN])
    # 2N+2B+2R+1Q per side = 14 non-pawn → phase 1.0
    assert game_phase(planes)[0] == pytest.approx(1.0)


def test_game_phase_kings_only_is_zero():
    planes = encode_corpus([EMPTY_KINGS_ONLY])
    assert game_phase(planes)[0] == 0.0


# ---------------- Mobility ----------------

def test_mobility_starting_position():
    fens = [STARTING_FEN]
    w_mob, b_mob = pseudo_legal_move_count(fens)
    # 20 legal first moves per side in starting position
    assert w_mob[0] == 20
    assert b_mob[0] == 20


# ---------------- Feature bank ----------------

def test_feature_bank_returns_expected_keys():
    planes = encode_corpus([STARTING_FEN, EMPTY_KINGS_ONLY])
    bank = chess_feature_bank(planes, fens=[STARTING_FEN, EMPTY_KINGS_ONLY])
    expected = {
        "WP_count", "WN_count", "WB_count", "WR_count", "WQ_count",
        "BP_count", "BN_count", "BB_count", "BR_count", "BQ_count",
        "material_net",
        "W_doubled_pawns", "B_doubled_pawns",
        "W_passed_pawns", "B_passed_pawns",
        "W_king_zone_enemy", "B_king_zone_enemy",
        "W_king_zone_own_pawns", "B_king_zone_own_pawns",
        "pawn_file_imbalance", "phase",
        "W_mobility", "B_mobility",
    }
    missing = expected - set(bank.keys())
    assert not missing, f"missing keys: {missing}"


def test_feature_bank_shapes_and_finite():
    fens = [STARTING_FEN, EMPTY_KINGS_ONLY, W_PASSED_PAWN_FEN, DOUBLED_PAWN_FEN]
    planes = encode_corpus(fens)
    bank = chess_feature_bank(planes, fens=fens)
    for name, arr in bank.items():
        assert arr.shape == (4,), f"{name} has shape {arr.shape}, expected (4,)"
        assert np.isfinite(arr).all(), f"{name} contains non-finite values"


def test_feature_bank_without_mobility_omits_mobility_keys():
    planes = encode_corpus([STARTING_FEN])
    bank = chess_feature_bank(planes, fens=None)
    assert "W_mobility" not in bank
    assert "B_mobility" not in bank


def test_material_net_starting_is_zero():
    planes = encode_corpus([STARTING_FEN])
    bank = chess_feature_bank(planes)
    assert bank["material_net"][0] == 0.0


def test_material_net_extra_white_queen():
    fen_extra_queen = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
    planes = encode_corpus([fen_extra_queen])
    bank = chess_feature_bank(planes)
    assert bank["material_net"][0] == 9.0
