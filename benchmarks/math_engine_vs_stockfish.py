"""Math-engine vs Stockfish tournament harness.

Plugs the cx=13 PySR-discovered eval (from `pysr_chess_stockfish.md`) into the
chess_engine negamax + alpha-beta search and plays a tournament against
Stockfish at several depths. Reports win/loss/draw counts and an Elo estimate.

The SR-discovered eval:

    eval = (material_net / 0.01460058)
         - ((W_mobility * central_BP) /
            ((1.3168311 - B_king_zone_own_pawns) - central_BB))

This is "iteration 0" of the self-improvement loop — the eval distilled from
500-position Stockfish d=15 labels at TEST R² = 0.391. Future iterations
should improve via Architecture C (self-play training + Stockfish-only-for-
measurement).
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

from symbolic_chess.expression_layer import Variable, Constant, add, sub, mul, div, compile_expr
from symbolic_chess.chess_engine import best_move, make_expr_evaluator, material_eval

from chess_corpus import find_stockfish  # type: ignore


# Material values matching board.py
_MATERIAL_PAWN = 1.0
_MATERIAL_KNIGHT = 3.0
_MATERIAL_BISHOP = 3.0
_MATERIAL_ROOK = 5.0
_MATERIAL_QUEEN = 9.0
# Central squares as (rank, file) — matches board.py CENTRAL_SQUARES
_CENTRAL_SQ_INDICES = [chess.D4, chess.E4, chess.D5, chess.E5]


OUT_REPORT = ROOT / "benchmarks" / "results" / "math_engine_vs_stockfish.md"


# ---------------- SR-discovered eval (cx=13 from pysr_chess_stockfish.md) ----------------

def sr_eval_expr_cx13():
    """Construct the cx=13 PySR Pareto equation as an Expr tree."""
    # Need any-length placeholder data for Variable construction
    placeholder = np.array([0.0])
    material_net = Variable("material_net", placeholder)
    W_mobility = Variable("W_mobility", placeholder)
    central_BP = Variable("central_BP", placeholder)
    B_king_zone_own_pawns = Variable("B_king_zone_own_pawns", placeholder)
    central_BB = Variable("central_BB", placeholder)

    # (material_net / 0.01460058) - ((W_mobility * central_BP) /
    #  ((1.3168311 - B_king_zone_own_pawns) - central_BB))
    term1 = div(material_net, Constant(0.01460058))
    inner_denom = sub(sub(Constant(1.3168311), B_king_zone_own_pawns), central_BB)
    term2 = div(mul(W_mobility, central_BP), inner_denom)
    return sub(term1, term2), [
        "material_net", "W_mobility", "central_BP",
        "B_king_zone_own_pawns", "central_BB",
    ]


def make_fast_eval_cx13():
    """Specialised fast eval for the cx=13 expression.

    Computes only the 5 features it actually uses, directly from chess.Board.
    Skips encode_position + full chess_feature_bank (300us → ~30us per call).
    """
    expr, names = sr_eval_expr_cx13()
    compiled = compile_expr(expr, names)
    # Indices into the feature array, matching `names` order
    IDX_MATERIAL = 0
    IDX_W_MOBILITY = 1
    IDX_CENTRAL_BP = 2
    IDX_B_KING_ZONE_OWN_PAWNS = 3
    IDX_CENTRAL_BB = 4
    feats = np.zeros(5, dtype=np.float64)

    def extract(board: chess.Board) -> float:
        # 1. material_net — iterate piece_map once
        material = 0.0
        central_bp = 0
        central_bb = 0
        for sq, piece in board.piece_map().items():
            pt = piece.piece_type
            if pt == chess.PAWN:
                v = _MATERIAL_PAWN
            elif pt == chess.KNIGHT:
                v = _MATERIAL_KNIGHT
            elif pt == chess.BISHOP:
                v = _MATERIAL_BISHOP
            elif pt == chess.ROOK:
                v = _MATERIAL_ROOK
            elif pt == chess.QUEEN:
                v = _MATERIAL_QUEEN
            else:
                continue   # king ignored for material
            if piece.color == chess.WHITE:
                material += v
            else:
                material -= v
            # central counts: black pawn / black bishop on central square
            if sq in _CENTRAL_SQ_INDICES and piece.color == chess.BLACK:
                if pt == chess.PAWN:
                    central_bp += 1
                elif pt == chess.BISHOP:
                    central_bb += 1

        # 2. W_mobility — toggle turn, count pseudo-legal moves, restore
        orig_turn = board.turn
        board.turn = chess.WHITE
        w_mobility = float(board.pseudo_legal_moves.count())
        board.turn = orig_turn

        # 3. B_king_zone_own_pawns — black king square, count BP within 1 sq
        bk_sq = board.king(chess.BLACK)
        if bk_sq is None:
            b_king_zone_own_pawns = 0.0
        else:
            bk_r = chess.square_rank(bk_sq)
            bk_f = chess.square_file(bk_sq)
            count = 0
            for dr in (-1, 0, 1):
                for df in (-1, 0, 1):
                    r, f = bk_r + dr, bk_f + df
                    if 0 <= r < 8 and 0 <= f < 8:
                        sq = chess.square(f, r)
                        p = board.piece_at(sq)
                        if p and p.color == chess.BLACK and p.piece_type == chess.PAWN:
                            count += 1
            b_king_zone_own_pawns = float(count)

        feats[IDX_MATERIAL] = material
        feats[IDX_W_MOBILITY] = w_mobility
        feats[IDX_CENTRAL_BP] = float(central_bp)
        feats[IDX_B_KING_ZONE_OWN_PAWNS] = b_king_zone_own_pawns
        feats[IDX_CENTRAL_BB] = float(central_bb)
        return float(compiled(feats))

    return extract


# ---------------- Game play ----------------

def play_game(
    white_move_fn, black_move_fn,
    max_plies: int = 200,
    random_plies: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[str, int]:
    """Play one game. Each move_fn takes a Board and returns a Move.

    `random_plies` opening moves are uniform random over legal moves before
    the engines take over. Needed to break determinism — without it, every
    deterministic engine pair plays the same game forever.

    Returns (result_str, n_plies). result_str ∈ {"1-0", "0-1", "1/2-1/2", "*"}.
    """
    board = chess.Board()
    plies = 0
    if random_plies and rng is None:
        rng = np.random.default_rng()
    while plies < random_plies and not board.is_game_over(claim_draw=True):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(legal[int(rng.integers(0, len(legal)))])
        plies += 1
    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        mover = white_move_fn if board.turn == chess.WHITE else black_move_fn
        mv = mover(board)
        if mv is None:
            break
        board.push(mv)
        plies += 1
    return board.result(claim_draw=True), plies


def make_math_engine_mover(eval_fn, search_depth: int):
    """Wrap our eval into a move-producing function for play_game."""
    def mover(board: chess.Board) -> chess.Move | None:
        return best_move(board, depth=search_depth, eval_fn=eval_fn)
    return mover


def make_stockfish_mover(engine: chess.engine.SimpleEngine, depth: int):
    """Wrap Stockfish into a move-producing function."""
    def mover(board: chess.Board) -> chess.Move | None:
        result = engine.play(board, chess.engine.Limit(depth=depth))
        return result.move
    return mover


# ---------------- Elo math ----------------

def score_to_elo_diff(score: float, n: int) -> tuple[float, float, float]:
    """Convert win-rate `score` (∈ [0, 1]) over n games to (Elo_diff, low_ci, high_ci).

    Elo diff = -400 * log10(1/score - 1).
    Wilson-style 95% CI on the win-rate, then converted to Elo bounds.
    Score 0 → -400 (lower cap), score 1 → +400 (upper cap).
    """
    def elo(s: float) -> float:
        if s <= 0.0: return -400.0
        if s >= 1.0: return 400.0
        return -400.0 * math.log10(1.0 / s - 1.0)

    # Wilson 95% CI on a binomial proportion (approximation; treats draws as
    # half-wins for the point estimate, then standard binomial CI on score)
    z = 1.96
    if n <= 0:
        return elo(score), -400.0, 400.0
    denom = 1.0 + z * z / n
    centre = (score + z * z / (2 * n)) / denom
    margin = z * math.sqrt((score * (1 - score) + z * z / (4 * n)) / n) / denom
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    return elo(score), elo(lo), elo(hi)


# Approximate published Stockfish-at-depth Elo (Computer Chess Rating List
# style estimates; exact values vary by Stockfish version and hardware)
STOCKFISH_ELO_AT_DEPTH = {
    1: 1750,
    2: 1950,
    3: 2150,
    4: 2300,
    5: 2400,
    6: 2500,
}


# ---------------- Tournament ----------------

def play_tournament(
    math_engine_eval_fn,
    math_engine_depth: int,
    sf_depth: int,
    sf_engine: chess.engine.SimpleEngine,
    n_games: int,
    seed: int = 0,
    random_plies: int = 4,
) -> dict:
    """Play n_games against Stockfish at sf_depth. Alternates colors each game.

    `random_plies` random opening moves are played before engines take over
    (default 4) — breaks determinism so the games are actually different.

    Returns dict with win/draw/loss counts and aggregate score.
    """
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    game_records = []
    for g in range(n_games):
        math_plays_white = (g % 2 == 0)
        math_mover = make_math_engine_mover(math_engine_eval_fn, math_engine_depth)
        sf_mover = make_stockfish_mover(sf_engine, sf_depth)
        if math_plays_white:
            white, black = math_mover, sf_mover
        else:
            white, black = sf_mover, math_mover
        t0 = time.time()
        result, n_plies = play_game(
            white, black, max_plies=200,
            random_plies=random_plies, rng=rng,
        )
        elapsed = time.time() - t0
        # Score from MATH engine's perspective
        if result == "1-0":
            score = 1.0 if math_plays_white else 0.0
        elif result == "0-1":
            score = 1.0 if not math_plays_white else 0.0
        elif result == "1/2-1/2":
            score = 0.5
        else:
            score = 0.5  # incomplete game → call it a draw
        if score == 1.0:
            wins += 1
        elif score == 0.5:
            draws += 1
        else:
            losses += 1
        game_records.append({
            "g": g, "math_white": math_plays_white, "result": result,
            "plies": n_plies, "score": score, "elapsed": elapsed,
        })
        print(f"  game {g+1}/{n_games} | "
              f"math={'W' if math_plays_white else 'B'} | "
              f"result={result:>7} | plies={n_plies:>3} | "
              f"score={score:.1f} | {elapsed:.1f}s",
              flush=True)

    total_score = wins + 0.5 * draws
    win_rate = total_score / n_games
    elo_pt, elo_lo, elo_hi = score_to_elo_diff(win_rate, n_games)
    return {
        "sf_depth": sf_depth,
        "n_games": n_games,
        "wins": wins, "draws": draws, "losses": losses,
        "score": total_score, "win_rate": win_rate,
        "elo_diff": elo_pt, "elo_lo": elo_lo, "elo_hi": elo_hi,
        "sf_elo_est": STOCKFISH_ELO_AT_DEPTH.get(sf_depth, None),
        "games": game_records,
    }


# ---------------- Report ----------------

def write_report(args, results: list[dict], elapsed: float):
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# Math engine vs Stockfish — iteration 0\n\n")
        f.write("Source eval: cx=13 PySR Pareto equation from "
                "`benchmarks/results/pysr_chess_stockfish.md` "
                "(500-position Stockfish d=15 distillation).\n\n")
        f.write(f"**Math engine search depth:** {args.math_depth}\n")
        f.write(f"**Games per opponent depth:** {args.n_games}\n\n")

        f.write("## Tournament results\n\n")
        f.write("| Stockfish depth | Stockfish Elo (est) | Wins | Draws | Losses | "
                "Math win-rate | Elo diff (95% CI) | Math Elo (est) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in results:
            sf_elo = r["sf_elo_est"]
            math_elo_pt = sf_elo + r["elo_diff"] if sf_elo else None
            math_elo_lo = sf_elo + r["elo_lo"] if sf_elo else None
            math_elo_hi = sf_elo + r["elo_hi"] if sf_elo else None
            ci_str = f"{r['elo_diff']:+.0f} ({r['elo_lo']:+.0f}, {r['elo_hi']:+.0f})"
            math_elo_str = (f"{math_elo_pt:.0f} ({math_elo_lo:.0f}, {math_elo_hi:.0f})"
                            if math_elo_pt is not None else "n/a")
            f.write(f"| {r['sf_depth']} | {sf_elo} | {r['wins']} | {r['draws']} | "
                    f"{r['losses']} | {r['win_rate']:.3f} | {ci_str} | {math_elo_str} |\n")

        f.write("\n## Eval expression\n\n")
        f.write("```\n")
        f.write("eval = (material_net / 0.01460058)\n")
        f.write("     - ((W_mobility * central_BP) /\n")
        f.write("        ((1.3168311 - B_king_zone_own_pawns) - central_BB))\n")
        f.write("```\n\n")
        f.write("Five features: material_net (centipawn balance), W_mobility "
                "(white pseudo-legal moves), central_BP (black pawns on "
                "d4/d5/e4/e5), B_king_zone_own_pawns (black king's pawn "
                "shield), central_BB (black bishops in center).\n\n")

        f.write("## Interpretation\n\n")
        f.write("- **Baseline reference.** Stockfish at d=1 is ~1750 Elo "
                "(strong amateur). Material-only at depth 3 would score "
                "roughly 1100-1300 Elo.\n")
        f.write("- **Math engine Elo estimate.** Win-rate against Stockfish "
                "at known-depth-Elo gives a calibrated point estimate. CI "
                "is wide on n=20-30 games — interpret accordingly.\n")
        f.write("- **Iteration 0 status.** This is the eval distilled from "
                "500 positions only. Self-improvement architecture (search-"
                "depth bootstrap or population coevolution) will produce "
                "iter 1+ evals; the right framing is whether these beat the "
                "iter-0 Elo here.\n")

        f.write(f"\n_Elapsed: {elapsed:.1f}s_\n")
    print(f"\nReport written: {OUT_REPORT}")


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-games", type=int, default=20,
                        help="games per Stockfish depth")
    parser.add_argument("--math-depth", type=int, default=3,
                        help="search depth for our engine")
    parser.add_argument("--sf-depths", type=int, nargs="+", default=[1, 2, 3],
                        help="Stockfish depths to play against")
    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-plies", type=int, default=4,
                        help="random opening plies to break determinism")
    args = parser.parse_args()

    print("=== Math engine vs Stockfish — iteration 0 ===\n")
    binary = find_stockfish(args.stockfish)

    expr, feature_names = sr_eval_expr_cx13()
    print(f"Eval expression (cx=13): {expr.to_string()}")
    print(f"Features used: {feature_names}")
    print("Using fast specialised eval (skips full feature bank)\n")

    eval_fn = make_fast_eval_cx13()

    t0 = time.time()
    results = []
    with chess.engine.SimpleEngine.popen_uci(binary) as sf_engine:
        for sf_depth in args.sf_depths:
            print(f"\n--- Stockfish depth {sf_depth} "
                  f"(~{STOCKFISH_ELO_AT_DEPTH.get(sf_depth, '?')} Elo) ---")
            r = play_tournament(
                eval_fn, args.math_depth, sf_depth, sf_engine,
                n_games=args.n_games, seed=args.seed + sf_depth,
                random_plies=args.random_plies,
            )
            results.append(r)
            print(f"  Result: W={r['wins']} D={r['draws']} L={r['losses']} "
                  f"| score={r['win_rate']:.3f} | "
                  f"Elo diff {r['elo_diff']:+.0f}")

    elapsed = time.time() - t0
    write_report(args, results, elapsed)
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
