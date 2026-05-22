"""Drive math-engine-cpp self-play via UCI to generate SR training data.

Iter-1+ of the self-improvement loop. Where Stockfish self-play generates
training data anchored to Stockfish's style, math-engine-cpp self-play
generates data anchored to OUR current eval — the prerequisite for true
self-discovery (Architecture A from docs/math_engine_cpp_v0.md).

Records identical schema to chess_engine/self_play.py:
  (game_id, ply_number, fen, outcome, plies_to_end, random_phase)

Usage:
    python benchmarks/cpp_engine_self_play.py --n 100 --depth 5 \
        --engine math-engine-cpp/build/math_engine.exe \
        --out data/chess/cpp_self_play_v0.parquet
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from symbolic_chess.chess_engine.self_play import generate_self_play_corpus


def main():
    parser = argparse.ArgumentParser(description="math-engine-cpp self-play corpus")
    parser.add_argument("--out", type=str,
                        default="data/chess/cpp_self_play.parquet")
    parser.add_argument("--n", type=int, default=100, help="number of games")
    parser.add_argument("--depth", type=int, default=5,
                        help="search depth for math-engine-cpp")
    parser.add_argument("--random-plies", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--engine", type=str,
                        default="math-engine-cpp/build/math_engine.exe",
                        help="path to math-engine-cpp binary")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    engine_path = str(Path(args.engine).resolve())
    if not Path(engine_path).is_file():
        raise FileNotFoundError(
            f"math-engine-cpp binary not found at {engine_path}. "
            f"Build it first: cd math-engine-cpp && "
            f"cmake -B build -S . && cmake --build build --config Release"
        )

    df = generate_self_play_corpus(
        Path(args.out),
        n_games=args.n,
        depth=args.depth,
        random_plies=args.random_plies,
        max_plies=args.max_plies,
        engine_path=engine_path,
        engine_name="math-engine-cpp",
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(df.head())
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
