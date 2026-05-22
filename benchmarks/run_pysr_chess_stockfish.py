"""Stockfish distillation benchmark.

Design doc: docs/benchmark_stockfish_distillation.md

Given (position, stockfish_eval) pairs, can PySR discover a closed-form
expression over our handcrafted spatial features that mimics Stockfish's
value function?

Pipeline:
  1. Load (or build) corpus via benchmarks/chess_corpus.py
  2. Encode FENs → planes via symbolic_chess.expression_layer.board.encode_corpus
  3. Build feature bank via chess_feature_bank
  4. TRAIN/TEST split (random for synthetic v0; by date later)
  5. Fit baselines: constant, material-LR, full-LR
  6. Run PySR; report Pareto front + comparison
  7. Write benchmarks/results/pysr_chess_stockfish.md

The `--mock` flag substitutes a deterministic material-based eval for
Stockfish. Use only to smoke-test the pipeline without the binary; results
under --mock are NOT a real test of SR distillation.
"""
from __future__ import annotations
import argparse
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau

from symbolic_chess.expression_layer import Variable, run_pysr, equation_table
from symbolic_chess.expression_layer.board import encode_corpus, chess_feature_bank

import chess_corpus  # benchmarks/chess_corpus.py


DEFAULT_CORPUS = ROOT / "data" / "chess" / "stockfish_eval.parquet"
OUT_REPORT = ROOT / "benchmarks" / "results" / "pysr_chess_stockfish.md"

PYSR_NITER = 60
PYSR_POPS = 25
PYSR_POP_SIZE = 40
PYSR_MAXSIZE = 35
PYSR_PARSIMONY = 0.0025


def mock_eval(planes: np.ndarray) -> np.ndarray:
    """Deterministic material-based eval used only when --mock is passed.

    Centipawn approximation: P=100, N=300, B=320, R=500, Q=900.
    Adds small noise so the regression target isn't perfectly linear.
    """
    from symbolic_chess.expression_layer.board import (
        piece_count, WP, WN, WB, WR, WQ, BP, BN, BB, BR, BQ
    )
    cp = (
        100 * (piece_count(planes, WP) - piece_count(planes, BP))
        + 300 * (piece_count(planes, WN) - piece_count(planes, BN))
        + 320 * (piece_count(planes, WB) - piece_count(planes, BB))
        + 500 * (piece_count(planes, WR) - piece_count(planes, BR))
        + 900 * (piece_count(planes, WQ) - piece_count(planes, BQ))
    )
    rng = np.random.default_rng(7)
    cp = cp + rng.normal(0, 30, size=cp.shape)
    return np.clip(cp, -1000, 1000)


def load_or_build_corpus(args) -> pd.DataFrame:
    """Load the corpus parquet, or generate it (mock or real stockfish)."""
    out = Path(args.corpus)
    if args.mock:
        # Mock path is separate to avoid contaminating real cache
        out = out.with_name(out.stem + "_mock.parquet")
        if out.exists() and not args.overwrite:
            print(f"Loading cached MOCK corpus from {out}")
            return pd.read_parquet(out)
        print(f"Generating {args.n} positions with MOCK eval (material + noise)")
        fens = chess_corpus.generate_random_fens(args.n, seed=args.seed)
        planes = encode_corpus(fens)
        evals = mock_eval(planes)
        df = pd.DataFrame({"fen": fens, "eval_cp": evals, "depth": -1})
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return df
    return chess_corpus.build_corpus(
        out, n_positions=args.n, depth=args.depth,
        stockfish_path=args.stockfish, seed=args.seed, overwrite=args.overwrite,
    )


def fit_baselines(X_train, y_train, X_test, y_test, feature_names, material_idx):
    """Fit constant + material-only LR + full LR; return dict of (test_r2, test_mse)."""
    out = {}

    # Constant predictor
    mean = float(y_train.mean())
    y_pred = np.full_like(y_test, mean)
    out["constant"] = {
        "r2": _r2(y_test, y_pred),
        "mse": float(np.mean((y_test - y_pred) ** 2)),
        "tau": 0.0,
        "expr": f"{mean:.2f}",
    }

    # Material-only LR (just material_net)
    lr_mat = LinearRegression()
    lr_mat.fit(X_train[:, material_idx:material_idx+1], y_train)
    y_pred = lr_mat.predict(X_test[:, material_idx:material_idx+1])
    tau = kendalltau(y_test, y_pred).correlation
    out["material_only_LR"] = {
        "r2": _r2(y_test, y_pred),
        "mse": float(np.mean((y_test - y_pred) ** 2)),
        "tau": float(tau) if np.isfinite(tau) else 0.0,
        "expr": f"{lr_mat.coef_[0]:.4f} * material_net + {lr_mat.intercept_:.2f}",
    }

    # Full LR on all features
    lr_full = LinearRegression()
    lr_full.fit(X_train, y_train)
    y_pred = lr_full.predict(X_test)
    tau = kendalltau(y_test, y_pred).correlation
    terms = [f"{c:+.3f}*{n}" for c, n in zip(lr_full.coef_, feature_names) if abs(c) > 1e-6]
    out["full_LR"] = {
        "r2": _r2(y_test, y_pred),
        "mse": float(np.mean((y_test - y_pred) ** 2)),
        "tau": float(tau) if np.isfinite(tau) else 0.0,
        "expr": " ".join(terms[:20]) + (" + ..." if len(terms) > 20 else ""),
    }
    return out


def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def run_pysr_distillation(X_train, y_train, X_test, y_test, feature_names):
    """Run PySR on (X_train, y_train); evaluate each Pareto equation on TEST."""
    variables = [Variable(name, X_train[:, j]) for j, name in enumerate(feature_names)]
    print(f"  PySR on {X_train.shape[0]} train rows, {len(feature_names)} features")
    try:
        model, _ = run_pysr(
            variables, y_train,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["tanh", "abs"],
            niterations=PYSR_NITER, populations=PYSR_POPS,
            population_size=PYSR_POP_SIZE, maxsize=PYSR_MAXSIZE,
            parsimony=PYSR_PARSIMONY,
            verbosity=1, procs=1, random_state=0,
        )
    except Exception as e:
        print(f"PySR failed: {type(e).__name__}: {e}")
        return None

    eqs = equation_table(model)
    # Evaluate each Pareto equation on TEST via the model's predict
    # (model.predict on a row vector → scalar; loop is fine at Pareto-size scale)
    eq_with_test = []
    for r in eqs:
        if not np.isfinite(r["loss"]):
            continue
        try:
            idx = next(i for i, row in enumerate(model.equations_.itertuples())
                       if row.complexity == r["complexity"] and abs(row.loss - r["loss"]) < 1e-12)
            y_pred = model.predict(X_test, index=idx)
            r2 = _r2(y_test, y_pred)
            tau = kendalltau(y_test, y_pred).correlation
            eq_with_test.append({
                **r,
                "test_r2": float(r2),
                "test_tau": float(tau) if np.isfinite(tau) else 0.0,
            })
        except Exception as e:
            print(f"  eval failed for cx={r['complexity']}: {e}")
    return eq_with_test


def write_report(args, df, baselines, eq_results, elapsed):
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# Stockfish distillation via PySR\n\n")
        f.write(f"Design doc: `docs/benchmark_stockfish_distillation.md`\n\n")
        mode = "MOCK (material + noise)" if args.mock else f"STOCKFISH depth={args.depth}"
        f.write(f"**Mode:** {mode}\n\n")
        f.write(f"**Corpus:** {len(df)} positions, eval range "
                f"[{df['eval_cp'].min():.0f}, {df['eval_cp'].max():.0f}] cp; "
                f"mean={df['eval_cp'].mean():.1f}, std={df['eval_cp'].std():.1f}\n\n")
        f.write(f"**Split:** random {int(args.train_frac*100)}/{int((1-args.train_frac)*100)}\n\n")

        f.write("## Baselines (TEST)\n\n")
        f.write("| Model | R2 | MSE | Kendall-τ | Expression |\n|---|---|---|---|---|\n")
        for name, b in baselines.items():
            f.write(f"| {name} | {b['r2']:.4f} | {b['mse']:.1f} | {b['tau']:.4f} | `{b['expr'][:120]}` |\n")

        f.write("\n## PySR Pareto front (TRAIN/TEST)\n\n")
        if not eq_results:
            f.write("_PySR did not produce evaluable equations (see log for cause)._\n")
        else:
            f.write("| Complexity | TRAIN loss | TEST R2 | TEST τ | Equation |\n|---|---|---|---|---|\n")
            eq_results_sorted = sorted(eq_results, key=lambda r: r["complexity"])
            for r in eq_results_sorted:
                eq_str = str(r['equation'])[:150]
                f.write(f"| {r['complexity']} | {r['loss']:.3e} | {r['test_r2']:.4f} | "
                        f"{r['test_tau']:.4f} | `{eq_str}` |\n")

            best_r2 = max(eq_results, key=lambda r: r["test_r2"])
            f.write(f"\n**Best PySR by TEST R2:** cx={best_r2['complexity']}, "
                    f"R2={best_r2['test_r2']:.4f}\n")

            f.write("\n## Acceptance (per design doc)\n\n")
            full_lr_r2 = baselines["full_LR"]["r2"]
            cond_1 = any(r["test_r2"] > 0.85 and r["complexity"] < 30 and
                         r["test_r2"] > full_lr_r2 + 0.03 for r in eq_results)
            cond_2 = any(r["test_r2"] > 0.75 and r["complexity"] <= 15 for r in eq_results)
            f.write(f"- (1) compact (cx<30) eq with TEST R2>0.85 beating full-LR by >0.03: "
                    f"{'PASS' if cond_1 else 'FAIL'}\n")
            f.write(f"- (2) very compact (cx≤15) eq with TEST R2>0.75: "
                    f"{'PASS' if cond_2 else 'FAIL'}\n")

        f.write(f"\n_Elapsed: {elapsed:.1f}s_\n")
    print(f"\nReport written: {OUT_REPORT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS))
    parser.add_argument("--n", type=int, default=500,
                        help="positions to generate if corpus doesn't exist")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mock", action="store_true",
                        help="use material-based mock eval (smoke test only, no stockfish needed)")
    parser.add_argument("--skip-pysr", action="store_true",
                        help="run baselines only — useful when PySR/Julia is broken")
    args = parser.parse_args()

    t0 = time.time()
    print("=== Stockfish distillation via PySR ===")
    if args.mock:
        print("[MOCK MODE] using material + noise as eval target. Smoke test only.")

    # 1. Corpus
    df = load_or_build_corpus(args)
    print(f"Loaded {len(df)} positions")

    # 2. Encode + build bank
    print("Encoding positions ...")
    planes = encode_corpus(df["fen"].tolist())
    print("Building feature bank ...")
    bank = chess_feature_bank(planes, fens=df["fen"].tolist())
    feature_names = list(bank.keys())
    X = np.column_stack([bank[n] for n in feature_names]).astype(np.float64)
    y = df["eval_cp"].to_numpy().astype(np.float64)
    print(f"Features: {len(feature_names)} | X shape: {X.shape}")

    # 3. Split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    n_train = int(args.train_frac * len(df))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # 4. Baselines
    material_idx = feature_names.index("material_net")
    print("\n--- Baselines ---")
    baselines = fit_baselines(X_train, y_train, X_test, y_test,
                               feature_names, material_idx)
    for name, b in baselines.items():
        print(f"  {name}: R2={b['r2']:.4f}, MSE={b['mse']:.1f}, tau={b['tau']:.4f}")

    # 5. PySR
    eq_results = []
    if not args.skip_pysr:
        print("\n--- PySR ---")
        eq_results = run_pysr_distillation(X_train, y_train, X_test, y_test, feature_names) or []
    else:
        print("\n--- PySR skipped (--skip-pysr) ---")

    elapsed = time.time() - t0

    # 6. Report
    write_report(args, df, baselines, eq_results, elapsed)
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
