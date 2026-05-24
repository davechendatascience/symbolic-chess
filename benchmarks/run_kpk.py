"""KPK equilibrium distillation via PySR.

Design doc: docs/kpk_equilibrium_distillation.md

Tests whether SR can recover the closed-form classifier for "white wins in
K+P-vs-K" — a closed, theory-known endgame. Acceptance: Pareto eq with
TEST accuracy >= 0.90 at complexity <= 20 — comparable to textbook
rule-of-the-square + key-squares theory.

Baselines:
  - Majority class (predict always-1)
  - Rule-of-the-square strict (predict 1 iff BK outside the square of the pawn)
  - Linear regression on all features
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

from symbolic_chess.expression_layer import (
    Variable, run_tessera, equation_table, predict_with_tree,
)

import kpk_corpus  # benchmarks/kpk_corpus.py
from kpk_corpus import FEATURE_NAMES, build_kpk_corpus


DEFAULT_CORPUS = ROOT / "data" / "chess" / "kpk_corpus.parquet"
OUT_REPORT = ROOT / "benchmarks" / "results" / "pysr_kpk.md"

PYSR_NITER = 80
PYSR_POPS = 25
PYSR_POP_SIZE = 50
PYSR_MAXSIZE = 30
PYSR_PARSIMONY = 0.0025


# ---------------- Theoretical baselines ----------------

def rule_of_square_strict(df: pd.DataFrame) -> np.ndarray:
    """Predict 1 iff BK is outside the square of the pawn.

    "Outside square" = max(|bk_file - wp_file|, 7 - bk_rank) > (7 - wp_rank) + wtm.
    BK cannot catch the pawn → white wins. Misses positions where BK
    is inside the square but white still wins via key-squares logic.
    """
    file_gap = np.abs(df["bk_file"] - df["wp_file"])
    rank_gap = 7 - df["bk_rank"]
    promo_plus_tempo = (7 - df["wp_rank"]) + df["wtm"]
    outside = np.maximum(file_gap, rank_gap) > promo_plus_tempo
    return outside.astype(np.float64).to_numpy()


def rule_of_square_optimistic(df: pd.DataFrame) -> np.ndarray:
    """Outside the square → 1 (wins). Inside the square → majority class (1 wins).

    More accurate than strict on real corpora where >50% of inside-square
    positions are still won via key squares. Equivalent to "predict 1 unless
    pawn definitively can't promote" — but in KPK with WK present, that
    case is rare.
    """
    return np.ones(len(df), dtype=np.float64)   # equivalent: predict everything as win


# ---------------- Accuracy ----------------

def binary_accuracy(y_true: np.ndarray, y_pred_raw: np.ndarray) -> float:
    """Threshold predictions at 0.5; compute classification accuracy."""
    y_pred = (np.asarray(y_pred_raw) >= 0.5).astype(np.float64)
    return float(np.mean(y_pred == y_true))


# ---------------- Baselines ----------------

def fit_baselines(train_df, test_df, X_train, X_test, y_train, y_test, feature_names):
    out = {}

    # Constant (majority class on TRAIN)
    majority = float(np.round(y_train.mean()))
    y_pred = np.full_like(y_test, majority)
    out["majority_class"] = {
        "acc": binary_accuracy(y_test, y_pred),
        "expr": f"{majority:.0f}",
    }

    # Rule of the square (strict)
    y_pred_strict = rule_of_square_strict(test_df)
    out["rule_of_square_strict"] = {
        "acc": binary_accuracy(y_test, y_pred_strict),
        "expr": "1 if max(|bk_file-wp_file|, 7-bk_rank) > (7-wp_rank)+wtm else 0",
    }

    # Linear regression on all features
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    terms = [f"{c:+.3f}*{n}" for c, n in zip(lr.coef_, feature_names) if abs(c) > 1e-3]
    out["full_LR"] = {
        "acc": binary_accuracy(y_test, y_pred_lr),
        "expr": " ".join(terms) + f" + {lr.intercept_:+.3f}",
    }
    return out


# ---------------- PySR ----------------

def run_pipeline(X_train, y_train, X_test, y_test, feature_names):
    """Returns list of {complexity, loss, equation, test_acc} dicts."""
    variables = [Variable(name, X_train[:, j]) for j, name in enumerate(feature_names)]
    print(f"  PySR on {X_train.shape[0]} train rows, {len(feature_names)} features")
    try:
        gp, _ = run_tessera(
            variables, y_train,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["abs", "tanh"],
            niterations=PYSR_NITER, populations=PYSR_POPS,
            population_size=PYSR_POP_SIZE, maxsize=PYSR_MAXSIZE,
            parsimony=PYSR_PARSIMONY,
            verbose=True, random_state=0,
        )
    except Exception as e:
        print(f"PySR failed: {type(e).__name__}: {e}")
        return None

    eqs = equation_table(gp)
    eq_with_test = []
    for r in eqs:
        if not np.isfinite(r["loss"]):
            continue
        try:
            y_pred = predict_with_tree(r["tree"], X_test, feature_names)
            eq_with_test.append({
                **r,
                "test_acc": binary_accuracy(y_test, y_pred),
            })
        except Exception as e:
            print(f"  eval failed for cx={r['complexity']}: {e}")
    return eq_with_test


# ---------------- Report ----------------

def write_report(args, df, baselines, eq_results, elapsed):
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    n_wins = int((df["label"] == 1.0).sum())
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# KPK equilibrium distillation via PySR\n\n")
        f.write("Design doc: `docs/kpk_equilibrium_distillation.md`\n\n")
        f.write(f"**Corpus:** {len(df)} positions @ Stockfish depth={args.depth}; "
                f"white_wins={n_wins} ({100*n_wins/len(df):.1f}%), "
                f"draws={len(df) - n_wins}\n\n")
        f.write(f"**Split:** random {int(args.train_frac*100)}/{int((1-args.train_frac)*100)}\n\n")

        f.write("## Baselines (TEST accuracy)\n\n")
        f.write("| Model | TEST Accuracy | Expression |\n|---|---|---|\n")
        for name, b in baselines.items():
            expr = b["expr"][:130]
            f.write(f"| {name} | {b['acc']:.4f} | `{expr}` |\n")

        f.write("\n## PySR Pareto front\n\n")
        if not eq_results:
            f.write("_PySR produced no evaluable equations (see log)._\n")
        else:
            f.write("| Complexity | TRAIN loss | TEST Accuracy | Equation |\n|---|---|---|---|\n")
            for r in sorted(eq_results, key=lambda r: r["complexity"]):
                eq_str = str(r["equation"])[:180]
                f.write(f"| {r['complexity']} | {r['loss']:.3e} | "
                        f"{r['test_acc']:.4f} | `{eq_str}` |\n")

            best = max(eq_results, key=lambda r: r["test_acc"])
            f.write(f"\n**Best PySR by TEST acc:** cx={best['complexity']}, "
                    f"acc={best['test_acc']:.4f}\n")

            f.write("\n## Acceptance\n\n")
            strong = any(r["test_acc"] >= 0.95 and r["complexity"] <= 15
                          for r in eq_results)
            ok = any(r["test_acc"] >= 0.90 and r["complexity"] <= 20
                     for r in eq_results)
            marginal = any(r["test_acc"] >= 0.85 and r["complexity"] <= 30
                            for r in eq_results)
            beats_lr = best["test_acc"] > baselines["full_LR"]["acc"]
            f.write(f"- Strong (cx<=15, acc>=0.95): **{'PASS' if strong else 'FAIL'}**\n")
            f.write(f"- Acceptable (cx<=20, acc>=0.90): **{'PASS' if ok else 'FAIL'}**\n")
            f.write(f"- Marginal (cx<=30, acc>=0.85): **{'PASS' if marginal else 'FAIL'}**\n")
            f.write(f"- Beats full-LR: **{'PASS' if beats_lr else 'FAIL'}** "
                    f"(PySR {best['test_acc']:.4f} vs LR {baselines['full_LR']['acc']:.4f})\n")

            f.write("\n## Theory vs SR — observations\n\n")
            f.write("_(Auto-generated section; fill in by inspecting top Pareto expressions.)_\n\n")
            f.write("- Rule-of-the-square strict baseline gives ")
            f.write(f"{baselines['rule_of_square_strict']['acc']:.3f} accuracy alone.\n")
            f.write("- Best SR expression at cx <= 15 captures: TBD (inspect after run).\n")
            f.write("- Whether opposition/key-square structure emerges: TBD.\n")

        f.write(f"\n_Elapsed: {elapsed:.1f}s_\n")
    print(f"\nReport written: {OUT_REPORT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default=str(DEFAULT_CORPUS))
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="cap per-position Stockfish time (seconds)")
    parser.add_argument("--stockfish", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-pysr", action="store_true",
                        help="run baselines only")
    args = parser.parse_args()

    t0 = time.time()
    print("=== KPK equilibrium distillation via PySR ===\n")

    # 1. Corpus
    df = build_kpk_corpus(
        Path(args.corpus), n_positions=args.n, depth=args.depth,
        time_limit=args.time_limit,
        stockfish_path=args.stockfish, seed=args.seed, overwrite=args.overwrite,
    )
    print(f"Loaded {len(df)} positions")

    # 2. Build X, y
    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.float64)
    print(f"Features: {len(FEATURE_NAMES)} | X shape: {X.shape} | "
          f"label mean: {y.mean():.3f}")

    # 3. Random split
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(df))
    n_train = int(args.train_frac * len(df))
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"Train: {len(train_idx)} | Test: {len(test_idx)}")

    # 4. Baselines
    print("\n--- Baselines (TEST) ---")
    baselines = fit_baselines(train_df, test_df, X_train, X_test, y_train, y_test,
                                FEATURE_NAMES)
    for name, b in baselines.items():
        print(f"  {name}: acc={b['acc']:.4f}")

    # 5. PySR
    eq_results = []
    if not args.skip_pysr:
        print("\n--- PySR ---")
        eq_results = run_pipeline(X_train, y_train, X_test, y_test, FEATURE_NAMES) or []
        if eq_results:
            best = max(eq_results, key=lambda r: r["test_acc"])
            print(f"\nBest PySR: cx={best['complexity']} | test_acc={best['test_acc']:.4f}")
            print(f"  {str(best['equation'])[:200]}")
    else:
        print("\n--- PySR skipped (--skip-pysr) ---")

    elapsed = time.time() - t0
    write_report(args, df, baselines, eq_results, elapsed)
    print(f"\nElapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
