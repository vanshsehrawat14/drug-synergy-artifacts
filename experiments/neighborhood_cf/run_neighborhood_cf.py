
"""Neighborhood Collaborative Filtering on raw O'Neil responses.

Uses the team's shared drug-pair folds. Hyperparameters are selected on
validation pairs, and predictions are evaluated on held-out test pairs.
"""

from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRELIM_DIR = ROOT / "experiments" / "preliminary"
sys.path.insert(0, str(PRELIM_DIR))

from common import load_oneil, split, ridge_additive, metrics
import pairlearners as L


DATA_FILE = ROOT / "results" / "neighborhood_cf" / "raw_response_targets.npz"
OUT_DIR = ROOT / "results" / "neighborhood_cf"

K_GRID = [3, 5, 10, 20]
SHRINK_GRID = [2.0, 5.0, 20.0]
RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def safe_metrics(pred, truth, H):
    """Compute the shared metrics, with explicit checks for finite values."""
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("Predictions or targets contain non-finite values.")
    return metrics(pred, truth, H)


def predict_cf(Y, H, aa, bb, tr, va, te):
    """Tune residual CF settings on validation pairs, then predict test pairs."""
    lam, beta = ridge_additive(H, Y, tr, va, lams=RIDGE_LAMBDAS)
    base = H @ beta
    residuals = Y - base

    best = None

    for shrink in SHRINK_GRID:
        M, W = L.residual_rows(
            residuals, tr, aa=aa, bb=bb, nd=H.shape[1]
        )
        S = L.drug_sim_from_resid(M, W, shrink=shrink)

        for k in K_GRID:
            val_residual_pred = L.knn(
                S, M, W, va, k, aa=aa, bb=bb
            )
            val_pred = base[va] + val_residual_pred
            val_mse = float(np.mean((val_pred - Y[va]) ** 2))

            if best is None or val_mse < best["val_mse"]:
                best = {
                    "val_mse": val_mse,
                    "k": k,
                    "shrink": shrink,
                    "lambda": lam,
                    "beta": beta,
                    "M": M,
                    "W": W,
                    "S": S,
                }

    test_residual_pred = L.knn(
        best["S"], best["M"], best["W"], te, best["k"],
        aa=aa, bb=bb
    )
    test_pred = base[te] + test_residual_pred

    return test_pred, base[te], {
        "k": best["k"],
        "shrink": best["shrink"],
        "lambda": best["lambda"],
        "validation_mse": best["val_mse"],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing target matrix: {DATA_FILE}\n"
            "Run the target-building script first."
        )

    with np.load(DATA_FILE, allow_pickle=False) as z:
        Y = z["Y"].astype(float)
        saved_pairs = [tuple(row) for row in z["pairs"].tolist()]
        saved_cells = z["cells"].tolist()
        folds = z["folds"].astype(int)

    shared = load_oneil()
    H = shared["H"]
    aa, bb = shared["aa"], shared["bb"]
    shared_pairs = [tuple(pair) for pair in shared["pairs"]]
    shared_cells = list(shared["cells"])

    # Verify that target rows/columns match the shared learner metadata.
    if saved_pairs != shared_pairs:
        raise ValueError("Target pair ordering does not match shared pair ordering.")
    if saved_cells != shared_cells:
        raise ValueError("Target cell ordering does not match shared cell ordering.")
    if Y.shape != (len(shared_pairs), len(shared_cells)):
        raise ValueError(f"Unexpected target matrix shape: {Y.shape}")
    if not np.isfinite(Y).all():
        raise ValueError("Target matrix contains missing or non-finite values.")
    if not np.array_equal(folds, shared["folds"]):
        raise ValueError("Saved folds do not match the shared folds.")

    n_pairs, n_cells = Y.shape
    cf_oof = np.full_like(Y, np.nan)
    baseline_oof = np.full_like(Y, np.nan)
    fold_results = []

    print("Neighborhood CF: raw-response evaluation")
    print(f"Pairs: {n_pairs}; cell lines: {n_cells}")
    print(f"Target range: {Y.min():.3f} to {Y.max():.3f}")
    print("\nRunning five outer folds...")

    for f in range(5):
        tr, va, te = split(folds, f)
        pred, baseline_pred, params = predict_cf(
            Y, H, aa, bb, tr, va, te
        )

        cf_oof[te] = pred
        baseline_oof[te] = baseline_pred

        fold_results.append({
            "fold": f,
            "n_train": int(len(tr)),
            "n_validation": int(len(va)),
            "n_test": int(len(te)),
            "cf_parameters": params,
            "cf": safe_metrics(pred, Y[te], H[te]),
            "baseline": safe_metrics(baseline_pred, Y[te], H[te]),
        })

        print(
            f"Fold {f}: CF MSE={fold_results[-1]['cf']['mse']:.4f}; "
            f"baseline MSE={fold_results[-1]['baseline']['mse']:.4f}; "
            f"k={params['k']}, shrink={params['shrink']}"
        )

    if not np.isfinite(cf_oof).all() or not np.isfinite(baseline_oof).all():
        raise ValueError("Some out-of-fold predictions were not generated.")

    overall = {
        "cf": safe_metrics(cf_oof, Y, H),
        "baseline": safe_metrics(baseline_oof, Y, H),
    }

    report = {
        "target": "raw combination inhibition",
        "n_pairs": int(n_pairs),
        "n_cells": int(n_cells),
        "folds": 5,
        "selection": {
            "validation_fold": "(test_fold + 1) mod 5",
            "k_grid": K_GRID,
            "shrink_grid": SHRINK_GRID,
            "ridge_lambdas": RIDGE_LAMBDAS,
        },
        "overall_out_of_fold_metrics": overall,
        "fold_results": fold_results,
    }

    np.savez_compressed(
        OUT_DIR / "cf_oof_predictions.npz",
        truth=Y,
        cf_prediction=cf_oof,
        baseline_prediction=baseline_oof,
        folds=folds,
        pairs=np.array(shared_pairs, dtype=str),
        cells=np.array(shared_cells, dtype=str),
    )
    (OUT_DIR / "cf_metrics.json").write_text(
        json.dumps(report, indent=2)
    )

    print("\nOVERALL OUT-OF-FOLD RESULTS")
    print(json.dumps(overall, indent=2))
    print(f"\nSaved predictions: {OUT_DIR / 'cf_oof_predictions.npz'}")
    print(f"Saved metrics:     {OUT_DIR / 'cf_metrics.json'}")


if __name__ == "__main__":
    main()

