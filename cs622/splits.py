"""Central canonical unordered-pair folds. No labels enter assignment."""
import argparse
import json
import re

import numpy as np
import pandas as pd

from .io import digest, write_json


def normalize_name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing identifier")
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def pair_id(a, b):
    # JSON encoding prevents delimiter collisions in arbitrary source identifiers.
    a, b = sorted((normalize_name(a), normalize_name(b)))
    return json.dumps([a, b], separators=(",", ":"))


def published_folds(path):
    df = pd.read_csv(path)
    df["pair_id"] = [pair_id(a, b) for a, b in zip(df.drug_a_name, df.drug_b_name)]
    if df.groupby("pair_id").fold.nunique().max() != 1:
        raise ValueError("Published pair crosses folds")
    folds = df[["pair_id", "fold"]].drop_duplicates().sort_values("pair_id")
    if set(folds.fold) != set(range(5)):
        raise ValueError("Expected published folds 0..4")
    return folds.reset_index(drop=True)


def generated_folds(pairs, seed=2026, n_folds=5):
    pairs = np.array(sorted(set(pairs)))
    if n_folds < 3 or len(pairs) < n_folds:
        raise ValueError("Need at least three nonempty folds")
    order = np.random.default_rng(seed).permutation(len(pairs))
    fold = np.empty(len(pairs), dtype=int)
    fold[order] = np.arange(len(pairs)) % n_folds
    return pd.DataFrame({"pair_id": pairs, "fold": fold})


def partitions(frame, test_fold, n_folds=5, unit="pair"):
    if test_fold not in range(n_folds):
        raise ValueError("Invalid test fold")
    validate_folds(frame, n_folds, unit)
    val_fold = (test_fold + 1) % n_folds
    folds = frame.fold.to_numpy()
    return tuple(np.flatnonzero(mask) for mask in (
        (folds != test_fold) & (folds != val_fold), folds == val_fold, folds == test_fold))


def validate_folds(frame, n_folds=5, unit="pair"):
    """unit "pair": the published held-out-pair contract; "pair_cell": each pair-cell in one fold."""
    if frame.empty or frame.fold.isna().any() or set(frame.fold) != set(range(n_folds)):
        raise ValueError("Missing or unexpected folds")
    keys = {"pair": ["pair_id"], "pair_cell": ["pair_id", "cell_line"]}[unit]
    if frame.groupby(keys, observed=True).fold.nunique().max() != 1:
        raise ValueError("Pair leakage across folds" if unit == "pair" else "Pair-cell leakage across folds")
    if "block_id" in frame and frame.groupby("block_id", observed=True).fold.nunique().max() != 1:
        raise ValueError("Block leakage across folds")
    if "observation_id" in frame and frame.observation_id.duplicated().any():
        raise ValueError("Duplicate observation IDs")
    return {str(k): {"rows": int(len(g)), "pairs": int(g.pair_id.nunique())}
            for k, g in frame.groupby("fold", observed=True)}


def validate_seen_drugs(frame, test_fold, n_folds=5, unit="pair"):
    tr, va, te = partitions(frame, test_fold, n_folds, unit)
    drugs = lambda ix: set(frame.iloc[ix].drug_a) | set(frame.iloc[ix].drug_b)
    missing = (drugs(va) | drugs(te)) - drugs(tr)
    if missing:
        raise ValueError(f"Held-out drugs absent from training: {sorted(missing)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", help="Prepared observations or panel Parquet")
    p.add_argument("--output")
    args = p.parse_args()
    df = pd.read_parquet(args.dataset)
    report = {"counts": validate_folds(df), "sha256": digest(args.dataset)}
    for fold in range(5):
        validate_seen_drugs(df, fold)
    report["pair_and_block_disjoint"] = True
    report["held_out_drugs_seen_in_training"] = True
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
