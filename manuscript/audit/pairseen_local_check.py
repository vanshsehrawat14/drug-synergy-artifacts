"""Local consistency checks of the pair-seen GPU runs from their returned metadata and metrics.

Checks provenance fields, training-set sizes against the held-out-pair protocol, monotherapy coverage and
the pooled response MSE of both procedures. Contrast-level R^2 needs the per-row predictions, which are
retained on the cluster (see pairseen_audit.py).

python manuscript/audit/pairseen_local_check.py --runs results/final_20261110/gpu_return/runs/f2_pair_seen \
    --observations data/processed/oneil_v1/observations.parquet --resolution RESOLUTION_DIR \
    --manifest results/final_20261110/gpu_return/full_return_manifest.json --output manuscript/audit/outputs/pairseen_local_check.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    """SHA-256 of the file with CRLF normalized to LF (git autocrlf rewrites line endings on Windows checkouts)."""
    return hashlib.sha256(Path(path).read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for k in ["runs", "observations", "resolution", "manifest", "output"]:
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    runs = Path(a.runs)
    manifest = {Path(f["path"]).as_posix(): f["sha256"] for f in json.loads(Path(a.manifest).read_text())["files"]}
    obs = pd.read_parquet(a.observations, columns=["fold", "is_combination"])
    fold_rows = obs[obs.is_combination].fold.value_counts().sort_index()
    rows = []
    for seed in (2026, 2027, 2028):
        for fold in range(5):
            d = runs / f"seed_{seed}" / f"fold_{fold}"
            meta = json.loads((d / "metadata.json").read_text())
            pooled = json.loads((d / "evaluation" / "metrics.json").read_text())["pooled"]
            rel = f"results/final_20261110/runs/f2_pair_seen/seed_{seed}/fold_{fold}"
            hashes = {name: sha256(d / name) == manifest.get(f"{rel}/{name}") for name in
                      ["metadata.json", "config.json", "evaluation/metrics.json", "checkpoints/history.json"]}
            pair_train = int(fold_rows.drop([fold, (fold + 1) % 5]).sum())
            rows.append(dict(seed=seed, fold=fold, commit=meta["git"].get("commit", meta["git"].get("revision")),
                             dirty=meta["git"].get("dirty"), split_sha=meta.get("split_assignment_sha256"),
                             train_rows=meta["partition_rows"].get("train"), validation_rows=meta["partition_rows"].get("validation"),
                             test_rows=meta["partition_rows"].get("test"), pair_split_train_rows=pair_train,
                             test_sse=pooled["sse"], test_n=pooled["n"], hashes_match_manifest=all(hashes.values()),
                             monotherapy=json.dumps(meta.get("monotherapy_coverage"))[:200]))
    t = pd.DataFrame(rows)
    per_seed = t.groupby("seed").apply(lambda g: g.test_sse.sum() / g.test_n.sum(), include_groups=False)
    res = Path(a.resolution)
    meta = json.loads((res / "inputs.json").read_text())
    with np.load(res / "cache.npz") as z:
        rpe = z["row_pair_error"]
    base_cols = [j for j, n in enumerate(meta["names"]) if n.startswith("neural_base_F2:")]
    base_mse = rpe[:, base_cols].sum(0) / meta["rows"]
    out = {"runs": len(t), "commits": sorted(set(map(str, t.commit))), "dirty_flags": sorted(set(map(str, t.dirty))),
           "split_sha256": sorted(set(map(str, t.split_sha))), "all_hashes_match_manifest": bool(t.hashes_match_manifest.all()),
           "test_rows_total_per_seed": t.groupby("seed").test_n.sum().astype(int).to_dict(),
           "train_rows_pair_cell_split": t.groupby("fold").train_rows.first().astype(int).to_dict(),
           "train_rows_pair_split": t.groupby("fold").pair_split_train_rows.first().astype(int).to_dict(),
           "pair_seen_mse_per_seed": {int(k): float(v) for k, v in per_seed.items()},
           "pair_seen_mse_mean": float(per_seed.mean()),
           "f2_base_mse_per_seed": [float(x) for x in base_mse], "f2_base_mse_mean": float(base_mse.mean()),
           "relative_change": float(per_seed.mean() / base_mse.mean() - 1),
           "monotherapy_coverage_example": t.monotherapy.iloc[0]}
    Path(a.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
