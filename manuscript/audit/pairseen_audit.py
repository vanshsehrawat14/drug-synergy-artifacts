"""Independent recomputation of the pair-seen comparison (B2) from per-row predictions.

Uses the operators of interaction_audit.py (same directory). Inputs: the O'Neil observation table, the
15 pair-seen prediction files, the pair-cell split, and a resolution cache holding unit means of truth,
the Bliss reference and the three held-out-pair f2_base seeds. Run where the predictions are stored.

python pairseen_audit.py --observations observations.parquet --runs RUNS/f2_pair_seen --resolution RESOLUTION_DIR \
    --split pair_cell_split.csv --noise-share 0.0978 --output pairseen_audit.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from interaction_audit import Operators, boot, jack, sha256  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "runs", "resolution", "split", "output"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--noise-share", type=float, required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=97)
    a = ap.parse_args()
    res, runs = Path(a.resolution), Path(a.runs)
    obs = pd.read_parquet(a.observations, columns=["observation_id", "block_id", "drug_a", "drug_b", "cell_line", "dose_a",
                                                   "dose_b", "inhibition", "is_combination", "fold", "pair_id"])
    obs = obs[obs.is_combination].reset_index(drop=True)
    for c in ["drug_a", "drug_b", "cell_line", "pair_id", "observation_id"]:
        obs[c] = obs[c].astype(str)
    obs["ra"] = obs.groupby("block_id").dose_a.rank(method="dense").astype(int) - 1
    obs["rb"] = obs.groupby("block_id").dose_b.rank(method="dense").astype(int) - 1
    obs["unit"], ukeys = pd.factorize(pd.MultiIndex.from_frame(obs[["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"]]), sort=True)
    units = pd.DataFrame(ukeys.tolist(), columns=["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"])
    units["n"] = np.bincount(obs.unit)
    first = obs.groupby("unit").head(1).sort_values("unit")
    for c in ["drug_a", "drug_b", "fold"]:
        units[c] = first[c].to_numpy()
    units["pc"] = pd.factorize(units.pair_id + "|" + units.cell_line, sort=True)[0]
    units["pair"] = pd.factorize(units.pair_id, sort=True)[0]
    y = obs.inhibition.to_numpy(float)
    n_u = units.n.to_numpy(float)
    truth = np.bincount(obs.unit, weights=y) / n_u

    meta = json.loads((res / "inputs.json").read_text())
    coords = pd.read_parquet(res / "coordinates.parquet")
    with np.load(res / "cache.npz") as z:
        values = z["values"]
    merged = units.reset_index().merge(coords[["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"]].astype({"ra": int, "rb": int})
                                       .reset_index().rename(columns={"index": "cache_row"}),
                                       on=["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"], how="left")
    if merged.cache_row.isna().any() or len(coords) != len(units):
        raise ValueError("Unit grouping differs from the cache")
    values = values[merged.sort_values("index").cache_row.to_numpy(int)]
    names = meta["names"]
    col = lambda nm: values[:, names.index(nm)]
    out = {"inputs": {"observations_sha256": sha256(a.observations), "cache_sha256": sha256(res / "cache.npz"),
                      "split_sha256": sha256(a.split)},
           "truth_max_abs_diff_vs_cache": float(np.abs(truth - col("truth")).max())}
    bliss = col("bliss")
    base = np.column_stack([col(f"neural_base_F2:{s}") for s in (2026, 2027, 2028)])

    # Pair-seen predictions: every combination row exactly once per seed.
    pid = pd.Series(np.arange(len(obs)), index=obs.observation_id)
    ps, row_mse, files = [], {}, {}
    for s in (2026, 2027, 2028):
        pred = np.full(len(obs), np.nan)
        for f in range(5):
            p = runs / f"seed_{s}" / f"fold_{f}" / "predictions.parquet"
            files[f"{s}/{f}"] = sha256(p)
            t = pd.read_parquet(p)
            pc = [c for c in t.columns if c in ("prediction", "pred", "y_pred")][0]
            idx = pid.reindex(t.observation_id.astype(str)).to_numpy()
            if np.isnan(idx).any() or not np.isnan(pred[idx.astype(int)]).all():
                raise ValueError("Prediction rows missing or duplicated")
            pred[idx.astype(int)] = t[pc].to_numpy(float)
        if np.isnan(pred).any():
            raise ValueError("Rows without a pair-seen prediction")
        row_mse[s] = float(np.mean((y - pred) ** 2))
        ps.append(np.bincount(obs.unit, weights=pred) / n_u)
    ps = np.column_stack(ps)
    out["prediction_files_sha256"] = files
    out["response_mse"] = {"pair_seen_per_seed": row_mse, "pair_seen_mean": float(np.mean(list(row_mse.values())))}

    split = pd.read_csv(a.split).astype({"pair_id": str, "cell_line": str}).rename(columns={"fold": "split_fold"})
    fixed = np.column_stack([truth, truth - bliss, truth[:, None] - base, truth[:, None] - ps])
    ib, ip = slice(2, 5), slice(5, 8)

    def evaluate(keep):
        ops = Operators(units, keep)
        ep, ew, _ = ops.energies(fixed)
        cells = ops.cells.merge(split, on=["pair_id", "cell_line"], how="left")
        return ep, ew, ops, cells

    ep, ew, ops, cells = evaluate(np.ones(len(units), bool))
    pc_fold = cells.split_fold.to_numpy(int)
    # Per-pair energies are needed per fold: recompute pair-cell residual energies per pair-cell.
    rp_full = None
    result = {}
    for level, E in [("pair_cell", ep), ("dose_shape", ew)]:
        for j, tname in enumerate(["raw", "bliss_residual"]):
            T = E[:, j]
            eb, epz = E[:, ib].mean(1), E[:, ip].mean(1)
            result[f"{level}:{tname}"] = {"f2_base_r2": float(1 - eb.sum() / T.sum()), "pair_seen_r2": float(1 - epz.sum() / T.sum()),
                                          "difference": float((eb.sum() - epz.sum()) / T.sum()),
                                          "pair_seen_r2_per_seed": [float(1 - E[:, 5 + k].sum() / T.sum()) for k in range(3)],
                                          "f2_base_r2_per_seed": [float(1 - E[:, 2 + k].sum() / T.sum()) for k in range(3)]}
    # Primary: Bliss-residual pair-cell contrast, with grouped intervals and per-fold differences.
    T = ep[:, 1]; eb = ep[:, ib].mean(1); epz = ep[:, ip].mean(1)
    rng = np.random.default_rng(a.seed)
    prim = result["pair_cell:bliss_residual"]
    prim["difference_pair_interval"] = boot(np.column_stack([eb, epz]), T[:, None], lambda x, y_: (x[0] - x[1]) / y_[0], a.boot, rng)
    prim["pair_seen_pair_interval"] = boot(epz[:, None], T[:, None], lambda x, y_: 1 - x[0] / y_[0], a.boot, rng)
    leaves = []
    for d in sorted(set(units.drug_a) | set(units.drug_b)):
        lp, _, _, _ = evaluate(((units.drug_a != d) & (units.drug_b != d)).to_numpy())
        leaves.append((lp[:, 1].sum(), lp[:, ib].mean(1).sum(), lp[:, ip].mean(1).sum()))
    leaves = np.array(leaves)
    prim["difference_drug_interval"] = jack(prim["difference"], (leaves[:, 1] - leaves[:, 2]) / leaves[:, 0])
    prim["pair_seen_drug_interval"] = jack(prim["pair_seen_r2"], 1 - leaves[:, 2] / leaves[:, 0])
    prim["noise_corrected_difference"] = float(prim["difference"] / (1 - a.noise_share))
    # Per pair-cell-split fold: residual energies restricted to the fold's pair-cells (projection fixed).
    rp = None
    m = ops.agg @ fixed[ops.rows]
    r = np.empty_like(m)
    for idx, sw, q in ops.panel_groups:
        x = m[idx] * sw[:, None]
        r[idx] = (x - q @ (q.T @ x)) / sw[:, None]
    wpc = ops.w_pc
    per_fold = {}
    for f in range(5):
        k = pc_fold == f
        Tf = (wpc[k] * r[k, 1] ** 2).sum()
        ebf = np.mean([(wpc[k] * r[k, 2 + s] ** 2).sum() for s in range(3)])
        epf = np.mean([(wpc[k] * r[k, 5 + s] ** 2).sum() for s in range(3)])
        per_fold[f] = {"f2_base_r2": float(1 - ebf / Tf), "pair_seen_r2": float(1 - epf / Tf), "difference": float((ebf - epf) / Tf)}
    prim["per_pair_cell_split_fold"] = per_fold
    prim["pair_cell_split_folds_positive"] = int(sum(v["difference"] > 0 for v in per_fold.values()))
    # Per original held-out-pair fold (the frozen B2 rule counts these).
    orig = cells.fold.to_numpy(int)
    per_orig = {}
    for f in range(5):
        k = orig == f
        Tf = (wpc[k] * r[k, 1] ** 2).sum()
        ebf = np.mean([(wpc[k] * r[k, 2 + s] ** 2).sum() for s in range(3)])
        epf = np.mean([(wpc[k] * r[k, 5 + s] ** 2).sum() for s in range(3)])
        per_orig[f] = {"f2_base_r2": float(1 - ebf / Tf), "pair_seen_r2": float(1 - epf / Tf), "difference": float((ebf - epf) / Tf)}
    prim["per_original_fold"] = per_orig
    prim["original_folds_positive"] = int(sum(v["difference"] > 0 for v in per_orig.values()))
    out["results"] = result
    Path(a.output).write_text(json.dumps(out, indent=1))
    print("done")


if __name__ == "__main__":
    main()
