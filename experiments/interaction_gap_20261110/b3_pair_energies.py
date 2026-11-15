"""Per-pair projected-contrast energies of the saved pair-seen predictions (B3), by pair-cell-split fold.

Run where the 15 prediction files are stored (no training). The output holds per-pair energy sums only, which
pair with the leak-free B1/B2 energies computed locally by b3_paired.py.

python b3_pair_energies.py --observations OBS --resolution RES --runs RUNS/f2_pair_seen \
    --split pair_cell_split.csv --pairseen pairseen_audit.json --output b3_pair_energies.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import Data, Panel, sha256, sha256_lf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "resolution", "runs", "split", "pairseen", "output"]:
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    data = Data(a.observations, a.resolution)
    P, C = data.shape
    panel = Panel(data)
    Mt, Mref = data.pc_mean_units(data.truth_u), data.pc_mean_units(data.cache["bliss"])
    T = panel(Mt - Mref)
    ids = pd.read_parquet(a.observations, columns=["observation_id", "is_combination", "pair_id"])
    ids = ids[ids.is_combination].reset_index(drop=True)
    if len(ids) != len(data.y) or not (ids.pair_id.astype(str).to_numpy() == data.comb.pair_id.to_numpy()).all():
        raise ValueError("Row order differs from the analysis table")
    pos = pd.Series(np.arange(len(ids)), index=ids.observation_id.astype(str))
    sp = pd.read_csv(a.split).astype({"pair_id": str, "cell_line": str})
    pi, ci = {p: i for i, p in enumerate(data.pairs)}, {c: i for i, c in enumerate(data.cells)}
    split = np.full((P, C), -1)
    split[sp.pair_id.map(pi).to_numpy(), sp.cell_line.map(ci).to_numpy()] = sp.fold.to_numpy()
    if (split < 0).any():
        raise ValueError("Pair-cell split incomplete")
    recorded = json.loads(Path(a.pairseen).read_text())["prediction_files_sha256"]
    E, match = np.zeros((3, P, C)), []
    for k, s in enumerate((2026, 2027, 2028)):
        pred = np.full(len(ids), np.nan)
        for f in range(5):
            path = Path(a.runs) / f"seed_{s}" / f"fold_{f}" / "predictions.parquet"
            match.append(sha256(path) == recorded[f"{s}/{f}"])
            t = pd.read_parquet(path)
            col = [c for c in t.columns if c in ("prediction", "pred", "y_pred")][0]
            idx = pos.reindex(t.observation_id.astype(str)).to_numpy()
            if np.isnan(idx).any() or not np.isnan(pred[idx.astype(int)]).all():
                raise ValueError("Prediction rows missing or duplicated")
            pred[idx.astype(int)] = t[col].to_numpy(float)
        if np.isnan(pred).any():
            raise ValueError("Rows without a prediction")
        E[k] = data.w * (T - panel(data.pc_mean_rows(pred) - Mref)) ** 2
    Tw, Em = data.w * T ** 2, E.mean(0)
    out = {"pairs": data.pairs, "prediction_files_match_recorded_sha256": int(sum(match)), "prediction_files": len(match),
           "observations_sha256": sha256(a.observations), "split_sha256_lf": sha256_lf(a.split),
           "b3_pooled_r2": float(1 - Em.sum() / Tw.sum()),
           "b3_seed_pooled_r2": [float(1 - E[k].sum() / Tw.sum()) for k in range(3)],
           "target_by_pair_fold": [[float(Tw[p, split[p] == f].sum()) for f in range(5)] for p in range(P)],
           "b3_by_pair_fold": [[float(Em[p, split[p] == f].sum()) for f in range(5)] for p in range(P)]}
    Path(a.output).write_text(json.dumps(out))
    print("b3_pooled_r2", out["b3_pooled_r2"], "files_match", out["prediction_files_match_recorded_sha256"])


if __name__ == "__main__":
    main()
