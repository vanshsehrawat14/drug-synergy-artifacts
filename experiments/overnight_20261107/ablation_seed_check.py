"""Exploratory: per-seed and per-fold response-MSE change of each mask vs full-input F2 (sprint_v1 f2_base), same seed."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from cs622.sprint_report import index, load_truth, out_of_fold

runs, out = Path(sys.argv[1]), Path(sys.argv[2])
df, sha, _ = load_truth("data/processed/oneil_v1")
t = index(df)
y = t["y"]
fold = df.fold.to_numpy() if "fold" in df else None
base = Path("results/neural/runs/sprint_v1/runs/f2_base")
rows = []
for s in (2026, 2027, 2028):
    ref = out_of_fold({k: base / f"seed_{s}" / f"fold_{k}" for k in range(5)}, t)
    for m in ("no_drug_identity", "no_cell_identity", "no_dose"):
        p = out_of_fold({k: runs / m / f"seed_{s}" / f"fold_{k}" for k in range(5)}, t)
        e_m, e_r = (y - p) ** 2, (y - ref) ** 2
        rows.append(dict(mask=m, seed=s, fold="all", change=e_m.sum() / e_r.sum() - 1))
        if fold is not None:
            for k in range(5):
                f = fold == k
                rows.append(dict(mask=m, seed=s, fold=k, change=e_m[f].sum() / e_r[f].sum() - 1))
pd.DataFrame(rows).to_csv(out, index=False)
d = pd.DataFrame(rows)
print(d[d.fold == "all"].pivot(index="mask", columns="seed", values="change").round(4))
if len(sys.argv) > 4:  # gpu_ablation_summary.csv: frozen contrasts for the masks plus per-seed response range
    c = pd.read_csv(sys.argv[3])
    c = c[c.model.isin(["no_drug_identity", "no_cell_identity", "no_dose"])].copy()
    a = d[d.fold == "all"].groupby("mask").change.agg(["min", "max"])
    c["response_seed_min"] = c.model.map(a["min"]).where(c.level == "response")
    c["response_seed_max"] = c.model.map(a["max"]).where(c.level == "response")
    c.to_csv(sys.argv[4], index=False)
