"""Phase 3.4 confirmatory seed 2029: frozen ablation statistics per seed, pooled 2026-2029, and the decision table.

python experiments/overnight_20261107/seed2029_report.py --runs <runs 2026-2028> <seed2029 root> --f2-runs <seed2029 root>/f2_base
    --resolution <resolution cache> --original ../handoff/gpu/ablation_comparisons.csv --output ../handoff/seed2029

Decision rules are the frozen ones in cs622.ablation_summary (materiality, Holm across masks, Phase 3.4 trigger).
The protocol does not say how the extra seed enters the decision; both the pooled four-seed estimate (frozen
seed-averaged estimator) and the seed-2029 replicate are reported. Drug-identity response is exploratory (not triggered).
"""
import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from cs622 import ablation_summary
from cs622.resolution import means
from cs622.sprint_report import index, load_truth, out_of_fold

TRIGGERED = [("no_drug_identity", "panel_contrast"), ("no_cell_identity", "panel_contrast"),
             ("no_cell_identity", "dose_contrast"), ("no_dose", "dose_contrast")]
EXPLORATORY = [("no_drug_identity", "response")]
COLS = ["change", "pair_low", "pair_high", "drug_low", "drug_high", "fold_min", "fold_max", "folds_same_sign",
        "p", "p_holm_masks", "material_change", "confirmatory_seed_trigger"]


def summarize(a, seeds, name):
    out = Path(a.output) / name
    if (out / "ablation_comparisons.csv").exists():  # resume after an interrupted report
        return pd.read_csv(out / "ablation_comparisons.csv").assign(analysis=name)
    ablation_summary.run(Namespace(runs=a.runs, output=str(out), pseudo=False, threads=a.threads, resolution=a.resolution,
                                   seeds=seeds, f2_runs=a.f2_runs))
    return pd.read_csv(out / "ablation_comparisons.csv").assign(analysis=name)


def check_f2_path(a):
    """The extra F2 column is built by the ablation path; it must reproduce the cached historical F2 columns."""
    base = Path(a.resolution)
    meta = json.loads((base / "inputs.json").read_text())
    with np.load(base / "cache.npz") as c:
        values, rowerr = c["values"], c["row_pair_error"]
    df, _, _ = load_truth("data/processed/oneil_v1")
    t = index(df)
    group = ablation_summary.unit_group(df, t)
    n = np.bincount(group)
    j = meta["names"].index("neural_base_F2:2026")
    p = out_of_fold({k: Path("results/neural/runs/sprint_v1/runs/f2_base/seed_2026") / f"fold_{k}" for k in range(5)}, t)
    re = np.bincount(t["pair"], weights=(t["y"] - p) ** 2, minlength=len(t["pairs"]))
    return dict(unit_means_max_abs_diff=float(np.abs(means(group, p, n) - values[:, j]).max()),
                pair_sse_max_rel_diff=float(np.abs(re / rowerr[:, j] - 1).max()))


def per_fold_response(a):
    df, _, _ = load_truth("data/processed/oneil_v1")
    t = index(df)
    y, fold = t["y"], df.fold.to_numpy()
    roots = [Path(r) for r in a.runs]
    rows = []
    for s in (2026, 2027, 2028, 2029):
        base = Path(a.f2_runs) if s == 2029 else Path("results/neural/runs/sprint_v1/runs/f2_base")
        ref = out_of_fold({k: base / f"seed_{s}" / f"fold_{k}" for k in range(5)}, t)
        for m in ablation_summary.MASKS:
            d = next(r / m for r in roots if (r / m / f"seed_{s}").exists())
            e_m, e_r = (y - out_of_fold({k: d / f"seed_{s}" / f"fold_{k}" for k in range(5)}, t)) ** 2, (y - ref) ** 2
            rows.append(dict(mask=m, seed=s, fold="all", change=e_m.sum() / e_r.sum() - 1))
            rows += [dict(mask=m, seed=s, fold=k, change=e_m[fold == k].sum() / e_r[fold == k].sum() - 1) for k in range(5)]
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--f2-runs", required=True)
    p.add_argument("--resolution", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "f2_path_check.json").write_text(json.dumps(check_f2_path(a), indent=2))
    comps = [summarize(a, [2026, 2027, 2028, 2029], "pooled_2026_2029")]
    comps += [summarize(a, [s], f"seed_{s}") for s in (2026, 2027, 2028, 2029)]
    allc = pd.concat(comps, ignore_index=True)
    allc.to_csv(out / "comparisons_by_analysis.csv", index=False)
    orig = pd.read_csv(a.original).assign(analysis="original_2026_2028")
    rows = []
    for (m, lv), role in [(c, "preregistered confirmation") for c in TRIGGERED] + [(c, "exploratory robustness") for c in EXPLORATORY]:
        o = orig[(orig.model == m) & (orig.level == lv)].iloc[0]
        r = dict(mask=m, level=lv, role=role, original_material=bool(o.material_change), original_change=o.change,
                 original_trigger=bool(o.confirmatory_seed_trigger))
        for name in ("pooled_2026_2029", "seed_2029"):
            x = allc[(allc.analysis == name) & (allc.model == m) & (allc.level == lv)].iloc[0]
            r.update({f"{name}_{k}": x[k] for k in COLS})
        r["classification_changed_pooled"] = r["pooled_2026_2029_material_change"] != r["original_material"]
        r["classification_changed_seed_2029"] = r["seed_2029_material_change"] != r["original_material"]
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "decision_table.csv", index=False)
    pr = per_fold_response(a)
    pr.to_csv(out / "response_change_by_seed_fold.csv", index=False)
    show = ["analysis", "model", "level", "change", "pair_low", "pair_high", "drug_low", "drug_high", "fold_min", "fold_max",
            "p_holm_masks", "material_change"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(allc[allc.model.isin(ablation_summary.MASKS)][show].round(4).to_string(index=False))
        print(pd.DataFrame(rows).T.to_string())
        print(pr[pr.fold == "all"].pivot(index="mask", columns="seed", values="change").round(4))


if __name__ == "__main__":
    main()
