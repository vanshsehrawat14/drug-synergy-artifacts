"""Summarize the GP validation study and apply the selection rule frozen in configs/gp_study.json.

python -m synergy_artifacts.gp_select --config configs/gp_study.json --study results/neural/runs/gp_study --output results/gp

Per method, each cell line uses its selected family (lowest validation RMSE, his rule). Pooled validation MSE
is the primary metric. Production cost is the measured study seconds over both families and all cells,
times five folds, divided by the study's worker count. Only validation rows are read.

Amendment configs/gp_study_amendment.json removes cap6000 exact from the production candidates (cost) and runs it
validation-only on eight cells; same_cells.csv compares every method on those cells.

Amendment configs/gp_study_amendment2.json applies the frozen kernel-variant clause that decide() only reported:
the best variant that beats the default by more than 1% at cap 1500 is validated at the selected cap and predictor
and replaces the default there if it still lowers validation MSE by more than 1%. finalize() writes decision_final.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import digest, write_json

BUDGET_HOURS = 8.0
TIE = 0.01
CANDIDATES = [("cap1500", "exact"), ("cap1500", "inducing"), ("cap3000", "exact"), ("cap3000", "inducing")]
VARIANTS = [("cap1500_w0", "exact"), ("cap1500_rho0", "exact"), ("cap1500_rho05", "exact")]


def summarize(study):
    study = Path(study)
    rows, cells = [], []
    for csv in sorted(study.glob("*.csv")):
        if csv.name.endswith(".partial.csv"):
            continue
        frame = pd.read_csv(csv)
        workers = json.loads(csv.with_suffix(".json").read_text())["workers"]
        cells.append(frame)
        for predictor, own in frame.groupby("predictor"):
            chosen = own[own.selected]
            if chosen.cell_line.duplicated().any():
                raise ValueError(f"{csv.name}: more than one selected family per cell")
            n = chosen.validation_n.to_numpy(float)
            seconds = own.seconds.sum()
            rows.append({"group": csv.stem, "predictor": predictor, "cells": len(chosen),
                         "training_rows_hyperparameters": float(chosen.n_train.median()),
                         "training_rows_conditioned": float(chosen.n_train_available.median() if predictor == "inducing" else chosen.n_train.median()),
                         "validation_rows": int(n.sum()), "validation_mse": float(chosen.validation_sse.sum() / n.sum()),
                         "validation_rmse": float(np.sqrt(chosen.validation_sse.sum() / n.sum())),
                         "validation_nll": float(np.average(chosen.validation_nll, weights=n)),
                         "validation_coverage90": float(np.average(chosen.validation_coverage90, weights=n)),
                         "matern_cells": int((chosen.family == "matern").sum()),
                         "seconds_both_families": float(seconds), "median_seconds_per_fit": float(own.seconds.median()),
                         "peak_mb_max": float(own.peak_mb.max()), "convergence_warnings": int(own.warnings.sum()),
                         "hyperparameters_at_bound": int(own.at_bound.sum()), "study_workers": workers,
                         "production_hours_estimate": float(seconds * 5 / workers / 3600)})
    return pd.DataFrame(rows), pd.concat(cells, ignore_index=True)


def same_cells(cells, subset="cap6000_subset"):
    """Pooled validation MSE of every method on the cap6000 subset's cells, each cell at its selected family."""
    chosen = cells[cells.selected]
    names = set(chosen[chosen.group == subset].cell_line)
    rows = []
    for (group, predictor), g in chosen[chosen.cell_line.isin(names)].groupby(["group", "predictor"]):
        rows.append({"group": group, "predictor": predictor, "cells": g.cell_line.nunique(),
                     "validation_mse": g.validation_sse.sum() / g.validation_n.sum(),
                     "validation_rmse": float(np.sqrt(g.validation_sse.sum() / g.validation_n.sum())),
                     "training_rows_conditioned": float((g.n_train_available if predictor == "inducing" else g.n_train).median())})
    return pd.DataFrame(rows)


def decide(summary, plan):
    table = summary.set_index(["group", "predictor"])
    missing = [c for c in CANDIDATES + VARIANTS if c not in table.index]
    if missing:
        raise ValueError(f"Study incomplete: {missing}")
    cand = table.loc[CANDIDATES].copy()
    cand["eligible"] = cand.production_hours_estimate <= BUDGET_HOURS
    eligible = cand[cand.eligible]
    best = eligible.validation_mse.idxmin()
    near = eligible[eligible.validation_mse <= eligible.validation_mse.min() * (1 + TIE)]
    chosen = near.production_hours_estimate.idxmin()
    base = table.loc[("cap1500", "exact")].validation_mse
    variants = {g: {"validation_mse": float(table.loc[(g, p)].validation_mse),
                    "change_vs_his_kernel": float(table.loc[(g, p)].validation_mse / base - 1)} for g, p in VARIANTS}
    variant_wins = [g for g, v in variants.items() if v["change_vs_his_kernel"] < -TIE]
    group, predictor = chosen
    method = {**plan["groups"][group]["fit"], "predictor": predictor}
    return {"rule": plan["selection"], "lowest_mse_eligible": list(best), "selected": [group, predictor], "method": method,
            "kernel_variants": variants, "kernel_variant_beats_default_by_more_than_1pct": variant_wins,
            "candidates": json.loads(cand.reset_index().to_json(orient="records"))}


def finalize(summary, decision, amendment):
    """The frozen variant clause at the selected scale (amendment 2)."""
    table = summary.set_index(["group", "predictor"])
    group, predictor = decision["selected"]
    at_scale = amendment["variant_at_scale"]
    base, variant = table.loc[(group, predictor)].validation_mse, table.loc[(at_scale, predictor)].validation_mse
    change = float(variant / base - 1)
    replace = change < -TIE
    method = {**decision["method"], **(amendment["groups"][at_scale]["fit"] if replace else {})}
    return {"amendment": amendment["rule"], "selected_scale": [group, predictor], "variant": decision_variant(decision),
            "variant_group": at_scale, "default_validation_mse": float(base), "variant_validation_mse": float(variant),
            "change": change, "variant_replaces_default": bool(replace), "method": method}


def decision_variant(decision):
    wins = decision["kernel_variant_beats_default_by_more_than_1pct"]
    return min(wins, key=lambda g: decision["kernel_variants"][g]["validation_mse"]) if wins else None


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/gp_study.json")
    p.add_argument("--study", default="results/neural/runs/gp_study")
    p.add_argument("--output", default="results/gp")
    p.add_argument("--summary-only", action="store_true", help="summarize a partial study without deciding")
    p.add_argument("--amendment", help="configs/gp_study_amendment2.json: apply the variant clause to decision.json")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    summary, cells = summarize(a.study)
    summary.to_csv(out / "study_summary.csv", index=False, float_format="%.6g", lineterminator="\n")
    cells.drop(columns=["kernel"]).assign(kernel=cells.kernel).to_csv(out / "study_cells.csv", index=False, float_format="%.6g", lineterminator="\n")
    subset = same_cells(cells)
    if len(subset):
        subset.to_csv(out / "same_cells.csv", index=False, float_format="%.6g", lineterminator="\n")
        print(subset.round(3).to_string())
    print(summary[["group", "predictor", "training_rows_conditioned", "validation_rmse", "validation_nll", "validation_coverage90",
                   "median_seconds_per_fit", "peak_mb_max", "production_hours_estimate"]].round(3).to_string())
    if a.summary_only:
        return
    if a.amendment:
        final = finalize(summary, json.loads((out / "decision.json").read_text()), json.loads(Path(a.amendment).read_text()))
        final["amendment_sha256"] = digest(a.amendment)
        write_json(out / "decision_final.json", final)
        print(json.dumps({k: final[k] for k in ["variant_group", "change", "variant_replaces_default", "method"]}))
        return
    decision = decide(summary, json.loads(Path(a.config).read_text()))
    decision["config_sha256"] = digest(a.config)
    write_json(out / "decision.json", decision)
    print(json.dumps({k: decision[k] for k in ["lowest_mse_eligible", "selected", "method", "kernel_variant_beats_default_by_more_than_1pct"]}))


if __name__ == "__main__":
    main()
