"""ALMANAC validation of the frozen ONEIL networks (configs/almanac_validation.json).

python -m synergy_artifacts.almanac_report --output results/almanac_validation

Same metrics, intervals and verdict rule as the ONEIL benchmark report, on ALMANAC rows with generated
pair folds. Baselines come from `python -m synergy_artifacts.cv run` with configs/almanac_{mean,additive,reference_bliss}.json.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, write_json
from .runner import load_experiment
from .neural_report import contrast_energies, from_sums, index, jackknife_interval, out_of_fold, paired, score
from .suite import run_dir

RUNS = Path("results/neural/runs")
INPUTS = {"f2_combined": "identities, doses, training-pair monotherapy", "f0_combined": "identities, doses",
          "mono_only": "training-pair monotherapy values only (learned, Bliss offset)",
          "bliss_reference": "training-pair monotherapy, fixed rule", "additive": "identities, doses (additive ridge)",
          "mean": "none (training mean)"}


def regime_confirmation(df, t, y, predictions, reference, seeds, declared="configs/almanac_regime_confirmation.json"):
    """Prospective test declared in configs/almanac_regime_confirmation.json (Bliss-level bins only)."""
    from .regimes import add_holm, calibrated_bliss, flatten, regime_masks, regime_metrics, regimes
    read_json(declared)
    region, level = regimes(df, reference)
    calibrated = calibrated_bliss(df, y, reference)
    rows = []
    for model in ["f2_combined", "f0_combined"]:
        own = [predictions[model, s] for s in seeds]
        for against, ref in [("bliss", reference), ("calibrated_bliss", calibrated)]:
            for family, name, mask in regime_masks(y, region, level):
                if family == "dose_region":
                    continue
                rows.append({"model": model, "against": against, "family": family, "regime": name,
                             **regime_metrics(t, mask, y, own, ref)})
    table = add_holm(pd.DataFrame(rows), [{"model": m, "against": a} for m in ["f2_combined", "f0_combined"]
                                          for a in ["bliss", "calibrated_bliss"]])
    primary = table[(table.model == "f2_combined") & (table.against == "bliss") & (table.regime == "R>=75")].iloc[0]
    table["h1_confirmed"] = None
    table.loc[primary.name, "h1_confirmed"] = bool(primary.interaction_residual_r2_jackknife[0] > 0
                                                   and min(primary.interaction_residual_r2_seeds) > 0)
    return flatten(table)


def site_check(output, site_map, protocol_path="configs/almanac_validation.json", suite=RUNS / "almanac_v1"):
    """Post hoc check declared in configs/almanac_site_check.json: beyond-Bliss gain with site main effects removed."""
    from .regimes import flatten, regime_masks, regime_metrics, regimes
    output, protocol = Path(output), read_json(protocol_path)
    df, _, _ = load_experiment({"dataset": protocol["shared_overrides"]["dataset"], "table": "observations", "target": "inhibition",
                                "test_fold": 0, "model": {"kind": "dose"}})
    t = index(df)
    y, seeds = t["y"], protocol["benchmark"]["seeds"]
    sites = pd.read_csv(site_map).set_index("pair_id").screener
    reference = out_of_fold({k: RUNS / "almanac_reference_bliss" / f"fold_{k}" for k in range(5)}, t)
    _, level = regimes(df, reference)
    code = t["panel_frame"].pair_id.map(sites).fillna("NA")
    stratified = dict(t, panel_frame=t["panel_frame"].assign(fold=t["panel_frame"].fold * 10 + code.map({"FF": 1, "FG": 2}).fillna(0).astype(int)))
    row_site = pd.Series(df.pair_id.astype(str).to_numpy()).map(sites).fillna("NA").to_numpy()
    rows = []
    for model in ["f2_combined", "f0_combined"]:
        own = [out_of_fold({k: run_dir(suite, model, s, k) for k in range(5)}, t) for s in seeds]
        masks = regime_masks(y, level, level) if model == "f2_combined" else [("all", "all", np.ones(len(y), bool))]
        for family, name, mask in masks:
            if family == "dose_region":
                continue
            rows.append({"model": model, "site": "within fold, cell and site", "family": family, "regime": name,
                         **regime_metrics(stratified, mask, y, own, reference)})
            for site in ["FF", "FG"]:
                rows.append({"model": model, "site": site, "family": family, "regime": name,
                             **regime_metrics(t, mask & (row_site == site), y, own, reference)})
    table = flatten(pd.DataFrame(rows))
    table.to_csv(output / "site_check.csv", index=False, float_format="%.6g")
    print(table[["model", "site", "regime", "panel_rows", "interaction_residual_r2", "interaction_residual_r2_jackknife",
                 "interaction_residual_r2_seeds"]].to_string())
    return table


def analyze(output, protocol_path="configs/almanac_validation.json", suite=RUNS / "almanac_v1", repeats=1000, seed=2026):
    """Protocols with a "runs" block (third screens) name their suite, baselines and declared tests; ALMANAC uses the defaults."""
    output, protocol = Path(output), read_json(protocol_path)
    dataset = protocol["shared_overrides"]["dataset"]
    runs = protocol.get("runs", {})
    suite = Path(runs.get("suite", suite))
    baselines = runs.get("baselines", {k: str(RUNS / f"almanac_{k}") for k in ["mean", "additive", "reference_bliss"]})
    df, _, _ = load_experiment({"dataset": dataset, "table": "observations", "target": "inhibition", "test_fold": 0,
                                "model": {"kind": "dose"}})
    t = index(df)
    y, seeds = t["y"], protocol["benchmark"]["seeds"]
    folds = lambda base: {k: Path(base) / f"fold_{k}" for k in range(5)}
    predictions = {(name, s): out_of_fold({k: run_dir(suite, name, s, k) for k in range(5)}, t)
                   for name in protocol["candidates"] for s in seeds}
    for name in ["mean", "additive", "reference_bliss"]:
        predictions[name if name != "reference_bliss" else "bliss_reference", None] = out_of_fold(folds(baselines[name]), t)
    reference = predictions["bliss_reference", None]
    status = out_of_fold({k: run_dir(suite, "f2_combined", seeds[0], k) for k in range(5)}, t, "monotherapy_status")
    imputed = status == "mean_imputed"
    r = y - reference
    sums = lambda w: np.bincount(t["pair"], weights=w, minlength=len(t["pairs"]))
    truth_sums = np.column_stack([sums(y), sums(y * y), sums(r), sums(r * r)])
    scores = {k: score(p, t, truth_sums, imputed) for k, p in predictions.items()}
    energy = contrast_energies(t, {"zero": np.zeros(len(t["panel_frame"])), **{k: v["panel_prediction"] for k, v in scores.items()}})
    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(len(t["pairs"]), size=len(t["pairs"])), minlength=len(t["pairs"]))
                        for _ in range(repeats)]).astype(float)
    jackknife = np.array([[d not in pair for pair in t["pair_drugs"]] for d in t["drugs"]], dtype=float)
    models = list(protocol["candidates"]) + ["bliss_reference", "additive", "mean"]
    keys_of = lambda m: [k for k in scores if k[0] == m]
    rows = []
    for model in models:
        keys = keys_of(model)
        own = [scores[k] for k in keys]
        mats = [o["sums"] for o in own]
        row = {"model": model, "inputs": INPUTS[model], "seeds": ";".join(str(k[1]) for k in keys if k[1] is not None) or "none",
               "n": own[0]["pooled"]["n"]}
        for m in ["rmse", "mse", "mae", "r2", "pearson"]:
            values = [o["pooled"][m] for o in own]
            row[m], row[m + "_seed_sd"] = float(np.mean(values)), (float(np.std(values, ddof=1)) if len(values) > 1 else None)
        boot = np.quantile(np.mean([from_sums(weights, x, truth_sums)["rmse"] for x in mats], 0), [.025, .975])
        jack = np.sqrt(np.mean([(jackknife @ x[:, 1]) / (jackknife @ x[:, 0]) for x in mats], 0))
        contrast = np.mean([1 - energy[k] / energy["zero"] for k in keys], 0)
        within = np.mean([1 - energy[k] / energy["bliss_reference", None] for k in keys], 0)
        versus = contrast - (1 - energy["bliss_reference", None] / energy["zero"])
        row.update({"rmse_ci_low": boot[0], "rmse_ci_high": boot[1],
                    "rmse_jackknife_low": jackknife_interval(row["rmse"], jack)[0], "rmse_jackknife_high": jackknife_interval(row["rmse"], jack)[1],
                    "residual_r2": float(np.mean([o["residual_r2"] for o in own])), "panel_rmse": float(np.mean([o["panel_rmse"] for o in own])),
                    "contrast_r2": contrast[0], "contrast_r2_jackknife_low": jackknife_interval(contrast[0], contrast[1:])[0],
                    "contrast_r2_jackknife_high": jackknife_interval(contrast[0], contrast[1:])[1],
                    "contrast_r2_minus_bliss": versus[0], "contrast_r2_minus_bliss_jackknife_low": jackknife_interval(versus[0], versus[1:])[0],
                    "contrast_r2_minus_bliss_jackknife_high": jackknife_interval(versus[0], versus[1:])[1],
                    "interaction_residual_r2": within[0], "interaction_residual_r2_jackknife_low": jackknife_interval(within[0], within[1:])[0],
                    "interaction_residual_r2_jackknife_high": jackknife_interval(within[0], within[1:])[1],
                    "imputed_rmse": float(np.mean([o["imputed_rmse"] for o in own])) if imputed.any() else None,
                    **{f"rmse_fold_{k}": float(np.mean([from_sums((t["pair_fold"] == k).astype(float), x, truth_sums)["rmse"] for x in mats]))
                       for k in range(5)}})
        rows.append(row)
    benchmark = pd.DataFrame(rows)
    mse = benchmark.set_index("model").mse
    comparisons = []
    for model, control, question in protocol["comparisons"]:
        change = paired([scores[k]["sums"] for k in keys_of(model)], [scores[k]["sums"] for k in keys_of(control)], t, weights, jackknife)
        comparisons.append({"model": model, "control": control, "question": question, "model_mse": mse[model], "control_mse": mse[control],
                            "relative_mse_change": change["delta_mse"] / mse[control], **change})
    comparisons = pd.DataFrame(comparisons)
    regimes_table = regime_confirmation(df, t, y, predictions, reference, seeds, runs.get("declared", "configs/almanac_regime_confirmation.json"))
    output.mkdir(parents=True, exist_ok=True)
    for name, table in {"benchmark": benchmark, "comparisons": comparisons, "regimes": regimes_table}.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.10g", lineterminator="\n")
    write_json(output / "summary.json", {
        "protocol": protocol["name"], "dataset": dataset, "rows": len(y), "pairs": len(t["pairs"]), "drugs": len(t["drugs"]),
        "cells": len(t["cells"]), "imputed_rows": int(imputed.sum()), "bootstrap": repeats,
        "monotherapy_status": {str(k): int(v) for k, v in pd.Series(status).value_counts().items()}})
    print(benchmark[["model", "rmse", "rmse_seed_sd", "rmse_ci_low", "rmse_ci_high", "r2", "pearson", "contrast_r2",
                     "interaction_residual_r2", "interaction_residual_r2_jackknife_low", "interaction_residual_r2_jackknife_high"]].to_string())
    print(comparisons[["model", "control", "delta_mse", "ci_low", "ci_high", "jackknife_low", "jackknife_high", "relative_mse_change",
                       "folds_lower", "seed_pairs_lower", "verdict"]].to_string())
    print(regimes_table[["model", "against", "regime", "dose_rows", "panel_rows", "interaction_residual_r2", "interaction_residual_r2_jackknife",
                         "interaction_residual_p_holm", "interaction_residual_r2_seeds", "h1_confirmed"]].to_string())
    return benchmark, comparisons


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True)
    p.add_argument("--protocol", default="configs/almanac_validation.json")
    p.add_argument("--site-map", help="pair_id,screener CSV: run the post hoc site check instead")
    a = p.parse_args()
    if a.site_map:
        site_check(a.output, a.site_map, a.protocol)
    else:
        analyze(a.output, a.protocol)


if __name__ == "__main__":
    main()
