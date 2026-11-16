"""Breakdowns of the shared comparison on identical rows: per cell line, per drug, by Bliss level, and cost.

python -m synergy_artifacts.neural_report verify --output results/unified
python -m synergy_artifacts.neural_report analyze --output results/unified --method gp=DIR [--method cf=DIR]
python -m synergy_artifacts.unified --output results/unified --method gp=DIR [--method cf=DIR]

Pooled metrics, grouped intervals and paired comparisons come from neural_report analyze. This module adds:
- metrics per cell line and per drug;
- MSE and interaction residual R² by Bliss-expected inhibition;
- training cost and information requirements.

A team method enters through the shared contract: DIR/fold_<k>.parquet with observation_id and prediction for
every test row of fold k. A run.json and selection.csv beside the folds, written by synergy_artifacts.gp_shared, add cost.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json
from .regimes import BLISS_BINS, regime_metrics, regimes
from .neural_report import CPU, DOSE, INPUTS, RUNS, index, load_truth, model_paths, out_of_fold, team_predictions

CORE = ["selected:F2", "selected:F0", DOSE, "additive", "bliss_reference"]
LEVELS = ["R<25", "25-50", "50-75", "R>=75"]


def load_models(t, methods, suite=RUNS / "benchmark_v1", protocol_path="configs/neural_benchmark.json"):
    """Out-of-fold predictions per model, one vector per seed."""
    protocol = read_json(protocol_path)
    chosen = read_json(Path(suite) / "selection.json")["selected"]
    seeds = protocol["benchmark"]["seeds"]
    reference = out_of_fold(model_paths("f2_residual", protocol["selection"]["seed"], chosen, suite), t, "reference_offset")
    dose_seeds = read_json(RUNS / DOSE / "protocol.json")["protocol"]["seeds"]
    models = {m: [out_of_fold(model_paths(m, s, chosen, suite), t) for s in seeds] for m in ["selected:F2", "selected:F0"]}
    models[DOSE] = [out_of_fold(model_paths(DOSE, s, chosen, suite), t) for s in dose_seeds]
    models["additive"] = [out_of_fold(model_paths("additive", None, chosen, suite), t)]
    models["bliss_reference"] = [reference]
    for name, directory in methods.items():
        models[name] = [team_predictions(directory, t)]
    return models, reference


def grouped(t, codes, names, models, level):
    """RMSE, R² and Pearson within each group, averaged over seeds."""
    rows = []
    y = t["y"]
    for g, name in enumerate(names):
        rows_g = codes == g if codes.ndim == 1 else codes[:, g]
        if not rows_g.any():
            continue
        yg = y[rows_g]
        for model, preds in models.items():
            stats = []
            for p in preds:
                e = yg - p[rows_g]
                stats.append([np.mean(e * e), 1 - np.sum(e * e) / np.sum((yg - yg.mean()) ** 2), np.corrcoef(yg, p[rows_g])[0, 1]])
            mse, r2, pearson = np.mean(stats, 0)
            rows.append({"level": level, "group": name, "model": model, "n": int(rows_g.sum()), "rmse": float(np.sqrt(mse)),
                         "mse": float(mse), "r2": float(r2), "pearson": float(pearson)})
    return pd.DataFrame(rows)


def by_drug(t, models):
    df = t["df"]
    a, b = df.drug_a.astype(str).to_numpy(), df.drug_b.astype(str).to_numpy()
    member = np.column_stack([(a == d) | (b == d) for d in t["drugs"]])
    return grouped(t, member, t["drugs"], models, "drug")


def by_level(t, models, reference):
    """Response MSE and interaction residual R² against Bliss within each Bliss-expected inhibition bin."""
    _, level = regimes(t["df"], reference)
    rows = []
    for name in ["all"] + LEVELS:
        mask = np.ones(len(t["y"]), bool) if name == "all" else level == name
        bliss_mse = float(np.mean((t["y"][mask] - reference[mask]) ** 2))
        for model, preds in models.items():
            mse = float(np.mean([np.mean((t["y"][mask] - p[mask]) ** 2) for p in preds]))
            row = {"bin": name, "model": model, "rows": int(mask.sum()), "rmse": float(np.sqrt(mse)), "mse": mse,
                   "mse_change_vs_bliss": mse / bliss_mse - 1}
            if model != "bliss_reference":
                m = regime_metrics(t, mask, t["y"], preds, reference)
                row.update({"interaction_residual_r2": m["interaction_residual_r2"],
                            "interaction_residual_r2_low": m["interaction_residual_r2_jackknife"][0],
                            "interaction_residual_r2_high": m["interaction_residual_r2_jackknife"][1],
                            "seeds_positive": int(sum(v > 0 for v in m["interaction_residual_r2_seeds"])), "seeds": len(preds),
                            "contrast_r2_model": m["contrast_r2_model"], "contrast_r2_bliss": m["contrast_r2_bliss"]})
            rows.append(row)
    return pd.DataFrame(rows)


def cost(methods, runs_csv="results/neural_benchmark/runs.csv"):
    """Information requirements, model size and measured training cost per out-of-fold benchmark."""
    runs = pd.read_csv(runs_csv)
    rows = []
    for model, label in [("selected:F2", "selected:F2"), ("selected:F0", "selected:F0"), (DOSE, DOSE)]:
        own = runs[runs.benchmark_models.fillna("").str.split(";").apply(lambda x: label in x)]
        per_seed = own.groupby("seed").fit_seconds.sum()
        rows.append({"model": model, "inputs": INPUTS.get(model), "unit": "one network per fold, all cell lines",
                     "parameters": int(own.parameters.median()), "training_rows_per_fit": "all training-fold rows",
                     "fit_seconds_five_folds": float(per_seed.mean()), "hardware": ";".join(sorted(own.device.dropna().astype(str).unique()))})
    rows.append({"model": "additive", "inputs": INPUTS.get("additive"), "unit": "one ridge per fold", "parameters": None,
                 "training_rows_per_fit": "all training-fold rows", "fit_seconds_five_folds": None, "hardware": "CPU"})
    rows.append({"model": "bliss_reference", "inputs": INPUTS.get("bliss_reference"), "unit": "fixed rule, no fit", "parameters": 0,
                 "training_rows_per_fit": "training-pair monotherapy", "fit_seconds_five_folds": 0.0, "hardware": "CPU"})
    for name, directory in methods.items():
        record, selection = Path(directory) / "run.json", Path(directory) / "selection.csv"
        row = {"model": name, "inputs": "team method", "unit": None, "parameters": None, "training_rows_per_fit": None,
               "fit_seconds_five_folds": None, "hardware": None}
        if record.exists() and read_json(record).get("model_family") == "neighborhood_cf":
            run, sel = read_json(record), pd.read_csv(selection)
            dose = run["settings"]["dose"] != "none"
            row.update({"inputs": "identities" + (", dose ranks" if dose else ""), "unit": "one model per fold, all cell lines",
                        "parameters": "38 x 39 additive ridge, training residual matrix" + (", dose profile per cell line" if dose else ""),
                        "training_rows_per_fit": "training-pair pair-cell means" + (" and training rows" if dose else ""),
                        "fit_seconds_five_folds": float(sel.seconds.sum()), "hardware": "CPU"})
        elif record.exists() and selection.exists() and "n_rows" in pd.read_csv(selection, nrows=1):
            # synergy_artifacts.gp_gpu: exact GP on every training row, both families timed
            run, sel = read_json(record), pd.read_csv(selection)
            sel = sel[sel.predictor == "exact"]
            chosen, settings = sel[sel.selected], run["settings"]
            row.update({"inputs": "identities, doses" + (", training-pair monotherapy" if run.get("regime") == "F2" else ""),
                        "unit": "one GP per cell line and fold",
                        "parameters": f"{4 if settings.get('monotherapy') else 3} kernel hyperparameters per GP; stores its training inputs",
                        "training_rows_per_fit": f"{chosen.n_rows.median():.0f} rows ({chosen.n_inputs.median():.0f} distinct inputs), exact",
                        "fit_seconds_five_folds": float((sel.fit_seconds + sel.predict_seconds).sum()), "hardware": run.get("device", "GPU")})
        elif record.exists() and selection.exists():
            run, sel = read_json(record), pd.read_csv(selection)
            chosen = sel[sel.selected] if "selected" in sel else sel
            settings = run["settings"]
            predictor = settings.get("predictor", "exact")
            rows_used = (f"{chosen.n_train.median():.0f} exact" if predictor == "exact"
                         else f"{chosen.n_train.median():.0f} for hyperparameters, {chosen.n_train_available.median():.0f} conditioned")
            row.update({"inputs": "identities, doses", "unit": "one GP per cell line and fold",
                        "parameters": "3 kernel hyperparameters per GP; stores its training inputs",
                        "training_rows_per_fit": rows_used, "fit_seconds_five_folds": float(sel.get("seconds", sel.get("fit_seconds")).sum()),
                        "hardware": "CPU"})
        rows.append(row)
    return pd.DataFrame(rows)


def run(output, methods):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df, _, _ = load_truth("data/processed/oneil_v1")
    t = index(df)
    models, reference = load_models(t, methods)
    cells = grouped(t, t["cell"], t["cells"], models, "cell")
    drugs = by_drug(t, models)
    pd.concat([cells, drugs]).to_csv(output / "breakdown.csv", index=False, float_format="%.6g", lineterminator="\n")
    levels = by_level(t, models, reference)
    levels.to_csv(output / "bliss_levels.csv", index=False, float_format="%.6g", lineterminator="\n")
    cost(methods).to_csv(output / "cost.csv", index=False, float_format="%.6g", lineterminator="\n")
    print(levels.pivot(index="model", columns="bin", values="rmse").round(2).to_string())
    print(levels.pivot(index="model", columns="bin", values="interaction_residual_r2").round(3).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default="results/unified")
    p.add_argument("--method", action="append", default=[], metavar="NAME=DIR")
    a = p.parse_args()
    run(a.output, dict(m.split("=", 1) for m in a.method))


if __name__ == "__main__":
    main()
