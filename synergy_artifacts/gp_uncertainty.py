"""Predictive uncertainty of GP out-of-fold predictions on the shared test rows.

python -m synergy_artifacts.gp_uncertainty --output results/gp_uncertainty --method gp_his=DIR [--method NAME=DIR ...]

DIR holds fold_<k>.parquet with observation_id, prediction and predictive_std (latent_std optional) for every
test row of fold k. Optional validation_<k>.parquet holds the same columns for the rows of fold (k+1)%5, the
validation fold of model k; only those rows set the variance scale used for recalibration.

Per method, on the 1,475,328 test rows:
- calibration: central-interval coverage and mean width at 50/80/90/95%, PIT histogram, NLL, z mean and SD;
- held-out pairs: coverage of each pair's 90% intervals;
- error versus predicted SD: RMSE within deciles of predicted SD, Spearman of |error| and SD across rows,
  within pair-cell units and across units;
- regimes: coverage, RMSE and mean SD by dose region and Bliss-expected inhibition;
- usefulness: RMSE of the most confident fraction of rows, and the share of the random-to-oracle gain in the
  area under that curve;
- latent SD (kernel posterior only) versus predictive SD (latent plus fitted noise);
- baseline: the same means with a constant SD per cell line and fold, equal to that model's validation RMSE on
  fold (f+1)%5 from selection.csv (validation rows only);
- recalibration, where validation predictions exist: one variance scale per fold from validation rows only;
- the selected neural F2 model's errors (seed-mean prediction) ranked by each GP's predictive SD.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .io import read_json, write_json
from .regimes import regimes
from .neural_report import RUNS, index, load_truth, model_paths, out_of_fold

LEVELS = [0.5, 0.8, 0.9, 0.95]
FRACTIONS = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def load(directory, t, prefix="fold"):
    """Arrays aligned to t's rows; validation files are aligned to the rows of their own fold."""
    columns, out = ["prediction", "predictive_std", "latent_std"], {}
    for k in range(5):
        path = Path(directory) / f"{prefix}_{k}.parquet"
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        rows_fold = k if prefix == "fold" else (k + 1) % 5
        ids = t["ids"][rows_fold]
        if frame.observation_id.duplicated().any() or set(frame.observation_id) != set(ids):
            raise ValueError(f"{path}: must cover its rows once each")
        frame = frame.set_index("observation_id").reindex(ids)
        for c in columns:
            if c in frame:
                out.setdefault(c, np.full(len(t["y"]), np.nan) if prefix == "fold" else {})
                if prefix == "fold":
                    out[c][t["rows"][k]] = frame[c].to_numpy(float)
                else:
                    out[c][k] = frame[c].to_numpy(float)
    return out


def calibration(y, mean, std):
    z = (y - mean) / std
    rows = [{"level": q, "coverage": float(np.mean(np.abs(z) <= norm.ppf(0.5 + q / 2))),
             "mean_width": float(2 * norm.ppf(0.5 + q / 2) * std.mean())} for q in LEVELS]
    pit = np.histogram(norm.cdf(z), bins=10, range=(0, 1))[0] / len(z)
    nll = float(np.mean(0.5 * np.log(2 * np.pi * std ** 2) + 0.5 * z ** 2))
    return rows, pit, {"nll": nll, "z_mean": float(z.mean()), "z_sd": float(z.std()), "mean_std": float(std.mean())}


def within_rank_correlation(groups, a, b):
    """Pooled Spearman correlation after centering ranks within each group."""
    frame = pd.DataFrame({"g": groups, "a": a, "b": b})
    ra = frame.groupby("g").a.rank()
    rb = frame.groupby("g").b.rank()
    ra, rb = ra - ra.groupby(frame.g).transform("mean"), rb - rb.groupby(frame.g).transform("mean")
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def risk(y, mean, score):
    sq = (y - mean) ** 2
    n = len(sq)
    curve = lambda order: np.cumsum(sq[order]) / np.arange(1, n + 1)
    model, oracle = curve(np.argsort(score, kind="stable")), curve(np.argsort(sq, kind="stable"))
    random = sq.mean()
    rows = [{"fraction": f, "rmse": float(np.sqrt(model[max(int(f * n), 1) - 1]))} for f in FRACTIONS]
    gain = float((random - model.mean()) / (random - oracle.mean()))
    return rows, gain


def cell_baseline(directory, t, df):
    """Per-row SD equal to the cell's validation RMSE for the model that predicted it (selected family)."""
    sel = pd.read_csv(Path(directory) / "selection.csv")
    if "predictor" in sel:
        sel = sel[sel.predictor == "exact"]
    if "selected" not in sel:  # earlier adapter layout: one row per cell and fold with both families' RMSE
        sel = sel.assign(selected=True, validation_rmse=[getattr(r, f"validation_rmse_{r.family}") for r in sel.itertuples()])
    rmse = sel[sel.selected].set_index(["cell_line", "fold"]).validation_rmse
    keys = pd.MultiIndex.from_arrays([df.cell_line.astype(str).to_numpy(), df.fold.to_numpy()])
    sd = rmse.reindex(keys).to_numpy(float)
    if not np.isfinite(sd).all():
        raise ValueError(f"{directory}: selection.csv lacks a validation RMSE for some cell and fold")
    return sd


def analyze(t, name, pred, reference, region, level, validation=None, baseline=None, other=None):
    y, mean, std = t["y"], pred["prediction"], pred["predictive_std"]
    err = y - mean
    out = {"method": name}
    cal, pit, extra = calibration(y, mean, std)
    out.update(extra)
    tables = {"calibration": [{"method": name, "sd": "predictive", **r} for r in cal],
              "pit": [{"method": name, "bin": i, "fraction": float(p)} for i, p in enumerate(pit)]}
    if "latent_std" in pred:
        lat, _, lat_extra = calibration(y, mean, np.maximum(pred["latent_std"], 1e-9))
        tables["calibration"] += [{"method": name, "sd": "latent", **r} for r in lat]
        out["latent_mean_std"], out["latent_nll"] = lat_extra["mean_std"], lat_extra["nll"]
        out["latent_vs_predictive_spearman"] = spearman(pred["latent_std"], std)
    z90 = np.abs(err) <= norm.ppf(0.95) * std
    pairs = pd.Series(z90).groupby(t["pair"]).mean()
    out["pair_coverage90"] = {"median": float(pairs.median()), "p10": float(pairs.quantile(0.1)),
                              "below_0.8": float((pairs < 0.8).mean()), "pairs": int(len(pairs))}
    deciles = pd.qcut(std, 10, labels=False, duplicates="drop")
    tables["deciles"] = [{"method": name, "decile": int(d), "mean_std": float(std[deciles == d].mean()),
                          "rmse": float(np.sqrt(np.mean(err[deciles == d] ** 2))),
                          "coverage90": float(z90[deciles == d].mean())} for d in np.unique(deciles)]
    unit = pd.DataFrame({"u": t["panel"], "sq": err ** 2, "s": std}).groupby("u").mean()
    out["spearman_abs_error_sd"] = {"rows": spearman(np.abs(err), std),
                                    "within_pair_cell": within_rank_correlation(t["panel"], np.abs(err), std),
                                    "across_pair_cell": spearman(unit.sq, unit.s)}
    rows, gain = risk(y, mean, std)
    tables["risk"] = [{"method": name, **r} for r in rows]
    out["risk_gain_vs_oracle"] = gain
    tables["regimes"] = []
    for kind, labels in [("dose_region", region), ("bliss_level", level)]:
        for lab in pd.unique(labels):
            m = labels == lab
            tables["regimes"].append({"method": name, "kind": kind, "regime": lab, "rows": int(m.sum()),
                                      "rmse": float(np.sqrt(np.mean(err[m] ** 2))), "mean_std": float(std[m].mean()),
                                      "coverage90": float(z90[m].mean()), "z_sd": float((err[m] / std[m]).std())})
    if validation is not None:
        scaled = np.empty(len(y), bool)
        scales = {}
        for k in range(5):
            vrows = t["rows"][(k + 1) % 5]
            z_val = (y[vrows] - validation["prediction"][k]) / validation["predictive_std"][k]
            scales[k] = float(np.sqrt(np.mean(z_val ** 2)))
            rows_k = t["rows"][k]
            scaled[rows_k] = np.abs(err[rows_k]) <= norm.ppf(0.95) * std[rows_k] * scales[k]
        factor = np.empty(len(y))
        for k in range(5):
            factor[t["rows"][k]] = scales[k]
        rec, _, rec_extra = calibration(y, mean, std * factor)
        tables["calibration"] += [{"method": name, "sd": "predictive, validation-scaled", **r} for r in rec]
        out["recalibrated"] = {"validation_scale": scales, "test_coverage90": float(scaled.mean()), "nll": rec_extra["nll"]}
    if baseline is not None:
        base, _, base_extra = calibration(y, mean, baseline)
        tables["calibration"] += [{"method": name, "sd": "validation RMSE per cell", **r} for r in base]
        _, base_gain = risk(y, mean, baseline)
        out["cell_baseline"] = {"nll": base_extra["nll"], "z_sd": base_extra["z_sd"], "risk_gain_vs_oracle": base_gain,
                                "spearman_abs_error_sd_rows": spearman(np.abs(err), baseline)}
    if other is not None:
        other_name, other_mean = other
        other_err = y - other_mean
        rows_o, gain_o = risk(y, other_mean, std)
        out[f"ranks_{other_name}_errors"] = {
            "spearman_rows": spearman(np.abs(other_err), std),
            "spearman_within_pair_cell": within_rank_correlation(t["panel"], np.abs(other_err), std),
            "risk_gain_vs_oracle": gain_o, "rmse_most_confident_half": next(r["rmse"] for r in rows_o if r["fraction"] == 0.5),
            "rmse_all": float(np.sqrt(np.mean(other_err ** 2)))}
    return out, tables


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True)
    p.add_argument("--method", action="append", required=True, help="NAME=DIR")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    df, _, _ = load_truth("data/processed/oneil_v1")
    t = index(df)
    suite, protocol = RUNS / "benchmark_v1", read_json("configs/neural_benchmark.json")
    chosen = read_json(suite / "selection.json")["selected"]
    reference = out_of_fold(model_paths("f2_residual", protocol["selection"]["seed"], chosen, suite), t, "reference_offset")
    region, level = regimes(df, reference)
    seeds = protocol["benchmark"]["seeds"]
    f2 = np.mean([out_of_fold(model_paths("selected:F2", s, chosen, suite), t) for s in seeds], axis=0)
    summaries, tables = [], {}
    for spec in a.method:
        name, directory = spec.split("=", 1)
        pred = load(directory, t)
        if pred is None or "predictive_std" not in pred:
            raise ValueError(f"{directory}: needs fold_<k>.parquet with predictive_std")
        baseline = cell_baseline(directory, t, df) if (Path(directory) / "selection.csv").exists() else None
        summary, tab = analyze(t, name, pred, reference, region, level, load(directory, t, "validation"), baseline,
                               ("selected_f2_seed_mean", f2))
        summaries.append(summary)
        for key, rows in tab.items():
            tables.setdefault(key, []).extend(rows)
    for key, rows in tables.items():
        pd.DataFrame(rows).to_csv(out / f"{key}.csv", index=False, float_format="%.6g", lineterminator="\n")
    write_json(out / "summary.json", summaries)
    print(json.dumps(summaries, indent=1)[:4000])


if __name__ == "__main__":
    main()
