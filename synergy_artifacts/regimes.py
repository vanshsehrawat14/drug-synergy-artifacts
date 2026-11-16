"""Beyond-Bliss signal by dose regime, Bliss level, drug and pair, from existing out-of-fold predictions.

python -m synergy_artifacts.regimes --output results/beyond_bliss_regimes

Declared in configs/beyond_bliss_regimes.json before any regime-level number was computed.
Exploratory: the pooled results were already known. Nothing is refit or retuned.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as student

from .io import read_json, write_json
from .neural_report import RUNS, index, jackknife_interval, load_truth, model_paths, out_of_fold

BLISS_BINS = [-np.inf, 25, 50, 75, np.inf]


def regimes(df, reference):
    """Row-level regime labels from inputs only: dose ranks within the block and the Bliss expectation."""
    ra = df.groupby("block_id", observed=True).dose_a.rank(method="dense").to_numpy()
    rb = df.groupby("block_id", observed=True).dose_b.rank(method="dense").to_numpy()
    region = np.where((ra <= 2) & (rb <= 2), "low", np.where((ra >= 3) & (rb >= 3), "high", "mixed"))
    level = pd.cut(reference, BLISS_BINS, right=False, labels=["R<25", "25-50", "50-75", "R>=75"]).astype(str)
    return region, np.asarray(level)


def projected(frame, values, drugs):
    """Residuals of each column after removing drug main effects within fold and cell, with every drug deleted in turn.

    Returns energies (1 + len(drugs), columns) and the all-drug per-row residuals (rows, columns).
    """
    energy = np.zeros((1 + len(drugs), values.shape[1]))
    rows_residual = np.zeros_like(values)
    for _, g in frame.groupby(["fold", "cell_line"], sort=True):
        rows, a, b = g.index.to_numpy(), g.drug_a.to_numpy(), g.drug_b.to_numpy()
        for j, drop in enumerate([None] + drugs):
            keep = np.ones(len(g), bool) if drop is None else (a != drop) & (b != drop)
            if not keep.any():
                continue
            lookup = {x: i for i, x in enumerate(sorted(set(a[keep]) | set(b[keep])))}
            h = np.zeros((keep.sum(), len(lookup)))
            ix = np.arange(keep.sum())
            np.add.at(h, (ix, [lookup[x] for x in a[keep]]), 1)
            np.add.at(h, (ix, [lookup[x] for x in b[keep]]), 1)
            z = values[rows[keep]]
            residual = z - h @ np.linalg.lstsq(h, z, rcond=None)[0]
            energy[j] += (residual ** 2).sum(0)
            if drop is None:
                rows_residual[rows] = residual
    return energy, rows_residual


def jackknife_p(full, replicates):
    replicates = np.asarray(replicates, float)
    n = len(replicates)
    se = np.sqrt((n - 1) / n * ((replicates - replicates.mean()) ** 2).sum())
    return float(2 * student.sf(abs(full) / se, n - 1)) if se > 0 else float("nan")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def panel_of(t, mask, columns):
    """Pair-cell means of each column over the dose rows in mask."""
    panel = t["panel"][mask]
    ids, inverse = np.unique(panel, return_inverse=True)
    n = np.bincount(inverse)
    means = np.column_stack([np.bincount(inverse, weights=c[mask]) / n for c in columns])
    frame = t["panel_frame"].iloc[ids].reset_index(drop=True)
    return frame, means


def regime_metrics(t, mask, y, predictions, reference):
    """Seed-averaged energies; delete-one-drug jackknife over the 38 drugs."""
    columns = [y - p for p in predictions] + [y - reference, y]
    frame, means = panel_of(t, mask, columns)
    energy, _ = projected(frame, means, t["drugs"])
    k = len(predictions)
    e_p, e_r, e_y = energy[:, :k].mean(1), energy[:, k], energy[:, k + 1]
    valid = e_r[1:] > 0
    irr = 1 - e_p / e_r
    contrast_p, contrast_r = 1 - e_p / e_y, 1 - e_r / e_y
    sse_p = np.mean([np.sum((y[mask] - p[mask]) ** 2) for p in predictions])
    out = {"dose_rows": int(mask.sum()), "panel_rows": len(frame), "interaction_residual_r2": float(irr[0]),
           "interaction_residual_r2_seeds": [float(1 - energy[0, s] / e_r[0]) for s in range(k)],
           "interaction_residual_r2_jackknife": [float(x) for x in jackknife_interval(irr[0], irr[1:][valid])],
           "interaction_residual_p": jackknife_p(irr[0], irr[1:][valid]),
           "contrast_r2_model": float(contrast_p[0]), "contrast_r2_bliss": float(contrast_r[0]),
           "contrast_r2_difference": float(contrast_p[0] - contrast_r[0]),
           "contrast_r2_difference_jackknife": [float(x) for x in jackknife_interval(contrast_p[0] - contrast_r[0], (contrast_p - contrast_r)[1:][valid])],
           "response_residual_r2": float(1 - sse_p / np.sum((y[mask] - reference[mask]) ** 2)),
           "bliss_interaction_share": float(e_r[0] / e_y[0])}
    return out


def pair_reliability(t, y, predictions, reference, seed=2026):
    """Per-pair interaction gains on two random cell halves and their rank agreement."""
    frame, means = panel_of(t, np.ones(len(y), bool), [y - p for p in predictions] + [y - reference])
    _, residual = projected(frame, means, [])
    gain_rows = residual[:, -1] ** 2 - (residual[:, :-1] ** 2).mean(1)
    cells = np.array(sorted(frame.cell_line.unique()))
    half = set(np.random.default_rng(seed).permutation(cells)[:len(cells) // 2])
    frame = frame.assign(gain=gain_rows, bliss=residual[:, -1] ** 2, half=frame.cell_line.isin(half).map({True: "A", False: "B"}))
    by = frame.groupby(["pair_id", "half"]).gain.sum().unstack()
    pairs = frame.drop_duplicates("pair_id").set_index("pair_id")[["drug_a", "drug_b"]].loc[by.index]
    rho = spearmanr(by.A, by.B).statistic
    replicates = [spearmanr(by.A[keep], by.B[keep]).statistic for keep in
                  [((pairs.drug_a != d) & (pairs.drug_b != d)).to_numpy() for d in t["drugs"]]]
    per_pair = frame.groupby(["pair_id", "drug_a", "drug_b"]).agg(gain=("gain", "sum"), bliss_energy=("bliss", "sum")).reset_index()
    per_pair = per_pair.merge(by.rename(columns={"A": "gain_half_a", "B": "gain_half_b"}), left_on="pair_id", right_index=True)
    per_pair["relative_gain"] = per_pair.gain / per_pair.bliss_energy
    agree = ((per_pair.gain_half_a > 0) == (per_pair.gain_half_b > 0)).mean()
    return {"pairs": len(by), "spearman_halves": float(rho), "spearman_jackknife": [float(x) for x in jackknife_interval(rho, replicates)],
            "sign_agreement": float(agree), "pairs_gain_both_halves": int(((per_pair.gain_half_a > 0) & (per_pair.gain_half_b > 0)).sum()),
            "pairs_loss_both_halves": int(((per_pair.gain_half_a < 0) & (per_pair.gain_half_b < 0)).sum())}, per_pair, frame


def drug_subsets(t, frame, draws=2000, seed=2026):
    rng = np.random.default_rng(seed)
    rows = []
    for d in t["drugs"]:
        g = frame[(frame.drug_a == d) | (frame.drug_b == d)]
        by = g.groupby("pair_id").agg(gain=("gain", "sum"), bliss=("bliss", "sum"))
        value = by.gain.sum() / by.bliss.sum()
        idx = rng.integers(len(by), size=(draws, len(by)))
        boot = by.gain.to_numpy()[idx].sum(1) / by.bliss.to_numpy()[idx].sum(1)
        p = min(1., 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
        rows.append({"drug": d, "pairs": len(by), "interaction_residual_r2": value, "ci_low": np.quantile(boot, .025),
                     "ci_high": np.quantile(boot, .975), "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    return out


def calibrated_bliss(df, y, reference):
    """Addendum control: isotonic recalibration of the Bliss rule, fitted per test fold on its three training folds."""
    from sklearn.isotonic import IsotonicRegression
    fold, out = df.fold.to_numpy(), np.full(len(y), np.nan)
    for k in range(5):
        train = (fold != k) & (fold != (k + 1) % 5)
        model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(reference[train], y[train])
        out[fold == k] = model.predict(reference[fold == k])
    return out


def regime_masks(y, region, level):
    return ([("all", "all", np.ones(len(y), bool))] + [("dose_region", r, region == r) for r in ["low", "mixed", "high"]]
            + [("bliss_level", b, level == b) for b in ["R<25", "25-50", "50-75", "R>=75"]])


def add_holm(table, keys):
    for key in keys:
        tested = (table.family != "all").to_numpy()
        for c, v in key.items():
            tested &= (table[c] == v).to_numpy()
        table.loc[tested, "interaction_residual_p_holm"] = holm(table.loc[tested, "interaction_residual_p"])
    return table


def flatten(table):
    flat = table.copy()
    for c in ["interaction_residual_r2_jackknife", "contrast_r2_difference_jackknife", "interaction_residual_r2_seeds"]:
        flat[c] = flat[c].map(lambda v: ";".join(f"{x:.4f}" for x in v))
    return flat


def addendum(output, suite=RUNS / "benchmark_v1", protocol_path="configs/neural_benchmark.json", dataset="data/processed/oneil_v1",
             declared="configs/beyond_bliss_regimes.json"):
    """Post hoc control declared in configs/beyond_bliss_regimes_addendum.json."""
    output, plan, protocol = Path(output), read_json(declared), read_json(protocol_path)
    output.mkdir(parents=True, exist_ok=True)
    chosen = read_json(Path(suite) / "selection.json")["selected"]
    df, _, _ = load_truth(dataset)
    t = index(df)
    y = t["y"]
    reference = out_of_fold(model_paths("f2_residual", protocol["selection"]["seed"], chosen, suite), t, "reference_offset")
    region, level = regimes(df, reference)
    calibrated = calibrated_bliss(df, y, reference)
    rows = [{"model": "calibrated_bliss", "against": "bliss", "family": f, "regime": n,
             **regime_metrics(t, m, y, [calibrated], reference)} for f, n, m in regime_masks(y, region, level)]
    for model in plan["models"]:
        predictions = [out_of_fold(model_paths(model, s, chosen, suite), t) for s in plan["seeds"]]
        rows += [{"model": model, "against": "calibrated_bliss", "family": f, "regime": n,
                  **regime_metrics(t, m, y, predictions, calibrated)} for f, n, m in regime_masks(y, region, level)]
    keys = [{"model": "calibrated_bliss", "against": "bliss"}] + [{"model": m, "against": "calibrated_bliss"} for m in plan["models"]]
    table = add_holm(pd.DataFrame(rows), keys)
    flat = flatten(table)
    flat.to_csv(output / "addendum_calibrated_bliss.csv", index=False, float_format="%.6g")
    print(flat[["model", "against", "family", "regime", "interaction_residual_r2", "interaction_residual_r2_jackknife",
                "interaction_residual_p_holm", "interaction_residual_r2_seeds", "response_residual_r2"]].to_string())
    return table


def addendum_mono(output, suite=RUNS / "benchmark_v1", protocol_path="configs/neural_benchmark.json", dataset="data/processed/oneil_v1",
                  declared="configs/beyond_bliss_regimes.json"):
    """Post hoc control declared in configs/beyond_bliss_regimes_addendum2.json: the identity-free mono_only network."""
    output, plan, protocol = Path(output), read_json(declared), read_json(protocol_path)
    chosen = read_json(Path(suite) / "selection.json")["selected"]
    df, _, _ = load_truth(dataset)
    t = index(df)
    y = t["y"]
    reference = out_of_fold(model_paths("f2_residual", protocol["selection"]["seed"], chosen, suite), t, "reference_offset")
    region, level = regimes(df, reference)
    mono = [out_of_fold(model_paths("mono_only", s, chosen, suite), t) for s in plan["seeds"]]
    f2 = [out_of_fold(model_paths("selected:F2", s, chosen, suite), t) for s in plan["seeds"]]
    rows = [{"model": "mono_only", "against": "bliss", "family": f, "regime": n, **regime_metrics(t, m, y, mono, reference)}
            for f, n, m in regime_masks(y, region, level)]
    # F2 against mono_only, seed by seed (seed s of F2 against seed s of mono_only), then averaged energies.
    for f, n, m in regime_masks(y, region, level):
        per_seed = [regime_metrics(t, m, y, [p], q) for p, q in zip(f2, mono)]
        frame, means = panel_of(t, m, [y - p for p in f2] + [y - q for q in mono])
        energy, _ = projected(frame, means, t["drugs"])
        k = len(f2)
        irr = 1 - energy[:, :k].mean(1) / energy[:, k:].mean(1)
        valid = energy[1:, k:].mean(1) > 0
        rows.append({"model": "selected:F2", "against": "mono_only", "family": f, "regime": n, "panel_rows": len(frame),
                     "interaction_residual_r2": float(irr[0]), "interaction_residual_r2_seeds": [x["interaction_residual_r2"] for x in per_seed],
                     "interaction_residual_r2_jackknife": [float(v) for v in jackknife_interval(irr[0], irr[1:][valid])],
                     "interaction_residual_p": jackknife_p(irr[0], irr[1:][valid]), "contrast_r2_difference_jackknife": [np.nan, np.nan],
                     "response_residual_r2": float(np.mean([1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - q[m]) ** 2) for p, q in zip(f2, mono)]))})
    keys = [{"model": "mono_only", "against": "bliss"}, {"model": "selected:F2", "against": "mono_only"}]
    table = add_holm(pd.DataFrame(rows), keys)
    flat = flatten(table)
    flat.to_csv(output / "addendum_mono_only.csv", index=False, float_format="%.6g")
    print(flat[["model", "against", "family", "regime", "interaction_residual_r2", "interaction_residual_r2_jackknife",
                "interaction_residual_p_holm", "interaction_residual_r2_seeds", "response_residual_r2"]].to_string())
    return table


def figure(output):
    """Interaction residual R2 by Bliss level with drug-jackknife intervals (pip install -e ".[plots]")."""
    import matplotlib.pyplot as plt
    output = Path(output)
    main, mono = pd.read_csv(output / "regimes.csv"), pd.read_csv(output / "addendum_mono_only.csv")
    bins = ["R<25", "25-50", "50-75", "R>=75"]
    series = [("selected F2 vs Bliss", main[main.model == "selected:F2"], "#2a78d6", "o"),
              ("selected F2 vs mono_only network", mono[(mono.model == "selected:F2") & (mono.against == "mono_only")], "#eb6834", "s"),
              ("selected F0 vs Bliss", main[main.model == "selected:F0"], "#1baf7a", "^")]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axhline(0, color="#52514e", lw=1, ls="--", zorder=1)
    for i, (label, frame, color, marker) in enumerate(series):
        frame = frame[frame.family == "bliss_level"].set_index("regime").loc[bins]
        low, high = np.array([[float(x) for x in v.split(";")] for v in frame.interaction_residual_r2_jackknife]).T
        x = np.arange(len(bins)) + (i - 1) * .18
        ax.errorbar(x, frame.interaction_residual_r2, yerr=[frame.interaction_residual_r2 - low, high - frame.interaction_residual_r2],
                    fmt=marker, color=color, ms=8, lw=2, capsize=0, mec="#fcfcfb", mew=1.5, label=label, zorder=3)
    ax.set_xticks(range(len(bins)), ["R < 25", "25-50", "50-75", "R >= 75"], color="#0b0b0b")
    ax.set_xlabel("Bliss expected inhibition of the dose row (training-pair monotherapy)", color="#52514e")
    ax.set_ylabel("interaction residual R² vs reference", color="#52514e")
    ax.set_title("ONEIL: pair-specific gain over the reference by expected effect (drug jackknife 95%)", color="#0b0b0b", fontsize=10)
    ax.grid(axis="y", color="#e4e3df", lw=.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor="#0b0b0b")
    fig.tight_layout()
    fig.savefig(output / "fig_regimes.png", dpi=180, facecolor=fig.get_facecolor())
    print(f"Saved {output / 'fig_regimes.png'}")


def analyze(output, suite=RUNS / "benchmark_v1", protocol_path="configs/neural_benchmark.json", dataset="data/processed/oneil_v1",
            declared="configs/beyond_bliss_regimes.json"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plan, protocol = read_json(declared), read_json(protocol_path)
    chosen = read_json(Path(suite) / "selection.json")["selected"]
    df, _, _ = load_truth(dataset)
    t = index(df)
    y = t["y"]
    reference = out_of_fold(model_paths("f2_residual", protocol["selection"]["seed"], chosen, suite), t, "reference_offset")
    region, level = regimes(df, reference)
    summary, tables, per_pair_all = {"declared": plan}, [], []
    for model in plan["models"]:
        predictions = [out_of_fold(model_paths(model, s, chosen, suite), t) for s in plan["seeds"]]
        for family, name, mask in regime_masks(y, region, level):
            tables.append({"model": model, "family": family, "regime": name, **regime_metrics(t, mask, y, predictions, reference)})
        reliability, per_pair, frame = pair_reliability(t, y, predictions, reference)
        summary[model] = {"pair_reliability": reliability}
        per_pair_all.append(per_pair.assign(model=model))
        drugs = drug_subsets(t, frame)
        drugs.assign(model=model).to_csv(output / f"drugs_{model.replace(':', '_')}.csv", index=False, float_format="%.6g")
        summary[model]["drugs_holm_significant"] = drugs[drugs.p_holm < .05].to_dict("records")
    table = add_holm(pd.DataFrame(tables), [{"model": m} for m in plan["models"]])
    rule = lambda r: (r.family != "all" and r.interaction_residual_p_holm < .05 and min(r.interaction_residual_r2_seeds) > 0
                      and r.interaction_residual_r2_jackknife[0] > 0)
    table["meaningful_improvement"] = [bool(rule(r)) if r.family != "all" else None for r in table.itertuples()]
    flat = flatten(table)
    flat.to_csv(output / "regimes.csv", index=False, float_format="%.6g")
    pd.concat(per_pair_all).to_csv(output / "pairs.csv", index=False, float_format="%.6g")
    summary["regimes"] = json.loads(table.to_json(orient="records"))
    summary = json.loads(json.dumps(summary, default=float).replace("NaN", "null"))
    write_json(output / "summary.json", summary)
    print(flat[["model", "family", "regime", "panel_rows", "interaction_residual_r2", "interaction_residual_r2_jackknife",
                "interaction_residual_p_holm", "contrast_r2_difference", "response_residual_r2", "meaningful_improvement"]].to_string())
    for model in plan["models"]:
        print(model, summary[model]["pair_reliability"], len(summary[model]["drugs_holm_significant"]), "drugs significant after Holm")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True)
    p.add_argument("--addendum", action="store_true", help="run the post hoc calibrated-Bliss and mono_only controls")
    p.add_argument("--figure", action="store_true", help="plot saved results")
    a = p.parse_args()
    if a.figure:
        figure(a.output)
    elif a.addendum:
        addendum(a.output)
        addendum_mono(a.output)
    else:
        analyze(a.output)


if __name__ == "__main__":
    main()
