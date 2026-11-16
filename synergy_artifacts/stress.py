"""High-inhibition stress tests declared in configs/stress_tests.json, on existing out-of-fold predictions.

python -m synergy_artifacts.stress --screen oneil --output results/stress_tests
python -m synergy_artifacts.stress --screen almanac --output results/stress_tests --batch almanac_unit_batch.csv

Target: interaction residual R2 vs Bliss at R >= 75 (and pooled) for selected F2 (ONEIL) and f2_combined
(ALMANAC). A control is survived only if its interval lies above 0 and every seed is positive.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json
from .regimes import calibrated_bliss, panel_of, regimes
from .neural_report import RUNS, index, jackknife_interval, load_truth, model_paths, out_of_fold
from .suite import run_dir

SEEDS = [2026, 2027, 2028]


def energies(frame, values, drugs, extra=None):
    """Residual energy after removing drug main effects (plus `extra` main effects) within fold and cell.

    Returns the full energy, the energy with each drug deleted, and the full-fit energy of each cell."""
    full, by_drug, by_cell = np.zeros(values.shape[1]), np.zeros((len(drugs), values.shape[1])), {}
    for (_, cell), g in frame.groupby(["fold", "cell_line"], sort=True):
        rows, a, b = g.index.to_numpy(), g.drug_a.to_numpy(), g.drug_b.to_numpy()
        levels = g[extra].to_numpy() if extra else None
        for j, drop in enumerate([None] + list(drugs)):
            keep = np.ones(len(g), bool) if drop is None else (a != drop) & (b != drop)
            if not keep.any() or (drop is not None and keep.all()):
                if drop is not None and keep.all():
                    by_drug[j - 1] += e_all
                continue
            names = sorted(set(a[keep]) | set(b[keep]))
            lookup = {x: i for i, x in enumerate(names)}
            columns = [np.zeros((keep.sum(), len(names)))]
            ix = np.arange(keep.sum())
            np.add.at(columns[0], (ix, [lookup[x] for x in a[keep]]), 1)
            np.add.at(columns[0], (ix, [lookup[x] for x in b[keep]]), 1)
            if extra:
                codes, inverse = np.unique(levels[keep], return_inverse=True)
                columns.append(np.eye(len(codes))[inverse][:, 1:])
            h = np.hstack(columns)
            z = values[rows[keep]]
            e = ((z - h @ np.linalg.lstsq(h, z, rcond=None)[0]) ** 2).sum(0)
            if drop is None:
                e_all = e
                full += e
                by_cell[cell] = by_cell.get(cell, 0) + e
            else:
                by_drug[j - 1] += e
    return full, by_drug, by_cell


def gain(t, mask, y, predictions, references, extra=None, frame_extra=None):
    """Interaction residual R2 of the predictions against the references (one shared reference, or one per seed)."""
    k = len(predictions)
    frame, means = panel_of(t, mask, [y - p for p in predictions] + [y - r for r in references])
    if frame_extra is not None:
        frame = frame.assign(batch=frame_extra(frame))
    full, by_drug, by_cell = energies(frame, means, t["drugs"], "batch" if extra else None)
    ratio = lambda e: 1 - e[:k].mean() / e[k:].mean()
    seeds = [1 - full[s] / full[k + s if len(references) == k else k] for s in range(k)]
    drug_reps = np.array([ratio(e) for e in by_drug if e[k:].mean() > 0])
    cells = sorted(by_cell)
    cell_reps = np.array([ratio(full - by_cell[c]) for c in cells]) if len(cells) > 2 else np.array([])
    value = ratio(full)
    influence = np.abs(drug_reps - value)
    out = {"panel_rows": len(frame), "interaction_residual_r2": float(value), "seeds": [float(s) for s in seeds],
           "drug_jackknife": [float(x) for x in jackknife_interval(value, drug_reps)],
           "max_drug_influence": float(influence.max()),
           "max_drug_influence_drug": t["drugs"][int(np.argmax([abs(ratio(e) - value) if e[k:].mean() > 0 else -1 for e in by_drug]))]}
    if len(cell_reps):
        out["cell_jackknife"] = [float(x) for x in jackknife_interval(value, cell_reps)]
    return out


def survives(row, interval="drug_jackknife"):
    return bool(row[interval][0] > 0 and min(row["seeds"]) > 0)


def calibrated_by_cell(df, y, reference):
    """Isotonic Bliss refitted per cell line on the three training folds of each test fold."""
    out = np.full(len(y), np.nan)
    cells = df.cell_line.astype(str).to_numpy()
    for c in np.unique(cells):
        m = cells == c
        out[m] = calibrated_bliss(df[m].reset_index(drop=True), y[m], reference[m])
    return out


def analog_similarity(t, observations):
    """Per pair: highest monotherapy-profile correlation between one drug and the partner of a training-fold pair sharing the other drug."""
    mono = observations[observations.is_monotherapy & observations.valid_target]
    treated = np.where(mono.dose_a > 0, mono.drug_a.astype(str), mono.drug_b.astype(str))
    profile = (pd.DataFrame({"drug": treated, "cell": mono.cell_line.astype(str).to_numpy(), "y": mono.inhibition.to_numpy(float)})
               .groupby(["drug", "cell"]).y.mean().unstack())
    corr = profile.T.corr(min_periods=5).reindex(index=t["drugs"], columns=t["drugs"])
    pairs = pd.DataFrame(t["pair_drugs"], columns=["a", "b"]).assign(fold=t["pair_fold"])
    sim = np.full(len(pairs), -1.)
    for k in range(5):
        train = pairs[~pairs.fold.isin([k, (k + 1) % 5])]
        partners = {}
        for a, b in zip(train.a, train.b):
            partners.setdefault(a, set()).add(b)
            partners.setdefault(b, set()).add(a)
        for i in np.flatnonzero(pairs.fold.to_numpy() == k):
            a, b = pairs.a[i], pairs.b[i]
            best = [corr.loc[b, list(partners.get(a, set()) - {b})].max() if partners.get(a) else np.nan,
                    corr.loc[a, list(partners.get(b, set()) - {a})].max() if partners.get(b) else np.nan]
            sim[i] = np.nanmax(best) if np.isfinite(best).any() else -1.
    return sim


def load(screen):
    if screen == "oneil":
        suite, protocol = RUNS / "benchmark_v1", read_json("configs/neural_benchmark.json")
        chosen = read_json(suite / "selection.json")["selected"]
        df, _, _ = load_truth("data/processed/oneil_v1")
        t = index(df)
        reference = out_of_fold(model_paths("f2_residual", protocol["selection"]["seed"], chosen, suite), t, "reference_offset")
        f2 = [out_of_fold(model_paths("selected:F2", s, chosen, suite), t) for s in SEEDS]
        mono = [out_of_fold(model_paths("mono_only", s, chosen, suite), t) for s in SEEDS]
        observations = pd.read_parquet("data/processed/oneil_v1/observations.parquet")
    else:
        from .runner import load_experiment
        dataset = read_json("configs/almanac_validation.json")["shared_overrides"]["dataset"]
        df, _, _ = load_experiment({"dataset": dataset, "table": "observations", "target": "inhibition", "test_fold": 0, "model": {"kind": "dose"}})
        t = index(df)
        reference = out_of_fold({k: RUNS / "almanac_reference_bliss" / f"fold_{k}" for k in range(5)}, t)
        f2 = [out_of_fold({k: run_dir(RUNS / "almanac_v1", "f2_combined", s, k) for k in range(5)}, t) for s in SEEDS]
        mono_root = RUNS / "almanac_mono_only"
        mono = ([out_of_fold({k: run_dir(mono_root, "mono_only", s, k) for k in range(5)}, t) for s in SEEDS]
                if all((run_dir(mono_root, "mono_only", s, k) / "evaluation").exists() for s in SEEDS for k in range(5)) else None)
        observations = pd.read_parquet(Path(dataset) / "observations.parquet")
    return df, t, reference, f2, mono, observations


def run(screen, output, batch=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df, t, reference, f2, mono, observations = load(screen)
    y = t["y"]
    region, level = regimes(df, reference)
    high, everything = level == "R>=75", np.ones(len(y), bool)
    rows = []

    def add(control, variant, regime, mask, references=None, interval="drug_jackknife", **kw):
        r = gain(t, mask, y, f2, references or [reference], **kw)
        rows.append({"screen": screen, "control": control, "variant": variant, "regime": regime, **r,
                     "interval": interval, "survives": survives(r, interval) if regime == "R>=75" else None})
        print(rows[-1]["control"], variant, regime, r["panel_rows"], round(r["interaction_residual_r2"], 4), r[interval], r["seeds"], flush=True)

    for regime, mask in [("R>=75", high), ("all", everything)]:
        add("none", "as reported", regime, mask)
        add("correlated pairs", "delete-one-cell jackknife", regime, mask, interval="cell_jackknife")
    for name in ["low", "mixed", "high"]:
        for lv in ["R<25", "25-50", "50-75", "R>=75"]:
            m = (region == name) & (level == lv)
            if m.any() and len(np.unique(t["panel"][m])) >= 200:
                add("dose regime", f"{name} doses", lv, m)
    cell_cal = calibrated_by_cell(df, y, reference)
    for regime, mask in [("R>=75", high), ("all", everything)]:
        add("calibration", "isotonic Bliss per cell line", regime, mask, references=[cell_cal])
        if mono is not None:
            add("calibration", "mono_only network as reference (seed by seed)", regime, mask, references=mono)
    sim = analog_similarity(t, observations)
    cut = np.median(sim)
    distant = sim[t["pair"]] < cut
    for regime, mask in [("R>=75", high), ("all", everything)]:
        add("correlated pairs", f"pairs below median analog similarity ({cut:.3f})", regime, mask & distant)
    if batch is not None:
        units = pd.read_csv(batch).astype({"pair_id": str, "cell": str})
        for column, label in [("TESTDATE", "NCI test date"), ("PLATE", "NCI plate")]:
            lookup = dict(zip(zip(units.pair_id, units.cell), units.SCREENER + ":" + units[column].astype(str)))
            def levels(frame, lookup=lookup):
                return [lookup.get((p, c), "unmatched") for p, c in zip(frame.pair_id, frame.cell_line)]
            for regime, mask in [("R>=75", high), ("all", everything)]:
                add("batch", f"{label} main effects removed", regime, mask, extra=True, frame_extra=levels)
    table = pd.DataFrame(rows)
    for c in ["seeds", "drug_jackknife", "cell_jackknife"]:
        table[c] = table[c].map(lambda v: ";".join(f"{x:.4f}" for x in v) if isinstance(v, list) else "")
    table.to_csv(output / f"{screen}.csv", index=False, float_format="%.6g", lineterminator="\n")
    pd.DataFrame({"pair_id": t["pairs"], "analog_similarity": sim}).to_csv(output / f"{screen}_analog_similarity.csv", index=False, float_format="%.6g", lineterminator="\n")
    return table


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--screen", choices=["oneil", "almanac"], required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch", help="ALMANAC pair-cell -> NCI SCREENER/TESTDATE/PLATE map")
    a = p.parse_args()
    run(a.screen, a.output, a.batch)


if __name__ == "__main__":
    main()
