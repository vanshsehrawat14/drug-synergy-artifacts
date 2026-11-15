"""Frozen input-ablation contrasts against the historical full-input F2 network (overnight 2026-11-07).

Each mask is compared with `neural_base_F2` (sprint_v1 f2_base, same folds and seeds) on identical rows,
using the frozen resolution projectors: response MSE, pair-cell contrast (P-A) and dose-shape contrast
(E-P-D) MSE, Bliss residual skill and pair-cell Bliss-excess recovery. Errors are averaged over seeds.
Whole-pair bootstrap, delete-one-drug jackknife with projector refits, five-fold direction, Holm across masks.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student
from threadpoolctl import threadpool_limits

from .io import read_json, write_json, digest
from .regimes import holm
from .resolution import means, projections, grouped_intervals
from .sprint_report import index, load_truth, out_of_fold, jackknife_interval

MASKS = ["no_drug_identity", "no_cell_identity", "no_dose"]
SEEDS = [2026, 2027, 2028]


def unit_group(df, t):
    ra = df.groupby("block_id", observed=True).dose_a.rank(method="dense").to_numpy(int) - 1
    rb = df.groupby("block_id", observed=True).dose_b.rank(method="dense").to_numpy(int) - 1
    keys = pd.MultiIndex.from_arrays([t["panel"], ra, rb, df.dose_a.to_numpy(), df.dose_b.to_numpy()])
    group, _ = pd.factorize(keys, sort=True)
    return group


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.resolution)
    meta = read_json(base / "inputs.json")
    frame = pd.read_parquet(base / "coordinates.parquet")
    with np.load(base / "cache.npz") as c:
        values, rowerr, rep = c["values"], c["row_pair_error"], c["repeat_error"]
    df, sha, _ = load_truth("data/processed/oneil_v1")
    if sha != meta["dataset_sha256"]:
        raise ValueError("Dataset differs from the frozen resolution cache")
    t = index(df)
    group = unit_group(df, t)
    n = np.bincount(group)
    if not np.allclose(means(group, t["y"], n), values[:, 0]):
        raise AssertionError("Unit mapping does not reproduce the cached truth")
    names, fams, seeds = list(meta["names"]), list(meta["families"]), list(meta["seeds"])
    seeds_run = list(a.seeds)
    if seeds_run == SEEDS:
        keep = [0] + [j for j, f in enumerate(fams) if f in ("bliss", "neural_base_F2", "neural_base_F0", "mono_only")]
    else:  # other seed sets: F2 control restricted to the same seeds; F0/mono controls have no matching seeds
        keep = [0] + [j for j, f in enumerate(fams) if f == "bliss" or (f == "neural_base_F2" and seeds[j] in seeds_run)]
    names, fams, seeds = [names[j] for j in keep], [fams[j] for j in keep], [seeds[j] for j in keep]
    values, rowerr, rep = values[:, keep], rowerr[:, keep], rep[:, keep]
    yg = values[:, 0]
    roots = [Path(r) for r in ([a.runs] if isinstance(a.runs, str) else a.runs)]
    validation = []
    extra_v, extra_r, extra_p = [], [], []
    ablations = {m: [r / m for r in roots] for m in MASKS}
    if a.pseudo:
        ablations = {"pseudo_f0_base": [Path("results/neural/runs/sprint_v1/runs/f0_base")]}
    cached_f2 = {seeds[j] for j, f in enumerate(fams) if f == "neural_base_F2"}
    if [s for s in seeds_run if s not in cached_f2]:  # matched F2 control trained for seeds absent from the cache
        ablations = {"neural_base_F2": [Path(a.f2_runs)], **ablations}
    for mask, dirs in ablations.items():
        for s in (seeds_run if mask != "neural_base_F2" else [s for s in seeds_run if s not in cached_f2]):
            d = next((x for x in dirs if (x / f"seed_{s}").exists()), dirs[0])
            paths = {k: d / f"seed_{s}" / f"fold_{k}" for k in range(5)}
            missing = [k for k, p in paths.items() if not (p / "predictions.parquet").exists()]
            if missing:
                validation.append(dict(mask=mask, seed=s, status="incomplete", missing_folds=str(missing)))
                continue
            p = out_of_fold(paths, t)  # raises unless IDs equal the frozen test fold order and all finite
            dup = sum(pd.read_parquet(paths[k] / "predictions.parquet", columns=["observation_id"]).observation_id.duplicated().sum() for k in range(5))
            cfg = [read_json(paths[k] / "config.json") for k in range(5)] if not a.pseudo else []
            validation.append(dict(mask=mask, seed=s, status="validated", rows=int(len(p)), duplicate_ids=int(dup),
                                   finite=bool(np.isfinite(p).all()),
                                   configs_seed_fold_ok=all(c["seed"] == s and c["test_fold"] == k for k, c in enumerate(cfg)),
                                   mask_flags=str({k: v for k, v in (cfg[0]["features"].items() if cfg else []) if k != "regime"})))
            pg = means(group, p, n)
            err = t["y"] - p
            names.append(f"{mask}:{s}"); fams.append(mask); seeds.append(s)
            extra_v.append(pg)
            extra_r.append(np.bincount(t["pair"], weights=err ** 2, minlength=len(t["pairs"])))
            extra_p.append(np.bincount(t["pair"], weights=(err - (yg - pg)[group]) ** 2, minlength=len(t["pairs"])))
    pd.DataFrame(validation).to_csv(out / "ablation_validation.csv", index=False)
    if extra_v:
        values = np.column_stack([values] + extra_v)
        rowerr = np.column_stack([rowerr] + extra_r)
        rep = np.column_stack([rep] + extra_p)
    fams = np.array(fams)
    z = values[:, :1] - values
    z[:, 0] = values[:, 0]
    w = frame.n.to_numpy(float)
    pair = frame.pair.to_numpy(int)
    panel = frame.panel.to_numpy(int)
    pn = np.bincount(panel, weights=w)
    pm = means(panel, z * w[:, None], pn)
    with threadpool_limits(limits=a.threads):
        ep, ew, rp, rw = projections(frame, z)
        leaves_p, leaves_w = [], []
        for d in meta["drugs"]:
            lp, lw, _, _ = projections(frame, z, d, False)
            leaves_p.append(lp); leaves_w.append(lw)
    leaves_p, leaves_w = np.array(leaves_p), np.array(leaves_w)
    pair_panel = np.column_stack([np.bincount(pair, weights=w * c ** 2) for c in rp.T])
    pair_within = np.column_stack([np.bincount(pair, weights=w * c ** 2) for c in rw.T])
    bl = np.flatnonzero(fams == "bliss")[0]
    # pair-cell Bliss excess: observed mean(y - r) and predicted mean(yhat - r) per panel
    excess_obs = pm[:, bl]  # z[:, bl] = y - r_bliss at unit level
    metrics = []
    for fam in dict.fromkeys(fams[1:]):
        ix = np.flatnonzero(fams == fam)
        pred_excess = [excess_obs - pm[:, j] for j in ix]  # (y - r) - (y - yhat) = yhat - r
        metrics.append(dict(model=fam, seeds=len(ix), row_mse=rowerr[:, ix].sum(0).mean() / w.sum(),
                            panel_mse_weighted=float((pm[:, ix] ** 2 * pn[:, None]).sum(0).mean() / pn.sum()),
                            panel_contrast_r2=1 - ep[ix].mean() / ep[0], dose_contrast_r2=1 - ew[ix].mean() / ew[0],
                            bliss_residual_skill=1 - rowerr[:, ix].sum(0).mean() / rowerr[:, bl].sum(),
                            panel_bliss_contrast_skill=1 - ep[ix].mean() / ep[bl],
                            excess_recovery_pearson=float(np.mean([np.corrcoef(excess_obs, q)[0, 1] for q in pred_excess])),
                            excess_recovery_pearson_min_seed=float(np.min([np.corrcoef(excess_obs, q)[0, 1] for q in pred_excess]))))
    pd.DataFrame(metrics).to_csv(out / "ablation_metrics.csv", index=False)
    comps = []
    controls = [m for m in dict.fromkeys(fams) if m in MASKS or m.startswith("pseudo")] + ["neural_base_F0", "mono_only"]
    pf = frame.groupby("pair", observed=True).fold.first().to_numpy()
    for model in controls:
        ia, ib = np.flatnonzero(fams == model), np.flatnonzero(fams == "neural_base_F2")
        if not len(ia):
            continue
        for level, errors, leave in [("response", rowerr, None), ("panel_contrast", pair_panel, leaves_p), ("dose_contrast", pair_within, leaves_w)]:
            r = grouped_intervals(errors[:, ia].mean(1), errors[:, ib].mean(1), meta["pair_drugs"])
            if leave is not None:
                jk = leave[:, ia].mean(1) / leave[:, ib].mean(1) - 1
                r["drug_low"], r["drug_high"] = map(float, jackknife_interval(r["change"], jk))
            else:
                pdg = np.asarray(meta["pair_drugs"])
                jk = np.array([errors[(pdg != d).all(1)][:, ia].sum(0).mean() / errors[(pdg != d).all(1)][:, ib].sum(0).mean() - 1 for d in meta["drugs"]])
            se = np.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum())
            r["p"] = float(2 * student.sf(abs(r["change"]) / se, len(jk) - 1)) if se else 1.0
            fd = [errors[pf == f][:, ia].sum(0).mean() / errors[pf == f][:, ib].sum(0).mean() - 1 for f in range(5)]
            comps.append(dict(model=model, control="neural_base_F2", level=level, **r, fold_min=min(fd), fold_max=max(fd),
                              folds_same_sign=bool(all(np.sign(x) == np.sign(r["change"]) for x in fd)),
                              fold_changes=" ".join(f"{x:.6f}" for x in fd)))
    comp = pd.DataFrame(comps)
    comp["p_holm_masks"] = np.nan
    for level in comp.level.unique():
        m = (comp.level == level) & comp.model.isin(MASKS)
        if m.any():
            comp.loc[m, "p_holm_masks"] = holm(comp.loc[m, "p"].to_numpy())
    comp["material_change"] = ((comp.change.abs() >= .05) & (np.sign(comp.pair_low) == np.sign(comp.pair_high)) &
                               (np.sign(comp.drug_low) == np.sign(comp.drug_high)) & (np.sign(comp.pair_low) == np.sign(comp.change)) &
                               comp.folds_same_sign & (comp.p_holm_masks.fillna(comp.p) < .05))
    # Phase 3.4 trigger (preregistered): classification decided by one fold, or a grouped bound within 0.5 pp of 5%
    near = lambda r: min(abs(abs(r.pair_low) - .05), abs(abs(r.pair_high) - .05), abs(abs(r.drug_low) - .05), abs(abs(r.drug_high) - .05)) < .005
    comp["confirmatory_seed_trigger"] = comp.apply(lambda r: r.model in MASKS and (near(r) or (sum(
        np.sign([r.fold_min, r.fold_max]) != np.sign(r.change)) == 1 and abs(r.change) >= .05)), axis=1)
    comp.to_csv(out / "ablation_comparisons.csv", index=False)
    write_json(out / "ablation_provenance.json", dict(runs=";".join(map(str, roots)), dataset_sha256=sha, resolution_cache=str(base),
               protocol_sha256=digest("configs/research_20261107.json"), names=names, pseudo=bool(a.pseudo),
               seeds=seeds_run, f2_runs=a.f2_runs))
    print(pd.DataFrame(metrics).to_string(index=False))
    print(comp[["model", "level", "change", "pair_low", "pair_high", "drug_low", "drug_high", "fold_min", "fold_max", "p", "material_change"]].to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", default=["results/research_20261107/runs"])
    p.add_argument("--output", required=True)
    p.add_argument("--pseudo", action="store_true", help="validation only: treat historical f0_base as an ablation")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--resolution", default="results/research_20261107/resolution")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="mask seeds; the F2 control uses the same seeds")
    p.add_argument("--f2-runs", help="f2_base runs (seed_*/fold_*) for seeds missing from the resolution cache")
    run(p.parse_args())
