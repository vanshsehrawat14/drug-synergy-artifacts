"""Exploratory prediction-resolution analysis of frozen out-of-fold predictions.

python -m synergy_artifacts.resolution --output results/interaction_analysis/resolution
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.stats import t as student
from threadpoolctl import threadpool_limits

from .io import read_json, write_json, digest
from .regimes import holm
from .neural_report import RUNS, index, load_truth, model_paths, out_of_fold, team_predictions, jackknife_interval


def means(codes, x, n=None):
    n = np.bincount(codes) if n is None else n
    if x.ndim == 1:
        return np.bincount(codes, weights=x, minlength=len(n)) / n
    return np.column_stack([means(codes, c, n) for c in x.T])


def residualize(h, values, weights):
    """Weighted orthogonal residual. QR/SVD tolerates the endpoint incidence null space."""
    sw = np.sqrt(weights)
    beta = lstsq(h * sw[:, None], values * sw[:, None], cond=1e-11, lapack_driver="gelsy")[0]
    return values - h @ beta


def grouped_intervals(numerator, denominator, pair_drugs, draws=2000, seed=20261107):
    """Relative change from pair-level sufficient statistics; paired sampling throughout."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    value = numerator.sum() / denominator.sum() - 1
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(numerator), size=(draws, len(numerator)))
    boot = numerator[ix].sum(1) / denominator[ix].sum(1) - 1
    drugs = sorted(set(np.asarray(pair_drugs).ravel()))
    pdg = np.asarray(pair_drugs)
    leave = np.array([numerator[(pdg != d).all(1)].sum() / denominator[(pdg != d).all(1)].sum() - 1 for d in drugs])
    low, high = jackknife_interval(value, leave)
    return dict(change=float(value), pair_low=float(np.quantile(boot, .025)), pair_high=float(np.quantile(boot, .975)),
                drug_low=float(low), drug_high=float(high))


def prepare(output):
    df, sha, _ = load_truth("data/processed/oneil_v1")
    t = index(df)
    ra = df.groupby("block_id", observed=True).dose_a.rank(method="dense").to_numpy(int) - 1
    rb = df.groupby("block_id", observed=True).dose_b.rank(method="dense").to_numpy(int) - 1
    if ra.min() != 0 or rb.min() != 0 or ra.max() != 3 or rb.max() != 3:
        raise ValueError("Expected complete 4 by 4 grids")
    keys = pd.MultiIndex.from_arrays([t["panel"], ra, rb, df.dose_a.to_numpy(), df.dose_b.to_numpy()])
    group, unique = pd.factorize(keys, sort=True)
    n = np.bincount(group)
    if (n == 0).any():
        raise ValueError("Missing panel dose-grid location")
    for dose in [df.dose_a.to_numpy(), df.dose_b.to_numpy()]:
        avg = means(group, dose, n)
        if np.max(np.abs(dose - avg[group])) > 1e-10:
            raise ValueError("Dose ranks do not identify identical concentrations within panel")
    protocol = read_json("configs/neural_benchmark.json")
    suite = RUNS / "benchmark_v1"
    chosen = read_json(suite / "selection.json")["selected"]
    names, families, seeds, values, row_pair_error, repeat_error = [], [], [], [], [], []
    y = t["y"]
    yg = means(group, y, n)
    values.append(yg)
    names.append("truth"); families.append("truth"); seeds.append(-1)
    repeat_truth = np.bincount(t["pair"], weights=(y - yg[group]) ** 2)
    repeat_error.append(repeat_truth)
    row_pair_error.append(np.zeros(len(t["pairs"])))

    def add(name, family, seed, p):
        names.append(name); families.append(family); seeds.append(seed)
        pg = means(group, p, n)
        values.append(pg)
        err = y - p
        row_pair_error.append(np.bincount(t["pair"], weights=err ** 2))
        repeat_error.append(np.bincount(t["pair"], weights=(err - (yg - pg)[group]) ** 2))
        print("loaded", name, flush=True)

    for label, model in [("neural_F2", "selected:F2"), ("neural_F0", "selected:F0"),
                         ("neural_base_F0", "f0_base"), ("neural_base_F2", "f2_base"),
                         ("mono_only", "mono_only"), ("mono_dose", "mono_dose")]:
        ss = protocol["benchmark"]["seeds"] if label != "mono_dose" else [2026]
        for seed in ss:
            add(f"{label}:{seed}", label, seed, out_of_fold(model_paths(model, seed, chosen, suite), t))
    refpaths = model_paths("f2_residual", 2026, chosen, suite)
    add("bliss", "bliss", -1, out_of_fold(refpaths, t, "reference_offset"))
    add("additive", "additive", -1, out_of_fold(model_paths("additive", None, chosen, suite), t))
    for name, directory in [("gp_F2", "gp_gpu_f2_residual_mono"), ("gp_F0", "gp_gpu_f0_drug_dose"),
                            ("gp_full", "gp_gpu_full"), ("gp_w0", "gp_gpu_full_w0"),
                            ("cf_selected", "cf_selected"), ("cf_original", "cf_shar"), ("cf_original_dose", "cf_shar_dose")]:
        add(name, name, -1, team_predictions(RUNS / directory, t))
    panel_ids = unique.get_level_values(0).to_numpy(int)
    frame = t["panel_frame"].iloc[panel_ids].reset_index(drop=True)
    frame["ra"], frame["rb"] = unique.get_level_values(1), unique.get_level_values(2)
    frame["dose_a"], frame["dose_b"] = unique.get_level_values(3), unique.get_level_values(4)
    frame["panel"] = panel_ids
    frame["pair"] = pd.Categorical(frame.pair_id, categories=t["pairs"]).codes
    frame["n"] = n
    frame.to_parquet(output / "coordinates.parquet", index=False)
    np.savez_compressed(output / "cache.npz", values=np.column_stack(values), row_pair_error=np.array(row_pair_error).T,
                        repeat_error=np.array(repeat_error).T)
    write_json(output / "inputs.json", {"dataset_sha256": sha, "rows": len(y), "panels": len(t["panel_frame"]),
               "pairs": len(t["pairs"]), "drugs": t["drugs"], "pair_drugs": t["pair_drugs"], "names": names,
               "families": families, "seeds": seeds, "protocol_sha256": digest("configs/interaction_analysis.json"),
               "grouping_uses_rank_and_actual_concentrations": True, "raw_y_mean": float(y.mean()), "raw_y_variance": float(y.var()),
               "note": "Historical networks are averaged over seed-specific errors, never predictions. Contrast projectors use test covariates and define evaluation estimands only."})


def projections(frame, z, drop=None, return_residual=True):
    """Two orthogonal contrasts: panel incidence and centered endpoint drug-dose rank.

    Within each cell/fold, centered endpoint indicators remove dose main effects.
    Panel centering makes this space orthogonal to *every* panel-constant vector.
    Weighting by counts preserves the dose-row estimand after replicate collapse.
    """
    keep = np.ones(len(frame), bool) if drop is None else ((frame.drug_a != drop) & (frame.drug_b != drop)).to_numpy()
    rpanel, rwithin = np.zeros_like(z), np.zeros_like(z)
    for _, g in frame[keep].groupby(["fold", "cell_line"], observed=True):
        ii = g.index.to_numpy()
        a, b = g.drug_a.to_numpy(), g.drug_b.to_numpy()
        drugs = sorted(set(a) | set(b)); lookup = {d: j for j, d in enumerate(drugs)}
        da, db = np.array([lookup[d] for d in a]), np.array([lookup[d] for d in b])
        panel, inv = np.unique(g.panel.to_numpy(), return_inverse=True)
        w = g.n.to_numpy(float); pn = np.bincount(inv, weights=w)
        pm = means(inv, z[ii] * w[:, None], pn)
        first = np.unique(inv, return_index=True)[1]
        hp = np.zeros((len(panel), len(drugs)))
        np.add.at(hp, (np.arange(len(panel)), da[first]), 1)
        np.add.at(hp, (np.arange(len(panel)), db[first]), 1)
        rp = residualize(hp, pm, pn)
        rpanel[ii] = rp[inv]
        h = np.zeros((len(g), len(drugs) * 4))
        rows = np.arange(len(g))
        np.add.at(h, (rows, da * 4 + g.ra.to_numpy(int)), 1)
        np.add.at(h, (rows, db * 4 + g.rb.to_numpy(int)), 1)
        h -= means(inv, h * w[:, None], pn)[inv]
        rwithin[ii] = residualize(h, z[ii] - pm[inv], w)
    w = frame.n.to_numpy(float) * keep
    ep = (rpanel ** 2 * w[:, None]).sum(0)
    ew = (rwithin ** 2 * w[:, None]).sum(0)
    return ep, ew, rpanel if return_residual else None, rwithin if return_residual else None


def analyze(output):
    meta = read_json(output / "inputs.json")
    frame = pd.read_parquet(output / "coordinates.parquet")
    with np.load(output / "cache.npz") as cache:
        v, rowerr, rep = cache["values"], cache["row_pair_error"], cache["repeat_error"]
    # Truth and all model errors: the same linear projection acts on every column.
    z = v[:, :1] - v; z[:, 0] = v[:, 0]
    n, pair, panel = frame.n.to_numpy(float), frame.pair.to_numpy(int), frame.panel.to_numpy(int)
    pn = np.bincount(panel, weights=n)
    pm = means(panel, z * n[:, None], pn)
    within = z - pm[panel]
    panel_sse = (pm ** 2 * pn[:, None]).sum(0)
    within_sse = (within ** 2 * n[:, None]).sum(0)
    ep, ew, rp, rw = projections(frame, z)
    np.savez_compressed(output / "contrasts.npz", panel=rp, within=rw)
    pair_panel = np.column_stack([np.bincount(pair, weights=n * c ** 2) for c in rp.T])
    pair_within = np.column_stack([np.bincount(pair, weights=n * c ** 2) for c in rw.T])
    primary = [("gp_F2", "neural_F2"), ("cf_selected", "neural_F0"), ("gp_F0", "neural_F0")]
    families = np.array(meta["families"])
    # Delete a drug and refit both projections; deleting residuals without refitting is insufficient.
    leaves_p, leaves_w = [], []
    for d in meta["drugs"]:
        a, b, _, _ = projections(frame, z, d, False)
        leaves_p.append(a); leaves_w.append(b)
        print("projected deletion", d, flush=True)
    leaves_p, leaves_w = np.array(leaves_p), np.array(leaves_w)
    np.savez_compressed(output / "jackknife.npz", panel=leaves_p, within=leaves_w)
    ref = np.flatnonzero(families == "bliss")[0]
    denominator_row = meta["raw_y_variance"] * n.sum()
    denominator_panel = ((pm[:, 0] - meta["raw_y_mean"]) ** 2 * pn).sum()
    rows = []
    for family in dict.fromkeys(meta["families"]):
        if family == "truth": continue
        ix = np.flatnonzero(families == family)
        e = rowerr[:, ix].sum(0).mean()
        panel_e = panel_sse[ix].mean()
        rows.append(dict(model=family, seeds=len(ix), row_mse=e / n.sum(), row_r2=1-e/denominator_row,
            panel_mse_weighted=panel_e / pn.sum(), panel_mse_equal=float((pm[:, ix] ** 2).mean()),
            panel_r2=1-panel_e/denominator_panel, within_mse=within_sse[ix].mean()/n.sum(),
            bliss_residual_mse_reduction=1-e/rowerr[:, ref].sum(),
            panel_contrast_r2=1-ep[ix].mean()/ep[0],
            panel_bliss_contrast_mse_reduction=1-ep[ix].mean()/ep[ref],
            dose_contrast_r2=1-ew[ix].mean()/ew[0],
            dose_bliss_contrast_mse_reduction=1-ew[ix].mean()/ew[ref],
            replicate_mse=rep[:, ix].sum(0).mean()/n.sum()))
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    comparisons=[]
    for a,b in primary + [("neural_base_F2","neural_base_F0"),("neural_F2","mono_only"),("mono_dose","mono_only"),
                          ("cf_original_dose","cf_original"),("gp_full","gp_w0")]:
        ia,ib=np.flatnonzero(families==a),np.flatnonzero(families==b)
        for level, errors, leave in [("response",rowerr,None),("panel_contrast",pair_panel,leaves_p),("dose_contrast",pair_within,leaves_w)]:
            r=grouped_intervals(errors[:,ia].mean(1),errors[:,ib].mean(1),meta["pair_drugs"])
            if leave is not None:
                jk=leave[:,ia].mean(1)/leave[:,ib].mean(1)-1
                r["drug_low"],r["drug_high"]=map(float,jackknife_interval(r["change"],jk))
                se=np.sqrt((len(jk)-1)/len(jk)*((jk-jk.mean())**2).sum())
                r["p"]=float(2*student.sf(abs(r["change"])/se,len(jk)-1)) if se else 1.
            pf=frame.groupby("pair",observed=True).fold.first().to_numpy()
            fd=[errors[pf==f][:,ia].sum(0).mean()/errors[pf==f][:,ib].sum(0).mean()-1 for f in range(5)]
            comparisons.append(dict(model=a,control=b,level=level,**r,fold_min=min(fd),fold_max=max(fd),
                                    bootstrap="whole pair; conditional on fitted contrast" if leave is not None else "whole pair"))
    comp=pd.DataFrame(comparisons)
    mask=(comp.level=="panel_contrast") & comp.apply(lambda x:(x.model,x.control) in primary,axis=1)
    comp.loc[mask,"p_holm_primary"]=holm(comp.loc[mask,"p"].to_numpy())
    comp["material_superiority"]=((comp.change<=-.05)&(comp.pair_high<0)&(comp.drug_high<0)&(comp.fold_max<0)&(comp.p_holm_primary<.05))
    comp.to_csv(output / "comparisons.csv",index=False)
    variance={"between_panel_additive":(denominator_panel-ep[0])/n.sum(),"between_panel_contrast":ep[0]/n.sum(),
              "within_panel_additive_dose":(within_sse[0]-ew[0])/n.sum(),"within_panel_dose_contrast":ew[0]/n.sum(),
              "replicate_deviations":rep[:,0].sum()/n.sum()}
    decomposition=[]
    for j,name in enumerate(meta["names"]):
        if j==0: continue
        components=[panel_sse[j]-ep[j],ep[j],within_sse[j]-ew[j],ew[j],rep[:,j].sum()]
        total=rowerr[:,j].sum()
        if not np.isclose(sum(components),total,rtol=1e-9): raise AssertionError("Error decomposition failed")
        decomposition.append(dict(model=name,total_mse=total/n.sum(),**dict(zip(variance,np.array(components)/n.sum()))))
    pd.DataFrame(decomposition).to_csv(output / "error_decomposition.csv",index=False)
    if not np.isclose(sum(variance.values()),meta["raw_y_variance"],rtol=1e-9): raise AssertionError("Variance decomposition failed")
    write_json(output / "variance.json",{"population_variance":meta["raw_y_variance"],"components":variance,
       "shares":{k:x/meta["raw_y_variance"] for k,x in variance.items()},"rows":int(n.sum()),"panels":len(pn),
       "unique_dose_units":len(frame),"replicate_counts":pd.Series(n).value_counts().sort_index().to_dict(),
       "identity_error_max":float(abs(sum(variance.values())-meta["raw_y_variance"]))})
    pd.DataFrame(rows).set_index("model").to_string(output / "metrics.txt")
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True);p.add_argument("--reuse",action="store_true")
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if not a.reuse: prepare(out)
        analyze(out)


if __name__=="__main__": main()
