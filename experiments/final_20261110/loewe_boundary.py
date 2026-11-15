"""Numerical checks of the Loewe existence criterion, boundary-layer law and summed-dose jump (MATHEMATICAL_FINDINGS).

python experiments/final_20261110/loewe_boundary.py --cases CASES --regimes REGIMES --output OUT

Co-increasing LL.4/LL.4 cases only. Arm i has the smaller upper plateau h_i < h_j.
Theorem: a root exists iff f_j(x_j) < h_i. Law: h_i - y* = |d_i - c_i| q/(1+q), q = (delta e_i/x_i)^|b_i|,
delta = 1 - x_j/D_j(h_i) evaluated at the plateau (exact as delta -> 0).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def curve(x, b, c, d, e):
    return c + (d - c) / (1 + (x / e) ** b)


def inverse(y, b, c, d, e):
    return e * ((d - y) / (y - c)) ** (1 / b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", required=True); p.add_argument("--regimes", required=True); p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    c = pd.read_parquet(a.cases)
    r = pd.read_csv(a.regimes, usecols=["case_id", "cert_status", "regime", "in_C1", "certified_root", "root_distance_to_asymptote",
                                        "native_reference", "hsa_fallback_reference"])
    # Continuity consequence (Proposition 2): error of each fallback on the 14,197 rejected admissible roots.
    rejected = r[r.in_C1]
    fallback_error = {}
    for label, g in [("all", rejected), ("within_0_01_of_asymptote", rejected[rejected.root_distance_to_asymptote < 0.01]),
                     ("interior", rejected[rejected.root_distance_to_asymptote >= 0.01])]:
        sd, hs = (g.native_reference - g.certified_root).abs(), (g.hsa_fallback_reference - g.certified_root).abs()
        fallback_error[label] = dict(cases=int(len(g)), summed_dose_median=float(sd.median()), summed_dose_mean=float(sd.mean()),
                                     hsa_median=float(hs.median()), hsa_mean=float(hs.mean()), share_hsa_closer=float((hs < sd).mean()))
    c = c.merge(r, on="case_id", how="inner", validate="one_to_one")
    ll = (c.model_a == "LL.4") & (c.model_b == "LL.4")
    arm = {}
    for s in "ab":
        b, cc, d = c[f"{s}_b"].to_numpy(), c[f"{s}_c"].to_numpy(), c[f"{s}_d"].to_numpy()
        lo, hi = np.where(b < 0, cc, d), np.where(b < 0, d, cc)
        arm[s] = dict(b=b, c=cc, d=d, e=c[f"{s}_e"].to_numpy(), lo=lo, hi=hi, x=c[f"dose_{s}"].to_numpy(), inc=hi > lo)
    use = (ll & arm["a"]["inc"] & arm["b"]["inc"] & c.cert_status.isin(["certified_root", "certified_no_root"])).to_numpy()
    i_is_a = arm["a"]["hi"] < arm["b"]["hi"]
    pick = lambda key, first: np.where(first, arm["a"][key], arm["b"][key])
    I = {k: pick(k, i_is_a) for k in ["b", "c", "d", "e", "lo", "hi", "x"]}
    J = {k: pick(k, ~i_is_a) for k in ["b", "c", "d", "e", "lo", "hi", "x"]}
    equal = arm["a"]["hi"] == arm["b"]["hi"]
    with np.errstate(all="ignore"):
        fj = curve(J["x"], J["b"], J["c"], J["d"], J["e"])
        predicted_root = equal | (fj < I["hi"])
    certified_root = (c.cert_status == "certified_root").to_numpy()
    agree = predicted_root == certified_root
    margin = np.abs(fj - I["hi"])
    existence = dict(cases=int(use.sum()), agreement=int((agree & use).sum()), disagreement=int((~agree & use).sum()),
                     disagreement_max_margin=float(margin[~agree & use].max()) if (~agree & use).any() else None,
                     equal_plateau_cases=int((equal & use).sum()))
    # Boundary layer: certified roots near the smaller plateau.
    with np.errstate(all="ignore"):
        delta = 1 - J["x"] / inverse(I["hi"], J["b"], J["c"], J["d"], J["e"])
        q = (delta * I["e"] / I["x"]) ** np.abs(I["b"])
        layer = np.abs(I["d"] - I["c"]) * q / (1 + q)
        actual = I["hi"] - c.certified_root.to_numpy()
    roots = use & certified_root & ~equal & (delta > 0) & np.isfinite(layer) & (actual > 0)
    f = pd.DataFrame({"delta": delta[roots], "slope": np.abs(I["b"][roots]), "predicted": layer[roots], "actual": actual[roots],
                      "c1": c.in_C1.to_numpy()[roots], "regime": c.regime.to_numpy()[roots]})
    f["ratio"] = f.actual / f.predicted
    f["delta_bin"] = pd.cut(f.delta, [0, 1e-3, 1e-2, 0.05, 0.2, 1], include_lowest=True)
    bins = f.groupby("delta_bin", observed=True).agg(cases=("ratio", "size"), median_ratio=("ratio", "median"),
                                                     p05=("ratio", lambda x: x.quantile(.05)), p95=("ratio", lambda x: x.quantile(.95)),
                                                     median_actual=("actual", "median"), c1_cases=("c1", "sum")).reset_index()
    bins["delta_bin"] = bins.delta_bin.astype(str)
    bins.to_csv(out / "boundary_layer.csv", index=False)
    near = f.actual < 0.01
    f[near][["delta", "slope", "predicted", "actual", "c1", "regime"]].to_csv(out / "near_boundary_roots.csv", index=False)
    corr = float(np.corrcoef(np.log10(f.actual[near]), np.log10(f.predicted[near]))[0, 1])
    # Summed-dose jump on recovered (C1) cases: native fallback minus certified root.
    c1 = (use & c.in_C1.to_numpy()).astype(bool)
    with np.errstate(all="ignore"):
        summed = np.minimum(np.maximum(curve(I["x"] + J["x"], I["b"], I["c"], I["d"], I["e"]),
                                       curve(I["x"] + J["x"], J["b"], J["c"], J["d"], J["e"])), 100)
    jump_obs = c.native_reference.to_numpy() - c.certified_root.to_numpy()
    jump_theory = summed - I["hi"]
    root = c.certified_root.to_numpy()
    upper = c1 & (I["hi"] - root < 0.01)
    lower = c1 & ~upper & (root - np.maximum(I["lo"], J["lo"]) < 0.01)
    interior = c1 & ~upper & ~lower
    jumps = dict(c1_cases=int(c1.sum()), upper_plateau=int(upper.sum()), lower_asymptote=int(lower.sum()), interior=int(interior.sum()),
                 median_jump=float(np.median(jump_obs[c1])), mean_jump=float(jump_obs[c1].mean()),
                 share_jump_above_1=float((jump_obs[c1] > 1).mean()),
                 upper_median_jump=float(np.median(jump_obs[upper])), upper_mean_jump=float(jump_obs[upper].mean()),
                 upper_max_abs_theory_minus_observed=float(np.abs(jump_theory[upper] - jump_obs[upper]).max()),
                 upper_median_abs_theory_minus_observed=float(np.median(np.abs(jump_theory[upper] - jump_obs[upper]))),
                 share_of_c1_absolute_shift_from_upper=float(np.abs(jump_obs[upper]).sum() / np.abs(jump_obs[c1]).sum()))
    summary = dict(existence=existence, boundary_layer=dict(cases=int(len(f)), near_cases=int(near.sum()),
                   log_log_correlation_near=corr, median_ratio_delta_below_0_01=float(f.ratio[f.delta < 1e-2].median()),
                   c1_share_with_slope_above_2=float((f.slope[f.c1] > 2).mean()) if f.c1.any() else None,
                   all_root_share_with_slope_above_2=float((f.slope > 2).mean())), summed_dose_jump=jumps,
                   fallback_error_on_rejected_roots=fallback_error)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)); print(bins.to_string(index=False))


if __name__ == "__main__":
    main()
