"""Independent numerical audit of the Loewe acceptance artifact.

From the cached single-agent coefficients, the native case table and the 100-digit certificates, this
script recomputes: the case populations behind each reported count; the existence criterion
(Proposition 1) against certificate status; an independent high-precision bracket for every rejected
admissible root and for a control sample of accepted roots; the native summed-dose and HSA fallback
values and their errors relative to the certified root; root location; the summed-dose jump; and the
boundary-layer law. No project module is imported.

python manuscript/audit/loewe_audit.py --cases surface_cases.parquet --certificates certificates_replayable.parquet \
    --verified cases_verified.parquet --regimes loewe_regimes.csv --output manuscript/audit/outputs/loewe_audit.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curve(x, b, c, d, e, loglogistic):
    """drc LL.4 (e > 0) or SynergyFinder L.4 (e on the log-dose scale), at raw dose x > 0."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = np.where(loglogistic, b * (np.log(x) - np.log(np.where(loglogistic, e, 1.0))), b * (np.log(x) - e))
        return c + (d - c) / (1 + np.exp(z))


def mp_inverse(y, b, c, d, e):
    """LL.4 inverse D(y) = e ((d - y)/(y - c))^(1/b) in mpmath."""
    return e * ((d - y) / (y - c)) ** (1 / b)


def mp_g(y, arms, doses):
    return sum(x / mp_inverse(y, *p) for p, x in zip(arms, doses)) - 1


def quantiles(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(np.mean(x)),
            "p10": float(np.quantile(x, .1)), "p90": float(np.quantile(x, .9)), "max": float(np.max(x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--certificates", required=True)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--regimes", required=True)
    ap.add_argument("--controls", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261110)
    ap.add_argument("--output", required=True)
    ap.add_argument("--derived", required=True, help="directory for case-level tables (kept out of version control)")
    a = ap.parse_args()
    Path(a.derived).mkdir(parents=True, exist_ok=True)
    out = {"inputs": {k: sha256(getattr(a, k)) for k in ["cases", "certificates", "verified", "regimes"]}}

    cases = pd.read_parquet(a.cases)
    cert = pd.read_parquet(a.certificates)
    ver = pd.read_parquet(a.verified)
    reg = pd.read_csv(a.regimes, usecols=["case_id", "in_C1", "in_C2_recovered", "native_reference", "hsa_fallback_reference",
                                          "certified_root", "regime"])
    df = cases.merge(cert, on=["case_id", "surface_id"], how="inner", suffixes=("", "_cert"))
    df = df.merge(ver[["case_id", "class", "admissible", "independent"]], on="case_id", how="inner")
    df = df.merge(reg, on="case_id", how="inner")
    if len(df) != len(cases) or len(df) != 367152:
        raise ValueError("Case tables do not align")
    fb = df.fallback.to_numpy(bool)
    status = df.cert_status.to_numpy()

    # 1. Populations.
    pop = {"cases": int(len(df)), "surfaces": int(df.surface_id.nunique()),
           "native_accepted": int((~fb).sum()), "native_fallback": int(fb.sum()),
           "native_code_counts": {str(k): int(v) for k, v in df.native_code.value_counts().sort_index().items()},
           "certificate_status": {k: int(v) for k, v in df.cert_status.value_counts().items()},
           "status_by_fallback": {f"{s}|{'fallback' if f else 'accepted'}": int(v)
                                  for (s, f), v in df.groupby(["cert_status", "fallback"]).size().items()},
           "historical_class": {k: int(v) for k, v in df["class"].value_counts().items()},
           "C1": int(df.in_C1.sum()), "C2_recovered": int(df.in_C2_recovered.sum()),
           "C1_equals_fallback_and_admissible": bool(np.array_equal(df.in_C1.to_numpy(bool), fb & df.admissible.to_numpy(bool))),
           "fallback_certified_roots": int((fb & (status == "certified_root")).sum()),
           "C1_or_C2_equals_fallback_certified_roots": bool(np.array_equal((df.in_C1 | df.in_C2_recovered).to_numpy(bool),
                                                                         fb & (status == "certified_root"))),
           "models_in_C1": {f"{m1}/{m2}": int(v) for (m1, m2), v in df[df.in_C1].groupby(["model_a", "model_b"]).size().items()},
           "direction_in_C1": {str(k): int(v) for k, v in df[df.in_C1].direction.value_counts().items()}}
    out["populations"] = pop

    # 2. Coefficients, monotonicity and plateaus.
    ll = {k: (df[f"model_{k}"] == "LL.4").to_numpy() for k in "ab"}
    P = {k: [df[f"{k}_{p}"].to_numpy(float) for p in "bcde"] for k in "ab"}
    xa, xb = df.dose_a.to_numpy(float), df.dose_b.to_numpy(float)
    inc = {k: ((P[k][0] < 0) & (P[k][2] > P[k][1])) | ((P[k][0] > 0) & (P[k][2] < P[k][1])) for k in "ab"}
    flat = {k: (P[k][0] == 0) | (P[k][1] == P[k][2]) for k in "ab"}
    lo = {k: np.minimum(P[k][1], P[k][2]) for k in "ab"}
    hi = {k: np.maximum(P[k][1], P[k][2]) for k in "ab"}
    co_inc = inc["a"] & inc["b"] & ~flat["a"] & ~flat["b"]
    co_dec = ~inc["a"] & ~inc["b"] & ~flat["a"] & ~flat["b"]
    both_ll = ll["a"] & ll["b"]
    fa = curve(xa, *P["a"], ll["a"]); fb_ = curve(xb, *P["b"], ll["b"])
    fsa = curve(xa + xb, *P["a"], ll["a"]); fsb = curve(xa + xb, *P["b"], ll["b"])
    r_sd = np.minimum(100, np.maximum(fsa, fsb))
    r_h = np.minimum(100, np.maximum(fa, fb_))
    native = df.native_below_3.to_numpy(float)
    out["fallback_formulas"] = {
        "summed_dose_max_abs_diff_on_native_fallbacks": float(np.nanmax(np.abs(r_sd[fb] - native[fb]))),
        "hsa_max_abs_diff_vs_regimes_table_on_fallbacks": float(np.nanmax(np.abs(r_h[fb] - df.hsa_fallback_reference.to_numpy(float)[fb]))),
        "native_reference_matches_regimes_table": float(np.nanmax(np.abs(native - df.native_reference.to_numpy(float))))}

    # 3. Existence criterion against certificate status (co-monotone, non-flat, both LL.4).
    defined = both_ll & (co_inc | co_dec)
    joint_lo = np.maximum(lo["a"], lo["b"]); joint_hi = np.minimum(hi["a"], hi["b"])
    empty = joint_lo >= joint_hi
    i_is_a = np.where(co_inc, hi["a"] <= hi["b"], lo["a"] >= lo["b"])
    fj = np.where(i_is_a, fb_, fa)
    level = np.where(co_inc, np.where(i_is_a, hi["a"], hi["b"]), np.where(i_is_a, lo["a"], lo["b"]))
    equal = np.where(co_inc, hi["a"] == hi["b"], lo["a"] == lo["b"])
    margin = np.where(co_inc, level - fj, fj - level)
    predicted = np.where(equal, True, margin > 0)
    # Resolve float64 near-ties at 60 digits.
    close = defined & ~empty & ~equal & (np.abs(margin) < 1e-9 * np.maximum(1, np.abs(level)))
    mp.mp.dps = 60
    for r in np.flatnonzero(close):
        j = "b" if i_is_a[r] else "a"
        b, c, d, e = (mp.mpf(float(v[r])) for v in P[j])
        x = mp.mpf(float(xb[r] if j == "b" else xa[r]))
        fjr = c + (d - c) / (1 + (x / e) ** b)
        lev = mp.mpf(float(level[r]))
        predicted[r] = bool(fjr < lev) if co_inc[r] else bool(fjr > lev)
    check = defined & ~empty & np.isin(status, ["certified_root", "certified_no_root"])
    truth = status == "certified_root"
    out["existence_criterion"] = {
        "population": "both arms LL.4, co-monotone, non-flat, nonempty common interval, certificate root or no root",
        "cases": int(check.sum()), "co_increasing": int((check & co_inc).sum()), "co_decreasing": int((check & co_dec).sum()),
        "agreements": int((predicted[check] == truth[check]).sum()),
        "disagreements": int((predicted[check] != truth[check]).sum()),
        "float64_near_ties_resolved_at_60_digits": int(close.sum()),
        "defined_nonempty_but_certificate_undefined": int((defined & ~empty & ~np.isin(status, ["certified_root", "certified_no_root"])).sum()),
        "empty_interval_cases_matching_status": int((defined & empty & (status == "undefined_empty_interval")).sum()),
        "excluded_any_L4_arm": int((~both_ll).sum())}

    # 4. Replay of the stored high-precision brackets with an independent evaluation of g. A sign change at
    # two points strictly inside the common interval proves a root there (g is continuous and monotone).
    dl, du = df.dist_lower.to_numpy(float), df.dist_upper.to_numpy(float)
    root = df.root.to_numpy(float)
    blo, bhi = df.bracket_lo.to_numpy(), df.bracket_hi.to_numpy()
    targets = np.flatnonzero(fb & (status == "certified_root"))
    rng = np.random.default_rng(a.seed)
    acc = np.flatnonzero(~fb & (status == "certified_root"))
    controls = rng.choice(acc, size=min(a.controls, len(acc)), replace=False)
    near_acc = acc[np.minimum(dl, du)[acc] < 0.01]
    cert_dist_up = np.full(len(df), np.nan)   # certified distance of the bracket midpoint to the upper end
    bracket = {}
    for name, rows in [("fallback_roots", targets), ("accepted_controls", controls), ("accepted_near_asymptote", near_acc)]:
        ok, inside, skipped, failed = 0, 0, 0, []
        for r in rows:
            if not (ll["a"][r] and ll["b"][r]):
                skipped += 1
                continue
            digits = max(len(str(blo[r])), len(str(bhi[r])))
            mp.mp.dps = digits + 30
            arms = [tuple(mp.mpf(float(v[r])) for v in P[k]) for k in "ab"]
            doses = [mp.mpf(float(xa[r])), mp.mpf(float(xb[r]))]
            lo_r = max(min(p[1], p[2]) for p in arms); hi_r = min(max(p[1], p[2]) for p in arms)
            y1, y2 = mp.mpf(str(blo[r])), mp.mpf(str(bhi[r]))
            y1, y2 = min(y1, y2), max(y1, y2)
            ins = lo_r < y1 < y2 < hi_r
            inside += int(ins)
            good = ins and mp.sign(mp_g(y1, arms, doses)) * mp.sign(mp_g(y2, arms, doses)) < 0
            ok += int(good)
            if good:
                cert_dist_up[r] = float(hi_r - (y1 + y2) / 2)
            elif len(failed) < 5:
                failed.append(str(df.case_id.iloc[r]))
        bracket[name] = {"cases": int(len(rows)), "LL4_checked": int(len(rows) - skipped), "inside_common_interval": inside,
                         "sign_change_confirmed": ok, "examples_not_confirmed": failed}
    out["independent_bracket_replay"] = bracket

    # 5. Error of each fallback on rejected admissible roots.
    capped_root = np.minimum(100, root)
    errs = {}
    for name, mask in [("C1_14197", df.in_C1.to_numpy(bool)), ("C2_1423", df.in_C2_recovered.to_numpy(bool)),
                       ("all_15620", fb & (status == "certified_root"))]:
        e_sd, e_h = np.abs(r_sd[mask] - capped_root[mask]), np.abs(r_h[mask] - capped_root[mask])
        near = np.minimum(dl, du)[mask] < 0.01
        errs[name] = {"summed_dose": quantiles(e_sd), "hsa": quantiles(e_h), "share_hsa_closer": float(np.mean(e_h < e_sd)),
                      "within_0_01_of_asymptote": int(near.sum()), "interior": int((~near).sum()),
                      "asymptote_distance": quantiles(np.minimum(dl, du)[mask])}
    out["fallback_error_on_rejected_roots"] = errs
    rej = fb & (status == "certified_root")
    pd.DataFrame({"case_id": df.case_id.to_numpy()[rej], "C1": df.in_C1.to_numpy(bool)[rej], "C2": df.in_C2_recovered.to_numpy(bool)[rej],
                  "error_summed_dose": np.abs(r_sd - capped_root)[rej], "error_hsa": np.abs(r_h - capped_root)[rej],
                  "asymptote_distance": np.minimum(dl, du)[rej]}).to_csv(Path(a.derived) / "rejected_roots.csv", index=False)

    # 6. Location of co-increasing LL.4 C1 roots and the summed-dose jump at the upper plateau.
    c1 = df.in_C1.to_numpy(bool) & co_inc & both_ll
    up, low_ = c1 & (du < 0.01), c1 & (dl < 0.01) & ~(du < 0.01)
    inter = c1 & ~up & ~low_
    jump = r_sd - joint_hi
    shift = np.abs(native - capped_root)
    below_cap = up & (joint_hi < 100)
    dev = native - capped_root - (jump + (joint_hi - root))
    out["c1_location"] = {"co_increasing_LL4_C1": int(c1.sum()), "upper": int(up.sum()), "lower": int(low_.sum()),
                          "interior": int(inter.sum()), "upper_with_plateau_below_100": int(below_cap.sum()),
                          "jump_upper_plateau_below_100": quantiles(jump[below_cap]),
                          "jump_positive_share_plateau_below_100": float(np.mean(jump[below_cap] > 0)),
                          "identity_native_minus_root_equals_jump_plus_distance_max_abs": float(np.abs(dev[below_cap]).max()),
                          "share_of_absolute_shift_upper": float(shift[up].sum() / shift[c1].sum()),
                          "share_of_absolute_shift_lower": float(shift[low_].sum() / shift[c1].sum()),
                          "share_of_absolute_shift_interior": float(shift[inter].sum() / shift[c1].sum())}

    # 7. Boundary-layer law on certified roots within 0.01 of the smaller upper plateau.
    near_up = (status == "certified_root") & co_inc & both_ll & (du < 0.01) & ~equal
    ii = np.where(i_is_a, 0, 1)
    bi = np.where(ii == 0, P["a"][0], P["b"][0]); ci = np.where(ii == 0, P["a"][1], P["b"][1])
    di = np.where(ii == 0, P["a"][2], P["b"][2]); ei = np.where(ii == 0, P["a"][3], P["b"][3])
    xi = np.where(ii == 0, xa, xb); xj = np.where(ii == 0, xb, xa)
    bj = np.where(ii == 0, P["b"][0], P["a"][0]); cj = np.where(ii == 0, P["b"][1], P["a"][1])
    dj = np.where(ii == 0, P["b"][2], P["a"][2]); ej = np.where(ii == 0, P["b"][3], P["a"][3])
    with np.errstate(all="ignore"):
        Dj = ej * ((dj - level) / (level - cj)) ** (1 / bj)
        delta = 1 - xj / Dj
        log_pred = np.log10(np.abs(di - ci)) + np.abs(bi) * np.log10(delta * ei / xi)
    good = near_up & np.isfinite(log_pred) & (delta > 0) & np.isfinite(cert_dist_up) & (cert_dist_up > 0)
    cd = cert_dist_up
    resid = log_pred[good] - np.log10(cd[good])
    out["boundary_layer"] = {"population": "certified roots, both LL.4, co-increasing, within 0.01 of the smaller upper plateau",
                             "cases": int(near_up.sum()), "evaluable": int(good.sum()),
                             "log10_ratio_pred_over_certified": quantiles(np.abs(resid)),
                             "share_within_factor_1_01": float(np.mean(np.abs(resid) < np.log10(1.01))),
                             "share_within_factor_2": float(np.mean(np.abs(resid) < np.log10(2))),
                             "loglog_pearson_r": float(np.corrcoef(log_pred[good], np.log10(cd[good]))[0, 1]),
                             "certified_distance_decades": [float(np.log10(cd[good]).min()), float(np.log10(cd[good]).max())],
                             "distance_source": "midpoint of the replayed certificate bracket"}
    pd.DataFrame({"case_id": df.case_id.to_numpy()[good], "abs_slope": np.abs(bi[good]), "log10_predicted": log_pred[good],
                  "log10_certified": np.log10(cd[good]), "fallback": fb[good], "C1": df.in_C1.to_numpy(bool)[good],
                  "C2": df.in_C2_recovered.to_numpy(bool)[good]}).to_csv(Path(a.derived) / "boundary_layer_cases.csv", index=False)
    slope_c1 = np.abs(bi[df.in_C1.to_numpy(bool) & co_inc & both_ll])
    slope_all = np.abs(bi[(status == "certified_root") & co_inc & both_ll])
    out["boundary_layer"]["share_abs_slope_above_2"] = {"C1_co_increasing_LL4": float(np.mean(slope_c1 > 2)),
                                                        "all_certified_roots_co_increasing_LL4": float(np.mean(slope_all > 2))}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["existence_criterion", "independent_bracket_replay", "c1_location"]}))


if __name__ == "__main__":
    main()
