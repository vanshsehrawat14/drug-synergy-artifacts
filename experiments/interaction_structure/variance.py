"""Experiment A: target audit, replicate and reference dependence, cross-fitted variance components, controls.

python experiments/interaction_structure/variance.py --observations data/processed/oneil_v1/observations.parquet \
    --resolution RESOLUTION_DIR --output results/interaction_structure
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (FAMILIES, Covariance, Data, Panel, boot_ratio, components, drug_leaves, jackknife,  # noqa: E402
                  loco_b2, sha256, shares, train_baseline)

LAMS, RANKS = [0.0, 0.25, 0.5, 0.75], [1, 2, 3, 5, 8, 12, 20, 38]
CONFIGS = [(1.0, 0)] + [(lam, r) for lam in LAMS for r in RANKS]


def with_intervals(t, w, v, data, draws, seed):
    st = components(t, w, v)
    out = {k: float(x) for k, x in shares(st).items()}
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(st), size=(draws, len(st)))
    bs = [shares(st[i].sum(0)) for i in ix]
    leaves = [shares(st[keep].sum(0)) for keep in drug_leaves(data).values()]
    cells = [shares(components(np.delete(t, c, 1), np.delete(w, c, 1), np.delete(v, c, 1)).sum(0)) for c in range(t.shape[1])]
    for k in ["noise", "pair_wide", "pair_by_cell", "descriptive_pair_mean"]:
        out[k + "_pair_interval"] = [float(np.quantile([b[k] for b in bs], q)) for q in (.025, .975)]
        out[k + "_drug_interval"] = jackknife(out[k], [lv[k] for lv in leaves])
        out[k + "_cell_interval"] = jackknife(out[k], [lv[k] for lv in cells])
    return out


def loco(data, t, v, select=True):
    """Leave-one-cell-out B1 and B2 per test fold; covariance from the other folds, B2 config chosen on fold f+1."""
    pred1, pred2, chosen = np.zeros_like(t), np.zeros_like(t), {}
    C = t.shape[1]
    for f in range(5):
        val = (f + 1) % 5
        test = data.fold == f
        cfg = (1.0, 0)
        if select:
            inner = ~np.isin(data.fold, [f, val])
            cov = Covariance(t[inner], v[inner])
            vm = data.fold == val
            sse = {}
            for lam, r in CONFIGS:
                p = loco_b2(cov.signal(lam, r), cov.N, t[vm][..., None])[..., 0]
                sse[(lam, r)] = float((data.w[vm] * (t[vm] - p) ** 2).sum())
            cfg = min(sse, key=sse.get)
        cov = Covariance(t[~test], v[~test])
        chosen[f] = list(cfg)
        pred2[test] = loco_b2(cov.signal(*cfg), cov.N, t[test][..., None])[..., 0]
        wk = cov.b1_weight(C - 1)
        pred1[test] = wk * (t[test].sum(1, keepdims=True) - t[test]) / (C - 1)
    return pred1, pred2, chosen


def r2_parts(data, t, preds, draws, seed):
    w = data.w
    T = (w * t ** 2).sum(1)
    out = {}
    for name, p in preds.items():
        E = (w * (t - p) ** 2).sum(1)
        leaves = [1 - E[k].sum() / T[k].sum() for k in drug_leaves(data).values()]
        r2 = 1 - E.sum() / T.sum()
        out[name] = {"r2": float(r2), "pair_interval": boot_ratio(E, T, draws, seed), "drug_interval": jackknife(r2, leaves)}
    if {"B1", "B2"} <= set(preds):
        E1, E2 = (w * (t - preds["B1"]) ** 2).sum(1), (w * (t - preds["B2"]) ** 2).sum(1)
        d = (E1.sum() - E2.sum()) / T.sum()
        leaves = [(E1[k].sum() - E2[k].sum()) / T[k].sum() for k in drug_leaves(data).values()]
        out["B2_minus_B1"] = {"r2": float(d), "pair_interval": boot_ratio(np.column_stack([E1, E2]), T, draws, seed,
                                                                           lambda n, dd: (n[0] - n[1]) / dd),
                              "drug_interval": jackknife(d, leaves)}
    return out


def pair_decomposition(w, t, p):
    """R2 of a prediction on pair means (across cells) and on pair-centred residuals."""
    d1 = w.sum(1)
    a, ah = (w * t).sum(1) / d1, (w * p).sum(1) / d1
    g, gh = t - a[:, None], p - ah[:, None]
    return {"pair_mean_r2": float(1 - (d1 * (a - ah) ** 2).sum() / (d1 * a ** 2).sum()),
            "residual_r2": float(1 - (w * (g - gh) ** 2).sum() / (w * g ** 2).sum()),
            "pair_mean_energy_share_of_target": float((d1 * a ** 2).sum() / (w * t ** 2).sum()),
            "total_r2": float(1 - (w * (t - p) ** 2).sum() / (w * t ** 2).sum())}


def eligible_retest(data, tol=0.02):
    """8-block pair-cells whose two batches use the same dose grid up to rounding."""
    comb = data.comb
    g = comb.groupby("block_id").agg(ga=("dose_a", lambda s: tuple(sorted(set(s)))), gb=("dose_b", lambda s: tuple(sorted(set(s)))))
    bl = data.blocks.join(g)
    cnt = bl.groupby("flat").size()
    out, excluded = [], []
    for flat in cnt[cnt == 8].index:
        b = bl[bl.flat == flat]
        g1, g2 = b[b.batch == 0], b[b.batch == 1]
        if g1.ga.nunique() != 1 or g2.ga.nunique() != 1 or len(g1) != 4 or len(g2) != 4:
            excluded.append(flat)
            continue
        ra = np.log(np.array(g2.ga.iloc[0]) / np.array(g1.ga.iloc[0]))
        rb = np.log(np.array(g2.gb.iloc[0]) / np.array(g1.gb.iloc[0]))
        (out if max(np.abs(ra).max(), np.abs(rb).max()) <= tol else excluded).append(flat)
    return np.array(sorted(out)), np.array(sorted(excluded))


def cross_batch(data, flats, M_base, row_ref, signs_batch, D_frozen, draws, seed):
    """Batch-specific targets for re-measured pair-cells and the cross-batch moments."""
    P, C = data.shape
    w_x = data.w.copy()
    w_x.flat[flats] = 64.0
    panel = Panel(data, w=w_x)
    in_el = np.isin(data.flat_row, flats)
    res = {}
    y_t = data.y - row_ref if row_ref is not None else data.y
    m = [data.pc_mean_rows(y_t, rows=in_el & (data.row_batch == b)) for b in (0, 1)]
    for adjust in ["none", "cell_shift"]:
        Ms = []
        for b in (0, 1):
            Mb = M_base.copy()
            vals = m[b].flat[flats]
            if adjust == "cell_shift" and b == 1:
                diff = pd.Series(m[1].flat[flats] - m[0].flat[flats]).groupby(flats % C).transform("mean").to_numpy()
                vals = vals - diff
            Mb.flat[flats] = vals
            Ms.append(Mb)
        T = [panel(Mb).reshape(-1)[flats] for Mb in Ms]
        # Noise: within-batch split at re-measured pair-cells, the frozen split elsewhere.
        PDs = []
        fp, fc = np.divmod(flats, C)
        for b in (0, 1):
            Db = D_frozen.copy()
            rows = in_el & (data.row_batch == b)
            for r in range(draws):
                s = signs_batch[data.row_block[rows], r] * data.y[rows]
                Db[fp, fc, r] = np.bincount(data.flat_row[rows], weights=s, minlength=P * C)[flats] / 64.0
            PDs.append(panel(Db).reshape(P * C, -1)[flats])
        v = [(pd_ ** 2).mean(1) for pd_ in PDs]
        xn = (PDs[0] * PDs[1]).mean(1)
        pid = flats // C
        upairs = np.unique(pid)
        stats = []
        for p in upairs:
            k = pid == p
            t1, t2, v1, v2, x0 = T[0][k], T[1][k], v[0][k], v[1][k], xn[k]
            n = k.sum()
            stats.append([((t1 ** 2 - v1) + (t2 ** 2 - v2)).sum() / 2, (t1 * t2 - x0).sum(), n,
                          ((t1.sum() ** 2 - (t1 ** 2).sum()) + (t2.sum() ** 2 - (t2 ** 2).sum())) / 2,
                          t1.sum() * t2.sum() - (t1 * t2).sum(), n * (n - 1), (t1 - t2).sum(), ((t1 - t2) ** 2).sum(),
                          (v1 + v2).sum()])
        stats = np.array(stats)

        def summarize(s):
            E1, X, n, Pw, Px, nn, dsum, d2, vsum = s.sum(0)
            E1, X, Pw, Px = E1 / n, X / n, Pw / nn, Px / nn
            return {"within_batch_reproducible": E1, "cross_batch_reproducible": X, "within_pair_wide": Pw,
                    "cross_pair_wide": Px, "within_pair_by_cell": E1 - Pw, "cross_pair_by_cell": X - Px,
                    "replication_overall": X / E1, "replication_pair_wide": Px / Pw,
                    "replication_pair_by_cell": (X - Px) / (E1 - Pw), "mean_shift": dsum / n,
                    "msd": d2 / n, "msd_over_within_noise": (d2 / n) / (vsum / n)}

        out = {k: float(v_) for k, v_ in summarize(stats).items()}
        rng = np.random.default_rng(seed)
        bs = [summarize(stats[rng.integers(len(stats), size=len(stats))]) for _ in range(2000)]
        for k in list(out):
            out[k + "_pair_interval"] = [float(np.nanquantile([b[k] for b in bs], q)) for q in (.025, .975)]
        a1 = pd.Series(T[0]).groupby(pid).mean().to_numpy()
        a2 = pd.Series(T[1]).groupby(pid).mean().to_numpy()
        g1, g2 = T[0] - a1[np.searchsorted(upairs, pid)], T[1] - a2[np.searchsorted(upairs, pid)]
        out.update(corr_pc=float(np.corrcoef(T[0], T[1])[0, 1]), corr_pair_means=float(np.corrcoef(a1, a2)[0, 1]),
                   corr_residuals=float(np.corrcoef(g1, g2)[0, 1]), noise_mean_square=float((v[0].mean() + v[1].mean()) / 2),
                   target_mean_square=float((T[0] ** 2).mean() / 2 + (T[1] ** 2).mean() / 2))
        res[adjust] = out
        res[adjust + "_values"] = {"t1": T[0].tolist(), "t2": T[1].tolist(), "pair": [data.pairs[i] for i in pid],
                                   "cell": [data.cells[i] for i in flats % C]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", required=True)
    ap.add_argument("--resolution", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--derived", default=None, help="directory for per-pair-cell targets (not committed)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    t0 = time.time()
    design = json.loads((Path(__file__).resolve().parent / "design.json").read_text())
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Data(a.observations, a.resolution)
    res = Path(a.resolution)
    out = {"inputs": {"observations_sha256": sha256(a.observations), "cache_sha256": sha256(res / "cache.npz"),
                      "coordinates_sha256": sha256(res / "coordinates.parquet")}}
    if out["inputs"]["observations_sha256"] != design["inputs"]["observations"]["sha256"]:
        raise ValueError("Observation table differs from the frozen design")
    P, C = data.shape
    w = data.w
    panel = Panel(data)

    # ---- Audit: units, blocks, batches, reference sharing.
    comb, bl = data.comb, data.blocks
    edge = data.mono_a + data.mono_b - data.mono_a * data.mono_b / 100
    edge_u = np.bincount(data.unit_of_row, weights=edge) / data.n_u
    diff_u = np.abs(edge_u - data.cache["bliss"])
    big = diff_u > 0.01
    flat_u = data.flat_unit
    nblocks = bl.groupby("flat").size()
    eligible, excluded = eligible_retest(data)
    flats8 = nblocks[nblocks == 8].index.to_numpy()
    out["audit"] = {
        "pairs": P, "cells": C, "pair_cells": int(P * C), "missing_pair_cells": 0, "drugs": len(data.drugs),
        "combination_rows": int(len(data.y)), "units": int(len(data.n_u)),
        "unit_replicate_counts": {int(k): int(v) for k, v in zip(*np.unique(data.n_u, return_counts=True))},
        "blocks": int(len(bl)), "blocks_per_pair_cell": {int(k): int(v) for k, v in nblocks.value_counts().items()},
        "batch2_blocks": int(bl.batch.sum()), "batch2_pair_cells": int(len(flats8)),
        "batch2_pairs": int(len(np.unique(flats8 // C))), "batch2_cells": int(len(np.unique(flats8 % C))),
        "retest_eligible_pair_cells": int(len(eligible)), "retest_eligible_pairs": int(len(np.unique(eligible // C))),
        "retest_dose_changed_pair_cells": int(len(excluded)),
        "retest_dose_changed_pairs": sorted({data.pairs[i] for i in np.unique(excluded // C)}),
        "row_weight_counts": {int(k): int(v) for k, v in zip(*np.unique(w, return_counts=True))},
        "truth_max_abs_diff_vs_cache": data.truth_cache_diff,
        "frozen_vs_block_edge_bliss": {"max_abs": float(diff_u.max()), "mean_abs": float(diff_u.mean()),
                                       "units_over_0.01": int(big.sum()),
                                       "pair_cells_with_units_over_0.01": int(len(np.unique(flat_u[big]))),
                                       "share_in_batch2_pair_cells": float(np.isin(flat_u[big], flats8).mean()) if big.any() else 0.0},
        "pair_folds": {int(f): int((data.fold == f).sum()) for f in range(5)}}
    # Monotherapy sharing: within pair-cell across blocks, and across pairs.
    keys = []
    for e in "ab":
        keys.append(pd.DataFrame({"drug": comb["drug_" + e].to_numpy(), "cell": comb.cell_line.to_numpy(),
                                  "dose": comb["dose_" + e].to_numpy(), "value": comb["mono_" + e].to_numpy(),
                                  "block": comb.block_id.to_numpy(), "flat": data.flat_row,
                                  "batch": data.row_batch}).drop_duplicates(["block", "dose", "drug"]))
    mono = pd.concat(keys).drop_duplicates(["block", "drug", "dose"])
    within = mono.groupby(["flat", "drug", "dose", "batch"]).value.nunique()
    across = mono[mono.batch == 0].groupby(["drug", "cell", "dose"]).value.agg(["std", "size"])
    out["audit"]["monotherapy"] = {
        "edge_values_per_block": int(len(mono)),
        "groups_within_pair_cell_batch": int(len(within)), "groups_with_one_distinct_value": int((within == 1).sum()),
        "drug_cell_dose_groups_across_pairs": int(len(across)),
        "across_pair_pooled_sd": float(np.sqrt((across["std"].fillna(0) ** 2 * (across["size"] - 1)).sum() / (across["size"] - 1).sum())),
        "max_across_pair_sd": float(across["std"].max())}
    el_rows = np.isin(data.flat_row, eligible)
    el = mono[mono.flat.isin(eligible)].sort_values("dose")
    tup = el.groupby(["flat", "drug", "batch"]).value.apply(lambda s: tuple(np.round(s.drop_duplicates().to_numpy(), 6))).unstack("batch")
    same = tup[0] == tup[1]
    m2 = tup
    out["audit"]["monotherapy"]["retest_edges_identical_to_batch1"] = {"drug_pair_cells": int(len(m2)), "identical": int(same.sum())}
    print("audit", round(time.time() - t0), flush=True)

    # ---- Targets.
    Mt = data.pc_mean_units(data.truth_u)
    Mref = data.pc_mean_units(data.cache["bliss"])
    hsa = np.maximum(data.mono_a, data.mono_b)
    T = {"bliss": panel(Mt - Mref), "raw": panel(Mt), "ref": panel(Mref),
         "hsa": panel(data.pc_mean_rows(data.y - hsa)), "bliss_edge": panel(data.pc_mean_rows(data.y - edge))}
    signs = data.split_signs(50, 97, "grid")
    D = data.split_differences(signs)
    PD = panel(D)
    V = (PD ** 2).mean(-1)
    signs_b = data.split_signs(50, 97, "batch")
    Db = data.split_differences(signs_b)
    Vb = (panel(Db) ** 2).mean(-1)
    # Cross-fitted train-only baseline residual: each fold's pairs against a baseline fitted on the other folds.
    Mbr = Mt - Mref
    Ttrain, Vtrain = np.empty_like(Mbr), np.empty_like(Mbr)
    for f in range(5):
        r = train_baseline(data, np.concatenate([Mbr[..., None], D], -1), {f})
        k = data.fold == f
        Ttrain[k] = r[k, :, 0]
        Vtrain[k] = (r[k, :, 1:] ** 2).mean(-1)
    T["train"] = Ttrain
    noise = {"bliss": V, "raw": V, "ref": np.zeros_like(V), "hsa": V, "bliss_edge": V, "train": Vtrain}
    out["reproduction"] = {"bliss_noise_share": float((w * V).sum() / (w * T["bliss"] ** 2).sum()),
                           "raw_noise_share": float((w * V).sum() / (w * T["raw"] ** 2).sum()),
                           "bliss_energy_per_row": float((w * T["bliss"] ** 2).sum() / w.sum()),
                           "bliss_unweighted_cross_cell_share": float((((T["bliss"].sum(1) ** 2 - (T["bliss"] ** 2).sum(1)) / (C - 1)).sum()) / (T["bliss"] ** 2).sum()),
                           "batch_split_bliss_noise_share": float((w * Vb).sum() / (w * T["bliss"] ** 2).sum()),
                           "frozen_reference": {"bliss_noise_share": 0.09780410529352931, "bliss_energy_per_row": 20.588451181506063,
                                                "bliss_cross_cell_share": 0.23374990682143693, "raw_noise_share": 0.06713301836787676}}
    print("targets", round(time.time() - t0), flush=True)

    # ---- Components with intervals.
    comp = {}
    for name in ["bliss", "raw", "ref", "hsa", "bliss_edge", "train"]:
        comp[name] = with_intervals(T[name], w, noise[name], data, a.boot, 11)
    comp["bliss_batch_split_noise"] = with_intervals(T["bliss"], w, Vb, data, a.boot, 11)
    # Reference part of the Bliss target's pair-wide structure: covariance algebra T_raw = T_ref + T_bliss.
    def tau2(x, y_):
        s = ((w * x).sum(1) * (w * y_).sum(1) - (w ** 2 * x * y_).sum(1)).sum()
        return s / ((w.sum(1) ** 2 - (w ** 2).sum(1)).sum())
    eb = (w * T["bliss"] ** 2).sum() / w.sum()
    comp["pair_wide_algebra"] = {"tau2_raw": tau2(T["raw"], T["raw"]), "tau2_bliss": tau2(T["bliss"], T["bliss"]),
                                 "tau2_ref": tau2(T["ref"], T["ref"]), "cross_ref_bliss": tau2(T["ref"], T["bliss"]),
                                 "bliss_energy_per_row": eb}
    pm = {nm: (w * T[nm]).sum(1) / w.sum(1) for nm in T}
    comp["pair_mean_correlations"] = {f"{x}~{y_}": float(np.corrcoef(pm[x], pm[y_])[0, 1]) for x, y_ in
                                      [("bliss", "raw"), ("bliss", "ref"), ("raw", "ref"), ("bliss", "hsa"), ("bliss", "bliss_edge"), ("bliss", "train")]}
    out["components"] = comp
    print("components", round(time.time() - t0), flush=True)

    # ---- Cross-cell covariance structure and leave-one-cell-out predictable shares.
    cov = Covariance(T["bliss"], V)
    ev = np.sort(np.linalg.eigvalsh(cov.Csig))[::-1]
    u1 = np.linalg.eigh(cov.Csig)[1][:, -1]
    out["covariance_structure"] = {"eigenvalues_noise_corrected": ev.tolist(), "trace": float(ev.sum()),
                                   "top_eigenvalue_share": float(ev[0] / ev.sum()),
                                   "top_eigenvector_cosine_with_constant": float(abs(u1.sum()) / np.sqrt(C)),
                                   "compound_symmetry_tau2": cov.tau2,
                                   "offdiagonal_sd": float(cov.S[~np.eye(C, dtype=bool)].std())}
    lo = {}
    for name in ["bliss", "raw", "ref", "train"]:
        p1, p2, chosen = loco(data, T[name], noise[name])
        lo[name] = r2_parts(data, T[name], {"B1": p1, "B2": p2}, a.boot, 11)
        lo[name]["b2_configs"] = chosen
        lo[name]["noise_share"] = float((w * noise[name]).sum() / (w * T[name] ** 2).sum())
        if name == "bliss":
            loco_bliss = (p1, p2)
    out["loco"] = lo
    print("loco", round(time.time() - t0), flush=True)

    # ---- Held-out-pair models: pair-mean versus pair-centred parts.
    dec = {}
    for fam, names in FAMILIES.items():
        parts = [pair_decomposition(w, T["bliss"], panel(Mt - Mref) - panel(Mt - data.pc_mean_units(data.cache[nm]))) for nm in names]
        dec[fam] = {k: float(np.mean([p_[k] for p_ in parts])) for k in parts[0]}
    dec["loco_B1"] = pair_decomposition(w, T["bliss"], loco_bliss[0])
    dec["loco_B2"] = pair_decomposition(w, T["bliss"], loco_bliss[1])
    out["held_out_pair_models"] = dec

    # ---- Negative controls.
    ctrl = {}
    rng = np.random.default_rng(17)
    perm = []
    for _ in range(20):
        tp, vp = T["bliss"].copy(), V.copy()
        for f in range(5):
            idx = np.flatnonzero(data.fold == f)
            for c in range(C):
                j = rng.permutation(idx)
                tp[idx, c], vp[idx, c] = T["bliss"][j, c], V[j, c]
        sh = shares(components(tp, w, vp).sum(0))
        p1, _, _ = loco(data, tp, vp, select=False)
        perm.append([sh["pair_wide"], sh["pair_by_cell"], 1 - (w * (tp - p1) ** 2).sum() / (w * tp ** 2).sum()])
    perm = np.array(perm)
    ctrl["pair_permutation"] = {"pair_wide_mean": float(perm[:, 0].mean()), "pair_wide_sd": float(perm[:, 0].std()),
                                "pair_wide_max": float(perm[:, 0].max()), "loco_b1_r2_mean": float(perm[:, 2].mean()),
                                "loco_b1_r2_max": float(perm[:, 2].max()), "permutations": 20}
    rng = np.random.default_rng(19)
    zi = []
    ybar = data.truth_u[data.unit_of_row]
    scale = np.sqrt(data.n_u / (data.n_u - 1))[data.unit_of_row]
    ref_row = data.cache["bliss"][data.unit_of_row]
    for _ in range(5):
        sb = rng.choice([-1.0, 1.0], size=len(bl))[data.row_block]
        ystar = ref_row + sb * (data.y - ybar) * scale
        tz = panel(data.pc_mean_rows(ystar - ref_row))
        vz = (panel(data.split_differences(signs[:, :20], ystar)) ** 2).mean(-1)
        sh = shares(components(tz, w, vz).sum(0))
        p1, p2, _ = loco(data, tz, vz)
        zi.append([sh["noise"], sh["pair_wide"], 1 - (w * (tz - p1) ** 2).sum() / (w * tz ** 2).sum(),
                   1 - (w * (tz - p2) ** 2).sum() / (w * tz ** 2).sum(), (w * tz ** 2).sum() / w.sum()])
    zi = np.array(zi)
    ctrl["zero_interaction"] = {"noise_share_mean": float(zi[:, 0].mean()), "noise_share_range": [float(zi[:, 0].min()), float(zi[:, 0].max())],
                                "pair_wide_mean": float(zi[:, 1].mean()), "loco_b1_r2_mean": float(zi[:, 2].mean()),
                                "loco_b2_r2_mean": float(zi[:, 3].mean()), "energy_per_row_mean": float(zi[:, 4].mean()),
                                "realizations": 5}
    # Reference perturbation: shared monotherapy values perturbed per (drug, cell, dose).
    rng = np.random.default_rng(23)
    ka = pd.Series(comb.drug_a.to_numpy()) + "|" + comb.cell_line.to_numpy() + "|" + comb.dose_a.astype(str).to_numpy()
    kb = pd.Series(comb.drug_b.to_numpy()) + "|" + comb.cell_line.to_numpy() + "|" + comb.dose_b.astype(str).to_numpy()
    allkeys = pd.Index(pd.concat([ka, kb]).unique())
    ia, ib = allkeys.get_indexer(ka), allkeys.get_indexer(kb)
    pert = {}
    base = T["bliss_edge"]
    for s in [2.0, 5.0, 10.0]:
        rows = []
        for _ in range(10):
            delta = rng.normal(0, s, len(allkeys))
            a_, b_ = np.clip(data.mono_a + delta[ia], -100, 200), np.clip(data.mono_b + delta[ib], -100, 200)
            tp = panel(data.pc_mean_rows(data.y - (a_ + b_ - a_ * b_ / 100)))
            dT = tp - base
            sh = shares(components(tp, w, V).sum(0))
            rows.append([(w * dT ** 2).sum() / (w * base ** 2).sum(), sh["pair_wide"],
                         np.corrcoef((w * tp).sum(1), (w * base).sum(1))[0, 1], shares(components(dT, w, 0 * V).sum(0))["pair_wide"]])
        rows = np.array(rows)
        pert[str(s)] = {"induced_energy_share": float(rows[:, 0].mean()), "pair_wide_share_perturbed": float(rows[:, 1].mean()),
                        "pair_mean_corr": float(rows[:, 2].mean()), "pair_wide_share_of_perturbation": float(rows[:, 3].mean())}
    ctrl["reference_perturbation"] = pert
    ctrl["reference_only"] = {"energy_share_of_raw": float((w * T["ref"] ** 2).sum() / (w * T["raw"] ** 2).sum())}
    out["controls"] = ctrl
    print("controls", round(time.time() - t0), flush=True)

    # ---- Cross-batch re-measurement.
    cb = {"eligible_pair_cells": int(len(eligible))}
    unit_ref_row = data.cache["bliss"][data.unit_of_row]
    for name, ref_row_, M_base in [("raw", None, Mt), ("bliss_frozen_reference", unit_ref_row, Mt - Mref),
                                   ("bliss_own_edges", edge, Mt - Mref)]:
        cb[name] = cross_batch(data, eligible, M_base, ref_row_, signs_b, D, 50, 13)
    # Representativeness of the re-tested pairs under the frozen target.
    keep = np.isin(np.arange(P), np.unique(eligible // C))
    cb["retested_pairs_frozen_components"] = {k: float(v) for k, v in shares(components(T["bliss"][keep], w[keep], V[keep]).sum(0)).items()}
    cb["all_pairs_frozen_components"] = {k: float(v) for k, v in shares(components(T["bliss"], w, V).sum(0)).items()}
    values = {f"{k}:{adj}": cb[k].pop(adj + "_values") for k in ["raw", "bliss_frozen_reference", "bliss_own_edges"]
              for adj in ["none", "cell_shift"]}
    out["cross_batch"] = cb
    print("cross-batch", round(time.time() - t0), flush=True)

    out["runtime_seconds"] = round(time.time() - t0, 1)
    (out_dir / "variance_components.json").write_text(json.dumps(out, indent=1))
    if a.derived:
        d = Path(a.derived)
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, p in enumerate(data.pairs):
            for c, cell in enumerate(data.cells):
                rows.append([p, cell, data.drug_a[i], data.drug_b[i], int(data.fold[i]), w[i, c],
                             *[T[k][i, c] for k in ["bliss", "raw", "ref", "hsa", "train"]], V[i, c], Vb[i, c], Vtrain[i, c]])
        pd.DataFrame(rows, columns=["pair_id", "cell_line", "drug_a", "drug_b", "fold", "w", "t_bliss", "t_raw", "t_ref", "t_hsa",
                                    "t_train", "noise_var", "noise_var_batch_split", "noise_var_train"]).to_parquet(d / "pair_cell_targets.parquet", index=False)
        np.save(d / "split_projected.npy", PD.astype(np.float32))
        first = next(iter(values.values()))
        pd.DataFrame({"pair": first["pair"], "cell": first["cell"],
                      **{f"{k}:t{b}": values[k][f"t{b}"] for k in values for b in (1, 2)}}).to_csv(d / "cross_batch_values.csv", index=False)
    print(json.dumps({k: out[k] for k in ["reproduction"]}, indent=1))


if __name__ == "__main__":
    main()
