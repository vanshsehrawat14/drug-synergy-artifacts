"""Independent audit of Loewe screening decisions under numerical and convention changes.

Block Loewe score = mean observed inhibition over the block's 16 combination wells minus the mean case
reference of its fitted surface. Pair-cell score = mean of its replicate blocks. Rankings are descending
with ties broken by (pair_id, cell_line). Reference versions:
  N   native SynergyFinder references (Newton acceptance, summed-dose fallback, cap 100)
  C1  N with the 14,197 rejected admissible roots restored (certified root, cap 100)
  I   N with all 15,620 certified fallback roots restored
  H   N with HSA instead of the summed-dose fallback for every fallback case
  IH  I with HSA for every remaining fallback case (certified root where one exists, else HSA)
No project module is imported.

python manuscript/audit/screening_audit.py --observations data/processed/oneil_v1/observations.parquet \
    --blocks blocks.parquet --cases surface_cases.parquet --certificates certificates_replayable.parquet \
    --verified cases_verified.parquet --regimes loewe_regimes.csv --output manuscript/audit/outputs/screening_audit.json
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def top(score, k):
    n = math.ceil(len(score) * k)
    m = np.zeros(len(score), bool)
    m[np.argsort(-score, kind="stable")[:n]] = True
    return m


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "blocks", "cases", "certificates", "verified", "regimes", "output", "derived"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--halves", type=int, default=200)
    ap.add_argument("--seed", type=int, default=611)
    a = ap.parse_args()
    out = {"inputs": {k: sha256(getattr(a, k)) for k in ["observations", "blocks", "cases", "certificates", "verified", "regimes"]}}

    obs = pd.read_parquet(a.observations, columns=["block_id", "inhibition", "is_combination"])
    T = obs[obs.is_combination].groupby("block_id").inhibition.agg(["mean", "size"])
    blocks = pd.read_parquet(a.blocks, columns=["block_id", "pair_id", "cell_line", "surface_id", "native_sum_score", "synergy_loewe"])
    blocks["pair_id"] = blocks.pair_id.astype(str); blocks["cell_line"] = blocks.cell_line.astype(str)
    blocks = blocks.join(T, on="block_id")
    if (blocks["size"] != 16).any():
        raise ValueError("Expected 16 combination wells per block")
    cases = pd.read_parquet(a.cases, columns=["case_id", "surface_id", "native_below_3", "native_hsa_fallback", "fallback"])
    cert = pd.read_parquet(a.certificates, columns=["case_id", "cert_status", "root"])
    ver = pd.read_parquet(a.verified, columns=["case_id", "admissible", "independent"])
    reg = pd.read_csv(a.regimes, usecols=["case_id", "in_C1", "in_C2_recovered"])
    c = cases.merge(cert, on="case_id").merge(ver, on="case_id").merge(reg, on="case_id")
    fb = c.fallback.to_numpy(bool)
    has_root = (c.cert_status == "certified_root").to_numpy()
    native = c.native_below_3.to_numpy(float)
    root = np.minimum(100, c.root.to_numpy(float))
    c1 = c.in_C1.to_numpy(bool)
    refs = {"N": native,
            "C1": np.where(c1, root, native),
            "I": np.where(fb & has_root, root, native),
            "H": c.native_hsa_fallback.to_numpy(float)}
    refs["IH"] = np.where(fb & ~has_root, refs["H"], refs["I"])
    out["checks"] = {"C1_cases": int(c1.sum()), "I_cases": int((fb & has_root).sum()),
                     "C1_certificate_vs_float64_independent_root_max_abs": float(np.abs(
                         root[c1] - np.minimum(100, c.independent.to_numpy(float)[c1])).max()),
                     "cases_per_surface": sorted(set(c.groupby("surface_id").size().tolist()))}
    mean_ref = {v: pd.Series(r).groupby(c.surface_id.to_numpy()).mean() for v, r in refs.items()}
    bscore = {v: blocks["mean"].to_numpy() - blocks.surface_id.map(m).to_numpy(float) for v, m in mean_ref.items()}
    out["checks"]["native_block_score_vs_reconstruction_max_abs"] = float(np.abs(bscore["N"] - blocks.native_sum_score).max())
    out["checks"]["native_block_score_vs_published_drugcomb_max_abs"] = float(np.abs(bscore["N"] - blocks.synergy_loewe).max())
    dp = bscore["N"] - blocks.synergy_loewe.to_numpy(float)
    out["checks"]["native_vs_published"] = {"rmse": float(np.sqrt(np.mean(dp ** 2))),
                                            "pearson": float(np.corrcoef(bscore["N"], blocks.synergy_loewe)[0, 1]),
                                            "share_within_1e-6": float(np.mean(np.abs(dp) < 1e-6)),
                                            "share_within_0.01": float(np.mean(np.abs(dp) < 0.01)),
                                            "share_within_1": float(np.mean(np.abs(dp) < 1))}
    bscore["P"] = blocks.synergy_loewe.to_numpy(float)

    # Block-level label changes.
    dC1 = bscore["C1"] - bscore["N"]
    out["block_level"] = {"blocks": int(len(blocks)), "changed_by_C1": int((np.abs(dC1) > 1e-9).sum()),
                          "changed_share": float(np.mean(np.abs(dC1) > 1e-9)),
                          "sign_reversals_C1": int((np.sign(bscore["N"]) != np.sign(bscore["C1"])).sum())}

    # Pair-cell scores, canonical order.
    f = blocks[["pair_id", "cell_line", "block_id"]].copy()
    for v in list(refs) + ["P"]:
        f[v] = bscore[v]
    f = f.sort_values(["pair_id", "cell_line", "block_id"]).reset_index(drop=True)
    unit, ukeys = pd.factorize(f.pair_id + "|" + f.cell_line, sort=True)
    size = np.bincount(unit)
    pos = f.groupby(unit).cumcount().to_numpy()
    M = {v: np.full((len(size), size.max()), np.nan) for v in list(refs) + ["P"]}
    for v in M:
        M[v][unit, pos] = f[v].to_numpy()
    score = {v: np.nanmean(M[v], axis=1) for v in M}
    nunits = len(size)
    out["pair_cells"] = {"units": int(nunits), "block_counts": {int(k): int(v) for k, v in pd.Series(size).value_counts().items()},
                         "sign_reversals_C1": int((np.sign(score["N"]) != np.sign(score["C1"])).sum())}

    # Selections and cutoff structure.
    sel = {}
    for k in BUDGETS:
        n = math.ceil(nunits * k)
        row = {"selected": n}
        for v in refs:
            sel[(v, k)] = top(score[v], k)
            s = np.sort(score[v])[::-1]
            row[f"{v}_cutoff"] = float(s[n - 1]); row[f"{v}_gap_to_next"] = float(s[n - 1] - s[n])
            row[f"{v}_ties_at_cutoff"] = int((score[v] == s[n - 1]).sum() - 1)
        sel[("P", k)] = top(score["P"], k)
        row["P_to_N_turnover"] = float((sel[("P", k)] & ~sel[("N", k)]).sum() / n)
        row["P_to_C1_turnover"] = float((sel[("P", k)] & ~sel[("C1", k)]).sum() / n)
        for v in ["C1", "I", "H", "IH"]:
            row[f"N_to_{v}_replaced"] = int((sel[("N", k)] & ~sel[(v, k)]).sum())
            row[f"N_to_{v}_turnover"] = float((sel[("N", k)] & ~sel[(v, k)]).sum() / n)
        row["I_to_IH_turnover"] = float((sel[("I", k)] & ~sel[("IH", k)]).sum() / n)
        out[f"budget_{k}"] = row
    k = 0.05
    n5 = math.ceil(nunits * k)
    ent = sel[("C1", k)] & ~sel[("N", k)]; ext = sel[("N", k)] & ~sel[("C1", k)]
    cutN = np.sort(score["N"])[::-1][n5 - 1]
    cutC = np.sort(score["C1"])[::-1][n5 - 1]
    out["c1_top5"] = {"entrants": int(ent.sum()), "exits": int(ext.sum()),
                      "entrant_native_margin_below_cutoff": {"median": float(np.median(cutN - score["N"][ent])),
                                                             "p90": float(np.quantile(cutN - score["N"][ent], .9)),
                                                             "max": float((cutN - score["N"][ent]).max()),
                                                             "within_1_point": int(((cutN - score["N"][ent]) < 1).sum())},
                      "exit_native_margin_above_cutoff": {"median": float(np.median(score["N"][ext] - cutN)),
                                                          "max": float((score["N"][ext] - cutN).max())},
                      "entrant_score_change": {"median": float(np.median((score["C1"] - score["N"])[ent])),
                                               "mean": float(np.mean((score["C1"] - score["N"])[ent])),
                                               "min": float(np.min((score["C1"] - score["N"])[ent]))},
                      "exit_score_change": {"median": float(np.median((score["C1"] - score["N"])[ext])),
                                            "max_abs": float(np.abs(score["C1"] - score["N"])[ext].max())},
                      "native_cutoff": float(cutN), "c1_cutoff": float(cutC),
                      "units_changed_by_C1": int((np.abs(score["C1"] - score["N"]) > 1e-9).sum())}

    # Replicate resampling: blocks are resampled within pair-cell; references are fixed.
    rng = np.random.default_rng(a.seed)
    width = size.max()
    use = np.arange(width)[None, :] < size[:, None]

    def means(v, idx, mask):
        vals = np.take_along_axis(M[v], idx, axis=1)
        return np.where(mask, vals, 0).sum(1) / mask.sum(1)

    within, noise, pN, pC = [], [], np.zeros(nunits), np.zeros(nunits)
    prev = None
    for _ in range(a.boot):
        idx = np.floor(rng.random((nunits, width)) * size[:, None]).astype(int)
        sN, sC = top(means("N", idx, use), k), top(means("C1", idx, use), k)
        within.append((sN & ~sC).sum() / n5)
        pN += sN; pC += sC
        if prev is not None:
            noise.append((prev & ~sN).sum() / n5)
        prev = sN
    pN /= a.boot; pC /= a.boot
    halves_within, halves_noise = [], []
    for _ in range(a.halves):
        perm = np.argsort(rng.random((nunits, width)) + ~use * 2, axis=1)
        half = size[:, None] // 2
        m1 = np.arange(width)[None, :] < half
        # second half: positions half..size-1 of the same permutation
        second = np.take_along_axis(perm, np.clip(np.arange(width)[None, :] + half, 0, width - 1), axis=1)
        a1N, a1C = top(means("N", perm, m1), k), top(means("C1", perm, m1), k)
        a2N = top(means("N", second, m1), k)
        halves_within.append((a1N & ~a1C).sum() / n5)
        halves_noise.append((a1N & ~a2N).sum() / n5)
    q = lambda x: {"mean": float(np.mean(x)), "low": float(np.quantile(x, .025)), "high": float(np.quantile(x, .975)),
                   "min": float(np.min(x))}
    kept = sel[("N", k)] & sel[("C1", k)]
    out["replicate_resampling"] = {
        "bootstrap_draws": a.boot, "half_splits": a.halves,
        "within_resample_N_to_C1_turnover_bootstrap": q(within),
        "within_resample_N_to_C1_turnover_half": q(halves_within),
        "noise_turnover_consecutive_bootstrap_N": q(noise),
        "noise_turnover_half_depth_N": q(halves_noise),
        "selection_probability": {"entrants": {"N": float(pN[ent].mean()), "C1": float(pC[ent].mean()),
                                               "share_N_below_0_1": float(np.mean(pN[ent] < .1)),
                                               "share_C1_at_least_0_5": float(np.mean(pC[ent] >= .5))},
                                  "exits": {"N": float(pN[ext].mean()), "C1": float(pC[ext].mean())},
                                  "kept": {"N": float(pN[kept].mean()), "C1": float(pC[kept].mean())}}}
    Path(a.derived).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pair_cell": np.asarray(ukeys), **{f"score_{v}": score[v] for v in score},
                  "selected_N": sel[("N", k)], "selected_C1": sel[("C1", k)], "p_select_N": pN, "p_select_C1": pC}).to_csv(
        Path(a.derived) / "pair_cell_scores.csv", index=False)
    Path(a.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["checks"]), json.dumps(out["c1_top5"]))


if __name__ == "__main__":
    main()
