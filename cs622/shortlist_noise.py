"""Combination-replicate noise in the Loewe pair-cell shortlist versus the fixed-data C1 change (configs/final_20261110.json, A2).

python -m cs622.shortlist_noise --verified VERIFIED --certificates CERTS --output OUT

Block scores use the historical fixed-fit references (cs622.decision_sensitivity). Only replicate blocks are
resampled; monotherapy, fits and references stay fixed, so this is combination-measurement noise alone.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_sensitivity import BUDGETS, KEY, references, selected, unit_scores
from .io import digest

SEED = 20261110
VERSIONS = ["N", "C1", "I"]


def block_matrix(blocks, scores):
    """Pair-cells (canonical KEY order) by replicate block, NaN-padded; columns follow block_id."""
    f = blocks[KEY + ["block_id"]].copy()
    for v, s in scores.items():
        f[v] = s
    f = f.sort_values(KEY + ["block_id"]).reset_index(drop=True)
    unit, units = pd.factorize(pd.MultiIndex.from_frame(f[KEY]), sort=True)
    position = f.groupby(unit).cumcount().to_numpy()
    size = np.bincount(unit)
    m = {v: np.full((len(size), size.max()), np.nan) for v in scores}
    for v in scores:
        m[v][unit, position] = f[v].to_numpy()
    return m, size, units.to_frame(index=False)


def top(score, k):
    return selected(score, k)[0]


def replaced(a, b):
    return (a & ~b).sum() / a.sum()


def resample_means(m, size, index):
    """index: units x max_blocks integer positions (only the first size[u] used per unit)."""
    cols = np.arange(m.shape[1])
    use = cols[None, :] < size[:, None]
    vals = np.take_along_axis(m, index, axis=1)
    return np.where(use, vals, 0).sum(1) / size


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cases = pd.read_parquet(a.cases)
    blocks = pd.read_parquet(a.blocks)
    for c in KEY:
        blocks[c] = blocks[c].astype(str)
    ref, _ = references(cases, pd.read_parquet(a.verified), pd.read_csv(a.precision), pd.read_parquet(a.certificates))
    scores = {v: unit_scores(cases, blocks, ref[v]) for v in VERSIONS}
    if not np.allclose(scores["N"], blocks.native_sum_score.to_numpy(float)):
        raise AssertionError("Native block scores not reproduced")
    m, size, units = block_matrix(blocks, scores)
    if not set(size) <= {4, 8}:
        raise ValueError("Unexpected replicate counts")
    full = {v: np.nanmean(m[v], axis=1) for v in VERSIONS}
    sel = {(v, k): top(full[v], k) for v in VERSIONS for k in BUDGETS}
    k5 = 0.05
    entrants = sel[("C1", k5)] & ~sel[("N", k5)]
    exits = sel[("N", k5)] & ~sel[("C1", k5)]
    if entrants.sum() != 67 or exits.sum() != 67:
        raise AssertionError("Historical 67/1,137 C1 change not reproduced")
    rng = np.random.default_rng(SEED)
    width = m["N"].shape[1]
    rows = []
    # Equal block halves within pair-cell: two independent half-depth screens.
    half = {key: [] for key in ["noise_N", "noise_C1", "noise_I", "N_to_C1_half"]}
    for _ in range(a.halves):
        perm = np.argsort(rng.random((len(size), width)) + (np.arange(width)[None, :] >= size[:, None]), axis=1)
        h = size // 2
        second = np.take_along_axis(perm, (np.arange(width)[None, :] % h[:, None]) + h[:, None], axis=1)
        first = np.take_along_axis(perm, np.arange(width)[None, :] % h[:, None], axis=1)
        s1 = {v: resample_means(m[v], h, first) for v in VERSIONS}
        s2 = {v: resample_means(m[v], h, second) for v in VERSIONS}
        for k in BUDGETS:
            for v in VERSIONS:
                half[f"noise_{v}"].append((k, replaced(top(s1[v], k), top(s2[v], k))))
            half["N_to_C1_half"].append((k, replaced(top(s1["N"], k), top(s1["C1"], k))))
    for key, values in half.items():
        arr = pd.DataFrame(values, columns=["budget", "value"])
        for k, g in arr.groupby("budget"):
            rows.append(dict(procedure="half_split", quantity=key, budget=k, mean=g.value.mean(),
                             q025=g.value.quantile(.025), q975=g.value.quantile(.975), draws=len(g)))
    # Block bootstrap within pair-cell: selection probabilities on shared resamples.
    count = {(v, k): np.zeros(len(size)) for v in ["N", "C1"] for k in BUDGETS}
    within, between, previous = {k: [] for k in BUDGETS}, {k: [] for k in BUDGETS}, None
    for b in range(a.bootstrap):
        index = np.floor(rng.random((len(size), width)) * size[:, None]).astype(int)
        s = {v: resample_means(m[v], size, index) for v in ["N", "C1"]}
        current = {}
        for k in BUDGETS:
            tn, tc = top(s["N"], k), top(s["C1"], k)
            count[("N", k)] += tn; count[("C1", k)] += tc
            within[k].append(replaced(tn, tc))
            current[k] = tn
            if previous is not None and b % 2 == 1:
                between[k].append(replaced(previous[k], tn))
        previous = current
    for k in BUDGETS:
        for key, values in [("N_to_C1_within_resample", within[k]), ("noise_N_independent_resamples", between[k])]:
            v = np.array(values)
            rows.append(dict(procedure="bootstrap", quantity=key, budget=k, mean=v.mean(), q025=np.quantile(v, .025),
                             q975=np.quantile(v, .975), draws=len(v)))
        rows.append(dict(procedure="fixed_data", quantity="N_to_C1", budget=k, mean=replaced(sel[("N", k)], sel[("C1", k)]),
                         q025=np.nan, q975=np.nan, draws=1))
    table = pd.DataFrame(rows)
    table.to_csv(out / "turnover.csv", index=False)
    p = units.copy()
    for v in ["N", "C1"]:
        p[f"score_{v}"] = full[v]
        p[f"p_select_{v}"] = count[(v, k5)] / a.bootstrap
    p["c1_entrant"], p["c1_exit"] = entrants, exits
    p["selected_N"], p["selected_C1"] = sel[("N", k5)], sel[("C1", k5)]
    p["blocks"] = size
    p[p.c1_entrant | p.c1_exit].to_csv(out / "changed_unit_selection.csv", index=False)
    groups = {"C1 entrants": entrants, "C1 exits": exits, "selected under both": sel[("N", k5)] & sel[("C1", k5)]}
    summary = []
    for name, mask in groups.items():
        for v in ["N", "C1"]:
            q = p.loc[mask, f"p_select_{v}"]
            summary.append(dict(group=name, convention=v, units=int(mask.sum()), mean=q.mean(), median=q.median(),
                                share_below_0_1=(q < .1).mean(), share_at_least_0_5=(q >= .5).mean()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "selection_probability.csv", index=False)
    w = table[(table.quantity == "N_to_C1_within_resample") & (table.budget == k5)].iloc[0]
    ent = summary[summary.group == "C1 entrants"].set_index("convention")["mean"]
    decision = dict(within_resample_q025=float(w.q025), entrant_mean_p_N=float(ent["N"]), entrant_mean_p_C1=float(ent["C1"]),
                    systematic=bool(w.q025 > 0 and ent["N"] < .1 and ent["C1"] >= .5))
    (out / "summary.json").write_text(json.dumps(dict(
        inputs={k: digest(getattr(a, k)) for k in ["cases", "blocks", "verified", "precision", "certificates"]},
        pair_cells=int(len(size)), replicate_counts={int(x): int((size == x).sum()) for x in np.unique(size)},
        halves=a.halves, bootstrap=a.bootstrap, seed=SEED, decision=decision,
        limitations="Combination replicate blocks only; monotherapy, fits and references fixed; shared plates would make noise an underestimate."),
        indent=2))
    with pd.option_context("display.width", 200):
        print(table[table.budget == k5].round(4).to_string(index=False)); print(summary.round(3).to_string(index=False)); print(decision)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="results/neural/runs/scientific_native_full/surface_cases.parquet")
    p.add_argument("--blocks", default="results/neural/runs/scientific_native_full/blocks.parquet")
    p.add_argument("--verified", required=True)
    p.add_argument("--precision", default="results/research_20261107/precision/certificates.csv")
    p.add_argument("--certificates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--halves", type=int, default=200)
    p.add_argument("--bootstrap", type=int, default=1000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
