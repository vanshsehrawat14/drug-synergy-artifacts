"""Prospective model-predicted Loewe scoring (response_analysis 2026-11-07, preregistered section 5).

Observed combination responses are replaced by fixed out-of-fold neural F2 predictions; only the
per-case Loewe reference changes between versions. Case arm a is the column drug (conc_c) and arm b the
row drug (conc_r); this mapping reproduces all 92,208 native block scores to <1e-13.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_sensitivity import BUDGETS, KEY, compare, references, selected
from .io import read_json
from .neural_report import RUNS, index, load_truth, model_paths, out_of_fold


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_parquet(a.cases)
    blocks = pd.read_parquet(a.blocks)
    for c in KEY:
        blocks[c] = blocks[c].astype(str)
    certs = pd.read_parquet(a.certificates)
    ref, _ = references(cases, pd.read_parquet(a.verified), pd.read_csv(a.precision), certs)
    df, sha, _ = load_truth("data/processed/oneil_v1")
    t = index(df)
    conc = pd.read_parquet("data/processed/oneil_v1/observations.parquet", columns=["observation_id", "conc_c", "conc_r"])
    obs = df[["observation_id", "block_id"]].copy()
    obs["row"] = np.arange(len(obs))
    obs = obs.merge(conc, on="observation_id", how="left", validate="one_to_one")
    obs = obs.merge(blocks[["block_id", "surface_id"]], on="block_id", validate="many_to_one")
    pos = cases[["surface_id", "dose_a", "dose_b"]].reset_index().rename(columns={"index": "case_row", "dose_a": "conc_c", "dose_b": "conc_r"})
    obs = obs.merge(pos, on=["surface_id", "conc_c", "conc_r"], how="left", validate="many_to_one")
    if obs.case_row.isna().any():
        raise ValueError("Unmapped observation")
    obs = obs.sort_values("row")
    case_row = obs.case_row.to_numpy(int)
    block_of = obs.block_id.to_numpy()
    # check: observed responses reproduce native block scores
    y = t["y"]
    bs = pd.Series(y - ref["N"][case_row]).groupby(block_of).mean()
    if np.abs(bs - blocks.set_index("block_id").native_sum_score.reindex(bs.index)).max() > 1e-9:
        raise AssertionError("Mapping does not reproduce native block scores")
    chosen = read_json(RUNS / "benchmark_v1/selection.json")["selected"]
    preds = {f"neural_F2_seed{s}": out_of_fold(model_paths("selected:F2", s, chosen, RUNS / "benchmark_v1"), t) for s in [2026, 2027, 2028]}
    preds["neural_F2_seed_mean"] = np.mean([preds[k] for k in list(preds)], axis=0)
    preds["observed"] = y
    bkey = blocks.set_index("block_id")[KEY]
    rows = []
    scores = {}
    for name, p in preds.items():
        for v in ["N", "C1", "I"]:
            blk = pd.Series(p - ref[v][case_row]).groupby(block_of).mean()
            f = bkey.loc[blk.index].assign(score=blk.to_numpy())
            scores[(name, v)] = f.groupby(KEY, sort=True, observed=True).score.mean().to_numpy()
        for base, alt in [("N", "C1"), ("N", "I")]:
            for k in BUDGETS:
                r = compare(scores[(name, base)], scores[(name, alt)], k)
                rows.append(dict(responses=name, kind="observed" if name == "observed" else "prospective_model_predicted",
                                 base=base, alternative=alt, budget=k, replaced=r["replaced"], selected=r["selected"],
                                 replacement_fraction=r["replacement_fraction"], jaccard=r["jaccard"]))
        for k in BUDGETS:  # overlap of model-predicted and observed native selections (context only)
            A, n = selected(scores[(name, "N")], k)
            B, _ = selected(scores[("observed", "N")] if ("observed", "N") in scores else scores[(name, "N")], k)
    table = pd.DataFrame(rows)
    ov = []
    for name in preds:
        if name == "observed":
            continue
        for k in BUDGETS:
            A, n = selected(scores[(name, "N")], k)
            B, _ = selected(scores[("observed", "N")], k)
            ov.append(dict(responses=name, budget=k, predicted_vs_observed_native_overlap=(A & B).sum() / n))
    table = table.merge(pd.DataFrame(ov), on=["responses", "budget"], how="left")
    table.to_csv(out / "prospective_scoring.csv", index=False)
    x = table.copy()
    x["replacement_fraction"] = (x.replacement_fraction * 100).round(2)
    print(x.pivot_table(index=["responses", "base", "alternative"], columns="budget", values="replacement_fraction").to_string())
    print(pd.DataFrame(ov).pivot_table(index="responses", columns="budget", values="predicted_vs_observed_native_overlap").round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="results/neural/runs/scientific_native_full/surface_cases.parquet")
    p.add_argument("--blocks", default="results/neural/runs/scientific_native_full/blocks.parquet")
    p.add_argument("--verified", required=True)
    p.add_argument("--precision", default="results/interaction_analysis/precision/certificates.csv")
    p.add_argument("--certificates", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())
