"""Paired, pair-clustered comparison of the neural pair-seen model (B3) with the leak-free B1 and B2.

Inputs: per-pair B3 energies by pair-cell-split fold (b3_pair_energies.py, run where the predictions are), and
the local leak-free matched predictors (anchors.matched with refitted contrasts). The target energies computed on
both sides must agree before any comparison is reported.

python experiments/interaction_structure/b3_paired.py --observations data/processed/oneil_v1/observations.parquet \
    --resolution RESOLUTION_DIR --split results/pair_information/oracle/pair_cell_split.csv \
    --pairseen manuscript/audit/outputs/pairseen_audit.json --b3 results/interaction_structure/b3_pair_energies.json \
    --output results/interaction_structure/b3_paired.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchors import matched  # noqa: E402
from core import FAMILIES, Data, Panel, drug_leaves, jackknife, sha256_lf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "resolution", "split", "pairseen", "b3", "output"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    data = Data(a.observations, a.resolution)
    P, C = data.shape
    panel = Panel(data)
    Mt, Mref = data.pc_mean_units(data.truth_u), data.pc_mean_units(data.cache["bliss"])
    T = panel(Mt - Mref)
    V = (panel(data.split_differences(data.split_signs(50, 97, "grid"))) ** 2).mean(-1)
    sp = pd.read_csv(a.split).astype({"pair_id": str, "cell_line": str})
    pi, ci = {p: i for i, p in enumerate(data.pairs)}, {c: i for i, c in enumerate(data.cells)}
    split = np.full((P, C), -1)
    split[sp.pair_id.map(pi).to_numpy(), sp.cell_line.map(ci).to_numpy()] = sp.fold.to_numpy()
    b3 = json.loads(Path(a.b3).read_text())
    if b3["pairs"] != data.pairs:
        raise ValueError("Pair order differs")
    by_fold = lambda E: np.array([[E[p, split[p] == f].sum() for f in range(5)] for p in range(P)])
    Tw = data.w * T ** 2
    T_pf = by_fold(Tw)
    t_diff = float(np.abs(T_pf - np.array(b3["target_by_pair_fold"])).max())
    if t_diff > 1e-6 * T_pf.max():
        raise ValueError(f"Target energies differ between sites: {t_diff}")
    pairseen = json.loads(Path(a.pairseen).read_text())
    res = matched(data, T, V, split, panel, pairseen, "bliss", M=Mt - Mref)
    oof = res.pop("oof_predictions")
    E = {"B1": by_fold(data.w * panel(T - oof["B1"]) ** 2), "B2": by_fold(data.w * panel(T - oof["B2"]) ** 2),
         "B3": np.array(b3["b3_by_pair_fold"]),
         "F2_held_out_pair": by_fold(np.mean([data.w * (T - panel(data.pc_mean_units(data.cache[nm]) - Mref)) ** 2
                                              for nm in FAMILIES["f2_base"]], 0))}
    tot = T_pf.sum(1)
    leaves = drug_leaves(data)
    rng = np.random.default_rng(11)
    ix = rng.integers(P, size=(a.boot, P))
    out = {"inputs": {"b3_energies_sha256_lf": sha256_lf(a.b3), "split_sha256_lf": sha256_lf(a.split)},
           "target_energy_max_abs_difference_between_sites": t_diff,
           "b3_prediction_files_matching_recorded_sha256": f"{b3['prediction_files_match_recorded_sha256']}/{b3['prediction_files']}",
           "b3_pooled_r2_cluster": b3["b3_pooled_r2"], "b3_seed_pooled_r2_cluster": b3["b3_seed_pooled_r2"],
           "b2_configs": res["b2_configs"], "models": {}, "differences": {}}
    for m, e in E.items():
        ep = e.sum(1)
        r2 = 1 - ep.sum() / tot.sum()
        out["models"][m] = {"r2": float(r2),
                            "pair_interval": [float(np.quantile(1 - ep[ix].sum(1) / tot[ix].sum(1), q)) for q in (.025, .975)],
                            "drug_interval": jackknife(r2, [1 - ep[k].sum() / tot[k].sum() for k in leaves.values()]),
                            "per_fold": [float(1 - e[:, f].sum() / T_pf[:, f].sum()) for f in range(5)]}
    for x, y in [("B3", "B1"), ("B3", "B2"), ("B2", "B1"), ("B3", "F2_held_out_pair"), ("B1", "F2_held_out_pair")]:
        ex, ey = E[x].sum(1), E[y].sum(1)
        d = (ey.sum() - ex.sum()) / tot.sum()
        bs = (ey[ix].sum(1) - ex[ix].sum(1)) / tot[ix].sum(1)
        out["differences"][f"{x}-{y}"] = {
            "r2_difference": float(d), "pair_interval": [float(np.quantile(bs, q)) for q in (.025, .975)],
            "pair_bootstrap_share_positive": float((bs > 0).mean()),
            "drug_interval": jackknife(d, [(ey[k].sum() - ex[k].sum()) / tot[k].sum() for k in leaves.values()]),
            "per_fold": [float((E[y][:, f].sum() - E[x][:, f].sum()) / T_pf[:, f].sum()) for f in range(5)],
            "pairs_where_first_is_better": float(np.mean(ex < ey))}
    out["b1_share_of_b3"] = out["models"]["B1"]["r2"] / out["models"]["B3"]["r2"]
    out["b1_share_of_b3_pair_interval"] = [float(np.quantile((1 - E["B1"].sum(1)[ix].sum(1) / tot[ix].sum(1)) /
                                                            (1 - E["B3"].sum(1)[ix].sum(1) / tot[ix].sum(1)), q)) for q in (.025, .975)]
    Path(a.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ["target_energy_max_abs_difference_between_sites", "b1_share_of_b3"]}))
    print(json.dumps(out["differences"], indent=1)[:2500])


if __name__ == "__main__":
    main()
