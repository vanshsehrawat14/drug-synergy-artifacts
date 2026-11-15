"""Post-hoc follow-ups to the preregistered decision analysis (labelled exploratory).

1. Edge-exclusion convention: treat certified roots closer than eps to an asymptote as undefined (keep fallback).
2. Near-tie density: units within the changed-unit margin of the cutoff, versus all units.
3. Attribution of pair-cell score change to boundary versus interior root fixes.
4. Concentration of changed decisions by drug and cell line.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_sensitivity import BUDGETS, KEY, aggregate, compare, references, selected, unit_scores


def run(a):
    out = Path(a.output)
    cases = pd.read_parquet(a.cases)
    blocks = pd.read_parquet(a.blocks)
    for c in KEY:
        blocks[c] = blocks[c].astype(str)
    certs = pd.read_parquet(a.certificates)
    ref, masks = references(cases, pd.read_parquet(a.verified), pd.read_csv(a.precision), certs)
    cert = certs.set_index("case_id").reindex(cases.case_id)
    dist = np.minimum(cert.dist_lower.to_numpy(float), cert.dist_upper.to_numpy(float))
    fix = masks["fallback"] & masks["has_root"]
    pcN, _ = aggregate(blocks, unit_scores(cases, blocks, ref["N"]))
    x = pcN.to_numpy()
    rows = []
    for eps in [0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
        use = fix & (dist >= eps)
        r = np.where(use, ref["I"], ref["N"])
        pc, pair = aggregate(blocks, unit_scores(cases, blocks, r))
        for k in BUDGETS:
            rows.append(dict(edge_exclusion_eps=eps, fixes_kept=int(use.sum()), budget=k,
                             **{m: v for m, v in compare(x, pc.to_numpy(), k).items()
                                if m in ("selected", "replaced", "replacement_fraction", "jaccard")}))
    pd.DataFrame(rows).to_csv(out / "edge_exclusion_sensitivity.csv", index=False)
    # attribution: boundary (<0.01) vs interior fixes
    parts = {}
    for name, m in [("boundary", fix & (dist < .01)), ("interior", fix & (dist >= .01))]:
        r = np.where(m, ref["I"], ref["N"])
        parts[name] = aggregate(blocks, unit_scores(cases, blocks, r))[0].to_numpy() - x
    pcI = aggregate(blocks, unit_scores(cases, blocks, ref["I"]))[0].to_numpy()
    total = pcI - x
    units = pcN.reset_index()[KEY]
    att = []
    for k in BUDGETS:
        A, n = selected(x, k)
        B, _ = selected(pcI, k)
        ch = A ^ B
        moved = ch & (np.abs(total) > 0)
        cut = np.sort(x)[::-1][n - 1]
        margin_ch = np.abs(x[ch] - cut)
        att.append(dict(budget=k, changed=int(ch.sum()), changed_with_own_score_change=int(moved.sum()),
                        changed_displaced_only=int((ch & (total == 0)).sum()),
                        boundary_share_of_abs_change=float(np.abs(parts["boundary"][moved]).sum() /
                                                           max(np.abs(total[moved]).sum(), 1e-300)),
                        changed_margin_median=float(np.median(margin_ch)), changed_margin_p90=float(np.quantile(margin_ch, .9)),
                        units_within_p90_margin=int((np.abs(x - cut) <= np.quantile(margin_ch, .9)).sum()),
                        cutoff_score=float(cut), score_sd_all=float(x.std()),
                        nonzero_change_units=int((np.abs(total) > 0).sum())))
    pd.DataFrame(att).to_csv(out / "change_attribution.csv", index=False)
    # concentration at top-5%
    A, _ = selected(x, .05)
    B, _ = selected(pcI, .05)
    ch = A ^ B
    u = units[ch]
    drugs = pd.concat([u.drug_a, u.drug_b]).value_counts()
    cells = u.cell_line.value_counts()
    base_d = pd.concat([units.drug_a, units.drug_b]).value_counts()
    conc = dict(changed=int(ch.sum()),
                top_drugs=[dict(drug=d, changed_units=int(v), units_total=int(base_d[d])) for d, v in drugs.head(8).items()],
                top_cells=[dict(cell=c, changed_units=int(v)) for c, v in cells.head(8).items()],
                distinct_drugs=int(drugs.size), distinct_cells=int(cells.size),
                share_top3_drugs=float(drugs.head(3).sum() / (2 * ch.sum())))
    (out / "change_concentration.json").write_text(json.dumps(conc, indent=2))
    print(pd.DataFrame(att).to_string(index=False))
    print(json.dumps(conc)[:1500])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="results/neural/runs/scientific_native_full/surface_cases.parquet")
    p.add_argument("--blocks", default="results/neural/runs/scientific_native_full/blocks.parquet")
    p.add_argument("--verified", required=True)
    p.add_argument("--precision", default="results/research_20261107/precision/certificates.csv")
    p.add_argument("--certificates", required=True)
    p.add_argument("--output", required=True)
    run(p.parse_args())


def fallback_value(cases):
    """Native summed-dose fallback max(f_a(xa+xb), f_b(xa+xb)), capped at 100 (SynergyFinder 2.0.x `fun`)."""
    x = cases.dose_a.to_numpy(float) + cases.dose_b.to_numpy(float)
    vals = []
    for arm in "ab":
        b, c, d, e = [cases[f"{arm}_{k}"].to_numpy(float) for k in "bcde"]
        ll4 = cases[f"model_{arm}"].to_numpy() == "LL.4"
        with np.errstate(all="ignore"):
            t = np.where(ll4, np.log(x) - np.log(e), np.log(x) - e)
            vals.append(c + (d - c) / (1 + np.exp(b * t)))
    return np.minimum(np.maximum(vals[0], vals[1]), 100)


def consistent_edge(a):
    """N versus consistent edge conventions: every certified root with distance >= eps is used; others fall back."""
    out = Path(a.output)
    cases = pd.read_parquet(a.cases)
    blocks = pd.read_parquet(a.blocks)
    for c in KEY:
        blocks[c] = blocks[c].astype(str)
    certs = pd.read_parquet(a.certificates)
    ref, masks = references(cases, pd.read_parquet(a.verified), pd.read_csv(a.precision), certs)
    cert = certs.set_index("case_id").reindex(cases.case_id)
    dist = np.minimum(cert.dist_lower.to_numpy(float), cert.dist_upper.to_numpy(float))
    fb = masks["fallback"]
    fbv = fallback_value(cases)
    check = np.abs(fbv[fb] - ref["N"][fb])
    root = np.minimum(cert.root.to_numpy(float), 100)
    pcN = aggregate(blocks, unit_scores(cases, blocks, ref["N"]))[0].to_numpy()
    rows = []
    for eps in [0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]:
        use = masks["has_root"] & (dist >= eps)
        # accepted cases without a usable root keep their native value only if not an excluded root
        r = np.where(use, root, np.where(masks["has_root"], fbv, ref["N"]))
        pc = aggregate(blocks, unit_scores(cases, blocks, r))[0].to_numpy()
        for k in BUDGETS:
            rows.append(dict(eps=eps, accepted_roots_excluded=int((~fb & masks["has_root"] & (dist < eps)).sum()),
                             fallback_roots_used=int((fb & use).sum()), budget=k,
                             **{m: v for m, v in compare(pcN, pc, k).items() if m in ("replaced", "replacement_fraction")}))
    t = pd.DataFrame(rows)
    t.to_csv(out / "consistent_edge_conventions.csv", index=False)
    print("fallback reconstruction max abs diff", float(np.nanmax(check)), "n", int(fb.sum()))
    t["replacement_fraction"] = (t.replacement_fraction * 100).round(2)
    print(t.pivot(index=["eps", "accepted_roots_excluded", "fallback_roots_used"], columns="budget", values="replacement_fraction").to_string())


if __name__ == "__main__" and "--consistent" in __import__("sys").argv:
    pass
