"""Screening-decision sensitivity of fixed-fit Loewe scoring (response_analysis 2026-11-07, preregistered).

Retrospective observed-response scoring only. All versions share cases, fits, inclusion and units;
only per-case references change. Undefined/no-root cases keep the native convention value.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
KEY = ["pair_id", "drug_a", "drug_b", "cell_line"]
SEED = 20261107


def order(score):
    """Descending score; ties by canonical (input) order."""
    return np.argsort(-np.asarray(score, float), kind="stable")


def ranks(score):
    r = np.empty(len(score), int)
    r[order(score)] = np.arange(len(score))
    return r


def selected(score, k):
    n = max(1, int(np.ceil(len(score) * k)))
    m = np.zeros(len(score), bool)
    m[order(score)[:n]] = True
    return m, n


def compare(a, b, k, groups=None):
    A, n = selected(a, k)
    B, _ = selected(b, k)
    ra, rb = ranks(a), ranks(b)
    move = np.abs(ra - rb)
    union = A | B
    cut_a = np.sort(np.asarray(a))[::-1][n - 1]
    changed = A ^ B
    out = dict(n_units=len(a), budget=k, selected=n,
               replaced=int((A & ~B).sum()), replacement_fraction=(A & ~B).sum() / n,
               symmetric_difference=int(changed.sum()), jaccard=(A & B).sum() / union.sum(),
               entered=int((B & ~A).sum()), exited=int((A & ~B).sum()),
               ties_at_cutoff=int((np.asarray(a) == cut_a).sum() - 1),
               rank_move_median_all=float(np.median(move)), rank_move_p95_all=float(np.quantile(move, .95)),
               rank_move_median_union=float(np.median(move[union])), rank_move_p95_union=float(np.quantile(move[union], .95)),
               sign_flips=int((np.sign(a) != np.sign(b)).sum()))
    if changed.any():
        margin = np.abs(np.asarray(a)[changed] - cut_a)
        delta = np.abs(np.asarray(b)[changed] - np.asarray(a)[changed])
        out.update(changed_margin_median=float(np.median(margin)), changed_margin_max=float(margin.max()),
                   changed_score_change_median=float(np.median(delta)),
                   changed_margin_below_change=float(np.mean(margin < delta)),
                   changed_rank_distance_median=float(np.median(np.abs(ra[changed] - (n - .5)))))
    return out


def global_rank(a, b):
    return dict(spearman=float(spearmanr(a, b).statistic), kendall_tau_b=float(kendalltau(a, b).statistic))


def references(cases, verified, precision, certs):
    native = cases.native_below_3.to_numpy(float)
    fb = cases.fallback.to_numpy()
    ver = verified.set_index("case_id").reindex(cases.case_id)
    c1 = fb & ver.admissible.to_numpy(bool)
    if c1.sum() != 14197:
        raise AssertionError("C1 scope must be 14,197")
    ref = {"N": native.copy()}
    ref["C1"] = np.where(c1, np.minimum(ver.independent.to_numpy(float), 100), native)
    p = precision[(precision.condition == "unchanged") & precision.certified & ~precision.control]
    pos = pd.Series(np.arange(len(cases)), index=cases.case_id)
    idx = pos.loc[p.case_id].to_numpy()
    rec = idx[~c1[idx]]
    if len(rec) != 1423:
        raise AssertionError("C2 recovered scope must be 1,423")
    ref["C2"] = ref["C1"].copy()
    ref["C2"][rec] = np.minimum(p.set_index("case_id").loc[cases.case_id.iloc[rec], "effect"].to_numpy(float), 100)
    cert = certs.set_index("case_id").reindex(cases.case_id)
    root = cert.root.to_numpy(float)
    has_root = (cert.cert_status == "certified_root").to_numpy()
    ref["I"] = np.where(fb & has_root, np.minimum(root, 100), native)
    ref["R"] = np.where(has_root, np.minimum(root, 100), native)
    ref["H"] = cases.native_hsa_fallback.to_numpy(float)
    masks = dict(c1=c1, c2_recovered=np.isin(np.arange(len(cases)), rec), has_root=has_root, fallback=fb)
    return ref, masks


def unit_scores(cases, blocks, ref):
    """Block score = native block score - mean case reference shift on its surface; pair-cell = mean of blocks."""
    shift = pd.Series(ref - cases.native_below_3.to_numpy(float), index=cases.surface_id).groupby(level=0).mean()
    block = blocks.native_sum_score.to_numpy(float) - blocks.surface_id.map(shift).to_numpy(float)
    return block


def aggregate(blocks, block_score):
    f = blocks[KEY].copy()
    f["score"] = block_score
    pc = f.groupby(KEY, sort=True, observed=True).score.mean()
    pair = pc.groupby(level="pair_id", sort=True).mean()
    return pc, pair


def interval_bounds(cases, blocks, certs, ref, masks):
    """Unit score intervals for version I: certified brackets, unresolved cases over {native, solver value}."""
    cert = certs.set_index("case_id").reindex(cases.case_id)
    lo = ref.copy()
    hi = ref.copy()
    fb = masks["fallback"]
    use = fb & masks["has_root"]
    blo = pd.to_numeric(cert.bracket_lo, errors="coerce").to_numpy(float)
    bhi = pd.to_numeric(cert.bracket_hi, errors="coerce").to_numpy(float)
    lo[use] = np.minimum(np.nextafter(blo[use], -np.inf), 100)
    hi[use] = np.minimum(np.nextafter(bhi[use], np.inf), 100)
    unresolved = (cert.cert_status == "unresolved").to_numpy()
    alt = np.where(np.isfinite(cases.native_solution.to_numpy(float)), np.minimum(cases.native_solution.to_numpy(float), 100), ref)
    lo[unresolved] = np.minimum(ref[unresolved], alt[unresolved])
    hi[unresolved] = np.maximum(ref[unresolved], alt[unresolved])
    # higher reference -> lower score
    s_hi = unit_scores(cases, blocks, lo)
    s_lo = unit_scores(cases, blocks, hi)
    return s_lo, s_hi, int(unresolved.sum())


def guaranteed(native_score, lo, hi, k):
    """Membership certainly-in / certainly-out of the corrected top-k given score intervals."""
    n = max(1, int(np.ceil(len(lo) * k)))
    A, _ = selected(native_score, k)
    # j certainly in if fewer than n others could exceed its lower bound
    hs = np.sort(hi)
    def earlier_ties(values, thresholds):
        positions = {}
        for i, value in enumerate(values):
            positions.setdefault(value, []).append(i)
        return np.array([np.searchsorted(positions.get(value, []), i) for i, value in enumerate(thresholds)])

    could_beat = len(hi) - np.searchsorted(hs, lo, side="right") - (hi > lo) + earlier_ties(hi, lo)
    certain_in = could_beat < n
    ls = np.sort(lo)
    surely_beat = len(lo) - np.searchsorted(ls, hi, side="right") + earlier_ties(lo, hi)
    certain_out = surely_beat >= n
    ambiguous = ~certain_in & ~certain_out
    return dict(guaranteed_exits=int((A & certain_out).sum()), guaranteed_entries=int((~A & certain_in).sum()),
                ambiguous_native_selected=int((A & ambiguous).sum()), ambiguous_native_unselected=int((~A & ambiguous).sum()))


def _replaced(a, b, k):
    A, n = selected(a, k)
    B, _ = selected(b, k)
    return (A & ~B).sum() / n


def pair_bootstrap(units, a, b, k, draws=2000):
    groups = [g for g in units.groupby("pair_id", sort=True).indices.values()]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(draws):
        ix = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        vals.append(_replaced(a[ix], b[ix], k))
    drugs = sorted(set(units.drug_a) | set(units.drug_b))
    loo = [_replaced(a[m], b[m], k) for d in drugs
           for m in [((units.drug_a != d) & (units.drug_b != d)).to_numpy()]]
    return dict(pair_ci_lo=float(np.quantile(vals, .025)), pair_ci_hi=float(np.quantile(vals, .975)),
                drug_loo_min=float(min(loo)), drug_loo_max=float(max(loo)))


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_parquet(a.cases)
    blocks = pd.read_parquet(a.blocks)
    for c in KEY:
        blocks[c] = blocks[c].astype(str)
    verified = pd.read_parquet(a.verified)
    precision = pd.read_csv(a.precision)
    certs = pd.read_parquet(a.certificates)
    ref, masks = references(cases, verified, precision, certs)
    block_scores = {v: unit_scores(cases, blocks, r) for v, r in ref.items()}
    block_scores["H"] = blocks.native_hsa_fallback_score.to_numpy(float)
    block_scores["P"] = blocks.synergy_loewe.to_numpy(float)
    if not np.allclose(block_scores["N"], blocks.native_sum_score.to_numpy(float)):
        raise AssertionError("Native block scores not reproduced")
    pc, pair = {}, {}
    for v, s in block_scores.items():
        pc[v], pair[v] = aggregate(blocks, s)
    units = pc["N"].reset_index()[KEY]
    comparisons = [("N", "C1"), ("N", "C2"), ("C1", "C2"), ("N", "I"), ("N", "R"), ("I", "R"), ("N", "H"), ("P", "N")]
    rows = []
    blk_key = blocks.block_id.to_numpy()
    blk_order = np.argsort(blk_key, kind="stable")
    for base, alt in comparisons:
        for level, x, y in [("pair_cell", pc[base].to_numpy(), pc[alt].to_numpy()),
                            ("drug_pair", pair[base].to_numpy(), pair[alt].to_numpy()),
                            ("block", block_scores[base][blk_order], block_scores[alt][blk_order])]:
            g = global_rank(x, y)
            for k in BUDGETS:
                rows.append(dict(level=level, base=base, alternative=alt,
                                 kind="convention" if "H" in (base, alt) else "reconstruction" if "P" in (base, alt) else "numerical",
                                 **compare(x, y, k), **g))
    table = pd.DataFrame(rows)
    # pair-clustered uncertainty and within-cell turnover for the numerical pair-cell comparisons
    extra = []
    for base, alt in [("N", "C1"), ("N", "C2"), ("N", "I"), ("N", "R")]:
        x, y = pc[base].to_numpy(), pc[alt].to_numpy()
        for k in BUDGETS:
            boot = pair_bootstrap(units, x, y, k, a.draws)
            cells = [_replaced(x[m], y[m], k) for _, m in units.groupby("cell_line").indices.items()]
            extra.append(dict(level="pair_cell", base=base, alternative=alt, budget=k, **boot,
                              within_cell_median=float(np.median(cells)), within_cell_min=float(min(cells)),
                              within_cell_max=float(max(cells))))
    table = table.merge(pd.DataFrame(extra), on=["level", "base", "alternative", "budget"], how="left")
    # guaranteed versus ambiguous changes for N -> I
    lo, hi, n_unres = interval_bounds(cases, blocks, certs, ref["I"], masks)
    lo_pc, _ = aggregate(blocks, lo)
    hi_pc, _ = aggregate(blocks, hi)
    gb = [dict(level="pair_cell", base="N", alternative="I_interval", budget=k, unresolved_cases=n_unres,
               max_unit_interval_width=float((hi_pc - lo_pc).max()),
               **guaranteed(pc["N"].to_numpy(), lo_pc.to_numpy(), hi_pc.to_numpy(), k)) for k in BUDGETS]
    pd.DataFrame(gb).to_csv(out / "guaranteed_changes.csv", index=False)
    table.to_csv(out / "ranking_stability.csv", index=False)
    # unit-level table with row IDs
    u = units.copy()
    for v in ["N", "C1", "C2", "I", "R", "H", "P"]:
        u[f"score_{v}"] = pc[v].to_numpy()
        u[f"rank_{v}"] = ranks(pc[v].to_numpy())
        for k in BUDGETS:
            u[f"top{int(k * 100)}_{v}"] = selected(pc[v].to_numpy(), k)[0]
    # case composition per pair-cell
    cert = certs.set_index("case_id").reindex(cases.case_id)
    near = (cert.dist_lower.to_numpy(float) < .01) | (cert.dist_upper.to_numpy(float) < .01)
    comp = pd.DataFrame({"surface_id": cases.surface_id.to_numpy(),
                         "c1": masks["c1"], "c2rec": masks["c2_recovered"],
                         "i_fix": masks["fallback"] & masks["has_root"],
                         "boundary_fix": masks["fallback"] & masks["has_root"] & near,
                         "abs_shift_I": np.abs(ref["I"] - ref["N"])}).groupby("surface_id").agg(
        c1=("c1", "sum"), c2rec=("c2rec", "sum"), i_fix=("i_fix", "sum"), boundary_fix=("boundary_fix", "sum"),
        abs_shift_I=("abs_shift_I", "mean"))
    bs = blocks[KEY + ["surface_id"]].drop_duplicates().merge(comp, left_on="surface_id", right_index=True, how="left")
    agg = bs.groupby(KEY, sort=True).agg(surfaces=("surface_id", "nunique"), c1_cases=("c1", "sum"),
                                         c2_recovered_cases=("c2rec", "sum"), i_fix_cases=("i_fix", "sum"),
                                         boundary_fix_cases=("boundary_fix", "sum")).reset_index()
    u = u.merge(agg, on=KEY, how="left", validate="one_to_one")
    u.to_csv(out / "ranking_stability_units.csv", index=False)
    # conditional effects by unit composition
    cond = []
    for base, alt in [("N", "C1"), ("N", "I"), ("C1", "C2")]:
        for k in BUDGETS:
            A, B = u[f"top{int(k * 100)}_{base}"].to_numpy(), u[f"top{int(k * 100)}_{alt}"].to_numpy()
            ch = A ^ B
            cls = np.select([u.boundary_fix_cases > 0, u.i_fix_cases > 0], ["contains_boundary_fix", "unique_root_fix_only"], "no_fix_displaced")
            for c in ["contains_boundary_fix", "unique_root_fix_only", "no_fix_displaced"]:
                cond.append(dict(base=base, alternative=alt, budget=k, unit_class=c,
                                 units=int((cls == c).sum()), changed=int((ch & (cls == c)).sum())))
    pd.DataFrame(cond).to_csv(out / "conditional_changes.csv", index=False)
    # reproduce the reported top-5% N->C1 turnover from row IDs
    t5 = u[u.top5_N ^ u.top5_C1][KEY + ["score_N", "score_C1", "rank_N", "rank_C1", "top5_N", "top5_C1", "c1_cases", "boundary_fix_cases"]]
    t5.to_csv(out / "top5_N_to_C1_changes.csv", index=False)
    summary = dict(top5_N_C1_exited=int((u.top5_N & ~u.top5_C1).sum()), top5_N_C1_selected=int(u.top5_N.sum()),
                   reproduces_5_89=bool(round((u.top5_N & ~u.top5_C1).sum() / u.top5_N.sum(), 6) == round(67 / 1137, 6)),
                   cases=len(cases), blocks=len(blocks), pair_cells=len(u), drug_pairs=int(u.pair_id.nunique()),
                   version_case_changes={v: int((ref[v] != ref["N"]).sum()) for v in ["C1", "C2", "I", "R", "H"]})
    (out / "decision_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="results/neural/runs/scientific_native_full/surface_cases.parquet")
    p.add_argument("--blocks", default="results/neural/runs/scientific_native_full/blocks.parquet")
    p.add_argument("--verified", required=True)
    p.add_argument("--precision", default="results/interaction_analysis/precision/certificates.csv")
    p.add_argument("--certificates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--draws", type=int, default=2000)
    run(p.parse_args())
