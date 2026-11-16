"""Replicate-noise ceilings, cross-cell pair oracles and the pair-seen comparison (configs/pair_information.json).

python -m synergy_artifacts.pair_analysis noise --resolution RES --output OUT
python -m synergy_artifacts.pair_analysis oracle --resolution RES --output OUT
python -m synergy_artifacts.pair_analysis pairseen --resolution RES --runs RUNS --oracle OUT_ORACLE --output OUT

All quantities are evaluation contrasts of fixed predictions. Oracles read the same pair in other
cells and are information diagnostics, never held-out-pair predictors.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .ablation_summary import unit_group
from .io import read_json, write_json, digest
from .pair_seen import SEEDS, assign, assignment_sha256, pair_cell_folds
from .resolution import means, projections, residualize
from .neural_report import index, load_truth, jackknife_interval

SEED = 20261110
MODELS = {"neural_F2": "selected F2", "neural_F0": "selected F0", "neural_base_F2": "production F2 (f2_base)",
          "mono_only": "monotherapy-only network", "gp_F2": "GP F2", "cf_selected": "CF extension", "bliss": "Bliss"}


def load(resolution):
    root = Path(resolution)
    meta = read_json(root / "inputs.json")
    frame = pd.read_parquet(root / "coordinates.parquet")
    df, sha, _ = load_truth("data/processed/oneil_v1")
    if sha != meta["dataset_sha256"]:
        raise ValueError("Dataset differs from the resolution cache")
    t = index(df)
    group = unit_group(df, t)
    n = np.bincount(group)
    with np.load(root / "cache.npz") as c:
        values = c["values"]
    yg = means(group, t["y"], n)
    if not np.allclose(yg, values[:, 0], atol=1e-9) or not np.array_equal(n, frame.n.to_numpy().astype(int)):
        raise ValueError("Unit grouping does not reproduce the cache")
    cols = [j for j, f in enumerate(meta["families"]) if f in MODELS]
    z = np.column_stack([values[:, 0]] + [values[:, 0] - values[:, j] for j in cols])
    names = ["truth"] + [meta["names"][j] for j in cols]
    fams = np.array(["truth"] + [meta["families"][j] for j in cols])
    return dict(meta=meta, frame=frame, df=df, t=t, group=group, n=n, z=z, names=names, fams=fams, sha=sha)


def noise_draws(d, draws, seed=SEED):
    """Balanced random block halves within each pair-cell; columns have the replicate-mean noise variance."""
    df, t, group, n = d["df"], d["t"], d["group"], d["n"]
    blocks = pd.DataFrame({"panel": t["panel"], "block": df.block_id.to_numpy()}).drop_duplicates().sort_values(["panel", "block"])
    sizes = blocks.groupby("panel").size().to_numpy()
    if not set(sizes) <= {4, 8}:
        raise ValueError("Unexpected replicate block counts")
    rng = np.random.default_rng(seed)
    position = blocks.groupby("panel").cumcount().to_numpy()
    row_block = pd.Series(np.arange(len(blocks)), index=blocks.block.to_numpy()).reindex(df.block_id.to_numpy()).to_numpy()
    out = np.empty((len(n), draws))
    panel_of_block = np.repeat(np.arange(len(sizes)), sizes)
    for r in range(draws):
        sign = np.empty(len(blocks))
        for size in (4, 8):
            panels = np.flatnonzero(sizes == size)
            perm = rng.permuted(np.tile(np.arange(size), (len(panels), 1)), axis=1)
            half = np.where(perm < size // 2, 1., -1.)
            rows = np.flatnonzero(np.isin(panel_of_block, panels))
            sign[rows] = half[np.searchsorted(panels, panel_of_block[rows]), position[rows]]
        out[:, r] = means(group, sign[row_block] * t["y"], n)
    return out


def pair_energy(frame, x):
    return np.column_stack([np.bincount(frame.pair.to_numpy(), weights=frame.n.to_numpy(float) * c * c,
                                        minlength=frame.pair.max() + 1) for c in x.T])


def ratio_interval(num, den, pair_drugs, leave_num, leave_den, transform, draws=2000, seed=SEED):
    """transform(numerator_sums, denominator_sums) on pair-level sufficient statistics; whole-pair bootstrap and drug jackknife."""
    as2 = lambda x: np.asarray(x, float).reshape(len(x), -1)
    num, den, leave_num, leave_den = as2(num), as2(den), as2(leave_num), as2(leave_den)
    value = transform(num.sum(0), den.sum(0))
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(num), size=(draws, len(num)))
    boot = np.array([transform(num[i].sum(0), den[i].sum(0)) for i in ix])
    leave = np.array([transform(a, b) for a, b in zip(leave_num, leave_den)])
    low, high = jackknife_interval(value, leave)
    return dict(value=float(value), pair_low=float(np.quantile(boot, .025)), pair_high=float(np.quantile(boot, .975)),
                drug_low=float(low), drug_high=float(high))


def noise(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    d = load(a.resolution)
    frame, z, fams, meta = d["frame"], d["z"], d["fams"], d["meta"]
    noise_units = noise_draws(d, a.draws)
    w = frame.n.to_numpy(float)
    # Pooled within-unit replicate variance: the row-level ceiling.
    rep = np.bincount(d["group"], weights=(d["t"]["y"] - z[:, 0][d["group"]]) ** 2)
    sigma2 = rep.sum() / (w - 1).sum()
    columns = np.column_stack([z, noise_units])
    k = z.shape[1]
    with threadpool_limits(limits=a.threads):
        ep, ew, rp, rw = projections(frame, columns)
        pe = {"panel": pair_energy(frame, rp), "dose": pair_energy(frame, rw)}
        del rp, rw
        leaves = {"panel": [], "dose": []}
        for drug in meta["drugs"]:
            lp, lw, _, _ = projections(frame, columns, drug, False)
            leaves["panel"].append(lp); leaves["dose"].append(lw)
            print("deleted", drug, flush=True)
    leaves = {key: np.array(v) for key, v in leaves.items()}
    ib = int(np.flatnonzero(fams == "bliss")[0])
    rows, pd_ = [], meta["pair_drugs"]
    for level in ["panel", "dose"]:
        e, lv = pe[level], leaves[level]
        noise_e, noise_l = e[:, k:].mean(1), lv[:, k:].mean(1)
        for target, den_col in [("raw", 0), ("bliss_residual", ib)]:
            share = ratio_interval(noise_e, e[:, den_col], pd_, noise_l, lv[:, den_col], lambda x, y: x[0] / y[0])
            rows.append(dict(level=level, target=target, model="noise_share", **share,
                             noise_energy=float(noise_e.sum() / w.sum()), target_energy=float(e[:, den_col].sum() / w.sum()),
                             noise_draw_sd=float(e[:, k:].sum(0).std() / w.sum())))
            for family in MODELS:
                ix = np.flatnonzero(fams == family)
                if target == "bliss_residual" and family == "bliss":
                    continue
                me, ml = e[:, ix].mean(1), lv[:, ix].mean(1)
                num = np.column_stack([me, noise_e]); den = e[:, den_col]
                lnum = np.column_stack([ml, noise_l])
                raw = ratio_interval(num, den, pd_, lnum, lv[:, den_col], lambda x, y: 1 - x[0] / y[0])
                corrected = ratio_interval(num, den, pd_, lnum, lv[:, den_col], lambda x, y: 1 - (x[0] - x[1]) / (y[0] - x[1]))
                rows.append(dict(level=level, target=target, model=family, **raw,
                                 **{f"corrected_{key}": v for key, v in corrected.items()}))
    table = pd.DataFrame(rows)
    table.to_csv(out / "noise_ceilings.csv", index=False)
    row_var = float(d["t"]["y"].var())
    write_json(out / "noise_summary.json", dict(
        dataset_sha256=d["sha"], resolution_inputs_sha256=digest(Path(a.resolution) / "inputs.json"), draws=a.draws, seed=SEED,
        replicate_variance=float(sigma2), within_unit_df=float((w - 1).sum()), row_variance=row_var,
        row_ceiling_r2=float(1 - sigma2 / row_var), names=d["names"],
        estimator="Equal block halves within each pair-cell; d = (mean half 1 - mean half 2)/2 per identical-input unit.",
        limitations="Combination replicate blocks only; shared plates or processing would make noise an underestimate. Monotherapy is shared across blocks."))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))


def panel_contrast(frame, values, drop=None):
    """(P - A) on pair-cell means within cell and original fold; the panel part of resolution.projections."""
    keep = np.ones(len(frame), bool) if drop is None else ((frame.drug_a != drop) & (frame.drug_b != drop)).to_numpy()
    result = np.zeros_like(values)
    for _, g in frame[keep].groupby(["fold", "cell_line"], observed=True):
        ii = g.index.to_numpy()
        a, b = g.drug_a.to_numpy(), g.drug_b.to_numpy()
        lookup = {x: j for j, x in enumerate(sorted(set(a) | set(b)))}
        da, db = np.array([lookup[x] for x in a]), np.array([lookup[x] for x in b])
        panel, inv = np.unique(g.panel.to_numpy(), return_inverse=True)
        w = g.n.to_numpy(float); pn = np.bincount(inv, weights=w)
        pm = means(inv, values[ii] * w[:, None], pn)
        first = np.unique(inv, return_index=True)[1]
        h = np.zeros((len(panel), len(lookup)))
        np.add.at(h, (np.arange(len(panel)), da[first]), 1)
        np.add.at(h, (np.arange(len(panel)), db[first]), 1)
        result[ii] = residualize(h, pm, pn)[inv]
    return result


def oracle_predictions(frame, t_unit, split, drop=None):
    """Unit-level oracle values from the same pair's pair-cell contrasts in other cells."""
    panel = frame.panel.to_numpy()
    first = np.unique(panel, return_index=True)[1]
    keep = np.ones(len(frame), bool) if drop is None else ((frame.drug_a != drop) & (frame.drug_b != drop)).to_numpy()
    p = frame.iloc[first][["pair_id", "cell_line"]].reset_index(drop=True)
    p["t"] = t_unit[first]
    p["keep"] = keep[first]
    p = p.merge(split, on=["pair_id", "cell_line"], how="left")
    if p.fold.isna().any():
        raise ValueError("Pair-cell without split fold")
    loco = np.zeros(len(p)); matched = np.zeros(len(p))
    for _, g in p[p.keep].groupby("pair_id"):
        ii, tv, fv = g.index.to_numpy(), g.t.to_numpy(), g.fold.to_numpy(int)
        loco[ii] = (tv.sum() - tv) / (len(tv) - 1)
        for f in range(5):
            train = (fv != f) & (fv != (f + 1) % 5)
            matched[ii[fv == f]] = tv[train].mean()
    inverse = np.unique(panel, return_inverse=True)[1]
    return loco[inverse], matched[inverse]


def oracle(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    d = load(a.resolution)
    frame, z, fams, meta = d["frame"], d["z"], d["fams"], d["meta"]
    split = pair_cell_folds(d["t"]["panel_frame"])
    ib = int(np.flatnonzero(fams == "bliss")[0])
    base = np.flatnonzero(fams == "neural_base_F2"); sel = np.flatnonzero(fams == "neural_F2")
    w = frame.n.to_numpy(float)

    def evaluate(drop=None):
        targets = np.column_stack([z[:, 0], z[:, ib]])  # raw truth and Bliss residual
        rp = panel_contrast(frame, targets, drop)
        cols, labels = [], []
        for j, name in enumerate(["raw", "bliss_residual"]):
            loco, matched = oracle_predictions(frame, rp[:, j], split, drop)
            cols += [targets[:, j] - loco, targets[:, j] - matched]
            labels += [f"{name}:loco", f"{name}:matched"]
        model_cols = np.column_stack([z[:, base], z[:, sel], z[:, ib]])
        allcols = np.column_stack([targets, np.column_stack(cols), model_cols])
        rpa = panel_contrast(frame, allcols, drop)
        return rp, pair_energy(frame, rpa), labels

    with threadpool_limits(limits=a.threads):
        rp, energy, labels = evaluate()
        leaves = []
        for drug in meta["drugs"]:
            leaves.append(evaluate(drug)[1].sum(0)); print("deleted", drug, flush=True)
    leaves = np.array(leaves)
    nb, ns = len(base), len(sel)
    col = dict(raw=0, bliss_residual=1)
    o0 = 2
    m0 = o0 + len(labels)
    rows = []
    for target in ["raw", "bliss_residual"]:
        den = energy[:, col[target]]; lden = leaves[:, col[target]]
        # Every column of the second projection is an error relative to the same target space.
        candidates = {f"oracle_{lab.split(':')[1]}": o0 + labels.index(lab) for lab in labels if lab.startswith(target + ":")}
        model_ix = {"neural_base_F2": list(range(m0, m0 + nb)), "neural_F2": list(range(m0 + nb, m0 + nb + ns))}
        if target == "raw":
            model_ix["bliss"] = [m0 + nb + ns]
        for name, ix in list(candidates.items()) + list(model_ix.items()):
            ix = np.atleast_1d(ix)
            num, lnum = energy[:, ix].mean(1), leaves[:, ix].mean(1)
            # For raw targets model errors are y - prediction; for residual targets the Bliss offset cancels.
            r = ratio_interval(num, den, meta["pair_drugs"], lnum, lden, lambda x, y: 1 - x[0] / y[0])
            rows.append(dict(target=target, predictor=name, **r))
        for oname in candidates:
            ix = candidates[oname]
            num = np.column_stack([energy[:, model_ix["neural_base_F2"]].mean(1), energy[:, ix]])
            lnum = np.column_stack([leaves[:, model_ix["neural_base_F2"]].mean(1), leaves[:, ix]])
            r = ratio_interval(num, den, meta["pair_drugs"], lnum, lden, lambda x, y: (x[0] - x[1]) / y[0])
            rows.append(dict(target=target, predictor=f"{oname}_minus_f2_base", **r))
    table = pd.DataFrame(rows)
    # Cross-cell covariance share on the full-data contrasts.
    panel = frame.panel.to_numpy(); first = np.unique(panel, return_index=True)[1]
    p = frame.iloc[first][["pair_id"]].reset_index(drop=True)
    shares = {}
    for j, target in enumerate(["raw", "bliss_residual"]):
        p["t"] = rp[first, j]
        g = p.groupby("pair_id").t
        s1, s2, c = g.sum(), g.apply(lambda x: (x ** 2).sum()), g.size()
        shares[target] = float(((s1 ** 2 - s2) / (c - 1)).sum() / s2.sum())
    gate_row = table[(table.target == "bliss_residual") & (table.predictor == "oracle_matched_minus_f2_base")].iloc[0]
    gate = bool(gate_row.value >= 0.10 and gate_row.pair_low > 0)
    table.to_csv(out / "oracle_r2.csv", index=False)
    split.to_csv(out / "pair_cell_split.csv", index=False)
    write_json(out / "oracle_summary.json", dict(
        dataset_sha256=d["sha"], split_sha256=assignment_sha256(split), cross_cell_covariance_share=shares,
        gate=dict(rule="matched oracle minus f2_base interaction-residual R^2 >= 0.10 and pair-bootstrap lower bound > 0",
                  value=float(gate_row.value), pair_low=float(gate_row.pair_low), passed=gate),
        note="Oracles average the same pair's pair-cell contrast over other cells (all, or the B2 training folds)."))
    print(table.round(4).to_string(index=False)); print(shares, "gate passed:", gate)


def pair_seen_predictions(runs, t, seed):
    """Out-of-fold pair-seen predictions in runner row order; every row exactly once."""
    f = pd.concat([pd.read_parquet(Path(runs) / "f2_pair_seen" / f"seed_{seed}" / f"fold_{k}" / "predictions.parquet",
                                   columns=["observation_id", "prediction"]) for k in range(5)], ignore_index=True)
    ids = t["df"].observation_id.to_numpy()
    if f.observation_id.duplicated().any() or set(f.observation_id) != set(ids):
        raise ValueError(f"Seed {seed}: pair-seen predictions do not cover every row once")
    p = f.set_index("observation_id").prediction.reindex(ids).to_numpy(float)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite pair-seen predictions")
    return p


def hsa_units(d):
    """Training-pair HSA reference from the historical f2_base seed-2026 feature states, as in information_audit."""
    from .features import FeatureBuilder
    df, hsa = d["df"], np.zeros(len(d["df"]))
    for fold in range(5):
        state = Path("results/neural/runs/benchmark_v1/runs/f2_base/seed_2026") / f"fold_{fold}" / "features.json"
        m = df.fold.to_numpy() == fold
        hsa[m] = FeatureBuilder.from_state_dict(read_json(state)).reference(df[m].drop(columns=["inhibition"]), "hsa")
    return means(d["group"], hsa, d["n"])


def pairseen(a):
    """Pair-seen F2 versus held-out-pair f2_base on identical rows and the original evaluation operators (B2)."""
    from .information_audit import remove_reference_span
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    d = load(a.resolution)
    frame, z, fams, meta, t = d["frame"], d["z"], d["fams"], d["meta"], d["t"]
    base, ib = np.flatnonzero(fams == "neural_base_F2"), int(np.flatnonzero(fams == "bliss")[0])
    split = pd.read_csv(Path(a.oracle) / "pair_cell_split.csv")
    split_sha = assignment_sha256(split)
    if split_sha != read_json(Path(a.oracle) / "oracle_summary.json")["split_sha256"]:
        raise ValueError("Split file differs from the oracle summary")
    seen, rowerr_seen = [], []
    for seed in SEEDS:
        for k in range(5):
            if read_json(Path(a.runs) / "f2_pair_seen" / f"seed_{seed}" / f"fold_{k}" / "metadata.json")["split_assignment_sha256"] != split_sha:
                raise ValueError("Run split hash differs")
        p = pair_seen_predictions(a.runs, t, seed)
        seen.append(z[:, 0] - means(d["group"], p, d["n"]))
        rowerr_seen.append(np.bincount(t["pair"], weights=(t["y"] - p) ** 2))
    rowerr_seen = np.column_stack(rowerr_seen)
    with np.load(Path(a.resolution) / "cache.npz") as c:
        rowerr_base = c["row_pair_error"][:, [j for j, f in enumerate(meta["families"]) if f == "neural_base_F2"]]
    hsa = hsa_units(d)
    w = frame.n.to_numpy(float)
    # Columns: 0 truth; 1 Bliss error (y - r_B); 2 HSA error; 3-5 f2_base errors; 6-8 pair-seen errors; 9 matched oracle error.
    fixed = np.column_stack([z[:, 0], z[:, ib], z[:, 0] - hsa, z[:, base], np.column_stack(seen)])

    def evaluate(drop=None):
        _, matched = oracle_predictions(frame, panel_contrast(frame, z[:, ib:ib + 1], drop)[:, 0], split, drop)
        columns = np.column_stack([fixed, z[:, ib] - matched])
        with threadpool_limits(limits=a.threads):
            _, _, rp, rw = projections(frame, columns, drop)
            ap = remove_reference_span(frame, rp, [1, 2], drop)
        return {"panel": rp, "dose": rw, "panel_reference_augmented": ap}

    full = evaluate()
    pe = {k: pair_energy(frame, x) for k, x in full.items()}
    del full
    leaves = {k: [] for k in pe}
    for drug in meta["drugs"]:
        for k, x in evaluate(drug).items():
            leaves[k].append((w[:, None] * x * x).sum(0))
        print("deleted", drug, flush=True)
    leaves = {k: np.array(v) for k, v in leaves.items()}
    noise_table = pd.read_csv(Path(a.noise) / "noise_ceilings.csv")
    noise_total = {lv: float(noise_table[(noise_table.level == lv) & (noise_table.model == "noise_share")].noise_energy.iloc[0] * w.sum())
                   for lv in ["panel", "dose"]}
    pf = frame.groupby("pair", observed=True).fold.first().to_numpy()
    groups = {"f2_base": [3, 4, 5], "pair_seen": [6, 7, 8], "oracle_matched": [9]}
    rows = []
    for level in ["panel", "dose", "panel_reference_augmented"]:
        e, lv = pe[level], leaves[level]
        targets = [("reference_augmented", 0)] if level == "panel_reference_augmented" else [("raw", 0), ("bliss_residual", 1)]
        for target, den in targets:
            models = groups if (level != "dose" and target != "raw") else {k: v for k, v in groups.items() if k != "oracle_matched"}
            for name, cols in models.items():
                r = ratio_interval(e[:, cols].mean(1), e[:, den], meta["pair_drugs"], lv[:, cols].mean(1), lv[:, den], lambda x, y: 1 - x[0] / y[0])
                rows.append(dict(level=level, target=target, model=name, quantity="r2", **r))
            num = np.column_stack([e[:, groups["f2_base"]].mean(1), e[:, groups["pair_seen"]].mean(1)])
            lnum = np.column_stack([lv[:, groups["f2_base"]].mean(1), lv[:, groups["pair_seen"]].mean(1)])
            r = ratio_interval(num, e[:, den], meta["pair_drugs"], lnum, lv[:, den], lambda x, y: (x[0] - x[1]) / y[0])
            folds = [float((num[pf == f, 0].sum() - num[pf == f, 1].sum()) / e[pf == f, den].sum()) for f in range(5)]
            noise = noise_total.get(level) if level != "panel_reference_augmented" else None
            corrected = float((num[:, 0].sum() - num[:, 1].sum()) / (e[:, den].sum() - noise)) if noise else np.nan
            rows.append(dict(level=level, target=target, model="pair_seen_minus_f2_base", quantity="delta_r2", **r,
                             fold_min=min(folds), fold_max=max(folds), folds_positive=int(sum(x > 0 for x in folds)),
                             noise_corrected_value=corrected))
    pdg = np.array(meta["pair_drugs"])
    leave_seen = [rowerr_seen[(pdg != dr).all(1)].mean(1).sum() for dr in meta["drugs"]]
    leave_base = [rowerr_base[(pdg != dr).all(1)].mean(1).sum() for dr in meta["drugs"]]
    resp = ratio_interval(rowerr_seen.mean(1), rowerr_base.mean(1), meta["pair_drugs"], leave_seen, leave_base, lambda x, y: x[0] / y[0] - 1)
    rows.append(dict(level="row", target="response", model="pair_seen_vs_f2_base", quantity="relative_mse_change", **resp,
                     pair_seen_mse=float(rowerr_seen.sum(0).mean() / len(t["y"])), f2_base_mse=float(rowerr_base.sum(0).mean() / len(t["y"]))))
    table = pd.DataFrame(rows)
    table.to_csv(out / "pair_seen_comparison.csv", index=False)
    per_seed = []
    for j, s in enumerate(SEEDS):
        for level in ["panel", "dose"]:
            e = pe[level]
            per_seed.append(dict(seed=s, level=level, **{f"{m}_{tg}": float(1 - e[:, groups[m][j]].sum() / e[:, den].sum())
                                                         for m in ["f2_base", "pair_seen"] for tg, den in [("raw", 0), ("bliss_residual", 1)]}))
    pd.DataFrame(per_seed).to_csv(out / "per_seed.csv", index=False)
    pick = lambda lv, tg, m: table[(table.level == lv) & (table.target == tg) & (table.model == m)].iloc[0]
    prim, orc, held = pick("panel", "bliss_residual", "pair_seen_minus_f2_base"), pick("panel", "bliss_residual", "oracle_matched"), pick("panel", "bliss_residual", "f2_base")
    exploitable = bool(prim.value >= 0.05 and prim.pair_low > 0 and prim.drug_low > 0 and prim.folds_positive >= 4)
    write_json(out / "decision.json", dict(
        primary_delta_r2=float(prim.value), pair_interval=[float(prim.pair_low), float(prim.pair_high)],
        drug_interval=[float(prim.drug_low), float(prim.drug_high)], folds_positive=int(prim.folds_positive),
        noise_corrected_delta_r2=float(prim.noise_corrected_value), oracle_fraction_captured=float(prim.value / (orc.value - held.value)),
        exploitable=exploitable, representation_limited=bool(not exploitable),
        rule="Exploitable: change >= 0.05, both grouped intervals above zero, at least 4/5 original folds positive.",
        split_sha256=split_sha, seeds=SEEDS,
        note="Reference-augmented rows were added after B1 was observed and are exploratory secondaries."))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(table.round(4).to_string(index=False))
    print("exploitable:", exploitable)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["noise", "oracle"]:
        s = sub.add_parser(name)
        s.add_argument("--resolution", required=True); s.add_argument("--output", required=True)
        s.add_argument("--threads", type=int, default=4)
        if name == "noise":
            s.add_argument("--draws", type=int, default=50)
    s = sub.add_parser("pairseen")
    for arg in ["--resolution", "--runs", "--oracle", "--noise", "--output"]:
        s.add_argument(arg, required=True)
    s.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    {"noise": noise, "oracle": oracle, "pairseen": pairseen}[a.command](a)


if __name__ == "__main__":
    main()
