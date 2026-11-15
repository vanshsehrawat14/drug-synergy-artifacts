"""Within-screen normalized and baseline-relative errors from saved out-of-fold predictions (overnight 2026-11-07).

No refitting. Training variance uses each test fold's training rows only (folds other than test f and
validation (f+1)%5). Screens are never pooled. Uncertainty: unordered-drug-pair cluster bootstrap and
delete-one-drug ranges. Stratum standardization uses input-defined strata only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [2026]  # primary: identical seed across screens; overridden by --seeds
BLISS_BINS = [-np.inf, 0, 20, 40, 60, 80, np.inf]
DOSE_BINS = [-np.inf, -2, -1, 0, 1, np.inf]
SEED = 20261107


def screens(oneil_root, other_root):
    o, x = Path(oneil_root), Path(other_root)
    spec = {"ONEIL": dict(data=o / "data/processed/oneil_v1", f2=o / "results/neural/runs/sprint_v1/runs/f2_combined",
                          f0=o / "results/neural/runs/sprint_v1/runs/f0_combined",
                          mono_only=o / "results/neural/runs/sprint_v1/runs/mono_only",
                          mean=o / "results/neural/runs/phase2_mean", additive=o / "results/neural/runs/phase2_additive",
                          bliss=(o / "results/neural/runs/sprint_v1/runs/f2_residual", "reference_offset"))}
    for name, run, pre in [("ALMANAC", "almanac_v1", "almanac"), ("NCATS_ES", "ncats_es_v2", "ncats_es_v2"),
                           ("NCATS_DIPG", "ncats_dipg_v2", "ncats_dipg_v2")]:
        r = x / "results/neural/runs"
        spec[name] = dict(data=x / "data/processed" / run, f2=r / run / "runs/f2_combined", f0=r / run / "runs/f0_combined",
                          mono_only=(r / run / "runs/mono_only") if "ncats" in run else None,
                          mean=r / f"{pre}_mean", additive=r / f"{pre}_additive", bliss=r / f"{pre}_reference_bliss")
    return spec


def load_folds(d, column="prediction", seeded=False):
    parts = []
    for fold in range(5):
        if seeded:
            for s in SEEDS:
                p = pd.read_parquet(d / f"seed_{s}/fold_{fold}/predictions.parquet", columns=["observation_id", column])
                parts.append(p.assign(seed=s))
        else:
            p = pd.read_parquet(d / f"fold_{fold}/predictions.parquet", columns=["observation_id", column])
            parts.append(p)
    f = pd.concat(parts, ignore_index=True)
    if seeded:
        f = f.pivot(index="observation_id", columns="seed", values=column)
        f.columns = [f"s{c}" for c in f.columns]
        return f
    if f.observation_id.duplicated().any():
        raise ValueError(f"duplicate predictions in {d}")
    return f.set_index("observation_id")[column]


def build(name, sp):
    obs = pd.read_parquet(sp["data"] / "observations.parquet",
                          columns=["observation_id", "pair_id", "drug_a", "drug_b", "cell_line", "fold", "dose_a", "dose_b",
                                   "inhibition", "is_combination"])
    obs = obs[obs.is_combination].set_index("observation_id")
    tv = {}
    for f in range(5):
        train = obs[(obs.fold != f) & (obs.fold != (f + 1) % 5)].inhibition
        tv[f] = dict(var=float(train.var(ddof=0)), mean=float(train.mean()), n=int(len(train)))
    models = {}
    for m in ["f2", "f0", "mono_only"]:
        if sp.get(m) is not None and Path(sp[m]).exists():
            try:
                models[m] = load_folds(Path(sp[m]), seeded=True)
            except FileNotFoundError as exc:
                print(f"{name} {m}: incomplete seeds, reported NA ({exc.filename})", flush=True)
    for m in ["mean", "additive"]:
        models[m] = load_folds(Path(sp[m]))
    bl = sp["bliss"]
    if isinstance(bl, tuple):
        b = load_folds(bl[0], bl[1], seeded=True)  # reference offset is seed-invariant (checked)
        if not np.allclose(b.min(1), b.max(1), equal_nan=True):
            raise ValueError("reference offset differs across seeds")
        models["bliss"] = b.iloc[:, 0]
        mono = pd.read_parquet(bl[0] / "seed_2026/fold_0/predictions.parquet", columns=["observation_id", "monotherapy_status"])
        monos = [pd.read_parquet(bl[0] / f"seed_2026/fold_{k}/predictions.parquet", columns=["observation_id", "monotherapy_status"]) for k in range(5)]
    else:
        models["bliss"] = load_folds(Path(bl))
        monos = [pd.read_parquet(Path(bl) / f"fold_{k}/predictions.parquet", columns=["observation_id", "monotherapy_status"]) for k in range(5)]
    mono = pd.concat(monos).set_index("observation_id").monotherapy_status.astype(str)
    ids = obs.index
    coverage = {}
    common = ids
    for m, v in models.items():
        idx = v.dropna().index if isinstance(v, pd.Series) else v.dropna(how="any").index
        coverage[m] = int(len(idx.intersection(ids)))
        common = common.intersection(idx)
    t = obs.loc[common].copy()
    t["mono"] = mono.reindex(common).to_numpy()
    err = {}
    for m, v in models.items():
        if isinstance(v, pd.Series):
            err[m] = (t.inhibition - v.loc[common]).to_numpy()[:, None]
        else:
            err[m] = t.inhibition.to_numpy()[:, None] - v.loc[common].to_numpy()
    t["bliss_ref"] = models["bliss"].loc[common].to_numpy()
    return t, err, tv, dict(screen=name, combination_rows=int(len(obs)), evaluable_rows=int(len(t)),
                            dropped_rows=int(len(obs) - len(t)), coverage=coverage)


def mse(e):
    return float(np.mean(e ** 2))  # seed-averaged squared error for seeded models


def per_pair(t, e):
    codes, uniq = pd.factorize(t.pair_id, sort=True)
    return codes, uniq, np.bincount(codes, weights=np.mean(e ** 2, axis=1), minlength=len(uniq)), np.bincount(codes, minlength=len(uniq))


def gain_ci(t, ea, eb, draws):
    codes, uniq, sa, n = per_pair(t, ea)
    _, _, sb, _ = per_pair(t, eb)
    rng = np.random.default_rng(SEED)
    g = []
    for _ in range(draws):
        w = np.bincount(rng.integers(len(uniq), size=len(uniq)), minlength=len(uniq))
        g.append(1 - (w @ sa) / (w @ sb))
    pairs = pd.Series(uniq)
    drugs = sorted(set(t.drug_a) | set(t.drug_b))
    da = t.groupby("pair_id").drug_a.first().reindex(uniq).to_numpy()
    db = t.groupby("pair_id").drug_b.first().reindex(uniq).to_numpy()
    loo = [1 - sa[(da != d) & (db != d)].sum() / sb[(da != d) & (db != d)].sum() for d in drugs]
    return float(np.quantile(g, .025)), float(np.quantile(g, .975)), float(min(loo)), float(max(loo))


def strata(t):
    lb = pd.cut(t.bliss_ref, BLISS_BINS, right=False, labels=False)
    with np.errstate(divide="ignore"):
        ld = pd.cut(np.log10(t.dose_a + t.dose_b), DOSE_BINS, right=False, labels=False)
    return lb.astype(int).astype(str) + "|" + ld.astype(int).astype(str) + "|" + t.mono.astype(str)


def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, cover, strat = [], [], {}
    for name, sp in screens(a.oneil_root, a.other_root).items():
        t, err, tv, cov = build(name, sp)
        cover.append({**cov, **{f"train_var_fold{f}": v["var"] for f, v in tv.items()}})
        fold = t.fold.to_numpy()
        var_row = np.array([tv[f]["var"] for f in fold])
        mean_row = np.array([tv[f]["mean"] for f in fold])
        const = (t.inhibition.to_numpy() - mean_row)[:, None]
        err["train_constant"] = const
        for m, e in err.items():
            se = np.mean(e ** 2, axis=1)
            pc = pd.DataFrame({"k": t.pair_id.astype(str) + "|" + t.cell_line.astype(str), "y": t.inhibition.to_numpy(),
                               "p": t.inhibition.to_numpy()[:, None].repeat(e.shape[1], 1).mean(1) - e.mean(1)})
            g = pc.groupby("k", observed=True)[["y", "p"]].mean()
            row = dict(screen=name, model=m, seeds=e.shape[1], rows=len(t), pairs=int(t.pair_id.nunique()),
                       cells=int(t.cell_line.nunique()), rmse=float(np.sqrt(se.mean())),
                       mae=float(np.mean(np.abs(e))), mse=float(se.mean()),
                       normalized_mse_train_var=float(np.mean(se / var_row)),
                       pair_cell_rmse=float(np.sqrt(((g.y - g.p) ** 2).mean())), pair_cells=len(g))
            for base in ["train_constant", "mean", "bliss", "additive"]:
                if base != m:
                    row[f"gain_vs_{base}"] = 1 - se.mean() / np.mean(err[base] ** 2)
            if m in ("f2", "f0") and "bliss" in err:
                lo, hi, dl, dh = gain_ci(t, e, err["bliss"], a.draws)
                row.update(gain_vs_bliss_pair_ci_lo=lo, gain_vs_bliss_pair_ci_hi=hi, gain_vs_bliss_drug_loo_min=dl,
                           gain_vs_bliss_drug_loo_max=dh)
                lo, hi, dl, dh = gain_ci(t, e, err["train_constant"], a.draws)
                row.update(gain_vs_train_constant_pair_ci_lo=lo, gain_vs_train_constant_pair_ci_hi=hi)
            rows.append(row)
        if "f2" not in err:
            continue
        s = strata(t)
        f2 = np.mean(err["f2"] ** 2, axis=1)
        bl = np.mean(err["bliss"] ** 2, axis=1)
        strat[name] = pd.DataFrame({"s": s, "f2": f2, "bliss": bl, "pair": t.pair_id.to_numpy()}).groupby("s", observed=True).agg(
            n=("f2", "size"), f2=("f2", "sum"), bliss=("bliss", "sum"), pairs=("pair", "nunique"))
        print(name, cov, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "cross_screen_errors.csv", index=False)
    pd.DataFrame(cover).to_csv(out / "data_coverage_predictions.csv", index=False)
    # standardization of the F2-vs-Bliss gain to O'Neil stratum weights on common support
    if "ONEIL" not in strat:
        print("ONEIL F2 unavailable for these seeds; standardization skipped")
        return
    ref = strat["ONEIL"]
    srows = []
    for name, st in strat.items():
        common = ref.index.intersection(st.index)
        w = ref.loc[common, "n"] / ref.loc[common, "n"].sum()
        raw = 1 - st.f2.sum() / st.bliss.sum()
        cs = st.loc[common]
        support = cs.n.sum() / st.n.sum()
        std = 1 - (w * cs.f2 / cs.n).sum() / (w * cs.bliss / cs.n).sum() if len(common) else np.nan
        ess = cs.n.sum() ** 2 / (cs.n ** 2).sum() if len(common) else 0
        srows.append(dict(screen=name, strata=len(st), common_strata=len(common), common_support_rows_share=float(support),
                          oneil_mass_covered=float(ref.loc[common, "n"].sum() / ref.n.sum()),
                          raw_gain_f2_vs_bliss=float(raw), standardized_gain_to_oneil_strata=float(std),
                          stratum_row_ess=float(ess)))
    st = pd.DataFrame(srows)
    o = st.set_index("screen").loc["ONEIL"]
    st["raw_gap_vs_oneil"] = st.raw_gain_f2_vs_bliss - o.raw_gain_f2_vs_bliss
    st["standardized_gap_vs_oneil"] = st.standardized_gain_to_oneil_strata - o.standardized_gain_to_oneil_strata
    st.to_csv(out / "cross_screen_standardization.csv", index=False)
    pd.concat({k: v for k, v in strat.items()}).to_csv(out / "cross_screen_strata.csv")
    print(table[["screen", "model", "rows", "rmse", "mae", "normalized_mse_train_var", "gain_vs_bliss", "gain_vs_train_constant"]].to_string(index=False))
    print(st.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oneil-root", required=True)
    p.add_argument("--other-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seeds", default="2026")
    a = p.parse_args()
    SEEDS[:] = [int(x) for x in a.seeds.split(",")]
    run(a)
