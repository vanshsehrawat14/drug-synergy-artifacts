"""Experiments B and C: few-cell same-pair information curves, controls, residual structure and the matched
pair-seen comparison.

python experiments/interaction_gap_20261110/anchors.py --observations data/processed/oneil_v1/observations.parquet \
    --resolution RESOLUTION_DIR --split results/final_20261110/oracle/pair_cell_split.csv \
    --pairseen manuscript/audit/outputs/pairseen_audit.json --output results/interaction_gap_20261110
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import (FAMILIES, Covariance, Data, Panel, anchor_masks, blup, boot_ratio, drug_leaves, jackknife,  # noqa: E402
                  pick, sha256, sha256_lf, train_baseline)
from variance import CONFIGS  # noqa: E402

KS = [0, 1, 2, 4, 8, 16]
SEEDS = list(range(20))
SEL_SEEDS = list(range(1000, 1005))
EVAL_START = 16
MODELS = ["B0", "B1", "B2", "B4_B1", "B4_B2", "B5a_B1", "B5a_B2", "B5b_B2"]


def donors(data, test, train, salt, seed):
    """Per test pair, a training pair sharing exactly one drug (salt 'B4') or none (salt 'B5a')."""
    out = np.empty(len(test), int)
    tr = np.flatnonzero(train)
    for i, p in enumerate(test):
        dp = {data.drug_a[p], data.drug_b[p]}
        shared = np.array([len(dp & {data.drug_a[q], data.drug_b[q]}) for q in tr])
        cand = tr[shared == (1 if salt == "B4" else 0)]
        out[i] = pick(data.pairs[p], cand, seed, salt)
    return out


def predict_all(cov, cfg, k, t, D, A, E, rows, d4, d5):
    """Predictions of every model and residual-noise draws for one (fold, seed, k)."""
    n = len(rows)
    tE = t[rows[:, None], E]
    out = {"B0": np.zeros_like(tE)}
    noise = {"B0": D[rows[:, None], E]}
    if k == 0:
        for m in MODELS[1:]:
            out[m] = np.zeros_like(tE)
        out["B5b_B2"] = np.full_like(tE, np.nan)
        noise["B1"] = noise["B2"] = noise["B0"]
        return out, noise
    wk = float(cov.b1_weight(k))
    K = cov.signal(*cfg)
    tA = t[rows[:, None], A]
    DA = D[rows[:, None], A]
    out["B1"] = np.repeat(wk * tA.mean(1, keepdims=True), E.shape[1], 1)
    rhs = np.concatenate([tA[..., None], DA], -1)
    sol = blup(K, cov.N, rhs, A, E)
    out["B2"] = sol[..., 0]
    noise["B1"] = noise["B0"] - wk * DA.mean(1, keepdims=True)
    noise["B2"] = noise["B0"] - sol[..., 1:]
    for name, q in [("B4", d4), ("B5a", d5)]:
        tq = t[q[:, None], A]
        out[name + "_B1"] = np.repeat(wk * tq.mean(1, keepdims=True), E.shape[1], 1)
        out[name + "_B2"] = blup(K, cov.N, tq[..., None], A, E)[..., 0]
    if k >= 2:
        out["B5b_B2"] = blup(K, cov.N, tA[..., None], np.roll(A, 1, axis=1), E)[..., 0]
    else:
        out["B5b_B2"] = np.full_like(tE, np.nan)
    return out, noise


def select_config(data, provider, f, k, w):
    v = (f + 1) % 5
    t, D, V = provider({f, v})
    tr, val = ~np.isin(data.fold, [f, v]), np.flatnonzero(data.fold == v)
    cov = Covariance(t[tr], V[tr])
    C = data.shape[1]
    sse = {cfg: 0.0 for cfg in CONFIGS}
    for s in SEL_SEEDS:
        masks = anchor_masks([data.pairs[i] for i in val], C, s)
        A, E = masks[:, :k], masks[:, EVAL_START:]
        tA, tE, wE = t[val[:, None], A], t[val[:, None], E], w[val[:, None], E]
        for cfg in CONFIGS:
            p = blup(cov.signal(*cfg), cov.N, tA[..., None], A, E)[..., 0]
            sse[cfg] += float((wE * (tE - p) ** 2).sum())
    best = min(sse, key=sse.get)
    return best, {f"{lam}:{r}": s_ for (lam, r), s_ in sse.items()}


def run_target(name, data, provider):
    C = data.shape[1]
    w = data.w
    recs, sel = [], {}
    for f in range(5):
        te = np.flatnonzero(data.fold == f)
        tr = data.fold != f
        t, D, V = provider({f})
        cov = Covariance(t[tr], V[tr])
        cfgs = {}
        for k in KS[1:]:
            cfgs[k], grid = select_config(data, provider, f, k, w)
            sel[f"{f}:{k}"] = {"config": list(cfgs[k]), "validation_sse": grid}
        for s in SEEDS:
            masks = anchor_masks([data.pairs[i] for i in te], C, s)
            E = masks[:, EVAL_START:]
            d4, d5 = donors(data, te, tr, "B4", s), donors(data, te, tr, "B5a", s)
            wE = w[te[:, None], E]
            tE = t[te[:, None], E]
            sse0 = (wE * tE ** 2).sum(1)
            for k in KS:
                A = masks[:, :k]
                preds, noise = predict_all(cov, cfgs.get(k, (1.0, 0)), k, t, D, A, E, te, d4, d5)
                n0 = (wE * (noise["B0"] ** 2).mean(-1)).sum(1)
                n1 = (wE * (noise["B1"] ** 2).mean(-1)).sum(1)
                n2 = (wE * (noise["B2"] ** 2).mean(-1)).sum(1)
                for m in MODELS:
                    sse = (wE * (tE - preds[m]) ** 2).sum(1)
                    recs.append(pd.DataFrame({"target": name, "fold": f, "seed": s, "k": k, "model": m,
                                              "pair_id": np.array(data.pairs)[te], "sse": sse, "sse0": sse0, "noise0": n0,
                                              "noise_resid_b1": n1, "noise_resid_b2": n2, "weight": wE.sum(1),
                                              "eval_cells": E.shape[1]}))
        print(name, "fold", f, flush=True)
    df = pd.concat(recs, ignore_index=True)
    for c in ["target", "model", "pair_id"]:
        df[c] = df[c].astype("category")
    return df, sel


def summarize(df, data, boot):
    """Seed-averaged per-pair energies; pooled R2 with pair-bootstrap and drug-jackknife intervals; contrasts."""
    per = df.groupby(["target", "k", "model", "pair_id"], observed=True)[["sse", "sse0", "noise0", "noise_resid_b1", "noise_resid_b2", "weight"]].mean().reset_index()
    pair_index = {p: i for i, p in enumerate(data.pairs)}
    keep = {d: np.array([m_[pair_index[p]] for p in data.pairs]) for d, m_ in drug_leaves(data).items()}
    out = {}
    for (tg, k), g in per.groupby(["target", "k"], observed=True):
        wide = g.pivot(index="pair_id", columns="model", values="sse").reindex(data.pairs)
        s0 = g[g.model == "B0"].set_index("pair_id").reindex(data.pairs)
        T = s0.sse0.to_numpy()
        entry = {"eval_noise_share": float(s0.noise0.sum() / T.sum())}
        for m in MODELS:
            E = wide[m].to_numpy()
            if np.isnan(E).all():
                continue
            r2 = 1 - E.sum() / T.sum()
            leaves = [1 - E[kp].sum() / T[kp].sum() for kp in keep.values()]
            ratio = 1 - E / T
            entry[m] = {"r2": float(r2), "pair_interval": boot_ratio(E, T, boot, 11), "drug_interval": jackknife(r2, leaves),
                        "mse": float(E.sum() / s0.weight.sum()), "rmse": float(np.sqrt(E.sum() / s0.weight.sum())),
                        "pair_macro_r2_mean": float(np.mean(ratio)), "pair_macro_r2_median": float(np.median(ratio))}
        for a_, b_ in [("B1", "B0"), ("B2", "B1"), ("B1", "B4_B1"), ("B1", "B5a_B1"), ("B2", "B4_B2"), ("B2", "B5a_B2"), ("B2", "B5b_B2")]:
            if a_ not in entry or b_ not in entry:
                continue
            Ea, Eb = wide[a_].to_numpy(), wide[b_].to_numpy()
            d = (Eb.sum() - Ea.sum()) / T.sum()
            leaves = [(Eb[kp].sum() - Ea[kp].sum()) / T[kp].sum() for kp in keep.values()]
            entry[f"{a_}-{b_}"] = {"r2_difference": float(d), "pair_interval": boot_ratio(np.column_stack([Eb, Ea]), T, boot, 11,
                                                                                          lambda n, dd: (n[0] - n[1]) / dd),
                                   "drug_interval": jackknife(d, leaves),
                                   "pairs_improved_share": float(np.mean(Eb > Ea))}
        if k > 0:
            E1, E2 = wide["B1"].to_numpy(), wide["B2"].to_numpy()
            n1 = g[g.model == "B1"].set_index("pair_id").reindex(data.pairs).noise_resid_b1.to_numpy()
            entry["experiment_c"] = {"residual_energy_after_B1": float(E1.sum() / T.sum()),
                                     "residual_noise_after_B1": float(n1.sum() / T.sum()),
                                     "residual_r2_B2_vs_B1": float(1 - E2.sum() / E1.sum()),
                                     "residual_r2_pair_interval": boot_ratio(E2, E1, boot, 11),
                                     "reproducible_residual_explained_by_B2": float((E1.sum() - E2.sum()) / (E1.sum() - n1.sum()))}
        out[f"{tg}:k={k}"] = entry
    return out, per


def masked_contrast(data, M, mask):
    """Panel contrast whose drug effects are fitted only on entries where mask is True."""
    dj = {d: j for j, d in enumerate(data.drugs)}
    out = np.empty_like(M)
    for f in range(5):
        idx = np.flatnonzero(data.fold == f)
        h = np.zeros((len(idx), len(dj)))
        h[np.arange(len(idx)), [dj[d] for d in data.drug_a[idx]]] = 1
        h[np.arange(len(idx)), [dj[d] for d in data.drug_b[idx]]] = 1
        for c in range(M.shape[1]):
            m = mask[idx, c]
            sw = np.sqrt(data.w[idx, c])
            coef = np.linalg.lstsq(h[m] * sw[m, None], M[idx[m], c] * sw[m], rcond=None)[0]
            out[idx, c] = M[idx, c] - h @ coef
    return out


def matched(data, t, V, split, panel, pairseen, name, M=None):
    """Pair-cell split anchors (folds not in {f, f+1}); B2 chosen on fold f+1; evaluated on fold f.

    With M (unprojected pair-cell means), anchors, covariance and validation targets use contrasts whose drug
    effects are fitted without test-fold entries (strictly leak-free); evaluation is unchanged.
    """
    P, C = data.shape
    w = data.w
    pred = {"B1": np.zeros_like(t), "B2": np.zeros_like(t)}
    chosen = {}
    t_eval = t
    for f in range(5):
        v = (f + 1) % 5
        trm = ~np.isin(split, [f, v])
        if M is not None:
            t_tr, t_val = masked_contrast(data, M, trm), masked_contrast(data, M, split != f)
        else:
            t_tr = t_val = t_eval
        t = t_tr
        cov = Covariance(t, V, mask=trm)
        sse = {cfg: 0.0 for cfg in CONFIGS}
        Ks = {cfg: cov.signal(*cfg) for cfg in CONFIGS}
        for p in range(P):
            A, E = np.flatnonzero(trm[p]), np.flatnonzero(split[p] == v)
            for cfg in CONFIGS:
                pr = blup(Ks[cfg], cov.N, t[p, A][None, :, None], A[None], E[None])[0, :, 0]
                sse[cfg] += float((w[p, E] * (t_val[p, E] - pr) ** 2).sum())
        cfg = min(sse, key=sse.get)
        chosen[f] = list(cfg)
        K = cov.signal(*cfg)
        for p in range(P):
            A, E = np.flatnonzero(trm[p]), np.flatnonzero(split[p] == f)
            pred["B2"][p, E] = blup(K, cov.N, t[p, A][None, :, None], A[None], E[None])[0, :, 0]
            pred["B1"][p, E] = float(cov.b1_weight(len(A))) * t[p, A].mean()
    res = {"b2_configs": chosen}
    t = t_eval
    T_all = (w * t ** 2)
    for m, pr in pred.items():
        for proj in ["unprojected", "projected"]:
            R = t - pr if proj == "unprojected" else panel(t - pr)
            E_ = w * R ** 2
            per_fold = {f: float(1 - E_[split == f].sum() / T_all[split == f].sum()) for f in range(5)}
            Ep, Tp = E_.sum(1), T_all.sum(1)
            res[f"{m}:{proj}"] = {"r2": float(1 - Ep.sum() / Tp.sum()), "pair_interval": boot_ratio(Ep, Tp, 2000, 11),
                                  "drug_interval": jackknife(1 - Ep.sum() / Tp.sum(),
                                                             [1 - Ep[kp].sum() / Tp[kp].sum() for kp in drug_leaves(data).values()]),
                                  "per_fold": per_fold}
    E1, E2 = (w * panel(t - pred["B1"]) ** 2).sum(1), (w * panel(t - pred["B2"]) ** 2).sum(1)
    res["B2-B1:projected"] = {"r2_difference": float((E1.sum() - E2.sum()) / T_all.sum()),
                              "pair_interval": boot_ratio(np.column_stack([E1, E2]), T_all.sum(1), 2000, 11, lambda n, dd: (n[0] - n[1]) / dd)}
    key = "pair_cell:bliss_residual" if name == "bliss" else "pair_cell:raw"
    ps = pairseen["results"][key]
    b3 = {"r2": ps["pair_seen_r2"], "f2_base_r2": ps["f2_base_r2"]}
    if "per_pair_cell_split_fold" in ps:
        b3["per_fold"] = {int(k): v["pair_seen_r2"] for k, v in ps["per_pair_cell_split_fold"].items()}
        for m in ["B1", "B2"]:
            diffs = [b3["per_fold"][f] - res[f"{m}:projected"]["per_fold"][f] for f in range(5)]
            res[f"B3-{m}:projected_per_fold"] = diffs
    b3["pair_interval"] = ps.get("pair_seen_pair_interval")
    res["B3"] = b3
    res["B3-B1"] = b3["r2"] - res["B1:projected"]["r2"]
    res["B3-B2"] = b3["r2"] - res["B2:projected"]["r2"]
    res["B1_share_of_B3"] = res["B1:projected"]["r2"] / b3["r2"]
    res["oof_predictions"] = pred
    return res


def main():
    ap = argparse.ArgumentParser()
    for k in ["observations", "resolution", "split", "pairseen", "output"]:
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--targets", default="bliss,raw,ref,train,f2_residual")
    a = ap.parse_args()
    t0 = time.time()
    out_dir = Path(a.output)
    data = Data(a.observations, a.resolution)
    P, C = data.shape
    panel = Panel(data)
    Mt = data.pc_mean_units(data.truth_u)
    Mref = data.pc_mean_units(data.cache["bliss"])
    signs = data.split_signs(50, 97, "grid")
    D = data.split_differences(signs)
    PD = panel(D)
    V = (PD ** 2).mean(-1)
    T = {"bliss": panel(Mt - Mref), "raw": panel(Mt), "ref": panel(Mref)}
    f2 = np.mean([panel(data.pc_mean_units(data.cache[nm]) - Mref) for nm in FAMILIES["f2_selected"]], 0)
    T["f2_residual"] = T["bliss"] - f2
    zeros = np.zeros_like(PD)

    def fixed(name):
        Dn = zeros if name == "ref" else PD
        Vn = (Dn ** 2).mean(-1)
        return lambda exclude: (T[name], Dn, Vn)

    def train(exclude):
        r = train_baseline(data, np.concatenate([(Mt - Mref)[..., None], D], -1), exclude)
        return r[..., 0], r[..., 1:], (r[..., 1:] ** 2).mean(-1)

    frames, selection = [], {}
    for name in a.targets.split(","):
        df, sel = run_target(name, data, train if name == "train" else fixed(name))
        frames.append(df)
        selection[name] = sel
        print(name, "done", round(time.time() - t0), flush=True)
    df = pd.concat(frames, ignore_index=True)
    summary, per = summarize(df, data, a.boot)
    print("summary", round(time.time() - t0), flush=True)

    # Curve table: (target, k, seed, fold, model) on identical evaluation rows.
    g = df.groupby(["target", "k", "seed", "fold", "model"], observed=True)
    curve = g.agg(n_pairs=("pair_id", "size"), eval_pair_cells=("eval_cells", "sum"), eval_weight=("weight", "sum"),
                  sse=("sse", "sum"), sse_b0=("sse0", "sum"), noise_energy=("noise0", "sum")).reset_index()
    ratio = df.assign(r=1 - df.sse / df.sse0).groupby(["target", "k", "seed", "fold", "model"], observed=True).r
    curve["pair_macro_r2_mean"] = ratio.mean().to_numpy()
    curve["pair_macro_r2_median"] = ratio.median().to_numpy()
    curve["mse"] = curve.sse / curve.eval_weight
    curve["rmse"] = np.sqrt(curve.mse)
    curve["r2_vs_b0"] = 1 - curve.sse / curve.sse_b0
    curve["eval_noise_share"] = curve.noise_energy / curve.sse_b0
    curve = curve.drop(columns="noise_energy")
    curve.to_csv(out_dir / "ANCHOR_INFORMATION_CURVE.csv", index=False, float_format="%.6g", lineterminator="\n")
    per.to_csv(out_dir / "anchor_pairs.csv.gz", index=False, float_format="%.6g", compression="gzip")

    # Matched pair-seen comparison.
    sp = pd.read_csv(a.split).astype({"pair_id": str, "cell_line": str})
    pi, ci = {p: i for i, p in enumerate(data.pairs)}, {c: i for i, c in enumerate(data.cells)}
    split = np.full((P, C), -1)
    split[sp.pair_id.map(pi).to_numpy(), sp.cell_line.map(ci).to_numpy()] = sp.fold.to_numpy()
    if (split < 0).any():
        raise ValueError("Pair-cell split incomplete")
    pairseen = json.loads(Path(a.pairseen).read_text())
    mres = {}
    Mfull = {"bliss": Mt - Mref, "raw": Mt}
    for name in ["bliss", "raw"]:
        r = matched(data, T[name], V, split, panel, pairseen, name)
        r.pop("oof_predictions")
        mres[name] = r
        r = matched(data, T[name], V, split, panel, pairseen, name, M=Mfull[name])
        r.pop("oof_predictions")
        mres[name + ":leak_free"] = r
    print("matched", round(time.time() - t0), flush=True)

    result = {"inputs": {"observations_sha256": sha256(a.observations), "cache_sha256": sha256(Path(a.resolution) / "cache.npz"),
                         "split_sha256_lf": sha256_lf(a.split), "pairseen_audit_sha256_lf": sha256_lf(a.pairseen)},
              "settings": {"k": KS, "seeds": SEEDS, "selection_seeds": SEL_SEEDS, "eval_start": EVAL_START,
                           "configs": [list(c) for c in sorted(CONFIGS)], "bootstrap": a.boot},
              "curves": summary, "selection": selection, "matched_pair_seen": mres,
              "runtime_seconds": round(time.time() - t0, 1)}
    (out_dir / "anchor_experiment.json").write_text(json.dumps(result, indent=1))
    print("done", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
