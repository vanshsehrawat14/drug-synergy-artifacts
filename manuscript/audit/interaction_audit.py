"""Independent recomputation of replicate-noise ceilings, noise-corrected R^2 and cross-cell oracles.

No project module is imported. Units, the two evaluation contrasts, the block split and the oracles are
reimplemented from their definitions. Inputs: the O'Neil observation table, the frozen resolution cache
of out-of-fold unit means (research_20261107/resolution) and the pair-cell split.

python manuscript/audit/interaction_audit.py --observations data/processed/oneil_v1/observations.parquet \
    --resolution RESOLUTION_DIR --split results/final_20261110/oracle/pair_cell_split.csv \
    --output manuscript/audit/outputs/interaction_audit.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import t as student


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def basis(x, tol=1e-10):
    """Orthonormal basis of the column space of x (SVD)."""
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, s > tol * max(s.max(), 1.0)]


class Operators:
    """Pair-cell contrast C_P and dose-shape contrast C_D within (original fold, cell line).

    Units are identical physical inputs (pair-cell, dose pair) with row counts n_u. C_P removes additive
    endpoint-drug effects from row-weighted pair-cell means; C_D removes pair-cell means and additive
    endpoint drug-by-dose-rank effects from unit means. Both are orthogonal projections in the row-count
    metric, so energies are sums of n_u r_u^2.
    """

    def __init__(self, units, keep):
        self.rows = np.flatnonzero(keep)
        u = units.iloc[self.rows]
        pcs, self.u_pc = np.unique(u.pc.to_numpy(), return_inverse=True)
        self.w_u = u.n.to_numpy(float)
        self.w_pc = np.bincount(self.u_pc, weights=self.w_u)
        first = np.unique(self.u_pc, return_index=True)[1]
        self.cells = u.iloc[first][["pc", "pair", "pair_id", "fold", "cell_line", "drug_a", "drug_b"]].reset_index(drop=True)
        nu, npc, npair = len(u), len(pcs), int(units.pair.max()) + 1
        self.agg = sparse.csr_matrix((self.w_u / self.w_pc[self.u_pc], (self.u_pc, np.arange(nu))), shape=(npc, nu))
        self.pair_of_pc = sparse.csr_matrix((self.w_pc, (self.cells.pair.to_numpy(), np.arange(npc))), shape=(npair, npc))
        self.pair_of_u = sparse.csr_matrix((self.w_u, (u.pair.to_numpy(), np.arange(nu))), shape=(npair, nu))
        self.panel_groups, self.unit_groups = [], []
        for _, g in self.cells.groupby(["fold", "cell_line"], sort=True):
            idx = g.index.to_numpy()
            drugs = {d: j for j, d in enumerate(sorted(set(g.drug_a) | set(g.drug_b)))}
            h = np.zeros((len(g), len(drugs)))
            h[np.arange(len(g)), [drugs[d] for d in g.drug_a]] = 1
            h[np.arange(len(g)), [drugs[d] for d in g.drug_b]] = 1
            sw = np.sqrt(self.w_pc[idx])
            self.panel_groups.append((idx, sw, basis(h * sw[:, None])))
            rows = np.flatnonzero(np.isin(self.u_pc, idx))
            ua = u.iloc[rows]
            hd = np.zeros((len(rows), 4 * len(drugs)))
            hd[np.arange(len(rows)), [4 * drugs[d] + r for d, r in zip(ua.drug_a, ua.ra)]] = 1
            hd[np.arange(len(rows)), [4 * drugs[d] + r for d, r in zip(ua.drug_b, ua.rb)]] = 1
            local = np.unique(self.u_pc[rows], return_inverse=True)[1]
            wu = self.w_u[rows]
            hm = np.zeros((local.max() + 1, hd.shape[1]))
            np.add.at(hm, local, hd * wu[:, None])
            hm /= np.bincount(local, weights=wu)[:, None]
            hd -= hm[local]
            swu = np.sqrt(wu)
            self.unit_groups.append((rows, swu, basis(hd * swu[:, None])))

    def energies(self, z):
        """Per-pair energies of C_P z and C_D z, and the pair-cell residuals."""
        z = z[self.rows]
        m = self.agg @ z
        rp = np.empty_like(m)
        for idx, sw, q in self.panel_groups:
            x = m[idx] * sw[:, None]
            rp[idx] = (x - q @ (q.T @ x)) / sw[:, None]
        x_all = z - m[self.u_pc]
        rw = np.empty_like(x_all)
        for rows, sw, q in self.unit_groups:
            x = x_all[rows] * sw[:, None]
            rw[rows] = (x - q @ (q.T @ x)) / sw[:, None]
        return self.pair_of_pc @ rp ** 2, self.pair_of_u @ rw ** 2, rp


def jack(full, leaves):
    leaves = np.asarray(leaves, float)
    k = len(leaves)
    half = student.ppf(.975, k - 1) * np.sqrt((k - 1) / k * ((leaves - leaves.mean()) ** 2).sum())
    return [float(full - half), float(full + half)]


def boot(num, den, f, draws, rng):
    ix = rng.integers(len(num), size=(draws, len(num)))
    vals = np.array([f(num[i].sum(0), den[i].sum(0)) for i in ix])
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--observations", required=True)
    ap.add_argument("--resolution", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=97)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    res = Path(a.resolution)
    out = {"inputs": {"observations_sha256": sha256(a.observations), "cache_sha256": sha256(res / "cache.npz"),
                      "coordinates_sha256": sha256(res / "coordinates.parquet"), "split_sha256": sha256(a.split)},
           "settings": {"draws": a.draws, "bootstrap": a.boot, "seed": a.seed}}

    cols = ["block_id", "drug_a", "drug_b", "cell_line", "dose_a", "dose_b", "inhibition", "is_combination", "fold", "pair_id"]
    obs = pd.read_parquet(a.observations, columns=cols)
    obs = obs[obs.is_combination].reset_index(drop=True)
    for c in ["drug_a", "drug_b", "cell_line", "pair_id"]:
        obs[c] = obs[c].astype(str)
    out["rows"] = int(len(obs)); out["blocks"] = int(obs.block_id.nunique())
    obs["ra"] = obs.groupby("block_id").dose_a.rank(method="dense").astype(int) - 1
    obs["rb"] = obs.groupby("block_id").dose_b.rank(method="dense").astype(int) - 1
    key = pd.MultiIndex.from_frame(obs[["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"]])
    obs["unit"], ukeys = pd.factorize(key, sort=True)
    units = pd.DataFrame(ukeys.tolist(), columns=["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"])
    units["n"] = np.bincount(obs.unit)
    first = obs.groupby("unit").head(1).sort_values("unit")
    for c in ["drug_a", "drug_b", "fold"]:
        units[c] = first[c].to_numpy()
    units["pc"] = pd.factorize(units.pair_id + "|" + units.cell_line, sort=True)[0]
    units["pair"] = pd.factorize(units.pair_id, sort=True)[0]
    y = obs.inhibition.to_numpy(float)
    unit_of_row = obs.unit.to_numpy()
    n_u = units.n.to_numpy(float)
    truth = np.bincount(unit_of_row, weights=y) / n_u

    meta = json.loads((res / "inputs.json").read_text())
    coords = pd.read_parquet(res / "coordinates.parquet")
    with np.load(res / "cache.npz") as z:
        values = z["values"]
    ckey = coords[["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"]].astype({"ra": int, "rb": int})
    merged = units.reset_index().merge(ckey.reset_index().rename(columns={"index": "cache_row"}),
                                       on=["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"], how="left")
    if merged.cache_row.isna().any() or merged.cache_row.duplicated().any() or len(coords) != len(units):
        raise ValueError("Unit grouping differs from the cache")
    order = merged.sort_values("index").cache_row.to_numpy(int)
    values = values[order]
    names = meta["names"]
    out["units"] = {"count": int(len(units)), "n_counts": {int(k): int(v) for k, v in units.n.value_counts().items()},
                    "pair_cells": int(units.pc.nunique()),
                    "truth_max_abs_diff_vs_cache": float(np.abs(truth - values[:, names.index("truth")]).max()),
                    "n_matches_cache": bool(np.array_equal(units.n.to_numpy(), coords.n.to_numpy()[order]))}
    col = lambda name: values[:, names.index(name)]
    fam = {"neural_F2": [f"neural_F2:{s}" for s in (2026, 2027, 2028)],
           "neural_base_F2": [f"neural_base_F2:{s}" for s in (2026, 2027, 2028)],
           "gp_F2": ["gp_F2"], "cf_selected": ["cf_selected"], "bliss": ["bliss"]}
    fixed = [truth, truth - col("bliss")]
    sl, s = {"target": slice(0, 2)}, 2
    for f, nm in fam.items():
        fixed += [truth - col(x) for x in nm]
        sl[f] = slice(s, s + len(nm)); s += len(nm)
    fixed = np.column_stack(fixed)

    # Block split. Blocks of a pair-cell that share one dose grid form a group; signs are balanced within groups.
    bl = obs.groupby("block_id").agg(first_unit=("unit", "first"), units=("unit", lambda s: tuple(sorted(s))))
    bl["pc"] = units.pc.to_numpy()[bl.first_unit.to_numpy()]
    bl["grid"] = pd.factorize(bl.pc.astype(str) + ":" + bl.units.astype(str))[0]
    gsize = bl.groupby("grid").size()
    out["split"] = {"grid_group_sizes": {int(k): int(v) for k, v in gsize.value_counts().items()},
                    "pair_cells_with_multiple_grids": int((bl.groupby("pc").grid.nunique() > 1).sum()),
                    "pair_cell_block_counts": {int(k): int(v) for k, v in bl.groupby("pc").size().value_counts().items()}}
    if (gsize % 2).any():
        raise ValueError("Odd grid group")
    rng = np.random.default_rng(a.seed)
    bgrid, bpc = bl.grid.to_numpy(), bl.pc.to_numpy()
    gs = gsize.to_numpy()[bgrid]; ps = bl.groupby("pc").size().reindex(bpc).to_numpy()
    signs = {"grid": np.empty((len(bl), a.draws), np.int8), "panel": np.empty((len(bl), a.draws), np.int8)}
    for r in range(a.draws):
        rg = pd.Series(rng.random(len(bl))).groupby(bgrid).rank(method="first").to_numpy() - 1
        rp = pd.Series(rng.random(len(bl))).groupby(bpc).rank(method="first").to_numpy() - 1
        signs["grid"][:, r] = np.where(rg < gs / 2, 1, -1); signs["panel"][:, r] = np.where(rp < ps / 2, 1, -1)
    row_block = pd.Series(np.arange(len(bl)), index=bl.index).reindex(obs.block_id).to_numpy()
    imb = {k: float(max(np.abs(np.bincount(unit_of_row, weights=v[row_block, r].astype(float))).max() for r in range(a.draws)))
           for k, v in signs.items()}
    out["split"]["max_unit_sign_imbalance"] = imb

    def noise_batches():
        for r0 in range(0, a.draws, a.batch):
            cols_ = []
            for kind in ("grid", "panel"):
                sg = signs[kind][row_block, r0:r0 + a.batch].astype(float)
                cols_.append(np.column_stack([np.bincount(unit_of_row, weights=sg[:, j] * y) / n_u for j in range(sg.shape[1])]))
            yield cols_

    split = pd.read_csv(a.split).astype({"pair_id": str, "cell_line": str}).rename(columns={"fold": "split_fold"})

    def evaluate(keep):
        ops = Operators(units, keep)
        ep, ew, rp = ops.energies(fixed)
        ng = [np.zeros(ep.shape[0]), np.zeros(ep.shape[0])]; nw = [np.zeros(ep.shape[0]), np.zeros(ep.shape[0])]
        for batch in noise_batches():
            for k, d in enumerate(batch):
                p_, w_, _ = ops.energies(d)
                ng[k] += p_.sum(1) / a.draws; nw[k] += w_.sum(1) / a.draws
        cells = ops.cells.merge(split, on=["pair_id", "cell_line"], how="left")
        if cells.split_fold.isna().any():
            raise ValueError("Pair-cell missing from split")
        orc = []
        for j in range(2):
            tv_all = rp[:, j]
            loco = np.zeros(len(tv_all)); matched = np.zeros(len(tv_all))
            for _, g in cells.groupby("pair_id"):
                ii = g.index.to_numpy(); tv = tv_all[ii]; fv = g.split_fold.to_numpy(int)
                loco[ii] = (tv.sum() - tv) / (len(tv) - 1)
                for f in range(5):
                    tr = (fv != f) & (fv != (f + 1) % 5)
                    matched[ii[fv == f]] = tv[tr].mean()
            orc += [loco, matched]
        orc = np.column_stack(orc)
        unit_or = np.zeros((len(units), 4))
        unit_or[ops.rows] = orc[ops.u_pc]
        tz = np.column_stack([truth, truth, fixed[:, 1], fixed[:, 1]]) - unit_or
        eo, _, _ = ops.energies(tz)
        share = []
        for j in range(2):
            g = pd.DataFrame({"p": cells.pair_id, "t": rp[:, j]}).groupby("p").t
            s1, s2, c = g.sum(), g.apply(lambda x: (x ** 2).sum()), g.size()
            share.append(float(((s1 ** 2 - s2) / (c - 1)).sum() / s2.sum()))
        return dict(ep=ep, ew=ew, ng=ng, nw=nw, eo=eo, share=share)

    full = evaluate(np.ones(len(units), bool))
    drugs = sorted(set(units.drug_a) | set(units.drug_b))
    leaves = []
    for d in drugs:
        lv = evaluate(((units.drug_a != d) & (units.drug_b != d)).to_numpy())
        leaves.append({k: (v if k == "share" else ([x.sum(0) for x in v] if isinstance(v, list) else v.sum(0))) for k, v in lv.items()})
        print("deleted", d, flush=True)

    rng = np.random.default_rng(a.seed + 1)
    result = {"cross_cell_covariance_share": {"raw": full["share"][0], "bliss_residual": full["share"][1]}}
    for level, E, N, lk, nk in [("pair_cell", full["ep"], full["ng"], "ep", "ng"), ("dose_shape", full["ew"], full["nw"], "ew", "nw")]:
        for j, tname in enumerate(["raw", "bliss_residual"]):
            T = E[:, j]
            # Noise of the target: the replicate split applies to y; the shared reference cancels.
            Ng, Np = N[0], N[1]
            lT = np.array([lv[lk][j] for lv in leaves]); lNg = np.array([lv[nk][0] for lv in leaves])
            entry = {"target_energy_per_row": float(T.sum() / n_u.sum()),
                     "noise_share_grid_split": float(Ng.sum() / T.sum()),
                     "noise_share_panel_split": float(Np.sum() / T.sum()),
                     "noise_share_grid_pair_interval": boot(Ng[:, None], T[:, None], lambda x, y_: x[0] / y_[0], a.boot, rng),
                     "noise_share_grid_drug_interval": jack(Ng.sum() / T.sum(), lNg / lT), "models": {}}
            for f in fam:
                if f == "bliss" and tname != "raw":
                    continue
                Ef = E[:, sl[f]].mean(1)
                entry["models"][f] = {"r2": float(1 - Ef.sum() / T.sum()),
                                      "noise_corrected_r2_grid": float(1 - (Ef.sum() - Ng.sum()) / (T.sum() - Ng.sum())),
                                      "noise_corrected_r2_panel": float(1 - (Ef.sum() - Np.sum()) / (T.sum() - Np.sum()))}
            result[f"{level}:{tname}"] = entry
    base = full["ep"][:, sl["neural_base_F2"]].mean(1)
    lb = np.array([lv["ep"][sl["neural_base_F2"]].mean() for lv in leaves])
    for j, (tname, kind) in enumerate([("raw", "loco"), ("raw", "matched"), ("bliss_residual", "loco"), ("bliss_residual", "matched")]):
        tj = 0 if tname == "raw" else 1
        T = full["ep"][:, tj]; lT = np.array([lv["ep"][tj] for lv in leaves])
        eo = full["eo"][:, j]; lo = np.array([lv["eo"][j] for lv in leaves])
        r2 = 1 - eo.sum() / T.sum(); gap = (base.sum() - eo.sum()) / T.sum()
        result[f"oracle:{tname}:{kind}"] = {
            "r2": float(r2), "pair_interval": boot(eo[:, None], T[:, None], lambda x, y_: 1 - x[0] / y_[0], a.boot, rng),
            "drug_interval": jack(r2, 1 - lo / lT), "minus_production_f2": float(gap),
            "minus_production_f2_pair_interval": boot(np.column_stack([base, eo]), T[:, None], lambda x, y_: (x[0] - x[1]) / y_[0], a.boot, rng),
            "minus_production_f2_drug_interval": jack(gap, (lb - lo) / lT)}
    for f in ["neural_base_F2", "neural_F2", "gp_F2", "cf_selected"]:
        Ef = full["ep"][:, sl[f]].mean(1); T = full["ep"][:, 1]
        lT = np.array([lv["ep"][1] for lv in leaves]); lE = np.array([lv["ep"][sl[f]].mean() for lv in leaves])
        result[f"pair_cell:bliss_residual:{f}:intervals"] = {
            "pair_interval": boot(Ef[:, None], T[:, None], lambda x, y_: 1 - x[0] / y_[0], a.boot, rng),
            "drug_interval": jack(1 - Ef.sum() / T.sum(), 1 - lE / lT)}
    out["results"] = result
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["units"]), json.dumps(out["split"]))


if __name__ == "__main__":
    main()
