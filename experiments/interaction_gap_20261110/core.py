"""Shared pieces of the interaction-gap analysis (design.json in this directory).

The O'Neil pair-cell matrix is complete (583 pairs x 39 cell lines), so targets are held as (P, C) arrays
with row-count weights w. The frozen pair-cell contrast removes additive endpoint-drug effects within each
(original pair fold, cell line) panel by row-weighted least squares; it is applied identically to targets,
replicate-split differences and predictions.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BATCH2_SOURCE_ROW = 2273700
FAMILIES = {"f2_base": [f"neural_base_F2:{s}" for s in (2026, 2027, 2028)],
            "f2_selected": [f"neural_F2:{s}" for s in (2026, 2027, 2028)], "gp": ["gp_F2"], "cf": ["cf_selected"]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lf(path):
    """Hash with CRLF normalized to LF (text files checked out on Windows)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def basis(x, tol=1e-10):
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    return u[:, s > tol * max(s.max(), 1.0)]


class Data:
    """Combination rows, units, block edges and the frozen cache, indexed onto the pair-cell matrix."""

    def __init__(self, observations, resolution):
        cols = ["block_id", "drug_a", "drug_b", "cell_line", "dose_a", "dose_b", "inhibition", "is_combination",
                "fold", "pair_id", "source_row"]
        obs = pd.read_parquet(observations, columns=cols)
        for c in ["drug_a", "drug_b", "cell_line", "pair_id"]:
            obs[c] = obs[c].astype(str)
        edges = []
        for e, o in [("a", "b"), ("b", "a")]:
            s = obs[(~obs.is_combination) & (obs["dose_" + e] > 0) & (obs["dose_" + o] == 0)]
            edges.append(s[["block_id", "dose_" + e, "inhibition"]].rename(columns={"inhibition": "mono_" + e}))
        comb = obs[obs.is_combination].reset_index(drop=True)
        for (e, edge) in zip("ab", edges):
            comb = comb.merge(edge, on=["block_id", "dose_" + e], how="left", validate="many_to_one")
        if comb[["mono_a", "mono_b"]].isna().any().any():
            raise ValueError("Missing block edge")
        comb["ra"] = comb.groupby("block_id").dose_a.rank(method="dense").astype(int) - 1
        comb["rb"] = comb.groupby("block_id").dose_b.rank(method="dense").astype(int) - 1
        key = pd.MultiIndex.from_frame(comb[["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"]])
        comb["unit"], ukeys = pd.factorize(key, sort=True)
        units = pd.DataFrame(ukeys.tolist(), columns=["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"])
        units["n"] = np.bincount(comb.unit)
        self.pairs = sorted(units.pair_id.unique())
        self.cells = sorted(units.cell_line.unique())
        pi = {p: i for i, p in enumerate(self.pairs)}
        ci = {c: i for i, c in enumerate(self.cells)}
        units["pi"] = units.pair_id.map(pi).to_numpy()
        units["ci"] = units.cell_line.map(ci).to_numpy()
        comb["pi"] = comb.pair_id.map(pi).to_numpy()
        comb["ci"] = comb.cell_line.map(ci).to_numpy()
        P, C = len(self.pairs), len(self.cells)
        self.shape = (P, C)
        first = comb.groupby("pi").head(1).sort_values("pi")
        self.drug_a, self.drug_b = first.drug_a.to_numpy(), first.drug_b.to_numpy()
        self.fold = first.fold.to_numpy(int)
        if (comb.groupby("pi").fold.nunique() > 1).any():
            raise ValueError("Pair spans folds")
        self.drugs = sorted(set(self.drug_a) | set(self.drug_b))
        self.flat_row = (comb.pi.to_numpy() * C + comb.ci.to_numpy())
        self.flat_unit = (units.pi.to_numpy() * C + units.ci.to_numpy())
        self.w = np.bincount(self.flat_row, minlength=P * C).reshape(P, C).astype(float)
        if (self.w == 0).any():
            raise ValueError("Incomplete pair-cell matrix")
        self.y = comb.inhibition.to_numpy(float)
        self.unit_of_row = comb.unit.to_numpy()
        self.n_u = units.n.to_numpy(float)
        self.truth_u = np.bincount(self.unit_of_row, weights=self.y) / self.n_u
        self.mono_a, self.mono_b = comb.mono_a.to_numpy(float), comb.mono_b.to_numpy(float)
        self.comb, self.units = comb, units
        # Frozen cache aligned to units.
        res = Path(resolution)
        meta = json.loads((res / "inputs.json").read_text())
        coords = pd.read_parquet(res / "coordinates.parquet")
        with np.load(res / "cache.npz") as z:
            values = z["values"]
        ck = coords[["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"]].astype({"ra": int, "rb": int}).reset_index()
        merged = units[["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"]].reset_index().merge(
            ck.rename(columns={"index": "cache_row"}), on=["pair_id", "cell_line", "ra", "rb", "dose_a", "dose_b"], how="left")
        if merged.cache_row.isna().any() or merged.cache_row.duplicated().any() or len(coords) != len(units):
            raise ValueError("Unit grouping differs from the cache")
        values = values[merged.sort_values("index").cache_row.to_numpy(int)]
        self.cache = {nm: values[:, k] for k, nm in enumerate(meta["names"])}
        self.truth_cache_diff = float(np.abs(self.truth_u - self.cache["truth"]).max())
        # Blocks: pair-cell, batch, dose-grid group.
        bl = comb.groupby("block_id").agg(flat=("pi", "first"), ci=("ci", "first"), src=("source_row", "min"),
                                          units=("unit", lambda s: tuple(sorted(s))))
        bl["flat"] = bl.flat.to_numpy() * C + bl.ci.to_numpy()
        bl["batch"] = (bl.src >= BATCH2_SOURCE_ROW).astype(int)
        bl["grid"] = pd.factorize(bl.flat.astype(str) + ":" + bl.units.astype(str))[0]
        bl["grid_batch"] = pd.factorize(bl.grid.astype(str) + ":" + bl.batch.astype(str))[0]
        self.blocks = bl.drop(columns="units")
        self.row_block = pd.Series(np.arange(len(bl)), index=bl.index).reindex(comb.block_id).to_numpy()
        self.row_batch = bl.batch.to_numpy()[self.row_block]

    # Aggregation onto the pair-cell matrix.
    def pc_mean_units(self, z):
        z = np.asarray(z, float)
        if z.ndim == 1:
            z = z[:, None]
        P, C = self.shape
        out = np.stack([np.bincount(self.flat_unit, weights=self.n_u * z[:, k], minlength=P * C) for k in range(z.shape[1])], -1)
        return (out.reshape(P, C, -1) / self.w[..., None]).squeeze(-1) if out.shape[-1] == 1 else out.reshape(P, C, -1) / self.w[..., None]

    def pc_mean_rows(self, y, rows=None, weights=None):
        """Row-level values averaged per pair-cell; rows restricts to a subset (weights then from the subset)."""
        P, C = self.shape
        flat = self.flat_row if rows is None else self.flat_row[rows]
        y = np.asarray(y, float) if rows is None else np.asarray(y, float)[rows]
        s = np.bincount(flat, weights=y, minlength=P * C).reshape(P, C)
        n = np.bincount(flat, minlength=P * C).reshape(P, C).astype(float)
        return s / np.where(n > 0, n, np.nan)

    def split_signs(self, draws=50, seed=97, groups="grid"):
        """Block signs balanced within groups; draws reproduce interaction_audit.py for 'grid' and 'panel'."""
        bl = self.blocks
        rng = np.random.default_rng(seed)
        g_grid, g_pc = bl.grid.to_numpy(), bl.flat.to_numpy()
        gs = np.bincount(g_grid)[g_grid]
        ps = np.bincount(pd.factorize(g_pc)[0])[pd.factorize(g_pc)[0]]
        out = np.empty((len(bl), draws), np.int8)
        gb = bl.grid_batch.to_numpy()
        gbs = np.bincount(gb)[gb]
        if (np.bincount(gb) % 2).any():
            raise ValueError("Odd batch group")
        for r in range(draws):
            rg = pd.Series(rng.random(len(bl))).groupby(g_grid).rank(method="first").to_numpy() - 1
            rp = pd.Series(rng.random(len(bl))).groupby(g_pc).rank(method="first").to_numpy() - 1
            if groups == "grid":
                out[:, r] = np.where(rg < gs / 2, 1, -1)
            elif groups == "panel":
                out[:, r] = np.where(rp < ps / 2, 1, -1)
            elif groups == "batch":
                rb = pd.Series(rg).groupby(gb).rank(method="first").to_numpy() - 1
                out[:, r] = np.where(rb < gbs / 2, 1, -1)
            else:
                raise ValueError(groups)
        return out

    def split_differences(self, signs, y=None):
        """Per pair-cell half-difference means d = sum(s y) / w, shape (P, C, draws)."""
        y = self.y if y is None else y
        P, C = self.shape
        out = np.empty((P, C, signs.shape[1]))
        for r in range(signs.shape[1]):
            s = signs[self.row_block, r] * y
            out[:, :, r] = np.bincount(self.flat_row, weights=s, minlength=P * C).reshape(P, C) / self.w
        return out


class Panel:
    """Row-weighted residual from additive endpoint-drug effects within (fold, cell) panels."""

    def __init__(self, data, w=None, fold=None):
        self.w = data.w if w is None else w
        fold = data.fold if fold is None else fold
        dj = {d: j for j, d in enumerate(data.drugs)}
        self.groups = []
        for f in np.unique(fold):
            idx = np.flatnonzero(fold == f)
            h = np.zeros((len(idx), len(dj)))
            h[np.arange(len(idx)), [dj[d] for d in data.drug_a[idx]]] = 1
            h[np.arange(len(idx)), [dj[d] for d in data.drug_b[idx]]] = 1
            for c in range(self.w.shape[1]):
                sw = np.sqrt(self.w[idx, c])
                self.groups.append((idx, c, sw, basis(h * sw[:, None])))

    def __call__(self, m):
        m = np.asarray(m, float)
        squeeze = m.ndim == 2
        if squeeze:
            m = m[..., None]
        out = np.empty_like(m)
        for idx, c, sw, q in self.groups:
            x = m[idx, c, :] * sw[:, None]
            out[idx, c, :] = (x - q @ (q.T @ x)) / sw[:, None]
        return out[..., 0] if squeeze else out


def train_baseline(data, m, exclude):
    """Additive drug-by-cell baseline per cell line, fitted on pairs whose fold is not in `exclude`.

    Returns m minus the baseline for every pair (out of sample for excluded folds). m: (P, C) or (P, C, K).
    """
    m = np.asarray(m, float)
    squeeze = m.ndim == 2
    if squeeze:
        m = m[..., None]
    dj = {d: j for j, d in enumerate(data.drugs)}
    h = np.zeros((len(data.pairs), len(dj)))
    h[np.arange(len(data.pairs)), [dj[d] for d in data.drug_a]] = 1
    h[np.arange(len(data.pairs)), [dj[d] for d in data.drug_b]] = 1
    tr = ~np.isin(data.fold, list(exclude))
    if not (h[tr].sum(0) > 0)[h[~tr].sum(0) > 0].all():
        raise ValueError("Held-out drug absent from training pairs")
    out = np.empty_like(m)
    for c in range(m.shape[1]):
        sw = np.sqrt(data.w[tr, c])
        coef = np.linalg.lstsq(h[tr] * sw[:, None], m[tr, c, :] * sw[:, None], rcond=None)[0]
        out[:, c, :] = m[:, c, :] - h @ coef
    return out[..., 0] if squeeze else out


def components(t, w, v):
    """Energy, noise, pair-wide (cross-cell U-statistic) and pair-by-cell parts; per-pair sufficient statistics."""
    e = (w * t ** 2).sum(1)
    n = (w * v).sum(1)
    s1, s2 = (w * t).sum(1), (w ** 2 * t ** 2).sum(1)
    d1, d2 = w.sum(1), (w ** 2).sum(1)
    return np.column_stack([e, n, s1 ** 2 - s2, d1 ** 2 - d2, d1, s1 ** 2 / d1])


def shares(stats):
    e, n, num, den, d1, between = stats.sum(0) if stats.ndim == 2 else stats
    tau2 = num / den
    pw = tau2 * d1 / e
    return {"energy_per_row": e / d1, "noise": n / e, "pair_wide": pw, "pair_by_cell": 1 - pw - n / e,
            "tau2": tau2, "descriptive_pair_mean": between / e}


def anchor_masks(pairs, n_cells, seed):
    """Per pair, a permutation of cell positions from a hash of (pair_id, seed); never reads targets."""
    out = np.empty((len(pairs), n_cells), int)
    for i, p in enumerate(pairs):
        h = int(hashlib.sha256(f"{p}|{seed}".encode()).hexdigest()[:16], 16)
        out[i] = np.random.default_rng(h).permutation(n_cells)
    return out


def pick(pairs, candidates, seed, salt):
    """Deterministic choice among sorted candidates from a hash of (pair, seed, salt)."""
    h = int(hashlib.sha256(f"{pairs}|{seed}|{salt}".encode()).hexdigest()[:16], 16)
    return candidates[np.random.default_rng(h).integers(len(candidates))]


class Covariance:
    """Cross-cell second moments of training targets (missing entries allowed) and the B1/B2 models."""

    def __init__(self, t, v, mask=None):
        mask = np.ones(t.shape, bool) if mask is None else mask
        x, m = np.where(mask, t, 0.0), mask.astype(float)
        self.S = (x.T @ x) / np.maximum(m.T @ m, 1)
        self.N = np.where(mask, v, 0.0).sum(0) / np.maximum(m.sum(0), 1)
        off = ~np.eye(len(self.N), dtype=bool)
        self.tau2 = float(self.S[off].mean())
        self.s2 = float(np.diag(self.S).mean() - self.tau2)
        self.Csig = self.S - np.diag(self.N)
        self._eig = np.linalg.eigh(self.Csig)

    def b1_weight(self, k):
        k = np.asarray(k, float)
        return np.where(k > 0, k * self.tau2 / (self.s2 + k * self.tau2), 0.0)

    def signal(self, lam, r):
        C = self.Csig
        d = np.diag(C)
        floor = 0.01 * d.mean()
        off = C[~np.eye(len(d), dtype=bool)].mean()
        kcs = np.full(C.shape, off)
        np.fill_diagonal(kcs, np.maximum(d, off + floor))
        if lam >= 1:
            return kcs
        ev, U = self._eig
        top = np.argsort(ev)[::-1][:r]
        top = top[ev[top] > 0]
        kr = (U[:, top] * ev[top]) @ U[:, top].T
        klr = kr + np.diag(np.maximum(d - np.diag(kr), floor))
        return lam * kcs + (1 - lam) * klr


def blup(K, N, t_anchor, A, E):
    """Best linear predictor of E cells from A cells; t_anchor (n, k, R), A (n, k), E (n, m) -> (n, m, R)."""
    KA = K[A[:, :, None], A[:, None, :]] + np.einsum("nk,kj->nkj", N[A], np.eye(A.shape[1]))
    KEA = K[E[:, :, None], A[:, None, :]]
    return KEA @ np.linalg.solve(KA, t_anchor)


def loco_b2(K, N, t):
    """Leave-one-cell-out B2 predictions for every pair-cell; t (P, C, R)."""
    C = K.shape[0]
    out = np.empty_like(t)
    for c in range(C):
        o = np.delete(np.arange(C), c)
        beta = np.linalg.solve(K[np.ix_(o, o)] + np.diag(N[o]), K[o, c])
        out[:, c, :] = np.einsum("pkr,k->pr", t[:, o, :], beta)
    return out


def boot_ratio(num, den, draws, seed, f=lambda n, d: 1 - n / d):
    """Pair-cluster percentile bootstrap of f(sum num, sum den); num/den per pair (P,) or (P, K)."""
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(den), size=(draws, len(den)))
    vals = np.array([f(num[i].sum(0), den[i].sum(0)) for i in ix])
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def jackknife(full, leaves):
    from scipy.stats import t as student
    leaves = np.asarray(leaves, float)
    k = len(leaves)
    half = student.ppf(.975, k - 1) * np.sqrt((k - 1) / k * ((leaves - leaves.mean()) ** 2).sum())
    return [float(full - half), float(full + half)]


def drug_leaves(data):
    """Boolean keep-masks over pairs, one per deleted drug."""
    return {d: (data.drug_a != d) & (data.drug_b != d) for d in data.drugs}
