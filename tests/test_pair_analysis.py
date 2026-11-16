import itertools

import numpy as np
import pandas as pd

from synergy_artifacts.pair_analysis import noise_draws, oracle_predictions, panel_contrast, ratio_interval
from synergy_artifacts.resolution import means, projections
from synergy_artifacts.shortlist_noise import block_matrix, resample_means


def curve(x, b, c, d, e):
    return c + (d - c) / (1 + (x / e) ** b)


def inverse(y, b, c, d, e):
    return e * ((d - y) / (y - c)) ** (1 / b)


def frame(seed=0):
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations("ABCDEF", 2))
    rows = []
    for panel, (cell, (a, b)) in enumerate(itertools.product(["X", "Y"], pairs)):
        fold = pairs.index((a, b)) % 2
        for ra, rb in itertools.product(range(4), range(4)):
            rows.append(dict(panel=panel, fold=fold, cell_line=cell, drug_a=a, drug_b=b, pair_id=a + b, ra=ra, rb=rb, n=4))
    f = pd.DataFrame(rows)
    f["pair"] = pd.factorize(f.pair_id)[0]
    return f, rng


def test_panel_contrast_matches_resolution_projection():
    f, rng = frame()
    z = rng.normal(size=(len(f), 3))
    _, _, rp, _ = projections(f, z)
    assert np.allclose(panel_contrast(f, z), rp)
    assert np.allclose(panel_contrast(f, z, "A"), projections(f, z, "A")[2])


def test_noise_proxy_cancels_means_and_matches_replicate_mean_variance():
    # 200 panels x 16 units x 4 blocks; whole-block shifts plus row noise.
    panels, units, blocks = 800, 16, 4
    rng = np.random.default_rng(1)
    panel = np.repeat(np.arange(panels), units * blocks)
    block = panel * blocks + np.tile(np.repeat(np.arange(blocks), units), panels)
    group = panel * units + np.tile(np.arange(units), panels * blocks)
    mu = rng.normal(0, 30, panels * units)
    eps = rng.normal(0, 3, len(group)) + rng.normal(0, 2, panels * blocks)[block]
    n = np.bincount(group)
    d = dict(df=pd.DataFrame({"block_id": block}), t={"panel": panel, "y": mu[group] + eps}, group=group, n=n)
    proxy = noise_draws(d, 40, seed=3)
    shifted = noise_draws(dict(d, t={"panel": panel, "y": eps}), 40, seed=3)
    assert np.allclose(proxy, shifted)
    # Unit-level replicate-mean noise variance is (9 + 4)/4; panel means keep the block term 4/4 + 9/64.
    assert abs(proxy.var() - 13 / 4) < 0.12
    pm = means(np.repeat(np.arange(panels), units), proxy)
    assert abs(pm.var() - (4 / 4 + 9 / 64)) < 0.12


def test_oracle_uses_other_cells_and_matched_training_folds():
    f, _ = frame()
    first = f.groupby("panel").head(1)
    t = np.zeros(len(f))
    values = {p: float(i) for i, p in enumerate(first.panel)}
    t = f.panel.map(values).to_numpy(float)
    split = first[["pair_id", "cell_line"]].assign(fold=0)
    split.loc[split.cell_line == "Y", "fold"] = 2
    loco, matched = oracle_predictions(f, t, split)
    g = f.groupby("panel").head(1).assign(loco=loco[f.groupby("panel").head(1).index], matched=matched[f.groupby("panel").head(1).index])
    for _, row in g.iterrows():
        same = g[(g.pair_id == row.pair_id) & (g.panel != row.panel)]
        assert np.isclose(row.loco, same.panel.map(values).mean())
    # Pair-cell fold 0 trains on folds 2, 3, 4: only cell Y panels of the same pair.
    x = g[(g.cell_line == "X")].iloc[0]
    assert np.isclose(x.matched, g[(g.pair_id == x.pair_id) & (g.cell_line == "Y")].panel.map(values).mean())


def test_ratio_interval_identity_and_shapes():
    num, den = np.arange(1, 11, dtype=float), np.full(10, 2.0)
    pairs = [("A", "B")] * 5 + [("C", "D")] * 5
    r = ratio_interval(num, den, pairs, num.sum() - np.array([15., 40, 15, 40]), den.sum() - np.array([10., 10, 10, 10]),
                       lambda x, y: x[0] / y[0])
    assert np.isclose(r["value"], 55 / 20) and r["pair_low"] <= r["value"] <= r["pair_high"]


def test_existence_criterion_against_bisection():
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(400):
        ca, cb = rng.uniform(-5, 5, 2)
        da, db = rng.uniform(20, 110, 2)
        ba, bb = -rng.uniform(0.5, 6, 2)
        ea, eb = 10 ** rng.uniform(-2, 1, 2)
        xa, xb = 10 ** rng.uniform(-2, 1.5, 2)
        lo, hi = max(ca, cb), min(da, db)
        i_is_a = da < db
        fj = curve(xb, bb, cb, db, eb) if i_is_a else curve(xa, ba, ca, da, ea)
        predicted = fj < hi
        # Roots approach the smaller plateau like delta^|b_i| (Proposition 3): add a geometric grid towards it.
        grid = np.concatenate([np.linspace(lo, hi, 20001)[1:-1], hi - (hi - lo) * np.logspace(-1, -13, 400)])
        with np.errstate(all="ignore"):
            g = xa / inverse(grid, ba, ca, da, ea) + xb / inverse(grid, bb, cb, db, eb) - 1
            bi, ci, di, ei, xi = (ba, ca, da, ea, xa) if i_is_a else (bb, cb, db, eb, xb)
            bj, cj, dj, ej, xj = (bb, cb, db, eb, xb) if i_is_a else (ba, ca, da, ea, xa)
            layer = abs(di - ci) * ((1 - xj / inverse(hi, bj, cj, dj, ej)) * ei / xi) ** abs(bi)
        finite = np.isfinite(g)
        if abs(fj - hi) < 1e-3 or not finite.any() or (predicted and not layer > 1e-11 * (hi - lo)):
            continue
        assert predicted == bool((g[finite] < 0).any()), (predicted, fj, hi)
        checked += 1
    assert checked > 300


def test_boundary_layer_identity_is_exact():
    # Root by bisection in extended precision; the exact identity uses delta at the root.
    from mpmath import mp, mpf
    mp.dps = 60
    bi, ci, di, ei, xi = mpf(-4), mpf(0), mpf(60), mpf(1), mpf("0.5")
    bj, cj, dj, ej = mpf(-2), mpf(0), mpf(100), mpf(2)
    Dj = lambda y: ej * ((dj - y) / (y - cj)) ** (1 / bj)
    Di = lambda y: ei * ((di - y) / (y - ci)) ** (1 / bi)
    xj = Dj(di) * (1 - mpf("1e-3"))
    lo, hi = mpf("1e-30"), di - mpf("1e-50")
    for _ in range(400):
        mid = (lo + hi) / 2
        if xi / Di(mid) + xj / Dj(mid) - 1 > 0:
            lo = mid
        else:
            hi = mid
    root = (lo + hi) / 2
    q_tilde = ((1 - xj / Dj(root)) * ei / xi) ** abs(bi)
    assert abs((di - root) - (di - ci) * q_tilde / (1 + q_tilde)) < mpf("1e-40")
    q = ((1 - xj / Dj(di)) * ei / xi) ** abs(bi)
    assert abs((di - root) / ((di - ci) * q / (1 + q)) - 1) < 1e-6


def test_block_matrix_and_resample_means():
    blocks = pd.DataFrame({"pair_id": ["p"] * 4 + ["q"] * 8, "drug_a": "a", "drug_b": "b", "cell_line": "c",
                           "block_id": np.arange(12)})
    m, size, units = block_matrix(blocks, {"N": np.arange(12, dtype=float)})
    assert size.tolist() == [4, 8] and np.isnan(m["N"][0, 4:]).all()
    index = np.tile(np.arange(8), (2, 1))
    assert np.allclose(resample_means(m["N"], size, index), [1.5, 7.5])
    assert np.allclose(resample_means(m["N"], size // 2, index), [0.5, 5.5])
