"""Synthetic checks of the interaction-gap estimators (experiments/interaction_structure)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "interaction_structure"))
from anchors import masked_contrast, predict_all  # noqa: E402
from core import Covariance, Panel, anchor_masks, blup, components, shares, train_baseline  # noqa: E402


def stub(n_drugs=12, n_cells=10, folds=5, seed=0):
    rng = np.random.default_rng(seed)
    drugs = [f"D{i:02d}" for i in range(n_drugs)]
    pairs = [(a, b) for i, a in enumerate(drugs) for b in drugs[i + 1:]]
    fold = rng.permutation(np.arange(len(pairs)) % folds)
    return SimpleNamespace(drugs=drugs, drug_a=np.array([p[0] for p in pairs]), drug_b=np.array([p[1] for p in pairs]),
                           pairs=[f"{a}|{b}" for a, b in pairs], fold=fold, w=np.full((len(pairs), n_cells), 64.0),
                           shape=(len(pairs), n_cells))


def drug_design(d):
    j = {x: i for i, x in enumerate(d.drugs)}
    h = np.zeros((len(d.pairs), len(j)))
    h[np.arange(len(d.pairs)), [j[x] for x in d.drug_a]] = 1
    h[np.arange(len(d.pairs)), [j[x] for x in d.drug_b]] = 1
    return h


def test_panel_is_a_projection_that_removes_additive_drug_effects():
    d = stub()
    rng = np.random.default_rng(1)
    panel = Panel(d)
    m = rng.normal(size=d.w.shape)
    r = panel(m)
    assert np.allclose(panel(r), r)
    additive = drug_design(d) @ rng.normal(size=(len(d.drugs), d.w.shape[1]))
    assert np.allclose(panel(additive), 0, atol=1e-10)
    assert np.allclose(panel(m + additive), r)


def test_masked_contrast_with_full_mask_equals_panel():
    d = stub()
    m = np.random.default_rng(2).normal(size=d.w.shape)
    assert np.allclose(masked_contrast(d, m, np.ones(d.w.shape, bool)), Panel(d)(m))


def test_components_recover_planted_variances():
    rng = np.random.default_rng(3)
    P, C, tau2, sig2, nu = 4000, 30, 2.0, 5.0, 1.0
    t = rng.normal(0, np.sqrt(tau2), (P, 1)) + rng.normal(0, np.sqrt(sig2), (P, C)) + rng.normal(0, np.sqrt(nu), (P, C))
    sh = shares(components(t, np.ones((P, C)), np.full((P, C), nu)).sum(0))
    total = tau2 + sig2 + nu
    assert sh["pair_wide"] == pytest.approx(tau2 / total, abs=0.01)
    assert sh["noise"] == pytest.approx(nu / total, abs=0.01)
    assert sh["pair_by_cell"] == pytest.approx(sig2 / total, abs=0.01)
    assert sh["descriptive_pair_mean"] > sh["pair_wide"]


def test_split_sign_estimator_is_unbiased_for_the_noise_of_a_mean():
    rng = np.random.default_rng(4)
    y = 3.0 + rng.normal(0, 2.0, size=(200000, 4))
    s = np.array([1, 1, -1, -1])
    d = (y * rng.permuted(np.tile(s, (len(y), 1)), axis=1)).mean(1)
    assert np.mean(d ** 2) == pytest.approx(4.0 / 4, rel=0.02)


def test_blup_with_compound_symmetry_equals_the_empirical_bayes_intercept():
    C, tau2, sig2, nu, k = 12, 1.5, 3.0, 0.5, 5
    K = np.full((C, C), tau2) + sig2 * np.eye(C)
    N = np.full(C, nu)
    A, E = np.arange(k)[None], np.arange(k, C)[None]
    x = np.random.default_rng(5).normal(size=(1, k, 1))
    p = blup(K, N, x, A, E)[0, :, 0]
    s2 = sig2 + nu
    assert np.allclose(p, k * tau2 / (s2 + k * tau2) * x.mean())


def test_b1_weight_formula():
    rng = np.random.default_rng(6)
    t = rng.normal(0, 1, (3000, 1)) + rng.normal(0, 2, (3000, 20))
    cov = Covariance(t, np.zeros_like(t))
    assert cov.tau2 == pytest.approx(1.0, abs=0.1)
    assert cov.s2 == pytest.approx(4.0, abs=0.2)
    for k in (1, 4, 16):
        assert float(cov.b1_weight(k)) == pytest.approx(k * cov.tau2 / (cov.s2 + k * cov.tau2))
    assert float(cov.b1_weight(0)) == 0.0


def test_masks_are_deterministic_and_never_read_targets():
    a = anchor_masks(["x", "y", "z"], 39, 7)
    assert np.array_equal(a, anchor_masks(["x", "y", "z"], 39, 7))
    assert not np.array_equal(a, anchor_masks(["x", "y", "z"], 39, 8))
    assert all(sorted(r) == list(range(39)) for r in a)
    # Evaluation cells are the same positions for every k.
    assert np.array_equal(a[:, 16:], anchor_masks(["x", "y", "z"], 39, 7)[:, 16:])


def structured(P, C, clusters, rng, strength):
    lab = np.arange(C) % clusters
    a = rng.normal(0, 1.0, (P, 1))
    u = rng.normal(0, 1.0, (P, clusters))[:, lab] * strength
    return a + u + rng.normal(0, 1.0, (P, C))


@pytest.mark.parametrize("strength,expect_gain", [(1.5, True), (0.0, False)])
def test_b2_gains_only_with_cell_structure(strength, expect_gain):
    rng = np.random.default_rng(8)
    P, C, k = 3000, 24, 8
    t = structured(P, C, 3, rng, strength)
    tr, te = np.arange(2000), np.arange(2000, P)
    cov = Covariance(t[tr], np.full((len(tr), C), 1.0))
    masks = np.tile(np.arange(C), (len(te), 1))
    A, E = masks[:, :k], masks[:, 16:]
    tA, tE = t[te][:, :k], t[te][:, 16:]
    b1 = float(cov.b1_weight(k)) * tA.mean(1, keepdims=True)
    b2 = blup(cov.signal(0.0, C), cov.N, tA[..., None], A, E)[..., 0]
    gain = ((tE - b1) ** 2).sum() / (tE ** 2).sum() - ((tE - b2) ** 2).sum() / (tE ** 2).sum()
    assert (gain > 0.05) if expect_gain else (abs(gain) < 0.01)


def test_predictions_never_use_evaluation_cells():
    rng = np.random.default_rng(9)
    P, C = 60, 24
    t = structured(P, C, 3, rng, 1.0)
    D = rng.normal(size=(P, C, 4))
    cov = Covariance(t[:40], np.ones((40, C)))
    rows = np.arange(40, 60)
    masks = anchor_masks([str(i) for i in rows], C, 1)
    A, E = masks[:, :8], masks[:, 16:]
    d4 = d5 = np.arange(len(rows))
    base, _ = predict_all(cov, (0.25, C), 8, t, D, A, E, rows, d4, d5)
    t2 = t.copy()
    t2[rows[:, None], E] += 100.0
    pert, _ = predict_all(cov, (0.25, C), 8, t2, D, A, E, rows, d4, d5)
    for m in ["B1", "B2"]:
        assert np.allclose(base[m], pert[m])


def test_train_baseline_ignores_excluded_folds():
    d = stub()
    rng = np.random.default_rng(10)
    m = rng.normal(size=d.w.shape)
    r = train_baseline(d, m, {0})
    m2 = m.copy()
    m2[d.fold == 0] += 50.0
    r2 = train_baseline(d, m2, {0})
    keep = d.fold != 0
    assert np.allclose(r[keep], r2[keep])
    assert np.allclose(r2[~keep] - r[~keep], 50.0)
