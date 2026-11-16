from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from synergy_artifacts import cf

REAL = Path("data/raw/deepsynergy/labels.csv").exists() and cf.TARGETS.exists()


def synthetic(n_drugs=6, cells=3, seed=0):
    """All pairs of n_drugs in every cell line, one full 4 x 4 dose grid per pair-cell, additive dose profile plus noise."""
    rng = np.random.default_rng(seed)
    effect = rng.normal(0, 10, (cells, n_drugs, 4))
    effect -= effect.mean(2, keepdims=True)
    pairs = list(combinations(range(n_drugs), 2))
    grid = np.array([(i, j) for i in range(4) for j in range(4)])
    rows = [(a, b, c, i, j, p % 5) for p, (a, b) in enumerate(pairs) for c in range(cells) for i, j in grid]
    da, db, cell, ra, rb, fold = (np.array(x) for x in zip(*rows))
    e = effect[cell, da, ra] + effect[cell, db, rb] + rng.normal(0, 0.1, len(da))
    pc = np.repeat(np.arange(len(da) // 16), 16)
    e -= (np.bincount(pc, weights=e) / 16)[pc]
    return {"da": da, "db": db, "cell": cell, "ra": ra, "rb": rb, "fold": fold, "e": e, "n_drugs": n_drugs, "cells": list(range(cells))}


@pytest.mark.parametrize("partner", [False, True])
def test_dose_design_sums_to_zero_within_every_pair_cell(partner):
    data = synthetic()
    X = cf.design(np.arange(len(data["e"])), data, partner)
    u = np.random.default_rng(1).normal(size=X.shape[1])
    profile = X @ u
    assert np.allclose(profile.reshape(-1, 16).sum(1), 0)
    assert np.allclose(np.asarray(X.sum(1)).ravel(), 0)


def test_dose_profile_recovers_an_additive_profile_from_training_rows_only():
    data = synthetic()
    fit = cf.dose_profile(data, 0, False, [1e-3, 1.0], predict_test=True)
    test = data["fold"] == 0
    assert fit["validation_sse"] / fit["validation_rows"] < 0.05
    assert np.mean((fit["test"] - data["e"][test]) ** 2) < 0.05
    noisy = {**data, "e": data["e"].copy()}
    noisy["e"][test] = 1e3
    again = cf.dose_profile(noisy, 0, False, [1e-3, 1.0], predict_test=True)
    assert np.array_equal(again["test"], fit["test"])


@pytest.mark.skipif(not REAL, reason="needs the DeepSynergy labels and Shar's target matrix")
def test_pair_cell_reproduces_her_predict_cf_and_ignores_test_pair_labels():
    data, her = cf.load(), cf.shar()
    spec = {"k": [3, 5, 10, 20], "shrink": [2.0, 5.0, 20.0], "similarity": "pearson"}
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    tr, va, te = her.split(data["folds"], 1)
    theirs, _, _ = her.predict_cf(data["Y"], data["H"], data["aa"], data["bb"], tr, va, te)
    mine = cf.pair_cell(data, 1, spec, lambdas, predict_test=True)
    assert np.array_equal(mine["test"], theirs)
    noisy = {**data, "Y": data["Y"].copy()}
    noisy["Y"][te] = 0.0
    for kind in ["pearson", "cell"]:
        a = cf.pair_cell(data, 1, {**spec, "similarity": kind}, lambdas, predict_test=True)
        b = cf.pair_cell(noisy, 1, {**spec, "similarity": kind}, lambdas, predict_test=True)
        assert np.array_equal(a["test"], b["test"])
