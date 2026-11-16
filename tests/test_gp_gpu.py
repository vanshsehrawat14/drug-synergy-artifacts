import numpy as np
import pandas as pd
import torch

from synergy_artifacts.gp_gpu import DEFAULT, Encoder, ExactGP, collapse, fit_cell
from synergy_artifacts.gp_shared import HIS, gp_module, make_model
from tests.test_gp_shared import synthetic

CPU = torch.device("cpu")


def replicated(seed=3, reps=3):
    """Synthetic cell with replicate blocks, as in ONEIL."""
    cell = synthetic(seed=seed)
    rng = np.random.default_rng(seed)
    copies = []
    for r in range(reps):
        c = cell.copy()
        c["observation_id"] = c.observation_id + f":r{r}"
        c["block_id"] = c.block_id.astype(int) * 10 + r
        c["block_key"] = c.cell_line + ":" + c.block_id.astype(str)
        c["inhibition"] = c.inhibition + rng.normal(0, 3, len(c))
        copies.append(c)
    return pd.concat(copies, ignore_index=True)


def sklearn_pair(family, rows):
    gp = gp_module()
    model = make_model(family, HIS, seed=42).fit(rows[gp.FEATURES], rows.inhibition)
    reg = model.named_steps["regressor"]
    return model, reg


def test_collapsed_likelihood_matches_sklearn_on_every_row():
    rows = replicated()
    X = Encoder().fit(rows).transform(rows)
    assert len(collapse(X, rows.inhibition.to_numpy())[0]) < len(rows) / 2
    for family in ["rbf", "matern"]:
        model, reg = sklearn_pair(family, rows)
        gp = ExactGP(family, DEFAULT, CPU)
        y = rows.inhibition.to_numpy(float)
        gp.y_mean, gp.y_std = float(y.mean()), float(y.std())
        data = gp._data(X, y)
        for theta in [reg.kernel_.theta, reg.kernel_.theta + np.array([0.3, -0.2, 0.5])]:
            want, grad = reg.log_marginal_likelihood(theta, eval_gradient=True)
            t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
            value = gp.nll(t, data)
            value.backward()
            assert np.isclose(-value.item(), want, rtol=1e-9, atol=1e-7)
            assert np.allclose(-t.grad.numpy(), grad, rtol=1e-6, atol=1e-6)


def test_fit_and_prediction_match_sklearn():
    rows = replicated(seed=4)
    train, query = rows[rows.fold != 0], rows[rows.fold == 0]
    enc = Encoder().fit(train)
    for family in ["rbf", "matern"]:
        model, reg = sklearn_pair(family, train)
        gp = ExactGP(family, DEFAULT, CPU).fit(enc.transform(train), train.inhibition)
        assert np.allclose(gp.theta, reg.kernel_.theta, atol=1e-3)
        assert np.isclose(gp.lml, reg.log_marginal_likelihood_value_, rtol=1e-6)
        mean, std, latent = gp.predict(enc.transform(query))
        want_mean, want_std = model.predict(query[gp_module().FEATURES], return_std=True)
        assert np.allclose(mean, want_mean, atol=1e-3 * np.abs(want_mean).max())
        assert np.allclose(std, want_std, rtol=1e-3)
        assert (latent < std).all()


def test_inducing_with_every_input_is_the_exact_posterior():
    rows = replicated(seed=5)
    train, query = rows[rows.fold != 0], rows[rows.fold == 0]
    enc = Encoder().fit(train)
    gp = ExactGP("matern", DEFAULT, CPU).fit(enc.transform(train), train.inhibition)
    exact = gp.predict(enc.transform(query))
    approx = gp.inducing(gp.Xu.numpy()).predict(enc.transform(query))
    assert np.allclose(approx[0], exact[0], atol=1e-3 * np.abs(exact[0]).max())
    assert np.allclose(approx[1], exact[1], rtol=1e-3)
    fewer = gp.inducing(gp.Xu.numpy()[::4]).predict(enc.transform(query))
    assert np.isfinite(fewer[0]).all() and (fewer[1] > 0).all()


def monotherapy(cell):
    rows = []
    for (pair, a, b), g in cell.groupby(["pair_id", "drug_a", "drug_b"]):
        for drug, dose_col in [(a, "dose_a"), (b, "dose_b")]:
            for dose in sorted(g[dose_col].unique()):
                rows.append({"pair_id": pair, "drug_a": a, "drug_b": b, "cell_line": cell.cell_line.iloc[0],
                             "dose_a": dose if drug == a else 0.0, "dose_b": dose if drug == b else 0.0,
                             "unit_a": "uM", "unit_b": "uM", "inhibition": 30 * dose})
    return pd.DataFrame(rows)


def test_fit_cell_hides_the_test_fold_and_reports_every_variant():
    cell = replicated(seed=6, reps=2)
    mono = monotherapy(cell)
    hidden = cell.copy()
    hidden.loc[hidden.fold == 0, "inhibition"] = np.nan
    hidden_mono = mono[mono.pair_id.isin(set(cell.pair_id[cell.fold != 0]))]
    for method in [{}, {"dose": "per_drug"}, {"monotherapy": True}, {"target": "bliss_residual"},
                   {"rows": 120, "inducing": [10]}, {"hyper_rows": 120}]:
        pred, val, records = fit_cell(cell, mono, 0, method, dev=CPU)
        again, _, _ = fit_cell(hidden, hidden_mono, 0, method, dev=CPU)
        assert sorted(pred.observation_id) == sorted(cell.observation_id[cell.fold == 0])
        assert set(val.observation_id) == set(cell.observation_id[cell.fold == 1])
        assert np.allclose(again.prediction, pred.prediction)
        assert sum(r["selected"] and r["predictor"] == "exact" for r in records) == 1


def test_uncertainty_summaries_on_known_distributions():
    from synergy_artifacts.gp_uncertainty import calibration, risk, within_rank_correlation
    rng = np.random.default_rng(0)
    std = rng.uniform(1, 5, 50000)
    y = rng.normal(0, std)
    rows, pit, extra = calibration(y, np.zeros_like(y), std)
    assert all(abs(r["coverage"] - r["level"]) < 0.01 for r in rows)
    assert np.allclose(pit, 0.1, atol=0.01) and abs(extra["z_sd"] - 1) < 0.02
    _, gain = risk(y, np.zeros_like(y), std)
    _, none = risk(y, np.zeros_like(y), rng.permutation(std))
    assert 0.2 < gain < 1 and abs(none) < 0.05
    groups = np.repeat(np.arange(500), 100)
    assert within_rank_correlation(groups, np.abs(y), std) > 0.2
    assert abs(within_rank_correlation(groups, np.abs(y), rng.permutation(std))) < 0.05
