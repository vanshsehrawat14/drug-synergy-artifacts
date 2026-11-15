import itertools

import numpy as np
import pandas as pd

from cs622.gp_shared import FAMILIES, InducingPredictor, fit_cell, gp_module, make_model, HIS, shared_frame


def synthetic(cell="C1", drugs=7, seed=0):
    rng = np.random.default_rng(seed)
    potency = rng.lognormal(0, .5, drugs)
    rows = []
    for block, (a, b) in enumerate(itertools.combinations(range(drugs), 2)):
        for da, db in itertools.product([.1, .3, 1.0], repeat=2):
            ea, eb = 90 * da / (da + potency[a]), 90 * db / (db + potency[b])
            rows.append({"observation_id": f"{cell}:{block}:{da}:{db}", "pair_id": f"D{a}|D{b}", "drug_a": f"D{a}",
                         "drug_b": f"D{b}", "cell_line": cell, "fold": block % 5, "block_id": block, "dose_a": da,
                         "dose_b": db, "unit_a": "uM", "unit_b": "uM", "inhibition": ea + eb - ea * eb / 90 + rng.normal(0, 2)})
    return shared_frame(pd.DataFrame(rows))


def test_fit_cell_predicts_each_test_row_once_and_never_sees_test_targets():
    cell = synthetic()
    predictions, rows = fit_cell(cell, fold=0, cap=200)
    test = cell[cell.fold == 0]
    assert sorted(predictions.observation_id) == sorted(test.observation_id)
    assert np.isfinite(predictions[["prediction", "predictive_std"]].to_numpy()).all()
    assert all(r["validation_fold"] == 1 and r["n_train"] <= 200 for r in rows)
    chosen = [r["family"] for r in rows if r["selected"]]
    assert chosen == [min(FAMILIES, key=lambda f: next(r["validation_rmse"] for r in rows if r["family"] == f))]

    # Test targets are not read: removing them leaves every prediction unchanged.
    hidden = cell.copy()
    hidden.loc[hidden.fold == 0, "inhibition"] = np.nan
    again, _ = fit_cell(hidden, fold=0, cap=200)
    assert np.array_equal(again.prediction.to_numpy(), predictions.prediction.to_numpy())



def test_inducing_predictor_with_all_training_inputs_matches_the_exact_gp():
    gp = gp_module()
    cell = synthetic(seed=1)
    train, query = cell[cell.fold != 0], cell[cell.fold == 0]
    model = make_model("rbf", HIS, seed=42).fit(train[gp.FEATURES], train.inhibition)
    exact_mean, exact_std = model.predict(query[gp.FEATURES], return_std=True)
    mean, std = InducingPredictor(model, train, train).predict(query)
    assert np.allclose(mean, exact_mean, atol=1e-3 * np.abs(exact_mean).max())
    assert np.allclose(std, exact_std, rtol=1e-3)


def test_inducing_study_never_predicts_the_test_fold():
    cell = synthetic(seed=2)
    hidden = cell.copy()
    hidden.loc[hidden.fold == 0, "inhibition"] = np.nan
    predictions, rows = fit_cell(hidden, fold=0, cap=120, method={"predictor": "inducing"}, predict_test=False)
    assert predictions is None
    assert all(r["n_inducing"] <= r["n_train"] and np.isfinite(r["validation_rmse"]) for r in rows)
