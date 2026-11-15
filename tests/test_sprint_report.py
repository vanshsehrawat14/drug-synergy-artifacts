import numpy as np
import pandas as pd

from cs622.io import digest, read_json
from cs622.metrics import regression_metrics
from cs622.runner import run
from cs622.sprint_report import from_sums, load_truth, pair_sums, paired, verify_run
from test_runner import config, dataset


def toy(seed=0):
    rng = np.random.default_rng(seed)
    pair = np.repeat(np.arange(6), 20)
    y = rng.normal(30, 20, len(pair))
    return {"y": y, "pair": pair, "pairs": np.arange(6), "pair_n": np.bincount(pair).astype(float), "pair_fold": np.arange(6) % 5,
            "pair_drugs": [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("A", "D"), ("B", "D")], "drugs": ["A", "B", "C", "D"]}, rng


def test_pair_sums_reproduce_pooled_and_residual_metrics():
    t, rng = toy()
    p, reference = t["y"] + rng.normal(0, 5, len(t["y"])), t["y"] + rng.normal(2, 9, len(t["y"]))
    r = t["y"] - reference
    total = lambda w: np.bincount(t["pair"], weights=w)
    truth = np.column_stack([total(t["y"]), total(t["y"] ** 2), total(r), total(r * r)])
    got, direct = from_sums(np.ones(6), pair_sums(p, t), truth), regression_metrics(t["y"], p)
    for key in ["mse", "rmse", "mae", "r2", "pearson"]:
        np.testing.assert_allclose(got[key], direct[key], rtol=1e-10)
    np.testing.assert_allclose(got["residual_r2"], 1 - np.sum((t["y"] - p) ** 2) / np.sum((r - r.mean()) ** 2), rtol=1e-10)
    weights = np.vstack([np.ones(6), [2, 0, 1, 1, 1, 1]])
    np.testing.assert_allclose(from_sums(weights, pair_sums(p, t), truth)["mse"][0], direct["mse"], rtol=1e-10)


def test_paired_change_is_seed_averaged_and_classified():
    t, rng = toy(1)
    jackknife = np.array([[d not in pair for pair in t["pair_drugs"]] for d in t["drugs"]], dtype=float)
    weights = np.stack([np.bincount(rng.integers(6, size=6), minlength=6) for _ in range(200)]).astype(float)
    worse = [pair_sums(t["y"] + rng.normal(0, 6, len(t["y"])), t) for _ in range(2)]
    better = [pair_sums(t["y"] + rng.normal(0, 2, len(t["y"])), t)]
    change = paired(better, worse, t, weights, jackknife)
    expected = better[0][:, 1].sum() / 120 - np.mean([w[:, 1].sum() for w in worse]) / 120
    np.testing.assert_allclose(change["delta_mse"], expected, rtol=1e-12)
    assert change["ci_low"] <= change["delta_mse"] <= change["ci_high"] and change["verdict"] == "lower error"
    assert change["seed_pairs_lower"] == "2/2" and change["folds_lower"] == 5
    assert paired(worse, better, t, weights, jackknife)["verdict"] == "higher error"
    same = paired(better, better, t, weights, jackknife)
    assert same["delta_mse"] == 0 and same["verdict"] == "not established"


def test_verify_run_accepts_a_clean_run_and_detects_tampering(dataset, tmp_path):
    c = config(dataset)
    c["training"]["device"] = "cpu"
    run(c, tmp_path / "run")
    meta = read_json(tmp_path / "run" / "metadata.json")
    df, sha, train_ids = load_truth(dataset)
    rows = df[df.fold == 0]
    arguments = (c, meta["git"], meta["code_sha256"], sha, rows, train_ids[0])
    clean = verify_run(tmp_path / "run", *arguments)
    assert all(v for v in clean.values() if isinstance(v, bool)) and clean["restore_max_abs_difference"] <= 1e-5
    assert digest(dataset / "observations.parquet") == sha
    predictions = pd.read_parquet(tmp_path / "run" / "predictions.parquet")
    predictions.loc[3, "prediction"] += .5
    predictions.to_parquet(tmp_path / "run" / "predictions.parquet", index=False)
    tampered = verify_run(tmp_path / "run", *arguments)
    assert not tampered["metrics"] and not tampered["checkpoint"] and tampered["predictions"]
    other = {**c, "seed": 10}
    assert not verify_run(tmp_path / "run", other, *arguments[1:])["config"]


def test_contrast_energies_match_interaction_contrast_and_drug_deletion():
    from cs622.metrics import interaction_contrast
    from cs622.sprint_report import contrast_energies
    rng = np.random.default_rng(3)
    drugs = list("ABCDEFGH")
    pairs = [(a, b) for i, a in enumerate(drugs) for b in drugs[i + 1:]]
    frame = pd.DataFrame([{"fold": k % 2, "pair_id": f"{a}{b}", "drug_a": a, "drug_b": b, "cell_line": c}
                          for k, (a, b) in enumerate(pairs) for c in ["X", "Y"] if k % 5])
    frame["response"] = rng.normal(size=len(frame))
    p = frame.response.to_numpy() + rng.normal(scale=.5, size=len(frame))
    energy = contrast_energies({"panel_frame": frame, "drugs": drugs}, {"zero": np.zeros(len(frame)), "p": p})
    full = interaction_contrast(frame.assign(prediction=p), "response")
    np.testing.assert_allclose([energy["p"][0], energy["zero"][0]], [full["contrast_sse"], full["contrast_ss"]], rtol=1e-10)
    kept = frame[(frame.drug_a != "C") & (frame.drug_b != "C")]
    reduced = interaction_contrast(kept.assign(prediction=p[kept.index]), "response")
    np.testing.assert_allclose(energy["p"][1 + drugs.index("C")], reduced["contrast_sse"], rtol=1e-10)


def test_team_predictions_align_by_identifier_and_require_full_coverage(tmp_path):
    import pytest
    from cs622.sprint_report import team_predictions
    t = {"y": np.zeros(5), "rows": {k: np.array([k]) for k in range(5)}, "ids": {k: [f"o{k}"] for k in range(5)}}
    for k in range(5):
        pd.DataFrame({"observation_id": [f"o{k}"], "prediction": [float(k)]}).to_parquet(tmp_path / f"fold_{k}.parquet")
    np.testing.assert_array_equal(team_predictions(tmp_path, t), np.arange(5.))
    pd.DataFrame({"observation_id": ["o4", "o4"], "prediction": [1., 1.]}).to_parquet(tmp_path / "fold_4.parquet")
    with pytest.raises(ValueError, match="exactly"):
        team_predictions(tmp_path, t)
