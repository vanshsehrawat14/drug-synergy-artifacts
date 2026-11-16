import pandas as pd
import pytest

from synergy_artifacts.cv import aggregate
from synergy_artifacts.evaluate import evaluate_files
from synergy_artifacts.export import export_bundle
from synergy_artifacts.io import read_json, write_json
from synergy_artifacts.runner import run
from test_runner import config, dataset


def test_external_prediction_alignment_and_metadata(dataset, tmp_path):
    c = config(dataset, "mean")
    output = tmp_path / "model"
    run(c, output)
    predictions = output / "predictions.parquet"
    report = evaluate_files(predictions, dataset / "observations.parquet", "inhibition", tmp_path / "evaluation")
    assert report["pooled"]["n"] == 48
    frame = pd.read_parquet(predictions).assign(dose_a=9999)
    wrong = tmp_path / "wrong.parquet"
    frame.to_parquet(wrong, index=False)
    with pytest.raises(ValueError, match="metadata mismatch"):
        evaluate_files(wrong, dataset / "observations.parquet", "inhibition", tmp_path / "bad_eval")


def test_cv_requires_comparable_disjoint_folds(dataset, tmp_path):
    c = config(dataset, "mean")
    paths = []
    for fold in range(5):
        c["test_fold"] = fold
        path = tmp_path / f"f{fold}"
        run(c, path)
        paths.append(path)
    report = aggregate(paths, tmp_path / "pooled")
    assert report["complete_five_fold_cv"]
    assert report["metrics"]["pooled"]["n"] == 240
    with pytest.raises(ValueError, match="Incomplete"):
        aggregate(paths[:2], tmp_path / "incomplete")
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate([paths[0], paths[0]], tmp_path / "duplicate")
    metadata = read_json(paths[-1] / "metadata.json")
    metadata["config"]["features"]["regime"] = "F2"
    write_json(paths[-1] / "metadata.json", metadata)
    with pytest.raises(ValueError, match="incompatible"):
        aggregate(paths, tmp_path / "mixed_information")


def test_teammate_bundle_never_exports_test_outcomes_as_features(dataset, tmp_path):
    c = config(dataset)
    c["features"]["regime"] = "F2"
    output = tmp_path / "bundle"
    export_bundle(c, output)
    train = pd.read_parquet(output / "train.parquet")
    test = pd.read_parquet(output / "test_features.parquet")
    mono = pd.read_parquet(output / "training_monotherapy.parquet")
    assert "inhibition" not in test
    assert "inhibition" in train
    assert not (set(mono.pair_id) & set(test.pair_id))
    assert set(mono.pair_id) <= set(train.pair_id)
