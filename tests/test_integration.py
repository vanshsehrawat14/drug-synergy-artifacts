import pandas as pd
import pytest

from synergy_artifacts.export import export_bundle
from synergy_artifacts.io import digest, read_json, write_json
from synergy_artifacts.integration import example, validate_bundle
from test_runner import config, dataset

@pytest.mark.parametrize("regime", ["F0", "F2"])
def test_independent_contract_and_incomplete_submission(dataset, tmp_path, regime):
    c = config(dataset)
    c["features"]["regime"] = regime
    bundle = tmp_path / "bundle"
    export_bundle(c, bundle)
    report = example(bundle, tmp_path / "dummy")
    assert report["partition_rows"] == {"train": 144, "validation": 48, "test": 48}
    assert set(report["rejected"]) == {"omitted_row", "duplicate_id", "nonfinite"}
    contract = read_json(bundle / "contract.json")
    features = pd.read_parquet(bundle / "test_features.parquet").assign(inhibition=0)
    features.to_parquet(bundle / "test_features.parquet", index=False)
    contract["artifacts"]["test_features.parquet"] = digest(bundle / "test_features.parquet")
    write_json(bundle / "contract.json", contract)
    with pytest.raises(ValueError, match="Held-out labels"):
        validate_bundle(bundle)


def test_external_evaluator_rejects_missing_test_rows(dataset,tmp_path):
    from synergy_artifacts.evaluate import evaluate_files
    observations=dataset/"observations.parquet"
    source=pd.read_parquet(observations)
    expected=source[source.is_combination & (source.fold==0)]
    prediction=expected[["observation_id","fold"]].assign(prediction=expected.inhibition)
    path=tmp_path/"predictions.parquet"
    prediction.iloc[:-1].to_parquet(path,index=False)
    with pytest.raises(ValueError,match="coverage"):
        evaluate_files(path,observations,"inhibition",tmp_path/"incomplete")
    prediction.to_parquet(path,index=False)
    result=evaluate_files(path,observations,"inhibition",tmp_path/"complete")
    assert result["pooled"]["n"]==len(expected)
    assert result["pooled"]["mse"]==0.
    # A label-isolated bundle may supply no fold in predictions at all.
    bundle=tmp_path/"test_targets.parquet";expected.to_parquet(bundle,index=False)
    prediction.drop(columns="fold").iloc[:-1].to_parquet(path,index=False)
    with pytest.raises(ValueError,match="coverage"):
        evaluate_files(path,bundle,"inhibition",tmp_path/"incomplete_bundle")
