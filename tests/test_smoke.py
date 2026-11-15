import pandas as pd
import pytest

def test_no_data_smoke_exports_complete_predictions(tmp_path):
    from cs622.smoke import smoke
    metadata, metrics = smoke(tmp_path / "smoke")
    assert metadata["partition_rows"] == {"train": 144, "validation": 48, "test": 48}
    assert metrics["pooled"]["n"] == 48
    predicted = pd.read_parquet(tmp_path / "smoke/run/predictions.parquet")
    assert "inhibition" not in predicted
    assert predicted.observation_id.is_unique


def test_external_metadata_checked_without_pair_column(tmp_path):
    from cs622.evaluate import evaluate_files
    truth = pd.DataFrame({"observation_id": ["1", "2"], "cell_line": ["X", "Y"],
                          "unit_a": ["uM", "uM"], "inhibition": [1., 2.]})
    truth.to_parquet(tmp_path / "truth.parquet", index=False)
    for field, value in [("cell_line", "WRONG"), ("unit_a", "nM")]:
        prediction = truth[["observation_id", field]].copy().assign(prediction=[1., 2.])
        prediction[field] = value
        prediction.to_parquet(tmp_path / "prediction.parquet", index=False)
        with pytest.raises(ValueError, match="metadata mismatch"):
            evaluate_files(tmp_path / "prediction.parquet", tmp_path / "truth.parquet", "inhibition", tmp_path / field)
