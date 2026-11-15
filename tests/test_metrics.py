import numpy as np
import pandas as pd
import pytest

from cs622.metrics import align_predictions, evaluate, interaction_contrast, regression_metrics


def test_hand_metrics():
    m = regression_metrics([1, 2, 3], [2, 2, 2])
    assert m["mse"] == pytest.approx(2 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert m["mae"] == pytest.approx(2 / 3)
    assert m["r2"] == 0
    assert m["pearson"] is None


def test_perfect_and_constant():
    assert regression_metrics([1, 2, 3], [1, 2, 3])["r2"] == 1
    assert regression_metrics([4, 4], [4, 4])["r2"] is None
    assert regression_metrics([4, 4], [4, 4])["mse"] == 0
    assert regression_metrics(np.arange(3712), np.full(3712, 45.123456789))["pearson"] is None
    assert regression_metrics(np.full(3712, 45.123456789), np.arange(3712))["r2"] is None


@pytest.mark.parametrize("y,p", [([1, np.nan], [1, 2]), ([1, 2], [1, np.inf]), ([], []), ([1], [1, 2])])
def test_invalid_metrics(y, p):
    with pytest.raises(ValueError):
        regression_metrics(y, p)


def panel():
    import itertools
    pairs = list(itertools.combinations("ABCDE", 2))
    return pd.DataFrame({"drug_a": [a for a, b in pairs], "drug_b": [b for a, b in pairs],
        "pair_id": [a + b for a, b in pairs], "cell_line": "C", "fold": 0,
        "observation_id": np.arange(10), "y": np.arange(10) ** 2, "prediction": np.arange(10)})


def test_contrast_matches_original_projector():
    df = panel()
    for column in ["drug_a", "drug_b"]:
        df[column] = pd.Categorical(df[column], categories=list("ABCDEF"))
    h = np.array([[int(d == a) + int(d == b) for d in "ABCDE"] for a, b in zip(df.drug_a, df.drug_b)])
    y = df.y.to_numpy(float)
    p = df.prediction.to_numpy(float)
    yr = y - h @ np.linalg.lstsq(h, y, rcond=None)[0]
    pr = p - h @ np.linalg.lstsq(h, p, rcond=None)[0]
    assert interaction_contrast(df, "y")["contrast_r2"] == pytest.approx(1 - sum((yr-pr)**2) / sum(yr**2))
    swapped = df.rename(columns={"drug_a": "drug_b", "drug_b": "drug_a"})
    assert interaction_contrast(swapped, "y")["contrast_r2"] == pytest.approx(interaction_contrast(df, "y")["contrast_r2"])


def test_aggregation_alignment_bootstrap():
    df = panel()
    pred = df[["observation_id", "prediction"]].sample(frac=1, random_state=1)
    aligned = align_predictions(df.drop(columns="prediction"), pred)
    assert aligned.prediction.tolist() == df.prediction.tolist()
    result = evaluate(aligned, "y", bootstrap=20)
    assert result["by_cell_line"][0]["mse"] == result["pooled"]["mse"]
    assert result["uncertainty"]["clusters"] == 10
    assert result == evaluate(aligned, "y", bootstrap=20)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate(pd.concat([df, df.iloc[[0]]]), "y")
    with pytest.raises(ValueError, match="panel"):
        interaction_contrast(pd.concat([df, df.iloc[[0]]]), "y")
