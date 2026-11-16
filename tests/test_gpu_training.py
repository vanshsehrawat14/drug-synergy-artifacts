import json

import numpy as np
import pandas as pd
import pytest
import torch

from synergy_artifacts.features import FeatureBuilder
from synergy_artifacts.gpu import check_step, check_tensor
from synergy_artifacts.io import read_json, write_json
from synergy_artifacts.models import DoseNetwork
from synergy_artifacts.runner import run
from synergy_artifacts.suite import benchmark_runs, candidate_config, initialize, merge, run_dir, select
from synergy_artifacts.training import NeuralRegressor, resolve_device
from test_runner import config, dataset


def monotherapy_frames():
    train = pd.DataFrame(dict(drug_a=["A", "A"], drug_b=["B", "B"], cell_line=["X", "X"], dose_a=[1., 10.],
                              dose_b=[1., 10.], unit_a=["uM"] * 2, unit_b=["uM"] * 2, pair_id=["AB", "AB"]))
    edges = pd.DataFrame(dict(drug_a=["A", "A", "A", "A"], drug_b=["B", "B", "B", "B"], cell_line=["X"] * 4,
                              dose_a=[1., 10., 0., 0.], dose_b=[0., 0., 1., 10.], inhibition=[10., 30., 20., 60.],
                              unit_a=["uM"] * 4, unit_b=["uM"] * 4, pair_id=["AB"] * 4))
    test = train.assign(dose_a=[10 ** .5, 100.], dose_b=[1., 10.])
    return train, edges, test


def test_monotherapy_interpolation_and_reference():
    train, edges, test = monotherapy_frames()
    mean = FeatureBuilder("F2").fit(train, edges)
    values, status = mean.monotherapy_lookup(test, "a")
    assert np.isnan(values).all() and (status == 2).all()
    interpolated = FeatureBuilder("F2", mono_missing="interpolate", reference_inputs=["bliss", "hsa"]).fit(train, edges)
    values, status = interpolated.monotherapy_lookup(test, "a")
    np.testing.assert_allclose(values, [20., 30.])  # log-dose midpoint; constant beyond the range
    assert status.tolist() == [1, 1]
    assert interpolated.monotherapy_status(test).tolist() == [1, 1]
    np.testing.assert_allclose(interpolated.reference(test, "bliss"), [20 + 20 - 4, 30 + 60 - 18])
    np.testing.assert_allclose(interpolated.reference(test, "hsa"), [20, 60])
    a, b, context = interpolated.transform(test)
    assert context.shape == (2, 1 + 2)
    restored = FeatureBuilder.from_state_dict(json.loads(json.dumps(interpolated.state_dict())))
    for x, y in zip(interpolated.transform(test), restored.transform(test)):
        np.testing.assert_array_equal(x, y)
    with pytest.raises(ValueError, match="training-pair"):
        mean.reference(test, "bliss")
    with pytest.raises(ValueError, match="require F2"):
        FeatureBuilder("F0", mono_missing="interpolate")
    coverage = interpolated.monotherapy_coverage(test)["a"]
    assert coverage == {"available": 0, "interpolated": 2, "unavailable": 0, "rows": 2}


def test_monotherapy_only_features_have_no_identity_or_dose():
    train, edges, test = monotherapy_frames()
    builder = FeatureBuilder("F2", identities=False, dose_input=False, mono_missing="interpolate").fit(train, edges)
    a, b, context = builder.transform(test)
    assert a.shape == (2, 2) and context.shape == (2, 0)
    with pytest.raises(ValueError, match="Ablation"):
        FeatureBuilder("F0", identities=False).fit(train).transform(test)


def test_encoder_depth_and_order_dependent_ablation():
    torch.manual_seed(0)
    default = DoseNetwork(5, 3)
    assert [k for k in default.state_dict() if k.startswith("encoder")] == ["encoder.0.weight", "encoder.0.bias"]
    deep = DoseNetwork(5, 3, encoder_layers=2, dropout=0.)
    ordered = DoseNetwork(5, 3, combine="concat", dropout=0.)
    a, b, c = torch.randn(7, 5), torch.randn(7, 5), torch.randn(7, 3)
    deep.eval(), ordered.eval()
    torch.testing.assert_close(deep(a, b, c), deep(b, a, c), rtol=0, atol=0)
    assert not torch.allclose(ordered(a, b, c), ordered(b, a, c))
    with pytest.raises(ValueError):
        DoseNetwork(5, 3, combine="product")


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_plateau_schedule_records_lr_and_resumes_exactly(tmp_path, device):
    rng = np.random.default_rng(3)
    x = tuple(rng.normal(size=(32, n)).astype(np.float32) for n in [3, 3, 2])
    y = x[0][:, 0] * 2 + rng.normal(size=32)
    spec = dict(kind="dose", endpoint_dim=3, cell_dim=2, hidden=[8], encoder_dim=4, dropout=.1)
    training = dict(epochs=12, batch_size=8, lr=.05, patience=100, threads=1, device=device,
                    scheduler={"kind": "plateau", "factor": .5, "patience": 0, "min_lr": 1e-4})
    full = NeuralRegressor(spec, training, 5).fit(x, y, (x, y + 5), tmp_path / "full", {})
    history = read_json(tmp_path / "full" / "history.json")
    assert {"lr", "seconds", "grad_norm_mean", "grad_norm_max"} <= set(history[0])
    assert history[-1]["lr"] < history[0]["lr"]
    NeuralRegressor(spec, {**training, "epochs": 5}, 5).fit(x, y, (x, y + 5), tmp_path / "split", {})
    resumed = NeuralRegressor(spec, training, 5).fit(x, y, (x, y + 5), tmp_path / "split", {},
                                                     resume=tmp_path / "split" / "latest.pt")
    np.testing.assert_array_equal(full.predict(x), resumed.predict(x))
    assert [r["lr"] for r in read_json(tmp_path / "split" / "history.json")] == [r["lr"] for r in history]


def test_unavailable_cuda_is_rejected():
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(ValueError, match="CUDA"):
            resolve_device("cuda")
    with pytest.raises(ValueError):
        resolve_device("mps")


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_device_execution_checks(device):
    device = torch.device(device)
    assert check_tensor(device)["passed"]
    assert check_step(device)["passed"]


def test_residual_monotherapy_and_ordered_runs(dataset, tmp_path):
    base = config(dataset)
    base["features"] = {"regime": "F2", "mono_missing": "interpolate", "reference_offset": "bliss"}
    _, metrics = run(base, tmp_path / "residual")
    frame = pd.read_parquet(tmp_path / "residual" / "evaluation" / "predictions_with_truth.parquet")
    assert {"reference_offset", "monotherapy_status"} <= set(frame)
    assert np.isfinite(frame.reference_offset).all() and metrics["pooled"]["n"] == len(frame)
    from synergy_artifacts.runner import predict_checkpoint
    predict_checkpoint(tmp_path / "residual" / "checkpoints" / "best.pt", tmp_path / "residual" / "predictions.parquet",
                       tmp_path / "restored.parquet")
    np.testing.assert_allclose(pd.read_parquet(tmp_path / "restored.parquet").prediction, frame.prediction, rtol=0, atol=1e-9)
    only = config(dataset)
    only["features"] = {"regime": "F2", "mono_missing": "interpolate", "identities": False, "dose_input": False,
                        "reference_offset": "bliss"}
    assert run(only, tmp_path / "mono_only")[0]["fit"]["parameters"] > 0
    ordered = config(dataset)
    ordered["model"]["combine"] = "concat"
    run(ordered, tmp_path / "ordered")
    swapped = pd.read_parquet(tmp_path / "ordered" / "predictions.parquet")
    assert np.isfinite(swapped.prediction_swapped).all()
    mean = config(dataset)
    mean["features"]["reference_offset"] = "bliss"
    mean["model"] = {"kind": "mean"}
    with pytest.raises(ValueError, match="residual"):
        run(mean, tmp_path / "bad_offset")


def protocol(dataset, tmp_path):
    base = config(dataset)
    write_json(tmp_path / "base.json", base)
    return {"base": {"F0": str(tmp_path / "base.json")}, "shared_overrides": {"training": {"eval_batch_size": 7}},
            "selection": {"seed": 9, "groups": {"F0": ["a", "b"]}},
            "benchmark": {"models": ["a", "selected:F0"], "seeds": [9, 10]},
            "candidates": {"a": {"base": "F0", "overrides": {}},
                           "b": {"base": "F0", "overrides": {"model": {"hidden": [4]}}}}}


def test_suite_configuration_selection_and_expansion(dataset, tmp_path):
    assert merge({"x": {"y": 1, "z": 2}}, {"x": {"y": 3}}) == {"x": {"y": 3, "z": 2}}
    spec = protocol(dataset, tmp_path)
    c = candidate_config(spec, "b", 2, 10, "cpu")
    assert (c["test_fold"], c["seed"], c["model"]["hidden"], c["training"]["eval_batch_size"], c["training"]["device"]) == (2, 10, [4], 7, "cpu")
    root = tmp_path / "suite"
    for name, mse in [("a", 2.), ("b", 1.)]:
        for fold in range(5):
            path = run_dir(root, name, 9, fold)
            write_json(path / "metadata.json", {"fit": {"validation_mse": mse if fold else 3 - mse, "parameters": 1}})
            write_json(path / "evaluation" / "metrics.json", {})
    chosen = select(spec, root, range(5))
    assert chosen["F0"]["0"]["selected"] == "a" and chosen["F0"]["1"]["selected"] == "b"
    runs = benchmark_runs(spec, chosen, [9, 10], range(5))
    assert ("a", 10, 0) in runs and ("b", 10, 1) in runs and len(runs) == len(set(runs))
    initialize(spec, root, None, range(5))
    initialize(spec, root, None, range(5))
    with pytest.raises(ValueError, match="different"):
        initialize(spec, root, {"training": {"epochs": 1}}, range(5))
