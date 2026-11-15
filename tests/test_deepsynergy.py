import gzip
import pickle
import pickletools

import numpy as np
import pandas as pd
import pytest
import torch

from cs622 import deepsynergy as ds
from cs622.features import load_independent_features
from cs622.io import digest, read_json, write_json
from test_runner import dataset


def encode(ops):
    """Serialize (opcode, argument) pairs; FRAME's free argument is set to zero."""
    table = {op.name: op for op in pickletools.opcodes}
    out = bytearray()
    for name, arg in ops:
        op = table[name]
        out += op.code.encode("latin-1")
        kind = op.arg.name if op.arg else None
        if kind in {"uint1", "uint2", "uint8"}:
            out += int(0 if arg is ds.ANY else arg).to_bytes({"uint1": 1, "uint2": 2, "uint8": 8}[kind], "little")
        elif kind == "int4":
            out += int(arg).to_bytes(4, "little", signed=True)
        elif kind == "unicodestring1":
            out += bytes([len(arg.encode())]) + arg.encode()
        elif kind == "bytes1":
            out += bytes([len(arg)]) + arg
    return bytes(out)


def declaration(size=ds.PAYLOAD_BYTES):
    return b"\x8e" + size.to_bytes(8, "little")


def test_original_header_whitelist():
    header = encode(ds.HEADER)
    assert len(header) == ds.HEADER_BYTES
    ds.validate_original_header(header + declaration())
    hostile = [("SHORT_BINUNICODE", {"numpy.core.multiarray": "posix", "_reconstruct": "system"}.get(a, a))
               if n == "SHORT_BINUNICODE" else (n, a) for n, a in ds.HEADER]
    for bad in [encode(hostile) + declaration(), pickle.dumps({"x": 1}, protocol=4).ljust(ds.HEADER_BYTES + 9, b".")]:
        with pytest.raises(ValueError, match="pickle structure"):
            ds.validate_original_header(bad)
    with pytest.raises(ValueError, match="payload"):
        ds.validate_original_header(header + declaration(ds.PAYLOAD_BYTES - 8))


def synthetic_archive(tmp_path, monkeypatch, corrupt=False, trailer=b"\x94t\x94b."):
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"drug_a_name": ["a-1", "a-1", "B"], "drug_b_name": ["B", "C", "C"],
                  "cell_line": ["x", "y", "x"]}).to_csv(labels, index=False)
    drug = {"A1": [1., 2.], "B": [3., 4.], "C": [5., 6.]}
    cell = {"X": [7.], "Y": [8.]}
    order = [("A1", "B", "X"), ("A1", "C", "Y"), ("B", "C", "X")]
    order += [(b, a, c) for a, b, c in order]
    payload = np.array([drug[a] + drug[b] + cell[c] for a, b, c in order])
    if corrupt:
        payload[4, 2] += 1e-12  # drug A1 in the swapped half no longer matches its first occurrence
    for name, value in dict(ROWS=6, COLUMNS=5, DRUG_FEATURES=2, HEADER_BYTES=0, LABEL_SHA=digest(labels)).items():
        monkeypatch.setattr(ds, name, value)
    monkeypatch.setattr(ds, "validate_original_header", lambda header: None)
    archive = tmp_path / "X.p.gz"
    with gzip.open(archive, "wb") as stream:
        stream.write(declaration(payload.nbytes) + payload.astype("<f8").tobytes() + trailer)
    return archive, labels, drug, cell


def test_extraction_aligns_both_orderings(tmp_path, monkeypatch):
    archive, labels, drug, cell = synthetic_archive(tmp_path, monkeypatch)
    provenance = ds.extract_original(archive, labels, tmp_path / "out")
    assert (provenance["drugs"], provenance["cells"], provenance["extraction"]["nonfinite_values"]) == (3, 2, 0)
    loaded = load_independent_features(tmp_path / "out" / "features.npz", tmp_path / "out" / "features.provenance.json")
    assert loaded["drug"] == drug and loaded["cell"] == cell
    with pytest.raises(FileExistsError):
        ds.extract_original(archive, labels, tmp_path / "out")


@pytest.mark.parametrize("change,message", [({"corrupt": True}, "alignment"),
                                            ({"trailer": b"cos\nsystem\n."}, "after the feature payload")])
def test_extraction_rejects_misalignment_and_trailing_code(tmp_path, monkeypatch, change, message):
    archive, labels, _, _ = synthetic_archive(tmp_path, monkeypatch, **change)
    with pytest.raises(ValueError, match=message):
        ds.extract_original(archive, labels, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("norm", ["norm", "tanh", "tanh_norm"])
def test_entity_normalization_equals_doubled_row_statistics(norm):
    rng = np.random.default_rng(0)
    table = rng.normal(size=(5, 4))
    table[:, 1] = 3.
    table[:, 2] = [1., 1., 9., 1., 50.]  # constant over training entities only
    counts = np.array([3., 1., 0., 2., 0.])
    rows = np.repeat(table, counts.astype(int), axis=0)
    keep = rows.std(0) > 0

    def standardize(x, reference):
        return (x - reference.mean(0)) / reference.std(0)
    expected = standardize(table[:, keep], rows[:, keep])
    if norm != "norm":
        train = np.tanh(standardize(rows[:, keep], rows[:, keep]))
        expected = np.tanh(expected)
        if norm == "tanh_norm":
            expected = standardize(expected, train)
    actual, kept = ds.normalize_entities(table, counts, norm)
    assert kept.tolist() == keep.tolist() == [True, False, False, True]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    with pytest.raises(ValueError):
        ds.normalize_entities(table, counts, "minmax")


def test_smoothed_epoch_is_centered_moving_average():
    losses = [9., 3., 8., 2., 7., 6., 5., 4., 6., 7.]
    assert ds.smoothed_epoch(losses, 1) == 3
    assert ds.smoothed_epoch(losses, 3) == 2  # mean of epochs 1..3 is 4.33; isolated minimum at epoch 3 is smoothed away
    assert ds.smoothed_epoch(losses[:2], 25) in {0, 1}


@pytest.mark.parametrize("name", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_stage_resume_matches_uninterrupted_fit(tmp_path, name):
    rng = np.random.default_rng(1)
    tables = rng.normal(size=(3, 4)), rng.normal(size=(2, 3))
    device = torch.device(name)
    rows = tuple(torch.tensor(r, device=device) for r in [[0, 1, 2, 1, 2, 0], [1, 2, 0, 0, 1, 2], [0, 1, 0, 0, 1, 0]])
    y, train = torch.randn(6, generator=torch.Generator().manual_seed(2)).to(device), np.arange(6)
    settings = dict(layers=[8], input_dropout=.2, dropout=.5, placement="paper", lr=1e-2, momentum=.5,
                    batch_size=2, checkpoint_every=1, threads=1)
    full = ds._Stage(tables, rows, settings, device, 3, tmp_path / "full.pt").fit(train, y, 4)
    ds._Stage(tables, rows, settings, device, 3, tmp_path / "split.pt").fit(train, y, 2)
    resumed = ds._Stage(tables, rows, settings, device, 3, tmp_path / "split.pt")
    assert resumed.epoch == 2
    resumed.fit(train, y, 4)
    index = torch.arange(6, device=device)
    torch.testing.assert_close(full.predict(index), resumed.predict(index), rtol=0, atol=0)
    assert [r["train_mse"] for r in full.history] == [r["train_mse"] for r in resumed.history]
    with pytest.raises(ValueError, match="settings"):
        ds._Stage(tables, rows, {**settings, "lr": 1.}, device, 3, tmp_path / "split.pt")


def test_reproduction_protocol_on_synthetic_features(tmp_path):
    rng = np.random.default_rng(4)
    drugs, cells = ["A", "B", "C", "D"], ["X", "Y", "Z"]
    drug_features, cell_features = rng.normal(size=(4, 6)), rng.normal(size=(3, 5))
    drug_features[:, 0] = 1.
    np.savez_compressed(tmp_path / "f.npz", drug_ids=np.array(drugs), drug_features=drug_features,
                        cell_ids=np.array(cells), cell_features=cell_features)
    write_json(tmp_path / "f.json", {"source": "synthetic", "independent_of_combination_outcomes": True,
                                     "sha256": digest(tmp_path / "f.npz")})
    pairs = [(a, b) for i, a in enumerate(drugs) for b in drugs[i + 1:]]
    labels = pd.DataFrame([{"drug_a_name": a, "drug_b_name": b, "cell_line": c, "fold": [0, 1, 2, 3, 4, 0][i],
                            "synergy": float(rng.normal(scale=10))} for i, (a, b) in enumerate(pairs) for c in cells])
    labels.to_csv(tmp_path / "labels.csv", index=False)
    settings = dict(layers=[8, 4], epochs=3, window=1, checkpoint_every=1, batch_size=4, threads=1)
    reports = [ds.reproduce(tmp_path / "f.npz", tmp_path / "f.json", tmp_path / name, folds=[0], device="cpu",
                            settings=settings, labels=tmp_path / "labels.csv") for name in ["a", "b"]]
    fold = tmp_path / "a" / "fold_0"
    metrics, predictions = read_json(fold / "metrics.json"), pd.read_parquet(fold / "predictions.parquet")
    assert metrics["test_samples"] == len(predictions) == (labels.fold == 0).sum()
    assert metrics["validation_fold"] == 1 and metrics["selected"]["final_epochs"] >= 1
    np.testing.assert_allclose(predictions.prediction, (predictions.prediction_ab + predictions.prediction_ba) / 2)
    np.testing.assert_allclose(metrics["symmetrized"]["mse"], np.mean((predictions.y - predictions.prediction) ** 2))
    assert not list(fold.glob("*.pt")) and "mean_over_folds" not in reports[0]
    pd.testing.assert_frame_equal(predictions, pd.read_parquet(tmp_path / "b" / "fold_0" / "predictions.parquet"))
    again = ds.reproduce(tmp_path / "f.npz", tmp_path / "f.json", tmp_path / "a", folds=[0], device="cpu",
                         settings=settings, labels=tmp_path / "labels.csv")
    assert again["folds"] == reports[0]["folds"]


def test_dose_adaptation_scores_the_benchmark_rows(dataset, tmp_path):
    from cs622.sprint_report import load_truth
    rng = np.random.default_rng(5)
    np.savez_compressed(tmp_path / "f.npz", drug_ids=np.array(list("ABCDEF")), drug_features=rng.normal(size=(6, 5)),
                        cell_ids=np.array(["X"]), cell_features=rng.normal(size=(1, 3)))
    write_json(tmp_path / "f.json", {"source": "synthetic", "independent_of_combination_outcomes": True,
                                     "sha256": digest(tmp_path / "f.npz")})
    settings = dict(layers=[8, 4], input_dropout=.2, dropout=.5, placement="paper", norm="tanh_norm", lr=1e-3, momentum=.5,
                    batch_size=16, epochs=4, patience=2, center_target=True, checkpoint_every=1, print_every=1,
                    eval_batch_size=64, threads=1)
    protocol = {"name": "test", "dataset": str(dataset), "features": str(tmp_path / "f.npz"), "provenance": str(tmp_path / "f.json"),
                "seeds": [3], "folds": [0], "bootstrap": 10, "settings": settings,
                "pilot": {"fold": 0, "seed": 3, "settings": {"epochs": 2, "patience": 2}}}
    write_json(tmp_path / "protocol.json", protocol)
    [pilot] = ds.dose_suite(tmp_path / "protocol.json", tmp_path / "pilot", device="cpu", pilot=True)
    assert pilot["fit"]["epochs_completed"] == 2 and not list((tmp_path / "pilot").rglob("predictions.parquet"))
    [meta] = ds.dose_suite(tmp_path / "protocol.json", tmp_path / "a", device="cpu")
    run = tmp_path / "a" / "seed_3" / "fold_0"
    prediction, test = pd.read_parquet(run / "predictions.parquet"), load_truth(dataset)[0].query("fold == 0")
    assert prediction.observation_id.tolist() == test.observation_id.tolist() and "inhibition" not in prediction
    assert (prediction.dose_a.to_numpy() == test.dose_a.to_numpy()).all() and meta["inputs"]["input_dim"] == (5 + 1) * 2
    np.testing.assert_allclose(prediction.prediction, (prediction.prediction_ab + prediction.prediction_ba) / 2)
    metrics = read_json(run / "evaluation" / "metrics.json")
    np.testing.assert_allclose(metrics["pooled"]["mse"], np.mean((test.inhibition.to_numpy() - prediction.prediction.to_numpy()) ** 2))
    losses = [r["validation_mse"] for r in read_json(run / "history.json")]
    assert meta["fit"]["best_epoch"] == int(np.argmin(losses)) and len(losses) == min(4, meta["fit"]["best_epoch"] + 3)
    best = torch.load(run / "checkpoints" / "best.pt", weights_only=True)
    assert best["epoch"] == meta["fit"]["best_epoch"] and not (run / "checkpoints" / "latest.pt").exists()
    assert meta["checkpoint_sha256"] == digest(run / "checkpoints" / "best.pt")
    ds.dose_suite(tmp_path / "protocol.json", tmp_path / "b", device="cpu")
    pd.testing.assert_frame_equal(prediction, pd.read_parquet(tmp_path / "b" / "seed_3" / "fold_0" / "predictions.parquet"))
    write_json(tmp_path / "protocol.json", {**protocol, "seeds": [4]})
    with pytest.raises(ValueError, match="different protocol"):
        ds.dose_suite(tmp_path / "protocol.json", tmp_path / "a", device="cpu")
