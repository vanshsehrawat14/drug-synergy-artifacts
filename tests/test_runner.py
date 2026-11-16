import itertools

import numpy as np
import pandas as pd
import pytest

from synergy_artifacts.data import prepare
from synergy_artifacts.io import digest, read_json, write_json
from synergy_artifacts.runner import run
from synergy_artifacts.splits import generated_folds, pair_id


@pytest.fixture
def dataset(tmp_path):
    pairs = list(itertools.combinations("ABCDEF", 2))
    # Five perfect matchings of K6: every training partition sees every drug.
    assignments = ["AB CD EF", "AC BE DF", "AD BF CE", "AE BD CF", "AF BC DE"]
    folds = {pair_id(pair[0], pair[1]): fold for fold, group in enumerate(assignments) for pair in group.split()}
    raw, summary, labels = [], [], []
    for block, (a, b) in enumerate(pairs, 1):
        for x, y in itertools.product([0, .1, 1, 10, 100], repeat=2):
            raw.append(dict(block_id=block, conc_r=x, conc_c=y, inhibition=ord(a)-60+ord(b)-60+np.log1p(x+y),
                drug_row=a, drug_col=b, conc_r_unit="uM", conc_c_unit="uM", cell_line_name="X",
                drug_row_cid=ord(a), drug_col_cid=ord(b), cellosaurus_accession="CVCL_TEST", study_name="ONEIL"))
        summary.append(dict(block_id=block, drug_row=a, drug_col=b, cell_line_name="X", conc_r_unit="uM", conc_c_unit="uM",
            synergy_bliss=1, synergy_hsa=2, synergy_loewe=3, synergy_zip=4, S=5, css=6))
        labels.append(dict(drug_a_name=a, drug_b_name=b, cell_line="X", synergy=float(block), fold=int(folds[pair_id(a,b)])))
    raw_path, summary_path, labels_path = [tmp_path / name for name in ["raw.csv", "summary.csv", "labels.csv"]]
    pd.DataFrame(raw).to_csv(raw_path, index=False)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    out = tmp_path / "prepared"
    manifest = prepare(dict(raw=str(raw_path), summary=str(summary_path), labels=str(labels_path), output=str(out), chunksize=70))
    assert manifest["combination_rows"] == 15 * 16
    return out


def config(dataset, kind="dose"):
    return dict(dataset=str(dataset), table="observations", target="inhibition", test_fold=0, seed=9,
        features={"regime": "F0"}, model={"kind": kind, **({"hidden": [8], "encoder_dim": 4, "dropout": 0.} if kind == "dose" else {})},
        training=dict(epochs=2, batch_size=64, patience=4, threads=1), bootstrap=10)


@pytest.mark.parametrize("kind", ["mean", "additive", "dose"])
def test_full_raw_to_export(dataset, tmp_path, kind):
    output = tmp_path / kind
    metadata, metrics = run(config(dataset, kind), output)
    assert metadata["partition_rows"] == dict(train=144, validation=48, test=48)
    assert metrics["pooled"]["n"] == 48
    prediction = pd.read_parquet(output / "predictions.parquet")
    evaluation = pd.read_parquet(output / "evaluation" / "predictions_with_truth.parquet")
    assert "inhibition" not in prediction
    assert prediction.observation_id.tolist() == evaluation.observation_id.tolist()
    with pytest.raises(FileExistsError):
        run(config(dataset, kind), output)


def test_test_labels_do_not_change_training(dataset, tmp_path):
    c = config(dataset)
    run(c, tmp_path / "original")
    path = dataset / "observations.parquet"
    df = pd.read_parquet(path)
    df.loc[(df.fold == 0) & df.is_combination, "inhibition"] += 100000
    df.to_parquet(path, index=False)
    # Declare the synthetic intervention as a new artifact, rather than bypassing
    # the runner's tamper check on prepared data.
    manifest = read_json(dataset / "manifest.json")
    manifest["artifacts"]["observations"] = digest(path)
    write_json(dataset / "manifest.json", manifest)
    run(c, tmp_path / "perturbed")
    a = pd.read_parquet(tmp_path / "original" / "predictions.parquet")
    b = pd.read_parquet(tmp_path / "perturbed" / "predictions.parquet")
    np.testing.assert_array_equal(a.prediction, b.prediction)
    assert read_json(tmp_path / "original" / "features.json") == read_json(tmp_path / "perturbed" / "features.json")


def test_duplicate_summary_rejected(dataset, tmp_path):
    source = tmp_path / "summary.csv"
    df = pd.read_csv(source)
    pd.concat([df, df.iloc[[0]]]).to_csv(source, index=False)
    c = read_json(dataset / "manifest.json")["settings"]
    c["output"] = str(tmp_path / "bad")
    with pytest.raises(pd.errors.MergeError):
        prepare(c)


def test_preparation_reuse_checks_integrity(dataset):
    manifest = read_json(dataset / "manifest.json")
    assert prepare(manifest["settings"]) == manifest
    path = dataset / "observations.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "inhibition"] += 1
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="artifact changed"):
        prepare(manifest["settings"])
    with pytest.raises(ValueError, match="checksum mismatch"):
        run(config(dataset), dataset.parent / "tampered_run")


def test_chunk_size_does_not_change_outputs(dataset):
    original = read_json(dataset / "manifest.json")
    changed = dict(original["settings"], chunksize=29, output=str(dataset.parent / "different_chunks"))
    rebuilt = prepare(changed)
    assert original["artifacts"] == rebuilt["artifacts"]


def test_runner_resume_can_extend_budget(dataset, tmp_path):
    c = config(dataset)
    c["training"]["epochs"] = 1
    output = tmp_path / "resume"
    run(c, output)
    c["training"]["epochs"] = 2
    metadata, _ = run(c, output, resume=output / "checkpoints" / "latest.pt")
    assert metadata["fit"]["epochs_completed"] == 2


@pytest.mark.parametrize("kind,table,target", [("dose", "observations", "inhibition"), ("deepsynergy", "published", "synergy")])
def test_independent_feature_lazy_runner(dataset, tmp_path, kind, table, target):
    rng = np.random.default_rng(55)
    path = tmp_path / "features.npz"
    provenance = tmp_path / "features.json"
    np.savez(path, drug_ids=np.array(list("ABCDEF")), drug_features=rng.normal(size=(6, 5)),
        cell_ids=np.array(["X"]), cell_features=rng.normal(size=(1, 3)))
    write_json(provenance, {"independent_of_combination_outcomes": True,
        "source": "synthetic fixed random features; not real descriptors", "sha256": digest(path)})
    c = config(dataset)
    c.update(table=table, target=target, features={"regime": "F1", "external": str(path), "provenance": str(provenance), "identities": False, "normalization": "tanh"})
    c["model"] = {"kind": kind, "hidden": [8], "dropout": 0.}
    if kind == "deepsynergy":
        c["model"]["input_dropout"] = 0.
    metadata, metrics = run(c, tmp_path / (kind + "_features"))
    assert np.isfinite(metrics["pooled"]["mse"])
