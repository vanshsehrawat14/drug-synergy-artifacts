import itertools

import numpy as np
import pandas as pd
import pytest

from cs622.data import prepare
from cs622.io import read_json
from cs622.pair_seen import assign, assignment_sha256, control_config, job, pair_cell_folds
from cs622.runner import run
from cs622.splits import pair_id, partitions, validate_folds

CELLS = ["X1", "X2", "X3", "X4", "X5"]


def units(pairs=15, cells=7):
    return pd.DataFrame([(f"P{p:02d}", f"C{c}") for p in range(pairs) for c in range(cells)], columns=["pair_id", "cell_line"])


def test_assignment_is_deterministic_and_balanced_within_pair():
    a, b = pair_cell_folds(units()), pair_cell_folds(units().sample(frac=1, random_state=3))
    assert assignment_sha256(a) == assignment_sha256(b)
    assert assignment_sha256(a) != assignment_sha256(pair_cell_folds(units(), seed=1))
    counts = a.groupby(["pair_id", "fold"]).size().unstack(fill_value=0)
    assert (counts.max(1) - counts.min(1)).max() <= 1 and counts.shape[1] == 5


def test_assign_keeps_rows_and_groups_pair_cells():
    frame = units().loc[np.repeat(np.arange(105), 3)].reset_index(drop=True)
    frame["fold"], frame["block_id"] = -1, np.arange(len(frame)) // 3
    out = assign(frame)
    assert out[["pair_id", "cell_line"]].equals(frame[["pair_id", "cell_line"]])
    assert out.groupby(["pair_id", "cell_line"]).fold.nunique().max() == 1
    validate_folds(out, unit="pair_cell")
    with pytest.raises(ValueError, match="Pair leakage"):
        validate_folds(out)


def test_job_matches_f2_base_except_split():
    seed, fold, config = job(7)
    assert (seed, fold) == (2027, 2) and config["split"] == {"kind": "pair_cell", "seed": 20261110}
    control = control_config(config)
    assert "split" not in control and control["features"] == {"regime": "F2", "mono_missing": "interpolate"}
    assert control["training"]["device"] == "cuda" and control["training"]["eval_batch_size"] == 16384
    with pytest.raises(ValueError):
        job(15)


@pytest.fixture
def dataset(tmp_path):
    pairs = list(itertools.combinations("ABCDEF", 2))
    assignments = ["AB CD EF", "AC BE DF", "AD BF CE", "AE BD CF", "AF BC DE"]
    folds = {pair_id(x[0], x[1]): f for f, group in enumerate(assignments) for x in group.split()}
    raw, summary, labels, block = [], [], [], 0
    for cell, (a, b) in itertools.product(CELLS, pairs):
        block += 1
        for x, y in itertools.product([0, .1, 1, 10, 100], repeat=2):
            raw.append(dict(block_id=block, conc_r=x, conc_c=y, inhibition=ord(a) - 60 + ord(b) - 60 + int(cell[1]) + np.log1p(x + y),
                            drug_row=a, drug_col=b, conc_r_unit="uM", conc_c_unit="uM", cell_line_name=cell, drug_row_cid=ord(a),
                            drug_col_cid=ord(b), cellosaurus_accession="CVCL_" + cell, study_name="ONEIL"))
        summary.append(dict(block_id=block, drug_row=a, drug_col=b, cell_line_name=cell, conc_r_unit="uM", conc_c_unit="uM",
                            synergy_bliss=1, synergy_hsa=2, synergy_loewe=3, synergy_zip=4, S=5, css=6))
        labels.append(dict(drug_a_name=a, drug_b_name=b, cell_line=cell, synergy=float(block), fold=int(folds[pair_id(a, b)])))
    paths = [tmp_path / n for n in ["raw.csv", "summary.csv", "labels.csv"]]
    for frame, path in zip([raw, summary, labels], paths):
        pd.DataFrame(frame).to_csv(path, index=False)
    out = tmp_path / "prepared"
    prepare(dict(raw=str(paths[0]), summary=str(paths[1]), labels=str(paths[2]), output=str(out), chunksize=500))
    return out


def test_pair_cell_runner_uses_training_blocks_only(dataset, tmp_path, monkeypatch):
    import cs622.runner as runner
    seen, fit = {}, runner.FeatureBuilder.fit
    def record(self, train, monotherapy=None):
        seen["mono"], seen["train"] = set(monotherapy.block_id), set(train.block_id)
        return fit(self, train, monotherapy)
    monkeypatch.setattr(runner.FeatureBuilder, "fit", record)
    config = dict(dataset=str(dataset), table="observations", target="inhibition", test_fold=1, seed=9,
                  features={"regime": "F2", "mono_missing": "interpolate"},
                  model={"kind": "dose", "hidden": [8], "encoder_dim": 4, "dropout": 0.},
                  training=dict(epochs=2, batch_size=64, patience=4, threads=1), bootstrap=10,
                  split={"kind": "pair_cell", "seed": 20261110})
    metadata, metrics = run(config, tmp_path / "pair_seen")
    assert metadata["split_rule"].startswith("pair-cell folds seed 20261110")
    assert metadata["partition_rows"] == dict(train=720, validation=240, test=240)
    pred = pd.read_parquet(tmp_path / "pair_seen" / "predictions.parquet")
    obs = assign(pd.read_parquet(dataset / "observations.parquet"))
    test = obs[obs.observation_id.isin(pred.observation_id)]
    train = obs[~obs.fold.isin([1, 2]) & obs.is_combination]
    assert set(test.fold) == {1} and set(test.pair_id) <= set(train.pair_id)
    assert not set(zip(test.pair_id, test.cell_line)) & set(zip(train.pair_id, train.cell_line))
    tr, va, te = partitions(obs[obs.is_combination].reset_index(drop=True), 1, unit="pair_cell")
    assert len(te) == len(pred) and metrics["pooled"]["n"] == len(pred)
    assert seen["mono"] and seen["mono"] <= seen["train"] and seen["train"] == set(train.block_id)
