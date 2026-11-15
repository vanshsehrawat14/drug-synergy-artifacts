import itertools

import pandas as pd
import pytest

from cs622.data import SCORES, prepare
from cs622.io import read_json


def write_sources(tmp_path):
    rows, summary, block = [], [], 0
    for study, drugs in [("ALMANAC", ["D1", "D2", "D3", "D4", "D5", "D6"]), ("ONEIL", ["D1", "D2"])]:
        for a, b in itertools.combinations(drugs, 2):
            for cell in ["C1", "C2"]:
                block += 1
                for x, y in itertools.product([0, .1, 1.0], repeat=2):
                    rows.append({"block_id": block, "conc_r": x, "conc_c": y, "inhibition": 10 * x + 20 * y, "drug_row": b,
                                 "drug_col": a, "conc_r_unit": "uM", "conc_c_unit": "uM", "cell_line_name": cell,
                                 "drug_row_cid": 1, "drug_col_cid": 2, "cellosaurus_accession": "X", "study_name": study})
                summary.append({"block_id": block, "drug_row": b, "drug_col": a, "cell_line_name": cell, "conc_r_unit": "uM",
                                "conc_c_unit": "uM", **{s: 1.0 for s in SCORES}})
    pd.DataFrame(rows).to_csv(tmp_path / "raw.csv", index=False)
    pd.DataFrame(summary).to_csv(tmp_path / "summary.csv", index=False)


def test_other_study_gets_label_free_generated_pair_folds(tmp_path):
    write_sources(tmp_path)
    config = {"study": "ALMANAC", "raw": str(tmp_path / "raw.csv"), "summary": str(tmp_path / "summary.csv"),
              "output": str(tmp_path / "out"), "fold_seed": 2026}
    manifest = prepare(config)
    obs = pd.read_parquet(tmp_path / "out" / "observations.parquet")
    assert obs.observation_id.str.startswith("dc14:ALMANAC:").all() and len(obs) == 15 * 2 * 9
    assert obs.groupby("pair_id", observed=True).fold.nunique().eq(1).all() and set(obs.fold) == set(range(5))
    assert not (tmp_path / "out" / "published.parquet").exists()
    assert manifest["fold_source"].startswith("generated_folds(seed=") and manifest["combination_rows_per_block"] == {"4": 30}
    assert read_json(tmp_path / "out" / "manifest.json")["dataset"] == "DrugComb v1.4 ALMANAC"


def test_oneil_still_requires_published_labels(tmp_path):
    write_sources(tmp_path)
    with pytest.raises(ValueError, match="published"):
        prepare({"raw": str(tmp_path / "raw.csv"), "summary": str(tmp_path / "summary.csv"), "output": str(tmp_path / "o")})


def test_sparse_drugs_are_excluded_and_held_out_drugs_stay_seen(tmp_path):
    write_sources(tmp_path)
    raw = pd.read_csv(tmp_path / "raw.csv")
    extra = raw[(raw.study_name == "ALMANAC") & (raw.block_id == 1)].assign(block_id=999, drug_row="RARE")
    pd.concat([raw, extra]).to_csv(tmp_path / "raw.csv", index=False)
    summary = pd.read_csv(tmp_path / "summary.csv")
    pd.concat([summary, summary.iloc[[0]].assign(block_id=999, drug_row="RARE")]).to_csv(tmp_path / "summary.csv", index=False)
    config = {"study": "ALMANAC", "raw": str(tmp_path / "raw.csv"), "summary": str(tmp_path / "summary.csv"),
              "output": str(tmp_path / "out"), "fold_seed": 2026, "min_pairs_per_drug": 3}
    manifest = prepare(config)
    obs = pd.read_parquet(tmp_path / "out" / "observations.parquet")
    assert manifest["excluded_sparse_drugs"]["drugs"] == ["RARE"] and "RARE" not in set(obs.drug_a) | set(obs.drug_b)
    for k in range(5):
        held = obs[obs.fold.isin([k, (k + 1) % 5])]
        train = obs[~obs.fold.isin([k, (k + 1) % 5])]
        assert set(held.drug_a) | set(held.drug_b) <= set(train.drug_a) | set(train.drug_b)


def test_drug_cell_coverage_drops_sparse_drug_cells_and_keeps_held_out_drug_cells_seen():
    from cs622.data import _covering_folds, _sparse_drug_cells
    units = [(a, b, "C1") for a, b in itertools.combinations([f"D{i}" for i in range(8)], 2)] + [("D0", "D1", "C2")]
    df = pd.DataFrame([{"pair_id": f"{a}|{b}", "drug_a": a, "drug_b": b, "cell_line": c} for a, b, c in units])
    excluded, kept = _sparse_drug_cells(df, 3)
    assert excluded["pair_cells"] == 1 and set(kept.cell_line) == {"C1"} and {"D0|C2", "D1|C2"} <= set(excluded["drug_cells"])
    folds, seed = _covering_folds(kept, 2026, by_cell=True)
    f = kept.merge(folds, on="pair_id")
    for k in range(5):
        held = f.fold.isin([k, (k + 1) % 5])
        assert set(f.drug_a[held]) | set(f.drug_b[held]) <= set(f.drug_a[~held]) | set(f.drug_b[~held])
