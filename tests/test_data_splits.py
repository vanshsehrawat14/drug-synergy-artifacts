import numpy as np
import pandas as pd
import pytest

from cs622.data import canonicalize
from cs622.splits import generated_folds, pair_id, partitions, published_folds, validate_folds


def test_pair_swap_and_dose_association():
    raw = pd.DataFrame(dict(drug_row=["A", "B"], drug_col=["B", "A"], cell_line_name=["cell"] * 2,
        conc_r=[1., 2.], conc_c=[2., 1.], conc_r_unit=["uM"] * 2, conc_c_unit=["uM"] * 2,
        drug_row_cid=[11, 22], drug_col_cid=[22, 11], inhibition=[-5., 105.]))
    df = canonicalize(raw)
    assert df.pair_id.nunique() == 1
    assert df.dose_a.tolist() == [1., 1.]
    assert df.cid_a.tolist() == [11, 11]
    assert df.inhibition.tolist() == [-5., 105.]
    assert pair_id("A-B", "C") == pair_id("c", "AB")


def test_deterministic_grouped_split():
    pairs = [pair_id(str(i), str(i + 1)) for i in range(30)]
    a = generated_folds(pairs, 4)
    pd.testing.assert_frame_equal(a, generated_folds(list(reversed(pairs)), 4))
    df = a.loc[a.index.repeat(12)].reset_index(drop=True)
    for f in range(5):
        tr, va, te = partitions(df, f)
        groups = [set(df.iloc[ix].pair_id) for ix in [tr, va, te]]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        assert sum(map(len, [tr, va, te])) == len(df)


def test_reject_leaking_folds():
    df = generated_folds([pair_id(str(i), "Z") for i in range(10)])
    leaked = pd.concat([df, df.iloc[[0]].assign(fold=(df.iloc[0].fold + 1) % 5)])
    with pytest.raises(ValueError, match="leakage"):
        validate_folds(leaked)


def test_reject_duplicate_observations():
    df = generated_folds([pair_id(str(i), "Z") for i in range(10)])
    df["observation_id"] = "same"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_folds(df)


def test_published_fold_conflict(tmp_path):
    df = pd.DataFrame(dict(drug_a_name=["A", "B"], drug_b_name=["B", "A"], fold=[0, 1]))
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="crosses"):
        published_folds(path)
