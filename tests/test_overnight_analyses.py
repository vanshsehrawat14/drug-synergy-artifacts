import numpy as np
import pandas as pd

from cs622.loewe_audit import loewe_residual, strict_reference
from cs622.regimes import holm, projected, regimes


def test_strict_loewe_root_matches_sham_combination():
    # Two copies of one LL.4 curve, f(x) = 100 x / (1 + x): Loewe additivity gives f(xa + xb).
    p = np.array([-1., 0., 100., 1.])
    root, status = strict_reference(.5, .5, "LL.4", p, "LL.4", p)
    assert status == "defined" and abs(root - 50) < 1e-8
    assert abs(loewe_residual(root, .5, .5, "LL.4", p, "LL.4", p)) < 1e-9
    assert strict_reference(.5, .5, "LL.4", np.array([1., 0., 100., 1.]), "LL.4", p)[1] == "not_increasing"


def test_projection_removes_additive_drug_effects():
    drugs = ["A", "B", "C", "D", "E"]
    rows = [(a, b) for i, a in enumerate(drugs) for b in drugs[i + 1:]]
    effect = dict(zip(drugs, [1., -2., 3., .5, 4.]))
    frame = pd.DataFrame({"fold": 0, "cell_line": "C1", "drug_a": [a for a, _ in rows], "drug_b": [b for _, b in rows]})
    additive = np.array([effect[a] + effect[b] for a, b in rows])
    interaction = np.zeros(len(rows))
    interaction[0] = 1.
    energy, residual = projected(frame, np.column_stack([additive, additive + interaction]), drugs)
    assert energy[0, 0] < 1e-20 and energy[0, 1] > 0 and np.abs(residual[:, 0]).max() < 1e-10


def test_holm_and_dose_regions():
    assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06])
    grid = pd.DataFrame([(1, a, b) for a in [1, 2, 3, 4] for b in [1, 2, 3, 4]], columns=["block_id", "dose_a", "dose_b"])
    region, level = regimes(grid, np.array([10., 30., 60., 80.] * 4))
    assert (region == "low").sum() == 4 and (region == "high").sum() == 4 and (region == "mixed").sum() == 8
    assert list(level[:4]) == ["R<25", "25-50", "50-75", "R>=75"]
