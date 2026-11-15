import numpy as np
import pandas as pd

from cs622.loewe_audit import strict_reference
from cs622.loewe_verify import admissible, theta_root


def frame(rows):
    cols = ["dose_a", "dose_b", "model_a", "a_b", "a_c", "a_d", "a_e", "model_b", "b_b", "b_c", "b_d", "b_e"]
    return pd.DataFrame(rows, columns=cols)


def test_forward_root_matches_inverse_root_and_scorer_equation():
    f = frame([[1.0, 2.0, "LL.4", -1.5, 0., 90., 3., "LL.4", -2., 5., 80., 4.],
               [0.3, 0.1, "L.4", -1.2, -2., 100., np.log(2.), "LL.4", -0.8, 0., 70., 1.]])
    effect, status = theta_root(f)
    assert list(status) == ["root_increasing", "root_increasing"]
    for y, r in zip(effect, f.itertuples()):
        strict, _ = strict_reference(r.dose_a, r.dose_b, r.model_a, np.array([r.a_b, r.a_c, r.a_d, r.a_e]),
                                     r.model_b, np.array([r.b_b, r.b_c, r.b_d, r.b_e]))
        assert abs(y - strict) < 1e-9
    residual, ok = admissible(f, effect)
    assert ok.all() and np.abs(residual).max() < 1e-9


def test_no_root_and_opposite_curves_have_no_expectation():
    # drug b alone at its dose already exceeds drug a's maximum: no Loewe root
    f = frame([[1.0, 50.0, "LL.4", -1., 0., 20., 5., "LL.4", -3., 0., 95., 1.],
               [1.0, 1.0, "LL.4", -1., 0., 50., 1., "LL.4", 1., 0., 50., 1.]])
    effect, status = theta_root(f)
    assert list(status) == ["no_root", "opposite_directions"]
    assert np.isnan(effect).all()
