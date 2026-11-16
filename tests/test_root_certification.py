import numpy as np
from mpmath import mp

from synergy_artifacts.decision_sensitivity import compare, guaranteed, selected
from synergy_artifacts.loewe_interval import certify


def case(b=(-1.0, -1.0), c=(0.0, 0.0), d=(100.0, 100.0), e=(1.0, 2.0), dose=(1.0, 1.0), model=("LL.4", "LL.4")):
    r = {"case_id": "x", "dose_a": dose[0], "dose_b": dose[1], "model_a": model[0], "model_b": model[1]}
    for i, arm in enumerate("ab"):
        r.update({f"{arm}_b": b[i], f"{arm}_c": c[i], f"{arm}_d": d[i], f"{arm}_e": e[i]})
    return r


def loewe_root(r):
    """Reference by direct high-precision bisection in effect coordinate."""
    mp.dps = 50
    lo, hi = mp.mpf(1e-30), mp.mpf(100) - mp.mpf(1e-30)
    def gg(y):
        s = mp.mpf(-1)
        for arm in "ab":
            b, c, d, e = (mp.mpf(r[f"{arm}_{k}"]) for k in "bcde")
            s += mp.mpf(r[f"dose_{arm}"]) / (e * ((d - y) / (y - c)) ** (1 / b))
        return s
    for _ in range(300):
        m = (lo + hi) / 2
        if gg(m) > 0:
            lo = m
        else:
            hi = m
    return float((lo + hi) / 2)


def test_certified_root_matches_direct_solution_and_brackets_it():
    r = case()
    out = certify(r, None)
    assert out["cert_status"] == "certified_root"
    assert abs(out["root"] - loewe_root(r)) < 1e-9
    assert float(out["bracket_lo"]) <= out["root"] <= float(out["bracket_hi"])


def test_hint_far_from_root_falls_back_to_bisection():
    out = certify(case(), 99.0)
    assert out["cert_status"] == "certified_root" and out["hint_source"] == "bisection"


def test_symmetric_arm_order():
    r = case(e=(1.0, 3.0), dose=(0.5, 2.0), b=(-1.5, -0.8))
    s = case(e=(3.0, 1.0), dose=(2.0, 0.5), b=(-0.8, -1.5))
    a, b = certify(r, None), certify(s, None)
    assert a["cert_status"] == b["cert_status"] == "certified_root"
    assert abs(a["root"] - b["root"]) < 1e-12


def test_opposite_and_empty_and_invalid_are_undefined():
    assert certify(case(b=(-1.0, 1.0)), None)["cert_status"] == "undefined_opposite_or_flat"
    assert certify(case(c=(0.0, 60.0), d=(50.0, 100.0)), None)["cert_status"] == "undefined_empty_interval"
    assert certify(case(e=(-1.0, 1.0)), None)["cert_status"] == "invalid_curve"


def test_no_root_when_strong_drug_alone_exceeds_weak_plateau():
    # weak plateau 20; strong drug at its dose alone already exceeds 20, so no interior root
    r = case(d=(20.0, 100.0), e=(1.0, 0.01), dose=(0.001, 1.0))
    assert certify(r, None)["cert_status"] == "certified_no_root"


def test_l4_uses_natural_log_dose():
    r = case(model=("L.4", "L.4"), e=(np.log(1.0), np.log(2.0)))
    q = case(model=("LL.4", "LL.4"), e=(1.0, 2.0))
    assert abs(certify(r, None)["root"] - certify(q, None)["root"]) < 1e-9


def test_selection_tie_rule_and_metrics():
    a = np.array([3.0, 2.0, 2.0, 1.0])
    m, n = selected(a, 0.5)
    assert n == 2 and m.tolist() == [True, True, False, False]
    r = compare(a, np.array([1.0, 2.0, 3.0, 0.0]), 0.5)
    assert r["replaced"] == 1 and r["entered"] == 1 and abs(r["jaccard"] - 1 / 3) < 1e-12


def test_guaranteed_changes_respect_intervals():
    native = np.array([4.0, 3.0, 2.0, 1.0])
    lo = np.array([0.0, 3.0, 2.5, 1.0])
    hi = np.array([0.1, 3.0, 2.5, 1.0])
    g = guaranteed(native, lo, hi, 0.5)
    assert g["guaranteed_exits"] == 1 and g["guaranteed_entries"] == 1
    hi2 = hi.copy(); hi2[0] = 2.75
    g2 = guaranteed(native, lo, hi2, 0.5)
    assert g2["ambiguous_native_selected"] == 1
