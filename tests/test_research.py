"""Scientific invariants for the resolution and numerical-reliability analyses."""
import numpy as np
import pandas as pd

from cs622.features import FeatureBuilder
from cs622.loewe_priority import certificate, rank_metrics
from cs622.loewe_verify import theta_root
from cs622.resolution import residualize
from cs622.research_jobs import job


def test_weighted_orthogonal_decomposition():
    rng=np.random.default_rng(8)
    h=rng.normal(size=(30,4));h[:,0]=1
    z=rng.normal(size=(30,3));w=rng.integers(1,9,size=30)
    r=residualize(h,z,w);fit=z-r
    np.testing.assert_allclose(h.T@(w[:,None]*r),0,atol=1e-12)
    np.testing.assert_allclose((w[:,None]*z*z).sum(0),(w[:,None]*(fit*fit+r*r)).sum(0),atol=1e-12)


def test_identical_drug_loewe_equation_and_certificate():
    f=pd.DataFrame({"model_a":["LL.4"],"model_b":["LL.4"],"dose_a":[1.],"dose_b":[2.]})
    for arm in "ab":
        for k,v in zip("bcde",[-1.,0.,100.,1.]): f[f"{arm}_{k}"]=v
    root,status=theta_root(f)
    np.testing.assert_allclose(root,[75.],atol=1e-10)
    r,ok=certificate(f,root)
    assert ok.all() and abs(r[0])<1e-10
    assert not certificate(f,np.array([100.]))[1].any()
    assert not certificate(f,np.array([80.]))[1].any()


def test_ranking_turnover_denominator():
    a=np.arange(100.);b=a.copy();b[0]=200
    assert rank_metrics(a,b)["turnover"] == 1-4/5
    assert rank_metrics(a,a)["retention"] == 1


def test_high_precision_boundary_root_survives_float_rounding():
    from cs622.loewe_precision import precise_root
    f=pd.DataFrame({"model_a":["LL.4"],"model_b":["LL.4"],"dose_a":[1e20],"dose_b":[2e20]})
    for arm in "ab":
        for k,v in zip("bcde",[-1.,0.,100.,1.]):f[f"{arm}_{k}"]=v
    answer=precise_root(f.iloc[0])
    assert answer["certified"] and answer["boundary_distance"]>0
    assert answer["effect"]==100. and not certificate(f,np.array([answer["effect"]]))[1][0]
    import mpmath as mp
    with mp.workdps(100):
        actual=mp.mpf(answer["effect_decimal"])
        expected=100*(mp.mpf(1e20)+mp.mpf(2e20))/(1+mp.mpf(1e20)+mp.mpf(2e20))
        assert abs(actual-expected)<mp.mpf("1e-50")
    f["b_b"]=1.
    assert not precise_root(f.iloc[0])["certified"]


def test_error_triage_ties_are_outcome_independent():
    from cs622.reliability import top_mass
    e=np.array([100.,2.,3.,4.,5.,6.,7.,8.,9.,10.])
    assert np.isclose(top_mass(e,np.ones(10)),.1)
    score=np.array([1.,1.,0.,0.,0.,0.,0.,0.,0.,0.])
    assert np.isclose(top_mass(e,score),.5*102/e.sum())


def test_dose_error_calibrator_never_uses_test_outcomes():
    from cs622.reliability import calibrator
    f=pd.DataFrame(dict(cell_line=["A","A","B","B"],mean_logdose=[-2.,-1.,0.,1.],
                        gap_logdose=[0.,1.,2.,1.],reference=[10.,20.,30.,40.],error=[1.,4.,9.,16.]))
    test=f.copy();a=calibrator(f,test,"dose_bliss");test.error=1e9
    np.testing.assert_array_equal(a,calibrator(f,test,"dose_bliss"))
    assert (a>0).all()


def test_masks_preserve_lookup_but_remove_direct_identity():
    f=pd.DataFrame({"drug_a":["A","A","B","B"],"drug_b":["B","B","A","A"],
                    "cell_line":["C","D","C","D"],"dose_a":[1.,2.,1.,2.],"dose_b":[2.,1.,2.,1.],
                    "unit_a":["uM"]*4,"unit_b":["uM"]*4})
    # Directly seed the train-only lookup to avoid introducing outcomes to transform.
    b=FeatureBuilder("F2",drug_identity=False,cell_identity=False)
    b.drugs=["A","B"];b.cells=["C","D"]
    b.mono={(d,c,x):10*x for d in b.drugs for c in b.cells for x in [1.,2.]};b.mono_mean=15.
    from cs622.features import Standardizer
    b.endpoint_scaler=Standardizer().fit(np.vstack([b._continuous(f,e) for e in "ab"]))
    a,c,cell=b.transform(f)
    assert cell.shape==(4,0) and a.shape==(4,4)
    assert b.monotherapy_values(f,"a").tolist()==[10.,20.,10.,20.]
    restored=FeatureBuilder.from_state_dict(b.state_dict())
    for old,new in zip((a,c,cell),restored.transform(f)):np.testing.assert_array_equal(old,new)


def test_frozen_job_coverage_and_control_settings():
    jobs=[job(i) for i in range(45)]
    assert len({(a,b,c) for a,b,c,_ in jobs})==45
    for _,_,_,c in jobs:
        assert c["training"]["epochs"]==200 and c["training"]["patience"]==20
        assert c["model"]["hidden"]==[128,64]


def test_runner_applies_each_mask(tmp_path):
    from cs622.smoke import smoke
    from cs622.runner import run
    from cs622.io import read_json
    smoke(tmp_path/"fixture")
    for task in [0,15,30]:
        mask,_,_,config=job(task)
        config["dataset"]=str(tmp_path/"fixture/prepared")
        config["training"].update(device="cpu",epochs=1,patience=1,threads=1)
        config["bootstrap"]=2
        run(config,tmp_path/mask)
        features=read_json(tmp_path/mask/"features.json")
        assert features["drug_identity"] == (task!=0)
        assert features["cell_identity"] == (task!=15)
        assert features["dose_input"] == (task!=30)
