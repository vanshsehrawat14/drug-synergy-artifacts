"""Scientific invariants for certificate replay, nested targets and matched designs."""
import numpy as np
import pandas as pd
import pytest

from synergy_artifacts.certificate_audit import bracket_structure,replay
from synergy_artifacts.loewe_interval import certify
from synergy_artifacts.information_audit import remove_reference_span
from synergy_artifacts.mechanism_analysis import linear_effect
from synergy_artifacts.mechanism_jobs import jobs,config_for,trigger_audit
from synergy_artifacts.resolution import projections
from synergy_artifacts.screen_audit import row_ess,standardized_gain,panel_errors,endpoint


def test_extreme_boundary_certificate_is_replayable_after_serialization():
    case=dict(case_id='boundary',dose_a=1e80,dose_b=2e80,model_a='LL.4',model_b='LL.4')
    for arm in 'ab':
        case.update({f'{arm}_{k}':v for k,v in zip('bcde',[-1.,0.,100.,1.])})
    cert=certify(case,dps=100)
    assert cert['cert_status']=='certified_root'
    assert cert['root']==100. and cert['dist_upper']>0
    assert bracket_structure(cert) and replay(case,cert)
    from mpmath import mp
    with mp.workdps(300):
        legacy={**cert,'bracket_lo':mp.nstr(mp.mpf(cert['bracket_lo']),40),'bracket_hi':mp.nstr(mp.mpf(cert['bracket_hi']),40)}
    assert not bracket_structure(legacy)


def test_guaranteed_selection_respects_canonical_cutoff_ties():
    from synergy_artifacts.decision_sensitivity import guaranteed
    g=guaranteed(np.array([1.,1.]),np.array([1.,1.]),np.array([1.,1.]),.5)
    assert g['guaranteed_exits']==g['guaranteed_entries']==0
    assert g['ambiguous_native_selected']==g['ambiguous_native_unselected']==0


def test_reference_projection_is_nested_weighted_and_outcome_independent():
    rng=np.random.default_rng(9);n=40
    frame=pd.DataFrame(dict(drug_a=['a']*n,drug_b=['b']*n,fold=[0]*n,cell_line=['c']*n,n=rng.integers(1,6,n)))
    truth=rng.normal(size=n);reference=rng.normal(size=n);other=rng.normal(size=n)
    # Columns represent truth and model errors; truth - reference-error = reference.
    c=np.column_stack([truth,truth-reference,other])
    q=remove_reference_span(frame,c,[1])
    w=frame.n.to_numpy()
    np.testing.assert_allclose(reference@(w[:,None]*q),0,atol=1e-12)
    assert np.all((w[:,None]*q*q).sum(0)<=(w[:,None]*c*c).sum(0)+1e-12)
    np.testing.assert_allclose(q[:,0],q[:,1],atol=1e-12)
    # Changing the truth/model-error columns together leaves the nuisance span fixed.
    changed=c.copy();changed[:,0]+=2;changed[:,1]+=2
    q2=remove_reference_span(frame,changed,[1])
    np.testing.assert_allclose(q2[:,2],q[:,2],atol=1e-12)


def test_noninteracting_bliss_has_nonzero_original_dose_contrast():
    import itertools
    rows=[];y=[];curves={d:np.array([0.,20.,60.,90.])*(1-.1*j) for j,d in enumerate('ABCD')}
    for panel,(a,b) in enumerate(itertools.combinations('ABCD',2)):
        for ra,rb in itertools.product(range(4),repeat=2):
            rows.append(dict(drug_a=a,drug_b=b,panel=panel,pair=panel,fold=0,cell_line='C',ra=ra,rb=rb,n=1))
            x,z=curves[a][ra],curves[b][rb];y.append(x+z-x*z/100)
    frame=pd.DataFrame(rows);v=np.array(y)
    _,_,_,c=projections(frame,np.column_stack([v,np.zeros(len(v))]))
    assert np.linalg.norm(c[:,0])>1
    # Truth is exactly Bliss; a reference-augmented evaluation has zero remaining truth.
    q=remove_reference_span(frame,c,[1])
    np.testing.assert_allclose(q[:,0],0,atol=1e-10)


def test_seed_average_panel_error_is_distinct_from_ensemble():
    t=pd.DataFrame(dict(pair_id=['A','B'],cell_line=['C','C']))
    seed,ensemble,n=panel_errors(t,np.array([[1.,-1.],[1.,-1.]]))
    assert seed==1 and ensemble==0 and n==2


def test_reweighting_ess_is_over_rows_and_denominators_are_reweighted():
    assert row_ess([50,50],[.5,.5])==100
    assert np.isclose(row_ess([90,10],[.5,.5]),36)
    g=standardized_gain([10,10],[10,90],[100,100],[.9,.1])
    assert np.isclose(g,.82)
    assert np.isnan(standardized_gain([0,10],[0,90],[0,100],[.9,.1]))
    a=[np.array([10.,20.]),np.array([10.,20.]),np.array([20.,40.])]
    raw,std,d=endpoint(a,a)
    assert raw==0 and std==0 and d==0


def test_frozen_task_design_has_matched_2029_control_and_no_search():
    j=jobs();assert len(j)==50 and len({x['output'] for x in j})==50
    seed=[x for x in j if x['campaign']=='seed2029']
    assert len(seed)==20 and {x['seed'] for x in seed}=={2029}
    for arm in ['full','no_drug_identity','no_cell_identity','no_dose']:
        assert {x['fold'] for x in seed if x['arm']==arm}==set(range(5))
    for x in j:
        assert x['config']['model']==config_for('full',2026,0)['model']
        assert x['config']['training']==config_for('full',2026,0)['training']
    assert config_for('residual_no_drug',2026,0)['features']==dict(regime='F2',mono_missing='interpolate',drug_identity=False,reference_offset='bliss')


def test_signed_factorial_effect_and_matching():
    errors=np.array([[10.,12.,14.,20.],[20.,24.,28.,40.]])
    num,den=linear_effect(errors,[(1,[3]),(-1,[1]),(-1,[2]),(1,[0])],[0])
    np.testing.assert_array_equal(num,[4.,8.]);assert num.sum()/den.sum()==.4


def test_seed_trigger_does_not_add_provisional_response_endpoint():
    row=dict(model='no_drug_identity',level='response',change=.0508,pair_low=.0367,pair_high=.066,drug_low=.0177,drug_high=.084,
             fold_min=.0196,fold_max=.0748,confirmatory_seed_trigger=False)
    assert not trigger_audit(pd.DataFrame([row]))[0]['trigger']
    row.update(level='panel_contrast',change=-.0554,pair_low=-.09,pair_high=-.0186,drug_low=-.1383,drug_high=.0275,
               fold_min=-.102,fold_max=.00418,confirmatory_seed_trigger=True)
    assert trigger_audit(pd.DataFrame([row]))[0]['trigger']


def test_new_training_arms_on_synthetic_data(tmp_path):
    from synergy_artifacts.smoke import smoke
    from synergy_artifacts.runner import run
    from synergy_artifacts.io import read_json
    smoke(tmp_path/'fixture')
    for arm in ['joint_identity','residual_no_drug']:
        c=config_for(arm,2026,0);c['dataset']=str(tmp_path/'fixture/prepared')
        c['training'].update(device='cpu',epochs=1,patience=1,threads=1);c['bootstrap']=2
        run(c,tmp_path/arm)
        f=read_json(tmp_path/arm/'features.json')
        assert not f['drug_identity'] and f['cell_identity']==(arm!='joint_identity')
        p=pd.read_parquet(tmp_path/arm/'predictions.parquet')
        assert np.isfinite(p.prediction).all()
        assert ('reference_offset' in p)==(arm=='residual_no_drug')


def test_return_collector_preserves_task_accounting_hashes_and_existing_files(tmp_path):
    import hashlib,json,tarfile
    from experiments.mechanism_analysis.collect_artifacts import collect
    repo=tmp_path/'repo';d=repo/'results/mechanism_analysis/runs'/jobs()[0]['output'];d.mkdir(parents=True)
    (d/'config.json').write_text(json.dumps(jobs()[0]['config']))
    (d/'metadata.json').write_text('{}');(d/'predictions.parquet').write_bytes(b'fixture')
    archive=tmp_path/'return.tar.gz';manifest=collect(repo,archive)
    assert manifest['complete_tasks']==1 and len(manifest['tasks'])==50
    with tarfile.open(archive) as tar:
        saved=json.load(tar.extractfile('artifact_manifest.json'))
        for f in saved['files']:
            assert hashlib.sha256(tar.extractfile(f['path']).read()).hexdigest()==f['sha256']
    with pytest.raises(FileExistsError):collect(repo,archive)
