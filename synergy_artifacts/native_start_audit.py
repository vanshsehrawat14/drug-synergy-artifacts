"""Prepare a fixed-fit native starting-point intervention and audit its reconstruction."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .io import digest,write_json


def run(a):
    out=Path(a.output)
    c=pd.read_parquet(a.cases);q=pd.read_parquet(a.certificates).set_index('case_id').reindex(c.case_id)
    has=q.cert_status.to_numpy()=='certified_root'
    distance=np.minimum(q.dist_lower.to_numpy(float),q.dist_upper.to_numpy(float))
    strata=np.where(has,np.where(distance<.01,'boundary','interior'),q.cert_status.to_numpy())
    strata=np.char.add(np.where(c.fallback.to_numpy(),'fallback:','accepted:'),strata.astype(str))
    rng=np.random.default_rng(20261108)
    ids=np.concatenate([rng.choice(np.flatnonzero(strata==s),size=min(50,int((strata==s).sum())),replace=False) for s in sorted(set(strata))])
    sample=c.iloc[ids].copy();sample['stratum']=strata[ids]
    if a.prepare:
        out.mkdir(parents=True,exist_ok=False)
        sample.to_csv(out/'cases.csv',index=False)
        write_json(out/'inputs.json',dict(cases_sha256=digest(a.cases),source_sha256=digest(Path(a.source)/'R/calculate_synergy_score.R'),
                   sample_seed=20261108,strata=sample.stratum.value_counts().to_dict(),case_input_sha256=digest(out/'cases.csv')))
        print('prepared',len(sample),'cases');return
    raw=pd.read_csv(out/'native/starts.csv').merge(sample[['case_id','stratum','native_code','native_below_3']],on='case_id',validate='many_to_one')
    original=raw[raw.start_rule=='original'].set_index('case_id')
    match_code=original.termcd.to_numpy()==original.native_code.to_numpy()
    reference_error=np.abs(original.reference-original.native_below_3)
    if not match_code.all() or reference_error.max()>1e-8:
        raise AssertionError(f'Original source reconstruction failed: code matches {match_code.sum()}/{len(match_code)}, reference max error {reference_error.max()}')
    raw['accepted']=raw.termcd<3
    cq=q.copy();cq.index=c.case_id
    raw['cert_status']=raw.case_id.map(cq.cert_status)
    raw['certified_reference']=np.minimum(raw.case_id.map(cq.root).to_numpy(float),100)
    raw['accepted_at_certified_root']=raw.accepted & (raw.cert_status=='certified_root') & (abs(raw.reference-raw.certified_reference)<=1e-5)
    raw['acceptance_changed']=raw.accepted.to_numpy()!=raw.case_id.map(original.termcd < 3).to_numpy()
    raw['reference_shift']=raw.reference.to_numpy()-raw.case_id.map(original.reference).to_numpy()
    raw.to_csv(out/'interventions.csv',index=False)
    tab=raw.groupby(['stratum','start_rule']).agg(cases=('case_id','size'),accepted=('accepted','sum'),
        acceptance_changes=('acceptance_changed','sum'),accepted_at_certified_root=('accepted_at_certified_root','sum'),reference_changes=('reference_shift',lambda x:int((abs(x)>1e-6).sum())),
        max_abs_reference_shift=('reference_shift',lambda x:float(abs(x).max()))).reset_index()
    tab.to_csv(out/'summary.csv',index=False)
    write_json(out/'reconstruction.json',dict(cases=len(original),native_code_mismatches=int((~match_code).sum()),
        native_reference_max_abs_error=float(reference_error.max()),
        affected_cases_any_start=int(raw[raw.acceptance_changed].case_id.nunique()),
        valid_root_cases_acceptance_changed=int(raw[raw.acceptance_changed & (raw.cert_status=='certified_root')].case_id.nunique()),
        large_reference_change_cases_any_start=int(raw[abs(raw.reference_shift)>1e-6].case_id.nunique()),
        interpretation='Stratified fixed-fit numerical intervention, not population incidence or biological evidence.'))
    print(tab.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true')
    p.add_argument('--cases',default='results/neural/runs/scientific_native_full/surface_cases.parquet')
    p.add_argument('--certificates',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
