"""Input-support and pair Bayesian-bootstrap sensitivities, declared in an amendment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from . import cross_screen as old
from .screen_audit import sufficient,arrays,endpoint,row_ess
from .io import write_json


def minimal(sp):
    cols=['observation_id','is_combination','inhibition','pair_id','drug_a','drug_b','cell_line','dose_a','dose_b']
    t=pd.read_parquet(Path(sp['data'])/'observations.parquet',columns=cols)
    t=t[t.is_combination].set_index('observation_id');f=old.load_folds(sp['f2'],seeded=True)
    b=sp['bliss'];bp,bcol,seeded=(b[0],b[1],True) if isinstance(b,tuple) else (b,'prediction',False)
    bl=old.load_folds(bp,bcol,seeded)
    if isinstance(bl,pd.DataFrame):bl=bl.iloc[:,0]
    paths=[bp/f'seed_2026/fold_{k}/predictions.parquet' if seeded else bp/f'fold_{k}/predictions.parquet' for k in range(5)]
    mono=pd.concat([pd.read_parquet(p,columns=['observation_id','monotherapy_status']) for p in paths]).set_index('observation_id').monotherapy_status
    if not t.index.is_unique or not f.index.is_unique or not bl.index.is_unique:raise ValueError('Duplicated ID')
    if set(t.index)!=set(f.index) or set(t.index)!=set(bl.index):raise ValueError('Incomplete prediction coverage')
    t['bliss_ref']=bl.reindex(t.index);t['mono']=mono.reindex(t.index).astype(str)
    e={'f2':t.inhibition.to_numpy()[:,None]-f.loc[t.index].to_numpy(),
       'bliss':(t.inhibition-t.bliss_ref).to_numpy()[:,None]}
    return sufficient(t,e)


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);old.SEEDS[:]=[2026]
    stats={};drugs={}
    for name,sp in old.screens(a.oneil_root,a.other_root).items():
        stats[name],drugs[name]=minimal(sp)
        stats[name].to_parquet(out/f'{name}_pair_stratum.parquet',index=False)
        drugs[name].to_csv(out/f'{name}_pair_drugs.csv')
        print('cached',name,flush=True)
    rows=[];loo=[]
    ref=stats['ONEIL'];rng=np.random.default_rng(20261108)
    for name,s in stats.items():
        if name=='ONEIL':continue
        common=sorted(set(ref.stratum)&set(s.stratum))
        rc=ref.groupby('stratum').pair.nunique();sc=s.groupby('stratum').pair.nunique()
        stable=[j for j in common if rc[j]>=10 and sc[j]>=10]
        for method,keys in [('stable_support_pair_bootstrap',stable),('full_support_pair_bayesian_bootstrap',common)]:
            rp,ra=arrays(ref,keys);sp,sa=arrays(s,keys)
            rt=[x.sum(0) for x in ra];st=[x.sum(0) for x in sa]
            point=endpoint(rt,st);boot=[]
            with threadpool_limits(limits=2):
                for _ in range(a.draws):
                    if method.startswith('full'):
                        rw=rng.exponential(size=len(rp));sw=rng.exponential(size=len(sp))
                    else:
                        rw=np.bincount(rng.integers(len(rp),size=len(rp)),minlength=len(rp))
                        sw=np.bincount(rng.integers(len(sp),size=len(sp)),minlength=len(sp))
                    boot.append(endpoint([rw@x for x in ra],[sw@x for x in sa]))
            boot=np.asarray(boot);valid=np.isfinite(boot).all(1);lo,hi=np.quantile(boot[valid],[.025,.975],axis=0)
            rows.append(dict(screen=name,method=method,strata=len(keys),oneil_mass=rt[0].sum()/ref.n.sum(),screen_mass=st[0].sum()/s.n.sum(),
                raw_gap=point[0],standardized_gap=point[1],absolute_gap_reduction=point[2],
                reduction_low=lo[2],reduction_high=hi[2],valid_draws=int(valid.sum()),invalid_draws=int((~valid).sum()),
                row_ess=row_ess(st[0],rt[0]/rt[0].sum())))
            for origin,pairs,mats in [('ONEIL',rp,ra),(name,sp,sa)]:
                d=drugs[origin].reindex(pairs)
                for drug in sorted(set(d.drug_a)|set(d.drug_b)):
                    keep=((d.drug_a!=drug)&(d.drug_b!=drug)).to_numpy()
                    removed=[x[keep].sum(0) for x in mats]
                    g=endpoint(removed,st) if origin=='ONEIL' else endpoint(rt,removed)
                    loo.append(dict(screen=name,method=method,deleted_from=origin,drug=drug,absolute_gap_reduction=g[2]))
    pd.DataFrame(rows).to_csv(out/'sensitivities.csv',index=False)
    pd.DataFrame(loo).to_csv(out/'drug_deletion.csv',index=False)
    write_json(out/'interpretation.json',dict(primary='Frozen full-support point estimate unchanged. Stable-support sensitivity changes target population and is exploratory.',
        intervals='Stable-support nonparametric pair-bootstrap confidence intervals; full-support exponential pair Bayesian-bootstrap percentile intervals. No dose-level iid inference.'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--oneil-root',default='.')
    p.add_argument('--other-root',required=True);p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=2000)
    run(p.parse_args())
