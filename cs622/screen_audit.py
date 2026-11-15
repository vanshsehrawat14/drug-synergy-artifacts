"""Within-screen error audit and clustered uncertainty for common-stratum gaps.

No screen outcomes are pooled. The baseline-relative denominator is reweighted
along with model SSE. Seed-average error and ensemble error are distinct.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import cross_screen as old
from .io import digest, write_json


def row_ess(count, target):
    """ESS for each row receiving target_s / count_s, normalized to sum one."""
    count,target=np.asarray(count,float),np.asarray(target,float)
    return float(target.sum()**2/np.sum(target**2/count))


def standardized_gain(n, model_sse, baseline_sse, target):
    n=np.asarray(n,float);target=np.asarray(target,float)
    model_sse=np.asarray(model_sse,float);baseline_sse=np.asarray(baseline_sse,float)
    active=target>0
    if np.any(n[active]<=0):return np.nan
    return float(1-np.sum(target[active]*model_sse[active]/n[active])/np.sum(target[active]*baseline_sse[active]/n[active]))


def panel_errors(t, errors):
    codes,keys=pd.factorize(pd.MultiIndex.from_frame(t[['pair_id','cell_line']]),sort=True)
    n=np.bincount(codes)
    m=np.column_stack([np.bincount(codes,weights=e)/n for e in errors.T])
    return float(np.mean(m*m)),float(np.mean(m.mean(1)**2)),len(keys)


def sufficient(t,e):
    f=pd.DataFrame(dict(pair=t.pair_id.to_numpy(),stratum=old.strata(t).to_numpy(),
                        f2=np.mean(e['f2']**2,axis=1),bliss=np.mean(e['bliss']**2,axis=1)))
    x=f.groupby(['pair','stratum'],observed=True).agg(n=('f2','size'),f2=('f2','sum'),bliss=('bliss','sum')).reset_index()
    drug=t.groupby('pair_id',observed=True)[['drug_a','drug_b']].first()
    return x,drug


def arrays(stats,common):
    pairs=np.sort(stats.pair.unique())
    mats=[stats.pivot(index='pair',columns='stratum',values=k).reindex(index=pairs,columns=common).fillna(0).to_numpy(float) for k in ['n','f2','bliss']]
    return pairs,mats


def endpoint(ref,screen):
    rn,rf,rb=ref;sn,sf,sb=screen
    w=rn/rn.sum()
    raw_gap=(1-sf.sum()/sb.sum())-(1-rf.sum()/rb.sum())
    std_gap=standardized_gain(sn,sf,sb,w)-standardized_gain(rn,rf,rb,w)
    return raw_gap,std_gap,abs(raw_gap)-abs(std_gap)


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    old.SEEDS[:]=[2026]
    allstats={};drugs={};metrics=[];coverage=[];sources=[]
    for name,sp in old.screens(a.oneil_root,a.other_root).items():
        t,e,tv,cov=old.build(name,sp)
        if not np.isfinite(t.inhibition).all() or any(not np.isfinite(x).all() for x in e.values()):raise ValueError('Nonfinite common rows')
        if not t.index.is_unique:raise ValueError('Duplicated observations')
        if (t.groupby('pair_id',observed=True).fold.nunique()>1).any():raise ValueError('Pair crosses test folds')
        coverage.append(cov)
        var=np.array([tv[f]['var'] for f in t.fold])
        for m,err in e.items():
            pc,ens,npan=panel_errors(t,err)
            metrics.append(dict(screen=name,model=m,seeds=err.shape[1],rows=len(t),pair_cells=npan,
                mse=float(np.mean(err**2)),rmse=float(np.sqrt(np.mean(err**2))),mae=float(np.mean(abs(err))),
                normalized_mse_train_var=float(np.mean(np.mean(err**2,axis=1)/var)),
                pair_cell_seed_average_mse=pc,pair_cell_ensemble_mse=ens,
                gain_vs_bliss=1-np.mean(err**2)/np.mean(e['bliss']**2),
                gain_vs_mean=1-np.mean(err**2)/np.mean(e['mean']**2),
                gain_vs_additive=1-np.mean(err**2)/np.mean(e['additive']**2)))
        allstats[name],drugs[name]=sufficient(t,e)
        sources.append(dict(screen=name,observations_sha256=digest(Path(sp['data'])/'observations.parquet'),
                            monotherapy_status=t.mono.value_counts().to_dict(),training_variance=tv))
        print(name,'rows',len(t),'pairs',len(drugs[name]),flush=True)
    pd.DataFrame(metrics).to_csv(out/'errors.csv',index=False)
    write_json(out/'coverage.json',coverage);write_json(out/'provenance.json',sources)
    rows=[];bootrows=[];loorows=[];stratarows=[]
    rng=np.random.default_rng(20261108)
    ref=allstats['ONEIL']
    for name,s in allstats.items():
        if name=='ONEIL':continue
        common=sorted(set(ref.stratum)&set(s.stratum))
        rp,ra=arrays(ref,common);sp,sa=arrays(s,common)
        rt=[x.sum(0) for x in ra];st=[x.sum(0) for x in sa];w=rt[0]/rt[0].sum()
        raw,std,reduction=endpoint(rt,st)
        # Raw all-row gap is retained separately from the common-support raw gap.
        raw_all=(1-s.f2.sum()/s.bliss.sum())-(1-ref.f2.sum()/ref.bliss.sum())
        row_weights=(sa[0]*(w/np.maximum(st[0],1))[None,:]).sum(1)
        boot=[]
        with threadpool_limits(limits=2):
            for b in range(a.draws):
                ri=rng.integers(len(rp),size=len(rp));si=rng.integers(len(sp),size=len(sp))
                rv=[x[ri].sum(0) for x in ra];sv=[x[si].sum(0) for x in sa]
                if rv[0].sum()==0 or sv[0].sum()==0:continue
                g=endpoint(rv,sv)
                boot.append(g)
        boot=np.asarray(boot);valid=np.isfinite(boot).all(1);bv=boot[valid]
        for j,(rg,sg,rd) in enumerate(boot):bootrows.append(dict(screen=name,draw=j,raw_gap=rg,standardized_gap=sg,absolute_gap_reduction=rd,valid=bool(valid[j])))
        rd=drugs['ONEIL'].reindex(rp);sd=drugs[name].reindex(sp)
        for origin,ps,mats,dd,other in [('ONEIL',rp,ra,rd,st),(name,sp,sa,sd,rt)]:
            for drug in sorted(set(dd.drug_a)|set(dd.drug_b)):
                keep=((dd.drug_a!=drug)&(dd.drug_b!=drug)).to_numpy()
                removed=[x[keep].sum(0) for x in mats]
                g=endpoint(removed,other) if origin=='ONEIL' else endpoint(other,removed)
                loorows.append(dict(screen=name,deleted_from=origin,drug=drug,raw_gap=g[0],standardized_gap=g[1],absolute_gap_reduction=g[2]))
        lo,hi=np.quantile(bv,[.025,.975],axis=0) if len(bv) else (np.full(3,np.nan),np.full(3,np.nan))
        rows.append(dict(screen=name,common_strata=len(common),oneil_mass_covered=rt[0].sum()/ref.n.sum(),
                         screen_mass_covered=st[0].sum()/s.n.sum(),row_ess=row_ess(st[0],w),
                         pair_weight_ess=float(row_weights.sum()**2/(row_weights@row_weights)),
                         min_pairs_in_stratum=int(s[s.stratum.isin(common)].groupby('stratum').pair.nunique().min()),
                         raw_all_row_gap=raw_all,raw_common_gap=raw,standardized_gap=std,absolute_gap_reduction=reduction,
                         raw_gap_pair_low=lo[0],raw_gap_pair_high=hi[0],std_gap_pair_low=lo[1],std_gap_pair_high=hi[1],
                         reduction_pair_low=lo[2],reduction_pair_high=hi[2],valid_bootstrap_draws=int(valid.sum()),
                         invalid_bootstrap_draws=int((~valid).sum()),frozen_reduction_criterion_met=bool(reduction>0 and lo[2]>0)))
        for j,key in enumerate(common):
            stratarows.append(dict(screen=name,stratum=key,target_mass=w[j],rows=int(st[0][j]),row_weight=w[j]/st[0][j],
                                   pair_count=int(s[s.stratum==key].pair.nunique()),model_mse=st[1][j]/st[0][j],bliss_mse=st[2][j]/st[0][j]))
    for filename,data in [('standardization.csv',rows),('bootstrap.csv',bootrows),('drug_deletion.csv',loorows),('strata.csv',stratarows)]:pd.DataFrame(data).to_csv(out/filename,index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    # Existing additional seeds: only complete families, without substituting models.
    old.SEEDS[:]=[2026,2027,2028]
    sensitivity=[]
    for name,sp in old.screens(a.oneil_root,a.other_root).items():
        t,e,_,_=old.build(name,sp)
        for m,err in e.items():
            if err.shape[1]<2:continue
            pc,ens,npan=panel_errors(t,err)
            sensitivity.append(dict(screen=name,model=m,seeds=err.shape[1],pair_cells=npan,
                                    pair_cell_seed_average_mse=pc,pair_cell_ensemble_mse=ens))
    pd.DataFrame(sensitivity).to_csv(out/'seed_vs_ensemble.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--oneil-root',default='.')
    p.add_argument('--other-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--draws',type=int,default=2000);run(p.parse_args())
