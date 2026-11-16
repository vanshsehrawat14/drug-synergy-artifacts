"""Analyze returned frozen mechanism jobs with matched controls and grouped errors.

Training and analysis are separate. Every requested run must be complete before
an endpoint is evaluated. Seed 2029 response is exploratory for drug identity.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student
from threadpoolctl import threadpool_limits

from .ablation_summary import unit_group,MASKS
from .information_audit import remove_reference_span,pair_energy
from .io import read_json,write_json,digest
from .mechanism_jobs import config_for,check_controls
from .resolution import means,projections,grouped_intervals
from .neural_report import index,load_truth,out_of_fold,jackknife_interval
from .features import FeatureBuilder
from .regimes import holm


def linear_effect(errors,terms,denominator):
    """Signed factorial error contrast / matched control; input is pair x model."""
    diff=sum(weight*errors[:,indices].mean(1) for weight,indices in terms)
    den=errors[:,denominator].mean(1)
    return diff,den


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    check_controls(a.campaign,a.baseline_root,a.ablations)
    root=Path(a.runs);base=Path(a.baseline_root);resolution=Path(a.resolution)
    df,sha,_=load_truth('data/processed/oneil_v1');t=index(df)
    meta=read_json(resolution/'inputs.json');frame=pd.read_parquet(resolution/'coordinates.parquet')
    if sha!=meta['dataset_sha256']:raise ValueError('Truth cache mismatch')
    group=unit_group(df,t);n=np.bincount(group);yg=means(group,t['y'],n)
    names=['truth'];families=['truth'];seeds=[-1];v=[yg];row=[np.zeros(len(t['pairs']))];sources=[]

    def add(family,seed,folder,arm,offset=False):
        paths={f:folder/f'seed_{seed}'/f'fold_{f}' for f in range(5)}
        for f,p in paths.items():
            expected=config_for(arm,seed,f)
            if offset:expected['features']['reference_offset']='bliss'
            if read_json(p/'config.json')!=expected:raise ValueError(f'Config mismatch {p}')
            m=read_json(p/'metadata.json')
            if m['dataset_sha256']!=sha:raise ValueError('Returned dataset hash differs')
            sources.append(dict(family=family,seed=seed,fold=f,predictions_sha256=digest(p/'predictions.parquet')))
        p=out_of_fold(paths,t);names.append(f'{family}:{seed}');families.append(family);seeds.append(seed)
        v.append(means(group,p,n));row.append(np.bincount(t['pair'],weights=(t['y']-p)**2))
    if a.campaign=='seed2029':
        for family in ['full']+MASKS:
            for seed in [2026,2027,2028,2029]:
                folder=(root/'seed2029'/family if seed==2029 else
                        base/'results/neural/runs/benchmark_v1/runs/f2_base' if family=='full' else Path(a.ablations)/family)
                add(family,seed,folder,family)
    elif a.campaign=='joint_identity':
        for family in ['full','no_drug_identity','no_cell_identity','joint_identity']:
            for seed in [2026,2027,2028]:
                folder=(root/'joint_identity/joint_identity' if family=='joint_identity' else
                        base/'results/neural/runs/benchmark_v1/runs/f2_base' if family=='full' else Path(a.ablations)/family)
                add(family,seed,folder,family)
    else:
        for family,arm in [('full_residual','full'),('residual_no_drug','residual_no_drug')]:
            for seed in [2026,2027,2028]:
                folder=base/'results/neural/runs/benchmark_v1/runs/f2_residual' if arm=='full' else root/'residual_drug_identity/residual_no_drug'
                add(family,seed,folder,arm,offset=True)
    ref_indices=[]
    for reference in ['bliss','hsa']:
        p=np.zeros(len(df))
        for fold in range(5):
            b=FeatureBuilder.from_state_dict(read_json(base/f'results/neural/runs/benchmark_v1/runs/f2_base/seed_2026/fold_{fold}/features.json'))
            m=df.fold.to_numpy()==fold;p[m]=b.reference(df[m].drop(columns=['inhibition']),reference)
        ref_indices.append(len(v));v.append(means(group,p,n));names.append(reference);families.append(reference);seeds.append(-1)
        row.append(np.bincount(t['pair'],weights=(t['y']-p)**2))
    v=np.column_stack(v);z=v[:,:1]-v;z[:,0]=v[:,0];row=np.column_stack(row)
    fam=np.array(families);seed=np.array(seeds)
    with threadpool_limits(limits=a.threads):
        _,_,rp,rw=projections(frame,z)
        arrays={'response':row,'panel_contrast':pair_energy(frame,rp),'dose_contrast':pair_energy(frame,rw),
                'panel_reference_residual':pair_energy(frame,remove_reference_span(frame,rp,ref_indices)),
                'dose_reference_residual':pair_energy(frame,remove_reference_span(frame,rw,ref_indices))}
        leave={k:[] for k in arrays if k!='response'}
        for d in meta['drugs']:
            ep,ew,p,w=projections(frame,z,d)
            leave['panel_contrast'].append(ep);leave['dose_contrast'].append(ew)
            for key,x in [('panel_reference_residual',p),('dose_reference_residual',w)]:
                q=remove_reference_span(frame,x,ref_indices,d)
                leave[key].append((frame.n.to_numpy()[:,None]*q*q).sum(0))
    specifications=[]
    if a.campaign=='seed2029':
        for scope,allowed in [('seed2029',[2029]),('four_seed_pool',[2026,2027,2028,2029])]:
            full=np.flatnonzero((fam=='full')&np.isin(seed,allowed))
            for mask in MASKS:
                ix=np.flatnonzero((fam==mask)&np.isin(seed,allowed))
                specifications.append((scope,mask,[(1,ix),(-1,full)],full))
    elif a.campaign=='joint_identity':
        ix={f:np.flatnonzero(fam==f) for f in ['full','no_drug_identity','no_cell_identity','joint_identity']}
        specifications=[('three_seed','joint_identity_factorial',[(1,ix['joint_identity']),(-1,ix['no_drug_identity']),(-1,ix['no_cell_identity']),(1,ix['full'])],ix['full'])]
    else:
        full=np.flatnonzero(fam=='full_residual');ix=np.flatnonzero(fam=='residual_no_drug')
        specifications=[('three_seed','residual_no_drug',[(1,ix),(-1,full)],full)]
    pf=frame.groupby('pair',observed=True).fold.first().to_numpy();pdg=np.array(meta['pair_drugs']);rows=[]
    for scope,model,terms,den in specifications:
        for level,energy in arrays.items():
            num,baseerr=linear_effect(energy,terms,den)
            r=grouped_intervals(baseerr+num,baseerr,meta['pair_drugs'],seed=20261108)
            if level=='response':
                jk=np.array([num[(pdg!=d).all(1)].sum()/baseerr[(pdg!=d).all(1)].sum() for d in meta['drugs']])
            else:
                el=np.array(leave[level]);ld=sum(weight*el[:,indices].mean(1) for weight,indices in terms)
                jk=ld/el[:,den].mean(1)
            r['drug_low'],r['drug_high']=map(float,jackknife_interval(r['change'],jk))
            se=np.sqrt((len(jk)-1)/len(jk)*((jk-jk.mean())**2).sum())
            p=float(2*student.sf(abs(r['change'])/se,len(jk)-1)) if se else 1.
            fd=[num[pf==f].sum()/baseerr[pf==f].sum() for f in range(5)]
            primary=(model=='joint_identity_factorial' and level=='response') or (model=='residual_no_drug' and level=='panel_reference_residual')
            rows.append(dict(scope=scope,model=model,endpoint=level,**r,p=p,fold_min=min(fd),fold_max=max(fd),
                folds_same_sign=all(np.sign(x)==np.sign(r['change']) for x in fd),primary=primary))
    tab=pd.DataFrame(rows)
    tab['p_holm']=np.nan
    if a.campaign=='seed2029':
        for _,ids in tab.groupby(['scope','endpoint']).groups.items():tab.loc[ids,'p_holm']=holm(tab.loc[ids,'p'].to_numpy())
        tab['historical_material_rule_met']=((abs(tab.change)>=.05)&(tab.p_holm<.05)&tab.folds_same_sign&
            (np.sign(tab.pair_low)==np.sign(tab.change))&(np.sign(tab.pair_high)==np.sign(tab.change))&
            (np.sign(tab.drug_low)==np.sign(tab.change))&(np.sign(tab.drug_high)==np.sign(tab.change)))
        triggered={('no_drug_identity','panel_contrast'),('no_cell_identity','panel_contrast'),('no_cell_identity','dose_contrast'),('no_dose','dose_contrast')}
        tab['frozen_seed_trigger_endpoint']=[(m,e) in triggered for m,e in zip(tab.model,tab.endpoint)]
    tab.to_csv(out/'comparisons.csv',index=False)
    write_json(out/'provenance.json',dict(campaign=a.campaign,dataset_sha256=sha,predictions=sources,
        interpretation='Seed-2029 drug response is exploratory. New experiment claims need Holm across both new primary endpoints; use --combine after both complete.'))
    print(tab.to_string(index=False))


def combine(paths,output):
    tab=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True);tab=tab[tab.primary].copy()
    if set(tab.model)!=set(['joint_identity_factorial','residual_no_drug']) or len(tab)!=2:raise ValueError('Both primary experiments required')
    tab['p_holm']=holm(tab.p.to_numpy())
    tab['material']=((abs(tab.change)>=.05)&(tab.p_holm<.05)&tab.folds_same_sign&
        (np.sign(tab.pair_low)==np.sign(tab.change))&(np.sign(tab.pair_high)==np.sign(tab.change))&
        (np.sign(tab.drug_low)==np.sign(tab.change))&(np.sign(tab.drug_high)==np.sign(tab.change)))
    if Path(output).exists():raise FileExistsError(output)
    tab.to_csv(output,index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',choices=['seed2029','joint_identity','residual_drug_identity'])
    p.add_argument('--runs');p.add_argument('--ablations');p.add_argument('--baseline-root',default='.')
    p.add_argument('--resolution');p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--combine',nargs='+');a=p.parse_args()
    if a.combine:combine(a.combine,a.output)
    else:run(a)
