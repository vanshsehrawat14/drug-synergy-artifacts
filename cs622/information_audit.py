"""Exact paired input-ablation error attribution and reference-augmented contrasts.

These are evaluation projections, not predictors fitted to test outcomes.
Reference vectors use training-pair monotherapy only. No contrast is causal synergy.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .ablation_summary import MASKS, SEEDS, unit_group
from .features import FeatureBuilder
from .io import read_json, write_json, digest
from .resolution import means, projections, residualize, grouped_intervals
from .sprint_report import index, load_truth, out_of_fold, jackknife_interval


def remove_reference_span(frame, contrast, ref_indices, drop=None):
    """Nested W-orthogonal projection off projected reference predictions, by cell/fold."""
    keep = np.ones(len(frame), bool) if drop is None else ((frame.drug_a != drop) & (frame.drug_b != drop)).to_numpy()
    result = np.zeros_like(contrast)
    refs = contrast[:, :1] - contrast[:, ref_indices]
    for _, g in frame[keep].groupby(['fold', 'cell_line'], observed=True):
        ii=g.index.to_numpy();w=g.n.to_numpy(float);h=refs[ii]
        scale=np.sqrt((w[:, None]*h*h).sum(0))
        h=h[:,scale>1e-10]/scale[scale>1e-10]
        result[ii]=residualize(h,contrast[ii],w) if h.shape[1] else contrast[ii]
    return result


def pair_energy(frame, x):
    return np.column_stack([np.bincount(frame.pair, weights=frame.n*c*c) for c in x.T])


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    root=Path(a.resolution);meta=read_json(root/'inputs.json')
    frame=pd.read_parquet(root/'coordinates.parquet')
    df,sha,_=load_truth('data/processed/oneil_v1');t=index(df)
    if sha!=meta['dataset_sha256']:raise ValueError('Dataset mismatch')
    group=unit_group(df,t);n=np.bincount(group);yg=means(group,t['y'],n)
    names=['truth'];fams=['truth'];values=[yg];row=[];repeat=[]
    with np.load(root/'cache.npz') as c:
        for family in ['neural_base_F2','neural_F2','gp_F2','cf_selected','mono_only','bliss']:
            for j,f in enumerate(meta['families']):
                if f==family:
                    names.append(meta['names'][j]);fams.append('full' if family=='neural_base_F2' else f)
                    values.append(c['values'][:,j]);row.append(c['row_pair_error'][:,j]);repeat.append(c['repeat_error'][:,j])
    sources=[]
    hsa=np.zeros(len(df))
    for fold in range(5):
        fp=Path('results/neural/runs/sprint_v1/runs/f2_base/seed_2026')/f'fold_{fold}'/'features.json'
        builder=FeatureBuilder.from_state_dict(read_json(fp));m=df.fold.to_numpy()==fold
        hsa[m]=builder.reference(df[m].drop(columns=['inhibition']), 'hsa')
        sources.append(dict(fold=fold,feature_state_sha256=digest(fp)))
    for label,p in [('hsa',hsa)]+[(f'{mask}:{seed}',out_of_fold({k:Path(a.runs)/mask/f'seed_{seed}'/f'fold_{k}' for k in range(5)},t)) for mask in MASKS for seed in SEEDS]:
        pg=means(group,p,n);err=t['y']-p
        names.append(label);fams.append(label.split(':')[0]);values.append(pg)
        row.append(np.bincount(t['pair'],weights=err**2))
        repeat.append(np.bincount(t['pair'],weights=(err-(yg-pg)[group])**2))
    v=np.column_stack(values);row=np.column_stack([np.zeros(len(t['pairs']))]+row)
    repeat=np.column_stack([np.bincount(t['pair'],weights=(t['y']-yg[group])**2)]+repeat)
    z=v[:,:1]-v;z[:,0]=v[:,0];fams=np.array(fams)
    w=frame.n.to_numpy(float);panel=frame.panel.to_numpy(int);pn=np.bincount(panel,weights=w)
    pm=means(panel,z*w[:,None],pn)[panel]
    refs=[names.index('bliss'),names.index('hsa')]
    with threadpool_limits(limits=a.threads):
        _,_,rp,rw=projections(frame,z)
        ap=remove_reference_span(frame,rp,refs);aw=remove_reference_span(frame,rw,refs)
        arrays={'additive_panel':pm-rp,'pair_contrast':rp,'additive_dose':z-pm-rw,'dose_contrast':rw}
        energies={k:pair_energy(frame,x) for k,x in arrays.items()}
        energies['repeat']=repeat
        recovered=sum(energies.values())
        if not np.allclose(recovered[:,1:].sum(0),row[:,1:].sum(0),atol=.01,rtol=1e-10):raise AssertionError('Error identity failed')
        paired={'panel_reference_residual':pair_energy(frame,ap),'dose_reference_residual':pair_energy(frame,aw)}
        leaves={key:[] for key in paired}
        for drug in meta['drugs']:
            _,_,p,d=projections(frame,z,drug)
            for key,x in [('panel_reference_residual',p),('dose_reference_residual',d)]:
                q=remove_reference_span(frame,x,refs,drug)
                leaves[key].append((w[:,None]*q*q).sum(0))
            print('projected deletion',drug,flush=True)
    components=[];contrasts=[];truth=[];seed_rows=[];model_rows=[];fold_rows=[]
    full=np.flatnonzero(fams=='full');total_rows=w.sum()
    pairfold=frame.groupby('pair',observed=True).fold.first().to_numpy()
    for family in dict.fromkeys(fams[1:]):
        ix=np.flatnonzero(fams==family)
        for key,energy in paired.items():
            r=grouped_intervals(energy[:,ix].mean(1),energy[:,0],meta['pair_drugs'],seed=20261108)
            leave=np.array(leaves[key]);jk=1-leave[:,ix].mean(1)/leave[:,0]
            low,high=map(float,jackknife_interval(-r['change'],jk))
            model_rows.append(dict(model=family,contrast=key,r2=-r['change'],pair_low=-r['pair_high'],pair_high=-r['pair_low'],drug_low=low,drug_high=high))
    for mask in MASKS:
        ix=np.flatnonzero(fams==mask)
        for key,energy in [('response',row),('panel_contrast',energies['pair_contrast']),('dose_contrast',energies['dose_contrast'])]:
            for fold in range(5):
                keep=pairfold==fold
                fold_rows.append(dict(mask=mask,endpoint=key,fold=fold,change=energy[keep][:,ix].sum(0).mean()/energy[keep][:,full].sum(0).mean()-1))
        net=(row[:,ix].sum(0).mean()-row[:,full].sum(0).mean())/total_rows
        for key,energy in energies.items():
            delta=(energy[:,ix].sum(0).mean()-energy[:,full].sum(0).mean())/total_rows
            components.append(dict(mask=mask,component=key,full_mse=energy[:,full].sum(0).mean()/total_rows,
                                   mask_mse=energy[:,ix].sum(0).mean()/total_rows,delta_mse=delta,share_net_delta=delta/net,net_delta_mse=net))
        for key,energy in paired.items():
            r=grouped_intervals(energy[:,ix].mean(1),energy[:,full].mean(1),meta['pair_drugs'],seed=20261108)
            leave=np.array(leaves[key]);jk=leave[:,ix].mean(1)/leave[:,full].mean(1)-1
            r['drug_low'],r['drug_high']=map(float,jackknife_interval(r['change'],jk))
            contrasts.append(dict(mask=mask,contrast=key,**r,full_r2=1-energy[:,full].sum(0).mean()/energy[:,0].sum(),
                                  mask_r2=1-energy[:,ix].sum(0).mean()/energy[:,0].sum()))
        for j in ix:
            matched=names.index('neural_base_F2:'+names[j].split(':')[1])
            for key,energy in energies.items():
                seed_rows.append(dict(mask=mask,seed=int(names[j].split(':')[1]),component=key,
                                      delta_mse=(energy[:,j].sum()-energy[:,matched].sum())/total_rows))
    for key,old,new in [('panel',rp,ap),('dose',rw,aw)]:
        before=(w*old[:,0]**2).sum();after=(w*new[:,0]**2).sum()
        truth.append(dict(contrast=key,original_truth_energy=before,residual_truth_energy=after,
                          share_removed_by_reference_span=1-after/before))
    pd.DataFrame(components).to_csv(out/'error_attribution.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'reference_residual_comparisons.csv',index=False)
    pd.DataFrame(truth).to_csv(out/'reference_energy.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'seed_attribution.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_changes.csv',index=False)
    pd.DataFrame(model_rows).to_csv(out/'model_reference_residuals.csv',index=False)
    write_json(out/'provenance.json',dict(dataset_sha256=sha,names=names,reference_sources=sources,
        max_error_identity_discrepancy=float(np.abs(recovered[:,1:].sum(0)-row[:,1:].sum(0)).max()),
        inference='Pair intervals condition on nuisance projections; drug deletion refits both original and reference-augmented projections. Seed-specific errors averaged.'))
    print(pd.DataFrame(components).to_string(index=False),flush=True)
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',required=True)
    p.add_argument('--resolution',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=2);run(p.parse_args())
