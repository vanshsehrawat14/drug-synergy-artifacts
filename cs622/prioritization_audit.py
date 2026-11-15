"""Named screening changes, threshold margins and fixed-fit mechanism attribution.

All scores are retrospective observed-response summaries. No biological efficacy
claim follows from a certified mathematical reference or a shortlist change.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from .decision_sensitivity import KEY,BUDGETS,references,unit_scores,aggregate,selected,ranks,compare,pair_bootstrap
from .decision_followup import fallback_value
from .io import digest,write_json


def change_table(units,x,y,budget,base,alternative):
    A,n=selected(x,budget);B,_=selected(y,budget);cut=np.sort(x)[::-1][n-1];cut_b=np.sort(y)[::-1][n-1]
    r=units.copy();r['score_native']=x;r['score_alternative']=y;r['score_change']=y-x
    r['native_rank']=ranks(x)+1;r['alternative_rank']=ranks(y)+1
    r['native_cutoff']=cut;r['alternative_cutoff']=cut_b
    r['native_margin']=abs(x-cut);r['alternative_margin']=abs(y-cut_b)
    r['native_signed_margin']=x-cut;r['alternative_signed_margin']=y-cut_b
    r['selection']=np.select([B&~A,A&~B],['entered','exited'],'unchanged')
    r['base']=base;r['alternative']=alternative;r['budget']=budget
    return r[A^B].copy()


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    c=pd.read_parquet(a.cases);b=pd.read_parquet(a.blocks)
    for key in KEY:b[key]=b[key].astype(str)
    cert=pd.read_parquet(a.certificates);q=cert.set_index('case_id').reindex(c.case_id)
    ref,masks=references(c,pd.read_parquet(a.verified),pd.read_csv(a.precision),cert)
    pc={v:aggregate(b,unit_scores(c,b,r))[0].to_numpy() for v,r in ref.items()}
    pc['H']=aggregate(b,b.native_hsa_fallback_score.to_numpy())[0].to_numpy()
    pc['P']=aggregate(b,b.synergy_loewe.to_numpy())[0].to_numpy()
    units=aggregate(b,b.native_sum_score.to_numpy())[0].reset_index()[KEY]
    dist=np.minimum(q.dist_lower.to_numpy(float),q.dist_upper.to_numpy(float))
    fb=fallback_value(c);root=np.minimum(q.root.to_numpy(float),100)
    # Only the root-domain edge convention changes; genuinely undefined cases
    # keep their historical values. A separate undefined-case intervention follows.
    for eps in [1e-6,.0001,.01]:
        r=np.where(masks['has_root']&(dist>=eps),root,np.where(masks['has_root'],fb,ref['N']))
        pc[f'edge_{eps:g}']=aggregate(b,unit_scores(c,b,r))[0].to_numpy()
    undefined=q.cert_status.str.startswith('undefined').to_numpy()
    r=np.where(undefined&~masks['fallback'],fb,ref['N'])
    pc['undefined_to_sum']=aggregate(b,unit_scores(c,b,r))[0].to_numpy()
    comparisons=[('N','C1'),('N','I'),('I','R'),('N','H'),('P','N'),('N','edge_1e-06'),('N','edge_0.0001'),('N','edge_0.01'),('N','undefined_to_sum')]
    changes=[];summ=[]
    for base,alt in comparisons:
        for k in BUDGETS:
            table=change_table(units,pc[base],pc[alt],k,base,alt);changes.append(table)
            for role,g in table.groupby('selection'):
                summ.append(dict(base=base,alternative=alt,budget=k,selection=role,n=len(g),
                    margin_median=float(g.native_margin.median()),margin_p90=float(g.native_margin.quantile(.9)),
                    margin_max=float(g.native_margin.max()),mean_score_change=float(g.score_change.mean()),
                    median_score_change=float(g.score_change.median()),
                    share_margin_le_01=float((g.native_margin<=.1).mean()),share_margin_le_05=float((g.native_margin<=.5).mean()),
                    share_margin_le_1=float((g.native_margin<=1).mean()),share_margin_le_5=float((g.native_margin<=5).mean()),
                    distinct_pairs=g.pair_id.nunique(),distinct_cells=g.cell_line.nunique()))
    changes=pd.concat(changes,ignore_index=True)
    # Exact linear score-shift attribution: cases -> surface mean -> block mean -> panel mean.
    effects={}
    fix=masks['fallback']&masks['has_root']
    for label,mask in [('boundary',fix&(dist<.01)),('interior',fix&(dist>=.01))]:
        r=np.where(mask,ref['I'],ref['N'])
        effects[label]=aggregate(b,unit_scores(c,b,r))[0].to_numpy()-pc['N']
    if not np.allclose(effects['boundary']+effects['interior'],pc['I']-pc['N'],atol=1e-11):raise AssertionError('Attribution identity')
    contribution=units.copy()
    contribution['boundary_delta']=effects['boundary'];contribution['interior_delta']=effects['interior']
    contribution['total_delta']=pc['I']-pc['N']
    changes=changes.merge(contribution,on=KEY,validate='many_to_one')
    changes.to_csv(out/'changed_selections.csv',index=False)
    pd.DataFrame(summ).to_csv(out/'margin_summary.csv',index=False)
    rounding=[]
    for alt in ['C1','I']:
        for k in BUDGETS:
            native,_=selected(pc['N'],k);corrected,_=selected(pc[alt],k);rounded,_=selected(np.round(pc['N'],0),k)
            enters=corrected&~native;rounded_enters=rounded&~native
            rounding.append(dict(alternative=alt,budget=k,corrected_entries=int(enters.sum()),integer_rounded_entries=int(rounded_enters.sum()),
                shared_entries=int((enters&rounded_enters).sum()),share_corrected_entries_also_from_rounding=float((enters&rounded_enters).sum()/max(enters.sum(),1))))
    pd.DataFrame(rounding).to_csv(out/'rounding_overlap.csv',index=False)
    contribution.to_csv(out/'score_attribution.csv',index=False)
    sensitivity=[];cores=[];cell_rows=[];drug_rows=[]
    for k in BUDGETS:
        native,n=selected(pc['N'],k);strict,_=selected(pc['I'],k);edge,_=selected(pc['edge_0.01'],k);hsa,_=selected(pc['H'],k)
        core=native&strict&edge;wide=core&hsa
        cores.append(dict(budget=k,selected=n,native_strict_edge_core=int(core.sum()),core_share=float(core.sum()/n),
                          core_with_hsa=int(wide.sum()),core_with_hsa_share=float(wide.sum()/n)))
        for alt in ['C1','I']:
            metrics=compare(pc['N'],pc[alt],k)
            boot=pair_bootstrap(units,pc['N'],pc[alt],k,a.draws)
            sensitivity.append(dict(alternative=alt,**metrics,**boot))
            for cell,ii in units.groupby('cell_line').indices.items():
                r=compare(pc['N'][ii],pc[alt][ii],k)
                cell_rows.append(dict(cell_line=cell,alternative=alt,budget=k,**{j:r[j] for j in ['n_units','selected','replaced','replacement_fraction']}))
            for cell in sorted(units.cell_line.unique()):
                m=(units.cell_line!=cell).to_numpy();r=compare(pc['N'][m],pc[alt][m],k)
                drug_rows.append(dict(deleted_cell=cell,alternative=alt,budget=k,replacement_fraction=r['replacement_fraction']))
    pd.DataFrame(sensitivity).to_csv(out/'budget_stability.csv',index=False)
    pd.DataFrame(cores).to_csv(out/'convention_core.csv',index=False)
    pd.DataFrame(cell_rows).to_csv(out/'within_cell.csv',index=False)
    pd.DataFrame(drug_rows).to_csv(out/'cell_deletion.csv',index=False)
    primary=changes[(changes.base=='N')&(changes.alternative=='C1')&(changes.budget==.05)]
    if len(primary)!=134 or (primary.selection=='entered').sum()!=67:raise AssertionError('Primary turnover failed')
    correction=compare(pc['I'],pc['R'],.05)
    named=primary[primary.selection=='entered'].nlargest(8,'score_change')
    named.to_csv(out/'largest_entering_selections.csv',index=False)
    # Pure reference-accuracy update of accepted roots, separately from fallback recovery.
    write_json(out/'summary.json',dict(cases_sha256=digest(a.cases),certificates_sha256=digest(a.certificates),
        primary_replacements=67,selected=1137,changed_pair_cells=len(primary),
        strict_replacements=int(((selected(pc['N'],.05)[0])&~selected(pc['I'],.05)[0]).sum()),
        accepted_root_accuracy_top5_replacements=correction['replaced'],
        accepted_root_accuracy_max_panel_score_shift=float(abs(pc['R']-pc['I']).max()),
        undefined_cases=int(undefined.sum()),no_root_cases=int((q.cert_status=='certified_no_root').sum()),
        corrected_cases=int(fix.sum()),corrected_boundary_cases=int((fix&(dist<.01)).sum()),
        all_corrected_pair_cells_increase=bool(np.all((pc['I']-pc['N'])[abs(pc['I']-pc['N'])>1e-12]>0)),
        units=len(units),note='Canonical ranking; pair bootstrap reselects within each draw; drug-LOO intervals are sensitivity ranges. Retrospective selections, no biological validation.'))
    print(pd.DataFrame(summ).query("base=='N' and alternative in ['C1','I'] and budget==.05").to_string(index=False),flush=True)
    print(named[KEY+['score_native','score_alternative','score_change','native_margin']].to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',default='results/neural/runs/scientific_native_full/surface_cases.parquet')
    p.add_argument('--blocks',default='results/neural/runs/scientific_native_full/blocks.parquet')
    p.add_argument('--precision',default='results/research_20261107/precision/certificates.csv')
    p.add_argument('--verified',required=True);p.add_argument('--certificates',required=True);p.add_argument('--output',required=True)
    p.add_argument('--draws',type=int,default=2000);run(p.parse_args())
