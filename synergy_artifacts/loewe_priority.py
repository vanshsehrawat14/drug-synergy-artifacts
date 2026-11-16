"""Fixed-fit Loewe root certification, prioritization and numerical sensitivity.

Uses the historical native outputs and the independent forward-curve solver.
No scoring convention is changed in the valid-root correction.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr

from .io import write_json, digest
from .loewe_verify import theta_root


def certificate(cases, effect, tolerance=1e-6):
    """Vectorized original equation, evaluated in log space only on its real domain."""
    y=np.asarray(effect,float)
    inside=np.isfinite(y);terms=[]
    for arm in "ab":
        b,c,d,e=[cases[f"{arm}_{k}"].to_numpy(float) for k in "bcde"]
        model=cases[f"model_{arm}"].to_numpy()
        valid=np.isfinite(np.c_[b,c,d,e]).all(1)&(b!=0)&np.isin(model,["LL.4","L.4"])
        valid&=(y>np.minimum(c,d))&(y<np.maximum(c,d))&((model!="LL.4")|(e>0))
        inside&=valid
        with np.errstate(all="ignore"):
            centre=np.where(model=="LL.4",np.log(e),e)
            inverse=centre+(np.log(np.abs(d-y))-np.log(np.abs(y-c)))/b
            terms.append(np.log(cases[f"dose_{arm}"].to_numpy(float))-inverse)
    with np.errstate(all="ignore"):
        residual=np.expm1(logsumexp(np.stack(terms),axis=0))
    residual=np.where(inside,residual,np.nan)
    return residual,inside&(np.abs(residual)<tolerance)


def top(a,fraction):
    n=max(1,int(np.ceil(len(a)*fraction)))
    return np.argsort(-np.asarray(a),kind="stable")[:n]


def rank_metrics(a,b,fraction=.05):
    x,y=top(a,fraction),top(b,fraction)
    overlap=len(np.intersect1d(x,y));n=len(x)
    return dict(n=len(a),selected=n,retention=overlap/n,turnover=1-overlap/n,
                spearman=float(spearmanr(a,b).statistic),sign_flips=int((np.sign(a)!=np.sign(b)).sum()))


def ranked_bootstrap(frame,a,b,draws=2000,seed=20261107):
    groups=[g.index.to_numpy() for _,g in frame.groupby("pair_id",sort=True)]
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(draws):
        ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        boot.append(rank_metrics(a[ix],b[ix])["turnover"])
    drugs=sorted(set(frame.drug_a)|set(frame.drug_b));leave=[]
    for d in drugs:
        ix=((frame.drug_a!=d)&(frame.drug_b!=d)).to_numpy()
        leave.append(rank_metrics(a[ix],b[ix])["turnover"])
    return dict(turnover_pair_ci=np.quantile(boot,[.025,.975]).tolist(),
                turnover_delete_drug_range=[min(leave),max(leave)],bootstrap_draws=draws)


def score_panel(cases,blocks,reference):
    shift=pd.Series(reference-cases.native_below_3.to_numpy(),index=cases.surface_id).groupby(level=0).mean()
    out=blocks[["pair_id","drug_a","drug_b","cell_line","native_sum_score","synergy_loewe","native_hsa_fallback_score"]].copy()
    out["corrected"]=out.native_sum_score-blocks.surface_id.map(shift).to_numpy()
    return out.groupby(["pair_id","drug_a","drug_b","cell_line"],sort=True,observed=True).mean().reset_index()


def run(output,cases_path,verified_path):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cases=pd.read_parquet(cases_path)
    verified=pd.read_parquet(verified_path).set_index("case_id").reindex(cases.case_id)
    if verified.independent.isna().all(): raise ValueError("No aligned independent roots")
    blocks=pd.read_parquet("results/neural/runs/scientific_native_full/blocks.parquet")
    for c in ["pair_id","drug_a","drug_b","cell_line"]: blocks[c]=blocks[c].astype(str)
    y=verified.independent.to_numpy();r,valid=certificate(cases,y)
    if not np.array_equal(valid,verified.admissible.to_numpy()):
        raise AssertionError("Independent certificate disagrees with verified historical result")
    fix=cases.fallback.to_numpy()&valid
    if fix.sum()!=14197: raise AssertionError("Full correction scope must be 14,197")
    ref=np.where(fix,np.minimum(y,100),cases.native_below_3)
    panel=score_panel(cases,blocks,ref)
    panel["native_rank"]=panel.native_sum_score.rank(ascending=False,method="first")
    panel["corrected_rank"]=panel.corrected.rank(ascending=False,method="first")
    panel["rank_change"]=panel.corrected_rank-panel.native_rank
    panel["score_change"]=panel.corrected-panel.native_sum_score
    panel.to_parquet(out/"panel.parquet",index=False)
    panel.reindex(panel.rank_change.abs().sort_values(ascending=False).index).head(100).to_csv(out/"largest_rank_changes.csv",index=False)
    a,b=panel.native_sum_score.to_numpy(),panel.corrected.to_numpy()
    rows=[]
    for fraction in [.01,.05,.10]:
        rows.append(dict(scope="pair_cell",control="numerical_valid_root_any",fraction=fraction,**rank_metrics(a,b,fraction)))
        rows.append(dict(scope="pair_cell",control="HSA_fallback_convention",fraction=fraction,**rank_metrics(a,panel.native_hsa_fallback_score.to_numpy(),fraction)))
        rows.append(dict(scope="pair_cell",control="published_vs_native",fraction=fraction,**rank_metrics(panel.synergy_loewe.to_numpy(),a,fraction)))
    for cell,g in panel.groupby("cell_line",sort=True):
        rows.append(dict(scope="cell:"+cell,control="numerical_valid_root_any",fraction=.05,**rank_metrics(g.native_sum_score.to_numpy(),g.corrected.to_numpy())))
    pairs=panel.groupby("pair_id",sort=True)[["native_sum_score","corrected"]].mean()
    rows.append(dict(scope="pair_average",control="numerical_valid_root_any",fraction=.05,**rank_metrics(pairs.native_sum_score.to_numpy(),pairs.corrected.to_numpy())))
    pd.DataFrame(rows).to_csv(out/"rankings.csv",index=False)
    summary={"cases":len(cases),"corrected_cases":int(fix.sum()),"panels":len(panel),
             "primary":{**rank_metrics(a,b),**ranked_bootstrap(panel,a,b)},
             "score_change":panel.score_change.describe(percentiles=[.01,.5,.95,.99]).to_dict(),
             "provenance":{"native_cases_sha256":digest(cases_path),"verified_cases_sha256":digest(verified_path)},
             "interpretation_boundary":"Rank changes measure scoring sensitivity, not improved biological hit discovery."}
    write_json(out/"summary.json",summary)
    print(summary["primary"],flush=True)
    acceptance=[]
    rn,native_ok=certificate(cases,cases.native_solution.to_numpy())
    for tol in [1e-4,1e-6,1e-8]:
        independently_ok=np.isfinite(r)&(np.abs(r)<tol)
        native_good=np.isfinite(rn)&(np.abs(rn)<tol)
        use=cases.fallback.to_numpy()&native_good
        p=score_panel(cases,blocks,np.where(use,np.minimum(cases.native_solution,100),cases.native_below_3))
        acceptance.append(dict(rule="native_residual_certificate",tolerance=tol,accepted=int(native_good.sum()),
            recovered_fallback=int(use.sum()),accepted_historical_uncertified=int((~cases.fallback.to_numpy()&~native_good).sum()),
            **rank_metrics(a,p.corrected.to_numpy())))
        use=cases.fallback.to_numpy()&independently_ok
        p=score_panel(cases,blocks,np.where(use,np.minimum(y,100),cases.native_below_3))
        acceptance.append(dict(rule="independent_bracket_certificate",tolerance=tol,accepted=int(independently_ok.sum()),
            recovered_fallback=int(use.sum()),accepted_historical_uncertified=int((~cases.fallback.to_numpy()&~independently_ok).sum()),
            **rank_metrics(a,p.corrected.to_numpy())))
    pd.DataFrame(acceptance).to_csv(out/"acceptance.csv",index=False)
    sensitivity=[]
    params=[f"{arm}_{k}" for arm in "ab" for k in "bcde"]
    for label in ["round6","round8","round10","dose_plus_1e-6","dose_minus_1e-6"]:
        changed=cases.copy()
        if label.startswith("round"):
            digits=int(label[5:]);x=changed[params].to_numpy()
            with np.errstate(all="ignore"):
                scale=10.**(digits-1-np.floor(np.log10(np.maximum(np.abs(x),1e-250))))
                changed[params]=np.round(x*scale)/scale
        else:
            changed[["dose_a","dose_b"]]*=(1+(1e-6 if "plus" in label else -1e-6))
        roots,status=theta_root(changed,steps=80)
        residual,ok=certificate(changed,roots)
        use=cases.fallback.to_numpy()&ok
        p=score_panel(cases,blocks,np.where(use,np.minimum(roots,100),cases.native_below_3))
        common=fix&use;diff=roots[common]-y[common]
        sensitivity.append(dict(perturbation=label,root_status_changes=int((ok!=valid).sum()),
            lost_corrected=int((fix&~use).sum()),new_corrected=int((use&~fix).sum()),
            common_root_abs_median=float(np.median(np.abs(diff))),common_root_abs_max=float(np.max(np.abs(diff))),
            score_abs_max=float(np.max(np.abs(p.corrected.to_numpy()-b))),**rank_metrics(b,p.corrected.to_numpy())))
        print(label,sensitivity[-1],flush=True)
        pd.DataFrame(sensitivity).to_csv(out/"numerical_sensitivity.csv",index=False)
    # Curve sample strata use historical root/fallback status and distance, never prioritization outcomes.
    distance=np.minimum.reduce([np.abs(y-cases[f"{arm}_{k}"].to_numpy()) for arm in "ab" for k in "cd"])
    strata=np.select([fix&(distance<.01),fix,~cases.fallback.to_numpy()&valid],["fallback_near_boundary","fallback_other","accepted_root"],"no_certified_root")
    sf=pd.DataFrame({"surface_id":cases.surface_id,"stratum":strata}).drop_duplicates()
    # One surface per stratum; overlaps removed before the next draw.
    selected=[];used=set();rng=np.random.default_rng(20261107)
    for s in ["fallback_near_boundary","fallback_other","accepted_root","no_certified_root"]:
        ids=np.array(sorted(set(sf.surface_id[sf.stratum==s])-used))
        choose=rng.choice(ids,size=min(40,len(ids)),replace=False);used.update(choose)
        selected.extend([{"surface_id":x,"stratum":s} for x in choose])
    selected=pd.DataFrame(selected).merge(blocks.sort_values("block_id").drop_duplicates("surface_id")[["surface_id","block_id","pair_id","cell_line"]],on="surface_id",validate="one_to_one")
    selected.to_csv(out/"refit_sample.csv",index=False)
    write_json(out/"sensitivity_summary.json",{"strata_sample":selected.stratum.value_counts().to_dict(),
       "native_answer_certified_recovery":int((cases.fallback.to_numpy()&native_ok).sum()),
       "distance_corrected_quantiles":np.quantile(distance[fix],[0,.25,.5,.75,1]).tolist()})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True)
    p.add_argument("--cases",default="results/neural/runs/scientific_native_full/surface_cases.parquet")
    p.add_argument("--verified",required=True);a=p.parse_args();run(a.output,a.cases,a.verified)
