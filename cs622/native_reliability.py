"""Analyze paired native single-agent refits and fixed-fit solver corrections.

Inputs are emitted by experiments/research_native_fit.R using the recorded sample
and perturbation protocol. A stratified sample is not a population ranking estimate.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from .io import digest, read_json, write_json
from .loewe_priority import certificate, rank_metrics
from .loewe_verify import theta_root


def prepare(directory,sample_path,sources):
    """Build deterministic shared-curve perturbations, without accessing combination labels."""
    out=Path(directory);out.mkdir(parents=True,exist_ok=True)
    if (out/"edges.csv").exists():raise FileExistsError("Use a fresh output directory")
    sample=pd.read_csv(sample_path)
    expected=read_json("results/neural/runs/phase3_native_fit_probe_v2/inputs.json")["source"]
    for name,sha in expected.items():
        if digest(Path(sources)/"R"/name)!=sha:raise ValueError("Historical source hash mismatch: "+name)
    obs=pd.read_parquet("data/processed/oneil_v1/observations.parquet",columns=["block_id","drug_a","drug_b","cell_line","dose_a","dose_b","inhibition"])
    obs=obs[obs.block_id.isin(sample.block_id)]
    edges=[];coordinates=[]
    for row in sample.itertuples():
        f=obs[obs.block_id==row.block_id]
        for arm,other in [("a","b"),("b","a")]:
            for r in f[f["dose_"+other]==0].sort_values("dose_"+arm).itertuples():
                edges.append(dict(original_block=row.block_id,endpoint=arm,dose=getattr(r,"dose_"+arm),response=r.inhibition,
                                  curve_key=(getattr(r,"drug_"+arm),r.cell_line,getattr(r,"dose_"+arm))))
        for r in f[(f.dose_a>0)&(f.dose_b>0)].itertuples():
            coordinates.append(dict(original_block=row.block_id,dose_a=r.dose_a,dose_b=r.dose_b))
    edge=pd.DataFrame(edges);coords=pd.DataFrame(coordinates)
    keys=sorted(set(edge.curve_key));rng=np.random.default_rng(20261107)
    noise={draw:dict(zip(keys,rng.normal(size=len(keys)))) for draw in range(5)}
    conditions=[(0.,-1)]+[(sigma,draw) for sigma in [.1,.5,1.,2.] for draw in range(5)]
    edges=[];coordinates=[];mapping=[];newid=0
    for sigma,draw in conditions:
        for r in sample.itertuples():
            newid+=1;g=edge[edge.original_block==r.block_id].copy()
            g["response"]+=0 if sigma==0 else sigma*g.curve_key.map(noise[draw]);g["block_id"]=newid
            edges.append(g[["block_id","endpoint","dose","response"]])
            c=coords[coords.original_block==r.block_id].copy();c["block_id"]=newid;c["case_id"]=[f"{newid}:{j}" for j in range(len(c))]
            coordinates.append(c[["case_id","block_id","dose_a","dose_b"]])
            mapping.append(dict(block_id=newid,original_block=r.block_id,surface_id=r.surface_id,stratum=r.stratum,sigma=sigma,draw=draw))
    pd.concat(edges).to_csv(out/"edges.csv",index=False);pd.concat(coordinates).to_csv(out/"coordinates.csv",index=False)
    pd.DataFrame(mapping).to_csv(out/"mapping.csv",index=False)
    write_json(out/"protocol.json",dict(sample_surfaces=len(sample),conditions=conditions,shared_curve_noise=True,
        independent_monotherapy_repeats=False,perturbation_interpretation="Stress scales; independent monotherapy repeat provenance is unavailable.",
        source_sha256=expected,driver_sha256=digest("experiments/research_native_fit.R"),edges_sha256=digest(out/"edges.csv"),seed=20261107))


def analyze(directory):
    out=Path(directory)
    mapping=pd.read_csv(out/"mapping.csv")
    cases=pd.read_csv(out/"native/cases.csv");fits=pd.read_csv(out/"native/fits.csv")
    for arm in "ab":
        f=fits[fits.endpoint==arm][["block_id","model","b","c","d","e"]]
        cases=cases.merge(f.rename(columns={c:("model_"+arm if c=="model" else arm+"_"+c)
                                          for c in ["model","b","c","d","e"]}),on="block_id",validate="many_to_one")
    cases=cases.merge(mapping,on="block_id",validate="many_to_one")
    root,status=theta_root(cases,80);residual,ok=certificate(cases,root)
    cases["independent"]=root;cases["certified"]=ok;cases["root_status"]=status
    cases["fallback"]=cases.native_code>=3
    cases["corrected_reference"]=np.where(cases.fallback&ok,np.minimum(root,100),cases.native_reference)
    old_cases=pd.read_parquet("results/neural/runs/scientific_native_full/surface_cases.parquet")
    baseline_cases=cases[cases.sigma==0].copy()
    old_cases=old_cases[old_cases.surface_id.isin(baseline_cases.surface_id)].copy()
    swaps=0
    for sid,g in baseline_cases.groupby("surface_id"):
        ix=old_cases.surface_id==sid;h=old_cases[ix]
        def matches(swap):
            return all(np.allclose(np.sort(g["dose_"+a].unique()),np.sort(h["dose_"+b].unique()),rtol=1e-12,atol=0)
                       for a,b in ([("a","b"),("b","a")] if swap else [("a","a"),("b","b")]))
        direct,reverse=matches(False),matches(True)
        if not direct and not reverse:raise AssertionError("Historical dose grids cannot be aligned")
        def mismatch(swap):
            return sum(abs(float(g.iloc[0][a+"_"+k])-float(h.iloc[0][b+"_"+k]))/max(1,abs(float(h.iloc[0][b+"_"+k])))
                       for a,b in ([("a","b"),("b","a")] if swap else [("a","a"),("b","b")]) for k in "bcde")
        if reverse and (not direct or mismatch(True)<mismatch(False)):
            swaps+=1
            for a,b in [("dose_a","dose_b"),("model_a","model_b")]+[("a_"+k,"b_"+k) for k in "bcde"]:
                old_cases.loc[ix,a],old_cases.loc[ix,b]=h[b].to_numpy(),h[a].to_numpy()
    for f in [baseline_cases,old_cases]:
        for arm in "ab":f["rank_"+arm]=f.groupby("surface_id")["dose_"+arm].rank(method="dense").astype(int)
    same=baseline_cases.merge(old_cases,on=["surface_id","rank_a","rank_b"],suffixes=("_refit","_old"),validate="one_to_one")
    if len(same)!=len(cases[cases.sigma==0]):raise AssertionError("Baseline coefficient alignment failed")
    for arm in "ab":np.testing.assert_allclose(same["dose_"+arm+"_refit"],same["dose_"+arm+"_old"],rtol=1e-12,atol=0)
    pars=[f"{a}_{k}" for a in "ab" for k in "bcde"]
    rel=np.column_stack([abs(same[p+"_refit"]-same[p+"_old"])/np.maximum(1,abs(same[p+"_old"])) for p in pars])
    write_json(out/"unchanged_refit_check.json",dict(cases=len(same),historical_endpoint_swaps=swaps,
        max_scaled_coefficient_difference=float(np.nanmax(rel)),median_scaled_coefficient_difference=float(np.nanmedian(rel)),
        native_code_changes=int((same.native_code_refit!=same.native_code_old).sum()),
        native_reference_max_difference=float(abs(same.native_reference-same.native_below_3).max()),
        interpretation="Unchanged-data refits are compared separately from coefficient-fixed solver corrections; small coefficient changes can change historical acceptance."))
    cols=["native_reference","corrected_reference","hsa_fallback_reference"]
    panel=cases.groupby("block_id")[cols].mean().reset_index().merge(mapping,on="block_id",validate="one_to_one")
    original=pd.read_parquet("results/neural/runs/scientific_native_full/blocks.parquet")
    obs=pd.read_parquet("data/processed/oneil_v1/observations.parquet",columns=["block_id","dose_a","dose_b","inhibition"])
    obs=obs[obs.block_id.isin(mapping.original_block)&(obs.dose_a>0)&(obs.dose_b>0)].groupby("block_id").inhibition.mean()
    panel["observed"]=panel.original_block.map(obs)
    for c in cols:panel[c.replace("reference","score")]=panel.observed-panel[c]
    baseline=panel[panel.sigma==0].set_index("surface_id")
    rows=[]
    for (sigma,draw),g in panel.groupby(["sigma","draw"]):
        g=g.set_index("surface_id").reindex(baseline.index)
        for name in ["native_score","corrected_score","hsa_fallback_score"]:
            a,b=baseline[name].to_numpy(),g[name].to_numpy();valid=np.isfinite(a)&np.isfinite(b)
            d=b[valid]-a[valid]
            rows.append(dict(sigma=sigma,draw=draw,method=name,complete_surfaces=int(valid.sum()),
                             mean_abs_shift=float(np.mean(abs(d))),median_abs_shift=float(np.median(abs(d))),
                             max_abs_shift=float(np.max(abs(d))),**rank_metrics(a[valid],b[valid])))
    pd.DataFrame(rows).to_csv(out/"refit_sensitivity.csv",index=False)
    paired=[]
    for (sigma,draw),g in panel.groupby(["sigma","draw"]):
        valid=g[["native_score","corrected_score"]].notna().all(1);g=g[valid]
        paired.append(dict(sigma=sigma,draw=draw,**rank_metrics(g.native_score.to_numpy(),g.corrected_score.to_numpy()),
                           mean_abs_shift=float(abs(g.native_score-g.corrected_score).mean())))
    pd.DataFrame(paired).to_csv(out/"within_fit_solver_effect.csv",index=False)
    status_table=cases.groupby(["sigma","draw","stratum"],observed=True).agg(cases=("case_id","size"),
        fallbacks=("fallback","sum"),certified=("certified","sum"),native_missing=("native_reference",lambda x:x.isna().sum()))
    cases["recovered"]=cases.fallback&cases.certified
    status_table["recovered"]=cases.groupby(["sigma","draw","stratum"],observed=True).recovered.sum()
    status_table.to_csv(out/"status.csv")
    old=original.set_index("block_id").loc[baseline.original_block,"native_sum_score"].to_numpy()
    diff=baseline.native_score.to_numpy()-old
    summary=dict(surfaces=len(baseline),conditions=len(panel)//len(baseline),fits=len(fits),fit_failures=int((~fits.success).sum()),
        cases=len(cases),official_native_max_abs_difference=float(abs(cases.native_reference-cases.official_reference).max()),
        unchanged_refit_vs_historical_max_abs_score=float(np.max(abs(diff))),
        unchanged_refit_vs_historical_median_abs_score=float(np.median(abs(diff))),
        unchanged_refit_vs_historical_top5=rank_metrics(old,baseline.native_score.to_numpy()),
        sample_only=True,independent_monotherapy_repeats=False,
        limitation="Status-stratified 160-surface sample; perturbation scales are numerical stress tests, not estimated monotherapy noise. Corrected scores retain unresolved historical fallbacks. High-precision certificate instability is analyzed separately.",
        provenance={name:digest(out/name) for name in ["protocol.json","mapping.csv","native/cases.csv","native/fits.csv"]})
    write_json(out/"summary.json",summary)
    panel.to_parquet(out/"panel.parquet",index=False)
    print(summary,flush=True)
    print(pd.DataFrame(rows).groupby(["sigma","method"])[["turnover","sign_flips","mean_abs_shift","max_abs_shift"]].mean().to_string(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",required=True)
    p.add_argument("--prepare",action="store_true");p.add_argument("--sample",default="results/research_20261107/loewe/refit_sample.csv")
    p.add_argument("--sources");a=p.parse_args()
    if a.prepare:
        if not a.sources:p.error("--prepare requires --sources with the pinned historical R files")
        prepare(a.directory,a.sample,a.sources)
    else:analyze(a.directory)
