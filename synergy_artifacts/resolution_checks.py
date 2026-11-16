"""Prespecified HSA, equal-panel weighting and replicate-dispersion sensitivities."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .features import FeatureBuilder
from .io import read_json, write_json
from .resolution import means, projections, grouped_intervals
from .neural_report import load_truth,index,model_paths,RUNS,jackknife_interval


def run(directory):
    out=Path(directory);meta=read_json(out/"inputs.json")
    f=pd.read_parquet(out/"coordinates.parquet")
    with np.load(out/"cache.npz") as z:v,rowerr,rep=z["values"],z["row_pair_error"],z["repeat_error"]
    n=f.n.to_numpy();panel=f.panel.to_numpy();pair=f.pair.to_numpy();pn=np.bincount(panel,weights=n)
    df,_,_=load_truth("data/processed/oneil_v1");t=index(df)
    suite=RUNS/"benchmark_v1";chosen=read_json(suite/"selection.json")["selected"]
    paths=model_paths("selected:F2",2026,chosen,suite);hsa=np.full(len(df),np.nan);bliss=hsa.copy()
    for k in range(5):
        fb=FeatureBuilder.from_state_dict(read_json(paths[k]/"features.json"));test=df.iloc[t["rows"][k]]
        hsa[t["rows"][k]]=fb.reference(test,"hsa");bliss[t["rows"][k]]=fb.reference(test,"bliss")
    # Exact-key grouping matches the main decomposition, including actual concentrations.
    ra=df.groupby("block_id",observed=True).dose_a.rank(method="dense").to_numpy(int)-1
    rb=df.groupby("block_id",observed=True).dose_b.rank(method="dense").to_numpy(int)-1
    group,_=pd.factorize(pd.MultiIndex.from_arrays([t["panel"],ra,rb,df.dose_a,df.dose_b]),sort=True)
    families=np.array(meta["families"]);bi=np.flatnonzero(families=="bliss")[0]
    if not np.allclose(means(group,bliss,n),v[:,bi],atol=1e-5):raise AssertionError("Reference lookup differs")
    hg=means(group,hsa,n);herror=(t["y"]-hsa)**2
    hp,hw,_,_=projections(f,(v[:,0]-hg)[:,None])
    with np.load(out/"contrasts.npz") as z:cp,cw=z["panel"],z["within"]
    rr=[]
    for family in dict.fromkeys(families):
        if family=="truth":continue
        ix=np.flatnonzero(families==family)
        rr.append(dict(model=family,hsa_residual_mse_reduction=1-rowerr[:,ix].sum(0).mean()/herror.sum(),
            hsa_panel_contrast_mse_reduction=1-(cp[:,ix]**2*n[:,None]).sum(0).mean()/hp[0],
            hsa_dose_contrast_mse_reduction=1-(cw[:,ix]**2*n[:,None]).sum(0).mean()/hw[0]))
    pd.DataFrame(rr).to_csv(out/"hsa_sensitivity.csv",index=False)
    # Each panel has total weight one, preserving replicate proportions inside a panel.
    ff=f.copy();ff["n"]=n/pn[panel]
    z=v[:,:1]-v;z[:,0]=v[:,0]
    ep,ew,rp,rw=projections(ff,z);w=ff.n.to_numpy()
    leaves_p=[];leaves_w=[]
    for drug in meta["drugs"]:
        a,b,_,_=projections(ff,z,drug,False);leaves_p.append(a);leaves_w.append(b)
        print("equal-panel deletion",drug,flush=True)
    lp,lw=np.array(leaves_p),np.array(leaves_w)
    rows=[]
    for family in dict.fromkeys(families):
        if family=="truth":continue
        ix=np.flatnonzero(families==family)
        rows.append(dict(model=family,panel_contrast_r2=1-ep[ix].mean()/ep[0],dose_contrast_r2=1-ew[ix].mean()/ew[0]))
    pd.DataFrame(rows).to_csv(out/"equal_panel_metrics.csv",index=False)
    comparisons=[]
    for a,b in [("gp_F2","neural_F2"),("cf_selected","neural_F0"),("gp_F0","neural_F0")]:
        ia,ib=np.flatnonzero(families==a),np.flatnonzero(families==b)
        for level,res,leave in [("panel_contrast",rp,lp),("dose_contrast",rw,lw)]:
            errors=np.column_stack([np.bincount(pair,weights=w*x*x) for x in res.T])
            r=grouped_intervals(errors[:,ia].mean(1),errors[:,ib].mean(1),meta["pair_drugs"])
            r["drug_low"],r["drug_high"]=map(float,jackknife_interval(r["change"],leave[:,ia].mean(1)/leave[:,ib].mean(1)-1))
            comparisons.append(dict(model=a,control=b,level=level,**r))
    pd.DataFrame(comparisons).to_csv(out/"equal_panel_comparisons.csv",index=False)
    dof=np.bincount(pair,weights=n-1);sse=rep[:,0]
    rng=np.random.default_rng(20261107);ix=rng.integers(len(sse),size=(2000,len(sse)))
    boot=sse[ix].sum(1)/dof[ix].sum(1)
    write_json(out/"replicate_dispersion.json",dict(pooled_within_input_variance=float(sse.sum()/dof.sum()),
        pair_ci=np.quantile(boot,[.025,.975]).tolist(),within_input_degrees_of_freedom=int(dof.sum()),
        identical_input_units=len(n),hsa_reference_mse=float(herror.mean()),
        limitation="Dispersion among repeat rows at identical inputs includes experimental and processing variation. Pooling weights by within-input degrees of freedom. Independence across plates and independent monotherapy repeat noise are not established; this is not a biological-noise floor."))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--directory",required=True);a=p.parse_args()
    with threadpool_limits(limits=1):run(a.directory)
