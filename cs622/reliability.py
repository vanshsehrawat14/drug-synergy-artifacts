"""Validation-calibrated uncertainty and pair-cell error triage from saved models."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .features import FeatureBuilder
from .gp_uncertainty import load as gp_load
from .io import read_json, write_json, digest
from .models import make_network
from .sprint_report import RUNS, index, load_truth, model_paths, out_of_fold, jackknife_interval


def predict_saved(path, frame, batch=16384):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    builder=FeatureBuilder.from_state_dict(ck["identity"]["context"]["features"])
    net=make_network(ck["identity"]["spec"]);net.load_state_dict(ck["best_model"]);net.eval()
    offset=ck["identity"]["context"]["config"]["features"].get("reference_offset")
    pred=[];refs=[]
    with torch.inference_mode():
        for start in range(0,len(frame),batch):
            f=frame.iloc[start:start+batch].drop(columns="inhibition",errors="ignore")
            p=net(*(torch.from_numpy(x) for x in builder.transform(f))).numpy().astype(float)*ck["target_scale"]+ck["target_mean"]
            if offset:p+=builder.reference(f,offset)
            pred.append(p);refs.append(builder.reference(f,"bliss"))
    return np.concatenate(pred),np.concatenate(refs)


def top_mass(error,score,fraction=.1):
    """Expected error mass at a fixed budget; ties are selected uniformly without outcome information."""
    error,score=np.asarray(error),np.asarray(score)
    k=max(1,int(np.ceil(len(score)*fraction)))
    threshold=np.partition(score,len(score)-k)[len(score)-k]
    above=score>threshold;equal=score==threshold
    selected=error[above].sum()+(k-above.sum())/equal.sum()*error[equal].sum()
    return float(selected/error.sum()) if error.sum()>0 else np.nan


def calibrator(train,test,kind):
    """Validation-only empirical error variance with 20-panel shrinkage."""
    global_mse=float(train.error.mean())
    if kind=="global":return np.full(len(test),global_mse)
    if kind=="dose_bliss":
        cells=sorted(train.cell_line.astype(str).unique())
        def features(f):
            return np.c_[f[["mean_logdose","gap_logdose","reference"]].to_numpy(float),
                         np.column_stack([(f.cell_line==c).to_numpy(float) for c in cells])]
        a,b=features(train),features(test);centre=a[:,:3].mean(0);scale=a[:,:3].std(0)
        scale=np.where(scale>1e-8,scale,1.)
        a[:,:3]=(a[:,:3]-centre)/scale;b[:,:3]=(b[:,:3]-centre)/scale
        model=Ridge(alpha=20).fit(a,np.log1p(train.error.to_numpy()))
        back=lambda x:np.maximum(np.expm1(np.clip(model.predict(x),-20,30)),1e-8)
        return back(b)*(global_mse/back(a).mean())
    keys=["cell_line"] if kind=="cell" else ["cell_line","bliss_bin"]
    table=train.groupby(keys,observed=True).error.agg(["sum","size"])
    table["estimate"]=(table["sum"]+20*global_mse)/(table["size"]+20)
    joined=test[keys].merge(table.reset_index(),on=keys,how="left",sort=False)
    return joined.estimate.fillna(global_mse).to_numpy()


def calibration(y,p,var):
    var=np.maximum(var,1e-8);z=(y-p)/np.sqrt(var)
    return dict(nll=float(np.mean(.5*np.log(2*np.pi*var)+.5*z*z)),coverage90=float(np.mean(abs(z)<=norm.ppf(.95))),
                width90=float(2*norm.ppf(.95)*np.sqrt(var).mean()),mse=float(np.mean((y-p)**2)))


def pair_inference(frame,a,b,draws=2000):
    err=frame.error.to_numpy();aa=frame[a].to_numpy();bb=frame[b].to_numpy()
    full=top_mass(err,aa)-top_mass(err,bb)
    groups=[g.index.to_numpy() for _,g in frame.groupby("pair_id",sort=True)]
    rng=np.random.default_rng(20261107);boot=[]
    for _ in range(draws):
        ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
        boot.append(top_mass(err[ix],aa[ix])-top_mass(err[ix],bb[ix]))
    leave=[]
    for d in sorted(set(frame.drug_a)|set(frame.drug_b)):
        ix=((frame.drug_a!=d)&(frame.drug_b!=d)).to_numpy()
        leave.append(top_mass(err[ix],aa[ix])-top_mass(err[ix],bb[ix]))
    return dict(difference=full,pair_low=float(np.quantile(boot,.025)),pair_high=float(np.quantile(boot,.975)),
                drug_low=float(jackknife_interval(full,leave)[0]),drug_high=float(jackknife_interval(full,leave)[1]))


def run(output,reuse=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    df,sha,_=load_truth("data/processed/oneil_v1");t=index(df)
    suite=RUNS/"sprint_v1";selected=read_json(suite/"selection.json")["selected"]
    gdir=RUNS/"gp_gpu_f2_residual_mono"
    gp=gp_load(gdir,t);gv=gp_load(gdir,t,"validation")
    paths=[model_paths("selected:F2",s,selected,suite) for s in [2026,2027,2028]]
    test_nn=np.array([out_of_fold(p,t) for p in paths])
    panels=[];row_cal=[];chosen=[];restores=[]
    for k in range(5):
        test=df.iloc[t["rows"][k]].copy();val=df.iloc[t["rows"][(k+1)%5]].copy()
        cache=out/f"validation_{k}.npz"
        if cache.exists() and reuse:
            with np.load(cache) as z:vp,vr=z["predictions"],z["reference"]
        else:
            vp=[];vr=None
            for s,p in zip([2026,2027,2028],paths):
                checkpoint=p[k]/"checkpoints/best.pt"
                # Verify restoration on test inputs before making unseen validation predictions.
                test_index=np.linspace(0,len(test)-1,500,dtype=int)
                check,_=predict_saved(checkpoint,test.iloc[test_index])
                delta=float(np.max(np.abs(check-test_nn[s-2026,t["rows"][k][test_index]])))
                if delta>1e-3:raise AssertionError(f"Checkpoint restore mismatch {delta}")
                restores.append(dict(fold=k,seed=s,max_abs_difference=delta,checkpoint_sha256=digest(checkpoint)))
                pred,ref=predict_saved(checkpoint,val);vp.append(pred)
                if vr is not None and not np.allclose(vr,ref,atol=1e-3):raise AssertionError("Seed monotherapy references differ")
                vr=ref
                print("validation restored",k,s,flush=True)
            vp=np.array(vp);np.savez_compressed(cache,predictions=vp,reference=vr)
        with torch.no_grad():
            fb=FeatureBuilder.from_state_dict(read_json(paths[0][k]/"features.json"))
            tr=fb.reference(test.drop(columns="inhibition"),"bliss")
        for name,pm,pv,sv,st in [
            ("neural_ensemble",test_nn[:,t["rows"][k]].mean(0),vp.mean(0),vp.var(0,ddof=1),test_nn[:,t["rows"][k]].var(0,ddof=1)),
            ("gp_F2",gp["prediction"][t["rows"][k]],gv["prediction"][k],gv["predictive_std"][k]**2,gp["predictive_std"][t["rows"][k]]**2),
            ("gp_F2_latent",gp["prediction"][t["rows"][k]],gv["prediction"][k],gv["latent_std"][k]**2,gp["latent_std"][t["rows"][k]]**2)]:
            tv=test[["pair_id","drug_a","drug_b","cell_line"]].copy();vv=val[["pair_id","drug_a","drug_b","cell_line"]].copy()
            tv["y"],vv["y"]=test.inhibition.to_numpy(),val.inhibition.to_numpy()
            tv["prediction"],vv["prediction"]=pm,pv
            tv["variance"],vv["variance"]=st,sv
            tv["reference"],vv["reference"]=tr,vr
            for dst,src in [(tv,test),(vv,val)]:
                loga,logb=np.log10(src.dose_a.to_numpy()),np.log10(src.dose_b.to_numpy())
                dst["mean_logdose"]=(loga+logb)/2;dst["gap_logdose"]=abs(loga-logb)
            tv["error"],vv["error"]=(tv.y-tv.prediction)**2,(vv.y-vv.prediction)**2
            tv["bliss_bin"],vv["bliss_bin"]=np.digitize(tr,[25,50,75]),np.digitize(vr,[25,50,75])
            # Row calibration. Variance fitting uses validation outcomes only.
            if name=="neural_ensemble":
                coef=nnls(np.c_[np.ones(len(sv)),sv],vv.error.to_numpy())[0]
                calibrated=coef[0]+coef[1]*st
            else:
                coef=np.array([0.,np.mean(vv.error.to_numpy()/np.maximum(sv,1e-9))]);calibrated=coef[1]*st
            tv["calibrated_variance"]=calibrated
            for method,var in [("raw_uncertainty",st),("validation_calibrated",calibrated)]+[(b,calibrator(vv,tv,b)) for b in ["global","cell","cell_bliss","dose_bliss"]]:
                row_cal.append(dict(model=name,fold=k,uncertainty=method,**calibration(tv.y.to_numpy(),pm,var),
                                    rows=len(tv),noise_floor=coef[0],scale=coef[1]))
            # Triage target is error of the *pair-cell mean*, not the mean squared dose error.
            keys=["pair_id","drug_a","drug_b","cell_line"]
            vt=vv.groupby(keys,observed=True).agg(y=("y","mean"),prediction=("prediction","mean"),reference=("reference","mean"),
                          variance=("variance","mean"),row_mse=("error","mean"),mean_logdose=("mean_logdose","mean"),gap_logdose=("gap_logdose","mean")).reset_index()
            tt=tv.groupby(keys,observed=True).agg(y=("y","mean"),prediction=("prediction","mean"),reference=("reference","mean"),
                          variance=("variance","mean"),row_mse=("error","mean"),mean_logdose=("mean_logdose","mean"),gap_logdose=("gap_logdose","mean")).reset_index()
            for f in [vt,tt]:f["error"]=(f.y-f.prediction)**2;f["bliss_bin"]=np.digitize(f.reference,[25,50,75])
            if name=="neural_ensemble":
                # Disagreement of panel means preserves within-network dose correlation.
                vc=np.column_stack([val.assign(p=p).groupby(keys,observed=True).p.mean().to_numpy() for p in vp])
                tc=np.column_stack([test.assign(p=p).groupby(keys,observed=True).p.mean().to_numpy() for p in test_nn[:,t["rows"][k]]])
                vt["variance"],tt["variance"]=vc.var(1,ddof=1),tc.var(1,ddof=1)
            # Select the simple baseline using a fixed half of *validation pairs*, then refit all validation pairs.
            ps=np.array(sorted(vt.pair_id.unique()));rng=np.random.default_rng(20261107+k)
            half=set(rng.permutation(ps)[:len(ps)//2]);train=vt.pair_id.isin(half)
            selection={b:top_mass(vt.loc[~train,"error"].to_numpy(),calibrator(vt[train],vt[~train],b)) for b in ["global","cell","cell_bliss"]}
            best=max(selection,key=lambda x:(selection[x],-len(x)))
            augmented=dict(selection,dose_bliss=top_mass(vt.loc[~train,"error"].to_numpy(),calibrator(vt[train],vt[~train],"dose_bliss")))
            extra=max(augmented,key=lambda x:(augmented[x],-len(x)))
            for b in selection:tt[b]=calibrator(vt,tt,b)
            tt["dose_bliss"]=calibrator(vt,tt,"dose_bliss");tt["augmented_baseline"]=tt[extra]
            tt["selected_baseline"]=tt[best];tt["model"]=name;tt["fold"]=k
            panels.append(tt);chosen.append(dict(model=name,fold=k,selected=best,validation_half_capture=selection,
                                                 augmented_selected=extra,augmented_half_capture=augmented))
        print("uncertainty fold",k,"complete",flush=True)
    panel=pd.concat(panels,ignore_index=True);panel.to_parquet(out/"panel.parquet",index=False)
    pd.DataFrame(row_cal).to_csv(out/"row_calibration.csv",index=False)
    write_json(out/"validation_selection.json",chosen)
    if restores:write_json(out/"restoration_checks.json",restores)
    summary=[];comparisons=[]
    for name,g in panel.groupby("model",sort=True):
        g=g.reset_index(drop=True)
        for method in ["variance","global","cell","cell_bliss","dose_bliss","selected_baseline","augmented_baseline"]:
            summary.append(dict(model=name,uncertainty=method,panels=len(g),panel_mse=float(g.error.mean()),
                top10_error_mass=top_mass(g.error,g[method]),top25_error_mass=top_mass(g.error,g[method],.25),
                top10_dose_error_mass=top_mass(g.row_mse,g[method]),spearman_error=float(spearmanr(g.error,g[method]).statistic)))
        for control in ["selected_baseline","augmented_baseline"]:
            comparisons.append(dict(model=name,control=control,**pair_inference(g,"variance",control)))
    pd.DataFrame(summary).to_csv(out/"triage.csv",index=False);pd.DataFrame(comparisons).to_csv(out/"comparisons.csv",index=False)
    write_json(out/"provenance.json",{"dataset_sha256":sha,"source":"frozen five-fold predictions and saved best checkpoints",
       "neural":"three-seed ensemble mean and sample disagreement; distinct from historical seed-average error",
       "gp":"mean marginal predictive variance ranks panels; no independence assumption or panel interval from marginal variances",
       "baseline_selection":"fixed 50/50 split of validation pairs; refit selected simple score on all validation pairs",
       "limitation":"Validation rows previously selected models/checkpoints; calibration is development calibration, not an untouched conformal guarantee."})
    print(pd.DataFrame(summary).to_string(index=False),flush=True);print(comparisons,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);p.add_argument("--reuse",action="store_true")
    a=p.parse_args();torch.set_num_threads(4)
    with threadpool_limits(limits=4):run(a.output,a.reuse)
