"""Exploratory high-precision certification under the recorded follow-up protocol."""
import argparse
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd

from .io import digest, write_json
from .loewe_priority import certificate, rank_metrics, score_panel
from .loewe_verify import theta_root


def precise_root(row, digits=100):
    """Independent bisection of the inverse equation in the common effect interval.

    Floats are converted exactly, so precision does not silently alter fitted inputs.
    The decimal result, rather than its rounded float, carries the certificate.
    """
    with mp.workdps(digits):
        arms=[]
        for arm in "ab":
            model=row[f"model_{arm}"]
            p=[float(row[f"{arm}_{k}"]) for k in "bcde"]
            dose=float(row[f"dose_{arm}"])
            if model not in ["LL.4","L.4"] or not np.isfinite(p+[dose]).all() or dose<=0:
                return {"certified":False,"status":"invalid"}
            b,c,d,e=map(mp.mpf,p)
            if b==0 or c==d or (model=="LL.4" and e<=0):
                return {"certified":False,"status":"invalid"}
            arms.append((b,c,d,mp.log(e) if model=="LL.4" else e,mp.log(mp.mpf(dose))))
        directions=[mp.sign(-b*(d-c)) for b,c,d,_,_ in arms]
        if directions[0]!=directions[1]:return {"certified":False,"status":"opposite"}
        lower=max(min(c,d) for _,c,d,_,_ in arms)
        upper=min(max(c,d) for _,c,d,_,_ in arms)
        if lower>=upper:return {"certified":False,"status":"empty_interval"}

        def evaluate(t):
            y=lower+(upper-lower)/(1+mp.exp(-t))
            if not lower<y<upper:return y,mp.nan
            r=sum(mp.exp(ld-centre-(mp.log(abs(d-y))-mp.log(abs(y-c)))/b)
                  for b,c,d,centre,ld in arms)-1
            return y,r

        lo,hi=mp.mpf(-200),mp.mpf(200)
        _,rl=evaluate(lo);_,rh=evaluate(hi);direction=directions[0]
        if not direction*rl>0 or not direction*rh<0:
            return {"certified":False,"status":"no_bracket"}
        for _ in range(400):
            mid=(lo+hi)/2;y,res=evaluate(mid)
            if abs(res)<mp.mpf("1e-45"):break
            if direction*res>0:lo=mid
            else:hi=mid
        return {"certified":bool(lower<y<upper and abs(res)<mp.mpf("1e-30")),
                "status":"solved", "effect":float(y),"effect_decimal":mp.nstr(y,digits),
                "residual_decimal":mp.nstr(res,15),"boundary_distance":float(min(y-lower,upper-y))}


def round_coefficients(frame,digits):
    changed=frame.copy();cols=[f"{a}_{k}" for a in "ab" for k in "bcde"]
    x=changed[cols].to_numpy()
    with np.errstate(all="ignore"):
        scale=10.**(digits-1-np.floor(np.log10(np.maximum(np.abs(x),1e-250))))
        changed[cols]=np.round(x*scale)/scale
    return changed


def run(output,verified_path,reuse=False):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    source=Path("results/neural/runs/scientific_native_full/surface_cases.parquet")
    cases=pd.read_parquet(source);changed=round_coefficients(cases,10)
    y,st=theta_root(cases,80);yr,sr=theta_root(changed,80)
    _,ok=certificate(cases,y,1e-8);_,okr=certificate(changed,yr,1e-8)
    _,accept=certificate(cases,y);_,acceptr=certificate(changed,yr)
    suspects=cases.fallback.to_numpy()&((np.char.startswith(st.astype(str),"root")&~ok)|
                                              (np.char.startswith(sr.astype(str),"root")&~okr))
    rng=np.random.default_rng(20261107)
    controls=rng.choice(np.flatnonzero(ok&okr&~suspects),100,replace=False)
    ids=np.r_[np.flatnonzero(suspects),controls];rows=[]
    if reuse:
        table=pd.read_csv(out/"certificates.csv")
        if len(table)!=2*len(ids) or set(table["index"])!=set(ids):raise ValueError("Precision cache scope differs")
    else:
        for n,i in enumerate(ids):
            for condition,frame in [("unchanged",cases),("round10",changed)]:
                rows.append(dict(case_id=cases.iloc[i].case_id,index=int(i),control=not suspects[i],condition=condition,
                                 double_accepted=bool((accept if condition=="unchanged" else acceptr)[i]),
                                 **precise_root(frame.iloc[i])))
            if n%100==0:print("precision",n,"/",len(ids),flush=True)
        table=pd.DataFrame(rows);table.to_csv(out/"certificates.csv",index=False)
    u=table[table.condition=="unchanged"].set_index("index")
    v=table[table.condition=="round10"].set_index("index")
    both=u.certified&v.certified
    delta=(u.effect-v.effect).abs()
    discontinuity=both&~u.control&(u.double_accepted!=v.double_accepted)&(delta<.001)
    verified=pd.read_parquet(verified_path).set_index("case_id").reindex(cases.case_id)
    fix=cases.fallback.to_numpy()&verified.admissible.to_numpy()
    oldref=np.where(fix,np.minimum(verified.independent.to_numpy(),100),cases.native_below_3)
    newref=oldref.copy()
    recovered=u.index[u.certified&~u.control&~fix[u.index]].to_numpy()
    newref[recovered]=np.minimum(u.loc[recovered,"effect"],100)
    blocks=pd.read_parquet("results/neural/runs/scientific_native_full/blocks.parquet")
    for c in ["pair_id","drug_a","drug_b","cell_line"]:blocks[c]=blocks[c].astype(str)
    base=score_panel(cases,blocks,oldref);revised=score_panel(cases,blocks,newref)
    high_ref=np.where(cases.fallback.to_numpy()&accept,np.minimum(y,100),cases.native_below_3)
    high_round=np.where(cases.fallback.to_numpy()&acceptr,np.minimum(yr,100),cases.native_below_3)
    for table_part,ref in [(u,high_ref),(v,high_round)]:
        ii=table_part.index[table_part.certified&~table_part.control]
        ref[ii]=np.minimum(table_part.loc[ii,"effect"],100)
    hp=score_panel(cases,blocks,high_ref);hr=score_panel(cases,blocks,high_round)
    summary=dict(suspects=int(suspects.sum()),controls=len(controls),
        control_certified=int((u.control&u.certified&v.certified).sum()),
        unchanged_suspects_certified=int((~u.control&u.certified).sum()),
        round10_suspects_certified=int((~v.control&v.certified).sum()),
        additional_recovered_cases=len(recovered),original_correction_cases=int(fix.sum()),
        certified_both_max_abs_effect_change=float(delta[both].max()),
        certification_discontinuities=int(discontinuity.sum()),
        high_precision_round10=rank_metrics(hp.corrected.to_numpy(),hr.corrected.to_numpy()),
        high_precision_round10_max_abs_panel_shift=float(abs(hp.corrected-hr.corrected).max()),
        original_to_augmented=rank_metrics(base.corrected.to_numpy(),revised.corrected.to_numpy()),
        native_to_augmented=rank_metrics(base.native_sum_score.to_numpy(),revised.corrected.to_numpy()),
        protocol_sha256=digest("configs/loewe_precision_amendment.json"),cases_sha256=digest(source))
    write_json(out/"summary.json",summary);print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);p.add_argument("--verified",required=True)
    p.add_argument("--reuse",action="store_true");a=p.parse_args();run(a.output,a.verified,a.reuse)
