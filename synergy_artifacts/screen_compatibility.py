"""Exact identifier, concentration and monotherapy support audit of inspected screens."""
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from .data import canonicalize, _sparse_drugs
from .io import digest, read_json, write_json

COLS=["block_id","drug_a","drug_b","cid_a","cid_b","cell_line","cellosaurus_accession",
      "dose_a","dose_b","unit_a","unit_b","inhibition","pair_id"]


def endpoint_table(frame,mono=False):
    rows=[]
    for a,b in [("a","b"),("b","a")]:
        f=frame[(frame["dose_"+a]>0)&((frame["dose_"+b]==0) if mono else (frame["dose_"+b]>0))]
        rows.append(f[["cid_"+a,"drug_"+a,"cellosaurus_accession","dose_"+a,"inhibition"]].set_axis(
                    ["cid","drug","cell","dose","response"],axis=1))
    return pd.concat(rows,ignore_index=True)


def run(output,almanac_raw,source):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    meta_path=out/"almanac_identifiers.parquet"
    if not meta_path.exists():
        parts=[];cols=["block_id","drug_row_cid","drug_col_cid","cellosaurus_accession","study_name"]
        for chunk in pd.read_csv(source,usecols=cols,chunksize=300000,low_memory=False):
            f=chunk[chunk.study_name=="ALMANAC"].drop(columns="study_name").drop_duplicates()
            if len(f):parts.append(f)
        meta=pd.concat(parts,ignore_index=True).drop_duplicates()
        if meta.block_id.duplicated().any():raise ValueError("Ambiguous block identifiers")
        meta.to_parquet(meta_path,index=False)
    raw=pd.read_parquet(almanac_raw).merge(pd.read_parquet(meta_path),on="block_id",validate="many_to_one")
    alm=canonicalize(raw,merge_variants=True);del raw
    cfg=read_json("configs/data_almanac.json")
    excluded=_sparse_drugs(alm,cfg.get("min_pairs_per_drug",0))["drugs"]
    alm=alm[~alm.drug_a.isin(excluded)&~alm.drug_b.isin(excluded)]
    frames={"ONEIL":pd.read_parquet("data/processed/oneil_v1/observations.parquet",columns=COLS),
            "ALMANAC":alm[COLS],
            "NCATS_ES":pd.read_parquet("data/processed/ncats_es_v2/observations.parquet",columns=COLS),
            "NCATS_DIPG":pd.read_parquet("data/processed/ncats_dipg_v2/observations.parquet",columns=COLS)}
    profiles=[];maps={};ranges={};combos={};identities=[]
    for name,f in frames.items():
        c=f[(f.dose_a>0)&(f.dose_b>0)&np.isfinite(f.inhibition)].copy()
        e=endpoint_table(c);m=endpoint_table(f,True)
        dr=e[["cid","drug"]].drop_duplicates();ce=c[["cell_line","cellosaurus_accession"]].drop_duplicates()
        for rr in dr.itertuples():identities.append(dict(screen=name,kind="drug",name=rr.drug,identifier=rr.cid))
        for rr in ce.itertuples():identities.append(dict(screen=name,kind="cell",name=rr.cell_line,identifier=rr.cellosaurus_accession))
        maps[name]=(dr,ce)
        ranges[name]=e.groupby(["cid","cell"],observed=True).dose.agg(["min","max"])
        combos[name]=c
        mono=m.groupby(["cid","cell","dose"],observed=True).response.agg(["mean","size","nunique"])
        exact=pd.MultiIndex.from_frame(e[["cid","cell","dose"]]).isin(mono.index)
        profiles.append(dict(screen=name,rows=len(c),blocks=c.block_id.nunique(),pairs=c.pair_id.nunique(),
            drugs=dr.drug.nunique(),positive_cids=dr.loc[dr.cid>0,"cid"].nunique(),cells=c.cell_line.nunique(),
            dose_min=float(e.dose.min()),dose_median=float(e.dose.median()),dose_max=float(e.dose.max()),
            response_mean=float(c.inhibition.mean()),response_sd=float(c.inhibition.std(ddof=0)),
            response_below0=float((c.inhibition<0).mean()),response_above100=float((c.inhibition>100).mean()),
            monotherapy_exact_endpoint_coverage=float(exact.mean()),
            monotherapy_keys=len(mono),mono_keys_multiple_values=int((mono["nunique"]>1).sum()),
            units=";".join(sorted(set(c.unit_a.astype(str))|set(c.unit_b.astype(str))))))
        print(name,profiles[-1],flush=True)
    overlaps=[]
    for source_name,target_name in combinations(frames,2):
        ds,cs=maps[source_name];dt,ct=maps[target_name]
        dc=set(ds.cid[ds.cid>0])&set(dt.cid[dt.cid>0])
        cc=(set(cs.cellosaurus_accession.dropna())&set(ct.cellosaurus_accession.dropna()))-{"","nan","None"}
        for a,b in [(source_name,target_name),(target_name,source_name)]:
            t=combos[b];hit=t.cid_a.isin(dc)&t.cid_b.isin(dc)&t.cellosaurus_accession.isin(cc)
            support=hit.to_numpy().copy()
            for arm in "ab":
                keys=pd.MultiIndex.from_arrays([t["cid_"+arm],t.cellosaurus_accession])
                limits=ranges[a].reindex(keys)
                support&=(t["dose_"+arm].to_numpy()>=limits["min"].to_numpy())&(t["dose_"+arm].to_numpy()<=limits["max"].to_numpy())
            overlaps.append(dict(source=a,target=b,shared_cids=len(dc),shared_cell_accessions=len(cc),
                shared_drug_names=len(set(ds.drug)&set(dt.drug)),shared_cell_names=len(set(cs.cell_line)&set(ct.cell_line)),
                target_rows=len(t),identity_supported_rows=int(hit.sum()),identity_and_range_rows=int(support.sum()),
                identity_supported_pair_cells=len(t.loc[hit,["pair_id","cell_line"]].drop_duplicates()),
                range_supported_pair_cells=len(t.loc[support,["pair_id","cell_line"]].drop_duplicates()),
                supported_mass=float(support.mean())))
    pd.DataFrame(profiles).to_csv(out/"profiles.csv",index=False)
    pd.DataFrame(overlaps).to_csv(out/"overlap.csv",index=False)
    pd.DataFrame(identities).to_csv(out/"identifiers.csv",index=False)
    write_json(out/"provenance.json",dict(almanac_raw_sha256=digest(almanac_raw),source_sha256=digest(source),
        excluded_almanac_drugs=excluded,identity_rule="Positive exact PubChem CID and exact Cellosaurus accession; no salt/parent or spelling-based equivalence inferred.",
        dose_support="Each endpoint within the source drug-cell observed concentration range; range overlap alone does not establish interpolation quality or response compatibility.",
        monotherapy="Full-screen availability is descriptive only; training features retain training-pair-only lookups. Repeated copied edges are not independent measurements.",
        transfer_status="Response-definition compatibility must additionally be established. Cross-screen predictions and standardized error gaps require retained source predictions; aggregate relative gains are insufficient."))
    print(pd.DataFrame(overlaps).to_string(index=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True);p.add_argument("--almanac-raw",required=True)
    p.add_argument("--source",default="data/raw/drugcomb/drugcomb_data_v1.4.csv")
    a=p.parse_args();run(a.output,a.almanac_raw,a.source)
