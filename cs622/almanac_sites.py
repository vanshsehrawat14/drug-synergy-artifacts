"""Pair -> NCI screening site for DrugComb ALMANAC rows.

python -m cs622.almanac_sites --nci ComboDrugGrowth_Nov2017.csv --output results/almanac_validation/pair_screener.csv

Matches DrugComb combination rows to the NCI ALMANAC file (DTP, ComboDrugGrowth_Nov2017.csv) on cell,
both concentrations (uM) and inhibition = 100 - PercentGrowthNoTZ, rounded to 0.01. Each pair takes
its majority screener; `purity` is that screener's share of matched rows. Ties (purity 0.5, 5 of
5,141 pairs) are broken arbitrarily; results/almanac_validation/pair_screener.csv is the map the
site check used.
"""
import argparse

import numpy as np
import pandas as pd

from .splits import normalize_name, pair_id

COLUMNS = ["block_id", "conc_r", "conc_c", "inhibition", "drug_row", "drug_col", "cell_line_name", "study_name"]


def site_map(nci, drugcomb, chunksize=1_000_000):
    raw = pd.read_csv(nci, usecols=["SCREENER", "CONC1", "CONCUNIT1", "CONC2", "CONCUNIT2", "PERCENTGROWTHNOTZ", "CELLNAME", "NSC2"])
    raw = raw[raw.NSC2.notna() & (raw.CONCUNIT1 == "M") & (raw.CONCUNIT2 == "M")].copy()
    raw["cell"] = raw.CELLNAME.astype(str).map(normalize_name)
    c1, c2 = raw.CONC1 * 1e6, raw.CONC2 * 1e6
    raw["lo"], raw["hi"], raw["inh"] = np.minimum(c1, c2).round(6), np.maximum(c1, c2).round(6), (100 - raw.PERCENTGROWTHNOTZ).round(2)
    parts = [c[(c.study_name == "ALMANAC") & (c.conc_r > 0) & (c.conc_c > 0)]
             for c in pd.read_csv(drugcomb, usecols=COLUMNS, chunksize=chunksize, low_memory=False)]
    dc = pd.concat(parts, ignore_index=True)
    dc["cell"] = dc.cell_line_name.astype(str).map(normalize_name)
    dc["lo"], dc["hi"], dc["inh"] = np.minimum(dc.conc_r, dc.conc_c).round(6), np.maximum(dc.conc_r, dc.conc_c).round(6), dc.inhibition.round(2)
    dc["pair_id"] = [pair_id(a, b) for a, b in zip(dc.drug_row, dc.drug_col)]
    keys = ["cell", "lo", "hi", "inh"]
    m = dc[keys + ["pair_id"]].merge(raw[keys + ["SCREENER"]].drop_duplicates(keys + ["SCREENER"]), on=keys).dropna(subset=["SCREENER"])
    shares = m.groupby("pair_id").SCREENER.value_counts(normalize=True).rename("share").reset_index()
    top = shares.sort_values(["pair_id", "share"], ascending=[True, False]).drop_duplicates("pair_id")
    print(f"{len(m):,} of {len(dc):,} combination rows matched; {len(top):,} pairs mapped")
    return top.rename(columns={"SCREENER": "screener", "share": "purity"})[["pair_id", "screener", "purity"]]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--nci", required=True)
    p.add_argument("--drugcomb", default="data/raw/drugcomb/drugcomb_data_v1.4.csv")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    site_map(a.nci, a.drugcomb).to_csv(a.output, index=False)


if __name__ == "__main__":
    main()
