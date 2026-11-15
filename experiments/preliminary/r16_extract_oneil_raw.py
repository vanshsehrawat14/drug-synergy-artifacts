"""R16: extract the O'Neil (study_name == 'ONEIL') dose-level rows from the DrugComb v1.4 raw file
(Zenodo 18449193, drugcomb_data_v1.4.csv, CC-BY-4.0) into a compact NPZ under data/processed/.

Run as: python -I r16_extract_oneil_raw.py <path to drugcomb_data_v1.4.csv>
Outputs: data/processed/drugcomb_oneil/oneil_dose_level.npz + manifest.json
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd

src = Path(sys.argv[1])
out = Path(__file__).resolve().parents[2] / 'data' / 'processed' / 'drugcomb_oneil'
out.mkdir(parents=True, exist_ok=True)
md5, sha = hashlib.md5(), hashlib.sha256()
with open(src, 'rb') as fh:
    for chunk in iter(lambda: fh.read(1 << 24), b''):
        md5.update(chunk); sha.update(chunk)
assert md5.hexdigest() == 'f5f071de351ec38ffe29c328519a7be5', 'md5 mismatch vs Zenodo record'
keep, studies, nrows = [], {}, 0
cols = ['block_id', 'conc_r', 'conc_c', 'inhibition', 'drug_row', 'drug_col', 'conc_r_unit', 'conc_c_unit', 'cell_line_name', 'study_name']
for ch in pd.read_csv(src, usecols=cols, chunksize=2_000_000, low_memory=False):
    nrows += len(ch)
    for k, v in ch.study_name.value_counts().items():
        studies[k] = studies.get(k, 0) + int(v)
    keep.append(ch[ch.study_name == 'ONEIL'])
df = pd.concat(keep, ignore_index=True)
assert set(df.conc_r_unit) == {'uM'} and set(df.conc_c_unit) == {'uM'}
drugs = sorted(set(df.drug_row) | set(df.drug_col)); cells = sorted(set(df.cell_line_name))
di = {d: i for i, d in enumerate(drugs)}; ci = {c: i for i, c in enumerate(cells)}
np.savez_compressed(out / 'oneil_dose_level.npz', block_id=df.block_id.to_numpy(np.int64),
                    conc_r=df.conc_r.to_numpy(float), conc_c=df.conc_c.to_numpy(float), inhibition=df.inhibition.to_numpy(float),
                    drug_row=df.drug_row.map(di).to_numpy(np.int16), drug_col=df.drug_col.map(di).to_numpy(np.int16),
                    cell=df.cell_line_name.map(ci).to_numpy(np.int16))
man = dict(source='Zenodo 18449193 drugcomb_data_v1.4.csv (CC-BY-4.0)', source_md5=md5.hexdigest(), source_sha256=sha.hexdigest(),
           source_rows=nrows, studies=studies, oneil_rows=len(df), oneil_blocks=int(df.block_id.nunique()),
           rows_per_block=df.groupby('block_id').size().value_counts().to_dict(), n_inhibition_nan=int(df.inhibition.isna().sum()),
           drugs=drugs, cells=cells)
(out / 'manifest.json').write_text(json.dumps(man, indent=1, default=str))
print(json.dumps({k: v for k, v in man.items() if k not in ('drugs', 'cells')}, indent=1, default=str))
