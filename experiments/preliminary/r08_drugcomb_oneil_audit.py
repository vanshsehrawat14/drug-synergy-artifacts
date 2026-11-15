"""R08: Locate the O'Neil screen inside DrugComb v1.4 (Zenodo 11102665, CC-BY-4.0) and build
alternative-metric and replicate-block panels aligned to the DeepSynergy 583 x 39 panel.

Run as: python -I r08_drugcomb_oneil_audit.py <path-to-summary_table_v1.4.csv>
(The downloaded CSV is treated as untrusted data: read with pandas only.)
Outputs: results/preliminary/r08_oneil_drugcomb_panels.npz, results/preliminary/r08_audit.json
"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import load_oneil, RES

csv = Path(sys.argv[1])
sha = hashlib.sha256(csv.read_bytes()).hexdigest()
d = load_oneil(); drugs = d['drugs']; cells = d['cells']; pairs = d['pairs']
cols = ['block_id', 'drug_row', 'drug_col', 'cell_line_name', 'synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S', 'css', 'ri_row', 'ri_col', 'ic50_row', 'ic50_col']
dc = pd.read_csv(csv, usecols=cols, low_memory=False)
norm = lambda s: s.astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
dc['r'] = norm(dc.drug_row); dc['c'] = norm(dc.drug_col); dc['cl'] = norm(dc.cell_line_name)
dmap = {''.join(ch for ch in x.upper() if ch.isalnum()): x for x in drugs}
cmap = {''.join(ch for ch in x.upper() if ch.isalnum()): x for x in cells}
audit = dict(sha256=sha, rows=len(dc), unmatched_oneil_drugs=[x for x in dmap if x not in set(dc.r) | set(dc.c)],
             unmatched_oneil_cells=[x for x in cmap if x not in set(dc.cl)])
sub = dc[dc.r.isin(dmap) & dc.c.isin(dmap) & dc.cl.isin(cmap)].copy()
sub['a'] = [min(dmap[x], dmap[y]) for x, y in zip(sub.r, sub.c)]; sub['b'] = [max(dmap[x], dmap[y]) for x, y in zip(sub.r, sub.c)]
sub['cell'] = sub.cl.map(cmap)
pairset = set(pairs); sub = sub[[(a, b) in pairset for a, b in zip(sub.a, sub.b)]]
for m in ['synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'S', 'css', 'ri_row', 'ri_col']:
    sub[m] = pd.to_numeric(sub[m], errors='coerce')
cnt = sub.groupby(['a', 'b', 'cell']).size()
audit.update(matched_rows=len(sub), matched_units=int(len(cnt)), blocks_per_unit=cnt.value_counts().sort_index().to_dict(),
             block_id_range=[int(sub.block_id.min()), int(sub.block_id.max())])
# Panels: per metric, mean over blocks; and first/second block for replicate analysis
pidx = {p: i for i, p in enumerate(pairs)}; cidx = {c: j for j, c in enumerate(cells)}
N, C = len(pairs), len(cells); panels = {}
sub = sub.sort_values('block_id')
sub['rep'] = sub.groupby(['a', 'b', 'cell']).cumcount()
for m in ['synergy_loewe', 'synergy_bliss', 'synergy_zip', 'synergy_hsa', 'S', 'css']:
    for tag, frame in [('mean', sub.groupby(['a', 'b', 'cell'])[m].mean().reset_index()),
                       ('rep0', sub[sub.rep == 0][['a', 'b', 'cell', m]]), ('rep1', sub[sub.rep == 1][['a', 'b', 'cell', m]])]:
        A = np.full((N, C), np.nan)
        for r in frame.itertuples(index=False):
            A[pidx[(r[0], r[1])], cidx[r[2]]] = r[3]
        panels[f'{m}_{tag}'] = A
# Agreement with DeepSynergy labels and replicate reliability
Y = d['Y']; L = panels['synergy_loewe_mean']; ok = ~np.isnan(L)
audit['coverage_of_583x39'] = float(ok.mean())
audit['corr_deepsynergy_vs_drugcomb_loewe_mean'] = float(np.corrcoef(Y[ok], L[ok])[0, 1])
rel = {}
for m in ['synergy_loewe', 'synergy_bliss', 'synergy_zip', 'synergy_hsa', 'S', 'css']:
    a, b = panels[f'{m}_rep0'], panels[f'{m}_rep1']; k = ~np.isnan(a) & ~np.isnan(b)
    if k.sum() > 100:
        rel[m] = dict(n=int(k.sum()), retest_corr=float(np.corrcoef(a[k], b[k])[0, 1]),
                      noise_var_single=float(np.var(a[k] - b[k]) / 2), total_var=float(np.var(np.r_[a[k], b[k]])),
                      noise_var_robust=float((np.median(np.abs((a[k] - b[k]) - np.median(a[k] - b[k]))) * 1.4826) ** 2 / 2))
audit['replicate_reliability'] = rel
mc = {}
names = ['synergy_loewe', 'synergy_bliss', 'synergy_zip', 'synergy_hsa', 'S']
for i in names:
    for j in names:
        if i < j:
            A_, B_ = panels[f'{i}_mean'], panels[f'{j}_mean']; k = ~np.isnan(A_) & ~np.isnan(B_)
            mc[f'{i}~{j}'] = float(np.corrcoef(A_[k], B_[k])[0, 1])
audit['metric_correlations'] = mc
np.savez_compressed(RES / 'r08_oneil_drugcomb_panels.npz', **panels)
(RES / 'r08_audit.json').write_text(json.dumps(audit, indent=1, default=str))
print(json.dumps(audit, indent=1, default=str))
