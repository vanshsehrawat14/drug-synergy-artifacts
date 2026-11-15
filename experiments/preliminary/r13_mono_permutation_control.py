"""R13: permutation control for R12. Reassign each drug the monotherapy profile of another drug
(one fixed permutation of drugs, applied to all cells), so features stay drug-consistent
identifiers but lose their pharmacological meaning. If mono_mlp's contrast skill on DrugComb
Loewe / S survives, R12's effect is identifier capacity, not monotherapy-reference structure.
Run: python -I r13_mono_permutation_control.py <summary_table_v1.4.csv>"""
import sys, json, runpy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
src = Path(__file__).resolve().parent / 'r12_reference_structure.py'
code = src.read_text()
# reuse R12 definitions without running its experiment loop
head = code.split('rows = []\nfor name, Y in panels.items():')[0]
g = {'__name__': 'r12defs', '__file__': str(src)}
sys.argv = [str(src), sys.argv[1]]
exec(compile(head, str(src), 'exec'), g)
from common import RES, ridge_additive, metrics, split
rows = []
for name in ['drugcomb_loewe', 'drugcomb_S']:
    Y = g['panels'][name]
    for f in range(5):
        tr, va, te = split(g['folds'], f); h = g['H'][te]
        RI, LIC = g['mono_tables'](np.r_[tr, va])
        lam, beta = ridge_additive(g['H'], Y, tr, va); f0 = g['H'] @ beta; R = Y - f0
        for perm_seed in [None, 1, 2, 3]:
            if perm_seed is None:
                RIp, LICp = RI, LIC
            else:
                pi = np.random.default_rng(perm_seed).permutation(g['nd']); RIp, LICp = RI[pi], LIC[pi]
            F = g['sym_feats'](RIp, LICp)
            mm = np.mean([g['fit_mono_mlp'](F, R, tr, va, s) for s in [622, 1622]], 0)
            m = metrics(f0[te] + mm[te], Y[te], h)
            rows.append(dict(panel=name, fold=f, perm=str(perm_seed), sse=m['contrast_sse'], ss=m['contrast_ss']))
        print(name, f, flush=True)
R_ = pd.DataFrame(rows); R_.to_csv(RES / 'r13_runs.csv', index=False)
s = R_.groupby(['panel', 'perm']).agg(sse=('sse', 'sum'), ss=('ss', 'sum')); s['pooled_cr2'] = 1 - s.sse / s.ss
print(s.pooled_cr2.round(4).to_string()); (RES / 'r13_summary.json').write_text(s.reset_index().to_json(orient='records', indent=1))
