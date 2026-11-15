"""R19: Gate 2 protocol step. R08's DrugComb panels averaged every summary block matching the drug and cell names; 612 units
carry one non-ONEIL block, so the ONEIL-only Loewe panel (R18) differs. Recompute R12's mono_mlp contrast R^2 (s0)
on the ONEIL-only panel with R12's own code (features unchanged), plus the same model on Bliss for reference.
Run: python -I r19_s0_oneil_only.py <summary_table_v1.4.csv>
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
src = Path(__file__).resolve().parent / 'r12_reference_structure.py'
head = src.read_text().split('rows = []\nfor name, Y in panels.items():')[0]
g = {'__name__': 'r12defs', '__file__': str(src)}
sys.argv = [str(src), sys.argv[1]]
exec(compile(head, str(src), 'exec'), g)
from common import RES, ridge_additive, metrics, split
U = np.load(RES / 'r18_units.npz')
rows = []
for name, Y in [('oneil_only_loewe', U['Y_L']), ('oneil_only_bliss', U['Y_B'])]:
    for f in range(5):
        tr, va, te = split(g['folds'], f); h = g['H'][te]
        RI, LIC = g['mono_tables'](np.r_[tr, va]); F = g['sym_feats'](RI, LIC)
        lam, beta = ridge_additive(g['H'], Y, tr, va); f0 = g['H'] @ beta; R = Y - f0
        mm = np.mean([g['fit_mono_mlp'](F, R, tr, va, s) for s in [622, 1622, 2622]], 0)
        m = metrics(f0[te] + mm[te], Y[te], h); rows.append(dict(panel=name, fold=f, sse=m['contrast_sse'], ss=m['contrast_ss']))
    print(name, flush=True)
R_ = pd.DataFrame(rows); s = R_.groupby('panel').agg(sse=('sse', 'sum'), ss=('ss', 'sum')); s['pooled_cr2'] = 1 - s.sse / s.ss
print(s.round(4).to_string()); R_.to_csv(RES / 'r19_runs.csv', index=False)
(RES / 'r19_summary.json').write_text(s.reset_index().to_json(orient='records', indent=1))
