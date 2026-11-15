"""R12: Is the learnable beyond-additive signal for unseen pairs explained by the synergy
reference model's own non-additive dependence on the two monotherapies?

Every synergy score is S_abc = E_abc - Ref(m_ac, m_bc). Bliss: Ref = I_a + I_b - I_a I_b
(non-additive part -I_a I_b); HSA: Ref = max(I_a, I_b) (non-additive part |I_a - I_b|/2);
Loewe: an implicit function of both Hill curves. If E_abc carried no pair-specific structure,
S would still contain a non-additive, drug-level-predictable term -Ref(m_ac, m_bc).

Test: per fold, estimate drug-in-cell monotherapy summaries (DrugComb ri = relative
inhibition, ic50) from TRAINING+VALIDATION blocks only, build symmetric non-additive features
of the two drugs' monotherapies, and compare contrast R^2 of
  mono_lin : additive ridge + per-cell ridge on 6 symmetric monotherapy features
  mono_mlp : additive ridge + small MLP on the same symmetric features (cell one-hot appended)
  knn_resid: identity-only neighbourhood CF (R03/R09 learner)
  knn_after_mono: knn on residuals left by mono_mlp (does identity CF add beyond it?)
for DeepSynergy Loewe and DrugComb Loewe/Bliss/HSA/ZIP/S panels.
Run: python -I r12_reference_structure.py <summary_table_v1.4.csv>
"""
import sys, json, time, itertools, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd, torch
from threadpoolctl import threadpool_limits
from common import load_oneil, RES, ridge_additive, proj_resid, metrics, split
import pairlearners as L

torch.set_num_threads(4); threadpool_limits(4)
csv = Path(sys.argv[1]); t0 = time.time()
d = load_oneil(); H, aa, bb, folds = d['H'], d['aa'], d['bb'], d['folds']; drugs = d['drugs']; cells = d['cells']
N, C = d['Y'].shape; nd = H.shape[1]
cols = ['drug_row', 'drug_col', 'cell_line_name', 'ri_row', 'ri_col', 'ic50_row', 'ic50_col']
dc = pd.read_csv(csv, usecols=cols, low_memory=False)
nrm = lambda s: s.astype(str).str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
dmap = {''.join(ch for ch in x.upper() if ch.isalnum()): x for x in drugs}; cmap = {''.join(ch for ch in x.upper() if ch.isalnum()): x for x in cells}
dc['r'] = nrm(dc.drug_row); dc['c'] = nrm(dc.drug_col); dc['cl'] = nrm(dc.cell_line_name)
dc = dc[dc.r.isin(dmap) & dc.c.isin(dmap) & dc.cl.isin(cmap)].copy()
for k in ['ri_row', 'ri_col', 'ic50_row', 'ic50_col']:
    dc[k] = pd.to_numeric(dc[k], errors='coerce')
pidx = {p: i for i, p in enumerate(d['pairs'])}; cidx = {c: j for j, c in enumerate(cells)}; didx = {x: i for i, x in enumerate(drugs)}
dc['pa'] = [min(dmap[x], dmap[y]) for x, y in zip(dc.r, dc.c)]; dc['pb'] = [max(dmap[x], dmap[y]) for x, y in zip(dc.r, dc.c)]
dc = dc[[(a, b) in pidx for a, b in zip(dc.pa, dc.pb)]]
dc['pair'] = [pidx[(a, b)] for a, b in zip(dc.pa, dc.pb)]; dc['cell'] = dc.cl.map(cmap).map(cidx)
# long table of monotherapy observations: (pair, drug, cell, ri, log ic50)
mono = pd.concat([pd.DataFrame(dict(pair=dc.pair, drug=dc.r.map(dmap).map(didx), cell=dc.cell, ri=dc.ri_row, lic=np.log10(dc.ic50_row.clip(lower=1e-6)))),
                  pd.DataFrame(dict(pair=dc.pair, drug=dc.c.map(dmap).map(didx), cell=dc.cell, ri=dc.ri_col, lic=np.log10(dc.ic50_col.clip(lower=1e-6))))])
mono = mono.dropna()
P = np.load(RES / 'r08_oneil_drugcomb_panels.npz')
panels = {'deepsynergy_loewe': d['Y']}
for m in ['synergy_loewe', 'synergy_bliss', 'synergy_hsa', 'synergy_zip', 'S']:
    panels['drugcomb_' + m.replace('synergy_', '')] = P[f'{m}_mean']


def mono_tables(allowed_pairs):
    m = mono[mono.pair.isin(set(allowed_pairs))].groupby(['drug', 'cell'])[['ri', 'lic']].mean()
    RI = np.full((nd, C), np.nan); LIC = np.full((nd, C), np.nan)
    for (dg, c), r in m.iterrows():
        RI[dg, c] = r.ri; LIC[dg, c] = r.lic
    for A in (RI, LIC):  # fill rare gaps with the drug's mean over cells (still training-only)
        rowm = np.nanmean(A, 1, keepdims=True); A[np.isnan(A)] = np.broadcast_to(rowm, A.shape)[np.isnan(A)]
    return RI / 100.0, LIC


def sym_feats(RI, LIC):
    """Per (pair, cell) symmetric NON-additive features (additive parts are absorbed by ridge)."""
    ra, rb, la, lb = RI[aa], RI[bb], LIC[aa], LIC[bb]
    return np.stack([ra * rb, np.abs(ra - rb), np.maximum(ra, rb), la * lb, np.abs(la - lb), np.maximum(la, lb)], -1)  # N x C x 6


def fit_mono_lin(F, R, tr, va):
    best = None
    for lam in [.1, 1, 10, 100, 1000]:
        pred = np.zeros_like(R)
        for c in range(C):
            X = F[:, c, :]; mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-9; Xs = (X - mu) / sd
            b = np.linalg.solve(Xs[tr].T @ Xs[tr] + lam * np.eye(X.shape[1]), Xs[tr].T @ (R[tr, c] - R[tr, c].mean()))
            pred[:, c] = R[tr, c].mean() + Xs @ b
        v = np.mean((pred[va] - R[va]) ** 2)
        if best is None or v < best[0]:
            best = (v, pred)
    return best[1]


def fit_mono_mlp(F, R, tr, va, seed):
    torch.manual_seed(seed)
    X = F.reshape(N * C, -1); mu = X.reshape(N, C, -1)[tr].reshape(-1, X.shape[1]).mean(0); sd = X.reshape(N, C, -1)[tr].reshape(-1, X.shape[1]).std(0) + 1e-9
    Xs = (X - mu) / sd; Xs = np.concatenate([Xs, np.tile(np.eye(C), (N, 1))], 1)
    Xt = torch.tensor(Xs, dtype=torch.float32).reshape(N, C, -1); sc = float(R[tr].std())
    y = torch.tensor(R / sc, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[-1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=.003, weight_decay=1e-4); best = np.inf; state = None; since = 0
    for ep in range(1500):
        opt.zero_grad(); loss = ((net(Xt[tr]).squeeze(-1) - y[tr]) ** 2).mean(); loss.backward(); opt.step()
        if ep % 20 == 0:
            with torch.no_grad():
                vl = float(((net(Xt[va]).squeeze(-1) - y[va]) ** 2).mean())
            if vl < best - 1e-6:
                best = vl; state = copy.deepcopy(net.state_dict()); since = 0
            else:
                since += 1
            if since >= 10:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        return net(Xt).squeeze(-1).numpy() * sc


def knn_pred(R, tr, va, te):
    M, W = L.residual_rows(R, tr); best = None
    for sh, k in itertools.product([2, 5, 20], [3, 5, 10, 20]):
        S = L.drug_sim_from_resid(M, W, sh); v = np.mean((L.knn(S, M, W, va, k) - R[va]) ** 2)
        if best is None or v < best[0]:
            best = (v, sh, k)
    return L.knn(L.drug_sim_from_resid(M, W, best[1]), M, W, te, best[2])


rows = []
for name, Y in panels.items():
    for f in range(5):
        tr, va, te = split(folds, f); h = H[te]
        RI, LIC = mono_tables(np.r_[tr, va]); F = sym_feats(RI, LIC)
        lam, beta = ridge_additive(H, Y, tr, va); f0 = H @ beta; R = Y - f0
        preds = {'ridge': f0[te]}
        ml = fit_mono_lin(F, R, tr, va); preds['mono_lin'] = f0[te] + ml[te]
        mm = np.mean([fit_mono_mlp(F, R, tr, va, s) for s in [622, 1622, 2622]], 0); preds['mono_mlp'] = f0[te] + mm[te]
        preds['knn_resid'] = f0[te] + knn_pred(R, tr, va, te)
        R2 = R - mm; preds['knn_after_mono'] = f0[te] + mm[te] + knn_pred(R2, tr, va, te)
        for k, p in preds.items():
            m = metrics(p, Y[te], h); rows.append(dict(panel=name, fold=f, model=k, mse=m['mse'], sse=m['contrast_sse'], ss=m['contrast_ss']))
    print(name, round(time.time() - t0, 1), flush=True)
R_ = pd.DataFrame(rows); R_.to_csv(RES / 'r12_runs.csv', index=False)
g = R_.groupby(['panel', 'model']).agg(mse=('mse', 'mean'), sse=('sse', 'sum'), ss=('ss', 'sum'))
g['pooled_cr2'] = 1 - g.sse / g.ss
print(g[['mse', 'pooled_cr2']].round(4).to_string())
(RES / 'r12_summary.json').write_text(g.reset_index().to_json(orient='records', indent=1))
