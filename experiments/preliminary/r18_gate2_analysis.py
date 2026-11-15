"""R18: Gate 2 analyses under the frozen protocol docs/gate2_protocol.md (SHA-256 cc3051ab...).
Run: python -I r18_gate2_analysis.py <summary_table_v1.4.csv>   (needs r16 extraction and r17 block scores)
Outputs: results/preliminary/r18_gate2.json, results/preliminary/r18_units.npz
"""
import sys, json, hashlib, itertools, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from common import load_oneil, RES, ROOT, RAW, ridge_additive, proj_resid, metrics, split
import pairlearners as L

PROTO_SHA = 'cc3051abd6e59205e5990088f66830bf050084773dadd19bdad799f7fe91c88b'
assert hashlib.sha256((ROOT / 'docs' / 'gate2_protocol.md').read_bytes()).hexdigest() == PROTO_SHA
t0 = time.time(); out = {}
d = load_oneil(); H, aa, bb, folds = d['H'], d['aa'], d['bb'], d['folds']
N, C = d['Y'].shape
man = json.loads((RAW / 'manifest.json').read_text())
blk = pd.read_csv(RES / 'r17_block_scores.csv.gz')
sm = pd.read_csv(sys.argv[1], usecols=['block_id', 'synergy_loewe', 'synergy_bliss', 'synergy_hsa', 'synergy_zip', 'S', 'css'], low_memory=False)
for c in sm.columns[1:]:
    sm[c] = pd.to_numeric(sm[c], errors='coerce')
blk = blk.merge(sm, on='block_id', how='left')
assert blk.synergy_loewe.notna().all() and blk.synergy_bliss.notna().all()

# ---------------- P1 identity checks
out['P1'] = dict(n_blocks=len(blk),
                 bliss_max_abs_diff=float(((blk['T'] - blk.ref_bliss) - blk.synergy_bliss).abs().max()),
                 hsa_max_abs_diff=float(((blk['T'] - blk.ref_hsa) - blk.synergy_hsa).abs().max()))
g = blk.assign(md=blk.synergy_loewe - blk.synergy_bliss).groupby(['drug_row', 'drug_col', 'cell']).md
out['P1']['loewe_minus_bliss_max_within_unit_sd'] = float(g.std().max())
out['P1']['units_oriented'] = int(g.ngroups)

# ---------------- unit-level panels (583 x 39), mean over the unit's ONEIL blocks
nrm = lambda x: ''.join(ch for ch in x.upper() if ch.isalnum())
dmap = {nrm(x): x for x in d['drugs']}; cmap = {nrm(x): x for x in d['cells']}
rawd = [dmap[nrm(x)] for x in man['drugs']]; rawc = [cmap[nrm(x)] for x in man['cells']]
pidx = {p: i for i, p in enumerate(d['pairs'])}; cidx = {c: j for j, c in enumerate(d['cells'])}
nr, ncl = blk.drug_row.map(lambda i: rawd[i]), blk.drug_col.map(lambda i: rawd[i])
blk['p'] = [pidx[(min(x, y), max(x, y))] for x, y in zip(nr, ncl)]; blk['c'] = blk.cell.map(lambda i: cidx[rawc[i]])
blk['R_L_DC'] = blk['T'] - blk.synergy_loewe
cols = dict(T='T', R_B='ref_bliss', R_H='ref_hsa', R_L_DC='R_L_DC', R_L_gen='ref_loewe_gen', R_L_emul='ref_loewe_sf2',
            R_L_exist='ref_loewe_exist', R_L_hsafb='ref_loewe_sf2_hsafb', R_B_fit='ref_bliss_fit',
            R_B_fit_h1='ref_bliss_fit_h1', R_B_fit_d100='ref_bliss_fit_d100', R_B_fit_ideal='ref_bliss_fit_ideal',
            R_L_gen_h1='ref_loewe_gen_h1', R_L_gen_d100='ref_loewe_gen_d100', R_L_gen_ideal='ref_loewe_gen_ideal',
            Y_L='synergy_loewe', Y_B='synergy_bliss', Y_H='synergy_hsa', Y_Z='synergy_zip', Y_S='S', Y_CSS='css',
            n_fb_emul='n_fallback')
gm = blk.groupby(['p', 'c'])
U = {}
for k, v in cols.items():
    s = gm[v].mean(); A = np.full((N, C), np.nan); A[s.index.get_level_values(0), s.index.get_level_values(1)] = s.to_numpy(); U[k] = A
nblk = np.zeros((N, C)); s = gm.size(); nblk[s.index.get_level_values(0), s.index.get_level_values(1)] = s.to_numpy()
fb_exist = blk.assign(fb=(blk.ref_loewe_exist - blk.ref_loewe_gen).abs() > 1e-6).groupby(['p', 'c']).fb.mean()
FB = np.zeros((N, C)); FB[fb_exist.index.get_level_values(0), fb_exist.index.get_level_values(1)] = fb_exist.to_numpy()
out['coverage'] = {k: float(np.isfinite(v).mean()) for k, v in U.items()}
R08 = np.load(RES / 'r08_oneil_drugcomb_panels.npz')
out['max_abs_diff_vs_r08_panel'] = {k: float(np.nanmax(np.abs(U[k] - R08[m]))) for k, m in
                                    [('Y_L', 'synergy_loewe_mean'), ('Y_B', 'synergy_bliss_mean'), ('Y_H', 'synergy_hsa_mean'), ('Y_S', 'S_mean')]}
for k in ['R_L_gen', 'R_L_emul', 'R_L_exist', 'R_L_hsafb', 'R_B_fit', 'R_L_gen_h1', 'R_L_gen_d100', 'R_L_gen_ideal',
          'R_B_fit_h1', 'R_B_fit_d100', 'R_B_fit_ideal']:   # rare non-finite fits: fill with the unit's Bliss reference (logged)
    bad = ~np.isfinite(U[k]); out.setdefault('filled_nonfinite', {})[k] = int(bad.sum()); U[k][bad] = U['R_B'][bad]
assert all(np.isfinite(U[k]).all() for k in ['T', 'R_B', 'R_H', 'R_L_DC', 'Y_L', 'Y_B', 'Y_H'])

M = dict(M_DC=U['R_B'] - U['R_L_DC'], M_alg=U['R_B'] - U['R_L_gen'], M_impl=U['R_L_gen'] - U['R_L_DC'],
         M_emul=U['R_B'] - U['R_L_emul'], M_exist=U['R_B'] - U['R_L_exist'], M_fb_exist=U['R_L_gen'] - U['R_L_exist'],
         M_hsafb=U['R_B'] - U['R_L_hsafb'],
         M_alg_fit=U['R_B_fit'] - U['R_L_gen'], M_alg_h1=U['R_B_fit_h1'] - U['R_L_gen_h1'],
         M_alg_d100=U['R_B_fit_d100'] - U['R_L_gen_d100'], M_alg_ideal=U['R_B_fit_ideal'] - U['R_L_gen_ideal'])
assert np.allclose(M['M_DC'], U['Y_L'] - U['Y_B'], atol=1e-6)


def pooled(fn):
    sse = ss = 0.0
    for f in range(5):
        tr, va, te = split(folds, f); a, b = fn(te, H[te]); sse += a; ss += b
    return sse, ss


def cr2(X, Y):
    sse, ss = pooled(lambda te, h: (metrics(X[te], Y[te], h)['contrast_sse'], metrics(X[te], Y[te], h)['contrast_ss']))
    return float(1 - sse / ss)


def css(X):
    return pooled(lambda te, h: (float(np.sum(proj_resid(X[te], h) ** 2)), 0.0))[0]


def nonadd_share(X):
    tot = pooled(lambda te, h: (float(np.sum((X[te] - X[te].mean(0)) ** 2)), 0.0))[0]
    return float(css(X) / tot)


ssYL = css(U['Y_L'])
out['P2'] = dict(cr2_M_DC=cr2(M['M_DC'], U['Y_L']), share_M_DC=css(M['M_DC']) / ssYL, share_Y_B=css(U['Y_B']) / ssYL,
                 cross_term_share=1 - css(M['M_DC']) / ssYL - css(U['Y_B']) / ssYL,
                 nonadd_share={k: nonadd_share(v) for k, v in [('Y_L', U['Y_L']), ('Y_B', U['Y_B']), ('T', U['T']), ('M_DC', M['M_DC'])]})
s0 = 0.3072
comp = {k: dict(cr2_vs_Y_L=cr2(v, U['Y_L']), contrast_ss_share_of_Y_L=css(v) / ssYL, nonadd_share=nonadd_share(v),
                corr_contrast_with_M_DC=None) for k, v in M.items()}
for k, v in M.items():   # correlation of contrast components with M_DC's contrast
    a = b = ab = 0.0
    for f in range(5):
        tr, va, te = split(folds, f); x = proj_resid(v[te], H[te]).ravel(); y = proj_resid(M['M_DC'][te], H[te]).ravel()
        a += x @ x; b += y @ y; ab += x @ y
    comp[k]['corr_contrast_with_M_DC'] = float(ab / np.sqrt(a * b + 1e-300))
out['components'] = comp
passes = {k: comp[k]['cr2_vs_Y_L'] >= 0.5 * s0 for k in ['M_alg', 'M_impl']}
out['P3'] = dict(s0=s0, threshold=0.5 * s0, cr2_M_DC=comp['M_DC']['cr2_vs_Y_L'], component_pass=passes,
                 gate2_pass=bool(comp['M_DC']['cr2_vs_Y_L'] >= 0.5 * s0 and any(passes.values())))
# where does M_impl live? share of its contrast energy in units where the exist-rule fallback fired in any block
Pimpl = {}
num = den = 0.0
for f in range(5):
    tr, va, te = split(folds, f); x = proj_resid(M['M_impl'][te], H[te]); den += float(np.sum(x ** 2)); num += float(np.sum((x ** 2)[FB[te] > 0]))
out['P3']['M_impl_energy_share_in_fallback_units'] = num / den
out['P3']['fraction_units_with_fallback'] = float((FB > 0).mean())
out['P3']['fraction_blocks_with_fallback_exist_rule'] = float(((blk.ref_loewe_exist - blk.ref_loewe_gen).abs() > 1e-6).mean())
out['fidelity_vs_DC_block'] = {k: float(np.corrcoef(blk[v], blk.R_L_DC)[0, 1]) for k, v in
                               [('emul_nleqslv_port', 'ref_loewe_sf2'), ('exist_rule', 'ref_loewe_exist'), ('generalised', 'ref_loewe_gen'), ('hsa_fallback', 'ref_loewe_sf2_hsafb')]}
out['fidelity_vs_DC_block_exact_frac'] = {k: float(((blk[v] - blk.R_L_DC).abs() < 0.01).mean()) for k, v in
                                          [('emul_nleqslv_port', 'ref_loewe_sf2'), ('exist_rule', 'ref_loewe_exist'), ('generalised', 'ref_loewe_gen')]}
print(json.dumps({k: out[k] for k in ['P1', 'P2', 'P3']}, indent=1), flush=True)


# ---------------- learners (R09 procedure)
def learners(Y, f):
    tr, va, te = split(folds, f)
    lam, beta = ridge_additive(H, Y, tr, va); f0 = H @ beta; R = Y - f0
    Mr, W = L.residual_rows(R, tr); best = None
    for sh, k in itertools.product([2, 5, 20], [3, 5, 10, 20]):
        S = L.drug_sim_from_resid(Mr, W, sh); v = np.mean((L.knn(S, Mr, W, va, k) - R[va]) ** 2)
        if best is None or v < best[0]:
            best = (v, sh, k)
    pk = f0[te] + L.knn(L.drug_sim_from_resid(Mr, W, best[1]), Mr, W, te, best[2])
    best = None
    for sh, lam2 in itertools.product([2, 5, 20], [.1, 1, 10, 100]):
        S = L.drug_sim_from_resid(Mr, W, sh); ev, evec = np.linalg.eigh(S); Kd = (evec * np.clip(ev, 0, None)) @ evec.T
        v = np.mean((L.krr(Kd, R, tr, va, lam2) - R[va]) ** 2)
        if best is None or v < best[0]:
            best = (v, lam2, Kd)
    pr = f0[te] + L.krr(best[2], R, tr, te, best[1])
    return te, dict(knn=pk, krr=pr)


def run(Y, invariant_ref=None, T_preds=None):
    """Pooled contrast SSE/SS for naive learners on Y; if T_preds given, also the invariant protocol T_hat - R."""
    res = {}
    for f in range(5):
        te, p = learners(Y, f) if T_preds is None else (T_preds[f][0], {k: v - invariant_ref[T_preds[f][0]] for k, v in T_preds[f][1].items()})
        for k, v in p.items():
            m = metrics(v, Y[te], H[te]); r = res.setdefault(k, [0.0, 0.0]); r[0] += m['contrast_sse']; r[1] += m['contrast_ss']
    return {k: dict(sse=v[0], ss=v[1], cr2=1 - v[0] / v[1]) for k, v in res.items()}


# P4 Bliss-true null
z = np.load(RAW / 'oneil_dose_level.npz')
raw = pd.DataFrame({k: z[k] for k in ['block_id', 'conc_r', 'conc_c', 'inhibition', 'drug_row', 'drug_col', 'cell']})
comb = raw[(raw.conc_r > 0) & (raw.conc_c > 0)]
gv = comb.groupby(['drug_row', 'drug_col', 'cell', 'conc_r', 'conc_c']).inhibition.agg(['var', 'count'])
gv = gv[gv['count'] > 1]
sig2_cell = float((gv['var'] * (gv['count'] - 1)).sum() / (gv['count'] - 1).sum())
gT = blk.groupby(['drug_row', 'drug_col', 'cell'])['T'].agg(['var', 'count']); gT = gT[gT['count'] > 1]
sig2_T = float((gT['var'] * (gT['count'] - 1)).sum() / (gT['count'] - 1).sum())
out['P4'] = dict(sigma2_cell=sig2_cell, sigma2_blockT=sig2_T, runs={})
for tag, var_u in [('cell_independent', sig2_cell / (16 * nblk)), ('block_T', sig2_T / nblk)]:
    eps = np.random.default_rng(2).normal(0, np.sqrt(var_u))
    for name, Y in [('loewe_sim', M['M_DC'] + eps), ('bliss_sim', eps)]:
        out['P4']['runs'][f'{tag}:{name}'] = run(Y)
        print(tag, name, {k: round(v['cr2'], 4) for k, v in out['P4']['runs'][f'{tag}:{name}'].items()}, round(time.time() - t0), flush=True)

# P5 metric-invariant view
Tp = [learners(U['T'], f) for f in range(5)]
out['P5'] = {}
refs = dict(T=np.zeros((N, C)), Y_B=U['R_B'], Y_H=U['R_H'], Y_L=U['R_L_DC'])
for zname, ref in refs.items():
    Z = U['T'] if zname == 'T' else U[zname]
    naive = run(Z) if zname != 'T' else {k: dict(sse=0, ss=0, cr2=0) for k in ['knn', 'krr']}
    inv = run(Z, ref, Tp)
    sse_ref = ss_ref = 0.0
    for f in range(5):
        tr, va, te = split(folds, f); m = metrics(-ref[te], Z[te], H[te]); sse_ref += m['contrast_sse']; ss_ref += m['contrast_ss']
    out['P5'][zname] = dict(contrast_ss=css(Z), naive=naive if zname != 'T' else inv, invariant=inv,
                            reference_only=dict(sse=sse_ref, ss=ss_ref, cr2=1 - sse_ref / ss_ref))
    print(zname, 'SS', round(css(Z)), 'naive', {k: round(v['cr2'], 4) for k, v in out['P5'][zname]['naive'].items()},
          'inv', {k: round(v['cr2'], 4) for k, v in inv.items()}, 'refonly', round(1 - sse_ref / ss_ref, 4), flush=True)
# ---------------- exploratory (added after the protocol was frozen; not part of the Gate 2 decision)
# S = CSS - RI_row - RI_col exactly (checked on the summary table), so S's contrast equals CSS's contrast.
# Does the Bliss expectation, a zero-parameter function of monotherapy, predict the non-additive part of the raw
# response T and of CSS/S?
out['exploratory'] = dict(cr2_RB_predicts_T=cr2(U['R_B'], U['T']), cr2_RB_predicts_CSS=cr2(U['R_B'], U['Y_CSS']),
                          cr2_RB_predicts_S=cr2(U['R_B'], U['Y_S']), cr2_RH_predicts_T=cr2(U['R_H'], U['T']),
                          cr2_RLDC_predicts_T=cr2(U['R_L_DC'], U['T']),
                          contrast_ss=dict(T=css(U['T']), R_B=css(U['R_B']), R_H=css(U['R_H']), R_L_DC=css(U['R_L_DC']),
                                           Y_S=css(U['Y_S']), Y_CSS=css(U['Y_CSS'])))
print('exploratory', json.dumps(out['exploratory'], indent=1), flush=True)
np.savez_compressed(RES / 'r18_units.npz', **U, **M, nblk=nblk, FB=FB)
(RES / 'r18_gate2.json').write_text(json.dumps(out, indent=1, default=float))
print('done', round(time.time() - t0, 1))
