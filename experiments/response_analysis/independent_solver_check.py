"""Phase 3.1: independent float64 Brent solver versus theta bisection, native value and interval certificate.

Fixed stratified set (seed 20261107): up to 200 cases per regime plus every case on surfaces of pair-cells
whose top-5% membership changes under C1. Differences are attributed to domain resolution, solver
tolerance or native fallback; curve fits are fixed, so fit differences cannot arise here.
"""
import sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
sys.path.insert(0, '.')

OUT = '../handoff'
reg = pd.read_csv(f'{OUT}/loewe_regimes.csv')
cases = pd.read_parquet('results/neural/runs/scientific_native_full/surface_cases.parquet')
ver = pd.read_parquet('../../2026-11-06_validation/loewe_verify/cases_verified.parquet').set_index('case_id').reindex(cases.case_id)
cert = pd.read_parquet(f'{OUT}/root_certificates/certificates.parquet').set_index('case_id').reindex(cases.case_id)
blocks = pd.read_parquet('results/neural/runs/scientific_native_full/blocks.parquet', columns=['surface_id', 'pair_id', 'cell_line'])
units = pd.read_csv(f'{OUT}/decision/top5_N_to_C1_changes.csv')
changed_surfaces = set(blocks.astype(str).merge(units[['pair_id', 'cell_line']].astype(str), on=['pair_id', 'cell_line']).surface_id.astype(int))
rng = np.random.default_rng(20261107)
pick = set()
for r, g in reg.groupby('regime'):
    ix = g.index.to_numpy()
    pick.update(rng.choice(ix, min(200, len(ix)), replace=False).tolist())
pick.update(np.flatnonzero(cases.surface_id.isin(changed_surfaces).to_numpy()).tolist())
pick = np.array(sorted(pick))


def g_factory(row):
    arms = []
    for a in 'ab':
        b, c, d, e = (row[f'{a}_{k}'] for k in 'bcde')
        centre = np.log(e) if row[f'model_{a}'] == 'LL.4' else e
        arms.append((b, c, d, centre, np.log(row[f'dose_{a}'])))
    def g(y):
        s = -1.0
        for b, c, d, centre, ld in arms:
            s += np.exp(ld - centre - (np.log(abs(d - y)) - np.log(abs(y - c))) / b)
        return s
    return g


rows = []
for i in pick:
    row = cases.iloc[i]
    cs = cert.cert_status.iloc[i]
    out = dict(case_id=row.case_id, regime=reg.regime.iloc[i], cert_status=cs, native_fallback=bool(row.fallback),
               cert_root=cert.root.iloc[i], theta_root=ver.independent.iloc[i], native=row.native_solution)
    brent, reason = np.nan, ''
    if cs in ('certified_root', 'certified_no_root'):
        lo, hi = cert.joint_lower.iloc[i], cert.joint_upper.iloc[i]
        delta = 1e-12 * max(1.0, abs(lo), abs(hi))
        g = g_factory(row)
        with np.errstate(all='ignore'):
            a, b = g(lo + delta), g(hi - delta)
        if np.isfinite(a) and np.isfinite(b) and a * b < 0:
            brent = brentq(g, lo + delta, hi - delta, xtol=1e-14, rtol=4 * np.finfo(float).eps, maxiter=500)
            reason = 'bracketed'
        else:
            reason = 'no_float_bracket'
    else:
        reason = 'undefined_by_definition'
    out.update(brent_root=brent, brent_status=reason)
    if cs == 'certified_root':
        d = abs(brent - out['cert_root']) if np.isfinite(brent) else np.nan
        out['brent_abs_error'] = d
        dist = min(cert.dist_lower.iloc[i], cert.dist_upper.iloc[i])
        out['root_distance_to_asymptote'] = dist
        if reason == 'no_float_bracket':
            out['attribution'] = 'domain: root within float resolution of asymptote' if dist < 1e-10 else 'domain: float bracket sign failure'
        elif d > 1e-8:
            out['attribution'] = 'solver tolerance'
        else:
            out['attribution'] = 'agree' + ('; native fallback discarded it' if row.fallback else '')
    elif cs == 'certified_no_root':
        out['attribution'] = 'agree: no root' if reason == 'no_float_bracket' else 'float false bracket'
    else:
        out['attribution'] = 'undefined'
    rows.append(out)
t = pd.DataFrame(rows)
t.to_csv(f'{OUT}/root_certificates/independent_solver_check.csv', index=False)
print('cases', len(t), 'changed-unit surfaces', len(changed_surfaces))
print(pd.crosstab(t.regime, t.attribution).to_string())
ok = t[t.cert_status == 'certified_root']
print('max |brent - cert| where bracketed', float(ok.brent_abs_error.max()), ' max |theta - cert|', float((ok.theta_root - ok.cert_root).abs().max()))
