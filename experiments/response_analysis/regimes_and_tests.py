"""loewe_regimes.csv plus mathematical identity checks (dose-zero/inverse identity, arm symmetry, monotonicity)."""
import json, sys
import numpy as np, pandas as pd
from mpmath import mp
sys.path.insert(0, '.')
from synergy_artifacts.loewe_interval import certify, COLS
from synergy_artifacts.decision_sensitivity import references

OUT = '../handoff'
cases = pd.read_parquet('results/neural/runs/scientific_native_full/surface_cases.parquet')
ver = pd.read_parquet('../../2026-11-06_validation/loewe_verify/cases_verified.parquet')
certs = pd.read_parquet(f'{OUT}/root_certificates/certificates.parquet')
ref, masks = references(cases, ver, pd.read_csv('results/interaction_analysis/precision/certificates.csv'), certs)
cert = certs.set_index('case_id').reindex(cases.case_id)
v = ver.set_index('case_id').reindex(cases.case_id)
st = cert.cert_status.to_numpy()
fb = cases.fallback.to_numpy()
root = cert.root.to_numpy(float)
dist = np.minimum(cert.dist_lower.to_numpy(float), cert.dist_upper.to_numpy(float))
nat = cases.native_solution.to_numpy(float)
err = np.where(~fb & (st == 'certified_root'), np.abs(nat - root), np.nan)
regime = np.select([
    st == 'invalid_curve', st == 'undefined_opposite_or_flat', st == 'undefined_empty_interval',
    ~fb & (st == 'certified_root') & (err <= .05), ~fb & (st != 'certified_root'), ~fb & (st == 'certified_root'),
    fb & (st == 'certified_root') & (dist >= .01), fb & (st == 'certified_root') & (dist < .01),
    fb & (st == 'certified_no_root')],
    ['invalid_or_missing_curve', 'undefined_opposite_or_flat', 'undefined_empty_interval',
     'native_accepted_certified_root', 'native_accepted_no_admissible_root', 'native_accepted_inaccurate_root',
     'fallback_certified_unique_root', 'fallback_boundary_certified_root', 'fallback_certified_no_root'], 'unresolved')
table = pd.DataFrame({
    'case_id': cases.case_id, 'surface_id': cases.surface_id, 'dose_a': cases.dose_a, 'dose_b': cases.dose_b,
    'model_a': cases.model_a, 'model_b': cases.model_b, 'native_code': cases.native_code, 'native_fallback': fb,
    'historical_class': v['class'].to_numpy(), 'cert_status': st, 'regime': regime,
    'in_C1': masks['c1'], 'in_C2_recovered': masks['c2_recovered'],
    'certified_root': root, 'root_distance_to_asymptote': dist, 'native_reference': ref['N'],
    'certified_reference_I': ref['I'], 'reference_change_I': ref['I'] - ref['N'],
    'native_accepted_abs_error': err, 'hsa_fallback_reference': ref['H']})
table.to_csv(f'{OUT}/loewe_regimes.csv', index=False)
summ = table.groupby('regime').agg(cases=('case_id', 'size'), native_accepted=('native_fallback', lambda s: int((~s).sum())),
                                   changed_in_I=('reference_change_I', lambda s: int((s != 0).sum())),
                                   abs_ref_change_sum=('reference_change_I', lambda s: float(s.abs().sum())),
                                   abs_ref_change_max=('reference_change_I', lambda s: float(s.abs().max())))
summ.to_csv(f'{OUT}/loewe_regime_summary.csv')
print(summ.to_string())

# 1. inverse identity: D(f(x)) == x at 60 digits for every distinct fitted arm (dose-zero reduction of the Loewe equation)
mp.dps = 60
arms = []
for a in 'ab':
    f = cases[[f'model_{a}', f'{a}_b', f'{a}_c', f'{a}_d', f'{a}_e', f'dose_{a}']].copy()
    f.columns = ['model', 'b', 'c', 'd', 'e', 'dose']
    arms.append(f)
arms = pd.concat(arms).drop_duplicates()
arms = arms[np.isfinite(arms[['b', 'c', 'd', 'e']]).all(1) & (arms.b != 0) & (arms.c != arms.d)]
rng = np.random.default_rng(20261107)
sample = arms.iloc[rng.choice(len(arms), min(20000, len(arms)), replace=False)]
worst = mp.mpf(0); bad = 0; checked = 0
for r in sample.itertuples():
    b, c, d, e, x = map(mp.mpf, (r.b, r.c, r.d, r.e, r.dose))
    t = mp.log(x) - (mp.log(e) if r.model == 'LL.4' else e)
    y = c + (d - c) / (1 + mp.exp(b * t))
    if not (min(c, d) < y < max(c, d)):
        continue  # effect saturated at this precision; inverse undefined
    centre = mp.log(e) if r.model == 'LL.4' else e
    D = mp.exp(centre + mp.log((d - y) / (y - c)) / b)
    rel = abs(D / x - 1); checked += 1
    worst = max(worst, rel); bad += rel > mp.mpf('1e-30')
tests = dict(inverse_identity=dict(arms_checked=checked, max_rel_error=mp.nstr(worst, 5), failures_above_1e_30=int(bad)))

# 2. arm-order symmetry on a seeded sample of certified roots and no-root cases
idx = rng.choice(len(cases), 20000, replace=False)
sw = cases.iloc[idx][COLS].copy()
ren = {f'a_{k}': f'b_{k}' for k in 'bcde'} | {f'b_{k}': f'a_{k}' for k in 'bcde'} | {'dose_a': 'dose_b', 'dose_b': 'dose_a', 'model_a': 'model_b', 'model_b': 'model_a'}
sw = sw.rename(columns=ren)[COLS]
mism = 0; maxd = 0.0
for i, r in zip(idx, sw.to_dict('records')):
    res = certify(r, root[i] if np.isfinite(root[i]) else None)
    if res['cert_status'] != st[i]:
        mism += 1
    elif st[i] == 'certified_root':
        maxd = max(maxd, abs(res['root'] - root[i]))
tests['arm_symmetry'] = dict(cases=len(idx), status_mismatches=mism, max_root_difference=maxd)

# 3. monotonicity of certified references along each dose axis within a surface
viol = 0; pairs = 0
df = pd.DataFrame({'s': cases.surface_id, 'xa': cases.dose_a, 'xb': cases.dose_b, 'y': root, 'dir': cert.direction.to_numpy(float), 'ok': st == 'certified_root'})
df = df[df.ok]
for axis, other in [('xa', 'xb'), ('xb', 'xa')]:
    g = df.sort_values(['s', other, axis])
    same = (g.s.values[1:] == g.s.values[:-1]) & (g[other].values[1:] == g[other].values[:-1])
    dy = np.diff(g.y.values)[same] * g.dir.values[1:][same]
    pairs += int(same.sum()); viol += int((dy < -1e-9).sum())
tests['monotonicity'] = dict(adjacent_pairs=pairs, violations_beyond_1e_9=viol,
                             note='Certified roots must be nondecreasing in each dose for co-increasing curves (nonincreasing for co-decreasing).')
json.dump(tests, open(f'{OUT}/root_certificates/identity_tests.json', 'w'), indent=2)
print(json.dumps(tests, indent=2))
