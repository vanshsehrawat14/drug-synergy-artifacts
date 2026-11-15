"""Phase 3.2 perturbation stability: coefficient rounding (certified), display rounding of scores, float32 arithmetic."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from cs622.decision_sensitivity import references, unit_scores, aggregate, compare, BUDGETS, KEY
from cs622.loewe_priority import certificate
from cs622.loewe_verify import theta_root
cases = pd.read_parquet('results/neural/runs/scientific_native_full/surface_cases.parquet')
blocks = pd.read_parquet('results/neural/runs/scientific_native_full/blocks.parquet')
for c in KEY: blocks[c] = blocks[c].astype(str)
certs = pd.read_parquet('../handoff/root_certificates/certificates.parquet')
ref, masks = references(cases, pd.read_parquet('../../2026-11-06_validation/loewe_verify/cases_verified.parquet'), pd.read_csv('results/research_20261107/precision/certificates.csv'), certs)
base = certs.set_index('case_id').reindex(cases.case_id)
pcI = aggregate(blocks, unit_scores(cases, blocks, ref['I']))[0].to_numpy()
pcN = aggregate(blocks, unit_scores(cases, blocks, ref['N']))[0].to_numpy()
scope = np.flatnonzero((cases.fallback | (base.cert_status == 'certified_root')).to_numpy())
rows = []
for digits in [6, 8, 10, 12]:
    t = pd.read_parquet(f'../handoff/root_certificates/rounded_{digits}.parquet')
    st0 = base.cert_status.to_numpy()[scope]; st1 = t.cert_status.to_numpy()
    r0 = base.root.to_numpy(float)[scope]; r1 = t.root.to_numpy(float)
    both = (st0 == 'certified_root') & (st1 == 'certified_root')
    refr = ref['I'].copy()
    fb = cases.fallback.to_numpy()[scope]
    newref = np.where(fb & (st1 == 'certified_root'), np.minimum(r1, 100), ref['N'][scope])
    refr[scope] = newref
    pc = aggregate(blocks, unit_scores(cases, blocks, refr))[0].to_numpy()
    for k in BUDGETS:
        m = compare(pcI, pc, k)
        rows.append(dict(perturbation=f'coef_round{digits}_certified', budget=k, status_changes=int((st0 != st1).sum()),
                         root_abs_max=float(np.nanmax(np.abs(r0[both] - r1[both]))), unresolved=int((st1 == 'unresolved').sum()),
                         max_abs_pair_cell_shift=float(np.abs(pc - pcI).max()), replaced=m['replaced'], replacement_fraction=m['replacement_fraction']))
# display rounding of final pair-cell scores (ties broken by canonical key)
for dec in [2, 1, 0]:
    for name, x in [('I', pcI), ('N', pcN)]:
        for k in BUDGETS:
            m = compare(x, np.round(x, dec), k)
            rows.append(dict(perturbation=f'display_round_{dec}dp_{name}', budget=k, replaced=m['replaced'], replacement_fraction=m['replacement_fraction'],
                             max_abs_pair_cell_shift=float(np.abs(np.round(x, dec) - x).max())))
# float32 arithmetic in the double-precision certificate and theta solver steps (stopping)
c32 = cases.copy()
cols = [f'{a}_{k}' for a in 'ab' for k in 'bcde'] + ['dose_a', 'dose_b']
c32[cols] = c32[cols].astype(np.float32).astype(float)
for label, frame, steps in [('float32_inputs_theta80', c32, 80), ('theta_steps40', cases, 40), ('theta_steps60', cases, 60), ('theta_steps200', cases, 200)]:
    y, status = theta_root(frame, steps)
    _, ok = certificate(frame, y)
    fb = cases.fallback.to_numpy()
    refx = np.where(fb & ok, np.minimum(y, 100), ref['N'])
    pc = aggregate(blocks, unit_scores(cases, blocks, refx))[0].to_numpy()
    for k in BUDGETS:
        m = compare(pcI, pc, k)
        rows.append(dict(perturbation=label + '_vs_I', budget=k, status_changes=int(((fb & ok) != (fb & masks['has_root'])).sum()),
                         max_abs_pair_cell_shift=float(np.abs(pc - pcI).max()), replaced=m['replaced'], replacement_fraction=m['replacement_fraction']))
out = pd.DataFrame(rows)
out.to_csv('../handoff/decision/perturbation_stability.csv', index=False)
out['replacement_fraction'] = (out.replacement_fraction * 100).round(2)
print(out[out.budget == 0.05].drop(columns='budget').to_string(index=False))
