"""Convention sensitivity: native accepted a solver value in 1,883 cases whose curves run in opposite directions or are flat
(no Loewe expectation by definition). Replace those values by the native summed-dose fallback and recompute selections."""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from synergy_artifacts.decision_sensitivity import references, unit_scores, aggregate, compare, BUDGETS, KEY
from synergy_artifacts.decision_followup import fallback_value
cases = pd.read_parquet('results/neural/runs/scientific_native_full/surface_cases.parquet')
blocks = pd.read_parquet('results/neural/runs/scientific_native_full/blocks.parquet')
for c in KEY: blocks[c] = blocks[c].astype(str)
certs = pd.read_parquet('../handoff/root_certificates/certificates.parquet')
ref, masks = references(cases, pd.read_parquet('../../2026-11-06_validation/loewe_verify/cases_verified.parquet'), pd.read_csv('results/interaction_analysis/precision/certificates.csv'), certs)
st = certs.set_index('case_id').reindex(cases.case_id).cert_status.to_numpy()
und = ~masks['fallback'] & (st == 'undefined_opposite_or_flat')
fb = fallback_value(cases)
rows = []
for name, base in [('N', ref['N']), ('I', ref['I'])]:
    alt = base.copy(); alt[und] = fb[und]
    a = aggregate(blocks, unit_scores(cases, blocks, base))[0].to_numpy()
    b = aggregate(blocks, unit_scores(cases, blocks, alt))[0].to_numpy()
    for k in BUDGETS:
        r = compare(a, b, k)
        rows.append(dict(base=name, change='undefined_accepted_to_fallback', cases=int(und.sum()), budget=k, replaced=r['replaced'],
                         replacement_fraction=r['replacement_fraction'], max_abs_case_change=float(np.abs(alt - base)[und].max()),
                         pair_cells_affected=int((np.abs(b - a) > 0).sum())))
t = pd.DataFrame(rows); t.to_csv('../handoff/decision/undefined_accepted_sensitivity.csv', index=False)
t['replacement_fraction'] = (t.replacement_fraction * 100).round(2)
print(t.to_string(index=False))
