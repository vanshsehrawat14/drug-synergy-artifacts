"""Certify roots after rounding curve coefficients to 6/8/10/12 significant digits (Phase 3.2)."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from synergy_artifacts.loewe_interval import certify, COLS
from synergy_artifacts.loewe_precision import round_coefficients

def work(args):
    frame, hints = args
    return [certify(r, h) for r, h in zip(frame.to_dict('records'), hints)]

if __name__ == '__main__':
    cases = pd.read_parquet('results/neural/runs/scientific_native_full/surface_cases.parquet')
    base = pd.read_parquet('../handoff/root_certificates/certificates.parquet').set_index('case_id').reindex(cases.case_id)
    # scope: every case whose status could matter (fallback or a certified root); undefined-by-definition cases are unchanged by rounding sign tests
    scope = np.flatnonzero((cases.fallback | (base.cert_status == 'certified_root')).to_numpy())
    for digits in [6, 8, 10, 12]:
        r = round_coefficients(cases.iloc[scope], digits)[COLS]
        hints = base.root.to_numpy(float)[scope]
        chunks = np.array_split(np.arange(len(r)), max(1, len(r) // 2000))
        rows = []
        with ProcessPoolExecutor(14) as ex:
            for part in ex.map(work, [(r.iloc[c], hints[c]) for c in chunks]):
                rows.extend(part)
        t = pd.DataFrame(rows)
        t.to_parquet(f'../handoff/root_certificates/rounded_{digits}.parquet', index=False)
        print(digits, t.cert_status.value_counts().to_dict(), flush=True)
