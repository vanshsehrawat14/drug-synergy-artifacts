"""Replay historical certificates without modifying them; repair lossy serialization.

Float coefficients remain exact binary rationals. A repaired artifact is a new
certificate, never a change to the fixed-fit Loewe definition or historical score.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from mpmath import mp

from .io import digest, write_json
from .loewe_interval import Eq, _arms, certify


def bracket_structure(row, dps=300):
    with mp.workdps(dps):
        try:
            lo, hi = mp.mpf(row['bracket_lo']), mp.mpf(row['bracket_hi'])
            return bool(mp.mpf(float(row['joint_lower'])) < lo < hi < mp.mpf(float(row['joint_upper'])))
        except (ValueError, TypeError):
            return False


def replay(case, cert):
    if not bracket_structure(cert):
        return False
    eq = Eq(_arms(case), max(100, int(cert['dps']) * 2))
    g0 = eq.g_iv(mp.mpf(cert['bracket_lo']))
    g1 = eq.g_iv(mp.mpf(cert['bracket_hi']))
    if g0 is None or g1 is None:
        return False
    return bool((g0.a > 0 and g1.b < 0) or (g1.a > 0 and g0.b < 0))


def _audit(item):
    case, old, structure = item
    valid = structure and replay(case, old)
    new = old if valid else certify(case, hint=None, dps=100)
    ok = new['cert_status'] == 'certified_root' and replay(case, new)
    return dict(case_id=case['case_id'], historical_structure_ok=structure,
                historical_replay_ok=valid, repaired=not valid, replay_ok=ok,
                root_shift=abs(new.get('root', np.nan)-old['root'])), new


def run(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    cases = pd.read_parquet(a.cases).set_index('case_id', drop=False)
    cert = pd.read_parquet(a.certificates).set_index('case_id', drop=False)
    roots = cert[cert.cert_status == 'certified_root']
    good = roots.apply(bracket_structure, axis=1)
    fallback = cases.loc[roots.index, 'fallback'].astype(bool)
    rng = np.random.default_rng(20261108)
    control = rng.choice(roots.index[~fallback], size=200, replace=False)
    ids = roots.index[fallback | ~good].union(control, sort=False)
    jobs = [(cases.loc[i].to_dict(), cert.loc[i].to_dict(), bool(good.loc[i])) for i in ids]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(_audit, jobs, chunksize=64))
    audit = pd.DataFrame([r for r, _ in results])
    audit.to_csv(out/'replay.csv', index=False)
    repairs = pd.DataFrame([r for (info, r) in results if info['repaired']])
    repairs.to_parquet(out/'repaired_certificates.parquet', index=False)
    repaired = cert.copy()
    for r in repairs.to_dict('records'):
        for k,v in r.items(): repaired.loc[r['case_id'], k] = v
    repaired.reset_index(drop=True).to_parquet(out/'certificates_replayable.parquet', index=False)
    summary = dict(input_sha256=digest(a.certificates), root_count=len(roots),
                   structurally_unreplayable=int((~good).sum()), audited_roots=len(ids),
                   corrected_fallback_roots=int(fallback.sum()), historical_equation_failures=int((~audit.historical_replay_ok).sum()),
                   repaired_count=int(audit.repaired.sum()), replay_failures=int((~audit.replay_ok).sum()),
                   max_float_root_shift=float(audit.root_shift.max()),
                   note='All root brackets checked structurally; equation replay covers every corrected fallback, every structurally invalid bracket, and 200 seeded accepted controls. Other accepted equations retain historical evidence.')
    write_json(out/'summary.json', summary)
    print(summary, flush=True)
    if not audit.replay_ok.all(): raise RuntimeError('Certificate replay remains incomplete')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', default='results/neural/runs/scientific_native_full/surface_cases.parquet')
    p.add_argument('--certificates', required=True);p.add_argument('--output', required=True)
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
