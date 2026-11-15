"""R20: check the derived prediction for the summed-concentration fallback (sf2_emul.exist_rule_loewe).

Derivation (increasing fitted curves f_k, upper asymptotes d_k). Let i be the drug with the smaller d (weaker) and j the
other. The isobole sum is decreasing on the joint domain, so a root exists iff f_j(x_j) < d_i. Otherwise:
  exact generalised Loewe  L_gen = f_j(x_j)              (drug i can add no dose equivalence beyond d_i)
  SynergyFinder 2 fallback L_fb  = max_k f_k(x_i + x_j)  (raw concentrations of two different drugs added)
so the reference shift is f_j(x_i + x_j) - f_j(x_j) >= 0. It grows with log(x_i / e_j): the partner's dose scale
measured in units of drug j's potency. That is a cross term between the two drugs, so it is non-additive in drug
identity, and it is fixed by monotherapy and the dose grids alone.
Checks: (1) per-cell regime counts and causes; (2) within fallback cells, Spearman(log10(x_i/e_j), shift);
(3) a worked example block. Run: python -I r20_fallback_mechanism.py
"""
import sys, json, math
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sf2_emul as S
import r17_sf2_rescore as R17

RES = R17.RES


def _fit(args):
    return S.fit_dose_response(list(args[0]), list(args[1]))


def main():
    blocks, man = R17.load_blocks()
    key = lambda dose, resp: (tuple(np.round(dose, 10)), tuple(np.round(resp, 9)))
    series = {}
    for b in blocks:
        b['kr'] = key(b['cr'], b['M'][:, 0]); b['kc'] = key(b['cc'], b['M'][0, :])
        series.setdefault(b['kr'], (b['cr'], b['M'][:, 0])); series.setdefault(b['kc'], (b['cc'], b['M'][0, :]))
    ks = list(series)
    with Pool(12) as pool:
        fits = dict(zip(ks, pool.map(_fit, [series[k] for k in ks], chunksize=20)))
    cells, seen = [], set()
    for b in blocks:
        u = (b['kr'], b['kc'])
        if u in seen:
            continue
        seen.add(u)
        fr, fc = fits[b['kr']], fits[b['kc']]
        for x2 in b['cr'][1:]:
            for x1 in b['cc'][1:]:
                ar, ac = S.active(*fr), S.active(*fc)
                gen = S.generalized_loewe(x1, x2, fc[0], fc[1], fr[0], fr[1]); ex = S.exist_rule_loewe(x1, x2, fc[0], fc[1], fr[0], fr[1])
                fb = abs(ex - gen) > 1e-6
                cause = 'root' if not fb else ('inactive_drug' if not (ar and ac) else 'stronger_exceeds_weaker_max')
                # identify the stronger drug j (larger upper asymptote among active curves) and partner dose x_i
                if ar and ac:
                    (xj, pj, mj), (xi, pi) = ((x1, fc[1], fc[0]), (x2, fr[1])) if fc[1][2] >= fr[1][2] else ((x2, fr[1], fr[0]), (x1, fc[1]))
                elif ac or ar:
                    (xj, pj, mj), xi = ((x1, fc[1], fc[0]), x2) if ac else ((x2, fr[1], fr[0]), x1)
                else:
                    xj = pj = mj = xi = None
                lr = math.log10(xi / pj[3]) if (xj is not None and mj == 'LL.4' and pj[3] > 0) else np.nan
                cells.append(dict(cause=cause, shift=ex - gen, log_ratio=lr, first_block=b['block_id']))
    C = pd.DataFrame(cells)
    out = dict(n_unique_surface_cells=len(C), cause_share=C.cause.value_counts(normalize=True).to_dict(),
               mean_shift_by_cause=C.groupby('cause')['shift'].mean().to_dict())
    fbk = C[(C.cause != 'root') & C.log_ratio.notna()]
    out['spearman_logratio_shift_fallback_cells'] = float(fbk[['log_ratio', 'shift']].corr(method='spearman').iloc[0, 1])
    q = pd.qcut(fbk.log_ratio, 5)
    out['shift_by_logratio_quintile'] = {str(k): float(v) for k, v in fbk.groupby(q, observed=True)['shift'].mean().items()}
    print(json.dumps(out, indent=1))
    (RES / 'r20_fallback_mechanism.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
