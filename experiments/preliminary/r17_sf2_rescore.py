"""R17: re-score every O'Neil block from DrugComb's raw doses with an emulation of DrugComb/SynergyFinder 2 (sf2_emul.py)
and with alternative reference conventions. Pure function of the dose-level data; no synergy labels are used here.

Per block (5 x 5 matrix, 4 doses per drug + zero):
  T          mean observed inhibition over the 16 combination cells (the common raw-response target)
  ref_bliss  mean Bliss reference on observed monotherapy values (SynergyFinder 2 Bliss)
  ref_hsa    mean HSA reference on observed monotherapy values
  ref_loewe_sf2        SF2 emulation: nleqslv root, summed-concentration fallback when termcd >= 3, cap at 100
  ref_loewe_sf2_hsafb  same, but a dimensionally consistent fallback (HSA of fitted curves at own doses)
  ref_loewe_sf2_nocap  SF2 emulation without the cap at 100
  ref_loewe_exist      simplified SF2: root when it exists in the joint domain, else summed-concentration fallback
  ref_loewe_gen        exact generalised Loewe (R15b convention) on the same fitted curves; inactive curves add no dose
  ref_bliss_fit        Bliss on fitted curves (removes observed-vs-fitted mismatch)
  ref_{bliss_fit,loewe_gen}_{h1,d100,ideal}  sub-ablations of the reference algebra (Hill slope 1 / Emax 100 / both + E0 0)
  n_fallback           number of the 16 cells where the SF2 fallback fired
Run: python -I r17_sf2_rescore.py [limit_blocks]
Output: results/preliminary/r17_block_scores.csv.gz, results/preliminary/r17_fits.csv
"""
import sys, json, time, math
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import sf2_emul as S

ROOT = Path(__file__).resolve().parents[2]; RES = ROOT / 'results' / 'preliminary'
RAW = ROOT / 'data' / 'processed' / 'drugcomb_oneil'


def load_blocks(limit=None):
    z = np.load(RAW / 'oneil_dose_level.npz'); man = json.loads((RAW / 'manifest.json').read_text())
    df = pd.DataFrame({k: z[k] for k in z.files})
    if limit:
        df = df[df.block_id <= limit]
    df = df.sort_values(['block_id', 'conc_r', 'conc_c'])
    blocks = []
    for bid, g in df.groupby('block_id', sort=True):
        cr = np.sort(g.conc_r.unique()); cc = np.sort(g.conc_c.unique())
        if len(cr) != 5 or len(cc) != 5 or len(g) != 25:
            blocks.append(dict(block_id=int(bid), bad=True)); continue
        M = g.pivot_table(index='conc_r', columns='conc_c', values='inhibition', aggfunc='mean').loc[cr, cc].to_numpy()
        blocks.append(dict(block_id=int(bid), bad=False, dr=int(g.drug_row.iloc[0]), dc=int(g.drug_col.iloc[0]), cell=int(g.cell.iloc[0]),
                           cr=cr, cc=cc, M=M))
    return blocks, man


def _fit(args):
    dose, resp = args
    return S.fit_dose_response(list(dose), list(resp))


def _loewe_block(args):
    cr, cc, fc, fr = args  # fc: (model, par) of column drug; fr: of row drug
    names = ['sf2', 'hsafb', 'nocap', 'exist', 'gen', 'gen_h1', 'gen_d100', 'gen_ideal']
    out = {k: np.full((4, 4), np.nan) for k in names}; out['code'] = np.zeros((4, 4), int)
    if fc[0] == 'FAIL' or fr[0] == 'FAIL':
        return out
    for i, x2 in enumerate(cr[1:]):        # row drug dose
        for j, x1 in enumerate(cc[1:]):    # column drug dose
            v, code = S.sf2_loewe_ref(x1, x2, fc[0], fc[1], fr[0], fr[1], 'summed', True)
            out['sf2'][i, j] = v; out['code'][i, j] = code
            out['nocap'][i, j] = S.sf2_loewe_ref(x1, x2, fc[0], fc[1], fr[0], fr[1], 'summed', False)[0]
            out['hsafb'][i, j] = S.sf2_loewe_ref(x1, x2, fc[0], fc[1], fr[0], fr[1], 'hsa_fit', True)[0]
            out['exist'][i, j] = S.exist_rule_loewe(x1, x2, fc[0], fc[1], fr[0], fr[1])
            out['gen'][i, j] = S.generalized_loewe(x1, x2, fc[0], fc[1], fr[0], fr[1])
            for k in ['h1', 'd100', 'ideal']:
                out['gen_' + k][i, j] = S.generalized_loewe(x1, x2, fc[0], S.ablate(fc[0], fc[1], k), fr[0], S.ablate(fr[0], fr[1], k))
    return out


def main(limit=None):
    t0 = time.time()
    blocks, man = load_blocks(limit)
    good = [b for b in blocks if not b['bad']]
    print('blocks', len(blocks), 'good', len(good), round(time.time() - t0, 1), flush=True)
    # unique monotherapy series -> one fit each (identical inputs give identical fits, as in DrugComb)
    key = lambda dose, resp: (tuple(np.round(dose, 10)), tuple(np.round(resp, 9)))
    series = {}
    for b in good:
        b['kr'] = key(b['cr'], b['M'][:, 0]); b['kc'] = key(b['cc'], b['M'][0, :])
        series.setdefault(b['kr'], (b['cr'], b['M'][:, 0])); series.setdefault(b['kc'], (b['cc'], b['M'][0, :]))
    keys = list(series)
    with Pool(12) as pool:
        fits = dict(zip(keys, pool.map(_fit, [series[k] for k in keys], chunksize=20)))
    print('fits', len(fits), 'models', pd.Series([f[0] for f in fits.values()]).value_counts().to_dict(), round(time.time() - t0, 1), flush=True)
    # unique (row fit, col fit, doses) -> one Loewe surface
    lkey = lambda b: (b['kr'], b['kc'])
    uniq = {}
    for b in good:
        uniq.setdefault(lkey(b), (b['cr'], b['cc'], fits[b['kc']], fits[b['kr']]))
    ukeys = list(uniq)
    with Pool(12) as pool:
        surf = dict(zip(ukeys, pool.map(_loewe_block, [uniq[k] for k in ukeys], chunksize=20)))
    print('loewe surfaces', len(surf), round(time.time() - t0, 1), flush=True)
    rows = []
    for b in good:
        M = b['M']; yr = M[1:, 0][:, None]; yc = M[0, 1:][None, :]; C = M[1:, 1:]
        fr, fc = fits[b['kr']], fits[b['kc']]; s = surf[lkey(b)]
        bliss = yr + yc - yr * yc / 100; hsa = np.maximum(yr, yc)
        row = dict(block_id=b['block_id'], drug_row=b['dr'], drug_col=b['dc'], cell=b['cell'], T=C.mean(),
                   ref_bliss=bliss.mean(), ref_hsa=hsa.mean(),
                   n_fallback=int((s['code'] >= 3).sum()), n_code2=int((s['code'] == 2).sum()), n_err=int((s['code'] == 999).sum()),
                   n_nan_sf2=int(np.isnan(s['sf2']).sum()), model_row=fr[0], model_col=fc[0],
                   active_row=bool(fr[0] != 'FAIL' and S.active(*fr)), active_col=bool(fc[0] != 'FAIL' and S.active(*fc)))
        for k in ['', '_h1', '_d100', '_ideal']:
            if fr[0] == 'FAIL' or fc[0] == 'FAIL':
                row['ref_bliss_fit' + k] = np.nan; continue
            pr_, pc_ = S.ablate(fr[0], fr[1], k[1:]), S.ablate(fc[0], fc[1], k[1:])
            f_r = np.array([S.curve(fr[0], pr_, x) for x in b['cr'][1:]])[:, None]; f_c = np.array([S.curve(fc[0], pc_, x) for x in b['cc'][1:]])[None, :]
            row['ref_bliss_fit' + k] = np.mean(f_r + f_c - f_r * f_c / 100)
        for k, v in [('sf2', 'ref_loewe_sf2'), ('hsafb', 'ref_loewe_sf2_hsafb'), ('nocap', 'ref_loewe_sf2_nocap'), ('exist', 'ref_loewe_exist'),
                     ('gen', 'ref_loewe_gen'), ('gen_h1', 'ref_loewe_gen_h1'), ('gen_d100', 'ref_loewe_gen_d100'), ('gen_ideal', 'ref_loewe_gen_ideal')]:
            row[v] = np.nanmean(s[k]) if np.isfinite(s[k]).any() else np.nan
            row['nnan_' + k] = int((~np.isfinite(s[k])).sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RES / ('r17_block_scores.csv.gz' if not limit else 'r17_block_scores_dev.csv.gz'), index=False)
    fr = pd.DataFrame([dict(n=len(k[0]), model=v[0], **({'b': v[1][0], 'c': v[1][1], 'd': v[1][2], 'e': v[1][3]} if v[1] else {}))
                       for k, v in fits.items()])
    fr.to_csv(RES / ('r17_fits.csv' if not limit else 'r17_fits_dev.csv'), index=False)
    print(out.describe().T.round(3).to_string()); print('done', round(time.time() - t0, 1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
