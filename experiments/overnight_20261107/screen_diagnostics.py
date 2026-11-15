"""Evidence for cross-screen mismatch: replicate structure, dose support, response level, monotherapy coverage, blocks per pair-cell."""
import numpy as np, pandas as pd
rows = []
for name, d in [('ONEIL', 'data/processed/oneil_v1'), ('ALMANAC', '../cluster_return/early/data/processed/almanac_v1'),
                ('NCATS_ES', '../cluster_return/early/data/processed/ncats_es_v2'), ('NCATS_DIPG', '../cluster_return/early/data/processed/ncats_dipg_v2')]:
    o = pd.read_parquet(f'{d}/observations.parquet', columns=['pair_id', 'cell_line', 'block_id', 'dose_a', 'dose_b', 'inhibition', 'is_combination', 'drug_a', 'drug_b'])
    c = o[o.is_combination]
    key = c.pair_id.astype(str) + '|' + c.cell_line.astype(str) + '|' + c.dose_a.round(12).astype(str) + '|' + c.dose_b.round(12).astype(str)
    g = c.groupby(key).inhibition
    n = g.transform('size')
    rep = c[n > 1]
    within = ((rep.inhibition - g.transform('mean')[n > 1]) ** 2).sum() / max((n[n > 1].groupby(key[n > 1]).first() - 1).sum(), 1)
    bpc = c.groupby([c.pair_id.astype(str), c.cell_line.astype(str)]).block_id.nunique()
    ld = np.log10(c.dose_a + c.dose_b)
    rows.append(dict(screen=name, combination_rows=len(c), pair_cells=len(bpc), blocks_per_pair_cell_median=float(bpc.median()),
                     share_pair_cells_single_block=float((bpc == 1).mean()), rows_with_replicate_at_identical_input=float((n > 1).mean()),
                     pooled_within_input_variance=float(within), inhibition_mean=float(c.inhibition.mean()), inhibition_sd=float(c.inhibition.std()),
                     inhibition_p01=float(c.inhibition.quantile(.01)), inhibition_p99=float(c.inhibition.quantile(.99)),
                     log10_dose_sum_p05=float(ld.quantile(.05)), log10_dose_sum_p95=float(ld.quantile(.95)),
                     doses_per_axis_median=float(c.groupby('block_id').dose_a.nunique().median()),
                     drugs=int(len(set(c.drug_a) | set(c.drug_b))), cells=int(c.cell_line.nunique())))
t = pd.DataFrame(rows)
t.to_csv('../handoff/cross_screen/screen_diagnostics.csv', index=False)
print(t.round(3).T.to_string())
