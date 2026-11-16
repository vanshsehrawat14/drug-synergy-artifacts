# Target and leakage audit

Scope: the O'Neil screen in DrugComb v1.4 (`data/processed/oneil_v1/observations.parquet`, SHA-256 `ea588405…ce8b`), the frozen resolution cache (`interaction_analysis/resolution`, cache `91f93811…6687`) and the frozen pair folds. Code: `experiments/interaction_structure/variance.py`, `dose_shape_check.py`; numbers: `results/interaction_structure/variance_components.json`, `dose_shape_check.json`. Design frozen before any outcome was computed: `experiments/interaction_structure/design.json` (commit c0a785c).

## Units

| Quantity | Value |
|---|---:|
| Drug pairs / drugs / cell lines | 583 / 38 / 39 |
| Pair-cells (complete matrix; missing) | 22,737 (0) |
| Combination rows | 1,475,328 |
| Units (pair-cell × dose pair): four rows / eight rows | 365,352: 361,872 / 3,480 |
| Blocks: pair-cells with 4 / 8 blocks | 92,208: 22,422 / 315 |
| Row weight w (rows per pair-cell): 64 / 128 | 22,422 / 315 |
| Pairs per original fold | 116, 116, 116, 116, 119 |

Plates, plating dates and operators are not in the export. A block (one 4 × 4 combination matrix with its single-agent edges) is the only recorded replicate unit. The 315 eight-block pair-cells are 21 pairs × 15 cell lines with a second set of four blocks deposited separately (source rows ≥ 2,273,700; 1,260 blocks), called batch 2 below. In 240 of them (16 pairs) both batches use the same dose grid up to rounding (largest |log dose ratio| ≤ 0.02, e.g. 1.08 vs 1.075 µM). In the other 75 (AZD1775+GEMCITABINE, AZD1775+L778123, GEMCITABINE+MK8776, L778123+MK8669, MRK003+VINBLASTINE) one drug was tested at different concentrations, so those blocks are not replicates.

## Target

For unit $u$ of pair-cell $(p,c)$ with rows $y_{u1},\dots,y_{un_u}$ (percent inhibition):

- $\bar y_u$ is the unit mean.
- $E_u = a_u + b_u - a_u b_u/100$ is the frozen Bliss reference. It takes single-agent inhibition $a_u$, $b_u$ at the unit's doses from a training-pair monotherapy lookup, with log-dose interpolation when a dose is absent, computed out of fold.
- The pair-cell mean is $m_{pc} = \sum_u n_u(\bar y_u - E_u)/w_{pc}$.
- The target is $t_{pc} = m_{pc} - \sum_d h_{pd}\hat\alpha_{dc}$, where $h$ marks the two drugs of $p$ and $\hat\alpha_{\cdot c}$ is the row-weighted least-squares fit of $m_{\cdot c}$ on drug indicators within the panel (pair fold of $p$, cell $c$).

$t$ is the frozen `pair_cell:bliss_residual` estimand. Energies are $\sum w t^2$. The same contrast of $\bar y_u$ is the reference-free measured-response target ($T_{raw}$); of $E_u$, the reference-only target ($T_{ref}$, with $T_{raw} = T_{ref} + T_{bliss}$); of $\bar y_u - \max(a_u,b_u)$, the HSA target. $T_{train}$ replaces the in-panel drug fit with one fitted on pairs from the other folds only.

## Reference sharing

- Single-agent edge values are identical across the replicate blocks of every pair-cell and batch: all 184,416 (pair-cell, batch, drug, dose) groups take one value.
- They are also identical across pairs: 5,928 (drug, cell, dose) groups have a pooled SD of 0.0006 and a maximum SD of 0.0012. Each drug-cell-dose has one single-agent number, reused by every pair and every replicate.
- The double-zero well is the constant −0.000626 in every block.
- The batch-2 blocks reuse the same single-agent values: across the 240 replicate pair-cells, the largest edge difference is 0.22 and the mean 0.002.
- The frozen reference equals the block-edge reference except at 604 units in 77 re-tested pair-cells (max |Δ| 76.9). There the lookup mixes batch-specific doses.

Consequences:

- Replicate agreement is technical reproducibility conditional on the reference. No reference uncertainty enters any noise estimate, and none can be estimated from this export.
- Reference errors would be shared by every pair containing the drug in that cell line, and by every replicate.
- A propagation analysis with assumed single-agent error (VARIANCE_COMPONENTS.md) bounds their effect instead.

## Replicate dependence

The two batches give the only independent re-measurement in the data. In the 240 replicate pair-cells, each batch alone (weight 64) was placed in the frozen panel contrast.

- **Unadjusted:** the mean squared batch difference of the contrast is 4.47 times what within-batch replicate dispersion predicts (pair interval 2.96–6.59). Batch 1 is higher by 4.9 contrast units on average.
- **After removing each cell line's batch shift:** the ratio is 1.79 (0.86–3.33). A complete re-run would remove that shift through the cell-line part of the contrast.

The four replicate blocks of a pair-cell therefore share a batch component that the replicate split cannot see. VARIANCE_COMPONENTS.md quantifies how much of the reproducible signal survives the change of batch.

## Reproduction of reported noise estimates

The block signs are the frozen draws (50 draws, seed 97). Pair-cell values are bit-identical to `manuscript/audit/outputs/interaction_audit.json`.

| Estimate | Frozen | Recomputed | Batch-aware split |
|---|---:|---:|---:|
| Pair-cell Bliss noise share | 0.0978041 | 0.0978041 | 0.0971 |
| Pair-cell raw noise share | 0.0671330 | 0.0671330 | |
| Pair-cell Bliss energy per row | 20.588451 | 20.588451 | |
| Cross-cell covariance share, unweighted (Bliss) | 0.2337499 | 0.2337499 | |
| Dose-shape Bliss noise share, grid split | 0.3633470 | 0.3633470 | 0.3631 |
| Dose-shape Bliss noise share, old panel split | 0.4473365 | 0.4473365 | |

How the dual-grid correction arises:

- The 210 dual-grid pair-cells are the 135 rounding-level grid changes plus the 75 changed-dose pair-cells. Their units split into two four-row groups.
- A split balanced only per pair-cell gives some units unbalanced signs, which leaks dose-shape signal into the noise estimate. That is the 0.447 → 0.363 correction.
- The batch-aware split additionally separates the 105 identical-grid eight-block pair-cells by batch. It changes the dose-shape share by 0.0003 and the pair-cell share by 0.0007.

The ~36% dose-shape figure stands. It is a dose-shape quantity and must not be confused with the pair-cell noise share (9.8%).

## Leakage

1. **Anchor masks** are permutations seeded by SHA-256 of (pair_id, seed). They never read targets. Evaluation cells are positions 16 to 38, identical for every k and every model (tested).
2. **Evaluated cells** never enter a prediction. Perturbing them leaves B1 and B2 unchanged (canary test). Covariance, the pair-intercept parameters and B2 settings use other folds only. B2 is selected on fold f+1 with covariance from the remaining three folds, then refitted on the four non-test folds.
3. **The frozen contrast depends on the test set.** Its drug effects are fitted within (pair fold, cell) panels that contain the evaluated pair-cell and other test entries.
   - **Original pair folds:** a held-out pair's anchors carry other held-out pairs' values at the anchor cells, never the pair's own evaluated cells. The train-only baseline target, which contains no test data at all, reproduces the anchor curves within 0.004 R² for k ≤ 16 (COMPARATIVE_CONTROLS.md).
   - **Pair-cell split** (the pair-seen comparison): every panel mixes training and test entries. Anchors built from the frozen contrast carry other pairs' test entries. A covariance estimated on training entries of the same panels picks up test entries through the drug fit.
     - With drug effects refitted on training entries only (`masked_contrast`), B1 falls from 0.209 to 0.177 and B2 from 0.309 to 0.216.
     - Fixed-setting attribution: the anchor path and the covariance path each add about 0.05 to B2; the anchor path adds 0.03 to B1.
     - The leak-free comparison is primary.
     - The previously reported matched oracle (0.205, mean of the pair's training-fold contrasts) carries the anchor-path inflation.
4. **Held-out-pair predictions as a target offset** (`f2_residual` sensitivity only): the out-of-fold F2 predictions for training pairs came from models trained with the test fold. This is an indirect path, so the target is reported as exploratory.
5. **Neural pair-seen predictions (B3):** they come from the frozen campaign and were not retrained. They are compared only in the matched pair-cell split, through pooled and per-fold values in `manuscript/audit/outputs/pairseen_audit.json`. Per-pair B3 energies remain with the prediction files on the cluster, so B3 differences have fold-level but no pair-bootstrap uncertainty.
6. **Dose grids, cell availability, shared drugs:**
   - Changed-dose pair-cells are excluded from every cross-batch analysis. In the frozen target they average both batches.
   - Cell availability is complete.
   - Shared drugs are handled with delete-one-drug jackknife intervals and donor-pair controls.
7. **Is the pair-wide offset inherited from the reference?** No. Pair-wide covariance decomposes as $\tau^2_{raw}$ 7.41 = $\tau^2_{ref}$ 2.44 + $\tau^2_{bliss}$ 4.99 + 2(−0.013).
   - Bliss pair means are uncorrelated with reference pair means (r = 0.009).
   - They track response-level pair means (r = 0.81), HSA pair means (0.98) and train-only pair means (0.80).
