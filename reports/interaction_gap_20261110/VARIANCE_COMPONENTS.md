# Variance components

Code: `variance.py`, `grounding.py`, `batch_transfer.py` (all in `experiments/interaction_gap_20261110/`). Numbers: `variance_components.json`, `grounding.json`, `batch_transfer.json` (all in `results/interaction_gap_20261110/`). Intervals:

- pair-cluster bootstrap (2,000 draws), written [a, b];
- delete-one-drug jackknife, written "drug [a, b]";
- delete-one-cell jackknife, written "cell [a, b]".

## Estimators

For the pair-cell matrix $t$ (583 × 39, row weights $w$), the target energy is $\sum w t^2$.

- **Noise:** $\sum w\,v$, with $v_{pc}$ the mean over 50 block-sign draws of the squared contrast of the half-difference mean. Equivalently, total energy minus the mean cross product of the two half-sample targets, $E[t_A t_B]$.
- **Pair-wide component:** the row-weighted cross-cell U-statistic $\tau^2 = \sum_p\sum_{c\ne c'} w t\,w' t' / \sum_p\sum_{c\ne c'} w w'$, as an energy share $\tau^2 \sum w / \sum w t^2$. It needs no assumption about the noise beyond independence across cell lines.
- **Pair-by-cell:** the remainder.

Under the contrast, cell-line means and additive drug-by-cell effects are zero by construction. A pure cell effect $b_c$ is therefore not estimable from this target; the decomposition is pair-wide plus pair-by-cell plus noise. The in-sample pair-mean share (classical ANOVA) is reported separately as descriptive.

Out-of-sample predictable shares come from leave-one-cell-out prediction for held-out pairs:
- **B1** uses the pair's other 38 cell lines through the empirical-Bayes intercept.
- **B2** uses them through the best linear predictor under a cross-cell covariance.

In both, covariance and settings are estimated on other pair folds.

## Components

| Target | Energy/row | Noise | Pair-wide | Pair-by-cell | Descriptive pair mean |
|---|---:|---:|---:|---:|---:|
| Bliss residual (primary) | 20.59 | 0.098 [0.089, 0.108] | 0.242 [0.197, 0.293] | 0.660 [0.613, 0.702] | 0.254 |
| drug jackknife | | [0.070, 0.126] | [0.151, 0.334] | [0.566, 0.754] | |
| cell jackknife | | [0.069, 0.127] | [0.160, 0.324] | [0.553, 0.767] | |
| Measured response | 29.99 | 0.067 [0.062, 0.073] | 0.247 [0.214, 0.282] | 0.686 [0.651, 0.718] | 0.263 |
| Train-only baseline residual | 31.72 | 0.103 [0.093, 0.115] | 0.249 [0.195, 0.304] | 0.648 [0.595, 0.697] | 0.255 |
| HSA residual | 21.32 | 0.095 | 0.248 | 0.657 | 0.259 |
| Reference only (Bliss) | 9.24 | 0 | 0.264 | 0.736 | 0.286 |

The decomposition is stable across target definitions. About a quarter of the reproducible contrast energy is a pair-wide offset shared across cell lines, about two thirds is pair-by-cell, and a tenth is replicate noise. Interval widths come mainly from drug and cell heterogeneity: drug and cell jackknife intervals are roughly twice the pair-bootstrap width.

Out-of-sample shares (leave one cell out, all 38 other cells as anchors):

| Target | B1 intercept | B2 covariance | B2 − B1 |
|---|---:|---:|---:|
| Bliss residual | 0.215 [0.175, 0.258]; drug [0.127, 0.303] | 0.360 [0.319, 0.400]; drug [0.263, 0.456] | 0.144 [0.116, 0.172]; drug [0.076, 0.213] |
| Measured response | 0.225 | 0.420 | 0.195 [0.171, 0.218] |
| Train-only baseline residual | 0.217 | 0.382 | 0.165 [0.132, 0.201]; drug [0.075, 0.255] |
| Reference only | 0.249 | 0.620 | 0.371 |

The intercept recovers almost all of the pair-wide component: 0.215 of 0.242. With equal true intercepts, the empirical-Bayes ceiling at k = 38 is 0.242 × 0.92. B2's settings are similar in every fold: shrinkage toward compound symmetry of 0.25 to 0.5, at full rank. The transferable cell structure is therefore not low-dimensional.

With all other cells as anchors, B2 recovers 0.144 of the 0.660 pair-by-cell share, about a fifth. About 0.54 of the Bliss contrast energy is reproducible within batch yet not linearly transferable from the same pair's other cell lines (`fig_decomposition`).

## Cross-cell covariance

The noise-corrected 39 × 39 covariance of the Bliss target has trace 725 and a smallest eigenvalue of −0.12, which is effectively positive semidefinite.

- The leading eigenvalue holds 30.6% of the trace. Its eigenvector has cosine 0.94 with the constant vector, i.e. the pair-wide direction.
- The next four hold 13.9%, 9.8%, 8.0% and 6.7%.
- The off-diagonal structure is reproducible. The cell-by-cell correlation of pair-centred targets, estimated on disjoint halves of the pairs, agrees with Spearman 0.62 (p = 0.0005; measured response 0.72).
- It is only weakly related to transcriptome-wide similarity of the cell lines (3,984 FARMS genes, independent of combination outcomes): Mantel Spearman 0.069 (p = 0.082) for the Bliss target and 0.121 (p = 0.002) for measured response.
- A leave-one-cell-out predictor whose cell similarity comes only from expression adds 0.002 [−0.002, 0.006] over B1. That is 1% of B2's gain.

The structure that B2 transfers is real within this screen but is not explained by global expression similarity. These data cannot separate biological cell context from shared screening runs.

## Held-out-pair models against the two components

| Prediction (Bliss contrast) | Pair-mean R² | Pair-centred R² | Total R² |
|---|---:|---:|---:|
| F2 production (f2_base) | −0.277 | 0.027 | −0.050 |
| F2 selected | −0.144 | 0.131 | 0.061 |
| GP | 0.059 | 0.017 | 0.028 |
| CF | 0.063 | 0.021 | 0.032 |
| B1, all other cells of the pair | 0.992 | −0.049 | 0.215 |
| B2, all other cells of the pair | 0.967 | 0.153 | 0.360 |

Pair means carry 25.4% of the target energy. Without same-pair data the models do not predict them: R² is at most 0.06, and the neural models score below zero. Their small skill comes from pair-centred, cell-specific structure. F2-selected reaches 0.131 there, close to B2's 0.153 with 38 anchor cells.

## Negative controls

| Control | Result |
|---|---|
| Pair labels permuted within each (fold, cell) panel, 20 permutations | pair-wide 0.0007 ± 0.0016 (max 0.003); leave-one-cell-out B1 R² −0.0001 (max 0.0002) |
| Zero interaction: frozen Bliss reference plus block-sign-flipped replicate deviations, 5 realizations | noise share 0.988 (0.974 to 1.004); pair-wide −0.0009; B1 0.000; B2 −0.001; energy 2.05 per row |
| Reference only | pair-wide 0.264 of its own energy; its pair means are uncorrelated with the Bliss target's (r = 0.009) |
| Single-agent values perturbed per (drug, cell, dose), SD 2 / 5 / 10 | induced energy 0.25% / 1.6% / 6.4% of the target; pair-wide share 0.246 / 0.243 / 0.234 (unperturbed edge target 0.247); pair-mean correlation 0.9999 / 0.9992 / 0.9967; the perturbation itself is 0.1–0.3% pair-wide |

The estimators are calibrated: noise-only data return a noise share near 1 and no pair-wide structure. Pair-wide structure needs pair identity across cells. A shared-reference error of plausible size moves the target by a few percent, almost all of it pair-by-cell.

## Within-batch versus cross-batch reproducibility

The 240 replicate pair-cells (16 pairs × 15 cell lines) were measured in two batches. Each batch alone was placed in the frozen contrast (single-agent values shared, as in the data). Moments, in contrast units squared, are per pair-cell:

| Bliss target | Unadjusted | Cell-line batch shift removed |
|---|---:|---:|
| Within-batch reproducible $E[t_b^2] - v_b$ | 38.5 | 41.0 |
| Cross-batch $E[t_1 t_2]$ | 18.3 | 36.0 |
| Ratio, overall | 0.48 [0.08, 0.74] | 0.88 [0.65, 1.01] |
| Pair-wide, within → cross | 19.3 → 5.2 | 25.5 → 22.8; ratio 0.89 [0.52, 0.97] |
| Pair-by-cell, within → cross | 19.2 → 13.1 | 15.5 → 13.2; ratio 0.85 [0.62, 1.16] |
| Mean squared difference / within-batch noise | 4.47 | 1.79 |
| Correlation of pair-cells / pair means / pair-centred | 0.64 / 0.79 / 0.55 | 0.71 / 0.81 / 0.66 |

- **Measured response:** after the shift the overall ratio is 0.89 [0.71, 1.01] and the correlation 0.77.
- **Reference choice:** using each batch's own single-agent edges changes nothing, because they are the same values.
- **Shift:** batch 2 is shifted by cell line (mean 4.9 contrast units). A complete re-run would remove this through the cell-line part of the contrast.
- **Beyond the shift:** about 12% of the within-batch reproducible energy is batch-specific (5.0 units² against a within-batch noise of 5.7 units² per batch). Both the pair-wide and pair-by-cell components replicate.

Cross-cell transfer also predicts the independent batch. These are leave-one-cell-out predictions from batch-1 contrasts at all other cells, for the 16 pairs:

| Scored against | B1 | B2 | B2 − B1 |
|---|---:|---:|---:|
| Batch 1, Bliss | 0.464 | 0.497 | 0.033 [−0.030, 0.106] |
| Batch 2, Bliss | 0.492 | 0.523 | 0.030 [−0.015, 0.086] |
| Batch 1, response | 0.389 | 0.471 | 0.082 [0.023, 0.173] |
| Batch 2, response | 0.438 | 0.482 | 0.045 [−0.004, 0.100] |

The re-tested pairs are not typical. Under the frozen target their pair-wide share is 0.449 (all pairs 0.242) and their energy 26.9 per row (all pairs 20.6); 10 of the 16 contain MK8776, AZD1775 or MK4827.

Two extrapolations bracket the cross-batch ceiling for the full screen:
- If batch variance scales with replicate noise (ratio 0.8 to 0.9), the batch-specific share is about 0.08 and the cross-batch reproducible share about 0.82.
- If batch variance is constant in absolute terms (4.5 to 5.0 units²), the batch-specific share is about 0.23 and the ceiling about 0.67.

The reported "90% reproducible" is a within-batch technical figure conditional on shared single-agent values. It is not a biological ceiling.

## What is and is not identifiable

- **Identifiable:** pair-wide versus pair-by-cell energy of the contrast; replicate noise within a batch; out-of-sample predictable shares under the stated predictors; batch-specific variance for 16 pairs.
- **Not identifiable from this export:**
  - single-agent (reference) measurement error;
  - plate, day or run structure, and therefore whether the transferable cell similarity reflects biology or co-processed cell lines;
  - batch variance for the 567 pairs that were never re-tested;
  - pure cell-line effects, which the contrast removes by definition.
