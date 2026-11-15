# Statistical and computational audit

## Units and dependence

- **Evaluation unit.** The held-out pair (583 pairs). Bootstrap intervals resample pairs (2,000 draws). Pair-cells within a pair share the pair offset and are never treated as independent.
- **Shared drugs.** Pairs sharing a drug are dependent. Every main interval is paired with a delete-one-drug jackknife; these are typically about twice as wide as the pair bootstrap. Cell-jackknife intervals are given for the variance components.
- **Folds.** The five folds are partitions of one screen, not independent studies. Fold-level values are reported as consistency checks, not as replication.
- **Missingness.** The pair-cell matrix is complete. 75 re-measured pair-cells with changed doses are excluded from replicate comparisons only.

## Targets and leakage

- **Test-set dependence of the frozen contrast.** It fits drug effects within (pair fold, cell line) panels that contain the evaluated entries. It is the estimand of record, so targets were not redefined. Two consequences were found and handled:
  1. **Original pair folds.** A held-out pair's anchors carry other held-out pairs' values. A train-only target, whose baseline is fitted on other folds and which contains no held-out data, reproduces the anchor curves within 0.004 for $k\le16$.
  2. **Pair-cell split.** Anchors and a covariance estimated in the same panels carry test entries. Matched B1 and B2 inflate from 0.177 and 0.216 to 0.209 and 0.309. The attribution in supplement Table S6 separates the two paths. All matched comparisons use contrasts refitted on training entries.
- **Canary test.** Setting evaluated cells to +100 leaves B1 and B2 predictions unchanged (`tests/test_interaction_gap.py`).
- **Hyperparameters.** B2 settings are chosen on validation fold $f+1$ with covariance from the remaining folds, then refitted. No held-out pair influences selection.
- **Anchor masks.** Masks are SHA-256 hashes of (pair, seed) and never read targets. Evaluation rows are identical across $k$ and models.
- **Exploratory target.** "Bliss minus held-out-pair F2 prediction" uses out-of-fold predictions whose training data included the test fold for the covariance pairs. It is labelled exploratory and is not used for any conclusion.

## Variance decomposition

- **Pair-wide component.** The cross-cell U-statistic uses no target twice and is unbiased under cross-cell independence of noise. Noise comes from Lemma 4. The pair-by-cell term is the remainder and inherits the uncertainty of both.
- **Descriptive versus predictive.** In-sample pair-mean shares (descriptive) are reported separately from out-of-sample predictable shares (leave-one-cell-out B1 and B2).
- **Calibration.**
  - Zero-interaction simulation: noise share 0.988, pair-wide −0.0009.
  - Within-panel pair permutation: pair-wide 0.0007 ± 0.0016.
- **Replicate noise is within batch.**
  - Unadjusted, the cross-batch mean squared difference is 4.47 times the within-batch prediction.
  - After the cell-line shift it is 1.79 times, and 88% of reproducible energy replicates.
  - The full-screen extrapolation (0.67–0.82) depends on how batch variance scales and rests on 16 atypical pairs. Both assumptions are stated.
- **Reference noise.** Not identifiable, because single-agent values are shared across replicates, pairs and both batches. Sensitivity: an assumed SD of 5 inhibition points moves 1.6% of target energy.

## The pair-seen comparison

- **B3 provenance.**
  - Per-pair errors were recomputed on RebelX from the 15 saved prediction files, with all SHA-256 hashes matching the campaign record.
  - The recomputed pooled $R^2$ equals the recorded 0.21823552298723448 exactly.
  - Target energies computed on the cluster and locally agree to 0.
  - No model was retrained.
- **Paired inference.** B1, B2 and B3 are evaluated on identical rows through the same contrast.

  | Difference | Estimate | Pair interval | Drug interval |
  |---|---:|---:|---:|
  | B3 − B1 | 0.041 | [0.004, 0.080] | [−0.058, 0.141] |
  | B3 − B2 | 0.003 | [−0.031, 0.036] | [−0.079, 0.084] |
  | B2 − B1 | 0.039 | [0.029, 0.049] | [0.014, 0.063] |

  The frozen materiality rule (difference at least 0.01, both intervals excluding 0) is met only by B2 − B1.
- **Drug jackknife without refitting.** It deletes pairs containing each drug but does not refit the projection. The earlier pair-seen audit refitted it, giving a different drug interval for the pair-seen gain ([0.069, 0.468] against [0.115, 0.422]). The manuscript quotes the refitted interval for the gain and the non-refitted one only for the new paired differences.
- **Training information.** Pair-seen and held-out-pair training use different supervision. The comparison isolates information, not capacity. Training-row counts differ by under 0.5%.

## Screening

- **Ties and budgets.** No ties at any cutoff. Turnover is stable across budgets.
- **Margins.** Entrants come from a median 3.60 below the cutoff; exits sit a median 0.17 above it, with unchanged scores.
- **Resampling.**
  - Combination replicates only: 1,000 bootstrap resamples and 200 half-depth splits.
  - The C1 change is present in every resample.
  - Single-agent noise cannot be resampled (shared values).
- **Fidelity.** Native reconstruction against published scores: RMSE 0.704, $r$ 0.9990, one top-5% candidate different.

## Multiplicity and status

- **Exploratory status.** Everything after the frozen October protocols is exploratory.
- **Pre-declared controls.**
  - The interaction-gap design, its controls and its materiality thresholds were committed before outcomes (commit c0a785c). The freeze file discloses the inspections that preceded it.
  - The matched B1-share rule (≥ 90% of B3) was not met (81%), and this is reported.
- **Confirmatory test.** The only one on an untouched screen (NCATS) failed and is reported as a main negative result.

## Reproducibility

- **Seeds and determinism.** All random procedures are seeded and documented (`REPRODUCIBILITY.md`). Deterministic recomputations reproduce frozen values bit for bit:
  - noise shares;
  - Gate 2 outputs;
  - the B3 pooled $R^2$.
- **Number check.** `scripts/check_numbers.py` passes 116/116.
- **Tests.** The full test suite passes; see `REPRODUCIBILITY.md` for the latest run.
