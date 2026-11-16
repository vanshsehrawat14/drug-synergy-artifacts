# Comparative controls

Code: `experiments/interaction_structure/anchors.py`.

Outputs, in `results/interaction_structure/`:
- `anchor_experiment.json` (pooled values, intervals, contrasts, selection grids);
- `ANCHOR_INFORMATION_CURVE.csv` (target × k × anchor seed × fold × model);
- `anchor_pairs.csv.gz` (per pair, seed-averaged);
- `figures/fig_anchor_information.*`.

## Design

- **Folds and anchors:** for each held-out pair of the frozen five pair folds, k anchor cell lines (k = 0, 1, 2, 4, 8, 16) come from a hash-seeded permutation (20 seeds).
- **Evaluation rows:** the 23 cell lines at positions 16 to 38, identical for every k and model. That is 13,409 pair-cells per seed.
- **R²:** 1 − SSE/SSE(B0) on those rows, row-weighted, pooled over the 583 pairs after averaging each pair's errors over seeds.
- **Uncertainty:** pair-cluster bootstrap and delete-one-drug jackknife intervals. Anchor-seed spread is small: the SD of pooled R² across seeds is 0.006 for B1 and 0.009 for B2 at k = 16.

Models:
- **B0:** zero, i.e. no same-pair information.
- **B1:** empirical-Bayes pair intercept $k\tau^2/(s^2+k\tau^2)\times$ anchor mean.
- **B2:** best linear predictor under a cross-cell covariance. It shrinks toward compound symmetry with weight λ and keeps r eigen-directions of the noise-corrected covariance, with λ and r chosen per (target, fold, k) on validation fold f+1.
- **B4:** B1 or B2 fed the anchors of a donor pair from the training folds that shares one drug.
- **B5a:** the same with a donor sharing no drug.
- **B5b:** B2 fed the pair's own anchor values on deranged cells.

## Anchor information curve (Bliss residual contrast)

| k | B1 | B2 | B2 − B1 | B4 (B2) | B5a (B2) | B5b (B2) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.049 [0.031, 0.067] | 0.081 [0.058, 0.106] | 0.033 [0.024, 0.041] | −0.087 | −0.091 | — |
| 2 | 0.083 [0.056, 0.110] | 0.130 [0.098, 0.163] | 0.047 [0.038, 0.056] | −0.132 | −0.136 | −0.013 |
| 4 | 0.123 [0.090, 0.157] | 0.178 [0.141, 0.215] | 0.055 [0.047, 0.063] | −0.176 | −0.193 | 0.040 |
| 8 | 0.164 [0.125, 0.203] | 0.227 [0.188, 0.267] | 0.063 [0.054, 0.072] | −0.213 | −0.231 | 0.089 |
| 16 | 0.194 [0.154, 0.236] | 0.280 [0.241, 0.321] | 0.086 [0.071, 0.102] | −0.267 | −0.286 | 0.085 |
| all 38 (LOCO) | 0.215 [0.175, 0.258] | 0.360 [0.319, 0.400] | 0.144 [0.116, 0.172] | | | |

At k = 16:

- **Drug-jackknife intervals:** B1 [0.107, 0.281], B2 [0.190, 0.370], B2 − B1 [0.048, 0.124].
- **Pairs improved:** B2 beats B1 for 70.5% of pairs and in all five folds. Per-fold B1 is 0.28, 0.12, 0.11, 0.30, 0.13; B2 is 0.35, 0.20, 0.22, 0.39, 0.20.
- **RMSE:** B0 4.54, B1 4.08, B2 3.85.
- **Pair-macro R² (mean/median):** B1 0.125/0.041, B2 0.180/0.133.
- **Noise:** the evaluated rows carry a noise share of 0.098 in every condition. Raw R² is not noise-corrected anywhere in this table.

| Contrast at k = 16 | Estimate | Pair bootstrap | Drug jackknife |
|---|---:|---:|---:|
| B1 − B0 (pair propensity) | 0.194 | [0.154, 0.236] | [0.107, 0.281] |
| B2 − B1 (transferable cell profile) | 0.086 | [0.071, 0.102] | [0.048, 0.124] |
| B1 − B4 (same pair vs donor sharing a drug) | 0.372 | [0.334, 0.412] | [0.271, 0.474] |
| B1 − B5a (same pair vs unrelated donor) | 0.390 | | |
| B2 − B5b (correct vs deranged anchor cells) | 0.195 | [0.174, 0.216] | [0.147, 0.243] |

All three contrasts are material by the frozen rule: at least 0.01, with both intervals excluding zero. B2 − B1 also meets the stricter "substantial" threshold (at least 0.02 at k = 16 and at LOCO).

What the controls show:
- **Donor pairs:** strongly negative at every k, and a shared drug does not help. B4 and B5a differ by 0.02 at most. After the contrast, drug identity carries no transferable interaction information, so the gain is specific to the same pair.
- **Deranged anchors:** B2 falls below B1. The extra skill depends on which cell line each anchor value came from.

## Target sensitivity (k = 16)

| Target | B1 | B2 | B2 − B1 | Residual R² of B2 over B1 (experiment C) |
|---|---:|---:|---:|---:|
| Bliss residual contrast (primary) | 0.194 | 0.280 | 0.086 [0.071, 0.102] | 0.107 [0.089, 0.126] |
| Train-only baseline residual (no test data in any value) | 0.196 | 0.283 | 0.087 [0.071, 0.105]; drug [0.043, 0.131] | 0.109 [0.088, 0.129] |
| Measured-response contrast | 0.204 | 0.317 | 0.113 [0.100, 0.128] | 0.142 [0.126, 0.159] |
| Bliss residual minus held-out-pair F2 prediction (exploratory) | 0.226 | 0.288 | 0.062 [0.053, 0.071] | 0.080 [0.069, 0.093] |
| Reference only | 0.229 | 0.476 | 0.247 [0.222, 0.270] | 0.320 |

- **Train-only target:** it is the protocol's train-only baseline and contains no held-out data anywhere. It reproduces the primary curve within 0.004 at every k up to 16. With all 38 other cells as anchors its B2 gain is larger (0.165 against 0.144).
- **After the held-out-pair model:** subtracting F2's prediction leaves B2's gain at 0.062. The transferable structure is mostly not what drug features already explain.
- **Reference only:** this deterministic monotherapy function is the most cell-structured target, as expected.

## Experiment C: residual after the pair intercept

At k = 16, the B1 residual on evaluated cells is 0.806 of the target energy, of which 0.102 is replicate noise (from the same split draws, propagated through B1). B2 removes:

- 10.7% [8.9, 12.6] of that residual;
- 12.2% of its within-batch reproducible part.

The share grows with anchors: 3.4% at k = 1, 5.1% at k = 2, 6.2% at k = 4, 7.5% at k = 8, 10.7% at k = 16. There is structured cell-context dependence beyond the intercept. Most of the reproducible residual is still not linearly transferable from the tested anchors.

## Neural pair-seen model (B3), matched condition

**Condition** (the pair-cell split of the frozen pair-seen campaign):
- Anchors are the pair's cell lines in the three training folds, about 23 per pair.
- B2 settings are chosen on validation fold f+1, as the network's early stopping was.
- Evaluation is on fold f with the panel contrast applied to out-of-fold predictions, as for B3.

The two B1/B2 versions differ only in how the anchors and covariance are built. The frozen contrast fits drug effects with test entries; "leak-free" refits them on training entries.

| Model, projected contrast R² | Frozen-contrast anchors and covariance | Leak-free (primary) |
|---|---:|---:|
| B1 pair intercept | 0.209 [0.168, 0.252] | 0.177 [0.143, 0.212]; drug [0.106, 0.248] |
| B2 cross-cell covariance | 0.309 [0.268, 0.349] | 0.216 [0.181, 0.250]; drug [0.140, 0.291] |
| B3 neural pair-seen (3 seeds) | 0.218 [0.162, 0.276] | 0.218 |
| F2 held-out pair (same rows) | −0.050 | −0.050 |

Leak-free, per pair-cell-split fold:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| B1 | 0.181 | 0.181 | 0.190 | 0.168 | 0.166 |
| B2 | 0.222 | 0.218 | 0.247 | 0.206 | 0.189 |
| B3 | 0.221 | 0.256 | 0.236 | 0.178 | 0.206 |
| B3 − B1 | 0.041 | 0.075 | 0.046 | 0.010 | 0.040 |
| B3 − B2 | 0.000 | 0.038 | −0.011 | −0.028 | 0.016 |

Under matched, leak-free information:
- **B1** reaches 81% of the neural model's R², below the frozen 90% threshold for "the intercept explains almost all of the pair-seen gain".
- **B3 − B1** is +0.041 and positive in 5/5 folds.
- **B3 − B2** is +0.003, with mixed signs.
- **B2 − B1** is +0.039 [0.029, 0.049].
- **Reading:** the network captures the pair intercept plus about as much cell-specific transfer as a linear covariance predictor. It does not leave linear cross-cell structure unused. B3 has no pair-bootstrap interval for these differences because its per-pair energies remain on the cluster.

The frozen-contrast column is inflated, and the inflation is attributed in TARGET_AND_LEAKAGE_AUDIT.md. With settings fixed at λ = 0.5 and full rank:

| Anchors / covariance | B1 | B2 |
|---|---:|---:|
| frozen / frozen | 0.209 | 0.308 |
| frozen / leak-free | 0.208 | 0.262 |
| leak-free / frozen | 0.183 | 0.256 |
| leak-free / leak-free | 0.177 | 0.209 |

**Why the matched gain is smaller than the curve's.** The matched B2 gain (0.039 at about 23 anchors) is below the anchor-curve gain at k = 16 (0.086). In the pair-cell split, the covariance is estimated from 60% of each pair's cells with pairwise-complete moments, and anchors carry drug effects fitted on fewer entries. In the original pair folds, the covariance comes from complete training pairs.

**Measured-response contrast.** B3 is 0.463 against leak-free B1 0.183 and B2 0.229. The network also predicts the monotherapy-determined reference part at the target cell (f2_base reaches 0.279 without the pair), which anchors of the same pair cannot supply. That comparison is not about same-pair information.

## Related-work boundary

Four papers set the boundary:

- **One-hot news** (Çandır et al., *Bioinformatics* 42(3) btag040, 2026, doi:10.1093/bioinformatics/btag040) reports that replacing drug and cell-line features with one-hot identities matches published synergy models.
- **SynVerse** (Tasnina, Haghani and Murali, *Brief. Bioinform.* 26(6) bbaf676, 2025, doi:10.1093/bib/bbaf676) evaluates models across four splitting strategies and ablations, and reports that biologically meaningful features do not drive performance.
- **BATCHIE** (Tosh et al., *Nat. Commun.* 16, 2025, doi:10.1038/s41467-024-55287-7) designs combination screens by Bayesian active learning.
- **The active-learning guide** (Wang, Allauzen, Nghe and Opuu, *Sci. Rep.* 2025, doi:10.1038/s41598-025-85600-3) finds that cellular-environment features matter more than molecular encoding.

Identity shortcuts, split sensitivity and adaptive selection are therefore established. The results here add a different kind of statement: an out-of-sample decomposition of what same-pair supervision supplies (pair intercept versus transferable cell profile versus irreducible pair-by-cell variation), with controls that exclude identity covariation and leakage, and an independent-batch check. These summaries are from the papers' abstracts; full texts were not reviewed here.
