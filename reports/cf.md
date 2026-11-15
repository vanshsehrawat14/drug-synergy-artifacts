# Neighborhood collaborative filtering on the shared benchmark

| Part | Author | Files |
|---|---|---|
| Raw-response CF evaluation: target, tuning grid, five-fold evaluation | Sharlene Kim | `experiments/neighborhood_cf/` (her commit `d6e9550` on `sharlenekim-neighborhood-cf`, cherry-picked with her authorship) |
| Residual kNN, additive ridge, folds and metrics it imports | shared preliminary pipeline | `experiments/preliminary/pairlearners.py`, `common.py`, unchanged from `main` |
| Benchmark adapter, checks, extension, pair-cell comparison | Vansh | `cs622/cf.py`, `configs/cf_study.json`, `tests/test_cf.py` |

## 1. Her method

| | |
|---|---|
| Target | Mean inhibition (percentage points) over the 16 positive-dose cells of each block, then over the pair-cell's 4–8 replicate blocks |
| Matrix | 583 pairs × 39 cell lines, complete: 22,737 values |
| Base model | Drug-in-cell additive ridge Y ≈ Hβ (H: two-drug indicator; β: 38 × 39), λ ∈ {0.01, 0.1, 1, 10, 100} |
| Neighbourhood | For a held-out pair (x, y): the k training partners z of x most similar to y. Their residuals Y(x, z) − Hβ are averaged with similarity weights (negative weights set to 0). Both orientations are averaged and the result added to the ridge prediction |
| Similarity | Pearson correlation of two drugs' training residual vectors (39 cells) over their common training partners, multiplied by n / (n + shrink) |
| Missing values | Only held-out pairs are missing. A mask excludes them from similarities and neighbour sets; they are never read as zeros. With no usable neighbour the residual is 0 |
| Splits | Shared DeepSynergy pair folds; validation (f + 1) mod 5; the other three folds train; no refit |
| Hyperparameters | k ∈ {3, 5, 10, 20}, shrink ∈ {2, 5, 20} and λ, per fold on validation pair-cell MSE |
| Her reported result | Pair-cell MSE 35.904, Pearson 0.960, contrast R² 0.143; additive ridge 49.473 |

## 2. Reproduction and leakage audit

**Reproduction.** Her two scripts were run unchanged on her branch.
- `target_validation.json` is identical.
- Every number in `cf_metrics.json` agrees to within 2.2e-16 relative.
- The per-fold k, shrink and λ are the same.

**Alignment with the benchmark.**
- Her target equals the mean of each pair-cell's benchmark rows (largest difference 1.4e-14) for all 22,737 pair-cells.
- Her pair folds equal the benchmark folds for all 583 pairs.
- Drug names differ only in punctuation (`5-FU` vs `5FU`).

**Held-out pairs.** The test pair enters neither the ridge, the similarities nor the neighbour sets, and validation pairs are not used for test predictions.
- **Check.** Refitting fold 0 with every test-pair label replaced by noise reproduces the saved predictions bitwise for all three runs (`results/cf/verification.json`).
- **Tests.** `tests/test_cf.py` checks the same invariance, and that `cs622.cf.pair_cell` reproduces her `predict_cf` bitwise.

**Contrast metric.** Her overall contrast R² (0.143) projects drug main effects off all 583 pairs at once, so fold-to-fold differences in the additive fit count as contrast; this is also why her additive ridge scores −0.194.
- Within each test fold her CF scores 0.23–0.31.
- The benchmark's projection within fold and cell gives 0.273.

**Verdict.** Her CF is a held-out-pair method for drugs seen in training, which is exactly the benchmark task, so no cold-start variant is needed. Its only mismatch with the benchmark is the target: it predicts the pair-cell mean, with no dose dimension.

## 3. Two tasks

60.7% of dose-level response variance lies within pair-cells (`results/summary/variance_shares.json`). Every model is therefore scored twice.
- **Her task.** Each model's predictions are averaged per pair and cell line and compared with the pair-cell mean response, each pair-cell weighted equally as she does (`results/compare/pair_cell.csv`).
- **Shared benchmark.** Her pair-cell value is given to every dose row of that pair-cell, the smallest possible adapter. No dose information is added (`results/compare/final.csv`).

## 4. Extension, selected on validation only

**Protocol.** `configs/cf_study.json` was committed (`ac491f2`) before any variant ran. The decision (`results/cf/decision.json`, `06f4ab3`) was committed before any test fold of the extension was predicted.

**CF-stage variants.**
- her grid (the default);
- wider k and shrink;
- uncentred (cosine) similarity;
- per-cell-line similarity, averaged with her all-cell similarity.

**Dose-stage variants.** A within-pair-cell profile fitted per cell line on training rows minus their pair-cell mean, centred so that it sums to zero in every pair-cell.
- **Additive.** Drug × own dose rank.
- **Partner.** Drug × own rank × partner's rank.
- **Inputs.** Identities, cell line and each drug's dose rank (1–4) within its block, all known for an unseen pair. Ridge λ per fold on validation.

**Rule.** Each stage keeps its default unless a challenger's pooled validation error is at least 1% lower. Because the profile sums to zero within each pair-cell, the two stages' errors add without a cross term.

| CF stage | Validation MSE, pair-cell part | Dose stage | Validation MSE, within pair-cells |
|---|---:|---|---:|
| Her grid | 36.26 | None (her prediction) | 699.9 |
| Wider k, shrink | 36.25 | Additive | 198.1 |
| Cosine | 36.33 | **Partner rank (selected)** | **130.8** |
| **Per-cell similarity (selected)** | **31.69** | | |

**Selected configuration.** Its validation MSE on dose rows is 162.5, against 736.2 for her prediction on every dose row. The dose ridge chose λ = 1, the smallest value in its grid, in every fold; there are about 75 training rows per parameter, so regularization barely matters.

**Ablation.** Her CF stage with the selected dose profile (`cf_shar_dose`) was also run on test, to separate the two stages. No decision uses it.

## 5. Five-fold results

**Her task: pair-cell means** (22,737 pair-cells; 95% pair bootstrap; networks average three seeds).

| Model | RMSE [95% CI] | R² | Pearson | MSE vs her CF |
|---|---|---:|---:|---|
| F2 network | 5.55 [5.28, 5.83] | 0.932 | 0.966 | −14.2%, lower |
| F2 GP, all rows (exploratory) | 5.62 [5.35, 5.92] | 0.930 | 0.965 | |
| **CF extension** | **5.65 [5.39, 5.93]** | 0.930 | 0.964 | −11.1%, lower |
| F0 network | 5.78 [5.50, 6.06] | 0.926 | 0.963 | −7.1%, not established |
| **Her CF** | **5.99 [5.74, 6.28]** | 0.921 | 0.960 | – |
| F0 GP, all rows (exploratory) | 7.05 [6.79, 7.33] | 0.890 | 0.944 | |
| Bliss reference | 7.63 [7.38, 7.90] | 0.871 | 0.939 | +62.3%, higher |
| Frozen GP selection | 7.65 [7.36, 7.93] | 0.871 | 0.933 | |
| DeepSynergy, dose-aware | 9.05 [8.71, 9.40] | 0.819 | 0.906 | |
| Additive ridge | 13.56 [13.26, 13.85] | 0.595 | 0.771 | +411.8%, higher |

Paired results on this task (`results/compare/pair_cell_comparisons.csv`).
- **Her CF.** It is 7.6% behind the F0 network (not established) and 16.6% behind the F2 network (higher error). It beats Bliss by 38.4% and the additive ridge by 80.5% (both lower error).
- **The extension.** Its −4.3% against the F0 network is not established: the bootstrap excludes 0 but the drug jackknife does not. The same holds for +3.6% against the F2 network and +1.1% against the F2 GP. It beats Bliss by 45.2% (lower error).

**Shared benchmark: every dose row** (1,475,328 rows).

| Model | Information | RMSE [95% CI] | R² | Pearson | MAE | Contrast R² [drug jackknife] | Interaction-residual R² | MSE vs Bliss | MSE vs F0 network |
|---|---|---|---:|---:|---:|---|---:|---|---|
| F0 network | identities, doses | 12.07 [11.82, 12.35] | 0.874 | 0.935 | 8.06 | 0.306 [0.173, 0.439] | −0.013 | −26.0%, lower | – |
| **CF extension** | identities, dose ranks | **12.78 [12.55, 13.05]** | 0.858 | 0.926 | 9.00 | 0.338 [0.229, 0.447] | 0.034 | −17.0%, lower | +12.2%, higher |
| Her CF stage + dose profile (ablation) | identities, dose ranks | 12.94 [12.70, 13.20] | 0.855 | 0.925 | 9.16 | 0.273 [0.183, 0.363] | −0.061 | −14.9%, lower | +14.9%, higher |
| Bliss reference | training-pair monotherapy | 14.03 [13.78, 14.29] | 0.829 | 0.914 | 9.40 | 0.315 [0.193, 0.437] | 0 | – | +35.1%, higher |
| Additive ridge | identities, doses | 24.18 [23.82, 24.53] | 0.493 | 0.702 | 19.26 | 0.001 | −0.459 | +197.0%, higher | |
| **Her CF, pair-cell value on every dose row** | identities | **27.13 [26.60, 27.61]** | 0.361 | 0.601 | 21.94 | 0.273 [0.183, 0.363] | −0.061 | +274.1%, higher | |

**By Bliss-expected inhibition** (RMSE; extension vs F0 network): 13.65 vs 13.20 at R < 25, 13.20 vs 13.35 at 25–50, 12.06 vs 11.71 at 50–75 and 11.69 vs 9.36 at R ≥ 75. The extension's deficit is concentrated at high inhibition, where it is 11.5% above Bliss in MSE.

**Compute, five folds, CPU.** Her CF takes 6 s; the extension takes 65 s, with the dose profile and the per-cell similarities dominating. By comparison, the frozen GP selection took 69,332 CPU-s and the networks 900–1,100 GPU-s.

## 6. Readings

- **The method transfers.** It needs only which training pairs were measured and their pair-cell means, so it applies directly to held-out pairs.
- **Identity neighbours carry the pair-cell mean.** Without doses or monotherapy, her CF reaches pair-cell R² 0.921, close to the F0 network (not established) and well above Bliss. This fits the variance decomposition: 76% of between-pair-cell variance is drug main effects within cell, which the ridge and the neighbours share across partners.
- **Pair-specific contrast.** Her CF reproduces 27% of the contrast and the extension 34%, against 31% for the F0 network and 36% for the F2 network; all intervals overlap. Of the contrast Bliss leaves, the extension explains 3.4% (interval [−6.7%, 13.4%]). As for the networks, there is no established interaction-specific gain.
- **The benchmark gap is the dose dimension.** Without doses her prediction misses the within-pair-cell 61% of variance, giving RMSE 27.1 on dose rows. A dose profile shared across partners recovers most of it (12.94). What remains against the F0 network (+12.2% MSE) sits at high inhibition, where a profile averaged over partners cannot place the plateau of a specific pair.
- **Per-cell similarity helps.** It lowers pair-cell MSE by 11.1% on test (lower error, every fold), matching the 12.6% validation gain. Drug neighbourhoods differ between cell lines.

## 7. Limitations

- **Status of the extension.** It was designed after the shared results were known. Its protocol was frozen before it ran, so it is a validation-selected extension, not a pre-registered hypothesis.
- **Inner selection.** k, shrink and λ are chosen on the same validation fold that scores the variant, so validation errors are optimistic, slightly more so for larger grids. The 1% margin and the test result (−11.1%) argue against this driving the choice.
- **No pair-specific dose shape.** The dose profile averages over partners.
- **Single run.** The method is deterministic, so there are no seeds. Intervals are grouped over pairs and drugs only.
- **No predictive SD.** CF has no uncertainty estimate, so the GP calibration analysis does not apply.

## Reproduction

```sh
python experiments/neighborhood_cf/build_raw_response_targets.py      # her target matrix (needs data/processed/drugcomb_oneil)
python experiments/neighborhood_cf/run_neighborhood_cf.py             # her evaluation, unchanged
python -m cs622.cf study --config configs/cf_study.json --output results/neural/runs/cf_study
python -m cs622.cf select --config configs/cf_study.json --study results/neural/runs/cf_study --output results/cf
python -m cs622.cf run --config configs/cf_study.json --cf shar --dose none --output results/neural/runs/cf_shar
python -m cs622.cf run --config configs/cf_study.json --decision results/cf/decision.json --output results/neural/runs/cf_selected
python -m cs622.cf run --config configs/cf_study.json --cf shar --dose partner --output results/neural/runs/cf_shar_dose
R=results/neural/runs
python -m cs622.cf verify --method cf_shar=$R/cf_shar --method cf_selected=$R/cf_selected --method cf_shar_dose=$R/cf_shar_dose --output results/cf/verification.json
M="--method gp_his=$R/gp_cap1500 --method gp_frozen=$R/gp_final --method gp_full=$R/gp_gpu_full --method gp_full_w0=$R/gp_gpu_full_w0 --method gp_f0=$R/gp_gpu_f0_drug_dose --method gp_f2=$R/gp_gpu_f2_residual_mono --method cf_shar=$R/cf_shar --method cf_shar_dose=$R/cf_shar_dose --method cf_selected=$R/cf_selected"
python -m cs622.sprint_report verify --output results/compare
python -m cs622.sprint_report analyze --output results/compare $M
python -m cs622.unified --output results/compare $M
python -m cs622.cf compare --method gp_frozen=$R/gp_final --method gp_f0=$R/gp_gpu_f0_drug_dose --method gp_f2=$R/gp_gpu_f2_residual_mono --method cf_shar=$R/cf_shar --method cf_selected=$R/cf_selected --output results/compare
```
