# Gaussian process: frozen study, full-data extension and shared comparison

## Who did what

| Part | Author | Code |
|---|---|---|
| Kernel, encoder, block cap, scikit-learn fit, RBF vs Matern 5/2 choice | Kingsley Ukandu | `GP_Regression/src/project_utils.py`, unchanged |
| Shared folds, validation fold (f+1)%5, no refit | Vansh, adapter | `cs622/gp_shared.py` (`fit_cell`, `run`) |
| Inducing-point predictor: his hyperparameters, conditioned on every training row | Vansh, extension | `cs622/gp_shared.py` (`InducingPredictor`) |
| Exact GP on every training row in float64 torch; dose, monotherapy and residual variants | Vansh, extension | `cs622/gp_gpu.py` |
| Selection, uncertainty and prediction checks | Vansh | `cs622/gp_select.py`, `cs622/gp_uncertainty.py`, `cs622/gp_verify.py` |

His model, as run everywhere below:
- one GP per cell line on drug identities and standardized log doses;
- an additive plus symmetrized-product kernel on unordered drug-dose arms (shared similarity ρ = 0.25, interaction weight w = 1);
- `ConstantKernel × pair kernel + WhiteKernel`, `normalize_y`, one L-BFGS-B run from his start values and bounds;
- RBF vs Matern 5/2 per cell by validation RMSE;
- training capped to whole blocks: 1,488 of the 22,464 training rows per cell (6.6%).

All choices below use validation fold (f+1)%5 only. The study fold is 0: train on folds 2–4, validate on fold 1, 39 cell lines, 295,296 validation rows. Test folds are scored once, after every choice.

## 1. Frozen CPU study

Plan: `configs/gp_study.json`, committed before any output. Amendment 1 dropped cap 6,000 for cost (about 24 h against an 8 h budget). Amendment 2 applied the frozen kernel-variant clause, which the first selection code reported but did not apply.

| Method | Rows conditioned | Val RMSE | Val MSE | NLL | 90% cover | s/fit | Est. 5-fold h |
|---|---:|---:|---:|---:|---:|---:|---:|
| His GP, cap 1,500, exact | 1,488 | 18.85 | 355.4 | 4.345 | 0.947 | 58 | 1.6 |
| Cap 1,500, inducing extension | 22,464 | 15.94 | 254.2 | 4.252 | 0.976 | 99 | 2.7 |
| Cap 3,000, exact | 2,992 | 17.16 | 294.6 | 4.248 | 0.948 | 171 | 4.6 |
| **Cap 3,000, inducing extension** | 22,464 | **15.82** | **250.1** | 4.199 | 0.966 | 198 | 5.4 |

- **Selection.** Cap 3,000 with the inducing extension has the lowest MSE among eligible candidates. Cap 1,500 inducing is 1.6% worse, outside the 1% tie rule.
- **Kernel variants at cap 1,500.**
  - w = 0 (no interaction term): −11.3% MSE, 90% coverage 0.864.
  - ρ = 0: −3.5%.
  - ρ = 0.5: +5.7%.
- **Variant check at the selected scale (amendment 2).** w = 0 changes MSE by −0.4% (249.1 vs 250.1), short of the 1% bar. The default kernel stands.
- **Frozen selection.** His kernel and hyperparameter fit on a 3,000-row block cap, then the inducing-point posterior conditioned on every training row.
- **Cap 6,000 on eight cells (amendment 1, run after production).** On the same eight cells (`results/gp/same_cells.csv`), cap 6,000 exact scores validation MSE 235.8. The frozen selection scores 225.7 and his GP 328.6. Dropping cap 6,000 for cost did not cost accuracy. Its 4.5 h estimate in `study_summary.csv` covers these eight cells only; all 39 cells would take about 22 h.
- **Run health.** No convergence warnings and no hyperparameters at a bound in any group.

## 2. GPU extension: his kernel on every row

**Why the exact GP is feasible.** ONEIL measures each dose pair in four replicate blocks. The 22,464 training rows of a cell are about 5,616 distinct inputs. Collapsing replicates into their mean, count and within-replicate sum of squares leaves the likelihood and the posterior unchanged.
- On A2058, the collapsed and direct log marginal likelihoods agree to 1.5e-15 relative. The direct version uses a 23,296-row matrix and 33 GB of GPU memory.
- No approximation is needed for coverage: an exact fit on all rows takes 2.6 s median on an A100, with 8.2 GB peak.

**Faithfulness to his model.** The torch port reads his start values, bounds and alpha from his `make_gp`. On his capped rows it reproduces his scikit-learn fits:
- unit tests: likelihood to 1e-9, gradients, fitted hyperparameters and predictions;
- study: identical pooled validation MSE at cap 1,500 (355.43) and cap 3,000 (294.55) over all 39 cells.

### Coverage and approximation (plan `configs/gp_gpu.json`)

| His kernel | Distinct inputs | Val MSE | RMSE | NLL | 90% cover | GPU s/fit |
|---|---:|---:|---:|---:|---:|---:|
| 1,488 rows | 1,344 | 355.4 | 18.85 | 4.345 | 0.947 | 0.12 |
| 2,992 rows | 2,368 | 294.6 | 17.16 | 4.248 | 0.948 | 0.31 |
| 6,000 rows | 3,808 | 264.3 | 16.26 | 4.188 | 0.943 | 0.98 |
| 12,000 rows | 5,376 | 251.2 | 15.85 | 4.157 | 0.938 | 2.2 |
| all 22,464 rows | 5,616 | 247.9 | 15.74 | 4.149 | 0.934 | 2.6 |
| his 1,488-row hyperparameters, exact on all rows | 5,616 | 249.7 | 15.80 | 4.180 | 0.955 | 0.16 |

- **Data, not hyperparameters.**
  - Conditioning on every row lowers validation MSE by 30% relative to his budget.
  - Of that gain, 98% is available with his own 1,488-row hyperparameters (249.7 vs 247.9).
  - The frozen inducing extension recovers it at 250.1.
- **Inducing points**, with the all-row hyperparameters:

  | Inducing points | 256 | 512 | 1,024 | 2,048 | 4,096 |
  |---|---:|---:|---:|---:|---:|
  | Val MSE | 289.7 | 265.1 | 254.9 | 248.2 | 247.4 |

  The exact all-row MSE is 247.9. So 2,048 inducing points reach the exact posterior, and the gain from more rows is not specific to the approximation.

### Features and kernel (amendment `configs/gp_gpu_amendment.json`, before any coverage or feature output)

| Variant, all rows | Information | Val MSE | RMSE | NLL | 90% cover |
|---|---|---:|---:|---:|---:|
| His kernel | F0 | 247.9 | 15.74 | 4.149 | 0.934 |
| **Per-drug standardized log dose** | F0 | **177.8** | 13.33 | 3.962 | 0.924 |
| No interaction term (w = 0) | F0 | 249.0 | 15.78 | 4.145 | 0.911 |
| Monotherapy as a third arm coordinate | F2 | 138.9 | 11.79 | 3.823 | 0.940 |
| Bliss-residual target | F2 | 146.3 | 12.09 | 3.840 | 0.916 |
| **Bliss residual + monotherapy coordinate** | F2 | **137.5** | 11.73 | 3.805 | 0.921 |
| Bliss residual, w = 0 | F2 | 145.7 | 12.07 | 3.838 | 0.916 |

**F0.**
- **Selected: per-drug dose.** It lowers MSE by 28%.
- **Why it helps.** His pooled standardization compares absolute concentrations across drugs. Per-drug standardization compares positions within each drug's tested range.
- **Information.** It uses only the training dose design, no outcomes.

**F2.**
- **Selected: Bliss residual plus the monotherapy coordinate.**
- **Information.** The monotherapy lookup is the neural F2 lookup: training-pair values with interpolation.

**Same rows, neural models.** On the identical fold-1 validation rows, the selected networks score MSE 143.5 (F0) and 135.9 (F2).

**Interaction term.**
- **Variants that drop it.** w = 0 is within 0.5% at full data for F0. For the F2 residual, w = 0 leaves 36 hyperparameter estimates at a bound.
- **The cap-1,500 result.** The 11% advantage of w = 0 at cap 1,500 does not persist with all rows.

**No drug-similarity features.** The repository has none: the F1 matrices are unavailable and only PubChem CIDs exist. No such variant was run.

## 3. Five-fold test results

All five folds and 39 cell lines; the same 1,475,328 rows for every model (`results/gp_compare/`). Intervals are 95% pair-cluster bootstrap. Verdicts use the shared rule.

**Prediction checks** (`gp_verification.json`):
- each fold covers exactly its test rows once, with finite predictions and SDs;
- every pair lies in one fold, so training, validation and test pairs are disjoint;
- the validation fold is (f+1)%5, with one selected family per cell and fold;
- the frozen decision recomputes identically from the study files;
- refitting A2058 fold 0 from the saved settings reproduces both runs bitwise (`reproduce_*.json`).

| Model | Information | RMSE [95% CI] | R² | Pearson | MAE | Contrast R² | Interaction-residual R² | MSE vs Bliss | MSE vs F2 network |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| Selected F2 network | F2 | 11.84 [11.59, 12.13] | 0.878 | 0.937 | 7.83 | 0.359 | 0.064 | −28.7%, lower | – |
| Full-data GP, F2: Bliss residual + monotherapy (exploratory) | F2 | 11.75 [11.49, 12.04] | 0.880 | 0.938 | 7.75 | 0.334 | 0.029 | −29.8%, lower | −1.6%, not established |
| Selected F0 network | F0 | 12.07 [11.82, 12.35] | 0.874 | 0.935 | 8.06 | 0.306 | −0.013 | −26.0%, lower | +3.8%, higher |
| Full-data GP, F0: per-drug dose (exploratory) | F0 | 13.45 [13.20, 13.70] | 0.843 | 0.918 | 9.70 | 0.001 | −0.459 | −8.1%, not established | +28.9%, higher |
| Bliss reference | monotherapy, no fit | 14.03 [13.78, 14.29] | 0.829 | 0.914 | 9.40 | 0.315 | 0 | – | +40.3%, higher |
| Full-data GP, his kernel, w = 0 (exploratory) | F0 | 15.76 [15.48, 16.03] | 0.785 | 0.886 | 12.08 | 0.000 | −0.459 | +26.2%, higher | +77.0%, higher |
| **Frozen GP selection** | F0 | **16.10 [15.82, 16.39]** | 0.775 | 0.880 | 12.30 | −0.147 | −0.675 | +31.6%, higher | +84.7%, higher |
| Full-data GP, his kernel (exploratory) | F0 | 16.12 [15.84, 16.42] | 0.774 | 0.880 | 12.33 | −0.257 | −0.834 | +32.1%, higher | +85.3%, higher |
| DeepSynergy, dose-aware | descriptors, expression | 18.19 [17.87, 18.53] | 0.713 | 0.851 | 13.89 | 0.283 | −0.047 | +68.2%, higher | +135.9%, higher |
| **Kingsley's GP** | F0 | **19.12 [18.83, 19.42]** | 0.683 | 0.832 | 14.97 | −0.180 | −0.722 | +85.8%, higher | +160.6%, higher |
| Additive ridge | F0 | 24.18 [23.82, 24.53] | 0.493 | 0.702 | 19.26 | 0.001 | −0.459 | +197.0%, higher | +316.6%, higher |

**Frozen selection versus his GP.**
- MSE falls by 29% (259.1 vs 365.6).
- It is lower in all 39 cell lines and all 38 drugs.

**Against the other models.**
- **Bliss.** The frozen GP beats it in only 4 cells and 4 drugs.
- **F0 network.** It beats it in none.
- **By Bliss level.** Its error is close to Bliss at low expected inhibition: +5.6% MSE at R < 25 and −23.2% at 25–50. At R ≥ 75 its MSE is three times Bliss's (RMSE 19.27 vs 11.07). Without monotherapy, the GP cannot place high responses that a single-agent curve already gives Bliss.

**Pair-specific structure.** Both CPU GP runs have negative contrast R² (−0.18 and −0.15): their pair-specific predictions are worse than predicting no pair-specific contrast at all.

**Full-data GPU variants** (exploratory). Their predictions were copied from the cluster (tar SHA-256 checked) and pass the same checks.
- **F2 GP vs F2 network.** MSE is 1.6% lower. The bootstrap interval excludes 0 ([−2.8%, −0.3%]), but the drug jackknife does not ([−3.4%, +0.3%]), and one fold of five goes the other way. Not established.
- **F0 GP.** It is 8.1% below Bliss (not established: the jackknife crosses 0) and 28.9% above the F0 network.
- **Rows, not the hyperparameter fit.** His kernel fitted and conditioned on every row (16.12) equals the frozen selection, which fits on 3,000 rows and conditions on all (16.10).
- **Interaction term.** Dropping it (w = 0) lowers test MSE by 4.5% (248.3 vs 259.9; lower in 4 of 5 folds) and moves contrast R² from −0.257 to 0.000. On the validation fold the two were within 0.5%.
- **Across machines.** The frozen production rerun on the cluster matches the local run to 3.2e-10 on all 1,475,328 rows, with the same family chosen in all 195 cell-folds (`reproduce_crossmachine.json`).

**Compute for five folds.**
- His GP: 22,120 CPU-s.
- Frozen selection: 69,332 CPU-s; 5.07 h wall time with 4 workers.
- Full-data GPU fits: 2.6 s each on an A100; 969–1,953 GPU-s per five-fold run, both families included.
- Networks: 898 (F2) and 1,102 (F0) GPU-s.

## 4. Predictive uncertainty

These are the GP's predictive SDs: latent variance plus the fitted noise, as in his `return_std`. Test fold, all rows (`results/gp_compare/uncertainty/`).

| | His GP | Frozen selection | F0 GP, all rows (exploratory) | F2 GP, all rows (exploratory) |
|---|---:|---:|---:|---:|
| Coverage at nominal 50 / 80 / 90 / 95% | 0.56 / 0.86 / 0.94 / 0.98 | 0.61 / 0.90 / 0.96 / 0.98 | 0.58 / 0.85 / 0.92 / 0.95 | 0.63 / 0.86 / 0.92 / 0.95 |
| z SD (1 if calibrated) | 0.88 | 0.82 | 0.98 | 1.00 |
| Gaussian NLL | 4.358 | 4.203 | 3.975 | 3.814 |
| Pairs with 90% coverage below 0.8 | 1.5% | 1.2% | 3.1% | 2.2% |
| Spearman, abs. error vs SD: rows | 0.17 | 0.17 | 0.16 | 0.18 |
| Same, within pair-cell | −0.05 | −0.07 | −0.05 | −0.08 |
| Same, across pair-cells | 0.42 | 0.42 | 0.43 | 0.47 |
| Risk-coverage gain (share of the random-to-oracle gain) | 0.33 | 0.32 | 0.36 | 0.42 |
| Latent SD only: 90% coverage, NLL | – | – | 0.84, 4.084 | 0.79, 4.176 |
| Predictive SD scaled on validation: 90% coverage | – | – | 0.915 | 0.920 |
| **Baseline:** constant SD per cell = that model's validation RMSE | | | | |
| Coverage at 50 / 80 / 90 / 95% | 0.50 / 0.80 / 0.90 / 0.95 | 0.52 / 0.82 / 0.91 / 0.95 | 0.57 / 0.84 / 0.91 / 0.95 | 0.63 / 0.86 / 0.92 / 0.95 |
| NLL | 4.348 | 4.167 | 3.976 | 3.817 |
| Risk-coverage gain | 0.30 | 0.33 | 0.35 | 0.41 |

**Readings.**
- **The variances are valid but conservative.** Intervals over-cover at every level. Test RMSE rises steadily with predicted SD (deciles 12.0 → 22.1 for the frozen selection), but the SD overstates error by about 3 points throughout.
- **They carry pair-level, not dose-level, information.**
  - SDs order errors across pair-cells (0.42) but not within a pair-cell's dose matrix (−0.07).
  - SD is nearly flat across dose regions and Bliss levels (19.0–19.7), while error is not: 13.8 at 25–50 against 19.3 at R ≥ 75. Coverage drops to 0.93 there.
- **Cell lines.** For the frozen selection, 90% coverage ranges from 0.926 (LNCAP, the hardest cell, RMSE 25.2) to 0.973, with no cell below 0.9. Across cells, mean SD tracks RMSE (correlation 0.97).
- **They add little beyond a validation-calibrated per-cell scale.** That baseline is better calibrated and has lower NLL for both runs. It ranks errors about as well: a risk-coverage gain of 0.30 and 0.33, against 0.33 and 0.32 for the GP's SD.
- **Across models.** The GP's SD also ranks the F2 network's errors (rows 0.08–0.09). On the half of rows where the frozen GP is most confident, the network's RMSE is 9.65, against 11.48 overall.
- **All-row GPs are close to calibrated** (z SD 0.98 and 1.00, 90% coverage 0.92). Over-coverage at 50% and near-nominal coverage at 95% reflect heavy-tailed errors, not a wrong scale. Scaling the SD by validation error changes little.
- **Latent versus predictive.** The latent SD alone under-covers (90% intervals cover 0.84 and 0.79), so most predictive variance is the fitted noise. Without the interaction term the latent SD is nearly 0 (mean 1.25): the drug-additive function is pinned down by all rows.
- **Same conclusion with all rows.** The per-cell baseline matches the GPU variants in NLL (3.976 vs 3.975, 3.817 vs 3.814) and risk-coverage gain, and within a pair-cell the SD still does not rank errors.

**Answer.** Despite its lower accuracy, the GP gives honest intervals: conservative at his row budget, near-nominal with all rows. Its variances, however, identify unreliable predictions little better than each cell line's validation error does.

## 5. What the GP learns about pair-specific interaction

- **At his 1,488-row budget,** removing the interaction term (w = 0) lowers validation MSE by 11.3%. At full data it changes validation MSE by +0.5% (249.0 vs 247.9).
- **On test,** the his-kernel GPs have negative contrast R² (−0.18, −0.15 and −0.26 with all rows): the interaction term adds pair-specific predictions that are worse than none. Without it, test MSE is 4.5% lower at full data.
- **Where accuracy comes from.**
  - With identities and doses, it comes from the dose scale (per-drug standardization: −28% MSE) and from data coverage, not from the interaction kernel.
  - With monotherapy, the Bliss-residual GP reaches the F2 network's accuracy (−1.6% MSE, not established) and its contrast R² (0.334 vs 0.359).
- **No interaction term in the F2 residual (w = 0).** It does not change MSE (145.7 vs 146.3) and leaves 36 hyperparameter estimates at a bound.

## 6. Fair-comparison caveats

- **Information.** The his-kernel GPs and the F0 networks see the same inputs. The F2 GP and the F2 network see the same training-pair monotherapy lookup.
- **Model scope.** The GP fits one model per cell line and cannot share drug effects across cells; the networks fit one model per fold over all cells.
- **Budget.** The frozen selection was chosen under an 8 h CPU budget. Kingsley's original cap of 1,488 rows came from CPU cost, not from a property of the method.
- **Selection.** GPU variants were chosen on one validation fold (fold 1 of the fold-0 split) and applied to all five folds. The networks chose per fold.
- **Determinism.** The networks report three seeds. The GP is deterministic given the cap seed.

## Reproduction

Frozen CPU study and production, on branch `integration/2026-11-06`:

```sh
for g in cap1500 cap1500_w0 cap1500_rho0 cap1500_rho05 cap3000; do
  python -m cs622.gp_shared study --config configs/gp_study.json --group $g --output results/neural/runs/gp_study
done
python -m cs622.gp_select --config configs/gp_study.json --study results/neural/runs/gp_study --output results/gp
python -m cs622.gp_shared study --config configs/gp_study_amendment2.json --group cap3000_w0 --output results/neural/runs/gp_study
python -m cs622.gp_select --config configs/gp_study.json --study results/neural/runs/gp_study --output results/gp --amendment configs/gp_study_amendment2.json
python -m cs622.gp_shared run --method results/gp/decision.json --output results/neural/runs/gp_final
```

GPU study and production on one A100, on this branch:

```sh
C="configs/gp_gpu.json configs/gp_gpu_amendment.json"
for g in demo coverage features interaction; do python -m cs622.gp_gpu study --config $C --group $g --output results/neural/runs/gp_gpu_study; done
python -m cs622.gp_gpu check --config $C --cell A2058 --output results/neural/runs/gp_gpu_study/collapse_check.json
python -m cs622.gp_gpu select --config $C --study results/neural/runs/gp_gpu_study --output results/gp_gpu
for v in f0_drug_dose f2_residual_mono full full_w0; do python -m cs622.gp_gpu run --config $C --variant $v --output results/neural/runs/gp_gpu_$v; done
```

Evaluation on the shared rows, all six GP runs (the GPU runs' fold files copied from the cluster):

```sh
R=results/neural/runs
M="--method gp_his=$R/gp_cap1500 --method gp_frozen=$R/gp_final --method gp_full=$R/gp_gpu_full --method gp_full_w0=$R/gp_gpu_full_w0 --method gp_f0=$R/gp_gpu_f0_drug_dose --method gp_f2=$R/gp_gpu_f2_residual_mono"
python -m cs622.gp_verify --output results/gp_compare/gp_verification.json $M --decision results/gp/decision.json --config configs/gp_study.json --study results/neural/runs/gp_study
python -m cs622.gp_verify --output results/gp_compare/reproduce_frozen.json --method gp_frozen=results/neural/runs/gp_final --reproduce A2058:0
python -m cs622.sprint_report analyze --output results/gp_compare $M      # needs verification.json from sprint_report verify
python -m cs622.unified --output results/gp_compare $M
python -m cs622.gp_uncertainty --output results/gp_compare/uncertainty $M
python -c "import json; from cs622.synthesis import variance_shares; print(json.dumps(variance_shares()))"
```
