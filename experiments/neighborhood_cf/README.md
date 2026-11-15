# Neighborhood Collaborative Filtering for Raw Drug-Combination Response Prediction

## Overview

This experiment evaluates a Neighborhood Collaborative Filtering (CF) model for predicting raw drug-combination inhibition responses using the O'Neil Drug Combination dataset.

The model predicts continuous combination inhibition values rather than derived synergy scores. Its performance is compared against an additive ridge-regression baseline.

## Dataset and target construction

- **Dataset:** O'Neil drug-combination screening data
- **Drug pairs:** 583
- **Cell lines:** 39
- **Target matrix:** 583 × 39
- **Complete pair-cell observations:** 22,737

For each replicate block:

1. Select the 16 dose observations where both drug concentrations are positive.
2. Calculate the mean raw inhibition across those 16 observations.
3. Average replicate-block means for each drug-pair/cell-line combination.

Each pair-cell target has 4–8 replicate blocks.

The resulting target matrix is aligned with the shared pair ordering, cell-line ordering, and five-fold assignments from `experiments/preliminary/common.py`.

## Model and evaluation

Neighborhood CF is evaluated using five-fold pair-grouped out-of-fold evaluation.

For each test fold, the next fold modulo five is used for validation:

`validation_fold = (test_fold + 1) mod 5`

The hyperparameter grids are:

- Neighborhood size (k): 3, 5, 10, 20
- Similarity shrinkage: 2, 5, 20
- Ridge regularization: 0.01, 0.1, 1, 10, 100

The additive ridge baseline is evaluated using the same fold structure.

Metrics include:
- Mean squared error (MSE)
- Pearson correlation
- Contrast sum of squared errors
- Contrast R²

## Overall out-of-fold results

| Metric | Neighborhood CF | Additive baseline |
|---|---:|---:|
| MSE | 35.9041 | 49.4726 |
| Pearson correlation | 0.9596 | 0.9439 |
| Contrast R² | 0.1435 | -0.1939 |

Neighborhood CF achieves approximately 27.4% lower MSE than the additive baseline.

The positive contrast R² indicates that CF captures some pair-specific response variation beyond the additive baseline. However, contrast R² remains modest and should not be interpreted as evidence of strong pair-specific predictive performance.

## Files

- `build_raw_response_targets.py`: Reconstructs the raw response target matrix from processed O'Neil dose-level data.
- `run_neighborhood_cf.py`: Runs the Neighborhood CF evaluation and additive baseline.
- `results/neighborhood_cf/target_validation.json`: Target construction and validation summary.
- `results/neighborhood_cf/cf_metrics.json`: Overall and fold-level model evaluation metrics.

Generated `.npz` files are excluded from Git tracking by the repository's `.gitignore` and must be regenerated locally.

## Reproduction

From the repository root:

1. Ensure the required O'Neil processed data and DeepSynergy labels file are available in the expected `data/` directories.
2. Build the target matrix:

   `python experiments/neighborhood_cf/build_raw_response_targets.py`

3. Run the model:

   `python experiments/neighborhood_cf/run_neighborhood_cf.py`

The scripts require the Python dependencies used by the shared project environment, including NumPy and pandas.

## Reproducibility and limitations

The rebuilt target matrix was compared against the original locally generated matrix.

- Target values were numerically equivalent (`np.allclose`, tolerance 1e-12).
- Pair ordering, cell-line ordering, replicate counts, and fold assignments matched exactly.
- All 22,737 pair-cell targets were complete.

The raw data files and generated NumPy archives are not included in Git. Teammates must obtain the required source data separately.

These results are from the current five-fold evaluation and should be interpreted within the team's shared evaluation protocol. They do not establish generalization to unseen drugs, datasets, or experimental settings.