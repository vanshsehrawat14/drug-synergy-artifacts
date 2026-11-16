# Interaction Signal and Loewe Reference Computation in Drug-Synergy Prediction on the O'Neil Screen

**[Paper (PDF)](manuscript/manuscript.pdf) · [Supplement (PDF)](manuscript/supplement.pdf) · [Code](synergy_artifacts/) · [Reproduction](manuscript/REPRODUCIBILITY.md)**

Vanshdeep Sehrawat, Kingsley Ukandu and Sharlene Kim — University of Nevada, Las Vegas.

We separate measured-response prediction, pair-specific interaction structure and numerical reference computation on the O'Neil screen: 583 drug pairs, 38 drugs and 39 cancer cell lines, with replicated dose matrices.

## Findings

- Measured-response $R^2$ reaches 0.88 across the evaluated neural, Gaussian-process and collaborative-filtering models; parameter-free Bliss reaches 0.83. Additive single-agent and drug-by-dose structure accounts for 81.8% of row variance.
- The evaluated models explain at most 0.07 of the replicate-reproducible pair-specific interaction contrast. Seeing a pair in other cell lines supplies information unavailable to held-out-pair predictors.
- Loewe labels remain learnable in a zero-interaction simulation. Exact root analysis and interval certificates distinguish reference structure from solver failures.
- Restoring 14,197 float64-verified roots changes 67 of 1,137 pair-cells in a top-5% shortlist. Numerical improvement does not establish better biological selection.
- The NCATS confirmation failed. Results remain limited to the evaluated screens, methods and inputs; analyses on previously inspected data are exploratory.

## Repository

| Path | Content |
|---|---|
| `synergy_artifacts/` | Data preparation, models, evaluation and analysis |
| `gaussian_process/src/` | Kingsley Ukandu's original Gaussian-process implementation |
| `configs/` | Model settings, study protocols and amendments |
| `experiments/` | Reference, response, interaction and pair-information analyses |
| `results/` | Summary tables, numerical certificates and validation records |
| `reports/` | Study methods, results and limitations |
| `manuscript/` | Paper, supplement, figures, audits and build/check scripts |
| `tests/` | Synthetic and implementation checks |

[AUTHORS.md](AUTHORS.md) records contributions. The [GP](reports/gp.md) and [CF](reports/cf.md) reports distinguish the original frameworks from their extensions. [Published-method comparability](docs/published_models.md) records targets, information regimes, splits and reproducibility limits.

## Install and reproduce

Python 3.11 or newer; CUDA for neural training. XeLaTeX is needed to build the PDFs.

```sh
python -m pip install -r requirements.txt -e ".[test]"
python -m synergy_artifacts.download deepsynergy drugcomb_summary drugcomb_raw
python -m synergy_artifacts.data --config configs/data_oneil.json
python -m synergy_artifacts.cv run --config configs/neural_dose_aware.json --output results/neural/runs/f0_cv
```

The [reproduction record](manuscript/REPRODUCIBILITY.md) and Supplement S14 specify input hashes, configurations, seeds and analysis commands. Full reconstruction requires regenerating the untracked out-of-fold predictions; they are not included in a fresh clone.

The committed summaries support the numerical checker directly:

```sh
python manuscript/scripts/check_numbers.py --resolution results/interaction_analysis/resolution
python manuscript/scripts/build.py
python -m pytest -q
```

The original CF real-data comparison is skipped when its untracked target matrix is absent. [Dataset provenance](docs/datasets.md) and [NOTICE.md](NOTICE.md) identify upstream access and terms. Original software is [MIT licensed](LICENSE); input datasets and model weights are downloaded or regenerated separately.

[Source provenance](docs/provenance.md) distinguishes historical execution hashes from the maintained layout.
