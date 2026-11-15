# Interaction Signal and Loewe Reference Computation in Drug-Synergy Prediction on the O'Neil Screen

Vanshdeep Sehrawat, Kingsley Ukandu, Sharlene Kim. University of Nevada, Las Vegas.

A synergy score is a measured combination response minus a non-interaction reference computed from single-agent data. Benchmark accuracy on held-out drug pairs can therefore reflect single-agent response, pair-specific interaction or the way the reference is computed. This repository separates the three on the O'Neil screen as distributed by DrugComb v1.4: 583 pairs of 38 drugs in 39 cancer cell lines, with replicated 4 × 4 dose matrices.

- Paper: [`manuscript/manuscript.pdf`](manuscript/manuscript.pdf), supplement [`manuscript/supplement.pdf`](manuscript/supplement.pdf)

## Questions

1. How much of held-out-pair accuracy comes from single-agent response, and how much from pair-specific interaction?
2. How much of the pair-specific interaction contrast is reproducible across replicates, and what does seeing the same pair in other cell lines supply?
3. How do the existence of the Loewe root, the solver's acceptance rule and its fallback affect Loewe labels and screening rankings?

## Principal results

All results are for this screen, the models and inputs evaluated, and pairs held out of training. They are exploratory: the screen had been examined before the analyses were designed.

- **Response accuracy.** Neural, Gaussian-process and collaborative-filtering models reach response $R^2$ up to 0.88 on held-out pairs; the Bliss reference with no fitted parameters reaches 0.83. Additive single-agent and drug-by-dose structure holds 81.8% of row variance.
- **Interaction accuracy.** On the pair-specific interaction contrast the same models explain at most 0.07 of the replicate-reproducible signal.
- **Decomposition.** The contrast energy splits into a pair-wide offset (24%), pair-by-cell variation (66%) and within-batch replicate noise (10%). With the pair seen in other cell lines, an empirical-Bayes offset reaches $R^2$ 0.177, a linear cross-cell predictor 0.216 and a neural network 0.218, against −0.050 for the network with the pair held out. About 54% of the contrast energy is reproducible but predicted by none of these procedures.
- **Loewe reference.** Loewe labels remain learnable on a simulated screen with no interaction (contrast $R^2$ 0.49–0.52); Bliss labels do not. An exact condition states when the Loewe root exists; 15,620 roots that the SynergyFinder solver rejected are certified by interval arithmetic; the summed-dose fallback makes the reference discontinuous at the existence boundary. Restoring 14,197 float64-verified roots replaces 67 of 1,137 pair-cells in a top-5% shortlist.
- **Negative results.** The pre-declared confirmation of high-inhibition interaction specificity on an untouched NCATS screen failed at its first step ([report](reports/third_screen_validation.md)). A drug-identity response penalty of at least 5% was not established (+4.61% pooled over four seeds). Neither is reopened here.

No biological or clinical claim is made.

## Authors and contributions

The manuscript and [AUTHORS.md](AUTHORS.md) state the same division. All three authors are included in [CITATION.cff](CITATION.cff).

| Author | Contribution | Implementation |
|---|---|---|
| Vanshdeep Sehrawat | Neural models and DeepSynergy reproduction; shared data pipeline, splits, metrics and runner; GP and CF adapters and extensions; interaction, replicate-noise, same-pair, Loewe and screening analyses; manuscript and scientific figures | `cs622/`, `experiments/`, `manuscript/` |
| Kingsley Ukandu | Gaussian-process model: pair kernel, drug-dose encoder, block cap, scikit-learn fit, RBF versus Matérn 5/2 choice | `GP_Regression/src/project_utils.py`, used unchanged |
| Sharlene Kim | Neighborhood collaborative filtering on raw response: target construction, tuning grid, five-fold evaluation | `experiments/neighborhood_cf/`, `results/neighborhood_cf/` |

[`reports/gp.md`](reports/gp.md) and [`reports/cf.md`](reports/cf.md) separate each collaborator's model from the adapters and extensions run on it.

## Data

Nothing in `data/raw/` or `data/processed/` is tracked. `python -m cs622.download` fetches the sources and checks their hashes ([`docs/datasets.md`](docs/datasets.md)).

| Source | Use | Terms |
|---|---|---|
| DrugComb v1.4 dose-level responses and block scores ([Zenodo 18449193](https://zenodo.org/records/18449193), [11102665](https://zenodo.org/records/11102665)) | O'Neil, NCATS and ALMANAC rows | CC BY 4.0 |
| DeepSynergy labels ([Preuer et al. 2018](https://www.bioinf.jku.at/software/DeepSynergy/)) | Published pair folds and O'Neil labels | No license stated; cited |
| NCI-ALMANAC ([NCI](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC)) | Screening-site check | NCI public data |

## Methods

- **Protocol** ([`docs/methodology.md`](docs/methodology.md)). Five published pair-grouped folds: fold *f* is test, fold *f*+1 (mod 5) validation, the rest training. Every held-out pair is unseen; each of its drugs appears in training pairs. Selection uses validation folds only. Information regimes: F0 (drug and cell identity, doses) and F2 (F0 plus single-agent responses from training pairs).
- **Models.** Dose-aware neural network and DeepSynergy (`cs622/models.py`, `cs622/deepsynergy.py`); Gaussian process (`GP_Regression/`, run through `cs622/gp_shared.py` and `cs622/gp_gpu.py`); neighborhood CF (`experiments/neighborhood_cf/`, `cs622/cf.py`); mean, additive and Bliss-reference baselines (`configs/mean.json`, `additive.json`, `reference_bliss.json`).
- **Interaction target.** Pair-cell mean of measured response minus the Bliss reference, projected off additive drug effects within each cell line and fold (`cs622/metrics.py`).
- **Analyses.** Replicate-noise ceilings by balanced block sign splits; same-pair predictors B0–B5 and a variance decomposition; a zero-interaction simulation; Loewe root existence, interval certificates (mpmath) and the fallback; screening-shortlist sensitivity.

## Layout

```
cs622/              package: data, splits, models, runner, metrics, analyses
GP_Regression/src/  Kingsley Ukandu's Gaussian process (imported by cs622.gp_shared)
configs/            frozen protocols and model configurations, one JSON per study or amendment
experiments/        study scripts (neighborhood_cf/ is Sharlene Kim's; preliminary/ is the Gate 2 pipeline)
results/            tracked summaries, tables and figures; predictions and checkpoints stay local
reports/            one report per study
manuscript/         paper and supplement (LaTeX), figures, number checker, independent audits
docs/               Gate 2 protocol and findings, methodology, neural results, published-method comparison
tests/              unit and synthetic tests
```

Study folders keep the identifier under which their protocol was frozen; the protocols, hash manifests and claim ledger cite those paths.

| Study | Protocol | Code | Results | Report |
|---|---|---|---|---|
| Gate 2: scoring-induced learnability | `docs/gate2_protocol.md` | `experiments/preliminary/` | `results/preliminary/` | [`docs/preliminary_findings.md`](docs/preliminary_findings.md) |
| Neural benchmark and DeepSynergy | `configs/neural_*.json`, `deepsynergy_*.json` | `cs622/` | `results/neural_sprint/` | [`docs/neural_results.md`](docs/neural_results.md) |
| Gaussian process | `configs/gp_study*.json`, `gp_gpu*.json` | `GP_Regression/`, `cs622/gp_*.py` | `results/gp/`, `gp_gpu/`, `gp_compare/` | [`reports/gp.md`](reports/gp.md) |
| Collaborative filtering | `configs/cf_study.json` | `experiments/neighborhood_cf/`, `cs622/cf.py` | `results/neighborhood_cf/`, `cf/`, `compare/` | [`reports/cf.md`](reports/cf.md) |
| Loewe fallbacks, beyond-Bliss regimes, ALMANAC | `configs/almanac_*.json`, `beyond_bliss_*.json` | `cs622/loewe_audit.py`, `regimes.py`, `almanac_*.py` | `results/loewe_audit/`, `beyond_bliss_regimes/`, `almanac_validation/` | [`reports/loewe_fallbacks_and_almanac.md`](reports/loewe_fallbacks_and_almanac.md), [`reports/loewe_note.md`](reports/loewe_note.md) |
| Loewe verification, stress tests, third screen | `configs/loewe_verification.json`, `stress_tests.json`, `third_screen_*.json` | `cs622/loewe_verify.py`, `stress.py`, `third_screen.py` | `results/loewe_verify/`, `stress_tests/`, `third_screen/` | [`reports/third_screen_validation.md`](reports/third_screen_validation.md) |
| Resolution and reliability | `configs/research_20261107.json` and amendments | `cs622/resolution.py`, `reliability.py`, `loewe_interval.py` | `results/research_20261107/` | [`reports/resolution_and_reliability.md`](reports/resolution_and_reliability.md) |
| Decision sensitivity and seed 2029 | `configs/research_20261107.json` | `experiments/overnight_20261107/` | `results/overnight_20261107/`, `seed2029_20261108/` | `results/overnight_20261107/research_findings.md` |
| Mechanisms and prioritization | `configs/mechanisms_20261108.json`, `gpu_mechanisms_20261108.json` | `experiments/mechanisms_20261108/`, `cs622/mechanism_*.py` | `results/mechanisms_20261108/` | [`reports/mechanisms.md`](reports/mechanisms.md), [`reports/loewe_prioritization.md`](reports/loewe_prioritization.md) |
| Replicate noise and pair information | `configs/final_20261110.json` | `experiments/final_20261110/`, `cs622/final_analysis.py`, `pair_seen.py` | `results/final_20261110/` | [`reports/final_20261110/`](reports/final_20261110/EXPERIMENTS.md) |
| Interaction gap | `experiments/interaction_gap_20261110/design.json` | `experiments/interaction_gap_20261110/` | `results/interaction_gap_20261110/` | [`reports/interaction_gap_20261110/`](reports/interaction_gap_20261110/REPORT.md) |

## Install

Python 3.11 or newer. GPU training needs CUDA; everything else runs on CPU.

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt -e ".[test]"
```

Building the paper also needs XeLaTeX (MiKTeX or TeX Live).

## Reproduce

From the repository root:

```sh
python -m cs622.download deepsynergy drugcomb_summary drugcomb_raw        # about 2.2 GB
python -m cs622.data --config configs/data_oneil.json                     # observation table, SHA-256 ea588405…ce8b
python -m cs622.cv run --config configs/neural_dose_aware.json --output results/neural/runs/f0_cv
python -m cs622.cv run --config configs/additive.json --output results/neural/runs/additive
```

Each study's report lists its full command sequence. The analyses behind the paper read saved out-of-fold predictions through the resolution cache (`python -m cs622.resolution --output results/research_20261107/resolution`); [`manuscript/REPRODUCIBILITY.md`](manuscript/REPRODUCIBILITY.md) gives the remaining steps, and Supplement S14 the hash of every input. Then:

```sh
python manuscript/scripts/check_numbers.py --resolution results/research_20261107/resolution   # 132 checks (116 original + 16 post-review checks)
python manuscript/scripts/build.py                                                            # manuscript and supplement PDFs
```

## Tests

```sh
python -m pytest -q
```

Expected: 140 passed, 1 skipped. The skipped test compares `cs622.cf` with Sharlene Kim's original CF code on her target matrix, `results/neighborhood_cf/raw_response_targets.npz`, which is not in the repository. Tests use synthetic fixtures except where they say otherwise.

## Availability and limits

- Raw data, processed tables, out-of-fold predictions and checkpoints are not tracked. Neural predictions came from GPU runs; their archives are retained by the authors and identified by SHA-256 in the reports and Supplement S14.
- Reproducing the paper's analyses without retraining needs those predictions; reproducing them from scratch needs the GPU runs in the study protocols.
- Original software is [MIT licensed](LICENSE). Upstream data retain their own terms; see [NOTICE.md](NOTICE.md).
- This repository contains the research code, protocols, scientific summaries and manuscript. Input datasets and model checkpoints are downloaded or regenerated separately.
- [`docs/published_models.md`](docs/published_models.md) records the published-model comparison and the decision not to add an incompatible experiment.

Source identity, file hashes and transformations are recorded in [SOURCE.json](SOURCE.json) and [MANIFEST.json](MANIFEST.json).
