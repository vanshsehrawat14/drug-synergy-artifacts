# Final campaign experiments

10 November 2026, branch `research/final-2026-11-10` from `124d95a`.
- **Protocol.** `configs/pair_information.json`, frozen in commit `4a6fbd8` at 08:36 PST, before any quantity below was computed. Preceding observations are listed inside it.
- **Status.** Every result is exploratory. O'Neil has been inspected extensively. No historical seed, threshold, configuration or result was changed, and no ablation seed was added.

Two directions were run:
- **A:** replicate-noise calibration (CPU only).
- **B:** pair-information ceiling (a CPU oracle and one gated GPU experiment).

## Ground truth at the start

| Item | State |
|---|---|
| Branches | `research/resolution-reliability` `d2c8f41`; `research/response_analysis-2026-11-07` `7090494`; `research/mechanisms-2026-11-08` `124d95a`. All worktrees clean. |
| Tests | 118 passed, 1 skipped on `124d95a`. The 107 + 1 count refers to `7090494`. The skip is the CF real-data test, which needs Shar's target matrix (ignored, absent from fresh worktrees). Earlier reports called it CUDA-dependent; it is not. On CPU, CUDA tests are not parametrized at all. |
| Data | `oneil_v1` observations SHA-256 `ea588405…3ce8b`. 1,475,328 combination rows, 92,208 blocks, 22,737 pair-cells. Pair-cells have 4 replicate blocks (22,422) or 8 (315). |
| Cached predictions | Resolution cache `interaction_analysis/resolution` (inputs SHA-256 recorded in each summary); historical `benchmark_v1` runs |
| Compute | Shared A100-SXM4-80GB GPUs; one allocated GPU per task. |
| Archives retained on RebelX | `research_return_20261107_final.tar.gz` (`2b75eeba…`), `…20261108_seed2029.tar.gz` (`454faac2…`), `…20261108_mechanisms.tar.gz` (`f05e4796…`) |

## A1. Target-specific replicate noise ceilings

**Design.** Within each pair-cell, blocks are split into equal halves, 50 random partitions (seed 20261110). The proxy $d=(\bar y_{h1}-\bar y_{h2})/2$ per identical-input unit has the covariance of the replicate-mean noise (Lemma 4). The cached evaluation operators are applied unchanged. Intervals: whole-pair bootstrap (2,000 draws) and delete-one-drug jackknife with every projection refitted.

```sh
python -m synergy_artifacts.pair_analysis noise --resolution <interaction_analysis/resolution> --output results/pair_information/noise
```

**Results** (`results/pair_information/noise/`). Row level: replicate variance 83.34 points² on 1,109,976 df; row ceiling $R^2$ 0.928.

| Target | Noise share [pair; drug] | Ceiling | Selected F2 $R^2$ → noise-corrected | GP F2 | CF ext. | Production F2 |
|---|---|---:|---|---|---|---|
| Pair-cell contrast | 0.069 [0.064, 0.075; 0.048, 0.091] | 0.93 | 0.356 → 0.382 | 0.333 → 0.357 | 0.335 → 0.360 | 0.279 → 0.300 |
| **Bliss-residual pair-cell contrast** | **0.101 [0.092, 0.111; 0.066, 0.136]** | **0.90** | 0.061 → 0.068 | 0.028 → 0.031 | 0.032 → 0.035 | −0.050 → −0.056 |
| Dose-shape contrast | 0.169 [0.147, 0.195; 0.101, 0.236] | 0.83 | 0.672 → 0.809 | 0.683 → 0.821 | 0.508 → 0.611 | 0.643 → 0.773 |
| Bliss-residual dose-shape contrast | 0.450 [0.399, 0.518; 0.333, 0.568] | 0.55 | 0.124 → 0.226 | 0.153 → 0.278 | −0.315 → −0.573 | 0.046 → 0.083 |

**Decision (frozen rule).** The upper bound of the primary noise share is below 0.2. Replicate noise therefore cannot explain interaction-residual $R^2$ near 0.06: the unexplained pair-cell interaction contrast is reproducible across replicate blocks. At dose-shape level about half the residual target is replicate noise, and GP/neural F2 capture about a quarter of the reliable remainder.

**Limitations.** Replicate blocks may share plates or processing, so these ceilings are upper bounds. Monotherapy noise is absent because monotherapy is shared. Cross-study synergy reproducibility is much lower than within-study reproducibility.

## A2. Loewe shortlist under combination-replicate noise

**Design.** Block scores under N, C1 and I come from the historical fixed-fit references (`decision_sensitivity.references`, `unit_scores`); native block scores are reproduced exactly. The unit is the pair-cell mean of its blocks. Procedures:
- 200 equal block half-splits;
- 1,000 within-pair-cell block bootstraps (seed 20261110), with N and C1 on the same resample.

```sh
python -m synergy_artifacts.shortlist_noise --verified <2026-11-06_validation/loewe_verify/cases_verified.parquet> \
  --certificates <response_analysis handoff root_certificates/certificates.parquet> --output results/pair_information/shortlist_noise
```

**Results** (`results/pair_information/shortlist_noise/`, top 5%, 1,137 units):

| Quantity | Mean [2.5%, 97.5%] |
|---|---|
| Fixed-data N → C1 turnover | 5.89% |
| N → C1 within each bootstrap resample | 5.88% [5.19%, 6.68%] |
| N → C1 within each half-split | 5.78% [5.01%, 6.60%] |
| Noise: independent half-depth replicate screens (N) | 28.5% [27.4%, 29.6%] |
| Noise: independent full-depth bootstrap resamples (N) | 16.7% [15.4%, 17.9%] |

| Units | Selection probability under N | Under C1 |
|---|---:|---:|
| 67 C1 entrants | 0.068 (74.6% below 0.1) | 0.776 (88.1% ≥ 0.5) |
| 67 C1 exits | 0.441 | 0.322 |
| 1,070 kept under both | 0.908 | 0.885 |

**Decision (frozen rule).** Systematic. The within-resample turnover is above zero in every resample, entrants' mean probability under N is below 0.1, and under C1 it is at least 0.5.

**Interpretation.**
- At this screen's replicate depth, a single replication changes about three times more top-5% choices than C1 does.
- The C1 change is a bias, not variance: it does not average out with more replicates.
- It replaces near-coin-flip units with units that native scoring excludes in almost every replicate resample.

**Limitations.** Combination replicates only; monotherapy, fits and references are fixed. Full-depth independent-replication turnover lies between the bootstrap (≈1.5× replicate-mean variance) and half-depth (4×) figures.

## B1. Cross-cell pair oracle and the B2 gate

**Design.** Target: $t=(P-A)(y-r_B)$ per pair-cell (primary), plus the raw pair-cell contrast.
- **LOCO oracle.** The mean of the same pair's $t$ over all other cells.
- **Matched oracle.** The mean over cells in the B2 training folds.

Pair-cell folds: seed 20261110, assignment SHA-256 `d32914a9…370ea`. Intervals as in A1, with the oracle rebuilt after each drug deletion.

```sh
python -m synergy_artifacts.pair_analysis oracle --resolution <interaction_analysis/resolution> --output results/pair_information/oracle
```

**Results** (`results/pair_information/oracle/`):

| Predictor (Bliss-residual pair-cell contrast) | $R^2$ [pair; drug] |
|---|---|
| LOCO oracle | 0.215 [0.173, 0.257; 0.104, 0.327] |
| Matched oracle | 0.205 [0.164, 0.248; 0.094, 0.317] |
| Selected F2, held-out pairs | 0.061 [0.022, 0.099; −0.039, 0.161] |
| Production F2, held-out pairs | −0.050 [−0.089, −0.014; −0.153, 0.053] |
| **Gate: matched oracle − production F2** | **0.256 [0.202, 0.308; 0.082, 0.429] → passed** |

- **Cross-cell covariance share.** 0.234 (Bliss residual), 0.244 (raw).
- **Raw contrast.** The oracle reaches 0.22, below Bliss (0.31) and selected F2 (0.36). Much of the raw contrast is monotherapy-derived; the pair-specific excess is not.
- **Post-hoc secondary** (computed in the B2 analysis). Removing the projected Bliss/HSA span leaves the matched oracle at 0.209. The pair-consistent signal is not Bliss/HSA-span structure.

## B2. Pair-seen F2 (GPU)

**Design.**
- **Config.** Production `f2_base` configuration, identical except `split = {"kind": "pair_cell", "seed": 20261110}`.
- **Split.** Every pair-cell, with all its blocks, is in one fold; test f, validation (f+1)%5, three folds train.
- **Monotherapy.** Training blocks only.
- **Tasks.** Seeds 2026–2028 × 5 folds = 15.
- **Code.** `synergy_artifacts.pair_seen`; the runner split option is backward compatible, and the default path is unchanged.
- **Preflight.** Every task config equals its `f2_base` control except the split. No pair-cell or block leakage. Every test drug-cell has training monotherapy. 19 tests pass on the cluster. A CPU real-data smoke run passes.
- **Evaluation.** Identical rows and the original evaluation operators. An equivalence test (pair-seen := `f2_base`) returns exactly 0 for every difference.

```sh
# From this checkout, after regenerating held-out-pair controls; TASK is 0..14.
python experiments/pair_information/check.py --controls results/neural/runs/benchmark_v1/runs/f2_base
python -m synergy_artifacts.pair_seen --task TASK --output results/pair_information/runs --controls results/neural/runs/benchmark_v1/runs/f2_base
# Evaluation; use the regenerated resolution cache and pair-seen run directory.
# Inputs: an exact column subset of the local resolution cache (truth, f2_base seeds, Bliss),
# shown locally to reproduce the full-cache output with zero difference.
python -m synergy_artifacts.pair_analysis pairseen --resolution <resolution_b2> --runs results/pair_information/runs \
  --oracle results/pair_information/oracle --noise results/pair_information/noise --output results/pair_information/pair_seen
```

**Execution.**
- **Array.** `133011` (`0-14%1`) on an A100-SXM4-80GB shared with other users, 08:50–10:09 PST.
- **Outcome.** 15/15 tasks exited 0, with no retries. Each task recorded commit `8bce416`, a clean tree and split hash `d32914a9…`.
- **Per-task records.** Job IDs, start/end times, training seconds, best and completed epochs, and validation/test MSE are in `results/pair_information/training_records/task_ledger.csv`.
- **Runtime.** Total training 4,228 s (1.17 GPU-hours), with tasks taking 167–570 s, against a 7.5 GPU-hour ceiling.
- **Archives.**
  - Full return, with checkpoints, retained on RebelX: `research_return_20261110_pairseen.tar.gz`, 28,320,304 B, SHA-256 `c19c035b02b338f89357c8e2fca2fab47aeb16f6d8868b392d12f45840a86cbb`, 135 files.
  - Summary return, downloaded and verified, all 97 files: `research_return_20261110_pairseen_summary.tar.gz`, 934,313 B, SHA-256 `2a63212ef21b60aa1ae123357bd99125c27f2a12f6751107fc8680690dcfe803`.

**Results** (`results/pair_information/pair_seen/`; seed-averaged errors, identical rows, original operators):

| Target | Held-out-pair f2_base | Pair-seen F2 | Difference [pair; drug] | Folds + |
|---|---:|---:|---|---:|
| **Bliss-residual pair-cell contrast (primary)** | −0.050 | **0.218** | **+0.268 [0.208, 0.324; 0.069, 0.468]** | 5/5 |
| Raw pair-cell contrast | 0.279 | 0.463 | +0.184 [0.141, 0.228; 0.036, 0.333] | 5/5 |
| Reference-augmented pair-cell contrast (post hoc) | −0.056 | 0.215 | +0.271 [0.211, 0.325; 0.071, 0.471] | 5/5 |
| Raw dose-shape contrast | 0.643 | 0.694 | +0.051 [0.040, 0.065; 0.020, 0.082] | 5/5 |
| Bliss-residual dose-shape contrast | 0.046 | 0.181 | +0.136 [0.109, 0.167; 0.068, 0.204] | 5/5 |

- **Response MSE.** 151.62 → 129.98, −14.3% [−17.0, −11.7; drug −20.3, −8.3].
- **Per seed** (Bliss-residual pair-cell). Pair-seen 0.191, 0.216, 0.248; held-out −0.041, −0.090, −0.020.
- **Oracle comparison.** Matched oracle 0.205; fraction of the oracle gain captured 1.05.
- **Noise correction.** Noise-corrected primary difference 0.298.

**Decision (frozen rule).** **Exploitable.** The difference is at least 0.05, both grouped intervals are above zero, and all 5 folds are positive. The frozen architecture represents the pair-consistent interaction when the pair is observed; the held-out-pair deficit is informational, not representational.

The remaining reproducible contrast (ceiling 0.90 against 0.22) is specific to the pair-cell. Neither cross-cell averaging nor pair-seen training predicts it.

**Limitations.**
- The comparison is between procedures that receive different information, on identical rows. It is not a deployable unseen-pair benchmark.
- Pair-cell (leave-triplet-out) splits are known to be optimistic.
- The architecture, optimizer and validation rule are the production ones. Validation-fold early stopping uses pair-cell folds.

## Loewe boundary checks (mathematical findings)

```sh
python experiments/pair_information/loewe_boundary.py --cases <scientific_native_full/surface_cases.parquet> \
  --regimes <response_analysis handoff loewe_regimes.csv> --output results/pair_information/loewe_boundary
```

- **Existence criterion.** 309,056/309,056 agreements with the 100-digit certificates.
- **Boundary-layer law.** Log–log $r=0.997$ on 12,964 near-plateau roots. 93.8% of C1-recovered roots have $|b_i|>2$, versus 43.3% of all roots.
- **Summed-dose jump.** For 10,712 upper-plateau C1 roots: median 3.67 points, mean 18.8; these roots carry 68.3% of the absolute C1 shift.

## Numerical implementation corrections

- The initial jump check assumed every C1 root lies at the upper plateau. It was corrected to restrict that check to upper-plateau roots and to report all three root locations separately.
- A linear existence-test grid missed boundary-layer roots (Proposition 3). A geometric grid was added; only cases below float64 resolution are skipped.

## Validation

| Run | Code | Result |
|---|---|---|
| Local CPU, Python 3.14.2 | `af69d62` | **130 passed**, 0 skipped, in 91.7 s. Shar's target matrix was linked (ignored file, SHA-256 `77cdbb57…f272bb`); without it, 129 passed and 1 skipped. |
| GPU execution `133134` (A100, torch 2.12.1+cu129, CUDA available) | `af69d62` | **132 passed, 1 skipped** in 379 s, exit 0. The CPU tests plus the three CUDA-parametrized variants ran. The skip is the CF real-data test (target matrix absent on RebelX). |
| Cluster preflight before B2 | `8bce416` | 19 passed (`test_pair_seen`, `test_runner`, `test_data_splits`); `check.py` passed for all 15 tasks |

- **New tests.** `tests/test_pair_seen.py` (4) and `tests/test_pair_analysis.py` (7) cover:
  - split determinism and balance, and leakage;
  - training-block monotherapy;
  - the exact `f2_base` match;
  - the panel-projection identity;
  - the noise-proxy variance (Lemma 4) and oracle construction;
  - Propositions 1 and 3 against bisection;
  - block resampling.
- **Equivalence checks.** A fake pair-seen run built from `f2_base` gives exactly zero differences. The reduced cache reproduces the full-cache B2 output exactly.


## Reproduction

- **Shared outputs.** Every analysis reads tracked configs and writes a fresh directory; existing outputs are refused.
- **Local inputs.** The resolution cache, `scientific_native_full` tables and certificate files stay untracked; their SHA-256 values are recorded in each `summary.json`.
- **Figures.** `python experiments/pair_information/figures.py --results results/pair_information`.
