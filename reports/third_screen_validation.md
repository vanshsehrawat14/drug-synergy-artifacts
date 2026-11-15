# Validation sprint, 2026-11-06

**Scope.** Four parts:
1. an independent check of the Loewe fallback finding;
2. stress tests of the high-inhibition beyond-Bliss result;
3. a third, untouched screen evaluated with a frozen protocol;
4. a correction list for PR #1, on `review/pr1-gp-integration`.

**Freezes and amendments.** Each was committed before the numbers it governs were computed:
- `acc77e1`: third-screen protocol and stress-test controls;
- `fa7e528`: the Loewe verification plan;
- `69fa7f8`: third-screen amendment 1, made before any test output;
- `0659fab`: third-screen and stress analysis code, committed before any third-screen output was read.

## 1. Loewe fallback: independent verification

Plan: `configs/loewe_verification.json`. Code: `cs622/loewe_verify.py`. Results: `results/loewe_verify/`.

**Admissibility under the scorer's own definition.** A root counts as admissible only if two conditions hold:
- the Loewe equation of `calculate_synergy_score.R` has residual < 1e-6 at the root;
- the root lies strictly inside both fitted effect ranges, where that equation is real-valued.

**Independent implementation.** The second root finder never inverts a curve:
- with θ = x_a / D_a(y), it solves f_a(x_a/θ) = f_b(x_b/(1−θ)) by bisection on logit θ;
- it also covers co-decreasing curve pairs, which the earlier strict rule excluded.

**Agreement.**
- On the 193,186 cases where both solvers find a root, they agree to 3.8e-13.
- The strict rule finds no root that the new solver misses.
- The new solver finds 2,022 more on co-increasing curves; 665 of these are admissible. The rest sit within float precision of an asymptote.

**All 367,152 cases, classified** (`classes.csv`):

| Class | Cases | Surfaces | Native outcome | Root − native reference (mean / median) |
|---|---:|---:|---|---|
| Accepted, matches the root | 180,364 | 17,667 | code 1–2 | 0 |
| Accepted, no admissible root | 1,883 | 417 | code 1–2 | n/a |
| **Artifact: solver at the root, answer discarded** | **12,477** | 4,848 | all code 3 | −19.8 / −3.7 |
| **Artifact: root exists, solver failed** | **1,720** | 448 | 253 code 3, 1,467 error | −28.5 / −6.6 |
| Boundary: root at an asymptote, residual ≥ 1e-6 | 1,423 | 704 | 1,389 code 3 | −21.5 / −4.0 |
| Definition: no root | 121,984 | 14,637 | fallback | n/a |
| Definition: opposite or flat curves | 47,301 | 3,063 | fallback | n/a |

**Genuine numerical artifacts: 14,197 cases** (the two artifact rows).
- In these cases the scorer's own equation has an admissible root and the scorer still falls back.
- In 12,477 of them, nleqslv had already returned that root (median distance 0.009). The scorer discards it because the termination code is 3.

**Definition-dependent cases: 169,285.**
- There is no root, or the two curves run in opposite directions.
- The fallback is then the scorer's stated rule, and other conventions differ legitimately.
- The 1,423 boundary roots are mathematically valid but numerically ill-conditioned, so they are not counted as artifacts.

**Score effects** (native block scores, references capped at 100 as in the scorer):

| Correction | Cases | Blocks changed | Mean shift | ≥ 5 pts | Sign flips | Max |
|---|---:|---:|---:|---:|---:|---:|
| Solver at the root only | 12,477 | 19,308 (20.9%) | +3.21 | 4,240 | 1,903 | 38.9 |
| All admissible roots, co-increasing | 13,888 | 20,088 (21.8%) | +3.68 | 4,828 | 1,986 | 51.1 |
| All admissible roots | 14,197 | 20,276 (22.0%) | +3.66 | 4,836 | 1,998 | 51.1 |
| *Convention:* HSA fallback instead of summed-dose fallback | all fallbacks | 76,204 (82.6%) | +9.38 | 31,708 | 20,974 | 93.8 |

- **Pair-cell panel.** Using all admissible roots changes 5,003 of 22,737 units, 1,196 of them by ≥ 5. The Spearman correlation with the published panel is 0.976.
- **In context.** The artifact is real but smaller than an ordinary choice between fallback conventions, which moves four times as many blocks.

**Downstream model effects** (identity panel network, `experiments/phase2/panel_neural.py`, seed 2026). Retraining reproduces the existing predictions exactly (max difference 0).

| Train on → score on | R², all units | R², 5,003 affected units | Pearson, affected | Contrast R² |
|---|---:|---:|---:|---:|
| Published → published | 0.593 | 0.508 | 0.712 | 0.338 |
| Published → corrected | 0.571 | 0.171 | 0.625 | 0.324 |
| Corrected → corrected | 0.579 | 0.335 | 0.629 | 0.316 |

- **Published labels flatter the model.** On the affected units, a model trained on published labels looks better than one trained on corrected labels can be (0.508 vs 0.335).
- **Why.** The fallback adds variance that can be predicted from drug identity, and the model learns part of it.
- **Benchmark effect.** Across all units the effect on the benchmark is modest: R² −0.014 and contrast R² −0.022.

**Limitations.**
- The cases use the cached Python fits, not native refits of every block.
- The generating SynergyFinder release is unknown.
- The 1,883 accepted cases with no admissible root are reported but left unchanged.

## 2. Stress tests of the high-inhibition result

Declared in `configs/stress_tests.json`. Code: `cs622/stress.py`. Results: `results/stress_tests/`.

**Rule.** A control is survived only if two things hold at R ≥ 75:
- the interval is above 0 (drug jackknife, or cell jackknife for that control);
- every seed is positive.

### ONEIL (selected F2, seeds 2026–2028)

| Control | Pair-cells | Interaction residual R² | Interval | Seeds | Survives |
|---|---:|---:|---|---|---|
| As reported | 14,380 | 0.163 | drug [0.055, 0.271] | 0.193 / 0.140 / 0.156 | yes |
| Cell jackknife (pairs share cells) | 14,380 | 0.163 | cell [0.091, 0.235] | same | yes |
| Largest single-drug influence | | 0.029 (PACLITAXEL) | | | |
| Isotonic Bliss per cell line | 14,380 | 0.168 | [0.059, 0.276] | 0.198 / 0.144 / 0.160 | yes |
| mono_only network as reference | 14,380 | 0.157 | [0.047, 0.267] | 0.187 / 0.133 / 0.149 | yes |
| Mixed doses | 12,480 | 0.171 | [0.028, 0.314] | 0.204 / 0.150 / 0.159 | yes |
| High doses | 14,362 | 0.124 | [0.016, 0.232] | 0.151 / 0.100 / 0.122 | yes |
| **Low doses** | 1,205 | **−0.180** | [−1.171, 0.811] | −0.004 / −0.382 / −0.154 | **no** |
| **Pairs below median analog similarity (0.629)** | 6,278 | **0.195** | [**−0.004**, 0.394] | 0.212 / 0.196 / 0.177 | **no** |
| Batch | n/a | n/a | n/a | n/a | not testable |

- **Two controls fail.**
  - *Low doses.* The cell is small and its interval spans ±1, so it carries almost no information. Under the declared rule it is still a failure.
  - *Analog exclusion.* This halves the units, and the drug interval then crosses 0 by 0.004, although the point estimate rises. Its cell-jackknife interval is [0.105, 0.285].
- **Batch is not testable here.** DrugComb has no ONEIL batch identifiers, and no local copy of the original batch file exists.
- **Pooled across all bins** (unchanged): 0.064, drug [−0.037, 0.165], cell [0.009, 0.119].

### ALMANAC (f2_combined, seeds 2026–2028; mono_only retrained on ALMANAC with the frozen sprint config, 15 runs)

| Control | Pair-cells | Interaction residual R² | Interval | Seeds | Survives |
|---|---:|---:|---|---|---|
| As reported | 68,627 | 0.373 | drug [0.259, 0.487] | 0.361 / 0.384 / 0.376 | yes |
| Cell jackknife | 68,627 | 0.373 | cell [0.319, 0.428] | same | yes |
| Largest single-drug influence | | 0.036 (topotecan) | | | |
| Site (stratified; computed earlier) | 68,627 | 0.328 | [0.165, 0.491] | all > 0 | yes |
| Isotonic Bliss per cell line | 68,627 | 0.376 | [0.263, 0.489] | 0.363 / 0.386 / 0.378 | yes |
| mono_only network as reference | 68,627 | 0.364 | [0.246, 0.482] | 0.350 / 0.374 / 0.366 | yes |
| Low doses | 4,863 | 0.229 | [0.012, 0.446] | 0.221 / 0.215 / 0.251 | yes |
| Mixed doses | 57,704 | 0.388 | [0.248, 0.528] | 0.381 / 0.401 / 0.383 | yes |
| High doses | 68,306 | 0.317 | [0.213, 0.420] | 0.297 / 0.325 / 0.328 | yes |
| Pairs below median analog similarity (0.835) | 33,341 | 0.292 | [0.139, 0.445] | 0.257 / 0.353 / 0.266 | yes |
| NCI test-date main effects removed | 68,627 | 0.328 | [0.222, 0.433] | 0.320 / 0.335 / 0.328 | yes |
| NCI plate main effects removed | 68,627 | 0.267 | [0.150, 0.384] | 0.274 / 0.264 / 0.264 | yes |

- **Batch map.** Test date and plate come from exact row matches to the NCI file: 285,150 pair-cells, 169 screener-date levels and 20,019 plates. The one ALMANAC pair with no matched NCI rows forms its own level.
- **Pooled results.** Pooled values stay positive under every control: plate 0.233 [0.156, 0.310], test date 0.354, mono_only 0.397.
- **What the controls remove.** Plate effects remove the most, about a third of the gain, so part of the ALMANAC signal is plate- or batch-level. What remains is still large.

**Reading.**
- **ALMANAC.** The gain survives every declared control.
- **ONEIL.** The gain survives calibration, the mono_only reference, the cell jackknife and the two larger dose regions. It fails the low-dose cell and analog exclusion, the latter by 0.004.
- **Batch remains untestable for ONEIL.**

## 3. Third screen (frozen protocol, GPU)

Plan: `configs/third_screen_confirmation.json` and amendment 1. Code: `cs622/third_screen.py`. Results: `results/third_screen/`.

**Screen choice.**

| Candidate | Status |
|---|---|
| AstraZeneca–Sanger DREAM (preferred) | Not accessible. Synapse `syn18496666` has a managed ACT access requirement and needs a login; no request was filed. Not in DrugComb v1.4. |
| GDSC combinations (Jaaks 2022) | Public processed files are curve-level summaries only. Dose-level values exist only as raw plate fluorescence (Figshare, internal-research licence), so a new normalization pipeline would be needed. |
| CLOUD (DrugComb) | 39,878 pairs, but one dose per drug and one cell line: no dose-level variation. |
| **NCATS_ES(Nampt+PARP)** (primary) | 6×6/10×10 matrices with block monotherapy. Runs through the frozen pipeline. |
| **NCATS_DIPG** (secondary) | Same laboratory, different cells. Needs one declared rule: four drug spellings are merged. |
| Wilson, NCATS_HL and other DrugComb studies | Under 500 eligible pairs. |

**Independence.**
- NCATS is a different laboratory and platform from Merck (ONEIL) and NCI DTP (ALMANAC).
- No cell line is shared with ONEIL. 6 and 12 drugs are shared.
- The frozen architectures were retrained on each screen. No ONEIL or ALMANAC model enters.

**Amendment 1** (`69fa7f8`, before any test output).
- **Trigger.** The Bliss reference and mono_only need training-pair monotherapy in the test cell. In both NCATS screens most pairs are measured in one cell only.
- **Change.** The seen-in-training rule moved from drugs to drug-cells: at least 10 pairs per drug-cell, and folds keep held-out drug-cells seen.
- **Effect.**
  - NCATS_ES becomes a single-cell (UHO1) all-pairs screen: 2,396 pairs, 72 drugs, 112,165 combination rows.
  - NCATS_DIPG keeps 3,506 pairs, 128 drugs, 2 cells and 224,708 rows.
  - No row was imputed; under 1.5% were interpolated.
- **Training.** 90 network runs; no run reached the epoch budget.
  - f0/f2 stopped after 22–35 epochs;
  - mono_only after 25–154.

**Pre-declared hypotheses** (fixed sequence; rule: drug-jackknife interval above 0 and every seed positive):

| Screen | H1: f2 vs Bliss at R ≥ 75 | H2: f2 vs Bliss, pooled | H3: f2 vs mono_only at R ≥ 75 |
|---|---|---|---|
| NCATS_ES (primary) | 0.057 [−0.006, 0.120]; seeds 0.065 / 0.042 / 0.064. **Not confirmed** | 0.281 [0.184, 0.378], all seeds > 0. Rule met, but not confirmed in sequence | 0.056 [−0.007, 0.118]. Not confirmed |
| NCATS_DIPG (secondary) | −0.005 [−0.078, 0.067]. Not confirmed | 0.021 [−0.026, 0.068]. Not confirmed | −0.008 [−0.081, 0.065]. Not confirmed |

**Bliss-level bins** (f2_combined vs Bliss, Holm over 4 bins):

| Screen | R < 25 | 25–50 | 50–75 | R ≥ 75 |
|---|---|---|---|---|
| NCATS_ES | **0.333** [0.232, 0.434] | 0.217 [0.135, 0.300] | 0.116 [0.037, 0.195] | 0.057 [−0.006, 0.120] |
| NCATS_DIPG | −0.005 | 0.012 | 0.045 | −0.005 |

- **NCATS_ES.** The gain runs the opposite way to ONEIL: largest at low expected effect, smallest at R ≥ 75.
- **Calibration.** Isotonic recalibration of Bliss changes no bin by more than 0.02 on either screen.
- **f0_combined** (no monotherapy) is below Bliss at R ≥ 75 on both screens (−0.089 and −0.238). On NCATS_DIPG it is below Bliss pooled as well (−0.147).

**Response level** (declared comparisons; verdict rule as in the sprint):

| Screen | Model | RMSE | R² | Pearson | ΔMSE vs Bliss | vs f0 | vs mono_only |
|---|---|---:|---:|---:|---|---|---|
| NCATS_ES | f2_combined | 22.73 | 0.732 | 0.856 | −11.4%, lower | −14.1%, lower | −6.8%, lower |
| | mono_only | 23.54 | 0.712 | 0.844 | −5.0%, lower | | |
| | f0_combined | 24.52 | 0.688 | 0.830 | +3.1%, not established | | |
| | Bliss | 24.15 | 0.697 | 0.843 | | | |
| NCATS_DIPG | f2_combined | 14.31 | 0.846 | 0.920 | −15.7%, lower | −10.2%, lower | −4.9%, lower |
| | mono_only | 14.67 | 0.838 | 0.915 | −11.4%, lower | | |
| | f0_combined | 15.10 | 0.828 | 0.910 | −6.2%, not established | | |
| | Bliss | 15.59 | 0.817 | 0.912 | | | |

The additive ridge (RMSE 31.49 and 24.34) and the training mean are far behind on both screens.

**Reading.**
- **The prospective high-inhibition hypothesis fails on the untouched screen.** H1 fails on both NCATS screens, so nothing is confirmed under the fixed sequence.
- **F2 does improve response prediction.** It beats Bliss by 11–16% MSE on both screens.
- **Much of that improvement needs no identities.** An identity-free monotherapy network already beats Bliss by 5–11%.
- **The pair-specific part is screen-dependent.**

| Screen | Interaction residual R² (pooled) |
|---|---:|
| ALMANAC | 0.40 |
| NCATS_ES | 0.28 |
| ONEIL | 0.06 |
| NCATS_DIPG | 0.02 |

- **Possible reasons for the spread** (all untested):
  - pairs per drug-cell;
  - single-cell vs multi-cell design;
  - batch structure, as the ALMANAC plate result shows.

## 4. PR #1 (kept separate)

`reports/kingz_pr_corrections.md` is on `review/pr1-gp-integration` (commit `baca661`). PR #1 is not merged.

The corrections it lists:
- the unsupported 35.90 / 49.47 figures;
- 2.63 GB of LFS objects in the PR commit, which must be removed by rewriting it, not by a deletion commit;
- unrelated history and a `.gitattributes` conflict;
- personal absolute paths;
- unpinned dependencies;
- provisional folds and the random inner split;
- the duplicate data pipeline;
- a version-dependent hash;
- stale text;
- scope limits: one cell and one fold, the 1,500-row cap, fixed ρ and w, and helpers that do not match the protocol.

## Reproduction

```sh
python -m cs622.loewe_verify --cases results/loewe_audit/full_cases.parquet --output results/loewe_verify [--retrained <panel predictions>]
python -m cs622.stress --screen oneil --output results/stress_tests
python -m cs622.stress --screen almanac --output results/stress_tests --batch almanac_unit_batch.csv
python -m cs622.data --config configs/data_ncats_es.json        # and data_ncats_dipg.json
python -m cs622.suite run --protocol configs/third_screen_validation.json --output results/neural/runs/ncats_es_v2 --stage all --device cuda
python -m cs622.suite run --protocol configs/third_screen_dipg_validation.json --output results/neural/runs/ncats_dipg_v2 --stage all --device cuda
python -m cs622.suite run --protocol configs/almanac_mono_only.json --output results/neural/runs/almanac_mono_only --stage all --device cuda
python -m cs622.cv run --config configs/ncats_es_mean.json --output results/neural/runs/ncats_es_v2_mean   # likewise additive, reference_bliss and ncats_dipg
python -m cs622.third_screen --protocol configs/third_screen_validation.json --output results/third_screen/ncats_es
python -m cs622.third_screen --protocol configs/third_screen_dipg_validation.json --output results/third_screen/ncats_dipg
```

**Inputs and artifacts kept elsewhere.**
- The retraining for section 1 uses `experiments/phase2/panel_neural.py` from `feat/cs622-neural-pipeline`, with `--targets published solver_root valid_root valid_root_any`.
- The ALMANAC batch map comes from exact row matches to `ComboDrugGrowth_Nov2017.csv`.
- Run directories, checkpoints and the ALMANAC analog-similarity table stay on RebelX.
