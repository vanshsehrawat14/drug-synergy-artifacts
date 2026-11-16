# Frozen GPU execution and return contract

The [manifest](../results/mechanism_analysis/gpu_manifest.json) specifies exactly **50 tasks**. The [scientific protocol](../configs/mechanism_training.json) was frozen before new training; all follow-ups on these inspected data are exploratory. No GPU job was launched by this continuation.

| Tasks | Campaign | New runs | Matched controls |
|---|---|---:|---|
| 0–4 | Seed-2029 full-input F2 | 5 | Necessary baseline for tasks 5–19 |
| 5–9 | Seed-2029 no drug identity | 5 | New full-input baseline, same seed/fold |
| 10–14 | Seed-2029 no cell identity | 5 | Same |
| 15–19 | Seed-2029 no dose | 5 | Same |
| 20–34 | Joint identity omission | 15 | Existing full F2, no drug and no cell; seeds 2026–2028, five folds |
| 35–49 | No drug identity, Bliss-residual objective | 15 | Existing `f2_residual`, same seeds/folds; raw-target masks as secondary controls |

For each arm, folds are ordered 0–4 within each seed. Tasks 20–34 and 35–49 use seeds 2026, 2027, 2028. Tasks 0–19 use seed 2029 only. Complete configuration objects and their canonical JSON hashes are in the manifest. The original dataset is `oneil_v1`, SHA-256 `ea588405019971906ac99c3caa547d5f4fac5b1be340356f0a8fe17a6aa3ce8b`.

## Scientific comparisons

**Seed closure.** The exact four endpoints are drug/pair-cell contrast, cell/pair-cell contrast, cell/dose-shape contrast and dose/dose-shape contrast. Task reuse covers the two cell endpoints. Run all matched baseline folds; comparing a new mask seed with a historical seed is invalid. Drug-identity response is an additional exploratory check sharing tasks 5–9. Report seed 2029 and four-seed pooled errors separately, retaining the original seeds, thresholds and Holm family.

**Joint identity experiment.** Hypothesis: conditional on per-dose monotherapy, drug and cell channels have nonadditive predictive error costs. The competing explanation is approximately additive costs under the frozen learning procedure. Primary effect:

$$
\frac{SSE_{\mathrm{both}}-SSE_{\mathrm{no\ drug}}-SSE_{\mathrm{no\ cell}}+SSE_{\mathrm{full}}}{SSE_{\mathrm{full}}}.
$$

A positive effect means joint removal costs more than the sum of single-removal penalties; a negative effect means less. This is an algorithm-conditional information diagnostic, not drug synergy. Parameter-count/initialization differences and retained lookup information remain limitations. Secondary decomposition localizes the effect to additive or reference-augmented contrasts.

**Residual-objective experiment.** Hypothesis: the apparent no-drug pair-cell contrast advantage persists with a fixed training-pair Bliss offset. Competing explanations are raw-target conditioning/optimization and a persistent identity-channel contrast penalty. New models fit `Y − R_Bliss` and add the same reference back. Primary metric is paired relative MSE change on the reference-augmented pair-cell contrast, versus the full-input `f2_residual` model. Original contrasts, response error and raw/residual differences are secondary. Failure or reversal weakens the proposed identity-overfitting account; persistence is not sufficient to identify its biological cause.

Both new experiments use the production architecture, optimizer and validation checkpoint rule. No architecture/hyperparameter search is permitted. One primary endpoint per new experiment; apply Holm across the two primary tests only after both are complete. A material new effect requires magnitude ≥5%, both pair/drug grouped intervals excluding zero, Holm p<0.05 and all five folds agreeing. This does not redefine historical endpoints. Missing tasks, source mismatch, leakage, nonfinite outputs or irrecoverable deterministic failure stop the affected campaign. No outcome-dependent seed extension is authorized by this packet.

## Resources and dependencies

Use one **allocated CUDA GPU**, four CPU cores, **16 GiB RAM** and a **30-minute cap per task**. The array permits **one active task**. Historical A100 training time is 59–509 seconds, median 161, mean 185. Estimates exclude preparation, queue and evaluation overhead:

| Campaign | Mean-based GPU hours | 2× historical-maximum allowance |
|---|---:|---:|
| 20 seed-2029 tasks | 1.03 | 5.66 |
| Each 15-task follow-up | 0.77 | 4.25 |
| Total | 2.57 | 14.15 |

Dependencies: existing Python/PyTorch CUDA environment, prepared O'Neil data, complete historical `f2_base` and `f2_residual` controls and all original ablations. `mpmath`, plotting and pytest dependencies are needed for packet checks. No new dataset or remote service is needed. Scripts fail if controls differ; they never substitute selected or combined models. All required existing controls passed local configuration/completeness checks.

## Reproduction

Run from this checkout after regenerating the frozen controls and ablations. `TASK` is an integer from 0 to 49. Use one allocated CUDA GPU, four CPU cores, 16 GiB RAM and a 30-minute limit per task; run tasks serially. The task table and stopping rules above apply unchanged.

```sh
python -m synergy_artifacts.mechanism_jobs --task TASK --check-only --baseline-root . --ablations results/interaction_analysis/runs
python -m synergy_artifacts.mechanism_jobs --task TASK --output results/mechanism_analysis/runs --baseline-root . --ablations results/interaction_analysis/runs
```

The commands validate matched controls and resume only matching checkpoints. Site-specific scheduler wrappers and transfer bundles are omitted; no additional repository or historical Git objects are needed for a new run. Stored commit identifiers describe the original execution.

## Analysis after return

Verify archive paths, manifest and hashes before extraction into a fresh directory. The following commands require local historical predictions and the original resolution cache; replace capitalized paths explicitly:

```sh
python -m synergy_artifacts.mechanism_analysis --campaign seed2029 --runs RETURNED_RUNS --ablations ORIGINAL_ABLATIONS --baseline-root HISTORICAL_ROOT --resolution RESOLUTION_CACHE --output NEW_SEED_ANALYSIS
python -m synergy_artifacts.mechanism_analysis --campaign joint_identity --runs RETURNED_RUNS --ablations ORIGINAL_ABLATIONS --baseline-root HISTORICAL_ROOT --resolution RESOLUTION_CACHE --output NEW_JOINT_ANALYSIS
python -m synergy_artifacts.mechanism_analysis --campaign residual_drug_identity --runs RETURNED_RUNS --ablations ORIGINAL_ABLATIONS --baseline-root HISTORICAL_ROOT --resolution RESOLUTION_CACHE --output NEW_RESIDUAL_ANALYSIS
python -m synergy_artifacts.mechanism_analysis --combine NEW_JOINT_ANALYSIS/comparisons.csv NEW_RESIDUAL_ANALYSIS/comparisons.csv --output NEW_TWO_TEST_DECISION.csv
```

Analysis checks configuration, dataset hash, exact out-of-fold row IDs and finite predictions. It averages seed-specific squared errors, refits both nuisance stages under drug deletion and uses whole-pair bootstrap. The combined decision requires both primary experiments; neither missing tasks nor the exploratory drug-response label is silently waived.
