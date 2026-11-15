# Seed-2029 confirmation, 2026-11-08

Follow-up to the [overnight analysis](../overnight_20261107/research_findings.md), originally executed from source `a5adc3d`. This record preserves the original execution; the reproduction commands below use the current checkout.

## Bottom line

- **Execution.** All 20 seed-2029 tasks completed: 3 masks plus the matched full-input F2 control, 5 folds each, RebelX array `132713`. All exit 0, with 0 failed, 0 unfinished and 0 retries. Every config equals its frozen reference except the seed.
- **Preregistered confirmations.** Using the frozen seed-averaged estimator pooled over seeds 2026–2029, none of the four triggered cells is material. That matches the original classification, so the recorded conclusions stand.
- **Seed 2029 alone.** One cell disagrees: removing drug identity reduces pair-cell contrast error by 12.6% [−16.7, −8.6; −22.0, −3.3]. All five folds are negative and Holm p = 0.029, so this single seed is material. The per-seed estimates are −4.8, −9.1, −2.6 and −12.6%. The direction is the same in every seed, but the magnitude depends on the seed. Pooled, the change is −7.4% and all five folds are now negative. It still fails two criteria: the drug-jackknife upper bound is +0.5% and Holm p = 0.20. This cell is **conditional, not confirmed**.
- **Exploratory: drug-identity response.** Seed 2029 alone gives +3.2%; the pooled estimate is +4.6%. Neither is material, because the pooled change is below 5%. The original +5.08% material call (claim C5) does not reproduce. Per-seed values are 6.2, 3.4, 5.7 and 3.2%. The seed-to-seed spread comes mostly from the full-input control: its row MSE ranges from 150.3 to 154.2 across seeds, while the masked model stays at 158.9–159.6.
- **Cell identity response.** Still material: pooled +13.1%, seed 2029 alone +12.3%.
- **Mechanisms packet (`106855c`).** Its tasks 20–49 ran as array `132733`, 30/30 complete (§5). Neither of its two primary endpoints is material: joint identity omission gives a factorial interaction of +2.2%; no drug identity under the Bliss-residual objective gives −3.9% on contrast, with the folds disagreeing. Its tasks 0–19 are exactly the 20 runs above, with identical configs, so they were not trained twice.

## 1. What the frozen protocol required

Rule (`preregistered_analyses.md` §6): "one extra seed (2029, five folds, unchanged config) for a mask only if its classification for an endpoint depends on one fold or a grouped bound lies within 0.5 pp of the 5% threshold." The coded trigger (`cs622/ablation_summary.py`, column `confirmatory_seed_trigger`) fired for four mask × endpoint cells in `handoff/gpu/ablation_comparisons.csv`:

| Cell | Why it fired (original 3 seeds) |
|---|---|
| Drug identity removed, pair-cell contrast | change −5.54%; fold range −10.20% to +0.42% (one sign disagrees with |change| ≥ 5%) |
| Cell identity removed, pair-cell contrast | pair-bootstrap upper bound 4.89% (within 0.5 pp of 5%) |
| Cell identity removed, dose-shape contrast | drug-jackknife upper bound 4.70% |
| Dose removed, dose-shape contrast | drug-jackknife upper bound 4.50% |

All three masks therefore needed seed 2029 (5 folds each). The comparisons are paired against full-input F2 at the same seed, and no seed-2029 F2 existed (`sprint_v1/f2_base` has seeds 2026–2028 only), so the matched F2 control was trained at seed 2029 too: 4 models × 5 folds = 20 tasks. Drug-identity response (+5.08%, not triggered) is reported separately as an exploratory robustness check; it reuses the same `no_drug_identity` seed-2029 runs, so it needed no extra GPU work.

The coded trigger reads "depends on one fold" as "one of the fold extremes has the opposite sign". The literal one-fold reading is reported per fold in `analysis/*/ablation_comparisons.csv` (`fold_changes`).

## 2. Cutoff, allocation and scheduling checks

- **08:20 cutoff.** It is the overnight directive's default cap ("no new compute jobs after 10 hours from start", constraint 3), not a site rule. It bound new jobs under that directive, and none were launched after it. This seed-2029 work is a separate request issued after the cap that explicitly asks for these runs. Partition `sxmq` has no time limit or QoS (`MaxTime=UNLIMITED`, `QoS=N/A`), and the RebelX compute guidelines set no time-of-day limit. No site limit is stricter.
- **Allocation.** No jobs of mine were queued or running at 09:20 PST. `sxm001` was drained, and `sxm002` was mixed with 14 of 32 CPUs allocated. SLURM does not track GPU gres for the other users' jobs. Every seed-2029 task was bound to GPU 0, which already carried 20.6 GB of other processes' memory out of 80 GB. The overnight array saw the same situation (20–42 GB). The worker uses the scheduler's binding (`SLURM_JOB_GPUS`) as the packet worker does.
- **Footprint.** One GPU at a time (`--array=0-19%1`). Each task gets 4 CPUs, 16 GB and an 8 h limit, the same as the packet `worker.sbatch`.

## 3. Execution

- **Driver.** `experiments/overnight_20261107/seed2029/seed2029_job.py` (local branch commit `77c77be`; byte-identical to the copy run on RebelX). It builds each config with the packet's own `cs622.research_jobs.job()` for the same mask and fold, then sets `seed = 2029`. For the full-input control it also removes the single mask key. Before training it asserts that the result equals the saved frozen config with only the seed replaced:
  - masks: the overnight `runs/<mask>/seed_2026/fold_k/config.json`;
  - control: the historical `sprint_v1/f2_base/seed_2026/fold_k/config.json`.

  All 20 passed. The full-input baseline at d2c8f41 matches the historical `f2_base` code: between `21ca03e` and `d2c8f41`, `features.py`/`runner.py` only add mask flags that default to on, and the `data.py` changes are in dataset preparation. The dataset SHA-256 is unchanged.
- **Launch.** `submit_seed2029.sh --partition=sxmq` repeats the packet `submit.sh` checks before `sbatch`: clone at `d2c8f41`, clean tree, no existing `seed2029/` output, packet tests (9 passed), and the config check.
- **Array.** `132713` (`0-19%1`) ran on node `sxm002`, GPU 0. Tasks 0–4 are `f2_base`, 5–9 `no_drug_identity`, 10–14 `no_cell_identity` and 15–19 `no_dose`, each covering folds 0–4.
  - Submitted at about 09:27; tasks ran from 09:27:50 to 10:26:23 PST.
  - Training took 59–349 s per task.
  - Per-task Slurm job IDs (132713–132732), times, exit codes and hashes are in `run_ledger_seed2029.csv`.
  - Each task's `.out` ends with an `=== exit` line written by an EXIT trap, so exit codes are logged even without Slurm accounting.
- **Validation before statistics.**
  - Each mask × seed and the F2 control: 1,475,328 out-of-fold rows in frozen test order, 0 duplicates, all finite, seed/fold/mask flags as expected (`analysis/*/ablation_validation.csv`).
  - Archive: all 227 files match its manifest.
  - Code path:
    - The seed-2029 control path, given seed-2026 copies, reproduces the cached seed-2026 statistics exactly (max abs diff 0.0).
    - Recomputed historical F2 unit means and pair SSE equal the resolution cache exactly (`analysis/f2_path_check.json`).
    - The default three-seed mode reproduces every value in the overnight `ablation_comparisons.csv`, `ablation_metrics.csv` and `ablation_validation.csv` (`analysis/original_2026_2028/`; the only change is the added `fold_changes` column).

**Deviations.**

1. **Dirty-tree flag.** The runs record `git.dirty = true`. The clone's `.gitignore` ignores only `results/research_20261107/{runs,slurm}/`, so the new untracked `seed2029/` output directory made the tree dirty. Checked: the 20 runs match the overnight d2c8f41 runs in commit, `code_sha256` for every module, dependencies, Python, dataset hash and split rule, and no tracked file is modified. Afterwards, `/results/research_20261107/seed2029/` was added to the clone's local `.git/info/exclude` (as was done for `/data` overnight), and the clone is clean again.
2. **How the extra seed enters the decision.** The protocol does not say. Both the pooled four-seed result (the frozen seed-averaged estimator) and the seed-2029-only replicate are reported, with all decision rules unchanged. Single-seed Holm p-values use the same within-seed delete-one-drug jackknife.
3. **Coded versus literal fold rule.** The coded trigger reads "depends on one fold" as "a fold extreme has the opposite sign". Per-fold changes are now saved (`fold_changes`) so the literal reading can be checked. For the one fold-triggered cell (drug identity, pair-cell contrast), exactly one fold disagreed in the original three seeds (fold 3, +0.42%; the others were −4.3 to −10.2%), so the two readings agree there. The other three cells were triggered by interval bounds.
4. **New trigger in the pooled result.** The pooled `no_dose` response now meets the coded trigger: pair upper bound 5.38%. The protocol authorizes a single extra seed, so no further seeds were run.

## 4. Results

Each cell gives the relative MSE change versus full-input F2 at the same seeds, folds and rows. A positive value means the masked model is worse. Brackets give the pair-bootstrap and drug-jackknife 95% intervals. Holm correction is across the three masks within each endpoint. "Material" is the frozen rule: magnitude ≥ 5%, Holm p < 0.05, both grouped intervals exclude zero, and all five folds have the same sign.

| Cell | Original (2026–2028) | Pooled 2026–2029 | Seed 2029 only | Classification |
|---|---|---|---|---|
| Drug identity, pair-cell contrast (triggered) | −5.54%, not material | −7.41% [−10.73, −3.94; −15.32, +0.50], Holm p 0.20, all 5 folds negative | **−12.63%** [−16.67, −8.57; −22.00, −3.27], Holm p 0.029, all 5 folds negative, **material** | Unchanged under pooled; material in the seed-2029 replicate. Conditional. |
| Cell identity, pair-cell contrast (triggered) | +1.19%, not material | −0.59% [−3.66, +2.70; −8.93, +7.74], p 0.89 | −5.56% [−9.77, −1.12; −15.44, +4.32], p 0.52 | Unchanged, not material |
| Cell identity, dose-shape contrast (triggered) | +2.09%, not material | +1.79% [0.84, 2.77; −0.29, 3.88], p 0.18 | +0.92% [−0.44, 2.36; −1.36, 3.19], p 0.42 | Unchanged, not material |
| Dose removed, dose-shape contrast (triggered) | +2.02%, not material | +2.17% [1.21, 3.18; 0.00, 4.34], p 0.15 | +2.63% [1.13, 4.12; −0.19, 5.46], p 0.20 | Unchanged, not material |
| Drug identity, response (exploratory) | +5.08%, material | +4.61% [3.21, 6.09; 1.50, 7.73], p 0.005 | +3.23% [1.31, 5.14; −0.30, 6.75], p 0.14 | Not material in either. The original call does not reproduce. |

Other cells, pooled 2026–2029:

| Mask | Response | Pair-cell contrast | Dose-shape contrast |
|---|---|---|---|
| Cell identity removed | **+13.14%**, material | −0.59% | +1.79% |
| Dose removed | +4.01% | +4.27% | +2.17% |

Full tables:

- `analysis/comparisons_by_analysis.csv`: every mask × endpoint for the pooled analysis and each single seed.
- `analysis/decision_table.csv`: the five cells above, with every decision statistic.
- `analysis/*/ablation_metrics.csv`: per-seed and pooled metrics, including response, contrast R², Bliss residual skill and excess recovery.
- `analysis/response_change_by_seed_fold.csv`: per-seed, per-fold response changes.

Response change by seed (masked vs F2 at the same seed):

| Mask | 2026 | 2027 | 2028 | 2029 |
|---|---|---|---|---|
| Drug identity | 6.21% | 3.36% | 5.71% | 3.23% |
| Cell identity | 14.11% | 10.89% | 15.30% | 12.32% |
| Dose | 3.59% | 3.40% | 6.44% | 2.65% |

Row MSE of the full-input F2 control by seed: 150.30, 153.73, 150.84 and 154.15. Seed 2029's control is the weakest of the four, which lowers every seed-2029 response change.

**Interpretation limits.** No new winner claims are made. The drug-identity contrast effect has a consistent direction but a seed-dependent size. Read it as evidence that this frozen network's drug-identity channel hurts pair-specific contrast prediction to a variable degree, not as an identified mechanism. The residual-objective experiment (§5) tests that account directly.

## 5. Mechanisms packet (branch `research/mechanisms-2026-11-08`, `106855c`)

**Acceptance and review.**

- Packet `cs622_mechanisms_106855c_r2.zip` (SHA-256 `71d64cb5dabcc71f3aa7a073ba14cf1b78f47d511c5d3deacd67c3d5b6be9db0`) was uploaded to `PROJECT_ROOT/mechanisms_106855c/`. All 11 files match `packet_manifest.json`.
- `prepare.sh` made a fresh clone `PROJECT_ROOT/cs622_gpu/wt_mechanisms_20261108` at `106855c` and ran:
  - `tests/test_mechanism_audits.py`: 11 passed;
  - the control checks: valid for all three campaigns.
- Requested resources: one GPU, 4 CPUs, 16 GiB, a 30-minute cap per task and one active task. These are within partition limits.
- Its tasks 0–19 are the same 20 seed-2029 configurations, verified identical dict for dict against array `132713`. As the packet's README directs, they were not resubmitted.

**Execution.**

- Array `132733`: `bash submit.sh --partition=sxmq --array=20-49%1`, submitted 10:44 PST. Tasks ran on `sxm002` and finished by about 12:35.
- 30/30 complete, 0 failed, 0 retries.
- An external `scontrol` logger (`~CLUSTER_PATH`) recorded `COMPLETED` and `ExitCode=0:0` for every task.
- The clone is still clean.
- Collected with the packet's `collect_return.py --checkpoints`:
  - `PROJECT_ROOT/cs622_gpu/research_return_20261108_mechanisms.tar.gz`
  - 67,511,247 B, SHA-256 `f05e479683c0c42c223d304822377eac31a27558f708a68adb73e514dad869c1`
  - 270 files, all hashes verified locally
  - its "incomplete" list is tasks 0–19, i.e. the seed-2029 runs held in the `132713` archive.

**Packet analysis.** Run with the packet's `cs622.mechanism_analysis` from a fresh clone of its bundle (`mechanisms_analysis/`). Both experiments are exploratory under the packet's own rule (magnitude ≥ 5%, both grouped intervals exclude zero, Holm p < 0.05 over the two primaries, all folds agree).

| Experiment | Primary endpoint | Change [pair; drug 95%] | Folds | Holm p | Material |
|---|---|---|---|---|---|
| Joint identity omission (tasks 20–34) | Factorial response-error interaction (SSE_both − SSE_no_drug − SSE_no_cell + SSE_full)/SSE_full | +2.21% [0.82, 3.59; −1.15, 5.57] | all positive | 0.38 | No |
| No drug identity, Bliss-residual objective (tasks 35–49) | Reference-augmented pair-cell contrast vs `f2_residual` | −3.85% [−7.03, −0.41; −11.17, 3.47] | disagree (−11.2 to +8.4%) | 0.38 | No |

**Reading.**

- **Joint omission.** Removing both identity channels costs slightly more than the sum of the two single removals, consistently across folds. The effect is below 5%, and the drug-jackknife interval spans zero and does not rule out a 5% effect. Inconclusive.
- **Residual objective.** With the Bliss offset fixed, the no-drug contrast advantage shrinks from −7.4% (raw target, pooled) to −3.9%, and the folds disagree. By the packet's registered reading this weakens the identity-overfitting account. It does not identify a cause.
- **Cross-check.** The packet's seed-2029 analysis, run on the `132713` runs, reproduces every change, Holm p and material classification in §4 exactly (`mechanisms_analysis/seed2029_crosscheck/`).

## 6. Artifacts, hashes and reproduction

| Archive (RebelX, retained) | Bytes | SHA-256 |
|---|---|---|
| `PROJECT_ROOT/cs622_gpu/research_return_20261108_seed2029.tar.gz` (array 132713, all outputs including checkpoints; 4 parts in `PROJECT_ROOT/research_d2c8f41/seed2029_parts/`) | 112,903,789 | `454faac2de5c236d03165a10682eca7e0bc5ef9ae97e9e1729fd27b51917d441` |
| `PROJECT_ROOT/cs622_gpu/research_return_20261108_mechanisms.tar.gz` (array 132733; 3 parts in `PROJECT_ROOT/mechanisms_106855c/return_parts/`) | 67,511,247 | `f05e479683c0c42c223d304822377eac31a27558f708a68adb73e514dad869c1` |
| `PROJECT_ROOT/cs622_gpu/research_return_20261107_final.tar.gz` (overnight, untouched) | 1,011,650,207 | `2b75eeba78743ea15e6d72039df6b90e985ad49186192ebcd0584de174fc5ce4` |

Both retained return archives were checksum-verified. Model checkpoints are retained by the authors and are not redistributed.

**Jobs.**

- `132713`: seed 2029, tasks 0–19; per-task job IDs 132713–132732 are in `run_ledger_seed2029.csv`.
- `132733`: mechanisms tasks 20–49; per-task job IDs are in the logger file.
- Failed tasks: none. Unfinished tasks: none.

**Experiments not executed.**

- No further confirmatory seeds were run, even though the pooled `no_dose` response now meets the trigger; the protocol allows one extra seed.
- No other campaigns were run.

**Code.**

- Local branch `research/overnight-2026-11-07`, commit `77c77be`:
  - `seed2029/` driver, sbatch, submit and pack scripts;
  - `experiments/overnight_20261107/seed2029_report.py`;
  - `cs622/ablation_summary.py`: `--seeds`, `--f2-runs`, multi-root `--runs`, `fold_changes`; default output unchanged.
- Tests: 107 passed, 1 CUDA skip (`logs/tests_seed2029.log`).

**Reproduction.**

Run from this checkout after regenerating the original full-input and ablation controls. `TASK` is an integer from 0 to 19; use one allocated CUDA GPU at a time. Original task identifiers and commit hashes above describe the recorded execution.

```sh
python experiments/overnight_20261107/seed2029/seed2029_job.py --check-only --historical results/neural/runs/sprint_v1/runs/f2_base
python experiments/overnight_20261107/seed2029/seed2029_job.py --task TASK --historical results/neural/runs/sprint_v1/runs/f2_base --output results/research_20261107/seed2029
python experiments/overnight_20261107/seed2029_report.py \
  --runs results/research_20261107/runs results/research_20261107/seed2029 \
  --f2-runs results/research_20261107/seed2029/f2_base --resolution RESOLUTION_CACHE \
  --original results/overnight_20261107/ablation_comparisons.csv --output NEW_ANALYSIS
```

The [mechanisms protocol](../../reports/mechanisms_gpu_runs.md) gives the other campaigns and matched analysis commands. Scheduler and transfer scripts from the original execution are omitted.
