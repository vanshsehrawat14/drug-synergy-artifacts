# Response analysis: findings

Base revision `d2c8f41`. The analysis plan was registered before any outcome was computed (`preregistered_analyses.md`). Methods, inputs and deviations are in `methods_and_provenance.md`. Every number below comes from a file in this directory. "Pair-cell" means an unordered drug pair × cell line (22,737 O'Neil units). "Top-k" means the k% highest Loewe scores, with ties broken by canonical key.

## Findings

The central claim survives in a narrower and more precise form. With curve fits fixed, the native SynergyFinder 2.0.x treatment of near-asymptote Loewe roots changes 5.9–9.2% of top-1% to top-20% pair-cell selections. Every one of those changes is mathematically certified.

The effect has three properties:

- **Convention, not accuracy.** It comes from the edge convention, not from root accuracy. Of the absolute score change behind changed decisions, 97.5–100% comes from roots within 0.01 inhibition points of a curve asymptote. Treating roots closer than ε as undefined removes most of it: ε = 1e-6 gives 4.40% top-5% turnover, ε = 1e-4 gives 2.11%, and ε = 1e-2 gives 0.09%.
- **Directional.** The native summed-dose fallback overstates the Loewe reference in 99% of these cases, so the affected combinations are systematically under-scored. Every one of the 5,009 changed pair-cells moves toward synergy.
- **Comparable to rounding.** It is about the size of rounding native pair-cell scores to integers (5.01% top-5% turnover) and far smaller than the HSA fallback convention (30.9%).

The numerically correct statement is: *a scorer's undocumented, solver-path-dependent edge rule shifts about 6–9% of prioritized combinations in one direction, and the right edge convention is a modelling choice that must be stated.* It is not a claim that one ranking is biologically better.

The other new findings:

- **Cross-screen:** input-strata standardization does not explain the cross-screen gap. This is a negative result.
- **Reliability:** results are unchanged and negative.
- **GPU ablations:** all 45 frozen tasks completed. Removing cell identity raises response MSE by 13.4%, a material and seed-robust change. Removing drug identity raises it by 5.1%, which is material by the frozen rule but marginal and seed-fragile. Removing dose raises it by 4.5%, below the threshold. No mask materially changes pair-cell or dose-shape contrast error (§4).

## 1. Claim ledger

### Verified positive

| # | Assertion | Evidence |
|---|---|---|
| P1 | Every one of the 367,152 dose cases has a mathematically certified status under fixed fits. There are 195,984 interval-certified roots (bracket ≤3.3e-11), 117,824 certified non-existences, 49,184 opposite/flat curves and 4,160 empty joint intervals, with 0 unresolved. | `root_certificates/certificates.parquet`; method in `methods_and_provenance.md` |
| P2 | The certificate independently selects exactly the 14,197 + 1,423 = 15,620 fallback cases previously corrected, no more and no fewer. Accepted native values lie within 2.7e-6 of certified roots. | `loewe_regime_summary.csv` |
| P3 | The reported 5.89% (67/1,137) top-5% N→C1 turnover reproduces exactly from row IDs. | `decision/top5_N_to_C1_changes.csv`, `decision/decision_summary.json` |
| P4 | Native → interval-certified top-k turnover is 9.21 / 7.91 / 6.95 / 6.73 / 5.91% at 1/2/5/10/20%. The pair-bootstrap lower bounds are 5.70 / 4.84 / 5.19 / 5.54 / 5.23%, and the drug-LOO minima are 6.57 / 4.46 / 6.20 / 6.24 / 5.56%. N→C1 gives 8.33 / 7.25 / 5.89 / 5.72 / 5.23%. | `decision/ranking_stability.csv` |
| P5 | All N→I changes are guaranteed. With certified brackets and no unresolved cases, the lower and upper bounds coincide; the maximum pair-cell score interval width is 8.3e-12. | `decision/guaranteed_changes.csv` |
| P6 | Root arithmetic is stable. Coefficients rounded to 6/8/10/12 significant digits produce 0 certification-status changes and 0 top-k changes; root movement is at most 1.2e-3 at 6 digits. The earlier 195 double-precision acceptance flips are an artifact of the double-precision residual test and do not appear under interval certification. | `decision/perturbation_stability.csv`, `root_certificates/rounded_*.parquet` |
| P7 | An independent float64 Brent solver agrees with the certificate to 8.5e-14 and the θ-solver to 2.3e-12 on 3,371 stratified cases. The only failures are 27 roots within float resolution of an asymptote. No difference is attributable to solver tolerance. | `root_certificates/independent_solver_check.csv` |
| P8 | Mathematical sanity checks pass. The inverse identity holds to 2.4e-29 at 60 digits; arm-order symmetry shows 0/20,000 mismatches; certified references show 0/271,240 dose-monotonicity violations. Fallback reconstruction matches native to 1.2e-11; block-score reconstruction matches to 1.5e-14. | `root_certificates/identity_tests.json`, `methods_and_provenance.md` |
| P9 | The correction is directional. References fall in 99.0% (boundary) and 96.3% (interior) of corrected cases. All 5,009 affected pair-cells increase (mean +4.04 points). Entering top-5% units gain +9.35 on average; exiting units are displaced (+0.001). | `decision/direction_and_boundary.json` |
| P10 | Prospective, model-predicted scoring keeps the effect. With out-of-fold neural F2 predictions held fixed as responses, N→C1 top-5% turnover is 6.42–6.77% across seeds (6.86% for the seed mean) and top-1% turnover is 11.0–12.7%. | `decision/prospective_scoring.csv` |
| P11 | Cell identity carries non-redundant response information beyond the F2 monotherapy features. Removing it raises dose-row MSE by 13.4% (pair interval [11.7, 15.2], drug interval [9.6, 17.2], Holm p = 5e-8). The increase is 10.9–15.3% in each seed and 8.4–17.6% in each seed × fold. | `gpu_ablation_summary.csv`, `gpu/ablation_seed_check.csv` |

### Verified negative

| # | Assertion | Evidence |
|---|---|---|
| N1 | Interior-root corrections alone (roots ≥0.01 from an asymptote) do not change top-k decisions materially: 0–0.44% turnover. The 1,631 interior cases are real but rarely decision-relevant. | `decision/edge_exclusion_sensitivity.csv` (ε=1e-2) |
| N2 | Decision-unit aggregation absorbs most of the effect. Drug-pair aggregates change 0–8.33% of very small lists (0/6, 1/12, 0/30, 2/59, 6/117). Block-level changes (6.6% at top-5%) track pair-cell changes. | `decision/ranking_stability.csv` |
| N3 | Input-strata standardization does not reduce cross-screen gaps. The F2-vs-Bliss gain gap relative to O'Neil moves from −0.173 to −0.180 (ES), −0.140 to −0.165 (DIPG) and +0.062 to +0.120 (ALMANAC) after reweighting to O'Neil strata (Bliss level × log dose × monotherapy availability; ≥99.9% common support). The frozen experiment-3 criterion is not met. | `cross_screen/cross_screen_standardization.csv` |
| N4 | The failed NCATS confirmation is unchanged. H1 (interaction residual R² at R ≥ 75) has a jackknife interval of [−0.006, 0.120], and the sequence stops there. This session's response-level gains do not bear on H1. | `wt_gpgpu/results/third_screen/ncats_es/hypotheses.json` (unchanged) |
| N5 | Reliability is unchanged. No uncertainty method meets the frozen utility rule (≥+5 pp and both grouped intervals >0). Neural disagreement reaches +6.51 pp, but its drug interval is [−0.59, 13.60]. | `reliability_summary.csv` |
| N6 | Published versus native reconstruction is not a material source: 0.09–0.88% turnover. | `decision/ranking_stability.csv` |
| N7 | No input mask materially changes interaction-level error. Pair-cell contrast changes are −5.5%, +1.2% and +7.1%, and dose-shape contrast changes are −1.5%, +2.1% and +2.0% (drug, cell and dose removed). Each fails at least one frozen criterion: no-dose pair-cell contrast has both grouped intervals above zero and agreeing folds, but Holm p = 0.093. The response-level information in these inputs does not show up as interaction-specific skill. | `gpu_ablation_summary.csv` |
| N8 | Removing dose raises response MSE by only 4.5% [3.0, 5.9] (Holm p = 0.001, below the 5% threshold) and dose-shape contrast error by 2.0%. A plausible explanation, not tested here, is that F2 already receives the monotherapy response at each tested dose. | `gpu_ablation_summary.csv` |

### Conditional

| # | Assertion | Condition |
|---|---|---|
| C1 | A material (≥5%) top-5% prioritization effect exists. | Holds only if roots closer than about 1e-7 inhibition points to an asymptote are accepted as Loewe expectations, i.e. the scorer's own strict-interior definition. Under any edge rule with ε ≥ 1e-6 it falls below 5%. |
| C2 | Native rankings are "wrong". | Only relative to the strict-interior definition. Native rankings sit within 0.22–1.36% of a *consistent* ε=0.01 exclusion convention, which also rejects the 2,896 near-edge roots native accepted. The native defect is inconsistency (solver-path-dependent edge treatment), not proximity to any reasonable convention. |
| C3 | Cross-screen differences reflect measurement and design heterogeneity. | Pooled replicate variance at identical nominal inputs is 83 (O'Neil), 134 (ALMANAC), 819 (ES) and 524 (DIPG) inhibition-points². NCATS has 1–2 cell lines and mostly single-block pair-cells. These are descriptive; replicate subsets differ by screen, so they are not noise floors. |
| C4 | 1,883 native-accepted values exist where Loewe is undefined (opposite/flat curves). Switching them to fallback changes 1.14% of the top-5%. | Convention choice; no correct value exists. |
| C5 | Drug identity materially improves response prediction: +5.08% MSE when removed (pair [3.7, 6.6], drug [1.8, 8.4], Holm p = 0.004, all folds agree). | Classification rests on a point estimate 0.08 pp above the 5% threshold. Per seed the change is 6.2, 3.4 and 5.7%, so seed 2027 alone would not qualify. The preregistered seed trigger checks interval bounds only, so it did not fire here. Removing drug identity also *lowers* pair-cell contrast error by 5.5% (drug interval [−13.8, 2.8], folds disagree). This is exploratory and not material. |

### Unresolved

| # | Question | Why |
|---|---|---|
| U1 | Which edge convention best predicts independently measured combination response? | Needs replicate plates or a held-out experimental shortlist (see next experiments). |
| U2 | How large is monotherapy measurement and fit uncertainty relative to the edge convention at population scale? | Only 160 status-stratified native refits exist, with no independent monotherapy repeats. Perturbation SD 0.1 already gives 7.5% (8-surface lists), so fit uncertainty may equal or exceed the numerical effect. Not a population estimate. |
| U3 | Does the effect generalize to other screens or SynergyFinder ≥2.4.16? | The 2.4.16+ grid solver differs. ALMANAC/NCATS Loewe scoring was not reconstructed here. |

## 2. Strongest counterarguments and refutation tests

1. **"This is a convention, not an error."** Largely accepted (C1, C2). The precise residual claim is that native scoring applies two incompatible edge rules depending on Newton's termination code. Native selections differ from the consistent ε = 0.01 rule by 0.22–1.36% and from the consistent ε = 0 rule by 5.91–9.21%, so native is close to one rule without implementing either. *Refuted if* a documented SynergyFinder rule reproduces the native acceptance pattern from coefficients alone, with no dependence on solver path. The certificates and `loewe_regimes.csv` allow this test.
2. **"The changes are near-ties."** Partly. The median changed-unit margin to the cutoff is 0.35 points (score SD 15.5). Rounding scores to integers alone gives 5.01%. But the changes are one-directional (P9) and certified (P5), so they are not noise. Near-tie reasoning bounds their practical importance; it does not reverse their sign.
3. **"Fit uncertainty swamps it."** Plausible (U2). The existing refits show 7.5–27.5% turnover in 8-surface stress lists at SD 0.1–1. No population estimate exists.
4. **"Few drugs drive it."** Not supported. Changed top-5% units involve 36 of 38 drugs and 35 cell lines; the top three drugs account for 22.8% of drug appearances. DASATINIB is most frequent at 33/158 units. The drug-LOO minimum stays above 4.46% at every budget.
5. **"Model-based prioritization would hide it."** Not supported (P10). In prospective use, however, model error dominates: predicted and observed native top-5% lists overlap only 48.5–52.6%.

## 3. Next experiments (prioritized)

1. **Replicate-anchored convention test.** In a screen with independent combination replicates (or the O'Neil within-block repeats), compare whether ε = 0 or ε = 0.01 references better predict held-out replicate excess at the affected near-plateau cases. This directly resolves U1.
2. **Population-scale fit uncertainty.** Repeat the native drc refit for all surfaces using R 3.6.3 (the environment must first be reconstructed) with replicate-grounded perturbations. Compare the fit-driven and edge-driven turnover of the same top-k lists.
3. **Version transport.** Score the same fixed fits with the SynergyFinder 2.4.16+ grid solver to show whether the edge effect persists in current software.

## 4. Frozen GPU input ablations

**Task accounting.** Array `132664` (45 sequential tasks on one A100-80GB) ran tasks 0–44 between 2026-11-07 21:57:30 and 2026-11-08 00:23 PST.

- 45/45 complete, 0 failed, 0 unstarted and 0 retries.
- Every saved config matches its frozen config hash (`run_ledger.csv`). Training took 59–509 s per task.
- The scheduler logger recorded exit code `0:0` for 39 tasks. It did not capture exit codes for tasks 0–4 and 44; RebelX has no accounting database for recovering them. Each of those six tasks has a complete metadata record and an end-of-run summary in the retained execution archive.
- Validation: each mask × seed has 1,475,328 out-of-fold rows in the frozen test order, with 0 duplicates, all values finite, and seed, fold and mask flags matching (`gpu/ablation_validation.csv`).

**Contrasts.** Relative MSE change versus full-input F2 (`benchmark_v1/f2_base`, same rows, folds and seeds), with errors averaged over seeds 2026–2028. A positive value means the masked model is worse. Brackets give the whole-pair bootstrap and delete-one-drug jackknife 95% intervals. Holm correction is across masks within each endpoint.

| Mask | Response | Pair-cell contrast (P−A) | Dose-shape contrast (E−P−D) |
|---|---|---|---|
| Drug identity removed | **+5.08%** [3.67, 6.60; 1.77, 8.39], Holm p = 0.004, material | −5.54% [−9.00, −1.86; −13.83, 2.75], p = 0.37, folds disagree | −1.45% [−2.51, −0.34; −3.62, 0.71], p = 0.33 |
| Cell identity removed | **+13.42%** [11.68, 15.21; 9.63, 17.21], p = 5e-8, material | +1.19% [−2.14, 4.89; −8.24, 10.62], p = 0.80 | +2.09% [1.06, 3.18; −0.52, 4.70], p = 0.33 |
| Dose removed | +4.47% [3.04, 5.94; 2.10, 6.84], p = 0.001 | +7.07% [3.37, 11.26; 0.68, 13.46], p = 0.093 | +2.02% [0.98, 3.17; −0.47, 4.50], p = 0.33 |

For scale, the existing controls give F0 versus F2 at +14.2% response and monotherapy-only versus F2 at +22.2% (`gpu/ablation_comparisons.csv`). The frozen protocol also lists a monotherapy-removal mask, but it is not among the 45 packet tasks; those two existing controls stand in for it. Figure: `figures/fig5_gpu_ablations.png`.

**Confirmatory-seed rule (Phase 3.4).** The coded trigger fired for four cells: drug-identity pair-cell contrast (one fold has the opposite sign), and cell-identity pair-cell contrast, cell-identity dose-shape contrast and dose-removed dose-shape contrast (a grouped bound within 0.5 pp of 5%).

**Update 2026-11-08:** seed 2029 was run afterwards (array `132713`, 20/20). The four triggered cells stay not material when pooled; seed 2029 alone makes the drug-identity pair-cell contrast material (−12.6%); the drug-identity response falls to +4.6% pooled, so it is no longer material. See `results/seed_validation/README.md`. The extra seed was not part of the original execution; the linked follow-up reports both the pooled and single-seed results.

## 5. Ranked follow-up questions

1. Which edge convention does SynergyFinder intend? Its documentation is silent, and its source accepts or rejects by termination code.
2. Do DrugComb releases after v1.4 reuse the 2.0.x Loewe path for O'Neil?
3. Is the near-plateau regime clinically meaningful at all, i.e. does any shortlisted combination owe its rank to a drug at its fitted maximum?
4. Why is NCATS ES replicate disagreement about 10× O'Neil's: plate layout, normalization or assay?
5. Should Loewe-based shortlists report a convention-robust core (units selected under both ε = 0 and ε = 0.01)?
6. Is the drug-identity response effect real? Run seed 2029 (and ideally 2030–2031) for `no_drug_identity`. The current classification depends on a point estimate 0.08 pp above threshold.
7. Does removing drug identity improve pair-cell contrast error reproducibly? The current estimate is −5.5% with disagreeing folds. If it does, identity embeddings may be overfitting pair-specific contrasts.

## 6. Task IDs, artifacts and reproduction

**Archives:**

| Archive | Bytes | SHA-256 |
|---|---|---|
| `research_return_20261107_final.tar.gz` (retained by the authors) | 1,011,650,207 | `2b75eeba78743ea15e6d72039df6b90e985ad49186192ebcd0584de174fc5ce4` |
| Transferred ablation subset `cluster_return/ablations/ablation_slim2.tar`: predictions, config, metadata, metrics and Slurm logs for all 45 runs, without checkpoints or `predictions_with_truth`; every member verified against `files_slim.sha256` | 123,320,320 | `6e770c2b3b1c57f31d4e6e8dd40a054aa509bde40e0f7285aa1f63f28d27adc0` |
| Early return (ALMANAC/NCATS runs, ALMANAC prepared data) | 777 MB | `5809c9ea76bf820aca3ddb4ca569ed081274b26f4a7403922de5ee95a7be8e17` |
| NCATS v2 prepared data | | `7e124d18de0b80ed1c4b5413cd57b008e2de5ce3e15567f19042afe788164ccd` |

The final archive was not downloaded. All analyses here use the verified subset.

**Code.** The original execution used source commits `20b78bb`, `3c397ec` and `a5adc3d` on top of `d2c8f41`. These identifiers record provenance; reproduction uses the code in this checkout.

**Historical validation.** The original execution passed 107 tests, with one CUDA-dependent skip.

**Reproduction.** The [reproduction record](../../manuscript/REPRODUCIBILITY.md) and Supplement S14 specify current commands and regenerated inputs. The original one-off analysis scripts retain their relative execution-input layout; they are not standalone commands for a fresh clone. `methods_and_provenance.md` records their input identities and original environment.

To re-run the ablations from this checkout, use `python -m synergy_artifacts.research_jobs --task TASK --output results/interaction_analysis/runs` inside a single-GPU allocation, with `TASK` from 0 to 44. Regenerate the matched full-input controls first. The environment workarounds recorded in `methods_and_provenance.md` describe the original scheduler packet.
