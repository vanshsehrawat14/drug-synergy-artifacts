# CS 622: mechanisms, information and independent-screen audit

Vansh Sehrawat (neural models and shared pipeline), Kingsley Ukandu (Gaussian process), Sharlene Kim (collaborative filtering). 8 November 2026.

This investigation starts at `a5adc3d`, following the November 8 overnight handoff. Historical experiments, authorship, thresholds and failed confirmations are retained. The new analyses are exploratory. The [analysis plan](../configs/mechanisms_20261108.json), [GPU protocols](../configs/gpu_mechanisms_20261108.json) and [support amendment](../configs/screen_support_amendment_20261108.json) record which observations preceded each test.

The strongest new findings are a native-source intervention that establishes solver-path dependence at fixed fits; an asymmetry between incoming and outgoing screening selections that was hidden by pooled margins; and a reference-augmented evaluation showing that neural and GP dose-shape prediction survives removal of much, but not all, reference-derived variation. Input-stratum standardization still does not explain the screen-performance gaps.

## 1. Provenance and design closure

The handoff SHA-256 is `f01719c7cead3c7fc06f08c169cd106b104ee4dd67ea696a5029de4cf746240d`. All 54 manifest entries and all 270 files in the returned ablation subset passed their hashes. All 45 frozen ablations remain complete; no new training result is reported here. See [verification](../results/mechanisms_20261108/input_verification.json).

Replaying the frozen seed-trigger code identifies exactly four result cells:

| Mask | Endpoint | Trigger evidence | Seed-2029 tasks |
|---|---|---|---|
| Drug identity removed | Pair-cell contrast | Fold 3 changes sign: +0.418%; the other four folds are negative | 5–9 |
| Cell identity removed | Pair-cell contrast | Pair-bootstrap upper bound +4.8934%, within 0.5 percentage points of 5% | 10–14 |
| Cell identity removed | Dose-shape contrast | Drug-jackknife upper bound +4.6956% | 10–14 |
| Dose removed | Dose-shape contrast | Drug-jackknife upper bound +4.5048% | 15–19 |

Tasks 0–4 supply the necessary **full-input seed-2029 baseline**. Four endpoint cells therefore require 20 unique runs, rather than 20 mask runs without matched controls. The code originally inferred sign disagreement from fold extrema; the complete five-fold table confirms exactly one opposing fold for the drug/pair-cell result. No threshold is changed and no extra trigger is invented for no-dose pair-cell error or drug-identity response.

The drug-identity response estimate remains **provisional**: +5.0826% response MSE on seeds 2026–2028. The five new no-drug runs also support a separately labeled exploratory response check. Sharing runs with a triggered contrast endpoint does not make the response endpoint preregistered. Report seed 2029 separately and then four-seed pooled errors, preserving every original seed. The [design audit](../results/mechanisms_20261108/seed2029_design_audit.json) and [execution manifest](../results/mechanisms_20261108/gpu_manifest.json) specify the comparison exactly.

## 2. Loewe prioritization: a stronger, narrower mechanism

The [numerical and prioritization audit](loewe_prioritization.md) gives the definitions, named selections and falsification controls. The primary correction still replaces **67/1,137 selections (5.8927%)**, pair-bootstrap 95% interval **[4.2216%, 7.3879%]**. Recovering all 15,620 certified fallback roots gives **79/1,137 (6.9481%)**, interval **[5.1891%, 8.7071%]**. The primary 14,197-root endpoint is preserved; the extra 1,423 remain a follow-up.

The pooled near-cutoff description is incomplete:

| Top-5% change | Count | Median native margin | Maximum native margin | Mean own score change |
|---|---:|---:|---:|---:|
| C1 entrants | 67 | 3.601 | 25.871 | +8.108 |
| C1 exits | 67 | 0.167 | 0.316 | +0.00034 |
| All-certified entrants | 79 | 4.231 | 35.396 | +9.352 |
| All-certified exits | 79 | 0.187 | 0.363 | +0.00128 |

Every C1 exit lies within 0.5 score points of the old cutoff, while **85.1% of entrants start more than one point below it**. Integer rounding has a similar *number* of changes, but shares only **3/67 incoming selections** with C1. This falsifies an explanation in which all incoming selections are merely near-tied integer-rounding substitutes. It does not establish their biological value.

The original R 3.6.3 / nleqslv 3.3.2 equations reproduce the original termination code on **400/400 status-stratified cases**, reference error at most **7.90×10⁻¹²**. Changing only Newton's starting point changes acceptance in **102 cases**, including **68 cases with certified roots**. In the 50 interior-fallback cases, a lower-bound-plus-0.1 start recovers 29 certified-root answers; an explicit boundary exclusion rule cannot explain these interior failures. These are controlled numerical interventions; the balanced strata are not a population sample.

Conversely, replacing accepted numerical answers by their certified roots changes pair-cell scores by at most **2.06×10⁻⁷** and changes no top-5% selection. The practical mechanism is **root recovery versus fallback**, rather than small error in already accepted roots. Uniform exclusion of roots close to an asymptote is a separate modeling convention. Most consequential recoveries are near asymptotes, so the biological interpretation remains convention-dependent.

Five historically stored brackets had been rounded so aggressively that their serialized endpoints were no longer strictly inside the effect interval. All 15,620 corrected fallback roots and 200 accepted controls were replayed with interval equations; re-certifying those five restores the brackets with **zero change to float reference values**. Recomputed score bounds guarantee all 79 augmented top-5% exits and entries, with no ambiguous memberships. The serialization repair and an exact-tie guarantee fix do not change any historical ranking.

## 3. What information do the inputs explain?

### Exact error attribution

Let (E) average identical physical inputs, (P) average pair-cell panels, (A) project onto endpoint-drug incidence within cell/fold, and (D) project onto panel-centered endpoint-drug/dose-rank incidence. With row weights retained, the mutually orthogonal error decomposition is

$$
\operatorname{MSE}(e)=N^{-1}\{\|Ae\|_W^2+\|(P-A)e\|_W^2+\|De\|_W^2+
\|(E-P-D)e\|_W^2+\|(I-E)e\|^2\}.
$$

This is an error identity, not a population variance decomposition. The population identity instead centers the additive panel term by the grand mean. Reference-relative reduction $1-\|e\|^2/\|y-r\|^2$ and contrast $R^2=1-\|Ce\|^2/\|Cy\|^2$ have different denominators. Subtracting the same reference from truth and prediction leaves the prediction error unchanged.

The new identity reproduces the original ablation contrasts. Summed error discrepancy is below $4.8\times10^{-7}$ across more than 1.4 million observations. The following values are **changes in MSE**, in inhibition-points², versus the same-seed production F2 baseline:

| Removed input | Additive panel | Pair-cell contrast | Additive dose | Dose-shape contrast | Total |
|---|---:|---:|---:|---:|---:|
| Drug identity | +3.821 | −1.197 | +5.692 | −0.609 | +7.707 |
| Cell identity | +10.819 | +0.257 | +8.392 | +0.875 | +20.343 |
| Dose | +0.097 | +1.528 | +4.310 | +0.845 | +6.779 |

Repeat-error changes are below $4\times10^{-10}$ MSE. **94.4%** of the cell-identity cost lies in the specified additive components. For drug identity, additive costs exceed the net response cost because contrast errors move in the opposite direction; this is why signed component shares may exceed 100%. The no-drug contrast changes do not satisfy the frozen superiority rule. No existing ablation establishes a material interaction-specific improvement, and the +5.1% response claim awaits the extra seed.

### A reference-derived contrast is not necessarily a combination effect

The original nuisance spaces do not remove multiplicative Bliss structure. For example, even a response exactly equal to (r_B=m_a+m_b-m_am_b/100) generally has nonzero (E-P-D) contrast. A synthetic falsification test demonstrates this directly.

For each original contrast $C\in\{P-A,E-P-D\}$, form $B_C=[Cr_B,Cr_H]$, where references use **training-pair monotherapy**. Within each cell/fold, let $Q_C$ be the weighted projector onto this span. The new evaluation target is

$$
C_*=(I-Q_C)C,\qquad R_*^2=1-\|C_*e\|_W^2/\|C_*y\|_W^2.
$$

The basis depends on references and input covariates, not test outcomes. Fitting it to the evaluated vectors defines an evaluation projection; it does not train a predictor on test labels. Both nuisance stages are refitted after each drug deletion. Bliss and HSA have exactly zero (R_*^2), an implementation control. This removes only their specified linear span, not every possible monotherapy-derived function.

The span removes **33.57% of original pair-cell contrast truth energy** and **63.46% of original dose-shape contrast truth energy**. Those are target-energy shares, not prediction-error reductions.

| Frozen model | Pair-cell reference-residual R² [drug interval] | Dose-shape reference-residual R² [drug interval] |
|---|---:|---:|
| Selected neural F2 | 0.056 [−0.044, 0.156] | **0.106 [0.058, 0.155]** |
| GP F2 | 0.025 [−0.018, 0.069] | **0.135 [0.091, 0.180]** |
| Selected CF | 0.064 [−0.031, 0.158] | **−0.218 [−0.369, −0.067]** |
| Production neural F2 | −0.056 [−0.159, 0.047] | 0.027 [−0.015, 0.070] |
| Monotherapy-only neural control | 0.003 [−0.003, 0.010] | 0.018 [0.010, 0.026] |

Neural and GP dose-shape prediction therefore does not disappear after these reference controls. CF's dose-shape skill on the original target does not survive this stronger target. No model's pair-cell interval excludes zero under drug deletion. These are exploratory intervals, not multiplicity-adjusted superiority declarations. Pair-bootstrap intervals condition on the fitted nuisance projections; drug intervals refit them. Predictors remain fixed, so neither captures all possible retraining uncertainty.

### Input provenance and comparison limits

All masks retain the same training-pair monotherapy lookup and identical test coverage. Removing a *direct channel* does not remove information implicit in that lookup: per-dose monotherapy values depend on drug, cell and dose. Thus no-dose F2 still receives dose-specific measurements, and no-cell/no-drug F2 still receives identity-specific measurements.

Masking also changes the network's parameter count: full 23,009; no drug 21,793; no cell 18,017; no dose 22,945. Seed matching does not give identical parameter initializations when shapes differ. The results characterize these frozen learning procedures and retained inputs, rather than an isolated biological information quantity or a capacity-invariant causal effect. The joint-identity and residual-objective follow-ups address defined predictive uncertainties; they do not resolve every initialization or capacity effect.

## 4. Independent-screen errors and support

All four screens have complete finite matched prediction coverage, valid positive combination doses and µM units. Configuration auditing confirms the same `f2_combined` architecture and seed 2026. These screen comparisons use that frozen model, not the selected neural model or the production model used in the input ablations. Responses are never pooled across screens.

| Screen | Rows | F2 RMSE | MSE / train-fold variance | Equal pair-cell MSE | Gain vs training-pair Bliss |
|---|---:|---:|---:|---:|---:|
| O'Neil | 1,475,328 | 11.805 | 0.1210 | 30.316 | 29.19% |
| ALMANAC | 2,871,834 | 9.523 | 0.1065 | 32.012 | 35.37% |
| NCATS ES | 112,165 | 22.665 | 0.2669 | 109.231 | 11.92% |
| NCATS DIPG | 224,708 | 14.352 | 0.1555 | 44.289 | 15.20% |

Training variance excludes test and validation folds. Baseline gains use ratios of summed errors on the same rows; model SSE and baseline SSE are both reweighted. Full mean/additive/Bliss denominators, MAE and monotherapy status counts are in [screen evidence](../results/mechanisms_20261108/screens/). Direct cross-screen transport still fails the identity/dose/assay compatibility gate; within-screen error normalization is not transfer validation.

Two corrections qualify the overnight results:

1. The original three-seed pair-cell calculation averaged predictions before squaring. It was an **ensemble** metric. The corrected seed-average pair-cell MSE is 31.65 rather than 28.61 for ALMANAC F2; 111.65 rather than 104.08 for ES; and 43.45 rather than 40.55 for DIPG. Seed-2026 primary estimates are unchanged. O'Neil `f2_combined` remains incomplete for the extra seeds and is reported unavailable, with no substitution.
2. Row effective sample size under stratum weights is $(\sum_i w_i)^2/\sum_i w_i^2$. The former 14–19 values summarized concentration of stratum counts. Correct row ESS is approximately **1,319,319**, **49,473** and **152,311** for ALMANAC, ES and DIPG; pair-weight ESS is 3,612, 1,030 and 1,562. These do not create that many independent experimental units.

All screens cover at least 99.89% of their rows on O'Neil's 52 input strata. Nevertheless, some strata contain only one or two pairs. Full-support pair bootstrap loses a required stratum in 175/2,000 ALMANAC draws, 719/2,000 ES draws and 1,084/2,000 DIPG draws. Percentiles of the remaining draws are **conditional**, not validly presented as unconditional full-support confidence intervals.

The pre-analysis support amendment therefore reports a separate sensitivity with at least ten pairs per common stratum in each screen. It retains at least 99.58% of O'Neil mass and 98.24% of other-screen mass, with all 2,000 pair-bootstrap draws valid:

| Screen | Raw common-support gain gap | Standardized gap | Reduction in absolute gap, pp [pair interval] |
|---|---:|---:|---:|
| ALMANAC | +6.50 pp | +11.89 pp | **−5.39 [−6.70, −4.12]** |
| NCATS ES | −16.98 pp | −18.12 pp | −1.14 [−2.40, +0.20] |
| NCATS DIPG | −14.01 pp | −16.67 pp | **−2.66 [−3.45, −1.84]** |

Negative reduction means a wider gap. A full-support pair Bayesian bootstrap gives the same qualitative result; its percentile intervals are labeled separately. The frozen explanation criterion is not met. The result rules out improvement from these particular strata; it does not identify assay heterogeneity, noise, or missing biology as the cause.

The NCATS H1 confirmation remains failed: interaction-residual R² 0.0571 with drug interval [−0.00593, 0.12013]. Later response-level gains and the exploratory residual targets above do not reopen that sequential confirmation. Its original scientific values are preserved in the historical-control table; the historical source is unchanged.

## 5. Next computation and evidentiary limits

The [GPU guide](mechanisms_gpu_runs.md) provides the exact 20-run seed closure and two 15-run follow-ups, hypotheses, controls, analysis commands and stopping rules. No GPU job was launched in this continuation. Existing controls were checked locally; missing or mismatched controls stop a campaign. The empirical training-time estimate is 2.57 GPU hours for all 50 tasks, excluding preparation, queue and evaluation overhead; the conservative 2× historical-maximum allowance is 14.2 GPU hours. Every task requests one GPU, four CPUs, 16 GiB and a 30-minute cap, with one active task.

The candidate methodological contribution is the version-specific, replayable chain from solver-path acceptance to named screening changes, together with diagnostics separating reference-derived prediction from remaining contrast skill. Root solving, projection theory, full response-surface prediction and the general sensitivity of synergy to curve fitting are established methods; no priority claim is established here. The previous report's [literature assessment](resolution_and_reliability.md) remains applicable.

The decisive biological test is an independently measured, prospectively fixed comparison of convention-discordant shortlists with plate-tagged monotherapy and combination repeats. Existing nominal repeats do not identify monotherapy uncertainty, and the 160-surface native-refit stress study is not a population uncertainty estimate. Additional GPU response training cannot by itself determine which Loewe convention improves experimental hit rate.

## Reproduction and validation

[Aggregate evidence](../results/mechanisms_20261108/), [four vector figure pages](figures/mechanisms_20261108/mechanisms_20261108.pdf), and their hash manifests accompany this report. Modules `certificate_audit`, `native_start_audit`, `prioritization_audit`, `information_audit`, `screen_audit` and `screen_support` expose explicit input/output arguments; fresh output paths are required. Native source interventions additionally require the pinned R runtime/source documented in the numerical audit. Figure rebuilding uses only shared aggregates:

```sh
python -m cs622.mechanism_figures
python -m pytest -q
```

The full suite passes **118 tests**, with one CUDA-dependent skip on the CPU host (62.49 seconds). An initial new test exposed list-input handling in a normalization helper; it was corrected before the passing suite. Certificate serialization, canonical ranking ties, weighted projection nesting, noninteracting-reference counterexamples, seed-versus-ensemble error, job coverage, both new training arms and return-archive integrity have dedicated tests. All three campaign control/dataset preflights also pass. No push, merge, publication or replacement of historical result files is part of this continuation.
