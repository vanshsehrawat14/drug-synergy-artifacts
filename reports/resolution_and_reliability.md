# Resolution and reliability

Vanshdeep Sehrawat (neural models and shared pipeline), Kingsley Ukandu (Gaussian process), Sharlene Kim (collaborative filtering). 7 November 2026.

The validated three-model milestone is `c56fda8`. This continuation uses its saved predictions and preserves its models, attribution and historical results. All follow-up analyses are exploratory. The failed NCATS confirmation remains failed. Protocols were recorded before the corresponding new endpoints: [main protocol](../configs/interaction_analysis.json), [precision amendment](../configs/loewe_precision_amendment.json), [dose-based calibration control](../configs/reliability_baseline_amendment.json). The amendments identify which earlier observations motivated them.

## 1. Prediction resolution

**Estimands.** Let $y\in\mathbb R^N$ be measured inhibition, $\hat y$ its out-of-fold prediction and $e=y-\hat y$. Define orthogonal projections in the dose-row inner product:

- $E$: means at identical drug-pair, cell and actual-dose inputs; $P$: pair-cell means; $J$: grand mean.
- $A$: additive endpoint-drug incidence on pair-cell means, fitted separately within cell and outer fold. Its column space contains the grand mean and lies inside $P$.
- $D$: endpoint drug-by-dose-rank incidence, centered within each panel and fitted within cell/fold. Its space lies inside $E-P$.

The five mutually orthogonal components give the exact identity

$$
\operatorname{Var}_N(y)=N^{-1}\{\|(A-J)y\|^2+\|(P-A)y\|^2+\|Dy\|^2
+\|(E-P-D)y\|^2+\|(I-E)y\|^2\}.
$$

The error identity replaces $y$ by $e$ and $A-J$ by $A$, retaining mean bias. It sums to MSE rather than centered error variance. The implementation collapses identical inputs and uses replicate-count weights; both identities agree to numerical precision. No physical-dose key is duplicated across collapsed units. Equal-panel sensitivity normalizes each panel's weights to one and refits every projection.

For a monotherapy reference $r$, the residual error is $(y-r)-(\hat y-r)=e$. Its relative reduction is $1-\|e\|^2/\|y-r\|^2$, **not** a share of population variance. For contrast $C=P-A$ or $C=E-P-D$, contrast $R^2=1-\|Ce\|^2/\|Cy\|^2$. Improvement beyond the reference instead uses denominator $\|C(y-r)\|^2$. These denominators answer different questions.

The projectors define evaluation contrasts, not predictors trained on test responses. Dose rank removes a specified additive nuisance space; it does not represent every continuous-dose effect. Remaining dose shape includes multiplicative Bliss effects and other nonlinear monotherapy structure, so neither contrast is a causal synergy parameter.

**Population decomposition.** The 1,475,328 rows contain 365,352 identical-input units and 22,737 pair-cell panels.

| Component | Population variance share |
|---|---:|
| Additive drug effects on panel means, within cell/fold | 36.66% |
| Remaining pair-cell contrast | 2.60% |
| Additive drug-by-dose-rank effects within panels | 45.12% |
| Remaining within-panel dose shape | 10.18% |
| Repeat deviations at identical inputs | 5.44% |

**Matched evaluation.** Neural metrics average seed-specific errors across three seeds; GP and CF have one saved prediction per row. F2 uses training-pair monotherapy, while F0 and CF omit monotherapy. CF's native information interface differs from F2, so cross-family results are not pure algorithm comparisons.

| Model | Dose-row MSE | Equal-panel mean MSE | Pair-cell contrast R² | Dose-shape contrast R² |
|---|---:|---:|---:|---:|
| Neural F2 | 140.29 | 30.80 | 0.356 | 0.672 |
| GP F2 | 138.07 | 31.57 | 0.333 | 0.683 |
| Neural F0 | 145.66 | 33.36 | 0.303 | 0.653 |
| GP F0 | 180.79 | 49.66 | 0.001 | 0.511 |
| CF selected extension | 163.41 | 31.92 | 0.335 | 0.508 |
| Bliss | 196.80 | 58.27 | 0.314 | 0.626 |

GP F2 versus neural F2 changes response MSE by −1.58%: whole-pair interval [−2.70%, −0.38%], drug-jackknife interval [−3.46%, 0.30%]. Both intervals lie inside the declared ±5% equivalence margin. This exploratory equivalence statement concerns the response endpoint only. The pair-cell contrast change is +3.59% [drug interval −5.59%, 12.77%; Holm p=0.738], while dose-shape contrast MSE improves by 3.22% [1.60%, 4.85%], below the 5% material-effect threshold.

CF selected versus neural F0 has 12.19% greater response MSE and 41.82% greater dose-contrast MSE, but a −4.69% pair-cell contrast difference whose drug interval crosses zero. GP F0 has 43.26% greater pair-cell contrast MSE than neural F0 [14.90%, 71.63%; Holm p=0.0114]. Equal-panel weighting preserves these conclusions. None of the three declared primary comparisons establishes superiority of the first-named method under the full decision rule.

The matched production-network F2/F0 comparison gives 12.44% lower response MSE and 15.57% lower pair-cell contrast MSE with monotherapy. Adding the fixed dose profile to original CF reduces response MSE by 77.26% while leaving pair-cell means and their contrasts unchanged. These controls demonstrate why resolution matters. Removing drug identity, cell identity and dose separately still requires the 45 declared GPU runs; those results are not yet available.

Neural F2 improves residual MSE by 28.71% against Bliss and 37.42% against HSA, but improves pair-cell contrast error by only 6.13% and 9.22%, respectively. The 6.13% estimate uses this continuation's replicate-aware weights; it does not replace the historical 6.4% result under its earlier aggregation. Pooled repeat-row dispersion is 83.34 inhibition-points² [pair bootstrap 80.94, 85.62], with 1,109,976 within-input degrees of freedom. Without independent plate/replicate provenance, this is not an identified biological-noise floor and cannot set monotherapy perturbation SD.

All comparison intervals cluster complete drug pairs. Drug jackknife refits the contrast projectors after each deletion. Contrast pair-bootstrap intervals condition on the fitted projectors. Dose rows are never treated as independent experimental units. Three primary panel-contrast tests receive Holm correction; other contrasts are exploratory sensitivities.

## 2. Loewe prioritization and numerical reliability

**Fixed fits.** Replacing only the original 14,197 rejected, independently certified roots changes 67 of 1,137 members of the native top-5% pair-cell list: **5.89% turnover**, whole-pair bootstrap [4.22%, 7.39%]. This meets the declared material-prioritization rule. There are 517 pair-cell sign changes and rank correlation 0.977. Top-1% and top-10% turnover are 8.33% and 5.72%. Median within-cell top-5% turnover is 3.33% (range 0–30%). The top-5% list of pair-averaged scores is unchanged, despite 17 pair-average sign changes. Consequences therefore depend on the prioritization unit.

Changing to HSA fallback produces 30.87% top-5% turnover; this is a **scoring-convention change**, not a numerical correction. Published-versus-native reconstruction changes only 0.09% of the top-5% list. The alternative acceptance table audits residual tolerances and recovery of historical fallbacks; it does not silently replace all historically accepted values.

**Boundary certification.** The follow-up solver uses an independent effect-coordinate bisection at 100 decimal digits. It converts the original float coefficients exactly, brackets within the common fitted effect interval, and requires strict interior membership and original inverse-equation residual below 10⁻³⁰. No opposite-direction or no-bracket case is certified by assumption.

All 2,579 boundary suspects and 100 seeded well-certified controls pass before and after rounding coefficients to ten significant digits. This certifies **1,423 additional fallback cases**, beyond the historical 14,197. Maximum root movement is 3.87 × 10⁻⁸ inhibition points; 195 double-precision acceptance decisions nevertheless flip. High-precision certification reduces the largest rounding-induced pair-cell score change from 7.96 points to 3.87 × 10⁻⁹, with no ranking or sign changes under that perturbation. Thus the observed discontinuity is in certification/fallback selection, not root movement. This is an exploratory, trigger-driven follow-up.

Adding the recovered cases changes 13 further top-list memberships relative to the original correction. Relative to native scores, 79 of 1,137 choices change (6.95%), with 586 pair-cell sign changes. The original 14,197-case endpoint remains the primary result; the new certificate extends rather than rewrites it. These are numerical-validity findings, not evidence of improved biological hit rate or a claim about every current SynergyFinder release.

**Native refits.** The pinned historical source, R 3.6.3, drc 3.0.1 and nleqslv 3.3.2 produced 6,720 successful curve fits and 53,760 dose cases: 160 status-stratified surfaces, an unchanged baseline and five paired draws at SD 0.1, 0.5, 1 and 2 inhibition points. Shared drug/cell/dose edges receive shared perturbations. Native reference values exactly reproduce the official scorer on these inputs.

| Perturbation SD | Historical top-5% turnover | Valid-root recovery turnover |
|---|---:|---:|
| 0.1 | 7.5% | 0.0% |
| 0.5 | 15.0% | 12.5% |
| 1.0 | 27.5% | 20.0% |
| 2.0 | 22.5% | 20.0% |

Entries average five draws; each list has only eight sampled surfaces. This stratified stress experiment is not a population turnover estimate. Unchanged-data refits retain the sample's original top-eight list, but individual panel scores can differ by 4.99 points from cached fits. Historical endpoint ordering is explicitly aligned before comparing coefficients. Fit changes and fixed-fit solver changes are reported separately. Perturbation SDs are sensitivity scales: independent monotherapy repeat uncertainty remains unmeasured. Neither recovery nor a smoother convention resolves that experimental uncertainty.

## 3. Cross-screen compatibility

Exact positive PubChem CIDs and Cellosaurus accessions identify 14 common drugs and nine common cells between O'Neil and ALMANAC. Source-supported endpoint concentration ranges retain only **1,999/2,871,834 ALMANAC rows (0.0696%)** when O'Neil is the source; the reverse direction retains 22,264/1,475,328 O'Neil rows (1.509%). These are range-support bounds, not proof of complete surface overlap. Neither NCATS screen shares a cell accession with either of these screens; ES and DIPG also have no common cell accession. Exact names alone overstate O'Neil–ES drug overlap (six names versus five CIDs). DIPG contains a drug-name/CID ambiguity, retained in the identifier audit.

Full-screen exact-dose monotherapy availability is 100% in O'Neil, ALMANAC and ES but only 24.23% in DIPG. Training-pair availability remains a separate constraint, and interpolation is not an observed measurement. Mean inhibition also differs substantially: 46.53, 20.19, 45.96 and 31.64 points, respectively. These measured differences support a compatibility concern; they do not quantify how much of the model-performance gap they explain.

DrugComb's common inhibition field and µM units do not establish equivalent assays, exposure times or normalization. ALMANAC's underlying protocols vary; a contributing primary study describes a 72-hour CellTiter-Glo implementation ([methods paper](https://doi.org/10.1177/2472555218812429)). Block-level correspondence to an assay-compatible O'Neil subset has not been established here. Direct identity-model transport therefore fails the declared compatibility gate. No new screen was sought.

The primary standardized error-gap analysis remains pending recovery of existing ALMANAC/NCATS row predictions. Published aggregate relative gains cannot be pooled without their squared-error denominators. Their prior failed NCATS confirmation is preserved without reinterpretation.

## 4. Predictive reliability

The three-seed neural ensemble has pair-cell mean MSE 26.69, distinct from the historical seed-average MSE 30.80. Checkpoint restoration was verified against saved test predictions before generating validation predictions. Calibration, scale fitting and simple-baseline selection use validation data only. Because validation previously selected models/checkpoints, this is development calibration, not untouched conformal coverage.

| Uncertainty | Error mass captured in top 10% | Gain over validation-selected simple baseline | Drug-jackknife interval for gain |
|---|---:|---:|---:|
| GP predictive variance | 25.66% | −7.37 pp | [−15.60, 0.87] pp |
| GP latent variance | 35.19% | +2.17 pp | [−4.97, 9.31] pp |
| Neural ensemble disagreement | 34.95% | +6.51 pp | [−0.59, 13.60] pp |

The simple candidates are global, cell-specific and cell/Bliss-bin validation errors. A predeclared follow-up adds ridge log-error calibration using cell, dose and Bliss; validation still selects the original baseline in every fold. The neural gain has a positive pair-bootstrap interval [2.06, 10.96] pp but a drug interval crossing zero. **None meets the declared utility rule** requiring at least +5 pp and both grouped intervals above zero.

Raw neural disagreement covers only 39.47% of observed dose responses with nominal 90% intervals; validation-fitted noise floor and scale increase coverage to 92.58%. GP predictive coverage is 92.08%, versus 78.56% for latent uncertainty alone. The cell/Bliss baseline has lower descriptive Gaussian NLL than model-specific recalibration for both models. Calibration does not establish useful error ranking. GP panel ranking uses mean marginal variance; no panel prediction interval is constructed by assuming dose independence. Error detection is a measurement-planning surrogate, not demonstrated biological discovery utility.

## 5. Novelty assessment and decisive next work

Full dose-response prediction is established in [comboFM](https://doi.org/10.1038/s41467-020-19950-z) and [DeepSynBa](https://pmc.ncbi.nlm.nih.gov/articles/PMC13341016/). Uncertainty-aware synergy estimation is established in [SynBa](https://pmc.ncbi.nlm.nih.gov/articles/PMC10311304/), and experimental sequential selection in [RECOVER](https://arxiv.org/abs/2202.04202). [Cheng et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC13587485/) already benchmark cross-screen heterogeneity, while explicitly leaving dose-range differences and cross-dataset generalization unresolved. These broader themes are not new claims of this project.

The strongest candidate contribution is the specific, independently certified chain from historical acceptance rules and floating-point boundary certification to drug-combination top-list changes, separated from native-refit uncertainty and scoring conventions. [Mind the Curve](https://pmc.ncbi.nlm.nih.gov/articles/PMC10253300/) already establishes that fitted curves can change synergy classifications across software; the present fixed-coefficient certificate discontinuity and benchmark prioritization consequences are narrower. The literature review found related work, not a guarantee of priority. The orthogonal multi-resolution comparison is an empirical diagnostic contribution, not new projection theory.

The next decisive computations are the frozen 45 GPU input ablations and common-stratum error standardization from existing screen predictions. The next decisive experiment is a prospectively specified comparison of historical versus corrected shortlist performance, with aligned dose grids and independent monotherapy/combination plate replicates. That would distinguish numerical improvement from actual selection utility and supply defensible perturbation magnitudes. A fixed-fit certificate should also be audited on historically accepted, unresolved cases before proposing a production scoring replacement.

## Evidence and execution

[Aggregate evidence](../results/interaction_analysis/) is retained. Source predictions and native raw outputs remain untracked. Numerical verification uses the `research` optional dependencies. Native-input generation is `python -m synergy_artifacts.native_reliability --prepare --directory OUTPUT --sources PINNED_SOURCE`; run `experiments/research_native_fit.R` on those inputs in the pinned R environment, then analyze with the same module without `--prepare`.

GPU results and standardized cross-screen performance differences are pending, not negative findings.
