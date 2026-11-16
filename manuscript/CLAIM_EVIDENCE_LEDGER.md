# Claim–evidence ledger

Quantitative claims in the manuscript and supplement trace to the artifacts below. `scripts/check_numbers.py` retains 116 original and 16 DeepSynergy checks and adds 15 checks for editorially clarified populations, energy normalization, the worked contrast and paired uncertainty. Validation is recorded in `REPRODUCIBILITY.md`.

Status labels:
- **verified:** recomputed independently;
- **narrowed:** restated to what the evidence supports;
- **corrected:** an earlier reported value was changed;
- **secondary:** taken from a committed report whose own artifact was not re-derived here.

## Central claims

| # | Claim (manuscript section) | Evidence | Status |
|---|---|---|---|
| 1 | Best response models reach $R^2$ 0.88; selected F2 RMSE 11.84 (§5) | `results/compare/selected_models.csv` | verified |
| 2 | GP-F2 meets the ±5% response-equivalence margin against selected F2: −1.6% MSE, −2.22 [−3.93, −0.46; drug −4.83, 0.40] (§5) | `results/compare/comparisons.csv` | verified |
| 3 | Additive structure 81.8%, pair-cell contrast 2.6%, dose-shape contrast 10.2%, replicate deviations 5.4% of row variance (§5) | `interaction_analysis/resolution/variance.json` | verified |
| 4 | Held-out-pair models explain at most 0.07 of the noise-corrected pair-cell contrast (§5) | `audit/outputs/interaction_audit.json` | verified |
| 5 | Within-batch replicate noise 9.8% of contrast energy; ceiling 0.90 (§6.1) | `interaction_audit.json`, `results/interaction_structure/variance_components.json` | verified, narrowed to "within batch, conditional on shared single-agent values" |
| 6 | Dose-shape noise share 36% (§6.1) | `interaction_audit.json`, `dose_shape_check.json` | corrected from 45% |
| 7 | Cross-batch: 88% [65, 101] of within-batch reproducible energy replicates; full-screen extrapolation 0.67–0.82 (§6.1) | `variance_components.json` (cross_batch) | verified; the extrapolation states both assumptions |
| 8 | Reference noise is not independently estimable (§6.1) | `variance_components.json` (audit.monotherapy); Proposition 5 | verified |
| 9 | Decomposition: pair-wide 0.242, pair-by-cell 0.660, noise 0.098 (§6.2) | `variance_components.json` | verified |
| 10 | Bliss-residual pair means have correlation 0.009 with reference-only pair means and 0.81 with measured-response pair means (§6.2) | `variance_components.json` | verified; no causal origin inferred from correlation |
| 11 | Leak-free matched: B1 0.177, B2 0.216, B3 0.218, held-out F2 −0.050, with paired intervals (§6.3, Table 3) | `results/interaction_structure/b3_paired_summary.json` (computed on RebelX from 15 hash-verified prediction files); `anchor_experiment.json` | verified |
| 12 | B1 is 81% [69, 97] of B3; B3 − B2 0.003 [−0.031, 0.036]; B3 − B1 not material (drug interval includes 0) (§6.3) | `b3_paired_summary.json` | verified |
| 13 | Matched oracle 0.205 was leaky; leak-free 0.177 (§6.3) | `anchor_experiment.json`; leak attribution (supplement Table S6) | corrected |
| 14 | Transfer grows with anchors (0.033 → 0.086 → 0.144); same-pair and same-cell specific (§6.3) | `anchor_experiment.json`, `variance_components.json` | verified |
| 15 | 54.3% of total contrast energy remains within-batch reproducible but unpredicted by the evaluated leave-one-cell-out B1/B2 procedures; 60.2% of estimated reproducible energy (§6.3) | 1 − 0.098 − 0.359 (`variance_components.json`) | verified, narrowed ("under the evaluated procedures") |
| 16 | Cross-cell structure barely associated with transcriptome-wide similarity (Mantel $\rho$ 0.069, $p$ 0.08; expression-only predictor 1% of B2 gain); other cell-line representations untested (§6.3) | `grounding.json` | verified, narrowed; the origin is stated as not identifiable |
| 17 | Loewe labels learnable in a zero-interaction screen (0.49–0.52); Bliss not; $M$ = 0.772 (§7) | `audit/outputs/gate2_reproduction.json` (full re-execution, max difference 0) | verified; the construction and limits are stated |
| 18 | Existence criterion, Proposition 1 (§8) | proof (S9); `loewe_audit.json`: 310,928/310,928 | verified |
| 19 | Continuity and jump at the boundary, Proposition 2 (§8) | proof (S9); jump positive in 58% of the near-plateau subset | verified, narrowed (the jump is guaranteed only at the boundary) |
| 20 | Boundary-layer law, Proposition 3 (§8) | proof (S9); 13,277 roots, $r$ 0.998 | verified |
| 21 | Counts: C1 14,197; C2 1,423; total 15,620; 15,588/15,588 bracket replays (§8.3) | `loewe_audit.json` | verified; populations reconciled in the disjoint population table |
| 22 | Fallback error medians: summed-dose 4.17, HSA 0.0040 (§8.3) | `loewe_audit.json` | corrected (0.0044 → 0.0040) |
| 23 | C1 replaces 67/1,137 top-5% candidates (5.89%); systematic under resampling (§9) | `audit/outputs/screening_audit.json` | verified |
| 24 | Native reconstruction differs from published DrugComb top-5% by one candidate (§9) | `screening_audit.json` (P_to_N_turnover 0.00088 = 1/1,137) | verified |
| 25 | Negative results (§10) | `failed_ncats_confirmation.json`; `pair_information/claim_ledger.json`; `results_summary.md` | verified (NCATS, drug identity); secondary (identity contrast −7.41%, uncertainty utility) |

## Corrections relative to earlier project reports

| Earlier statement | Corrected statement | Reason |
|---|---|---|
| "About 90% of the interaction contrast is reproducible" | 90% within batch, conditional on shared single-agent values; cross-batch plausibly 0.67–0.82 | second-batch re-measurement; shared references |
| Matched cross-cell oracle 0.205 | 0.177 (leak-free) | drug effects in the frozen contrast were fitted with test entries |
| Dose-shape noise share 45% | 36% | per-pair-cell sign split unbalanced within units of dual-grid pair-cells |
| Pair-cell noise share 10.1% [9.2, 11.1] | 9.8% [8.9, 10.8] | same split correction |
| HSA fallback median error 0.0044 | 0.0040 | recomputation on the 14,197 C1 roots |
| "The summed-dose fallback always jumps" | Jumps at the existence boundary; positive in 58% of near-plateau rejected roots | saturated stronger arms give $J\approx0$ |
| "Generalized Loewe" | "continuous extension (equal to HSA in the no-root region)" | name conflicts with BIGL |
| "The interaction wall is an information limit" | Not predicted by any evaluated procedure; no claim of impossibility | evidence concerns these inputs and models |
| "Pair-seen network captures the oracle (1.05 of the gain)" | Pair offset 81% of network $R^2$; B3 − B2 unresolved: 0.003, pair [−0.031, 0.036], drug [−0.079, 0.084] | leak-free matched comparison with paired intervals |
| Existence criterion on 309,056 cases | 310,928 (co-increasing and co-decreasing) | co-decreasing cases now checked |
| Cross-study Pearson 0.09–0.25 | removed | not verifiable in the cited source |
| DrugComb v1.4 used SynergyFinder 2.0.3 (release note) | Findings concern our reconstruction: the 2.0.12 Loewe code (solver branch identical to 2.0.3) under R 3.6.3 and nleqslv 3.3.2, matching published v1.4 Loewe scores (RMSE 0.704, $r$ 0.9990); the release DrugComb ran is not established | release note could not be retrieved; the reconstruction, not the historical release, was verified |
| "The gap is not a property of model capacity" | The same base architecture improves with same-pair supervision; other capacity and representation limitations remain possible | larger or differently structured models were not evaluated |
| "Unrelated to transcriptome-wide similarity" (abstract) | Barely associated with transcriptome-wide expression similarity; an expression-only predictor recovers 1%; other representations untested | one representation tested; Mantel $\rho$ 0.069, $p$ 0.08 |
| DeepSynergy "contrast $R^2$ 0.283, close to Bliss 0.315" beside the response table's −0.047 and 0 | Named as the measured-response contrast ($E_u=0$); The response table's column is the same comparison on the Bliss pair-cell contrast, $(0.283-0.315)/(1-0.315)$ | two targets were reported under one name |
| "Every within-unit replicate statistic has the same distribution under all conventions" (Proposition 5) | Statistics that depend on a unit's block scores through their differences are unchanged; unit means change with the reference | the proof gives the differences only |
| 79.8% of dose cases use fallback | 79.8% of 22,737 pair-cells contain a block flagged by the historical existence-rule emulation; native fallback is separately 184,905/367,152 cases (50.4%) | `r18_gate2_analysis.py` groups the block indicator by pair-cell; native counts use termination codes |
| 54.3% of reproducible energy remains unresolved | 54.3% of total energy, or 60.2% of estimated within-batch reproducible energy | divide by 1 − 0.0978041 after subtracting B2 R² 0.3594570 |
| Zero-interaction dose blocks rescored by the native pipeline | Pair-cell means simulated as Bliss plus noise; both references held fixed | frozen Gate 2 P4 code, seed 2; no synthetic curve fitting or solver rerun |
| No fitted parameters in the reference difference | No parameters fitted to combination outcomes; single-agent curves are fitted | reference construction |
