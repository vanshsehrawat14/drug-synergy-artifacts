# Preregistered overnight analyses, 2026-11-07

Written 2026-11-07 22:45 PST, before any new ablation metric, new ranking table, new certificate or cross-screen error table was computed or inspected. Base revision `d2c8f41` (`research/resolution-reliability`); new code lives on the local branch `research/overnight-2026-11-07`, created from `d2c8f41`. Already-known results that inform this registration: the 2026-11-07 research report, `results/research_20261107/**` aggregates, and file presence/exit status of ablation tasks 0–5. No ablation prediction or metric had been opened.

## 0. Frozen thresholds (copied, not to be changed)

From `configs/research_20261107.json` (SHA-256 recorded in `environment.txt`):

| Item | Frozen rule |
|---|---|
| Response equivalence | Both grouped 95% intervals (whole-pair bootstrap, delete-one-drug) entirely within ±5% relative MSE |
| Material superiority | ≥5% contrast-MSE reduction, Holm p<0.05, both grouped intervals exclude zero, same direction in all five folds |
| Material prioritization (Loewe) | ≥5% turnover of native top-5% pair-cell list and pair-bootstrap lower bound >0 |
| Useful error triage | ≥+5 pp error mass captured at 10% budget, both grouped intervals >0 |
| Precision attribution (`loewe_precision_amendment.json`) | Both HP roots satisfy equation, move <1e-3, double-precision accept/reject differs |
| Bootstrap | 2,000 draws, seed 20261107, unordered drug pair as resampling cluster |

## 1. Units and scoring levels (O'Neil, DrugComb v1.4)

- **Dose case** (`case_id`, 367,152): one unique positive-dose combination `(dose_a>0, dose_b>0)` on one fitted monotherapy surface. It is the unit at which a Loewe reference is solved. Observed responses at a case may repeat across blocks.
- **Surface** (`surface_id`, 22,947): one pair of fitted single-agent curves plus its dose grid. Several blocks may share a surface.
- **Block / full-grid screen unit** (`block_id`, 92,208): one DrugComb dose-response matrix (one ordered drug pair in one cell line, one replicate matrix). Block Loewe score = mean over its positive-dose cases of observed inhibition − reference, as in the native scorer; zero-dose rows/columns are not scored.
- **Pair-cell decision** (`pair_id` × `cell_line`, 22,737): unordered drug pair × cell line. Score = unweighted mean of block scores for that pair-cell. This is the primary screening decision unit.
- **Drug-pair aggregate** (`pair_id`, 583): unweighted mean of pair-cell scores across cell lines.
- Dose-case reference changes, block-score changes and pair-cell selection changes are reported separately and never equated.

## 2. Loewe score versions (identical cases, fits, inclusion and units)

| Version | Definition | Kind |
|---|---|---|
| N (native) | `native_sum_score`: SynergyFinder 2.0.x Loewe path, accept nleqslv code <3, summed-dose fallback, cap 100 | historical routine |
| C1 (primary correction) | N with the 14,197 independently verified admissible roots replacing fallbacks | numerical correction |
| C2 (augmented) | C1 plus the 1,423 high-precision recovered fallbacks | numerical correction (exploratory) |
| I (interval-certified) | N with every fallback whose root passes the new interval certificate (§3) replaced; uncertified cases keep N | numerical correction, rigorous |
| H (convention) | HSA fallback instead of summed-dose fallback | scoring-convention change, reported separately |
| P (published) | DrugComb `synergy_loewe` | reconstruction control |

No-root, opposite/flat and undefined cases keep their native convention value in every numerical version; they are never assigned a "true" Loewe value.

## 3. Root certificate (new, defined before running)

For a case with both arms LL.4 or L.4 (float coefficients treated as exact rationals), co-monotone and a non-empty open joint effect interval (ℓ,u):

1. g(y)=Σ_arm x_arm/D_arm(y) − 1 is strictly monotone on (ℓ,u) because each D_arm is strictly monotone with the same direction; uniqueness follows.
2. A **certified root** is an interval [y₀,y₁] ⊂ (ℓ,u), with ℓ<y₀ and y₁<u proved by interval arithmetic, and g(y₀), g(y₁) enclosed by mpmath interval arithmetic (≥60 digits) with strictly opposite signs. Reported: bracket endpoints, enclosures of g at both ends, width, distance to ℓ and u, precision.
3. **Certified non-existence**: same-sign enclosure of the finite end limit plus the known infinite limit at the other end, with interval arithmetic; otherwise **unresolved**.
4. Opposite/flat curves, invalid parameters or empty intervals are **undefined by definition**, not certified non-existence of a root of some other equation.
5. A high-precision point estimate with small residual is not a certificate. A case is certified only through step 2 or 3.
6. Cap at 100 and fallback conventions are unchanged.

Certified set: all 14,197 C1 roots, all 1,423 C2 roots, all 1,883 "accepted, no admissible root" cases, all 2,896 accepted near-edge roots, and every case on a surface contributing to a pair-cell whose membership changes at any budget under C1, C2 or I. Remaining accepted roots are certified if time allows (reported coverage either way).

## 4. Loewe regimes (mutually exclusive, first match wins)

1. `invalid_or_missing_curve` (missing/invalid model or coefficients)
2. `undefined_opposite_or_flat`
3. `undefined_empty_interval`
4. `native_accepted_certified_root` (native code 1–2 and native value inside the certified bracket after rounding to the cap)
5. `native_accepted_no_admissible_root` (native code 1–2, no certified root at that value)
6. `fallback_certified_unique_root` (native fallback, certified root, distance to boundary ≥0.01)
7. `fallback_boundary_certified_root` (native fallback, certified root within 0.01 of ℓ or u)
8. `fallback_certified_no_root`
9. `unresolved` (neither certified root nor certified non-existence)

Multiple roots cannot occur for co-monotone LL.4/L.4 (step 1). Zero-dose cases are not in the case table. Each correction and each changed decision is attributed to the regimes of its cases (share of absolute score change).

## 5. Decision sensitivity (primary paper analysis)

- Budgets k ∈ {1, 2, 5, 10, 20}% with n_k = ceil(N·k). Score direction: higher Loewe score = more synergistic = selected. Tie rule: descending score, ties broken by ascending canonical key (`pair_id, drug_a, drug_b, cell_line`; `pair_id` for pairs; `block_id` for blocks). Exact score ties at the cutoff are counted and reported.
- Comparisons: N→C1 (primary), N→C2, C1→C2, N→I, N→H (convention), P→N (reconstruction), plus coefficient rounding 6/8/10/12 significant digits under I.
- Per budget and level (pair-cell primary; block and drug-pair secondary): |A\B|/|A|, |AΔB|, Jaccard, Spearman and Kendall τ-b over all units, median and 95th percentile absolute rank movement (all units and A∪B), entered/exited counts, pre-correction distance from cutoff of changed units (score points and rank), within-cell top-5% turnover distribution.
- Conditional effects: changed decisions split by whether the unit contains certified-root corrections only, boundary corrections, or no correction (pure displacement).
- Near-tie analysis: for changed units, margin = |score_N − cutoff_N|; compare to |score change|; concentration by drug, cell line, fit quality (min distance to asymptote), dose range and regime.
- Guaranteed vs ambiguous: a membership change is **guaranteed** if it holds when every certified reference ranges over its certified bracket and every unresolved case ranges over {native fallback, solver value}; **ambiguous** otherwise. Report lower bound (guaranteed) and upper bound (guaranteed + ambiguous).
- Pair-clustered bootstrap (2,000 draws) for turnover and drug leave-one-out ranges. No row-level p-values.
- Reproduce 67/1,137 (5.89%) from row IDs and list them.
- All analyses here are **retrospective observed-response scoring**. A prospective model-predicted scoring analysis (fixed out-of-fold neural F2 predictions substituted for observed inhibition, references varied only) runs only if the case-to-prediction mapping is exact; otherwise it is reported as not done.

## 6. Frozen GPU ablations (45 tasks)

- Validation before any metric: 45/45 coverage of (mask, seed, fold); test row IDs identical to historical full-input F2 for the same fold; finite predictions; no duplicated row IDs; pair-disjoint train/validation/test; seed, mask and config hash match `cs622.research_jobs.job(task)`.
- Contrasts versus historical full-input F2 (same fold, seed, rows): relative change in dose-row MSE, pair-cell contrast MSE (P−A) and dose-shape contrast MSE (E−P−D), and residual skill against training-pair Bliss (1−‖e‖²/‖y−r_Bliss‖²). Synergy recovery: pair-cell mean Bliss excess, observed versus predicted, MSE and Pearson r.
- Seed-averaged errors as in the report. Inference: whole-pair bootstrap, delete-one-drug jackknife, five-fold direction consistency. Holm across the three masks within each endpoint. Descriptive classification uses the frozen materiality rule (≥5%, both intervals exclude zero, same direction in all folds). No new winner claims; the GP–neural equivalence result is untouched.
- Incomplete tasks are listed, never averaged as complete.
- Confirmatory seed rule (Phase 3.4): one extra seed (2029, five folds, unchanged config) for a mask only if its classification for an endpoint depends on one fold or a grouped bound lies within 0.5 pp of the 5% threshold. Otherwise no rerun.

## 7. Cross-screen standardization

- Screens: O'Neil (`oneil_v1`), ALMANAC (`almanac_v1`), NCATS ES (`ncats_es_v2`), NCATS DIPG (`ncats_dipg_v2`), each with its own saved out-of-fold predictions. No refitting, no pooling of endpoints.
- Per screen, native scale: RMSE and MAE (inhibition points), dose rows and pair-cell means.
- Normalized: MSE / training-fold target variance (variance computed on each fold's training rows only, averaged over test folds), and baseline-relative gain 1 − MSE_model/MSE_baseline for each saved baseline (`*_mean`, `*_additive`, `*_reference_bliss`), on identical rows.
- Primary standardized gap (frozen exp. 3): between-screen difference in F2-vs-Bliss relative gain before and after reweighting to common input strata: Bliss reference bins (<0, [0,20), [20,40), [40,60), [60,80), ≥80), log10 µM dose-sum bins (<−2, [−2,−1), [−1,0), [0,1), ≥1), monotherapy availability. Strata are defined from inputs only. Report common-support mass and effective sample size; NA when support is empty.
- Pair-clustered bootstrap intervals and drug leave-one-out sensitivity. The failed NCATS confirmation is reported unchanged.

## 8. Reliability

No new uncertainty fitting. `reliability_summary.csv` compiles the declared tests with their frozen utility rule. Dense dose rows are not treated as independent replicates.

## 9. Phase 3 confirmatory registrations

1. **Independent reference check**: fixed stratified set with seed 20261107: 200 cases per regime of §4 (all if fewer) plus every case on surfaces of pair-cells whose top-5% membership changes under C1. Solvers: (a) float64 Brent in effect coordinate (`scipy.optimize.brentq`, xtol 1e-14), (b) existing θ-bisection, (c) native value, (d) interval certificate. Attribute differences to domain, tolerance, fallback or fit.
2. **Perturbation stability**: unchanged curves/responses; coefficient rounding 6/8/10/12 digits; θ-bisection steps 40/60/80/200; float32 arithmetic; display rounding of final pair-cell scores to 2 and 1 decimals with the tie rule. Separate display rounding, arithmetic precision and stopping.
3. **Curve-fit uncertainty**: only with existing native refits; no new resampling units. Report the constraint.
4. **Confirmatory seed**: §6 rule only.

## 10. Stop rules

Ambiguous protocol, source mismatch, leakage, corrupted outputs or persistent deterministic failure stop the affected analysis; artifacts are preserved and the blocker is reported.
