# Discarded Loewe roots in DrugComb v1.4 ONEIL scores

**Summary.** DrugComb v1.4 Loewe scores come from SynergyFinder 2.0.3. Its Loewe scorer keeps a root only when `nleqslv` returns termination code 1 or 2. Otherwise it substitutes a summed-dose fallback.

In the ONEIL study, 14,197 dose cases fell back although the scorer's own equation has an admissible root there. These roots sit at the edge of the range where the equation is defined, a median 0.0002 inhibition points inside it; usually the edge is the weaker drug's plateau. Newton's method steps past the edge, where the equation is undefined, and stops with code 3.

The same near-edge situation is accepted in 2,896 other cases. Which way a case goes depends on the solver's path, not on the mathematics. We call this an acceptance-rule artifact, not a software defect.

Restoring the roots has three effects:
- 22.0% of block scores change and 1,998 change sign.
- On the affected units, an identity-based benchmark model scores R² 0.508 on published labels but 0.341 when trained and scored with every admissible root restored.
- Across the whole benchmark the effect is small.

**What this is not.**
- **Not a large benchmark effect.** Over all 22,737 pair-cell units, R² changes by 0.012.
- **Not the largest source of label variation.** Choosing the HSA fallback instead of the summed-dose fallback moves four times as many blocks.
- **Not robust to every convention.** A rule that treats roots within 0.01 points of the range edge as undefined would be a legitimate alternative; under it, 1,631 of the 14,197 cases remain.
- **Version-specific.** The inspected SynergyFinder 2.4.16 source replaces this solver with a response grid.
- **Not relevant to response-level models.**

## Software and data

**Data.** `drugcomb_data_v1.4.csv`, study ONEIL:
- 92,208 blocks covering 583 pairs and 39 cell lines;
- 367,152 unique dose cases on 22,947 fitted monotherapy surfaces.

**Which software produced the scores.**
- **DrugComb's statement.** DrugComb's release notes say that release 1.4 (Nov 2019) recalculated all scores with SynergyFinder 2.0.3 on Bioconductor 3.10 and R 3.6 [1].
- **SynergyFinder 2.0.3 itself.** It is commit `aa414fa` on the Bioconductor `RELEASE_3_10` branch (2019-11-12). Its DESCRIPTION reads `Version: 2.0.3` and imports `nleqslv(>= 3.0)` [2]. In `R/calculate_synergy_score.R`, the Loewe branch:
  - solves with `nleqslv::nleqslv(x, eq, method = "Newton", ...)`;
  - keeps the answer only `if (slv$termcd < 3)`;
  - otherwise takes `max(y.loewe1, y.loewe2)`, the larger single-drug curve at the summed dose;
  - sets the termination code to 999 on error;
  - caps the expectation at 100.
- **Neighbouring releases.** The same branch appears unchanged in 1.10.7 (`4d5ec7f`), 2.0.12 (`a09d792`) and 2.2.4. Release 1.8.0 accepted only code 1. From 2.4.16 (Bioconductor 3.12) the solver is replaced by a 100-step response grid.
- **nleqslv.** 3.3.2 was the CRAN release from 2018-05-17 to 2022-08-07 [3].
- **DrugComb's pipeline.** TidyComb 0.0.14 (`22d887c`) calls `Loewe(response.mat, drug.row.model, drug.col.model)`. That matches the 2.0.x signature.

**Native reproduction.** We ran the 2.0.12 Loewe code under R 3.6.3 with nleqslv 3.3.2 on cached curve coefficients. Its solver branch is identical to 2.0.3.
- It reproduces the published block Loewe scores: RMSE 0.70, r = 0.9995, 92.3% of blocks within 0.01 (n = 92,208).
- Accepting only code 1 fits worse (RMSE 0.74); an HSA fallback fits much worse (RMSE 15.5).

## Equation and procedure

Each drug's response is an LL.4 curve, f(x) = c + (d − c) / (1 + exp(b (ln x − ln e))). L.4 uses the same form on log dose. Its inverse is D(y) = e · ((d − y)/(y − c))^(1/b), defined only for y strictly between c and d.

For a dose pair (x₁, x₂), the Loewe expectation y solves

  x₁ / D₁(y) + x₂ / D₂(y) = 1.

For two increasing curves, y must lie in the joint range: above both lower asymptotes and below both plateaus. The upper end is the weaker drug's plateau; the lower end is the larger lower asymptote. Near either end one drug's equivalent dose runs to zero or infinity and the equation becomes very steep. Outside the range the equation is undefined: a negative base is raised to a fractional power.

The 2.0.3 scorer, for each dose pair:
1. starts Newton's method at the larger lower asymptote plus 1;
2. keeps the solver's value if the termination code is 1 or 2;
3. otherwise, including errors, uses the larger single-drug curve at the summed dose x₁ + x₂;
4. caps the expectation at 100.

The case score is observed inhibition minus the expectation. A block score averages its cases.

## Why admissible roots are discarded

**Admissible root.** A root counts only if the scorer's equation has residual below 1e-6 there and it lies strictly inside both effect ranges.

| Class, all 367,152 cases | Cases | Native outcome | Root to range edge, median | Within 0.01 of edge (plateau / lower) |
|---|---:|---|---:|---:|
| Accepted, matches the root | 180,364 | code 1–2 | 7.26 | 2,896 (1,868 / 1,028) |
| **Artifact: solver at the root, answer discarded** | **12,477** | code 3 | 0.00018 | 11,660 (10,381 / 1,279) |
| **Artifact: root exists, solver failed** | **1,720** | 253 code 3, 1,467 error | 0.0058 | 906 (531 / 375) |
| Boundary: root at an asymptote, residual ≥ 1e-6 | 1,423 | mostly code 3 | ~0 | 1,423 |
| Accepted, no admissible root | 1,883 | code 1–2 | — | — |
| No root, or opposite or flat curves | 169,285 | fallback | — | — |

**How the solver fails.**
- In 11,850 of the 12,477 "solver at the root" cases (95.0%), the native iterate lies just outside the joint range.
- There the equation returns NaN. Newton's line search finds no better point, so `nleqslv` returns code 3, "no better point found", although the iterate is within a median 0.009 points of the root.
- The acceptance rule reads only the termination code, never the residual or the domain, and sends these cases to the fallback.

**Inconsistent treatment.** Near-edge roots are not rejected as a class.
- Within 0.01 points of the edge, 2,896 roots are accepted because the solver happened to return code 1 or 2, and 12,566 are discarded.
- The scorer has no rule about the edge; the outcome follows the solver path. This is the sense in which the finding is an acceptance-rule artifact.

**Worked example** (native drc fit, case `37833:2:3`). Dasatinib 3.0 µM with MK-2206 0.9 µM in MSTO; dasatinib's plateau is 58.5475.
1. The Loewe root is y = 58.54747. Brent's method and the independent bisection agree to 1.7e-13, and the equation's residual there is 2.4e-9. The root is 1.9e-5 points below dasatinib's plateau. The equation's slope there is −13,234 per inhibition point.
2. `nleqslv` stopped at y = 58.5621, past the plateau, with code 3.
3. The scorer used the fallback instead: MK-2206's curve at 3.9 µM, 63.688. It matches the reference the scorer returned to 5e-10.
4. Restoring the root lowers the case's reference by 5.14 points, so its synergy score rises by 5.14. With the six corrected cases of the block, the block score moves from 30.25 to 30.64.

The figure `results/summary/figures/fig6_loewe_mechanism.png` shows this case and the distance of every root to the range edge.

**Root-level experiment.** The 98 probe cases are native drc fits under R 3.6.3, every one a fallback with an admissible root. `python -m cs622.loewe_example` recomputes each from the stored coefficients alone, without R:
- **Solvers.** The two independent solvers agree to 2.4e-13; the largest residual at a root is 5.9e-7.
- **Code-3 cases.** In 75 of the 82, the native iterate lies outside the joint range. The median root is 0.0003 points from the range edge. The median iterate is 0.008 points from the root.
- **Error cases.** Their roots are also near the edge (median 0.011).
- **Fallback recomputed.** It matches the returned reference to 1e-6 in 70 cases, and to within 0.56 in all. The residual difference comes from the stored coefficients, which are not byte-identical to the native fits.
- **Direction.** Fallback minus root has median +2.98 and maximum +92.7. It is positive in 94 of 98 cases, so the fallback mostly understates synergy.

## Effect on scores

All shifts are on native block scores, with references capped at 100.

| Correction | Blocks changed | Mean shift | ≥ 5 points | ≥ 10 points | Sign flips | Max |
|---|---:|---:|---:|---:|---:|---:|
| Solver at the root only (12,477 cases) | 19,308 (20.9%) | +3.21 | 4,240 | 1,764 | 1,903 | 38.9 |
| All admissible roots (14,197 cases) | 20,276 (22.0%) | +3.66 | 4,836 | 2,172 | 1,998 | 51.1 |
| Roots meeting nleqslv ftol 1e-8 (13,151 cases) | 20,248 (22.0%) | +3.39 | 4,508 | 1,888 | 1,835 | 51.1 |
| *Convention:* HSA instead of summed-dose fallback | 76,204 (82.6%) | +9.38 | 31,708 | 22,124 | 20,974 | 93.8 |

**Pair-cell panel.** 5,003 of 22,737 units change, and 1,196 of them by at least 5 points. The Spearman correlation with the published panel is 0.976.

**Artifact versus convention.** Conventions are legitimate choices made before scoring. Three are in play here:
- **What to use when no root exists.** Summed-dose or HSA fallback.
- **Whether near-edge roots count.** For example, roots within ε of the range edge could be called undefined. At ε = 0.01 that would leave 1,631 artifact cases and change 2,896 accepted ones.
- **Tolerances.** At nleqslv's own ftol 1e-8, 13,151 cases remain.

The artifact is different in kind. Under the rule the 2.0.x scorer states (root if it converges, fallback if no root), these cases have a root, and whether it is kept depends on solver path.

## Effect on a benchmark

We used an identity-only panel network with drug-pair folds, seed 2026. Retraining it on published labels reproduces the existing predictions exactly.

| Train on → score on | R², all units | R², 5,003 affected | Pearson, affected | Contrast R² |
|---|---:|---:|---:|---:|
| Published → published | 0.593 | 0.508 | 0.712 | 0.338 |
| Published → all admissible roots | 0.571 | 0.169 | 0.624 | 0.324 |
| All admissible roots → same labels | 0.581 | 0.341 | 0.632 | 0.320 |

- **Why the model benefits.** Whether a case falls back depends on the drug-cell monotherapy curves, so part of the artifact is predictable from drug identity. Published labels therefore overstate accuracy on the affected units.
- **Whole benchmark.** The change is small: R² falls by 0.012 and contrast R² by 0.019. These rows use `valid_root_any` (all 14,197 cases, including co-decreasing curves). The restricted `valid_root` sensitivity uses 13,888 co-increasing cases: affected-unit R² is 0.335, whole-benchmark R² falls by 0.014 and contrast R² by 0.022.
- **Response-level models.** They are unaffected. Models that predict inhibition and then score it share the label's reference, so the artifact cancels in their error.

## Verification

- **Independent solver.** It never inverts a curve: with θ = x₁/D₁(y), it solves f₁(x₁/θ) = f₂(x₂/(1 − θ)) by bisection on logit θ.
  - It agrees with Brent's method on inverses to 3.8e-13 on the 193,186 cases where both find a root.
  - The largest residual at an admissible root is 9.9e-7, and the median over the 14,197 artifact roots is 1.4e-12.
- **Reproducibility.** Re-running the verification from the saved case table reproduces all 292 fields of `loewe_verify.json` and a byte-identical `classes.csv`.
- **Native-fit probe.** A 200-block probe with native drc fits under R 3.6.3 confirmed all 98 fallback cases that have an admissible root: 82 ended with code 3 and 16 with an error.

## Scope and limits

- **Curve coefficients.** The 367,152 cases use cached Python re-fits of the drc models. Native fits were checked on the 200-block probe only.
- **Per-block settings.** DrugComb does not publish them, for example the noise seed or the imputation of missing cells.
- **Other studies.** Only ONEIL was audited. Other v1.4 studies went through the same scorer and were not checked.
- **Software versions.** The finding concerns published DrugComb v1.4 labels and SynergyFinder 1.10.7–2.2.4, including 2.0.3. It does not apply to the inspected grid-based 2.4.16 implementation.

## Reproduction

The upstream stage runs on `feat/cs622-neural-pipeline` with an R 3.6.3 / nleqslv 3.3.2 runtime and the synergyfinder 2.0.12 source:
- `experiments/scientific_validation/native_full.py`;
- `experiments/phase3/native_fit.py`;
- `experiments/phase2/panel_neural.py`; the retraining adds `--targets published solver_root valid_root valid_root_any`.

| Input | SHA-256 |
|---|---|
| `scientific_native_full/surface_cases.parquet` | `69e2c76476c8d14aa3c289b0a98ad6703b40b7eb9ad903f9e2df2cd1d4e23528` |
| `scientific_native_full/blocks.parquet` | `866ae92c09829e25dde86e139a83d6a9be11fe335b024b530141243d7f13f169` |
| `phase3_native_attribution_v3/strict_cases.parquet` | `6b1d0a7ecf2578cd0951905ca86f2e921ceb5e22271483c498cd93ffe93af31a` |
| `phase2_panel_neural/predictions.parquet` | `089766e461e991f15e524bad57f7384d66cf9049f8c1930f46a7e2b5660c42f5` |
| retrained panel predictions | `8f9f25eceabfdd7158d88d5c9a87cc2a3f468f7abe1cac2898d5f9ae8358ab46` |
| `loewe_audit/full_cases.parquet` | `14db04818f36abd0eb38adcea24a6ac5af68e65d98949703ee43042aaad5ba85` |
| `loewe_verify/cases_verified.parquet` | `384a2bd51b4c9ffeca7d4d1768237a6212bf5cf126b2c16a382af87d72acf438` |

Then, on this branch:

```sh
python -m cs622.loewe_audit --output results/loewe_audit
python -m cs622.loewe_verify --cases results/loewe_audit/full_cases.parquet --output results/loewe_verify --retrained <retrained panel predictions>
python -m cs622.loewe_example --output results/loewe_audit --cases results/loewe_audit/full_cases.parquet --verified results/loewe_verify/cases_verified.parquet
```

The verification plan was frozen in `configs/loewe_verification.json` before it ran. The worked example and the root-level experiment need only the tracked `results/loewe_audit/probe_98_cases.csv`.

## Before circulating outside the team

1. Native drc refits of all 92,208 blocks under R 3.6.3.
2. The same audit on the other v1.4 studies, ALMANAC first.
3. Send the case list to the DrugComb maintainers.

[1] DrugComb news, Release 1.4 (Nov 2019), https://drugcomb.fimm.fi/news
[2] Bioconductor synergyfinder, branch RELEASE_3_10, commit aa414fa86bd3060d73aaa6e2912e622ddf08421f, https://github.com/bioc/synergyfinder
[3] CRAN archive, https://cran.r-project.org/src/contrib/Archive/nleqslv/
