# Loewe prioritization and numerical-certification audit

Vansh Sehrawat, Kingsley Ukandu, Sharlene Kim. 8 November 2026. Exploratory continuation from `a5adc3d`; fixed observed O'Neil responses and historical single-agent fits.

## Estimand and solver definition

For arm $i$, $f_i(x)=c_i+(d_i-c_i)/(1+\exp[b_i(\log x-\eta_i)])$, with $\eta_i=\log e_i$ for LL.4 and $\eta_i=e_i$ for L.4. The inverse is

$$
\log D_i(y)=\eta_i+b_i^{-1}\log[(d_i-y)/(y-c_i)],\qquad
g(y)=x_a/D_a(y)+x_b/D_b(y)-1.
$$

Co-monotone, non-flat curves define an open common effect interval $(\ell,u)$. Each inverse is strictly monotone with the same orientation, so $g$ is strictly monotone. Opposite signs at strict-interior bracket endpoints prove one unique root. A strictly positive finite-end limit, together with the known infinite limit at the other end, proves no root. Unsupported curves, opposite/flat directions and empty common intervals are separate statuses. These definitions concern the registered co-monotone Loewe reference, not every algebraic extension of the equations.

Coefficients and doses are their original binary floating-point values treated as exact rationals. The inhibition cap at 100, native fallback formula, case inclusion and averaging are unchanged in each controlled numerical comparison. This isolates solver acceptance from refitting and scoring-convention changes.

There are 367,152 dose cases, 22,947 fitted surfaces, 92,208 blocks, 22,737 unordered pair-cell decisions and 583 pair aggregates. Case synergy is observed inhibition minus reference; blocks average positive-dose case scores; pair-cells average blocks; pairs average pair-cells. Decisions rank descending scores with canonical-key tie breaking. Dose observations are never independent uncertainty units.

## Reference versions and attribution

| Version | Changes from native | Interpretation |
|---|---|---|
| N | None | Historical Newton/termination-code routine |
| C1 | Recover 14,197 previously verified admissible fallback roots | Frozen primary numerical endpoint |
| I | Recover all 15,620 interval-certified fallback roots | Secondary extension, including 1,423 additional roots |
| R | I, plus replace accepted answers by certified root values | Pure accepted-root accuracy control |
| Edge ε | Use every root at least ε from an asymptote, otherwise summed-dose fallback | Explicit convention, applied to accepted and fallback roots |
| H | HSA instead of summed-dose fallback | Different scoring convention |
| Undefined → fallback | Replace the 1,883 accepted values outside the registered co-monotone definition | Undefined-reference convention sensitivity |
| P | Published DrugComb score | Source-reconstruction control |

The numerical corrections preserve all 117,824 certified no-root cases and 53,344 cases outside the registered definition. The 1,883 originally accepted opposite/flat cases are not silently assigned certified roots. The undefined-reference intervention changes 13/1,137 native top-5% selections (1.14%) and is kept separate.

Among fallback roots, 13,989 lie within 0.01 points of an asymptote and 1,631 lie farther away. The near-boundary subset accounts for approximately 97.8% of absolute score movement among the augmented top-5% changed units. Near-boundary location does not imply that the fallback and root give similar reference values.

## Actual selections and threshold margins

The [changed-selection table](../results/mechanism_analysis/prioritization/changed_selections.csv) lists canonical pair/cell identifiers, before/after score, rank, both cutoff margins, entry/exit and boundary/interior score contributions for every comparison and budget. The primary C1 result contains 67 incoming and 67 outgoing units, or 67 replaced members of a 1,137-member list.

Examples among C1 entrants, all selected retrospectively:

| Drug pair | Cell | N | C1 | Score increase | Native distance below cutoff |
|---|---|---:|---:|---:|---:|
| Sorafenib + temozolomide | SKMES1 | −14.785 | 12.299 | 27.084 | 25.871 |
| AZD1775 + dasatinib | A2780 | 2.183 | 24.098 | 21.915 | 8.903 |
| Bortezomib + paclitaxel | COLO320DM | −7.949 | 12.491 | 20.439 | 19.034 |
| Bortezomib + oxaliplatin | MSTO | 0.300 | 20.607 | 20.307 | 10.786 |
| Cyclophosphamide + sunitinib | NCIH2122 | −3.552 | 15.092 | 18.644 | 14.638 |

These names identify the changed screening decisions; they are not recommendations for experimental or clinical use. The fitted curves can extrapolate poorly, and the appropriate reference convention remains unresolved.

Incoming C1 selections have median native margin 3.601 points, 90th percentile 11.069 and maximum 25.871; outgoing selections have median 0.167 and maximum 0.316. Only 10/67 entrants are within one point of the cutoff, versus every exit. With the augmented correction, incoming median margin rises to 4.231 and maximum to 35.396. The pooled median of changed units largely describes displaced exits and obscures the incoming-score mechanism.

Integer rounding alone replaces 57 native top-5% choices. Its incoming set overlaps C1 in just **three units**, and overlaps the augmented incoming set in the same three. Equal turnover percentages therefore do not establish that the correction is equivalent to rounding, even though near-tied exits are expected when any ranking moves. [Rounding overlap](../results/mechanism_analysis/prioritization/rounding_overlap.csv) and the [margin figure](figures/mechanism_analysis/selection_margins.pdf) make this distinction explicit.

## Stability across budgets and clusters

| Budget | Selections | N → C1 turnover | N → I turnover | N → I pair-bootstrap interval |
|---|---:|---:|---:|---:|
| 1% | 228 | 8.33% | 9.21% | [5.70%, 13.60%] |
| 2% | 455 | 7.25% | 7.91% | [4.84%, 10.33%] |
| 5% | 1,137 | 5.89% | 6.95% | [5.19%, 8.71%] |
| 10% | 2,274 | 5.72% | 6.73% | [5.54%, 7.87%] |
| 20% | 4,548 | 5.23% | 5.91% | [5.23%, 6.77%] |

Bootstrap resamples whole drug pairs, retaining all their cell decisions and reselecting each draw's lists. At top 5%, deleting any one drug leaves C1 turnover in [5.26%, 7.14%] and I turnover in [6.20%, 8.26%]. Deleting any one cell leaves C1 in [5.32%, 6.23%] and I in [6.23%, 7.31%]. These are sensitivity ranges, not confidence intervals. Within-cell lists and budgets are retained in [within-cell evidence](../results/mechanism_analysis/prioritization/within_cell.csv); small lists have coarse, heterogeneous turnover.

Changing the unit matters: the historical top-5% list of 30 drug-pair averages does not change under C1, even though pair-cell choices do. Neither pair bootstrap nor drug/cell deletion supplies independent biological replication. The effect is a finite-screen ranking consequence under a stated convention.

A descriptive convention-robust core contains **1,045/1,137 native choices (91.9%)** selected under native, strict-interior and uniform ε=0.01 rules. Including HSA fallback reduces the common core to **771/1,137 (67.8%)**. This describes agreement among named rules; it does not identify biological truth or a uniformly optimal shortlist.

## Native-source reconstruction and intervention

The pinned `calculate_synergy_score.R` is SynergyFinder 2.0.12, SHA-256 `ebc914c816ce543c382678d047f315181545fc1f58d43b9084934369ce9aebe7`. The registered path is fitted monotherapy → LL.4/L.4 inverse equation → `nleqslv(method="Newton")` started at `max(c_a,c_b)+1` → accept `termcd < 3`, otherwise summed-dose fallback → cap 100 → dose/block/pair-cell averaging.

The package documentation distinguishes function convergence from step convergence and warns that termination code 2 needs a separate residual check. A termination code is not itself a mathematical root certificate. [Official nleqslv documentation](https://search.r-project.org/CRAN/refmans/nleqslv/help/nleqslv.html)

The existing full-screen reconstruction checks are preserved: summed-dose fallback error ≤1.2×10⁻¹¹; native block reconstruction error ≤1.5×10⁻¹⁴; 20,000 arm swaps with no status mismatch; and 271,240 dose-adjacency checks with no monotonicity violation. Source hashes and the unchanged historical artifacts substantiate what was reconstructed, without making claims about later software versions.

The new experiment samples 50 cases in each of eight observed status strata, using seed 20261108 before interventions. It sources the original equations and fallback function; no surrogate inverse is substituted for the native call. R 3.6.3 and nleqslv 3.3.2 are enforced. Changing only the starting point gives:

| Original status, n=50 each | Lower +0.1 acceptance changes | Lower +5 | Midpoint |
|---|---:|---:|---:|
| Accepted boundary root | 10 | 20 | 12 |
| Accepted interior root | 0 | 0 | 0 |
| Fallback boundary root | 5 | 3 | 6 |
| Fallback interior root | 29 | 12 | 25 |
| Accepted opposite/flat | 7 | 33 | 2 |

The other three fallback strata—certified no root, empty interval and opposite/flat—have zero acceptance changes. Union across starts: 102 cases change acceptance; 68 have certified roots. Alternate acceptance on opposite/flat cases is not counted as mathematical recovery. All 29 interior-root recoveries at lower+0.1 match the independent certified reference within 10⁻⁵ points. Original accepted interior roots change only by small approximation error, with unchanged acceptance.

This is direct evidence of solver-path dependence. It corrects the overly broad response_analysis characterization “convention, not accuracy”: **failure to find an admissible root** is numerical, while **choosing to exclude an otherwise valid boundary root** is a convention. Native code implements the former acceptance path, not a documented uniform ε exclusion. Interior recovery is a particularly useful falsification control. The intervention does not estimate the population fraction sensitive to alternate starts, and varying starts is not proposed as a production acceptance rule.

The historical R runtime was available in the local validation environment, contrary to the response_analysis environment note. Initial execution needed its existing library directory and the host emitted a locale warning. The exact original-code/reference reconstruction passed after locating the library; no curve fitting or source replacement was needed.

## Precision, serialization and legitimate convention sensitivity

All 195,984 root brackets were checked structurally. Five 40-significant-digit serialized brackets had collapsed at extreme asymptotes. The interval calculation had retained positive boundary distances, but the stored decimal endpoints were not independently replayable. The fix serializes at working precision plus ten digits. The five are re-certified independently at 100+ digits, and all **15,620 corrected roots plus 200 accepted controls** pass the stricter equation replay. Root values used in ranking are unchanged. Other accepted roots retain their historical interval evidence; their equations were not all rerun in this continuation.

With repaired brackets, the maximum propagated pair-cell score width is **8.33×10⁻¹²**. All 21/36/79/153/269 augmented replacements at budgets 1/2/5/10/20% have guaranteed entries and exits and zero ambiguous units. Canonical ties are now handled in the guarantee function; the actual global cutoffs have no exact ties, so historical memberships do not change.

The historical perturbation tests remain verified by their hashes: interval certification has no status or top-k changes under coefficient rounding to 6/8/10/12 significant digits. The earlier 195 double-precision residual-acceptance flips therefore do not imply unstable true roots. Their interpretation is an unstable residual/certification decision followed by a large fallback switch. Updating already accepted root values changes no top-5% memberships, with maximum pair-cell shift 2.05×10⁻⁷ points.

Conventions remain consequential. Recovering only roots farther than 10⁻⁶ from an asymptote gives 4.40% top-5% turnover, below the frozen 5% materiality threshold. At 0.01, recovery of interior fallbacks alone gives 0.09%. A genuinely **uniform** ε=0.01 rule additionally rejects 2,896 native-accepted edge roots and differs from native by 1.23% at top 5%. HSA fallback differs by 30.87%. These are separate comparisons; a domain-exclusion threshold must not be presented as tighter numerical accuracy or as a repair of an undefined reference.

## What this establishes and what it does not

Established for this fixed historical pipeline: admissible fallback roots, replayable certificates, native source-path reproduction, starting-point sensitivity, named shortlist changes, asymmetry of entry/exit margins and their clustered sensitivity. The original ≥5% primary prioritization rule is met. The strict-interior result is mathematically defensible and survives independent precision checks.

Unresolved: which edge/undefined-reference convention has greater experimental value, how independent monotherapy uncertainty compares at population scale, whether current software has the same mechanism, and whether the named entrants are better biological hits. The existing 160-surface refits use status-stratified sampling and stress SDs; they cannot calibrate a screen-wide uncertainty distribution. Near-asymptote extrapolation can be scientifically fragile even when its arithmetic is certified.

## Reproduction

Use fresh output directories and the original input hashes. The returned handoff contains the complete historical certificates; local predictions/data remain untracked.

```sh
python -m synergy_artifacts.certificate_audit --certificates CERTIFICATES --output NEW_CERT_AUDIT --workers 4
python -m synergy_artifacts.native_start_audit --prepare --certificates CERTIFICATES --source PINNED_SOURCE --output NEW_START_AUDIT
Rscript --vanilla experiments/native_start_audit.R NEW_START_AUDIT/cases.csv PINNED_SOURCE NEW_START_AUDIT/native
python -m synergy_artifacts.native_start_audit --certificates CERTIFICATES --source PINNED_SOURCE --output NEW_START_AUDIT
python -m synergy_artifacts.prioritization_audit --verified VERIFIED_CASES --precision PRECISION_CERTIFICATES --certificates NEW_CERT_AUDIT/certificates_replayable.parquet --output NEW_DECISION_AUDIT
```

The original R library must expose nleqslv 3.3.2. No native refits are performed by these commands. Numerical outputs, source hashes, detailed interventions and named changed selections are in [machine-readable evidence](../results/mechanism_analysis/).
