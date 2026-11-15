# Preliminary findings: the DrugComb Loewe fallback

O'Neil screen inside DrugComb v1.4. Analyses follow a protocol frozen before any learnability number was computed ([gate2_protocol.md](gate2_protocol.md), 2026-10-09). Numbers come from `results/preliminary/`.

## Setup

- **Units:** 22,737 pair × cell-line units, giving a 583 × 39 panel. Each DrugComb block is a 5 × 5 dose matrix: 4 doses per drug plus zero.
- **Raw response:** *T* is the mean observed % inhibition over the 16 combination cells.
- **References:** *R*<sub>B</sub> and *R*<sub>H</sub> are the Bliss and HSA references on observed monotherapy. *R*<sub>L,DC</sub> = *T* − DrugComb Loewe is DrugComb's own Loewe reference.
- **Evaluation:** DeepSynergy's five pair-grouped folds; a test pair is unseen, but both of its drugs appear in training.
- **Contrast R²:** on each test fold, project predictions and labels off the additive drug-in-cell design (least squares on drug indicators, per cell line). Then R² = 1 − ‖Pŷ − Py‖² / ‖Py‖², pooled over folds. It scores only what is beyond additive single-drug effects, which is the part a synergy model claims to learn.

## Identities

- DrugComb Bliss = *T* − *R*<sub>B</sub> and HSA = *T* − *R*<sub>H</sub>, exact to 7e-14 over 92,208 blocks.
- Monotherapy rows are identical across a drug's blocks in a cell line (SD 0.0005). Single-agent data are therefore drug-in-cell information that exists before any pair is measured.
- For any two response-minus-reference scores, *S*<sub>m</sub> − *S*<sub>n</sub> = *R*<sub>n</sub> − *R*<sub>m</sub>. Score differences are reference differences.

## Decomposition

Let *M* = *R*<sub>B</sub> − *R*<sub>L,DC</sub>, which equals DrugComb Loewe − DrugComb Bliss. Split it exactly as *M* = *M*<sub>alg</sub> + *M*<sub>impl</sub>:

- *M*<sub>alg</sub> = *R*<sub>B</sub> − *R*<sub>L,gen</sub> is the textbook algebra: exact generalized Loewe on drc-emulated LL.4 fits.
- *M*<sub>impl</sub> = *R*<sub>L,gen</sub> − *R*<sub>L,DC</sub> is DrugComb's deviation from exact Loewe.

| Component, used as a zero-parameter predictor of the Loewe label | Contrast R² |
|---|---|
| *M* | 0.772 |
| *M*<sub>impl</sub> | 0.752 |
| *M*<sub>alg</sub> | −0.047 |
| *M*<sub>alg</sub>, idealized curves (Hill slope 1, E<sub>max</sub> 100, or both) | −0.021 / −0.019 / 0.002 |
| Our reimplementation of DrugComb's existence rule, from monotherapy only (post hoc) | 0.703 |

Compare the trained learners: kernel ridge 0.416, neighborhood CF 0.377, single-agent MLP 0.307.

*M*<sub>impl</sub> concentrates where the fallback fires: 92% of its contrast energy sits in fallback units, and 80% of units have at least one fallback cell.

## Mechanism

For increasing fitted curves *f*<sub>k</sub> with upper asymptotes *d*<sub>k</sub>, let *i* be the drug with the smaller *d*.

- The isobole sum is decreasing on the joint domain. A Loewe root therefore exists if and only if *f*<sub>j</sub>(*x*<sub>j</sub>) < *d*<sub>i</sub>.
- Without a root, exact Loewe gives *f*<sub>j</sub>(*x*<sub>j</sub>).
- DrugComb (SynergyFinder 2.0.12, `nleqslv` termcd ≥ 3) instead returns max<sub>k</sub> *f*<sub>k</sub>(*x*<sub>i</sub> + *x*<sub>j</sub>). It adds the raw concentrations of two different drugs.
- The resulting shift, *f*<sub>j</sub>(*x*<sub>i</sub> + *x*<sub>j</sub>) − *f*<sub>j</sub>(*x*<sub>j</sub>), is a sigmoid in log(*x*<sub>i</sub> / *e*<sub>j</sub>). That is a cross term between the two drugs, so it is non-additive in drug identity, and it is fixed by monotherapy and the dose grid alone.

Checked on 367,152 unique dose cells:

| Regime | Share of cells | Mean inflation of the Loewe reference (% inhibition) |
|---|---|---|
| Root exists | 53% | 0 |
| Stronger drug's effect exceeds the weaker drug's maximum | 33% | 11.5 |
| One fitted curve inactive | 14% | 24.0 |

The shift rises across quintiles of log<sub>10</sub>(partner dose / stronger-drug IC<sub>50</sub>) and then plateaus: 1.7, 12.8, 21.2, 21.8 and 19.5 (Spearman 0.32).

## Controls

| Labels | Neighborhood CF | Kernel ridge |
|---|---|---|
| DrugComb Loewe, real | 0.377 | 0.416 |
| Loewe, simulated zero-interaction screen (*T* = Bliss expectation + noise) | 0.489 | 0.515 |
| DrugComb Bliss, real | 0.156 | 0.149 |
| Bliss, simulated zero-interaction screen | −0.099 | −0.001 |

- **Drug permutation:** permuting the drug ↔ monotherapy-profile assignment removes the single-agent MLP's Loewe skill. It falls from 0.304 to about 0.
- **Common response model:** training on *T* and subtracting each score's exact reference gives identical absolute error for every score (kernel ridge SSE 479k). Cross-score differences in R² therefore reflect the reference and the denominator, not more learnable biology.
- **Bliss reference:** the Bliss expectation alone explains 31% of *T*'s non-additive variance. A reference can be a useful offset (Bliss) or a dominant nuisance (DrugComb Loewe).

## Open

- Scope is O'Neil within DrugComb v1.4 only. Fallback rates in other studies (different dose designs) are untested; this is milestone 1.
- The fits are Python ports of `drc` and `nleqslv`, not the R binaries. Agreement with DrugComb's block-level Loewe reference is r = 0.988, with 77% of blocks exact to 0.01.
- DeepSynergy's Loewe labels (Combenefit) show no monotherapy-only skill. Combenefit's handling of undefined regions has not been inspected.
- Novelty rests on targeted searches. MuSyC (Meyer et al. 2019) notes that Loewe is undefined when maximal effects differ; no source found reports this fallback or its effect on learning.
