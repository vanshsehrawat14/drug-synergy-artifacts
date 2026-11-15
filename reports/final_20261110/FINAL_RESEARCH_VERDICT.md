# Final research verdict

Vansh Sehrawat (neural models, shared pipeline), Kingsley Ukandu (Gaussian process), Sharlene Kim (collaborative filtering). 10 November 2026. Evidence: `results/final_20261110/`, `EXPERIMENTS.md`, `MATHEMATICAL_FINDINGS.md`, plus the historical branches listed there. All results after 6 November are exploratory; O'Neil and ALMANAC are examined screens.

**Bottom line.** We have reached the computational ceiling of these screens.
- **The interaction wall is an information limit.** It is not measurement noise: about 90% of the pair-cell interaction contrast is reproducible. It is not a capacity limit of the network: the same frozen network recovers the pair-consistent part ($R^2$ 0.22, equal to the cross-cell oracle) once the pair is seen. For unseen pairs, three model families do not infer it from drug-level inputs. This is evidence, not proof, that those inputs do not carry it.
- **The Loewe acceptance artifact is closed.** It has an exact mechanism and is a systematic, replicate-robust shortlist bias.
- **Next steps are not computational.** The remaining questions need new measurements or new drug-level information.

## Established claims

Scope is the O'Neil held-out-pair protocol unless stated otherwise.

**E1. Response accuracy comes from single-agent and drug-level structure.**
- Bliss with no fitting reaches $R^2$ 0.829; selected F2 reaches 0.878.
- Additive drug and drug-by-dose effects hold 81.8% of row variance. The pair-cell contrast holds 2.6% and dose-shape contrast 10.2%.
- Holds for neural, GP and CF families on identical rows.

**E2. The pair-cell interaction contrast is reproducible, and models do not predict it.**
- Replicate noise is 10.1% of the Bliss-residual pair-cell contrast energy [9.2, 11.1; drug 6.6, 13.6], so the ceiling is 0.90.
- Noise-corrected interaction-residual $R^2$ is at most 0.068: selected F2 0.068, CF 0.035, GP 0.031, production F2 −0.056.
- Weak interaction prediction is therefore not a measurement-noise ceiling.
- At dose-shape level, about 45% of the Bliss-residual target is replicate noise. GP and neural F2 capture 0.23–0.28 of the reliable remainder.

**E3. About a fifth of the interaction contrast is pair-consistent across cells.**
- The same pair's contrast in other cells predicts it with $R^2$ 0.205–0.215 [drug 0.094–0.327].
- The cross-cell covariance share is 0.234.
- It survives removal of the projected Bliss/HSA span (0.209).
- Held-out-pair models reach −0.05 to 0.06 on the same target. The gap is information available only if the pair has been seen.

**E4. Pair-seen training.**
- With each pair observed in other cells (pair-cell folds), the frozen production F2 reaches interaction-residual $R^2$ **0.218** [pair 0.157, 0.273; drug 0.051, 0.385], against −0.050 with pairs held out.
- The difference is **+0.268** [0.208, 0.324; drug 0.069, 0.468]. It is positive in all five folds and all three seeds (0.19/0.22/0.25 against −0.04/−0.09/−0.02); noise-corrected it is 0.298.
- It equals the matched cross-cell oracle (0.205): 1.05 of the oracle gain is captured.
- It persists after removal of the Bliss/HSA span (+0.271) and at dose-shape level (Bliss-residual 0.046 → 0.181). Response MSE falls 14.3% [−17.0, −11.7].
- **Reading.** The held-out-pair deficit is informational, not representational.
- **Remainder.** About 0.68 of the target energy is reproducible but not pair-consistent (ceiling 0.90 minus 0.22). Neither cross-cell averaging nor pair-seen training predicts it; within these inputs only measurements of that pair-cell would.

**E5. DrugComb Loewe labels contain scoring-induced learnable structure** (October Gate 2, `docs/preliminary_findings.md`).
- A zero-parameter monotherapy term explains contrast $R^2$ 0.77 of the label.
- With zero true interaction, learners reach 0.49–0.52.
- Response-trained models are exactly invariant to the reference.

**E6. The Loewe acceptance artifact and its mechanism.**
- **Scale.** In 14,197 cases an admissible root exists but is rejected by the termination-code rule. They are independently certified and replayable, and reproduced by the native source on 400/400 stratified cases.
- **Mechanism, exact for co-increasing LL.4 arms.**
  - A root exists iff $f_j(x_j)<h_i$ (309,056/309,056 agreements).
  - The native summed-dose fallback jumps at the existence boundary, while HSA and generalized Loewe are continuous.
  - Roots approach the smaller plateau as $\delta^{|b_i|}$ (log–log $r$ 0.997). 94% of recovered roots have $|b_i|>2$.
- **Effect on the reference.** A rejected root moves the reference by a median 4.17 points under the native fallback, versus 0.004 under a continuous fallback.

**E7. The C1 shortlist change is a systematic bias, not replicate variance.**
- C1 replaces 67/1,137 top-5% pair-cells (5.89%). The change is present in every replicate resample: 5.19–6.68%.
- Entrants are selected with probability 0.07 under native scoring and 0.78 under C1. Exits are near coin flips (0.44).
- Combination-replicate noise alone turns over 17–28% of the list.

**E8. Monotherapy-informed networks beat Bliss at response level on all four screens** (existing): −11% to −36% MSE.

## Negative results

| Hypothesis | Outcome |
|---|---|
| ONEIL high-inhibition specificity confirms on untouched NCATS screens | **Failed** (ES 0.057 [−0.006, 0.120]; DIPG −0.005) |
| Drug identity costs ≥5% response MSE | Not established: pooled four-seed +4.61%; seed 2029 alone +3.23% |
| Removing drug identity improves pair-cell contrast (identity overfitting) | Not supported: pooled −7.41% [drug −15.3, +0.5]; under a Bliss-residual objective −3.85% with disagreeing folds |
| Drug and cell identity channels interact | Not established (+2.21%, Holm p 0.38) |
| Any input ablation shows material interaction-specific improvement | None meets the frozen rule |
| Model uncertainty has practical screening utility | Frozen utility rule unmet for GP and neural ensembles |
| Input strata explain cross-screen gains | Frozen criterion not met; gaps widen after standardization |
| Weak interaction prediction is a replicate-noise ceiling | **Rejected** (A1) |
| Shortlist changes are near-tie or rounding substitutes | Rejected (incoming margins; 3/67 overlap with rounding; A2 selection probabilities) |
| Replicate agreement can identify the better Loewe convention | Not identified when monotherapy is shared (Proposition 5) |
| Weak interaction prediction reflects limited capacity of the frozen network | **Rejected** (B2): pair-seen F2 matches the cross-cell oracle |

## Open questions

1. **Biological value of either shortlist** (ledger B1). This needs prospective, plate-tagged repeats with independent monotherapy.
2. **Whether drug-level information can predict pair-consistent interaction for unseen pairs.** Identity, dose and monotherapy inputs do not.
3. **Why interaction prediction differs across screens** (ALMANAC versus ONEIL versus DIPG).
4. **Independent-repeat noise.** True cross-study noise for these contrasts; within-study ceilings are upper bounds.
5. **Current SynergyFinder releases.** Whether they keep a discontinuous fallback. The inspected 2.4.16 source uses a response grid.

## Confidence and limitations

- **High confidence in the numerical and mathematical claims** (E5, E6). They rest on proofs, 100-digit certificates, native-source reproduction and independent solvers.
- **Moderate-to-high confidence in E1–E3 and E7 as statements about this screen and protocol.**
  - **Intervals.** They are whole-pair bootstrap and drug-jackknife, with projections refitted.
  - **E2–E3 limits.** They assume block-exchangeable replicate noise and use within-study replicates.
  - **E7 limits.** It uses combination replicates only.
- **Low confidence in any generalization beyond O'Neil.** The one untouched-screen confirmation failed.
- **Not identified.** Biological synergy, causal mechanism, clinical relevance and novel-drug generalization. The task holds out pairs of seen drugs.

## Synthesis: the ten questions

1. **Strongest contribution.**
   - A calibrated account of what held-out-pair combination models learn: single-agent and drug-level response; a reproducible interaction residual they do not predict; and how much of it is information about the pair itself.
   - Together with an exact mechanism for one widely used label pipeline's learnable artifacts.
2. **What is demonstrated.** E1–E8, with the scopes stated above.
3. **Original hypotheses that failed.**
   - High-inhibition specificity (NCATS);
   - drug-identity materiality;
   - identity overfitting;
   - uncertainty utility;
   - the input-stratum explanation of screen gaps;
   - the capacity explanation of the interaction wall (B2).
4. **Does the Loewe prioritization finding survive?** Yes, narrowly. It holds as a numerical-validity finding with a systematic shortlist bias for a fixed historical pipeline, now with an exact mechanism. It does not survive as a claim of better hits. Its materiality depends on accepting near-plateau roots: $\varepsilon\ge10^{-6}$ gives 4.40%.
5. **A stronger alternative central contribution?** Yes: the calibrated evaluation (noise ceiling, cross-cell pair oracle, pair-seen control) is more general than the 5.9% turnover. It should lead, with the Loewe mechanism as the second, label-side pillar.
6. **What is novel.**
   - The SynergyFinder acceptance and fallback mechanism, its exact boundary analysis and its learnability and shortlist consequences. Targeted searches found no prior report.
   - Contrast-level, noise-corrected and pair-information-calibrated evaluation.
   - Not novel: noise ceilings, the optimism of random splits, unequal-$E_{\max}$ Loewe failure.
7. **What remains unverified.**
   - Biological utility;
   - cross-study noise;
   - Conjecture 3a (Newton exit mechanism);
   - Gate 2 numbers on the current checkout (scripts exist; re-run before use).
8. **Are more experiments justified?** No more computation on these screens. The decisive next experiment is wet-lab.
9. **Credible methodological paper.** "What do drug-combination models learn?", built on E1–E7, with the failed confirmation reported as a main result.
10. **What a skeptical reviewer would reject.**
    - Any biological or clinical reading of shortlist changes;
    - "interaction learning" without the noise and oracle calibration;
    - generalization beyond O'Neil;
    - treating oracle or pair-seen gains as achievable for unseen pairs;
    - treating exploratory O'Neil analyses as confirmatory.

## Stopping recommendation

**Stop computational experimentation and consolidate.**
