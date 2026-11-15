# Gate 2 protocol (frozen before any contrast/learnability metric is computed on the new quantities)

Written 2026-10-09 after: downloading DrugComb v1.4 raw doses (Zenodo 18449193, md5 f5f071de…, SHA-256 58c3893c…); extracting the 92,208 ONEIL blocks; reading SynergyFinder 2.0.12 / 1.8.0, nleqslv 3.3.2 and drc 3.0-1 source; and checking reconstruction fidelity on blocks 1–400 only (my Bliss and HSA equal DrugComb's to 1e-4; my Loewe emulation r = 0.91). No contrast R², learner or mono-feature analysis of the quantities below has been run.

## Label-free facts already observed (design inputs)
- Every ONEIL block is a 5 × 5 matrix: 4 doses per drug plus zero.
- Replicate blocks of a unit share identical monotherapy rows.
- A drug's monotherapy responses in a cell are the same in every block that contains it (SD 0.0005). Monotherapy is therefore drug-in-cell information that is available before any pair is measured.

## Definitions (block level, then unit mean over the unit's ONEIL blocks)
- T = mean observed inhibition over the 16 combination cells (the common raw response).
- R_B, R_H = mean Bliss and HSA references on observed monotherapy (SynergyFinder 2). DrugComb Bliss = T − R_B and HSA = T − R_H, verified.
- R_L,DC = T − (DrugComb Loewe score): DrugComb's own mean Loewe reference, recovered exactly from its published score.
- R_L,gen = exact generalised Loewe (R15b convention) on my drc-emulated LL.4 fits. A non-increasing fitted curve is "inactive": its isobole term is 0, so L = the other drug's fitted effect; if both drugs are inactive, L = the larger fitted effect.
- R_L,emul = my SynergyFinder 2 emulation: nleqslv port, summed-concentration fallback, cap at 100.
- M_DC = R_B − R_L,DC = DrugComb Loewe − DrugComb Bliss.
- M_alg = R_B − R_L,gen: reference algebra on real fitted curves.
- M_impl = R_L,gen − R_L,DC: DrugComb's deviation from exact generalised Loewe (solver, fallback, cap).
- These satisfy M_DC = M_alg + M_impl.
- Labels: Y_m = unit mean of DrugComb score m over ONEIL blocks.
- Folds: the five published O'Neil folds (common.split).
- CR²(X) = pooled contrast R² of X as a zero-parameter predictor (P_T on each test fold, common.metrics).

## P1 Identity checks
- (a) Bliss and HSA rebuilt from raw doses match DrugComb for all blocks: max |diff| < 1e-3 means "exact".
- (b) Loewe − Bliss is identical across replicate blocks of a unit.

## P2 Accounting
CR²(M_DC) for Y_L, plus the contrast-SS shares ‖P M_DC‖² / ‖P Y_L‖² and ‖P Y_B‖² / ‖P Y_L‖².

## P3 Mechanism (Gate 2 decision)
Baseline monotherapy-only skill s0 is R12's mono_mlp on DrugComb Loewe, 0.307. If the ONEIL-only label panel changes R12's number by more than 0.01, s0 is recomputed with R12's code on the ONEIL-only panel.

**PASS (mechanism identified)** iff both of the following hold:
- CR²(M_DC) ≥ 0.5 s0.
- At least one named component X ∈ {M_alg, M_impl} has CR²(M_X) ≥ 0.5 s0 on its own, with a stated algebraic or algorithmic reason why X is non-additive in drug identity.

Secondary attributions within the winning component:
- For M_impl: the fallback regime from my emulation (blocks with ≥ 1 fallback cell vs none), and the share of ‖P M_impl‖² in fallback blocks.
- For M_alg: sub-ablations of generalised Loewe with Hill slopes set to 1, and with upper asymptotes set to 100 for active curves.

**FAIL** if no single component reaches 0.5 s0. Then H2 is reported as an accounting identity, with the empirical shares and no mechanism claim.

## P4 Bliss-true null (scoring-induced learnability)
- Set T_sim = R_B + ε, with ε the unit mean of independent Gaussian cell-level noise.
- σ is the pooled replicate SD of ONEIL combination cells; noise is averaged over 16 cells × the unit's blocks.
- Then Loewe_sim = T_sim − R_L,DC and Bliss_sim = T_sim − R_B.
- Fit KNN-resid and KRR-resid (pairlearners, R03/R09 grids) on the 5 folds.
- Prediction: CR²(Loewe_sim) ≥ 0.2 with zero true interaction; CR²(Bliss_sim) ≤ 0.01.

## P5 Metric-invariant view (response to the external review)
For Z ∈ {T, Y_B, Y_H, Y_L}, report:
- the contrast SS;
- naive KNN/KRR CR², trained on Z;
- the invariant protocol: train on T, then predict Z as T̂ − R_Z;
- the reference-only predictor −R_Z, which assumes T has no non-additive structure.

Expectations:
- Invariant contrast SSE is identical across Z by construction.
- Naive SSE differs, and the difference measures how much learnability is scoring-induced.

## Not changed after freezing
- Thresholds, folds and component definitions.
- The learner grids, which are R09's.

Emulation fidelity is reported but is not used to choose among the components, because P3 uses DrugComb's exact R_L,DC.
