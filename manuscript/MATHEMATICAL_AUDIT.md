# Mathematical audit

Scope: Propositions 1–3 and 5 and Lemma 4 (supplement S3 and S9), and the numerical claims built on them. Each item was re-derived from its definitions. Numerical checks use the cached historical fits and the stored 100-digit interval certificates (`audit/loewe_audit.py`, output `audit/outputs/loewe_audit.json`).

## Proposition 1: existence and uniqueness of the Loewe root

- **Statement checked.** For co-increasing LL.4 arms with nonempty common interval $(\ell,h)$ and $h_i<h_j$, a root exists if and only if $f_j(x_j)<h_i$. It is unique when it exists. If $h_i=h_j$, a unique root always exists.
- **Assumptions made explicit.**
  - $e>0$, $b\neq0$, $c\neq d$; both arms increasing.
  - Common interval nonempty.
  - Doses strictly positive.
  - The inverse $D_k$ exists on the open interval between the arm's asymptotes, which contains $(\ell,h)$.
- **Existence versus uniqueness.** Uniqueness follows from strict monotonicity of $g$. Existence follows from the limits and the intermediate value theorem. Both are proved.
- **Boundary and degenerate cases.**
  - $f_j(x_j)=h_i$ exactly gives $g(h_i^-)=0$; strict decrease leaves $g>0$ inside the interval, so no admissible root exists. The statement correctly uses strict inequality.
  - Equal upper plateaus always give a root.
  - Opposite monotonicity, flat arms ($c=d$), L.4 arms and empty intervals are outside the statement and are excluded from the numerical check (49,184 opposite or flat cases; 4,160 empty intervals; 4,576 cases with an L.4 arm).
- **Co-decreasing arms.** Reduced by $y\mapsto-y$. Verified numerically on 1,872 cases, which earlier reports had not checked.
- **Numerical agreement.** 310,928/310,928 against independent certificates, 0 disagreements, 0 near-ties needing 60-digit arithmetic.
- **Novelty.** Equivalent forms are likely known. The manuscript presents the criterion as a characterization used for certification, not as a new theorem.

## Proposition 2: continuity and the summed-dose jump

- **Statement checked.**
  - As doses approach $B=\{f_j(x_j)=h_i\}$ from the root side, $y^*\to h_i$.
  - The extension equal to $y^*$ where a root exists and to $f_j(x_j)$ otherwise coincides with HSA on the no-root side.
  - At $B$ the summed-dose fallback exceeds $h_i$ by $J>0$ when $h_i<100$.
- **Issue found and resolved.** Earlier text implied the fallback always jumps for rejected roots. The proof gives $J>0$ only on $B$. Off the boundary, a saturated stronger arm can give $J\approx0$. In the 10,646 rejected near-plateau roots with plateau below 100, $J$ is positive in 58%. The manuscript states this.
- **Identity.** Native − certified $=J+(h_i-y^*)$ holds to $3.1\times10^{-12}$ on those roots.
- **Naming.** "Generalized Loewe" conflicts with BIGL's name. It is replaced by "continuous extension (equal to HSA in the no-root region)". HSA substitution for an undefined Loewe reference is available in existing software and is not claimed as new.

## Proposition 3: boundary-layer law

- **Statement checked.**
  - The exact identity $h_i-y^*=|d_i-c_i|\,\tilde q/(1+\tilde q)$.
  - The asymptotic $h_i-y^*\sim|d_i-c_i|(\delta e_i/x_i)^{|b_i|}$ for $|b_i|>1$ as $\delta\to0$.
- **Conditions.** Requires $D_j$ to be $C^1$ near $h_i$ (true for LL.4 arms inside the open interval) and $|b_i|>1$ for the ratio to converge. For $|b_i|\le1$ only the exact identity is used.
- **Proof completed (11 November, editorial pass).** Step 2 of the supplement proof used $h_i-y^*=O(\delta^{|b_i|})$ without deriving it. The proof now obtains it first from $\delta(y^*)<\delta(h_i)$ ($\delta(y)$ is increasing) and the exact identity, and ends with the passage from $\tilde q$ to $q$. The statement is unchanged; it now names $h_i<h_j$ and fixed $x_i$ as hypotheses, and the arms are written as drugs 1 and 2 because the arm label $b$ clashed with the Hill slope $b$.
- **Numerical agreement.** On 13,277 certified roots within 0.01 of the smaller plateau:
  - median absolute $\log_{10}$ ratio $2.7\times10^{-5}$;
  - 94.0% within 1%;
  - log–log correlation 0.998;
  - distances spanning $10^{-58}$ to $10^{-2}$.
  - Distances come from bracket midpoints, avoiding the float64 storage floor.

## Lemma 4: replicate-split noise estimator

- **Statement checked.** With balanced signs independent of independent, identically distributed block-error vectors, $\operatorname{Cov}(d)=\operatorname{Cov}(\bar\varepsilon)$. Hence the expected projected energy of $d$ equals the noise energy of any fixed linear contrast.
- **Issue found and resolved.** The lemma needs balance within every unit. A split balanced only per pair-cell violates it in the 210 pair-cells whose blocks carry two dose grids. That inflated the dose-shape noise share (0.447 → 0.363). The grid-balanced split satisfies the lemma, and the batch-aware split changes the result by less than 0.001.
- **Limitation.** The lemma conditions on shared inputs and requires independent block-error vectors within a batch, with arbitrary within-block covariance. Exchangeability alone is insufficient. Effects shared across blocks are absent from the noise estimate. The 16-pair cross-batch analysis provides an assumption-dependent extrapolation, not a universal biological ceiling.

## Proposition 5: invariance of within-unit replicate differences

- **Statement checked.** Within-unit replicate contrasts are identical under all conventions whose reference uses only shared inputs.
- **Correction (11 November).** The first bullet claimed that every within-unit replicate statistic has the same distribution under all conventions. The proof gives only the differences, so the statement now covers statistics that depend on a unit's block scores through their differences (variances, ranges, sign-split contrasts) and says that unit means change with the reference. The second bullet describes a reliability statistic that can change. The consequence is limited to shift-invariant within-unit statistics.
- **Correction.** Reliability is $V/(V+\sigma^2/m)$ with $V=\operatorname{Var}(\mu-R)$. The earlier wording "increases with the variance of the reference" was imprecise and now reads "adding a deterministic artifact uncorrelated with $\mu-R$ increases $V$ and therefore reliability".
- **Consequence used.** Single-agent values are identical across within-batch replicates and nearly identical across pairs. Their shared error is absent from replicate differences. Such differences cannot distinguish conventions; other reliability statistics can change, and agreement alone does not identify a biologically preferable reference.

## Cross-cell variance estimator

- **U-statistic.** $\hat\tau^2$ is unbiased for the pair-wide variance if noise and pair-by-cell terms are uncorrelated across cell lines.
- **Projection dependence.** The panel projection mixes pairs within a cell line, not across cell lines, so cross-cell independence of noise is preserved.
- **Cell-similarity structure.** If the pair-by-cell term has cell-similarity structure, its mean off-diagonal covariance is absorbed into $\hat\tau^2$. The manuscript reports the transferable structure separately (B2), so this does not change any conclusion. Since the editorial pass (11 November), §6.2 states this as the definition: $\tau^2$ is the mean cross-cell product, so the cross-cell U-statistic estimates a named statistical moment under independent noise. This does not identify a unique biological random-effects mechanism.

## Numerical certification and solver behaviour

- **Solver failure versus undefined reference.** The two are distinguished by certificate class (main-text population table): rejected certified roots (15,620) versus certified no root (117,824) versus undefined (53,344). The two never overlap.
- **Population counts (11 November).** The earlier main text said the population rows "must not be added". Its top-level rows are disjoint and sum to 367,152 (S10); only C1 and C2 are nested. Text, caption and the deck backup slide were corrected.
- **C1 and C2.**
  - C1 (14,197) is verified with float64 checks: 12,477 solver-at-root plus 1,720 solver-failed.
  - C2 (1,423) needs 100-digit certification.
  - Their union equals the 15,620 rejected certified roots exactly.
- **Bracket replay.** 15,588/15,588 rejected roots with two LL.4 arms; the 32 others involve an L.4 arm and are not replayed by the LL.4 code.
- **Unresolved.** The Newton exit mechanism (iterates leaving the interval when the boundary layer is thinner than the first step) is a conjecture supported by slope concentration and a start-point intervention. It remains explicitly conjectural. Propositions 1–3 establish properties of roots, boundary continuation and the fallback; they do not fully characterize iterative solver trajectories.

## Editorial reconciliation

Proof and lemma blocks are preserved. The native accepted total (182,247) includes 180,364 certified roots and 1,883 opposite/flat-arm outcomes outside the inverse formulation. The native fallback total is 184,905 of 367,152 cases (50.4%). Main-text population rows are disjoint; C1/C2 are described separately as nested subgroups.
