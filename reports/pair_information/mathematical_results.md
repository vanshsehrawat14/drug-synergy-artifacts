# Mathematical findings

Final campaign, 10 November 2026, branch `research/final-2026-11-10`. Status labels: **proved** (complete proof below), **asymptotic** (proved to leading order, with stated conditions), **elementary/known** (true and short, but not a contribution), **conjecture** (supported numerically, not proved). Numerical checks use the historical fixed fits; scripts and outputs are listed with each item.

## Definitions

**Arm.** An LL.4 curve $f(x)=c+(d-c)/(1+(x/e)^b)$, $x>0$, $e>0$, $b\neq0$, $c\neq d$. It is increasing iff $b<0,d>c$ or $b>0,d<c$. Its plateaus are $\ell_f=\lim_{x\to0^+}f$ and $h_f=\lim_{x\to\infty}f$. On the open interval between $c$ and $d$ its inverse is $D(y)=e\{(d-y)/(y-c)\}^{1/b}$.

**Loewe equation.** For co-increasing arms $a,b$ at doses $x_a,x_b>0$, with common effect interval $(\ell,h)=(\max_k\ell_k,\min_kh_k)$ assumed nonempty,
$$g(y)=\frac{x_a}{D_a(y)}+\frac{x_b}{D_b(y)}-1,\qquad y\in(\ell,h).$$
Label the arms so that $h_i\le h_j$ ($i$ has the smaller upper plateau).

**References for one dose case.** Loewe root $y^*$ (when it exists). Summed-dose fallback, the native SynergyFinder 2.0 rule: $R_{SD}=\min\{100,\max_kf_k(x_a+x_b)\}$. HSA fallback: $R_H=\max_kf_k(x_k)$. Generalized Loewe: $R_G=y^*$ if a root exists, otherwise $f_j(x_j)$. Native scoring uses $y^*$ only when Newton's method returns termination code 1 or 2 and otherwise uses $R_{SD}$. C1 is native scoring with the 14,197 independently verified admissible fallback roots restored.

**Evaluation operators.** $E$ averages identical physical inputs, $P$ averages pair-cell panels, $A$ projects panel means onto endpoint-drug incidence within cell and original pair fold, and $D$ projects onto panel-centred endpoint drug-by-dose-rank incidence. The pair-cell contrast is $C_P=P-A$ and the dose-shape contrast is $C_D=E-P-D$; both are fixed by the design. The Bliss-residual target is $C(y-r_B)$, with $r_B$ built from training-pair monotherapy.

## 1. Existence criterion for the Loewe root (proved; equivalent forms are likely known)

**Proposition 1.** For co-increasing arms with $h_i<h_j$, the function $g$ is continuous and strictly decreasing on $(\ell,h_i)$, with
$$\lim_{y\to\ell^+}g(y)=+\infty,\qquad \lim_{y\to h_i^-}g(y)=x_j/D_j(h_i)-1 .$$
Hence a root exists iff $f_j(x_j)<h_i$, and it is unique. If $h_i=h_j$, a unique root always exists.

*Proof.*
1. Each $D_k$ is continuous and strictly increasing on its own range, which contains $(\ell,h)$. So each term $x_k/D_k$ is strictly decreasing, and so is $g$.
2. As $y\to\ell^+$, every arm with $\ell_k=\ell$ has $D_k\to0^+$, so its term tends to $+\infty$. The other term is positive.
3. As $y\to h_i^-$, $D_i\to\infty$, so the first term tends to 0. Since $h_i$ lies inside the range of arm $j$, $D_j(h_i)$ is finite and positive.
4. The limit $x_j/D_j(h_i)-1$ is negative iff $x_j<D_j(h_i)$, iff $f_j(x_j)<h_i$, because $f_j$ is strictly increasing with $f_j(D_j(h_i))=h_i$.
5. If $h_i=h_j$, both $D_k\to\infty$ and the limit is $-1$.
6. The intermediate value theorem and strict monotonicity give existence and uniqueness. $\square$

The condition constrains the **doses**: the stronger-plateau arm alone must not already reach the weaker arm's plateau. It is not a condition on a "target response", which is the unknown.

**Numerical check.** On every co-increasing LL.4/LL.4 case with an interval certificate, the criterion agrees with the independent 100-digit certificates in **309,056/309,056** cases, with no disagreement (`experiments/pair_information/loewe_boundary.py`, `results/pair_information/loewe_boundary/summary.json`).

## 2. Continuity, and the summed-dose jump (proved)

**Proposition 2.** Take co-increasing arms with $h_i<h_j$ and $h_i<100$, and let the doses approach the boundary $B=\{f_j(x_j)=h_i\}$ from the root side. Then:
1. $y^*\to h_i$, so $R_G$ is continuous across $B$.
2. On the no-root side $R_H=R_G$, so the HSA fallback is also continuous across $B$.
3. The summed-dose fallback jumps across $B$ by
$$J=\min\{100,\max_kf_k(x_a+x_b)\}-h_i>0 .$$

*Proof.*
1. Let the doses converge to a point $(\bar x_i,\bar x_j)\in B$, so $\bar x_j=D_j(h_i)$.
   - Fix $\varepsilon>0$. By continuity in the doses, $g(h_i-\varepsilon)$ converges to $G(\varepsilon)=\bar x_i/D_i(h_i-\varepsilon)+D_j(h_i)/D_j(h_i-\varepsilon)-1$.
   - $D_j$ is strictly increasing, so the second ratio exceeds 1. Hence $G(\varepsilon)>\bar x_i/D_i(h_i-\varepsilon)>0$.
   - For doses close enough to the limit, $g(h_i-\varepsilon)>0$. Since $g$ is decreasing, $y^*\in(h_i-\varepsilon,h_i)$.
   - As $\varepsilon$ is arbitrary, $y^*\to h_i=\lim f_j(x_j)$.
2. On the no-root side $f_j(x_j)\ge h_i>f_i(x_i)$, so $R_H=\max_kf_k(x_k)=f_j(x_j)=R_G$.
3. On $B$, $x_a+x_b>x_j$ and $f_j$ is strictly increasing, so $\max_kf_k(x_a+x_b)\ge f_j(x_a+x_b)>f_j(x_j)=h_i$. Since $h_i<100$, $J>0$. $\square$

A rejected root just inside $B$ replaces $y^*\approx h_i$ by $R_{SD}$. The resulting reference error, $J+(h_i-y^*)$, is essentially the jump. This is the mechanism behind the large score increases of C1 entrants: a mean own-score change of +8.1 points, with near-tied exits.

**Numerical check.** Among co-increasing LL.4 C1 cases:
- **Upper-plateau roots.** 10,712 lie within 0.01 of $h_i$. For these, native − certified root equals $J$ up to the root's distance from the plateau: median deviation $1.8\times10^{-4}$, maximum below 0.01.
- **Jump size.** Median 3.67 points, mean 18.8.
- **Share of the shift.** These cases carry 68.3% of the absolute C1 reference shift. Another 1,608 roots lie near the lower asymptote and 1,536 in the interior; there the jump formula does not apply.

**Practical consequence.** A rejected root is costly only because the native fallback is discontinuous. On all 14,197 rejected admissible roots:

| Fallback | Median error | Mean error |
|---|---:|---:|
| Native summed-dose | 4.17 points | 20.9 points |
| Continuous HSA | 0.0044 points | 0.73 points |

The HSA fallback is closer in 71.7% of cases (`fallback_error_on_rejected_roots`). The two remedies point the same way:
- certify roots before falling back;
- use a fallback that is continuous at $B$.

Changing the fallback for the 117,824 genuine no-root cases is a separate convention change: 30.9% top-5% turnover.

## 3. Boundary-layer law (asymptotic)

**Proposition 3.** Under the conditions of Proposition 2, let $\delta=1-x_j/D_j(h_i)\in(0,1)$ and $q=(\delta e_i/x_i)^{|b_i|}$. The root satisfies the exact identity
$$h_i-y^*=|d_i-c_i|\,\frac{\tilde q}{1+\tilde q},\qquad \tilde q=\{\delta(y^*)e_i/x_i\}^{|b_i|},\quad \delta(y)=1-x_j/D_j(y).$$
If $|b_i|>1$, then $\tilde q/q\to1$ as $\delta\to0$, so $h_i-y^*\sim|d_i-c_i|\,q$.

*Proof.*
1. The root equation is $x_i/D_i(y^*)=\delta(y^*)$. Substituting $D_i$ gives $\{(h_i-y^*)/(y^*-\ell_i)\}=\tilde q$ in both sign conventions of $b_i$. Solving for $h_i-y^*$ gives the identity.
2. $D_j$ is $C^1$ near $h_i$, so $|\delta(y^*)-\delta|=O(h_i-y^*)=O(\delta^{|b_i|})$.
3. Hence $\tilde q/q=(\delta(y^*)/\delta)^{|b_i|}=1+O(\delta^{|b_i|-1})\to1$ when $|b_i|>1$. $\square$

Admissible roots therefore sit in a layer of width proportional to $\delta^{|b_i|}$. Steep Hill slopes drive them far below the resolution at which the plateau variable can be represented. For example, $|b_i|=8$ and $\delta e_i/x_i=0.01$ give $q=10^{-16}$.

**Numerical check.**
- **Agreement.** On 12,964 certified roots within 0.01 of the smaller plateau, the leading-order law and the certified distance agree with log–log correlation **0.997** over ten decades. They depart only at the $\sim10^{-12}$ floor of float64 root storage (`fig3_boundary_layer`).
- **Steep slopes.** 93.8% of C1-recovered roots have $|b_i|>2$, versus 43.3% of all certified roots.
- **Reading.** The numerical failures are structurally predictable from $(\delta,|b_i|)$; they are not random. This replaces the proposed generic condition-number analysis with an exact local law.

**Conjecture 3a.** Native Newton rejection, started at $\max(c_a,c_b)+1$, is driven by iterates leaving $(\ell,h_i)$ when the layer is thinner than the first step. Evidence:
- the slope concentration above;
- the start-point intervention (102/400 acceptance changes, 68 with certified roots).

No convergence-basin proof is given.

## 4. A replicate-split noise proxy that preserves block correlation (proved)

**Lemma 4.** Fix a panel $p$ with an even number $n$ of replicate blocks and units $u$. Assume:
- $y_{ub}=\mu_u+\varepsilon_{ub}$;
- block error vectors $\varepsilon_{\cdot b}$ are independent across blocks and across panels, identically distributed within the panel, and mean zero;
- within-block covariance $\Sigma_p$ is arbitrary.

Draw a sign vector $\sigma$ with $n/2$ entries equal to $+1$ and $n/2$ equal to $-1$, independently of $\varepsilon$. Set $d_u=n^{-1}\sum_b\sigma_by_{ub}$. Then $E d=0$ and $\operatorname{Cov}(d)=\operatorname{Cov}(\bar\varepsilon)=\Sigma_p/n$. Consequently, for every fixed linear operator $C$ and weight $W$, $E\|Cd\|_W^2=E\|C\bar\varepsilon\|_W^2$.

*Proof.*
1. $\sum_b\sigma_b=0$ removes $\mu$.
2. $\operatorname{Cov}(d_u,d_{u'})=n^{-2}\sum_b\sigma_b^2\Sigma_p[u,u']=\Sigma_p[u,u']/n$.
3. This equals $\operatorname{Cov}(\bar\varepsilon_u,\bar\varepsilon_{u'})$.
4. Then $E\|Cd\|^2=\operatorname{tr}(C^\top WC\operatorname{Cov}(d))$. $\square$

Signs are assigned to **blocks**, not rows. Whole-block shifts, the noise that matters most for pair-cell contrasts, are therefore counted. The noise-corrected statistics
$$R^2_{\text{corr}}=1-\frac{\|Ce\|^2-\hat N}{\|Cy\|^2-\hat N}$$
are ratios of unbiased estimators: consistent, not unbiased. If blocks share plates or processing, $\hat N$ underestimates independent-repeat noise, and every ceiling is an upper bound. Monotherapy is shared across O'Neil blocks, so its noise is absent from $\hat N$.

## 5. Replicates sharing a reference cannot rank reference conventions (proved)

**Proposition 5.** Let block scores be $S^v_{ub}=y_{ub}-R^v_u$, where each convention's reference $R^v_u$ depends only on inputs shared by all blocks of unit $u$. Then:
1. $S^v_{ub}-S^v_{ub'}=y_{ub}-y_{ub'}$ for every convention $v$. Every statistic of within-unit replicate contrasts has the same distribution under all conventions.
2. With unit means over $m$ blocks, the intraclass reliability
$$\rho_v=\frac{\operatorname{Var}_u(\mu_u-R^v_u)}{\operatorname{Var}_u(\mu_u-R^v_u)+\sigma^2/m}$$
is increasing in the between-unit variance of the deterministic reference. Adding to $R^v$ any artifact uncorrelated with $\mu-R^v$ strictly increases $\rho_v$. $\square$

Replicate reproducibility therefore rewards deterministic reference variance; it cannot validate a convention when monotherapy is shared. The replicate-anchored convention test proposed in the November 7 response_analysis handoff is **not identified** on O'Neil. It needs independent monotherapy replicates or external outcomes.

Consistent with this, A2 finds:
- half-depth replicate turnover of the top-5% list is the same under N (28.5%) and C1 (28.8%);
- the N → C1 change is identical (5.9%) in every resample.

## 6. Reference cancellation (elementary/known)

If $\hat S=\hat Y-R$ and $S=Y-R$ for any reference fixed before outcomes, then $\hat S-S=\hat Y-Y$. Every loss and every linear contrast of the error is the same for every response-minus-reference score; only denominators differ. The identity is in `docs/preliminary_findings.md` (main) and the November 7 report. It is the reason response-trained models are unaffected by the Loewe acceptance artifact.

## 7. Counterexamples and corrections

1. **"No real positive root for target responses above $\min E_{\max}$".** As stated, this is imprecise: the response is the unknown. Proposition 1 gives the exact dose condition. The 117,824 certified no-root cases are definitional, and distinct from the 15,620 certified roots that the native solver rejected.
2. **"With monotherapy retained, identity is needed only for the cross term."** The exact error attribution is a counterexample (`results/mechanism_analysis/information/error_attribution.csv`). Removing drug identity raises the additive panel and additive dose errors (+3.82 and +5.69 points²) while lowering both contrast errors (−1.20 and −0.61). In F2, identity mainly carries additive calibration. A response-level ablation cannot show that interaction is "orthogonal" to what the model learns.
3. **"$\delta_{num}$ is predictable on held-out pairs, which proves leakage."** $\delta_{num}=R_{\text{native}}-R_{\text{certified}}$ is a deterministic function of pre-label inputs: monotherapy fits and doses of drugs seen in training. Its predictability follows by construction for any learner with monotherapy access. It measures scoring-induced learnability (the null-world control in `docs/preliminary_findings.md`: Loewe contrast $R^2$ 0.49–0.52 with zero true interaction). Under the declared task, unseen pairs of seen drugs, it is not leakage.
4. **Reliability as validation.** Proposition 5 gives explicit cases in which a deterministic artifact raises replicate reliability.

## 8. Unresolved

- **Conjecture 3a** (Newton exit mechanism).
- **Cross-cell covariance share.** The share $\kappa=\sum_p\sum_{c\ne c'}t_{pc}t_{pc'}/\{(C_p-1)\sum_pt_{pc}^2\}$ estimates the pair-consistent share of contrast energy under $t_{pc}=a_p+b_{pc}+e_{pc}$. The $(P-A)$ projection induces weak dependence across cells, so this is approximate, not unbiased. It is 0.234 for the Bliss-residual and 0.244 for the raw pair-cell contrast.
- **Scope of Propositions 1–3.** Co-decreasing arms follow by symmetry but were not checked numerically. cases with an L.4 arm (at most 4,592) and opposite or flat pairs (49,184 cases) are outside these statements.

## Numerical assumptions

- All analyses are float64.
- Certificates are the historical 100-digit interval brackets (`interaction_analysis`, `mechanism_analysis`).
- Certified roots are stored as float64, which creates the distance floor near $10^{-12}$.
- Every check uses the cached fixed fits, not native refits.
