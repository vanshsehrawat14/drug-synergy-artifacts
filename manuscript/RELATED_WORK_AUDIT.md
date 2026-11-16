# Related-work audit

## Bibliographic verification

`references.json` holds 55 entries. Every entry records how it was verified:
- 46 against Crossref, including four checked against primary full texts;
- 3 against PubMed;
- 2 against arXiv;
- 1 against PubMed Central (a bioRxiv preprint);
- 1 against PMLR, 1 against CRAN and 1 against the GitHub source (SynergyFinder).

Cached metadata is in `audit/outputs/reference_metadata/`. Two entries were added for this version and checked against Crossref on 2026-11-10: Tosh et al. 2025 (*Nat. Commun.* 16:156) and Wang et al. 2025 (*Sci. Rep.* 15:3484). The SynergyFinder source entry no longer lists authors, which could not be verified for the source code itself. Asiaee et al. 2026 (bioRxiv, version 2) was added on 2026-11-11 from its PubMed Central record (PMC13060067), which gives the authors, DOI and full text.

The revised manuscript cites 41 entries and the supplement eight; their union is 41, leaving 14 unused records outside the generated reference lists. The 37 cited DOI records were rechecked against registered titles and publication dates. The other four cited references are the two PubMed-indexed reviews and the versioned software sources. The underlying reference records are unchanged.

## What each cited work is used for

| Theme | Works | Use in the manuscript | Basis |
|---|---|---|---|
| Screens and resources | O'Neil 2016; Holbeck 2017; DrugComb 2019/2021 | data source; context | metadata; data counts recomputed from the export |
| Scoring software | SynergyFinder 2017/2020; source 2.0.3 and 3.7.4; nleqslv | solver behaviour: start $\max(c)+1$, accept codes < 3, cap 100; 3.7.4 response grid with summed-dose fallback | source code read directly |
| Reference models | Bliss 1939; Loewe 1926/1953; Greco 1995; Berenbaum 1989; Geary 2013; Tang 2015; Lederer 2019; Vlot 2019; Grabovsky 2004 | definitions; known disagreements and unequal-maximum failure | standard content |
| Synergy testing without a parametric reference | Asiaee 2026 (bioRxiv) | §2 pointer and §11 contrast with our analysis: they replace the reference with an isotonic null and a wild-bootstrap test; on DrugCombDB, higher replicate concordance than Bliss, HSA, Loewe and ZIP; Loewe non-finite in 20.9% of experiments from failed single-agent fits; no prediction model trained | full text (PMC, version 2) |
| Alternatives for unequal maxima | BIGL; BRAID; MuSyC (two papers); asymptotic Loewe (Twarog 2020) | prior work on the same problem; HSA substitution in the `synergy` package (Wooten 2021) | package documentation and source modes (`delta_hsa`, `delta_weakest`) |
| Prediction methods | DeepSynergy; MatchMaker; DeepDDS; JointSyn; DeepSynBa; comboFM; BATCHIE; active-learning guide | target/protocol comparison and bounded reproduction decision; possible future experimental design | primary papers, supplements and official implementations for the comparison (S15 and `docs/published_models.md`) |
| Evaluation pitfalls | Park & Marcotte 2012; Bernett 2024; One-hot news 2026; SynVerse 2025; Zhang 2023 (*Commun. Biol.*) | split sensitivity; leakage; identity shortcuts; within- versus across-study reproducibility | abstracts; Zhang 2023 full text checked previously (intra-study replicate correlation Loewe 0.938 vs Bliss 0.778) |
| Noise ceilings | Lage-Castellanos 2019; Schoppe 2016 | ceilings are not new | abstracts |
| Collaborative filtering | Sarwar 2001; Koren 2008 | CF method lineage | metadata |

## Claims removed or narrowed

- **"Pearson 0.09–0.25 across studies"** (attributed to the *Commun. Biol.* 2023 paper) was not found in the source and is removed.
- **DrugComb's statement that v1.4 used SynergyFinder 2.0.3** could not be retrieved. The manuscript therefore describes its findings as concerning the reconstruction (the 2.0.12 Loewe code, whose solver branch is identical to 2.0.3's, matching published v1.4 scores with RMSE 0.704 and $r$ 0.9990) and states that the release DrugComb ran was not established.
- **Prior art for HSA substitution.** The `synergy` package already substitutes HSA or the weaker drug when Loewe is undefined, so no novelty is claimed for HSA substitution.
- **Name clash.** "Generalized Loewe" clashes with BIGL and is renamed.
- **Identity-shortcut papers.** The early audit used the One-hot news and SynVerse abstracts. The post-review implementation audit below supersedes that scope with primary papers, methods and available code; their target and split distinctions remain explicit.

## Post-review implementation audit, 15 November 2026

The primary papers and available supplementary methods for DeepSynergy, MatchMaker, DeepDDS, JointSyn, DeepSynBa, comboFM, One-hot news and SynVerse are now reviewed in [`docs/published_models.md`](../docs/published_models.md). This supersedes the abstract-only scope above for these works. Four new DOI entries are verified against Crossref and the primary full texts in `audit/outputs/reference_metadata/published_models.json`.

The main text adds a compact task-comparability table. Supplement S15 separates the two existing DeepSynergy experiments, reports the original-label port in its own table, retains the dose adaptation’s uncertainty and budget limitation, and records the failed full-population/configuration gate for an additional response model. No new fits or frozen-result changes were made. Published aggregate-score and classification values are not used as scores on the inhibition/Bliss-residual task.

## Novelty boundary

**Not claimed:**
- noise ceilings;
- split sensitivity and identity shortcuts;
- failure of Loewe additivity for unequal maxima;
- HSA substitution;
- active selection.

**Claimed, and how far the evidence supports it:**
1. **The interaction decomposition.** Statistical variance components and out-of-sample tests of same-pair supervision, with replicate/batch calibration and leak-free paired comparisons. B1/B3 = 81% is a ratio of predictive R² values. In the leave-one-cell-out B1/B2 setting, unpredicted reproducible energy is 54.3% of total energy, or 60.2% of estimated within-batch reproducible energy. Single screen; no biological mechanism identified.
2. **The Loewe analysis.** Exact root-existence, continuation, boundary-law and fallback-discontinuity results, numerical certification of 15,620 rejected roots and the screening consequence (67/1,137). Findings use the 2.0.12 reconstruction, whose solver branch matches 2.0.3; the mechanism of native Newton rejection remains conjectural. No complete characterization of iterative solver behavior or biological benefit is claimed.
3. **Reference-induced learnability under a zero-interaction control.** Reproduced from the October protocol; a demonstration, not a general theorem.
