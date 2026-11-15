# Manuscript

"Interaction Signal and Loewe Reference Computation in Drug-Synergy Prediction on the O'Neil Screen."

| File | Content |
|---|---|
| `manuscript.tex`, `manuscript.pdf` | Main text (title page and abstract; no contents page; 8 figures, 6 tables) |
| `supplement.tex`, `supplement.pdf` | Supplement S1–S15 (contents page retained, 5 figures): data, operators, proofs, certification, extended tables, reproducibility with full input hashes and commands |
| `style.tex` | Shared layout: Latin Modern 11 pt, Computer Modern math, 0.85 in margins, 1.25 line spacing, shared departmental title block, author-year citations; contents in supplement only |
| `references.json` | 55 verified references; `build.py` lists the cited ones alphabetically in the plainnat layout |
| `figures/` | Figure PDFs (publication) and PNGs (preview), `MANIFEST.json` (provenance), `README.md` (conventions, storyboard, case-selection rules) |
| `CLAIM_EVIDENCE_LEDGER.md` | Every central claim with its artifact and status, plus corrections to earlier reports |
| `MATHEMATICAL_AUDIT.md` | Propositions 1–3 and 5 and Lemma 4: assumptions, boundary cases, numerical checks |
| `STATISTICAL_AUDIT.md` | Units, dependence, leakage, variance decomposition, paired inference, screening |
| `RELATED_WORK_AUDIT.md` | Reference verification and novelty boundary |
| `REPRODUCIBILITY.md` | Inputs with hashes, commands, seeds |
| `audit/` | Independent audit scripts and outputs |
| [`SOURCE.json`](../SOURCE.json) | Exact artifact source commit, hashes and validation |
| `scripts/` | `make_figures.py` with `figstyle.py`, `fig_measure.py`, `fig_interaction.py`, `fig_loewe.py`; `check_numbers.py` (132 checks: 116 original, 16 post-review); `build.py` (XeLaTeX) |

Build: `make_figures.py` (arguments in `REPRODUCIBILITY.md`), then `python manuscript/scripts/check_numbers.py --resolution RES` and `python manuscript/scripts/build.py`.
