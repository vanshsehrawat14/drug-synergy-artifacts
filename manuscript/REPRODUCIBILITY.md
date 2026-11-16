# Reproducibility

All commands run from the repository root. Python 3.14, numpy 2.4.1, pandas 2.3.3, scipy 1.17.0, matplotlib 3.10.8, mpmath 1.3.0. pdfLaTeX (MiKTeX) with Computer Modern; figures use matplotlib's bundled Computer Modern (cmr10, cmb10).

## Frozen inputs

Supplement S14 gives the full SHA-256 of every input (Tables S14 and S15) and the complete command sequence.

| Input | Location | SHA-256 |
|---|---|---|
| O'Neil observation table (DrugComb v1.4) | `data/processed/oneil_v1/observations.parquet` (from `python -m synergy_artifacts.download deepsynergy drugcomb_summary drugcomb_raw` and `python -m synergy_artifacts.data --config configs/data_oneil.json`; not tracked) | `ea588405019971906ac99c3caa547d5f4fac5b1be340356f0a8fe17a6aa3ce8b` |
| Resolution cache (out-of-fold unit means) | `results/interaction_analysis/resolution/`, written by `python -m synergy_artifacts.resolution` at commit d2c8f41 (`cache.npz` is not tracked) | `cache.npz` `91f93811dedf5c0721198146638c21f25b1616d43f43ae9322cf8790a7e86687` |
| Pair-cell split | `results/pair_information/oracle/pair_cell_split.csv` | `d32914a94d8ead90e12d8cb10b6beed15f93339b0de2200ce2075a67a79370ea` (LF, as committed; a CRLF checkout gives `794db1fe9acf49316aad2e9789b04aa57f5b8e49c075fd0fcbf85eb37494e7e1`) |
| Pair-seen predictions (15 files) | Cluster run directory `results/pair_information/runs/f2_pair_seen/` (not tracked) | listed in `audit/outputs/pairseen_audit.json`; all 15 re-verified on 2026-11-10 |
| Final campaign archive | `final_return_20261110.tar.gz` (retained by the authors, not tracked) | `c2e369a075c798d66561d58f77ada9bc4230400c31944899fa4e44c8d7172b77`, re-verified (154 entries) |
| Pair-seen summary return | `research_return_20261110_pairseen_summary.tar.gz` (retained by the authors, not tracked) | prefix `2a63212e`, re-verified |
| Expression features | `data/processed/deepsynergy/features.npz` | `4ece7882cbc3e1576bc7400d5c2f53d48d466de9eaafbc5d69ff1bb547d7f40d` |

## Steps

Let `RES` be the resolution cache directory and `DERIVED` a directory outside the repository.

1. **Independent audits** (outputs in `manuscript/audit/outputs/`). Complete commands are in Supplement S14; the Loewe and screening audits also need `--derived`, which their docstrings omit:
   - `manuscript/audit/interaction_audit.py`;
   - `loewe_audit.py`;
   - `screening_audit.py`;
   - `gate2_compare.py` (after re-executing `experiments/preliminary/r16–r21` in a clean source checkout).
2. **Interaction-gap analyses** (outputs in `results/interaction_structure/`):
   ```
   X=experiments/interaction_structure; OBS=data/processed/oneil_v1/observations.parquet; OUT=results/interaction_structure
   python $X/variance.py --observations $OBS --resolution $RES --output $OUT --derived $DERIVED
   python $X/anchors.py --observations $OBS --resolution $RES --split results/pair_information/oracle/pair_cell_split.csv \
       --pairseen manuscript/audit/outputs/pairseen_audit.json --output $OUT
   python $X/grounding.py --observations $OBS --resolution $RES --features data/processed/deepsynergy/features.npz --output $OUT
   python $X/batch_transfer.py --observations $OBS --resolution $RES --output $OUT
   python $X/dose_shape_check.py --observations $OBS --resolution $RES --output $OUT
   ```
3. **Paired pair-seen comparison** (on the cluster, where the predictions are; no training):
   ```
   python b3_pair_energies.py --observations OBS --resolution RES_B2 --runs RUNS/f2_pair_seen --split SPLIT \
       --pairseen pairseen_audit.json --output b3_pair_energies.json
   python b3_paired.py --observations OBS --resolution RES_B2 --split SPLIT --pairseen pairseen_audit.json \
       --b3 b3_pair_energies.json --output b3_paired.json
   ```
   The scripts used are `core.py`, `anchors.py`, `variance.py`, `b3_pair_energies.py` and `b3_paired.py` from this repository, uploaded as bundles with SHA-256 prefixes `8e250eb9604d2b27` and `08f23608b68acb21`.
   - Cluster outputs: `b3_pair_energies.json` (`411b68bc8af94286…`) and `b3_paired.json` (`a8fc4bff417c54a7…`).
   - The summary is transcribed in `results/interaction_structure/b3_paired_summary.json`.
   - The recomputed pooled B3 $R^2$ equals the recorded value exactly.
4. **Figures** (all 13 main and supplementary figures; about 40 s):
   ```
   N=NATIVE_RUN   # results/neural/runs/scientific_native_full of a research checkout
   python manuscript/scripts/make_figures.py --observations data/processed/oneil_v1/observations.parquet \
       --targets $DERIVED/pair_cell_targets.parquet --cross-batch $DERIVED/cross_batch_values.csv \
       --surfaces $N/surface_cases.parquet --native-panel $N/panel.parquet --derived DERIVED_AUDIT --resolution $RES \
       --gate2 GATE2_RERUN/results/preliminary --labels data/raw/deepsynergy/labels.csv
   ```
   - `DERIVED` is the `--derived` directory of `variance.py` (step 2); `DERIVED_AUDIT` holds the per-case tables written by the Loewe and screening audits.
   - The figure manifest records exact current input, output and script hashes plus case-selection notes. Editorial figure redesigns intentionally change output bytes; scientific data are checked separately against their preceding figures and frozen inputs.
   - Conventions, storyboard and selection rules: `manuscript/figures/README.md`.
5. **Number check and build:**
   ```
   python manuscript/scripts/check_numbers.py --resolution $RES      # 147 checks (116 original, 16 DeepSynergy, 15 editorial)
   python manuscript/scripts/build.py                                 # manuscript.tex, supplement.tex -> PDFs
   ```
6. **Tests:** `python -m pytest tests -q`.

## Determinism

| Procedure | Seed |
|---|---|
| Block signs | 97 |
| Pair bootstrap | 11 |
| Cross-batch bootstrap | 13 |
| Permutations | 17 |
| Zero-interaction realizations | 19 |
| Reference perturbations | 23 |
| Mantel tests | 29–41 |
| Batch-transfer bootstrap | 43 |
| Anchor masks | SHA-256 of (pair, seed) |
| Network seeds | 2026–2028 |

Recomputations reproduce frozen values bit for bit: pair-cell and dose-shape noise shares, Gate 2 outputs (maximum difference 0) and the pair-seen pooled $R^2$.

## Not redistributed

Per-pair-cell targets, per-case Loewe tables and the observation table are derived from the DrugComb export and are regenerated by the scripts rather than committed.

## Manuscript revision validation

The editorial revision retains all frozen results, research code, model configurations, proofs, certificates and input hashes. No model was retrained. The full suite passes 142 tests with the documented original CF real-data test skipped; the numerical checker passes 147/147 checks.

The main manuscript has 30 pages and the supplement 29, with no missing glyphs, undefined references, overfull boxes or oversized floats. Eight figures were revised for information access, evaluation populations, simulation definitions or visual clarity. Thirty-three panel-level comparisons preserve the original plotted scientific arrays, including the population panels moved from Figure 7 to S5. Five unmodified figures retain their original PDF/PNG hashes. Current figure hashes are in `figures/MANIFEST.json`; exact source and document hashes are in the repository's `SOURCE.json` and `MANIFEST.json`.
