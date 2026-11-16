# Preliminary pipeline

This pipeline produces the numbers in [docs/preliminary_findings.md](../../docs/preliminary_findings.md). Scripts are numbered by the original analysis log; gaps are exploratory runs that were not carried over. Run from the repo root after `python -m synergy_artifacts.download`. The total is about 30 min. r17 and r20 start 12 worker processes; lower `Pool(12)` if memory is short.

| Step | Command | Output (`results/preliminary/`) |
|---|---|---|
| Extract O'Neil doses | `python -I experiments/preliminary/r16_extract_oneil_raw.py data/raw/drugcomb/drugcomb_data_v1.4.csv` | `data/processed/drugcomb_oneil/` |
| Align DrugComb scores to the DeepSynergy panel | `python -I experiments/preliminary/r08_drugcomb_oneil_audit.py data/raw/drugcomb/summary_table_v1.4.csv` | `r08_*` |
| Rescore every block (SynergyFinder 2 emulation + alternative references) | `python -I experiments/preliminary/r17_sf2_rescore.py` | `r17_*` (~9 min) |
| Monotherapy-feature learners per score | `python -I experiments/preliminary/r12_reference_structure.py data/raw/drugcomb/summary_table_v1.4.csv` | `r12_*` |
| Drug-permutation control | `python -I experiments/preliminary/r13_mono_permutation_control.py data/raw/drugcomb/summary_table_v1.4.csv` | `r13_*` |
| Gate 2 analyses (P1–P5) | `python -I experiments/preliminary/r18_gate2_analysis.py data/raw/drugcomb/summary_table_v1.4.csv` | `r18_*` |
| s0 on the O'Neil-only panel | `python -I experiments/preliminary/r19_s0_oneil_only.py data/raw/drugcomb/summary_table_v1.4.csv` | `r19_*` |
| Fallback mechanism check | `python -I experiments/preliminary/r20_fallback_mechanism.py` | `r20_*` (~8 min) |
| Figure | `python experiments/preliminary/r21_gate2_figure.py` | `gate2_mechanism.{png,pdf}` |

Shared modules:
- `common.py`: paths, `load_oneil`, `split`, `metrics`, the additive ridge, and the MLP.
- `pairlearners.py`: symmetric pair-kernel ridge and neighborhood CF.
- `sf2_emul.py`: a Python port of SynergyFinder 2.0.12's Loewe, covering the `drc` LL.4 fit, the `nleqslv` root and the fallback.
