# Data

`python -m cs622.download [name ...]` fetches into `data/raw/` and verifies checksums. With no arguments it fetches everything.

| Name | File | Size | Source | License | Used for |
|---|---|---|---|---|---|
| `deepsynergy` | `deepsynergy/labels.csv` | 1.6 MB | [DeepSynergy](https://www.bioinf.jku.at/software/DeepSynergy/) (Preuer et al. 2018) | none stated; cite | O'Neil Loewe labels and the five pair folds |
| `drugcomb_summary` | `drugcomb/summary_table_v1.4.csv` | 193 MB | [Zenodo 11102665](https://zenodo.org/records/11102665) | CC-BY-4.0 | DrugComb block-level scores (Bliss, HSA, Loewe, ZIP, S) |
| `drugcomb_raw` | `drugcomb/drugcomb_data_v1.4.csv` | 2.0 GB | [Zenodo 18449193](https://zenodo.org/records/18449193) | CC-BY-4.0 | Dose-level % inhibition for all DrugComb studies |
| `almanac` | `almanac/ComboDrugGrowth_Nov2017.csv` | 615 MB | [NCI-ALMANAC](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC) (Holbeck et al. 2017) | NCI public data | Independent screen |

`data/processed/` holds derived files. `experiments/preliminary/r16_extract_oneil_raw.py` writes the O'Neil dose-level subset there (92,208 blocks × 25 dose cells, 11 MB).

O'Neil is matched across sources by normalized drug and cell-line names. DeepSynergy has 315 duplicated pair–cell units; these are averaged.
