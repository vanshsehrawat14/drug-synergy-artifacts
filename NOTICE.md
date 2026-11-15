# Licensing and data provenance

Original software and its usage documentation are released under the [MIT license](LICENSE), with copyright held by Vanshdeep Sehrawat, Kingsley Ukandu and Sharlene Kim. This software license does not relicense upstream datasets or the manuscript and supplement; the latter retain the authors' copyright unless separately licensed.

| Resource | Use | Terms and access |
|---|---|---|
| DrugComb v1.4 [dose-level responses](https://doi.org/10.5281/zenodo.18449193) and [block summaries](https://doi.org/10.5281/zenodo.11102665) | O'Neil measurements and reference scores; derived scientific summaries in this repository | Deposits specify CC BY 4.0. Cite DrugComb and O'Neil as listed in `manuscript/references.json`. |
| [DeepSynergy](https://www.bioinf.jku.at/software/DeepSynergy/) | Published O'Neil labels/folds, descriptors and expression inputs | No redistribution license was identified for these input assets. They are not included; download from the source using the documented commands and retain their provenance. |
| [NCI-ALMANAC](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/338237347/NCI-ALMANAC) | External screening checks | NCI public data; input data are downloaded separately. |

The repository distributes analysis code and derived scientific summaries, not original response datasets, downloaded feature archives, prediction checkpoints or model weights. Dataset URLs, input hashes and transformations are documented in [docs/datasets.md](docs/datasets.md), [manuscript/REPRODUCIBILITY.md](manuscript/REPRODUCIBILITY.md) and Supplement S14. Dependencies retain their own licenses. Published architectures are cited in the methods; no external model implementation was added by the published-method review.
