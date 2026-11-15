# Methodology

## Task and data

The primary task predicts measured inhibition at a drug-dose pair and cancer
cell line for completely withheld unordered drug pairs. The secondary task
studies how synergy definitions and numerical implementations change evaluation.
Predicting a computed synergy score does not establish biological interaction.

DrugComb v1.4 ONEIL has 1,475,328 positive-dose observations, 92,208 experimental
blocks, 583 pairs, 38 drugs and 39 cells. Inhibition is the exported percentage
response in percentage points, retained without clipping. Doses are in uM.
Drug identity, pair, cell, dose observation, block and replicate are different
units; all blocks/replicates of a pair stay in the same partition.

Sources and file checksums:

| Input | Source | SHA-256 |
|---|---|---|
| Dose grids | [DrugComb raw](https://zenodo.org/records/18449193) | `58c3893c9b8973eaea22aee819c0f268b3739d091a1d451187c1488c1058fe1d` |
| Block scores | [DrugComb summary](https://zenodo.org/records/11102665) | `e08e3d35bfa4bea011afe1b05b7025acde5c669c5e8d265866b5e12b08037a2f` |
| Labels/folds | [DeepSynergy](https://github.com/KristinaPreuer/DeepSynergy) | `4ef40678fff51faa84fa4de63344c5b51e2e4abe7c28a417af24ec2b50fa135a` |

`cs622.download` acquires files under `data/raw/`; `cs622.data` streams them
into `data/processed/oneil_v1/`. Preparation records source/output hashes,
settings, counts and exclusions in `manifest.json`. Do not edit raw files.
Drug names are uppercased with nonalphanumeric characters removed; ambiguous
mappings fail. Associated doses/identifiers swap with drug order. Cell names,
source row IDs, original coordinates and block IDs are retained for audit.
The export is curated data, not original instrument measurements.

## Splits, inputs and training

The five pair folds come from the published DeepSynergy labels. For test fold
f, validation is (f+1)%5 and the other three folds train. No pair may cross
partitions, including other cells, doses, orderings and replicates. Each held-out
drug must occur in other training pairs: this is not leave-one-drug-out.
Fold 0 has 888,576/295,296/291,456 train/validation/test rows.

- **F0:** drug/cell identities and associated doses, available before screening.
- **F1:** F0 and independently sourced molecular/cell features, with explicit
  provenance. Supported in code; the original feature matrices are unavailable.
- **F2:** F0 and single-agent values averaged at exact drug/cell/dose keys using
  training-pair block edges only. Missing values use the training lookup mean
  and missingness indicators. No held-out-pair edges enter the lookup.

F2 acquisition validity remains conditional. DrugComb describes separately
measured ONEIL monotherapies interpolated onto combination doses, but the full
original-assay-to-export processing path is not verified. Recover original
assays and interpolation/imputation/baseline settings before asserting fully
independent availability. All 3,360 imputed-input out-of-fold rows are retained:
F2 RMSE 26.451 versus F0 19.537 there. Missingness indicators never activate
during training. The cause is the training-mean imputation: with identical
training, log-dose interpolation of the same drug/cell values gives 19.6 there
([neural results](neural_results.md)).

Fit vocabularies, normalization, target scaling and imputation on training only.
F0/F2 settings: seed 2026, encoder 32, hidden layers 128/64, dropout 0.1,
Adam lr 0.001, weight decay 1e-5, batch 1024, maximum 200 epochs, patience 20.
Save strict validation-MSE improvements and evaluate the best checkpoint.
GP/CF choose their settings using the same validation groups and report their
search/runtime budgets. Never select a model or change features using test errors.

For one fold, use `python -m cs622.runner --config ... --output ...`.
For five sequential folds and pooled evaluation:

```sh
python -m cs622.cv run --config configs/neural_dose_aware.json --output results/neural/runs/f0_cv
python -m cs622.cv aggregate --runs results/neural/runs/f0_cv/fold_0 results/neural/runs/f0_cv/fold_1 results/neural/runs/f0_cv/fold_2 results/neural/runs/f0_cv/fold_3 results/neural/runs/f0_cv/fold_4 --output results/neural/runs/f0_cv/aggregate
```

Replace the config with `mean.json`, `additive.json` or
`neural_monotherapy.json` for the corresponding experiment. Do not repeat
completed production training just to verify installation. Runs record resolved
configuration, data/code hashes, dependencies, partition counts, histories,
checkpoint, runtime, predictions and separate evaluation artifacts.
Optional diagnostic plots require `python -m pip install -e ".[plots]"`.

## Shared prediction contract

Build a label-isolated bundle from the same config and test fold:

```sh
python -m cs622.export --config configs/neural_dose_aware.json --output data/processed/integration/f0_fold0
python -m cs622.integration --bundle data/processed/integration/f0_fold0 --output results/neural/runs/contract_example
```

The second command validates the bundle and submits shuffled training-mean dummy
predictions. It also verifies rejection of missing/duplicate/nonfinite submissions.
`train.parquet` and `validation.parquet` include inhibition; `test_features.parquet`
does not. F2 adds `training_monotherapy.parquet`, restricted to training pairs.
**Never use `evaluation/test_targets.parquet` for fitting, tuning or features.**
Hash checks need the source dataset at the path recorded in `contract.json`.

Each independent model returns a Parquet table with unique `observation_id` and
finite scalar `prediction` for **every** test row. Predictions are unclipped
inhibition in percentage points. Retain `pair_id`, `cell_line`, `fold`, drug
identities, `block_id`, associated doses and units for coordinate checks. Row
order is arbitrary. Record regime, target, seed, source revision and validation
selection with the submission. The evaluator aligns IDs and rejects incomplete
coverage and inconsistent supplied metadata:

```sh
python -m cs622.evaluate --predictions results/neural/runs/contract_example/predictions.parquet --observations data/processed/integration/f0_fold0/evaluation/test_targets.parquet --target inhibition --output results/neural/runs/contract_check
```

## Metrics and comparisons

For aligned finite outcomes y and predictions p, MSE = sum((y-p)^2)/n,
RMSE = sqrt(MSE), MAE = mean(abs(y-p)), and
R^2 = 1 - sum((y-p)^2)/sum((y-mean(y))^2). Pearson correlation is the centered
cross-product divided by the product of centered vector norms. Constant targets
have undefined R^2; constant vectors have undefined correlation, represented
as null. Invalid values and duplicate/missing IDs fail rather than disappearing.

Pool out-of-fold rows to compute micro metrics; macro-pair MSE averages pair
MSEs equally. Do not average fold R^2. Report pair/cell breakdowns and paired
loss changes on identical observations. Pair bootstrap intervals condition on
this screen and miss shared-drug dependence; delete-one-drug sensitivity does
not establish unseen-drug performance. Only 38 drugs and repeated cells/blocks
limit population inference. Five folds are not five independent datasets.

The optional retrospective interaction-contrast metric uses one block-averaged
row per pair/cell/fold: within each fold/cell let H be drug incidence and
P=I-HH^+. Contrast R^2 = 1-sum(||P(y-p)||^2)/sum(||Py||^2).
It is not a dose-row metric or an input feature; zero residual energy is undefined.

## Existing results and limitations

Completed five-fold pooled response RMSE: mean 33.961, additive 24.176,
F0 neural 13.000, F2 neural 12.277. These are development results from an already
examined screen, not independent confirmation. F2 has additional information;
the difference is not a pure architecture comparison. Small per-fold metrics
and prediction/dataset hashes are retained in [response_metrics.csv](../results/response_metrics.csv).
The frozen-protocol GPU benchmark and the DeepSynergy reproduction are reported in
[neural_results.md](neural_results.md).
Original runs, checkpoints, scoring analyses and full records remain on the
preserved `feat/cs622-neural-pipeline` development branch/local data, not in this
team snapshot. No production predictions or scientific results were changed.

Bliss reference is a+b-ab/100; HSA reference is max(a,b). Scores subtract a
reference from measured response. Within-block edges may be used retrospectively
to reconstruct scores, never as held-out predictive features. Published Loewe,
ZIP and reconstructed Bliss/HSA are distinct targets. For an exact known offset,
subtracting it from both truth and prediction preserves squared error, while
R^2 can change. Cross-target R^2 is not directly comparable biological information.

Native Loewe forward reproduction remains approximate (full cached-fit block
RMSE 0.704), with original generating fits/settings unresolved. The Python solver
proxy failed native equivalence. Ordinary Bliss/Loewe reference disagreement
does not itself prove a numerical artifact. No stable model-ranking reversal
or independent-study confirmation is established. Detailed scorer investigations
are retained separately from the shared ML implementation.

`cs622.deepsynergy reproduce` runs the published protocol on the original chemical
and expression features; its audit is in [neural_results.md](neural_results.md).
`deepsynergy_paper.json` is an uncompleted adaptation configuration, not a faithful
numerical reproduction. `neural_onehot.json` predicts published synergy, a
different task from dose-level response. Keep these result tables separate.

For future confirmation, agree on hypotheses, eligible data, exclusions, code,
features, model selection and primary metrics before inspecting outcomes.
ONEIL and previously examined ALMANAC are not untouched confirmation cohorts.
Data source terms and a project license still require review before external
redistribution. Datasets and third-party solver code are not bundled here.

References: [O'Neil 2016](https://doi.org/10.1158/1535-7163.MCT-15-0843),
[DeepSynergy 2018](https://doi.org/10.1093/bioinformatics/btx806),
[DrugComb 2019](https://doi.org/10.1093/nar/gkz337),
[SynergyFinder 2.0](https://doi.org/10.1093/nar/gkaa216).
