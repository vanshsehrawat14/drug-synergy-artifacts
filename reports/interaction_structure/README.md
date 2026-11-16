# Interaction gap: what same-pair information supplies

This is a CPU-only analysis of why held-out-pair models miss interaction structure that same-pair information in other cell lines recovers. It uses the O'Neil screen, the frozen pair folds, the frozen resolution cache and the saved pair-seen results. No model was trained.

The design was frozen before outcomes were computed, in `experiments/interaction_structure/design.json` (commit c0a785c).

| Document | Content |
|---|---|
| `REPORT.md` | Full report |
| `TARGET_AND_LEAKAGE_AUDIT.md` | Units, target algebra, reference sharing, replicate dependence, reproduced noise estimates, leakage |
| `VARIANCE_COMPONENTS.md` | Cross-fitted components, predictable shares, controls, within- versus cross-batch reproducibility |
| `COMPARATIVE_CONTROLS.md` | Anchor curves, B0 to B5 contrasts, matched pair-seen comparison, related-work boundary |

Outputs live in `results/interaction_structure/`:
- `variance_components.json`, `anchor_experiment.json`, `grounding.json`, `batch_transfer.json`, `dose_shape_check.json`;
- `ANCHOR_INFORMATION_CURVE.csv`: target, k, anchor seed, fold, model, evaluated pair-cells, SSE, SSE(B0), MSE, RMSE, R², pair-macro R², noise share;
- `anchor_pairs.csv.gz`: per pair, seed-averaged;
- `figures/`;
- `MANIFEST.json`: SHA-256 of inputs, scripts and outputs.

Per-pair-cell targets and the cross-batch values are derived from the screen data and are not committed. The scripts regenerate them with `--derived`.

## Reproduce

Run from the repository root:
- `RES` is the frozen resolution cache: `results/interaction_analysis/resolution` on branch research/resolution-reliability (d2c8f41), with `cache.npz` SHA-256 `91f93811…6687`.
- `DERIVED` is any directory outside the repository.

```
OBS=data/processed/oneil_v1/observations.parquet
OUT=results/interaction_structure
X=experiments/interaction_structure
python $X/variance.py --observations $OBS --resolution $RES --output $OUT --derived $DERIVED
python $X/anchors.py --observations $OBS --resolution $RES --split results/pair_information/oracle/pair_cell_split.csv \
    --pairseen manuscript/audit/outputs/pairseen_audit.json --output $OUT
python $X/grounding.py --observations $OBS --resolution $RES --features data/processed/deepsynergy/features.npz --output $OUT
python $X/batch_transfer.py --observations $OBS --resolution $RES --output $OUT
python $X/dose_shape_check.py --observations $OBS --resolution $RES --output $OUT
python $X/figures.py --results $OUT --cross-batch $DERIVED/cross_batch_values.csv
python $X/manifest.py --resolution $RES --output $OUT
python -m pytest tests/test_interaction_gap.py
```

Total runtime is about five minutes on 16 CPU threads. All randomness is seeded:
- block signs: 97;
- pair bootstrap: 11;
- cross-batch bootstrap: 13;
- permutations: 17;
- zero-interaction realizations: 19;
- reference perturbations: 23;
- Mantel tests: 29 to 41;
- batch-transfer bootstrap: 43;
- anchor masks: SHA-256 of (pair_id, seed).
