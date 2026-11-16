"""Preflight for the 15 pair-seen tasks: configs, controls, split hash and leakage on the real data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from synergy_artifacts.io import read_json  # noqa: E402
from synergy_artifacts.pair_seen import SPLIT_SEED, assignment_sha256, control_config, job, pair_cell_folds  # noqa: E402
from synergy_artifacts.runner import load_experiment  # noqa: E402
from synergy_artifacts.splits import partitions  # noqa: E402

EXPECTED_SPLIT = "d32914a94d8ead90e12d8cb10b6beed15f93339b0de2200ce2075a67a79370ea"

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--controls", required=True)
a = p.parse_args()
report = {}
for task in range(15):
    seed, fold, config = job(task)
    control = Path(a.controls) / f"seed_{seed}" / f"fold_{fold}"
    if read_json(control / "config.json") != control_config(config) or not (control / "predictions.parquet").exists():
        raise SystemExit(f"Control mismatch or incomplete: {control}")
_, _, config = job(0)
df, mono, _ = load_experiment(config)
if assignment_sha256(pair_cell_folds(df, SPLIT_SEED)) != EXPECTED_SPLIT:
    raise SystemExit("Split assignment hash differs")
for fold in range(5):
    tr, va, te = partitions(df, fold, unit="pair_cell")
    key = lambda ix: set(zip(df.pair_id.iloc[ix].astype(str), df.cell_line.iloc[ix].astype(str)))
    if key(te) & (key(tr) | key(va)) or key(va) & key(tr):
        raise SystemExit("Pair-cell leakage")
    blocks = set(df.block_id.iloc[tr])
    if set(df.block_id.iloc[te]) & blocks:
        raise SystemExit("Block leakage")
    train_mono = mono[mono.block_id.isin(blocks)]
    drug_cell = set(zip(train_mono.drug_a.astype(str), train_mono.cell_line.astype(str))) | set(zip(train_mono.drug_b.astype(str), train_mono.cell_line.astype(str)))
    test = df.iloc[te]
    need = set(zip(test.drug_a.astype(str), test.cell_line.astype(str))) | set(zip(test.drug_b.astype(str), test.cell_line.astype(str)))
    report[fold] = dict(train=len(tr), validation=len(va), test=len(te), test_drug_cells_without_training_monotherapy=len(need - drug_cell))
print(json.dumps(dict(tasks=15, split_sha256=EXPECTED_SPLIT, folds=report), indent=1))
