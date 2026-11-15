"""Pair-cell folds and the frozen pair-seen F2 control (configs/final_20261110.json, direction B2).

python -m cs622.pair_seen --task 0 --output results/final_20261110/runs
Tasks 0..14: seeds 2026-2028 by five pair-cell test folds. Every pair-cell, with all of its
blocks, lies in one fold; a test pair is therefore seen in other cells during training.
"""
import argparse
import hashlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from .io import read_json, write_json, digest

SPLIT_SEED = 20261110
SEEDS = [2026, 2027, 2028]


def pair_cell_folds(frame, seed=SPLIT_SEED, n_folds=5):
    """Per pair, sorted cells are permuted and dealt round-robin from a random offset."""
    units = frame[["pair_id", "cell_line"]].astype(str).drop_duplicates().sort_values(["pair_id", "cell_line"])
    rng = np.random.default_rng(seed)
    parts = []
    for pair, g in units.groupby("pair_id", sort=True):
        cells = g.cell_line.to_numpy()
        order, offset = rng.permutation(len(cells)), int(rng.integers(n_folds))
        fold = np.empty(len(cells), dtype=int)
        fold[order] = (np.arange(len(cells)) + offset) % n_folds
        parts.append(pd.DataFrame({"pair_id": pair, "cell_line": cells, "fold": fold}))
    return pd.concat(parts, ignore_index=True)


def assignment_sha256(assignment):
    text = assignment.sort_values(["pair_id", "cell_line"]).to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def assign(frame, seed=SPLIT_SEED):
    """Replace the published pair folds by pair-cell folds; rows keep their order."""
    folds = pair_cell_folds(frame, seed)
    key = pd.MultiIndex.from_arrays([frame.pair_id.astype(str), frame.cell_line.astype(str)])
    lookup = pd.Series(folds.fold.to_numpy(), index=pd.MultiIndex.from_frame(folds[["pair_id", "cell_line"]]))
    out = frame.copy()
    out["fold"] = lookup.reindex(key).to_numpy()
    if out.fold.isna().any():
        raise ValueError("Unassigned pair-cell")
    out["fold"] = out.fold.astype(int)
    return out


def job(task):
    jobs = [(seed, fold) for seed in SEEDS for fold in range(5)]
    if not 0 <= task < len(jobs):
        raise ValueError("Task must be in 0..14")
    seed, fold = jobs[task]
    config = deepcopy(read_json("configs/neural_monotherapy.json"))
    config.update(seed=seed, test_fold=fold)
    config["features"].update(mono_missing="interpolate")
    config["training"].update(device="cuda", eval_batch_size=16384)
    config["split"] = {"kind": "pair_cell", "seed": SPLIT_SEED}
    return seed, fold, config


def control_config(config):
    """The held-out-pair f2_base configuration this task is matched to."""
    control = deepcopy(config)
    control.pop("split")
    return control


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--controls", default="results/neural/runs/sprint_v1/runs/f2_base")
    a = p.parse_args()
    import torch
    from .runner import run
    if not torch.cuda.is_available():
        raise RuntimeError("This job requires an allocated CUDA GPU")
    seed, fold, config = job(a.task)
    control = read_json(Path(a.controls) / f"seed_{seed}" / f"fold_{fold}" / "config.json")
    if control != control_config(config):
        raise ValueError("Pair-seen task differs from its f2_base control beyond the split")
    out = Path(a.output) / "f2_pair_seen" / f"seed_{seed}" / f"fold_{fold}"
    if (out / "metadata.json").exists():
        if read_json(out / "config.json") != config:
            raise ValueError("Existing run has different settings")
        print("Already completed", out)
        return
    last = out / "checkpoints" / "latest.pt"
    run(config, out, resume=str(last) if last.exists() else None)
    write_json(out / "pair_seen_job.json", {"task": a.task, "seed": seed, "fold": fold,
               "protocol_sha256": digest("configs/final_20261110.json"),
               "control": f"f2_base seed {seed} fold {fold}, identical except the split"})


if __name__ == "__main__":
    main()
