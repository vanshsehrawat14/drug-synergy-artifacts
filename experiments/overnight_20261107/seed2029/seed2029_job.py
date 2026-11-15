"""Phase 3.4 confirmatory seed (preregistered_analyses.md section 6): the frozen input-ablation configs at seed 2029,
plus the matched full-input F2 control (historical sprint_v1 f2_base config at seed 2029). Nothing else changes.

Run from the d2c8f41 clone (tree stays clean):
  python seed2029_job.py --task K --output results/research_20261107/seed2029 --historical <sprint_v1 f2_base dir>
  python seed2029_job.py --check-only --historical <dir>   # verifies all 20 configs, no training
Tasks 0-4: f2_base folds 0-4; 5-19: no_drug_identity, no_cell_identity, no_dose, folds 0-4 each.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from cs622.io import read_json, write_json, digest  # noqa: E402
from cs622.research_jobs import MASKS, job  # noqa: E402

SEED = 2029
MODELS = ["f2_base"] + list(MASKS)


def config(task, historical):
    if not 0 <= task < 20:
        raise ValueError("Task must be in 0..19")
    model, fold = MODELS[task // 5], task % 5
    mask = model if model in MASKS else "no_drug_identity"
    m, s, f, cfg = job(list(MASKS).index(mask) * 15 + fold)  # frozen seed-2026 task, same mask and fold
    assert (m, s, f) == (mask, 2026, fold)
    cfg["seed"] = SEED
    if model == "f2_base":
        for k in MASKS[mask]:
            del cfg["features"][k]
        ref = read_json(Path(historical) / "seed_2026" / f"fold_{fold}" / "config.json")
    else:
        ref = read_json(Path("results/research_20261107/runs") / mask / "seed_2026" / f"fold_{fold}" / "config.json")
    if dict(ref, seed=SEED) != cfg:
        raise ValueError(f"Task {task}: config differs from its frozen reference beyond the seed")
    return model, fold, cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", type=int, default=-1)
    p.add_argument("--output", default="results/research_20261107/seed2029")
    p.add_argument("--historical", required=True)
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()
    if a.check_only:
        for t in range(20):
            model, fold, _ = config(t, a.historical)
            print(t, model, SEED, fold, "config = frozen reference with seed 2029")
        return
    import torch
    from cs622.runner import run
    if not torch.cuda.is_available():
        raise RuntimeError("This job requires an allocated CUDA GPU")
    model, fold, cfg = config(a.task, a.historical)
    out = Path(a.output) / model / f"seed_{SEED}" / f"fold_{fold}"
    if (out / "metadata.json").exists():
        if read_json(out / "config.json") != cfg:
            raise ValueError("Existing run has different settings")
        print("Already completed", out)
        return
    last = out / "checkpoints/latest.pt"
    run(cfg, out, resume=str(last) if last.exists() else None)
    write_json(out / "research_job.json", {
        "task": a.task, "model": model, "seed": SEED, "fold": fold,
        "phase": "3.4 confirmatory seed, preregistered_analyses.md section 6",
        "protocol_sha256": digest("configs/research_20261107.json"), "driver_sha256": digest(__file__),
        "control": "f2_base seed 2029 trained by this driver, same fold" if model != "f2_base" else "matched control"})


if __name__ == "__main__":
    main()
