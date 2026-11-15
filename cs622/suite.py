"""Frozen neural sprint: validation-selected candidates, then multi-seed five-fold runs.

python -m cs622.suite plan --protocol configs/neural_sprint.json
python -m cs622.suite run --protocol configs/neural_sprint.json --output results/neural/runs/sprint_v1 --stage all --device cuda
python -m cs622.suite pack --output results/neural/runs/sprint_v1 --archive sprint_v1.zip

Stage "select" trains every candidate once (selection seed) on all five folds.
Selection is mechanical: per outer fold, the lowest validation MSE within a group.
Stage "benchmark" adds seeds for the declared models. Completed runs are skipped;
an interrupted run resumes from its last epoch checkpoint.
"""
import argparse
import copy
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

from .io import read_json, write_json
from .runner import git_state, run

SMOKE = {"subset": {"cells": ["A2058", "A2780"], "blocks_per_pair_cell": 1},
         "training": {"epochs": 2, "patience": 3, "batch_size": 256}, "bootstrap": 10}


def merge(base, overrides):
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def protocol_hash(protocol):
    return hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()


def candidate_config(protocol, name, fold, seed, device, budget=None):
    spec = protocol["candidates"][name]
    config = merge(read_json(protocol["base"][spec["base"]]), protocol.get("shared_overrides", {}))
    config = merge(config, spec["overrides"])
    config.update(test_fold=fold, seed=seed)
    config["training"]["device"] = device
    return merge(config, budget) if budget else config


def run_dir(root, name, seed, fold):
    return Path(root) / "runs" / name / f"seed_{seed}" / f"fold_{fold}"


def complete(path):
    return (path / "metadata.json").exists() and (path / "evaluation" / "metrics.json").exists()


def selection_runs(protocol, folds):
    seed = protocol["selection"]["seed"]
    return [(name, seed, fold) for name in protocol["candidates"] for fold in folds]


def select(protocol, root, folds):
    seed = protocol["selection"]["seed"]
    table = {}
    for group, names in protocol["selection"]["groups"].items():
        for fold in folds:
            scores = []
            for name in names:
                path = run_dir(root, name, seed, fold)
                if not complete(path):
                    raise ValueError(f"Selection requires the completed run {path}")
                fit = read_json(path / "metadata.json")["fit"]
                scores.append((fit["validation_mse"], fit["parameters"], name))
            score, _, chosen = min(scores)
            table.setdefault(group, {})[str(fold)] = {"selected": chosen, "validation_mse": {n: s for s, _, n in scores}}
    return table


def benchmark_runs(protocol, chosen, seeds, folds):
    runs = []
    for model in protocol["benchmark"]["models"]:
        for fold in folds:
            name = chosen[model.split(":", 1)[1]][str(fold)]["selected"] if model.startswith("selected:") else model
            runs += [(name, seed, fold) for seed in seeds if (name, seed, fold) not in runs]
    return runs


def _archive(path, root):
    target = Path(root) / "failed" / f"{path.parent.parent.name}_{path.parent.name}_{path.name}_{time.strftime('%Y%m%d%H%M%S')}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(target))
    return target


def _log(root, record):
    with open(Path(root) / "progress.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def execute(protocol, root, runs, device, shard=(0, 1), budget=None):
    index, count = shard
    for i, (name, seed, fold) in enumerate(runs):
        path = run_dir(root, name, seed, fold)
        if i % count != index or complete(path):
            continue
        config = candidate_config(protocol, name, fold, seed, device, budget)
        latest = path / "checkpoints" / "latest.pt"
        resume = latest if latest.exists() else None
        if path.exists() and resume is None:
            _archive(path, root)
        started = time.perf_counter()
        print(f"[{i + 1}/{len(runs)}] {name} seed={seed} fold={fold}{' (resume)' if resume else ''}", flush=True)
        try:
            run(config, path, resume=resume)
        except ValueError:
            if resume is None:
                raise
            # A checkpoint that cannot be resumed (e.g. exhausted budget) is retained, then rerun.
            _archive(path, root)
            run(config, path)
        fit = read_json(path / "metadata.json")["fit"]
        _log(root, {"candidate": name, "seed": seed, "fold": fold, "seconds": time.perf_counter() - started,
                    "validation_mse": fit["validation_mse"], "epochs": fit["epochs_completed"], "resumed": bool(resume)})


def wait_for(root, runs, hours=24, poll=60):
    """Sharded processes reach selection only after every shard's selection runs finish."""
    deadline = time.time() + hours * 3600
    while True:
        pending = [r for r in runs if not complete(run_dir(root, *r))]
        if not pending:
            return
        if time.time() > deadline:
            raise TimeoutError(f"{len(pending)} selection runs still incomplete, e.g. {pending[0]}")
        print(f"Waiting for {len(pending)} selection runs from other shards", flush=True)
        time.sleep(poll)


def initialize(protocol, root, budget, folds):
    root = Path(root)
    manifest = {"protocol": protocol, "protocol_sha256": protocol_hash(protocol), "budget": budget, "folds": list(folds),
                "base_configs": {k: read_json(v) for k, v in protocol["base"].items()}}
    path = root / "suite.json"
    if path.exists():
        existing = read_json(path)
        if {k: existing.get(k) for k in manifest} != manifest:
            raise ValueError("Output belongs to a different protocol, budget or fold set")
        return
    root.mkdir(parents=True, exist_ok=True)
    write_json(path, {**manifest, "git": git_state(), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")})


def pack(root, archive):
    root, archive = Path(root), Path(archive)
    if archive.exists():
        raise FileExistsError(archive)
    def keep(p):
        if p.name == "best.pt":
            return True
        return p.suffix in {".json", ".jsonl", ".parquet", ".log"} and p.name != "predictions_with_truth.parquet"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file() and keep(p):
                z.write(p, p.relative_to(root.parent))
    print(f"Packed {archive} ({archive.stat().st_size / 1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--protocol", default="configs/neural_sprint.json")
    go = sub.add_parser("run")
    go.add_argument("--protocol", default="configs/neural_sprint.json")
    go.add_argument("--output", required=True)
    go.add_argument("--stage", choices=["select", "benchmark", "all"], default="all")
    go.add_argument("--device", default="cuda")
    go.add_argument("--shard", default="0/1", help="i/n: this process runs every n-th pending run, offset i")
    go.add_argument("--extension-seeds", action="store_true", help="also run the declared extension seeds")
    go.add_argument("--smoke", action="store_true", help="tiny real-data subset and 2 epochs; engineering check only")
    go.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    out = sub.add_parser("pack")
    out.add_argument("--output", required=True)
    out.add_argument("--archive", required=True)
    a = p.parse_args()
    if a.command == "pack":
        return pack(a.output, a.archive)
    protocol = read_json(a.protocol)
    if a.command == "plan":
        folds = range(5)
        stage1 = selection_runs(protocol, folds)
        dummy = {g: {str(f): {"selected": names[0]} for f in folds} for g, names in protocol["selection"]["groups"].items()}
        seeds = protocol["benchmark"]["seeds"]
        stage2 = [r for r in benchmark_runs(protocol, dummy, seeds, folds) if r not in stage1]
        print(json.dumps({"protocol_sha256": protocol_hash(protocol), "selection_runs": len(stage1),
                          "benchmark_runs_upper_bound": len(stage2) + 2 * 5 * (len(seeds) - 1),
                          "candidates": list(protocol["candidates"])}, indent=2))
        return
    shard = tuple(int(x) for x in a.shard.split("/"))
    budget = SMOKE if a.smoke else None
    initialize(protocol, a.output, budget, a.folds)
    if a.stage in {"select", "all"}:
        execute(protocol, a.output, selection_runs(protocol, a.folds), a.device, shard, budget)
    if a.stage in {"benchmark", "all"}:
        if shard[1] > 1:
            wait_for(a.output, selection_runs(protocol, a.folds))
        chosen = select(protocol, a.output, a.folds)
        temporary = Path(a.output) / f"selection.{shard[0]}.tmp"
        write_json(temporary, {"rule": protocol["selection"], "selected": chosen})
        temporary.replace(Path(a.output) / "selection.json")
        seeds = protocol["benchmark"]["seeds"] + (protocol["benchmark"]["extension_seeds"] if a.extension_seeds else [])
        execute(protocol, a.output, benchmark_runs(protocol, chosen, seeds, a.folds), a.device, shard, budget)
    print(f"Stage {a.stage} finished for shard {a.shard}: {a.output}")


if __name__ == "__main__":
    main()
