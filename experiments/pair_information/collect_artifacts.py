"""Collect the pair-seen runs and Slurm logs with per-file hashes and task accounting."""
import argparse
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path


def collect(repo, output, checkpoints=False):
    repo, output = Path(repo).resolve(), Path(output).resolve()
    root = repo / "results/pair_information"
    sys.path.insert(0, str(repo))
    from synergy_artifacts.pair_seen import job
    if output.exists():
        raise FileExistsError(output)
    tasks, files = [], []
    for task in range(15):
        seed, fold, config = job(task)
        d = root / "runs" / "f2_pair_seen" / f"seed_{seed}" / f"fold_{fold}"
        complete = all((d / k).exists() for k in ["predictions.parquet", "config.json", "metadata.json"])
        if complete and json.loads((d / "config.json").read_text()) != config:
            raise ValueError(f"Config differs: {d}")
        tasks.append(dict(task=task, seed=seed, fold=fold, complete=complete))
    for p in sorted(x for folder in [root / "runs", root / "slurm"] if folder.exists() for x in folder.rglob("*")):
        if not p.is_file() or p.is_symlink() or p.name in {"predictions_with_truth.parquet", "latest.pt"}:
            continue
        if p.suffix in {".json", ".csv", ".out", ".log"} or p.name == "predictions.parquet" or (checkpoints and p.name == "best.pt"):
            files.append(p)
    manifest = dict(tasks=tasks, complete_tasks=sum(x["complete"] for x in tasks), files=[])
    with tarfile.open(output, "w:gz") as tar:
        for p in files:
            name = p.relative_to(repo).as_posix()
            manifest["files"].append(dict(path=name, bytes=p.stat().st_size, sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest()))
            tar.add(p, arcname=name, recursive=False)
        data = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("artifact_manifest.json"); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(json.dumps(dict(archive=str(output), bytes=output.stat().st_size,
                          sha256=hashlib.file_digest(output.open("rb"), "sha256").hexdigest(),
                          complete_tasks=manifest["complete_tasks"], incomplete=[x["task"] for x in tasks if not x["complete"]]), indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="."); p.add_argument("--output", required=True); p.add_argument("--checkpoints", action="store_true")
    a = p.parse_args(); collect(a.repo, a.output, a.checkpoints)
