"""Build run_ledger.csv for the 45 frozen input ablations (run on RebelX from the clone root)."""
import csv, glob, hashlib, json, os, re, sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from synergy_artifacts.research_jobs import job  # frozen task -> config mapping

ALLOC = Path.home() / "scratch/synergy_artifacts/research_d2c8f41/alloc_132664.log"
RUNS = Path("results/interaction_analysis/runs")
SLURM = Path("results/interaction_analysis/slurm")
ARRAY = "132664"


def canon(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def alloc_records():
    """Last observed scontrol record per array task."""
    rec = {}
    if not ALLOC.exists():
        return rec
    text = ALLOC.read_text()
    for block in re.split(r"\n(?=JobId=)", text):
        m = re.search(r"JobId=(\d+) ArrayJobId=%s ArrayTaskId=(\d+)\b" % ARRAY, block)
        if not m:
            continue
        r = rec.setdefault(int(m.group(2)), {})
        r["job_id"] = m.group(1)
        for key in ["JobState", "StartTime", "EndTime", "ExitCode"]:
            v = re.search(key + r"=(\S+)", block)
            if v:
                r[key] = v.group(1)
        g = re.search(r"Nodes=(\S+) .*GRES=gpu:1\(IDX:(\d+)\)", block)
        if g:
            r["node"], r["gpu_idx"] = g.group(1), g.group(2)
    return rec


def main(out):
    alloc = alloc_records()
    queued = os.popen("squeue -h -j %s -o '%%i %%T' 2>/dev/null" % ARRAY).read()
    rows = []
    for task in range(45):
        mask, seed, fold, config = job(task)
        d = RUNS / mask / f"seed_{seed}" / f"fold_{fold}"
        meta = d / "metadata.json"
        saved = d / "config.json"
        saved_hash = canon(json.loads(saved.read_text())) if saved.exists() else ""
        a = alloc.get(task, {})
        log = SLURM / f"{ARRAY}_{task}.out"
        elapsed = ""
        if meta.exists():
            elapsed = json.loads(meta.read_text()).get("elapsed_seconds", "")
        complete = meta.exists() and (d / "predictions.parquet").exists() and (d / "research_job.json").exists()
        running = f"{ARRAY}_{task} RUNNING" in queued
        status = "complete" if complete else "running" if running else (
            "failed" if log.exists() else "unstarted")
        rows.append(dict(
            task_id=task, mask=mask, seed=seed, fold=fold, target=config["target"],
            features=json.dumps(config["features"], sort_keys=True),
            config_sha256=canon(config), saved_config_sha256=saved_hash,
            config_match=(saved_hash == canon(config)) if saved_hash else "",
            intended_output=str(d), status=status, slurm_array=ARRAY,
            slurm_job_id=a.get("job_id", ""), node=a.get("node", ""), gpu_idx=a.get("gpu_idx", ""),
            launch_time=a.get("StartTime", ""), end_time=a.get("EndTime", "") if a.get("JobState") in
            ("COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL") else "",
            scheduler_state=a.get("JobState", ""), exit_code=a.get("ExitCode", ""),
            training_seconds=elapsed, retry_reason="",
            predictions_sha256=sha(d / "predictions.parquet"), metadata_sha256=sha(meta),
            log=str(log) if log.exists() else "", log_sha256=sha(log)))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(counts, "config mismatches:", [r["task_id"] for r in rows if r["config_match"] is False])


if __name__ == "__main__":
    main(sys.argv[1])
