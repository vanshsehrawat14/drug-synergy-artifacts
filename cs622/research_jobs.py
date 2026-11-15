"""Frozen, targeted neural input ablations; no model search.

python -m cs622.research_jobs --task 0 --output results/research_20261107/runs
Tasks 0..44: three masks x three seeds x five held-out-pair folds.
"""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from .io import read_json, write_json, digest
from .runner import run


MASKS = {"no_drug_identity": {"drug_identity": False},
         "no_cell_identity": {"cell_identity": False},
         "no_dose": {"dose_input": False}}


def job(task):
    jobs = [(mask, seed, fold) for mask in MASKS for seed in [2026, 2027, 2028] for fold in range(5)]
    if not 0 <= task < len(jobs): raise ValueError("Task must be in 0..44")
    mask, seed, fold = jobs[task]
    config = deepcopy(read_json("configs/neural_monotherapy.json"))
    config.update(seed=seed, test_fold=fold)
    config["features"].update(mono_missing="interpolate", **MASKS[mask])
    config["training"].update(device="cuda", eval_batch_size=16384)
    return mask, seed, fold, config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--task",type=int,required=True)
    p.add_argument("--output",required=True);a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("This job requires an allocated CUDA GPU")
    mask,seed,fold,config=job(a.task)
    out=Path(a.output)/mask/f"seed_{seed}"/f"fold_{fold}"
    if (out/"metadata.json").exists():
        if read_json(out/"config.json") != config: raise ValueError("Existing run has different settings")
        print("Already completed",out);return
    last=out/"checkpoints/latest.pt"
    run(config,out,resume=str(last) if last.exists() else None)
    write_json(out/"research_job.json",{"task":a.task,"mask":mask,"seed":seed,"fold":fold,
        "protocol_sha256":digest("configs/research_20261107.json"),"control":"historical f2_base, same fold and seed",
        "input_note":"Identities remain available only to the training-pair monotherapy lookup when masked from network inputs."})


if __name__=="__main__": main()
