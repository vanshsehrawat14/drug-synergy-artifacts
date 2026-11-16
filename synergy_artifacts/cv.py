"""Sequential central-fold execution and strict out-of-fold aggregation."""
import argparse
import copy
from pathlib import Path

import pandas as pd

from .io import read_json, write_json
from .metrics import drug_jackknife, evaluate
from .runner import run


def run_folds(config, output, folds=range(5), epochs=None):
    for fold in folds:
        current = copy.deepcopy(config)
        current["test_fold"] = fold
        if epochs is not None:
            current["training"]["epochs"] = epochs
        run(current, Path(output) / f"fold_{fold}")


def aggregate(runs, output, allow_partial=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    frames, signatures, folds, sources, implementations = [], [], [], [], []
    for path in map(Path, runs):
        meta = read_json(path / "metadata.json")
        config = copy.deepcopy(meta["config"])
        folds.append(config.pop("test_fold"))
        signatures.append(config)
        sources.append(meta["dataset_sha256"])
        implementations.append(meta["code_sha256"])
        frame = pd.read_parquet(path / "evaluation" / "predictions_with_truth.parquet")
        if set(frame.fold) != {folds[-1]}:
            raise ValueError("Run predictions contain the wrong test fold")
        frames.append(frame)
    if (not signatures or any(s != signatures[0] for s in signatures) or len(set(sources)) != 1
            or any(code != implementations[0] for code in implementations)):
        raise ValueError("Cannot pool incompatible datasets, targets, feature regimes or model budgets")
    if len(folds) != len(set(folds)):
        raise ValueError("Duplicate test folds")
    complete = set(folds) == set(range(5))
    if not complete and not allow_partial:
        raise ValueError("Incomplete cross-validation; provide all five folds or --allow-partial")
    frame = pd.concat(frames, ignore_index=True)
    if frame.groupby("pair_id", observed=True).fold.nunique().max() != 1:
        raise ValueError("Pair leakage across out-of-fold predictions")
    config = signatures[0]
    metrics = evaluate(frame, config["target"], config.get("bootstrap", 1000), config["seed"],
                       contrast=config["table"] in {"panel", "published"})
    metrics["drug_jackknife"] = drug_jackknife(frame, config["target"])
    report = {"complete_five_fold_cv": complete, "folds": sorted(folds), "config": config,
              "runs": [str(p) for p in runs], "metrics": metrics}
    evaluation = output / "evaluation"
    evaluation.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(evaluation / "predictions_with_truth.parquet", index=False)
    write_json(output / "aggregate.json", report)
    print({"complete_five_fold_cv": complete, "folds": sorted(folds), **metrics["pooled"]})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("run")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    train.add_argument("--epochs", type=int)
    pool = sub.add_parser("aggregate")
    pool.add_argument("--runs", nargs="+", required=True)
    pool.add_argument("--output", required=True)
    pool.add_argument("--allow-partial", action="store_true")
    a = p.parse_args()
    if a.command == "run":
        run_folds(read_json(a.config), a.output, a.folds, a.epochs)
    else:
        aggregate(a.runs, a.output, a.allow_partial)


if __name__ == "__main__":
    main()
