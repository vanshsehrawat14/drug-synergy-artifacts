"""Export the same split contract for teammate algorithms, with isolated test labels."""
import argparse
from pathlib import Path

from .io import digest, read_json, write_json
from .runner import load_experiment
from .splits import partitions


def export_bundle(config, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    frame, mono, source = load_experiment(config)
    tr, va, te = partitions(frame, config["test_fold"])
    output.mkdir(parents=True, exist_ok=True)
    frame.iloc[tr].to_parquet(output / "train.parquet", index=False)
    frame.iloc[va].to_parquet(output / "validation.parquet", index=False)
    target = config["target"]
    frame.iloc[te].drop(columns=target).to_parquet(output / "test_features.parquet", index=False)
    evaluation = output / "evaluation"
    evaluation.mkdir(exist_ok=True)
    frame.iloc[te].to_parquet(evaluation / "test_targets.parquet", index=False)
    if config["features"]["regime"] == "F2":
        if mono is None:
            raise ValueError("F2 export requires dose data")
        mono = mono[mono.pair_id.isin(frame.iloc[tr].pair_id)]
        mono.to_parquet(output / "training_monotherapy.parquet", index=False)
    write_json(output / "contract.json", {"config": config, "source_sha256": digest(source),
        "target": target, "split_unit": "canonical unordered drug pair",
        "partition_rows": dict(zip(["train", "validation", "test"], map(len, [tr, va, te]))),
        "test_labels": "evaluation/test_targets.parquet; evaluation only",
        "prediction_contract": "unique observation_id and finite scalar prediction for every test feature row",
        "artifacts": {str(p.relative_to(output)): digest(p) for p in output.rglob("*.parquet")}})
    print(f"Exported {target} fold {config['test_fold']} to {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    export_bundle(read_json(args.config), args.output)


if __name__ == "__main__":
    main()
