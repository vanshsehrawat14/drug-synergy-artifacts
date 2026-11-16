"""Verify an exported teammate contract and exercise the strict evaluator."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synergy_artifacts.evaluate import evaluate_files
from synergy_artifacts.features import FeatureBuilder
from synergy_artifacts.io import digest, read_json, write_json
from synergy_artifacts.metrics import align_predictions
from synergy_artifacts.runner import load_experiment
from synergy_artifacts.splits import partitions


def validate_bundle(bundle):
    bundle = Path(bundle)
    contract = read_json(bundle / "contract.json")
    for name, expected in contract["artifacts"].items():
        if digest(bundle / name) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    source, mono, path = load_experiment(contract["config"])
    if digest(path) != contract["source_sha256"]:
        raise ValueError("Source provenance differs")
    target = contract["target"]
    if target != "inhibition" or contract["config"]["table"] != "observations":
        raise ValueError("Expected measured dose-level inhibition")
    frames = {name: pd.read_parquet(bundle / filename) for name, filename in
              [("train", "train.parquet"), ("validation", "validation.parquet"),
               ("test", "test_features.parquet")]}
    truth = pd.read_parquet(bundle / "evaluation/test_targets.parquet")
    if target in frames["test"] or any(c.startswith("synergy") for c in frames["test"]):
        raise ValueError("Held-out labels in test features")
    groups = []
    for (name, frame), indices in zip(frames.items(), partitions(source, contract["config"]["test_fold"])):
        expected = source.iloc[indices].copy()
        if name == "test":
            expected.drop(columns=target, inplace=True)
        if len(frame) != contract["partition_rows"][name] or frame.observation_id.duplicated().any():
            raise ValueError("Partition size or identifiers differ")
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected.reset_index(drop=True))
        group = set(frame.pair_id)
        if any(group & previous for previous in groups):
            raise ValueError("Pair crosses partitions")
        groups.append(group)
    pd.testing.assert_frame_equal(truth.drop(columns=target), frames["test"])
    pd.testing.assert_series_equal(truth[target], source.iloc[partitions(source, contract["config"]["test_fold"])[2]][target].reset_index(drop=True))
    coverage = None
    if contract["config"]["features"]["regime"] == "F2":
        exported = pd.read_parquet(bundle / "training_monotherapy.parquet")
        expected = mono[mono.pair_id.isin(groups[0])]
        pd.testing.assert_frame_equal(exported.reset_index(drop=True), expected.reset_index(drop=True))
        builder = FeatureBuilder(regime="F2").fit(frames["train"], exported)
        coverage = {name: builder.monotherapy_coverage(frame) for name, frame in frames.items()}
    return {"bundle": str(bundle), "source_sha256": digest(path), "target": target,
            "regime": contract["config"]["features"]["regime"],
            "partition_rows": {k: len(v) for k, v in frames.items()},
            "partition_pairs": dict(zip(frames, map(len, groups))),
            "checks": "checksums, exact source rows, pair isolation, test-label isolation, F2 training-only edges",
            "coverage": coverage}


def example(bundle, output):
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError(output)
    verified = validate_bundle(bundle)
    output.mkdir(parents=True)
    # This mock implementation reads only training labels and test features.
    train = pd.read_parquet(bundle / "train.parquet")
    features = pd.read_parquet(bundle / "test_features.parquet")
    prediction = features[["observation_id", "pair_id", "cell_line", "fold", "dose_a", "dose_b"]].copy()
    prediction["prediction"] = float(train.inhibition.mean())
    prediction = prediction.sample(frac=1, random_state=2026).reset_index(drop=True)
    prediction.to_parquet(output / "predictions.parquet", index=False)
    # Only the separate evaluation step opens held-out labels.
    report = evaluate_files(output / "predictions.parquet", bundle / "evaluation/test_targets.parquet",
                            "inhibition", output / "evaluation")
    truth = pd.read_parquet(bundle / "evaluation/test_targets.parquet")
    rejected = {}
    for name, invalid in [("omitted_row", prediction.iloc[1:]),
                          ("duplicate_id", pd.concat([prediction, prediction.iloc[[0]]])),
                          ("nonfinite", prediction.assign(prediction=np.nan))]:
        try:
            aligned = align_predictions(truth, invalid)
            from synergy_artifacts.metrics import regression_metrics
            regression_metrics(aligned.inhibition, aligned.prediction)
        except ValueError as error:
            rejected[name] = str(error)
        else:
            raise AssertionError(f"Invalid {name} accepted")
    verified.update(dummy_model="training mean; shuffled output order", metrics=report["pooled"], rejected=rejected)
    write_json(output / "contract_validation.json", verified)
    return verified


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(example(a.bundle, a.output))


if __name__ == "__main__":
    main()
