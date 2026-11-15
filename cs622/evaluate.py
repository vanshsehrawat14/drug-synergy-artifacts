"""Evaluate external model predictions against centrally aligned observations."""
import argparse
from pathlib import Path

import pandas as pd

from .io import digest, write_json
from .metrics import align_predictions, evaluate


def evaluate_files(predictions, observations, target, output, contrast=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    predicted = pd.read_parquet(predictions)
    source = pd.read_parquet(observations)
    # Select the declared evaluation folds, never the submitted observation IDs:
    # an incomplete submission must not shrink its own evaluation population.
    truth = source.copy()
    if "fold" in predicted and "fold" in truth:
        if predicted.fold.isna().any():
            raise ValueError("Missing prediction fold")
        truth = truth[truth.fold.isin(predicted.fold.unique())]
    if "is_combination" in truth:
        truth = truth[truth.is_combination]
    columns = [c for c in ["observation_id", "pair_id", "drug_a", "drug_b", "cell_line", "fold", "block_id", "dose_a", "dose_b", "unit_a", "unit_b", target] if c in truth]
    aligned = align_predictions(truth[columns], predicted)
    reported = predicted.set_index("observation_id").reindex(aligned.observation_id)
    for col in ["pair_id", "drug_a", "drug_b", "cell_line", "fold", "block_id", "dose_a", "dose_b", "unit_a", "unit_b"]:
        if col in predicted and col in aligned:
            if aligned[col].astype(str).tolist() != reported[col].astype(str).tolist():
                raise ValueError(f"Prediction metadata mismatch: {col}")
    report = evaluate(aligned, target, contrast=contrast)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", report)
    write_json(output / "provenance.json", {"prediction_sha256": digest(predictions), "observations_sha256": digest(observations), "target": target})
    print(report["pooled"])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--contrast", action="store_true")
    a = p.parse_args()
    evaluate_files(a.predictions, a.observations, a.target, a.output, a.contrast)


if __name__ == "__main__":
    main()
