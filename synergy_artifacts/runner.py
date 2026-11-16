"""One experiment contract for held-out-pair baselines and neural models.

python -m synergy_artifacts.runner --config configs/smoke.json --output results/neural/runs/smoke
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import FeatureBuilder, LazyFeatures, load_independent_features
from .io import digest, read_json, write_json
from .metrics import evaluate
from .models import AdditiveRidge, MeanModel
from .runtime import peak_memory_bytes
from .splits import partitions, validate_folds, validate_seen_drugs
from .training import NeuralRegressor, device_report

IDENTIFIERS = ["observation_id", "pair_id", "drug_a", "drug_b", "cell_line", "fold"]
DOSE_COLUMNS = ["block_id", "dose_a", "dose_b", "unit_a", "unit_b", "inhibition",
                "is_combination", "is_monotherapy", "valid_target", "valid_dose"]


def git_state():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def split_unit(config):
    split = config.get("split", {"kind": "pair"})
    if split.get("kind") not in {"pair", "pair_cell"} or (split["kind"] == "pair_cell" and config["table"] != "observations"):
        raise ValueError("Unknown split")
    return split["kind"]


def load_experiment(config):
    table = config["table"]
    if table not in {"observations", "panel", "published", "published_rows"}:
        raise ValueError("Unknown target table")
    target = config["target"]
    if table == "observations" and target != "inhibition":
        raise ValueError("Dose track predicts measured inhibition only")
    if table.startswith("published") and target != "synergy":
        raise ValueError("Published DeepSynergy track predicts published synergy only")
    if config["model"]["kind"] == "deepsynergy" and table == "observations":
        raise ValueError("Published DeepSynergy is a panel synergy track; use dose model for dose response")
    path = Path(config["dataset"]) / (table + ".parquet")
    actual_hash = digest(path)
    manifest_path = path.with_suffix(".manifest.json") if table == "published_rows" else path.parent / "manifest.json"
    manifest = read_json(manifest_path)
    expected = manifest["output_sha256"] if table == "published_rows" else manifest["artifacts"][table]
    if actual_hash != expected:
        raise ValueError("Prepared artifact checksum mismatch; rebuild or declare a new dataset version")
    columns = IDENTIFIERS + (DOSE_COLUMNS if table == "observations" else [target])
    df = pd.read_parquet(path, columns=list(dict.fromkeys(columns)))
    if config.get("subset", {}).get("cells"):
        cells = config["subset"]["cells"]
        df = df[df.cell_line.isin(cells)].copy()
    monotherapy = None
    if table == "observations":
        monotherapy = df[df.is_monotherapy & df.valid_target & df.valid_dose].copy()
        df = df[df.is_combination & df.valid_target & df.valid_dose].copy()
        limit = config.get("subset", {}).get("blocks_per_pair_cell")
        if limit:
            selected = df[["pair_id", "cell_line", "block_id"]].drop_duplicates().sort_values("block_id")
            selected = selected.groupby(["pair_id", "cell_line"], observed=True).head(limit)
            df = df[df.block_id.isin(selected.block_id)].copy()
    df = df.sort_values("observation_id").reset_index(drop=True)
    unit = split_unit(config)
    if unit == "pair_cell":
        # Exploratory pair-seen control: pair-cell folds replace the published pair folds.
        from .pair_seen import assign
        df = assign(df, config["split"]["seed"])
    validate_folds(df, unit=unit)
    validate_seen_drugs(df, config["test_fold"], unit=unit)
    if not np.isfinite(df[target].to_numpy(float)).all():
        raise ValueError("Nonfinite targets; define explicit exclusions before running")
    return df, monotherapy, path


def run(config, output, resume=None):
    started = time.perf_counter()
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("Run output exists; choose a new run ID or explicitly resume")
    output.mkdir(parents=True, exist_ok=True)
    df, mono, path = load_experiment(config)
    unit = split_unit(config)
    tr, va, te = partitions(df, config["test_fold"], unit=unit)
    train, validation = df.iloc[tr], df.iloc[va]
    feature_config = config["features"]
    external = None
    if feature_config.get("external"):
        external = load_independent_features(feature_config["external"], feature_config["provenance"])
    builder = FeatureBuilder(feature_config["regime"], feature_config.get("dose", config["table"] == "observations"),
                             feature_config.get("identities", True), external,
                             feature_config.get("normalization", "standard"), feature_config.get("remove_constant", False),
                             feature_config.get("mono_missing", "mean"), feature_config.get("dose_input", True),
                             feature_config.get("reference_inputs", []), feature_config.get("drug_identity", True),
                             feature_config.get("cell_identity", True))
    if mono is not None:
        # Pair folds: monotherapy from training pairs. Pair-cell folds: from training blocks only.
        mono = mono[mono.pair_id.isin(train.pair_id)] if unit == "pair" else mono[mono.block_id.isin(train.block_id)]
    builder.fit(train.drop(columns=config["target"]), mono)
    kind = config["model"]["kind"]
    lazy = external is not None and kind in {"dose", "deepsynergy"}
    def features(frame):
        frame = frame.drop(columns=config["target"], errors="ignore")
        return LazyFeatures(builder, frame) if lazy else builder.transform(frame)
    offset_kind = feature_config.get("reference_offset")
    if offset_kind and kind not in {"dose", "deepsynergy"}:
        raise ValueError("Reference offsets are defined for neural residual models only")
    def offset(frame):
        # Residual learning: fit Y - R on training-pair references; predictions add R back.
        return builder.reference(frame, offset_kind) if offset_kind else np.zeros(len(frame))
    xtrain, xval = features(train), features(validation)
    ytrain, yval = train[config["target"]].to_numpy(float), validation[config["target"]].to_numpy(float)
    # Checkpoint context includes training-derived transforms, not held-out labels.
    recovery_config = copy.deepcopy(config)
    recovery_config.get("training", {}).pop("epochs", None)
    context = {"dataset_sha256": digest(path), "config": recovery_config, "features": builder.state_dict(),
               "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
               "training_ids_sha256": hashlib.sha256("\n".join(train.observation_id).encode()).hexdigest()}
    model_config = config["model"].copy()
    kind = model_config.pop("kind")
    fit_summary = {}
    if kind in {"dose", "deepsynergy"}:
        sample = builder.transform(train.iloc[:1].drop(columns=config["target"]))
        spec = {"kind": kind, "endpoint_dim": sample[0].shape[1], "cell_dim": sample[2].shape[1], **model_config}
        model = NeuralRegressor(spec, config["training"], config["seed"])
        model.fit(xtrain, ytrain - offset(train), (xval, yval - offset(validation)), output / "checkpoints", context, resume)
        fit_summary = model.fit_summary
    elif kind.startswith("reference_"):
        if builder.regime != "F2":
            raise ValueError("Reference prediction requires explicit training-only F2 monotherapy")
        if kind not in {"reference_bliss", "reference_hsa"}:
            raise ValueError("Only observed-monotherapy Bliss/HSA references are implemented")
        model = None
    else:
        if resume:
            raise ValueError("Resume is supported for neural models only")
        if kind == "mean":
            model = MeanModel()
        elif kind == "additive":
            model = AdditiveRidge(**model_config)
        else:
            raise ValueError(f"Unknown model: {kind}")
        model.fit(xtrain, ytrain, (xval, yval))
        np.savez(output / "baseline.npz", **({"mean": model.mean} if kind == "mean" else {"coefficients": model.coefficients}))
    write_json(output / "features.json", builder.state_dict())
    write_json(output / "config.json", config)
    # Test outcomes are accessed for scoring only after all fitting/model selection.
    test_features = df.iloc[te].drop(columns=config["target"])
    if kind.startswith("reference_"):
        from .controls import bliss_reference, hsa_reference
        a, b = [builder.monotherapy_values(test_features, endpoint) for endpoint in ["a", "b"]]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Reference-only predictor has missing training-only monotherapy; set mono_missing=interpolate explicitly")
        prediction = (bliss_reference if kind == "reference_bliss" else hsa_reference)(a, b)
    else:
        prediction = model.predict(features(test_features)) + offset(test_features)
    identifiers = IDENTIFIERS + (["block_id", "dose_a", "dose_b", "unit_a", "unit_b"] if config["table"] == "observations" else [])
    predictions = df.iloc[te][identifiers].copy()
    predictions["experiment_id"], predictions["dataset"], predictions["study"] = output.name, "DeepSynergy-published" if config["table"].startswith("published") else "DrugComb-v1.4", "ONEIL"
    predictions["target_name"], predictions["feature_regime"] = config["target"], feature_config["regime"]
    predictions["prediction"] = prediction
    if builder.regime == "F2":
        status = builder.monotherapy_status(test_features)
        unavailable = "mean_imputed"
        predictions["monotherapy_status"] = np.array(["exact", "interpolated", unavailable])[status]
    if offset_kind:
        predictions["reference_offset"] = offset(test_features)
    if kind == "dose" and config["model"].get("combine") == "concat":
        # Order-dependent ablation: the same unordered test pair scored in the opposite order.
        rename = {f"{p}_{e}": f"{p}_{'b' if e == 'a' else 'a'}" for p in ["drug", "dose", "unit"] for e in "ab"}
        swapped = test_features.rename(columns=rename)
        predictions["prediction_swapped"] = model.predict(features(swapped)) + offset(swapped)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    evaluation = predictions.copy()
    evaluation[config["target"]] = df.iloc[te][config["target"]].to_numpy(float)
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    evaluation.to_parquet(evaluation_dir / "predictions_with_truth.parquet", index=False)
    # Source-row reproduction keeps 315 duplicate pair/cell labels. Panel contrast
    # is reported only after an explicit aggregation, never treating those as independent units.
    metrics = evaluate(evaluation, config["target"], config.get("bootstrap", 200), config["seed"],
                       contrast=config["table"] == "panel" or config["table"] == "published")
    write_json(evaluation_dir / "metrics.json", metrics)
    metadata = {"config": config, "git": git_state(), "python": platform.python_version(),
                "dependencies": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "torch", "pyarrow"]},
                "dataset_sha256": context["dataset_sha256"], "fold_counts": validate_folds(df, unit=unit),
                "partition_rows": dict(zip(["train", "validation", "test"], map(len, [tr, va, te]))),
                "fit": fit_summary, "elapsed_seconds": time.perf_counter() - started,
                "device": device_report(model.device) if kind in {"dose", "deepsynergy"} else device_report("cpu"),
                "process_peak_resident_bytes": peak_memory_bytes(),
                "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                "split_rule": "published test f, validation (f+1)%5, other folds train" if unit == "pair" else
                              f"pair-cell folds seed {config['split']['seed']}: test f, validation (f+1)%5, other folds train",
                "pilot": bool(config.get("subset"))}
    if unit == "pair_cell":
        from .pair_seen import assignment_sha256, pair_cell_folds
        metadata["split_assignment_sha256"] = assignment_sha256(pair_cell_folds(df, config["split"]["seed"]))
    if builder.regime == "F2":
        metadata["monotherapy_coverage"] = {name: builder.monotherapy_coverage(df.iloc[ix])
                                           for name, ix in zip(["train", "validation", "test"], [tr, va, te])}
    write_json(output / "metadata.json", metadata)
    print(json.dumps({"output": str(output), "pooled": metrics["pooled"], "fit": fit_summary,
                      "seconds": metadata["elapsed_seconds"]}, indent=2))
    return metadata, metrics


def predict_checkpoint(checkpoint, inputs, output, device="cpu"):
    if Path(output).exists():
        raise FileExistsError(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    model, context = NeuralRegressor.load(checkpoint, device)
    builder = FeatureBuilder.from_state_dict(context["features"])
    frame = pd.read_parquet(inputs)
    target = context["config"]["target"]
    frame = frame.drop(columns=[target], errors="ignore")
    columns = [c for c in IDENTIFIERS + ["block_id", "dose_a", "dose_b", "unit_a", "unit_b"] if c in frame]
    result = frame[columns].copy()
    inputs = LazyFeatures(builder, frame) if builder.external is not None else builder.transform(frame)
    offset_kind = context["config"]["features"].get("reference_offset")
    result["prediction"] = model.predict(inputs) + (builder.reference(frame, offset_kind) if offset_kind else 0.)
    result.to_parquet(output, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--predict-checkpoint")
    parser.add_argument("--inputs")
    args = parser.parse_args()
    if args.predict_checkpoint:
        predict_checkpoint(args.predict_checkpoint, args.inputs, args.output)
    else:
        run(read_json(args.config), args.output, args.resume)


if __name__ == "__main__":
    main()
