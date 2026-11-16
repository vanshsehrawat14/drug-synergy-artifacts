"""Verified tables and figures for the neural GPU benchmark and the DeepSynergy reproduction.

python -m synergy_artifacts.neural_report verify --output results/neural_benchmark
python -m synergy_artifacts.neural_report analyze --output results/neural_benchmark
python -m synergy_artifacts.neural_report figures --output results/neural_benchmark

verify checks imported runs against the frozen protocol, data, code revision, predictions,
metrics, histories and checkpoints. analyze scores out-of-fold predictions after selection is
fixed and never changes a model choice. figures reads only the saved tables.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student

from .deepsynergy import LABEL_SHA, original_rows, protocol_sha, smoothed_epoch
from .features import FeatureBuilder
from .io import digest, read_json, write_json
from .metrics import evaluate, interaction_contrast, regression_metrics
from .models import make_network
from .runner import load_experiment
from .suite import benchmark_runs, candidate_config, protocol_hash, run_dir, select, selection_runs

RUNS = Path("results/neural/runs")
CPU = {"cpu_f0": "phase2_f0", "cpu_f2": "phase2_f2", "additive": "phase2_additive", "mean": "phase2_mean"}
BENCHMARK = ["f0_base", "selected:F0", "f2_base_mean", "f2_base", "selected:F2", "f2_residual", "mono_only"]
COORDINATES = ["pair_id", "drug_a", "drug_b", "cell_line", "fold", "block_id", "dose_a", "dose_b", "unit_a", "unit_b"]
# Fixed before test outcomes were read: each candidate differs from its control in the stated factor.
CONTROL = {"f0_plateau": "f0_base", "f0_deep": "f0_base", "f0_wide": "f0_base", "f0_regularized": "f0_base",
           "f0_combined": "f0_base", "f0_ordered": "f0_base", "f2_base": "f0_base", "f2_base_mean": "f2_base",
           "f2_plateau": "f2_base", "f2_deep": "f2_base", "f2_wide": "f2_base", "f2_regularized": "f2_base",
           "f2_combined": "f2_base", "f2_residual": "f2_base", "f2_refinput": "f2_base",
           "mono_dose": "f2_residual", "mono_only": "mono_dose"}
COMPARISONS = [
    ("f0_base", "cpu_f0", "replication: production F0 on GPU seeds vs frozen CPU run"),
    ("f2_base_mean", "cpu_f2", "replication: production F2 on GPU seeds vs frozen CPU run"),
    ("selected:F0", "cpu_f0", "validation-selected F0 vs frozen CPU F0"),
    ("selected:F2", "cpu_f2", "validation-selected F2 vs frozen CPU F2"),
    ("selected:F0", "f0_base", "capacity and optimization without monotherapy"),
    ("selected:F2", "f2_base", "capacity and optimization with monotherapy"),
    ("f2_base", "f2_base_mean", "absent monotherapy doses: interpolation vs training mean"),
    ("f2_base", "f0_base", "monotherapy inputs at the production architecture"),
    ("selected:F2", "selected:F0", "monotherapy inputs at the selected architectures"),
    ("f2_residual", "f2_base", "residual learning over the training-pair Bliss reference"),
    ("mono_only", "bliss_reference", "learned function of monotherapy values vs fixed Bliss"),
    ("f2_residual", "mono_only", "drug/cell identities and doses beyond monotherapy values"),
    ("selected:F0", "bliss_reference", "identities and doses alone vs monotherapy-based Bliss"),
    ("selected:F2", "bliss_reference", "best network vs monotherapy-based Bliss"),
]
# Declared with the dose-aware DeepSynergy protocol, before any of its test predictions existed.
DOSE = "deepsynergy_dose"
DOSE_COMPARISONS = [
    ("selected:F0", DOSE, "identity embeddings vs published DeepSynergy on original features, both without monotherapy"),
    ("selected:F2", DOSE, "selected F2 vs dose-aware DeepSynergy"),
    (DOSE, "f0_base", "dose-aware DeepSynergy vs the production F0 network"),
    (DOSE, "bliss_reference", "dose-aware DeepSynergy vs monotherapy-based Bliss"),
]
# Team methods added with --method are compared with these, on identical rows.
TEAM_CONTROLS = ["selected:F2", "selected:F0", "additive", "bliss_reference"]
INPUTS = {"selected:F2": "identities, doses, training-pair monotherapy", "selected:F0": "identities, doses",
          DOSE: "chemical descriptors, expression, doses", "f2_base_mean": "identities, doses, training-pair monotherapy",
          "f0_base": "identities, doses", "cpu_f2": "identities, doses, training-pair monotherapy", "cpu_f0": "identities, doses",
          "mono_only": "training-pair monotherapy", "bliss_reference": "training-pair monotherapy, fixed rule",
          "additive": "identities, doses (additive ridge)", "mean": "none (training mean)"}
FINAL = ["selected:F2", "selected:F0", DOSE, "f2_base_mean", "f0_base", "cpu_f2", "cpu_f0", "bliss_reference", "additive", "mean"]
DS_SETTINGS = {"layers": [8192, 4096], "input_dropout": .2, "dropout": .5, "placement": "paper", "norm": "tanh_norm",
               "lr": 1e-5, "momentum": .5, "batch_size": 64, "epochs": 1000, "window": 25, "checkpoint_every": 50,
               "threads": 4, "allow_tf32": False, "keep_checkpoints": False}


def close(a, b, rtol=1e-9):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k], rtol) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(close(x, y, rtol) for x, y in zip(a, b))
    if isinstance(a, (float, int)) and isinstance(b, (float, int)) and not isinstance(a, bool):
        return bool(np.isclose(a, b, rtol=rtol, atol=0))
    return a == b


def code_hashes(commit):
    files = subprocess.check_output(["git", "ls-tree", "--name-only", commit, "synergy_artifacts/"], text=True).split()
    return {Path(f).name: hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{f}"])).hexdigest()
            for f in files if f.endswith(".py")}


def load_truth(dataset):
    """Combination rows in runner order (sorted observation IDs) and per-fold training-ID hashes."""
    df, _, path = load_experiment({"dataset": str(dataset), "table": "observations", "target": "inhibition",
                                   "test_fold": 0, "model": {"kind": "dose"}})
    train_ids = {}
    for k in range(5):
        train = df.observation_id[~df.fold.isin([k, (k + 1) % 5])]
        train_ids[k] = hashlib.sha256("\n".join(train).encode()).hexdigest()
    return df, digest(path), train_ids


def restore(checkpoint, frame):
    """Best-checkpoint predictions from a weights-only load; no pickled code is executed."""
    identity = checkpoint["identity"]
    builder = FeatureBuilder.from_state_dict(identity["context"]["features"])
    network = make_network(identity["spec"])
    network.load_state_dict(checkpoint["best_model"])
    network.eval()
    with torch.no_grad():
        output = network(*(torch.as_tensor(x) for x in builder.transform(frame))).numpy().astype(float)
    value = output * checkpoint["target_scale"] + checkpoint["target_mean"]
    kind = identity["context"]["config"]["features"].get("reference_offset")
    return value + builder.reference(frame, kind) if kind else value


def verify_run(path, config, git, code, dataset_sha, rows, train_ids_sha, progress=None, sample=2000):
    """Checks for one run; rows are the test-fold truth rows in runner order."""
    meta = read_json(path / "metadata.json")
    fit = meta["fit"]
    result = {"config": meta["config"] == config and read_json(path / "config.json") == config,
              "git": meta["git"] == git, "dataset": meta["dataset_sha256"] == dataset_sha, "code": meta["code_sha256"] == code,
              "device": (fit["device"].startswith(config["training"]["device"]) and meta["device"]["deterministic_algorithms"]
                         and meta["device"]["float32_matmul_precision"] == "highest")}
    pred = pd.read_parquet(path / "predictions.parquet")
    p = pred.prediction.to_numpy(float)
    result["predictions"] = (pred.observation_id.tolist() == rows.observation_id.tolist() and np.isfinite(p).all()
                             and all(pred[c].astype(str).tolist() == rows[c].astype(str).tolist() for c in COORDINATES))
    stored = read_json(path / "evaluation" / "metrics.json")
    again = evaluate(rows[["observation_id", "pair_id", "cell_line", "inhibition"]].assign(prediction=p),
                     "inhibition", config["bootstrap"], config["seed"])
    result["metrics"] = all(close(again[k], stored[k]) for k in ["pooled", "uncertainty", "macro_mse"])
    history = read_json(path / "checkpoints" / "history.json")
    losses, training = [r["validation_mse"] for r in history], config["training"]
    result["history"] = (len(losses) == fit["epochs_completed"] == min(training["epochs"], fit["best_epoch"] + 1 + training["patience"])
                         and int(np.argmin(losses)) == fit["best_epoch"] and min(losses) == fit["validation_mse"])
    try:
        checkpoint = torch.load(path / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
        identity, context = checkpoint["identity"], checkpoint["identity"]["context"]
        recovery = json.loads(json.dumps(config))
        recovery["training"].pop("epochs")
        index = np.sort(np.random.default_rng(0).choice(len(rows), size=min(sample, len(rows)), replace=False))
        difference = float(np.abs(restore(checkpoint, rows.iloc[index].drop(columns="inhibition")) - p[index]).max())
        result["checkpoint"] = (checkpoint["best_epoch"] == fit["best_epoch"] and checkpoint["best"] == fit["validation_mse"]
                                and identity["seed"] == config["seed"] and identity["training"] == recovery["training"]
                                and all(identity["spec"][k] == v for k, v in config["model"].items() if k != "kind")
                                and context["config"] == recovery and context["dataset_sha256"] == dataset_sha
                                and context["code_sha256"] == code and context["training_ids_sha256"] == train_ids_sha
                                and sum(v.numel() for v in checkpoint["best_model"].values()) == fit["parameters"]
                                and difference <= 1e-3)
        result["restore_max_abs_difference"] = difference
    except Exception as error:  # A failed load is a failed check, recorded with its reason.
        result["checkpoint"], result["checkpoint_error"] = False, f"{type(error).__name__}: {error}"
    if progress is not None:
        result["progress"] = (progress["validation_mse"] == fit["validation_mse"]
                              and progress["epochs"] == fit["epochs_completed"] and progress["resumed"] is False)
    return result


def found_runs(suite):
    return sorted((p.parent.parent.name, int(p.parent.name[5:]), int(p.name[5:])) for p in Path(suite).glob("runs/*/seed_*/fold_*"))


def planned_runs(protocol, chosen):
    selection = selection_runs(protocol, range(5))
    return selection + [r for r in benchmark_runs(protocol, chosen, protocol["benchmark"]["seeds"], range(5)) if r not in selection]


def verify_deepsynergy(root, labels, features, provenance):
    root, report = Path(root), read_json(Path(root) / "reproduction.json")
    frame = original_rows("published", labels, None)
    checks = {"settings": report["settings"] == DS_SETTINGS and report["target"] == "published" and report["seed"] == 2026,
              "inputs": digest(labels) == LABEL_SHA and report["features_sha256"] == digest(features) == read_json(provenance)["sha256"],
              "device": report["device"]["type"] == "cuda" and report["device"]["deterministic_algorithms"]
                        and report["device"]["float32_matmul_precision"] == "highest"}
    folds = []
    for k in range(5):
        d = root / f"fold_{k}"
        metrics, chosen, stage1 = read_json(d / "metrics.json"), read_json(d / "selection.json"), read_json(d / "stage1_history.json")
        history, pred = read_json(d / "history.json"), pd.read_parquet(d / "predictions.parquet")
        rows = frame[frame.fold == k].reset_index(drop=True)
        losses = [r["validation_mse"] for r in stage1]
        expected = {"window": 25, "epoch_index": smoothed_epoch(losses, 25), "notebook_window_15": smoothed_epoch(losses, 15),
                    "raw_minimum": int(np.argmin(losses))}
        expected["final_epochs"] = max(expected["epoch_index"], 1)
        y, p = rows.y.to_numpy(float), pred.prediction.to_numpy(float)
        mse = float(np.mean((y - p) ** 2))
        folds.append({"rows": len(pred) == len(rows) == metrics["test_samples"]
                      and all((pred[c].to_numpy() == rows[c].to_numpy()).all() for c in ["a", "b", "cell", "y", "fold"]),
                      "symmetrized": np.array_equal(p, (pred.prediction_ab.to_numpy() + pred.prediction_ba.to_numpy()) / 2),
                      "selection": chosen == expected == metrics["selected"] and len(stage1) == 1000
                                   and history["stage1"] == stage1 and len(history["stage2"]) == chosen["final_epochs"],
                      "metrics": metrics["validation_fold"] == (k + 1) % 5 and close(
                          {"mse": mse, "pearson": float(np.corrcoef(y, p)[0, 1]), "rmse": float(np.sqrt(mse))}, metrics["symmetrized"])})
    mse = [read_json(root / f"fold_{k}" / "metrics.json")["symmetrized"] for k in range(5)]
    again = {"mse": float(np.mean([m["mse"] for m in mse])), "rmse_mean": float(np.mean([m["rmse"] for m in mse])),
             "rmse_sd": float(np.std([m["rmse"] for m in mse], ddof=1)),
             "pearson_mean": float(np.mean([m["pearson"] for m in mse])), "pearson_sd": float(np.std([m["pearson"] for m in mse], ddof=1))}
    checks.update({key: all(f[key] for f in folds) for key in folds[0]})
    checks["mean_over_folds"] = close(again, report["mean_over_folds"])
    return checks


def verify_dose(root, protocol_path, folds, dataset_sha, features, pilot=None):
    """Frozen-protocol, input, code, prediction, metric and stopping checks for the dose-aware runs.

    Best checkpoints stay on the training machine; each run predicted its test rows from the
    reloaded best checkpoint and recorded its SHA-256.
    """
    root, protocol = Path(root), read_json(protocol_path)
    manifest, settings = read_json(root / "protocol.json"), protocol["settings"]
    checks = {"dose_protocol_frozen": (manifest["protocol"] == protocol and manifest["protocol_sha256"] == protocol_sha(protocol)
                                       and manifest["pilot"] is False and None not in (settings["epochs"], settings["patience"]))}
    planned = [(s, k) for s in protocol["seeds"] for k in protocol["folds"]]
    checks["dose_run_coverage"] = sorted((int(p.parent.name[5:]), int(p.name[5:])) for p in root.glob("seed_*/fold_*")) == sorted(planned)
    runs, commits, hashes = {}, set(), {}
    for s, k in planned:
        d, rows = root / f"seed_{s}" / f"fold_{k}", folds[k]
        meta, pred = read_json(d / "metadata.json"), pd.read_parquet(d / "predictions.parquet")
        commit = meta["git"]["commit"]
        commits.add(commit)
        if commit not in hashes:
            hashes[commit] = code_hashes(commit)
        p, fit = pred.prediction.to_numpy(float), meta["fit"]
        losses = [r["validation_mse"] for r in read_json(d / "history.json")]
        stored = read_json(d / "evaluation" / "metrics.json")
        again = evaluate(rows[["observation_id", "pair_id", "cell_line", "inhibition"]].assign(prediction=p), "inhibition",
                         protocol["bootstrap"], s)
        runs[f"seed_{s}/fold_{k}"] = {
            "config": (meta["settings"] == settings and meta["protocol_sha256"] == manifest["protocol_sha256"]
                       and (meta["seed"], meta["fold"], meta["pilot"]) == (s, k, False)),
            "git": meta["git"]["dirty"] is False, "code": meta["code_sha256"] == hashes[commit],
            "inputs": meta["dataset_sha256"] == dataset_sha and meta["features_sha256"] == digest(features),
            "device": (meta["device"]["type"] == "cuda" and meta["device"]["deterministic_algorithms"]
                       and meta["device"]["float32_matmul_precision"] == "highest"),
            "predictions": (pred.observation_id.tolist() == rows.observation_id.tolist() and np.isfinite(p).all()
                            and all(pred[c].astype(str).tolist() == rows[c].astype(str).tolist() for c in COORDINATES)
                            and np.array_equal(p, (pred.prediction_ab.to_numpy() + pred.prediction_ba.to_numpy()) / 2)),
            "metrics": all(close(again[x], stored[x]) for x in ["pooled", "uncertainty", "macro_mse"]),
            "history": (len(losses) == fit["epochs_completed"] == min(settings["epochs"], fit["best_epoch"] + 1 + settings["patience"])
                        and int(np.argmin(losses)) == fit["best_epoch"] and min(losses) == fit["validation_mse"]),
            "checkpoint_recorded": len(meta.get("checkpoint_sha256", "")) == 64}
    for key in next(iter(runs.values())):
        checks[f"dose_runs_{key}"] = all(r[key] for r in runs.values())
    checks["dose_single_commit"] = len(commits) == 1
    if pilot is not None:
        record = read_json(Path(pilot) / f"seed_{protocol['pilot']['seed']}" / f"fold_{protocol['pilot']['fold']}" / "pilot.json")
        checks["dose_pilot_validation_only"] = record["pilot"] is True and not list(Path(pilot).rglob("predictions.parquet"))
    return checks, runs


def verify(output, suite=RUNS / "benchmark_v1", deepsynergy=RUNS / "deepsynergy_original", protocol_path="configs/neural_benchmark.json",
           dataset="data/processed/oneil_v1", labels="data/raw/deepsynergy/labels.csv",
           features="data/processed/deepsynergy/features.npz", provenance="data/processed/deepsynergy/features.provenance.json",
           cpu_table="results/response_metrics.csv", sample=2000, dose=RUNS / DOSE, dose_protocol="configs/deepsynergy_dose.json"):
    suite, checks = Path(suite), {}
    protocol, manifest = read_json(protocol_path), read_json(suite / "suite.json")
    checks["protocol_frozen"] = (manifest["protocol"] == protocol and manifest["protocol_sha256"] == protocol_hash(protocol)
                                 and manifest["base_configs"] == {k: read_json(v) for k, v in protocol["base"].items()}
                                 and manifest["budget"] is None and manifest["folds"] == list(range(5)))
    git = manifest["git"]
    checks["suite_git_clean"] = git["dirty"] is False and git["commit"] is not None
    stored, chosen = read_json(suite / "selection.json"), select(protocol, suite, range(5))
    checks["selection_recomputed"] = stored["selected"] == chosen and stored["rule"] == protocol["selection"]
    planned = planned_runs(protocol, chosen)
    checks["run_coverage"] = sorted(planned) == found_runs(suite)
    checks["no_failed_runs"] = not (suite / "failed").exists()
    progress = {(r["candidate"], r["seed"], r["fold"]): r for r in map(json.loads, (suite / "progress.jsonl").read_text().splitlines())}
    checks["progress_complete"] = sorted(progress) == sorted(planned)
    df, dataset_sha, train_ids = load_truth(dataset)
    code = code_hashes(git["commit"])
    folds = {k: df[df.fold == k] for k in range(5)}
    runs = {}
    for name, seed, fold in planned:
        runs[f"{name}/seed_{seed}/fold_{fold}"] = verify_run(
            run_dir(suite, name, seed, fold), candidate_config(protocol, name, fold, seed, "cuda"), git, code, dataset_sha,
            folds[fold], train_ids[fold], progress.get((name, seed, fold)), sample)
        print(name, seed, fold, all(v for k, v in runs[f"{name}/seed_{seed}/fold_{fold}"].items() if isinstance(v, bool)), flush=True)
    for key in ["config", "git", "dataset", "code", "device", "predictions", "metrics", "history", "checkpoint", "progress"]:
        checks[f"runs_{key}"] = all(r[key] for r in runs.values())
    table = pd.read_csv(cpu_table)
    checks["cpu_reference_predictions"] = all(digest(RUNS / CPU["cpu_" + r.model if r.model in {"f0", "f2"} else r.model] /
                                                     f"fold_{r.test_fold}" / "predictions.parquet") == r.prediction_sha256
                                              for r in table.itertuples())
    checks.update({"deepsynergy_" + k: v for k, v in verify_deepsynergy(deepsynergy, labels, features, provenance).items()})
    dose_runs = {}
    if Path(dose).exists():
        dose_checks, dose_runs = verify_dose(dose, dose_protocol, folds, dataset_sha, features, RUNS / f"{DOSE}_pilot")
        checks.update(dose_checks)
    differences = [r["restore_max_abs_difference"] for r in runs.values() if "restore_max_abs_difference" in r]
    report = {"passed": all(checks.values()), "checks": checks, "commit": git["commit"], "dataset_sha256": dataset_sha,
              "protocol_sha256": manifest["protocol_sha256"], "runs": len(planned), "selection_runs": len(selection_runs(protocol, range(5))),
              "checkpoint_restore": {"rows_per_run": sample, "device": "cpu", "max_abs_difference": max(differences), "tolerance": 1e-3},
              "failures": {k: v for k, v in {**runs, **{f"{DOSE}/{k}": v for k, v in dose_runs.items()}}.items()
                           if not all(x for x in v.values() if isinstance(x, bool))},
              DOSE: "verified" if dose_runs else "absent"}
    write_json(Path(output) / "verification.json", report)
    print(json.dumps({k: report[k] for k in ["passed", "checks", "checkpoint_restore"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    return report


# ---- analysis -------------------------------------------------------------------------------

def index(df):
    """Integer codes for pairs, cells and pair-cell panel rows, with truth sums per pair."""
    pairs, pair = np.unique(df.pair_id.astype(str).to_numpy(), return_inverse=True)
    cells, cell = np.unique(df.cell_line.astype(str).to_numpy(), return_inverse=True)
    _, panel = np.unique(pair * len(cells) + cell, return_inverse=True)
    first = df.assign(panel=panel).groupby("panel").head(1).sort_values("panel")
    y = df.inhibition.to_numpy(float)
    meta = df.assign(pair=pair).groupby("pair").head(1).sort_values("pair")
    drugs = sorted(set(df.drug_a.astype(str)) | set(df.drug_b.astype(str)))
    pair_drugs = list(zip(meta.drug_a.astype(str), meta.drug_b.astype(str)))
    panel_n = np.bincount(panel)
    return {"df": df, "y": y, "pairs": pairs, "pair": pair, "cells": cells, "cell": cell, "panel": panel, "panel_n": panel_n,
            "panel_frame": first[["fold", "pair_id", "drug_a", "drug_b", "cell_line"]].astype(
                {"fold": int, "pair_id": str, "drug_a": str, "drug_b": str, "cell_line": str}).reset_index(drop=True)
                .assign(response=np.bincount(panel, weights=y) / panel_n),
            "pair_n": np.bincount(pair).astype(float), "pair_fold": meta.fold.to_numpy(int), "pair_drugs": pair_drugs,
            "drugs": drugs, "rows": {k: np.flatnonzero(df.fold.to_numpy() == k) for k in range(5)},
            "ids": {k: df.observation_id.to_numpy()[df.fold.to_numpy() == k].tolist() for k in range(5)}}


def model_paths(model, seed, chosen, suite):
    if model in CPU:
        return {k: RUNS / CPU[model] / f"fold_{k}" for k in range(5)}
    if model == DOSE:
        return {k: RUNS / DOSE / f"seed_{seed}" / f"fold_{k}" for k in range(5)}
    name = lambda k: chosen[model.split(":", 1)[1]][str(k)]["selected"] if model.startswith("selected:") else model
    return {k: run_dir(suite, name(k), seed, k) for k in range(5)}


def out_of_fold(paths, t, column="prediction"):
    vector = np.full(len(t["y"]), np.nan) if column != "monotherapy_status" else np.empty(len(t["y"]), dtype=object)
    for k, path in paths.items():
        frame = pd.read_parquet(path / "predictions.parquet", columns=["observation_id", column])
        if frame.observation_id.tolist() != t["ids"][k]:
            raise ValueError(f"{path}: predictions are not aligned with test fold {k}")
        vector[t["rows"][k]] = frame[column].to_numpy()
    if column != "monotherapy_status" and not np.isfinite(vector).all():
        raise ValueError("Incomplete out-of-fold predictions")
    return vector


def team_predictions(directory, t):
    """Out-of-fold predictions under the shared contract: fold_<k>.parquet holds observation_id and
    prediction for every test row of fold k, once, in any order."""
    vector = np.full(len(t["y"]), np.nan)
    for k in range(5):
        frame = pd.read_parquet(Path(directory) / f"fold_{k}.parquet", columns=["observation_id", "prediction"])
        if frame.observation_id.duplicated().any() or set(frame.observation_id) != set(t["ids"][k]):
            raise ValueError(f"{directory}: fold {k} must cover exactly its test rows, once each")
        vector[t["rows"][k]] = frame.set_index("observation_id").prediction.reindex(t["ids"][k]).to_numpy(float)
    if not np.isfinite(vector).all():
        raise ValueError(f"{directory}: nonfinite predictions")
    return vector


def contrast_energies(t, columns):
    """||P(y - c)||^2 on the pair-cell panel for each column c, with all drugs and with each drug deleted.

    P removes drug main effects within each fold and cell (the methodology interaction contrast).
    Row 0 uses every pair; row j deletes the pairs containing drug j-1. Pass c = 0 for ||Py||^2
    and c = R for the Bliss residual energy.
    """
    frame, names = t["panel_frame"], list(columns)
    d = frame.response.to_numpy()[:, None] - np.column_stack([columns[n] for n in names])
    energy = np.zeros((1 + len(t["drugs"]), len(names)))
    for _, g in frame.groupby(["fold", "cell_line"], sort=True):
        rows, a, b = g.index.to_numpy(), g.drug_a.to_numpy(), g.drug_b.to_numpy()
        for j, drop in enumerate([None] + t["drugs"]):
            keep = np.ones(len(g), bool) if drop is None else (a != drop) & (b != drop)
            if not keep.any():
                continue
            drugs = {x: i for i, x in enumerate(sorted(set(a[keep]) | set(b[keep])))}
            h = np.zeros((keep.sum(), len(drugs)))
            ix = np.arange(keep.sum())
            np.add.at(h, (ix, [drugs[x] for x in a[keep]]), 1)
            np.add.at(h, (ix, [drugs[x] for x in b[keep]]), 1)
            z = d[rows[keep]]
            residual = z - h @ np.linalg.lstsq(h, z, rcond=None)[0]
            energy[j] += (residual ** 2).sum(0)
    return dict(zip(names, energy.T))


def jackknife_interval(full, replicates):
    """Delete-one-drug t interval centred on the full-data value."""
    replicates = np.asarray(replicates, float)
    half = student.ppf(.975, len(replicates) - 1) * np.sqrt((len(replicates) - 1) / len(replicates)
                                                             * ((replicates - replicates.mean()) ** 2).sum())
    return full - half, full + half


def monotherapy_audit(dataset):
    """Provenance and missing-dose behaviour of the F2 monotherapy inputs, from monotherapy rows only.

    No combination outcome is used. Absent keys are scored against the shared single-agent value
    stored in the held-out pair's own block edges; leave-one-dose-out uses training-pair keys only.
    """
    columns = ["drug_a", "drug_b", "cell_line", "dose_a", "dose_b", "inhibition", "pair_id", "block_id", "fold",
               "is_combination", "is_monotherapy", "valid_target", "valid_dose"]
    o = pd.read_parquet(Path(dataset) / "observations.parquet", columns=columns)
    o = o[o.valid_target & o.valid_dose]
    combination, mono = o[o.is_combination], o[o.is_monotherapy]
    edges = []
    for e, other in [("a", "b"), ("b", "a")]:
        m = mono[(mono["dose_" + e] > 0) & (mono["dose_" + other] == 0)]
        edges.append(pd.DataFrame({"drug": m["drug_" + e].astype(str).to_numpy(), "cell": m.cell_line.astype(str).to_numpy(),
                                   "dose": m["dose_" + e].to_numpy(), "value": m.inhibition.to_numpy(),
                                   "pair": m.pair_id.astype(str).to_numpy(), "block": m.block_id.to_numpy()}))
    edges = pd.concat(edges, ignore_index=True)
    keys = edges.groupby(["drug", "cell", "dose"]).agg(blocks=("block", "nunique"), pairs=("pair", "nunique"),
                                                        spread=("value", lambda v: v.max() - v.min()), value=("value", "mean"))
    curves = keys.groupby(level=[0, 1]).size()
    rows = [("all", "monotherapy_rows", len(mono)), ("all", "drug_cell_dose_keys", len(keys)),
            ("all", "blocks_per_key_min", keys.blocks.min()), ("all", "blocks_per_key_median", keys.blocks.median()),
            ("all", "blocks_per_key_max", keys.blocks.max()), ("all", "max_spread_across_blocks_pp", keys.spread.max()),
            ("all", "keys_in_one_pair", int((keys.pairs == 1).sum())), ("all", "drug_cell_curves", len(curves))]
    rows += [("all", f"curves_with_{n}_doses", int(c)) for n, c in curves.value_counts().sort_index().items()]
    errors = {"absent_mean": [], "absent_interpolated": [], "lodo_mean": [], "lodo_interpolated": [],
              "lodo_interior_mean": [], "lodo_interior_interpolated": []}
    for k in range(5):
        train_pairs = set(combination.pair_id[~combination.fold.isin([k, (k + 1) % 5])].astype(str))
        lookup = edges[edges.pair.isin(train_pairs)].groupby(["drug", "cell", "dose"]).value.mean()
        mean = float(lookup.mean())
        curve = {key: (np.log10(g.index.get_level_values("dose").to_numpy()), g.to_numpy()) for key, g in lookup.groupby(level=[0, 1])}
        test = combination[combination.fold == k]
        absent = np.zeros(len(test), bool)
        found = set()
        for e in "ab":
            key = list(zip(test["drug_" + e].astype(str), test.cell_line.astype(str), test["dose_" + e]))
            miss = ~pd.MultiIndex.from_tuples(key).isin(lookup.index)
            absent |= miss
            found |= {x for x, m in zip(key, miss) if m}
        for drug, cell, dose in sorted(found):
            x, v = curve[drug, cell]
            truth = keys.value[drug, cell, dose]
            errors["absent_mean"].append(mean - truth)
            errors["absent_interpolated"].append(np.interp(np.log10(dose), x, v) - truth)
        for x, v in curve.values():
            for i in range(len(x)) if len(x) >= 3 else []:
                error_mean, error_interpolated = mean - v[i], np.interp(x[i], np.delete(x, i), np.delete(v, i)) - v[i]
                errors["lodo_mean"].append(error_mean)
                errors["lodo_interpolated"].append(error_interpolated)
                if 0 < i < len(x) - 1:
                    errors["lodo_interior_mean"].append(error_mean)
                    errors["lodo_interior_interpolated"].append(error_interpolated)
        sub_rows = test[absent]
        rows += [(f"fold_{k}", "test_rows_with_absent_key", int(absent.sum())), (f"fold_{k}", "pairs", sub_rows.pair_id.nunique()),
                 (f"fold_{k}", "cells", sub_rows.cell_line.nunique()), (f"fold_{k}", "absent_keys", len(found)),
                 (f"fold_{k}", "training_lookup_mean", mean)]
    for name, values in errors.items():
        values = np.asarray(values)
        rows += [("pooled", f"{name}_n", len(values)), ("pooled", f"{name}_rmse", float(np.sqrt(np.mean(values ** 2)))),
                 ("pooled", f"{name}_mae", float(np.mean(np.abs(values))))]
    return pd.DataFrame(rows, columns=["scope", "statistic", "value"])


def pair_sums(p, t):
    e = t["y"] - p
    total = lambda w: np.bincount(t["pair"], weights=w, minlength=len(t["pairs"]))
    return np.column_stack([t["pair_n"], total(e * e), total(np.abs(e)), total(p), total(p * p), total(t["y"] * p)])


def from_sums(w, model, truth):
    """Pooled metrics for pair weights w (one vector or a matrix of bootstrap/jackknife weights)."""
    n, sse, sae, sp, spp, syp = np.moveaxis(w @ model, -1, 0)
    sy, syy, sr, srr = np.moveaxis(w @ truth, -1, 0)
    ss = syy - sy * sy / n
    with np.errstate(invalid="ignore"):  # near-constant predictors (training mean) have no correlation
        pearson = (syp - sy * sp / n) / np.sqrt(ss * (spp - sp * sp / n))
    return {"mse": sse / n, "rmse": np.sqrt(sse / n), "mae": sae / n, "r2": 1 - sse / ss, "pearson": pearson,
            "residual_r2": 1 - sse / (srr - sr * sr / n)}


def paired(a, b, t, weights, jackknife):
    """Seed-averaged squared-error change a - b on identical rows; negative favours a."""
    n = t["pair_n"]
    d = np.mean([m[:, 1] for m in a], 0) - np.mean([m[:, 1] for m in b], 0)
    point = d.sum() / n.sum()
    boot = np.quantile(weights @ d / (weights @ n), [.025, .975])
    jack = jackknife @ d / (jackknife @ n)
    half = student.ppf(.975, len(jack) - 1) * np.sqrt((len(jack) - 1) / len(jack) * ((jack - jack.mean()) ** 2).sum())
    folds = [d[t["pair_fold"] == k].sum() / n[t["pair_fold"] == k].sum() for k in range(5)]
    seeds = [(x[:, 1].sum() - y[:, 1].sum()) / n.sum() for x in a for y in b]
    evidence = [*boot, point - half, point + half, *folds, *seeds]
    verdict = "lower error" if max(evidence) < 0 else "higher error" if min(evidence) > 0 else "not established"
    return {"delta_mse": point, "ci_low": boot[0], "ci_high": boot[1], "jackknife_low": point - half, "jackknife_high": point + half,
            "folds_lower": int(sum(f < 0 for f in folds)), "fold_delta_min": min(folds), "fold_delta_max": max(folds),
            "seed_pairs_lower": f"{sum(s < 0 for s in seeds)}/{len(seeds)}", "seed_delta_min": min(seeds), "seed_delta_max": max(seeds),
            "pairs_lower_fraction": float(np.mean(d / n < 0)), "verdict": verdict}


def score(p, t, truth_sums, imputed):
    frame = t["panel_frame"].assign(prediction=np.bincount(t["panel"], weights=p) / t["panel_n"])
    panel, sums, e = regression_metrics(frame.response, frame.prediction), pair_sums(p, t), t["y"] - p
    return {"pooled": regression_metrics(t["y"], p), "sums": sums, "panel_prediction": frame.prediction.to_numpy(),
            "residual_r2": float(from_sums(np.ones(len(t["pairs"])), sums, truth_sums)["residual_r2"]),
            "cell_sse": np.bincount(t["cell"], weights=e * e, minlength=len(t["cells"])),
            "panel_rmse": panel["rmse"], "panel_pearson": panel["pearson"], "panel_r2": panel["r2"],
            "contrast_r2": interaction_contrast(frame, "response")["contrast_r2"],
            "imputed_rmse": float(np.sqrt(np.mean(e[imputed] ** 2))), "covered_rmse": float(np.sqrt(np.mean(e[~imputed] ** 2)))}


def run_rows(suite, protocol, chosen, planned):
    rows, seeds = [], protocol["benchmark"]["seeds"]
    selected = {(v["selected"], int(k)): group for group, folds in chosen.items() for k, v in folds.items()}
    for name, seed, fold in planned:
        path = run_dir(suite, name, seed, fold)
        meta, metrics = read_json(path / "metadata.json"), read_json(path / "evaluation" / "metrics.json")
        labels = [m for m in BENCHMARK if seed in seeds and model_paths(m, seed, chosen, suite)[fold] == path]
        rows.append({"model": name, "regime": meta["config"]["features"]["regime"], "seed": seed, "fold": fold,
                     "role": "selection" if seed == protocol["selection"]["seed"] else "benchmark",
                     "benchmark_models": ";".join(labels), "selected_in": selected.get((name, fold), "") if seed == protocol["selection"]["seed"] else "",
                     **run_fields(meta, metrics)})
    for path in sorted((RUNS / DOSE).glob("seed_*/fold_*")):
        meta, metrics = read_json(path / "metadata.json"), read_json(path / "evaluation" / "metrics.json")
        fit, pooled = meta["fit"], metrics["pooled"]
        rows.append({"model": DOSE, "regime": "F1", "seed": meta["seed"], "fold": meta["fold"], "role": DOSE, "benchmark_models": DOSE,
                     "selected_in": "", "validation_fold": meta["validation_fold"], "n": pooled["n"],
                     **{k: pooled[k] for k in ["mse", "rmse", "mae", "r2", "pearson"]}, "macro_pair_mse": metrics["macro_mse"]["pair"],
                     "rmse_ci_low": metrics["uncertainty"]["intervals"]["rmse"][0], "rmse_ci_high": metrics["uncertainty"]["intervals"]["rmse"][1],
                     "validation_mse": fit["validation_mse"], "best_epoch": fit["best_epoch"], "epochs_completed": fit["epochs_completed"],
                     "stopped": fit["stopped"], "parameters": fit["parameters"], "fit_seconds": fit["fit_seconds"],
                     "elapsed_seconds": meta["elapsed_seconds"], "cuda_max_allocated_mb": fit["cuda_max_memory_allocated_bytes"] / 2 ** 20,
                     "device": meta["device"]["name"], "commit": meta["git"]["commit"], "dirty": meta["git"]["dirty"]})
    for model, folder in CPU.items():
        for fold in range(5):
            path = RUNS / folder / f"fold_{fold}"
            meta, metrics = read_json(path / "metadata.json"), read_json(path / "evaluation" / "metrics.json")
            rows.append({"model": model, "regime": meta["config"]["features"]["regime"], "seed": meta["config"]["seed"], "fold": fold,
                         "role": "cpu_reference", "benchmark_models": model, "selected_in": "", **run_fields(meta, metrics)})
    return pd.DataFrame(rows)


def run_fields(meta, metrics):
    fit, pooled = meta["fit"], metrics["pooled"]
    epochs = meta["config"].get("training", {}).get("epochs")
    return {"validation_fold": (meta["config"]["test_fold"] + 1) % 5, "n": pooled["n"],
            **{k: pooled[k] for k in ["mse", "rmse", "mae", "r2", "pearson"]}, "macro_pair_mse": metrics["macro_mse"]["pair"],
            "rmse_ci_low": metrics["uncertainty"]["intervals"]["rmse"][0], "rmse_ci_high": metrics["uncertainty"]["intervals"]["rmse"][1],
            "validation_mse": fit.get("validation_mse"), "best_epoch": fit.get("best_epoch"), "epochs_completed": fit.get("epochs_completed"),
            "stopped": ("budget" if fit.get("epochs_completed") == epochs else "patience") if epochs else "",
            "parameters": fit.get("parameters"), "fit_seconds": fit.get("fit_seconds"), "elapsed_seconds": meta["elapsed_seconds"],
            "cuda_max_allocated_mb": fit["cuda_max_memory_allocated_bytes"] / 2 ** 20 if "cuda_max_memory_allocated_bytes" in fit else None,
            "device": meta["device"].get("name", meta["device"]["type"]) if "device" in meta else fit.get("device", "cpu"),
            "commit": meta["git"]["commit"], "dirty": meta["git"]["dirty"]}


def deepsynergy_table(root, labels):
    root, frame = Path(root), original_rows("published", labels, None)
    rows = []
    for k in range(5):
        metrics, pred = read_json(root / f"fold_{k}" / "metrics.json"), pd.read_parquet(root / f"fold_{k}" / "predictions.parquet")
        y, p = pred.y.to_numpy(float), pred.prediction.to_numpy(float)
        train_mean = frame.y[frame.fold != k].mean()
        rows.append({"fold": k, "validation_fold": metrics["validation_fold"], "test_rows": len(y),
                     "selected_epochs": metrics["selected"]["final_epochs"], "raw_minimum_epoch": metrics["selected"]["raw_minimum"],
                     "mse": metrics["symmetrized"]["mse"], "rmse": metrics["symmetrized"]["rmse"], "pearson": metrics["symmetrized"]["pearson"],
                     "r2": regression_metrics(y, p)["r2"], "doubled_rows_mse": metrics["doubled_rows_mse"],
                     "train_mean_mse": float(np.mean((y - train_mean) ** 2)), "target_sd": float(y.std()),
                     "stage1_seconds": metrics["stage_seconds"][0], "stage2_seconds": metrics["stage_seconds"][1]})
    table = pd.DataFrame(rows)
    pooled = pd.concat([pd.read_parquet(root / f"fold_{k}" / "predictions.parquet") for k in range(5)])
    whole = regression_metrics(pooled.y, pooled.prediction)
    summary = read_json(root / "reproduction.json")["mean_over_folds"]
    paper = read_json(root / "reproduction.json")["paper"]
    extra = pd.DataFrame([
        {"fold": "mean_over_folds", "test_rows": len(pooled), "mse": summary["mse"], "rmse": summary["rmse_mean"], "rmse_sd": summary["rmse_sd"],
         "pearson": summary["pearson_mean"], "pearson_sd": summary["pearson_sd"], "r2": table.r2.mean(),
         "doubled_rows_mse": table.doubled_rows_mse.mean(), "train_mean_mse": table.train_mean_mse.mean()},
        {"fold": "pooled_rows", "test_rows": whole["n"], "mse": whole["mse"], "rmse": whole["rmse"], "pearson": whole["pearson"], "r2": whole["r2"]},
        {"fold": "paper_table2", "mse": paper["mse"], "mse_ci_low": paper["mse_ci95"][0], "mse_ci_high": paper["mse_ci95"][1],
         "rmse": paper["rmse_mean"], "rmse_sd": paper["rmse_sd"], "pearson": paper["pearson_mean"], "pearson_sd": paper["pearson_sd"]}])
    return pd.concat([table, extra], ignore_index=True)


def learning_curves(suite, planned, deepsynergy):
    rows = []
    for name, seed, fold in planned:
        for r in read_json(run_dir(suite, name, seed, fold) / "checkpoints" / "history.json"):
            rows.append({"model": name, "seed": seed, "fold": fold, "epoch": r["epoch"], "train_mse": r["train_mse"],
                         "validation_mse": r["validation_mse"], "lr": r["lr"]})
    for name in [DOSE, DOSE + "_pilot"]:
        for path in sorted((RUNS / name).glob("seed_*/fold_*")):
            for r in read_json(path / "history.json"):
                rows.append({"model": name, "seed": int(path.parent.name[5:]), "fold": int(path.name[5:]), "epoch": r["epoch"],
                             "train_mse": r["train_mse"], "validation_mse": r["validation_mse"], "lr": DS_SETTINGS["lr"]})
    for k in range(5):
        for r in read_json(Path(deepsynergy) / f"fold_{k}" / "stage1_history.json"):
            rows.append({"model": "deepsynergy_stage1", "seed": 2026, "fold": k, "epoch": r["epoch"], "train_mse": r["train_mse"],
                         "validation_mse": r["validation_mse"], "lr": DS_SETTINGS["lr"]})
    return pd.DataFrame(rows)


def analyze(output, suite=RUNS / "benchmark_v1", deepsynergy=RUNS / "deepsynergy_original", protocol_path="configs/neural_benchmark.json",
            dataset="data/processed/oneil_v1", labels="data/raw/deepsynergy/labels.csv", repeats=1000, seed=2026, methods=None):
    output, suite, methods = Path(output), Path(suite), methods or {}
    report = read_json(output / "verification.json")
    dose_seeds = read_json(RUNS / DOSE / "protocol.json")["protocol"]["seeds"] if report.get(DOSE) == "verified" else []
    if not read_json(output / "verification.json")["passed"]:
        raise ValueError("Verification failed; run verify and resolve every failure first")
    protocol, chosen = read_json(protocol_path), read_json(suite / "selection.json")["selected"]
    planned = planned_runs(protocol, chosen)
    df, _, _ = load_truth(dataset)
    t = index(df)
    panel = pd.read_parquet(Path(dataset) / "panel.parquet", columns=["pair_id", "cell_line", "response"]).astype({"pair_id": str, "cell_line": str})
    check = t["panel_frame"].astype({"pair_id": str, "cell_line": str}).merge(panel, on=["pair_id", "cell_line"], validate="one_to_one")
    if len(check) != len(panel) or not np.allclose(check.response_x, check.response_y, rtol=0, atol=1e-9):
        raise ValueError("Row-mean pair-cell response differs from the prepared panel")

    # Training-pair Bliss reference and monotherapy status are training-data functions, identical across seeds.
    selection_seed, seeds = protocol["selection"]["seed"], protocol["benchmark"]["seeds"]
    reference = out_of_fold(model_paths("f2_residual", selection_seed, chosen, suite), t, "reference_offset")
    for model, s in [("f2_residual", seeds[1]), ("f2_residual", seeds[2]), ("mono_only", selection_seed)]:
        if not np.array_equal(reference, out_of_fold(model_paths(model, s, chosen, suite), t, "reference_offset")):
            raise ValueError("Bliss reference differs between runs")
    mean_status = out_of_fold(model_paths("f2_base_mean", selection_seed, chosen, suite), t, "monotherapy_status")
    interpolated_status = out_of_fold(model_paths("f2_base", selection_seed, chosen, suite), t, "monotherapy_status")
    imputed = mean_status == "mean_imputed"
    if not np.array_equal(imputed, interpolated_status == "interpolated"):
        raise ValueError("Mean-imputed and interpolated rows differ")
    r = t["y"] - reference
    sums = lambda w: np.bincount(t["pair"], weights=w, minlength=len(t["pairs"]))
    truth_sums = np.column_stack([sums(t["y"]), sums(t["y"] ** 2), sums(r), sums(r * r)])

    keys = [(name, selection_seed) for name in protocol["candidates"]] + [(m, s) for m in BENCHMARK for s in seeds]
    keys += [(m, read_json(RUNS / CPU[m] / "fold_0" / "config.json")["seed"]) for m in CPU] + [(DOSE, s) for s in dose_seeds]
    scores = {}
    for model, s in dict.fromkeys(keys):
        scores[model, s] = score(out_of_fold(model_paths(model, s, chosen, suite), t), t, truth_sums, imputed)
    scores["bliss_reference", None] = score(reference, t, truth_sums, imputed)
    for name, directory in methods.items():
        scores[name, None] = score(team_predictions(directory, t), t, truth_sums, imputed)
    reference_panel = np.bincount(t["panel"], weights=reference) / t["panel_n"]
    energy = contrast_energies(t, {"zero": np.zeros(len(reference_panel)), **{k: v["panel_prediction"] for k, v in scores.items()}})
    seeds_of = lambda m: [s for (x, s) in scores if x == m and (m not in BENCHMARK or s in seeds)]

    rng = np.random.default_rng(seed)
    weights = np.stack([np.bincount(rng.integers(len(t["pairs"]), size=len(t["pairs"])), minlength=len(t["pairs"])) for _ in range(repeats)]).astype(float)
    jackknife = np.array([[d not in pair for pair in t["pair_drugs"]] for d in t["drugs"]], dtype=float)

    # Runs, selection and controlled single-seed ablations (selection seed only).
    runs = run_rows(suite, protocol, chosen, planned)
    selection = []
    for group, folds in chosen.items():
        for k, entry in folds.items():
            ranked = sorted(entry["validation_mse"], key=lambda name: entry["validation_mse"][name])
            for name in ranked:
                fit = read_json(run_dir(suite, name, selection_seed, int(k)) / "metadata.json")["fit"]
                selection.append({"group": group, "fold": int(k), "candidate": name, "validation_mse": entry["validation_mse"][name],
                                  "parameters": fit["parameters"], "rank": ranked.index(name) + 1, "selected": name == entry["selected"]})
    selection = pd.DataFrame(selection)
    validation = runs[runs.role == "selection"].pivot(index="fold", columns="model", values="validation_mse")
    candidates = []
    for name, spec in protocol["candidates"].items():
        control = CONTROL.get(name)
        group = next((g for g, names in protocol["selection"]["groups"].items() if name in names), "")
        pooled = scores[name, selection_seed]["pooled"]
        own = runs[(runs.model == name) & (runs.role == "selection")]
        row = {"candidate": name, "regime": own.regime.iloc[0], "group": group or "ablation only", "purpose": spec["purpose"],
               "control": control or "", "parameters": int(own.parameters.iloc[0]), "mean_validation_mse": validation[name].mean(),
               "folds_selected": int(selection[(selection.candidate == name) & selection.selected].shape[0]),
               "mean_epochs": own.epochs_completed.mean(), "budget_stops": int((own.stopped == "budget").sum()),
               **{"test_" + k: pooled[k] for k in ["rmse", "mse", "mae", "r2", "pearson"]},
               "test_residual_r2": scores[name, selection_seed]["residual_r2"]}
        if control:
            delta = validation[name] - validation[control]
            change = paired([scores[name, selection_seed]["sums"]], [scores[control, selection_seed]["sums"]], t, weights, jackknife)
            row.update({"validation_delta_mean": delta.mean(), "validation_delta_min": delta.min(), "validation_delta_max": delta.max(),
                        "validation_folds_lower": int((delta < 0).sum()), "test_delta_mse": change["delta_mse"],
                        "test_delta_ci_low": change["ci_low"], "test_delta_ci_high": change["ci_high"], "test_folds_lower": change["folds_lower"]})
        if name == "f0_ordered":
            swapped = out_of_fold(model_paths(name, selection_seed, chosen, suite), t, "prediction_swapped")
            own_pred = out_of_fold(model_paths(name, selection_seed, chosen, suite), t)
            row["order_rms_difference"] = float(np.sqrt(np.mean((own_pred - swapped) ** 2)))
        candidates.append(row)
    candidates = pd.DataFrame(candidates)

    # Benchmark summary: every planned model and the frozen references on identical out-of-fold rows.
    labels_map = {"selected:F0": "selected F0 (" + "/".join(sorted({v["selected"] for v in chosen["F0"].values()})) + ")",
                  "selected:F2": "selected F2 (" + "/".join(sorted({v["selected"] for v in chosen["F2"].values()})) + ")",
                  DOSE: "DeepSynergy, dose-aware"}
    benchmark, groups = [], []
    listed = BENCHMARK + ([DOSE] if dose_seeds else []) + ["cpu_f0", "cpu_f2", "additive", "mean", "bliss_reference"] + list(methods)
    for model in listed:
        keys = [(model, s) for s in (seeds_of(model) if model != "bliss_reference" else [None])]
        own = [scores[k] for k in keys]
        mats = [o["sums"] for o in own]
        boot = {m: np.quantile(np.mean([from_sums(weights, x, truth_sums)[m] for x in mats], 0), [.025, .975]) for m in ["mse", "rmse", "residual_r2"]}
        fold = {f"rmse_fold_{k}": float(np.mean([from_sums((t["pair_fold"] == k).astype(float), x, truth_sums)["rmse"] for x in mats])) for k in range(5)}
        pair_mse = np.mean([x[:, 1] for x in mats], 0) / t["pair_n"]
        cell_mse = np.mean([o["cell_sse"] for o in own], 0) / np.bincount(t["cell"])
        row = {"model": model, "label": labels_map.get(model, model), "seeds": ";".join(str(s) for _, s in keys if s is not None) or "none",
               "n": own[0]["pooled"]["n"]}
        for m in ["rmse", "mse", "mae", "r2", "pearson", "contrast_r2"]:
            values = [o["pooled"][m] if m in o["pooled"] else o[m] for o in own]
            row[m], row[m + "_seed_sd"] = float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None
        jack_mse = np.mean([(jackknife @ x[:, 1]) / (jackknife @ x[:, 0]) for x in mats], 0)
        contrast = np.mean([1 - energy[k] / energy["zero"] for k in keys], 0)
        within = np.mean([1 - energy[k] / energy["bliss_reference", None] for k in keys], 0)
        versus = contrast - (1 - energy["bliss_reference", None] / energy["zero"])
        row.update({"rmse_jackknife_low": jackknife_interval(row["rmse"], np.sqrt(jack_mse))[0],
                    "rmse_jackknife_high": jackknife_interval(row["rmse"], np.sqrt(jack_mse))[1],
                    "contrast_r2_jackknife_low": jackknife_interval(contrast[0], contrast[1:])[0],
                    "contrast_r2_jackknife_high": jackknife_interval(contrast[0], contrast[1:])[1],
                    "contrast_r2_minus_bliss": versus[0], "contrast_r2_minus_bliss_jackknife_low": jackknife_interval(versus[0], versus[1:])[0],
                    "contrast_r2_minus_bliss_jackknife_high": jackknife_interval(versus[0], versus[1:])[1],
                    "interaction_residual_r2": within[0], "interaction_residual_r2_jackknife_low": jackknife_interval(within[0], within[1:])[0],
                    "interaction_residual_r2_jackknife_high": jackknife_interval(within[0], within[1:])[1]})
        row.update({"rmse_ci_low": boot["rmse"][0], "rmse_ci_high": boot["rmse"][1], "mse_ci_low": boot["mse"][0], "mse_ci_high": boot["mse"][1],
                    "residual_r2": float(np.mean([o["residual_r2"] for o in own])), "residual_r2_ci_low": boot["residual_r2"][0],
                    "residual_r2_ci_high": boot["residual_r2"][1], "macro_pair_mse": float(pair_mse.mean()),
                    "median_pair_rmse": float(np.median(np.sqrt(pair_mse))), "macro_cell_mse": float(cell_mse.mean()),
                    **{k: float(np.mean([o[k] for o in own])) for k in ["panel_rmse", "panel_pearson", "panel_r2", "imputed_rmse", "covered_rmse"]},
                    **fold})
        benchmark.append(row)
        groups += [{"model": model, "level": "pair", "group": p, "fold": int(f), "drug_a": a, "drug_b": b, "n": int(n), "mse": v}
                   for p, f, (a, b), n, v in zip(t["pairs"], t["pair_fold"], t["pair_drugs"], t["pair_n"], pair_mse)]
        groups += [{"model": model, "level": "cell", "group": c, "n": int(n), "mse": v} for c, n, v in zip(t["cells"], np.bincount(t["cell"]), cell_mse)]
    benchmark, groups = pd.DataFrame(benchmark), pd.DataFrame(groups)
    mse = benchmark.set_index("model").mse
    comparisons = []
    declared = COMPARISONS + (DOSE_COMPARISONS if dose_seeds else []) + [(m, c, f"team method vs {c}") for m in methods for c in TEAM_CONTROLS]
    for model, control, question in declared:
        a = [scores[k]["sums"] for k in scores if k[0] == model and (k[1] in seeds or model not in BENCHMARK)]
        b = [scores[k]["sums"] for k in scores if k[0] == control and (k[1] in seeds or control not in BENCHMARK)]
        change = paired(a, b, t, weights, jackknife)
        comparisons.append({"model": model, "control": control, "question": question, "model_mse": mse[model], "control_mse": mse[control],
                            "delta_rmse": benchmark.set_index("model").rmse[model] - benchmark.set_index("model").rmse[control],
                            "relative_mse_change": change["delta_mse"] / mse[control], **change})
    comparisons = pd.DataFrame(comparisons)

    # One table for the team report: every method on identical rows, seed spread kept apart from grouped intervals.
    table, final = benchmark.set_index("model"), []
    for model in [m for m in FINAL if m in table.index] + list(methods):
        r = table.loc[model]
        own = [scores[k]["sums"] for k in scores if k[0] == model and (k[1] in seeds or model not in BENCHMARK)]
        entry = {"model": model, "label": r.label, "inputs": INPUTS.get(model, "team method"), "seeds": r.seeds,
                 **{c: r[c] for c in ["rmse", "rmse_seed_sd", "rmse_ci_low", "rmse_ci_high", "rmse_jackknife_low", "rmse_jackknife_high",
                                      "mae", "r2", "pearson", "panel_rmse", "panel_pearson", "residual_r2", "contrast_r2",
                                      "contrast_r2_jackknife_low", "contrast_r2_jackknife_high", "interaction_residual_r2"]}}
        for control, prefix in [("bliss_reference", "vs_bliss"), ("selected:F2", "vs_selected_f2")]:
            if model != control:
                other = [scores[k]["sums"] for k in scores if k[0] == control and (k[1] in seeds or control not in BENCHMARK)]
                change = paired(own, other, t, weights, jackknife)
                entry.update({f"{prefix}_mse_change": change["delta_mse"] / table.mse[control], f"{prefix}_verdict": change["verdict"]})
        final.append(entry)
    final = pd.DataFrame(final)

    output.mkdir(parents=True, exist_ok=True)
    tables = {"final": final, "runs": runs, "selection": selection, "candidates": candidates, "benchmark": benchmark, "comparisons": comparisons,
              "monotherapy": monotherapy_audit(dataset),
              "groups": groups, "learning_curves": learning_curves(suite, planned, deepsynergy), "deepsynergy": deepsynergy_table(deepsynergy, labels)}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False, float_format="%.10g", lineterminator="\n")
    print(benchmark[["label", "rmse", "rmse_seed_sd", "rmse_ci_low", "rmse_ci_high", "r2", "pearson", "residual_r2"]].to_string())
    print(comparisons[["model", "control", "delta_mse", "ci_low", "ci_high", "jackknife_low", "jackknife_high", "verdict"]].to_string())
    return tables


# ---- figures (saved tables only) ---------------------------------------------------------------

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SHORT = {"f0_plateau": "F0 plateau LR", "f0_deep": "F0 two-layer dose encoder", "f0_wide": "F0 wide (512/256, encoder 64)",
         "f0_regularized": "F0 dropout 0.3, wd 1e-4", "f0_combined": "F0 all changes", "f0_ordered": "F0 ordered pair input*",
         "f2_base": "monotherapy added (F2 vs F0)", "f2_base_mean": "F2 mean-imputed absent doses", "f2_plateau": "F2 plateau LR",
         "f2_deep": "F2 two-layer dose encoder", "f2_wide": "F2 wide (512/256, encoder 64)", "f2_regularized": "F2 dropout 0.3, wd 1e-4",
         "f2_combined": "F2 all changes", "f2_residual": "F2 Bliss residual target", "f2_refinput": "F2 Bliss/HSA head inputs",
         "mono_dose": "no identities (vs F2 residual)*", "mono_only": "no doses either (vs no identities)*"}


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": .8,
                         "axes.labelcolor": SECONDARY, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titlecolor": INK, "axes.titlelocation": "left",
                         "legend.frameon": False, "lines.linewidth": 1.4, "text.color": INK})
    return plt


def _save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")


def figures(output):
    plt, output = _pyplot(), Path(output)
    read = lambda name: pd.read_csv(output / f"{name}.csv")
    curves, selection, runs, candidates = read("learning_curves"), read("selection"), read("runs"), read("candidates")
    benchmark, comparisons, groups, ds = read("benchmark"), read("comparisons"), read("groups"), read("deepsynergy")
    dose = [DOSE] if DOSE in set(benchmark.model) else []
    team = read("final").query("inputs == 'team method'").model.tolist()
    (output / "figures").mkdir(exist_ok=True)

    # 1. Validation learning curves of the selection runs (selection seed), best epoch marked.
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8), sharex=True, sharey="row", constrained_layout=True)
    for i, group in enumerate(["F0", "F2"]):
        names = selection[selection.group == group].candidate.unique()
        shown = [f"{group.lower()}_base", f"{group.lower()}_wide", f"{group.lower()}_combined"]
        own = curves[curves.model.isin(names) & (curves.seed == 2026)]
        low = own.groupby(["model", "fold"]).validation_mse.min().min()
        for k in range(5):
            ax = axes[i, k]
            others = [n for n in names if n not in shown]
            for name in others:
                c = own[(own.model == name) & (own.fold == k)]
                ax.plot(c.epoch, c.validation_mse, color=AXIS, lw=.8, label="other candidates" if name == others[0] else None)
            for j, name in enumerate(shown):
                c = own[(own.model == name) & (own.fold == k)]
                ax.plot(c.epoch, c.validation_mse, color=SERIES[j], label=name)
                best = c.loc[c.validation_mse.idxmin()]
                ax.plot(best.epoch, best.validation_mse, "o", ms=4, color=SERIES[j], mec=SURFACE, mew=1)
            chosen = selection[(selection.group == group) & (selection.fold == k) & selection.selected].candidate.iloc[0]
            ax.set_title(f"{group} · test fold {k}\nselected: {chosen}", color=INK)
            ax.set_ylim(low - 5, low + 85)
            ax.set_xlabel("epoch" if i else "")
        axes[i, 0].set_ylabel("validation MSE")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    axes[1, 0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Validation learning curves, selection seed 2026 (dot: best epoch, the evaluated checkpoint; y-axis clipped above)",
                 x=0, ha="left", fontsize=9, color=INK)
    _save(fig, output / "figures" / "fig1_learning_curves.png")

    # 2. Controlled ablations: one factor against its control, validation (selection data) and test (seed 2026).
    rows = candidates[candidates.control.notna()].copy()
    rows["order"] = rows.candidate.map({n: i for i, n in enumerate(SHORT)})
    rows = rows.sort_values("order", ascending=False)
    own = runs[runs.role == "selection"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True, constrained_layout=True)
    for ax, column, title in [(axes[0], "validation_mse", "Validation MSE change (used for selection)"),
                              (axes[1], "mse", "Test MSE change, selection seed (not used for selection)")]:
        table = own.pivot(index="fold", columns="model", values=column)
        for y, r in enumerate(rows.itertuples()):
            delta = table[r.candidate] - table[r.control]
            ax.scatter(delta, np.full(5, y), s=14, color=SERIES[0], alpha=.45, linewidths=0, zorder=2)
            if column == "mse":
                ax.plot([r.test_delta_ci_low, r.test_delta_ci_high], [y, y], color=SERIES[0], lw=1.6, zorder=3)
                ax.plot(r.test_delta_mse, y, "D", ms=5, color=SERIES[0], mec=SURFACE, mew=1, zorder=4)
            else:
                ax.plot(delta.mean(), y, "D", ms=5, color=SERIES[0], mec=SURFACE, mew=1, zorder=4)
        ax.axvline(0, color=SECONDARY, lw=.8, zorder=1)
        ax.set_title(title)
        ax.set_xlabel("candidate minus control (negative = lower error)")
    axes[0].set_yticks(range(len(rows)), [f"{SHORT[n]}" for n in rows.candidate])
    fig.suptitle("Controlled single-factor changes (dots: folds; diamond: mean or pooled with pair-bootstrap 95% CI; * ablation, not eligible)",
                 x=0, ha="left", fontsize=9)
    _save(fig, output / "figures" / "fig2_ablations.png")

    # 3. Five-fold, multi-seed benchmark on identical rows; points are individual fold/seed runs.
    order = ["cpu_f0", "f0_base", "selected:F0", *dose, "cpu_f2", "f2_base_mean", "f2_base", "f2_residual", "selected:F2", "mono_only", *team]
    table = benchmark.set_index("model")
    fig, ax = plt.subplots(figsize=(8, .45 * len(order) + .2), constrained_layout=True)
    for y, model in enumerate(reversed(order)):
        points = runs[runs.benchmark_models.fillna("").str.split(";").apply(lambda labels: model in labels)]
        ax.scatter(points.rmse, y + (points.seed - 2027) * .12, s=12, color=AXIS, linewidths=0, zorder=2)
        r = table.loc[model]
        ax.plot([r.rmse_ci_low, r.rmse_ci_high], [y, y], color=SERIES[0], lw=1.8, zorder=3)
        ax.plot(r.rmse, y, "D", ms=6, color=SERIES[0], mec=SURFACE, mew=1, zorder=4)
        ax.annotate(f"{r.rmse:.2f}", (r.rmse, y), xytext=(0, 7), textcoords="offset points", ha="center", color=SECONDARY, fontsize=7)
    ax.set_yticks(range(len(order)), [table.loc[m, "label"] + (" [CPU, 1 seed]" if m.startswith("cpu") else "") for m in reversed(order)])
    ax.set_xlabel("test RMSE, inhibition percentage points")
    ax.set_title("Five-fold out-of-fold RMSE (grey: one fold and seed; diamond: seed-mean pooled RMSE, pair-bootstrap 95% CI)")
    _save(fig, output / "figures" / "fig3_benchmark.png")

    # 4. Paired per-pair squared-error changes for the declared comparisons.
    pair = groups[groups.level == "pair"].pivot(index="group", columns="model", values="mse")
    shown = comparisons[~comparisons.question.str.startswith("replication")].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    for y, r in enumerate(shown.itertuples()):
        delta = pair[r.model] - pair[r.control]
        q = np.quantile(delta, [.05, .25, .5, .75, .95])
        ax.plot([q[0], q[4]], [y, y], color=AXIS, lw=1, zorder=1)
        ax.add_patch(plt.Rectangle((q[1], y - .18), q[3] - q[1], .36, color=GRID, zorder=2))
        ax.plot([q[2], q[2]], [y - .18, y + .18], color=SECONDARY, lw=1, zorder=3)
        ax.plot([r.ci_low, r.ci_high], [y, y], color=SERIES[0], lw=1.8, zorder=4)
        ax.plot(r.delta_mse, y, "D", ms=5.5, color=SERIES[0], mec=SURFACE, mew=1, zorder=5)
        ax.annotate(f"{r.pairs_lower_fraction:.0%} of pairs lower", (q[4], y), xytext=(6, 0), textcoords="offset points",
                    va="center", color=SECONDARY, fontsize=7)
    ax.axvline(0, color=SECONDARY, lw=.8, zorder=0)
    ax.set_yticks(range(len(shown)), [f"{r.model} − {r.control}" for r in shown.itertuples()])
    ax.set_xlabel("MSE change per drug pair (model minus control; negative = lower error)")
    ax.set_title("Per-pair error change: box 25–75%, whiskers 5–95%, line median; diamond: pooled change with pair-bootstrap 95% CI")
    _save(fig, output / "figures" / "fig4_paired_errors.png")

    # 5. Prediction beyond the training-pair Bliss reference and beyond additive drug effects.
    order = ["bliss_reference", "mono_only", "cpu_f0", "f0_base", "selected:F0", *dose, "cpu_f2", "f2_base_mean", "f2_base", "f2_residual",
             "selected:F2", *team]
    fig, axes = plt.subplots(1, 3, figsize=(13, .4 * len(order) + .5), sharey=True, constrained_layout=True)
    panels = [("residual_r2", "residual_r2_ci_low", "residual_r2_ci_high"),
              ("contrast_r2", "contrast_r2_jackknife_low", "contrast_r2_jackknife_high"),
              ("interaction_residual_r2", "interaction_residual_r2_jackknife_low", "interaction_residual_r2_jackknife_high")]
    for y, model in enumerate(reversed(order)):
        r = table.loc[model]
        for ax, (value, low, high) in zip(axes, panels):
            ax.plot([r[low], r[high]], [y, y], color=SERIES[0], lw=1.8)
            ax.plot(r[value], y, "D", ms=5.5, color=SERIES[0], mec=SURFACE, mew=1)
    for ax in axes:
        ax.axvline(0, color=SECONDARY, lw=.8, zorder=0)
    axes[1].axvline(table.loc["bliss_reference", "contrast_r2"], color=MUTED, lw=.8, ls="--", zorder=0)
    axes[0].set_yticks(range(len(order)), [table.loc[m, "label"] for m in reversed(order)])
    axes[0].set_title("Dose-level R² of y − Bliss(training-pair monotherapy)")
    axes[1].set_title("Pair-cell interaction-contrast R² (drug main effects removed;\ndashed: Bliss)")
    axes[2].set_title("Share of Bliss interaction-contrast error removed")
    axes[0].set_xlabel("residual R² (95% pair-bootstrap CI)")
    axes[1].set_xlabel("contrast R² (95% delete-one-drug jackknife CI)")
    axes[2].set_xlabel("interaction residual R² (95% delete-one-drug jackknife CI)")
    _save(fig, output / "figures" / "fig5_beyond_reference.png")

    # 6. DeepSynergy reproduction: stage-1 validation curves and per-fold test MSE against the paper.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
    folds = ds[ds.fold.astype(str).str.isdigit()].astype({"fold": int})
    stage = curves[curves.model == "deepsynergy_stage1"]
    for k in range(5):
        c = stage[stage.fold == k]
        axes[0].plot(c.epoch + 1, c.validation_mse, color=SERIES[k], lw=1, label=f"fold {k}")
        chosen = int(folds.set_index("fold").selected_epochs[k])
        axes[0].plot(chosen, c.validation_mse.iloc[chosen - 1], "o", ms=4.5, color=SERIES[k], mec=SURFACE, mew=1)
    axes[0].set(xlabel="epoch", ylabel="validation MSE (published synergy)", ylim=(stage.validation_mse.min() - 20, stage.validation_mse.min() + 260))
    axes[0].legend(ncol=5, loc="upper right", fontsize=7)
    axes[0].set_title("Stage 1 (3 training folds), dot: smoothed epoch choice used for the 4-fold refit")
    paper, mean = ds.set_index("fold").loc["paper_table2"], ds.set_index("fold").loc["mean_over_folds"]
    axes[1].axvspan(paper.mse_ci_low, paper.mse_ci_high, color=GRID, zorder=0)
    axes[1].axvline(paper.mse, color=SECONDARY, lw=1, zorder=1)
    axes[1].scatter(folds.mse, np.ones(5), s=16, color=AXIS, zorder=2)
    axes[1].plot(mean.mse, 1, "D", ms=6, color=SERIES[0], mec=SURFACE, mew=1, zorder=3)
    axes[1].annotate(f"reproduction {mean.mse:.1f}", (mean.mse, 1), xytext=(0, 9), textcoords="offset points", ha="center", color=SECONDARY, fontsize=7)
    axes[1].annotate(f"paper {paper.mse:.1f}\n95% CI band", (paper.mse_ci_high, 1.3), xytext=(4, 0), textcoords="offset points",
                     ha="left", color=SECONDARY, fontsize=7)
    axes[1].set(yticks=[], ylim=(.5, 1.5), xlabel="test MSE, published Loewe synergy")
    axes[1].set_title("Folds (grey) and mean over folds vs paper Table 2")
    _save(fig, output / "figures" / "fig6_deepsynergy.png")

    # 7. Dose-aware DeepSynergy on the shared rows: validation curves and per-fold test RMSE against the networks and Bliss.
    if dose:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
        own, pilot = curves[curves.model == DOSE], curves[curves.model == DOSE + "_pilot"]
        for (seed, k), c in own.groupby(["seed", "fold"]):
            axes[0].plot(c.epoch + 1, c.validation_mse, color=SERIES[k], lw=.9, alpha=.85, label=f"fold {k}" if seed == 2026 else None)
            best = c.loc[c.validation_mse.idxmin()]
            axes[0].plot(best.epoch + 1, best.validation_mse, "o", ms=3.5, color=SERIES[k], mec=SURFACE, mew=.8)
        if len(pilot):
            axes[0].plot(pilot.epoch + 1, pilot.validation_mse, color=INK, lw=1, ls="--", label="pilot (fold 0)")
        axes[0].set(xlabel="epoch", ylabel="validation MSE (order-averaged)")
        axes[0].legend(ncol=3, loc="upper right", fontsize=7)
        axes[0].set_title("Validation curves, 3 seeds per fold; dot: best epoch, the evaluated checkpoint")
        for j, model in enumerate(["bliss_reference", DOSE, "selected:F0", "selected:F2"]):
            r = table.loc[model]
            axes[1].plot(range(5), [r[f"rmse_fold_{k}"] for k in range(5)], "o-", ms=4, color=SERIES[j], label=r.label)
        axes[1].set(xticks=range(5), xlabel="test fold", ylabel="test RMSE, seed mean")
        axes[1].legend(loc="best", fontsize=7)
        axes[1].set_title("Per-fold test RMSE on identical rows")
        _save(fig, output / "figures" / "fig7_deepsynergy_dose.png")
    plt.close("all")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["verify", "analyze", "figures"])
    p.add_argument("--output", default="results/neural_benchmark")
    p.add_argument("--suite", default=str(RUNS / "benchmark_v1"))
    p.add_argument("--deepsynergy", default=str(RUNS / "deepsynergy_original"))
    p.add_argument("--method", action="append", default=[], metavar="NAME=DIR",
                   help="team predictions: DIR/fold_<k>.parquet with observation_id and prediction for every test row")
    a = p.parse_args()
    if a.command == "verify":
        verify(a.output, a.suite, a.deepsynergy)
    elif a.command == "analyze":
        analyze(a.output, a.suite, a.deepsynergy, methods=dict(m.split("=", 1) for m in a.method))
    else:
        figures(a.output)


if __name__ == "__main__":
    main()
