"""GPU readiness: device execution, real-data training, restoration and CPU agreement.

python -m synergy_artifacts.gpu --output results/neural/runs/gpu_check

Tolerances are float32 rounding scales or, for training, a fraction of the
prediction change caused by a different initialization seed. Bitwise CPU/GPU
equality is not expected.
"""
import argparse
import copy
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .features import FeatureBuilder
from .io import read_json, write_json
from .models import make_network
from .runner import run
from .training import NeuralRegressor, device_report, resolve_device

SUBSET = {"cells": ["A2058", "A2780"], "blocks_per_pair_cell": 1}
REFERENCE_RUNS = ["results/neural/runs/phase2_f0/fold_0", "results/neural/runs/phase2_f2/fold_0"]


def driver_report():
    for name, args in [("nvidia-smi", ["--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"]),
                       ("rocm-smi", ["--showproductname", "--showdriverversion", "--showmeminfo", "vram"])]:
        exe = shutil.which(name)
        if exe:
            try:
                return {"tool": name, "output": subprocess.run([exe, *args], capture_output=True, text=True, timeout=60).stdout.strip()}
            except (OSError, subprocess.SubprocessError) as error:
                return {"tool": name, "error": str(error)}
    return None


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def check_tensor(device):
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2048, 1024, generator=g), torch.randn(1024, 512, generator=g)
    exact = a.double() @ b.double()
    product = a.to(device) @ b.to(device)
    _sync(device)
    location = str(product.device)
    error = lambda x: float((x.double().cpu() - exact).norm() / exact.norm())
    cpu_error, device_error = error(a @ b), error(product)
    tolerance = max(10 * cpu_error, 1e-6)
    return {"computed_on": location, "relative_error_vs_float64": {"cpu": cpu_error, "device": device_error},
            "tolerance": tolerance, "rationale": "float32 GEMM rounding; device error within 10x CPU error",
            "passed": product.device.type == device.type and device_error <= tolerance}


def check_step(device):
    spec = dict(kind="dose", endpoint_dim=42, cell_dim=39, hidden=[128, 64], encoder_dim=32, dropout=0.)
    torch.manual_seed(1)
    reference = make_network(spec)
    g = torch.Generator().manual_seed(2)
    inputs = [torch.randn(1024, 42, generator=g), torch.randn(1024, 42, generator=g), torch.randn(1024, 39, generator=g)]
    target = torch.randn(1024, generator=g)
    outcome = {}
    for name, where in [("cpu", torch.device("cpu")), ("device", device)]:
        for optimizer_name in ["sgd", "adam"]:
            model = copy.deepcopy(reference).to(where)
            optimizer = (torch.optim.SGD(model.parameters(), lr=1e-2) if optimizer_name == "sgd"
                         else torch.optim.Adam(model.parameters(), lr=1e-3))
            output = model(*[x.to(where) for x in inputs])
            loss = (output - target.to(where)).square().mean()
            loss.backward()
            gradients = [p.grad.detach().cpu().clone() for p in model.parameters()]
            optimizer.step()
            _sync(where)
            outcome[name, optimizer_name] = {"output": output.detach().cpu(), "loss": float(loss.detach()), "gradients": gradients,
                                             "parameters": [p.detach().cpu().clone() for p in model.parameters()],
                                             "on": str(next(model.parameters()).device)}
    cpu, dev = outcome["cpu", "sgd"], outcome["device", "sgd"]
    relative = lambda x, y: float((x - y).norm() / max(float(y.norm()), 1e-30))
    forward = relative(dev["output"], cpu["output"])
    gradient = max(relative(x, y) for x, y in zip(dev["gradients"], cpu["gradients"]))
    sgd = max(float((x - y).abs().max()) for x, y in zip(dev["parameters"], cpu["parameters"]))
    adam = max(float((x - y).abs().max()) for x, y in zip(outcome["device", "adam"]["parameters"], outcome["cpu", "adam"]["parameters"]))
    changed = all(not torch.equal(x, y) for x, y in zip(dev["parameters"], [p.detach() for p in reference.parameters()]))
    passed = forward <= 1e-5 and gradient <= 1e-4 and sgd <= 1e-6 and adam <= 2e-3 and changed and dev["on"].startswith(device.type)
    return {"parameters_on": dev["on"], "relative_forward_difference": forward, "max_relative_gradient_difference": gradient,
            "sgd_step_max_parameter_difference": sgd, "adam_step_max_parameter_difference": adam,
            "all_parameters_updated": changed, "loss": {"cpu": cpu["loss"], "device": dev["loss"]},
            "tolerances": {"forward": 1e-5, "gradient": 1e-4, "sgd_step": 1e-6, "adam_step": 2e-3},
            "rationale": "float32 reduction order; Adam's first step is ~lr*sign(gradient), so near-zero gradients may differ by 2*lr",
            "passed": passed}


def subset_config(base, device, epochs, seed=2026, dropout=None, **features):
    config = copy.deepcopy(read_json(base))
    config.update(seed=seed, subset=dict(SUBSET), bootstrap=20)
    config["features"].update(features)
    config["training"].update(epochs=epochs, batch_size=256, patience=epochs + 1, device=str(device), threads=4)
    if dropout is not None:
        config["model"]["dropout"] = dropout
    return config


def _predictions(path):
    return pd.read_parquet(Path(path) / "predictions.parquet").sort_values("observation_id").reset_index(drop=True)


def restore(run_dir, device, frame):
    model, context = NeuralRegressor.load(Path(run_dir) / "checkpoints" / "best.pt", device)
    builder = FeatureBuilder.from_state_dict(context["features"])
    offset_kind = context["config"]["features"].get("reference_offset")
    offset = builder.reference(frame, offset_kind) if offset_kind else 0.
    return model.predict(builder.transform(frame)) + offset


def check_real_training(root, device, base_f0, base_f2):
    report = {}
    for name, config in [("f0", subset_config(base_f0, device, 5)),
                         ("f2_interpolated", subset_config(base_f2, device, 5, mono_missing="interpolate"))]:
        output = root / f"real_{name}"
        metadata, metrics = run(config, output)
        frame = _predictions(output)
        same_device = restore(output, device, frame)
        on_cpu = restore(output, "cpu", frame)
        reported = frame.prediction.to_numpy()
        report[name] = {"device": metadata["fit"]["device"], "train_rows": metadata["partition_rows"]["train"],
                        "test_rows": metrics["pooled"]["n"], "test_rmse": metrics["pooled"]["rmse"],
                        "epochs": metadata["fit"]["epochs_completed"], "fit_seconds": metadata["fit"]["fit_seconds"],
                        "cuda_max_memory_allocated_bytes": metadata["fit"].get("cuda_max_memory_allocated_bytes"),
                        "restored_same_device_exact": bool(np.array_equal(same_device, reported)),
                        "restored_cpu_max_abs_difference": float(np.abs(on_cpu - reported).max()),
                        "tolerance_cpu_restore": 1e-3,
                        "passed": metadata["fit"]["device"].startswith(device.type) and np.isfinite(reported).all()
                        and np.array_equal(same_device, reported) and float(np.abs(on_cpu - reported).max()) <= 1e-3}
    return report


def _rms(x, y):
    return float(np.sqrt(np.mean((np.asarray(x) - np.asarray(y)) ** 2)))


def check_agreement(root, device, base_f0):
    """Identical seed/batch order on CPU and device versus a different-seed CPU run."""
    report = {}
    for dropout in [0., .1]:
        runs = {}
        for name, where, seed in [("cpu", "cpu", 2026), ("device", device, 2026), ("cpu_other_seed", "cpu", 2027)]:
            output = root / f"agreement_dropout{dropout:g}_{name}"
            run(subset_config(base_f0, where, 6, seed, dropout), output)
            runs[name] = (_predictions(output).prediction.to_numpy(),
                          [r["validation_mse"] for r in read_json(output / "checkpoints" / "history.json")])
        device_change = _rms(runs["device"][0], runs["cpu"][0])
        seed_change = _rms(runs["cpu_other_seed"][0], runs["cpu"][0])
        trajectory = max(abs(a - b) / b for a, b in zip(runs["device"][1], runs["cpu"][1]))
        entry = {"device_vs_cpu_prediction_rms": device_change, "seed_vs_seed_prediction_rms": seed_change,
                 "ratio": device_change / seed_change, "max_relative_validation_mse_difference": trajectory}
        if dropout == 0.:
            entry.update(tolerance_ratio=.1, passed=device_change <= .1 * seed_change,
                         rationale="without dropout, device rounding must stay an order of magnitude below seed variation")
        else:
            entry.update(passed=True, rationale="dropout masks use device-specific random streams; informational")
        report[f"dropout_{dropout:g}"] = entry
    return report


def check_reference_runs(device, runs):
    report = {}
    for run_dir in map(Path, runs):
        if not (run_dir / "checkpoints" / "best.pt").exists():
            report[str(run_dir)] = {"skipped": "reference CPU run absent", "passed": True}
            continue
        truth = pd.read_parquet(run_dir / "evaluation" / "predictions_with_truth.parquet")
        stored = truth.prediction.to_numpy()
        started = time.perf_counter()
        restored = restore(run_dir, device, truth.drop(columns=["inhibition", "prediction"]))
        seconds = time.perf_counter() - started
        y = truth.inhibition.to_numpy()
        rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))
        difference = float(np.abs(restored - stored).max())
        report[str(run_dir)] = {"rows": len(truth), "max_abs_prediction_difference": difference,
                                "cpu_rmse": rmse(stored), "device_rmse": rmse(restored),
                                "inference_seconds": seconds, "tolerances": {"prediction": 1e-3, "rmse": 1e-4},
                                "passed": difference <= 1e-3 and abs(rmse(stored) - rmse(restored)) <= 1e-4}
    return report


def check_resume(root, device, base_f0):
    config = subset_config(base_f0, device, 4)
    run(config, root / "resume_full")
    run(config, root / "resume_repeat")
    short = copy.deepcopy(config)
    short["training"]["epochs"] = 2
    run(short, root / "resume_split")
    run(config, root / "resume_split", resume=root / "resume_split" / "checkpoints" / "latest.pt")
    full, repeat, split = (_predictions(root / name).prediction.to_numpy() for name in ["resume_full", "resume_repeat", "resume_split"])
    return {"repeat_bitwise_identical": bool(np.array_equal(full, repeat)),
            "resume_bitwise_identical": bool(np.array_equal(full, split)),
            "resume_max_abs_difference": float(np.abs(full - split).max()),
            "passed": bool(np.array_equal(full, split) and np.array_equal(full, repeat))}


def check_throughput(root, device, configs, epochs):
    report = {}
    for name, (base, overrides) in configs.items():
        config = copy.deepcopy(read_json(base))
        for section, values in overrides.items():
            config[section].update(values)
        config["training"].update(epochs=epochs, patience=epochs + 1, device=str(device), eval_batch_size=16384)
        config["bootstrap"] = 20
        metadata, _ = run(config, root / f"throughput_{name}")
        history = read_json(root / f"throughput_{name}" / "checkpoints" / "history.json")
        seconds = [r["seconds"] for r in history]
        report[name] = {"train_rows": metadata["partition_rows"]["train"], "parameters": metadata["fit"]["parameters"],
                        "epoch_seconds": seconds, "train_rows_per_second": metadata["partition_rows"]["train"] / float(np.median(seconds)),
                        "cuda_max_memory_allocated_bytes": metadata["fit"].get("cuda_max_memory_allocated_bytes"),
                        "cuda_max_memory_reserved_bytes": metadata["fit"].get("cuda_max_memory_reserved_bytes"),
                        "elapsed_seconds": metadata["elapsed_seconds"], "passed": True}
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reference-runs", nargs="*", default=REFERENCE_RUNS)
    p.add_argument("--throughput-epochs", type=int, default=2)
    p.add_argument("--f0-config", default="configs/neural_dose_aware.json")
    p.add_argument("--f2-config", default="configs/neural_monotherapy.json")
    a = p.parse_args()
    root = Path(a.output)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    device = resolve_device(a.device)
    report = {"platform": platform.platform(), "python": platform.python_version(), "device": device_report(device),
              "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count(),
              "driver": driver_report(), "checks": {}}
    write_json(root / "report.json", report)
    checks = [("tensor_execution", lambda: check_tensor(device)),
              ("forward_backward_optimizer", lambda: check_step(device)),
              ("real_subset_training_and_restore", lambda: check_real_training(root, device, a.f0_config, a.f2_config)),
              ("cpu_device_training_agreement", lambda: check_agreement(root, device, a.f0_config)),
              ("production_checkpoint_inference", lambda: check_reference_runs(device, a.reference_runs)),
              ("resume_and_repeat_determinism", lambda: check_resume(root, device, a.f0_config)),
              ("full_fold_throughput", lambda: check_throughput(root, device, {
                  "f0_base": (a.f0_config, {}),
                  "f2_wide_deep": (a.f2_config, {"features": {"mono_missing": "interpolate"},
                                                 "model": {"hidden": [512, 256], "encoder_dim": 64, "encoder_layers": 2}})},
                  a.throughput_epochs))]
    for name, check in checks:
        started = time.perf_counter()
        try:
            result = check()
        except Exception as error:  # Record and continue: the report is the deliverable.
            result = {"error": f"{type(error).__name__}: {error}", "passed": False}
        result = result if "passed" in result else {**result, "passed": all(v.get("passed", False) for v in result.values())}
        result["seconds"] = time.perf_counter() - started
        report["checks"][name] = result
        write_json(root / "report.json", report)
        print(f"{name}: {'PASS' if result['passed'] else 'FAIL'} ({result['seconds']:.1f}s)", flush=True)
    report["passed"] = all(c["passed"] for c in report["checks"].values())
    write_json(root / "report.json", report)
    print(f"GPU check {'PASSED' if report['passed'] else 'FAILED'}: {root / 'report.json'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
