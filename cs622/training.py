"""Bounded minibatch training, validation stopping and exact epoch-boundary recovery."""
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from .features import LazyFeatures
from .models import make_network

# cuBLAS reads this when CUDA initializes; deterministic CUDA matrix products need it.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def resolve_device(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices are cpu and cuda (CUDA or ROCm builds)")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def device_report(device):
    device = torch.device(device)
    report = {"type": device.type, "torch": torch.__version__, "cuda": torch.version.cuda,
              "hip": getattr(torch.version, "hip", None), "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        report.update(name=properties.name, capability=f"{properties.major}.{properties.minor}",
                      total_memory_bytes=int(properties.total_memory), cudnn=torch.backends.cudnn.version(),
                      index=device.index if device.index is not None else torch.cuda.current_device())
    return report


def seed_all(seed, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    # IEEE float32 products on every device; TF32 would silently change CUDA results.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _tensors(features, device="cpu"):
    return tuple(torch.as_tensor(x, dtype=torch.float32).to(device) for x in features)


def feature_rows(features):
    return len(features) if isinstance(features, LazyFeatures) else len(features[0])


def feature_batch(features, indices, device):
    if isinstance(features, LazyFeatures):
        arrays = features.batch(indices.cpu().numpy() if isinstance(indices, torch.Tensor) else indices)
    else:
        arrays = [x[indices] for x in features]
    return [torch.as_tensor(x, dtype=torch.float32, device=device) for x in arrays]


def make_scheduler(optimizer, spec):
    if not spec:
        return None
    if spec.get("kind") != "plateau":
        raise ValueError("Only validation-plateau learning-rate reduction is implemented")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=spec.get("factor", .5), patience=spec.get("patience", 5),
        threshold=spec.get("threshold", 1e-4), min_lr=spec.get("min_lr", 0.))


class NeuralRegressor:
    def __init__(self, model_spec, training, seed=2026):
        self.spec, self.config, self.seed = model_spec, training, seed
        seed_all(seed, training.get("threads", 4))
        self.device = resolve_device(training.get("device", "auto"))
        self.model = make_network(model_spec).to(self.device)

    def predict(self, features):
        self.model.eval()
        values = []
        batch_size = self.config.get("eval_batch_size", self.config.get("batch_size", 256))
        with torch.no_grad():
            for start in range(0, feature_rows(features), batch_size):
                batch = feature_batch(features, slice(start, start+batch_size), self.device)
                output = self.model(*batch)
                if not torch.isfinite(output).all():
                    raise ValueError("Nonfinite neural prediction")
                values.append(output.cpu().numpy())
        return np.concatenate(values) * self.target_scale + self.target_mean

    def fit(self, features, target, validation, output, context, resume=None):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        y = np.asarray(target, dtype=np.float32)
        xv, yv = validation
        yv = np.asarray(yv, dtype=float)
        if not len(y) or not len(yv) or not np.isfinite(y).all() or not np.isfinite(yv).all():
            raise ValueError("Training/validation targets must be finite and nonempty")
        if feature_rows(features) != len(y) or feature_rows(xv) != len(yv):
            raise ValueError("Feature/target row mismatch")
        for data in [features, xv]:
            if not isinstance(data, LazyFeatures) and not all(np.isfinite(x).all() for x in data):
                raise ValueError("Nonfinite training/validation features")
        self.target_mean = float(y.mean()) if self.config.get("normalize_target", True) else 0.
        self.target_scale = max(float(y.std()), 1e-8) if self.config.get("normalize_target", True) else 1.
        # Dense features stay resident on the training device; batch order is still drawn on CPU,
        # so CPU and CUDA runs with one seed visit identical minibatches.
        resident = not isinstance(features, LazyFeatures)
        x = features if not resident else _tensors(features, self.device)
        if not isinstance(xv, LazyFeatures):
            xv = _tensors(xv, self.device)
        y_tensor = torch.tensor((y - self.target_mean) / self.target_scale)
        if resident:
            y_tensor = y_tensor.to(self.device)
        optimizer_name = self.config.get("optimizer", "adam")
        kwargs = {"lr": self.config.get("lr", .001), "weight_decay": self.config.get("weight_decay", 0.)}
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), momentum=self.config.get("momentum", 0.), **kwargs)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.model.parameters(), **kwargs)
        else:
            raise ValueError("Unknown optimizer")
        scheduler = make_scheduler(optimizer, self.config.get("scheduler"))
        generator = torch.Generator().manual_seed(self.seed)
        best, since, start, best_epoch = float("inf"), 0, 0, -1
        best_state = None
        history = []
        identity = {"spec": self.spec, "context": context, "seed": self.seed,
                    "training": {k: v for k, v in self.config.items() if k != "epochs"}}
        if resume:
            checkpoint = torch.load(resume, map_location=self.device, weights_only=False)
            if checkpoint["identity"] != identity:
                raise ValueError("Resume data/features/config differ from checkpoint")
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            self.target_mean, self.target_scale = checkpoint["target_mean"], checkpoint["target_scale"]
            generator.set_state(checkpoint["generator"].cpu())
            torch.set_rng_state(checkpoint["rng"].cpu())
            if self.device.type == "cuda":
                # map_location moved the saved states to the device; generators accept CPU byte tensors only.
                torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
            start, best, since = checkpoint["epoch"] + 1, checkpoint["best"], checkpoint["since"]
            best_state, best_epoch, history = checkpoint["best_model"], checkpoint["best_epoch"], checkpoint["history"]
        if start >= self.config["epochs"]:
            raise ValueError("Resume budget must exceed completed epochs")
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        batch_size = self.config.get("batch_size", 256)
        patience = self.config.get("patience", 20)
        parameters = list(self.model.parameters())
        fit_started = time.perf_counter()
        for epoch in range(start, self.config["epochs"]):
            if since >= patience:
                break
            epoch_started = time.perf_counter()
            self.model.train()
            order = torch.randperm(len(y), generator=generator)
            if resident:
                order = order.to(self.device)
            total, norms = 0., []
            for indices in order.split(batch_size):
                batch = feature_batch(x, indices, self.device)
                truth = y_tensor[indices].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                if self.model.augment:
                    # Two supervised orderings, as in the published doubled training set.
                    loss = ((self.model.ordered(*batch) - truth).square().mean()
                            + (self.model.ordered(batch[1], batch[0], batch[2]) - truth).square().mean()) / 2
                else:
                    loss = (self.model(*batch) - truth).square().mean()
                loss.backward()
                gradients = [p.grad for p in parameters if p.grad is not None]
                finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all()
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in gradients]))
                # One host synchronization per step: loss value, gradient norm, finiteness.
                loss_value, norm_value, finite_value = torch.stack([loss.detach(), norm, finite.float()]).tolist()
                if not math.isfinite(loss_value):
                    raise ValueError("Nonfinite training loss")
                if finite_value != 1.:
                    raise ValueError("Nonfinite gradient")
                if self.config.get("gradient_clip"):
                    torch.nn.utils.clip_grad_norm_(parameters, self.config["gradient_clip"])
                optimizer.step()
                total += loss_value * len(indices)
                norms.append(norm_value)
            validation_mse = float(np.mean((self.predict(xv) - yv) ** 2))
            if not np.isfinite(validation_mse):
                raise ValueError("Nonfinite validation loss")
            record = {"epoch": epoch, "train_mse": total / len(y) * self.target_scale ** 2,
                      "validation_mse": validation_mse, "lr": optimizer.param_groups[0]["lr"],
                      "grad_norm_mean": float(np.mean(norms)), "grad_norm_max": float(np.max(norms))}
            if scheduler is not None:
                scheduler.step(validation_mse)
            history.append(record)
            if validation_mse < best - self.config.get("min_delta", 0.):
                best, since, best_epoch = validation_mse, 0, epoch
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                since += 1
            record["seconds"] = time.perf_counter() - epoch_started
            payload = {"identity": identity, "model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict() if scheduler is not None else None,
                       "target_mean": self.target_mean, "target_scale": self.target_scale, "epoch": epoch,
                       "best": best, "since": since, "best_epoch": best_epoch, "best_model": best_state,
                       "history": history, "generator": generator.get_state(), "rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else []}
            save_checkpoint(output / "latest.pt", payload)
            if since == 0:
                save_checkpoint(output / "best.pt", payload)
            (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if epoch == start or (epoch + 1) % 10 == 0:
                print(f"epoch={epoch} train_mse={record['train_mse']:.5g} validation_mse={best:.5g}", flush=True)
        self.model.load_state_dict(best_state)
        self.fit_summary = {"best_epoch": best_epoch, "validation_mse": best, "epochs_completed": len(history),
                            "device": str(self.device), "parameters": sum(p.numel() for p in self.model.parameters()),
                            "fit_seconds": time.perf_counter() - fit_started}
        if self.device.type == "cuda":
            self.fit_summary["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated(self.device))
            self.fit_summary["cuda_max_memory_reserved_bytes"] = int(torch.cuda.max_memory_reserved(self.device))
        return self

    @classmethod
    def load(cls, path, device="cpu"):
        # Checkpoints are trusted local artifacts, never arbitrary downloaded pickles.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        identity = checkpoint["identity"]
        config = {**identity["training"], "device": device}
        obj = cls(identity["spec"], config, identity["seed"])
        obj.model.load_state_dict(checkpoint["best_model"])
        obj.target_mean, obj.target_scale = checkpoint["target_mean"], checkpoint["target_scale"]
        return obj, identity["context"]
