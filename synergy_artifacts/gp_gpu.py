"""Exact GP on every training row of a cell line, in float64 torch: an extension of Kingsley Ukandu's GP.

python -m synergy_artifacts.gp_gpu study --config configs/gp_gpu.json --group coverage --output results/neural/runs/gp_gpu_study
python -m synergy_artifacts.gp_gpu run --config configs/gp_gpu.json --variant full --output results/neural/runs/gp_full

Kept from his model (gaussian_process/src/project_utils.py): the kernel ConstantKernel * SymmetricDrugDoseKernel
+ WhiteKernel with the start values and bounds read from his make_gp, alpha, normalize_y, one L-BFGS-B run from
his start values, one GP per cell line, his ArmDoseTransformer and cap_training_blocks. As in synergy_artifacts.gp_shared,
RBF vs Matern 5/2 is chosen per cell by validation RMSE on fold (f+1)%5 and the selected fit predicts the test fold.

Changed, per declared variant:
- rows: hyperparameters are fit on every training row (or a declared cap). Replicates share an input, so each
  distinct input enters through its mean, count and within-replicate sum of squares. The likelihood and posterior
  are unchanged; about 22,500 rows become about 5,700 inputs per cell.
- hyper_rows: fit hyperparameters on a cap, then condition exactly on every row.
- dose "per_drug": log dose standardized within each drug on training rows.
- monotherapy (F2): the training-pair single-agent response as a third arm coordinate with its own length scale.
- target "bliss_residual" (F2): the GP models inhibition minus the Bliss reference, which is added back.
  Both F2 options use synergy_artifacts.features.FeatureBuilder's F2 lookup with interpolation, the information of the
  neural F2 models and the Bliss reference.
- inducing: DTC/VFE predictive distributions (Titsias 2009) under the full fit's hyperparameters, with random
  subsets of the distinct training inputs, to measure approximation error against the exact posterior.
"""
import argparse
import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from .controls import bliss_reference
from .features import FeatureBuilder
from .gp_shared import KEEP, _provenance, gp_module, shared_frame, uncertainty_metrics
from .io import digest, write_json
from .runner import load_experiment

DEFAULT = {"rows": "all", "hyper_rows": None, "dose": "pooled", "monotherapy": False, "target": "inhibition",
           "families": ["rbf", "matern"], "interaction_weight": 1.0, "shared_similarity": 0.25, "inducing": []}
NAMES = ["constant", "length_scale", "noise", "mono_length_scale"]
LOG2PI = math.log(2 * math.pi)


@functools.cache
def his_settings():
    """Start values, bounds and alpha of his make_gp; the monotherapy length scale reuses the dose bounds."""
    reg = gp_module().make_gp("rbf").named_steps["regressor"]
    c, pair, white = reg.kernel.k1.k1, reg.kernel.k1.k2, reg.kernel.k2
    start = {"constant": c.constant_value, "length_scale": pair.length_scale, "noise": white.noise_level}
    bounds = {"constant": c.constant_value_bounds, "length_scale": pair.length_scale_bounds,
              "noise": white.noise_level_bounds}
    start["mono_length_scale"], bounds["mono_length_scale"] = pair.length_scale, pair.length_scale_bounds
    return start, {k: tuple(map(float, v)) for k, v in bounds.items()}, float(reg.alpha)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder:
    """His ArmDoseTransformer, optionally with per-drug doses and a monotherapy coordinate. Fit on training rows.

    transform returns (rows, 2 arms, coordinates): drug code, dose, and optionally monotherapy.
    """

    def __init__(self, dose="pooled", mono=None):
        if dose not in {"pooled", "per_drug"}:
            raise ValueError("dose must be pooled or per_drug")
        self.dose, self.mono = dose, mono

    def _logs(self, frame):
        return pd.DataFrame({"drug": np.r_[frame.drug_a.to_numpy(), frame.drug_b.to_numpy()],
                             "x": np.log10(np.r_[frame.dose_a.to_numpy(float), frame.dose_b.to_numpy(float)])})

    def _mono(self, frame):
        values = np.column_stack([self.mono.monotherapy_values(frame, e) for e in ["a", "b"]])
        if not np.isfinite(values).all():
            raise ValueError("Missing training-pair monotherapy after interpolation")
        return values

    def fit(self, train):
        self.pre = gp_module().ArmDoseTransformer().fit(train[gp_module().FEATURES])
        if self.dose == "per_drug":
            stats = self._logs(train).groupby("drug").x.agg(["mean", "std"])
            self.drug_mean = stats["mean"].to_dict()
            self.drug_std = stats["std"].fillna(0).where(lambda s: s > 0, 1.0).to_dict()
        if self.mono is not None:
            values = self._mono(train)
            self.mono_mean, self.mono_std = float(values.mean()), float(values.std()) or 1.0
        return self

    def transform(self, frame):
        X = self.pre.transform(frame[gp_module().FEATURES])
        arms = [[X[:, 0], X[:, 1]], [X[:, 2], X[:, 3]]]
        if self.dose == "per_drug":
            for arm, e in zip(arms, ["a", "b"]):
                drugs = frame["drug_" + e]
                arm[1] = (np.log10(frame["dose_" + e].to_numpy(float)) - drugs.map(self.drug_mean).to_numpy(float)) \
                    / drugs.map(self.drug_std).to_numpy(float)
        if self.mono is not None:
            values = (self._mono(frame) - self.mono_mean) / self.mono_std
            arms[0].append(values[:, 0])
            arms[1].append(values[:, 1])
        return np.stack([np.column_stack(arms[0]), np.column_stack(arms[1])], axis=1)


def collapse(X, y):
    """Distinct inputs up to arm exchange, with replicate means, counts and within-replicate sums of squares."""
    a, b = X[:, 0], X[:, 1]
    swap, tied = np.zeros(len(X), bool), np.ones(len(X), bool)
    for j in range(X.shape[2]):
        swap |= tied & (a[:, j] > b[:, j])
        tied &= a[:, j] == b[:, j]
    canon = np.where(swap[:, None, None], X[:, ::-1], X)
    flat, inverse, counts = np.unique(canon.reshape(len(X), -1), axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    mean = np.bincount(inverse, y) / counts
    ss = np.bincount(inverse, (y - mean[inverse]) ** 2)
    return flat.reshape(len(flat), 2, -1), mean, counts.astype(float), ss


def _dose(t, family):
    if family == "rbf":
        return torch.exp(-0.5 * t * t)
    r = math.sqrt(5) * t
    return (1 + r + r * r / 3) * torch.exp(-r)


class PairKernel:
    """His SymmetricDrugDoseKernel in torch; every continuous arm coordinate gets a product dose-kernel factor."""

    def __init__(self, family, shared_similarity, interaction_weight):
        if family not in {"rbf", "matern"}:
            raise ValueError("family must be rbf or matern")
        self.family, self.rho, self.w = family, shared_similarity, interaction_weight

    def arm(self, u, v, scales):
        k = self.rho + (1 - self.rho) * (u[:, None, 0] == v[None, :, 0]).to(u.dtype)
        for j in range(1, u.shape[1]):
            k = k * _dose((u[:, None, j] - v[None, :, j]).abs() / scales[j - 1], self.family)
        return k

    def __call__(self, X, Y, scales):
        ac, ad = self.arm(X[:, 0], Y[:, 0], scales), self.arm(X[:, 0], Y[:, 1], scales)
        bc, bd = self.arm(X[:, 1], Y[:, 0], scales), self.arm(X[:, 1], Y[:, 1], scales)
        return (ac + ad + bc + bd) / 4 + self.w * (ac * bd + ad * bc) / 2

    def diag(self, X, scales):
        u, v = X[:, 0], X[:, 1]
        cross = self.rho + (1 - self.rho) * (u[:, 0] == v[:, 0]).to(u.dtype)
        for j in range(1, u.shape[1]):
            cross = cross * _dose((u[:, j] - v[:, j]).abs() / scales[j - 1], self.family)
        return (1 + cross) / 2 + self.w * (1 + cross * cross) / 2


class ExactGP:
    """ML-II on collapsed replicates, then the exact posterior; outputs in the original target units."""

    def __init__(self, family, method, dev=None):
        self.method, self.dev = method, dev or device()
        self.kernel = PairKernel(family, method["shared_similarity"], method["interaction_weight"])
        self.start, self.bounds, self.alpha = his_settings()

    def _t(self, x):
        return torch.as_tensor(x, dtype=torch.float64, device=self.dev)

    def _unpack(self, theta):
        p = torch.exp(theta)
        return p[0], p[[1] + ([3] if len(p) > 3 else [])], p[2]

    def _data(self, X, y):
        Xu, ybar, m, ss = collapse(X, (np.asarray(y, float) - self.y_mean) / self.y_std)
        return self._t(Xu), self._t(ybar), self._t(m), self._t(ss)

    def nll(self, theta, data):
        """Negative log marginal likelihood of all rows, from the collapsed statistics; None if not PD."""
        Xu, ybar, m, ss = data
        c, scales, noise = self._unpack(theta)
        s2 = noise + self.alpha
        K = c * self.kernel(Xu, Xu, scales) + torch.diag(s2 / m)
        L, info = torch.linalg.cholesky_ex(K)
        if info.item():
            return None
        a = torch.cholesky_solve(ybar[:, None], L)[:, 0]
        n = len(ybar)
        lml = (-0.5 * ybar @ a - torch.log(torch.diagonal(L)).sum() - 0.5 * n * LOG2PI
               - 0.5 * ((m - 1) * (LOG2PI + torch.log(s2))).sum() - 0.5 * torch.log(m).sum() - (ss / (2 * s2)).sum())
        return -lml

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.y_mean, self.y_std = float(y.mean()), float(y.std()) or 1.0
        data = self._data(X, y)
        names = NAMES[:4 if X.shape[2] > 2 else 3]
        theta0 = np.log([self.start[k] for k in names])
        bounds = np.log([self.bounds[k] for k in names])
        self.evaluations = 0

        def objective(theta):
            self.evaluations += 1
            t = self._t(theta).requires_grad_(True)
            value = self.nll(t, data)
            if value is None:
                return np.inf, np.zeros_like(theta)
            value.backward()
            return value.item(), t.grad.cpu().numpy()

        self.result = minimize(objective, theta0, method="L-BFGS-B", jac=True, bounds=bounds)
        self.theta, self.names, self.n_hyper_rows, self.n_hyper_inputs = self.result.x, names, len(y), len(data[0])
        self.at_bound = int(np.isclose(self.theta[:, None], bounds).any(1).sum())
        self.lml = -float(self.result.fun)
        return self.condition(X, y)

    def condition(self, X, y):
        """Exact posterior on these rows under the fitted hyperparameters and target normalization."""
        self.Xu, ybar, m, _ = self._data(X, y)
        c, self.scales, self.noise = self._unpack(self._t(self.theta))
        self.c = c
        K = c * self.kernel(self.Xu, self.Xu, self.scales) + torch.diag((self.noise + self.alpha) / m)
        self.L = torch.linalg.cholesky(K)
        self.a = torch.cholesky_solve(ybar[:, None], self.L)[:, 0]
        self.n_rows, self.n_inputs = len(y), len(ybar)
        self._m, self._ybar = m, ybar
        return self

    def hyperparameters(self):
        return dict(zip(self.names, np.exp(self.theta).tolist()))

    def predict(self, X, chunk=4096):
        """Mean, predictive std (latent plus noise, his return_std) and latent std, in target units."""
        out = []
        with torch.no_grad():
            for s in range(0, len(X), chunk):
                Xs = self._t(X[s:s + chunk])
                Ks = self.c * self.kernel(self.Xu, Xs, self.scales)
                v = torch.linalg.solve_triangular(self.L, Ks, upper=False)
                var = (self.c * self.kernel.diag(Xs, self.scales) - (v * v).sum(0)).clamp_min(0)
                out.append(torch.stack([Ks.T @ self.a, torch.sqrt(var + self.noise), torch.sqrt(var)], 1).cpu().numpy())
        mean, std, latent = np.concatenate(out).T
        return mean * self.y_std + self.y_mean, std * self.y_std, latent * self.y_std

    def inducing(self, Z, jitter=1e-6):
        return InducingPosterior(self, Z, jitter)


class InducingPosterior:
    """DTC/VFE predictive distribution with inducing inputs Z under an ExactGP's hyperparameters and rows."""

    def __init__(self, gp, Z, jitter=1e-6):
        self.gp = gp
        with torch.no_grad():
            self.Z = gp._t(Z)
            Kuu = gp.c * gp.kernel(self.Z, self.Z, gp.scales)
            scale = float(torch.diagonal(Kuu).mean())
            for j in [jitter, 1e-5, 1e-4, 1e-3]:
                self.L, info = torch.linalg.cholesky_ex(Kuu + j * scale * torch.eye(len(Kuu), dtype=Kuu.dtype, device=Kuu.device))
                if not info.item():
                    self.jitter = j
                    break
            else:
                raise RuntimeError("Inducing covariance is not positive definite")
            A = torch.linalg.solve_triangular(self.L, gp.c * gp.kernel(self.Z, gp.Xu, gp.scales), upper=False)
            dinv = gp._m / (gp.noise + gp.alpha)
            self.LB = torch.linalg.cholesky(torch.eye(len(Z), dtype=A.dtype, device=A.device) + (A * dinv) @ A.T)
            inner = torch.cholesky_solve((A @ (dinv * gp._ybar))[:, None], self.LB)
            self.w = torch.linalg.solve_triangular(self.L.T, inner, upper=True)[:, 0]

    def predict(self, X, chunk=4096):
        gp, out = self.gp, []
        with torch.no_grad():
            for s in range(0, len(X), chunk):
                Xs = gp._t(X[s:s + chunk])
                Kzs = gp.c * gp.kernel(self.Z, Xs, gp.scales)
                v = torch.linalg.solve_triangular(self.L, Kzs, upper=False)
                u = torch.linalg.solve_triangular(self.LB, v, upper=False)
                var = (gp.c * gp.kernel.diag(Xs, gp.scales) - (v * v).sum(0) + (u * u).sum(0)).clamp_min(0)
                out.append(torch.stack([Kzs.T @ self.w, torch.sqrt(var + gp.noise), torch.sqrt(var)], 1).cpu().numpy())
        mean, std, latent = np.concatenate(out).T
        return mean * gp.y_std + gp.y_mean, std * gp.y_std, latent * gp.y_std


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def fit_cell(cell, mono, fold, method, seed=42, dev=None, predict_test=True):
    """One cell line and test fold. cell: that cell's shared combination rows; mono: its monotherapy rows.

    Returns (test predictions or None, validation predictions of the selected family, one row per fit/predictor).
    """
    gp, dev = gp_module(), dev or device()
    method = {**DEFAULT, **method}
    name, val_fold = cell.cell_line.iloc[0], (fold + 1) % 5
    available = cell[~cell.fold.isin([fold, val_fold])]
    validation, test = cell[cell.fold == val_fold], cell[cell.fold == fold]
    gp.assert_split(available, validation)
    gp.assert_split(available, test)
    rows = available if method["rows"] == "all" else gp.cap_training_blocks(available, int(method["rows"]), seed + fold)
    hyper = rows if method["hyper_rows"] is None else gp.cap_training_blocks(available, int(method["hyper_rows"]), seed + fold)
    builder = None
    if method["monotherapy"] or method["target"] == "bliss_residual":
        pairs = set(available.pair_id)
        builder = FeatureBuilder("F2", mono_missing="interpolate").fit(available, mono[mono.pair_id.isin(pairs)])

    def offset(frame):
        if method["target"] != "bliss_residual":
            return np.zeros(len(frame))
        a, b = [builder.monotherapy_values(frame, e) for e in ["a", "b"]]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("Missing training-pair monotherapy for the Bliss offset")
        return bliss_reference(a, b)

    enc = Encoder(method["dose"], builder if method["monotherapy"] else None).fit(rows)
    X_rows, X_hyper, X_val = enc.transform(rows), enc.transform(hyper), enc.transform(validation)
    y_rows, y_hyper = rows.inhibition.to_numpy(float) - offset(rows), hyper.inhibition.to_numpy(float) - offset(hyper)
    val_offset = offset(validation)
    fits, records = {}, []
    for family in method["families"]:
        if dev.type == "cuda":
            torch.zeros(1, device=dev)
            torch.cuda.reset_peak_memory_stats(dev)
        _sync(dev)
        started = time.perf_counter()
        model = ExactGP(family, method, dev).fit(X_hyper, y_hyper)
        if method["hyper_rows"] is not None:
            model.condition(X_rows, y_rows)
        _sync(dev)
        fitted = time.perf_counter()
        mean, std, latent = model.predict(X_val)
        _sync(dev)
        base = {"cell_line": name, "fold": fold, "validation_fold": val_fold, "family": family,
                "n_rows_hyper": model.n_hyper_rows, "n_inputs_hyper": model.n_hyper_inputs, "n_rows": model.n_rows,
                "n_inputs": model.n_inputs, "n_validation": len(validation), "n_test": len(test), **model.hyperparameters(),
                "lml": model.lml, "at_bound": model.at_bound, "nit": int(model.result.nit), "evaluations": model.evaluations,
                "converged": bool(model.result.success), "fit_seconds": fitted - started,
                "predict_seconds": time.perf_counter() - fitted,
                "peak_gpu_mb": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None}
        fits[family] = (model, mean + val_offset, std, latent)
        records.append({**base, "predictor": "exact", "n_inducing": None,
                        **{f"validation_{k}": v for k, v in uncertainty_metrics(validation.inhibition, mean + val_offset, std).items()}})
        rng = np.random.default_rng(seed + fold)
        for size in method["inducing"]:
            pick = rng.permutation(len(model.Xu))[:size]
            started = time.perf_counter()
            post = model.inducing(model.Xu[pick].cpu().numpy())
            m, s, _ = post.predict(X_val)
            _sync(dev)
            records.append({**base, "predictor": f"inducing_{size}", "n_inducing": len(pick), "jitter": post.jitter,
                            "inducing_seconds": time.perf_counter() - started,
                            **{f"validation_{k}": v for k, v in uncertainty_metrics(validation.inhibition, m + val_offset, s).items()}})
    family = min(method["families"], key=lambda f: next(r["validation_rmse"] for r in records
                                                         if r["family"] == f and r["predictor"] == "exact"))
    for r in records:
        r["selected"] = r["family"] == family
    model, val_mean, val_std, val_latent = fits[family]
    val_pred = validation[KEEP].assign(prediction=val_mean, predictive_std=val_std, latent_std=val_latent)
    predictions = None
    if predict_test:
        mean, std, latent = model.predict(enc.transform(test))
        predictions = test[KEEP].assign(prediction=mean + offset(test), predictive_std=std, latent_std=latent)
    return predictions, val_pred, records


def load_cells(dataset, cells=None):
    df, mono, path = load_experiment({"dataset": str(dataset), "table": "observations", "target": "inhibition",
                                      "test_fold": 0, "model": {"kind": "dose"}})
    frame = shared_frame(df)
    names = cells or sorted(frame.cell_line.unique())
    return {c: (frame[frame.cell_line == c], mono[mono.cell_line == c]) for c in names}, frame, path


def _provenance_gpu(path, dev):
    info = _provenance(path)
    info["versions"]["torch"] = torch.__version__
    info["device"] = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    return info


def load_plan(configs):
    """A config and its amendments: variants and groups are added, a later selection replaces an earlier one."""
    configs = [configs] if isinstance(configs, (str, Path)) else list(configs)
    plan = json.loads(Path(configs[0]).read_text())
    for path in configs[1:]:
        extra = json.loads(Path(path).read_text())
        plan["variants"] = {**plan["variants"], **extra.get("variants", {})}
        plan["groups"] = {**plan["groups"], **extra.get("groups", {})}
        plan["selection"] = extra.get("selection", plan["selection"])
    return plan


def study(config, group, output, gpu=0):
    """Validation-only fits of one declared group on the study fold; the test fold is never predicted."""
    plan = load_plan(config)
    spec = plan["groups"][group]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{group}.csv"
    if target.exists():
        raise FileExistsError(f"{target} exists; choose a new output")
    dev = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    cells, _, path = load_cells(plan["dataset"], spec.get("cells"))
    started, rows = time.perf_counter(), []
    for variant in spec["variants"]:
        method = plan["variants"][variant]
        for name, (cell, mono) in cells.items():
            _, _, records = fit_cell(cell, mono, plan["study_fold"], method, plan["seed"], dev, predict_test=False)
            rows += [{"group": group, "variant": variant, **r} for r in records]
            pd.DataFrame(rows).to_csv(output / f"{group}.partial.csv", index=False)
            print(f"{time.strftime('%H:%M:%S')} {variant} {name} {records[0]['fit_seconds']:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(target, index=False)
    (output / f"{group}.partial.csv").unlink()
    configs = [config] if isinstance(config, (str, Path)) else list(config)
    write_json(output / f"{group}.json", {"config": [str(c) for c in configs], "config_sha256": [digest(c) for c in configs], "group": group,
                                          "spec": spec, "seconds": time.perf_counter() - started,
                                          **_provenance_gpu(path, dev)})


def run(config, variant, output, folds, gpu=0):
    """Five-fold production of one declared variant: test and validation predictions of the selected family."""
    plan = load_plan(config)
    configs = [config] if isinstance(config, (str, Path)) else list(config)
    method = {**DEFAULT, **plan["variants"][variant]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"fold_{k}.parquet").exists() for k in folds):
        raise FileExistsError("Fold predictions exist; choose a new output")
    dev = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    cells, frame, path = load_cells(plan["dataset"])
    started, summary = time.perf_counter(), []
    for k in folds:
        tests, vals = [], []
        for name, (cell, mono) in cells.items():
            pred, val, records = fit_cell(cell, mono, k, method, plan["seed"], dev)
            tests.append(pred)
            vals.append(val)
            summary += records
            print(f"{time.strftime('%H:%M:%S')} fold {k} {name}", flush=True)
        fold = pd.concat(tests, ignore_index=True)
        if fold.observation_id.duplicated().any() or set(fold.observation_id) != set(frame.observation_id[frame.fold == k]):
            raise ValueError(f"Fold {k} predictions do not cover its test rows once each")
        if not np.isfinite(fold[["prediction", "predictive_std", "latent_std"]].to_numpy()).all():
            raise ValueError(f"Fold {k} has nonfinite predictions")
        fold.to_parquet(output / f"fold_{k}.parquet", index=False)
        pd.concat(vals, ignore_index=True).to_parquet(output / f"validation_{k}.parquet", index=False)
        pd.DataFrame(summary).to_csv(output / "selection.csv", index=False)
    regime = "F2" if method["monotherapy"] or method["target"] == "bliss_residual" else "F0"
    write_json(output / "run.json", {
        "method": f"Exact GP on all training rows (extension of Kingsley Ukandu's GP), variant {variant}",
        "regime": regime, "target": "inhibition", "settings": {**method, "seed": plan["seed"],
        "selection": "family with lowest validation RMSE on fold (f+1)%5; no refit"},
        "config": [str(c) for c in configs], "config_sha256": [digest(c) for c in configs], "folds": list(folds),
        "seconds": time.perf_counter() - started, **_provenance_gpu(path, dev),
        "prediction_sha256": {f"fold_{k}": digest(output / f"fold_{k}.parquet") for k in folds}})


def check_collapse(config, cell_name, family="rbf", gpu=0, output=None):
    """Fit one cell on the study fold, then compare the collapsed likelihood with a direct one on every row."""
    plan = load_plan(config)
    dev = torch.device(f"cuda:{gpu}") if torch.cuda.is_available() else torch.device("cpu")
    (cell, _), = load_cells(plan["dataset"], [cell_name])[0].values()
    fold = plan["study_fold"]
    rows = cell[~cell.fold.isin([fold, (fold + 1) % 5])]
    X, y = Encoder().fit(rows).transform(rows), rows.inhibition.to_numpy(float)
    gp = ExactGP(family, DEFAULT, dev).fit(X, y)
    with torch.no_grad():
        c, scales, noise = gp._unpack(gp._t(gp.theta))
        Xt, yn = gp._t(X), gp._t((y - gp.y_mean) / gp.y_std)
        K = c * gp.kernel(Xt, Xt, scales)
        K.diagonal().add_(noise + gp.alpha)
        L = torch.linalg.cholesky(K)
        del K
        a = torch.cholesky_solve(yn[:, None], L)[:, 0]
        direct = float(-0.5 * yn @ a - torch.log(torch.diagonal(L)).sum() - 0.5 * len(yn) * LOG2PI)
    result = {"cell_line": cell_name, "family": family, "rows": len(y), "inputs": gp.n_inputs,
              "lml_collapsed": gp.lml, "lml_direct": direct, "relative_difference": abs(direct - gp.lml) / abs(direct),
              "peak_gpu_mb_direct": torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else None}
    print(json.dumps(result))
    if output:
        write_json(output, result)
    return result


def summarize(study_dir):
    """Pooled validation metrics per group, variant and predictor, each cell at its selected family."""
    frames = [pd.read_csv(p) for p in sorted(Path(study_dir).glob("*.csv")) if not p.name.endswith(".partial.csv")]
    cells = pd.concat(frames, ignore_index=True)
    rows = []
    for (group, variant, predictor), g in cells.groupby(["group", "variant", "predictor"], sort=False):
        chosen, n = g[g.selected], g[g.selected].validation_n.to_numpy(float)
        exact = cells[(cells.group == group) & (cells.variant == variant) & (cells.predictor == "exact")]
        rows.append({"group": group, "variant": variant, "predictor": predictor, "cells": len(chosen),
                     "rows_hyper_median": float(chosen.n_rows_hyper.median()), "rows_median": float(chosen.n_rows.median()),
                     "inputs_median": float(chosen.n_inputs.median()), "n_inducing": chosen.n_inducing.median(),
                     "validation_rows": int(n.sum()), "validation_mse": float(chosen.validation_sse.sum() / n.sum()),
                     "validation_rmse": float(np.sqrt(chosen.validation_sse.sum() / n.sum())),
                     "validation_nll": float(np.average(chosen.validation_nll, weights=n)),
                     "validation_coverage90": float(np.average(chosen.validation_coverage90, weights=n)),
                     "matern_cells": int((chosen.family == "matern").sum()),
                     "fit_seconds_median": float(exact.fit_seconds.median()), "fit_seconds_total": float(exact.fit_seconds.sum()),
                     "peak_gpu_mb_max": float(exact.peak_gpu_mb.max()), "not_converged": int((~exact.converged.astype(bool)).sum()),
                     "at_bound": int(exact.at_bound.sum())})
    return pd.DataFrame(rows), cells


def decide(summary, plan):
    exact = summary[(summary.predictor == "exact") & (summary.group != "demo")].set_index("variant").validation_mse
    rule = plan["selection"]
    f0 = rule["F0"]["default"]
    challengers = rule["F0"].get("challengers", [rule["F0"].get("challenger")])
    better = [v for v in challengers if exact[v] < exact[f0] * (1 - rule["F0"]["margin"])]
    if better:
        f0 = min(better, key=lambda v: exact[v])
    f2 = min(rule["F2"]["candidates"], key=lambda v: exact[v])
    return {"rule": rule, "F0": f0, "F2": f2, "validation_mse": {k: float(v) for k, v in exact.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("study")
    s.add_argument("--config", nargs="+", required=True)
    s.add_argument("--group", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--gpu", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--config", nargs="+", required=True)
    r.add_argument("--variant", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    r.add_argument("--gpu", type=int, default=0)
    c = sub.add_parser("check")
    c.add_argument("--config", nargs="+", required=True)
    c.add_argument("--cell", required=True)
    c.add_argument("--family", default="rbf")
    c.add_argument("--output")
    c.add_argument("--gpu", type=int, default=0)
    d = sub.add_parser("select")
    d.add_argument("--config", nargs="+", required=True)
    d.add_argument("--study", required=True)
    d.add_argument("--output", required=True)
    a = p.parse_args()
    if a.command == "study":
        study(a.config, a.group, a.output, a.gpu)
    elif a.command == "run":
        run(a.config, a.variant, a.output, a.folds, a.gpu)
    elif a.command == "check":
        check_collapse(a.config, a.cell, a.family, a.gpu, a.output)
    else:
        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        summary, cells = summarize(a.study)
        summary.to_csv(out / "study_summary.csv", index=False, float_format="%.6g", lineterminator="\n")
        cells.to_csv(out / "study_cells.csv", index=False, float_format="%.6g", lineterminator="\n")
        print(summary.round(3).to_string())
        plan = load_plan(a.config)
        if set(plan["groups"]) - {"demo"} <= set(summary.group):
            decision = {**decide(summary, plan), "config_sha256": [digest(c) for c in a.config]}
            write_json(out / "decision.json", decision)
            print(json.dumps({k: decision[k] for k in ["F0", "F2"]}))


if __name__ == "__main__":
    main()
