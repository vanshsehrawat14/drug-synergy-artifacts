"""Kingsley Ukandu's Gaussian process on the shared protocol, with a sparse extension.

python -m cs622.gp_shared study --config configs/gp_study.json --group cap1500 --output results/neural/runs/gp_study
python -m cs622.gp_shared run --method configs/gp_final.json --output results/neural/runs/gp_final --folds 0 1 2 3 4

His model is GP_Regression/src/project_utils.py, imported unchanged:
- one GP per cell line on drug identities and standardized log doses;
- an additive plus symmetrized product kernel;
- training capped to whole blocks by cap_training_blocks (seed 42 + test fold);
- RBF vs Matern 5/2 chosen per cell by validation RMSE.

Shared-protocol changes:
- published pair folds;
- validation fold (f+1)%5 instead of his random inner split;
- the selected fit predicts the test fold without refitting.

Extension, not part of his method: InducingPredictor keeps his fitted hyperparameters and conditions on
every training row of the cell through an inducing-point predictor. It uses the DTC/VFE predictive
distribution (Titsias 2009), with the capped training blocks as inducing inputs.
"""
import argparse
import functools
import importlib.metadata
import importlib.util
import json
import time
import tracemalloc
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_solve, cholesky, solve_triangular

from .io import digest, write_json
from .runner import git_state, load_experiment

GP_SOURCE = Path(__file__).resolve().parent.parent / "GP_Regression" / "src" / "project_utils.py"
FAMILIES = ["rbf", "matern"]
KEEP = ["observation_id", "pair_id", "drug_a", "drug_b", "cell_line", "fold", "block_id", "dose_a", "dose_b", "unit_a", "unit_b"]
HIS = {"cap": 1500, "families": FAMILIES, "interaction_weight": 1.0, "shared_similarity": 0.25, "predictor": "exact"}
Z90 = 1.6448536269514722


@functools.cache
def gp_module():
    spec = importlib.util.spec_from_file_location("gp_project_utils", GP_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_frame(df):
    """Shared combination rows in the column names his functions use."""
    if not (df.unit_a.astype(str).eq("uM").all() and df.unit_b.astype(str).eq("uM").all()):
        raise ValueError("Doses must be in uM")
    frame = df[KEEP + ["inhibition"]].copy()
    for col in ["observation_id", "pair_id", "drug_a", "drug_b", "cell_line", "unit_a", "unit_b"]:
        frame[col] = frame[col].astype(str)
    frame["block_key"] = frame.cell_line + ":" + frame.block_id.astype(str)
    return frame.reset_index(drop=True)


def canonical_unique(X):
    """Distinct encoded inputs up to arm exchange, under which the kernel is invariant."""
    a, b = X[:, :2], X[:, 2:]
    swap = (a[:, 0] > b[:, 0]) | ((a[:, 0] == b[:, 0]) & (a[:, 1] > b[:, 1]))
    return np.unique(np.where(swap[:, None], np.hstack([b, a]), X), axis=0)


class InducingPredictor:
    """DTC/VFE predictive distribution under a fitted model's hyperparameters, conditioned on all rows.

    With inducing inputs equal to the training inputs it reproduces the exact GP.
    """

    def __init__(self, model, train, inducing, chunk=2000, jitter=1e-6):
        features = gp_module().FEATURES
        pre, reg = model.named_steps["preprocessor"], model.named_steps["regressor"]
        self.pre, self.k, self.chunk = pre, reg.kernel_.k1, chunk
        self.noise = float(reg.kernel_.k2.noise_level + reg.alpha)
        self.y_mean, self.y_scale = float(reg._y_train_mean), float(reg._y_train_std)
        self.Z = canonical_unique(pre.transform(inducing[features]))
        X = pre.transform(train[features])
        y = (train.inhibition.to_numpy(float) - self.y_mean) / self.y_scale
        Kuu = self.k(self.Z)
        for scale in [1e-6, 1e-5, 1e-4, 1e-3]:
            self.jitter = jitter * scale / 1e-6 * float(np.mean(np.diag(Kuu)))
            try:
                self.L = cholesky(Kuu + self.jitter * np.eye(len(Kuu)), lower=True)
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise np.linalg.LinAlgError("Inducing covariance is not positive definite")
        m = len(self.Z)
        AAT, Ay = np.zeros((m, m)), np.zeros(m)
        for s in range(0, len(X), chunk):
            A = solve_triangular(self.L, self.k(self.Z, X[s:s + chunk]), lower=True, check_finite=False)
            AAT += A @ A.T
            Ay += A @ y[s:s + chunk]
        self.LB = cholesky(np.eye(m) + AAT / self.noise, lower=True)
        self.w = solve_triangular(self.L, cho_solve((self.LB, True), Ay), lower=True, trans="T") / self.noise
        self.n_conditioned, self.n_inducing = len(X), m

    def predict(self, frame):
        X = self.pre.transform(frame[gp_module().FEATURES])
        means, variances = [], []
        for s in range(0, len(X), self.chunk):
            Xc = X[s:s + self.chunk]
            Kzx = self.k(self.Z, Xc)
            v = solve_triangular(self.L, Kzx, lower=True, check_finite=False)
            u = solve_triangular(self.LB, v, lower=True, check_finite=False)
            means.append(Kzx.T @ self.w)
            variances.append(self.k.diag(Xc) - (v * v).sum(0) + (u * u).sum(0) + self.noise)
        mean, var = np.concatenate(means), np.maximum(np.concatenate(variances), 0)
        return mean * self.y_scale + self.y_mean, np.sqrt(var) * self.y_scale


def uncertainty_metrics(y, mean, std):
    err = np.asarray(y, float) - mean
    var = np.maximum(std, 1e-12) ** 2
    return {"n": len(err), "sse": float(err @ err), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "nll": float(np.mean(0.5 * np.log(2 * np.pi * var) + err ** 2 / (2 * var))),
            "coverage90": float(np.mean(np.abs(err) <= Z90 * std)), "mean_std": float(np.mean(std))}


def make_model(family, method, seed):
    model = gp_module().make_gp(family, interaction_weight=method["interaction_weight"], seed=seed)
    return model.set_params(regressor__kernel__k1__k2__shared_similarity=method["shared_similarity"])


def fit_cell(cell, fold, cap=1500, seed=42, threads=1, method=None, predict_test=True):
    """One cell line and test fold. cell is that cell's shared rows from every fold.

    method defaults to his procedure; cap is kept as an argument for the original call signature.
    Returns test predictions (None if predict_test is False) and one summary row per family.
    """
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    gp = gp_module()
    method = {**HIS, "cap": cap, **(method or {})}
    name, val_fold = cell.cell_line.iloc[0], (fold + 1) % 5
    if cell.cell_line.nunique() != 1:
        raise ValueError("fit_cell takes one cell line")
    available = cell[~cell.fold.isin([fold, val_fold])]
    train = gp.cap_training_blocks(available, method["cap"], seed + fold)
    validation, test = cell[cell.fold == val_fold], cell[cell.fold == fold]
    gp.assert_split(available, validation)
    gp.assert_split(available, test)
    fits = {}
    with threadpool_limits(limits=threads):
        for family in method["families"]:
            tracemalloc.start()
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = make_model(family, method, seed).fit(train[gp.FEATURES], train.inhibition)
            fitted = time.perf_counter()
            if method["predictor"] == "exact":
                predictor = None
                mean, std = gp.predict_batches(model, validation[gp.FEATURES])
            else:
                predictor = InducingPredictor(model, available, train)
                mean, std = predictor.predict(validation)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            kernel = model.named_steps["regressor"].kernel_
            fits[family] = {"model": model, "predictor": predictor, "fit_seconds": fitted - started,
                            "seconds": time.perf_counter() - started, "peak_mb": peak / 2**20, "kernel": str(kernel),
                            "at_bound": int(np.isclose(kernel.theta[:, None], kernel.bounds).any(1).sum()),
                            "warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
                            **{f"validation_{k}": v for k, v in uncertainty_metrics(validation.inhibition, mean, std).items()}}
        # His rule: lowest validation RMSE, ties to the first family listed.
        family = min(method["families"], key=lambda f: fits[f]["validation_rmse"])
        predictions = None
        if predict_test:
            chosen = fits[family]
            mean, std = (gp.predict_batches(chosen["model"], test[gp.FEATURES]) if chosen["predictor"] is None
                         else chosen["predictor"].predict(test))
            predictions = test[KEEP].assign(prediction=mean, predictive_std=std)
    rows = []
    for f, fit in fits.items():
        rows.append({"cell_line": name, "fold": fold, "validation_fold": val_fold, "family": f, "selected": f == family,
                     "n_train": len(train), "n_train_available": len(available), "n_validation": len(validation),
                     "n_test": len(test), "n_inducing": fit["predictor"].n_inducing if fit["predictor"] else None,
                     "jitter": fit["predictor"].jitter if fit["predictor"] else None,
                     **{k: v for k, v in fit.items() if k not in {"model", "predictor"}}})
    return predictions, rows


def _task(args):
    return fit_cell(*args)


def _cells(dataset):
    df, _, path = load_experiment({"dataset": str(dataset), "table": "observations", "target": "inhibition",
                                   "test_fold": 0, "model": {"kind": "dose"}})
    frame = shared_frame(df)
    return {c: g for c, g in frame.groupby("cell_line", sort=True)}, frame, path


def _provenance(path):
    return {"dataset_sha256": digest(path), "gp_source_sha256": digest(GP_SOURCE), "git": git_state(),
            "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "scikit-learn"]}}


def study(config, group, output, workers=4, threads=2):
    """Validation-only runs of one declared group; the test fold is never predicted."""
    plan = json.loads(Path(config).read_text())
    spec = plan["groups"][group]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{group}.csv"
    if target.exists():
        raise FileExistsError(f"{target} exists; choose a new output")
    cells, _, path = _cells(plan["dataset"])
    methods = [{**HIS, **spec["fit"], "predictor": p} for p in spec["predictors"]]
    names = spec.get("cells", list(cells))
    tasks = [(cells[c], plan["study_fold"], m["cap"], plan["seed"], threads, m, False) for m in methods for c in names]
    started, rows = time.perf_counter(), []
    with ProcessPoolExecutor(workers) as pool:
        for (cell, *_rest, method, _), (_, result) in zip(tasks, pool.map(_task, tasks)):
            rows += [{"group": group, "predictor": method["predictor"], **r} for r in result]
            pd.DataFrame(rows).to_csv(output / f"{group}.partial.csv", index=False)
    pd.DataFrame(rows).to_csv(target, index=False)
    (output / f"{group}.partial.csv").unlink()
    write_json(output / f"{group}.json", {"config": str(config), "config_sha256": digest(config), "group": group,
                                          "spec": spec, "workers": workers, "threads": threads,
                                          "seconds": time.perf_counter() - started, **_provenance(path)})


def run(output, folds, dataset="data/processed/oneil_v1", cap=1500, seed=42, workers=6, threads=2, method=None):
    method = {**HIS, "cap": cap, **(method or {})}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"fold_{k}.parquet").exists() for k in folds):
        raise FileExistsError("Fold predictions exist; choose a new output")
    started = time.perf_counter()
    cells, frame, path = _cells(dataset)
    tasks = [(cells[c], k, method["cap"], seed, threads, method, True) for k in folds for c in cells]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_task, tasks))
    summary = pd.DataFrame([r for _, rows in results for r in rows])
    for k in folds:
        fold = pd.concat([p for p, rows in results if rows[0]["fold"] == k], ignore_index=True)
        expected = set(frame.observation_id[frame.fold == k])
        if fold.observation_id.duplicated().any() or set(fold.observation_id) != expected:
            raise ValueError(f"Fold {k} predictions do not cover its test rows once each")
        if not np.isfinite(fold[["prediction", "predictive_std"]].to_numpy()).all():
            raise ValueError(f"Fold {k} has nonfinite predictions")
        fold.to_parquet(output / f"fold_{k}.parquet", index=False)
    previous = output / "selection.csv"
    if previous.exists():
        summary = pd.concat([pd.read_csv(previous), summary]).drop_duplicates(["cell_line", "fold", "family"], keep="last")
    summary.sort_values(["fold", "cell_line", "family"]).to_csv(previous, index=False)
    write_json(output / "run.json", {
        "method": "Gaussian process (Kingsley Ukandu), GP_Regression/src/project_utils.py",
        "regime": "F0: drug identities and doses, one model per cell line", "target": "inhibition",
        "settings": {**method, "seed": seed, "cap_seed": "seed + test fold",
                     "selection": "family with lowest validation RMSE on fold (f+1)%5; no refit",
                     "workers": workers, "threads": threads},
        "folds": sorted(set(summary.fold)), **_provenance(path), "seconds_this_call": time.perf_counter() - started,
        "prediction_sha256": {f"fold_{k}": digest(output / f"fold_{k}.parquet") for k in range(5) if (output / f"fold_{k}.parquet").exists()},
    })
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", required=True)
    r.add_argument("--dataset", default="data/processed/oneil_v1")
    r.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    r.add_argument("--cap", type=int, default=1500)
    r.add_argument("--method", help="JSON file with a 'method' object; defaults to his procedure")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--threads", type=int, default=2)
    s = sub.add_parser("study")
    s.add_argument("--config", required=True)
    s.add_argument("--group", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    if a.command == "study":
        study(a.config, a.group, a.output, a.workers, a.threads)
        return
    method = json.loads(Path(a.method).read_text())["method"] if a.method else None
    summary = run(a.output, a.folds, a.dataset, a.cap, workers=a.workers, threads=a.threads, method=method)
    chosen = summary[summary.selected]
    print(chosen.groupby("fold").agg(cells=("cell_line", "size"), matern=("family", lambda f: int((f == "matern").sum())),
                                     at_bound=("at_bound", "sum"), seconds=("seconds", "sum")).to_string())


if __name__ == "__main__":
    main()
