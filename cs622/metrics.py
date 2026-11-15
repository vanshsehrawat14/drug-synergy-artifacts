"""Shared regression metrics and retrospective panel interaction contrasts."""
import numpy as np
import pandas as pd


def regression_metrics(truth, prediction):
    y, p = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != p.shape or len(y) == 0:
        raise ValueError("Expected matching nonempty one-dimensional predictions and targets")
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError("Nonfinite targets or predictions; exclusion must be explicit upstream")
    error = y - p
    sse = float(error @ error)
    centered = y - y.mean()
    ss = float(centered @ centered) if np.ptp(y) > 0 else 0.
    pearson = None
    if len(y) > 1 and ss > 0 and np.ptp(p) > 0:
        pearson = float(np.corrcoef(y, p)[0, 1])
    return {"n": len(y), "mse": sse / len(y), "rmse": float(np.sqrt(sse / len(y))),
            "mae": float(np.abs(error).mean()), "pearson": pearson,
            "r2": 1 - sse / ss if ss > 0 else None, "sse": sse, "target_ss": ss}


def align_predictions(observations, predictions):
    for frame in [observations, predictions]:
        if frame.observation_id.isna().any() or frame.observation_id.duplicated().any():
            raise ValueError("Duplicate or missing observation IDs")
    if set(observations.observation_id) != set(predictions.observation_id):
        raise ValueError("Prediction coverage differs from evaluation observations")
    return observations.merge(predictions[["observation_id", "prediction"]], on="observation_id", validate="one_to_one")


def group_metrics(frame, target, group):
    return [{group: str(key), **regression_metrics(g[target], g.prediction)}
            for key, g in frame.groupby(group, observed=True, sort=True)]


def pair_bootstrap(frame, target, repeats=500, seed=2026):
    """Percentile intervals from pair-cluster resampling, not dose-row resampling.

    Conditional on this drug/cell collection; shared-drug dependence remains.
    """
    if repeats < 1:
        return None
    error = frame[target].to_numpy(float) - frame.prediction.to_numpy(float)
    stats = pd.DataFrame({"pair": frame.pair_id.to_numpy(), "n": 1,
                          "sse": error ** 2, "sae": abs(error)}).groupby("pair", observed=True).sum()
    if len(stats) < 2:
        return None
    values = stats[["n", "sse", "sae"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.array([values[rng.integers(len(values), size=len(values))].sum(axis=0) for _ in range(repeats)])
    metrics = {"mse": samples[:, 1] / samples[:, 0], "mae": samples[:, 2] / samples[:, 0]}
    metrics["rmse"] = np.sqrt(metrics["mse"])
    return {"method": "pair-cluster percentile bootstrap", "clusters": len(stats), "repeats": repeats,
            "seed": seed, "confidence": .95,
            "intervals": {k: np.quantile(v, [.025, .975]).tolist() for k, v in metrics.items()}}


def interaction_contrast(frame, target):
    """Exact common.metrics projector, independently within each test fold/cell.

    This is a retrospective diagnostic. Both truth and predictions are projected
    on the evaluated pair incidence design; no coefficients may become features.
    Valid only for one observation per unordered pair/cell/fold (not dose rows).
    """
    keys = ["fold", "pair_id", "cell_line"]
    if frame.duplicated(keys).any():
        raise ValueError("Contrast requires a pair-cell panel; aggregate doses/blocks first")
    sse, ss = 0., 0.
    for _, g in frame.groupby(["fold", "cell_line"], observed=True, sort=True):
        drugs = sorted(set(g.drug_a) | set(g.drug_b))
        lookup = {d: i for i, d in enumerate(drugs)}
        h = np.zeros((len(g), len(drugs)))
        ix = np.arange(len(g))
        np.add.at(h, (ix, g.drug_a.map(lookup).to_numpy(dtype=np.int64)), 1)
        np.add.at(h, (ix, g.drug_b.map(lookup).to_numpy(dtype=np.int64)), 1)
        z = g[[target, "prediction"]].to_numpy(float)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite contrast inputs")
        residual = z - h @ np.linalg.lstsq(h, z, rcond=None)[0]
        sse += float(np.sum((residual[:, 0] - residual[:, 1]) ** 2))
        ss += float(np.sum(residual[:, 0] ** 2))
    # Exact additive targets can leave machine-roundoff residuals.
    tolerance = np.finfo(float).eps * max(1., float(np.sum(frame[target].to_numpy(float) ** 2)))
    return {"contrast_sse": sse, "contrast_ss": ss,
            "contrast_r2": 1 - sse / ss if ss > tolerance else None,
            "use": "retrospective evaluation only"}


def evaluate(frame, target, bootstrap=500, seed=2026, contrast=False):
    if frame.observation_id.duplicated().any():
        raise ValueError("Duplicate observations")
    result = {"pooled": regression_metrics(frame[target], frame.prediction),
              "by_cell_line": group_metrics(frame, target, "cell_line"),
              "by_pair": group_metrics(frame, target, "pair_id"),
              "uncertainty": pair_bootstrap(frame, target, bootstrap, seed)}
    if contrast:
        result["interaction_contrast"] = interaction_contrast(frame, target)
    result["macro_mse"] = {"pair": float(np.mean([m["mse"] for m in result["by_pair"]])),
                           "cell_line": float(np.mean([m["mse"] for m in result["by_cell_line"]]))}
    return result


def drug_jackknife(frame, target):
    """Delete-one-drug uncertainty, conditional on cells and dose design.

    Removes every pair containing the deleted drug. Intervals are approximate;
    this small collection of drugs does not support population-wide claims.
    """
    from scipy.stats import t
    drugs = sorted(set(frame.drug_a) | set(frame.drug_b))
    if len(drugs) < 4:
        return None
    estimates = []
    for drug in drugs:
        selected = frame[(frame.drug_a != drug) & (frame.drug_b != drug)]
        if selected.empty:
            return None
        estimates.append(regression_metrics(selected[target], selected.prediction))
    point = regression_metrics(frame[target], frame.prediction)
    intervals = {}
    for metric in ["mse", "rmse", "mae"]:
        values = np.array([m[metric] for m in estimates])
        se = np.sqrt((len(drugs)-1) / len(drugs) * np.sum((values-values.mean())**2))
        half = float(t.ppf(.975, len(drugs)-1) * se)
        intervals[metric] = [point[metric]-half, point[metric]+half]
    return {"method": "delete-one-drug jackknife, approximate t interval", "drugs": len(drugs),
            "confidence": .95, "intervals": intervals}
