"""Standalone prediction plots and a real-data drug/dose exchange audit."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureBuilder, LazyFeatures
from .io import read_json, write_json
from .training import NeuralRegressor


def symmetry_audit(run, output, limit=10000):
    run, output = Path(run), Path(output)
    model, context = NeuralRegressor.load(run / "checkpoints" / "best.pt")
    builder = FeatureBuilder.from_state_dict(context["features"])
    frame = pd.read_parquet(run / "predictions.parquet").iloc[:limit]
    rename = {}
    for prefix in ["drug", "dose", "unit", "cid"]:
        if prefix + "_a" in frame:
            rename.update({prefix + "_a": prefix + "_b", prefix + "_b": prefix + "_a"})
    swapped = frame.rename(columns=rename)
    def features(data):
        return LazyFeatures(builder, data) if builder.external is not None else builder.transform(data)
    difference = np.abs(model.predict(features(frame)) - model.predict(features(swapped)))
    report = {"rows": len(frame), "max_absolute_prediction_difference": float(difference.max()),
              "exact_exchange_invariance": bool((difference == 0).all()),
              "swapped_together": sorted(rename), "checkpoint": str(run / "checkpoints" / "best.pt")}
    write_json(output, report)
    if not report["exact_exchange_invariance"]:
        raise ValueError("Real-data exchange invariance audit failed")
    print(report)
    return report


def plot_run(run, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run, output = Path(run), Path(output)
    if output.exists():
        raise FileExistsError(output)
    config = read_json(run / "config.json")
    target = config["target"]
    df = pd.read_parquet(run / "evaluation" / "predictions_with_truth.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    counts = axes[0].hexbin(df[target], df.prediction, gridsize=65, mincnt=1, bins="log", cmap="viridis")
    bounds = [float(min(df[target].min(), df.prediction.min())), float(max(df[target].max(), df.prediction.max()))]
    axes[0].plot(bounds, bounds, color="black", linewidth=1, linestyle="--")
    axes[0].set(xlabel=f"Observed {target}", ylabel=f"Predicted {target}", title="Held-out observations")
    fig.colorbar(counts, ax=axes[0], label="Count (log color scale)")
    rmse = df.assign(squared_error=(df[target]-df.prediction)**2).groupby("cell_line", observed=True).squared_error.mean().pow(.5).sort_values()
    rmse.tail(15).plot.barh(ax=axes[1], color="#306b85")
    axes[1].set(xlabel="RMSE", ylabel="Cell line", title="Cell-line error (up to 15 largest)")
    fig.suptitle(f"{config['model']['kind']} · {config['features']['regime']} · test fold {config['test_fold']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved {output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--symmetry", action="store_true")
    args = p.parse_args()
    if args.symmetry:
        symmetry_audit(args.run, args.output)
    else:
        plot_run(args.run, args.output)


if __name__ == "__main__":
    main()
