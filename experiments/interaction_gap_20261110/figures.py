"""Figures for the interaction-gap analysis.

python experiments/interaction_gap_20261110/figures.py --results results/interaction_gap_20261110 \
    --cross-batch DERIVED/cross_batch_values.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REST, NOISE = "#c9c8c2", "#8d8c86"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def anchor_curves(res, var, out):
    cur = res["curves"]
    ks = [1, 2, 4, 8, 16]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    series = [("B2", ORANGE, "B2 cross-cell covariance"), ("B1", BLUE, "B1 pair intercept"),
              ("B5b_B2", AQUA, "B2, anchor cells deranged"), ("B4_B2", YELLOW, "B2, donor pair sharing a drug")]
    for m, col, lab in series:
        y = [cur[f"bliss:k={k}"].get(m, {}).get("r2", np.nan) for k in ks]
        lo = [cur[f"bliss:k={k}"].get(m, {}).get("pair_interval", [np.nan] * 2)[0] for k in ks]
        hi = [cur[f"bliss:k={k}"].get(m, {}).get("pair_interval", [np.nan] * 2)[1] for k in ks]
        ax.plot(ks, y, color=col, lw=2, marker="o", ms=4, label=lab)
        if m in ("B1", "B2"):
            ax.fill_between(ks, lo, hi, color=col, alpha=0.15, lw=0)
        ax.annotate(lab.split(" ")[0] if m in ("B1", "B2") else "", (ks[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    lo = var["loco"]["bliss"]
    for m, col in [("B1", BLUE), ("B2", ORANGE)]:
        ax.errorbar([38], [lo[m]["r2"]], yerr=[[lo[m]["r2"] - lo[m]["pair_interval"][0]], [lo[m]["pair_interval"][1] - lo[m]["r2"]]],
                    color=col, marker="s", ms=5, capsize=2, lw=1.2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 38])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "all 38"])
    ax.set_xlabel("anchor cell lines of the held-out pair (k)")
    ax.set_ylabel("R² vs. no same-pair information")
    ax.set_title("a  Same-pair information, held-out pairs", loc="left", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_ylim(-0.32, 0.5)
    # b: matched pair-cell split, leak-free.
    ax = axes[1]
    m = res["matched_pair_seen"]["bliss:leak_free"]
    vals = [("B1 intercept", m["B1:projected"]["r2"], m["B1:projected"]["pair_interval"], BLUE),
            ("B2 covariance", m["B2:projected"]["r2"], m["B2:projected"]["pair_interval"], ORANGE),
            ("B3 neural", m["B3"]["r2"], m["B3"]["pair_interval"], VIOLET)]
    for i, (lab, v, ci, col) in enumerate(vals):
        ax.bar(i, v, color=col, width=0.6)
        if ci:
            ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], color=INK, capsize=3, lw=1)
        ax.text(i, 0.006, f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels([v[0] for v in vals], fontsize=8)
    ax.set_ylabel("R² (projected contrast)")
    ax.set_title("b  Pair-cell split, about 23 anchors", loc="left", fontsize=9, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_ylim(0, 0.32)
    fig.tight_layout()
    save(fig, out, "fig_anchor_information")


def decomposition(var, out):
    names = [("bliss", "Bliss residual"), ("raw", "measured response"), ("train", "train-only baseline")]
    fig, ax = plt.subplots(figsize=(7.4, 2.4))
    for i, (k, lab) in enumerate(names):
        c, lo = var["components"][k], var["loco"][k]
        b1, gain = lo["B1"]["r2"], lo["B2_minus_B1"]["r2"]
        noise = c["noise"]
        rest = 1 - noise - b1 - gain
        left = 0
        for v, col in [(b1, BLUE), (gain, ORANGE), (rest, REST), (noise, NOISE)]:
            ax.barh(i, v, left=left, color=col, height=0.62, edgecolor=SURFACE, lw=1.5)
            if v > 0.06:
                ax.text(left + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if col != REST else INK)
            left += v
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n[1] for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of pair-cell contrast energy (leave-one-cell-out, all other cells as anchors)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, REST, NOISE)]
    ax.legend(handles, ["pair intercept (B1)", "cell-specific transfer (B2 - B1)", "reproducible, not transferred", "replicate noise"],
              frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.tight_layout()
    save(fig, out, "fig_decomposition")


def cross_batch(path, var, out):
    df = pd.read_csv(path)
    x, y = df["bliss_frozen_reference:cell_shift:t1"], df["bliss_frozen_reference:cell_shift:t2"]
    pm = df.assign(x=x, y=y).groupby("pair")[["x", "y"]].mean()
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    lim = [min(x.min(), y.min()) - 1, max(x.max(), y.max()) + 1]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax.scatter(x, y, s=10, color=BLUE, alpha=0.55, lw=0, label="pair-cell")
    ax.scatter(pm.x, pm.y, s=34, color=ORANGE, edgecolor=SURFACE, lw=1.2, label="pair mean over 15 cells")
    cb = var["cross_batch"]["bliss_frozen_reference"]["cell_shift"]
    ax.text(0.03, 0.97, f"r = {cb['corr_pc']:.2f} (pair-cells)\nr = {cb['corr_pair_means']:.2f} (pair means)\n"
                        f"cross/within energy = {cb['replication_overall']:.2f}", transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("contrast, batch 1")
    ax.set_ylabel("contrast, batch 2 (cell shift removed)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(color=GRID, lw=0.6)
    fig.tight_layout()
    save(fig, out, "fig_cross_batch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cross-batch", required=True)
    a = ap.parse_args()
    r = Path(a.results)
    out = r / "figures"
    out.mkdir(exist_ok=True)
    res = json.loads((r / "anchor_experiment.json").read_text())
    var = json.loads((r / "variance_components.json").read_text())
    anchor_curves(res, var, out)
    decomposition(var, out)
    cross_batch(a.cross_batch, var, out)


if __name__ == "__main__":
    main()
