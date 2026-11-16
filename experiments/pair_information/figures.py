"""Figures for the final campaign from tracked aggregates only.

python experiments/pair_information/figures.py --results results/pair_information
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def save(fig, out, name):
    for ext in ["png", "pdf"]:
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def information_ladder(root, out):
    noise = pd.read_csv(root / "noise" / "noise_ceilings.csv")
    oracle = pd.read_csv(root / "oracle" / "oracle_r2.csv")
    share = noise[(noise.level == "panel") & (noise.target == "bliss_residual") & (noise.model == "noise_share")].iloc[0]
    rows = []
    o = oracle[oracle.target == "bliss_residual"].set_index("predictor")
    rows += [("Cross-cell oracle, all other cells", o.loc["oracle_loco"], ORANGE),
             ("Cross-cell oracle, B2 training cells", o.loc["oracle_matched"], ORANGE)]
    seen = root / "pair_seen" / "pair_seen_comparison.csv"
    if seen.exists():
        s = pd.read_csv(seen)
        s = s[(s.level == "panel") & (s.target == "bliss_residual") & (s.quantity == "r2")].set_index("model")
        rows.append(("Pair-seen F2 (pair-cell folds)", s.loc["pair_seen"].rename({"value": "value"}), AQUA))
    n = noise[(noise.level == "panel") & (noise.target == "bliss_residual")].set_index("model")
    rows += [("Selected F2, held-out pairs", n.loc["neural_F2"], BLUE), ("GP F2, held-out pairs", n.loc["gp_F2"], BLUE),
             ("CF extension, held-out pairs", n.loc["cf_selected"], BLUE), ("Production F2, held-out pairs", n.loc["neural_base_F2"], BLUE)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(rows))[::-1]
    for yi, (label, r, color) in zip(y, rows):
        ax.errorbar(r["value"], yi, xerr=[[r["value"] - r["pair_low"]], [r["pair_high"] - r["value"]]], fmt="o", color=color,
                    ms=6, lw=2, capsize=0)
        ax.annotate(f"{r['value']:.2f}", (r["pair_high"], yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontsize=8)
    ceiling = 1 - share["value"]
    ax.axvline(ceiling, color=GRAY, lw=2, ls="--")
    ax.annotate(f"replicate ceiling {ceiling:.2f}", (ceiling, y.max() + 0.6), ha="right", va="bottom", color=MUTED, fontsize=8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-0.15, 1.0); ax.set_ylim(-0.7, y.max() + 1.2)
    ax.set_xlabel("R² on the Bliss-residual pair-cell contrast (95% whole-pair bootstrap)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig1_information_ladder")


def shortlist(root, out):
    t = pd.read_csv(root / "shortlist_noise" / "turnover.csv")
    t = t[t.budget == 0.05].set_index("quantity")
    sel = pd.read_csv(root / "shortlist_noise" / "changed_unit_selection.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.15, 1]})
    items = [("Independent half-depth replicates", t.loc["noise_N"], BLUE), ("Independent full-depth bootstrap", t.loc["noise_N_independent_resamples"], BLUE),
             ("C1 correction, same resample", t.loc["N_to_C1_within_resample"], ORANGE)]
    y = np.arange(len(items))[::-1]
    for yi, (label, r, color) in zip(y, items):
        a.barh(yi, 100 * r["mean"], color=color, height=0.55)
        a.errorbar(100 * r["mean"], yi, xerr=[[100 * (r["mean"] - r["q025"])], [100 * (r["q975"] - r["mean"])]], fmt="none", ecolor=INK, lw=1)
        a.annotate(f"{100 * r['mean']:.1f}%", (100 * r["q975"], yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8)
    a.set_yticks(y, [i[0] for i in items]); a.set_xlabel("Top-5% pair-cell turnover (%)"); a.set_xlim(0, 36); a.grid(axis="y", visible=False)
    for k, (mask, label) in enumerate([(sel.c1_entrant, "C1 entrants"), (sel.c1_exit, "C1 exits")]):
        g = sel[mask]
        for v, x, color in [("N", 0, BLUE), ("C1", 1, ORANGE)]:
            jitter = (np.random.default_rng(k).random(len(g)) - 0.5) * 0.18
            b.scatter(x + 2.4 * k + jitter, g[f"p_select_{v}"], s=8, color=color, alpha=0.6, lw=0)
            b.plot([x + 2.4 * k - 0.22, x + 2.4 * k + 0.22], [g[f"p_select_{v}"].mean()] * 2, color=INK, lw=2)
        b.annotate(label, (0.5 + 2.4 * k, 1.04), ha="center", fontsize=8)
    b.set_xticks([0, 1, 2.4, 3.4], ["native", "C1", "native", "C1"]); b.set_ylim(-0.03, 1.12)
    b.set_ylabel("Bootstrap selection probability"); b.grid(axis="x", visible=False)
    save(fig, out, "fig2_shortlist_noise")


def boundary(root, out):
    f = pd.read_csv(root / "loewe_boundary" / "near_boundary_roots.csv")
    s = json.loads((root / "loewe_boundary" / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    for mask, label, color in [(~f.c1, "accepted or other roots", BLUE), (f.c1, "C1-recovered roots", ORANGE)]:
        ax.scatter(f.predicted[mask], f.actual[mask], s=3, color=color, alpha=0.4, lw=0, label=label, rasterized=True)
    lim = [max(1e-16, f[["predicted", "actual"]].min().min()), 1e-1]
    ax.plot(lim, lim, color=GRAY, lw=1.2, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Leading-order law  |d−c| q/(1+q)"); ax.set_ylabel("Certified distance to smaller plateau")
    ax.legend(frameon=False, loc="upper left", markerscale=3, fontsize=8)
    ax.annotate(f"log-log r = {s['boundary_layer']['log_log_correlation_near']:.3f}\nn = {s['boundary_layer']['near_cases']:,}",
                (0.97, 0.05), xycoords="axes fraction", ha="right", fontsize=8, color=MUTED)
    save(fig, out, "fig3_boundary_layer")


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--results", required=True); a = p.parse_args()
    root = Path(a.results); out = root / "figures"; out.mkdir(parents=True, exist_ok=True)
    information_ladder(root, out); shortlist(root, out); boundary(root, out)
    print(sorted(x.name for x in out.iterdir()))


if __name__ == "__main__":
    main()
