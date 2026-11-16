"""Cross-screen summary table and report figures from saved results. Nothing is refit.

python -m synergy_artifacts.synthesis --output results/summary [--loewe-cases CASES --loewe-verified VERIFIED]

The table separates four questions:
- response accuracy: F2 and F0 error;
- prediction beyond the monotherapy reference: MSE change against Bliss on the same training-pair monotherapy;
- interaction-specific prediction: interaction residual R² against Bliss, pooled and at Bliss-expected inhibition >= 75;
- Loewe scoring artifacts, which concern score labels, not these response models (reports/loewe_note.md).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

SCREENS = [
    # name, benchmark, comparisons, regimes, F2 and F0 model names, status of the high-inhibition test
    ("ONEIL", "results/neural_benchmark/benchmark.csv", "results/neural_benchmark/comparisons.csv", "results/beyond_bliss_regimes/regimes.csv",
     "selected:F2", "selected:F0", "exploratory (found here)"),
    ("ALMANAC", "results/almanac_validation/benchmark.csv", "results/almanac_validation/comparisons.csv",
     "results/almanac_validation/regimes.csv", "f2_combined", "f0_combined", "rule met; previously explored screen"),
    ("NCATS_ES", "results/third_screen/ncats_es/benchmark.csv", "results/third_screen/ncats_es/comparisons.csv",
     "results/third_screen/ncats_es/regimes.csv", "f2_combined", "f0_combined", "failed prospectively (H1)"),
    ("NCATS_DIPG", "results/third_screen/ncats_dipg/benchmark.csv", "results/third_screen/ncats_dipg/comparisons.csv",
     "results/third_screen/ncats_dipg/regimes.csv", "f2_combined", "f0_combined", "failed prospectively (H1)"),
]


def _interval(text):
    return [float(x) for x in str(text).split(";")]


def cross_screen():
    rows = []
    for name, bench, comp, reg, f2, f0, status in SCREENS:
        b, c, r = pd.read_csv(bench).set_index("model"), pd.read_csv(comp), pd.read_csv(reg)
        if "against" in r:
            r = r[r.against == "bliss"]
        r = r[r.model == f2]

        def change(model, control="bliss_reference"):
            hit = c[(c.model == model) & (c.control == control)]
            if hit.empty:
                return [np.nan] * 3 + [""]
            h = hit.iloc[0]
            return [h.relative_mse_change, h.ci_low / h.control_mse, h.ci_high / h.control_mse, h.verdict]

        def irr(family, regime):
            hit = r[(r.family == family) & (r.regime == regime)].iloc[0]
            low, high = _interval(hit.interaction_residual_r2_jackknife)
            return hit.interaction_residual_r2, low, high

        f2c, f0c, monoc = change(f2), change(f0), change("mono_only")
        pooled, high = irr("all", "all"), irr("bliss_level", "R>=75")
        rows.append({"screen": name, "dose_rows": int(b.loc[f2, "n"]), "f2_rmse": b.loc[f2, "rmse"], "f2_r2": b.loc[f2, "r2"],
                     "f0_rmse": b.loc[f0, "rmse"], "bliss_rmse": b.loc["bliss_reference", "rmse"],
                     "f2_vs_bliss_mse_change": f2c[0], "f2_vs_bliss_low": f2c[1], "f2_vs_bliss_high": f2c[2], "f2_vs_bliss": f2c[3],
                     "f0_vs_bliss_mse_change": f0c[0], "f0_vs_bliss": f0c[3],
                     "mono_only_vs_bliss_mse_change": monoc[0], "mono_only_vs_bliss_low": monoc[1], "mono_only_vs_bliss_high": monoc[2],
                     "interaction_r2_pooled": pooled[0], "interaction_r2_pooled_low": pooled[1], "interaction_r2_pooled_high": pooled[2],
                     "interaction_r2_r75": high[0], "interaction_r2_r75_low": high[1], "interaction_r2_r75_high": high[2],
                     "high_inhibition_test": status})
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(width, height, panels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "axes.labelcolor": INK, "text.color": INK})
    fig, axes = plt.subplots(1, panels, figsize=(width, height), facecolor=SURFACE, constrained_layout=True)
    for ax in np.atleast_1d(axes):
        _style(ax)
    return fig, axes


def _dots(ax, y, values, low, high, color, label, marker="o", filled=True):
    ax.errorbar(values, y, xerr=[np.subtract(values, low), np.subtract(high, values)], fmt="none", ecolor=color, elinewidth=2, capsize=0)
    ax.scatter(values, y, s=48, marker=marker, color=color if filled else SURFACE, edgecolors=color, linewidths=2, zorder=3, label=label)


def figure_cross_screen(table, path):
    fig, (a, b) = _figure(7.4, 3.3, 2)
    labels = [f"{r.screen}\n{r.high_inhibition_test}" for r in table.itertuples()]
    y = np.arange(len(table))[::-1].astype(float)
    _dots(a, y + 0.12, 100 * table.f2_vs_bliss_mse_change, 100 * table.f2_vs_bliss_low, 100 * table.f2_vs_bliss_high, BLUE, "F2 network")
    mono = table.mono_only_vs_bliss_mse_change.notna().to_numpy()
    _dots(a, y[mono] - 0.12, 100 * table.mono_only_vs_bliss_mse_change[mono], 100 * table.mono_only_vs_bliss_low[mono],
          100 * table.mono_only_vs_bliss_high[mono], ORANGE, "monotherapy-only network")
    a.axvline(0, color=MUTED, linewidth=1)
    a.set_yticks(y, labels)
    a.set_xlabel("Response MSE change vs Bliss (%)")
    a.set_title("a  Response error vs Bliss")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    _dots(b, y + 0.12, table.interaction_r2_pooled, table.interaction_r2_pooled_low, table.interaction_r2_pooled_high, BLUE, "F2, all rows")
    _dots(b, y - 0.12, table.interaction_r2_r75, table.interaction_r2_r75_low, table.interaction_r2_r75_high, BLUE,
          "F2, Bliss-expected inhibition ≥ 75", marker="D", filled=False)
    b.axvline(0, color=MUTED, linewidth=1)
    b.set_yticks(y, [""] * len(y))
    b.set_xlabel("Interaction residual R² vs Bliss (F2)")
    b.set_title("b  Interaction-specific signal")
    b.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def loewe_blocks(cases_path, verified_path):
    """Published and corrected native block scores with every admissible root restored."""
    from .loewe_verify import RUNS, score_effects
    full, ver = pd.read_parquet(cases_path), pd.read_parquet(verified_path)
    if not (full.case_id.to_numpy() == ver.case_id.to_numpy()).all():
        raise ValueError("Case tables are not aligned")
    cases = full.assign(independent=ver.independent.to_numpy())
    blocks = pd.read_parquet(RUNS / "scientific_native_full/blocks.parquet", columns=["surface_id", "native_sum_score"])
    effect, shift = score_effects(cases, blocks, ver["class"].str.startswith("artifact").to_numpy(), "all admissible")
    return blocks.native_sum_score.to_numpy(), blocks.native_sum_score.to_numpy() + shift, effect


def figure_loewe(published, corrected, path, verify_json="results/loewe_verify/loewe_verify.json"):
    effects = json.loads(Path(verify_json).read_text())["score_effects"]
    fig, (a, b) = _figure(7.4, 3.0, 2)
    moved = np.abs(corrected - published) > 1e-9
    flips = moved & (np.sign(corrected) != np.sign(published))
    keep = moved & ~flips
    a.scatter(published[keep], corrected[keep], s=4, color=BLUE, alpha=0.35, linewidths=0, label=f"same sign ({keep.sum():,})")
    a.scatter(published[flips], corrected[flips], s=6, color=ORANGE, alpha=0.7, linewidths=0, label=f"sign reversed ({flips.sum():,})")
    lim = [min(published[moved].min(), corrected[moved].min()), max(published[moved].max(), corrected[moved].max())]
    a.plot(lim, lim, color=MUTED, linewidth=1)
    a.axhline(0, color=GRID, linewidth=1, zorder=0)
    a.axvline(0, color=GRID, linewidth=1, zorder=0)
    a.grid(axis="y", color=GRID, linewidth=0.8)
    a.set_xlabel("Native block Loewe score (reproduces published)")
    a.set_ylabel("With admissible roots restored")
    a.set_title("a  Blocks whose score changes")
    a.legend(frameon=False, loc="upper left", fontsize=8, markerscale=3)
    rows = [("Admissible roots restored", effects[2]["blocks_affected"], effects[2]["block_sign_flips"]),
            ("Solver already at the root", effects[0]["blocks_affected"], effects[0]["block_sign_flips"]),
            ("HSA vs summed-dose fallback\n(ordinary convention choice)", effects[3]["blocks_affected"], effects[3]["block_sign_flips"])]
    y = np.arange(len(rows))[::-1]
    total = 92208
    b.barh(y + 0.18, [100 * r[1] / total for r in rows], height=0.34, color=BLUE, label="score changed")
    b.barh(y - 0.18, [100 * r[2] / total for r in rows], height=0.34, color=ORANGE, label="sign reversed")
    for yi, r in zip(y, rows):
        b.text(100 * r[1] / total + 1, yi + 0.18, f"{r[1]:,}", va="center", color=INK, fontsize=8)
        b.text(100 * r[2] / total + 1, yi - 0.18, f"{r[2]:,}", va="center", color=INK, fontsize=8)
    b.set_yticks(y, [r[0] for r in rows])
    b.set_xlim(0, 100)
    b.set_xlabel("Share of 92,208 ONEIL blocks (%)")
    b.set_title("b  Blocks affected")
    b.legend(frameon=False, loc="upper right", fontsize=8)
    fig.savefig(path, dpi=200, facecolor=SURFACE)


BENCH = [("selected:F2", "F2 network: identities, doses, monotherapy"), ("gp_f2", "GP, F2, all rows (exploratory)"),
         ("selected:F0", "F0 network: identities, doses"), ("cf_selected", "CF, extended: cell similarity, dose profile"),
         ("gp_f0", "GP, F0, per-drug dose (exploratory)"), ("cf_shar", "CF, Shar: pair-cell mean on every dose"),
         ("gp_full", "GP, his kernel, all rows (exploratory)"), ("gp_frozen", "GP, frozen pick: 3k-row fit, inducing"),
         ("gp_his", "GP, his budget: 1,488 rows"), ("deepsynergy_dose", "DeepSynergy, dose-aware: descriptors, expression"),
         ("bliss_reference", "Bliss: monotherapy, no fit"), ("additive", "Additive ridge: identities, doses")]
LEVEL_SERIES = [("selected:F2", "F2 network", BLUE), ("gp_f2", "F2 GP, all rows", AQUA), ("selected:F0", "F0 network", ORANGE),
                ("cf_selected", "CF, extended", YELLOW), ("gp_f0", "F0 GP, all rows", INK), ("gp_frozen", "frozen GP selection", MUTED)]


def figure_benchmark(final, levels, path):
    fig, (a, b) = _figure(9.0, 3.9, 2)
    f = final.set_index("model")
    rows = sorted([(m, label) for m, label in BENCH if m in f.index], key=lambda r: f.rmse[r[0]])
    y = np.arange(len(rows))[::-1].astype(float)
    _dots(a, y, [f.rmse[m] for m, _ in rows], [f.rmse_ci_low[m] for m, _ in rows], [f.rmse_ci_high[m] for m, _ in rows], MUTED, None)
    for yi, (m, _) in zip(y, rows):
        a.text(f.rmse_ci_high[m] + 0.3, yi, f"{f.rmse[m]:.2f}", va="center", color=INK, fontsize=8)
    a.set_yticks(y, [label for _, label in rows])
    a.set_xlabel("Test RMSE, inhibition points (95% pair bootstrap)")
    a.set_title("a  Test RMSE on identical rows")
    bins = ["R<25", "25-50", "50-75", "R>=75"]
    x = np.arange(len(bins))
    for model, label, color in LEVEL_SERIES:
        own = levels[levels.model == model].set_index("bin")
        if own.empty:
            continue
        b.plot(x, 100 * own.mse_change_vs_bliss[bins], color=color, linewidth=2, marker="o", markersize=7, label=label)
    b.axhline(0, color=MUTED, linewidth=1)
    b.set_xticks(x, ["< 25", "25–50", "50–75", "≥ 75"])
    b.grid(axis="y", color=GRID, linewidth=0.8)
    b.grid(axis="x", visible=False)
    b.set_xlabel("Bliss-expected inhibition")
    b.set_ylabel("MSE change vs Bliss (%)")
    b.set_title("b  MSE vs Bliss by Bliss level")
    b.legend(frameon=False, fontsize=8, loc="best")
    fig.savefig(path, dpi=200, facecolor=SURFACE)


TASKS = [("selected:F2", "F2 network"), ("gp_f2", "GP, F2 (exploratory)"), ("cf_selected", "CF, extended"), ("selected:F0", "F0 network"),
         ("cf_shar", "CF, Shar"), ("gp_f0", "GP, F0 (exploratory)"), ("bliss_reference", "Bliss"), ("gp_frozen", "GP, frozen pick"),
         ("deepsynergy_dose", "DeepSynergy, dose-aware"), ("additive", "Additive ridge")]


def figure_tasks(pair_cell, final, path):
    """The same predictions scored on Shar's task (pair-cell mean response) and on the dose rows."""
    fig, (a, b) = _figure(8.2, 3.4, 2)
    p, f = pair_cell.set_index("model"), final.set_index("model")
    rows = sorted([(m, label) for m, label in TASKS if m in p.index and m in f.index], key=lambda r: p.rmse[r[0]])
    y = np.arange(len(rows))[::-1].astype(float)
    for ax, table, title, xlabel in [(a, p, "a  Pair-cell mean (her target)", "RMSE of pair-cell means"),
                                     (b, f, "b  Every dose row (shared benchmark)", "RMSE on dose rows")]:
        for yi, (m, _) in zip(y, rows):
            color = YELLOW if m.startswith("cf_") else MUTED
            _dots(ax, [yi], [table.rmse[m]], [table.rmse_ci_low[m]], [table.rmse_ci_high[m]], color, None)
            ax.text(table.rmse_ci_high[m] + 0.02 * table.rmse.max(), yi, f"{table.rmse[m]:.2f}", va="center", color=INK, fontsize=8)
        ax.set_xlabel(xlabel + " (95% pair bootstrap)")
        ax.set_title(title)
        ax.set_xlim(0, table.rmse_ci_high[[m for m, _ in rows]].max() * 1.15)
    a.set_yticks(y, [label for _, label in rows])
    b.set_yticks(y, [""] * len(rows))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def figure_gp_scaling(frozen, gpu, path):
    """Validation RMSE of Kingsley's GP kernel against training rows and inducing points (fold 0 study, 39 cell lines)."""
    fig, (a, c, b) = _figure(9.6, 3.2, 3)
    g = gpu[(gpu.predictor == "exact") & (gpu.group != "demo")].set_index("variant")
    caps = [v for v in ["cap1500", "cap3000", "cap6000", "cap12000", "full"] if v in g.index]
    a.plot(g.rows_median[caps], g.validation_rmse[caps], color=BLUE, linewidth=2, marker="o", markersize=7, label="exact GP (GPU)")
    f = frozen.set_index(["group", "predictor"])
    cpu = [f.loc[(k, "exact")] for k in ["cap1500", "cap3000"] if (k, "exact") in f.index]
    a.scatter([r.training_rows_conditioned for r in cpu], [r.validation_rmse for r in cpu], s=110, facecolors="none",
              edgecolors=INK, linewidths=1.2, zorder=4, label="exact GP, frozen CPU study")
    a.set_xscale("log")
    a.set_xticks([1500, 3000, 6000, 12000, 22500], ["1.5k", "3k", "6k", "12k", "22.5k"])
    a.minorticks_off()
    a.grid(axis="y", color=GRID, linewidth=0.8)
    a.set_xlabel("Training rows per cell line")
    a.set_ylabel("Validation RMSE")
    a.set_title("a  His kernel, more rows")
    a.legend(frameon=False, fontsize=7, loc="upper right")
    rows = [("exact, all-row fit", g.validation_rmse.get("full"), BLUE, "o"),
            ("exact, 1.5k-row fit", g.validation_rmse.get("his_hyper_all_rows"), BLUE, "o"),
            ("inducing, 3k-row fit (frozen)", f.validation_rmse.get(("cap3000", "inducing")), ORANGE, "D"),
            ("inducing, 1.5k-row fit", f.validation_rmse.get(("cap1500", "inducing")), ORANGE, "D"),
            ("exact, no interaction term", g.validation_rmse.get("full_w0"), AQUA, "s")]
    rows = [r for r in rows if r[1] is not None and np.isfinite(r[1])]
    y = np.arange(len(rows))[::-1]
    for yi, (label, value, color, marker) in zip(y, rows):
        c.scatter(value, yi, s=55, color=color, marker=marker, zorder=3)
        c.text(value + 0.02, yi, f"{value:.2f}", va="center", fontsize=7, color=INK)
    c.set_yticks(y, [r[0] for r in rows], fontsize=7)
    c.set_xlabel("Validation RMSE")
    c.set_title("b  Conditioned on all 22.5k rows")
    left = min(r[1] for r in rows)
    c.set_xlim(left - 0.1, max(r[1] for r in rows) + 0.15)
    sparse = gpu[gpu.predictor.str.startswith("inducing") & (gpu.variant == "full") & (gpu.group != "demo")].sort_values("n_inducing")
    b.plot(sparse.n_inducing, sparse.validation_rmse, color=ORANGE, linewidth=2, marker="D", markersize=6, label="inducing-point posterior")
    if "full" in g.index:
        b.axhline(g.validation_rmse["full"], color=BLUE, linewidth=1.5, linestyle="--", label="exact posterior")
    b.set_xscale("log", base=2)
    b.set_xticks(sparse.n_inducing.tolist(), [f"{int(m):,}" for m in sparse.n_inducing])
    b.minorticks_off()
    b.grid(axis="y", color=GRID, linewidth=0.8)
    b.set_xlabel("Inducing points")
    b.set_ylabel("Validation RMSE")
    b.set_title("c  Inducing vs exact, all rows")
    b.legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def figure_uncertainty(calibration, deciles, labels, path):
    fig, (a, b) = _figure(7.4, 3.1, 2)
    colors = [BLUE, ORANGE, AQUA, INK]
    a.plot([0.5, 0.95], [0.5, 0.95], color=MUTED, linewidth=1, linestyle="--")
    for (name, label), color in zip(labels, colors):
        own = calibration[(calibration.method == name) & (calibration.sd == "predictive")]
        a.plot(own.level, own.coverage, color=color, linewidth=2, marker="o", markersize=6, label=label)
        base = calibration[(calibration.method == name) & (calibration.sd == "validation RMSE per cell")]
        if len(base):
            a.plot(base.level, base.coverage, color=color, linewidth=1.5, linestyle=":", marker="s", markersize=4)
    a.plot([], [], color=MUTED, linewidth=2, marker="o", markersize=6, label="solid: GP SD")
    a.plot([], [], color=MUTED, linewidth=1.5, linestyle=":", marker="s", markersize=4, label="dotted: per-cell baseline")
    a.set_xlabel("Nominal central-interval level")
    a.set_ylabel("Test coverage")
    a.grid(axis="y", color=GRID, linewidth=0.8)
    a.set_title("a  Calibration on held-out pairs")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    for (name, label), color in zip(labels, colors):
        own = deciles[deciles.method == name]
        b.plot(own.mean_std, own.rmse, color=color, linewidth=2, marker="o", markersize=5, label=label)
    top = float(max(deciles.mean_std.max(), deciles.rmse.max()))
    b.plot([0, top], [0, top], color=MUTED, linewidth=1, linestyle="--")
    b.set_xlabel("Mean predicted SD (decile)")
    b.set_ylabel("Test RMSE in decile")
    b.grid(axis="y", color=GRID, linewidth=0.8)
    b.set_title("b  Error against predicted SD")
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def figure_loewe_mechanism(probe_path, example_path, hist_path, path):
    """Worked fallback case (native fit) and the distance of every root to the edge of the joint effect range."""
    from .loewe_verify import curve
    example = json.loads(Path(example_path).read_text())
    ex, near = example["example"], example["all_cases_by_class"]
    probe = pd.read_csv(probe_path)
    r = probe[probe.case_id == ex["case_id"]].iloc[0]
    fig, (a, b) = _figure(7.6, 3.3, 2)
    dose = np.logspace(-3, 1.5, 300)
    for arm, color in [("a", BLUE), ("b", ORANGE)]:
        pars = [getattr(r, f"{arm}_{k}") for k in "bcde"]
        a.plot(dose, curve(np.log(dose), np.array(getattr(r, f"model_{arm}")), *pars), color=color, linewidth=2,
               label=f"{getattr(r, f'drug_{arm}')}, single agent")
    a.axhline(min(max(r.a_c, r.a_d), max(r.b_c, r.b_d)), color=MUTED, linewidth=1, linestyle="--")
    total = r.dose_a + r.dose_b
    a.scatter([total], [ex["fallback_recomputed"]], s=60, marker="D", color=INK, zorder=4)
    a.annotate(f"fallback {ex['fallback_recomputed']:.2f}\n(stronger curve at summed dose)", (total, ex["fallback_recomputed"]),
               textcoords="offset points", xytext=(-10, 10), ha="right", fontsize=7)
    a.scatter([total], [ex["root_brent"]], s=60, marker="o", color=AQUA, zorder=4)
    a.annotate(f"Loewe root {ex['root_brent']:.3f}, {ex['root_to_edge']:.0e} below the plateau.\n"
               f"nleqslv stopped at {ex['native_iterate']:.3f}, past the plateau\nwhere the equation is undefined: code 3",
               (total, ex["root_brent"]), xytext=(0.97, 0.06), textcoords="axes fraction", ha="right", va="bottom", fontsize=7,
               arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})
    a.set_xscale("log")
    a.set_xlabel("Dose (uM)")
    a.set_ylabel("Inhibition (%)")
    a.grid(axis="y", color=GRID, linewidth=0.8)
    a.set_title(f"a  {r.drug_a} {r.dose_a:g} uM + {r.drug_b} {r.dose_b:g} uM, {r.cell_line}")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    hist = pd.read_csv(hist_path)
    artifact = ["artifact: solver at root, answer discarded", "artifact: root exists, solver failed"]
    groups = [("accepted root", ["accepted: matches root"], BLUE), ("discarded admissible root", artifact, ORANGE)]
    for label, classes, color in groups:
        g = hist[hist["class"].isin(classes)].groupby("log10_low").cases.sum()
        n = sum(near[c]["cases"] for c in classes)
        b.step(g.index + 0.25, g / g.sum(), where="mid", color=color, linewidth=2, label=f"{label} (n = {n:,})")
    accepted = near["accepted: matches root"]["within_0.01"]
    discarded = sum(near[c]["within_0.01"] for c in artifact)
    b.axvline(-2, color=MUTED, linewidth=1, linestyle=":")
    b.text(-2.3, 0.2, f"within 0.01 points:\n{accepted:,} accepted\n{discarded:,} discarded", ha="right", va="center", fontsize=7)
    b.set_xlabel("log10 distance from root to edge of joint effect range")
    b.set_ylabel("Share of cases")
    b.grid(axis="y", color=GRID, linewidth=0.8)
    b.set_title("b  Discarded roots sit at the range edge")
    b.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def variance_shares(dataset="data/processed/oneil_v1"):
    """Where ONEIL response variance sits: dose and replicate rows within a pair-cell, cell lines, drug main
    effects within fold and cell, and the pair-specific interaction contrast that remains."""
    from .regimes import panel_of, projected
    from .neural_report import index, load_truth
    df, _, _ = load_truth(dataset)
    t = index(df)
    y = t["y"]
    unit_mean = (np.bincount(t["panel"], weights=y) / t["panel_n"])[t["panel"]]
    total, within = float(((y - y.mean()) ** 2).sum()), float(((y - unit_mean) ** 2).sum())
    frame, means = panel_of(t, np.ones(len(y), bool), [y])
    z = means[:, 0]
    panel_total = float(((z - z.mean()) ** 2).sum())
    cell_mean = frame.assign(z=z).groupby(["fold", "cell_line"]).z.transform("mean").to_numpy()
    within_cell = float(((z - cell_mean) ** 2).sum())
    energy, _ = projected(frame, means, [])
    contrast = float(energy[0, 0])
    return {"dose_rows": len(y), "pair_cell_units": len(z),
            "dose_level": {"within_pair_cell": within / total, "between_pair_cell": 1 - within / total},
            "pair_cell_level": {"fold_and_cell": 1 - within_cell / panel_total,
                                "drug_main_effects_within_cell": (within_cell - contrast) / panel_total,
                                "pair_specific_contrast": contrast / panel_total},
            "contrast_share_of_dose_variance": (1 - within / total) * contrast / panel_total}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", default="results/summary")
    p.add_argument("--loewe-cases", help="loewe_audit/full_cases.parquet")
    p.add_argument("--loewe-verified", help="loewe_verify/cases_verified.parquet")
    p.add_argument("--unified", default="results/unified")
    p.add_argument("--gp", default="results/gp")
    p.add_argument("--gp-gpu", default="results/gp_gpu")
    p.add_argument("--uncertainty", default="results/gp_compare/uncertainty")
    p.add_argument("--uncertainty-label", action="append", default=[], metavar="METHOD=LABEL")
    p.add_argument("--variance-shares", action="store_true", help="also decompose ONEIL response variance (needs the dataset)")
    p.add_argument("--pair-cell", default="results/compare/pair_cell.csv", help="synergy_artifacts.cf compare output")
    a = p.parse_args()
    out = Path(a.output)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    table = cross_screen()
    table.to_csv(out / "cross_screen.csv", index=False, float_format="%.6g", lineterminator="\n")
    if a.variance_shares:
        (out / "variance_shares.json").write_text(json.dumps(variance_shares(), indent=2) + "\n")
    figure_cross_screen(table, out / "figures" / "fig1_cross_screen.png")
    unified, gp = Path(a.unified), Path(a.gp)
    if (unified / "selected_models.csv").exists() and (unified / "bliss_levels.csv").exists():
        figure_benchmark(pd.read_csv(unified / "selected_models.csv"), pd.read_csv(unified / "bliss_levels.csv"), out / "figures" / "fig2_benchmark.png")
    if (gp / "study_summary.csv").exists() and Path(a.gp_gpu, "study_summary.csv").exists():
        figure_gp_scaling(pd.read_csv(gp / "study_summary.csv"), pd.read_csv(Path(a.gp_gpu, "study_summary.csv")),
                          out / "figures" / "fig3_gp_scaling.png")
    if Path(a.pair_cell).exists() and (unified / "selected_models.csv").exists():
        figure_tasks(pd.read_csv(a.pair_cell), pd.read_csv(unified / "selected_models.csv"), out / "figures" / "fig7_two_tasks.png")
    uncertainty = Path(a.uncertainty)
    if (uncertainty / "calibration.csv").exists():
        labels = [tuple(x.split("=", 1)) for x in a.uncertainty_label]
        figure_uncertainty(pd.read_csv(uncertainty / "calibration.csv"), pd.read_csv(uncertainty / "deciles.csv"), labels,
                           out / "figures" / "fig5_gp_uncertainty.png")
    if a.loewe_cases:
        published, corrected, effect = loewe_blocks(a.loewe_cases, a.loewe_verified)
        figure_loewe(published, corrected, out / "figures" / "fig4_loewe.png")
        print(effect)
    if Path("results/loewe_audit/edge_histogram.csv").exists():
        figure_loewe_mechanism("results/loewe_audit/probe_98_cases.csv", "results/loewe_audit/example.json",
                               "results/loewe_audit/edge_histogram.csv", out / "figures" / "fig6_loewe_mechanism.png")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
