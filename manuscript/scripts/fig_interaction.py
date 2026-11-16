"""Figures 3, 4, 5, S2 and S3: response versus interaction accuracy, decomposition and same-pair transfer."""
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import figstyle as S

ADDITIVE, PAIRCELL, DOSESHAPE = "#9e9e9e", "#b2182b", "#ef8a62"


def save(fig, path):
    S.save(fig, path)


def minus(s):
    """Minus signs for negative numbers; hyphens in words are kept."""
    return S.minus(s)


# ------------------------------------------------------------------------------------------------- Figure 3

def response_r2(inp):
    meta = json.loads(inp["resolution_inputs"].read_text())
    with np.load(inp["resolution_cache"]) as z:
        rpe = z["row_pair_error"]
    var_y = meta["raw_y_variance"]

    def resp(names):
        idx = [meta["names"].index(n) for n in names]
        return 1 - rpe[:, idx].sum(0).mean() / meta["rows"] / var_y
    return resp, var_y


def fig3(inp, path):
    var = inp.json("variance_shares")
    ed = pd.read_csv(inp["error_decomposition"]).set_index("model")
    ia = inp.json("interaction_audit")["results"]
    ps = inp.json("pairseen_audit")
    sh = var["shares"]
    comp = [("Additive drug and drug $\\times$ dose", sh["between_panel_additive"] + sh["within_panel_additive_dose"], ADDITIVE),
            ("Pair-cell contrast", sh["between_panel_contrast"], PAIRCELL),
            ("Dose-shape contrast", sh["within_panel_dose_contrast"], DOSESHAPE),
            ("Replicate deviations", sh["replicate_deviations"], "white")]
    fig = plt.figure(figsize=(S.WIDTH, 3.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.32, 1], height_ratios=[0.42, 1.5], left=0.205, right=0.985,
                          bottom=0.115, top=0.86, wspace=0.42, hspace=1.25)
    # (a) allocation of row variance
    ax = fig.add_subplot(gs[0, 0])
    x = 0
    for name, v, col in comp:
        ax.barh(0, 100 * v, left=x, color=col, height=0.62, edgecolor=S.MUTED if col == "white" else "white",
                lw=0.5, hatch="////" if col == "white" else ".." if col == DOSESHAPE else None)
        x += 100 * v
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of row variance (%)")
    ax.text(100 * comp[0][1] / 2, 0, f"{100 * comp[0][1]:.1f}%", ha="center", va="center", fontsize=6.8, color="white")
    xs = 100 * comp[0][1]
    for k, (name, v, col) in enumerate(comp[1:]):
        mid = xs + 100 * v / 2
        ax.annotate(f"{100 * v:.1f}%", (mid, 0.31), (mid + [-13, -4, 5][k], 0.75), fontsize=6.3, ha="center",
                    va="bottom", arrowprops=dict(arrowstyle="-", lw=0.5, color=S.MUTED, shrinkA=0, shrinkB=0))
        xs += 100 * v
    ax.set_ylim(-0.5, 1.25)
    fig.text(0.012, 0.935, "a", fontsize=9, fontfamily=S.BOLD, va="bottom")
    fig.text(0.04, 0.935, "Measured response: allocation of the variance of 1,475,328 dose rows", fontsize=7,
             va="bottom")
    # (b) where each model's error lies
    ax = fig.add_subplot(gs[1, 0])
    models = [("Selected F2 network", ["neural_F2:2026", "neural_F2:2027", "neural_F2:2028"]),
              ("GP F2", ["gp_F2"]), ("CF, dose-aware", ["cf_selected"]),
              ("Base F2, interpolated", ["neural_base_F2:2026", "neural_base_F2:2027", "neural_base_F2:2028"]),
              ("Bliss reference", ["bliss"])]
    parts = [("between_panel_additive", "within_panel_additive_dose"), ("between_panel_contrast",),
             ("within_panel_dose_contrast",), ("replicate_deviations",)]
    for i, (name, rows) in enumerate(models):
        y = len(models) - 1 - i
        e = ed.loc[rows].mean()
        left = 0
        for (pk, (cname, _, col)) in zip(parts, comp):
            v = float(sum(e[k] for k in pk))
            ax.barh(y, v, left=left, color=col, height=0.62, edgecolor=S.MUTED if col == "white" else "white", lw=0.5,
                    hatch="////" if col == "white" else ".." if col == DOSESHAPE else None)
            left += v
        ax.text(left + 3, y, f"{left:.0f}", va="center", fontsize=6.2)
    ax.axvline(var["components"]["replicate_deviations"], color=S.MUTED, lw=0.6, ls=":")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[0] for m in models[::-1]])
    ax.set_xlim(0, 215)
    ax.set_xlabel("Mean squared error (points$^2$), by component")
    ax.text(var["components"]["replicate_deviations"] + 2, len(models) - 0.45, "replicate floor", fontsize=6,
            color=S.MUTED, va="bottom")
    ax.set_ylim(-0.6, len(models) - 0.2)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=S.MUTED if c == "white" else "none", lw=0.5,
                             hatch="////" if c == "white" else ".." if c == DOSESHAPE else None) for _, _, c in comp]
    ax.legend(handles, [c[0] for c in comp], loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2, fontsize=6.2,
              handlelength=1.2, columnspacing=1.0)
    pos = ax.get_position()
    fig.text(0.012, pos.y1 + 0.115, "b", fontsize=9, fontfamily=S.BOLD, va="bottom")
    fig.text(0.04, pos.y1 + 0.115, "Where each held-out-pair model's error lies", fontsize=7, va="bottom")
    # (c) response versus interaction accuracy
    resp, var_y = response_r2(inp)
    pc = ia["pair_cell:bliss_residual"]
    ax = fig.add_subplot(gs[:, 1])
    pts = [("Selected F2", resp([f"neural_F2:{s}" for s in (2026, 2027, 2028)]), "neural_F2", "o"),
           ("GP F2", resp(["gp_F2"]), "gp_F2", "D"), ("CF", resp(["cf_selected"]), "cf_selected", "^"),
           ("Base F2", resp([f"neural_base_F2:{s}" for s in (2026, 2027, 2028)]), "neural_base_F2", "s")]
    share = pc["noise_share_grid_split"]
    lo, hi = pc["noise_share_grid_pair_interval"]
    ax.axhspan(1 - hi, 1 - lo, color=S.UNCERT_FILL, lw=0)
    ax.axhline(1 - share, color=S.UNCERT, lw=0.8)
    ax.text(0.8015, 1 - share + 0.025, f"interaction ceiling {1 - share:.2f} (within batch)", fontsize=6.2,
            color=S.MUTED, va="bottom")
    rc = 1 - var["shares"]["replicate_deviations"]
    ax.axvline(rc, color=S.UNCERT, lw=0.8)
    ax.text(rc - 0.002, 0.45, "response ceiling\n(1 $-$ replicate share)", fontsize=6.0, color=S.MUTED,
            rotation=90, ha="right", va="center")
    offs = {"Selected F2": (14, 22, "left"), "GP F2": (16, 2, "left"), "CF": (-10, 18, "right"),
            "Base F2": (12, -8, "left")}
    for name, x, key, mk in pts:
        iv = ia[f"pair_cell:bliss_residual:{key}:intervals"]["pair_interval"]
        ax.plot([x, x], iv, color=S.HELDOUT, lw=1.6, solid_capstyle="butt")
        ax.scatter(x, pc["models"][key]["r2"], color=S.HELDOUT, marker=mk, s=20, edgecolor="white", lw=0.5, zorder=3)
        dx, dy, ha = offs[name]
        ax.annotate(name, (x, pc["models"][key]["r2"]), xytext=(dx, dy), textcoords="offset points", fontsize=6.2,
                    ha=ha, va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color=S.MUTED, shrinkA=0, shrinkB=2))
    xb = resp(["bliss"])
    ax.scatter(xb, 0, color="white", edgecolor=S.INK, marker="s", s=18, lw=0.8, zorder=3)
    ax.annotate("Bliss reference", (xb, 0), xytext=(0, -11), textcoords="offset points", fontsize=6.2, ha="center")
    r3 = ps["results"]["pair_cell:bliss_residual"]
    xp = 1 - ps["response_mse"]["pair_seen_mean"] / var_y
    ax.plot([xp, xp], r3["pair_seen_pair_interval"], color=S.B3, lw=1.6, solid_capstyle="butt")
    ax.scatter(xp, r3["pair_seen_r2"], color=S.B3, s=20, edgecolor="white", lw=0.5, zorder=3)
    ax.annotate("F2, pair seen\nin other cell lines", (xp, r3["pair_seen_r2"]), xytext=(-5, 0),
                textcoords="offset points", fontsize=6.2, ha="right", va="center", color=S.INK)
    ax.axhline(0, color=S.MUTED, lw=0.5)
    ax.set_xlim(0.80, 0.955)
    ax.set_ylim(-0.12, 1.0)
    ax.set_xlabel("Response $R^2$ (all dose rows)")
    ax.set_ylabel("Interaction $R^2$ (Bliss pair-cell contrast)")
    fig.text(ax.get_position().x0 - 0.075, 0.935, "c", fontsize=9, fontfamily=S.BOLD, va="bottom")
    fig.text(ax.get_position().x0 - 0.047, 0.935, "Response versus interaction accuracy", fontsize=7, va="bottom")
    save(fig, path)


# ------------------------------------------------------------------------------------------------- Figure 4

def pair_matrix(inp, col="t_bliss"):
    t = pd.read_parquet(inp["targets"], columns=["pair_id", "cell_line", "w", col])
    T = t.pivot(index="pair_id", columns="cell_line", values=col)
    Wm = t.pivot(index="pair_id", columns="cell_line", values="w")
    pm = (T * Wm).sum(1) / Wm.sum(1)
    return T, Wm, pm


def fig4(inp, path):
    vc = inp.json("variance_components")
    T, Wm, pm = pair_matrix(inp)
    order = pm.sort_values().index
    Ts = T.loc[order].to_numpy()
    centred = Ts - pm.loc[order].to_numpy()[:, None]
    lim = float(np.ceil(np.percentile(np.abs(Ts), 98)))
    nrm = TwoSlopeNorm(0, -lim, lim)
    W, H = S.WIDTH, 4.5
    fig = plt.figure(figsize=(W, H))

    def ax_(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def head(x, y, letter, text):
        fig.text(x / W, y / H, letter, fontsize=9, fontfamily=S.BOLD, va="bottom")
        fig.text((x + 0.19) / W, y / H, text, fontsize=7, va="bottom")

    hy, hh = 0.75, 3.3
    axm = ax_(0.42, hy, 0.13, hh)
    axm.imshow(pm.loc[order].to_numpy()[:, None], aspect="auto", cmap=S.RESIDUAL, norm=nrm, origin="lower")
    axm.set_xticks([])
    axm.set_yticks([0, 291, 582])
    axm.set_yticklabels(["1", "292", "583"])
    axm.set_ylabel("Drug pairs, ordered by pair mean", labelpad=1)
    axm.set_title("pair\nmean", fontsize=6.2)
    for sp in axm.spines.values():
        sp.set_visible(False)
    for k, (M, ttl) in enumerate([(Ts, "contrast $t_{pc}$"), (centred, "$t_{pc}$ minus pair mean")]):
        ax = ax_(0.62 + k * 1.2, hy, 1.08, hh)
        im = ax.imshow(M, aspect="auto", cmap=S.RESIDUAL, norm=nrm, origin="lower", interpolation="nearest")
        ax.set_yticks([])
        ax.set_xticks([0, 19, 38])
        ax.set_xticklabels(["1", "20", "39"])
        ax.set_xlabel("Cell lines", labelpad=1)
        ax.set_title(ttl, fontsize=6.8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    head(0.03, hy + hh + 0.3, "a", "Bliss contrast, 583 pairs $\\times$ 39 cell lines")
    cax = ax_(0.62, 0.29, 2.28, 0.06)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.set_label("Bliss pair-cell contrast (points)", fontsize=6.2, labelpad=1.5)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(0.4)
    # (b) energy shares by target with controls
    ax = ax_(4.12, 2.6, 2.03, 1.2)
    rows = [("Bliss residual", vc["components"]["bliss"]), ("Measured response", vc["components"]["raw"]),
            ("Train-only baseline", vc["components"]["train"]), ("HSA residual", vc["components"]["hsa"]),
            ("Reference only", vc["components"]["ref"])]
    cols = [("noise", "white"), ("pair_wide", S.INK), ("pair_by_cell", S.UNCERT)]
    for i, (name, c) in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0
        for key, col in cols:
            v = max(c[key], 0)
            ax.barh(y, v, left=left, color=col, height=0.62, edgecolor=S.MUTED if col == "white" else "white", lw=0.5,
                    hatch="////" if col == "white" else None)
            left += v
        if i == 0:
            lo, hi = c["pair_wide_pair_interval"]
            dlo, dhi = c["pair_wide_drug_interval"]
            x0 = c["noise"]
            yy = y + 0.5
            ax.plot([x0 + dlo, x0 + dhi], [yy, yy], color=S.INK, lw=0.6)
            ax.plot([x0 + lo, x0 + hi], [yy, yy], color=S.INK, lw=1.8, solid_capstyle="butt")
            for v, xx, tc in [(c["pair_wide"], c["noise"] + c["pair_wide"] / 2, "white"),
                              (c["pair_by_cell"], c["noise"] + c["pair_wide"] + c["pair_by_cell"] / 2, S.INK)]:
                ax.text(xx, y, f"{v:.2f}", ha="center", va="center", fontsize=6, color=tc)
            ax.text(c["noise"] / 2, y + 0.42, f"{c['noise']:.2f}", ha="center", va="bottom", fontsize=6)
    perm = vc["controls"]["pair_permutation"]["pair_wide_mean"]
    zero = vc["controls"]["zero_interaction"]["noise_share_mean"]
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(rows) - 0.05)
    ax.set_xlabel("Share of contrast energy", labelpad=1)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=S.MUTED if c == "white" else "none", lw=0.5,
                             hatch="////" if c == "white" else None) for _, c in cols]
    ax.legend(handles, ["replicate noise", "pair-wide offset", "pair-by-cell"], ncol=3, loc="lower left",
              bbox_to_anchor=(-0.5, 1.03), fontsize=6.1, handlelength=1.1, columnspacing=0.9)
    head(3.35, hy + hh + 0.3, "b", "Energy shares of the pair-cell contrast")
    fig.text(6.15 / W, 2.24 / H, minus(f"Controls: permuted pair labels, pair-wide {perm:.4f}; simulated zero "
                                       f"interaction, noise {zero:.3f}"), ha="right", va="top", fontsize=5.9,
             color=S.CONTROL)
    # (c) cross-batch replication
    cb_ = pd.read_csv(inp["cross_batch"])
    cs = vc["cross_batch"]["bliss_frozen_reference"]["cell_shift"]
    head(3.35, 1.72, "c", "Re-measured pair-cells and the ceiling")
    ax = ax_(3.72, 0.3, 1.12, 1.12)
    x1, x2 = cb_["bliss_frozen_reference:cell_shift:t1"], cb_["bliss_frozen_reference:cell_shift:t2"]
    m = float(np.ceil(max(np.abs(x1).max(), np.abs(x2).max()) / 5) * 5)
    ax.plot([-m, m], [-m, m], color=S.RULE, lw=0.6)
    ax.scatter(x1, x2, s=3.5, color=S.INK, alpha=0.6, lw=0)
    ax.set_xlim(-m, m)
    ax.set_ylim(-m, m)
    ax.set_xlabel("Batch 1 contrast", labelpad=1)
    ax.set_ylabel("Batch 2 contrast", labelpad=1)
    lo, hi = cs["replication_overall_pair_interval"]
    ax.text(-m * 0.95, m * 0.95, f"240 pair-cells, 16 pairs\nratio {cs['replication_overall']:.2f} "
            f"[{lo:.2f}, {hi:.2f}]", fontsize=5.8, va="top")
    ax = ax_(5.25, 0.3, 1.0, 1.12)
    nb = vc["components"]["bliss"]
    within = 1 - nb["noise"]
    wi = [1 - nb["noise_pair_interval"][1], 1 - nb["noise_pair_interval"][0]]
    ax.bar(0, within, color=S.UNCERT_FILL, width=0.6, edgecolor=S.UNCERT, lw=0.5)
    ax.plot([0, 0], wi, color=S.INK, lw=1.6, solid_capstyle="butt")
    ax.text(0, within + 0.03, f"{within:.2f}", ha="center", va="bottom", fontsize=6)
    ax.add_patch(plt.Rectangle((0.7, 0.67), 0.6, 0.82 - 0.67, facecolor=S.UNCERT, edgecolor="none"))
    ax.text(1.0, 0.85, "0.67$\\endash$\n0.82", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["within\nbatch", "across\nbatches"], fontsize=6)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Reproducible share", labelpad=1)
    save(fig, path)
    return {"heatmap_clip": lim, "pairs": int(T.shape[0]), "cells": int(T.shape[1])}


# ------------------------------------------------------------------------------------------------- Figure 5

def curve(cur, target, model, ks=(1, 2, 4, 8, 16)):
    y = np.array([cur[f"{target}:k={k}"].get(model, {}).get("r2", np.nan) for k in ks])
    lo = np.array([cur[f"{target}:k={k}"].get(model, {}).get("pair_interval", [np.nan] * 2)[0] for k in ks])
    hi = np.array([cur[f"{target}:k={k}"].get(model, {}).get("pair_interval", [np.nan] * 2)[1] for k in ks])
    return np.array(ks, float), y, lo, hi


def anchor_panel(ax, cur, loco, target, controls=True, small=False):
    xl = 64
    if loco is not None:
        ax.axvspan(26, 90, color="#f3f3f3", lw=0, zorder=0)
        ax.axvline(32, color=S.MUTED, lw=0.5, ls=":")
    for m, col in [("B1", S.B1), ("B2", S.B2)]:
        x, y, lo, hi = curve(cur, target, m)
        ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(x, y, color=col, lw=1.2, marker="o", ms=2.6 if small else 3.2)
        if loco is not None:
            v = loco[m]
            ax.plot([xl, xl], v["pair_interval"], color=col, lw=1.6, solid_capstyle="butt")
            ax.scatter(xl, v["r2"], color=col, marker="s", s=14, zorder=3, edgecolor="white", lw=0.4)
    if controls:
        for m, ls, name in [("B4_B2", "--", "donor sharing a drug"), ("B5a_B2", ":", "unrelated donor"),
                            ("B5b_B2", "-.", "anchors deranged")]:
            x, y, lo, hi = curve(cur, target, m)
            ax.plot(x, y, color=S.CONTROL, lw=1.0, ls=ls, marker="o", ms=2.2, label=f"B2, {name}")
    ax.axhline(0, color=S.MUTED, lw=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16] + ([xl] if loco is not None else []))
    ax.set_xticklabels(["1", "2", "4", "8", "16"] + (["38\nLOCO"] if loco is not None else []))
    ax.minorticks_off()
    ax.set_xlim(0.8, 90 if loco is not None else 20)


def fig5(inp, path):
    ae = inp.json("anchor_experiment")
    vc = inp.json("variance_components")
    b3 = inp.json("b3_paired")
    cur = ae["curves"]
    fig = plt.figure(figsize=(S.WIDTH, 3.95))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[2.6, 0.62], left=0.09, right=0.985,
                          bottom=0.1, top=0.92, wspace=0.55, hspace=0.75)
    # (a) anchor curves
    ax = fig.add_subplot(gs[0, 0])
    anchor_panel(ax, cur, vc["loco"]["bliss"], "bliss")
    ax.text(4, -0.385, "23 fixed evaluation cells", ha="center", va="bottom", fontsize=5.6, color=S.MUTED)
    ax.text(64, -0.385, "39 evaluation\ncell lines", ha="center", va="bottom", fontsize=5.6, color=S.MUTED)
    ax.set_ylim(-0.4, 0.62)
    ax.set_xlabel("Anchor cell lines of the held-out pair ($k$)")
    ax.set_ylabel("$R^2$, Bliss pair-cell contrast")
    x, y, _, _ = curve(cur, "bliss", "B2")
    ax.text(17.5, y[-1] + 0.005, "B2", color=S.B2, fontsize=6.5, va="center")
    x, y1, _, _ = curve(cur, "bliss", "B1")
    ax.text(17.5, y1[-1] - 0.01, "B1", color=S.B1, fontsize=6.5, va="center")
    ax.legend(loc="upper left", fontsize=5.9, handlelength=3.0, borderaxespad=0.1, title="Controls",
              title_fontsize=5.9, alignment="left")
    S.head(ax, "a", "Held-out pairs: anchors of the same pair", dx=-0.075, dy=0.02)
    # (b) matched comparison and paired differences
    ax = fig.add_subplot(gs[0, 1])
    mods = [("B1 pair offset", b3["models"]["B1_pair_intercept"], S.B1),
            ("B2 cross-cell linear", b3["models"]["B2_cross_cell_covariance"], S.B2),
            ("B3 pair-seen network", b3["models"]["B3_pair_seen_network"], S.B3),
            ("F2, pair held out", b3["models"]["F2_held_out_pair"], S.HELDOUT)]
    full = dict(b3["differences"]["B3-B1"])
    full["r2_difference"] = (inp.json("pairseen_audit")["results"]["pair_cell:bliss_residual"]["pair_seen_r2"]
                             - ae["matched_pair_seen"]["bliss:leak_free"]["B1:projected"]["r2"])
    diffs = [("B2 $-$ B1", b3["differences"]["B2-B1"]), ("B3 $-$ B1", full),
             ("B3 $-$ B2", b3["differences"]["B3-B2"])]
    ys = [7.2, 6.2, 5.2, 4.2, 2.4, 1.4, 0.4]
    for (name, m, col), y in zip(mods, ys[:4]):
        S.interval(ax, y, m["r2"], m["pair_interval"], m["drug_interval"], color=col)
        ax.text(m["drug_interval"][1] + 0.012, y, minus(f"{m['r2']:.3f}"), va="center", fontsize=6)
    for (name, d), y in zip(diffs, ys[4:]):
        S.interval(ax, y, d["r2_difference"], d["pair_interval"], d["drug_interval"], color=S.INK, marker="D", size=12)
        ax.text(d["drug_interval"][1] + 0.012, y, minus(f"{d['r2_difference']:+.3f}"), va="center",
                fontsize=6)
    ax.set_yticks(ys)
    ax.set_yticklabels([m[0] for m in mods] + [d[0] for d in diffs])
    ax.axvline(0, color=S.MUTED, lw=0.5)
    ax.axhline(3.3, color=S.RULE, lw=0.5)
    ax.set_xlim(-0.16, 0.42)
    ax.set_ylim(-0.2, 7.8)
    ax.set_xlabel("$R^2$ or paired difference")
    ax.text(-0.155, 3.15, "paired differences", fontsize=5.9, color=S.MUTED, va="top")
    ax.text(-0.155, 7.75, "same rows, about 23 anchors", fontsize=5.9, color=S.MUTED, va="top")
    S.head(ax, "b", "Pair-cell split: matched comparison", dx=-0.2, dy=0.02)
    # (c) energy budget, leave one cell line out
    ax = fig.add_subplot(gs[1, :])
    lo = vc["loco"]["bliss"]
    noise = vc["components"]["bliss"]["noise"]
    segs = [("B1 pair\noffset", lo["B1"]["r2"], S.B1, "white"),
            ("B2 transfer\nbeyond offset", lo["B2"]["r2"] - lo["B1"]["r2"], S.B2, "white"),
            ("within-batch reproducible,\nunpredicted by B1/B2", 1 - noise - lo["B2"]["r2"], "white", S.INK),
            ("replicate\nnoise", noise, S.UNCERT_FILL, S.INK)]
    left = 0
    for name, v, col, tc in segs:
        ax.barh(0, v, left=left, color=col, height=0.5, edgecolor=S.MUTED, lw=0.5)
        ax.text(left + v / 2, 0, f"{v:.3f}", ha="center", va="center", fontsize=6.5, color=tc)
        ax.text(left + v / 2, 0.36, name, ha="center", va="bottom", fontsize=6.3)
        left += v
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 0.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Share of total Bliss contrast energy; 39 cells evaluated leave-one-cell-out, 38 anchors")
    S.head(ax, "c", "Total-energy budget for the B1/B2 leave-one-cell-out analysis", dx=-0.075, dy=0.06)
    save(fig, path)


# ------------------------------------------------------------------------------------------------- Figure S2

def figS2(inp, path):
    ae = inp.json("anchor_experiment")
    vc = inp.json("variance_components")
    b3 = inp.json("b3_paired")
    cur = ae["curves"]
    fig = plt.figure(figsize=(S.WIDTH, 3.9))
    gs = fig.add_gridspec(2, 4, left=0.085, right=0.985, bottom=0.11, top=0.92, wspace=0.45, hspace=0.8,
                          height_ratios=[1, 1])
    targets = [("train", "Train-only baseline", vc["loco"].get("train")), ("raw", "Measured response", vc["loco"].get("raw")),
               ("f2_residual", "Bliss minus held-out F2", None), ("ref", "Reference only", vc["loco"].get("ref"))]
    for k, (t, name, lo) in enumerate(targets):
        ax = fig.add_subplot(gs[0, k])
        anchor_panel(ax, cur, lo, t, controls=False, small=True)
        ax.set_title(name, fontsize=6.8)
        ax.set_ylim(-0.05, 0.62)
        if k == 0:
            ax.set_ylabel("$R^2$")
            S.label(ax, "a", dx=-28)
        ax.set_xlabel("$k$")
    ax = fig.add_subplot(gs[1, :2])
    fo = np.arange(5)
    for j, (key, col, mk, name) in enumerate([("B1_pair_intercept", S.B1, "o", "B1"),
                                              ("B2_cross_cell_covariance", S.B2, "s", "B2"),
                                              ("B3_pair_seen_network", S.B3, "D", "B3"),
                                              ("F2_held_out_pair", S.HELDOUT, "^", "F2, pair held out")]):
        ax.scatter(fo + (j - 1.5) * 0.12, b3["models"][key]["per_fold"], color=col, marker=mk, s=14, label=name,
                   edgecolor="white", lw=0.4)
    ax.axhline(0, color=S.MUTED, lw=0.5)
    ax.set_xticks(fo)
    ax.set_xticklabels([f"fold {f}" for f in fo])
    ax.set_ylabel("$R^2$")
    ax.set_ylim(-0.12, 0.3)
    ax.legend(ncol=4, loc="upper left", fontsize=6, bbox_to_anchor=(0, 1.18), handletextpad=0.2, columnspacing=1.0)
    S.label(ax, "b", dx=-28, dy=12)
    ax = fig.add_subplot(gs[1, 2:])
    ks = [1, 2, 4, 8, 16]
    sd1 = [cur[f"bliss:k={k}"]["B2-B1"]["r2_difference"] for k in ks]
    iv = [cur[f"bliss:k={k}"]["B2-B1"]["pair_interval"] for k in ks]
    ax.fill_between(ks, [v[0] for v in iv], [v[1] for v in iv], color=S.B2, alpha=0.18, lw=0)
    ax.plot(ks, sd1, color=S.B2, marker="o", ms=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.axhline(0, color=S.MUTED, lw=0.5)
    ax.set_ylim(0, 0.12)
    ax.set_xlabel("$k$")
    ax.set_ylabel("B2 $-$ B1, Bliss contrast")
    S.label(ax, "c", dx=-28, dy=12)
    save(fig, path)


# ------------------------------------------------------------------------------------------------- Figure S3

def figS3(inp, path):
    vc = inp.json("variance_components")["cross_batch"]["bliss_frozen_reference"]
    cb_ = pd.read_csv(inp["cross_batch"])
    fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH, 2.25), gridspec_kw={"width_ratios": [1, 1, 1.25], "wspace": 0.5})
    m = 0
    for k in ("none", "cell_shift"):
        m = max(m, cb_[f"bliss_frozen_reference:{k}:t1"].abs().max(), cb_[f"bliss_frozen_reference:{k}:t2"].abs().max())
    m = float(np.ceil(m / 5) * 5)
    for ax, k, ttl, s in [(axes[0], "none", "As measured", "a"), (axes[1], "cell_shift", "Cell-line batch shift removed", "b")]:
        x1, x2 = cb_[f"bliss_frozen_reference:{k}:t1"], cb_[f"bliss_frozen_reference:{k}:t2"]
        ax.plot([-m, m], [-m, m], color=S.RULE, lw=0.6)
        ax.scatter(x1, x2, s=4, color=S.INK, alpha=0.6, lw=0)
        ax.set_xlim(-m, m)
        ax.set_ylim(-m, m)
        ax.set_aspect("equal")
        ax.set_xlabel("Batch 1 contrast")
        ax.set_ylabel("Batch 2 contrast")
        ax.set_title(ttl, fontsize=6.8)
        ax.text(0.04, 0.96, f"ratio {vc[k]['replication_overall']:.2f}", transform=ax.transAxes, fontsize=6.2, va="top")
        S.label(ax, s, dx=-28)
    ax = axes[2]
    keys = [("replication_overall", "overall"), ("replication_pair_wide", "pair-wide"),
            ("replication_pair_by_cell", "pair-by-cell")]
    for j, k in enumerate(("none", "cell_shift")):
        for i, (key, name) in enumerate(keys):
            v = vc[k][key]
            iv = vc[k].get(key + "_pair_interval")
            x = i + (j - 0.5) * 0.3
            col = S.UNCERT if k == "none" else S.INK
            if iv:
                ax.plot([x, x], iv, color=col, lw=1.4, solid_capstyle="butt")
            ax.scatter(x, v, color=col, s=14, zorder=3, label=("as measured" if k == "none" else "shift removed")
                       if i == 0 else None)
    ax.axhline(1, color=S.RULE, lw=0.6)
    ax.set_xticks(range(3))
    ax.set_xticklabels([k[1] for k in keys])
    ax.set_ylabel("Cross / within-batch energy")
    ax.set_ylim(-0.4, 1.3)
    ax.legend(loc="lower right", fontsize=6)
    S.label(ax, "c", dx=-28)
    save(fig, path)
