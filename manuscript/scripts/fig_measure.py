"""Figures 1, 2 and S1: measured dose matrices, references and the evaluation design."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge
from matplotlib.colors import TwoSlopeNorm, Normalize

import figstyle as S

PCTS = (5, 50, 95)
PCT_NAME = {5: "5th percentile", 50: "median", 95: "95th percentile"}


def save(fig, path):
    S.save(fig, path)


def ll4(x, b, c, d, e):
    return c + (d - c) / (1 + (x / e) ** b)


def select_cases(inp):
    """Pair-cells measured in one batch whose Bliss contrast is closest to the 5th, 50th and 95th percentiles."""
    t = pd.read_parquet(inp["targets"], columns=["pair_id", "cell_line", "drug_a", "drug_b", "w", "t_bliss"])
    q = np.percentile(t.t_bliss, PCTS)
    one = t[t.w == 64].reset_index(drop=True)
    out = []
    for p, v in zip(PCTS, q):
        k = int(np.argmin(np.abs(one.t_bliss.to_numpy() - v)))
        r = one.iloc[k]
        out.append(dict(pct=p, quantile=float(v), pair_id=str(r.pair_id), drug_a=str(r.drug_a), drug_b=str(r.drug_b),
                        cell_line=str(r.cell_line), t_bliss=float(r.t_bliss)))
    return out, len(t), len(one)


def load_matrices(inp, cases):
    cols = ["block_id", "drug_a", "drug_b", "cell_line", "dose_a", "dose_b", "inhibition"]
    o = pd.read_parquet(inp["observations"], columns=cols)
    for c in cases:
        b = o[(o.drug_a == c["drug_a"]) & (o.drug_b == c["drug_b"]) & (o.cell_line == c["cell_line"])]
        da, db = np.sort(b.dose_a.unique()), np.sort(b.dose_b.unique())
        assert len(da) == 5 and len(db) == 5 and b.block_id.nunique() == 4
        piv = b.pivot_table(index="dose_a", columns="dose_b", values="inhibition", aggfunc="mean").loc[da, db]
        reps = {k: g.pivot_table(index="dose_a", columns="dose_b", values="inhibition").loc[da, db].to_numpy()
                for k, g in b.groupby("block_id")}
        Y = piv.to_numpy()
        a, bb = Y[:, 0], Y[0, :]
        E = a[:, None] + bb[None, :] - a[:, None] * bb[None, :] / 100
        c.update(doses_a=da, doses_b=db, Y=Y, E=E, R=Y - E, reps=reps)
    return cases


def attach_fits(inp, cases):
    """LL.4 fits of the native pipeline, oriented to drug_a / drug_b by matching dose sets."""
    p = pd.read_parquet(inp["native_panel"], columns=["drug_a", "drug_b", "cell_line", "surface_id"])
    s = pd.read_parquet(inp["surfaces"])
    for c in cases:
        sid = p[(p.drug_a == c["drug_a"]) & (p.drug_b == c["drug_b"]) & (p.cell_line == c["cell_line"])].surface_id
        assert len(sid) == 1
        x = s[s.surface_id == int(sid.iloc[0])]
        fa = tuple(x.iloc[0][["a_b", "a_c", "a_d", "a_e"]].astype(float))
        fb = tuple(x.iloc[0][["b_b", "b_c", "b_d", "b_e"]].astype(float))
        sa, sb = set(np.round(x.dose_a, 6)), set(np.round(x.dose_b, 6))
        if sa == set(np.round(c["doses_a"][1:], 6)) and sb == set(np.round(c["doses_b"][1:], 6)):
            c["fit_a"], c["fit_b"] = fa, fb
        else:
            assert sb == set(np.round(c["doses_a"][1:], 6)) and sa == set(np.round(c["doses_b"][1:], 6))
            c["fit_a"], c["fit_b"] = fb, fa
        c["surface_id"] = int(sid.iloc[0])
        c["surface"] = x
    return cases


def nice(name):
    return {"5FU": "5-FU"}.get(name, name)


def dose_unit(doses):
    """Axis unit for one drug: nM when every tested dose is below 0.1 uM."""
    return ("nM", 1000.0) if max(doses) < 0.1 else ("$\\mu$M", 1.0)


def dose_label(v, scale=1.0):
    if v == 0:
        return "0"
    x = v * scale
    return f"{x:.3g}" if x >= 0.01 else f"{x:.2g}"


def cell_text(v, fmt):
    s = fmt.format(v)
    if float(s) == 0:
        s = fmt.format(0.0)
    return S.minus(s)


def heat(ax, M, doses_a, doses_b, cmap, norm, mask_edges=False, fmt="{:.0f}", fontsize=5.4):
    A = np.array(M, dtype=float)
    shown = A.copy()
    if mask_edges:
        shown[0, :] = np.nan
        shown[:, 0] = np.nan
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#f0f0f0")
    ax.imshow(np.ma.masked_invalid(shown), origin="lower", cmap=cm, norm=norm, aspect="equal")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isnan(shown[i, j]):
                continue
            rgba = cm(norm(shown[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(j, i, cell_text(shown[i, j], fmt), ha="center", va="center", fontsize=fontsize,
                    color="white" if lum < 0.5 else S.INK)
    ax.axhline(0.5, color="white", lw=1.4)
    ax.axvline(0.5, color="white", lw=1.4)
    ua, sa = dose_unit(doses_a[1:])
    ub, sb = dose_unit(doses_b[1:])
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels([dose_label(v, sb) for v in doses_b], rotation=90)
    ax.set_yticklabels([dose_label(v, sa) for v in doses_a])
    ax.tick_params(length=0, pad=1.2, labelsize=5.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return ua, ub


def fig1(inp, path):
    cases, n_all, n_one = select_cases(inp)
    cases = attach_fits(inp, load_matrices(inp, cases))
    inh = Normalize(*S.INHIBITION_RANGE)
    lim = max(np.nanmax(np.abs(c["R"][1:, 1:])) for c in cases)
    lim = float(np.ceil(lim / 10) * 10)
    res = TwoSlopeNorm(0, -lim, lim)
    W, H = S.WIDTH, 6.35
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(3, 5, width_ratios=[1.25, 1, 1, 1, 0.92], left=0.075, right=0.99, bottom=0.125,
                          top=0.895, wspace=0.4, hspace=0.62)
    heads = ["Single agents", r"Measured $\bar y_u$", "Bliss reference $E_u$", r"Residual $\bar y_u - E_u$",
             "Pair-cell target"]
    for r, c in enumerate(cases[::-1]):
        A, B = nice(c["drug_a"]), nice(c["drug_b"])
        ax = fig.add_subplot(gs[r, 0])
        for fit, doses, vals, solid, mk, name in [(c["fit_a"], c["doses_a"], c["Y"][:, 0], True, "o", A),
                                                  (c["fit_b"], c["doses_b"], c["Y"][0, :], False, "s", B)]:
            xs = np.logspace(np.log10(doses[1] / 4), np.log10(doses[-1] * 4), 200)
            col = S.INK if solid else S.MUTED
            fy = ll4(xs, *fit)
            ax.plot(xs, np.where(fy > 108, np.nan, np.clip(fy, -20, None)), color=col, lw=0.9, ls="-" if solid else "--", zorder=2)
            ax.scatter(doses[1:], vals[1:], s=10, marker=mk, color=col if solid else "white", edgecolor=col,
                       linewidth=0.8, zorder=3, label=name)
        ax.set_xscale("log")
        ax.set_ylim(-10, 137)  # headroom above 100% keeps the legend clear of every curve
        ax.spines["left"].set_bounds(-10, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Inhibition (%)")
        ax.set_xlabel("Dose ($\\mu$M)")
        ax.axhline(0, color=S.RULE, lw=0.5, zorder=1)
        ax.legend(loc="upper left", fontsize=5.8, handletextpad=0.2, borderaxespad=0.1, labelspacing=0.25)
        axes = [fig.add_subplot(gs[r, k]) for k in (1, 2, 3)]
        heat(axes[0], c["Y"], c["doses_a"], c["doses_b"], S.INHIBITION, inh)
        heat(axes[1], c["E"], c["doses_a"], c["doses_b"], S.INHIBITION, inh)
        ua, ub = heat(axes[2], c["R"], c["doses_a"], c["doses_b"], S.RESIDUAL, res, mask_edges=True)
        for ax2 in axes:
            ax2.set_xlabel(f"{B} ({ub})", fontsize=6.2, labelpad=1.5)
        axes[0].set_ylabel(f"{A} ({ua})", fontsize=6.2, labelpad=1.5)
        for ax2 in axes[1:]:
            ax2.set_yticklabels([])
        # pair-cell target: residual mean, additive drug effects of the panel, contrast
        m = float(np.mean(c["R"][1:, 1:]))
        t = c["t_bliss"]
        axs = fig.add_subplot(gs[r, 4])
        vals = [("$m_{pc}$", m, S.MUTED), ("drug\neffects", m - t, S.UNCERT), ("$t_{pc}$", t, S.INK)]
        for k, (nm, v, col) in enumerate(vals):
            axs.bar(k, v, color=col, width=0.62)
            axs.text(k, v + (0.7 if v >= 0 else -0.7), cell_text(v, "{:.1f}"), ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=5.8)
        axs.set_xticks(range(3))
        axs.set_xticklabels([v[0] for v in vals], fontsize=6)
        axs.tick_params(axis="x", length=0, pad=2)
        axs.axhline(0, color=S.MUTED, lw=0.6)
        axs.set_ylim(-9, 13)
        axs.set_yticks([-5, 0, 5, 10])
        axs.set_ylabel("Points", fontsize=6.2, labelpad=1)
        axs.spines["bottom"].set_visible(False)
        pct = PCT_NAME[c["pct"]]
        y_title = max(a_.get_position().y1 for a_ in [ax] + axes) + 0.02
        t1 = fig.text(0.012, y_title, f"{pct.capitalize()} of $t_{{pc}}$:", fontsize=7, ha="left", va="bottom",
                      fontfamily=S.BOLD)
        fig.canvas.draw()
        x1 = fig.transFigure.inverted().transform(t1.get_window_extent().get_points())[1, 0]
        fig.text(x1 + 0.008, y_title, f"{A} + {B} in {c['cell_line']}", fontsize=7, ha="left", va="bottom")
        if r == 0:
            for k, (ax2, ttl) in enumerate(zip([ax] + axes + [axs], heads)):
                pos = ax2.get_position()
                x = pos.x0 if k else pos.x0 - 0.045
                fig.text(x, 0.952, "abcde"[k], fontsize=9, fontfamily=S.BOLD, va="bottom")
                fig.text(x + 0.024, 0.952, ttl, fontsize=7.2, va="bottom")
    fig.add_artist(plt.Line2D([0.012, 0.99], [0.942, 0.942], color=S.RULE, lw=0.5))
    p1, p2, p3 = (fig.axes[k].get_position() for k in (1, 2, 3))
    cax = fig.add_axes([p1.x0, 0.04, p2.x1 - p1.x0, 0.011])
    cb = fig.colorbar(plt.cm.ScalarMappable(inh, S.INHIBITION), cax=cax, orientation="horizontal")
    cb.set_label("Inhibition (%), columns b and c", fontsize=6.2, labelpad=1.5)
    cax2 = fig.add_axes([p3.x0, 0.04, p3.width, 0.011])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(res, S.RESIDUAL), cax=cax2, orientation="horizontal")
    cb2.set_label("Residual (points)", fontsize=6.2, labelpad=1.5)
    for c_ in (cb, cb2):
        c_.outline.set_linewidth(0.4)
        c_.ax.tick_params(labelsize=5.8, length=2)
    save(fig, path)
    return {"selection": "pair-cells measured in one batch (n = %d of %d) with Bliss pair-cell contrast closest to the "
                         "5th, 50th and 95th percentiles of all values" % (n_one, n_all),
            "cases": [{k: c[k] for k in ("pct", "quantile", "drug_a", "drug_b", "cell_line", "t_bliss", "surface_id")}
                      | {"m_pc": float(np.mean(c["R"][1:, 1:]))} for c in cases]}


def figS1(inp, path):
    cases, _, _ = select_cases(inp)
    c = load_matrices(inp, [cases[1]])[0]
    inh = Normalize(*S.INHIBITION_RANGE)
    A, B = nice(c["drug_a"]), nice(c["drug_b"])
    fig, axes = plt.subplots(1, 5, figsize=(S.WIDTH, 1.75), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.05],
                                                                         "wspace": 0.18})
    for k, (bid, M) in enumerate(sorted(c["reps"].items())):
        ua, ub = heat(axes[k], M, c["doses_a"], c["doses_b"], S.INHIBITION, inh, fontsize=5.2)
        axes[k].set_title(f"Replicate block {k + 1}", fontsize=6.8)
        axes[k].set_xlabel(f"{B} ({ub})", fontsize=6.2)
        if k:
            axes[k].set_yticklabels([])
    axes[0].set_ylabel(f"{A} ({ua})", fontsize=6.2)
    stack = np.stack(list(c["reps"].values()))
    sd = stack.std(axis=0, ddof=1)
    ax = axes[4]
    nrm = Normalize(0, float(np.ceil(sd.max())))
    heat(ax, sd, c["doses_a"], c["doses_b"], "Greys", nrm, fmt="{:.1f}", fontsize=5.0)
    ax.set_title("Replicate SD (points)", fontsize=6.8)
    ax.set_xlabel(f"{B} ({ub})", fontsize=6.2)
    ax.set_yticklabels([])
    save(fig, path)
    return {"case": {k: c[k] for k in ("drug_a", "drug_b", "cell_line", "t_bliss")},
            "edge_sd_max": float(max(sd[0, :].max(), sd[:, 0].max())), "combination_sd_median": float(np.median(sd[1:, 1:]))}


# ----------------------------------------------------------------------------------------------- Figure 2 (design)

def _grid(ax, colors, x0, y0, cw, ch, edge="white"):
    for i, row in enumerate(colors):
        for j, col in enumerate(row):
            ax.add_patch(Rectangle((x0 + j * cw, y0 - (i + 1) * ch), cw, ch, facecolor=col, edgecolor=edge, lw=0.5))


def _dot(ax, x, y, kind, r=0.042):
    if kind == 1:
        ax.add_patch(plt.Circle((x, y), r, facecolor=S.INK, edgecolor=S.INK, lw=0.8))
    elif kind == 2:
        ax.add_patch(plt.Circle((x, y), r, facecolor="white", edgecolor=S.INK, lw=0.8))
        ax.add_patch(Wedge((x, y), r, 90, 270, facecolor=S.INK, edgecolor=S.INK, lw=0.8))
    else:
        ax.plot([x - 0.03, x + 0.03], [y, y], color=S.UNCERT, lw=0.8)


def fig2(inp, path):
    H = 4.45
    fig = plt.figure(figsize=(S.WIDTH, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, S.WIDTH)
    ax.set_ylim(0, H)
    ax.axis("off")
    npair, ncell, cw = 10, 14, 0.115
    folds = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    k = 2
    tr, va, te = S.SPLIT["train"], S.SPLIT["validation"], S.SPLIT["test"]
    role = lambda f: te if f == k else va if f == (k + 1) % 5 else tr
    top = 4.0

    def head(x, letter, text, y):
        ax.text(x, y, letter, fontsize=9, fontfamily=S.BOLD, va="baseline")
        ax.text(x + 0.17, y, text, fontsize=7.5, va="baseline")

    # (a) held-out-pair split
    x0 = 0.42
    head(0.02, "a", "Held-out pairs (the benchmark)", 4.3)
    _grid(ax, [[role(folds[p])] * ncell for p in range(npair)], x0, top, cw, cw)
    ax.text(x0 - 0.07, top - npair * cw / 2, "Drug pairs", rotation=90, ha="right", va="center", fontsize=6.5)
    ax.text(x0 + ncell * cw / 2, top + 0.05, "Cell lines", ha="center", va="bottom", fontsize=6.5)
    ax.text(x0, top - npair * cw - 0.1, "Whole pairs are held out. Five published pair folds:\n"
            "test fold $k$, validation fold $k{+}1$, training on the\nother three. "
            "A test pair has no outcome in training.", fontsize=6.2, va="top", linespacing=1.4)
    # (b) pair-cell split
    x1 = 3.05
    head(2.65, "b", "Pair seen in other cell lines (B1$\\endash$B3)", 4.3)
    _grid(ax, [[role((j + p) % 5) for j in range(ncell)] for p in range(npair)], x1, top, cw, cw)
    pr = 3
    tj = [j for j in range(ncell) if (j + pr) % 5 == k][1]
    for j in range(ncell):
        if (j + pr) % 5 not in (k, (k + 1) % 5):
            ax.add_patch(Rectangle((x1 + j * cw + 0.017, top - (pr + 1) * cw + 0.017), cw - 0.034, cw - 0.034,
                                   facecolor="none", edgecolor=S.B2, lw=0.9))
    ax.add_patch(Rectangle((x1 + tj * cw, top - (pr + 1) * cw), cw, cw, facecolor="none", edgecolor=S.NATIVE, lw=1.4))
    ax.annotate("anchors", (x1 + ncell * cw, top - (pr + 0.5) * cw),
                (x1 + ncell * cw + 0.12, top - (pr + 0.5) * cw), color=S.B2, fontsize=6.0, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=S.B2))
    ax.annotate("evaluated\npair-cell", (x1 + (tj + 0.5) * cw, top), (x1 + (tj + 0.5) * cw, top + 0.08),
                color=S.NATIVE, fontsize=6.0, ha="center", va="bottom")
    ax.text(x1, top - npair * cw - 0.1, "Each pair's cell lines are dealt round-robin to five\n"
            "folds. The evaluated pair is measured in about 23\ntraining cell lines (anchors).", fontsize=6.2,
            va="top", linespacing=1.4)
    lx, ly = 5.33, 3.62
    ax.text(lx, ly + 0.2, "Fold role", fontsize=6.3, va="bottom")
    for i, (name, col) in enumerate([("training", tr), ("validation", va), ("test", te)]):
        ax.add_patch(Rectangle((lx, ly - i * 0.19), 0.12, 0.12, facecolor=col, edgecolor="white"))
        ax.text(lx + 0.18, ly - i * 0.19 + 0.06, name, va="center", fontsize=6.2)
    ax.text(lx, ly - 0.62, "Schematic: 10 of 583\npairs, 14 of 39 cell\nlines; $k$ = 3", fontsize=5.8, color=S.MUTED,
            va="top", linespacing=1.35)
    ax.plot([0.02, S.WIDTH - 0.02], [2.4, 2.4], color=S.RULE, lw=0.5)
    # (c) information matrix
    head(0.02, "c", "Information used to predict pair $p$ in cell line $c$", 2.17)
    cols_c = ["Outcomes of\nother pairs", "Single-agent\nresponses\n(direct input)", "Same pair,\nother cell lines"]
    rows_c = [("Bliss reference (no fitting)", S.INK, [0, 1, 0]),
              ("F0, GP-F0, ridge, DeepSynergy", S.HELDOUT, [1, 0, 0]),
              ("F2 and GP-F2", S.HELDOUT, [1, 1, 0]),
              ("Dose-aware CF", S.HELDOUT, [1, 0, 0]),
              ("B1 pair offset", S.B1, [1, 0, 1]),
              ("B2 cross-cell linear predictor", S.B2, [1, 0, 1]),
              ("B3 pair-seen F2 network", S.B3, [1, 1, 1])]
    cx0, cy0, dx, dy = 2.22, 1.72, 0.58, 0.205
    for j, t in enumerate(cols_c):
        ax.text(cx0 + j * dx, cy0 + 0.04, t, ha="center", va="bottom", fontsize=5.9, linespacing=1.2)
    for i, (name, col, flags) in enumerate(rows_c):
        y = cy0 - (i + 0.5) * dy - (0.05 if i else 0)
        ax.add_patch(Rectangle((0.2, y - 0.04), 0.08, 0.08, facecolor=col, edgecolor="none"))
        ax.text(0.34, y, name, va="center", fontsize=6.1, linespacing=1.2)
        for j, f in enumerate(flags):
            _dot(ax, cx0 + j * dx, y, f)
        if i == 0:
            ax.plot([0.2, cx0 + 2 * dx + 0.28], [y - dy / 2 - 0.025] * 2, color=S.RULE, lw=0.4)
    yl = 0.14
    _dot(ax, 0.24, yl, 1, r=0.03)
    ax.text(0.31, yl, "used", va="center", fontsize=5.8, color=S.MUTED)
    _dot(ax, 0.68, yl, 0, r=0.03)
    ax.text(0.75, yl, "not used; evaluated pair-cell outcomes withheld", va="center", fontsize=5.8,
            color=S.MUTED)
    # (d) targets and metrics
    head(4.02, "d", "Interaction target and metrics", 2.17)
    steps = [(r"Replicate mean $\bar y_u$ at dose unit $u$", "$\\rightarrow$ response $R^2$ over all dose rows"),
             (r"Residual $\bar y_u - E_u$;  Bliss $E_u = a_u + b_u - a_u b_u/100$", None),
             (r"Pair-cell mean $m_{pc}$ over measured dose units", None),
             ("Contrast $t_{pc}$: $m_{pc}$ minus additive drug\neffects fitted in its (fold, cell line) panel",
              r"$\rightarrow$ contrast $R^2 = 1 - \sum w\,(t - \hat t)^2 / \sum w\,t^2$")]
    bx, bw, y = 4.05, 2.22, 2.0
    for i, (txt, metric) in enumerate(steps):
        h = 0.36 if "\n" in txt else 0.24
        ax.add_patch(FancyBboxPatch((bx, y - h), bw, h, boxstyle="round,pad=0,rounding_size=0.025",
                                    facecolor="#fafafa", edgecolor=S.MUTED, lw=0.5))
        ax.text(bx + 0.06, y - h / 2, txt, va="center", fontsize=5.8, linespacing=1.3)
        y -= h
        gap = 0.22 if i < 3 else 0
        if metric:
            ax.text(bx + 0.24, y - 0.11, metric, va="center", fontsize=5.8, color=S.B2)
        if i < len(steps) - 1:
            ax.annotate("", (bx + 0.11, y - gap + 0.015), (bx + 0.11, y - 0.015),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=S.MUTED, mutation_scale=6,
                                        shrinkA=0, shrinkB=0))
        y -= gap
    save(fig, path)
    return {"type": "schematic; fold assignments are illustrative, not the published folds"}
