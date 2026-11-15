"""Figures 6, 7, 8, S4 and S5: reference-driven labels, the Loewe root and screening consequences."""
import hashlib
import json

import mpmath as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle

import figstyle as S
import fig_measure as FM

WORKED_SURFACE = 9458
LABELS_SHA = "4ef40678fff51faa84fa4de63344c5b51e2e4abe7c28a417af24ec2b50fa135a"


def save(fig, path):
    S.save(fig, path)


def minus(s):
    """Minus signs for negative numbers; hyphens in words are kept."""
    return S.minus(s)


def ll4(x, b, c, d, e):
    return c + (d - c) / (1 + (x / e) ** b)


def box(ax, x, y, w, h, text, fc="#fafafa", ec=S.MUTED, fs=6.0, color=S.INK, ha="center", weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03", facecolor=fc, edgecolor=ec,
                                lw=0.6))
    tx = x + w / 2 if ha == "center" else x + 0.06
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color, linespacing=1.3,
            fontfamily=S.BOLD if weight == "bold" else None)


def arrow(ax, a, b, color=S.MUTED, lw=0.6):
    ax.annotate("", b, a, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=6, shrinkA=0,
                                          shrinkB=0))


# ------------------------------------------------------------------------------------------------- Figure 6

def gate2_contrasts(inp):
    """Projected contrasts of the observed Loewe label and of the zero-parameter term M (Gate 2 convention)."""
    lab = inp["labels"]
    assert hashlib.sha256(lab.read_bytes()).hexdigest() == LABELS_SHA
    df = pd.read_csv(lab)
    df["a"] = df[["drug_a_name", "drug_b_name"]].min(axis=1)
    df["b"] = df[["drug_a_name", "drug_b_name"]].max(axis=1)
    g = df.groupby(["a", "b", "cell_line"], sort=True).agg(y=("synergy", "mean"), fold=("fold", "first")).reset_index()
    drugs = sorted(set(g.a) | set(g.b))
    D = {x: i for i, x in enumerate(drugs)}
    wide = g.pivot(index=["a", "b"], columns="cell_line", values="y")
    pairs = list(wide.index)
    H = np.eye(len(D))[[D[a] for a, b in pairs]] + np.eye(len(D))[[D[b] for a, b in pairs]]
    folds = g.groupby(["a", "b"]).fold.first().reindex(wide.index).to_numpy()
    with np.load(inp["gate2_units"]) as z:
        U = {k: z[k] for k in ("Y_L", "Y_B", "R_B", "R_L_DC", "M_DC")}
    xs, ys = [], []
    for f in range(5):
        te = np.flatnonzero(folds == f)
        h = H[te]
        proj = lambda Z: Z - h @ np.linalg.lstsq(h, Z, rcond=None)[0]
        xs.append(proj(U["M_DC"][te]).ravel())
        ys.append(proj(U["Y_L"][te]).ravel())
    x, y = np.concatenate(xs), np.concatenate(ys)
    return x, y, float(1 - np.sum((y - x) ** 2) / np.sum(y ** 2))


def fig6(inp, path):
    g = inp.json("gate2")
    cases, _, _ = FM.select_cases(inp)
    c = FM.attach_fits(inp, FM.load_matrices(inp, [cases[2]]))[0]
    sf = c["surface"]
    # native Loewe reference on the 4 x 4 combination grid, oriented to drug_a rows / drug_b columns
    RL = np.full((4, 4), np.nan)
    FB = np.zeros((4, 4), bool)
    da, db = np.round(c["doses_a"][1:], 6), np.round(c["doses_b"][1:], 6)
    swap = c["fit_a"] != tuple(sf.iloc[0][["a_b", "a_c", "a_d", "a_e"]].astype(float))
    for _, r in sf.iterrows():
        xa, xb = (r.dose_b, r.dose_a) if swap else (r.dose_a, r.dose_b)
        i, j = int(np.flatnonzero(da == round(xa, 6))[0]), int(np.flatnonzero(db == round(xb, 6))[0])
        RL[i, j] = r.native_below_3
        FB[i, j] = r.native_code >= 3
    E = c["E"][1:, 1:]
    W, H = S.WIDTH, 4.55
    fig = plt.figure(figsize=(W, H))

    def ax_(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    def head(x, y, letter, text):
        fig.text(x / W, y / H, letter, fontsize=9, fontfamily=S.BOLD, va="bottom")
        fig.text((x + 0.19) / W, y / H, text, fontsize=7, va="bottom")

    # (a) references of one measured block
    head(0.03, 4.3, "a", f"Two references for the same block: {FM.nice(c['drug_a'])} + {FM.nice(c['drug_b'])} in "
                          f"{c['cell_line']} (Figure 1, top row)")
    inh = Normalize(*S.INHIBITION_RANGE)
    diff = E - RL
    dl = float(np.ceil(np.nanmax(np.abs(diff)) / 5) * 5)
    dn = TwoSlopeNorm(0, -dl, dl)
    titles = ["Bliss $E_u$", "Loewe, native $R_u$", "$E_u - R_u$ = Loewe label $-$ Bliss label"]
    for k, (M, cmap, nrm) in enumerate([(E, S.INHIBITION, inh), (RL, S.INHIBITION, inh), (diff, S.RESIDUAL, dn)]):
        ax = ax_(0.55 + k * 1.28, 2.85, 1.05, 1.05)
        full = np.full((5, 5), np.nan)
        full[1:, 1:] = M
        FM.heat(ax, full, c["doses_a"], c["doses_b"], cmap, nrm, mask_edges=True, fontsize=5.4)
        ax.set_xlim(0.5, 4.5)
        ax.set_ylim(0.5, 4.5)
        ua, ub = FM.dose_unit(c["doses_a"][1:]), FM.dose_unit(c["doses_b"][1:])
        ax.set_xlabel(f"{FM.nice(c['drug_b'])} ({ub[0]})", fontsize=6.0, labelpad=1)
        if k == 0:
            ax.set_ylabel(f"{FM.nice(c['drug_a'])} ({ua[0]})", fontsize=6.0, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title(titles[k], fontsize=6.4, pad=3)
        if k == 1:
            for i in range(4):
                for j in range(4):
                    if FB[i, j]:
                        ax.add_patch(Rectangle((j + 1 - 0.5, i + 1 - 0.5), 1, 1, facecolor="none", edgecolor=S.NATIVE,
                                               lw=0.9))
    ax = ax_(4.45, 2.85, 1.8, 1.05)
    ax.axis("off")
    ax.text(0, 0.98, f"Outlined: solver rejected, summed-dose\nfallback used ({int(FB.sum())} of 16 cells).\n\n"
            "The difference between the two labels\ndoes not involve the measured\ncombination: it is fixed by the\n"
            "single agents and the solver.", fontsize=6.0, va="top", linespacing=1.35)
    # (b) zero-interaction screen
    head(0.03, 2.3, "b", "Zero-interaction screen (schematic)")
    axd = ax_(0.03, 0.05, 2.2, 2.2)
    axd.set_xlim(0, 2.2)
    axd.set_ylim(0, 2.2)
    axd.axis("off")
    box(axd, 0.1, 1.72, 2.0, 0.36, "Responses = reference + replicate noise\n(no interaction by construction)",
        ec=S.CONTROL)
    box(axd, 0.1, 1.0, 0.92, 0.38, "Score with\nnative Loewe", ec=S.NATIVE)
    box(axd, 1.18, 1.0, 0.92, 0.38, "Score with\nBliss", ec=S.INK)
    arrow(axd, (0.56, 1.72), (0.56, 1.38))
    arrow(axd, (1.64, 1.72), (1.64, 1.38))
    box(axd, 0.1, 0.28, 2.0, 0.38, "Contrast of each label; CF and kernel ridge\ntrained on held-out pairs")
    arrow(axd, (0.56, 1.0), (0.56, 0.66))
    arrow(axd, (1.64, 1.0), (1.64, 0.66))
    axd.text(1.1, 0.12, "Any skill comes from the reference alone", ha="center", fontsize=5.9, color=S.MUTED)
    # (c) learnability
    head(2.5, 2.3, "c", "Contrast $R^2$ on held-out pairs")
    ax = ax_(3.42, 0.35, 1.1, 1.8)
    p4, p5 = g["P4"]["runs"], g["P5"]
    rows = [("Loewe, observed", p5["Y_L"]["naive"]["knn"]["cr2"], p5["Y_L"]["naive"]["krr"]["cr2"], S.NATIVE),
            ("Loewe, zero\ninteraction", p4["cell_independent:loewe_sim"]["knn"]["cr2"],
             p4["cell_independent:loewe_sim"]["krr"]["cr2"], S.NATIVE),
            ("Bliss, observed", p5["Y_B"]["naive"]["knn"]["cr2"], p5["Y_B"]["naive"]["krr"]["cr2"], S.INK),
            ("Bliss, zero\ninteraction", p4["cell_independent:bliss_sim"]["knn"]["cr2"],
             p4["cell_independent:bliss_sim"]["krr"]["cr2"], S.INK)]
    for i, (name, knn, krr, col) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([knn, krr], [y + 0.12, y - 0.12], color=S.RULE, lw=0.8, zorder=1)
        ax.scatter(knn, y + 0.12, color=col, marker="o", s=16, zorder=3, edgecolor="white", lw=0.4,
                   label="neighbourhood CF" if i == 0 else None)
        ax.scatter(krr, y - 0.12, color="white", edgecolor=col, marker="D", s=13, zorder=3, lw=0.8,
                   label="kernel ridge" if i == 0 else None)
    ax.axvline(0, color=S.MUTED, lw=0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows[::-1]], fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.15, 0.62)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlabel("$R^2$", labelpad=1)
    leg = ax.legend(loc="lower right", fontsize=5.6, handletextpad=0.1, borderaxespad=0.0)
    for h_ in leg.legend_handles:
        h_.set_color(S.MUTED)
        h_.set_edgecolor(S.MUTED)
    # (d) zero-parameter term against the Loewe label
    x, y, r2 = gate2_contrasts(inp)
    head(4.62, 2.3, "d", "A term with no fitted parameters")
    ax = ax_(4.95, 0.35, 1.3, 1.3)
    lim = float(np.percentile(np.abs(np.r_[x, y]), 99.5))
    ax.plot([-lim, lim], [-lim, lim], color=S.RULE, lw=0.6)
    ax.scatter(x, y, s=1.2, color=S.NATIVE, alpha=0.25, lw=0, rasterized=True)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Contrast of $M = E - R$", labelpad=1)
    ax.set_ylabel("Contrast of Loewe label", labelpad=1)
    ax.text(-lim * 0.95, lim * 0.95, f"$R^2$ = {r2:.3f}", fontsize=6.2, va="top")
    comp = g["components"]
    fig.text(4.66 / W, 2.18 / H, minus(f"$R^2$ of its parts: implementation {comp['M_impl']['cr2_vs_Y_L']:.3f},\n"
                                       f"existence rule {comp['M_exist']['cr2_vs_Y_L']:.3f}, exact algebra "
                                       f"{comp['M_alg']['cr2_vs_Y_L']:.3f}"), fontsize=5.7, va="top", linespacing=1.35,
             color=S.MUTED)
    save(fig, path)
    return {"case": {k: c[k] for k in ("drug_a", "drug_b", "cell_line", "surface_id")},
            "fallback_cells": int(FB.sum()), "M_contrast_r2_recomputed": r2,
            "M_contrast_r2_recorded": comp["M_DC"]["cr2_vs_Y_L"]}


# ------------------------------------------------------------------------------------------------- Figure 7

class Worked:
    """Exact Loewe quantities for the worked surface (arm i: weaker upper plateau)."""

    def __init__(self, surfaces):
        s = surfaces[surfaces.surface_id == WORKED_SURFACE].copy()
        self.cases = s
        A = tuple(s.iloc[0][["a_b", "a_c", "a_d", "a_e"]].astype(float))
        B = tuple(s.iloc[0][["b_b", "b_c", "b_d", "b_e"]].astype(float))
        assert max(A[1], A[2]) < max(B[1], B[2])
        self.Pi, self.Pj = A, B             # i = dasatinib (dose_a), j = MK-2206 (dose_b)
        self.hi = max(A[1], A[2])
        self.lo = max(min(A[1], A[2]), min(B[1], B[2]))
        mp.mp.dps = 80
        self.mi = [mp.mpf(v) for v in A]
        self.mj = [mp.mpf(v) for v in B]
        self.mhi = max(self.mi[1], self.mi[2])
        self.xstar = float(self.D(self.mhi, self.mj))

    @staticmethod
    def D(y, p):
        b, c, d, e = p
        return e * ((d - y) / (y - c)) ** (1 / b)

    def g(self, xi, xj, u):
        y = self.mhi - mp.mpf(u)
        return mp.mpf(xi) / self.D(y, self.mi) + mp.mpf(xj) / self.D(y, self.mj) - 1

    def root_gap(self, xi, xj):
        """h_i - y* by bisection in log(u); None if no root."""
        if xj >= self.xstar:
            return None
        lo, hi = mp.mpf(-200), mp.log(self.mhi - mp.mpf(self.lo)) - mp.mpf("1e-30")
        for _ in range(400):
            mid = (lo + hi) / 2
            if self.g(xi, xj, mp.e ** mid) < 0:   # u too small: still past the root
                lo = mid
            else:
                hi = mid
        return mp.e ** ((lo + hi) / 2)

    def law(self, xi, xj):
        b, c, d, e = self.Pi
        delta = 1 - xj / self.xstar
        return abs(d - c) * (delta * e / xi) ** abs(b)


def fig7(inp, path):
    s_all = pd.read_parquet(inp["surfaces"])
    w = Worked(s_all)
    rr = pd.read_csv(inp["rejected_roots"])
    bl = pd.read_csv(inp["boundary_layer"])
    la = inp.json("loewe_audit")
    obs = pd.read_parquet(inp["observations"], columns=["drug_a", "drug_b", "cell_line", "dose_a", "dose_b", "inhibition"])
    ob = obs[(obs.drug_a == "DASATINIB") & (obs.drug_b == "MK2206") & (obs.cell_line == "MSTO")]
    sa = ob[ob.dose_b == 0].groupby("dose_a").inhibition.mean()
    sb = ob[ob.dose_a == 0].groupby("dose_b").inhibition.mean()
    cases = w.cases.merge(rr[["case_id", "C1", "C2"]], on="case_id", how="left")
    cases["C1"] = cases["C1"].astype("boolean").fillna(False).astype(bool)
    xi_slice = 3.0
    W, H = S.WIDTH, 6.35
    fig = plt.figure(figsize=(W, H))

    def ax_(x, y, wd, ht):
        return fig.add_axes([x / W, y / H, wd / W, ht / H])

    def head(x, y, letter, text):
        fig.text(x / W, y / H, letter, fontsize=9, fontfamily=S.BOLD, va="bottom")
        fig.text((x + 0.19) / W, y / H, text, fontsize=7, va="bottom")

    fig.text(0.03 / W, 6.22 / H, "Worked surface: dasatinib (arm $i$, weaker plateau) and MK-2206 (arm $j$) in MSTO; "
             "native LL.4 fits", fontsize=6.6, color=S.MUTED, va="bottom")
    # (a) single agents and the existence threshold
    head(0.03, 5.94, "a", "Existence threshold")
    ax = ax_(0.42, 4.32, 1.55, 1.45)
    xs = np.logspace(-3, 1.6, 400)
    ax.plot(xs, ll4(xs, *w.Pi), color=S.INK, lw=1.0, label="dasatinib, $f_i$")
    ax.plot(xs, ll4(xs, *w.Pj), color=S.MUTED, lw=1.0, ls="--", label="MK-2206, $f_j$")
    ax.scatter(sa.index[1:], sa.values[1:], s=9, color=S.INK, zorder=3)
    ax.scatter(sb.index[1:], sb.values[1:], s=9, marker="s", color="white", edgecolor=S.MUTED, lw=0.8, zorder=3)
    ax.axhline(w.hi, color=S.CORRECTED, lw=0.7, ls=":")
    ax.text(1.2e-3, w.hi + 2, f"$h_i$ = {w.hi:.2f}", color=S.CORRECTED, fontsize=6, va="bottom")
    ax.axvline(w.xstar, color=S.CORRECTED, lw=0.7, ls=":")
    ax.axvspan(w.xstar, 40, color="#f3f3f3", lw=0, zorder=0)
    ax.text(w.xstar * 1.3, 97, "no\nroot", fontsize=6, color=S.MUTED, va="top")
    ax.text(w.xstar / 1.15, 4, "$x_j^* = D_j(h_i)$\n= " + f"{w.xstar:.2f} $\\mu$M", fontsize=5.8, color=S.CORRECTED,
            ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 40)
    ax.set_ylim(-5, 100)
    ax.set_xlabel("Dose ($\\mu$M)", labelpad=1)
    ax.set_ylabel("Inhibition (%)", labelpad=1)
    ax.legend(loc="upper left", fontsize=5.8, handlelength=1.8, borderaxespad=0.1)
    # (b) existence region and native outcomes on the grid
    head(2.2, 5.94, "b", "Native outcome on the dose grid")
    ax = ax_(2.55, 4.32, 1.55, 1.45)
    ax.axvspan(w.xstar, 30, color="#f3f3f3", lw=0, zorder=0)
    ax.axvline(w.xstar, color=S.CORRECTED, lw=0.7, ls=":")
    ax.axhline(xi_slice, color=S.RULE, lw=2.6, zorder=1)
    ax.text(26, xi_slice * 1.3, "slice in d", fontsize=5.6, color=S.MUTED, ha="right")
    kinds = {"accepted root": dict(marker="o", color=S.CORRECTED, edgecolor=S.CORRECTED),
             "rejected root (C1)": dict(marker="X", color=S.NATIVE, edgecolor=S.NATIVE),
             "no root (fallback)": dict(marker="s", color="white", edgecolor=S.MUTED)}
    for _, r in cases.iterrows():
        if r.C1:
            k = "rejected root (C1)"
        elif r.native_code in (1, 2):
            k = "accepted root"
        else:
            k = "no root (fallback)"
        ax.scatter(r.dose_b, r.dose_a, s=26, lw=0.8, zorder=3, **kinds[k])
    for k, kw in kinds.items():
        ax.scatter([], [], s=22, lw=0.8, label=k, **kw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.012, 30)
    ax.set_ylim(0.01, 400)
    ax.set_xlabel("MK-2206, $x_j$ ($\\mu$M)", labelpad=1)
    ax.set_ylabel("Dasatinib, $x_i$ ($\\mu$M)", labelpad=1)
    ax.legend(loc="upper left", fontsize=5.6, ncol=1, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.25)
    # (c) g(y) near the plateau
    head(4.32, 5.94, "c", "$g(y)$ near the plateau, $x_i$ = 3 $\\mu$M")
    ax = ax_(4.72, 4.32, 1.53, 1.45)
    us = np.logspace(-17, 1.5, 260)
    xjs = [(0.2, 0.25), (0.9, 0.45), (2.9, 0.7), (3.03, 0.92)]
    cmap = plt.get_cmap("Blues")
    ulp = float(np.spacing(w.hi))
    ax.axvspan(1e-18, ulp, color="#ececec", lw=0)
    for xj, shade in xjs:
        gv = np.array([float(w.g(xi_slice, xj, u)) for u in us])
        ax.plot(us, gv, color=cmap(shade), lw=1.0, label=f"$x_j$ = {xj:g}")
        u0 = float(w.root_gap(xi_slice, xj))
        ax.scatter(u0, 0, s=12, color=cmap(shade), zorder=3, edgecolor="white", lw=0.4)
    gv = np.array([float(w.g(xi_slice, 4.0, u)) for u in us])
    ax.plot(us, gv, color=S.MUTED, lw=1.0, ls="--", label="$x_j$ = 4 (no root)")
    ax.axhline(0, color=S.INK, lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1, linscale=1.2)
    ax.set_xlim(10 ** 1.5, 1e-18)
    ax.set_ylim(-1.05, 60)
    ax.set_yticks([-1, 0, 1, 10])
    ax.set_yticklabels([minus(-1), "0", "1", "10"])
    ax.set_xticks([1e0, 1e-5, 1e-10, 1e-15])
    ax.set_xlabel("$h_i - y$ (points)", labelpad=1)
    ax.set_ylabel("$g(y)$", labelpad=1)
    ax.legend(loc="upper right", fontsize=5.3, handlelength=1.3, borderaxespad=0.1, labelspacing=0.1,
              bbox_to_anchor=(0.86, 1.0))
    # (d) the reference along the slice
    head(0.03, 3.72, "d", "Reference along the slice $x_i$ = 3 $\\mu$M")
    ax = ax_(0.42, 2.12, 3.55, 1.4)
    xj = np.logspace(-2, np.log10(25), 500)
    gap = np.array([float(w.root_gap(xi_slice, v)) if v < w.xstar else np.nan for v in xj])
    root = w.hi - gap
    ext = np.where(np.isnan(root), ll4(xj, *w.Pj), root)
    sd = np.minimum(100, np.maximum(ll4(xi_slice + xj, *w.Pi), ll4(xi_slice + xj, *w.Pj)))
    left = xj < w.xstar
    ax.plot(xj[left], ext[left], color=S.CORRECTED, lw=1.4, label="certified root $y^*$")
    ax.plot(xj[~left], ext[~left], color=S.CORRECTED, lw=1.4, ls=(0, (4, 1.5)), label="continuous extension $f_j(x_j)$ (= HSA)")
    ax.plot(xj, sd, color=S.NATIVE, lw=1.2, ls="--", label="summed-dose fallback")
    grid = cases[cases.dose_a == xi_slice].sort_values("dose_b")
    ax.scatter(grid.dose_b, grid.native_below_3, s=22, color=S.NATIVE, zorder=4, edgecolor="white", lw=0.5,
               label="native reference at grid doses")
    gr = grid[grid.dose_b < w.xstar]
    ax.scatter(gr.dose_b, [w.hi - float(w.root_gap(xi_slice, v)) for v in gr.dose_b], s=20, facecolor="white",
               edgecolor=S.CORRECTED, lw=0.9, zorder=5, label="certified root at grid doses")
    jb = float(np.minimum(100, max(ll4(xi_slice + w.xstar, *w.Pi), ll4(xi_slice + w.xstar, *w.Pj))))
    ax.annotate("", (w.xstar, jb), (w.xstar, w.hi), arrowprops=dict(arrowstyle="<->", lw=0.7, color=S.INK,
                                                                     shrinkA=0, shrinkB=0))
    ax.text(w.xstar * 1.08, (jb + w.hi) / 2, f"jump $J$ = {jb - w.hi:.1f}", fontsize=6.2, va="center")
    ax.axvline(w.xstar, color=S.CORRECTED, lw=0.6, ls=":")
    ax.axhline(w.hi, color=S.CORRECTED, lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 25)
    ax.set_ylim(56, 84)
    ax.set_xlabel("MK-2206, $x_j$ ($\\mu$M)", labelpad=1)
    ax.set_ylabel("Loewe reference (%)", labelpad=1)
    ax.legend(loc="upper left", fontsize=5.8, borderaxespad=0.1, handlelength=2.2)
    # (e) boundary layer along the slice
    head(4.32, 3.72, "e", "Boundary layer (Proposition 3)")
    ax = ax_(4.72, 2.12, 1.53, 1.4)
    deltas = np.logspace(-6, np.log10(0.99), 60)
    gaps = np.array([float(w.root_gap(xi_slice, (1 - d) * w.xstar)) for d in deltas])
    law = np.array([w.law(xi_slice, (1 - d) * w.xstar) for d in deltas])
    ax.axhspan(1e-80, ulp, color="#ececec", lw=0)
    ax.plot(deltas, gaps, color=S.CORRECTED, lw=1.4, label="certified $h_i - y^*$")
    ax.plot(deltas, law, color=S.INK, lw=0.8, ls=(0, (2, 1.5)), label=r"$|d_i - c_i|\,(\delta e_i/x_i)^{|b_i|}$")
    for d0 in (1 - grid.dose_b[grid.dose_b < w.xstar] / w.xstar):
        ax.scatter(d0, float(w.root_gap(xi_slice, (1 - d0) * w.xstar)), s=18, color=S.NATIVE, marker="X", zorder=4,
                   lw=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-6, 1.2)
    ax.set_ylim(1e-22, 1e-3)
    ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5])
    ax.set_xlabel(r"$\delta = 1 - x_j/x_j^*$", labelpad=1)
    ax.set_ylabel("$h_i - y^*$ (points)", labelpad=1)
    ax.text(2e-6, ulp * 3, "float64 spacing", fontsize=5.5, color=S.MUTED, va="bottom")
    ax.text(3e-3, 1e-12, f"slope $|b_i|$ = {abs(w.Pi[0]):.2f}", fontsize=5.9, rotation=0)
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.8, borderaxespad=0.1)
    # (f) the law across all near-plateau certified roots
    fig.add_artist(plt.Line2D([0.03 / W, 6.27 / W], [1.74 / H, 1.74 / H], color=S.RULE, lw=0.5))
    fig.text(0.03 / W, 1.6 / H, "All cases", fontsize=6.6, color=S.MUTED, va="bottom")
    head(0.03, 1.36, "f", f"{la['boundary_layer']['cases']:,} certified roots near a plateau")
    ax = ax_(0.42, 0.34, 1.55, 0.9)
    acc = ~bl.fallback.astype(bool)
    ax.scatter(bl.log10_predicted[~acc], bl.log10_certified[~acc], s=1.5, color=S.NATIVE, alpha=0.35, lw=0,
               rasterized=True, label=f"rejected (fallback), {int((~acc).sum()):,}")
    ax.scatter(bl.log10_predicted[acc], bl.log10_certified[acc], s=1.5, color=S.CORRECTED, alpha=0.6, lw=0,
               rasterized=True, label=f"accepted by solver, {int(acc.sum()):,}")
    ax.plot([-62, 0], [-62, 0], color=S.INK, lw=0.5)
    fs = np.log10(ulp)
    ax.axhline(fs, color=S.MUTED, lw=0.5, ls=":")
    ax.set_xlim(-62, 0)
    ax.set_ylim(-62, 0)
    ax.set_xticks([-60, -40, -20, 0])
    ax.set_yticks([-60, -40, -20, 0])
    ax.set_xlabel("Predicted $\\log_{10}(h_i - y^*)$", labelpad=1)
    ax.set_ylabel(r"Certified $\log_{10}(h_i - y^*)$", labelpad=1)
    ax.text(-60, -6, f"r = {la['boundary_layer']['loglog_pearson_r']:.3f}", fontsize=5.9, va="top")
    lg = ax.legend(loc="lower right", fontsize=5.5, markerscale=4, handletextpad=0.1, borderaxespad=0.1)
    for h_ in lg.legend_handles:
        h_.set_alpha(1)
    # (g) slopes of rejected roots
    head(2.2, 1.36, "g", "Steep weaker arm")
    ax = ax_(2.6, 0.34, 0.95, 0.9)
    sh = la["boundary_layer"]["share_abs_slope_above_2"]
    vals = [("all certified\nroots", sh["all_certified_roots_co_increasing_LL4"], S.CORRECTED),
            ("rejected\n(C1)", sh["C1_co_increasing_LL4"], S.NATIVE)]
    for k, (name, v, col) in enumerate(vals):
        ax.bar(k, 100 * v, color=col, width=0.6)
        ax.text(k, 100 * v + 2, f"{100 * v:.1f}%", ha="center", fontsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([v[0] for v in vals], fontsize=5.8)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Share with $|b_i| > 2$ (%)", labelpad=1)
    # (h) fallback error on C1
    head(3.95, 1.36, "h", f"Fallback error, {la['populations']['C1']:,} C1 roots")
    ax = ax_(4.4, 0.34, 1.85, 0.9)
    c1 = rr[rr.C1]
    fe = la["fallback_error_on_rejected_roots"]["C1_14197"]
    for col_, color, name, key in [("error_summed_dose", S.NATIVE, "summed dose", "summed_dose"),
                                   ("error_hsa", S.CORRECTED, "HSA (continuous)", "hsa")]:
        v = np.sort(np.maximum(c1[col_].to_numpy(), 1e-9))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=1.2,
                label=f"{name}; median {fe[key]['median']:.4f}" if key == "hsa" else f"{name}; median {fe[key]['median']:.2f}")
        ax.axvline(fe[key]["median"], color=color, lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 300)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([1e-8, 1e-4, 1e0])
    ax.set_xlabel("$|$reference $-$ certified root$|$ (points)", labelpad=1)
    ax.set_ylabel("Cumulative share", labelpad=1)
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.4, borderaxespad=0.1)
    save(fig, path)
    return {"surface": WORKED_SURFACE, "x_j_star_uM": w.xstar, "h_i": w.hi, "jump_at_boundary_slice": jb - w.hi,
            "selection": "illustrative surface: co-increasing LL.4 arms, weaker plateau below 100, at least four grid "
                         "cases each of accepted roots, C1 rejected roots and certified no-root cases"}


# ------------------------------------------------------------------------------------------------- Figure 8

def pc_name(s):
    pair, cell = s.split("|")
    a, b = json.loads(pair)
    return f"{a} + {b}, {cell}"


def fig8(inp, path):
    sc = inp.json("screening_audit")
    u = pd.read_csv(inp["pair_cell_scores"])
    top = sc["c1_top5"]
    cutN, cutC = top["native_cutoff"], top["c1_cutoff"]
    ent = u.selected_C1 & ~u.selected_N
    ext = u.selected_N & ~u.selected_C1
    kept = u.selected_N & u.selected_C1
    W, H = S.WIDTH, 5.0
    fig = plt.figure(figsize=(W, H))

    def ax_(x, y, wd, ht):
        return fig.add_axes([x / W, y / H, wd / W, ht / H])

    def head(x, y, letter, text):
        fig.text(x / W, y / H, letter, fontsize=9, fontfamily=S.BOLD, va="bottom")
        fig.text((x + 0.19) / W, y / H, text, fontsize=7, va="bottom")

    # (a) scores near the cutoff
    head(0.03, 4.75, "a", "Pair-cell Loewe scores near the top-5% cutoff")
    ax = ax_(0.45, 2.75, 1.75, 1.8)
    win = (u.score_N > cutN - 28) & (u.score_N < cutN + 14)
    other = win & ~ent & ~ext
    ax.scatter(u.score_N[other], u.score_C1[other], s=1.5, color=S.UNCERT, alpha=0.5, lw=0, rasterized=True,
               label="other pair-cells")
    ax.scatter(u.score_N[ext], u.score_C1[ext], s=7, color=S.NATIVE, lw=0, label=f"exits ({int(ext.sum())})")
    ax.scatter(u.score_N[ent], u.score_C1[ent], s=7, color=S.CORRECTED, lw=0, label=f"entrants ({int(ent.sum())})")
    ax.axvline(cutN, color=S.NATIVE, lw=0.6, ls=":")
    ax.axhline(cutC, color=S.CORRECTED, lw=0.6, ls=":")
    ax.plot([cutN - 28, cutN + 14], [cutN - 28, cutN + 14], color=S.RULE, lw=0.5)
    ax.set_xlim(cutN - 28, cutN + 14)
    ax.set_ylim(cutN - 28, cutN + 22)
    ax.set_xlabel("Native score", labelpad=1)
    ax.set_ylabel("Score with C1 roots restored", labelpad=1)
    ax.legend(loc="upper left", fontsize=5.7, markerscale=1.6, handletextpad=0.1, borderaxespad=0.1)
    # (b) entrant score shifts
    head(2.55, 4.75, "b", "The 67 entrants: native $\\rightarrow$ C1 score")
    ax = ax_(2.95, 2.75, 1.45, 1.8)
    e = u[ent].sort_values("score_C1").reset_index(drop=True)
    for i, r in e.iterrows():
        ax.plot([r.score_N, r.score_C1], [i, i], color=S.RULE, lw=0.6, zorder=1)
    ax.scatter(e.score_N, e.index, s=6, color=S.NATIVE, zorder=2, lw=0, label="native")
    ax.scatter(e.score_C1, e.index, s=6, color=S.CORRECTED, zorder=3, lw=0, label="C1")
    ax.axvline(cutN, color=S.NATIVE, lw=0.6, ls=":")
    ax.axvline(cutC, color=S.CORRECTED, lw=0.6, ls=":")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Pair-cell Loewe score", labelpad=1)
    ax.legend(loc="lower right", fontsize=5.7, markerscale=1.8, handletextpad=0.1, borderaxespad=0.0)
    xr = max(e.score_C1) + 2
    for k, i in enumerate(e.index[-3:]):
        yl = len(e) + 3.5 + 5.2 * k
        ax.annotate(pc_name(e.pair_cell[i]), (e.score_C1[i], i), (xr - 30, yl), fontsize=5.0, va="center",
                    ha="left", arrowprops=dict(arrowstyle="-", lw=0.4, color=S.MUTED, shrinkA=0, shrinkB=1.5),
                    bbox=dict(fc="white", ec="none", pad=0.4), zorder=5)
    ax.set_xlim(cutN - 27, max(e.score_C1) + 6)
    ax.set_ylim(-1, len(e) + 17)
    # (c) selection probabilities
    head(4.55, 4.75, "c", "Selection under 1,000 resamples")
    ax = ax_(4.95, 2.75, 1.3, 1.8)
    ax.plot([0, 1], [0, 1], color=S.RULE, lw=0.5)
    ax.scatter(u.p_select_N[kept], u.p_select_C1[kept], s=1.5, color=S.UNCERT, alpha=0.5, lw=0, rasterized=True)
    ax.scatter(u.p_select_N[ext], u.p_select_C1[ext], s=7, color=S.NATIVE, lw=0)
    ax.scatter(u.p_select_N[ent], u.p_select_C1[ent], s=7, color=S.CORRECTED, lw=0)
    sp = sc["replicate_resampling"]["selection_probability"]
    for k in ("entrants", "exits"):
        ax.scatter(sp[k]["N"], sp[k]["C1"], s=40, marker="+", color=S.INK, lw=1.0, zorder=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("P(selected), native", labelpad=1)
    ax.set_ylabel("P(selected), C1", labelpad=1)
    # (d) turnover by change at the 5% budget
    head(0.03, 2.05, "d", "Share of the top-5% list (1,137 pair-cells) replaced")
    ax = ax_(1.85, 0.4, 2.35, 1.45)
    b5 = sc["budget_0.05"]
    rs = sc["replicate_resampling"]
    n = b5.get("selected", 1137)
    bars = [("C1: restore 14,197 float64-verified roots", b5["N_to_C1_turnover"], S.CORRECTED, None),
            ("I: restore all 15,620 rejected roots", b5["N_to_I_turnover"], S.CORRECTED, "////"),
            ("H: HSA for all 117,824 no-root cases", b5["N_to_H_turnover"], "white", None),
            ("replicate bootstrap (native)", rs["noise_turnover_consecutive_bootstrap_N"]["mean"], S.UNCERT_FILL, None),
            ("half-depth replicates (native)", rs["noise_turnover_half_depth_N"]["mean"], S.UNCERT_FILL, None)]
    for i, (name, v, col, hatch) in enumerate(bars):
        y = len(bars) - 1 - i
        ax.barh(y, 100 * v, color=col, height=0.62, edgecolor=S.MUTED if col in ("white", S.UNCERT_FILL) else col,
                lw=0.5, hatch=hatch)
        lab = f"{100 * v:.2f}%" if i < 3 else f"{100 * v:.1f}%"
        if i < 3:
            lab += f"  ({round(v * n)} of {n:,})"
        ax.text(100 * v + 0.6, y, lab, va="center", fontsize=5.8)
    w5 = rs["within_resample_N_to_C1_turnover_bootstrap"]
    ax.plot([100 * w5["low"], 100 * w5["high"]], [len(bars) - 1 + 0.42] * 2, color=S.INK, lw=1.2)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([b[0] for b in bars[::-1]], fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 44)
    ax.set_xlabel("Turnover (%)", labelpad=1)
    # (e) by budget
    head(4.55, 2.05, "e", "C1 turnover by budget")
    ax = ax_(4.95, 0.4, 1.3, 1.45)
    bud = [(1, sc["budget_0.01"]), (2, sc["budget_0.02"]), (5, sc["budget_0.05"]), (10, sc["budget_0.1"]),
           (20, sc["budget_0.2"])]
    xb = [b for b, _ in bud]
    yb = [100 * v["N_to_C1_turnover"] for _, v in bud]
    ax.plot(xb, yb, color=S.CORRECTED, marker="o", ms=3, lw=1.0)
    for b, v in bud:
        ax.text(b, 100 * v["N_to_C1_turnover"] + 0.35, f"{v['selected']:,}", ha="center", fontsize=5.3,
                color=S.MUTED)
    ax.set_xscale("log")
    ax.set_xticks(xb)
    ax.set_xticklabels([f"{b}%" for b in xb])
    ax.minorticks_off()
    ax.set_ylim(0, 10)
    ax.set_xlabel("Budget (top share of pair-cells)", labelpad=1)
    ax.set_ylabel("Turnover (%)", labelpad=1)
    save(fig, path)
    return {"entrants": int(ent.sum()), "exits": int(ext.sum()),
            "labelled_entrants": [pc_name(x) for x in e.pair_cell[-3:]]}


# ------------------------------------------------------------------------------------------------- Figure S4

def figS4(inp, path):
    la = inp.json("loewe_audit")
    p = la["populations"]
    cs, sb = p["certificate_status"], p["status_by_fallback"]
    hc = p["historical_class"]
    W, H = S.WIDTH, 3.0
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    f = lambda n: f"{n:,}"
    box(ax, 2.15, 2.5, 2.0, 0.38, f"All Loewe cases: {f(p['cases'])}\non {f(p['surfaces'])} surfaces",
        weight="bold")
    lvl1 = [("Certified root", cs["certified_root"]), ("Certified no root", cs["certified_no_root"]),
            ("Undefined: opposite\nor flat arms", cs["undefined_opposite_or_flat"]),
            ("Undefined: empty\ncommon interval", cs["undefined_empty_interval"])]
    xs = [0.12, 1.72, 3.32, 4.92]
    for x, (name, n) in zip(xs, lvl1):
        box(ax, x, 1.62, 1.36, 0.42, f"{name}\n{f(n)}")
        arrow(ax, (3.15, 2.5), (x + 0.68, 2.04))
    lvl2 = [(0.12, f"accepted\n{f(sb['certified_root|accepted'])}", S.CORRECTED),
            (0.82, f"rejected\n{f(sb['certified_root|fallback'])}", S.NATIVE),
            (1.72, f"fallback\n{f(sb['certified_no_root|fallback'])}", S.MUTED),
            (3.32, f"accepted\n{f(sb['undefined_opposite_or_flat|accepted'])}", S.MUTED),
            (4.02, f"fallback\n{f(sb['undefined_opposite_or_flat|fallback'])}", S.MUTED),
            (4.92, f"fallback\n{f(sb['undefined_empty_interval|fallback'])}", S.MUTED)]
    for x, t, col in lvl2:
        wd = 0.66 if x not in (1.72, 4.92) else 1.36
        box(ax, x, 0.92, wd, 0.4, t, ec=col)
    for (x0, x1) in [(0.8, 0.45), (0.8, 1.15), (2.4, 2.4), (4.0, 3.65), (4.0, 4.35), (5.6, 5.6)]:
        arrow(ax, (x0, 1.62), (x1, 1.32))
    box(ax, 0.05, 0.12, 0.88, 0.48, f"C1: {f(p['C1'])}\nfloat64-verified", ec=S.CORRECTED)
    box(ax, 1.0, 0.12, 0.95, 0.48, f"C2: {f(p['C2_recovered'])}\n100-digit only", ec=S.CORRECTED)
    arrow(ax, (1.15, 0.92), (0.5, 0.6))
    arrow(ax, (1.15, 0.92), (1.47, 0.6))
    ax.text(2.1, 0.36, f"C1 = solver stopped at the root, answer discarded "
            f"({f(hc['artifact: solver at root, answer discarded'])})\n"
            f"       + root exists, solver failed ({f(hc['artifact: root exists, solver failed'])})\n"
            "C2 = root within about $10^{-13}$ of a plateau", fontsize=5.9, va="center", linespacing=1.4)
    save(fig, path)


# ------------------------------------------------------------------------------------------------- Figure S5

def figS5(inp, path):
    la = inp.json("loewe_audit")
    rr = pd.read_csv(inp["rejected_roots"])
    fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH, 2.2), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.55})
    fig.subplots_adjust(bottom=0.2, top=0.9)
    ax = axes[0]
    for pop, ls in [("C1", "-"), ("C2", "--")]:
        sub = rr[rr[pop]]
        for col_, color, name in [("error_summed_dose", S.NATIVE, "summed dose"), ("error_hsa", S.CORRECTED, "HSA")]:
            v = np.sort(np.maximum(sub[col_].to_numpy(), 1e-16))
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, lw=1.1, ls=ls,
                    label=f"{pop} (n = {len(sub):,}), {name}")
    ax.set_xscale("log")
    ax.set_xlim(1e-16, 300)
    ax.set_ylim(0, 1)
    ax.set_xlabel("$|$fallback $-$ certified root$|$ (points)")
    ax.set_ylabel("Cumulative share")
    ax.legend(loc="upper left", fontsize=5.8)
    S.label(ax, "a", dx=-28)
    ax = axes[1]
    loc = la["c1_location"]
    vals = [("upper plateau", loc["upper"]), ("lower asymptote", loc["lower"]), ("interior", loc["interior"])]
    for i, (name, v) in enumerate(vals):
        ax.barh(2 - i, v, color=S.NATIVE if i == 0 else S.UNCERT, height=0.6)
        ax.text(v + 150, 2 - i, f"{v:,}", va="center", fontsize=6)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([v[0] for v in vals])
    ax.set_xlim(0, 13500)
    ax.set_xlabel(f"C1 roots, co-increasing LL.4 (n = {loc['co_increasing_LL4_C1']:,})")
    S.label(ax, "b", dx=-60)
    save(fig, path)
