"""Shared visual conventions for the manuscript figures (see manuscript/figures/README.md)."""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.text import Text  # noqa: E402

WIDTH = 6.3          # printed figure width, inches (the text block is 6.8 in)
BOLD = "cmb10"       # Computer Modern bold; cmr10 has no bold weight
INK = "#1a1a1a"      # measured data, exact quantities
MUTED = "#5c5c5c"    # axes, secondary text
RULE = "#c8c8c8"     # reference lines
GRID = "#ececec"
UNCERT = "#9a9a9a"   # intervals, noise
UNCERT_FILL = "#e3e3e3"
HELDOUT = "#E69F00"  # held-out-pair benchmark models
B1 = "#56B4E9"       # pair offset
B2 = "#0072B2"       # cross-cell linear predictor
B3 = "#009E73"       # pair-seen network
NATIVE = "#D55E00"   # historical SynergyFinder pipeline
CORRECTED = "#0072B2"  # certified roots, continuous extension, C1
CONTROL = "#CC79A7"  # controls and nulls
INHIBITION = "cividis"
INHIBITION_RANGE = (-10, 100)
RESIDUAL = "RdBu_r"
SPLIT = {"train": "#d9d9d9", "validation": "#a6a6a6", "test": "#1a1a1a"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titleweight": "normal", "axes.titlepad": 4, "axes.labelpad": 2.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.pad": 2, "ytick.major.pad": 2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.5, "lines.linewidth": 1.1, "lines.markersize": 3.5,
    "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 600,
    "figure.dpi": 150, "image.interpolation": "nearest", "hatch.linewidth": 0.5,
})


def label(ax, s, x=-0.02, y=1.0, **kw):
    """Bold lower-case panel label placed at the upper left of an axes, outside the plotting area."""
    ax.annotate(s, (0, 1), xycoords="axes fraction", xytext=(kw.pop("dx", -22), kw.pop("dy", 6)),
                textcoords="offset points", fontsize=9, fontfamily=BOLD, va="bottom", ha="left", **kw)


def head(ax, letter, text, dx=-0.0, dy=0.0):
    """Panel letter and short heading above the upper-left corner of an axes (figure coordinates)."""
    fig = ax.figure
    pos = ax.get_position()
    x = pos.x0 + dx
    y = pos.y1 + dy
    fig.text(x, y, letter, fontsize=9, fontfamily=BOLD, va="bottom")
    fig.text(x + 0.12 / fig.get_figwidth() * 2.2, y, text, fontsize=7, va="bottom")


def fig_label(fig, s, x, y):
    fig.text(x, y, s, fontsize=9, fontfamily=BOLD, va="bottom", ha="left")


def hgrid(ax):
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def vgrid(ax):
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def interval(ax, y, v, pair, drug=None, color=INK, marker="o", size=18, horizontal=True, z=3):
    """Point estimate with thick pair-bootstrap and thin drug-jackknife intervals."""
    if horizontal:
        if drug is not None:
            ax.plot(drug, [y, y], color=color, lw=0.7, solid_capstyle="butt", zorder=z - 1)
        ax.plot(pair, [y, y], color=color, lw=2.2, solid_capstyle="butt", zorder=z - 1)
        ax.scatter([v], [y], s=size, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=z)
    else:
        if drug is not None:
            ax.plot([y, y], drug, color=color, lw=0.7, solid_capstyle="butt", zorder=z - 1)
        ax.plot([y, y], pair, color=color, lw=2.2, solid_capstyle="butt", zorder=z - 1)
        ax.scatter([y], [v], s=size, color=color, marker=marker, edgecolor="white", linewidth=0.6, zorder=z)


def residual_cmap():
    return plt.get_cmap(RESIDUAL)


def sequential(color, n=256):
    return LinearSegmentedColormap.from_list("seq", ["#ffffff", color], N=n)


def minus(s):
    """Math minus for negative numbers outside $...$; hyphens in words stay hyphens."""
    parts = re.split(r"((?<!\\)\$[^$]*\$)", str(s))
    return "".join(p if k % 2 else re.sub(r"(?<![A-Za-z0-9])-(?=[0-9.])", "$-$", p) for k, p in enumerate(parts))


def check_glyphs(fig):
    """cmr10 uses the TeX text encoding: outside $...$ these characters print as other glyphs or not at all."""
    fig.canvas.draw()
    bad = set('<>|"_\\{}`')
    for t in fig.findobj(Text):
        s = t.get_text()
        plain = "".join(re.split(r"(?<!\\)\$[^$]*\$", s))
        wrong = {c for c in plain if c in bad or ord(c) > 127}
        if wrong:
            raise ValueError(f"{fig.get_label() or 'figure'}: {sorted(wrong)} outside math in {s!r}")
    tight = fig.get_tightbbox(fig.canvas.get_renderer())  # everything drawn, in inches
    w, h = fig.get_size_inches()
    if tight.x0 < -0.01 or tight.y0 < -0.01 or tight.x1 > w + 0.01 or tight.y1 > h + 0.01:
        raise ValueError(f"drawn content runs past the {w:.2f} x {h:.2f} in figure: {tight.bounds}")


def save(fig, path):
    """Vector PDF (deterministic: no creation date) and a 220 dpi PNG preview."""
    check_glyphs(fig)
    fig.savefig(path.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=220, metadata={"Software": None})
    plt.close(fig)
