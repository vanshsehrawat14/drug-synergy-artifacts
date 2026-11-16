"""Response analysis figures; every figure writes its exact plotting table beside it (table view for low-contrast slots)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

H = Path('../handoff'); F = H / 'figures'; F.mkdir(exist_ok=True)
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # validated categorical slots 1-4 (light surface)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'lines.linewidth': 2, 'lines.markersize': 7})


def label_end(ax, x, y, text, color_ink=INK):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=color_ink)


def fig1():
    t = pd.read_csv(H / 'decision/ranking_stability.csv')
    t = t[(t.level == 'pair_cell')]
    sel = [('N', 'C1', 'Primary correction (14,197 roots)'), ('N', 'I', 'Interval-certified (15,620 roots)'),
           ('N', 'H', 'HSA fallback (convention change)'), ('P', 'N', 'Published vs native (reconstruction)')]
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    nudge = {'Primary correction (14,197 roots)': -1.0, 'Interval-certified (15,620 roots)': 1.0}
    for (b, a, lab), col, mk in zip(sel, S, 'osD^'):
        g = t[(t.base == b) & (t.alternative == a)].sort_values('budget')
        x, y = g.budget * 100, g.replacement_fraction * 100
        ax.plot(x, y, color=col, marker=mk)
        if g.pair_ci_lo.notna().any():
            ax.vlines(x, g.pair_ci_lo * 100, g.pair_ci_hi * 100, color=col, linewidth=1.2)
        label_end(ax, x.iloc[-1], y.iloc[-1] + nudge.get(lab, 0), lab)
        rows.append(g.assign(series=lab)[['series', 'budget', 'replacement_fraction', 'pair_ci_lo', 'pair_ci_hi', 'replaced', 'selected']])
    ax.axhline(5, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.annotate('frozen 5% materiality line (applies at top-5%)', (2.2, 5), xytext=(0, -12), textcoords='offset points', fontsize=8, color=INK2)
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1%', '2%', '5%', '10%', '20%'])
    ax.set_xlabel('Selection budget (share of 22,737 pair-cells)'); ax.set_ylabel('Selected pair-cells replaced (%)')
    ax.set_xlim(.9, 60); ax.set_ylim(0, 36)
    ax.set_title('Native top-k list turnover by Loewe scoring version (bars: pair bootstrap 95%)', fontsize=10, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(F / 'fig1_turnover_by_budget.png', dpi=200); plt.close(fig)
    pd.concat(rows).to_csv(F / 'fig1_turnover_by_budget.csv', index=False)


def fig2():
    a = pd.read_csv(H / 'decision/edge_exclusion_sensitivity.csv'); a = a[a.budget == .05]
    b = pd.read_csv(H / 'decision/consistent_edge_conventions.csv'); b = b[b.budget == .05]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rows = []
    for g, col, mk, lab, xcol in [(a, S[0], 'o', 'Exclude near-edge fallback roots only', 'edge_exclusion_eps'),
                                  (b, S[1], 's', 'Exclude near-edge roots consistently (incl. accepted)', 'eps')]:
        g = g[g[xcol] > 0].sort_values(xcol)
        x, y = g[xcol], g.replacement_fraction * 100
        ax.plot(x, y, color=col, marker=mk)
        label_end(ax, x.iloc[0], y.iloc[0], '')
        rows.append(pd.DataFrame({'series': lab, 'eps': x, 'top5_replacement_pct': y}))
    ax.annotate('fallback roots only', (1e-8, 5.28), xytext=(8, 6), textcoords='offset points', fontsize=9, color=INK)
    ax.annotate('consistent exclusion\n(native-accepted near-edge too)', (1e-2, 1.23), xytext=(-60, 22), textcoords='offset points', fontsize=9, color=INK)
    ax.axhline(5, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
    ax.axhline(6.95, color=INK2, linestyle=':', linewidth=1)
    ax.annotate('ε = 0 (all certified roots): 6.95%', (1e-12, 6.95), xytext=(0, 3), textcoords='offset points', fontsize=8, color=INK2)
    ax.annotate('5% materiality', (1e-12, 5), xytext=(0, -11), textcoords='offset points', fontsize=8, color=INK2)
    ax.set_xscale('log'); ax.set_xlabel('ε: certified roots within ε inhibition points of an asymptote treated as undefined')
    ax.set_ylabel('Native top-5% replaced (%)'); ax.set_ylim(0, 8)
    ax.set_title('Decision effect depends on the edge convention, not root accuracy', fontsize=10, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(F / 'fig2_edge_convention.png', dpi=200); plt.close(fig)
    pd.concat(rows).to_csv(F / 'fig2_edge_convention.csv', index=False)


def fig3():
    p = pd.read_csv(H / 'decision/prospective_scoring.csv')
    p = p[(p.base == 'N') & (p.alternative == 'C1')]
    obs = p[p.responses == 'observed'].sort_values('budget')
    seeds = p[p.responses.str.contains('seed20')]
    rng = seeds.groupby('budget').replacement_fraction.agg(['min', 'max']).reset_index()
    mean = p[p.responses == 'neural_F2_seed_mean'].sort_values('budget')
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = obs.budget * 100
    ax.fill_between(rng.budget * 100, rng['min'] * 100, rng['max'] * 100, color=S[1], alpha=.18, linewidth=0)
    ax.plot(x, mean.replacement_fraction * 100, color=S[1], marker='s')
    ax.plot(x, obs.replacement_fraction * 100, color=S[0], marker='o')
    label_end(ax, x.iloc[-1], mean.replacement_fraction.iloc[-1] * 100, 'Predicted (F2 seed mean; band: seeds)')
    label_end(ax, x.iloc[-1], obs.replacement_fraction.iloc[-1] * 100 - .6, 'Observed responses')
    ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1%', '2%', '5%', '10%', '20%'])
    ax.set_xlim(.9, 110); ax.set_ylim(0, 16)
    ax.set_xlabel('Selection budget'); ax.set_ylabel('Top-k replaced, native → C1 (%)')
    ax.set_title('Same reference change, retrospective vs prospective scoring', fontsize=10, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(F / 'fig3_prospective.png', dpi=200); plt.close(fig)
    p.to_csv(F / 'fig3_prospective.csv', index=False)


def fig4():
    e = pd.read_csv(H / 'cross_screen/cross_screen_errors.csv'); e = e[e.model == 'f2']
    s = pd.read_csv(H / 'cross_screen/cross_screen_standardization.csv')
    d = e.merge(s[['screen', 'standardized_gain_to_oneil_strata']], on='screen')
    order = ['ONEIL', 'ALMANAC', 'NCATS_ES', 'NCATS_DIPG']
    d = d.set_index('screen').loc[order].reset_index()
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hlines(y, d.gain_vs_bliss_pair_ci_lo * 100, d.gain_vs_bliss_pair_ci_hi * 100, color=S[0], linewidth=2)
    ax.plot(d.gain_vs_bliss * 100, y, 'o', color=S[0], label='Raw (pair bootstrap 95%)')
    ax.plot(d.standardized_gain_to_oneil_strata * 100, y - .18, 's', color=S[1], label="Reweighted to O'Neil input strata")
    for yi, r in zip(y, d.itertuples()):
        ax.annotate(f'{r.gain_vs_bliss*100:.1f}', (r.gain_vs_bliss_pair_ci_hi * 100, yi), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(["O'Neil", 'ALMANAC', 'NCATS ES', 'NCATS DIPG'])
    ax.set_xlabel('F2 MSE reduction vs training-pair Bliss reference (%)'); ax.set_xlim(0, 50)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    ax.set_title('Within-screen gains; input-strata standardization does not close gaps', fontsize=10, color=INK, loc='left')
    fig.tight_layout(); fig.savefig(F / 'fig4_cross_screen.png', dpi=200); plt.close(fig)
    d.to_csv(F / 'fig4_cross_screen.csv', index=False)


def fig5():
    c = pd.read_csv(H / 'gpu_ablation_summary.csv')
    masks = ['no_drug_identity', 'no_cell_identity', 'no_dose']
    labels = ['Drug identity removed', 'Cell identity removed', 'Dose removed']
    levels = [('response', 'Dose-row response'), ('panel_contrast', 'Pair-cell contrast (P−A)'), ('dose_contrast', 'Dose-shape contrast (E−P−D)')]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), sharey=True)
    y = np.arange(3)[::-1]
    for ax, (lv, title) in zip(axes, levels):
        g = c[c.level == lv].set_index('model').loc[masks]
        ax.hlines(y, g.drug_low * 100, g.drug_high * 100, color=INK2, linewidth=1)
        ax.hlines(y, g.pair_low * 100, g.pair_high * 100, color=S[0], linewidth=3.5)
        ax.plot(g.change * 100, y, 'o', color=S[0], markeredgecolor=SURF, markeredgewidth=1.5)
        for yi, r in zip(y, g.itertuples()):
            ax.annotate(f'{r.change*100:+.1f}' + (' *' if r.material_change else ''), (r.change * 100, yi), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=8, color=INK, zorder=5,
                        bbox=dict(fc=SURF, ec='none', pad=1))
        for v in (-5, 5):
            ax.axvline(v, color=INK2, linestyle=(0, (4, 3)), linewidth=1)
        ax.axvline(0, color=INK2, linewidth=.8)
        ax.set_title(title, fontsize=9.5, color=INK, loc='left'); ax.set_xlim(-16, 31); ax.set_ylim(-.6, 2.6)
        ax.set_xlabel('MSE change vs full-input F2 (%)')
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels)
    fig.suptitle('Frozen input ablations (3 seeds × 5 folds). Thick: pair bootstrap 95%; thin: drug jackknife 95%; * frozen material rule',
                 fontsize=9, color=INK2, x=.01, ha='left')
    fig.tight_layout(); fig.savefig(F / 'fig5_gpu_ablations.png', dpi=200); plt.close(fig)
    c.to_csv(F / 'fig5_gpu_ablations.csv', index=False)


if __name__ == '__main__':
    for f in sys.argv[1:] or ['fig1', 'fig2', 'fig3', 'fig4', 'fig5']:
        globals()[f]()
