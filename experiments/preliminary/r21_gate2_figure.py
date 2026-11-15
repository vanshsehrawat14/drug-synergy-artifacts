"""R21: one-page Gate 2 figure from R18/R19/R20 outputs. Run: python r21_gate2_figure.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; RES = ROOT / 'results' / 'preliminary'; FIG = RES
o = json.loads((RES / 'r18_gate2.json').read_text()); r20 = json.loads((RES / 'r20_fallback_mechanism.json').read_text())
s0 = {r['panel']: r['pooled_cr2'] for r in json.loads((RES / 'r19_summary.json').read_text())}['oneil_only_loewe']
BLUE, ORANGE, GREEN, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#9aa1a9'
INK, INK2, GRID = '#1f2328', '#59636e', '#e6e8eb'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold'})
fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
c = o['components']; p5 = o['P5']; p4 = o['P4']['runs']


def style(a, axis='x'):
    a.grid(axis=axis, color=GRID, lw=.8); a.set_axisbelow(True)
    for sp in ['top', 'right']:
        a.spines[sp].set_visible(False)


# (a) derived zero-parameter predictors vs learned models
a = ax[0, 0]
rows = [('Loewe − Bliss reference difference', c['M_DC']['cr2_vs_Y_L'], ORANGE),
        ('  its SynergyFinder-implementation part', c['M_impl']['cr2_vs_Y_L'], ORANGE),
        ('  summed-dose fallback, re-derived from code', c['M_exist']['cr2_vs_Y_L'], ORANGE),
        ('  textbook Loewe vs Bliss algebra, real curves', c['M_alg']['cr2_vs_Y_L'], ORANGE),
        ('  same algebra, idealised curves', c['M_alg_ideal']['cr2_vs_Y_L'], ORANGE),
        ('Kernel ridge trained on labels', p5['Y_L']['naive']['krr']['cr2'], BLUE),
        ('Neighbour CF trained on labels', p5['Y_L']['naive']['knn']['cr2'], BLUE),
        ('Monotherapy MLP trained on labels', s0, BLUE)]
y = list(range(len(rows)))[::-1]
for yi, (lab, v, col) in zip(y, rows):
    a.barh(yi, v, color=col, height=.62)
    a.text(v + (.012 if v >= 0 else -.012), yi, f'{v:.2f}', va='center', ha='left' if v >= 0 else 'right', color=INK, fontsize=9)
a.set_yticks(y); a.set_yticklabels([r[0] for r in rows]); a.axvline(0, color=INK2, lw=.8)
a.set_xlim(-.15, .9); a.set_xlabel('Contrast R² on held-out pairs (DrugComb Loewe, O\'Neil)')
a.set_title('(a) Monotherapy alone, no fitting, beats every trained model', loc='left')
a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
         labels=['Monotherapy only, no fitting', 'Trained on labels'], loc='lower right', bbox_to_anchor=(1.0, 0.02), frameon=False, fontsize=9)
style(a)

# (b) Bliss-true null
a = ax[0, 1]
groups = [('Loewe\nreal labels', p5['Y_L']['naive']), ('Loewe\nno true interaction', p4['cell_independent:loewe_sim']),
          ('Bliss\nreal labels', p5['Y_B']['naive']), ('Bliss\nno true interaction', p4['cell_independent:bliss_sim'])]
w = .36
for i, (lab, r) in enumerate(groups):
    for j, (m, col, nm) in enumerate([('knn', BLUE, 'Neighbour CF'), ('krr', GREEN, 'Kernel ridge')]):
        v = r[m]['cr2']; xpos = i + (j - .5) * (w + .04)
        a.bar(xpos, v, width=w, color=col, label=nm if i == 0 else None)
        a.text(xpos, v + (.012 if v >= 0 else -.03), f'{v:.2f}'.replace('-0.00', '0.00'), ha='center', fontsize=9, color=INK)
a.set_xticks(range(4)); a.set_xticklabels([g[0] for g in groups]); a.axhline(0, color=INK2, lw=.8)
a.set_ylim(-.15, .62); a.set_ylabel('Contrast R² on held-out pairs')
a.set_title('(b) With no true interaction, Loewe still looks learnable', loc='left')
a.legend(frameon=False, loc='upper right', fontsize=9); style(a, 'y')

# (c) trained learner vs reference-only, per metric
a = ax[1, 0]
ssT = p5['T']['contrast_ss']
mets = [('Bliss', 'Y_B'), ('HSA', 'Y_H'), ('DrugComb Loewe', 'Y_L')]
for i, (lab, k) in enumerate(mets):
    for j, (v, col, nm) in enumerate([(p5[k]['naive']['krr']['cr2'], BLUE, 'Kernel ridge trained on the score'),
                                      (p5[k]['reference_only']['cr2'], ORANGE, 'Reference only, no combination data')]):
        xpos = i + (j - .5) * (w + .04)
        a.bar(xpos, v, width=w, color=col, label=nm if i == 0 else None)
        a.text(xpos, v + (.015 if v >= 0 else -.045), f'{v:.2f}', ha='center', fontsize=9, color=INK)
a.set_xticks(range(3))
a.set_xticklabels([f'{lab}\n{p5[k]["contrast_ss"] / ssT:.1f}×' for lab, k in mets])
a.set_xlabel('Score, with its non-additive variance as a multiple of the raw response\'s')
a.axhline(0, color=INK2, lw=.8); a.set_ylim(-.6, .85); a.set_ylabel('Contrast R² on held-out pairs')
a.set_title('(c) For Loewe, the reference alone beats a trained model', loc='left')
a.legend(frameon=False, loc='upper left', fontsize=9); style(a, 'y')

# (d) derived prediction: shift grows with partner dose in units of the stronger drug's IC50
a = ax[1, 1]
q = r20['shift_by_logratio_quintile']; labs = list(q); vals = [q[k] for k in labs]
a.plot(range(5), vals, color=ORANGE, lw=2, marker='o', ms=8, markeredgecolor='white', markeredgewidth=2)
for i, v in enumerate(vals):
    a.text(i, v + 1.0, f'{v:.1f}', ha='center', fontsize=9, color=INK)
nice = ['≤ −0.9', '−0.9 to 0.1', '0.1 to 1.1', '1.1 to 2.2', '> 2.2']
a.set_xticks(range(5)); a.set_xticklabels(nice); a.set_ylim(0, 27)
a.set_xlabel('log10(partner dose ÷ stronger drug IC50), quintiles of fallback cells')
a.set_ylabel('Loewe reference inflation (inhibition points)')
a.set_title(f'(d) Fallback inflation tracks partner dose ÷ IC50 ({100 * (1 - r20["cause_share"]["root"]):.0f}% of cells)', loc='left')
style(a, 'y')
fig.tight_layout(pad=2.0)
FIG.mkdir(exist_ok=True)
fig.savefig(FIG / 'gate2_mechanism.png', dpi=160); fig.savefig(FIG / 'gate2_mechanism.pdf')
print('saved')
