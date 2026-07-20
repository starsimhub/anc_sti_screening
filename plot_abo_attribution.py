"""
ABO attribution figure — PTB and LBW panels; each panel shows the
3 scenarios × 4 assumption regimes grid.

PTB/LBW stacks (mutually exclusive, sum to n_outcome):
  sole_syph | sole_ct | sole_tv | sole_ng | shared_across_STIs | no_STI_attribution

Rows grouped by assumption regime; within each regime the 3 scenarios
(SOC, 1-screen 90%, 2-screen 90%) appear as sub-rows. This layout makes
regime-to-regime comparison the primary read while still showing
per-scenario impact within each regime.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parent
DISEASES = ('syph', 'ct', 'tv', 'ng')
DISEASE_COLORS = {
    'syph':   '#4e79a7',
    'ct':     '#f28e2b',
    'tv':     '#59a14f',
    'ng':     '#e15759',
    'shared': '#af7aa1',
    'none':   '#bab0ac',
}
LABELS = {
    'syph':   'Syphilis (sole)',
    'ct':     'Chlamydia (sole)',
    'tv':     'Trichomoniasis (sole)',
    'ng':     'Gonorrhoea (sole)',
    'shared': 'Shared (2+ STIs)',
    'none':   'No STI attribution',
}
SCENARIO_SHORT = {
    'soc':                'SOC',
    'anc_1screen_90cov':  '1-screen 90%',
    'anc_2screen_90cov':  '2-screen 90%',
}
REGIME_LABEL = {
    'no_treatment_effect': 'No treatment effect',
    'weak_effects':        'Weak effects',
    'central_reversible':  'Central + reversible',
    'strong_effects':      'Strong effects',
}
# regime display order (worst → best assumption for interventions)
REGIME_ORDER = ['no_treatment_effect', 'weak_effects', 'central_reversible', 'strong_effects']
SCENARIO_ORDER = ['soc', 'anc_1screen_90cov', 'anc_2screen_90cov']


def _row_label(regime, scenario, show_regime):
    scen = SCENARIO_SHORT[scenario]
    return f'{REGIME_LABEL[regime]}\n{scen}' if show_regime else f'  {scen}'


def _panel_ptb_lbw(ax, attrib, totals, outcome, title):
    ax.set_title(title, fontsize=11)
    y = 0
    tick_pos, tick_labels = [], []

    for r_i, regime in enumerate(REGIME_ORDER):
        for s_i, sc in enumerate(SCENARIO_ORDER):
            sub = attrib[(attrib.scenario == sc) & (attrib.assumption == regime) & (attrib.outcome == outcome)]
            if sub.empty:
                y += 1
                continue
            sole = {d: int(sub[sub.disease == d].n_sole.median()) for d in DISEASES}
            no_attrib = int(sub[sub.disease == 'none'].n_total.median())
            n_total_row = totals[(totals.scenario == sc) & (totals.assumption == regime)]
            n_total = int(n_total_row[f'n_{outcome}'].median()) if not n_total_row.empty else 0
            shared = max(0, n_total - sum(sole.values()) - no_attrib)

            left = 0.0
            for d in DISEASES:
                v = sole[d]
                if v:
                    ax.barh(y, v, left=left, color=DISEASE_COLORS[d], edgecolor='white',
                            linewidth=0.5,
                            label=LABELS[d] if (r_i == 0 and s_i == 0) else None)
                    left += v
            if shared:
                ax.barh(y, shared, left=left, color=DISEASE_COLORS['shared'], edgecolor='white',
                        linewidth=0.5, label=LABELS['shared'] if (r_i == 0 and s_i == 0) else None)
                left += shared
            if no_attrib:
                ax.barh(y, no_attrib, left=left, color=DISEASE_COLORS['none'], edgecolor='white',
                        linewidth=0.5, label=LABELS['none'] if (r_i == 0 and s_i == 0) else None)

            if n_total > 0:
                ax.text(n_total * 1.01, y, f'{n_total/1e6:.2f}M' if n_total > 1e6 else f'{n_total/1e3:.0f}K',
                        va='center', fontsize=8)

            tick_pos.append(y)
            tick_labels.append(_row_label(regime, sc, show_regime=(s_i == 0)))
            y += 1
        y += 0.5  # separator between regimes

    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=8)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlabel('Count (cumulative 2028-2045, population scale)')


def main():
    attrib = pd.read_csv(REPO / 'results' / 'abo_attribution.csv')
    totals = pd.read_csv(REPO / 'results' / 'abo_totals.csv')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    _panel_ptb_lbw(axes[0], attrib, totals, 'ptb', 'Preterm birth (PTB)')
    _panel_ptb_lbw(axes[1], attrib, totals, 'lbw', 'Low birth weight (LBW)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle('Attributable adverse birth outcomes: 3 scenarios × 4 effect-size regimes '
                 '(cumulative 2028-2045)', fontsize=13)

    (REPO / 'figures').mkdir(exist_ok=True)
    outpath = REPO / 'figures' / 'abo_attribution.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Saved {outpath}')


if __name__ == '__main__':
    main()
