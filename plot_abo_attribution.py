"""
ABO attribution figure — 4 panels (PTB, LBW, Stillbirth, NND).

PTB/LBW stacks (mutually exclusive, sum to total outcome count):
  sole_syph | sole_ct | sole_tv | sole_ng | shared_across_STIs | no_STI_attribution

"Sole" = only this STI exposed the pregnancy. "Shared" = pregnancy exposed
to 2+ STIs (we don't split it further because a shared PTB with NG+CT
can't be uniquely attributed to either).

Stillbirth/NND: syph-only (model gap: NG/CT/TV don't produce them).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent
DISEASES = ('syph', 'ct', 'tv', 'ng')  # display order
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
SCENARIO_LABELS = {
    'soc':                'SOC (syph RPR only)',
    'anc_1screen_90cov':  '1-screen (enrolment) 90%',
    'anc_2screen_90cov':  '2-screen (enrolment + 3rd tri) 90%',
}


def _panel_ptb_lbw(ax, df_attrib, df_totals, outcome, scenarios, title):
    ax.set_title(title, fontsize=11)
    y = np.arange(len(scenarios))
    n_total_col = f'n_{outcome}'

    for i, sc in enumerate(scenarios):
        sub = df_attrib[(df_attrib.scenario == sc) & (df_attrib.outcome == outcome)]
        sole = {d: int(sub[sub.disease == d].n_sole.iloc[0]) for d in DISEASES}
        no_attrib = int(sub[sub.disease == 'none'].n_total.iloc[0])
        n_total = int(df_totals.loc[sc, n_total_col])
        shared = max(0, n_total - sum(sole.values()) - no_attrib)

        left = 0.0
        for d in DISEASES:
            v = sole[d]
            if v:
                ax.barh(i, v, left=left, color=DISEASE_COLORS[d], edgecolor='white',
                        linewidth=0.5, label=LABELS[d] if i == 0 else None)
                left += v
        if shared:
            ax.barh(i, shared, left=left, color=DISEASE_COLORS['shared'], edgecolor='white',
                    linewidth=0.5, label=LABELS['shared'] if i == 0 else None)
            left += shared
        if no_attrib:
            ax.barh(i, no_attrib, left=left, color=DISEASE_COLORS['none'], edgecolor='white',
                    linewidth=0.5, label=LABELS['none'] if i == 0 else None)

        ax.text(n_total * 1.01, i, f'{n_total/1e6:.2f}M', va='center', fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlabel('Count (cumulative 2028-2045, population scale)')


def _panel_syph_only(ax, df_attrib, scenarios, outcome, title):
    ax.set_title(title, fontsize=11)
    y = np.arange(len(scenarios))
    for i, sc in enumerate(scenarios):
        row = df_attrib[(df_attrib.scenario == sc) & (df_attrib.outcome == outcome)]
        n = int(row.n_total.iloc[0]) if not row.empty else 0
        ax.barh(i, n, color=DISEASE_COLORS['syph'], edgecolor='white', linewidth=0.5)
        ax.text(n * 1.01, i, f'{n:,}', va='center', fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_xlabel('Count (cumulative 2028-2045, population scale)')


def main():
    df_attrib = pd.read_csv(REPO / 'results' / 'abo_attribution.csv')
    df_totals = pd.read_csv(REPO / 'results' / 'abo_totals.csv').set_index('scenario')
    scenarios = list(SCENARIO_LABELS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True)
    _panel_ptb_lbw(axes[0, 0], df_attrib, df_totals, 'ptb', scenarios, 'Preterm birth (PTB)')
    _panel_ptb_lbw(axes[0, 1], df_attrib, df_totals, 'lbw', scenarios, 'Low birth weight (LBW)')
    _panel_syph_only(axes[1, 0], df_attrib, scenarios, 'stillbirth',
                     'Stillbirth (syph-attributable — NG/CT/TV pathway not modelled)')
    _panel_syph_only(axes[1, 1], df_attrib, scenarios, 'nnd',
                     'Neonatal death (syph-attributable — NG/CT/TV pathway not modelled)')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle('Attributable adverse birth outcomes by scenario '
                 '(central_reversible, 1 draw × 1 seed, 2028-2045)',
                 fontsize=13)

    (REPO / 'figures').mkdir(exist_ok=True)
    outpath = REPO / 'figures' / 'abo_attribution.png'
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f'Saved {outpath}')


if __name__ == '__main__':
    main()
