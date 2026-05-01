"""
Cost-effectiveness acceptability curve + EVPI curve.

P(NMB > 0) and EVPI as functions of WTP threshold. Answers:
  - at what WTP does the intervention become plausibly CE?
  - how much decision uncertainty is resolvable by perfect information at each WTP?

Data source: results/voi_draws.df from run_voi.py
"""

import numpy as np
import sciris as sc
import matplotlib.pyplot as pl
from utils import set_font

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

WTP_GRID = np.concatenate([np.arange(0, 2001, 100), np.arange(2200, 10001, 200)])

GDP_PER_CAPITA = 1500  # Zimbabwe 2023, World Bank (approximate)


def plot(draws_df=None):
    if draws_df is None:
        draws_df = sc.loadobj(f'{RESULTS_DIR}/voi_draws.df')

    d = draws_df['delta_dalys'].values
    c = draws_df['delta_costs'].values

    prob_ce = np.array([(wtp * d - c > 0).mean() for wtp in WTP_GRID])
    evpi    = np.array([np.mean(np.maximum(wtp * d - c, 0))
                        - max(np.mean(wtp * d - c), 0) for wtp in WTP_GRID])

    set_font(size=16)
    fig, axes = pl.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(WTP_GRID, prob_ce * 100, color='#2ca02c', linewidth=2.5)
    ax.axhline(50, color='grey', linewidth=0.8, linestyle=':')
    for x, label in [(GDP_PER_CAPITA, '1× GDP/cap'), (3 * GDP_PER_CAPITA, '3× GDP/cap')]:
        ax.axvline(x, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.text(x, 95, f'  {label}', fontsize=11, color='grey', va='top')
    ax.set_xlabel('WTP ($/DALY averted)')
    ax.set_ylabel('P(intervention is cost-effective) (%)')
    ax.set_title('Cost-effectiveness acceptability curve')
    ax.set_ylim(-2, 102)
    ax.set_xlim(0, WTP_GRID[-1])

    ax = axes[1]
    ax.plot(WTP_GRID, evpi, color='#d62728', linewidth=2.5)
    for x, label in [(GDP_PER_CAPITA, '1× GDP/cap'), (3 * GDP_PER_CAPITA, '3× GDP/cap')]:
        ax.axvline(x, color='grey', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.text(x, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0,
                f'  {label}', fontsize=11, color='grey', va='top')
    ax.set_xlabel('WTP ($/DALY averted)')
    ax.set_ylabel('EVPI ($ per simulated cohort)')
    ax.set_title('Expected value of perfect information')
    ax.set_xlim(0, WTP_GRID[-1])

    for i, ax in enumerate(axes):
        ax.text(-0.08, 1.06, chr(65 + i), transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='top')

    fig.tight_layout()
    sc.savefig(f'{FIGURES_DIR}/fig_ceac.png', dpi=200)
    print(f'Saved {FIGURES_DIR}/fig_ceac.png')
    return fig


if __name__ == '__main__':
    plot()
