"""
Figure 1: Prior predictive distribution of incremental NMB.

Histogram of NMB across parameter draws at multiple WTP thresholds.
Colored green (NMB > 0, intervention CE) and red (NMB < 0, not CE).

Key message: quantifies current decision uncertainty — how often does the
intervention appear cost-effective given what we know before the trial?

Data source: results/voi_draws.df from run_voi.py
"""

import numpy as np
import sciris as sc
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from utils import set_font

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

# WTP thresholds to display (subset of those in run_voi.py)
WTP_DISPLAY = [100, 500, 1000, 2000]

# Colors
GREEN = '#2ca02c'
RED = '#d62728'


def plot(draws_df=None, wtp_thresholds=None):
    if draws_df is None:
        draws_df = sc.loadobj(f'{RESULTS_DIR}/voi_draws.df')
    if wtp_thresholds is None:
        wtp_thresholds = WTP_DISPLAY

    set_font(size=18)
    n_wtp = len(wtp_thresholds)
    fig, axes = pl.subplots(1, n_wtp, figsize=(5 * n_wtp, 5), sharey=True)
    if n_wtp == 1:
        axes = [axes]

    for ax, wtp in zip(axes, wtp_thresholds):
        nmb = wtp * draws_df['delta_dalys'] - draws_df['delta_costs']
        nmb_pos = nmb[nmb >= 0]
        nmb_neg = nmb[nmb < 0]

        bins = np.linspace(nmb.min(), nmb.max(), 40)
        ax.hist(nmb_pos, bins=bins, color=GREEN, alpha=0.75, edgecolor='white', linewidth=0.5, label='CE')
        ax.hist(nmb_neg, bins=bins, color=RED, alpha=0.75, edgecolor='white', linewidth=0.5, label='Not CE')
        ax.axvline(0, color='k', linewidth=1, linestyle='--')

        prob_ce = np.mean(nmb > 0)
        mean_nmb = np.mean(nmb)
        ax.set_title(f'WTP = ${wtp:,}/DALY')
        ax.set_xlabel('Incremental NMB ($)')
        ax.text(0.95, 0.95, f'P(CE) = {prob_ce:.0%}\nE[NMB] = ${mean_nmb:,.0f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=14,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[0].set_ylabel('Number of draws')
    axes[0].legend(frameon=False, fontsize=13, loc='upper left')

    # Panel labels
    for i, ax in enumerate(axes):
        ax.text(-0.08, 1.06, chr(65 + i), transform=ax.transAxes,
                fontsize=24, fontweight='bold', va='top')

    fig.tight_layout()
    sc.savefig(f'{FIGURES_DIR}/fig1_nmb.png', dpi=200)
    print(f'Saved {FIGURES_DIR}/fig1_nmb.png')

    return fig


if __name__ == '__main__':
    plot()
