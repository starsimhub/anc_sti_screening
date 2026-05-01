"""
Figure 2: Cost-effectiveness plane.

Scatter of (ΔDALYs averted, ΔCosts) across parameter draws, with WTP
threshold lines. Points in the lower-right quadrant (more DALYs averted,
lower costs) are dominant.

Key message: joint uncertainty in costs and effects — where do draws land
relative to the WTP threshold?

Data source: results/voi_draws.df from run_voi.py
"""

import numpy as np
import sciris as sc
import matplotlib.pyplot as pl
from utils import set_font

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

# WTP lines to overlay
WTP_LINES = [500, 1000, 2000]


def plot(draws_df=None):
    if draws_df is None:
        draws_df = sc.loadobj(f'{RESULTS_DIR}/voi_draws.df')

    set_font(size=18)
    fig, ax = pl.subplots(figsize=(8, 7))

    delta_dalys = draws_df['delta_dalys']
    delta_costs = draws_df['delta_costs']

    # Color by whether intervention is cost-effective at middle WTP
    wtp_mid = WTP_LINES[len(WTP_LINES) // 2]
    nmb = wtp_mid * delta_dalys - delta_costs
    colors = np.where(nmb > 0, '#2ca02c', '#d62728')

    ax.scatter(delta_dalys, delta_costs, c=colors, alpha=0.5, s=25, edgecolors='none')

    # WTP threshold lines
    x_range = np.array([0, max(delta_dalys.max() * 1.1, 0.1)])
    for wtp in WTP_LINES:
        ax.plot(x_range, wtp * x_range, '--', color='grey', linewidth=1, alpha=0.7)
        # Label at the right end
        x_label = x_range[1] * 0.85
        y_label = wtp * x_label
        if y_label < delta_costs.max() * 1.3:
            ax.text(x_label, y_label, f'${wtp:,}', fontsize=11,
                    color='grey', ha='left', va='bottom', rotation=np.degrees(np.arctan(wtp * x_range[1] / (delta_costs.max() * 1.3 or 1))))

    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    ax.set_xlabel('DALYs averted (ΔDALYs)')
    ax.set_ylabel('Incremental cost (ΔCosts, $)')
    ax.set_title('Cost-effectiveness plane')

    # Quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    label_kw = dict(fontsize=11, color='grey', alpha=0.6, ha='center', va='center')
    ax.text(xlim[1] * 0.75, ylim[0] * 0.75, 'Dominant\n(better & cheaper)', **label_kw)
    ax.text(xlim[0] * 0.75 if xlim[0] < 0 else xlim[1] * 0.25,
            ylim[1] * 0.75, 'Trade-off', **label_kw)

    fig.tight_layout()
    sc.savefig(f'{FIGURES_DIR}/fig2_ceplane.png', dpi=200)
    print(f'Saved {FIGURES_DIR}/fig2_ceplane.png')

    return fig


if __name__ == '__main__':
    plot()
