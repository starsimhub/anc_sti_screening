"""
EVPPI across multiple WTP thresholds.

For each parameter group, computes EVPPI at a range of WTPs and plots
each group as a line. Shows how parameter importance shifts with WTP:
where is the decision sensitive to what?

Data source: results/voi_draws.df from run_voi.py
"""

import numpy as np
import sciris as sc
import pandas as pd
import matplotlib.pyplot as pl
from utils import set_font
from plot_fig3_evppi import PARAM_GROUPS, GROUP_COLORS, compute_evppi_group

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

WTP_GRID = [1000, 2000, 3000, 4000, 5000, 7500, 10000]


def plot(draws_df=None, wtps=None):
    if draws_df is None:
        draws_df = sc.loadobj(f'{RESULTS_DIR}/voi_draws.df')
    if wtps is None:
        wtps = WTP_GRID

    rows = []
    for wtp in wtps:
        print(f'WTP=${wtp:,}')
        for gname, cols in PARAM_GROUPS.items():
            evppi = compute_evppi_group(draws_df, cols, wtp=wtp)
            rows.append(dict(group=gname, wtp=wtp, evppi=evppi))
            print(f'  {gname}: ${evppi:,.0f}')
    df = pd.DataFrame(rows)

    set_font(size=15)
    fig, ax = pl.subplots(figsize=(11, 7))
    for gname in PARAM_GROUPS.keys():
        sub = df[df.group == gname]
        ax.plot(sub.wtp, sub.evppi, marker='o', linewidth=2,
                color=GROUP_COLORS.get(gname, '#888'), label=gname, alpha=0.9)

    ax.set_xlabel('WTP ($/DALY averted)')
    ax.set_ylabel('EVPPI ($ per simulated cohort)')
    ax.set_title('Parameter importance across willingness-to-pay thresholds')
    ax.legend(loc='upper left', frameon=False, fontsize=11, ncol=2)

    fig.tight_layout()
    sc.savefig(f'{FIGURES_DIR}/fig_evppi_wtps.png', dpi=200)
    sc.saveobj(f'{RESULTS_DIR}/voi_evppi_wtps.df', df)
    print(f'Saved {FIGURES_DIR}/fig_evppi_wtps.png')
    return fig, df


if __name__ == '__main__':
    plot()
