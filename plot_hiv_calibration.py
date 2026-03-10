"""
HIV calibration figure (supplementary).

2x3 grid: HIV prevalence, new infections, ART, PLHIV, deaths, population.
Uses calibration stats from run_msim.py with data overlay.
"""

import sciris as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from utils import set_font

# Settings
RESULTS_DIR = 'results'
DATA_DIR = 'data'
FIGURES_DIR = 'figures'
START_YEAR = 1990
END_YEAR = 2025

# Colors
HIV_COLOR = '#2171b5'
DATA_COLOR = 'k'
BAND_ALPHA = 0.2


def load_data(scenario='soc'):
    cs = sc.loadobj(f'{RESULTS_DIR}/zimbabwe_calib_stats_{scenario}.df')
    data_hiv = pd.read_csv(f'{DATA_DIR}/zimbabwe_hiv_calib.csv')
    return cs, data_hiv


def get_stats(cs, col, lo='10%', hi='90%'):
    return cs[(col, '50%')], cs[(col, lo)], cs[(col, hi)]


def plot_panel(ax, cs, col, data_df=None, data_col=None, title='', color='C0',
               ylabel=None, si_ticks=False, ylim_bottom=True):
    """ Plot a single calibration panel with model band and data points """
    med, lo, hi = get_stats(cs, col)
    years = cs.index

    ax.fill_between(years, lo, hi, alpha=BAND_ALPHA, color=color, linewidth=0)
    ax.plot(years, med, color=color, linewidth=1.5, label='Model (median)')

    if data_df is not None and data_col is not None:
        d = data_df[['time', data_col]].dropna()
        if len(d):
            ax.scatter(d.time, d[data_col], color=DATA_COLOR, s=15, zorder=5,
                       label='Data', edgecolors='none')

    ax.set_title(title, fontsize=18, pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlim(START_YEAR, END_YEAR)
    if ylim_bottom:
        ax.set_ylim(bottom=0)
    if si_ticks:
        sc.SIticks(ax, axis='y')
    ax.tick_params(labelsize=13)

    return


def plot_calibration_figure(scenario='soc'):
    set_font(size=18)
    cs, data_hiv = load_data(scenario)

    fig = pl.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, left=0.06, right=0.98, bottom=0.06, top=0.93,
                  wspace=0.28, hspace=0.35)

    # Row 1
    ax = fig.add_subplot(gs[0, 0])
    plot_panel(ax, cs, 'hiv.prevalence_15_49', data_hiv, 'hiv.prevalence_15_49',
               '(A) HIV prevalence (15-49)', HIV_COLOR, ylabel='Prevalence')
    ax.legend(fontsize=12, loc='upper right', frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    plot_panel(ax, cs, 'hiv.new_infections', data_hiv, 'hiv.new_infections',
               '(B) HIV new infections', HIV_COLOR, si_ticks=True)

    ax = fig.add_subplot(gs[0, 2])
    plot_panel(ax, cs, 'hiv.n_on_art', data_hiv, 'hiv.n_on_art',
               '(C) People on ART', HIV_COLOR, si_ticks=True)

    # Row 2
    ax = fig.add_subplot(gs[1, 0])
    plot_panel(ax, cs, 'hiv.n_infected', data_hiv, 'hiv.n_infected',
               '(D) People living with HIV', HIV_COLOR, si_ticks=True)

    ax = fig.add_subplot(gs[1, 1])
    plot_panel(ax, cs, 'hiv.new_deaths', data_hiv, 'hiv.new_deaths',
               '(E) HIV-related deaths', HIV_COLOR, si_ticks=True)

    ax = fig.add_subplot(gs[1, 2])
    plot_panel(ax, cs, 'n_alive', data_hiv, 'n_alive',
               '(F) Total population', '#555555', si_ticks=True)

    pl.savefig(f'{FIGURES_DIR}/hiv_calibration.png', dpi=200, bbox_inches='tight')
    print(f'Saved {FIGURES_DIR}/hiv_calibration.png')

    return


if __name__ == '__main__':
    plot_calibration_figure()
