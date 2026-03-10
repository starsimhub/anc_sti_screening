"""
STI epidemiology figure for calibration validation.

Panels:
    A: NG prevalence by age and sex (bar chart)
    B: CT prevalence by age and sex (bar chart)
    C: TV prevalence by age and sex (bar chart)
    D: NG prevalence over time (band plot with data)
    E: CT prevalence over time (band plot with data)
    F: TV prevalence over time (band plot with data)

Data sources:
    - Calibration stats from run_msim.py (percentile bands)
    - Age/sex epi data extracted by save_stats() in model.py
    - Zimbabwe STI survey data (zimbabwe_sti_data.csv)
"""

import sciris as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from utils import set_font

RESULTS_DIR = 'results'
DATA_DIR = 'data'
FIGURES_DIR = 'figures'

# Colors
F_COLOR = '#d46e9c'
M_COLOR = '#4a90d9'
BAND_ALPHA = 0.2
DATA_COLOR = 'k'

START_YEAR = 2000
END_YEAR   = 2025


def get_stats(cs, col, lo='10%', hi='90%'):
    return cs[(col, '50%')], cs[(col, lo)], cs[(col, hi)]


def plot_prev_by_age(epi_df, disease, ax):
    """
    Plot prevalence by age and sex for a single disease.

    Args:
        epi_df (DataFrame): age/sex/prevalence data from save_stats()
        disease (str):      'ng', 'ct', or 'tv'
        ax:                 matplotlib axes
    """
    thisdf = epi_df.loc[
        (epi_df.disease == disease) &
        (epi_df.age != '0-15') &
        (epi_df.age != '65+')
    ].copy()
    thisdf['prevalence'] *= 100
    sns.barplot(data=thisdf, x='age', y='prevalence', hue='sex',
                ax=ax, palette=[F_COLOR, M_COLOR])
    ax.set_title(f'{disease.upper()} prevalence by age')
    ax.set_ylabel('Prevalence (%)')
    ax.set_xlabel('')
    ax.set_ylim(bottom=0)

    return


def plot_prev_timeseries(cs, disease, ax, sti_data=None):
    """
    Plot prevalence time series with uncertainty band.

    Args:
        cs:        calibration stats DataFrame
        disease:   'ng', 'ct', or 'tv'
        ax:        matplotlib axes
        sti_data:  optional survey data for overlay
    """
    col = f'{disease}.prevalence'
    try:
        med, lo, hi = get_stats(cs, col)
    except KeyError:
        # Try alternative column name
        col = f'{disease}.prevalence_f'
        med, lo, hi = get_stats(cs, col)

    years = cs.index
    mask = (years >= START_YEAR) & (years <= END_YEAR)

    dis_colors = {'ng': '#e6550d', 'ct': '#3182bd', 'tv': '#31a354'}
    color = dis_colors.get(disease, 'C0')

    ax.fill_between(years[mask], lo[mask] * 100, hi[mask] * 100,
                     alpha=BAND_ALPHA, color=color, linewidth=0)
    ax.plot(years[mask], med[mask] * 100, color=color, linewidth=1.5, label='Model')

    if sti_data is not None:
        dcol = f'{disease}_prevalence'
        if dcol in sti_data.columns:
            d = sti_data[['time', dcol]].dropna()
            d = d[(d.time >= START_YEAR) & (d.time <= END_YEAR)]
            if len(d):
                ax.scatter(d.time, d[dcol] * 100, color=DATA_COLOR, s=20,
                           zorder=5, label='Data', edgecolors='none')

    ax.set_title(f'{disease.upper()} prevalence (%)')
    ax.set_xlim(START_YEAR, END_YEAR)
    ax.set_ylim(bottom=0)
    ax.set_ylabel('')
    ax.set_xlabel('')

    return


def plot_epi_figure(scenario='soc'):
    set_font(size=20)

    # Load data
    cs = sc.loadobj(f'{RESULTS_DIR}/zimbabwe_calib_stats_{scenario}.df')
    try:
        epi_df = sc.loadobj(f'{RESULTS_DIR}/epi_df_{scenario}.df')
    except FileNotFoundError:
        epi_df = None
    try:
        sti_data = pd.read_csv(f'{DATA_DIR}/zimbabwe_sti_data.csv')
    except FileNotFoundError:
        sti_data = None

    diseases = ['ng', 'ct', 'tv']
    fig, axes = pl.subplots(2, 3, figsize=(20, 10))

    # Top row: prevalence by age and sex
    if epi_df is not None:
        for i, disease in enumerate(diseases):
            plot_prev_by_age(epi_df, disease, axes[0, i])
            if i == 0:
                axes[0, i].legend(frameon=False, fontsize=14)
            else:
                axes[0, i].get_legend().set_visible(False)
    else:
        for i in range(3):
            axes[0, i].text(0.5, 0.5, 'Run save_stats() first',
                           transform=axes[0, i].transAxes, ha='center', va='center')

    # Bottom row: prevalence time series
    for i, disease in enumerate(diseases):
        plot_prev_timeseries(cs, disease, axes[1, i], sti_data)
        if i == 0:
            axes[1, i].legend(fontsize=12, frameon=False)

    fig.tight_layout()

    # Panel labels
    labels = 'ABCDEF'
    positions = [(0.02, 0.94), (0.35, 0.94), (0.68, 0.94),
                 (0.02, 0.47), (0.35, 0.47), (0.68, 0.47)]
    for label, (x, y) in zip(labels, positions):
        pl.figtext(x, y, label, fontsize=32, ha='center', va='center', fontweight='bold')

    pl.savefig(f'{FIGURES_DIR}/sti_epi.png', dpi=200, bbox_inches='tight')
    print(f'Saved {FIGURES_DIR}/sti_epi.png')

    return


if __name__ == '__main__':
    plot_epi_figure()
