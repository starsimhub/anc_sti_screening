"""
Plot prior predictive coverage vs calibration data.

Reads results from run.py and produces one panel per calibration target showing
the 5th–95th percentile envelope from the prior draws alongside the data points.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sciris as sc

RESULTS = Path(__file__).parent / 'results'
DATA    = Path(__file__).resolve().parents[2] / 'data' / 'zimbabwe_sti_data.csv'

TARGETS = {
    'ng.prevalence':       'NG prevalence',
    'ct.prevalence_f_25_30': 'CT prevalence (F 25–30)',
    'tv.prevalence':       'TV prevalence',
}


def load_results():
    dfs   = sc.load(RESULTS / 'coverage_dfs.obj')
    data  = pd.read_csv(DATA).set_index('time')
    return dfs, data


def make_envelope(dfs, col):
    """ Stack all draws for a column and compute percentile bands by year. """
    wide = pd.concat([df.set_index('time')[col].rename(i)
                      for i, df in enumerate(dfs) if col in df.columns], axis=1)
    return wide.quantile([0.05, 0.25, 0.5, 0.75, 0.95], axis=1).T


def plot_coverage(dfs, data):
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4), sharey=False)

    for ax, (col, label) in zip(axes, TARGETS.items()):
        env = make_envelope(dfs, col)

        ax.fill_between(env.index, env[0.05], env[0.95], alpha=0.2, color='steelblue', label='5–95%')
        ax.fill_between(env.index, env[0.25], env[0.75], alpha=0.3, color='steelblue', label='25–75%')
        ax.plot(env.index, env[0.5], color='steelblue', lw=1.5, label='median')

        if col in data.columns:
            ax.scatter(data.index, data[col], color='black', s=20, zorder=5, label='data')

        ax.set_title(label)
        ax.set_xlabel('Year')
        ax.set_xlim(2000, 2026)

    axes[0].set_ylabel('Prevalence')
    axes[-1].legend(fontsize=8, loc='upper right')
    fig.suptitle('Prior predictive coverage check', fontsize=12)
    fig.tight_layout()

    out = RESULTS / 'coverage_check.png'
    fig.savefig(out, dpi=150)
    print(f'Saved {out}')
    return fig


if __name__ == '__main__':
    dfs, data = load_results()
    plot_coverage(dfs, data)
    plt.show()
