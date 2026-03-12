"""
Figure 4: Prior and posterior parameter distributions.

Three panels of subplots:
    A: Calibrated epi parameters — uniform prior (light) + posterior from Optuna (dark)
    B: Birth outcome priors — delivery shifts, growth penalties, treatment reversibility (prior only)
    C: Cost parameter priors (prior only, flagged as placeholders)

Data sources:
    - Calibration priors from priors.py (uniform bounds)
    - Posterior samples from results/zimbabwe_pars.df
    - Prior-only distributions from priors.py
"""

import numpy as np
import sciris as sc
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from scipy import stats
from utils import set_font
from priors import calib_pars, birth_outcome_pars, cost_pars

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

# Colors — distinct hues for each panel
PRIOR_COLOR = '#b0c4de'        # Light steel blue for uniform priors
POSTERIOR_COLOR = '#2c5f8a'    # Dark blue for posteriors
BIRTH_COLOR = '#5a8c6f'        # Sage green for birth outcome priors
COST_COLOR = '#c4956a'         # Tan for cost priors


def plot_calib_panel(axes, posterior_df):
    """Plot calibrated parameters: uniform prior + KDE posterior."""
    for ax, (col, (label, lo, hi, is_log)) in zip(axes.flat, calib_pars.items()):
        x_prior = np.linspace(lo, hi, 200)
        y_prior = np.ones_like(x_prior) / (hi - lo)
        ax.fill_between(x_prior, y_prior, alpha=0.35, color=PRIOR_COLOR, label='Prior')

        samples = posterior_df[col].values
        if is_log:
            log_samples = np.log(samples)
            kde = stats.gaussian_kde(log_samples)
            x_post = np.linspace(lo, hi, 200)
            y_post = kde(np.log(x_post)) / x_post
        else:
            kde = stats.gaussian_kde(samples)
            x_post = np.linspace(lo, hi, 200)
            y_post = kde(x_post)

        ax.fill_between(x_post, y_post, alpha=0.6, color=POSTERIOR_COLOR, label='Posterior')
        ax.plot(x_post, y_post, color=POSTERIOR_COLOR, lw=1.5)
        ax.set_title(label)
        ax.set_yticks([])
        for sp in ['top', 'right', 'left']:
            ax.spines[sp].set_visible(False)


def plot_prior_panel(axes, params, color, placeholder=False):
    """Plot prior-only parameters as density curves."""
    for ax, (name, (label, dist)) in zip(axes.flat, params.items()):
        lo = dist.ppf(0.001)
        hi = dist.ppf(0.999)
        x = np.linspace(lo, hi, 200)
        y = dist.pdf(x)
        ax.fill_between(x, y, alpha=0.45, color=color)
        ax.plot(x, y, color=color, lw=1.5)
        ax.set_title(label + (' *' if placeholder else ''))
        ax.set_yticks([])
        for sp in ['top', 'right', 'left']:
            ax.spines[sp].set_visible(False)

    for ax in axes.flat[len(params):]:
        ax.set_visible(False)


def plot():
    set_font(size=20)
    posterior_df = sc.loadobj(f'{RESULTS_DIR}/zimbabwe_pars.df')

    fig = pl.figure(figsize=(22, 16))

    # Outer grid: left strip for labels + right area for subplots
    gs_outer = GridSpec(3, 2, figure=fig, width_ratios=[0.05, 1],
                        height_ratios=[3, 3, 2], hspace=0.35, wspace=0.02,
                        top=0.97, bottom=0.04, left=0.01, right=0.98)

    # Section labels as colored strips on the far left with rotated text
    panels = [
        ('(A) Epi parameters',          POSTERIOR_COLOR),
        ('(B) Birth outcome priors',     BIRTH_COLOR),
        ('(C) Cost priors *',            COST_COLOR),
    ]
    for row, (label, color) in enumerate(panels):
        ax_label = fig.add_subplot(gs_outer[row, 0])
        ax_label.set_facecolor(color + '40')
        ax_label.text(0.5, 0.5, label, fontsize=20, fontweight='bold',
                      ha='center', va='center', rotation=90,
                      transform=ax_label.transAxes, color=color)
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        for sp in ax_label.spines.values():
            sp.set_visible(False)

    # Panel A: Calibrated parameters (3×4)
    gs_a = gs_outer[0, 1].subgridspec(3, 4, hspace=0.65, wspace=0.25)
    axes_a = np.array([fig.add_subplot(gs_a[i, j]) for i in range(3) for j in range(4)])
    plot_calib_panel(axes_a, posterior_df)
    axes_a[0].legend(loc='upper right', fontsize=14, frameon=False)

    # Panel B: Birth outcome priors (3×4)
    gs_b = gs_outer[1, 1].subgridspec(3, 4, hspace=0.65, wspace=0.25)
    axes_b = np.array([fig.add_subplot(gs_b[i, j]) for i in range(3) for j in range(4)])
    plot_prior_panel(axes_b, birth_outcome_pars, BIRTH_COLOR)

    # Panel C: Cost priors (2×4)
    gs_c = gs_outer[2, 1].subgridspec(2, 4, hspace=0.65, wspace=0.25)
    axes_c = np.array([fig.add_subplot(gs_c[i, j]) for i in range(2) for j in range(4)])
    plot_prior_panel(axes_c, cost_pars, COST_COLOR, placeholder=True)

    sc.savefig(f'{FIGURES_DIR}/fig4_priors_posteriors.png', dpi=200)
    pl.show()
    print('Done.')


if __name__ == '__main__':
    plot()
