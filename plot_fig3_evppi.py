"""
Figure 3: EVPPI tornado chart.

Horizontal bars sorted by EVPPI value for each parameter group.
Shows which parameters drive decision uncertainty and where further
information would be most valuable.

Key message: which parameters drive decision uncertainty? Do reversibility
or cost parameters dominate?

Data source: results/voi_draws.df from run_voi.py
Requires: scikit-learn for gradient-boosted regression (Strong et al. 2014)
"""

import numpy as np
import sciris as sc
import pandas as pd
import matplotlib.pyplot as pl
from utils import set_font

RESULTS_DIR = 'results'
FIGURES_DIR = 'figures'

# Default WTP for EVPPI computation
DEFAULT_WTP = 1000

# Parameter groups for EVPPI (name → list of column prefixes/names)
PARAM_GROUPS = sc.objdict({
    'Network structure':       ['structuredsexual.prop_f0', 'structuredsexual.prop_m0', 'structuredsexual.m1_conc'],
    'NG transmission':         ['ng.beta_m2f', 'ng.p_symp'],
    'CT transmission':         ['ct.beta_m2f', 'ct.p_symp'],
    'TV transmission':         ['tv.beta_m2f', 'tv.p_symp'],
    'Delivery timing shifts':  ['ptb_shift_ng', 'ptb_shift_ct', 'ptb_shift_tv', 'ptb_shift_std'],
    'Growth penalties':        ['growth_penalty_ng', 'growth_penalty_ct', 'growth_penalty_tv'],
    'Tx reversibility (T1)': ['tx_residual_growth_tri1', 'tx_residual_timing_tri1'],
    'Tx reversibility (T2)': ['tx_residual_growth_tri2', 'tx_residual_timing_tri2'],
    'Tx reversibility (T3)': ['tx_residual_growth_tri3', 'tx_residual_timing_tri3'],
    'Cost parameters':         ['cost_poc_test', 'cost_tx_ng', 'cost_tx_ct', 'cost_tx_tv',
                                'cost_anc_visit', 'cost_ptb_mgmt', 'cost_lbw_mgmt', 'cost_partner_notif'],
})

# Colors for groups (epi vs birth outcome vs cost)
GROUP_COLORS = {
    'Network structure':        '#4a90d9',
    'NG transmission':          '#4a90d9',
    'CT transmission':          '#4a90d9',
    'TV transmission':          '#4a90d9',
    'Delivery timing shifts':   '#5a8c6f',
    'Growth penalties':         '#5a8c6f',
    'Tx reversibility (T1)': '#5a8c6f',
    'Tx reversibility (T2)': '#5a8c6f',
    'Tx reversibility (T3)': '#5a8c6f',
    'Cost parameters':          '#c4956a',
}


def compute_evppi_group(draws_df, group_cols, wtp=DEFAULT_WTP, n_folds=5):
    """
    Estimate EVPPI for a parameter group using gradient-boosted regression
    (Strong et al. 2014 nonparametric method).

    EVPPI(φ) = E[max(E[NMB|φ], 0)] − max(E[NMB], 0)

    Uses cross-validated predictions to avoid overfitting.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    nmb = (wtp * draws_df['delta_dalys'] - draws_df['delta_costs']).values

    # Get the parameter columns that exist in the data
    available = [c for c in group_cols if c in draws_df.columns]
    if not available:
        return 0.0

    X = draws_df[available].values
    n = len(nmb)

    # Cross-validated predictions of E[NMB | φ]
    cv_preds = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        gbr = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=42, subsample=0.8,
        )
        gbr.fit(X[train_idx], nmb[train_idx])
        cv_preds[test_idx] = gbr.predict(X[test_idx])

    # EVPPI = E[max(E[NMB|φ], 0)] - max(E[NMB], 0)
    evppi = np.mean(np.maximum(cv_preds, 0)) - max(np.mean(nmb), 0)
    return max(evppi, 0)  # Floor at 0


def compute_all_evppi(draws_df, wtp=DEFAULT_WTP):
    """Compute EVPPI for all parameter groups."""
    results = []
    for group_name, cols in PARAM_GROUPS.items():
        evppi = compute_evppi_group(draws_df, cols, wtp=wtp)
        results.append(dict(group=group_name, evppi=evppi))
        print(f'  {group_name}: EVPPI = ${evppi:,.2f}')
    return pd.DataFrame(results).sort_values('evppi', ascending=True)


def plot(draws_df=None, wtp=DEFAULT_WTP):
    if draws_df is None:
        draws_df = sc.loadobj(f'{RESULTS_DIR}/voi_draws.df')

    print(f'Computing EVPPI at WTP=${wtp:,}/DALY...')
    evppi_df = compute_all_evppi(draws_df, wtp=wtp)

    # Also load EVPI for context
    try:
        evpi_df = sc.loadobj(f'{RESULTS_DIR}/voi_evpi.df')
        evpi_row = evpi_df[evpi_df.wtp == wtp]
        evpi_val = float(evpi_row.evpi.values[0]) if len(evpi_row) else None
    except FileNotFoundError:
        evpi_val = None

    set_font(size=18)
    fig, ax = pl.subplots(figsize=(10, 7))

    colors = [GROUP_COLORS.get(g, '#888888') for g in evppi_df['group']]
    bars = ax.barh(evppi_df['group'], evppi_df['evppi'], color=colors, alpha=0.85,
                   edgecolor='white', linewidth=0.5)

    # Add EVPI reference line
    if evpi_val is not None:
        ax.axvline(evpi_val, color='k', linewidth=1.5, linestyle='--', alpha=0.6)
        ax.text(evpi_val, len(evppi_df) - 0.5, f'  EVPI = ${evpi_val:,.2f}',
                fontsize=13, va='bottom', ha='left')

    # Value labels on bars
    for bar, val in zip(bars, evppi_df['evppi']):
        if val > 0:
            ax.text(bar.get_width() + ax.get_xlim()[1] * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'${val:,.2f}', va='center', ha='left', fontsize=13)

    ax.set_xlabel('EVPPI ($/woman)')
    ax.set_title(f'Expected value of partial perfect information (WTP = ${wtp:,}/DALY)')

    # Legend for group types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4a90d9', alpha=0.85, label='Epi parameters'),
        Patch(facecolor='#5a8c6f', alpha=0.85, label='Birth outcome parameters'),
        Patch(facecolor='#c4956a', alpha=0.85, label='Cost parameters'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=13)

    fig.tight_layout()

    # Save both the figure and the EVPPI results
    sc.savefig(f'{FIGURES_DIR}/fig3_evppi.png', dpi=200)
    sc.saveobj(f'{RESULTS_DIR}/voi_evppi.df', evppi_df)
    print(f'Saved {FIGURES_DIR}/fig3_evppi.png')
    print(f'Saved {RESULTS_DIR}/voi_evppi.df')

    return fig, evppi_df


if __name__ == '__main__':
    plot()
