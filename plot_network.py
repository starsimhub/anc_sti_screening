"""
Network structure figure (supplementary).

2x3 grid showing StructuredSexual network properties:
    A: Lifetime partner distribution by sex
    B: Age mixing pattern (age difference by female age group)
    C: Risk group composition by sex
    D: Cumulative distribution of age at sexual debut
    E: Female partnership status by age
    F: Condom use over time by partnership type
"""

import sciris as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from utils import set_font

RESULTS_DIR = 'results'
DATA_DIR = 'data'
FIGURES_DIR = 'figures'

# Colors
F_COLOR = '#d46e9c'
M_COLOR = '#4a90d9'
RG_COLORS = ['#4daf4a', '#ff7f00', '#e41a1c']  # Low, Mid, High risk
RG_LABELS = ['Low risk', 'Medium risk', 'High risk']


def plot_lifetime_partners(data, ax):
    """ Panel A: Lifetime partner distribution by sex (debuted agents only) """
    max_partners = 20
    bins = np.arange(max_partners + 2) - 0.5
    width = 0.35
    bin_centers = np.arange(max_partners + 1)

    f_vals = np.clip(data.lifetime_partners_f, 0, max_partners)
    m_vals = np.clip(data.lifetime_partners_m, 0, max_partners)

    f_counts, _ = np.histogram(f_vals, bins=bins)
    m_counts, _ = np.histogram(m_vals, bins=bins)
    f_prop = f_counts / f_counts.sum()
    m_prop = m_counts / m_counts.sum()

    ax.bar(bin_centers - width/2, f_prop, width, color=F_COLOR, alpha=0.85, label='Female')
    ax.bar(bin_centers + width/2, m_prop, width, color=M_COLOR, alpha=0.85, label='Male')

    ax.set_xlabel('Lifetime partners')
    ax.set_ylabel('Proportion')
    ax.set_title('Lifetime partner\ndistribution')
    ax.set_xlim(-1, max_partners + 1)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels(['0', '5', '10', '15', '20+'])
    ax.legend(frameon=False, fontsize=12)

    return


def plot_age_mixing(data, ax):
    """ Panel B: Age mixing pattern (age difference by female age group) """
    labels = {'teens': '<20', 'young': '20\u201325', 'adult': '25+'}
    colors = ['#4daf4a', '#ff7f00', '#984ea3']

    plot_data = []
    for key in labels:
        vals = data.age_diffs.get(key, np.array([]))
        plot_data.append(vals)

    parts = ax.violinplot(plot_data, positions=[1, 2, 3], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(list(labels.values()))
    ax.set_xlabel('Female age group')
    ax.set_ylabel('Age difference (M \u2212 F)')
    ax.set_title('Age mixing pattern')
    ax.set_ylim(top=15)
    ax.axhline(0, color='grey', linestyle='--', alpha=0.3)

    return


def plot_risk_groups(data, ax):
    """ Panel C: Risk group composition by sex """
    rg = data.risk_group_data
    sexes = ['Female', 'Male']
    x = np.arange(len(sexes))
    width = 0.5
    bottom = np.zeros(len(sexes))

    for rg_idx in [0, 1, 2]:
        vals = np.array([rg[(sex, rg_idx)] / rg[(sex, 'total')] * 100 for sex in sexes])
        ax.bar(x, vals, width, bottom=bottom, color=RG_COLORS[rg_idx],
               label=RG_LABELS[rg_idx], alpha=0.85)
        for xi, v in zip(x, vals):
            if v > 5:
                ax.text(xi, bottom[xi] + v/2, f'{v:.0f}%', ha='center', va='center', fontsize=12)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(sexes)
    ax.set_ylabel('Proportion (%)')
    ax.set_title('Risk group\ncomposition')
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, fontsize=11, loc='upper right')

    return


def plot_debut_age(data, ax):
    """ Panel D: Cumulative distribution of age at sexual debut """
    for sex_label, color in [('Female', F_COLOR), ('Male', M_COLOR)]:
        ages = np.sort(data.debut_data[sex_label])
        cdf = np.arange(1, len(ages) + 1) / len(ages)
        ax.plot(ages, cdf, color=color, linewidth=2, label=sex_label)

        for q in [0.25, 0.50, 0.75]:
            val = np.percentile(ages, q * 100)
            ax.plot(val, q, 'o', color=color, markersize=6, zorder=5)

    f_ages = data.debut_data['Female']
    m_ages = data.debut_data['Male']
    f25, f50, f75 = [np.percentile(f_ages, q) for q in [25, 50, 75]]
    m25, m50, m75 = [np.percentile(m_ages, q) for q in [25, 50, 75]]

    ax.annotate(f'F: {f25:.1f}  M: {m25:.1f}', xy=(max(f25, m25) + 0.5, 0.25),
                fontsize=11, va='center', ha='left')
    ax.annotate(f'F: {f50:.1f}  M: {m50:.1f}', xy=(max(f50, m50) + 0.5, 0.50),
                fontsize=11, va='center', ha='left')
    ax.annotate(f'F: {f75:.1f}  M: {m75:.1f}', xy=(max(f75, m75) + 0.5, 0.75),
                fontsize=11, va='center', ha='left')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Cumulative proportion')
    ax.set_title('Age at sexual debut')
    ax.set_xlim(12, 35)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='grey', linestyle='--', alpha=0.3, linewidth=0.8)
    ax.legend(frameon=False, fontsize=12)

    return


def plot_partnership_by_age(data, ax):
    """ Panel E: Female partnership status by age """
    pba = data.partnership_by_age
    ages = pba['age_bins']

    ax.plot(ages, pba['prop_stable'] * 100, color='#2171b5', linewidth=2, label='Stable partner')
    ax.plot(ages, pba['prop_casual'] * 100, color='#ff7f00', linewidth=2, label='1+ casual partner')

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Proportion of females (%)')
    ax.set_title('Female partnership\nstatus by age')
    ax.set_xlim(15, 50)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=11)

    return


def plot_condom_use(ax):
    """ Panel F: Condom use over time by partnership type """
    df = pd.read_csv(f'{DATA_DIR}/condom_use.csv')
    year_cols = [c for c in df.columns if c != 'partnership']
    years = [int(c) for c in year_cols]

    groups = {
        'Stable (low risk)': ['(0,0)'],
        'Cross-risk':        ['(0,1)', '(0,2)', '(1,0)', '(1,2)', '(2,0)', '(2,1)'],
        'High risk':         ['(1,1)', '(2,2)'],
        'FSW\u2013client':   ['(fsw,client)'],
    }
    group_colors = ['#4daf4a', '#ff7f00', '#e41a1c', '#984ea3']

    for (label, pairings), color in zip(groups.items(), group_colors):
        subset = df[df.partnership.isin(pairings)]
        vals = subset[year_cols].mean(axis=0).values
        ax.plot(years, vals, color=color, linewidth=2, label=label, marker='o', markersize=4)

    ax.set_xlabel('Year')
    ax.set_ylabel('Condom use probability')
    ax.set_title('Condom use over time')
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=10, loc='upper left')

    return


if __name__ == '__main__':

    data = sc.loadobj(f'{RESULTS_DIR}/network_data.obj')

    set_font(size=18)
    fig = pl.figure(figsize=(22, 14))
    gs = GridSpec(2, 3, left=0.06, right=0.98, bottom=0.06, top=0.93,
                  wspace=0.28, hspace=0.38)

    plot_lifetime_partners(data, fig.add_subplot(gs[0, 0]))
    plot_age_mixing(data, fig.add_subplot(gs[0, 1]))
    plot_risk_groups(data, fig.add_subplot(gs[0, 2]))
    plot_debut_age(data, fig.add_subplot(gs[1, 0]))
    plot_partnership_by_age(data, fig.add_subplot(gs[1, 1]))
    plot_condom_use(fig.add_subplot(gs[1, 2]))

    for i, ax in enumerate(fig.axes):
        ax.text(-0.10, 1.06, chr(65 + i), transform=ax.transAxes,
                fontsize=28, fontweight='bold', va='top')

    pl.savefig(f'{FIGURES_DIR}/network.png', dpi=200, bbox_inches='tight')
    print(f'Saved {FIGURES_DIR}/network.png')
