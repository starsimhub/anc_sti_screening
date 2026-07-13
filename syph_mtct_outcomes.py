"""Pooled two-seed MTCT-outcomes figure: cascade + treated/untreated heatmaps.

Loads results/syph_preg_log_343000.pkl and results/syph_preg_log_343003.pkl
(produced by diag_syph_preg.py + the PregnancyLog analyzer), pools all
syph-exposed pregnancies delivered/lost in the given window, and produces a
3-panel figure at Zimbabwe scale (pop_scale=870).

  Panel 1: detection cascade per 100 syph+ pregnancies.
  Panel 2: heatmap for TREATED mothers (mother's stage at MTCT x cs_outcome).
  Panel 3: heatmap for UNTREATED mothers (same axes, same color scale).

Usage:
    python syph_mtct_outcomes.py               # default 2020-2025
    python syph_mtct_outcomes.py 2000 2045
"""
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import sciris as sc
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

POP_SCALE = 870
LO = int(sys.argv[1]) if len(sys.argv) > 1 else 2020
HI = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
YEARS = HI - LO

REPO = Path(__file__).resolve().parent

SEEDS = [343000, 343003]
frames = []
for s in SEEDS:
    with open(REPO / 'results' / f'syph_preg_log_{s}.pkl', 'rb') as f:
        d = pickle.load(f)
    fr = pd.DataFrame([r for r in d['records'] if r['outcome'] is not None])
    fr['seed'] = s
    frames.append(fr)
df = pd.concat(frames, ignore_index=True)
N_SEEDS = len(SEEDS)

win = df[(df['outcome_year'] >= LO) & (df['outcome_year'] < HI)].copy()
sp = win[win['syph_positive_ever']].copy()
n_syph = len(sp)
if n_syph == 0:
    print(f'No syph+ pregnancies in window {LO}-{HI}. Aborting.')
    sys.exit(1)

# --- Cascade ---
n_tested   = int((sp['tests'].map(len) > 0).sum())
n_diag     = int(sp['tests'].map(lambda ts: any(t[2] for t in ts)).sum())
n_treat    = int(sp['treated'].sum())
n_treat_normal = int((sp['treated'] & (sp['outcome'] == 'normal')).sum())

# Note: SyphilisANCTimer schedules a visit for every pregnancy, so an
# "ANC-visited" step would be 100% by construction. Actual attendance is
# filtered by SyphTest's per-visit probabilities — captured directly in
# the "Any syph test" bar.
cascade = [
    ('Syph+',                    n_syph),
    ('Any syph\ntest',           n_tested),
    ('Diagnosed',                n_diag),
    ('Treated',                  n_treat),
    ('Treated →\nnormal outcome', n_treat_normal),
]
cascade_pct = [100 * c[1] / n_syph for c in cascade]

# --- Heatmaps ---
STAGES = ['primary', 'secondary', 'early', 'late']
STAGE_LABEL = {'primary': 'Primary', 'secondary': 'Secondary',
               'early': 'Early latent', 'late': 'Late latent'}
OUTCOMES = [1, 2, 3, 4]
OUTCOME_LABEL = {1: 'NND', 2: 'Stillborn', 3: 'Congenital', 4: 'Normal'}


def build_matrix(group):
    m = group[group['cs_outcome'].notna() & group['mtct_stage'].notna()].copy()
    m['cs_int'] = m['cs_outcome'].astype(int)
    if len(m) == 0:
        return pd.DataFrame(0, index=STAGES, columns=OUTCOMES), 0
    mat = m.pivot_table(index='mtct_stage', columns='cs_int',
                        values='mother_uid', aggfunc='count', fill_value=0)
    mat = mat.reindex(index=STAGES, columns=OUTCOMES, fill_value=0)
    return mat, len(m)


treated = sp[sp['treated']]
untreated = sp[~sp['treated']]
mat_t, n_mtct_t = build_matrix(treated)
mat_u, n_mtct_u = build_matrix(untreated)

scale = POP_SCALE / (N_SEEDS * YEARS)
mat_t_yr = mat_t * scale
mat_u_yr = mat_u * scale

# Shared color scale across both heatmaps for comparability
vmax = max(mat_t_yr.values.max(), mat_u_yr.values.max(), 1)

# --- Font ---
FONT = REPO / 'assets' / 'LibertinusSans-Regular.otf'
if FONT.exists():
    sc.fonts(add=str(FONT))
    sc.options(font='Libertinus Sans', fontsize=11)


def fmt_yr(v):
    if v <= 0:
        return '—'
    if v < 100:
        return f'{v:.0f}'
    if v < 10_000:
        return f'{v:,.0f}'
    return f'{v/1000:.1f}k'


def draw_heatmap(ax, mat_yr, mat_raw, title):
    data = mat_yr.values
    ax.imshow(data, aspect='auto', cmap='Reds', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(OUTCOMES)))
    ax.set_xticklabels([OUTCOME_LABEL[o] for o in OUTCOMES], fontsize=9)
    ax.set_yticks(range(len(STAGES)))
    ax.set_yticklabels([STAGE_LABEL[s] for s in STAGES], fontsize=9)
    ax.set_xlabel("Newborn outcome", fontsize=10)
    ax.set_title(title, fontsize=11)

    row_probs = mat_raw.div(mat_raw.sum(axis=1).replace(0, 1), axis=0)
    row_totals = mat_yr.sum(axis=1)
    col_totals = mat_yr.sum(axis=0)
    grand_total = mat_yr.values.sum()

    for i, s_key in enumerate(STAGES):
        for j, o in enumerate(OUTCOMES):
            v = data[i, j]
            p = row_probs.loc[s_key, o]
            color = 'white' if v > vmax * 0.55 else '#222'
            ax.text(j, i - 0.14, fmt_yr(v), ha='center', va='center',
                    fontsize=9.5, color=color, fontweight='bold')
            ax.text(j, i + 0.20, f'({p*100:.0f}%)', ha='center', va='center',
                    fontsize=8, color=color)

    for i, s_key in enumerate(STAGES):
        ax.text(len(OUTCOMES) - 0.35, i, f'  {row_totals[s_key]:,.0f}/yr',
                va='center', ha='left', fontsize=8.5, color='#333')

    n_rows = len(STAGES)
    for j, o in enumerate(OUTCOMES):
        ax.text(j, n_rows - 0.35, f'{col_totals[o]:,.0f}/yr',
                ha='center', va='top', fontsize=8.5, color='#333',
                transform=ax.transData)
    # Grand total at bottom-right
    ax.text(len(OUTCOMES) - 0.35, n_rows - 0.35, f'  {grand_total:,.0f}/yr',
            va='top', ha='left', fontsize=9, color='#000', fontweight='bold',
            transform=ax.transData)
    ax.set_xlim(-0.5, len(OUTCOMES) - 0.5 + 1.0)
    ax.set_ylim(n_rows - 0.15, -0.5)
    ax.tick_params(top=False, bottom=True, left=True, right=False)
    for sp_ in ('top', 'right'):
        ax.spines[sp_].set_visible(False)


# --- Figure: cascade + two heatmaps ---
fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.6, 3, 3])

ax1 = fig.add_subplot(gs[0, 0])
xs = np.arange(len(cascade))
colors = ['#4e79a7', '#79a3c2', '#a3c3d9', '#cfdfec', '#59a14f']
ax1.bar(xs, cascade_pct, color=colors, edgecolor='#333', linewidth=0.5, width=0.7)
for i, (lab, ct) in enumerate(cascade):
    ax1.text(i, cascade_pct[i] + 2, f'{cascade_pct[i]:.0f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xticks(xs)
ax1.set_xticklabels([c[0] for c in cascade], fontsize=9, rotation=25, ha='right')
ax1.set_ylabel('Per 100 syphilis+ pregnancies', fontsize=10)
ax1.set_title(f'Detection cascade under SOC ({n_syph} syph+ preg)', fontsize=11)
ax1.set_ylim(0, 115)
ax1.axhline(100, color='#888', linestyle=':', linewidth=0.7)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

ax2 = fig.add_subplot(gs[0, 1])
draw_heatmap(ax2, mat_t_yr, mat_t, 'TREATED mothers — MTCT events/yr')
ax2.set_ylabel("Mother's stage at MTCT", fontsize=10)

ax3 = fig.add_subplot(gs[0, 2])
draw_heatmap(ax3, mat_u_yr, mat_u, 'UNTREATED mothers — MTCT events/yr')

out = REPO / 'figures' / f'syph_pregnancy_{LO}_{HI}_hot_tx_vs_untx.png'
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=180, bbox_inches='tight')
print(f'Wrote {out}')
print()
print(f'Window {LO}-{HI}, pooled across {N_SEEDS} seeds: {SEEDS}')
print(f'  Syph+ pregnancies:  {n_syph}')
print(f'  TREATED:            {len(treated)} ({100*len(treated)/n_syph:.1f}%)  '
      f'MTCT fired: {n_mtct_t} ({100*n_mtct_t/max(len(treated),1):.1f}%)')
print(f'  UNTREATED:          {len(untreated)} ({100*len(untreated)/n_syph:.1f}%)  '
      f'MTCT fired: {n_mtct_u} ({100*n_mtct_u/max(len(untreated),1):.1f}%)')
print()

def summarize(mat, label):
    tot = mat.sum().sum()
    if tot == 0:
        print(f'  {label}: no events')
        return
    print(f'  {label} (n_raw={tot}):')
    for stage in STAGES:
        s_tot = mat.loc[stage].sum()
        if s_tot == 0: continue
        parts = [f'{OUTCOME_LABEL[o]}={mat.loc[stage, o]} ({100*mat.loc[stage, o]/s_tot:.0f}%)'
                 for o in OUTCOMES if mat.loc[stage, o] > 0]
        print(f'    {stage:<14} n={s_tot:>4}  {", ".join(parts)}')

summarize(mat_t, 'Treated mothers')
summarize(mat_u, 'Untreated mothers')
