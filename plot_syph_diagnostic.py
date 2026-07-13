"""9x5 diagnostic figure: syphilis-in-pregnancy under SOC.

Pools K=5 sims from results/diag_k5_soc.pkl. Window is CLI arg.

Usage:
    python plot_syph_diagnostic.py 2020 2025
    python plot_syph_diagnostic.py 2025 2045
"""
import sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

POP_SCALE = 870
LO = int(sys.argv[1]) if len(sys.argv) > 1 else 2020
HI = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
YEARS = HI - LO

with open('results/diag_k5_soc.pkl', 'rb') as f:
    sims = pickle.load(f)

# Optional seed filter (comma-separated in sys.argv[3])
if len(sys.argv) > 3:
    keep = set(int(s) for s in sys.argv[3].split(','))
    sims = [s for s in sims if s['seed'] in keep]

N_SEEDS = len(sims)

# Combine into one big dataframe, tagged by seed
frames = []
mtct_all = []
tests_tested = {}
tests_positive = {}
treats = {}
for s in sims:
    df_s = pd.DataFrame(s['finished'])
    df_s['seed'] = s['seed']
    df_s['conception_year'] = 1985 + df_s['conception_ti'] / 12
    frames.append(df_s)
    for m in s['mtct_log']:
        mtct_all.append({**m, 'seed': s['seed']})
    # Namespace uids by seed so cross-seed uids don't collide
    # tests_tested / tests_positive may be lists of (source, ti) tuples OR bare ti ints
    def _ti(x): return x[1] if isinstance(x, tuple) else x
    for uid, tis in s['tests_tested'].items():
        tests_tested[(s['seed'], uid)] = [_ti(t) for t in tis]
    for uid, tis in s['tests_positive'].items():
        tests_positive[(s['seed'], uid)] = [_ti(t) for t in tis]
    for uid, tis in s['treats'].items():
        treats[(s['seed'], uid)] = tis

df = pd.concat(frames, ignore_index=True)
df_win = df[(df['conception_year'] >= LO) & (df['conception_year'] < HI)].copy()

def any_in(seed, uid, ci, di, d_):
    return any(ci <= t <= di for t in d_.get((seed, uid), []))

df_win['syph_any'] = (df_win['infected_at_conception'] | df_win['infected_at_delivery']
                      | df_win['mtct_stage'].notna())

def _tested(r):    return any_in(r['seed'], int(r['mother_uid']), int(r['conception_ti']), int(r['delivery_ti']), tests_tested)
def _diagnosed(r): return any_in(r['seed'], int(r['mother_uid']), int(r['conception_ti']), int(r['delivery_ti']), tests_positive)
def _treated(r):   return any_in(r['seed'], int(r['mother_uid']), int(r['conception_ti']), int(r['delivery_ti']), treats)

df_win_syph = df_win[df_win['syph_any']].copy()
df_win_syph['tested']    = df_win_syph.apply(_tested, axis=1)
df_win_syph['diagnosed'] = df_win_syph.apply(_diagnosed, axis=1)
df_win_syph['treated']   = df_win_syph.apply(_treated, axis=1)

n_syph  = len(df_win_syph)
n_tested = int(df_win_syph['tested'].sum())
n_diag   = int(df_win_syph['diagnosed'].sum())
n_treat  = int(df_win_syph['treated'].sum())

cascade = [('Syph+', n_syph), ('Tested', n_tested), ('Diagnosed', n_diag), ('Treated', n_treat)]

# --- MTCT ---
STAGES = ['primary', 'secondary', 'early_latent', 'late_latent']
STAGE_LABEL = {'primary':'Primary', 'secondary':'Secondary',
               'early_latent':'Early latent', 'late_latent':'Late latent'}
OUTCOMES = [1, 2, 3, 4]
OUTCOME_LABEL = {1:'NND', 2:'Stillborn', 3:'Congenital', 4:'Normal'}
OUTCOME_COLOR = {1:'#7a0018', 2:'#d43a2e', 3:'#f28e2b', 4:'#59a14f'}

df_mtct = df_win[df_win['mtct_stage'].notna()]
mat = df_mtct.pivot_table(index='mtct_stage', columns='cs_outcome',
                          values='mother_uid', aggfunc='count', fill_value=0)
mat = mat.reindex(index=STAGES, columns=OUTCOMES, fill_value=0)

# --- Miss attribution ---
adverse = df_mtct[df_mtct['cs_outcome'].isin([1, 2, 3])].copy()
buckets = {'Never tested\nduring preg': 0, 'Tested, always\nnegative': 0,
           'Treated too late\n(after MTCT)': 0, 'Tested positive,\nnot treated': 0,
           'Treated before MTCT,\nbut failed': 0}
for _, r in adverse.iterrows():
    s = r['seed']; u = int(r['mother_uid']); ci = int(r['conception_ti']); di = int(r['delivery_ti']); mt = int(r['mtct_ti'])
    tested_any = any_in(s, u, ci, di, tests_tested)
    pos_any = any_in(s, u, ci, di, tests_positive)
    tx_any = any_in(s, u, ci, di, treats)
    tx_before_mtct = any(ci <= t < mt for t in treats.get((s, u), []))
    if not tested_any: buckets['Never tested\nduring preg'] += 1
    elif not pos_any: buckets['Tested, always\nnegative'] += 1
    elif not tx_any: buckets['Tested positive,\nnot treated'] += 1
    elif not tx_before_mtct: buckets['Treated too late\n(after MTCT)'] += 1
    else: buckets['Treated before MTCT,\nbut failed'] += 1
miss = list(buckets.items())

# Scale: convert per-K-total to per-year Zimbabwe-scale
def scale(n):
    return n * POP_SCALE / (N_SEEDS * YEARS)

# ============ FIGURE ============
fig = plt.figure(figsize=(9, 5), constrained_layout=True)
gs = GridSpec(1, 3, figure=fig, width_ratios=[1.9, 3.4, 2.2])

# Panel 1: cascade
ax1 = fig.add_subplot(gs[0, 0])
labels = [c[0] for c in cascade]
vals = [scale(c[1]) for c in cascade]
colors = ['#4e79a7', '#79a3c2', '#a3c3d9', '#cfdfec']
ax1.bar(range(len(cascade)), vals, color=colors, edgecolor='#333', linewidth=0.5, width=0.7)
for i, v_raw in enumerate([c[1] for c in cascade]):
    pct = v_raw / cascade[0][1] * 100 if cascade[0][1] > 0 else 0
    ax1.text(i, vals[i] + max(vals)*0.025, f'{vals[i]/1000:.1f}k',
             ha='center', fontsize=8, fontweight='bold')
    if i > 0:
        ax1.text(i, vals[i]/2, f'{pct:.0f}%', ha='center', fontsize=9,
                 color='white', fontweight='bold')
ax1.set_xticks(range(len(cascade)))
ax1.set_xticklabels(labels, fontsize=8, rotation=30, ha='right')
ax1.set_ylabel('Pregnancies / year (Zim scale)', fontsize=8)
ax1.set_title('Detection cascade under SOC', fontsize=9.5)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='y', labelsize=7.5)
ax1.set_ylim(0, max(vals) * 1.15 if max(vals) > 0 else 1)

# Panel 2: MTCT stacked bars
ax2 = fig.add_subplot(gs[0, 1])
y_pos = np.arange(len(STAGES))
left = np.zeros(len(STAGES))
totals_scaled = scale(mat.sum(axis=1).values)
max_total = totals_scaled.max() if len(totals_scaled) else 1
for oc in OUTCOMES:
    vals = scale(mat[oc].values)
    ax2.barh(y_pos, vals, left=left, color=OUTCOME_COLOR[oc],
             label=OUTCOME_LABEL[oc], edgecolor='white', linewidth=0.5)
    for i, v in enumerate(vals):
        if v > max_total * 0.04:
            ax2.text(left[i] + v/2, y_pos[i], f'{v:.0f}',
                     ha='center', va='center', fontsize=7.5,
                     color='white' if oc in (1,2,4) else 'black', fontweight='bold')
    left += vals
for i, stage in enumerate(STAGES):
    tot = totals_scaled[i]; n_events = int(mat.loc[stage].sum())
    ax2.text(left[i] + max_total*0.015, y_pos[i],
             f'{tot:.0f}/yr (n={n_events})',
             va='center', fontsize=7.5, color='#333')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([STAGE_LABEL[s] for s in STAGES], fontsize=8.5)
ax2.set_xlabel('Vertical transmission events / year (Zim scale)', fontsize=8)
ax2.set_title("Mother's stage at transmission × newborn outcome", fontsize=9.5)
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', labelsize=7.5)
ax2.set_xlim(0, max_total * 1.42 if max_total > 0 else 1)
ax2.legend(loc='lower right', fontsize=7.5, frameon=False, ncol=2)

# Panel 3: miss attribution
ax3 = fig.add_subplot(gs[0, 2])
labels = [m[0] for m in miss]
vals_raw = [m[1] for m in miss]
vals = [scale(v) for v in vals_raw]
tot = sum(vals_raw) if sum(vals_raw) > 0 else 1
colors = ['#c1272d', '#e67e22', '#95a5a6', '#f39c12', '#7f8c8d']
y_pos = np.arange(len(miss))
ax3.barh(y_pos, vals, color=colors, edgecolor='#333', linewidth=0.5, height=0.7)
for i, (v, v_raw) in enumerate(zip(vals, vals_raw)):
    pct = v_raw / tot * 100
    ax3.text(v + (max(vals) if max(vals) > 0 else 1)*0.03, y_pos[i],
             f'{v:.0f}/yr ({pct:.0f}%)', va='center', fontsize=7.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(labels, fontsize=7.5)
ax3.invert_yaxis()
ax3.set_xlabel('Adverse MTCT outcomes / year', fontsize=8)
ax3.set_title(f'Why adverse outcomes weren\'t averted\n(n={len(adverse)} events)', fontsize=9.5)
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
ax3.tick_params(axis='x', labelsize=7.5)
ax3.set_xlim(0, max(vals) * 1.55 if max(vals) > 0 else 1)

# Callout
n_cong = int(mat[3].sum())
n_cong_known = 0
for _, r in df_mtct[df_mtct['cs_outcome'] == 3].iterrows():
    if any_in(r['seed'], int(r['mother_uid']), int(r['conception_ti']),
              int(r['delivery_ti']), tests_positive):
        n_cong_known += 1
callout = (f'  {n_cong_known} of {n_cong} clinical congenital syphilis cases '
           f'(~{scale(n_cong):.0f} / year at Zimbabwe scale) '
           f'were diagnosed during that pregnancy.  ')
fig.text(0.5, -0.02, callout, ha='center', va='top',
         fontsize=9.5, color='white',
         bbox=dict(boxstyle='round,pad=0.55', facecolor='#7a0018', edgecolor='none'))

fig.suptitle(f'Syphilis in pregnancy under SOC ({LO}-{HI}, K={N_SEEDS} averaged, Zim scale)',
             fontsize=11, y=1.03)

tag = f'_hot' if len(sys.argv) > 3 else ''
out = f'figures/syph_pregnancy_diagnostic_{LO}_{HI}{tag}.png'
fig.savefig(out, dpi=180, bbox_inches='tight')
print(f'Wrote {out}')
print(f'  cascade: syph+={n_syph}, tested={n_tested}, diag={n_diag}, treat={n_treat}')
print(f'  MTCT events: {len(df_mtct)}  ({len(df_mtct)/(N_SEEDS*YEARS):.1f}/sim/yr agent-scale)')
print(f'  adverse: {len(adverse)}, congenital: {n_cong}, known-mother: {n_cong_known}')
