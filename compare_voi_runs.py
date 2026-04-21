"""
Compare two VoI runs.

Prints the provenance of each run (stisim / starsim / repo git state), then
summary decision statistics at standard WTP thresholds, then the delta
between them.

Usage:
    python compare_voi_runs.py path/to/run_a/voi_draws.df path/to/run_b/voi_draws.df

To snapshot a run before a dependency upgrade:
    cp -r results results_snapshots/baseline-<date>-<stisim_sha>

Then after the upgrade, re-run and:
    python compare_voi_runs.py results_snapshots/baseline-*/voi_draws.df results/voi_draws.df
"""

import sys
import numpy as np
import pandas as pd
import sciris as sc

from version_utils import load_meta

WTPS = [1000, 2000, 3000, 5000, 10000]


def summary(df):
    d = df['delta_dalys'].values
    c = df['delta_costs'].values
    rows = []
    for wtp in WTPS:
        nmb = wtp * d - c
        rows.append({
            'WTP':      wtp,
            'P(CE)':    float((nmb > 0).mean()),
            'E[NMB]':   float(nmb.mean()),
            'EVPI':     float(np.mean(np.maximum(nmb, 0)) - max(nmb.mean(), 0)),
        })
    return pd.DataFrame(rows)


def schema_diff(df_a, df_b):
    a, b = set(df_a.columns), set(df_b.columns)
    return {
        'only_in_a':  sorted(a - b),
        'only_in_b':  sorted(b - a),
        'shared':     sorted(a & b),
    }


def _gitstr(g):
    if not g: return 'no git info'
    d = ' (dirty)' if g.get('dirty') else ''
    return f"{g.get('branch','?')}@{g.get('sha','?')[:8]}{d}"


def print_provenance(label, meta):
    print(f'\n--- {label} ---')
    if meta is None:
        print('  (no metadata)')
        return
    print(f"  timestamp    {meta.get('timestamp','?')}")
    sti  = meta.get('stisim')  or {}
    ss   = meta.get('starsim') or {}
    repo = meta.get('anc_sti_repo') or {}
    print(f"  stisim       {sti.get('version','?'):<8} {_gitstr(sti.get('git'))}")
    print(f"  starsim      {ss.get('version','?'):<8} {_gitstr(ss.get('git'))}")
    print(f"  anc_sti_repo          {_gitstr(repo)}")
    if 'run' in meta:
        print(f"  run          {meta['run']}")


def compare(path_a, path_b):
    df_a = sc.loadobj(path_a)
    df_b = sc.loadobj(path_b)
    meta_a = load_meta(path_a)
    meta_b = load_meta(path_b)

    print('=' * 70)
    print(f'A: {path_a}  (n={len(df_a)})')
    print(f'B: {path_b}  (n={len(df_b)})')
    print('=' * 70)
    print_provenance('A provenance', meta_a)
    print_provenance('B provenance', meta_b)

    schema = schema_diff(df_a, df_b)
    if schema['only_in_a'] or schema['only_in_b']:
        print('\n--- Schema changes ---')
        if schema['only_in_a']:
            print(f"  only in A: {schema['only_in_a']}")
        if schema['only_in_b']:
            print(f"  only in B: {schema['only_in_b']}")

    sum_a = summary(df_a)
    sum_b = summary(df_b)
    print('\n--- Run A ---')
    print(sum_a.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))
    print('\n--- Run B ---')
    print(sum_b.to_string(index=False, float_format=lambda x: f'{x:,.2f}'))

    diff = sum_b.copy()
    for col in ['P(CE)', 'E[NMB]', 'EVPI']:
        diff[col] = sum_b[col] - sum_a[col]
    print('\n--- Delta (B − A) ---')
    print(diff.to_string(index=False, float_format=lambda x: f'{x:+,.2f}'))

    icer_a = df_a['delta_costs'] / df_a['delta_dalys']
    icer_b = df_b['delta_costs'] / df_b['delta_dalys']
    print('\n--- ICER distribution ($/DALY) ---')
    print(f"  A: median {icer_a.median():>8,.0f}   mean {icer_a.mean():>8,.0f}   "
          f"5-95% [{icer_a.quantile(0.05):,.0f}, {icer_a.quantile(0.95):,.0f}]")
    print(f"  B: median {icer_b.median():>8,.0f}   mean {icer_b.mean():>8,.0f}   "
          f"5-95% [{icer_b.quantile(0.05):,.0f}, {icer_b.quantile(0.95):,.0f}]")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
