"""
Aggregate scenarios.jsonl into K-averaged scalars.

Reads results/scenarios.jsonl (one row per sim); groups by
(scenario_id, assumption_id, draw_idx); computes seed-mean of every
scalar column; writes results/scenarios.kavg.csv.

Usage:
    python aggregate_scenarios.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO    = Path(__file__).resolve().parent
IN_PATH = REPO / 'results' / 'scenarios.jsonl'
OUT_CSV = REPO / 'results' / 'scenarios.kavg.csv'


def main():
    if not IN_PATH.exists():
        raise SystemExit(f'Not found: {IN_PATH}')

    df = pd.read_json(IN_PATH, lines=True)
    print(f'Loaded {len(df)} rows from {IN_PATH.name}')

    # Filter out any error rows
    if 'error' in df.columns:
        errors = df[df['error'].notna()]
        if len(errors):
            print(f'WARN: {len(errors)} sims errored; dropping from kavg.')
            for _, e in errors.head(3).iterrows():
                print(f'  {e["scenario_id"]}/{e["assumption_id"]}/d{e["draw_idx"]}/s{e["seed"]}: {e["error"]}')
        df = df[df['error'].isna()]

    key_cols = ['scenario_id', 'assumption_id', 'draw_idx']
    num_cols = [c for c in df.columns
                 if c not in key_cols + ['seed', 'error', 'traceback']
                 and pd.api.types.is_numeric_dtype(df[c])]

    kavg = df.groupby(key_cols)[num_cols].mean().reset_index()
    kavg.to_csv(OUT_CSV, index=False)
    print(f'Wrote {OUT_CSV}: {len(kavg)} rows.')

    n_cells = kavg[['scenario_id', 'assumption_id']].drop_duplicates()
    print(f'Cells: {len(n_cells)} (expected 7 × 4 = 28).')


if __name__ == '__main__':
    main()
