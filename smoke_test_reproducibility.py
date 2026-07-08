"""
Phase-2 reproducibility gate.

Runs 3 draws x 5 seeds through the ported model 1985-2025, extracts
per-draw K=5-mean scalars using sti_notification's own
extract_calibration_summary function (imported directly), and compares
to sti_notification/experiments/06_2026-06-24_kseed_calibration/outputs/per_draw_means.csv.

Pass condition: all compared metrics match within TOLERANCE relative
tolerance. Larger deviation means something in the port has shifted
model dynamics; investigate before proceeding.

Usage:
    conda run -n starsim python smoke_test_reproducibility.py

Environment overrides:
    N_DRAWS   (default 3)
    N_SEEDS   (default 5)
    N_WORKERS (default min(N_DRAWS * N_SEEDS, 20))
    STOP      (default 2025)
    TOLERANCE (default 1e-6)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sti_notification's calibration pipeline importable so we can reuse
# extract_calibration_summary and guarantee byte-for-byte extraction
# consistency with the truth file.
STINOTIF_PIPELINE = Path('/home/robyn/sti_notification/calibration/artifacts/scripts')
if str(STINOTIF_PIPELINE) not in sys.path:
    sys.path.insert(0, str(STINOTIF_PIPELINE))

from _pipeline import extract_calibration_summary  # noqa: E402

from apply_draw import row_to_sim_pars, set_pars_local
from model import make_sim


REPO = Path(__file__).resolve().parent
DRAWS_CSV = REPO / 'data' / 'calibration_draws.csv'
STINOTIF_TRUTH = Path(
    '/home/robyn/sti_notification/experiments/'
    '06_2026-06-24_kseed_calibration/outputs/per_draw_means.csv'
)
RESULTS = REPO / 'results'
RESULTS.mkdir(exist_ok=True)

N_DRAWS   = int(os.environ.get('N_DRAWS', 3))
N_SEEDS   = int(os.environ.get('N_SEEDS', 5))
STOP      = int(os.environ.get('STOP', 2025))
N_AGENTS  = int(os.environ.get('N_AGENTS', 10_000))
TOLERANCE = float(os.environ.get('TOLERANCE', 1e-6))

# Columns of per_draw_means.csv we compare against. These are the ones
# obtainable from a 1985-2025 sim (i.e. exclude pf/ni_2035_2040 which
# need running to 2040).
COMPARE_METRICS = [
    'hiv_prev_15_49_2010_2020',
    'trep_f_2016',
    'nontrep_f_2016',
    'hiv_trep_ratio_2016',
    'fsw_prev_2019',
    'primary_share',
    'secondary_share',
    'early_lat_share',
]


def run_one(args):
    draw_idx, seed, row = args
    sim = make_sim(seed=seed, n_agents=N_AGENTS, start=1985, stop=STOP,
                   which='all', fetal_health=False, verbose=-1)
    sim_pars = row_to_sim_pars(row)
    set_pars_local(sim, sim_pars)
    sim.run()
    summary = extract_calibration_summary(sim, draw_idx, seed)
    # Keep only the fields we care about + identifiers
    out = {'draw_idx': int(draw_idx), 'seed': int(seed)}
    for m in COMPARE_METRICS:
        out[m] = summary.get(m, float('nan'))
    return out


def main():
    if not STINOTIF_TRUTH.exists():
        sys.exit(f'Truth file not found: {STINOTIF_TRUTH}')
    if not DRAWS_CSV.exists():
        sys.exit(f'Draws file not found: {DRAWS_CSV}')

    draws = pd.read_csv(DRAWS_CSV).head(N_DRAWS)
    truth = pd.read_csv(STINOTIF_TRUTH).set_index('draw_idx')

    tasks = []
    for _, row in draws.iterrows():
        d = int(row['draw_idx'])
        for sub in range(N_SEEDS):
            seed = d * 1000 + sub
            tasks.append((d, seed, row.to_dict()))

    n_workers = int(os.environ.get('N_WORKERS', min(len(tasks), 20)))
    print(f'Running {len(tasks)} sims '
          f'({N_DRAWS} draws x {N_SEEDS} seeds) on {n_workers} workers, '
          f'stop={STOP}, n_agents={N_AGENTS} ...')

    with mp.Pool(n_workers) as pool:
        rows = pool.map(run_one, tasks)

    df = pd.DataFrame(rows)
    kmean = df.groupby('draw_idx').mean(numeric_only=True)

    print('\nOur K=5 means:')
    print(kmean[COMPARE_METRICS])

    truth_sub = truth.loc[kmean.index, COMPARE_METRICS]
    print('\nsti_notification truth (same draws):')
    print(truth_sub)

    diffs = (kmean[COMPARE_METRICS] - truth_sub).abs()
    denom = truth_sub.abs().clip(lower=1e-9)
    rel = diffs / denom
    max_rel = float(rel.max().max())

    print(f'\nMax abs diff:      {float(diffs.max().max()):.4e}')
    print(f'Max rel diff:      {max_rel:.4e}')
    print(f'Tolerance:         {TOLERANCE:.4e}')

    out_csv = RESULTS / 'smoke_test_results.csv'
    combined = kmean[COMPARE_METRICS].copy()
    combined.columns = [f'ours_{c}' for c in combined.columns]
    for c in COMPARE_METRICS:
        combined[f'truth_{c}'] = truth_sub[c]
    combined.to_csv(out_csv)
    print(f'\nWrote {out_csv}')

    if max_rel < TOLERANCE:
        print('\nPASS: reproduction within tolerance')
        return 0
    else:
        print(f'\nFAIL: reproduction diverges beyond tolerance ({max_rel:.4e} > {TOLERANCE:.4e})')
        # Show per-metric relative diffs to help diagnose
        print('\nRelative diffs by metric:')
        print(rel)
        return 1


if __name__ == '__main__':
    sys.exit(main())
