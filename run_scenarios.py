"""
Dispatch the (draw × seed × scenario × assumption) grid.

Builds a sim via scenarios.build_scenario_sim, runs it,
extracts scalar summary, writes one JSON row to results/scenarios.jsonl.

Usage:
    # Small-N validation: 1 draw × 1 seed × 7 scenarios × 4 assumptions = 28 sims
    N_DRAWS=1 N_SEEDS=1 python run_scenarios.py

    # Full first run: 5 draws × 5 seeds × 7 scenarios × 4 assumptions = 700 sims
    N_DRAWS=5 N_SEEDS=5 python run_scenarios.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios import (INTERVENTION_SCENARIOS, EFFECT_SIZE_ASSUMPTIONS,
                        build_scenario_sim)


REPO = Path(__file__).resolve().parent
RESULTS = REPO / 'results'
RESULTS.mkdir(exist_ok=True)

DRAWS_CSV = REPO / 'data' / 'calibration_draws.csv'
N_DRAWS   = int(os.environ.get('N_DRAWS', 5))
N_SEEDS   = int(os.environ.get('N_SEEDS', 5))
N_WORKERS = int(os.environ.get('N_WORKERS', min(80, mp.cpu_count())))
START     = int(os.environ.get('START', 1985))
STOP      = int(os.environ.get('STOP', 2045))
N_AGENTS  = int(os.environ.get('N_AGENTS', 10_000))


def _endpoint_sum(res, path, field):
    """Safe endpoint sum: returns nan if path or field missing."""
    try:
        cur = res
        for step in path:
            cur = cur[step]
        return float(np.sum(cur[field].values))
    except (KeyError, AttributeError, TypeError):
        return float('nan')


def extract_scalars(sim, cell_id, assumption_id, draw_idx, seed):
    """Pull the small scalar summary needed for the JSONL archive."""
    r = sim.results
    return {
        'draw_idx': int(draw_idx),
        'seed': int(seed),
        'scenario_id': cell_id,
        'assumption_id': assumption_id,
        # ABO
        'n_deliveries':  _endpoint_sum(r, ['birth_outcome_dalys'], 'n_deliveries'),
        'n_ptb':         _endpoint_sum(r, ['birth_outcome_dalys'], 'n_ptb'),
        'n_lbw':         _endpoint_sum(r, ['birth_outcome_dalys'], 'n_lbw'),
        'n_stillbirths': _endpoint_sum(r, ['birth_outcome_dalys'], 'n_stillbirths'),
        'dalys':         _endpoint_sum(r, ['birth_outcome_dalys'], 'dalys'),
        # Epi endpoints (last-year values)
        'hiv_prev_final':  float(r['hiv']['prevalence'].values[-1]) if 'hiv' in r else float('nan'),
        'syph_prev_final': float(r['syph']['prevalence_f'].values[-1]) if 'syph' in r else float('nan'),
        'ng_prev_final':   float(r['ng']['prevalence_f'].values[-1]) if 'ng' in r else float('nan'),
        'ct_prev_final':   float(r['ct']['prevalence_f'].values[-1]) if 'ct' in r else float('nan'),
        'tv_prev_final':   float(r['tv']['prevalence_f'].values[-1]) if 'tv' in r else float('nan'),
    }


def run_one(task):
    draw_idx, seed, scenario_id, assumption_id, row = task
    try:
        sim = build_scenario_sim(
            seed=seed, scenario_id=scenario_id,
            assumption_id=assumption_id, draw_row=row,
            start=START, stop=STOP, n_agents=N_AGENTS,
        )
        sim.run()
        return extract_scalars(sim, scenario_id, assumption_id, draw_idx, seed)
    except Exception as e:
        import traceback
        return {'draw_idx': int(draw_idx), 'seed': int(seed),
                 'scenario_id': scenario_id, 'assumption_id': assumption_id,
                 'error': f'{type(e).__name__}: {e}',
                 'traceback': traceback.format_exc()}


def main():
    draws = pd.read_csv(DRAWS_CSV).head(N_DRAWS)
    tasks = []
    for _, row in draws.iterrows():
        d = int(row['draw_idx'])
        for sub in range(N_SEEDS):
            seed = d * 1000 + sub
            for sc_id in INTERVENTION_SCENARIOS.keys():
                for ax_id in EFFECT_SIZE_ASSUMPTIONS.keys():
                    tasks.append((d, seed, sc_id, ax_id, row.to_dict()))

    print(f'Grid: {len(draws)} draws × {N_SEEDS} seeds × '
          f'{len(INTERVENTION_SCENARIOS)} scenarios × '
          f'{len(EFFECT_SIZE_ASSUMPTIONS)} assumptions = {len(tasks)} sims '
          f'| workers={N_WORKERS}')

    t0 = time.time()
    out_path = RESULTS / 'scenarios.jsonl'
    with open(out_path, 'w') as f, mp.Pool(N_WORKERS) as pool:
        for i, row_out in enumerate(pool.imap(run_one, tasks, chunksize=1)):
            f.write(json.dumps(row_out) + '\n')
            f.flush()
            if (i+1) % 10 == 0 or i+1 == len(tasks):
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                eta = (len(tasks) - i - 1) / rate if rate > 0 else float('inf')
                print(f'  {i+1}/{len(tasks)} sims done '
                      f'({elapsed:.0f}s, {rate:.2f} sims/s, ETA {eta/60:.1f} min)',
                      flush=True)

    print(f'Done: {len(tasks)} sims in {time.time()-t0:.0f}s. Wrote {out_path}.')


if __name__ == '__main__':
    sys.exit(main())
