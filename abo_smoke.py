"""
Small-N ABO attribution driver: 3 scenarios × 4 effect-size regimes × K seeds,
1985-2045, 10k agents. K=1 for the smoke; bump to 5 for the deliverable.

Writes:
  results/abo_attribution.csv  — per (scenario × assumption × seed × outcome × disease)
  results/abo_totals.csv        — per (scenario × assumption × seed) totals
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios import build_scenario_sim, EFFECT_SIZE_ASSUMPTIONS


REPO = Path(__file__).resolve().parent
STOP = 2045
WINDOW_START = 2028
N_AGENTS = 10_000
SCENARIOS = ['soc', 'anc_1screen_90cov', 'anc_2screen_90cov']
ASSUMPTIONS = list(EFFECT_SIZE_ASSUMPTIONS.keys())  # 4 regimes
DISEASES = ('ng', 'ct', 'tv', 'syph')
K_SEEDS = 1  # bump to 5 for the deliverable


def _cum(res, key, mask):
    r = res.get(key)
    if r is None:
        return 0
    return int(np.asarray(r.values)[mask].sum())


def run_one(args):
    scenario_id, assumption_id, sub_seed = args
    df = pd.read_csv(REPO / 'data' / 'calibration_draws.csv')
    row = df.iloc[0].to_dict()
    seed = int(row['draw_idx']) * 1000 + sub_seed
    t0 = time.time()
    sim = build_scenario_sim(
        seed=seed, scenario_id=scenario_id, assumption_id=assumption_id,
        draw_row=row, start=1985, stop=STOP, n_agents=N_AGENTS,
        bundled_prevention=False,
    )
    sim.run()
    elapsed = time.time() - t0

    yrs = np.asarray([t.year for t in sim.results.timevec])
    mask = yrs >= WINDOW_START

    attrib = sim.analyzers['birth_outcome_attribution'].results
    dalys  = sim.analyzers['birth_outcome_dalys'].results

    tag = dict(scenario=scenario_id, assumption=assumption_id, seed=seed)
    rows = []
    for outcome in ('ptb', 'lbw'):
        for d in DISEASES:
            rows.append({**tag, 'outcome': outcome, 'disease': d,
                'n_total': _cum(attrib, f'n_{outcome}_{d}', mask),
                'n_sole':  _cum(attrib, f'n_{outcome}_sole_{d}', mask),
                'n_shared':_cum(attrib, f'n_{outcome}_shared_{d}', mask)})
        rows.append({**tag, 'outcome': outcome, 'disease': 'none',
            'n_total': _cum(attrib, f'n_{outcome}_no_attribution', mask),
            'n_sole':  _cum(attrib, f'n_{outcome}_no_attribution', mask),
            'n_shared': 0})
    rows.append({**tag, 'outcome': 'stillbirth', 'disease': 'syph',
        'n_total': _cum(attrib, 'n_stillbirth_syph', mask),
        'n_sole':  _cum(attrib, 'n_stillbirth_syph', mask), 'n_shared': 0})
    rows.append({**tag, 'outcome': 'nnd', 'disease': 'syph',
        'n_total': _cum(attrib, 'n_nnd_syph', mask),
        'n_sole':  _cum(attrib, 'n_nnd_syph', mask), 'n_shared': 0})

    totals = {**tag, 'wall_sec': round(elapsed, 1),
        'n_deliveries':  _cum(dalys, 'n_deliveries', mask),
        'n_ptb':         _cum(dalys, 'n_ptb', mask),
        'n_lbw':         _cum(dalys, 'n_lbw', mask),
        'n_ptb_only':    _cum(dalys, 'n_ptb_only', mask),
        'n_lbw_only':    _cum(dalys, 'n_lbw_only', mask),
        'n_ptb_and_lbw': _cum(dalys, 'n_ptb_and_lbw', mask),
        'n_stillbirths': _cum(dalys, 'n_stillbirths', mask),
        'dalys': int(np.asarray(dalys['dalys'].values)[mask].sum())}
    return rows, totals


def main():
    tasks = [(sid, aid, k)
             for sid in SCENARIOS
             for aid in ASSUMPTIONS
             for k in range(K_SEEDS)]
    t0 = time.time()
    workers = min(len(tasks), 80)
    with mp.Pool(workers) as pool:
        results = pool.map(run_one, tasks)

    attrib_rows, total_rows = [], []
    for rows, totals in results:
        attrib_rows.extend(rows)
        total_rows.append(totals)

    (REPO / 'results').mkdir(exist_ok=True)
    pd.DataFrame(attrib_rows).to_csv(REPO / 'results' / 'abo_attribution.csv', index=False)
    totals_df = pd.DataFrame(total_rows)
    totals_df.to_csv(REPO / 'results' / 'abo_totals.csv', index=False)

    print(f'\nAll {len(tasks)} sims done in {time.time()-t0:.0f}s ({workers} workers)')
    print(f'\n=== ABO totals (median across K={K_SEEDS}, cumulative {WINDOW_START}-{STOP}) ===')
    pivot = totals_df.groupby(['assumption', 'scenario'])[['n_ptb', 'n_lbw', 'n_stillbirths', 'dalys']].median().unstack(0)
    print(pivot.to_string())


if __name__ == '__main__':
    main()
