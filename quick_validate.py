"""
Quick validation with rich per-disease diagnostics.

Runs SOC + two intervention arms (1-screen 90%, 2-screen 90%) at central
assumption, 1 seed, 1985-2032. Reports:
- ABO counts, DALYs
- ANC screen activity per disease
- Per-STI prevalence at end of sim
- Treatment-intervention totals

Sanity indicators:
- SOC and intervention arms should differ meaningfully
- 1-screen vs 2-screen should differ (since window fix)
- n_positive should be non-zero and roughly match sum of per-disease detections
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios import build_scenario_sim


REPO = Path(__file__).resolve().parent
STOP = 2032
N_AGENTS = 10_000
ASSUMPTION = 'central_reversible'
SCENARIOS = ['soc', 'anc_1screen_90cov', 'anc_2screen_90cov']


def _sum(res, key):
    r = res.get(key)
    if r is None:
        return 0
    try:
        return int(np.asarray(r.values).sum())
    except Exception:
        return 0


def _prev_final(sim, disease):
    r = sim.results.get(disease)
    if r is None:
        return float('nan')
    prev = r.get('prevalence_f')
    if prev is None:
        prev = r.get('prevalence')
    if prev is None:
        return float('nan')
    return float(np.asarray(prev.values)[-1])


def run_one(scenario_id):
    df = pd.read_csv(REPO / 'data' / 'calibration_draws.csv')
    row = df.iloc[0].to_dict()
    t0 = time.time()
    sim = build_scenario_sim(
        seed=int(row['draw_idx']) * 1000,
        scenario_id=scenario_id,
        assumption_id=ASSUMPTION,
        draw_row=row,
        start=1985, stop=STOP, n_agents=N_AGENTS,
        bundled_prevention=True,
    )
    sim.run()
    elapsed = time.time() - t0

    ana = sim.analyzers['birth_outcome_dalys']
    ana_res = ana.results
    result = {
        'scenario': scenario_id,
        'wall_sec': round(elapsed, 1),
        'n_deliveries': _sum(ana_res, 'n_deliveries'),
        'n_ptb':        _sum(ana_res, 'n_ptb'),
        'n_lbw':        _sum(ana_res, 'n_lbw'),
        'n_still':      _sum(ana_res, 'n_stillbirths'),
        'dalys':        float(np.asarray(ana_res['dalys'].values).sum()),
    }

    # Per-STI end-of-sim prevalence in females
    for d in ('hiv', 'syph', 'ng', 'ct', 'tv', 'bv'):
        result[f'{d}_prev_final'] = _prev_final(sim, d)

    # ANC screen activity: per-disease detections
    for name in ('anc_enroll', 'anc_tri3'):
        intv = sim.interventions.get(name)
        if intv is None:
            result[f'{name}_screened'] = 0
            result[f'{name}_positive'] = 0
            for d in ('ng', 'ct', 'tv'):
                result[f'{name}_{d}_true_pos'] = 0
            continue
        result[f'{name}_screened'] = _sum(intv.results, 'n_screened')
        result[f'{name}_positive'] = _sum(intv.results, 'n_positive')
        for d in ('ng', 'ct', 'tv'):
            result[f'{name}_{d}_true_pos']  = _sum(intv.results, f'n_{d}_true_pos')

    # Treatment intervention totals — how many times each tx dispatched
    for tx_name in ('ng_tx', 'ct_tx', 'metronidazole', 'syph_tx'):
        tx = sim.interventions.get(tx_name)
        if tx is None:
            result[f'{tx_name}_total'] = 0
            continue
        # `new_treated` is per-step dispatch; `n_treated` is point-in-time
        # active count (sums to 0 for instantaneous treatments).
        result[f'{tx_name}_total'] = _sum(tx.results, 'new_treated')

    return result


def main():
    t0 = time.time()
    with mp.Pool(len(SCENARIOS)) as pool:
        rows = pool.map(run_one, SCENARIOS)
    df = pd.DataFrame(rows).set_index('scenario')
    print(f'\nAll {len(SCENARIOS)} sims done in {time.time()-t0:.0f}s')
    print()
    print('=== ABO counts + DALYs ===')
    print(df[['n_deliveries', 'n_ptb', 'n_lbw', 'n_still', 'dalys']].to_string())
    print()
    print('=== End-of-sim (2032) STI prev in females ===')
    prev_cols = [c for c in df.columns if c.endswith('_prev_final')]
    print(df[prev_cols].round(4).to_string())
    print()
    print('=== ANC screen activity (per-disease detections) ===')
    for name in ('anc_enroll', 'anc_tri3'):
        cols = [c for c in df.columns if c.startswith(name)]
        print(f'-- {name} --')
        print(df[cols].to_string())
    print()
    print('=== Treatment dispatch totals ===')
    tx_cols = [c for c in df.columns if c.endswith('_total')]
    print(df[tx_cols].to_string())


if __name__ == '__main__':
    main()
