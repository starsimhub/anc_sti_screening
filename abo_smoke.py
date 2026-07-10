"""
Small-N ABO attribution driver: SOC vs 1-screen vs 2-screen at 90% coverage,
central_reversible, 1 draw × 1 seed, 1985-2045. Writes
`results/abo_attribution.csv` for aggregate_abo/plot_abo_attribution.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios import build_scenario_sim


REPO = Path(__file__).resolve().parent
STOP = 2045
WINDOW_START = 2028
N_AGENTS = 10_000
ASSUMPTION = 'central_reversible'
SCENARIOS = ['soc', 'anc_1screen_90cov', 'anc_2screen_90cov']
DISEASES = ('ng', 'ct', 'tv', 'syph')


def _cum(res, key, mask):
    r = res.get(key)
    if r is None:
        return 0
    return int(np.asarray(r.values)[mask].sum())


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
        bundled_prevention=False,
    )
    sim.run()
    elapsed = time.time() - t0

    yrs = np.asarray([t.year for t in sim.results.timevec])
    mask = yrs >= WINDOW_START

    attrib = sim.analyzers['birth_outcome_attribution'].results
    dalys  = sim.analyzers['birth_outcome_dalys'].results

    rows = []
    for outcome in ('ptb', 'lbw'):
        for d in DISEASES:
            rows.append(dict(
                scenario=scenario_id,
                outcome=outcome,
                disease=d,
                n_total=_cum(attrib, f'n_{outcome}_{d}', mask),
                n_sole=_cum(attrib, f'n_{outcome}_sole_{d}', mask),
                n_shared=_cum(attrib, f'n_{outcome}_shared_{d}', mask),
            ))
        rows.append(dict(
            scenario=scenario_id, outcome=outcome, disease='none',
            n_total=_cum(attrib, f'n_{outcome}_no_attribution', mask),
            n_sole=_cum(attrib, f'n_{outcome}_no_attribution', mask),
            n_shared=0,
        ))
    rows.append(dict(
        scenario=scenario_id, outcome='stillbirth', disease='syph',
        n_total=_cum(attrib, 'n_stillbirth_syph', mask),
        n_sole=_cum(attrib, 'n_stillbirth_syph', mask),
        n_shared=0,
    ))
    rows.append(dict(
        scenario=scenario_id, outcome='nnd', disease='syph',
        n_total=_cum(attrib, 'n_nnd_syph', mask),
        n_sole=_cum(attrib, 'n_nnd_syph', mask),
        n_shared=0,
    ))

    totals = dict(
        scenario=scenario_id,
        wall_sec=round(elapsed, 1),
        n_deliveries=_cum(dalys, 'n_deliveries', mask),
        n_ptb=_cum(dalys, 'n_ptb', mask),
        n_lbw=_cum(dalys, 'n_lbw', mask),
        n_ptb_only=_cum(dalys, 'n_ptb_only', mask),
        n_lbw_only=_cum(dalys, 'n_lbw_only', mask),
        n_ptb_and_lbw=_cum(dalys, 'n_ptb_and_lbw', mask),
        n_stillbirths=_cum(dalys, 'n_stillbirths', mask),
        dalys=int(np.asarray(dalys['dalys'].values)[mask].sum()),
    )
    return rows, totals


def main():
    t0 = time.time()
    with mp.Pool(len(SCENARIOS)) as pool:
        results = pool.map(run_one, SCENARIOS)

    attrib_rows, total_rows = [], []
    for rows, totals in results:
        attrib_rows.extend(rows)
        total_rows.append(totals)

    (REPO / 'results').mkdir(exist_ok=True)
    pd.DataFrame(attrib_rows).to_csv(REPO / 'results' / 'abo_attribution.csv', index=False)
    totals_df = pd.DataFrame(total_rows).set_index('scenario')
    totals_df.to_csv(REPO / 'results' / 'abo_totals.csv')

    print(f'\nAll {len(SCENARIOS)} sims done in {time.time()-t0:.0f}s')
    print('\n=== ABO totals (cumulative {}-{}, population scale) ==='.format(WINDOW_START, STOP))
    print(totals_df.to_string())


if __name__ == '__main__':
    main()
