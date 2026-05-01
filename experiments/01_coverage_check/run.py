"""
Prior predictive coverage check.

Draws N_DRAWS parameter sets uniformly from the calibration prior bounds and
runs the model with each.  Saves a list of per-draw result dataframes for
plotting by plot.py.

Run locally with N_DRAWS=10 for a quick sanity check; use N_DRAWS=100+ on
chinchilla (set N_WORKERS to match available cores).
"""

import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import sciris as sc
from stisim.calibration import set_sim_pars, make_df
from model import make_sim
from priors import calib_pars

RESULT_COLS = ['ng.prevalence', 'ct.prevalence_f_25_30', 'tv.prevalence',
               'hiv.prevalence_15_49']
N_DRAWS   = 100
N_WORKERS = 1   # set to core count on chinchilla
RESULTS   = Path(__file__).parent / 'results'


def run_one(args):
    draw_pars, seed = args
    sim = make_sim(verbose=-1, seed=seed, start=1990, stop=2026)
    set_sim_pars(sim, draw_pars)
    sim.init()
    sim.run()
    return make_df(sim, df_res_list=RESULT_COLS)


def sample_draws(n, seed=42):
    rng = np.random.default_rng(seed)
    draws = []
    for name, (label, low, high, log_scale) in calib_pars.items():
        if log_scale:
            vals = np.exp(rng.uniform(np.log(low), np.log(high), n))
        else:
            vals = rng.uniform(low, high, n)
        draws.append((name, vals))
    return [{name: float(vals[i]) for name, vals in draws} for i in range(n)]


if __name__ == '__main__':
    sc.heading(f'Coverage check: {N_DRAWS} prior draws')
    draw_list = sample_draws(N_DRAWS)
    args = [(d, i) for i, d in enumerate(draw_list)]

    if N_WORKERS > 1:
        dfs = sc.parallelize(run_one, iterarg=args, ncpus=N_WORKERS)
    else:
        dfs = [run_one(a) for a in sc.progressbar(args)]

    sc.save(RESULTS / 'coverage_dfs.obj', dfs)
    sc.save(RESULTS / 'coverage_draws.obj', draw_list)
    print(f'Saved to {RESULTS}')
