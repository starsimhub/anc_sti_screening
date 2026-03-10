"""
Run calibrations for the ANC STI screening model.

Calibrates HIV + NG/CT/TV transmission parameters and network structure
to Zimbabwe epidemiological data using Optuna via the STIsim v15 calibration API.
"""

# NumPy threading — must be set before numpy is imported
import os
os.environ.update(
    OMP_NUM_THREADS='1',
    OPENBLAS_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1',
    MKL_NUM_THREADS='1',
)

# %% Imports
import numpy as np
import sciris as sc
import stisim as sti
import pandas as pd
from model import make_sim

# Constants
LOCATION = 'zimbabwe'
DATA_DIR = 'data'
RESULTS_DIR = 'results'

# Run settings
TOTAL_TRIALS = 2000
N_WORKERS = 100
STORAGE = None


def make_calibration():

    # Define calibration parameters — dot notation for v15 API
    ckw = dict(suggest_type='suggest_float')
    calib_pars = dict(
        # HIV transmission
        hiv=dict(
            beta_m2f=      dict(low=0.002, high=0.014, guess=0.006, **ckw),
            eff_condom=    dict(low=0.5,   high=0.9,   guess=0.75,  **ckw),
            rel_init_prev= dict(low=2,     high=15,    guess=8,     **ckw),
        ),
        # Network structure
        structuredsexual=dict(
            prop_f0= dict(low=0.55, high=0.9,  guess=0.7,  **ckw),
            prop_m0= dict(low=0.55, high=0.85, guess=0.83, **ckw),
            m1_conc= dict(low=0.05, high=0.3,  guess=0.15, **ckw),
        ),
        # STI transmission
        ng=dict(
            beta_m2f= dict(low=0.02, high=0.25, guess=0.08, log=True, **ckw),
            p_symp=   dict(low=0.05, high=0.30, guess=0.15, **ckw),
        ),
        ct=dict(
            beta_m2f= dict(low=0.02, high=0.25, guess=0.06, log=True, **ckw),
            p_symp=   dict(low=0.10, high=0.35, guess=0.25, **ckw),
        ),
        tv=dict(
            beta_m2f= dict(low=0.02, high=0.25, guess=0.07, log=True, **ckw),
            p_symp=   dict(low=0.15, high=0.75, guess=0.45, **ckw),
        ),
    )

    # Extra results to track during calibration
    sres = []
    for dis in ['ng', 'ct', 'tv']:
        for res in ['prevalence', 'new_infections', 'n_infected']:
            for sk in ['', '_f', '_m']:
                sres.append(f'{dis}.{res}{sk}')
    sres += ['hiv.n_on_art', 'n_alive', 'hiv.new_deaths']

    # Build sim and load data
    sim = make_sim(verbose=-1, seed=1, start=1990, stop=2026)
    data = pd.read_csv(f'{DATA_DIR}/{LOCATION}_sti_data.csv')

    weights = dict(
        ng_prevalence=2,
        ct_prevalence=2,
        tv_prevalence=1,
        ng_new_infections=0,
        ct_new_infections=0,
        tv_new_infections=0,
    )

    # Post-sim check: reject if STIs die out or HIV too low
    def check_fn(sim):
        if sim is None:
            return False
        for dis in ['ng', 'ct', 'tv']:
            ni = sim.results[dis].new_infections[-60:]
            if np.sum(ni) == 0:
                return False
        hiv_prev = sim.results.hiv.prevalence_15_49[-60:]
        if np.median(hiv_prev) < 0.05:
            return False
        return True

    calib = sti.Calibration(
        calib_pars=calib_pars,
        extra_results=sres,
        weights=weights,
        sim=sim,
        data=data,
        check_fn=check_fn,
        study_name=f'{LOCATION}_anc_sti_calibration',
        total_trials=TOTAL_TRIALS,
        n_workers=N_WORKERS,
        die=False, reseed=False, storage=STORAGE,
        save_results=True,
        continue_db=True, keep_db=True,
    )

    return sim, calib


if __name__ == '__main__':

    do_shrink = True

    sim, calib = make_calibration()
    sc.heading(f'Running calibration, {TOTAL_TRIALS} trials')
    calib.calibrate()
    print(f'Best pars are {calib.best_pars}')
    calib.remove_db()
    calib.save(
        f'{RESULTS_DIR}/{LOCATION}_calib.obj',
        shrink=do_shrink,
        pars_filename=f'{RESULTS_DIR}/{LOCATION}_pars.df',
    )

    print('Done!')
