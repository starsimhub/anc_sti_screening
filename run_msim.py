"""
Run multi-sim analysis with calibrated parameters.

Loads the top-N calibrated parameter sets, runs each, and generates
percentile statistics for plotting.
"""

# NumPy threading
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
from utils import percentiles

# Constants
LOCATION = 'zimbabwe'
RESULTS_DIR = 'results'


def load_calib_pars(path=None):
    """ Load calibrated parameters as a DataFrame with dot-notation columns """
    if path is None:
        path = f'{RESULTS_DIR}/{LOCATION}_pars.df'
    df = sc.loadobj(path)
    return df


def check_stis_alive(sim):
    """ Post-run filter: reject sims where STIs died out """
    if sim is None:
        return False
    for dis in ['ng', 'ct', 'tv']:
        ni = sim.results[dis].new_infections[-60:]
        if np.sum(ni) == 0:
            return False
    return True


def prune_columns(df):
    """ Keep only the columns needed for analysis """
    keep_prefixes = [
        'time',
        'n_alive',
        'hiv.prevalence', 'hiv.n_infected', 'hiv.new_infections', 'hiv.n_on_art',
        'ng.prevalence', 'ng.n_infected', 'ng.new_infections', 'ng.new_treated',
        'ct.prevalence', 'ct.n_infected', 'ct.new_infections', 'ct.new_treated',
        'tv.prevalence', 'tv.n_infected', 'tv.new_infections', 'tv.new_treated',
        'bv.prevalence', 'bv.n_infected',
        'total_symptomatic.',
        'pregnancy_sti_stats.',
        'sw_stats.',
        'syndromic_vds.', 'syndromic_uds.',
        'anc_screen.',
    ]
    cols = [c for c in df.columns if any(c.startswith(p) for p in keep_prefixes)]
    return df[cols]


def run_msim(n_pars=200, scenario='soc', start=1990, stop=2026):
    """
    Run multi-sim with top calibrated parameter sets.

    Args:
        n_pars (int):     number of parameter sets to use
        scenario (str):   scenario name
        start/stop (int): simulation time range
    """
    print(f'Loading calibrated parameters...')
    pars_df = load_calib_pars()
    print(f'Loaded {len(pars_df)} parameter sets, using top {n_pars}')

    print(f'Building base sim: scenario={scenario}, {start}-{stop}')
    base = make_sim(scenario=scenario, start=start, stop=stop, verbose=-1)

    print(f'Running {n_pars} sims...')
    msim = sti.make_calib_sims(
        calib_pars=pars_df,
        sim=base,
        n_parsets=n_pars,
        check_fn=check_stis_alive,
    )

    print(f'{len(msim.sims)}/{n_pars} sims passed check_fn')
    return msim.sims


def save_results(sims, scenario='soc'):
    """ Convert sim results to DataFrames and compute percentile statistics """
    print(f'Extracting results from {len(sims)} sims...')
    dfs = sc.autolist()
    for sim in sims:
        df = sim.to_df(resample='year', use_years=True, sep='.')
        df = prune_columns(df)
        df['par_idx'] = sim.par_idx
        dfs += df

    resdf = pd.concat(dfs)
    print(f'Computing percentile statistics ({len(resdf)} rows, {len(resdf.columns)} columns)...')
    cs = resdf.groupby(resdf.time).describe(percentiles=percentiles)
    path = f'{RESULTS_DIR}/{LOCATION}_calib_stats_{scenario}.df'
    sc.saveobj(path, cs)
    print(f'Saved {path}')

    return cs


if __name__ == '__main__':

    n_pars   = 200
    scenario = 'soc'

    sims = run_msim(n_pars=n_pars, scenario=scenario)
    cs = save_results(sims, scenario=scenario)

    print('Done!')
