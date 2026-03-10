"""
Run ANC screening scenarios with calibrated parameters.

Compares screening strategies across multiple calibrated parameter sets
and random seeds. Saves treatment and adverse outcome metrics per scenario.
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
from run_msim import load_calib_pars, check_stis_alive
from utils import scenarios, scenlabels

# Constants
LOCATION = 'zimbabwe'
RESULTS_DIR = 'results'


def run_scenario(scenario='soc', n_pars=10, seeds_per_par=5, start=1990, stop=2040):
    """
    Run a single scenario with multiple calibrated parameter sets and seeds.

    Args:
        scenario (str):      scenario name
        n_pars (int):        number of calibrated parameter sets
        seeds_per_par (int): random seeds per parameter set
        start/stop (int):    simulation time range
    """
    pars_df = load_calib_pars()
    base = make_sim(scenario=scenario, start=start, stop=stop, verbose=-1)

    msim = sti.make_calib_sims(
        calib_pars=pars_df,
        sim=base,
        n_parsets=n_pars,
        seeds_per_par=seeds_per_par,
        check_fn=check_stis_alive,
    )

    return msim.sims


def extract_results(sims, scenario):
    """
    Extract key results from completed scenario sims.

    Returns a long-format DataFrame with columns:
        scenario, par_idx, year, metric, value
    """
    all_rows = []
    for sim in sims:
        par_idx = sim.par_idx
        yearvec = sim.t.yearvec

        # Disease-level results
        for dis in ['ng', 'ct', 'tv', 'bv']:
            for metric in ['new_infections', 'prevalence', 'new_treated']:
                key = f'{dis}.{metric}'
                try:
                    vals = sim.results[dis][metric]
                    for year, val in zip(yearvec, vals):
                        all_rows.append(dict(scenario=scenario, par_idx=par_idx,
                                             year=year, metric=key, value=float(val)))
                except (KeyError, AttributeError):
                    pass

        # ANC screening results (if present)
        if 'anc_screen' in sim.interventions:
            anc = sim.results.anc_screen
            for metric_key in ['n_screened', 'n_positive']:
                for year, val in zip(yearvec, anc[metric_key]):
                    all_rows.append(dict(scenario=scenario, par_idx=par_idx,
                                         year=year, metric=f'anc.{metric_key}', value=float(val)))
            for dis in ['ng', 'ct', 'tv']:
                for suffix in ['detected', 'true_pos', 'false_neg']:
                    mkey = f'n_{dis}_{suffix}'
                    try:
                        for year, val in zip(yearvec, anc[mkey]):
                            all_rows.append(dict(scenario=scenario, par_idx=par_idx,
                                                 year=year, metric=f'anc.{mkey}', value=float(val)))
                    except KeyError:
                        pass

        # Pregnancy STI stats (if present)
        if 'pregnancy_sti_stats' in sim.analyzers:
            ps = sim.results.pregnancy_sti_stats
            for mkey in ['n_pregnant', 'n_pregnant_any_sti', 'pregnant_sti_prev']:
                for year, val in zip(yearvec, ps[mkey]):
                    all_rows.append(dict(scenario=scenario, par_idx=par_idx,
                                         year=year, metric=f'preg.{mkey}', value=float(val)))
            for dis in ['ng', 'ct', 'tv']:
                for year, val in zip(yearvec, ps[f'pregnant_{dis}_prev']):
                    all_rows.append(dict(scenario=scenario, par_idx=par_idx,
                                         year=year, metric=f'preg.{dis}_prev', value=float(val)))

    df = pd.DataFrame(all_rows)
    return df


def save_scenario_results(sims, scenario):
    """ Extract and save results for a single scenario """
    df = extract_results(sims, scenario)
    sc.saveobj(f'{RESULTS_DIR}/scenario_{scenario}.df', df)
    print(f'Saved {RESULTS_DIR}/scenario_{scenario}.df ({len(df)} rows)')
    return df


if __name__ == '__main__':

    n_pars       = 10
    seeds_per_par = 5
    stop         = 2040

    all_dfs = []
    for scenario in scenarios:
        sc.heading(f'Running scenario: {scenlabels[scenario]}')
        sims = run_scenario(scenario=scenario, n_pars=n_pars,
                           seeds_per_par=seeds_per_par, stop=stop)
        df = save_scenario_results(sims, scenario)
        all_dfs.append(df)

    # Combine all scenarios
    combined = pd.concat(all_dfs, ignore_index=True)
    sc.saveobj(f'{RESULTS_DIR}/all_scenarios.df', combined)
    print(f'\nSaved combined results: {len(combined)} rows')
    print('Done!')
