"""
Run Value of Information analysis.

For each draw from the parameter space:
  1. Sample epi parameters from calibration posterior
  2. Sample birth outcome + cost parameters from priors
  3. Run SOC sim + intervention sim (CRN via shared seed)
  4. Compute incremental DALYs and costs → INMB

Outputs:
  - results/voi_draws.df: per-draw NMB, DALYs, costs, all parameter values
  - results/voi_summary.obj: EVPI at multiple WTP thresholds
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
import pandas as pd
from model import make_sim
from analyzers import birth_outcome_dalys, intervention_costs
from priors import sample_priors
from stisim.calibration import set_sim_pars
from version_utils import save_with_meta

# Constants
LOCATION = 'zimbabwe'
RESULTS_DIR = 'results'

# VoI settings
N_DRAWS = 200           # Number of outer-loop parameter draws
INTV_SCENARIO = 'twice' # PROMISE: enrollment + third-trimester screens
SOC_SCENARIO = 'soc'
START = 1990
STOP = 2040
INTV_YEAR = 2027        # Year screening begins (DALYs/costs accumulated from here)
WTP_THRESHOLDS = [50, 100, 200, 500, 1000, 2000, 5000]


def load_posterior(n_top=200):
    """Load top calibrated parameter sets from Optuna."""
    path = f'{RESULTS_DIR}/{LOCATION}_pars.df'
    df = sc.loadobj(path)
    return df.head(n_top)


def make_connector_pars(draw):
    """Build sti_fetal connector pars from sampled birth outcome parameters."""
    return dict(
        ptb_shift_mean=sc.objdict(
            ng=float(draw['ptb_shift_ng']),
            ct=float(draw['ptb_shift_ct']),
            tv=float(draw['ptb_shift_tv']),
        ),
        ptb_shift_std=float(draw['ptb_shift_std']),
        growth_penalty=sc.objdict(
            ng=float(draw['growth_penalty_ng']),
            ct=float(draw['growth_penalty_ct']),
            tv=float(draw['growth_penalty_tv']),
        ),
        tx_residual_growth=sc.objdict(
            tri1=float(draw['tx_residual_growth_tri1']),
            tri2=float(draw['tx_residual_growth_tri2']),
            tri3=float(draw['tx_residual_growth_tri3']),
        ),
        tx_residual_timing=sc.objdict(
            tri1=float(draw['tx_residual_timing_tri1']),
            tri2=float(draw['tx_residual_timing_tri2']),
            tri3=float(draw['tx_residual_timing_tri3']),
        ),
    )


def make_cost_analyzer(draw):
    """Construct intervention_costs analyzer with sampled cost parameters."""
    return intervention_costs(
        cost_poc_test=float(draw['cost_poc_test']),
        cost_anc_visit=float(draw['cost_anc_visit']),
        cost_tx_ng=float(draw['cost_tx_ng']),
        cost_tx_ct=float(draw['cost_tx_ct']),
        cost_tx_tv=float(draw['cost_tx_tv']),
        cost_ptb_mgmt=float(draw['cost_ptb_mgmt']),
        cost_lbw_mgmt=float(draw['cost_lbw_mgmt']),
        cost_partner_notif=float(draw.get('cost_partner_notif', 5.0)),
        start=INTV_YEAR,
    )


def make_sim_pair(epi_pars, draw, seed):
    """
    Build a SOC + intervention sim pair with shared seed (CRN)
    and sampled birth outcome parameters.

    Args:
        epi_pars (dict): calibrated epi parameters (flat dot-notation)
        draw (dict):     sampled birth outcome + cost parameters
        seed (int):      shared random seed for CRN
    """
    conn_pars = make_connector_pars(draw)
    daly_soc  = birth_outcome_dalys(start=INTV_YEAR)
    daly_intv = birth_outcome_dalys(start=INTV_YEAR)
    cost_soc  = make_cost_analyzer(draw)
    cost_intv = make_cost_analyzer(draw)

    sim_soc = make_sim(
        scenario=SOC_SCENARIO, seed=seed, start=START, stop=STOP,
        verbose=-1, analyzers=[daly_soc, cost_soc],
        connector_pars=conn_pars,
    )

    sim_intv = make_sim(
        scenario=INTV_SCENARIO, seed=seed, start=START, stop=STOP,
        verbose=-1, analyzers=[daly_intv, cost_intv],
        connector_pars=conn_pars,
    )

    # Apply calibrated epi parameters
    set_sim_pars(sim_soc, epi_pars)
    set_sim_pars(sim_intv, epi_pars)

    return sim_soc, sim_intv


def extract_outcomes(sim):
    """Extract cumulative DALYs and costs from a completed sim."""
    dalys = 0.0
    costs = 0.0
    try:
        dalys = float(sim.analyzers['birth_outcome_dalys'].results['cum_dalys'][-1])
    except (KeyError, AttributeError):
        pass
    try:
        costs = float(sim.analyzers['intervention_costs'].results['cum_cost'][-1])
    except (KeyError, AttributeError):
        pass
    return dalys, costs


def run_single_draw(i, epi_pars, draw, seed):
    """Run one SOC + intervention pair and return results."""
    T = sc.timer()
    sim_soc, sim_intv = make_sim_pair(epi_pars, draw, seed)

    sim_soc.run()
    sim_intv.run()

    dalys_soc, costs_soc = extract_outcomes(sim_soc)
    dalys_intv, costs_intv = extract_outcomes(sim_intv)

    delta_dalys = dalys_soc - dalys_intv  # Positive = DALYs averted
    delta_costs = costs_intv - costs_soc  # Positive = intervention costs more

    row = dict(
        draw=i, seed=seed,
        dalys_soc=dalys_soc, dalys_intv=dalys_intv,
        costs_soc=costs_soc, costs_intv=costs_intv,
        delta_dalys=delta_dalys, delta_costs=delta_costs,
    )

    # Store all parameter values for EVPPI regression
    for k, v in epi_pars.items():
        row[k] = v
    for k, v in draw.items():
        row[k] = float(v)

    elapsed = T.toc(output=True)
    print(f'  Draw {i}: ΔDALYs={delta_dalys:+.3f}, ΔCosts=${delta_costs:+,.0f} ({elapsed:0.1f}s)')
    return row


def compute_evpi(draws_df, wtp_thresholds=None):
    """
    Compute EVPI from NMB samples.

    EVPI(λ) = E[max(NMB, 0)] − max(E[NMB], 0)
    """
    if wtp_thresholds is None:
        wtp_thresholds = WTP_THRESHOLDS

    results = []
    for wtp in wtp_thresholds:
        nmb = wtp * draws_df['delta_dalys'] - draws_df['delta_costs']
        evpi = np.mean(np.maximum(nmb, 0)) - max(np.mean(nmb), 0)
        prob_ce = np.mean(nmb > 0)
        results.append(dict(
            wtp=wtp,
            mean_nmb=float(np.mean(nmb)),
            evpi=float(evpi),
            prob_ce=float(prob_ce),
        ))
    return pd.DataFrame(results)


def run_voi(n_draws=N_DRAWS, parallel=True):
    """
    Run the full VoI outer loop.
    """
    T = sc.timer()

    posterior = load_posterior()
    n_posterior = len(posterior)
    print(f'Loaded {n_posterior} posterior parameter sets')

    # Sample birth outcome + cost parameters
    prior_draws = sample_priors(n=n_draws, seed=42)
    print(f'Sampled {n_draws} prior draws')
    print(f'Running {n_draws} draws × 2 sims = {n_draws * 2} total sims '
          f'({SOC_SCENARIO} vs {INTV_SCENARIO}, {START}–{STOP}, CRN)')

    # Time a single draw to estimate total runtime
    print('\nTiming first draw...')
    T0 = sc.timer()
    par_idx = 0
    epi_pars_0 = posterior.iloc[par_idx].to_dict()
    draw_0 = {k: v[0] for k, v in prior_draws.items()}
    row_0 = run_single_draw(0, epi_pars_0, draw_0, seed=1000)
    t_one = T0.toc(output=True)

    if parallel:
        import multiprocessing
        n_cpus = multiprocessing.cpu_count()
        est_mins = (n_draws - 1) * t_one / n_cpus / 60
        print(f'\n  Single draw: {t_one:.0f}s. {n_cpus} CPUs available.')
        print(f'  Estimated wall time for remaining {n_draws - 1} draws (parallel): ~{est_mins:.0f} min')
    else:
        est_mins = (n_draws - 1) * t_one / 60
        print(f'\n  Single draw: {t_one:.0f}s.')
        print(f'  Estimated wall time for remaining {n_draws - 1} draws (serial): ~{est_mins:.0f} min')

    all_rows = [row_0]

    def _run_one(i):
        # Cycle through posterior samples
        par_idx = i % n_posterior
        epi_pars = posterior.iloc[par_idx].to_dict()
        draw = {k: v[i] for k, v in prior_draws.items()}
        seed = 1000 + i  # Unique seed per draw for CRN
        return run_single_draw(i, epi_pars, draw, seed)

    remaining = np.arange(1, n_draws)
    print(f'\nRunning draws 1–{n_draws - 1}...')
    if parallel:
        all_rows += sc.parallelize(_run_one, remaining)
    else:
        for i in remaining:
            all_rows.append(_run_one(i))

    draws_df = pd.DataFrame(all_rows)

    # Compute EVPI
    evpi_df = compute_evpi(draws_df)

    # Save with reproducibility metadata sidecar (see version_utils.py)
    run_meta = dict(
        n_draws=int(n_draws),
        intv_scenario=INTV_SCENARIO,
        soc_scenario=SOC_SCENARIO,
        start=int(START),
        stop=int(STOP),
        intv_year=int(INTV_YEAR),
        wtp_thresholds=WTP_THRESHOLDS,
    )
    save_with_meta(draws_df, f'{RESULTS_DIR}/voi_draws.df', run=run_meta)
    save_with_meta(evpi_df,  f'{RESULTS_DIR}/voi_evpi.df',  run=run_meta)
    print(f'\nSaved {RESULTS_DIR}/voi_draws.df ({len(draws_df)} rows)')
    print(f'Saved {RESULTS_DIR}/voi_evpi.df')
    print(f'Total time: {T.toc(output=True)/60:.1f} min')

    # Print summary
    print('\n--- EVPI Summary ---')
    for _, row in evpi_df.iterrows():
        print(f'  WTP=${row.wtp:,.0f}: EVPI=${row.evpi:,.2f}/woman, '
              f'P(CE)={row.prob_ce:.1%}, E[NMB]=${row.mean_nmb:,.2f}')

    return draws_df, evpi_df


if __name__ == '__main__':

    n_draws  = 200
    parallel = True

    draws_df, evpi_df = run_voi(n_draws=n_draws, parallel=parallel)
    print('Done!')
