"""
ANC STI Screening Model

Agent-based model for evaluating antenatal care (ANC) screening strategies
for asymptomatic STIs (NG, CT, TV, BV) during pregnancy in Zimbabwe.
Built on STIsim/Starsim with HIV co-circulation.
"""

# %% Imports and settings
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import stisim as sti

from interventions import make_testing
from analyzers import make_analyzers
from fetal_health import FetalHealth
from connectors import sti_fetal

# Constants
LOCATION = 'zimbabwe'
DATA_DIR = 'data'
RESULTS_DIR = 'results'


# %% Disease setup
def make_stis():
    """ Create the four discharging STIs """
    ng = sti.Gonorrhea(eff_condom=0.7)
    ct = sti.Chlamydia(eff_condom=0.8)
    tv = sti.Trichomoniasis(eff_condom=0.8)
    bv = sti.SimpleBV()
    return ng, ct, tv, bv


def make_hiv():
    """ Create HIV disease module with Zimbabwe-specific initial prevalence """
    hiv = sti.HIV(
        beta_m2f=0.035,
        eff_condom=0.95,
        init_prev_data=pd.read_csv(f'{DATA_DIR}/init_prev_hiv.csv'),
        rel_init_prev=1.,
    )
    return hiv


def make_hiv_intvs():
    """ Create HIV testing, treatment, and prevention interventions """
    scaleup_years = np.arange(1990, 2021)
    years = np.arange(1990, 2041)
    n_years = len(scaleup_years)
    fsw_prob    = np.concatenate([np.linspace(0, 0.75, n_years), np.linspace(0.75, 0.85, len(years) - n_years)])
    low_cd4_prob = np.concatenate([np.linspace(0, 0.85, n_years), np.linspace(0.85, 0.95, len(years) - n_years)])
    gp_prob     = np.concatenate([np.linspace(0, 0.5, n_years), np.linspace(0.5, 0.6, len(years) - n_years)])

    # FSW testing
    def fsw_eligibility(sim):
        return sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art

    fsw_testing = sti.HIVTest(
        years=years, test_prob_data=fsw_prob,
        name='fsw_testing', eligibility=fsw_eligibility, label='fsw_testing',
    )

    # General population testing
    def other_eligibility(sim):
        return ~sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art

    other_testing = sti.HIVTest(
        years=years, test_prob_data=gp_prob,
        name='other_testing', eligibility=other_eligibility, label='other_testing',
    )

    # Low CD4 testing
    def low_cd4_eligibility(sim):
        return (sim.diseases.hiv.cd4 < 200) & ~sim.diseases.hiv.diagnosed

    low_cd4_testing = sti.HIVTest(
        years=years, test_prob_data=low_cd4_prob,
        name='low_cd4_testing', eligibility=low_cd4_eligibility, label='low_cd4_testing',
    )

    # Treatment and prevention
    n_art  = pd.read_csv(f'{DATA_DIR}/n_art.csv').set_index('year')
    n_vmmc = pd.read_csv(f'{DATA_DIR}/n_vmmc.csv').set_index('year')
    art  = sti.ART(coverage=n_art)
    vmmc = sti.VMMC(coverage=n_vmmc)
    prep = sti.Prep()

    return [fsw_testing, other_testing, low_cd4_testing, art, vmmc, prep]


def make_diseases():
    """
    Create all diseases and connectors.

    Returns:
        diseases (list):   [hiv, ng, ct, tv, bv]
        connectors (list): HIV-STI coinfection interactions
    """
    hiv = make_hiv()
    ng, ct, tv, bv = make_stis()
    diseases = [hiv, ng, ct, tv, bv]
    connectors = [sti.hiv_ng(hiv, ng), sti.hiv_ct(hiv, ct), sti.hiv_tv(hiv, tv)]
    return diseases, connectors


# %% Sim construction
def make_sim(scenario='soc', seed=1, start=1990, stop=2030, n_agents=None, debug=False,
             verbose=1/12, analyzers=None):
    """
    Build the simulation.

    Args:
        scenario (str):   screening scenario name (e.g. 'soc', 'anc_screen')
        seed (int):       random seed
        start (int):      simulation start year
        stop (int):       simulation stop year
        n_agents (int):   population size (default 10k, 500 for debug)
        debug (bool):     if True, use small population
        verbose (float):  print frequency in years
        analyzers (list): additional analyzers to include
    """
    total_pop = {1970: 5.203e6, 1980: 7.05e6, 1985: 8.691e6, 1990: 9980999, 2000: 11.83e6}[start]
    if n_agents is None:
        n_agents = [int(10e3), int(5e2)][debug]

    # Demographics
    fertility_data = pd.read_csv(f'{DATA_DIR}/asfr.csv')
    pregnancy = ss.Pregnancy(fertility_rate=fertility_data)
    death_data = pd.read_csv(f'{DATA_DIR}/deaths.csv')
    death = ss.Deaths(death_rate=death_data, rate_units=1)

    # People and networks
    ppl = ss.People(n_agents, age_data=pd.read_csv(f'{DATA_DIR}/age_dist_{start}.csv', index_col='age')['value'])
    sexual = sti.StructuredSexual(
        prop_f0=0.79,
        prop_m0=0.83,
        f1_conc=0.16,
        m1_conc=0.11,
        p_pair_form=0.58,
        condom_data=pd.read_csv(f'{DATA_DIR}/condom_use.csv'),
    )
    maternal = ss.MaternalNet()

    # Diseases and connectors
    diseases, connectors = make_diseases()

    # Fetal health module + connector
    # Check if a custom FetalHealth was passed in analyzers
    has_custom_fh = False
    if analyzers is not None:
        for a in sc.tolist(analyzers):
            if isinstance(a, FetalHealth):
                has_custom_fh = True
                break
    if not has_custom_fh:
        fetal_health = FetalHealth()
        analyzers = sc.mergelists(fetal_health, analyzers)
    connectors.append(sti_fetal())

    # Interventions: HIV + STI testing
    hiv_intvs = make_hiv_intvs()
    ng, ct, tv, bv = diseases[1], diseases[2], diseases[3], diseases[4]
    sti_intvs = make_testing(ng, ct, tv, bv, scenario=scenario, stop=stop)
    interventions = hiv_intvs + sti_intvs

    # Analyzers
    analyzers = make_analyzers(extra_analyzers=analyzers)

    # Build sim
    simpars = dict(
        use_migration=False,
        rand_seed=seed,
        n_agents=n_agents,
        start=start,
        stop=stop,
        verbose=verbose,
    )

    sim = sti.Sim(
        pars=simpars,
        total_pop=total_pop,
        people=ppl,
        demographics=[pregnancy, death],
        diseases=diseases,
        networks=[sexual, maternal],
        connectors=connectors,
        interventions=interventions,
        analyzers=analyzers,
    )
    sim.scenario = scenario

    return sim


if __name__ == '__main__':

    debug = False
    seed  = 1
    scenario = 'soc'

    sim = make_sim(scenario=scenario, seed=seed, debug=debug, start=1990, stop=2030)
    sim.run()

    df = sim.to_df(resample='year', use_years=True, sep='.')
    sc.saveobj(f'{RESULTS_DIR}/{scenario}_sim.df', df)
    print('Done!')
