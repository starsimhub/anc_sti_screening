"""
Create HIV model and interventions
"""

import numpy as np
import pandas as pd
import stisim as sti


def get_testing_products():
    scaleup_years = np.arange(1990, 2021)
    years = np.arange(1990, 2041)
    n_years = len(scaleup_years)
    fsw_prob = np.concatenate([np.linspace(0, 0.75, n_years), np.linspace(0.75, 0.85, len(years) - n_years)])
    low_cd4_prob = np.concatenate([np.linspace(0, 0.85, n_years), np.linspace(0.85, 0.95, len(years) - n_years)])
    gp_prob = np.concatenate([np.linspace(0, 0.5, n_years), np.linspace(0.5, 0.6, len(years) - n_years)])

    def fsw_eligibility(sim):
        return sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art

    def other_eligibility(sim):
        return ~sim.networks.structuredsexual.fsw & ~sim.diseases.hiv.diagnosed & ~sim.diseases.hiv.on_art

    def low_cd4_eligibility(sim):
        return (sim.diseases.hiv.cd4 < 200) & ~sim.diseases.hiv.diagnosed

    fsw_testing = sti.HIVTest(years=years, test_prob_data=fsw_prob, name='fsw_testing',
                              eligibility=fsw_eligibility, label='fsw_testing')
    other_testing = sti.HIVTest(years=years, test_prob_data=gp_prob, name='other_testing',
                                eligibility=other_eligibility, label='other_testing')
    low_cd4_testing = sti.HIVTest(years=years, test_prob_data=low_cd4_prob, name='low_cd4_testing',
                                  eligibility=low_cd4_eligibility, label='low_cd4_testing')

    return fsw_testing, other_testing, low_cd4_testing


def make_hiv():
    return sti.HIV(
        beta_m2f=0.008,
        eff_condom=0.95,
        init_prev_data=pd.read_csv('data/init_prev_hiv.csv'),
        rel_init_prev=1.,
    )


def make_hiv_intvs():
    n_art = pd.read_csv('data/n_art.csv').set_index('year')
    n_vmmc = pd.read_csv('data/n_vmmc.csv').set_index('year')
    fsw_testing, other_testing, low_cd4_testing = get_testing_products()
    art = sti.ART(coverage=n_art)
    vmmc = sti.VMMC(coverage=n_vmmc)
    prep = sti.Prep()

    return [fsw_testing, other_testing, low_cd4_testing, art, vmmc, prep]
