"""
Build a Zimbabwe sim with HIV + 4 discharging STIs (NG/CT/TV/BV) +
syphilis + GUD placeholder. Used to evaluate partner-notification and
care-seeking strategies for STI undertreatment.
"""

import os

import starsim as ss
import sciris as sc
import pandas as pd
import stisim as sti

from interventions import make_testing, make_syph_testing, make_pn
from hiv_model import make_hiv, make_hiv_intvs
from connectors import sti_fetal
from analyzers import SyphTransmissionEvents, CareTimingAnalyzer

LOCATION = 'zimbabwe'
DATA_DIR = 'data'


def make_discharging_stis(care_seek_mult=1.0):
    """Build NG/CT/TV/BV. `care_seek_mult` scales `p_symp_care` on NG/CT/TV
    (scalar or (F, M) tuple), clipped to [0, 1].
    """
    if hasattr(care_seek_mult, '__len__'):
        mult_f, mult_m = float(care_seek_mult[0]), float(care_seek_mult[1])
    else:
        mult_f = mult_m = float(care_seek_mult)
    def scaled(care_pair):
        return [min(1.0, care_pair[0] * mult_f),
                min(1.0, care_pair[1] * mult_m)]
    ng = sti.Gonorrhea(eff_condom=0.7, beta_m2f=0.12, p_symp=[0.13, 0.65],
                      p_symp_care=scaled([0.49, 0.83]))
    ct = sti.Chlamydia(eff_condom=0.8, beta_m2f=0.15, p_symp=[0.30, 0.54],
                      p_symp_care=scaled([0.49, 0.83]))
    tv = sti.Trichomoniasis(eff_condom=0.8, beta_m2f=0.14, p_symp=[0.6, 0.5],
                          p_symp_care=scaled([0.49, 0.27]))
    bv = sti.SimpleBV()
    return ng, ct, tv, bv


def make_ulcerative_stis():
    init_prev_path = f'{DATA_DIR}/init_prev_syph.csv'
    init_prev_latent_path = f'{DATA_DIR}/init_prev_latent_syph.csv'
    syph = sti.Syphilis(
        beta_m2f=0.15,
        beta_m2c=1.0,   # guaranteed transmission during syph+ pregnancy for the MTCT-outcomes diagnostic
        eff_condom=0.5,
        rel_trans_primary=5,
        rel_trans_secondary=1,
        rel_trans_latent_half_life=ss.months(6),
        p_symp_primary=[0.3, 0.8],
        anc_detection=1.,
        rel_init_prev=0.2,
        # 15-64 matches the ZIMPHIA household-survey denominator for the
        # trep / nontrep targets in the calibration pipeline.
        age_range=[15, 64],
        init_prev_data=pd.read_csv(init_prev_path) if os.path.exists(init_prev_path) else None,
        init_prev_latent_data=pd.read_csv(init_prev_latent_path) if os.path.exists(init_prev_latent_path) else None,
    )
    # name='gudp' avoids the buggy `gud_syph` auto-connector match.
    gud = sti.GUDPlaceholder(prevalence=0.01, name='gudp')
    return syph, gud


def make_diseases(which='all', care_seek_mult=1.0,
                  care_timing_windows_months=(3, 6)):
    """Build the disease set + matching analyzers. Returns (dict, analyzers).

    care_timing_windows_months: (lo, hi) months for the per-episode
    "treated within N months of acquisition" analyzer.
    """
    d = sc.objdict(hiv=make_hiv())
    analyzers = []
    if which in ('discharging', 'all'):
        d.ng, d.ct, d.tv, d.bv = make_discharging_stis(care_seek_mult=care_seek_mult)
        analyzers.append(sti.sw_stats(diseases=['ng', 'ct', 'tv']))
    if which in ('ulcerative', 'all'):
        d.syph, d.gudp = make_ulcerative_stis()
        analyzers.append(sti.coinfection_stats('syph', 'hiv', name='syph_hiv_coinfection'))
        # ZIMPHIA-matched: trep and nontrep at 15-64
        analyzers.append(sti.coinfection_stats(
            'syph', 'hiv', disease1_infected_state_name='trep',
            age_limits=[15, 64], name='syph_hiv_trep'))
        analyzers.append(sti.coinfection_stats(
            'syph', 'hiv', disease1_infected_state_name='nontrep',
            age_limits=[15, 64], name='syph_hiv_nontrep'))
        analyzers.append(SyphTransmissionEvents())

    if which == 'all':
        analyzers.append(CareTimingAnalyzer(
            disease_names=['ng', 'ct', 'tv', 'syph'],
            treatment_disease_map={
                'ng_tx': 'ng',
                'ct_tx': 'ct',
                'metronidazole': 'tv',
                'syph_tx': 'syph',
            },
            windows_months=care_timing_windows_months,
        ))
    elif which == 'discharging':
        analyzers.append(CareTimingAnalyzer(
            disease_names=['ng', 'ct', 'tv'],
            treatment_disease_map={
                'ng_tx': 'ng', 'ct_tx': 'ct', 'metronidazole': 'tv',
            },
            windows_months=care_timing_windows_months,
        ))
    elif which == 'ulcerative':
        analyzers.append(CareTimingAnalyzer(
            disease_names=['syph'],
            treatment_disease_map={'syph_tx': 'syph'},
            windows_months=care_timing_windows_months,
        ))
    return d, analyzers


def make_networks(dur_recall=ss.years(0.25)):
    sexual = sti.StructuredSexual(
        prop_f0=0.67, prop_m0=0.55,
        prop_f2=0.10, prop_m2=0.20,
        concurrency_dist=ss.nbinom(n=2, p=0.5),
        f1_conc=0.15, m1_conc=0.20,
        f2_conc=1.0, m2_conc=4.4,
        recall_prior=True,
        condom_data=pd.read_csv(f'{DATA_DIR}/condom_use.csv'),
        fsw_shares=ss.bernoulli(p=0.10),
        client_shares=ss.bernoulli(p=0.20),
        sw_seeking_rate=ss.permonth(20),
    )
    return [sexual, sti.PriorPartners(dur_recall=dur_recall), ss.MaternalNet()]


def make_interventions(diseases, which='all', poc=None, poc_syph=None,
                       pn_pars=None, stop=2040,
                       syph_symp_test_prob=None, syph_anc_probs=None,
                       syph_anc_windows=None):
    """Build intervention list: HIV → NG/CT/TV testing+treatment → PN → syph testing+treatment.

    poc_syph falls back to poc if None. poc controls NG/CT/TV SymptomaticTesting;
    poc_syph controls the syph ulcer-channel product swap independently.
    syph_anc_windows is a list of (low_wk, high_wk) tuples passed to
    SyphilisANCTimer; defaults to [(8, 32)] (SOC arbitrary timing).
    """
    if poc_syph is None:
        poc_syph = poc
    intvs = make_hiv_intvs()
    if which in ('discharging', 'all'):
        intvs += make_testing(poc=poc, stop=stop)
    # PN must run BEFORE make_syph_testing: syph_tx reads ti_positive set the
    # same step by syph_pn_test, and would miss PN-driven positives if ordered after.
    if which in ('discharging', 'all'):
        intvs.append(make_pn(poc=poc, pn_pars=pn_pars))
    if which in ('ulcerative', 'all'):
        intvs += make_syph_testing(stop=stop, symp_test_prob=syph_symp_test_prob,
                                   anc_probs=syph_anc_probs, poc=bool(poc_syph),
                                   syph_anc_windows=syph_anc_windows)
    return intvs


def make_sim_parts(seed=1, n_agents=5e3, start=1985, stop=2030,
                   pn_pars=None, poc=None, poc_syph=None, which='all',
                   dur_recall=ss.years(0.25),
                   fetal_health=True, care_seek_mult=1.0, verbose=1/12,
                   syph_symp_test_prob=None, syph_anc_probs=None,
                   syph_anc_windows=None):
    """Return a dict of Sim kwargs ready for sti.Sim(**parts).

    Callers can inspect or mutate the returned dict before constructing
    the Sim — useful for adding scenario interventions that need
    references to the built disease and treatment modules.
    """
    diseases, analyzers = make_diseases(which, care_seek_mult=care_seek_mult)
    networks = make_networks(dur_recall)
    interventions = make_interventions(diseases, which=which, poc=poc,
                                       poc_syph=poc_syph,
                                       pn_pars=pn_pars, stop=stop,
                                       syph_symp_test_prob=syph_symp_test_prob,
                                       syph_anc_probs=syph_anc_probs,
                                       syph_anc_windows=syph_anc_windows)

    # sti_fetal translates STI events into FetalHealth API calls.
    custom = [ss.FetalHealth(), sti_fetal()] if fetal_health else None

    # total_pop=8.7e6 overrides the auto-derived value: the demographics CSV
    # is in thousands so starsim would otherwise infer ~8686 literal agents.
    simpars = dict(
        rand_seed=seed, n_agents=n_agents,
        start=start, stop=stop,
        use_migration=False, verbose=verbose,
        total_pop=8.7e6,
    )
    return dict(
        pars=simpars,
        datafolder=f'{DATA_DIR}/',
        demographics=LOCATION,
        diseases=list(diseases.values()),
        networks=networks,
        interventions=interventions,
        analyzers=analyzers,
        custom=custom,
    )


def make_sim(interventions=(), analyzers=(), custom=(), **parts_kwargs):
    """Convenience wrapper: builds parts, appends extras, returns sti.Sim.

    For scenarios that need references to the built disease/treatment
    modules when constructing extras, call make_sim_parts directly.
    """
    parts = make_sim_parts(**parts_kwargs)
    if interventions:
        parts['interventions'] = list(parts['interventions']) + list(interventions)
    if analyzers:
        parts['analyzers'] = list(parts['analyzers']) + list(analyzers)
    if custom:
        base = parts.get('custom') or []
        parts['custom'] = list(base) + list(custom)
    return sti.Sim(**parts)


if __name__ == '__main__':
    sim = make_sim(seed=1, which='all', start=1985, stop=1990, n_agents=1000)
    sim.run()
    print(f'Diseases: {list(sim.diseases.keys())}')
    print(f'Connectors: {list(sim.connectors.keys()) if sim.connectors else "none"}')
    print(f'HIV prev (final): {sim.results.hiv.prevalence[-1]:.4f}')
