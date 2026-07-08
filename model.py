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
    """
    Build NG/CT/TV/BV disease modules. ``care_seek_mult`` scales the
    symptomatic care-seeking probability for NG/CT/TV (clipped to [0,1]),
    used as a scenario lever for demand-generation sweeps. Accepts:
      * scalar: applied equally to F and M
      * (mult_f, mult_m) tuple: sex-specific scaling
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
        beta_m2c=0.075,
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

    care_timing_windows_months controls the CareTimingAnalyzer windows
    (per-episode "treated within N months of acquisition" metric); the
    analyzer records one result per window per disease. Defaults to
    (3, 6) so both policy thresholds are captured in one pass.
    """
    d = sc.objdict(hiv=make_hiv())
    analyzers = []
    if which in ('discharging', 'all'):
        d.ng, d.ct, d.tv, d.bv = make_discharging_stis(care_seek_mult=care_seek_mult)
        analyzers.append(sti.sw_stats(diseases=['ng', 'ct', 'tv']))
    if which in ('ulcerative', 'all'):
        d.syph, d.gudp = make_ulcerative_stis()
        # Original analyzer kept for back-compat (tracks syph.infected, 15-49)
        analyzers.append(sti.coinfection_stats('syph', 'hiv', name='syph_hiv_coinfection'))
        # ZIMPHIA-matched: trep and nontrep at 15-64
        analyzers.append(sti.coinfection_stats(
            'syph', 'hiv', disease1_infected_state_name='trep',
            age_limits=[15, 64], name='syph_hiv_trep'))
        analyzers.append(sti.coinfection_stats(
            'syph', 'hiv', disease1_infected_state_name='nontrep',
            age_limits=[15, 64], name='syph_hiv_nontrep'))
        # Transmission event recorder (Lorenz + transmission matrix)
        analyzers.append(SyphTransmissionEvents())

    # Per-episode "treated within N months of acquisition" — stricter
    # than tx_success / new_inf (which counts treatment events not
    # episodes). Only meaningful when treatments exist; gate on which.
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
                       fsw_outreach=False, fsw_coverage_per_step=0.10):
    """Orchestrate intervention construction.

    Layout (top to bottom in the returned list):
      1. HIV interventions
      2. NG/CT/TV testing + treatment (from make_testing)
      3. Syph testing + treatment (from make_syph_testing)
      4. PN intervention (from make_pn), shared across all diseases

    PN is built once at this level with explicit pn_pars routing — it
    is NOT built inside make_testing or make_syph_testing. POCPN /
    SyndromicPN look up the per-disease tests / panels they route to
    by name at step time.

    poc controls the NG/CT/TV SymptomaticTesting panel; poc_syph
    controls the syph ulcer-channel product swap (syndromic_gud →
    gud2) at intv_year. Separated so an experiment can enable one
    without the other. If poc_syph is None, it falls back to poc.
    """
    if poc_syph is None:
        poc_syph = poc
    intvs = make_hiv_intvs()
    if which in ('discharging', 'all'):
        intvs += make_testing(poc=poc, stop=stop,
                              fsw_outreach=fsw_outreach,
                              fsw_coverage_per_step=fsw_coverage_per_step)
    # Insert PN AFTER make_testing but BEFORE make_syph_testing. This
    # order matters: POCPN.notify_attendees fires syph_pn_test.step on
    # attending partners, which sets ti_positive. syph_tx (last
    # intervention in make_syph_testing) reads ti_positive == ti to
    # decide who to treat. If PN ran AFTER syph_tx, those PN-driven
    # syph positives would never be treated — same-step they're missed
    # (syph_tx already ran), next-step they're stale (ti_positive value
    # no longer matches syph_tx.ti). NG/CT/TV cascades survive that bug
    # because their treatments key off persistent tx.eligibility rather
    # than per-step ti_positive.
    if which in ('discharging', 'all'):
        intvs.append(make_pn(poc=poc, pn_pars=pn_pars))
    if which in ('ulcerative', 'all'):
        intvs += make_syph_testing(stop=stop, symp_test_prob=syph_symp_test_prob,
                                   anc_probs=syph_anc_probs, poc=bool(poc_syph))
    # syph_care_seek_mult is applied as a multiplier on top of the
    # (possibly calibrated) rel_test in build_sim, after set_pars_local
    # has overridden any per-draw rel_test values. Doing it at
    # construction would lose to set_pars_local. ANC and PN syph tests
    # are not scaled — ANC is opportunistic, PN tests fire on notified
    # attendees regardless of care-seeking.
    return intvs


def make_sim(seed=1, n_agents=5e3, start=1985, stop=2030,
             pn_pars=None, poc=None, poc_syph=None, which='all',
             dur_recall=ss.years(0.25),
             fetal_health=True, care_seek_mult=1.0, verbose=1/12,
             syph_symp_test_prob=None, syph_anc_probs=None,
             fsw_outreach=False, fsw_coverage_per_step=0.10):
    """care_seek_mult scales NG/CT/TV symptomatic care-seeking
    (`p_symp_care` on each disease module). Syph symptomatic
    care-seeking is scaled separately, at the experiment level — see
    build_sim in run.py: after set_pars_local applies any calibrated
    rel_test, the multiplier is composed on top via
    `syph_symp_test.pars.rel_test *= care_seek_mult` (same for
    syph_symp_test_poc + syph_rash_test). Doing it here at construction
    would lose to set_pars_local overwriting it.
    """

    diseases, analyzers = make_diseases(which, care_seek_mult=care_seek_mult)
    networks = make_networks(dur_recall)
    interventions = make_interventions(diseases, which=which, poc=poc,
                                       poc_syph=poc_syph,
                                       pn_pars=pn_pars, stop=stop,
                                       syph_symp_test_prob=syph_symp_test_prob,
                                       syph_anc_probs=syph_anc_probs,
                                       fsw_outreach=fsw_outreach,
                                       fsw_coverage_per_step=fsw_coverage_per_step)

    # FetalHealth tracks adverse birth outcomes (LBW, SGA, SVN, timing); the
    # sti_fetal connector translates STI infections + treatments into
    # FetalHealth API calls. Both go in `custom` so the standard
    # auto-connector machinery (hiv_*, etc.) still runs.
    custom = [ss.FetalHealth(), sti_fetal()] if fetal_health else None

    # The Zimbabwe demographics CSV in data/ encodes age-cohort values in
    # thousands of people, so starsim's auto-derived total_pop comes out as
    # ~8686 (literal). Override to the actual 1985 Zimbabwe population
    # (~8.7M) so per-agent count outputs (new_infections, n_alive, etc.)
    # scale to absolute people rather than thousands-of-thousands.
    simpars = dict(
        rand_seed=seed, n_agents=n_agents,
        start=start, stop=stop,
        use_migration=False, verbose=verbose,
        total_pop=8.7e6,
    )
    # Coinfection connectors auto-added by sti.Sim. GUDPlaceholder is named
    # 'gudp' so the buggy `gud_syph` auto-connector isn't matched.
    sim = sti.Sim(
        pars=simpars,
        datafolder=f'{DATA_DIR}/',
        demographics=LOCATION,
        diseases=list(diseases.values()),
        networks=networks,
        interventions=interventions,
        analyzers=analyzers,
        custom=custom,
    )
    return sim


if __name__ == '__main__':
    sim = make_sim(seed=1, which='all', start=1985, stop=1990, n_agents=1000)
    sim.run()
    print(f'Diseases: {list(sim.diseases.keys())}')
    print(f'Connectors: {list(sim.connectors.keys()) if sim.connectors else "none"}')
    print(f'HIV prev (final): {sim.results.hiv.prevalence[-1]:.4f}')
