"""
Interventions for the ANC STI screening model.

Ported from sti_notification's interventions.py: make_testing, make_syph_testing,
SyphilisANCTimer, CareSeekScaler, and module-level constants. PN classes
(SyndromicPN, POCPN, PNIntensitySwitch) excluded; partner notification is out
of scope for this project's first run. ANCScreen is a project-specific addition.
DxRiskRedux unifies the old CondomCounseling (treatment-triggered) and
ANCBundledPrevention (screen-triggered) mechanisms.
"""

import stisim as sti
import starsim as ss
import numpy as np
import pandas as pd
import sciris as sc

from pn import PartnerNotification, pn_rates


class SyphilisANCTimer(ss.Intervention):
    """Schedule one ANC syph test event per pregnancy at a realistic week.

    This intervention draws a single visit-week for each newly-conceived
    woman from Uniform(8, 32) and marks her as ANC-test-eligible on
    that timestep. The `SyphTest` intervention reads from ``today_uids``
    to perform the actual test.

    States:
        ti_anc_visit (FloatArr): timestep on which the woman will
            attend her ANC visit. NaN if not pregnant / not scheduled.

    Properties:
        today_uids: UIDs whose ti_anc_visit == current ti and who are
            still alive + still pregnant.

    Pars:
        visit_week_low  (int): lower bound of visit-week draw. Default 8.
        visit_week_high (int): upper bound. Default 32.
    """

    def __init__(self, pars=None, name='syph_anc_timer', **kwargs):
        super().__init__(name=name)
        self.define_pars(
            visit_week=ss.uniform(low=ss.weeks(8), high=ss.weeks(32)),  
        )
        self.update_pars(pars=pars, **kwargs)
        self.define_states(
            ss.FloatArr('ti_anc_visit', label='ti of scheduled ANC visit'),
        )
        return

    def _schedule(self, uids):
        """Draw a visit-week per woman."""
        if len(uids) == 0:
            return
        preg = self.sim.demographics.pregnancy
        weeks = self.pars.visit_week.rvs(uids)
        self.ti_anc_visit[uids] = preg.ti_pregnant[uids] + weeks

    def init_post(self):
        super().init_post()
        if hasattr(self.sim.demographics, 'pregnancy'):
            preg = self.sim.demographics.pregnancy
            self._schedule(preg.pregnant.uids)

    def step(self):
        if not hasattr(self.sim.demographics, 'pregnancy'):
            return
        preg = self.sim.demographics.pregnancy
        new_preg = preg.pregnant.uids[preg.ti_pregnant[preg.pregnant.uids] == self.ti]
        self._schedule(new_preg)

    @property
    def today_uids(self):
        if not hasattr(self.sim.demographics, 'pregnancy'):
            return ss.uids()
        preg = self.sim.demographics.pregnancy
        candidates = self.ti_anc_visit.notnan.uids
        if len(candidates) == 0:
            return ss.uids()
        due = candidates[self.ti_anc_visit[candidates] == self.ti]
        if len(due) == 0:
            return ss.uids()
        # Still pregnant + still alive at this ti
        return due[preg.pregnant[due]]


ANC_PROBS_REALISTIC = [0.20, 0.30, 0.40, 0.35, 0.55, 0.70, 0.85]
ANC_PROBS_POC = [0.05, 0.10, 0.15, 0.15, 0.20, 0.20, 0.20]
ANC_YEARS = [1980, 1990, 1999, 2008, 2012, 2018, 2040]


def make_syph_testing(stop=2040, symp_test_prob=None, rdt_year=2012,
                      anc_probs=None, anc_years=None,
                      poc=False, intv_year=2027):
    """
    Symptomatic + ANC syphilis testing pathways.

    Three channels feed into a single SyphTx:
      1. Symptomatic test (GUD): agents with chancre or rash visible.
      2. ANC RPR screen (1980-rdt_year): serology for pregnant women.
      3. ANC dual RDT screen (rdt_year-stop): treponemal rapid test.

    Args:
        anc_probs: per-visit ANC testing probabilities at the calendar
                   years in ``anc_years``. Default = ANC_PROBS_REALISTIC
                   (peak 70% by 2018, 85% by 2040 — defensible Zimbabwe
                   coverage matching reported EMTCT scale-up). For
                   bifurcation analysis use ANC_PROBS_POC, the
                   non-defensible proof-of-concept ramp from exps 22-23.
    """
    if symp_test_prob is None:
        symp_test_prob = pd.read_csv('data/symp_test_prob_soc.csv')
    if anc_probs is None:
        anc_probs = ANC_PROBS_REALISTIC
    if anc_years is None:
        anc_years = ANC_YEARS

    syph_dx_df = pd.read_csv(f'data/syph_dx.csv')
    # Two syndromic syph channels: ulcer (chancre|gudp.symptomatic → syndromic_gud
    # at 0.8; presumptive treatment for any ulcer-presenter) and rash (rash_visible
    # → syndromic_rash at 0.1; secondary syph rarely reaches STI-clinic treatment).
    gud_dx  = sti.SyphDx(syph_dx_df[syph_dx_df.name == 'syndromic_gud'],
                         name='SyphDx_gud')
    rash_dx = sti.SyphDx(syph_dx_df[syph_dx_df.name == 'syndromic_rash'],
                         name='SyphDx_rash')
    rpr_dx  = sti.SyphDx(syph_dx_df[syph_dx_df.name == 'rpr'],  name='SyphDx_rpr')
    dual_dx = sti.SyphDx(syph_dx_df[syph_dx_df.name == 'dual'], name='SyphDx_dual')

    def syph_dx_eligibility(sim):
        """Treat anyone newly diagnosed positive by any treatment-triggering syph test.

        ANC pathway: pre-intv_year or non-POC uses syph_anc_rdt directly; POC
        arms after intv_year switch to syph_anc_confirm (rpr of dual-RDT positives)
        so treponemal-antibody-carrying cured women aren't re-treated.
        """
        intv = sim.interventions
        treat_tests = ['syph_symp_test', 'syph_symp_test_poc',
                       'syph_rash_test', 'syph_anc_rpr',
                       'syph_pn_test']
        confirm = intv.get('syph_anc_confirm')
        # Keep anc_rdt as treatment-trigger until confirm actually starts —
        # otherwise pre-2027 POC sims lose ANC syph treatment silently.
        if confirm is not None and sim.now >= confirm.start:
            treat_tests.append('syph_anc_confirm')
        else:
            treat_tests.append('syph_anc_rdt')
        tests = [intv.get(n) for n in treat_tests]
        tests = [t for t in tests if t is not None]
        if not tests:
            return ss.uids()
        pos = tests[0].ti_positive == tests[0].ti
        for t in tests[1:]:
            pos = pos | (t.ti_positive == t.ti)
        return pos.uids

    syph_tx = sti.SyphTx(name='syph_tx', label='syph_tx', eligibility=syph_dx_eligibility)

    # --- Ulcer channel: chancre + non-syph GUD presenters ---
    def syph_symp_eligibility(sim):
        syph = sim.diseases.syph
        gudp = sim.diseases.gudp
        return syph.chancre_visible | gudp.symptomatic

    # dt_scale=False because CSV values are per-episode; stisim's default
    # dt_scale=True would divide by 12 and effectively drop primary-syph treatment.
    syph_symp_test = sti.SyphTest(
        name='syph_symp_test', label='syph_symp_test',
        product=gud_dx,
        test_prob_data=symp_test_prob,
        eligibility=syph_symp_eligibility,
        dt_scale=False,
    )

    # POC ulcer channel (poc=True): syph_symp_test_poc uses gud2 (0.95/0.95/0.05)
    # for ulcer presenters after intv_year; syph_pn_test uses rpr (not dual)
    # because dual has 0.20 primary-sens and 0.95 FP on cured patients — bad
    # for PN attendees who are typically primary or previously-treated.
    syph_symp_test_poc = None
    syph_pn_test = None
    if poc:
        gud2_dx = sti.SyphDx(syph_dx_df[syph_dx_df.name == 'gud2'],
                              name='SyphDx_gud2')
        rpr_pn_dx = sti.SyphDx(syph_dx_df[syph_dx_df.name == 'rpr'],
                                name='SyphDx_rpr_pn')
        syph_symp_test.stop = intv_year
        syph_symp_test_poc = sti.SyphTest(
            name='syph_symp_test_poc', label='syph_symp_test_poc',
            product=gud2_dx,
            test_prob_data=symp_test_prob,
            eligibility=syph_symp_eligibility,
            dt_scale=False,
        )
        syph_symp_test_poc.start = intv_year

        def _never_eligible(_sim):
            return ss.uids()

        syph_pn_test = sti.SyphTest(
            name='syph_pn_test', label='syph_pn_test',
            product=rpr_pn_dx,
            test_prob_data=1.0,
            eligibility=_never_eligible,
            dt_scale=False,
        )
        syph_pn_test.start = intv_year

    # --- Rash channel: secondary syph rash presenters (weak) ---
    def syph_rash_eligibility(sim):
        return sim.diseases.syph.rash_visible

    syph_rash_test = sti.SyphTest(
        name='syph_rash_test', label='syph_rash_test',
        product=rash_dx,
        test_prob_data=symp_test_prob,
        eligibility=syph_rash_eligibility,
        dt_scale=False,
    )

    # --- ANC channels (era-gated). SyphilisANCTimer picks one visit-week per
    # pregnancy; SyphTests read today_uids with dt_scale=False so anc_probs are
    # per-visit probabilities.
    syph_anc_timer = SyphilisANCTimer()

    def anc_eligibility(sim):
        sched = sim.interventions.get('syph_anc_timer')
        if sched is None:
            return ss.uids()
        return sched.today_uids

    syph_anc_rpr = sti.SyphTest(
        name='syph_anc_rpr', label='syph_anc_rpr',
        product=rpr_dx,
        years=anc_years,
        test_prob_data=anc_probs,
        eligibility=anc_eligibility,
        dt_scale=False,
    )
    syph_anc_rpr.stop = rdt_year

    syph_anc_rdt = sti.SyphTest(
        name='syph_anc_rdt', label='syph_anc_rdt',
        product=dual_dx,
        years=anc_years,
        test_prob_data=anc_probs,
        eligibility=anc_eligibility,
        dt_scale=False,
    )
    syph_anc_rdt.start = rdt_year

    # POC-arm ANC confirmatory test: RPR follow-up on dual-RDT positives.
    # Cuts the re-treatment loop where treponemal-antibody-carrying cured
    # women trigger the RDT on every ANC visit.
    syph_anc_confirm = None
    if poc:
        def anc_confirm_eligibility(sim):
            rdt = sim.interventions.get('syph_anc_rdt')
            if rdt is None:
                return ss.uids()
            return (rdt.ti_positive == rdt.ti).uids

        # Reuse rpr_pn_dx if it was built above (poc=True branch); else
        # build a new rpr product reference.
        try:
            anc_confirm_dx = rpr_pn_dx
        except NameError:
            anc_confirm_dx = sti.SyphDx(syph_dx_df[syph_dx_df.name == 'rpr'],
                                        name='SyphDx_rpr_anc_confirm')
        syph_anc_confirm = sti.SyphTest(
            name='syph_anc_confirm', label='syph_anc_confirm',
            product=anc_confirm_dx,
            test_prob_data=1.0,
            eligibility=anc_confirm_eligibility,
            dt_scale=False,
        )
        syph_anc_confirm.start = intv_year

    # syph_tx is listed last so its eligibility callback picks up
    # ti_positive == ti from every treatment-triggering test that fired
    # this step. Order matters: syph_anc_confirm runs AFTER syph_anc_rdt
    # (its eligibility reads rdt.ti_positive == ti).
    intvs = [syph_anc_timer, syph_symp_test, syph_rash_test,
             syph_anc_rpr, syph_anc_rdt]
    if syph_symp_test_poc is not None:
        intvs.append(syph_symp_test_poc)
    if syph_pn_test is not None:
        intvs.append(syph_pn_test)
    if syph_anc_confirm is not None:
        intvs.append(syph_anc_confirm)
    intvs.append(syph_tx)
    return intvs


class SyndromicPN(PartnerNotification):
    """
    Partner notification adapted for syndromic STI treatment.

    On attendance, routes partners by sex through the appropriate
    syndromic-management intervention; partners are treated per the
    syndromic algorithm on the next timestep.

    Cycle prevention and the new_attended_no_sti / new_index_no_sti
    diagnostic results are provided by the base
    :class:`sti.PartnerNotification`; this subclass only overrides
    ``notify_attendees`` to route attendees by sex.

    Args:
        eligibility: Index-case selector, e.g. just-treated agents.
        syndromic_vds_name: name of the women's syndromic-mgmt intervention.
        syndromic_uds_name: name of the men's syndromic-mgmt intervention.
    """
    def __init__(self, eligibility,
                 syndromic_vds_name='syndromic_vds',
                 syndromic_uds_name='syndromic_uds', **kwargs):
        super().__init__(eligibility=eligibility, test=None, **kwargs)
        self._syndromic_vds_name = syndromic_vds_name
        self._syndromic_uds_name = syndromic_uds_name
        return

    def notify_attendees(self, uids):
        ppl = self.sim.people
        f_uids = uids[ppl.female[uids]]
        m_uids = uids[ppl.male[uids]]
        vds = self.sim.interventions.get(self._syndromic_vds_name)
        uds = self.sim.interventions.get(self._syndromic_uds_name)
        if len(f_uids) and vds is not None:
            vds.step(uids=f_uids)
        if len(m_uids) and uds is not None:
            uds.step(uids=m_uids)
        return



class POCPN(PartnerNotification):
    """
    Partner notification for the POC arm, switching at ``intv_year``.

    The POC arm must be identical to the SOC arm before ``intv_year``:
    pre-switch, attendees are routed through syndromic management exactly
    as :class:`SyndromicPN` does (by sex, to syndromic_vds/uds). Only at
    ``intv_year`` does routing switch to the POC etiological cascade:
      1. The POC NG/CT/TV panel (etiological dx, replaces syndromic_vds/uds).
      2. The POC syph PN test (rpr, non-treponemal RDT; 0.90 sens across
         primary/secondary/latent/tertiary, 0.05 FP on cured).

    Without this time switch, pre-2027 PN attendees in the POC arm would be
    routed to ``panel``/``syph_pn_test`` (both gated to ``start=intv_year``),
    so they'd receive no treatment while the SOC arm treats the same
    attendees via syndromic management — a deterministic pre-2027 divergence.

    Looks up routed interventions by name through ``self.sim`` at step time.
    Stashing refs at construction would bind to instances that the sim has
    since cloned (their state arrays would be stale / unallocated).

    Cycle prevention + diagnostic results come from the base
    :class:`sti.PartnerNotification`.

    Args:
        eligibility: Index-case selector (same as SyndromicPN).
        panel_name: name of the symptomatic-testing panel intervention to
            route NG/CT/TV testing through (defaults to ``'panel'``).
        syph_pn_test_name: name of the syph PN test (rpr product).
        syndromic_vds_name: women's syndromic-mgmt intervention (pre-switch).
        syndromic_uds_name: men's syndromic-mgmt intervention (pre-switch).
        intv_year: year the POC routing switches on. Default 2027.
    """
    def __init__(self, eligibility, panel_name='panel',
                 syph_pn_test_name='syph_pn_test',
                 syndromic_vds_name='syndromic_vds',
                 syndromic_uds_name='syndromic_uds',
                 intv_year=2027, **kwargs):
        super().__init__(eligibility=eligibility, test=None, **kwargs)
        self._panel_name = panel_name
        self._syph_pn_test_name = syph_pn_test_name
        self._syndromic_vds_name = syndromic_vds_name
        self._syndromic_uds_name = syndromic_uds_name
        self._intv_year = intv_year

    def notify_attendees(self, uids):
        if not len(uids):
            return
        # Pre-switch: route through syndromic management, identical to the
        # SOC arm (SyndromicPN), so the POC arm matches SOC before intv_year.
        if self.sim.now < self._intv_year:
            ppl = self.sim.people
            vds = self.sim.interventions.get(self._syndromic_vds_name)
            uds = self.sim.interventions.get(self._syndromic_uds_name)
            f_uids = uids[ppl.female[uids]]
            m_uids = uids[ppl.male[uids]]
            if len(f_uids) and vds is not None:
                vds.step(uids=f_uids)
            if len(m_uids) and uds is not None:
                uds.step(uids=m_uids)
            return
        # Post-switch: POC etiological cascade.
        panel = self.sim.interventions.get(self._panel_name)
        if panel is not None:
            panel.step(uids=uids)
        syph_pn_test = self.sim.interventions.get(self._syph_pn_test_name)
        if syph_pn_test is not None:
            syph_pn_test.step(uids=uids)
        return


# Baseline PN rates: per-edge notification + per-(edge, partner-sex) attendance.
# Stable = marital; casual partnerships have lower notify + attend rates.
# Shared between make_testing's baseline_pn_eligibility callable and make_pn.
BASELINE_NOTIFY = {'stable': 0.20, 'casual': 0.10}
BASELINE_ATTEND = {'stable': {'f': 0.80, 'm': 0.50},
                   'casual': {'f': 0.50, 'm': 0.25}}


def baseline_pn_eligibility(sim):
    """Index-case selector for the PN intervention: any agent whose
    NG/CT/TV/syph treatment fired this step. Cycle prevention is handled
    inside the upstream :class:`sti.PartnerNotification` (drops
    ``(index, partner)`` edges where ``last_notifier[index] == partner``),
    so no time-windowed filter is applied here.
    """
    intv = sim.interventions
    masks = []
    for name in ('ng_tx', 'ct_tx', 'metronidazole', 'syph_tx'):
        tx = intv.get(name)
        if tx is not None:
            masks.append(tx.ti_treated == tx.ti)
    if not masks:
        return ss.uids()
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return combined.uids


def make_pn(poc=None, pn_pars=None):
    """Build the shared partner-notification intervention.

    PN is shared across all diseases — index pool draws from
    NG/CT/TV/syph treatments collectively, and notify/attend rates are
    set once (no per-disease stratification). Routing of attendees is
    poc-aware:

      * Non-POC (arm A): :class:`SyndromicPN` routes attendees through
        syndromic_vds/uds, which apply the empiric NG/CT/TV/BV
        treatment algorithm. Syph attendees fall out of the syndromic
        pathway unless they happen to present with a chancre.
      * POC (arms B/C/...): :class:`POCPN` routes attendees through the
        POC etiological NG/CT/TV panel + `syph_pn_test` (rpr product),
        applied unconditionally on attending uids. So a notified
        attendee gets the full POC workup regardless of symptoms.

    Cycle prevention and the new_attended_no_sti / new_index_no_sti
    diagnostic results are provided by the upstream
    :class:`sti.PartnerNotification`; we just pass ``diseases`` and
    ``index_treatments`` so the upstream class can compute them.

    Args:
        poc: True for arms B/C/...; False for arm A.
        pn_pars: optional dict of overrides. Recognized keys:
            ``notify_rates`` (dict edge→prob), ``attendance_rates``
            (dict edge→{f, m}→prob). Remaining keys forwarded to the
            PN class.
    """
    overrides = (pn_pars or {}).copy()
    notify = overrides.pop('notify_rates', BASELINE_NOTIFY)
    attend = overrides.pop('attendance_rates', BASELINE_ATTEND)
    pn_pars_built = dict(
        p_notify_current=ss.bernoulli(p=pn_rates(notify)),
        p_attends_current=ss.bernoulli(p=pn_rates(attend)),
        p_notify_previous=ss.bernoulli(p=0),   # current channel only
        p_attends_previous=ss.bernoulli(p=0),
    )
    if poc:
        pn = POCPN(
            eligibility=baseline_pn_eligibility,
            panel_name='panel',
            syph_pn_test_name='syph_pn_test',
            name='pn', label='pn',
            pars=pn_pars_built,
            **overrides,
        )
    else:
        pn = SyndromicPN(
            eligibility=baseline_pn_eligibility,
            syndromic_vds_name='syndromic_vds',
            syndromic_uds_name='syndromic_uds',
            name='pn', label='pn',
            pars=pn_pars_built,
            **overrides,
        )
    return pn


SYNDROMIC_TX_MIX_CERV = dict(
    all3=[0.50, 0.10],
    ngct=[0.20, 0.80],
    mtnz=[0.15, 0.00],
    none=[0.15, 0.10],
)
SYNDROMIC_TX_MIX_NONCERV = dict(
    all3=[0.40, 0.10],
    ngct=[0.10, 0.80],
    mtnz=[0.25, 0.00],
    none=[0.25, 0.10],
)
# POC etiological-test accuracy used for the symptomatic-testing panel
# and for FSW outreach. sti.SymptomaticTesting expects
# {disease: [F, M]} dicts.
POC_SENS = {'ng': [0.95, 0.95], 'ct': [0.95, 0.95], 'tv': [0.95, 0.95]}
POC_SPEC = {'ng': [0.95, 0.95], 'ct': [0.95, 0.95], 'tv': [0.95, 0.95]}


def make_testing(poc=None, stop=2040):

    intv_year = 2027

    # Keep syndromic_vds/uds .stop == sim stop: shortening it makes their step
    # wipe ng/ct/tv_tx.eligibility every post-stop tick, blocking POC treatment.
    # POC mode gates via the eligibility callable instead (see below).
    synd_end = stop

    # Symptomatic care-seekers, baseline (pre-POC) — used by both
    # syndromic_vds/uds and the POC panel.
    def _raw_seeking_care_vds(sim):
        dis = sim.diseases
        female = sim.people.female
        ng_care = dis.ng.symptomatic & (dis.ng.ti_seeks_care == dis.ng.ti) & female
        tv_care = dis.tv.symptomatic & (dis.tv.ti_seeks_care == dis.tv.ti) & female
        ct_care = dis.ct.symptomatic & (dis.ct.ti_seeks_care == dis.ct.ti) & female
        bv_care = dis.bv.symptomatic & (dis.bv.ti_seeks_care == dis.bv.ti) & female
        return (ng_care | ct_care | tv_care | bv_care).uids

    def _raw_seeking_care_uds(sim):
        dis = sim.diseases
        male = sim.people.male
        ng_care = dis.ng.symptomatic & (dis.ng.ti_seeks_care == dis.ng.ti) & male
        tv_care = dis.tv.symptomatic & (dis.tv.ti_seeks_care == dis.tv.ti) & male
        ct_care = dis.ct.symptomatic & (dis.ct.ti_seeks_care == dis.ct.ti) & male
        return (ng_care | ct_care | tv_care).uids

    if poc:
        def seeking_care_vds(sim):
            if sim.now >= intv_year:
                return ss.uids()
            return _raw_seeking_care_vds(sim)

        def seeking_care_uds(sim):
            if sim.now >= intv_year:
                return ss.uids()
            return _raw_seeking_care_uds(sim)

        def seeking_care_any(sim):
            return _raw_seeking_care_vds(sim) | _raw_seeking_care_uds(sim)
    else:
        seeking_care_vds = _raw_seeking_care_vds
        seeking_care_uds = _raw_seeking_care_uds

        def seeking_care_any(sim):
            return seeking_care_vds(sim) | seeking_care_uds(sim)

    ng_tx = sti.GonorrheaTreatment(name='ng_tx', label='ng_tx')
    ct_tx = sti.STITreatment(diseases='ct', name='ct_tx', label='ct_tx')
    metronidazole = sti.STITreatment(diseases=['tv', 'bv'], name='metronidazole', label='metronidazole')
    treatments = [ng_tx, ct_tx, metronidazole]
    outcome_tx_map = dict(
        all3=treatments,
        ngct=[ng_tx, ct_tx],
        mtnz=[metronidazole],
        none=[],
    )

    # Syndromic management of VDS and UDS. Use upstream
    # sti.SyndromicManagement with our project-specific tx_mix values.
    syndromic_pars = dict(
        tx_mix_cerv=SYNDROMIC_TX_MIX_CERV,
        tx_mix_noncerv=SYNDROMIC_TX_MIX_NONCERV,
    )
    syndromic_vds = sti.SyndromicManagement(
        name='syndromic_vds',
        label='syndromic_vds',
        stop=synd_end,
        diseases=['ng', 'ct', 'tv', 'bv'],
        eligibility=seeking_care_vds,
        treatments=treatments,
        outcome_tx_map=outcome_tx_map,
        pars=syndromic_pars,
    )

    syndromic_uds = sti.SyndromicManagement(
        name='syndromic_uds',
        label='syndromic_uds',
        stop=synd_end,
        diseases=['ng', 'ct', 'tv'],
        eligibility=seeking_care_uds,
        treatments=treatments,
        outcome_tx_map=outcome_tx_map,
        pars=syndromic_pars,
    )

    intvs = [syndromic_vds, syndromic_uds, ng_tx, ct_tx, metronidazole]
    if poc:
        # POC etiological panel replaces syndromic_vds/uds after intv_year:
        # single eligibility for both sexes, per-pathogen molecular tests, no
        # presumptive metronidazole (empty negative_treatments=[]).
        disease_treatment_map = {'ng': ng_tx, 'ct': ct_tx, 'tv': metronidazole}
        panel = sti.SymptomaticTesting(
            name='panel', label='panel',
            start=intv_year,
            diseases=['ng', 'ct', 'tv'],
            eligibility=seeking_care_any,
            treatments=treatments,
            disease_treatment_map=disease_treatment_map,
            negative_treatments=[],
            pars=dict(sens=POC_SENS, spec=POC_SPEC),
        )
        intvs.append(panel)

    # PN is built by make_pn() at the top level (make_interventions).
    return intvs


class DxRiskRedux(ss.Intervention):
    """Post-event rel_sus reduction (condom counselling / bundled prevention).
    Args:
        triggers (iterable):  intervention names to listen to.
        trigger_attr (str):   name of the `ti_*` attribute on each trigger
                              intervention to poll (e.g. 'ti_treated' for
                              treatment interventions, 'ti_tested' for screens).
        diseases (iterable):  disease names whose rel_sus gets multiplied.
    """

    def __init__(self, pars=None, triggers=(), trigger_attr='ti_treated',
                 diseases=('ng', 'ct', 'tv', 'syph'),
                 start=2027, name='dx_risk_redux', **kwargs):
        super().__init__(name=name)
        self.define_pars(
            coverage=ss.bernoulli(p=0.5),
            eff=0.5,
            dur=ss.constant(ss.months(6)),
        )
        self.update_pars(pars, **kwargs)
        self.diseases = list(diseases)
        self.triggers = list(triggers)
        self.trigger_attr = trigger_attr
        self.start = start
        self.define_states(
            ss.FloatArr('ti_protect_end', default=np.nan),
        )
        return

    def _newly_triggered(self):
        ti = self.ti
        uids = ss.uids()
        for name in self.triggers:
            intv = self.sim.interventions.get(name)
            if intv is None:
                continue
            ti_arr = getattr(intv, self.trigger_attr, None)
            if ti_arr is None:
                continue
            uids = uids | (ti_arr == ti).uids
        return uids

    def step(self):
        sim = self.sim
        if sim.now < self.start:
            return
        ti = self.ti

        triggered = self._newly_triggered()
        n_enrolled = 0
        if len(triggered):
            enroll = self.pars.coverage.filter(triggered)
            n_enrolled = len(enroll)
            if n_enrolled:
                self.ti_protect_end[enroll] = ti + self.pars.dur.rvs(enroll)
        self._n_enrolled_this_step = n_enrolled

        protected = (self.ti_protect_end > ti).uids
        if not len(protected):
            return
        factor = 1.0 - float(self.pars.eff)
        for d in self.diseases:
            dis = sim.diseases.get(d)
            if dis is None:
                continue
            dis.rel_sus[protected] *= factor
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_protected', dtype=int, label='Currently protected',
                      auto_plot=False),
            ss.Result('new_enrolled', dtype=int, label='Newly enrolled',
                      auto_plot=False),
        )
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        self.results['n_protected'][ti] = int((self.ti_protect_end > ti).sum())
        self.results['new_enrolled'][ti] = int(getattr(self, '_n_enrolled_this_step', 0))
        return


class CareSeekScaler(ss.Intervention):
    """Defer the symptomatic care-seeking multiplier to intv_year.
    """

    def __init__(self, mult=1.0, start=2027,
                 diseases=('ng', 'ct', 'tv'),
                 syph_test_interventions=('syph_symp_test', 'syph_symp_test_poc',
                                          'syph_rash_test'),
                 name='care_seek_scaler', *args, **kwargs):
        super().__init__(name=name)
        self.update_pars(*args, **kwargs)
        if hasattr(mult, '__len__'):
            self.mult_f = float(mult[0])
            self.mult_m = float(mult[1])
        else:
            self.mult_f = self.mult_m = float(mult)
        self.start = start
        self.diseases = list(diseases)
        self.syph_test_interventions = list(syph_test_interventions)
        self._fired = False
        return

    def _find_intv(self, name):
        intvs = self.sim.interventions
        if hasattr(intvs, 'get'):
            found = intvs.get(name)
            if found is not None:
                return found
        for cand in intvs:
            if getattr(cand, 'name', None) == name:
                return cand
        return None

    def _find_disease(self, name):
        ds = self.sim.diseases
        if hasattr(ds, 'get'):
            found = ds.get(name)
            if found is not None:
                return found
        for cand in ds:
            if getattr(cand, 'name', None) == name:
                return cand
        return None

    def step(self):
        if self._fired or self.sim.now < self.start:
            return
        for d_name in self.diseases:
            disease = self._find_disease(d_name)
            if disease is None:
                continue
            curr = disease.pars.p_symp_care
            disease.pars.p_symp_care = [min(1.0, float(curr[0]) * self.mult_f),
                                        min(1.0, float(curr[1]) * self.mult_m)]
        mult_scalar = (self.mult_f + self.mult_m) / 2.0
        if mult_scalar != 1.0:
            for intv_name in self.syph_test_interventions:
                intv = self._find_intv(intv_name)
                if intv is None or not hasattr(intv, 'pars'):
                    continue
                if hasattr(intv.pars, 'rel_test'):
                    intv.pars.rel_test = intv.pars.rel_test * mult_scalar
        self._fired = True
        return


# % ANCScreen class

class ANCScreen(sti.STITest):
    """
    ANC-based screening for asymptomatic STIs during pregnancy.

    Screens pregnant women for NG, CT, TV (and optionally BV) within a
    gestational age window. Multiple instances can be used for timed
    screens (e.g. enrollment ≤24w + third-trimester 32-34w).

    Args:
        diseases (list):               disease name strings (resolved to modules in init_pre)
        treatments (list):             treatment name strings (resolved to modules in init_pre)
        disease_treatment_map (dict):  disease name → treatment name
        test_sensitivity (dict):       per-disease-name test sensitivity
        screen_prob (float/array):     probability of being screened at ANC
        screen_prob_data (array):      time-varying screening probability
        years (array):                 years corresponding to screen_prob_data
        ga_min (float):                minimum gestational age (weeks) for eligibility
        ga_max (float):                maximum gestational age (weeks) for eligibility

    Diseases and treatments are stored as names rather than instances because
    `sti.Sim(**parts)` deep-copies modules on construction, invalidating any
    references captured beforehand. Names get resolved against `sim.diseases`
    and `sim.interventions` inside `init_pre`.
    """

    def __init__(self, pars=None, diseases=None, treatments=None,
                 disease_treatment_map=None, test_sensitivity=None,
                 screen_prob=None, screen_prob_data=None,
                 ga_min=None, ga_max=None,
                 years=None, start=None, stop=None,
                 name=None, label=None, **kwargs):
        super().__init__(years=years, start=start, stop=stop, name=name, label=label)
        self.define_pars(
            screen_prob=ss.bernoulli(p=screen_prob if screen_prob is not None else 0.5),
            dt_scale=False,
        )
        self.update_pars(pars, **kwargs)

        self._disease_names = list(sc.tolist(diseases))
        self._treatment_names = list(sc.tolist(treatments))
        self.disease_treatment_map = dict(disease_treatment_map) if disease_treatment_map else {}
        self.diseases = []
        self.treatments = []

        # Per-disease test sensitivity (default: perfect test), keyed by disease name
        if test_sensitivity is None:
            test_sensitivity = {n: 1.0 for n in self._disease_names}
        self.test_sensitivity = test_sensitivity

        # GA window for eligibility (in weeks); None means no constraint
        self.ga_min = ga_min
        self.ga_max = ga_max

        # Distribution for sensitivity sampling - overwritten below with the values from pars
        self._sens_dist = ss.bernoulli(p=0.5)

        self.screen_prob_data = screen_prob_data
        self._screen_prob_interp = None

        return

    def init_pre(self, sim):
        # Resolve name strings → module instances BEFORE super().init_pre,
        # since init_results iterates self.diseases to build per-disease Result keys.
        self.diseases = [sim.diseases[n] for n in self._disease_names]
        self.treatments = [sim.interventions[n] for n in self._treatment_names]
        self.disease_treatment_map = {
            dname: sim.interventions[tname]
            for dname, tname in self.disease_treatment_map.items()
        }

        super().init_pre(sim)

        # Interpolate screening probability over time if data provided
        if self.screen_prob_data is not None and self.pars.years is not None:
            self._screen_prob_interp = sc.smoothinterp(
                sim.t.yearvec, self.pars.years, self.screen_prob_data, smoothness=0
            )
        return

    def init_results(self):
        super().init_results()
        results = sc.autolist()
        results += ss.Result('n_screened', dtype=int, label='Women screened')
        results += ss.Result('n_positive', dtype=int, label='Women testing positive')
        for d in self.diseases:
            results += ss.Result(f'n_{d.name}_detected', dtype=int, label=f'{d.name.upper()} detected')
            results += ss.Result(f'n_{d.name}_true_pos',  dtype=int, label=f'{d.name.upper()} true positive')
            results += ss.Result(f'n_{d.name}_false_pos', dtype=int, label=f'{d.name.upper()} false positive')
            results += ss.Result(f'n_{d.name}_false_neg', dtype=int, label=f'{d.name.upper()} false negative')
        self.define_results(*results)
        return

    def step(self):
        sim = self.sim
        ppl = sim.people
        ti = self.ti

        if sim.now < self.start or sim.now >= self.stop:
            return

        # Identify pregnant women eligible for screening
        # Pregnant women who haven't been screened this pregnancy (by this instance)
        pregnant = ppl.pregnancy.pregnant
        never_tested = self.ti_tested.isnan
        tested_before_this_pregnancy = self.ti_tested < ppl.pregnancy.ti_pregnant
        eligible = pregnant & ppl.female & (never_tested | tested_before_this_pregnancy)

        if not eligible.any():
            return

        eligible_uids = eligible.uids

        # Filter by gestational age window (in weeks) if specified
        if self.ga_min is not None or self.ga_max is not None:
            ga_weeks = ppl.pregnancy.gestation[eligible_uids]
            in_window = np.ones(len(eligible_uids), dtype=bool)
            if self.ga_min is not None:
                in_window &= ga_weeks >= self.ga_min
            if self.ga_max is not None:
                in_window &= ga_weeks <= self.ga_max
            eligible_uids = eligible_uids[in_window]

        if len(eligible_uids) == 0:
            return

        # Determine who gets screened based on probability
        if self._screen_prob_interp is not None:
            self.pars.screen_prob.set(p=self._screen_prob_interp[ti])

        screened = self.pars.screen_prob.rvs(eligible_uids)
        screened_uids = eligible_uids[screened]

        if len(screened_uids) == 0:
            return

        self.ti_tested[screened_uids] = ti
        self.results['n_screened'][ti] = len(screened_uids)

        # Test each disease and route to treatment
        any_positive = ss.uids()
        for disease in self.diseases:
            dname = disease.name
            sensitivity = self.test_sensitivity.get(dname, 1.0)

            # True infection status
            infected_uids = screened_uids[disease.infected[screened_uids]]
            uninfected_uids = screened_uids[~disease.infected[screened_uids]]

            # Apply test sensitivity (imperfect test)
            if len(infected_uids) and sensitivity < 1.0:
                self._sens_dist.set(p=sensitivity)
                detected = self._sens_dist.rvs(infected_uids)
                true_pos_uids  = infected_uids[detected]
                false_neg_uids = infected_uids[~detected]
            else:
                true_pos_uids  = infected_uids
                false_neg_uids = ss.uids()

            # Assume perfect specificity (no false positives)
            false_pos_uids = ss.uids()

            # Record results
            detected_uids = true_pos_uids | false_pos_uids
            self.results[f'n_{dname}_detected'][ti]  = len(detected_uids)
            self.results[f'n_{dname}_true_pos'][ti]   = len(true_pos_uids)
            self.results[f'n_{dname}_false_pos'][ti]  = len(false_pos_uids)
            self.results[f'n_{dname}_false_neg'][ti]  = len(false_neg_uids)

            # Record in disease results
            for pkey, pattr in disease.sex_keys.items():
                skk = '' if pkey == '' else f'_{pkey}'
                disease.results[f'new_true_pos{skk}'][ti]  += len(true_pos_uids & ppl[pattr])
                disease.results[f'new_false_pos{skk}'][ti] += len(false_pos_uids & ppl[pattr])
                disease.results[f'new_false_neg{skk}'][ti] += len(false_neg_uids & ppl[pattr])
                disease.results[f'new_true_neg{skk}'][ti]  += len(uninfected_uids & ppl[pattr])

            # Route to treatment
            if dname in self.disease_treatment_map and len(detected_uids):
                tx = self.disease_treatment_map[dname]
                tx.eligibility = tx.eligibility | detected_uids

            any_positive = any_positive | detected_uids

        self.results['n_positive'][ti] = len(any_positive)

        return
