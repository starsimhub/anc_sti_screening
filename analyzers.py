"""
Analyzers for the ANC STI screening model.

Custom result tracking beyond the standard disease/intervention outputs.
"""

import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti


class total_symptomatic(ss.Analyzer):
    """ Track overall symptomatic prevalence across all STIs, stratified by sex and HIV status """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'total_symptomatic'
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_symptoms',       dtype=int,   label='Symptomatic incidence'),
            ss.Result('n_symptomatic',      dtype=int,   label='Number with symptoms'),
            ss.Result('symp_prev',          scale=False, label='Adult symptomatic prevalence'),
            ss.Result('symp_prev_f',        scale=False, label='Vaginal discharge prevalence'),
            ss.Result('symp_prev_m',        scale=False, label='Urethral discharge prevalence'),
            ss.Result('symp_prev_no_hiv',   scale=False, label='Symptomatic prevalence HIV-'),
            ss.Result('symp_prev_has_hiv',  scale=False, label='Symptomatic prevalence HIV+'),
            ss.Result('symp_prev_no_hiv_f', scale=False, label='Symptomatic prevalence HIV- F'),
            ss.Result('symp_prev_has_hiv_f', scale=False, label='Symptomatic prevalence HIV+ F'),
            ss.Result('symp_prev_no_hiv_m', scale=False, label='Symptomatic prevalence HIV- M'),
            ss.Result('symp_prev_has_hiv_m', scale=False, label='Symptomatic prevalence HIV+ M'),
        )
        return

    @staticmethod
    def cond_prob(numerator, denominator):
        numer = len((denominator & numerator).uids)
        denom = len(denominator.uids)
        return sc.safedivide(numer, denom)

    def step(self):
        sim = self.sim
        ti  = self.ti
        adults = (sim.people.age >= 15) & (sim.people.age <= 65)
        women  = adults & sim.people.female
        men    = adults & sim.people.male
        hiv    = sim.diseases.hiv

        new_symptoms = (sim.people.ng.ti_symptomatic == ti) | (sim.people.ct.ti_symptomatic == ti) | (sim.people.tv.ti_symptomatic == ti) | (sim.people.bv.ti_symptomatic == ti)
        any_symptoms = sim.people.ng.symptomatic | sim.people.ct.symptomatic | sim.people.tv.symptomatic | sim.people.bv.symptomatic

        has_hiv   = adults & hiv.infected
        has_hiv_f = has_hiv & women
        has_hiv_m = has_hiv & men
        no_hiv    = adults & hiv.susceptible
        no_hiv_f  = no_hiv & women
        no_hiv_m  = no_hiv & men

        n_symp   = any_symptoms & adults
        n_symp_f = any_symptoms & women
        n_symp_m = any_symptoms & men

        self.results['new_symptoms'][ti]  = np.count_nonzero(new_symptoms)
        self.results['n_symptomatic'][ti] = np.count_nonzero(any_symptoms)
        self.results['symp_prev'][ti]     = sc.safedivide(np.count_nonzero(n_symp), np.count_nonzero(adults))
        self.results['symp_prev_f'][ti]   = sc.safedivide(np.count_nonzero(n_symp_f), np.count_nonzero(women))
        self.results['symp_prev_m'][ti]   = sc.safedivide(np.count_nonzero(n_symp_m), np.count_nonzero(men))

        self.results['symp_prev_no_hiv'][ti]    = self.cond_prob(n_symp, no_hiv)
        self.results['symp_prev_has_hiv'][ti]   = self.cond_prob(n_symp, has_hiv)
        self.results['symp_prev_no_hiv_f'][ti]  = self.cond_prob(n_symp_f, no_hiv_f)
        self.results['symp_prev_has_hiv_f'][ti] = self.cond_prob(n_symp_f, has_hiv_f)
        self.results['symp_prev_no_hiv_m'][ti]  = self.cond_prob(n_symp_m, no_hiv_m)
        self.results['symp_prev_has_hiv_m'][ti] = self.cond_prob(n_symp_m, has_hiv_m)

        return


class pregnancy_sti_stats(ss.Analyzer):
    """ Track STI prevalence among pregnant women — the key outcome for ANC screening evaluation """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = 'pregnancy_sti_stats'
        return

    def init_results(self):
        super().init_results()
        results = sc.autolist()
        results += ss.Result('n_pregnant',        dtype=int, label='Pregnant women')
        results += ss.Result('n_pregnant_any_sti', dtype=int, label='Pregnant with any STI')
        results += ss.Result('pregnant_sti_prev',  scale=False, label='STI prevalence in pregnant women')
        for dis in ['ng', 'ct', 'tv', 'bv']:
            results += ss.Result(f'pregnant_{dis}_prev', scale=False, label=f'{dis.upper()} prevalence in pregnant women')
            results += ss.Result(f'pregnant_{dis}_symp', scale=False, label=f'{dis.upper()} symptomatic in pregnant women')
        results += ss.Result('pregnant_hiv_prev', scale=False, label='HIV prevalence in pregnant women')
        self.define_results(*results)
        return

    def step(self):
        sim = self.sim
        ti  = self.ti
        ppl = sim.people
        pregnant = ppl.pregnancy.pregnant & ppl.female

        n_preg = np.count_nonzero(pregnant)
        self.results['n_pregnant'][ti] = n_preg

        if n_preg == 0:
            return

        any_sti = pregnant & (ppl.ng.infected | ppl.ct.infected | ppl.tv.infected | ppl.bv.infected)
        self.results['n_pregnant_any_sti'][ti] = np.count_nonzero(any_sti)
        self.results['pregnant_sti_prev'][ti]  = sc.safedivide(np.count_nonzero(any_sti), n_preg)

        for dis in ['ng', 'ct', 'tv', 'bv']:
            infected = pregnant & ppl[dis].infected
            self.results[f'pregnant_{dis}_prev'][ti] = sc.safedivide(np.count_nonzero(infected), n_preg)
            if hasattr(ppl[dis], 'symptomatic'):
                symp = pregnant & ppl[dis].symptomatic
                self.results[f'pregnant_{dis}_symp'][ti] = sc.safedivide(np.count_nonzero(symp), n_preg)

        hiv_preg = pregnant & ppl.hiv.infected
        self.results['pregnant_hiv_prev'][ti] = sc.safedivide(np.count_nonzero(hiv_preg), n_preg)

        return


class birth_outcome_dalys(ss.Analyzer):
    """
    Compute DALYs from adverse birth outcomes (PTB and LBW) each timestep.

    Reads classification flags from the FetalHealth module at the point of
    delivery and applies disability weights and durations. Parameters are
    exposed so they can be varied across VoI draws.

    YLD = n_ptb * dw_ptb * dur_ptb  +  n_lbw_only * dw_lbw * dur_lbw

    PTB+LBW co-occurrences accrue the higher disability weight (PTB) only,
    to avoid double-counting.

    Args:
        dw_ptb  (float): disability weight for preterm birth (default GBD 2019: 0.15)
        dw_lbw  (float): disability weight for LBW without PTB (default GBD 2019: 0.10)
        dur_ptb (float): DALY accrual duration in years for PTB (default: 1.0)
        dur_lbw (float): DALY accrual duration in years for LBW (default: 1.0)
        start   (float): year from which to start accumulating DALYs
    """

    def __init__(self, dw_ptb=0.15, dw_lbw=0.10, dur_ptb=1.0, dur_lbw=1.0, start=None, **kwargs):
        super().__init__(**kwargs)
        self.name = 'birth_outcome_dalys'
        self.dw_ptb  = dw_ptb
        self.dw_lbw  = dw_lbw
        self.dur_ptb = dur_ptb
        self.dur_lbw = dur_lbw
        self.start   = start

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.start is None:
            self.start = sim.t.yearvec[0]

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_deliveries',  dtype=int,   label='Deliveries'),
            ss.Result('n_ptb',         dtype=int,   label='Preterm births'),
            ss.Result('n_lbw',         dtype=int,   label='LBW births'),
            ss.Result('n_ptb_lbw',     dtype=int,   label='PTB + LBW'),
            ss.Result('yld_ptb',       scale=False, label='YLD — preterm birth'),
            ss.Result('yld_lbw',       scale=False, label='YLD — LBW only'),
            ss.Result('dalys',         scale=False, label='DALYs'),
            ss.Result('cum_dalys',     scale=False, label='Cumulative DALYs'),
        )

    def step(self):
        sim = self.sim
        ti  = self.ti

        if sim.t.yearvec[ti] < self.start:
            return

        try:
            fh = sim.analyzers['fetal_health']
        except (KeyError, AttributeError):
            return

        preg = sim.people.pregnancy
        # Pregnancy.step() clears pregnant before analyzers run; just-delivered women
        # are identified by ti_delivery == ti and not pregnant
        delivering = (preg.ti_delivery == ti) & ~preg.pregnant
        if not delivering.any():
            return

        uids = delivering.uids
        is_ptb = np.asarray(fh.is_preterm[uids], dtype=bool)
        is_lbw = np.asarray(fh.is_lbw[uids], dtype=bool)

        n_ptb     = int(np.sum(is_ptb))
        n_lbw     = int(np.sum(is_lbw))
        n_ptb_lbw = int(np.sum(is_ptb & is_lbw))

        # Avoid double-counting: LBW-only accrues dw_lbw; PTB accrues dw_ptb regardless
        n_lbw_only = n_lbw - n_ptb_lbw

        yld_ptb = n_ptb     * self.dw_ptb * self.dur_ptb
        yld_lbw = n_lbw_only * self.dw_lbw * self.dur_lbw
        dalys   = yld_ptb + yld_lbw

        self.results['n_deliveries'][ti] = len(uids)
        self.results['n_ptb'][ti]        = n_ptb
        self.results['n_lbw'][ti]        = n_lbw
        self.results['n_ptb_lbw'][ti]    = n_ptb_lbw
        self.results['yld_ptb'][ti]      = yld_ptb
        self.results['yld_lbw'][ti]      = yld_lbw
        self.results['dalys'][ti]        = dalys

    def finalize(self):
        super().finalize()
        self.results['cum_dalys'][:] = np.cumsum(self.results['dalys'].values)


class intervention_costs(ss.Analyzer):
    """
    Track resource costs from STI interventions each timestep.

    Tallies tests administered, treatments given, and adverse outcome
    management costs. All unit cost parameters are exposed so they can be
    varied across VoI draws.

    Cost components:
        - POC test: n_screened × cost_poc_test (covers all STIs in one visit)
        - Marginal ANC visit cost: n_screened × cost_anc_visit
        - Treatment: treatments per disease × disease-specific unit cost
        - Adverse outcome management: n_ptb × cost_ptb_mgmt + n_lbw × cost_lbw_mgmt

    Args:
        cost_poc_test    (float): cost per POC screening test (all STIs)
        cost_anc_visit   (float): marginal cost of integrating STI testing into ANC
        cost_tx_ng       (float): treatment cost per NG case
        cost_tx_ct       (float): treatment cost per CT case
        cost_tx_tv       (float): treatment cost per TV case
        cost_ptb_mgmt    (float): cost of managing a preterm birth
        cost_lbw_mgmt    (float): cost of managing a LBW birth
        anc_screen_name  (str):   name of the ANCScreen intervention in the sim
        start            (float): year from which to start accumulating costs
    """

    def __init__(self, cost_poc_test=8.0, cost_anc_visit=3.0,
                 cost_tx_ng=5.0, cost_tx_ct=3.0, cost_tx_tv=2.0,
                 cost_ptb_mgmt=300.0, cost_lbw_mgmt=200.0,
                 cost_partner_notif=5.0,
                 anc_screen_names=None, start=None, **kwargs):
        super().__init__(**kwargs)
        self.name = 'intervention_costs'
        self.cost_poc_test      = cost_poc_test
        self.cost_anc_visit     = cost_anc_visit
        self.cost_tx_ng         = cost_tx_ng
        self.cost_tx_ct         = cost_tx_ct
        self.cost_tx_tv         = cost_tx_tv
        self.cost_ptb_mgmt      = cost_ptb_mgmt
        self.cost_lbw_mgmt      = cost_lbw_mgmt
        self.cost_partner_notif = cost_partner_notif
        self.anc_screen_names   = anc_screen_names or ['anc_enroll', 'anc_tri3']
        self.start              = start

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.start is None:
            self.start = sim.t.yearvec[0]

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_screened',      dtype=int,   label='Women screened (ANC)'),
            ss.Result('n_treated_ng',    dtype=int,   label='NG treatments'),
            ss.Result('n_treated_ct',    dtype=int,   label='CT treatments'),
            ss.Result('n_treated_tv',    dtype=int,   label='TV treatments'),
            ss.Result('cost_screening',  scale=False, label='Screening costs ($)'),
            ss.Result('cost_treatment',  scale=False, label='Treatment costs ($)'),
            ss.Result('cost_outcomes',   scale=False, label='Adverse outcome management costs ($)'),
            ss.Result('total_cost',      scale=False, label='Total costs ($)'),
            ss.Result('cum_cost',        scale=False, label='Cumulative costs ($)'),
        )

    def step(self):
        sim = self.sim
        ti  = self.ti

        if sim.t.yearvec[ti] < self.start:
            return

        # --- Screening costs (sum across all ANC screens) ---
        n_screened = 0
        for screen_name in self.anc_screen_names:
            anc = sim.interventions.get(screen_name)
            if anc is not None:
                n_screened += int(anc.results['n_screened'][ti])
        cost_screening = n_screened * (self.cost_poc_test + self.cost_anc_visit)

        # --- Treatment costs ---
        n_treated_ng = self._count_treatments(sim, 'ng_tx', 'ng')
        n_treated_ct = self._count_treatments(sim, 'ct_tx', 'ct')
        n_treated_tv = self._count_treatments(sim, 'metronidazole', 'tv')
        cost_treatment = (n_treated_ng * self.cost_tx_ng
                        + n_treated_ct * self.cost_tx_ct
                        + n_treated_tv * self.cost_tx_tv)

        # --- Adverse outcome management costs ---
        n_ptb = n_lbw = 0
        try:
            fh = sim.analyzers['fetal_health']
            n_ptb = int(fh.results['n_preterm'][ti])
            n_lbw = int(fh.results['n_lbw'][ti])
        except (KeyError, AttributeError):
            pass
        cost_outcomes = n_ptb * self.cost_ptb_mgmt + n_lbw * self.cost_lbw_mgmt

        # --- Partner notification costs ---
        n_partners_treated = 0
        pn = sim.interventions.get('partner_notif')
        if pn is not None:
            n_partners_treated = int(pn.results['n_partners_treated'][ti])
        cost_pn = n_partners_treated * self.cost_partner_notif

        total_cost = cost_screening + cost_treatment + cost_outcomes + cost_pn

        self.results['n_screened'][ti]     = n_screened
        self.results['n_treated_ng'][ti]   = n_treated_ng
        self.results['n_treated_ct'][ti]   = n_treated_ct
        self.results['n_treated_tv'][ti]   = n_treated_tv
        self.results['cost_screening'][ti] = cost_screening
        self.results['cost_treatment'][ti] = cost_treatment
        self.results['cost_outcomes'][ti]  = cost_outcomes
        self.results['total_cost'][ti]     = total_cost

    def finalize(self):
        super().finalize()
        self.results['cum_cost'][:] = np.cumsum(self.results['total_cost'].values)

    def _count_treatments(self, sim, tx_name, disease_name):
        """Count successful + unnecessary treatments for a given intervention."""
        tx = sim.interventions.get(tx_name)
        if tx is None or not hasattr(tx, 'outcomes'):
            return 0
        oc = tx.outcomes.get(disease_name, sc.objdict())
        return len(oc.get('successful', [])) + len(oc.get('unnecessary', []))


def make_analyzers(extra_analyzers=None):
    """
    Create the default set of analyzers for this model.

    Args:
        extra_analyzers: additional analyzers to include
    """
    analyzers = [
        sti.sw_stats(diseases=['ng', 'ct', 'tv', 'hiv']),
        total_symptomatic(),
        pregnancy_sti_stats(),
    ]
    if extra_analyzers is not None:
        analyzers += sc.tolist(extra_analyzers)
    return analyzers
