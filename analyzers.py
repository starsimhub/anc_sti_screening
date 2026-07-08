"""
Analyzers for the ANC STI screening model.

Custom result tracking beyond the standard disease/intervention outputs.
"""

from collections import defaultdict
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
            ss.Result('n_deliveries',   dtype=int,   label='Deliveries'),
            ss.Result('n_ptb',          dtype=int,   label='Preterm births'),
            ss.Result('n_lbw',          dtype=int,   label='LBW births'),
            ss.Result('n_ptb_lbw',      dtype=int,   label='PTB + LBW'),
            ss.Result('n_stillbirths',  dtype=int,   label='Stillbirths'),
            ss.Result('yld_ptb',        scale=False, label='YLD — preterm birth'),
            ss.Result('yld_lbw',        scale=False, label='YLD — LBW only'),
            ss.Result('dalys',          scale=False, label='DALYs'),
            ss.Result('cum_dalys',      scale=False, label='Cumulative DALYs'),
        )

    def step(self):
        sim = self.sim
        ti  = self.ti
        if sim.t.yearvec[ti] < self.start:
            return

        # Prefer the module's own per-timestep results (starsim Pregnancy tracks
        # n_preterm / n_very_preterm / stillbirths natively; FetalHealth tracks
        # n_lbw / n_vlbw / n_sga natively).
        preg_res = sim.demographics.pregnancy.results if hasattr(sim.demographics, 'pregnancy') else None
        fh_res   = sim.custom['fetal_health'].results if 'fetal_health' in sim.custom else None

        n_deliv = int(preg_res['births'].values[ti]) if preg_res and 'births' in preg_res else 0
        n_ptb   = int(preg_res['n_preterm'].values[ti]) if preg_res and 'n_preterm' in preg_res else 0
        n_lbw   = int(fh_res['n_lbw'].values[ti]) if fh_res and 'n_lbw' in fh_res else 0
        n_still = int(preg_res['stillbirths'].values[ti]) if preg_res and 'stillbirths' in preg_res else 0

        # PTB and LBW overlap is common; we can't easily count from module-level
        # results alone. Approximate: assume all LBW are also PTB (mechanistic
        # model produces this) — so n_ptb_lbw ~= n_lbw.
        n_ptb_lbw = n_lbw
        n_lbw_only = max(0, n_lbw - n_ptb_lbw)

        yld_ptb = n_ptb      * self.dw_ptb * self.dur_ptb
        yld_lbw = n_lbw_only * self.dw_lbw * self.dur_lbw
        dalys   = yld_ptb + yld_lbw

        self.results['n_deliveries'][ti]  = n_deliv
        self.results['n_ptb'][ti]         = n_ptb
        self.results['n_lbw'][ti]         = n_lbw
        self.results['n_ptb_lbw'][ti]     = n_ptb_lbw
        self.results['n_stillbirths'][ti] = n_still
        self.results['yld_ptb'][ti]       = yld_ptb
        self.results['yld_lbw'][ti]       = yld_lbw
        self.results['dalys'][ti]         = dalys

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
            fh = sim.custom['fetal_health']
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


class SyphTransmissionEvents(ss.Analyzer):
    """Aggregate syph transmission counts for Lorenz + transmission matrix.

    Args:
        events_window: (year_start, year_end) for the transmission matrix
            aggregation. Per-source counts always cover the full sim.
        name: analyzer name (default 'syph_transmission_events')
    """

    def __init__(self, events_window=(2010, 2025), name='syph_transmission_events',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.events_window = events_window
        self.src_count = defaultdict(int)
        self.matrix = defaultdict(int)
        self.by_year = defaultdict(lambda: defaultdict(int))
        return

    def init_post(self):
        super().init_post()
        sim = self.sim
        # The model may not have syph (e.g. discharging-only runs); be defensive
        if not hasattr(sim.diseases, 'syph'):
            return
        syph = sim.diseases.syph
        nw = sim.networks.structuredsexual
        ppl = sim.people
        original = syph.set_prognoses
        src_count = self.src_count
        matrix = self.matrix
        by_year = self.by_year
        events_lo, events_hi = self.events_window

        def categorize(uid):
            if ppl.female[uid]:
                return 'F_fsw' if nw.fsw[uid] else 'F_other'
            return 'M_client' if nw.client[uid] else 'M_other'

        def stage_of(uid):
            if syph.primary[uid]: return 'primary'
            if syph.secondary[uid]: return 'secondary'
            if syph.early[uid]: return 'early_latent'
            if syph.late[uid]: return 'late_latent'
            return 'unknown'

        def instrumented(uids, source_uids=None, ti=None):
            if source_uids is not None and len(source_uids) > 0:
                cti = ti if ti is not None else syph.ti
                try:
                    year = int(syph.t.timevec[cti].year)
                except Exception:
                    year = -1
                src_arr = np.atleast_1d(source_uids)
                dst_arr = np.atleast_1d(uids)
                in_window = events_lo <= year < events_hi
                for s, d in zip(src_arr, dst_arr):
                    src_count[int(s)] += 1
                    if in_window:
                        key = (categorize(s), categorize(d), stage_of(s))
                        matrix[key] += 1
                        by_year[year][key] += 1
            return original(uids, source_uids, ti)

        syph.set_prognoses = instrumented
        return

    def step(self):
        # All work happens in the monkey-patched set_prognoses.
        return

    def as_dict(self):
        """Serializable snapshot for outputs."""
        return {
            'src_count': dict(self.src_count),
            'matrix': {f'{k[0]}|{k[1]}|{k[2]}': v
                       for k, v in self.matrix.items()},
            'by_year': {str(y): {f'{k[0]}|{k[1]}|{k[2]}': v
                                  for k, v in d.items()}
                        for y, d in self.by_year.items()},
            'events_window': list(self.events_window),
        }


class CareTimingAnalyzer(ss.Analyzer):
    """Per-episode "treated within N months of acquisition" metric, for
    one or more windows simultaneously (3mo + 6mo, etc.).

    Stricter than ``tx_success / new_inf`` (which counts ALL successful
    treatments and ALL new infections in window — re-infections inflate
    both num and denom, and treatments of pre-window infections inflate
    only the numerator). This metric is per-episode:

      ``{d}_inf_treated_within_{N}mo`` = "agent was newly infected at
      time T then successfully treated within N months of T".

    For each disease tracks a per-agent ``ti_last_inf`` (overwritten on
    every new infection event for that agent), then on each step
    inspects every linked treatment's ``outcomes[disease].successful``
    uids; for each successful uid checks whether
    ``(ti - ti_last_inf) <= window_steps_N`` for each window N. If yes,
    increments the corresponding result. A cure at 4 months counts for
    the 6mo result but not the 3mo result.

    Args:
        disease_names: list of disease names to track.
        treatment_disease_map: dict mapping treatment intervention name
            to disease name.
        windows_months: list of cure-timing windows in months
            (default [3, 6]).
        name: analyzer name (default 'care_timing').

    Reads:
        sim.diseases[d].ti_infected per step.
        sim.interventions[tx].outcomes[d].successful per step.

    Writes (per disease, per window):
        results[f'{d}_inf_treated_within_{N}mo'], indexed by the
        timestep on which the CURE happened. Sum over window for the
        numerator; pair with sim.results[d].new_infections for the
        denominator.

    Backwards-compat: accepts ``window_months=`` (singular) as well;
    converted to a one-element list internally.
    """
    def __init__(self, disease_names, treatment_disease_map,
                 windows_months=None, window_months=None,
                 name='care_timing', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.disease_names = list(disease_names)
        self.treatment_disease_map = dict(treatment_disease_map)
        if windows_months is None and window_months is not None:
            windows_months = [int(window_months)]
        if windows_months is None:
            windows_months = [3, 6]
        self.windows_months = [int(w) for w in windows_months]
        states = [ss.FloatArr(f'{d}_ti_last_inf', default=np.nan)
                  for d in self.disease_names]
        self.define_states(*states)

    def init_results(self):
        super().init_results()
        results = sc.autolist()
        for d in self.disease_names:
            for w in self.windows_months:
                results += [
                    ss.Result(f'{d}_inf_treated_within_{w}mo', dtype=int,
                              label=(f'{d} infections treated within '
                                     f'{w}mo of acquisition'),
                              auto_plot=False),
                ]
        self.define_results(*results)

    def step(self):
        sim = self.sim
        ti = self.ti
        dt_year = sim.t.dt_year if sim.t.dt_year else 1/12
        window_steps = {w: max(1, int(round(w / 12.0 / dt_year)))
                        for w in self.windows_months}

        # 1. Update ti_last_inf for agents newly infected this step.
        for d in self.disease_names:
            disease = sim.diseases.get(d)
            if disease is None:
                continue
            ti_arr = getattr(self, f'{d}_ti_last_inf')
            new_inf = (disease.ti_infected == ti).uids
            if len(new_inf):
                ti_arr[new_inf] = ti

        # 2. For each tracked treatment, check window membership.
        for tx_name, d in self.treatment_disease_map.items():
            tx = sim.interventions.get(tx_name)
            if tx is None:
                continue
            outcomes = getattr(tx, 'outcomes', None)
            if outcomes is None:
                continue
            disease_out = outcomes.get(d) if hasattr(outcomes, 'get') else None
            if disease_out is None:
                continue
            succ = disease_out.get('successful') if hasattr(disease_out, 'get') \
                   else getattr(disease_out, 'successful', None)
            if succ is None or len(succ) == 0:
                continue
            ti_arr = getattr(self, f'{d}_ti_last_inf')
            last_inf = ti_arr[succ]
            valid = ~np.isnan(last_inf)
            gap = ti - last_inf
            for w, n_steps in window_steps.items():
                in_window = valid & (gap <= n_steps)
                n_in = int(in_window.sum())
                if n_in:
                    self.results[f'{d}_inf_treated_within_{w}mo'][ti] += n_in
        return
