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
