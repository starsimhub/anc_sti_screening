"""
Interventions for the ANC STI screening model.

Contains:
    - ANCScreen: GA-windowed, per-timestep STI screening (PROMISE trial design)
    - STIPartnerNotification: notify/treat partners of ANC-positive women
    - make_testing: factory function to assemble all STI interventions

Note: ANCScreen here is intentionally project-specific. The general-purpose
ANC intervention for real-world modelling (scheduled once per pregnancy,
auto-detects HIV + syphilis) is sti.ANCTest in core stisim.

SyndromicMgmt is in core stisim — use sti.SyndromicMgmt directly.
"""

import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti

SyndromicMgmt = sti.SyndromicMgmt  # re-export for any local code that imports by name


class ANCScreen(sti.STITest):
    """
    GA-windowed ANC screening for the PROMISE trial VoI analysis.

    Screens pregnant women for NG, CT, TV (and optionally BV) within a
    gestational age window each timestep. Multiple instances model timed
    screens (enrollment ≤24w + third-trimester 32-34w). Each instance
    maintains its own ti_tested so a woman is screened at most once per
    visit type per pregnancy.

    This is a project-specific class for the PROMISE trial design.
    For general real-world ANC modelling use sti.ANCTest instead.
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

        self.diseases = sc.tolist(diseases)
        self.treatments = sc.tolist(treatments)
        self.disease_treatment_map = disease_treatment_map or {}
        self.test_sensitivity = test_sensitivity or {d.name: 1.0 for d in self.diseases}
        self.ga_min = ga_min
        self.ga_max = ga_max
        self._sens_dist = ss.bernoulli(p=0.5)
        self.screen_prob_data = screen_prob_data
        self._screen_prob_interp = None
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.screen_prob_data is not None and self.pars.years is not None:
            self._screen_prob_interp = sc.smoothinterp(
                sim.t.yearvec, self.pars.years, self.screen_prob_data, smoothness=0,
            )
        return

    def init_results(self):
        super().init_results()
        results = sc.autolist()
        results += ss.Result('n_screened', dtype=int, label='Women screened')
        results += ss.Result('n_positive', dtype=int, label='Women testing positive')
        for d in self.diseases:
            results += [
                ss.Result(f'n_{d.name}_detected',  dtype=int, label=f'{d.name.upper()} detected'),
                ss.Result(f'n_{d.name}_true_pos',  dtype=int, label=f'{d.name.upper()} true positive'),
                ss.Result(f'n_{d.name}_false_pos', dtype=int, label=f'{d.name.upper()} false positive'),
                ss.Result(f'n_{d.name}_false_neg', dtype=int, label=f'{d.name.upper()} false negative'),
            ]
        self.define_results(*results)
        return

    def step(self):
        sim = self.sim
        ppl = sim.people
        ti  = self.ti

        if sim.now < self.start or sim.now >= self.stop:
            return

        pregnant = ppl.pregnancy.pregnant
        never_tested = np.isnan(self.ti_tested.values)
        tested_before_preg = self.ti_tested < ppl.pregnancy.ti_pregnant
        eligible = pregnant & ppl.female & (never_tested | tested_before_preg)

        if not eligible.any():
            return

        eligible_uids = eligible.uids

        if self.ga_min is not None or self.ga_max is not None:
            ga = np.asarray(ppl.pregnancy.gestation[eligible_uids], dtype=float)
            mask = np.ones(len(eligible_uids), dtype=bool)
            if self.ga_min is not None: mask &= ga >= self.ga_min
            if self.ga_max is not None: mask &= ga <= self.ga_max
            eligible_uids = eligible_uids[mask]

        if not len(eligible_uids):
            return

        if self._screen_prob_interp is not None:
            self.pars.screen_prob.set(p=self._screen_prob_interp[ti])

        screened_uids = eligible_uids[self.pars.screen_prob.rvs(eligible_uids)]
        if not len(screened_uids):
            return

        self.ti_tested[screened_uids] = ti
        self.results['n_screened'][ti] = len(screened_uids)

        any_positive = ss.uids()
        for disease in self.diseases:
            dname = disease.name
            sens  = self.test_sensitivity.get(dname, 1.0)

            infected   = screened_uids[disease.infected[screened_uids]]
            uninfected = screened_uids[~disease.infected[screened_uids]]

            if len(infected) and sens < 1.0:
                self._sens_dist.set(p=sens)
                det       = self._sens_dist.rvs(infected)
                true_pos  = infected[det]
                false_neg = infected[~det]
            else:
                true_pos  = infected
                false_neg = ss.uids()
            false_pos = ss.uids()
            detected  = true_pos | false_pos

            self.results[f'n_{dname}_detected'][ti]  = len(detected)
            self.results[f'n_{dname}_true_pos'][ti]  = len(true_pos)
            self.results[f'n_{dname}_false_pos'][ti] = len(false_pos)
            self.results[f'n_{dname}_false_neg'][ti] = len(false_neg)

            if hasattr(disease, 'sex_keys'):
                for pkey, pattr in disease.sex_keys.items():
                    skk = '' if pkey == '' else f'_{pkey}'
                    disease.results[f'new_true_pos{skk}'][ti]  += len(true_pos  & ppl[pattr])
                    disease.results[f'new_false_pos{skk}'][ti] += len(false_pos & ppl[pattr])
                    disease.results[f'new_false_neg{skk}'][ti] += len(false_neg & ppl[pattr])
                    disease.results[f'new_true_neg{skk}'][ti]  += len(uninfected & ppl[pattr])

            if dname in self.disease_treatment_map and len(detected):
                self.disease_treatment_map[dname].eligibility |= detected

            any_positive = any_positive | detected

        self.results['n_positive'][ti] = len(any_positive)
        return


# %% Partner notification
class STIPartnerNotification(ss.Intervention):
    """
    Notify and treat sexual partners of women testing positive at ANC.

    When a woman tests positive for an STI at ANC screening, her current
    sexual partner(s) are identified via the network and probabilistically
    notified and treated. This reduces reinfection risk between screens.

    Args:
        p_partner_tx (float):   probability that a partner is successfully
                                notified AND treated (default 0.3)
        anc_screens (list):     names of ANCScreen interventions to monitor
        treatments (list):      treatment modules to apply to partners
        disease_treatment_map:  maps disease name → treatment module
    """

    def __init__(self, p_partner_tx=0.3, anc_screens=None,
                 treatments=None, disease_treatment_map=None,
                 name=None, label=None, start=None, **kwargs):
        super().__init__(name=name, label=label, **kwargs)
        self.define_pars(
            p_notify_treat=ss.bernoulli(p=p_partner_tx),
        )
        self.anc_screen_names = anc_screens or ['anc_enroll', 'anc_tri3']
        self.treatments = treatments or []
        self.disease_treatment_map = disease_treatment_map or {}
        self.start = start

        self.define_states(
            ss.FloatArr('ti_notified', label='Time partner was notified'),
        )
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        if self.start is None:
            self.start = sim.t.yearvec[0]

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_index_cases',      dtype=int, label='Index cases (ANC positive)'),
            ss.Result('n_partners_found',   dtype=int, label='Partners identified'),
            ss.Result('n_partners_treated', dtype=int, label='Partners notified & treated'),
        )

    def step(self):
        sim = self.sim
        ti = self.ti

        if sim.now < self.start:
            return

        # Find women who just tested positive at any ANC screen this timestep
        index_uids = ss.uids()
        for screen_name in self.anc_screen_names:
            anc = sim.interventions.get(screen_name)
            if anc is None:
                continue
            # Women screened this timestep who tested positive for any disease
            just_screened = anc.ti_tested == ti
            for disease in anc.diseases:
                positive = just_screened & disease.infected
                index_uids = index_uids | positive.uids

        if len(index_uids) == 0:
            return

        self.results['n_index_cases'][ti] = len(index_uids)

        # Find current sexual partners via network
        nw = sim.networks.structuredsexual
        partners = nw.find_contacts(index_uids, as_array=False)
        partners = ss.uids(partners)

        self.results['n_partners_found'][ti] = len(partners)

        if len(partners) == 0:
            return

        # Probabilistically notify and treat
        treated_partners = self.pars.p_notify_treat.filter(partners)
        self.ti_notified[treated_partners] = ti
        self.results['n_partners_treated'][ti] = len(treated_partners)

        # Route to treatment — treat for all diseases the index was positive for
        if len(treated_partners):
            for dname, tx in self.disease_treatment_map.items():
                # Treat partners who are infected (even if asymptomatic)
                infected_partners = treated_partners[sim.diseases[dname].infected[treated_partners]]
                if len(infected_partners):
                    tx.eligibility = tx.eligibility | infected_partners

        return


# %% Factory functions
def seeking_care_vds(sim):
    """ Eligibility: women with symptomatic vaginal discharge seeking care """
    dis = sim.diseases
    female = sim.people.female
    ng_care = dis.ng.symptomatic & (dis.ng.ti_seeks_care == dis.ng.ti) & female
    tv_care = dis.tv.symptomatic & (dis.tv.ti_seeks_care == dis.tv.ti) & female
    ct_care = dis.ct.symptomatic & (dis.ct.ti_seeks_care == dis.ct.ti) & female
    bv_care = dis.bv.symptomatic & (dis.bv.ti_seeks_care == dis.bv.ti) & female
    return (ng_care | ct_care | tv_care | bv_care).uids


def seeking_care_uds(sim):
    """ Eligibility: men with symptomatic urethral discharge seeking care """
    dis = sim.diseases
    male = sim.people.male
    ng_care = dis.ng.symptomatic & (dis.ng.ti_seeks_care == dis.ng.ti) & male
    tv_care = dis.tv.symptomatic & (dis.tv.ti_seeks_care == dis.tv.ti) & male
    ct_care = dis.ct.symptomatic & (dis.ct.ti_seeks_care == dis.ct.ti) & male
    return (ng_care | ct_care | tv_care).uids


def make_testing(ng, ct, tv, bv, scenario='soc', stop=2040):
    """
    Create all STI testing and treatment interventions.

    Args:
        ng, ct, tv, bv: disease module instances
        scenario (str):  screening scenario
        stop (int):      simulation end year

    Scenarios:
        soc:        syndromic management only (standard of care)
        enroll:     single enrollment screen (≤24w GA)
        tri3:       single third-trimester screen (32-34w GA)
        twice:      both screens (PROMISE trial design)
        partner_tx: both screens + partner notification/treatment
    """
    # Treatments
    ng_tx = sti.GonorrheaTreatment(
        name='ng_tx',
        rel_treat_unsucc=0.005,
        rel_treat_unneed=0.0005,
    )
    ct_tx = sti.STITreatment(diseases='ct', name='ct_tx', label='ct_tx')
    metronidazole = sti.STITreatment(diseases=['tv', 'bv'], name='metronidazole', label='metronidazole')
    treatments = [ng_tx, ct_tx, metronidazole]

    outcome_treatment_map = dict(
        all3=treatments,
        ngct=[ng_tx, ct_tx],
        mtnz=[metronidazole],
        none=[],
    )

    # Syndromic management of VDS
    syndromic_vds = SyndromicMgmt(
        name='syndromic_vds',
        label='syndromic_vds',
        stop=stop,
        diseases=[ng, ct, tv, bv],
        eligibility=seeking_care_vds,
        treatments=treatments,
        outcome_treatment_map=outcome_treatment_map,
    )

    # Syndromic management of UDS
    syndromic_uds = SyndromicMgmt(
        name='syndromic_uds',
        label='syndromic_uds',
        stop=stop,
        diseases=[ng, ct, tv],
        eligibility=seeking_care_uds,
        treatments=treatments,
        outcome_treatment_map=outcome_treatment_map,
    )

    intvs = [syndromic_vds, syndromic_uds, ng_tx, ct_tx, metronidazole]

    # ANC screening scenarios — mapped to PROMISE trial design:
    #   soc:         syndromic management only (standard of care)
    #   enroll:      single enrollment screen (≤24w)
    #   tri3:        single third-trimester screen (32-34w)
    #   twice:       both enrollment + third-trimester screens (PROMISE design)
    #   partner_tx:  twice + partner notification/treatment

    disease_treatment_map = {'ng': ng_tx, 'ct': ct_tx, 'tv': metronidazole}
    default_sensitivity  = {'ng': 0.95, 'ct': 0.95, 'tv': 0.90}
    intv_year = 2027

    def _make_anc_screen(name, ga_min=None, ga_max=None):
        return ANCScreen(
            name=name, label=name,
            start=intv_year,
            diseases=[ng, ct, tv],
            treatments=treatments,
            disease_treatment_map=disease_treatment_map,
            screen_prob=0.5,
            test_sensitivity=default_sensitivity,
            ga_min=ga_min, ga_max=ga_max,
        )

    if scenario == 'enroll':
        # Single enrollment screen: ≤24 weeks GA
        intvs.append(_make_anc_screen('anc_enroll', ga_max=24))

    elif scenario == 'tri3':
        # Single third-trimester screen: ≥28 weeks GA (third trimester)
        intvs.append(_make_anc_screen('anc_tri3', ga_min=28))

    elif scenario in ['twice', 'partner_tx']:
        # PROMISE design: enrollment (≤24w) + third-trimester (≥28w)
        # Each instance has its own ti_tested, so women are screened once per visit
        intvs.append(_make_anc_screen('anc_enroll', ga_max=24))
        intvs.append(_make_anc_screen('anc_tri3', ga_min=28))

        if scenario == 'partner_tx':
            pn = STIPartnerNotification(
                name='partner_notif', label='partner_notif',
                p_partner_tx=0.3,
                anc_screens=['anc_enroll', 'anc_tri3'],
                treatments=treatments,
                disease_treatment_map=disease_treatment_map,
                start=intv_year,
            )
            intvs.append(pn)

    # soc: no ANC screening (default — intvs unchanged)

    return intvs
