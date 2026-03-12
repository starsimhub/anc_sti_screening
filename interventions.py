"""
Interventions for the ANC STI screening model.

Contains:
    - SyndromicMgmt: syndromic management of VDS/UDS
    - ANCScreen: GA-windowed ANC screening for asymptomatic STIs
    - STIPartnerNotification: notify/treat partners of ANC-positive women
    - make_testing: factory function to assemble all STI interventions
"""

import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti


def count(arr):
    return np.count_nonzero(arr)


# %% Syndromic management
class SyndromicMgmt(sti.STITest):
    """
    Syndromic management of vaginal/urethral discharge syndromes.

    When a symptomatic patient presents, they are probabilistically assigned
    one of four treatment outcomes based on whether they have cervical
    infection (NG/CT) and their sex:
        - all3: treated for NG + CT + TV/BV
        - ngct: treated for NG + CT only
        - mtnz: treated with metronidazole only (TV/BV)
        - none: dismissed without treatment
    """

    def __init__(self, pars=None, treatments=None, diseases=None,
                 outcome_treatment_map=None, treat_prob_data=None,
                 years=None, start=None, stop=None, eligibility=None,
                 name=None, label=None, **kwargs):
        super().__init__(years=years, start=start, stop=stop, eligibility=eligibility, name=name, label=label)
        self.define_pars(
            tx_mix_cerv=dict(
                all3=[0.50, 0.05],
                ngct=[0.20, 0.80],
                mtnz=[0.20, 0.00],
                none=[0.10, 0.15],
            ),
            tx_mix_noncerv=dict(
                all3=[0.40, 0.05],
                ngct=[0.10, 0.80],
                mtnz=[0.20, 0.00],
                none=[0.30, 0.15],
            ),
            tx_cerv_f=ss.choice(a=4),
            tx_cerv_m=ss.choice(a=4),
            tx_noncerv_f=ss.choice(a=4),
            tx_noncerv_m=ss.choice(a=4),
            dt_scale=False,
        )
        self.update_pars(pars, **kwargs)
        self.fvals_cerv    = [v[0] for v in self.pars.tx_mix_cerv.values()]
        self.mvals_cerv    = [v[1] for v in self.pars.tx_mix_cerv.values()]
        self.fvals_noncerv = [v[0] for v in self.pars.tx_mix_noncerv.values()]
        self.mvals_noncerv = [v[1] for v in self.pars.tx_mix_noncerv.values()]

        # Store treatments and diseases
        self.treatments = sc.tolist(treatments)
        self.diseases = diseases
        if outcome_treatment_map is None:
            outcome_treatment_map = dict(
                all3=self.treatments,
                ngct=[self.treatments[0], self.treatments[1]],
                mtnz=[self.treatments[1]],
                none=[],
            )
        self.outcome_treatment_map = outcome_treatment_map

        self.define_states(
            ss.FloatArr('ti_referred'),
            ss.FloatArr('ti_dismissed'),
        )
        self.treat_prob_data = treat_prob_data
        self.treat_prob = None
        self.treated_by_uid = None

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.pars.tx_cerv_f.set(p=self.fvals_cerv)
        self.pars.tx_cerv_m.set(p=self.mvals_cerv)
        self.pars.tx_noncerv_f.set(p=self.fvals_noncerv)
        self.pars.tx_noncerv_m.set(p=self.mvals_noncerv)
        return

    def init_results(self):
        super().init_results()
        results = sc.autolist()
        sexkeys = ['', 'f', 'm']
        for sk in sexkeys:
            skk = '' if sk == '' else f'_{sk}'
            skl = '' if sk == '' else f' - {sk.upper()}'
            results += [
                ss.Result('new_care_seekers' + skk, dtype=int, label='Care seekers' + skl),
                ss.Result('new_tx0' + skk, dtype=int, label='No treatment' + skl),
                ss.Result('new_tx1' + skk, dtype=int, label='1 treatment' + skl),
                ss.Result('new_tx2' + skk, dtype=int, label='2 treatments' + skl),
                ss.Result('new_tx3' + skk, dtype=int, label='3 treatments' + skl),
            ]
        self.define_results(*results)
        return

    def step(self, uids=None):
        sim = self.sim
        ppl = sim.people
        self.treated_by_uid = None

        if sim.now >= self.stop:
            for treatment in self.treatments:
                treatment.eligibility = ss.uids()
            return

        if sim.now >= self.start:

            if uids is None:
                uids = self.check_eligibility()
                self.ti_tested[uids] = self.ti

            if len(uids):
                f_uids = uids[ppl.female[uids]]
                m_uids = uids[ppl.male[uids]]

                # Determine who has symptomatic cervical infection
                is_cerv = ppl.ng.symptomatic | ppl.ct.symptomatic

                # Treatment outcomes by cervical status and sex
                f_cerv_uids    = f_uids[is_cerv[f_uids]]
                f_noncerv_uids = f_uids[~is_cerv[f_uids]]

                ofc  = self.pars.tx_cerv_f.rvs(f_cerv_uids)
                ofnc = self.pars.tx_noncerv_f.rvs(f_noncerv_uids)
                om   = self.pars.tx_cerv_m.rvs(m_uids)

                outcomes = dict(
                    all3=f_cerv_uids[ofc == 0] | f_noncerv_uids[ofnc == 0] | m_uids[om == 0],
                    ngct=f_cerv_uids[ofc == 1] | f_noncerv_uids[ofnc == 1] | m_uids[om == 1],
                    mtnz=f_cerv_uids[ofc == 2] | f_noncerv_uids[ofnc == 2] | m_uids[om == 2],
                    none=f_cerv_uids[ofc == 3] | f_noncerv_uids[ofnc == 3] | m_uids[om == 3],
                )

                # Track diagnostic accuracy per disease
                for disease in self.diseases:
                    for pkey, pattr in disease.sex_keys.items():
                        skk = '' if pkey == '' else f'_{pkey}'
                        disease.results[f'new_true_pos{skk}'][self.ti]  += len(outcomes['all3'] & disease.treatable & ppl[pattr])
                        disease.results[f'new_false_pos{skk}'][self.ti] += len(outcomes['all3'] & disease.susceptible & ppl[pattr])
                        disease.results[f'new_true_neg{skk}'][self.ti]  += len(outcomes['none'] & disease.susceptible & ppl[pattr])
                        disease.results[f'new_false_neg{skk}'][self.ti] += len(outcomes['none'] & disease.treatable & ppl[pattr])

                # Additional NG/CT-specific accuracy
                for disease in [sim.diseases.ng, sim.diseases.ct]:
                    for pkey, pattr in disease.sex_keys.items():
                        skk = '' if pkey == '' else f'_{pkey}'
                        disease.results[f'new_true_pos{skk}'][self.ti]  += len(outcomes['ngct'] & disease.treatable & ppl[pattr])
                        disease.results[f'new_false_pos{skk}'][self.ti] += len(outcomes['ngct'] & disease.susceptible & ppl[pattr])
                        disease.results[f'new_false_neg{skk}'][self.ti] += len(outcomes['mtnz'] & disease.treatable & ppl[pattr])
                        disease.results[f'new_true_neg{skk}'][self.ti]  += len(outcomes['mtnz'] & disease.susceptible & ppl[pattr])

                for disease in [sim.diseases.tv]:
                    for pkey, pattr in disease.sex_keys.items():
                        skk = '' if pkey == '' else f'_{pkey}'
                        disease.results[f'new_true_pos{skk}'][self.ti]  += len(outcomes['mtnz'] & disease.treatable & ppl[pattr])
                        disease.results[f'new_false_pos{skk}'][self.ti] += len(outcomes['mtnz'] & disease.susceptible & ppl[pattr])
                        disease.results[f'new_false_neg{skk}'][self.ti] += len(outcomes['ngct'] & disease.treatable & ppl[pattr])
                        disease.results[f'new_true_neg{skk}'][self.ti]  += len(outcomes['ngct'] & disease.susceptible & ppl[pattr])

                # Update treatment eligibility
                for outcome, txs in self.outcome_treatment_map.items():
                    for tx in txs:
                        tx.eligibility = tx.eligibility | outcomes[outcome]

                # Track referral/dismissal
                referred_uids  = outcomes['all3'] | outcomes['ngct'] | outcomes['mtnz']
                dismissed_uids = outcomes['none']
                self.ti_referred[referred_uids] = self.ti
                self.ti_dismissed[dismissed_uids] = self.ti
                self.treated_by_uid = outcomes

            self.store_results()

        return

    def store_results(self):
        """ Store results before treatments are applied, so infection status is accurate """
        ti = self.ti
        ppl = self.sim.people
        just_tested = self.ti_tested == ti
        self.results['new_care_seekers'][ti]   += count(just_tested)
        self.results['new_care_seekers_f'][ti] += count(just_tested & ppl.female)
        self.results['new_care_seekers_m'][ti] += count(just_tested & ppl.male)

        sexdict = {'': 'alive', 'f': 'female', 'm': 'male'}
        if self.treated_by_uid is not None:
            for sk, sl in sexdict.items():
                skk = '' if sk == '' else f'_{sk}'
                self.results['new_tx0' + skk][ti] += count(ppl[sl][self.treated_by_uid['none']])
                self.results['new_tx1' + skk][ti] += count(ppl[sl][self.treated_by_uid['mtnz']])
                self.results['new_tx2' + skk][ti] += count(ppl[sl][self.treated_by_uid['ngct']])
                self.results['new_tx3' + skk][ti] += count(ppl[sl][self.treated_by_uid['all3']])

        return


# %% ANC screening
class ANCScreen(sti.STITest):
    """
    ANC-based screening for asymptomatic STIs during pregnancy.

    Screens pregnant women for NG, CT, TV (and optionally BV) within a
    gestational age window. Multiple instances can be used for timed
    screens (e.g. enrollment ≤24w + third-trimester 32-34w).

    Args:
        diseases (list):               disease modules to screen for
        treatments (list):             treatment modules to apply
        disease_treatment_map (dict):  maps disease name → treatment module
        test_sensitivity (dict):       per-disease test sensitivity
        screen_prob (float/array):     probability of being screened at ANC
        screen_prob_data (array):      time-varying screening probability
        years (array):                 years corresponding to screen_prob_data
        ga_min (float):                minimum gestational age (weeks) for eligibility
        ga_max (float):                maximum gestational age (weeks) for eligibility
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
        if disease_treatment_map is None and treatments is not None:
            disease_treatment_map = {}
        self.disease_treatment_map = disease_treatment_map

        # Per-disease test sensitivity (default: perfect test)
        if test_sensitivity is None:
            test_sensitivity = {d.name: 1.0 for d in self.diseases}
        self.test_sensitivity = test_sensitivity

        # GA window for eligibility (in weeks); None means no constraint
        self.ga_min = ga_min
        self.ga_max = ga_max

        # Distribution for sensitivity sampling
        self._sens_dist = ss.bernoulli(p=0.5)

        self.screen_prob_data = screen_prob_data
        self._screen_prob_interp = None

        return

    def init_pre(self, sim):
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
        never_tested = np.isnan(self.ti_tested.values)
        tested_before_this_pregnancy = self.ti_tested < ppl.pregnancy.ti_pregnant
        eligible = pregnant & ppl.female & (never_tested | tested_before_this_pregnancy)

        if not eligible.any():
            return

        eligible_uids = eligible.uids

        # Filter by gestational age window (in weeks) if specified
        if self.ga_min is not None or self.ga_max is not None:
            ga_weeks = np.asarray(ppl.pregnancy.gestation[eligible_uids], dtype=float)
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
