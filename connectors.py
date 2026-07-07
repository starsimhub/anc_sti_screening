"""
Connectors between STI diseases and the FetalHealth module.

Adapted from the anc_sti_screening repo. The sti_fetal connector holds all
disease-specific parameters (delivery timing shifts, growth penalties,
treatment reversibility) and translates STI events into calls to FetalHealth's
generic API. Compared to the ANC version, this adds syphilis as a covered
disease — syphilis has the largest fetal-health effect of the seven STIs in
this model.

Performance note: this runs once per timestep and only operates on women
who were newly infected or treated that step, so cost scales with incident
events, not population size.
"""

import numpy as np
import sciris as sc
import starsim as ss


class sti_fetal(ss.Connector):
    """
    Connect STI disease modules to FetalHealth.

    Args:
        diseases (list):             disease names to monitor
            (default: ng, ct, tv, syph)
        treatments (list):           treatment intervention names
        treatment_disease_map (dict): maps treatment name -> disease name
        ptb_shift_mean (dict):       mean weeks delivery brought forward, per disease
        ptb_shift_std (float):       individual heterogeneity in timing shift
        growth_penalty (dict):       fractional weight reduction per infection, per disease
        tx_residual_growth (dict):   fraction of growth damage persisting after treatment, per trimester
        tx_residual_timing (dict):   fraction of timing damage persisting after treatment, per trimester
    """

    def __init__(self, diseases=None, treatments=None, treatment_disease_map=None,
                 pars=None, **kwargs):
        super().__init__()
        self.name = 'sti_fetal'
        self.disease_names = diseases or ['ng', 'ct', 'tv', 'syph']
        self.treatment_names = treatments or ['ng_tx', 'ct_tx', 'metronidazole', 'syph_tx']
        if treatment_disease_map is None:
            treatment_disease_map = {
                'ng_tx':         'ng',
                'ct_tx':         'ct',
                'metronidazole': 'tv',
                'syph_tx':       'syph',
            }
        self.treatment_disease_map = treatment_disease_map

        self.define_pars(
            # Delivery timing shift per infection (mean weeks brought forward)
            # Syphilis carries the largest effect in untreated pregnancies; NG/CT/TV
            # are smaller modifiers (Gomez 2013, Newman 2013, Korenromp 2019).
            ptb_shift_mean=sc.objdict(ng=2.0, ct=1.5, tv=1.0, syph=4.0),
            ptb_shift_std=1.0,

            # Growth restriction per infection (fractional weight reduction)
            growth_penalty=sc.objdict(ng=0.08, ct=0.03, tv=0.03, syph=0.12),

            # Treatment reversibility -- fraction of damage PERSISTING after treatment, by trimester
            # T1: most reversible (early treatment), T3: least reversible (damage locked in)
            tx_residual_growth=sc.objdict(tri1=0.25, tri2=0.40, tri3=0.60),
            tx_residual_timing=sc.objdict(tri1=0.35, tri2=0.55, tri3=0.75),

            # Distribution for sampling individual timing shifts (mean/std set dynamically per disease)
            ptb_shift_dist=ss.lognorm_ex(mean=1.0, std=1.0),
        )
        self.update_pars(pars, **kwargs)

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        try:
            fh = sim.custom['fetal_health']
            fh.add_conception_callback(self._on_conception)
        except (KeyError, AttributeError):
            pass
        return

    def _get_fh(self):
        try:
            return self.sim.custom['fetal_health']
        except (KeyError, AttributeError):
            return None

    def _on_conception(self, uids):
        """Check for pre-existing infections at the start of pregnancy."""
        for dname in self.disease_names:
            try:
                disease = self.sim.diseases[dname]
                infected_uids = uids[disease.infected[uids]]
                if len(infected_uids):
                    self._apply_infection(infected_uids, dname)
            except (KeyError, AttributeError):
                pass
        return

    def _apply_infection(self, uids, disease_name):
        """Apply infection effects on fetal health (timing shift + growth restriction)."""
        fh = self._get_fh()
        if fh is None:
            return

        shift_mean = self.pars.ptb_shift_mean.get(disease_name, 0)
        if shift_mean > 0:
            self.pars.ptb_shift_dist.set(mean=shift_mean, std=float(self.pars.ptb_shift_std))
            shifts = self.pars.ptb_shift_dist.rvs(uids)
            fh.apply_timing_shift(uids, shifts)

        penalty = self.pars.growth_penalty.get(disease_name, 0)
        if penalty > 0:
            fh.apply_growth_restriction(uids, penalty)

        return

    def _apply_treatment(self, uids, disease_name):
        """Reverse fetal health damage when an infection is treated during pregnancy."""
        fh = self._get_fh()
        if fh is None or not len(uids):
            return

        # Trimester from gestational age
        preg = self.sim.people.pregnancy
        ga_weeks   = preg.gestation[uids]
        boundaries = preg.pars.trimesters
        b1 = boundaries[0].weeks
        b2 = boundaries[1].weeks
        trimester = np.ones(len(uids), dtype=int)
        trimester[ga_weeks >= b1] = 2
        trimester[ga_weeks >= b2] = 3

        # Growth restriction reversal
        penalty = self.pars.growth_penalty.get(disease_name, 0)
        if penalty > 0:
            residual = np.select(
                [trimester == 1, trimester == 2, trimester == 3],
                [self.pars.tx_residual_growth.tri1, self.pars.tx_residual_growth.tri2, self.pars.tx_residual_growth.tri3],
            )
            reversible = penalty * (1 - residual)
            fh.reverse_growth_restriction(uids, reversible)

        # Timing shift reversal
        timing_residual = np.select(
            [trimester == 1, trimester == 2, trimester == 3],
            [self.pars.tx_residual_timing.tri1, self.pars.tx_residual_timing.tri2, self.pars.tx_residual_timing.tri3],
        )
        fh.reverse_timing_shift(uids, 1 - timing_residual)

        return

    def step(self):
        sim = self.sim
        ti = self.ti
        ppl = sim.people

        fh = self._get_fh()
        if fh is None:
            return

        preg = ppl.pregnancy
        if not preg.pregnant.any():
            return

        pregnant_uids = preg.pregnant.uids

        for dname in self.disease_names:
            try:
                disease = sim.diseases[dname]
            except (KeyError, AttributeError):
                continue
            newly_infected = disease.ti_infected == ti
            affected = pregnant_uids[newly_infected[pregnant_uids]]
            if len(affected):
                self._apply_infection(affected, dname)

        return

    def update_results(self):
        """ Detect treatments and reverse fetal damage. """
        super().update_results()
        sim = self.sim
        ti = self.ti

        fh = self._get_fh()
        if fh is None:
            return

        preg = sim.people.pregnancy
        if not preg.pregnant.any():
            return

        pregnant_uids = preg.pregnant.uids

        for tx_name in self.treatment_names:
            try:
                tx = sim.interventions[tx_name]
            except (KeyError, AttributeError):
                continue
            if not hasattr(tx, 'ti_treated'):
                continue
            just_treated = tx.ti_treated == ti
            treated_pregnant = pregnant_uids[just_treated[pregnant_uids]]
            if len(treated_pregnant):
                dname = self.treatment_disease_map.get(tx_name)
                if dname:
                    self._apply_treatment(treated_pregnant, dname)

        return
