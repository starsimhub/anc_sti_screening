"""
Connectors between STI diseases and the FetalHealth module.

The sti_fetal connector holds all disease-specific parameters (delivery timing
shifts, growth penalties, treatment reversibility) and translates STI events
into calls to FetalHealth's generic API.

Performance note: this runs once per timestep and only operates on
the set of women who were newly infected or treated that step, so the
cost is proportional to incident events, not population size.
"""

import numpy as np
import sciris as sc
import starsim as ss


class sti_fetal(ss.Connector):
    """
    Connect STI disease modules to FetalHealth.

    Monitors new infections and treatments in pregnant women each timestep
    and modifies fetal health accordingly. Also registers a conception callback
    to handle pre-existing infections at the start of pregnancy.

    Args:
        diseases (list):             disease names to monitor (default: ng, ct, tv)
        treatments (list):           treatment intervention names
        treatment_disease_map (dict): maps treatment name -> disease name
        ptb_shift_mean (dict):       mean weeks delivery brought forward, per disease
        ptb_shift_std (float):       individual heterogeneity in timing shift
        growth_penalty (dict):       fractional weight reduction per infection, per disease
        tx_residual_growth_early/late (float): fraction of growth damage persisting after treatment
        tx_residual_timing_early/late (float): fraction of timing damage persisting after treatment
        early_late_cutoff (dur):     GA cutoff for early vs late treatment
    """

    def __init__(self, diseases=None, treatments=None, treatment_disease_map=None,
                 pars=None, **kwargs):
        super().__init__()
        self.name = 'sti_fetal'
        self.disease_names = diseases or ['ng', 'ct', 'tv']
        self.treatment_names = treatments or ['ng_tx', 'ct_tx', 'metronidazole']
        if treatment_disease_map is None:
            treatment_disease_map = {
                'ng_tx':         'ng',
                'ct_tx':         'ct',
                'metronidazole': 'tv',
            }
        self.treatment_disease_map = treatment_disease_map

        self.define_pars(
            # Delivery timing shift per infection (mean weeks brought forward)
            ptb_shift_mean=sc.objdict(ng=2.0, ct=1.5, tv=1.0),
            ptb_shift_std=1.0,

            # Growth restriction per infection (fractional weight reduction)
            growth_penalty=sc.objdict(ng=0.08, ct=0.03, tv=0.03),

            # Treatment reversibility -- fraction of damage PERSISTING after treatment
            tx_residual_growth_early=0.33,
            tx_residual_growth_late=0.50,
            tx_residual_timing_early=0.50,
            tx_residual_timing_late=0.71,

            # GA cutoff for early vs late treatment
            early_late_cutoff=ss.weeks(24),

            # Distribution for sampling individual timing shifts (mean/std set dynamically per disease)
            ptb_shift_dist=ss.lognorm_ex(mean=1.0, std=1.0),
        )
        self.update_pars(pars, **kwargs)

        return

    def init_pre(self, sim):
        super().init_pre(sim)
        # Register conception callback with FetalHealth
        try:
            fh = sim.custom['fetal_health']
            fh.add_conception_callback(self._on_conception)
        except (KeyError, AttributeError):
            pass
        return

    def _get_fh(self):
        """ Get the FetalHealth module """
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

        # Timing shift
        shift_mean = self.pars.ptb_shift_mean.get(disease_name, 0)
        if shift_mean > 0:
            self.pars.ptb_shift_dist.set(mean=shift_mean, std=float(self.pars.ptb_shift_std))
            shifts = self.pars.ptb_shift_dist.rvs(uids)
            fh.apply_timing_shift(uids, shifts)

        # Growth restriction
        penalty = self.pars.growth_penalty.get(disease_name, 0)
        if penalty > 0:
            fh.apply_growth_restriction(uids, penalty)

        return

    def _apply_treatment(self, uids, disease_name):
        """Reverse fetal health damage when an infection is treated during pregnancy."""
        fh = self._get_fh()
        if fh is None:
            return

        preg = self.sim.people.pregnancy
        pregnant_uids = uids[preg.pregnant[uids]]
        if not len(pregnant_uids):
            return

        # Use the Pregnancy module's gestation tracker (already in weeks)
        ga_weeks = np.asarray(preg.gestation[pregnant_uids], dtype=float)
        cutoff = self.pars.early_late_cutoff.weeks
        is_early = ga_weeks <= cutoff

        # Growth restriction reversal
        penalty = self.pars.growth_penalty.get(disease_name, 0)
        if penalty > 0:
            residual = np.where(
                is_early,
                self.pars.tx_residual_growth_early,
                self.pars.tx_residual_growth_late,
            )
            reversible = penalty * (1 - residual)
            fh.reverse_growth_restriction(pregnant_uids, reversible)

        # Timing shift reversal
        timing_residual = np.where(
            is_early,
            self.pars.tx_residual_timing_early,
            self.pars.tx_residual_timing_late,
        )
        fh.reverse_timing_shift(pregnant_uids, 1 - timing_residual)

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

        # --- New infections in pregnant women ---
        for dname in self.disease_names:
            try:
                disease = sim.diseases[dname]
            except (KeyError, AttributeError):
                continue

            newly_infected = disease.ti_infected == ti
            affected = pregnant_uids[newly_infected[pregnant_uids]]
            if len(affected):
                self._apply_infection(affected, dname)

        # --- Treatments in pregnant women ---
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
