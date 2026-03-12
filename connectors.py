"""
Connectors between STI diseases and the FetalHealth module.

The sti_fetal connector monitors infections and treatments in pregnant women
and routes them to FetalHealth.apply_infection_effects() and
apply_treatment_effects().

Performance note: this runs once per timestep and only operates on
the set of women who were newly infected or treated that step, so the
cost is proportional to incident events, not population size.
"""

import numpy as np
import starsim as ss


class sti_fetal(ss.Connector):
    """
    Connect STI disease modules to the FetalHealth module.

    Monitors new infections and treatments in pregnant women each timestep
    and applies fetal health effects accordingly.

    Args:
        diseases (list): list of disease names to monitor (e.g. ['ng', 'ct', 'tv'])
        treatments (list): list of treatment intervention names (e.g. ['ng_tx', 'ct_tx', 'metronidazole'])
        treatment_disease_map (dict): maps treatment name → disease name(s) it treats
    """

    def __init__(self, diseases=None, treatments=None, treatment_disease_map=None, **kwargs):
        super().__init__(**kwargs)
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

        return

    def _get_ga_weeks(self, uids):
        """Compute gestational age in weeks for pregnant UIDs."""
        preg = self.sim.people.pregnancy
        dt_years = float(self.sim.pars.dt)
        ga_ts = self.sim.ti - np.array(preg.ti_pregnant[uids], dtype=float)
        return ga_ts * dt_years * 365.25 / 7

    def step(self):
        sim = self.sim
        ti = self.ti
        ppl = sim.people

        # Get fetal health module (registered as an analyzer)
        try:
            fh = sim.analyzers['fetal_health']
        except (KeyError, AttributeError):
            return  # FetalHealth not present, skip

        preg = ppl.pregnancy
        if not preg.pregnant.any():
            return

        pregnant_uids = preg.pregnant.uids

        # --- Monitor new infections in pregnant women ---
        for dname in self.disease_names:
            try:
                disease = sim.diseases[dname]
            except (KeyError, AttributeError):
                continue

            # Find pregnant women newly infected this timestep
            newly_infected = disease.ti_infected == ti
            affected = pregnant_uids[newly_infected[pregnant_uids]]

            if len(affected):
                fh.apply_infection_effects(affected, dname)

        # --- Monitor treatments in pregnant women ---
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
                    ga_weeks = self._get_ga_weeks(treated_pregnant)
                    fh.apply_treatment_effects(treated_pregnant, dname, ga_weeks=ga_weeks)

        return
