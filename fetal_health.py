"""
Fetal Health Module

Tracks fetal growth and adverse birth outcomes dynamically during pregnancy.
Works alongside the Pregnancy module to model how maternal STI infections
affect delivery timing (preterm birth) and fetal size (LBW/SGA).

Design:
    - Each pregnancy gets a baseline fetal weight percentile (individual heterogeneity)
    - Infections apply damage via two separate levers:
        1. Delivery timing: bring ti_delivery forward (PTB risk)
        2. Growth restriction: accumulate growth penalty — partially reversible by treatment
    - Treatment partially reverses both growth restriction and delivery timing shift,
      with the degree of reversal depending on gestational age at treatment (early vs late)
    - Reinfection compounds the damage
    - At delivery, birth weight is computed from gestational age + percentile + restriction
    - Outcomes classified: preterm (<37w), LBW (<2500g), SGA (<10th percentile for GA)

Parameters:
    All mechanistic birth outcome parameters (delivery timing shifts, growth penalties,
    treatment reversibility) are accepted via pars and can be varied across VoI draws.
    See priors.py and promise-voi-plan-2.md Section 3b-3d for definitions and priors.

Usage:
    fh = FetalHealth(ptb_shift_mean=dict(ng=2.0, ct=1.5, tv=1.0), ...)
    sim = sti.Sim(..., analyzers=[fh])
"""

import numpy as np
import sciris as sc
import starsim as ss


# %% Reference data

# Fetal weight by gestational age (weeks) — approximate 50th percentile (grams)
# Source: Hadlock 1991 / INTERGROWTH-21st
WEIGHT_BY_GA = {
    24: 600,  25: 700,  26: 800,  27: 900,  28: 1000,
    29: 1150, 30: 1300, 31: 1500, 32: 1700, 33: 1900,
    34: 2100, 35: 2400, 36: 2600, 37: 2850, 38: 3050,
    39: 3250, 40: 3400, 41: 3500, 42: 3550,
}
GA_WEEKS = np.array(list(WEIGHT_BY_GA.keys()), dtype=float)
WEIGHT_GRAMS = np.array(list(WEIGHT_BY_GA.values()), dtype=float)

# 10th percentile ratio (for SGA classification) — approximately 0.80 of median
SGA_RATIO = 0.80

# GA cutoff (weeks) for early vs late treatment reversibility
# Corresponds to PROMISE enrollment screen (≤24w) vs third-trimester screen (32-34w)
EARLY_LATE_GA_CUTOFF = 24.0


# %% Module
class FetalHealth(ss.Module):
    """
    Track fetal health outcomes during pregnancy.

    States are defined on all agents but only meaningful for pregnant women
    and their newborns.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(name='fetal_health')
        self.define_pars(
            # Delivery timing shift: mean weeks brought forward per infection
            ptb_shift_mean=sc.objdict(ng=2.0, ct=1.5, tv=1.0, bv=0.0),
            ptb_shift_std=1.0,

            # Growth restriction: fractional weight reduction per infection
            growth_penalty=sc.objdict(ng=0.08, ct=0.03, tv=0.03, bv=0.0),

            # Treatment reversibility — fraction of damage PERSISTING after treatment
            # Split by early (≤24w GA) vs late (>24w GA)
            tx_residual_growth_early=0.33,   # Growth penalty residual, early treatment
            tx_residual_growth_late=0.50,    # Growth penalty residual, late treatment
            tx_residual_timing_early=0.50,   # Delivery shift residual, early treatment
            tx_residual_timing_late=0.71,    # Delivery shift residual, late treatment

            # GA cutoff for early vs late (weeks)
            early_late_cutoff=EARLY_LATE_GA_CUTOFF,

            # Classification thresholds
            sga_ratio=SGA_RATIO,
            lbw_threshold=2500,   # grams
            preterm_threshold=37, # weeks
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            # Maternal states (set during pregnancy)
            ss.FloatArr('weight_percentile', label='Fetal weight percentile'),
            ss.FloatArr('growth_restriction', label='Cumulative growth restriction'),
            ss.FloatArr('n_infections_in_preg', label='Infections during pregnancy'),
            ss.FloatArr('timing_shift_applied', label='Total delivery shift applied (timesteps)'),

            # Newborn states (set at delivery)
            ss.FloatArr('birth_weight', label='Birth weight (grams)'),
            ss.BoolState('is_preterm', label='Preterm birth'),
            ss.BoolState('is_lbw', label='Low birth weight'),
            ss.BoolState('is_sga', label='Small for gestational age'),
        )

        # Internal distribution for PTB shift — initialized in init_pre
        self._ptb_shift_dist = ss.lognorm_ex(mean=1.0, std=1.0)

        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_deliveries',   dtype=int, label='Deliveries'),
            ss.Result('n_preterm',      dtype=int, label='Preterm births'),
            ss.Result('n_lbw',          dtype=int, label='Low birth weight'),
            ss.Result('n_sga',          dtype=int, label='Small for gestational age'),
            ss.Result('mean_birth_weight', scale=False, label='Mean birth weight (g)'),
            ss.Result('mean_ga_at_birth',  scale=False, label='Mean gestational age (weeks)'),
            ss.Result('preterm_rate',      scale=False, label='Preterm birth rate'),
            ss.Result('lbw_rate',          scale=False, label='LBW rate'),
            ss.Result('sga_rate',          scale=False, label='SGA rate'),
        )
        return

    def _get_ga_weeks(self, uids):
        """Get current gestational age in weeks for given UIDs."""
        preg = self.sim.people.pregnancy
        dt_years = float(self.sim.pars.dt)
        ga_ts = self.ti - np.array(preg.ti_pregnant[uids], dtype=float)
        return ga_ts * dt_years * 365.25 / 7

    def _ts_per_week(self):
        """Conversion factor: timesteps per week."""
        dt_years = float(self.sim.pars.dt)
        return 1.0 / (dt_years * 365.25 / 7)

    def on_conception(self, uids):
        """
        Called when women become pregnant. Sets baseline fetal weight percentile
        and resets pregnancy-specific states.
        """
        self.weight_percentile[uids] = np.random.lognormal(mean=0, sigma=0.1, size=len(uids))
        self.growth_restriction[uids] = 0.0
        self.n_infections_in_preg[uids] = 0
        self.timing_shift_applied[uids] = 0.0

        # Check for pre-existing infections and apply effects
        for dname in ['ng', 'ct', 'tv']:
            try:
                disease = self.sim.diseases[dname]
                infected_uids = uids[disease.infected[uids]]
                if len(infected_uids):
                    self.apply_infection_effects(infected_uids, dname)
            except (KeyError, AttributeError):
                pass

        return

    def apply_infection_effects(self, mother_uids, disease_name):
        """
        Apply adverse effects of a new STI infection on fetal health.

        Args:
            mother_uids (ss.uids): UIDs of infected pregnant women
            disease_name (str):    'ng', 'ct', or 'tv'
        """
        sim = self.sim
        preg = sim.people.pregnancy

        # Filter to only pregnant women
        pregnant_uids = mother_uids[preg.pregnant[mother_uids]]
        if len(pregnant_uids) == 0:
            return

        # --- Lever 1: Delivery timing shift ---
        shift_mean = self.pars.ptb_shift_mean.get(disease_name, 0)
        if shift_mean > 0:
            ts_per_wk = self._ts_per_week()
            shift_mean_ts = shift_mean * ts_per_wk
            shift_std_ts  = float(self.pars.ptb_shift_std) * ts_per_wk
            self._ptb_shift_dist.set(mean=shift_mean_ts, std=shift_std_ts)
            shifts = self._ptb_shift_dist.rvs(pregnant_uids)

            # Bring delivery forward but don't go below 24 weeks gestation
            min_dur_ts = 24 * ts_per_wk
            new_delivery = np.array(preg.ti_delivery[pregnant_uids], dtype=float) - shifts
            min_delivery_ti = np.array(preg.ti_pregnant[pregnant_uids], dtype=float) + min_dur_ts
            new_delivery = np.maximum(new_delivery, min_delivery_ti)

            # One-way ratchet: only bring forward, never push back
            current_delivery = np.array(preg.ti_delivery[pregnant_uids], dtype=float)
            actually_shifted = np.maximum(0, current_delivery - new_delivery)
            preg.ti_delivery[pregnant_uids] = np.minimum(current_delivery, new_delivery)
            self.timing_shift_applied[pregnant_uids] += actually_shifted

        # --- Lever 2: Growth restriction — cumulative ---
        penalty = self.pars.growth_penalty.get(disease_name, 0)
        if penalty > 0:
            current = self.growth_restriction[pregnant_uids]
            self.growth_restriction[pregnant_uids] = current + (1 - current) * penalty

        self.n_infections_in_preg[pregnant_uids] += 1

        return

    def apply_treatment_effects(self, mother_uids, disease_name, ga_weeks=None):
        """
        Partially reverse damage when an infection is treated during pregnancy.

        Both growth restriction and delivery timing shift can be partially reversed,
        with the degree depending on gestational age at treatment (early vs late).

        Args:
            mother_uids (ss.uids): UIDs of treated pregnant women
            disease_name (str):    'ng', 'ct', or 'tv'
            ga_weeks (np.ndarray): gestational age in weeks at treatment for each uid.
                                   If None, computed from current sim time.
        """
        preg = self.sim.people.pregnancy
        pregnant_uids = mother_uids[preg.pregnant[mother_uids]]
        if len(pregnant_uids) == 0:
            return

        # Compute GA if not provided
        if ga_weeks is None:
            ga_weeks = self._get_ga_weeks(pregnant_uids)

        cutoff = self.pars.early_late_cutoff
        is_early = ga_weeks <= cutoff

        # --- Growth restriction reversal ---
        penalty = self.pars.growth_penalty.get(disease_name, 0)
        if penalty > 0:
            # Select residual based on early/late
            residual = np.where(
                is_early,
                self.pars.tx_residual_growth_early,
                self.pars.tx_residual_growth_late,
            )
            reversible = penalty * (1 - residual)
            current = np.array(self.growth_restriction[pregnant_uids], dtype=float)
            self.growth_restriction[pregnant_uids] = np.maximum(0, current - reversible)

        # --- Delivery timing reversal (relaxing the ratchet) ---
        timing_residual = np.where(
            is_early,
            self.pars.tx_residual_timing_early,
            self.pars.tx_residual_timing_late,
        )
        # Recover a fraction of the accumulated timing shift
        current_shift = np.array(self.timing_shift_applied[pregnant_uids], dtype=float)
        recoverable = current_shift * (1 - timing_residual)

        # Only recover if there's shift to recover
        has_shift = recoverable > 0
        if np.any(has_shift):
            recover_uids = pregnant_uids[has_shift]
            recover_ts = recoverable[has_shift]

            # Push delivery back (later) by the recovered amount
            current_delivery = np.array(preg.ti_delivery[recover_uids], dtype=float)
            preg.ti_delivery[recover_uids] = current_delivery + recover_ts

            # Update tracked shift
            self.timing_shift_applied[recover_uids] -= recover_ts

        return

    def compute_birth_weight(self, mother_uids):
        """
        Compute birth weight for deliveries occurring this timestep.

        Birth weight = baseline_weight_for_ga × weight_percentile × (1 - growth_restriction)
        """
        preg = self.sim.people.pregnancy
        ga_ts = np.array(preg.ti_delivery[mother_uids] - preg.ti_pregnant[mother_uids], dtype=float)

        dt_years = float(self.sim.pars.dt)
        ga_weeks_val = ga_ts * dt_years * 365.25 / 7

        baseline_weight = np.interp(ga_weeks_val, GA_WEEKS, WEIGHT_GRAMS)

        percentile = np.array(self.weight_percentile[mother_uids], dtype=float)
        restriction = np.array(self.growth_restriction[mother_uids], dtype=float)

        birth_weight = baseline_weight * percentile * (1 - restriction)

        return birth_weight, ga_weeks_val

    def step(self):
        """
        Main step function. Two responsibilities:
            1. Initialize fetal health for new pregnancies
            2. Classify outcomes for deliveries happening this timestep
        """
        sim = self.sim
        ti = self.ti
        preg = sim.people.pregnancy

        # 1. Handle new conceptions — women who just became pregnant
        just_conceived = preg.pregnant & (preg.ti_pregnant == ti)
        if just_conceived.any():
            self.on_conception(just_conceived.uids)

        # 2. Handle deliveries — women delivering this timestep.
        # Pregnancy.step() processes deliveries BEFORE analyzers run.
        # After delivery, pregnant is cleared and ti_delivery == ti.
        delivering = (preg.ti_delivery == ti) & ~preg.pregnant
        if not delivering.any():
            return

        deliver_uids = delivering.uids
        birth_weights, ga_weeks = self.compute_birth_weight(deliver_uids)

        self.birth_weight[deliver_uids] = birth_weights

        # Classify outcomes
        preterm_thresh = self.pars.preterm_threshold
        lbw_thresh     = self.pars.lbw_threshold

        is_preterm = ga_weeks < preterm_thresh
        is_lbw     = birth_weights < lbw_thresh

        sga_threshold = np.interp(ga_weeks, GA_WEEKS, WEIGHT_GRAMS) * self.pars.sga_ratio
        is_sga = birth_weights < sga_threshold

        self.is_preterm[deliver_uids] = is_preterm
        self.is_lbw[deliver_uids]     = is_lbw
        self.is_sga[deliver_uids]     = is_sga

        # Update results
        n = len(deliver_uids)
        self.results['n_deliveries'][ti]     = n
        self.results['n_preterm'][ti]         = np.sum(is_preterm)
        self.results['n_lbw'][ti]             = np.sum(is_lbw)
        self.results['n_sga'][ti]             = np.sum(is_sga)
        self.results['mean_birth_weight'][ti] = np.mean(birth_weights) if n > 0 else 0
        self.results['mean_ga_at_birth'][ti]  = np.mean(ga_weeks) if n > 0 else 0
        self.results['preterm_rate'][ti]      = np.mean(is_preterm) if n > 0 else 0
        self.results['lbw_rate'][ti]          = np.mean(is_lbw) if n > 0 else 0
        self.results['sga_rate'][ti]          = np.mean(is_sga) if n > 0 else 0

        return
