"""
Fetal Health Module

Tracks fetal growth and adverse birth outcomes dynamically during pregnancy.
Works alongside the Pregnancy module to model how maternal STI infections
affect delivery timing (preterm birth) and fetal size (LBW/SGA).

Design:
    - Each pregnancy gets a baseline fetal weight percentile (individual heterogeneity)
    - Infections apply damage via two separate levers:
        1. Delivery timing: bring ti_delivery forward (PTB risk) — one-way ratchet
        2. Growth restriction: accumulate growth penalty — partially reversible by treatment
    - Treatment partially reverses growth restriction but leaves a residual
    - Reinfection compounds the damage
    - At delivery, birth weight is computed from gestational age + percentile + restriction
    - Outcomes classified: preterm (<37w), LBW (<2500g), SGA (<10th percentile for GA)

Usage:
    Module is added to the sim like any other module. It listens for infections
    and treatments via apply_infection_effects() and apply_treatment_effects(),
    which should be called from connectors or disease/intervention step methods.

    fh = FetalHealth()
    sim = sti.Sim(..., analyzers=[fh])  # Or add as a module
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

# Relative risks of preterm birth by disease
# These determine how much ti_delivery is shifted forward
# Source: DALY attribution parameters doc
RR_PTB = sc.objdict(
    ng=1.5,   # Vallely 2021, Taylor 2023
    ct=1.3,   # He 2020, Olson-Chen 2018
    tv=1.3,   # Silver 2014
    bv=1.0,   # Not modeled for PTB currently
)

# Growth restriction severity by disease (fractional reduction in weight)
# Applied as a multiplier: effective_weight *= (1 - growth_penalty)
GROWTH_PENALTY = sc.objdict(
    ng=0.08,  # NG → ~8% weight reduction (maps to 22.3% LBW rate)
    ct=0.03,  # CT → ~3% weight reduction (maps to 2.7% LBW rate)
    tv=0.03,  # TV → ~3% weight reduction (maps to 2.7% SGA rate)
    bv=0.0,
)

# What fraction of growth penalty persists after successful treatment
# 0.0 = full recovery, 1.0 = treatment has no effect on growth
TREATMENT_RESIDUAL = 0.3

# PTB: mean weeks brought forward on infection
PTB_SHIFT_MEAN = sc.objdict(
    ng=2.0,   # NG: mean 2 weeks earlier
    ct=1.5,   # CT: mean 1.5 weeks earlier
    tv=1.0,   # TV: mean 1 week earlier
    bv=0.0,
)
PTB_SHIFT_STD = 1.0  # Standard deviation of shift (weeks)


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
            rr_ptb=RR_PTB,
            growth_penalty=GROWTH_PENALTY,
            treatment_residual=TREATMENT_RESIDUAL,
            ptb_shift_mean=PTB_SHIFT_MEAN,
            ptb_shift_std=PTB_SHIFT_STD,
            sga_ratio=SGA_RATIO,
            lbw_threshold=2500,  # grams
            preterm_threshold=37, # weeks
        )
        self.update_pars(pars, **kwargs)

        self.define_states(
            # Maternal states (set during pregnancy)
            ss.FloatArr('weight_percentile', label='Fetal weight percentile'),      # Baseline heterogeneity (drawn at conception)
            ss.FloatArr('growth_restriction', label='Cumulative growth restriction'), # Accumulated damage from infections
            ss.FloatArr('n_infections_in_preg', label='Infections during pregnancy'), # Count of infection events this pregnancy
            ss.FloatArr('ti_ptb_shift', label='Last PTB shift time'),                # Track when last shift was applied

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

    def on_conception(self, uids):
        """
        Called when women become pregnant. Sets baseline fetal weight percentile
        and resets pregnancy-specific states.

        Should be called from the Pregnancy module or via a hook.
        """
        # Draw baseline weight percentile — log-normal around 1.0
        # This captures individual heterogeneity in fetal growth
        self.weight_percentile[uids] = np.random.lognormal(mean=0, sigma=0.1, size=len(uids))
        self.growth_restriction[uids] = 0.0
        self.n_infections_in_preg[uids] = 0
        self.ti_ptb_shift[uids] = np.nan

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

        Called when a pregnant woman becomes infected (or is already infected
        at conception). Modifies both delivery timing and growth.

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

        # --- Lever 1: Delivery timing (PTB) — one-way ratchet ---
        shift_mean = self.pars.ptb_shift_mean.get(disease_name, 0)
        if shift_mean > 0:
            # Convert weeks to timesteps
            dt_years = float(sim.pars.dt)
            ts_per_week = 1.0 / (dt_years * 365.25 / 7)  # timesteps per week
            shift_mean_ts = shift_mean * ts_per_week
            shift_std_ts  = float(self.pars.ptb_shift_std) * ts_per_week
            self._ptb_shift_dist.set(mean=shift_mean_ts, std=shift_std_ts)
            shifts = self._ptb_shift_dist.rvs(pregnant_uids)

            # Bring delivery forward but don't go below 24 weeks gestation
            min_dur_ts = 24 * ts_per_week  # 24 weeks in timesteps
            new_delivery = np.array(preg.ti_delivery[pregnant_uids], dtype=float) - shifts
            min_delivery_ti = np.array(preg.ti_pregnant[pregnant_uids], dtype=float) + min_dur_ts
            new_delivery = np.maximum(new_delivery, min_delivery_ti)

            # One-way ratchet: only bring forward, never push back
            current_delivery = np.array(preg.ti_delivery[pregnant_uids], dtype=float)
            preg.ti_delivery[pregnant_uids] = np.minimum(current_delivery, new_delivery)
            self.ti_ptb_shift[pregnant_uids] = self.ti

        # --- Lever 2: Growth restriction — cumulative ---
        penalty = self.pars.growth_penalty.get(disease_name, 0)
        if penalty > 0:
            # Compound: each infection adds penalty on remaining capacity
            # So first infection: 0 → 0.08, second: 0.08 → 0.08 + 0.92*0.08 = 0.154
            current = self.growth_restriction[pregnant_uids]
            self.growth_restriction[pregnant_uids] = current + (1 - current) * penalty

        self.n_infections_in_preg[pregnant_uids] += 1

        return

    def apply_treatment_effects(self, mother_uids, disease_name):
        """
        Partially reverse growth restriction when an infection is treated.

        Treatment reduces the growth penalty but leaves a residual, reflecting
        that even treated infections may have lasting effects on fetal growth.
        Does NOT reverse PTB shift (one-way ratchet).

        Args:
            mother_uids (ss.uids): UIDs of treated pregnant women
            disease_name (str):    'ng', 'ct', or 'tv'
        """
        preg = self.sim.people.pregnancy
        pregnant_uids = mother_uids[preg.pregnant[mother_uids]]
        if len(pregnant_uids) == 0:
            return

        penalty = self.pars.growth_penalty.get(disease_name, 0)
        residual = self.pars.treatment_residual

        # Remove the reversible portion of the most recent penalty
        # Residual fraction stays (e.g., 30% of the penalty persists)
        reversible = penalty * (1 - residual)
        current = self.growth_restriction[pregnant_uids]
        self.growth_restriction[pregnant_uids] = np.maximum(0, current - reversible)

        return

    def compute_birth_weight(self, mother_uids):
        """
        Compute birth weight for deliveries occurring this timestep.

        Birth weight = baseline_weight_for_ga × weight_percentile × (1 - growth_restriction)

        Args:
            mother_uids (ss.uids): UIDs of women delivering this timestep

        Returns:
            birth_weights (np.ndarray): birth weight in grams for each delivery
        """
        preg = self.sim.people.pregnancy
        ga_ts = np.array(preg.dur_pregnancy[mother_uids], dtype=float)

        # Convert dur_pregnancy (in timestep units) to weeks
        dt_years = float(self.sim.pars.dt)
        ga_weeks_val = ga_ts * dt_years * 365.25 / 7

        # Look up baseline weight for gestational age (interpolate)
        baseline_weight = np.interp(ga_weeks_val, GA_WEEKS, WEIGHT_GRAMS)

        # Apply individual heterogeneity and growth restriction
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

        # 2. Handle deliveries — women delivering this timestep
        delivering = preg.pregnant & (preg.ti_delivery <= ti)
        if not delivering.any():
            return

        deliver_uids = delivering.uids
        birth_weights, ga_weeks = self.compute_birth_weight(deliver_uids)

        # Store birth weight on mother (will be transferred to newborn in _post_delivery)
        self.birth_weight[deliver_uids] = birth_weights

        # Classify outcomes
        preterm_thresh = self.pars.preterm_threshold
        lbw_thresh     = self.pars.lbw_threshold

        is_preterm = ga_weeks < preterm_thresh
        is_lbw     = birth_weights < lbw_thresh

        # SGA: weight below 10th percentile for gestational age
        # Use baseline weight × SGA ratio as threshold
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
