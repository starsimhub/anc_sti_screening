"""
Centralized prior distributions for the VoI analysis.

All prior distributions used across the analysis pipeline are defined here.
This includes calibration parameter bounds, mechanistic birth outcome parameters
(delivery timing shifts, growth penalties, treatment reversibility), and cost
parameters.

Sources are documented inline; see also promise-voi-plan-2.md Section 3.
"""

from scipy import stats
import sciris as sc


# %% Calibration parameters (uniform priors — bounds for Optuna)
# Format: {column_name: (label, low, high, log_scale)}
calib_pars = sc.objdict({
    # HIV
    'hiv.beta_m2f':             ('HIV β (M→F)',        0.002, 0.014, False),
    'hiv.eff_condom':           ('HIV condom eff.',     0.5,   0.9,   False),
    'hiv.rel_init_prev':        ('HIV rel. init prev',  2,    15,    False),
    # Network
    'structuredsexual.prop_f0': ('Prop F low-risk',    0.55,  0.9,   False),
    'structuredsexual.prop_m0': ('Prop M low-risk',    0.55,  0.85,  False),
    'structuredsexual.m1_conc': ('M1 concurrency',     0.05,  0.3,   False),
    # NG
    'ng.beta_m2f':              ('NG β (M→F)',         0.02,  0.25,  True),
    'ng.p_symp':                ('NG p(symp)',         0.05,  0.30,  False),
    # CT
    'ct.beta_m2f':              ('CT β (M→F)',         0.02,  0.25,  True),
    'ct.p_symp':                ('CT p(symp)',         0.10,  0.35,  False),
    # TV
    'tv.beta_m2f':              ('TV β (M→F)',         0.02,  0.25,  True),
    'tv.p_symp':                ('TV p(symp)',         0.15,  0.75,  False),
})


# %% Delivery timing shift parameters (Section 3b)
# Mean weeks delivery is brought forward by each STI infection.
# These are the mechanistic equivalent of the RR_PTB parameters from meta-analyses.
ptb_shift_pars = sc.objdict(
    ptb_shift_ng  = ('PTB shift NG (wk)',  stats.lognorm(s=0.4, scale=2.0)),  # Vallely 2021: RR 1.40 (1.14-1.73)
    ptb_shift_ct  = ('PTB shift CT (wk)',  stats.lognorm(s=0.4, scale=1.5)),  # He 2020: OR 1.35 (1.11-1.63)
    ptb_shift_tv  = ('PTB shift TV (wk)',  stats.lognorm(s=0.4, scale=1.0)),  # Silver 2014: RR 1.42 (1.15-1.75)
    ptb_shift_std = ('PTB shift SD (wk)',  stats.lognorm(s=0.3, scale=1.0)),  # Individual heterogeneity
)


# %% Growth restriction parameters (Section 3c)
# Fractional reduction in birth weight per infection.
growth_penalty_pars = sc.objdict(
    growth_penalty_ng = ('Growth penalty NG', stats.beta(4, 46)),   # mean ~0.08; Vallely 2021 RR 2.23 (1.34-3.71)
    growth_penalty_ct = ('Growth penalty CT', stats.beta(2, 65)),   # mean ~0.03; He 2020 OR 1.49 (0.90-2.47)
    growth_penalty_tv = ('Growth penalty TV', stats.beta(2, 65)),   # mean ~0.03; Silver 2014 RR 1.51 (1.32-1.73)
)


# %% Treatment reversibility parameters (Section 3d)
# Fraction of damage that PERSISTS after treatment (0 = full recovery, 1 = no effect).
# Split by trimester: T1 (<13w), T2 (13-26w), T3 (≥26w).
tx_residual_pars = sc.objdict(
    tx_residual_growth_tri1 = ('Tx residual growth (T1)', stats.beta(2, 6)),  # mean ~0.25; early treatment reverses most
    tx_residual_growth_tri2 = ('Tx residual growth (T2)', stats.beta(2, 3)),  # mean ~0.40; moderate reversibility
    tx_residual_growth_tri3 = ('Tx residual growth (T3)', stats.beta(3, 2)),  # mean ~0.60; damage largely locked in
    tx_residual_timing_tri1 = ('Tx residual timing (T1)', stats.beta(2, 4)),  # mean ~0.35; timing most recoverable early
    tx_residual_timing_tri2 = ('Tx residual timing (T2)', stats.beta(3, 3)),  # mean ~0.55; moderate recovery
    tx_residual_timing_tri3 = ('Tx residual timing (T3)', stats.beta(5, 2)),  # mean ~0.75; late treatment minimal timing recovery
)


# %% All birth outcome priors combined (for plotting and sampling)
birth_outcome_pars = sc.objdict()
birth_outcome_pars.update(ptb_shift_pars)
birth_outcome_pars.update(growth_penalty_pars)
birth_outcome_pars.update(tx_residual_pars)


# %% DALY weights (fixed, not sampled)

daly_weights = sc.objdict(
    dw_ptb  = ('DW preterm birth', 0.15),   # GBD 2019
    dw_lbw  = ('DW low birth weight', 0.10),  # GBD 2019
    dur_ptb = ('Duration PTB (years)', 1.0),
    dur_lbw = ('Duration LBW (years)', 1.0),
)


# %% Cost priors (all PLACEHOLDERS — need Zimbabwe-specific data)

cost_pars = sc.objdict(
    cost_poc_test      = ('POC test ($)',      stats.gamma(a=4,     scale=2)),      # μ=8, σ=4
    cost_tx_ct         = ('Tx CT ($)',         stats.gamma(a=2.25,  scale=4/3)),    # μ=3, σ=2
    cost_tx_ng         = ('Tx NG ($)',         stats.gamma(a=25/9,  scale=9/5)),    # μ=5, σ=3
    cost_tx_tv         = ('Tx TV ($)',         stats.gamma(a=4,     scale=0.5)),    # μ=2, σ=1
    cost_partner_notif = ('Partner notif ($)', stats.gamma(a=25/16, scale=16/5)),   # μ=5, σ=4
    cost_anc_visit     = ('ANC visit ($)',     stats.gamma(a=2.25,  scale=4/3)),    # μ=3, σ=2
    cost_ptb_mgmt      = ('PTB mgmt ($)',      stats.gamma(a=2.25,  scale=400/3)),  # μ=300, σ=200
    cost_lbw_mgmt      = ('LBW mgmt ($)',      stats.gamma(a=16/9,  scale=112.5)),  # μ=200, σ=150
)


# %% Sampling helper

def sample_priors(n=1, seed=None):
    """
    Draw n samples from all prior-only parameters.

    Returns:
        sc.objdict with parameter names as keys, arrays of length n as values.
    """
    import numpy as np
    if seed is not None:
        np.random.seed(seed)

    draws = sc.objdict()
    for group in [birth_outcome_pars, cost_pars]:
        for name, (label, dist) in group.items():
            draws[name] = dist.rvs(size=n)
    return draws
