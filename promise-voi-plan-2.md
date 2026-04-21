# PROMISE Trial: Value of Information Analysis Using STIsim

## Overview

This document outlines the modeling pipeline for a pre-trial value of information (VoI) analysis of the PROMISE trial, using the STIsim agent-based framework calibrated to Zimbabwe. The goal is to quantify decision uncertainty around the cost-effectiveness of integrated antenatal STI screening and identify which parameters drive that uncertainty — before trial readouts are available.

The analysis proceeds in seven steps:

1. Define prior distributions on key epidemiological parameters (already done — these are the `calib_pars`)
2. Calibrate the STIsim Zimbabwe model to prevalence data using Optuna
3. Combine calibrated (posterior) epi parameters with prior distributions on birth outcome, treatment-of-birth-outcome, and cost parameters
4. Build a DALY analyzer to track birth outcomes and compute DALYs
5. Build the cost-effectiveness mapping from model outputs to incremental net monetary benefit
6. Run the VoI analysis (EVPI, EVPPI, EVSI)
7. Generate figures and reporting

---

## Step 1: Epidemiological Parameters (calib_pars)

These are already defined in the repo. There are 15 calibration parameters governing HIV transmission, sexual network structure, and STI transmission/symptomatology:

### HIV transmission

| Parameter | Low | High | Guess | Notes |
|-----------|-----|------|-------|-------|
| `hiv.beta_m2f` | 0.002 | 0.014 | 0.006 | Per-act M→F transmission probability |
| `hiv.eff_condom` | 0.5 | 0.9 | 0.75 | Condom efficacy for HIV |
| `hiv.rel_init_prev` | 2 | 15 | 8 | Relative initial prevalence (seeding) |

### Network structure

| Parameter | Low | High | Guess | Notes |
|-----------|-----|------|-------|-------|
| `structuredsexual.prop_f0` | 0.55 | 0.9 | 0.7 | Proportion of women in lowest activity group |
| `structuredsexual.prop_m0` | 0.55 | 0.85 | 0.83 | Proportion of men in lowest activity group |
| `structuredsexual.m1_conc` | 0.05 | 0.3 | 0.15 | Concurrency among higher-activity men |

### STI transmission and symptomatology

| Parameter | Low | High | Guess | Log | Notes |
|-----------|-----|------|-------|-----|-------|
| `ng.beta_m2f` | 0.02 | 0.25 | 0.08 | Yes | Ng per-act M→F transmission |
| `ng.p_symp` | 0.05 | 0.30 | 0.15 | No | Proportion of Ng infections symptomatic |
| `ct.beta_m2f` | 0.02 | 0.25 | 0.06 | Yes | Ct per-act M→F transmission |
| `ct.p_symp` | 0.10 | 0.35 | 0.25 | No | Proportion of Ct infections symptomatic |
| `tv.beta_m2f` | 0.02 | 0.25 | 0.07 | Yes | Tv per-act M→F transmission |
| `tv.p_symp` | 0.15 | 0.75 | 0.45 | No | Proportion of Tv infections symptomatic |

### Notes

- The `calib_pars` structure maps directly to the Optuna search space in Step 2. No changes needed.
- Diagnostic sensitivity parameters are NOT included because the SOC is no ANC screening. The intervention arm's POC test performance is fixed at 95/95 (Step 3e).
- Treatment efficacy for clearing STIs is well-established and already handled within STIsim. What is NOT known is the extent to which treating an STI during pregnancy reverses the damage to birth outcomes — this is addressed in Step 3.
- **HIV parameters are calibrated but fixed for the VoI analysis.** The three HIV parameters remain in `calib_pars` because HIV affects network structure and background STI dynamics, but they are not part of the VoI decision space. After calibration, HIV parameters are fixed at their posterior point estimates and do not contribute to uncertainty propagation in Steps 5-7.

---

## Step 2: Calibrate to Prevalence Data Using Optuna

### Calibration targets

| Target | Estimate | 95% CI | Source |
|--------|----------|--------|--------|
| HIV prevalence (women 15-49) | 14.5% | (13.0%, 16.0%) | ZIMPHIA 2020 |
| Ng prevalence (ANC women) | 3.5% | (1.5%, 6.0%) | Mudau et al. 2018; Harare ANC data |
| Ct prevalence (ANC women) | 8.0% | (5.0%, 12.0%) | Mudau et al. 2018; Peters et al. 2021 |
| Tv prevalence (ANC women) | 18.0% | (12.0%, 25.0%) | Masha et al. 2019; Harare data |
| HIV prevalence (ANC women) | 16.0% | (14.0%, 19.0%) | Zimbabwe PMTCT data |

**TODO:** Confirm exact calibration targets with the PROMISE study team. Additional targets (e.g., prevalence by age group, co-infection rates) should be added if data are available.

### Calibration approach

Use Optuna to search the 15-dimensional parameter space defined by `calib_pars`, minimizing a weighted sum of squared deviations between model-predicted prevalences and the calibration targets. The Optuna search bounds are already specified in `calib_pars` (low/high for each parameter).

The objective function should:
- Accept an Optuna trial and suggest values for all 15 parameters using the bounds and log-scale flags from `calib_pars`
- Build and run the STIsim Zimbabwe model (adapting `stisim_vddx_zim` as template)
- Extract model-predicted prevalences at the calibration timepoint
- Compute weighted least-squares goodness-of-fit, with weights inversely proportional to the uncertainty in each target

Start with ~500 Optuna trials. The top ~100-200 parameter sets (ranked by goodness of fit) form the approximate posterior.

### Outputs from Step 2

- **Posterior parameter samples**: Top ~100-200 parameter sets consistent with observed prevalence data.
- **Calibration diagnostics**: Model vs. data comparison for each top parameter set.
- **Prior vs. posterior comparison**: Marginal distributions before and after calibration (feeds into Figure 4).

---

## Step 3: Define Priors on Birth Outcome, Treatment Reversibility, and Cost Parameters

These parameters are NOT informed by the STIsim calibration. The epi model tells us who is infected with what and when; the links from infection to adverse pregnancy outcomes, the reversibility of damage through treatment, and costs all come from external evidence or are genuinely unknown.

### 3a. How birth outcomes work in the model

The `FetalHealth` module (`starsim.demographics.FetalHealth`) is a generic Starsim module that tracks fetal health during pregnancy via two damage pathways. It registers conception and delivery callbacks with the Pregnancy module. All disease-specific logic lives in the `sti_fetal` connector, which calls FetalHealth's public API.

**Architecture:**
- `FetalHealth` (generic, in Starsim): tracks `timing_shift`, `growth_restriction`, and `weight_percentile` per agent. Provides `apply_timing_shift()`, `apply_growth_restriction()`, `reverse_timing_shift()`, `reverse_growth_restriction()`. At delivery, calls `compute_birth_weight()` to classify PTB/LBW/SGA.
- `sti_fetal` connector (disease-specific, in this repo): registers a conception callback to handle pre-existing infections. In `step()`, monitors new infections and treatments in pregnant women, calling FetalHealth's API. Holds all pathogen-specific parameters (shifts, penalties, reversibility).

**Two damage pathways:**

1. **Delivery timing (→ PTB)**: When a pregnant woman acquires an STI, the `sti_fetal` connector shifts her delivery date earlier by a stochastic amount drawn from a lognormal distribution. The mean shift depends on the pathogen (Ng shifts more than Ct or Tv). Reinfection compounds the shift. Treatment can partially reverse the timing shift, with reversibility depending on gestational age at treatment. A birth is classified as preterm if delivery occurs before 37 weeks' gestation.

2. **Fetal growth restriction (→ LBW/SGA)**: Each infection accumulates a growth penalty that reduces birth weight as a fraction of the gestational-age-appropriate weight. Treatment partially reverses this penalty, but leaves a residual that depends on gestational age at treatment (early treatment recovers more). Birth weight is computed at delivery as: `baseline_weight_for_GA × individual_percentile × (1 − growth_restriction)`. A birth is classified as LBW if birth weight < 2500g, and SGA if below the 10th percentile for gestational age.

This mechanistic approach is richer than a simple probability/RR model because outcomes emerge from the interaction of infection timing, treatment timing, baseline fetal growth trajectory, and gestational age. It also means the VoI analysis runs pairs of full simulations (SOC vs. intervention) for each parameter draw, rather than replaying event logs post-hoc.

**Two infection timing scenarios handled:**
- **Pre-existing infection at conception**: The `sti_fetal` connector registers a conception callback with FetalHealth. When a woman becomes pregnant, the callback checks each disease for existing infections and applies damage immediately.
- **New infection during pregnancy**: The connector's `step()` method checks `disease.ti_infected == ti` each timestep and applies damage to newly infected pregnant women. Similarly, it checks `tx.ti_treated == ti` to detect treatments and apply partial reversal based on gestational age (early ≤24w uses `tx_residual_*_early`, late uses `tx_residual_*_late`).

### 3b. Delivery timing shift parameters (→ preterm birth)

When a pregnant woman is infected with an STI, her delivery date is shifted forward by a stochastic amount. The mean shift (in weeks) is pathogen-specific and uncertain. These are the mechanistic equivalent of the RR_PTB parameters from the literature — they naturally produce PTB rates that depend on the woman's baseline gestational trajectory.

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `ptb_shift_ng` | Mean delivery shift from Ng infection (weeks) | LogNormal(μ=log(2.0), σ=0.4) | Current default 2.0w; Vallely 2021 meta-analysis RR 1.40 (CI 1.14-1.73) |
| `ptb_shift_ct` | Mean delivery shift from Ct infection (weeks) | LogNormal(μ=log(1.5), σ=0.4) | Current default 1.5w; He 2020 meta-analysis OR 1.35 (CI 1.11-1.63) |
| `ptb_shift_tv` | Mean delivery shift from Tv infection (weeks) | LogNormal(μ=log(1.0), σ=0.4) | Current default 1.0w; Silver 2014 RR 1.42 (CI 1.15-1.75) |
| `ptb_shift_std` | SD of individual-level shift (weeks) | LogNormal(μ=log(1.0), σ=0.3) | Controls heterogeneity in PTB shift across women |

### 3c. Growth restriction parameters (→ LBW/SGA)

Each STI infection accumulates a growth penalty that reduces fetal weight as a fraction of gestational-age-appropriate weight.

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `growth_penalty_ng` | Fractional weight reduction from Ng | Beta(α=4, β=46) → mean ~0.08 | Current default 0.08; maps to ~22% LBW rate; Vallely 2021 RR 2.23 (CI 1.34-3.71) |
| `growth_penalty_ct` | Fractional weight reduction from Ct | Beta(α=2, β=65) → mean ~0.03 | Current default 0.03; He 2020 OR 1.49 (CI 0.90-2.47, NS) |
| `growth_penalty_tv` | Fractional weight reduction from Tv | Beta(α=2, β=65) → mean ~0.03 | Current default 0.03; Silver 2014 RR 1.51 (CI 1.32-1.73) |

### 3d. Treatment reversibility parameters

When an STI is treated during pregnancy, two questions arise: does treatment reverse the delivery timing shift, and how much of the growth restriction is recovered? Reversibility is timing-dependent, reflecting the two PROMISE screening timepoints.

**Growth restriction reversibility**: The `treatment_residual` parameter captures the fraction of growth penalty that persists after successful treatment. 0.0 = full recovery; 1.0 = treatment has no effect on growth.

**Delivery timing reversibility**: The current FetalHealth code assumes zero reversibility (one-way ratchet). We introduce a parameter to relax this, allowing early treatment to partially recover delivery timing.

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `tx_residual_growth_early` | Fraction of growth penalty persisting after treatment ≤24w | Beta(α=2, β=4) → mean ~0.33 | Current default TREATMENT_RESIDUAL=0.3; lower residual for early treatment |
| `tx_residual_growth_late` | Fraction of growth penalty persisting after treatment 32-34w | Beta(α=3, β=3) → mean ~0.50 | Less recovery time in third trimester |
| `tx_residual_timing_early` | Fraction of delivery shift persisting after treatment ≤24w | Beta(α=3, β=3) → mean ~0.50 | Relaxes the one-way ratchet; uncertain |
| `tx_residual_timing_late` | Fraction of delivery shift persisting after treatment 32-34w | Beta(α=5, β=2) → mean ~0.71 | Late treatment unlikely to recover timing |

**Note on the ratchet**: The current code treats delivery timing as a pure one-way ratchet (`tx_residual_timing = 1.0`). Our priors allow partial reversibility, especially with early treatment. If the EVPPI for timing reversibility parameters is high, that tells us this modeling choice matters and warrants clinical input.

**Discussion**: The early/late split directly addresses the PROMISE design question: how much value comes from the enrollment screen vs. the third-trimester screen? If late reversibility is low, the first screen is doing most of the work — with implications for simplified (single-screen) implementation.

### 3e. Intervention parameters (screening-specific)

The intervention arm introduces POC testing at two ANC timepoints. SOC has no ANC STI screening.

POC test sensitivity and specificity are fixed at 95%/95% for all three STIs. This assumes a high-quality test platform and keeps the parameter space manageable.

| Parameter | Description | Value | Notes |
|-----------|-------------|-------|-------|
| `sens_poc` | POC test sensitivity (all STIs) | 0.95 (fixed) | Assumed high-quality test |
| `spec_poc` | POC test specificity (all STIs) | 0.95 (fixed) | Assumed high-quality test |

The remaining uncertain intervention parameter is partner notification success:

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `p_partner_tx` | Partner notification + treatment success | Beta(α=3, β=7) → mean ~0.30 | Ferreira et al. 2013; highly uncertain |

Note: Reinfection between the two screening timepoints emerges from STIsim dynamics — it is not an exogenous parameter.

### 3f. Cost parameters

**Note: We do not currently have cost data for this setting. All cost priors below are placeholders. These need to be refined with input from the PROMISE study team and Zimbabwe-specific health system cost data.**

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `cost_poc_test` | Cost per POC test (3 STIs) | Gamma(μ=$8, σ=$4) | **PLACEHOLDER** |
| `cost_tx_ct` | Treatment cost, Ct (azithromycin) | Gamma(μ=$3, σ=$2) | **PLACEHOLDER** |
| `cost_tx_ng` | Treatment cost, Ng (dual therapy) | Gamma(μ=$5, σ=$3) | **PLACEHOLDER** |
| `cost_tx_tv` | Treatment cost, Tv (metronidazole) | Gamma(μ=$2, σ=$1) | **PLACEHOLDER** |
| `cost_partner_notif` | Cost per partner notification episode | Gamma(μ=$5, σ=$4) | **PLACEHOLDER** |
| `cost_anc_visit` | Marginal cost of integrating STI testing into ANC visit | Gamma(μ=$3, σ=$2) | **PLACEHOLDER** |
| `cost_ptb_mgmt` | Cost of managing preterm birth | Gamma(μ=$300, σ=$200) | **PLACEHOLDER** |
| `cost_lbw_mgmt` | Cost of managing LBW | Gamma(μ=$200, σ=$150) | **PLACEHOLDER** |
| `cost_neonatal_death` | Cost of neonatal death management | Gamma(μ=$250, σ=$180) | **PLACEHOLDER** |

### 3g. DALY weights

| Outcome | DALY weight | Duration | Source |
|---------|-------------|----------|--------|
| Preterm birth | 0.15 | Acute + sequelae | GBD 2019 |
| LBW | 0.10 | Acute + sequelae | GBD 2019 |

**Note on PTB vs LBW double-counting**: In the mechanistic model, nearly all LBW births are also preterm (preterm infants are mechanically <2500g via the fetal weight curve). PTB+LBW co-occurrences accrue PTB disability weight only. In practice, DALYs ≈ n_ptb × dw_ptb × dur_ptb.

### 3h. Co-infection interaction terms (structural uncertainty)

We will run two model structures:
- **Multiplicative model**: Effects for each STI compound independently. A woman infected with Ct + Tv gets both delivery shifts and both growth penalties applied.
- **Interaction model**: Include a co-infection interaction parameter that modifies the compounding, allowing synergistic or antagonistic effects.

Structural uncertainty handled via model averaging in the VoI analysis.

### VoI parameter count: 32 (8×4 grid)

- 9 calibrated epi parameters (3 network + 2 Ng + 2 Ct + 2 Tv; HIV fixed at posterior)
- 4 delivery timing shifts (ng, ct, tv, std)
- 3 growth penalties (ng, ct, tv)
- 4 treatment residuals (growth_early, growth_late, timing_early, timing_late)
- 2 baseline rates (p_ptb_base, p_lbw_base — used for validation, not directly in the mechanistic model)
- 1 partner treatment (p_partner_tx)
- 9 cost parameters

---

## Step 4: Build a DALY Analyzer

The DALY analyzer is a Starsim `Analyzer` that runs alongside the simulation and translates the birth outcomes produced by the mechanistic `FetalHealth` module into DALYs.

### What the analyzer needs to do

1. **Receive birth outcome classifications from FetalHealth**: At each delivery, FetalHealth classifies whether the birth is preterm (<37 weeks), LBW (<2500g), and/or SGA (<10th percentile). The DALY analyzer reads these flags.

2. **Compute DALYs per birth**: Apply DALY weights to each adverse outcome. For PTB+LBW co-occurrences, use PTB disability weight only to avoid double-counting (see 3g note).

3. **Track costs**: Since the analyzer sees each delivery's outcomes and each woman's treatment history, it also tallies costs (testing, treatment, partner notification, adverse outcome management). This produces the full (ΔDALYs, ΔCosts) pair needed for the NMB calculation.

4. **Aggregate results**: Store per-woman outcomes and population-level summaries (total deliveries, adverse outcomes by type, total DALYs, total costs) at each timestep and cumulatively.

### Design considerations

- **FetalHealth does the mechanistic heavy lifting**: It handles infection timing, treatment effects, growth restriction, and delivery timing shifts. The DALY analyzer is a thin accounting layer on top.
- **Each parameter draw requires a full sim pair**: Because the mechanistic birth outcome parameters (delivery shifts, growth penalties, treatment residuals) affect sim dynamics (e.g., a larger PTB shift can cause delivery before the third-trimester screen), each draw requires running the sim. Cost parameters do NOT affect dynamics and CAN be swept post-hoc.
- **Discounting**: Use 3% per WHO-CHOICE convention for any life-years-lost calculations.
- **Compatibility**: Implement as Starsim `Analyzer` with `initialize()`, `step()`, `finalize()`. Needs access to `FetalHealth` states and the pregnancy module.

### Outputs

Per simulation run:
- Per-woman table: woman_id, ga_at_delivery, birth_weight, is_preterm, is_lbw, is_sga, dalys, costs
- Population summaries: total births, adverse outcomes by type, total DALYs, total costs

---

## Step 5: Build the Cost-Effectiveness Mapping

For each draw from the full parameter space, we run a pair of STIsim simulations (SOC vs. intervention) and compute incremental cost-effectiveness.

### Simulation design

For each parameter draw (combining a posterior epi parameter set from Step 2 with a draw from the birth outcome and cost priors in Step 3):

1. **Run STIsim (SOC arm)**: Simulate with current syndromic management only (no ANC STI screening). FetalHealth mechanistically tracks delivery timing shifts and growth restriction. DALY analyzer records outcomes at delivery.

2. **Run STIsim (intervention arm)**: Same population (using common random numbers), same birth outcome parameters, but with integrated POC testing at enrollment (≤24 weeks) and third trimester (32-34 weeks). Women testing positive receive directed treatment. Treatment residual values are early or late depending on gestational age at screen. Partner notification with probability `p_partner_tx`.

3. **Compute incremental costs and effects**: DALY analyzer provides per-arm totals. Compute ΔDALYs and ΔCosts.

### Incremental NMB calculation

For each parameter draw `i` and WTP threshold `λ`:

    NMB_i(λ) = λ × ΔDALYs_i − ΔCosts_i

### Compute budget

- **Total parameter draws**: Target ~1,000-2,000 draws for stable EVPI/EVPPI.
- **STIsim runs per draw**: 2 (SOC + intervention, using CRN)
- **Total runs**: ~2,000-4,000
- **Runtime per run**: ~30-60 seconds (n_agents=10,000, ~15 year burn-in)
- **Total**: ~30-60 hours single-machine. Embarrassingly parallel — 16 cores → ~2-4 hours wall-clock.

**Strategy**: Start with ~200 draws for prototyping. Scale to 1,000+ for final results. CRN dramatically reduces variance of incremental estimates.

**Note on cost parameters**: Cost parameters do NOT affect sim dynamics — they only enter the NMB calculation. For a given sim pair, cost uncertainty CAN be swept post-hoc. The birth outcome parameters (shifts, penalties, residuals) DO affect dynamics and require re-running.

---

## Step 6: VoI Analysis

### 6a. EVPI (Expected Value of Perfect Information)

EVPI(λ) = E_θ[max(NMB(θ), 0)] − max(E_θ[NMB(θ)], 0)

Computed directly from NMB samples. Report as $/woman and multiply by decision population for population EVPI.

### 6b. EVPPI (Expected Value of Partial Perfect Information)

Use the Strong et al. (2014) nonparametric regression method (GAM or gradient-boosted regression) to estimate EVPPI without nested Monte Carlo.

Parameter groups for EVPPI:

1. **Network structure** (calibrated): prop_f0, prop_m0, m1_conc
2. **Ng transmission/symptoms** (calibrated): ng.beta_m2f, ng.p_symp
3. **Ct transmission/symptoms** (calibrated): ct.beta_m2f, ct.p_symp
4. **Tv transmission/symptoms** (calibrated): tv.beta_m2f, tv.p_symp
5. **Delivery timing shifts** (prior only): ptb_shift_ng, ptb_shift_ct, ptb_shift_tv, ptb_shift_std
6. **Growth restriction penalties** (prior only): growth_penalty_ng, growth_penalty_ct, growth_penalty_tv
7. **Early treatment reversibility** (prior only): tx_residual_growth_early, tx_residual_timing_early
8. **Late treatment reversibility** (prior only): tx_residual_growth_late, tx_residual_timing_late
9. **Partner treatment**: p_partner_tx
10. **Cost parameters** (prior only, PLACEHOLDER): all cost parameters jointly

### 6c. EVSI (Expected Value of Sample Information)

EVSI estimates the expected value of the PROMISE trial specifically (N=12,780).

PROMISE will provide direct information on: STI prevalence in ANC women, the composite adverse birth outcome rate, the treatment effect (difference between arms), partner notification uptake/success.

PROMISE will NOT directly resolve: individual-STI-specific causal effects, reversibility parameters (entangled with causal effects in the composite outcome), generalizability to other settings, costs (unless the CE component produces transferable unit costs).

Use regression-based EVSI (Heath et al. 2020). Prioritize EVPI and EVPPI first; EVSI can be bounded (0 ≤ EVSI ≤ EVPI).

---

## Step 7: Figures and Reporting

### Figure 1: Prior predictive distribution of incremental NMB (`plot_fig1_nmb.py`) ✓
- 4 panels at WTP = $100, $500, $1000, $2000/DALY
- Histogram colored green (NMB > 0, cost-effective) / red (NMB < 0, not CE)
- Annotated with P(CE) and E[NMB] per panel
- Key message: quantifies current decision uncertainty

### Figure 2: Cost-effectiveness plane (`plot_fig2_ceplane.py`) ✓
- Scatter of (ΔDALYs averted, ΔCosts) across all parameter draws
- WTP threshold lines at $500, $1000, $2000
- Points colored by CE status at middle WTP, quadrant labels
- Key message: joint uncertainty in costs and effects

### Figure 3: EVPPI tornado chart (`plot_fig3_evppi.py`) ✓
- Horizontal bars sorted by EVPPI value for 9 parameter groups
- Color-coded: blue (epi), green (birth outcome), tan (cost)
- EVPI reference line for context
- Uses gradient-boosted regression with 5-fold cross-validation (Strong et al. 2014)
- Key message: which parameters drive decision uncertainty. Do reversibility or cost parameters dominate?

### Figure 4: Prior and posterior parameter distributions (`plot_fig4_priors_posteriors.py`) ✓
- Panel A (3×4): Calibrated epi parameters — uniform prior (light) + KDE posterior (dark)
- Panel B (3×4): Birth outcome priors — delivery shifts, growth penalties, treatment reversibility
- Panel C (2×4): Cost priors (flagged as placeholders with *)
- Key message: what we know vs. don't know

### Figure 5: Threshold analysis heatmap (TODO)
- NMB over 2 high-EVPPI parameter pairs, with decision boundary
- Key message: where the decision changes
- Implementation: depends on which parameter groups have highest EVPPI from Fig 3

### Figure 6: EVPI and EVSI curves (TODO)
- EVPI vs. WTP threshold — can be plotted directly from `voi_evpi.df`
- EVSI vs. WTP — requires regression-based EVSI (Heath et al. 2020), can be bounded by EVPI initially
- Key message: value of the PROMISE trial's information

### Supplementary figures ✓
- `plot_hiv_calibration.py` — HIV calibration validation (2×3: prevalence, infections, ART, PLHIV, deaths, population)
- `plot_sti_epi.py` — STI prevalence by age/sex + time series (2×3: NG, CT, TV)
- `plot_network.py` — Network structure (2×3: lifetime partners, age mixing, risk groups, debut, partnerships, condom use)

---

## Progress and Status

### Completed

- [x] **Step 1**: Epidemiological parameters defined (15 calib_pars in `priors.py`)
- [x] **Step 2**: Calibration infrastructure and execution
  - `run_calibrations.py` working — dot-notation parameter routing, scalar `set_pars` fix in stisim
  - Full calibration run complete (~2000 Optuna trials). Top 200 parameter sets saved to `results/zimbabwe_pars.df`
  - `run_msim.py` runs top 200 pars → percentile stats for validation
- [x] **Step 3**: Prior distributions defined and implemented
  - All prior-only parameters codified in `priors.py` with `sample_priors()` helper
  - Birth outcome priors: 4 delivery timing shifts, 3 growth penalties, 4 treatment reversibility
  - Cost priors: 8 parameters (all placeholders — need Zimbabwe-specific data)
  - DALY weights fixed at GBD 2019 values
  - Stillbirths dropped from scope
- [x] **Step 4a**: `FetalHealth` module — **upstreamed to Starsim** (`starsim.demographics.FetalHealth`)
  - Now a proper `ss.Module` (not an analyzer), registered with Pregnancy via conception/delivery callbacks
  - Tracks fetal growth restriction and PTB shift during pregnancy, classifies birth outcomes at delivery
  - Public API: `apply_timing_shift()`, `apply_growth_restriction()`, `reverse_timing_shift()`, `reverse_growth_restriction()`
  - Results: `n_deliveries`, `n_preterm`, `n_lbw`, `n_sga`, `mean_birth_weight`, `mean_ga_at_birth`, `preterm_rate`, `lbw_rate`
  - Starsim PR: https://github.com/starsimhub/starsim/pull/1244 (branch: `fetal-health`)
  - Test in `starsim/tests/test_demographics.py::test_fetal_health` covers baseline, disease, and treatment scenarios
- [x] **Step 4b**: `sti_fetal` connector (`connectors.py`) — routes new infections and treatments in pregnant women to FetalHealth
  - Registers conception callback to handle pre-existing infections at start of pregnancy
  - `step()` checks for new infections and new treatments each timestep
  - Treatment reversibility depends on gestational age (early ≤24w vs late) with separate residual parameters for growth and timing
  - All birth outcome parameters (`ptb_shift_mean`, `growth_penalty`, `tx_residual_*`) are constructor arguments, overridable per VoI draw via `connector_pars` in `make_sim()`
- [x] **Step 4c**: `birth_outcome_dalys` analyzer (`analyzers.py`) — computes YLD from PTB and LBW at delivery
  - Bug fixed: `cum_dalys` computed in `finalize()` via `np.cumsum`, not inline (early-return skipped non-delivery timesteps)
  - Note: `yld_lbw` is effectively 0 because nearly all LBW births are also PTB in this model (preterm infants are mechanically <2500g). DALYs are driven by PTB count. This is epidemiologically correct.
- [x] **Step 4d**: `intervention_costs` analyzer (`analyzers.py`) — tracks screening, treatment, partner notification, and adverse outcome management costs
  - Bug fixed: `cum_cost` computed in `finalize()` via `np.cumsum` (same early-return fix as cum_dalys)
  - All unit cost parameters are constructor arguments, overridable per VoI draw
- [x] **Model integration**: `model.py` `make_sim()` includes FetalHealth (via `custom=`) + sti_fetal connector by default; DALY and cost analyzers passed as `extra_analyzers` for VoI runs; `connector_pars` argument allows overriding birth outcome parameters per draw
- [x] **Step 5**: `run_voi.py` built and running
  - Loads calibration posterior (top 200 from `zimbabwe_pars.df`)
  - Samples birth outcome + cost parameters from priors per draw
  - Builds SOC + intervention (twice) sim pairs with shared seed (CRN)
  - Applies calibrated epi parameters via `set_sim_pars()`
  - Computes ΔDALYs and ΔCosts per draw
  - Stores all parameter values per draw for EVPPI regression
  - Outputs: `voi_draws.df`, `voi_evpi.df`
- [x] **Step 6a**: EVPI computed from NMB samples across 7 WTP thresholds ($50–$5000)
  - Implemented in `compute_evpi()` within `run_voi.py`
- [x] **Step 6b**: EVPPI implemented via gradient-boosted regression (Strong et al. 2014)
  - `plot_fig3_evppi.py` computes EVPPI for 9 parameter groups using cross-validated GBR
  - Groups: network structure, NG/CT/TV transmission, delivery timing shifts, growth penalties, early/late treatment reversibility, cost parameters
- [x] **Step 7 (partial)**: Figure scripts written
  - `plot_fig1_nmb.py` — NMB histograms at 4 WTP thresholds, colored green/red
  - `plot_fig2_ceplane.py` — Cost-effectiveness plane scatter with WTP lines
  - `plot_fig3_evppi.py` — EVPPI tornado chart with EVPI reference line
  - `plot_fig4_priors_posteriors.py` — Prior and posterior parameter distributions (8×4 grid)
  - Supplementary: `plot_hiv_calibration.py`, `plot_sti_epi.py`, `plot_network.py`
- [x] **Interventions**: Full PROMISE intervention suite implemented in `interventions.py`
  - `SyndromicMgmt`: syndromic management of VDS/UDS (SOC)
  - `ANCScreen`: GA-windowed ANC screening with per-disease sensitivity/specificity
  - `STIPartnerNotification`: notify/treat partners of ANC-positive women
  - `make_testing()`: factory assembles all interventions for a given scenario
  - 5 scenarios: soc, enroll, tri3, twice, partner_tx

### Design decisions made

- **Stillbirths dropped from scope**: Focus on PTB and LBW (and SGA). Simplifies model and parameter space.
- **PTB vs LBW double-counting**: PTB+LBW co-occurrences accrue PTB disability weight only. In practice, nearly all LBW is PTB-driven in the mechanistic model.
- **FetalHealth is a Starsim Module, not an analyzer**: Upstreamed to `starsim.demographics.FetalHealth`. Added via `custom=` so it registers conception/delivery callbacks with Pregnancy. This is generic infrastructure reusable by any Starsim model with pregnancy.
- **Connector-based treatment reversal**: The `sti_fetal` connector handles both infection damage and treatment reversal in its `step()` method. No new callback machinery was needed in FetalHealth — the connector checks `tx.ti_treated == ti` for newly treated pregnant women and calls `fh.reverse_growth_restriction()` / `fh.reverse_timing_shift()`. This keeps FetalHealth generic while the connector holds all disease-specific logic.
- **Mechanistic model over post-hoc replay**: FetalHealth mechanistically shifts delivery timing and accumulates growth restriction. Birth outcome parameters affect sim dynamics (e.g., larger PTB shift → delivery before third-trimester screen), so each parameter draw requires a full sim pair. This is more expensive than post-hoc replay but preserves the mechanistic interactions that make the ABM valuable.
- **POC test fixed at 95/95**: Not in the VoI parameter space.
- **HIV parameters fixed at posterior**: Calibrated but not propagated into VoI uncertainty.
- **CRN for variance reduction**: SOC and intervention sims share a random seed per draw, so incremental differences reflect the intervention effect rather than stochastic noise.
- **Cost parameters don't affect dynamics**: They only enter the NMB calculation. In principle they could be swept post-hoc, but for simplicity they are sampled per draw alongside epi parameters.

### In progress

- [ ] **VoI runs on VMs**: `run_voi.py` running with 200 draws (400 sim runs). Awaiting completion.

### Next steps

- [ ] **Review VoI outputs**: Inspect `voi_draws.df` and `voi_evpi.df` for plausibility — are ΔDALYs and ΔCosts in reasonable ranges?
- [ ] **Generate Figures 1-4**: Run plot scripts once VoI results are available
- [ ] **Figure 5 (threshold analysis heatmap)**: Not yet implemented — NMB over 2 high-EVPPI parameter pairs
- [ ] **Figure 6 (EVPI/EVSI curves)**: EVPI curve can be plotted from `voi_evpi.df`; EVSI not yet implemented (can be bounded by EVPI)
- [ ] **Step 6c**: EVSI via regression-based method (Heath et al. 2020) — lower priority, bounded by EVPI
- [ ] **Scale up**: Increase from 200 to 1000+ draws for stable EVPPI estimates
- [ ] Collect Zimbabwe-specific cost data to replace placeholders
- [ ] Get clinical input on treatment reversibility framing (especially timing ratchet)
- [ ] Run validation plots once `run_msim.py` calib stats are available on VMs

---

## Key Discussion Points for Check-in

1. **Are the calibration targets right?** The PROMISE team may have site-specific prevalence data from the Harare clinics.

2. **Treatment reversibility — is the framing right?** Two key questions for clinical collaborators: (a) Can treatment reverse delivery timing shifts at all, or is the one-way ratchet correct? (b) Should reversibility depend on gestational age at treatment (early vs. late), on the specific STI, or both? The EVPPI analysis will tell us how much these choices matter for the decision.

3. **Cost data — what do we have?** All cost priors are placeholders. The PROMISE CE component may have budgeted costs. If costs dominate EVPPI, that's an actionable finding — costing work is a priority.

4. **What WTP threshold for Zimbabwe?** Old WHO 1-3× GDP/capita gives ~$1,800-$5,400. Opportunity-cost estimates (Woods 2016; Ochalek 2018) suggest $50-$500/DALY. This matters enormously.

5. **Population EVPI scaling**: What decision population — all ANC women in Zimbabwe (~400k/year)? Regionally?

6. **What will PROMISE tell us vs. not?** The trial measures the composite intervention effect. It cannot separately identify STI-specific causal effects vs. reversibility parameters. The VoI analysis clarifies whether this matters.

---

## Dependencies and Setup

- **starsim** (`fetal-health` branch) — includes `FetalHealth` module in demographics
- **stisim** (`prep-uplift` branch) — calibration API (`set_sim_pars`, `make_calib_sims`), disease modules, interventions
- **sciris** (>=3.1.6) — utilities, parallelization, file I/O
- **optuna** — Bayesian calibration
- **scikit-learn** — gradient-boosted regression for EVPPI (Strong et al. 2014)
- Standard scientific Python: scipy, numpy, pandas, matplotlib, seaborn
