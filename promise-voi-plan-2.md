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

The `FetalHealth` module uses a mechanistic approach with two separate damage pathways:

1. **Delivery timing (→ PTB)**: When a pregnant woman acquires an STI, her delivery date is shifted earlier by a stochastic amount drawn from a lognormal distribution. The mean shift depends on the pathogen (Ng shifts more than Ct or Tv). This is currently a one-way ratchet — reinfection compounds the shift, and treatment may or may not partially reverse it. A birth is classified as preterm if delivery occurs before 37 weeks' gestation.

2. **Fetal growth restriction (→ LBW/SGA)**: Each infection accumulates a growth penalty that reduces birth weight as a fraction of the gestational-age-appropriate weight. Treatment partially reverses this penalty, but leaves a residual. Birth weight is computed at delivery as: `baseline_weight_for_GA × individual_percentile × (1 − growth_restriction)`. A birth is classified as LBW if birth weight < 2500g, and SGA if below the 10th percentile for gestational age.

This mechanistic approach is richer than a simple probability/RR model because outcomes emerge from the interaction of infection timing, treatment timing, baseline fetal growth trajectory, and gestational age. It also means the VoI analysis runs pairs of full simulations (SOC vs. intervention) for each parameter draw, rather than replaying event logs post-hoc.

**Two infection timing scenarios to handle:**
- **Pre-existing infection at conception**: A woman already infected when she becomes pregnant. FetalHealth applies infection effects at conception via `on_conception()`. If treated early (≤24w), use early reversibility parameters; if treated late, use late parameters; if untreated, full damage applies at delivery.
- **New infection during pregnancy**: FetalHealth applies infection effects when the infection is acquired, via the `sti_fetal` connector. Same reversibility logic based on gestational age at treatment.

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

### Figure 1: Prior predictive distribution of incremental NMB
- Histogram colored green (NMB > 0) / red (NMB < 0), at multiple WTP thresholds
- Key message: quantifies current decision uncertainty

### Figure 2: Cost-effectiveness plane
- Scatter of (ΔDALYs, ΔCosts) with WTP line
- Key message: joint uncertainty in costs and effects

### Figure 3: EVPPI tornado chart
- Horizontal bars sorted by EVPPI value
- Key message: which parameters drive decision uncertainty. Do reversibility or cost parameters dominate?

### Figure 4: Prior and posterior parameter distributions (8×4 panel grid)
- Epi parameters: prior (light) + posterior (dark). Birth outcome and cost: prior only, labeled.
- Key message: what we know vs. don't know

### Figure 5: Threshold analysis heatmap
- NMB over 2 high-EVPPI parameter pairs, with decision boundary
- Key message: where the decision changes

### Figure 6: EVPI and EVSI curves
- EVPI and EVSI vs. WTP threshold, with uncertainty band on EVSI
- Key message: value of the PROMISE trial's information

### Supplementary: Calibration diagnostics
- Model vs. data for each target
- Optuna convergence
- Pairwise posterior correlations

---

## Progress and Status

### Completed

- [x] **Step 1**: Epidemiological parameters defined (15 calib_pars)
- [x] **Step 2 infrastructure**: `run_calibrations.py` working — dot-notation parameter routing, scalar `set_pars` fix in stisim, CSV column names updated to dot notation, weights dict updated
- [x] **Step 3 (plan)**: Prior parameters defined in this document (stillbirths dropped)
- [x] **Step 4a**: `FetalHealth` module (`fetal_health.py`) — tracks fetal growth restriction and PTB shift during pregnancy, classifies birth outcomes at delivery
  - Bug fixed: GA computation uses `ti_delivery - ti_pregnant` (actual) not `dur_pregnancy` (scheduled)
  - Bug fixed: delivery detection uses `(ti_delivery == ti) & ~pregnant` (analyzers run after pregnancy.step clears pregnant)
- [x] **Step 4b**: `sti_fetal` connector (`connectors.py`) — routes new infections and treatments in pregnant women to FetalHealth
- [x] **Step 4c**: `birth_outcome_dalys` analyzer (`analyzers.py`) — computes YLD from PTB and LBW at delivery
  - Bug fixed: `cum_dalys` computed in `finalize()` via `np.cumsum`, not inline (early-return skipped non-delivery timesteps)
  - Note: `yld_lbw` is effectively 0 because nearly all LBW births are also PTB in this model (preterm infants are mechanically <2500g). DALYs are driven by PTB count. This is epidemiologically correct.
- [x] **Step 4d**: `intervention_costs` analyzer (`analyzers.py`) — tracks screening, treatment, and adverse outcome management costs
  - Bug fixed: `cum_cost` computed in `finalize()` via `np.cumsum` (same early-return fix as cum_dalys)
- [x] **Model integration**: `model.py` `make_sim()` includes FetalHealth + sti_fetal connector by default; DALY and cost analyzers passed as `extra_analyzers` for VoI runs

### Design decisions made

- **Stillbirths dropped from scope**: Focus on PTB and LBW (and SGA). Simplifies model and parameter space.
- **PTB vs LBW double-counting**: PTB+LBW co-occurrences accrue PTB disability weight only. In practice, nearly all LBW is PTB-driven in the mechanistic model.
- **FetalHealth is an analyzer, not a module**: Added via `analyzers=` list so it runs at func_order ~97 (after pregnancy.step processes deliveries).
- **Mechanistic model over post-hoc replay**: FetalHealth mechanistically shifts delivery timing and accumulates growth restriction. Birth outcome parameters affect sim dynamics (e.g., larger PTB shift → delivery before third-trimester screen), so each parameter draw requires a full sim pair. This is more expensive than post-hoc replay but preserves the mechanistic interactions that make the ABM valuable.
- **POC test fixed at 95/95**: Not in the VoI parameter space.
- **HIV parameters fixed at posterior**: Calibrated but not propagated into VoI uncertainty.

### Next steps

- [ ] **Smoke test end-to-end**: Run SOC and intervention scenarios with DALY and cost analyzers attached, verify outputs
- [ ] **Wire FetalHealth to accept uncertain parameters at runtime**: The mechanistic parameters (ptb_shift_mean, growth_penalty, treatment_residual) are currently hardcoded constants. They need to accept values from the prior draws. Specifically:
  - `ptb_shift_mean` dict (ng, ct, tv) → from `ptb_shift_ng/ct/tv` priors
  - `ptb_shift_std` → from `ptb_shift_std` prior
  - `growth_penalty` dict (ng, ct, tv) → from `growth_penalty_ng/ct/tv` priors
  - `treatment_residual` → replaced by `tx_residual_growth_early/late` and `tx_residual_timing_early/late`
  - This requires modifying `apply_treatment_effects()` to accept gestational age and select early vs. late residual
  - Also requires modifying `apply_treatment_effects()` to optionally reverse delivery timing shift (relaxing the one-way ratchet)
- [ ] **Step 2**: Run full calibration (~500-2000 Optuna trials) on VMs
- [ ] **Step 5**: Build `run_voi.py` — loads calibration posterior (top 200), samples birth outcome + cost parameters per posterior draw, builds SOC + intervention sim pairs with CRN, runs with analyzers, computes INMB
- [ ] **Step 6a**: EVPI from NMB samples
- [ ] **Step 6b**: EVPPI via nonparametric regression (Strong et al. 2014)
- [ ] **Step 6c**: EVSI (can be bounded by EVPI initially)
- [ ] **Step 7**: Figures 1-6 + supplementary calibration diagnostics
- [ ] Collect Zimbabwe-specific cost data to replace placeholders
- [ ] Get clinical input on treatment reversibility framing (especially timing ratchet)

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

The analysis requires STIsim and dependencies (Starsim), Optuna for calibration, and standard scientific Python (scipy, numpy, pandas, matplotlib). The `stisim_vddx_zim` example is the starting template.

For EVPPI: pygam (GAM per Strong et al. 2014) or scikit-learn (gradient-boosted regression).
