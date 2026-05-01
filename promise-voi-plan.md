# PROMISE Trial: Value of Information Analysis Using STIsim

## Overview

This document outlines the modeling pipeline for a pre-trial value of information (VoI) analysis of the PROMISE trial, using the STIsim agent-based framework calibrated to Zimbabwe. The goal is to quantify decision uncertainty around the cost-effectiveness of integrated antenatal STI screening and identify which parameters drive that uncertainty — before trial readouts are available.

The analysis proceeds in seven steps:

1. Define prior distributions on key epidemiological parameters (already done — these are the `calib_pars`)
2. Calibrate the STIsim Zimbabwe model to prevalence data using Optuna
3. Combine calibrated (posterior) epi parameters with prior distributions on birth outcome, treatment-of-birth-outcome, and cost parameters
4. Build a DALY analyzer to track infection/treatment histories and compute birth outcome DALYs
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
- Diagnostic sensitivity parameters (e.g., `sens_poc_ct`) are NOT included here because the standard of care is no ANC screening — there is no diagnostic to characterize in the SOC arm. The intervention arm's diagnostic performance enters at Step 3.
- Treatment efficacy for clearing STIs is well-established and is already handled within STIsim (azithromycin clears Ct, dual therapy clears Ng, metronidazole clears Tv at known rates). What is NOT known is the extent to which treating an STI during pregnancy reverses the damage to birth outcomes — this is a separate question addressed in Step 3.
- **HIV parameters are calibrated but fixed for the VoI analysis.** The three HIV parameters (beta_m2f, eff_condom, rel_init_prev) remain in `calib_pars` because HIV affects network structure and background STI dynamics. However, they are not part of the VoI decision space — the decision is about STI screening, not HIV management. After calibration, HIV parameters are fixed at their posterior point estimates (or posterior median) and do not contribute to the uncertainty propagation in Steps 5-6. This keeps the VoI parameter space at 40 uncertain parameters.

---

## Step 2: Calibrate to Prevalence Data Using Optuna

### Calibration targets

We calibrate against observed prevalence data from Zimbabwe, primarily from ANC surveillance and population-based surveys:

| Target | Estimate | 95% CI | Source |
|--------|----------|--------|--------|
| HIV prevalence (women 15-49) | 14.5% | (13.0%, 16.0%) | ZIMPHIA 2020 |
| Ng prevalence (ANC women) | 3.5% | (1.5%, 6.0%) | Mudau et al. 2018; Harare ANC data |
| Ct prevalence (ANC women) | 8.0% | (5.0%, 12.0%) | Mudau et al. 2018; Peters et al. 2021 |
| Tv prevalence (ANC women) | 18.0% | (12.0%, 25.0%) | Masha et al. 2019; Harare data |
| HIV prevalence (ANC women) | 16.0% | (14.0%, 19.0%) | Zimbabwe PMTCT data |

**TODO:** Confirm exact calibration targets with the PROMISE study team. Additional targets (e.g., prevalence by age group, co-infection rates) should be added if data are available, as they help identify network parameters.

### Calibration approach

Use Optuna to search the 15-dimensional parameter space defined by `calib_pars`, minimizing a weighted sum of squared deviations between model-predicted prevalences and the calibration targets above. The Optuna search bounds are already specified in the `calib_pars` dictionary (low/high for each parameter).

The objective function should:
- Accept an Optuna trial and suggest values for all 15 parameters using the bounds and log-scale flags from `calib_pars`
- Build and run the STIsim Zimbabwe model (adapting the `stisim_vddx_zim` example as a template)
- Extract model-predicted prevalences at the calibration timepoint
- Compute a weighted least-squares goodness-of-fit, with weights inversely proportional to the uncertainty in each target

Start with ~500 Optuna trials. The top ~100-200 parameter sets (ranked by goodness of fit) form the approximate posterior for the epi parameters.

### Outputs from Step 2

- **Posterior parameter samples**: Top ~100-200 parameter sets from Optuna, representing combinations of epi parameters consistent with observed prevalence data.
- **Calibration diagnostics**: For each top parameter set, model vs. data comparison.
- **Prior vs. posterior comparison**: Marginal distributions of each epi parameter before and after calibration (feeds into Figure 4).

---

## Step 3: Define Priors on Birth Outcome, Treatment-of-Outcome, and Cost Parameters

These parameters are NOT informed by the STIsim calibration. The epi model tells us who is infected with what and when; the links from infection to adverse pregnancy outcomes, the reversibility of that damage through treatment, and costs all come from external evidence or are genuinely unknown.

### 3a. How birth outcomes work in the model

The mechanism within STIsim is as follows:

1. When a woman becomes pregnant, she is assigned birth outcome probabilities based on baseline risks (PTB, LBW, stillbirth).
2. **If she acquires an STI during pregnancy**, her birth outcome probabilities are updated (worsened) according to the RRs in Section 3b. The model resets her birth outcomes at the point of infection.
3. **If she is then treated and the STI is cleared**, the question is: how much do her birth outcome probabilities recover? Treatment clears the infection (this is well-established), but it may not fully reverse the inflammatory damage, placental changes, or other mechanisms through which the STI affects the pregnancy.

This means we need a parameter that captures the **degree of birth outcome reversibility upon treatment** — not treatment efficacy for clearing the infection itself (which is known), but the extent to which clearing the infection restores birth outcome probabilities toward baseline.

### 3b. Relative risks: STI infection → adverse birth outcomes

These RRs are applied at the point of infection to worsen birth outcome probabilities.

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `rr_ng_ptb` | RR of preterm birth given Ng infection | LogNormal(μ=log(1.4), σ=0.35) | Vallely 2021 meta-analysis: RR 1.40 (CI 1.14-1.73); Taylor 2023; Felske 2022. DALY doc: 14.7% PTB rate |
| `rr_ct_ptb` | RR of preterm birth given Ct infection | LogNormal(μ=log(1.35), σ=0.3) | He 2020 meta-analysis: OR 1.35 (CI 1.11-1.63); Olson-Chen 2018. DALY doc: 16.1% PTB rate |
| `rr_tv_ptb` | RR of preterm birth given Tv infection | LogNormal(μ=log(1.4), σ=0.25) | Silver 2014 meta-analysis: RR 1.42 (CI 1.15-1.75). DALY doc: 13.6% PTB rate |
| `rr_ng_lbw` | RR of LBW given Ng infection | LogNormal(μ=log(2.2), σ=0.4) | Vallely 2021 meta-analysis: RR 2.23 (CI 1.34-3.71); Heumann 2017. DALY doc: 22.3% LBW rate. Wide prior reflects sparse evidence |
| `rr_ct_lbw` | RR of LBW given Ct infection | LogNormal(μ=log(1.5), σ=0.4) | He 2020: OR 1.49 (CI 0.90-2.47, not significant). Wide prior reflects uncertainty |
| `rr_tv_lbw` | RR of LBW given Tv infection | LogNormal(μ=log(1.5), σ=0.25) | Silver 2014 meta-analysis: RR 1.51 (CI 1.32-1.73). Also reports SGA RR 1.15 |
| ~~`rr_ct_sb`~~ | ~~RR of stillbirth given Ct infection~~ | ~~LogNormal(μ=log(1.2), σ=0.4)~~ | **DROPPED — stillbirths out of scope** |
| ~~`rr_ng_sb`~~ | ~~RR of stillbirth given Ng infection~~ | ~~LogNormal(μ=log(1.3), σ=0.4)~~ | **DROPPED** |
| ~~`rr_tv_sb`~~ | ~~RR of stillbirth given Tv infection~~ | ~~LogNormal(μ=log(1.1), σ=0.3)~~ | **DROPPED** |

### 3c. Reversibility of birth outcome risk upon STI treatment

When a pregnant woman's STI is treated and cleared, her birth outcome probabilities are reset — but potentially not all the way back to baseline. The reversibility parameter `κ` (kappa) captures this:

- κ = 1.0: Full reversibility. Treatment fully restores birth outcome probabilities to what they would have been without infection. (Optimistic: assumes all damage is mediated by ongoing infection.)
- κ = 0.0: No reversibility. Treatment clears the infection but birth outcome probabilities remain elevated. (Pessimistic: assumes damage is done at the point of infection and is irreversible.)
- κ between 0 and 1: Partial reversibility. After treatment, the woman's RR moves from `rr_sti` partway back toward 1.0. Specifically, post-treatment RR = 1 + (1 − κ)(rr_sti − 1).

Reversibility is split by treatment timing to reflect the two PROMISE screening timepoints (enrollment ≤24 weeks vs. third trimester 32-34 weeks):

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `kappa_ptb_early` | Reversibility of PTB risk, treatment ≤24 weeks | Beta(α=4, β=2) → mean ~0.67 | Earlier treatment → more recovery; weak evidence from timing-of-treatment subgroup analyses |
| `kappa_ptb_late` | Reversibility of PTB risk, treatment 32-34 weeks | Beta(α=2, β=2) → mean ~0.50 | Less time for recovery; limited opportunity to reverse late-pregnancy inflammation |
| `kappa_lbw_early` | Reversibility of LBW risk, treatment ≤24 weeks | Beta(α=4, β=2) → mean ~0.67 | Similar rationale to PTB |
| `kappa_lbw_late` | Reversibility of LBW risk, treatment 32-34 weeks | Beta(α=2, β=2) → mean ~0.50 | Fetal growth restriction in third trimester may be less reversible |
| ~~`kappa_sb`~~ | ~~Reversibility of stillbirth risk upon STI treatment~~ | ~~Beta(α=2, β=2) → mean ~0.50~~ | **DROPPED — stillbirths out of scope** |

**Discussion:** The early/late split is directly relevant to the PROMISE design question: how much of the intervention's value comes from the enrollment screen vs. the third-trimester screen? If κ_late is substantially lower than κ_early, that suggests the first screen is doing most of the work, which has implications for a simplified (single-screen) implementation. The EVPPI analysis will tell us how much the early vs. late distinction matters for the decision.

**Note on priors:** The early κ priors are shifted higher (more optimistic) than the late κ priors, reflecting the intuition that earlier treatment is more effective at restoring outcomes. But both priors are wide, reflecting genuine uncertainty. The prior means are illustrative — clinical collaborators should review these.

### 3d. Baseline birth outcome risks (without STI)

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `p_ptb_base` | Baseline preterm birth rate | Beta(α=12, β=88) → mean ~0.12 | Zimbabwe DHS; Blencowe et al. 2012 |
| `p_lbw_base` | Baseline LBW rate | Beta(α=10, β=90) → mean ~0.10 | Zimbabwe DHS |
| ~~`p_sb_base`~~ | ~~Baseline stillbirth rate~~ | ~~Beta(α=2, β=98) → mean ~0.02~~ | **DROPPED — stillbirths out of scope** |

### 3e. Intervention parameters (screening-specific)

The intervention arm introduces POC testing at two ANC timepoints. SOC has no ANC STI screening.

POC test sensitivity and specificity are fixed at 95%/95% for all three STIs (Ct, Ng, Tv). This assumes a high-quality test platform and keeps the parameter space manageable. The diagnostic performance is relatively well-characterized and is not where the interesting decision uncertainty lies.

| Parameter | Description | Value | Notes |
|-----------|-------------|-------|-------|
| `sens_poc` | POC test sensitivity (all STIs) | 0.95 (fixed) | Assumed high-quality test |
| `spec_poc` | POC test specificity (all STIs) | 0.95 (fixed) | Assumed high-quality test |

The remaining uncertain intervention parameter is partner notification success:

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `p_partner_tx` | Partner notification + treatment success | Beta(α=3, β=7) → mean ~0.30 | Ferreira et al. 2013; highly uncertain |

Note: Reinfection between the two screening timepoints is NOT an exogenous parameter — it emerges from STIsim dynamics, conditional on partner treatment success and community prevalence. This is a key advantage of the agent-based approach.

### 3f. Cost parameters

**Note: We do not currently have cost data for this setting. All cost priors below are placeholders based on general estimates from similar sub-Saharan African settings. These need to be refined with input from the PROMISE study team and Zimbabwe-specific health system cost data. The cost priors are likely among the weakest inputs to this analysis.**

| Parameter | Description | Prior distribution | Source / rationale |
|-----------|-------------|-------------------|-------------------|
| `cost_poc_test` | Cost per POC test (3 STIs) | Gamma(μ=$8, σ=$4) | **PLACEHOLDER** — depends heavily on device platform and volume |
| `cost_tx_ct` | Treatment cost, Ct (azithromycin) | Gamma(μ=$3, σ=$2) | **PLACEHOLDER** |
| `cost_tx_ng` | Treatment cost, Ng (dual therapy) | Gamma(μ=$5, σ=$3) | **PLACEHOLDER** — AMR may require more expensive regimens |
| `cost_tx_tv` | Treatment cost, Tv (metronidazole) | Gamma(μ=$2, σ=$1) | **PLACEHOLDER** |
| `cost_partner_notif` | Cost per partner notification episode | Gamma(μ=$5, σ=$4) | **PLACEHOLDER** — highly variable by implementation model |
| `cost_anc_visit` | Marginal cost of integrating STI testing into ANC visit | Gamma(μ=$3, σ=$2) | **PLACEHOLDER** — staff time, consumables beyond the test |
| `cost_ptb_mgmt` | Cost of managing preterm birth | Gamma(μ=$300, σ=$200) | **PLACEHOLDER** — enormous range depending on gestational age and facility level |
| `cost_lbw_mgmt` | Cost of managing LBW | Gamma(μ=$200, σ=$150) | **PLACEHOLDER** — overlaps substantially with PTB costs |
| ~~`cost_sb_mgmt`~~ | ~~Cost of stillbirth management~~ | ~~Gamma(μ=$150, σ=$100)~~ | **DROPPED — stillbirths out of scope** |
| ~~`cost_neonatal_death`~~ | ~~Cost of neonatal death management~~ | ~~Gamma(μ=$250, σ=$180)~~ | **DROPPED** |

The wide priors on costs are deliberate — they reflect genuine ignorance and will likely show up as high-EVPPI parameters, which itself is a useful finding (it tells us that costing data is a priority input).

### 3g. DALY weights

| Outcome | DALY weight | Duration | Source |
|---------|-------------|----------|--------|
| Preterm birth | 0.15 | Acute + sequelae | GBD 2019 |
| LBW | 0.10 | Acute + sequelae | GBD 2019 |
| ~~Stillbirth~~ | ~~15.0 life-years~~ | ~~Full life lost~~ | **DROPPED — stillbirths out of scope** |
| ~~Neonatal death~~ | ~~15.0 life-years~~ | ~~Full life lost~~ | **DROPPED** |

### 3h. Co-infection interaction terms (structural uncertainty)

We will run two model structures:
- **Multiplicative model**: RRs for each STI multiply independently. A woman with Ct + Tv has risk = baseline × rr_ct × rr_tv.
- **Interaction model**: Include a co-infection interaction parameter, `rr_coinfection_modifier ~ LogNormal(μ=0, σ=0.15)`, allowing synergistic or antagonistic effects.

The structural uncertainty between these models will be handled via model averaging in the VoI analysis.

---

## Step 4: Build a DALY Analyzer

Before we can compute cost-effectiveness, we need a Starsim analyzer that runs alongside the simulation and computes DALYs from birth outcomes. This analyzer will be attached to each STIsim run and will track, for each pregnant woman, her infection history, treatment history, and resulting birth outcomes — then translate those into DALYs.

### What the analyzer needs to do

The DALY analyzer should operate as a Starsim `Analyzer` subclass that hooks into the simulation at each timestep (or at key events: infection, treatment, birth). Specifically:

1. **Track pregnant women**: Identify women who become pregnant during the simulation. Record their infection status at conception and monitor for new STI acquisitions during pregnancy.

2. **Record infection events during pregnancy**: When a pregnant woman acquires Ct, Ng, or Tv, log the infection type, gestational age at infection, and the resulting adjustment to her birth outcome probabilities (applying the RRs from Step 3b). Handle co-infections by applying RRs multiplicatively (or with the interaction modifier, if using the interaction model structure).

3. **Record treatment events during pregnancy**: When a pregnant woman's STI is treated and cleared, log the treatment timepoint (gestational age) and apply the κ-based reversibility to partially restore birth outcome probabilities (Step 3c). Post-treatment RR = 1 + (1 − κ)(rr_sti − 1) for each outcome.

4. **Resolve birth outcomes**: At the time of birth, sample each woman's outcomes (PTB, LBW, stillbirth) from her final adjusted probabilities. These are not mutually exclusive — a birth can be both preterm and low birth weight, for example. Stillbirth precludes neonatal outcomes.

5. **Compute DALYs per birth**: For each adverse outcome, apply the DALY weights from Step 3g. For the composite:
    - PTB (live birth): disability weight × duration for acute and long-term sequelae
    - LBW (live birth): disability weight × duration
    - Stillbirth: full discounted life-years lost
    - Neonatal death (if modeled as a secondary outcome): full discounted life-years lost
    - For births with multiple adverse outcomes (e.g., preterm + LBW), DALYs should be computed to avoid double-counting — use the maximum disability weight or an additive approach with adjustment, depending on the GBD methodology preferred.

6. **Aggregate and store results**: The analyzer should store per-woman records (for downstream post-processing across the Step 3 parameter draws) and population-level summaries (total DALYs, total adverse outcomes by type, broken down by arm).

### Design considerations

- **Separation of concerns**: The analyzer computes DALYs from birth outcomes given the RRs and κ values. For the VoI analysis, we want to run each STIsim simulation once per posterior epi parameter set, then sweep across many draws of the birth outcome / κ / cost parameters in post-processing. This means the analyzer should store enough intermediate information (each woman's infection and treatment history) that the DALY calculation can be re-run with different RR and κ values without re-running STIsim. In practice, this means storing a table of (woman_id, infection_events[], treatment_events[], gestational_ages[]) that can be replayed.

- **Discounting**: DALYs for stillbirth and neonatal death involve discounting future life-years. Use a standard discount rate (3% per WHO-CHOICE convention). The choice of discount rate has a small effect and can be tested in sensitivity analysis.

- **Overlap between PTB and LBW**: Many preterm births are also low birth weight. The analyzer should track both conditions independently and handle the DALY computation carefully to avoid double-counting the overlapping component. One approach: compute DALYs for the "small vulnerable newborn" composite (which PROMISE also measures as a secondary outcome), which captures the joint distribution.

- **Compatibility with STIsim/Starsim**: The analyzer should follow the Starsim analyzer pattern — implementing `initialize()`, `step()`, and `finalize()` methods. It will need access to the pregnancy module's state (which women are pregnant, gestational age) and the disease modules' state (infection status, treatment events).

### Outputs

The analyzer produces, for each simulation run:
- A per-woman table: woman_id, arm, infection_history, treatment_history, birth_outcomes, DALYs
- Population summaries: total births, adverse outcomes by type, total DALYs
- These feed directly into the cost-effectiveness mapping in Step 5.

---



## Step 5: Build the Cost-Effectiveness Mapping

For each combination of (posterior epi parameters from Step 2) × (sampled birth outcome, reversibility, and cost parameters from Step 3), we compute the incremental cost-effectiveness of the PROMISE screening strategy vs. standard of care. The DALY analyzer from Step 4 provides the per-woman intermediate outputs that make this step fast.

### Simulation design

For each posterior epi parameter draw:

1. **Run STIsim (standard of care arm)**: Simulate the population with current syndromic management only (no ANC STI screening). The DALY analyzer records each pregnant woman's infection and treatment history.

2. **Run STIsim (intervention arm)**: Same population (using common random numbers), but with integrated POC testing at enrollment (≤24 weeks) and third trimester (32-34 weeks). Women who test positive receive directed treatment. The DALY analyzer records the same per-woman data.

For each pair of STIsim runs, sweep across the Step 3 parameter space in post-processing (this is fast because it replays the per-woman records from the DALY analyzer with different RR, κ, and cost values):

3. **Replay birth outcomes**: For each draw of RRs and κ values, recompute each woman's birth outcome probabilities and DALYs from the stored infection/treatment histories.

4. **Compute costs and effects**: Tally incremental costs (testing, treatment, partner notification, averted adverse outcome management costs) and incremental effects (adverse outcomes averted, DALYs averted).

### Incremental NMB calculation

For each parameter draw `i` and WTP threshold `λ`:

    NMB_i(λ) = λ × ΔDALYs_i − ΔCosts_i

Where ΔDALYs_i = DALYs averted per woman screened (intervention vs. SOC) and ΔCosts_i = incremental cost per woman screened.

### Compute budget

- **Posterior epi samples**: ~200 (from Step 2)
- **Birth outcome / reversibility / cost prior draws**: ~100 per epi sample (fast — just sampling + arithmetic)
- **Total parameter combinations**: ~20,000
- **STIsim runs**: ~400 (200 SOC + 200 intervention, using CRN). The birth outcome and cost layer is fast post-processing.
- **Estimated runtime**: With n_agents=10,000 and ~15 year burn-in, expect ~30-60 seconds per run. Total: ~6-12 hours on a single machine. Parallelizable.

---

## Step 6: VoI Analysis

### 6a. EVPI (Expected Value of Perfect Information)

EVPI(λ) = E_θ[max(NMB(θ), 0)] − max(E_θ[NMB(θ)], 0)

Computed directly from the NMB samples. Report as $/woman and multiply by the relevant decision population (number of ANC women in Zimbabwe, or regionally) to get population EVPI.

### 6b. EVPPI (Expected Value of Partial Perfect Information)

Use the Strong et al. (2014) nonparametric regression method to estimate EVPPI for each parameter group without nested Monte Carlo. For each parameter group, fit a flexible regression of NMB on that group's parameters, then compute EVPPI from the fitted conditional expectations.

Parameter groups for EVPPI:

1. **Network structure** (calibrated): structuredsexual.prop_f0, structuredsexual.prop_m0, structuredsexual.m1_conc
2. **Ng transmission/symptoms** (calibrated): ng.beta_m2f, ng.p_symp
3. **Ct transmission/symptoms** (calibrated): ct.beta_m2f, ct.p_symp
4. **Tv transmission/symptoms** (calibrated): tv.beta_m2f, tv.p_symp
5. **RRs for preterm birth** (prior only): rr_ct_ptb, rr_ng_ptb, rr_tv_ptb
6. **RRs for LBW** (prior only): rr_ct_lbw, rr_ng_lbw, rr_tv_lbw
7. **RRs for stillbirth** (prior only): rr_ct_sb, rr_ng_sb, rr_tv_sb
8. **Reversibility parameters** (prior only): kappa_ptb, kappa_lbw, kappa_sb
9. **Partner treatment**: p_partner_tx
10. **Cost parameters** (prior only, PLACEHOLDER): all cost parameters jointly

### 6c. EVSI (Expected Value of Sample Information)

EVSI estimates the expected value of the PROMISE trial specifically, accounting for its finite sample size (N=12,780) and the specific quantities it will measure.

PROMISE will provide direct information on:
- STI prevalence in ANC women (both arms, both timepoints)
- The composite adverse birth outcome rate (both arms)
- The difference in adverse birth outcome rates (the treatment effect)
- Partner notification uptake and success

PROMISE will NOT directly resolve uncertainty about:
- The individual-STI-specific RRs (it measures the composite intervention effect, not the causal mechanism per STI)
- The reversibility parameters κ (these are entangled with the RRs in the trial's composite outcome)
- Long-term reinfection dynamics beyond the pregnancy period
- Generalizability to non-Harare settings
- Costs (unless the trial's cost-effectiveness component produces transferable unit cost estimates)

Use the regression-based EVSI method (Heath et al. 2020) to estimate EVSI. Prioritize EVPI and EVPPI for the first check-in; EVSI can be a placeholder with bounds (0 ≤ EVSI ≤ EVPI).

---

## Step 7: Figures and Reporting

### Figure 1: Prior predictive distribution of incremental NMB

- **Data**: Full set of NMB samples from Step 4
- **Format**: Histogram, colored green (NMB > 0, screening favored) and red (NMB < 0, SOC favored)
- **Variants**: Show at multiple WTP thresholds
- **Key message**: Quantifies current decision uncertainty

### Figure 2: Cost-effectiveness plane

- **Data**: (ΔDALYs, ΔCosts) pairs from Step 4
- **Format**: Scatter plot with WTP line; color by cost-effective (yes/no)
- **Key message**: Joint uncertainty in costs and effects

### Figure 3: EVPPI tornado chart

- **Data**: EVPPI estimates from Step 5b for each parameter group
- **Format**: Horizontal bar chart, sorted by EVPPI value
- **Key message**: Identifies which parameters drive the most decision uncertainty. Of particular interest: do the reversibility parameters (κ) and/or cost parameters dominate? If so, that motivates specific evidence-gathering priorities alongside the trial.

### Figure 4: Prior and posterior parameter distributions

- **Data**: Prior distributions (Steps 1 & 3) and posterior samples (Step 2)
- **Format**: Multi-panel. Epi parameters show prior (light) overlaid with calibrated posterior (dark). Birth outcome RRs, κ parameters, and cost parameters show prior only, clearly labeled.
- **Annotation**: Source for each parameter; cost panels flagged as placeholders
- **Key message**: Transparency about what we know and don't know

### Figure 5: Threshold analysis heatmap

- **Data**: Grid evaluation of NMB over 2 high-EVPPI parameters
- **Format**: 2D heatmap with decision boundary
- **Variants**: Multiple panels for different parameter pairs. Likely candidates include (κ_ptb vs. rr_ct_ptb), (Ct prevalence vs. cost_poc_test), or whichever pairs emerge from the EVPPI analysis.
- **Key message**: Shows the parameter combinations where the decision changes

### Figure 6: EVPI and EVSI curves

- **Data**: EVPI and EVSI as a function of WTP threshold
- **Format**: Line plot with uncertainty band around EVSI
- **Key message**: The value of the PROMISE trial's information, and the residual uncertainty after the trial

### Supplementary: Calibration diagnostics

- Model vs. data comparison for each calibration target
- Trace plots / convergence from Optuna
- Pairwise posterior correlations between epi parameters

---

## Progress and Status

### Completed

- [x] **Step 1**: Epidemiological parameters defined (15 calib_pars)
- [x] **Step 2 infrastructure**: `run_calibrations.py` working — dot-notation parameter routing, scalar `set_pars` fix in stisim, CSV column names updated to dot notation, weights dict updated
- [x] **Step 3 (plan)**: Prior parameters defined in this document (stillbirths dropped — see note below)
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

### Design decisions

- **Stillbirths dropped from scope**: The fetal health model focuses on PTB and LBW (and SGA). Stillbirth parameters (rr_*_sb, kappa_sb, cost_sb_mgmt, cost_neonatal_death) are excluded. This simplifies the model and the prior parameter space from ~25 to ~20 extra parameters.
- **PTB vs LBW double-counting**: PTB+LBW co-occurrences accrue PTB disability weight only. In practice, nearly all LBW is PTB-driven in this model, so DALYs ≈ n_ptb × dw_ptb × dur_ptb.
- **FetalHealth is an analyzer, not a module**: Added via `analyzers=` list so it runs at func_order ~97 (after pregnancy.step processes deliveries).

### Next steps

- [ ] **Smoke test end-to-end**: Run SOC and intervention scenarios with both DALY and cost analyzers attached, verify outputs make sense
- [ ] **Step 2**: Run full calibration (~500-2000 Optuna trials) on VMs
- [ ] **Step 3 (implementation)**: Wire up the ~20 prior parameters (RRs, κ values, costs) so they can be sampled and passed to analyzers/FetalHealth at runtime
- [ ] **Step 5**: Build `run_voi.py` — loads calibration posterior (top 200), samples 20 prior params per posterior draw, builds SOC + intervention sims with CRN, runs with both analyzers, computes INMB
- [ ] **Step 5 (cost-effectiveness)**: Implement INMB calculation from DALY and cost analyzer outputs
- [ ] **Step 6a**: EVPI from NMB samples
- [ ] **Step 6b**: EVPPI via nonparametric regression (Strong et al. 2014)
- [ ] **Step 6c**: EVSI (can be bounded by EVPI initially)
- [ ] **Step 7**: Figures 1-6 + supplementary calibration diagnostics
- [ ] Collect Zimbabwe-specific cost data to replace placeholders
- [ ] Get clinical input on κ (reversibility) framing

---

## Key Discussion Points for Check-in

1. **Are the calibration targets right?** The PROMISE team may have site-specific prevalence data from the Harare clinics.

2. **The κ (reversibility) framing — is it right?** The key modeling question is: when a pregnant woman's STI is treated, how much do her birth outcome probabilities recover? The κ parameter captures this, but clinical collaborators should weigh in on whether κ should vary by outcome (PTB vs. LBW vs. stillbirth), by STI, by gestational age at treatment, or some combination. The EVPPI analysis will tell us how much this matters for the decision.

3. **Cost data — what do we have?** All cost priors are placeholders. The PROMISE trial includes a cost-effectiveness component, so the study team may have budgeted costs or preliminary unit cost estimates. Even rough figures from the Harare clinics would be much better than the current priors. If costs dominate the EVPPI, that's a finding in itself.

4. **What WTP threshold is relevant for Zimbabwe?** WHO's old 1-3× GDP/capita gives ~$1,800-$5,400, but opportunity-cost-based estimates (Woods et al. 2016; Ochalek et al. 2018) suggest $50-$500/DALY for sub-Saharan African health systems. This matters enormously for the VoI framing.

5. **Population EVPI scaling**: Individual-level EVPI × decision population = total value of information. What's the right decision population — all ANC women in Zimbabwe (~400,000/year)? Regionally?

6. **What will PROMISE actually tell us?** The trial measures the composite intervention effect, not the individual causal pathways. Notably, the trial cannot separately identify the RRs and the κ parameters — it observes their joint effect. The VoI analysis can clarify whether this matters for the decision, or whether the composite effect is sufficient.

---

## Dependencies and Setup

The analysis requires STIsim and its dependencies (Starsim), Optuna for calibration, and standard scientific Python libraries (scipy, numpy, pandas, matplotlib). The `stisim_vddx_zim` example in the STIsim repository is the starting template.

For the EVPPI regression step, either pygam (for GAM-based estimation per Strong et al. 2014) or scikit-learn (gradient-boosted regression) can be used.



## Things to resolve
Birth outcomes:
- STIsim uses the Starsim pregnancy module. Within this module, when a woman becomes pregnant we draw her duration of pregnancy and set the time of delivery. 
- ONE modification to make is that if a woman who ALREADY has an becomes pregnant, we could use a modified distribution to determine her birth outcomes.
- ANOTHER modification is that if a woman gets pregnant and THEN gets an STI, we will need some way of modifying her outcomes. We could easily overwrite the ti_delivery and the fetal_health tracker but need to figure out what parameters we actually need for that, and keep them tractable. 
