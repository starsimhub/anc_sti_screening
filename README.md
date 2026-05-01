# ANC STI Screening — PROMISE VoI Analysis

Agent-based model for evaluating antenatal care (ANC) screening strategies for asymptomatic bacterial STIs during pregnancy in Zimbabwe, built on [STIsim](https://github.com/starsimhub/stisim) / [Starsim](https://github.com/starsimhub/starsim).

Pre-trial value of information (VoI) analysis for the PROMISE trial: quantifies decision uncertainty around cost-effectiveness of integrated ANC STI screening and identifies which parameters drive that uncertainty.

## Diseases modeled
- HIV (with ART, VMMC, PrEP)
- Gonorrhea (NG)
- Chlamydia (CT)
- Trichomoniasis (TV)
- Bacterial vaginosis (BV)

## Fetal health

The `FetalHealth` module (from Starsim) mechanistically tracks two damage pathways during pregnancy:

1. **Delivery timing shift → PTB**: STI infection shifts delivery earlier by a stochastic amount (pathogen-specific). Classified as preterm if <37 weeks GA.
2. **Growth restriction → LBW/SGA**: Each infection accumulates a fractional weight penalty. Birth weight = baseline_for_GA × percentile × (1 − restriction).

The `sti_fetal` connector routes new infections and treatments in pregnant women to FetalHealth. Treatment partially reverses damage, with reversibility depending on gestational age at treatment (early ≤24w vs late 32-34w).

## Repo structure

```
anc_sti_screening/
├── data/                        Zimbabwe demographic + epidemiological input data
├── results/                     Calibration outputs, VoI results
├── figures/                     Generated plots
├── assets/                      Fonts
│
├── model.py                     Sim construction (make_sim, make_diseases, make_hiv_intvs)
├── interventions.py             SyndromicMgmt, ANCScreen, STIPartnerNotification
├── analyzers.py                 total_symptomatic, pregnancy_sti_stats, birth_outcome_dalys, intervention_costs
├── connectors.py                sti_fetal connector: routes infections/treatments to FetalHealth
├── priors.py                    Prior distributions for VoI (epi, birth outcome, cost)
├── utils.py                     Plotting helpers, scenario definitions
│
├── run_calibrations.py          Optuna calibration (15 params, dot notation API)
├── run_msim.py                  Multi-sim with top 200 calibrated parameter sets
├── run_scenarios.py             Compare SOC vs ANC screening scenarios
├── run_voi.py                   Value of information analysis (EVPI, sim pairs with CRN)
│
├── plot_fig1_nmb.py             Fig 1: Prior predictive NMB distribution
├── plot_fig2_ceplane.py         Fig 2: Cost-effectiveness plane
├── plot_fig3_evppi.py           Fig 3: EVPPI tornado chart
├── plot_fig4_priors_posteriors.py  Fig 4: Prior and posterior parameter distributions
├── plot_hiv_calibration.py      Supplementary: HIV calibration validation
├── plot_sti_epi.py              Supplementary: STI prevalence by age/sex + time series
└── plot_network.py              Supplementary: Network structure
```

## Pipeline

```
1. run_calibrations.py        Calibrate HIV + NG/CT/TV to Zimbabwe data (2000 trials)
2. run_msim.py                Run top 200 pars → percentile stats for validation
3. Validation figures:
   - plot_hiv_calibration.py  HIV fit (2×3 grid)
   - plot_sti_epi.py          STI fit (prevalence by age/sex + time series)
   - plot_network.py          Network structure validation
4. run_voi.py                 VoI analysis: 200 draws × 2 sims (SOC + intervention, CRN)
                              → voi_draws.df, voi_evpi.df
5. VoI figures:
   - plot_fig1_nmb.py         NMB histograms at multiple WTP thresholds
   - plot_fig2_ceplane.py     ΔDALYs vs ΔCosts scatter
   - plot_fig3_evppi.py       EVPPI tornado (which parameters matter most)
   - plot_fig4_priors_posteriors.py  Prior/posterior distributions (8×4 grid)
```

## Scenarios

| Scenario | Screening | Description |
|----------|-----------|-------------|
| `soc` | Syndromic management only | Standard of care (baseline) |
| `enroll` | Single enrollment screen (≤24w GA) | Early screen only |
| `tri3` | Single third-trimester screen (≥28w GA) | Late screen only |
| `twice` | Both screens | PROMISE trial design |
| `partner_tx` | Both screens + partner notification | PROMISE + partner treatment |

## VoI parameter space (32 parameters)

- **9 calibrated epi parameters**: HIV transmission (3), network structure (3), STI transmission/symptoms (6)
- **4 delivery timing shifts**: pathogen-specific mean shifts + individual SD
- **3 growth penalties**: pathogen-specific fractional weight reduction
- **4 treatment reversibility**: growth and timing residuals, early vs late
- **8 cost parameters**: POC test, treatments, ANC visit, adverse outcome management (placeholders)

See `priors.py` for all distributions and `promise-voi-plan-2.md` for detailed rationale.

## Dependencies

- `stisim` (prep-uplift branch)
- `starsim` (fetal-health branch, >=3.2.0)
- `sciris` (>=3.1.6)
- `pandas`, `numpy`, `scipy`, `optuna`, `seaborn`, `scikit-learn`

## Usage

```python
from model import make_sim

# Single sim
sim = make_sim(scenario='twice', n_agents=10000, start=1990, stop=2040)
sim.run()

# VoI analysis
from run_voi import run_voi
draws_df, evpi_df = run_voi(n_draws=200)
```
