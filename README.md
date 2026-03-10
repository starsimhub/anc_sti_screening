# ANC STI Screening

Agent-based model for evaluating antenatal care (ANC) screening strategies for asymptomatic bacterial STIs during pregnancy in Zimbabwe, built on [STIsim](https://github.com/starsimhub/stisim) / [Starsim](https://github.com/starsimhub/starsim).

## Diseases modeled
- HIV
- Gonorrhea (NG)
- Chlamydia (CT)
- Trichomoniasis (TV)
- Bacterial vaginosis (BV)

## Repo structure

```
anc_sti_screening/
├── data/                      Zimbabwe demographic + epidemiological input data
├── results/                   Calibration outputs, scenario results
├── figures/                   Generated plots
├── assets/                    Fonts
├── model.py                   Sim construction (make_sim, make_diseases, make_hiv_intvs)
├── interventions.py           SyndromicMgmt (VDS/UDS) + ANCScreen intervention
├── analyzers.py               total_symptomatic, pregnancy_sti_stats
├── fetal_health.py            FetalHealth module: dynamic birth weight + PTB modeling
├── connectors.py              sti_fetal connector: routes infections/treatments to FetalHealth
├── utils.py                   Plotting helpers, scenario definitions
├── run_calibrations.py        Optuna calibration (v15 API, dot notation)
├── run_msim.py                Multi-sim with top calibrated parameter sets
├── run_scenarios.py           Compare ANC screening scenarios
├── plot_hiv_calibration.py    HIV calibration validation figure
├── plot_network.py            Network structure figure
└── plot_sti_epi.py            STI prevalence by age/sex + time series
```

## Pipeline

```
1. run_calibrations.py     Calibrate HIV + NG/CT/TV to Zimbabwe data (2000 trials)
2. run_msim.py             Run top 200 pars → percentile stats for validation
3. plot_hiv_calibration.py Validate HIV fit
   plot_sti_epi.py         Validate STI fit (prevalence by age/sex + time series)
   plot_network.py         Validate network structure
4. run_scenarios.py        Compare SOC vs ANC screening scenarios
                           FetalHealth module dynamically tracks birth outcomes
```

## Scenarios

| Scenario | Screening | Research question |
|----------|-----------|-------------------|
| `soc` | Syndromic management only | Baseline |
| `anc_all` | ANC screen for NG + CT + TV | Pathogen priority |
| `anc_ng_only` | ANC screen for NG only | Pathogen priority |
| `anc_ng_ct` | ANC screen for NG + CT | Pathogen priority |

## Dependencies

- `stisim` (v15 branch)
- `starsim` (>=3.2.0)
- `sciris` (>=3.1.6)
- `pandas`, `numpy`, `scipy`, `optuna`, `seaborn`

## Usage

```python
from model import make_sim

sim = make_sim(scenario='anc_all', n_agents=10000, start=1990, stop=2040)
sim.run()
```
