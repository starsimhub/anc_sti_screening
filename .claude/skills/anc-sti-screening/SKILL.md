---
name: anc-sti-screening
description: Use when working in the anc_sti_screening repo — the PROMISE trial pre-trial VoI analysis of ANC STI screening in Zimbabwe. Covers model structure, FetalHealth mechanism, calibration, VoI pipeline, scenarios, priors, analyzers, and gotchas.
---

# ANC STI Screening — PROMISE Trial VoI

Pre-trial value-of-information analysis evaluating asymptomatic bacterial STI screening (NG/CT/TV/BV) at ANC in Zimbabwe. Built on STIsim/Starsim. The key framing is not "is it cost-effective?" but "how confident are we, and what evidence would change that confidence?"

## Repo layout

| File | Purpose |
|------|---------|
| `model.py` | `make_sim()`, `make_diseases()`, `make_hiv_intvs()`, `make_stis()` |
| `interventions.py` | `SyndromicMgmt`, `ANCScreen`, `STIPartnerNotification`, `make_testing()` |
| `connectors.py` | `sti_fetal` — routes STI events to FetalHealth |
| `analyzers.py` | `total_symptomatic`, `pregnancy_sti_stats`, `birth_outcome_dalys`, `intervention_costs` |
| `priors.py` | All prior distributions (epi bounds + birth outcome + cost) |
| `run_calibrations.py` | Optuna calibration (15 epi params) |
| `run_msim.py` | Multi-sim validation with top 200 posteriors |
| `run_voi.py` | EVPI/EVPPI using CRN simulation pairs |
| `run_scenarios.py` | Per-scenario outcomes for all 5 scenarios |
| `plot_fig*.py` | Figure scripts (NMB, CE plane, EVPPI tornado, priors/posteriors) |

## Pipeline

```
run_calibrations.py  →  results/zimbabwe_pars.df (200 posteriors)
run_msim.py          →  msim validation
run_voi.py           →  results/voi_draws.df + voi_evpi.df
run_scenarios.py     →  per-scenario DALY/cost breakdowns
plot_fig*.py         →  figures/fig*.png
```

## Scenarios

| Name | Description |
|------|-------------|
| `soc` | Syndromic management only (no ANC screening) — VoI baseline |
| `enroll` | Single screen at enrollment (≤24w) |
| `tri3` | Single screen in third trimester (≥28w) |
| `twice` | Both screens (PROMISE protocol) — primary VoI comparator |
| `partner_tx` | Both screens + partner notification |

Diagnostic sensitivity fixed at 95/95 (POC); not in the VoI parameter space.

## Diseases and network

```python
ng  = sti.Gonorrhea(eff_condom=0.7)
ct  = sti.Chlamydia(eff_condom=0.8)
tv  = sti.Trichomoniasis(eff_condom=0.8)
bv  = sti.SimpleBV()
hiv = sti.HIV(beta_m2f=0.035, eff_condom=0.95, init_prev_data=...)
```

Network: `sti.StructuredSexual()` with Zimbabwe demographic data. HIV gets full intervention stack (FSW testing, GP testing, low-CD4 testing, ART, VMMC, PrEP).

## FetalHealth mechanism

FetalHealth (`ss.FetalHealth`, accessed via `sim.custom['fetal_health']`) holds each pregnant woman's delivery date and birth weight percentile. The `sti_fetal` connector modifies these via two pathways:

**1. Delivery timing shift → PTB**
Each new STI infection pulls delivery forward by a lognormal draw (mean = `ptb_shift_mean[disease]`, std = `ptb_shift_std`). Reinfection compounds. Treatment reverses `1 - tx_residual_timing[trimester]` of the remaining shift.

**2. Growth restriction → LBW/SGA**
Each infection reduces birth weight percentile by `growth_penalty[disease]`. Treatment recovers `1 - tx_residual_growth[trimester]`.

Trimester boundaries come from `preg.pars.trimesters` (~13w, ~26w).

## VoI parameter space

- **Calibrated epi (15 params)**: sampled from `zimbabwe_pars.df` posteriors — HIV β, condom eff, rel_init_prev; network prop_f0/m0/m1_conc; NG/CT/TV β and p_symp.
- **Birth outcome priors (13 params)**: `ptb_shift_pars` (4), `growth_penalty_pars` (3), `tx_reversibility_pars` (6).
- **Cost priors (8 params)**: gamma distributions — all still placeholders needing Zimbabwe-specific values.

All priors defined in `priors.py` as `scipy.stats` distributions. `sample_priors()` returns a flat dict of draws.

## Key patterns

**Building a sim**:
```python
from model import make_sim
sim = make_sim(scenario='twice', calib_pars=row, connector_pars=cpars)
sim.run()
```

**CRN for VoI**:
```python
seed = np.random.randint(0, 1e6)
soc_sim  = make_sim('soc',   seed=seed, ...)
intv_sim = make_sim('twice', seed=seed, ...)
# Run both; INMB = intv - soc
```

**Building connector pars** — always use `sc.objdict` for nested dicts:
```python
connector_pars = dict(
    ptb_shift_mean=sc.objdict(ng=draw['ptb_shift_ng'], ct=draw['ptb_shift_ct'], tv=draw['ptb_shift_tv']),
    ptb_shift_std=float(draw['ptb_shift_std']),
    growth_penalty=sc.objdict(ng=draw['growth_penalty_ng'], ...),
    tx_residual_growth=sc.objdict(tri1=draw['tx_res_g_t1'], tri2=draw['tx_res_g_t2'], tri3=draw['tx_res_g_t3']),
    tx_residual_timing=sc.objdict(tri1=draw['tx_res_t_t1'], ...),
)
```
**Never use plain `dict`** for nested connector pars — the connector accesses them via attribute (`.tri1`), so plain dicts raise `AttributeError`.

## Analyzers

- `birth_outcome_dalys` — DALYs from PTB and LBW/SGA, applied from `INTV_YEAR` onward
- `intervention_costs` — cost per screen, per treatment episode
- `pregnancy_sti_stats` — STI prevalence among pregnant women by trimester
- `total_symptomatic` — overall symptomatic prevalence stratified by sex and HIV status

## Standard imports

```python
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import stisim as sti
from starsim import FetalHealth
from connectors import sti_fetal
from interventions import make_testing
from analyzers import make_analyzers
from priors import sample_priors, calib_pars
```

## Running locally vs HPC

- Calibration (Optuna, N≥500 trials) and full VoI (N=200 draws × 2 sims) go on HPC.
- Local: test with `n_agents=500`, `n_draws=5`, short `dur`.
- Results committed to git if <10 MB; large calibration objects excluded.
