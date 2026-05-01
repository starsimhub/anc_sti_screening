---
name: anc-scenarios
description: Use when configuring, running, comparing, or interpreting ANC screening scenarios in the anc_sti_screening repo — including scenario definitions, make_sim() options, run_scenarios.py, result extraction, and cross-scenario comparisons.
context: fork
---

# ANC STI Screening — Scenarios

You are helping configure, run, and compare the five ANC screening scenarios for the PROMISE trial analysis.

## Scenario definitions

| Name | Description | Screening | Partner Tx |
|------|-------------|-----------|------------|
| `soc` | Standard of care — syndromic management only | None | No |
| `enroll` | Single ANC screen at enrollment (≤24w GA) | Once | No |
| `tri3` | Single ANC screen in third trimester (32-34w GA) | Once | No |
| `twice` | Both screens (PROMISE protocol) — **primary VoI comparator** | Both | No |
| `partner_tx` | Both screens plus partner notification and treatment | Both | Yes |

Scenario labels and ordering are defined in `utils.py`:
```python
from utils import scenarios, scenlabels
# scenarios = ['soc', 'enroll', 'tri3', 'twice', 'partner_tx']
```

## Building a sim

```python
from model import make_sim
sim = make_sim(scenario='twice', start=1990, stop=2040, n_agents=5000)
sim.run()
```

`make_sim()` assembles: diseases + connectors, HIV interventions, the scenario-appropriate STI interventions, and analyzers. Pass `calib_pars=row` (a Series from `zimbabwe_pars.df`) to apply a calibrated parameter set. Pass `connector_pars=cpars` (a dict with `sc.objdict` nested values) to override `sti_fetal` parameters.

## Scenario intervention logic (interventions.py)

**`SyndromicMgmt`** — runs in all scenarios. Treats symptomatic vaginal/urethral discharge. Outcome probabilities differ by sex and whether cervical infection is present (`tx_mix_cerv`, `tx_mix_noncerv`).

**`ANCScreen`** — added for `enroll`, `tri3`, `twice`, `partner_tx`:
- `enroll` screen: GA window ≤24w
- `tri3` screen: GA window 32-34w
- Tests for NG, CT, TV; treats positives with matched treatments
- Sensitivity 95%, specificity 95% (POC test; not in VoI parameter space)

**`STIPartnerNotification`** — added only for `partner_tx`. Notifies and treats partners of ANC-positive women.

## Running scenarios with calibrated parameters

```python
from run_scenarios import run_scenario, extract_results

sims = run_scenario(scenario='twice', n_pars=10, seeds_per_par=5)
df   = extract_results(sims, scenario='twice')
# df is long-format: columns = scenario, par_idx, year, metric, value
```

`run_scenario()` uses `sti.make_calib_sims()` internally: it applies each calibrated parameter set to a base sim and runs `seeds_per_par` replicates.

## Result metrics available in extract_results()

| Metric pattern | Description |
|----------------|-------------|
| `{dis}.new_infections` | Incidence (ng/ct/tv/bv) |
| `{dis}.prevalence` | Prevalence |
| `{dis}.new_treated` | New treatments |
| `anc_enroll.n_screened` | Women screened at enrollment |
| `anc_enroll.n_positive` | Positives detected at enrollment |
| `anc_enroll.n_{dis}_detected` | Per-disease positives at enrollment |
| `anc_tri3.*` | Same metrics for third-trimester screen |
| `partner_notif.n_index_cases` | Index cases triggering notification |
| `partner_notif.n_partners_found` | Partners reached |
| `partner_notif.n_partners_treated` | Partners treated |

VoI-specific metrics (DALYs, costs) come from `birth_outcome_dalys` and `intervention_costs` analyzers, accessible via `sim.results`.

## CRN simulation pairs (for VoI)

To compute incremental NMB correctly, run SOC and intervention with the same seed:
```python
seed = np.random.randint(0, 1_000_000)
soc_sim  = make_sim('soc',   seed=seed, calib_pars=row, connector_pars=cpars)
intv_sim = make_sim('twice', seed=seed, calib_pars=row, connector_pars=cpars)
sc.parallelize([soc_sim.run, intv_sim.run])  # or run sequentially
delta_dalys = intv_sim.results.birth_outcome_dalys.total - soc_sim.results.birth_outcome_dalys.total
delta_costs = intv_sim.results.intervention_costs.total  - soc_sim.results.intervention_costs.total
```
This CRN (common random numbers) structure ensures scenario differences aren't confounded by stochastic noise.

## Comparing scenarios

For a quick cross-scenario comparison (not full VoI):
```python
import sciris as sc
results = {}
for scen in scenarios:
    sims = run_scenario(scenario=scen, n_pars=5, seeds_per_par=3)
    results[scen] = extract_results(sims, scen)
combined = pd.concat(results.values())
```

Use `utils.percentile_pairs` for uncertainty bands in time-series plots.

## Compute scale / what to run locally

| Task | Local OK? | Recommended settings |
|------|-----------|---------------------|
| Single scenario test | Yes | `n_pars=2, seeds_per_par=2, n_agents=500` |
| Scenario comparison | Yes (small) | `n_pars=5, seeds_per_par=3, n_agents=1000` |
| Full scenario results | HPC | `n_pars=200, seeds_per_par=5, n_agents=5000` |
| VoI (N=200 draws) | HPC | See `run_voi.py` |

Always set single-threaded numpy before heavy runs (already in `run_scenarios.py`):
```python
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', ...)
```
