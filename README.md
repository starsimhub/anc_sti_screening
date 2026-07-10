# ANC STI Screening

Agent-based model for evaluating antenatal care (ANC) screening strategies for asymptomatic bacterial STIs during pregnancy in Zimbabwe, built on [STIsim](https://github.com/starsimhub/stisim) / [Starsim](https://github.com/starsimhub/starsim).

Calibration + intervention infrastructure is ported from the sibling repo `sti_notification`. This project uses the ported calibrated ensemble; it does NOT re-run the calibration itself. Current work is on branch `port/stinotif-calibration`.

**Read `HANDOFF.md` first** for current state, small-N findings, and the pending pivot to ABO/APO-first reporting.

## Diseases modeled
- HIV
- Syphilis (with GUD-placeholder for non-syphilitic genital ulcer)
- Gonorrhea (NG)
- Chlamydia (CT)
- Trichomoniasis (TV)
- Bacterial vaginosis (BV)

## Repo structure

```
anc_sti_screening/
├── data/                      Zimbabwe demographic + epi data + calibration_draws.csv
├── results/                   Scenario outputs (gitignored beyond a small kavg CSV)
├── figures/                   Generated plots
├── docs/                      Design spec + implementation plan (superpowers/)
├── model.py                   make_sim_parts / make_sim (Zimbabwe sim factory)
├── interventions.py           SyphilisANCTimer, make_syph_testing, make_testing,
│                              make_pn, CareSeekScaler, DxRiskRedux, ANCScreen
├── connectors.py              sti_fetal — routes STI events to FetalHealth
├── analyzers.py               birth_outcome_dalys, SyphTransmissionEvents,
│                              CareTimingAnalyzer
├── pn.py                      Edge-stratified PartnerNotification base class
├── hiv_model.py               HIV disease + interventions
├── apply_draw.py              row_to_sim_pars, set_pars_local for calibration draws
├── scenarios.py               INTERVENTION_SCENARIOS × EFFECT_SIZE_ASSUMPTIONS +
│                              build_scenario_sim factory
├── run_scenarios.py           Multiprocessing dispatch of the scenario grid
├── aggregate_scenarios.py     Assemble K-avg CSV + timeseries/snapshot parquets
├── quick_validate.py          Small-N smoke test with rich diagnostics
├── smoke_test_reproducibility.py  Verify sti_notification calibration reproduction
└── plot_*.py                  Figure scripts (adapted from sti_notification)
```

## Scenarios

7 intervention cells × 4 effect-size assumption sets (see `scenarios.py`):

| Scenario ID          | Screening                    |
| -------------------- | ---------------------------- |
| `soc`                | Syndromic + syph RPR only    |
| `anc_1screen_50cov`  | 1 ANC screen, 0-24w, 50% cov |
| `anc_1screen_75cov`  | 1 ANC screen, 0-24w, 75% cov |
| `anc_1screen_90cov`  | 1 ANC screen, 0-24w, 90% cov |
| `anc_2screen_50cov`  | + 3rd tri screen (30-36w), 50% cov |
| `anc_2screen_75cov`  | + 3rd tri screen (30-36w), 75% cov |
| `anc_2screen_90cov`  | + 3rd tri screen (30-36w), 90% cov |

All ANC screens use an NG+CT+TV panel at 95/95 sensitivity/specificity. Syph RPR-at-ANC is always on (SOC).

Effect-size assumption sets: `no_treatment_effect` (ratchet), `central_reversible`, `weak_effects` (lower CIs), `strong_effects` (upper CIs). See docstring at `scenarios.py::EFFECT_SIZE_ASSUMPTIONS`.

## Usage

Build a scenario sim directly:

```python
import pandas as pd
from scenarios import build_scenario_sim

df = pd.read_csv('data/calibration_draws.csv')
row = df.iloc[0].to_dict()
sim = build_scenario_sim(
    seed=int(row['draw_idx']) * 1000,
    scenario_id='anc_2screen_90cov',
    assumption_id='central_reversible',
    draw_row=row, start=1985, stop=2045, n_agents=10_000,
)
sim.run()
```

Run the full grid (5 draws × 5 seeds × 7 scenarios × 4 assumptions = 700 sims):

```bash
python run_scenarios.py    # writes results/scenarios.jsonl
python aggregate_scenarios.py    # writes results/scenarios.kavg.csv (+ parquets)
```

Fast smoke test:

```bash
python quick_validate.py    # 3 scenarios, K=1 seed, ~15 min
```

## Dependencies

- `stisim` on branch `rc1.5.9` (PR 505 baseline PN included)
- `starsim` (>=3.3.0)
- `sciris` (>=3.1.6)
- `pandas`, `numpy`, `scipy`
- Conda env: `starsim`
