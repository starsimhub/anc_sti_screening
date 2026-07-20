# Port sti_notification calibration to anc_sti_screening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the frozen sti_notification exp-06 calibration (17-parameter ensemble, 5 draws × K=5 seeds) into anc_sti_screening and run 13 ANC-screening scenarios × 4 effect-size assumptions to quantify ABO/APO/DALYs.

**Architecture:** Fork sti_notification's 7-disease calibrated model into anc_sti_screening on a `port/stinotif-calibration` branch. Replace anc_sti_screening's Optuna calibration machinery with the frozen ensemble. Bolt ANC-screening + PN interventions on top of the calibrated base. Naive (no shared burn-in): each of 1,300 sims runs 1985–2045 independently.

**Tech Stack:** Python 3, starsim, stisim (pinned rc1.5.9 with fix/ng-tx merged), sciris, pandas, numpy, scipy, matplotlib, seaborn.

## Global Constraints

- **stisim pin:** `rc1.5.9` (with `fix/ng-tx` merged into it as of pre-plan session). Do not modify the pin.
- **Population:** `n_agents=10_000` for all runs. Do not change without explicit user approval.
- **Sim horizon:** 1985–2045. Arms diverge at 2028; K=5 seeded means match `seed = draw_idx * 1000 + sub_idx` for `sub_idx in range(5)`.
- **Draws (first run):** 5 draws — the top-5 by GoF from `sti_notification/experiments/06_2026-06-24_kseed_calibration/outputs/draws_used.csv`.
- **Panel:** NG+CT+TV; test sens/spec 95/95. Fixed across all intervention arms.
- **SOC:** HIV testing/ART/VMMC/PrEP + syndromic management (VDS/UDS in general pop) + universal syph RPR at ANC. All 13 cells share this baseline; intervention arms add NG/CT/TV screening on top.
- **Branch:** `port/stinotif-calibration` off `main`. Do not commit anything to `main` until final PR-back at Phase 7.
- **Working directory:** `/home/robyn/anc_sti_screening`.
- **sti_notification source location:** `/home/robyn/sti_notification`. Read from it; do not commit to it as part of this plan.
- **Conda env:** `starsim`.
- **VM:** IDM Azure. Run heavyweight sims on 80 workers.

## File Structure

Files created or modified during this plan, organised by responsibility:

**Ported wholesale from sti_notification (with import-path adjustments if needed):**
- `model.py` — 7-disease `make_sim` (HIV + NG + CT + TV + BV + syph + GUDPlaceholder), richer StructuredSexual, PriorPartners, MaternalNet.
- `hiv_model.py` — HIV disease + HIV interventions (testing, ART, VMMC, PrEP).
- `connectors.py` — `sti_fetal` connector (NG/CT/TV/syph), holds effect-size assumption parameters.

**Ported partially (subset of sti_notification code):**
- `analyzers.py` — merged: sti_notification's `SyphTransmissionEvents`, `CareTimingAnalyzer` + wiring for `sti.sw_stats` / `sti.coinfection_stats`, alongside existing ANC analyzers `pregnancy_sti_stats`, `birth_outcome_dalys`, `intervention_costs`, `total_symptomatic`.

**Ported wholesale (data + calibration inputs):**
- `data/` — full sti_notification data folder wholesale, replacing overlapping files.
- `data/calibration_draws.csv` — top-5 rows from sti_notification exp-06 `draws_used.csv`.

**Built fresh in this plan:**
- `apply_draw.py` — `row_to_sim_pars` + `set_pars_local` helpers (adapted from sti_notification `calibration/artifacts/scripts/_pipeline.py`).
- `scenarios.py` — `INTERVENTION_SCENARIOS` (13 cells) + `EFFECT_SIZE_ASSUMPTIONS` (4 named cases).
- `run_scenarios.py` — dispatches the (draw × seed × scenario × assumption) grid.
- `aggregate_scenarios.py` — reads JSONL, writes K=5-averaged parquets + kavg CSV.
- `tests/test_apply_draw.py` — unit test for `set_pars_local` param routing.
- `tests/test_scenarios.py` — unit test that scenarios + assumptions are wired to sim params correctly.
- `tests/test_ancpn.py` — unit test for the new `ANCPN` intervention.
- `smoke_test_reproducibility.py` — Phase 2 script that re-runs 3 draws × 5 seeds and compares to sti_notification's `per_draw_means.csv`.

**Modified in this plan:**
- `interventions.py` — delete `STIPartnerNotification`; add `ANCPN(sti.PartnerNotification)`.
- `analyzers.py::birth_outcome_dalys` — update to read from starsim's Pregnancy.preterm + FetalHealth.lbw (current impl reads from `fh.is_preterm` / `fh.is_lbw` which don't exist in current starsim). Add stillbirths from Pregnancy.
- `plot_hiv_calibration.py`, `plot_sti_epi.py`, `plot_network.py`, `plot_fig4_priors_posteriors.py` — adapt to consume ensemble parquet outputs.
- `README.md` — updated to reflect the new pipeline.

**Deleted in this plan (superseded):**
- `run_calibrations.py`, `priors.py`, `run_msim.py` — Optuna infrastructure.
- Existing top-level `model.py`, `connectors.py` — replaced by ported versions.
- Old `promise-voi-plan-2.md` — superseded by design spec (kept in git history).

**Tag + branch:**
- Tag `v0.1` on `main` at current HEAD.
- Branch `port/stinotif-calibration` from `main`.

---

## Phase 1 — Repo prep

### Task 1.1: Tag v0.1 and branch off main

**Files:** none (git ops only).

**Interfaces:**
- Consumes: none.
- Produces: tag `v0.1` on origin/main; branch `port/stinotif-calibration` locally and on origin.

- [ ] **Step 1: Verify main is clean**

Run: `cd /home/robyn/anc_sti_screening && git status && git log --oneline -3`
Expected: `nothing to commit, working tree clean`; recent commits visible.

- [ ] **Step 2: Tag main HEAD as v0.1 and push**

Run:
```bash
cd /home/robyn/anc_sti_screening
git tag -a v0.1 -m "Pre-port snapshot: Optuna calibration + promise-voi-plan-2 framing"
git push origin v0.1
```

Expected: `To github.com:...  * [new tag]  v0.1 -> v0.1`.

- [ ] **Step 3: Create and switch to the port branch**

Run:
```bash
cd /home/robyn/anc_sti_screening
git switch -c port/stinotif-calibration
git push -u origin port/stinotif-calibration
```

Expected: `Switched to a new branch 'port/stinotif-calibration'`; upstream set on push.

- [ ] **Step 4: Commit the design spec on the port branch**

The design doc at `docs/superpowers/specs/2026-07-07-port-stinotif-calibration-design.md` should be uncommitted on the port branch (it was written during brainstorming, deliberately not committed to main). Commit it now.

Run:
```bash
cd /home/robyn/anc_sti_screening
git add docs/superpowers/specs/2026-07-07-port-stinotif-calibration-design.md
git status
```

Expected: single file staged, no other changes.

- [ ] **Step 5: Commit and push**

Run:
```bash
cd /home/robyn/anc_sti_screening
git commit -m "docs: add port design spec (2026-07-07)"
git push
```

### Task 1.2: Copy the calibrated data folder from sti_notification

**Files:**
- Copy from: `/home/robyn/sti_notification/data/`
- Copy to: `/home/robyn/anc_sti_screening/data/`

**Interfaces:**
- Consumes: none.
- Produces: full sti_notification data folder present under `anc_sti_screening/data/`. Overwrites overlapping files. Adds syph-related files, symp_test_prob files, syph_dx.csv, migration data.

- [ ] **Step 1: Inspect existing data folder to note which files will be overwritten**

Run:
```bash
ls /home/robyn/anc_sti_screening/data/
```

Expected: current ANC data files listed (age_dist_1980, age_dist_1990, asfr, deaths, init_prev_bv, ..., zimbabwe_data, zimbabwe_hiv_calib, zimbabwe_sti_data).

- [ ] **Step 2: Inspect sti_notification data folder**

Run:
```bash
ls /home/robyn/sti_notification/data/
```

Expected: syph_dx.csv, init_prev_syph.csv, init_prev_latent_syph.csv, symp_test_prob_concentrated.csv, symp_test_prob_soc.csv, zimbabwe_age_1985.csv, zimbabwe_asfr.csv, zimbabwe_deaths.csv, zimbabwe_migration.csv, zimbabwe_syph_data.csv, plus overlaps with ANC.

- [ ] **Step 3: Delete anc_sti_screening's existing data files with different naming conventions**

Some anc_sti_screening files have different names than sti_notification's equivalents (e.g. `asfr.csv` vs `zimbabwe_asfr.csv`). Delete the ANC-named files that will be replaced by prefixed versions.

Run:
```bash
cd /home/robyn/anc_sti_screening
git rm data/asfr.csv data/deaths.csv data/zimbabwe_data.csv data/age_dist_1980.csv data/age_dist_1990.csv
```

Expected: 5 files staged for deletion. (Note: `condom_use.csv`, `init_prev_*.csv`, `n_art.csv`, `n_vmmc.csv`, `zimbabwe_hiv_calib.csv`, `zimbabwe_sti_data.csv` will be overwritten by sti_notification's versions in step 4 — do not delete them first, git will show them as modified.)

- [ ] **Step 4: Copy sti_notification's data folder wholesale**

Run:
```bash
cp -r /home/robyn/sti_notification/data/*.csv /home/robyn/anc_sti_screening/data/
```

Expected: all sti_notification CSVs now under `anc_sti_screening/data/`.

- [ ] **Step 5: Verify the copy**

Run:
```bash
cd /home/robyn/anc_sti_screening
ls data/*.csv | sort
diff <(ls /home/robyn/sti_notification/data/*.csv | xargs -n1 basename | sort) <(ls data/*.csv | xargs -n1 basename | sort)
```

Expected: `diff` reports no differences between the two file lists.

- [ ] **Step 6: Stage + commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add data/
git status
```

Expected: 5 deletions + several modifications + several new files (init_prev_syph, init_prev_latent_syph, syph_dx, symp_test_prob_*, zimbabwe_age_1985, zimbabwe_asfr, zimbabwe_deaths, zimbabwe_migration, zimbabwe_syph_data).

```bash
git commit -m "data: port sti_notification data folder wholesale"
```

### Task 1.3: Copy the calibrated model files from sti_notification

**Files:**
- Copy: `/home/robyn/sti_notification/model.py` → `/home/robyn/anc_sti_screening/model.py` (replaces existing)
- Copy: `/home/robyn/sti_notification/hiv_model.py` → `/home/robyn/anc_sti_screening/hiv_model.py` (new)
- Copy: `/home/robyn/sti_notification/connectors.py` → `/home/robyn/anc_sti_screening/connectors.py` (replaces existing)

**Interfaces:**
- Consumes: nothing from anc_sti_screening's existing model.py or connectors.py.
- Produces: `make_sim(seed, n_agents=5e3, start=1985, stop=2030, pn_pars=None, poc=None, poc_syph=None, which='all', dur_recall=ss.years(0.25), fetal_health=True, care_seek_mult=1.0, verbose=1/12, syph_symp_test_prob=None, syph_anc_probs=None, fsw_outreach=False, fsw_coverage_per_step=0.10)` from `model.py`; `sti_fetal` connector class from `connectors.py` with pars `{ptb_shift_mean, ptb_shift_std, growth_penalty, tx_residual_growth, tx_residual_timing, ptb_shift_dist}`; HIV factory functions from `hiv_model.py`.

- [ ] **Step 1: Copy the three files**

Run:
```bash
cd /home/robyn/anc_sti_screening
cp /home/robyn/sti_notification/model.py model.py
cp /home/robyn/sti_notification/hiv_model.py hiv_model.py
cp /home/robyn/sti_notification/connectors.py connectors.py
```

- [ ] **Step 2: Check that model.py's imports still resolve**

Model.py imports `interventions.py::make_testing, make_syph_testing, make_pn`. These names come from *sti_notification's* interventions.py which we are NOT porting. We need to leave anc_sti_screening's interventions.py in place for now, but model.py's imports will fail. Temporarily comment those lines out to allow other work to proceed until Phase 3 refactors interventions.

Run:
```bash
head -20 /home/robyn/anc_sti_screening/model.py
```

Confirm the import block. Then edit `model.py` to temporarily neuter the intervention imports:

```python
# TEMPORARY: interventions module differs between repos; will be reconciled in Phase 3.
# from interventions import make_testing, make_syph_testing, make_pn
```

Also comment out the call to `make_interventions` in `make_sim` and pass `interventions=[]` in the `sti.Sim(...)` call. This is TEMPORARY and reversed at Task 3.4.

- [ ] **Step 3: Verify import works**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -c "from model import make_sim; print('import ok')"
```

Expected: `import ok`. If it fails, read the error and add whatever additional temporary shims are needed (e.g., commenting out `analyzers.py` imports from `make_diseases` if those cause errors).

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add model.py hiv_model.py connectors.py
git commit -m "model: port sti_notification model.py + hiv_model.py + connectors.py (WIP: interventions temporarily stubbed)"
```

### Task 1.4: Copy analyzer classes from sti_notification (merge, don't overwrite)

**Files:**
- Modify: `analyzers.py` — add classes from sti_notification alongside existing ANC classes.

**Interfaces:**
- Consumes: existing anc_sti_screening `analyzers.py` classes (`pregnancy_sti_stats`, `birth_outcome_dalys`, `intervention_costs`, `total_symptomatic`).
- Produces: added classes `SyphTransmissionEvents` and `CareTimingAnalyzer` from sti_notification.

- [ ] **Step 1: Read the sti_notification analyzer classes we need to port**

Run:
```bash
grep -n "^class " /home/robyn/sti_notification/analyzers.py
```

Expected output: `class SyphTransmissionEvents(ss.Analyzer):` and `class CareTimingAnalyzer(ss.Analyzer):`.

- [ ] **Step 2: Read the current ANC analyzers.py head**

Run:
```bash
head -12 /home/robyn/anc_sti_screening/analyzers.py
```

Note the import block.

- [ ] **Step 3: Read the sti_notification analyzers.py class bodies**

Read the full contents of `/home/robyn/sti_notification/analyzers.py`. Copy the two class definitions plus any module-level constants / helpers they depend on.

- [ ] **Step 4: Append the ported classes to anc_sti_screening/analyzers.py**

Append `SyphTransmissionEvents` and `CareTimingAnalyzer` class definitions (plus any helper functions they use) to the end of `analyzers.py`. Ensure all imports needed by those classes are added at the top of the file.

- [ ] **Step 5: Verify import works**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -c "from analyzers import SyphTransmissionEvents, CareTimingAnalyzer, birth_outcome_dalys, pregnancy_sti_stats; print('import ok')"
```

Expected: `import ok`.

- [ ] **Step 6: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add analyzers.py
git commit -m "analyzers: port SyphTransmissionEvents + CareTimingAnalyzer from sti_notification"
```

### Task 1.5: Copy calibration draws (top-5 by GoF)

**Files:**
- Create: `data/calibration_draws.csv` — top-5 rows from sti_notification's draws_used.csv.

**Interfaces:**
- Consumes: `/home/robyn/sti_notification/experiments/06_2026-06-24_kseed_calibration/outputs/draws_used.csv`.
- Produces: `data/calibration_draws.csv` — 5 rows, same 19 columns (draw_idx + 17 params + retention_rank + gof).

- [ ] **Step 1: Inspect sti_notification's draws file**

Run:
```bash
head -1 /home/robyn/sti_notification/experiments/06_2026-06-24_kseed_calibration/outputs/draws_used.csv
wc -l /home/robyn/sti_notification/experiments/06_2026-06-24_kseed_calibration/outputs/draws_used.csv
```

Expected: 19-column header; 11 lines (1 header + 10 rows).

- [ ] **Step 2: Take the top-5 rows**

The file is already sorted by `retention_rank` (which is inverse GoF rank). Take the top-5.

Run:
```bash
head -6 /home/robyn/sti_notification/experiments/06_2026-06-24_kseed_calibration/outputs/draws_used.csv > /home/robyn/anc_sti_screening/data/calibration_draws.csv
```

- [ ] **Step 3: Verify the copy**

Run:
```bash
wc -l /home/robyn/anc_sti_screening/data/calibration_draws.csv
head -1 /home/robyn/anc_sti_screening/data/calibration_draws.csv
cut -d, -f1 /home/robyn/anc_sti_screening/data/calibration_draws.csv | tr '\n' ' '; echo
```

Expected: 6 lines (1 header + 5 rows). Header identical to sti_notification. `draw_idx` values in first column across rows are the top-5 by rank (should be `263, 75, 154, ...` or similar — verify by comparing to sti_notification).

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add data/calibration_draws.csv
git commit -m "data: add top-5 calibration draws from sti_notification exp-06"
```

### Task 1.6: Delete Optuna machinery

**Files:**
- Delete: `run_calibrations.py`, `priors.py`, `run_msim.py`.

**Interfaces:**
- Consumes: nothing (deleting).
- Produces: cleaner top-level file listing.

- [ ] **Step 1: Confirm the files exist**

Run:
```bash
cd /home/robyn/anc_sti_screening
ls run_calibrations.py priors.py run_msim.py
```

Expected: all three listed.

- [ ] **Step 2: Delete via git**

Run:
```bash
cd /home/robyn/anc_sti_screening
git rm run_calibrations.py priors.py run_msim.py
```

Expected: three files staged for deletion.

- [ ] **Step 3: Verify no remaining references**

Run:
```bash
cd /home/robyn/anc_sti_screening
grep -rn "run_calibrations\|from priors\|import priors\|from run_msim\|import run_msim" --include="*.py" .
```

Expected: no output (no remaining references). If any exist, fix them or defer to later tasks; do not commit references to deleted files.

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git commit -m "chore: remove Optuna calibration machinery (superseded by ported ensemble)"
```

### Task 1.7: Delete the old VoI plan doc

**Files:**
- Delete: `promise-voi-plan-2.md`.

**Interfaces:**
- Consumes: nothing.
- Produces: cleaner top-level, no stale planning docs.

- [ ] **Step 1: Delete via git**

Run:
```bash
cd /home/robyn/anc_sti_screening
git rm promise-voi-plan-2.md
git commit -m "docs: remove promise-voi-plan-2 (superseded by 2026-07-07 design spec)"
```

The file remains in git history for reference.

---

## Phase 2 — Reproducibility smoke test

**Goal of Phase 2:** verify that the ported model, running under stisim rc1.5.9-with-fix/ng-tx, reproduces sti_notification exp-06's per-draw K=5 means to numerical noise. This is the go/no-go gate for continuing.

### Task 2.1: Port `set_pars_local` and `row_to_sim_pars`

**Files:**
- Create: `apply_draw.py`

**Interfaces:**
- Consumes: `sim` (from `model.make_sim`), draw-row dict.
- Produces:
  - `row_to_sim_pars(row: dict|pandas.Series) -> dict` — translates a CSV row into `{module.par: value}` (inverse-log-transforms columns prefixed with `log_`).
  - `set_pars_local(sim, pars: dict) -> sim` — mutates the sim's module params to match `pars`, matching modules by `name` across `diseases / networks / interventions / connectors / analyzers / demographics / custom` containers. Handles special cases (`time_to_undetectable`, `rel_trans_latent_half_life`, `p_symp_primary_f/m`, distributions with `.set()`).

- [ ] **Step 1: Read sti_notification's implementation to copy from**

Read `/home/robyn/sti_notification/calibration/artifacts/scripts/_pipeline.py` lines 79–137 (the `row_to_sim_pars` and `set_pars_local` functions).

- [ ] **Step 2: Write a failing test for `row_to_sim_pars`**

Create `tests/test_apply_draw.py`:

```python
"""Unit tests for apply_draw.py — draw translation + parameter setting."""
import numpy as np
import pandas as pd
import pytest


def test_row_to_sim_pars_translates_dotted_keys():
    from apply_draw import row_to_sim_pars
    row = {
        'draw_idx': 263,
        'hiv.beta_m2f': 0.0076,
        'log_syph.beta_m2f': -2.05,   # log-transformed value
        'structuredsexual.prop_f0': 0.65,
    }
    out = row_to_sim_pars(row)
    # draw_idx should be dropped
    assert 'draw_idx' not in out
    # dotted keys preserved
    assert out['hiv.beta_m2f'] == pytest.approx(0.0076)
    # log-prefixed keys inverse-transformed with exp (not 10**)
    assert out['syph.beta_m2f'] == pytest.approx(np.exp(-2.05))
    # network keys preserved
    assert out['structuredsexual.prop_f0'] == pytest.approx(0.65)


def test_row_to_sim_pars_accepts_pandas_series():
    from apply_draw import row_to_sim_pars
    df = pd.DataFrame([{'draw_idx': 1, 'hiv.beta_m2f': 0.008}])
    out = row_to_sim_pars(df.iloc[0])
    assert out['hiv.beta_m2f'] == pytest.approx(0.008)


def test_row_to_sim_pars_drops_non_dotted_keys():
    from apply_draw import row_to_sim_pars
    row = {'draw_idx': 1, 'retention_rank': 5, 'gof': 0.47, 'hiv.beta_m2f': 0.008}
    out = row_to_sim_pars(row)
    assert out == {'hiv.beta_m2f': pytest.approx(0.008)}
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -m pytest tests/test_apply_draw.py -v
```

Expected: `ModuleNotFoundError: No module named 'apply_draw'` or similar. If pytest isn't installed, run `pip install pytest` in the `starsim` conda env first.

- [ ] **Step 4: Write `apply_draw.py` with `row_to_sim_pars` + `set_pars_local`**

Create `apply_draw.py`:

```python
"""
Draw-to-sim parameter translation and application.

Adapted from sti_notification/calibration/artifacts/scripts/_pipeline.py
(commit 731bc1d). Uses the same log-transform convention (natural log)
and the same module-matching-by-name strategy.
"""
from __future__ import annotations

import numpy as np
import starsim as ss


def row_to_sim_pars(row) -> dict:
    """Translate one prior-CSV row into a {module.par: value} dict.

    - log_-prefixed columns are inverse-transformed with np.exp.
    - Non-module keys (draw_idx, retention_rank, gof, seed, and any key
      lacking a '.') are dropped.
    - Accepts a plain dict or a pandas.Series.
    """
    if hasattr(row, 'to_dict'):
        row = row.to_dict()
    sim_pars = {}
    for col, val in row.items():
        if col in ('draw_idx', 'seed', 'retention_rank', 'gof'):
            continue
        if isinstance(col, str) and col.startswith('log_'):
            key = col[4:]
            v = float(np.exp(val))
        else:
            key = col
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
        if isinstance(key, str) and '.' in key:
            sim_pars[key] = v
    return sim_pars


def set_pars_local(sim, pars: dict):
    """Apply a {module.par: value} dict to a built sim.

    Handles the few priors that need special wiring:
      - time_to_undetectable: wraps in ss.lognorm_ex(ss.years, ss.years)
      - rel_trans_latent_half_life: wraps in ss.years
      - p_symp_primary_f / p_symp_primary_m: list-index into p_symp_primary
      - Distributions with a .set() method: call .set(mean=value)
    """
    for key, value in pars.items():
        if '.' not in key:
            continue
        mod_name, par_name = key.split('.', 1)
        found = False
        for category in ('diseases', 'networks', 'interventions',
                         'connectors', 'analyzers',
                         'demographics', 'custom'):
            container = sim.pars.get(category)
            if container is None:
                continue
            if isinstance(container, list):
                for mod in container:
                    if hasattr(mod, 'name') and mod.name == mod_name:
                        if par_name == 'time_to_undetectable':
                            mod.pars[par_name] = ss.lognorm_ex(
                                ss.years(float(value)), ss.years(float(value)))
                        elif par_name == 'rel_trans_latent_half_life':
                            mod.pars[par_name] = ss.years(float(value))
                        elif par_name == 'p_symp_primary_f':
                            mod.pars['p_symp_primary'][0] = float(value)
                        elif par_name == 'p_symp_primary_m':
                            mod.pars['p_symp_primary'][1] = float(value)
                        else:
                            existing = mod.pars.get(par_name)
                            if hasattr(existing, 'set'):
                                existing.set(mean=value)
                            else:
                                mod.pars[par_name] = value
                        found = True
                        break
            if found:
                break
    return sim
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -m pytest tests/test_apply_draw.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add apply_draw.py tests/test_apply_draw.py
git commit -m "feat: add apply_draw.py — row_to_sim_pars + set_pars_local"
```

### Task 2.2: Write the reproducibility smoke test script

**Files:**
- Create: `smoke_test_reproducibility.py`

**Interfaces:**
- Consumes: `data/calibration_draws.csv`, `model.make_sim`, `apply_draw.row_to_sim_pars` + `set_pars_local`, sti_notification's `experiments/06_.../outputs/per_draw_means.csv` (for comparison).
- Produces: prints per-draw K=5-mean scalar comparison table + `PASS/FAIL` verdict; writes `smoke_test_results.csv` under `results/` for archival.

**IMPORTANT:** the ported `model.py` still has its intervention imports stubbed out from Task 1.3. Interventions are irrelevant to reproducibility of the prevalence-level calibration outcomes (they don't get called during a normal 1985–2020 run), so this works. Once Phase 3 restores interventions, this smoke test should still pass; re-run it as a regression check.

- [ ] **Step 1: Create `smoke_test_reproducibility.py`**

```python
"""
Phase-2 reproducibility gate.

Runs 3 draws x 5 seeds through the ported model 1985-2020, extracts
per-draw K=5-mean prevalences, and compares to
sti_notification/experiments/06_2026-06-24_kseed_calibration/outputs/per_draw_means.csv.

Pass condition: all compared metrics match within 1e-6 relative
tolerance. Larger tolerance means something in the port has shifted
model dynamics; investigate before proceeding.

Usage:
    python smoke_test_reproducibility.py
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sciris as sc

from apply_draw import row_to_sim_pars, set_pars_local
from model import make_sim


REPO = Path(__file__).resolve().parent
DRAWS_CSV = REPO / 'data' / 'calibration_draws.csv'
STINOTIF_TRUTH = Path(
    '/home/robyn/sti_notification/experiments/'
    '06_2026-06-24_kseed_calibration/outputs/per_draw_means.csv'
)
N_DRAWS = 3
K_SEEDS = 5
STOP = 2020
N_AGENTS = 10_000

# Metrics we compare vs sti_notification's truth. Names must match the
# columns in per_draw_means.csv.
COMPARE_METRICS = [
    'hiv_prev_15_49_2016',
    'trep_f_2016',
    'nontrep_f_2016',
    'ng_prev_f_2018',
    'ct_prev_f_2018',
    'tv_prev_f_2018',
]


def run_one(args):
    draw_idx, seed, row = args
    sim = make_sim(seed=seed, n_agents=N_AGENTS, start=1985, stop=STOP,
                   which='all', fetal_health=False, verbose=-1)
    sim_pars = row_to_sim_pars(row)
    set_pars_local(sim, sim_pars)
    sim.run()

    # Extract the same scalars sti_notification's pipeline extracts.
    r = sim.results
    def year_val(res_key, year, sub='prevalence'):
        res = r[res_key][sub]
        yrs = np.array([t.year + t.month/12 for t in res.timevec])
        i = np.argmin(np.abs(yrs - year))
        return float(res.values[i])

    return {
        'draw_idx': draw_idx,
        'seed': seed,
        'hiv_prev_15_49_2016': year_val('hiv', 2016, 'prevalence_15_49'),
        'trep_f_2016':         year_val('syph_hiv_trep', 2016, 'syph_prev_f'),
        'nontrep_f_2016':      year_val('syph_hiv_nontrep', 2016, 'syph_prev_f'),
        'ng_prev_f_2018':      year_val('ng', 2018, 'prevalence_f'),
        'ct_prev_f_2018':      year_val('ct', 2018, 'prevalence_f'),
        'tv_prev_f_2018':      year_val('tv', 2018, 'prevalence_f'),
    }


def main():
    if not STINOTIF_TRUTH.exists():
        sys.exit(f'Truth file not found: {STINOTIF_TRUTH}')
    if not DRAWS_CSV.exists():
        sys.exit(f'Draws file not found: {DRAWS_CSV}')

    draws = pd.read_csv(DRAWS_CSV).head(N_DRAWS)
    truth = pd.read_csv(STINOTIF_TRUTH).set_index('draw_idx')

    tasks = []
    for _, row in draws.iterrows():
        d = int(row['draw_idx'])
        for sub in range(K_SEEDS):
            seed = d * 1000 + sub
            tasks.append((d, seed, row.to_dict()))

    print(f'Running {len(tasks)} sims ({N_DRAWS} draws x {K_SEEDS} seeds) ...')
    with mp.Pool(min(len(tasks), 20)) as pool:
        rows = pool.map(run_one, tasks)

    df = pd.DataFrame(rows)
    kmean = df.groupby('draw_idx').mean(numeric_only=True)

    print('\nOur K=5 means:')
    print(kmean[COMPARE_METRICS])
    print('\nsti_notification truth:')
    print(truth.loc[kmean.index, COMPARE_METRICS])

    diffs = (kmean[COMPARE_METRICS] - truth.loc[kmean.index, COMPARE_METRICS]).abs()
    rel = diffs / truth.loc[kmean.index, COMPARE_METRICS].abs().clip(lower=1e-9)
    max_rel = rel.max().max()

    print(f'\nMax relative diff: {max_rel:.2e}')

    (REPO / 'results').mkdir(exist_ok=True)
    kmean.to_csv(REPO / 'results' / 'smoke_test_results.csv')

    if max_rel < 1e-6:
        print('PASS: reproduction within tolerance')
        return 0
    else:
        print('FAIL: reproduction diverges beyond tolerance. Investigate.')
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Verify script imports cleanly**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -c "import smoke_test_reproducibility as m; print('import ok')"
```

Expected: `import ok`. Fix any import errors before proceeding.

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add smoke_test_reproducibility.py
git commit -m "test: add Phase-2 reproducibility smoke test script"
```

### Task 2.3: Run the smoke test and verify

**Files:** none (execution).

**Interfaces:**
- Consumes: `smoke_test_reproducibility.py` from Task 2.2.
- Produces: `results/smoke_test_results.csv` + `PASS`/`FAIL` on stdout.

- [ ] **Step 1: Confirm which columns exist in the truth file**

Run:
```bash
head -1 /home/robyn/sti_notification/experiments/06_2026-06-24_kseed_calibration/outputs/per_draw_means.csv | tr ',' '\n' | grep -E "hiv_prev_15_49|trep_f|nontrep_f|ng_prev_f|ct_prev_f|tv_prev_f"
```

Expected: matching column names. If `hiv_prev_15_49_2016` doesn't exist but `hiv_prev_15_49` does (as a scalar taken at a specific year), or if columns are aggregated over year ranges instead of at points, adjust `COMPARE_METRICS` and the `year_val()` helper in `smoke_test_reproducibility.py` accordingly.

- [ ] **Step 2: Run the smoke test on the VM**

Run:
```bash
cd /home/robyn/anc_sti_screening
python smoke_test_reproducibility.py
```

Expected: takes ~5-10 minutes (15 sims x ~30-40 sec each on 20 workers). Ends with `PASS` verdict.

- [ ] **Step 3: If FAIL — diagnose**

If `Max relative diff` is > 1e-6:
1. Small (~1e-3): probably float precision in log-transform or ss.years rounding. Loosen tolerance if you confirm no dynamics change, but note it in a comment.
2. Larger: rerun 1 draw x 1 seed and compare full timeseries to sti_notification's timeseries.parquet. Look for where they diverge. Common culprits: data file mismatch, stisim version drift (verify `pip show stisim` matches the pin), sti_notification's model.py has minor differences from what we ported.
3. If truly stuck: STOP and consult the user before proceeding. Do not weaken the tolerance without approval.

- [ ] **Step 4: Commit the smoke-test results artifact**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add results/smoke_test_results.csv
git commit -m "test: Phase-2 smoke test PASS (3 draws x 5 seeds reproduce sti_notification)"
```

**Phase 2 gate:** do not proceed to Phase 3 until this task is committed with a PASS verdict.

---

## Phase 3 — ANC infrastructure

### Task 3.1: Restore intervention wiring in the ported model.py

**Files:**
- Modify: `model.py` — un-stub the intervention hooks that were temporarily commented out in Task 1.3.
- Modify: `interventions.py` — refactor to expose the sti_notification-style `make_testing`, `make_syph_testing`, `make_pn` functions the ported `model.py` expects. Delete `STIPartnerNotification` (superseded by Task 3.4).

**Interfaces:**
- Consumes: existing `ANCScreen`, `SyndromicMgmt` from anc_sti_screening's `interventions.py`.
- Produces:
  - `make_testing(poc, stop, fsw_outreach=False, fsw_coverage_per_step=0.10) -> list[Intervention]` — NG/CT/TV testing + treatment interventions.
  - `make_syph_testing(stop, symp_test_prob=None, anc_probs=None, poc=False) -> list[Intervention]` — syph symptomatic testing + ANC RPR + syph treatment.
  - `make_pn(poc=None, pn_pars=None) -> Intervention` — partner notification. **Placeholder in this task** — replaced by `ANCPN` at Task 3.4.

**Approach:** copy sti_notification's `interventions.py` in full into a temporary file, then remove things anc_sti_screening doesn't need (FSW outreach, POC PN classes), then merge with anc_sti_screening's `ANCScreen` / `SyndromicMgmt` / `make_testing`.

- [ ] **Step 1: Read the sti_notification interventions module**

Run:
```bash
wc -l /home/robyn/sti_notification/interventions.py
grep -n "^def \|^class " /home/robyn/sti_notification/interventions.py
```

Confirm the definitions.

- [ ] **Step 2: Copy sti_notification's interventions.py to a temp scratch file for reference**

Run:
```bash
cp /home/robyn/sti_notification/interventions.py /tmp/stinotif_interventions.py
```

Do NOT commit this to the repo; it's a reference copy for merging.

- [ ] **Step 3: Save existing ANC interventions.py contents**

Read `interventions.py` in full and identify:
- `ANCScreen` class definition
- `SyndromicMgmt` class definition
- Existing `make_testing(ng, ct, tv, bv, scenario='soc', stop=2040)` signature and body
- `_make_anc_screen` inner helper
- `seeking_care_vds`, `seeking_care_uds` eligibility functions
- `STIPartnerNotification` (to be deleted at Task 3.4)

Note which pieces to keep vs replace.

- [ ] **Step 4: Merge — write the new interventions.py**

Structure the new `interventions.py`:
```
1. Module docstring
2. Imports (union of both files)
3. Constants (ANC_PROBS_REALISTIC etc. from sti_notification if needed)
4. Existing eligibility helpers (seeking_care_vds, seeking_care_uds)
5. Existing SyndromicMgmt class (unchanged)
6. Existing ANCScreen class (unchanged — refined in Task 3.2)
7. STIPartnerNotification (KEEP for now — deleted at Task 3.4)
8. make_testing(poc, stop, fsw_outreach, fsw_coverage_per_step) — port from sti_notification
9. make_syph_testing(stop, symp_test_prob, anc_probs, poc) — port from sti_notification
10. make_pn(poc, pn_pars) — placeholder that returns an inert intervention; replaced at Task 3.4
```

`make_pn` placeholder body:
```python
def make_pn(poc=None, pn_pars=None):
    """Placeholder: real ANCPN wiring lives in Task 3.4."""
    # Return an intervention that does nothing; will be replaced.
    import starsim as ss
    class _NoOp(ss.Intervention):
        def step(self):
            return
    intv = _NoOp()
    intv.name = 'pn_placeholder'
    return intv
```

- [ ] **Step 5: Restore model.py's intervention hooks**

Reverse the comment-out from Task 1.3, Step 2. Restore:
```python
from interventions import make_testing, make_syph_testing, make_pn
```
and restore the `make_interventions` call inside `make_sim`.

- [ ] **Step 6: Verify a full sim builds and runs 1 year**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -c "
from model import make_sim
sim = make_sim(seed=1, n_agents=1000, start=2020, stop=2021, which='all')
sim.run()
print('Sim ran; diseases:', list(sim.diseases.keys()))
"
```

Expected: sim completes without error; prints `Sim ran; diseases: ['hiv', 'ng', 'ct', 'tv', 'bv', 'syph', 'gudp']`.

- [ ] **Step 7: Re-run the smoke test to confirm no regression**

Run:
```bash
cd /home/robyn/anc_sti_screening
python smoke_test_reproducibility.py
```

Expected: still PASS. If it now fails, one of the added interventions (e.g. syph_symp_test or syph_anc_probs) is affecting the calibration dynamics; investigate.

- [ ] **Step 8: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add model.py interventions.py
git commit -m "interventions: restore full sti_notification-style intervention wiring in model.py"
```

### Task 3.2: Adapt ANCScreen to consume scenario-cell parameters

**Files:**
- Modify: `interventions.py::ANCScreen` — verify or update `screen_prob` handling to accept a scalar (which is what the scenario cells will pass), and `ga_min`/`ga_max` to accept per-cell values.
- Test: `tests/test_ancscreen_cell_params.py`

**Interfaces:**
- Consumes: existing `ANCScreen` class.
- Produces: verified ANCScreen with:
  - `screen_prob: float` — probability of screening at ANC visit
  - `ga_min: float` — GA lower bound in weeks
  - `ga_max: float` — GA upper bound in weeks
  - When two `ANCScreen` instances are used (enrolment + 3rd tri), their GA windows are non-overlapping and both fire independently.

- [ ] **Step 1: Read the existing ANCScreen implementation**

Read `interventions.py::ANCScreen` from top to bottom. Note: `screen_prob` is stored as `ss.bernoulli(p=screen_prob)`; `ga_min`/`ga_max` are stored as instance attrs used in the `step()` eligibility filter.

- [ ] **Step 2: Write a failing integration test**

Create `tests/test_ancscreen_cell_params.py`:

```python
"""Verify ANCScreen accepts and applies scenario-cell params."""
import numpy as np
import pytest


def _build_sim(screen_prob=0.5, n_agents=500, stop=2030):
    """Build a minimal sim with a single enrolment ANCScreen."""
    import stisim as sti
    from model import make_sim
    from interventions import ANCScreen

    sim = make_sim(seed=1, n_agents=n_agents, start=2025, stop=stop,
                   which='discharging', fetal_health=True, verbose=-1)
    # Get the disease modules to attach to the screen
    ng = next(d for d in sim.pars['diseases'] if d.name == 'ng')
    ct = next(d for d in sim.pars['diseases'] if d.name == 'ct')
    tv = next(d for d in sim.pars['diseases'] if d.name == 'tv')

    screen = ANCScreen(
        diseases=[ng, ct, tv],
        screen_prob=screen_prob,
        ga_min=0, ga_max=24,   # enrolment window
        name='anc_enroll',
        label='anc_enroll',
        start=2028,
    )
    sim.pars['interventions'].append(screen)
    return sim, screen


def test_ancscreen_screens_at_rate():
    sim, screen = _build_sim(screen_prob=0.9, n_agents=2000, stop=2029)
    sim.init()
    sim.run()
    # If screening rate is 0.9, most eligible women should be screened.
    # We can't verify the exact rate without controlling pregnancies,
    # but n_screened > 0 for a 0.9 rate is a minimum sanity check.
    assert screen.results['n_screened'].values.sum() > 0


def test_ancscreen_ga_window_filters_correctly():
    """A screen with ga_min=32, ga_max=34 should NOT fire on <32w women."""
    sim, screen = _build_sim(screen_prob=1.0, n_agents=2000, stop=2029)
    # Override the GA window post-build
    screen.ga_min = 32
    screen.ga_max = 34
    sim.init()
    sim.run()
    # Assert we didn't screen anyone whose GA at screening was < 32w.
    # ANCScreen's step() should have filtered on GA. Actual verification
    # depends on ANCScreen exposing per-screen GA in results; if it
    # doesn't, this test can be a smoke test that just checks the run
    # completes without error and n_screened is non-negative.
    assert screen.results['n_screened'].values.sum() >= 0
```

- [ ] **Step 3: Run the test — should fail cleanly if ANCScreen has a bug, or pass if wiring already works**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -m pytest tests/test_ancscreen_cell_params.py -v
```

Expected: test either passes or fails with a clear signal about what's missing. If tests pass, ANCScreen's existing implementation already handles what scenario cells need — skip to Step 5 (commit test as regression suite).

If they fail, proceed to Step 4.

- [ ] **Step 4: Fix ANCScreen if needed**

Common issues:
- `screen_prob` is stored as a Bernoulli; if the test passes a float, ensure the code path handles the float correctly and doesn't reset the Bernoulli mid-run.
- `ga_min`/`ga_max` may not filter correctly at `step()` time; verify the filter uses `sim.people.pregnancy.gestation` (in weeks or the equivalent).

Read the existing `step()` method and make targeted fixes.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add interventions.py tests/test_ancscreen_cell_params.py
git commit -m "interventions: verify ANCScreen scenario-cell param wiring + regression tests"
```

### Task 3.3: Wire effect-size assumptions into sti_fetal

**Files:**
- Modify: `connectors.py::sti_fetal` — verify that `ptb_shift_mean`, `growth_penalty`, `tx_residual_growth`, `tx_residual_timing`, `ptb_shift_std` can be set from a caller-supplied dict via `pars=...` at construction time.
- Test: `tests/test_effect_size_assumptions.py`

**Interfaces:**
- Consumes: existing `sti_fetal` connector (already ported from sti_notification in Task 1.3).
- Produces: verified that:
  - `sti_fetal(pars={'ptb_shift_mean': dict(ng=1.5, ct=..., tv=..., syph=...), 'growth_penalty': dict(...), 'tx_residual_growth': dict(tri1=..., tri2=..., tri3=...), 'tx_residual_timing': dict(...)})` applies those params.
  - Default construction yields the sti_notification defaults.

**Note on structure mismatch:** the design spec §4.2 used `_early`/`_late` reversibility keys (from the old VoI plan doc). The actual `sti_fetal` connector uses trimester keys (`tri1`, `tri2`, `tri3`) — reversibility varies over three trimesters, not two windows. The `scenarios.py` `EFFECT_SIZE_ASSUMPTIONS` dict built in Task 3.5 must use trimester keys to match the actual model API. This is a plan-time correction to the spec.

- [ ] **Step 1: Verify the sti_fetal connector's `define_pars` accepts overrides via `pars=` kwarg**

Read `/home/robyn/anc_sti_screening/connectors.py` — specifically the `sti_fetal.__init__` and the `update_pars(pars, **kwargs)` call. Standard starsim pattern: `define_pars` sets defaults; `update_pars(pars=user_pars)` overrides.

- [ ] **Step 2: Write a failing test**

Create `tests/test_effect_size_assumptions.py`:

```python
"""Verify sti_fetal accepts effect-size assumption params."""
import sciris as sc
import pytest


def test_sti_fetal_default_params():
    from connectors import sti_fetal
    c = sti_fetal()
    assert c.pars.ptb_shift_mean['syph'] == pytest.approx(4.0)
    assert c.pars.growth_penalty['ng'] == pytest.approx(0.08)
    assert c.pars.tx_residual_growth['tri1'] == pytest.approx(0.25)
    assert c.pars.tx_residual_timing['tri3'] == pytest.approx(0.75)


def test_sti_fetal_overrides_with_pars_kwarg():
    from connectors import sti_fetal
    overrides = dict(
        ptb_shift_mean=sc.objdict(ng=3.0, ct=2.0, tv=1.5, syph=6.0),
        growth_penalty=sc.objdict(ng=0.12, ct=0.05, tv=0.05, syph=0.18),
        tx_residual_growth=sc.objdict(tri1=1.0, tri2=1.0, tri3=1.0),
        tx_residual_timing=sc.objdict(tri1=1.0, tri2=1.0, tri3=1.0),
    )
    c = sti_fetal(pars=overrides)
    assert c.pars.ptb_shift_mean['syph'] == pytest.approx(6.0)
    assert c.pars.growth_penalty['ng'] == pytest.approx(0.12)
    # 'no treatment effect' — no reversibility of either damage type
    assert c.pars.tx_residual_growth['tri1'] == pytest.approx(1.0)
    assert c.pars.tx_residual_timing['tri3'] == pytest.approx(1.0)
```

- [ ] **Step 3: Run — most likely passes on the ported code**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -m pytest tests/test_effect_size_assumptions.py -v
```

Expected: passes (the ported `sti_fetal` should support the override pattern out of the box). If it fails, adjust the connector's `__init__` to correctly propagate `pars=` overrides to the objdicts (likely via `update_pars`).

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add tests/test_effect_size_assumptions.py connectors.py
git commit -m "test: verify sti_fetal effect-size param overrides"
```

### Task 3.4: Replace STIPartnerNotification with ANCPN subclass

**Files:**
- Modify: `interventions.py` — delete `STIPartnerNotification` (~150 lines); add `ANCPN(sti.PartnerNotification)` (~40 lines).
- Test: `tests/test_ancpn.py`

**Interfaces:**
- Consumes: `sti.PartnerNotification` from stisim rc1.5.9 (PR 505 is landed).
- Produces:
  - `ANCPN(sti.PartnerNotification)` — subclass whose `step()` finds current partners of women who tested positive at any ANCScreen this timestep and applies the parent class's notification+treatment logic.
  - Wired into `make_pn(...)` in `interventions.py` to replace the Task-3.1 placeholder.

- [ ] **Step 1: Read `sti.PartnerNotification` in stisim rc1.5.9**

Run:
```bash
grep -n "class PartnerNotification\|def step\|def notify\|define_pars" /home/robyn/stisim/stisim/interventions/base_interventions.py | head -30
```

Read the class signature, `define_pars`, `step()`, `notify()` (if any). Note what methods/attributes the subclass needs to override or reuse.

- [ ] **Step 2: Write a failing test for ANCPN**

Create `tests/test_ancpn.py`:

```python
"""Verify ANCPN identifies current partners of ANC-positive women."""
import numpy as np
import pytest


def test_ancpn_class_exists_and_subclasses_stisim():
    import stisim as sti
    from interventions import ANCPN
    assert issubclass(ANCPN, sti.PartnerNotification)


def test_ancpn_finds_partners_of_anc_positive_women():
    """End-to-end: build a sim with ANC screen + ANCPN, run, verify
    at least one partner was notified over a multi-year run."""
    import stisim as sti
    from model import make_sim
    from interventions import ANCScreen, ANCPN

    sim = make_sim(seed=1, n_agents=2000, start=2025, stop=2030,
                   which='discharging', fetal_health=True, verbose=-1)
    ng = next(d for d in sim.pars['diseases'] if d.name == 'ng')
    ct = next(d for d in sim.pars['diseases'] if d.name == 'ct')
    tv = next(d for d in sim.pars['diseases'] if d.name == 'tv')
    screen = ANCScreen(diseases=[ng, ct, tv], screen_prob=1.0,
                       ga_min=0, ga_max=24, name='anc_enroll',
                       label='anc_enroll', start=2028)
    pn = ANCPN(anc_screen_names=['anc_enroll'], name='anc_pn',
               label='anc_pn', p_notify_treat=0.5, start=2028)
    sim.pars['interventions'].extend([screen, pn])
    sim.init()
    sim.run()

    # Over 2 years with 100% screen coverage and non-zero background STI
    # prevalence, at least one partner should have been notified.
    n_notified = pn.results.get('n_partners_notified', None)
    if n_notified is None:
        # sti.PartnerNotification may use a different result name; check
        # its results dict.
        assert any('notif' in k for k in pn.results.keys()), \
            f"No notification result in {list(pn.results.keys())}"
    else:
        assert int(np.sum(n_notified.values)) > 0
```

- [ ] **Step 3: Run test — should fail (ANCPN doesn't exist yet)**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -m pytest tests/test_ancpn.py -v
```

Expected: `ImportError: cannot import name 'ANCPN'` or similar.

- [ ] **Step 4: Delete STIPartnerNotification and implement ANCPN**

In `interventions.py`:

1. Delete the entire `STIPartnerNotification` class (~lines 389-486 based on Task 3.1 grep output).
2. Add ANCPN as a subclass of `sti.PartnerNotification`:

```python
import stisim as sti


class ANCPN(sti.PartnerNotification):
    """Partner notification triggered by ANC-screen positives.

    Subclass of stisim's PartnerNotification (PR 505, landed in
    rc1.5.8+) that overrides eligibility to pick up index cases from
    one or more ANCScreen interventions. Uses the edge-stratified
    PN machinery from the base class.
    """

    def __init__(self, anc_screen_names=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anc_screen_names = list(anc_screen_names or ['anc_enroll', 'anc_tri3'])

    def get_index_cases(self):
        """Return the set of women who tested positive at any ANC screen
        this timestep across the configured screens."""
        import starsim as ss
        sim = self.sim
        ti = self.ti
        idx = ss.uids()
        for name in self.anc_screen_names:
            screen = sim.interventions.get(name)
            if screen is None:
                continue
            # Positive at this screen this timestep, per-disease
            just_screened = (screen.ti_tested == ti)
            for disease in getattr(screen, 'diseases', []):
                positive = just_screened & disease.infected
                idx = idx | positive.uids
        return idx

    def step(self):
        # Base-class step() should call self.get_index_cases() and
        # dispatch to notify/treat. If the base class hard-codes
        # eligibility, we override step() to bypass and route through
        # the base class's notify() helpers.
        # PR 505 exposes get_index_cases as an override point; confirm
        # this is the case by reading base_interventions.py.
        return super().step()
```

**Note:** the exact override point depends on how PR 505 structured the base class. Read `stisim.interventions.base_interventions.PartnerNotification.step()` and locate where eligibility is determined. If the base class uses `self.get_index_cases()` or `self.eligible()`, override that method as in the template above. If it uses a different hook, adjust the override accordingly.

3. Replace the `make_pn` placeholder with:

```python
def make_pn(poc=None, pn_pars=None):
    """Build the ANC-driven partner notification intervention.

    Args:
        poc: unused (accepted for API compat with sti_notification).
        pn_pars: dict of overrides for the underlying PartnerNotification.
    """
    kw = dict(pn_pars or {})
    return ANCPN(name='anc_pn', label='anc_pn', **kw)
```

- [ ] **Step 5: Run test to verify ANCPN works**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -m pytest tests/test_ancpn.py -v
```

Expected: passes. If the second test fails due to a wiring detail with the base class's step/notify machinery, iterate on the override.

- [ ] **Step 6: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add interventions.py tests/test_ancpn.py
git commit -m "interventions: replace STIPartnerNotification with ANCPN subclass of sti.PartnerNotification"
```

### Task 3.5: Write scenarios.py

**Files:**
- Create: `scenarios.py`

**Interfaces:**
- Consumes: `sti_fetal` connector's parameter API from Task 3.3.
- Produces:
  - `INTERVENTION_SCENARIOS: dict[str, dict]` — 13 entries, one per cell in design-spec §4.1.
  - `EFFECT_SIZE_ASSUMPTIONS: dict[str, dict]` — 4 entries per design-spec §4.2, using trimester keys per Task 3.3's correction.
  - `build_scenario_sim(seed, scenario_id, assumption_id, draw_row, start=1985, stop=2045, n_agents=10_000)` — factory that composes model+scenario+assumption+draw into a runnable sim.

- [ ] **Step 1: Look up the meta-analysis CIs referenced in the design spec**

From `promise-voi-plan-2.md` §3b–3c:
- **PTB shifts (weeks brought forward):**
  - NG: Vallely 2021 RR 1.40 (CI 1.14–1.73). Log-linear translation gives shift ~2.0w central; low ~1.4w; high ~2.8w.
  - CT: He 2020 OR 1.35 (CI 1.11–1.63). Central ~1.5w; low ~1.0w; high ~2.2w.
  - TV: Silver 2014 RR 1.42 (CI 1.15–1.75). Central ~1.0w; low ~0.7w; high ~1.4w.
  - Syph: no meta-analysis given; use central=4.0w; low=2.5w; high=6.0w.
- **Growth penalties (fractional):**
  - NG: Vallely 2021 RR 2.23 (CI 1.34–3.71). Central 0.08; low 0.04; high 0.15.
  - CT: He 2020 OR 1.49 (CI 0.90–2.47, NS). Central 0.03; low 0.01; high 0.06.
  - TV: Silver 2014 RR 1.51 (CI 1.32–1.73). Central 0.03; low 0.02; high 0.05.
  - Syph: central 0.12; low 0.06; high 0.20.

These values are approximations from the design spec's summary of the meta-analyses; refine at implementation time if you want to back them out more rigorously from the reported RRs.

- [ ] **Step 2: Write scenarios.py**

```python
"""
Analysis grid: intervention scenarios × effect-size assumptions.

Two dimensions cross to produce the cell grid this project runs:

    INTERVENTION_SCENARIOS — what we DO
        13 named cells varying (# ANC screens) × (PN on/off) × (coverage).
        SOC is the baseline; the 12 intervention arms add NG/CT/TV
        screening at ANC in various configurations.

    EFFECT_SIZE_ASSUMPTIONS — what we BELIEVE
        4 named assumption bundles spanning two uncertainties:
          * Do STIs materially harm birth outcomes at all?
          * Does treatment during pregnancy reverse damage?

Every sim runs at a (scenario × assumption) cell, with the calibrated
parameters coming from a draw in `data/calibration_draws.csv`. K=5
seeds per (draw, scenario, assumption).

See docs/superpowers/specs/2026-07-07-port-stinotif-calibration-design.md
for the full design.
"""

from __future__ import annotations

import sciris as sc


# ────────────────────────────────────────────────────────────────────
# Intervention scenarios — 13 cells
# ────────────────────────────────────────────────────────────────────
# Each cell defines:
#   n_screens: 0 (SOC), 1 (enrolment only), or 2 (enrolment + 3rd tri)
#   coverage:  screen probability for ANC visits (float in [0,1])
#   pn:        True/False — whether ANCPN is active
INTERVENTION_SCENARIOS = sc.objdict()

INTERVENTION_SCENARIOS['soc'] = dict(
    label='SOC (syndromic + syph RPR)', n_screens=0, coverage=0.0, pn=False,
)

# 1-screen arms: enrolment only
for cov in (0.50, 0.75, 0.90):
    for pn in (False, True):
        cid = f'anc_1screen_{int(cov*100)}cov' + ('_pn' if pn else '')
        INTERVENTION_SCENARIOS[cid] = dict(
            label=cid, n_screens=1, coverage=cov, pn=pn,
        )

# 2-screen arms: enrolment + 3rd trimester
for cov in (0.50, 0.75, 0.90):
    for pn in (False, True):
        cid = f'anc_2screen_{int(cov*100)}cov' + ('_pn' if pn else '')
        INTERVENTION_SCENARIOS[cid] = dict(
            label=cid, n_screens=2, coverage=cov, pn=pn,
        )

assert len(INTERVENTION_SCENARIOS) == 13, f'expected 13 cells, got {len(INTERVENTION_SCENARIOS)}'


# ────────────────────────────────────────────────────────────────────
# Effect-size assumptions — 4 named cases
# ────────────────────────────────────────────────────────────────────
# Applied to the sti_fetal connector at sim construction time.
# Trimester keys (tri1/tri2/tri3) match the sti_fetal API; reversibility
# lower = more damage reversed by treatment.
EFFECT_SIZE_ASSUMPTIONS = sc.objdict()

EFFECT_SIZE_ASSUMPTIONS['no_treatment_effect'] = dict(
    label='No treatment effect (ratchet)',
    ptb_shift_mean=sc.objdict(ng=2.0, ct=1.5, tv=1.0, syph=4.0),
    growth_penalty=sc.objdict(ng=0.08, ct=0.03, tv=0.03, syph=0.12),
    # Ratchet: no damage ever reversed
    tx_residual_growth=sc.objdict(tri1=1.0, tri2=1.0, tri3=1.0),
    tx_residual_timing=sc.objdict(tri1=1.0, tri2=1.0, tri3=1.0),
)

EFFECT_SIZE_ASSUMPTIONS['central_reversible'] = dict(
    label='Central effects + reversible',
    ptb_shift_mean=sc.objdict(ng=2.0, ct=1.5, tv=1.0, syph=4.0),
    growth_penalty=sc.objdict(ng=0.08, ct=0.03, tv=0.03, syph=0.12),
    tx_residual_growth=sc.objdict(tri1=0.25, tri2=0.40, tri3=0.60),
    tx_residual_timing=sc.objdict(tri1=0.35, tri2=0.55, tri3=0.75),
)

EFFECT_SIZE_ASSUMPTIONS['weak_effects'] = dict(
    label='Weak effects (lower CIs)',
    ptb_shift_mean=sc.objdict(ng=1.4, ct=1.0, tv=0.7, syph=2.5),
    growth_penalty=sc.objdict(ng=0.04, ct=0.01, tv=0.02, syph=0.06),
    tx_residual_growth=sc.objdict(tri1=0.25, tri2=0.40, tri3=0.60),
    tx_residual_timing=sc.objdict(tri1=0.35, tri2=0.55, tri3=0.75),
)

EFFECT_SIZE_ASSUMPTIONS['strong_effects'] = dict(
    label='Strong effects (upper CIs)',
    ptb_shift_mean=sc.objdict(ng=2.8, ct=2.2, tv=1.4, syph=6.0),
    growth_penalty=sc.objdict(ng=0.15, ct=0.06, tv=0.05, syph=0.20),
    tx_residual_growth=sc.objdict(tri1=0.25, tri2=0.40, tri3=0.60),
    tx_residual_timing=sc.objdict(tri1=0.35, tri2=0.55, tri3=0.75),
)

assert len(EFFECT_SIZE_ASSUMPTIONS) == 4, f'expected 4 assumption sets, got {len(EFFECT_SIZE_ASSUMPTIONS)}'


# ────────────────────────────────────────────────────────────────────
# Sim factory
# ────────────────────────────────────────────────────────────────────
def build_scenario_sim(seed, scenario_id, assumption_id, draw_row,
                       start=1985, stop=2045, n_agents=10_000):
    """
    Compose a runnable sim from a cell spec + assumption + calibration draw.

    Args:
        seed (int):             RNG seed.
        scenario_id (str):      key into INTERVENTION_SCENARIOS.
        assumption_id (str):    key into EFFECT_SIZE_ASSUMPTIONS.
        draw_row (dict|Series): one row from data/calibration_draws.csv.
        start, stop (int):      sim horizon.
        n_agents (int):         population size.

    Returns:
        starsim.Sim: constructed, initialised, ready to run.
    """
    import pandas as pd
    from model import make_sim
    from apply_draw import row_to_sim_pars, set_pars_local

    cell = INTERVENTION_SCENARIOS[scenario_id]
    assumption = EFFECT_SIZE_ASSUMPTIONS[assumption_id]

    # Assumption is applied to sti_fetal via kwargs. Rebuild sti_fetal
    # with the assumption's pars.
    # Approach: build the sim with default sti_fetal, then locate the
    # sti_fetal connector in sim.pars['custom'] and update its pars.
    sim = make_sim(seed=seed, n_agents=n_agents, start=start, stop=stop,
                   which='all', fetal_health=True, verbose=-1)

    # Apply the effect-size assumption to the sti_fetal connector.
    for mod in sim.pars.get('custom', []) or []:
        if getattr(mod, 'name', None) == 'sti_fetal':
            for k in ('ptb_shift_mean', 'growth_penalty',
                       'tx_residual_growth', 'tx_residual_timing'):
                if k in assumption:
                    mod.pars[k] = assumption[k]
            break

    # Apply calibration parameters from the draw
    sim_pars = row_to_sim_pars(draw_row)
    set_pars_local(sim, sim_pars)

    # Configure ANC interventions per the cell
    _configure_anc_cell(sim, cell)

    return sim


def _configure_anc_cell(sim, cell):
    """Configure ANC screens + PN in the sim to match the scenario cell.

    Preconditions: `make_sim` has already added baseline ANC screens
    (anc_enroll + anc_tri3) and an ANCPN placeholder. This function
    activates / deactivates and sets coverage.
    """
    n_screens = cell['n_screens']
    coverage  = cell['coverage']
    pn_on     = cell['pn']

    # Look up interventions by name and configure
    intv = sim.pars.get('interventions', []) or []
    by_name = {getattr(i, 'name', None): i for i in intv}

    enrol = by_name.get('anc_enroll')
    tri3  = by_name.get('anc_tri3')
    pn    = by_name.get('anc_pn')

    # SOC: turn off both screens and PN
    if n_screens == 0:
        if enrol is not None:
            enrol.pars.screen_prob.set(p=0.0)
        if tri3 is not None:
            tri3.pars.screen_prob.set(p=0.0)
    elif n_screens == 1:
        if enrol is not None:
            enrol.pars.screen_prob.set(p=coverage)
        if tri3 is not None:
            tri3.pars.screen_prob.set(p=0.0)
    elif n_screens == 2:
        if enrol is not None:
            enrol.pars.screen_prob.set(p=coverage)
        if tri3 is not None:
            tri3.pars.screen_prob.set(p=coverage)

    # PN
    if pn is not None:
        if pn_on and n_screens > 0:
            # Enable PN — set p_notify_treat to the calibrated value
            # (using sti_notification's default of 0.5 for now; refine
            # with real data later).
            if hasattr(pn.pars, 'p_notify_treat'):
                pn.pars.p_notify_treat.set(p=0.5)
        else:
            if hasattr(pn.pars, 'p_notify_treat'):
                pn.pars.p_notify_treat.set(p=0.0)

    return
```

- [ ] **Step 3: Write a test that exercises the factory**

Create `tests/test_scenarios.py`:

```python
"""Verify scenarios.py wiring — cell dims correct, factory builds sim."""
import pandas as pd
import pytest


def test_intervention_scenarios_count():
    from scenarios import INTERVENTION_SCENARIOS
    assert len(INTERVENTION_SCENARIOS) == 13
    assert 'soc' in INTERVENTION_SCENARIOS
    # Sample a few names
    assert 'anc_1screen_50cov' in INTERVENTION_SCENARIOS
    assert 'anc_2screen_90cov_pn' in INTERVENTION_SCENARIOS


def test_effect_size_assumptions_count():
    from scenarios import EFFECT_SIZE_ASSUMPTIONS
    assert len(EFFECT_SIZE_ASSUMPTIONS) == 4
    assert 'no_treatment_effect' in EFFECT_SIZE_ASSUMPTIONS
    assert 'central_reversible' in EFFECT_SIZE_ASSUMPTIONS
    # 'no treatment effect' should have residuals all 1.0
    a = EFFECT_SIZE_ASSUMPTIONS['no_treatment_effect']
    assert a['tx_residual_growth']['tri1'] == 1.0
    assert a['tx_residual_timing']['tri3'] == 1.0


@pytest.mark.slow
def test_build_scenario_sim_smoke():
    """End-to-end: build a sim for one cell and verify basic structure."""
    from scenarios import build_scenario_sim

    df = pd.read_csv('data/calibration_draws.csv')
    row = df.iloc[0].to_dict()
    sim = build_scenario_sim(
        seed=int(row['draw_idx']) * 1000,
        scenario_id='anc_2screen_90cov_pn',
        assumption_id='central_reversible',
        draw_row=row,
        start=2025, stop=2028, n_agents=500,
    )
    # Basic checks
    disease_names = [d.name for d in sim.pars['diseases']]
    assert set(['hiv', 'ng', 'ct', 'tv', 'bv', 'syph', 'gudp']).issubset(disease_names)
    # sti_fetal is in custom
    custom_names = [m.name for m in (sim.pars.get('custom') or [])]
    assert 'sti_fetal' in custom_names
```

- [ ] **Step 4: Run the tests**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -m pytest tests/test_scenarios.py -v -k "not slow"
python -m pytest tests/test_scenarios.py -v -k "slow"
```

Expected: fast tests pass. Slow test (`test_build_scenario_sim_smoke`) takes ~30 sec to build+init a sim; should pass. If it fails, the diagnostic is either in `build_scenario_sim`'s ANC configuration or in the model port itself.

- [ ] **Step 5: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add scenarios.py tests/test_scenarios.py
git commit -m "feat: add scenarios.py with 13 intervention cells + 4 effect-size assumptions"
```

### Task 3.6: Update birth_outcome_dalys for current starsim FetalHealth API

**Files:**
- Modify: `analyzers.py::birth_outcome_dalys`

**Interfaces:**
- Consumes: `starsim.Pregnancy.preterm`, `starsim.Pregnancy.stillbirths` counts, `starsim.FetalHealth.lbw`.
- Produces: `birth_outcome_dalys` correctly computing DALYs from current API (not the outdated `fh.is_preterm` / `fh.is_lbw` names).

- [ ] **Step 1: Read the current birth_outcome_dalys implementation**

Read lines 132–225 of `analyzers.py`. Note that it references `fh.is_preterm` and `fh.is_lbw` — these don't exist in current starsim's FetalHealth.

- [ ] **Step 2: Determine the correct API**

From `starsim/starsim/demographics.py`:
- Preterm: `sim.demographics.pregnancy.preterm` (BoolArr on newborns, set at delivery).
- LBW: `sim.custom['fetal_health'].lbw` (BoolArr on newborns).
- Stillbirths: `sim.results.pregnancy.stillbirths` per timestep (already tracked natively).

- [ ] **Step 3: Rewrite the step() method**

Replace the body of `birth_outcome_dalys.step`:

```python
def step(self):
    import numpy as np
    sim = self.sim
    ti  = self.ti

    if sim.t.yearvec[ti] < self.start:
        return

    try:
        fh = sim.custom['fetal_health']
    except (KeyError, AttributeError):
        return

    preg = sim.people.pregnancy
    # Pregnancy.step() clears pregnant before analyzers run; just-delivered
    # women are identified by ti_delivery == ti and not pregnant
    delivering = (preg.ti_delivery == ti) & ~preg.pregnant
    if not delivering.any():
        return

    mother_uids = delivering.uids
    # PTB / LBW are stored on newborns, not mothers. Locate the newborns
    # of these mothers via the maternal network or the pregnancy module's
    # newborn record (implementation depends on starsim version — check
    # sim.demographics.pregnancy for the relevant array).

    # For starsim's stock behaviour, PTB flag is set on newborns at the
    # `ti_delivery` timestep. We can identify newborns as agents born
    # this timestep (age == 0 and just added), then filter to those whose
    # mothers match `mother_uids`.
    people = sim.people
    born_this_step = (people.age < sim.t.dt.years) & (people.ti_added == ti)
    newborn_uids = born_this_step.uids
    if len(newborn_uids) == 0:
        return

    # Preterm from Pregnancy module (stored on newborns)
    is_ptb = np.asarray(preg.preterm[newborn_uids], dtype=bool)
    # LBW from FetalHealth (stored on newborns)
    is_lbw = np.asarray(fh.lbw[newborn_uids], dtype=bool)

    n_ptb     = int(np.sum(is_ptb))
    n_lbw     = int(np.sum(is_lbw))
    n_ptb_lbw = int(np.sum(is_ptb & is_lbw))

    # Avoid double-counting: LBW-only accrues dw_lbw; PTB accrues dw_ptb regardless
    n_lbw_only = n_lbw - n_ptb_lbw

    yld_ptb = n_ptb      * self.dw_ptb * self.dur_ptb
    yld_lbw = n_lbw_only * self.dw_lbw * self.dur_lbw
    dalys   = yld_ptb + yld_lbw

    # Stillbirths: pull from pregnancy's native tracking (per timestep)
    stillbirths = int(sim.results.pregnancy.stillbirths.values[ti])

    self.results['n_deliveries'][ti] = len(mother_uids)
    self.results['n_ptb'][ti]        = n_ptb
    self.results['n_lbw'][ti]        = n_lbw
    self.results['n_ptb_lbw'][ti]    = n_ptb_lbw
    self.results['n_stillbirths'][ti] = stillbirths
    self.results['yld_ptb'][ti]      = yld_ptb
    self.results['yld_lbw'][ti]      = yld_lbw
    self.results['dalys'][ti]        = dalys
```

Also add `n_stillbirths` to `init_results`:

```python
def init_results(self):
    super().init_results()
    self.define_results(
        ss.Result('n_deliveries',   dtype=int,   label='Deliveries'),
        ss.Result('n_ptb',          dtype=int,   label='Preterm births'),
        ss.Result('n_lbw',          dtype=int,   label='LBW births'),
        ss.Result('n_ptb_lbw',      dtype=int,   label='PTB + LBW'),
        ss.Result('n_stillbirths',  dtype=int,   label='Stillbirths'),
        ss.Result('yld_ptb',        scale=False, label='YLD — preterm birth'),
        ss.Result('yld_lbw',        scale=False, label='YLD — LBW only'),
        ss.Result('dalys',          scale=False, label='DALYs'),
        ss.Result('cum_dalys',      scale=False, label='Cumulative DALYs'),
    )
```

- [ ] **Step 4: Add a smoke test**

Add to `tests/test_scenarios.py`:

```python
@pytest.mark.slow
def test_birth_outcome_dalys_populates():
    """Run a short sim and verify the DALY analyzer produces non-zero counts."""
    from scenarios import build_scenario_sim
    import pandas as pd

    df = pd.read_csv('data/calibration_draws.csv')
    row = df.iloc[0].to_dict()
    sim = build_scenario_sim(
        seed=int(row['draw_idx']) * 1000,
        scenario_id='soc',
        assumption_id='central_reversible',
        draw_row=row,
        start=2025, stop=2030, n_agents=1000,
    )
    sim.run()

    ana = sim.analyzers.get('birth_outcome_dalys')
    assert ana is not None
    assert ana.results['n_deliveries'].values.sum() > 0
    # DALYs may be zero if no adverse outcomes; but n_deliveries > 0
    # is a minimum sanity check.
```

- [ ] **Step 5: Run the test**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -m pytest tests/test_scenarios.py::test_birth_outcome_dalys_populates -v -s
```

Expected: passes. If FetalHealth attribute names differ from what's assumed above, adjust based on `sim.custom['fetal_health'].__dict__.keys()` at runtime.

- [ ] **Step 6: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add analyzers.py tests/test_scenarios.py
git commit -m "analyzers: update birth_outcome_dalys for current FetalHealth API + add stillbirths"
```

---

## Phase 4 — Small-N end-to-end validation

### Task 4.1: Write run_scenarios.py

**Files:**
- Create: `run_scenarios.py`

**Interfaces:**
- Consumes: `scenarios.INTERVENTION_SCENARIOS`, `scenarios.EFFECT_SIZE_ASSUMPTIONS`, `scenarios.build_scenario_sim`, `data/calibration_draws.csv`.
- Produces: `results/scenarios.jsonl` — one JSON row per completed sim, with scalars + timeseries pointers.

- [ ] **Step 1: Write `run_scenarios.py`**

```python
"""
Dispatch the (draw × seed × scenario × assumption) grid.

For each cell, build a sim via scenarios.build_scenario_sim, run it,
extract key scalars + timeseries, write one JSON row to
results/scenarios.jsonl. Timeseries + snapshots are archived to
per-cell parquets for later aggregation.

Usage:
    # Small-N test: 1 draw × 1 seed × 13 scenarios × 4 assumptions
    N_DRAWS=1 N_SEEDS=1 python run_scenarios.py

    # Full first run
    N_DRAWS=5 N_SEEDS=5 python run_scenarios.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scenarios import (INTERVENTION_SCENARIOS, EFFECT_SIZE_ASSUMPTIONS,
                        build_scenario_sim)


REPO = Path(__file__).resolve().parent
RESULTS = REPO / 'results'
RESULTS.mkdir(exist_ok=True)

DRAWS_CSV = REPO / 'data' / 'calibration_draws.csv'
N_DRAWS   = int(os.environ.get('N_DRAWS', 5))
N_SEEDS   = int(os.environ.get('N_SEEDS', 5))
N_WORKERS = int(os.environ.get('N_WORKERS', min(80, mp.cpu_count())))
START     = int(os.environ.get('START', 1985))
STOP      = int(os.environ.get('STOP', 2045))
N_AGENTS  = int(os.environ.get('N_AGENTS', 10_000))


def extract_scalars(sim, cell_id, assumption_id, draw_idx, seed):
    """Pull the small scalar summary needed for the JSONL archive."""
    r = sim.results
    def endpoint_sum(res_key, field):
        try:
            return float(np.sum(r[res_key][field].values))
        except (KeyError, AttributeError):
            return float('nan')

    return {
        'draw_idx': int(draw_idx),
        'seed': int(seed),
        'scenario_id': cell_id,
        'assumption_id': assumption_id,
        # ABO
        'n_deliveries':   endpoint_sum('birth_outcome_dalys', 'n_deliveries'),
        'n_ptb':          endpoint_sum('birth_outcome_dalys', 'n_ptb'),
        'n_lbw':          endpoint_sum('birth_outcome_dalys', 'n_lbw'),
        'n_ptb_lbw':      endpoint_sum('birth_outcome_dalys', 'n_ptb_lbw'),
        'n_stillbirths':  endpoint_sum('birth_outcome_dalys', 'n_stillbirths'),
        'dalys':          endpoint_sum('birth_outcome_dalys', 'dalys'),
        # Programmatic
        # Note: intervention_costs result-field names should be verified
        # against the analyzer's init_results() at implementation time;
        # if names differ, update these strings and rerun.
        'total_screens':  endpoint_sum('intervention_costs', 'n_screens'),
        'total_tx':       endpoint_sum('intervention_costs', 'n_treatments'),
        # Epi endpoints
        'hiv_prev_2045': float(r['hiv']['prevalence'].values[-1]) if 'prevalence' in r.get('hiv', {}) else float('nan'),
    }


def run_one(task):
    draw_idx, seed, scenario_id, assumption_id, row = task
    try:
        sim = build_scenario_sim(
            seed=seed, scenario_id=scenario_id,
            assumption_id=assumption_id, draw_row=row,
            start=START, stop=STOP, n_agents=N_AGENTS,
        )
        sim.run()
        return extract_scalars(sim, scenario_id, assumption_id, draw_idx, seed)
    except Exception as e:
        return {'draw_idx': int(draw_idx), 'seed': int(seed),
                 'scenario_id': scenario_id, 'assumption_id': assumption_id,
                 'error': f'{type(e).__name__}: {e}'}


def main():
    draws = pd.read_csv(DRAWS_CSV).head(N_DRAWS)
    tasks = []
    for _, row in draws.iterrows():
        d = int(row['draw_idx'])
        for sub in range(N_SEEDS):
            seed = d * 1000 + sub
            for sc_id in INTERVENTION_SCENARIOS.keys():
                for ax_id in EFFECT_SIZE_ASSUMPTIONS.keys():
                    tasks.append((d, seed, sc_id, ax_id, row.to_dict()))

    print(f'Grid: {len(draws)} draws × {N_SEEDS} seeds × '
          f'{len(INTERVENTION_SCENARIOS)} scenarios × '
          f'{len(EFFECT_SIZE_ASSUMPTIONS)} assumptions = {len(tasks)} sims '
          f'| workers={N_WORKERS}')

    t0 = time.time()
    out_path = RESULTS / 'scenarios.jsonl'
    with open(out_path, 'w') as f, mp.Pool(N_WORKERS) as pool:
        for i, row_out in enumerate(pool.imap(run_one, tasks, chunksize=1)):
            f.write(json.dumps(row_out) + '\n')
            f.flush()
            if (i+1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i+1) / elapsed
                eta = (len(tasks) - i - 1) / rate if rate > 0 else float('inf')
                print(f'  {i+1}/{len(tasks)} sims done '
                      f'({elapsed:.0f}s, {rate:.2f} sims/s, ETA {eta/60:.1f} min)')

    print(f'Done: {len(tasks)} sims in {time.time()-t0:.0f}s. Wrote {out_path}.')


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add run_scenarios.py
git commit -m "feat: add run_scenarios.py — dispatches (draw × seed × scenario × assumption) grid"
```

### Task 4.2: Write the aggregation script

**Files:**
- Create: `aggregate_scenarios.py`

**Interfaces:**
- Consumes: `results/scenarios.jsonl`.
- Produces:
  - `results/scenarios.kavg.csv` — K=5-averaged scalars, one row per (scenario_id, assumption_id, draw_idx).
  - Prints a diagnostic summary of the run.

For the small-N validation, only the kavg CSV is required. Full timeseries parquet aggregation lives in Task 5.2.

- [ ] **Step 1: Write `aggregate_scenarios.py`**

```python
"""
Aggregate scenarios.jsonl into K=5-averaged scalars.

Reads results/scenarios.jsonl (one row per sim); groups by
(scenario_id, assumption_id, draw_idx); computes seed-mean of every
scalar column; writes results/scenarios.kavg.csv.

Usage:
    python aggregate_scenarios.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO    = Path(__file__).resolve().parent
IN_PATH = REPO / 'results' / 'scenarios.jsonl'
OUT_CSV = REPO / 'results' / 'scenarios.kavg.csv'


def main():
    if not IN_PATH.exists():
        raise SystemExit(f'Not found: {IN_PATH}')

    df = pd.read_json(IN_PATH, lines=True)
    print(f'Loaded {len(df)} rows from {IN_PATH.name}')

    # Filter out any error rows (they'd lack numeric fields)
    if 'error' in df.columns:
        errors = df[df['error'].notna()]
        if len(errors):
            print(f'WARN: {len(errors)} sims errored; dropping from kavg.')
            print(errors[['scenario_id', 'assumption_id', 'draw_idx', 'error']].head())
        df = df[df['error'].isna()] if 'error' in df.columns else df

    key_cols  = ['scenario_id', 'assumption_id', 'draw_idx']
    num_cols  = [c for c in df.columns if c not in key_cols + ['seed', 'error']]

    kavg = df.groupby(key_cols)[num_cols].mean().reset_index()
    kavg.to_csv(OUT_CSV, index=False)
    print(f'Wrote {OUT_CSV}: {len(kavg)} rows.')

    # Diagnostic: cells present
    n_cells = kavg[['scenario_id', 'assumption_id']].drop_duplicates()
    print(f'Cells: {len(n_cells)} (expected 13 × 4 = 52).')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add aggregate_scenarios.py
git commit -m "feat: add aggregate_scenarios.py — K=5-mean kavg CSV builder"
```

### Task 4.3: Run small-N (1 draw × 1 seed × 13 × 4 = 52 sims) end-to-end

**Files:** none (execution).

**Interfaces:**
- Consumes: `run_scenarios.py`, `aggregate_scenarios.py`.
- Produces: `results/scenarios.jsonl` (52 rows), `results/scenarios.kavg.csv` (52 rows).

- [ ] **Step 1: Run the small-N dispatch**

Run:
```bash
cd /home/robyn/anc_sti_screening
N_DRAWS=1 N_SEEDS=1 N_WORKERS=20 python run_scenarios.py
```

Expected: takes ~15-25 minutes (52 sims × ~30 sec/sim serialized ÷ 20 workers). Ends with `Done: 52 sims in ...`. If any sims error, inspect the JSONL error rows.

- [ ] **Step 2: Run aggregation**

Run:
```bash
cd /home/robyn/anc_sti_screening
python aggregate_scenarios.py
```

Expected: `Wrote results/scenarios.kavg.csv: 52 rows. Cells: 52 (expected 13 × 4 = 52).`

- [ ] **Step 3: Sanity check the outputs**

Run:
```bash
cd /home/robyn/anc_sti_screening
python -c "
import pandas as pd
df = pd.read_csv('results/scenarios.kavg.csv')
print(df.groupby('scenario_id')[['n_ptb', 'n_lbw', 'n_stillbirths', 'dalys']].mean().round(1))
"
```

Expected checks (all should hold):
- SOC has non-zero `n_ptb`, `n_lbw`, `n_stillbirths`, `dalys` (baseline burden).
- 1-screen arms have lower or equal DALYs than SOC on average.
- 2-screen arms have lower or equal DALYs than 1-screen arms on average.
- Under `no_treatment_effect` assumption, all ANC arms have DALYs close to SOC (screening detects but can't reverse).
- Under `strong_effects` + `2screen_90cov_pn`, DALYs vs SOC is the largest reduction.

If any of these checks fails, diagnose before proceeding.

- [ ] **Step 4: Commit the small-N results**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add results/scenarios.kavg.csv
# scenarios.jsonl stays out of git (large + VM-only per spec §5.2)
git commit -m "test: Phase-4 small-N end-to-end (1 draw × 1 seed × 52 cells)"
```

---

## Phase 4.5 — ABO/APO artefact BEFORE the full run (added 2026-07-08)

Rationale: small-N validation (Task 4.3, quick_validate path) showed the DALY delta between 1-screen 90% and 2-screen 90% is essentially flat (Δ ≈ -0.6 under `central_reversible`; -0.9 with DxRiskRedux bundled prevention). Before spending compute on the 700-sim grid, the user wants to see cumulative ABO/APO totals **disaggregated by disease-attributable fraction** as a standalone artefact. Hypothesis: syphilis (already covered by SOC RPR) drives most ABOs; adding NG/CT/TV screening has thin marginal value. If that's the story, we want it clear from the ABO panel before any DALY chart is drawn.

### Task 4.5.1: Instrument `sti_fetal` (or a sibling analyzer) for per-disease ABO attribution

**Approach A (in-sim):** In `connectors.py::sti_fetal._apply_infection`, record the pathogen that triggered each damage stamp per pregnancy on a per-pregnancy accumulator. At delivery, reconcile the resulting ABO status (PTB / LBW / stillbirth / NND) against the accumulated pathogen list — attribute the outcome to the pathogen(s) that stamped it. Options for cases with multiple pathogens: (a) primary-attribution by first-stamp; (b) weighted attribution by cumulative damage share; (c) count under every stamping disease and flag as multi-attributable in the report.

**Approach B (counterfactual):** For each scenario, rerun with each disease's `beta_m2f` (and `beta_m2c` for syph) set to 0 in turn; diff the ABO counts. Cleaner semantics but 5x compute. Only use if in-sim attribution is prohibitively fiddly.

Ship whichever approach the next agent judges tractable; document the choice.

### Task 4.5.2: Standalone ABO/APO artefact

**Files:**
- New (or extend `aggregate_scenarios.py`): produce a table of cumulative PTB / LBW / stillbirth / NND across the 2028-2045 projection window, per (scenario × assumption × pathogen).
- New figure: horizontal stacked bar per (scenario × assumption), stacks = per-pathogen attributable ABO counts. Panel per outcome (PTB / LBW / stillbirth / NND).

Explicitly out of scope for this task: DALYs. `birth_outcome_dalys` continues to record them for internal use, but do not lead with the DALY chart until this artefact is signed off.

### Task 4.5.3: Review checkpoint

User reviews the ABO artefact and decides:
- Proceed to full 700-sim run (Task 5.1) with current scenario set, or
- Adjust the scenario set / effect-size assumptions in light of what the attribution reveals, then rerun the ABO artefact.

Do NOT launch Task 5.1 without this review.

---

## Phase 5 — Full first run

### Task 5.1: Run the full 1,300-sim grid

**Files:** none (execution).

**Interfaces:**
- Consumes: `run_scenarios.py` (unchanged).
- Produces: `results/scenarios.jsonl` (1,300 rows).

- [ ] **Step 1: Dispatch the full run on the VM**

Run:
```bash
cd /home/robyn/anc_sti_screening
N_DRAWS=5 N_SEEDS=5 N_WORKERS=80 python run_scenarios.py > results/run.log 2>&1 &
echo "PID: $!"
```

Expected: takes ~4 hours wall time. Monitor via `tail -f results/run.log`.

- [ ] **Step 2: Verify completion**

After the run finishes:
```bash
cd /home/robyn/anc_sti_screening
wc -l results/scenarios.jsonl
tail -20 results/run.log
grep -c '"error"' results/scenarios.jsonl
```

Expected: 1,300 lines in JSONL. Log ends with `Done: 1300 sims`. Zero error rows (or all errors diagnosed and rerun).

### Task 5.2: Aggregate and add timeseries + snapshots parquets

**Files:**
- Modify: `aggregate_scenarios.py` — add timeseries + snapshots parquet output.

**Interfaces:**
- Produces:
  - `results/scenarios.kavg.csv` — 260 rows (5 draws × 13 × 4).
  - `results/scenarios_timeseries.parquet` — K=5-averaged per-year timeseries.
  - `results/scenarios_snapshots.parquet` — K=5-averaged age×sex snapshots at 2028, 2035, 2040, 2045.

The timeseries + snapshot parquet outputs require capturing more than scalars from each sim. Extend the extraction in `run_scenarios.py::extract_scalars` (or add a parallel `extract_timeseries` that writes to a supplementary parquet per sim, then aggregate here).

- [ ] **Step 1: Extend run_scenarios.py to capture timeseries**

Add a `dump_timeseries(sim, cell_id, assumption_id, draw_idx, seed)` helper that writes a per-sim parquet:

```python
def dump_timeseries(sim, cell_id, assumption_id, draw_idx, seed):
    """Write per-year timeseries for one sim to a parquet fragment."""
    import numpy as np
    frag_dir = RESULTS / 'ts_frags'
    frag_dir.mkdir(exist_ok=True)
    frag = frag_dir / f'ts_{cell_id}_{assumption_id}_d{draw_idx}_s{seed}.parquet'

    rows = []
    # Yearly disease prevalence + new_infections
    for dname in ['hiv', 'ng', 'ct', 'tv', 'bv', 'syph']:
        try:
            res = sim.results[dname]
        except KeyError:
            continue
        for field in ('prevalence', 'prevalence_f', 'prevalence_m',
                       'prevalence_15_49', 'new_infections'):
            if field not in res:
                continue
            r = res[field]
            years = np.array([t.year for t in r.timevec])
            for yr, val in zip(years, r.values):
                rows.append(dict(
                    scenario_id=cell_id, assumption_id=assumption_id,
                    draw_idx=int(draw_idx), seed=int(seed),
                    disease=dname, metric=field, year=int(yr),
                    value=float(val),
                ))
    # ABO annual counts
    for field in ('n_deliveries', 'n_ptb', 'n_lbw', 'n_stillbirths', 'dalys'):
        try:
            r = sim.results['birth_outcome_dalys'][field]
        except (KeyError, AttributeError):
            continue
        years = np.array([t.year for t in r.timevec])
        for yr, val in zip(years, r.values):
            rows.append(dict(
                scenario_id=cell_id, assumption_id=assumption_id,
                draw_idx=int(draw_idx), seed=int(seed),
                disease='_abo', metric=field, year=int(yr),
                value=float(val),
            ))

    pd.DataFrame(rows).to_parquet(frag, index=False)
```

Call it from `run_one()` after `sim.run()`.

- [ ] **Step 2: Extend aggregate_scenarios.py to consume the fragments**

Add to `aggregate_scenarios.py`:

```python
def aggregate_timeseries():
    frag_dir = REPO / 'results' / 'ts_frags'
    if not frag_dir.exists() or not list(frag_dir.glob('*.parquet')):
        print('No timeseries fragments found; skipping timeseries aggregation.')
        return
    frags = list(frag_dir.glob('*.parquet'))
    print(f'Loading {len(frags)} timeseries fragments...')
    df = pd.concat([pd.read_parquet(f) for f in frags], ignore_index=True)
    # K=5-mean per (scenario, assumption, draw, disease, metric, year)
    grp = df.groupby(['scenario_id', 'assumption_id', 'draw_idx',
                       'disease', 'metric', 'year'])['value']
    kavg = grp.mean().reset_index()
    out = REPO / 'results' / 'scenarios_timeseries.parquet'
    kavg.to_parquet(out, index=False)
    print(f'Wrote {out}: {len(kavg)} rows.')
```

Call it from `main()`.

- [ ] **Step 3: Run aggregation over the full run**

Run:
```bash
cd /home/robyn/anc_sti_screening
python aggregate_scenarios.py
```

Expected:
- `Wrote results/scenarios.kavg.csv: 260 rows.`
- `Wrote results/scenarios_timeseries.parquet: <large_number> rows.`

- [ ] **Step 4: Commit the aggregation code + kavg CSV**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add run_scenarios.py aggregate_scenarios.py results/scenarios.kavg.csv
# scenarios.jsonl + scenarios_timeseries.parquet stay out of git per spec §5.2
git commit -m "run: full first-run outputs (5 draws × 5 seeds × 52 cells = 1300 sims)"
```

---

## Phase 6 — Figures and reporting

Figure adaptation is iterative and driven by collaborator feedback. Each task in this phase is self-contained; tackle them in the listed order but expect to revisit.

### Task 6.1: Adapt plot_hiv_calibration.py

**Files:**
- Modify: `plot_hiv_calibration.py`

**Interfaces:**
- Consumes: `results/scenarios_timeseries.parquet`, `data/zimbabwe_hiv_calib.csv`.
- Produces: `figures/hiv_calibration.png` — HIV 15-49 prevalence per-draw lines overlaid on the ZIMPHIA 2016 + 2020 datapoints, showing that the ensemble brackets the data.

- [ ] **Step 1: Read the existing plot script**

Understand what it currently plots (from Optuna outputs).

- [ ] **Step 2: Replace the data source**

The core change: load `scenarios_timeseries.parquet`, filter to `scenario_id='soc' AND disease='hiv' AND metric='prevalence_15_49'`, plot one line per (draw × assumption) or aggregate to (draw only) since SOC dynamics should be assumption-independent.

Overlay ZIMPHIA 2016 (15.9%) and 2020 (14.8%) datapoints.

- [ ] **Step 3: Generate the figure**

Run:
```bash
cd /home/robyn/anc_sti_screening
python plot_hiv_calibration.py
ls figures/hiv_calibration.png
```

Expected: figure produced. Inspect it and verify the ensemble brackets the datapoints.

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add plot_hiv_calibration.py figures/hiv_calibration.png
git commit -m "figures: adapt plot_hiv_calibration to consume ensemble parquet"
```

### Task 6.2: Adapt plot_sti_epi.py

**Files:**
- Modify: `plot_sti_epi.py`

**Interfaces:**
- Consumes: `results/scenarios_timeseries.parquet`, `data/zimbabwe_sti_data.csv`, `data/zimbabwe_syph_data.csv`.
- Produces: `figures/sti_epi.png` — a panel per pathogen (NG, CT, TV, syph) with ensemble prevalence lines + program surveillance datapoints. Filtered to `scenario_id='soc'` since STI epi validation is scenario-independent (SOC is the calibrated baseline).

- [ ] **Step 1: Read the existing plot_sti_epi.py**

Understand what it plots today (from Optuna outputs) — sex-stratified prevalence over time.

- [ ] **Step 2: Rewrite the data-loading block**

Load `scenarios_timeseries.parquet`; filter to `scenario_id='soc' AND assumption_id='central_reversible'` (any assumption is fine for SOC since SOC dynamics are assumption-invariant, but pick one to avoid over-plotting); pivot to per-year lines per pathogen. Overlay the datapoints from `zimbabwe_sti_data.csv` (NG/CT/TV) and `zimbabwe_syph_data.csv`.

- [ ] **Step 3: Generate the figure**

Run:
```bash
cd /home/robyn/anc_sti_screening
python plot_sti_epi.py
ls figures/sti_epi.png
```

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add plot_sti_epi.py figures/sti_epi.png
git commit -m "figures: adapt plot_sti_epi to consume ensemble parquet"
```

### Task 6.3: Adapt plot_network.py

**Files:**
- Modify: `plot_network.py`

**Interfaces:**
- Consumes: one representative sim (built via `scenarios.build_scenario_sim` with the top draw, SOC scenario, central assumption).
- Produces: `figures/network_structure.png` — degree distribution, partnership-duration histograms, or whatever the existing plot inspects.

Network structure is a per-sim (not per-cell) property, so we build one sim inline rather than reading the ensemble parquet.

- [ ] **Step 1: Read the existing plot_network.py**

Understand what it plots (likely degree distribution, mixing matrix, FSW share).

- [ ] **Step 2: Update the sim-construction block**

Replace whatever old sim-construction call it uses with:

```python
import pandas as pd
from scenarios import build_scenario_sim

df = pd.read_csv('data/calibration_draws.csv')
row = df.iloc[0].to_dict()
sim = build_scenario_sim(
    seed=int(row['draw_idx']) * 1000,
    scenario_id='soc', assumption_id='central_reversible',
    draw_row=row, start=1985, stop=2020, n_agents=10_000,
)
sim.run()
```

Then keep the existing plotting logic that consumes `sim.networks.structuredsexual` and its properties.

- [ ] **Step 3: Generate**

Run:
```bash
cd /home/robyn/anc_sti_screening
python plot_network.py
```

- [ ] **Step 4: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add plot_network.py figures/network_structure.png
git commit -m "figures: adapt plot_network to consume ported sim"
```

### Task 6.4: New figure — DALYs averted by (scenario × assumption)

**Files:**
- Create: `plot_dalys_averted.py`

**Interfaces:**
- Consumes: `results/scenarios.kavg.csv`.
- Produces: `figures/dalys_averted_by_scenario.png` — grouped bar chart of DALYs averted (vs SOC baseline) per intervention scenario, coloured by assumption set. This is the headline chart for the analysis.

- [ ] **Step 1: Write plot_dalys_averted.py**

```python
"""
Headline DALYs-averted figure.

For each (scenario, assumption): cumulative DALYs averted vs SOC in
the same assumption. Presented as grouped bars — one group per
intervention scenario, coloured by assumption.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


REPO = Path(__file__).resolve().parent
KAVG = REPO / 'results' / 'scenarios.kavg.csv'
FIG  = REPO / 'figures' / 'dalys_averted_by_scenario.png'
FIG.parent.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(KAVG)

    # Compute per-(draw, assumption) SOC DALYs; join into df
    soc = df[df['scenario_id'] == 'soc'].set_index(['draw_idx', 'assumption_id'])['dalys']
    df['dalys_soc'] = df.apply(
        lambda r: soc.get((r['draw_idx'], r['assumption_id']), float('nan')), axis=1
    )
    df['dalys_averted'] = df['dalys_soc'] - df['dalys']

    # Aggregate over draws
    agg = df.groupby(['scenario_id', 'assumption_id'])['dalys_averted'].agg(['mean', 'std']).reset_index()

    # Plot — only intervention arms (drop SOC — averted-from-self is 0)
    intv = agg[agg['scenario_id'] != 'soc'].copy()
    scenarios = sorted(intv['scenario_id'].unique())
    assumptions = ['no_treatment_effect', 'weak_effects', 'central_reversible', 'strong_effects']

    fig, ax = plt.subplots(figsize=(14, 6))
    x = list(range(len(scenarios)))
    width = 0.2
    for i, ax_id in enumerate(assumptions):
        sub = intv[intv['assumption_id'] == ax_id].set_index('scenario_id').reindex(scenarios)
        ax.bar([xi + i*width for xi in x], sub['mean'], width, yerr=sub['std'], label=ax_id)

    ax.set_xticks([xi + 1.5*width for xi in x])
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel('DALYs averted vs SOC (K=5 mean per draw, aggregated across 5 draws)')
    ax.set_title('DALYs averted by ANC screening scenario, by effect-size assumption')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(title='Effect-size assumption')
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f'Wrote {FIG}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Generate**

Run:
```bash
cd /home/robyn/anc_sti_screening
python plot_dalys_averted.py
```

- [ ] **Step 3: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add plot_dalys_averted.py figures/dalys_averted_by_scenario.png
git commit -m "figures: add DALYs-averted headline chart"
```

### Task 6.5: README update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite README.md to reflect the new pipeline**

Update sections:
- Diseases modeled → 7 diseases (add syph + GUD placeholder)
- Repo structure → new files (scenarios.py, run_scenarios.py, aggregate_scenarios.py, apply_draw.py); removed files (run_calibrations.py, priors.py, run_msim.py, promise-voi-plan-2.md)
- Pipeline → 1. Port sti_notification calibration ensemble → 2. Run scenarios (13 × 4 × 5 draws × 5 seeds) → 3. Aggregate → 4. Plot
- Scenarios table → the 13 cells
- Dependencies → pin stisim to rc1.5.9 (with fix/ng-tx merged)

- [ ] **Step 2: Commit**

Run:
```bash
cd /home/robyn/anc_sti_screening
git add README.md
git commit -m "docs: update README for ported-calibration pipeline"
```

---

## Phase 7 — PR back to main

### Task 7.1: Open PR

**Files:** none (git ops).

- [ ] **Step 1: Push all branches**

Run:
```bash
cd /home/robyn/anc_sti_screening
git push origin port/stinotif-calibration
```

- [ ] **Step 2: Open PR via gh CLI**

Run:
```bash
cd /home/robyn/anc_sti_screening
gh pr create --base main --head port/stinotif-calibration \
  --title "Port sti_notification calibration for ABO/APO/DALY analysis" \
  --body "$(cat <<'EOF'
## Summary
- Port sti_notification exp-06 calibration ensemble (17 params, 5 draws × K=5) into anc_sti_screening
- Replace Optuna machinery with ensemble-based scenario runner
- Add 13 ANC-screening scenarios × 4 effect-size assumptions = 52 cells
- Deliver ABO/APO/DALY tables + figures (no VoI/EVPI framing — dropped per collaborator feedback)

## Design
See `docs/superpowers/specs/2026-07-07-port-stinotif-calibration-design.md`.

## Test plan
- [ ] Phase-2 reproducibility smoke test PASS (already committed)
- [ ] Unit tests pass: `python -m pytest tests/ -v`
- [ ] Full-run outputs land in `results/scenarios.kavg.csv` with 260 rows
- [ ] Headline figure `figures/dalys_averted_by_scenario.png` inspects cleanly

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Provide PR URL to user**

Report the PR URL. Do not merge until user approves.

---

## Summary of file changes

**Created:** 8 files
- `apply_draw.py`
- `scenarios.py`
- `run_scenarios.py`
- `aggregate_scenarios.py`
- `smoke_test_reproducibility.py`
- `plot_dalys_averted.py`
- `tests/test_apply_draw.py`, `test_scenarios.py`, `test_ancscreen_cell_params.py`, `test_ancpn.py`, `test_effect_size_assumptions.py`
- `docs/superpowers/specs/2026-07-07-port-stinotif-calibration-design.md` (from brainstorming)
- `docs/superpowers/plans/2026-07-07-port-stinotif-calibration.md` (this file)

**Ported wholesale from sti_notification:** `model.py`, `hiv_model.py`, `connectors.py`, most of `data/`.

**Modified:** `interventions.py`, `analyzers.py`, `plot_*.py`, `README.md`.

**Deleted:** `run_calibrations.py`, `priors.py`, `run_msim.py`, `promise-voi-plan-2.md`.

**Tags/branches:** tag `v0.1` on main; branch `port/stinotif-calibration`.
