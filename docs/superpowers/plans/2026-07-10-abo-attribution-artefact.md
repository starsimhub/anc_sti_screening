# ABO/APO attribution artefact — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a standalone ABO/APO artefact — cumulative preterm birth (PTB), low birth weight (LBW), stillbirth, and neonatal death (NND) counts across the 2028-2045 projection window, **disaggregated by which STI exposed the pregnancy to damage**. Fix the analyzer's existing LBW=PTB approximation as a byproduct. This is Phase 4.5 of `2026-07-07-port-stinotif-calibration.md` — the go/no-go gate for launching the full 700-sim grid.

**Hypothesis under test:** Syphilis (already covered by SOC RPR-at-ANC) drives most ABOs. If that's true, adding NG/CT/TV screening will show thin marginal effect against the SOC syph baseline, and the story is about the syph pathway rather than the NG/CT/TV panel.

**Outcome scope (locked in with user):**
- **PTB and LBW**: per-disease attribution across NG / CT / TV / syph (all four). Both are mechanistically produced by `sti_fetal` for all four covered diseases via delivery-timing shift and growth restriction.
- **Stillbirth and NND**: syph-only. `sti.Syphilis` provides per-newborn `ti_stillborn` and `ti_nnd` states drawn from stage-specific `birth_outcomes` distributions; NG/CT/TV have no stillbirth or NND pathway in this model. The `min_ga=24w` floor in FetalHealth prevents timing shifts from bringing delivery below viability, so `sti_fetal` cannot indirectly produce stillbirths. This model gap is a stated limitation in the artefact prose.

**Attribution rule (locked in with user):** For PTB and LBW, count each ABO under *every* pathogen that exposed that pregnancy. Rows sum to ≥ n_ABO. Report a companion "sole vs shared" split so multi-pathogen bookkeeping is visible. Stillbirth and NND have only one contributing disease (syph), so no sole/shared split.

**Accumulator location (locked in with user):** On `sti_fetal` as `ss.BoolArr` states — one per disease, named `exposed_<disease>`. Reconciliation happens at delivery via a Pregnancy delivery callback.

**LBW math (locked in with user):** Fix now. Since attribution needs per-birth flags at delivery anyway, compute the true PTB / LBW / PTB+LBW breakdown from those flags and update `birth_outcome_dalys` accordingly.

## Global Constraints

- **Branch:** continue on `port/stinotif-calibration`. No rebase; new commits on top.
- **stisim pin:** `rc1.5.9` (with `fix/ng-tx` merged) — unchanged.
- **Working directory:** `/home/robyn/anc_sti_screening`.
- **Conda env:** `starsim`.
- **Test target:** small-N validation over the 3 scenarios already exercised in `quick_validate.py` (SOC / 1-screen 90% / 2-screen 90%), 1 draw × 1 seed, `central_reversible`. Do NOT launch the full 700-sim grid — that's Task 5.1, gated on user acceptance of this artefact.

## Model-mechanic caveats to preserve in the report

- `sti_fetal` applies damage uniformly across trimesters (no syph-specific per-GA congenital gradient). Treatment reversal IS trimester-graded (tri1 residual 0.25/growth × 0.35/timing, tri3 residual 0.60/0.75). Downstream reader should not over-interpret per-pathogen shares as biology.
- NG/CT/TV have no stillbirth or NND pathway. Real biology has one (chorioamnionitis, PROM, sepsis) but it isn't in the model. The artefact reports stillbirth+NND as syph-only and flags this as a known limitation.
- `exposed_<disease>` is set on the mother whenever `_apply_infection` fires for that disease during pregnancy, regardless of whether the effect-size params (`shift_mean`, `penalty`) are non-zero. This is deliberate — the report attributes damage-*exposure*, not damage-*magnitude*.
- Exposure marks are **not** unticked by treatment. A treated pregnancy still counts under the pathogen that infected it — with any residual damage. This matches the "damage attribution" reading, not "net-effect attribution".

## File Structure

**Modified:**
- `connectors.py::sti_fetal` — add per-pregnancy `exposed_<disease>` BoolArr states + reset in `_on_conception` + set in `_apply_infection`.
- `analyzers.py::birth_outcome_dalys` — use exact per-birth PTB/LBW flags via a delivery callback; DALY math becomes non-approximate. Add new sibling `birth_outcome_attribution` analyzer that emits per-disease attributable counts.
- `scenarios.py::build_scenario_sim` — wire the new `birth_outcome_attribution` analyzer alongside `birth_outcome_dalys`.

**Created:**
- `tests/test_abo_attribution.py` — unit + integration tests: single-pathogen exposure, dual-pathogen (shared attribution), term-SGA-LBW-without-PTB, no-infection sanity, cross-pregnancy reset.
- `aggregate_abo.py` — reads the small-N run's `results/abo_smoke.jsonl`, produces a per-(scenario × disease × outcome) tidy CSV.
- `plot_abo_attribution.py` — horizontal stacked bar per outcome (PTB / LBW attributed across 4 diseases; syph-stillbirth + syph-NND single-column).
- `results/abo_smoke.jsonl` — small-N run output (gitignored; see .gitignore update).
- `figures/abo_attribution.png` — the artefact figure (regenerable, gitignored).

**Optional:**
- `.gitignore` — add `results/*.jsonl` and `results/*.parquet` if not already covered.

---

## Phase A — Instrument `sti_fetal` with per-pregnancy exposure marks

### Task A.1: Add `exposed_<disease>` BoolArr states

**Files:** `connectors.py`

**Interfaces produced:**
- `sti_fetal.exposed_<disease>` — one `ss.BoolArr` per disease in `self.disease_names`; True on the mother's uid while she is carrying a pregnancy that has been exposed to that disease's damaging effects.

- [ ] **Step 1: Add `define_states` block to `sti_fetal.__init__`**

  Insert after the existing `update_pars(pars, **kwargs)` call:

  ```python
  self.define_states(*[
      ss.BoolArr(f'exposed_{d}', label=f'Pregnancy exposed to {d.upper()}')
      for d in self.disease_names
  ])
  ```

  Rationale: one BoolArr per covered disease. Names are derived from `self.disease_names` so the mapping stays in sync if the disease list changes.

- [ ] **Step 2: Verify state registration**

  ```bash
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate starsim
  python -c "
  from model import make_sim_parts
  parts = make_sim_parts(seed=1, n_agents=200, start=1985, stop=1986, which='all', verbose=-1)
  import stisim as sti
  sim = sti.Sim(**parts); sim.init()
  connector = [c for c in sim.pars['custom'] if getattr(c, 'name', None) == 'sti_fetal'][0]
  print([n for n in dir(connector) if n.startswith('exposed_')])
  "
  ```

  Expected: `['exposed_ct', 'exposed_ng', 'exposed_syph', 'exposed_tv']` (alphabetized by starsim's state registry).

### Task A.2: Set `exposed_<disease>` in `_apply_infection`

**Files:** `connectors.py`

**Interfaces:**
- `_apply_infection(uids, disease_name)` — same signature; side-effect sets `self.exposed_<disease_name>[uids] = True` before returning.

- [ ] **Step 1: Modify `_apply_infection`**

  Insert at the top of `_apply_infection`, before the `fh = self._get_fh()` call:

  ```python
  # Mark this pregnancy as exposed to this disease's damaging effects
  arr = getattr(self, f'exposed_{disease_name}', None)
  if arr is not None:
      arr[uids] = True
  ```

  Why at the top: exposure records the fact of infection during pregnancy, regardless of whether the effect-size params (`shift_mean`, `penalty`) are non-zero. This is deliberate — the report attributes damage-*exposure*, not damage-*magnitude*. Under `no_treatment_effect` all four assumption sets still expose on infection.

- [ ] **Step 2: Verify a single infection sets the exposure flag**

  In a Python shell:

  ```python
  # ... build a small sim, run some months, ...
  connector = [c for c in sim.pars['custom'] if getattr(c, 'name', None) == 'sti_fetal'][0]
  print('n exposed to ng:', int(connector.exposed_ng.sum()))
  ```

  Expected: > 0 by end of a multi-month run with ng active and pregnancies happening.

### Task A.3: Reset `exposed_<disease>` in `_on_conception`

**Files:** `connectors.py`

**Interfaces:**
- `_on_conception(uids)` — same signature; existing pre-existing-infection check runs, but now resets exposure flags to False on `uids` FIRST.

- [ ] **Step 1: Prepend reset**

  Insert at the top of `_on_conception`, before the existing per-disease loop:

  ```python
  # New pregnancy: clear any prior-pregnancy exposure flags for these mothers.
  for dname in self.disease_names:
      arr = getattr(self, f'exposed_{dname}', None)
      if arr is not None:
          arr[uids] = False
  ```

  Rationale: agents can have multiple pregnancies over the sim lifetime. Without a reset, a mother's exposure flags from pregnancy N carry into pregnancy N+1.

- [ ] **Step 2: Verify reset**

  Unit test that runs 2 back-to-back pregnancies for the same agent, infects between them, and asserts `exposed_ng` is True during pregnancy N and False at the start of pregnancy N+1. (See Task D.5.)

---

## Phase B — Reconciliation analyzer

### Task B.1: New `birth_outcome_attribution` analyzer skeleton

**Files:** `analyzers.py`

**Interfaces produced:**
- Class `birth_outcome_attribution(ss.Analyzer)` with per-timestep integer results:
  - **PTB** per-disease attribution (all four diseases): `n_ptb_<d>`, `n_ptb_sole_<d>`, `n_ptb_shared_<d>` for `d in (ng, ct, tv, syph)`, plus `n_ptb_no_attribution`.
  - **LBW** per-disease attribution (all four diseases): same triple + `n_lbw_no_attribution`.
  - **Stillbirth**: syph-only. Single result `n_stillbirth_syph`. No sole/shared split — only syph contributes in this model.
  - **NND**: syph-only. Single result `n_nnd_syph`. Same reasoning.
- Totals sanity check: `sum(n_ptb_<d>) - sum(n_ptb_shared_<d>) / 2 * (n_diseases - 1) ≈ n_ptb - n_ptb_no_attribution` (loose bound; exact equality gets messy with 3+ way overlaps but the artefact prose will spell out how to read the columns).

- [ ] **Step 1: Write class skeleton with Result definitions**

  Add to `analyzers.py`. Class has `__init__(diseases=('ng','ct','tv','syph'), start=None)`, `init_pre`, `init_results`, `_on_delivery(mother_uids, newborn_uids)` callback.

  Result declarations: `sc.autolist()` accumulator. For each `d` in `self.diseases`, add three PTB results (`n_ptb_<d>`, `n_ptb_sole_<d>`, `n_ptb_shared_<d>`) and three LBW results. Then add four scalars: `n_ptb_no_attribution`, `n_lbw_no_attribution`, `n_stillbirth_syph`, `n_nnd_syph`.

- [ ] **Step 2: Register the delivery callback**

  In `init_pre`, after `super().init_pre(sim)`:

  ```python
  self._sti_fetal = next(
      (c for c in sim.pars.get('custom') or []
       if getattr(c, 'name', None) == 'sti_fetal'),
      None,
  )
  if self._sti_fetal is None:
      raise ValueError('birth_outcome_attribution requires sti_fetal in sim.pars["custom"].')
  self._pregnancy = sim.demographics.pregnancy
  self._fh = sim.custom['fetal_health']
  self._syph = sim.diseases.get('syph')
  self._pregnancy.add_delivery_callback(self._on_delivery)
  ```

### Task B.2: Delivery-callback reconciliation

**Files:** `analyzers.py::birth_outcome_attribution._on_delivery`

**Interfaces consumed:**
- `mother_uids, newborn_uids` from the Pregnancy delivery callback.
- `self._sti_fetal.exposed_<disease>[parent_uids]` — boolean per-mother exposure flags.
- `self._pregnancy.preterm[newborn_uids]` — boolean per-newborn (PTB).
- `self._fh.lbw[newborn_uids]` — boolean per-newborn (LBW).
- `self._syph.ti_stillborn[newborn_uids]` — float per-newborn; matches current `ti` if syph killed this fetus.
- `self._syph.ti_nnd[newborn_uids]` — float per-newborn; matches current `ti` (or a small window after) if syph caused NND.

**Interfaces produced:** per-timestep integer bumps to the Result columns declared in Task B.1.

- [ ] **Step 1: Read per-birth outcome flags**

  ```python
  ti = self.ti
  is_ptb = self._pregnancy.preterm[newborn_uids]
  is_lbw = self._fh.lbw[newborn_uids]
  if self._syph is not None:
      is_syph_stillborn = self._syph.ti_stillborn[newborn_uids] == ti
      is_syph_nnd       = self._syph.ti_nnd[newborn_uids] == ti
  else:
      is_syph_stillborn = np.zeros(len(newborn_uids), dtype=bool)
      is_syph_nnd       = np.zeros(len(newborn_uids), dtype=bool)
  ```

  **Verification during Phase E:** cross-check per-timestep totals against `syph.results['new_stillborns']` and `syph.results['new_nnds']`. If they diverge, the syph module fires the events on a different timestep than delivery — in that case switch to a small trailing window (e.g. `ti_stillborn >= ti - 1`) or accumulate deferred deaths in a separate `step()` path.

- [ ] **Step 2: Build the per-birth exposure matrix**

  ```python
  parents = self.sim.people.parent[newborn_uids]
  exposure_cols = {}
  for d in self.diseases:
      arr = getattr(self._sti_fetal, f'exposed_{d}', None)
      exposure_cols[d] = arr[parents] if arr is not None else np.zeros(len(newborn_uids), dtype=bool)
  n_exposures_per_birth = np.sum(list(exposure_cols.values()), axis=0)
  ```

- [ ] **Step 3: Tally per-disease PTB + LBW counts**

  ```python
  for outcome_name, outcome_mask in (('ptb', is_ptb), ('lbw', is_lbw)):
      for d in self.diseases:
          mask_d = exposure_cols[d]
          self.results[f'n_{outcome_name}_{d}'][ti] += int((outcome_mask & mask_d).sum())
          self.results[f'n_{outcome_name}_sole_{d}'][ti] += int((outcome_mask & mask_d & (n_exposures_per_birth == 1)).sum())
          self.results[f'n_{outcome_name}_shared_{d}'][ti] += int((outcome_mask & mask_d & (n_exposures_per_birth > 1)).sum())
      self.results[f'n_{outcome_name}_no_attribution'][ti] += int((outcome_mask & (n_exposures_per_birth == 0)).sum())
  ```

- [ ] **Step 4: Tally syph-only stillbirth + NND**

  ```python
  self.results['n_stillbirth_syph'][ti] += int(is_syph_stillborn.sum())
  self.results['n_nnd_syph'][ti] += int(is_syph_nnd.sum())
  ```

- [ ] **Step 5: Reset exposure flags at delivery**

  After all reads are complete:

  ```python
  for d in self.diseases:
      arr = getattr(self._sti_fetal, f'exposed_{d}', None)
      if arr is not None:
          arr[mother_uids] = False
  ```

  Defensive complement to the `_on_conception` reset in Task A.3 — covers the delivery→next-conception gap.

### Task B.3: Wire the analyzer into scenarios.py

**Files:** `scenarios.py::build_scenario_sim`

**Interfaces:**
- The scenario factory adds `birth_outcome_attribution(start=start)` to `parts['analyzers']` alongside `birth_outcome_dalys(start=start)`.

- [ ] **Step 1: Update the analyzers list**

  Change:

  ```python
  parts['analyzers'] = list(parts['analyzers']) + [birth_outcome_dalys(start=start)]
  ```

  to:

  ```python
  from analyzers import birth_outcome_dalys, birth_outcome_attribution
  parts['analyzers'] = list(parts['analyzers']) + [
      birth_outcome_dalys(start=start),
      birth_outcome_attribution(start=start),
  ]
  ```

---

## Phase C — Fix `birth_outcome_dalys` LBW/PTB reconciliation

### Task C.1: Rewrite `step()` (and add a delivery callback)

**Files:** `analyzers.py::birth_outcome_dalys`

**Interfaces:**
- Replace the per-timestep module-level count reads with a delivery callback that reads exact per-birth flags, matching the attribution analyzer's approach.
- Emit `n_ptb`, `n_lbw`, `n_ptb_only`, `n_lbw_only`, `n_ptb_and_lbw`, `n_stillbirths`, `dalys` — all with exact overlap accounting.

- [ ] **Step 1: Switch from module-count reads to a delivery callback**

  Restructure `birth_outcome_dalys` to register `on_delivery(mother_uids, newborn_uids)` in `init_pre`, similar to Task B.1's Step 2 setup. The callback reads `preg.preterm[newborn_uids]` and `fh.lbw[newborn_uids]` and computes:

  ```python
  ptb_only    = is_ptb & ~is_lbw
  lbw_only    = is_lbw & ~is_ptb
  ptb_and_lbw = is_ptb & is_lbw
  ```

- [ ] **Step 2: Update result definitions**

  Replace the old `n_ptb_lbw` result (which was ~= n_lbw by approximation) with:
  - `n_ptb_only` — preterm without LBW
  - `n_lbw_only` — term but LBW (SGA/IUGR term birth)
  - `n_ptb_and_lbw` — preterm and LBW
  - Keep `n_ptb`, `n_lbw`, `n_stillbirths` totals.

- [ ] **Step 3: Fix DALY math**

  ```python
  yld_ptb = (int(ptb_only.sum()) + int(ptb_and_lbw.sum())) * self.dw_ptb * self.dur_ptb
  yld_lbw = int(lbw_only.sum()) * self.dw_lbw * self.dur_lbw
  dalys = yld_ptb + yld_lbw
  ```

  Rationale: co-occurring PTB+LBW births accrue only the higher weight (PTB), matching design spec §5.1. Term-SGA-LBW births now correctly accrue the LBW weight instead of being silently zeroed.

- [ ] **Step 4: Handle stillbirths**

  In the current model, stillbirths come only from syph (via `sti.Syphilis.ti_stillborn`). Aggregate `n_stillbirths` in `birth_outcome_dalys` by reading either the syph module's per-timestep result (`syph.results['new_stillborns']`) or by counting `is_syph_stillborn` at delivery using the same pattern as Task B.2. Do NOT read `pregnancy.results['stillbirths']` — starsim's Pregnancy has `p_loss=0` by default so its stillbirth counter is always zero here.

- [ ] **Step 5: Verify DALY total is same order of magnitude as before**

  Under the small-N validation, cumulative DALYs should shift slightly upward (yld_lbw goes from 0 to positive) but by <10% relative — LBW-only-without-PTB is a minority of ABOs.

---

## Phase D — TDD tests

### Task D.1: Unit test — single-pathogen exposure

**Files:** `tests/test_abo_attribution.py`

- [ ] **Step 1: Write a test that infects one mother with NG only, runs to delivery**

  ```python
  def test_single_pathogen_exposure():
      # Build small sim with 1 mother uid; force ng infection; run 1 pregnancy;
      # assert exposed_ng == True, exposed_ct/tv/syph == False.
  ```

  Prefer directly constructing a small `sti.Sim` + patching `sti_fetal.exposed_ng` for the mother uid rather than trying to induce a natural infection in a small sim (which is stochastic).

- [ ] **Step 2: Run test**

  ```bash
  python -m pytest tests/test_abo_attribution.py::test_single_pathogen_exposure -v
  ```

  Expected: pass.

### Task D.2: Unit test — dual-pathogen shared attribution

- [ ] **Step 1: Test that both counters increment + shared counter fires**

  Set `exposed_ng` AND `exposed_ct` True for the same mother; deliver; assert `n_ptb_ng` and `n_ptb_ct` both incremented; `n_ptb_shared_ng` and `n_ptb_shared_ct` both incremented; `n_ptb_sole_ng` NOT incremented.

### Task D.3: Term-SGA-LBW-without-PTB

- [ ] **Step 1: Test the DALY math fix**

  Build a delivery scenario where the newborn is LBW but term. Assert `birth_outcome_dalys.results['n_lbw_only'][ti] > 0`, `n_ptb_only` unchanged, `yld_lbw > 0`.

  This tests that we no longer zero out yld_lbw.

### Task D.4: Zero-infection sanity check

- [ ] **Step 1: A pregnancy with no STI infections yields no attributions**

  Assert all `exposed_*` remain False; all `n_ptb_<disease>` == 0 for that birth; `n_ptb_no_attribution` bumps by 1 if PTB fires from background causes.

### Task D.5: Reset across successive pregnancies

- [ ] **Step 1: A mother whose N-th pregnancy was exposed to NG should start pregnancy N+1 with exposed_ng False**

  Force conception → infect → deliver → conception → assert `exposed_ng[mother_uid]` is False right after 2nd conception.

---

## Phase E — Small-N validation + figure

### Task E.1: Run small-N over 3 scenarios

**Files:** none (execution).

- [ ] **Step 1: Run**

  ```bash
  source $(conda info --base)/etc/profile.d/conda.sh && conda activate starsim
  cd /home/robyn/anc_sti_screening
  python quick_validate.py 2>&1 | tee results/abo_smoke.log
  ```

  Update `quick_validate.py` if needed to dump the new attribution Results to the JSONL row (or write a new one-off driver `abo_smoke.py` that dumps a superset).

- [ ] **Step 2: Sanity checks on stdout**

  - Each scenario's `n_ptb_syph + n_ptb_ng + n_ptb_ct + n_ptb_tv + n_ptb_no_attribution` should be ≥ the total PTB count from `birth_outcome_dalys` (≥ because of shared attribution).
  - `n_ptb_syph_sole + n_ptb_syph_shared` should equal `n_ptb_syph`.
  - `n_syph_nnd` should be non-zero if syph NND API exists.

### Task E.2: Aggregation script

**Files:** `aggregate_abo.py`

- [ ] **Step 1: Read the JSONL, produce a tidy CSV**

  Columns: `scenario_id, disease, outcome, count, count_sole, count_shared, cumulative_2028_2045`.

  Format: one row per (scenario × disease × outcome). Outcome ∈ {ptb, lbw, stillbirth, syph_nnd}.

- [ ] **Step 2: Add a "no attribution" row per (scenario × outcome)**

  Distinguishes background ABOs from STI-attributable ones. Reader should see (e.g.) "SOC: 12,300 PTBs of which 4,500 are syph-attributable, 800 NG-attributable, 6,900 no-attribution".

### Task E.3: Horizontal stacked bar figure

**Files:** `plot_abo_attribution.py`

- [ ] **Step 1: One panel per outcome (PTB, LBW, stillbirth, syph-NND)**

  Y axis: 3 scenarios (SOC, 1-screen 90%, 2-screen 90%). X axis: cumulative count 2028-2045. Stacks: per-pathogen contribution + "no attribution" as the last (grey) segment.

- [ ] **Step 2: Add a companion "sole vs shared" small-multiples panel**

  Same 3 scenarios × 4 outcomes; split each pathogen bar into hatched (shared) + solid (sole) segments so the reader can distinguish "syph drove this alone" from "syph and NG/CT/TV both exposed this pregnancy".

- [ ] **Step 3: Save to `figures/abo_attribution.png`**

  Regeneratable; gitignored.

---

## Phase F — Review checkpoint

- [ ] **Step 1: Present the artefact to the user**

  Include:
  - The 4-outcome horizontal-bar figure.
  - The sole/shared companion figure.
  - A short summary paragraph — what fraction of each outcome is syph-attributable across scenarios? Does 1→2 screen move any pathogen's contribution meaningfully?

- [ ] **Step 2: User decides**

  - Proceed to Task 5.1 (full 700-sim run), OR
  - Adjust scenarios / assumptions in light of what the attribution reveals, then rerun this artefact.

  **Do NOT launch Task 5.1 without user acceptance of this artefact.**

---

## Commit strategy

Suggested commits (roughly one per phase):

1. `feat(connectors): sti_fetal records per-pregnancy exposure per disease` — Phase A (all 3 tasks).
2. `feat(analyzers): birth_outcome_attribution + delivery-callback reconciliation` — Phase B tasks B.1-B.3.
3. `feat(scenarios): wire birth_outcome_attribution into build_scenario_sim` — Phase B.4 (tiny, could bundle with 2).
4. `fix(analyzers): birth_outcome_dalys uses exact per-birth PTB/LBW flags` — Phase C.
5. `test: ABO attribution + DALY reconciliation tests` — Phase D.
6. `feat(reporting): aggregate_abo.py + plot_abo_attribution.py + small-N run` — Phase E.

Keep test additions bundled with the feature commit if they land in the same session; split only if the test set grows large.
