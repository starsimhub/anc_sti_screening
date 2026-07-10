# Handoff — anc_sti_screening (2026-07-08)

Branch: `port/stinotif-calibration` (unpushed). Nothing on this branch is committed since 2026-07-07 evening; all Session-2026-07-08 work described here is uncommitted.

## Where we are

Phases 1-4 of `docs/superpowers/plans/2026-07-07-port-stinotif-calibration.md` are done. Small-N validation via `quick_validate.py` (SOC / 1-screen 90% / 2-screen 90% × `central_reversible`, K=1 seed, 1985-2032, 10k agents) ran cleanly.

**Results (single draw × single seed, so ±1 DALY is within noise):**

| scenario           | DALYs | n_ptb  | n_lbw  | NG prev | CT prev | TV prev |
| ------------------ | ----- | ------ | ------ | ------- | ------- | ------- |
| SOC                | 588.9 | 3.42M  | 1.36M  | 1.36%   | 12.3%   | 21.2%   |
| anc_1screen_90cov  | 573.2 | 3.32M  | 1.29M  | 1.06%   | 10.8%   | 18.3%   |
| anc_2screen_90cov  | 572.6 | 3.32M  | 1.29M  | 1.05%   | 10.5%   | 17.5%   |

Adding `DxRiskRedux` (screen-triggered rel_sus × 0.1 for 3mo, i.e. bundled prevention triggered by ANC attendance) barely widened the 1-vs-2-screen gap (Δ = -0.90 with BP vs -0.60 without). Small-N results are **not yet accepted by the user**.

## Substantive finding + user's next-step directive

Enrol-time (0-24w GA) prevalent-infection pool dominates the DALY damage. The tri3 incident-infection pool is small AND has short remaining-damage window, so both "catch-and-treat" and "rel_sus-boost" mechanisms give tiny 2-screen benefit under `central_reversible`.

**User's directive for next steps (verbatim from the 2026-07-08 conversation):**

1. **Concentrate on ABO/APO metrics** — PTB, LBW, stillbirths, NNDs, and any other adverse birth or pregnancy outcome — **as a standalone artefact, not DALYs**. Report DALYs later, after this artefact is agreed.
2. **Report cumulative totals AND disaggregate by disease-attributable fraction.** Hypothesis: maybe the numbers don't move much as we add ANC screening for NG/CT/TV because most ABOs are caused by syphilis. If that's the story, syph ANC screening (already in SOC via RPR) is the whole game and adding NG/CT/TV screening has thin marginal value.

Concretely, this means before running the full 700-sim grid the next agent should:
- Extend the ABO analyzer (`analyzers.py::birth_outcome_dalys`, likely a rename or a new sibling) to record **per-disease attributable PTB / LBW / stillbirth / NND counts**. Two candidate approaches:
  - **In-sim attribution**: instrument `connectors.py::sti_fetal` to record the pathogen that triggered each damage stamp per pregnancy, then reconcile at delivery against the resulting ABO status.
  - **Counterfactual**: rerun each scenario with each disease's `beta` set to 0 in turn, diff.
- Produce a **standalone ABO report** (cumulative totals + disaggregation), agreed with user, BEFORE returning to DALY aggregation.

## Model-mechanic caveats to know before interpreting

- `sti_fetal` applies damage **per infection**, uniform across trimesters. NO trimester-graded damage. Treatment REVERSAL is trimester-graded (tri1 residual 0.25, tri3 residual 0.60).
- No syph-specific per-GA congenital-transmission gradient. Real biology has one — worth calling out in any comparison of syph vs NG/CT/TV attributable ABOs.
- `n_ptb_lbw = n_lbw` in the current analyzer, so `yld_lbw` is structurally zero (matches design spec §5.1). DALYs are effectively PTB-only.

## Recent structural changes on this branch (uncommitted)

Details in `.superpowers/sdd/progress.md` "Session 2026-07-08". Highlights:

- `ANCScreen` now accepts disease/treatment NAMES (strings) and resolves them in `init_pre(sim)`. Reason: `sti.Sim(**parts)` deep-copies modules on construction. See `memory/feedback_sti_sim_deep_copy.md`.
- `make_sim` refactored to `make_sim_parts` (returns kwargs dict) + thin `make_sim(interventions=..., analyzers=..., custom=..., **parts_kwargs)` merging wrapper.
- `_build_anc_screens(cell)` takes only the cell; screens carry name strings.
- Effect-size assumption mutation on `sti_fetal` runs POST-`sti.Sim(**parts)` (same deep-copy reason).
- ANC 3rd-tri window widened from PROMISE 32-34w → 30-36w. Monthly timestep advances GA by ~4.3w; a 2-week window skips every woman. See `memory/feedback_ga_window_timestep.md`.
- `CondomCounseling` + `ANCBundledPrevention` merged into `DxRiskRedux` (`triggers=(...)` + `trigger_attr='ti_treated'|'ti_tested'`).
- `FSWOutreach` class and `fsw_outreach` kwargs removed as dead code.
- `quick_validate.py`: treatment total lookup fixed to use `new_treated` (dispatch) not `n_treated` (point-in-time active count, sums to 0).

## Task 5.1 (full 700-sim run) status

NOT launched. Blocked pending user acceptance of the ABO artefact above. When ready, `run_scenarios.py` should already work at scale; `aggregate_scenarios.py` produces the K-avg CSV.

## What to commit when the next agent picks up

The Session-2026-07-08 refactors above should be committed in coherent chunks once the ABO artefact direction is set — likely:
1. ANCScreen name-string refactor + `sti.Sim` deep-copy fix
2. `make_sim_parts` refactor
3. GA-window widening
4. `DxRiskRedux` unification
5. FSWOutreach removal
6. `quick_validate.py` fix + comment pruning
