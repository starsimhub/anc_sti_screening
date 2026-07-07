# Design: Port the sti_notification calibration into anc_sti_screening

**Date:** 2026-07-07
**Status:** Design approved by user; awaiting implementation plan.
**Supersedes:** `promise-voi-plan-2.md` (VoI framing dropped by collaborator feedback).

---

## 1. Overview and goals

**Goal.** Use the frozen sti_notification exp-06 calibration ensemble inside
`anc_sti_screening` to quantify how three ANC-STI screening design levers
change adverse birth outcomes (ABO), adverse pregnancy outcomes (APO), DALYs,
and downstream STI/HIV dynamics in Zimbabwe. The three design levers:

1. Number of ANC screens per pregnancy (1 vs 2)
2. Partner notification of ANC-positive women (off vs on)
3. Coverage of ANC screening among ANC attendees (50%, 75%, 90%)

Report point estimates + uncertainty (from the calibration ensemble) under 4
named effect-size assumption sets, including a mandatory "no treatment
effect" pessimistic counterfactual.

**Not doing.** VoI / EVPI / EVPPI / EVSI. Optuna calibration. NMB / WTP
thresholds. Cost-effectiveness planes. Everything decision-theoretic in the
prior `promise-voi-plan-2.md` framing is replaced by simple named-case
sensitivity reporting; that framing didn't land with collaborators.

**Compute envelope (first run).** 5 draws × 5 seeds × 13 scenarios × 4
assumptions = 1,300 sims, each 60 sim-years. Approximately 4 hours wall on
80 workers on the IDM Azure VM.

---

## 2. Repo strategy

1. On `anc_sti_screening/main` at current HEAD: create tag `v0.1` and push.
   This preserves the current Optuna-calibration codebase for rollback.
2. Branch `port/stinotif-calibration` from main.
3. On that branch, do the model port (§3), the ANC infrastructure work
   (§3), and the runs (§6). Old Optuna machinery gets deleted; old plot
   scripts get adapted.
4. When the port passes the reproducibility smoke test (§6, Phase 2) and
   the first full run's figures are stable, PR the branch to main. Main
   becomes the new version.
5. `promise-voi-plan-2.md` is superseded by this spec; the file stays in
   git history for reference (do not hard-delete).

---

## 3. Model port

### 3.1 stisim pin

Pin to stisim `rc1.5.9` with `fix/ng-tx` merged in (the branch state the
user established prior to this design being written). This is one release
ahead of sti_notification's calibration base (`fix/ng-tx@731bc1d` off
`rc1.5.8`), but with `fix/ng-tx` merged in, all sti_notification-relevant
changes are present.

Reproducibility of the calibration under this pin is verified by the
Phase 2 smoke test (§6) before any scenario runs.

### 3.2 Files copied from sti_notification

| From sti_notification | To anc_sti_screening | Notes |
| --- | --- | --- |
| `model.py` | `model.py` (replaces existing) | Full 7-disease `make_sim` — HIV + NG + CT + TV + BV + syph + GUDPlaceholder. `StructuredSexual` with FSW/client populations, `PriorPartners`, `MaternalNet`. |
| `hiv_model.py` | `hiv_model.py` | HIV disease + HIV interventions (testing, ART, VMMC, PrEP). |
| `connectors.py` | `connectors.py` (replaces existing) | `sti_fetal` connector covering NG/CT/TV **and syph** (syph has `ptb_shift=4.0w`, `growth_penalty=0.12` — largest fetal effect of the seven STIs). |
| `data/` (full folder) | `data/` (replaces overlapping files) | Includes syph inputs (`init_prev_syph.csv`, `init_prev_latent_syph.csv`, `zimbabwe_syph_data.csv`), time-varying symptomatic-test-prob CSVs (`symp_test_prob_{soc,concentrated}.csv`), `syph_dx.csv`, `zimbabwe_migration.csv`. |
| `experiments/06_2026-06-24_kseed_calibration/outputs/draws_used.csv` | `data/calibration_draws.csv` | 5 rows for the first run (top-5 by GoF from the 10-row file). |
| Analyzer classes from `analyzers.py`: `SyphTransmissionEvents`, `CareTimingAnalyzer`, plus the `sw_stats` / `coinfection_stats` wiring | Merged into `analyzers.py` alongside existing ANC analyzers | Analyzers scoped to what the model needs; drop `syph_hiv_trep`/`nontrep` variants if not used by any figure. |
| The `set_pars_local` routing helper (currently in sti_notification's `run.py` / calibration pipeline) | New helper: `apply_draw.py`, or in `run_scenarios.py` | Takes a `calibration_draws.csv` row and applies the 17 params. Handles the log-prefixed columns (`log_syph.beta_m2f`, `log_ng.beta_m2f`, etc.) via `10**value`. |

### 3.3 Files NOT copied from sti_notification

- **`interventions.py`** — sti_notification's interventions are
  undertreatment-project-specific (`SyndromicPN`, `POCPN`, FSW-outreach,
  POC ulcer-channel product swap). anc_sti_screening keeps its own
  `interventions.py` (see §3.5).
- **`scenarios.py`** and **`run_scenarios.py`** — sti_notification's
  scenario factorial is the wrong shape for ANC. Rewrite fresh (§3.6).
- **`pn.py`** — PR 505 landed in stisim rc1.5.8+. Use
  `sti.PartnerNotification` from stisim directly; do not port the local
  mirror.
- The `experiments/` folder and its history stay in sti_notification.

### 3.4 Files deleted from anc_sti_screening on the port branch

- `run_calibrations.py` — Optuna infrastructure, superseded.
- `priors.py` — Optuna prior distributions, superseded.
- `run_msim.py` — top-N-parameter-sets msim, superseded by ensemble scenario runner.
- Existing `model.py` and `connectors.py` — replaced by sti_notification versions.

### 3.5 anc_sti_screening files adapted (not replaced)

- **`interventions.py`**
  - Keep: `ANCScreen`, `SyndromicMgmt`, `make_testing`.
  - Delete: `STIPartnerNotification` class.
  - Add: `ANCPN(sti.PartnerNotification)` subclass whose eligibility
    triggers on ANC-screen positives. Eligible-partner rate lookup uses
    the calibrated `pn_rates` machinery (edge-stratified marital vs
    casual) — same base class sti_notification uses.
- **`analyzers.py`**
  - Keep: `pregnancy_sti_stats`, `birth_outcome_dalys`,
    `intervention_costs`, `total_symptomatic`.
  - Merge in the ported sti_notification analyzers.
  - Update `birth_outcome_dalys` to include syph and stillbirths as ABO
    contributors.
- **Plot scripts** — adapt to consume the ensemble parquet outputs (not
  Optuna trial outputs):
  - `plot_hiv_calibration.py` — overlay HIV calibration bands on the
    ensemble.
  - `plot_sti_epi.py` — STI prevalence panels by age/sex + timeseries,
    ensemble bands.
  - `plot_network.py` — network validation figure.
  - `plot_fig4_priors_posteriors.py` — becomes ensemble-marginals figure
    (17 calibrated params from the ensemble; effect-size assumption
    values overlaid as vertical lines / bands).

### 3.6 anc_sti_screening files built fresh

- **`scenarios.py`** — see §4.
- **`run_scenarios.py`** — dispatches the (draw × seed × scenario ×
  assumption) grid. Writes JSONL per sim.
- **Aggregation script** — reads JSONL, writes K=5-averaged parquet
  (timeseries + snapshots) + a scalar `kavg.csv` for scenario endpoints.
  Follows sti_notification's aggregation pattern.

---

## 4. Scenario space and effect-size assumptions

Both live in a single `scenarios.py` file with a top-of-file docstring
explaining the two-axis design. This keeps the whole analysis grid
visible in one place.

### 4.1 Intervention scenarios (13 cells)

| # | cell_id | # screens | Panel | PN | Coverage |
|---|---|---|---|---|---|
| 1 | `soc` | 0 | — (syndromic + syph RPR only) | off | — |
| 2 | `anc_1screen_50cov` | 1 (enrolment ≤24w) | NG+CT+TV | off | 50% |
| 3 | `anc_1screen_75cov` | 1 | NG+CT+TV | off | 75% |
| 4 | `anc_1screen_90cov` | 1 | NG+CT+TV | off | 90% |
| 5 | `anc_1screen_50cov_pn` | 1 | NG+CT+TV | on | 50% |
| 6 | `anc_1screen_75cov_pn` | 1 | NG+CT+TV | on | 75% |
| 7 | `anc_1screen_90cov_pn` | 1 | NG+CT+TV | on | 90% |
| 8 | `anc_2screen_50cov` | 2 (enrol + 3rd tri 32–34w) | NG+CT+TV | off | 50% |
| 9 | `anc_2screen_75cov` | 2 | NG+CT+TV | off | 75% |
| 10 | `anc_2screen_90cov` | 2 | NG+CT+TV | off | 90% |
| 11 | `anc_2screen_50cov_pn` | 2 | NG+CT+TV | on | 50% |
| 12 | `anc_2screen_75cov_pn` | 2 | NG+CT+TV | on | 75% |
| 13 | `anc_2screen_90cov_pn` | 2 | NG+CT+TV | on | 90% |

**Constants across all cells:** syph RPR at ANC on (Zim SOC); HIV
testing/ART/VMMC/PrEP; syndromic management (VDS/UDS) in general
population; POC test sensitivity/specificity 95/95; K=5 seeds per
(draw, cell); simulation runs 1985–2045, arms diverge at 2028, reporting
window 2028–2045.

### 4.2 Effect-size assumption sets (4)

Each set is a dict of FetalHealth parameters. Applied to `FetalHealth` at
sim construction time.

| Assumption | Delivery shifts (weeks) | Growth penalties (fraction) | Reversibility |
|---|---|---|---|
| **`no_treatment_effect`** | Central (medians): NG=2.0, CT=1.5, TV=1.0, syph=4.0 | Central: NG=0.08, CT=0.03, TV=0.03, syph=0.12 | Ratchet — `tx_residual_*_early = tx_residual_*_late = 1.0` for both timing and growth. Treatment never reverses damage. |
| **`central_reversible`** (base) | Central | Central | Central: `tx_residual_growth_early=0.33`, `tx_residual_growth_late=0.50`, `tx_residual_timing_early=0.50`, `tx_residual_timing_late=0.71`. |
| **`weak_effects`** | Lower CIs from meta-analyses (Vallely 2021, He 2020, Silver 2014; syph from clinical literature) | Lower CIs | Central reversibility |
| **`strong_effects`** | Upper CIs from meta-analyses | Upper CIs | Central reversibility |

Exact CI-derived values will be fixed at implementation time in
`scenarios.py`.

**Design rationale for the set.** `no_treatment_effect` is the pessimistic
counterfactual — if treatment never reverses damage, is screening still
worth doing (via case detection alone)? Weak / central / strong bracket
effect-size uncertainty holding reversibility at central. Central +
reversible anchors reporting. 4 sets keeps the total sim count tractable
while covering both key uncertainties.

### 4.3 Ensemble propagation

For each row in `data/calibration_draws.csv`:

- 17 columns give the calibrated parameter values (some log-transformed
  — column names prefixed with `log_`).
- For each of K=5 seeds (`seed = draw_idx * 1000 + sub_idx`, `sub_idx ∈
  [0, 5)`): apply the draw's params to a fresh sim via `set_pars_local`,
  run 1985–2045, extract results.
- Seed convention matches sti_notification's calibration exactly, so
  SOC in these scenarios reproduces sti_notification's exp-06 SOC bit-for-bit.

---

## 5. Metrics, analyzers, and output

### 5.1 Metrics tracked per sim

**Adverse birth outcomes (ABO)** — from `starsim.Pregnancy` +
`FetalHealth`:

- `n_preterm` (<37w GA), `n_very_preterm` (<32w GA), `preterm_rate`
- `n_lbw` (<2500g), `lbw_rate`, `n_sga` (<10th percentile for GA)
- `n_stillbirths` (loss ≥20w GA) — native to `Pregnancy`

**Adverse pregnancy outcomes (APO)** — from `pregnancy_sti_stats` +
custom:

- STI prevalence at conception and at delivery, per pathogen
- Incident STI infections during pregnancy, per pathogen
- Incident HIV infections during pregnancy (MTCT precursor)
- Per-pregnancy count of syndromic + ANC treatment events

**DALYs** — from `birth_outcome_dalys`:

- YLD per PTB and per LBW (with the double-counting rule: PTB+LBW
  co-occurrence gets PTB weight only)
- Cumulative DALYs 2028→2045

**Programmatic outputs** — from `intervention_costs` + additions:

- Tests administered per pregnancy
- Treatments administered, per pathogen, per pregnancy
- Partners notified and treated
- False-positive treatments (specificity-driven)
- Per-woman ANC visit count

**Costs (descriptive)** — placeholder values from `promise-voi-plan-2.md`
§3f, clearly flagged as placeholder in output:

- Total cost per arm (screening + treatment + PN + adverse-outcome mgmt)
- Cost per DALY averted (vs SOC)
- Cost per 1000 women screened

**Epidemiological validation** — ensemble sanity across the runs:

- HIV prev 15–49, per year
- NG/CT/TV/syph prev, per year, sex-stratified
- FSW prev, each disease
- Syph stage shares (primary / secondary / latent)

### 5.2 Output file structure

Follows sti_notification's pattern.

| File | Committed? | Purpose |
|---|---|---|
| `results/scenarios.jsonl` | no (VM only, regenerable) | One JSON row per sim: `{draw, seed, scenario_id, assumption_id, scalars: {...}}`. Raw archive; source of everything below. |
| `results/scenarios_timeseries.parquet` | no (VM only) | K=5-averaged per-year timeseries, keyed by `(scenario_id, assumption_id, draw, year, disease, metric)`. Source for time-series plots. |
| `results/scenarios_snapshots.parquet` | no (VM only) | K=5-averaged age×sex snapshots at 2028, 2035, 2040, 2045. Source for age-structured figures. |
| `results/scenarios.kavg.csv` | yes | K=5-averaged scalar table, one row per `(scenario_id, assumption_id, draw)`. Small; supports scalar/bar plots without needing the parquet. |

---

## 6. Compute plan and phasing

### 6.1 Compute

- **Machine:** IDM Azure VM, 120 cores available; 80 workers used.
- **Sims:** 1,300 (5 × 5 × 13 × 4).
- **Sim-years:** ~78,000. Each sim runs 1985–2045.
- **Expected wall:** ~4 hours at the sti_notification exp-06 rate.
- **Population size:** 10,000 agents (matches sti_notification calibration base).
- **Burn-in strategy:** Naive. Each sim runs 1985–2045 independently
  under its assumption set. This is scientifically correct: each
  effect-size assumption implies a different accumulated historical
  burden of ABO/APO, which is part of what the analysis measures. Shared
  burn-in would falsely equalize pre-2028 histories across assumption
  sets.

### 6.2 Phased implementation

**Phase 1 — Repo prep** (~1 session, no sim runs)

1. Tag `v0.1` on `anc_sti_screening/main`; push.
2. Branch `port/stinotif-calibration` from main.
3. Copy files per §3.2; delete files per §3.4.
4. Verify `python -c "from model import make_sim"` succeeds.

**Phase 2 — Reproducibility smoke test** (~1 session, ~30 min VM run)

1. Port `set_pars_local` and load `calibration_draws.csv`.
2. Run 3 draws × 5 seeds, 1985 → 2020, no ANC interventions attached.
3. Compare per-draw K=5-mean scalars (HIV prev 15–49, NG/CT/TV/syph
   prevalences, syph stage shares) to sti_notification's
   `experiments/06_2026-06-24_kseed_calibration/outputs/per_draw_means.csv`
   for the same draws.
4. **Go/no-go gate.** Any material divergence blocks further work until
   diagnosed. Expected: exact match to numerical noise (identical model
   + identical stisim + identical params + identical seed).

**Phase 3 — ANC infrastructure** (~1–2 sessions, no sim runs)

1. Write `scenarios.py` with `INTERVENTION_SCENARIOS` and
   `EFFECT_SIZE_ASSUMPTIONS` dicts.
2. Adapt `ANCScreen` to consume scenario-cell params (`screen_prob`,
   `ga_min`/`ga_max` windows). Verify parameter propagation.
3. Replace `STIPartnerNotification` with `ANCPN(sti.PartnerNotification)`.
4. Wire effect-size assumptions into FetalHealth (currently hardcoded
   per the plan-doc TODO — needs to accept `ptb_shift`, `growth_penalty`,
   and the four `tx_residual_*` params at construction).
5. Update `birth_outcome_dalys` to include syph and stillbirths.

**Phase 4 — Small-N end-to-end** (~1 session, ~30 min VM run)

1. Run 1 draw × 1 seed × 13 scenarios × 4 assumptions = 52 sims, full
   1985–2045.
2. Verify aggregation: JSONL rows correct shape, timeseries parquet
   builds, kavg CSV correct.
3. Spot-check: SOC gives ~0 ANC tests per woman; intervention arms give
   expected coverage-scaled test counts; PN-on cells produce partner
   treatment counts; `no_treatment_effect` assumption gives higher
   residual damage than `central_reversible` for the same intervention.

**Phase 5 — Full first run** (~4 h wall)

1. Run the full 1,300-sim grid on 80 workers.
2. Aggregate outputs to parquet + kavg CSV.

**Phase 6 — Figures and reporting** (session-by-session, iterative)

1. Adapt existing plot scripts to consume ensemble outputs.
2. New figures:
   - ABO / APO / DALYs by (scenario × assumption), difference-from-SOC bars.
   - Cost tables (placeholders flagged inline).
   - Cumulative DALYs averted vs SOC, per assumption set.
3. Optional: ensemble-marginals figure showing the 17 calibrated
   parameters' distribution across the 5-draw ensemble, with effect-size
   assumption values overlaid.

**Phase 7 — PR back to main**

1. Once collaborators are happy with figures, PR
   `port/stinotif-calibration` → `main`.
2. Old `v0.1` tag stays as the pre-port rollback point.

---

## 7. Open items deferred

Items intentionally out of scope for the first run, to be revisited if
collaborators ask:

- **Real Zimbabwe cost data.** Costs stay as placeholders per
  `promise-voi-plan-2.md` §3f. If collaborators want cost-tables to be
  policy-actionable, revisit before Phase 6 reporting.
- **Sensitivity to test performance.** Fixed at 95/95 for now. Vary
  in a follow-up if collaborators ask about cheaper / less-accurate
  tests.
- **Alternative screen timings** (e.g., very-early 12–16w screen).
  Fixed at PROMISE-design timings for the first run.
- **BV screening as a panel option.** Not in scope; BV has no partner-tx
  pathway and different pathophysiology.
- **Shared burn-in optimization.** Naive is fine at 1,300 sims. If a
  follow-up scales up to (say) 30 draws × 10 assumptions × 100
  scenarios, shared burn-in via `sim.run(until=)` + `sc.dcp` becomes
  worth the engineering; not before.
- **Recalibration on rc1.5.9.** Only if Phase 2 smoke test fails to
  reproduce sti_notification exp-06 numbers. Expensive (~10 hr on 60
  workers per sti_notification exp-06).

---

## 8. Explicit non-goals

- Full VoI machinery (EVPI/EVPPI/EVSI). Dropped after collaborator
  feedback that these framings didn't land.
- Optuna-based calibration. Superseded by the sti_notification exp-06
  ensemble.
- NMB / WTP-threshold analysis. Superseded by direct ABO/APO/DALY
  reporting.
- Congenital syphilis modeling as its own tracked outcome. Syph
  contributes to ABO via the same `sti_fetal` pathway as NG/CT/TV
  (with the largest delivery shift and growth penalty of the four).
- Any new model calibration. The exp-06 ensemble is the calibrated
  baseline; this project consumes it.
