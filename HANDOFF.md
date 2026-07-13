# Handoff — anc_sti_screening (2026-07-13)

Branch: `port/stinotif-calibration`. Head: `a4b7522 fix(SyphilisANCTimer): floor scheduled ti so equality-match to sim.ti fires`. Behind push by 27 commits.

## Read this first

**The user is unconvinced by the current results.** Do NOT rush into interpretation or write-up. Approach the next steps with skepticism about the model outputs. Specific reasons for skepticism, from the 2026-07-12–13 session:

1. The pre-fix K=5 smoke reported ~43% PTB reduction under central regime for 1-screen 90%. After fixing a real bug in `SyphilisANCTimer` (see below), the same reduction dropped to **~13%**. The magnitudes are noticeably smaller than the user expected for a real intervention.
2. The 1-vs-2 screen delta is modest (2.6–4.3 percentage points across regimes). Post-fix this is larger than pre-fix (~1 pp) but still small.
3. Stochastic syph extinction at 10k agents is severe — 3 of 5 K-seeds went extinct in the SOC diagnostic; the K=5 average pools hot and extinct sims.
4. `n_stillbirths = 0` across every row of the 700-sim output. There is a separate counter bug that has not been fixed yet (see below).
5. The absolute-syphilis-prevalence structural ceiling in the model (documented in the parent-repo CLAUDE.md) means the model overpredicts syph seroprevalence vs empirical ZIMPHIA. All syph results here should be treated as relative-effect contrasts, not absolute calibration.

Any handover-analysis or narrative-writing steps should confront these caveats before presenting anything.

## Current state

- **700-sim grid complete.** 5 draws × 5 seeds × 7 scenarios × 4 assumptions = 700 sims. Wall time 15,403s (4h 17m) on 80 workers. Zero errors. Output: `results/scenarios.jsonl` (700 rows, gitignored) and `results/scenarios.kavg.csv` (140 rows, committed).
- **SyphilisANCTimer floor-fix committed** (`a4b7522`). Pre-fix, ANC syph testing fired ~99% below expected (fractional-ti vs integer-`self.ti` equality mismatch). Post-fix, ANC RDT fires ~2M pop-scaled tests/2020–2025 in SOC, matching ~70% per-pregnancy coverage from `ANC_PROBS_REALISTIC`.
- **Syph-in-pregnancy diagnostic** for SOC over three time windows rendered (K=2 hot-seed avg from a K=5 SOC-only sim):
  - `figures/syph_pregnancy_diagnostic_2012_2020_hot.png`
  - `figures/syph_pregnancy_diagnostic_2020_2025_hot.png`
  - `figures/syph_pregnancy_diagnostic_2025_2045_hot.png`

## 700-sim headline (median across 5 draws, K=5 seed-averaged)

PTB reduction from SOC, %:

| Scenario | no_treatment | weak | central | strong |
|---|---:|---:|---:|---:|
| 1-screen 50% | 1.0 | 2.9 | 12.2 | 12.2 |
| 1-screen 75% | 1.1 | 3.0 | 12.9 | 13.0 |
| 1-screen 90% | 1.0 | 3.0 | 13.0 | 13.2 |
| 2-screen 50% | 3.0 | 3.9 | 14.9 | 16.3 |
| 2-screen 75% | 3.1 | 4.2 | 15.4 | 17.3 |
| 2-screen 90% | 3.1 | 4.2 | 15.6 | 17.5 |

Coverage 50→90 gives only ~1 pp incremental. 2-screen vs 1-screen adds 2.6–4.3 pp. Weak / no-treatment regimes correctly show near-null. **User is skeptical of the small absolute magnitudes** — the 3rd-tri screen delta is what they specifically flagged as smaller than expected.

## Known open issues

### 1. Prenatal-stillbirth counter always reads 0

Root cause: `Pregnancy.add_delivery_callback(fn)` only fires for live births; syph-driven stillbirths kill the fetus prenatally via `sim.people.request_death(fetus)` → `Pregnancy.finish_step → process_prenatal_deaths → step_die(mother)`, which sets `pregnant=False, ti_delivery=NaN`. The mother never reaches the delivery block, so my analyzer's callback never sees the stillborn.

Both `birth_outcome_dalys.n_stillbirths` and `birth_outcome_attribution.n_stillbirth_syph` use this callback pathway, so both are broken.

`preg.results['stillbirths']` (aggregate, any cause) does record correctly (~150k pop-scaled in a hot 45yr sim). Fix path options in memory `feedback_delivery_callback_misses_stillbirths.md`.

### 2. Stochastic syph extinction at 10k agents

Under SOC, only 2 of 5 K-seeds at draw 343 sustained. Draws with lower `log_syph.beta_m2f` (e.g. draw 263) went extinct at every seed. The K=5 average pooling hot + extinct sims may not reflect the intended "sustaining ensemble" behaviour.

Options for next agent: (a) scale up to 20k agents (halves extinction rate empirically); (b) filter to sustaining seeds only and report conditional-on-sustaining; (c) accept as model property, note prominently in write-up.

### 3. `run_scenarios.py` doesn't dump timeseries or snapshots

Only scalar summary written to JSONL. Plan doc's Task 5.2 (per-year timeseries parquet + age × sex snapshot parquet) not yet implemented in `run_scenarios.py` or `aggregate_scenarios.py`. If timeseries plots are needed, this needs adding + rerun.

### 4. Congenital-syph outcome distribution semantics

`syph.cs_outcome=4` ("normal") is 67% of MTCT events in stisim's model. These are counted as "vertical transmission events" but produce clinically normal births. Requires care when framing "N MTCT events" vs "N clinical congenital syphilis cases" — they differ by ~3x.

## SOC-diagnostic key findings (from figures)

Same instrumented K=5 sim, per-pregnancy trajectory recorded via `sim.loop.insert(...)` after `syph_tx.update_results`:

- Under SOC, ANC syph testing coverage per pregnancy: **58-74%** across time windows (matches expected `anc_probs` 0.70–0.85 with some drop from timing / eligibility).
- The dominant miss reason for adverse outcomes is **tested-but-negative** (44-58% of miss cases) — RPR misses the mother due to window period or waning latent-titre.
- "Never tested during pregnancy": **26-45%** — grows over time (unexplained; may be extinction dilution artefact).
- Congenital-syph cases where mother was known-positive during pregnancy: **11-17%**. Even under SOC, most clinical congenital syph slips through.
- Bottom line: **detection is the bottleneck, not treatment.**

## Recent commits (2026-07-11 → 2026-07-13)

```
a4b7522 fix(SyphilisANCTimer): floor scheduled ti so equality-match fires
b61b10e tune: weak_effects near-null STI harm; K=5 for ABO grid deliverable
c418341 feat: ABO artefact — 3 scenarios × 4 effect-size regimes grid
c79ad52 feat: SyphilisANCTimer schedules one visit per pregnancy per configured window
264c2e9 feat: Phase 4.5 ABO artefact — small-N driver + attribution figure
6be71f3 fix(analyzers): birth_outcome_dalys uses exact per-birth PTB/LBW flags
39cfabf feat(analyzers): birth_outcome_attribution — per-disease ABO reconciliation
f7f1b27 feat(connectors): sti_fetal records per-pregnancy exposure per disease
```

## Uncommitted work (2026-07-13, needs curation)

**Diagnostic scripts** — all one-shot instrumentation used during the SyphilisANCTimer investigation:
- `diag_syph_k5.py` — main K=5 SOC instrumentation with loop.insert. **Recommend commit** (reference for future diagnostics).
- `plot_syph_diagnostic.py` — 3-panel figure generator, CLI arg for window. **Recommend commit**.
- `diag_syph_hunt.py` — hot-seed hunt. **Recommend commit** (small, useful helper).
- `diag_syph_pregnancy.py` — earlier single-sim variant, superseded by k5. **Recommend drop**.
- `diag_syph_reanalyse.py` — one-off pickle re-analysis. **Recommend drop**.
- `diag_2screen.py`, `diag_test_sources.py`, `diag_timer.py` — one-off debug scripts used to locate the fractional-ti bug. **Recommend drop** (bug is now documented in memory; scripts add clutter).

**Modified files**:
- `results/abo_attribution.csv`, `results/abo_totals.csv` — pre-fix outputs from `abo_smoke.py`. Small (~50 KB total). May want to REGENERATE post-fix before committing, or discard.

**Untracked results/**:
- `results/diag_k5_soc.pkl` (17 MB) — K=5 SOC per-pregnancy records. **Gitignored via new `*.pkl` rule.**
- `results/diag_records.pkl` (580 KB) — single-sim records, superseded. Gitignored.
- `results/scenarios.jsonl` (276 KB) — 700-sim raw output. Gitignored (regenerable).
- `results/scenarios.kavg.csv` (28 KB) — K-averaged summary of the 700 sims. **Recommend commit** (small, useful, matches the parent-repo pattern).
- `results/hot_seed.txt` (33 B) — trivial marker file, seed=343000. **Recommend drop** or commit for reproducibility.

## Suggested next steps for the next agent

1. **Confront the skepticism first.** Read the "Read this first" section carefully. Do NOT reflexively write up until the user's concerns have been either resolved or explicitly acknowledged in the framing.
2. **Consider fixing the stillbirth counter.** The two obvious paths are (a) monkey-patch `process_prenatal_deaths`, (b) add a `add_prenatal_death_callback` to starsim (would require an upstream branch). Once fixed, need to re-run at least the SOC-only diagnostic to see if the story changes.
3. **Understand the small 3rd-tri screen benefit.** Is it real or an artefact of the model design (tx_residual_growth[tri3]=0.60 means 60% of damage is locked in by tri3)? Consider a sensitivity run.
4. **Check the extinction-vs-sustaining split of the ensemble.** For each of the 5 draws in the current run, report whether syph sustained. If most are extinct, the "K=5 median" story is misleading.
5. **Timeseries plots** — if needed for the write-up, extend `run_scenarios.py::extract_scalars` to dump per-year TS parquets, per plan doc Task 5.2.

## Files added to memory this session

- `feedback_delivery_callback_misses_stillbirths.md`
- `feedback_ti_scheduling_integer_match.md`

## Environment

- Conda env: `starsim`
- stisim: `fix/ng-tx@731bc1d` (pinned)
- Run this on the IDM Azure VM (`agouti120` or similar). Local Mac cannot run the 700-sim grid.
