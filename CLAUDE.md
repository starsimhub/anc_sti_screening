# anc_sti_screening — Next Steps

## Session Summary (2026-04-30)

Completed smoke test infrastructure:
- **Created `smoke_test.py`** — validates SOC vs. twice scenarios with birth_outcome_dalys and intervention_costs analyzers
- **Fixed analyzer scaling bugs** — count fields (n_deliveries, n_ptb, n_lbw, n_screened, n_treated_*) now have `scale=False`
- **Documented in SUMMARY.md** — experiments/01_coverage_check/SUMMARY.md logs findings and known issues

Results:
- Both analyzers produce correct unscaled counts
- Screening cost differential working correctly (twice: +$7722)
- Ready to proceed to Step 2 (calibration)

## Priority: Step 2 — Calibration

**Next task:** Run `run_calibrations.py` on HPC/VM with N_TRIALS ~500-2000 to generate posterior epi parameters.

1. **Setup:**
   - Verify `run_calibrations.py` is ready (loads calib_pars, runs Optuna)
   - Ensure Zimbabwe prevalence targets are correct (check with PROMISE team if possible)
   
2. **Run calibration:**
   - Start with N_TRIALS=500, N_WORKERS=8-16 on a VM
   - Monitor convergence of top parameter sets
   
3. **Save posteriors:**
   - Extract top ~200 parameter sets (ranked by goodness-of-fit)
   - Save as `results/zimbabwe_pars.df` (load as pickle in run_voi.py)

## Known issues (not blockers)

- **Birth outcome concordance:** SOC and twice show identical adverse birth outcomes. FetalHealth connector may need parameter initialization or integration check.
- **Treatment count discrepancy:** n_treated same across SOC and twice; needs investigation in intervention code.

## Files to commit after this session

- analyzers.py (scale=False fixes)
- smoke_test.py
- experiments/01_coverage_check/SUMMARY.md

## Session context

- Branch: `upstream-fetal-health`
- Stack: stisim rc1.5.4, starsim 3.3.2 (age-ranges)
- Working directory: /Users/robynstuart/gf/anc_sti_screening
