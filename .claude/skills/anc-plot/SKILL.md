---
name: anc-plot
description: Use when creating, debugging, or modifying figures in the anc_sti_screening repo — including the four manuscript figures (NMB histograms, CE plane, EVPPI tornado, priors/posteriors), utility functions, and any new exploratory plots.
context: fork
---

# ANC STI Screening — Plotting

You are helping produce figures for the PROMISE trial VoI analysis. All figure scripts live in the repo root and pull from `results/` using `sc.loadobj()`.

## Figure inventory

| Script | Output | Input | Key message |
|--------|--------|-------|-------------|
| `plot_fig1_nmb.py` | `fig1_nmb.png` | `voi_draws.df` | P(CE) and E[NMB] at multiple WTPs — histograms, green/red |
| `plot_fig2_ceplane.py` | `fig2_ceplane.png` | `voi_draws.df` | CE scatter with WTP threshold lines |
| `plot_fig3_evppi.py` | `fig3_evppi.png` | `voi_draws.df` | EVPPI tornado by parameter group (Strong 2014 GBR) |
| `plot_fig4_priors_posteriors.py` | `fig4_priors_posteriors.png` | `voi_draws.df` + `zimbabwe_pars.df` | Prior vs posterior marginals for calibrated + VoI params |
| `plot_sti_epi.py` | epidemiological validation | `results/*.df` | Calibration fit plots |
| `plot_hiv_calibration.py` | HIV calibration | `results/*.df` | HIV-specific calibration |
| `plot_ceac.py` | CEAC curve | `voi_draws.df` | P(CE) as continuous function of WTP |
| `plot_evppi_wtps.py` | EVPPI vs WTP | `voi_draws.df` | How EVPPI varies across WTP thresholds |

## Key conventions

**Font and style** — always call `set_font()` from `utils.py` before plotting:
```python
from utils import set_font
set_font(size=18)  # uses Libertinus Sans from assets/
```

**Saving** — use `sc.savefig()` not `plt.savefig()`:
```python
sc.savefig(f'{FIGURES_DIR}/figN_name.png', dpi=200)
```

**Color palette**:
```python
GREEN = '#2ca02c'      # CE (NMB > 0)
RED   = '#d62728'      # Not CE
PRIOR_COLOR     = '#b0c4de'   # Uniform priors (light steel blue)
POSTERIOR_COLOR = '#2c5f8a'   # Posteriors (dark blue)
BIRTH_COLOR     = '#5a8c6f'   # Birth outcome params (sage green)
COST_COLOR      = '#c4956a'   # Cost params (tan)
# EVPPI group colors:
EPI_COLOR       = '#4a90d9'   # Network/transmission params
BIRTH_OUT_COLOR = '#5a8c6f'   # Delivery/growth/reversibility params
COST_GROUP_COLOR= '#c4956a'   # Cost parameters
```

**Panel labels** — bold A/B/C in top-left:
```python
for i, ax in enumerate(axes):
    ax.text(-0.08, 1.06, chr(65 + i), transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='top')
```

**Loading results**:
```python
import sciris as sc
draws_df = sc.loadobj('results/voi_draws.df')      # per-draw NMB, DALYs, costs, params
pars_df  = sc.loadobj('results/zimbabwe_pars.df')  # posteriors from calibration
```

## Key columns in `voi_draws.df`

| Column | Description |
|--------|-------------|
| `delta_dalys` | DALYs averted (intervention minus SOC) |
| `delta_costs` | Incremental cost |
| `nmb_{wtp}` | Pre-computed NMB at each WTP threshold |
| `ptb_shift_ng/ct/tv` | Sampled delivery timing shift (weeks) |
| `growth_penalty_ng/ct/tv` | Sampled birth weight penalty |
| `tx_residual_growth_tri{1,2,3}` | Sampled treatment reversibility |
| `tx_residual_timing_tri{1,2,3}` | Sampled timing reversibility |
| `cost_*` | Sampled cost parameters |
| Epi params | e.g. `ng.beta_m2f`, `structuredsexual.prop_f0` |

## EVPPI computation (fig3)

Uses nonparametric GBR regression (Strong et al. 2014 via scikit-learn):
```
EVPPI(φ) = E[max(E[NMB|φ], 0)] − max(E[NMB], 0)
```
Parameters are grouped (`PARAM_GROUPS` dict in `plot_fig3_evppi.py`). Cross-validated (K-fold) to avoid overfitting. Returns a scalar per group per WTP.

## CEAC (plot_ceac.py)

Computes P(CE) = mean(NMB > 0) at each WTP. Sweep `WTP_THRESHOLDS` from `run_voi.py`:
```python
wtp_grid = np.linspace(0, 5000, 200)
prob_ce  = [np.mean(wtp * draws_df['delta_dalys'] - draws_df['delta_costs'] > 0) for wtp in wtp_grid]
```

## Adding a new figure

1. Use `set_font(size=18)` and `sc.savefig(..., dpi=200)`.
2. Load from `results/` with `sc.loadobj()` — don't rerun simulations in plot scripts.
3. Keep the function signature `def plot(draws_df=None, ...)` with lazy loading so scripts can be called programmatically or standalone.
4. Save to `figures/` directory.
5. No inline simulation runs — plot scripts are pure visualization.

## Uncertainty bands (scenario plots)

Use the `percentile_pairs` from `utils.py` for fan charts:
```python
from utils import percentile_pairs, percentiles
# percentile_pairs = [[.01,.99],[.1,.9],[.25,.75]]
# Draw bands from outer to inner, decreasing alpha
```
