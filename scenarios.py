"""
Analysis grid: intervention scenarios × effect-size assumptions.

Two dimensions cross to produce the cell grid this project runs:

    INTERVENTION_SCENARIOS — what we DO
        7 named cells: SOC + {1-screen, 2-screen} × {50%, 75%, 90% coverage}.
        PN dimension dropped from first run per user directive.
        All arms use NG+CT+TV panel, test 95/95, syph RPR at ANC always on.

    EFFECT_SIZE_ASSUMPTIONS — what we BELIEVE
        4 named assumption bundles spanning two uncertainties:
          * Do STIs materially harm birth outcomes at all?
          * Does treatment during pregnancy reverse damage?
        Meta-analysis-derived CIs (Vallely 2021, He 2020, Silver 2014).

Every sim runs at a (scenario × assumption) cell, with calibrated
parameters coming from a draw in `data/calibration_draws.csv`.
K=5 seeds per (draw, scenario, assumption).
"""

from __future__ import annotations

import sciris as sc


# ────────────────────────────────────────────────────────────────────
# Intervention scenarios — 7 cells (PN dropped from first run)
# ────────────────────────────────────────────────────────────────────
INTERVENTION_SCENARIOS = sc.objdict()

INTERVENTION_SCENARIOS['soc'] = dict(
    label='SOC (syndromic + syph RPR only)',
    n_screens=0, coverage=0.0,
)

for cov in (0.50, 0.75, 0.90):
    cid = f'anc_1screen_{int(cov*100)}cov'
    INTERVENTION_SCENARIOS[cid] = dict(label=cid, n_screens=1, coverage=cov)

for cov in (0.50, 0.75, 0.90):
    cid = f'anc_2screen_{int(cov*100)}cov'
    INTERVENTION_SCENARIOS[cid] = dict(label=cid, n_screens=2, coverage=cov)

assert len(INTERVENTION_SCENARIOS) == 7


# ────────────────────────────────────────────────────────────────────
# Effect-size assumptions — 4 named cases
# ────────────────────────────────────────────────────────────────────
EFFECT_SIZE_ASSUMPTIONS = sc.objdict()

EFFECT_SIZE_ASSUMPTIONS['no_treatment_effect'] = dict(
    label='No treatment effect (ratchet)',
    ptb_shift_mean=sc.objdict(ng=2.0, ct=1.5, tv=1.0, syph=4.0),
    growth_penalty=sc.objdict(ng=0.08, ct=0.03, tv=0.03, syph=0.12),
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

assert len(EFFECT_SIZE_ASSUMPTIONS) == 4


# ────────────────────────────────────────────────────────────────────
# Sim factory
# ────────────────────────────────────────────────────────────────────
def build_scenario_sim(seed, scenario_id, assumption_id, draw_row,
                       start=1985, stop=2045, n_agents=10_000,
                       bundled_prevention=False):
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
        starsim.Sim: initialised sim, ready to run.
    """
    import pandas as pd
    import stisim as sti
    from model import make_sim_parts
    from apply_draw import row_to_sim_pars, set_pars_local
    from interventions import ANC_PROBS_REALISTIC
    from analyzers import birth_outcome_dalys

    cell = INTERVENTION_SCENARIOS[scenario_id]
    assumption = EFFECT_SIZE_ASSUMPTIONS[assumption_id]

    # Load the symptomatic-test-prob for syph testing (calibrated dep)
    from pathlib import Path
    repo = Path(__file__).resolve().parent
    symp_test = pd.read_csv(repo / 'data' / 'symp_test_prob_concentrated.csv')

    # Build the standard sim parts (with FetalHealth on)
    parts = make_sim_parts(
        seed=seed, n_agents=n_agents, start=start, stop=stop,
        which='all', fetal_health=True, verbose=-1,
        syph_symp_test_prob=symp_test,
        syph_anc_probs=ANC_PROBS_REALISTIC,
    )

    # ANC screens carry disease/treatment names (strings); ANCScreen resolves them
    # against sim.diseases / sim.interventions in init_pre, since sti.Sim(**parts)
    # deep-copies modules on construction.
    anc_screens = _build_anc_screens(cell)
    parts['interventions'] = list(parts['interventions']) + anc_screens
    if bundled_prevention and anc_screens:
        from interventions import DxRiskRedux
        import starsim as ss
        parts['interventions'].append(DxRiskRedux(
            name='anc_bundled_prevention',
            triggers=('anc_enroll', 'anc_tri3'),
            trigger_attr='ti_tested',
            diseases=('ng', 'ct', 'tv'),
            start=2028,
            coverage=ss.bernoulli(p=1.0),
            eff=0.9,
            dur=ss.constant(ss.months(3)),
        ))

    parts['analyzers'] = list(parts['analyzers']) + [birth_outcome_dalys(start=start)]

    sim = sti.Sim(**parts)

    # Apply effect-size assumption to the post-construction sti_fetal instance.
    for mod in sim.pars.get('custom') or []:
        if getattr(mod, 'name', None) == 'sti_fetal':
            for k in ('ptb_shift_mean', 'growth_penalty',
                       'tx_residual_growth', 'tx_residual_timing'):
                if k in assumption:
                    mod.pars[k] = assumption[k]
            break

    # Apply calibration params (set_pars_local finds modules by name across containers)
    sim_pars = row_to_sim_pars(draw_row)
    set_pars_local(sim, sim_pars)

    # Enable FSW/client syph prev tracking (matches calibration)
    for mod in sim.pars['diseases']:
        if getattr(mod, 'name', None) == 'syph':
            mod.store_sw = True
            break

    return sim


def _build_anc_screens(cell):
    """Build ANCScreen instances for a scenario cell using name strings.

    n_screens: 0=SOC (returns []), 1=enrolment (0-24w), 2=enrolment + 3rd tri (30-36w).

    3rd-tri window is 30-36w rather than PROMISE-design 32-34w: the monthly
    timestep advances GA by ~4.3 weeks so no woman's GA sequence lands inside
    a 2-week window. Widen to 6 weeks to catch at least one timestep.
    """
    from interventions import ANCScreen
    n_screens = cell['n_screens']
    coverage = cell['coverage']

    if n_screens == 0:
        return []

    diseases = ['ng', 'ct', 'tv']
    tx_map = {'ng': 'ng_tx', 'ct': 'ct_tx', 'tv': 'metronidazole'}
    treatments = list(tx_map.values())

    common = dict(
        diseases=diseases,
        treatments=treatments,
        disease_treatment_map=tx_map,
        screen_prob=coverage,
        start=2028,
    )
    screens = [ANCScreen(ga_min=0, ga_max=24, name='anc_enroll', label='anc_enroll', **common)]
    if n_screens == 2:
        screens.append(ANCScreen(ga_min=30, ga_max=36, name='anc_tri3', label='anc_tri3', **common))
    return screens
