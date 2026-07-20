"""Regression tests for scenarios.py wiring."""
import pandas as pd
import pytest


def test_intervention_scenarios_count():
    from scenarios import INTERVENTION_SCENARIOS
    assert len(INTERVENTION_SCENARIOS) == 7
    assert 'soc' in INTERVENTION_SCENARIOS
    assert 'anc_1screen_50cov' in INTERVENTION_SCENARIOS
    assert 'anc_2screen_90cov' in INTERVENTION_SCENARIOS


def test_effect_size_assumptions_count():
    from scenarios import EFFECT_SIZE_ASSUMPTIONS
    assert len(EFFECT_SIZE_ASSUMPTIONS) == 4
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
        scenario_id='anc_2screen_90cov',
        assumption_id='central_reversible',
        draw_row=row,
        start=1985, stop=2029, n_agents=500,
    )
    sim.init()
    # After init, diseases and interventions live in sim.diseases / sim.interventions
    disease_names = list(sim.diseases.keys())
    for expected in ('hiv', 'ng', 'ct', 'tv', 'bv', 'syph', 'gudp'):
        assert expected in disease_names
    intv_names = list(sim.interventions.keys())
    assert 'anc_enroll' in intv_names
    assert 'anc_tri3' in intv_names


@pytest.mark.slow
def test_soc_scenario_no_anc_interventions():
    """SOC scenario should have no ANC screens added."""
    from scenarios import build_scenario_sim
    df = pd.read_csv('data/calibration_draws.csv')
    row = df.iloc[0].to_dict()
    sim = build_scenario_sim(
        seed=int(row['draw_idx']) * 1000,
        scenario_id='soc',
        assumption_id='central_reversible',
        draw_row=row,
        start=1985, stop=2029, n_agents=500,
    )
    sim.init()
    intv_names = list(sim.interventions.keys())
    assert 'anc_enroll' not in intv_names
    assert 'anc_tri3' not in intv_names
