"""
Test the fetal health pipeline: FetalHealth, sti_fetal connector, birth_outcome_dalys analyzer.

Verifies in isolation and together:
  1. FetalHealth classifies PTB/LBW from timing shifts and growth restriction
  2. The sti_fetal connector applies infection damage and detects treatments
  3. The birth_outcome_dalys analyzer reads FetalHealth flags correctly
  4. End-to-end: intervention arm produces fewer adverse outcomes than SOC
"""

import numpy as np
import sciris as sc
import starsim as ss
import stisim as sti
import pylab as pl
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from model import make_sim
from connectors import sti_fetal
from analyzers import birth_outcome_dalys

n_agents = 500
do_plot  = False
sc.options(interactive=False)


def make_fh_sim(n_agents=n_agents, start=2020, stop=2025, seed=1):
    """ Minimal sim with FetalHealth + Pregnancy (no diseases) """
    sim = ss.Sim(
        dt=1/12,  # Monthly — required for FetalHealth to resolve delivery timing within pregnancy
        n_agents=n_agents, start=start, stop=stop,
        people=ss.People(n_agents),
        demographics=[ss.Pregnancy(fertility_rate=100), ss.Deaths(death_rate=5)],
        custom=ss.FetalHealth(),
        networks=ss.MaternalNet(),
        rand_seed=seed,
        verbose=-1,
    )
    return sim


def make_full_sim(scenario='soc', seed=1, start=1990, stop=2035, analyzers=None, connector_pars=None):
    """ Full ANC model sim with diseases, connectors, interventions """
    sim = make_sim(
        scenario=scenario, seed=seed, start=start, stop=stop,
        debug=True, verbose=-1, analyzers=analyzers,
        connector_pars=connector_pars,
    )
    return sim


# %% FetalHealth unit tests

@sc.timer()
def test_fh_init():
    """ FetalHealth initializes and registers callbacks """
    sc.heading('Testing FetalHealth initialization...')
    sim = make_fh_sim()
    sim.init()
    fh = sim.custom['fetal_health']
    assert fh.initialized, 'FetalHealth should be initialized'
    assert hasattr(fh, 'growth_restriction'), 'Expected growth_restriction state'
    assert hasattr(fh, 'timing_shift'),       'Expected timing_shift state'
    assert hasattr(fh, 'lbw'),                  'Expected lbw state'
    assert hasattr(fh, 'sga'),                  'Expected sga state'
    return sim


@sc.timer()
def test_fh_runs():
    """ FetalHealth runs without errors and records deliveries """
    sc.heading('Testing FetalHealth runs...')
    sim = make_fh_sim(stop=2030)
    sim.run()
    fh = sim.custom['fetal_health']
    n_dlv = np.sum(fh.results['n_births'].values)
    assert n_dlv > 0, 'Expected at least one delivery with fertility_rate=100'
    return sim


@sc.timer()
def test_conception_resets():
    """ Fetal health states reset to zero at conception """
    sc.heading('Testing conception resets...')
    sim = make_fh_sim(stop=2030)
    sim.run()
    fh   = sim.custom['fetal_health']
    preg = sim.demographics.pregnancy
    pregnant = preg.pregnant.uids
    if len(pregnant):
        gr = np.asarray(fh.growth_restriction[pregnant])
        assert np.all(gr == 0), f'Expected growth_restriction=0 for pregnant women (no damage source), got max={gr.max():.4f}'
    return sim


@sc.timer()
def test_timing_shift():
    """ apply_timing_shift brings delivery forward and records the shift """
    sc.heading('Testing timing shift...')
    sim = make_fh_sim()
    sim.init()
    for _ in range(50):
        sim.run_one_step()
    fh   = sim.custom['fetal_health']
    preg = sim.demographics.pregnancy
    pregnant = preg.pregnant.uids
    if len(pregnant) == 0:
        pytest.skip('No pregnant women found')

    uids         = pregnant[:min(5, len(pregnant))]
    old_delivery = np.array(preg.ti_delivery[uids])

    fh.apply_timing_shift(uids, 2.0)

    new_delivery = np.array(preg.ti_delivery[uids])
    assert np.all(new_delivery <= old_delivery), 'Expected delivery to move earlier after timing shift'
    assert np.any(fh.timing_shift[uids] > 0),   'Expected timing_shift to be recorded'
    return sim


@sc.timer()
def test_growth_restriction():
    """ apply_growth_restriction accumulates with diminishing returns """
    sc.heading('Testing growth restriction...')
    sim = make_fh_sim()
    sim.init()
    for _ in range(50):
        sim.run_one_step()
    fh   = sim.custom['fetal_health']
    preg = sim.demographics.pregnancy
    pregnant = preg.pregnant.uids
    if len(pregnant) == 0:
        pytest.skip('No pregnant women found')

    uids = pregnant[:min(5, len(pregnant))]

    fh.apply_growth_restriction(uids, 0.10)
    gr1 = np.array(fh.growth_restriction[uids])
    assert np.allclose(gr1, 0.10), f'Expected 0.10, got {gr1}'

    # Second application: compound (diminishing)
    fh.apply_growth_restriction(uids, 0.10)
    gr2      = np.array(fh.growth_restriction[uids])
    expected = 0.10 + (1 - 0.10) * 0.10  # = 0.19
    assert np.allclose(gr2, expected), f'Expected {expected:.4f} (compound), got {gr2}'
    return sim


@sc.timer()
def test_reverse_growth():
    """ reverse_growth_restriction reduces accumulated restriction """
    sc.heading('Testing growth restriction reversal...')
    sim = make_fh_sim()
    sim.init()
    for _ in range(50):
        sim.run_one_step()
    fh   = sim.custom['fetal_health']
    preg = sim.demographics.pregnancy
    pregnant = preg.pregnant.uids
    if len(pregnant) == 0:
        pytest.skip('No pregnant women found')

    uids = pregnant[:min(5, len(pregnant))]
    fh.apply_growth_restriction(uids, 0.20)
    fh.reverse_growth_restriction(uids, 0.10)
    gr = np.array(fh.growth_restriction[uids])
    assert np.allclose(gr, 0.10), f'Expected 0.10 after reversing 0.10 from 0.20, got {gr}'
    return sim


@sc.timer()
def test_reverse_timing():
    """ reverse_timing_shift recovers a fraction of the accumulated shift """
    sc.heading('Testing timing shift reversal...')
    sim = make_fh_sim()
    sim.init()
    for _ in range(50):
        sim.run_one_step()
    fh   = sim.custom['fetal_health']
    preg = sim.demographics.pregnancy
    pregnant = preg.pregnant.uids
    if len(pregnant) == 0:
        pytest.skip('No pregnant women found')

    uids         = pregnant[:min(5, len(pregnant))]
    old_delivery = np.array(preg.ti_delivery[uids])

    fh.apply_timing_shift(uids, 4.0)
    mid_delivery = np.array(preg.ti_delivery[uids])

    fh.reverse_timing_shift(uids, 0.50)
    new_delivery = np.array(preg.ti_delivery[uids])

    assert np.all(new_delivery >= mid_delivery), 'Expected delivery to move later after partial reversal'
    assert np.all(new_delivery <= old_delivery), 'Expected delivery still earlier than original'
    return sim


@sc.timer()
def test_lbw_classification():
    """ Large growth restriction → elevated LBW rate at delivery """
    sc.heading('Testing LBW classification...')
    sim = make_fh_sim(stop=2030)
    sim.init()
    fh = sim.custom['fetal_health']

    def restrict_all(uids):
        fh.apply_growth_restriction(uids, 0.30)
    fh.add_conception_callback(restrict_all)

    sim.run()
    n_lbw = np.sum(fh.results['n_lbw'].values)
    n_dlv = np.sum(fh.results['n_births'].values)
    lbw_rate = n_lbw / n_dlv if n_dlv > 0 else 0
    assert lbw_rate > 0.3, f'Expected elevated LBW rate with 30% restriction, got {lbw_rate:.2%}'
    return sim


@sc.timer()
def test_ptb_classification():
    """ Large timing shift → elevated PTB rate at delivery """
    sc.heading('Testing PTB classification...')
    sim = make_fh_sim(stop=2030)
    sim.init()
    fh = sim.custom['fetal_health']

    # Apply a large shift at conception
    def shift_at_conception(uids):
        fh.apply_timing_shift(uids, 10.0)
    fh.add_conception_callback(shift_at_conception)

    sim.run()
    preg   = sim.demographics.pregnancy
    n_ptb  = np.sum(preg.results['n_preterm'].values)
    n_dlv  = np.sum(fh.results['n_births'].values)
    mean_ga = fh.results['mean_ga_at_birth'].values
    mean_ga = mean_ga[mean_ga > 0].mean() if np.any(mean_ga > 0) else 0
    ptb_rate = n_ptb / n_dlv if n_dlv > 0 else 0
    print(f'  PTB: {n_ptb:.0f}/{n_dlv:.0f} = {ptb_rate:.2%}, mean GA={mean_ga:.1f} weeks')

    # With a 10-week shift, many deliveries should be preterm
    assert ptb_rate > 0.1, f'Expected elevated PTB rate with 10-week shift, got {ptb_rate:.2%}'
    return sim


# %% Connector tests

@sc.timer()
def test_connector_init():
    """ sti_fetal connector initializes with correct default parameters """
    sc.heading('Testing connector initialization...')
    conn = sti_fetal()
    assert conn.name == 'sti_fetal',        f'Expected name sti_fetal, got {conn.name}'
    assert 'ng' in conn.disease_names,      'Expected ng in disease_names'
    assert 'ct' in conn.disease_names,      'Expected ct in disease_names'
    assert 'tv' in conn.disease_names,      'Expected tv in disease_names'
    assert 'ng_tx' in conn.treatment_names, 'Expected ng_tx in treatment_names'
    return conn


@sc.timer()
def test_connector_in_sim():
    """ Connector is present, initialized, and conception callback registered """
    sc.heading('Testing connector in full sim...')
    sim = make_full_sim(scenario='twice', stop=1995)
    sim.init()

    assert 'sti_fetal' in sim.connectors, 'Expected sti_fetal in sim.connectors'
    conn = sim.connectors['sti_fetal']
    assert conn.initialized, 'Expected connector to be initialized'

    fh = sim.custom['fetal_health']
    cb_names = [cb.__name__ for cb in fh._conception_callbacks]
    assert '_on_conception' in cb_names, f'Expected _on_conception callback, got {cb_names}'
    return sim


@sc.timer()
def test_execution_order():
    """
    Verify that treatment detection (in update_results) runs AFTER interventions.

    The connector's step() handles infection detection and runs in the connector
    slot (before interventions). Treatment detection is in update_results(),
    which runs after interventions and disease steps in the loop.
    """
    sc.heading('Testing execution order...')
    sim = make_full_sim(scenario='twice', stop=1992)
    sim.init()
    plan = sim.loop.to_df()

    # Connector update_results should appear after all intervention steps
    conn_ur = plan[plan.label == 'sti_fetal.update_results']
    assert len(conn_ur) > 0, 'Expected sti_fetal.update_results in the loop'
    conn_ur_pos = conn_ur.iloc[0].name

    intv_positions = []
    for tx_name in ['ng_tx', 'ct_tx', 'metronidazole']:
        rows = plan[plan.label == f'{tx_name}.step']
        if len(rows):
            intv_positions.append(rows.iloc[0].name)

    assert len(intv_positions) > 0, 'Expected at least one treatment step in the loop'

    last_intv = max(intv_positions)
    print(f'  update_results position: {conn_ur_pos}')
    print(f'  Last intervention position: {last_intv}')

    assert conn_ur_pos > last_intv, (
        f'sti_fetal.update_results (pos {conn_ur_pos}) should run AFTER the last '
        f'intervention (pos {last_intv})'
    )
    return sim


@sc.timer()
def test_treatments_administered():
    """ Verify treatments are actually being administered in the twice scenario """
    sc.heading('Testing treatments administered...')
    sim = make_full_sim(scenario='twice', start=1990, stop=2035)
    sim.run()

    n_treated = 0
    for tx_name in ['ng_tx', 'ct_tx', 'metronidazole']:
        tx = sim.interventions.get(tx_name)
        if tx is not None:
            n = np.count_nonzero(~np.isnan(tx.ti_treated.values))
            n_treated += n
            print(f'  {tx_name}: {n} treatments')

    assert n_treated > 0, 'Expected at least one treatment in the twice scenario'
    return sim


# %% Analyzer tests

@sc.timer()
def test_daly_analyzer():
    """ birth_outcome_dalys runs and detects deliveries """
    sc.heading('Testing DALY analyzer...')
    sim = make_full_sim(analyzers=[birth_outcome_dalys(start=2027)])
    sim.run()
    da = sim.analyzers['birth_outcome_dalys']
    cum_dalys = da.results['cum_dalys'].values[-1]
    n_dlv     = np.sum(da.results['n_deliveries'].values)

    assert n_dlv > 0,     'Expected at least one delivery detected by analyzer'
    assert cum_dalys >= 0, f'Expected non-negative cumulative DALYs, got {cum_dalys:.4f}'
    print(f'  {n_dlv:.0f} deliveries, {cum_dalys:.4f} cum DALYs')
    return sim


@sc.timer()
def test_dalys_increase_with_damage():
    """ Higher fetal damage parameters → more DALYs """
    sc.heading('Testing DALYs increase with damage...')

    # Use more agents to ensure enough STI-infected pregnancies with deliveries
    n = 5_000

    # Zero damage: connector applies no fetal effects
    sim_lo = make_sim(
        scenario='soc', seed=1, start=1990, stop=2035, n_agents=n, verbose=-1,
        analyzers=[birth_outcome_dalys(start=2000)],
        connector_pars=dict(
            ptb_shift_mean=dict(ng=0.0, ct=0.0, tv=0.0),
            growth_penalty=dict(ng=0.0, ct=0.0, tv=0.0),
        ),
    )

    # High damage
    sim_hi = make_sim(
        scenario='soc', seed=1, start=1990, stop=2035, n_agents=n, verbose=-1,
        analyzers=[birth_outcome_dalys(start=2000)],
        connector_pars=dict(
            ptb_shift_mean=dict(ng=5.0, ct=5.0, tv=5.0),
            growth_penalty=dict(ng=0.20, ct=0.20, tv=0.20),
        ),
    )

    sim_lo.run()
    sim_hi.run()

    dalys_lo = sim_lo.analyzers['birth_outcome_dalys'].results['cum_dalys'].values[-1]
    dalys_hi = sim_hi.analyzers['birth_outcome_dalys'].results['cum_dalys'].values[-1]
    print(f'  DALYs: lo={dalys_lo:.4f}, hi={dalys_hi:.4f}')

    assert dalys_hi > dalys_lo, (
        f'Expected higher damage pars to produce more DALYs: lo={dalys_lo:.4f}, hi={dalys_hi:.4f}'
    )
    return sim_lo, sim_hi


# %% End-to-end tests

@sc.timer()
def test_crn_reproducibility():
    """ Identical sims with same seed produce identical results """
    sc.heading('Testing CRN reproducibility...')
    kw = dict(scenario='soc', seed=42, start=1990, stop=2035, analyzers=[birth_outcome_dalys(start=2025)])
    sim1 = make_full_sim(**kw)
    sim2 = make_full_sim(**kw)
    sim1.run()
    sim2.run()

    d1 = sim1.analyzers['birth_outcome_dalys'].results['cum_dalys'].values[-1]
    d2 = sim2.analyzers['birth_outcome_dalys'].results['cum_dalys'].values[-1]
    assert d1 == d2, f'Expected identical results from same seed: {d1} vs {d2}'
    return sim1, sim2


@sc.timer()
def test_intervention_reduces_dalys(do_plot=do_plot):
    """
    CRITICAL: twice scenario should produce fewer DALYs than SOC.
    Tests the full pipeline: infections → damage → treatment → reversal → fewer adverse outcomes.
    """
    sc.heading('Testing intervention reduces DALYs...')
    seed      = 42
    start     = 1990
    stop      = 2040
    intv_year = 2027

    sim_soc  = make_full_sim(scenario='soc',   seed=seed, start=start, stop=stop, analyzers=[birth_outcome_dalys(start=intv_year)])
    sim_intv = make_full_sim(scenario='twice', seed=seed, start=start, stop=stop, analyzers=[birth_outcome_dalys(start=intv_year)])
    sim_soc.run()
    sim_intv.run()

    dalys_soc  = sim_soc.analyzers['birth_outcome_dalys'].results['cum_dalys'].values[-1]
    dalys_intv = sim_intv.analyzers['birth_outcome_dalys'].results['cum_dalys'].values[-1]
    delta      = dalys_soc - dalys_intv

    print(f'  SOC  DALYs: {dalys_soc:.4f}')
    print(f'  INTV DALYs: {dalys_intv:.4f}')
    print(f'  Delta: {delta:+.4f} (positive = intervention better)')

    if do_plot:
        fig, axes = pl.subplots(1, 2, figsize=(12, 5))
        for ax, sim, label in [(axes[0], sim_soc, 'SOC'), (axes[1], sim_intv, 'Intervention')]:
            da = sim.analyzers['birth_outcome_dalys']
            ax.plot(sim.t.yearvec, da.results['cum_dalys'].values, label='Cumulative DALYs')
            ax.set_title(label)
            ax.set_xlabel('Year')
            ax.set_ylabel('Cumulative DALYs')
            ax.legend()
        fig.tight_layout()

    assert dalys_intv <= dalys_soc, (
        f'Intervention arm has MORE DALYs than SOC: '
        f'SOC={dalys_soc:.4f}, INTV={dalys_intv:.4f}. '
        f'Treatment reversal may not be working.'
    )
    return sim_soc, sim_intv


# %% Run

if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    # FetalHealth unit tests
    test_fh_init()
    test_fh_runs()
    test_conception_resets()
    test_timing_shift()
    test_growth_restriction()
    test_reverse_growth()
    test_reverse_timing()
    test_lbw_classification()
    test_ptb_classification()

    # Connector tests
    test_connector_init()
    test_connector_in_sim()
    test_execution_order()
    test_treatments_administered()

    # Analyzer tests
    test_daly_analyzer()
    test_dalys_increase_with_damage()

    # End-to-end tests
    test_crn_reproducibility()
    sim_soc, sim_intv = test_intervention_reduces_dalys(do_plot=do_plot)

    T.toc()

    if do_plot:
        pl.show()
