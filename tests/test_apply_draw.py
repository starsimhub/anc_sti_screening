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
    # Only the dotted module key survives
    assert set(out.keys()) == {'hiv.beta_m2f'}
    assert out['hiv.beta_m2f'] == pytest.approx(0.008)
