"""
Draw-to-sim parameter translation and application.

Adapted from sti_notification/calibration/artifacts/scripts/_pipeline.py.
Uses the same log-transform convention (natural log via np.exp) and the
same module-matching-by-name strategy.
"""
from __future__ import annotations

import numpy as np
import starsim as ss


def row_to_sim_pars(row) -> dict:
    """Translate one prior-CSV row into a {module.par: value} dict.

    - log_-prefixed columns are inverse-transformed with np.exp.
    - Non-module keys (draw_idx, retention_rank, gof, seed, and any key
      lacking a '.') are dropped.
    - Accepts a plain dict or a pandas.Series.
    """
    if hasattr(row, 'to_dict'):
        row = row.to_dict()
    sim_pars = {}
    for col, val in row.items():
        if col in ('draw_idx', 'seed', 'retention_rank', 'gof'):
            continue
        if isinstance(col, str) and col.startswith('log_'):
            key = col[4:]
            v = float(np.exp(val))
        else:
            key = col
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
        if isinstance(key, str) and '.' in key:
            sim_pars[key] = v
    return sim_pars


def set_pars_local(sim, pars: dict):
    """Apply a {module.par: value} dict to a built sim.

    Handles the few priors that need special wiring:
      - time_to_undetectable: wraps in ss.lognorm_ex(ss.years, ss.years)
      - rel_trans_latent_half_life: wraps in ss.years
      - p_symp_primary_f / p_symp_primary_m: list-index into p_symp_primary
      - Distributions with a .set() method: call .set(mean=value)
    """
    for key, value in pars.items():
        if '.' not in key:
            continue
        mod_name, par_name = key.split('.', 1)
        found = False
        for category in ('diseases', 'networks', 'interventions',
                         'connectors', 'analyzers',
                         'demographics', 'custom'):
            container = sim.pars.get(category)
            if container is None:
                continue
            if isinstance(container, list):
                for mod in container:
                    if hasattr(mod, 'name') and mod.name == mod_name:
                        if par_name == 'time_to_undetectable':
                            mod.pars[par_name] = ss.lognorm_ex(
                                ss.years(float(value)), ss.years(float(value)))
                        elif par_name == 'rel_trans_latent_half_life':
                            mod.pars[par_name] = ss.years(float(value))
                        elif par_name == 'p_symp_primary_f':
                            mod.pars['p_symp_primary'][0] = float(value)
                        elif par_name == 'p_symp_primary_m':
                            mod.pars['p_symp_primary'][1] = float(value)
                        else:
                            existing = mod.pars.get(par_name)
                            if hasattr(existing, 'set'):
                                existing.set(mean=value)
                            else:
                                mod.pars[par_name] = value
                        found = True
                        break
            if found:
                break
    return sim
