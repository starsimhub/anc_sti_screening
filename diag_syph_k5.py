"""K=5 instrumented SOC sims, 1985-2045.

Each sim records per-pregnancy trajectory + intervention state.
Combined pickle lets us slice arbitrary time windows without re-running.
"""
from __future__ import annotations
import multiprocessing as mp
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd

STOP = 2045
N_AGENTS = 10_000
DRAW_IDX = 343     # hot seed identified earlier
N_SEEDS = 5


def stage_of(syph, uid):
    if syph.primary[uid]: return 'primary'
    if syph.secondary[uid]: return 'secondary'
    if syph.early[uid]: return 'early_latent'
    if syph.late[uid]: return 'late_latent'
    return None


def run_one(args):
    seed, row = args
    from scenarios import build_scenario_sim
    sim = build_scenario_sim(
        seed=seed, scenario_id='soc', assumption_id='central_reversible',
        draw_row=row, start=1985, stop=STOP, n_agents=N_AGENTS,
    )
    sim.init()
    syph = sim.diseases.syph
    preg = sim.demographics.pregnancy

    active = {}
    finished = []
    mtct_log = []

    def on_conception(uids):
        for u in np.asarray(uids):
            u = int(u)
            active[u] = dict(
                mother_uid=u, conception_ti=int(sim.ti),
                stage_at_conception=stage_of(syph, u),
                infected_at_conception=bool(syph.infected[u]),
                mtct_stage=None, mtct_ti=None,
            )

    def on_delivery(mother_uids, newborn_uids):
        m_arr = np.asarray(mother_uids); n_arr = np.asarray(newborn_uids)
        for m, n in zip(m_arr, n_arr):
            m = int(m); n = int(n)
            rec = active.pop(m, None)
            if rec is None: continue
            rec['delivery_ti'] = int(sim.ti)
            rec['newborn_uid'] = n
            rec['stage_at_delivery'] = stage_of(syph, m)
            rec['infected_at_delivery'] = bool(syph.infected[m])
            cs = syph.cs_outcome[n]
            rec['cs_outcome'] = int(cs) if not np.isnan(cs) else -1
            finished.append(rec)

    preg.add_conception_callback(on_conception)
    preg.add_delivery_callback(on_delivery)

    # Record EVERY test / treatment event via loop.insert after syph_tx.step
    tests_tested = {}       # uid -> list of (source, ti)
    tests_positive = {}     # uid -> list of (source, ti)
    treats = {}             # uid -> list of ti

    def record_events(sim):
        for src in ('syph_anc_rpr', 'syph_anc_rdt', 'syph_symp_test', 'syph_rash_test'):
            iv = sim.interventions.get(src)
            if iv is None: continue
            ti_iv = int(iv.ti)
            for u in (iv.ti_tested == ti_iv).uids:
                tests_tested.setdefault(int(u), []).append((src, ti_iv))
            for u in (iv.ti_positive == ti_iv).uids:
                tests_positive.setdefault(int(u), []).append((src, ti_iv))
        tx = sim.interventions.get('syph_tx')
        if tx is not None:
            ti_tx = int(tx.ti)
            for u in (tx.ti_treated == ti_tx).uids:
                treats.setdefault(int(u), []).append(ti_tx)

    sim.loop.insert(record_events, label='syph_tx.update_results')

    orig_set_congenital = syph.set_congenital
    def spy_set_congenital(target_uids, source_uids=None):
        t_arr = np.asarray(target_uids)
        s_arr = np.asarray(source_uids) if source_uids is not None else np.full(len(t_arr), -1)
        for t, s in zip(t_arr, s_arr):
            t = int(t); s = int(s)
            stage = stage_of(syph, s) if s >= 0 else None
            mtct_log.append(dict(fetus_uid=t, mother_uid=s,
                                 mother_stage=stage, ti=int(sim.ti)))
        return orig_set_congenital(target_uids, source_uids)
    syph.set_congenital = spy_set_congenital

    t0 = time.time()
    sim.run()
    elapsed = time.time() - t0

    # Attach MTCT stage to finished pregnancies
    mtct_by_fetus = {r['fetus_uid']: r for r in mtct_log}
    for rec in finished:
        m = mtct_by_fetus.get(rec['newborn_uid'])
        if m is not None:
            rec['mtct_stage'] = m['mother_stage']
            rec['mtct_ti'] = m['ti']

    # Yearly syph prev (for hot/extinct diagnosis)
    yrs = np.array([t.year for t in sim.t.timevec])
    prev = np.asarray(sim.results.syph.prevalence.values)

    return dict(
        seed=seed, elapsed=elapsed,
        finished=finished, mtct_log=mtct_log,
        tests_tested=tests_tested, tests_positive=tests_positive, treats=treats,
        years=yrs.tolist(), syph_prev=prev.tolist(),
    )


def main():
    df = pd.read_csv('data/calibration_draws.csv')
    row = df[df['draw_idx'] == DRAW_IDX].iloc[0].to_dict()
    tasks = [(DRAW_IDX * 1000 + sub, row) for sub in range(N_SEEDS)]

    t0 = time.time()
    with mp.Pool(N_SEEDS) as pool:
        results = pool.map(run_one, tasks)

    for r in results:
        prev_2020 = r['syph_prev'][np.abs(np.array(r['years']) - 2020).argmin()]
        prev_end  = r['syph_prev'][-1]
        n_mtct = len(r['mtct_log'])
        print(f'seed={r["seed"]}: '
              f'prev_2020={prev_2020:.4f}  prev_end={prev_end:.4f}  '
              f'mtct_events={n_mtct}  ({r["elapsed"]:.0f}s)')

    out = Path('results/diag_k5_soc.pkl')
    with open(out, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nSaved {out}. Total wall={time.time()-t0:.0f}s.')


if __name__ == '__main__':
    main()
