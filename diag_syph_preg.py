"""Run one hot SOC sim (draw 343, seed 343000) with PregnancyLog attached.

Writes results/syph_preg_log.pkl (list of per-pregnancy records).

Usage:
    python diag_syph_preg.py                        # draw 343, seed 343000, 1985-2045
    python diag_syph_preg.py --seed 343003
    python diag_syph_preg.py --draw 343 --seed 343000 --stop 2030
"""
import argparse
import pickle
import time
from pathlib import Path
import pandas as pd

from scenarios import build_scenario_sim
from syph_preg_analyzer import PregnancyLog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draw', type=int, default=343)
    ap.add_argument('--seed', type=int, default=343000)
    ap.add_argument('--start', type=int, default=1985)
    ap.add_argument('--stop', type=int, default=2045)
    ap.add_argument('--n-agents', type=int, default=10_000)
    ap.add_argument('--out', default='results/syph_preg_log.pkl')
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    draws = pd.read_csv(repo / 'data' / 'calibration_draws.csv')
    row = draws[draws['draw_idx'] == args.draw].iloc[0].to_dict()

    analyzer = PregnancyLog()
    sim = build_scenario_sim(
        seed=args.seed,
        scenario_id='soc',
        assumption_id='central_reversible',
        draw_row=row,
        start=args.start, stop=args.stop, n_agents=args.n_agents,
        extra_analyzers=[analyzer],
    )

    t0 = time.time()
    sim.run()
    dt = time.time() - t0

    records = sim.analyzers['pregnancy_log'].to_records()
    out = repo / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'wb') as f:
        pickle.dump({
            'records': records,
            'meta': dict(
                draw=args.draw, seed=args.seed,
                start=args.start, stop=args.stop,
                n_agents=args.n_agents, wall_seconds=dt,
            ),
        }, f)

    n_open = sum(1 for r in records if r['outcome'] is None)
    n_closed = len(records) - n_open
    print(f'Wrote {out} — {len(records)} logged pregnancies '
          f'({n_closed} closed, {n_open} still open at sim end) — '
          f'{dt:.0f}s wall')


if __name__ == '__main__':
    main()
