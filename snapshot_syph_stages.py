"""Snapshot syph stage distribution at a target year.

Runs SOC sim to target year, prints stage breakdown for:
  - All adult women 15-49
  - Currently-pregnant women (any age)

Usage:
    python snapshot_syph_stages.py                        # draw 343 seed 343000, year 2025
    python snapshot_syph_stages.py --seed 343003 --year 2025
"""
import argparse
import pandas as pd
from pathlib import Path

from scenarios import build_scenario_sim


def stage_breakdown(syph, uids, label):
    n = int(len(uids))
    if n == 0:
        print(f'\n{label}: n=0')
        return
    counters = {
        'susceptible':  int((~syph.infected[uids]).sum()),
        'exposed':      int(syph.exposed[uids].sum()),
        'primary':      int(syph.primary[uids].sum()),
        'secondary':    int(syph.secondary[uids].sum()),
        'early_latent': int(syph.early[uids].sum()),
        'late_latent':  int(syph.late[uids].sum()),
        'tertiary':     int(syph.tertiary[uids].sum()),
    }
    total_infected = int(syph.infected[uids].sum())
    print(f'\n{label}  (n={n}, infected={total_infected} = {100*total_infected/n:.1f}%)')
    for k, v in counters.items():
        print(f'  {k:<15} {v:>5}  ({100*v/n:5.2f}% of {label.split()[0]}; '
              f'{100*v/max(total_infected,1):5.1f}% of infected)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draw', type=int, default=343)
    ap.add_argument('--seed', type=int, default=343000)
    ap.add_argument('--year', type=int, default=2025)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    draws = pd.read_csv(repo / 'data' / 'calibration_draws.csv')
    row = draws[draws['draw_idx'] == args.draw].iloc[0].to_dict()

    sim = build_scenario_sim(
        seed=args.seed, scenario_id='soc', assumption_id='central_reversible',
        draw_row=row, start=1985, stop=args.year + 1, n_agents=10_000,
    )
    sim.run()

    ppl = sim.people
    syph = sim.diseases.syph
    preg = sim.demographics.pregnancy

    adult_female = ppl.female & (ppl.age >= 15) & (ppl.age < 50) & ppl.alive
    women_uids = adult_female.uids
    preg_uids = preg.pregnant.uids

    print(f'== Syph stage snapshot at year {args.year} '
          f'(draw {args.draw}, seed {args.seed}) ==')
    stage_breakdown(syph, women_uids, 'Adult women 15-49')
    stage_breakdown(syph, preg_uids, 'Currently-pregnant women')


if __name__ == '__main__':
    main()
