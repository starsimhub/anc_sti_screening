"""Find a hot syph seed. Small pop, short window, iterate draws until sustaining."""
import numpy as np, pandas as pd, time
from scenarios import build_scenario_sim

STOP = 2025
N_AGENTS = 10_000

df = pd.read_csv('data/calibration_draws.csv')
df = df.sort_values('log_syph.beta_m2f', ascending=False)  # highest beta first
for _, row in df.iterrows():
    for sub in range(3):
        seed = int(row['draw_idx']) * 1000 + sub
        t0 = time.time()
        sim = build_scenario_sim(
            seed=seed, scenario_id='soc', assumption_id='central_reversible',
            draw_row=row.to_dict(), start=1985, stop=STOP, n_agents=N_AGENTS,
        )
        sim.run()
        prev = float(sim.results.syph.prevalence.values[-1])
        prev_2020 = float(sim.results.syph.prevalence.values[np.abs(
            np.array([t.year for t in sim.t.timevec]) - 2020).argmin()])
        print(f'draw={int(row["draw_idx"])} sub={sub} seed={seed}: '
              f'prev_2020={prev_2020:.4f} prev_2025={prev:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        if prev > 0.05:
            print(f'  ==> HOT seed={seed}')
            with open('results/hot_seed.txt', 'w') as f:
                f.write(f'{int(row["draw_idx"])},{sub},{seed},{prev}\n')
            raise SystemExit(0)
print('NO HOT SEED FOUND')
