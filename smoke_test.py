"""Smoke test: SOC + intervention scenarios with DALY and cost analyzers."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', MKL_NUM_THREADS='1')
import numpy as np, sciris as sc
from model import make_sim
from analyzers import birth_outcome_dalys, intervention_costs

RESULTS_DIR, SEED, N_AGENTS, START, STOP = 'results', 42, 500, 1990, 2040

def run_smoke_test():
    results = {}
    for scenario in ['soc', 'twice']:
        sc.heading(f'Smoke test: {scenario} scenario')
        sim = make_sim(scenario=scenario, seed=SEED, start=START, stop=STOP, n_agents=N_AGENTS, debug=False, verbose=-1, analyzers=[birth_outcome_dalys(), intervention_costs()])
        sim.run()
        scenario_results = {'n_agents': sim.pars.n_agents, 'n_timesteps': len(sim.t), 'final_year': int(sim.t.year)}
        if 'birth_outcome_dalys' in sim.results:
            dalys = sim.results.birth_outcome_dalys
            scenario_results['dalys'] = {'n_deliveries': int(np.sum(dalys.n_deliveries)), 'n_ptb': int(np.sum(dalys.n_ptb)), 'n_lbw': int(np.sum(dalys.n_lbw)), 'total_yld_ptb': float(np.sum(dalys.yld_ptb)), 'total_yld_lbw': float(np.sum(dalys.yld_lbw)), 'total_dalys': float(np.sum(dalys.cum_dalys[-1])) if len(dalys.cum_dalys) > 0 else 0}
        if 'intervention_costs' in sim.results:
            costs = sim.results.intervention_costs
            n_treated = int(np.sum(costs.n_treated_ng) + np.sum(costs.n_treated_ct) + np.sum(costs.n_treated_tv))
            scenario_results['costs'] = {'n_screened': int(np.sum(costs.n_screened)), 'n_treated': n_treated, 'cost_screening': float(np.sum(costs.cost_screening)), 'cost_treatment': float(np.sum(costs.cost_treatment)), 'cost_outcomes': float(np.sum(costs.cost_outcomes)), 'total_cost': float(np.sum(costs.cum_cost[-1])) if len(costs.cum_cost) > 0 else 0}
        ng_summary = {'prev': float(sim.results.ng.prevalence[-1] if hasattr(sim.results.ng, 'prevalence') else np.nan), 'n_infec': int(np.sum(sim.results.ng.new_infections) if hasattr(sim.results.ng, 'new_infections') else 0)}
        scenario_results['ng_summary'] = ng_summary
        results[scenario] = scenario_results
    return results

if __name__ == '__main__':
    results = run_smoke_test()
    sc.save(f'{RESULTS_DIR}/smoke_test_results.obj', results)
    print(f'Smoke test complete. Saved to {RESULTS_DIR}/smoke_test_results.obj')
