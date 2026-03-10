"""
Adverse birth outcome attribution and DALY estimation.

Implements attributable burden estimation for STIs during pregnancy.
Given STI prevalence and treatment coverage, estimates the reduction
in adverse birth outcomes (preterm birth, low birth weight, stillbirth)
attributable to screening and treatment.

Parameters from: "DALY attribution parameters" document (2024-08-29),
which converts literature RRs to % affected using:
    P1 = RR * P0 / (1 + P0 * (RR - 1))
where P0 = baseline outcome proportion, P1 = proportion with STI.

Key reference for approach: attributable fraction estimation.
"""

import numpy as np
import sciris as sc
import pandas as pd


# %% Adverse outcome parameters
# Percent of untreated pregnancies with each outcome, by STI
# Source: DALY attribution parameters doc, converted from RR to attributable %

ADVERSE_OUTCOME_PARS = sc.objdict(

    # Gonorrhea outcomes in births to untreated mothers
    ng=sc.objdict(
        preterm_birth = 0.147,   # 14.7%
        low_birth_weight = 0.223, # 22.3%
        stillbirth    = 0.03,    # 3%
        eye_infection = 0.388,   # 38.8% (no blindness) + 1.2% (blindness)
        eye_blindness = 0.012,   # 1.2%
        ectopic_pregnancy = 0.019, # 1.9% (in pregnant women)
    ),

    # Chlamydia outcomes
    ct=sc.objdict(
        preterm_birth = 0.161,   # 16.1%
        low_birth_weight = 0.027, # 2.7%
        sga           = 0.022,   # 2.2% small for gestational age
        eye_infection = 0.388,   # 38.8%
        eye_blindness = 0.012,   # 1.2%
        ectopic_pregnancy = 0.039, # 3.9%
        pneumonia     = 0.15,    # 10-20% (midpoint)
    ),

    # Trichomoniasis outcomes
    tv=sc.objdict(
        preterm_birth = 0.136,   # 13.6%
        sga           = 0.027,   # 2.7%
    ),
)

# Baseline rates (Zimbabwe context)
BASELINE_RATES = sc.objdict(
    preterm_birth    = 0.128,  # 12.8% baseline preterm birth rate (Chawanpaiboon 2023)
    low_birth_weight = 0.145,  # 14.5% baseline LBW rate (UNICEF)
    stillbirth       = 0.019,  # 19 per 1000 births (Hug 2021)
)

# GBD disability weights
DISABILITY_WEIGHTS = sc.objdict(
    preterm_birth    = 0.106,  # Moderate preterm (28-36 weeks)
    low_birth_weight = 0.106,  # Same as preterm (highly correlated)
    stillbirth       = 1.0,    # Full DALY (life lost)
    eye_infection    = 0.004,  # Mild infection
    eye_blindness    = 0.187,  # Moderate vision impairment
    sga              = 0.106,  # Similar to LBW
    pneumonia        = 0.051,  # Lower respiratory infection, mild
    ectopic_pregnancy = 0.549, # Acute abdomen
)

# Duration of disability (years)
DISABILITY_DURATION = sc.objdict(
    preterm_birth    = 1.0,    # First year complications
    low_birth_weight = 1.0,
    stillbirth       = 72.0,   # Full life expectancy lost
    eye_infection    = 0.08,   # ~1 month
    eye_blindness    = 72.0,   # Lifetime
    sga              = 1.0,
    pneumonia        = 0.08,   # ~1 month
    ectopic_pregnancy = 0.02,  # ~1 week acute
)


# %% Core functions
def outcomes_averted(n_births, sti_prev, treatment_coverage, treatment_efficacy,
                     disease='ng', outcome='low_birth_weight'):
    """
    Estimate adverse outcomes averted by screening and treating.

    The ADVERSE_OUTCOME_PARS values represent the % of untreated STI+
    pregnancies that experience the outcome *attributable to the STI*
    (i.e., the excess risk, already converted from RR via attributable
    risk formula in the source document).

    So: outcomes averted = n_infected_treated * p_outcome * efficacy

    Args:
        n_births (int):             number of births in the population
        sti_prev (float):           STI prevalence among pregnant women
        treatment_coverage (float): fraction of infected women detected and treated
        treatment_efficacy (float): fraction of treatments that cure the infection
        disease (str):              'ng', 'ct', or 'tv'
        outcome (str):              adverse outcome name

    Returns:
        dict with key metrics
    """
    pars = ADVERSE_OUTCOME_PARS[disease]
    if outcome not in pars:
        return sc.objdict(baseline_outcomes=0, averted=0, paf=0, nnt=np.inf)

    p_excess = pars[outcome]  # Excess risk attributable to STI

    # Number of STI+ pregnancies
    n_infected = n_births * sti_prev

    # Outcomes attributable to STI (without intervention)
    attributable_outcomes = n_infected * p_excess

    # Outcomes averted = infected women treated * excess risk * efficacy
    n_treated = n_infected * treatment_coverage
    averted = n_treated * p_excess * treatment_efficacy

    # PAF: fraction of all outcomes attributable to this STI
    p_baseline = BASELINE_RATES.get(outcome, 0.01)
    total_outcomes = n_births * p_baseline + attributable_outcomes
    paf = attributable_outcomes / total_outcomes if total_outcomes > 0 else 0

    # Number needed to treat to avert one outcome
    nnt = n_treated / averted if averted > 0 else np.inf

    return sc.objdict(
        baseline_outcomes=total_outcomes,
        attributable_outcomes=attributable_outcomes,
        averted=averted,
        paf=paf,
        nnt=nnt,
        n_treated=n_treated,
    )


def dalys_averted(n_births, sti_prev, treatment_coverage, treatment_efficacy,
                  disease='ng', discount_rate=0.03):
    """
    Estimate DALYs averted across all adverse outcomes for a disease.

    Args:
        n_births:             number of births
        sti_prev:             STI prevalence among pregnant women
        treatment_coverage:   fraction detected and treated
        treatment_efficacy:   treatment cure rate
        disease:              'ng', 'ct', or 'tv'
        discount_rate:        annual discount rate for future DALYs

    Returns:
        dict mapping outcome → DALYs averted, plus total
    """
    pars = ADVERSE_OUTCOME_PARS[disease]
    result = sc.objdict()
    total = 0

    for outcome in pars.keys():
        oa = outcomes_averted(n_births, sti_prev, treatment_coverage,
                              treatment_efficacy, disease, outcome)
        dw = DISABILITY_WEIGHTS.get(outcome, 0)
        dur = DISABILITY_DURATION.get(outcome, 1.0)

        # Discounted DALYs
        if discount_rate > 0 and dur > 1:
            disc_dur = (1 - np.exp(-discount_rate * dur)) / discount_rate
        else:
            disc_dur = dur

        dalys = oa.averted * dw * disc_dur
        result[outcome] = sc.objdict(
            averted=oa.averted,
            dalys=dalys,
            paf=oa.paf,
            nnt=oa.nnt,
        )
        total += dalys

    result['total_dalys'] = total
    return result


def estimate_from_sim(sim, disease='ng', treatment_coverage=0.5, treatment_efficacy=0.95):
    """
    Estimate adverse outcomes averted using prevalence from a completed simulation.

    Args:
        sim:                  completed Sim object
        disease:              'ng', 'ct', or 'tv'
        treatment_coverage:   screening coverage
        treatment_efficacy:   cure rate

    Returns:
        dict of outcomes averted and DALYs
    """
    ps = sim.results.pregnancy_sti_stats
    n_preg = float(np.mean(ps.n_pregnant[-60:]))  # Average over last 5 years
    sti_prev = float(np.mean(ps[f'pregnant_{disease}_prev'][-60:]))

    # Births per year (approximate)
    n_births = n_preg * 12  # Monthly pregnancies → annual births (rough)

    result = dalys_averted(n_births, sti_prev, treatment_coverage,
                           treatment_efficacy, disease)
    result['inputs'] = sc.objdict(
        n_births=n_births,
        sti_prev=sti_prev,
        treatment_coverage=treatment_coverage,
        treatment_efficacy=treatment_efficacy,
        disease=disease,
    )

    return result


def compare_scenarios(scenario_results, n_births=400000):
    """
    Compare DALY impact across scenarios using scenario result DataFrames.

    Args:
        scenario_results: dict of scenario_name → DataFrame from run_scenarios
        n_births:         annual births in Zimbabwe (~400K)

    Returns:
        comparison DataFrame
    """
    rows = []
    for scenario, df in scenario_results.items():
        # Get post-intervention prevalence among pregnant women
        post = df[(df.year >= 2030) & (df.year <= 2040)]
        for disease in ['ng', 'ct', 'tv']:
            prev_key = f'preg.{disease}_prev'
            prev_data = post[post.metric == prev_key]
            if len(prev_data) == 0:
                continue
            sti_prev = prev_data.value.median()

            for outcome in ADVERSE_OUTCOME_PARS.get(disease, {}).keys():
                oa = outcomes_averted(n_births, sti_prev, 1.0, 0.95, disease, outcome)
                rows.append(dict(
                    scenario=scenario,
                    disease=disease,
                    outcome=outcome,
                    sti_prev=sti_prev,
                    paf=oa.paf,
                    averted=oa.averted,
                    nnt=oa.nnt,
                ))

    return pd.DataFrame(rows)


# %% Example usage
if __name__ == '__main__':

    # Example: estimate for NG with typical Zimbabwe parameters
    print('=== NG adverse outcomes averted ===')
    print(f'Assuming: 400K births/yr, 2% NG prevalence in pregnant women')
    print(f'          50% screening coverage, 95% treatment efficacy')
    print()

    result = dalys_averted(
        n_births=400000,
        sti_prev=0.02,
        treatment_coverage=0.5,
        treatment_efficacy=0.95,
        disease='ng',
    )

    for outcome, vals in result.items():
        if outcome == 'total_dalys':
            print(f'\nTotal DALYs averted: {vals:.0f}')
        else:
            print(f'{outcome:25s}: {vals.averted:8.0f} averted, '
                  f'{vals.dalys:8.1f} DALYs, '
                  f'PAF={vals.paf:.3f}, NNT={vals.nnt:.1f}')

    # CT example
    print('\n=== CT adverse outcomes averted ===')
    result_ct = dalys_averted(400000, 0.03, 0.5, 0.95, disease='ct')
    for outcome, vals in result_ct.items():
        if outcome == 'total_dalys':
            print(f'\nTotal DALYs averted: {vals:.0f}')
        else:
            print(f'{outcome:25s}: {vals.averted:8.0f} averted, '
                  f'{vals.dalys:8.1f} DALYs, '
                  f'PAF={vals.paf:.3f}, NNT={vals.nnt:.1f}')
