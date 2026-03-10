# Data

Zimbabwe demographic and epidemiological input data for the ANC STI screening model.

## Demographic inputs

| File | Description |
|------|-------------|
| `age_dist_1980.csv` | Age distribution at 1980 (for early start sims) |
| `age_dist_1990.csv` | Age distribution at 1990 (default start) |
| `asfr.csv` | Age-specific fertility rates by year |
| `deaths.csv` | Age/sex-specific mortality rates by year |

## Disease inputs

| File | Description |
|------|-------------|
| `init_prev_hiv.csv` | Initial HIV prevalence by age and sex |
| `init_prev_ng.csv` | Initial gonorrhea prevalence by age and sex |
| `init_prev_ct.csv` | Initial chlamydia prevalence by age and sex |
| `init_prev_tv.csv` | Initial trichomoniasis prevalence by age and sex |
| `init_prev_bv.csv` | Initial BV prevalence by age and sex |

## Intervention inputs

| File | Description |
|------|-------------|
| `condom_use.csv` | Condom use probability by partnership type and year |
| `n_art.csv` | ART coverage targets by year |
| `n_vmmc.csv` | VMMC (circumcision) coverage targets by year |

## Calibration targets

| File | Description |
|------|-------------|
| `zimbabwe_data.csv` | General Zimbabwe epidemiological data |
| `zimbabwe_hiv_calib.csv` | HIV calibration targets (UNAIDS, ZIMPHIA) |
| `zimbabwe_sti_data.csv` | STI prevalence calibration targets |
