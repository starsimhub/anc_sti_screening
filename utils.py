"""
Utilities and defaults for the ANC STI screening model.
"""
import numpy as np
import sciris as sc

# Percentiles for plotting uncertainty bands (wide to narrow, for alpha shading)
percentile_pairs = [[.01, .99], [.1, .9], [.25, .75]]
percentiles = [.5] + [p for pair in percentile_pairs for p in pair]

# Scenario definitions
scenarios = ['soc', 'anc_all', 'anc_ng_only', 'anc_ng_ct']
scenlabels = {
    'soc':         'Standard of care',
    'anc_all':     'ANC screen (NG+CT+TV)',
    'anc_ng_only': 'ANC screen (NG only)',
    'anc_ng_ct':   'ANC screen (NG+CT)',
}

# Treatment labels
treatments = ['ng_tx', 'ct_tx', 'metronidazole']
tx_labels = {'ng_tx': 'NG', 'ct_tx': 'CT', 'metronidazole': 'MTNZ'}

# Results to exclude from analysis
unneeded_results = [
    'pregnancy', 'deaths', 'structuredsexual', 'maternalnet', 'new_deaths', 'cum_deaths',
    'fsw_testing', 'other_testing', 'low_cd4_testing', 'art', 'vmmc', 'hivdx',
]


# Helper functions
def set_font(size=None, font='Libertinus Sans'):
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return


def count(arr):
    return np.count_nonzero(arr)


def get_y(df, which, rname):
    if which == 'single':
        y = df[rname]
    elif which == 'multi':
        y = df[(rname, '50%')]
    return y
