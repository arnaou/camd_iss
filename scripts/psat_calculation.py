########################################################################################################################
#
#
#
#
#
#
#
#
########################################################################################################################
#%%
# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.gc import load_gc, construct_mixture, prop_calc, pyo_solve_antoine, pyo_unifac_bubbleT

#%%


# load group contributions
df_gc, df_a = load_gc('data/unifac.xlsx', 'dataframe')

# define a compound
mixture = {'CH3': [0, 1],
           'CH3O': [0, 1],
           'COO': [0, 1],
           'CH3OH': [1, 0]
           }
tags = ['MeOH', 'DMC']
df_mix = construct_mixture(tags, mixture)
df_mix.reset_index(drop=True)
# Select properties
prop_lst = ['Tb', 'Tc', 'Ac', 'Pc']

# calculate pure component properties
df_props = prop_calc(df_mix, prop_lst, df_gc)

# convert pc to MPa
df_props['Pc'] = df_props['Pc'] / 10


# determine the Antoine parameters
antoine_params = {'A': {}, 'B': {}, 'C': {}}
solv_stat = {}
term_cond = {}

for i in df_props.index:
    # extract the data
    Tb = df_props['Tb'].loc[i]
    Tc = df_props['Tc'].loc[i]
    Pc = df_props['Pc'].loc[i]
    P0 = 0.101325
    w = df_props['Ac'].loc[i]
    # solve the model
    output = pyo_solve_antoine(Tb, Tc, Pc, P0, w, verbose=False)
    # extract the solutions
    antoine_params['A'][i] = output['A']
    antoine_params['B'][i] = output['B']
    antoine_params['C'][i] = output['C']
    solv_stat[i] = output['solv_stat']
    term_cond[i] = output['term_cond']
    print(i)
    print(output['solv_stat'])
    print(output['term_cond'])

# convert dicts into dataframes and arrays
df_antoine = pd.DataFrame(antoine_params)
array_antoine = np.array(df_antoine)


# %%
