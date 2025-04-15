"""
Importing packages
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.gc import load_gc, construct_mixture, prop_calc, pyo_solve_antoine, pyo_unifac_bubbleT

"""
Load the data
"""
# load group contributions
df_gc, df_a = load_gc('../trash/unifac.xlsx', 'dataframe')

"""
Define the system
"""
# define the azeotropic pair
# dict_binary = {'CH3': [1, 0],
#                'CH3CO': [1, 0],
#                'CH3OH': [0, 1]
#                }
# tags = ['Acet', 'MeOH']
# dict_binary = {'CH3': [1, 0],
#                'CH2': [1, 0],
#                'OH': [1, 0],
#                'H2O': [0, 1]
#                 }
# tags = ['EtOH', 'Water']
dict_binary = {'acrylnitrile': [1, 0],
               'H2O': [0, 1]
                }
tags = ['acetonit', 'Water']
df_azeo = construct_mixture(tags, dict_binary)

"""
Pure component property prediction
"""
# Select properties
prop_lst = ['Tb', 'Tc', 'Ac', 'Pc']

# calculate pure component properties
df_props = prop_calc(df_azeo, prop_lst, df_gc)
# convert pc to MPa
df_props['Pc'] = df_props['Pc'] / 10
# # # correcting DMC properties
# df_props.loc['Acet', 'Tb'] = 329.15 # correct for acetone
# df_props.loc['Acet', 'Tc'] = 508.20 # correct for acetone
# df_props.loc['EtOH', 'Tb'] = 351.5 # correct for acetone
# df_props.loc['EtOH', 'Tc'] = 514.0 # correct for acetone
# df_props.loc['EtOH', 'Pc'] = 6.3 # correct for acetone


df_azeo.fillna(0.0, inplace=True)
df_azeo.reset_index(drop=True, inplace=True)
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

# # recheck the saturated vapour pressure
# aveva_psat_meoh = pd.read_excel('../data/methanolPsat.xls', skiprows=3)
# aveva_psat_meoh.drop(aveva_psat_meoh.columns[0], axis=1, inplace=True)
# aveva_psat_meoh.columns = ['T', 'P']
# aveva_psat_meoh['PP'] = 10**(df_antoine.loc['MeOH', 'A'] - df_antoine.loc['MeOH', 'B'] / (df_antoine.loc['MeOH', 'C']+aveva_psat_meoh.loc[:, 'T']))
# aveva_psat_dmc = pd.read_excel('../data/dmcPsat.xls', skiprows=3)
# aveva_psat_dmc.drop(aveva_psat_dmc.columns[0], axis=1, inplace=True)
# aveva_psat_dmc.columns = ['T', 'P']
# aveva_psat_dmc['PP'] = 10**(df_antoine.loc['DMC', 'A'] - df_antoine.loc['DMC', 'B'] / (df_antoine.loc['DMC', 'C']+aveva_psat_dmc.loc[:, 'T']))

# #%% make plot of Psat
# plt.plot(aveva_psat_meoh['T'], aveva_psat_meoh['P'], 'ro', label='aveva')
# plt.plot(aveva_psat_meoh['T'], aveva_psat_meoh['PP'], 'k-', label='our work')
# plt.xlabel('T, K')
# plt.ylabel('Psat, MPa')
# plt.title('Methanol')
# plt.legend()
# plt.show()
#
# plt.plot(aveva_psat_dmc['T'], aveva_psat_dmc['P'], 'ro', label='aveva')
# plt.plot(aveva_psat_dmc['T'], aveva_psat_dmc['PP'], 'k-', label='our work')
# plt.xlabel('T, K')
# plt.ylabel('Psat, MPa')
# plt.title('DMC')
# plt.legend()
# plt.show()

#%%

#%% perform vapour liquid equilibrium using bubble-T
n_data = 100
P = 0.101325
x0 = np.stack((np.linspace(1, 0, n_data), 1 - np.linspace(1, 0, n_data)), axis=1)
T0 = np.linspace(df_props['Tb'][0], df_props['Tb'][1], n_data).flatten()
solution, model = pyo_unifac_bubbleT(x0, T0, P, array_antoine[(0, 1), :], df_azeo.iloc[:2, :], verbose=True)
# retrieving the solutions
y = np.empty_like(x0)
gamma = np.empty_like(x0)
T = np.empty_like(T0)
psat = np.empty_like(x0)
#%%
for j in model.y.keys():
    y[j] = model.y[j].value

for j in model.gamma.keys():
    gamma[j] = model.gamma[j].value

for j in model.T.keys():
    T[j] = model.T[j].value

for j in model.psat.keys():
    psat[j] = model.psat[j].value

#%%
# lad data from aveva
# df_aveva = pd.read_excel('../data/BinarymethanolDMC.xlsx', skiprows=2)
# df_aveva.columns = ['no', 'T', 'P', 'x_meoh', 'y_meoh', 'x_dmc', 'y_dmc']

# check whether they have an azeotrope
#plt.plot(df_aveva['x_meoh'], df_aveva['y_meoh'], 'ro', label='aveva')
plt.plot(np.linspace(0, 1, n_data), np.linspace(0, 1, n_data), '-k')

plt.plot(x0[:, 0], y[:, 0], '-b', label='our work')
plt.xlabel('x methanol')
plt.ylabel('y methanol')

plt.title('x-y methanol')
plt.show()