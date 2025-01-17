# Project: camd_iss
#
# Title: testing initialization of bubble-T procedure
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
########################################################################################################################

# import python native modules
import pandas as pd
import numpy as np
import pyomo.environ as pyo

from matplotlib import pyplot as plt
# import costume modules
from utils.gc import load_gc, prop_calc, inspect_interractions, solve_antoine
from utils.pyom import ensure_file_exists_in_cwd, parse_pyomo_log

# load the data
df_gc, df_a = load_gc('../data/data_unifac.xlsx', 'dataframe')

# define mixture
df_mix = pd.DataFrame({'CH3': [2, 2],
                       'CH2': [1, 5],
                       'CH2NH': [1, 0]})
df_mix.index = ['DEA', 'HEP']
# extract groups
grp_list = df_mix.columns
# inspect missing interactions
miss_int = inspect_interractions(df_a.loc[grp_list, grp_list])
print(f' following groups have missing unifac interactions {miss_int}. These have been set to 0')

# set up x values to be evaluated
n_data = 11
x0 = np.stack((np.linspace(1.0, 0.0, n_data), 1-(np.linspace(1.0, 0.0, n_data)).T), axis=1)
T0 = 308.15
mixture = df_mix.reset_index(drop=True)

#%% need to validate the initialization function
v = np.array(mixture)
Rk = np.array(df_gc['Rk'][mixture.columns])  # np.array(df_gc['Rk'][df_mix.columns])
Qk = np.array(df_gc['Qk'][mixture.columns])
a = np.array(df_a[mixture.columns].loc[mixture.columns])
# begin calculations
# combinatorial part
r = np.dot(v, Rk)
q = np.dot(v, Qk)
j = np.mat(r) / np.mat(np.dot(x0, r)).T
L = np.mat(q) / np.mat(np.dot(x0, q)).T
gamma_c = 1 - j + np.log(j) - 5 * np.multiply(q, (1 - j / L + np.log(j / L)))
# residual part
e = ((v * Qk).T / q).T
tau = np.exp(-a / T0)
beta = ((np.dot(v * Qk, tau)).T / q).T
theta = np.dot(x0, v * Qk) / np.dot(x0, q)
omega = np.dot(theta, tau)
gamma_r = q * (1 - np.sum(theta * beta / omega - e * np.log(beta / omega), axis=1))
gamma = np.array(np.exp(gamma_c + gamma_r))




# #%%
# # create the model
# model = pyo.ConcreteModel('bubble-T UNIFAC')
# # define the parameters of the model
# n_comp = x0.shape[1]
# n_data = x0.shape[0]
# # load the unifac parameters
# file_path = '../data/data_unifac.xlsx'
# df_gc, df_a = load_gc(file_path, output_format='dataframe')
# # define sets
# model.k = pyo.Set(initialize=df_mix.columns, name='Groups present')
# model.i = pyo.RangeSet(0, n_comp - 1, name=' no. of Compounds')
# model.dp = pyo.RangeSet(0, n_data - 1, name='no. of data points')
# model.grp = pyo.Set(initialize=df_a.columns, name='all Groups')
# # define the constants/parameters
# model.A = pyo.Param(model.i, initialize=antoine_params[:, 0], mutable=False, name='Antoine A', domain=pyo.Reals)
# model.B = pyo.Param(model.i, initialize=antoine_params[:, 1], mutable=False, name='Antoine B', domain=pyo.Reals)
# model.C = pyo.Param(model.i, initialize=antoine_params[:, 2], mutable=False, name='Antoine C', domain=pyo.Reals)
# model.a = pyo.Param(model.grp, model.grp, initialize=df_a.stack().to_dict(), mutable=False,
#                     name='unifac interaction a_mk', domain=pyo.Reals)
# model.Rk = pyo.Param(model.grp, initialize=df_gc.Rk.to_dict(), mutable=False, name='unifac Rk parameters')
# model.Qk = pyo.Param(model.grp, initialize=df_gc.Qk.to_dict(), mutable=False, name='unifac Qk parameters')
# model.x = pyo.Param(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(x0)}, mutable=False,
#                     name='Liquid composition x', domain=pyo.Reals)
# model.P = pyo.Param(initialize=P0, mutable=False, name='Total pressure in MPa', domain=pyo.PositiveReals)
# model.v = pyo.Param(model.i, model.k, initialize=mixture.stack().to_dict(), mutable=False, name='compounds')
