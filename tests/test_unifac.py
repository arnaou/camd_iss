# # Project: camd_iss
# #
# # Title: testing the implementation of the UNIFAC model.
# # The model is based on: https://doi.org/10.1016/j.ces.2022.118125
# # The example is from smith and Van Ness
# #
# # Author: Adem Rosenkvist Nielsen Aouichaoui
# #
# # Email: arnaou@dtu.dk
# ########################################################################################################################
#
# # import python native modules
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# from matplotlib import pyplot as plt
#
# # import costume modules
# from utils.gc import load_gc, prop_calc, inspect_interractions, unifac,make_log_gamma_plot
# from utils.pyom import ensure_file_exists_in_cwd, parse_pyomo_log
#
#
# # load the data
# df_gc, df_a = load_gc('../data/data_unifac.xlsx', 'dataframe')
#
# # define the components:
# df_mix = pd.DataFrame({'CH3': [2, 2],
#                        'CH2': [1, 5],
#                        'CH2NH': [1, 0]})
# df_mix.index = ['DEA', 'HEP']  # diethylamine: CCNCC, heptane: CCCCCCC,
#
# # extract groups
# grp_list = df_mix.columns
# # inspect missing interactions
# miss_int = inspect_interractions(df_a.loc[grp_list, grp_list])
# print(f' following groups have missing unifac interactions {miss_int}. These have been set to 0')
#
# # define evaluation point from smith etl a. 2001
# x0 = np.array([0.4, 0.6])
# T0 = 308.15
# mixture = df_mix
# # calculate gamma
# gamma = unifac(df_mix, x0, T0)
# # print the results
# print(gamma)
# # evaluating the trend of log_gamma
# n_data = 10
# x0 = np.stack((np.linspace(1.0, 0.0, n_data), 1-(np.linspace(1.0, 0.0, n_data)).T), axis=1)
#
# output = make_log_gamma_plot(df_mix, x0, T0)
# output['fig'].show()
# print(output['gamma'])
#
#
#
#
# #%%
#
#
# # extract data
# # v = np.array(mixture)
# # Rk = np.array(df_gc['Rk'][mixture.columns])
# # Qk = np.array(df_gc['Qk'][mixture.columns])
# # a = np.array(df_a[mixture.columns].loc[mixture.columns])
# # # begin calculations: combinatorial term
# # r = np.dot(v, Rk)
# # q = np.dot(v, Qk)
# # L = np.mat(q) / np.mat(np.dot(x0, q)).T
# # j = np.mat(r) / np.mat(np.dot(x0, r)).T
# # gamma_c = 1 - j + np.log(j) - 5 * np.multiply(q, (1 - j / L + np.log(j / L)))
# # # begin calculation: residual term
# # e = ((v * Qk).T / q).T
# # tau = np.exp(-a / T0)
# # beta = ((np.dot(v * Qk, tau)).T/q).T
# # theta = np.dot(x0, v * Qk) / np.dot(x0, q)
# # omega = np.dot(theta, tau)
# # gamma_r = q * (1 - np.sum(theta * beta/omega - e * np.log(beta/omega),axis=1))
# # gamma = np.array(np.exp(gamma_c + gamma_r))
# #print(gamma)
