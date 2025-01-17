########################################################################################################################
# Project: camd_iss
#
# Title: Green Solvent screening for entrainers for acetone-methanol system
#
# Author: Adem Rosenkvist Nielsen Aouichaoui
#
# Email: arnaou@kt.dtu.dk
########################################################################################################################
"""
Importing packages
"""
import pandas as pd
import numpy as np
from utils.gc import load_gc, prop_calc, pyo_solve_antoine, pyo_unifac_bubbleT, iss_method, unifac, iss_method_new
from datetime import datetime
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)
startTime = datetime.now()
"""
user input
"""
# define the azeotropic pair: acetone methanol
dict_binary = {'CH3': [1, 0],
               'CH3CO': [1, 0],
               'CH3OH': [0, 1]
               }
df_binary = pd.DataFrame(dict_binary)
tags = ['Acet', 'MeOH']
# define the bounds on pure component properties
prop_lim = pd.DataFrame()
prop_lim['tm'] = [0.0, 298.15]
prop_lim['tb'] = [368, 500]
prop_lim['hv'] = [-np.inf, 67]
# define bonds on mixture properties
#
"""
Load the data
"""
# load interaction parameters
df_gc, df_a = load_gc('../data/unifac.xlsx', 'dataframe')
# load the entrainers to be screened
# load green solvent data
df0 = pd.read_excel('../data/GreenSolvBank.xlsx')
# select the compounds to be evaluated
df1 = df0[df0['Final Criteria'] == 'To be evaluated']
# remove columns that are only nans
df2 = df1.dropna(axis=1, how='all')
# set the remaining nans to 0.0
df3 = df2.fillna(0.0)
# select only the contribution data
df_entrainer = df3.iloc[:, 7:]
# compose design space
df_space = pd.concat([df_binary, df_entrainer])
df_space.fillna(0.0, inplace=True)
df_space.reset_index(drop=True, inplace=True)
# remove an entrainer in case it is also an entrainer
identical_to_row_0 = df_space.apply(lambda row: row.equals(df_space.iloc[0]) and row.name != 0, axis=1)
identical_to_row_1 = df_space.apply(lambda row: row.equals(df_space.iloc[1]) and row.name != 1, axis=1)
identical_rows = identical_to_row_0 | identical_to_row_1
df_space = df_space[~identical_rows]
# check for valency and connectivity

# check if interaction parameters and group contributions are available

"""
Pure component property calculation & screening
"""
# Calculate pure component properties
prop_lst = ['Tb', 'Tc', 'Tm', 'Ac', 'Pc', 'hv']
df_props = prop_calc(df_space, prop_lst, df_gc)
df_props.loc[0, 'Tb'] = 329.15  # correct for acetone
df_props.loc[0, 'Tc'] = 508.20  # correct for acetone
df_props.loc[:, 'Pc'] = df_props['Pc'] / 10  # convert to MPa
# screen based on pure component properties
death_note = []
temp = list(df_props[2:][df_props[2:]['Tb'] >= df_props[2:]['Tc']].index)
death_note += temp
temp = list(df_props[2:][df_props[2:]['Tm'] >= 298].index)
death_note += temp
temp = list(df_props[2:][df_props[2:]['Tb'] <= 368].index)
death_note += temp
temp = list(df_props[2:][df_props[2:]['Tb'] >= 500].index)
death_note += temp
death_note = set(death_note)
# remove compounds not adhering to the pure component criteria
df_pure_const = pd.DataFrame(True, index=range(df_space.shape[0]), columns=['success'])
df_pure_const.loc[list(death_note)] = False
idx_pure = df_pure_const[df_pure_const['success'] == True].index
# calculate the antoine parameters
antoine_params = {}
solv_stat = {}
term_cond = {}
# solve for the Antoine parameters for each compound
for i in idx_pure:
    Tb = df_props['Tb'].loc[i]
    Tc = df_props['Tc'].loc[i]
    Pc = df_props['Pc'].loc[i]
    P0 = 0.101325
    w = df_props['Ac'].loc[i]

    output = pyo_solve_antoine(Tb, Tc, Pc, P0, w, verbose=False)

    antoine_params[i] = {}
    antoine_params[i]['A'] = output['A']
    antoine_params[i]['B'] = output['B']
    antoine_params[i]['C'] = output['C']
    solv_stat[i] = output['solv_stat']
    term_cond[i] = output['term_cond']

for i in solv_stat.keys():
    if solv_stat[i] != 'ok' or term_cond[i] != 'optimal':
        print('Something is wrong with compound: ' + str(i))

# organize the antoine coefficients
df_antoine = pd.DataFrame(antoine_params)
# %%
"""
Mixture property calculation and screening
"""
#  define the total pressure: atmospheric in MPa
P = 0.101325
# evaluate the presence of azeotropes
x0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# initialize dataframe
df_azeotrope = pd.DataFrame(False, index=df_antoine.columns, columns=['Azeo_AE', 'Azeo_BE'])
for i in idx_pure[2:]:
    # define the mixture
    df_mix = df_space.iloc[[0, 1, i]].reset_index(drop=True)
    # extract antoine parameters
    antoine_params = np.array(df_antoine[[0, 1, i]]).T
    # make a guess for the temperature
    T0 = [df_props.loc[0, 'Tb'], df_props.loc[1, 'Tb'], df_props.loc[i, 'Tb']]
    # evaluate gama and psat at the given points at the extremities of the composition
    a = unifac(df_mix, x0, T0[0], df_gc, df_a)
    gamma0 = a[0, :]
    psat0 = (10 ** (antoine_params[:, 0][:, None] - antoine_params[:, 1][:, None] / (
                antoine_params[:, 2][:, None] + T0[0]))).T
    a = unifac(df_mix, x0, T0[1], df_gc, df_a)
    gamma1 = a[1, :]
    psat1 = (10 ** (antoine_params[:, 0][:, None] - antoine_params[:, 1][:, None] / (
                antoine_params[:, 2][:, None] + T0[1]))).T
    a = unifac(df_mix, x0, T0[2], df_gc, df_a)
    gamma2 = a[2, :]
    psat2 = (10 ** (antoine_params[:, 0][:, None] - antoine_params[:, 1][:, None] / (
                antoine_params[:, 2][:, None] + T0[2]))).T
    # reconstruct the gamma and psat so it looks nice
    gamma = np.vstack((gamma0, gamma1, gamma2))
    psat = np.vstack((psat0, psat1, psat2))
    # azeotrope detection
    Az_AB = np.log(gamma[1, 0] * psat[1, 0] / P) * -np.log(gamma[0, 1] * psat[0, 1] / P)
    Az_AE = np.log(gamma[2, 0] * psat[2, 0] / P) * -np.log(gamma[0, 2] * psat[0, 2] / P)
    Az_BE = np.log(gamma[2, 1] * psat[2, 1] / P) * -np.log(gamma[1, 2] * psat[1, 2] / P)
    # fill dataframe
    df_azeotrope.loc[i, 'Azeo_AE'] = Az_AE < 0
    df_azeotrope.loc[i, 'Azeo_BE'] = Az_BE < 0

# identify compounds that pass the azeotrope test
idx_azeo = df_azeotrope[(df_azeotrope['Azeo_AE'] == False) & (df_azeotrope['Azeo_BE'] == False)].index

# now we can perform mixture stability and miscibility analysis

# define the number of composition points to be evaluated
n_data = 5
# construct the compoistions
x0 = np.vstack((
    np.stack((np.linspace(1.0, 0.0, n_data), np.zeros(n_data), 1.0 - np.linspace(1.0, 0.0, n_data)), axis=1),
    np.stack((np.zeros(n_data), np.linspace(1.0, 0.0, n_data), 1.0 - np.linspace(1.0, 0.0, n_data)), axis=1)))
# initialize lists for results
cond_G_AE = []  # miscibility criteria between A-E
cond_G_BE = []  # miscibility criteria between B-E
S_inf = []  # Selectivity at infinite dilution
C_inf = []  # capacity at infinite dilution
SxC = []  # product of selectivity and capacity
solv_stat = []
term_cond = []
solvent_no = []
df_mix_const = pd.DataFrame(np.nan, index=idx_azeo, columns=['misc_AE', 'misc_BE', 'S', 'C', 'SxC',
                                                             'solver_stat', 'term_cond', 'Rmin',
                                                             'ED'])
# perform bubble-T VLE calculation along with mixture property
for i in idx_azeo[2:]:
    print(i)
    # generate initial guess for the temperature
    T0 = np.hstack((
        np.linspace(df_props['Tb'][0], df_props['Tb'][i], n_data).flatten(),
        np.linspace(df_props['Tb'][1], df_props['Tb'][i], n_data).flatten()))
    # define the mixture only
    df_mix = df_space.iloc[[0, 1, i]].reset_index(drop=True)
    # solve the bubble-T
    try:
        solution, model = pyo_unifac_bubbleT(x0, T0, P, np.array(df_antoine[[0, 1, i]]).T, df_mix, verbose=True)

        # retrieving the solutions
        y = np.empty_like(x0)
        gamma = np.empty_like(x0)
        T = np.empty_like(T0)
        psat = np.empty_like(x0)

        for j in model.y.keys():
            y[j] = model.y[j].value

        for j in model.gamma.keys():
            gamma[j] = model.gamma[j].value

        for j in model.T.keys():
            T[j] = model.T[j].value

        for j in model.psat.keys():
            psat[j] = model.psat[j].value

        # define epsilon for a safeguard
        eps = 1e-5
        # Calculate Gibbs energy of mixing
        Gmix = np.sum(x0 * np.log(gamma), axis=1) + np.sum(np.ma.log(x0).filled(0) * x0, axis=1)
        Gmix_AE = Gmix[:n_data]
        Gmix_BE = Gmix[n_data:]
        # calculate the second derivative of the Gibbs energy of mixing
        Gmix2_AE = np.zeros_like(Gmix_AE)
        Gmix2_BE = np.zeros_like(Gmix_AE)
        # Calculate second order Gibbs energy of mixing
        for j in range(int(Gmix_AE.shape[0]) - 2):
            Gmix2_AE[j + 1] = Gmix_AE[j + 2] - 2 * Gmix_AE[j + 1] + Gmix_AE[j]
            Gmix2_BE[j + 1] = Gmix_BE[j + 2] - 2 * Gmix_BE[j + 1] + Gmix_BE[j]
        # Check the Gibbs energy and the second order Gibbs energy of mixing
        cond_G_AE.append(all(Gmix_AE <= 0 + eps) and all(Gmix2_AE >= 0))
        cond_G_BE.append(all(Gmix_BE <= 0 + eps) and all(Gmix2_BE >= 0))

        # calculation of selectivity and capacity at infinite dilution
        S_inf.append(gamma[-1, 0] / gamma[-1, 1])
        C_inf.append(1 / gamma[-1, 1])
        SxC.append(gamma[-1, 0] / gamma[-1, 1] ** 2)

        df_mix_const.loc[i, 'misc_AE'] = cond_G_AE[-1]
        df_mix_const.loc[i, 'misc_BE'] = cond_G_BE[-1]
        df_mix_const.loc[i, 'S'] = S_inf[-1]
        df_mix_const.loc[i, 'C'] = C_inf[-1]
        df_mix_const.loc[i, 'SxC'] = SxC[-1]

    except ValueError:
        print('fuck')
        solution.solver.status = 'ERROR'
        solution.solver.termination_condition = 'ERROR'

    solv_stat.append(str(solution.solver.status))
    term_cond.append(str(solution.solver.termination_condition))
    solvent_no.append(i)
    df_mix_const.loc[i, 'solver_stat'] = solv_stat[-1]
    df_mix_const.loc[i, 'term_cond'] = term_cond[-1]

# check the compounds that pass the miscibility and stability tests
idx_misc = df_mix_const[(df_mix_const['misc_AE'] == True) & (df_mix_const['misc_BE'] == True)].index
print(datetime.now() - startTime)
#%%
"""
ISS procedure
"""
# perform ISS

# define the total pressure
P = 0.101325  # atmospheric in MPa
# define the initial guess for the composition
x0 = np.array([0.50, 0.30, 0.20]).reshape(1, 3)
# initialize containers for the result
df_perf = pd.DataFrame(np.nan, index=idx_misc, columns=['ED', 'R', 'S', 'C', 'SxC', 'T_iso', 'T', 'sol_stat', 'term_cond'])

ED = np.zeros(df_entrainer.shape[0])
R = np.zeros(df_entrainer.shape[0])
T = np.zeros(df_entrainer.shape[0])
x = np.zeros((df_entrainer.shape[0], 2, 3))

for i in idx_misc[2:]:
    print(i)
    # define the mixture
    df_mix = df_space.iloc[[0, 1, i]].reset_index(drop=True)
    # initial guess for the temperature
    T0 = 320 #df_props['Tb'][0]#(df_props['Tb'][0] + df_props['Tb'][i]) / 2
    # antoine parameters
    antoine_params = np.array(df_antoine[[0, 1, i]]).T
    # evaluate iss
    try:
        solution, model = iss_method_new(x0, T0, P, antoine_params, df_mix, verbose=False)
    except ValueError:
        pass
    # print status
    print('ISS procedure for (' + str(i) + '). solver status: ' + str(solution.solver.status) +
          ' & Termination condition: ' + str(solution.solver.termination_condition))

    # for j in model.x.keys():
    #     x[i-2, j[0], j[1]] = model.x[j].value
    # for j, k in enumerate(model.T.keys()):
    #     T[i - 2] = model.T[k].value
    # for j, k in enumerate(model.ED.keys()):
    #     ED[i - 2] = model.ED[k].value
    # for j, k in enumerate(model.R.keys()):
    #     R[i - 2] = model.R[k].value
    # for j, k in enumerate(model.x.keys()):
    #     x[i - 2][j // 3][j % 3] = model.x[k].value

    df_perf.loc[i, 'R'] = model.R.value
    df_perf.loc[i, 'ED'] = model.ED.value
    df_perf.loc[i, 'S'] = df_mix_const.loc[i, 'S']
    df_perf.loc[i, 'C'] = df_mix_const.loc[i, 'C']
    df_perf.loc[i, 'SxC'] = df_mix_const.loc[i, 'SxC']
    df_perf.loc[i, 'T_iso'] = model.T[0].value
    df_perf.loc[i, 'T'] = model.T[1].value
    df_perf.loc[i, 'sol_stat'] = str(solution.solver.status)
    df_perf.loc[i, 'term_cond'] = str(solution.solver.termination_condition)
#
# # # Save the data from the iss procedure
# # index_1 = ['iso_x_'+str(i) for i in range(x.shape[0])]
# # index_2 = ['min_x_'+str(i) for i in range(x.shape[0])]
# # index_3 = []
# # df_iss = pd.DataFrame()
# # for i, j in zip(index_1, index_2):
# #     index_3.append(i)
# #     index_3.append(j)
# # df_iss = pd.DataFrame(x.reshape(10, 3), columns=['A', 'B', 'E'])
# # df_iss['info'] = index_3
# # df_iss = df_iss[['info', 'A', 'B', 'E']]
#%%
a = df_perf[df_perf['sol_stat']=='ok']