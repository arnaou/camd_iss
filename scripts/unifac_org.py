########################################################################################################################
#
#   Scripts that implements a pyomo model for the UNIFAC model as described by Smith and Van Ness
#
#   Example: binary system of diethylamine(1)/n-heptane(2) at 308.15 K and x1= 0.4, x2=0.6
#
#
#
#   Adem R.N. Aouichaoui, arnaou@kt.dtu.dk
########################################################################################################################

# import packages
import pandas as pd
import numpy as np
from utils.gc import load_gc, unifac
import pyomo.environ as pyo
from utils.pyom import parse_pyomo_log, ensure_file_exists_in_cwd

# Define the binary system
df_mix = pd.DataFrame({'CH3': [2, 2], 'CH2': [1, 5], 'CH2NH': [1, 0]})
df_mix['compound'] = ['DEA', 'HEP']
df_mix.set_index('compound', inplace=True)

# define the temperature
T = 308.15

# define the compositions
x = np.array([[0.4, 0.6]])

# load the unifac parameters
file_path = '../data/unifac.xlsx'
df_gc, df_a = load_gc(file_path, output_format='dataframe')

# define no of components and data points
n_data = x.shape[0]
n_comp = x.shape[1]

# calculate activity coefficients using unifac
gamma = unifac(df_mix, x, T, df_gc, df_a)

# calculate unifac activity coefficient using the numerically stable version
def unifac_stable(mixture: pd.DataFrame, x0: float, T0: float, df_gc: pd.DataFrame, df_a: pd.DataFrame) -> np.array:
    """

    :param mixture:
    :param x0:
    :param T0:
    :param df_gc:
    :param df_a:
    :return:
    """
    v = np.array(mixture)
    Rk = np.array(df_gc['Rk'][mixture.columns])  # np.array(df_gc['Rk'][df_mix.columns])
    Qk = np.array(df_gc['Qk'][mixture.columns])
    a = np.array(df_a[mixture.columns].loc[mixture.columns])
    # start calculation: combinatorial term
    ri = np.dot(v, Rk)
    qi = np.dot(v, Qk)
    Ji = np.mat(ri) / np.mat(np.dot(x0, ri)).T
    Li = np.mat(qi) / np.mat(np.dot(x0, qi)).T
    ln_gamma_c = 1 - Ji + np.log(Ji) - 5 * np.multiply(qi, (1 - Ji / Li + np.log(Ji / Li)))
    # start calculation: residual term
    ei = v * Qk / qi
    psi = np.exp(-a / T0)
    beta = 0
    theta = 0
    omega = 0

    ln_gamma_r = 0

    gamma = np.array(np.exp(ln_gamma_c + ln_gamma_r))
    return gamma












#
# # create the model
# model = pyo.ConcreteModel('unifac smith and van ness - example')
#
# # define sets
# model.k = pyo.Set(initialize=df_mix.columns, name='Groups present')
# model.i = pyo.RangeSet(0, n_comp - 1, name=' no. of Compounds')
# model.dp = pyo.RangeSet(0, n_data - 1, name='no. of data points')
# model.grp = pyo.Set(initialize=df_a.columns, name='all Groups')
#
# # define parameters
# model.a = pyo.Param(model.grp, model.grp, initialize=df_a.stack().to_dict(), mutable=False, name='unifac interaction a_mk', domain=pyo.Reals)
# model.Rk = pyo.Param(model.grp, initialize=df_gc.Rk.to_dict(), mutable=False, name='unifac Rk parameters')
# model.Qk = pyo.Param(model.grp, initialize=df_gc.Qk.to_dict(), mutable=False, name='unifac Qk parameters')
# model.x = pyo.Param(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(x)}, mutable=False, name='Liquid composition x', domain=pyo.Reals)
# model.v = pyo.Param(model.i, model.k, initialize=df_mix.stack().to_dict(), mutable=False, name='compounds')
# model.T = pyo.Param(model.i, initialize=T, mutable=False, name='temperature in K')
#
