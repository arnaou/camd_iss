"""
script for testing unifac implementation:
    first a vectorized unifac model is implemented
    then a pyomo version is made
"""

# imports
import json
import warnings
from pathlib import Path
import numpy as np
import pyomo.environ as pyo

#%% define the path for the data
data_path = Path(__file__).parent / "data" / "json"

# Load R/Q parameters
with open(data_path / 'rk.json', 'r') as f:
    rk_data = json.load(f).get("data", {})
with open(data_path / 'qk.json', 'r') as f:
    qk_data = json.load(f).get("data", {})
# Load interaction parameters
with open(data_path / 'a.json', 'r') as f:
    interaction_data = json.load(f).get("data", {})

#%% define the system

mixture  = {
        'CH3':   [2, 2],      # 2 CH3 groups in diethylamine, 2 in heptane
        'CH2':   [1, 5],      # 1 CH2 groups in diethylamine, 5 in heptane
        'CH2NH': [1, 0],      # 1 CH2NH group in diethylamine only
    }

# Component mole fractions x_i
x = np.array([[0.4, 0.6]])  # 40% diethylamine, 60% heptane
T = np.array([308.15])  # Temperature T_t in Kelvin
# start the unifac calculations
groups = list(mixture.keys())

def unifac(mixture, x, T, rk,qk, interaction_data):
    """

    """
    #unifac starts here
    v = np.array([mixture[group] for group in groups])  # Shape: (n_groups, n_components)
    Rk = np.array([rk_data[group] for group in groups])
    Qk = np.array([qk_data[group] for group in groups])
    # Build interaction parameter matrix
    n_groups = len(groups)
    a = np.zeros((n_groups, n_groups))
    for i, group_i in enumerate(groups):
        for j, group_j in enumerate(groups):
            a[i, j] = interaction_data[group_i][group_j]
    # calculate ri
    r = np.dot(v.T, Rk)
    # calculate qi
    q = np.dot(v.T, Qk)
    # calculate Li
    L = q / np.dot(x, q).T
    # calculate Ji
    J = r / np.dot(x, r).T 
    # calculate gammaC
    log_gamma_c = 1 - J + np.log(J) - 5 * np.multiply(q, (1 - J / L + np.log(J / L)))
    # caculate ei,k
    e = (v.T * Qk) / q[:, np.newaxis]
    # calculate theta
    theta = np.dot(x, v.T * Qk) / np.dot(x, q)
    # calculate psi
    psi = np.exp(-a / T)
    # calculate beta
    beta = np.dot(v.T * Qk, psi) / q[:, np.newaxis]
    # calculate omega
    omega = np.dot(theta, psi) 
    # calculate gammaR
    log_gamma_r = q * (1 - np.sum(theta * beta / omega - e * np.log(beta / omega), axis=1))
    # calculate total gamma
    gamma = np.exp(log_gamma_c + log_gamma_r)
    # print for validation
    print("Activity coefficients: ", gamma.flatten())
    # wrap the results in a dictionary
    results = {
        "J": J,
        "L": L,
        "log_gamma_c": log_gamma_c,
        "e": e,
        "theta": theta,
        "psi": psi,
        "omega": omega,
        "log_gamma_r": log_gamma_r,
        "gamma": gamma.flatten()
    }
    return results

results = unifac(mixture, x, T, rk_data, qk_data, interaction_data)
# %%

def unifac_wrapper(mixture, x_input, T_input, rk_data, 
  qk_data, interaction_data):
      """
      Flexible UNIFAC wrapper that handles various input 
  combinations.
      
      Args:
          mixture: Dict {group: [counts for each component]}
          x_input: Single composition array or multiple 
  compositions
          T_input: Single temperature or multiple temperatures
          rk_data, qk_data, interaction_data: UNIFAC parameter
   dictionaries
      
      Returns:
          gamma: Activity coefficients array with appropriate 
  shape
      """

      # Normalize inputs to consistent shapes
      x = np.atleast_2d(x_input)  # Shape: (n_datapoints, n_components)
      T = np.atleast_1d(T_input)  # Shape: (n_temperatures,)

      # Validate input combinations
      if len(T) > 1 and x.shape[0] > 1:
          if len(T) != x.shape[0]:
              raise ValueError(f"Multiple x ({x.shape[0]}) and T ({len(T)}) must have same length")

      # Determine calculation mode
      single_x = x.shape[0] == 1
      single_T = len(T) == 1

      if single_x and single_T:
          # Case 1: Single x, single T
          result = unifac(mixture, x, T, rk_data, qk_data,
  interaction_data)
          return result["gamma"]

      elif single_x and not single_T:
          # Case 2: Single x, multiple T  
          gamma_list = []
          for t in T:
              result = unifac(mixture, x, np.array([t]),
  rk_data, qk_data, interaction_data)
              gamma_list.append(result["gamma"])
          return np.array(gamma_list).T  # Shape: (n_components, n_temperatures)

      elif not single_x and single_T:
          # Case 3: Multiple x, single T
          gamma_list = []
          for x_i in x:
              result = unifac(mixture, np.array([x_i]), T,
  rk_data, qk_data, interaction_data)
              gamma_list.append(result["gamma"])
          return np.array(gamma_list)  # Shape: (n_datapoints, n_components)

      else:
          # Case 4: Multiple x, multiple T (same length)
          gamma_list = []
          for x_i, t_i in zip(x, T):
              result = unifac(mixture, np.array([x_i]),
  np.array([t_i]), rk_data, qk_data, interaction_data)
              gamma_list.append(result["gamma"])
          return np.array(gamma_list)  # Shape: (n_datapoints, n_components)
          
#%% testing
#single x single T
gamma = unifac_wrapper(mixture, [0.4, 0.6], 308.15, rk_data,
   qk_data, interaction_data)

# single x and mutiple T
gamma = unifac_wrapper(mixture, [0.4, 0.6], [308.15, 150.15], rk_data,
   qk_data, interaction_data)
# # Create flattened interaction data after loading JSON
# interaction_params = {}
# for group_i, group_dict in interaction_data.items():
#     for group_j, value in group_dict.items():
#         interaction_params[(group_i, group_j)] = value

# # as a dictionary comprehension:
# e_dict = {(i, group): e[i, k]
#         for i in range(e.shape[0])
#         for k, group in enumerate(groups)}

# theta_dict = {(i, group): theta[i,k]
#             for i in range(len(x))
#             for k, group in enumerate(groups)}

# psi_dict = {(t, group_i, group_j): psi[i, j]
#             for t in range(1)
#             for i, group_i in enumerate(groups)
#             for j, group_j in enumerate(groups)}

# omega_dict ={(t, group): omega[t, k]
#               for t in range(1)
#               for i, group in enumerate(groups)}
# #%%
# # create the model
# model = pyo.ConcreteModel()
# n_comp = x.shape[1]
# n_data = x.shape[0]

# # define the sets
# model.k = pyo.Set(initialize=groups, name='groups')
# model.i = pyo.RangeSet(0, n_comp - 1, name=' no. of Compounds')
# model.dp = pyo.RangeSet(0, n_data - 1, name='no. of data points')
# model.grp = pyo.Set(initialize=interaction_data.keys(), name='all groups')
# # define data
# model.Rk = pyo.Param(model.grp, initialize=rk_data, mutable=False, name='unifac Rk parameters')
# model.Qk = pyo.Param(model.grp, initialize=qk_data, mutable=False, name='unifac Qk parameters')
# model.a = pyo.Param(model.grp, model.grp, initialize=interaction_params, mutable=False,
#                     name='unifac interaction a_mk', domain=pyo.Reals)
# model.T = pyo.Param(model.dp, initialize=T, mutable=False, name='Temperature in K')
# model.x = pyo.Param(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(x)}, mutable=False,
#                     name='Liquid composition x', domain=pyo.Reals)
# # define constants: these variable do not change w.r.t. x and T
# model.r = pyo.Param(model.i, initialize={index: v for index, v in np.ndenumerate(r)}, mutable=False, name='unifac r_i')
# model.q = pyo.Param(model.i, initialize={index: v for index, v in np.ndenumerate(q)}, mutable=False, name='unifac q_i')
# model.e = pyo.Param(model.i, model.k, initialize=e_dict, mutable=False, name='unifac e_ik')

# # define variable, change with x but not T
# model.J = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(J)}, within=pyo.PositiveReals, name='unifac J_i')
# model.L = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(L)}, within=pyo.PositiveReals, name='unifac L_i')
# model.gamma_c = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(gamma_c)}, within=pyo.Reals, name='unifac gamma combinatorial')
# model.theta = pyo.Var(model.dp, model.k, initialize=theta_dict, within=pyo.Reals, name='unifac theta_k')

# # define variables that are dependent on x and T
# model.psi = pyo.Var(model.dp, model.k, model.k, initialize=psi_dict, within=pyo.PositiveReals, name='unifac psi_k')
# model.omega = pyo.Var(model.dp, model.k, initialize=omega_dict, within=pyo.PositiveReals, name='unifac omega_k')
# model.gamma_r = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(np.matrix(gamma_r))}, within=pyo.Reals, name='unifac gamma residual')
# model.gamma = pyo.Var(model.dp, model.i, initialize={index: v for index, v in np.ndenumerate(gamma)}, within=pyo.PositiveReals, name='unifac gamma total')

# # define the equations
# def eq_J():
#     return 

# def eq_L():
#     return m.L[dp, i] == 

# def eq_gamma_c():
#     return

# def eq_theta():
#     return

# def eq_psi():
#     return

# def eq_omega():
#     return

# def eq_gamma_r():
#     return

# def eq_gamma(m, dp, i):
#     return m.gamma[dp, i] == pyo.exp(m.gamma_r[dp, i] + m.gamma_c[dp, i])

# %%
