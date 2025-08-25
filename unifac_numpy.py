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
        "gamma": gamma
    }
    return results

results = unifac(mixture, x, T, rk_data, qk_data, interaction_data)
# %%
          
#%% testing
#%%
# Import from the src package
from src import unifac_wrapper, load_unifac_data

# Load data once
rk_data, qk_data, interaction_data =load_unifac_data()

# Define your mixture
mixture = {
    'CH3': [2, 2],
    'CH2': [1, 5],
    'CH2NH': [1, 0]
}

# Use wrapper for different input combinations
gamma_single = unifac_wrapper(mixture, [0.4, 0.6],
308.15, rk_data, qk_data, interaction_data)
gamma_multi_T = unifac_wrapper(mixture, [0.4, 0.6],
[308.15, 318.15], rk_data, qk_data,
interaction_data)
gamma_multi_x = unifac_wrapper(mixture, [[0.4, 0.6],
[0.3, 0.7]], 308.15, rk_data, qk_data,
interaction_data)
gamma_multi_both = unifac_wrapper(mixture,
[[0.4, 0.6], [0.3, 0.7]], [308.15, 318.15], rk_data,
qk_data, interaction_data)

# %%

