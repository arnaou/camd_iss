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
x = np.array([0.4, 0.6])  # 40% diethylamine, 60% heptane
T = np.array([308.15, 310, 400])  # Temperature T_t in Kelvin
T = T[:, np.newaxis, np.newaxis] 
# start the unifac calculations
groups = list(mixture.keys())

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
L = np.matrix(q) / np.matrix(np.dot(x, q)).T
# calculate Ji
J = np.matrix(r) / np.matrix(np.dot(x, r)).T 
# calculate gammaC
gamma_c = 1 - J + np.log(J) - 5 * np.multiply(q, (1 - J / L + np.log(J / L)))
# caculate ei,k
e = (v.T * Qk) / q[:, np.newaxis]
# calculate theta
theta = np.dot(x, v.T * Qk) / np.dot(x, q)
# calculate psi
psi = np.exp(-a / T)
# calculate beta
#beta = np.dot(v.T * Qk, psi) / q[:, np.newaxis]
vQk = v.T * Qk  # Shape: [n_components, n_groups]
beta = np.einsum('ik,tmk->it', vQk, psi) / q[:, np.newaxis]
# calculate omega
omega = np.einsum('k,tkm->t', theta, psi) 
# calculate gammaR
#gamma_r = q * (1 - np.sum(theta * beta / omega - e * np.log(beta / omega), axis=1))
gamma_r_term = (theta[:, np.newaxis, np.newaxis] * beta[np.newaxis, :, :] / omega[np.newaxis, np.newaxis, :] -
                e[np.newaxis, :, :] * np.log(beta[np.newaxis, :, :] / omega[np.newaxis, np.newaxis, :]))  # Shape: [n_temp, n_components]
gamma_r = q[:, np.newaxis] * (1 - np.sum(gamma_r_term, axis=0))
# calculate total gamma
#gamma = np.exp(gamma_c + gamma_r)
gamma = np.exp(gamma_c[:, np.newaxis] + gamma_r.T) 
# %%
