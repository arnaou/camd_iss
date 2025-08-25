"""
Core UNIFAC calculation function.

Implements the UNIFAC method for calculating activity coefficients in liquid mixtures
following the Smith et al. (2001) formulation.
"""
import numpy as np


def unifac(mixture, x, T, rk_data, qk_data, interaction_data):
    """
    Calculate UNIFAC activity coefficients for given mixture, composition, and temperature.
    
    Args:
        mixture: Dict {group: [counts for each component]}
        x: Component mole fractions array (2D: n_datapoints × n_components)
        T: Temperature array (1D: n_temperatures)
        rk_data, qk_data: Group parameter dictionaries
        interaction_data: Binary interaction parameter dictionary
    
    Returns:
        Dict with calculation results including 'gamma', intermediate values
    """
    # Get groups from mixture
    groups = list(mixture.keys())
    
    # UNIFAC calculations
    v = np.array([mixture[group] for group in groups])  # Shape: (n_groups, n_components)
    Rk = np.array([rk_data[group] for group in groups])
    Qk = np.array([qk_data[group] for group in groups])
    
    # Build interaction parameter matrix
    n_groups = len(groups)
    a = np.zeros((n_groups, n_groups))
    for i, group_i in enumerate(groups):
        for j, group_j in enumerate(groups):
            a[i, j] = interaction_data[group_i][group_j]
    
    # Calculate ri
    r = np.dot(v.T, Rk)
    # Calculate qi
    q = np.dot(v.T, Qk)
    # Calculate Li
    L = q / np.dot(x, q).T
    # Calculate Ji
    J = r / np.dot(x, r).T 
    # Calculate gammaC
    log_gamma_c = 1 - J + np.log(J) - 5 * np.multiply(q, (1 - J / L + np.log(J / L)))
    # Calculate ei,k
    e = (v.T * Qk) / q[:, np.newaxis]
    # Calculate theta
    theta = np.dot(x, v.T * Qk) / np.dot(x, q)
    # Calculate psi
    psi = np.exp(-a / T)
    # Calculate beta
    beta = np.dot(v.T * Qk, psi) / q[:, np.newaxis]
    # Calculate omega
    omega = np.dot(theta, psi) 
    # Calculate gammaR
    log_gamma_r = q * (1 - np.sum(theta * beta / omega - e * np.log(beta / omega), axis=1))
    # Calculate total gamma
    gamma = np.exp(log_gamma_c + log_gamma_r)
    
    # Wrap the results in a dictionary
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