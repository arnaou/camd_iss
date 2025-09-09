"""
UNIFAC activity coefficient calculations.

This module provides UNIFAC activity coefficient calculations following the
Smith et al. (2001) formulation. The main interface is the `unifac` function
which handles array inputs and validation.

Example:
    from src.mixture.unifac import unifac
    from src.data import load_unifac_data
    
    rk_data, qk_data, interaction_data = load_unifac_data()
    
    mixture = {
        'heptane': {'CH3': 2, 'CH2': 5},
        'DEA': {'CH3': 2, 'CH2': 1, 'CH2NH': 1}
    }
    
    x = np.array([[0.6, 0.4]])  # Shape: (1, 2)
    T = np.array([308.15])      # Shape: (1,)
    
    results = unifac(mixture, x, T, rk_data, qk_data, interaction_data)
    gamma = results["gamma"]  # Activity coefficients
    
    # Access all intermediate results for debugging:
    # results["J"], results["L"], results["log_gamma_c"], 
    # results["log_gamma_r"], results["e"], results["theta"], 
    # results["psi"], results["omega"]
"""

from .interface import unifac
from .equations import unifac_equations

__all__ = ['unifac', 'unifac_equations']