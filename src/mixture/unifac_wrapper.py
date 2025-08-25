"""
Flexible UNIFAC wrapper for various input combinations.

Handles single/multiple compositions and temperatures with appropriate output shapes.
"""
import numpy as np
from .unifac_core import unifac


def unifac_wrapper(mixture, x_input, T_input, rk_data, qk_data, interaction_data):
    """
    Flexible UNIFAC wrapper that handles various input combinations.
    
    Args:
        mixture: Dict {group: [counts for each component]}
        x_input: Single composition array or multiple compositions
                - Single: [0.4, 0.6] or [[0.4, 0.6]]
                - Multiple: [[0.4, 0.6], [0.3, 0.7], ...]
        T_input: Single temperature or multiple temperatures
                - Single: 308.15 or [308.15]
                - Multiple: [308.15, 318.15, ...]
        rk_data, qk_data, interaction_data: UNIFAC parameter dictionaries
    
    Returns:
        gamma: Activity coefficients array
               Shape depends on inputs:
               - Single x, single T: (n_components,)
               - Single x, multiple T: (n_components, n_temperatures)
               - Multiple x, single T: (n_datapoints, n_components) 
               - Multiple x, multiple T: (n_datapoints, n_components)
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
        result = unifac(mixture, x, T, rk_data, qk_data, interaction_data)
        return result["gamma"]
        
    elif single_x and not single_T:
        # Case 2: Single x, multiple T  
        gamma_list = []
        for t in T:
            result = unifac(mixture, x, np.array([t]), rk_data, qk_data, interaction_data)
            gamma_list.append(result["gamma"])
        return np.array(gamma_list).T  # Shape: (n_components, n_temperatures)
        
    elif not single_x and single_T:
        # Case 3: Multiple x, single T
        gamma_list = []
        for x_i in x:
            result = unifac(mixture, np.array([x_i]), T, rk_data, qk_data, interaction_data)
            gamma_list.append(result["gamma"])
        return np.array(gamma_list)  # Shape: (n_datapoints, n_components)
        
    else:
        # Case 4: Multiple x, multiple T (same length)
        gamma_list = []
        for x_i, t_i in zip(x, T):
            result = unifac(mixture, np.array([x_i]), np.array([t_i]), rk_data, qk_data, interaction_data)
            gamma_list.append(result["gamma"])
        return np.array(gamma_list)  # Shape: (n_datapoints, n_components)