"""
Simplified UNIFAC wrapper for paired temperature and composition calculations.

Handles single points or arrays of equal length T and x with consistent output shapes.
"""
import numpy as np
from .equations import unifac_equations


def _convert_mixture_format(mixture):
    """
    Convert mixture from compound format to group format.
    
    Args:
        mixture: Compound format {compound_name: {group: count}}
    
    Returns:
        Group format {group: [counts for each component]}
    """
    # Extract compounds (the nested dicts)
    compounds = list(mixture.values())
    
    # Get all unique groups across all compounds
    all_groups = set()
    for compound in compounds:
        all_groups.update(compound.keys())
    all_groups = sorted(all_groups)
    
    # Build group format: {group: [count_in_comp1, count_in_comp2, ...]}
    group_format = {}
    for group in all_groups:
        group_format[group] = [compound.get(group, 0) for compound in compounds]
    
    return group_format


def unifac(mixture, x, T, rk_data, qk_data, interaction_data):
    """
    Simplified UNIFAC wrapper for paired (T, x) calculations.
    
    Args:
        mixture: Compound format {compound_name: {group: count}}
                Example: {'heptane': {'CH3': 2, 'CH2': 5}, 'DEA': {'CH3': 2, 'CH2': 1, 'CH2NH': 1}}
        x: Numpy array of compositions with shape (n_points, n_components)
        T: Numpy array of temperatures with shape (n_points,)
        rk_data, qk_data, interaction_data: UNIFAC parameter dictionaries
    
    Returns:
        gamma: Activity coefficients array with shape (n_points, n_components)
    """
    
    # Auto-convert mixture format 
    mixture = _convert_mixture_format(mixture)
    
    # Validate input types
    if not isinstance(x, np.ndarray):
        raise TypeError(f"x must be numpy array, got {type(x)}")
    if not isinstance(T, np.ndarray):
        raise TypeError(f"T must be numpy array, got {type(T)}")
    
    # Validate input shapes
    if x.ndim != 2:
        raise ValueError(f"x must be 2D array with shape (n_points, n_components), got shape {x.shape}")
    if T.ndim != 1:
        raise ValueError(f"T must be 1D array with shape (n_points,), got shape {T.shape}")
    
    # Validate input lengths
    if len(T) != x.shape[0]:
        raise ValueError(f"T and x must have same length: len(T)={len(T)}, x.shape[0]={x.shape[0]}")
    
    # Handle single point case
    if len(T) == 1:
        result = unifac_equations(mixture, x, T, rk_data, qk_data, interaction_data)
        # Ensure gamma shape is (1, n_components) not flattened
        gamma = result["gamma"]
        if gamma.ndim == 1:
            gamma = gamma.reshape(1, -1)
            result["gamma"] = gamma
        return result
    
    # Handle multiple points case
    results_list = []
    for i in range(len(T)):
        x_i = x[i:i+1]  # Keep 2D shape: (1, n_components)
        T_i = T[i:i+1]  # Keep 1D shape: (1,)
        result = unifac_equations(mixture, x_i, T_i, rk_data, qk_data, interaction_data)
        results_list.append(result)
    
    # Combine all results
    combined_results = {}
    for key in results_list[0].keys():
        if key in ["r", "q", "J", "L", "log_gamma_c", "log_gamma_r", "gamma"]:
            # Stack component-wise results
            combined_results[key] = np.vstack([r[key] for r in results_list])
        elif key in ["e"]:
            # Stack group-component results  
            combined_results[key] = np.vstack([r[key] for r in results_list])
        elif key in ["theta", "psi", "omega"]:
            # Stack data-point wise results
            combined_results[key] = np.array([r[key] for r in results_list])
    
    return combined_results