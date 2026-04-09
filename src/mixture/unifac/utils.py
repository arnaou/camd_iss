"""
Shared utility functions for UNIFAC modules.
"""

import numpy as np


def convert_mixture_format(mixture):
    """
    Convert mixture from compound format to group format.

    Args:
        mixture: Compound format {compound_name: {group: count}}

    Returns:
        Group format {group: [counts for each component]}
    """
    compounds = list(mixture.values())

    all_groups = set()
    for compound in compounds:
        all_groups.update(compound.keys())
    all_groups = sorted(all_groups)

    group_format = {}
    for group in all_groups:
        group_format[group] = [compound.get(group, 0) for compound in compounds]

    return group_format


def convert_antoine_format(antoine_params):
    """
    Convert Antoine parameters from nested dict to array format.

    Args:
        antoine_params: Dict format {compound_name: {'A': val, 'B': val, 'C': val}}

    Returns:
        Array format (n_components, 3) with [A, B, C] for each component
    """
    compounds = list(antoine_params.keys())
    n_comp = len(compounds)

    antoine_array = np.zeros((n_comp, 3))
    for i, compound in enumerate(compounds):
        antoine_array[i, 0] = antoine_params[compound]['A']
        antoine_array[i, 1] = antoine_params[compound]['B']
        antoine_array[i, 2] = antoine_params[compound]['C']

    return antoine_array


def generate_initialization_values(x, T, mixture, rk_data, qk_data, interaction_data,
                                   antoine_params, include_K=False):
    """
    Generate initialization values using the UNIFAC interface for better solver convergence.

    Used by both pyo_vle (include_K=False) and pyo_iss (include_K=True).

    Args:
        x: liquid composition array (n_data, n_components)
        T: temperature array (n_data,) in Kelvin
        mixture: compound format {compound_name: {group: count}}
        rk_data, qk_data, interaction_data: UNIFAC parameter dictionaries
        antoine_params: Antoine parameters dict
        include_K: if True, also return K-value initialization array (for ISS)

    Returns:
        Dict with initialization values, or None if calculation failed.
    """
    try:
        from .interface import unifac
        results = unifac(mixture, x, T, rk_data, qk_data, interaction_data)

        antoine_array = convert_antoine_format(antoine_params)
        n_data, n_comp = x.shape

        # Vapour pressures via Antoine equation
        psat_init = np.zeros((n_data, n_comp))
        for j in range(n_data):
            for i in range(n_comp):
                A, B, C = antoine_array[i]
                psat_init[j, i] = 10 ** (A - B / (C + T[j]))

        P = 1.01325  # bar
        K_init = np.zeros((n_data, n_comp))
        y_init = np.zeros((n_data, n_comp))
        for j in range(n_data):
            for i in range(n_comp):
                K_init[j, i] = results['gamma'][j, i] * psat_init[j, i] / P
                y_init[j, i] = K_init[j, i] * x[j, i]
            y_init[j, :] /= np.sum(y_init[j, :])

        init_values = {
            'r': results['r'],
            'q': results['q'],
            'J': results['J'],
            'L': results['L'],
            'e': results['e'],
            'theta': results['theta'],
            'psi': results['psi'],
            'beta': results['beta'],
            'omega': results['omega'],
            'log_gamma_c': results['log_gamma_c'],
            'log_gamma_r': results['log_gamma_r'],
            'gamma': results['gamma'],
            'psat': psat_init,
            'y': y_init,
        }

        if include_K:
            init_values['K'] = K_init

        return init_values

    except Exception as e:
        print(f"Warning: Failed to generate initialization values: {e}")
        print("Using default initialization values instead.")
        return None
