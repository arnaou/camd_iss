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
