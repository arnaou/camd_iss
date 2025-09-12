"""
UNIFAC parameter compatibility checking.

This module provides functions to check if compounds and mixtures have
all required UNIFAC parameters available for thermodynamic calculations.
"""

import warnings
from typing import Dict, List, Tuple, Set, Optional, Union
from .load_unifac import load_unifac_data


def check_unifac_parameters(
    compounds: Union[Dict[str, Dict[str, int]], List[Dict[str, int]]],
    check_system_groups: Optional[Dict[str, Dict[str, int]]] = None,
    verbose: bool = True
) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """
    Check UNIFAC R/Q parameters and binary interaction parameters for compounds.
    
    Validates that all groups in the compounds have:
    1. R (volume) and Q (surface area) parameters for UNIFAC calculations
    2. Binary interaction parameters between different groups
    3. If check_system_groups provided, validates interactions with system components
    
    Args:
        compounds: Either a dict of compounds {'compound_name': {'group': count}}
                  or a list of compound dicts [{'group': count}, ...]
        check_system_groups: Optional system compounds to check interactions with
                           (e.g., binary mixture A+B components)
        verbose: Print detailed information about missing parameters
    
    Returns:
        tuple: (missing_rq_groups, missing_interactions, problem_summary)
               - missing_rq_groups: List of groups missing R/Q parameters
               - missing_interactions: List of group pairs missing interactions  
               - problem_summary: List of problem descriptions
    """
    # Normalize input to list of compound dictionaries
    if isinstance(compounds, dict):
        # Convert from compound format to list format
        compound_list = list(compounds.values())
        compound_names = list(compounds.keys())
    else:
        compound_list = compounds
        compound_names = [f"compound_{i}" for i in range(len(compound_list))]
    
    if not compound_list:
        if verbose:
            print("No compounds to check")
        return [], [], ["No compounds provided"]
    
    # Collect all unique groups from compounds
    all_compound_groups = set()
    for compound in compound_list:
        all_compound_groups.update(compound.keys())
    
    # Add system groups if provided
    all_groups = set(all_compound_groups)
    system_groups = set()
    if check_system_groups:
        for system_compound in check_system_groups.values():
            system_groups.update(system_compound.keys())
        all_groups.update(system_groups)
    
    if verbose:
        print(f"Checking UNIFAC parameters for {len(compound_list)} compounds")
        print(f"Groups to validate: {sorted(all_groups)}")
        if system_groups:
            print(f"System groups: {sorted(system_groups)}")
    
    # Load UNIFAC data
    try:
        rk_data, qk_data, interaction_data = load_unifac_data()
    except Exception as e:
        warning_msg = f"Could not load UNIFAC data: {e}"
        if verbose:
            warnings.warn(warning_msg, UserWarning)
        return list(all_groups), [], [warning_msg]
    
    # Check R/Q parameters
    missing_rq_groups = []
    for group in all_groups:
        rk_missing = group not in rk_data or rk_data[group] is None
        qk_missing = group not in qk_data or qk_data[group] is None
        
        if rk_missing or qk_missing:
            missing_rq_groups.append(group)
            if verbose:
                rk_status = "missing" if group not in rk_data else "None" if rk_data[group] is None else f"{rk_data[group]}"
                qk_status = "missing" if group not in qk_data else "None" if qk_data[group] is None else f"{qk_data[group]}"
                print(f"    Missing R/Q for '{group}': RK={rk_status}, QK={qk_status}")
    
    # Check binary interaction parameters
    missing_interactions = []
    groups_list = list(all_groups)
    
    for i, group_i in enumerate(groups_list):
        for j, group_j in enumerate(groups_list):
            if i <= j:  # Check upper triangular part including diagonal
                # Check if interaction exists
                interaction_found = False
                
                if (group_i in interaction_data and 
                    group_j in interaction_data[group_i] and 
                    interaction_data[group_i][group_j] is not None):
                    interaction_found = True
                elif (group_j in interaction_data and 
                      group_i in interaction_data[group_j] and 
                      interaction_data[group_j][group_i] is not None):
                    interaction_found = True
                
                if not interaction_found and group_i != group_j:
                    # Only report missing interactions between different groups
                    missing_interactions.append((group_i, group_j))
                    if verbose:
                        print(f"    Missing interaction: {group_i} <-> {group_j}")
    
    # Generate problem summary
    problem_summary = []
    if missing_rq_groups:
        problem_summary.append(f"Missing R/Q parameters for {len(missing_rq_groups)} groups: {missing_rq_groups}")
    if missing_interactions:
        problem_summary.append(f"Missing {len(missing_interactions)} interaction parameters")
    
    if verbose:
        if not missing_rq_groups and not missing_interactions:
            print("✓ All UNIFAC parameters are available")
        else:
            print(f"⚠ Found {len(missing_rq_groups)} missing R/Q groups and {len(missing_interactions)} missing interactions")
    
    return missing_rq_groups, missing_interactions, problem_summary


def check_binary_compatibility(
    mixture: Dict[str, Dict[str, int]],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Check if a binary mixture has all required UNIFAC parameters.
    
    This function checks binary system compatibility for CAMD workflows by
    ensuring all required R/Q parameters and interaction data are available.
    
    Args:
        mixture: Binary mixture in compound format {'compound_name': {'group': count}}
        verbose: Print compatibility information
    
    Returns:
        dict: Compatibility results with keys:
              - 'compatible': bool, True if system is compatible
              - 'groups': list of groups in the system
              - 'missing_rq': list of groups missing R/Q parameters
              - 'missing_interactions': list of missing interaction pairs
              - 'summary': list of problem descriptions
    """
    if len(mixture) != 2:
        warning_msg = f"Expected binary mixture (2 compounds), got {len(mixture)}"
        if verbose:
            print(f"⚠ {warning_msg}")
        return {
            'compatible': False,
            'groups': [],
            'missing_rq': [],
            'missing_interactions': [],
            'summary': [warning_msg]
        }
    
    compound_names = list(mixture.keys())
    if verbose:
        print(f"Checking binary compatibility: {compound_names[0]} + {compound_names[1]}")
    
    # Check UNIFAC parameters
    missing_rq, missing_interactions, problems = check_unifac_parameters(
        mixture, verbose=verbose
    )
    
    # Get all groups
    all_groups = set()
    for compound_groups in mixture.values():
        all_groups.update(compound_groups.keys())
    
    is_compatible = len(missing_rq) == 0 and len(missing_interactions) == 0
    
    return {
        'compatible': is_compatible,
        'groups': sorted(all_groups),
        'missing_rq': missing_rq,
        'missing_interactions': missing_interactions, 
        'summary': problems
    }


def check_ternary_compatibility(
    binary_mixture: Dict[str, Dict[str, int]],
    solvent_candidates: List[Dict[str, int]],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Check if solvent candidates are compatible with a binary mixture for ternary VLE.
    
    Args:
        binary_mixture: Binary mixture {'compound_name': {'group': count}}
        solvent_candidates: List of solvent dictionaries [{'group': count}, ...]
        verbose: Print detailed information
    
    Returns:
        dict: Results with keys:
              - 'compatible_solvents': list of indices of compatible solvents
              - 'incompatible_solvents': list of indices with missing parameters
              - 'missing_parameters': dict mapping solvent index to missing parameters
    """
    if verbose:
        print(f"Checking ternary compatibility for {len(solvent_candidates)} solvents with binary mixture")
    
    # First check the binary system compatibility
    binary_check = check_binary_compatibility(binary_mixture, verbose=False)
    if not binary_check['compatible']:
        if verbose:
            print("⚠ Binary system is not compatible for UNIFAC calculations")
        return {
            'compatible_solvents': [],
            'incompatible_solvents': list(range(len(solvent_candidates))),
            'missing_parameters': {i: binary_check['summary'] for i in range(len(solvent_candidates))}
        }
    
    compatible_solvents = []
    incompatible_solvents = []
    missing_parameters = {}
    
    for i, solvent in enumerate(solvent_candidates):
        # Create ternary system by combining binary mixture with solvent
        ternary_system = dict(binary_mixture)  # Copy binary mixture
        ternary_system[f'solvent_{i}'] = solvent  # Add solvent
        
        # Check UNIFAC parameters for ternary system
        missing_rq, missing_interactions, problems = check_unifac_parameters(
            ternary_system, verbose=False
        )
        
        if len(missing_rq) == 0 and len(missing_interactions) == 0:
            compatible_solvents.append(i)
        else:
            incompatible_solvents.append(i)
            missing_parameters[i] = problems
    
    if verbose:
        print(f"✓ {len(compatible_solvents)} compatible solvents")
        print(f"⚠ {len(incompatible_solvents)} incompatible solvents")
    
    return {
        'compatible_solvents': compatible_solvents,
        'incompatible_solvents': incompatible_solvents,
        'missing_parameters': missing_parameters
    }