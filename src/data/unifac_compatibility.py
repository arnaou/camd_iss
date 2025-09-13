"""
UNIFAC parameter compatibility checking.

This module provides functions to check if compounds and mixtures have
all required UNIFAC parameters available for thermodynamic calculations.
"""

import warnings
import pickle
from pathlib import Path
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


def check_binary_interaction(
    mixture: Optional[Dict[str, Dict[str, int]]] = None,
    solvent_candidates: Optional[List[Dict[str, int]]] = None,
    data_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Enhanced UNIFAC binary interaction parameter checking with main group logic.

    This function replaces both check_binary_compatibility and check_ternary_compatibility
    with sophisticated main group interaction logic. "Binary" refers to pairwise group
    interactions, not the number of components in the mixture.

    Args:
        mixture: Base mixture in compound format {'compound_name': {'group': count}}
                Can be binary, ternary, or any number of components
        solvent_candidates: Optional list of additional solvents [{'group': count}, ...]
                          If provided, each solvent is tested with the base mixture
        data_path: Optional path to data directory containing UNIFAC files
        verbose: Print detailed compatibility information

    Returns:
        dict: Enhanced compatibility results with keys:
              - 'base_mixture_compatible': bool, compatibility of base mixture
              - 'compatible_solvents': list of indices of compatible solvents (if applicable)
              - 'incompatible_solvents': list of indices with critical issues (if applicable)
              - 'base_analysis': detailed analysis of base mixture
              - 'solvent_analysis': list of detailed analysis for each solvent (if applicable)
              - 'summary': overall summary statistics
    """
    results = {
        'base_mixture_compatible': False,
        'compatible_solvents': [],
        'incompatible_solvents': [],
        'base_analysis': {},
        'solvent_analysis': [],
        'summary': {}
    }

    # Validate inputs
    if mixture is None:
        if verbose:
            print("⚠ No base mixture provided")
        results['summary'] = {'error': 'No mixture provided'}
        return results

    if verbose:
        compound_names = list(mixture.keys())
        print(f"Enhanced binary interaction check for {len(compound_names)}-component system: {', '.join(compound_names)}")
        if solvent_candidates:
            print(f"  + {len(solvent_candidates)} solvent candidates to evaluate")

    # Check base mixture compatibility
    base_check = check_interaction_compatibility(
        compounds=mixture,
        data_path=data_path,
        verbose=verbose
    )

    results['base_mixture_compatible'] = base_check['compatible']
    results['base_analysis'] = base_check

    # If base mixture has critical issues, don't proceed with solvent checking
    if not base_check['compatible']:
        if verbose:
            print("⚠ Base mixture has critical compatibility issues - skipping solvent evaluation")
            for issue in base_check['summary']:
                print(f"  - {issue}")

        if solvent_candidates:
            results['incompatible_solvents'] = list(range(len(solvent_candidates)))
            results['summary'] = {
                'base_issues': base_check['summary'],
                'solvent_candidates': len(solvent_candidates),
                'compatible_solvents': 0,
                'compatibility_rate': 0.0
            }

        return results

    # If solvent candidates provided, test each one
    if solvent_candidates:
        if verbose:
            print(f"✓ Base mixture compatible - evaluating {len(solvent_candidates)} solvent candidates")

        compatible_solvents = []
        incompatible_solvents = []

        for i, solvent in enumerate(solvent_candidates):
            # Create extended system by adding solvent to base mixture
            extended_system = dict(mixture)  # Copy base mixture
            extended_system[f'candidate_{i}'] = solvent  # Add solvent candidate

            # Check compatibility of extended system
            solvent_check = check_interaction_compatibility(
                compounds=extended_system,
                data_path=data_path,
                verbose=False  # Suppress individual output for batch processing
            )

            # Store detailed analysis
            analysis_entry = {
                'candidate_index': i,
                'candidate_groups': list(solvent.keys()),
                'compatible': solvent_check['compatible'],
                'main_groups': solvent_check['main_groups'],
                'critical_missing_interactions': len(solvent_check['missing_interactions']),
                'total_interaction_gaps': sum(len(gaps) for gaps in solvent_check['interaction_matrix_gaps'].values()),
                'issues': solvent_check['summary'],
                'detailed_results': solvent_check
            }
            results['solvent_analysis'].append(analysis_entry)

            if solvent_check['compatible']:
                compatible_solvents.append(i)
                if verbose:
                    solvent_groups = list(solvent.keys())
                    print(f"    ✓ Candidate {i}: {solvent_groups}")
            else:
                incompatible_solvents.append(i)
                if verbose:
                    solvent_groups = list(solvent.keys())
                    critical_count = len(solvent_check['missing_interactions'])
                    print(f"    ⚠ Candidate {i}: {solvent_groups} - {critical_count} critical issues")

        results['compatible_solvents'] = compatible_solvents
        results['incompatible_solvents'] = incompatible_solvents

        # Generate summary statistics
        compatibility_rate = len(compatible_solvents) / len(solvent_candidates) if solvent_candidates else 0
        results['summary'] = {
            'base_compatible': True,
            'solvent_candidates': len(solvent_candidates),
            'compatible_solvents': len(compatible_solvents),
            'incompatible_solvents': len(incompatible_solvents),
            'compatibility_rate': compatibility_rate
        }

        if verbose:
            print(f"\nSummary:")
            print(f"  ✓ {len(compatible_solvents)} compatible candidates ({compatibility_rate:.1%})")
            print(f"  ⚠ {len(incompatible_solvents)} incompatible candidates")

            if incompatible_solvents:
                # Analyze common issues
                issue_counts = {}
                for analysis in results['solvent_analysis']:
                    if not analysis['compatible']:
                        for issue in analysis['issues']:
                            issue_counts[issue] = issue_counts.get(issue, 0) + 1

                print(f"  Most common issues:")
                for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    - {issue} (affects {count} candidates)")

    else:
        # Only base mixture analysis requested
        results['summary'] = {
            'base_compatible': base_check['compatible'],
            'base_groups': len(base_check['groups']),
            'base_main_groups': len(base_check['main_groups']),
            'critical_issues': len(base_check['missing_interactions']),
            'rq_issues': len(base_check['missing_rq'])
        }

        if verbose:
            print(f"Base mixture analysis complete:")
            if base_check['compatible']:
                print(f"  ✓ All critical parameters available")
                print(f"  Groups: {len(base_check['groups'])}, Main groups: {len(base_check['main_groups'])}")
            else:
                print(f"  ⚠ {len(base_check['missing_interactions'])} critical interaction issues")
                print(f"  ⚠ {len(base_check['missing_rq'])} R/Q parameter issues")

    return results


def _load_main_group_mapping(data_path: Optional[Path] = None) -> Dict[str, int]:
    """
    Load UNIFAC main group mapping from pickle file.

    Args:
        data_path: Optional path to directory containing unifac_groups.pkl
                  Defaults to ../../../data relative to this file

    Returns:
        dict: Mapping from group name to main group number
              e.g., {'CH3': 1, 'CH2': 1, 'OH': 5, ...}

    Raises:
        FileNotFoundError: If pickle file is not found
        pickle.UnpicklingError: If pickle file is malformed
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data"
    else:
        data_path = Path(data_path)

    pickle_file = data_path / "unifac_groups.pkl"

    try:
        with open(pickle_file, 'rb') as f:
            unifac_dict = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"UNIFAC group mapping file not found: {pickle_file}")
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Invalid pickle file: {e}")

    # Convert from {id: {'main_g': int, 'grp': str}} to {'grp': main_g}
    group_to_main = {}
    for entry in unifac_dict.values():
        if 'grp' in entry and 'main_g' in entry:
            group_to_main[entry['grp']] = entry['main_g']

    return group_to_main


def check_interaction_compatibility(
    compounds: Union[Dict[str, Dict[str, int]], List[Dict[str, int]]],
    check_system_groups: Optional[Dict[str, Dict[str, int]]] = None,
    data_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Enhanced UNIFAC parameter checking with main group interaction logic.

    This function implements the sophisticated interaction checking logic from
    inspect_interractions, which only flags missing interactions between different
    main functional groups (e.g., alkyl vs alcohol), not within the same main group.

    Args:
        compounds: Either a dict of compounds {'compound_name': {'group': count}}
                  or a list of compound dicts [{'group': count}, ...]
        check_system_groups: Optional system compounds to check interactions with
        data_path: Optional path to data directory containing UNIFAC files
        verbose: Print detailed information about missing parameters

    Returns:
        dict: Enhanced compatibility results with keys:
              - 'compatible': bool, True if system is fully compatible
              - 'groups': list of all functional groups in the system
              - 'main_groups': list of all main groups in the system
              - 'missing_rq': list of groups missing R/Q parameters
              - 'missing_interactions': list of missing interaction pairs (main groups)
              - 'missing_subgroup_interactions': list of missing subgroup pairs
              - 'interaction_matrix_gaps': dict with detailed gap analysis
              - 'summary': list of problem descriptions
    """
    # Normalize input to list of compound dictionaries
    if isinstance(compounds, dict):
        compound_list = list(compounds.values())
        compound_names = list(compounds.keys())
    else:
        compound_list = compounds
        compound_names = [f"compound_{i}" for i in range(len(compound_list))]

    if not compound_list:
        if verbose:
            print("No compounds to check")
        return {
            'compatible': False,
            'groups': [],
            'main_groups': [],
            'missing_rq': [],
            'missing_interactions': [],
            'missing_subgroup_interactions': [],
            'interaction_matrix_gaps': {},
            'summary': ["No compounds provided"]
        }

    # Collect all unique groups
    all_compound_groups = set()
    for compound in compound_list:
        all_compound_groups.update(compound.keys())

    all_groups = set(all_compound_groups)
    system_groups = set()
    if check_system_groups:
        for system_compound in check_system_groups.values():
            system_groups.update(system_compound.keys())
        all_groups.update(system_groups)

    if verbose:
        print(f"Enhanced interaction checking for {len(compound_list)} compounds")
        print(f"Groups to validate: {sorted(all_groups)}")
        if system_groups:
            print(f"System groups: {sorted(system_groups)}")

    # Load UNIFAC data and main group mapping
    try:
        rk_data, qk_data, interaction_data = load_unifac_data(data_path)
        group_to_main = _load_main_group_mapping(data_path)
    except Exception as e:
        warning_msg = f"Could not load UNIFAC data: {e}"
        if verbose:
            warnings.warn(warning_msg, UserWarning)
        return {
            'compatible': False,
            'groups': list(all_groups),
            'main_groups': [],
            'missing_rq': list(all_groups),
            'missing_interactions': [],
            'missing_subgroup_interactions': [],
            'interaction_matrix_gaps': {},
            'summary': [warning_msg]
        }

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

    # Enhanced interaction parameter checking with main group logic
    missing_main_group_interactions = []
    missing_subgroup_interactions = []
    interaction_matrix_gaps = {}

    # Get main groups for all functional groups
    main_groups = set()
    group_main_mapping = {}
    for group in all_groups:
        if group in group_to_main:
            main_group = group_to_main[group]
            main_groups.add(main_group)
            group_main_mapping[group] = main_group
        else:
            if verbose:
                print(f"    Warning: Group '{group}' not found in main group mapping")
            group_main_mapping[group] = None

    groups_list = list(all_groups)
    total_interactions = len(groups_list) * (len(groups_list) - 1) // 2
    missing_count = 0

    # Check interaction parameters (upper triangular matrix)
    for i, group_i in enumerate(groups_list):
        for j in range(i + 1, len(groups_list)):
            group_j = groups_list[j]

            # Check if interaction parameter exists
            interaction_found = False

            if (group_i in interaction_data and
                group_j in interaction_data[group_i] and
                interaction_data[group_i][group_j] is not None and
                interaction_data[group_i][group_j] != 0):
                interaction_found = True
            elif (group_j in interaction_data and
                  group_i in interaction_data[group_j] and
                  interaction_data[group_j][group_i] is not None and
                  interaction_data[group_j][group_i] != 0):
                interaction_found = True

            # If interaction is missing or zero
            if not interaction_found:
                missing_count += 1
                main_g1 = group_main_mapping.get(group_i)
                main_g2 = group_main_mapping.get(group_j)

                # Only flag as problematic if between different main groups
                if main_g1 is not None and main_g2 is not None and main_g1 != main_g2:
                    missing_main_group_interactions.append((main_g1, main_g2))
                    missing_subgroup_interactions.append((group_i, group_j))

                    if verbose:
                        print(f"    Missing interaction between main groups {main_g1}-{main_g2}: {group_i} <-> {group_j}")

                # Store all missing interactions for detailed analysis
                if group_i not in interaction_matrix_gaps:
                    interaction_matrix_gaps[group_i] = []
                interaction_matrix_gaps[group_i].append({
                    'partner_group': group_j,
                    'main_groups': (main_g1, main_g2),
                    'critical': main_g1 != main_g2 if main_g1 and main_g2 else True
                })

    # Generate enhanced problem summary
    problem_summary = []
    if missing_rq_groups:
        problem_summary.append(f"Missing R/Q parameters for {len(missing_rq_groups)} groups: {missing_rq_groups}")

    if missing_main_group_interactions:
        unique_main_pairs = list(set(missing_main_group_interactions))
        problem_summary.append(f"Missing {len(missing_main_group_interactions)} critical interactions between {len(unique_main_pairs)} main group pairs")
        problem_summary.append(f"Critical main group pairs: {unique_main_pairs}")

    if missing_count > len(missing_main_group_interactions):
        within_group_missing = missing_count - len(missing_main_group_interactions)
        problem_summary.append(f"Additional {within_group_missing} missing interactions within same main groups (non-critical)")

    is_compatible = len(missing_rq_groups) == 0 and len(missing_main_group_interactions) == 0

    if verbose:
        if is_compatible:
            print("✓ All critical UNIFAC parameters are available")
            if missing_count > 0:
                print(f"  Note: {missing_count} non-critical within-main-group interactions missing")
        else:
            print(f"⚠ Found {len(missing_rq_groups)} missing R/Q groups and {len(missing_main_group_interactions)} critical interactions")
            print(f"  Total interaction gaps: {missing_count}/{total_interactions}")

    return {
        'compatible': is_compatible,
        'groups': sorted(all_groups),
        'main_groups': sorted(main_groups),
        'missing_rq': missing_rq_groups,
        'missing_interactions': list(set(missing_main_group_interactions)),  # Remove duplicates
        'missing_subgroup_interactions': missing_subgroup_interactions,
        'interaction_matrix_gaps': interaction_matrix_gaps,
        'summary': problem_summary
    }