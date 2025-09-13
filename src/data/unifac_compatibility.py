"""
UNIFAC parameter compatibility checking with main group logic.

Minimal implementation focusing on essential compatibility information:
- Missing R/Q parameters
- Missing interactions between different main groups
- Clean, focused output
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from .load_unifac import load_unifac_data


def check_unifac_parameters(
    mixture: Optional[Dict[str, Dict[str, int]]] = None,
    solvent_candidates: Optional[List[Dict[str, int]]] = None,
    data_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Check UNIFAC parameter availability focusing on essential missing parameters.

    Args:
        mixture: Base mixture {'compound_name': {'group': count}}
        solvent_candidates: Optional solvents [{'group': count}, ...]
        data_path: Optional path to data directory

    Returns:
        dict with keys:
        - 'compatible': bool
        - 'missing_rq': list of groups missing R/Q parameters
        - 'missing_interactions_groups': list of (group1, group2) tuples
        - 'missing_interactions_main': list of (main_group1, main_group2) tuples
        - 'compatible_solvents': list of compatible solvent indices (if applicable)
        - 'solvent_issues': dict mapping solvent index to missing parameters (if applicable)
    """
    if mixture is None:
        return {'compatible': False, 'error': 'No mixture provided'}

    # Check mixture
    check = _check_groups(mixture, data_path)

    results = {
        'compatible': check['compatible'],
        'missing_rq': check['missing_rq'],
        'missing_interactions_groups': check['missing_interactions_groups'],
        'missing_interactions_main': check['missing_interactions_main']
    }

    # If mixture incompatible, return early
    if not check['compatible']:
        if solvent_candidates:
            results['compatible_solvents'] = []
            results['solvent_issues'] = {i: 'Mixture incompatible' for i in range(len(solvent_candidates))}
        return results

    # Check solvent candidates if provided
    if solvent_candidates:
        compatible_solvents = []
        solvent_issues = {}

        for i, solvent in enumerate(solvent_candidates):
            extended_system = dict(mixture)
            extended_system[f'candidate_{i}'] = solvent

            solvent_check = _check_groups(extended_system, data_path)

            if solvent_check['compatible']:
                compatible_solvents.append(i)
            else:
                issues = {}
                if solvent_check['missing_rq']:
                    issues['missing_rq'] = solvent_check['missing_rq']
                if solvent_check['missing_interactions_groups']:
                    issues['missing_interactions_groups'] = solvent_check['missing_interactions_groups']
                if solvent_check['missing_interactions_main']:
                    issues['missing_interactions_main'] = solvent_check['missing_interactions_main']
                solvent_issues[i] = issues

        results['compatible_solvents'] = compatible_solvents
        results['solvent_issues'] = solvent_issues

    return results


def _check_groups(
    compounds: Union[Dict[str, Dict[str, int]], List[Dict[str, int]]],
    data_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Core compatibility checking - no verbose output."""

    # Normalize input
    if isinstance(compounds, dict):
        compound_list = list(compounds.values())
    else:
        compound_list = compounds

    if not compound_list:
        return {'compatible': False, 'missing_rq': [], 'missing_interactions_groups': [], 'missing_interactions_main': []}

    # Collect all groups
    all_groups = set()
    for compound in compound_list:
        all_groups.update(compound.keys())

    # Load data
    try:
        rk_data, qk_data, interaction_data = load_unifac_data(data_path)

        # Load main group mapping
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data"

        with open(Path(data_path) / "unifac_groups.pkl", 'rb') as f:
            unifac_dict = pickle.load(f)

        group_to_main = {}
        for entry in unifac_dict.values():
            if 'grp' in entry and 'main_g' in entry:
                group_to_main[entry['grp']] = entry['main_g']

    except Exception:
        return {
            'compatible': False,
            'missing_rq': list(all_groups),
            'missing_interactions_groups': [],
            'missing_interactions_main': []
        }

    # Check R/Q parameters
    missing_rq = []
    for group in all_groups:
        if (group not in rk_data or rk_data[group] is None or
            group not in qk_data or qk_data[group] is None):
            missing_rq.append(group)

    # Check interactions between different main groups only
    missing_interactions_groups = []
    missing_interactions_main = []
    groups_list = list(all_groups)

    for i, group_i in enumerate(groups_list):
        for j in range(i + 1, len(groups_list)):
            group_j = groups_list[j]

            # Check if interaction exists
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

            # Only flag if missing AND between different main groups
            if not interaction_found:
                main_g1 = group_to_main.get(group_i)
                main_g2 = group_to_main.get(group_j)

                if main_g1 is not None and main_g2 is not None and main_g1 != main_g2:
                    missing_interactions_groups.append((group_i, group_j))
                    missing_interactions_main.append((main_g1, main_g2))

    # Remove duplicate main group pairs
    missing_interactions_main = list(set(missing_interactions_main))

    is_compatible = len(missing_rq) == 0 and len(missing_interactions_main) == 0

    return {
        'compatible': is_compatible,
        'missing_rq': missing_rq,
        'missing_interactions_groups': missing_interactions_groups,
        'missing_interactions_main': missing_interactions_main
    }