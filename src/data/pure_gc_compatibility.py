"""
Pure component group contribution compatibility checker.

Checks if all functional groups have valid group contributions for pure component properties.
A contribution of 0.0 means the contribution is not fitted/absent.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def check_pure_gc_parameters(
    mixture: Optional[Dict[str, Dict[str, int]]] = None,
    solvent_candidates: Optional[List[Dict[str, int]]] = None,
    properties: Optional[List[str]] = None,
    data_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Check if all groups have valid group contributions for pure component properties.

    Args:
        mixture: Mixture composition as {compound: {group: count}}
        solvent_candidates: List of solvent compositions as [{group: count}, ...]
        properties: List of properties to check ['tc', 'pc', 'tb', 'ac', 'tm', 'hv']
        data_path: Path to group contribution data

    Returns:
        Dictionary with compatibility results and missing contributions
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data" / "json"

    if properties is None:
        properties = ['tc', 'pc', 'tb', 'ac']

    # Load group contribution data
    gc_data = {}
    for prop in properties:
        file_path = data_path / f"{prop.capitalize()}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                data = json.load(f)
                gc_data[prop] = data.get('data', {})

    # Collect all unique groups
    all_groups = set()
    if mixture:
        for compound_groups in mixture.values():
            all_groups.update(compound_groups.keys())
    if solvent_candidates:
        for candidate in solvent_candidates:
            all_groups.update(candidate.keys())

    # Check missing/zero contributions
    missing_contributions = {}
    for prop in properties:
        if prop not in gc_data:
            missing_contributions[prop] = list(all_groups)
        else:
            prop_data = gc_data[prop]
            missing = []
            for group in all_groups:
                if group not in prop_data or prop_data[group] == 0.0:
                    missing.append(group)
            if missing:
                missing_contributions[prop] = missing

    # Check individual solvents
    compatible_solvents = []
    solvent_issues = {}

    if solvent_candidates:
        for i, candidate in enumerate(solvent_candidates):
            candidate_issues = {}
            is_compatible = True

            for prop in properties:
                if prop not in gc_data:
                    candidate_issues[prop] = list(candidate.keys())
                    is_compatible = False
                else:
                    prop_data = gc_data[prop]
                    missing = []
                    for group in candidate.keys():
                        if group not in prop_data or prop_data[group] == 0.0:
                            missing.append(group)
                    if missing:
                        candidate_issues[prop] = missing
                        is_compatible = False

            if is_compatible:
                compatible_solvents.append(i)
            else:
                solvent_issues[i] = candidate_issues

    return {
        'compatible': len(missing_contributions) == 0,
        'missing_contributions': missing_contributions,
        'compatible_solvents': compatible_solvents,
        'solvent_issues': solvent_issues
    }