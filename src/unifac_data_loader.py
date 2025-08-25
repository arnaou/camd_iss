"""
UNIFAC data loading utilities.

Handles loading of UNIFAC parameters (Rk, Qk, interaction parameters) from JSON files.
"""
import json
from pathlib import Path


def load_unifac_data(data_path=None):
    """
    Load UNIFAC parameters from JSON files.
    
    Args:
        data_path: Path to directory containing JSON files (optional)
                  Defaults to ../data/json relative to this file
    
    Returns:
        tuple: (rk_data, qk_data, interaction_data)
               - rk_data: Dict of group van der Waals volumes
               - qk_data: Dict of group surface areas  
               - interaction_data: Dict of binary interaction parameters
    
    Raises:
        FileNotFoundError: If JSON files are not found
        json.JSONDecodeError: If JSON files are malformed
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "json"
    else:
        data_path = Path(data_path)
    
    # Load R/Q parameters
    try:
        with open(data_path / 'rk.json', 'r') as f:
            rk_data = json.load(f).get("data", {})
        with open(data_path / 'qk.json', 'r') as f:
            qk_data = json.load(f).get("data", {})
        # Load interaction parameters
        with open(data_path / 'a.json', 'r') as f:
            interaction_data = json.load(f).get("data", {})
    except FileNotFoundError as e:
        raise FileNotFoundError(f"UNIFAC data file not found: {e}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in UNIFAC data file: {e}")
    
    return rk_data, qk_data, interaction_data


def get_available_groups(data_path=None):
    """
    Get list of available UNIFAC groups.
    
    Args:
        data_path: Path to directory containing JSON files (optional)
    
    Returns:
        list: Available group names
    """
    rk_data, _, _ = load_unifac_data(data_path)
    return list(rk_data.keys())


def validate_mixture_groups(mixture, data_path=None):
    """
    Validate that all groups in mixture are available in UNIFAC data.
    
    Args:
        mixture: Dict {group: [counts for each component]}
        data_path: Path to directory containing JSON files (optional)
    
    Returns:
        tuple: (is_valid, missing_groups)
               - is_valid: bool indicating if all groups are available
               - missing_groups: list of groups not found in data
    """
    available_groups = set(get_available_groups(data_path))
    mixture_groups = set(mixture.keys())
    missing_groups = list(mixture_groups - available_groups)
    
    return len(missing_groups) == 0, missing_groups