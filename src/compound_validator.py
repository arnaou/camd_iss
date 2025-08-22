"""
Compound Validator routine

Validates compound structures using Odele and Macchietto structural feasibility rules.
https://doi.org/10.1016/0378-3812(93)87127-M
This module can be used standalone for validating individual compounds or integrated
into the CAMD workflow.

Example Usage:
    # Standalone usage
    validator = CompoundValidator(data_path="./data/json")
    compound = {'CH3': 2, 'CH2': 1, 'OH': 1}  # Ethanol
    is_valid = validator.validate_compound(compound)
    print(f"Compound is valid: {is_valid}")
    
    # Batch validation
    compounds = [{'CH3': 2, 'OH': 1}, {'CH3': 3, 'CH2': 2}]
    valid_compounds = validator.filter_valid_compounds(compounds)
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Tuple


class CompoundValidator:
    """
    Validator for compound structures using Odele and Macchietto rules.
    
    Implements:
    1. Octet rule (valency satisfaction): 2*m = sum((2 - noc_i) * x_i)
    2. Connectivity rule: sum(x_j where j != i) >= (noc_i - 2) * x_i + 2
    
    Where:
    - m: molecular type (1=acyclic, 0=monocyclic, -1=bicyclic)
    - noc_i: number of connections for group i (valence)
    - x_i: occurrence count of group i
    """
    
    def __init__(self, data_path: Union[str, Path] = None, molecular_type: int = 1):
        """
        Initialize the compound validator.
        
        Args:
            data_path: Path to directory containing noc.json file
            molecular_type: 1=acyclic, 0=monocyclic, -1=bicyclic (default: 1)
        """
        # Validate molecular_type
        if molecular_type not in [-1, 0, 1]:
            raise ValueError("molecular_type must be -1 (bicyclic), 0 (monocyclic), or 1 (acyclic)")
        
        self.molecular_type = molecular_type
        self.noc_data = {}
        
        if data_path is None:
            # Default to data directory relative to this file
            data_path = Path(__file__).parent.parent / "data" / "json"
        else:
            data_path = Path(data_path)
            
        self.data_path = data_path
        self._load_noc_data()
    
    def _load_noc_data(self):
        """Load number of connections (noc) data from JSON file."""
        noc_file = self.data_path / "noc.json"
        
        if noc_file.exists():
            try:
                with open(noc_file, 'r') as f:
                    noc_json = json.load(f)
                self.noc_data = noc_json.get("data", {})
                print(f"Loaded noc data: {len(self.noc_data)} groups")
            except Exception as e:
                print(f"Error loading noc.json: {e}")
                self.noc_data = {}
        else:
            print(f"Warning: noc.json not found at {noc_file}")
            self.noc_data = {}
    
    def validate_compound(self, compound: Dict[str, int], verbose: bool = False) -> bool:
        """
        Validate a single compound using structural feasibility rules.
        
        Args:
            compound: Dict with group names as keys and counts as values
            verbose: Print detailed validation messages
            
        Returns:
            True if compound is structurally valid, False otherwise
        """
        if not compound:
            if verbose:
                print("Empty compound")
            return False
        
        if not self.noc_data:
            if verbose:
                print("Warning: No noc data available, skipping validation")
            return True  # Assume valid if no data to check against
        
        try:
            # Check for standalone molecules (NOC = 0)
            if not self._check_standalone_molecules(compound, verbose):
                return False
            
            # Check if this is a pure standalone molecule
            if self._is_pure_standalone_molecule(compound):
                if verbose:
                    print(f"Compound {compound} is a valid standalone molecule")
                return True
            
            # For non-standalone molecules, check octet and connectivity rules
            # Check octet rule
            if not self._check_octet_rule(compound, verbose):
                return False
            
            # Check connectivity rule
            if not self._check_connectivity_rule(compound, verbose):
                return False
            
            if verbose:
                print(f"Compound {compound} is structurally valid")
            return True
            
        except Exception as e:
            if verbose:
                print(f"Validation error for {compound}: {e}")
            return False
    
    def _check_standalone_molecules(self, compound: Dict[str, int], verbose: bool = False) -> bool:
        """
        Check for standalone molecules (groups with NOC = 0).
        
        Rules for standalone molecules:
        1. A single standalone molecule with count=1 is valid (e.g., {'H2O': 1})
        2. Multiple standalone molecules are invalid (e.g., {'H2O': 2})
        3. Standalone molecules mixed with other groups are invalid (e.g., {'H2O': 1, 'CH3': 1})
        
        Args:
            compound: Compound to validate
            verbose: Print detailed messages
            
        Returns:
            True if standalone molecule rules are satisfied, False otherwise
        """
        standalone_groups = []
        other_groups = []
        
        for group, count in compound.items():
            if group in self.noc_data and self.noc_data[group] == 0:
                standalone_groups.append((group, count))
            else:
                other_groups.append((group, count))
        
        # If no standalone molecules, validation passes
        if not standalone_groups:
            return True
        
        # If there are standalone molecules, check the rules
        if len(standalone_groups) > 1:
            # Multiple different standalone molecules
            if verbose:
                group_names = [group for group, _ in standalone_groups]
                print(f"Invalid compound: Multiple standalone molecules: {group_names}")
                print("Only one type of standalone molecule is allowed per compound")
            return False
        
        if len(standalone_groups) == 1:
            group, count = standalone_groups[0]
            
            # Check if count > 1
            if count > 1:
                if verbose:
                    print(f"Invalid compound: Standalone molecule '{group}' has count > 1 ({count})")
                    print("Standalone molecules must have count = 1")
                return False
            
            # Check if mixed with other groups
            if other_groups:
                if verbose:
                    other_group_names = [group for group, _ in other_groups]
                    print(f"Invalid compound: Standalone molecule '{group}' mixed with other groups: {other_group_names}")
                    print("Standalone molecules cannot be part of larger structures")
                return False
        
        return True
    
    def _is_pure_standalone_molecule(self, compound: Dict[str, int]) -> bool:
        """
        Check if compound is a pure standalone molecule (single group with NOC=0 and count=1).
        
        Args:
            compound: Compound to check
            
        Returns:
            True if it's a pure standalone molecule
        """
        if len(compound) != 1:
            return False
        
        group, count = next(iter(compound.items()))
        return (group in self.noc_data and 
                self.noc_data[group] == 0 and 
                count == 1)
    
    def _check_octet_rule(self, compound: Dict[str, int], verbose: bool = False) -> bool:
        """
        Check octet rule: 2*m = sum((2 - noc_i) * x_i)
        
        Args:
            compound: Compound to validate
            verbose: Print detailed messages
            
        Returns:
            True if octet rule is satisfied
        """
        total_valency = 0
        missing_groups = []
        
        for group, count in compound.items():
            if group in self.noc_data:
                noc = self.noc_data[group]
                total_valency += (2 - noc) * count
            else:
                missing_groups.append(group)
        
        if missing_groups:
            if verbose:
                print(f"Warning: Missing noc data for groups: {missing_groups}")
            return False  # Cannot validate without noc data
        
        expected_valency = 2 * self.molecular_type
        
        if total_valency != expected_valency:
            if verbose:
                molecular_type_names = {-1: 'bicyclic', 0: 'monocyclic', 1: 'acyclic'}
                type_name = molecular_type_names[self.molecular_type]
                print(f"Octet rule violation: valency sum = {total_valency}, "
                      f"expected = {expected_valency} (for {type_name} molecule)")
            return False
        
        if verbose:
            print(f"Octet rule satisfied: valency sum = {total_valency}")
        return True
    
    def _check_connectivity_rule(self, compound: Dict[str, int], verbose: bool = False) -> bool:
        """
        Check connectivity rule: sum(x_j where j != i) >= (noc_i - 2) * x_i + 2
        
        Args:
            compound: Compound to validate
            verbose: Print detailed messages
            
        Returns:
            True if connectivity rule is satisfied for all groups
        """
        for group_i, count_i in compound.items():
            if group_i not in self.noc_data:
                continue  # Skip groups without noc data
                
            noc_i = self.noc_data[group_i]
            
            # Sum of all other groups
            sum_other_groups = sum(count for group, count in compound.items() if group != group_i)
            
            # Required minimum connections
            min_required = (noc_i - 2) * count_i + 2
            
            if sum_other_groups < min_required:
                if verbose:
                    print(f"Connectivity rule violation for group '{group_i}': "
                          f"other groups sum = {sum_other_groups}, minimum required = {min_required}")
                return False
        
        if verbose:
            print("Connectivity rule satisfied for all groups")
        return True
    
    def filter_valid_compounds(self, compounds: List[Dict[str, int]], 
                             verbose: bool = False) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
        """
        Filter a list of compounds, returning valid and invalid ones separately.
        
        Args:
            compounds: List of compounds to validate
            verbose: Print validation details
            
        Returns:
            Tuple of (valid_compounds, invalid_compounds)
        """
        valid_compounds = []
        invalid_compounds = []
        
        for i, compound in enumerate(compounds):
            if verbose:
                print(f"\nValidating compound {i}: {compound}")
            
            if self.validate_compound(compound, verbose=verbose):
                valid_compounds.append(compound)
            else:
                invalid_compounds.append(compound)
        
        if verbose:
            print(f"\nValidation summary: {len(valid_compounds)} valid, {len(invalid_compounds)} invalid")
        
        return valid_compounds, invalid_compounds
    
    def get_validation_stats(self, compounds: List[Dict[str, int]]) -> Dict[str, int]:
        """
        Get validation statistics for a list of compounds.
        
        Args:
            compounds: List of compounds to analyze
            
        Returns:
            Dict with validation statistics
        """
        stats = {
            'total': len(compounds),
            'valid': 0,
            'invalid': 0,
            'standalone_violations': 0,
            'octet_violations': 0,
            'connectivity_violations': 0,
            'missing_data': 0
        }
        
        for compound in compounds:
            if self.validate_compound(compound):
                stats['valid'] += 1
            else:
                stats['invalid'] += 1
                
                # Check specific violation types
                try:
                    if not self._check_standalone_molecules(compound):
                        stats['standalone_violations'] += 1
                    elif not self._check_octet_rule(compound):
                        stats['octet_violations'] += 1
                    elif not self._check_connectivity_rule(compound):
                        stats['connectivity_violations'] += 1
                except:
                    stats['missing_data'] += 1
        
        return stats
    
    def set_molecular_type(self, molecular_type: int):
        """
        Set the molecular type for validation.
        
        Args:
            molecular_type: 1=acyclic, 0=monocyclic, -1=bicyclic
        """
        if molecular_type not in [-1, 0, 1]:
            raise ValueError("molecular_type must be -1 (bicyclic), 0 (monocyclic), or 1 (acyclic)")
        
        self.molecular_type = molecular_type
        molecular_type_names = {-1: 'bicyclic', 0: 'monocyclic', 1: 'acyclic'}
        print(f"Set molecular type to {molecular_type} ({molecular_type_names[molecular_type]})")


# Convenience functions
def validate_compound(compound: Dict[str, int], data_path: Union[str, Path] = None, 
                     molecular_type: int = 1, verbose: bool = False) -> bool:
    """
    Convenience function to validate a single compound.
    
    Args:
        compound: Compound to validate
        data_path: Optional path to data directory
        molecular_type: 1=acyclic, 0=monocyclic, -1=bicyclic
        verbose: Print validation details
        
    Returns:
        True if compound is valid
    """
    validator = CompoundValidator(data_path=data_path, molecular_type=molecular_type)
    return validator.validate_compound(compound, verbose=verbose)


def filter_valid_compounds(compounds: List[Dict[str, int]], data_path: Union[str, Path] = None,
                          molecular_type: int = 1, verbose: bool = False) -> List[Dict[str, int]]:
    """
    Convenience function to filter valid compounds from a list.
    
    Args:
        compounds: List of compounds to validate
        data_path: Optional path to data directory  
        molecular_type: 1=acyclic, 0=monocyclic, -1=bicyclic
        verbose: Print validation details
        
    Returns:
        List of valid compounds
    """
    validator = CompoundValidator(data_path=data_path, molecular_type=molecular_type)
    valid_compounds, _ = validator.filter_valid_compounds(compounds, verbose=verbose)
    return valid_compounds


if __name__ == "__main__":
    # Example usage
    print("=== Compound Validator Example ===")
    
    # Test compounds
    test_compounds = [
        {'H2O': 1},                                # water - valid standalone molecule
        {'H2O': 2},                                # invalid - multiple standalone molecules
        {'CH3': 1, 'CH2': 1, 'OH': 1},             # Ethanol - should be valid
        {'CH3': 1, 'CH3CO': 1},                    # Acetone - should be valid
        {'CH3': 10},                               # Too many CH3 - not valid
        {'OH': 5},                                 # Too many OH - not valid
        {'CH3': 1, 'CH2': 2, 'OH': 1},             # Propanol - should be valid
        {'CH3': 1, 'CH2': 3, 'H2O': 1, 'CHO': 1},  # invalid - standalone molecule mixed with others
        {'H2O': 1, 'DOH': 1},                      # invalid - multiple different standalone molecules
    ]
    
    validator = CompoundValidator()
    
    print(f"Testing {len(test_compounds)} compounds with molecular type = 1 (acyclic)")
    
    valid_compounds, invalid_compounds = validator.filter_valid_compounds(test_compounds, verbose=True)
    
    print(f"\n=== Results ===")
    print(f"Valid compounds ({len(valid_compounds)}):")
    for compound in valid_compounds:
        print(f"  {compound}")
    
    print(f"Invalid compounds ({len(invalid_compounds)}):")  
    for compound in invalid_compounds:
        print(f"  {compound}")
    
    # Validation statistics
    stats = validator.get_validation_stats(test_compounds)
    print(f"\n=== Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")