"""
Ultra-Minimal Structural Validator

Validates compound structures using Odele and Macchietto structural feasibility rules.
https://doi.org/10.1016/0378-3812(93)87127-M

Example Usage:
    # Simple validation (default: acyclic molecules)
    validator = StructuralValidator()
    compound = {'CH3': 1, 'CH2': 1, 'OH': 1}  # Ethanol
    is_valid = validator.is_valid(compound)
    
    # Validate cyclic molecules
    cyclic_validator = StructuralValidator(molecular_type=0)  # monocyclic
    
    # Batch validation
    compounds = [{'CH3': 2, 'OH': 1}, {'CH3': 3, 'CH2': 2}]
    results = validator.validate_batch(compounds)
"""

import json
from pathlib import Path
from typing import Dict, List, Union, Optional


class StructuralValidator:
    """Ultra-minimal validator using Odele and Macchietto rules."""
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None, molecular_type: int = 1):
        """
        Initialize structural validator.
        
        Args:
            data_path: Path to directory containing noc.json file
            molecular_type: Molecular structure type:
                           1 = acyclic (linear/branched, no rings) - default
                           0 = monocyclic (one ring)
                          -1 = bicyclic (two rings)
        """
        if molecular_type not in [-1, 0, 1]:
            raise ValueError("molecular_type must be -1 (bicyclic), 0 (monocyclic), or 1 (acyclic)")
        
        self.molecular_type = molecular_type
        self.noc_data = {}
        
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "json"
        else:
            data_path = Path(data_path)
            
        self._load_noc_data(data_path)
    
    def _load_noc_data(self, data_path: Path):
        """Load number of connections data."""
        noc_file = data_path / "noc.json"
        
        if noc_file.exists():
            try:
                with open(noc_file, 'r') as f:
                    noc_json = json.load(f)
                self.noc_data = noc_json.get("data", {})
            except Exception:
                self.noc_data = {}
    
    def is_valid(self, compound: Dict[str, int]) -> bool:
        """Check if compound is structurally valid."""
        if not compound:
            return False
        if not self.noc_data:
            return False  # Cannot validate without NOC data
        
        try:
            # Check standalone molecules first
            standalone_groups = [(g, c) for g, c in compound.items() 
                               if g in self.noc_data and self.noc_data[g] == 0]
            
            if standalone_groups:
                return self._validate_standalone(compound, standalone_groups)
            
            # Check structural rules for connected molecules
            return self._validate_connected(compound)
            
        except Exception:
            return False
    
    def _validate_standalone(self, compound: Dict[str, int], standalone_groups: List) -> bool:
        """Validate standalone molecules (NOC = 0)."""
        if len(standalone_groups) > 1 or len(compound) > 1:
            return False
        
        _, count = standalone_groups[0]
        return count == 1
    
    def _validate_connected(self, compound: Dict[str, int]) -> bool:
        """Validate connected molecules using octet and connectivity rules."""
        # Check for missing noc data
        missing_groups = [g for g in compound.keys() if g not in self.noc_data]
        if missing_groups:
            return False
        
        # Octet rule: 2*m = sum((2 - noc_i) * x_i)
        total_valency = sum((2 - self.noc_data[group]) * count 
                           for group, count in compound.items())
        
        if total_valency != 2 * self.molecular_type:
            return False
        
        # Connectivity rule: sum(x_j where j != i) >= (noc_i - 2) * x_i + 2
        for group_i, count_i in compound.items():
            noc_i = self.noc_data[group_i]
            sum_others = sum(count for group, count in compound.items() if group != group_i)
            min_required = (noc_i - 2) * count_i + 2
            
            if sum_others < min_required:
                return False
        
        return True
    
    def validate_batch(self, compounds: List[Dict[str, int]]) -> Dict[str, List[Dict[str, int]]]:
        """Validate a batch of compounds."""
        valid = []
        invalid = []
        
        for compound in compounds:
            if self.is_valid(compound):
                valid.append(compound)
            else:
                invalid.append(compound)
        
        return {'valid': valid, 'invalid': invalid}