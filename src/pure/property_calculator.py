"""
Pure Component Property Calculator

Calculates pure component properties using estimation methods (default: Constantinou-Gani).

Example Usage:
    # Simple calculation (default method)
    calculator = PropertyCalculator()
    compound = {'CH3': 1, 'CH2': 1, 'OH': 1}  # Ethanol
    properties = calculator.calculate(compound)  # Auto-detects single compound
    
    # Batch calculation (auto-detected)
    compounds = {
        'ethanol': {'CH3': 1, 'CH2': 1, 'OH': 1},
        'acetone': {'CH3': 1, 'CH3CO': 1},
        'methanol': {'CH3OH': 1}
    }
    batch_results = calculator.calculate(compounds)  # Auto-detects batch
"""

from typing import Dict, List, Optional, Union, overload
from pathlib import Path

from .methods.base_method import PropertyMethod
from .methods.constantinou_gani import ConstantinouGani


class PropertyCalculator:
    """Calculator for pure component properties using pluggable estimation methods."""
    
    def __init__(self, method: Optional[PropertyMethod] = None, 
                 data_path: Optional[Union[str, Path]] = None,
                 missing_group_action: str = "remove_compounds"):
        """Initialize with optional method and data path."""
        if method is None:
            method = ConstantinouGani(data_path=data_path, 
                                    missing_group_action=missing_group_action)
        
        self.method = method
    
    def calculate(self, input_data: Union[Dict[str, int], Dict[str, Dict[str, int]]], 
                 properties: Optional[List[str]] = None) -> Union[Optional[Dict[str, Optional[float]]], Dict[str, Optional[Dict[str, Optional[float]]]]]:
        """Calculate properties for single compound or batch. Auto-detects input type."""
        if not input_data:
            return None if self._is_single_compound(input_data) else {}
        
        if self._is_single_compound(input_data):
            return self.method.calculate_properties(input_data, properties)
        else:
            # Inline batch processing
            results = {}
            for compound_name, compound_groups in input_data.items():
                results[compound_name] = self.method.calculate_properties(compound_groups, properties)
            return results
    
    
    def _is_single_compound(self, input_data: Union[Dict[str, int], Dict[str, Dict[str, int]]]) -> bool:
        """Determine if input is single compound or batch."""
        if not input_data:
            return True
        return isinstance(next(iter(input_data.values())), int)
