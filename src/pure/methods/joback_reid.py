"""
Joback and Reid Group Contribution Method

Placeholder for future implementation of Joback and Reid method.
Reference: Joback, K.G. and Reid, R.C. (1987) Estimation of Pure-Component Properties from Group-Contributions
"""

from typing import Dict, List, Optional, Union
from pathlib import Path

from .base_method import PropertyMethod


class JobackReid(PropertyMethod):
    """Joback and Reid method for pure component property estimation (placeholder)."""
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None, 
                 missing_group_action: str = "remove_compounds"):
        super().__init__(data_path, missing_group_action)
        # TODO: Load Joback-Reid specific data
        
    def get_method_name(self) -> str:
        return "Joback-Reid"
    
    def get_available_properties(self) -> List[str]:
        """Get list of properties that can be calculated."""
        # TODO: Implement when data is available
        return []
    
    def get_property_units(self) -> Dict[str, str]:
        """Get units for each property."""
        # TODO: Define units when properties are implemented
        return {}
    
    def calculate_properties(self, compound: Dict[str, int], 
                           properties: Optional[List[str]] = None) -> Optional[Dict[str, Optional[float]]]:
        """Calculate properties for a compound."""
        # TODO: Implement Joback-Reid calculations
        raise NotImplementedError("Joback-Reid method not yet implemented")