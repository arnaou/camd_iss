"""
Abstract base class for property estimation methods.

Defines the interface that all property estimation methods must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from pathlib import Path


class PropertyMethod(ABC):
    """Abstract base class for property estimation methods."""
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None, 
                 missing_group_action: str = "remove_compounds"):
        """
        Initialize property method.
        
        Args:
            data_path: Path to data directory
            missing_group_action: Action for missing groups ("remove_compounds" or "set_zero")
        """
        self.missing_group_action = missing_group_action
        self.missing_groups_summary = {}
        
    @abstractmethod
    def calculate_properties(self, compound: Dict[str, int], 
                           properties: Optional[List[str]] = None) -> Optional[Dict[str, Optional[float]]]:
        """
        Calculate properties for a compound.
        
        Args:
            compound: Dict with group names as keys and counts as values
            properties: List of properties to calculate (None = all available)
            
        Returns:
            Dict with property names and values, or None if compound should be removed
        """
        pass
    
    @abstractmethod
    def get_available_properties(self) -> List[str]:
        """Get list of properties that can be calculated."""
        pass
    
    @abstractmethod
    def get_property_units(self) -> Dict[str, str]:
        """Get units for each property."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of this estimation method."""
        pass
    
    def calculate_batch(self, compounds: Dict[str, Dict[str, int]], 
                       properties: Optional[List[str]] = None) -> Dict[str, Optional[Dict[str, Optional[float]]]]:
        """
        Calculate properties for multiple named compounds.
        
        Args:
            compounds: Dict of {compound_name: {group: count}} 
                      e.g., {'acetone': {'CH3': 1, 'CH3CO': 1}, 'methanol': {'CH3OH': 1}}
            properties: List of properties to calculate (None = all available)
            
        Returns:
            Dict mapping compound names to property results
        """
        results = {}
        
        for compound_name, compound_groups in compounds.items():
            results[compound_name] = self.calculate_properties(compound_groups, properties)
            
        return results
    
    def _handle_missing_groups(self, compound: Dict[str, int], missing_groups: list, 
                              property_name: str) -> bool:
        """Handle missing groups according to configured action."""
        if not missing_groups:
            return True
            
        # Track missing groups for summary
        if property_name not in self.missing_groups_summary:
            self.missing_groups_summary[property_name] = set()
        self.missing_groups_summary[property_name].update(missing_groups)
        
        if self.missing_group_action == "remove_compounds":
            return False
        elif self.missing_group_action == "set_zero":
            return True
        
        return True
    
    def print_missing_groups_summary(self):
        """Print summary of missing groups encountered."""
        if not self.missing_groups_summary:
            print("No missing group contributions found.")
            return
            
        print("\n=== Missing Group Contributions Summary ===")
        for property_name, missing_groups in self.missing_groups_summary.items():
            print(f"{property_name.upper()}: {sorted(missing_groups)}")
        print(f"Action taken: {self.missing_group_action}")
        print("=" * 45)