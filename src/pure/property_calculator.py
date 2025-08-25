"""
Pure Component Property Calculator - Extensible Framework

Framework for calculating pure component properties using different estimation methods.
Currently supports Constantinou-Gani method with extensible architecture for future methods.

Example Usage:
    # Using default Constantinou-Gani method
    calculator = PropertyCalculator()
    compound = {'CH3': 1, 'CH2': 1, 'OH': 1}  # Ethanol
    properties = calculator.calculate(compound)
    units = calculator.get_property_units()
    
    # Using specific method
    from .constantinou_gani import ConstantinouGani
    method = ConstantinouGani()
    calculator = PropertyCalculator(method)
    properties = calculator.calculate(compound, ['tb', 'tc'])
    
    # Display with units
    for prop, value in properties.items():
        unit = units.get(prop, '')
        print(f"{prop}: {value:.2f} {unit}")
    
    # Batch calculation
    compounds = {
        'acetone': {'CH3': 1, 'CH3CO': 1},
        'methanol': {'CH3OH': 1},
        'ethanol': {'CH3': 1, 'CH2': 1, 'OH': 1}
    }
    batch_results = calculator.calculate_batch(compounds, ['tb', 'tc'])
    
    for compound_name, props in batch_results.items():
        print(f"{compound_name}: {props}")
"""

from typing import Dict, List, Optional, Union
from pathlib import Path

from .methods.base_method import PropertyMethod
from .methods.constantinou_gani import ConstantinouGani


class PropertyCalculator:
    """
    Extensible calculator for pure component properties.
    
    Uses pluggable property estimation methods for flexibility and maintainability.
    """
    
    def __init__(self, method: Optional[PropertyMethod] = None, 
                 data_path: Optional[Union[str, Path]] = None,
                 missing_group_action: str = "remove_compounds"):
        """
        Initialize property calculator.
        
        Args:
            method: Property estimation method (defaults to Constantinou-Gani)
            data_path: Path to data directory (passed to method if provided)
            missing_group_action: Action for missing groups
        """
        if method is None:
            method = ConstantinouGani(data_path=data_path, 
                                    missing_group_action=missing_group_action)
        
        self.method = method
    
    def calculate(self, compound: Dict[str, int], 
                 properties: Optional[List[str]] = None) -> Optional[Dict[str, Optional[float]]]:
        """
        Calculate properties for a compound using the configured method.
        
        Args:
            compound: Dict with group names as keys and counts as values
            properties: List of properties to calculate (None = all available)
            
        Returns:
            Dict with property names and values, or None if compound should be removed
        """
        return self.method.calculate_properties(compound, properties)
    
    def get_available_properties(self) -> List[str]:
        """Get list of properties that can be calculated."""
        return self.method.get_available_properties()
    
    def get_property_units(self) -> Dict[str, str]:
        """Get units for each property."""
        return self.method.get_property_units()
    
    def get_method_name(self) -> str:
        """Get the name of the current estimation method."""
        return self.method.get_method_name()
    
    def print_missing_groups_summary(self):
        """Print summary of missing groups encountered during calculations."""
        self.method.print_missing_groups_summary()
    
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
        return self.method.calculate_batch(compounds, properties)
    
    def print_batch_results(self, results: Dict[str, Optional[Dict[str, Optional[float]]]], 
                           title: str = "Batch Property Calculation Results"):
        """
        Print batch calculation results in a formatted table.
        
        Args:
            results: Results from calculate_batch()
            title: Title for the output table
        """
        print(f"\n=== {title} ===")
        print(f"Method: {self.get_method_name()}")
        
        units = self.get_property_units()
        
        for compound_name, properties in results.items():
            print(f"\n{compound_name.upper()}:")
            if properties is None:
                print("  Compound removed due to missing data")
                continue
            
            for prop, value in properties.items():
                if value is not None:
                    unit = units.get(prop, '')
                    print(f"  {prop.upper()}: {value:.2f} {unit}")
                else:
                    print(f"  {prop.upper()}: Not available")


# Convenience function for quick calculations
def calculate_properties(compound: Dict[str, int], 
                        data_path: Optional[Union[str, Path]] = None,
                        method_name: str = "Constantinou-Gani") -> Optional[Dict[str, Optional[float]]]:
    """
    Convenience function to calculate all properties for a compound.
    
    Args:
        compound: Dict with group names as keys and counts as values
        data_path: Optional path to property data directory
        method_name: Name of estimation method to use
        
    Returns:
        Dict with calculated properties, or None if compound should be removed
    """
    if method_name == "Constantinou-Gani":
        method = ConstantinouGani(data_path=data_path)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    calculator = PropertyCalculator(method)
    return calculator.calculate(compound)


if __name__ == "__main__":
    # Example usage
    print("=== Property Calculator Example ===")
    
    # Test compounds
    test_compounds = {
        'ethanol': {'CH3': 1, 'CH2': 1, 'OH': 1},
        'acetone': {'CH3': 1, 'CH3CO': 1},
        'water': {'H2O': 1},
        'methanol': {'CH3OH': 1}
    }
    
    calculator = PropertyCalculator()
    print(f"Using method: {calculator.get_method_name()}")
    print(f"Available properties: {calculator.get_available_properties()}")
    
    units = calculator.get_property_units()
    
    for name, compound in test_compounds.items():
        print(f"\n{name.upper()}: {compound}")
        properties = calculator.calculate(compound)
        
        if properties is None:
            print("  Compound removed due to missing group contributions")
            continue

        for prop_name, value in properties.items():
            if value is not None:
                unit = units.get(prop_name, '')
                print(f"  {prop_name.upper()}: {value:.2f} {unit}")
            else:
                print(f"  {prop_name.upper()}: Not available")
