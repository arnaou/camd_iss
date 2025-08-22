"""
Pure Component Property Calculator

Calculates pure component properties using the Constantinou-Gani method.
original paper: https://doi.org/10.1002/aic.690401011
extension: https://doi.org/10.1016/0378-3812(94)02593-P 

This module can be used standalone for calculating properties of individual compounds
or integrated into the CAMD workflow.

Example Usage:
    # Standalone usage
    calculator = PropertyCalculator(data_path="./data/json")
    compound = {'CH3': 2, 'OH': 1}  # Ethanol
    properties = calculator.calculate_all_properties(compound)
    print(f"Boiling point: {properties['tb']:.2f} K")
    
    # Individual property calculation
    tb = calculator.calculate_tb(compound)
    print(f"Tb: {tb:.2f} K")
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional, Union


class PropertyCalculator:
    """
    Calculator for pure component properties using Constantinou-Gani method.
    
    Calculates (so far):
    - Tm: Melting point [K]
    - Tb: Boiling point [K] 
    - Tc: Critical temperature [K]
    - Pc: Critical pressure [bar]
    - Ac: Acentric factor [-]
    - Hv: Heat of vaporization [kJ/mol]
    """
    
    def __init__(self, data_path: Union[str, Path] = None, pure_method: str = "Constantinou-Gani", 
                 missing_group_action: str = "remove_compounds"):
        """
        Initialize the property calculator.
        
        Args:
            data_path: Path to directory containing property data JSON files
            pure_method: Method for pure component properties (currently only "Constantinou-Gani")
            missing_group_action: Action for missing groups: "remove_compounds" (default) or "set_zero"
        """
        self.pure_method = pure_method
        self.pure_properties = {}
        self.missing_group_action = missing_group_action
        self.missing_groups_summary = {}  # Track missing groups across all calculations
        
        if missing_group_action not in ["remove_compounds", "set_zero"]:
            raise ValueError(f"missing_group_action must be 'remove_compounds' or 'set_zero', got '{missing_group_action}'")
        
        if data_path is None:
            # Default to data directory relative to this file
            data_path = Path(__file__).parent.parent / "data" / "json"
        else:
            data_path = Path(data_path)
            
        self.data_path = data_path
        self._load_property_data()
    
    def _load_property_data(self):
        """Load pure component property data from JSON files."""
        if self.pure_method != "Constantinou-Gani":
            raise NotImplementedError(f"Pure method {self.pure_method} not implemented")
        
        print(f"Loading property data from: {self.data_path}")
        
        # Property files to load
        property_files = {
            'scalars': 'scalars.json',
            'tm': 'Tm.json',
            'tb': 'Tb.json', 
            'tc': 'Tc.json',
            'pc': 'Pc.json',
            'ac': 'Ac.json',
            'hv': 'hv.json'
        }
        
        for prop_name, filename in property_files.items():
            prop_file = self.data_path / filename
            if prop_file.exists():
                try:
                    with open(prop_file, 'r') as f:
                        prop_data = json.load(f)
                    self.pure_properties[prop_name] = prop_data.get("data", {})
                    print(f"  Loaded {prop_name}: {len(self.pure_properties[prop_name])} entries")
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
                    self.pure_properties[prop_name] = {}
            else:
                print(f"  Warning: {filename} not found")
                self.pure_properties[prop_name] = {}
    
    def _handle_missing_groups(self, compound: Dict[str, int], missing_groups: list, property_name: str) -> bool:
        """
        Handle missing groups according to the configured action.
        
        Args:
            compound: The compound being calculated
            missing_groups: List of groups with missing contributions
            property_name: Name of the property being calculated
            
        Returns:
            bool: True if calculation should continue, False if compound should be removed
        """
        if not missing_groups:
            return True
            
        # Track missing groups for summary
        if property_name not in self.missing_groups_summary:
            self.missing_groups_summary[property_name] = set()
        self.missing_groups_summary[property_name].update(missing_groups)
        
        if self.missing_group_action == "remove_compounds":
            print(f"Warning: Compound {compound} has missing {property_name.upper()} data for groups: {missing_groups}")
            print(f"         Action: Removing compound (missing_group_action='remove_compounds')")
            return False
        elif self.missing_group_action == "set_zero":
            print(f"Warning: Compound {compound} has missing {property_name.upper()} data for groups: {missing_groups}")
            print(f"         Action: Setting missing contributions to 0 (missing_group_action='set_zero')")
            return True
        
        return True
    
    def print_missing_groups_summary(self):
        """Print a summary of all missing groups encountered during calculations."""
        if not self.missing_groups_summary:
            print("No missing group contributions found.")
            return
            
        print("\n=== Missing Group Contributions Summary ===")
        for property_name, missing_groups in self.missing_groups_summary.items():
            print(f"{property_name.upper()}: {sorted(missing_groups)}")
        print(f"Action taken: {self.missing_group_action}")
        print("=" * 45)
    
    def calculate_all_properties(self, compound: Dict[str, int]) -> Optional[Dict[str, Optional[float]]]:
        """
        Calculate all available pure component properties for a compound.
        
        Args:
            compound: Dict with group names as keys and counts as values
                     Example: {'CH3': 2, 'OH': 1} for ethanol
            
        Returns:
            Dict with property names as keys and calculated values as values, or None if compound should be removed
        """
        properties = {}
        
        # Calculate each property and check for missing groups
        tm_result = self.calculate_tm(compound)
        if tm_result is None and self.missing_group_action == "remove_compounds":
            return None
        properties['tm'] = tm_result
        
        tb_result = self.calculate_tb(compound)
        if tb_result is None and self.missing_group_action == "remove_compounds":
            return None
        properties['tb'] = tb_result
        
        tc_result = self.calculate_tc(compound)
        if tc_result is None and self.missing_group_action == "remove_compounds":
            return None
        properties['tc'] = tc_result
        
        pc_result = self.calculate_pc(compound)
        if pc_result is None and self.missing_group_action == "remove_compounds":
            return None
        properties['pc'] = pc_result
        
        ac_result = self.calculate_ac(compound)
        if ac_result is None and self.missing_group_action == "remove_compounds":
            return None
        properties['ac'] = ac_result
        
        hv_result = self.calculate_hv(compound)
        if hv_result is None and self.missing_group_action == "remove_compounds":
            return None
        properties['hv'] = hv_result
        
        return properties
    
    def calculate_tm(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate melting point: Tm0 * ln(sum(contrib.Tm[i] * mol[i]))
        
        Returns:
            Melting point in Kelvin [K] or None if calculation fails
        """
        if 'tm' not in self.pure_properties or 'scalars' not in self.pure_properties:
            return None
            
        tm_data = self.pure_properties['tm']
        scalars = self.pure_properties['scalars']
        
        tm_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in tm_data:
                tm_sum += count * tm_data[group]
            else:
                missing_groups.append(group)
        
        # Handle missing groups according to configured action
        if not self._handle_missing_groups(compound, missing_groups, 'tm'):
            return None
        
        # if tm_sum <= 0:
        #     return None
            
        # Get scalar for Tm calculation
        tm0 = scalars.get('Tm0')
        return tm0 * np.log(tm_sum)
    
    def calculate_tb(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate boiling point: Tb0 * ln(sum(contrib.Tb[i] * mol[i]))
        
        Returns:
            Boiling point in Kelvin [K] or None if calculation fails
        """
        if 'tb' not in self.pure_properties or 'scalars' not in self.pure_properties:
            return None
            
        tb_data = self.pure_properties['tb']
        scalars = self.pure_properties['scalars']
        
        tb_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in tb_data:
                tb_sum += count * tb_data[group]
            else:
                missing_groups.append(group)
        
        # Handle missing groups according to configured action
        if not self._handle_missing_groups(compound, missing_groups, 'tb'):
            return None
        
        # if tb_sum <= 0:
        #     return None
            
        # Get scalar for Tb calculation (should be 204.359)
        tb0 = scalars.get('Tb0')
        return tb0 * np.log(tb_sum)
    
    def calculate_tc(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate critical temperature: Tc0 * ln(sum(contrib.Tc[i] * mol[i]))
        
        Returns:
            Critical temperature in Kelvin [K] or None if calculation fails
        """
        if 'tc' not in self.pure_properties or 'scalars' not in self.pure_properties:
            return None
            
        tc_data = self.pure_properties['tc']
        scalars = self.pure_properties['scalars']
        
        tc_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in tc_data:
                tc_sum += count * tc_data[group]
            else:
                missing_groups.append(group)
        
        # Handle missing groups according to configured action
        if not self._handle_missing_groups(compound, missing_groups, 'tc'):
            return None
        
        # if tc_sum <= 0:
        #     return None
            
        # Get scalar for Tc calculation (should be 181.128)
        tc0 = scalars.get('Tc0')
        return tc0 * np.log(tc_sum)
    
    def calculate_pc(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate critical pressure: 1 / (sum(contrib.Pc[i] * mol[i]) + pc02)^2 + pc01
        
        Returns:
            Critical pressure in bar [bar] or None if calculation fails
        """
        if 'pc' not in self.pure_properties or 'scalars' not in self.pure_properties:
            return None
            
        pc_data = self.pure_properties['pc']
        scalars = self.pure_properties['scalars']
        
        pc_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in pc_data:
                pc_sum += count * pc_data[group]
            else:
                missing_groups.append(group)
        
        # Handle missing groups according to configured action
        if not self._handle_missing_groups(compound, missing_groups, 'pc'):
            return None
        
        # Get scalars for Pc calculation
        pc01 = scalars.get('Pc01')
        pc02 = scalars.get('Pc02')
        
        # Pc = 1 / (sum(contrib.Pc[i] * mol[i]) + pc02)^2 + pc01
        denominator = pc_sum + pc02

        return 1 / (denominator ** 2) + pc01 
    
        # if denominator > 0:
        #     return 1 / (denominator ** 2) + pc01
        # else:
        #     return pc01  # Return just the constant if calculation fails
    
    def calculate_ac(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate acentric factor: w01 * ln(sum(contrib.Ac[i] * mol[i]) + w03)^(1/w02)
        
        Returns:
            Acentric factor (dimensionless) [-] or None if calculation fails
        """
        if 'ac' not in self.pure_properties or 'scalars' not in self.pure_properties:
            return None
            
        ac_data = self.pure_properties['ac']
        scalars = self.pure_properties['scalars']
        
        ac_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in ac_data:
                ac_sum += count * ac_data[group]
            else:
                missing_groups.append(group)
        
        # Handle missing groups according to configured action
        if not self._handle_missing_groups(compound, missing_groups, 'ac'):
            return None
        
        # Get scalars for Ac calculation
        w01 = scalars.get('w01', 0.4085)
        w02 = scalars.get('w02', 0.505)
        w03 = scalars.get('w03', 1.1507)
        
        # acc = w01 * ln(sum(contrib.Ac[i] * mol[i]) + w03)^(1/w02)
        argument = ac_sum + w03

        return w01 * (np.log(argument) ** (1/w02))

    
    def calculate_hv(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate heat of vaporization: Hv0 + sum(contrib.Hv[i] * mol[i])
        
        Returns:
            Heat of vaporization in kJ/mol [kJ/mol] or None if calculation fails
        """
        if 'hv' not in self.pure_properties or 'scalars' not in self.pure_properties:
            return None
            
        hv_data = self.pure_properties['hv']
        scalars = self.pure_properties['scalars']
        
        hv_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in hv_data:
                hv_sum += count * hv_data[group]
            else:
                missing_groups.append(group)
        
        # Handle missing groups according to configured action
        if not self._handle_missing_groups(compound, missing_groups, 'hv'):
            return None
        
        # Get scalar for Hv calculation
        hv0 = scalars.get('hv0')
        # Return in kJ/mol
        return (hv0 + hv_sum)

    def get_available_properties(self) -> list:
        """Get list of properties that can be calculated based on loaded data."""
        available = []
        if 'tm' in self.pure_properties and 'scalars' in self.pure_properties:
            available.append('tm')
        if 'tb' in self.pure_properties and 'scalars' in self.pure_properties:
            available.append('tb')
        if 'tc' in self.pure_properties and 'scalars' in self.pure_properties:
            available.append('tc')
        if 'pc' in self.pure_properties and 'scalars' in self.pure_properties:
            available.append('pc')
        if 'ac' in self.pure_properties and 'scalars' in self.pure_properties:
            available.append('ac')
        if 'hv' in self.pure_properties and 'scalars' in self.pure_properties:
            available.append('hv')
        return available


# Convenience function for quick calculations
def calculate_properties(compound: Dict[str, int], data_path: Union[str, Path] = None) -> Optional[Dict[str, Optional[float]]]:
    """
    Convenience function to calculate all properties for a compound.
    
    Args:
        compound: Dict with group names as keys and counts as values
        data_path: Optional path to property data directory
        
    Returns:
        Dict with calculated properties, or None if compound should be removed
        
    Example:
        >>> compound = {'CH3': 1, 'CH2': 1, 'OH': 1}  # Ethanol
        >>> props = calculate_properties(compound)
        >>> if props:
        >>>     print(f"Tb: {props['tb']:.1f} K")
    """
    calculator = PropertyCalculator(data_path=data_path)
    return calculator.calculate_all_properties(compound)


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
    
    for name, compound in test_compounds.items():
        print(f"\n{name.upper()}: {compound}")
        properties = calculator.calculate_all_properties(compound)
        
        if properties is None:
            print("  Compound removed due to missing group contributions")
            continue

        units = {'tm': 'K', 'tb': 'K', 'tc': 'K', 'pc': 'bar', 'ac': '-', 'hv': 'kJ/mol'}

        for prop_name, value in properties.items():
            if value is not None:
                unit = units.get(prop_name, '')
                print(f"  {prop_name.upper()}: {value:.2f} {unit}")
            else:
                print(f"  {prop_name.upper()}: Not available")
