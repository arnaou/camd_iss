"""
Pure Component Property Calculator for CAMD

Calculates pure component properties using the Constantinou-Gani method.
This module can be used standalone for testing properties of individual compounds
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
    
    Calculates:
    - Tm: Melting point [K]
    - Tb: Boiling point [K] 
    - Tc: Critical temperature [K]
    - Pc: Critical pressure [Pa]
    - Ac: Acentric factor [-]
    - Hv: Heat of vaporization [J/mol]
    """
    
    def __init__(self, data_path: Union[str, Path] = None, pure_method: str = "Constantinou-Gani"):
        """
        Initialize the property calculator.
        
        Args:
            data_path: Path to directory containing property data JSON files
            pure_method: Method for pure component properties (currently only "Constantinou-Gani")
        """
        self.pure_method = pure_method
        self.pure_properties = {}
        
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
    
    def calculate_all_properties(self, compound: Dict[str, int]) -> Dict[str, Optional[float]]:
        """
        Calculate all available pure component properties for a compound.
        
        Args:
            compound: Dict with group names as keys and counts as values
                     Example: {'CH3': 2, 'OH': 1} for ethanol
            
        Returns:
            Dict with property names as keys and calculated values as values
        """
        properties = {}
        
        # Calculate each property
        properties['tm'] = self.calculate_tm(compound)
        properties['tb'] = self.calculate_tb(compound)
        properties['tc'] = self.calculate_tc(compound)
        properties['pc'] = self.calculate_pc(compound)
        properties['ac'] = self.calculate_ac(compound)
        properties['hv'] = self.calculate_hv(compound)
        
        return properties
    
    def calculate_tm(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate melting point: Tm0 * ln(sum(contrib.Tm[i] * mol[i]))"""
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
        
        if missing_groups:
            print(f"Warning: Missing Tm data for groups: {missing_groups}")
        
        if tm_sum <= 0:
            return None
            
        # Get scalar for Tm calculation (should be 102.425)
        tm_scalar = scalars.get('Tm0', 102.425)
        return tm_scalar * np.log(tm_sum)
    
    def calculate_tb(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate boiling point: Tb0 * ln(sum(contrib.Tb[i] * mol[i]))"""
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
        
        if missing_groups:
            print(f"Warning: Missing Tb data for groups: {missing_groups}")
        
        if tb_sum <= 0:
            return None
            
        # Get scalar for Tb calculation (should be 204.359)
        tb_scalar = scalars.get('Tb0', 204.359)
        return tb_scalar * np.log(tb_sum)
    
    def calculate_tc(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate critical temperature: Tc0 * ln(sum(contrib.Tc[i] * mol[i]))"""
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
        
        if missing_groups:
            print(f"Warning: Missing Tc data for groups: {missing_groups}")
        
        if tc_sum <= 0:
            return None
            
        # Get scalar for Tc calculation (should be 181.128)
        tc_scalar = scalars.get('Tc0', 181.128)
        return tc_scalar * np.log(tc_sum)
    
    def calculate_pc(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate critical pressure: 1 / (sum(contrib.Pc[i] * mol[i]) + pc02)^2 + pc01"""
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
        
        if missing_groups:
            print(f"Warning: Missing Pc data for groups: {missing_groups}")
        
        # Get scalars for Pc calculation
        pc01 = scalars.get('Pc01', 1.3705)
        pc02 = scalars.get('Pc02', 0.10022)
        
        # Pc = 1 / (sum(contrib.Pc[i] * mol[i]) + pc02)^2 + pc01
        denominator = pc_sum + pc02
        if denominator > 0:
            return 1 / (denominator ** 2) + pc01
        else:
            return pc01  # Return just the constant if calculation fails
    
    def calculate_ac(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate acentric factor: w01 * ln(sum(contrib.Ac[i] * mol[i]) + w03)^(1/w02)"""
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
        
        if missing_groups:
            print(f"Warning: Missing Ac data for groups: {missing_groups}")
        
        # Get scalars for Ac calculation
        w01 = scalars.get('w01', 0.4085)
        w02 = scalars.get('w02', 0.505)
        w03 = scalars.get('w03', 1.1507)
        
        # acc = w01 * ln(sum(contrib.Ac[i] * mol[i]) + w03)^(1/w02)
        argument = ac_sum + w03
        if argument > 0:
            return w01 * (np.log(argument) ** (1/w02))
        else:
            return 0.0  # Return 0 if calculation fails
    
    def calculate_hv(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate heat of vaporization: Hv0 + sum(contrib.Hv[i] * mol[i])"""
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
        
        if missing_groups:
            print(f"Warning: Missing Hv data for groups: {missing_groups}")
        
        # Get scalar for Hv calculation (should be 6.829)
        hv_scalar = scalars.get('Hv0', 6.829)
        # Return in kJ/mol (no conversion needed)
        return (hv_scalar + hv_sum)
    
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
def calculate_properties(compound: Dict[str, int], data_path: Union[str, Path] = None) -> Dict[str, Optional[float]]:
    """
    Convenience function to calculate all properties for a compound.
    
    Args:
        compound: Dict with group names as keys and counts as values
        data_path: Optional path to property data directory
        
    Returns:
        Dict with calculated properties
        
    Example:
        >>> compound = {'CH3': 2, 'OH': 1}  # Ethanol
        >>> props = calculate_properties(compound)
        >>> print(f"Tb: {props['tb']:.1f} K")
    """
    calculator = PropertyCalculator(data_path=data_path)
    return calculator.calculate_all_properties(compound)


if __name__ == "__main__":
    # Example usage
    print("=== Property Calculator Example ===")
    
    # Test compounds
    test_compounds = {
        'ethanol': {'CH3': 1, 'CH2': 1, 'OH': 1},
        'acetone': {'CH3': 2, 'CH3CO': 1},
        'water': {'H2O': 1},
    }
    
    calculator = PropertyCalculator()
    
    for name, compound in test_compounds.items():
        print(f"\n{name.upper()}: {compound}")
        properties = calculator.calculate_all_properties(compound)
        
        units = {'tm': 'K', 'tb': 'K', 'tc': 'K', 'pc': 'Pa', 'ac': '-', 'hv': 'J/mol'}
        
        for prop_name, value in properties.items():
            if value is not None:
                unit = units.get(prop_name, '')
                print(f"  {prop_name.upper()}: {value:.2f} {unit}")
            else:
                print(f"  {prop_name.upper()}: Not available")