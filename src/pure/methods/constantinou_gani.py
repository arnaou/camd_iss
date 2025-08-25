"""
Constantinou-Gani Group Contribution Method

Implementation of the Constantinou-Gani method for estimating pure component properties.
Original paper: https://doi.org/10.1002/aic.690401011
Extension: https://doi.org/10.1016/0378-3812(94)02593-P

Calculates:
- tm: Melting point [K]
- tb: Boiling point [K] 
- tc: Critical temperature [K]
- pc: Critical pressure [bar]
- ac: Acentric factor [-]
- hv: Heat of vaporization [kJ/mol]
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .base_method import PropertyMethod


class ConstantinouGani(PropertyMethod):
    """Constantinou-Gani method for pure component property estimation."""
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None, 
                 missing_group_action: str = "remove_compounds"):
        super().__init__(data_path, missing_group_action)
        
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "json"
        else:
            data_path = Path(data_path)
            
        self.data_path = data_path
        self.property_data = {}
        self._load_data()
    
    def _load_data(self):
        """Load Constantinou-Gani property data from JSON files."""
        property_files = {
            'scalars': 'scalars.json',
            'tm': 'tm.json',
            'tb': 'tb.json', 
            'tc': 'tc.json',
            'pc': 'pc.json',
            'ac': 'ac.json',
            'hv': 'hv.json'
        }
        
        for prop_name, filename in property_files.items():
            prop_file = self.data_path / filename
            if prop_file.exists():
                try:
                    with open(prop_file, 'r') as f:
                        prop_data = json.load(f)
                    self.property_data[prop_name] = prop_data.get("data", {})
                except Exception:
                    self.property_data[prop_name] = {}
            else:
                self.property_data[prop_name] = {}
    
    def get_method_name(self) -> str:
        return "Constantinou-Gani"
    
    def get_available_properties(self) -> List[str]:
        """Get list of properties that can be calculated."""
        available = []
        required_data = ['scalars']
        
        for prop in ['tm', 'tb', 'tc', 'pc', 'ac', 'hv']:
            if prop in self.property_data and self.property_data[prop]:
                if all(req in self.property_data for req in required_data):
                    available.append(prop)
        return available
    
    def get_property_units(self) -> Dict[str, str]:
        """Get units for each property."""
        return {
            'tm': 'K',           # Melting point [Kelvin]
            'tb': 'K',           # Boiling point [Kelvin]
            'tc': 'K',           # Critical temperature [Kelvin]
            'pc': 'bar',         # Critical pressure [bar]
            'ac': '-',           # Acentric factor [dimensionless]
            'hv': 'kJ/mol'       # Heat of vaporization [kJ/mol]
        }
    
    def calculate_properties(self, compound: Dict[str, int], 
                           properties: Optional[List[str]] = None) -> Optional[Dict[str, Optional[float]]]:
        """Calculate all or specified properties for a compound."""
        if properties is None:
            properties = self.get_available_properties()
        
        results = {}
        
        for prop in properties:
            if prop in ['tm', 'tb', 'tc']:
                results[prop] = self._calculate_logarithmic(compound, prop)
            elif prop == 'pc':
                results[prop] = self._calculate_pc(compound)
            elif prop == 'ac':
                results[prop] = self._calculate_ac(compound)
            elif prop == 'hv':
                results[prop] = self._calculate_hv(compound)
            else:
                results[prop] = None
                
            # Check if compound should be removed
            if results[prop] is None and self.missing_group_action == "remove_compounds":
                return None
        
        return results
    
    def _calculate_logarithmic(self, compound: Dict[str, int], property_name: str) -> Optional[float]:
        """Calculate properties using logarithmic formula: scalar * ln(sum(contrib[i] * count[i]))"""
        if property_name not in self.property_data or 'scalars' not in self.property_data:
            return None
            
        prop_data = self.property_data[property_name]
        scalars = self.property_data['scalars']
        
        prop_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in prop_data:
                prop_sum += count * prop_data[group]
            else:
                missing_groups.append(group)
        
        if not self._handle_missing_groups(compound, missing_groups, property_name):
            return None
        
        # Get scalar
        scalar_key = f"{property_name.capitalize()}0"  # Tm0, Tb0, Tc0
        scalar = scalars.get(scalar_key)
        
        if scalar is None or prop_sum <= 0:
            return None
            
        return scalar * np.log(prop_sum)
    
    def _calculate_pc(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate critical pressure: 1 / (sum(contrib[i] * count[i]) + pc02)^2 + pc01"""
        if 'pc' not in self.property_data or 'scalars' not in self.property_data:
            return None
            
        pc_data = self.property_data['pc']
        scalars = self.property_data['scalars']
        
        pc_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in pc_data:
                pc_sum += count * pc_data[group]
            else:
                missing_groups.append(group)
        
        if not self._handle_missing_groups(compound, missing_groups, 'pc'):
            return None
        
        pc01 = scalars.get('Pc01')
        pc02 = scalars.get('Pc02')
        
        if pc01 is None or pc02 is None:
            return None
        
        denominator = pc_sum + pc02
        return 1 / (denominator ** 2) + pc01
    
    def _calculate_ac(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate acentric factor: w01 * ln(sum(contrib[i] * count[i]) + w03)^(1/w02)"""
        if 'ac' not in self.property_data or 'scalars' not in self.property_data:
            return None
            
        ac_data = self.property_data['ac']
        scalars = self.property_data['scalars']
        
        ac_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in ac_data:
                ac_sum += count * ac_data[group]
            else:
                missing_groups.append(group)
        
        if not self._handle_missing_groups(compound, missing_groups, 'ac'):
            return None
        
        w01 = scalars.get('w01', 0.4085)
        w02 = scalars.get('w02', 0.505)
        w03 = scalars.get('w03', 1.1507)
        
        argument = ac_sum + w03
        return w01 * (np.log(argument) ** (1/w02))
    
    def _calculate_hv(self, compound: Dict[str, int]) -> Optional[float]:
        """Calculate heat of vaporization: Hv0 + sum(contrib[i] * count[i])"""
        if 'hv' not in self.property_data or 'scalars' not in self.property_data:
            return None
            
        hv_data = self.property_data['hv']
        scalars = self.property_data['scalars']
        
        hv_sum = 0.0
        missing_groups = []
        
        for group, count in compound.items():
            if group in hv_data:
                hv_sum += count * hv_data[group]
            else:
                missing_groups.append(group)
        
        if not self._handle_missing_groups(compound, missing_groups, 'hv'):
            return None
        
        hv0 = scalars.get('hv0')
        if hv0 is None:
            return None
            
        return hv0 + hv_sum