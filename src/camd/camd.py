import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import pickle
import json

from ..pure.property_calculator import PropertyCalculator
from ..molecules.compound_validator import CompoundValidator
from ..pure.antoine_calculator import AntoineCalculator


class CAMD:
    """
    Computer-Aided Molecular Design (CAMD) class for solvent screening.
    
    Supports VLE (Vapor-Liquid Equilibrium) calculations using UNIFAC-LY method.
    """
    
    def __init__(self, title: str = "", type: str = "VLE", output_dir: str = "./results", 
                 pure_method: str = "Constantinou-Gani", mixture_method: str = "UNIFAC-LY",
                 missing_group_action: str = "remove_compounds"):
        """
        Initialize CAMD instance.
        
        Args:
            title: Name associated with the CAMD problem
            type: Type of calculation - currently only "VLE" supported  
            output_dir: Directory for results output
            pure_method: Method for pure component properties (default: "Constantinou-Gani")
            mixture_method: Method for mixture properties (default: "UNIFAC-LY")
            missing_group_action: Action for missing groups: "remove_compounds" (default) or "set_zero"
        """
        if type not in ["VLE", "LLE", "SLE"]:
            raise ValueError(f"Type must be one of ['VLE', 'LLE', 'SLE'], got {type}")
        
        if type != "VLE":
            raise NotImplementedError(f"Only VLE is currently implemented, got {type}")
            
        self.title = title
        self.type = type
        self.output_dir = Path(output_dir)
        self.pure_method = pure_method
        self.mixture_method = mixture_method
        self.missing_group_action = missing_group_action
        
        self.groups = []
        self.compounds = {}
        self.system = {}
        self.available_groups = []
        self.design_groups = []  # Groups selected for compound generation
        self.structural_constraints = {}
        self.pure_properties = {}
        self.mixture_properties = {}
        self.generated_compounds = []
        self.property_constraints = {}
        self.calculated_properties = {}  # Store calculated properties for each candidate
        self.calculated_antoine = {}  # Store calculated Antoine parameters for each candidate
        self._generated_count = 0  # Track number of generated (vs predefined) candidates
        self.results = None
        
        # Initialize property calculator, Antoine calculator, and validator
        data_path = Path(__file__).parent.parent / "data" / "json"
        self.property_calculator = PropertyCalculator(data_path=data_path, pure_method=pure_method, 
                                                   missing_group_action=missing_group_action)
        self.antoine_calculator = AntoineCalculator(missing_group_action=missing_group_action, verbose=False)
        self.compound_validator = CompoundValidator(data_path=data_path, molecular_type=1)  # Default to acyclic
        
        self._create_output_directory()
        self._load_data()
    
    def _create_output_directory(self):
        """Create output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.output_dir}")
    
    def _load_data(self):
        """Load pure component and mixture property data based on specified methods."""
        print(f"Loading data for pure method: {self.pure_method}, mixture method: {self.mixture_method}")
        
        try:
            data_path = Path(__file__).parent.parent / "data" / "json"
            
            # Load available groups from smarts.json (common for all methods)
            smarts_file = data_path / "smarts.json"
            if smarts_file.exists():
                with open(smarts_file, 'r') as f:
                    smarts_data = json.load(f)
                self.available_groups = list(smarts_data.get("data", {}).keys())
                print(f"Loaded {len(self.available_groups)} available groups")
            else:
                raise FileNotFoundError("smarts.json not found")
            
            # Load pure component properties based on pure_method
            if self.pure_method == "Constantinou-Gani":
                self._load_constantinou_gani_data(data_path)
            else:
                raise NotImplementedError(f"Pure method {self.pure_method} not implemented")
            
            # Load mixture properties based on mixture_method  
            if self.mixture_method == "UNIFAC-LY":
                self._load_unifac_data(data_path)
            else:
                raise NotImplementedError(f"Mixture method {self.mixture_method} not implemented")
                
            print("Data loading complete")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.available_groups = []
            self.pure_properties = {}
            self.mixture_properties = {}
    
    def _load_constantinou_gani_data(self, data_path: Path):
        """Load Constantinou-Gani pure component property data."""
        cg_properties = {
            'scalars': 'scalars.json',
            'hv': 'hv.json',
            'tc': 'Tc.json', 
            'pc': 'Pc.json',
            'tb': 'Tb.json',
            'tm': 'Tm.json',
            'ac': 'Ac.json'
        }
        
        for prop_name, filename in cg_properties.items():
            prop_file = data_path / filename
            if prop_file.exists():
                with open(prop_file, 'r') as f:
                    prop_data = json.load(f)
                self.pure_properties[prop_name] = prop_data.get("data", {})
                print(f"  Loaded {prop_name} ({len(self.pure_properties[prop_name])} entries)")
            else:
                print(f"  Warning: {filename} not found")
                self.pure_properties[prop_name] = {}
    
    def _load_unifac_data(self, data_path: Path):
        """Load UNIFAC mixture interaction parameters."""
        unifac_properties = {
            'noc': 'noc.json'  # Number of connections (needed for UNIFAC)
            # TODO: Add interaction parameters when available
        }
        
        for prop_name, filename in unifac_properties.items():
            prop_file = data_path / filename
            if prop_file.exists():
                with open(prop_file, 'r') as f:
                    prop_data = json.load(f)
                self.mixture_properties[prop_name] = prop_data.get("data", {})
                print(f"  Loaded {prop_name} ({len(self.mixture_properties[prop_name])} entries)")
            else:
                print(f"  Warning: {filename} not found")
                self.mixture_properties[prop_name] = {}
    
    def add_groups(self, groups: List[str]):
        """
        Add functional groups to define the design space.
        
        Args:
            groups: List of UNIFAC group names (e.g., ['CH3', 'CH2', 'OH'])
        """
        if not isinstance(groups, list):
            raise TypeError("groups must be a list")
        
        # Validate groups exist in database
        invalid_groups = []
        if self.group_data is not None:
            available_groups = self.group_data.index.tolist()
            invalid_groups = [g for g in groups if g not in available_groups]
        
        if invalid_groups:
            raise ValueError(f"Invalid groups not found in database: {invalid_groups}")
        
        self.groups.extend(groups)
        self.groups = list(set(self.groups))  # Remove duplicates
        print(f"Added groups: {groups}")
        print(f"Current groups: {self.groups}")
    
    def set_design_groups(self, groups: List[str]):
        """
        Set the groups available for compound generation.
        
        Args:
            groups: List of UNIFAC group names to use in compound generation
        """
        if not isinstance(groups, list):
            raise TypeError("groups must be a list")
        
        # Validate groups exist in database
        invalid_groups = []
        if self.available_groups:
            invalid_groups = [g for g in groups if g not in self.available_groups]
        
        if invalid_groups:
            raise ValueError(f"Invalid groups not found in database: {invalid_groups}")
        
        self.design_groups = list(set(groups))  # Remove duplicates
        print(f"Set design groups: {self.design_groups}")
    
    def add_constraint_structural(self, 
                                min_number_of_groups: int = 1,
                                max_number_of_groups: int = 10,
                                min_number_of_repetitions: int = 0,
                                max_number_of_repetitions: int = 5,
                                min_func_groups: int = 1,
                                max_func_groups: int = 3,
                                molecular_type: int = 1):
        """
        Add structural constraints for compound generation.
        
        Args:
            min_number_of_groups: Minimum total number of groups in a compound
            max_number_of_groups: Maximum total number of groups in a compound
            min_number_of_repetitions: Minimum times a group can appear in a compound
            max_number_of_repetitions: Maximum times a group can appear in a compound
            min_func_groups: Minimum number of functional groups (non-alkyl)
            max_func_groups: Maximum number of functional groups (non-alkyl)
            molecular_type: 1=acyclic, 0=monocyclic, -1=bicyclic
        """
        if min_number_of_groups < 1:
            raise ValueError("min_number_of_groups must be at least 1")
        
        if max_number_of_groups < min_number_of_groups:
            raise ValueError("max_number_of_groups must be >= min_number_of_groups")
        
        if min_number_of_repetitions < 0:
            raise ValueError("min_number_of_repetitions must be >= 0")
        
        if max_number_of_repetitions < min_number_of_repetitions:
            raise ValueError("max_number_of_repetitions must be >= min_number_of_repetitions")
        
        if min_func_groups < 0:
            raise ValueError("min_func_groups must be >= 0")
        
        if max_func_groups < min_func_groups:
            raise ValueError("max_func_groups must be >= min_func_groups")
        
        if molecular_type not in [-1, 0, 1]:
            raise ValueError("molecular_type must be -1 (bicyclic), 0 (monocyclic), or 1 (acyclic)")
        
        self.structural_constraints = {
            'min_groups': min_number_of_groups,
            'max_groups': max_number_of_groups,
            'min_repetitions': min_number_of_repetitions,
            'max_repetitions': max_number_of_repetitions,
            'min_func_groups': min_func_groups,
            'max_func_groups': max_func_groups,
            'molecular_type': molecular_type
        }
        
        molecular_type_names = {-1: 'bicyclic', 0: 'monocyclic', 1: 'acyclic'}
        
        print(f"Set structural constraints:")
        print(f"  Total groups per compound: {min_number_of_groups}-{max_number_of_groups}")
        print(f"  Group repetitions: {min_number_of_repetitions}-{max_number_of_repetitions}")
        print(f"  Functional groups: {min_func_groups}-{max_func_groups}")
        print(f"  Molecular type: {molecular_type} ({molecular_type_names[molecular_type]})")
    
    def add_constraint_properties(self, property_constraints: Dict[str, List[float]]):
        """
        Add pure component property constraints for candidate screening.
        
        Args:
            property_constraints: Dict where keys are property names and values are
                                [lower_bound, upper_bound] lists.
                                Available properties: 'tb', 'tc', 'pc', 'ac', 'hv'
                                Units: tb/tc [K], pc [Pa], ac [-], hv [J/mol]
                                
        Example:
            constraints = {
                'tb': [300, 450],    # Boiling point 300-450 K
                'tc': [400, 600],    # Critical temperature 400-600 K  
                'pc': [1e5, 5e6],    # Critical pressure 1-50 bar
                'ac': [0.1, 0.8],    # Acentric factor 0.1-0.8
                'hv': [20000, 60000] # Heat of vaporization 20-60 kJ/mol
            }
        """
        if not isinstance(property_constraints, dict):
            raise TypeError("property_constraints must be a dictionary")
        
        # Validate available properties
        valid_properties = {'tm', 'tb', 'tc', 'pc', 'ac', 'hv'}
        invalid_props = set(property_constraints.keys()) - valid_properties
        if invalid_props:
            raise ValueError(f"Invalid property names: {invalid_props}. "
                           f"Valid properties: {valid_properties}")
        
        # Validate constraint format
        for prop_name, bounds in property_constraints.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(f"Bounds for '{prop_name}' must be a list of [lower_bound, upper_bound]")
            
            lower, upper = bounds
            if not isinstance(lower, (int, float)) or not isinstance(upper, (int, float)):
                raise ValueError(f"Bounds for '{prop_name}' must be numeric")
            
            if lower >= upper:
                raise ValueError(f"Lower bound must be < upper bound for '{prop_name}': {bounds}")
            
            # Only restrict negative values for pressure (pc) - other properties can be negative
            if lower < 0 and prop_name in ['pc']:
                raise ValueError(f"'{prop_name}' cannot have negative values: {bounds}")
            
            # Note: tm, tb, tc, ac, hv can have negative values in certain contexts
        
        self.property_constraints = property_constraints.copy()
        
        print(f"Set property constraints:")
        for prop_name, bounds in property_constraints.items():
            units = {'tm': 'K', 'tb': 'K', 'tc': 'K', 'pc': 'Pa', 'ac': '-', 'hv': 'J/mol'}
            unit = units.get(prop_name, '')
            print(f"  {prop_name.upper()}: {bounds[0]}-{bounds[1]} {unit}")
    
    def generate_candidates(self, 
                          max_solutions: int = 10000,
                          verbose: bool = False):
        """
        Generate all feasible compounds using Mixed-Integer Linear Programming (MILP).
        Inspired by the generate_mols function from old_code/src/unifac_utils.py
        
        Args:
            max_solutions: Maximum number of solutions to generate
            verbose: Print solver information
            
        Returns:
            List of generated compound dictionaries
        """
        if not self.design_groups:
            raise ValueError("No design groups set. Use set_design_groups() first.")
        
        if not self.structural_constraints:
            raise ValueError("No structural constraints set. Use add_constraint_structural() first.")
        
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise ImportError("Gurobi not available. Install with: pip install gurobipy")
        
        # Get parameters from structural constraints
        molecular_type = self.structural_constraints['molecular_type']
        min_func_groups = self.structural_constraints['min_func_groups']
        max_func_groups = self.structural_constraints['max_func_groups']
        
        molecular_type_names = {-1: 'bicyclic', 0: 'monocyclic', 1: 'acyclic'}
        
        print(f"Generating compounds using MILP approach...")
        print(f"Design groups: {self.design_groups}")
        print(f"Molecular type: {molecular_type} ({molecular_type_names[molecular_type]})")
        print(f"Functional groups: {min_func_groups}-{max_func_groups}")
        
        # Filter out groups with noc=0 (complete molecules) from MILP
        milp_groups = []
        if 'noc' in self.mixture_properties:
            noc_data = self.mixture_properties['noc']
            for group in self.design_groups:
                if group in noc_data and noc_data[group] == 0:
                    print(f"Excluding complete molecule {group} (noc=0) from MILP")
                else:
                    milp_groups.append(group)
        else:
            milp_groups = self.design_groups[:]
            print("Warning: No noc data available, using all groups in MILP")
        
        if not milp_groups:
            print("No groups available for MILP (all are complete molecules)")
            return []
        
        # Create Gurobi model
        model = gp.Model('Generate_Molecules')
        
        # Convert filtered groups to Gurobi tuplelist
        groups = gp.tuplelist(milp_groups)
        
        # Define variables: group occurrence counts
        x = model.addVars(groups, 
                         lb=self.structural_constraints['min_repetitions'], 
                         ub=self.structural_constraints['max_repetitions'], 
                         vtype=GRB.INTEGER, 
                         name=groups)
        
        # Set solver options
        model.setParam(GRB.Param.PoolSolutions, max_solutions)
        model.setParam(GRB.Param.PoolSearchMode, 2)  # Find multiple solutions
        
        if not verbose:
            model.setParam(GRB.Param.OutputFlag, 0)
        
        # Set objective (feasibility study)
        model.setObjective(1.0, GRB.MINIMIZE)
        
        # Add constraint: minimum total groups
        model.addConstr(x.sum() >= self.structural_constraints['min_groups'], 
                       name="minimum nr. of groups")
        
        # Add constraint: maximum total groups  
        model.addConstr(x.sum() <= self.structural_constraints['max_groups'], 
                       name="maximum nr. of groups")
        
        # Add constraint: Octet rule (Odele-Macchietto Rule 1)
        # 2*m = sum((2 - noc_i) * x_i)
        if 'noc' in self.mixture_properties:
            noc_data = self.mixture_properties['noc']
            model.addConstr(
                gp.quicksum((2 - noc_data.get(j, 2)) * x[j] for j in groups 
                           if j in noc_data) == 2 * molecular_type,
                name="octet rule"
            )
        else:
            print("Warning: No noc data available, skipping octet rule constraint")
        
        # Add constraint: Connectivity rule (Odele-Macchietto Rule 2)
        # sum(x_i where i != j) >= (noc_j - 2) * x_j + 2
        if 'noc' in self.mixture_properties:
            noc_data = self.mixture_properties['noc']
            model.addConstrs(
                gp.quicksum(x[i] for i in groups if i != j) >= 
                (noc_data.get(j, 2) - 2) * x[j] + 2 
                for j in groups if j in noc_data
            )
        
        # Add constraint: functional groups limits (exactly like original code)
        try:
            model.addConstr(
                x.sum() - x['CH3'] - x['CH2'] - x['CH'] - x['C'] >= min_func_groups,
                name="minimum nr. of functional groups"
            )
            model.addConstr(
                x.sum() - x['CH3'] - x['CH2'] - x['CH'] - x['C'] <= max_func_groups,
                name="maximum nr. of functional groups"
            )
        except KeyError as e:
            print(f"Warning: Required alkyl group {e} not in design groups. Skipping functional group constraints.")
        
        # Solve the model
        model.optimize()
        
        # Extract solutions
        nSolutions = model.SolCount
        print(f'Number of feasible compounds found: {nSolutions}')
        
        if nSolutions == 0:
            print("No feasible compounds found with current constraints")
            return []
        
        # Organize solutions
        compounds = []
        for e in range(nSolutions):
            model.setParam(GRB.Param.SolutionNumber, e)
            
            # Extract solution values
            compound = {}
            for group in groups:
                count = int(model.getVarByName(group).Xn)
                if count > 0:  # Only include groups with positive counts
                    compound[group] = count
            
            if compound:  # Only add non-empty compounds
                compounds.append(compound)
        
        self.generated_compounds = compounds
        self._generated_count = len(compounds)  # Track generated count
        print(f"Generated {len(compounds)} unique compounds")
        
        return compounds
    
    def set_candidates(self, candidates: List[Dict[str, int]]):
        """
        Add predefined candidate compounds to the candidate pool.
        
        - If no candidates were previously generated, this becomes the candidate pool
        - If candidates were already generated, predefined ones are added at the top
        - Validates structural rules and removes duplicates (including A, B)
        
        Args:
            candidates: List of candidate compounds, where each compound is a dict
                       with group names as keys and occurrence counts as values.
                       Example: [{'CH3': 2, 'CH2': 1, 'OH': 1}, {'CH3': 3, 'CH2': 2}]
        """
        if not isinstance(candidates, list):
            raise TypeError("candidates must be a list")
        
        if not candidates:
            raise ValueError("candidates list cannot be empty")
        
        valid_candidates = []
        
        for i, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise TypeError(f"Candidate {i} must be a dictionary")
            
            if not candidate:  # Skip empty candidates
                continue
                
            # Validate groups exist in available groups
            invalid_groups = []
            if self.available_groups:
                invalid_groups = [g for g in candidate.keys() if g not in self.available_groups]
            
            if invalid_groups:
                print(f"Warning: Skipping candidate {i} with invalid groups: {invalid_groups}")
                continue
            
            # Validate structural rules (octet and connectivity)
            try:
                self._validate_single_compound(candidate, i)
            except ValueError as e:
                print(f"Warning: Skipping candidate {i} due to structural violation: {e}")
                continue
            
            # Check if candidate is identical to A or B
            if self._is_compound_excluded(candidate):
                print(f"Warning: Skipping candidate {i} - identical to system component A or B")
                continue
            
            # Check if candidate already exists in generated compounds
            if self._is_duplicate_candidate(candidate):
                print(f"Warning: Skipping candidate {i} - already exists in candidate pool")
                continue
            
            valid_candidates.append(candidate)
        
        # Add valid predefined candidates
        if valid_candidates:
            if self.generated_compounds:
                # Add to existing generated candidates (at the top)
                self.generated_compounds = valid_candidates + self.generated_compounds
                print(f"Added {len(valid_candidates)} predefined candidates to existing pool")
            else:
                # No generated candidates, use predefined as the pool
                self.generated_compounds = valid_candidates
                print(f"Set {len(valid_candidates)} predefined candidates as candidate pool")
                
            print(f"Total candidates: {len(self.generated_compounds)}")
        else:
            print("No valid predefined candidates to add")
        
        return valid_candidates
    
    def _is_duplicate_candidate(self, candidate: Dict[str, int]) -> bool:
        """
        Check if a candidate already exists in the generated compounds list.
        
        Args:
            candidate: Candidate to check
            
        Returns:
            True if candidate is a duplicate, False otherwise
        """
        for existing_candidate in self.generated_compounds:
            if self._compounds_equal(candidate, existing_candidate):
                return True
        return False
    
    def calculate_pure_properties(self):
        """
        Calculate pure component properties for ALL candidates using PropertyCalculator.
        
        Stores results in self.calculated_properties where keys are candidate indices
        and values are dicts with properties: {'tm': value, 'tb': value, 'tc': value, 'pc': value, 'ac': value, 'hv': value}
        
        Returns:
            int: Number of candidates with properties calculated successfully
        """
        if not self.generated_compounds:
            print("No candidates to calculate properties for")
            return 0
        
        print(f"Calculating pure component properties for {len(self.generated_compounds)} candidates...")
        
        self.calculated_properties = {}  # Reset stored properties
        successful_calculations = 0
        compounds_to_remove = []  # Track compounds that should be removed
        
        for i, compound in enumerate(self.generated_compounds):
            try:
                properties = self.property_calculator.calculate_all_properties(compound)
                if properties is None:  # Compound marked for removal due to missing groups
                    compounds_to_remove.append(i)
                else:
                    self.calculated_properties[i] = properties
                    successful_calculations += 1
            except Exception as e:
                print(f"Warning: Could not calculate properties for candidate {i}: {e}")
                self.calculated_properties[i] = {
                    'tm': None, 'tb': None, 'tc': None, 'pc': None, 'ac': None, 'hv': None
                }
        
        # Remove compounds with missing groups if action is "remove_compounds"
        if compounds_to_remove and self.missing_group_action == "remove_compounds":
            print(f"\nRemoving {len(compounds_to_remove)} compounds due to missing group contributions...")
            # Remove in reverse order to maintain indices
            for i in reversed(compounds_to_remove):
                del self.generated_compounds[i]
                # Update _generated_count if removing generated compounds
                if i < self._generated_count:
                    self._generated_count -= 1
            
            # Recalculate properties with new indices
            print(f"Recalculating properties for remaining {len(self.generated_compounds)} candidates...")
            self.calculated_properties = {}
            successful_calculations = 0
            
            for i, compound in enumerate(self.generated_compounds):
                try:
                    properties = self.property_calculator.calculate_all_properties(compound)
                    if properties is not None:
                        self.calculated_properties[i] = properties
                        successful_calculations += 1
                except Exception as e:
                    print(f"Warning: Could not calculate properties for candidate {i}: {e}")
                    self.calculated_properties[i] = {
                        'tm': None, 'tb': None, 'tc': None, 'pc': None, 'ac': None, 'hv': None
                    }
        
        # Print missing groups summary
        self.property_calculator.print_missing_groups_summary()
        
        print(f"Property calculation complete: {successful_calculations}/{len(self.generated_compounds)} successful")
        return successful_calculations
    
    def calculate_antoine_parameters(self):
        """
        Calculate Antoine parameters for ALL candidates using calculated pure component properties.
        
        Requires that calculate_pure_properties() has been called first.
        Stores results in self.calculated_antoine where keys are candidate indices
        and values are dicts with Antoine parameters: {'A': value, 'B': value, 'C': value}
        
        Returns:
            int: Number of candidates with Antoine parameters calculated successfully
        """
        if not self.calculated_properties:
            print("No calculated properties available. Run calculate_pure_properties() first.")
            return 0
        
        print(f"Calculating Antoine parameters for {len(self.calculated_properties)} candidates...")
        
        self.calculated_antoine = {}  # Reset stored Antoine parameters
        successful_calculations = 0
        
        for i, properties in self.calculated_properties.items():
            if i >= len(self.generated_compounds):
                continue  # Skip if compound was removed
                
            compound = self.generated_compounds[i]
            
            try:
                antoine_params = self.antoine_calculator.calculate_antoine_parameters(compound, properties)
                
                if antoine_params is None:  # Compound marked for removal due to missing critical properties
                    self.calculated_antoine[i] = None
                else:
                    self.calculated_antoine[i] = antoine_params
                    successful_calculations += 1
                    
            except Exception as e:
                print(f"Warning: Could not calculate Antoine parameters for candidate {i}: {e}")
                self.calculated_antoine[i] = None
        
        # Print missing properties summary
        self.antoine_calculator.print_missing_properties_summary()
        
        print(f"Antoine parameter calculation complete: {successful_calculations}/{len(self.calculated_properties)} successful")
        return successful_calculations
    
    def set_pure_property(self, candidate_index: int, property_name: str, value: float):
        """
        Manually set or override a pure component property for a specific candidate.
        
        Args:
            candidate_index: Index of the candidate in the generated_compounds list
            property_name: Property name ('tm', 'tb', 'tc', 'pc', 'ac', 'hv')
            value: Property value to set
        """
        if candidate_index < 0 or candidate_index >= len(self.generated_compounds):
            raise ValueError(f"Invalid candidate index {candidate_index}. Must be 0 to {len(self.generated_compounds)-1}")
        
        valid_properties = {'tm', 'tb', 'tc', 'pc', 'ac', 'hv'}
        if property_name not in valid_properties:
            raise ValueError(f"Invalid property name '{property_name}'. Valid properties: {valid_properties}")
        
        if not isinstance(value, (int, float)):
            raise TypeError(f"Property value must be numeric, got {type(value)}")
        
        # Initialize properties dict for this candidate if it doesn't exist
        if candidate_index not in self.calculated_properties:
            self.calculated_properties[candidate_index] = {
                'tm': None, 'tb': None, 'tc': None, 'pc': None, 'ac': None, 'hv': None
            }
        
        # Set the property
        old_value = self.calculated_properties[candidate_index].get(property_name)
        self.calculated_properties[candidate_index][property_name] = value
        
        compound = self.generated_compounds[candidate_index]
        units = {'tm': 'K', 'tb': 'K', 'tc': 'K', 'pc': 'Pa', 'ac': '-', 'hv': 'J/mol'}
        unit = units.get(property_name, '')
        
        if old_value is not None:
            print(f"Updated {property_name.upper()} for candidate {candidate_index} ({compound}): {old_value} → {value} {unit}")
        else:
            print(f"Set {property_name.upper()} for candidate {candidate_index} ({compound}): {value} {unit}")
    
    def set_binary_system_property(self, compound: str, property_name: str, value: float):
        """
        Manually set or override a pure component property for compound A or B in the binary system.
        
        Args:
            compound: 'A' or 'B' to specify which compound in the binary system
            property_name: Property name ('tm', 'tb', 'tc', 'pc', 'ac', 'hv')
            value: Property value to set
        """
        if not self.system:
            raise ValueError("No binary system defined. Use add_binary_system() first.")
        
        if compound not in ['A', 'B']:
            raise ValueError("Compound must be 'A' or 'B'")
        
        if compound not in self.system:
            raise ValueError(f"Compound {compound} not found in binary system")
        
        valid_properties = {'tm', 'tb', 'tc', 'pc', 'ac', 'hv'}
        if property_name not in valid_properties:
            raise ValueError(f"Invalid property name '{property_name}'. Valid properties: {valid_properties}")
        
        if not isinstance(value, (int, float)):
            raise TypeError(f"Property value must be numeric, got {type(value)}")
        
        # Initialize binary system properties if not exists
        if not hasattr(self, 'binary_system_properties'):
            self.binary_system_properties = {'A': {}, 'B': {}}
        
        # Set the property
        old_value = self.binary_system_properties[compound].get(property_name)
        self.binary_system_properties[compound][property_name] = value
        
        compound_formula = self.system[compound]
        units = {'tm': 'K', 'tb': 'K', 'tc': 'K', 'pc': 'bar', 'ac': '-', 'hv': 'kJ/mol'}
        unit = units.get(property_name, '')
        
        if old_value is not None:
            print(f"Updated {property_name.upper()} for compound {compound} ({compound_formula}): {old_value} → {value} {unit}")
        else:
            print(f"Set {property_name.upper()} for compound {compound} ({compound_formula}): {value} {unit}")
    
    def get_binary_system_property(self, compound: str, property_name: str):
        """
        Get a property value for compound A or B, either manually set or calculated.
        
        Args:
            compound: 'A' or 'B' to specify which compound in the binary system
            property_name: Property name ('tm', 'tb', 'tc', 'pc', 'ac', 'hv')
            
        Returns:
            Property value if available, None otherwise
        """
        if not self.system or compound not in self.system:
            return None
        
        # Check for manually set properties first
        if hasattr(self, 'binary_system_properties') and compound in self.binary_system_properties:
            manual_value = self.binary_system_properties[compound].get(property_name)
            if manual_value is not None:
                return manual_value
        
        # Calculate property using property calculator
        compound_composition = self.system[compound]
        all_properties = self.property_calculator.calculate_all_properties(compound_composition)
        return all_properties.get(property_name)
    
    def show_binary_system_properties(self):
        """Display calculated and manually set properties for compounds A and B."""
        if not self.system:
            print("No binary system defined.")
            return
        
        print("\n=== Binary System Properties ===")
        
        for compound in ['A', 'B']:
            if compound in self.system:
                composition = self.system[compound]
                print(f"\nCompound {compound}: {composition}")
                
                # Get all properties
                properties = ['tm', 'tb', 'tc', 'pc', 'ac', 'hv']
                units = {'tm': 'K', 'tb': 'K', 'tc': 'K', 'pc': 'bar', 'ac': '-', 'hv': 'kJ/mol'}
                
                for prop in properties:
                    value = self.get_binary_system_property(compound, prop)
                    if value is not None:
                        unit = units.get(prop, '')
                        
                        # Check if this is a manually set value
                        is_manual = (hasattr(self, 'binary_system_properties') and 
                                   compound in self.binary_system_properties and 
                                   self.binary_system_properties[compound].get(prop) is not None)
                        
                        manual_indicator = " (manual)" if is_manual else ""
                        
                        # Format value appropriately
                        if prop == 'hv' and value is not None:
                            display_value = value / 1000  # Convert J/mol to kJ/mol
                        else:
                            display_value = value
                            
                        print(f"  {prop.upper()}: {display_value:.2f} {unit}{manual_indicator}")
                    else:
                        print(f"  {prop.upper()}: Not available")
    
    def add_compounds(self, compounds: Dict[str, List[int]]):
        """
        Add compounds defined by group occurrences.
        
        Args:
            compounds: Dict where keys are group names and values are lists of 
                      occurrence counts for each compound
                      Example: {'CH3': [2, 2, 2], 'CH2': [1, 2, 3]}
        """
        if not isinstance(compounds, dict):
            raise TypeError("compounds must be a dictionary")
        
        # Check if groups exist
        invalid_groups = [g for g in compounds.keys() if g not in self.groups]
        if invalid_groups:
            raise ValueError(f"Groups not in design space: {invalid_groups}. "
                           f"Add them first with add_groups()")
        
        # TODO: Add valency validation
        self._validate_compounds(compounds)
        
        self.compounds.update(compounds)
        print(f"Added compounds with groups: {list(compounds.keys())}")
    
    def add_binary_system(self, compound_A: Dict[str, int], compound_B: Dict[str, int]):
        """
        Add binary mixture definition with compounds A and B.
        
        Args:
            compound_A: Groups and counts for compound A
            compound_B: Groups and counts for compound B
        """
        if not isinstance(compound_A, dict) or not isinstance(compound_B, dict):
            raise TypeError("Both compounds must be dictionaries")
        
        # Validate groups exist in available groups
        all_groups = set(compound_A.keys()) | set(compound_B.keys())
        invalid_groups = all_groups - set(self.available_groups)
        if invalid_groups:
            raise ValueError(f"Invalid groups not in database: {invalid_groups}")
        
        # Convert to compound format for validation
        system_compounds = {}
        for group in all_groups:
            count_A = compound_A.get(group, 0)
            count_B = compound_B.get(group, 0)
            system_compounds[group] = [count_A, count_B]
        
        # Validate structural feasibility
        self._validate_compounds(system_compounds)
        
        # Store the system
        self.system = {
            'A': compound_A,
            'B': compound_B
        }
        self.compounds = system_compounds
        
        print(f"Added binary system:")
        print(f"  Compound A: {compound_A}")
        print(f"  Compound B: {compound_B}")
    
    def add_system(self, system: Dict[str, List[int]]):
        """
        Add system definition (legacy method, use add_binary_system for better interface).
        
        Args:
            system: Same format as add_compounds
        """
        if not isinstance(system, dict):
            raise TypeError("system must be a dictionary")
            
        # For now, only binary systems
        if self.system and len(self.system) >= 2:
            raise NotImplementedError("Currently only binary systems are supported")
        
        self.add_compounds(system)
        self.system.update(system)
        print(f"Added system: {list(system.keys())}")
    
    def _validate_compounds(self, compounds: Dict[str, List[int]]):
        """
        Validate compound definitions using Odele and Macchietto structural feasibility rules.
        
        Args:
            compounds: Compound definitions to validate
        """
        # Get number of compounds
        compound_counts = list(compounds.values())
        if not compound_counts:
            return
            
        n_compounds = len(compound_counts[0])
        
        # Check all groups have same number of compounds
        for group, counts in compounds.items():
            if len(counts) != n_compounds:
                raise ValueError(f"All groups must have same number of compounds. "
                               f"Group {group} has {len(counts)}, expected {n_compounds}")
        
        # Validate each compound using structural feasibility rules
        for i in range(n_compounds):
            compound_groups = {group: counts[i] for group, counts in compounds.items() if counts[i] > 0}
            self._validate_single_compound(compound_groups, i)
        
        print(f"Validated {n_compounds} compounds using Odele-Macchietto rules")
    
    def _validate_single_compound(self, compound_groups: Dict[str, int], compound_index: int):
        """
        Validate a single compound using CompoundValidator.
        
        Args:
            compound_groups: Groups and their counts for this compound
            compound_index: Index of compound for error reporting
        """
        if not compound_groups:
            return
        
        # Use CompoundValidator to check structural feasibility
        is_valid = self.compound_validator.validate_compound(compound_groups, verbose=False)
        if not is_valid:
            raise ValueError(f"Compound {compound_index} violates structural feasibility rules")
    
    def validate_system(self):
        """Check system consistency and thermodynamic validity."""
        if not self.system:
            raise ValueError("No system defined. Use add_binary_system() first.")
        
        # Filter out candidates identical to compounds A or B
        if self.generated_compounds:
            original_count = len(self.generated_compounds)
            filtered_candidates = []
            excluded_count = 0
            
            for candidate in self.generated_compounds:
                if self._is_compound_excluded(candidate):
                    excluded_count += 1
                else:
                    filtered_candidates.append(candidate)
            
            self.generated_compounds = filtered_candidates
            
            if excluded_count > 0:
                print(f"Excluded {excluded_count} candidates identical to system components A or B")
                print(f"Remaining candidates: {len(self.generated_compounds)}")
        
        print("System validation passed")
        return True
    
    def check_unifac(self, compounds=None, check_ternary_systems=True, 
                     missing_interaction_action="remove_compounds", verbose=True):
        """
        Check UNIFAC R/Q parameters and binary interaction parameters for compounds.
        
        Validates that all groups in the compounds have:
        1. R (volume) and Q (surface area) parameters for UNIFAC calculations
        2. Binary interaction parameters between different main groups
        3. If check_ternary_systems=True, validates A-B-solvent interactions for VLE
        
        Args:
            compounds: List of compound dictionaries or None to use generated_compounds
                      Each compound should be {'group': count} format
            check_ternary_systems: Check interactions for A+B+each_solvent systems
            missing_interaction_action: Action for missing interactions:
                                       "remove_compounds" (default) or "set_zero"
            verbose: Print detailed information about missing parameters
        
        Returns:
            tuple: (missing_rq_groups, missing_interactions, compounds_to_remove)
                   - missing_rq_groups: List of groups missing R/Q parameters
                   - missing_interactions: List of main group pairs missing interactions  
                   - compounds_to_remove: List of compound indices with missing parameters
        """
        import warnings
        import pandas as pd
        import json
        from pathlib import Path
        
        # Use generated compounds if none provided
        if compounds is None:
            compounds = self.generated_compounds
        
        if not compounds:
            if verbose:
                print("No compounds to check")
            return [], [], []
        
        # Collect all unique groups from compounds
        all_groups = set()
        for compound in compounds:
            all_groups.update(compound.keys())
        
        # Load UNIFAC group data
        try:
            data_path = Path(__file__).parent.parent / "data"
            with open(data_path / 'unifac_groups.pkl', 'rb') as f:
                unifac_dict = pickle.load(f)
        except FileNotFoundError as e:
            if verbose:
                warnings.warn(f"Could not load UNIFAC groups: {e}", UserWarning)
            return list(all_groups), [], list(range(len(compounds)))
        
        # Load R/Q parameters and interaction data from JSON files
        try:
            # Load R and Q parameters from JSON files
            with open(data_path / 'json' / 'rk.json', 'r') as f:
                rk_data = json.load(f).get("data", {})
            with open(data_path / 'json' / 'qk.json', 'r') as f:
                qk_data = json.load(f).get("data", {})
            
            # Load interaction data from JSON file
            with open(data_path / 'json' / 'a.json', 'r') as f:
                interaction_data = json.load(f).get("data", {})
            
        except Exception as e:
            if verbose:
                warnings.warn(f"Could not load UNIFAC data: {e}", UserWarning)
            return list(all_groups), [], list(range(len(compounds)))
        
        # Check R/Q parameters
        missing_rq_groups = []
        for group in all_groups:
            rk_missing = group not in rk_data or rk_data[group] is None
            qk_missing = group not in qk_data or qk_data[group] is None
            
            if rk_missing or qk_missing:
                missing_rq_groups.append(group)
                if verbose:
                    rk_status = "missing" if group not in rk_data else f"None" if rk_data[group] is None else f"{rk_data[group]}"
                    qk_status = "missing" if group not in qk_data else f"None" if qk_data[group] is None else f"{qk_data[group]}"
                    print(f"    Group '{group}': RK={rk_status}, QK={qk_status}")
        
        # Check binary interaction parameters using JSON data
        # Use only the groups that appear in the CAMD compounds
        compound_groups = list(all_groups)
        missing_interactions = []
        missing_sub_groups = []
        
        # Check interactions between pairs of compound groups (upper triangle + diagonal)
        for i, group_i in enumerate(compound_groups):
            for j, group_j in enumerate(compound_groups):
                if i <= j:  # Check upper triangular part including diagonal
                    # Check if groups exist in interaction data
                    if group_i in interaction_data and group_j in interaction_data[group_i]:
                        interaction_value = interaction_data[group_i][group_j]
                    elif group_j in interaction_data and group_i in interaction_data[group_j]:
                        # Check symmetric entry
                        interaction_value = interaction_data[group_j][group_i]
                    else:
                        # No interaction data found - treat as missing
                        interaction_value = None
                    
                    # Check if the interaction is missing (None) or zero
                    if interaction_value is None or interaction_value == 0:
                        # Find main groups for these groups
                        main_g1 = next((value['main_g'] for value in unifac_dict.values() 
                                       if value['grp'] == group_i), None)
                        main_g2 = next((value['main_g'] for value in unifac_dict.values() 
                                       if value['grp'] == group_j), None)
                        
                        # If the main groups are different, this is a missing interaction
                        # (Zero interactions within the same main group are expected)
                        if main_g1 and main_g2 and main_g1 != main_g2:
                            missing_interactions.append((main_g1, main_g2))
                            missing_sub_groups.append((group_i, group_j))
        
        # Check ternary system interactions (A+B+each solvent) if requested
        ternary_missing_interactions = []
        ternary_compounds_to_remove = []
        
        if check_ternary_systems and hasattr(self, 'system') and self.system:
            # Get groups from compounds A and B
            compound_A_groups = set(self.system['A'].keys())
            compound_B_groups = set(self.system['B'].keys())
            system_groups = compound_A_groups | compound_B_groups
            
            if verbose:
                print(f"\n   Checking ternary systems (A+B+solvent):")
                print(f"   Compound A groups: {sorted(compound_A_groups)}")
                print(f"   Compound B groups: {sorted(compound_B_groups)}")
            
            # Check each solvent candidate
            for i, solvent in enumerate(compounds):
                solvent_groups = set(solvent.keys())
                ternary_system_groups = system_groups | solvent_groups
                missing_for_solvent = []
                
                # Check all interactions in the ternary system
                for group_i in ternary_system_groups:
                    for group_j in ternary_system_groups:
                        if group_i <= group_j:  # Upper triangle + diagonal
                            # Check if interaction exists
                            if group_i in interaction_data and group_j in interaction_data[group_i]:
                                interaction_value = interaction_data[group_i][group_j]
                            elif group_j in interaction_data and group_i in interaction_data[group_j]:
                                interaction_value = interaction_data[group_j][group_i]
                            else:
                                interaction_value = None
                            
                            # Check if interaction is missing
                            if interaction_value is None or interaction_value == 0:
                                # Find main groups
                                main_g1 = next((value['main_g'] for value in unifac_dict.values() 
                                               if value['grp'] == group_i), None)
                                main_g2 = next((value['main_g'] for value in unifac_dict.values() 
                                               if value['grp'] == group_j), None)
                                
                                # Only flag if different main groups
                                if main_g1 and main_g2 and main_g1 != main_g2:
                                    missing_for_solvent.append((group_i, group_j, main_g1, main_g2))
                
                # If solvent has missing interactions, mark for removal
                if missing_for_solvent:
                    ternary_compounds_to_remove.append(i)
                    ternary_missing_interactions.extend(missing_for_solvent)
                    if verbose:
                        print(f"   Solvent {i} ({solvent}): {len(missing_for_solvent)} missing interactions")
                        for g1, g2, mg1, mg2 in missing_for_solvent[:3]:  # Show first 3
                            print(f"     Missing: {g1}↔{g2} (main groups {mg1}↔{mg2})")
                        if len(missing_for_solvent) > 3:
                            print(f"     ... and {len(missing_for_solvent)-3} more")
            
            if verbose and ternary_compounds_to_remove:
                print(f"   Ternary check: {len(ternary_compounds_to_remove)}/{len(compounds)} solvents have missing interactions")
        
        # Validate missing_interaction_action parameter
        if missing_interaction_action not in ["remove_compounds", "set_zero"]:
            raise ValueError(f"missing_interaction_action must be 'remove_compounds' or 'set_zero', got '{missing_interaction_action}'")
        
        # Combine results based on the chosen action for missing interactions
        compounds_to_remove = []
        
        for i, compound in enumerate(compounds):
            remove_reasons = []
            
            # Always remove compounds with missing R/Q parameters (can't be fixed by setting interactions to zero)
            missing_in_compound = [g for g in compound.keys() if g in missing_rq_groups]
            if missing_in_compound:
                remove_reasons.append(f"missing R/Q parameters for {missing_in_compound}")
            
            # Handle missing interactions based on the chosen action
            if missing_interaction_action == "remove_compounds":
                # Check if any main group pairs in this compound have missing interactions (general check)
                compound_main_groups = set()
                for group in compound.keys():
                    main_g = next((value['main_g'] for value in unifac_dict.values() 
                                  if value['grp'] == group), None)
                    if main_g:
                        compound_main_groups.add(main_g)
                
                # Check for missing interactions between main groups in this compound
                has_missing_interaction = False
                for interaction in missing_interactions:
                    mg1, mg2 = interaction
                    if mg1 in compound_main_groups and mg2 in compound_main_groups:
                        has_missing_interaction = True
                        break
                
                if has_missing_interaction:
                    remove_reasons.append("missing general binary interactions")
                
                # Check ternary system issues
                if check_ternary_systems and i in ternary_compounds_to_remove:
                    remove_reasons.append("missing ternary system interactions")
            
            # For "set_zero" action, we don't remove compounds due to missing interactions
            # (they will be handled by setting interaction parameters to 0.0 in VLE calculations)
            
            # Add to removal list if any issues found
            if remove_reasons:
                compounds_to_remove.append(i)
                if verbose:
                    print(f"Compound {i} ({compound}): {'; '.join(remove_reasons)}")
        
        # Calculate total possible interactions for compound groups (upper triangle + diagonal)
        total_compound_pairs = len(compound_groups) * (len(compound_groups) + 1) // 2
        
        # Print detailed summary (similar to inspect_interractions)
        if verbose:
            print(f"\n=== UNIFAC Parameter Check Results ===")
            
            # R/Q parameter summary
            if missing_rq_groups:
                print(f"❌ Groups missing R/Q parameters: {len(missing_rq_groups)} groups")
                print(f"   Missing groups: {missing_rq_groups}")
            else:
                print(f"✅ All compound groups have R/Q parameters ({len(compound_groups)} groups)")
            
            # Binary interaction parameter summary - commented out for cleaner display
            # Focus on A-B binary and ternary system validation instead
            # if missing_interactions:
            #     print(f"\n❌ Binary interaction parameters:")
            #     print(f"   Total compound group pairs checked: {total_compound_pairs}")
            #     print(f"   Missing interactions: {len(missing_interactions)} main group pairs")
            #     print(f"   Missing interaction percentage: {len(missing_interactions)/total_compound_pairs*100:.1f}%")
            #     
            #     # Show specific missing main group interactions
            #     print(f"   Missing main group pairs: {missing_interactions}")
            #     
            #     # Show specific missing subgroup interactions
            #     if missing_sub_groups:
            #         print(f"   Missing subgroup pairs: {missing_sub_groups}")
            #         
            #     # Display interaction matrix
            #     # Commented out for cleaner display - may be needed later
            #     # if len(compound_groups) <= 20:  # Only show matrix for reasonably sized systems
            #     #     print(f"\n   Interaction Matrix ({len(compound_groups)}x{len(compound_groups)}):")
            #     #     print("   " + "".join(f"{g:>8}" for g in compound_groups))
            #     #     for i, group_i in enumerate(compound_groups):
            #     #         row_values = []
            #     #         for j, group_j in enumerate(compound_groups):
            #     #             if group_i in interaction_data and group_j in interaction_data[group_i]:
            #     #                 value = interaction_data[group_i][group_j]
            #     #             elif group_j in interaction_data and group_i in interaction_data[group_j]:
            #     #                 value = interaction_data[group_j][group_i]
            #     #             else:
            #     #                 value = None
            #     #             
            #     #             if value is None:
            #     #                 row_values.append("  MISS")
            #     #             elif value == 0:
            #     #                 row_values.append("     0")
            #     #             else:
            #     #                 row_values.append(f"{value:6.1f}")
            #     #         
            #     #         print(f"{group_i:>6} " + "".join(f"{v:>8}" for v in row_values))
            #     
            #     # Issue warnings like inspect_interractions
            #     # warnings.warn(f"\nWarning: A total of {len(missing_interactions)} out of {total_compound_pairs} "
            #     #             f"interaction parameters are missing for compound groups", UserWarning)
            #     # warnings.warn(f"\nWarning: The interaction between the following main groups "
            #     #             f"{missing_interactions} have been set to 0", UserWarning)
            #     # if missing_sub_groups:
            #     #     warnings.warn(f"\nWarning: The interaction between the following sub groups "
            #     #                 f"{missing_sub_groups} have been set to 0", UserWarning)
            # else:
            #     print(f"✅ All binary interactions available ({total_compound_pairs} pairs checked)")
            
            # Show A-B system interaction matrix if ternary system check is enabled
            if check_ternary_systems and hasattr(self, 'system') and self.system:
                # Get groups from compounds A and B
                compound_A_groups = sorted(self.system['A'].keys())
                compound_B_groups = sorted(self.system['B'].keys())
                system_groups = sorted(set(compound_A_groups + compound_B_groups))
                
                # Get all solvent groups (excluding A and B groups)
                solvent_groups = sorted([g for g in compound_groups if g not in system_groups])
                
                if system_groups and solvent_groups:
                    print(f"\n   A-B System Interaction Matrix:")
                    print(f"   System groups (A+B): {system_groups}")
                    print(f"   Solvent groups: {solvent_groups[:10]}{'...' if len(solvent_groups) > 10 else ''}")
                    
                    # Create matrix with system groups as rows and solvent groups as columns
                    if len(solvent_groups) <= 20:  # Show matrix for A-B system interactions
                        print(f"\n   Interactions (System groups × Solvent groups):")
                        
                        # Header with solvent groups
                        header = "".join(f"{g:>8}" for g in solvent_groups)
                        print(f"{'':>8}{header}")
                        
                        # Each system group as a row
                        for sys_group in system_groups:
                            row_values = []
                            missing_count = 0
                            
                            for solv_group in solvent_groups:
                                # Check interaction between system group and solvent group
                                if sys_group in interaction_data and solv_group in interaction_data[sys_group]:
                                    value = interaction_data[sys_group][solv_group]
                                elif solv_group in interaction_data and sys_group in interaction_data[solv_group]:
                                    value = interaction_data[solv_group][sys_group]
                                else:
                                    value = None
                                
                                if value is None:
                                    row_values.append("  MISS")
                                    missing_count += 1
                                elif value == 0:
                                    row_values.append("     0")
                                else:
                                    row_values.append(f"{value:6.1f}")
                            
                            # Print row with system group name and values
                            row_str = "".join(f"{v:>8}" for v in row_values)
                            status = f" ({missing_count} missing)" if missing_count > 0 else " (complete)"
                            print(f"{sys_group:>8}{row_str}{status}")
                    else:
                        print(f"   Matrix too large to display ({len(system_groups)}×{len(solvent_groups)})")
                        
                        # Show summary statistics instead
                        total_interactions = len(system_groups) * len(solvent_groups)
                        missing_interactions_count = 0
                        
                        for sys_group in system_groups:
                            for solv_group in solvent_groups:
                                if sys_group in interaction_data and solv_group in interaction_data[sys_group]:
                                    value = interaction_data[sys_group][solv_group]
                                elif solv_group in interaction_data and sys_group in interaction_data[solv_group]:
                                    value = interaction_data[solv_group][sys_group]
                                else:
                                    value = None
                                
                                if value is None:
                                    missing_interactions_count += 1
                        
                        print(f"   Total A-B system interactions: {total_interactions}")
                        print(f"   Missing interactions: {missing_interactions_count} ({missing_interactions_count/total_interactions*100:.1f}%)")
                        print(f"   Available interactions: {total_interactions - missing_interactions_count} ({(total_interactions - missing_interactions_count)/total_interactions*100:.1f}%)")
            
            # Compound removal summary
            if compounds_to_remove:
                print(f"\n❌ Compounds affected:")
                print(f"   Compounds to remove: {len(compounds_to_remove)} out of {len(compounds)}")
                print(f"   Removal percentage: {len(compounds_to_remove)/len(compounds)*100:.1f}%")
                print(f"   Remaining compounds: {len(compounds) - len(compounds_to_remove)}")
                
                # Show action taken for missing interactions
                if missing_interaction_action == "set_zero" and (missing_interactions or ternary_missing_interactions):
                    print(f"   Missing interactions handling: Set to 0.0 (missing_interaction_action='set_zero')")
                    if ternary_missing_interactions:
                        print(f"   Ternary interactions set to 0.0: {len(ternary_missing_interactions)} pairs")
                elif check_ternary_systems:
                    print(f"   Ternary system issues: {len(ternary_compounds_to_remove)} compounds removed")
            else:
                print(f"\n✅ All compounds have complete UNIFAC parameters")
                if check_ternary_systems:
                    if missing_interaction_action == "set_zero" and (missing_interactions or ternary_missing_interactions):
                        print(f"✅ All compounds usable (missing interactions set to 0.0)")
                    else:
                        print(f"✅ All compounds compatible with A+B ternary systems")
            
            print("=" * 60)
        
        return missing_rq_groups, missing_interactions, compounds_to_remove
    
    def filter_pure_properties(self):
        """
        Filter candidates based on pure component property constraints using stored properties.
        
        Uses properties calculated by calculate_pure_properties() or set manually with set_pure_property().
        
        Returns:
            Dict: Detailed filtering statistics
        """
        if not self.property_constraints:
            return {
                'total': len(self.generated_compounds),
                'generated': self._generated_count,
                'predefined': len(self.generated_compounds) - self._generated_count,
                'passed': len(self.generated_compounds),
                'failed': 0,
                'constraints': [],
                'failure_breakdown': {},
                'first_failure_stats': {},
                'skipped': True,
                'skip_reason': 'No property constraints defined'
            }
            
        if not self.generated_compounds:
            return {
                'total': 0, 'generated': 0, 'predefined': 0, 'passed': 0, 'failed': 0,
                'constraints': [], 'failure_breakdown': {}, 'first_failure_stats': {},
                'skipped': True, 'skip_reason': 'No candidates to filter'
            }
            
        if not self.calculated_properties:
            return {
                'total': len(self.generated_compounds),
                'generated': self._generated_count,
                'predefined': len(self.generated_compounds) - self._generated_count,
                'passed': len(self.generated_compounds),
                'failed': 0,
                'constraints': list(self.property_constraints.keys()),
                'failure_breakdown': {},
                'first_failure_stats': {},
                'skipped': True,
                'skip_reason': 'No calculated properties available'
            }
            
        # Initialize statistics tracking
        original_count = len(self.generated_compounds)
        generated_count = self._generated_count
        predefined_count = original_count - generated_count
        
        filtered_candidates = []
        filtered_properties = {}
        failure_breakdown = {}
        
        # Track which constraint fails first for each candidate
        constraint_order = list(self.property_constraints.keys())
        first_failure_stats = {prop: 0 for prop in constraint_order}
        first_failure_stats['missing_properties'] = 0
        first_failure_stats['passed_all'] = 0
        
        for i, candidate in enumerate(self.generated_compounds):
            # Get stored properties for this candidate
            if i not in self.calculated_properties:
                failure_breakdown['no_properties'] = failure_breakdown.get('no_properties', 0) + 1
                first_failure_stats['missing_properties'] += 1
                continue
                
            properties = self.calculated_properties[i]
            
            # Check each property constraint in order
            passes_constraints = True
            first_failed_constraint = None
            
            for prop_name in constraint_order:
                bounds = self.property_constraints[prop_name]
                prop_value = properties.get(prop_name)
                
                if prop_value is None:
                    passes_constraints = False
                    failure_breakdown[f'{prop_name}_missing'] = failure_breakdown.get(f'{prop_name}_missing', 0) + 1
                    if first_failed_constraint is None:
                        first_failed_constraint = f'{prop_name}_missing'
                    break
                
                lower_bound, upper_bound = bounds
                if not (lower_bound <= prop_value <= upper_bound):
                    passes_constraints = False
                    failure_breakdown[prop_name] = failure_breakdown.get(prop_name, 0) + 1
                    if first_failed_constraint is None:
                        first_failed_constraint = prop_name
                    break
            
            # Record results
            if passes_constraints:
                first_failure_stats['passed_all'] += 1
                filtered_candidates.append(candidate)
                # Keep properties for filtered candidates with updated indices
                filtered_properties[len(filtered_candidates) - 1] = properties
            else:
                if first_failed_constraint:
                    if first_failed_constraint.endswith('_missing'):
                        prop_name = first_failed_constraint.replace('_missing', '')
                        first_failure_stats[prop_name] += 1
                    else:
                        first_failure_stats[first_failed_constraint] += 1
        
        # Update candidates and properties
        self.generated_compounds = filtered_candidates
        self.calculated_properties = filtered_properties
        
        # Return detailed statistics
        passed_count = len(filtered_candidates)
        failed_count = original_count - passed_count
        
        return {
            'total': original_count,
            'generated': generated_count,
            'predefined': predefined_count,
            'passed': passed_count,
            'failed': failed_count,
            'constraints': constraint_order,
            'constraint_details': {prop: bounds for prop, bounds in self.property_constraints.items()},
            'failure_breakdown': failure_breakdown,
            'first_failure_stats': first_failure_stats,
            'skipped': False,
            'skip_reason': None
        }
    
    def _is_compound_excluded(self, compound: Dict[str, int]) -> bool:
        """
        Check if a compound is identical to any system component (A or B).
        
        Args:
            compound: Compound to check
            
        Returns:
            True if compound should be excluded, False otherwise
        """
        if not self.system:
            return False
            
        # Check against compound A
        if 'A' in self.system:
            if self._compounds_equal(compound, self.system['A']):
                return True
                
        # Check against compound B  
        if 'B' in self.system:
            if self._compounds_equal(compound, self.system['B']):
                return True
                
        return False
    
    def _compounds_equal(self, comp1: Dict[str, int], comp2: Dict[str, int]) -> bool:
        """
        Check if two compounds are identical (same groups with same counts).
        
        Args:
            comp1: First compound
            comp2: Second compound
            
        Returns:
            True if compounds are identical, False otherwise
        """
        # Handle empty compounds
        if not comp1 and not comp2:
            return True
        if not comp1 or not comp2:
            return False
            
        # Check if all groups and counts match
        all_groups = set(comp1.keys()) | set(comp2.keys())
        for group in all_groups:
            count1 = comp1.get(group, 0)
            count2 = comp2.get(group, 0)
            if count1 != count2:
                return False
                
        return True
    
    def run(self):
        """Execute the CAMD screening calculation."""
        self.validate_system()
        
        print(f"Running CAMD screening: {self.title}")
        print(f"Method: {self.method}")
        print(f"Type: {self.type}")
        
        # TODO: Implement actual UNIFAC calculations
        self.results = {
            'status': 'completed',
            'method': self.method,
            'groups': self.groups,
            'compounds': self.compounds,
            'system': self.system
        }
        
        print("CAMD screening completed")
    
    def get_results(self):
        """Retrieve calculation results."""
        if self.results is None:
            raise RuntimeError("No results available. Run calculation first with run()")
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save results to file."""
        if self.results is None:
            raise RuntimeError("No results to save. Run calculation first.")
        
        if filename is None:
            filename = f"{self.title.replace(' ', '_')}_results.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {filepath}")