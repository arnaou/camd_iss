import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import pickle
import json


class CAMD:
    """
    Computer-Aided Molecular Design (CAMD) class for solvent screening.
    
    Supports VLE (Vapor-Liquid Equilibrium) calculations using UNIFAC-LY method.
    """
    
    def __init__(self, title: str = "", type: str = "VLE", output_dir: str = "./results", 
                 pure_method: str = "Constantinou-Gani", mixture_method: str = "UNIFAC-LY"):
        """
        Initialize CAMD instance.
        
        Args:
            title: Name associated with the CAMD problem
            type: Type of calculation - currently only "VLE" supported  
            output_dir: Directory for results output
            pure_method: Method for pure component properties (default: "Constantinou-Gani")
            mixture_method: Method for mixture properties (default: "UNIFAC-LY")
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
        
        self.groups = []
        self.compounds = {}
        self.system = {}
        self.available_groups = []
        self.design_groups = []  # Groups selected for compound generation
        self.structural_constraints = {}
        self.pure_properties = {}
        self.mixture_properties = {}
        self.generated_compounds = []
        self.results = None
        
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
        Validate a single compound using Odele and Macchietto rules.
        
        Args:
            compound_groups: Groups and their counts for this compound
            compound_index: Index of compound for error reporting
        """
        if not compound_groups:
            return
            
        # Rule 1: Octet rule (valency satisfaction)
        self._check_octet_rule(compound_groups, compound_index)
        
        # Rule 2: Connectivity rule  
        self._check_connectivity_rule(compound_groups, compound_index)
    
    def _check_octet_rule(self, compound_groups: Dict[str, int], compound_index: int):
        """
        Check octet rule: 2*m = sum((2 - noc_i) * x_i) where m is molecular type.
        For acyclic molecules: m = 1
        
        Args:
            compound_groups: Groups and their counts for this compound
            compound_index: Index of compound for error reporting
        """
        if 'noc' not in self.mixture_properties:
            print(f"Warning: Number of connections (noc) data not available, skipping octet rule check")
            return
            
        noc_data = self.mixture_properties['noc']
        total_valency = 0
        
        for group, count in compound_groups.items():
            if group in noc_data:
                noc = noc_data[group]
                total_valency += (2 - noc) * count
            else:
                raise ValueError(f"Number of connections not found for group '{group}' in compound {compound_index}")
        
        # For acyclic molecules (m=1), the sum should equal 2
        m = 1  # Assuming acyclic molecules for now
        expected_valency = 2 * m
        
        if total_valency != expected_valency:
            raise ValueError(f"Compound {compound_index} violates octet rule: "
                           f"valency sum = {total_valency}, expected = {expected_valency}")
    
    def _check_connectivity_rule(self, compound_groups: Dict[str, int], compound_index: int):
        """
        Check connectivity rule: sum(x_j where j != i) >= (noc_i - 2) * x_i + 2 for each group i.
        
        Args:
            compound_groups: Groups and their counts for this compound  
            compound_index: Index of compound for error reporting
        """
        if 'noc' not in self.mixture_properties:
            print(f"Warning: Number of connections (noc) data not available, skipping connectivity rule check")
            return
            
        noc_data = self.mixture_properties['noc']
        
        for group_i, count_i in compound_groups.items():
            if group_i not in noc_data:
                continue
                
            noc_i = noc_data[group_i]
            
            # Sum of all other groups
            sum_other_groups = sum(count for group, count in compound_groups.items() if group != group_i)
            
            # Required minimum connections
            min_required = (noc_i - 2) * count_i + 2
            
            if sum_other_groups < min_required:
                raise ValueError(f"Compound {compound_index} violates connectivity rule for group '{group_i}': "
                               f"other groups sum = {sum_other_groups}, minimum required = {min_required}")
    
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