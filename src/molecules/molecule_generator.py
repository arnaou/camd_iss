"""
Molecule generation using Mixed-Integer Linear Programming (MILP).

This module provides standalone functionality for generating feasible molecular
candidates using MILP optimization with structural and chemical feasibility constraints.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from .structural_validator import StructuralValidator


class MoleculeGenerator:
    """
    Generate molecular candidates using MILP optimization.
    
    This class implements the Odele-Macchietto rules for structural feasibility
    and uses Gurobi to solve the MILP problem for generating candidate molecules.
    """
    
    def __init__(self, load_data: bool = True):
        """
        Initialize the molecule generator.
        
        Args:
            load_data: Whether to automatically load required data (noc, etc.)
        """
        self.design_groups = []
        self.structural_constraints = {}
        self.noc_data = {}
        self.validator = StructuralValidator()
        
        if load_data:
            self._load_molecular_data()
    
    def _load_molecular_data(self):
        """Load molecular connectivity data (noc) required for constraints."""
        try:
            # Try multiple possible paths
            possible_paths = [
                Path(__file__).parent.parent / "data" / "json" / 'noc.json',  # From src/molecules/
                Path.cwd() / "data" / "json" / 'noc.json',                   # From current directory
                Path.cwd().parent / "data" / "json" / 'noc.json',            # From scripts/
            ]
            
            noc_path = None
            for path in possible_paths:
                if path.exists():
                    noc_path = path
                    break
            
            if noc_path is None:
                raise FileNotFoundError("noc.json not found in any expected location")
            
            # Load noc (number of connections) data
            with open(noc_path, 'r') as f:
                noc_file = json.load(f)
                self.noc_data = noc_file.get("data", {})
            
            print(f"Loaded noc data for {len(self.noc_data)} groups from {noc_path}")
            
        except FileNotFoundError as e:
            print(f"Warning: Could not load molecular data: {e}")
            print("Some constraints may be skipped")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in molecular data: {e}")
    
    def set_design_groups(self, groups: List[str]):
        """
        Set the groups available for molecular design.
        
        Args:
            groups: List of group names to use in molecule generation
        """
        self.design_groups = groups[:]
        print(f"Set design groups: {len(groups)} groups")
    
    def set_structural_constraints(self,
                                 min_groups: int = 2,
                                 max_groups: int = 10,
                                 min_func_groups: int = 1,
                                 max_func_groups: int = 3,
                                 molecular_type: int = 1):
        """
        Set structural constraints for molecule generation.
        
        Args:
            min_groups: Minimum total number of groups
            max_groups: Maximum total number of groups
            min_func_groups: Minimum number of functional groups
            max_func_groups: Maximum number of functional groups  
            molecular_type: Molecular type (1=acyclic, 0=monocyclic, -1=bicyclic)
        """
        self.structural_constraints = {
            'min_groups': min_groups,
            'max_groups': max_groups,
            'min_func_groups': min_func_groups,
            'max_func_groups': max_func_groups,
            'molecular_type': molecular_type
        }
        
        molecular_type_names = {-1: 'bicyclic', 0: 'monocyclic', 1: 'acyclic'}
        print(f"Set structural constraints:")
        print(f"  Groups: {min_groups}-{max_groups}")
        print(f"  Functional groups: {min_func_groups}-{max_func_groups}")
        print(f"  Molecular type: {molecular_type} ({molecular_type_names[molecular_type]})")
    
    def generate_candidates(self, 
                          max_solutions: int = 10000,
                          verbose: bool = False) -> List[Dict[str, int]]:
        """
        Generate all feasible compounds using Mixed-Integer Linear Programming (MILP).
        
        Uses Gurobi to solve the MILP optimization problem with Odele-Macchietto
        structural feasibility rules and user-defined constraints.
        
        Args:
            max_solutions: Maximum number of solutions to generate
            verbose: Print detailed solver information
            
        Returns:
            List of generated compound dictionaries
            
        Raises:
            ValueError: If design groups or constraints not set
            ImportError: If Gurobi is not available
        """
        if not self.design_groups:
            raise ValueError("No design groups set. Use set_design_groups() first.")
        
        if not self.structural_constraints:
            raise ValueError("No structural constraints set. Use set_structural_constraints() first.")
        
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
        print(f"Design groups: {len(self.design_groups)} groups")
        print(f"Molecular type: {molecular_type} ({molecular_type_names[molecular_type]})")
        print(f"Functional groups: {min_func_groups}-{max_func_groups}")
        
        # Filter out groups with noc=0 (complete molecules) from MILP
        milp_groups = []
        if self.noc_data:
            for group in self.design_groups:
                if group in self.noc_data and self.noc_data[group] == 0:
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
        # Use 0 as lower bound and max_groups as upper bound for individual groups
        x = model.addVars(groups, 
                         lb=0, 
                         ub=self.structural_constraints['max_groups'], 
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
        if self.noc_data:
            model.addConstr(
                gp.quicksum((2 - self.noc_data.get(j, 2)) * x[j] for j in groups 
                           if j in self.noc_data) == 2 * molecular_type,
                name="octet rule"
            )
        else:
            print("Warning: No noc data available, skipping octet rule constraint")
        
        # Add constraint: Connectivity rule (Odele-Macchietto Rule 2)
        # sum(x_i where i != j) >= (noc_j - 2) * x_j + 2
        if self.noc_data:
            model.addConstrs(
                gp.quicksum(x[i] for i in groups if i != j) >= 
                (self.noc_data.get(j, 2) - 2) * x[j] + 2 
                for j in groups if j in self.noc_data
            )
        
        # Add constraint: functional groups limits 
        # Assumes CH3, CH2, CH, C are alkyl groups (non-functional)
        alkyl_groups = ['CH3', 'CH2', 'CH', 'C']
        alkyl_in_design = [g for g in alkyl_groups if g in groups]
        
        if alkyl_in_design:
            try:
                # Functional groups = Total groups - Alkyl groups
                alkyl_sum = gp.quicksum(x[g] for g in alkyl_in_design)
                
                model.addConstr(
                    x.sum() - alkyl_sum >= min_func_groups,
                    name="minimum nr. of functional groups"
                )
                model.addConstr(
                    x.sum() - alkyl_sum <= max_func_groups,
                    name="maximum nr. of functional groups"
                )
            except Exception as e:
                print(f"Warning: Could not add functional group constraints: {e}")
        else:
            print("Warning: No alkyl groups in design groups. Skipping functional group constraints.")
        
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
        
        print(f"Generated {len(compounds)} unique compounds")
        return compounds
    
    def validate_candidates(self, 
                          candidates: List[Dict[str, int]],
                          exclude_compounds: Optional[List[Dict[str, int]]] = None,
                          verbose: bool = True) -> List[Dict[str, int]]:
        """
        Validate generated candidates for structural feasibility.
        
        Args:
            candidates: List of candidate compound dictionaries
            exclude_compounds: List of compounds to exclude (e.g., A, B from binary system)
            verbose: Print validation information
            
        Returns:
            List of valid compounds after filtering
        """
        if not candidates:
            return []
        
        if verbose:
            print(f"Validating {len(candidates)} candidates...")
        
        valid_candidates = []
        excluded_count = 0
        invalid_count = 0
        
        for i, candidate in enumerate(candidates):
            # Check structural validity
            try:
                self._validate_single_compound(candidate, i)
            except ValueError as e:
                if verbose:
                    print(f"  Candidate {i} invalid: {e}")
                invalid_count += 1
                continue
            
            # Check if compound should be excluded
            if exclude_compounds and self._is_compound_excluded(candidate, exclude_compounds):
                if verbose:
                    print(f"  Candidate {i} excluded (identical to system component)")
                excluded_count += 1
                continue
            
            valid_candidates.append(candidate)
        
        if verbose:
            print(f"Validation results:")
            print(f"  Valid compounds: {len(valid_candidates)}")
            print(f"  Invalid compounds: {invalid_count}")
            print(f"  Excluded compounds: {excluded_count}")
        
        return valid_candidates
    
    def _validate_single_compound(self, compound_groups: Dict[str, int], compound_index: int):
        """
        Validate a single compound using StructuralValidator.
        
        Args:
            compound_groups: Groups and their counts for this compound
            compound_index: Index of compound for error reporting
            
        Raises:
            ValueError: If compound violates structural rules
        """
        if not compound_groups:
            return
        
        # Use StructuralValidator to check structural feasibility
        is_valid = self.validator.is_valid(compound_groups)
        if not is_valid:
            raise ValueError(f"Compound {compound_index} violates structural feasibility rules")
    
    def _is_compound_excluded(self, 
                            compound: Dict[str, int], 
                            exclude_compounds: List[Dict[str, int]]) -> bool:
        """
        Check if a compound is identical to any in the exclusion list.
        
        Args:
            compound: Compound to check
            exclude_compounds: List of compounds to exclude
            
        Returns:
            True if compound should be excluded, False otherwise
        """
        for exclude_comp in exclude_compounds:
            if self._compounds_equal(compound, exclude_comp):
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


def generate_molecules(design_groups: List[str],
                      min_groups: int = 2,
                      max_groups: int = 10,
                      min_func_groups: int = 1,
                      max_func_groups: int = 3,
                      molecular_type: int = 1,
                      max_solutions: int = 10000,
                      exclude_compounds: Optional[List[Dict[str, int]]] = None,
                      verbose: bool = True) -> List[Dict[str, int]]:
    """
    Convenience function to generate molecules with default settings.
    
    Args:
        design_groups: List of group names to use in molecule generation
        min_groups: Minimum total number of groups
        max_groups: Maximum total number of groups  
        min_func_groups: Minimum number of functional groups
        max_func_groups: Maximum number of functional groups
        molecular_type: Molecular type (1=acyclic, 0=monocyclic, -1=bicyclic)
        max_solutions: Maximum number of solutions to generate
        exclude_compounds: List of compounds to exclude from results
        verbose: Print generation information
        
    Returns:
        List of valid compound dictionaries
    """
    generator = MoleculeGenerator()
    
    generator.set_design_groups(design_groups)
    generator.set_structural_constraints(
        min_groups=min_groups,
        max_groups=max_groups,
        min_func_groups=min_func_groups,
        max_func_groups=max_func_groups,
        molecular_type=molecular_type
    )
    
    # Generate candidates
    candidates = generator.generate_candidates(max_solutions=max_solutions, verbose=verbose)
    
    # Add complete molecules (noc=0) to the top of the list
    complete_molecules = []
    if generator.noc_data:
        for group in design_groups:
            if group in generator.noc_data and generator.noc_data[group] == 0:
                complete_molecules.append({group: 1})
    
    if complete_molecules:
        if verbose:
            print(f"Adding {len(complete_molecules)} complete molecules to candidates")
        candidates = complete_molecules + candidates
    
    # Filter exclusions only (MILP should ensure structural validity)
    if candidates and exclude_compounds:
        original_count = len(candidates)
        candidates = [c for c in candidates if not generator._is_compound_excluded(c, exclude_compounds)]
        excluded_count = original_count - len(candidates)
        if verbose and excluded_count > 0:
            print(f"Excluded {excluded_count} candidates identical to system components")
    
    return candidates