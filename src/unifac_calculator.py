"""
UNIFAC Calculator

Calculates activity coefficients for liquid mixtures using the UNIFAC group contribution method
following Smith et al. (2001) formulation. This module integrates with the CAMD workflow
but can also be used as standalone functionality.

Example Usage:
    # Standalone usage
    calculator = UNIFACCalculator(missing_group_action="remove_compounds")
    mixture_composition = {
        'CH3': [2, 2],    # Component 1: 2 CH3, Component 2: 2 CH3
        'CH2': [1, 5],    # Component 1: 1 CH2, Component 2: 5 CH2
        'CH2NH': [1, 0]   # Component 1: 1 CH2NH, Component 2: 0 CH2NH
    }
    x = np.array([0.4, 0.6])  # Mole fractions
    T = 308.15  # Temperature in K
    gamma = calculator.calculate_activity_coefficients(mixture_composition, x, T)
    print(f"Activity coefficients: {gamma}")
    
    # Calculate for compound dictionaries
    compounds = [{'CH3': 2, 'CH2': 1, 'CH2NH': 1}, {'CH3': 2, 'CH2': 5}]
    gamma = calculator.calculate_from_compounds(compounds, x, T)
"""

import numpy as np
import json
import warnings
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import pyomo.environ as pyo


class UNIFACCalculator:
    """
    Calculator for UNIFAC activity coefficients using Smith et al. (2001) formulation.
    
    Calculates activity coefficients γ_i for liquid mixtures using the UNIFAC 
    group contribution method. Follows equations (A.1) through (A.16) from 
    Smith et al. (2001) Appendix A.

    Requires group composition data and interaction parameters.
    """
    
    def __init__(self, data_path: Optional[str] = None, missing_group_action: str = "remove_compounds", verbose: bool = False):
        """
        Initialize the UNIFAC activity coefficient calculator.
        
        Args:
            data_path: Path to data directory containing JSON files (defaults to ../data/json)
            missing_group_action: Action for compounds with missing group data:
                                "remove_compounds" (default) or "set_zero"
            verbose: Enable verbose output
        """
        if data_path is None:
            self.data_path = Path(__file__).parent.parent / "data" / "json"
        else:
            self.data_path = Path(data_path)
        self.missing_group_action = missing_group_action
        self.verbose = verbose
        self.missing_groups_summary = {}  # Track missing groups
        
        if missing_group_action not in ["remove_compounds", "set_zero"]:
            raise ValueError(f"missing_group_action must be 'remove_compounds' or 'set_zero', got '{missing_group_action}'")
        
        self.rk_data = None
        self.qk_data = None  
        self.interaction_data = None
        self._load_data()
    
    def _load_data(self):
        """Load UNIFAC parameters from JSON files."""
        try:
            # Load R/Q parameters
            with open(self.data_path / 'rk.json', 'r') as f:
                self.rk_data = json.load(f).get("data", {})
            with open(self.data_path / 'qk.json', 'r') as f:
                self.qk_data = json.load(f).get("data", {})
            
            # Load interaction parameters
            with open(self.data_path / 'a.json', 'r') as f:
                self.interaction_data = json.load(f).get("data", {})
                
            if self.verbose:
                print(f"Loaded UNIFAC data:")
                print(f"  R parameters: {len(self.rk_data)} groups")
                print(f"  Q parameters: {len(self.qk_data)} groups") 
                print(f"  Interaction parameters: {len(self.interaction_data)} groups")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load UNIFAC data from {self.data_path}: {e}")
    
    def _handle_missing_groups(self, compound: Dict[str, int], missing_groups: list) -> bool:
        """
        Handle missing group parameters according to the configured action.
        
        Args:
            compound: The compound being calculated
            missing_groups: List of missing groups
            
        Returns:
            bool: True if calculation should continue, False if compound should be removed
        """
        if not missing_groups:
            return True
            
        # Track missing groups for summary
        compound_str = str(compound)
        if compound_str not in self.missing_groups_summary:
            self.missing_groups_summary[compound_str] = set()
        self.missing_groups_summary[compound_str].update(missing_groups)
        
        print(f"Warning: Compound {compound} has missing group parameters: {missing_groups}")
        
        if self.missing_group_action == "remove_compounds":
            print(f"         Action: Removing compound from calculation")
            return False
        else:  # set_zero
            print(f"         Action: Setting missing group contributions to zero")
            return True
    
    def print_missing_groups_summary(self):
        """Print a summary of all missing groups encountered during calculations."""
        if not self.missing_groups_summary:
            print("No missing group parameters found for UNIFAC calculations.")
            return
            
        print("\n=== Missing Group Parameters Summary (UNIFAC Calculator) ===")
        for compound, missing_groups in self.missing_groups_summary.items():
            print(f"Compound {compound}: {sorted(missing_groups)}")
        print(f"Action taken: {self.missing_group_action}")
        print("=" * 65)
    
    def _handle_calculation_error(self, error: Exception, context: str = "") -> None:
        """Handle calculation errors with consistent logging."""
        if self.verbose:
            context_str = f" during {context}" if context else ""
            print(f"UNIFAC calculation failed{context_str}: {error}")
    
    def _has_required_groups(self, groups: list) -> tuple[bool, list]:
        """Check if all groups have required R/Q parameters and interaction data."""
        missing_groups = []
        for group in groups:
            if group not in self.rk_data or group not in self.qk_data:
                missing_groups.append(group)
        return len(missing_groups) == 0, missing_groups
    
    def get_interaction_parameter(self, group_i: str, group_j: str) -> float:
        """Get interaction parameter a_mk between groups."""
        if self.interaction_data and group_i in self.interaction_data and group_j in self.interaction_data[group_i]:
            a_mk = self.interaction_data[group_i][group_j]
        elif self.interaction_data and group_j in self.interaction_data and group_i in self.interaction_data[group_j]:
            a_mk = self.interaction_data[group_j][group_i]
        else:
            if self.verbose:
                warnings.warn(f"Missing interaction parameter for {group_i}-{group_j}, using 0.0")
            a_mk = 0.0
        return a_mk if a_mk is not None else 0.0
    
    def calculate_activity_coefficients(self, mixture_composition: Dict[str, List[float]], 
                                      x: np.ndarray, T: float) -> Optional[np.ndarray]:
        """
        Calculate activity coefficients using Smith et al. (2001) UNIFAC method.
        
        Args:
            mixture_composition: Group composition matrix {group: [n_i_k for each component]}
            x: Component mole fractions
            T: Temperature in Kelvin
            
        Returns:
            gamma: Activity coefficients array or None if calculation fails
        """
        try:
            # Check for required group parameters
            groups = list(mixture_composition.keys())
            has_groups, missing_groups = self._has_required_groups(groups)
            
            if not has_groups:
                print(f"Missing group parameters for UNIFAC calculation: {missing_groups}")
                return None
            
            # Setup - extract groups and compositions 
            v = np.array([mixture_composition[group] for group in groups])  # Shape: (n_groups, n_components)
            
            # Get group parameters
            Rk = np.array([self.rk_data[group] for group in groups])
            Qk = np.array([self.qk_data[group] for group in groups])
            
            # Build interaction parameter matrix
            n_groups = len(groups)
            a = np.zeros((n_groups, n_groups))
            for i, group_i in enumerate(groups):
                for j, group_j in enumerate(groups):
                    a[i, j] = self.get_interaction_parameter(group_i, group_j)
            
            # Begin UNIFAC calculations following Smith et al. (2001)
            
            # Molecular parameters (Equations A.3, A.4)
            r = np.dot(v.T, Rk)  # r_i = Σ n_{i,k} R_k, i ∈ C^K (A.3)
            q = np.dot(v.T, Qk)  # q_i = Σ n_{i,k} Q_k, i ∈ C^K (A.4)
            
            # Combinatorial term (Equations A.2, A.9, A.10)
            L = np.matrix(q) / np.matrix(np.dot(x, q)).T  # L_{i,t} = q_i / Σ x_τ q_τ (A.10)
            J = np.matrix(r) / np.matrix(np.dot(x, r)).T  # J_{i,t} = r_i / Σ x_τ r_τ (A.9)
            
            # ln γ^C_{i,t} = 1 - J_{i,t} + ln J_{i,t} - 5q_i(1 - J_{i,t}/L_{i,t} + ln J_{i,t}/L_{i,t}) (A.2)
            gamma_c = 1 - J + np.log(J) - 5 * np.multiply(q, (1 - J / L + np.log(J / L)))
            
            # Residual term (Equations A.11-A.16) - vectorized numpy operations
            tau = np.exp(-a / T)  # ψ_{m,k,t} = exp(-a_{m,k}/T_t), m ∈ K; k ∈ K; t ∈ T (A.16)
            
            # Calculate e_{i,k} = n_{i,k}Q_k / q_i for all components and groups (A.12)
            # Broadcasting: v.T shape (n_comp, n_groups) * Qk shape (n_groups,) / q shape (n_comp, 1)
            e = (v.T * Qk) / q[:, np.newaxis]
            
            # Calculate β_{i,k,t} for all components and groups (A.13)
            # β_{i,k,t} = Σ_{m∈K} n_{i,m} Q_m ψ_{m,k,t} / q_i
            vQk = v.T * Qk  # Component-group contributions: (n_comp, n_groups)
            beta = np.dot(vQk, tau) / q[:, np.newaxis]  # Matrix multiplication + broadcasting
            
            # Calculate θ_{k,t} - group surface area fractions in mixture (A.14)
            theta = np.dot(x, v.T * Qk) / np.dot(x, q)
            
            # Calculate ω_{k,t} = Σ θ_{m,t} ψ_{m,k,t} (A.15)
            omega = np.dot(theta, tau)
            
            # ln γ^R_{i,t} = q_i [1 - Σ_{k∈K} (θ_{k,t} β_{i,k,t}/ω_{k,t} - e_{i,k} ln β_{i,k,t}/ω_{k,t})] (A.11)
            gamma_r = q * (1 - np.sum(theta * beta / omega - e * np.log(beta / omega), axis=1))
            
            # Total activity coefficient ln γ_{i,t} = ln γ^C_{i,t} + ln γ^R_{i,t} (A.1)
            gamma = np.array(np.exp(gamma_c + gamma_r)).flatten()
            
            return gamma
            
        except Exception as e:
            self._handle_calculation_error(e, "activity coefficient calculation")
            return None
    
    def calculate_from_compounds(self, compounds: List[Dict[str, int]], 
                               x: np.ndarray, T: float) -> Optional[np.ndarray]:
        """
        Calculate activity coefficients from a list of compound dictionaries.
        
        Args:
            compounds: List of compound dictionaries with group names as keys and counts as values
            x: Component mole fractions
            T: Temperature in Kelvin
            
        Returns:
            gamma: Activity coefficients array or None if calculation fails
            
        Example:
            compounds = [
                {'CH3': 2, 'CH2': 1, 'CH2NH': 1},  # Diethylamine
                {'CH3': 2, 'CH2': 5}                # Heptane
            ]
            x = np.array([0.4, 0.6])
            gamma = calculator.calculate_from_compounds(compounds, x, 308.15)
        """
        # Validate inputs
        if len(compounds) != len(x):
            raise ValueError(f"Number of compounds ({len(compounds)}) must match number of mole fractions ({len(x)})")
        
        if not np.isclose(np.sum(x), 1.0, atol=1e-6):
            raise ValueError(f"Mole fractions must sum to 1.0, got {np.sum(x):.6f}")
        
        # Extract all unique groups from compounds
        all_groups = set()
        for compound in compounds:
            all_groups.update(compound.keys())
        all_groups = sorted(all_groups)
        
        # Check for missing group parameters
        valid_compounds = []
        valid_x = []
        
        for i, compound in enumerate(compounds):
            compound_groups = list(compound.keys())
            has_groups, missing_groups = self._has_required_groups(compound_groups)
            
            if not has_groups and not self._handle_missing_groups(compound, missing_groups):
                continue  # Skip this compound
            
            valid_compounds.append(compound)
            valid_x.append(x[i])
        
        if not valid_compounds:
            print("No valid compounds remaining after handling missing groups")
            return None
        
        # Renormalize mole fractions if compounds were removed
        valid_x = np.array(valid_x)
        if len(valid_x) < len(x):
            valid_x = valid_x / np.sum(valid_x)
            print(f"Renormalized mole fractions after removing invalid compounds: {valid_x}")
        
        # Build mixture composition matrix
        mixture_composition = {}
        for group in all_groups:
            mixture_composition[group] = []
            for compound in valid_compounds:
                mixture_composition[group].append(compound.get(group, 0))
        
        return self.calculate_activity_coefficients(mixture_composition, valid_x, T)
    
    def create_pyomo_unifac_model(self, mixture_composition: Dict[str, List[float]], 
                                  x: np.ndarray, T: float, 
                                  model_name: str = "UNIFAC_Model") -> Optional[pyo.ConcreteModel]:
        """
        Create a Pyomo model for UNIFAC activity coefficient calculations.
        
        This creates a symbolic representation of the UNIFAC equations that can be used
        in optimization problems or for automatic differentiation.
        
        Args:
            mixture_composition: Group composition matrix {group: [n_i_k for each component]}
            x: Component mole fractions (initial values)
            T: Temperature in Kelvin
            model_name: Name for the Pyomo model
            
        Returns:
            Pyomo ConcreteModel with UNIFAC equations or None if setup fails
            
        Example:
            model = calculator.create_pyomo_unifac_model(mixture_comp, x, T)
            if model:
                # Access activity coefficients
                gamma_values = [pyo.value(model.gamma[i]) for i in model.I]
                print(f"Activity coefficients: {gamma_values}")
        """
        try:
            # Check for required group parameters
            groups = list(mixture_composition.keys())
            has_groups, missing_groups = self._has_required_groups(groups)
            
            if not has_groups:
                print(f"Missing group parameters for Pyomo UNIFAC model: {missing_groups}")
                return None
            
            # Create Pyomo model
            model = pyo.ConcreteModel(model_name)
            
            # Sets
            model.I = pyo.Set(initialize=range(len(x)), doc="Component indices")
            model.K = pyo.Set(initialize=range(len(groups)), doc="Group indices")
            
            # Parameters - Group data
            Rk_data = {k: self.rk_data[groups[k]] for k in range(len(groups))}
            Qk_data = {k: self.qk_data[groups[k]] for k in range(len(groups))}
            model.Rk = pyo.Param(model.K, initialize=Rk_data, doc="Group van der Waals volumes")
            model.Qk = pyo.Param(model.K, initialize=Qk_data, doc="Group surface areas")
            
            # Parameters - Composition matrix v[i,k] = number of groups k in component i
            v_data = {}
            for i in range(len(x)):
                for k, group in enumerate(groups):
                    v_data[(i, k)] = mixture_composition[group][i]
            model.v = pyo.Param(model.I, model.K, initialize=v_data, 
                               doc="Number of groups k in component i")
            
            # Parameters - Interaction parameters a[k,m]
            a_data = {}
            for k in range(len(groups)):
                for m in range(len(groups)):
                    a_data[(k, m)] = self.get_interaction_parameter(groups[k], groups[m])
            model.a = pyo.Param(model.K, model.K, initialize=a_data, 
                               doc="UNIFAC interaction parameters")
            
            # Parameters - Temperature and mole fractions
            model.T = pyo.Param(initialize=T, doc="Temperature [K]")
            model.x = pyo.Param(model.I, initialize={i: x[i] for i in range(len(x))}, 
                               mutable=True, doc="Component mole fractions")
            
            # Calculate initial values to avoid division by zero
            v_array = np.array([mixture_composition[group] for group in groups])
            Rk_array = np.array([self.rk_data[groups[k]] for k in range(len(groups))])
            Qk_array = np.array([self.qk_data[groups[k]] for k in range(len(groups))])
            
            r_init = np.dot(v_array.T, Rk_array)
            q_init = np.dot(v_array.T, Qk_array)
            
            # Variables - Molecular parameters with proper initialization
            model.r = pyo.Var(model.I, domain=pyo.PositiveReals, 
                             initialize={i: r_init[i] for i in range(len(x))},
                             doc="Component van der Waals volumes")
            model.q = pyo.Var(model.I, domain=pyo.PositiveReals, 
                             initialize={i: q_init[i] for i in range(len(x))},
                             doc="Component surface areas")
            
            # Variables - Combinatorial terms
            model.L = pyo.Var(model.I, domain=pyo.PositiveReals, doc="L_i term")
            model.J = pyo.Var(model.I, domain=pyo.PositiveReals, doc="J_i term")
            model.gamma_c = pyo.Var(model.I, doc="Combinatorial activity coefficient (ln)")
            
            # Variables - Residual terms
            model.tau = pyo.Var(model.K, model.K, domain=pyo.PositiveReals, 
                               doc="Interaction parameter tau[k,m]")
            model.e = pyo.Var(model.I, model.K, domain=pyo.NonNegativeReals, 
                             doc="Group surface fraction e[i,k]")
            model.beta = pyo.Var(model.I, model.K, domain=pyo.PositiveReals, 
                                doc="Beta parameter beta[i,k]")
            model.theta = pyo.Var(model.K, domain=pyo.NonNegativeReals, 
                                 doc="Group surface area fraction theta[k]")
            model.omega = pyo.Var(model.K, domain=pyo.PositiveReals, 
                                 doc="Omega parameter omega[k]")
            model.gamma_r = pyo.Var(model.I, doc="Residual activity coefficient (ln)")
            
            # Final activity coefficients
            model.ln_gamma = pyo.Var(model.I, doc="Total activity coefficient (ln)")
            model.gamma = pyo.Var(model.I, domain=pyo.PositiveReals, 
                                 doc="Total activity coefficient")
            
            # Constraints - Molecular parameters (Equations A.3, A.4)
            def r_rule(model, i):
                return model.r[i] == sum(model.v[i, k] * model.Rk[k] for k in model.K)
            model.r_constraint = pyo.Constraint(model.I, rule=r_rule)
            
            def q_rule(model, i):
                return model.q[i] == sum(model.v[i, k] * model.Qk[k] for k in model.K)
            model.q_constraint = pyo.Constraint(model.I, rule=q_rule)
            
            # Constraints - Combinatorial terms (Equations A.9, A.10)
            def J_rule(model, i):
                return model.J[i] == model.r[i] / sum(model.x[j] * model.r[j] for j in model.I)
            model.J_constraint = pyo.Constraint(model.I, rule=J_rule)
            
            def L_rule(model, i):
                return model.L[i] == model.q[i] / sum(model.x[j] * model.q[j] for j in model.I)
            model.L_constraint = pyo.Constraint(model.I, rule=L_rule)
            
            # Combinatorial activity coefficient (Equation A.2)
            def gamma_c_rule(model, i):
                return model.gamma_c[i] == (1 - model.J[i] + pyo.log(model.J[i]) - 
                                           5 * model.q[i] * (1 - model.J[i]/model.L[i] + 
                                                            pyo.log(model.J[i]/model.L[i])))
            model.gamma_c_constraint = pyo.Constraint(model.I, rule=gamma_c_rule)
            
            # Constraints - Residual terms
            # Interaction parameter tau (Equation A.16)
            def tau_rule(model, k, m):
                return model.tau[k, m] == pyo.exp(-model.a[k, m] / model.T)
            model.tau_constraint = pyo.Constraint(model.K, model.K, rule=tau_rule)
            
            # Group surface fraction e (Equation A.12)
            def e_rule(model, i, k):
                return model.e[i, k] == model.v[i, k] * model.Qk[k] / model.q[i]
            model.e_constraint = pyo.Constraint(model.I, model.K, rule=e_rule)
            
            # Beta parameter (Equation A.13)
            def beta_rule(model, i, k):
                return model.beta[i, k] == sum(model.v[i, m] * model.Qk[m] * model.tau[m, k] 
                                             for m in model.K) / model.q[i]
            model.beta_constraint = pyo.Constraint(model.I, model.K, rule=beta_rule)
            
            # Group surface area fraction theta (Equation A.14)
            def theta_rule(model, k):
                return model.theta[k] == (sum(model.x[i] * model.v[i, k] * model.Qk[k] 
                                            for i in model.I) / 
                                        sum(model.x[i] * model.q[i] for i in model.I))
            model.theta_constraint = pyo.Constraint(model.K, rule=theta_rule)
            
            # Omega parameter (Equation A.15)
            def omega_rule(model, k):
                return model.omega[k] == sum(model.theta[m] * model.tau[m, k] for m in model.K)
            model.omega_constraint = pyo.Constraint(model.K, rule=omega_rule)
            
            # Residual activity coefficient (Equation A.11)
            def gamma_r_rule(model, i):
                return model.gamma_r[i] == model.q[i] * (1 - sum(model.theta[k] * model.beta[i, k] / model.omega[k] - 
                                                                model.e[i, k] * pyo.log(model.beta[i, k] / model.omega[k]) 
                                                                for k in model.K))
            model.gamma_r_constraint = pyo.Constraint(model.I, rule=gamma_r_rule)
            
            # Total activity coefficient (Equation A.1)
            def ln_gamma_rule(model, i):
                return model.ln_gamma[i] == model.gamma_c[i] + model.gamma_r[i]
            model.ln_gamma_constraint = pyo.Constraint(model.I, rule=ln_gamma_rule)
            
            def gamma_rule(model, i):
                return model.gamma[i] == pyo.exp(model.ln_gamma[i])
            model.gamma_constraint = pyo.Constraint(model.I, rule=gamma_rule)
            
            if self.verbose:
                print(f"Created Pyomo UNIFAC model '{model_name}' with:")
                print(f"  Components: {len(x)}")
                print(f"  Groups: {len(groups)} ({groups})")
                print(f"  Variables: {len(list(model.component_objects(pyo.Var)))}")
                print(f"  Constraints: {len(list(model.component_objects(pyo.Constraint)))}")
            
            return model
            
        except Exception as e:
            print(f"Error creating Pyomo UNIFAC model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def solve_pyomo_unifac_model(self, model: pyo.ConcreteModel, 
                                solver_name: str = "ipopt") -> Optional[Dict[str, float]]:
        """
        Solve the Pyomo UNIFAC model to calculate activity coefficients.
        
        Args:
            model: Pyomo UNIFAC model from create_pyomo_unifac_model()
            solver_name: Solver to use (default: "ipopt")
            
        Returns:
            Dict with solution results or None if solve fails
        """
        try:
            # Create solver
            opt = pyo.SolverFactory(solver_name)
            
            if solver_name.lower() == "ipopt":
                opt.options["halt_on_ampl_error"] = "yes"
                opt.options["max_iter"] = 10000
                opt.options["tol"] = 1e-8
            
            # Solve model
            results = opt.solve(model, tee=self.verbose)
            
            # Check solver status
            if results.solver.status != pyo.SolverStatus.ok:
                print(f"Solver failed with status: {results.solver.status}")
                return None
            
            if results.solver.termination_condition != pyo.TerminationCondition.optimal:
                print(f"Solver did not find optimal solution: {results.solver.termination_condition}")
                return None
            
            # Extract results
            solution = {
                'solver_status': str(results.solver.status),
                'termination_condition': str(results.solver.termination_condition),
                'gamma': [pyo.value(model.gamma[i]) for i in model.I],
                'ln_gamma': [pyo.value(model.ln_gamma[i]) for i in model.I],
                'gamma_c': [pyo.value(model.gamma_c[i]) for i in model.I],
                'gamma_r': [pyo.value(model.gamma_r[i]) for i in model.I],
                'r': [pyo.value(model.r[i]) for i in model.I],
                'q': [pyo.value(model.q[i]) for i in model.I]
            }
            
            if self.verbose:
                print(f"Pyomo UNIFAC solve completed successfully")
                print(f"Activity coefficients: {solution['gamma']}")
            
            return solution
            
        except Exception as e:
            print(f"Error solving Pyomo UNIFAC model: {e}")
            import traceback
            traceback.print_exc()
            return None
    


# Convenience function for quick calculations  
def calculate_activity_coefficients(mixture_composition: Dict[str, List[float]], 
                                  x: np.ndarray, T: float,
                                  missing_group_action: str = "remove_compounds") -> Optional[np.ndarray]:
    """
    Convenience function to calculate activity coefficients for a mixture.
    
    Args:
        mixture_composition: Group composition matrix {group: [n_i_k for each component]}
        x: Component mole fractions
        T: Temperature in Kelvin
        missing_group_action: Action for missing group parameters
        
    Returns:
        Array of activity coefficients or None if calculation fails
        
    Example:
        mixture_composition = {
            'CH3': [2, 2],
            'CH2': [1, 5],
            'CH2NH': [1, 0]
        }
        x = np.array([0.4, 0.6])
        gamma = calculate_activity_coefficients(mixture_composition, x, 308.15)
    """
    calculator = UNIFACCalculator(missing_group_action=missing_group_action)
    return calculator.calculate_activity_coefficients(mixture_composition, x, T)


def calculate_from_compounds(compounds: List[Dict[str, int]], 
                           x: np.ndarray, T: float,
                           missing_group_action: str = "remove_compounds") -> Optional[np.ndarray]:
    """
    Convenience function to calculate activity coefficients from compound dictionaries.
    
    Args:
        compounds: List of compound dictionaries
        x: Component mole fractions
        T: Temperature in Kelvin
        missing_group_action: Action for missing group parameters
        
    Returns:
        Array of activity coefficients or None if calculation fails
        
    Example:
        compounds = [{'CH3': 2, 'CH2': 1, 'CH2NH': 1}, {'CH3': 2, 'CH2': 5}]
        x = np.array([0.4, 0.6])
        gamma = calculate_from_compounds(compounds, x, 308.15)
    """
    calculator = UNIFACCalculator(missing_group_action=missing_group_action)
    return calculator.calculate_from_compounds(compounds, x, T)


if __name__ == "__main__":
    # Example usage
    print("=== UNIFAC Calculator Example ===")
    
    # Test compounds
    test_compounds = [
        {'CH3': 2, 'CH2': 1, 'CH2NH': 1},  # Diethylamine: (C2H5)2NH
        {'CH3': 2, 'CH2': 5}                # Heptane: C7H16
    ]
    
    component_names = ['Diethylamine', 'Heptane']
    x = np.array([0.4, 0.6])  # 40% diethylamine, 60% heptane
    T = 308.15  # Temperature in K (35°C)
    
    print(f"Mixture: {component_names[0]}({x[0]*100:.0f}%) + {component_names[1]}({x[1]*100:.0f}%)")
    print(f"Temperature: {T:.2f} K ({T-273.15:.1f}°C)")
    print(f"Compounds:")
    for i, (name, compound) in enumerate(zip(component_names, test_compounds)):
        comp_text = " + ".join([f"{count} {group}" for group, count in compound.items()])
        print(f"  {name}: {comp_text}")
    
    # Initialize calculator
    calculator = UNIFACCalculator(verbose=True)
    
    try:
        # Method 1: Direct numpy calculation
        print("\n=== Method 1: Direct NumPy Calculation ===")
        gamma_numpy = calculator.calculate_from_compounds(test_compounds, x, T)
        
        if gamma_numpy is not None:
            print(f"Activity coefficients (NumPy):")
            for i, (comp, g) in enumerate(zip(component_names, gamma_numpy)):
                print(f"  γ_{i+1} ({comp}) = {g:.4f}")
        else:
            print("  NumPy calculation failed")
        
        # Method 2: Pyomo model calculation
        print("\n=== Method 2: Pyomo Model Calculation ===")
        
        # Create mixture composition matrix
        mixture_composition = {
            'CH3': [2.0, 2.0],
            'CH2': [1.0, 5.0],
            'CH2NH': [1.0, 0.0]
        }
        
        # Create Pyomo model
        model = calculator.create_pyomo_unifac_model(mixture_composition, x, T, "UNIFAC_Example")
        
        if model is not None:
            # Solve the model
            solution = calculator.solve_pyomo_unifac_model(model)
            
            if solution is not None:
                gamma_pyomo = solution['gamma']
                print(f"Activity coefficients (Pyomo):")
                for i, (comp, g) in enumerate(zip(component_names, gamma_pyomo)):
                    print(f"  γ_{i+1} ({comp}) = {g:.4f}")
                
                # Compare methods
                if gamma_numpy is not None:
                    print(f"\nComparison (NumPy vs Pyomo):")
                    for i, comp in enumerate(component_names):
                        diff = abs(gamma_numpy[i] - gamma_pyomo[i])
                        print(f"  {comp}: Δγ = {diff:.6f}")
                
                # Show additional Pyomo results
                print(f"\nDetailed Pyomo Results:")
                print(f"  Solver: {solution['solver_status']}")
                print(f"  Termination: {solution['termination_condition']}")
                print(f"  Combinatorial γ^C: {[f'{g:.4f}' for g in solution['gamma_c']]}")
                print(f"  Residual γ^R: {[f'{g:.4f}' for g in solution['gamma_r']]}")
                print(f"  r values: {[f'{r:.4f}' for r in solution['r']]}")
                print(f"  q values: {[f'{q:.4f}' for q in solution['q']]}")
            else:
                print("  Pyomo model solve failed")
        else:
            print("  Pyomo model creation failed")
            
    except Exception as e:
        print(f"Calculation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    calculator.print_missing_groups_summary()