"""
Antoine Parameter Calculator for CAMD

Calculates Antoine equation parameters (A, B, C) using pure component critical properties.
This module integrates with the CAMD workflow to determine vapor pressure parameters
needed for VLE calculations.

Example Usage:
    # Standalone usage
    calculator = AntoineCalculator(missing_group_action="remove_compounds")
    compound = {'CH3': 2, 'OH': 1}  # Ethanol
    properties = {'tc': 513.9, 'pc': 6.137e6, 'ac': 0.645, 'tb': 351.4}  # Critical props from PropertyCalculator
    antoine_params = calculator.calculate_antoine_parameters(compound, properties)
    print(f"Antoine A: {antoine_params['A']:.3f}")
    
    # Individual parameter calculation
    params = calculator.solve_antoine(tb=351.4, tc=513.9, pc=6.137, p0=0.101325, w=0.645)
    print(f"A: {params['A']:.3f}, B: {params['B']:.3f}, C: {params['C']:.3f}")
"""

import numpy as np
import pyomo.environ as pyo
from pathlib import Path
from typing import Dict, Optional, Union
import sys
import os

# Add utils to path for pyomo utilities
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from pyom import parse_pyomo_log, ensure_file_exists_in_cwd


class AntoineCalculator:
    """
    Calculator for Antoine equation parameters using critical properties.
    
    Calculates Antoine parameters A, B, C from the equation:
    log10(P*) = A - B / (C + T)
    where P* is saturation pressure in MPa and T is temperature in K.
    
    Requires critical properties: Tc, Pc, ac (acentric factor), and Tb (boiling point)
    """
    
    def __init__(self, missing_group_action: str = "remove_compounds", verbose: bool = False):
        """
        Initialize the Antoine parameter calculator.
        
        Args:
            missing_group_action: Action for compounds with missing critical properties:
                                "remove_compounds" (default) or "set_zero"
            verbose: Enable verbose solver output
        """
        self.missing_group_action = missing_group_action
        self.verbose = verbose
        self.missing_properties_summary = {}  # Track missing properties
        
        if missing_group_action not in ["remove_compounds", "set_zero"]:
            raise ValueError(f"missing_group_action must be 'remove_compounds' or 'set_zero', got '{missing_group_action}'")
    
    def _handle_missing_properties(self, compound: Dict[str, int], missing_properties: list) -> bool:
        """
        Handle missing critical properties according to the configured action.
        
        Args:
            compound: The compound being calculated
            missing_properties: List of missing critical properties
            
        Returns:
            bool: True if calculation should continue, False if compound should be removed
        """
        if not missing_properties:
            return True
            
        # Track missing properties for summary
        compound_str = str(compound)
        if compound_str not in self.missing_properties_summary:
            self.missing_properties_summary[compound_str] = set()
        self.missing_properties_summary[compound_str].update(missing_properties)
        
        if self.missing_group_action == "remove_compounds":
            print(f"Warning: Compound {compound} has missing critical properties: {missing_properties}")
            print(f"         Action: Removing compound (missing_group_action='remove_compounds')")
            return False
        elif self.missing_group_action == "set_zero":
            print(f"Warning: Compound {compound} has missing critical properties: {missing_properties}")
            print(f"         Action: Cannot calculate Antoine parameters with missing critical properties")
            return False  # Antoine calculation requires all critical properties
        
        return True
    
    def print_missing_properties_summary(self):
        """Print a summary of all missing properties encountered during calculations."""
        if not self.missing_properties_summary:
            print("No missing critical properties found for Antoine calculations.")
            return
            
        print("\n=== Missing Critical Properties Summary (Antoine Calculator) ===")
        for compound, missing_props in self.missing_properties_summary.items():
            print(f"Compound {compound}: {sorted(missing_props)}")
        print(f"Action taken: {self.missing_group_action}")
        print("=" * 65)
    
    def calculate_antoine_parameters(self, compound: Dict[str, int], 
                                   critical_properties: Dict[str, Optional[float]]) -> Optional[Dict[str, float]]:
        """
        Calculate Antoine parameters for a compound using its critical properties.
        
        Args:
            compound: Dict with group names as keys and counts as values
            critical_properties: Dict with critical properties from PropertyCalculator
                                Required: 'tc', 'pc', 'ac', 'tb'
                                
        Returns:
            Dict with Antoine parameters {'A', 'B', 'C'} or None if compound should be removed
        """
        # Check for required critical properties
        required_props = ['tc', 'pc', 'ac', 'tb']
        missing_props = []
        
        for prop in required_props:
            if prop not in critical_properties or critical_properties[prop] is None:
                missing_props.append(prop)
        
        # Handle missing properties
        if not self._handle_missing_properties(compound, missing_props):
            return None
        
        if missing_props:  # If we get here, we're in "set_zero" mode, but can't proceed
            return None
            
        # Extract critical properties
        tc = critical_properties['tc']  # Critical temperature [K]
        pc = critical_properties['pc'] / 1e6  # Critical pressure [Pa] -> [MPa]  
        ac = critical_properties['ac']  # Acentric factor [-]
        tb = critical_properties['tb']  # Boiling point [K]
        
        # Standard atmospheric pressure in MPa
        p0 = 0.101325
        
        # Calculate Antoine parameters
        try:
            result = self.solve_antoine(Tb=tb, Tc=tc, Pc=pc, P0=p0, w=ac, verbose=self.verbose)
            
            if result['solv_stat'] != 'ok' or result['term_cond'] != 'optimal':
                print(f"Warning: Antoine parameter solving failed for compound {compound}")
                print(f"         Solver status: {result['solv_stat']}, Termination: {result['term_cond']}")
                return None
                
            return {
                'A': result['A'],
                'B': result['B'], 
                'C': result['C'],
                'solver_status': result['solv_stat'],
                'termination_condition': result['term_cond']
            }
            
        except Exception as e:
            print(f"Error calculating Antoine parameters for compound {compound}: {e}")
            return None
    
    def solve_antoine(self, Tb: float, Tc: float, Pc: float, P0: float, w: float, verbose: bool = None) -> dict:
        """
        Solve for Antoine equation parameters A, B, C using pyomo optimization.
        
        Based on the solve_antoine function from utils/gc.py
        
        Args:
            Tb: Normal boiling point in Kelvin
            Tc: Critical temperature in Kelvin  
            Pc: Critical pressure in MPa
            P0: Atmospheric pressure in MPa
            w: Acentric factor [-]
            verbose: Printing of solver progress (overrides instance default if provided)
            
        Returns:
            Dict containing: solver_stat, term_cond, A0, B0, C0, A, B, C
            
        Note: Antoine equation: P(MPa) = 10^(A - B / (C + T))
        """
        if verbose is None:
            verbose = self.verbose
            
        # Step 1: Calculate saturated pressure at Tr = 0.7 (acentric factor definition)
        Pw = np.power(10, -(w + 1)) * Pc
        
        # Step 2: Initial guess using selected-points method
        t1 = Tb
        t2 = 0.7 * Tc  
        t3 = Tc
        p1 = P0
        p2 = Pw
        p3 = Pc
        y1 = np.log10(p1)
        y2 = np.log10(p2)
        y3 = np.log10(p3)
        
        # Initial parameter estimates
        X = (y3 - y2) / (y2 - y1) * (t2 - t1) / (t3 - t2)
        Y = t1 - t3
        C0 = Y / (X - 1) - t3
        B0 = (y3 - y1) / (t3 - t1) * (t1 + C0) * (t3 - C0)
        A0 = y2 + B0 / (t2 + C0)
        
        # Step 3: Pyomo optimization model
        model = pyo.ConcreteModel('Antoine parameter optimization')
        
        # Parameters
        model.tb = pyo.Param(initialize=Tb, mutable=False, domain=pyo.PositiveReals)
        model.tc = pyo.Param(initialize=Tc, mutable=False, domain=pyo.PositiveReals)
        model.pc = pyo.Param(initialize=Pc, mutable=False, domain=pyo.PositiveReals)
        model.w = pyo.Param(initialize=w, mutable=False, domain=pyo.PositiveReals)
        model.p0 = pyo.Param(initialize=P0, mutable=False, domain=pyo.PositiveReals)
        model.pw = pyo.Param(initialize=Pw, mutable=False, domain=pyo.PositiveReals)
        
        # Variables
        model.A = pyo.Var(within=pyo.Reals, initialize=A0, name='Antoine parameter A')
        model.B = pyo.Var(within=pyo.Reals, initialize=B0, name='Antoine parameter B')
        model.C = pyo.Var(within=pyo.Reals, initialize=C0, name='Antoine parameter C')
        
        # Constraints
        def eq_p0(m):
            """Constraint for atmospheric pressure at boiling point"""
            return pyo.log10(m.p0) == m.A - m.B / (m.C + m.tb)
        model.eq_p0 = pyo.Constraint(rule=eq_p0)
        
        def eq_pc(m):
            """Constraint for critical pressure at critical temperature"""
            return pyo.log10(m.pc) == m.A - m.B / (m.C + m.tc)
        model.eq_pc = pyo.Constraint(rule=eq_pc)
        
        def eq_pw(m):
            """Constraint for pressure at Tr = 0.7 (acentric factor definition)"""
            return pyo.log10(m.pw) == m.A - m.B / (m.C + m.tc * 0.7)
        model.eq_pw = pyo.Constraint(rule=eq_pw)
        
        # Solver configuration
        opt = pyo.SolverFactory('ipopt')
        opt.options["halt_on_ampl_error"] = "yes"
        #opt.options["hsllib"] = "libcoinhsl.so"
        #opt.options["linear_solver"] = "MA27"
        opt.options["max_iter"] = 10000
        
        # Solve
        logfile = "pyomo_antoine_log"
        solution = opt.solve(model, tee=verbose, symbolic_solver_labels=True, logfile=logfile)
        
        # Extract results
        solv_stat = str(solution.solver.status)
        term_cond = str(solution.solver.termination_condition)
        
        if solv_stat != 'ok' or term_cond != 'optimal':
            print('Warning: Antoine parameter optimization did not converge optimally')
            print(f'Solver status: {solv_stat}, Termination condition: {term_cond}')
        
        # Parse iterations from log file
        try:
            num_iter = parse_pyomo_log(logfile)
        except:
            num_iter = None
        
        output = {
            'solv_stat': solv_stat,
            'term_cond': term_cond,
            'num_iter': num_iter,
            'A0': A0,  # Initial guess
            'B0': B0,  # Initial guess
            'C0': C0,  # Initial guess
            'A': model.A.value,  # Optimized value
            'B': model.B.value,  # Optimized value
            'C': model.C.value   # Optimized value
        }
        
        return output
    
    def antoine_correlation(self, A: float, B: float, C: float, T: float) -> float:
        """
        Calculate saturation pressure using Antoine equation.
        
        Args:
            A, B, C: Antoine parameters
            T: Temperature in Kelvin
            
        Returns:
            Saturation pressure in MPa
        """
        return 10 ** (A - B / (C + T))
    
    def validate_antoine_parameters(self, A: float, B: float, C: float, 
                                  tb: float, tc: float, pc: float) -> Dict[str, float]:
        """
        Validate Antoine parameters by calculating pressures at key points.
        
        Args:
            A, B, C: Antoine parameters
            tb: Boiling point [K]
            tc: Critical temperature [K]
            pc: Critical pressure [MPa]
            
        Returns:
            Dict with validation results
        """
        p_at_tb = self.antoine_correlation(A, B, C, tb)
        p_at_tc = self.antoine_correlation(A, B, C, tc)
        
        return {
            'pressure_at_boiling_point': p_at_tb,
            'pressure_at_critical_temp': p_at_tc,
            'error_at_boiling_point': abs(p_at_tb - 0.101325),  # Should be ~0.101325 MPa
            'error_at_critical_temp': abs(p_at_tc - pc),  # Should equal pc
            'relative_error_critical': abs(p_at_tc - pc) / pc * 100  # Percentage error
        }


# Convenience function for quick calculations  
def calculate_antoine_parameters(compound: Dict[str, int], 
                               critical_properties: Dict[str, Optional[float]],
                               missing_group_action: str = "remove_compounds") -> Optional[Dict[str, float]]:
    """
    Convenience function to calculate Antoine parameters for a compound.
    
    Args:
        compound: Dict with group names as keys and counts as values
        critical_properties: Dict with critical properties (tc, pc, ac, tb)
        missing_group_action: Action for missing properties
        
    Returns:
        Dict with Antoine parameters or None if compound should be removed
        
    Example:
        >>> compound = {'CH3': 2, 'OH': 1}  # Ethanol
        >>> props = {'tc': 513.9, 'pc': 6.137e6, 'ac': 0.645, 'tb': 351.4}
        >>> antoine_params = calculate_antoine_parameters(compound, props)
        >>> if antoine_params:
        >>>     print(f"A: {antoine_params['A']:.3f}")
    """
    calculator = AntoineCalculator(missing_group_action=missing_group_action)
    return calculator.calculate_antoine_parameters(compound, critical_properties)


if __name__ == "__main__":
    # Example usage
    print("=== Antoine Calculator Example ===")
    
    # Test compounds with known critical properties
    test_cases = {
        'ethanol': {
            'compound': {'CH3': 1, 'CH2': 1, 'OH': 1},
            'properties': {'tc': 513.9, 'pc': 6.137e6, 'ac': 0.645, 'tb': 351.4}
        },
        'acetone': {
            'compound': {'CH3': 2, 'CH3CO': 1}, 
            'properties': {'tc': 508.2, 'pc': 4.7e6, 'ac': 0.307, 'tb': 329.4}
        }
    }
    
    calculator = AntoineCalculator(verbose=False)
    
    for name, data in test_cases.items():
        print(f"\n{name.upper()}: {data['compound']}")
        print(f"Critical properties: Tc={data['properties']['tc']:.1f}K, "
              f"Pc={data['properties']['pc']/1e6:.2f}MPa, "
              f"w={data['properties']['ac']:.3f}, Tb={data['properties']['tb']:.1f}K")
        
        antoine_params = calculator.calculate_antoine_parameters(
            data['compound'], data['properties']
        )
        
        if antoine_params is None:
            print("  Antoine parameters could not be calculated")
            continue
            
        print(f"  Antoine parameters:")
        print(f"    A: {antoine_params['A']:.4f}")
        print(f"    B: {antoine_params['B']:.4f}")
        print(f"    C: {antoine_params['C']:.4f}")
        
        # Validate results
        validation = calculator.validate_antoine_parameters(
            antoine_params['A'], antoine_params['B'], antoine_params['C'],
            data['properties']['tb'], data['properties']['tc'], 
            data['properties']['pc']/1e6
        )
        
        print(f"  Validation:")
        print(f"    P at boiling point: {validation['pressure_at_boiling_point']:.6f} MPa (target: 0.101325)")
        print(f"    P at critical temp: {validation['pressure_at_critical_temp']:.3f} MPa "
              f"(target: {data['properties']['pc']/1e6:.3f})")
        print(f"    Relative error at Tc: {validation['relative_error_critical']:.2f}%")
    
    # Print summary
    calculator.print_missing_properties_summary()