"""
Antoine Calculator

Calculates Antoine equation parameters (A, B, C) and the saturated vapour pressure
using pure component critical properties.
This module integrates with the CAMD workflow but can also be used as standalone functio.

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
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Union
import sys
import os

# Add utils to path for pyomo utilities
sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))
from pyom import parse_pyomo_log, ensure_file_exists_in_cwd


class AntoineCalculator:
    """
    Calculator for Antoine equation parameters using critical properties.
    
    Calculates Antoine parameters A, B, C from the equation:
    log10(P*) = A - B / (C + T)
    where P* is saturation pressure in bar and T is temperature in K.

    Requires critical properties: Tc [K], Pc [bar], ac (acentric factor), and Tb [K]
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
        
        print(f"Warning: Compound {compound} has missing critical properties: {missing_properties}")
        print(f"         Action: Cannot calculate Antoine parameters - all critical properties required")
        return False
    
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
    
    def _handle_calculation_error(self, error: Exception, compound: Dict[str, int], context: str = "") -> None:
        """Handle calculation errors with consistent logging."""
        if self.verbose:
            context_str = f" during {context}" if context else ""
            print(f"Calculation failed for {compound}{context_str}: {error}")
    
    def _has_required_properties(self, properties: Dict[str, Optional[float]], 
                               required: list = ['tc', 'pc', 'ac', 'tb']) -> tuple[bool, list]:
        """Check if all required properties are present and valid."""
        missing_props = []
        for prop in required:
            if prop not in properties or properties[prop] is None:
                missing_props.append(prop)
        return len(missing_props) == 0, missing_props
    
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
        # Check for required critical properties using helper method
        has_props, missing_props = self._has_required_properties(critical_properties)
        
        # Handle missing properties - Antoine calculation requires all critical properties
        if not has_props:
            self._handle_missing_properties(compound, missing_props)
            return None
            
        # Extract critical properties
        tc = critical_properties['tc']  # Critical temperature [K]
        pc = critical_properties['pc']  # Critical pressure [Pa] 
        ac = critical_properties['ac']  # Acentric factor [-]
        tb = critical_properties['tb']  # Boiling point [K]

        # Standard atmospheric pressure in bar
        p0 = 1.01325

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
            self._handle_calculation_error(e, compound, "Antoine parameter calculation")
            return None
    
    def solve_antoine(self, Tb: float, Tc: float, Pc: float, P0: float, w: float, verbose: bool = None) -> dict:
        """
        Solve for Antoine equation parameters A, B, C using pyomo optimization.
        
        this is based on the selected point method described in: https://doi.org/10.1021/cr60119a001
        
        Args:
            Tb: Normal boiling point in Kelvin
            Tc: Critical temperature in Kelvin  
            Pc: Critical pressure in bar
            P0: Atmospheric pressure in bar
            w: Acentric factor [-]
            verbose: Printing of solver progress (overrides instance default if provided)
            
        Returns:
            Dict containing: solver_stat, term_cond, A0, B0, C0, A, B, C

        Note: Antoine equation: P(bar) = 10^(A - B / (C + T))
        """
        if verbose is None:
            verbose = self.verbose
            
        # Step 1: Calculate saturated pressure at Tr = 0.7 (acentric factor definition)
        Pw = np.power(10, -(w + 1)) * Pc

        # Step 2: set three characteristic temperatures (t) and three characteristic pressures (y)
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
            'A': pyo.value(model.A),  # Optimized value
            'B': pyo.value(model.B),  # Optimized value
            'C': pyo.value(model.C)   # Optimized value
        }
        
        return output
    
    def antoine_correlation(self, A: float, B: float, C: float, T: float) -> float:
        """
        Calculate saturation pressure using Antoine equation.
        
        Args:
            A, B, C: Antoine parameters
            T: Temperature in Kelvin
            
        Returns:
            Saturation pressure in bar
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
            pc: Critical pressure [bar]
            
        Returns:
            Dict with validation results
        """
        p_at_tb = self.antoine_correlation(A, B, C, tb)
        p_at_tc = self.antoine_correlation(A, B, C, tc)
        
        return {
            'pressure_at_boiling_point': p_at_tb,
            'pressure_at_critical_temp': p_at_tc,
            'error_at_boiling_point': abs(p_at_tb - 1.01325),  # Should be ~1.01325 bar
            'error_at_critical_temp': abs(p_at_tc - pc),  # Should equal pc
            'relative_error_critical': abs(p_at_tc - pc) / pc * 100  # Percentage error
        }
    
    def calculate_psat_vector(self, T_vector: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
        """
        Vectorized vapor pressure calculation using Antoine equation.
        
        Args:
            T_vector: Array of temperatures [K]
            A, B, C: Antoine equation parameters
            
        Returns:
            Array of vapor pressures [bar]
            
        Example:
            >>> import numpy as np
            >>> T_range = np.linspace(250, 400, 100)
            >>> P_range = calculator.calculate_psat_vector(T_range, A, B, C)
        """
        return 10**(A - B/(T_vector + C))
    
    def plot_psat(self, A: float, B: float, C: float, 
                  tb: float, tc: float, pc: float, ac: float,
                  compound: Optional[Dict[str, int]] = None, compound_name: str = "Compound") -> None:
        """
        Plot saturation pressure vs temperature showing Antoine equation validation.
        
        Args:
            A, B, C: Antoine parameters
            tb: Boiling point [K]
            tc: Critical temperature [K]
            pc: Critical pressure [bar]
            ac: Acentric factor [-]
            compound: Dict with group names and counts (optional)
            compound_name: Name for plot title (used if compound not provided)
        """
        try:
            # Temperature range from Tb to Tc
            T_range = np.linspace(tb, tc, 1000)
            
            # Calculate Psat using vectorized Antoine equation (more efficient)
            P_antoine = self.calculate_psat_vector(T_range, A, B, C)
            
            # Key points
            T_ac = 0.7 * tc  # Temperature where acentric factor is defined (Tr = 0.7)
            P_tb = 1.01325  # Atmospheric pressure at boiling point bar
            P_ac = np.power(10, -(ac + 1)) * pc  # Pressure at Tr = 0.7
            
            # Calculate Antoine values at key points
            P_antoine_tb = self.antoine_correlation(A, B, C, tb)
            P_antoine_ac = self.antoine_correlation(A, B, C, T_ac)
            P_antoine_tc = self.antoine_correlation(A, B, C, tc)
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(T_range, P_antoine, 'b-', linewidth=2, label='Antoine equation')
            plt.plot(tb, P_antoine_tb, 'ro', markersize=8, label=f'Boiling point (Tb={tb:.1f}K)')
            plt.plot(T_ac, P_antoine_ac, 'go', markersize=8, label=f'Tr=0.7 (T={T_ac:.1f}K, ω={ac:.3f})')
            plt.plot(tc, P_antoine_tc, 'mo', markersize=8, label=f'Critical point (Tc={tc:.1f}K)')
            
            # Add horizontal lines for reference pressures
            plt.axhline(y=P_tb, color='r', linestyle='--', alpha=0.5)
            plt.axhline(y=P_ac, color='g', linestyle='--', alpha=0.5)
            plt.axhline(y=pc, color='m', linestyle='--', alpha=0.5)
            
            plt.xlabel('Temperature [K]')
            plt.ylabel('Saturation Pressure [bar]')
            plt.title(f'Psat vs T (Antoine Parameters: A={A:.3f}, B={B:.3f}, C={C:.3f})')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add compound composition box
            if compound:
                # Create compound composition text
                comp_text = "Compound Make Up:\n"
                comp_text += " + ".join([f"{count} {group}" for group, count in compound.items()])
            else:
                comp_text = f"Compound: {compound_name}"
            
            # Add text box with compound composition
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
            plt.text(0.02, 0.75, comp_text, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Warning: matplotlib not available. Skipping Psat plot.")
        except Exception as e:
            print(f"Error creating Psat plot: {e}")


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
    
    # Test compounds
    test_compounds = {
        'ethanol': {'CH3': 1, 'CH2': 1, 'OH': 1},
        'acetone': {'CH3': 2, 'CH3CO': 1}, 
        'water': {'H2O': 1},
        'methanol': {'CH3OH': 1}
    }
    
    # Calculate critical properties using PropertyCalculator
    print("--- Calculating Critical Properties ---")
    try:
        from property_calculator import PropertyCalculator
        
        prop_calc = PropertyCalculator(missing_group_action="remove_compounds")
        test_cases = {}
        
        for name, compound in test_compounds.items():
            print(f"  Calculating properties for {name}: {compound}")
            calc_props = prop_calc.calculate(compound)
            
            if calc_props and all(calc_props[p] is not None for p in ['tc', 'pc', 'ac', 'tb']):
                # PropertyCalculator units: Tc[K], Pc[bar], Ac[-], Tb[K]
                test_cases[name] = {
                    'compound': compound,
                    'properties': {
                        'tc': calc_props['tc'],
                        'pc': calc_props['pc'], 
                        'ac': calc_props['ac'],
                        'tb': calc_props['tb']
                    }
                }
                print(f"    ✓ Success: Tc={calc_props['tc']:.1f}K, Pc={calc_props['pc']:.1f}bar")
            else:
                print(f"    ✗ Failed: Missing group contributions")
        
        if not test_cases:
            print("  No compounds could be calculated. Exiting example.")
            exit()
            
    except ImportError:
        print("  PropertyCalculator not available. Cannot run example without calculated properties.")
        print("  To use experimental properties instead, uncomment the lines below:")
        print("  # experimental_cases = {")
        print("  #     'ethanol': {")
        print("  #         'compound': {'CH3': 1, 'CH2': 1, 'OH': 1},")
        print("  #         'properties': {'tc': 514.0, 'pc': 61.48, 'ac': 0.644, 'tb': 351.44}")
        print("  #     }")
        print("  # }")
        print("  # test_cases = experimental_cases")
        exit()
    
    calculator = AntoineCalculator(verbose=False)
    
    for name, data in test_cases.items():
        print(f"\n{name.upper()}: {data['compound']}")
        props = data['properties']
        print(f"Properties: Tc={props['tc']:.1f}K, Pc={props['pc']:.2f}bar, ω={props['ac']:.3f}, Tb={props['tb']:.1f}K")
        
        antoine_params = calculator.calculate_antoine_parameters(
            data['compound'], data['properties']
        )
        
        if antoine_params is None:
            print("  Antoine parameters could not be calculated")
            continue
            
        A, B, C = antoine_params['A'], antoine_params['B'], antoine_params['C']
        print(f"  Antoine: A={A:.4f}, B={B:.4f}, C={C:.4f}")
        
        # Validate results
        validation = calculator.validate_antoine_parameters(
            A, B, C, props['tb'], props['tc'], props['pc']
        )

        print(f"  Validation: P(Tb)={validation['pressure_at_boiling_point']:.6f} bar, "
              f"P(Tc)={validation['pressure_at_critical_temp']:.3f} bar, "
              f"Error={validation['relative_error_critical']:.2f}%")
        
        # Create diagnostic plot
        calculator.plot_psat(
            A, B, C, props['tb'], props['tc'], props['pc'], props['ac'],
            compound=data['compound'], compound_name=name
        )
    
    # Print summary
    calculator.print_missing_properties_summary()