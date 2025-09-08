"""
Antoine Calculator Visualizations

Plotting and visualization functions for Antoine equation parameters and validation.
This module is separated from the main antoine_calculator.py to keep the core 
calculation logic focused and lightweight.

Example Usage:
    from antoine_calculator import AntoineCalculator
    from antoine_visualizations import plot_psat
    
    calculator = AntoineCalculator()
    compound = {'CH3': 2, 'OH': 1}  # Ethanol
    properties = {'tc': 513.9, 'pc': 61.37, 'ac': 0.645, 'tb': 351.4}
    antoine_params = calculator.calculate_antoine_parameters(compound, properties)
    
    if antoine_params:
        plot_psat(antoine_params['A'], antoine_params['B'], antoine_params['C'],
                  properties['tb'], properties['tc'], properties['pc'], properties['ac'],
                  compound=compound)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_psat(A: float, B: float, C: float, 
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
        P_antoine = 10**(A - B/(T_range + C))
        
        # Key points
        T_ac = 0.7 * tc  # Temperature where acentric factor is defined (Tr = 0.7)
        P_tb = 1.01325  # Atmospheric pressure at boiling point bar
        P_ac = np.power(10, -(ac + 1)) * pc  # Pressure at Tr = 0.7
        
        # Calculate Antoine values at key points
        P_antoine_tb = antoine_correlation(A, B, C, tb)
        P_antoine_ac = antoine_correlation(A, B, C, T_ac)
        P_antoine_tc = antoine_correlation(A, B, C, tc)
        
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


def antoine_correlation(A: float, B: float, C: float, T: float) -> float:
    """
    Calculate saturation pressure using Antoine equation.
    
    Args:
        A, B, C: Antoine parameters
        T: Temperature in Kelvin
        
    Returns:
        Saturation pressure in bar
    """
    return 10 ** (A - B / (C + T))


