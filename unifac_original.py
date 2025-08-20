#!/usr/bin/env python3
"""
UNIFAC Model Implementation - Original Formulation

This script implements the UNIFAC method based on the original working implementation
from utils/gc.py, but structured as a class similar to the Smith2001 version.

This is the "reference" implementation that was working correctly before the Smith
modifications. It uses the exact same vectorized operations as the original.

References:
- Original implementation in utils/gc.py
- Numerical stability improvements from https://doi.org/10.1016/j.ces.2022.118125
"""

import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

class UNIFACOriginal:
    """
    UNIFAC Model using the original working formulation from utils/gc.py.
    
    This implementation uses the exact same vectorized operations as the original
    but packaged in a class structure for consistency with other implementations.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize UNIFAC model with group contribution data.
        
        Args:
            data_path: Path to data directory containing JSON files
        """
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "json"
        else:
            data_path = Path(data_path)
        
        self.data_path = data_path
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
                
            print(f"Loaded UNIFAC Original data:")
            print(f"  R parameters: {len(self.rk_data)} groups")
            print(f"  Q parameters: {len(self.qk_data)} groups") 
            print(f"  Interaction parameters: {len(self.interaction_data)} groups")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load UNIFAC data: {e}")
    
    def get_interaction_parameter(self, group_i: str, group_j: str) -> float:
        """Get interaction parameter between groups."""
        if group_i in self.interaction_data and group_j in self.interaction_data[group_i]:
            a_mk = self.interaction_data[group_i][group_j]
        elif group_j in self.interaction_data and group_i in self.interaction_data[group_j]:
            a_mk = self.interaction_data[group_j][group_i]
        else:
            warnings.warn(f"Missing interaction parameter for {group_i}-{group_j}, using 0.0")
            a_mk = 0.0
        return a_mk if a_mk is not None else 0.0
    
    def unifac(self, mixture_composition: Dict[str, List[float]], 
               x0: np.ndarray, T0: float) -> np.ndarray:
        """
        Calculate activity coefficients using the original UNIFAC method.
        
        This is the exact implementation from utils/gc.py lines 527-545, but
        adapted to use JSON data loading instead of Excel files.
        
        Args:
            mixture_composition: Group composition matrix {group: [n_i_k for each component]}
            x0: Component mole fractions  
            T0: Temperature in Kelvin
            
        Returns:
            gamma: Activity coefficients
        """
        # Setup - extract groups and compositions (equivalent to pd.DataFrame -> np.array)
        groups = list(mixture_composition.keys())
        v = np.array([mixture_composition[group] for group in groups])  # Shape: (n_groups, n_comp)
        
        # Get group parameters (equivalent to df_gc['Rk'][mixture.columns])
        Rk = np.array([self.rk_data[group] for group in groups])
        Qk = np.array([self.qk_data[group] for group in groups])
        
        # Build interaction parameter matrix (equivalent to df_a[mixture.columns].loc[mixture.columns])
        n_groups = len(groups)
        a = np.zeros((n_groups, n_groups))
        for i, group_i in enumerate(groups):
            for j, group_j in enumerate(groups):
                a[i, j] = self.get_interaction_parameter(group_i, group_j)
        
        # ========== ORIGINAL UNIFAC IMPLEMENTATION ==========
        # Lines 532-545 from utils/gc.py - exact copy with variable name consistency
        
        # Begin calculations: combinatorial term
        r = np.dot(v.T, Rk)  # Changed from np.dot(v, Rk) to match our v.T layout
        q = np.dot(v.T, Qk)  # Changed from np.dot(v, Qk) to match our v.T layout
        L = np.matrix(q) / np.matrix(np.dot(x0, q)).T
        j = np.matrix(r) / np.matrix(np.dot(x0, r)).T
        gamma_c = 1 - j + np.log(j) - 5 * np.multiply(q, (1 - j / L + np.log(j / L)))
        
        # Begin calculation: residual term
        e = (v.T * Qk) / q[:, np.newaxis]  # Use same vectorization as Smith2001
        tau = np.exp(-a / T0)
        beta = np.dot(v.T * Qk, tau) / q[:, np.newaxis]  # Use same vectorization as Smith2001
        theta = np.dot(x0, v.T * Qk) / np.dot(x0, q)  # Use v.T for consistency
        omega = np.dot(theta, tau)
        gamma_r = q * (1 - np.sum(theta * beta / omega - e * np.log(beta / omega), axis=1))
        
        # Total activity coefficient
        gamma = np.array(np.exp(gamma_c + gamma_r)).flatten()
        
        return gamma

def example_usage():
    """
    Example usage of the Original UNIFAC implementation.
    """
    print("=== UNIFAC Original Implementation (from utils/gc.py) ===")
    
    # Initialize model
    unifac = UNIFACOriginal()
    
    # Define binary mixture: diethylamine + heptane
    # Diethylamine: (C2H5)2NH -> CH3(2) + CH2(1) + CH2NH(1)
    # Heptane: C7H16 -> CH3(2) + CH2(5)
    mixture_composition = {
        'CH3':   [2, 2],      # 2 CH3 groups in diethylamine, 2 in heptane
        'CH2':   [1, 5],      # 1 CH2 group in diethylamine, 5 in heptane
        'CH2NH': [1, 0],      # 1 CH2NH group in diethylamine only
    }
    
    # Component mole fractions x_i
    x0 = np.array([0.4, 0.6])  # 40% diethylamine, 60% heptane
    T0 = 308.15  # Temperature in Kelvin

    print(f"Mixture: Diethylamine(40%) + Heptane(60%)")
    print(f"Temperature: {T0} K ({T0-273.15:.1f}°C)")
    print(f"Group composition matrix:")
    for group, composition in mixture_composition.items():
        print(f"  {group}: {composition}")

    try:
        # Calculate activity coefficients γ_i
        gamma = unifac.unifac(mixture_composition, x0, T0)
        
        print(f"\nActivity coefficients:")
        components = ['Diethylamine', 'Heptane']
        for i, comp in enumerate(components):
            g = gamma[i]
            print(f"  γ_{i+1} ({comp}) = {g:.4f}")
            
        # Analysis of mixture behavior
        print(f"\nMixture analysis:")
        if np.allclose(gamma, 1.0, rtol=0.05):
            print("  Behavior: Nearly ideal (γ ≈ 1)")
        else:
            print("  Behavior: Non-ideal mixture")
            for i, comp in enumerate(components):
                g = gamma[i]
                if g > 1.05:
                    print(f"  {comp}: Positive deviation (γ = {g:.3f}) - less favorable mixing")
                elif g < 0.95:
                    print(f"  {comp}: Negative deviation (γ = {g:.3f}) - more favorable mixing")
                else:
                    print(f"  {comp}: Nearly ideal behavior (γ = {g:.3f})")
                    
    except Exception as e:
        print(f"Calculation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    example_usage()