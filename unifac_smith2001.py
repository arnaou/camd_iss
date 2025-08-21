#!/usr/bin/env python3
"""
UNIFAC Model Implementation - Smith et al. (2001) Formulation

This script implements the UNIFAC method exactly as formulated by Smith et al. (2001)
with the equations from Appendix A. The equations are in a convenient form for 
programming purposes and are slightly modified to avoid numerical difficulties.

References:
- Smith, J. M., Van Ness, H. C., & Abbott, M. M. (2001). Introduction to chemical engineering thermodynamics
- Appendix A: UNIFAC Model equations (A.1) - (A.16)
"""

import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class UNIFACSmith2001:
    """
    UNIFAC Model following Smith et al. (2001) formulation exactly.
    
    Implements equations (A.1) through (A.16) from the appendix for calculating
    liquid activity coefficients in non-ideal mixtures.
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
                
            print(f"Loaded UNIFAC Smith2001 data:")
            print(f"  R parameters: {len(self.rk_data)} groups")
            print(f"  Q parameters: {len(self.qk_data)} groups") 
            print(f"  Interaction parameters: {len(self.interaction_data)} groups")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load UNIFAC data: {e}")
    
    def get_interaction_parameter(self, group_i: str, group_j: str) -> float:
        """Get interaction parameter a_mk between groups."""
        if group_i in self.interaction_data and group_j in self.interaction_data[group_i]:
            a_mk = self.interaction_data[group_i][group_j]
        elif group_j in self.interaction_data and group_i in self.interaction_data[group_j]:
            a_mk = self.interaction_data[group_j][group_i]
        else:
            warnings.warn(f"Missing interaction parameter for {group_i}-{group_j}, using 0.0")
            a_mk = 0.0
        return a_mk if a_mk is not None else 0.0
    
    def unifac(self, mixture_composition: Dict[str, List[float]], 
               x: np.ndarray, T: float) -> np.ndarray:
        """
        Calculate activity coefficients using Smith et al. (2001) UNIFAC method.
        
        This implementation follows the working approach from utils/gc.py but with
        Smith et al. (2001) equation references for clarity.
        
        Args:
            mixture_composition: Group composition matrix {group: [n_i_k for each component]}
            x: Component mole fractions
            T: Temperature in Kelvin
            
        Returns:
            gamma: Activity coefficients
        """
        # Setup - extract groups and compositions 
        groups = list(mixture_composition.keys())
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

def example_usage():
    """
    Example usage of the Smith et al. (2001) UNIFAC implementation.
    """
    print("=== UNIFAC Smith et al. (2001) Implementation ===")
    
    # Initialize model
    unifac = UNIFACSmith2001()
    
    # Define binary mixture: diethylamine + heptane
    # Diethylamine: (C2H5)2NH -> CH3(2) + CH2(2) + CH2NH(1)
    # Heptane: C7H16 -> CH3(2) + CH2(5)
    mixture_composition = {
        'CH3':   [2, 2],      # 2 CH3 groups in diethylamine, 2 in heptane
        'CH2':   [1, 5],      # 1 CH2 groups in diethylamine, 5 in heptane
        'CH2NH': [1, 0],      # 1 CH2NH group in diethylamine only
    }
    # Component mole fractions x_i
    x = np.array([0.4, 0.6])  # 40% diethylamine, 60% heptane
    T = [308.15, 310]  # Temperature T_t in Kelvin

    print(f"Mixture: Diethylamine(40%) + Heptane(60%)")
    print(f"Temperature: {T} K ({T-273.15:.1f}°C)")
    print(f"Group composition matrix:")
    for group, composition in mixture_composition.items():
        print(f"  {group}: {composition}")
    
    try:
        # Calculate activity coefficients γ_i
        gamma = unifac.unifac(mixture_composition, x, T)
        
        print(f"\nActivity coefficients (Smith et al. 2001 formulation):")
        components = ['Diethylamine', 'Heptane']
        for i, comp in enumerate(components):
            g = float(gamma[i])
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