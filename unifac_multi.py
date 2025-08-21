#!/usr/bin/env python3
"""
UNIFAC Model Implementation - Extended for Multiple T and x Values

This script extends the UNIFAC Smith2001 implementation to handle multiple
temperature and composition data points simultaneously. The length of T 
and x arrays must be the same.

Based on: unifac_smith2001.py
Author: Extended implementation for batch calculations
"""

import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt

class UNIFACMulti:
    """
    Extended UNIFAC Model for multiple T and x calculations.
    
    This implementation can handle:
    - Single T, single x (original functionality)
    - Multiple T, multiple x (same length arrays)
    - Vectorized calculations for efficiency
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
                
            print(f"Loaded UNIFAC Multi data:")
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
    
    def _validate_inputs(self, x: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and normalize inputs for multiple data points.
        
        Args:
            x: Composition array - shape (n_points, n_components) or (n_components,)
            T: Temperature array - shape (n_points,) or scalar
            
        Returns:
            x_norm, T_norm: Normalized arrays with consistent shapes
        """
        # Convert to numpy arrays
        x = np.atleast_2d(x)
        T = np.atleast_1d(T)
        
        # Handle single composition with multiple temperatures
        if x.shape[0] == 1 and T.shape[0] > 1:
            x = np.repeat(x, T.shape[0], axis=0)
        
        # Handle multiple compositions with single temperature  
        elif x.shape[0] > 1 and T.shape[0] == 1:
            T = np.repeat(T, x.shape[0])
        
        # Validate dimensions
        if x.shape[0] != T.shape[0]:
            raise ValueError(f"Inconsistent array lengths: x has {x.shape[0]} points, T has {T.shape[0]} points")
        
        # Validate compositions sum to 1
        x_sums = np.sum(x, axis=1)
        if not np.allclose(x_sums, 1.0, rtol=1e-6):
            raise ValueError(f"Compositions must sum to 1.0. Found sums: {x_sums}")
        
        return x, T
    
    def unifac_single(self, mixture_composition: Dict[str, List[float]], 
                     x: np.ndarray, T: float) -> np.ndarray:
        """
        Calculate activity coefficients for a single data point.
        
        This is the core UNIFAC calculation based on Smith et al. (2001).
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
        r = np.dot(v.T, Rk)  # r_i = Σ n_{i,k} R_k
        q = np.dot(v.T, Qk)  # q_i = Σ n_{i,k} Q_k
        
        # Combinatorial term (Equations A.2, A.9, A.10)
        L = np.matrix(q) / np.matrix(np.dot(x, q)).T  # L_{i,t}
        J = np.matrix(r) / np.matrix(np.dot(x, r)).T  # J_{i,t}
        
        # ln γ^C_{i,t}
        gamma_c = 1 - J + np.log(J) - 5 * np.multiply(q, (1 - J / L + np.log(J / L)))
        
        # Residual term (Equations A.11-A.16)
        tau = np.exp(-a / T)  # ψ_{m,k,t}
        
        # Calculate e_{i,k}, β_{i,k,t}, θ_{k,t}, ω_{k,t}
        e = (v.T * Qk) / q[:, np.newaxis]
        vQk = v.T * Qk
        beta = np.dot(vQk, tau) / q[:, np.newaxis]
        theta = np.dot(x, vQk) / np.dot(x, q)
        omega = np.dot(theta, tau)
        
        # ln γ^R_{i,t}
        gamma_r = q * (1 - np.sum(theta * beta / omega - e * np.log(beta / omega), axis=1))
        
        # Total activity coefficient ln γ_{i,t} = ln γ^C_{i,t} + ln γ^R_{i,t}
        gamma = np.array(np.exp(gamma_c + gamma_r)).flatten()
        
        return gamma
    
    def unifac(self, mixture_composition: Dict[str, List[float]], 
               x: Union[np.ndarray, List], T: Union[np.ndarray, List, float]) -> np.ndarray:
        """
        Calculate activity coefficients for multiple T and x values.
        
        Args:
            mixture_composition: Group composition matrix {group: [n_i_k for each component]}
            x: Component mole fractions - shape (n_points, n_components) or (n_components,)
            T: Temperature(s) in Kelvin - shape (n_points,) or scalar
            
        Returns:
            gamma: Activity coefficients - shape (n_points, n_components)
        """
        # Convert inputs to numpy arrays and validate
        x = np.array(x)
        T = np.array(T)
        x, T = self._validate_inputs(x, T)
        
        n_points, n_components = x.shape
        gamma_results = np.zeros((n_points, n_components))
        
        # Calculate activity coefficients for each data point
        for i in range(n_points):
            gamma_results[i] = self.unifac_single(mixture_composition, x[i], T[i])
        
        # Return single row if only one data point
        if n_points == 1:
            return gamma_results[0]
        
        return gamma_results
    
    def plot_gamma_vs_temperature(self, mixture_composition: Dict[str, List[float]],
                                 x: np.ndarray, T_range: Tuple[float, float], 
                                 n_points: int = 50, component_names: Optional[List[str]] = None):
        """
        Plot activity coefficients vs temperature at fixed composition.
        
        Args:
            mixture_composition: Group composition matrix
            x: Fixed composition
            T_range: Temperature range (T_min, T_max) in Kelvin
            n_points: Number of temperature points
            component_names: Names of components for legend
        """
        T_array = np.linspace(T_range[0], T_range[1], n_points)
        gamma_results = self.unifac(mixture_composition, x, T_array)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        n_components = len(x)
        
        for i in range(n_components):
            label = f"Component {i+1}"
            if component_names:
                label = f"{component_names[i]} (γ_{i+1})"
            plt.plot(T_array, gamma_results[:, i], 'o-', label=label)
        
        plt.xlabel('Temperature (K)')
        plt.ylabel('Activity Coefficient γ')
        plt.title('Activity Coefficients vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_gamma_vs_composition(self, mixture_composition: Dict[str, List[float]],
                                 T: float, n_points: int = 50, 
                                 component_names: Optional[List[str]] = None):
        """
        Plot activity coefficients vs composition at fixed temperature (binary systems only).
        
        Args:
            mixture_composition: Group composition matrix
            T: Fixed temperature in Kelvin
            n_points: Number of composition points
            component_names: Names of components for legend
        """
        # Generate composition points for binary system
        x1_array = np.linspace(0.01, 0.99, n_points)
        x_array = np.column_stack([x1_array, 1 - x1_array])
        
        gamma_results = self.unifac(mixture_composition, x_array, T)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        for i in range(2):  # Binary system
            label = f"Component {i+1}"
            if component_names:
                label = f"{component_names[i]} (γ_{i+1})"
            plt.plot(x1_array, gamma_results[:, i], 'o-', label=label)
        
        plt.xlabel('Mole fraction x₁')
        plt.ylabel('Activity Coefficient γ')
        plt.title(f'Activity Coefficients vs Composition at T = {T} K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def example_usage():
    """
    Example usage of the extended UNIFAC implementation.
    """
    print("=== UNIFAC Multi Implementation - Examples ===\n")
    
    # Initialize model
    unifac = UNIFACMulti()
    
    # Define binary mixture: diethylamine + heptane
    mixture_composition = {
        'CH3':   [2, 2],      # 2 CH3 groups in diethylamine, 2 in heptane
        'CH2':   [1, 5],      # 1 CH2 group in diethylamine, 5 in heptane
        'CH2NH': [1, 0],      # 1 CH2NH group in diethylamine only
    }
    
    component_names = ['Diethylamine', 'Heptane']
    
    print("1. Single point calculation:")
    x_single = np.array([0.4, 0.6])
    T_single = 308.15
    gamma_single = unifac.unifac(mixture_composition, x_single, T_single)
    print(f"   x = {x_single}, T = {T_single} K")
    print(f"   γ = {gamma_single}")
    
    print("\n2. Multiple temperatures, single composition:")
    T_multi = np.array([298.15, 308.15, 318.15, 328.15])
    x_fixed = np.array([0.4, 0.6])
    gamma_T = unifac.unifac(mixture_composition, x_fixed, T_multi)
    print(f"   x = {x_fixed} (fixed)")
    print(f"   T = {T_multi} K")
    print(f"   γ results shape: {gamma_T.shape}")
    for i, (T_val, gamma_val) in enumerate(zip(T_multi, gamma_T)):
        print(f"     T = {T_val} K: γ = {gamma_val}")
    
    print("\n3. Multiple compositions, single temperature:")
    x_multi = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]])
    T_fixed = 308.15
    gamma_x = unifac.unifac(mixture_composition, x_multi, T_fixed)
    print(f"   T = {T_fixed} K (fixed)")
    print(f"   x compositions shape: {x_multi.shape}")
    print(f"   γ results shape: {gamma_x.shape}")
    for i, (x_val, gamma_val) in enumerate(zip(x_multi, gamma_x)):
        print(f"     x = {x_val}: γ = {gamma_val}")
    
    print("\n4. Multiple temperatures and compositions (same length):")
    T_varying = np.array([298.15, 308.15, 318.15, 328.15])
    x_varying = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]])
    gamma_both = unifac.unifac(mixture_composition, x_varying, T_varying)
    print(f"   Combined T and x arrays (length {len(T_varying)}):")
    print(f"   γ results shape: {gamma_both.shape}")
    for i, (T_val, x_val, gamma_val) in enumerate(zip(T_varying, x_varying, gamma_both)):
        print(f"     T = {T_val} K, x = {x_val}: γ = {gamma_val}")
    
    # Optional: Create plots (uncomment to generate plots)
    print("\n5. Plotting examples (commented out - uncomment to run):")
    print("   # unifac.plot_gamma_vs_temperature(mixture_composition, [0.4, 0.6], (290, 340), component_names=component_names)")
    print("   # unifac.plot_gamma_vs_composition(mixture_composition, 308.15, component_names=component_names)")


if __name__ == "__main__":
    example_usage()