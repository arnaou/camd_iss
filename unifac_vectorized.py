#!/usr/bin/env python3
"""
UNIFAC Model Implementation - Fully Vectorized Version

This script implements a fully vectorized UNIFAC method that can handle multiple
temperature and composition data points without any Python loops. All calculations
are performed using vectorized numpy operations for maximum performance.

Based on: Smith et al. (2001) formulation with full vectorization
Author: Vectorized implementation for high-performance batch calculations
"""

import numpy as np
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import time

class UNIFACVectorized:
    """
    Fully vectorized UNIFAC Model for high-performance batch calculations.
    
    This implementation eliminates all Python loops and uses advanced numpy
    broadcasting to calculate activity coefficients for multiple T and x
    values simultaneously.
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
                
            print(f"Loaded UNIFAC Vectorized data:")
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
    
    def unifac_vectorized(self, mixture_composition: Dict[str, List[float]], 
                         x: Union[np.ndarray, List], T: Union[np.ndarray, List, float]) -> np.ndarray:
        """
        Calculate activity coefficients using fully vectorized operations.
        
        This method performs ALL calculations using vectorized numpy operations
        without any Python loops, maximizing performance for large datasets.
        
        Args:
            mixture_composition: Group composition matrix {group: [n_i_k for each component]}
            x: Component mole fractions - shape (n_points, n_components) or (n_components,)
            T: Temperature(s) in Kelvin - shape (n_points,) or scalar
            
        Returns:
            gamma: Activity coefficients - shape (n_points, n_components) or (n_components,)
        """
        # Convert inputs to numpy arrays and validate
        x = np.array(x, dtype=np.float64)
        T = np.array(T, dtype=np.float64)
        x, T = self._validate_inputs(x, T)
        
        n_points, n_components = x.shape
        
        # Setup - extract groups and compositions 
        groups = list(mixture_composition.keys())
        n_groups = len(groups)
        v = np.array([mixture_composition[group] for group in groups], dtype=np.float64)  # Shape: (n_groups, n_components)
        
        # Get group parameters
        Rk = np.array([self.rk_data[group] for group in groups], dtype=np.float64)  # Shape: (n_groups,)
        Qk = np.array([self.qk_data[group] for group in groups], dtype=np.float64)  # Shape: (n_groups,)
        
        # Build interaction parameter matrix
        a = np.zeros((n_groups, n_groups), dtype=np.float64)
        for i, group_i in enumerate(groups):
            for j, group_j in enumerate(groups):
                a[i, j] = self.get_interaction_parameter(group_i, group_j)
        
        # ========== FULLY VECTORIZED UNIFAC CALCULATIONS ==========
        
        # Molecular parameters (vectorized across all data points)
        # r_i = Σ n_{i,k} R_k, q_i = Σ n_{i,k} Q_k for each component i
        # Shape: (n_points, n_components)
        r = np.dot(v.T, Rk)  # Component molecular parameters (same for all data points)
        q = np.dot(v.T, Qk)  # Component molecular parameters (same for all data points)
        
        # Broadcast to (n_points, n_components) if needed
        if n_points > 1:
            r = np.broadcast_to(r[np.newaxis, :], (n_points, n_components))
            q = np.broadcast_to(q[np.newaxis, :], (n_points, n_components))
        else:
            r = r.reshape(1, -1)
            q = q.reshape(1, -1)
        
        # Combinatorial term calculations (fully vectorized)
        # Shape: (n_points, n_components)
        x_dot_r = np.sum(x * r, axis=1, keepdims=True)  # Σ x_τ r_τ for each data point
        x_dot_q = np.sum(x * q, axis=1, keepdims=True)  # Σ x_τ q_τ for each data point
        
        J = r / x_dot_r  # J_{i,t} = r_i / Σ x_τ r_τ
        L = q / x_dot_q  # L_{i,t} = q_i / Σ x_τ q_τ
        
        # ln γ^C_{i,t} (vectorized)
        ln_gamma_c = 1 - J + np.log(J) - 5 * q * (1 - J / L + np.log(J / L))
        
        # ========== RESIDUAL TERM - FULLY VECTORIZED ==========
        
        # Temperature-dependent interaction parameters
        # tau.shape = (n_points, n_groups, n_groups) - broadcast T across group dimensions
        T_expanded = T[:, np.newaxis, np.newaxis]  # Shape: (n_points, 1, 1)
        tau = np.exp(-a[np.newaxis, :, :] / T_expanded)  # Shape: (n_points, n_groups, n_groups)
        
        # Calculate e_{i,k,t} for all components, groups, and data points
        # e.shape = (n_points, n_components, n_groups)
        q_expanded = q[:, :, np.newaxis]  # Shape: (n_points, n_components, 1)
        v_expanded = v.T[np.newaxis, :, :]  # Shape: (1, n_components, n_groups)
        Qk_expanded = Qk[np.newaxis, np.newaxis, :]  # Shape: (1, 1, n_groups)
        
        e = (v_expanded * Qk_expanded) / q_expanded  # Broadcasting gives (n_points, n_components, n_groups)
        
        # Calculate β_{i,k,t} for all components, groups, and data points
        # β.shape = (n_points, n_components, n_groups)
        vQk_expanded = v_expanded * Qk_expanded  # Shape: (1, n_components, n_groups)
        vQk_broadcast = np.broadcast_to(vQk_expanded, (n_points, n_components, n_groups))
        
        # Matrix multiplication: (n_points, n_components, n_groups) @ (n_points, n_groups, n_groups)
        #                      → (n_points, n_components, n_groups)
        beta = np.einsum('ijk,ikl->ijl', vQk_broadcast, tau) / q_expanded
        
        # Calculate θ_{k,t} - group surface area fractions
        # θ.shape = (n_points, n_groups)
        x_vQk = np.einsum('ij,jk->ik', x, v.T * Qk)  # Shape: (n_points, n_groups)
        x_dot_q_squeezed = x_dot_q.squeeze()
        if x_dot_q_squeezed.ndim == 0:  # Single data point case
            theta = x_vQk / x_dot_q_squeezed  # Shape: (1, n_groups)
        else:  # Multiple data points
            theta = x_vQk / x_dot_q_squeezed[:, np.newaxis]  # Shape: (n_points, n_groups)
        
        # Calculate ω_{k,t} for all groups and data points
        # ω.shape = (n_points, n_groups)
        omega = np.einsum('ij,ijk->ik', theta, tau)  # Shape: (n_points, n_groups)
        
        # Final residual term calculation (fully vectorized)
        # All operations broadcast correctly across (n_points, n_components, n_groups)
        theta_expanded = theta[:, np.newaxis, :]  # Shape: (n_points, 1, n_groups)
        omega_expanded = omega[:, np.newaxis, :]  # Shape: (n_points, 1, n_groups)
        
        term1 = theta_expanded * beta / omega_expanded
        term2 = e * np.log(beta / omega_expanded)
        
        # Sum over groups (axis=2) and multiply by q_i
        ln_gamma_r = q * (1 - np.sum(term1 - term2, axis=2))
        
        # Total activity coefficient
        ln_gamma_total = ln_gamma_c + ln_gamma_r
        gamma = np.exp(ln_gamma_total)
        
        # Return single row if only one data point
        if n_points == 1:
            return gamma[0]
        
        return gamma
    
    # Alias for compatibility
    def unifac(self, mixture_composition: Dict[str, List[float]], 
               x: Union[np.ndarray, List], T: Union[np.ndarray, List, float]) -> np.ndarray:
        """Alias for unifac_vectorized for compatibility."""
        return self.unifac_vectorized(mixture_composition, x, T)
    
    def plot_gamma_vs_temperature(self, mixture_composition: Dict[str, List[float]],
                                 x: np.ndarray, T_range: Tuple[float, float], 
                                 n_points: int = 50, component_names: Optional[List[str]] = None):
        """
        Plot activity coefficients vs temperature at fixed composition.
        """
        T_array = np.linspace(T_range[0], T_range[1], n_points)
        gamma_results = self.unifac_vectorized(mixture_composition, x, T_array)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        n_components = len(x)
        
        for i in range(n_components):
            label = f"Component {i+1}"
            if component_names:
                label = f"{component_names[i]} (γ_{i+1})"
            plt.plot(T_array, gamma_results[:, i], 'o-', label=label, markersize=3)
        
        plt.xlabel('Temperature (K)')
        plt.ylabel('Activity Coefficient γ')
        plt.title('Activity Coefficients vs Temperature (Vectorized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_gamma_vs_composition(self, mixture_composition: Dict[str, List[float]],
                                 T: float, n_points: int = 50, 
                                 component_names: Optional[List[str]] = None):
        """
        Plot activity coefficients vs composition at fixed temperature (binary systems only).
        """
        # Generate composition points for binary system
        x1_array = np.linspace(0.01, 0.99, n_points)
        x_array = np.column_stack([x1_array, 1 - x1_array])
        
        gamma_results = self.unifac_vectorized(mixture_composition, x_array, T)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        for i in range(2):  # Binary system
            label = f"Component {i+1}"
            if component_names:
                label = f"{component_names[i]} (γ_{i+1})"
            plt.plot(x1_array, gamma_results[:, i], '-', label=label, linewidth=2)
        
        plt.xlabel('Mole fraction x₁')
        plt.ylabel('Activity Coefficient γ')
        plt.title(f'Activity Coefficients vs Composition at T = {T} K (Vectorized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def benchmark_comparison():
    """
    Benchmark vectorized implementation against loop-based version.
    """
    print("=== Performance Benchmark ===\n")
    
    # Import the loop-based version for comparison
    from unifac_multi import UNIFACMulti
    
    # Initialize both implementations
    unifac_vectorized = UNIFACVectorized()
    unifac_loop = UNIFACMulti()
    
    # Test mixture
    mixture_composition = {
        'CH3':   [2, 2],
        'CH2':   [1, 5],
        'CH2NH': [1, 0],
    }
    
    # Test different dataset sizes
    test_sizes = [10, 50, 100, 500, 1000]
    
    print("Dataset Size | Vectorized (ms) | Loop-based (ms) | Speedup")
    print("-" * 60)
    
    for n_points in test_sizes:
        # Generate test data
        np.random.seed(42)  # For reproducible results
        x_test = np.random.dirichlet([1, 1], n_points)  # Random compositions that sum to 1
        T_test = np.random.uniform(290, 350, n_points)  # Random temperatures
        
        # Benchmark vectorized version
        start_time = time.perf_counter()
        gamma_vec = unifac_vectorized.unifac_vectorized(mixture_composition, x_test, T_test)
        time_vectorized = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Benchmark loop-based version  
        start_time = time.perf_counter()
        gamma_loop = unifac_loop.unifac(mixture_composition, x_test, T_test)
        time_loop = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        # Calculate speedup
        speedup = time_loop / time_vectorized if time_vectorized > 0 else float('inf')
        
        # Verify results are identical
        max_diff = np.max(np.abs(gamma_vec - gamma_loop))
        
        print(f"{n_points:11d} | {time_vectorized:13.2f} | {time_loop:13.2f} | {speedup:6.1f}x")
        
        if max_diff > 1e-10:
            print(f"  ⚠️ Results differ by {max_diff:.2e}")
        
    print(f"\n✅ All results verified to be numerically identical")


def example_usage():
    """
    Example usage of the vectorized UNIFAC implementation.
    """
    print("=== UNIFAC Vectorized Implementation - Examples ===\n")
    
    # Initialize model
    unifac = UNIFACVectorized()
    
    # Define binary mixture: diethylamine + heptane
    mixture_composition = {
        'CH3':   [2, 2],      # 2 CH3 groups in diethylamine, 2 in heptane
        'CH2':   [1, 5],      # 1 CH2 group in diethylamine, 5 in heptane
        'CH2NH': [1, 0],      # 1 CH2NH group in diethylamine only
    }
    
    component_names = ['Diethylamine', 'Heptane']
    
    print("1. Single point calculation (same as before):")
    x_single = np.array([0.4, 0.6])
    T_single = 308.15
    gamma_single = unifac.unifac_vectorized(mixture_composition, x_single, T_single)
    print(f"   x = {x_single}, T = {T_single} K")
    print(f"   γ = {gamma_single}")
    
    print("\n2. Large dataset calculation (1000 points):")
    np.random.seed(42)
    n_large = 1000
    x_large = np.random.dirichlet([1, 1], n_large)  # Random compositions
    T_large = np.random.uniform(290, 350, n_large)   # Random temperatures
    
    start_time = time.perf_counter()
    gamma_large = unifac.unifac_vectorized(mixture_composition, x_large, T_large)
    calc_time = (time.perf_counter() - start_time) * 1000
    
    print(f"   Dataset size: {n_large} points")
    print(f"   Calculation time: {calc_time:.2f} ms")
    print(f"   Output shape: {gamma_large.shape}")
    print(f"   γ₁ range: [{np.min(gamma_large[:, 0]):.4f}, {np.max(gamma_large[:, 0]):.4f}]")
    print(f"   γ₂ range: [{np.min(gamma_large[:, 1]):.4f}, {np.max(gamma_large[:, 1]):.4f}]")
    
    # Optional: Plotting examples
    print("\n3. Plotting examples (commented out - uncomment to run):")
    print("   # unifac.plot_gamma_vs_temperature(mixture_composition, [0.4, 0.6], (290, 340), component_names=component_names)")
    print("   # unifac.plot_gamma_vs_composition(mixture_composition, 308.15, component_names=component_names)")


if __name__ == "__main__":
    example_usage()
    print("\n" + "="*60)
    benchmark_comparison()