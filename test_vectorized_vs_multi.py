#!/usr/bin/env python3
"""
Comprehensive test comparing vectorized vs loop-based UNIFAC implementations.
"""

import numpy as np
import time
from unifac_vectorized import UNIFACVectorized
from unifac_multi import UNIFACMulti
from unifac_smith2001 import UNIFACSmith2001

def test_accuracy():
    """Test numerical accuracy across different scenarios."""
    print("=== Accuracy Testing ===\n")
    
    # Initialize all implementations
    unifac_vec = UNIFACVectorized()
    unifac_multi = UNIFACMulti()
    unifac_smith = UNIFACSmith2001()
    
    mixture_composition = {
        'CH3':   [2, 2],
        'CH2':   [1, 5],
        'CH2NH': [1, 0],
    }
    
    # Test 1: Single point
    print("1. Single point calculation:")
    x_single = np.array([0.4, 0.6])
    T_single = 308.15
    
    gamma_vec = unifac_vec.unifac(mixture_composition, x_single, T_single)
    gamma_multi = unifac_multi.unifac(mixture_composition, x_single, T_single)
    gamma_smith = unifac_smith.unifac(mixture_composition, x_single, T_single)
    
    print(f"   Vectorized: {gamma_vec}")
    print(f"   Multi loop: {gamma_multi}")
    print(f"   Smith2001:  {gamma_smith}")
    
    diff_vec_multi = np.max(np.abs(gamma_vec - gamma_multi))
    diff_vec_smith = np.max(np.abs(gamma_vec - gamma_smith))
    
    print(f"   Max diff (vec vs multi): {diff_vec_multi:.2e}")
    print(f"   Max diff (vec vs smith): {diff_vec_smith:.2e}")
    print(f"   ✅ Single point: {'PASS' if diff_vec_multi < 1e-10 else 'FAIL'}")
    
    # Test 2: Multiple temperatures
    print("\n2. Multiple temperatures:")
    T_array = np.array([298.15, 308.15, 318.15, 328.15])
    
    gamma_vec_T = unifac_vec.unifac(mixture_composition, x_single, T_array)
    gamma_multi_T = unifac_multi.unifac(mixture_composition, x_single, T_array)
    
    diff_T = np.max(np.abs(gamma_vec_T - gamma_multi_T))
    print(f"   Shape: {gamma_vec_T.shape}")
    print(f"   Max difference: {diff_T:.2e}")
    print(f"   ✅ Multiple T: {'PASS' if diff_T < 1e-10 else 'FAIL'}")
    
    # Test 3: Multiple compositions
    print("\n3. Multiple compositions:")
    x_array = np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]])
    
    gamma_vec_x = unifac_vec.unifac(mixture_composition, x_array, T_single)
    gamma_multi_x = unifac_multi.unifac(mixture_composition, x_array, T_single)
    
    diff_x = np.max(np.abs(gamma_vec_x - gamma_multi_x))
    print(f"   Shape: {gamma_vec_x.shape}")
    print(f"   Max difference: {diff_x:.2e}")
    print(f"   ✅ Multiple x: {'PASS' if diff_x < 1e-10 else 'FAIL'}")
    
    # Test 4: Large dataset
    print("\n4. Large dataset (500 points):")
    np.random.seed(42)
    n_test = 500
    x_large = np.random.dirichlet([1, 1], n_test)
    T_large = np.random.uniform(290, 350, n_test)
    
    gamma_vec_large = unifac_vec.unifac(mixture_composition, x_large, T_large)
    gamma_multi_large = unifac_multi.unifac(mixture_composition, x_large, T_large)
    
    diff_large = np.max(np.abs(gamma_vec_large - gamma_multi_large))
    print(f"   Shape: {gamma_vec_large.shape}")
    print(f"   Max difference: {diff_large:.2e}")
    print(f"   ✅ Large dataset: {'PASS' if diff_large < 1e-10 else 'FAIL'}")

def performance_benchmark():
    """Detailed performance benchmark."""
    print("\n=== Performance Benchmark ===\n")
    
    unifac_vec = UNIFACVectorized()
    unifac_multi = UNIFACMulti()
    
    mixture_composition = {
        'CH3':   [2, 2],
        'CH2':   [1, 5],
        'CH2NH': [1, 0],
    }
    
    # Test different dataset sizes
    test_sizes = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    
    print("Dataset Size | Vec Time (ms) | Loop Time (ms) | Speedup | Memory Efficiency")
    print("-" * 80)
    
    results = []
    
    for n_points in test_sizes:
        # Generate test data
        np.random.seed(42)
        x_test = np.random.dirichlet([1, 1], n_points)
        T_test = np.random.uniform(290, 350, n_points)
        
        # Benchmark vectorized version (3 runs, take average)
        times_vec = []
        for _ in range(3):
            start_time = time.perf_counter()
            gamma_vec = unifac_vec.unifac(mixture_composition, x_test, T_test)
            times_vec.append((time.perf_counter() - start_time) * 1000)
        time_vec = np.mean(times_vec)
        
        # Benchmark loop-based version (3 runs, take average)
        times_loop = []
        for _ in range(3):
            start_time = time.perf_counter()
            gamma_loop = unifac_multi.unifac(mixture_composition, x_test, T_test)
            times_loop.append((time.perf_counter() - start_time) * 1000)
        time_loop = np.mean(times_loop)
        
        speedup = time_loop / time_vec if time_vec > 0 else float('inf')
        
        # Memory efficiency (ops per ms)
        ops_per_ms_vec = n_points / time_vec if time_vec > 0 else float('inf')
        ops_per_ms_loop = n_points / time_loop if time_loop > 0 else float('inf')
        memory_eff = ops_per_ms_vec / ops_per_ms_loop if ops_per_ms_loop > 0 else float('inf')
        
        print(f"{n_points:11d} | {time_vec:11.2f} | {time_loop:12.2f} | {speedup:6.1f}x | {memory_eff:13.1f}x")
        
        results.append({
            'size': n_points,
            'time_vec': time_vec,
            'time_loop': time_loop,
            'speedup': speedup,
            'memory_eff': memory_eff
        })
        
        # Verify accuracy
        max_diff = np.max(np.abs(gamma_vec - gamma_loop))
        if max_diff > 1e-10:
            print(f"   ⚠️ Accuracy issue: max diff = {max_diff:.2e}")
    
    # Summary statistics
    print(f"\nSummary Statistics:")
    speedups = [r['speedup'] for r in results if r['speedup'] != float('inf')]
    print(f"   Average speedup: {np.mean(speedups):.1f}x")
    print(f"   Maximum speedup: {np.max(speedups):.1f}x")
    print(f"   Speedup std dev: {np.std(speedups):.1f}x")
    
    # Performance scaling analysis
    large_datasets = [r for r in results if r['size'] >= 100]
    if large_datasets:
        large_speedups = [r['speedup'] for r in large_datasets]
        print(f"   Large dataset avg speedup (≥100 points): {np.mean(large_speedups):.1f}x")

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\n=== Edge Case Testing ===\n")
    
    unifac_vec = UNIFACVectorized()
    
    mixture_composition = {
        'CH3':   [2, 2],
        'CH2':   [1, 5],
        'CH2NH': [1, 0],
    }
    
    # Test 1: Very dilute compositions
    print("1. Very dilute compositions:")
    try:
        x_dilute = np.array([[0.001, 0.999], [0.999, 0.001]])
        T_test = np.array([308.15, 308.15])
        gamma_dilute = unifac_vec.unifac(mixture_composition, x_dilute, T_test)
        print(f"   Dilute results: {gamma_dilute}")
        print("   ✅ Dilute compositions: PASS")
    except Exception as e:
        print(f"   ❌ Dilute compositions: FAIL - {e}")
    
    # Test 2: High temperatures
    print("\n2. High temperatures:")
    try:
        x_test = np.array([0.5, 0.5])
        T_high = np.array([500.0, 600.0])
        gamma_high_T = unifac_vec.unifac(mixture_composition, x_test, T_high)
        print(f"   High T results: {gamma_high_T}")
        print("   ✅ High temperatures: PASS")
    except Exception as e:
        print(f"   ❌ High temperatures: FAIL - {e}")
    
    # Test 3: Low temperatures
    print("\n3. Low temperatures:")
    try:
        T_low = np.array([250.0, 275.0])
        gamma_low_T = unifac_vec.unifac(mixture_composition, x_test, T_low)
        print(f"   Low T results: {gamma_low_T}")
        print("   ✅ Low temperatures: PASS")
    except Exception as e:
        print(f"   ❌ Low temperatures: FAIL - {e}")

if __name__ == "__main__":
    test_accuracy()
    performance_benchmark() 
    test_edge_cases()
    print("\n=== All tests completed ===")