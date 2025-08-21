#!/usr/bin/env python3
"""
Test script to verify UNIFAC Multi implementation against original implementations.
"""

import numpy as np
from unifac_multi import UNIFACMulti
from unifac_smith2001 import UNIFACSmith2001
from unifac_original import UNIFACOriginal

def test_consistency():
    """Test that UNIFACMulti gives same results as original implementations."""
    print("=== Testing UNIFAC Multi Consistency ===\n")
    
    # Test mixture
    mixture_composition = {
        'CH3':   [2, 2],
        'CH2':   [1, 5], 
        'CH2NH': [1, 0],
    }
    
    # Test conditions
    x = np.array([0.4, 0.6])
    T = 308.15
    
    # Initialize all implementations
    unifac_multi = UNIFACMulti()
    unifac_smith = UNIFACSmith2001()
    unifac_orig = UNIFACOriginal()
    
    # Calculate with all implementations
    gamma_multi = unifac_multi.unifac(mixture_composition, x, T)
    gamma_smith = unifac_smith.unifac(mixture_composition, x, T)  
    gamma_orig = unifac_orig.unifac(mixture_composition, x, T)
    
    print("Single point comparison:")
    print(f"Multi:     γ = {gamma_multi}")
    print(f"Smith2001: γ = {gamma_smith}")
    print(f"Original:  γ = {gamma_orig}")
    
    # Check consistency
    tol = 1e-10
    multi_vs_smith = np.allclose(gamma_multi, gamma_smith, rtol=tol)
    multi_vs_orig = np.allclose(gamma_multi, gamma_orig, rtol=tol)
    
    print(f"\nConsistency check (tolerance = {tol}):")
    print(f"Multi vs Smith2001: {'✓ PASS' if multi_vs_smith else '❌ FAIL'}")
    print(f"Multi vs Original:   {'✓ PASS' if multi_vs_orig else '❌ FAIL'}")
    
    if multi_vs_smith and multi_vs_orig:
        print("🎯 All implementations are numerically identical!")
    else:
        print("⚠️ Implementations show differences")
        max_diff_smith = np.max(np.abs(gamma_multi - gamma_smith))
        max_diff_orig = np.max(np.abs(gamma_multi - gamma_orig))
        print(f"Max difference vs Smith2001: {max_diff_smith}")
        print(f"Max difference vs Original:  {max_diff_orig}")

def test_multi_functionality():
    """Test extended functionality for multiple T and x values."""
    print("\n=== Testing Multi Functionality ===\n")
    
    unifac = UNIFACMulti()
    mixture_composition = {
        'CH3':   [2, 2],
        'CH2':   [1, 5],
        'CH2NH': [1, 0],
    }
    
    # Test 1: Multiple temperatures
    print("1. Multiple temperatures:")
    T_array = np.array([298.15, 308.15, 318.15])
    x_fixed = np.array([0.4, 0.6])
    
    try:
        gamma_T = unifac.unifac(mixture_composition, x_fixed, T_array)
        print(f"   Input: T = {T_array}, x = {x_fixed}")
        print(f"   Output shape: {gamma_T.shape}")
        print(f"   Results: γ₁ = {gamma_T[:, 0]}, γ₂ = {gamma_T[:, 1]}")
        print("   ✓ SUCCESS")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 2: Multiple compositions
    print("\n2. Multiple compositions:")
    x_array = np.array([[0.3, 0.7], [0.5, 0.5], [0.7, 0.3]])
    T_fixed = 308.15
    
    try:
        gamma_x = unifac.unifac(mixture_composition, x_array, T_fixed)
        print(f"   Input: x = {x_array.tolist()}, T = {T_fixed}")
        print(f"   Output shape: {gamma_x.shape}")
        for i, (x_val, gamma_val) in enumerate(zip(x_array, gamma_x)):
            print(f"   x = {x_val}: γ = {gamma_val}")
        print("   ✓ SUCCESS")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Test 3: Simultaneous T and x variation
    print("\n3. Simultaneous T and x variation:")
    T_multi = np.array([298.15, 308.15, 318.15])
    x_multi = np.array([[0.3, 0.7], [0.5, 0.5], [0.7, 0.3]])
    
    try:
        gamma_both = unifac.unifac(mixture_composition, x_multi, T_multi)
        print(f"   Input: T = {T_multi}, x = {x_multi.tolist()}")
        print(f"   Output shape: {gamma_both.shape}")
        for i, (T_val, x_val, gamma_val) in enumerate(zip(T_multi, x_multi, gamma_both)):
            print(f"   T = {T_val} K, x = {x_val}: γ = {gamma_val}")
        print("   ✓ SUCCESS")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")

def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n=== Testing Error Handling ===\n")
    
    unifac = UNIFACMulti()
    mixture_composition = {
        'CH3':   [2, 2],
        'CH2':   [1, 5],
        'CH2NH': [1, 0],
    }
    
    # Test mismatched array lengths
    print("1. Mismatched array lengths:")
    try:
        T_mismatch = np.array([298.15, 308.15])  # Length 2
        x_mismatch = np.array([[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]])  # Length 3
        gamma = unifac.unifac(mixture_composition, x_mismatch, T_mismatch)
        print("   ❌ Should have failed but didn't!")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error type: {e}")
    
    # Test invalid composition (doesn't sum to 1)
    print("\n2. Invalid composition:")
    try:
        x_invalid = np.array([0.4, 0.7])  # Sums to 1.1
        T_valid = 308.15
        gamma = unifac.unifac(mixture_composition, x_invalid, T_valid)
        print("   ❌ Should have failed but didn't!")
    except ValueError as e:
        print(f"   ✓ Correctly caught error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error type: {e}")

if __name__ == "__main__":
    test_consistency()
    test_multi_functionality()
    test_error_handling()
    print("\n=== All tests completed ===")