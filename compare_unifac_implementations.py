#!/usr/bin/env python3
"""
Compare UNIFAC implementations to verify consistency.

This script tests two UNIFAC implementations:
1. unifac_smith2001.py - Smith et al. (2001) formulation with equation references
2. unifac_original.py - Original working formulation from utils/gc.py

Both should give identical results for the diethylamine-heptane system.
"""

import numpy as np
from unifac_smith2001 import UNIFACSmith2001
from unifac_original import UNIFACOriginal

def main():
    print("=== UNIFAC Implementation Comparison ===\n")
    
    # Test system: diethylamine + heptane
    # Diethylamine: (C2H5)2NH -> CH3(2) + CH2(1) + CH2NH(1)
    # Heptane: C7H16 -> CH3(2) + CH2(5)
    mixture_composition = {
        'CH3':   [2, 2],      # 2 CH3 groups in diethylamine, 2 in heptane
        'CH2':   [1, 5],      # 1 CH2 group in diethylamine, 5 in heptane  
        'CH2NH': [1, 0],      # 1 CH2NH group in diethylamine only
    }
    
    x = np.array([0.4, 0.6])  # 40% diethylamine, 60% heptane
    T = 308.15  # 35°C
    
    print(f"Test System: Diethylamine(40%) + Heptane(60%) at {T}K")
    print(f"Group composition: {mixture_composition}\n")
    
    # Expected results
    expected = [1.133, 1.047]
    print(f"Expected results: γ₁ = {expected[0]:.3f}, γ₂ = {expected[1]:.3f}\n")
    
    results = {}
    
    # Test Implementation 1: Smith2001
    print("1. Testing unifac_smith2001.py (Smith et al. formulation)...")
    try:
        unifac1 = UNIFACSmith2001()
        gamma1 = unifac1.unifac(mixture_composition, x, T)
        results['smith2001'] = gamma1
        print(f"   γ₁ = {gamma1[0]:.4f}, γ₂ = {gamma1[1]:.4f} ✓")
    except Exception as e:
        print(f"   Failed: {e} ❌")
        results['smith2001'] = None
    
    # Test Implementation 2: Original
    print("\n2. Testing unifac_original.py (Original gc.py formulation)...")
    try:
        unifac2 = UNIFACOriginal()
        gamma2 = unifac2.unifac(mixture_composition, x, T)
        results['original'] = gamma2
        print(f"   γ₁ = {gamma2[0]:.4f}, γ₂ = {gamma2[1]:.4f} ✓")
    except Exception as e:
        print(f"   Failed: {e} ❌")
        results['original'] = None
    
    # Comparison Analysis
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS")
    print("="*60)
    
    # Create comparison table
    print(f"{'Implementation':<20} {'γ₁ (Diethylamine)':<18} {'γ₂ (Heptane)':<15} {'Status'}")
    print("-" * 60)
    print(f"{'Expected':<20} {expected[0]:<18.3f} {expected[1]:<15.3f} {'Target'}")
    
    working_implementations = []
    
    for name, gamma in results.items():
        if gamma is not None:
            diff1 = abs(gamma[0] - expected[0])
            diff2 = abs(gamma[1] - expected[1])
            accuracy1 = (1 - diff1/expected[0]) * 100
            accuracy2 = (1 - diff2/expected[1]) * 100
            
            status = "✓ PASS" if (diff1 < 0.005 and diff2 < 0.005) else "⚠ INACCURATE"
            print(f"{name:<20} {gamma[0]:<18.4f} {gamma[1]:<15.4f} {status}")
            
            if status == "✓ PASS":
                working_implementations.append(name)
        else:
            print(f"{name:<20} {'FAILED':<18} {'FAILED':<15} {'❌ ERROR'}")
    
    # Cross-comparison between working implementations
    if len(working_implementations) > 1:
        print(f"\n📊 Cross-Implementation Consistency:")
        gamma_values = [results[impl] for impl in working_implementations]
        
        max_diff1 = max([abs(g[0] - gamma_values[0][0]) for g in gamma_values])
        max_diff2 = max([abs(g[1] - gamma_values[0][1]) for g in gamma_values])
        
        print(f"   Maximum difference in γ₁: {max_diff1:.6f}")
        print(f"   Maximum difference in γ₂: {max_diff2:.6f}")
        
        if max_diff1 < 1e-4 and max_diff2 < 1e-4:
            print(f"   🎯 All implementations are numerically consistent!")
        else:
            print(f"   ⚠️  Implementations show numerical differences")
    
    # Final Summary
    print(f"\n📋 SUMMARY:")
    print(f"   Working implementations: {len(working_implementations)}")
    print(f"   Failed implementations: {len(results) - len(working_implementations)}")
    
    if working_implementations:
        print(f"   ✅ Recommended: {working_implementations}")
    else:
        print(f"   ❌ No working implementations found!")

if __name__ == "__main__":
    main()