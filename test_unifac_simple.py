#!/usr/bin/env python3
"""
Simple test of UNIFAC bubble temperature calculation
"""
import numpy as np
import pandas as pd
from unifac_bubble_temperature import UnifacBubbleTCalculator

# Test with acetone and methanol
def test_simple():
    print("Testing simple UNIFAC calculation...")
    
    # Simple composition
    x = np.array([[0.5, 0.5]])  # 50% acetone, 50% methanol
    T0 = np.array([330.0])      # Initial temperature guess
    P = 0.101325                # Atmospheric pressure
    
    # Antoine parameters (from previous calculation)
    antoine_params = np.array([
        [8.9841, 8882.21, 547.58],  # Acetone
        [4.9021, 2031.46, 55.95]    # Methanol
    ])
    
    # Mixture groups
    mixture_data = {
        'CH3': [2, 1],      # 2 CH3 groups in acetone, 1 in methanol
        'CH3CO': [1, 0],    # 1 CH3CO group in acetone, 0 in methanol
        'OH': [0, 1]        # 0 OH groups in acetone, 1 in methanol
    }
    mixture = pd.DataFrame(mixture_data, index=['acetone', 'methanol'])
    
    print(f"Composition: {x[0]}")
    print(f"Initial T: {T0[0]} K")
    print(f"Pressure: {P} MPa")
    
    try:
        calculator = UnifacBubbleTCalculator()
        
        # Set up the model
        calculator.setup_model(x, T0, P, antoine_params, mixture)
        
        print("\nModel setup successful!")
        print("Attempting to solve...")
        
        # Solve with verbose output
        solution, model = calculator.solve(verbose=True)
        
        print(f"\nSolver status: {solution.solver.status}")
        print(f"Termination condition: {solution.solver.termination_condition}")
        
        if str(solution.solver.status) == 'ok':
            results = calculator.get_results()
            print(f"\nResults:")
            print(f"Temperature: {results['temperature'][0]:.2f} K")
            print(f"Vapor fractions: {results['vapour_fractions'][0]}")
            print(f"Activity coefficients: {results['activity_coefficients'][0]}")
        else:
            print("Solver failed!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()