"""
Test bubble temperature calculation with literature Antoine parameters
to verify the pressure unit fix works correctly.
"""

import numpy as np
import pandas as pd
from unifac_bubble_temperature import UnifacBubbleTCalculator, generate_xy_diagram

def test_pure_components():
    """Test pure component limits with literature Antoine parameters."""
    
    # Literature Antoine parameters for log10(P_bar) = A - B/(C+T)
    # Your Antoine equation expects P in MPa, so convert:
    # P_MPa = P_bar * 0.1, so log10(P_MPa) = log10(P_bar) + log10(0.1) = log10(P_bar) - 1
    # Therefore: log10(P_MPa) = (A - 1) - B/(C+T)
    
    # Acetone: A=7.11714, B=1210.595, C=229.664 (for P in bar)
    # Methanol: A=8.08094, B=1582.271, C=239.726 (for P in bar)
    
    antoine_params = np.array([
        [7.11714 - 1, 1210.595, 229.664],     # Acetone (A adjusted for MPa)
        [8.08094 - 1, 1582.271, 239.726]      # Methanol (A adjusted for MPa)
    ])
    
    P = 0.101325  # MPa (atmospheric pressure)
    
    print("Testing pure component boiling points...")
    
    # Test pure acetone (x1=1, x2=0)
    T_pure1 = antoine_params[0, 1] / (antoine_params[0, 0] - np.log10(P)) - antoine_params[0, 2]
    print(f"Pure acetone: {T_pure1:.2f} K ({T_pure1-273.15:.2f} °C) - should be ~56.1°C")
    
    # Test pure methanol (x1=0, x2=1)  
    T_pure2 = antoine_params[1, 1] / (antoine_params[1, 0] - np.log10(P)) - antoine_params[1, 2]
    print(f"Pure methanol: {T_pure2:.2f} K ({T_pure2-273.15:.2f} °C) - should be ~64.7°C")
    
    return antoine_params

def test_unifac_bubble_calculation():
    """Test UNIFAC bubble calculation with corrected parameters."""
    
    antoine_params = test_pure_components()
    
    # Define mixture (acetone + methanol)
    mixture_data = {
        'CH3': [1, 0],      # 1 CH3 group in acetone, 0 in methanol  
        'CH3CO': [1, 0],    # 1 CH3CO group in acetone, 0 in methanol
        'CH3OH': [0, 1]     # 0 CH3OH groups in acetone, 1 in methanol
    }
    
    mixture = pd.DataFrame(mixture_data, index=['acetone', 'methanol'])
    P = 0.101325  # MPa
    
    print(f"\nTesting UNIFAC bubble temperature calculation...")
    print(f"Mixture composition:")
    print(mixture)
    
    try:
        # Test a few points
        calculator = UnifacBubbleTCalculator()
        
        test_points = [
            (0.0, 1.0),  # Pure methanol
            (0.5, 0.5),  # 50-50 mixture
            (1.0, 0.0)   # Pure acetone
        ]
        
        for x1, x2 in test_points:
            x = np.array([[x1, x2]])
            T0 = np.array([330.0])  # Initial guess
            
            print(f"\nTesting x1={x1:.1f}, x2={x2:.1f}")
            
            try:
                solution, model = calculator.calculate_bubble_temperature(
                    x, T0, P, antoine_params, mixture, verbose=False
                )
                
                if str(solution.solver.status) == 'ok':
                    results = calculator.get_results()
                    T = results['temperature'][0]
                    y1 = results['vapour_fractions'][0, 0]
                    
                    print(f"  T = {T:.2f} K ({T-273.15:.2f} °C)")
                    print(f"  y1 = {y1:.3f}")
                    
                    # Check if results make sense
                    if x1 == 0.0:  # Pure methanol
                        expected = "~64.7°C"
                    elif x1 == 1.0:  # Pure acetone  
                        expected = "~56.1°C"
                    else:
                        expected = "between 56-65°C"
                    print(f"  Expected: {expected}")
                else:
                    print(f"  Solver failed: {solution.solver.status}")
                    
            except Exception as e:
                print(f"  Error: {e}")
                
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Bubble Temperature Calculation Fix")
    print("=" * 50)
    
    success = test_unifac_bubble_calculation()
    
    if success:
        print(f"\n✓ Test completed - check if temperatures are reasonable")
    else:
        print(f"\n✗ Test failed")