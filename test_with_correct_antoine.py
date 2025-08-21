"""
Test bubble temperature calculation with correct literature Antoine parameters
to verify the pressure unit fix and UNIFAC implementation work correctly.
"""

import numpy as np
import pandas as pd
from unifac_bubble_temperature import UnifacBubbleTCalculator

def test_with_correct_antoine_params():
    """Test with literature Antoine parameters properly converted to MPa."""
    
    # Literature Antoine parameters: log10(P_bar) = A - B/(C+T)
    # Acetone: A=7.11714, B=1210.595, C=229.664 (P in bar)
    # Methanol: A=8.08094, B=1582.271, C=239.726 (P in bar)
    
    # Convert to log10(P_MPa) = A - B/(C+T):
    # P_MPa = P_bar * 0.1, so log10(P_MPa) = log10(P_bar) + log10(0.1) = log10(P_bar) - 1
    # Therefore A_MPa = A_bar - 1
    
    antoine_params = np.array([
        [7.11714 - 1, 1210.595, 229.664],     # Acetone (converted to MPa)
        [8.08094 - 1, 1582.271, 239.726]      # Methanol (converted to MPa)
    ])
    
    P = 0.101325  # MPa (atmospheric pressure)
    
    print("Testing pure component boiling points with corrected Antoine parameters...")
    
    # Test pure acetone (x1=1, x2=0)
    T_pure1 = antoine_params[0, 1] / (antoine_params[0, 0] - np.log10(P)) - antoine_params[0, 2]
    print(f"Pure acetone: {T_pure1:.2f} K ({T_pure1-273.15:.2f} °C) - should be ~56.1°C")
    
    # Test pure methanol (x1=0, x2=1)  
    T_pure2 = antoine_params[1, 1] / (antoine_params[1, 0] - np.log10(P)) - antoine_params[1, 2]
    print(f"Pure methanol: {T_pure2:.2f} K ({T_pure2-273.15:.2f} °C) - should be ~64.7°C")
    
    if abs(T_pure1 - 329.2) < 1 and abs(T_pure2 - 337.9) < 1:
        print("✓ Antoine parameters are correct!")
        
        # Define mixture (acetone + methanol)
        mixture_data = {
            'CH3': [1, 0],      # 1 CH3 group in acetone, 0 in methanol  
            'CH3CO': [1, 0],    # 1 CH3CO group in acetone, 0 in methanol
            'CH3OH': [0, 1]     # 0 CH3OH groups in acetone, 1 in methanol
        }
        
        mixture = pd.DataFrame(mixture_data, index=['acetone', 'methanol'])
        
        print(f"\nTesting UNIFAC bubble temperature calculation...")
        
        try:
            # Test pure component limits
            calculator = UnifacBubbleTCalculator()
            
            test_cases = [
                (1.0, 0.0, "Pure acetone"),
                (0.0, 1.0, "Pure methanol"),
                (0.5, 0.5, "50-50 mixture")
            ]
            
            for x1, x2, description in test_cases:
                x = np.array([[x1, x2]])
                T0 = np.array([330.0])  # Initial guess
                
                print(f"\n{description} (x1={x1}, x2={x2}):")
                
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
                        if x1 == 1.0:  # Pure acetone
                            if abs(T - 329.2) < 2:
                                print(f"  ✓ Temperature close to expected ~56.1°C")
                            else:
                                print(f"  ✗ Temperature should be ~56.1°C")
                        elif x1 == 0.0:  # Pure methanol
                            if abs(T - 337.9) < 2:
                                print(f"  ✓ Temperature close to expected ~64.7°C")
                            else:
                                print(f"  ✗ Temperature should be ~64.7°C")
                        else:
                            if 329 < T < 338:
                                print(f"  ✓ Temperature between pure components")
                            else:
                                print(f"  ✗ Temperature should be between 56-65°C")
                                
                    else:
                        print(f"  ✗ Solver failed: {solution.solver.status}")
                        
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    
        except Exception as e:
            print(f"Error: {e}")
            
    else:
        print("✗ Antoine parameters still incorrect")

if __name__ == "__main__":
    print("Testing Bubble Temperature with Correct Antoine Parameters")
    print("=" * 60)
    
    test_with_correct_antoine_params()