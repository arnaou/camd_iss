""" script for performing binary vle of mixtures"""

# imports to enable the notebook to access the src folder
import sys
from pathlib import Path

# Add project root to path (parent of scripts/ directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
# import the necessary packages
from src.molecules import StructuralValidator
from src.pure import PropertyCalculator
from src.pure import AntoineCalculator
from src.data import load_unifac_data
from src.mixture.unifac.pyomo_vle import pyo_vle
import numpy as np
import matplotlib.pyplot as plt

##################################
# step1: define the compounds
#################################
mixture = {'acetone': {'CH3':1, 'CH3CO':1},
           'methanol': {'CH3OH':1},
           #'water': {'H2O':1}
           #'E': {'CH2': 3, 'HCO': 1, 'CH3COO': 1}
           'E': {'CH2':1, 'HCO':2}
           #'E': {'CH3':1, 'CH2':2, 'CH3CO':1, 'CH2O':1}
           #'E': {'CH3':3, 'CH':1, 'CH2COO':1}
           }
# validate the structures
validator = StructuralValidator()
validator.validate(mixture)



###############################
# step2: calculate the required properties
###############################
prop_list = ['tc', 'pc', 'tb', 'ac']
pure_calc = PropertyCalculator()
pure_props = pure_calc.calculate(mixture, prop_list)
# fix the properties of acetone
pure_props['acetone']['tc'] = 508.1
pure_props['acetone']['pc'] = 48.0
pure_props['acetone']['tb'] = 329.3
pure_props

###############################
# step3: calculate the Antoine coefficients
###############################
antoine_params_ini = {}
antoine_params = {}
antoine_calc = AntoineCalculator()
for compound, props in pure_props.items():
    results = antoine_calc.calculate_antoine_parameters(mixture, critical_properties=props, method='acentric_factor')
    print('optimization results was', results['termination_condition'] )
    antoine_params[compound] = {}
    antoine_params_ini[compound] = {}
    # antoine_params_ini[compound]['A'] = results['A0']
    # antoine_params_ini[compound]['B'] = results['B0']
    # antoine_params_ini[compound]['C'] = results['C0']
    antoine_params[compound]['A'] = results['A']
    antoine_params[compound]['B'] = results['B']
    antoine_params[compound]['C'] = results['C']
print(antoine_params_ini)
print(antoine_params)

###############################
# step4: perform the vle
###############################
# define number of data points
n_data = 100
# set composition
# generate all binary combinations for ternary system
# x1-x2 binary (x3=0)
x0 = 0.0
xf = 1-x0
x1_12 = np.linspace(x0, xf, n_data)
x2_12 = 1 - x1_12
x3_12 = np.zeros(n_data)

# x1-x3 binary (x2=0)
x1_13 = np.linspace(x0, xf, n_data)
x2_13 = np.zeros(n_data)
x3_13 = 1 - x1_13

# x2-x3 binary (x1=0)
x1_23 = np.zeros(n_data)
x2_23 = np.linspace(x0, xf, n_data)
x3_23 = 1 - x2_23

# combine all binary combinations
x = np.vstack([
    np.column_stack([x1_12, x2_12, x3_12]),
    np.column_stack([x1_13, x2_13, x3_13]),
    np.column_stack([x1_23, x2_23, x3_23])
])
# provide nitial guess for temperature
# Extract boiling points for each compound
tb_values = [pure_props[compound]['tb'] for compound in mixture.keys()]
tb1, tb2, tb3 = tb_values  # assuming 3 components
# Create T0 for each binary combination
# x1-x2 binary: T ranges from tb2 (x1=0, x2=1) to tb1 (x1=1, x2=0)
T0_12 = np.linspace(tb2, tb1, n_data)

# x1-x3 binary: T ranges from tb3 (x1=0, x3=1) to tb1 (x1=1, x3=0)
T0_13 = np.linspace(tb3, tb1, n_data)

# x2-x3 binary: T ranges from tb3 (x2=0, x3=1) to tb2 (x2=1, x3=0)
T0_23 = np.linspace(tb3, tb2, n_data)

# Combine all T0 values
T0 = np.concatenate([T0_12, T0_13, T0_23])
# Load all UNIFAC parameters
rk_data, qk_data, interaction_data = load_unifac_data()
# perform the VLE calculation
results, model = pyo_vle(x, T0, mixture, rk_data, qk_data, interaction_data, antoine_params, verbose=True)


###############################
# step5: lets plot the results in T-xy diagram and inspect the gamma values
###############################


# Extract results
# retrieving the solutions
gamma_pyo = np.empty_like(x)
T_pyo = np.empty_like(T0)
y_pyo = np.empty_like(x)
psat_pyo = np.empty_like(x)

for i in model.gamma.keys():
    gamma_pyo[i] = model.gamma[i].value

for i in model.T.keys():
    T_pyo[i] = model.T[i].value

for i in model.y.keys():
    y_pyo[i] = model.y[i].value

for i in model.psat.keys():
    psat_pyo[i] = model.psat[i].value

# Create T-xy plots for all three binary combinations
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Component 1-2 diagram (first 100 points)
axes[0].plot(y_pyo[:n_data, 0], T_pyo[:n_data], 'b-', label='Vapor')
axes[0].plot(x[:n_data, 0], T_pyo[:n_data], 'r-', label='Liquid')
axes[0].set_ylabel('Temperature (K)')
axes[0].set_xlabel('Component 1 Fraction')
axes[0].set_title('T-xy Diagram: Component 1-2')
axes[0].legend()
axes[0].grid()

# Component 1-3 diagram (points 100-200)
axes[1].plot(y_pyo[n_data:2*n_data, 0], T_pyo[n_data:2*n_data], 'b-', label='Vapor')
axes[1].plot(x[n_data:2*n_data, 0], T_pyo[n_data:2*n_data], 'r-', label='Liquid')
axes[1].set_ylabel('Temperature (K)')
axes[1].set_xlabel('Component 1 Fraction')
axes[1].set_title('T-xy Diagram: Component 1-3')
axes[1].legend()
axes[1].grid()

# Component 2-3 diagram (points 200-300)
axes[2].plot(y_pyo[2*n_data:3*n_data, 1], T_pyo[2*n_data:3*n_data], 'b-', label='Vapor')
axes[2].plot(x[2*n_data:3*n_data, 1], T_pyo[2*n_data:3*n_data], 'r-', label='Liquid')
axes[2].set_ylabel('Temperature (K)')
axes[2].set_xlabel('Component 2 Fraction')
axes[2].set_title('T-xy Diagram: Component 2-3')
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()