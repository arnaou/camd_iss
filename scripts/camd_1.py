# imports
import sys
import os
import numpy as np
# Add the parent directory to path to access src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.molecules import StructuralValidator, generate_molecules
from src.pure import PropertyCalculator, AntoineCalculator
from src.data import load_unifac_data, check_unifac_parameters


# Step 1: Define binary system (compounds A and B to be separated)
mixture = {'acetone': {'CH3':1, 'CH3CO':1},
           'methanol': {'CH3OH':1}}

# step 2.a: validate that the binary mixture: structurally
print("Step 2.a: Validating molecular structures...")
validator = StructuralValidator()
binary_validation = validator.validate(mixture)
print(f"Structural validation of binary mixture: {binary_validation}")

# step 2.b: validate that the binary mixture: pure component properties
print("\nStep 2.b: Calculating pure component properties...")
prop_list = ['tc', 'pc', 'tb', 'ac']
pure_calc = PropertyCalculator()
pure_props = pure_calc.calculate(mixture, prop_list)

# step 2.c: adjust the poure component properties if needed (e.g. missing critical properties)
pure_props['acetone']['tc'] = 508.1
pure_props['acetone']['pc'] = 48.0
pure_props['acetone']['tb'] = 329.3

print(f"Pure component properties calculated/adjusted successfully:")
for compound, props in pure_props.items():
    print(f"  {compound}: Tc={props['tc']:.1f}K, Pc={props['pc']:.1f}bar, Tb={props['tb']:.1f}K, ac={props['ac']:.3f}")

# Calculate Antoine parameters
print("Calculating Antoine parameters...")
antoine_calc = AntoineCalculator()
antoine_params = {}
for compound, props in pure_props.items():
    compound_dict = {compound: mixture[compound]}  # Single compound dict
    results = antoine_calc.calculate_antoine_parameters(compound_dict, critical_properties=props)
    if results['termination_condition'] == 'optimal':
        antoine_params[compound] = {
            'A': results['A'],
            'B': results['B'], 
            'C': results['C']
        }
        print(f"  {compound}: A={results['A']:.3f}, B={results['B']:.1f}, C={results['C']:.1f}")
    else:
        print(f"  WARNING: Antoine parameter calculation failed for {compound}")

print(f"Antoine parameters calculated for {len(antoine_params)} compounds")


# step 3: check if all interaction parameters of the mixture groups are available
print("\nStep 3: Checking binary mixture UNIFAC parameters...")
compatibility_result = check_unifac_parameters(mixture=mixture)

if compatibility_result['compatible']:
    print("✅ Binary mixture has all required UNIFAC parameters")
else:
    print("❌ Binary mixture has missing UNIFAC parameters:")
    if compatibility_result['missing_rq']:
        print(f"   Missing R/Q parameters: {compatibility_result['missing_rq']}")
    if compatibility_result['missing_interactions_groups']:
        print(f"   Missing interaction groups: {compatibility_result['missing_interactions_groups']}")
    if compatibility_result['missing_interactions_main']:
        print(f"   Missing main group interactions: {compatibility_result['missing_interactions_main']}")
    print("⚠ Proceeding may lead to calculation errors")


# step 4: define the groups from which the solvents will be designed
design_groups = ["CH3", "CH2", "CH", "C", "OH", "CHO", "HCO", "CH3COO", "CH2COO", "HCOO", "CH3CO", "CH2O", "COOH", "COO",
                 'H2O', 'DMSO', 'DOH']

# step 5: define the structural constraints
min_number_of_groups = 3      # Adjust min total groups for solvents
max_number_of_groups = 6      # Adjust max total groups for solvents
min_func_groups = 1           # Adjust min functional groups
max_func_groups = 2           # Adjust max functional groups
molecular_type = 1            # 1=acyclic, 0=monocyclic, -1=bicyclic

# step 6: generate the solvents
print("\nStep 6: Generating solvent candidates...")

# Convert binary mixture to exclusion format for generator
binary_components = list(mixture.values())

# Generate molecules using the constraints defined in step 5
try:
    candidates = generate_molecules(
        design_groups=design_groups,
        min_groups=min_number_of_groups,
        max_groups=max_number_of_groups,
        min_func_groups=min_func_groups,
        max_func_groups=max_func_groups,
        molecular_type=molecular_type,
        max_solutions=1000,  # Generate up to 1000 candidates
        exclude_compounds=binary_components,  # Exclude A and B
        verbose=True
    )
    
    print(f"Generated {len(candidates)} valid solvent candidates")
    if candidates:
        print("Example candidates:")
        for i, candidate in enumerate(candidates[:5]):  # Show first 5
            groups_str = ", ".join([f"{k}:{v}" for k, v in candidate.items()])
            print(f"  Candidate {i+1}: {groups_str}")
    else:
        print("⚠ No valid candidates generated with current constraints")
        print("Consider adjusting structural constraints or design groups")
        
except Exception as e:
    print(f"❌ Molecule generation failed: {e}")
    candidates = []


# step 7: define pure component property constraints
pure_props_constraints = {
    'tb': [368.0, 500.0],      # Boiling point 368-500 K
    'tm': [0.0, 298.0],        # Melting point 298K+ (noß upper limit)ß
    'hv': [-9999.0, 67.0]      # Heat of vaporization range
}
# step 8: calculate the pure component properties of the solvents and then the antoine parameters
print("\nStep 8: Calculating pure component properties and Antoine parameters for solvent candidates...")

if candidates:
    # Create a dictionary format for solvent candidates
    solvent_compounds = {}
    for i, candidate in enumerate(candidates):
        candidate_name = f"solvent_{i+1}"
        solvent_compounds[candidate_name] = candidate
    
    print(f"Calculating properties for {len(solvent_compounds)} solvent candidates...")
    
    # Calculate properties for solvents
    prop_list = ['tc', 'pc', 'tb', 'tm', 'ac', 'hv']
    try:
        solvent_props = pure_calc.calculate(solvent_compounds, prop_list)
        
        print(f"✅ Successfully calculated properties for {len(solvent_props)} solvents")
        
        # Show example properties
        print("\nExample calculated properties:")
        for i, (name, props) in enumerate(list(solvent_props.items())[:3]):
            candidate_idx = int(name.split('_')[1]) - 1
            groups_str = ", ".join([f"{k}:{v}" for k, v in candidates[candidate_idx].items()])
            print(f"  {name} ({groups_str}):")
            print(f"    Tb={props.get('tb', 'N/A'):.1f}K, Tc={props.get('tc', 'N/A'):.1f}K, Pc={props.get('pc', 'N/A'):.1f}bar")
            
    except Exception as e:
        print(f"❌ Property calculation failed: {e}")
        solvent_props = {}

    # Calculate Antoine parameters for solvents with complete properties
    print("\nCalculating Antoine parameters for solvent candidates...")
    solvent_antoine_params = {}
    
    if solvent_props:
        for name, props in solvent_props.items():
            # Check if we have the required properties for Antoine calculation
            required_props = ['tc', 'pc', 'tb', 'ac']
            if all(prop in props and props[prop] is not None for prop in required_props):
                try:
                    candidate_idx = int(name.split('_')[1]) - 1
                    candidate_dict = {name: candidates[candidate_idx]}
                    
                    # Use the calculated properties for Antoine parameter estimation
                    results = antoine_calc.calculate_antoine_parameters(
                        candidate_dict, 
                        critical_properties=props
                    )
                    
                    if results['termination_condition'] == 'optimal':
                        solvent_antoine_params[name] = {
                            'A': results['A'],
                            'B': results['B'], 
                            'C': results['C']
                        }
                        print(f"  ✅ {name}: A={results['A']:.3f}, B={results['B']:.1f}, C={results['C']:.1f}")
                    else:
                        print(f"  ⚠ Antoine parameter calculation failed for {name}: {results['termination_condition']}")
                        
                except Exception as e:
                    print(f"  ❌ Error calculating Antoine parameters for {name}: {e}")
            else:
                missing_props = [prop for prop in required_props if prop not in props or props[prop] is None]
                print(f"  ⚠ {name}: Missing required properties for Antoine calculation: {missing_props}")
        
        print(f"\n✅ Antoine parameters calculated for {len(solvent_antoine_params)} out of {len(solvent_props)} solvents")
        
        if len(solvent_antoine_params) == 0:
            print("⚠ No solvents have complete Antoine parameters")
            print("  This may affect VLE calculations in subsequent steps")
    else:
        print("⚠ No solvent properties available for Antoine parameter calculation")
        
else:
    print("⚠ No candidates available for property calculation")
    solvent_props = {}
    solvent_antoine_params = {}

# step 9: screen the solvents based on pure component properties
print("\nStep 9: Screening solvents based on pure component properties...")

if solvent_props:
    screened_solvents = {}
    
    for name, props in solvent_props.items():
        # Check if properties meet constraints
        meets_constraints = True
        constraint_results = {}
        
        for prop, (min_val, max_val) in pure_props_constraints.items():
            prop_value = props.get(prop)
            if prop_value is not None:
                constraint_results[prop] = min_val <= prop_value <= max_val
                if not constraint_results[prop]:
                    meets_constraints = False
            else:
                constraint_results[prop] = False
                meets_constraints = False
        
        if meets_constraints:
            screened_solvents[name] = {
                'groups': candidates[int(name.split('_')[1]) - 1],
                'properties': props,
                'constraint_check': constraint_results
            }
    
    print(f"✅ {len(screened_solvents)} solvents passed property screening out of {len(solvent_props)}")
    
    # Show screened solvents
    if screened_solvents:
        print("\nScreened solvent candidates:")
        for i, (name, data) in enumerate(list(screened_solvents.items())[:5]):
            groups_str = ", ".join([f"{k}:{v}" for k, v in data['groups'].items()])
            props = data['properties']
            print(f"  {name}: {groups_str}")
            print(f"    Tb={props['tb']:.1f}K, Tm={props['tm']:.1f}K, Hv={props['hv']:.1f}kJ/mol")
    else:
        print("❌ No solvents passed the screening criteria")
        print("Consider relaxing property constraints or generating more candidates")

else:
    print("⚠ No solvent properties available for screening")
    screened_solvents = {}

# Step 10: Check UNIFAC parameters for screened solvents
print("\nStep 10: Checking UNIFAC parameters for screened solvents...")

if screened_solvents:
    compatible_solvents = {}
    incompatible_solvents = {}

    # Extract solvent candidates in the format expected by check_unifac_parameters
    solvent_candidates = [data['groups'] for data in screened_solvents.values()]
    solvent_names = list(screened_solvents.keys())

    print(f"Screening {len(solvent_candidates)} solvents for UNIFAC parameter availability...")

    # Check UNIFAC parameters for all solvents at once
    unifac_results = check_unifac_parameters(
        mixture=mixture,
        solvent_candidates=solvent_candidates
    )

    # Process results
    for i, solvent_name in enumerate(solvent_names):
        if i in unifac_results['compatible_solvents']:
            compatible_solvents[solvent_name] = screened_solvents[solvent_name]
            groups_str = ", ".join([f"{k}:{v}" for k, v in screened_solvents[solvent_name]['groups'].items()])
            print(f"✅ {solvent_name}: {groups_str}")
        else:
            incompatible_solvents[solvent_name] = screened_solvents[solvent_name]
            groups_str = ", ".join([f"{k}:{v}" for k, v in screened_solvents[solvent_name]['groups'].items()])
            print(f"❌ {solvent_name}: {groups_str}")

            # Show specific issues for this solvent
            if i in unifac_results['solvent_issues']:
                issues = unifac_results['solvent_issues'][i]
                if 'missing_rq' in issues:
                    print(f"    Missing R/Q parameters: {issues['missing_rq']}")
                if 'missing_interactions_groups' in issues:
                    print(f"    Missing interaction groups: {issues['missing_interactions_groups']}")
                if 'missing_interactions_main' in issues:
                    print(f"    Missing main group interactions: {issues['missing_interactions_main']}")

    print(f"\n✅ {len(compatible_solvents)} solvents have complete UNIFAC parameters")
    print(f"❌ {len(incompatible_solvents)} solvents have missing parameters")

    if compatible_solvents:
        print("\nCompatible solvents ready for VLE calculations:")
        for name, data in list(compatible_solvents.items())[:5]:  # Show first 5
            groups_str = ", ".join([f"{k}:{v}" for k, v in data['groups'].items()])
            print(f"  {name}: {groups_str}")
    else:
        print("⚠ No solvents have complete UNIFAC parameters for VLE calculations")
        print("  Consider:")
        print("  - Using different functional groups in design_groups")
        print("  - Relaxing structural constraints")
        print("  - Checking UNIFAC parameter database coverage")

else:
    print("⚠ No screened solvents available for UNIFAC parameter check")
    compatible_solvents = {}
    incompatible_solvents = {}

# step 11.a: calculate the VLE based on bubble-T of A+B+solvent system at the specified conditions (P= 1.01325)
# define the liquid composition and an intial guess for the temperature
n_data = 50
# x1-x3 binary (x2=0) 
x1_13 = np.linspace(0, 1, n_data)
x2_13 = np.zeros(n_data)
x3_13 = 1 - x1_13

# x2-x3 binary (x1=0)
x1_23 = np.zeros(n_data)
x2_23 = np.linspace(0, 1, n_data)
x3_23 = 1 - x2_23

# combine binary combinations AE & BE
x = np.vstack([
    np.column_stack([x1_13, x2_13, x3_13]),
    np.column_stack([x1_23, x2_23, x3_23])
])



# step 11.b: calculate mixture properties based on the VLE
# check for azeotropes

# calculate Gmix

# calculate G2mix

# calculate selectivity at infinite dilution

# calculate capacity at infinite dilution

# calculate relative volatility at infinite dilution

# step 12: screen based on mixture properties


# step 13: perform the iss-methodology


# step 14: print the results