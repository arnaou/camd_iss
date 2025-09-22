# imports
import sys
import os
import numpy as np
from datetime import datetime
# Add the parent directory to path to access src module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.molecules import StructuralValidator, generate_molecules
from src.pure import PropertyCalculator, AntoineCalculator
from src.data import load_unifac_data, check_unifac_parameters
from src.mixture.unifac import pyo_vle
from src.mixture.unifac.pyomo_iss import pyo_iss
import pyomo.environ as pyo


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
prop_list = ['tc', 'pc', 'tb', 'ac', 'hv']
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
    results = antoine_calc.calculate_antoine_parameters(compound_dict, critical_properties=props, method='enthalpy_vap')
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
    
    # # Additional constraint: Filter out compounds where Tb > 0.7*Tc
    # print("\nFiltering compounds based on Tb/Tc ratio...")

    # eliminated_compounds = {}
    # valid_compounds = {}

    # if solvent_props:
    #     for name, props in solvent_props.items():
    #         tb = props.get('tb')
    #         tc = props.get('tc')

    #         if tb is not None and tc is not None and tc > 0:
    #             tb_tc_ratio = tb / tc
    #             candidate_idx = int(name.split('_')[1]) - 1
    #             groups_str = ", ".join([f"{k}:{v}" for k, v in candidates[candidate_idx].items()])

    #             if tb > 0.7 * tc:
    #                 eliminated_compounds[name] = {
    #                     'groups': candidates[candidate_idx],
    #                     'properties': props,
    #                     'tb_tc_ratio': tb_tc_ratio,
    #                     'reason': f'Tb ({tb:.1f}K) > 0.7*Tc ({0.7*tc:.1f}K)'
    #                 }
    #             else:
    #                 valid_compounds[name] = props
    #         else:
    #             # If missing critical properties, keep for now but flag
    #             valid_compounds[name] = props

    #     print(f"Eliminated {len(eliminated_compounds)} compounds where Tb > 0.7*Tc")
    #     print(f"Retained {len(valid_compounds)} compounds for further screening")

    #     # Print overview of eliminated compounds
    #     if eliminated_compounds:
    #         print("\nEliminated compounds (Tb > 0.7*Tc):")
    #         for name, data in eliminated_compounds.items():
    #             groups_str = ", ".join([f"{k}:{v}" for k, v in data['groups'].items()])
    #             print(f"  {name}: {groups_str}")
    #             print(f"    {data['reason']}, Ratio: {data['tb_tc_ratio']:.3f}")

    #     # Update solvent_props to only include valid compounds
    #     solvent_props = valid_compounds
    # else:
    #     print("No solvent properties available for Tb/Tc filtering")

    # Calculate Antoine parameters for solvents with complete properties
    print("\nCalculating Antoine parameters for solvent candidates...")
    solvent_antoine_params = {}
    
    if solvent_props:
        for name, props in solvent_props.items():
            # Check if we have the required properties for Antoine calculation
            required_props = ['tc', 'pc', 'tb', 'ac', 'hv']
            if all(prop in props and props[prop] is not None for prop in required_props):
                try:
                    candidate_idx = int(name.split('_')[1]) - 1
                    candidate_dict = {name: candidates[candidate_idx]}
                    
                    # Use the calculated properties for Antoine parameter estimation
                    results = antoine_calc.calculate_antoine_parameters(
                                                                        candidate_dict, 
                                                                        critical_properties=props,  
                                                                        method='enthalpy_vap'
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
        
        # Sanity check: Tb should be below Tc
        tb_value = props.get('tb')
        tc_value = props.get('tc')
        if tb_value is not None and tc_value is not None:
            if tb_value >= tc_value:
                print(f"  ❌ {name}: Tb ({tb_value:.1f}K) >= Tc ({tc_value:.1f}K) - discarded")
                meets_constraints = False

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

# Step 11: VLE calculations for each compatible solvent
print("\nStep 11: Performing VLE calculations for compatible solvents...")

if compatible_solvents:
    # Load UNIFAC data for VLE calculations
    rk_data, qk_data, interaction_data = load_unifac_data()

    # Operating pressure (bar)
    P = 1.01325

    # Number of composition points for each binary edge
    n_data = 50

    vle_results = {}
    failed_solvents = []

    # Get binary mixture boiling points
    binary_compounds = list(mixture.keys())
    tb_A = pure_props[binary_compounds[0]]['tb']
    tb_B = pure_props[binary_compounds[1]]['tb']

    successful_solvents = []

    # DEBUG: Only process first 10 solvents for testing: only 10 first
    solvent_items = list(compatible_solvents.items())#[:10]
    print(f"DEBUG: Testing only first {len(solvent_items)} solvents out of {len(compatible_solvents)} total")

    for solvent_name, solvent_data in solvent_items:
        groups_str = ", ".join([f"{k}:{v}" for k, v in solvent_data['groups'].items()])
        print(f"Calculating VLE for {solvent_name} ({groups_str})...")

        try:
            # Create ternary mixture: A + B + Solvent
            ternary_mixture = dict(mixture)
            ternary_mixture[solvent_name] = solvent_data['groups']

            # Get solvent boiling point and Antoine parameters
            if solvent_name not in solvent_props or solvent_name not in solvent_antoine_params:
                failed_solvents.append(solvent_name)
                print(f"  {solvent_name}: Missing properties - Failed")
                continue

            tb_E = solvent_props[solvent_name]['tb']
            ternary_antoine = dict(antoine_params)
            ternary_antoine[solvent_name] = solvent_antoine_params[solvent_name]

            # Define composition grids for ternary VLE (A-E and B-E binary edges)
            # A-E binary edge (B=0): x1 + x3 = 1
            x1_ae = np.linspace(0.00, 1.00, n_data)
            x2_ae = np.zeros(n_data)
            x3_ae = 1 - x1_ae

            # B-E binary edge (A=0): x2 + x3 = 1
            x1_be = np.zeros(n_data)
            x2_be = np.linspace(0.00, 1.00, n_data)
            x3_be = 1 - x2_be

            x_all = np.vstack([
                np.column_stack([x1_ae, x2_ae, x3_ae]),  # A-E edge
                np.column_stack([x1_be, x2_be, x3_be])   # B-E edge
            ])

            # Temperature initialization based on pure component boiling points
            # A-E edge: When A=0,E=1 → T=Tb(E), When A=1,E=0 → T=Tb(A)
            T0_ae = tb_E * (1 - x1_ae) + tb_A * x1_ae
            # B-E edge: When B=0,E=1 → T=Tb(E), When B=1,E=0 → T=Tb(B)
            T0_be = tb_E * (1 - x2_be) + tb_B * x2_be
            T_guess = np.concatenate([T0_ae, T0_be])

            # Perform VLE calculation using pyo_vle
            vle_solution, vle_model = pyo_vle(
                x_all, T_guess, ternary_mixture,
                rk_data, qk_data, interaction_data, ternary_antoine, verbose=False
            )

            # Extract results
            if vle_solution is not None and vle_model is not None:
                solv_stat = str(vle_solution.solver.status)
                term_cond = str(vle_solution.solver.termination_condition)

                if solv_stat != 'ok' or term_cond != 'optimal':
                    solver_success = False
                    print(f"  {solvent_name}: Solver failed - Status: {solv_stat}, Condition: {term_cond}")
                else:
                    solver_success = True
            else:
                solver_success = False

            if solver_success:
                # Extract results
                n_points = len(x_all)
                T_bubble = np.zeros(n_points)
                y_vapor = np.zeros((n_points, 3))
                gamma = np.zeros((n_points, 3))
                psat = np.zeros((n_points, 3))

                for i in range(n_points):
                    T_bubble[i] = vle_model.T[i].value if vle_model.T[i].value else T_guess[i]
                    for j in range(3):
                        y_vapor[i, j] = vle_model.y[i, j].value #if vle_model.y[i, j].value else 0.0
                        gamma[i, j] = vle_model.gamma[i, j].value #if vle_model.gamma[i, j].value else 1.0
                        psat[i, j] = vle_model.psat[i, j].value #if vle_model.psat[i, j].value else 0.0

                vle_results[solvent_name] = {
                    'x': x_all,
                    'T': T_bubble,
                    'y': y_vapor,
                    'gamma': gamma,
                    'compound_names': list(ternary_mixture.keys())
                }
                successful_solvents.append(solvent_name)
                print(f"  {solvent_name}: Solver succeeded")
            else:
                failed_solvents.append(solvent_name)
                print(f"  {solvent_name}: Solver failed")

        except Exception:
            failed_solvents.append(solvent_name)
            print(f"  {solvent_name}: Solver failed")

        # Calculate mixture properties from the vle_model
        P0 = 1.01325  # Reference pressure (bar)
        # selectivity at infinite dilution
        vle_results[solvent_name]['S'] = gamma[0, 0] / gamma[0, 1]
        # capacity at infinite dilution
        vle_results[solvent_name]['C'] = 1 / gamma[0, 1]
        # SxC product
        vle_results[solvent_name]['SxC'] = vle_results[solvent_name]['S'] * vle_results[solvent_name]['C']
        # relative volatility at infinite dilution
        vle_results[solvent_name]['alpha'] = gamma[0, 0]/psat[0, 0] * gamma[0, 1]/psat[0, 1]
        # azeotrope detection AE
        vle_results[solvent_name]['az_AE'] = np.log(gamma[0,0] * psat[0,0]/P0) * (-np.log(gamma[n_data-1,2] * psat[n_data-1,2]/P0))
        # azeotrope detection BE
        vle_results[solvent_name]['az_BE'] = np.log(gamma[n_data,1] * psat[n_data,1]/P0) * (-np.log(gamma[-1,2] * psat[-1,2]/P0))
        # deltaG mix: TODO: fix it so it does AE and then BE cause otherwise it does it wrongly. lets valiudate in vle ternary
        vle_results[solvent_name]['G_mix'] = (np.sum(x_all * np.log(gamma), axis=1) + np.sum(np.ma.log(x_all).filled(0) * x_all, axis=1))
        vle_results[solvent_name]['G_mix_AE'] = vle_results[solvent_name]['G_mix'][:n_data]
        vle_results[solvent_name]['G_mix_BE'] = vle_results[solvent_name]['G_mix'][n_data:]
        # delta2G mix
        vle_results[solvent_name]['G2_mix_AE'] = np.diff(np.diff(vle_results[solvent_name]['G_mix_AE']))
        vle_results[solvent_name]['G2_mix_BE'] = np.diff(np.diff(vle_results[solvent_name]['G_mix_BE']))
    # VLE Overview
    print(f"\nVLE Overview:")
    print(f"  Successful: {successful_solvents}")
    print(f"  Failed: {failed_solvents}")
    print(f"  Success rate: {len(successful_solvents)}/{len(compatible_solvents)} ({len(successful_solvents)/len(compatible_solvents)*100:.0f}%)")
    # print vle_results for all successful solvents
    if successful_solvents:
        print(f"\nVLE results for all {len(successful_solvents)} successful solvents:")
        for solvent in successful_solvents:
            print(f"\n{solvent}:")
            print(f"  Groups: {compatible_solvents[solvent]['groups']}")
            print(f"  Compounds: {vle_results[solvent]['compound_names']}")
            print(f"  S: {vle_results[solvent]['S']:.3f}, C: {vle_results[solvent]['C']:.3f}, SxC: {vle_results[solvent]['SxC']:.3f}, alpha: {vle_results[solvent]['alpha']:.3f}, az_AE: {vle_results[solvent]['az_AE']:.3f}, az_BE: {vle_results[solvent]['az_BE']:.3f}")
else:
    print("⚠ No compatible solvents available for VLE calculations")
    vle_results = {}
    failed_solvents = []

# step 12: screen based on mixture properties
print("\nStep 12: Screening solvents based on mixture properties...")

if vle_results:
    screened_solvents = {}
    rejected_solvents = {}

    for solvent_name, results in vle_results.items():
        if solvent_name in successful_solvents:
            az_AE = results['az_AE']
            az_BE = results['az_BE']
            G_mix_AE = results['G_mix_AE']
            G_mix_BE = results['G_mix_BE']
            G2_mix_AE = results['G2_mix_AE']
            G2_mix_BE = results['G2_mix_BE']
            

            # Check constraints: az_AE > 0, az_BE > 0, G_mix <= 0, G2_mix >= 0
            az_AE_valid = az_AE > 0
            az_BE_valid = az_BE > 0
            G_mix_AE_valid = np.all(G_mix_AE[1:-1] <= 0)
            G_mix_BE_valid = np.all(G_mix_BE[1:-1] <= 0)
            G_mix_valid = G_mix_AE_valid and G_mix_BE_valid
            G2_mix_AE_valid = np.all(G2_mix_AE[1:-1] >= 0)
            G2_mix_BE_valid = np.all(G2_mix_BE[1:-1] >= 0)
            G2_mix_valid = G2_mix_AE_valid and G2_mix_BE_valid
            S_valid = results['S'] >= 2.0  # Example: selectivity greater than 2

            if az_AE_valid and az_BE_valid and G_mix_valid and G2_mix_valid and S_valid:
                screened_solvents[solvent_name] = results
                status = "✓ PASSED"
            else:
                rejected_solvents[solvent_name] = {
                    'results': results,
                    'reasons': []
                }
                if not az_AE_valid:
                    rejected_solvents[solvent_name]['reasons'].append(f"az_AE={az_AE:.3f} ≤ 0")
                if not az_BE_valid:
                    rejected_solvents[solvent_name]['reasons'].append(f"az_BE={az_BE:.3f} ≤ 0")
                if not G_mix_valid:
                    rejected_solvents[solvent_name]['reasons'].append(f"G_mix has values > 0")
                if not G2_mix_valid:
                    rejected_solvents[solvent_name]['reasons'].append(f"G2_mix has values < 0")
                status = "✗ REJECTED: " + "; ".join(rejected_solvents[solvent_name]['reasons'])

            print(f"  {solvent_name}: {status}")

    print(f"\nMixture Properties Screening Results:")
    print(f"  Passed screening: {len(screened_solvents)}/{len(successful_solvents)}")
    print(f"  Rejected: {len(rejected_solvents)}")

    if screened_solvents:
        print(f"  Solvents passing all constraints: {list(screened_solvents.keys())}")
    else:
        print("  ⚠ No solvents passed all mixture property constraints")
else:
    print("  ⚠ No VLE results available for screening")
    screened_solvents = {}
    rejected_solvents = {}


# step 13: perform the iss-methodology
print("\nStep 13: Performing ISS methodology on screened solvents...")

if screened_solvents:
    iss_results = {}

    # Ensure UNIFAC data is loaded (should already be loaded from step 11)
    if 'rk_data' not in locals():
        rk_data, qk_data, interaction_data = load_unifac_data()

    # Define initial conditions for ISS
    boiling_points = [pure_props[compound]['tb'] for compound in mixture.keys()]
    T_min = min(boiling_points)
    T_max = max(boiling_points)
    T0 = np.array([(T_max + T_min) / 2, (T_max + T_min) / 2])  # Two temperatures for two data points
    x_iso = 0.75
    xA = 0.5
    x0 = np.array([[x_iso, 0.0, 1-x_iso],    # Point 0: iso-volatility point (A, B, E)
                   [xA, 0.0, 1-xA]])          # Point 1: minimum point (A, B, E)

    print(f"Running ISS calculations for {len(screened_solvents)} screened solvents...")

    for solvent_name in screened_solvents.keys():
        groups_str = ", ".join([f"{k}:{v}" for k, v in compatible_solvents[solvent_name]['groups'].items()])
        print(f"\nCalculating ISS for {solvent_name} ({groups_str})...")

        try:
            # Create ternary mixture: A + B + Solvent
            ternary_mixture = dict(mixture)
            ternary_mixture[solvent_name] = compatible_solvents[solvent_name]['groups']

            # Get ternary Antoine parameters
            ternary_antoine = dict(antoine_params)
            ternary_antoine[solvent_name] = solvent_antoine_params[solvent_name]

            # Perform ISS calculation
            results, model = pyo_iss(x0, T0, ternary_mixture, rk_data, qk_data, interaction_data, ternary_antoine, verbose=False)

            # Check solver status
            if results is not None and model is not None:
                solv_stat = str(results.solver.status)
                term_cond = str(results.solver.termination_condition)

                if solv_stat != 'ok' or term_cond != 'optimal':
                    solver_success = False
                    print(f"  {solvent_name}: Solver failed - Status: {solv_stat}, Condition: {term_cond}")
                else:
                    solver_success = True
            else:
                solver_success = False

            if solver_success:
                # Extract ED and R values
                ED = pyo.value(model.obj)
                R_value = pyo.value(model.R)

                iss_results[solvent_name] = {
                    'ED': ED,
                    'R': R_value,
                    'groups': compatible_solvents[solvent_name]['groups']
                }

                print(f"  ED (E/D): {ED:.6f}")
                print(f"  R value: {R_value:.6f}")
            else:
                iss_results[solvent_name] = {
                    'ED': None,
                    'R': None,
                    'groups': compatible_solvents[solvent_name]['groups'],
                    'error': 'Solver failed'
                }

        except Exception as e:
            print(f"  ❌ ISS calculation failed for {solvent_name}: {e}")
            iss_results[solvent_name] = {
                'ED': None,
                'R': None,
                'groups': compatible_solvents[solvent_name]['groups'],
                'error': str(e)
            }

    print(f"\n✅ ISS calculations completed for {len(iss_results)} solvents")

    # Show summary of ISS results
    successful_iss = {k: v for k, v in iss_results.items() if v['ED'] is not None}
    failed_iss = {k: v for k, v in iss_results.items() if v['ED'] is None}

    if successful_iss:
        print(f"\nISS Results Summary ({len(successful_iss)} successful):")
        for solvent_name, data in successful_iss.items():
            groups_str = ", ".join([f"{k}:{v}" for k, v in data['groups'].items()])
            print(f"  {solvent_name}: ED={data['ED']:.6f}, R={data['R']:.6f} ({groups_str})")

    if failed_iss:
        print(f"\nFailed ISS calculations ({len(failed_iss)}):")
        for solvent_name, data in failed_iss.items():
            print(f"  {solvent_name}: {data['error']}")

else:
    print("⚠ No screened solvents available for ISS calculations")
    iss_results = {}

# step 14: print the results
