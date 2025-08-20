#!/usr/bin/env python3
"""
Test case 1: CAMD implementation testing script

This script demonstrates the basic workflow for using the CAMD class:
1. Initialize CAMD instance
2. Define binary system (compounds A and B to be separated)
3. Set design groups for solvent candidate generation
4. Add structural constraints
5. Generate solvent candidates using MILP
6. Alternative: Set predefined solvent candidates
7. Run screening calculations

Fill in the empty lists/dicts with actual values for testing.
"""

from src.camd import CAMD

def main():
    print("=== CAMD Implementation Test Case 1 ===\n")
    
    # Step 1: Initialize CAMD instance
    print("1. Initializing CAMD...")
    camd = CAMD(
        title="Test Case 1 - Binary Separation",
        type="VLE",
        output_dir="./results/case_1",
        pure_method="Constantinou-Gani",
        mixture_method="UNIFAC-LY",
        missing_group_action="remove_compounds"  # Options: "remove_compounds" or "set_zero"
    )
    
    # Step 2: Define binary system (compounds A and B to be separated)
    print("\n2. Defining binary system to be separated...")
    compound_A = {'CH3':1, 'CH3CO':1}  # Acetone
    compound_B = {'CH3OH':1}  # Methanol
    
    if compound_A and compound_B:
        try:
            camd.add_binary_system(compound_A, compound_B)
            print("   Binary system added successfully")
            print(f"   Compound A: {compound_A}")
            print(f"   Compound B: {compound_B}")
        except Exception as e:
            print(f"   Error adding binary system: {e}")
    else:
        print("   No binary system defined - add compounds to compound_A and compound_B")
    
    # Step 3: Set design groups for solvent candidate generation
    print("\n3. Setting design groups for solvent generation...")
    design_groups = ["CH3", "CH2", "CH", "C", "OH", "CHO", "HCO", "CH3COO", "CH2COO", "HCOO", "CH3CO", "CH2O", "COOH", "COO"]  # TODO: Add groups like ['CH3', 'CH2', 'OH', 'COOH', 'CH3CO']
    if design_groups:
        camd.set_design_groups(design_groups)
        print(f"   Design groups set: {design_groups}")
    else:
        print("   No design groups defined - add groups to design_groups list")
    
    # Step 4: Add structural constraints for solvent generation
    print("\n4. Adding structural constraints for solvents...")
    camd.add_constraint_structural(
        min_number_of_groups=3,      # Adjust min total groups for solvents
        max_number_of_groups=6,      # Adjust max total groups for solvents
        min_number_of_repetitions=0, # Adjust min group repetitions
        max_number_of_repetitions=100, # Adjust max group repetitions
        min_func_groups=1,           # Adjust min functional groups
        max_func_groups=2,           # Adjust max functional groups
        molecular_type=1             # 1=acyclic, 0=monocyclic, -1=bicyclic
    )
    
    # Step 5: Generate solvent candidates using MILP (Option A)
    print("\n5a. Generating solvent candidates using MILP...")
    try:
        if design_groups:
            generated_candidates = camd.generate_candidates(
                max_solutions=300000,   # Adjust max solutions (have it as high as possible)
                verbose=False       # Set True for solver output
            )
            print(f"   Generated {len(generated_candidates)} solvent candidates")
        else:
            print("   Skipping candidate generation - no design groups set")
            generated_candidates = []
    except Exception as e:
        print(f"   Error generating candidates: {e}")
        generated_candidates = []
    
    # Step 6: Alternative - Set predefined solvent candidates (Option B)
    print("\n5b. Alternative: Setting predefined solvent candidates...")
    predefined_solvents = [
        # Add solvent candidates as dicts
        {'H2O': 1},    # water
        {'DMSO': 1},   # DMSO
        {'DOH': 1},    # Ethylene glycol
        {'HCOOH': 1},    # Formic acid
    ]
    
    if predefined_solvents:
        try:
            camd.set_candidates(predefined_solvents)
            print(f"   Set {len(predefined_solvents)} predefined solvent candidates")
        except Exception as e:
            print(f"   Error setting solvent candidates: {e}")
    else:
        print("   No predefined solvents defined - add to predefined_solvents list")
    
    # Step 6: Validate system (after all candidates are set)
    print("\n6. Validating system...")
    try:
        if compound_A and compound_B:
            camd.validate_system()
            print("   System validation successful")
        else:
            print("   Skipping validation - no binary system defined")
    except Exception as e:
        print(f"   Error in system validation: {e}")
    
    # Step 7: Set pure component property constraints
    print("\n7. Setting pure component property constraints...")
    property_constraints = {
        'tb': [368.0, 500.0],      # Boiling point 368-500 K
        'tm': [0.0, 298.0],    # Melting point 298K+ (no upper limit)
        #'tc': [400.0, 650.0],     # Critical temperature 400-650 K  
        #'pc': [1e5, 5e6],         # Critical pressure 1-50 bar (1e5-5e6 Pa)
        #'ac': [0.1, 0.8],         # Acentric factor 0.1-0.8
        'hv': [-9999.0, 67.0]      # Heat of vaporization range
    }
    
    if property_constraints:
        try:
            camd.add_constraint_properties(property_constraints)
            print("   Property constraints set successfully")
        except Exception as e:
            print(f"   Error setting property constraints: {e}")
    else:
        print("   No property constraints defined")
    
    # Step 8: Calculate pure component properties for all candidates
    print("\n8. Calculating pure component properties...")
    try:
        if camd.generated_compounds:
            successful_calcs = camd.calculate_pure_properties()
            print(f"   Properties calculated for {successful_calcs} candidates")
        else:
            print("   No candidates available for property calculation")
    except Exception as e:
        print(f"   Error calculating properties: {e}")
    
    # Step 9: Manual property adjustments for compounds A and B (Optional)
    print("\n9. Manual property adjustments for compounds A and B (Optional)...")
    
    # Show calculated properties first
    if compound_A and compound_B:
        try:
            camd.show_binary_system_properties()
        except Exception as e:
            print(f"   Error displaying properties: {e}")
    
    # Example manual adjustments for compounds A and B
    # Uncomment the examples below to override specific property values
    # 
    # # Acetone: Set experimental data for acetone (compound A)
    try:
        camd.set_binary_system_property('A', 'tb', 329.25)  # Experimental Tb for acetone
        camd.set_binary_system_property('A', 'tc', 508.15)  # Experimental Tc for acetone
        print("   Set experimental property values for compound A")
    except Exception as e:
        print(f"   Error setting property for compound A: {e}")
    # 
    # # Example 2: Set experimental critical pressure for methanol (compound B)  
    # try:
    #     camd.set_binary_system_property('B', 'pc', 80.84)  # Experimental Pc for methanol in bar
    #     print("   Set experimental critical pressure for compound B (methanol)")
    # except Exception as e:
    #     print(f"   Error setting property for compound B: {e}")

    
    #print("   No manual adjustments active - uncomment examples above to override calculated values")
    
    # Step 10: Filter candidates based on pure component properties
    print("\n10. Filtering candidates by pure component properties...")
    try:
        filtering_stats = camd.filter_pure_properties()
        
        if filtering_stats['skipped']:
            print(f"   Skipping property filtering: {filtering_stats['skip_reason']}")
        else:
            # Display comprehensive filtering diagnostics
            print(f"\n=== Property Filtering Diagnostics ===")
            print(f"Candidate Source Breakdown:")
            print(f"  Generated candidates (MILP): {filtering_stats['generated']}")
            print(f"  Predefined candidates: {filtering_stats['predefined']}")
            print(f"  Total candidates: {filtering_stats['total']}")
            
            print(f"\nProperty Constraints (in evaluation order):")
            units = {'tm': 'K', 'tb': 'K', 'tc': 'K', 'pc': 'Pa', 'ac': '-', 'hv': 'kJ/mol'}
            for i, prop_name in enumerate(filtering_stats['constraints'], 1):
                bounds = filtering_stats['constraint_details'][prop_name]
                unit = units.get(prop_name, '')
                print(f"  {i}. {prop_name.upper()}: {bounds[0]} - {bounds[1]} {unit}")
            
            print(f"\n=== Filtering Results ===")
            total = filtering_stats['total']
            passed = filtering_stats['passed']
            failed = filtering_stats['failed']
            
            print(f"✅ PASSED all constraints: {passed}/{total} ({passed/total*100:.1f}%)")
            print(f"❌ FAILED constraints: {failed}/{total} ({failed/total*100:.1f}%)")
            
            if failed > 0:
                print(f"\nFailure Analysis (by FIRST constraint that failed):")
                for i, prop_name in enumerate(filtering_stats['constraints'], 1):
                    count = filtering_stats['first_failure_stats'].get(prop_name, 0)
                    if count > 0:
                        percentage = count/total*100
                        print(f"  {i}. {prop_name.upper()}: {count} candidates ({percentage:.1f}%) failed here first")
                
                missing_props = filtering_stats['first_failure_stats'].get('missing_properties', 0)
                if missing_props > 0:
                    percentage = missing_props/total*100
                    print(f"  Missing properties: {missing_props} candidates ({percentage:.1f}%) had no calculated data")
            
            print(f"\nFinal Result: {passed} candidates available for screening")
        
    except Exception as e:
        print(f"   Error in property filtering: {e}")
    
    # Step 11: Calculation of Antoine parameters for VLE calculation
    print("\n11. Calculation of Antoine parameters for VLE calculation...")
    try:
        if camd.calculated_properties:
            successful_antoine = camd.calculate_antoine_parameters()
            print(f"   Antoine parameters calculated for {successful_antoine} candidates")
        else:
            print("   No calculated properties available for Antoine parameter calculation")
    except Exception as e:
        print(f"   Error calculating Antoine parameters: {e}")
    
    # Step 12: Check UNIFAC R/Q parameters and binary interaction parameters
    print("\n12. Checking UNIFAC parameters...")
    try:
        if camd.generated_compounds:
            missing_rq, missing_interactions, compounds_to_remove = camd.check_unifac(verbose=True)
            
            if missing_rq or missing_interactions or compounds_to_remove:
                print(f"   Found UNIFAC parameter issues:")
                if missing_rq:
                    print(f"     Groups missing R/Q parameters: {len(missing_rq)} groups")
                if missing_interactions:
                    print(f"     Missing binary interactions: {len(missing_interactions)} main group pairs")
                if compounds_to_remove:
                    print(f"     Compounds to remove: {len(compounds_to_remove)} candidates")
                    
                    # Remove compounds with missing UNIFAC parameters
                    remaining_compounds = [comp for i, comp in enumerate(camd.generated_compounds) 
                                         if i not in compounds_to_remove]
                    camd.generated_compounds = remaining_compounds
                    
                    # Also update calculated_properties to maintain consistency
                    updated_properties = {}
                    new_index = 0
                    for old_index, properties in camd.calculated_properties.items():
                        if old_index not in compounds_to_remove:
                            updated_properties[new_index] = properties
                            new_index += 1
                    camd.calculated_properties = updated_properties
                    
                    print(f"     Remaining candidates after UNIFAC filtering: {len(camd.generated_compounds)}")
            else:
                print("   All candidates have complete UNIFAC parameters")
        else:
            print("   No candidates available for UNIFAC parameter check")
    except Exception as e:
        print(f"   Error checking UNIFAC parameters: {e}")
    
    # Step 13: Run screening calculations
    print("\n13. Running screening calculations...")
    # Commented out until VLE calculations are implemented
    # try:
    #     if compound_A and compound_B and camd.generated_compounds:
    #         camd.run()
    #         
    #         # Get and save results
    #         results = camd.get_results()
    #         camd.save_results()
    #         print("   Screening calculations completed successfully")
    #     else:
    #         print("   Skipping calculations - no complete system defined")
    # except Exception as e:
    #     print(f"   Error in calculations: {e}")
    print("   Screening calculations not implemented yet")
    
    # Step 14: Summary
    print("\n=== Summary ===")
    print(f"Title: {camd.title}")
    print(f"Binary system: A={compound_A}, B={compound_B}")
    print(f"Available groups: {len(camd.available_groups)}")
    print(f"Design groups: {len(camd.design_groups)}")
    print(f"Generated candidates: {len(camd.generated_compounds)}")
    print(f"Property constraints: {len(camd.property_constraints)} properties")
    print(f"Calculated properties: {len(camd.calculated_properties)} candidates")
    print(f"Output directory: {camd.output_dir}")
    
    if camd.calculated_properties:
        print("\n=== Property Calculation Results ===")
        for i, properties in camd.calculated_properties.items():
            if i < 3:  # Show first 3 candidates
                compound = camd.generated_compounds[i]
                print(f"Candidate {i} ({compound}):")
                for prop, value in properties.items():
                    if value is not None:
                        units = {'tb': 'K', 'tc': 'K', 'pc': 'Pa', 'ac': '-', 'hv': 'kJ/mol'}
                        unit = units.get(prop, '')
                        print(f"  {prop.upper()}: {value:.2f} {unit}")
        if len(camd.calculated_properties) > 3:
            print(f"  ... and {len(camd.calculated_properties) - 3} more candidates")
    
    print("\n=== Workflow Completed ===")
    print("1. ✓ Binary system defined (compounds A and B)")
    print("2. ✓ Design groups set for solvent generation")
    print("3. ✓ Structural constraints applied")
    print("4. ✓ Candidates generated/set")
    print("5. ✓ Property constraints defined")
    print("6. ✓ Pure component properties calculated")
    print("7. ✓ Property-based filtering applied")
    print("8. ✓ System validated")
    print("9. ✓ Antoine parameters calculated for VLE")
    print("10. ✓ UNIFAC parameters validated (R/Q + interactions)")
    print("11. ⏳ VLE screening calculations (to be implemented)")
    
    print("\n=== Next Development Steps ===")
    print("1. Implement VLE calculations using UNIFAC-LY method")
    print("2. Add mixture property calculations")
    print("3. Implement solvent selection criteria")
    print("4. Add results export and visualization")
    print("5. Add options for handling missing UNIFAC interactions")

if __name__ == "__main__":
    main()