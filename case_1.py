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
        mixture_method="UNIFAC-LY"
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
        {'HCOOH': 1},    # Ethylene glycol
    ]
    
    if predefined_solvents:
        try:
            camd.set_candidates(predefined_solvents)
            print(f"   Set {len(predefined_solvents)} predefined solvent candidates")
        except Exception as e:
            print(f"   Error setting solvent candidates: {e}")
    else:
        print("   No predefined solvents defined - add to predefined_solvents list")
    
    # Step 6: Set pure component property constraints
    print("\n6. Setting pure component property constraints...")
    property_constraints = {
        'tb': [300, 450],      # Boiling point 300-450 K
        'tc': [400, 650],      # Critical temperature 400-650 K  
        'pc': [1e5, 5e6],      # Critical pressure 1-50 bar (1e5-5e6 Pa)
        'ac': [0.1, 0.8],      # Acentric factor 0.1-0.8
        # 'hv': [20000, 60000]   # Heat of vaporization 20-60 kJ/mol (optional)
    }
    
    if property_constraints:
        try:
            camd.add_constraint_properties(property_constraints)
            print("   Property constraints set successfully")
        except Exception as e:
            print(f"   Error setting property constraints: {e}")
    else:
        print("   No property constraints defined")
    
    # Step 7: Calculate pure component properties for all candidates
    print("\n7. Calculating pure component properties...")
    try:
        if camd.generated_compounds:
            successful_calcs = camd.calculate_pure_properties()
            print(f"   Properties calculated for {successful_calcs} candidates")
        else:
            print("   No candidates available for property calculation")
    except Exception as e:
        print(f"   Error calculating properties: {e}")
    
    # Step 8: Manual property adjustments (Optional)
    print("\n8. Manual property adjustments (Optional)...")
    # Example: Override a specific property value
    # if len(camd.generated_compounds) > 0:
    #     try:
    #         camd.set_pure_property(candidate_index=0, property_name='tb', value=373.15)
    #         print("   Example: Set boiling point for candidate 0 to 373.15 K")
    #     except Exception as e:
    #         print(f"   Error setting manual property: {e}")
    print("   No manual adjustments defined - uncomment example above to test")
    
    # Step 9: Filter candidates based on pure component properties
    print("\n9. Filtering candidates by pure component properties...")
    try:
        if camd.property_constraints and camd.calculated_properties:
            remaining_candidates = camd.filter_pure_properties()
            print(f"   Filtering complete: {remaining_candidates} candidates remain")
        else:
            print("   Skipping property filtering - no constraints or calculated properties")
    except Exception as e:
        print(f"   Error in property filtering: {e}")
    
    # Step 10: Validate system
    print("\n10. Validating system...")
    try:
        if compound_A and compound_B:
            camd.validate_system()
            print("   System validation successful")
        else:
            print("   Skipping validation - no binary system defined")
    except Exception as e:
        print(f"   Error in system validation: {e}")
    
    # Step 11: Run screening calculations
    print("\n11. Running screening calculations...")
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
    
    # Step 12: Summary
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
                        units = {'tb': 'K', 'tc': 'K', 'pc': 'Pa', 'ac': '-', 'hv': 'J/mol'}
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
    print("9. ⏳ VLE screening calculations (to be implemented)")
    
    print("\n=== Next Development Steps ===")
    print("1. Implement VLE calculations using UNIFAC-LY method")
    print("2. Add mixture property calculations")
    print("3. Implement solvent selection criteria")
    print("4. Add results export and visualization")

if __name__ == "__main__":
    main()