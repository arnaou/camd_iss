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
    
    # Step 7: Validate and run screening calculations
    print("\n6. Running screening calculations...")
    try:
        if compound_A and compound_B:
            camd.validate_system()
            camd.run()
            
            # Get and save results
            results = camd.get_results()
            camd.save_results()
            print("   Screening calculations completed successfully")
        else:
            print("   Skipping calculations - no binary system defined")
    except Exception as e:
        print(f"   Error in calculations: {e}")
    
    # Step 8: Summary
    print("\n=== Summary ===")
    print(f"Title: {camd.title}")
    print(f"Binary system: A={compound_A}, B={compound_B}")
    print(f"Available groups: {len(camd.available_groups)}")
    print(f"Design groups: {len(camd.design_groups)}")
    print(f"Solvent candidates: {len(camd.generated_compounds)}")
    print(f"Output directory: {camd.output_dir}")
    
    print("\n=== Next Steps ===")
    print("1. Fill in the compounds to be separated:")
    print("   - compound_A: e.g., {'CH3': 2, 'OH': 1} for ethanol")
    print("   - compound_B: e.g., {'CH3': 6} for hexane")
    print("2. Define solvent design space:")
    print("   - design_groups: Add UNIFAC groups like ['CH3', 'CH2', 'OH', 'COOH']")
    print("3. Optional: Add predefined solvent candidates")
    print("4. Adjust constraint parameters for desired solvent properties")
    print("5. Run the script to test the implementation")

if __name__ == "__main__":
    main()