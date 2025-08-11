# Repository Plan

## Overview
Tool for solvents and green solvents screening using Computer-Aided Molecular Design (CAMD)

## Architecture

### CAMD Class Design

#### Initialization
```python
CAMD(title="", type="VLE", output_dir="./results", 
     pure_method="Constantinou-Gani", mixture_method="UNIFAC-LY")
```
- **Type**: VLE (Vapor-Liquid Equilibrium) - LLE and SLE planned for future
- **Pure method**: Constantinou-Gani group contribution for pure component properties
- **Mixture method**: UNIFAC-LY for mixture thermodynamics
- Creates output directory and loads all property data from JSON files

#### Data Loading (Automatic)
- **Available groups**: Loaded from `data/json/smarts.json` 
- **Pure properties**: Constantinou-Gani data (Tc, Pc, Tb, Hv, etc.) from individual JSON files
- **Mixture properties**: UNIFAC interaction parameters and group connectivity data
- **Scalar constants**: Thermodynamic constants for property calculations

#### Design Space Definition
```python
CAMD.set_design_groups(['CH3', 'CH2', 'OH', 'CH3CO'])
```
- Define which UNIFAC groups can be used in compound generation
- All groups validated against loaded database

#### Structural Constraints
```python
CAMD.add_constraint_structural(
    min_number_of_groups=3, max_number_of_groups=8,
    min_number_of_repetitions=0, max_number_of_repetitions=5,
    min_func_groups=1, max_func_groups=3,
    molecular_type=1  # 1=acyclic, 0=monocyclic, -1=bicyclic
)
```
- **Total groups**: Min/max groups per compound
- **Group repetitions**: How many times each group can appear
- **Functional groups**: Min/max non-alkyl groups (excludes CH3, CH2, CH, C)
- **Molecular type**: Currently supports acyclic molecules

#### Compound Generation (MILP)
```python
compounds = CAMD.generate_compounds(max_solutions=10000, verbose=False)
```
- **Mixed-Integer Linear Programming** using Gurobi optimizer
- **Odele-Macchietto structural feasibility rules**:
  - Octet rule: `2*m = Σ(2 - noc_i) * x_i` 
  - Connectivity rule: `Σ(x_j ≠ i) ≥ (noc_i - 2) * x_i + 2`
- **Complete molecules** (noc=0) like H2O, I, Br automatically excluded from MILP
- **Functional group constraints** exactly matching original implementation

#### Binary System Definition  
```python
CAMD.add_binary_system(
    compound_A={'CH3': 2, 'CH2': 1, 'OH': 1},  # Ethanol
    compound_B={'H2O': 1}                        # Water
)
```
- Define binary mixtures with structural validation
- Both compounds validated using Odele-Macchietto rules

#### Data Validation & Structural Rules
- **Group validation**: All groups must exist in UNIFAC database
- **Structural feasibility**: Odele-Macchietto rules enforced
- **Complete molecules**: Groups with noc=0 handled separately
- **Error handling**: Clear messages for constraint violations

## Thermodynamic Models
- UNIFAC-LY implementation
- Future model integration points

## Output & Visualization
- Results export formats
- Plotting capabilities

## Goals

## Current Status ✅

### Completed Implementation
- [x] **CAMD Class Structure** - Complete with pure/mixture method separation
- [x] **Data Loading System** - JSON-based property data loading (scalars, Hv exported)
- [x] **Group Validation** - Against UNIFAC database from smarts.json
- [x] **Structural Constraints** - Comprehensive constraint management
- [x] **MILP Compound Generation** - Gurobi-based with Odele-Macchietto rules
- [x] **Binary System Definition** - With structural validation
- [x] **Complete Molecule Handling** - noc=0 groups properly excluded from MILP

### Data Export System
- [x] **Property Exporter** - Flexible single/multiple property export from Excel to JSON
- [x] **Scalar Constants** - Special handling for constants with different structure
- [x] **JSON Data Structure** - Consistent format with metadata and sample entries

## Next Steps

### Immediate Priority
- [ ] **Create methanol-acetone case study script** - For monitoring implementation progress
- [ ] **Pure component constraints and calculations** - Implement Constantinou-Gani property screening
- [ ] **Export UNIFAC interaction parameters (a-matrix)** to JSON format

### Thermodynamic Implementation  
- [ ] Implement UNIFAC-LY mixture calculations
- [ ] Implement VLE flash calculations
- [ ] Add comprehensive property constraints for compound screening

### System Integration
- [ ] Develop results export and visualization
- [ ] Add performance benchmarking against literature data

## Dependencies/Requirements
- **Python Packages**: pandas, numpy, gurobipy, openpyxl, pathlib
- **Gurobi License**: Required for MILP compound generation
- **Data Files**: Excel thermodynamic database, exported JSON property files

## Case Study: Methanol-Acetone Separation

A benchmark case study to monitor implementation progress and validate against literature data.

### System Definition
- **Compound A**: Methanol - `{'CH3': 1, 'OH': 1}`
- **Compound B**: Acetone - `{'CH3': 2, 'CH3CO': 1}`  
- **Literature data**: Available for validation in `old_code/case_study/`
- **Target**: VLE calculations and entrainer screening

### Implementation Monitoring
This case study will serve as a continuous benchmark to ensure:
- Pure property calculations match expected values
- UNIFAC activity coefficients are correct
- VLE predictions align with experimental data
- Generated entrainers show reasonable performance

## Complete Usage Example
```python
# Initialize CAMD with methods
camd = CAMD(
    title="Ethanol-Water Separation", 
    type="VLE",
    pure_method="Constantinou-Gani", 
    mixture_method="UNIFAC-LY",
    output_dir="./results"
)

# Set design space
camd.set_design_groups(['CH3', 'CH2', 'OH', 'CH3CO', 'H2O'])

# Add structural constraints
camd.add_constraint_structural(
    min_number_of_groups=2, max_number_of_groups=6,
    min_func_groups=1, max_func_groups=2,
    molecular_type=1  # acyclic
)

# Generate feasible compounds
compounds = camd.generate_compounds(max_solutions=5000)
print(f"Generated {len(compounds)} feasible compounds")

# Define binary system for screening
camd.add_binary_system(
    compound_A={'CH3': 2, 'CH2': 1, 'OH': 1},  # Ethanol  
    compound_B={'H2O': 1}                        # Water
)

# Run screening (future implementation)
# camd.run()
# results = camd.get_results()
```

## Notes
