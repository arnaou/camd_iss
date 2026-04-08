# CAMD-ISS

**Computer-Aided Molecular Design for Extractive Distillation using the Infinitely Sharp Split (ISS) Method**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## About

Extractive distillation separates azeotropic mixtures by introducing an entrainer that alters relative volatilities. Traditional entrainer selection evaluates selectivity and capacity at infinite dilution but neglects recovery column dynamics.

CAMD-ISS provides a framework that:

- **Designs** new entrainer candidates using MILP-based molecular generation with UNIFAC group contributions
- **Screens** predefined entrainer databases against thermodynamic criteria
- Evaluates entrainers using the **Infinitely Sharp Split (ISS)** method, which accounts for both extractive and recovery column performance
- Performs **miscibility screening**, **VLE calculations**, and **azeotrope detection** using Pyomo-based UNIFAC models

## Installation

### Prerequisites

- Python >= 3.9
- [IPOPT](https://coin-or.github.io/Ipopt/) solver (required for VLE/ISS optimization)

```bash
# Install IPOPT via conda (recommended)
conda install -c conda-forge ipopt

# Install the package
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

CAMD-ISS operates in two modes, both driven by YAML configuration files.

### Design Mode

Automatically generate and screen entrainer candidates:

```bash
python camd_iss.py config/example_design.yaml -v
```

The design config specifies the binary mixture (as UNIFAC groups), allowed design groups, structural constraints, and screening thresholds:

```yaml
mode: "design"

mixture:
  component_a:
    groups: {CH3: 1, CH3CO: 1}  # acetone
  component_b:
    groups: {CH3OH: 1}          # methanol

design:
  groups: ["CH3", "CH2", "CH", "C", "OH", "HCO", "DMSO", "DOH"]
  constraints:
    min_groups: 3
    max_groups: 6
    min_func_groups: 1
    max_func_groups: 2
    molecular_type: 1  # acyclic
```

### Screen Mode

Evaluate a predefined set of entrainers:

```bash
python screen_iss.py config/example_screen.yaml -v
```

```yaml
mode: "screen"

mixture:
  component_a:
    groups: {CH3: 1, CH3CO: 1}
  component_b:
    groups: {CH3OH: 1}

screen:
  entrainers:
    - name: "ethylene_glycol"
      groups: {DOH: 1}
    - name: "DMSO"
      groups: {DMSO: 1}
```

### Workflow

Both modes follow the same pipeline:

1. Validate binary mixture and check UNIFAC parameter availability
2. Calculate pure component properties (Constantinou-Gani method)
3. Calculate Antoine parameters via optimization
4. Generate candidates (design) or load predefined entrainers (screen)
5. Screen by pure component property constraints
6. Miscibility screening over temperature range
7. VLE calculations (bubble point)
8. Screen by mixture property constraints (selectivity, azeotropes)
9. ISS calculation for surviving candidates

Results are saved as CSV and JSON in the configured output directory.

### Analyzing Results

Use the screening log analyzer to understand why candidates were eliminated:

```bash
python analyze_screening_log.py results/design/camd_run_YYYYMMDD_HHMMSS.log
```

## Project Structure

```
camd_iss/
  camd_iss.py          # Design mode entry point
  screen_iss.py        # Screen mode entry point
  config/              # YAML configuration examples
  data/                # UNIFAC parameters and property data
    json/              # JSON parameter files (rk, qk, Tc, Tb, etc.)
  src/                 # Core library
    molecules/         # Structure validation and MILP generation
    pure/              # Property estimation (Constantinou-Gani)
    mixture/unifac/    # UNIFAC activity coefficients, VLE, ISS (Pyomo)
    data/              # Parameter loading and compatibility checking
    camd/              # CAMD engine
  tutorials/           # Jupyter notebooks with examples
  tests/               # Unit tests
```

## Built With

- [Pyomo](https://www.pyomo.org/) - Optimization modeling
- [IPOPT](https://coin-or.github.io/Ipopt/) - Nonlinear optimization solver
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) / [Pandas](https://pandas.pydata.org/)

## Acknowledgements

This work originated as an M.Sc. thesis by Adem R.N. Aouichaoui under the supervision of Assoc. Prof. Jens Abildskov at the Process Systems Engineering Center (PROSYS), Technical University of Denmark (DTU).

Collaborators:

- Ivonne Rodriguez-Donis, Laboratoire de Chimie Agro-industrielle (LCA), Universite de Toulouse, INRA
- Nataliya Shcherbakova, Laboratoire de Genie Chimique, Universite de Toulouse, CNRS, INP, UPS
- Vincent Gerbaud, Laboratoire de Genie Chimique, Universite de Toulouse, CNRS, INP, UPS

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the GNU General Public License v3. See [LICENSE](LICENSE) for details.
