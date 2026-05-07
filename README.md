# CAMD-ISS

**Computer-Aided Molecular Design for Extractive Distillation using the Infinitely Sharp Split (ISS) Method**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## About

Extractive distillation separates azeotropic and close-boiling mixtures by adding an entrainer that modifies relative volatilities. Classical entrainer selection criteria — selectivity and capacity at infinite dilution — ignore the recovery column and can miss feasibility constraints.

CAMD-ISS addresses this by coupling molecular design with the Infinitely Sharp Split (ISS) method, which simultaneously accounts for the extractive and recovery columns. The framework:

- **Designs** new entrainer candidates via MILP-based molecular generation (UNIFAC group contributions)
- **Screens** predefined entrainer sets against thermodynamic and structural constraints
- Evaluates candidates using the **ISS objective** (entrainer-to-distillate ratio *ED*), computed via Pyomo/IPOPT optimization
- Performs **miscibility screening**, **bubble-point VLE**, and **azeotrope detection** along the way

## Installation

### Prerequisites

| Solver | Required for | License |
|--------|-------------|---------|
| [IPOPT](https://coin-or.github.io/Ipopt/) | VLE, ISS, Antoine (all modes) | Open source |
| [Gurobi](https://www.gurobi.com/) | Molecule generation (design mode only) | Commercial — free academic license available |

**Screen mode** (`screen_iss.py`) only requires IPOPT.  
**Design mode** (`camd_iss.py`) additionally requires a Gurobi license for the MILP-based candidate generation.

A free academic license for Gurobi can be obtained at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

```bash
# Install IPOPT via conda (recommended)
conda install -c conda-forge ipopt

# Install Gurobi (design mode only)
pip install gurobipy
# Then activate your license: grbgetkey <your-key>

# Install package and dependencies
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Usage

Both modes are driven by YAML configuration files and produce CSV/JSON output.

### Design Mode — generate candidates

```bash
python camd_iss.py config/paper1/design_acetone_methanol.yaml -v
```

The config specifies the binary mixture, allowed UNIFAC groups, structural constraints, and screening thresholds:

```yaml
mode: "design"

mixture:
  component_a:
    groups: {CH3: 1, CH3CO: 1}   # acetone
  component_b:
    groups: {CH3OH: 1}            # methanol

design:
  groups: ["CH3", "CH2", "CH", "C", "OH", "HCO", "DMSO", "DOH"]
  constraints:
    min_groups: 3
    max_groups: 6
    min_func_groups: 1
    max_func_groups: 2
    molecular_type: 1   # 1 = acyclic
```

### Screen Mode — evaluate predefined entrainers

```bash
python screen_iss.py config/paper2/screen_homo_acetone_methanol.yaml -v
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

### Pipeline

Both modes run the same 14-step screening pipeline:

| Step | Description |
|------|-------------|
| 1 | Validate binary mixture (structural check) |
| 2 | Check UNIFAC parameter availability |
| 3 | Calculate pure component properties (Constantinou-Gani) |
| 4 | Apply user property overrides |
| 5 | Calculate Antoine parameters (Pyomo optimization) |
| 6 | Generate candidates (design) or load entrainers (screen) |
| 7 | Check UNIFAC parameters for all candidates |
| 8 | Calculate pure properties for candidates |
| 9 | Screen by pure property constraints (Tb, Tc, Pc, …) |
| 10 | Calculate Antoine parameters for candidates |
| 11 | Miscibility screening over temperature range |
| 12 | Bubble-point VLE calculations (ternary) |
| 13 | Screen by mixture property constraints (selectivity, azeotropes) |
| 14 | ISS calculation — minimize entrainer-to-distillate ratio *ED* |

Results are written to the configured `output.dir` as `results.csv` and `results.json`.

## Project Structure

```
camd_iss/
├── camd_iss.py          # Design mode entry point
├── screen_iss.py        # Screen mode entry point
├── config/
│   ├── camd_minimal.yaml         # Minimal working config template
│   ├── paper1/                   # Acetone/methanol study
│   └── paper2/                   # Multi-system study
├── data/
│   └── json/            # UNIFAC parameters (rk, qk, interactions, Tc, Tb, …)
├── src/
│   ├── pipeline.py      # Shared 14-step pipeline
│   ├── molecules/       # Structure validation and MILP generation
│   ├── pure/            # Property estimation and Antoine calculation
│   ├── mixture/unifac/  # UNIFAC, VLE, ISS (Pyomo/IPOPT)
│   └── data/            # Parameter loading and UNIFAC compatibility
├── case_studies/        # Analysis notebooks (paper 1)
├── tutorials/           # Step-by-step Jupyter notebooks (01–09)
└── tests/               # Unit tests
```

## Tutorials

The `tutorials/` folder contains nine notebooks covering the full workflow:

| Notebook | Topic |
|----------|-------|
| `01_structures.ipynb` | UNIFAC group structures and validation |
| `02_pure_properties.ipynb` | Constantinou-Gani property estimation |
| `03_antoine.ipynb` | Antoine parameter determination (three methods) |
| `04_unifac.ipynb` | UNIFAC activity coefficient model |
| `05_vle_binary.ipynb` | Binary VLE calculations |
| `06_vle_ternary.ipynb` | Ternary VLE calculations |
| `07_azeotrope.ipynb` | Azeotrope detection |
| `08_iss.ipynb` | ISS optimization |
| `09_analyze_screening.ipynb` | Screening results analysis |

## Built With

- [Pyomo](https://www.pyomo.org/) — optimization modeling
- [IPOPT](https://coin-or.github.io/Ipopt/) — nonlinear optimization solver
- [NumPy](https://numpy.org/) / [SciPy](https://scipy.org/) / [Pandas](https://pandas.pydata.org/)

## Acknowledgements

This work originated as an M.Sc. thesis by Adem R.N. Aouichaoui under the supervision of Assoc. Prof. Jens Abildskov at the Process Systems Engineering Center (PROSYS), Technical University of Denmark (DTU).

Collaborators:

- Ivonne Rodriguez-Donis — Laboratoire de Chimie Agro-industrielle (LCA), Université de Toulouse, INRA
- Nataliya Shcherbakova — Laboratoire de Génie Chimique, Université de Toulouse, CNRS, INP, UPS
- Vincent Gerbaud — Laboratoire de Génie Chimique, Université de Toulouse, CNRS, INP, UPS

## Citation

If you use this software in your research, please cite the relevant paper(s):

**Paper 1 — CAMD-ISS framework (acetone/methanol case study):**

```bibtex
@article{iss_camd2026,
  author  = {Adem R. N. Aouichaoui and Ivonne Rodriguez-Donis and Nataliya Shcherbakova
             and Edoardo Parascandolo and Vincent Gerbaud and Jens Abildskov},
  title   = {Computer-Aided Entrainer Design for Extractive Distillation Using Infinitely Sharp Splits},
  journal = {ChemRxiv},
  year    = {2026},
  doi     = {10.26434/chemrxiv.15002406/v1},
  url     = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15002406/v1},
}
```

**Paper 2 — Extended case studies (placeholder):**

```bibtex
@article{iss_camd2_placeholder,
  author  = {},
  title   = {},
  journal = {},
  year    = {},
  doi     = {},
}
```

## Contributing

Issues and pull requests are welcome. Please open an issue first to discuss proposed changes.

## License

GNU General Public License v3. See [LICENSE](LICENSE) for details.
