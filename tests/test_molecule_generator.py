import importlib.util
from pathlib import Path

import pytest
import yaml

from src.molecules import generate_molecules

CONFIG_DIR = Path(__file__).parent.parent / "config"
SOLVER_MODULES = {"gurobi": "gurobipy", "cplex": "cplex"}

# Total candidates, cross-checked between Gurobi and CPLEX. Includes complete
# molecules (noc=0 groups, e.g. DMSO) that bypass the MILP.
CASES = [
    ("camd_minimal.yaml", 141),
    ("IECR/design_acetone_methanol.yaml", 478),
]


def _run(config_file, solver):
    design = yaml.safe_load(open(CONFIG_DIR / config_file))["design"]
    c = design["constraints"]
    return generate_molecules(
        design_groups=design["groups"], min_groups=c["min_groups"], max_groups=c["max_groups"],
        min_func_groups=c["min_func_groups"], max_func_groups=c["max_func_groups"],
        molecular_type=c["molecular_type"], max_solutions=design["max_candidates"],
        verbose=False, solver=solver,
    )


def _available(solver):
    return importlib.util.find_spec(SOLVER_MODULES[solver]) is not None


@pytest.mark.parametrize("solver", SOLVER_MODULES)
@pytest.mark.parametrize("config_file,n_expected", CASES)
def test_solver_finds_all_compositions(solver, config_file, n_expected):
    if not _available(solver):
        pytest.skip(f"{solver} not installed")
    candidates = _run(config_file, solver)
    assert len(candidates) == n_expected
    assert len({frozenset(c.items()) for c in candidates}) == n_expected


@pytest.mark.parametrize("config_file,_", CASES)
def test_gurobi_and_cplex_agree(config_file, _):
    if not (_available("gurobi") and _available("cplex")):
        pytest.skip("needs both gurobi and cplex")
    as_set = lambda cands: {frozenset(c.items()) for c in cands}
    assert as_set(_run(config_file, "gurobi")) == as_set(_run(config_file, "cplex"))


def test_unknown_solver_rejected():
    with pytest.raises(ValueError, match="Unknown solver"):
        _run("camd_minimal.yaml", "scip")
