"""
Test suite for src.unifac_reverse: UNIFAC group composition -> molecule SMILES.

Ported from unifac_reverse.run_tests(), which remains as an informal
CLI self-check (`python -m src.unifac_reverse`) but wasn't wired into the
pytest suite.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.unifac_reverse import load_unifac_groups, reverse_engineer_by_name

DB_PATH = Path(__file__).parent.parent / "data" / "json" / "unifac_ly.json"

# (description, composition, expected SMILES list or None, topology)
CONSTITUTIONAL_CASES = [
    # Acyclic
    ("Ethane",         {"CH3": 2},                              ["CC"],                "acyclic"),
    ("Propane",        {"CH3": 2, "CH2": 1},                    ["CCC"],               "acyclic"),
    ("Isobutane",      {"CH3": 3, "CH": 1},                     ["CC(C)C"],            "acyclic"),
    ("Neopentane",     {"CH3": 4, "C": 1},                      ["CC(C)(C)C"],         "acyclic"),
    ("Ethanol",        {"CH3": 1, "CH2": 1, "OH": 1},           ["CCO"],               "acyclic"),
    ("Acetone",        {"CH3": 1, "CH3CO": 1},                  ["CC(C)=O"],           "acyclic"),
    ("Acetic acid",    {"CH3": 1, "COOH": 1},                   ["CC(=O)O"],           "acyclic"),
    ("Ethyl acetate",  {"CH3": 2, "CH2COO": 1},                 ["CCC(=O)OC"],         "acyclic"),
    ("Diethyl ether",  {"CH3": 2, "CH2O": 1},                   ["CCOC"],              "acyclic"),
    ("Propene",        {"CH3": 1, "CH2=CH": 1},                 ["C=CC"],              "acyclic"),
    ("1-Bromopropane", {"CH3": 1, "CH2": 2, "Br": 1},           ["CCCBr"],             "acyclic"),
    ("C6 branched",    {"CH3": 3, "CH2": 2, "CH": 1},           None,                  "acyclic"),  # expect >=2

    # Alicyclic
    ("Cyclohexane",        {"CH2": 6},                          ["C1CCCCC1"],          "monocyclic"),
    ("Methylcyclohexane",  {"CH3": 1, "CH": 1, "CH2": 5},       None,                  "monocyclic"),  # expect >=1
    ("Cyclopentane",       {"CH2": 5},                          ["C1CCCC1"],           "monocyclic"),

    # Aromatic
    ("Benzene",        {"ACH": 6},                              ["c1ccccc1"],          "acyclic"),
    ("Toluene",        {"ACH": 5, "ACCH3": 1},                  ["Cc1ccccc1"],         "acyclic"),
    ("Xylenes",        {"ACH": 4, "ACCH3": 2},                  None,                  "acyclic"),  # expect 3
    ("Phenol",         {"ACH": 5, "ACOH": 1},                   ["Oc1ccccc1"],         "acyclic"),
    ("Chlorobenzene",  {"ACH": 5, "ACCl": 1},                   ["Clc1ccccc1"],        "acyclic"),
    ("Aniline",         {"ACH": 5, "ACNH2": 1},                 ["Nc1ccccc1"],         "acyclic"),

    # Mixed aromatic + aliphatic
    ("Ethylbenzene",   {"ACH": 5, "AC": 1, "CH2": 1, "CH3": 1}, ["CCc1ccccc1"],        "acyclic"),
    ("Styrene",        {"ACH": 5, "AC": 1, "CH2=CH": 1},        ["C=Cc1ccccc1"],       "acyclic"),
    ("Benzaldehyde",   {"ACH": 5, "AC": 1, "HCO": 1},           ["O=Cc1ccccc1"],       "acyclic"),
    ("Benzoic acid",   {"ACH": 5, "AC": 1, "COOH": 1},          ["O=C(O)c1ccccc1"],    "acyclic"),

    # Naphthalene
    ("Naphthalene",    {"ACH": 8, "AC": 2},                     ["c1ccc2ccccc2c1"],    "acyclic"),

    # Multi-ring aromatic
    ("Diphenylmethane", {"ACH": 10, "AC": 2, "CH2": 1},         ["c1ccc(Cc2ccccc2)cc1"], "acyclic"),
    ("Biphenyl",         {"ACH": 10, "AC": 2},                  ["c1ccc(-c2ccccc2)cc1"], "acyclic"),
]

# (description, composition, topology, min stereoisomer count)
STEREO_CASES = [
    ("2-butene E/Z",      {"CH3": 2, "CH=CH": 1},                        "acyclic",  2),
    ("cis/trans decalin", {"CH2": 8, "CH": 2},                           "bicyclic", 2),
    ("2-bromobutane R/S", {"CH3": 2, "CH": 1, "CH2": 1, "Br": 1},        "acyclic",  3),
]


@pytest.fixture(scope="module")
def group_db():
    return load_unifac_groups(str(DB_PATH))


@pytest.mark.parametrize("desc,composition,expected,topology", CONSTITUTIONAL_CASES,
                         ids=[c[0] for c in CONSTITUTIONAL_CASES])
def test_constitutional_isomers(group_db, desc, composition, expected, topology):
    results = reverse_engineer_by_name(
        composition, group_db, topology=topology, min_ring_size=5, verbose=False
    )
    if expected is not None:
        for e in expected:
            assert e in results, f"{desc}: expected {e!r} in {results}"
    else:
        assert len(results) >= 2, f"{desc}: expected >=2 isomers, got {results}"


def test_methylcyclohexane_at_least_one_isomer(group_db):
    results = reverse_engineer_by_name(
        {"CH3": 1, "CH": 1, "CH2": 5}, group_db, topology="monocyclic",
        min_ring_size=5, verbose=False
    )
    assert len(results) >= 1


@pytest.mark.parametrize("desc,composition,topology,min_count", STEREO_CASES,
                         ids=[c[0] for c in STEREO_CASES])
def test_stereoisomers(group_db, desc, composition, topology, min_count):
    results = reverse_engineer_by_name(
        composition, group_db, topology=topology, min_ring_size=5,
        enumerate_stereo=True, verbose=False
    )
    assert len(results) >= min_count, f"{desc}: expected >={min_count}, got {results}"


def test_unknown_group_returns_empty(group_db):
    assert reverse_engineer_by_name({"NOT_A_REAL_GROUP": 1}, group_db, verbose=False) == []
