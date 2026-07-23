"""
Test suite for PropertyCalculator class.

Tests pure component property calculations using the Constantinou-Gani method.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pure.property_calculator import PropertyCalculator


class TestPropertyCalculator:

    @pytest.fixture
    def calc(self):
        return PropertyCalculator()

    def test_initialization(self, calc):
        assert calc.method is not None
        assert calc.method.get_method_name() == "Constantinou-Gani"

    # --- Single compound: known values ---

    @pytest.mark.parametrize("compound,prop,expected,tol", [
        ({'H2O': 1},                    'tm', 273.15, 0.10),
        ({'DMSO': 1},                   'tm', 291.65, 0.10),
        ({'CH3OH': 1},                  'tm', 176.00, 0.10),
        ({'DOH': 1},                    'tm', 260.15, 0.10),
        ({'CH3': 2, 'CH2': 5},          'tm', 175.55, 0.10),
        ({'H2O': 1},                    'tb', 373.15, 0.10),
        ({'DMSO': 1},                   'tb', 463.10, 0.10),
        ({'CH3OH': 1},                  'tb', 337.85, 0.10),
        ({'DOH': 1},                    'tb', 470.45, 0.10),
        ({'CH3': 2, 'CH2': 5},          'tb', 379.07, 0.10),
        ({'H2O': 1},                    'tc', 647.00, 0.10),
        ({'CH3OH': 1},                  'tc', 513.00, 0.10),
        ({'CH3': 2, 'CH2': 5},          'tc', 549.86, 0.10),
        ({'H2O': 1},                    'pc', 220.64, 0.10),
        ({'CH3OH': 1},                  'pc',  81.00, 0.10),
        ({'CH3': 2, 'CH2': 5},          'pc',  28.27, 0.10),
        ({'H2O': 1},                    'hv',  43.99, 0.10),
        ({'CH3OH': 1},                  'hv',  37.47, 0.10),
    ])
    def test_property_values(self, calc, compound, prop, expected, tol):
        result = calc.calculate(compound)
        assert result is not None, f"calculate() returned None for {compound}"
        assert prop in result, f"Property '{prop}' missing from result"
        val = result[prop]
        assert val is not None and not np.isnan(val), f"{prop} is None/NaN"
        assert abs(val - expected) < tol, f"{prop} = {val:.4f}, expected {expected}"

    def test_all_properties_returned(self, calc):
        result = calc.calculate({'H2O': 1})
        assert result is not None
        for key in ['tm', 'tb', 'tc', 'pc', 'ac', 'hv']:
            assert key in result

    def test_missing_group_returns_none(self, calc):
        result = calc.calculate({'UNKNOWN_GROUP_XYZ': 1})
        assert result is None

    # --- Batch calculation ---

    def test_batch_calculate(self, calc):
        batch = {
            'water': {'H2O': 1},
            'methanol': {'CH3OH': 1},
        }
        results = calc.calculate(batch)
        assert isinstance(results, dict)
        assert 'water' in results
        assert 'methanol' in results
        assert results['water'] is not None
        assert results['methanol'] is not None

    def test_batch_with_invalid_compound(self, calc):
        batch = {
            'water': {'H2O': 1},
            'unknown': {'NO_SUCH_GROUP': 1},
        }
        results = calc.calculate(batch)
        assert results['water'] is not None
        assert results['unknown'] is None

    def test_empty_compound(self, calc):
        result = calc.calculate({})
        assert result is None

    # --- Selective property calculation ---

    def test_selective_properties(self, calc):
        result = calc.calculate({'CH3OH': 1}, properties=['tb', 'tc'])
        assert result is not None
        assert 'tb' in result
        assert 'tc' in result
        # Other properties should not be present
        assert 'tm' not in result
        assert 'hv' not in result

    def test_get_available_properties(self, calc):
        available = calc.method.get_available_properties()
        assert isinstance(available, list)
        assert len(available) > 0
        for p in available:
            assert p in ['tm', 'tb', 'tc', 'pc', 'ac', 'hv']
