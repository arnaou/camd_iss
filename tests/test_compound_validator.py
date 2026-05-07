"""
Test suite for StructuralValidator class.

Tests compound validation using Odele and Macchietto structural feasibility rules.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.molecules.structural_validator import StructuralValidator


class TestStructuralValidator:

    @pytest.fixture
    def validator(self):
        return StructuralValidator()

    def test_initialization_defaults(self, validator):
        assert validator.molecular_type == 1
        assert isinstance(validator.noc_data, dict)
        assert len(validator.noc_data) > 0

    def test_invalid_molecular_type(self):
        with pytest.raises(ValueError):
            StructuralValidator(molecular_type=2)

    def test_molecular_type_options(self):
        for mt in [-1, 0, 1]:
            v = StructuralValidator(molecular_type=mt)
            assert v.molecular_type == mt

    # --- Standalone molecules (NOC = 0) ---

    def test_water_valid(self, validator):
        assert validator.is_valid({'H2O': 1}) is True

    def test_water_multiple_invalid(self, validator):
        assert validator.is_valid({'H2O': 2}) is False

    def test_standalone_mixed_with_others_invalid(self, validator):
        assert validator.is_valid({'H2O': 1, 'CH3': 1}) is False

    # --- Connected acyclic molecules ---

    @pytest.mark.parametrize("compound", [
        {'CH3': 1, 'CH2': 1, 'OH': 1},      # ethanol
        {'CH3': 1, 'CH3CO': 1},             # acetone
        {'CH3': 1, 'CH2': 2, 'OH': 1},      # propanol
        {'CH3OH': 1},                        # methanol (standalone)
    ])
    def test_valid_acyclic_compounds(self, validator, compound):
        assert validator.is_valid(compound) is True

    @pytest.mark.parametrize("compound", [
        {'CH3': 10},          # octet rule violation
        {'OH': 5},            # octet rule violation
        {},                   # empty
    ])
    def test_invalid_compounds(self, validator, compound):
        assert validator.is_valid(compound) is False

    # --- validate() auto-detection ---

    def test_validate_single_compound(self, validator):
        result = validator.validate({'CH3': 1, 'CH3CO': 1})
        assert result is True

    def test_validate_batch(self, validator):
        batch = {
            'ethanol': {'CH3': 1, 'CH2': 1, 'OH': 1},
            'invalid': {'CH3': 10},
        }
        results = validator.validate(batch)
        assert isinstance(results, dict)
        assert results['ethanol'] is True
        assert results['invalid'] is False

    def test_validate_empty(self, validator):
        assert validator.validate({}) is False

    # --- Missing NOC data ---

    def test_missing_noc_data_returns_false(self, validator):
        # Groups not in noc.json cannot be validated
        assert validator.is_valid({'UNKNOWN_GROUP_XYZ': 1}) is False
