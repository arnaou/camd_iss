"""
Test suite for CompoundValidator class.

Tests the compound validation using Odele and Macchietto structural feasibility rules.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from compound_validator import CompoundValidator, validate_compound, filter_valid_compounds


class TestCompoundValidator:
    """Test class for CompoundValidator functionality."""
    
    @pytest.fixture
    def validator(self):
        """Create a CompoundValidator instance for testing."""
        return CompoundValidator()
    
    @pytest.fixture
    def test_compounds(self):
        """Test compounds with known validity status."""
        return {
            'valid': [
                {'H2O': 1},                         # water - valid standalone molecule
                {'CH3': 1, 'CH2': 1, 'OH': 1},      # Ethanol - valid
                {'CH3': 1, 'CH3CO': 1},             # Acetone - valid
                {'CH3': 1, 'CH2': 2, 'OH': 1},      # Propanol - valid
            ],
            'invalid': [
                {'H2O': 2},                                # invalid - multiple standalone molecules
                {'CH3': 10},                               # Too many CH3 - not valid
                {'OH': 5},                                 # Too many OH - not valid
                {'CH3': 1, 'CH2': 3, 'H2O': 1, 'CHO': 1}, # invalid - standalone molecule mixed with others
                {'H2O': 1, 'CO2': 1},                     # invalid - multiple different standalone molecules
            ]
        }
    
    def test_initialization(self):
        """Test CompoundValidator initialization."""
        validator = CompoundValidator()
        assert validator.molecular_type == 1  # Default acyclic
        assert isinstance(validator.data_path, Path)
        assert isinstance(validator.noc_data, dict)
    
    def test_initialization_with_custom_molecular_type(self):
        """Test initialization with different molecular types."""
        validator_mono = CompoundValidator(molecular_type=0)
        assert validator_mono.molecular_type == 0
        
        validator_bi = CompoundValidator(molecular_type=-1)
        assert validator_bi.molecular_type == -1
    
    def test_invalid_molecular_type(self):
        """Test that invalid molecular_type raises ValueError."""
        with pytest.raises(ValueError):
            CompoundValidator(molecular_type=2)
    
    def test_set_molecular_type(self, validator):
        """Test setting molecular type after initialization."""
        validator.set_molecular_type(0)
        assert validator.molecular_type == 0
        
        with pytest.raises(ValueError):
            validator.set_molecular_type(5)
    
    # Test specific valid compounds
    @pytest.mark.parametrize("compound", [
        {'H2O': 1},                         # Valid standalone molecule
        {'CH3': 1, 'CH2': 1, 'OH': 1},      # Ethanol
        {'CH3': 1, 'CH3CO': 1},             # Acetone
        {'CH3': 1, 'CH2': 2, 'OH': 1},      # Propanol
    ])
    def test_valid_compounds(self, validator, compound):
        """Test that valid compounds pass validation."""
        assert validator.validate_compound(compound) == True
        assert validator.validate_compound(compound, verbose=True) == True
    
    # Test specific invalid compounds
    @pytest.mark.parametrize("compound", [
        {'H2O': 2},                                # Multiple standalone molecules
        {'CH3': 10},                               # Octet rule violation
        {'OH': 5},                                 # Octet rule violation
        {'CH3': 1, 'CH2': 3, 'H2O': 1, 'CHO': 1}, # Standalone molecule mixed with others
        {'H2O': 1, 'CO2': 1},                     # Multiple different standalone molecules
    ])
    def test_invalid_compounds(self, validator, compound):
        """Test that invalid compounds fail validation."""
        assert validator.validate_compound(compound) == False
        assert validator.validate_compound(compound, verbose=True) == False
    
    def test_standalone_molecule_validation(self, validator):
        """Test specific standalone molecule validation rules."""
        # Valid: single standalone molecule
        assert validator.validate_compound({'H2O': 1}) == True
        
        # Invalid: multiple copies of standalone molecule
        assert validator.validate_compound({'H2O': 2}) == False
        
        # Invalid: standalone molecule mixed with others
        assert validator.validate_compound({'H2O': 1, 'CH3': 1}) == False
        
        # Invalid: multiple different standalone molecules
        assert validator.validate_compound({'H2O': 1, 'CO2': 1}) == False
    
    def test_empty_compound(self, validator):
        """Test validation of empty compound."""
        assert validator.validate_compound({}) == False
    
    def test_filter_valid_compounds(self, validator, test_compounds):
        """Test filtering valid compounds from a list."""
        all_compounds = test_compounds['valid'] + test_compounds['invalid']
        
        valid_compounds, invalid_compounds = validator.filter_valid_compounds(all_compounds)
        
        # Check counts
        assert len(valid_compounds) == len(test_compounds['valid'])
        assert len(invalid_compounds) == len(test_compounds['invalid'])
        
        # Check that all expected valid compounds are in the valid list
        for compound in test_compounds['valid']:
            assert compound in valid_compounds
        
        # Check that all expected invalid compounds are in the invalid list
        for compound in test_compounds['invalid']:
            assert compound in invalid_compounds
    
    def test_get_validation_stats(self, validator, test_compounds):
        """Test validation statistics generation."""
        all_compounds = test_compounds['valid'] + test_compounds['invalid']
        
        stats = validator.get_validation_stats(all_compounds)
        
        # Check required keys
        required_keys = ['total', 'valid', 'invalid', 'standalone_violations', 
                        'octet_violations', 'connectivity_violations', 'missing_data']
        for key in required_keys:
            assert key in stats
        
        # Check basic counts
        assert stats['total'] == len(all_compounds)
        assert stats['valid'] == len(test_compounds['valid'])
        assert stats['invalid'] == len(test_compounds['invalid'])
        assert stats['valid'] + stats['invalid'] == stats['total']
        
        # Should have some standalone violations
        assert stats['standalone_violations'] > 0
        
        # Should have some octet violations
        assert stats['octet_violations'] > 0
    
    def test_octet_rule_violations(self, validator):
        """Test specific octet rule violations."""
        # Too many CH3 groups
        assert validator.validate_compound({'CH3': 10}) == False
        
        # Too many OH groups
        assert validator.validate_compound({'OH': 5}) == False
    
    def test_molecular_type_effects(self):
        """Test how different molecular types affect validation."""
        # Test compound that might be valid for different molecular types
        compound = {'CH3': 1, 'CH2': 1, 'OH': 1}  # Ethanol
        
        # Should be valid for acyclic (default)
        validator_acyclic = CompoundValidator(molecular_type=1)
        assert validator_acyclic.validate_compound(compound) == True
        
        # Test with monocyclic
        validator_mono = CompoundValidator(molecular_type=0)
        # This might be invalid due to different octet rule expectations
        result_mono = validator_mono.validate_compound(compound)
        assert isinstance(result_mono, bool)  # Should return a boolean
    
    def test_convenience_functions(self, test_compounds):
        """Test standalone convenience functions."""
        # Test validate_compound function
        assert validate_compound({'H2O': 1}) == True
        assert validate_compound({'H2O': 2}) == False
        
        # Test filter_valid_compounds function
        all_compounds = test_compounds['valid'] + test_compounds['invalid']
        valid_filtered = filter_valid_compounds(all_compounds)
        
        assert len(valid_filtered) == len(test_compounds['valid'])
        for compound in test_compounds['valid']:
            assert compound in valid_filtered
    
    def test_verbose_output(self, validator, capsys):
        """Test that verbose mode produces output."""
        # Test with valid compound
        validator.validate_compound({'H2O': 1}, verbose=True)
        captured = capsys.readouterr()
        assert "valid standalone molecule" in captured.out
        
        # Test with invalid compound
        validator.validate_compound({'H2O': 2}, verbose=True)
        captured = capsys.readouterr()
        assert "Invalid compound" in captured.out
    
    def test_missing_noc_data(self):
        """Test behavior when groups have missing NOC data."""
        # Create validator with non-existent data path
        validator = CompoundValidator(data_path="/nonexistent/path")
        
        # Should handle gracefully and assume valid
        result = validator.validate_compound({'UNKNOWN_GROUP': 1})
        assert result == True  # Should assume valid when no data available
    
    def test_is_pure_standalone_molecule(self, validator):
        """Test the helper method for detecting pure standalone molecules."""
        # Valid standalone molecule
        assert validator._is_pure_standalone_molecule({'H2O': 1}) == True
        
        # Not standalone (multiple groups)
        assert validator._is_pure_standalone_molecule({'H2O': 1, 'CH3': 1}) == False
        
        # Not standalone (count > 1)
        assert validator._is_pure_standalone_molecule({'H2O': 2}) == False
        
        # Not standalone (NOC != 0)
        assert validator._is_pure_standalone_molecule({'CH3': 1}) == False


if __name__ == "__main__":
    # Run basic test if executed directly
    validator = CompoundValidator()
    
    print("=== Quick Validation Test ===")
    
    test_cases = [
        ({'H2O': 1}, True, "Valid standalone molecule"),
        ({'H2O': 2}, False, "Invalid - multiple standalone molecules"),
        ({'CH3': 1, 'CH2': 1, 'OH': 1}, True, "Valid - ethanol"),
        ({'CH3': 1, 'H2O': 1}, False, "Invalid - mixed with standalone"),
        ({'CH3': 10}, False, "Invalid - octet rule violation"),
    ]
    
    for compound, expected, description in test_cases:
        result = validator.validate_compound(compound)
        status = "✓" if result == expected else "✗"
        print(f"{status} {description}: {compound} -> {result}")
    
    print("\nAll basic tests completed!")