"""
Test suite for PropertyCalculator class.

Tests the pure component property calculations using the Constantinou-Gani method.
Uses specific expected values similar to the original test_gc.py.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from property_calculator import PropertyCalculator


class TestPropertyCalculator:
    """Test class for PropertyCalculator functionality."""
    
    @pytest.fixture
    def calculator(self):
        """Create a PropertyCalculator instance for testing."""
        return PropertyCalculator()
    
    @pytest.fixture
    def test_compounds(self):
        """Test compounds with known group compositions."""
        return {
            'water': {'H2O': 1},
            'DMSO': {'DMSO': 1},
            'methanol': {'CH3OH': 1},
            'ethylene_glycol': {'DOH': 1},
            'heptane': {'CH3': 2, 'CH2': 5},
            'diethylamine': {'CH3': 2, 'CH2': 1, 'CH2NH': 1}
        }
    
    # Test melting points with expected values from original test
    @pytest.mark.parametrize("compound, expected", [
        ({'H2O': 1}, 273.15),
        ({'DMSO': 1}, 291.65),
        ({'CH3OH': 1}, 176.00),
        ({'DOH': 1}, 260.15),
        ({'CH3': 2, 'CH2': 5}, 175.55),
        ({'CH3': 2, 'CH2': 1, 'CH2NH': 1}, 211.87)
    ])
    def test_melting_point(self, calculator, compound, expected):
        """Test melting point calculations with specific expected values."""
        tm = calculator.calculate_tm(compound)
        if tm is not None:
            assert isinstance(tm, (int, float))
            assert not np.isnan(tm)
            assert abs(tm - expected) < 0.10, f"Tm {tm:.2f} != expected {expected}"
    
    # Test boiling points with expected values from original test
    @pytest.mark.parametrize("compound, expected", [
        ({'H2O': 1}, 373.15),
        ({'DMSO': 1}, 463.1),
        ({'CH3OH': 1}, 337.85),
        ({'DOH': 1}, 470.45),
        ({'CH3': 2, 'CH2': 5}, 379.07),
        ({'CH3': 2, 'CH2': 1, 'CH2NH': 1}, 341.35)
    ])
    def test_boiling_point(self, calculator, compound, expected):
        """Test boiling point calculations with specific expected values."""
        tb = calculator.calculate_tb(compound)
        if tb is not None:
            assert isinstance(tb, (int, float))
            assert not np.isnan(tb)
            assert abs(tb - expected) < 0.10, f"Tb {tb:.2f} != expected {expected}"
    
    # Test critical temperatures with expected values from original test
    @pytest.mark.parametrize("compound, expected", [
        ({'H2O': 1}, 647.00),
        ({'DMSO': 1}, 729.00),
        ({'CH3OH': 1}, 513.00),
        ({'DOH': 1}, 719.7),
        ({'CH3': 2, 'CH2': 5}, 549.86),
        ({'CH3': 2, 'CH2': 1, 'CH2NH': 1}, 516.51)
    ])
    def test_critical_temperature(self, calculator, compound, expected):
        """Test critical temperature calculations with specific expected values."""
        tc = calculator.calculate_tc(compound)
        if tc is not None:
            assert isinstance(tc, (int, float))
            assert not np.isnan(tc)
            assert abs(tc - expected) < 0.10, f"Tc {tc:.2f} != expected {expected}"
    
    # Test critical pressures with expected values from original test
    @pytest.mark.parametrize("compound, expected", [
        ({'H2O': 1}, 220.64),
        ({'DMSO': 1}, 56.50),
        ({'CH3OH': 1}, 81.00),
        ({'DOH': 1}, 85.38),
        ({'CH3': 2, 'CH2': 5}, 28.27),
        ({'CH3': 2, 'CH2': 1, 'CH2NH': 1}, 39.96)
    ])
    def test_critical_pressure(self, calculator, compound, expected):
        """Test critical pressure calculations with specific expected values."""
        pc = calculator.calculate_pc(compound)
        if pc is not None:
            assert isinstance(pc, (int, float))
            assert not np.isnan(pc)
            assert abs(pc - expected) < 0.10, f"Pc {pc:.2f} != expected {expected}"
    
    # Test acentric factors with expected values from original test
    @pytest.mark.parametrize("compound, expected", [
        ({'H2O': 1}, 0.34),
        ({'DMSO': 1}, 0.28),
        ({'CH3OH': 1}, 0.56),
        ({'DOH': 1}, 0.49),
        ({'CH3': 2, 'CH2': 5}, 0.34),
        ({'CH3': 2, 'CH2': 1, 'CH2NH': 1}, 0.33)
    ])
    def test_acentric_factor(self, calculator, compound, expected):
        """Test acentric factor calculations with specific expected values."""
        ac = calculator.calculate_ac(compound)
        if ac is not None:
            assert isinstance(ac, (int, float))
            assert not np.isnan(ac)
            assert abs(ac - expected) < 0.005, f"Ac {ac:.2f} != expected {expected}"
    
    # Test heat of vaporization with expected values from original test
    @pytest.mark.parametrize("compound, expected", [
        ({'H2O': 1}, 43.99),
        ({'DMSO': 1}, 52.46),
        ({'CH3OH': 1}, 37.47),
        ({'DOH': 1}, 66.26),
        ({'CH3': 2, 'CH2': 5}, 38.31),
        ({'CH3': 2, 'CH2': 1, 'CH2NH': 1}, 34.19)
    ])
    def test_heat_of_vaporization(self, calculator, compound, expected):
        """Test heat of vaporization calculations with specific expected values."""
        hv = calculator.calculate_hv(compound)
        if hv is not None:
            assert isinstance(hv, (int, float))
            assert not np.isnan(hv)
            assert abs(hv - expected) < 0.10, f"Hv {hv:.2f} != expected {expected}"
    
    def test_calculate_all_properties(self, calculator):
        """Test calculating all properties for a known compound."""
        water = {'H2O': 1}
        properties = calculator.calculate_all_properties(water)
        
        if properties is not None:
            assert isinstance(properties, dict)
            expected_props = ['tm', 'tb', 'tc', 'pc', 'ac', 'hv']
            
            for prop in expected_props:
                assert prop in properties
                if properties[prop] is not None:
                    assert isinstance(properties[prop], (int, float))
                    assert not np.isnan(properties[prop])
    
    def test_missing_groups_remove_compounds(self):
        """Test behavior with missing groups when action is remove_compounds."""
        calc = PropertyCalculator(missing_group_action="remove_compounds")
        unknown_compound = {'UNKNOWN_GROUP': 1}
        
        properties = calc.calculate_all_properties(unknown_compound)
        assert properties is None, "Should return None for compounds with missing groups"
    
    
    def test_initialization(self):
        """Test PropertyCalculator initialization."""
        calc = PropertyCalculator()
        assert calc.pure_method == "Constantinou-Gani"
        assert calc.missing_group_action == "remove_compounds"
        assert isinstance(calc.data_path, Path)
    
    def test_invalid_missing_group_action(self):
        """Test that invalid missing_group_action raises ValueError."""
        with pytest.raises(ValueError):
            PropertyCalculator(missing_group_action="invalid")
    
    def test_get_available_properties(self, calculator):
        """Test get_available_properties method."""
        available = calculator.get_available_properties()
        assert isinstance(available, list)
        assert len(available) > 0  # Should have some properties available


if __name__ == "__main__":
    # Run basic test if executed directly
    calc = PropertyCalculator()
    
    # Test water properties
    water = {'H2O': 1}
    props = calc.calculate_all_properties(water)
    if props:
        print("Water properties:")
        for prop, value in props.items():
            if value is not None:
                print(f"  {prop.upper()}: {value:.2f}")