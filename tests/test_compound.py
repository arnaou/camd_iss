import unittest
import pandas as pd

from sgreen.core import Compound

class TestCompound(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load group and valency data
        Compound.load_groups("data/data_unifac.xlsx", sheet_name="Groups")
        Compound.load_valency("data/data_unifac.xlsx", sheet_name="noc")

    def test_valid_compound(self):
        # Test creation of a valid compound
        water = Compound(name="H2O", groups={"H2O": 1})
        self.assertEqual(water.name, "H2O")
        self.assertEqual(water.groups, {"H2O": 1})

    def test_invalid_compound(self):
        # Test creation of an invalid compound
        with self.assertRaises(ValueError) as context:
            Compound(name="Invalid", groups={"XYZ": 1, "CH3": 2})
        self.assertIn("Invalid groups detected", str(context.exception))

    def test_connectivity(self):
        # Test design space check failure
        with self.assertRaises(ValueError) as context:
            Compound(name="Invalid", groups={"CH3": 1, "CH2": 1}, m=1)
        self.assertIn("connectivity check failed", str(context.exception))
            # Test design space check failure for CH3 and H2O
        with self.assertRaises(ValueError) as context:
            Compound(name="Invalid", groups={"CH3": 1, "H2O": 1}, m=1)
        self.assertIn("connectivity check failed", str(context.exception))