import unittest
import pandas as pd

from sgreen.core import Compound, Compounds

class TestCompounds(unittest.TestCase):
    def test_add_compound(self):
        # Test adding compounds to the collection
        water = Compound(name="H2O", groups={"H2O": 1})
        methanol = Compound(name="CH3OH", groups={"CH3": 1, "OH": 1})
        collection = Compounds(compounds=[water])
        collection.add_compound(methanol)
        self.assertIn("H2O", collection.compounds)
        self.assertIn("CH3OH", collection.compounds)

    def test_add_duplicate_compound(self):
        # Test adding a duplicate compound
        water = Compound(name="H2O", groups={"H2O": 1})
        methanol = Compound(name="CH3OH", groups={"CH3": 1, "OH": 1})
        duplicate_methanol = Compound(name="MethanolDuplicate", groups={"CH3": 1, "OH": 1})
        collection = Compounds(compounds=[water, methanol])
        with self.assertRaises(ValueError) as context:
            collection.add_compound(duplicate_methanol)
        self.assertIn("A compound with the same group occurrence already exists", str(context.exception))

    def test_remove_compound(self):
        # Test removing a compound
        water = Compound(name="H2O", groups={"H2O": 1})
        collection = Compounds(compounds=[water])
        collection.remove_compound("H2O")
        self.assertNotIn("H2O", collection.compounds)

    def test_as_dataframe(self):
        # Test converting the collection to a DataFrame
        water = Compound(name="H2O", groups={"H2O": 1})
        methanol = Compound(name="CH3OH", groups={"CH3": 1, "OH": 1})
        collection = Compounds(compounds=[water, methanol])
        df = collection.as_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("H2O", df.columns)
        self.assertIn("CH3OH", df.columns)

    def test_from_dataframe(self):
        # Test constructing a collection from a DataFrame
        df_comps = pd.DataFrame({
            'H2O':   [1, 0, 0, 0, 0, 0],
            'DMSO':  [0, 1, 0, 0, 0, 0],
            'CH3OH': [0, 0, 1, 0, 0, 0],
            'DOH':   [0, 0, 0, 1, 0, 0],
            'CH3':   [0, 0, 0, 0, 2, 2],
            'CH2':   [0, 0, 0, 0, 5, 1],
            'CH2NH': [0, 0, 0, 0, 0, 1]
        })
        collection = Compounds.from_dataframe(df_comps.T)
        self.assertEqual(len(collection.compounds), 7)
        self.assertIn("H2O", collection.compounds)
        self.assertIn("CH3OH", collection.compounds)