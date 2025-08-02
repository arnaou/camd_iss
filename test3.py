import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from sgreen.dataset import Dataset

class Calculator:
    def __init__(self, dataset):
        """
        Initializes the Calculator by using the provided Dataset object.

        :param dataset: A Dataset object containing group contribution data and UNIFAC parameters.
        """
        self.dataset = dataset
        self.contrib = self.dataset.get_contribs(output_format="dataframe")  # Load group contribution data
        self.unifac = self.dataset.get_unifac(output_format="dataframe")  # Load UNIFAC parameters
        self.property_methods = {
            'mw': self._calculate_mw,
            'tm': self._calculate_tm,
            'tb': self._calculate_tb,
            'tc': self._calculate_tc,
            'pc': self._calculate_pc,
            'vc': self._calculate_vc,
            'gf': self._calculate_gf,
            'hf': self._calculate_hf,
            'hv': self._calculate_hv,
            'vm': self._calculate_vm,
            'ac': self._calculate_ac,
            'zc': self._calculate_zc,
            'vm_tb': self._calculate_vm_tb,
            'tfc': self._calculate_tfc,
            'tfo': self._calculate_tfo,
            'hds': self._calculate_hds,
            'hps': self._calculate_hps,
            'hhs': self._calculate_hhs,
            'sol_par': self._calculate_sol_par,
            'lc50': self._calculate_lc50,
        }
        self.property_descriptions = {
            'mw': 'Molar weight in g/mol',
            'tm': 'Normal melting point in K',
            'tb': 'Normal boiling point in K',
            'tc': 'Critical temperature in K',
            'pc': 'Critical pressure in bar',
            'vc': 'Critical volume in cm³/mol',
            'gf': 'Gibbs free energy of formation in kJ/mol',
            'hf': 'Enthalpy of formation in kJ/mol',
            'hv': 'Enthalpy of vaporization in kJ/mol',
            'vm': 'Molar volume in cm³/mol',
            'ac': 'Acentric factor (unitless)',
            'zc': 'Critical compressibility factor (unitless)',
            'vm_tb': 'Molar volume at boiling point in cm³/mol',
            'tfc': 'Closed cup flash point in K',
            'tfo': 'Open cup flash point in K',
            'hds': 'Hansen dispersion solubility parameter in MPa^0.5',
            'hps': 'Hansen polar solubility parameter in MPa^0.5',
            'hhs': 'Hansen hydrogen bonding solubility parameter in MPa^0.5',
            'sol_par': 'Hildebrand Solubility parameter in MPa^0.5',
            'lc50': 'Median lethal concentration (LC50) in mg/L',
        }

    def inspect_contribs(self, compound_collection, prop_list, verbose=True):
        """
        Checks for missing contributions in the group contribution data for present groups and selected properties.

        :param compound_collection: A CompoundCollection object containing compounds.
        :param prop_list: A list of properties to inspect.
        :param verbose: If True, warnings are printed for missing contributions.
        :return: Dictionary of properties with missing contributions and the corresponding groups.
        """
        # Map prop_list to match self.contrib column names
        prop_mapping = {
            'mw': 'Mw',
            'tm': 'Tm',
            'tb': 'Tb',
            'tc': 'Tc',
            'pc': 'Pc',
            'vc': 'Vc',
            'gf': 'Gf',
            'hf': 'Hf',
            'hv': 'Hv',
            'vm': 'Vm',
            'ac': 'Ac',
            'zc': 'Zc',
            'vm_tb': 'Vm_tb',
            'tfc': 'tfc',
            'tfo': 'tfo',
            'hds': 'FDX',
            'hps': 'FPX',
            'hhs': 'FHX',
            'sol_par': 'sol_par',
            'lc50': 'LC50',
        }
        mapped_props = [prop_mapping[prop] for prop in prop_list if prop in prop_mapping]

        # Get group occurrences from the CompoundCollection
        group_occurrences = compound_collection.get_group_occurrences()
        present_groups = [group for group, count in group_occurrences.items() if count > 0]

        # Filter the group contribution data for present groups
        contrib_filtered = self.contrib.loc[present_groups, mapped_props]

        # Find zero entries
        zero_entries = contrib_filtered == 0

        # Set missing contributions to 0
        contrib_filtered[zero_entries] = 0

        # Stack the DataFrame to convert it into a Series with multi-index
        zero_entries_stacked = zero_entries.stack()

        # Filter where values are True (i.e., where the original values are 0)
        zero_tuples = zero_entries_stacked[zero_entries_stacked].index.tolist()

        # Group by column
        grouped_by_column = defaultdict(list)
        for idx, col in zero_tuples:
            grouped_by_column[col].append(idx)
            if col not in ["noc", "ccsub"] and verbose:
                warnings.warn(f"Warning: some groups have a 0 contribution for the property: Column '{col}'")
        grouped_by_column = dict(grouped_by_column)
        if bool(grouped_by_column):
            print('Some properties and groups have missing contributions.')
        else:
            print('All contributions for each group and property are present.')

        return grouped_by_column

    def calculate_properties(self, compound_collection, prop_list, round_digits=None):
        """
        Calculates various thermochemical properties using group contribution methods.

        :param compound_collection: A CompoundCollection object containing compounds.
        :param prop_list: A list of properties to calculate.
        :param round_digits: Number of decimal places to round the results to (Optional).
        :return: A pandas DataFrame with the calculated properties for each compound.
        """
        # Check for missing contributions
        missing_contribs = self.inspect_contribs(compound_collection, prop_list, verbose=False)
        if missing_contribs:
            print(f"Warning: Missing contributions detected for the following properties: {missing_contribs}")

        results = {}

        # Iterate over each compound in the collection
        for compound_name, compound in compound_collection.compounds.items():
            compound_results = {}
            for prop in prop_list:
                if prop in self.property_methods:
                    compound_results[prop] = self.property_methods[prop](compound.groups)
                else:
                    raise ValueError(f"Property '{prop}' is not supported.")
            results[compound_name] = compound_results

        # Convert results to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index')

        # Apply rounding if specified
        if round_digits is not None:
            results_df = results_df.round(round_digits)

        return results_df
    

    def info(self):
        """
        Prints an overview of the implemented properties and their descriptions.
        """
        print("Available Properties:")
        for prop, desc in self.property_descriptions.items():
            print(f"- {prop}: {desc}")

    # Individual property calculation methods
    def _calculate_mw(self, group_occurrences):
        """Calculates the molar weight"""
        return sum(self.contrib['Mw'][group] * count for group, count in group_occurrences.items())

    def _calculate_tm(self, group_occurrences):
        """Calculates the normal melting point"""
        return 102.425 * np.log(sum(self.contrib['Tm'][group] * count for group, count in group_occurrences.items()))

    def _calculate_tb(self, group_occurrences):
        """Calculates the normal boiling point"""
        return 204.359 * np.log(sum(self.contrib['Tb'][group] * count for group, count in group_occurrences.items()))

    def _calculate_tc(self, group_occurrences):
        """Calculates the critical temperature"""
        return 181.128 * np.log(sum(self.contrib['Tc'][group] * count for group, count in group_occurrences.items()))

    def _calculate_pc(self, group_occurrences):
        """Calculates the critical pressure"""
        return 1 / (sum(self.contrib['Pc'][group] * count for group, count in group_occurrences.items()) + 0.100220) ** 2 + 1.3705

    def _calculate_vc(self, group_occurrences):
        """Calculates the critical volume"""
        return -4.35 + 1000 * sum(self.contrib['Vc'][group] * count for group, count in group_occurrences.items())

    def _calculate_gf(self, group_occurrences):
        """Calculates the Gibbs free energy of formation"""
        return -14.828 + sum(self.contrib['Gf'][group] * count for group, count in group_occurrences.items())

    def _calculate_hf(self, group_occurrences):
        """Calculates the enthalpy of formation"""
        return 10.835 + sum(self.contrib['Hf'][group] * count for group, count in group_occurrences.items())

    def _calculate_hv(self, group_occurrences):
        """Calculates the enthalpy of vaporization"""
        return 6.829 + sum(self.contrib['Hv'][group] * count for group, count in group_occurrences.items())

    def _calculate_vm(self, group_occurrences):
        """Calculates the molar volume"""
        return 12.11 + sum(1000 * self.contrib['Vm'][group] * count for group, count in group_occurrences.items())

    def _calculate_ac(self, group_occurrences):
        """Calculates the acentric factor"""
        return 0.4085 * np.log(sum(self.contrib['Ac'][group] * count for group, count in group_occurrences.items()) + 1.1507) ** 1.9802

    def _calculate_zc(self, group_occurrences):
        """Calculates the critical compressibility factor"""
        pc = self._calculate_pc(group_occurrences)
        vc = self._calculate_vc(group_occurrences)
        tc = self._calculate_tc(group_occurrences)
        return pc * vc / (83.14 * tc)

    def _calculate_vm_tb(self, group_occurrences):
        """Calculates the molar volume at boiling point"""
        vc = self._calculate_vc(group_occurrences)
        return 0.285 * vc ** 1.048

    def _calculate_tfc(self, group_occurrences):
        """Calculates the closed cup flash point"""
        tb = self._calculate_tb(group_occurrences)
        return -2.03 * sum(self.contrib['tfc'][group] * count for group, count in group_occurrences.items()) + 0.659 * tb + 20.00

    def _calculate_tfo(self, group_occurrences):
        """Calculates the open cup flash point"""
        tb = self._calculate_tb(group_occurrences)
        return 3.63 * sum(self.contrib['tfo'][group] * count for group, count in group_occurrences.items()) + 0.409 * tb + 88.43

    def _calculate_hds(self, group_occurrences):
        """Calculates the Hansen dispersion solubility parameter"""
        vm = self._calculate_vm(group_occurrences)
        return sum(self.contrib['FDX'][group] * count for group, count in group_occurrences.items()) / vm

    def _calculate_hps(self, group_occurrences):
        """Calculates the Hansen polar solubility parameter"""
        vm = self._calculate_vm(group_occurrences)
        return (sum(self.contrib['FPX'][group] ** 2 * count for group, count in group_occurrences.items())) ** 0.5 / vm

    def _calculate_hhs(self, group_occurrences):
        """Calculates the Hansen hydrogen bonding solubility parameter"""
        vm = self._calculate_vm(group_occurrences)
        return (sum(self.contrib['FHX'][group] * count for group, count in group_occurrences.items()) / vm) ** 0.5

    def _calculate_sol_par(self, group_occurrences):
        """Calculates the Hildebrand (total) solubility parameter"""
        hv = self._calculate_hv(group_occurrences)
        vm = self._calculate_vm(group_occurrences)
        return ((1000 * hv - 8.314 * 298.15) / vm) ** 0.5

    def _calculate_lc50(self, group_occurrences):
        """Calculates the median lethal concentration (LC50)"""
        return sum(self.contrib['LC50'][group] * count for group, count in group_occurrences.items())


from sgreen.dataset import Dataset
from sgreen.camd import CompoundCollection, Compound


def main():
    # Load allowed groups and valency
    Compound.load_groups(file_path="data/data_unifac.xlsx", sheet_name="Groups")
    Compound.load_valency(file_path="data/data_unifac.xlsx", sheet_name="noc")

    # Initialize the Dataset
    dataset = Dataset(file_path="data/data_unifac.xlsx", output_format="dataframe")

    # Initialize the Calculator
    calculator = Calculator(dataset)

    # Create a CompoundCollection
    compound_collection = CompoundCollection()
    compound_collection.add_compound(Compound(name="H2O", groups={"H2O": 1}))
    compound_collection.add_compound(Compound(name="CH3OH", groups={"CH3": 1, "OH": 1}))

    # Calculate properties
    properties = calculator.calculate_properties(compound_collection, ['mw', 'tm', 'tb', 'vc'], round_digits=2)
    print("Properties:")
    print(properties)

if __name__ == "__main__":
    main()