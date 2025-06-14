import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("utils"))
from utils.gc import load_gc
import time

class Calculator:
    def __init__(self):
        """
        Initializes the pure component Calculator by loading group contribution data.

        :param file_path: Path to the Excel file containing group contribution data.
        :param output_format: Format of the loaded data ('dataframe' or 'dict').
        """
        self.contrib, _ = load_gc('data/data_unifac.xlsx', 'dataframe')
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

    def calculate_properties(self, mol, prop_list, round_digits=None):
        """
        Calculates various thermochemical properties using group contribution methods.

        :param mol: A pandas DataFrame with the occurrence of the groups.
        :param prop_list: A list of properties to calculate.
        :param round_digits: Number of decimal places to round the results to (Optional).
        :return: A pandas DataFrame with the calculated properties.
        """
        if not isinstance(mol, pd.DataFrame):
            raise ValueError("mol must be a pandas DataFrame.")
        if not isinstance(prop_list, list):
            raise ValueError("prop_list must be a list of property names.")

        idx = mol.columns
        results = {}

        for prop in prop_list:
            if prop in self.property_methods:
                results[prop] = self.property_methods[prop](mol, idx)
            else:
                raise ValueError(f"Property '{prop}' is not supported.")

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

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
    def _calculate_mw(self, mol, idx):
        """Calculates the molar weight"""
        return sum(self.contrib['Mw'][i] * mol[i] for i in idx)

    def _calculate_tm(self, mol, idx):
        """Calculates the normal melting point"""
        return 102.425 * np.log(sum(self.contrib['Tm'][i] * mol[i] for i in idx))

    def _calculate_tb(self, mol, idx):
        """Calculates the normal boiling point"""
        return 204.359 * np.log(sum(self.contrib['Tb'][i] * mol[i] for i in idx))

    def _calculate_tc(self, mol, idx):
        """Calculates the critical temperature"""
        return 181.128 * np.log(sum(self.contrib['Tc'][i] * mol[i] for i in idx))

    def _calculate_pc(self, mol, idx):
        """Calculates the critical pressure"""
        return 1 / (sum(self.contrib['Pc'][i] * mol[i] for i in idx) + 0.100220) ** 2 + 1.3705

    def _calculate_vc(self, mol, idx):
        """Calculates the critical volume"""
        return -4.35 + 1000 * sum(self.contrib['Vc'][i] * mol[i] for i in idx)

    def _calculate_gf(self, mol, idx):
        """Calculates the Gibbs free energy of formation"""
        return -14.828 + sum(self.contrib['Gf'][i] * mol[i] for i in idx)

    def _calculate_hf(self, mol, idx):
        """Calculates the enthalpy of formation"""
        return 10.835 + sum(self.contrib['Hf'][i] * mol[i] for i in idx)

    def _calculate_hv(self, mol, idx):
        """Calculates the enthalpy of vaporization"""
        return 6.829 + sum(self.contrib['Hv'][i] * mol[i] for i in idx)

    def _calculate_vm(self, mol, idx):
        """Calculates the molar volume"""
        return 12.11 + sum(1000 * self.contrib['Vm'][i] * mol[i] for i in idx)

    def _calculate_ac(self, mol, idx):
        """Calculates the acentric factor"""
        return 0.4085 * np.log(sum(self.contrib['Ac'][i] * mol[i] for i in idx) + 1.1507) ** 1.9802

    def _calculate_zc(self, mol, idx):
        """Calculates the critical compressibility factor"""
        pc = self._calculate_pc(mol, idx)
        vc = self._calculate_vc(mol, idx)
        tc = self._calculate_tc(mol, idx)
        return pc * vc / (83.14 * tc)

    def _calculate_vm_tb(self, mol, idx):
        """Calculates the molar volume at boiling point"""
        vc = self._calculate_vc(mol, idx)
        return 0.285 * vc ** 1.048

    def _calculate_tfc(self, mol, idx):
        """Calculates the closed cup flash point"""
        tb = self._calculate_tb(mol, idx)
        return -2.03 * sum(self.contrib['tfc'][i] * mol[i] for i in idx) + 0.659 * tb + 20.00
    
    def _calculate_tfo(self, mol, idx):
        """Calculates the open cup flash point"""
        tb = self._calculate_tb(mol, idx)
        return  3.63 * sum(self.contrib['tfo'][i] * mol[i] for i in idx) + 0.409 * tb + 88.43

    def _calculate_hds(self, mol, idx):
        """Calculates the Hansen dispersion solubility parameter"""
        vm = self._calculate_vm(mol, idx)
        return (sum(self.contrib['FDX'][i] * mol[i] for i in idx))/ vm

    def _calculate_hps(self, mol, idx):
        """Calculates the Hansen polar solubility parameter"""
        vm = self._calculate_vm(mol, idx)
        return (sum(self.contrib['FPX'][i] * self.contrib['FPX'][i] * mol[i] for i in idx)) ** 0.5 / vm

    def _calculate_hhs(self, mol, idx):
        """Calculates the Hansen hydrogen bonding solubility parameter"""
        vm = self._calculate_vm(mol, idx)
        return (sum(self.contrib['FHX'][i] * mol[i] for i in idx) / vm) ** 0.5
    
    def _calculate_sol_par(self, mol, idx):
        """Calculates the Hildebrand (total) solubility parameter"""
        hv = self._calculate_hv(mol, idx)
        vm = self._calculate_vm(mol, idx)
        return ((1000 * hv - 8.314 * 298.15) / vm) ** 0.5

    def _calculate_lc50(self, mol, idx):
        """Calculates the median lethal concentration (LC50)"""
        return sum(self.contrib['LC50'][i] * mol[i] for i in idx)



# %%
class Compound:
    def __init__(self, name, groups):
        """
        Initializes a Compound object.

        :param name: Name of the compound.
        :param groups: Dictionary of group occurrences (e.g., {'CH3': 2, 'CH2': 5}).
        """
        self.name = name
        self.groups = groups
        self.df = self._to_dataframe()

    def _to_dataframe(self):
        """
        Converts the group dictionary to a pandas DataFrame.

        :return: A pandas DataFrame with the group occurrences.
        """
        df = pd.DataFrame({self.name: self.groups}).T
        return df.fillna(0)  # Fill missing values with 0

    def as_dict(self):
        """
        Returns the group occurrences as a dictionary.

        :return: Dictionary of group occurrences.
        """
        return self.groups

    def as_dataframe(self):
        """
        Returns the group occurrences as a pandas DataFrame.

        :return: A pandas DataFrame with the group occurrences.
        """
        return self.df

    def __repr__(self):
        return f"Compound(name={self.name}, groups={self.groups})"
    



class CompoundCollection:
    def __init__(self, compounds=None):
        """
        Initializes a CompoundCollection object.

        :param compounds: List of Compound objects (optional).
        """
        self.compounds = {compound.name: compound for compound in (compounds or [])}

    def add_compound(self, compound):
        """
        Adds a new compound to the collection.

        :param compound: A Compound object.
        :raises ValueError: If a compound with the same group contributions already exists.
        """
        for existing_compound in self.compounds.values():
            if existing_compound.groups == compound.groups:
                raise ValueError(f"A compound with the same group occurrence already exists: {existing_compound.name}")
        self.compounds[compound.name] = compound

    def remove_compound(self, name):
        """
        Removes a compound from the collection by name.

        :param name: Name of the compound to remove.
        :raises ValueError: If the compound does not exist in the collection.
        """
        if name in self.compounds:
            del self.compounds[name]
        else:
            raise ValueError(f"Compound '{name}' not found in the collection.")

    def get_compound(self, name):
        """
        Retrieves a compound from the collection by name.

        :param name: Name of the compound to retrieve.
        :return: The Compound object.
        """
        return self.compounds.get(name, None)

    def as_dataframe(self):
        """
        Combines all compounds into a single pandas DataFrame.

        :return: A pandas DataFrame representing the collection.
        """
        return pd.concat([compound.as_dataframe() for compound in self.compounds.values()]).fillna(0)

    def as_dict(self):
        """
        Returns the collection as a dictionary of compounds and their group contributions.

        :return: Dictionary representation of the collection.
        """
        return {name: compound.as_dict() for name, compound in self.compounds.items()}

    def __repr__(self):
        return f"CompoundCollection(compounds={list(self.compounds.keys())})"



def main():
    # Initialize the Calculator
    pure_comp = Calculator()
    print("Available Properties:")
    pure_comp.info()

    # Create Compound objects
    water = Compound(name="H2O", groups={"H2O": 1})
    methanol = Compound(name="CH3OH", groups={"CH3": 1, "OH": 1})
    duplicate_methanol = Compound(name="MethanolDuplicate", groups={"CH3": 1, "OH": 1})

    # Create a CompoundCollection object
    collection = CompoundCollection(compounds=[water])

    # Add a new compound
    collection.add_compound(methanol)

    # Attempt to add a duplicate compound (based on groups)
    try:
        collection.add_compound(duplicate_methanol)
    except ValueError as e:
        print(e)  # Output: A compound with the same group contributions already exists: CH3OH

    # Access the collection as a DataFrame
    print("Compound Collection as DataFrame:")
    print(collection.as_dataframe())

    # Define the input DataFrame for property calculations
    df_comps = pd.DataFrame({
        'H2O':   [1, 0, 0, 0, 0, 0],
        'DMSO':  [0, 1, 0, 0, 0, 0],
        'CH3OH': [0, 0, 1, 0, 0, 0],
        'DOH':   [0, 0, 0, 1, 0, 0],
        'CH3':   [0, 0, 0, 0, 2, 2],
        'CH2':   [0, 0, 0, 0, 5, 1],
        'CH2NH': [0, 0, 0, 0, 0, 1]
    })

    # Measure execution time for property calculations
    start_time = time.time()
    pred = pure_comp.calculate_properties(
        df_comps,
        ['mw', 'tm', 'tb', 'tc', 'pc', 'vc', 'gf', 'hf', 'hv', 'vm', 'ac', 'zc', 'vm_tb', 'tfc', 'tfo', 'sol_par', 'lc50'],
        round_digits=2
    )
    end_time = time.time()

    # Print the execution time
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    # Print the calculated properties
    print("Calculated Properties:")
    print(pred)

# Entry point for the script
if __name__ == "__main__":
    main()