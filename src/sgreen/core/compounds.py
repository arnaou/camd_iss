"""Compounds class for managing collections of chemical compounds."""

import pandas as pd
from .compound import Compound


class Compounds:
    def __init__(self, compounds=None):
        """
        Initializes a Compounds object.

        :param compounds: List of Compound objects (optional).
        """
        self.compounds = {compound.name: compound for compound in (compounds or [])}
        self.counter = len(self.compounds)  # Counter for unnamed compounds

    def add_compound(self, compound):
        """
        Adds a new compound to the collection.

        :param compound: A Compound object.
        :raises ValueError: If a compound with the same group contributions already exists.
        """
        if not compound.name:
            compound.name = f"idx_{self.counter}"
            self.counter += 1

        # Check for duplicate group occurrences
        if any(existing_compound.groups == compound.groups for existing_compound in self.compounds.values()):
            raise ValueError(
                f"A compound with the same group occurrence already exists: {compound.name}"
            )

        self.compounds[compound.name] = compound
        
    def get_group_occurrences(self):
        """
        Retrieves the group occurrences for all compounds in the collection.

        :return: A dictionary where keys are group names and values are their occurrences.
        """
        group_occurrences = {}
        for compound in self.compounds.values():
            for group, count in compound.groups.items():
                group_occurrences[group] = group_occurrences.get(group, 0) + count
        return group_occurrences

    def get_compound_names(self):
        """
        Retrieves the names of all compounds in the collection.

        :return: A list of compound names.
        """
        return list(self.compounds.keys())

    def remove_compound(self, name):
        """
        Removes a compound from the collection by name.

        :param name: Name of the compound to remove.
        :raises ValueError: If the compound does not exist in the collection.
        """
        if name not in self.compounds:
            raise ValueError(f"Compound '{name}' not found in the collection.")
        del self.compounds[name]

    def get_compound(self, name):
        """
        Retrieves a compound from the collection by name.

        :param name: Name of the compound to retrieve.
        :return: The Compound object, or None if not found.
        """
        return self.compounds.get(name)

    def as_dataframe(self):
        """
        Combines all compounds into a single pandas DataFrame.

        :return: A pandas DataFrame representing the collection.
        """
        return pd.concat([compound.as_dataframe() for compound in self.compounds.values()], axis=0).fillna(0)

    def as_dict(self):
        """
        Returns the collection as a dictionary of compounds and their group contributions.

        :return: Dictionary representation of the collection.
        """
        return {name: compound.as_dict() for name, compound in self.compounds.items()}

    @classmethod
    def from_dataframe(cls, df):
        """
        Constructs a Compounds from a pandas DataFrame.

        :param df: A pandas DataFrame where rows represent groups and columns represent compounds.
        :return: A Compounds object.
        """
        compounds = [
            Compound(name=name if isinstance(name, str) and name.strip() else f"idx_{idx}", groups=df[name].to_dict())
            for idx, name in enumerate(df.columns)
        ]
        return cls(compounds=compounds)

    def __repr__(self):
        """
        Returns a string representation of the Compounds.

        :return: String representation of the collection.
        """
        return f"Compounds(compounds={list(self.compounds.keys())})"