"""     """

# imports
import pandas as pd


# define classes

class Compound:
    ALLOWED_GROUPS = None  # Allowed groups placeholder
    GROUP_VALENCY = None  # Group valency placeholder

    @classmethod
    def load_groups(cls, file_path, sheet_name="Groups"):
        """
        Loads the allowed groups from an Excel file.

        :param file_path: Path to the Excel file containing the allowed groups.
        :param sheet_name: Name of the sheet in the Excel file (default is 'Groups').
        """
        try:
            cls.ALLOWED_GROUPS = pd.read_excel(file_path, sheet_name=sheet_name).iloc[:, 0].tolist()
        except Exception as e:
            raise ValueError(f"Failed to load allowed groups from {file_path}: {e}")

    @classmethod
    def load_valency(cls, file_path, sheet_name="noc"):
        """
        Loads the group valency (number of connections) from an Excel file.

        :param file_path: Path to the Excel file containing the group valency.
        :param sheet_name: Name of the sheet in the Excel file (default is 'noc').
        """
        try:
            valency_df = pd.read_excel(file_path, sheet_name=sheet_name)
            cls.GROUP_VALENCY = valency_df.set_index("grp")["noc"].to_dict()
        except Exception as e:
            raise ValueError(f"Failed to load group valency from {file_path}: {e}")

    def __init__(self, name, groups, m=1):
        """
        Initializes a Compound object.

        :param name: Name of the compound.
        :param groups: Dictionary of group occurrences (e.g., {'CH3': 2, 'CH2': 5}).
        :param m: Parameter for the octet rule validation (default is 1).
        """
        self.name = name
        self.groups = groups
        self.m = m
        self._validate_groups()
        self._validate_connectivity()
        self.df = self._to_dataframe()

    def _validate_groups(self):
        """
        Validates that all groups in the compound are part of the allowed groups.

        :raises ValueError: If any group is invalid.
        """
        invalid_groups = [group for group in self.groups.keys() if group not in self.ALLOWED_GROUPS]
        if invalid_groups:
            raise ValueError(f"Invalid groups detected: {invalid_groups}. Allowed groups are: {self.ALLOWED_GROUPS}")

    def _validate_connectivity(self):
        """
        Validates the connectivity of the compound based on two checks:
        1. Octet rule: sum_over_groups((2 - number_of_connections) * occurrence) = 2 * m
        2. Connectivity inequality: Σ n_i ≥ n_j(v_j - 1) + 2 for each group.
        based on the odele and macchietto rule (1993)

        :raises ValueError: If either of the checks fails.
        """
        # Octet rule check
        total_octet = sum(
            (2 - Compound.GROUP_VALENCY.get(group, 0)) * occurrence
            for group, occurrence in self.groups.items()
        )
        if total_octet != 2 * self.m:
            raise ValueError(
                f"Octet rule check failed for compound '{self.name}'. "
                f"Calculated total: {total_octet}, Expected: {2 * self.m}. "
                f"Defined groups and occurrences: {self.groups}"
            )

        # Connectivity inequality check
        total_occurrences = sum(self.groups.values())  # Σ n_i
        for group, occurrence in {k: v for k, v in self.groups.items() if v > 0}.items():
            valency = Compound.GROUP_VALENCY.get(group, 0)
            if valency == 0:  # Skip groups with noc = 0
                continue
            rhs = occurrence * (valency - 1) + 2  # n_j(v_j - 1) + 2
            if total_occurrences < rhs:
                raise ValueError(
                    f"Connectivity inequality check failed for compound '{self.name}'. "
                    f"Group: {group}, Occurrence: {occurrence}, Σ n_i: {total_occurrences}, RHS: {rhs}. "
                    f"Defined groups and occurrences: {self.groups}"
                )

    def _to_dataframe(self):
        """
        Converts the group dictionary to a pandas DataFrame.

        :return: A pandas DataFrame with the group occurrences.
        """
        df = pd.DataFrame({self.name: self.groups}).T
        return df.fillna(0)

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
        Constructs a CompoundCollection from a pandas DataFrame.

        :param df: A pandas DataFrame where rows represent groups and columns represent compounds.
        :return: A CompoundCollection object.
        """
        compounds = [
            Compound(name=name if isinstance(name, str) and name.strip() else f"idx_{idx}", groups=df[name].to_dict())
            for idx, name in enumerate(df.columns)
        ]
        return cls(compounds=compounds)

    def __repr__(self):
        """
        Returns a string representation of the CompoundCollection.

        :return: String representation of the collection.
        """
        return f"CompoundCollection(compounds={list(self.compounds.keys())})"