"""Compound class for representing individual chemical compounds with group contributions."""

from typing import Dict, Optional, Union, Any, TYPE_CHECKING
from copy import deepcopy
import pandas as pd
import json
from .exceptions import CompoundError, ValidationError, ConnectivityError, GroupDataError

if TYPE_CHECKING:
    try:
        from rdkit import Chem
        from PIL import Image
    except ImportError:
        pass


class Compound:
    """
    Represents a chemical compound using group contribution methods.
    
    A compound is defined by its name and functional group occurrences,
    following the UNIFAC group contribution method. The class validates
    molecular connectivity using the Odele and Macchietto (1993) rules.
    
    Attributes:
        name (str): Name or identifier of the compound
        groups (Dict[str, int]): Dictionary mapping group names to their occurrences
        m (int): Number of molecules for octet rule validation (default: 1)
        
    Class Attributes:
        ALLOWED_GROUPS (Optional[List[str]]): List of valid functional groups
        GROUP_VALENCY (Optional[Dict[str, int]]): Mapping of groups to their valencies
        
    Examples:
        >>> # Load group data first
        >>> Compound.load_groups("data/unifac_groups.xlsx")
        >>> Compound.load_valency("data/unifac_valency.xlsx")
        >>> Compound.load_smarts("data/unifac_smarts.json")
        >>> 
        >>> # Create compounds
        >>> water = Compound.from_groups("H2O", {"H2O": 1})
        >>> ethanol = Compound.from_groups("Ethanol", {"CH3": 1, "CH2": 1, "OH": 1})
        >>> heptanol = Compound.from_groups("1-Heptanol", {"CH3": 1, "CH2": 6, "OH": 1})
        >>> 
        >>> print(water.name)  # "H2O"
        >>> print(ethanol.groups)  # {"CH3": 1, "CH2": 1, "OH": 1}
        
    References:
        Odele, O., & Macchietto, S. (1993). Computer aided molecular design: 
        A novel method for optimal solvent selection. Fluid Phase Equilibria, 
        82, 47-54.
    """
    ALLOWED_GROUPS: Optional[list[str]] = None
    GROUP_VALENCY: Optional[Dict[str, int]] = None
    GROUP_SMARTS: Optional[Dict[str, str]] = None

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

    @classmethod
    def load_smarts(cls, file_path: str) -> None:
        """
        Loads the SMARTS patterns for groups from a JSON file.

        Args:
            file_path: Path to the JSON file containing SMARTS patterns
            
        Raises:
            GroupDataError: If loading fails due to file or format issues
            
        Examples:
            >>> Compound.load_smarts("data/unifac_smarts.json")
            >>> Compound.load_smarts("data/group_patterns.json")
            
        Note:
            Expected JSON format: {"group_name": "smarts_pattern", ...}
            Example: {"CH3": "[CH3;X4]", "CH2": "[CH2;X4]", "OH": "[OH]"}
            Based on patterns from: https://github.com/simonmb/fragmentation_algorithm
        """
        if not isinstance(file_path, str):
            raise GroupDataError("file_path must be a string")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                smarts_data = json.load(f)
            
            if not isinstance(smarts_data, dict):
                raise GroupDataError("JSON file must contain a dictionary")
            if not smarts_data:
                raise GroupDataError("JSON file is empty or contains no data")
                
            # Validate that all values are strings
            for group, smarts in smarts_data.items():
                if not isinstance(group, str):
                    raise GroupDataError(f"Group name must be string, got {type(group)}")
                if not isinstance(smarts, str):
                    raise GroupDataError(f"SMARTS pattern for '{group}' must be string, got {type(smarts)}")
                    
            cls.GROUP_SMARTS = smarts_data.copy()
            
        except FileNotFoundError:
            raise GroupDataError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise GroupDataError(f"Invalid JSON format in {file_path}: {e}")
        except Exception as e:
            raise GroupDataError(f"Failed to load SMARTS patterns from {file_path}: {e}")

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
    
    # Visualization and molecular structure methods
    def to_smiles(self) -> str:
        """
        Convert compound to SMILES representation.
        
        Returns:
            SMILES string representation of the compound
            
        Raises:
            NotImplementedError: This method is not yet implemented
            ValidationError: If SMARTS data is not loaded
            
        Examples:
            >>> # After loading SMARTS data
            >>> Compound.load_smarts("data/unifac_smarts.json")
            >>> ethanol = Compound.from_groups("Ethanol", {"CH3": 1, "CH2": 1, "OH": 1})
            >>> smiles = ethanol.to_smiles()  # "CCO"
            
        Note:
            Requires SMARTS patterns to be loaded via load_smarts().
            Implementation will:
            1. Map groups to SMARTS fragments
            2. Build molecular graph using valency constraints
            3. Convert graph to SMILES using RDKit
        """
        if self.GROUP_SMARTS is None:
            raise ValidationError(
                "SMARTS data not loaded. Call Compound.load_smarts() first."
            )
        
        raise NotImplementedError(
            "SMILES generation from UNIFAC groups is not yet implemented. "
            "This will require molecular graph construction and RDKit integration."
        )
    
    def to_mol(self) -> 'Chem.Mol':
        """
        Convert compound to RDKit Mol object.
        
        Returns:
            RDKit Mol object representing the compound
            
        Raises:
            NotImplementedError: This method is not yet implemented
            ImportError: If RDKit is not installed
            
        Examples:
            >>> # After loading SMARTS data and installing RDKit
            >>> ethanol = Compound.from_groups("Ethanol", {"CH3": 1, "CH2": 1, "OH": 1})
            >>> mol = ethanol.to_mol()
            >>> # Use mol with RDKit functions
            
        Note:
            Requires RDKit to be installed: pip install rdkit
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError(
                "RDKit is required for molecular object generation. "
                "Install with: pip install rdkit"
            )
        
        raise NotImplementedError(
            "RDKit Mol generation is not yet implemented. "
            "This will be built on top of to_smiles() functionality."
        )
    
    def visualize(self, size: tuple[int, int] = (300, 300), **kwargs) -> 'Image':
        """
        Generate 2D molecular structure visualization.
        
        Args:
            size: Image size as (width, height) tuple (default: (300, 300))
            **kwargs: Additional arguments passed to RDKit drawing functions
            
        Returns:
            PIL Image object containing the 2D molecular structure
            
        Raises:
            NotImplementedError: This method is not yet implemented
            ImportError: If RDKit or PIL is not installed
            
        Examples:
            >>> # After loading SMARTS data and installing dependencies
            >>> ethanol = Compound.from_groups("Ethanol", {"CH3": 1, "CH2": 1, "OH": 1})
            >>> img = ethanol.visualize(size=(400, 400))
            >>> img.save("ethanol.png")
            >>> img.show()  # Display in Jupyter notebook
            
        Note:
            Requires RDKit and PIL: pip install rdkit pillow
            Uses RDKit's drawing capabilities for 2D structure generation
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                f"Required packages not installed: {e}. "
                "Install with: pip install rdkit pillow"
            )
        
        raise NotImplementedError(
            "Molecular visualization is not yet implemented. "
            "This will use RDKit's Draw module to generate 2D structures "
            "from the molecular graph constructed from UNIFAC groups."
        )
    
    def _build_molecular_graph(self) -> dict:
        """
        Build molecular graph from UNIFAC groups using connectivity rules.
        
        Returns:
            Dictionary representing molecular graph structure
            
        Raises:
            NotImplementedError: This method is not yet implemented
            
        Note:
            This is a core method that will:
            1. Map each group to its SMARTS pattern
            2. Determine connection points using valency data
            3. Build a connected molecular graph
            4. Validate connectivity using Odele & Macchietto rules
            5. Return graph suitable for SMILES/RDKit conversion
        """
        raise NotImplementedError(
            "Molecular graph construction is not yet implemented. "
            "This will be the core algorithm for connecting UNIFAC groups "
            "into valid molecular structures."
        )