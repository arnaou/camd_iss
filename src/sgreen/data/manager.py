"""Data management module for Excel->JSON workflow and data loading."""

import json
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
from datetime import datetime
import hashlib

from .exporter import DataExporter, convert_excel_to_json
from ..core.exceptions import GroupDataError


class DataManager:
    """
    Manages the Excel->JSON data workflow for UNIFAC group data.
    
    This class handles:
    - Automatic detection of data updates
    - Excel to JSON conversion when needed
    - Loading data for compound calculations
    - Data validation and integrity checks
    """
    
    def __init__(self, excel_file: Union[str, Path], json_dir: Union[str, Path]):
        """
        Initialize the data manager.
        
        Args:
            excel_file: Path to the source Excel file
            json_dir: Directory where JSON files are stored/will be created
        """
        self.excel_file = Path(excel_file)
        self.json_dir = Path(json_dir)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
    def ensure_json_data(self, force_update: bool = False) -> Dict[str, Path]:
        """
        Ensure JSON data files are up-to-date.
        
        Args:
            force_update: If True, force re-export even if JSON seems current
            
        Returns:
            Dictionary mapping data types to JSON file paths
            
        Examples:
            >>> manager = DataManager("data/data_unifac.xlsx", "data/json/")
            >>> files = manager.ensure_json_data()
            >>> print(files)
            {'groups': PosixPath('data/json/unifac_groups.json'), ...}
        """
        json_files = {
            'groups': self.json_dir / 'unifac_groups.json',
            'valency': self.json_dir / 'unifac_valency.json',
            'smarts': self.json_dir / 'unifac_smarts.json'
        }
        
        # Check if update is needed
        if force_update or self._needs_update(json_files):
            print(f"Updating JSON data from {self.excel_file}...")
            exported_files = convert_excel_to_json(self.excel_file, self.json_dir)
            
            # Update our file mapping with actually exported files
            for data_type, file_path in exported_files.items():
                if data_type in json_files:
                    json_files[data_type] = file_path
            
            print(f"Exported {len(exported_files)} data files to {self.json_dir}")
        
        return json_files
    
    def load_all_data(self, force_update: bool = False) -> Tuple[Dict, Dict, Dict]:
        """
        Load all UNIFAC data (groups, valency, SMARTS) from JSON files.
        
        Args:
            force_update: If True, force re-export from Excel first
            
        Returns:
            Tuple of (groups_data, valency_data, smarts_data) dictionaries
            
        Examples:
            >>> manager = DataManager("data/data_unifac.xlsx", "data/json/")
            >>> groups, valency, smarts = manager.load_all_data()
            >>> print(f"Loaded {len(groups['groups'])} groups")
        """
        json_files = self.ensure_json_data(force_update)
        
        groups_data = self._load_json_safe(json_files['groups'])
        valency_data = self._load_json_safe(json_files['valency'])
        smarts_data = self._load_json_safe(json_files['smarts'])
        
        return groups_data, valency_data, smarts_data
    
    def setup_compound_data(self, force_update: bool = False) -> None:
        """
        Load all data into the Compound class for immediate use.
        
        Args:
            force_update: If True, force re-export from Excel first
            
        Examples:
            >>> manager = DataManager("data/data_unifac.xlsx", "data/json/")
            >>> manager.setup_compound_data()
            >>> # Now you can create compounds without manual data loading
            >>> compound = Compound.from_groups("Ethanol", {"CH3": 1, "CH2": 1, "OH": 1})
        """
        from ..core import Compound
        
        json_files = self.ensure_json_data(force_update)
        
        # Load data into Compound class
        if json_files['groups'].exists():
            Compound.load_groups(str(json_files['groups']))
        
        if json_files['valency'].exists():
            Compound.load_valency(str(json_files['valency']))
        
        if json_files['smarts'].exists():
            try:
                Compound.load_smarts(str(json_files['smarts']))
            except Exception:
                # SMARTS data might not exist yet
                pass
        
        print("Compound class loaded with current data")
    
    def get_data_info(self) -> Dict:
        """
        Get information about current data files.
        
        Returns:
            Dictionary with data file information and statistics
        """
        json_files = self.ensure_json_data()
        info = {
            'excel_file': str(self.excel_file),
            'excel_modified': datetime.fromtimestamp(self.excel_file.stat().st_mtime).isoformat(),
            'json_dir': str(self.json_dir),
            'files': {}
        }
        
        for data_type, json_file in json_files.items():
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    file_info = {
                        'path': str(json_file),
                        'modified': datetime.fromtimestamp(json_file.stat().st_mtime).isoformat(),
                        'size_bytes': json_file.stat().st_size
                    }
                    
                    # Add metadata if available
                    if 'metadata' in data:
                        file_info.update(data['metadata'])
                    
                    info['files'][data_type] = file_info
                    
                except Exception as e:
                    info['files'][data_type] = {'error': str(e)}
        
        return info
    
    def _needs_update(self, json_files: Dict[str, Path]) -> bool:
        """Check if JSON files need to be updated from Excel."""
        excel_mtime = self.excel_file.stat().st_mtime
        
        for json_file in json_files.values():
            if not json_file.exists():
                return True
            
            json_mtime = json_file.stat().st_mtime
            if excel_mtime > json_mtime:
                return True
        
        return False
    
    def _load_json_safe(self, file_path: Path) -> Dict:
        """Safely load JSON file, return empty dict if file doesn't exist."""
        if not file_path.exists():
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return {}


def create_sample_smarts_file(output_file: Union[str, Path]) -> None:
    """
    Create a sample SMARTS file with common UNIFAC groups.
    
    Args:
        output_file: Path where to save the sample SMARTS JSON file
        
    Note:
        This creates a basic SMARTS mapping based on common UNIFAC groups.
        You should customize this based on your specific needs and the
        patterns from https://github.com/simonmb/fragmentation_algorithm
    """
    sample_smarts = {
        # Alkyl groups
        "CH3": "[CH3;X4]",
        "CH2": "[CH2;X4]", 
        "CH": "[CH1;X4]",
        "C": "[CH0;X4]",
        
        # Alcohols
        "OH": "[OH]",
        "CH3OH": "[CH3][OH]",
        
        # Water
        "H2O": "O",
        
        # Aromatic
        "ACH": "[cH]",
        "AC": "[cH0]",
        "ACCH3": "[c][CH3;X4]",
        
        # Carbonyls
        "CH2CO": "[CH2][C](=O)",
        "CHO": "[CH](=O)",
        "COOH": "C(=O)[OH]",
        
        # Common functional groups
        "NH2": "[NH2]",
        "NH": "[NH]",
        "N": "[N]",
        "COO": "[C](=O)[O]",
        "CH2O": "[CH2][O]",
        "CHO_alcohol": "[CH][OH]",
        "CH2OH": "[CH2][OH]"
    }
    
    output_data = {
        "smarts": sample_smarts,
        "metadata": {
            "description": "Sample SMARTS patterns for UNIFAC groups",
            "source": "Basic patterns - customize as needed",
            "reference": "https://github.com/simonmb/fragmentation_algorithm",
            "created": datetime.now().isoformat(),
            "count": len(sample_smarts)
        }
    }
    
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created sample SMARTS file: {output_file}")


# Convenience function for quick setup
def setup_data_workflow(excel_file: Union[str, Path], json_dir: Union[str, Path]) -> DataManager:
    """
    Convenience function to set up the complete data workflow.
    
    Args:
        excel_file: Path to the source Excel file
        json_dir: Directory for JSON files
        
    Returns:
        Configured DataManager instance
        
    Examples:
        >>> manager = setup_data_workflow("data/data_unifac.xlsx", "data/json/")
        >>> # Data is now ready for use with Compound class
    """
    manager = DataManager(excel_file, json_dir)
    manager.setup_compound_data()
    
    # Create sample SMARTS file if it doesn't exist
    smarts_file = Path(json_dir) / "unifac_smarts.json"
    if not smarts_file.exists():
        create_sample_smarts_file(smarts_file)
    
    return manager