"""Data export utilities for converting Excel data to JSON format."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import hashlib

from ..core.exceptions import GroupDataError


class DataExporter:
    """
    Utility class for exporting Excel data to JSON format.
    
    This class handles the conversion of UNIFAC group data from Excel files
    to JSON format for efficient loading and processing.
    """
    
    def __init__(self, excel_file: Union[str, Path]):
        """
        Initialize the data exporter.
        
        Args:
            excel_file: Path to the Excel file containing UNIFAC data
        """
        self.excel_file = Path(excel_file)
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_file}")
    
    def export_groups(self, sheet_name: str = "Groups", output_file: Optional[Union[str, Path]] = None) -> Dict[str, List[str]]:
        """
        Export allowed groups from Excel to JSON.
        
        Args:
            sheet_name: Name of the sheet containing groups (default: 'Groups')
            output_file: Optional output JSON file path
            
        Returns:
            Dictionary with groups data and metadata
            
        Raises:
            GroupDataError: If export fails
        """
        try:
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            if df.empty:
                raise GroupDataError(f"No data found in sheet '{sheet_name}'")
            
            # Extract groups from first column
            groups = df.iloc[:, 0].dropna().astype(str).tolist()
            if not groups:
                raise GroupDataError("No valid groups found")
            
            # Create output structure
            output_data = {
                "groups": groups,
                "metadata": {
                    "source_file": str(self.excel_file),
                    "sheet_name": sheet_name,
                    "export_timestamp": datetime.now().isoformat(),
                    "count": len(groups),
                    "checksum": self._calculate_checksum(groups)
                }
            }
            
            # Save to file if specified
            if output_file:
                self._save_json(output_data, output_file)
            
            return output_data
            
        except Exception as e:
            raise GroupDataError(f"Failed to export groups from {self.excel_file}: {e}")
    
    def export_valency(self, sheet_name: str = "noc", output_file: Optional[Union[str, Path]] = None) -> Dict[str, Union[Dict[str, int], Dict]]:
        """
        Export group valency data from Excel to JSON.
        
        Args:
            sheet_name: Name of the sheet containing valency data (default: 'noc')
            output_file: Optional output JSON file path
            
        Returns:
            Dictionary with valency data and metadata
            
        Raises:
            GroupDataError: If export fails
        """
        try:
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            if df.empty:
                raise GroupDataError(f"No data found in sheet '{sheet_name}'")
            
            if "grp" not in df.columns or "noc" not in df.columns:
                raise GroupDataError("Expected columns 'grp' and 'noc' not found")
            
            # Convert to dictionary
            valency_dict = df.set_index("grp")["noc"].to_dict()
            if not valency_dict:
                raise GroupDataError("No valid valency data found")
            
            # Create output structure
            output_data = {
                "valency": valency_dict,
                "metadata": {
                    "source_file": str(self.excel_file),
                    "sheet_name": sheet_name,
                    "export_timestamp": datetime.now().isoformat(),
                    "count": len(valency_dict),
                    "checksum": self._calculate_checksum(str(valency_dict))
                }
            }
            
            # Save to file if specified
            if output_file:
                self._save_json(output_data, output_file)
            
            return output_data
            
        except Exception as e:
            raise GroupDataError(f"Failed to export valency from {self.excel_file}: {e}")
    
    def export_sheets(self, sheets: Optional[List[str]] = None, output_dir: Union[str, Path] = None) -> Dict[str, Path]:
        """
        Export specified sheets or all sheets to separate JSON files.
        
        Args:
            sheets: List of sheet names to export. If None, exports all sheets.
            output_dir: Directory to save JSON files
            
        Returns:
            Dictionary mapping sheet names to output file paths
            
        Raises:
            GroupDataError: If export fails
        """
        if output_dir is None:
            output_dir = Path(self.excel_file).parent / "json"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        try:
            excel_file = pd.ExcelFile(self.excel_file)
            available_sheets = excel_file.sheet_names
            
            # If no sheets specified, export all
            if sheets is None:
                sheets_to_export = available_sheets
            else:
                # Validate that requested sheets exist
                missing_sheets = [s for s in sheets if s not in available_sheets]
                if missing_sheets:
                    raise GroupDataError(f"Sheets not found: {missing_sheets}. Available: {available_sheets}")
                sheets_to_export = sheets
            
            for sheet_name in sheets_to_export:
                try:
                    # Handle special cases
                    if sheet_name.lower() == "groups":
                        output_file = output_dir / "unifac_groups.json"
                        self.export_groups(sheet_name, output_file)
                        exported_files["groups"] = output_file
                    elif sheet_name.lower() == "noc":
                        output_file = output_dir / "unifac_valency.json"
                        self.export_valency(sheet_name, output_file)
                        exported_files["valency"] = output_file
                    else:
                        # Generic sheet export
                        output_file = output_dir / f"unifac_{sheet_name.lower()}.json"
                        self._export_generic_sheet(sheet_name, output_file)
                        exported_files[sheet_name.lower()] = output_file
                        
                except Exception as e:
                    print(f"Warning: Failed to export sheet '{sheet_name}': {e}")
                    continue
            
            return exported_files
            
        except Exception as e:
            raise GroupDataError(f"Failed to export sheets: {e}")
    
    def export_all_sheets(self, output_dir: Union[str, Path]) -> Dict[str, Path]:
        """
        Export all sheets to separate JSON files.
        
        Args:
            output_dir: Directory to save JSON files
            
        Returns:
            Dictionary mapping sheet names to output file paths
            
        Raises:
            GroupDataError: If export fails
        """
        return self.export_sheets(sheets=None, output_dir=output_dir)
    
    def _export_generic_sheet(self, sheet_name: str, output_file: Path) -> None:
        """Export a generic sheet to JSON format."""
        df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
        if df.empty:
            return
        
        # Fix column names - rename first column to 'group' if it's unnamed
        if df.columns[0].startswith('Unnamed:'):
            df = df.rename(columns={df.columns[0]: 'group'})
        
        # Convert DataFrame to dictionary
        data = df.to_dict(orient='records')
        
        output_data = {
            "data": data,
            "metadata": {
                "source_file": str(self.excel_file),
                "sheet_name": sheet_name,
                "export_timestamp": datetime.now().isoformat(),
                "rows": len(data),
                "columns": list(df.columns)
            }
        }
        
        self._save_json(output_data, output_file)
    
    def _save_json(self, data: Dict, output_file: Union[str, Path]) -> None:
        """Save data to JSON file with proper formatting."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _calculate_checksum(self, data: Union[str, List]) -> str:
        """Calculate MD5 checksum for data integrity verification."""
        if isinstance(data, list):
            data = str(sorted(data))
        return hashlib.md5(data.encode()).hexdigest()


def convert_excel_to_json(
    excel_file: Union[str, Path],
    output_dir: Union[str, Path],
    sheets: Optional[Union[List[str], Dict[str, str]]] = None
) -> Dict[str, Path]:
    """
    Convenience function to convert Excel UNIFAC data to JSON format.
    
    Args:
        excel_file: Path to Excel file
        output_dir: Directory for JSON output files
        sheets: Optional list of sheet names or mapping of sheet names to export
               If list: exports those specific sheets
               If dict: uses mapping {"sheet_name": "output_name"}
               If None: exports default sheets {"Groups": "groups", "noc": "valency"}
    
    Returns:
        Dictionary mapping data types to output file paths
        
    Examples:
        >>> # Export specific sheets
        >>> files = convert_excel_to_json("data/data_unifac.xlsx", "data/json/", 
        ...                               sheets=["Groups", "noc", "Tm", "Tb"])
        
        >>> # Export all sheets
        >>> files = convert_excel_to_json("data/data_unifac.xlsx", "data/json/", 
        ...                               sheets=None)
        
        >>> # Custom mapping
        >>> files = convert_excel_to_json("data/data_unifac.xlsx", "data/json/",
        ...                               sheets={"Groups": "groups", "noc": "valency"})
    """
    exporter = DataExporter(excel_file)
    
    if sheets is None:
        # Default behavior: export Groups and noc only
        sheets = {"Groups": "groups", "noc": "valency"}
    
    if isinstance(sheets, list):
        # Export specific sheets using new method
        return exporter.export_sheets(sheets=sheets, output_dir=output_dir)
    
    elif isinstance(sheets, dict):
        # Legacy behavior: use mapping
        output_dir = Path(output_dir)
        output_files = {}
        
        for sheet_name, data_type in sheets.items():
            try:
                output_file = output_dir / f"unifac_{data_type}.json"
                
                if data_type == "groups":
                    exporter.export_groups(sheet_name, output_file)
                elif data_type == "valency":
                    exporter.export_valency(sheet_name, output_file)
                else:
                    exporter._export_generic_sheet(sheet_name, output_file)
                
                output_files[data_type] = output_file
                
            except Exception as e:
                print(f"Warning: Failed to export {sheet_name}: {e}")
        
        return output_files
    
    else:
        raise ValueError("sheets must be a list of sheet names or dict mapping")