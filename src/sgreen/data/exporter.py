"""Data export utilities for converting Excel data to JSON format."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from ..core.exceptions import GroupDataError


class DataExporter:
    """
    Utility class for exporting Excel data to JSON format with flat key-value structure.
    
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
    
    def export_sheet(self, sheet_name: str, output_file: Optional[Union[str, Path]] = None) -> Path:
        """
        Export a single sheet to JSON format with flat key-value structure.
        
        Args:
            sheet_name: Name of the sheet to export
            output_file: Optional output JSON file path. If None, uses sheet_name.json
            
        Returns:
            Path to the exported JSON file
            
        Raises:
            GroupDataError: If export fails
        """
        if output_file is None:
            output_file = f"{sheet_name}.json"
        
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            if df.empty:
                raise GroupDataError(f"No data found in sheet '{sheet_name}'")
            
            # Fix column names - rename first column to 'group' if it's unnamed
            if df.columns[0].startswith('Unnamed:'):
                df = df.rename(columns={df.columns[0]: 'group'})
            
            # Special handling for SMARTS sheet
            if sheet_name.lower() == 'smarts':
                output_data = self._export_smarts_sheet(df, sheet_name)
            elif len(df.columns) == 2:
                # Simple two-column data (group, value)
                group_col = df.columns[0]
                value_col = df.columns[1]
                
                # Create flat dictionary: {group: value}
                data_dict = df.set_index(group_col)[value_col].to_dict()
                
                output_data = {
                    "metadata": {
                        "source_file": str(self.excel_file),
                        "sheet_name": sheet_name,
                        "export_timestamp": datetime.now().isoformat(),
                        "rows": len(data_dict),
                        "group_column": group_col,
                        "value_column": value_col,
                        "sample_entries": dict(list(data_dict.items())[:3]) if data_dict else {}
                    },
                    "data": data_dict
                }
            else:
                # Multi-column data - convert first column to keys, rest to nested values
                group_col = df.columns[0]
                value_cols = df.columns[1:]
                
                data_dict = {}
                for _, row in df.iterrows():
                    group_name = row[group_col]
                    if len(value_cols) == 1:
                        data_dict[group_name] = row[value_cols[0]]
                    else:
                        data_dict[group_name] = {col: row[col] for col in value_cols}
                
                output_data = {
                    "metadata": {
                        "source_file": str(self.excel_file),
                        "sheet_name": sheet_name,
                        "export_timestamp": datetime.now().isoformat(),
                        "rows": len(data_dict),
                        "group_column": group_col,
                        "value_columns": list(value_cols),
                        "sample_entries": dict(list(data_dict.items())[:3]) if data_dict else {}
                    },
                    "data": data_dict
                }
            
            self._save_json(output_data, output_file)
            return output_file
            
        except Exception as e:
            raise GroupDataError(f"Failed to export sheet '{sheet_name}' from {self.excel_file}: {e}")
    
    def _export_smarts_sheet(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Special handling for SMARTS sheet which may have multiple patterns per group.
        
        Args:
            df: DataFrame containing SMARTS data
            sheet_name: Name of the sheet
            
        Returns:
            Dictionary with processed SMARTS data
        """
        group_col = df.columns[0]  # 'groups' column
        data_dict = {}
        
        # Count groups with multiple patterns
        single_pattern_count = 0
        multiple_pattern_count = 0
        
        for _, row in df.iterrows():
            group_name = row[group_col]
            
            # Collect all non-null SMARTS patterns for this group
            patterns = []
            for col in df.columns[1:]:  # Skip 'groups' column
                if pd.notna(row[col]):
                    patterns.append(str(row[col]).strip())
            
            if len(patterns) == 1:
                # Single pattern - store as string
                data_dict[group_name] = patterns[0]
                single_pattern_count += 1
            elif len(patterns) > 1:
                # Multiple patterns - store as list
                data_dict[group_name] = patterns
                multiple_pattern_count += 1
            else:
                # No patterns found - this shouldn't happen but handle it
                data_dict[group_name] = None
        
        return {
            "metadata": {
                "source_file": str(self.excel_file),
                "sheet_name": sheet_name,
                "export_timestamp": datetime.now().isoformat(),
                "rows": len(data_dict),
                "group_column": group_col,
                "sample_entries": dict(list(data_dict.items())[:3]) if data_dict else {}
            },
            "data": data_dict
        }
    
    def export_sheets(self, sheets: Optional[List[str]] = None, output_dir: Union[str, Path] = None) -> Dict[str, Path]:
        """
        Export multiple sheets to JSON files.
        
        Args:
            sheets: List of sheet names to export. If None, exports all sheets except 'a' and 'a-reduced'.
            output_dir: Directory to save JSON files. If None, uses current directory.
            
        Returns:
            Dictionary mapping sheet names to output file paths
            
        Raises:
            GroupDataError: If export fails
        """
        if output_dir is None:
            output_dir = Path(".")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        try:
            excel_file = pd.ExcelFile(self.excel_file)
            available_sheets = excel_file.sheet_names
            
            # If no sheets specified, export all except problematic ones
            if sheets is None:
                sheets_to_export = [s for s in available_sheets if s not in ['a', 'a-reduced']]
            else:
                # Validate that requested sheets exist
                missing_sheets = [s for s in sheets if s not in available_sheets]
                if missing_sheets:
                    raise GroupDataError(f"Sheets not found: {missing_sheets}. Available: {available_sheets}")
                sheets_to_export = sheets
            
            for sheet_name in sheets_to_export:
                try:
                    sheet_name_str = str(sheet_name)
                    output_file = output_dir / f"{sheet_name_str}.json"
                    exported_file = self.export_sheet(sheet_name_str, output_file)
                    exported_files[sheet_name_str] = exported_file
                        
                except Exception as e:
                    print(f"Warning: Failed to export sheet '{sheet_name}': {e}")
                    continue
            
            return exported_files
            
        except Exception as e:
            raise GroupDataError(f"Failed to export sheets: {e}")
    
    def _save_json(self, data: Dict, output_file: Union[str, Path]) -> None:
        """Save data to JSON file with proper formatting."""
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def convert_excel_to_json(
    excel_file: Union[str, Path],
    output_dir: Union[str, Path],
    sheets: Optional[List[str]] = None
) -> Dict[str, Path]:
    """
    Convenience function to convert Excel UNIFAC data to JSON format.
    
    Args:
        excel_file: Path to Excel file
        output_dir: Directory for JSON output files
        sheets: Optional list of sheet names to export. If None, exports all sheets except 'a' and 'a-reduced'.
    
    Returns:
        Dictionary mapping sheet names to output file paths
        
    Examples:
        >>> # Export specific sheets
        >>> files = convert_excel_to_json("data/data_unifac.xlsx", "data/json/", 
        ...                               sheets=["Tm", "Tb", "noc"])
        
        >>> # Export all sheets (except problematic ones)
        >>> files = convert_excel_to_json("data/data_unifac.xlsx", "data/json/")
    """
    exporter = DataExporter(excel_file)
    return exporter.export_sheets(sheets=sheets, output_dir=output_dir)