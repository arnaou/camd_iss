#!/usr/bin/env python3
"""
Script to export thermodynamic property data from Excel to JSON format.
Supports exporting single or multiple properties.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union


class PropertyExporter:
    """Export thermodynamic properties from Excel to JSON format."""
    
    def __init__(self, excel_file: str = "data/data_unifac.xlsx", output_dir: str = "data/json"):
        """
        Initialize the property exporter.
        
        Args:
            excel_file: Path to the Excel file containing property data
            output_dir: Directory to save JSON files
        """
        self.excel_file = Path(excel_file)
        self.output_dir = Path(output_dir)
        
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_file}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Excel file to get available sheets
        self.excel_data = pd.ExcelFile(self.excel_file)
        self.available_sheets = self.excel_data.sheet_names
        
        print(f"Available sheets in {self.excel_file.name}:")
        for i, sheet in enumerate(self.available_sheets, 1):
            print(f"  {i:2d}. {sheet}")
    
    def export_property(self, sheet_name: str, group_column: str = "groups") -> bool:
        """
        Export a single property sheet to JSON.
        
        Args:
            sheet_name: Name of the Excel sheet to export
            group_column: Column name containing the group names
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            if sheet_name not in self.available_sheets:
                print(f"Error: Sheet '{sheet_name}' not found in Excel file")
                return False
            
            # Read the sheet
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            
            # Check if group column exists
            if group_column not in df.columns:
                print(f"Warning: Column '{group_column}' not found in sheet '{sheet_name}'")
                print(f"Available columns: {list(df.columns)}")
                # Try to find a suitable column
                possible_group_cols = [col for col in df.columns if 'group' in col.lower()]
                if possible_group_cols:
                    group_column = possible_group_cols[0]
                    print(f"Using column '{group_column}' instead")
                else:
                    print(f"Error: No suitable group column found")
                    return False
            
            # Determine the property column (assume it's the second column if not specified)
            property_columns = [col for col in df.columns if col != group_column]
            
            if len(property_columns) == 1:
                property_column = property_columns[0]
            else:
                print(f"Multiple property columns found: {property_columns}")
                property_column = property_columns[0]  # Take the first one
                print(f"Using column '{property_column}'")
            
            # Create the data dictionary
            data_dict = {}
            for _, row in df.iterrows():
                group = row[group_column]
                value = row[property_column]
                
                # Skip NaN values
                if pd.notna(group) and pd.notna(value):
                    data_dict[str(group)] = value
            
            # Create metadata
            metadata = {
                "source_file": str(self.excel_file),
                "sheet_name": sheet_name,
                "export_timestamp": datetime.now().isoformat(),
                "rows": len(data_dict),
                "group_column": group_column,
                "property_column": property_column,
                "sample_entries": dict(list(data_dict.items())[:3])  # First 3 entries
            }
            
            # Create JSON structure
            json_data = {
                "metadata": metadata,
                "data": data_dict
            }
            
            # Save to JSON file
            output_file = self.output_dir / f"{sheet_name.lower()}.json"
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            print(f"✓ Exported {len(data_dict)} entries from sheet '{sheet_name}' to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error exporting sheet '{sheet_name}': {e}")
            return False
    
    def export_properties(self, sheet_names: Union[str, List[str]], group_column: str = "groups") -> dict:
        """
        Export multiple properties to JSON.
        
        Args:
            sheet_names: Single sheet name or list of sheet names to export
            group_column: Column name containing the group names
            
        Returns:
            dict: Results of each export (sheet_name -> success/failure)
        """
        if isinstance(sheet_names, str):
            sheet_names = [sheet_names]
        
        results = {}
        
        print(f"\nExporting {len(sheet_names)} properties...")
        for sheet_name in sheet_names:
            success = self.export_property(sheet_name, group_column)
            results[sheet_name] = success
        
        # Summary
        successful = sum(results.values())
        print(f"\nExport Summary:")
        print(f"  Successful: {successful}/{len(sheet_names)}")
        print(f"  Failed: {len(sheet_names) - successful}/{len(sheet_names)}")
        
        return results
    
    def export_all(self, group_column: str = "groups", exclude: Optional[List[str]] = None) -> dict:
        """
        Export all available sheets to JSON.
        
        Args:
            group_column: Column name containing the group names
            exclude: List of sheet names to exclude from export
            
        Returns:
            dict: Results of each export
        """
        if exclude is None:
            exclude = []
        
        sheets_to_export = [sheet for sheet in self.available_sheets if sheet not in exclude]
        
        return self.export_properties(sheets_to_export, group_column)


def main():
    """Interactive main function for property export."""
    try:
        exporter = PropertyExporter()
        
        print(f"\nProperty Export Options:")
        print("1. Export single property")
        print("2. Export multiple properties")
        print("3. Export all properties")
        print("4. Exit")
        
        while True:
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == "1":
                sheet_name = input("Enter sheet name: ").strip()
                exporter.export_property(sheet_name)
                
            elif choice == "2":
                sheets_input = input("Enter sheet names (comma-separated): ").strip()
                sheet_names = [name.strip() for name in sheets_input.split(",")]
                exporter.export_properties(sheet_names)
                
            elif choice == "3":
                exclude_input = input("Enter sheets to exclude (comma-separated, or press Enter for none): ").strip()
                exclude = [name.strip() for name in exclude_input.split(",")] if exclude_input else []
                exporter.export_all(exclude=exclude)
                
            elif choice == "4":
                print("Exiting...")
                break
                
            else:
                print("Invalid option. Please select 1-4.")
    
    except KeyboardInterrupt:
        print("\nExport cancelled by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()