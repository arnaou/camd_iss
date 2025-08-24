#!/usr/bin/env python3
"""
Unified Data Exporter

Exports thermodynamic property data and UNIFAC parameters from Excel to JSON format.
Combines functionality from export_properties.py, export_scalars.py, export_selected.py, 
and export_interactions.py into a single comprehensive module.

Example Usage:
    # Initialize exporter
    exporter = DataExporter("data/data_unifac.xlsx", "data/json")
    
    # Export all properties
    exporter.export_all()
    
    # Export specific properties
    exporter.export_properties(["Rk", "Qk", "Hv"])
    
    # Export scalars
    exporter.export_scalars()
    
    # Export interaction parameters
    exporter.export_interactions()
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Union, Dict


class DataExporter:
    """
    Unified exporter for thermodynamic properties and UNIFAC parameters.
    
    Handles export of:
    - Standard properties (Rk, Qk, Hv, etc.) from Excel sheets
    - Scalar constants from scalars sheet
    - Binary interaction parameters from interactions matrix
    """
    
    def __init__(self, excel_file: str = "data/data_unifac.xlsx", output_dir: str = "data/json"):
        """
        Initialize the data exporter.
        
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
    
    def export_scalars(self) -> bool:
        """
        Export the scalars sheet with const/val structure.
        
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            # Read the scalars sheet
            df = pd.read_excel(self.excel_file, sheet_name='scalars')
            
            # Create data dictionary
            data_dict = {}
            for _, row in df.iterrows():
                const = row['const']
                value = row['val']
                
                # Skip NaN values
                if pd.notna(const) and pd.notna(value):
                    data_dict[str(const)] = value
            
            # Create metadata
            metadata = {
                "source_file": str(self.excel_file),
                "sheet_name": "scalars",
                "export_timestamp": datetime.now().isoformat(),
                "rows": len(data_dict),
                "constant_column": "const",
                "value_column": "val",
                "sample_entries": dict(list(data_dict.items())[:3])  # First 3 entries
            }
            
            # Create JSON structure
            json_data = {
                "metadata": metadata,
                "data": data_dict
            }
            
            # Save to JSON file
            output_file = self.output_dir / 'scalars.json'
            
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)
            
            print(f"✓ Exported {len(data_dict)} scalar constants to {output_file}")
            
            # Display the constants
            print(f"  Exported constants:")
            for const, val in data_dict.items():
                print(f"    {const}: {val}")
            
            return True
            
        except Exception as e:
            print(f"Error exporting scalars: {e}")
            return False
    
    def export_interactions(self, sheet_name: str = "a") -> bool:
        """
        Export UNIFAC binary interaction parameters from Excel to JSON format.
        
        Args:
            sheet_name: Name of the sheet containing interaction parameters (default: "a")
            
        Returns:
            bool: True if export successful, False otherwise
        """
        try:
            print(f"Reading interaction parameters from sheet: {sheet_name}")
            
            # Read the interaction parameters sheet
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name, index_col=0, header=[0, 1])
            
            print(f"  Original DataFrame shape: {df.shape}")
            
            # The DataFrame has multi-level columns due to two header rows
            # We want the first level (group names), skip the second level (main group tags)
            if df.columns.nlevels > 1:
                # Take only the first level of column headers (group names)
                df.columns = df.columns.get_level_values(0)
            
            # Remove any unnamed or NaN columns/rows
            df = df.dropna(how='all', axis=0)  # Remove rows that are all NaN
            df = df.dropna(how='all', axis=1)  # Remove columns that are all NaN
            
            # Convert to dictionary format for JSON export
            interactions_dict = {}
            
            # Convert DataFrame to nested dictionary: {group1: {group2: value}}
            for i, row_group in enumerate(df.index):
                if pd.notna(row_group):  # Skip NaN index values
                    interactions_dict[row_group] = {}
                    for j, col_group in enumerate(df.columns):
                        if pd.notna(col_group):  # Skip NaN column values
                            value = df.iloc[i, j]
                            # Convert numpy types to Python types for JSON serialization
                            if pd.notna(value):
                                interactions_dict[row_group][col_group] = float(value)
                            else:
                                interactions_dict[row_group][col_group] = 0.0
            
            # Create metadata
            metadata = {
                "source_file": str(self.excel_file),
                "sheet_name": sheet_name,
                "export_timestamp": datetime.now().isoformat(),
                "description": "UNIFAC binary interaction parameters (group-group)",
                "format": "nested dictionary {group1: {group2: interaction_value}}",
                "groups_count": len(interactions_dict),
                "notes": [
                    "Symmetric matrix: a_ij should equal a_ji",
                    "Zero values may indicate missing parameters or same main group",
                    "Main group information available in unifac_groups.pkl"
                ],
                "sample_groups": list(interactions_dict.keys())[:5] if interactions_dict else []
            }
            
            # Prepare final JSON structure
            json_data = {
                "metadata": metadata,
                "data": interactions_dict
            }
            
            # Write to JSON file
            output_file = self.output_dir / f'{sheet_name.lower()}.json'
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Exported interaction parameters to {output_file}")
            print(f"  Groups exported: {len(interactions_dict)}")
            print(f"  Total interactions: {sum(len(group_dict) for group_dict in interactions_dict.values())}")
            
            # Show some sample data
            if interactions_dict:
                sample_group = list(interactions_dict.keys())[0]
                sample_interactions = list(interactions_dict[sample_group].items())[:3]
                print(f"  Sample interactions for '{sample_group}':")
                for partner, value in sample_interactions:
                    print(f"    {sample_group} ↔ {partner}: {value}")
            
            return True
            
        except Exception as e:
            print(f"Error exporting interaction parameters: {e}")
            return False
    
    def export_properties(self, sheet_names: Union[str, List[str]], group_column: str = "groups") -> Dict[str, bool]:
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
    
    def export_all(self, group_column: str = "groups", exclude: Optional[List[str]] = None,
                   include_scalars: bool = True, include_interactions: bool = True) -> Dict[str, bool]:
        """
        Export all available sheets to JSON.
        
        Args:
            group_column: Column name containing the group names
            exclude: List of sheet names to exclude from export
            include_scalars: Whether to export scalars sheet
            include_interactions: Whether to export interaction parameters
            
        Returns:
            dict: Results of each export
        """
        if exclude is None:
            exclude = []
        
        # Add scalars and interaction sheet to exclusions for standard property export
        standard_excludes = ["scalars", "a"]  # These are handled separately
        all_excludes = list(set(exclude + standard_excludes))
        
        sheets_to_export = [sheet for sheet in self.available_sheets if sheet not in all_excludes]
        
        results = {}
        
        # Export standard properties
        if sheets_to_export:
            property_results = self.export_properties(sheets_to_export, group_column)
            results.update(property_results)
        
        # Export scalars if requested and available
        if include_scalars and "scalars" in self.available_sheets and "scalars" not in exclude:
            results["scalars"] = self.export_scalars()
        
        # Export interactions if requested and available
        if include_interactions and "a" in self.available_sheets and "a" not in exclude:
            results["interactions"] = self.export_interactions("a")
        
        return results
    
    def export_selected(self, properties: List[str]) -> Dict[str, bool]:
        """
        Export selected properties (convenience method).
        
        Args:
            properties: List of property names to export
            
        Returns:
            dict: Results of each export
        """
        return self.export_properties(properties)
    
    def list_available_sheets(self) -> List[str]:
        """Return list of available sheets in the Excel file."""
        return self.available_sheets.copy()
    
    def validate_excel_structure(self, sheet_name: str) -> Dict[str, any]:
        """
        Validate the structure of a sheet and provide information.
        
        Args:
            sheet_name: Name of the sheet to validate
            
        Returns:
            dict: Validation results and sheet information
        """
        try:
            if sheet_name not in self.available_sheets:
                return {
                    "valid": False,
                    "error": f"Sheet '{sheet_name}' not found",
                    "available_sheets": self.available_sheets
                }
            
            df = pd.read_excel(self.excel_file, sheet_name=sheet_name)
            
            info = {
                "valid": True,
                "sheet_name": sheet_name,
                "shape": df.shape,
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "sample_data": df.head().to_dict() if not df.empty else {}
            }
            
            return info
            
        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "sheet_name": sheet_name
            }


# Convenience functions for backward compatibility
def export_properties(sheet_names: Union[str, List[str]], 
                     excel_file: str = "data/data_unifac.xlsx",
                     output_dir: str = "data/json",
                     group_column: str = "groups") -> Dict[str, bool]:
    """
    Convenience function to export properties.
    
    Args:
        sheet_names: Single sheet name or list of sheet names to export
        excel_file: Path to the Excel file
        output_dir: Output directory for JSON files
        group_column: Column name containing the group names
        
    Returns:
        dict: Results of each export
    """
    exporter = DataExporter(excel_file, output_dir)
    return exporter.export_properties(sheet_names, group_column)


def export_scalars(excel_file: str = "data/data_unifac.xlsx",
                  output_dir: str = "data/json") -> bool:
    """
    Convenience function to export scalars.
    
    Args:
        excel_file: Path to the Excel file
        output_dir: Output directory for JSON files
        
    Returns:
        bool: True if export successful
    """
    exporter = DataExporter(excel_file, output_dir)
    return exporter.export_scalars()


def export_interactions(excel_file: str = "data/data_unifac.xlsx",
                       output_dir: str = "data/json",
                       sheet_name: str = "a") -> bool:
    """
    Convenience function to export interaction parameters.
    
    Args:
        excel_file: Path to the Excel file
        output_dir: Output directory for JSON files
        sheet_name: Name of the sheet containing interaction parameters
        
    Returns:
        bool: True if export successful
    """
    exporter = DataExporter(excel_file, output_dir)
    return exporter.export_interactions(sheet_name)


def main():
    """Interactive main function for data export."""
    try:
        print("=== Unified Data Exporter ===")
        exporter = DataExporter()
        
        print(f"\nExport Options:")
        print("1. Export single property")
        print("2. Export multiple properties")
        print("3. Export scalars")
        print("4. Export interaction parameters")
        print("5. Export all data")
        print("6. Export selected properties (scalars + Hv)")
        print("7. Validate sheet structure")
        print("8. Exit")
        
        while True:
            choice = input("\nSelect an option (1-8): ").strip()
            
            if choice == "1":
                sheet_name = input("Enter sheet name: ").strip()
                exporter.export_property(sheet_name)
                
            elif choice == "2":
                sheets_input = input("Enter sheet names (comma-separated): ").strip()
                sheet_names = [name.strip() for name in sheets_input.split(",")]
                exporter.export_properties(sheet_names)
                
            elif choice == "3":
                exporter.export_scalars()
                
            elif choice == "4":
                exporter.export_interactions()
                
            elif choice == "5":
                exclude_input = input("Enter sheets to exclude (comma-separated, or press Enter for none): ").strip()
                exclude = [name.strip() for name in exclude_input.split(",")] if exclude_input else []
                exporter.export_all(exclude=exclude)
                
            elif choice == "6":
                exporter.export_selected(["scalars", "Hv"])
                
            elif choice == "7":
                sheet_name = input("Enter sheet name to validate: ").strip()
                info = exporter.validate_excel_structure(sheet_name)
                print(f"\nValidation Results:")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                
            elif choice == "8":
                print("Exiting...")
                break
                
            else:
                print("Invalid option. Please select 1-8.")
    
    except KeyboardInterrupt:
        print("\nExport cancelled by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()