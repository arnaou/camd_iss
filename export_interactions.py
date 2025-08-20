#!/usr/bin/env python3
"""
Export UNIFAC binary interaction parameters from Excel to JSON format.

Reads the "a" sheet from data_unifac.xlsx and exports the binary interaction
parameters to a JSON file. The Excel sheet has:
- Top row: group names
- Second row: main group tags (to be skipped)
- Matrix: binary interaction parameters (group-group)
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def export_interactions():
    """Export UNIFAC binary interaction parameters to JSON format."""
    
    # Input and output paths
    input_file = Path("data/data_unifac.xlsx")
    output_file = Path("data/json/interactions.json")
    
    print(f"Reading interaction parameters from: {input_file}")
    
    try:
        # Read the "a" sheet - binary interaction parameters
        df = pd.read_excel(input_file, sheet_name='a', index_col=0, header=[0, 1])
        
        print(f"Original DataFrame shape: {df.shape}")
        print(f"Column levels: {df.columns.nlevels}")
        print(f"First few columns: {df.columns[:5].tolist()}")
        print(f"First few rows: {df.index[:5].tolist()}")
        
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
            "source_file": str(input_file),
            "sheet_name": "a",
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
        
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to JSON file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Successfully exported interaction parameters!")
        print(f"Output file: {output_file}")
        print(f"Groups exported: {len(interactions_dict)}")
        print(f"Total interactions: {sum(len(group_dict) for group_dict in interactions_dict.values())}")
        
        # Show some sample data
        if interactions_dict:
            sample_group = list(interactions_dict.keys())[0]
            sample_interactions = list(interactions_dict[sample_group].items())[:3]
            print(f"\nSample interactions for '{sample_group}':")
            for partner, value in sample_interactions:
                print(f"  {sample_group} ↔ {partner}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error exporting interaction parameters: {e}")
        return False

if __name__ == "__main__":
    print("=== UNIFAC Binary Interaction Parameters Export ===")
    success = export_interactions()
    
    if success:
        print("\n=== Export completed successfully ===")
        print("You can now use the JSON file in CAMD calculations.")
    else:
        print("\n=== Export failed ===")
        print("Please check the error messages above.")