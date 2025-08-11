#!/usr/bin/env python3
"""
Export scalars sheet which has a different structure (const, val)
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

def export_scalars():
    """Export the scalars sheet with const/val structure."""
    try:
        # Read the scalars sheet
        df = pd.read_excel('data/data_unifac.xlsx', sheet_name='scalars')
        
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
            "source_file": "data/data_unifac.xlsx",
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
        output_dir = Path('data/json')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / 'scalars.json'
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"✓ Exported {len(data_dict)} scalar constants to {output_file}")
        
        # Display the constants
        print(f"\nExported constants:")
        for const, val in data_dict.items():
            print(f"  {const}: {val}")
        
        return True
        
    except Exception as e:
        print(f"Error exporting scalars: {e}")
        return False

if __name__ == "__main__":
    export_scalars()