#!/usr/bin/env python3
"""
Data Exporter

Exports thermodynamic property data and UNIFAC parameters from Excel to JSON.

Example Usage:
    # Export all sheets
    exporter = DataExporter("data/data_unifac.xlsx", "data/json")
    results = exporter.export()
    
    # Export specific sheets
    results = exporter.export(["Rk", "Qk", "scalars"])
    
    # Export single sheet
    result = exporter.export("Hv")
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Union, Dict, Optional


class DataExporter:
    """exporter for UNIFAC data with auto-detection."""
    
    def __init__(self, excel_file: str = "data/data_unifac.xlsx", output_dir: str = "data/json"):
        self.excel_file = Path(excel_file)
        self.output_dir = Path(output_dir)
        
        if not self.excel_file.exists():
            raise FileNotFoundError(f"Excel file not found: {self.excel_file}")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.excel_data = pd.ExcelFile(self.excel_file)
    
    def export(self, sheets: Optional[Union[str, List[str]]] = None) -> Dict[str, bool]:
        """Export specified sheets or all sheets if none specified."""
        if sheets is None:
            sheets = self.excel_data.sheet_names
        elif isinstance(sheets, str):
            sheets = [sheets]
        
        results = {}
        for sheet in sheets:
            if sheet not in self.excel_data.sheet_names:
                results[sheet] = False
                continue
                
            try:
                if sheet == "scalars":
                    results[sheet] = self._export_scalars()
                elif sheet == "a":
                    results[sheet] = self._export_interactions()
                else:
                    results[sheet] = self._export_standard(sheet)
            except Exception:
                results[sheet] = False
        
        successful = sum(results.values())
        if len(results) > 1:
            print(f"Export Summary: {successful}/{len(results)} successful")
        return results
    
    def _export_standard(self, sheet: str) -> bool:
        """Export standard property sheets."""
        df = pd.read_excel(self.excel_file, sheet_name=sheet)
        
        # Find group column
        group_col = None
        for col in df.columns:
            if 'group' in str(col).lower():
                group_col = col
                break
        if not group_col:
            group_col = df.columns[0]
        
        # Get property column
        prop_cols = [col for col in df.columns if col != group_col]
        if not prop_cols:
            return False
        prop_col = prop_cols[0]
        
        # Create data dictionary
        data = {}
        for _, row in df.iterrows():
            group, value = row[group_col], row[prop_col]
            if pd.notna(group) and pd.notna(value):
                data[str(group)] = value
        
        # Save JSON
        self._save_json(data, sheet)
        print(f"✓ Exported {len(data)} entries from '{sheet}'")
        return True
    
    def _export_scalars(self) -> bool:
        """Export scalars sheet."""
        df = pd.read_excel(self.excel_file, sheet_name='scalars')
        
        data = {}
        for _, row in df.iterrows():
            const, value = row['const'], row['val']
            if pd.notna(const) and pd.notna(value):
                data[str(const)] = value
        
        self._save_json(data, 'scalars')
        print(f"✓ Exported {len(data)} scalar constants")
        return True
    
    def _export_interactions(self) -> bool:
        """Export interaction parameters matrix."""
        df = pd.read_excel(self.excel_file, sheet_name='a', index_col=0, header=[0, 1])
        
        # Simplify headers and clean data
        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # Convert to nested dictionary
        data = {}
        for i, row_group in enumerate(df.index):
            if pd.notna(row_group):
                data[row_group] = {}
                for j, col_group in enumerate(df.columns):
                    if pd.notna(col_group):
                        value = df.iloc[i, j]
                        try:
                            data[row_group][col_group] = float(value) if pd.notna(value) else 0.0
                        except (ValueError, TypeError):
                            data[row_group][col_group] = 0.0
        
        self._save_json(data, 'a')
        total_interactions = sum(len(group_dict) for group_dict in data.values())
        print(f"✓ Exported interaction parameters: {len(data)} groups, {total_interactions} interactions")
        return True
    
    def _save_json(self, data: dict, sheet_name: str):
        """Save data to JSON with minimal metadata."""
        json_data = {
            "metadata": {
                "source_file": str(self.excel_file),
                "sheet_name": sheet_name,
                "export_timestamp": datetime.now().isoformat(),
                "rows": len(data)
            },
            "data": data
        }
        
        output_file = self.output_dir / f"{sheet_name.lower()}.json"
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)