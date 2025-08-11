#!/usr/bin/env python3
"""
Export specific properties: scalars and Hv
"""

from export_properties import PropertyExporter

def main():
    # Initialize the exporter
    exporter = PropertyExporter()
    
    # Export scalars and Hv
    properties_to_export = ["scalars", "Hv"]
    
    print("Exporting selected properties...")
    results = exporter.export_properties(properties_to_export)
    
    print("\nExport completed!")
    for prop, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {prop}")

if __name__ == "__main__":
    main()