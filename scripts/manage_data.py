#!/usr/bin/env python3
"""
Command-line script for managing UNIFAC data (Excel -> JSON workflow).

This script helps you:
1. Convert Excel data to JSON format
2. Update JSON files when Excel changes
3. Validate data integrity
4. Get data statistics

Usage:
    python scripts/manage_data.py --help
    python scripts/manage_data.py export --excel data/data_unifac.xlsx --output data/json/
    python scripts/manage_data.py update --excel data/data_unifac.xlsx --json-dir data/json/
    python scripts/manage_data.py info --excel data/data_unifac.xlsx --json-dir data/json/
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sgreen.data import DataManager, convert_excel_to_json, create_sample_smarts_file


def cmd_export(args):
    """Export Excel data to JSON format."""
    try:
        # Determine which sheets to export
        if args.all_sheets:
            from sgreen.data.exporter import DataExporter
            exporter = DataExporter(args.excel)
            files = exporter.export_sheets(sheets=None, output_dir=args.output)
        elif args.sheets:
            # Export specific sheets
            sheets_list = [s.strip() for s in args.sheets.split(',')]
            files = convert_excel_to_json(args.excel, args.output, sheets=sheets_list)
        else:
            # Default export (Groups and noc only)
            files = convert_excel_to_json(args.excel, args.output)
        
        print(f"✅ Successfully exported {len(files)} files:")
        for data_type, file_path in files.items():
            print(f"   {data_type}: {file_path}")
        
        # Create sample SMARTS if requested
        if args.create_smarts:
            smarts_file = Path(args.output) / "unifac_smarts.json"
            if not smarts_file.exists():
                create_sample_smarts_file(smarts_file)
                print(f"   smarts: {smarts_file} (sample)")
                
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1
    
    return 0


def cmd_update(args):
    """Update JSON files if Excel has changed."""
    try:
        manager = DataManager(args.excel, args.json_dir)
        files = manager.ensure_json_data(force_update=args.force)
        
        print(f"✅ JSON data is up-to-date:")
        for data_type, file_path in files.items():
            status = "✓" if file_path.exists() else "✗" 
            print(f"   {status} {data_type}: {file_path}")
            
    except Exception as e:
        print(f"❌ Update failed: {e}")
        return 1
    
    return 0


def cmd_info(args):
    """Show information about data files."""
    try:
        manager = DataManager(args.excel, args.json_dir)
        info = manager.get_data_info()
        
        print(f"📊 Data Information:")
        print(f"   Excel file: {info['excel_file']}")
        print(f"   Modified: {info['excel_modified']}")
        print(f"   JSON directory: {info['json_dir']}")
        print()
        
        for data_type, file_info in info['files'].items():
            if 'error' in file_info:
                print(f"   ❌ {data_type}: {file_info['error']}")
            else:
                print(f"   ✅ {data_type}:")
                print(f"      Path: {file_info['path']}")
                print(f"      Modified: {file_info.get('modified', 'Unknown')}")
                print(f"      Size: {file_info.get('size_bytes', 0)} bytes")
                if 'count' in file_info:
                    print(f"      Records: {file_info['count']}")
                print()
                
    except Exception as e:
        print(f"❌ Info retrieval failed: {e}")
        return 1
    
    return 0


def cmd_setup(args):
    """Set up the complete data workflow."""
    try:
        from sgreen.data import setup_data_workflow
        
        manager = setup_data_workflow(args.excel, args.json_dir)
        print(f"✅ Data workflow set up successfully!")
        print(f"   Excel: {args.excel}")
        print(f"   JSON: {args.json_dir}")
        print()
        print("You can now use:")
        print("   from sgreen.core import Compound")
        print("   compound = Compound.from_groups('Ethanol', {'CH3': 1, 'CH2': 1, 'OH': 1})")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage UNIFAC Excel->JSON data workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export Excel to JSON')
    export_parser.add_argument('--excel', required=True, help='Excel file path')
    export_parser.add_argument('--output', required=True, help='Output directory for JSON files')
    export_parser.add_argument('--sheets', help='Comma-separated list of sheet names to export (e.g., "Groups,noc,Tm,Tb")')
    export_parser.add_argument('--all-sheets', action='store_true', help='Export all sheets in the Excel file')
    export_parser.add_argument('--create-smarts', action='store_true', 
                              help='Create sample SMARTS file if missing')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update JSON files from Excel')
    update_parser.add_argument('--excel', required=True, help='Excel file path')
    update_parser.add_argument('--json-dir', required=True, help='JSON directory')
    update_parser.add_argument('--force', action='store_true', help='Force update even if current')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show data file information')
    info_parser.add_argument('--excel', required=True, help='Excel file path')
    info_parser.add_argument('--json-dir', required=True, help='JSON directory')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Set up complete workflow')
    setup_parser.add_argument('--excel', required=True, help='Excel file path')
    setup_parser.add_argument('--json-dir', required=True, help='JSON directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'export':
        return cmd_export(args)
    elif args.command == 'update':
        return cmd_update(args)
    elif args.command == 'info':
        return cmd_info(args)
    elif args.command == 'setup':
        return cmd_setup(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())