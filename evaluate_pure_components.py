#!/usr/bin/env python3
"""
Pure Component Property Evaluation Script

This script evaluates pure component properties for a predefined list of compounds
using the Constantinou-Gani method. It provides detailed property calculations
and comparison capabilities for selected components.

Usage:
    python evaluate_pure_components.py

The script will evaluate all compounds defined in the SELECTED_COMPONENTS list
and display their calculated properties in a formatted table.
"""

import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from property_calculator import PropertyCalculator


# Define your selected components here as group composition dictionaries
SELECTED_COMPONENTS = [

    {'name': 'HEP', 'composition': {'CH3': 2, 'CH2':5}},
    {'name': 'DEA', 'composition': {'CH3': 2, 'CH2':1, 'CH2NH':1}},
    {'name': 'Methanol', 'composition': {'CH3OH': 1}},
    {'name': 'Acetone', 'composition': {'CH3': 1, 'CH3CO': 1}},
    {'name': 'Water', 'composition': {'H2O':1}},
    {'name': 'Ethylene Glycol', 'composition': {'DOH': 1}},
    
    
    # Add your own compounds here following the same format:
    # {'name': 'Compound Name', 'composition': {'GROUP1': count1, 'GROUP2': count2}},
]


class PureComponentEvaluator:
    """
    Evaluator for pure component properties of selected compounds.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            data_path: Path to property data directory (optional)
        """
        self.calculator = PropertyCalculator(data_path=data_path)
        self.results = []
        
    def evaluate_component(self, name: str, composition: Dict[str, int]) -> Dict:
        """
        Evaluate properties for a single component.
        
        Args:
            name: Component name
            composition: Group composition dictionary
            
        Returns:
            Dictionary with component info and calculated properties
        """
        print(f"Evaluating {name}: {composition}")
        
        # Calculate all properties
        properties = self.calculator.calculate_all_properties(composition)
        
        # Create result entry
        result = {
            'name': name,
            'composition': composition,
            'formula': self._get_molecular_formula(composition),
            **properties
        }
        
        return result
    
    def evaluate_all_components(self, components: List[Dict]) -> List[Dict]:
        """
        Evaluate properties for all selected components.
        
        Args:
            components: List of component dictionaries with 'name' and 'composition'
            
        Returns:
            List of results dictionaries
        """
        print(f"=== Evaluating {len(components)} Pure Components ===\n")
        
        self.results = []
        
        for component in components:
            try:
                result = self.evaluate_component(
                    component['name'], 
                    component['composition']
                )
                self.results.append(result)
                print("✓ Success\n")
                
            except Exception as e:
                print(f"✗ Error: {e}\n")
                # Add failed result
                self.results.append({
                    'name': component['name'],
                    'composition': component['composition'],
                    'formula': self._get_molecular_formula(component['composition']),
                    'tm': None, 'tb': None, 'tc': None, 'pc': None, 'ac': None, 'hv': None,
                    'error': str(e)
                })
        
        return self.results
    
    def _get_molecular_formula(self, composition: Dict[str, int]) -> str:
        """
        Generate a simple molecular formula representation.
        
        Args:
            composition: Group composition dictionary
            
        Returns:
            String representation of molecular formula
        """
        formula_parts = []
        for group, count in composition.items():
            if count == 1:
                formula_parts.append(group)
            else:
                formula_parts.append(f"{group}[{count}]")
        
        return " + ".join(formula_parts)
    
    def display_results(self, sort_by: str = None, ascending: bool = True):
        """
        Display results in a formatted table.
        
        Args:
            sort_by: Property to sort by ('name', 'tb', 'tm', etc.)
            ascending: Sort in ascending order if True
        """
        if not self.results:
            print("No results to display. Run evaluate_all_components() first.")
            return
        
        # Create DataFrame for better formatting
        df_data = []
        for result in self.results:
            row = {
                'Name': result['name'],
                'Tm (K)': f"{result['tm']:.2f}" if result['tm'] is not None else "N/A",
                'Tb (K)': f"{result['tb']:.2f}" if result['tb'] is not None else "N/A",
                'Tc (K)': f"{result['tc']:.2f}" if result['tc'] is not None else "N/A",
                'Pc (bar)': f"{result['pc']:.2f}" if result['pc'] is not None else "N/A",
                'Ac (-)': f"{result['ac']:.2f}" if result['ac'] is not None else "N/A",
                'Hv (kJ/mol)': f"{result['hv']:.2f}" if result['hv'] is not None else "N/A",
                'Formula': result['formula'],
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Sort only if explicitly requested
        if sort_by is not None:
            if sort_by in ['tm', 'tb', 'tc', 'pc', 'ac', 'hv']:
                # Convert to numeric for sorting, handling N/A values
                numeric_col = f"{sort_by.title()} (K)" if sort_by in ['tm', 'tb', 'tc'] else \
                             f"{sort_by.title()} (bar)" if sort_by == 'pc' else \
                             f"{sort_by.title()} (-)" if sort_by == 'ac' else \
                             f"{sort_by.title()} (kJ/mol)"
                
                # Create a temporary column for sorting
                df['_sort_temp'] = df[numeric_col].replace('N/A', None)
                df['_sort_temp'] = pd.to_numeric(df['_sort_temp'], errors='coerce')
                df = df.sort_values('_sort_temp', ascending=ascending, na_position='last')
                df = df.drop('_sort_temp', axis=1)
            elif sort_by == 'name':
                df = df.sort_values('Name', ascending=ascending)
        
        print("=== Pure Component Property Evaluation Results ===\n")
        print(df.to_string(index=False))
        print(f"\nEvaluated {len(self.results)} components")
        
        # Show summary statistics
        self._display_summary()
    
    def _display_summary(self):
        """Display summary statistics for the evaluated components."""
        if not self.results:
            return
        
        print("\n=== Summary Statistics ===")
        
        # Count successful calculations
        properties = ['tm', 'tb', 'tc', 'pc', 'ac', 'hv']
        prop_names = ['Tm', 'Tb', 'Tc', 'Pc', 'Ac', 'Hv']
        
        for prop, name in zip(properties, prop_names):
            successful = sum(1 for r in self.results if r[prop] is not None)
            total = len(self.results)
            print(f"{name}: {successful}/{total} successful ({successful/total*100:.1f}%)")
    
    def save_results(self, filename: str = "pure_component_evaluation.xlsx"):
        """
        Save results to an Excel file.
        
        Args:
            filename: Output filename
        """
        if not self.results:
            print("No results to save. Run evaluate_all_components() first.")
            return
        
        # Prepare data for Excel
        excel_data = []
        for result in self.results:
            row = {
                'Name': result['name'],
                'Molecular Formula': result['formula'],
                'Group Composition': str(result['composition']),
                'Tm (K)': result['tm'],
                'Tb (K)': result['tb'],
                'Tc (K)': result['tc'],
                'Pc (Pa)': result['pc'],
                'Ac (-)': result['ac'],
                'Hv (J/mol)': result['hv'],
            }
            
            if 'error' in result:
                row['Error'] = result['error']
            
            excel_data.append(row)
        
        df = pd.DataFrame(excel_data)
        df.to_excel(filename, index=False)
        print(f"Results saved to {filename}")
    
    def filter_by_property(self, property_name: str, min_value: float = None, 
                          max_value: float = None) -> List[Dict]:
        """
        Filter components by property constraints.
        
        Args:
            property_name: Property to filter by ('tm', 'tb', 'tc', 'pc', 'ac', 'hv')
            min_value: Minimum value (optional)
            max_value: Maximum value (optional)
            
        Returns:
            List of components that meet the criteria
        """
        if not self.results:
            print("No results available. Run evaluate_all_components() first.")
            return []
        
        filtered = []
        for result in self.results:
            value = result.get(property_name)
            if value is None:
                continue
                
            # Check constraints
            if min_value is not None and value < min_value:
                continue
            if max_value is not None and value > max_value:
                continue
                
            filtered.append(result)
        
        return filtered
    
    def compare_components(self, component_names: List[str]):
        """
        Compare specific components side by side.
        
        Args:
            component_names: List of component names to compare
        """
        if not self.results:
            print("No results available. Run evaluate_all_components() first.")
            return
        
        # Find matching components
        matching = [r for r in self.results if r['name'] in component_names]
        
        if not matching:
            print("No matching components found.")
            return
        
        print(f"=== Comparison of {len(matching)} Components ===\n")
        
        # Display comparison table
        comparison_data = []
        for result in matching:
            row = {
                'Component': result['name'],
                'Tm (K)': f"{result['tm']:.2f}" if result['tm'] is not None else "N/A",
                'Tb (K)': f"{result['tb']:.2f}" if result['tb'] is not None else "N/A",
                'Tc (K)': f"{result['tc']:.2f}" if result['tc'] is not None else "N/A",
                'Pc (bar)': f"{result['pc']:.2f}" if result['pc'] is not None else "N/A",
                'Ac (-)': f"{result['ac']:.2f}" if result['ac'] is not None else "N/A",
                'Hv (kJ/mol)': f"{result['hv']:.2f}" if result['hv'] is not None else "N/A",
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))


def main():
    """Main function to run the pure component evaluation."""
    print("Pure Component Property Evaluation Script")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = PureComponentEvaluator()
    
    # Display available properties
    available_props = evaluator.calculator.get_available_properties()
    print(f"Available properties: {', '.join(available_props)}")
    print(f"Property calculation method: {evaluator.calculator.pure_method}\n")
    
    # Evaluate all selected components
    results = evaluator.evaluate_all_components(SELECTED_COMPONENTS)
    
    # Display results (preserve original order)
    evaluator.display_results()
    
    # Save results to Excel
    output_file = "pure_component_evaluation_results.xlsx"
    evaluator.save_results(output_file)
    
    # Example filtering and comparison
    print("\n=== Example Analysis ===")
    
    # Filter components with boiling point between 350-400 K
    print("\nComponents with Tb between 350-400 K:")
    tb_filtered = evaluator.filter_by_property('tb', 350.0, 400.0)
    for comp in tb_filtered:
        print(f"  {comp['name']}: {comp['tb']:.2f} K")
    
    # Compare alcohols
    print("\nComparing alcohols:")
    evaluator.compare_components(['Methanol', 'Ethanol', 'Propanol', 'Butanol'])
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Results saved to: {output_file}")
    print(f"Total components evaluated: {len(results)}")


if __name__ == "__main__":
    main()