#!/usr/bin/env python3
"""
Analyze CAMD screening log and create rejection reasons bar plot.

Reads a CAMD log file and creates a bar plot showing how many compounds
fail due to different property constraints (Tb, Tm, Tc, Hv, Azeotrope, Miscibility, Selectivity).
"""

import re
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from collections import defaultdict


def parse_rejection_reasons(log_path):
    """
    Parse CAMD log file to extract rejection reasons.

    Returns:
        tuple: (pure_rejections, mixture_rejections) dictionaries with counts
    """
    with open(log_path, 'r') as f:
        content = f.read()

    pure_rejections = defaultdict(int)
    mixture_rejections = defaultdict(int)

    # Parse Step 9: Pure Property Rejections
    pure_section = re.search(
        r'STEP 9: Screen Pure Properties.*?Rejection reasons:(.*?)(?=Summary:|STEP 10:)',
        content, re.DOTALL
    )

    if pure_section:
        rejection_text = pure_section.group(1)

        # Count rejections by property
        for line in rejection_text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('solvent_'):
                if 'tb=' in line.lower() and 'outside' in line.lower():
                    pure_rejections['Tb'] += 1
                elif 'tm=' in line.lower() and 'outside' in line.lower():
                    pure_rejections['Tm'] += 1
                elif 'tc=' in line.lower() and 'outside' in line.lower():
                    pure_rejections['Tc'] += 1
                elif 'pc=' in line.lower() and 'outside' in line.lower():
                    pure_rejections['Pc'] += 1
                elif 'hv=' in line.lower() and 'outside' in line.lower():
                    pure_rejections['Hv'] += 1
                elif 'tb (' in line.lower() and '>= tc' in line.lower():
                    pure_rejections['Tb≥Tc'] += 1

    # Parse Step 13: Mixture Property Rejections
    mixture_section = re.search(
        r'STEP 13: Screen Mixture Properties.*?Rejection reasons:(.*?)(?=Summary:|STEP 14:)',
        content, re.DOTALL
    )

    if mixture_section:
        rejection_text = mixture_section.group(1)

        # Count rejections by property
        for line in rejection_text.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('solvent_'):
                if 'azeotrope' in line.lower():
                    mixture_rejections['Azeotrope'] += 1
                elif 'phase split' in line.lower() or 'instability' in line.lower():
                    mixture_rejections['Miscibility'] += 1
                elif 'low selectivity' in line.lower() or 'selectivity' in line.lower():
                    mixture_rejections['Selectivity'] += 1

    return pure_rejections, mixture_rejections


def create_rejection_bar_plot(pure_rejections, mixture_rejections, output_path=None):
    """
    Create a grouped bar plot showing rejection reasons.

    Args:
        pure_rejections: Dictionary with pure property rejection counts
        mixture_rejections: Dictionary with mixture property rejection counts
        output_path: Optional path to save the figure
    """
    # Prepare data
    pure_properties = ['Tb', 'Tm', 'Tc', 'Pc', 'Hv', 'Tb≥Tc']
    mixture_properties = ['Azeotrope', 'Miscibility', 'Selectivity']

    pure_counts = [pure_rejections.get(prop, 0) for prop in pure_properties]
    mixture_counts = [mixture_rejections.get(prop, 0) for prop in mixture_properties]

    # Filter out zero counts
    pure_data = [(prop, count) for prop, count in zip(pure_properties, pure_counts) if count > 0]
    mixture_data = [(prop, count) for prop, count in zip(mixture_properties, mixture_counts) if count > 0]

    # Combine all data
    all_labels = [f"{prop}\n(Pure)" for prop, _ in pure_data] + [f"{prop}\n(Mixture)" for prop, _ in mixture_data]
    all_counts = [count for _, count in pure_data] + [count for _, count in mixture_data]
    all_colors = ['#2ca02c'] * len(pure_data) + ['#d62728'] * len(mixture_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Create bar chart
    x_pos = range(len(all_labels))
    bars = ax.bar(x_pos, all_counts, color=all_colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, count in zip(bars, all_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{int(count)}', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_labels, fontsize=11, fontweight='normal')
    ax.set_ylabel('Number of Rejected Compounds', fontsize=13, fontweight='normal')
    ax.set_title('Screening Rejection Reasons', fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', edgecolor='black', linewidth=1.5, alpha=0.8, label='Pure Properties'),
        Patch(facecolor='#d62728', edgecolor='black', linewidth=1.5, alpha=0.8, label='Mixture Properties')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True,
              fancybox=False, edgecolor='black', framealpha=1)

    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.5)

    plt.tight_layout()

    # Save figure
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none', format='pdf')
        plt.savefig(output_path.replace('.pdf', '.png'), dpi=300,
                   bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"\nFigure saved as:")
        print(f"  - {output_path}")
        print(f"  - {output_path.replace('.pdf', '.png')}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze CAMD screening log and create rejection reasons bar plot'
    )
    parser.add_argument('log_file', help='Path to CAMD log file')
    parser.add_argument('-o', '--output', default='rejection_reasons.pdf',
                       help='Output file path (default: rejection_reasons.pdf)')
    args = parser.parse_args()

    # Check if log file exists
    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return

    print(f"Analyzing log file: {log_path}")

    # Parse rejection reasons
    pure_rejections, mixture_rejections = parse_rejection_reasons(log_path)

    # Print statistics
    print("\n" + "="*60)
    print("PURE PROPERTY REJECTIONS:")
    print("="*60)
    total_pure = 0
    for prop, count in sorted(pure_rejections.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prop:10s}: {count:4d}")
        total_pure += count
    print(f"  {'TOTAL':10s}: {total_pure:4d}")

    print("\n" + "="*60)
    print("MIXTURE PROPERTY REJECTIONS:")
    print("="*60)
    total_mixture = 0
    for prop, count in sorted(mixture_rejections.items(), key=lambda x: x[1], reverse=True):
        print(f"  {prop:15s}: {count:4d}")
        total_mixture += count
    print(f"  {'TOTAL':15s}: {total_mixture:4d}")

    print("\n" + "="*60)
    print(f"TOTAL REJECTIONS: {total_pure + total_mixture}")
    print("="*60)

    # Create plot
    print(f"\nCreating rejection reasons bar plot...")
    create_rejection_bar_plot(pure_rejections, mixture_rejections, args.output)
    print("\nDone!")


if __name__ == '__main__':
    main()
