#!/usr/bin/env python3
"""
Check-Azeotrope: Independent test for whether a binary mixture forms an azeotrope.

Standalone diagnostic, separate from the design/screen pipelines. Solves for
the binary azeotropic composition and temperature via UNIFAC (using the same
mixture/pure-property/Antoine setup as iss.py), reports
whether a physically valid azeotrope was found, and plots the xy and Txy
diagrams for the binary pair with the azeotrope point marked.

The azeotrope search is a root-finding problem with a single initial guess
(x = 0.5, T = midpoint of Tb_A/Tb_B) — it can miss an azeotrope that lies far
from that guess, or fail to converge for a genuinely non-azeotropic pair. A
converged result with a boundary composition (x near 0 or 1) means no
azeotrope exists in the usual sense.

Usage:
    python check_azeotrope.py config/IECR/screen_acetone_methanol.yaml -v
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import (
    load_config, validate_mixture, check_mixture_parameters,
    calculate_mixture_properties, override_mixture_properties,
    calculate_mixture_antoine, _fmt_groups,
)
from src.data import load_unifac_data
from src.mixture.unifac import pyo_vle, pyo_azeo

BOUNDARY_TOL = 1e-3
N_CURVE_POINTS = 50


def check_azeotrope(config, verbose=False):
    """
    Run the binary azeotrope check and compute the xy/Txy VLE curve.

    Returns:
        Dict with:
            'found', 'x_A', 'T' (azeotrope point, x_A/T are None if solver failed)
            'solver_status', 'termination_condition'
            'curve_x', 'curve_y', 'curve_T' (VLE curve arrays for plotting)
            'compound_names': [name_A, name_B]
    """
    mixture = validate_mixture(config, verbose)
    check_mixture_parameters(mixture, verbose)
    binary_props = calculate_mixture_properties(mixture, verbose)
    binary_props = override_mixture_properties(binary_props, config, verbose)
    binary_antoine = calculate_mixture_antoine(mixture, binary_props, verbose)

    tb_A = binary_props['A']['tb']
    tb_B = binary_props['B']['tb']

    rk_data, qk_data, interaction_data = load_unifac_data()

    # Azeotrope search (single initial guess at x=0.5)
    T0 = (tb_A + tb_B) / 2
    solution, model = pyo_azeo(
        np.array([[0.5, 0.5]]), np.array([T0]), mixture,
        rk_data, qk_data, interaction_data, binary_antoine, verbose=verbose
    )

    status = str(solution.solver.status)
    term_cond = str(solution.solver.termination_condition)

    result = {
        'solver_status': status,
        'termination_condition': term_cond,
        'found': False,
        'x_A': None,
        'T': None,
        'compound_names': list(mixture.keys()),
    }

    if status == 'ok' and term_cond == 'optimal':
        x_A = model.x[0, 0].value
        T = model.T[0].value
        result['x_A'] = x_A
        result['T'] = T
        result['found'] = BOUNDARY_TOL < x_A < 1 - BOUNDARY_TOL

    # VLE curve across the full composition range (for xy/Txy diagrams)
    x1 = np.linspace(0.0, 1.0, N_CURVE_POINTS)
    x_curve = np.column_stack([x1, 1 - x1])
    T_guess = tb_A * x1 + tb_B * (1 - x1)

    vle_solution, vle_model = pyo_vle(
        x_curve, T_guess, mixture, rk_data, qk_data, interaction_data,
        binary_antoine, verbose=False
    )

    curve_T = np.array([vle_model.T[j].value for j in range(N_CURVE_POINTS)])
    curve_y = np.array([vle_model.y[j, 0].value for j in range(N_CURVE_POINTS)])

    result['curve_x'] = x1
    result['curve_y'] = curve_y
    result['curve_T'] = curve_T

    return result


def plot_xy_txy(result, groups_A, groups_B, save_path=None):
    """Plot side-by-side xy and Txy diagrams with the azeotrope point marked."""
    fig, (ax_xy, ax_txy) = plt.subplots(1, 2, figsize=(11, 5))

    x = result['curve_x']
    y = result['curve_y']
    T = result['curve_T']

    # xy diagram
    ax_xy.plot(x, y, 'b-', label='VLE curve')
    ax_xy.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='y = x')
    ax_xy.set_xlabel('x_A (liquid)')
    ax_xy.set_ylabel('y_A (vapor)')
    ax_xy.set_title('xy Diagram')
    ax_xy.set_xlim(0, 1)
    ax_xy.set_ylim(0, 1)

    # Txy diagram
    ax_txy.plot(x, T, 'b-', label='Bubble T (liquid, x)')
    ax_txy.plot(y, T, 'r-', label='Dew T (vapor, y)')
    ax_txy.set_xlabel('A composition')
    ax_txy.set_ylabel('T [K]')
    ax_txy.set_title('Txy Diagram')
    ax_txy.set_xlim(0, 1)

    if result['found']:
        ax_xy.plot(result['x_A'], result['x_A'], 'g*', markersize=14, label='Azeotrope')
        ax_txy.axvline(result['x_A'], color='g', linestyle=':', linewidth=1)
        ax_txy.plot(result['x_A'], result['T'], 'g*', markersize=14, label='Azeotrope')

    ax_xy.legend()
    ax_txy.legend()
    fig.suptitle(f'A ({groups_A}) / B ({groups_B}) Binary VLE')
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"✓ Saved: {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Check-Azeotrope: test whether a binary mixture forms an azeotrope'
    )
    parser.add_argument('config', help='Path to YAML configuration file (uses the mixture section)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-plot', action='store_true', help='Skip xy/Txy plotting')
    args = parser.parse_args()

    config = load_config(args.config)
    result = check_azeotrope(config, verbose=args.verbose)

    mixture_cfg = config['mixture']
    groups_A = _fmt_groups(mixture_cfg['component_a']['groups'])
    groups_B = _fmt_groups(mixture_cfg['component_b']['groups'])

    print("\n" + "=" * 60)
    print("AZEOTROPE CHECK")
    print("=" * 60)
    print(f"Component A: {groups_A}")
    print(f"Component B: {groups_B}")
    print(f"Solver: {result['solver_status']}/{result['termination_condition']}")

    if result['x_A'] is None:
        print("\nResult: Could not determine — solver did not converge")
    elif result['found']:
        print(f"\nResult: Azeotrope found at x_A={result['x_A']:.4f}, T={result['T']:.2f}K")
    else:
        print(f"\nResult: No azeotrope (solution at composition boundary, "
              f"x_A={result['x_A']:.4f}) — mixture is zeotropic across the full range")
    print("=" * 60)

    if not args.no_plot:
        save_path = None
        if 'output' in config and 'dir' in config['output']:
            output_dir = Path(config['output']['dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / 'xy_txy_diagram.png'
        plot_xy_txy(result, groups_A, groups_B, save_path)


if __name__ == '__main__':
    main()
