#!/usr/bin/env python3
"""
Screen-ISS: Screen a predefined set of entrainers for ISS performance

Config-driven script for screening and evaluating a limited set of entrainers.
No molecule generation - all entrainers are predefined in the config file.

Workflow Steps:
1. Check defined mixture for validity
2. Check mixture for pure component and binary parameters
3. Calculate pure component properties (mixture)
4. Override if properties are provided by user
5. Calculate Antoine parameters (mixture)
6. Load and validate predefined entrainer candidates
7. Check UNIFAC parameters for entrainers
8. Calculate pure component properties (entrainers)
9. Screen based on pure property constraints
10. Calculate Antoine parameters (entrainers)
11. Perform miscibility screening over temperature range
12. Perform VLE (only on miscibility-compatible entrainers)
13. Screen based on mixture constraints
14. Perform ISS calculation
"""

import sys
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.molecules import StructuralValidator
from src.pure import PropertyCalculator, AntoineCalculator
from src.data import load_unifac_data, check_unifac_parameters
from src.mixture.unifac import pyo_vle
from src.mixture.unifac.pyomo_iss import pyo_iss
from src.mixture.unifac import unifac


class TeeLogger:
    """Logger that writes to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_dir(config):
    """Create output directory."""
    output_dir = Path(config['output']['dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_step(step_num, title, verbose):
    """Print step header."""
    if verbose:
        print("\n" + "="*60)
        print(f"STEP {step_num}: {title}")
        print("="*60)


# === STEP 1: Check defined mixture for validity ===
def validate_mixture(config, verbose=False):
    """Check defined mixture for structural validity."""
    print_step(1, "Validate Binary Mixture", verbose)

    mixture = {
        'A': config['mixture']['component_a']['groups'],
        'B': config['mixture']['component_b']['groups']
    }

    validator = StructuralValidator()

    # Validate A
    is_valid_A = validator.is_valid(mixture['A'])
    # Validate B
    is_valid_B = validator.is_valid(mixture['B'])

    if verbose:
        groups_A = ", ".join([f"{k}:{v}" for k, v in mixture['A'].items()])
        groups_B = ", ".join([f"{k}:{v}" for k, v in mixture['B'].items()])
        print(f"Component A: {groups_A}")
        print(f"  {'✓ Valid' if is_valid_A else '✗ Invalid'}")
        print(f"Component B: {groups_B}")
        print(f"  {'✓ Valid' if is_valid_B else '✗ Invalid'}")
        print(f"\nSummary: {'Both components valid' if is_valid_A and is_valid_B else 'Validation failed'}")

    if not (is_valid_A and is_valid_B):
        raise ValueError("Binary mixture validation failed")

    return mixture


# === STEP 2: Check mixture for pure component and binary parameters ===
def check_mixture_parameters(mixture, verbose=False):
    """Check UNIFAC parameters for binary mixture."""
    print_step(2, "Check Binary Mixture Parameters", verbose)

    result = check_unifac_parameters(mixture=mixture)

    if verbose:
        if result['compatible']:
            print("✓ All UNIFAC parameters available")
        else:
            print("✗ Missing UNIFAC parameters")
        print(f"\nSummary: {'Compatible' if result['compatible'] else 'Missing parameters'}")

    return result['compatible']


# === STEP 3: Calculate pure component properties ===
def calculate_mixture_pure_properties(mixture, verbose=False):
    """Calculate pure component properties for binary mixture."""
    print_step(3, "Calculate Pure Component Properties (Mixture)", verbose)

    prop_list = ['tc', 'pc', 'tb', 'tm', 'ac', 'hv']
    pure_calc = PropertyCalculator()
    pure_props = pure_calc.calculate(mixture, prop_list)

    if verbose:
        for comp, props in pure_props.items():
            groups_str = ", ".join([f"{k}:{v}" for k, v in mixture[comp].items()])
            print(f"\n{comp} ({groups_str}):")
            print(f"  Tc: {props['tc']:.2f} K")
            print(f"  Pc: {props['pc']:.2f} bar")
            print(f"  Tb: {props['tb']:.2f} K")
            print(f"  Tm: {props['tm']:.2f} K")
            print(f"  Ac: {props['ac']:.4f}")
            print(f"  Hv: {props['hv']:.2f} kJ/mol")
        print(f"\nSummary: Calculated {len(prop_list)} properties for {len(pure_props)} components")

    return pure_props


# === STEP 4: Override if properties are provided by user ===
def override_mixture_properties(pure_props, config, verbose=False):
    """Apply user-provided property overrides."""
    print_step(4, "Override Mixture Properties (if provided)", verbose)

    overrides_applied = 0

    if 'pure_properties' in config:
        for comp in ['component_a', 'component_b']:
            if comp in config['pure_properties']:
                comp_key = 'A' if comp == 'component_a' else 'B'
                overrides = config['pure_properties'][comp]
                for prop, value in overrides.items():
                    if value is not None:
                        pure_props[comp_key][prop] = value
                        overrides_applied += 1
                        if verbose:
                            print(f"Override {comp_key}.{prop} = {value}")

    if verbose:
        if overrides_applied > 0:
            print(f"\nSummary: {overrides_applied} property overrides applied")
        else:
            print("\nSummary: No overrides (using calculated values)")

    return pure_props


# === STEP 5: Calculate Antoine parameters ===
def calculate_mixture_antoine(mixture, pure_props, verbose=False):
    """Calculate Antoine parameters for mixture components."""
    print_step(5, "Calculate Antoine Parameters (Mixture)", verbose)

    antoine_calc = AntoineCalculator()
    antoine_params = {}
    failed = []

    for compound, props in pure_props.items():
        compound_dict = {compound: mixture[compound]}
        results = antoine_calc.calculate_antoine_parameters(
            compound_dict, critical_properties=props, method='enthalpy_vap'
        )

        if results['termination_condition'] == 'optimal':
            antoine_params[compound] = {
                'A': results['A'],
                'B': results['B'],
                'C': results['C']
            }
            if verbose:
                print(f"✓ {compound}: A={results['A']:.3f}, B={results['B']:.1f}, C={results['C']:.1f}")
        else:
            failed.append(compound)
            if verbose:
                print(f"✗ {compound}: Failed")

    if verbose:
        print(f"\nSummary: {len(antoine_params)}/{len(pure_props)} successful")

    if len(antoine_params) != len(pure_props):
        raise ValueError("Antoine parameter calculation failed for mixture components")

    return antoine_params


# === STEP 6: Load and validate entrainer candidates ===
def load_and_validate_entrainers(config, verbose=False):
    """Load and validate predefined entrainers."""
    print_step(6, "Load and Validate Entrainer Candidates", verbose)

    screen_config = config['screen']
    entrainers = []
    entrainer_names = []
    validator = StructuralValidator()

    valid_count = 0
    invalid_count = 0

    if verbose:
        print("Validating predefined entrainers:")

    for entrainer in screen_config['entrainers']:
        name = entrainer['name']
        groups = entrainer['groups']
        is_valid = validator.is_valid(groups)

        if is_valid:
            entrainers.append(groups)
            entrainer_names.append(name)
            valid_count += 1
            if verbose:
                groups_str = ", ".join([f"{k}:{v}" for k, v in groups.items()])
                print(f"  ✓ {name}: {groups_str}")
        else:
            invalid_count += 1
            if verbose:
                groups_str = ", ".join([f"{k}:{v}" for k, v in groups.items()])
                print(f"  ✗ {name}: {groups_str} - Invalid")

    if verbose:
        print(f"\nSummary: {valid_count} valid, {invalid_count} invalid")

    return entrainers, entrainer_names


# === STEP 7: Check UNIFAC parameters ===
def check_entrainer_parameters(entrainers, mixture, verbose=False):
    """Check UNIFAC parameters for entrainer candidates."""
    print_step(7, "Check Entrainer Parameters (UNIFAC)", verbose)

    result = check_unifac_parameters(mixture=mixture, solvent_candidates=entrainers)

    compatible_indices = result['compatible_solvents']
    compatible = [entrainers[i] for i in compatible_indices]

    if verbose:
        print(f"Checked: {len(entrainers)}")
        print(f"Compatible: {len(compatible)}")
        print(f"Incompatible: {len(entrainers) - len(compatible)}")
        print(f"\nSummary: {len(compatible)}/{len(entrainers)} have complete UNIFAC parameters")

    return compatible, compatible_indices


# === STEP 8: Calculate pure component properties (entrainers) ===
def calculate_entrainer_pure_properties(entrainers, config, verbose=False):
    """Calculate pure component properties for entrainers."""
    print_step(8, "Calculate Pure Component Properties (Entrainers)", verbose)

    prop_calc = PropertyCalculator()
    entrainer_compounds = {f"entrainer_{i+1}": ent for i, ent in enumerate(entrainers)}

    prop_list = ['tc', 'pc', 'tb', 'tm', 'ac', 'hv']
    entrainer_props_raw = prop_calc.calculate(entrainer_compounds, prop_list)

    # Filter None values (missing groups)
    entrainer_props = {name: props for name, props in entrainer_props_raw.items() if props is not None}

    if verbose:
        dropped = len(entrainer_props_raw) - len(entrainer_props)
        if dropped > 0:
            print(f"Dropped: {dropped} (missing group contributions)")
        print(f"\nSummary: {len(entrainer_props)}/{len(entrainers)} properties calculated")

    return entrainer_props


# === STEP 9: Screen based on pure property constraints ===
def screen_pure_properties(entrainers, entrainer_props, config, verbose=False):
    """Screen entrainers based on pure property constraints."""
    print_step(9, "Screen Pure Properties", verbose)

    constraints = config['screening']['pure']
    screened = {}
    rejected = {}

    for name, props in entrainer_props.items():
        idx = int(name.split('_')[1]) - 1
        groups = entrainers[idx]
        groups_str = ", ".join([f"{k}:{v}" for k, v in groups.items()])

        # Sanity check: Tb < Tc
        if props.get('tb') and props.get('tc') and props['tb'] >= props['tc']:
            rejected[name] = {
                'groups': groups_str,
                'reason': f"Tb ({props['tb']:.1f}K) >= Tc ({props['tc']:.1f}K)"
            }
            continue

        # Check constraints
        passes = True
        rejection_reason = None

        for prop, (min_val, max_val) in constraints.items():
            if prop in props and props[prop] is not None:
                if not (min_val <= props[prop] <= max_val):
                    passes = False
                    rejection_reason = f"{prop}={props[prop]:.1f} outside [{min_val}, {max_val}]"
                    break
            else:
                passes = False
                rejection_reason = f"Missing {prop}"
                break

        if passes:
            screened[name] = {
                'groups': groups,
                'properties': props
            }
        else:
            rejected[name] = {
                'groups': groups_str,
                'reason': rejection_reason
            }

    if verbose:
        print(f"Initial: {len(entrainer_props)}")
        print(f"Passed: {len(screened)}")
        print(f"Rejected: {len(rejected)}")

        if rejected:
            print(f"\nRejection reasons:")
            for name, data in rejected.items():
                print(f"  {name} ({data['groups']}): {data['reason']}")

        print(f"\nSummary: {len(screened)}/{len(entrainer_props)} passed pure property screening")

    return screened


# === STEP 10: Calculate Antoine parameters (entrainers) ===
def calculate_entrainer_antoine(screened_entrainers, verbose=False):
    """Calculate Antoine parameters for screened entrainers."""
    print_step(10, "Calculate Antoine Parameters (Entrainers)", verbose)

    antoine_calc = AntoineCalculator()
    antoine_params = {}
    failed = 0

    for name, data in screened_entrainers.items():
        props = data['properties']
        groups = data['groups']

        required = ['tc', 'pc', 'tb', 'ac', 'hv']
        if all(prop in props and props[prop] is not None for prop in required):
            compound_dict = {name: groups}
            results = antoine_calc.calculate_antoine_parameters(
                compound_dict, critical_properties=props, method='enthalpy_vap'
            )

            if results['termination_condition'] == 'optimal':
                antoine_params[name] = {
                    'A': results['A'],
                    'B': results['B'],
                    'C': results['C']
                }
            else:
                failed += 1
        else:
            failed += 1

    if verbose:
        print(f"Successful: {len(antoine_params)}")
        print(f"Failed: {failed}")
        print(f"\nSummary: {len(antoine_params)}/{len(screened_entrainers)} calculated")

    return antoine_params


# === STEP 11: Perform miscibility screening over temperature range ===
def screen_miscibility(screened_entrainers, mixture, binary_props, rk_data, qk_data, interaction_data, verbose=False):
    """
    Screen entrainers for miscibility over temperature ranges for A-E and B-E pairs.

    Uses UNIFAC to evaluate activity coefficients and check:
    - Gibbs mixing energy (G_mix) for phase stability
    - Second derivative of G_mix for thermodynamic consistency

    Tests two binary pairs:
    - A-E: from Tb_A to Tb_E
    - B-E: from Tb_B to Tb_E

    Removes entrainers that show phase splitting or instability at any temperature.
    """
    print_step(11, "Screen Miscibility (Temperature Range)", verbose)

    # Composition grid for binary testing
    n_data = 100
    x0 = 0.0
    xf = 1.0
    x1 = np.linspace(x0, xf, n_data)
    T_step = 1.0

    passed_miscibility = {}
    failed_miscibility = {}

    tb_A = binary_props['A']['tb']
    tb_B = binary_props['B']['tb']

    for entrainer_name, entrainer_data in screened_entrainers.items():
        groups_str = ", ".join([f"{k}:{v}" for k, v in entrainer_data['groups'].items()])

        try:
            tb_E = entrainer_data['properties']['tb']

            # Determine temperature ranges for A-E and B-E pairs
            T_min_AE = min(tb_A, tb_E)
            T_max_AE = max(tb_A, tb_E)
            T_range_AE = np.arange(T_min_AE, T_max_AE + T_step, T_step)

            T_min_BE = min(tb_B, tb_E)
            T_max_BE = max(tb_B, tb_E)
            T_range_BE = np.arange(T_min_BE, T_max_BE + T_step, T_step)

            miscible_for_all_temps = True
            failing_pair = None
            failing_temps = []

            # Test A-E miscibility
            for T_test in T_range_AE:
                x_AE = np.column_stack([x1, 1 - x1])

                results = unifac(
                    {'A': mixture['A'], 'E': entrainer_data['groups']},
                    x_AE,
                    np.full(n_data, T_test),
                    rk_data, qk_data, interaction_data
                )

                gamma = results['gamma']

                # Calculate Gibbs mixing energy
                G_mix = (np.sum(x_AE * np.log(gamma), axis=1) +
                         np.sum(np.ma.log(x_AE).filled(0) * x_AE, axis=1))

                # Second derivative test
                if len(G_mix) > 2:
                    G2_mix = np.diff(np.diff(G_mix))

                    # Check miscibility criteria (excluding endpoints)
                    G_mix_check = np.all(G_mix[1:-1] <= 0)
                    G2_mix_check = np.all(G2_mix >= 0)

                    if not (G_mix_check and G2_mix_check):
                        miscible_for_all_temps = False
                        failing_pair = "A-E"
                        failing_temps.append(T_test)

            if miscible_for_all_temps:
                failing_temps = []
                # Test B-E miscibility (only if A-E passed)
                for T_test in T_range_BE:
                    x_BE = np.column_stack([x1, 1 - x1])

                    results = unifac(
                        {'B': mixture['B'], 'E': entrainer_data['groups']},
                        x_BE,
                        np.full(n_data, T_test),
                        rk_data, qk_data, interaction_data
                    )

                    gamma = results['gamma']

                    # Calculate Gibbs mixing energy
                    G_mix = (np.sum(x_BE * np.log(gamma), axis=1) +
                             np.sum(np.ma.log(x_BE).filled(0) * x_BE, axis=1))

                    # Second derivative test
                    if len(G_mix) > 2:
                        G2_mix = np.diff(np.diff(G_mix))

                        # Check miscibility criteria (excluding endpoints)
                        G_mix_check = np.all(G_mix[1:-1] <= 0)
                        G2_mix_check = np.all(G2_mix >= 0)

                        if not (G_mix_check and G2_mix_check):
                            miscible_for_all_temps = False
                            failing_pair = "B-E"
                            failing_temps.append(T_test)

            if miscible_for_all_temps:
                passed_miscibility[entrainer_name] = entrainer_data
            else:
                if len(failing_temps) > 1:
                    T_min = failing_temps[0]
                    T_max = failing_temps[-1]
                    temp_range_str = f"{T_min:.1f}-{T_max:.1f}K"
                else:
                    temp_range_str = f"{failing_temps[0]:.1f}K"
                failed_miscibility[entrainer_name] = {
                    'groups': groups_str,
                    'reason': f"Phase instability {failing_pair} T={temp_range_str}"
                }

        except Exception as e:
            failed_miscibility[entrainer_name] = {
                'groups': groups_str,
                'reason': f"Miscibility test error: {type(e).__name__}"
            }

    if verbose:
        print(f"Input: {len(screened_entrainers)}")
        print(f"Passed: {len(passed_miscibility)}")
        print(f"Failed: {len(failed_miscibility)}")

        if failed_miscibility:
            print(f"\nFailed miscibility:")
            for name, data in failed_miscibility.items():
                print(f"  {name} ({data['groups']}): {data['reason']}")

        print(f"\nSummary: {len(passed_miscibility)}/{len(screened_entrainers)} passed miscibility screening")

    return passed_miscibility


# === STEP 12: Perform VLE ===
def perform_vle(screened_entrainers, entrainer_antoine, mixture, binary_antoine, binary_props, verbose=False):
    """Perform VLE calculations for entrainers."""
    print_step(12, "Perform VLE Calculations", verbose)

    rk_data, qk_data, interaction_data = load_unifac_data()
    P = 1.01325  # bar
    n_data = 50

    tb_A = binary_props['A']['tb']
    tb_B = binary_props['B']['tb']

    vle_results = {}
    successful = []
    failed = {}

    total = len([name for name in screened_entrainers.keys() if name in entrainer_antoine])

    for entrainer_name, entrainer_data in screened_entrainers.items():
        groups_str = ", ".join([f"{k}:{v}" for k, v in entrainer_data['groups'].items()])

        if entrainer_name not in entrainer_antoine:
            failed[entrainer_name] = {
                'groups': groups_str,
                'reason': "No Antoine parameters"
            }
            continue

        try:
            # Build ternary mixture
            ternary_mixture = dict(mixture)
            ternary_mixture[entrainer_name] = entrainer_data['groups']

            # Antoine parameters
            ternary_antoine = dict(binary_antoine)
            ternary_antoine[entrainer_name] = entrainer_antoine[entrainer_name]

            tb_E = entrainer_data['properties']['tb']

            # Composition grids
            x1_ae = np.linspace(0.0, 1.0, n_data)
            x2_ae = np.zeros(n_data)
            x3_ae = 1 - x1_ae

            x1_be = np.zeros(n_data)
            x2_be = np.linspace(0.0, 1.0, n_data)
            x3_be = 1 - x2_be

            x_all = np.vstack([
                np.column_stack([x1_ae, x2_ae, x3_ae]),
                np.column_stack([x1_be, x2_be, x3_be])
            ])

            # Temperature guess
            T0_ae = tb_E * (1 - x1_ae) + tb_A * x1_ae
            T0_be = tb_E * (1 - x2_be) + tb_B * x2_be
            T_guess = np.concatenate([T0_ae, T0_be])

            # VLE calculation
            vle_solution, vle_model = pyo_vle(
                x_all, T_guess, ternary_mixture,
                rk_data, qk_data, interaction_data, ternary_antoine, verbose=False
            )

            if vle_solution is None or vle_model is None:
                failed[entrainer_name] = {
                    'groups': groups_str,
                    'reason': "VLE solver returned None"
                }
                continue

            status = str(vle_solution.solver.status)
            cond = str(vle_solution.solver.termination_condition)

            if status != 'ok' or cond != 'optimal':
                failed[entrainer_name] = {
                    'groups': groups_str,
                    'reason': f"VLE solver: {status}/{cond}"
                }
                continue

            # Extract results
            n_points = len(x_all)
            T_bubble = np.zeros(n_points)
            y_vapor = np.zeros((n_points, 3))
            gamma = np.zeros((n_points, 3))
            psat = np.zeros((n_points, 3))

            for i in range(n_points):
                T_bubble[i] = vle_model.T[i].value if vle_model.T[i].value else T_guess[i]
                for j in range(3):
                    y_vapor[i, j] = vle_model.y[i, j].value
                    gamma[i, j] = vle_model.gamma[i, j].value
                    psat[i, j] = vle_model.psat[i, j].value

            # Calculate mixture properties
            S = gamma[0, 0] / gamma[0, 1]
            C = 1 / gamma[0, 1]
            SxC = S * C
            alpha = (gamma[0, 0] / psat[0, 0]) / (gamma[0, 1] / psat[0, 1])

            # Azeotrope detection
            az_AE = np.log(gamma[0, 0] * psat[0, 0] / P) * (-np.log(gamma[n_data-1, 2] * psat[n_data-1, 2] / P))
            az_BE = np.log(gamma[n_data, 1] * psat[n_data, 1] / P) * (-np.log(gamma[-1, 2] * psat[-1, 2] / P))

            # Gibbs mixing
            G_mix = (np.sum(x_all * np.log(gamma), axis=1) +
                    np.sum(np.ma.log(x_all).filled(0) * x_all, axis=1))
            G_mix_AE = G_mix[:n_data]
            G_mix_BE = G_mix[n_data:]
            G2_mix_AE = np.diff(np.diff(G_mix_AE))
            G2_mix_BE = np.diff(np.diff(G_mix_BE))

            vle_results[entrainer_name] = {
                'x': x_all,
                'T': T_bubble,
                'y': y_vapor,
                'gamma': gamma,
                'psat': psat,
                'S': S,
                'C': C,
                'SxC': SxC,
                'alpha': alpha,
                'az_AE': az_AE,
                'az_BE': az_BE,
                'G_mix_AE': G_mix_AE,
                'G_mix_BE': G_mix_BE,
                'G2_mix_AE': G2_mix_AE,
                'G2_mix_BE': G2_mix_BE,
                'compound_names': list(ternary_mixture.keys())
            }
            successful.append(entrainer_name)

        except Exception as e:
            failed[entrainer_name] = {
                'groups': groups_str,
                'reason': f"VLE error: {type(e).__name__}"
            }

    if verbose:
        print(f"Total: {total}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print(f"\nFailure reasons:")
            for name, data in failed.items():
                print(f"  {name} ({data['groups']}): {data['reason']}")

        print(f"\nSummary: {len(successful)}/{total} VLE calculations successful")

    return vle_results, successful


# === STEP 13: Screen based on mixture constraints ===
def screen_mixture_properties(vle_results, successful, screened_entrainers, config, verbose=False):
    """Screen entrainers based on mixture property constraints."""
    print_step(13, "Screen Mixture Properties", verbose)

    mixture_config = config['screening']['mixture']
    passed = {}
    rejected = {}

    for name in successful:
        results = vle_results[name]
        groups_str = ", ".join([f"{k}:{v}" for k, v in screened_entrainers[name]['groups'].items()])

        # Check constraints
        rejection_reasons = []

        if mixture_config.get('no_azeotrope_AE', False):
            if results['az_AE'] <= 0:
                rejection_reasons.append(f"Azeotrope A-E (az_AE={results['az_AE']:.3f})")

        if mixture_config.get('no_azeotrope_BE', False):
            if results['az_BE'] <= 0:
                rejection_reasons.append(f"Azeotrope B-E (az_BE={results['az_BE']:.3f})")

        if mixture_config.get('miscibility', False):
            if not np.all(results['G_mix_AE'][1:-1] <= 0):
                rejection_reasons.append("Phase split A-E (G_mix > 0)")
            if not np.all(results['G_mix_BE'][1:-1] <= 0):
                rejection_reasons.append("Phase split B-E (G_mix > 0)")
            if not np.all(results['G2_mix_AE'][1:-1] >= 0):
                rejection_reasons.append("Instability A-E (G2_mix < 0)")
            if not np.all(results['G2_mix_BE'][1:-1] >= 0):
                rejection_reasons.append("Instability B-E (G2_mix < 0)")

        S_min = mixture_config.get('selectivity_min', 0)
        if results['S'] < S_min:
            rejection_reasons.append(f"Low selectivity (S={results['S']:.2f} < {S_min})")

        if not rejection_reasons:
            passed[name] = results
        else:
            rejected[name] = {
                'groups': groups_str,
                'reason': "; ".join(rejection_reasons)
            }

    if verbose:
        print(f"Input: {len(successful)}")
        print(f"Passed: {len(passed)}")
        print(f"Rejected: {len(rejected)}")

        if rejected:
            print(f"\nRejection reasons:")
            for name, data in rejected.items():
                print(f"  {name} ({data['groups']}): {data['reason']}")

        print(f"\nSummary: {len(passed)}/{len(successful)} passed mixture screening")

    return passed


# === STEP 14: Perform ISS calculation ===
def perform_iss(screened_mixture, compatible_entrainers, mixture, binary_antoine,
                entrainer_antoine, binary_props, config, verbose=False):
    """Perform ISS calculations."""
    print_step(14, "Perform ISS Calculations", verbose)

    rk_data, qk_data, interaction_data = load_unifac_data()

    iss_config = config['iss']
    x_iso = iss_config['x_iso']
    xA = iss_config['x_min']

    boiling_points = [binary_props['A']['tb'], binary_props['B']['tb']]
    T_min = min(boiling_points)
    T_max = max(boiling_points)
    T0 = np.array([(T_max + T_min) / 2, (T_max + T_min) / 2])
    x0 = np.array([[x_iso, 0.0, 1 - x_iso],
                   [xA, 0.0, 1 - xA]])

    iss_results = {}
    failed = {}

    for entrainer_name in screened_mixture.keys():
        groups_str = ", ".join([f"{k}:{v}" for k, v in compatible_entrainers[entrainer_name]['groups'].items()])

        try:
            # Build ternary
            ternary_mixture = dict(mixture)
            ternary_mixture[entrainer_name] = compatible_entrainers[entrainer_name]['groups']

            ternary_antoine = dict(binary_antoine)
            ternary_antoine[entrainer_name] = entrainer_antoine[entrainer_name]

            # ISS calculation
            results, model = pyo_iss(x0, T0, ternary_mixture, rk_data, qk_data,
                                    interaction_data, ternary_antoine, verbose=False)

            if results is None or model is None:
                failed[entrainer_name] = {
                    'groups': groups_str,
                    'reason': "ISS solver returned None"
                }
                continue

            status = str(results.solver.status)
            cond = str(results.solver.termination_condition)

            if status != 'ok' or cond != 'optimal':
                failed[entrainer_name] = {
                    'groups': groups_str,
                    'reason': f"ISS solver: {status}/{cond}"
                }
                continue

            import pyomo.environ as pyo
            ED = pyo.value(model.obj)
            R_value = pyo.value(model.R)

            # Extract detailed results for both operating points
            n_comp = len(model.i)  # Number of components
            n_data = len(model.dp)  # Number of operating points

            if verbose:
                print(f"  ISS Results for {entrainer_name}: n_comp={n_comp}, n_data={n_data}")

            # Initialize arrays for the 2 operating points
            gamma_vals = np.zeros((n_data, n_comp))
            K_vals = np.zeros((n_data, n_comp))
            x_vals = np.zeros((n_data, n_comp))
            y_vals = np.zeros((n_data, n_comp))
            T_vals = np.zeros(n_data)

            # Extract values from model
            for j in range(n_data):
                T_vals[j] = pyo.value(model.T[j])
                for i in range(n_comp):
                    gamma_vals[j, i] = pyo.value(model.gamma[j, i])
                    K_vals[j, i] = pyo.value(model.K[j, i])
                    x_vals[j, i] = pyo.value(model.x[j, i])
                    y_vals[j, i] = pyo.value(model.y[j, i])

            iss_results[entrainer_name] = {
                'ED': ED,
                'R': R_value,
                'groups': compatible_entrainers[entrainer_name]['groups'],
                'gamma': gamma_vals,
                'K': K_vals,
                'x': x_vals,
                'y': y_vals,
                'T': T_vals,
                'compound_names': list(ternary_mixture.keys())
            }

        except Exception as e:
            failed[entrainer_name] = {
                'groups': groups_str,
                'reason': f"ISS error: {type(e).__name__}"
            }

    if verbose:
        print(f"Input: {len(screened_mixture)}")
        print(f"Successful: {len(iss_results)}")
        print(f"Failed: {len(failed)}")

        if failed:
            print(f"\nFailure reasons:")
            for name, data in failed.items():
                print(f"  {name} ({data['groups']}): {data['reason']}")

        print(f"\nSummary: {len(iss_results)}/{len(screened_mixture)} ISS calculations successful")

    return iss_results


# === Save results ===
def save_results(iss_results, screened_mixture, output_dir, config, verbose=False):
    """Save results to files."""
    import json
    import csv

    if verbose:
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

    formats = config['output']['formats']

    # Prepare data
    results_list = []
    for name, iss_data in iss_results.items():
        groups_str = ", ".join([f"{k}:{v}" for k, v in iss_data['groups'].items()])
        mixture_data = screened_mixture[name]

        results_list.append({
            'entrainer': name,
            'groups': groups_str,
            'ED': iss_data['ED'],
            'R': iss_data['R'],
            'S': mixture_data['S'],
            'C': mixture_data['C'],
            'SxC': mixture_data['SxC'],
            'alpha': mixture_data['alpha']
        })

    # Sort by ED
    results_list.sort(key=lambda x: x['ED'])

    # Save CSV
    if 'csv' in formats:
        csv_path = output_dir / 'results.csv'
        with open(csv_path, 'w', newline='') as f:
            if results_list:
                writer = csv.DictWriter(f, fieldnames=results_list[0].keys())
                writer.writeheader()
                writer.writerows(results_list)
        if verbose:
            print(f"✓ Saved: {csv_path}")

    # Save JSON - Summary
    if 'json' in formats:
        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        if verbose:
            print(f"✓ Saved: {json_path}")

    # Save detailed JSON with all ISS data
    if 'json' in formats:
        detailed_results = []
        for name, iss_data in iss_results.items():
            groups_str = ", ".join([f"{k}:{v}" for k, v in iss_data['groups'].items()])
            mixture_data = screened_mixture[name]

            # Build operating points data with labeled components (A, B, E)
            operating_points = []
            n_comp = iss_data['x'].shape[1] if iss_data['x'].ndim > 1 else 1
            comp_labels = ['A', 'B', 'E'][:n_comp]  # Use available labels

            for j in range(len(iss_data['T'])):
                op_point = {
                    'T': float(iss_data['T'][j]),
                    'x': {},
                    'y': {},
                    'gamma': {},
                    'K': {}
                }

                # Extract component data with proper indexing
                for i, label in enumerate(comp_labels):
                    op_point['x'][label] = float(iss_data['x'][j, i]) if n_comp > 1 else float(iss_data['x'][j])
                    op_point['y'][label] = float(iss_data['y'][j, i]) if n_comp > 1 else float(iss_data['y'][j])
                    op_point['gamma'][label] = float(iss_data['gamma'][j, i]) if n_comp > 1 else float(iss_data['gamma'][j])
                    op_point['K'][label] = float(iss_data['K'][j, i]) if n_comp > 1 else float(iss_data['K'][j])

                operating_points.append(op_point)

            detailed_results.append({
                'entrainer': name,
                'groups': groups_str,
                'compound_names': iss_data['compound_names'],
                'ED': float(iss_data['ED']),
                'R': float(iss_data['R']),
                'S': float(mixture_data['S']),
                'C': float(mixture_data['C']),
                'SxC': float(mixture_data['SxC']),
                'alpha': float(mixture_data['alpha']),
                'iss_data': {
                    'operating_point_1': {
                        'description': 'Iso-volatility point',
                        **operating_points[0]
                    },
                    'operating_point_2': {
                        'description': 'ISS point',
                        **operating_points[1]
                    }
                }
            })

        # Sort by ED
        detailed_results.sort(key=lambda x: x['ED'])

        detailed_json_path = output_dir / 'results_detailed.json'
        with open(detailed_json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        if verbose:
            print(f"✓ Saved: {detailed_json_path}")

    if verbose:
        print(f"\nSummary: Saved {len(results_list)} results in {len(formats)} format(s)")


def main():
    parser = argparse.ArgumentParser(description='Screen-ISS: Screen predefined entrainers for ISS performance')
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    verbose = args.verbose or config['output'].get('verbose', False)

    # Setup output and logging
    output_dir = setup_output_dir(config)

    # Setup logging to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"screen_run_{timestamp}.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger

    try:
        # Header
        print("="*60)
        print("Screen-ISS: Entrainer Screening")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Config: {args.config}")
        print(f"Log file: {log_file}")

        # === BINARY MIXTURE WORKFLOW ===
        mixture = validate_mixture(config, verbose)
        check_mixture_parameters(mixture, verbose)
        binary_props = calculate_mixture_pure_properties(mixture, verbose)

        # Validate boiling point order: Tb_A < Tb_B
        tb_A = binary_props['A']['tb']
        tb_B = binary_props['B']['tb']
        if tb_A >= tb_B:
            raise ValueError(f"Invalid boiling point order: Tb_A ({tb_A:.2f}K) must be < Tb_B ({tb_B:.2f}K)")
        if verbose:
            print(f"\n✓ Boiling point validation: Tb_A ({tb_A:.2f}K) < Tb_B ({tb_B:.2f}K)")

        binary_props = override_mixture_properties(binary_props, config, verbose)
        binary_antoine = calculate_mixture_antoine(mixture, binary_props, verbose)

        # === ENTRAINER WORKFLOW ===
        # Step 6: Load predefined entrainers
        entrainers, entrainer_names = load_and_validate_entrainers(config, verbose)

        if not entrainers:
            print("\n⚠ No entrainers to process")
            return

        # Step 7-10: Initial screening pipeline
        compatible, compatible_indices = check_entrainer_parameters(entrainers, mixture, verbose)

        if not compatible:
            print("\n⚠ No compatible entrainers found")
            return

        entrainer_props = calculate_entrainer_pure_properties(compatible, config, verbose)
        screened_pure = screen_pure_properties(compatible, entrainer_props, config, verbose)

        if not screened_pure:
            print("\n⚠ No entrainers passed pure property screening")
            return

        entrainer_antoine = calculate_entrainer_antoine(screened_pure, verbose)

        if not entrainer_antoine:
            print("\n⚠ No entrainers have valid Antoine parameters")
            return

        # Step 11: Early miscibility screening
        rk_data, qk_data, interaction_data = load_unifac_data()
        screened_miscibility = screen_miscibility(
            screened_pure, mixture, binary_props,
            rk_data, qk_data, interaction_data, verbose
        )

        if not screened_miscibility:
            print("\n⚠ No entrainers passed miscibility screening")
            return

        # Filter entrainer_antoine to only include miscibility-passing entrainers
        entrainer_antoine_filtered = {k: v for k, v in entrainer_antoine.items()
                                     if k in screened_miscibility}

        # Step 12: VLE (now only on miscibility-compatible entrainers)
        vle_results, successful = perform_vle(screened_miscibility, entrainer_antoine_filtered, mixture,
                                             binary_antoine, binary_props, verbose)

        if not successful:
            print("\n⚠ No successful VLE calculations")
            return

        # Step 13: Mixture property screening
        screened_mixture = screen_mixture_properties(vle_results, successful, screened_miscibility, config, verbose)

        if not screened_mixture:
            print("\n⚠ No entrainers passed mixture property screening")
            return

        # Step 14: ISS calculation
        iss_results = perform_iss(screened_mixture, screened_miscibility, mixture, binary_antoine,
                                 entrainer_antoine_filtered, binary_props, config, verbose)

        # === RESULTS ===
        if iss_results:
            save_results(iss_results, screened_mixture, output_dir, config, verbose)

            if verbose:
                print("\n" + "="*60)
                print("TOP ENTRAINERS")
                print("="*60)
                sorted_entrainers = sorted(iss_results.items(), key=lambda x: x[1]['ED'])
                for i, (name, data) in enumerate(sorted_entrainers[:5], 1):
                    groups_str = ", ".join([f"{k}:{v}" for k, v in data['groups'].items()])
                    print(f"{i}. {name}")
                    print(f"   Groups: {groups_str}")
                    print(f"   ED: {data['ED']:.6f}, R: {data['R']:.6f}")
                print(f"\nTotal: {len(iss_results)} entrainers evaluated")
        else:
            print("\n⚠ No entrainers completed ISS calculations")

        # Footer
        print("\n" + "="*60)
        print("Screen-ISS COMPLETE")
        print(f"Output: {output_dir}")
        print("="*60)

    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()


if __name__ == '__main__':
    main()
