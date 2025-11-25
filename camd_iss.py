#!/usr/bin/env python3
"""
CAMD-ISS: Computer-Aided Molecular Design for Infinite-dilution Selectivity and Solubility

Config-driven script for solvent design and screening.

Workflow Steps:
1. Check defined mixture for validity
2. Check mixture for pure component and binary parameters
3. Calculate pure component properties (mixture)
4. Override if properties are provided by user
5. Calculate Antoine parameters (mixture)
6. [Screen mode] Test defined solvent candidates OR [Design mode] Generate solvent candidates
7. Check and handle missing pure component and binary parameters (solvents)
8. Calculate pure component properties (solvents)
9. Screen based on pure property constraints
10. Calculate Antoine parameters (solvents)
11. Perform VLE
12. Screen based on mixture constraints
13. Perform ISS calculation
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

from src.molecules import StructuralValidator, generate_molecules
from src.pure import PropertyCalculator, AntoineCalculator
from src.data import load_unifac_data, check_unifac_parameters
from src.mixture.unifac import pyo_vle
from src.mixture.unifac.pyomo_iss import pyo_iss


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


# === STEP 6a: Test defined solvent candidates (screen mode) ===
def load_and_validate_solvents(config, verbose=False):
    """Load and validate predefined solvents."""
    print_step(6, "Load and Validate Solvent Candidates", verbose)

    screen_config = config['screen']
    solvents = []
    solvent_names = []
    validator = StructuralValidator()

    valid_count = 0
    invalid_count = 0

    if verbose:
        print("Validating predefined solvents:")

    for solvent in screen_config['solvents']:
        name = solvent['name']
        groups = solvent['groups']
        is_valid = validator.is_valid(groups)

        if is_valid:
            solvents.append(groups)
            solvent_names.append(name)
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

    return solvents, solvent_names


# === STEP 6b: Generate solvent candidates (design mode) ===
def generate_solvent_candidates(config, mixture, verbose=False):
    """Generate solvent candidates using MILP."""
    print_step(6, "Generate Solvent Candidates", verbose)

    design_config = config['design']
    binary_components = list(mixture.values())

    if verbose:
        print(f"Design groups: {', '.join(design_config['groups'])}")
        print(f"Constraints: {design_config['constraints']['min_groups']}-{design_config['constraints']['max_groups']} total groups, "
              f"{design_config['constraints']['min_func_groups']}-{design_config['constraints']['max_func_groups']} functional groups")

    candidates = generate_molecules(
        design_groups=design_config['groups'],
        min_groups=design_config['constraints']['min_groups'],
        max_groups=design_config['constraints']['max_groups'],
        min_func_groups=design_config['constraints']['min_func_groups'],
        max_func_groups=design_config['constraints']['max_func_groups'],
        molecular_type=design_config['constraints']['molecular_type'],
        max_solutions=design_config['max_candidates'],
        exclude_compounds=binary_components,
        verbose=False
    )

    if verbose:
        print(f"\nSummary: Generated {len(candidates)} candidates")

    return candidates


# === STEP 7: Check and handle missing pure component and binary parameters ===
def check_solvent_parameters(solvents, mixture, verbose=False):
    """Check UNIFAC parameters for solvent candidates."""
    print_step(7, "Check Solvent Parameters (UNIFAC)", verbose)

    result = check_unifac_parameters(mixture=mixture, solvent_candidates=solvents)

    compatible_indices = result['compatible_solvents']
    compatible = [solvents[i] for i in compatible_indices]

    if verbose:
        print(f"Checked: {len(solvents)}")
        print(f"Compatible: {len(compatible)}")
        print(f"Incompatible: {len(solvents) - len(compatible)}")
        print(f"\nSummary: {len(compatible)}/{len(solvents)} have complete UNIFAC parameters")

    return compatible, compatible_indices


# === STEP 8: Calculate pure component properties (solvents) ===
def calculate_solvent_pure_properties(solvents, config, verbose=False):
    """Calculate pure component properties for solvents."""
    print_step(8, "Calculate Pure Component Properties (Solvents)", verbose)

    prop_calc = PropertyCalculator()
    solvent_compounds = {f"solvent_{i+1}": sol for i, sol in enumerate(solvents)}

    prop_list = ['tc', 'pc', 'tb', 'tm', 'ac', 'hv']
    solvent_props_raw = prop_calc.calculate(solvent_compounds, prop_list)

    # Filter None values (missing groups)
    solvent_props = {name: props for name, props in solvent_props_raw.items() if props is not None}

    if verbose:
        dropped = len(solvent_props_raw) - len(solvent_props)
        if dropped > 0:
            print(f"Dropped: {dropped} (missing group contributions)")
        print(f"\nSummary: {len(solvent_props)}/{len(solvents)} properties calculated")

    return solvent_props


# === STEP 9: Screen based on pure property constraints ===
def screen_pure_properties(solvents, solvent_props, config, verbose=False):
    """Screen solvents based on pure property constraints."""
    print_step(9, "Screen Pure Properties", verbose)

    constraints = config['screening']['pure']
    screened = {}
    rejected = {}

    for name, props in solvent_props.items():
        idx = int(name.split('_')[1]) - 1
        groups = solvents[idx]
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
        print(f"Initial: {len(solvent_props)}")
        print(f"Passed: {len(screened)}")
        print(f"Rejected: {len(rejected)}")

        if rejected:
            print(f"\nRejection reasons:")
            for name, data in rejected.items():
                print(f"  {name} ({data['groups']}): {data['reason']}")

        print(f"\nSummary: {len(screened)}/{len(solvent_props)} passed pure property screening")

    return screened


# === STEP 10: Calculate Antoine parameters (solvents) ===
def calculate_solvent_antoine(screened_solvents, verbose=False):
    """Calculate Antoine parameters for screened solvents."""
    print_step(10, "Calculate Antoine Parameters (Solvents)", verbose)

    antoine_calc = AntoineCalculator()
    antoine_params = {}
    failed = 0

    for name, data in screened_solvents.items():
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
        print(f"\nSummary: {len(antoine_params)}/{len(screened_solvents)} calculated")

    return antoine_params


# === STEP 11: Perform VLE ===
def perform_vle(screened_solvents, solvent_antoine, mixture, binary_antoine, binary_props, verbose=False):
    """Perform VLE calculations for solvents."""
    print_step(11, "Perform VLE Calculations", verbose)

    rk_data, qk_data, interaction_data = load_unifac_data()
    P = 1.01325  # bar
    n_data = 50

    tb_A = binary_props['A']['tb']
    tb_B = binary_props['B']['tb']

    vle_results = {}
    successful = []
    failed = {}

    total = len([name for name in screened_solvents.keys() if name in solvent_antoine])

    for solvent_name, solvent_data in screened_solvents.items():
        groups_str = ", ".join([f"{k}:{v}" for k, v in solvent_data['groups'].items()])

        if solvent_name not in solvent_antoine:
            failed[solvent_name] = {
                'groups': groups_str,
                'reason': "No Antoine parameters"
            }
            continue

        try:
            # Build ternary mixture
            ternary_mixture = dict(mixture)
            ternary_mixture[solvent_name] = solvent_data['groups']

            # Antoine parameters
            ternary_antoine = dict(binary_antoine)
            ternary_antoine[solvent_name] = solvent_antoine[solvent_name]

            tb_E = solvent_data['properties']['tb']

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
                failed[solvent_name] = {
                    'groups': groups_str,
                    'reason': "VLE solver returned None"
                }
                continue

            status = str(vle_solution.solver.status)
            cond = str(vle_solution.solver.termination_condition)

            if status != 'ok' or cond != 'optimal':
                failed[solvent_name] = {
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

            vle_results[solvent_name] = {
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
            successful.append(solvent_name)

        except Exception as e:
            failed[solvent_name] = {
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


# === STEP 12: Screen based on mixture constraints ===
def screen_mixture_properties(vle_results, successful, screened_solvents, config, verbose=False):
    """Screen solvents based on mixture property constraints."""
    print_step(12, "Screen Mixture Properties", verbose)

    mixture_config = config['screening']['mixture']
    passed = {}
    rejected = {}

    for name in successful:
        results = vle_results[name]
        groups_str = ", ".join([f"{k}:{v}" for k, v in screened_solvents[name]['groups'].items()])

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


# === STEP 13: Perform ISS calculation ===
def perform_iss(screened_mixture, compatible_solvents, mixture, binary_antoine,
                solvent_antoine, binary_props, config, verbose=False):
    """Perform ISS calculations."""
    print_step(13, "Perform ISS Calculations", verbose)

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

    for solvent_name in screened_mixture.keys():
        groups_str = ", ".join([f"{k}:{v}" for k, v in compatible_solvents[solvent_name]['groups'].items()])

        try:
            # Build ternary
            ternary_mixture = dict(mixture)
            ternary_mixture[solvent_name] = compatible_solvents[solvent_name]['groups']

            ternary_antoine = dict(binary_antoine)
            ternary_antoine[solvent_name] = solvent_antoine[solvent_name]

            # ISS calculation
            results, model = pyo_iss(x0, T0, ternary_mixture, rk_data, qk_data,
                                    interaction_data, ternary_antoine, verbose=False)

            if results is None or model is None:
                failed[solvent_name] = {
                    'groups': groups_str,
                    'reason': "ISS solver returned None"
                }
                continue

            status = str(results.solver.status)
            cond = str(results.solver.termination_condition)

            if status != 'ok' or cond != 'optimal':
                failed[solvent_name] = {
                    'groups': groups_str,
                    'reason': f"ISS solver: {status}/{cond}"
                }
                continue

            import pyomo.environ as pyo
            ED = pyo.value(model.obj)
            R_value = pyo.value(model.R)

            iss_results[solvent_name] = {
                'ED': ED,
                'R': R_value,
                'groups': compatible_solvents[solvent_name]['groups']
            }

        except Exception as e:
            failed[solvent_name] = {
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
            'solvent': name,
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

    # Save JSON
    if 'json' in formats:
        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        if verbose:
            print(f"✓ Saved: {json_path}")

    if verbose:
        print(f"\nSummary: Saved {len(results_list)} results in {len(formats)} format(s)")


def main():
    parser = argparse.ArgumentParser(description='CAMD-ISS: Config-driven solvent design/screening')
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
    log_file = output_dir / f"camd_run_{timestamp}.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger

    try:
        # Header
        print("="*60)
        print("CAMD-ISS: Computer-Aided Molecular Design")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {config['mode'].upper()}")
        print(f"Config: {args.config}")
        print(f"Log file: {log_file}")

        # === BINARY MIXTURE WORKFLOW ===
        mixture = validate_mixture(config, verbose)
        check_mixture_parameters(mixture, verbose)
        binary_props = calculate_mixture_pure_properties(mixture, verbose)
        binary_props = override_mixture_properties(binary_props, config, verbose)
        binary_antoine = calculate_mixture_antoine(mixture, binary_props, verbose)

        # === SOLVENT WORKFLOW ===
        # Step 6: Get candidates (mode-dependent)
        if config['mode'] == 'screen':
            candidates, solvent_names = load_and_validate_solvents(config, verbose)
        elif config['mode'] == 'design':
            candidates = generate_solvent_candidates(config, mixture, verbose)
        else:
            raise ValueError(f"Invalid mode: {config['mode']}. Must be 'design' or 'screen'")

        if not candidates:
            print("\n⚠ No candidates to process")
            return

        # Step 7-13: Screening pipeline
        compatible, compatible_indices = check_solvent_parameters(candidates, mixture, verbose)

        if not compatible:
            print("\n⚠ No compatible solvents found")
            return

        solvent_props = calculate_solvent_pure_properties(compatible, config, verbose)
        screened_pure = screen_pure_properties(compatible, solvent_props, config, verbose)

        if not screened_pure:
            print("\n⚠ No solvents passed pure property screening")
            return

        solvent_antoine = calculate_solvent_antoine(screened_pure, verbose)

        if not solvent_antoine:
            print("\n⚠ No solvents have valid Antoine parameters")
            return

        vle_results, successful = perform_vle(screened_pure, solvent_antoine, mixture,
                                             binary_antoine, binary_props, verbose)

        if not successful:
            print("\n⚠ No successful VLE calculations")
            return

        screened_mixture = screen_mixture_properties(vle_results, successful, screened_pure, config, verbose)

        if not screened_mixture:
            print("\n⚠ No solvents passed mixture property screening")
            return

        iss_results = perform_iss(screened_mixture, screened_pure, mixture, binary_antoine,
                                 solvent_antoine, binary_props, config, verbose)

        # === RESULTS ===
        if iss_results:
            save_results(iss_results, screened_mixture, output_dir, config, verbose)

            if verbose:
                print("\n" + "="*60)
                print("TOP SOLVENTS")
                print("="*60)
                sorted_solvents = sorted(iss_results.items(), key=lambda x: x[1]['ED'])
                for i, (name, data) in enumerate(sorted_solvents[:5], 1):
                    groups_str = ", ".join([f"{k}:{v}" for k, v in data['groups'].items()])
                    print(f"{i}. {name}")
                    print(f"   Groups: {groups_str}")
                    print(f"   ED: {data['ED']:.6f}, R: {data['R']:.6f}")
                print(f"\nTotal: {len(iss_results)} solvents found")
        else:
            print("\n⚠ No solvents completed ISS calculations")

        # Footer
        print("\n" + "="*60)
        print("CAMD-ISS COMPLETE")
        print(f"Output: {output_dir}")
        print("="*60)

    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()


if __name__ == '__main__':
    main()
