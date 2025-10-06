#!/usr/bin/env python3
"""
CAMD-ISS: Computer-Aided Molecular Design for Infinite-dilution Selectivity and Solubility

Config-driven script for solvent design and screening.
"""

import sys
import os
import yaml
import argparse
import numpy as np
import logging
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


def get_mixture(config):
    """Extract binary mixture from config."""
    mixture = {
        'A': config['mixture']['component_a']['groups'],
        'B': config['mixture']['component_b']['groups']
    }
    return mixture


def calculate_pure_properties(mixture, config, verbose=False):
    """Calculate pure component properties for binary mixture."""
    if verbose:
        print("\n" + "="*60)
        print("STEP 1: Pure Component Properties")
        print("="*60)

    prop_list = ['tc', 'pc', 'tb', 'ac', 'hv']
    pure_calc = PropertyCalculator()
    pure_props = pure_calc.calculate(mixture, prop_list)

    # Apply manual overrides if provided
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
                            print(f"  Override {comp_key}.{prop} = {value}")

    if verbose:
        print("\nCalculated properties:")
        for comp, props in pure_props.items():
            print(f"  {comp}: Tc={props['tc']:.1f}K, Pc={props['pc']:.1f}bar, Tb={props['tb']:.1f}K")
        print(f"\nSummary: {len(pure_props)} components, {overrides_applied} overrides applied")

    return pure_props


def calculate_antoine_parameters(mixture, pure_props, verbose=False):
    """Calculate Antoine parameters for components."""
    if verbose:
        print("\n" + "="*60)
        print("STEP 2: Antoine Parameters")
        print("="*60)

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
                print(f"  ✓ {compound}: A={results['A']:.3f}, B={results['B']:.1f}, C={results['C']:.1f}")
        else:
            failed.append(compound)
            if verbose:
                print(f"  ✗ {compound}: Failed")

    if verbose:
        print(f"\nSummary: {len(antoine_params)}/{len(pure_props)} successful")
        if failed:
            print(f"  Failed: {', '.join(failed)}")

    return antoine_params


def check_binary_unifac(mixture, verbose=False):
    """Check UNIFAC parameters for binary mixture."""
    if verbose:
        print("\n" + "="*60)
        print("STEP 3: Binary Mixture UNIFAC Check")
        print("="*60)

    result = check_unifac_parameters(mixture=mixture)

    if result['compatible']:
        if verbose:
            print("  ✓ All UNIFAC parameters available")
            print("\nSummary: Binary mixture is UNIFAC-compatible")
        return True
    else:
        if verbose:
            print("  ✗ Missing UNIFAC parameters")
            print("\nSummary: Binary mixture has missing UNIFAC parameters")
        return False


def generate_solvent_candidates(config, mixture, verbose=False):
    """Generate solvent candidates using MILP (design mode)."""
    if verbose:
        print("\n" + "="*60)
        print("STEP 4: Generate Solvent Candidates (Design Mode)")
        print("="*60)

    design_config = config['design']

    # Extract binary components to exclude
    binary_components = list(mixture.values())

    candidates = generate_molecules(
        design_groups=design_config['groups'],
        min_groups=design_config['constraints']['min_groups'],
        max_groups=design_config['constraints']['max_groups'],
        min_func_groups=design_config['constraints']['min_func_groups'],
        max_func_groups=design_config['constraints']['max_func_groups'],
        molecular_type=design_config['constraints']['molecular_type'],
        max_solutions=design_config['max_candidates'],
        exclude_compounds=binary_components,
        verbose=False  # Suppress internal verbosity
    )

    if verbose:
        print(f"\nSummary: Generated {len(candidates)} candidate solvents")

    return candidates


def get_predefined_solvents(config, verbose=False):
    """Get predefined solvents from config (screen mode)."""
    if verbose:
        print("\n=== Step 4: Loading Predefined Solvents ===")

    screen_config = config['screen']
    solvents = [s['groups'] for s in screen_config['solvents']]
    solvent_names = [s['name'] for s in screen_config['solvents']]

    if verbose:
        print(f"  Loaded {len(solvents)} predefined solvents")
        for name, groups in zip(solvent_names, solvents):
            groups_str = ", ".join([f"{k}:{v}" for k, v in groups.items()])
            print(f"    {name}: {groups_str}")

    return solvents, solvent_names


def screen_pure_properties(candidates, config, verbose=False):
    """Screen candidates based on pure property constraints."""
    if verbose:
        print("\n=== Step 5: Pure Property Screening ===")

    constraints = config['screening']['pure']
    prop_calc = PropertyCalculator()

    # Create compound dict
    solvent_compounds = {f"solvent_{i+1}": cand for i, cand in enumerate(candidates)}

    # Calculate properties
    prop_list = ['tc', 'pc', 'tb', 'tm', 'ac', 'hv']
    solvent_props_raw = prop_calc.calculate(solvent_compounds, prop_list)

    # Filter None values
    solvent_props = {name: props for name, props in solvent_props_raw.items() if props is not None}

    # Screen based on constraints
    screened = {}
    for name, props in solvent_props.items():
        # Sanity: Tb < Tc
        if props.get('tb') and props.get('tc') and props['tb'] >= props['tc']:
            continue

        # Check constraints
        passes = True
        for prop, (min_val, max_val) in constraints.items():
            if prop in props and props[prop] is not None:
                if not (min_val <= props[prop] <= max_val):
                    passes = False
                    break
            else:
                passes = False
                break

        if passes:
            idx = int(name.split('_')[1]) - 1
            screened[name] = {
                'groups': candidates[idx],
                'properties': props
            }

    if verbose:
        print(f"  {len(screened)}/{len(candidates)} candidates passed")

    return screened


def check_solvent_unifac(screened_solvents, mixture, verbose=False):
    """Check UNIFAC compatibility for screened solvents."""
    if verbose:
        print("\n=== Step 6: UNIFAC Compatibility Check ===")

    compatible = {}
    solvent_candidates = [data['groups'] for data in screened_solvents.values()]
    solvent_names = list(screened_solvents.keys())

    result = check_unifac_parameters(mixture=mixture, solvent_candidates=solvent_candidates)

    for i, name in enumerate(solvent_names):
        if i in result['compatible_solvents']:
            compatible[name] = screened_solvents[name]

    if verbose:
        print(f"  {len(compatible)}/{len(screened_solvents)} solvents compatible")

    return compatible


def calculate_solvent_antoine(compatible_solvents, verbose=False):
    """Calculate Antoine parameters for compatible solvents."""
    if verbose:
        print("\n=== Step 7: Solvent Antoine Parameters ===")

    antoine_calc = AntoineCalculator()
    antoine_params = {}

    for name, data in compatible_solvents.items():
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

    if verbose:
        print(f"  Calculated Antoine for {len(antoine_params)} solvents")

    return antoine_params


def run_vle_calculations(compatible_solvents, solvent_antoine, mixture, binary_antoine,
                        binary_props, n_data=50, verbose=False):
    """Perform VLE calculations for all compatible solvents."""
    if verbose:
        print("\n=== Step 8: VLE Calculations ===")

    rk_data, qk_data, interaction_data = load_unifac_data()
    P = 1.01325  # bar

    tb_A = binary_props['A']['tb']
    tb_B = binary_props['B']['tb']

    vle_results = {}
    successful = []

    for solvent_name, solvent_data in compatible_solvents.items():
        if solvent_name not in solvent_antoine:
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
                continue

            status = str(vle_solution.solver.status)
            cond = str(vle_solution.solver.termination_condition)

            if status != 'ok' or cond != 'optimal':
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
            if verbose:
                print(f"  Failed: {solvent_name}")
            continue

    if verbose:
        print(f"  {len(successful)} VLE calculations succeeded")

    return vle_results, successful


def screen_mixture_properties(vle_results, successful, config, verbose=False):
    """Screen solvents based on mixture property constraints."""
    if verbose:
        print("\n=== Step 9: Mixture Property Screening ===")

    mixture_config = config['screening']['mixture']
    passed = {}

    for name in successful:
        results = vle_results[name]

        # Check constraints
        az_AE_ok = results['az_AE'] > 0 if mixture_config.get('no_azeotrope_AE', False) else True
        az_BE_ok = results['az_BE'] > 0 if mixture_config.get('no_azeotrope_BE', False) else True

        G_mix_AE_ok = np.all(results['G_mix_AE'][1:-1] <= 0) if mixture_config.get('miscibility', False) else True
        G_mix_BE_ok = np.all(results['G_mix_BE'][1:-1] <= 0) if mixture_config.get('miscibility', False) else True
        G_mix_ok = G_mix_AE_ok and G_mix_BE_ok

        G2_mix_AE_ok = np.all(results['G2_mix_AE'][1:-1] >= 0) if mixture_config.get('miscibility', False) else True
        G2_mix_BE_ok = np.all(results['G2_mix_BE'][1:-1] >= 0) if mixture_config.get('miscibility', False) else True
        G2_mix_ok = G2_mix_AE_ok and G2_mix_BE_ok

        S_min = mixture_config.get('selectivity_min', 0)
        S_ok = results['S'] >= S_min

        if az_AE_ok and az_BE_ok and G_mix_ok and G2_mix_ok and S_ok:
            passed[name] = results

    if verbose:
        print(f"  {len(passed)}/{len(successful)} solvents passed")

    return passed


def run_iss_calculations(screened_mixture, compatible_solvents, mixture, binary_antoine,
                        solvent_antoine, binary_props, config, verbose=False):
    """Run ISS methodology on screened solvents."""
    if verbose:
        print("\n=== Step 10: ISS Calculations ===")

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

    for solvent_name in screened_mixture.keys():
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
                continue

            status = str(results.solver.status)
            cond = str(results.solver.termination_condition)

            if status != 'ok' or cond != 'optimal':
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
            continue

    if verbose:
        print(f"  {len(iss_results)} ISS calculations succeeded")

    return iss_results


def save_results(iss_results, screened_mixture, compatible_solvents, output_dir, config):
    """Save results to files."""
    import json
    import csv

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
        print(f"  Saved: {csv_path}")

    # Save JSON
    if 'json' in formats:
        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results_list, f, indent=2)
        print(f"  Saved: {json_path}")


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
        if verbose:
            print("=" * 60)
            print("CAMD-ISS: Solvent Design & Screening")
            print("=" * 60)
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Mode: {config['mode']}")
            print(f"Config: {args.config}")
            print(f"Log file: {log_file}")

        # Setup
        mixture = get_mixture(config)

        # Binary mixture calculations
        binary_props = calculate_pure_properties(mixture, config, verbose)
        binary_antoine = calculate_antoine_parameters(mixture, binary_props, verbose)
        check_binary_unifac(mixture, verbose)

        # Get candidates based on mode
        if config['mode'] == 'design':
            candidates = generate_solvent_candidates(config, mixture, verbose)
            solvent_names = None
        elif config['mode'] == 'screen':
            candidates, solvent_names = get_predefined_solvents(config, verbose)
        else:
            raise ValueError(f"Invalid mode: {config['mode']}. Must be 'design' or 'screen'")

        if not candidates:
            print("No candidates to process")
            return

        # Screening pipeline
        screened_pure = screen_pure_properties(candidates, config, verbose)
        compatible_solvents = check_solvent_unifac(screened_pure, mixture, verbose)
        solvent_antoine = calculate_solvent_antoine(compatible_solvents, verbose)

        vle_results, successful = run_vle_calculations(
            compatible_solvents, solvent_antoine, mixture, binary_antoine, binary_props,
            n_data=50, verbose=verbose
        )

        screened_mixture = screen_mixture_properties(vle_results, successful, config, verbose)

        iss_results = run_iss_calculations(
            screened_mixture, compatible_solvents, mixture, binary_antoine,
            solvent_antoine, binary_props, config, verbose
        )

        # Save results
        if verbose:
            print("\n" + "="*60)
            print("FINAL RESULTS")
            print("="*60)

        if iss_results:
            save_results(iss_results, screened_mixture, compatible_solvents, output_dir, config)

            if verbose:
                print(f"\nTop 5 solvents (ranked by ED):")
                sorted_solvents = sorted(iss_results.items(), key=lambda x: x[1]['ED'])
                for i, (name, data) in enumerate(sorted_solvents[:5]):
                    groups_str = ", ".join([f"{k}:{v}" for k, v in data['groups'].items()])
                    print(f"  {i+1}. {name}: {groups_str}")
                    print(f"     ED={data['ED']:.6f}, R={data['R']:.6f}")
                print(f"\nTotal solvents found: {len(iss_results)}")
        else:
            print("No solvents passed all screening steps")

        if verbose:
            print("\n" + "=" * 60)
            print("CAMD-ISS Complete")
            print(f"Output directory: {output_dir}")
            print("=" * 60)

    finally:
        # Restore stdout and close log file
        sys.stdout = logger.terminal
        logger.close()


if __name__ == '__main__':
    main()
