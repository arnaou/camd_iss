"""
Shared screening pipeline for CAMD-ISS.

Contains all step functions used by iss.py for both design and screen
modes. Each step is a standalone function called from the entry point.

Steps:
    1. validate_mixture           - Check mixture structural validity
    2. check_mixture_parameters   - Check UNIFAC parameter availability
    3. calculate_mixture_properties - Calculate pure component properties
    4. override_mixture_properties - Apply user-provided overrides
    5. calculate_mixture_antoine  - Calculate Antoine parameters for mixture
    7. check_candidate_parameters - Check UNIFAC params for candidates
    8. calculate_candidate_properties - Calculate pure props for candidates
    9. screen_pure_properties     - Screen by pure property constraints
   10. calculate_candidate_antoine - Calculate Antoine params for candidates
   11. screen_miscibility         - Miscibility screening over T range
   12. perform_vle                - Bubble point VLE calculations
   13. screen_mixture_properties  - Screen by mixture property constraints
   14. perform_iss                - ISS calculations
"""

import sys
import os
import time
import yaml
import json
import csv
import numpy as np
import pyomo.environ as pyo
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from .molecules import StructuralValidator, generate_molecules
from .pure import PropertyCalculator, AntoineCalculator
from .data import load_unifac_data, check_unifac_parameters
from .mixture.unifac import pyo_vle
from .mixture.unifac.pyomo_iss import pyo_iss
from .mixture.unifac import unifac


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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


@contextmanager
def step(step_num, title, verbose):
    """Print step header/footer (with elapsed time) around the wrapped block."""
    if verbose:
        print("\n" + "="*60)
        print(f"STEP {step_num}: {title}")
        print("="*60)
    t0 = time.time()
    yield
    if verbose:
        print(f"  [{time.time() - t0:.1f}s]")


def _fmt_groups(groups):
    """Format a UNIFAC groups dict as 'GROUP:count, ...'."""
    return ", ".join(f"{k}:{v}" for k, v in groups.items())


# ---------------------------------------------------------------------------
# Step 1: Validate binary mixture
# ---------------------------------------------------------------------------

def validate_mixture(config, verbose=False):
    """Check defined mixture for structural validity."""
    with step(1, "Validate Binary Mixture", verbose):
        mixture = {
            'A': config['mixture']['component_a']['groups'],
            'B': config['mixture']['component_b']['groups']
        }

        validator = StructuralValidator()

        is_valid_A = validator.is_valid(mixture['A'])
        is_valid_B = validator.is_valid(mixture['B'])

        if verbose:
            groups_A = _fmt_groups(mixture['A'])
            groups_B = _fmt_groups(mixture['B'])
            print(f"Component A: {groups_A}")
            print(f"  {'✓ Valid' if is_valid_A else '✗ Invalid'}")
            print(f"Component B: {groups_B}")
            print(f"  {'✓ Valid' if is_valid_B else '✗ Invalid'}")
            print(f"\nSummary: {'Both components valid' if is_valid_A and is_valid_B else 'Validation failed'}")

        if not (is_valid_A and is_valid_B):
            raise ValueError("Binary mixture validation failed")

    return mixture


# ---------------------------------------------------------------------------
# Step 2: Check mixture UNIFAC parameters
# ---------------------------------------------------------------------------

def check_mixture_parameters(mixture, verbose=False):
    """Check UNIFAC parameters for binary mixture."""
    with step(2, "Check Binary Mixture Parameters", verbose):
        result = check_unifac_parameters(mixture=mixture)

        if verbose:
            if result['compatible']:
                print("✓ All UNIFAC parameters available")
            else:
                print("✗ Missing UNIFAC parameters")
            print(f"\nSummary: {'Compatible' if result['compatible'] else 'Missing parameters'}")

    return result['compatible']


# ---------------------------------------------------------------------------
# Step 3: Calculate pure component properties (mixture)
# ---------------------------------------------------------------------------

def calculate_mixture_properties(mixture, verbose=False):
    """Calculate pure component properties for binary mixture."""
    with step(3, "Calculate Pure Component Properties (Mixture)", verbose):
        prop_list = ['tc', 'pc', 'tb', 'tm', 'ac', 'hv']
        pure_calc = PropertyCalculator()
        pure_props = pure_calc.calculate(mixture, prop_list)

        if verbose:
            for comp, props in pure_props.items():
                groups_str = _fmt_groups(mixture[comp])
                print(f"\n{comp} ({groups_str}):")
                print(f"  Tc: {props['tc']:.2f} K")
                print(f"  Pc: {props['pc']:.2f} bar")
                print(f"  Tb: {props['tb']:.2f} K")
                print(f"  Tm: {props['tm']:.2f} K")
                print(f"  Ac: {props['ac']:.4f}")
                print(f"  Hv: {props['hv']:.2f} kJ/mol")
            print(f"\nSummary: Calculated {len(prop_list)} properties for {len(pure_props)} components")

    return pure_props


# ---------------------------------------------------------------------------
# Step 4: Override mixture properties
# ---------------------------------------------------------------------------

def override_mixture_properties(pure_props, config, verbose=False):
    """Apply user-provided property overrides."""
    with step(4, "Override Mixture Properties (if provided)", verbose):
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


# ---------------------------------------------------------------------------
# Step 5: Calculate Antoine parameters (mixture)
# ---------------------------------------------------------------------------

def calculate_mixture_antoine(mixture, pure_props, verbose=False):
    """Calculate Antoine parameters for mixture components."""
    with step(5, "Calculate Antoine Parameters (Mixture)", verbose):
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


# ---------------------------------------------------------------------------
# Step 6: Load and validate predefined candidates
# ---------------------------------------------------------------------------

def load_and_validate_candidates(config, config_key='screen', list_key='entrainers', verbose=False):
    """
    Load and validate predefined candidates from config.

    Args:
        config: Parsed YAML config dict.
        config_key: Top-level key in config ('screen').
        list_key: Key within config_key for the candidate list
                  ('entrainers' or 'solvents').
    """
    with step(6, "Load and Validate Candidates", verbose):
        screen_config = config[config_key]
        candidates = []
        candidate_names = []
        validator = StructuralValidator()

        valid_count = 0
        invalid_count = 0

        if verbose:
            print("Validating predefined candidates:")

        for entry in screen_config[list_key]:
            name = entry['name']
            groups = entry['groups']
            is_valid = validator.is_valid(groups)

            if is_valid:
                candidates.append(groups)
                candidate_names.append(name)
                valid_count += 1
                if verbose:
                    groups_str = _fmt_groups(groups)
                    print(f"  ✓ {name}: {groups_str}")
            else:
                invalid_count += 1
                if verbose:
                    groups_str = _fmt_groups(groups)
                    print(f"  ✗ {name}: {groups_str} - Invalid")

        if verbose:
            print(f"\nSummary: {valid_count} valid, {invalid_count} invalid")

    return candidates, candidate_names


def generate_solvent_candidates(config, mixture, verbose=False):
    """Generate solvent candidates using MILP (design mode's step 6)."""
    with step(6, "Generate Solvent Candidates", verbose):
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
            verbose=False,
            solver=design_config.get('solver', 'gurobi'),
        )

        if verbose:
            print(f"\nSummary: Generated {len(candidates)} candidates")

    return candidates


# ---------------------------------------------------------------------------
# Step 7: Check UNIFAC parameters for candidates
# ---------------------------------------------------------------------------

def check_candidate_parameters(candidates, mixture, verbose=False):
    """Check UNIFAC parameters for candidates."""
    with step(7, "Check Candidate Parameters (UNIFAC)", verbose):
        result = check_unifac_parameters(mixture=mixture, solvent_candidates=candidates)

        compatible_indices = result['compatible_solvents']
        compatible = [candidates[i] for i in compatible_indices]

        if verbose:
            print(f"Input: {len(candidates)}")
            print(f"Passed: {len(compatible)}")
            print(f"Rejected: {len(candidates) - len(compatible)}")
            print(f"\nSummary: {len(compatible)}/{len(candidates)} have complete UNIFAC parameters")

    return compatible, compatible_indices


# ---------------------------------------------------------------------------
# Step 8: Calculate pure component properties (candidates)
# ---------------------------------------------------------------------------

def calculate_candidate_properties(candidates, config, prefix='candidate', verbose=False):
    """
    Calculate pure component properties for candidates.

    Args:
        candidates: List of group dicts.
        config: Parsed YAML config (unused currently, kept for interface consistency).
        prefix: Name prefix for internal labelling ('candidate', 'solvent', 'entrainer').
    """
    with step(8, "Calculate Pure Component Properties (Candidates)", verbose):
        prop_calc = PropertyCalculator()
        compounds = {f"{prefix}_{i+1}": sol for i, sol in enumerate(candidates)}

        prop_list = ['tc', 'pc', 'tb', 'tm', 'ac', 'hv']
        props_raw = prop_calc.calculate(compounds, prop_list)

        # Filter None values (missing groups)
        props = {name: p for name, p in props_raw.items() if p is not None}

        if verbose:
            dropped = len(props_raw) - len(props)
            print(f"Input: {len(candidates)}")
            if dropped > 0:
                print(f"Dropped: {dropped} (missing group contributions)")
            print(f"\nSummary: {len(props)}/{len(candidates)} properties calculated")

    return props


# ---------------------------------------------------------------------------
# Step 9: Screen based on pure property constraints
# ---------------------------------------------------------------------------

def screen_pure_properties(candidates, candidate_props, config, verbose=False):
    """Screen candidates based on pure property constraints."""
    with step(9, "Screen Pure Properties", verbose):
        constraints = config['screening'].get('pure') or {}
        screened = {}
        rejected = {}

        for name, props in candidate_props.items():
            idx = int(name.split('_')[1]) - 1
            groups = candidates[idx]
            groups_str = _fmt_groups(groups)

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
            print(f"Input: {len(candidate_props)}")
            print(f"Passed: {len(screened)}")
            print(f"Rejected: {len(rejected)}")

            if rejected:
                print(f"\nRejection reasons:")
                for name, data in rejected.items():
                    print(f"  {name} ({data['groups']}): {data['reason']}")

            print(f"\nSummary: {len(screened)}/{len(candidate_props)} passed pure property screening")

    return screened


# ---------------------------------------------------------------------------
# Step 10: Calculate Antoine parameters (candidates)
# ---------------------------------------------------------------------------

def calculate_candidate_antoine(screened_candidates, verbose=False):
    """Calculate Antoine parameters for screened candidates."""
    with step(10, "Calculate Antoine Parameters (Candidates)", verbose):
        antoine_calc = AntoineCalculator()
        antoine_params = {}
        failed = 0

        for name, data in screened_candidates.items():
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
            print(f"Input: {len(screened_candidates)}")
            print(f"Successful: {len(antoine_params)}")
            print(f"Failed: {failed}")
            print(f"\nSummary: {len(antoine_params)}/{len(screened_candidates)} calculated")

    return antoine_params


# ---------------------------------------------------------------------------
# Step 11: Miscibility screening over temperature range
# ---------------------------------------------------------------------------

def screen_miscibility(screened_candidates, mixture, binary_props,
                       rk_data, qk_data, interaction_data, verbose=False):
    """
    Screen candidates for miscibility over temperature ranges for A-E and B-E pairs.

    Uses UNIFAC to evaluate activity coefficients and check:
    - Gibbs mixing energy (G_mix) for phase stability
    - Second derivative of G_mix for thermodynamic consistency

    Tests two binary pairs:
    - A-E: from Tb_A to Tb_E
    - B-E: from Tb_B to Tb_E

    Removes candidates that show phase splitting or instability at any temperature.
    """
    with step(11, "Screen Miscibility (Temperature Range)", verbose):
        n_data = 100
        T_step = 1.0

        # Composition grid — fixed for all candidates and temperatures, built once
        x1 = np.linspace(0.0, 1.0, n_data)
        x_binary = np.column_stack([x1, 1 - x1])  # (n_data, 2)

        passed_miscibility = {}
        failed_miscibility = {}

        tb_A = binary_props['A']['tb']
        tb_B = binary_props['B']['tb']

        for cand_name, cand_data in screened_candidates.items():
            groups_str = _fmt_groups(cand_data['groups'])

            try:
                tb_E = cand_data['properties']['tb']

                T_range_AE = np.arange(min(tb_A, tb_E), max(tb_A, tb_E) + T_step, T_step)
                T_range_BE = np.arange(min(tb_B, tb_E), max(tb_B, tb_E) + T_step, T_step)

                def _check_miscibility_batched(comp_key, comp_groups, T_range):
                    """
                    Batch all temperature steps into a single unifac() call.
                    Returns (is_miscible, failing_temps).
                    """
                    N_T = len(T_range)
                    # Stack composition and temperature arrays across all T steps
                    x_batch = np.tile(x_binary, (N_T, 1))          # (N_T*n_data, 2)
                    T_batch = np.repeat(T_range, n_data)            # (N_T*n_data,)

                    res = unifac(
                        {comp_key: comp_groups, 'E': cand_data['groups']},
                        x_batch, T_batch, rk_data, qk_data, interaction_data
                    )
                    gamma_batch = res['gamma'].reshape(N_T, n_data, 2)   # (N_T, n_data, 2)
                    x_3d = x_binary[np.newaxis, :, :]                    # (1, n_data, 2)

                    # G_mix for each T slice: shape (N_T, n_data)
                    G_mix_all = (np.sum(x_3d * np.log(gamma_batch), axis=2) +
                                 np.sum(np.ma.log(x_3d).filled(0) * x_3d, axis=2))

                    failing_temps = []
                    for t_idx, T_test in enumerate(T_range):
                        G_mix = G_mix_all[t_idx]
                        if len(G_mix) > 2:
                            G2_mix = np.diff(np.diff(G_mix))
                            if not (np.all(G_mix[1:-1] <= 0) and np.all(G2_mix >= 0)):
                                failing_temps.append(T_test)

                    return len(failing_temps) == 0, failing_temps

                # Test A-E then B-E (B-E only if A-E passes)
                miscible_AE, failing_AE = _check_miscibility_batched(
                    'A', mixture['A'], T_range_AE
                )

                if not miscible_AE:
                    failing_temps = failing_AE
                    failing_pair = "A-E"
                else:
                    miscible_BE, failing_BE = _check_miscibility_batched(
                        'B', mixture['B'], T_range_BE
                    )
                    failing_temps = failing_BE
                    failing_pair = "B-E" if failing_BE else None

                if not failing_temps:
                    passed_miscibility[cand_name] = cand_data
                else:
                    if len(failing_temps) > 1:
                        temp_range_str = f"{failing_temps[0]:.1f}-{failing_temps[-1]:.1f}K"
                    else:
                        temp_range_str = f"{failing_temps[0]:.1f}K"
                    failed_miscibility[cand_name] = {
                        'groups': groups_str,
                        'reason': f"Phase instability {failing_pair} T={temp_range_str}"
                    }

            except Exception as e:
                failed_miscibility[cand_name] = {
                    'groups': groups_str,
                    'reason': f"Miscibility test error: {type(e).__name__}"
                }

        if verbose:
            print(f"Input: {len(screened_candidates)}")
            print(f"Passed: {len(passed_miscibility)}")
            print(f"Rejected: {len(failed_miscibility)}")

            if failed_miscibility:
                print(f"\nRejection reasons:")
                for name, data in failed_miscibility.items():
                    print(f"  {name} ({data['groups']}): {data['reason']}")

            print(f"\nSummary: {len(passed_miscibility)}/{len(screened_candidates)} passed miscibility screening")

    return passed_miscibility


# ---------------------------------------------------------------------------
# Step 12: Perform VLE
# ---------------------------------------------------------------------------

def perform_vle(screened_candidates, candidate_antoine, mixture,
                binary_antoine, binary_props,
                rk_data, qk_data, interaction_data, verbose=False):
    """Perform VLE calculations for candidates."""
    with step(12, "Perform VLE Calculations", verbose):
        P = 1.01325  # bar
        n_data = 50

        tb_A = binary_props['A']['tb']
        tb_B = binary_props['B']['tb']

        vle_results = {}
        successful = []
        failed = {}

        total = len([name for name in screened_candidates.keys() if name in candidate_antoine])

        for cand_name, cand_data in screened_candidates.items():
            groups_str = _fmt_groups(cand_data['groups'])

            if cand_name not in candidate_antoine:
                failed[cand_name] = {
                    'groups': groups_str,
                    'reason': "No Antoine parameters"
                }
                continue

            try:
                # Build ternary mixture
                ternary_mixture = dict(mixture)
                ternary_mixture[cand_name] = cand_data['groups']

                # Antoine parameters
                ternary_antoine = dict(binary_antoine)
                ternary_antoine[cand_name] = candidate_antoine[cand_name]

                tb_E = cand_data['properties']['tb']

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
                    failed[cand_name] = {
                        'groups': groups_str,
                        'reason': "VLE solver returned None"
                    }
                    continue

                status = str(vle_solution.solver.status)
                cond = str(vle_solution.solver.termination_condition)

                if status != 'ok' or cond != 'optimal':
                    failed[cand_name] = {
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

                vle_results[cand_name] = {
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
                successful.append(cand_name)

            except Exception as e:
                failed[cand_name] = {
                    'groups': groups_str,
                    'reason': f"VLE error: {type(e).__name__}"
                }

        if verbose:
            print(f"Input: {total}")
            print(f"Successful: {len(successful)}")
            print(f"Failed: {len(failed)}")

            if failed:
                print(f"\nFailure reasons:")
                for name, data in failed.items():
                    print(f"  {name} ({data['groups']}): {data['reason']}")

            print(f"\nSummary: {len(successful)}/{total} VLE calculations successful")

    return vle_results, successful


# ---------------------------------------------------------------------------
# Step 13: Screen based on mixture constraints
# ---------------------------------------------------------------------------

def screen_mixture_properties(vle_results, successful, screened_candidates, config, verbose=False):
    """Screen candidates based on mixture property constraints."""
    with step(13, "Screen Mixture Properties", verbose):
        mixture_config = config['screening']['mixture']
        passed = {}
        rejected = {}

        for name in successful:
            results = vle_results[name]
            groups_str = _fmt_groups(screened_candidates[name]['groups'])

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


# ---------------------------------------------------------------------------
# Step 14: Perform ISS calculation
# ---------------------------------------------------------------------------

def perform_iss(screened_mixture, compatible_candidates, mixture, binary_antoine,
                candidate_antoine, binary_props, config,
                rk_data, qk_data, interaction_data, verbose=False, detailed=False):
    """
    Perform ISS calculations.

    Args:
        detailed: If True, extract full gamma/K/x/y arrays per operating point
                  (used by screen mode for detailed JSON output).
    """
    with step(14, "Perform ISS Calculations", verbose):
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

        for cand_name in screened_mixture.keys():
            groups_str = _fmt_groups(compatible_candidates[cand_name]['groups'])

            try:
                # Build ternary
                ternary_mixture = dict(mixture)
                ternary_mixture[cand_name] = compatible_candidates[cand_name]['groups']

                ternary_antoine = dict(binary_antoine)
                ternary_antoine[cand_name] = candidate_antoine[cand_name]

                # ISS calculation
                results, model = pyo_iss(x0, T0, ternary_mixture, rk_data, qk_data,
                                        interaction_data, ternary_antoine, verbose=False)

                if results is None or model is None:
                    failed[cand_name] = {
                        'groups': groups_str,
                        'reason': "ISS solver returned None"
                    }
                    continue

                status = str(results.solver.status)
                cond = str(results.solver.termination_condition)

                if status != 'ok' or cond != 'optimal':
                    failed[cand_name] = {
                        'groups': groups_str,
                        'reason': f"ISS solver: {status}/{cond}"
                    }
                    continue

                ED = pyo.value(model.obj)
                R_value = pyo.value(model.R)

                entry = {
                    'ED': ED,
                    'R': R_value,
                    'groups': compatible_candidates[cand_name]['groups']
                }

                if detailed:
                    n_comp = len(model.i)
                    n_data = len(model.dp)

                    gamma_vals = np.zeros((n_data, n_comp))
                    K_vals = np.zeros((n_data, n_comp))
                    x_vals = np.zeros((n_data, n_comp))
                    y_vals = np.zeros((n_data, n_comp))
                    T_vals = np.zeros(n_data)

                    for j in range(n_data):
                        T_vals[j] = pyo.value(model.T[j])
                        for i in range(n_comp):
                            gamma_vals[j, i] = pyo.value(model.gamma[j, i])
                            K_vals[j, i] = pyo.value(model.K[j, i])
                            x_vals[j, i] = pyo.value(model.x[j, i])
                            y_vals[j, i] = pyo.value(model.y[j, i])

                    entry['gamma'] = gamma_vals
                    entry['K'] = K_vals
                    entry['x'] = x_vals
                    entry['y'] = y_vals
                    entry['T'] = T_vals
                    entry['compound_names'] = list(ternary_mixture.keys())

                iss_results[cand_name] = entry

            except Exception as e:
                failed[cand_name] = {
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


# ---------------------------------------------------------------------------
# Step 15: Resolve SMILES (experimental)
# ---------------------------------------------------------------------------

# Single-group molecules that are complete structures — the assembler
# cannot handle these since they have no free valences to connect.
_SINGLE_GROUP_SMILES = {
    'CH3OH': 'CO',
    'H2O':   'O',
    'DMSO':  'CS(C)=O',
    'DOH':   'OCCO',
    'THF':   'C1CCCO1',
}


def resolve_smiles(iss_results, compatible_candidates, verbose=False):
    """
    [Experimental] Resolve UNIFAC group compositions to SMILES strings.

    Uses src.unifac_reverse for multi-group molecules and a hardcoded
    fallback table for single-group whole-molecule groups (DOH, DMSO,
    CH3OH, H2O, THF).

    A UNIFAC group composition is a multiset of groups, not a unique
    connectivity — different constitutional isomers can share the exact
    same group counts (e.g. {CH3:1, CH2:1, CH:1, HCO:2} matches both
    2-methylbutanedial and ethylmalonaldehyde) and get identical estimated
    properties/ISS results, since UNIFAC is blind to connectivity. All
    isomers found are returned, joined by " | ", rather than picking one
    arbitrarily — the property estimates apply to all of them equally.

    Args:
        iss_results: Dict of ISS results (keys = candidate names).
        compatible_candidates: Dict with 'groups' per candidate.

    Returns:
        Dict mapping candidate name -> SMILES string, possibly containing
        multiple isomers joined by " | " (or None if resolution failed).
    """
    with step(15, "Resolve SMILES [experimental]", verbose):
        smiles_map = {}
        ok = 0
        failed = 0

        try:
            from .unifac_reverse import load_unifac_groups, reverse_engineer_by_name
            data_path = Path(__file__).parent.parent / "data" / "json" / "unifac_ly.json"
            db = load_unifac_groups(str(data_path))
            assembler_available = True
        except Exception as e:
            if verbose:
                print(f"Warning: SMILES resolver unavailable ({e})")
            assembler_available = False
            db = None

        for name in iss_results:
            groups = compatible_candidates[name]['groups']

            # Single-group whole-molecule fallback
            if len(groups) == 1:
                group_name = next(iter(groups))
                if group_name in _SINGLE_GROUP_SMILES:
                    smiles_map[name] = _SINGLE_GROUP_SMILES[group_name]
                    ok += 1
                    continue

            # Multi-group: use assembler
            if assembler_available:
                try:
                    smiles_list = reverse_engineer_by_name(groups, db)
                    if smiles_list:
                        smiles_map[name] = " | ".join(smiles_list)
                        ok += 1
                    else:
                        smiles_map[name] = None
                        failed += 1
                except Exception:
                    smiles_map[name] = None
                    failed += 1
            else:
                smiles_map[name] = None
                failed += 1

        if verbose:
            print(f"Resolved: {ok}/{len(iss_results)}")
            if failed:
                unresolved = [n for n, s in smiles_map.items() if s is None]
                print(f"Unresolved: {', '.join(unresolved)}")
            for name, smi in smiles_map.items():
                if smi:
                    groups_str = _fmt_groups(compatible_candidates[name]['groups'])
                    print(f"  {groups_str} -> {smi}")

    return smiles_map


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _build_mixture_info(binary_mixture, binary_props, binary_antoine):
    """Build mixture A/B info dict for output."""
    mixture_info = {}
    for comp in ['A', 'B']:
        comp_entry = {}
        if binary_mixture and comp in binary_mixture:
            comp_entry['groups'] = _fmt_groups(binary_mixture[comp])
        if binary_props and comp in binary_props:
            props = binary_props[comp]
            comp_entry['tc'] = props.get('tc')
            comp_entry['pc'] = props.get('pc')
            comp_entry['tb'] = props.get('tb')
            comp_entry['tm'] = props.get('tm')
            comp_entry['ac'] = props.get('ac')
            comp_entry['hv'] = props.get('hv')
        if binary_antoine and comp in binary_antoine:
            ant = binary_antoine[comp]
            comp_entry['ant_A'] = ant.get('A')
            comp_entry['ant_B'] = ant.get('B')
            comp_entry['ant_C'] = ant.get('C')
        mixture_info[comp] = comp_entry
    return mixture_info


def save_results(iss_results, screened_mixture, output_dir, config, verbose=False,
                 binary_props=None, binary_antoine=None, binary_mixture=None,
                 candidate_props_map=None, candidate_antoine=None,
                 smiles_map=None, label='solvent'):
    """
    Save results to CSV and JSON files.

    Args:
        label: Key name used in output ('solvent' for design, 'entrainer' for screen).
    """
    if verbose:
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)

    formats = config['output']['formats']

    # Prepare results list
    results_list = []
    for name, iss_data in iss_results.items():
        groups_str = _fmt_groups(iss_data['groups'])
        mixture_data = screened_mixture[name]

        entry = {
            label: name,
            'groups': groups_str,
            'ED': iss_data['ED'],
            'R': iss_data['R'],
            'S': mixture_data['S'],
            'C': mixture_data['C'],
            'SxC': mixture_data['SxC'],
            'alpha': mixture_data['alpha']
        }

        # Add pure component properties
        if candidate_props_map and name in candidate_props_map:
            props = candidate_props_map[name]['properties']
            entry['tc'] = props.get('tc')
            entry['pc'] = props.get('pc')
            entry['tb'] = props.get('tb')
            entry['tm'] = props.get('tm')
            entry['ac'] = props.get('ac')
            entry['hv'] = props.get('hv')

        # Add Antoine parameters
        if candidate_antoine and name in candidate_antoine:
            ant = candidate_antoine[name]
            entry['ant_A'] = ant.get('A')
            entry['ant_B'] = ant.get('B')
            entry['ant_C'] = ant.get('C')

        # Add SMILES (experimental)
        if smiles_map and name in smiles_map:
            entry['smiles'] = smiles_map[name]

        results_list.append(entry)

    # Sort by ED
    results_list.sort(key=lambda x: x['ED'])

    mixture_info = _build_mixture_info(binary_mixture, binary_props, binary_antoine)

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
        output = {
            'mixture': mixture_info,
            'results': results_list
        }
        with open(json_path, 'w') as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"✓ Saved: {json_path}")

    if verbose:
        print(f"\nSummary: Saved {len(results_list)} results in {len(formats)} format(s)")

    return results_list, mixture_info


def save_detailed_results(iss_results, screened_mixture, output_dir, mixture_info,
                          candidate_props_map=None, candidate_antoine=None,
                          verbose=False):
    """Save detailed JSON with full ISS operating point data (screen mode)."""
    detailed_results = []
    for name, iss_data in iss_results.items():
        groups_str = _fmt_groups(iss_data['groups'])
        mixture_data = screened_mixture[name]

        # Build operating points data with labeled components (A, B, E)
        operating_points = []
        n_comp = iss_data['x'].shape[1] if iss_data['x'].ndim > 1 else 1
        comp_labels = ['A', 'B', 'E'][:n_comp]

        for j in range(len(iss_data['T'])):
            op_point = {
                'T': float(iss_data['T'][j]),
                'x': {},
                'y': {},
                'gamma': {},
                'K': {}
            }

            for i, lbl in enumerate(comp_labels):
                op_point['x'][lbl] = float(iss_data['x'][j, i]) if n_comp > 1 else float(iss_data['x'][j])
                op_point['y'][lbl] = float(iss_data['y'][j, i]) if n_comp > 1 else float(iss_data['y'][j])
                op_point['gamma'][lbl] = float(iss_data['gamma'][j, i]) if n_comp > 1 else float(iss_data['gamma'][j])
                op_point['K'][lbl] = float(iss_data['K'][j, i]) if n_comp > 1 else float(iss_data['K'][j])

            operating_points.append(op_point)

        detail_entry = {
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
        }

        # Add pure component properties and Antoine for the candidate
        if candidate_props_map and name in candidate_props_map:
            detail_entry['pure_props'] = candidate_props_map[name]['properties']
        if candidate_antoine and name in candidate_antoine:
            detail_entry['antoine'] = candidate_antoine[name]

        detailed_results.append(detail_entry)

    # Sort by ED
    detailed_results.sort(key=lambda x: x['ED'])

    detailed_json_path = output_dir / 'results_detailed.json'
    with open(detailed_json_path, 'w') as f:
        json.dump({'mixture': mixture_info, 'results': detailed_results}, f, indent=2)
    if verbose:
        print(f"✓ Saved: {detailed_json_path}")


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(results_list, label='candidate', top_n=None):
    """
    Print a single-line-per-entry summary table of screening results.

    Columns: rank, groups, ED, R, S, C, SxC, alpha,
             Tb, Tc, Pc, Tm, ac, Hv, Antoine A/B/C

    Args:
        results_list: List of result dicts from save_results (sorted by ED).
        label: Header label used in title ('solvent', 'entrainer', 'candidate').
        top_n: If set, only print the top N entries.
    """
    if not results_list:
        return

    entries = results_list[:top_n] if top_n else results_list
    count = len(results_list)

    def _fv(v, w, d):
        """Format a float value with given width and decimals, or 'n/a'."""
        return f"{v:{w}.{d}f}" if v is not None else f"{'n/a':>{w}}"

    # Column widths
    G = 30   # groups
    W = 140  # total separator line width

    has_smiles = any(e.get('smiles') for e in entries)
    S = 30  # smiles column width

    header = (
        f"{'#':>3}  {'Groups':<{G}}  "
        + (f"{'SMILES':<{S}}  " if has_smiles else "")
        + f"{'ED':>8}  {'R':>6}  {'S':>6}  {'C':>6}  {'SxC':>6}  {'alpha':>6}  "
        f"{'Tb(K)':>6}  {'Tc(K)':>6}  {'Pc(b)':>5}  {'Tm(K)':>6}  "
        f"{'ac':>6}  {'Hv(kJ)':>6}  {'Ant-A':>7}  {'Ant-B':>8}  {'Ant-C':>7}"
    )

    title = f"TOP {label.upper()}S" if top_n is None else f"TOP {top_n} {label.upper()}S"
    print("\n" + "=" * W)
    print(f"{title} (sorted by ED)".center(W))
    print("=" * W)
    print(header)
    print("-" * W)

    for i, entry in enumerate(entries, 1):
        groups = entry.get('groups', 'n/a')
        if len(groups) > G:
            groups = groups[:G - 3] + "..."

        smi = entry.get('smiles') or ''
        if has_smiles and len(smi) > S:
            smi = smi[:S - 3] + "..."

        line = (
            f"{i:>3}  {groups:<{G}}  "
            + (f"{smi:<{S}}  " if has_smiles else "")
            + f"{_fv(entry.get('ED'),    8, 4)}  "
            f"{_fv(entry.get('R'),     6, 3)}  "
            f"{_fv(entry.get('S'),     6, 3)}  "
            f"{_fv(entry.get('C'),     6, 3)}  "
            f"{_fv(entry.get('SxC'),   6, 3)}  "
            f"{_fv(entry.get('alpha'), 6, 3)}  "
            f"{_fv(entry.get('tb'),    6, 1)}  "
            f"{_fv(entry.get('tc'),    6, 1)}  "
            f"{_fv(entry.get('pc'),    5, 2)}  "
            f"{_fv(entry.get('tm'),    6, 1)}  "
            f"{_fv(entry.get('ac'),    6, 4)}  "
            f"{_fv(entry.get('hv'),    6, 2)}  "
            f"{_fv(entry.get('ant_A'), 7, 4)}  "
            f"{_fv(entry.get('ant_B'), 8, 3)}  "
            f"{_fv(entry.get('ant_C'), 7, 3)}"
        )
        print(line)

    print("=" * W)
    print(f"Total: {count} {label}(s) evaluated")
    print("=" * W)


# ---------------------------------------------------------------------------
# Full screening pipeline (steps 7-14)
# ---------------------------------------------------------------------------

def run_screening_pipeline(candidates, mixture, binary_props, binary_antoine,
                           config, verbose=False, prefix='candidate',
                           detailed_iss=False):
    """
    Run the full screening pipeline from step 7 through step 14.

    Returns a dict with all intermediate results, or None if the pipeline
    is exhausted at any step.

    Args:
        candidates: List of group dicts (from design or screen mode).
        prefix: Naming prefix for internal labels.
        detailed_iss: Whether to extract full ISS data (for screen mode).
    """
    # Step 7: Check UNIFAC parameters
    compatible, compatible_indices = check_candidate_parameters(candidates, mixture, verbose)
    if not compatible:
        print("\n⚠ No compatible candidates found")
        return None

    # Step 8: Calculate pure component properties
    candidate_props = calculate_candidate_properties(compatible, config, prefix=prefix, verbose=verbose)

    # Step 9: Screen pure properties
    screened_pure = screen_pure_properties(compatible, candidate_props, config, verbose)
    if not screened_pure:
        print("\n⚠ No candidates passed pure property screening")
        return None

    # Step 10: Calculate Antoine parameters
    candidate_antoine = calculate_candidate_antoine(screened_pure, verbose)
    if not candidate_antoine:
        print("\n⚠ No candidates have valid Antoine parameters")
        return None

    # Load UNIFAC data once — shared by steps 11, 12, 14
    rk_data, qk_data, interaction_data = load_unifac_data()

    # Step 11: Miscibility screening
    screened_miscibility = screen_miscibility(
        screened_pure, mixture, binary_props,
        rk_data, qk_data, interaction_data, verbose
    )
    if not screened_miscibility:
        print("\n⚠ No candidates passed miscibility screening")
        return None

    # Filter Antoine to miscibility-passing candidates
    antoine_filtered = {k: v for k, v in candidate_antoine.items()
                        if k in screened_miscibility}

    # Step 12: VLE
    vle_results, successful = perform_vle(
        screened_miscibility, antoine_filtered, mixture,
        binary_antoine, binary_props, rk_data, qk_data, interaction_data, verbose
    )
    if not successful:
        print("\n⚠ No successful VLE calculations")
        return None

    # Step 13: Mixture property screening
    screened_mixture = screen_mixture_properties(
        vle_results, successful, screened_miscibility, config, verbose
    )
    if not screened_mixture:
        print("\n⚠ No candidates passed mixture property screening")
        return None

    # Step 14: ISS calculation
    iss_results = perform_iss(
        screened_mixture, screened_miscibility, mixture, binary_antoine,
        antoine_filtered, binary_props, config,
        rk_data, qk_data, interaction_data, verbose, detailed=detailed_iss
    )

    # Step 15: Resolve SMILES [experimental]
    smiles_map = resolve_smiles(iss_results, screened_miscibility, verbose)

    return {
        'iss_results': iss_results,
        'screened_mixture': screened_mixture,
        'screened_miscibility': screened_miscibility,
        'antoine_filtered': antoine_filtered,
        'smiles_map': smiles_map,
    }
