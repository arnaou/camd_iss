#!/usr/bin/env python3
"""
CAMD-ISS: Computer-Aided Molecular Design for Extractive Distillation

Config-driven script for solvent design and screening with early miscibility screening.

Workflow Steps:
1. Validate binary mixture
2. Check UNIFAC parameters (mixture)
3. Calculate pure component properties (mixture)
4. Override properties if provided by user
5. Calculate Antoine parameters (mixture)
6. Generate solvent candidates (MILP) or load predefined solvents
7-14. Screening pipeline (UNIFAC params, pure props, Antoine, miscibility, VLE, ISS)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.molecules import generate_molecules
from src.pipeline import (
    TeeLogger, load_config, setup_output_dir, step,
    validate_mixture, check_mixture_parameters,
    calculate_mixture_properties, override_mixture_properties,
    calculate_mixture_antoine, load_and_validate_candidates,
    run_screening_pipeline, save_results, print_results_table,
)


def generate_solvent_candidates(config, mixture, verbose=False):
    """Generate solvent candidates using MILP."""
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
            verbose=False
        )

        if verbose:
            print(f"\nSummary: Generated {len(candidates)} candidates")

    return candidates


def main():
    parser = argparse.ArgumentParser(
        description='CAMD-ISS: Config-driven solvent design/screening with early miscibility screening'
    )
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    config = load_config(args.config)
    verbose = args.verbose or config['output'].get('verbose', False)

    output_dir = setup_output_dir(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"camd_run_{timestamp}.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger

    try:
        print("="*60)
        print("CAMD-ISS: Computer-Aided Molecular Design")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Mode: {config['mode'].upper()}")
        print(f"Config: {args.config}")
        print(f"Log file: {log_file}")

        # Steps 1-5: Binary mixture workflow
        mixture = validate_mixture(config, verbose)
        check_mixture_parameters(mixture, verbose)
        binary_props = calculate_mixture_properties(mixture, verbose)
        binary_props = override_mixture_properties(binary_props, config, verbose)

        tb_A = binary_props['A']['tb']
        tb_B = binary_props['B']['tb']
        if tb_A >= tb_B:
            raise ValueError(f"Invalid boiling point order: Tb_A ({tb_A:.2f}K) must be < Tb_B ({tb_B:.2f}K)")
        if verbose:
            print(f"\n✓ Boiling point validation: Tb_A ({tb_A:.2f}K) < Tb_B ({tb_B:.2f}K)")

        binary_antoine = calculate_mixture_antoine(mixture, binary_props, verbose)

        # Step 6: Get candidates (mode-dependent)
        if config['mode'] == 'screen':
            candidates, _ = load_and_validate_candidates(
                config, config_key='screen', list_key='solvents', verbose=verbose
            )
        elif config['mode'] == 'design':
            candidates = generate_solvent_candidates(config, mixture, verbose)
        else:
            raise ValueError(f"Invalid mode: {config['mode']}. Must be 'design' or 'screen'")

        if not candidates:
            print("\n⚠ No candidates to process")
            return

        # Steps 7-14: Screening pipeline
        pipeline_result = run_screening_pipeline(
            candidates, mixture, binary_props, binary_antoine,
            config, verbose, prefix='solvent'
        )

        if pipeline_result is None or not pipeline_result['iss_results']:
            print("\n⚠ No solvents completed the screening pipeline")
            return

        iss_results = pipeline_result['iss_results']
        screened_mixture = pipeline_result['screened_mixture']

        # Save results
        results_list, _ = save_results(
            iss_results, screened_mixture, output_dir, config, verbose,
            binary_props=binary_props, binary_antoine=binary_antoine,
            binary_mixture=mixture,
            candidate_props_map=pipeline_result['screened_miscibility'],
            candidate_antoine=pipeline_result['antoine_filtered'],
            smiles_map=pipeline_result.get('smiles_map'),
            label='solvent'
        )

        print_results_table(results_list, label='solvent')

        print("\n" + "="*60)
        print("CAMD-ISS COMPLETE")
        print(f"Output: {output_dir}")
        print("="*60)

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == '__main__':
    main()
