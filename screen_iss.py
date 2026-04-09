#!/usr/bin/env python3
"""
Screen-ISS: Screen a predefined set of entrainers for ISS performance.

Config-driven script for screening and evaluating a limited set of entrainers.
No molecule generation — all entrainers are predefined in the config file.

Workflow Steps:
1. Validate binary mixture
2. Check UNIFAC parameters (mixture)
3. Calculate pure component properties (mixture)
4. Override properties if provided by user
5. Calculate Antoine parameters (mixture)
6. Load and validate predefined entrainer candidates
7-14. Screening pipeline (UNIFAC params, pure props, Antoine, miscibility, VLE, ISS)
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import (
    TeeLogger, load_config, setup_output_dir,
    validate_mixture, check_mixture_parameters,
    calculate_mixture_properties, override_mixture_properties,
    calculate_mixture_antoine, load_and_validate_candidates,
    run_screening_pipeline, save_results, save_detailed_results,
    print_results_table, _build_mixture_info,
)


def main():
    parser = argparse.ArgumentParser(
        description='Screen-ISS: Screen predefined entrainers for ISS performance'
    )
    parser.add_argument('config', help='Path to YAML configuration file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    config = load_config(args.config)
    verbose = args.verbose or config['output'].get('verbose', False)

    output_dir = setup_output_dir(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"screen_run_{timestamp}.log"
    logger = TeeLogger(log_file)
    sys.stdout = logger

    try:
        print("="*60)
        print("Screen-ISS: Entrainer Screening")
        print("="*60)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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

        # Step 6: Load predefined entrainers
        entrainers, entrainer_names = load_and_validate_candidates(
            config, config_key='screen', list_key='entrainers', verbose=verbose
        )

        if not entrainers:
            print("\n⚠ No entrainers to process")
            return

        # Steps 7-14: Screening pipeline (with detailed ISS extraction)
        pipeline_result = run_screening_pipeline(
            entrainers, mixture, binary_props, binary_antoine,
            config, verbose, prefix='entrainer', detailed_iss=True
        )

        if pipeline_result is None or not pipeline_result['iss_results']:
            print("\n⚠ No entrainers completed the screening pipeline")
            return

        iss_results = pipeline_result['iss_results']
        screened_mixture = pipeline_result['screened_mixture']

        # Save summary results (CSV + JSON)
        results_list, mixture_info = save_results(
            iss_results, screened_mixture, output_dir, config, verbose,
            binary_props=binary_props, binary_antoine=binary_antoine,
            binary_mixture=mixture,
            candidate_props_map=pipeline_result['screened_miscibility'],
            candidate_antoine=pipeline_result['antoine_filtered'],
            label='entrainer'
        )

        # Save detailed JSON with full ISS operating point data
        save_detailed_results(
            iss_results, screened_mixture, output_dir, mixture_info,
            candidate_props_map=pipeline_result['screened_miscibility'],
            candidate_antoine=pipeline_result['antoine_filtered'],
            verbose=verbose
        )

        print_results_table(results_list, label='entrainer')

        print("\n" + "="*60)
        print("Screen-ISS COMPLETE")
        print(f"Output: {output_dir}")
        print("="*60)

    finally:
        sys.stdout = logger.terminal
        logger.close()


if __name__ == '__main__':
    main()
