from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from compare_reconstruction import ValidationConfig, run_validation
from config import CONFIG
from generate_truth_data import TRUTH_CONFIG, generate_truth_data
from run_estimator import run_estimator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EKF reconstruction and validation for a selected scenario.",
    )
    parser.add_argument(
        "--scenario",
        choices=("no-leak", "leak"),
        default="leak",
        help="Scenario folder under truth_outputs to use.",
    )
    parser.add_argument(
        "--skip-truth-generation",
        action="store_true",
        help="Use existing truth files instead of regenerating them first.",
    )
    return parser


def run_scenario_validation(scenario: str, skip_truth_generation: bool = False) -> None:
    if not skip_truth_generation:
        generate_truth_data(TRUTH_CONFIG)

    truth_dir = TRUTH_CONFIG.output_dir / scenario
    inp_name = "base3.inp" if scenario == "no-leak" else TRUTH_CONFIG.leak_inp_filename
    scenario_inp_path = truth_dir / inp_name
    scenario_measurements_path = truth_dir / "truth_measurements.csv"
    reconstruction_dir = Path(f"outputs_{scenario}")
    validation_dir = Path(f"validation_outputs_{scenario}")

    estimator_config = replace(
        CONFIG,
        inp_path=scenario_inp_path,
        measurements_path=scenario_measurements_path,
        output_dir=reconstruction_dir,
        plots_dir=reconstruction_dir / "plots",
    )
    run_estimator(estimator_config)

    validation_config = ValidationConfig(
        truth_dir=truth_dir,
        reconstruction_dir=reconstruction_dir,
        output_dir=validation_dir,
    )
    run_validation(validation_config)


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_scenario_validation(args.scenario, skip_truth_generation=args.skip_truth_generation)
