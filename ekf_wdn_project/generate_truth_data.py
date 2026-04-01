from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

import pandas as pd

from config import CONFIG, EstimatorConfig
from load_model import extract_model_metadata, load_water_network

try:
    import wntr
except ImportError:  # pragma: no cover
    wntr = None


@dataclass(frozen=True)
class TruthGenerationConfig:
    estimator_config: EstimatorConfig = CONFIG
    output_dir: Path = Path("truth_outputs")
    no_leak_folder_name: str = "no-leak"
    leak_folder_name: str = "leak"
    leak_inp_filename: str = "leak.inp"


TRUTH_CONFIG = TruthGenerationConfig()


def generate_truth_data(config: TruthGenerationConfig = TRUTH_CONFIG) -> dict[str, dict[str, pd.DataFrame]]:
    if wntr is None:
        raise ImportError(
            "WNTR is required but not installed. Install dependencies with "
            "`py -3 -m pip install -r requirements.txt`."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)

    no_leak_dir = config.output_dir / config.no_leak_folder_name
    leak_dir = config.output_dir / config.leak_folder_name
    no_leak_dir.mkdir(parents=True, exist_ok=True)
    leak_dir.mkdir(parents=True, exist_ok=True)

    no_leak_inp = no_leak_dir / config.estimator_config.inp_path.name
    shutil.copyfile(config.estimator_config.inp_path, no_leak_inp)

    leak_inp = leak_dir / config.leak_inp_filename
    if not leak_inp.exists():
        raise FileNotFoundError(
            f"Leak scenario input file not found: {leak_inp}. "
            "Edit this file manually, then rerun the generator."
        )

    scenario_outputs = {
        config.no_leak_folder_name: _generate_truth_for_inp(no_leak_inp, no_leak_dir, config.estimator_config),
        config.leak_folder_name: _generate_truth_for_inp(leak_inp, leak_dir, config.estimator_config),
    }
    return scenario_outputs


def _generate_truth_for_inp(
    inp_path: Path,
    output_dir: Path,
    estimator_config: EstimatorConfig,
) -> dict[str, pd.DataFrame]:
    metadata = extract_model_metadata(estimator_config)
    wn = load_water_network(inp_path)
    wn.options.time.hydraulic_timestep = estimator_config.hydraulic_timestep_seconds
    wn.options.time.report_timestep = estimator_config.hydraulic_timestep_seconds

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    timestamp = pd.Series(results.node["head"].index, name="timestamp")

    truth_heads = _frame_with_timestamp(
        timestamp,
        results.node["head"].loc[:, list(metadata.all_report_nodes)],
    )
    truth_pressures = _frame_with_timestamp(
        timestamp,
        results.node["pressure"].loc[:, list(metadata.all_report_nodes)],
    )
    truth_demands = _frame_with_timestamp(
        timestamp,
        results.node["demand"].loc[:, list(metadata.demand_nodes)],
        rename_map={node: f"D{node}" for node in metadata.demand_nodes},
    )
    truth_pipe_flows = _frame_with_timestamp(
        timestamp,
        results.link["flowrate"].loc[:, list(metadata.report_flow_links)],
    )
    truth_measurements = pd.DataFrame(
        {
            "timestamp": timestamp,
            "P4": results.node["pressure"].loc[:, metadata.measured_pressure_node].to_numpy(),
            "Q1a": results.link["flowrate"].loc[:, metadata.measured_flow_links[0]].to_numpy(),
            "Q3a": results.link["flowrate"].loc[:, metadata.measured_flow_links[1]].to_numpy(),
        }
    )

    outputs = {
        "truth_node_heads.csv": truth_heads,
        "truth_node_pressures.csv": truth_pressures,
        "truth_demands.csv": truth_demands,
        "truth_pipe_flows.csv": truth_pipe_flows,
        "truth_measurements.csv": truth_measurements,
    }
    for filename, df in outputs.items():
        df.to_csv(output_dir / filename, index=False)

    return outputs
def _frame_with_timestamp(
    timestamp: pd.Series,
    df: pd.DataFrame,
    rename_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    output = df.copy()
    if rename_map:
        output = output.rename(columns=rename_map)
    output = output.reset_index(drop=True)
    output.insert(0, "timestamp", timestamp.to_numpy())
    return output


if __name__ == "__main__":
    generated = generate_truth_data()
    print(f"Saved scenario truth files to {TRUTH_CONFIG.output_dir}")
    for scenario_name, scenario_outputs in generated.items():
        print(f"{scenario_name}: {len(scenario_outputs)} files")
