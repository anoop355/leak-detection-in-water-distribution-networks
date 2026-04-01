from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import CONFIG, EstimatorConfig
from ekf import ExtendedKalmanFilter
from hydraulic_interface import HydraulicInterface
from jacobians import numerical_jacobian
from load_model import build_initial_state, extract_model_metadata
from plot_results import generate_plots


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logging.getLogger("wntr.epanet.io").setLevel(logging.ERROR)


def load_measurements(path: Path) -> pd.DataFrame:
    measurements = pd.read_csv(path)
    required_columns = {"timestamp", "P4", "Q1a", "Q3a"}
    missing = required_columns.difference(measurements.columns)
    if missing:
        raise ValueError(f"Measurement file is missing required columns: {sorted(missing)}")
    return measurements


def run_estimator(config: EstimatorConfig = CONFIG) -> None:
    configure_logging()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)

    metadata = extract_model_metadata(config)
    hydraulic = HydraulicInterface(config, metadata)
    measurements = load_measurements(config.measurements_path)

    initial_snapshot = hydraulic.simulate_snapshot(config.initial_demands, timestamp_seconds=0)
    initial_state = build_initial_state(initial_snapshot.head_state_vector(metadata), config)
    measurement_noise = hydraulic.build_measurement_noise()

    ekf = ExtendedKalmanFilter(
        initial_state=initial_state,
        initial_covariance=config.initial_covariance,
        process_noise=config.process_noise,
        measurement_noise=measurement_noise,
        config=config,
    )

    state_rows: list[dict[str, float]] = []
    predicted_rows: list[dict[str, float]] = []
    residual_rows: list[dict[str, float]] = []
    node_head_rows: list[dict[str, float]] = []
    node_pressure_rows: list[dict[str, float]] = []
    pipe_flow_rows: list[dict[str, float]] = []
    demand_count = len(metadata.demand_nodes)

    for row_index, measurement_row in measurements.iterrows():
        timestamp = float(measurement_row["timestamp"])
        measurement = np.array(
            [
                float(measurement_row["P4"]),
                float(measurement_row["Q1a"]),
                float(measurement_row["Q3a"]),
            ],
            dtype=float,
        )

        LOGGER.info("Processing timestep %s at t=%ss", row_index, timestamp)
        jacobian_cache: dict[tuple[float, ...], np.ndarray] = {}

        def hydraulic_response(demand_state: np.ndarray) -> np.ndarray:
            snapshot = hydraulic.simulate_snapshot(demand_state, timestamp_seconds=timestamp)
            return np.concatenate(
                [
                    snapshot.head_state_vector(metadata),
                    snapshot.measurement_vector(metadata),
                ]
            )

        def transition_function(state: np.ndarray) -> np.ndarray:
            demand_state = state[demand_count:]
            response = hydraulic_response(demand_state)
            return np.concatenate([response[:demand_count], demand_state])

        def measurement_function(state: np.ndarray) -> np.ndarray:
            response = hydraulic_response(state[demand_count:])
            return response[demand_count:]

        def transition_jacobian_function(state: np.ndarray) -> np.ndarray:
            demand_state = state[demand_count:]
            cache_key = tuple(np.asarray(demand_state, dtype=float))
            response_jacobian = jacobian_cache.get(cache_key)
            if response_jacobian is None:
                response_jacobian = numerical_jacobian(hydraulic_response, demand_state, config)
                jacobian_cache[cache_key] = response_jacobian
            transition_jacobian = np.zeros((config.state_size, config.state_size), dtype=float)
            transition_jacobian[:demand_count, demand_count:] = response_jacobian[:demand_count, :]
            transition_jacobian[demand_count:, demand_count:] = np.eye(demand_count, dtype=float)
            return transition_jacobian

        def measurement_jacobian_function(state: np.ndarray) -> np.ndarray:
            demand_state = state[demand_count:]
            cache_key = tuple(np.asarray(demand_state, dtype=float))
            response_jacobian = jacobian_cache.get(cache_key)
            if response_jacobian is None:
                response_jacobian = numerical_jacobian(hydraulic_response, demand_state, config)
                jacobian_cache[cache_key] = response_jacobian
            measurement_jacobian = np.zeros((config.measurement_size, config.state_size), dtype=float)
            measurement_jacobian[:, demand_count:] = response_jacobian[demand_count:, :]
            return measurement_jacobian

        try:
            step_result = ekf.step(
                measurement=measurement,
                transition_function=transition_function,
                measurement_function=measurement_function,
                transition_jacobian_function=transition_jacobian_function,
                measurement_jacobian_function=measurement_jacobian_function,
            )
        except Exception as exc:
            LOGGER.exception("EKF step failed at timestep %s: %s", row_index, exc)
            raise

        if np.any(step_result.x_upd[demand_count:] < 0.0):
            LOGGER.warning("Negative demand estimate detected after update at timestep %s.", row_index)

        updated_snapshot = hydraulic.simulate_snapshot(
            step_result.x_upd[demand_count:],
            timestamp_seconds=timestamp,
        )
        ekf.x[:demand_count] = updated_snapshot.head_state_vector(metadata)

        residual_norm = float(np.linalg.norm(step_result.residual))
        LOGGER.info("Residual magnitude at timestep %s: %.6f", row_index, residual_norm)

        state_rows.append(_build_state_row(timestamp, ekf.x, metadata))
        predicted_rows.append(
            {
                "timestamp": timestamp,
                "P4_measured": measurement[0],
                "P4_predicted": step_result.y_pred[0],
                "Q1a_measured": measurement[1],
                "Q1a_predicted": step_result.y_pred[1],
                "Q3a_measured": measurement[2],
                "Q3a_predicted": step_result.y_pred[2],
            }
        )
        residual_rows.append(
            {
                "timestamp": timestamp,
                "residual_P4": step_result.residual[0],
                "residual_Q1a": step_result.residual[1],
                "residual_Q3a": step_result.residual[2],
                "nr_P4": step_result.normalized_residual[0],
                "nr_Q1a": step_result.normalized_residual[1],
                "nr_Q3a": step_result.normalized_residual[2],
                "innovation_trace": float(np.trace(step_result.innovation_covariance)),
            }
        )
        node_head_rows.append(_series_to_row(timestamp, updated_snapshot.heads, metadata.all_report_nodes))
        node_pressure_rows.append(
            _series_to_row(timestamp, updated_snapshot.pressures, metadata.all_report_nodes)
        )
        pipe_flow_rows.append(_series_to_row(timestamp, updated_snapshot.flows, metadata.report_flow_links))

    state_estimates = pd.DataFrame(state_rows)
    predicted_measurements = pd.DataFrame(predicted_rows)
    residuals = pd.DataFrame(residual_rows)
    all_node_heads = pd.DataFrame(node_head_rows)
    all_node_pressures = pd.DataFrame(node_pressure_rows)
    all_pipe_flows = pd.DataFrame(pipe_flow_rows)

    state_estimates.to_csv(config.output_dir / "state_estimates.csv", index=False)
    predicted_measurements.to_csv(config.output_dir / "predicted_measurements.csv", index=False)
    residuals.to_csv(config.output_dir / "residuals.csv", index=False)
    all_node_heads.to_csv(config.output_dir / "all_node_heads.csv", index=False)
    all_node_pressures.to_csv(config.output_dir / "all_node_pressures.csv", index=False)
    all_pipe_flows.to_csv(config.output_dir / "all_pipe_flows.csv", index=False)

    generate_plots(predicted_measurements, residuals, state_estimates, config.plots_dir)
    LOGGER.info("Estimator run completed. Outputs saved to %s", config.output_dir.resolve())


def _build_state_row(timestamp: float, state: np.ndarray, metadata) -> dict[str, float]:
    heads = state[: len(metadata.demand_nodes)]
    demands = state[len(metadata.demand_nodes) :]
    row = {"timestamp": timestamp}
    for node_name, head in zip(metadata.demand_nodes, heads):
        row[f"H{node_name}"] = float(head)
    for node_name, demand in zip(metadata.demand_nodes, demands):
        row[f"D{node_name}"] = float(demand)
    return row


def _series_to_row(timestamp: float, series: pd.Series, names: tuple[str, ...]) -> dict[str, float]:
    row = {"timestamp": timestamp}
    for name in names:
        row[name] = float(series.loc[name])
    return row


if __name__ == "__main__":
    run_estimator()
