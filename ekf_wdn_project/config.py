from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EstimatorConfig:
    inp_path: Path = Path("base3.inp")
    measurements_path: Path = Path("measurements.csv")
    output_dir: Path = Path("outputs")
    plots_dir: Path = Path("outputs/plots")
    hydraulic_timestep_seconds: int = 15 * 60
    demand_nodes: tuple[str, ...] = ("2", "3", "4", "5", "6")
    helper_nodes: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5")
    measured_pressure_node: str = "4"
    measured_flow_links: tuple[str, ...] = ("1a", "3a")
    report_flow_links: tuple[str, ...] = (
        "1a",
        "1b",
        "2a",
        "2b",
        "3a",
        "3b",
        "4a",
        "4b",
        "5a",
        "5b",
    )
    initial_demands: np.ndarray = field(
        default_factory=lambda: np.array([0.0015, 0.0010, 0.0005, 0.0010, 0.0010], dtype=float)
    )
    initial_head_variance: float = 1.0
    initial_demand_variance: float = 0.25
    head_process_variance: float = 0.01
    demand_process_variance: float = 0.0025
    pressure_sensor_std: float = 0.5
    flow_sensor_relative_std: float = 0.05
    jacobian_relative_step: float = 1e-4
    jacobian_min_step: float = 1e-6
    innovation_regularization: float = 1e-8
    covariance_regularization: float = 1e-10
    minimum_demand: float = 0.0
    maximum_demand: float | None = None
    bad_data_sigma_limit: float | None = None

    @property
    def state_size(self) -> int:
        return len(self.demand_nodes) * 2

    @property
    def measurement_size(self) -> int:
        return 1 + len(self.measured_flow_links)

    @property
    def initial_covariance(self) -> np.ndarray:
        head_vars = np.full(len(self.demand_nodes), self.initial_head_variance, dtype=float)
        demand_vars = np.full(len(self.demand_nodes), self.initial_demand_variance, dtype=float)
        return np.diag(np.concatenate([head_vars, demand_vars]))

    @property
    def process_noise(self) -> np.ndarray:
        head_vars = np.full(len(self.demand_nodes), self.head_process_variance, dtype=float)
        demand_vars = np.full(len(self.demand_nodes), self.demand_process_variance, dtype=float)
        return np.diag(np.concatenate([head_vars, demand_vars]))


CONFIG = EstimatorConfig()
