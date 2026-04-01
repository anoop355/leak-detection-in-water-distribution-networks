from __future__ import annotations

from dataclasses import dataclass
import logging
import warnings

import numpy as np
import pandas as pd

from config import EstimatorConfig
from load_model import ModelMetadata, load_water_network

LOGGER = logging.getLogger(__name__)


@dataclass
class HydraulicSnapshot:
    heads: pd.Series
    pressures: pd.Series
    flows: pd.Series

    def measurement_vector(self, metadata: ModelMetadata) -> np.ndarray:
        pressure = float(self.pressures.loc[metadata.measured_pressure_node])
        flows = [float(self.flows.loc[link_name]) for link_name in metadata.measured_flow_links]
        return np.array([pressure, *flows], dtype=float)

    def head_state_vector(self, metadata: ModelMetadata) -> np.ndarray:
        return self.heads.loc[list(metadata.demand_nodes)].to_numpy(dtype=float)


class HydraulicInterface:
    def __init__(self, config: EstimatorConfig, metadata: ModelMetadata):
        self.config = config
        self.metadata = metadata

    def simulate_snapshot(self, demands: np.ndarray, timestamp_seconds: float | int | None = None) -> HydraulicSnapshot:
        demand_vector = np.asarray(demands, dtype=float).copy()
        demand_vector = np.maximum(demand_vector, self.config.minimum_demand)
        if self.config.maximum_demand is not None:
            demand_vector = np.minimum(demand_vector, self.config.maximum_demand)

        wn = load_water_network(self.config.inp_path)
        wn.options.time.duration = 0
        wn.options.time.hydraulic_timestep = self.config.hydraulic_timestep_seconds
        wn.options.time.report_timestep = self.config.hydraulic_timestep_seconds

        for node_name, demand_value in zip(self.metadata.demand_nodes, demand_vector):
            self._set_node_demand(wn, node_name, demand_value)

        if timestamp_seconds is not None and hasattr(wn.options.time, "pattern_start"):
            wn.options.time.pattern_start = int(timestamp_seconds)

        sim = self._build_simulator(wn)
        results = sim.run_sim()

        heads = results.node["head"].iloc[-1].copy()
        pressures = results.node["pressure"].iloc[-1].copy()
        flows = results.link["flowrate"].iloc[-1].copy()

        return HydraulicSnapshot(heads=heads, pressures=pressures, flows=flows)

    def build_measurement_noise(self) -> np.ndarray:
        nominal_snapshot = self.simulate_snapshot(self.config.initial_demands, timestamp_seconds=0)
        nominal_flows = [
            abs(float(nominal_snapshot.flows.loc[link_name]))
            for link_name in self.metadata.measured_flow_links
        ]
        flow_stds = [
            max(self.config.flow_sensor_relative_std * flow, self.config.jacobian_min_step)
            for flow in nominal_flows
        ]
        stds = np.array([self.config.pressure_sensor_std, *flow_stds], dtype=float)
        return np.diag(stds**2)

    def _build_simulator(self, wn):
        try:
            from wntr.sim import EpanetSimulator

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                return EpanetSimulator(wn)
        except Exception:  # pragma: no cover - depends on external install details
            LOGGER.warning("Falling back to WNTRSimulator because EPANET simulator was unavailable.")
            from wntr.sim import WNTRSimulator

            return WNTRSimulator(wn)

    def _set_node_demand(self, wn, node_name: str, demand_value: float) -> None:
        node = wn.get_node(node_name)
        demand_list = getattr(node, "demand_timeseries_list", [])
        if len(demand_list) == 0:
            node.add_demand(demand_value, pattern_name=None, category="EKF")
            return

        demand_list[0].base_value = float(demand_value)
        for demand_ts in demand_list:
            if hasattr(demand_ts, "pattern_name"):
                demand_ts.pattern_name = None
