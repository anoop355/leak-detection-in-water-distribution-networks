from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import EstimatorConfig

try:
    import wntr
except ImportError:  # pragma: no cover - exercised at runtime when dependency is missing
    wntr = None


@dataclass(frozen=True)
class ModelMetadata:
    demand_nodes: tuple[str, ...]
    helper_nodes: tuple[str, ...]
    all_report_nodes: tuple[str, ...]
    measured_pressure_node: str
    measured_flow_links: tuple[str, ...]
    report_flow_links: tuple[str, ...]
    elevations: dict[str, float]
    base_demands: dict[str, float]


def _require_wntr() -> None:
    if wntr is None:
        raise ImportError(
            "WNTR is required but not installed. Install dependencies with "
            "`py -3 -m pip install -r requirements.txt`."
        )


def load_water_network(inp_path: Path | str):
    _require_wntr()
    return wntr.network.WaterNetworkModel(str(inp_path))


def extract_model_metadata(config: EstimatorConfig) -> ModelMetadata:
    wn = load_water_network(config.inp_path)

    elevations: dict[str, float] = {}
    base_demands: dict[str, float] = {}
    for node_name in config.demand_nodes + config.helper_nodes:
        node = wn.get_node(node_name)
        elevations[node_name] = float(getattr(node, "elevation", 0.0))
        demand_list = getattr(node, "demand_timeseries_list", [])
        if node_name in config.demand_nodes and len(demand_list) > 0:
            base_demands[node_name] = float(demand_list[0].base_value)
        elif node_name in config.demand_nodes:
            base_demands[node_name] = 0.0

    return ModelMetadata(
        demand_nodes=config.demand_nodes,
        helper_nodes=config.helper_nodes,
        all_report_nodes=config.demand_nodes + config.helper_nodes,
        measured_pressure_node=config.measured_pressure_node,
        measured_flow_links=config.measured_flow_links,
        report_flow_links=config.report_flow_links,
        elevations=elevations,
        base_demands=base_demands,
    )


def build_initial_state(initial_heads: np.ndarray, config: EstimatorConfig) -> np.ndarray:
    return np.concatenate([np.asarray(initial_heads, dtype=float), config.initial_demands.copy()])
