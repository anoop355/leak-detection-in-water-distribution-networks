from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wntr


@dataclass(frozen=True)
class MeasurementConfig:
    inp_path: Path = Path("base3.inp")
    output_path: Path = Path("measurements.csv")
    include_noise: bool = False
    pressure_noise_std: float = 0.5
    flow_noise_relative_std: float = 0.05
    random_seed: int | None = 42
    report_timestep_seconds: int = 15 * 60


CONFIG = MeasurementConfig()


def _apply_sensor_noise(df: pd.DataFrame, config: MeasurementConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.random_seed)
    noisy = df.copy()

    noisy["P4"] = noisy["P4"] + rng.normal(
        loc=0.0,
        scale=config.pressure_noise_std,
        size=len(noisy),
    )

    for flow_col in ("Q1a", "Q3a"):
        flow_std = config.flow_noise_relative_std * noisy[flow_col].abs().to_numpy()
        noisy[flow_col] = noisy[flow_col] + rng.normal(
            loc=0.0,
            scale=flow_std,
            size=len(noisy),
        )

    return noisy


def generate_measurements(config: MeasurementConfig = CONFIG) -> pd.DataFrame:
    wn = wntr.network.WaterNetworkModel(str(config.inp_path))
    wn.options.time.hydraulic_timestep = config.report_timestep_seconds
    wn.options.time.report_timestep = config.report_timestep_seconds

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    pressure = results.node["pressure"]["4"]
    flow_1a = results.link["flowrate"]["1a"]
    flow_3a = results.link["flowrate"]["3a"]

    measurements = pd.DataFrame(
        {
            "timestamp": pressure.index,
            "P4": pressure.to_numpy(),
            "Q1a": flow_1a.to_numpy(),
            "Q3a": flow_3a.to_numpy(),
        }
    )

    if config.include_noise:
        measurements = _apply_sensor_noise(measurements, config)

    measurements.to_csv(config.output_path, index=False)
    return measurements


if __name__ == "__main__":
    df = generate_measurements()
    print(f"Saved {len(df)} rows to {CONFIG.output_path}")
