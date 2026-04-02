from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ensure ekf_wdn_project modules are importable regardless of cwd
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import EstimatorConfig
from ekf import ExtendedKalmanFilter
from hydraulic_interface import HydraulicInterface
from jacobians import numerical_jacobian
from load_model import ModelMetadata, build_initial_state, extract_model_metadata

logging.basicConfig(level=logging.WARNING)
logging.getLogger("wntr").setLevel(logging.ERROR)
logging.getLogger("wntr.epanet.io").setLevel(logging.ERROR)

# PATHS

_PARENT       = _HERE.parent
DATASET_ROOT  = _PARENT / "stgcn_dataset_v2" / "scenarios"
MANIFEST_TEST = _PARENT / "stgcn_dataset_v2" / "manifests" / "manifest_test.csv"
INP_PATH      = _HERE / "base3.inp"
EVAL_CSV_PATH = _PARENT / "ekf_eval_results.csv"
PLOT_DIR      = _PARENT / "ekf_reconstruction_plots"

ALL_SENSORS     = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
MONITORED       = ["P4", "Q1a", "Q3a"]
UNMONITORED     = ["P2", "P3", "P5", "P6", "Q2a", "Q4a", "Q5a"]

PRESSURE_COL_MAP = {"P2": "2", "P3": "3", "P5": "5", "P6": "6"}
FLOW_COL_MAP     = {"Q2a": "2a", "Q4a": "4a", "Q5a": "5a"}

NUM_SAMPLE_NOLEAK = 3
NUM_SAMPLE_LEAK   = 3



# EKF CONFIG 

def make_config() -> EstimatorConfig:
    return EstimatorConfig(
        inp_path=INP_PATH,
        measurements_path=INP_PATH,   # not used in batch mode — kept for compat
        output_dir=Path("_batch_tmp"),
        plots_dir=Path("_batch_tmp/plots"),
    )

# CORE: run EKF on a single scenario and return reconstructed pressures/flows

def run_ekf_scenario(
    measurements_df: pd.DataFrame,
    config: EstimatorConfig,
    metadata: ModelMetadata,
    hydraulic: HydraulicInterface,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    initial_snapshot = hydraulic.simulate_snapshot(
        config.initial_demands, timestamp_seconds=0
    )
    initial_state    = build_initial_state(
        initial_snapshot.head_state_vector(metadata), config
    )
    measurement_noise = hydraulic.build_measurement_noise()

    ekf = ExtendedKalmanFilter(
        initial_state=initial_state,
        initial_covariance=config.initial_covariance,
        process_noise=config.process_noise,
        measurement_noise=measurement_noise,
        config=config,
    )

    demand_count       = len(metadata.demand_nodes)
    node_pressure_rows = []
    pipe_flow_rows     = []

    for _, row in measurements_df.iterrows():
        timestamp   = float(row["timestamp_s"])
        measurement = np.array(
            [float(row["P4"]), float(row["Q1a"]), float(row["Q3a"])],
            dtype=float,
        )

        jacobian_cache: dict[tuple[float, ...], np.ndarray] = {}

        def hydraulic_response(demand_state: np.ndarray) -> np.ndarray:
            snap = hydraulic.simulate_snapshot(
                demand_state, timestamp_seconds=timestamp
            )
            return np.concatenate(
                [snap.head_state_vector(metadata), snap.measurement_vector(metadata)]
            )

        def transition_function(state: np.ndarray) -> np.ndarray:
            d = state[demand_count:]
            resp = hydraulic_response(d)
            return np.concatenate([resp[:demand_count], d])

        def measurement_function(state: np.ndarray) -> np.ndarray:
            resp = hydraulic_response(state[demand_count:])
            return resp[demand_count:]

        def transition_jacobian_function(state: np.ndarray) -> np.ndarray:
            d = state[demand_count:]
            key = tuple(np.asarray(d, dtype=float))
            J = jacobian_cache.get(key)
            if J is None:
                J = numerical_jacobian(hydraulic_response, d, config)
                jacobian_cache[key] = J
            F = np.zeros((config.state_size, config.state_size), dtype=float)
            F[:demand_count, demand_count:] = J[:demand_count, :]
            F[demand_count:, demand_count:] = np.eye(demand_count, dtype=float)
            return F

        def measurement_jacobian_function(state: np.ndarray) -> np.ndarray:
            d = state[demand_count:]
            key = tuple(np.asarray(d, dtype=float))
            J = jacobian_cache.get(key)
            if J is None:
                J = numerical_jacobian(hydraulic_response, d, config)
                jacobian_cache[key] = J
            H = np.zeros((config.measurement_size, config.state_size), dtype=float)
            H[:, demand_count:] = J[demand_count:, :]
            return H

        try:
            step_result = ekf.step(
                measurement=measurement,
                transition_function=transition_function,
                measurement_function=measurement_function,
                transition_jacobian_function=transition_jacobian_function,
                measurement_jacobian_function=measurement_jacobian_function,
            )
        except Exception:
            # On failure, keep previous state and skip this timestep
            step_result = None

        # Re-simulate with updated demands to get full network state
        demands_upd = ekf.x[demand_count:]
        updated_snap = hydraulic.simulate_snapshot(
            demands_upd, timestamp_seconds=timestamp
        )
        ekf.x[:demand_count] = updated_snap.head_state_vector(metadata)

        # Record pressures
        prow = {"timestamp": timestamp}
        for name in metadata.all_report_nodes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prow[name] = float(updated_snap.pressures.loc[name])
        node_pressure_rows.append(prow)

        # Record flows
        frow = {"timestamp": timestamp}
        for name in metadata.report_flow_links:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frow[name] = float(updated_snap.flows.loc[name])
        pipe_flow_rows.append(frow)

    return pd.DataFrame(node_pressure_rows), pd.DataFrame(pipe_flow_rows)

# HELPERS

def load_scenario(folder: Path) -> tuple[pd.DataFrame, dict]:
    data   = pd.read_csv(folder / "data.csv")
    import json
    with open(folder / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    return data, labels

def is_no_leak(labels: dict) -> bool:
    return int(labels.get("label_detection", 0)) == 0

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)

def build_measurements_df(data: pd.DataFrame) -> pd.DataFrame:
    """Convert scenario data.csv into measurements DataFrame for EKF."""
    return pd.DataFrame({
        "timestamp_s": data["t"].values * 60.0,   # minutes -> seconds
        "P4":  data["P4"].values,
        "Q1a": data["Q1a"].values,
        "Q3a": data["Q3a"].values,
    })

def extract_ekf_reconstruction(
    pressures_df: pd.DataFrame,
    flows_df: pd.DataFrame,
) -> dict[str, np.ndarray]:
    recon = {}
    for sensor, col in PRESSURE_COL_MAP.items():
        recon[sensor] = pressures_df[col].values.astype(np.float32)
    for sensor, col in FLOW_COL_MAP.items():
        # Take abs() — WNTR flows can be signed by direction
        recon[sensor] = np.abs(flows_df[col].values.astype(np.float32))
    return recon

# PLOTTING

def plot_scenario(scenario_id, actual_raw, ekf_recon, label_str, plot_dir):
    T      = actual_raw.shape[0]
    t_axis = np.arange(T) * 15
    fig, axes = plt.subplots(len(UNMONITORED), 1,
                             figsize=(10, 2.5 * len(UNMONITORED)), sharex=True)

    for ax, sensor in zip(axes, UNMONITORED):
        ax.plot(t_axis, actual_raw[sensor], "b-",  linewidth=1.5, label="Actual")
        ax.plot(t_axis, ekf_recon[sensor],  "r--", linewidth=1.5, label="EKF")
        ax.set_ylabel(sensor, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.suptitle(f"EKF | Scenario {scenario_id}  [{label_str}]", fontsize=11, y=1.01)
    fig.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"scenario_{scenario_id}_{label_str}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out

# PER-SCENARIO PROCESSING  

def _process_one_scenario(folder: Path) -> dict | None:

    try:
        cfg      = make_config()
        metadata = extract_model_metadata(cfg)
        hydraulic = HydraulicInterface(cfg, metadata)

        data, labels = load_scenario(folder)
        is_nl     = is_no_leak(labels)
        label_str = "noleak" if is_nl else "leak"
        meas_df   = build_measurements_df(data)
        actual    = {s: data[s].values.astype(np.float32) for s in ALL_SENSORS}

        pressures_df, flows_df = run_ekf_scenario(meas_df, cfg, metadata, hydraulic)
        ekf_recon = extract_ekf_reconstruction(pressures_df, flows_df)

        row = {"scenario_id": folder.name, "label": label_str}
        for sensor in UNMONITORED:
            y_true = actual[sensor]
            y_pred = ekf_recon[sensor]
            n      = min(len(y_true), len(y_pred))
            y_true, y_pred = y_true[:n], y_pred[:n]
            row[f"{sensor}_mae"]  = round(float(np.mean(np.abs(y_true - y_pred))), 6)
            row[f"{sensor}_rmse"] = round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 6)
            row[f"{sensor}_r2"]   = round(r2_score(y_true, y_pred), 6)

        # Store actual+recon arrays for optional plotting (serialisable as lists)
        actual_lists  = {s: actual[s].tolist()    for s in ALL_SENSORS}
        recon_lists   = {s: ekf_recon[s].tolist() for s in UNMONITORED}

        return {
            "row":       row,
            "is_nl":     is_nl,
            "label_str": label_str,
            "actual":    actual_lists,
            "recon":     recon_lists,
        }
    except Exception as exc:
        return {"error": str(exc), "scenario_id": folder.name}


# Top-level picklable wrapper required by ProcessPoolExecutor
def _scenario_worker(folder_str: str) -> dict | None:
    # Re-insert module path in each subprocess (needed on some systems)
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    return _process_one_scenario(Path(folder_str))

# MAIN

def main():
    parser = argparse.ArgumentParser(description="EKF batch evaluation")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel worker processes (default: 1)")
    args = parser.parse_args()

    if not INP_PATH.exists():
        raise FileNotFoundError(f"EPANET model not found: {INP_PATH}")
    if not MANIFEST_TEST.exists():
        raise FileNotFoundError(f"Test manifest not found: {MANIFEST_TEST}")

    print(f"EKF model    : {INP_PATH}")
    print(f"Dataset      : {DATASET_ROOT}")
    print(f"Monitored    : {MONITORED}")
    print(f"Unmonitored  : {UNMONITORED}")
    print(f"Workers      : {args.workers}")

    # Load test scenario folders
    manifest = pd.read_csv(MANIFEST_TEST)
    test_folders = []
    for scn_id in manifest["scenario_id"].values:
        p = DATASET_ROOT / f"scenario_{int(scn_id):05d}"
        if (p / "data.csv").exists() and (p / "labels.json").exists():
            test_folders.append(p)

    print(f"Test scenarios: {len(test_folders)}")
    print("Starting batch evaluation. This may take a long time on CPU.\n")

    rows            = []
    all_mae_noleak  = {s: [] for s in UNMONITORED}
    all_mae_leak    = {s: [] for s in UNMONITORED}
    no_leak_samples = []
    leak_samples    = []
    failed          = 0
    completed       = 0

    def _collect(result: dict):
        nonlocal failed, completed
        completed += 1
        if result is None or "error" in result:
            scn = result.get("scenario_id", "unknown") if result else "unknown"
            err = result.get("error", "unknown error") if result else "None returned"
            print(f"  [WARN] {scn} failed: {err}")
            failed += 1
            return

        row       = result["row"]
        is_nl     = result["is_nl"]
        label_str = result["label_str"]
        actual    = {s: np.array(v, dtype=np.float32) for s, v in result["actual"].items()}
        ekf_recon = {s: np.array(v, dtype=np.float32) for s, v in result["recon"].items()}

        for sensor in UNMONITORED:
            mae = row[f"{sensor}_mae"]
            (all_mae_noleak if is_nl else all_mae_leak)[sensor].append(mae)
        rows.append(row)

        scn_id = row["scenario_id"]
        if is_nl and len(no_leak_samples) < NUM_SAMPLE_NOLEAK:
            no_leak_samples.append((scn_id, actual, ekf_recon, label_str))
        elif not is_nl and len(leak_samples) < NUM_SAMPLE_LEAK:
            leak_samples.append((scn_id, actual, ekf_recon, label_str))

        if completed % 10 == 0 or completed == len(test_folders):
            print(f"  Processed {completed}/{len(test_folders)} scenarios "
                  f"({failed} failed)...")

    if args.workers <= 1:
        for folder in test_folders:
            _collect(_process_one_scenario(folder))
    else:
        folder_strs = [str(f) for f in test_folders]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_scenario_worker, fs): fs
                       for fs in folder_strs}
            for future in as_completed(futures):
                try:
                    _collect(future.result())
                except Exception as exc:
                    print(f"  [WARN] Worker exception: {exc}")
                    failed += 1
                    completed += 1

    # Save CSV
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(EVAL_CSV_PATH, index=False)
    print(f"\n[OK] Saved per-scenario results -> {EVAL_CSV_PATH}")
    if failed:
        print(f"  ({failed} scenarios failed and were excluded)")

    # Summary metrics
    print("\n=== Per-Node Test Metrics — EKF (denormalised, unmonitored nodes only) ===")
    print(f"{'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}")
    print("-" * 42)
    node_maes = {}
    for sensor in UNMONITORED:
        mean_mae  = float(np.mean(eval_df[f"{sensor}_mae"].values))
        mean_rmse = float(np.mean(eval_df[f"{sensor}_rmse"].values))
        mean_r2   = float(np.mean(eval_df[f"{sensor}_r2"].values))
        node_maes[sensor] = mean_mae
        print(f"{sensor:<8}  {mean_mae:>10.4f}  {mean_rmse:>10.4f}  {mean_r2:>8.4f}")

    overall_mae = float(np.mean(list(node_maes.values())))
    worst       = max(node_maes, key=node_maes.get)
    print(f"\nMean MAE across all unmonitored nodes : {overall_mae:.4f}")
    print(f"Worst-reconstructed node              : {worst} (MAE={node_maes[worst]:.4f})")

    # Leak vs No-Leak
    print("\n=== Leak vs. No-Leak Reconstruction MAE ===")
    print(f"{'Sensor':<8}  {'No-Leak MAE':>12}  {'Leak MAE':>10}")
    print("-" * 36)
    ratio_check = {}
    for sensor in UNMONITORED:
        nl = float(np.mean(all_mae_noleak[sensor])) if all_mae_noleak[sensor] else float("nan")
        lk = float(np.mean(all_mae_leak[sensor]))   if all_mae_leak[sensor]   else float("nan")
        print(f"{sensor:<8}  {nl:>12.4f}  {lk:>10.4f}")
        if all_mae_noleak[sensor] and all_mae_leak[sensor] and nl > 1e-10:
            ratio_check[sensor] = lk / nl

    if ratio_check:
        print("\nLeak/NoLeak MAE ratio (>1.05 means anomaly signal preserved):")
        for sensor, ratio in ratio_check.items():
            flag = " [anomaly signal preserved]" if ratio > 1.05 else ""
            print(f"  {sensor:<8}: {ratio:.3f}{flag}")

    # Plots
    for scn_id, actual, ekf_recon, label_str in no_leak_samples + leak_samples:
        out = plot_scenario(scn_id, actual, ekf_recon, label_str, PLOT_DIR)
        print(f"  Saved plot -> {out}")

    print(f"\n[OK] Plots saved -> {PLOT_DIR}/")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
