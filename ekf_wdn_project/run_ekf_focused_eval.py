"""
run_ekf_focused_eval.py

Focused EKF evaluation on 12 representative scenarios from test_dataset:
  - 2 no-leak scenarios
  - 2 leak scenarios per pipe (pipes 1-5) = 10 leak scenarios

Metrics: MAE, RMSE, R^2 per scenario per unmonitored sensor.
Summary: per-scenario table + grouped totals (no-leak, pipe-1..5).

Placement: ekf_wdn_project/   (so existing module imports work without modification)

Usage (from ekf_wdn_project/):
    python run_ekf_focused_eval.py

Outputs (written one level up, next to the autoencoder results):
    ../ekf_focused_eval_results.csv
    ../ekf_focused_eval_plots/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── ensure ekf_wdn_project modules are importable ────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import EstimatorConfig
from ekf import ExtendedKalmanFilter
from hydraulic_interface import HydraulicInterface
from jacobians import numerical_jacobian
from load_model import ModelMetadata, build_initial_state, extract_model_metadata

# ── suppress noisy wntr / epanet logging ─────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logging.getLogger("wntr").setLevel(logging.ERROR)
logging.getLogger("wntr.epanet.io").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
_PARENT      = _HERE.parent
DATASET_ROOT = _PARENT / "test_dataset" / "scenarios"
MANIFEST     = _PARENT / "test_dataset" / "manifests" / "manifest.csv"
INP_PATH     = _HERE / "base3.inp"
EVAL_CSV     = _PARENT / "ekf_focused_eval_results.csv"
PLOT_DIR     = _PARENT / "ekf_focused_eval_plots"

ALL_SENSORS = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
MONITORED   = ["P4", "Q1a", "Q3a"]
UNMONITORED = ["P2", "P3", "P5", "P6", "Q2a", "Q4a", "Q5a"]

# EKF output column -> sensor name
PRESSURE_COL_MAP = {"P2": "2", "P3": "3", "P5": "5", "P6": "6"}
FLOW_COL_MAP     = {"Q2a": "2a", "Q4a": "4a", "Q5a": "5a"}

# Number of scenarios per category
N_NOLEAK_PICK = 2
N_LEAK_PICK   = 2   # per pipe


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def select_scenarios(manifest_path: Path, dataset_root: Path) -> list[dict]:
    """
    Pick N_NOLEAK_PICK no-leak + N_LEAK_PICK per pipe (1-5) scenarios.
    Returns list of dicts with keys: folder, label_detection, label_pipe.
    """
    df = pd.read_csv(manifest_path)

    selected = []

    # No-leak
    noleak_rows = df[df["label_detection"] == 0].head(N_NOLEAK_PICK)
    for _, row in noleak_rows.iterrows():
        folder = dataset_root / f"scenario_{int(row['scenario_id']):05d}"
        if (folder / "data.csv").exists() and (folder / "labels.json").exists():
            selected.append({
                "folder":          folder,
                "scenario_id":     int(row["scenario_id"]),
                "label_detection": 0,
                "label_pipe":      -1,
                "group":           "no-leak",
            })

    # Leak per pipe
    for pipe in range(1, 6):
        pipe_rows = df[
            (df["label_detection"] == 1) & (df["label_pipe"] == pipe)
        ].head(N_LEAK_PICK)
        for _, row in pipe_rows.iterrows():
            folder = dataset_root / f"scenario_{int(row['scenario_id']):05d}"
            if (folder / "data.csv").exists() and (folder / "labels.json").exists():
                selected.append({
                    "folder":          folder,
                    "scenario_id":     int(row["scenario_id"]),
                    "label_detection": 1,
                    "label_pipe":      pipe,
                    "group":           f"pipe-{pipe}",
                })

    return selected


# ─────────────────────────────────────────────────────────────────────────────
# EKF CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def make_config() -> EstimatorConfig:
    return EstimatorConfig(
        inp_path=INP_PATH,
        measurements_path=INP_PATH,
        output_dir=Path("_focused_tmp"),
        plots_dir=Path("_focused_tmp/plots"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORE: run EKF on one scenario
# ─────────────────────────────────────────────────────────────────────────────
def run_ekf_scenario(
    measurements_df: pd.DataFrame,
    config: EstimatorConfig,
    metadata: ModelMetadata,
    hydraulic: HydraulicInterface,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameters
    ----------
    measurements_df : columns [timestamp_s, P4, Q1a, Q3a]

    Returns
    -------
    node_pressures_df : columns [timestamp, 2, 3, 4, 5, 6, L1..L5]
    pipe_flows_df     : columns [timestamp, 1a, 1b, 2a, ...]
    """
    initial_snapshot  = hydraulic.simulate_snapshot(config.initial_demands, timestamp_seconds=0)
    initial_state     = build_initial_state(initial_snapshot.head_state_vector(metadata), config)
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
            snap = hydraulic.simulate_snapshot(demand_state, timestamp_seconds=timestamp)
            return np.concatenate(
                [snap.head_state_vector(metadata), snap.measurement_vector(metadata)]
            )

        def transition_function(state: np.ndarray) -> np.ndarray:
            d    = state[demand_count:]
            resp = hydraulic_response(d)
            return np.concatenate([resp[:demand_count], d])

        def measurement_function(state: np.ndarray) -> np.ndarray:
            resp = hydraulic_response(state[demand_count:])
            return resp[demand_count:]

        def transition_jacobian_function(state: np.ndarray) -> np.ndarray:
            d   = state[demand_count:]
            key = tuple(np.asarray(d, dtype=float))
            J   = jacobian_cache.get(key)
            if J is None:
                J = numerical_jacobian(hydraulic_response, d, config)
                jacobian_cache[key] = J
            F = np.zeros((config.state_size, config.state_size), dtype=float)
            F[:demand_count, demand_count:] = J[:demand_count, :]
            F[demand_count:, demand_count:] = np.eye(demand_count, dtype=float)
            return F

        def measurement_jacobian_function(state: np.ndarray) -> np.ndarray:
            d   = state[demand_count:]
            key = tuple(np.asarray(d, dtype=float))
            J   = jacobian_cache.get(key)
            if J is None:
                J = numerical_jacobian(hydraulic_response, d, config)
                jacobian_cache[key] = J
            H = np.zeros((config.measurement_size, config.state_size), dtype=float)
            H[:, demand_count:] = J[demand_count:, :]
            return H

        try:
            ekf.step(
                measurement=measurement,
                transition_function=transition_function,
                measurement_function=measurement_function,
                transition_jacobian_function=transition_jacobian_function,
                measurement_jacobian_function=measurement_jacobian_function,
            )
        except Exception:
            pass  # keep previous state

        # Re-simulate with updated demands to get full network state
        demands_upd  = ekf.x[demand_count:]
        updated_snap = hydraulic.simulate_snapshot(demands_upd, timestamp_seconds=timestamp)
        ekf.x[:demand_count] = updated_snap.head_state_vector(metadata)

        prow = {"timestamp": timestamp}
        for name in metadata.all_report_nodes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prow[name] = float(updated_snap.pressures.loc[name])
        node_pressure_rows.append(prow)

        frow = {"timestamp": timestamp}
        for name in metadata.report_flow_links:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                frow[name] = float(updated_snap.flows.loc[name])
        pipe_flow_rows.append(frow)

    return pd.DataFrame(node_pressure_rows), pd.DataFrame(pipe_flow_rows)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def build_measurements_df(data: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp_s": data["t"].values * 60.0,
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
        recon[sensor] = np.abs(flows_df[col].values.astype(np.float32))
    return recon

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_scenario(scenario_id: int, group: str,
                  actual: dict[str, np.ndarray],
                  recon: dict[str, np.ndarray],
                  plot_dir: Path) -> Path:
    T      = len(next(iter(actual.values())))
    t_axis = np.arange(T) * 15

    fig, axes = plt.subplots(len(UNMONITORED), 1,
                             figsize=(10, 2.5 * len(UNMONITORED)), sharex=True)
    for ax, sensor in zip(axes, UNMONITORED):
        ax.plot(t_axis, actual[sensor], "b-",  linewidth=1.5, label="Actual")
        ax.plot(t_axis, recon[sensor],  "r--", linewidth=1.5, label="EKF")
        ax.set_ylabel(sensor, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.suptitle(f"EKF | Scenario {scenario_id}  [{group}]", fontsize=11, y=1.01)
    fig.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"scenario_{scenario_id:05d}_{group}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTING
# ─────────────────────────────────────────────────────────────────────────────
def print_group_summary(group_name: str, rows: list[dict]) -> None:
    """Print mean MAE / RMSE / R^2 for a group of scenario result rows."""
    print(f"\n  Group: {group_name}  (n={len(rows)})")
    print(f"    {'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}")
    print("    " + "-" * 40)
    for sensor in UNMONITORED:
        maes  = [r[f"{sensor}_mae"]  for r in rows]
        rmses = [r[f"{sensor}_rmse"] for r in rows]
        r2s   = [r[f"{sensor}_r2"]   for r in rows]
        print(f"    {sensor:<8}  {np.mean(maes):>10.4f}  "
              f"{np.mean(rmses):>10.4f}  {np.mean(r2s):>8.4f}")
    all_maes = [r[f"{s}_mae"] for r in rows for s in UNMONITORED]
    print(f"    {'OVERALL':<8}  {np.mean(all_maes):>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not INP_PATH.exists():
        raise FileNotFoundError(f"EPANET model not found: {INP_PATH}")
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    print(f"EKF model    : {INP_PATH}")
    print(f"Dataset      : {DATASET_ROOT}")
    print(f"Monitored    : {MONITORED}")
    print(f"Unmonitored  : {UNMONITORED}")

    scenarios = select_scenarios(MANIFEST, DATASET_ROOT)
    print(f"\nSelected {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"  scenario_{s['scenario_id']:05d}  group={s['group']}")

    # Build EKF objects once (shared for all scenarios)
    cfg      = make_config()
    metadata = extract_model_metadata(cfg)
    hydraulic = HydraulicInterface(cfg, metadata)

    rows_by_group: dict[str, list[dict]] = {}
    all_rows = []

    for idx, scn in enumerate(scenarios, 1):
        folder = scn["folder"]
        scn_id = scn["scenario_id"]
        group  = scn["group"]

        print(f"\n[{idx}/{len(scenarios)}] Scenario {scn_id:05d}  ({group}) ...")

        data   = pd.read_csv(folder / "data.csv")
        actual = {s: data[s].values.astype(np.float32) for s in ALL_SENSORS}
        meas_df = build_measurements_df(data)

        try:
            pressures_df, flows_df = run_ekf_scenario(meas_df, cfg, metadata, hydraulic)
            recon = extract_ekf_reconstruction(pressures_df, flows_df)
        except Exception as exc:
            print(f"  [ERROR] EKF failed: {exc}")
            continue

        row = {
            "scenario_id":     scn_id,
            "group":           group,
            "label_detection": scn["label_detection"],
            "label_pipe":      scn["label_pipe"],
        }
        for sensor in UNMONITORED:
            mae, rmse, r2 = compute_metrics(actual[sensor], recon[sensor])
            row[f"{sensor}_mae"]  = round(mae,  6)
            row[f"{sensor}_rmse"] = round(rmse, 6)
            row[f"{sensor}_r2"]   = round(r2,   6)
            print(f"  {sensor:<8}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

        all_rows.append(row)
        rows_by_group.setdefault(group, []).append(row)

        # Plot
        out = plot_scenario(scn_id, group, actual, recon, PLOT_DIR)
        print(f"  Plot -> {out}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    eval_df = pd.DataFrame(all_rows)
    eval_df.to_csv(EVAL_CSV, index=False)
    print(f"\n[OK] Per-scenario results -> {EVAL_CSV}")

    # ── Per-scenario summary table ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-SCENARIO RESULTS")
    print("=" * 70)
    hdr = f"{'ScnID':>8}  {'Group':<10}  " + "  ".join(
        f"{s+'_MAE':>10}" for s in UNMONITORED
    ) + f"  {'OverallMAE':>11}"
    print(hdr)
    print("-" * len(hdr))
    for row in all_rows:
        maes = [row[f"{s}_mae"] for s in UNMONITORED]
        cols = f"{row['scenario_id']:>8}  {row['group']:<10}  " + "  ".join(
            f"{m:>10.4f}" for m in maes
        ) + f"  {np.mean(maes):>11.4f}"
        print(cols)

    # ── Grouped summaries ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GROUPED SUMMARIES")
    print("=" * 70)

    group_order = ["no-leak"] + [f"pipe-{p}" for p in range(1, 6)]
    for g in group_order:
        if g in rows_by_group:
            print_group_summary(g, rows_by_group[g])

    # Overall
    print_group_summary("ALL", all_rows)

    # Leak vs No-Leak MAE ratio
    if "no-leak" in rows_by_group:
        leak_rows = [r for g, rs in rows_by_group.items()
                     if g != "no-leak" for r in rs]
        if leak_rows:
            print("\n--- Leak/No-Leak MAE ratio per sensor ---")
            for sensor in UNMONITORED:
                nl_mae = np.mean([r[f"{sensor}_mae"] for r in rows_by_group["no-leak"]])
                lk_mae = np.mean([r[f"{sensor}_mae"] for r in leak_rows])
                ratio  = lk_mae / nl_mae if nl_mae > 1e-10 else float("nan")
                flag   = " [anomaly signal present]" if ratio > 1.05 else ""
                print(f"  {sensor:<8}: {ratio:.3f}{flag}")

    print(f"\n[OK] Plots saved -> {PLOT_DIR}/")
    print("Focused EKF evaluation complete.")


if __name__ == "__main__":
    main()
