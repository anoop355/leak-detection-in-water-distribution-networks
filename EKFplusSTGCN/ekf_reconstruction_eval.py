"""
- Evaluates ability to reconstruct 7 unmonitored sensors across in test_dataset.
- 3 monitored sensors (P4, Q1a, Q3a) are fed to the EKF
- It reconstructs [P2, P3, P5, P6, Q2a, Q4a, Q5a].  
- Compare against the masked ground-truth columns.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── module paths ──────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_ROOT    = _HERE.parent
_EKF_DIR = _ROOT / "ekf_wdn_project"
for p in [str(_EKF_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from config import EstimatorConfig
from ekf import ExtendedKalmanFilter
from hydraulic_interface import HydraulicInterface
from jacobians import numerical_jacobian
from load_model import ModelMetadata, build_initial_state, extract_model_metadata

logging.basicConfig(level=logging.WARNING)
logging.getLogger("wntr").setLevel(logging.ERROR)
logging.getLogger("wntr.epanet.io").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
DATASET_ROOT = _ROOT / "test_dataset" / "scenarios"
MANIFEST     = _ROOT / "test_dataset" / "manifests" / "manifest.csv"
INP_PATH     = _EKF_DIR / "base3.inp"
RESULTS_DIR  = _HERE / "results"

ALL_SENSORS = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
MONITORED   = ["P4", "Q1a", "Q3a"]
UNMONITORED = ["P2", "P3", "P5", "P6", "Q2a", "Q4a", "Q5a"]

# EKF output column -> sensor name
PRESSURE_COL_MAP = {"P2": "2", "P3": "3", "P5": "5", "P6": "6"}
FLOW_COL_MAP     = {"Q2a": "2a", "Q4a": "4a", "Q5a": "5a"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_config(inp_path: Path | None = None) -> EstimatorConfig:
    """Each parallel worker passes its own isolated inp_path copy."""
    p = inp_path if inp_path is not None else INP_PATH
    return EstimatorConfig(
        inp_path=p,
        measurements_path=p,
        output_dir=Path("_ekf_recon_tmp"),
        plots_dir=Path("_ekf_recon_tmp/plots"),
    )

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)

def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    n = min(len(y_true), len(y_pred))
    yt, yp = y_true[:n], y_pred[:n]
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2   = r2_score(yt, yp)
    return mae, rmse, r2


# ─────────────────────────────────────────────────────────────────────────────
# EKF RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_ekf_scenario(measurements_df: pd.DataFrame,
                     config: EstimatorConfig,
                     metadata: ModelMetadata,
                     hydraulic: HydraulicInterface):
    initial_snap  = hydraulic.simulate_snapshot(config.initial_demands, timestamp_seconds=0)
    initial_state = build_initial_state(initial_snap.head_state_vector(metadata), config)
    meas_noise    = hydraulic.build_measurement_noise()
    ekf = ExtendedKalmanFilter(
        initial_state=initial_state,
        initial_covariance=config.initial_covariance,
        process_noise=config.process_noise,
        measurement_noise=meas_noise,
        config=config,
    )
    dc = len(metadata.demand_nodes)
    prows, frows = [], []

    for _, row in measurements_df.iterrows():
        ts   = float(row["timestamp_s"])
        meas = np.array([float(row["P4"]), float(row["Q1a"]), float(row["Q3a"])], dtype=float)
        cache: dict = {}

        def hyd_resp(d):
            s = hydraulic.simulate_snapshot(d, timestamp_seconds=ts)
            return np.concatenate([s.head_state_vector(metadata), s.measurement_vector(metadata)])

        def trans_fn(state):
            d = state[dc:]; r = hyd_resp(d)
            return np.concatenate([r[:dc], d])

        def meas_fn(state):
            return hyd_resp(state[dc:])[dc:]

        def trans_jac(state):
            d = state[dc:]; key = tuple(d)
            J = cache.get(key)
            if J is None:
                J = numerical_jacobian(hyd_resp, d, config); cache[key] = J
            F = np.zeros((config.state_size, config.state_size))
            F[:dc, dc:] = J[:dc, :]; F[dc:, dc:] = np.eye(dc)
            return F

        def meas_jac(state):
            d = state[dc:]; key = tuple(d)
            J = cache.get(key)
            if J is None:
                J = numerical_jacobian(hyd_resp, d, config); cache[key] = J
            H = np.zeros((config.measurement_size, config.state_size))
            H[:, dc:] = J[dc:, :]; return H

        try:
            ekf.step(measurement=meas,
                     transition_function=trans_fn,
                     measurement_function=meas_fn,
                     transition_jacobian_function=trans_jac,
                     measurement_jacobian_function=meas_jac)
        except Exception:
            pass

        upd_snap = hydraulic.simulate_snapshot(ekf.x[dc:], timestamp_seconds=ts)
        ekf.x[:dc] = upd_snap.head_state_vector(metadata)

        pr = {"timestamp": ts}
        for name in metadata.all_report_nodes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pr[name] = float(upd_snap.pressures.loc[name])
        prows.append(pr)

        fr = {"timestamp": ts}
        for name in metadata.report_flow_links:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fr[name] = float(upd_snap.flows.loc[name])
        frows.append(fr)

    return pd.DataFrame(prows), pd.DataFrame(frows)


def extract_recon(pressures_df: pd.DataFrame,
                  flows_df: pd.DataFrame) -> dict[str, np.ndarray]:
    recon = {}
    for sensor, col in PRESSURE_COL_MAP.items():
        recon[sensor] = pressures_df[col].values.astype(np.float32)
    for sensor, col in FLOW_COL_MAP.items():
        recon[sensor] = np.abs(flows_df[col].values.astype(np.float32))
    return recon


# ─────────────────────────────────────────────────────────────────────────────
# PER-SCENARIO WORKER
# ─────────────────────────────────────────────────────────────────────────────
def _process_scenario(folder: Path, inp_path: Path | None = None) -> dict | None:
    try:
        import json
        cfg       = make_config(inp_path)
        metadata  = extract_model_metadata(cfg)
        hydraulic = HydraulicInterface(cfg, metadata)

        data_df = pd.read_csv(folder / "data.csv")
        with open(folder / "labels.json", encoding="utf-8") as f:
            labels = json.load(f)

        label_det  = int(labels.get("label_detection", 0))
        label_pipe = int(labels.get("label_pipe", -1))
        group = "no-leak" if label_det == 0 else f"pipe-{label_pipe}"

        meas_df = pd.DataFrame({
            "timestamp_s": data_df["t"].values * 60.0,
            "P4":  data_df["P4"].values,
            "Q1a": data_df["Q1a"].values,
            "Q3a": data_df["Q3a"].values,
        })

        pressures_df, flows_df = run_ekf_scenario(meas_df, cfg, metadata, hydraulic)
        recon = extract_recon(pressures_df, flows_df)

        row = {
            "scenario_id":     folder.name,
            "label_detection": label_det,
            "label_pipe":      label_pipe,
            "group":           group,
        }
        for sensor in UNMONITORED:
            y_true = data_df[sensor].values.astype(np.float32)
            y_pred = recon[sensor]
            mae, rmse, r2 = metrics(y_true, y_pred)
            row[f"{sensor}_mae"]  = round(mae,  6)
            row[f"{sensor}_rmse"] = round(rmse, 6)
            row[f"{sensor}_r2"]   = round(r2,   6)

        return {"row": row}
    except Exception as exc:
        return {"error": str(exc), "scenario_id": folder.name}


def _worker(folder_str: str) -> dict | None:
    if str(_EKF_DIR) not in sys.path:
        sys.path.insert(0, str(_EKF_DIR))
    # Each worker process gets its own temp dir. WNTR writes temp.inp/.bin/.rpt
    # relative to CWD, so we chdir there to avoid cross-process collisions.
    tmp_dir = Path(tempfile.mkdtemp(prefix="ekf_recon_"))
    try:
        local_inp = tmp_dir / "base3.inp"
        shutil.copy2(str(INP_PATH), str(local_inp))
        os.chdir(str(tmp_dir))
        return _process_scenario(Path(folder_str), inp_path=local_inp)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTING + SAVING
# ─────────────────────────────────────────────────────────────────────────────
def group_summary(df: pd.DataFrame, group_label: str) -> dict:
    row = {"group": group_label, "n": len(df)}
    for sensor in UNMONITORED:
        row[f"{sensor}_mae"]  = round(float(df[f"{sensor}_mae"].mean()),  4)
        row[f"{sensor}_rmse"] = round(float(df[f"{sensor}_rmse"].mean()), 4)
        row[f"{sensor}_r2"]   = round(float(df[f"{sensor}_r2"].mean()),   4)
    all_maes = df[[f"{s}_mae" for s in UNMONITORED]].values.flatten()
    row["overall_mae"] = round(float(all_maes.mean()), 4)
    return row


def print_group(name: str, df: pd.DataFrame):
    print(f"\n  {name}  (n={len(df)})")
    print(f"    {'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}")
    print("    " + "-" * 40)
    for s in UNMONITORED:
        print(f"    {s:<8}  {df[f'{s}_mae'].mean():>10.4f}  "
              f"{df[f'{s}_rmse'].mean():>10.4f}  {df[f'{s}_r2'].mean():>8.4f}")
    all_mae = df[[f"{s}_mae" for s in UNMONITORED]].values.flatten().mean()
    print(f"    {'OVERALL':<8}  {all_mae:>10.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EKF reconstruction evaluation (all test_dataset)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default: 1)")
    args = parser.parse_args()

    if not INP_PATH.exists():
        raise FileNotFoundError(f"EPANET model not found: {INP_PATH}")
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    folders  = []
    for scn_id in manifest["scenario_id"].values:
        p = DATASET_ROOT / f"scenario_{int(scn_id):05d}"
        if (p / "data.csv").exists() and (p / "labels.json").exists():
            folders.append(p)

    print(f"EKF model      : {INP_PATH}")
    print(f"Test scenarios : {len(folders)}")
    print(f"Workers        : {args.workers}")
    print(f"Unmonitored    : {UNMONITORED}\n")
    print("Starting evaluation. This will take significant time on CPU.\n")

    rows      = []
    failed    = 0
    completed = 0
    total     = len(folders)

    def collect(result: dict):
        nonlocal failed, completed
        completed += 1
        if result is None or "error" in result:
            scn = (result or {}).get("scenario_id", "unknown")
            err = (result or {}).get("error", "None returned")
            print(f"  [WARN] {scn}: {err}")
            failed += 1
            return
        rows.append(result["row"])
        if completed % 50 == 0 or completed == total:
            print(f"  Progress: {completed}/{total}  ({failed} failed)")

    if args.workers <= 1:
        for folder in folders:
            collect(_process_scenario(folder))
    else:
        folder_strs = [str(f) for f in folders]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, fs): fs for fs in folder_strs}
            for fut in as_completed(futures):
                try:
                    collect(fut.result())
                except Exception as exc:
                    print(f"  [WARN] Worker exception: {exc}")
                    failed += 1; completed += 1

    if not rows:
        print("No results to save."); return

    df = pd.DataFrame(rows)

    # ── Save per-scenario CSV ────────────────────────────────────────────────
    per_scn_path = RESULTS_DIR / "ekf_recon_per_scenario.csv"
    df.to_csv(per_scn_path, index=False)
    print(f"\n[OK] Per-scenario results -> {per_scn_path}")
    if failed:
        print(f"     ({failed} scenarios failed and excluded)")

    # ── Per-sensor overall summary ────────────────────────────────────────────
    sensor_rows = []
    for s in UNMONITORED:
        sensor_rows.append({
            "sensor":     s,
            "mean_mae":   round(float(df[f"{s}_mae"].mean()),  4),
            "mean_rmse":  round(float(df[f"{s}_rmse"].mean()), 4),
            "mean_r2":    round(float(df[f"{s}_r2"].mean()),   4),
            "median_mae": round(float(df[f"{s}_mae"].median()), 4),
            "std_mae":    round(float(df[f"{s}_mae"].std()),    4),
        })
    sensor_df = pd.DataFrame(sensor_rows)
    sensor_path = RESULTS_DIR / "ekf_recon_summary_sensor.csv"
    sensor_df.to_csv(sensor_path, index=False)

    # ── Grouped summary by pipe ───────────────────────────────────────────────
    group_rows = []
    for g in ["no-leak"] + [f"pipe-{p}" for p in range(1, 6)] + ["ALL"]:
        sub = df[df["group"] == g] if g != "ALL" else df
        if len(sub) == 0:
            continue
        group_rows.append(group_summary(sub, g))
    group_df = pd.DataFrame(group_rows)
    group_path = RESULTS_DIR / "ekf_recon_summary_pipe.csv"
    group_df.to_csv(group_path, index=False)
    print(f"[OK] Sensor summary      -> {sensor_path}")
    print(f"[OK] Pipe-group summary  -> {group_path}")

    # ── Console report ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EKF RECONSTRUCTION EVALUATION — test_dataset (all scenarios)")
    print("=" * 65)

    print("\n--- Per-Sensor Overall (mean across all scenarios) ---")
    print(f"  {'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}  {'Median MAE':>11}")
    print("  " + "-" * 48)
    for _, sr in sensor_df.iterrows():
        print(f"  {sr['sensor']:<8}  {sr['mean_mae']:>10.4f}  {sr['mean_rmse']:>10.4f}  "
              f"{sr['mean_r2']:>8.4f}  {sr['median_mae']:>11.4f}")
    overall_mae = float(df[[f"{s}_mae" for s in UNMONITORED]].values.mean())
    print(f"\n  Overall mean MAE (all sensors, all scenarios): {overall_mae:.4f}")

    print("\n--- Grouped by Pipe ---")
    for g in ["no-leak"] + [f"pipe-{p}" for p in range(1, 6)] + ["ALL"]:
        sub = df[df["group"] == g] if g != "ALL" else df
        if len(sub) > 0:
            print_group(g, sub)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
