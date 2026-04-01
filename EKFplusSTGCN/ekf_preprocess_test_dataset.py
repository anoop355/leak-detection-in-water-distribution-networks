"""
ekf_preprocess_test_dataset.py
================================
Preprocesses test_dataset through the EKF -> test_dataset_ekf/
(mirrors ekf_preprocess_stgcn_dataset.py but uses test_dataset as source).

Output per scenario (test_dataset_ekf/scenarios/scenario_XXXXX/):
  data.csv   — t, P2..Q5a (actual monitored + EKF-reconstructed) + inn_P4, inn_Q1a, inn_Q3a
  labels.json — copied from source

Usage:
    python EKFplusSTGCN/ekf_preprocess_test_dataset.py
    python EKFplusSTGCN/ekf_preprocess_test_dataset.py --workers 4
    python EKFplusSTGCN/ekf_preprocess_test_dataset.py --workers 4 --resume
"""

from __future__ import annotations

import argparse
import json
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

_HERE    = Path(__file__).resolve().parent
_ROOT    = _HERE.parent
_EKF_DIR = _ROOT / "ekf_wdn_project"
for _p in [str(_EKF_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import EstimatorConfig
from ekf import ExtendedKalmanFilter
from hydraulic_interface import HydraulicInterface
from jacobians import numerical_jacobian
from load_model import build_initial_state, extract_model_metadata

logging.basicConfig(level=logging.WARNING)
logging.getLogger("wntr").setLevel(logging.ERROR)
logging.getLogger("wntr.epanet.io").setLevel(logging.ERROR)

INP_PATH     = _EKF_DIR / "base3.inp"
SRC_ROOT     = _ROOT / "test_dataset"
DST_ROOT     = _ROOT / "test_dataset_ekf"
SRC_MANIFEST = SRC_ROOT / "manifests" / "manifest.csv"

MONITORED        = ["P4", "Q1a", "Q3a"]
ALL_SENSORS      = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
PRESSURE_COL_MAP = {"P2": "2", "P3": "3", "P5": "5", "P6": "6"}
FLOW_COL_MAP     = {"Q2a": "2a", "Q4a": "4a", "Q5a": "5a"}


def make_config(inp_path: Path) -> EstimatorConfig:
    return EstimatorConfig(
        inp_path=inp_path,
        measurements_path=inp_path,
        output_dir=inp_path.parent / "_tmp_out",
        plots_dir=inp_path.parent / "_tmp_plots",
    )


def run_ekf_scenario(measurements_df, config, metadata, hydraulic):
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
    prows, frows, irows = [], [], []

    for _, row in measurements_df.iterrows():
        ts   = float(row["timestamp_s"])
        meas = np.array([float(row["P4"]), float(row["Q1a"]), float(row["Q3a"])], dtype=float)
        cache: dict = {}

        def hyd_resp(d, _ts=ts):
            s = hydraulic.simulate_snapshot(d, timestamp_seconds=_ts)
            return np.concatenate([s.head_state_vector(metadata), s.measurement_vector(metadata)])

        def trans_fn(state):
            d = state[dc:]; return np.concatenate([hyd_resp(d)[:dc], d])

        def meas_fn(state): return hyd_resp(state[dc:])[dc:]

        def trans_jac(state):
            d = state[dc:]; key = tuple(d)
            J = cache.get(key)
            if J is None: J = numerical_jacobian(hyd_resp, d, config); cache[key] = J
            F = np.zeros((config.state_size, config.state_size))
            F[:dc, dc:] = J[:dc, :]; F[dc:, dc:] = np.eye(dc); return F

        def meas_jac(state):
            d = state[dc:]; key = tuple(d)
            J = cache.get(key)
            if J is None: J = numerical_jacobian(hyd_resp, d, config); cache[key] = J
            H = np.zeros((config.measurement_size, config.state_size))
            H[:, dc:] = J[dc:, :]; return H

        residual = np.zeros(3, dtype=float)
        try:
            result = ekf.step(measurement=meas, transition_function=trans_fn,
                              measurement_function=meas_fn,
                              transition_jacobian_function=trans_jac,
                              measurement_jacobian_function=meas_jac)
            residual = result.residual
        except Exception:
            pass

        upd = hydraulic.simulate_snapshot(ekf.x[dc:], timestamp_seconds=ts)
        ekf.x[:dc] = upd.head_state_vector(metadata)

        pr = {"timestamp": ts}
        for name in metadata.all_report_nodes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pr[name] = float(upd.pressures.loc[name])
        prows.append(pr)

        fr = {"timestamp": ts}
        for name in metadata.report_flow_links:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fr[name] = float(upd.flows.loc[name])
        frows.append(fr)

        irows.append({"inn_P4": float(residual[0]),
                      "inn_Q1a": float(residual[1]),
                      "inn_Q3a": float(residual[2])})

    return pd.DataFrame(prows), pd.DataFrame(frows), pd.DataFrame(irows)


def _process_scenario(src_folder: Path, dst_folder: Path, inp_path: Path) -> dict:
    try:
        cfg      = make_config(inp_path)
        metadata = extract_model_metadata(cfg)
        hydraulic = HydraulicInterface(cfg, metadata)

        data_df = pd.read_csv(src_folder / "data.csv")
        with open(src_folder / "labels.json", encoding="utf-8") as f:
            labels = json.load(f)

        meas_df = pd.DataFrame({
            "timestamp_s": data_df["t"].values * 60.0,
            "P4":  data_df["P4"].values,
            "Q1a": data_df["Q1a"].values,
            "Q3a": data_df["Q3a"].values,
        })

        T = len(data_df)
        pressures_df, flows_df, innovations_df = run_ekf_scenario(
            meas_df, cfg, metadata, hydraulic)

        out = pd.DataFrame({"t": data_df["t"].values})
        for sensor, col in PRESSURE_COL_MAP.items():
            out[sensor] = pressures_df[col].values[:T] if col in pressures_df.columns else np.nan
        out["P4"]  = data_df["P4"].values
        for sensor, col in FLOW_COL_MAP.items():
            out[sensor] = np.abs(flows_df[col].values[:T]) if col in flows_df.columns else np.nan
        out["Q1a"] = data_df["Q1a"].values
        out["Q3a"] = data_df["Q3a"].values
        out["inn_P4"]  = innovations_df["inn_P4"].values[:T]
        out["inn_Q1a"] = innovations_df["inn_Q1a"].values[:T]
        out["inn_Q3a"] = innovations_df["inn_Q3a"].values[:T]

        col_order = ["t"] + ALL_SENSORS + ["inn_P4", "inn_Q1a", "inn_Q3a"]
        out = out[col_order]

        dst_folder.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst_folder / "data.csv", index=False)
        with open(dst_folder / "labels.json", "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2)

        return {"scenario_id": src_folder.name, "ok": True}

    except Exception as exc:
        return {"scenario_id": src_folder.name, "ok": False, "error": str(exc)}


def _worker(args: tuple) -> dict:
    src_str, dst_str = args
    if str(_EKF_DIR) not in sys.path:
        sys.path.insert(0, str(_EKF_DIR))
    tmp_dir = Path(tempfile.mkdtemp(prefix="ekf_test_prep_"))
    try:
        local_inp = tmp_dir / "base3.inp"
        shutil.copy2(str(INP_PATH), str(local_inp))
        os.chdir(str(tmp_dir))
        return _process_scenario(Path(src_str), Path(dst_str), local_inp)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true",
                        help="Skip scenarios whose dst folder already exists")
    args = parser.parse_args()

    if not INP_PATH.exists():
        raise FileNotFoundError(f"EKF model not found: {INP_PATH}")
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC_ROOT}")

    manifest = pd.read_csv(SRC_MANIFEST)
    src_scenarios_dir = SRC_ROOT / "scenarios"
    dst_scenarios_dir = DST_ROOT / "scenarios"

    work = []
    for scn_id in manifest["scenario_id"].values:
        src = src_scenarios_dir / f"scenario_{int(scn_id):05d}"
        dst = dst_scenarios_dir / f"scenario_{int(scn_id):05d}"
        if not (src / "data.csv").exists():
            continue
        if args.resume and (dst / "data.csv").exists():
            continue
        work.append((str(src), str(dst)))

    # Copy manifests
    dst_manifests = DST_ROOT / "manifests"
    dst_manifests.mkdir(parents=True, exist_ok=True)
    for mf in (SRC_ROOT / "manifests").glob("*.csv"):
        shutil.copy2(str(mf), str(dst_manifests / mf.name))

    print(f"Source : {SRC_ROOT}")
    print(f"Output : {DST_ROOT}")
    print(f"Scenarios to process : {len(work)}  ({'resume' if args.resume else 'full run'})")
    print(f"Workers : {args.workers}")
    print("Starting...\n")

    completed = 0; failed = 0; total = len(work)

    def collect(r):
        nonlocal completed, failed
        completed += 1
        if not r.get("ok"):
            print(f"  [WARN] {r['scenario_id']}: {r.get('error', '?')}")
            failed += 1
        if completed % 100 == 0 or completed == total:
            print(f"  Progress: {completed}/{total}  ({failed} failed)")

    if args.workers <= 1:
        for src_str, dst_str in work:
            tmp_dir = Path(tempfile.mkdtemp(prefix="ekf_test_prep_"))
            try:
                local_inp = tmp_dir / "base3.inp"
                shutil.copy2(str(INP_PATH), str(local_inp))
                collect(_process_scenario(Path(src_str), Path(dst_str), local_inp))
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_worker, item): item for item in work}
            for fut in as_completed(futures):
                try:
                    collect(fut.result())
                except Exception as exc:
                    print(f"  [WARN] Worker exception: {exc}")
                    failed += 1; completed += 1

    print(f"\nDone. {completed - failed}/{total} scenarios preprocessed successfully.")
    print(f"Output: {DST_ROOT}")


if __name__ == "__main__":
    main()
