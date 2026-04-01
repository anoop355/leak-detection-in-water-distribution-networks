"""
pipeline_ekf_stgcn.py

Full pipeline evaluation: EKF reconstruction -> ST-GCN S10-A detection/localisation.

On each of the 12 focused test_dataset scenarios:
  1. EKF reconstructs the 7 unmonitored sensors from [P4, Q1a, Q3a]
  2. Reconstructed + actual monitored signals form the full 10-sensor array
  3. ST-GCN S10-A predicts: leak/no-leak, pipe location, leak size, position
  4. Predictions are compared against ground-truth labels

Outputs:
  pipeline_ekf_stgcn_results.csv
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── EKF module path ───────────────────────────────────────────────────────────
_EKF_DIR = Path(__file__).resolve().parent / "ekf_wdn_project"
if str(_EKF_DIR) not in sys.path:
    sys.path.insert(0, str(_EKF_DIR))

from config import EstimatorConfig
from ekf import ExtendedKalmanFilter
from hydraulic_interface import HydraulicInterface
from jacobians import numerical_jacobian
from load_model import ModelMetadata, build_initial_state, extract_model_metadata

logging.basicConfig(level=logging.WARNING)
logging.getLogger("wntr").setLevel(logging.ERROR)
logging.getLogger("wntr.epanet.io").setLevel(logging.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
STGCN_BUNDLE = "stgcn_placement_bundles/stgcn_bundle_S10-A.pt"
INP_PATH     = _EKF_DIR / "base3.inp"
DATASET_ROOT = "test_dataset/scenarios"
MANIFEST     = "test_dataset/manifests/manifest.csv"
OUT_CSV      = "pipeline_ekf_stgcn_results.csv"

ALL_SENSORS = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
MONITORED   = ["P4", "Q1a", "Q3a"]
UNMONITORED = ["P2", "P3", "P5", "P6", "Q2a", "Q4a", "Q5a"]

# EKF column mappings
PRESSURE_COL_MAP = {"P2": "2", "P3": "3", "P5": "5", "P6": "6"}
FLOW_COL_MAP     = {"Q2a": "2a", "Q4a": "4a", "Q5a": "5a"}

N_NOLEAK_PICK = 2
N_LEAK_PICK   = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# ST-GCN MODEL  (identical to train_stgcn_sensor_placement.py)
# ─────────────────────────────────────────────────────────────────────────────
PIPE_CLASSES  = 6
SIZE_CLASSES  = 4
PIPE_NONE_IDX = 5

class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (1, kernel_size),
                              padding=(0, pad), dilation=(1, dilation))
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.act(self.bn(self.conv(x)))
        return x.permute(0, 3, 2, 1)


class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.einsum("ij,btjc->btic", self.A, x)
        return self.act(self.ln(self.lin(x)))


class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp     = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph    = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout  = nn.Dropout(dropout)
        self.out_act  = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.res_proj(x)
        y = self.dropout(self.graph(self.temp(x)))
        return self.out_act(y + r)


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * num_nodes, 1)

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat  = x.reshape(B, T, N * C)
        weights = torch.softmax(self.attn(x_flat), dim=1)
        return (x_flat * weights).sum(dim=1)


class SingleLeakSTGCN(nn.Module):
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, 4, dropout)
        head_in     = num_nodes * hidden_2
        head_hidden = 64
        self.temporal_pool = TemporalAttentionPool(hidden_2, num_nodes)

        def _head(out_size):
            return nn.Sequential(
                nn.Linear(head_in, head_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, out_size)
            )

        self.detect_head = _head(2)
        self.pipe_head   = _head(PIPE_CLASSES)
        self.size_head   = _head(SIZE_CLASSES)
        self.pos_head    = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        z = self.temporal_pool(x)
        return (self.detect_head(z), self.pipe_head(z),
                self.size_head(z), self.pos_head(z).squeeze(1))


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def select_scenarios(manifest_path, dataset_root):
    df = pd.read_csv(manifest_path)
    selected = []
    for _, row in df[df["label_detection"] == 0].head(N_NOLEAK_PICK).iterrows():
        folder = os.path.join(dataset_root, f"scenario_{int(row['scenario_id']):05d}")
        if os.path.isfile(os.path.join(folder, "data.csv")):
            selected.append({"folder": folder, "scenario_id": int(row["scenario_id"]),
                             "label_detection": 0, "label_pipe": -1, "group": "no-leak"})
    for pipe in range(1, 6):
        pipe_rows = df[(df["label_detection"] == 1) & (df["label_pipe"] == pipe)].head(N_LEAK_PICK)
        for _, row in pipe_rows.iterrows():
            folder = os.path.join(dataset_root, f"scenario_{int(row['scenario_id']):05d}")
            if os.path.isfile(os.path.join(folder, "data.csv")):
                selected.append({"folder": folder, "scenario_id": int(row["scenario_id"]),
                                 "label_detection": 1, "label_pipe": pipe,
                                 "group": f"pipe-{pipe}"})
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# EKF
# ─────────────────────────────────────────────────────────────────────────────
def make_ekf_config():
    return EstimatorConfig(inp_path=INP_PATH, measurements_path=INP_PATH,
                           output_dir=Path("_pipeline_tmp"),
                           plots_dir=Path("_pipeline_tmp/plots"))

def run_ekf_scenario(measurements_df, config, metadata, hydraulic):
    initial_snap  = hydraulic.simulate_snapshot(config.initial_demands, timestamp_seconds=0)
    initial_state = build_initial_state(initial_snap.head_state_vector(metadata), config)
    meas_noise    = hydraulic.build_measurement_noise()
    ekf = ExtendedKalmanFilter(initial_state=initial_state,
                               initial_covariance=config.initial_covariance,
                               process_noise=config.process_noise,
                               measurement_noise=meas_noise, config=config)
    dc = len(metadata.demand_nodes)
    prows, frows = [], []

    for _, row in measurements_df.iterrows():
        ts  = float(row["timestamp_s"])
        meas = np.array([float(row["P4"]), float(row["Q1a"]), float(row["Q3a"])], dtype=float)
        cache = {}

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
            if J is None: J = numerical_jacobian(hyd_resp, d, config); cache[key] = J
            F = np.zeros((config.state_size, config.state_size))
            F[:dc, dc:] = J[:dc, :]; F[dc:, dc:] = np.eye(dc)
            return F

        def meas_jac(state):
            d = state[dc:]; key = tuple(d)
            J = cache.get(key)
            if J is None: J = numerical_jacobian(hyd_resp, d, config); cache[key] = J
            H = np.zeros((config.measurement_size, config.state_size))
            H[:, dc:] = J[dc:, :]; return H

        try:
            ekf.step(measurement=meas, transition_function=trans_fn,
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


def ekf_to_full_array(data_df, pressures_df, flows_df, sensor_names):
    """
    Build (T, N) array: actual values for monitored sensors,
    EKF-reconstructed values for unmonitored sensors.
    """
    T = len(data_df)
    N = len(sensor_names)
    out = np.zeros((T, N), dtype=np.float32)
    idx = {s: i for i, s in enumerate(sensor_names)}

    # Actual monitored
    for s in MONITORED:
        out[:, idx[s]] = data_df[s].values.astype(np.float32)

    # EKF unmonitored
    for sensor, col in PRESSURE_COL_MAP.items():
        if col in pressures_df.columns:
            out[:, idx[sensor]] = pressures_df[col].values[:T].astype(np.float32)
    for sensor, col in FLOW_COL_MAP.items():
        if col in flows_df.columns:
            out[:, idx[sensor]] = np.abs(flows_df[col].values[:T].astype(np.float32))

    return out   # (T, N)


# ─────────────────────────────────────────────────────────────────────────────
# STGCN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def make_stgcn_features(raw_signals, baseline_template, mu, sigma, window):
    """
    raw_signals      : (T, N) float32
    baseline_template: (W, N) float32  (window-length template)
    Returns normalised tensor (1, W, N, 2) ready for model input.
    """
    T = raw_signals.shape[0]
    W = min(window, T)
    raw_win  = raw_signals[:W]                      # (W, N)
    base_win = baseline_template[:W]                # (W, N)
    dev_win  = raw_win - base_win                   # (W, N)
    feats    = np.stack([raw_win, dev_win], axis=-1).astype(np.float32)  # (W, N, 2)
    feats    = (feats - mu[None]) / (sigma[None] + 1e-8)
    return torch.tensor(feats[None], dtype=torch.float32, device=DEVICE)  # (1, W, N, 2)


@torch.no_grad()
def predict_stgcn(model, x_tensor):
    model.eval()
    det_logits, pipe_logits, size_logits, pos_pred = model(x_tensor)
    det_pred  = int(det_logits.argmax(dim=1).item())
    pipe_pred = int(pipe_logits.argmax(dim=1).item())
    size_pred = int(size_logits.argmax(dim=1).item())
    pos_pred  = float(pos_pred.item())
    det_prob  = float(torch.softmax(det_logits, dim=1)[0, 1].item())
    return det_pred, pipe_pred, size_pred, pos_pred, det_prob


def decode_predictions(det_pred, pipe_pred):
    """Return (predicted_detection, predicted_pipe_number_1indexed_or_None)."""
    pred_pipe = None if det_pred == 0 else (pipe_pred + 1 if pipe_pred < 5 else None)
    return det_pred, pred_pipe


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.isfile(STGCN_BUNDLE):
        raise FileNotFoundError(f"STGCN bundle not found: {STGCN_BUNDLE}")
    if not INP_PATH.exists():
        raise FileNotFoundError(f"EPANET model not found: {INP_PATH}")

    # ── Load STGCN bundle ────────────────────────────────────────────────────
    print(f"Loading STGCN bundle: {STGCN_BUNDLE}")
    stgcn_bundle = torch.load(STGCN_BUNDLE, map_location="cpu", weights_only=False)

    stgcn_adj       = stgcn_bundle["adjacency"]
    stgcn_mu        = stgcn_bundle["mu"]                # (N, 2)
    stgcn_sigma     = stgcn_bundle["sigma"]             # (N, 2)
    stgcn_baseline  = stgcn_bundle["baseline_template"] # (W, N)
    stgcn_sensors   = stgcn_bundle["sensor_names"]
    stgcn_window    = stgcn_bundle["window"]
    hidden_1        = stgcn_bundle["hidden_1"]
    hidden_2        = stgcn_bundle["hidden_2"]
    kernel_size     = stgcn_bundle["kernel_size"]
    dropout         = stgcn_bundle["dropout"]
    node_feats      = stgcn_bundle["node_feats"]
    num_nodes       = len(stgcn_sensors)

    stgcn_model = SingleLeakSTGCN(
        stgcn_adj, num_nodes, hidden_1, hidden_2, kernel_size, dropout, node_feats
    ).to(DEVICE)
    stgcn_model.load_state_dict(stgcn_bundle["model_state_dict"])
    stgcn_model.eval()
    print(f"STGCN model loaded. Sensors: {stgcn_sensors}")

    # ── Build EKF objects ────────────────────────────────────────────────────
    print(f"Initialising EKF (EPANET model: {INP_PATH})")
    ekf_cfg  = make_ekf_config()
    metadata = extract_model_metadata(ekf_cfg)
    hydraulic = HydraulicInterface(ekf_cfg, metadata)

    # ── Select 12 scenarios ──────────────────────────────────────────────────
    scenarios = select_scenarios(MANIFEST, DATASET_ROOT)
    print(f"\nSelected {len(scenarios)} scenarios\n")

    rows = []

    for idx, scn in enumerate(scenarios, 1):
        folder = scn["folder"]
        scn_id = scn["scenario_id"]
        group  = scn["group"]
        true_det  = scn["label_detection"]
        true_pipe = scn["label_pipe"]    # -1 for no-leak, 1-5 for leak

        print(f"[{idx}/{len(scenarios)}] Scenario {scn_id:05d}  ({group})")

        data_df = pd.read_csv(os.path.join(folder, "data.csv"))

        # ── EKF reconstruction ───────────────────────────────────────────────
        meas_df = pd.DataFrame({
            "timestamp_s": data_df["t"].values * 60.0,
            "P4":  data_df["P4"].values,
            "Q1a": data_df["Q1a"].values,
            "Q3a": data_df["Q3a"].values,
        })

        try:
            pressures_df, flows_df = run_ekf_scenario(meas_df, ekf_cfg, metadata, hydraulic)
            full_raw = ekf_to_full_array(data_df, pressures_df, flows_df, stgcn_sensors)
            ekf_ok   = True
        except Exception as exc:
            print(f"  [ERROR] EKF failed: {exc}")
            full_raw = np.zeros((len(data_df), len(stgcn_sensors)), dtype=np.float32)
            ekf_ok   = False

        # ── STGCN inference ──────────────────────────────────────────────────
        x_tensor = make_stgcn_features(full_raw, stgcn_baseline, stgcn_mu, stgcn_sigma,
                                       stgcn_window)
        det_pred, pipe_pred_raw, size_pred, pos_pred, det_prob = predict_stgcn(
            stgcn_model, x_tensor)
        pred_det, pred_pipe = decode_predictions(det_pred, pipe_pred_raw)

        # ── Evaluate ─────────────────────────────────────────────────────────
        det_correct  = (pred_det == true_det)
        pipe_correct = False
        if true_det == 1:   # only meaningful for leak scenarios
            pipe_correct = (pred_pipe == true_pipe)

        print(f"  True : det={true_det}  pipe={true_pipe if true_det else 'N/A'}")
        print(f"  Pred : det={pred_det} (p={det_prob:.2f})  pipe={pred_pipe}")
        mark = "OK" if det_correct else "WRONG"
        if true_det == 1:
            mark += f"  pipe={'OK' if pipe_correct else 'WRONG'}"
        print(f"  Result: [{mark}]")

        SIZE_MAP = {0: "S", 1: "M", 2: "L", 3: "none"}
        rows.append({
            "scenario_id":      scn_id,
            "group":            group,
            "true_detection":   true_det,
            "true_pipe":        true_pipe,
            "pred_detection":   pred_det,
            "pred_pipe":        pred_pipe,
            "pred_size":        SIZE_MAP.get(size_pred, "?"),
            "pred_position":    round(pos_pred, 4),
            "det_prob_leak":    round(det_prob, 4),
            "det_correct":      det_correct,
            "pipe_correct":     pipe_correct,
            "ekf_ok":           ekf_ok,
        })

    # ── Save results ─────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Results saved -> {OUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS: EKF -> ST-GCN S10-A")
    print("=" * 60)

    print(f"\n{'ScnID':>8}  {'Group':<10}  {'TrueDet':>7}  {'PredDet':>7}  "
          f"{'TruePipe':>8}  {'PredPipe':>8}  {'DetOK':>6}  {'PipeOK':>6}")
    print("-" * 75)
    for r in rows:
        print(f"{r['scenario_id']:>8}  {r['group']:<10}  {r['true_detection']:>7}  "
              f"{r['pred_detection']:>7}  "
              f"{str(r['true_pipe']):>8}  {str(r['pred_pipe']):>8}  "
              f"{'yes' if r['det_correct'] else 'no':>6}  "
              f"{'yes' if r['pipe_correct'] else ('N/A' if r['true_detection']==0 else 'no'):>6}")

    det_acc   = np.mean([r["det_correct"] for r in rows])
    noleak_rows = [r for r in rows if r["true_detection"] == 0]
    leak_rows   = [r for r in rows if r["true_detection"] == 1]
    noleak_acc  = np.mean([r["det_correct"] for r in noleak_rows]) if noleak_rows else float("nan")
    leak_recall = np.mean([r["det_correct"] for r in leak_rows])   if leak_rows   else float("nan")
    pipe_acc    = np.mean([r["pipe_correct"] for r in leak_rows])  if leak_rows   else float("nan")

    print(f"\nDetection accuracy  (all 12) : {det_acc:.2%}  "
          f"({sum(r['det_correct'] for r in rows)}/{len(rows)})")
    print(f"No-leak specificity (n={len(noleak_rows):2d}) : {noleak_acc:.2%}")
    print(f"Leak recall         (n={len(leak_rows):2d}) : {leak_recall:.2%}")
    print(f"Pipe accuracy       (n={len(leak_rows):2d}) : {pipe_acc:.2%}  "
          f"(among all leak scenarios)")
    print(f"Pipe acc (det=correct, n={sum(r['det_correct'] and r['true_detection']==1 for r in rows):2d}) "
          f": {np.mean([r['pipe_correct'] for r in leak_rows if r['det_correct']]):.2%}"
          if any(r['det_correct'] and r['true_detection']==1 for r in rows) else "")

    print("\nPer-group pipe accuracy:")
    for pipe in range(1, 6):
        grp = [r for r in rows if r["group"] == f"pipe-{pipe}"]
        if grp:
            pacc = np.mean([r["pipe_correct"] for r in grp])
            print(f"  pipe-{pipe}: {pacc:.2%}  "
                  f"({sum(r['pipe_correct'] for r in grp)}/{len(grp)})")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
