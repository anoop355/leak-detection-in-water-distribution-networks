"""
- Feed [P4, Q1a, Q3a] to the EKF
- EKF reconstructs [P2, P3, P5, P6, Q2a, Q4a, Q5a]
- Combine reconstructed + actual monitored into full 10-sensor array
- Build 2-channel STGCN features [raw, deviation_from_baseline]
- ST-GCN S10-A predicts: detection (0/1), pipe (1-5 or none), size, position
- Compare against scenario ground-truth labels
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
import torch
import torch.nn as nn

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
STGCN_BUNDLE = _ROOT / "stgcn_placement_bundles" / "stgcn_bundle_S10-A.pt"
DATASET_ROOT = _ROOT / "test_dataset" / "scenarios"
MANIFEST     = _ROOT / "test_dataset" / "manifests" / "manifest.csv"
INP_PATH     = _EKF_DIR / "base3.inp"
RESULTS_DIR  = _HERE / "results"

ALL_SENSORS = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
MONITORED   = ["P4", "Q1a", "Q3a"]
UNMONITORED = ["P2", "P3", "P5", "P6", "Q2a", "Q4a", "Q5a"]
PRESSURE_COL_MAP = {"P2": "2", "P3": "3", "P5": "5", "P6": "6"}
FLOW_COL_MAP     = {"Q2a": "2a", "Q4a": "4a", "Q5a": "5a"}

NUM_PIPES     = 5
PIPE_NONE_IDX = 5
PIPE_CLASSES  = 6
SIZE_CLASSES  = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# ST-GCN MODEL  (SingleLeakSTGCN — must match train_stgcn_sensor_placement.py)
# ─────────────────────────────────────────────────────────────────────────────
class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (1, kernel_size),
                              padding=(0, pad), dilation=(1, dilation))
        self.bn  = nn.BatchNorm2d(out_ch); self.act = nn.ReLU()
    def forward(self, x):
        x = x.permute(0, 3, 2, 1); x = self.act(self.bn(self.conv(x)))
        return x.permute(0, 3, 2, 1)

class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch); self.ln = nn.LayerNorm(out_ch); self.act = nn.ReLU()
    def forward(self, x):
        x = torch.einsum("ij,btjc->btic", self.A, x)
        return self.act(self.ln(self.lin(x)))

class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp     = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph    = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout  = nn.Dropout(dropout); self.out_act = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        r = self.res_proj(x); y = self.dropout(self.graph(self.temp(x)))
        return self.out_act(y + r)

class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * num_nodes, 1)
    def forward(self, x):
        B, T, N, C = x.shape; x_flat = x.reshape(B, T, N * C)
        return (x_flat * torch.softmax(self.attn(x_flat), dim=1)).sum(dim=1)

class SingleLeakSTGCN(nn.Module):
    def __init__(self, adj, num_nodes, h1=16, h2=32, ks=5, drop=0.25, nf=2):
        super().__init__()
        self.block1 = STBlock(nf, h1, adj, ks, 1, drop)
        self.block2 = STBlock(h1, h2, adj, ks, 2, drop)
        self.block3 = STBlock(h2, h2, adj, ks, 4, drop)
        hi = num_nodes * h2; hh = 64
        self.temporal_pool = TemporalAttentionPool(h2, num_nodes)
        def _h(o): return nn.Sequential(nn.Linear(hi, hh), nn.ReLU(), nn.Dropout(drop), nn.Linear(hh, o))
        self.detect_head = _h(2); self.pipe_head = _h(PIPE_CLASSES)
        self.size_head   = _h(SIZE_CLASSES)
        self.pos_head    = nn.Sequential(nn.Linear(hi, hh), nn.ReLU(), nn.Dropout(drop),
                                         nn.Linear(hh, 1), nn.Sigmoid())
    def forward(self, x):
        x = self.block3(self.block2(self.block1(x))); z = self.temporal_pool(x)
        return self.detect_head(z), self.pipe_head(z), self.size_head(z), self.pos_head(z).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS HELPERS  (matching evaluate_stgcn_model1.py style)
# ─────────────────────────────────────────────────────────────────────────────
def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2*p*r / (p+r) if (p+r) > 0 else 0.0
    return round(p, 4), round(r, 4), round(f, 4)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12: return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)

def compute_detection_metrics(true_det, pred_det):
    """Binary detection: accuracy, precision, recall, F1, confusion matrix."""
    true_arr = np.array(true_det, dtype=int)
    pred_arr = np.array(pred_det, dtype=int)
    acc = float(np.mean(true_arr == pred_arr))
    tp = int(np.sum((true_arr == 1) & (pred_arr == 1)))
    fp = int(np.sum((true_arr == 0) & (pred_arr == 1)))
    fn = int(np.sum((true_arr == 1) & (pred_arr == 0)))
    tn = int(np.sum((true_arr == 0) & (pred_arr == 0)))
    p, r, f1 = prf(tp, fp, fn)
    conf = np.array([[tn, fp], [fn, tp]], dtype=int)
    conf_norm = conf / conf.sum(axis=1, keepdims=True).clip(min=1)
    return {
        "accuracy": round(acc, 4), "precision": p, "recall": r, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "specificity": round(tn / (tn + fp) if (tn + fp) > 0 else 0.0, 4),
        "n_total": len(true_arr),
        "n_leak": int(np.sum(true_arr == 1)),
        "n_noleak": int(np.sum(true_arr == 0)),
    }, conf, conf_norm

def compute_pipe_metrics(per_rows):
    """Per-pipe TP/FP/FN for leak scenarios where detection was correct."""
    pipe_stats = {pid: {"tp": 0, "fp": 0, "fn": 0, "conf": {}} for pid in range(1, 6)}
    for r in per_rows:
        if r["true_detection"] != 1:
            continue
        true_p = r["true_pipe"]
        pred_p = r["pred_pipe"]   # None if predicted no-leak
        # Update confusion cell
        if pred_p is not None and 1 <= pred_p <= 5:
            pipe_stats[true_p]["conf"][pred_p] = (
                pipe_stats[true_p]["conf"].get(pred_p, 0) + 1)
        if pred_p == true_p:
            pipe_stats[true_p]["tp"] += 1
        else:
            pipe_stats[true_p]["fn"] += 1
            if pred_p is not None and 1 <= pred_p <= 5:
                pipe_stats[pred_p]["fp"] += 1
    # Build summary rows
    rows = []
    f1_scores = []
    for pid in range(1, 6):
        tp = pipe_stats[pid]["tp"]; fp = pipe_stats[pid]["fp"]; fn = pipe_stats[pid]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        f1_scores.append(f1)
        rows.append({"pipe": pid, "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f1": f1,
                     "support": tp + fn})
    macro_f1 = round(float(np.mean(f1_scores)), 4)
    # 5x5 normalised confusion matrix
    raw = np.zeros((5, 5), dtype=float)
    for i, true_p in enumerate(range(1, 6)):
        for j, pred_p in enumerate(range(1, 6)):
            raw[i, j] = pipe_stats[true_p]["conf"].get(pred_p, 0)
    row_sums = raw.sum(axis=1, keepdims=True)
    pipe_conf_norm = np.where(row_sums > 0, raw / row_sums, 0.0)
    return pd.DataFrame(rows), macro_f1, raw.astype(int), pipe_conf_norm

def compute_position_metrics(per_rows):
    """MAE, RMSE, R^2 on pipe-matched (true_pipe == pred_pipe) leak scenarios."""
    pairs = [(r["true_position"], r["pred_position"])
             for r in per_rows
             if r["true_detection"] == 1 and r["pred_pipe"] == r["true_pipe"]
             and r["true_position"] is not None]
    if not pairs:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}
    yt = np.array([p[0] for p in pairs], dtype=float)
    yp = np.array([p[1] for p in pairs], dtype=float)
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2   = r2_score(yt, yp)
    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6), "n": len(pairs)}


# ─────────────────────────────────────────────────────────────────────────────
# EKF HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def make_config(inp_path: Path | None = None):
    p = inp_path if inp_path is not None else INP_PATH
    return EstimatorConfig(inp_path=p, measurements_path=p,
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
        ts   = float(row["timestamp_s"])
        meas = np.array([float(row["P4"]), float(row["Q1a"]), float(row["Q3a"])], dtype=float)
        cache: dict = {}
        def hyd_resp(d):
            s = hydraulic.simulate_snapshot(d, timestamp_seconds=ts)
            return np.concatenate([s.head_state_vector(metadata), s.measurement_vector(metadata)])
        def trans_fn(state):
            d = state[dc:]; r = hyd_resp(d); return np.concatenate([r[:dc], d])
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
        try:
            ekf.step(measurement=meas, transition_function=trans_fn,
                     measurement_function=meas_fn,
                     transition_jacobian_function=trans_jac,
                     measurement_jacobian_function=meas_jac)
        except Exception: pass
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
    return pd.DataFrame(prows), pd.DataFrame(frows)

def build_full_array(data_df, pressures_df, flows_df, sensor_names):
    """Actual monitored + EKF reconstructed -> (T, N) float32."""
    T = len(data_df); N = len(sensor_names)
    idx = {s: i for i, s in enumerate(sensor_names)}
    out = np.zeros((T, N), dtype=np.float32)
    for s in MONITORED:
        out[:, idx[s]] = data_df[s].values.astype(np.float32)
    for sensor, col in PRESSURE_COL_MAP.items():
        if col in pressures_df.columns:
            out[:, idx[sensor]] = pressures_df[col].values[:T].astype(np.float32)
    for sensor, col in FLOW_COL_MAP.items():
        if col in flows_df.columns:
            out[:, idx[sensor]] = np.abs(flows_df[col].values[:T].astype(np.float32))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STGCN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def make_stgcn_features(raw, baseline, mu, sigma, window):
    T = raw.shape[0]; W = min(window, T)
    dev   = raw[:W] - baseline[:W]
    feats = np.stack([raw[:W], dev], axis=-1).astype(np.float32)
    feats = (feats - mu[None]) / (sigma[None] + 1e-8)
    return torch.tensor(feats[None], dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def predict_stgcn(model, x):
    model.eval()
    dl, pl, sl, pp = model(x)
    det  = int(dl.argmax(dim=1).item())
    pipe = int(pl.argmax(dim=1).item())
    pos  = float(pp.item())
    prob = float(torch.softmax(dl, dim=1)[0, 1].item())
    pred_pipe = None if det == 0 else (pipe + 1 if pipe < 5 else None)
    return det, pred_pipe, pos, prob


# ─────────────────────────────────────────────────────────────────────────────
# PER-SCENARIO WORKER
# ─────────────────────────────────────────────────────────────────────────────
# Bundle cache shared within a process
_stgcn_cache: dict = {}

def _get_stgcn():
    if "model" not in _stgcn_cache:
        bundle = torch.load(str(STGCN_BUNDLE), map_location="cpu", weights_only=False)
        adj    = bundle["adjacency"]
        nn_    = len(bundle["sensor_names"])
        model  = SingleLeakSTGCN(adj, nn_, bundle["hidden_1"], bundle["hidden_2"],
                                 bundle["kernel_size"], bundle["dropout"],
                                 bundle["node_feats"]).to(DEVICE)
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()
        _stgcn_cache["model"]    = model
        _stgcn_cache["mu"]       = bundle["mu"]
        _stgcn_cache["sigma"]    = bundle["sigma"]
        _stgcn_cache["baseline"] = bundle["baseline_template"]
        _stgcn_cache["sensors"]  = bundle["sensor_names"]
        _stgcn_cache["window"]   = bundle["window"]
    return (_stgcn_cache["model"], _stgcn_cache["mu"], _stgcn_cache["sigma"],
            _stgcn_cache["baseline"], _stgcn_cache["sensors"], _stgcn_cache["window"])


def _process_scenario(folder: Path, inp_path: Path | None = None) -> dict | None:
    try:
        import json
        cfg       = make_config(inp_path=inp_path)
        metadata  = extract_model_metadata(cfg)
        hydraulic = HydraulicInterface(cfg, metadata)

        data_df = pd.read_csv(folder / "data.csv")
        with open(folder / "labels.json", encoding="utf-8") as f:
            labels = json.load(f)

        true_det  = int(labels.get("label_detection", 0))
        true_pipe = int(labels.get("label_pipe", -1))
        true_pos  = labels.get("label_position", None)
        if true_pos is not None:
            true_pos = float(true_pos)
        group = "no-leak" if true_det == 0 else f"pipe-{true_pipe}"

        meas_df = pd.DataFrame({
            "timestamp_s": data_df["t"].values * 60.0,
            "P4": data_df["P4"].values,
            "Q1a": data_df["Q1a"].values,
            "Q3a": data_df["Q3a"].values,
        })

        pressures_df, flows_df = run_ekf_scenario(meas_df, cfg, metadata, hydraulic)
        model, mu, sigma, baseline, sensors, window = _get_stgcn()
        full_raw = build_full_array(data_df, pressures_df, flows_df, sensors)
        x_tensor = make_stgcn_features(full_raw, baseline, mu, sigma, window)
        pred_det, pred_pipe, pred_pos, det_prob = predict_stgcn(model, x_tensor)

        return {
            "scenario_id":    folder.name,
            "group":          group,
            "true_detection": true_det,
            "true_pipe":      true_pipe,
            "true_position":  true_pos,
            "pred_detection": pred_det,
            "pred_pipe":      pred_pipe,
            "pred_position":  round(pred_pos, 4),
            "det_prob_leak":  round(det_prob, 4),
            "det_correct":    (pred_det == true_det),
            "pipe_correct":   (pred_pipe == true_pipe) if true_det == 1 else None,
        }
    except Exception as exc:
        return {"error": str(exc), "scenario_id": folder.name}


def _worker(folder_str: str) -> dict | None:
    if str(_EKF_DIR) not in sys.path:
        sys.path.insert(0, str(_EKF_DIR))
    # Each worker process gets its own temp dir. WNTR writes temp.inp/.bin/.rpt
    # relative to CWD, so we chdir there to avoid cross-process collisions.
    tmp_dir = Path(tempfile.mkdtemp(prefix="ekf_pipe_"))
    try:
        local_inp = tmp_dir / "base3.inp"
        shutil.copy2(str(INP_PATH), str(local_inp))
        os.chdir(str(tmp_dir))
        return _process_scenario(Path(folder_str), inp_path=local_inp)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="EKF -> STGCN pipeline evaluation (all test_dataset)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default: 1)")
    args = parser.parse_args()

    for path in [STGCN_BUNDLE, INP_PATH, MANIFEST]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    folders  = []
    for scn_id in manifest["scenario_id"].values:
        p = DATASET_ROOT / f"scenario_{int(scn_id):05d}"
        if (p / "data.csv").exists() and (p / "labels.json").exists():
            folders.append(p)

    print(f"STGCN bundle   : {STGCN_BUNDLE}")
    print(f"EKF model      : {INP_PATH}")
    print(f"Test scenarios : {len(folders)}")
    print(f"Workers        : {args.workers}")
    print("Starting pipeline evaluation...\n")

    per_rows  = []
    failed    = 0
    completed = 0
    total     = len(folders)

    def collect(result):
        nonlocal failed, completed
        completed += 1
        if result is None or "error" in result:
            scn = (result or {}).get("scenario_id", "unknown")
            err = (result or {}).get("error", "None returned")
            print(f"  [WARN] {scn}: {err}")
            failed += 1; return
        per_rows.append(result)
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

    if not per_rows:
        print("No results to save."); return

    # ── Save per-scenario CSV ────────────────────────────────────────────────
    df = pd.DataFrame(per_rows)
    per_csv = RESULTS_DIR / "ekf_stgcn_per_scenario.csv"
    df.to_csv(per_csv, index=False)
    print(f"\n[OK] Per-scenario results -> {per_csv}")

    # ── Detection metrics ────────────────────────────────────────────────────
    det_metrics, det_conf, det_conf_norm = compute_detection_metrics(
        df["true_detection"].tolist(), df["pred_detection"].tolist())
    det_df = pd.DataFrame([det_metrics])
    det_df.to_csv(RESULTS_DIR / "ekf_stgcn_detection_metrics.csv", index=False)
    pd.DataFrame(det_conf, index=["true_0", "true_1"],
                 columns=["pred_0", "pred_1"]).to_csv(
        RESULTS_DIR / "ekf_stgcn_detection_confusion.csv")

    # ── Pipe metrics ─────────────────────────────────────────────────────────
    pipe_df, macro_f1, pipe_conf_raw, pipe_conf_norm = compute_pipe_metrics(per_rows)
    pipe_df["macro_f1"] = ""
    pipe_df.iloc[-1, pipe_df.columns.get_loc("macro_f1")] = macro_f1
    pipe_df.to_csv(RESULTS_DIR / "ekf_stgcn_pipe_metrics.csv", index=False)
    # Pipe confusion matrices
    pipes = [f"pipe{i}" for i in range(1, 6)]
    pd.DataFrame(pipe_conf_raw, index=[f"true_{p}" for p in pipes],
                 columns=[f"pred_{p}" for p in pipes]).to_csv(
        RESULTS_DIR / "ekf_stgcn_pipe_confusion_raw.csv")
    pd.DataFrame(np.round(pipe_conf_norm, 4), index=[f"true_{p}" for p in pipes],
                 columns=[f"pred_{p}" for p in pipes]).to_csv(
        RESULTS_DIR / "ekf_stgcn_pipe_confusion_norm.csv")

    # ── Position metrics ─────────────────────────────────────────────────────
    pos_metrics = compute_position_metrics(per_rows)
    pd.DataFrame([pos_metrics]).to_csv(
        RESULTS_DIR / "ekf_stgcn_position_metrics.csv", index=False)

    # ── Pipe exact-match accuracy ─────────────────────────────────────────────
    leak_rows     = [r for r in per_rows if r["true_detection"] == 1]
    pipe_correct  = [r for r in leak_rows if r["pipe_correct"]]
    pipe_exact_acc = len(pipe_correct) / len(leak_rows) if leak_rows else float("nan")

    # ── Per-group detection and pipe accuracy ─────────────────────────────────
    group_rows = []
    for g in ["no-leak"] + [f"pipe-{p}" for p in range(1, 6)] + ["ALL"]:
        sub = [r for r in per_rows if r["group"] == g] if g != "ALL" else per_rows
        if not sub: continue
        n         = len(sub)
        det_acc   = np.mean([r["det_correct"] for r in sub])
        pipe_rows = [r for r in sub if r["true_detection"] == 1]
        pipe_acc  = np.mean([r["pipe_correct"] for r in pipe_rows]) if pipe_rows else float("nan")
        group_rows.append({"group": g, "n": n,
                           "det_accuracy": round(det_acc, 4),
                           "n_leak": len(pipe_rows),
                           "pipe_accuracy": round(pipe_acc, 4) if not np.isnan(pipe_acc) else "N/A"})
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(RESULTS_DIR / "ekf_stgcn_group_summary.csv", index=False)

    # ── Console report ────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 65)
    lines.append("EKF -> ST-GCN S10-A PIPELINE EVALUATION  (test_dataset, all)")
    lines.append("=" * 65)
    lines.append(f"Total scenarios : {len(per_rows)}  ({failed} failed/excluded)")
    lines.append(f"No-leak         : {det_metrics['n_noleak']}")
    lines.append(f"Leak            : {det_metrics['n_leak']}")
    lines.append("")
    lines.append("--- Detection ---")
    lines.append(f"  Accuracy    : {det_metrics['accuracy']:.4f}")
    lines.append(f"  Precision   : {det_metrics['precision']:.4f}")
    lines.append(f"  Recall      : {det_metrics['recall']:.4f}")
    lines.append(f"  F1          : {det_metrics['f1']:.4f}")
    lines.append(f"  Specificity : {det_metrics['specificity']:.4f}  (no-leak correctly rejected)")
    lines.append(f"  TP={det_metrics['tp']}  FP={det_metrics['fp']}  "
                 f"FN={det_metrics['fn']}  TN={det_metrics['tn']}")
    lines.append("")
    lines.append("  Confusion matrix (rows=true, cols=predicted):")
    lines.append(f"             pred_0   pred_1")
    lines.append(f"  true_0  :  {det_conf[0,0]:>6}   {det_conf[0,1]:>6}")
    lines.append(f"  true_1  :  {det_conf[1,0]:>6}   {det_conf[1,1]:>6}")
    lines.append("")
    lines.append("--- Pipe Localisation (among leak scenarios) ---")
    lines.append(f"  Pipe exact-match accuracy : {pipe_exact_acc:.4f}  "
                 f"({len(pipe_correct)}/{len(leak_rows)})")
    lines.append(f"  Macro F1 (pipes 1-5)      : {macro_f1:.4f}")
    lines.append("")
    lines.append(f"  {'Pipe':<6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'Support':>8}")
    lines.append("  " + "-" * 58)
    for _, pr in pipe_df.iterrows():
        lines.append(f"  {int(pr['pipe']):<6}  {int(pr['tp']):>4}  {int(pr['fp']):>4}  "
                     f"{int(pr['fn']):>4}  {pr['precision']:>7.4f}  {pr['recall']:>7.4f}  "
                     f"{pr['f1']:>7.4f}  {int(pr['support']):>8}")
    lines.append("")
    lines.append("  Pipe confusion matrix (normalised by true pipe):")
    header = "           " + "".join(f"  pred_{i}" for i in range(1, 6))
    lines.append(header)
    for i, tp in enumerate(range(1, 6)):
        row_str = f"  true_{tp} :  " + "  ".join(f"{pipe_conf_norm[i, j]:>6.3f}"
                                                   for j in range(5))
        lines.append(row_str)
    lines.append("")
    lines.append("--- Position Regression (pipe-matched pairs) ---")
    lines.append(f"  MAE  : {pos_metrics['mae']}")
    lines.append(f"  RMSE : {pos_metrics['rmse']}")
    lines.append(f"  R^2  : {pos_metrics['r2']}")
    lines.append(f"  n    : {pos_metrics['n']}")
    lines.append("")
    lines.append("--- Per-Group Summary ---")
    lines.append(f"  {'Group':<12}  {'n':>4}  {'DetAcc':>8}  {'n_leak':>6}  {'PipeAcc':>8}")
    lines.append("  " + "-" * 46)
    for _, gr in group_df.iterrows():
        lines.append(f"  {gr['group']:<12}  {int(gr['n']):>4}  {float(gr['det_accuracy']):>8.4f}  "
                     f"{int(gr['n_leak']):>6}  {str(gr['pipe_accuracy']):>8}")

    report = "\n".join(lines)
    print("\n" + report)

    summary_path = RESULTS_DIR / "ekf_stgcn_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\n[OK] Summary -> {summary_path}")
    print("Pipeline evaluation complete.")


if __name__ == "__main__":
    main()
