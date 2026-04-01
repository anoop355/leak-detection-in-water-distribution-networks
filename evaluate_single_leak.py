"""
Evaluation script for single-leak detection / localisation models
==================================================================
Supports both LeakTCN and SingleLeakSTGCN bundles.
Works with the test_dataset/ folder structure produced by
generate_test_dataset.py:

    test_dataset/scenarios/scenario_XXXXX/
        data.csv
        labels.json   (flat single-leak format)

Metrics computed
----------------
  Detection          : accuracy, precision, recall, F1, 2x2 confusion matrix
  Pipe identification: macro F1, micro F1, exact-match accuracy,
                       normalised 5x5 confusion matrix, per-pipe P/R/F1
  Position regression: overall MAE / RMSE / R2,
                       per-pipe MAE / RMSE / R2,
                       scatter CSV (true_pos, pred_pos, pipe_id)
  Tolerance-based    : pipe+position F1 at --pos_tol threshold

Usage
-----
    python evaluate_single_leak.py --bundle stgcn_bundle_optionB_25ep.pt
    python evaluate_single_leak.py --bundle leak_tcn_bundle.pt
    python evaluate_single_leak.py --bundle stgcn_bundle_optionB_25ep.pt \\
        --test_dir test_dataset --pos_tol 0.20
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ===========================================================================
# Model definitions (must match training scripts exactly)
# ===========================================================================

NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES  = NUM_PIPES + 1
SIZE_CLASSES  = 4


class LeakTCN(nn.Module):
    """Plain TCN — matches train_tcn_detection_localisation3.py."""
    def __init__(self, C):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=4,  dilation=1), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=8,  dilation=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=16, dilation=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.detect_head = nn.Linear(32, 2)
        self.pipe_head   = nn.Linear(32, PIPE_CLASSES)
        self.size_head   = nn.Linear(32, SIZE_CLASSES)
        self.pos_head    = nn.Linear(32, 1)

    def forward(self, x):
        z = self.backbone(x)
        return (
            self.detect_head(z),
            self.pipe_head(z),
            self.size_head(z),
            self.pos_head(z).squeeze(1),
        )


class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (1, kernel_size),
                              padding=(0, pad), dilation=(1, dilation))
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):                     # (B,T,N,C)
        x = x.permute(0, 3, 2, 1)            # (B,C,N,T)
        x = self.act(self.bn(self.conv(x)))
        return x.permute(0, 3, 2, 1)         # (B,T,N,C)


class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):                                       # (B,T,N,C)
        x = torch.einsum("ij,btjc->btic", self.A, x)
        return self.act(self.ln(self.lin(x)))


class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5,
                 dilation=1, dropout=0.25):
        super().__init__()
        self.temp    = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph   = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout = nn.Dropout(dropout)
        self.out_act = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.res_proj(x)
        y = self.dropout(self.graph(self.temp(x)))
        return self.out_act(y + r)


class SingleLeakSTGCN(nn.Module):
    """ST-GCN — matches train_stgtcn_detection_localisation3.py."""
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, 4, dropout)

        head_in     = num_nodes * hidden_2
        head_hidden = 64

        def _head(out_size):
            return nn.Sequential(
                nn.Linear(head_in, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(head_hidden, out_size)
            )

        self.detect_head = _head(2)
        self.pipe_head   = _head(PIPE_CLASSES)
        self.size_head   = _head(SIZE_CLASSES)
        self.pos_head    = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1), nn.Sigmoid()
        )

    def forward(self, x):                              # (B,T,N,2)
        x = self.block3(self.block2(self.block1(x)))
        z = x.mean(dim=1).reshape(x.size(0), -1)      # (B, N*H2)
        return (
            self.detect_head(z),
            self.pipe_head(z),
            self.size_head(z),
            self.pos_head(z).squeeze(1),
        )


class TemporalAttentionPool(nn.Module):
    """Soft attention over time axis — matches train_stgtcn_detection_localisation4.py."""
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * num_nodes, 1)

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat  = x.reshape(B, T, N * C)
        scores  = self.attn(x_flat)
        weights = torch.softmax(scores, dim=1)
        return (x_flat * weights).sum(dim=1)


class SingleLeakSTGCNv4(nn.Module):
    """ST-GCN with temporal attention pooling — matches train_stgtcn_detection_localisation4.py."""
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
                nn.Linear(head_in, head_hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(head_hidden, out_size)
            )

        self.detect_head = _head(2)
        self.pipe_head   = _head(PIPE_CLASSES)
        self.size_head   = _head(SIZE_CLASSES)
        self.pos_head    = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 1), nn.Sigmoid()
        )

    def forward(self, x):                              # (B,T,N,2)
        x = self.block3(self.block2(self.block1(x)))
        z = self.temporal_pool(x)                      # (B, N*H2)
        return (
            self.detect_head(z),
            self.pipe_head(z),
            self.size_head(z),
            self.pos_head(z).squeeze(1),
        )


# ===========================================================================
# Bundle loading
# ===========================================================================

def load_bundle(bundle_path: Path, device: str):
    bundle = torch.load(str(bundle_path), map_location=device, weights_only=False)
    model_type = bundle.get("model_type", "tcn")

    if "stgcn" in model_type.lower():
        adj        = np.array(bundle["adjacency"], dtype=np.float32)
        num_nodes  = len(bundle["sensor_names"])
        hidden_1   = int(bundle.get("hidden_1", 16))
        hidden_2   = int(bundle.get("hidden_2", 32))
        kernel_sz  = int(bundle.get("kernel_size", 5))
        dropout    = float(bundle.get("dropout", 0.25))
        node_feats = int(bundle.get("node_feats", 2))
        if model_type == "stgcn_single_leak_v4":
            model = SingleLeakSTGCNv4(adj, num_nodes, hidden_1, hidden_2,
                                      kernel_sz, dropout, node_feats).to(device)
        else:
            model = SingleLeakSTGCN(adj, num_nodes, hidden_1, hidden_2,
                                    kernel_sz, dropout, node_feats).to(device)
    else:
        sensor_names = bundle["feature_cols"]
        model = LeakTCN(C=len(sensor_names)).to(device)

    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle, model_type


# ===========================================================================
# Input preprocessing
# ===========================================================================

def preprocess_tcn(raw: np.ndarray, mu: np.ndarray,
                   sigma: np.ndarray) -> torch.Tensor:
    """raw: (T, C) → tensor (1, C, T)"""
    x = (raw - mu) / (sigma + 1e-8)
    return torch.tensor(x.T, dtype=torch.float32).unsqueeze(0)  # (1,C,T)


def preprocess_stgcn(raw: np.ndarray, baseline: np.ndarray,
                     mu: np.ndarray, sigma: np.ndarray,
                     node_feats: int = 2) -> torch.Tensor:
    """
    raw:        (T, N)
    baseline:   (T_base, N)
    mu/sigma:   (N, node_feats)
    node_feats: 2 → [raw, deviation]; 3 → [raw, deviation, first_diff]
    → tensor (1, T, N, node_feats)
    """
    T = raw.shape[0]
    base = baseline[:T]
    channels = [raw, raw - base]
    if node_feats == 3:
        diff = np.zeros_like(raw)
        diff[1:] = raw[1:] - raw[:-1]
        channels.append(diff)
    feats = np.stack(channels, axis=-1).astype(np.float32)  # (T,N,node_feats)
    feats = (feats - mu[None]) / (sigma[None] + 1e-8)
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


# ===========================================================================
# Single-scenario inference
# ===========================================================================

def predict_scenario(data_csv: Path, model: nn.Module, bundle: dict,
                     model_type: str, device: str) -> Dict:
    sensor_names = bundle.get("sensor_names") or bundle.get("feature_cols")
    mu    = np.array(bundle["mu"],    dtype=np.float32)
    sigma = np.array(bundle["sigma"], dtype=np.float32)

    df  = pd.read_csv(data_csv)
    raw = df[sensor_names].to_numpy(dtype=np.float32)  # (T, C or N)

    if "stgcn" in model_type.lower():
        baseline   = np.array(bundle["baseline_template"], dtype=np.float32)
        node_feats = int(bundle.get("node_feats", 2))
        x_t = preprocess_stgcn(raw, baseline, mu, sigma, node_feats).to(device)
    else:
        x_t = preprocess_tcn(raw, mu, sigma).to(device)

    with torch.no_grad():
        detect_logits, pipe_logits, _, pos_pred = model(x_t)

    pred_detect = int(detect_logits.argmax(dim=1).item())
    pred_pipe   = int(pipe_logits.argmax(dim=1).item())    # 0..4 or 5=NONE
    pred_pos    = float(pos_pred.item())

    # Convert pipe index back to pipe_id (1..5); if NONE treat as no prediction
    pred_pipe_id = (pred_pipe + 1) if pred_pipe < NUM_PIPES else None

    return {
        "pred_detect":  pred_detect,
        "pred_pipe_id": pred_pipe_id,
        "pred_pos":     max(0.0, min(1.0, pred_pos)),
    }


# ===========================================================================
# Metrics helpers
# ===========================================================================

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_detection_metrics(true_det: List[int],
                               pred_det: List[int]) -> Dict:
    t = np.array(true_det, dtype=int)
    p = np.array(pred_det, dtype=int)

    acc = float(np.mean(t == p))
    tp  = int(np.sum((t == 1) & (p == 1)))
    fp  = int(np.sum((t == 0) & (p == 1)))
    fn  = int(np.sum((t == 1) & (p == 0)))
    tn  = int(np.sum((t == 0) & (p == 0)))
    prec, rec, f1 = prf(tp, fp, fn)

    return {
        "accuracy":  round(acc,  4),
        "precision": round(prec, 4),
        "recall":    round(rec,  4),
        "f1":        round(f1,   4),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_scenarios": len(t),
    }


def compute_pipe_metrics(pipe_stats: Dict) -> Dict:
    f1_scores, total_tp, total_fp, total_fn = [], 0, 0, 0
    for pid in range(1, NUM_PIPES + 1):
        tp = pipe_stats[pid]["tp"]
        fp = pipe_stats[pid]["fp"]
        fn = pipe_stats[pid]["fn"]
        _, _, f1 = prf(tp, fp, fn)
        f1_scores.append(f1)
        total_tp += tp; total_fp += fp; total_fn += fn

    macro_f1 = float(np.mean(f1_scores))
    _, _, micro_f1 = prf(total_tp, total_fp, total_fn)
    return {"pipe_macro_f1": round(macro_f1, 4),
            "pipe_micro_f1": round(micro_f1, 4)}


def compute_regression_metrics(pos_pairs: List[Tuple]) -> Dict:
    if not pos_pairs:
        return {"mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "n_matched": 0}
    tv = np.array([p[0] for p in pos_pairs], dtype=float)
    pv = np.array([p[1] for p in pos_pairs], dtype=float)
    err = pv - tv
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((tv - tv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": round(mae, 6), "rmse": round(rmse, 6),
            "r2": round(r2, 6), "n_matched": len(pos_pairs)}


def compute_per_pipe_regression(pos_pairs: List[Tuple]) -> pd.DataFrame:
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        pp = [(t, p) for t, p, pipe in pos_pairs if pipe == pid]
        if not pp:
            rows.append({"pipe_id": pid, "n_matched": 0,
                         "mae": float("nan"), "rmse": float("nan"),
                         "r2": float("nan")})
            continue
        tv = np.array([x[0] for x in pp])
        pv = np.array([x[1] for x in pp])
        err = pv - tv
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((tv - tv.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({"pipe_id": pid, "n_matched": len(pp),
                     "mae": round(mae, 6), "rmse": round(rmse, 6),
                     "r2": round(r2, 6)})
    return pd.DataFrame(rows)


def pipe_prf_df(pipe_stats: Dict) -> pd.DataFrame:
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        tp = pipe_stats[pid]["tp"]
        fp = pipe_stats[pid]["fp"]
        fn = pipe_stats[pid]["fn"]
        pr, rc, f1 = prf(tp, fp, fn)
        rows.append({"pipe_id": pid, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(pr, 4), "recall": round(rc, 4),
                     "f1": round(f1, 4),
                     "support_true": tp + fn, "predicted_total": tp + fp})
    return pd.DataFrame(rows)


def pipe_confusion_normalised(pipe_stats: Dict) -> pd.DataFrame:
    raw = np.zeros((NUM_PIPES, NUM_PIPES), dtype=float)
    for i, tp in enumerate(range(1, NUM_PIPES + 1)):
        for j, pp in enumerate(range(1, NUM_PIPES + 1)):
            raw[i, j] = pipe_stats[tp]["conf"].get(pp, 0)
    row_sums = raw.sum(axis=1, keepdims=True)
    norm = np.where(row_sums > 0, raw / row_sums, 0.0)
    return pd.DataFrame(np.round(norm, 4),
                        index=[f"true_pipe_{p}" for p in range(1, NUM_PIPES + 1)],
                        columns=[f"pred_pipe_{p}" for p in range(1, NUM_PIPES + 1)])


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate single-leak LeakTCN or SingleLeakSTGCN on test_dataset."
    )
    parser.add_argument("--test_dir", type=str, default="test_dataset",
                        help="Root of test_dataset/ (contains scenarios/ sub-folder)")
    parser.add_argument("--bundle", type=str,
                        default="stgcn_bundle_optionB_25ep.pt",
                        help="Path to trained model bundle (.pt)")
    parser.add_argument("--pos_tol", type=float, default=0.25,
                        help="Position tolerance for pipe+pos F1 metric")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    test_dir    = Path(args.test_dir)
    bundle_path = Path(args.bundle)
    scenarios_dir = test_dir / "scenarios"

    if not scenarios_dir.exists():
        raise FileNotFoundError(f"Scenarios folder not found: {scenarios_dir}")
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    print(f"Bundle     : {bundle_path}")
    print(f"Test dir   : {scenarios_dir}")
    print(f"Pos. tol.  : {args.pos_tol}")
    print(f"Device     : {args.device}")
    print()

    model, bundle, model_type = load_bundle(bundle_path, args.device)
    print(f"Model type : {model_type}")
    print(f"Sensors    : {bundle.get('sensor_names') or bundle.get('feature_cols')}")
    print()

    # Output directory
    bundle_stem = bundle_path.stem
    out_dir = test_dir / f"evaluation_{bundle_stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all valid scenario folders
    scn_dirs = sorted(
        [p for p in scenarios_dir.iterdir()
         if p.is_dir() and p.name.startswith("scenario_")],
        key=lambda p: int(p.name.split("_")[1])
    )
    if not scn_dirs:
        raise RuntimeError(f"No scenario_* folders found in {scenarios_dir}")

    # Accumulators
    per_rows     = []
    true_det_all = []
    pred_det_all = []
    all_pos_pairs: List[Tuple] = []     # (true_pos, pred_pos, pipe_id)

    pipe_only_stats = {pid: {"tp": 0, "fp": 0, "fn": 0,
                             "conf": {p: 0 for p in range(1, NUM_PIPES + 1)}}
                       for pid in range(1, NUM_PIPES + 1)}

    pipe_pos_stats  = {pid: {"tp": 0, "fp": 0, "fn": 0, "pos_errors": [],
                             "conf": {p: 0 for p in range(1, NUM_PIPES + 1)}}
                       for pid in range(1, NUM_PIPES + 1)}

    exact_pipe_match_count = 0
    n_leak_scenarios       = 0

    print(f"Evaluating {len(scn_dirs)} scenarios...")

    for scn_dir in scn_dirs:
        data_path  = scn_dir / "data.csv"
        label_path = scn_dir / "labels.json"
        if not data_path.exists() or not label_path.exists():
            continue

        labels = json.loads(label_path.read_text(encoding="utf-8"))
        scn_id       = int(labels["scenario_id"])
        true_detect  = int(labels["label_detection"])
        true_pipe_id = int(labels["label_pipe"])        # -1 if no-leak
        true_pos     = float(labels["label_position"])  # -1.0 if no-leak
        true_size    = str(labels["label_size"])

        pred = predict_scenario(data_path, model, bundle, model_type, args.device)
        pred_detect  = pred["pred_detect"]
        pred_pipe_id = pred["pred_pipe_id"]   # 1..5 or None
        pred_pos     = pred["pred_pos"]

        true_det_all.append(true_detect)
        pred_det_all.append(pred_detect)

        # Pipe and position metrics — leak scenarios only
        tp_pipe = fp_pipe = fn_pipe = 0
        tp_pos  = fp_pos  = fn_pos  = 0
        pos_err = float("nan")
        pipe_correct = False

        if true_detect == 1:
            n_leak_scenarios += 1

            # Pipe-only match
            if pred_pipe_id == true_pipe_id:
                tp_pipe = 1
                pipe_correct = True
                pipe_only_stats[true_pipe_id]["tp"] += 1
            else:
                fn_pipe = 1
                pipe_only_stats[true_pipe_id]["fn"] += 1
                if pred_pipe_id is not None:
                    fp_pipe = 1
                    pipe_only_stats[pred_pipe_id]["fp"] += 1

            # Confusion (pipe identification)
            if pred_pipe_id is not None and 1 <= pred_pipe_id <= NUM_PIPES:
                pipe_only_stats[true_pipe_id]["conf"][pred_pipe_id] += 1

            # Position error (only where pipe matched)
            if pipe_correct:
                pos_err = abs(pred_pos - true_pos)
                all_pos_pairs.append((true_pos, pred_pos, true_pipe_id))

            # Exact pipe match check
            if pipe_correct and fp_pipe == 0:
                exact_pipe_match_count += 1

            # Tolerance-based pipe+pos match
            if pipe_correct and pos_err <= args.pos_tol:
                tp_pos = 1
                pipe_pos_stats[true_pipe_id]["tp"] += 1
                pipe_pos_stats[true_pipe_id]["pos_errors"].append(pos_err)
            else:
                fn_pos = 1
                pipe_pos_stats[true_pipe_id]["fn"] += 1
                if pred_pipe_id is not None:
                    fp_pos = 1
                    if 1 <= pred_pipe_id <= NUM_PIPES:
                        pipe_pos_stats[pred_pipe_id]["fp"] += 1

        elif pred_detect == 1 and pred_pipe_id is not None:
            # False alarm: predicted leak on a no-leak scenario
            if 1 <= pred_pipe_id <= NUM_PIPES:
                pipe_only_stats[pred_pipe_id]["fp"] += 1
                pipe_pos_stats[pred_pipe_id]["fp"] += 1

        per_rows.append({
            "scenario_id":   scn_id,
            "scenario":      scn_dir.name,
            "source_inp":    labels.get("source_inp", ""),
            "true_detect":   true_detect,
            "pred_detect":   pred_detect,
            "detect_correct": int(true_detect == pred_detect),
            "true_pipe":     true_pipe_id,
            "pred_pipe":     pred_pipe_id if pred_pipe_id else -1,
            "pipe_correct":  int(pipe_correct),
            "true_pos":      true_pos,
            "pred_pos":      round(pred_pos, 4),
            "pos_error":     round(pos_err, 4) if not np.isnan(pos_err) else float("nan"),
            "true_size":     true_size,
            "tp_pipe":       tp_pipe, "fp_pipe": fp_pipe, "fn_pipe": fn_pipe,
            "tp_pos":        tp_pos,  "fp_pos":  fp_pos,  "fn_pos":  fn_pos,
        })

        # Write per-scenario prediction JSON
        pred_out = {
            "scenario_id":  scn_id,
            "true_detect":  true_detect,
            "pred_detect":  pred_detect,
            "pred_pipe_id": pred_pipe_id,
            "pred_pos":     round(pred_pos, 4),
        }
        (scn_dir / f"prediction_{bundle_path.stem}.json").write_text(
            json.dumps(pred_out, indent=2), encoding="utf-8"
        )

    # -------------------------------------------------------------------------
    # Detection metrics
    # -------------------------------------------------------------------------
    det_metrics = compute_detection_metrics(true_det_all, pred_det_all)

    # -------------------------------------------------------------------------
    # Pipe identification metrics
    # -------------------------------------------------------------------------
    pipe_summary     = compute_pipe_metrics(pipe_only_stats)
    pipe_exact_acc   = (exact_pipe_match_count / n_leak_scenarios
                        if n_leak_scenarios > 0 else float("nan"))
    pipe_id_summary  = {
        **pipe_summary,
        "pipe_exact_match_accuracy": round(float(pipe_exact_acc), 4),
        "pipe_exact_match_count":    exact_pipe_match_count,
        "n_leak_scenarios":          n_leak_scenarios,
    }

    # -------------------------------------------------------------------------
    # Position regression metrics
    # -------------------------------------------------------------------------
    regression_overall  = compute_regression_metrics(all_pos_pairs)
    regression_per_pipe = compute_per_pipe_regression(all_pos_pairs)

    # Tolerance-based pipe+pos summary (micro)
    tp_d = sum(r["tp_pos"] for r in per_rows)
    fp_d = sum(r["fp_pos"] for r in per_rows)
    fn_d = sum(r["fn_pos"] for r in per_rows)
    p_pos_micro, r_pos_micro, f1_pos_micro = prf(tp_d, fp_d, fn_d)

    # -------------------------------------------------------------------------
    # Build DataFrames
    # -------------------------------------------------------------------------
    per_df = pd.DataFrame(per_rows).sort_values("scenario_id")

    # False positive rate on no-leak scenarios
    no_leak_rows = per_df[per_df["true_detect"] == 0]
    fpr = float((no_leak_rows["pred_detect"] > 0).mean()) if len(no_leak_rows) > 0 else float("nan")

    # Detection confusion matrix
    det_conf = pd.DataFrame(
        [[det_metrics["tn"], det_metrics["fp"]],
         [det_metrics["fn"], det_metrics["tp"]]],
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"]
    )

    # Overall summary
    overall_summary = {
        "bundle":      str(bundle_path.name),
        "model_type":  model_type,
        "n_scenarios": len(per_rows),
        "pos_tol":     args.pos_tol,

        # Detection
        "detect_accuracy":  det_metrics["accuracy"],
        "detect_precision": det_metrics["precision"],
        "detect_recall":    det_metrics["recall"],
        "detect_f1":        det_metrics["f1"],
        "detect_tp":  det_metrics["tp"],  "detect_fp":  det_metrics["fp"],
        "detect_fn":  det_metrics["fn"],  "detect_tn":  det_metrics["tn"],
        "false_positive_rate_no_leak": round(fpr, 4),

        # Pipe identification
        **pipe_id_summary,

        # Position regression (pipe-matched pairs only)
        "pos_mae":  regression_overall["mae"],
        "pos_rmse": regression_overall["rmse"],
        "pos_r2":   regression_overall["r2"],
        "pos_n_matched": regression_overall["n_matched"],

        # Tolerance-based pipe+pos
        "pipe_pos_tol_precision_micro": round(p_pos_micro, 4),
        "pipe_pos_tol_recall_micro":    round(r_pos_micro, 4),
        "pipe_pos_tol_f1_micro":        round(f1_pos_micro, 4),
    }

    # -------------------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------------------
    per_csv = out_dir / "per_scenario_metrics.csv"
    per_df.to_csv(per_csv, index=False)

    summary_path = out_dir / "overall_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    det_conf_path = out_dir / "detection_confusion_matrix.csv"
    det_conf.to_csv(det_conf_path)

    pipe_prf_path = out_dir / "per_pipe_pipe_only.csv"
    pipe_prf_df(pipe_only_stats).to_csv(pipe_prf_path, index=False)

    pipe_conf_path = out_dir / "pipe_confusion_matrix_normalised.csv"
    pipe_confusion_normalised(pipe_only_stats).to_csv(pipe_conf_path)

    pipe_id_path = out_dir / "pipe_identification_summary.json"
    with open(pipe_id_path, "w", encoding="utf-8") as f:
        json.dump(pipe_id_summary, f, indent=2)

    reg_summary_path = out_dir / "position_regression_summary.json"
    with open(reg_summary_path, "w", encoding="utf-8") as f:
        json.dump(regression_overall, f, indent=2)

    reg_pipe_path = out_dir / "position_regression_per_pipe.csv"
    regression_per_pipe.to_csv(reg_pipe_path, index=False)

    if all_pos_pairs:
        scatter_df = pd.DataFrame(all_pos_pairs,
                                  columns=["true_position", "pred_position", "pipe_id"])
        scatter_df["pipe_id"] = scatter_df["pipe_id"].astype(int)
    else:
        scatter_df = pd.DataFrame(columns=["true_position", "pred_position", "pipe_id"])
    scatter_path = out_dir / "position_scatter_data.csv"
    scatter_df.to_csv(scatter_path, index=False)

    # Per-pipe pos-error stats (tolerance-based matches)
    pos_err_rows = []
    for pid in range(1, NUM_PIPES + 1):
        errs = np.array(pipe_pos_stats[pid]["pos_errors"], dtype=float)
        pos_err_rows.append({
            "pipe_id":    pid,
            "n_matched":  int(errs.size),
            "pos_mae":    round(float(np.mean(errs)),   6) if errs.size > 0 else float("nan"),
            "pos_median": round(float(np.median(errs)), 6) if errs.size > 0 else float("nan"),
            "pos_std":    round(float(np.std(errs)),    6) if errs.size > 0 else float("nan"),
            "pos_max":    round(float(np.max(errs)),    6) if errs.size > 0 else float("nan"),
        })
    pd.DataFrame(pos_err_rows).to_csv(out_dir / "per_pipe_pos_error_stats.csv", index=False)

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  DETECTION  (n={len(per_rows)})")
    print(f"{'='*60}")
    print(f"  Accuracy : {det_metrics['accuracy']:.4f}")
    print(f"  Precision: {det_metrics['precision']:.4f}")
    print(f"  Recall   : {det_metrics['recall']:.4f}")
    print(f"  F1       : {det_metrics['f1']:.4f}")
    print(f"  FPR (no-leak scenarios): {fpr:.4f}")
    print(f"\n{'='*60}")
    print(f"  PIPE IDENTIFICATION  (leak scenarios only, n={n_leak_scenarios})")
    print(f"{'='*60}")
    print(f"  Macro F1      : {pipe_id_summary['pipe_macro_f1']:.4f}")
    print(f"  Micro F1      : {pipe_id_summary['pipe_micro_f1']:.4f}")
    print(f"  Exact match   : {pipe_id_summary['pipe_exact_match_accuracy']:.4f}  "
          f"({exact_pipe_match_count}/{n_leak_scenarios})")
    print(f"\n{'='*60}")
    print(f"  POSITION REGRESSION  (pipe-matched pairs, n={regression_overall['n_matched']})")
    print(f"{'='*60}")
    print(f"  MAE  : {regression_overall['mae']}")
    print(f"  RMSE : {regression_overall['rmse']}")
    print(f"  R²   : {regression_overall['r2']}")
    print(f"\n  Pipe+pos F1 (tol={args.pos_tol}): {f1_pos_micro:.4f}")
    print(f"\n{'='*60}")
    print(f"  OUTPUT: {out_dir.resolve()}")
    print(f"{'='*60}")
    print(f"  {per_csv.name}")
    print(f"  {summary_path.name}")
    print(f"  {det_conf_path.name}")
    print(f"  {pipe_prf_path.name}")
    print(f"  {pipe_conf_path.name}")
    print(f"  {pipe_id_path.name}")
    print(f"  {reg_summary_path.name}")
    print(f"  {reg_pipe_path.name}")
    print(f"  {scatter_path.name}")


if __name__ == "__main__":
    main()
