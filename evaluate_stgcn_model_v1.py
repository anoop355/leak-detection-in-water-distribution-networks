"""
Updates in this version:
- loads a trained ST-GCN bundle from disk
- runs windowed evaluation over the selected test scenarios
- aggregates predictions into per-scenario and summary outputs
"""

import os
import json
import argparse
from pathlib import Path
import traceback
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


MAX_LEAKS     = 3
NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES       # index 5 = NONE
PIPE_CLASSES  = NUM_PIPES + 1   # 6
SIZE_CLASSES  = 4
NODE_FEATS    = 2


class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size),
            padding=(0, pad)
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)   # (B, C, N, T)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.permute(0, 3, 2, 1)   # (B, T, N, C)
        return x


class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.einsum("ij,btjc->btic", self.A, x)
        x = self.lin(x)
        x = self.ln(x)
        x = self.act(x)
        return x


class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dropout=0.25):
        super().__init__()
        self.temp    = TemporalConvLayer(in_ch, out_ch, kernel_size)
        self.graph   = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout = nn.Dropout(dropout)
        self.out_act = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        y = self.temp(x)
        y = self.graph(y)
        y = self.dropout(y)
        y = y + residual
        y = self.out_act(y)
        return y


class MultiTaskSTGCN(nn.Module):
    def __init__(self, adj_matrix, hidden_1=16, hidden_2=32, kernel_size=5, dropout=0.25):
        super().__init__()
        self.block1 = STBlock(NODE_FEATS, hidden_1, adj_matrix, kernel_size, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, dropout)

        head_hidden = 64

        self.count_head = nn.Sequential(
            nn.Linear(hidden_2, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, 4),
        )
        self.pipe_head = nn.Sequential(
            nn.Linear(hidden_2, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, MAX_LEAKS * PIPE_CLASSES),
        )
        self.size_head = nn.Sequential(
            nn.Linear(hidden_2, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, MAX_LEAKS * SIZE_CLASSES),
        )
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_2, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, MAX_LEAKS),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        z = x.mean(dim=(1, 2))   # global average pool over time and nodes
        count_logits = self.count_head(z)
        pipe_logits  = self.pipe_head(z).view(-1, MAX_LEAKS, PIPE_CLASSES)
        size_logits  = self.size_head(z).view(-1, MAX_LEAKS, SIZE_CLASSES)
        pos_pred     = self.pos_head(z).view(-1, MAX_LEAKS)
        return count_logits, pipe_logits, size_logits, pos_pred


def majority_vote(arr: np.ndarray, valid_mask: np.ndarray = None, default=None):
    if valid_mask is not None:
        arr = arr[valid_mask]
    if arr.size == 0:
        return default
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def mean_over_valid(values: np.ndarray, valid_mask: np.ndarray, default=0.0):
    v = values[valid_mask]
    if v.size == 0:
        return float(default)
    return float(np.mean(v))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def make_node_features_for_eval(
    signals: pd.DataFrame,
    sensor_names: List[str],
    baseline_template: np.ndarray
) -> np.ndarray:
    """Build (T, N, 2) node features: raw reading and deviation from baseline."""
    missing = [c for c in sensor_names if c not in signals.columns]
    if missing:
        raise ValueError(f"signals.csv missing required columns: {missing}")

    raw = signals[sensor_names].to_numpy(dtype=np.float32)
    T   = raw.shape[0]

    if baseline_template.shape[0] < T:
        raise ValueError(
            f"baseline_template shorter than scenario length: baseline={baseline_template.shape[0]}, scenario={T}"
        )

    base = baseline_template[:T]
    dev  = raw - base
    return np.stack([raw, dev], axis=-1).astype(np.float32)


def predict_from_signals_df(
    signals: pd.DataFrame,
    model: nn.Module,
    sensor_names: List[str],
    baseline_template: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    window: int,
    stride: int,
    device: str = "cpu"
) -> Dict:
    X = make_node_features_for_eval(signals, sensor_names, baseline_template)

    if len(X) < window:
        return {"predicted_leak_count": 0, "predicted_leaks": [], "num_windows": 0}

    windows = []
    for end in range(window, len(X) + 1, stride):
        x_win = X[end-window:end, :, :]
        x_win = (x_win - mu[None, :, :]) / (sigma[None, :, :] + 1e-8)
        windows.append(torch.tensor(x_win, dtype=torch.float32).unsqueeze(0))

    X_batch = torch.cat(windows, dim=0).to(device)

    with torch.no_grad():
        count_logits, pipe_logits, _size_logits, pos_pred = model(X_batch)

    count_pred = count_logits.argmax(dim=1).cpu().numpy().astype(int)
    pipe_pred  = pipe_logits.argmax(dim=2).cpu().numpy().astype(int)
    pos_pred   = pos_pred.cpu().numpy().astype(np.float32)

    final_count = majority_vote(count_pred, default=0)

    final_leaks = []
    for slot in range(MAX_LEAKS):
        slot_pipe = pipe_pred[:, slot]
        slot_pos  = pos_pred[:, slot]
        valid     = slot_pipe != PIPE_NONE_IDX
        pipe_vote = majority_vote(slot_pipe, valid_mask=valid, default=PIPE_NONE_IDX)
        pos_mean  = mean_over_valid(slot_pos, valid_mask=valid, default=0.0)
        if pipe_vote == PIPE_NONE_IDX:
            continue
        final_leaks.append({"pipe_id": int(pipe_vote + 1), "position": clamp01(pos_mean)})

    final_leaks = sorted(final_leaks, key=lambda d: (d["pipe_id"], d["position"]))
    final_leaks = final_leaks[:final_count] if final_count > 0 else []

    return {
        "predicted_leak_count": int(final_count),
        "predicted_leaks": final_leaks,
        "num_windows": int(X_batch.shape[0])
    }


def match_leaks_pipe_and_pos(true_leaks: List[Dict], pred_leaks: List[Dict], pos_tol: float):
    used_pred  = set()
    pos_errors = []
    TP = 0
    true_sorted = sorted(true_leaks, key=lambda d: (int(d["pipe_id"]), float(d["position"])))

    for t in true_sorted:
        t_pipe, t_pos = int(t["pipe_id"]), float(t["position"])
        best_j, best_err = None, None
        for j, p in enumerate(pred_leaks):
            if j in used_pred or int(p["pipe_id"]) != t_pipe:
                continue
            err = abs(float(p["position"]) - t_pos)
            if err <= pos_tol and (best_err is None or err < best_err):
                best_err, best_j = err, j
        if best_j is not None:
            used_pred.add(best_j)
            TP += 1
            pos_errors.append(float(best_err))

    return TP, len(pred_leaks) - TP, len(true_leaks) - TP, pos_errors


def match_leaks_pipe_and_pos_no_tol(true_leaks: List[Dict], pred_leaks: List[Dict]):
    used_pred  = set()
    pos_errors = []
    pos_pairs  = []
    TP = 0
    true_sorted = sorted(true_leaks, key=lambda d: (int(d["pipe_id"]), float(d["position"])))

    for t in true_sorted:
        t_pipe, t_pos = int(t["pipe_id"]), float(t["position"])
        best_j, best_err, best_pred_pos = None, None, None
        for j, p in enumerate(pred_leaks):
            if j in used_pred or int(p["pipe_id"]) != t_pipe:
                continue
            err = abs(float(p["position"]) - t_pos)
            if best_err is None or err < best_err:
                best_err, best_j, best_pred_pos = err, j, float(p["position"])
        if best_j is not None:
            used_pred.add(best_j)
            TP += 1
            pos_errors.append(float(best_err))
            pos_pairs.append((t_pos, best_pred_pos, t_pipe))

    return TP, len(pred_leaks) - TP, len(true_leaks) - TP, pos_errors, pos_pairs


def match_pipes_only(true_leaks: List[Dict], pred_leaks: List[Dict]):
    true_pipes = [int(l["pipe_id"]) for l in true_leaks]
    pred_pipes = [int(l["pipe_id"]) for l in pred_leaks]
    TP, used = 0, set()
    for tp in sorted(true_pipes):
        for j, pp in enumerate(pred_pipes):
            if j in used:
                continue
            if pp == tp:
                TP += 1
                used.add(j)
                break
    return TP, len(pred_pipes) - TP, len(true_pipes) - TP


def prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_count_classification_metrics(true_counts: List[int], pred_counts: List[int]):
    classes  = [0, 1, 2, 3]
    true_arr = np.array(true_counts, dtype=int)
    pred_arr = np.array(pred_counts, dtype=int)
    accuracy = float(np.mean(true_arr == pred_arr))

    conf_matrix = np.zeros((4, 4), dtype=int)
    for t, p in zip(true_arr, pred_arr):
        if 0 <= t <= 3 and 0 <= p <= 3:
            conf_matrix[t, p] += 1

    row_sums = conf_matrix.sum(axis=1, keepdims=True).astype(float)
    conf_matrix_norm = np.where(row_sums > 0, conf_matrix / row_sums, 0.0)

    rows = []
    f1_scores = []
    total_tp_micro = total_fp_micro = total_fn_micro = 0

    for c in classes:
        tp = int(np.sum((true_arr == c) & (pred_arr == c)))
        fp = int(np.sum((true_arr != c) & (pred_arr == c)))
        fn = int(np.sum((true_arr == c) & (pred_arr != c)))
        support = int(np.sum(true_arr == c))
        p, r, f1 = prf(tp, fp, fn)
        f1_scores.append(f1)
        total_tp_micro += tp
        total_fp_micro += fp
        total_fn_micro += fn
        rows.append({
            "leak_count_class": c,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(f1, 4), "support": support
        })

    macro_f1 = float(np.mean(f1_scores))
    p_micro, r_micro, micro_f1 = prf(total_tp_micro, total_fp_micro, total_fn_micro)

    summary = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(p_micro, 4),
        "micro_recall": round(r_micro, 4),
        "num_scenarios": int(len(true_arr))
    }
    return pd.DataFrame(rows), summary, conf_matrix, conf_matrix_norm


def compute_pipe_macro_f1(pipe_stats: Dict) -> Dict:
    f1_scores = []
    total_tp = total_fp = total_fn = 0
    for pid in range(1, NUM_PIPES + 1):
        tp, fp, fn = pipe_stats[pid]["tp"], pipe_stats[pid]["fp"], pipe_stats[pid]["fn"]
        _, _, f1 = prf(tp, fp, fn)
        f1_scores.append(f1)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    macro_f1 = float(np.mean(f1_scores))
    _, _, micro_f1 = prf(total_tp, total_fp, total_fn)
    return {"pipe_macro_f1": round(macro_f1, 4), "pipe_micro_f1": round(micro_f1, 4)}


def compute_pipe_exact_match(per_rows: List[Dict]) -> Dict:
    leak_rows = [r for r in per_rows if r["true_count"] > 0]
    if len(leak_rows) == 0:
        return {"pipe_exact_match_accuracy": float("nan"), "n_leak_scenarios": 0}
    exact = sum(
        1 for r in leak_rows
        if r["tp_pipe"] == r["true_count"] and r["fp_pipe"] == 0 and r["fn_pipe"] == 0
    )
    return {
        "pipe_exact_match_accuracy": round(exact / len(leak_rows), 4),
        "pipe_exact_match_count": exact,
        "n_leak_scenarios": len(leak_rows)
    }


def compute_pipe_confusion_normalised(pipe_stats: Dict) -> np.ndarray:
    raw = np.zeros((NUM_PIPES, NUM_PIPES), dtype=float)
    for i, true_pid in enumerate(range(1, NUM_PIPES + 1)):
        for j, pred_pid in enumerate(range(1, NUM_PIPES + 1)):
            raw[i, j] = float(pipe_stats[true_pid]["conf"].get(pred_pid, 0))
    row_sums = raw.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, raw / row_sums, 0.0)


def compute_regression_metrics(pos_pairs: List[Tuple]) -> Dict:
    if len(pos_pairs) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n_matched": 0}
    true_vals  = np.array([p[0] for p in pos_pairs], dtype=float)
    pred_vals  = np.array([p[1] for p in pos_pairs], dtype=float)
    errors     = pred_vals - true_vals
    mae        = float(np.mean(np.abs(errors)))
    rmse       = float(np.sqrt(np.mean(errors ** 2)))
    ss_res     = float(np.sum(errors ** 2))
    ss_tot     = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
    r2         = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6), "n_matched": int(len(pos_pairs))}


def compute_per_pipe_regression_metrics(pos_pairs: List[Tuple]) -> pd.DataFrame:
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        pipe_pairs = [(t, p) for t, p, pipe in pos_pairs if pipe == pid]
        if len(pipe_pairs) == 0:
            rows.append({"pipe_id": pid, "n_matched": 0, "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")})
            continue
        true_vals = np.array([x[0] for x in pipe_pairs], dtype=float)
        pred_vals = np.array([x[1] for x in pipe_pairs], dtype=float)
        errors    = pred_vals - true_vals
        mae       = float(np.mean(np.abs(errors)))
        rmse      = float(np.sqrt(np.mean(errors ** 2)))
        ss_res    = float(np.sum(errors ** 2))
        ss_tot    = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
        r2        = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rows.append({"pipe_id": pid, "n_matched": int(len(pipe_pairs)),
                     "mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6)})
    return pd.DataFrame(rows)


def init_pipe_stats():
    return {pid: {"tp": 0, "fp": 0, "fn": 0, "pos_errors": [],
                  "conf": {p: 0 for p in range(1, NUM_PIPES + 1)}}
            for pid in range(1, NUM_PIPES + 1)}


def update_pipe_only(pipe_stats, true_leaks, pred_leaks):
    true_pipes = [int(l["pipe_id"]) for l in true_leaks]
    pred_pipes = [int(l["pipe_id"]) for l in pred_leaks]
    used_pred  = set()
    for tp in true_pipes:
        match_j = None
        for j, pp in enumerate(pred_pipes):
            if j in used_pred:
                continue
            if pp == tp:
                match_j = j
                break
        if match_j is not None:
            used_pred.add(match_j)
            pipe_stats[tp]["tp"] += 1
        else:
            pipe_stats[tp]["fn"] += 1
    for j, pp in enumerate(pred_pipes):
        if j not in used_pred:
            pipe_stats[pp]["fp"] += 1
    for tp in true_pipes:
        for pp in pred_pipes:
            if 1 <= tp <= NUM_PIPES and 1 <= pp <= NUM_PIPES:
                pipe_stats[tp]["conf"][pp] += 1


def update_pipe_pos(pipe_stats, true_leaks, pred_leaks, pos_tol):
    used_pred   = set()
    true_sorted = sorted([(int(l["pipe_id"]), float(l["position"])) for l in true_leaks],
                         key=lambda x: (x[0], x[1]))
    for t_pipe, t_pos in true_sorted:
        best_j, best_err = None, None
        for j, p in enumerate(pred_leaks):
            if j in used_pred or int(p["pipe_id"]) != t_pipe:
                continue
            err = abs(float(p["position"]) - t_pos)
            if err <= pos_tol and (best_err is None or err < best_err):
                best_err, best_j = err, j
        if best_j is not None:
            used_pred.add(best_j)
            pipe_stats[t_pipe]["tp"] += 1
            pipe_stats[t_pipe]["pos_errors"].append(float(best_err))
        else:
            pipe_stats[t_pipe]["fn"] += 1
    for j, p in enumerate(pred_leaks):
        if j not in used_pred:
            pp = int(p["pipe_id"])
            if 1 <= pp <= NUM_PIPES:
                pipe_stats[pp]["fp"] += 1
    true_pipes = [int(l["pipe_id"]) for l in true_leaks]
    pred_pipes = [int(l["pipe_id"]) for l in pred_leaks]
    for tp in true_pipes:
        for pp in pred_pipes:
            if 1 <= tp <= NUM_PIPES and 1 <= pp <= NUM_PIPES:
                pipe_stats[tp]["conf"][pp] += 1


def pipe_prf_df(pipe_stats):
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        tp, fp, fn = pipe_stats[pid]["tp"], pipe_stats[pid]["fp"], pipe_stats[pid]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        rows.append({"pipe_id": pid, "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f1": f1,
                     "support_true": tp + fn, "predicted_total": tp + fp})
    return pd.DataFrame(rows)


def pipe_pos_error_df(pipe_stats):
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        errs = np.array(pipe_stats[pid]["pos_errors"], dtype=float)
        rows.append({
            "pipe_id": pid, "n_matched": int(errs.size),
            "pos_mae":    float(np.mean(errs))   if errs.size > 0 else np.nan,
            "pos_median": float(np.median(errs)) if errs.size > 0 else np.nan,
            "pos_std":    float(np.std(errs))    if errs.size > 0 else np.nan,
            "pos_max":    float(np.max(errs))    if errs.size > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def pipe_confusion_df(pipe_stats):
    rows = []
    for tp in range(1, NUM_PIPES + 1):
        row = {"true_pipe": tp}
        for pp in range(1, NUM_PIPES + 1):
            row[f"pred_pipe_{pp}"] = pipe_stats[tp]["conf"][pp]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="test_data_results")
    parser.add_argument("--bundle",      type=str, default="multileak_stgcn_bundle_v1.pt")
    parser.add_argument("--pos_tol",     type=float, default=0.25)
    parser.add_argument("--device",      type=str, default="cpu")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    bundle_path = Path(args.bundle)
    device      = args.device

    if not results_dir.exists():
        raise RuntimeError(f"results_dir not found: {results_dir.resolve()}")
    if not bundle_path.exists():
        raise RuntimeError(f"bundle not found: {bundle_path.resolve()}")

    bundle = torch.load(str(bundle_path), map_location=device, weights_only=False)

    mu                = np.array(bundle["mu"],                dtype=np.float32)
    sigma             = np.array(bundle["sigma"],             dtype=np.float32)
    sensor_names      = bundle["sensor_names"]
    baseline_template = np.array(bundle["baseline_template"], dtype=np.float32)
    adjacency         = np.array(bundle["adjacency"],         dtype=np.float32)
    window      = int(bundle["window"])
    stride      = int(bundle.get("stride", 10))
    hidden_1    = int(bundle.get("hidden_1", 16))
    hidden_2    = int(bundle.get("hidden_2", 32))
    kernel_size = int(bundle.get("kernel_size", 5))
    dropout     = float(bundle.get("dropout", 0.25))

    model = MultiTaskSTGCN(
        adj_matrix=adjacency, hidden_1=hidden_1, hidden_2=hidden_2,
        kernel_size=kernel_size, dropout=dropout
    ).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    scn_dirs = sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("scn_")],
        key=lambda p: int(p.name.split("_")[1])
    )
    if len(scn_dirs) == 0:
        raise RuntimeError(f"No scn_* folders found in {results_dir.resolve()}")

    out_dir = results_dir / "evaluation_stgcn_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rows        = []
    by_true_count   = {0: [], 1: [], 2: [], 3: []}
    pipe_only_stats = init_pipe_stats()
    pipe_pos_stats  = init_pipe_stats()
    all_true_counts = []
    all_pred_counts = []
    all_pos_pairs   = []

    for scn_dir in scn_dirs:
        signals_path = scn_dir / "signals.csv"
        labels_path  = scn_dir / "labels.json"
        if not signals_path.exists() or not labels_path.exists():
            continue

        signals    = pd.read_csv(signals_path)
        labels     = json.loads(labels_path.read_text(encoding="utf-8", errors="ignore"))
        true_leaks = labels.get("Leaks", [])
        if not isinstance(true_leaks, list):
            true_leaks = labels.get("leaks", [])
        true_count = int(len(true_leaks))

        pred = predict_from_signals_df(
            signals=signals, model=model, sensor_names=sensor_names,
            baseline_template=baseline_template, mu=mu, sigma=sigma,
            window=window, stride=stride, device=device
        )
        pred_leaks = pred["predicted_leaks"]
        pred_count = int(pred["predicted_leak_count"])

        all_true_counts.append(true_count)
        all_pred_counts.append(pred_count)

        tp_pipe, fp_pipe, fn_pipe = match_pipes_only(true_leaks, pred_leaks)
        p_pipe, r_pipe, f1_pipe   = prf(tp_pipe, fp_pipe, fn_pipe)

        tp_pos, fp_pos, fn_pos, _pos_errs_tol = match_leaks_pipe_and_pos(
            true_leaks, pred_leaks, pos_tol=args.pos_tol
        )
        p_pos, r_pos, f1_pos = prf(tp_pos, fp_pos, fn_pos)

        tp_loc, fp_loc, fn_loc, pos_errs_all, scn_pos_pairs = match_leaks_pipe_and_pos_no_tol(
            true_leaks, pred_leaks
        )
        pos_mae = float(np.mean(pos_errs_all)) if len(pos_errs_all) > 0 else np.nan
        all_pos_pairs.extend(scn_pos_pairs)

        count_error = abs(pred_count - true_count)
        exact_count = 1 if pred_count == true_count else 0
        exact_match = 1 if (true_count == pred_count == tp_pos and fp_pos == 0 and fn_pos == 0) else 0

        true_pipes_sorted = sorted([int(l["pipe_id"]) for l in true_leaks])
        pred_pipes_sorted = sorted([int(l["pipe_id"]) for l in pred_leaks])

        row = {
            "scenario":   scn_dir.name,
            "scn_number": int(labels.get("scn_number", scn_dir.name.split("_")[1])),
            "source_inp": labels.get("source_inp", ""),
            "true_count": true_count, "pred_count": pred_count,
            "count_error": int(count_error), "exact_count": int(exact_count), "exact_match": int(exact_match),
            "tp_pipe": tp_pipe, "fp_pipe": fp_pipe, "fn_pipe": fn_pipe,
            "precision_pipe": p_pipe, "recall_pipe": r_pipe, "f1_pipe": f1_pipe,
            "tp_pos": tp_pos, "fp_pos": fp_pos, "fn_pos": fn_pos,
            "precision_pos": p_pos, "recall_pos": r_pos, "f1_pos": f1_pos,
            "pos_mae_matched": pos_mae, "num_windows": pred["num_windows"],
            "true_pipes": str(true_pipes_sorted), "pred_pipes": str(pred_pipes_sorted),
        }
        per_rows.append(row)
        if true_count in by_true_count:
            by_true_count[true_count].append(row)

        update_pipe_only(pipe_only_stats, true_leaks, pred_leaks)
        update_pipe_pos(pipe_pos_stats, true_leaks, pred_leaks, pos_tol=args.pos_tol)

        with open(scn_dir / "prediction.json", "w", encoding="utf-8") as f:
            json.dump({
                "source_inp": labels.get("source_inp", ""),
                "scn_number": int(labels.get("scn_number", -1)),
                "predicted_leak_count": pred_count,
                "predicted_leaks": pred_leaks,
            }, f, indent=2)

    per_df  = pd.DataFrame(per_rows).sort_values("scn_number")
    per_csv = out_dir / "per_scenario_metrics.csv"
    per_df.to_csv(per_csv, index=False)

    import ast
    misclassification_rows = []
    for row in per_rows:
        if row["true_count"] == 0 or (row["fp_pipe"] == 0 and row["fn_pipe"] == 0):
            continue
        true_pipes_list = ast.literal_eval(row["true_pipes"])
        pred_pipes_list = ast.literal_eval(row["pred_pipes"])
        used, matched_true = set(), []
        for tp in sorted(true_pipes_list):
            for j, pp in enumerate(pred_pipes_list):
                if j in used:
                    continue
                if pp == tp:
                    matched_true.append(tp)
                    used.add(j)
                    break
        missed   = sorted([p for p in true_pipes_list if p not in matched_true
                           or true_pipes_list.count(p) > matched_true.count(p)])
        spurious = sorted([pred_pipes_list[j] for j in range(len(pred_pipes_list)) if j not in used])
        misclassification_rows.append({
            "scn_number": row["scn_number"], "source_inp": row["source_inp"],
            "true_count": row["true_count"], "pred_count": row["pred_count"],
            "true_pipes": str(true_pipes_list), "pred_pipes": str(pred_pipes_list),
            "missed_pipes": str(missed), "spurious_pipes": str(spurious),
            "tp_pipe": row["tp_pipe"], "fp_pipe": row["fp_pipe"], "fn_pipe": row["fn_pipe"],
        })
    misclass_df   = pd.DataFrame(misclassification_rows).sort_values("scn_number")
    misclass_path = out_dir / "pipe_misclassification_log.csv"
    misclass_df.to_csv(misclass_path, index=False)

    count_per_class_df, count_summary, count_conf_raw, count_conf_norm = \
        compute_count_classification_metrics(all_true_counts, all_pred_counts)
    count_per_class_path  = out_dir / "count_classification_per_class.csv"
    count_summary_path    = out_dir / "count_classification_summary.json"
    count_conf_raw_path   = out_dir / "count_confusion_matrix_raw.csv"
    count_conf_norm_path  = out_dir / "count_confusion_matrix_normalised.csv"
    count_per_class_df.to_csv(count_per_class_path, index=False)
    with open(count_summary_path, "w", encoding="utf-8") as f:
        json.dump(count_summary, f, indent=2)
    pd.DataFrame(count_conf_raw,  index=[f"true_{c}" for c in range(4)],
                 columns=[f"pred_{c}" for c in range(4)]).to_csv(count_conf_raw_path)
    pd.DataFrame(np.round(count_conf_norm, 4), index=[f"true_{c}" for c in range(4)],
                 columns=[f"pred_{c}" for c in range(4)]).to_csv(count_conf_norm_path)

    pipe_id_summary_path = out_dir / "pipe_identification_summary.json"
    with open(pipe_id_summary_path, "w", encoding="utf-8") as f:
        json.dump({**compute_pipe_macro_f1(pipe_only_stats), **compute_pipe_exact_match(per_rows)}, f, indent=2)

    pipe_conf_norm = compute_pipe_confusion_normalised(pipe_only_stats)
    pipe_conf_norm_path = out_dir / "pipe_confusion_matrix_normalised.csv"
    pd.DataFrame(np.round(pipe_conf_norm, 4),
                 index=[f"true_pipe_{p}" for p in range(1, NUM_PIPES + 1)],
                 columns=[f"pred_pipe_{p}" for p in range(1, NUM_PIPES + 1)]).to_csv(pipe_conf_norm_path)

    overall_regression       = compute_regression_metrics(all_pos_pairs)
    per_pipe_regression_df   = compute_per_pipe_regression_metrics(all_pos_pairs)
    regression_summary_path  = out_dir / "position_regression_summary.json"
    per_pipe_regression_path = out_dir / "position_regression_per_pipe.csv"
    with open(regression_summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_regression, f, indent=2)
    per_pipe_regression_df.to_csv(per_pipe_regression_path, index=False)

    scatter_df   = pd.DataFrame(all_pos_pairs, columns=["true_position", "pred_position", "pipe_id"]) \
                   if len(all_pos_pairs) > 0 else pd.DataFrame(columns=["true_position", "pred_position", "pipe_id"])
    scatter_df["pipe_id"] = scatter_df["pipe_id"].astype(int)
    scatter_path = out_dir / "position_scatter_data.csv"
    scatter_df.to_csv(scatter_path, index=False)

    def summarize(rows: List[Dict]) -> Dict:
        if len(rows) == 0:
            return {}
        df = pd.DataFrame(rows)
        tp_p, fp_p, fn_p = int(df["tp_pipe"].sum()), int(df["fp_pipe"].sum()), int(df["fn_pipe"].sum())
        tp_d, fp_d, fn_d = int(df["tp_pos"].sum()),  int(df["fp_pos"].sum()),  int(df["fn_pos"].sum())
        p_pipe, r_pipe, f1_pipe = prf(tp_p, fp_p, fn_p)
        p_pos,  r_pos,  f1_pos  = prf(tp_d, fp_d, fn_d)
        pos_mae_vals = df["pos_mae_matched"].dropna().to_numpy()
        return {
            "num_scenarios": int(len(df)),
            "exact_count_rate": float(df["exact_count"].mean()),
            "mean_count_error": float(df["count_error"].mean()),
            "exact_match_rate": float(df["exact_match"].mean()),
            "pipe_only_precision_micro": float(p_pipe), "pipe_only_recall_micro": float(r_pipe),
            "pipe_only_f1_micro": float(f1_pipe),
            "pipe_pos_precision_micro": float(p_pos), "pipe_pos_recall_micro": float(r_pos),
            "pipe_pos_f1_micro": float(f1_pos),
            "localization_mae_on_matched": float(np.mean(pos_mae_vals)) if pos_mae_vals.size > 0 else float("nan"),
        }

    overall = summarize(per_rows)
    bycount = {k: summarize(v) for k, v in by_true_count.items()}

    fpr = None
    if len(by_true_count[0]) > 0:
        fpr = float((pd.DataFrame(by_true_count[0])["pred_count"] > 0).mean())
    overall["false_positive_rate_no_leak"] = float(fpr) if fpr is not None else float("nan")
    overall["position_tolerance"] = float(args.pos_tol)
    overall["window"] = int(window)
    overall["stride"] = int(stride)
    overall["bundle"] = str(bundle_path.name)

    with open(out_dir / "overall_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)

    bycount_df  = pd.DataFrame([{"true_count": k, **v} for k, v in bycount.items() if v])
    bycount_csv = out_dir / "summary_by_true_count.csv"
    bycount_df.to_csv(bycount_csv, index=False)

    pipe_only_df = pipe_prf_df(pipe_only_stats)
    pipe_pos_df  = pipe_prf_df(pipe_pos_stats)
    pos_err_df   = pipe_pos_error_df(pipe_pos_stats)
    conf_only_df = pipe_confusion_df(pipe_only_stats)
    conf_pos_df  = pipe_confusion_df(pipe_pos_stats)

    pipe_only_path = out_dir / "per_pipe_pipe_only.csv"
    pipe_pos_path  = out_dir / "per_pipe_pipe_pos_tol.csv"
    pos_err_path   = out_dir / "per_pipe_pos_error_stats.csv"
    conf_only_path = out_dir / "per_pipe_confusions_pipe_only.csv"
    conf_pos_path  = out_dir / "per_pipe_confusions_pipe_pos.csv"

    pipe_only_df.to_csv(pipe_only_path, index=False)
    pipe_pos_df.to_csv(pipe_pos_path,   index=False)
    pos_err_df.to_csv(pos_err_path,     index=False)
    conf_only_df.to_csv(conf_only_path, index=False)
    conf_pos_df.to_csv(conf_pos_path,   index=False)

    print(f"[OK] per_scenario_metrics.csv       : {per_csv}")
    print(f"[OK] pipe_misclassification_log.csv : {misclass_path}")
    print(f"[OK] overall_summary.json           : {out_dir / 'overall_summary.json'}")
    print(f"[OK] summary_by_true_count.csv      : {bycount_csv}")
    print(f"[INFO] position tolerance: {args.pos_tol}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        Path("FAILED_evaluate_model_log.txt").write_text(tb, encoding="utf-8")
        raise
