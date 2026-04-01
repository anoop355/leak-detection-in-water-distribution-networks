"""
multi_model_evaluate.py  (REVISED)

Changes vs original:
  FIX 1  : Removed redundant correct_pipe_f1_micro metric. match_pipes_only and
            match_correct_pipe_any_pos produced identical F1 scores because both
            perform greedy pipe-ID-only matching. Only match_pipes_only is retained
            for pipe identification F1. The position error collection is still handled
            via match_correct_pipe_any_pos (pipe-correct, any position), which is
            correct for MAE/RMSE/R2.

  FIX 2  : Added pipe macro F1 to per-model summary and global comparison table.
            Macro F1 averages per-pipe F1 scores equally across all five pipes,
            consistent with the single-model evaluator (evaluate_model.py).

  FIX 3  : Added count classification macro F1 and per-class precision/recall/F1
            to per-model outputs (count_classification_per_class.csv,
            count_classification_summary.json). Consistent with evaluate_model.py.

  FIX 4  : Added position RMSE and R2 (overall and per-pipe) to per-model outputs.
            pos_pairs (true_pos, pred_pos, pipe_id) are now collected per model,
            enabling regression metrics and scatter plot data export.

  FIX 5  : Changed POS_TOL from 0.10 to 0.25 (normalised pipe length), per WASA
            confirmation that a 25% pipe-length search zone is operationally
            acceptable for field crew deployment.

  FIX 6  : Added pipe+position macro F1 (pipe_pos_tol_macro_f1). Computed by
            averaging per-pipe F1 scores equally across all five pipes using the
            tolerance-based matching stats (pipe_tol_stats). This is the primary
            ranking criterion for sensor configuration comparison in Section X.3.2.
            Consistent with how pipe_macro_f1 is computed from pipe_correct_stats.

  NOTE   : FPR=1.0 for several reduced-sensor models (S6-C, S4-C, S4-E, S2-D,
            S2-F, S2-G, S1-E) is a genuine result, not a code error. Those sensor
            subsets produce leak-like signatures even under no-leak conditions,
            making the trained TCN unable to distinguish normal operation.
"""

import os
import json
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ============================================================
# USER SETTINGS (EDIT THESE ONLY)
# ============================================================

TEST_RESULTS_DIR      = Path("test_data_results")
TRAINED_MODELS_DIR    = Path("trained_models")
SENSOR_PLACEMENTS_CSV = Path("sensor_placements.csv")
OUTPUT_ROOT           = Path("sensor_placement_results")
POS_TOL               = 0.25   # FIX 5: changed from 0.10 to 0.25 (WASA-confirmed threshold)
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PER_SCENARIO_PREDICTIONS = False

# ============================================================
# MODEL (must match training)
# ============================================================
MAX_LEAKS     = 3
NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES  = NUM_PIPES + 1
SIZE_CLASSES  = 4


class MultiLeakTCN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=4,  dilation=1), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=8,  dilation=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=16, dilation=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.count_head = nn.Linear(32, 4)
        self.pipe_head  = nn.Linear(32, MAX_LEAKS * PIPE_CLASSES)
        self.size_head  = nn.Linear(32, MAX_LEAKS * SIZE_CLASSES)
        self.pos_head   = nn.Linear(32, MAX_LEAKS)

    def forward(self, x):
        z = self.backbone(x)
        count_logits = self.count_head(z)
        pipe_logits  = self.pipe_head(z).view(-1, MAX_LEAKS, PIPE_CLASSES)
        size_logits  = self.size_head(z).view(-1, MAX_LEAKS, SIZE_CLASSES)
        pos_pred     = self.pos_head(z).view(-1, MAX_LEAKS)
        return count_logits, pipe_logits, size_logits, pos_pred


# ============================================================
# Helpers
# ============================================================

def safe_float(x, default=np.nan):
    try:    return float(x)
    except: return float(default)

def safe_int(x, default=-1):
    try:    return int(x)
    except: return int(default)

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def majority_vote(arr, valid_mask=None, default=None):
    if valid_mask is not None:
        arr = arr[valid_mask]
    if arr.size == 0:
        return default
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])

def mean_over_valid(values, valid_mask, default=0.0):
    v = values[valid_mask]
    return float(np.mean(v)) if v.size > 0 else float(default)

def prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0.0
    return precision, recall, f1

def load_labels(path):
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))

def extract_leaks(labels):
    return labels.get("Leaks", labels.get("leaks", [])) or []

def list_scn_dirs(results_dir):
    return sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("scn_")],
        key=lambda p: safe_int(p.name.split("_")[1], default=10**9)
    )


# ============================================================
# Prediction
# ============================================================

def predict_from_signals_df(signals, model, feature_cols, mu, sigma, window, stride, device="cpu"):
    missing = [c for c in feature_cols if c not in signals.columns]
    if missing:
        raise ValueError(f"signals.csv missing columns: {missing}")

    X = signals[feature_cols].to_numpy(dtype=np.float32)
    if len(X) < window:
        return {"predicted_leak_count": 0, "predicted_leaks": [], "num_windows": 0}

    windows = []
    for end in range(window, len(X) + 1, stride):
        x_win = (X[end-window:end, :] - mu) / (sigma + 1e-8)
        windows.append(torch.tensor(x_win, dtype=torch.float32).transpose(0, 1).unsqueeze(0))

    X_batch = torch.cat(windows, dim=0).to(device)

    with torch.no_grad():
        count_logits, pipe_logits, _, pos_pred = model(X_batch)

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

    return {"predicted_leak_count": int(final_count),
            "predicted_leaks": final_leaks,
            "num_windows": int(X_batch.shape[0])}


# ============================================================
# Matching
# ============================================================

def match_pipes_only(true_leaks, pred_leaks):
    """Greedy pipe-ID-only matching. Used for pipe identification F1."""
    true_pipes = [safe_int(l.get("pipe_id", -1)) for l in true_leaks]
    pred_pipes = [safe_int(l.get("pipe_id", -1)) for l in pred_leaks]
    TP, used = 0, set()
    for tp in sorted(true_pipes):
        for j, pp in enumerate(pred_pipes):
            if j in used: continue
            if pp == tp:
                TP += 1; used.add(j); break
    return TP, len(pred_pipes) - TP, len(true_pipes) - TP


def match_pipe_and_pos_with_tol(true_leaks, pred_leaks, pos_tol):
    """Pipe must match AND position error <= pos_tol."""
    used_pred, pos_errors, TP = set(), [], 0
    for t in sorted(true_leaks, key=lambda d: (safe_int(d.get("pipe_id", 999)), safe_float(d.get("position", 0)))):
        t_pipe, t_pos = safe_int(t.get("pipe_id", -1)), safe_float(t.get("position", np.nan))
        best_j, best_err = None, None
        for j, p in enumerate(pred_leaks):
            if j in used_pred: continue
            if safe_int(p.get("pipe_id", -1)) != t_pipe: continue
            err = abs(safe_float(p.get("position", 0.0)) - t_pos)
            if err <= pos_tol and (best_err is None or err < best_err):
                best_err, best_j = err, j
        if best_j is not None:
            used_pred.add(best_j); TP += 1; pos_errors.append(float(best_err))
    return TP, len(pred_leaks) - TP, len(true_leaks) - TP, pos_errors


def match_correct_pipe_any_pos(true_leaks, pred_leaks):
    """
    Pipe must match, no position threshold.
    Returns TP/FP/FN, position errors, AND raw (true_pos, pred_pos, pipe_id) pairs
    needed for RMSE and R2 computation.
    FIX 4: Now also returns pos_pairs for regression metrics.
    """
    used_pred, pos_errors, pos_pairs, TP = set(), [], [], 0
    for t in sorted(true_leaks, key=lambda d: (safe_int(d.get("pipe_id", 999)), safe_float(d.get("position", 0)))):
        t_pipe  = safe_int(t.get("pipe_id", -1))
        t_pos   = safe_float(t.get("position", np.nan))
        best_j, best_err, best_pred_pos = None, None, None
        for j, p in enumerate(pred_leaks):
            if j in used_pred: continue
            if safe_int(p.get("pipe_id", -1)) != t_pipe: continue
            err = abs(safe_float(p.get("position", 0.0)) - t_pos)
            if best_err is None or err < best_err:
                best_err, best_j = err, j
                best_pred_pos = safe_float(p.get("position", 0.0))
        if best_j is not None:
            used_pred.add(best_j); TP += 1
            pos_errors.append(float(best_err))
            pos_pairs.append((t_pos, best_pred_pos, t_pipe))  # FIX 4
    return TP, len(pred_leaks) - TP, len(true_leaks) - TP, pos_errors, pos_pairs


# ============================================================
# FIX 2 + FIX 3: Count classification and pipe macro F1 helpers
# (ported from evaluate_model.py for consistency)
# ============================================================

def compute_count_classification_metrics(true_counts, pred_counts):
    """Per-class precision/recall/F1, macro F1, micro F1, accuracy. (FIX 3)"""
    classes  = [0, 1, 2, 3]
    true_arr = np.array(true_counts, dtype=int)
    pred_arr = np.array(pred_counts, dtype=int)

    accuracy    = float(np.mean(true_arr == pred_arr))
    conf_matrix = np.zeros((4, 4), dtype=int)
    for t, p in zip(true_arr, pred_arr):
        if 0 <= t <= 3 and 0 <= p <= 3:
            conf_matrix[t, p] += 1

    row_sums  = conf_matrix.sum(axis=1, keepdims=True).astype(float)
    conf_norm = np.where(row_sums > 0, conf_matrix / row_sums, 0.0)

    rows, f1_scores = [], []
    total_tp = total_fp = total_fn = 0
    for c in classes:
        tp = int(np.sum((true_arr == c) & (pred_arr == c)))
        fp = int(np.sum((true_arr != c) & (pred_arr == c)))
        fn = int(np.sum((true_arr == c) & (pred_arr != c)))
        p, r, f1 = prf(tp, fp, fn)
        f1_scores.append(f1)
        total_tp += tp; total_fp += fp; total_fn += fn
        rows.append({"leak_count_class": c, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(p, 4), "recall": round(r, 4),
                     "f1": round(f1, 4), "support": int(np.sum(true_arr == c))})

    p_micro, r_micro, micro_f1 = prf(total_tp, total_fp, total_fn)
    summary = {
        "accuracy":        round(accuracy, 4),
        "macro_f1":        round(float(np.mean(f1_scores)), 4),
        "micro_f1":        round(micro_f1, 4),
        "micro_precision": round(p_micro, 4),
        "micro_recall":    round(r_micro, 4),
        "num_scenarios":   int(len(true_arr))
    }
    return pd.DataFrame(rows), summary, conf_matrix, conf_norm


def compute_pipe_macro_f1(pipe_stats):
    """
    Macro F1 averaged equally over all five pipes. (FIX 2)
    Used for pipe-only identification (no position threshold).
    """
    f1_scores, total_tp, total_fp, total_fn = [], 0, 0, 0
    for pid in range(1, NUM_PIPES + 1):
        tp, fp, fn = pipe_stats[pid]["tp"], pipe_stats[pid]["fp"], pipe_stats[pid]["fn"]
        _, _, f1 = prf(tp, fp, fn)
        f1_scores.append(f1)
        total_tp += tp; total_fp += fp; total_fn += fn
    _, _, micro_f1 = prf(total_tp, total_fp, total_fn)
    return {"pipe_macro_f1": round(float(np.mean(f1_scores)), 4),
            "pipe_micro_f1": round(micro_f1, 4)}


def compute_pipe_pos_tol_macro_f1(pipe_stats):
    """
    FIX 6: Pipe+position macro F1 averaged equally over all five pipes.
    Uses pipe_tol_stats, where a match requires both correct pipe identification
    AND position error within POS_TOL (0.25 normalised pipe length).
    This is the primary ranking criterion for sensor configuration comparison
    in Section X.3.2.
    Micro F1 also returned for supplementary comparison.
    """
    f1_scores, total_tp, total_fp, total_fn = [], 0, 0, 0
    for pid in range(1, NUM_PIPES + 1):
        tp, fp, fn = pipe_stats[pid]["tp"], pipe_stats[pid]["fp"], pipe_stats[pid]["fn"]
        _, _, f1 = prf(tp, fp, fn)
        f1_scores.append(f1)
        total_tp += tp; total_fp += fp; total_fn += fn
    _, _, micro_f1 = prf(total_tp, total_fp, total_fn)
    return {"pipe_pos_tol_macro_f1": round(float(np.mean(f1_scores)), 4),
            "pipe_pos_tol_micro_f1": round(micro_f1, 4)}


def compute_pipe_exact_match(per_rows):
    leak_rows = [r for r in per_rows if r["true_count"] > 0]
    if not leak_rows:
        return {"pipe_exact_match_accuracy": float("nan"), "n_leak_scenarios": 0}
    exact = sum(1 for r in leak_rows
                if r["tp_pipe"] == r["true_count"] and r["fp_pipe"] == 0 and r["fn_pipe"] == 0)
    return {"pipe_exact_match_accuracy": round(exact / len(leak_rows), 4),
            "pipe_exact_match_count": exact,
            "n_leak_scenarios": len(leak_rows)}


# ============================================================
# FIX 4: Regression metrics (RMSE, R2)
# ============================================================

def compute_regression_metrics(pos_pairs):
    """Overall MAE, RMSE, R2 from (true_pos, pred_pos, pipe_id) tuples."""
    if not pos_pairs:
        return {"mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "n_matched": 0}
    true_vals = np.array([p[0] for p in pos_pairs], dtype=float)
    pred_vals = np.array([p[1] for p in pos_pairs], dtype=float)
    errors    = pred_vals - true_vals
    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": round(mae, 6), "rmse": round(rmse, 6),
            "r2": round(r2, 6), "n_matched": int(len(pos_pairs))}


def compute_per_pipe_regression_metrics(pos_pairs):
    """MAE, RMSE, R2 per pipe."""
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        pairs = [(t, p) for t, p, pipe in pos_pairs if pipe == pid]
        if not pairs:
            rows.append({"pipe_id": pid, "n_matched": 0,
                         "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")})
            continue
        true_v = np.array([x[0] for x in pairs], dtype=float)
        pred_v = np.array([x[1] for x in pairs], dtype=float)
        errors = pred_v - true_v
        mae  = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        ss_res = float(np.sum(errors ** 2))
        ss_tot = float(np.sum((true_v - np.mean(true_v)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        rows.append({"pipe_id": pid, "n_matched": len(pairs),
                     "mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6)})
    return pd.DataFrame(rows)


# ============================================================
# Per-pipe stats helpers (unchanged from original)
# ============================================================

def init_pipe_stats():
    return {pid: {"tp": 0, "fp": 0, "fn": 0, "pos_errors": [],
                  "conf": {p: 0 for p in range(1, NUM_PIPES + 1)}}
            for pid in range(1, NUM_PIPES + 1)}


def update_pipe_stats_correct_pipe_any_pos(pipe_stats, true_leaks, pred_leaks):
    true_list = [(safe_int(l.get("pipe_id", -1)), safe_float(l.get("position", np.nan))) for l in true_leaks]
    pred_list = [(safe_int(l.get("pipe_id", -1)), safe_float(l.get("position", np.nan))) for l in pred_leaks]

    used_pred, matched_pairs = set(), []
    for ti, (t_pipe, t_pos) in enumerate(true_list):
        best, best_err = None, None
        for pj, (p_pipe, p_pos) in enumerate(pred_list):
            if pj in used_pred or p_pipe != t_pipe: continue
            err = abs(p_pos - t_pos)
            if best_err is None or err < best_err:
                best_err, best = err, pj
        if best is not None:
            used_pred.add(best); matched_pairs.append((ti, best))
            pipe_stats[t_pipe]["tp"] += 1
            pipe_stats[t_pipe]["pos_errors"].append(float(best_err))

    matched_true  = {ti for ti, _ in matched_pairs}
    unmatched_true = [ti for ti in range(len(true_list)) if ti not in matched_true]
    unmatched_pred = [pj for pj in range(len(pred_list)) if pj not in used_pred]

    for ti in unmatched_true:
        t_pipe, _ = true_list[ti]
        if 1 <= t_pipe <= NUM_PIPES: pipe_stats[t_pipe]["fn"] += 1
    for pj in unmatched_pred:
        p_pipe, _ = pred_list[pj]
        if 1 <= p_pipe <= NUM_PIPES: pipe_stats[p_pipe]["fp"] += 1

    conf_pairs = matched_pairs.copy()
    if unmatched_true and unmatched_pred:
        candidates = sorted([(abs(pred_list[pj][1] - true_list[ti][1]), ti, pj)
                              for ti in unmatched_true for pj in unmatched_pred])
        used_t = set(ti for ti, _ in matched_pairs)
        used_p = set(pj for _, pj in matched_pairs)
        for _, ti, pj in candidates:
            if ti in used_t or pj in used_p: continue
            used_t.add(ti); used_p.add(pj); conf_pairs.append((ti, pj))

    for ti, pj in conf_pairs:
        t_pipe, _ = true_list[ti]; p_pipe, _ = pred_list[pj]
        if 1 <= t_pipe <= NUM_PIPES and 1 <= p_pipe <= NUM_PIPES:
            pipe_stats[t_pipe]["conf"][p_pipe] += 1


def update_pipe_stats_pipe_pos_tol(pipe_stats, true_leaks, pred_leaks, pos_tol):
    true_list = [(safe_int(l.get("pipe_id", -1)), safe_float(l.get("position", np.nan))) for l in true_leaks]
    pred_list = [(safe_int(l.get("pipe_id", -1)), safe_float(l.get("position", np.nan))) for l in pred_leaks]

    used_pred, matched_pairs = set(), []
    for ti, (t_pipe, t_pos) in enumerate(true_list):
        best, best_err = None, None
        for pj, (p_pipe, p_pos) in enumerate(pred_list):
            if pj in used_pred or p_pipe != t_pipe: continue
            err = abs(p_pos - t_pos)
            if err <= pos_tol and (best_err is None or err < best_err):
                best_err, best = err, pj
        if best is not None:
            used_pred.add(best); matched_pairs.append((ti, best))
            pipe_stats[t_pipe]["tp"] += 1
            pipe_stats[t_pipe]["pos_errors"].append(float(best_err))

    matched_true  = {ti for ti, _ in matched_pairs}
    unmatched_true = [ti for ti in range(len(true_list)) if ti not in matched_true]
    unmatched_pred = [pj for pj in range(len(pred_list)) if pj not in used_pred]

    for ti in unmatched_true:
        t_pipe, _ = true_list[ti]
        if 1 <= t_pipe <= NUM_PIPES: pipe_stats[t_pipe]["fn"] += 1
    for pj in unmatched_pred:
        p_pipe, _ = pred_list[pj]
        if 1 <= p_pipe <= NUM_PIPES: pipe_stats[p_pipe]["fp"] += 1

    conf_pairs = matched_pairs.copy()
    if unmatched_true and unmatched_pred:
        candidates = sorted([(abs(pred_list[pj][1] - true_list[ti][1]), ti, pj)
                              for ti in unmatched_true for pj in unmatched_pred])
        used_t = set(ti for ti, _ in matched_pairs)
        used_p = set(pj for _, pj in matched_pairs)
        for _, ti, pj in candidates:
            if ti in used_t or pj in used_p: continue
            used_t.add(ti); used_p.add(pj); conf_pairs.append((ti, pj))

    for ti, pj in conf_pairs:
        t_pipe, _ = true_list[ti]; p_pipe, _ = pred_list[pj]
        if 1 <= t_pipe <= NUM_PIPES and 1 <= p_pipe <= NUM_PIPES:
            pipe_stats[t_pipe]["conf"][p_pipe] += 1


def pipe_prf_df(pipe_stats):
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        tp, fp, fn = pipe_stats[pid]["tp"], pipe_stats[pid]["fp"], pipe_stats[pid]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        rows.append({"pipe_id": pid, "tp": tp, "fp": fp, "fn": fn,
                     "precision": p, "recall": r, "f1": f1,
                     "support_true": tp + fn, "predicted_total": tp + fp})
    return pd.DataFrame(rows)


def pipe_pos_error_df(pipe_stats, prefix=""):
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        errs = np.array(pipe_stats[pid]["pos_errors"], dtype=float)
        rows.append({"pipe_id": pid,
                     f"{prefix}n_matched":  int(errs.size),
                     f"{prefix}pos_mae":    float(np.mean(errs))   if errs.size > 0 else np.nan,
                     f"{prefix}pos_median": float(np.median(errs)) if errs.size > 0 else np.nan,
                     f"{prefix}pos_std":    float(np.std(errs))    if errs.size > 0 else np.nan,
                     f"{prefix}pos_max":    float(np.max(errs))    if errs.size > 0 else np.nan})
    return pd.DataFrame(rows)


def pipe_confusion_df(pipe_stats):
    rows = []
    for tp in range(1, NUM_PIPES + 1):
        row = {"true_pipe": tp}
        for pp in range(1, NUM_PIPES + 1):
            row[f"pred_pipe_{pp}"] = pipe_stats[tp]["conf"][pp]
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Summary builder
# ============================================================

def summarize_per_scenario(per_rows):
    if not per_rows:
        return {}
    df  = pd.DataFrame(per_rows)
    out = {"num_scenarios": int(len(df))}

    out["exact_count_rate"] = float(df["exact_count"].mean())
    out["mean_count_error"] = float(df["count_error"].mean())

    # FIX 1: Only report pipe_only micro F1 (correct_pipe was a duplicate)
    tp_p, fp_p, fn_p = int(df["tp_pipe"].sum()), int(df["fp_pipe"].sum()), int(df["fn_pipe"].sum())
    p_p, r_p, f1_p = prf(tp_p, fp_p, fn_p)
    out["pipe_only_precision_micro"] = float(p_p)
    out["pipe_only_recall_micro"]    = float(r_p)
    out["pipe_only_f1_micro"]        = float(f1_p)

    # Tolerance-based localization micro
    tp_t, fp_t, fn_t = int(df["tp_pos_tol"].sum()), int(df["fp_pos_tol"].sum()), int(df["fn_pos_tol"].sum())
    p_t, r_t, f1_t = prf(tp_t, fp_t, fn_t)
    out["pipe_pos_tol_precision_micro"] = float(p_t)
    out["pipe_pos_tol_recall_micro"]    = float(r_t)
    out["pipe_pos_tol_f1_micro"]        = float(f1_t)

    # MAE on correct pipe (tolerance-free)
    loc_vals = df["pos_mae_correct_pipe"].dropna().to_numpy()
    out["localization_mae_on_correct_pipe"] = float(np.mean(loc_vals)) if loc_vals.size > 0 else float("nan")

    out["exact_match_rate_with_tol"] = float(df["exact_match_with_tol"].mean())

    df0 = df[df["true_count"] == 0]
    out["false_positive_rate_no_leak"] = float((df0["pred_count"] > 0).mean()) if len(df0) > 0 else float("nan")

    return out


# ============================================================
# Main
# ============================================================

def main():
    for p in [TEST_RESULTS_DIR, TRAINED_MODELS_DIR, SENSOR_PLACEMENTS_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p.resolve()}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    placements = pd.read_csv(SENSOR_PLACEMENTS_CSV)
    for c in ["model_name", "k_budget", "placement_strategy", "configuration"]:
        if c not in placements.columns:
            raise ValueError(f"sensor_placements.csv missing column: {c}")
    placements["model_name"] = placements["model_name"].astype(str)

    scn_dirs = list_scn_dirs(TEST_RESULTS_DIR)
    if not scn_dirs:
        raise RuntimeError(f"No scn_* folders found in {TEST_RESULTS_DIR.resolve()}")

    all_models_summary_rows  = []
    all_models_per_pipe_rows = []

    for _, row in placements.iterrows():
        model_name = str(row["model_name"]).strip()
        if not model_name:
            continue

        bundle_path = TRAINED_MODELS_DIR / model_name / f"multileak_tcn_bundle_{model_name}.pt"
        if not bundle_path.exists():
            print(f"[WARN] Skipping {model_name}: bundle not found at {bundle_path}")
            continue

        model_eval_dir = OUTPUT_ROOT / model_name / "evaluation"
        model_eval_dir.mkdir(parents=True, exist_ok=True)

        bundle       = torch.load(str(bundle_path), map_location=DEVICE, weights_only=False)
        mu           = np.array(bundle["mu"], dtype=np.float32)
        sigma        = np.array(bundle["sigma"], dtype=np.float32)
        feature_cols = list(bundle["feature_cols"])
        window       = int(bundle["window"])
        stride       = int(bundle.get("stride", 10))

        model = MultiLeakTCN(C=len(feature_cols)).to(DEVICE)
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        per_rows      = []
        by_true_count = {0: [], 1: [], 2: [], 3: []}
        pipe_correct_stats = init_pipe_stats()   # pipe-only, any position
        pipe_tol_stats     = init_pipe_stats()   # pipe + position within POS_TOL

        # FIX 3: collectors for count classification
        all_true_counts = []
        all_pred_counts = []

        # FIX 4: collector for regression metrics
        all_pos_pairs = []

        for scn_dir in scn_dirs:
            signals_path = scn_dir / "signals.csv"
            labels_path  = scn_dir / "labels.json"
            if not signals_path.exists() or not labels_path.exists():
                continue

            signals    = pd.read_csv(signals_path)
            labels     = load_labels(labels_path)
            true_leaks = extract_leaks(labels)
            true_count = int(len(true_leaks))

            pred       = predict_from_signals_df(signals, model, feature_cols,
                                                 mu, sigma, window, stride, DEVICE)
            pred_leaks = pred["predicted_leaks"]
            pred_count = int(pred["predicted_leak_count"])

            # FIX 3: accumulate counts
            all_true_counts.append(true_count)
            all_pred_counts.append(pred_count)

            # Pipe-only matching (FIX 1)
            tp_pipe, fp_pipe, fn_pipe = match_pipes_only(true_leaks, pred_leaks)
            p_pipe, r_pipe, f1_pipe   = prf(tp_pipe, fp_pipe, fn_pipe)

            # Correct pipe, any position — for MAE and pos_pairs (FIX 4)
            tp_cp, fp_cp, fn_cp, pos_errs_cp, scn_pos_pairs = match_correct_pipe_any_pos(
                true_leaks, pred_leaks)
            pos_mae_cp = float(np.mean(pos_errs_cp)) if pos_errs_cp else np.nan
            all_pos_pairs.extend(scn_pos_pairs)  # FIX 4

            # Tolerance-based (FIX 5: now uses POS_TOL=0.25)
            tp_tol, fp_tol, fn_tol, pos_errs_tol = match_pipe_and_pos_with_tol(
                true_leaks, pred_leaks, POS_TOL)
            p_tol, r_tol, f1_tol = prf(tp_tol, fp_tol, fn_tol)
            pos_mae_tol = float(np.mean(pos_errs_tol)) if pos_errs_tol else np.nan

            count_error          = abs(pred_count - true_count)
            exact_count          = 1 if pred_count == true_count else 0
            exact_match_with_tol = 1 if (true_count == pred_count == tp_tol
                                          and fp_tol == 0 and fn_tol == 0) else 0

            true_pipes_sorted = sorted([safe_int(l.get("pipe_id", -1)) for l in true_leaks])
            pred_pipes_sorted = sorted([safe_int(l.get("pipe_id", -1)) for l in pred_leaks])

            per_rows.append({
                "model_name":           model_name,
                "scenario":             scn_dir.name,
                "scn_number":           safe_int(labels.get("scn_number", scn_dir.name.split("_")[1])),
                "source_inp":           str(labels.get("source_inp", "")),
                "true_count":           true_count,
                "pred_count":           pred_count,
                "count_error":          int(count_error),
                "exact_count":          int(exact_count),
                "exact_match_with_tol": int(exact_match_with_tol),
                "true_pipes":           str(true_pipes_sorted),
                "pred_pipes":           str(pred_pipes_sorted),
                "tp_pipe":              tp_pipe,
                "fp_pipe":              fp_pipe,
                "fn_pipe":              fn_pipe,
                "precision_pipe":       p_pipe,
                "recall_pipe":          r_pipe,
                "f1_pipe":              f1_pipe,
                "pos_mae_correct_pipe": pos_mae_cp,
                "tp_pos_tol":           tp_tol,
                "fp_pos_tol":           fp_tol,
                "fn_pos_tol":           fn_tol,
                "precision_pos_tol":    p_tol,
                "recall_pos_tol":       r_tol,
                "f1_pos_tol":           f1_tol,
                "pos_mae_within_tol":   pos_mae_tol,
                "num_windows":          int(pred["num_windows"]),
            })

            if true_count in by_true_count:
                by_true_count[true_count].append(per_rows[-1])

            update_pipe_stats_correct_pipe_any_pos(pipe_correct_stats, true_leaks, pred_leaks)
            update_pipe_stats_pipe_pos_tol(pipe_tol_stats, true_leaks, pred_leaks, POS_TOL)

            if SAVE_PER_SCENARIO_PREDICTIONS:
                out_pred = model_eval_dir / f"{scn_dir.name}_prediction.json"
                out_pred.write_text(json.dumps({
                    "model_name":          model_name,
                    "scn_number":          safe_int(labels.get("scn_number", -1)),
                    "predicted_leak_count": pred_count,
                    "predicted_leaks":     pred_leaks,
                }, indent=2), encoding="utf-8")

        # Convenience: all output filenames are prefixed with model_name
        # e.g. S8-A_per_scenario_metrics.csv, S8-A_pipe_identification_summary.json
        def mf(name):
            """Return model_eval_dir / '{model_name}_{name}'."""
            return model_eval_dir / f"{model_name}_{name}"

        # Per-scenario CSV
        per_df = pd.DataFrame(per_rows).sort_values("scn_number")
        per_df.to_csv(mf("per_scenario_metrics.csv"), index=False)

        # Pipe misclassification log — one row per leak scenario with imperfect pipe identification.
        # Shows true_pipes, pred_pipes, missed_pipes, and spurious_pipes for error analysis.
        import ast
        misclassification_rows = []
        for r in per_rows:
            if r["true_count"] == 0:
                continue  # skip no-leak scenarios
            if r["fp_pipe"] == 0 and r["fn_pipe"] == 0:
                continue  # perfect pipe match — skip
            true_pipes_list = ast.literal_eval(r["true_pipes"])
            pred_pipes_list = ast.literal_eval(r["pred_pipes"])
            used = set()
            matched_true = []
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
                "scn_number":     r["scn_number"],
                "source_inp":     r["source_inp"],
                "true_count":     r["true_count"],
                "pred_count":     r["pred_count"],
                "true_pipes":     str(true_pipes_list),
                "pred_pipes":     str(pred_pipes_list),
                "missed_pipes":   str(missed),
                "spurious_pipes": str(spurious),
                "tp_pipe":        r["tp_pipe"],
                "fp_pipe":        r["fp_pipe"],
                "fn_pipe":        r["fn_pipe"],
            })
        pd.DataFrame(misclassification_rows).sort_values("scn_number").to_csv(
            mf("pipe_misclassification_log.csv"), index=False)

        # FIX 3: Count classification outputs
        count_per_class_df, count_summary, count_conf_raw, count_conf_norm = \
            compute_count_classification_metrics(all_true_counts, all_pred_counts)
        count_per_class_df.to_csv(mf("count_classification_per_class.csv"), index=False)
        mf("count_classification_summary.json").write_text(
            json.dumps(count_summary, indent=2), encoding="utf-8")
        pd.DataFrame(count_conf_raw,
                     index=[f"true_{c}" for c in range(4)],
                     columns=[f"pred_{c}" for c in range(4)]
                     ).to_csv(mf("count_confusion_matrix_raw.csv"))
        pd.DataFrame(np.round(count_conf_norm, 4),
                     index=[f"true_{c}" for c in range(4)],
                     columns=[f"pred_{c}" for c in range(4)]
                     ).to_csv(mf("count_confusion_matrix_normalised.csv"))

        # FIX 2: Pipe-only macro F1 and exact match
        pipe_macro_dict = compute_pipe_macro_f1(pipe_correct_stats)
        pipe_exact_dict = compute_pipe_exact_match(per_rows)
        pipe_id_summary = {**pipe_macro_dict, **pipe_exact_dict}
        mf("pipe_identification_summary.json").write_text(
            json.dumps(pipe_id_summary, indent=2), encoding="utf-8")

        # FIX 6: Pipe+position tolerance macro F1
        pipe_pos_tol_macro_dict = compute_pipe_pos_tol_macro_f1(pipe_tol_stats)
        mf("pipe_pos_tol_macro_f1_summary.json").write_text(
            json.dumps({
                **pipe_pos_tol_macro_dict,
                "pos_tol_used": float(POS_TOL)
            }, indent=2), encoding="utf-8")

        # FIX 4: Regression outputs
        overall_regression  = compute_regression_metrics(all_pos_pairs)
        per_pipe_regression = compute_per_pipe_regression_metrics(all_pos_pairs)
        mf("position_regression_summary.json").write_text(
            json.dumps(overall_regression, indent=2), encoding="utf-8")
        per_pipe_regression.to_csv(mf("position_regression_per_pipe.csv"), index=False)

        if all_pos_pairs:
            scatter_df = pd.DataFrame(all_pos_pairs,
                                      columns=["true_position", "pred_position", "pipe_id"])
            scatter_df["pipe_id"] = scatter_df["pipe_id"].astype(int)
        else:
            scatter_df = pd.DataFrame(columns=["true_position", "pred_position", "pipe_id"])
        scatter_df.to_csv(mf("position_scatter_data.csv"), index=False)

        # Overall summary JSON
        overall = summarize_per_scenario(per_rows)
        overall.update({
            "model_name":          model_name,
            "k_budget":            safe_int(row.get("k_budget", np.nan), default=-1),
            "placement_strategy":  str(row.get("placement_strategy", "")),
            "configuration":       str(row.get("configuration", "")),
            "fitness":             row.get("fitness", ""),
            # FIX 2: pipe-only macro F1
            "pipe_macro_f1":       pipe_macro_dict["pipe_macro_f1"],
            # FIX 6: pipe+position macro F1 (primary ranking criterion for X.3.2)
            "pipe_pos_tol_macro_f1": pipe_pos_tol_macro_dict["pipe_pos_tol_macro_f1"],
            "pipe_pos_tol_micro_f1": pipe_pos_tol_macro_dict["pipe_pos_tol_micro_f1"],
            # FIX 3: count classification
            "count_macro_f1":      count_summary["macro_f1"],
            "count_accuracy":      count_summary["accuracy"],
            # FIX 4: full position metrics
            "position_rmse":       overall_regression["rmse"],
            "position_r2":         overall_regression["r2"],
            "pos_tol_used":        float(POS_TOL),
            "window":              int(window),
            "stride":              int(stride),
            "bundle_path":         str(bundle_path),
        })
        mf("overall_summary.json").write_text(
            json.dumps(overall, indent=2), encoding="utf-8")

        # Summary by true leak count
        bycount_rows = []
        for k in sorted(by_true_count):
            s = summarize_per_scenario(by_true_count[k])
            if s:
                s["true_count"] = k
                bycount_rows.append(s)
        pd.DataFrame(bycount_rows).to_csv(mf("summary_by_true_count.csv"), index=False)

        # Per-pipe CSVs
        pipe_prf_df(pipe_correct_stats).to_csv(
            mf("per_pipe_correct_pipe_any_pos.csv"), index=False)
        pipe_pos_error_df(pipe_correct_stats, prefix="correct_pipe_").to_csv(
            mf("per_pipe_pos_error_correct_pipe_any_pos.csv"), index=False)
        pipe_confusion_df(pipe_correct_stats).to_csv(
            mf("per_pipe_confusions_correct_pipe_any_pos.csv"), index=False)
        pipe_prf_df(pipe_tol_stats).to_csv(
            mf("per_pipe_pipe_pos_tol.csv"), index=False)
        pipe_pos_error_df(pipe_tol_stats, prefix=f"tol_{POS_TOL}_").to_csv(
            mf("per_pipe_pos_error_within_tol.csv"), index=False)
        pipe_confusion_df(pipe_tol_stats).to_csv(
            mf("per_pipe_confusions_pipe_pos_tol.csv"), index=False)

        # Collect for global table
        pipe_prf2 = pipe_prf_df(pipe_correct_stats).copy()
        for col, val in [("model_name",          model_name),
                         ("k_budget",             overall["k_budget"]),
                         ("placement_strategy",   overall["placement_strategy"]),
                         ("configuration",        overall["configuration"])]:
            pipe_prf2.insert(0, col, val)
        all_models_per_pipe_rows.append(pipe_prf2)
        all_models_summary_rows.append(overall)

        print(f"[OK] {model_name} -> {model_eval_dir}")

    # Global comparison table
    if all_models_summary_rows:
        all_df = pd.DataFrame(all_models_summary_rows)
        key_cols = [
            "model_name", "k_budget", "placement_strategy", "configuration", "fitness",
            # FIX 2: pipe-only macro F1
            "pipe_macro_f1", "pipe_only_f1_micro",
            # FIX 6: pipe+position macro F1 (primary ranking criterion)
            "pipe_pos_tol_macro_f1", "pipe_pos_tol_micro_f1",
            # FIX 3: count classification
            "count_macro_f1", "count_accuracy", "exact_count_rate",
            # FIX 4: full position metrics
            "localization_mae_on_correct_pipe", "position_rmse", "position_r2",
            # Existing tolerance micro metrics (retained for reference)
            "pipe_pos_tol_f1_micro", "exact_match_rate_with_tol",
            "false_positive_rate_no_leak",
            "pos_tol_used", "window", "stride", "bundle_path"
        ]
        cols = [c for c in key_cols if c in all_df.columns] + \
               [c for c in all_df.columns if c not in key_cols]
        all_df[cols].to_csv(OUTPUT_ROOT / "all_models_summary.csv", index=False)
        print(f"[OK] {OUTPUT_ROOT / 'all_models_summary.csv'}")

    if all_models_per_pipe_rows:
        pd.concat(all_models_per_pipe_rows, ignore_index=True).to_csv(
            OUTPUT_ROOT / "all_models_per_pipe_correct_pipe_any_pos.csv", index=False)
        print(f"[OK] {OUTPUT_ROOT / 'all_models_per_pipe_correct_pipe_any_pos.csv'}")

    print(f"[INFO] Position tolerance used: {POS_TOL}")
    print("[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "FAILED_multi_model_evaluate_log.txt").write_text(tb, encoding="utf-8")
        raise