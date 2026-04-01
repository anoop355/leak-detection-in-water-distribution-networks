"""
multi_model_evaluate.py

Evaluates ALL trained TCN bundles listed in sensor_placements.csv, one-by-one,
on the SAME test dataset, and saves per-model outputs into:

  sensor_placement_outputs/<model_name>/evaluation/

It also creates a global comparison file:

  sensor_placement_outputs/all_models_summary.csv

Key change vs your older evaluator:
- Computes localization error WITHOUT tolerance gating:
    "MAE on correct pipe" = |pos_pred - pos_true| for matches where pipe_id is correct,
    regardless of whether the error is <= pos_tol.

It still keeps the pos_tol-based "localized correctly" metrics as OPTIONAL additional info.

------------------------------------------------------------
YOU ONLY NEED TO EDIT THE SETTINGS SECTION BELOW.
------------------------------------------------------------
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

# Test dataset folder containing scn_* folders with signals.csv and labels.json
TEST_RESULTS_DIR = Path("test_data_results")

# Folder containing model subfolders:
#   trained_models/<model_name>/multileak_tcn_bundle_<model_name>.pt
TRAINED_MODELS_DIR = Path("trained_models")

# CSV describing each trained model and its sensor configuration
# (your table: model_name, k_budget, placement_strategy, configuration, fitness, ...)
SENSOR_PLACEMENTS_CSV = Path("sensor_placements.csv")

# Where to save all evaluation outputs
OUTPUT_ROOT = Path("sensor_placement_outputs")

# Position tolerance for "localized correctly" (pipe + position within tolerance)
# This is used ONLY for the tolerance-based metrics (not for "MAE on correct pipe").
POS_TOL = 0.10

# Device: "cpu" or "cuda"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# If True, saves per-scenario prediction.json inside each scn folder (audit)
SAVE_PER_SCENARIO_PREDICTIONS = False

# ============================================================
# MODEL (must match training)
# ============================================================
MAX_LEAKS = 3
NUM_PIPES = 5
PIPE_NONE_IDX = NUM_PIPES         # 5  (0..4 are pipes 1..5, 5 means NONE)
PIPE_CLASSES = NUM_PIPES + 1      # 6
SIZE_CLASSES = 4                  # exists in bundle but ignored for output


class MultiLeakTCN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=4, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=8, dilation=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=16, dilation=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.count_head = nn.Linear(32, 4)  # 0..3
        self.pipe_head  = nn.Linear(32, MAX_LEAKS * PIPE_CLASSES)  # 3*6
        self.size_head  = nn.Linear(32, MAX_LEAKS * SIZE_CLASSES)  # 3*4 (ignored)
        self.pos_head   = nn.Linear(32, MAX_LEAKS)                 # 3

    def forward(self, x):
        z = self.backbone(x)
        count_logits = self.count_head(z)
        pipe_logits = self.pipe_head(z).view(-1, MAX_LEAKS, PIPE_CLASSES)
        size_logits = self.size_head(z).view(-1, MAX_LEAKS, SIZE_CLASSES)
        pos_pred = self.pos_head(z).view(-1, MAX_LEAKS)
        return count_logits, pipe_logits, size_logits, pos_pred


# ============================================================
# Helpers
# ============================================================

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)

def safe_int(x, default=-1):
    try:
        return int(x)
    except Exception:
        return int(default)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

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

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def load_labels(labels_path: Path) -> Dict:
    return json.loads(labels_path.read_text(encoding="utf-8", errors="ignore"))

def extract_leaks(labels: Dict) -> List[Dict]:
    # supports "Leaks" or "leaks"
    if "Leaks" in labels:
        return labels.get("Leaks", []) or []
    return labels.get("leaks", []) or []

def list_scn_dirs(results_dir: Path) -> List[Path]:
    return sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("scn_")],
        key=lambda p: safe_int(p.name.split("_")[1], default=10**9)
    )


# ============================================================
# Prediction for one signals.csv
# ============================================================

def predict_from_signals_df(
    signals: pd.DataFrame,
    model: nn.Module,
    feature_cols: List[str],
    mu: np.ndarray,
    sigma: np.ndarray,
    window: int,
    stride: int,
    device: str = "cpu",
) -> Dict:
    missing = [c for c in feature_cols if c not in signals.columns]
    if missing:
        raise ValueError(f"signals.csv missing required columns: {missing}")

    X = signals[feature_cols].to_numpy(dtype=np.float32)
    if len(X) < window:
        return {"predicted_leak_count": 0, "predicted_leaks": [], "num_windows": 0}

    windows = []
    for end in range(window, len(X) + 1, stride):
        x_win = X[end-window:end, :]
        x_win = (x_win - mu) / (sigma + 1e-8)
        x_t = torch.tensor(x_win, dtype=torch.float32).transpose(0, 1).unsqueeze(0)  # (1,C,T)
        windows.append(x_t)

    X_batch = torch.cat(windows, dim=0).to(device)  # (N,C,T)

    with torch.no_grad():
        count_logits, pipe_logits, _size_logits, pos_pred = model(X_batch)

    count_pred = count_logits.argmax(dim=1).cpu().numpy().astype(int)   # (N,)
    pipe_pred  = pipe_logits.argmax(dim=2).cpu().numpy().astype(int)    # (N,3)
    pos_pred   = pos_pred.cpu().numpy().astype(np.float32)              # (N,3)

    final_count = majority_vote(count_pred, default=0)

    final_leaks = []
    for slot in range(MAX_LEAKS):
        slot_pipe = pipe_pred[:, slot]
        slot_pos  = pos_pred[:, slot]

        valid = slot_pipe != PIPE_NONE_IDX
        pipe_vote = majority_vote(slot_pipe, valid_mask=valid, default=PIPE_NONE_IDX)
        pos_mean  = mean_over_valid(slot_pos, valid_mask=valid, default=0.0)

        if pipe_vote == PIPE_NONE_IDX:
            continue

        pipe_id = int(pipe_vote + 1)  # 0..4 -> 1..5
        final_leaks.append({"pipe_id": pipe_id, "position": clamp01(pos_mean)})

    # Sort + enforce count
    final_leaks = sorted(final_leaks, key=lambda d: (int(d["pipe_id"]), float(d["position"])))
    if final_count > 0:
        final_leaks = final_leaks[:final_count]
    else:
        final_leaks = []

    return {
        "predicted_leak_count": int(final_count),
        "predicted_leaks": final_leaks,
        "num_windows": int(X_batch.shape[0]),
    }


# ============================================================
# Matching for metrics
# ============================================================

def match_pipes_only(true_leaks: List[Dict], pred_leaks: List[Dict]) -> Tuple[int, int, int]:
    true_pipes = [safe_int(l.get("pipe_id", -1)) for l in true_leaks]
    pred_pipes = [safe_int(l.get("pipe_id", -1)) for l in pred_leaks]

    TP = 0
    used = set()
    for tp in sorted(true_pipes):
        for j, pp in enumerate(pred_pipes):
            if j in used:
                continue
            if pp == tp:
                TP += 1
                used.add(j)
                break

    FP = len(pred_pipes) - TP
    FN = len(true_pipes) - TP
    return TP, FP, FN


def match_pipe_and_pos_with_tol(true_leaks: List[Dict], pred_leaks: List[Dict], pos_tol: float):
    """
    Pipe must match AND |pos_pred-pos_true| <= pos_tol.
    One-to-one greedy matching by smallest position error.
    """
    used_pred = set()
    pos_errors = []
    TP = 0

    true_sorted = sorted(true_leaks, key=lambda d: (safe_int(d.get("pipe_id", 999)), safe_float(d.get("position", 0.0))))

    for t in true_sorted:
        t_pipe = safe_int(t.get("pipe_id", -1))
        t_pos = safe_float(t.get("position", np.nan))

        best_j = None
        best_err = None

        for j, p in enumerate(pred_leaks):
            if j in used_pred:
                continue
            if safe_int(p.get("pipe_id", -1)) != t_pipe:
                continue
            err = abs(safe_float(p.get("position", 0.0)) - t_pos)
            if err <= pos_tol:
                if best_err is None or err < best_err:
                    best_err = err
                    best_j = j

        if best_j is not None:
            used_pred.add(best_j)
            TP += 1
            pos_errors.append(float(best_err))

    FP = len(pred_leaks) - TP
    FN = len(true_leaks) - TP
    return TP, FP, FN, pos_errors


def match_correct_pipe_any_pos(true_leaks: List[Dict], pred_leaks: List[Dict]):
    """
    Pipe must match, position can be ANYTHING.
    Returns:
      TP, FP, FN, pos_errors_for_correct_pipe_matches
    One-to-one greedy matching by smallest abs position error, but only among same-pipe candidates.
    """
    used_pred = set()
    pos_errors = []
    TP = 0

    true_sorted = sorted(true_leaks, key=lambda d: (safe_int(d.get("pipe_id", 999)), safe_float(d.get("position", 0.0))))

    for t in true_sorted:
        t_pipe = safe_int(t.get("pipe_id", -1))
        t_pos  = safe_float(t.get("position", np.nan))

        best_j = None
        best_err = None

        for j, p in enumerate(pred_leaks):
            if j in used_pred:
                continue
            if safe_int(p.get("pipe_id", -1)) != t_pipe:
                continue
            err = abs(safe_float(p.get("position", 0.0)) - t_pos)
            if best_err is None or err < best_err:
                best_err = err
                best_j = j

        if best_j is not None:
            used_pred.add(best_j)
            TP += 1
            pos_errors.append(float(best_err))

    FP = len(pred_leaks) - TP
    FN = len(true_leaks) - TP
    return TP, FP, FN, pos_errors


# ============================================================
# Per-pipe stats
# ============================================================

def init_pipe_stats():
    stats = {}
    for pid in range(1, NUM_PIPES + 1):
        stats[pid] = {
            "tp": 0, "fp": 0, "fn": 0,
            "pos_errors": [],
            "conf": {p: 0 for p in range(1, NUM_PIPES + 1)}  # true->pred
        }
    return stats


def update_pipe_stats_correct_pipe_any_pos(pipe_stats, true_leaks, pred_leaks):
    """
    Updates per-pipe TP/FP/FN + per-pipe pos_errors (only for correct pipe matches).
    Confusion uses a simple greedy assignment:
      - if correct-pipe match exists -> use it
      - otherwise, match remaining preds to remaining truths by smallest abs pos error
        to produce a sensible true->pred confusion count (instead of an inflated cross-product).
    """
    true_list = [(safe_int(l.get("pipe_id", -1)), safe_float(l.get("position", np.nan))) for l in true_leaks]
    pred_list = [(safe_int(l.get("pipe_id", -1)), safe_float(l.get("position", np.nan))) for l in pred_leaks]

    used_pred = set()
    matched_pairs = []  # (t_idx, p_idx)

    # 1) First, match by same pipe with smallest position error
    for ti, (t_pipe, t_pos) in enumerate(true_list):
        best = None
        best_err = None
        for pj, (p_pipe, p_pos) in enumerate(pred_list):
            if pj in used_pred:
                continue
            if p_pipe != t_pipe:
                continue
            err = abs(p_pos - t_pos)
            if best_err is None or err < best_err:
                best_err = err
                best = pj
        if best is not None:
            used_pred.add(best)
            matched_pairs.append((ti, best))

            pipe_stats[t_pipe]["tp"] += 1
            pipe_stats[t_pipe]["pos_errors"].append(float(best_err))

    # 2) Remaining truths -> FN initially (may be “explained” by confusion matching below)
    matched_true = {ti for ti, _ in matched_pairs}
    unmatched_true = [ti for ti in range(len(true_list)) if ti not in matched_true]

    # 3) Remaining preds -> FP initially
    unmatched_pred = [pj for pj in range(len(pred_list)) if pj not in used_pred]

    # Set FN/FP counts (for correct-pipe metric)
    for ti in unmatched_true:
        t_pipe, _ = true_list[ti]
        if 1 <= t_pipe <= NUM_PIPES:
            pipe_stats[t_pipe]["fn"] += 1

    for pj in unmatched_pred:
        p_pipe, _ = pred_list[pj]
        if 1 <= p_pipe <= NUM_PIPES:
            pipe_stats[p_pipe]["fp"] += 1

    # 4) Confusion: build a greedy assignment between remaining truths and remaining preds
    # by smallest abs position error (ignoring pipe equality), so confusion is meaningful.
    # If there are no remaining preds, those truths just won't add a confusion entry.
    conf_pairs = matched_pairs.copy()

    if unmatched_true and unmatched_pred:
        # build all candidate costs
        candidates = []
        for ti in unmatched_true:
            t_pipe, t_pos = true_list[ti]
            for pj in unmatched_pred:
                p_pipe, p_pos = pred_list[pj]
                cost = abs(p_pos - t_pos)
                candidates.append((cost, ti, pj))
        candidates.sort(key=lambda x: x[0])

        used_t = set(ti for ti, _ in matched_pairs)
        used_p = set(pj for _, pj in matched_pairs)

        for cost, ti, pj in candidates:
            if ti in used_t or pj in used_p:
                continue
            used_t.add(ti)
            used_p.add(pj)
            conf_pairs.append((ti, pj))

    # Write confusion counts (true_pipe -> predicted_pipe)
    for ti, pj in conf_pairs:
        t_pipe, _ = true_list[ti]
        p_pipe, _ = pred_list[pj]
        if 1 <= t_pipe <= NUM_PIPES and 1 <= p_pipe <= NUM_PIPES:
            pipe_stats[t_pipe]["conf"][p_pipe] += 1


def update_pipe_stats_pipe_pos_tol(pipe_stats, true_leaks, pred_leaks, pos_tol: float):
    """
    Per-pipe TP/FP/FN + pos_errors ONLY when (pipe matches AND pos error <= tol).
    Confusion uses the same logic as above but for tol-matched TP.
    """
    true_list = [(safe_int(l.get("pipe_id", -1)), safe_float(l.get("position", np.nan))) for l in true_leaks]
    pred_list = [(safe_int(l.get("pipe_id", -1)), safe_float(l.get("position", np.nan))) for l in pred_leaks]

    used_pred = set()
    matched_pairs = []

    # tol-match only
    for ti, (t_pipe, t_pos) in enumerate(true_list):
        best = None
        best_err = None
        for pj, (p_pipe, p_pos) in enumerate(pred_list):
            if pj in used_pred:
                continue
            if p_pipe != t_pipe:
                continue
            err = abs(p_pos - t_pos)
            if err <= pos_tol and (best_err is None or err < best_err):
                best_err = err
                best = pj
        if best is not None:
            used_pred.add(best)
            matched_pairs.append((ti, best))

            pipe_stats[t_pipe]["tp"] += 1
            pipe_stats[t_pipe]["pos_errors"].append(float(best_err))

    matched_true = {ti for ti, _ in matched_pairs}
    unmatched_true = [ti for ti in range(len(true_list)) if ti not in matched_true]
    unmatched_pred = [pj for pj in range(len(pred_list)) if pj not in used_pred]

    for ti in unmatched_true:
        t_pipe, _ = true_list[ti]
        if 1 <= t_pipe <= NUM_PIPES:
            pipe_stats[t_pipe]["fn"] += 1

    for pj in unmatched_pred:
        p_pipe, _ = pred_list[pj]
        if 1 <= p_pipe <= NUM_PIPES:
            pipe_stats[p_pipe]["fp"] += 1

    # Confusion (greedy on remaining by abs position error)
    conf_pairs = matched_pairs.copy()
    if unmatched_true and unmatched_pred:
        candidates = []
        for ti in unmatched_true:
            t_pipe, t_pos = true_list[ti]
            for pj in unmatched_pred:
                p_pipe, p_pos = pred_list[pj]
                cost = abs(p_pos - t_pos)
                candidates.append((cost, ti, pj))
        candidates.sort(key=lambda x: x[0])

        used_t = set(ti for ti, _ in matched_pairs)
        used_p = set(pj for _, pj in matched_pairs)

        for cost, ti, pj in candidates:
            if ti in used_t or pj in used_p:
                continue
            used_t.add(ti)
            used_p.add(pj)
            conf_pairs.append((ti, pj))

    for ti, pj in conf_pairs:
        t_pipe, _ = true_list[ti]
        p_pipe, _ = pred_list[pj]
        if 1 <= t_pipe <= NUM_PIPES and 1 <= p_pipe <= NUM_PIPES:
            pipe_stats[t_pipe]["conf"][p_pipe] += 1


def pipe_prf_df(pipe_stats):
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        tp = pipe_stats[pid]["tp"]
        fp = pipe_stats[pid]["fp"]
        fn = pipe_stats[pid]["fn"]
        p, r, f1 = prf(tp, fp, fn)
        rows.append({
            "pipe_id": pid,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f1,
            "support_true": tp + fn,
            "predicted_total": tp + fp
        })
    return pd.DataFrame(rows)


def pipe_pos_error_df(pipe_stats, prefix: str = ""):
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        errs = np.array(pipe_stats[pid]["pos_errors"], dtype=float)
        rows.append({
            "pipe_id": pid,
            f"{prefix}n_matched": int(errs.size),
            f"{prefix}pos_mae": float(np.mean(errs)) if errs.size > 0 else np.nan,
            f"{prefix}pos_median": float(np.median(errs)) if errs.size > 0 else np.nan,
            f"{prefix}pos_std": float(np.std(errs)) if errs.size > 0 else np.nan,
            f"{prefix}pos_max": float(np.max(errs)) if errs.size > 0 else np.nan,
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


# ============================================================
# Summaries
# ============================================================

def summarize_per_scenario(per_rows: List[Dict]) -> Dict:
    if not per_rows:
        return {}

    df = pd.DataFrame(per_rows)

    out = {}
    out["num_scenarios"] = int(len(df))

    # Count metrics
    out["exact_count_rate"] = float(df["exact_count"].mean())
    out["mean_count_error"] = float(df["count_error"].mean())

    # Pipe-only micro
    tp_p = int(df["tp_pipe"].sum())
    fp_p = int(df["fp_pipe"].sum())
    fn_p = int(df["fn_pipe"].sum())
    p_pipe, r_pipe, f1_pipe = prf(tp_p, fp_p, fn_p)
    out["pipe_only_precision_micro"] = float(p_pipe)
    out["pipe_only_recall_micro"] = float(r_pipe)
    out["pipe_only_f1_micro"] = float(f1_pipe)

    # Correct-pipe (any position) micro
    tp_cp = int(df["tp_correct_pipe"].sum())
    fp_cp = int(df["fp_correct_pipe"].sum())
    fn_cp = int(df["fn_correct_pipe"].sum())
    p_cp, r_cp, f1_cp = prf(tp_cp, fp_cp, fn_cp)
    out["correct_pipe_precision_micro"] = float(p_cp)
    out["correct_pipe_recall_micro"] = float(r_cp)
    out["correct_pipe_f1_micro"] = float(f1_cp)

    # Localization error on correct pipe (no tolerance)
    loc_vals = df["pos_mae_correct_pipe"].dropna().to_numpy()
    out["localization_mae_on_correct_pipe"] = float(np.mean(loc_vals)) if loc_vals.size > 0 else float("nan")

    # Tolerance-based pipe+pos micro
    tp_t = int(df["tp_pos_tol"].sum())
    fp_t = int(df["fp_pos_tol"].sum())
    fn_t = int(df["fn_pos_tol"].sum())
    p_t, r_t, f1_t = prf(tp_t, fp_t, fn_t)
    out["pipe_pos_tol_precision_micro"] = float(p_t)
    out["pipe_pos_tol_recall_micro"] = float(r_t)
    out["pipe_pos_tol_f1_micro"] = float(f1_t)

    # Localization error within tol (matched)
    tol_vals = df["pos_mae_within_tol"].dropna().to_numpy()
    out["localization_mae_within_tol_only"] = float(np.mean(tol_vals)) if tol_vals.size > 0 else float("nan")

    # Exact match (strict using tol TP/FP/FN)
    out["exact_match_rate_with_tol"] = float(df["exact_match_with_tol"].mean())

    # False positive rate on no-leak
    if "true_count" in df.columns and "pred_count" in df.columns:
        df0 = df[df["true_count"] == 0]
        out["false_positive_rate_no_leak"] = float((df0["pred_count"] > 0).mean()) if len(df0) > 0 else float("nan")

    return out


# ============================================================
# Main evaluation loop (ALL MODELS)
# ============================================================

def main():
    # -------- Validate paths --------
    if not TEST_RESULTS_DIR.exists():
        raise FileNotFoundError(f"TEST_RESULTS_DIR not found: {TEST_RESULTS_DIR.resolve()}")
    if not TRAINED_MODELS_DIR.exists():
        raise FileNotFoundError(f"TRAINED_MODELS_DIR not found: {TRAINED_MODELS_DIR.resolve()}")
    if not SENSOR_PLACEMENTS_CSV.exists():
        raise FileNotFoundError(f"SENSOR_PLACEMENTS_CSV not found: {SENSOR_PLACEMENTS_CSV.resolve()}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # -------- Load placements table --------
    placements = pd.read_csv(SENSOR_PLACEMENTS_CSV)

    required_cols = {"model_name", "k_budget", "placement_strategy", "configuration"}
    missing = [c for c in required_cols if c not in placements.columns]
    if missing:
        raise ValueError(f"sensor_placements.csv missing required columns: {missing}")

    # Normalize model_name to string
    placements["model_name"] = placements["model_name"].astype(str)

    # -------- Find scenarios once --------
    scn_dirs = list_scn_dirs(TEST_RESULTS_DIR)
    if not scn_dirs:
        raise RuntimeError(f"No scn_* folders found in {TEST_RESULTS_DIR.resolve()}")

    all_models_summary_rows = []

    # Optional: store per-pipe merged summary across all models
    all_models_per_pipe_rows = []

    # -------- Loop through every model --------
    for i, row in placements.iterrows():
        model_name = str(row["model_name"]).strip()
        if not model_name:
            continue

        model_folder = TRAINED_MODELS_DIR / model_name
        bundle_path = model_folder / f"multileak_tcn_bundle_{model_name}.pt"
        if not bundle_path.exists():
            print(f"[WARN] Skipping {model_name}: bundle not found at {bundle_path}")
            continue

        # output folder for this model
        model_out_root = OUTPUT_ROOT / model_name
        model_eval_dir = model_out_root / "evaluation"
        model_eval_dir.mkdir(parents=True, exist_ok=True)

        # load bundle
        bundle = torch.load(str(bundle_path), map_location=DEVICE, weights_only=False)
        mu = np.array(bundle["mu"], dtype=np.float32)
        sigma = np.array(bundle["sigma"], dtype=np.float32)
        feature_cols = list(bundle["feature_cols"])
        window = int(bundle["window"])
        stride = int(bundle.get("stride", 10))

        # build model
        model = MultiLeakTCN(C=len(feature_cols)).to(DEVICE)
        model.load_state_dict(bundle["model_state_dict"])
        model.eval()

        per_rows = []
        by_true_count = {0: [], 1: [], 2: [], 3: []}

        # Per-pipe stats:
        # 1) correct-pipe-any-pos (what you want for MAE without tolerance)
        pipe_correct_stats = init_pipe_stats()
        # 2) pipe+pos within tolerance
        pipe_tol_stats = init_pipe_stats()

        for scn_dir in scn_dirs:
            signals_path = scn_dir / "signals.csv"
            labels_path = scn_dir / "labels.json"
            if not signals_path.exists() or not labels_path.exists():
                continue

            signals = pd.read_csv(signals_path)
            labels = load_labels(labels_path)
            true_leaks = extract_leaks(labels)
            true_count = int(len(true_leaks))

            # predict
            pred = predict_from_signals_df(
                signals=signals,
                model=model,
                feature_cols=feature_cols,
                mu=mu,
                sigma=sigma,
                window=window,
                stride=stride,
                device=DEVICE,
            )
            pred_leaks = pred["predicted_leaks"]
            pred_count = int(pred["predicted_leak_count"])

            # --- Metrics: pipe-only ---
            tp_pipe, fp_pipe, fn_pipe = match_pipes_only(true_leaks, pred_leaks)
            p_pipe, r_pipe, f1_pipe = prf(tp_pipe, fp_pipe, fn_pipe)

            # --- Metrics: correct pipe (any position) + MAE ---
            tp_cp, fp_cp, fn_cp, pos_errs_cp = match_correct_pipe_any_pos(true_leaks, pred_leaks)
            p_cp, r_cp, f1_cp = prf(tp_cp, fp_cp, fn_cp)
            pos_mae_cp = float(np.mean(pos_errs_cp)) if len(pos_errs_cp) > 0 else np.nan

            # --- Metrics: pipe+pos within tolerance ---
            tp_tol, fp_tol, fn_tol, pos_errs_tol = match_pipe_and_pos_with_tol(true_leaks, pred_leaks, pos_tol=POS_TOL)
            p_tol, r_tol, f1_tol = prf(tp_tol, fp_tol, fn_tol)
            pos_mae_tol = float(np.mean(pos_errs_tol)) if len(pos_errs_tol) > 0 else np.nan

            count_error = abs(pred_count - true_count)
            exact_count = 1 if pred_count == true_count else 0
            exact_match_with_tol = 1 if (true_count == pred_count == tp_tol and fp_tol == 0 and fn_tol == 0) else 0

            per_rows.append({
                "model_name": model_name,
                "scenario": scn_dir.name,
                "scn_number": safe_int(labels.get("scn_number", scn_dir.name.split("_")[1])),
                "source_inp": str(labels.get("source_inp", "")),

                "true_count": true_count,
                "pred_count": pred_count,
                "count_error": int(count_error),
                "exact_count": int(exact_count),
                "exact_match_with_tol": int(exact_match_with_tol),

                # pipe-only
                "tp_pipe": tp_pipe, "fp_pipe": fp_pipe, "fn_pipe": fn_pipe,
                "precision_pipe": p_pipe, "recall_pipe": r_pipe, "f1_pipe": f1_pipe,

                # correct-pipe any position
                "tp_correct_pipe": tp_cp, "fp_correct_pipe": fp_cp, "fn_correct_pipe": fn_cp,
                "precision_correct_pipe": p_cp, "recall_correct_pipe": r_cp, "f1_correct_pipe": f1_cp,
                "pos_mae_correct_pipe": pos_mae_cp,

                # pipe+pos tolerance
                "tp_pos_tol": tp_tol, "fp_pos_tol": fp_tol, "fn_pos_tol": fn_tol,
                "precision_pos_tol": p_tol, "recall_pos_tol": r_tol, "f1_pos_tol": f1_tol,
                "pos_mae_within_tol": pos_mae_tol,

                "num_windows": int(pred["num_windows"]),
            })

            if true_count in by_true_count:
                by_true_count[true_count].append(per_rows[-1])

            # Per-pipe stats updates
            update_pipe_stats_correct_pipe_any_pos(pipe_correct_stats, true_leaks, pred_leaks)
            update_pipe_stats_pipe_pos_tol(pipe_tol_stats, true_leaks, pred_leaks, pos_tol=POS_TOL)

            # Optional: save per-scenario predictions
            if SAVE_PER_SCENARIO_PREDICTIONS:
                pred_out = model_eval_dir / f"{scn_dir.name}_prediction.json"
                pred_out.write_text(json.dumps({
                    "model_name": model_name,
                    "scn_number": safe_int(labels.get("scn_number", -1)),
                    "predicted_leak_count": pred_count,
                    "predicted_leaks": pred_leaks,
                }, indent=2), encoding="utf-8")

        # Save per-scenario CSV for this model
        per_df = pd.DataFrame(per_rows).sort_values("scn_number")
        per_df.to_csv(model_eval_dir / "per_scenario_metrics.csv", index=False)

        # Save overall summary JSON for this model
        overall = summarize_per_scenario(per_rows)
        overall["model_name"] = model_name
        overall["k_budget"] = safe_int(row.get("k_budget", np.nan), default=-1)
        overall["placement_strategy"] = str(row.get("placement_strategy", ""))
        overall["configuration"] = str(row.get("configuration", ""))
        overall["fitness"] = row.get("fitness", "")

        overall["pos_tol_used"] = float(POS_TOL)
        overall["window"] = int(window)
        overall["stride"] = int(stride)
        overall["bundle_path"] = str(bundle_path)

        (model_eval_dir / "overall_summary.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")

        # Save summary by true leak count
        bycount_rows = []
        for k in sorted(by_true_count.keys()):
            s = summarize_per_scenario(by_true_count[k])
            if s:
                s["true_count"] = k
                bycount_rows.append(s)
        pd.DataFrame(bycount_rows).to_csv(model_eval_dir / "summary_by_true_count.csv", index=False)

        # Save per-pipe CSV outputs
        # A) correct pipe (any position)
        pipe_correct_prf = pipe_prf_df(pipe_correct_stats)
        pipe_correct_err = pipe_pos_error_df(pipe_correct_stats, prefix="correct_pipe_")
        pipe_correct_conf = pipe_confusion_df(pipe_correct_stats)

        # B) pipe+pos within tol
        pipe_tol_prf = pipe_prf_df(pipe_tol_stats)
        pipe_tol_err = pipe_pos_error_df(pipe_tol_stats, prefix=f"tol_{POS_TOL}_")
        pipe_tol_conf = pipe_confusion_df(pipe_tol_stats)

        pipe_correct_prf.to_csv(model_eval_dir / "per_pipe_correct_pipe_any_pos.csv", index=False)
        pipe_correct_err.to_csv(model_eval_dir / "per_pipe_pos_error_correct_pipe_any_pos.csv", index=False)
        pipe_correct_conf.to_csv(model_eval_dir / "per_pipe_confusions_correct_pipe_any_pos.csv", index=False)

        pipe_tol_prf.to_csv(model_eval_dir / "per_pipe_pipe_pos_tol.csv", index=False)
        pipe_tol_err.to_csv(model_eval_dir / "per_pipe_pos_error_within_tol.csv", index=False)
        pipe_tol_conf.to_csv(model_eval_dir / "per_pipe_confusions_pipe_pos_tol.csv", index=False)

        # Collect for global comparison table
        all_models_summary_rows.append(overall)

        # Collect per-pipe merged rows (so you can later compare pipe-wise across models)
        pipe_correct_prf2 = pipe_correct_prf.copy()
        pipe_correct_prf2.insert(0, "model_name", model_name)
        pipe_correct_prf2.insert(1, "k_budget", overall["k_budget"])
        pipe_correct_prf2.insert(2, "placement_strategy", overall["placement_strategy"])
        pipe_correct_prf2.insert(3, "configuration", overall["configuration"])
        all_models_per_pipe_rows.append(pipe_correct_prf2)

        print(f"[OK] Evaluated {model_name} -> {model_eval_dir}")

    # -------- Save global comparison outputs --------
    if all_models_summary_rows:
        all_df = pd.DataFrame(all_models_summary_rows)

        # Put key columns first (nice for Excel)
        key_cols = [
            "model_name", "k_budget", "placement_strategy", "configuration", "fitness",
            "pipe_only_f1_micro", "correct_pipe_f1_micro",
            "localization_mae_on_correct_pipe",
            "pipe_pos_tol_f1_micro",
            "exact_count_rate", "exact_match_rate_with_tol",
            "false_positive_rate_no_leak",
            "pos_tol_used", "window", "stride", "bundle_path"
        ]
        cols = [c for c in key_cols if c in all_df.columns] + [c for c in all_df.columns if c not in key_cols]
        all_df = all_df[cols]

        all_df.to_csv(OUTPUT_ROOT / "all_models_summary.csv", index=False)
        print(f"[OK] Saved: {OUTPUT_ROOT / 'all_models_summary.csv'}")

    if all_models_per_pipe_rows:
        per_pipe_all = pd.concat(all_models_per_pipe_rows, ignore_index=True)
        per_pipe_all.to_csv(OUTPUT_ROOT / "all_models_per_pipe_correct_pipe_any_pos.csv", index=False)
        print(f"[OK] Saved: {OUTPUT_ROOT / 'all_models_per_pipe_correct_pipe_any_pos.csv'}")

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
