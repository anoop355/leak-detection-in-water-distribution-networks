"""
evaluate_stgcn_sensor_placement.py
------------------------------------
Evaluates all ST-GCN sensor-placement bundles against the external test dataset.
Reads bundle paths and sensor configurations from sensor_placements.csv, runs
inference per scenario, and writes per-model metrics plus a global comparison table.

Output structure:
    stgcn_placement_results/
        {model_name}/evaluation/
            {model_name}_per_scenario_metrics.csv
            {model_name}_overall_summary.json
            {model_name}_detection_confusion_matrix.csv
            {model_name}_per_pipe_pipe_only.csv
            {model_name}_pipe_confusion_matrix_normalised.csv
            {model_name}_pipe_identification_summary.json
            {model_name}_position_regression_summary.json
            {model_name}_position_regression_per_pipe.csv
            {model_name}_position_scatter_data.csv
        all_models_summary.csv
        all_models_per_pipe.csv
"""

import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn



# SETTINGS

BUNDLES_DIR           = Path("stgcn_placement_bundles")
SENSOR_PLACEMENTS_CSV = Path("sensor_placements.csv")
TEST_DIR              = Path("test_dataset")
OUTPUT_ROOT           = Path("stgcn_placement_results")
POS_TOL               = 0.25
DEVICE                = "cuda" if torch.cuda.is_available() else "cpu"


NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES  = NUM_PIPES + 1
SIZE_CLASSES  = 4



# MODEL DEFINITIONS

class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (1, kernel_size),
                              padding=(0, pad), dilation=(1, dilation))
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)   # (B, C, N, T)
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
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5,
                 dilation=1, dropout=0.25):
        super().__init__()
        self.temp     = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph    = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout  = nn.Dropout(dropout)
        self.out_act  = nn.ReLU()
        self.res_proj = (nn.Linear(in_ch, out_ch) if in_ch != out_ch
                         else nn.Identity())

    def forward(self, x):
        r = self.res_proj(x)
        y = self.dropout(self.graph(self.temp(x)))
        return self.out_act(y + r)


class SingleLeakSTGCN(nn.Module):
    """Average-pool backbone — matches stgcn_single_leak_v3."""
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, 4, dropout)
        head_in, head_hidden = num_nodes * hidden_2, 64

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
        x = self.block3(self.block2(self.block1(x)))
        z = x.mean(dim=1).reshape(x.size(0), -1)
        return (self.detect_head(z), self.pipe_head(z),
                self.size_head(z), self.pos_head(z).squeeze(1))


class TemporalAttentionPool(nn.Module):
    """Soft attention pooling over the time axis."""
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * num_nodes, 1)

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat  = x.reshape(B, T, N * C)
        weights = torch.softmax(self.attn(x_flat), dim=1)
        return (x_flat * weights).sum(dim=1)


class SingleLeakSTGCNv4(nn.Module):
    """Temporal-attention-pool backbone — matches stgcn_single_leak_v4."""
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, 4, dropout)
        head_in, head_hidden = num_nodes * hidden_2, 64

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
        x = self.block3(self.block2(self.block1(x)))
        z = self.temporal_pool(x)
        return (self.detect_head(z), self.pipe_head(z),
                self.size_head(z), self.pos_head(z).squeeze(1))


def load_bundle(bundle_path: Path):
    bundle     = torch.load(str(bundle_path), map_location=DEVICE, weights_only=False)
    model_type = bundle.get("model_type", "stgcn_single_leak_v3")
    adj        = np.array(bundle["adjacency"],        dtype=np.float32)
    num_nodes  = len(bundle["sensor_names"])
    hidden_1   = int(bundle.get("hidden_1",    16))
    hidden_2   = int(bundle.get("hidden_2",    32))
    kernel_sz  = int(bundle.get("kernel_size",  5))
    dropout    = float(bundle.get("dropout",  0.25))
    node_feats = int(bundle.get("node_feats",   2))

    if model_type == "stgcn_single_leak_v4":
        model = SingleLeakSTGCNv4(adj, num_nodes, hidden_1, hidden_2,
                                  kernel_sz, dropout, node_feats).to(DEVICE)
    else:
        model = SingleLeakSTGCN(adj, num_nodes, hidden_1, hidden_2,
                                kernel_sz, dropout, node_feats).to(DEVICE)

    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle, model_type


def preprocess_stgcn(raw: np.ndarray, baseline: np.ndarray,
                     mu: np.ndarray, sigma: np.ndarray,
                     node_feats: int = 2) -> torch.Tensor:
    T        = raw.shape[0]
    base     = baseline[:T]
    channels = [raw, raw - base]
    if node_feats == 3:
        # temporal difference channel
        diff = np.zeros_like(raw)
        diff[1:] = raw[1:] - raw[:-1]
        channels.append(diff)
    feats = np.stack(channels, axis=-1).astype(np.float32)
    feats = (feats - mu[None]) / (sigma[None] + 1e-8)
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


def predict_scenario(data_csv: Path, model: nn.Module,
                     bundle: dict, node_feats: int) -> Dict:
    sensor_names = bundle["sensor_names"]
    mu           = np.array(bundle["mu"],                dtype=np.float32)
    sigma        = np.array(bundle["sigma"],             dtype=np.float32)
    baseline     = np.array(bundle["baseline_template"], dtype=np.float32)

    raw = pd.read_csv(data_csv)[sensor_names].to_numpy(dtype=np.float32)
    x_t = preprocess_stgcn(raw, baseline, mu, sigma, node_feats).to(DEVICE)

    with torch.no_grad():
        detect_logits, pipe_logits, _, pos_pred = model(x_t)

    pred_detect  = int(detect_logits.argmax(dim=1).item())
    pred_pipe    = int(pipe_logits.argmax(dim=1).item())
    pred_pos     = float(pos_pred.item())
    pred_pipe_id = (pred_pipe + 1) if pred_pipe < NUM_PIPES else None

    return {
        "pred_detect":  pred_detect,
        "pred_pipe_id": pred_pipe_id,
        "pred_pos":     max(0.0, min(1.0, pred_pos)),
    }


# METRICS

def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_detection_metrics(true_det: List[int], pred_det: List[int]) -> Dict:
    t = np.array(true_det, dtype=int)
    p = np.array(pred_det, dtype=int)
    acc = float(np.mean(t == p))
    tp  = int(np.sum((t == 1) & (p == 1)))
    fp  = int(np.sum((t == 0) & (p == 1)))
    fn  = int(np.sum((t == 1) & (p == 0)))
    tn  = int(np.sum((t == 0) & (p == 0)))
    pr, rc, f1 = prf(tp, fp, fn)
    return {"accuracy": round(acc, 4), "precision": round(pr, 4),
            "recall": round(rc, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_scenarios": len(t)}


def compute_pipe_metrics(pipe_stats: Dict) -> Dict:
    f1_scores, total_tp, total_fp, total_fn = [], 0, 0, 0
    for pid in range(1, NUM_PIPES + 1):
        tp, fp, fn = (pipe_stats[pid]["tp"], pipe_stats[pid]["fp"],
                      pipe_stats[pid]["fn"])
        _, _, f1 = prf(tp, fp, fn)
        f1_scores.append(f1)
        total_tp += tp; total_fp += fp; total_fn += fn
    _, _, micro_f1 = prf(total_tp, total_fp, total_fn)
    return {"pipe_macro_f1": round(float(np.mean(f1_scores)), 4),
            "pipe_micro_f1": round(micro_f1, 4)}


def compute_regression_metrics(pos_pairs: List[Tuple]) -> Dict:
    if not pos_pairs:
        return {"mae": float("nan"), "rmse": float("nan"),
                "r2": float("nan"), "n_matched": 0}
    tv = np.array([p[0] for p in pos_pairs], dtype=float)
    pv = np.array([p[1] for p in pos_pairs], dtype=float)
    err    = pv - tv
    mae    = float(np.mean(np.abs(err)))
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((tv - tv.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": round(mae, 6), "rmse": round(rmse, 6),
            "r2": round(r2, 6), "n_matched": len(pos_pairs)}


def compute_per_pipe_regression(pos_pairs: List[Tuple]) -> pd.DataFrame:
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        pp = [(t, p) for t, p, pipe in pos_pairs if pipe == pid]
        if not pp:
            rows.append({"pipe_id": pid, "n_matched": 0, "mae": float("nan"),
                         "rmse": float("nan"), "r2": float("nan")})
            continue
        tv  = np.array([x[0] for x in pp])
        pv  = np.array([x[1] for x in pp])
        err = pv - tv
        mae    = float(np.mean(np.abs(err)))
        rmse   = float(np.sqrt(np.mean(err ** 2)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((tv - tv.mean()) ** 2))
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append({"pipe_id": pid, "n_matched": len(pp),
                     "mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6)})
    return pd.DataFrame(rows)


def pipe_prf_df(pipe_stats: Dict) -> pd.DataFrame:
    rows = []
    for pid in range(1, NUM_PIPES + 1):
        tp, fp, fn = (pipe_stats[pid]["tp"], pipe_stats[pid]["fp"],
                      pipe_stats[pid]["fn"])
        pr, rc, f1 = prf(tp, fp, fn)
        rows.append({"pipe_id": pid, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(pr, 4), "recall": round(rc, 4),
                     "f1": round(f1, 4),
                     "support_true": tp + fn, "predicted_total": tp + fp})
    return pd.DataFrame(rows)


def pipe_confusion_normalised(pipe_stats: Dict) -> pd.DataFrame:
    """Row-normalised 5x5 confusion matrix for pipe identification."""
    raw = np.zeros((NUM_PIPES, NUM_PIPES), dtype=float)
    for i, tp in enumerate(range(1, NUM_PIPES + 1)):
        for j, pp in enumerate(range(1, NUM_PIPES + 1)):
            raw[i, j] = pipe_stats[tp]["conf"].get(pp, 0)
    row_sums = raw.sum(axis=1, keepdims=True)
    norm = np.where(row_sums > 0, raw / row_sums, 0.0)
    return pd.DataFrame(np.round(norm, 4),
                        index=[f"true_pipe_{p}" for p in range(1, NUM_PIPES + 1)],
                        columns=[f"pred_pipe_{p}" for p in range(1, NUM_PIPES + 1)])


# EVALUATE ONE MODEL

def evaluate_one_model(model_name: str, bundle_path: Path,
                       scn_dirs: List[Path], model_eval_dir: Path) -> Dict:

    model, bundle, model_type = load_bundle(bundle_path)
    node_feats   = int(bundle.get("node_feats", 2))
    sensor_names = bundle["sensor_names"]

    per_rows     = []
    true_det_all = []
    pred_det_all = []
    all_pos_pairs: List[Tuple] = []

    pipe_only_stats = {pid: {"tp": 0, "fp": 0, "fn": 0,
                             "conf": {p: 0 for p in range(1, NUM_PIPES + 1)}}
                       for pid in range(1, NUM_PIPES + 1)}
    pipe_pos_stats  = {pid: {"tp": 0, "fp": 0, "fn": 0, "pos_errors": [],
                             "conf": {p: 0 for p in range(1, NUM_PIPES + 1)}}
                       for pid in range(1, NUM_PIPES + 1)}

    exact_pipe_match_count = 0
    n_leak_scenarios       = 0

    for scn_dir in scn_dirs:
        data_path  = scn_dir / "data.csv"
        label_path = scn_dir / "labels.json"
        if not data_path.exists() or not label_path.exists():
            continue

        labels       = json.loads(label_path.read_text(encoding="utf-8"))
        scn_id       = int(labels["scenario_id"])
        true_detect  = int(labels["label_detection"])
        true_pipe_id = int(labels["label_pipe"])
        true_pos     = float(labels["label_position"])
        true_size    = str(labels["label_size"])

        pred         = predict_scenario(data_path, model, bundle, node_feats)
        pred_detect  = pred["pred_detect"]
        pred_pipe_id = pred["pred_pipe_id"]
        pred_pos     = pred["pred_pos"]

        true_det_all.append(true_detect)
        pred_det_all.append(pred_detect)

        tp_pipe = fp_pipe = fn_pipe = 0
        tp_pos  = fp_pos  = fn_pos  = 0
        pos_err      = float("nan")
        pipe_correct = False

        if true_detect == 1:
            n_leak_scenarios += 1

            if pred_pipe_id == true_pipe_id:
                tp_pipe      = 1
                pipe_correct = True
                pipe_only_stats[true_pipe_id]["tp"] += 1
            else:
                fn_pipe = 1
                pipe_only_stats[true_pipe_id]["fn"] += 1
                if pred_pipe_id is not None:
                    fp_pipe = 1
                    pipe_only_stats[pred_pipe_id]["fp"] += 1

            if pred_pipe_id is not None and 1 <= pred_pipe_id <= NUM_PIPES:
                pipe_only_stats[true_pipe_id]["conf"][pred_pipe_id] += 1

            if pipe_correct:
                pos_err = abs(pred_pos - true_pos)
                all_pos_pairs.append((true_pos, pred_pos, true_pipe_id))
                exact_pipe_match_count += 1

            # tolerance-based pipe+pos match
            if pipe_correct and pos_err <= POS_TOL:
                tp_pos = 1
                pipe_pos_stats[true_pipe_id]["tp"] += 1
                pipe_pos_stats[true_pipe_id]["pos_errors"].append(pos_err)
            else:
                fn_pos = 1
                pipe_pos_stats[true_pipe_id]["fn"] += 1
                if pred_pipe_id is not None and 1 <= pred_pipe_id <= NUM_PIPES:
                    fp_pos = 1
                    pipe_pos_stats[pred_pipe_id]["fp"] += 1

        elif pred_detect == 1 and pred_pipe_id is not None:
            # false positive — count against predicted pipe
            if 1 <= pred_pipe_id <= NUM_PIPES:
                pipe_only_stats[pred_pipe_id]["fp"] += 1
                pipe_pos_stats[pred_pipe_id]["fp"]  += 1

        per_rows.append({
            "scenario_id":    scn_id,
            "scenario":       scn_dir.name,
            "source_inp":     labels.get("source_inp", ""),
            "true_detect":    true_detect,
            "pred_detect":    pred_detect,
            "detect_correct": int(true_detect == pred_detect),
            "true_pipe":      true_pipe_id,
            "pred_pipe":      pred_pipe_id if pred_pipe_id else -1,
            "pipe_correct":   int(pipe_correct),
            "true_pos":       true_pos,
            "pred_pos":       round(pred_pos, 4),
            "pos_error":      round(pos_err, 4) if not np.isnan(pos_err) else float("nan"),
            "true_size":      true_size,
            "tp_pipe": tp_pipe, "fp_pipe": fp_pipe, "fn_pipe": fn_pipe,
            "tp_pos":  tp_pos,  "fp_pos":  fp_pos,  "fn_pos":  fn_pos,
        })

    per_df = pd.DataFrame(per_rows).sort_values("scenario_id")

    det_metrics  = compute_detection_metrics(true_det_all, pred_det_all)
    no_leak_rows = per_df[per_df["true_detect"] == 0]
    fpr = float((no_leak_rows["pred_detect"] > 0).mean()) if len(no_leak_rows) > 0 else float("nan")

    pipe_summary   = compute_pipe_metrics(pipe_only_stats)
    pipe_exact_acc = (exact_pipe_match_count / n_leak_scenarios
                      if n_leak_scenarios > 0 else float("nan"))
    pipe_id_summary = {
        **pipe_summary,
        "pipe_exact_match_accuracy": round(float(pipe_exact_acc), 4),
        "pipe_exact_match_count":    exact_pipe_match_count,
        "n_leak_scenarios":          n_leak_scenarios,
    }

    regression_overall  = compute_regression_metrics(all_pos_pairs)
    regression_per_pipe = compute_per_pipe_regression(all_pos_pairs)

    # tolerance-based pipe+pos (micro)
    tp_d = sum(r["tp_pos"] for r in per_rows)
    fp_d = sum(r["fp_pos"] for r in per_rows)
    fn_d = sum(r["fn_pos"] for r in per_rows)
    p_micro, r_micro, f1_pos_micro = prf(tp_d, fp_d, fn_d)

    # tolerance-based pipe+pos (macro over pipes)
    f1_per_pipe_pos = []
    for pid in range(1, NUM_PIPES + 1):
        _, _, f1_p = prf(pipe_pos_stats[pid]["tp"],
                         pipe_pos_stats[pid]["fp"],
                         pipe_pos_stats[pid]["fn"])
        f1_per_pipe_pos.append(f1_p)
    f1_pos_macro = float(np.mean(f1_per_pipe_pos))

    # save per-model outputs
    def mf(name):
        return model_eval_dir / f"{model_name}_{name}"

    per_df.to_csv(mf("per_scenario_metrics.csv"), index=False)

    det_conf = pd.DataFrame(
        [[det_metrics["tn"], det_metrics["fp"]],
         [det_metrics["fn"], det_metrics["tp"]]],
        index=["true_0", "true_1"], columns=["pred_0", "pred_1"]
    )
    det_conf.to_csv(mf("detection_confusion_matrix.csv"))

    pipe_prf_df(pipe_only_stats).to_csv(mf("per_pipe_pipe_only.csv"), index=False)
    pipe_confusion_normalised(pipe_only_stats).to_csv(
        mf("pipe_confusion_matrix_normalised.csv"))
    mf("pipe_identification_summary.json").write_text(
        json.dumps(pipe_id_summary, indent=2), encoding="utf-8")
    mf("position_regression_summary.json").write_text(
        json.dumps(regression_overall, indent=2), encoding="utf-8")
    regression_per_pipe.to_csv(mf("position_regression_per_pipe.csv"), index=False)

    if all_pos_pairs:
        scatter_df = pd.DataFrame(all_pos_pairs,
                                  columns=["true_position", "pred_position", "pipe_id"])
        scatter_df["pipe_id"] = scatter_df["pipe_id"].astype(int)
    else:
        scatter_df = pd.DataFrame(columns=["true_position", "pred_position", "pipe_id"])
    scatter_df.to_csv(mf("position_scatter_data.csv"), index=False)

    overall_summary = {
        "model_name":   model_name,
        "model_type":   model_type,
        "sensor_names": sensor_names,
        "n_sensors":    len(sensor_names),
        "n_scenarios":  len(per_rows),
        "pos_tol":      POS_TOL,
        "detect_accuracy":             det_metrics["accuracy"],
        "detect_precision":            det_metrics["precision"],
        "detect_recall":               det_metrics["recall"],
        "detect_f1":                   det_metrics["f1"],
        "detect_tp": det_metrics["tp"], "detect_fp": det_metrics["fp"],
        "detect_fn": det_metrics["fn"], "detect_tn": det_metrics["tn"],
        "false_positive_rate_no_leak": round(fpr, 4),
        **pipe_id_summary,
        "pos_mae":       regression_overall["mae"],
        "pos_rmse":      regression_overall["rmse"],
        "pos_r2":        regression_overall["r2"],
        "pos_n_matched": regression_overall["n_matched"],
        "pipe_pos_tol_f1_micro":        round(f1_pos_micro, 4),
        "pipe_pos_tol_f1_macro":        round(f1_pos_macro, 4),
        "pipe_pos_tol_precision_micro": round(p_micro, 4),
        "pipe_pos_tol_recall_micro":    round(r_micro, 4),
    }
    mf("overall_summary.json").write_text(
        json.dumps(overall_summary, indent=2), encoding="utf-8")

    return overall_summary, pipe_prf_df(pipe_only_stats)


# MAIN

def main():
    print(f"Device : {DEVICE}")
    print(f"Bundles: {BUNDLES_DIR.resolve()}")
    print(f"Tests  : {TEST_DIR.resolve()}")
    print()

    for p in [BUNDLES_DIR, SENSOR_PLACEMENTS_CSV, TEST_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Not found: {p.resolve()}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    placements = pd.read_csv(SENSOR_PLACEMENTS_CSV)
    placements["model_name"]    = placements["model_name"].astype(str).str.strip()
    placements["configuration"] = placements["configuration"].astype(str).str.strip()

    scenarios_dir = TEST_DIR / "scenarios"
    scn_dirs = sorted(
        [p for p in scenarios_dir.iterdir()
         if p.is_dir() and p.name.startswith("scenario_")],
        key=lambda p: int(p.name.split("_")[1])
    )
    if not scn_dirs:
        raise RuntimeError(f"No scenario_* folders found in {scenarios_dir}")
    print(f"Test scenarios: {len(scn_dirs)}\n")

    all_summary_rows  = []
    all_per_pipe_rows = []

    for _, row in placements.iterrows():
        model_name = str(row["model_name"]).strip()
        if not model_name:
            continue

        bundle_path = BUNDLES_DIR / f"stgcn_bundle_{model_name}.pt"
        if not bundle_path.exists():
            print(f"[WARN] Skipping {model_name}: bundle not found at {bundle_path}")
            continue

        model_eval_dir = OUTPUT_ROOT / model_name / "evaluation"
        model_eval_dir.mkdir(parents=True, exist_ok=True)

        print(f"Evaluating {model_name} ...")
        overall_summary, pipe_df = evaluate_one_model(
            model_name, bundle_path, scn_dirs, model_eval_dir
        )

        # attach placement metadata from the CSV
        overall_summary["k_budget"]           = int(row.get("k_budget", -1))
        overall_summary["placement_strategy"] = str(row.get("placement_strategy", ""))
        overall_summary["configuration"]      = str(row.get("configuration", ""))
        overall_summary["fitness"]            = str(row.get("fitness", ""))

        print(f"  Detection F1 : {overall_summary['detect_f1']:.4f}  "
              f"(acc={overall_summary['detect_accuracy']:.4f}, "
              f"FPR={overall_summary['false_positive_rate_no_leak']:.4f})")
        print(f"  Pipe Macro F1: {overall_summary['pipe_macro_f1']:.4f}  "
              f"exact={overall_summary['pipe_exact_match_accuracy']:.4f}")
        print(f"  Pos MAE      : {overall_summary['pos_mae']}  "
              f"R2={overall_summary['pos_r2']}")
        print(f"  Pipe+Pos F1  : micro={overall_summary['pipe_pos_tol_f1_micro']:.4f}  "
              f"macro={overall_summary['pipe_pos_tol_f1_macro']:.4f}")
        print(f"  [OK] -> {model_eval_dir}")
        print()

        for col, val in [("model_name",          model_name),
                         ("k_budget",             overall_summary["k_budget"]),
                         ("placement_strategy",   overall_summary["placement_strategy"]),
                         ("configuration",        overall_summary["configuration"])]:
            pipe_df.insert(0, col, val)
        all_per_pipe_rows.append(pipe_df)
        all_summary_rows.append(overall_summary)

    # global comparison table
    if all_summary_rows:
        all_df = pd.DataFrame(all_summary_rows)
        key_cols = [
            "model_name", "k_budget", "placement_strategy", "configuration", "fitness",
            "n_sensors",
            "detect_accuracy", "detect_f1", "false_positive_rate_no_leak",
            "pipe_macro_f1", "pipe_micro_f1", "pipe_exact_match_accuracy",
            "pos_mae", "pos_rmse", "pos_r2", "pos_n_matched",
            "pipe_pos_tol_f1_micro", "pipe_pos_tol_f1_macro",
            "pos_tol",
        ]
        cols = ([c for c in key_cols if c in all_df.columns] +
                [c for c in all_df.columns if c not in key_cols])
        summary_path = OUTPUT_ROOT / "all_models_summary.csv"
        all_df[cols].to_csv(summary_path, index=False)
        print(f"[OK] {summary_path}")

    if all_per_pipe_rows:
        per_pipe_path = OUTPUT_ROOT / "all_models_per_pipe.csv"
        pd.concat(all_per_pipe_rows, ignore_index=True).to_csv(per_pipe_path, index=False)
        print(f"[OK] {per_pipe_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / "FAILED_evaluate_stgcn_sensor_placement_log.txt").write_text(
            tb, encoding="utf-8")
        raise