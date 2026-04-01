"""
train_stgcn_ekf.py
====================
Trains ST-GCN S10-A on EKF-preprocessed data (stgcn_dataset_ekf).

Combines two distribution-mismatch fixes:
  Option 1 — Train/test alignment: model is trained on EKF-reconstructed
             sensor signals, matching exactly what it sees at inference.
  Option 2 — EKF innovations as 3rd feature channel: captures the
             physics-based "surprise" signal at monitored sensors [P4,Q1a,Q3a]
             which is strongly correlated with leak location.

Feature channels per node per timestep:
  channel 0: raw sensor value (actual for monitored, EKF-reconstructed for rest)
  channel 1: deviation from no-leak baseline
  channel 2: EKF innovation (residual)  — non-zero only at P4, Q1a, Q3a;
             zero for the 7 reconstructed nodes

Input dataset : stgcn_dataset_ekf/   (produced by ekf_preprocess_stgcn_dataset.py)
Output bundle : EKFplusSTGCN/stgcn_bundle_S10-A-EKF.pt

Usage (from First_WDN/):
    python EKFplusSTGCN/train_stgcn_ekf.py
"""

from __future__ import annotations

import bisect
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ===========================================================================
# CONFIG
# ===========================================================================

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

DATASET_ROOT   = str(_ROOT / "stgcn_dataset_ekf" / "scenarios")
MANIFEST_TRAIN = str(_ROOT / "stgcn_dataset_ekf" / "manifests" / "manifest_train.csv")
MANIFEST_VAL   = str(_ROOT / "stgcn_dataset_ekf" / "manifests" / "manifest_val.csv")
MANIFEST_TEST  = str(_ROOT / "stgcn_dataset_ekf" / "manifests" / "manifest_test.csv")

BUNDLE_OUT = _HERE / "stgcn_bundle_S10-A-EKF.pt"

ALL_SENSORS = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
MONITORED   = ["P4", "Q1a", "Q3a"]       # sensors with EKF innovations
INNOVATION_COLS = ["inn_P4", "inn_Q1a", "inn_Q3a"]

# Set NODE_FEATS=2 for Option 1 only (raw+deviation, no innovations).
# Set NODE_FEATS=3 for Options 1+2 (raw+deviation+innovation).
NODE_FEATS  = 3

WINDOW      = 12
STRIDE      = 1
BATCH_SIZE  = 64
EPOCHS      = 25
LR          = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT     = 0.25
SEED        = 42

HIDDEN_1    = 16
HIDDEN_2    = 32
KERNEL_SIZE = 5

NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES  = NUM_PIPES + 1

SIZE_TO_IDX  = {"S": 0, "M": 1, "L": 2}
SIZE_NONE_IDX = 3
SIZE_CLASSES  = 4

LAMBDA_DETECT = 1.0
LAMBDA_PIPE   = 2.0
LAMBDA_SIZE   = 1.0
LAMBDA_POS    = 0.5

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = DEVICE == "cuda"


# ===========================================================================
# REPRODUCIBILITY
# ===========================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ===========================================================================
# LABEL HELPERS
# ===========================================================================

def load_labels(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_no_leak(labels: dict) -> bool:
    return int(labels.get("label_detection", 0)) == 0

def encode_labels(labels: dict):
    detect = int(labels.get("label_detection", 0))
    if detect == 1:
        pipe_t = int(labels.get("label_pipe", 1)) - 1
        pos_t  = float(labels.get("label_position", 0.0))
        sl     = str(labels.get("label_size", "S")).upper()
        size_t = SIZE_TO_IDX.get(sl, 0)
    else:
        pipe_t = PIPE_NONE_IDX
        pos_t  = 0.0
        size_t = SIZE_NONE_IDX
    return detect, pipe_t, pos_t, size_t


# ===========================================================================
# FILE HELPERS
# ===========================================================================

def folders_from_manifest(manifest_path: str) -> list[str]:
    df = pd.read_csv(manifest_path)
    folders = []
    for scn_id in df["scenario_id"].values:
        p   = os.path.join(DATASET_ROOT, f"scenario_{int(scn_id):05d}")
        sig = os.path.join(p, "data.csv")
        lab = os.path.join(p, "labels.json")
        if os.path.isfile(sig) and os.path.isfile(lab):
            folders.append(p)
    return folders


def read_signals(folder: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    raw : (T, N) float32   — 10 sensor values (actual monitored + EKF-reconstructed)
    inn : (T, N) float32   — EKF innovations broadcast to all nodes
                             (non-zero only at monitored-sensor columns)
    """
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    raw = df[ALL_SENSORS].to_numpy(dtype=np.float32)          # (T, 10)

    inn = np.zeros_like(raw)                                    # (T, 10)
    for k, (inn_col, sensor) in enumerate(
            zip(INNOVATION_COLS, MONITORED)):
        if inn_col in df.columns:
            col_idx = ALL_SENSORS.index(sensor)
            inn[:, col_idx] = df[inn_col].to_numpy(dtype=np.float32)

    return raw, inn


# ===========================================================================
# BASELINE + NORMALISATION
# ===========================================================================

def compute_baseline_template(train_folders: list[str]) -> np.ndarray:
    arrays = []
    for folder in train_folders:
        labels = load_labels(os.path.join(folder, "labels.json"))
        if is_no_leak(labels):
            raw, _ = read_signals(folder)
            arrays.append(raw)
    if not arrays:
        raise RuntimeError("No no-leak scenarios found in training split.")
    min_len = min(a.shape[0] for a in arrays)
    return np.mean(np.stack([a[:min_len] for a in arrays], axis=0),
                   axis=0).astype(np.float32)   # (T, N)


def make_node_features(raw: np.ndarray,
                        baseline: np.ndarray,
                        inn: np.ndarray,
                        node_feats: int = NODE_FEATS) -> np.ndarray:
    """
    raw      : (T, N)
    baseline : (T, N)   no-leak template
    inn      : (T, N)   EKF innovations (0 for unmonitored nodes)

    Returns (T, N, node_feats) float32
    """
    T   = raw.shape[0]
    dev = raw - baseline[:T]
    if node_feats == 2:
        return np.stack([raw, dev], axis=-1).astype(np.float32)
    # node_feats == 3
    return np.stack([raw, dev, inn], axis=-1).astype(np.float32)


def compute_mu_sigma(train_folders: list[str],
                      baseline: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Per-(node, channel) mean and std across training set."""
    N = len(ALL_SENSORS)
    sum_x  = np.zeros((N, NODE_FEATS), dtype=np.float64)
    sum_x2 = np.zeros((N, NODE_FEATS), dtype=np.float64)
    total_T = 0
    for folder in train_folders:
        raw, inn = read_signals(folder)
        feats    = make_node_features(raw, baseline, inn)
        sum_x  += feats.sum(axis=0)
        sum_x2 += (feats ** 2).sum(axis=0)
        total_T += feats.shape[0]
    mu  = sum_x / total_T
    var = np.maximum((sum_x2 / total_T) - mu ** 2, 1e-8)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


# ===========================================================================
# GRAPH
# ===========================================================================

_FULL_EDGES = [
    ("Q1a", "P2"), ("P2",  "Q2a"), ("Q2a", "P3"),
    ("P3",  "Q3a"), ("Q3a", "P4"), ("P4",  "Q4a"),
    ("Q4a", "P5"),  ("P4",  "Q5a"), ("Q5a", "P6"),
]

def build_adjacency(sensor_names: list[str]) -> np.ndarray:
    idx = {n: i for i, n in enumerate(sensor_names)}
    N   = len(sensor_names)
    A   = np.zeros((N, N), dtype=np.float32)
    for a, b in _FULL_EDGES:
        if a in idx and b in idx:
            A[idx[a], idx[b]] = 1.0
            A[idx[b], idx[a]] = 1.0
    A  += np.eye(N, dtype=np.float32)
    deg = A.sum(axis=1)
    d   = np.where(deg > 0, deg ** -0.5, 0.0)
    return (np.diag(d) @ A @ np.diag(d)).astype(np.float32)


# ===========================================================================
# DATASET
# ===========================================================================

class EKFScenarioDataset(Dataset):
    def __init__(self, folders: list[str],
                  baseline: np.ndarray,
                  mu: np.ndarray,
                  sigma: np.ndarray,
                  window: int = WINDOW,
                  stride: int = STRIDE):
        self.window = window
        self.stride = stride

        self.features      = []
        self.targets       = []
        self.names         = []
        self.window_counts = []
        self.cum_counts    = [0]

        for folder in folders:
            raw, inn = read_signals(folder)
            feats = make_node_features(raw, baseline, inn)
            feats = (feats - mu[None]) / (sigma[None] + 1e-8)
            feats = feats.astype(np.float32)

            labels = load_labels(os.path.join(folder, "labels.json"))
            tgt    = encode_labels(labels)

            T = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            if n_windows == 0:
                continue

            self.features.append(feats)
            self.targets.append(tgt)
            self.names.append(os.path.basename(folder))
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        # Scenario-level oversampling for class balance
        det_per_scen = [t[0] for t in self.targets]
        class_counts = Counter(det_per_scen)
        max_scen     = max(class_counts.values())

        nf, nt, nn_ = [], [], []
        for i, det in enumerate(det_per_scen):
            reps = max(1, round(max_scen / class_counts[det]))
            nf.extend([self.features[i]] * reps)
            nt.extend([self.targets[i]]   * reps)
            nn_.extend([self.names[i]]    * reps)

        self.features = nf
        self.targets  = nt
        self.names    = nn_

        self.window_counts = []
        self.cum_counts    = [0]
        for feats in self.features:
            T = feats.shape[0]
            n = max(0, (T - window) // stride + 1)
            self.window_counts.append(n)
            self.cum_counts.append(self.cum_counts[-1] + n)

    def __len__(self):
        return self.cum_counts[-1]

    def __getitem__(self, idx):
        scn_idx  = bisect.bisect_right(self.cum_counts, idx) - 1
        local    = idx - self.cum_counts[scn_idx]
        start    = local * self.stride
        x        = self.features[scn_idx][start: start + self.window]
        detect, pipe_t, pos_t, size_t = self.targets[scn_idx]
        return (
            torch.tensor(x,      dtype=torch.float32),
            torch.tensor(detect, dtype=torch.long),
            torch.tensor(pipe_t, dtype=torch.long),
            torch.tensor(pos_t,  dtype=torch.float32),
            torch.tensor(size_t, dtype=torch.long),
        )


# ===========================================================================
# MODEL  (identical architecture to train_stgcn_sensor_placement.py)
# ===========================================================================

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
    def __init__(self, in_ch, out_ch, adj_matrix,
                  kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp    = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph   = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout = nn.Dropout(dropout)
        self.out_act = nn.ReLU()
        self.res_proj = (nn.Linear(in_ch, out_ch)
                         if in_ch != out_ch else nn.Identity())

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
    def __init__(self, adj, num_nodes, hidden_1=16, hidden_2=32,
                  kernel_size=5, dropout=0.25, node_feats=3):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj, kernel_size, 4, dropout)

        head_in = num_nodes * hidden_2
        hh      = 64

        self.temporal_pool = TemporalAttentionPool(hidden_2, num_nodes)

        def _head(out_size):
            return nn.Sequential(
                nn.Linear(head_in, hh), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hh, out_size),
            )

        self.detect_head = _head(2)
        self.pipe_head   = _head(PIPE_CLASSES)
        self.size_head   = _head(SIZE_CLASSES)
        self.pos_head    = nn.Sequential(
            nn.Linear(head_in, hh), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hh, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        z = self.temporal_pool(x)
        return (
            self.detect_head(z),
            self.pipe_head(z),
            self.size_head(z),
            self.pos_head(z).squeeze(1),
        )


# ===========================================================================
# TRAINING + EVALUATION
# ===========================================================================

def train_and_save():
    print("=" * 70)
    print(f"ST-GCN EKF-trained model  |  sensors: {ALL_SENSORS}")
    print(f"NODE_FEATS = {NODE_FEATS}  "
          f"({'raw+dev+innovation' if NODE_FEATS == 3 else 'raw+dev'})")
    print("=" * 70)

    train_folders = folders_from_manifest(MANIFEST_TRAIN)
    val_folders   = folders_from_manifest(MANIFEST_VAL)
    test_folders  = folders_from_manifest(MANIFEST_TEST)

    if not train_folders:
        raise RuntimeError(
            f"No training scenarios found. Did you run "
            f"ekf_preprocess_stgcn_dataset.py first?\n"
            f"  Expected: {DATASET_ROOT}")

    print(f"Train: {len(train_folders)}  "
          f"Val: {len(val_folders)}  "
          f"Test: {len(test_folders)}\n")

    baseline = compute_baseline_template(train_folders)
    mu, sigma = compute_mu_sigma(train_folders, baseline)

    train_ds = EKFScenarioDataset(train_folders, baseline, mu, sigma,
                                   WINDOW, STRIDE)
    test_ds  = EKFScenarioDataset(test_folders,  baseline, mu, sigma,
                                   WINDOW, STRIDE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    num_nodes = len(ALL_SENSORS)
    adj       = build_adjacency(ALL_SENSORS)

    model = SingleLeakSTGCN(
        adj, num_nodes, HIDDEN_1, HIDDEN_2, KERNEL_SIZE, DROPOUT, NODE_FEATS
    ).to(DEVICE)

    # Class-weighted detection loss
    wcc = Counter(t[0] for i, t in enumerate(train_ds.targets)
                  for _ in range(train_ds.window_counts[i]))
    cw  = np.array([1.0 / max(wcc.get(c, 1), 1) for c in range(2)],
                    dtype=np.float32)
    cw  = torch.tensor(cw / cw.sum() * 2.0, dtype=torch.float32,
                        device=DEVICE)

    optimiser = torch.optim.Adam(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    l_detect = nn.CrossEntropyLoss(weight=cw)
    l_pipe   = nn.CrossEntropyLoss(reduction="none")
    l_size   = nn.CrossEntropyLoss(reduction="none")
    l_pos    = nn.SmoothL1Loss(reduction="none")

    for ep in range(EPOCHS):
        model.train()
        running = 0.0
        for x, detect, pipe_t, pos_t, size_t in train_loader:
            x      = x.to(DEVICE, non_blocking=True)
            detect = detect.to(DEVICE, non_blocking=True)
            pipe_t = pipe_t.to(DEVICE, non_blocking=True)
            pos_t  = pos_t.to(DEVICE,  non_blocking=True)
            size_t = size_t.to(DEVICE, non_blocking=True)

            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                dl, pl, sl, pp = model(x)
                Ld = l_detect(dl, detect)
                lm = detect.float()
                dm = lm.sum().clamp(min=1.0)
                Lp = (l_pipe(pl, pipe_t) * lm).sum() / dm
                Ls = (l_size(sl, size_t) * lm).sum() / dm
                Lr = (l_pos(pp, pos_t)   * lm).sum() / dm
                loss = (LAMBDA_DETECT * Ld + LAMBDA_PIPE * Lp
                        + LAMBDA_SIZE * Ls + LAMBDA_POS * Lr)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            running += float(loss.item())

        print(f"Epoch {ep + 1:>2}/{EPOCHS}  loss={running / len(train_loader):.4f}")

    # ── Evaluation on test split ────────────────────────────────────────────
    model.eval()
    pipe_true_all, pipe_pred_all   = [], []
    det_true_all,  det_pred_all    = [], []
    pos_true_all,  pos_pred_all    = [], []

    with torch.no_grad():
        for x, detect, pipe_t, pos_t, size_t in test_loader:
            x = x.to(DEVICE)
            dl, pl, _, pp = model(x)
            det_true_all.append(detect.numpy())
            det_pred_all.append(dl.argmax(dim=1).cpu().numpy())
            pipe_true_all.append(pipe_t.numpy())
            pipe_pred_all.append(pl.argmax(dim=1).cpu().numpy())
            pos_true_all.append(pos_t.numpy())
            pos_pred_all.append(pp.cpu().numpy())

    det_true  = np.concatenate(det_true_all)
    det_pred  = np.concatenate(det_pred_all)
    pipe_true = np.concatenate(pipe_true_all)
    pipe_pred = np.concatenate(pipe_pred_all)
    pos_true  = np.concatenate(pos_true_all)
    pos_pred  = np.concatenate(pos_pred_all)

    det_acc   = float(np.mean(det_true == det_pred))
    leak_mask = det_true > 0.5
    f1_pipe   = f1_score(pipe_true[leak_mask], pipe_pred[leak_mask],
                          average="macro",
                          labels=list(range(NUM_PIPES)))
    pos_mae   = float(np.mean(np.abs(
        pos_true[leak_mask] - pos_pred[leak_mask])))

    print(f"\n--- Test split (window-level) ---")
    print(f"  Detection accuracy : {det_acc:.4f}")
    print(f"  Pipe F1 (macro)    : {f1_pipe:.4f}")
    print(f"  Position MAE       : {pos_mae:.4f}")

    # ── Save bundle ──────────────────────────────────────────────────────────
    BUNDLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_type":        "stgcn_ekf_v1",
        "model_name":        "S10-A-EKF",
        "model_state_dict":  model.state_dict(),
        "adjacency":         adj,
        "mu":                mu,
        "sigma":             sigma,
        "baseline_template": baseline,
        "sensor_names":      ALL_SENSORS,
        "monitored":         MONITORED,
        "innovation_cols":   INNOVATION_COLS,
        "window":            WINDOW,
        "stride":            STRIDE,
        "node_feats":        NODE_FEATS,
        "hidden_1":          HIDDEN_1,
        "hidden_2":          HIDDEN_2,
        "kernel_size":       KERNEL_SIZE,
        "dropout":           DROPOUT,
        "pipe_classes":      PIPE_CLASSES,
        "size_classes":      SIZE_CLASSES,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "lr":                LR,
        "weight_decay":      WEIGHT_DECAY,
        "lambda_detect":     LAMBDA_DETECT,
        "lambda_pipe":       LAMBDA_PIPE,
        "seed":              SEED,
        "dataset_root":      DATASET_ROOT,
        "test_det_accuracy": round(det_acc, 4),
        "test_pipe_f1":      round(f1_pipe, 4),
        "test_pos_mae":      round(pos_mae, 4),
    }
    torch.save(bundle, str(BUNDLE_OUT))
    print(f"\n[OK] Bundle saved -> {BUNDLE_OUT}")


if __name__ == "__main__":
    print(f"CUDA available : {torch.cuda.is_available()}")
    print(f"Device         : {DEVICE}\n")
    train_and_save()
