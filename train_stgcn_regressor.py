"""
train_stgcn_regressor.py

Direct ST-GCN regressor for sparse-sensor reconstruction.

Takes the 3-block ST-GCN backbone from train_stgtcn_detection_localisation5.py
(dilations 1, 2, 4 — identical STBlock components) and replaces the
classification/regression heads with a single pointwise Linear(32 -> 1) head
that produces a full (B, T, N=10, 1) reconstruction directly from the masked input.

No encoder-decoder, no LSTM, no temporal pooling.

Forward pass:
    x_masked : (B, T, N=10, F=1)  -- 7 unmonitored nodes zeroed
    block1   : (B, T, N, 16)   dilation=1
    block2   : (B, T, N, 32)   dilation=2
    block3   : (B, T, N, 32)   dilation=4
    head     : (B, T, N, 1)    Linear(32 -> 1) applied pointwise

Loss : MSE on unmonitored nodes only (normalised space).
Dataset: stgcn_dataset_v2  (2 204 train scenarios, T=24, WINDOW=12 -> 13 windows each)
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT   = "stgcn_dataset_v2/scenarios"
MANIFEST_TRAIN = "stgcn_dataset_v2/manifests/manifest_train.csv"
MANIFEST_VAL   = "stgcn_dataset_v2/manifests/manifest_val.csv"
MANIFEST_TEST  = "stgcn_dataset_v2/manifests/manifest_test.csv"
SAVE_PATH      = "stgcn_regressor.pt"

ALL_SENSORS     = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
NUM_NODES       = len(ALL_SENSORS)
NODE_FEATS      = 1

MONITORED_IDX   = [2, 5, 7]               # P4, Q1a, Q3a
UNMONITORED_IDX = [0, 1, 3, 4, 6, 8, 9]  # P2, P3, P5, P6, Q2a, Q4a, Q5a

WINDOW       = 12
STRIDE       = 1
BATCH_SIZE   = 64
EPOCHS       = 100
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.25
SEED         = 42

HIDDEN_1    = 16
HIDDEN_2    = 32
KERNEL_SIZE = 5

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = DEVICE == "cuda"

# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# HELPERS
# ============================================================
def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_no_leak(labels: dict) -> bool:
    return int(labels.get("label_detection", 0)) == 0

def folders_from_manifest(manifest_path: str):
    df = pd.read_csv(manifest_path)
    folders = []
    for scn_id in df["scenario_id"].values:
        path = os.path.join(DATASET_ROOT, f"scenario_{int(scn_id):05d}")
        if (os.path.isfile(os.path.join(path, "data.csv")) and
                os.path.isfile(os.path.join(path, "labels.json"))):
            folders.append(path)
    return folders

def read_signals_all(folder: str) -> np.ndarray:
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    missing = [c for c in ALL_SENSORS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {folder}: {missing}")
    return df[ALL_SENSORS].to_numpy(dtype=np.float32)   # (T, N)

# ============================================================
# NORMALISATION + BASELINE
# ============================================================
def compute_mu_sigma(train_folders):
    sum_x   = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    sum_x2  = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    total_T = 0
    for folder in train_folders:
        raw    = read_signals_all(folder)
        feats  = raw[:, :, None]
        sum_x  += feats.sum(axis=0)
        sum_x2 += (feats ** 2).sum(axis=0)
        total_T += feats.shape[0]
    mu    = (sum_x / total_T).astype(np.float32)
    sigma = np.sqrt(np.maximum((sum_x2 / total_T) - mu.astype(np.float64) ** 2,
                               1e-8)).astype(np.float32)
    return mu, sigma   # (N, 1), (N, 1)

def compute_baseline_template(train_folders):
    arrays = []
    for folder in train_folders:
        labels = load_labels(os.path.join(folder, "labels.json"))
        if is_no_leak(labels):
            arrays.append(read_signals_all(folder))
    if not arrays:
        raise RuntimeError("No no-leak scenarios in training split.")
    min_len = min(a.shape[0] for a in arrays)
    return np.mean(np.stack([a[:min_len] for a in arrays], axis=0),
                   axis=0).astype(np.float32)   # (T, N)

# ============================================================
# GRAPH
# ============================================================
def build_sensor_adjacency() -> np.ndarray:
    sensor_set = set(ALL_SENSORS)
    idx = {name: i for i, name in enumerate(ALL_SENSORS)}
    A   = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    def connect(a, b):
        if a in sensor_set and b in sensor_set:
            i, j = idx[a], idx[b]
            A[i, j] = A[j, i] = 1.0

    connect("Q1a", "P2");  connect("P2",  "Q2a")
    connect("Q2a", "P3");  connect("P3",  "Q3a")
    connect("Q3a", "P4");  connect("P4",  "Q4a")
    connect("Q4a", "P5");  connect("P4",  "Q5a")
    connect("Q5a", "P6")

    A += np.eye(NUM_NODES, dtype=np.float32)
    deg          = A.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
    D            = np.diag(deg_inv_sqrt)
    return (D @ A @ D).astype(np.float32)

# ============================================================
# DATASET
# ============================================================
class RegressorDataset(Dataset):
    def __init__(self, folders, mu, sigma, window=12, stride=1):
        self.samples = []
        for folder in folders:
            raw   = read_signals_all(folder)
            feats = ((raw[:, :, None] - mu[None]) / (sigma[None] + 1e-8)).astype(np.float32)
            T     = feats.shape[0]
            for w in range(max(0, (T - window) // stride + 1)):
                s = w * stride
                self.samples.append(feats[s:s + window])   # (W, N, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)

# ============================================================
# MODEL  — 3-block ST-GCN backbone + pointwise linear head
# ============================================================
class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (1, kernel_size),
                              padding=(0, pad), dilation=(1, dilation))
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, N, C)
        x = x.permute(0, 3, 2, 1)          # (B, C, N, T)
        x = self.act(self.bn(self.conv(x)))
        return x.permute(0, 3, 2, 1)        # (B, T, N, C)


class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, N, C)
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
        residual = self.res_proj(x)
        y = self.temp(x)
        y = self.graph(y)
        y = self.dropout(y)
        y = y + residual
        return self.out_act(y)


class STGCNRegressor(nn.Module):
    """
    3-block ST-GCN backbone with a pointwise linear reconstruction head.
    No temporal pooling, no LSTM, no encoder-decoder structure.

    Input : (B, T, N=10, 1)  — masked (7 nodes zeroed)
    Output: (B, T, N=10, 1)  — full network state estimate
    """
    def __init__(self, adj_matrix, node_feats=1,
                 hidden_1=16, hidden_2=32, kernel_size=5, dropout=0.25):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, dilation=1, dropout=dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, dilation=2, dropout=dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, dilation=4, dropout=dropout)
        self.head   = nn.Linear(hidden_2, node_feats)

    def forward(self, x_masked):
        # x_masked: (B, T, N, 1)
        z = self.block1(x_masked)   # (B, T, N, 16)
        z = self.block2(z)          # (B, T, N, 32)
        z = self.block3(z)          # (B, T, N, 32)
        return self.head(z)         # (B, T, N, 1)

# ============================================================
# TRAIN / EVAL
# ============================================================
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total = 0.0
    for x_full in loader:
        x_full = x_full.to(DEVICE, non_blocking=True)
        masked = x_full.clone()
        masked[:, :, UNMONITORED_IDX, :] = 0.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
            recon = model(masked)
            loss  = F.mse_loss(recon[:, :, UNMONITORED_IDX, :],
                               x_full[:, :, UNMONITORED_IDX, :])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total = 0.0
    for x_full in loader:
        x_full = x_full.to(DEVICE, non_blocking=True)
        masked = x_full.clone()
        masked[:, :, UNMONITORED_IDX, :] = 0.0
        recon  = model(masked)
        total += F.mse_loss(recon[:, :, UNMONITORED_IDX, :],
                            x_full[:, :, UNMONITORED_IDX, :]).item()
    return total / len(loader)

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_folders = folders_from_manifest(MANIFEST_TRAIN)
    val_folders   = folders_from_manifest(MANIFEST_VAL)
    test_folders  = folders_from_manifest(MANIFEST_TEST)

    if not train_folders:
        raise RuntimeError(f"No valid scenarios found via {MANIFEST_TRAIN}")
    print(f"Train: {len(train_folders)}  Val: {len(val_folders)}  Test: {len(test_folders)}")

    print("Computing normalisation statistics...")
    mu, sigma = compute_mu_sigma(train_folders)

    print("Computing baseline template from no-leak training scenarios...")
    baseline = compute_baseline_template(train_folders)
    print(f"  baseline shape={baseline.shape}")

    adj = build_sensor_adjacency()
    print(f"Adjacency matrix shape: {adj.shape}")

    print("Building datasets...")
    train_ds = RegressorDataset(train_folders, mu, sigma, window=WINDOW, stride=STRIDE)
    val_ds   = RegressorDataset(val_folders,   mu, sigma, window=WINDOW, stride=STRIDE)
    test_ds  = RegressorDataset(test_folders,  mu, sigma, window=WINDOW, stride=STRIDE)
    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val   samples: {len(val_ds)}")
    print(f"  Test  samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = STGCNRegressor(adj, node_feats=NODE_FEATS, hidden_1=HIDDEN_1,
                               hidden_2=HIDDEN_2, kernel_size=KERNEL_SIZE,
                               dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    print("\n--- Training ---")
    best_val_loss = float("inf")
    best_state    = None

    for ep in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler)
        val_loss   = evaluate_loss(model, val_loader)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker        = "  *** best ***"

        print(f"Epoch {ep+1:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest val loss: {best_val_loss:.6f}")

    bundle = {
        "model_type":        "stgcn_regressor",
        "model_state_dict":  model.state_dict(),
        "adjacency":         adj,
        "mu":                mu,
        "sigma":             sigma,
        "baseline_template": baseline,
        "sensor_names":      ALL_SENSORS,
        "monitored_idx":     MONITORED_IDX,
        "unmonitored_idx":   UNMONITORED_IDX,
        "window":            WINDOW,
        "node_feats":        NODE_FEATS,
        "hidden_1":          HIDDEN_1,
        "hidden_2":          HIDDEN_2,
        "kernel_size":       KERNEL_SIZE,
        "dropout":           DROPOUT,
    }
    torch.save(bundle, SAVE_PATH)
    print(f"[OK] Saved bundle -> {SAVE_PATH}")

    test_loss = evaluate_loss(model, test_loader)
    print(f"Test MSE (normalised, unmonitored nodes): {test_loss:.6f}")


if __name__ == "__main__":
    main()
