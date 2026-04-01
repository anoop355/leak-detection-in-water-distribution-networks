"""
train_stgcn_autoencoder.py

ST-GCN Autoencoder — self-supervised reconstruction of the full 10-sensor
water-distribution-network state from only the 3 active sparse sensors.

Active (monitored) sensors : P4, Q1a, Q3a  =>  indices [2, 5, 7]
Unmonitored sensors        : P2, P3, P5, P6, Q2a, Q4a, Q5a  =>  indices [0,1,3,4,6,8,9]

Architecture
  Encoder  : 2 × ST-Block  (in=1 → 16 → 32)
  Decoder  : per-node LSTM  (hidden=32, layers=2)
  Head     : Linear(32 → 1) per node per timestep

Loss : MSE on unmonitored nodes only (normalised space)
"""

import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT   = "stgcn_dataset/scenarios"
MANIFEST_TRAIN = "stgcn_dataset/manifests/manifest_train.csv"
MANIFEST_VAL   = "stgcn_dataset/manifests/manifest_val.csv"
MANIFEST_TEST  = "stgcn_dataset/manifests/manifest_test.csv"
SAVE_PATH      = "stgcn_autoencoder.pt"

ALL_SENSORS     = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
NUM_NODES       = len(ALL_SENSORS)       # 10
NODE_FEATS      = 1                      # raw sensor reading only

MONITORED_IDX   = [2, 5, 7]             # P4, Q1a, Q3a
UNMONITORED_IDX = [0, 1, 3, 4, 6, 8, 9] # P2, P3, P5, P6, Q2a, Q4a, Q5a

WINDOW      = 12
STRIDE      = 1
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT     = 0.25
SEED        = 42

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
# LABEL / FILE HELPERS  (unchanged from detection script)
# ============================================================
def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_no_leak(labels: dict) -> bool:
    return int(labels.get("label_detection", 0)) == 0

def folders_from_manifest(manifest_path: str):
    """Return list of scenario folder paths listed in a manifest CSV."""
    df = pd.read_csv(manifest_path)
    folders = []
    for scn_id in df["scenario_id"].values:
        path = os.path.join(DATASET_ROOT, f"scenario_{int(scn_id):05d}")
        sig  = os.path.join(path, "data.csv")
        lab  = os.path.join(path, "labels.json")
        if os.path.isfile(sig) and os.path.isfile(lab):
            folders.append(path)
    return folders

def read_signals_all(folder: str) -> np.ndarray:
    """Read all 10 sensor columns from a scenario's data.csv.
    Returns: (T, 10) float32 array of raw sensor readings.
    """
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    missing = [c for c in ALL_SENSORS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {folder}: {missing}")
    return df[ALL_SENSORS].to_numpy(dtype=np.float32)  # (T, N)

# ============================================================
# GRAPH  —  full 10-node topology
# ============================================================
def build_sensor_adjacency() -> np.ndarray:
    """
    Build the symmetrically-normalised adjacency matrix for all 10 sensors.

    Physical topology:
        Q1a -- P2 -- Q2a -- P3 -- Q3a -- P4
                                        /   \\
                                     Q4a     Q5a
                                      |       |
                                     P5      P6
    Self-loops added; symmetric D^{-1/2} A D^{-1/2} normalisation applied.
    """
    sensor_set = set(ALL_SENSORS)
    idx = {name: i for i, name in enumerate(ALL_SENSORS)}
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    def connect(a, b):
        if a in sensor_set and b in sensor_set:
            i, j = idx[a], idx[b]
            A[i, j] = 1.0
            A[j, i] = 1.0

    connect("Q1a", "P2")
    connect("P2",  "Q2a")
    connect("Q2a", "P3")
    connect("P3",  "Q3a")
    connect("Q3a", "P4")
    connect("P4",  "Q4a")
    connect("Q4a", "P5")
    connect("P4",  "Q5a")
    connect("Q5a", "P6")

    A = A + np.eye(NUM_NODES, dtype=np.float32)           # self-loops

    deg = A.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-10)), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return A_hat.astype(np.float32)

# ============================================================
# NORMALISATION  —  raw sensor readings, 1 feature per node
# ============================================================
def compute_mu_sigma(train_folders):
    """
    Per-channel (per-node) mean and std over training data only.
    Returns: mu (N, 1), sigma (N, 1) — both float32.
    """
    sum_x  = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    sum_x2 = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    total_T = 0

    for folder in train_folders:
        raw   = read_signals_all(folder)          # (T, N)
        feats = raw[:, :, None]                   # (T, N, 1)
        sum_x  += feats.sum(axis=0)
        sum_x2 += (feats ** 2).sum(axis=0)
        total_T += feats.shape[0]

    mu  = (sum_x / total_T).astype(np.float32)
    var = (sum_x2 / total_T) - (mu.astype(np.float64) ** 2)
    var = np.maximum(var, 1e-8)
    sigma = np.sqrt(var).astype(np.float32)

    return mu, sigma   # (N, 1), (N, 1)

def compute_baseline_template(train_folders):
    """
    Average of all no-leak training scenarios in raw (un-normalised) space.
    Returns: (T, N) float32 — saved in bundle for the downstream detector.
    """
    no_leak_arrays = []
    for folder in train_folders:
        labels = load_labels(os.path.join(folder, "labels.json"))
        if is_no_leak(labels):
            arr = read_signals_all(folder)         # (T, N)
            no_leak_arrays.append(arr)

    if len(no_leak_arrays) == 0:
        raise RuntimeError("No no-leak scenarios found in training split.")

    min_len = min(a.shape[0] for a in no_leak_arrays)
    cropped = [a[:min_len] for a in no_leak_arrays]
    baseline = np.mean(np.stack(cropped, axis=0), axis=0).astype(np.float32)
    return baseline   # (T, N)

# ============================================================
# DATASET
# ============================================================
class AutoencoderDataset(Dataset):
    """
    Loads normalised (T, N, 1) tensors from scenarios.
    Each scenario with T=12 and WINDOW=12 yields exactly 1 sample.
    """
    def __init__(self, folders, mu, sigma, window=12, stride=1):
        self.samples = []   # list of (W, N, 1) float32 arrays

        for folder in folders:
            raw   = read_signals_all(folder)           # (T, N)
            feats = raw[:, :, None]                    # (T, N, 1)
            feats = (feats - mu[None]) / (sigma[None] + 1e-8)
            feats = feats.astype(np.float32)

            T = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            for w in range(n_windows):
                s = w * stride
                e = s + window
                self.samples.append(feats[s:e])        # (W, N, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)

# ============================================================
# MODEL COMPONENTS  (verbatim from train_stgtcn_detection_localisation5.py)
# ============================================================
class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            dilation=(1, dilation)
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, N, C)
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
        # x: (B, T, N, C)
        x = torch.einsum("ij,btjc->btic", self.A, x)
        x = self.lin(x)
        x = self.ln(x)
        x = self.act(x)
        return x


class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp    = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph   = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout = nn.Dropout(dropout)
        self.out_act = nn.ReLU()

        if in_ch != out_ch:
            self.res_proj = nn.Linear(in_ch, out_ch)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x):
        # x: (B, T, N, C_in)
        residual = self.res_proj(x)
        y = self.temp(x)
        y = self.graph(y)
        y = self.dropout(y)
        y = y + residual
        y = self.out_act(y)
        return y

# ============================================================
# AUTOENCODER
# ============================================================
class STGCNAutoencoder(nn.Module):
    """
    Encoder:  2 × STBlock  (1 → 16 → 32)
    Decoder:  per-node LSTM(input=32, hidden=32, layers=2)
    Head:     Linear(32 → 1)  applied pointwise
    """
    def __init__(self, adj_matrix, node_feats=1,
                 hidden_1=16, hidden_2=32, kernel_size=5, dropout=0.25):
        super().__init__()

        self.encoder_block1 = STBlock(
            in_ch=node_feats, out_ch=hidden_1,
            adj_matrix=adj_matrix, kernel_size=kernel_size, dilation=1, dropout=dropout
        )
        self.encoder_block2 = STBlock(
            in_ch=hidden_1, out_ch=hidden_2,
            adj_matrix=adj_matrix, kernel_size=kernel_size, dilation=2, dropout=dropout
        )

        # Per-node temporal decoder — operates on (B*N, T, H)
        self.lstm = nn.LSTM(
            input_size=hidden_2,
            hidden_size=hidden_2,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.recon_head = nn.Linear(hidden_2, node_feats)

    def forward(self, x_masked):
        # x_masked: (B, T, N, F) — unmonitored nodes zeroed
        z = self.encoder_block1(x_masked)       # (B, T, N, hidden_1)
        z = self.encoder_block2(z)              # (B, T, N, hidden_2)

        B, T, N, H = z.shape
        z_flat  = z.reshape(B * N, T, H)        # (B*N, T, H)
        z_dec, _ = self.lstm(z_flat)            # (B*N, T, H)
        z_dec   = z_dec.reshape(B, T, N, H)    # (B, T, N, H)

        recon = self.recon_head(z_dec)          # (B, T, N, F)
        return recon

# ============================================================
# TRAINING LOOP
# ============================================================
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    total_loss = 0.0

    for x_full in loader:
        x_full = x_full.to(DEVICE, non_blocking=True)  # (B, T, N, 1)

        masked_input = x_full.clone()
        masked_input[:, :, UNMONITORED_IDX, :] = 0.0

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
            recon = model(masked_input)   # (B, T, N, 1)
            loss  = F.mse_loss(
                recon[:, :, UNMONITORED_IDX, :],
                x_full[:, :, UNMONITORED_IDX, :]
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader):
    model.eval()
    total_loss = 0.0

    for x_full in loader:
        x_full = x_full.to(DEVICE, non_blocking=True)
        masked_input = x_full.clone()
        masked_input[:, :, UNMONITORED_IDX, :] = 0.0

        recon = model(masked_input)
        loss  = F.mse_loss(
            recon[:, :, UNMONITORED_IDX, :],
            x_full[:, :, UNMONITORED_IDX, :]
        )
        total_loss += loss.item()

    return total_loss / len(loader)

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ----------------------------------------------------------
    # Load scenario folders
    # ----------------------------------------------------------
    train_folders = folders_from_manifest(MANIFEST_TRAIN)
    val_folders   = folders_from_manifest(MANIFEST_VAL)
    test_folders  = folders_from_manifest(MANIFEST_TEST)

    if len(train_folders) == 0:
        raise RuntimeError(f"No valid scenarios found via {MANIFEST_TRAIN}")

    print(f"Train: {len(train_folders)}  Val: {len(val_folders)}  Test: {len(test_folders)}")

    # ----------------------------------------------------------
    # Normalisation stats from train data only
    # ----------------------------------------------------------
    print("Computing normalisation statistics...")
    mu, sigma = compute_mu_sigma(train_folders)
    print(f"  mu shape={mu.shape}  sigma shape={sigma.shape}")

    # Baseline template (no-leak average) — saved for downstream detector
    print("Computing baseline template from no-leak training scenarios...")
    baseline = compute_baseline_template(train_folders)
    print(f"  baseline shape={baseline.shape}")

    # ----------------------------------------------------------
    # Adjacency matrix  —  full 10-node topology
    # ----------------------------------------------------------
    adj = build_sensor_adjacency()
    print(f"Adjacency matrix shape: {adj.shape}")

    # ----------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------
    print("Building datasets...")
    train_ds = AutoencoderDataset(train_folders, mu, sigma, window=WINDOW, stride=STRIDE)
    val_ds   = AutoencoderDataset(val_folders,   mu, sigma, window=WINDOW, stride=STRIDE)
    test_ds  = AutoencoderDataset(test_folders,  mu, sigma, window=WINDOW, stride=STRIDE)

    print(f"  Train samples: {len(train_ds)}")
    print(f"  Val   samples: {len(val_ds)}")
    print(f"  Test  samples: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ----------------------------------------------------------
    # Model, optimiser, scaler
    # ----------------------------------------------------------
    model     = STGCNAutoencoder(adj, node_feats=NODE_FEATS, hidden_1=HIDDEN_1,
                                 hidden_2=HIDDEN_2, kernel_size=KERNEL_SIZE,
                                 dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------
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
            marker = "  *** best ***"

        print(f"Epoch {ep+1:3d}/{EPOCHS}  "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{marker}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest val loss: {best_val_loss:.6f}")

    # ----------------------------------------------------------
    # Save bundle
    # ----------------------------------------------------------
    bundle = {
        "model_state_dict":  model.state_dict(),
        "adjacency":         adj,
        "mu":                mu,               # (N, 1)
        "sigma":             sigma,            # (N, 1)
        "baseline_template": baseline,         # (T, N)  — for downstream detector
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
    print(f"\n[OK] Saved bundle -> {SAVE_PATH}")

    # ----------------------------------------------------------
    # Quick test-set loss report
    # ----------------------------------------------------------
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loss   = evaluate_loss(model, test_loader)
    print(f"Test MSE (normalised, unmonitored nodes): {test_loss:.6f}")


if __name__ == "__main__":
    main()
