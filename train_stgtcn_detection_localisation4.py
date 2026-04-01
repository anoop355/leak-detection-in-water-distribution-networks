import os
import json
import math
import random
import bisect
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT   = "stgcn_dataset/scenarios"
MANIFEST_TRAIN = "stgcn_dataset/manifests/manifest_train.csv"
MANIFEST_VAL   = "stgcn_dataset/manifests/manifest_val.csv"
MANIFEST_TEST  = "stgcn_dataset/manifests/manifest_test.csv"
SAVE_PATH      = "stgcn_bundle_v4_temporal_attn.pt"

SENSOR_NAMES = ["P4", "Q1a", "Q3a"]
NUM_NODES = len(SENSOR_NAMES)
NODE_FEATS = 2   # [raw, deviation_from_baseline]

WINDOW = 12      # each scenario has exactly 12 timesteps (15-min intervals)
STRIDE = 1
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
SEED = 42

# Compact model
HIDDEN_1 = 16
HIDDEN_2 = 32
KERNEL_SIZE = 5

NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES       # 5 => NONE
PIPE_CLASSES  = NUM_PIPES + 1   # 6

SIZE_TO_IDX  = {"S": 0, "M": 1, "L": 2}
SIZE_NONE_IDX = 3
SIZE_CLASSES  = 4

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
# LABEL HELPERS
# ============================================================
def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_no_leak(labels: dict):
    return int(labels.get("label_detection", 0)) == 0

def encode_labels_from_json(labels: dict):
    """
    Converts a single-leak labels.json into scalar targets.

    Returns:
      detect:    int  0=no-leak, 1=leak
      pipe_t:    int  0..4 = pipe 1..5,  5 = NONE
      pos_t:     float  0.0 if no leak
      size_t:    int  0=S,1=M,2=L,  3 = NONE
    """
    detect = int(labels.get("label_detection", 0))

    if detect == 1:
        pipe_t = int(labels.get("label_pipe", 1)) - 1           # 1..5 -> 0..4
        pos_t  = float(labels.get("label_position", 0.0))
        sl     = str(labels.get("label_size", "S")).upper()
        size_t = SIZE_TO_IDX.get(sl, 0)
    else:
        pipe_t = PIPE_NONE_IDX
        pos_t  = 0.0
        size_t = SIZE_NONE_IDX

    return detect, pipe_t, pos_t, size_t

# ============================================================
# FILE / SCENARIO HELPERS
# ============================================================
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

def read_signals(folder: str):
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    missing = [c for c in SENSOR_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {folder}: {missing}")
    return df[SENSOR_NAMES].to_numpy(dtype=np.float32)

# ============================================================
# BASELINE + NORMALISATION
# ============================================================
def compute_baseline_template(train_folders):
    """
    Baseline template = average of all NO-LEAK training scenarios.
    Uses only training data to avoid leakage.
    """
    no_leak_arrays = []
    for folder in train_folders:
        labels = load_labels(os.path.join(folder, "labels.json"))
        if is_no_leak(labels):
            arr = read_signals(folder)
            no_leak_arrays.append(arr)

    if len(no_leak_arrays) == 0:
        raise RuntimeError("No no-leak scenarios found in training split. Cannot build baseline template.")

    min_len = min(a.shape[0] for a in no_leak_arrays)
    cropped  = [a[:min_len] for a in no_leak_arrays]
    baseline = np.mean(np.stack(cropped, axis=0), axis=0).astype(np.float32)  # (T, N)

    return baseline

def make_node_features(raw_signals: np.ndarray, baseline_template: np.ndarray):
    """
    raw_signals:       (T, N)
    baseline_template: (Tb, N)

    Returns:
      feats: (T, N, 2)
        feat 0 = raw sensor reading
        feat 1 = deviation from baseline
    """
    T  = raw_signals.shape[0]
    Tb = baseline_template.shape[0]

    if Tb < T:
        raise ValueError(
            f"Baseline template shorter than scenario length: baseline={Tb}, scenario={T}"
        )

    base = baseline_template[:T]   # (T, N)
    dev  = raw_signals - base

    feats = np.stack([raw_signals, dev], axis=-1).astype(np.float32)  # (T, N, 2)
    return feats

def compute_mu_sigma(train_folders, baseline_template):
    """
    Streaming mean/std over train data only.
    Stats are computed per node-feature channel => shape (N, 2)
    """
    sum_x  = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    sum_x2 = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    total_T = 0

    for folder in train_folders:
        raw   = read_signals(folder)
        feats = make_node_features(raw, baseline_template)  # (T, N, 2)

        sum_x  += feats.sum(axis=0)
        sum_x2 += (feats ** 2).sum(axis=0)
        total_T += feats.shape[0]

    mu  = sum_x / total_T
    var = (sum_x2 / total_T) - (mu ** 2)
    var = np.maximum(var, 1e-8)
    sigma = np.sqrt(var)

    return mu.astype(np.float32), sigma.astype(np.float32)

# ============================================================
# GRAPH
# ============================================================
def build_sensor_adjacency():
    """
    Physical sensor graph based on the true network structure:

        Q1a -- P2 -- Q2a -- P3 -- Q3a -- P4
                                       /   \
                                    Q4a     Q5a
                                     |       |
                                    P5      P6

    Only edges where both endpoints are in SENSOR_NAMES are included,
    so the matrix adapts automatically to any sensor subset.
    """
    sensor_set = set(SENSOR_NAMES)
    idx = {name: i for i, name in enumerate(SENSOR_NAMES)}
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    def connect(a, b, w=1.0):
        if a in sensor_set and b in sensor_set:
            i, j = idx[a], idx[b]
            A[i, j] = w
            A[j, i] = w

    # True physical topology
    connect("Q1a", "P2")
    connect("P2",  "Q2a")
    connect("Q2a", "P3")
    connect("P3",  "Q3a")
    connect("Q3a", "P4")
    connect("P4",  "Q4a")
    connect("Q4a", "P5")
    connect("P4",  "Q5a")
    connect("Q5a", "P6")

    # Self-loops
    A = A + np.eye(NUM_NODES, dtype=np.float32)

    # Symmetric normalization
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return A_hat.astype(np.float32)

# ============================================================
# MEMORY-EFFICIENT WINDOW DATASET
# ============================================================
class ScenarioWindowDataset(Dataset):
    def __init__(self, folders, baseline_template, mu, sigma, window=12, stride=1):
        self.window = window
        self.stride = stride

        self.features     = []     # each entry: (T, N, 2)
        self.targets      = []     # each entry: tuple(detect, pipe_t, pos_t, size_t)
        self.names        = []
        self.window_counts = []
        self.cum_counts   = [0]

        for folder in folders:
            raw   = read_signals(folder)
            feats = make_node_features(raw, baseline_template)
            feats = (feats - mu[None, :, :]) / (sigma[None, :, :] + 1e-8)
            feats = feats.astype(np.float32)

            labels = load_labels(os.path.join(folder, "labels.json"))
            tgt    = encode_labels_from_json(labels)

            T         = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            if n_windows == 0:
                continue

            self.features.append(feats)
            self.targets.append(tgt)
            self.names.append(os.path.basename(folder))
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        # ------------------------------------------------------------------
        # Scenario-level oversampling so no-leak and leak classes are balanced.
        # ------------------------------------------------------------------
        detect_per_scenario = [t[0] for t in self.targets]
        class_scenario_counts = Counter(detect_per_scenario)
        max_scenarios = max(class_scenario_counts.values())

        print(f"  Scenario counts before oversampling: {dict(sorted(class_scenario_counts.items()))}")

        new_features = []
        new_targets  = []
        new_names    = []

        for i, det in enumerate(detect_per_scenario):
            count_for_class = class_scenario_counts[det]
            repeats = max(1, round(max_scenarios / count_for_class))
            new_features.extend([self.features[i]] * repeats)
            new_targets.extend([self.targets[i]]   * repeats)
            new_names.extend([self.names[i]]        * repeats)

        self.features = new_features
        self.targets  = new_targets
        self.names    = new_names

        # Rebuild window counts and cumulative index after oversampling
        self.window_counts = []
        self.cum_counts    = [0]
        for feats in self.features:
            T         = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        new_det = [t[0] for t in self.targets]
        print(f"  Scenario counts after  oversampling: {dict(sorted(Counter(new_det).items()))}")

    def __len__(self):
        return self.cum_counts[-1]

    def _locate_index(self, idx):
        scenario_idx = bisect.bisect_right(self.cum_counts, idx) - 1
        local_idx    = idx - self.cum_counts[scenario_idx]
        return scenario_idx, local_idx

    def __getitem__(self, idx):
        scenario_idx, local_idx = self._locate_index(idx)
        x_full = self.features[scenario_idx]
        start  = local_idx * self.stride
        end    = start + self.window
        x      = x_full[start:end]   # (W, N, 2)

        detect, pipe_t, pos_t, size_t = self.targets[scenario_idx]

        return (
            torch.tensor(x, dtype=torch.float32),                  # (W, N, 2)
            torch.tensor(detect, dtype=torch.long),
            torch.tensor(pipe_t, dtype=torch.long),
            torch.tensor(pos_t,  dtype=torch.float32),
            torch.tensor(size_t, dtype=torch.long),
        )

# ============================================================
# MODEL
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

class TemporalAttentionPool(nn.Module):
    """
    Learnable soft-attention pooling over the time axis.
    Input:  (B, T, N, C)
    Output: (B, N * C)   — same shape as x.mean(dim=1).reshape(B, -1)
    """
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * num_nodes, 1)

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat  = x.reshape(B, T, N * C)           # (B, T, N*C)
        scores  = self.attn(x_flat)                 # (B, T, 1)
        weights = torch.softmax(scores, dim=1)      # (B, T, 1)
        z       = (x_flat * weights).sum(dim=1)     # (B, N*C)
        return z


class SingleLeakSTGCN(nn.Module):
    """
    Shared ST-GCN backbone + single-leak output heads:
      - detect: 2 classes (0=no-leak, 1=leak)
      - pipe:   6 classes (5 pipes + NONE)
      - size:   4 classes (S/M/L + NONE)
      - pos:    scalar regression
    """
    def __init__(self, adj_matrix):
        super().__init__()

        self.block1 = STBlock(
            in_ch=NODE_FEATS, out_ch=HIDDEN_1,
            adj_matrix=adj_matrix, kernel_size=KERNEL_SIZE, dilation=1, dropout=DROPOUT
        )
        self.block2 = STBlock(
            in_ch=HIDDEN_1, out_ch=HIDDEN_2,
            adj_matrix=adj_matrix, kernel_size=KERNEL_SIZE, dilation=2, dropout=DROPOUT
        )
        self.block3 = STBlock(
            in_ch=HIDDEN_2, out_ch=HIDDEN_2,
            adj_matrix=adj_matrix, kernel_size=KERNEL_SIZE, dilation=4, dropout=DROPOUT
        )

        head_in     = NUM_NODES * HIDDEN_2   # 10 * 32 = 320
        head_hidden = 64

        self.detect_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, 2),
        )

        self.pipe_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, PIPE_CLASSES),
        )

        self.size_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, SIZE_CLASSES),
        )

        self.pos_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid(),   # keep positions in [0,1]
        )

        self.temporal_pool = TemporalAttentionPool(HIDDEN_2, NUM_NODES)

    def forward(self, x):
        # x: (B, T, N, 2)
        x = self.block1(x)   # (B, T, N, HIDDEN_1)
        x = self.block2(x)   # (B, T, N, HIDDEN_2)
        x = self.block3(x)   # (B, T, N, HIDDEN_2)

        # Temporal attention pooling over time, retaining node dimension.
        z = self.temporal_pool(x)            # (B, N * HIDDEN_2)

        detect_logits = self.detect_head(z)           # (B, 2)
        pipe_logits   = self.pipe_head(z)             # (B, PIPE_CLASSES)
        size_logits   = self.size_head(z)             # (B, SIZE_CLASSES)
        pos_pred      = self.pos_head(z).squeeze(1)   # (B,)

        return detect_logits, pipe_logits, size_logits, pos_pred

# ============================================================
# EVAL HELPERS
# ============================================================
@torch.no_grad()
def scenario_level_detection_accuracy(model, dataset, batch_size=256):
    model.eval()

    true_labels = []
    pred_labels = []

    for i in range(len(dataset.features)):
        x_full = dataset.features[i]    # (T, N, 2)
        detect, _, _, _ = dataset.targets[i]

        T         = x_full.shape[0]
        n_windows = max(0, (T - dataset.window) // dataset.stride + 1)
        if n_windows == 0:
            continue

        windows = []
        for w in range(n_windows):
            s = w * dataset.stride
            e = s + dataset.window
            windows.append(x_full[s:e])
        windows = np.stack(windows, axis=0).astype(np.float32)

        batch_preds = []
        for start in range(0, windows.shape[0], batch_size):
            xb = torch.tensor(windows[start:start+batch_size], dtype=torch.float32, device=DEVICE)
            detect_logits, _, _, _ = model(xb)
            batch_preds.extend(detect_logits.argmax(dim=1).cpu().numpy().tolist())

        # Majority vote across windows
        vals, counts = np.unique(np.array(batch_preds), return_counts=True)
        final_pred   = int(vals[np.argmax(counts)])

        true_labels.append(detect)
        pred_labels.append(final_pred)

    return accuracy_score(true_labels, pred_labels)

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: NO GPU")

    train_folders = folders_from_manifest(MANIFEST_TRAIN)
    val_folders   = folders_from_manifest(MANIFEST_VAL)
    test_folders  = folders_from_manifest(MANIFEST_TEST)

    if len(train_folders) == 0:
        raise RuntimeError(f"No valid scenario folders found via {MANIFEST_TRAIN}")

    print(f"Train: {len(train_folders)}  Val: {len(val_folders)}  Test: {len(test_folders)}")

    # Build baseline + norm stats from TRAIN ONLY
    baseline_template = compute_baseline_template(train_folders)
    mu, sigma = compute_mu_sigma(train_folders, baseline_template)

    # Build datasets
    print("Building train dataset...")
    train_ds = ScenarioWindowDataset(
        train_folders, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )
    print("Building val dataset...")
    val_ds = ScenarioWindowDataset(
        val_folders, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )
    print("Building test dataset...")
    test_ds = ScenarioWindowDataset(
        test_folders, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )

    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows:   {len(val_ds)}")
    print(f"Test windows:  {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    adj   = build_sensor_adjacency()
    model = SingleLeakSTGCN(adj).to(DEVICE)

    # Class weights for detection loss computed at window level
    window_class_counts = Counter()
    for i, tgt in enumerate(train_ds.targets):
        detect   = tgt[0]
        n_windows = train_ds.window_counts[i]
        window_class_counts[detect] += n_windows

    print(f"Window-level class counts (train): {dict(sorted(window_class_counts.items()))}")

    class_weights = []
    for c in range(2):
        class_weights.append(1.0 / max(window_class_counts.get(c, 1), 1))
    class_weights = np.array(class_weights, dtype=np.float32)
    class_weights = class_weights / class_weights.sum() * 2.0
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler    = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    # Loss functions.
    # Pipe and size use reduction="none" so the detection mask can be applied
    # before averaging, excluding no-leak scenarios from localisation gradients.
    loss_detect = nn.CrossEntropyLoss(weight=class_weights)
    loss_pipe   = nn.CrossEntropyLoss(reduction="none")
    loss_size   = nn.CrossEntropyLoss(reduction="none")
    loss_pos    = nn.SmoothL1Loss(reduction="none")

    LAMBDA_DETECT = 2.0
    LAMBDA_PIPE   = 1.0
    LAMBDA_SIZE   = 1.0
    LAMBDA_POS    = 0.5

    # Train
    for ep in range(EPOCHS):
        model.train()
        running = 0.0

        for x, detect, pipe_t, pos_t, size_t in train_loader:
            x      = x.to(DEVICE, non_blocking=True)       # (B, W, N, 2)
            detect = detect.to(DEVICE, non_blocking=True)  # (B,)
            pipe_t = pipe_t.to(DEVICE, non_blocking=True)  # (B,)
            pos_t  = pos_t.to(DEVICE,  non_blocking=True)  # (B,)
            size_t = size_t.to(DEVICE, non_blocking=True)  # (B,)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                detect_logits, pipe_logits, size_logits, pos_pred = model(x)

                Ld = loss_detect(detect_logits, detect)

                # Pipe/size/pos losses masked to leak scenarios only
                leak_mask = detect.float()                                 # (B,)
                denom     = leak_mask.sum().clamp(min=1.0)

                Lp = (loss_pipe(pipe_logits, pipe_t) * leak_mask).sum() / denom
                Ls = (loss_size(size_logits, size_t) * leak_mask).sum() / denom
                Lr = (loss_pos(pos_pred, pos_t)      * leak_mask).sum() / denom

                loss = LAMBDA_DETECT * Ld + LAMBDA_PIPE * Lp + LAMBDA_SIZE * Ls + LAMBDA_POS * Lr

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())

        print(f"Epoch {ep+1}/{EPOCHS}, loss={running/len(train_loader):.4f}")

    # Save bundle
    bundle = {
        "model_type": "stgcn_single_leak_v4",
        "model_state_dict": model.state_dict(),
        "adjacency": adj,
        "mu": mu,
        "sigma": sigma,
        "baseline_template": baseline_template,
        "sensor_names": SENSOR_NAMES,
        "window": WINDOW,
        "stride": STRIDE,
        "node_feats": NODE_FEATS,
        "hidden_1": HIDDEN_1,
        "hidden_2": HIDDEN_2,
        "kernel_size": KERNEL_SIZE,
        "dilations": [1, 2, 4],
        "num_blocks": 3,
        "head_in": NUM_NODES * HIDDEN_2,
        "dropout": DROPOUT,
        "dataset_root": DATASET_ROOT,
        "pipe_classes": PIPE_CLASSES,
        "size_classes": SIZE_CLASSES,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
    }

    torch.save(bundle, SAVE_PATH)
    print(f"[OK] Saved {SAVE_PATH}")

    # Scenario-level detection accuracy (majority vote across windows)
    test_det_acc = scenario_level_detection_accuracy(model, test_ds, batch_size=256)
    print("\n=== SCENARIO-LEVEL DETECTION ===")
    print(f"Accuracy: {test_det_acc:.4f}")

    # Window-level pipe / position evaluation
    model.eval()
    pipe_true_all, pipe_pred_all = [], []
    pos_true_all,  pos_pred_all  = [], []
    detect_true_all = []

    with torch.no_grad():
        for x, detect, pipe_t, pos_t, size_t in test_loader:
            x = x.to(DEVICE)
            _, pipe_logits, _, pos_pred = model(x)

            pipe_true_all.append(pipe_t.numpy())
            pipe_pred_all.append(pipe_logits.argmax(dim=1).cpu().numpy())
            pos_true_all.append(pos_t.numpy())
            pos_pred_all.append(pos_pred.cpu().numpy())
            detect_true_all.append(detect.numpy())

    pipe_true_all   = np.concatenate(pipe_true_all)
    pipe_pred_all   = np.concatenate(pipe_pred_all)
    pos_true_all    = np.concatenate(pos_true_all)
    pos_pred_all    = np.concatenate(pos_pred_all)
    detect_true_all = np.concatenate(detect_true_all)

    # All localisation metrics use leak scenarios only
    leak_mask = detect_true_all > 0.5

    # Pipe F1 Macro — leak scenarios only, excluding NONE class
    pipe_true_leak = pipe_true_all[leak_mask]
    pipe_pred_leak = pipe_pred_all[leak_mask]
    f1_pipe = f1_score(pipe_true_leak, pipe_pred_leak, average="macro", labels=list(range(NUM_PIPES)))
    print(f"\n=== PIPE F1 MACRO (leak scenarios only) ===")
    print(f"F1 Macro: {f1_pipe:.4f}")

    # Position MAE / RMSE — leak scenarios only
    pos_true_leak = pos_true_all[leak_mask]
    pos_pred_leak = pos_pred_all[leak_mask]
    if len(pos_true_leak) > 0:
        mae  = float(np.mean(np.abs(pos_true_leak - pos_pred_leak)))
        rmse = float(np.sqrt(np.mean((pos_true_leak - pos_pred_leak) ** 2)))
        print(f"\n=== POSITION (leak scenarios only) ===")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
