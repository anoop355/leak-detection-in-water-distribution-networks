"""
Trains SingleLeakSTGCN model per row in sensor_placements.csv using the following 
implementation details:
  - Temporal attention pooling
  - LAMBDA_DETECT = 1.0,  LAMBDA_PIPE = 2.0
  - Sensor-subset adjacency (edges retained only when both endpoints are sensors)
  - 2-channel node features: [raw, deviation_from_baseline]
  - Scenario-level oversampling
  - 25 epochs, same stgcn_dataset/ manifests

Bundles are saved to:  stgcn_placement_bundles/stgcn_bundle_{model_name}.pt


"""

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


# CONFIG 

DATASET_ROOT   = "stgcn_dataset/scenarios"
MANIFEST_TRAIN = "stgcn_dataset/manifests/manifest_train.csv"
MANIFEST_VAL   = "stgcn_dataset/manifests/manifest_val.csv"
MANIFEST_TEST  = "stgcn_dataset/manifests/manifest_test.csv"

PLACEMENTS_CSV = Path("sensor_placements.csv")
MODELS_ROOT    = Path("stgcn_placement_bundles")

# Set to a model_name string (e.g. "S6-B") to resume from that row,
# or None to start from the first row.
START_FROM_MODEL = None

NODE_FEATS  = 2        # [raw, deviation_from_baseline]
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

# REPRODUCIBILITY

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# LABEL HELPERS 

def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_no_leak(labels: dict):
    return int(labels.get("label_detection", 0)) == 0

def encode_labels_from_json(labels: dict):
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


# FILE / SCENARIO HELPERS

def folders_from_manifest(manifest_path: str):
    df = pd.read_csv(manifest_path)
    folders = []
    for scn_id in df["scenario_id"].values:
        path = os.path.join(DATASET_ROOT, f"scenario_{int(scn_id):05d}")
        sig  = os.path.join(path, "data.csv")
        lab  = os.path.join(path, "labels.json")
        if os.path.isfile(sig) and os.path.isfile(lab):
            folders.append(path)
    return folders

def read_signals(folder: str, sensor_names: list):
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    missing = [c for c in sensor_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {folder}: {missing}")
    return df[sensor_names].to_numpy(dtype=np.float32)


# BASELINE + NORMALISATION  (parameterised by sensor_names)

def compute_baseline_template(train_folders, sensor_names):
    no_leak_arrays = []
    for folder in train_folders:
        labels = load_labels(os.path.join(folder, "labels.json"))
        if is_no_leak(labels):
            arr = read_signals(folder, sensor_names)
            no_leak_arrays.append(arr)
    if not no_leak_arrays:
        raise RuntimeError("No no-leak scenarios found in training split.")
    min_len = min(a.shape[0] for a in no_leak_arrays)
    cropped = [a[:min_len] for a in no_leak_arrays]
    return np.mean(np.stack(cropped, axis=0), axis=0).astype(np.float32)

def make_node_features(raw_signals: np.ndarray, baseline_template: np.ndarray):
    T = raw_signals.shape[0]
    base = baseline_template[:T]
    dev  = raw_signals - base
    return np.stack([raw_signals, dev], axis=-1).astype(np.float32)  # (T, N, 2)

def compute_mu_sigma(train_folders, baseline_template, sensor_names):
    num_nodes = len(sensor_names)
    sum_x  = np.zeros((num_nodes, NODE_FEATS), dtype=np.float64)
    sum_x2 = np.zeros((num_nodes, NODE_FEATS), dtype=np.float64)
    total_T = 0
    for folder in train_folders:
        raw   = read_signals(folder, sensor_names)
        feats = make_node_features(raw, baseline_template)
        sum_x  += feats.sum(axis=0)
        sum_x2 += (feats ** 2).sum(axis=0)
        total_T += feats.shape[0]
    mu  = sum_x / total_T
    var = np.maximum((sum_x2 / total_T) - (mu ** 2), 1e-8)
    return mu.astype(np.float32), np.sqrt(var).astype(np.float32)


# GRAPH  (parameterised by sensor_names)

# Full physical topology of the WDN (all nodes including non-sensor ones)
_FULL_EDGES = [
    ("Q1a", "P2"), ("P2",  "Q2a"), ("Q2a", "P3"),
    ("P3",  "Q3a"), ("Q3a", "P4"), ("P4",  "Q4a"),
    ("Q4a", "P5"),  ("P4",  "Q5a"), ("Q5a", "P6"),
]

def build_sensor_adjacency(sensor_names: list) -> np.ndarray:
    """
    Sensor-subset adjacency: edges are kept only when BOTH endpoints are
    present in sensor_names (same rule as v5 / train_stgtcn_detection_localisation5.py).
    Self-loops added; symmetric normalisation applied.
    """
    sensor_set = set(sensor_names)
    idx = {name: i for i, name in enumerate(sensor_names)}
    N   = len(sensor_names)
    A   = np.zeros((N, N), dtype=np.float32)

    for a, b in _FULL_EDGES:
        if a in sensor_set and b in sensor_set:
            i, j = idx[a], idx[b]
            A[i, j] = 1.0
            A[j, i] = 1.0

    A = A + np.eye(N, dtype=np.float32)

    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    return (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)


# DATASET

class ScenarioWindowDataset(Dataset):
    def __init__(self, folders, sensor_names, baseline_template, mu, sigma,
                 window=12, stride=1):
        self.window  = window
        self.stride  = stride

        self.features      = []
        self.targets       = []
        self.names         = []
        self.window_counts = []
        self.cum_counts    = [0]

        for folder in folders:
            raw   = read_signals(folder, sensor_names)
            feats = make_node_features(raw, baseline_template)
            feats = (feats - mu[None]) / (sigma[None] + 1e-8)
            feats = feats.astype(np.float32)

            labels = load_labels(os.path.join(folder, "labels.json"))
            tgt    = encode_labels_from_json(labels)

            T = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            if n_windows == 0:
                continue

            self.features.append(feats)
            self.targets.append(tgt)
            self.names.append(os.path.basename(folder))
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        # Scenario-level oversampling
        detect_per_scenario = [t[0] for t in self.targets]
        class_counts = Counter(detect_per_scenario)
        max_scenarios = max(class_counts.values())

        new_features, new_targets, new_names = [], [], []
        for i, det in enumerate(detect_per_scenario):
            repeats = max(1, round(max_scenarios / class_counts[det]))
            new_features.extend([self.features[i]] * repeats)
            new_targets.extend([self.targets[i]]   * repeats)
            new_names.extend([self.names[i]]        * repeats)

        self.features = new_features
        self.targets  = new_targets
        self.names    = new_names

        self.window_counts = []
        self.cum_counts    = [0]
        for feats in self.features:
            T = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

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
        x      = x_full[start : start + self.window]
        detect, pipe_t, pos_t, size_t = self.targets[scenario_idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(detect, dtype=torch.long),
            torch.tensor(pipe_t, dtype=torch.long),
            torch.tensor(pos_t,  dtype=torch.float32),
            torch.tensor(size_t, dtype=torch.long),
        )


# MODEL  (TemporalAttentionPool + SingleLeakSTGCN)

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
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix,
                              kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix,
                              kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix,
                              kernel_size, 4, dropout)

        head_in     = num_nodes * hidden_2
        head_hidden = 64

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
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        z = self.temporal_pool(x)
        return (
            self.detect_head(z),
            self.pipe_head(z),
            self.size_head(z),
            self.pos_head(z).squeeze(1),
        )


# TRAIN ONE CONFIGURATION

def train_one_configuration(model_name: str, sensor_names: list,
                             train_folders, val_folders, test_folders):
    print("=" * 70)
    print(f"Training {model_name}  |  sensors ({len(sensor_names)}): {sensor_names}")
    print("=" * 70)

    # Build baseline + normalisation stats from train split only
    baseline_template = compute_baseline_template(train_folders, sensor_names)
    mu, sigma = compute_mu_sigma(train_folders, baseline_template, sensor_names)

    # Datasets
    train_ds = ScenarioWindowDataset(
        train_folders, sensor_names, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )
    test_ds = ScenarioWindowDataset(
        test_folders, sensor_names, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # Adjacency
    num_nodes = len(sensor_names)
    adj = build_sensor_adjacency(sensor_names)

    # Model
    model = SingleLeakSTGCN(
        adj, num_nodes, HIDDEN_1, HIDDEN_2, KERNEL_SIZE, DROPOUT, NODE_FEATS
    ).to(DEVICE)

    # Class weights (window-level)
    window_class_counts = Counter()
    for i, tgt in enumerate(train_ds.targets):
        window_class_counts[tgt[0]] += train_ds.window_counts[i]

    cw = np.array([1.0 / max(window_class_counts.get(c, 1), 1)
                   for c in range(2)], dtype=np.float32)
    cw = torch.tensor(cw / cw.sum() * 2.0, dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scaler    = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    loss_detect = nn.CrossEntropyLoss(weight=cw)
    loss_pipe   = nn.CrossEntropyLoss(reduction="none")
    loss_size   = nn.CrossEntropyLoss(reduction="none")
    loss_pos    = nn.SmoothL1Loss(reduction="none")

    # Training loop
    for ep in range(EPOCHS):
        model.train()
        running = 0.0
        for x, detect, pipe_t, pos_t, size_t in train_loader:
            x      = x.to(DEVICE, non_blocking=True)
            detect = detect.to(DEVICE, non_blocking=True)
            pipe_t = pipe_t.to(DEVICE, non_blocking=True)
            pos_t  = pos_t.to(DEVICE,  non_blocking=True)
            size_t = size_t.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
                detect_logits, pipe_logits, size_logits, pos_pred = model(x)
                Ld = loss_detect(detect_logits, detect)
                leak_mask = detect.float()
                denom     = leak_mask.sum().clamp(min=1.0)
                Lp = (loss_pipe(pipe_logits, pipe_t) * leak_mask).sum() / denom
                Ls = (loss_size(size_logits, size_t) * leak_mask).sum() / denom
                Lr = (loss_pos(pos_pred, pos_t)      * leak_mask).sum() / denom
                loss = (LAMBDA_DETECT * Ld + LAMBDA_PIPE * Lp
                        + LAMBDA_SIZE * Ls + LAMBDA_POS * Lr)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())

        print(f"[{model_name}] Epoch {ep+1}/{EPOCHS}, "
              f"loss={running/len(train_loader):.4f}")

    # Quick in-script evaluation (window-level pipe F1 on test split)

    model.eval()
    pipe_true_all, pipe_pred_all, detect_true_all = [], [], []
    pos_true_all, pos_pred_all = [], []

    with torch.no_grad():
        for x, detect, pipe_t, pos_t, size_t in test_loader:
            x = x.to(DEVICE)
            _, pipe_logits, _, pos_pred = model(x)
            pipe_true_all.append(pipe_t.numpy())
            pipe_pred_all.append(pipe_logits.argmax(dim=1).cpu().numpy())
            pos_true_all.append(pos_t.numpy())
            pos_pred_all.append(pos_pred.cpu().numpy())
            detect_true_all.append(detect.numpy())

    pipe_true   = np.concatenate(pipe_true_all)
    pipe_pred   = np.concatenate(pipe_pred_all)
    pos_true    = np.concatenate(pos_true_all)
    pos_pred_np = np.concatenate(pos_pred_all)
    det_true    = np.concatenate(detect_true_all)

    leak_mask = det_true > 0.5
    f1_pipe = f1_score(pipe_true[leak_mask], pipe_pred[leak_mask],
                       average="macro", labels=list(range(NUM_PIPES)))
    mae = float(np.mean(np.abs(pos_true[leak_mask] - pos_pred_np[leak_mask])))

    print(f"[{model_name}] Pipe F1 Macro (train split): {f1_pipe:.4f}")
    print(f"[{model_name}] Position MAE  (train split): {mae:.4f}")

    # Save bundle
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    bundle_path = MODELS_ROOT / f"stgcn_bundle_{model_name}.pt"

    bundle = {
        "model_type":        "stgcn_single_leak_v4",
        "model_name":        model_name,
        "model_state_dict":  model.state_dict(),
        "adjacency":         adj,
        "mu":                mu,
        "sigma":             sigma,
        "baseline_template": baseline_template,
        "sensor_names":      sensor_names,
        "window":            WINDOW,
        "stride":            STRIDE,
        "node_feats":        NODE_FEATS,
        "hidden_1":          HIDDEN_1,
        "hidden_2":          HIDDEN_2,
        "kernel_size":       KERNEL_SIZE,
        "dilations":         [1, 2, 4],
        "num_blocks":        3,
        "head_in":           num_nodes * HIDDEN_2,
        "dropout":           DROPOUT,
        "dataset_root":      DATASET_ROOT,
        "pipe_classes":      PIPE_CLASSES,
        "size_classes":      SIZE_CLASSES,
        "seed":              SEED,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "lr":                LR,
        "weight_decay":      WEIGHT_DECAY,
        "lambda_detect":     LAMBDA_DETECT,
        "lambda_pipe":       LAMBDA_PIPE,
    }

    torch.save(bundle, str(bundle_path))
    print(f"[OK] Saved bundle: {bundle_path}\n")


# MAIN

def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {DEVICE}\n")

    if not PLACEMENTS_CSV.exists():
        raise FileNotFoundError(f"sensor_placements.csv not found: {PLACEMENTS_CSV}")

    # Load manifests once — all configurations share the same dataset split
    train_folders = folders_from_manifest(MANIFEST_TRAIN)
    val_folders   = folders_from_manifest(MANIFEST_VAL)
    test_folders  = folders_from_manifest(MANIFEST_TEST)

    if not train_folders:
        raise RuntimeError(f"No valid scenario folders found via {MANIFEST_TRAIN}")

    print(f"Dataset  —  train: {len(train_folders)}  "
          f"val: {len(val_folders)}  test: {len(test_folders)}\n")

    placements = pd.read_csv(PLACEMENTS_CSV)
    placements["model_name"]    = placements["model_name"].astype(str).str.strip()
    placements["configuration"] = placements["configuration"].astype(str).str.strip()

    # Skip S10-A (full instrumentation baseline — too wide for a single ST-GCN config)
    placements = placements[placements["model_name"] != "S10-A"].copy()

    if START_FROM_MODEL is not None:
        model_list = placements["model_name"].tolist()
        if START_FROM_MODEL not in model_list:
            raise ValueError(
                f"START_FROM_MODEL '{START_FROM_MODEL}' not found in "
                f"sensor_placements.csv"
            )
        start_idx  = model_list.index(START_FROM_MODEL)
        placements = placements.iloc[start_idx:].copy()

    print(f"Configurations to train: {len(placements)}")
    print(placements[["model_name", "k_budget", "configuration"]].to_string(index=False))
    print()

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    for _, row in placements.iterrows():
        model_name   = row["model_name"]
        sensor_names = [s.strip() for s in row["configuration"].split(",")
                        if s.strip()]

        if not sensor_names:
            print(f"[WARN] Skipping {model_name}: empty configuration")
            continue

        bundle_path = MODELS_ROOT / f"stgcn_bundle_{model_name}.pt"
        if bundle_path.exists():
            print(f"[SKIP] {model_name} — bundle already exists at {bundle_path}")
            continue

        train_one_configuration(
            model_name, sensor_names,
            train_folders, val_folders, test_folders
        )

    print("[DONE] Finished training all ST-GCN sensor placement configurations.")


if __name__ == "__main__":
    main()
