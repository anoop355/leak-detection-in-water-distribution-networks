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
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT = "/content/training_cases_output"
DRIVE_SAVE_PATH = "/content/drive/MyDrive/colab_upload/multileak_stgcn_bundle_v3.pt"

SENSOR_NAMES = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
NUM_NODES = len(SENSOR_NAMES)
NODE_FEATS = 2   # [raw, deviation_from_baseline]

WINDOW = 180
STRIDE = 20      # increased from 10 — halves window count, allows more epochs
BATCH_SIZE = 64
EPOCHS = 25      # increased from 10 — gives rare classes more gradient exposure
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25
SEED = 42

# Compact model to stay closer to ~1 hour
HIDDEN_1 = 16
HIDDEN_2 = 32
KERNEL_SIZE = 5

MAX_LEAKS = 3
NUM_PIPES = 5
PIPE_NONE_IDX = NUM_PIPES         # 5 => NONE
PIPE_CLASSES = NUM_PIPES + 1      # 6

SIZE_TO_IDX = {"S": 0, "M": 1, "L": 2}
SIZE_NONE_IDX = 3
SIZE_CLASSES = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

def get_leaks_list(labels: dict):
    if "Leaks" in labels and isinstance(labels["Leaks"], list):
        return labels["Leaks"]
    if "leaks" in labels and isinstance(labels["leaks"], list):
        return labels["leaks"]
    return []

def get_leak_count(labels: dict):
    return len(get_leaks_list(labels))

def is_no_leak(labels: dict):
    return get_leak_count(labels) == 0

def encode_labels_from_json(labels: dict):
    leaks = get_leaks_list(labels)
    leak_count = int(len(leaks))

    pipe_targets = [PIPE_NONE_IDX] * MAX_LEAKS
    pos_targets  = [0.0] * MAX_LEAKS
    size_targets = [SIZE_NONE_IDX] * MAX_LEAKS
    slot_mask    = [0.0] * MAX_LEAKS

    leaks_sorted = sorted(leaks, key=lambda x: int(x.get("pipe_id", 999)))

    for i, lk in enumerate(leaks_sorted[:MAX_LEAKS]):
        pipe_id = int(lk.get("pipe_id", 1))          # 1..5
        pos = float(lk.get("position", 0.0))         # 0..1
        size_level = str(lk.get("size_level", "S")).upper()

        pipe_targets[i] = pipe_id - 1
        pos_targets[i]  = pos
        size_targets[i] = SIZE_TO_IDX.get(size_level, 0)
        slot_mask[i]    = 1.0

    return leak_count, pipe_targets, pos_targets, size_targets, slot_mask

# ============================================================
# FILE / SCENARIO HELPERS
# ============================================================
def list_valid_scenario_folders(root: str):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    folders = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        sig = p / "signals.csv"
        lab = p / "labels.json"
        if sig.exists() and lab.exists():
            folders.append(str(p))
    folders = sorted(folders)
    return folders

def get_stratify_labels(folders):
    y = []
    for folder in folders:
        labels = load_labels(os.path.join(folder, "labels.json"))
        y.append(get_leak_count(labels))   # stratify by 0/1/2/3 leak count
    return y

def read_signals(folder: str):
    df = pd.read_csv(os.path.join(folder, "signals.csv"))
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
    cropped = [a[:min_len] for a in no_leak_arrays]
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
    T = raw_signals.shape[0]
    Tb = baseline_template.shape[0]

    if Tb < T:
        raise ValueError(
            f"Baseline template shorter than scenario length: baseline={Tb}, scenario={T}"
        )

    base = baseline_template[:T]  # (T, N)
    dev = raw_signals - base

    feats = np.stack([raw_signals, dev], axis=-1).astype(np.float32)  # (T, N, 2)
    return feats

def compute_mu_sigma(train_folders, baseline_template):
    """
    Streaming mean/std over train data only.
    Stats are computed per node-feature channel => shape (N, 2)
    """
    sum_x = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    sum_x2 = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    total_T = 0

    for folder in train_folders:
        raw = read_signals(folder)
        feats = make_node_features(raw, baseline_template)  # (T, N, 2)

        sum_x += feats.sum(axis=0)
        sum_x2 += (feats ** 2).sum(axis=0)
        total_T += feats.shape[0]

    mu = sum_x / total_T
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
    """
    idx = {name: i for i, name in enumerate(SENSOR_NAMES)}
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    def connect(a, b, w=1.0):
        i, j = idx[a], idx[b]
        A[i, j] = w
        A[j, i] = w

    # True physical topology
    connect("Q1a", "P2")
    connect("P2", "Q2a")
    connect("Q2a", "P3")
    connect("P3", "Q3a")
    connect("Q3a", "P4")
    connect("P4", "Q4a")
    connect("Q4a", "P5")
    connect("P4", "Q5a")
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
    def __init__(self, folders, baseline_template, mu, sigma, window=180, stride=10):
        self.window = window
        self.stride = stride

        self.features = []     # each entry: (T, N, 2)
        self.targets = []      # each entry: tuple(labels...)
        self.names = []
        self.window_counts = []
        self.cum_counts = [0]

        for folder in folders:
            raw = read_signals(folder)
            feats = make_node_features(raw, baseline_template)
            feats = (feats - mu[None, :, :]) / (sigma[None, :, :] + 1e-8)
            feats = feats.astype(np.float32)

            labels = load_labels(os.path.join(folder, "labels.json"))
            tgt = encode_labels_from_json(labels)

            T = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            if n_windows == 0:
                continue

            self.features.append(feats)
            self.targets.append(tgt)
            self.names.append(os.path.basename(folder))
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        # ------------------------------------------------------------------
        # Scenario-level oversampling for minority classes.
        #
        # The window-level class distribution is dominated by classes 2 and 3
        # because there are far more leak scenarios than no-leak scenarios.
        # Class weighting in the loss cannot fully compensate when the ratio
        # is as extreme as 1:340 (class 0 vs class 3 windows).
        #
        # Strategy: count scenarios per class, find the maximum, then repeat
        # each minority-class scenario enough times so that every class has
        # roughly the same number of scenarios before window expansion.
        # This is scenario-level oversampling — each repeated scenario
        # contributes the same windows as the original, so no new data is
        # fabricated, but the gradient signal from rare classes is amplified.
        # ------------------------------------------------------------------
        leak_counts_per_scenario = [t[0] for t in self.targets]
        class_scenario_counts = Counter(leak_counts_per_scenario)
        max_scenarios = max(class_scenario_counts.values())

        print(f"  Scenario counts before oversampling: {dict(sorted(class_scenario_counts.items()))}")

        new_features = []
        new_targets  = []
        new_names    = []

        for i, lc in enumerate(leak_counts_per_scenario):
            # How many times to repeat this scenario so its class reaches
            # approximately max_scenarios total entries.
            count_for_class = class_scenario_counts[lc]
            repeats = max(1, round(max_scenarios / count_for_class))
            new_features.extend([self.features[i]] * repeats)
            new_targets.extend([self.targets[i]]   * repeats)
            new_names.extend([self.names[i]]        * repeats)

        self.features = new_features
        self.targets  = new_targets
        self.names    = new_names

        # Rebuild window counts and cumulative index from scratch after oversampling
        self.window_counts = []
        self.cum_counts    = [0]
        for feats in self.features:
            T = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        new_lc = [t[0] for t in self.targets]
        print(f"  Scenario counts after  oversampling: {dict(sorted(Counter(new_lc).items()))}")

    def __len__(self):
        return self.cum_counts[-1]

    def _locate_index(self, idx):
        scenario_idx = bisect.bisect_right(self.cum_counts, idx) - 1
        local_idx = idx - self.cum_counts[scenario_idx]
        return scenario_idx, local_idx

    def __getitem__(self, idx):
        scenario_idx, local_idx = self._locate_index(idx)
        x_full = self.features[scenario_idx]
        start = local_idx * self.stride
        end = start + self.window
        x = x_full[start:end]   # (W, N, 2)

        leak_count, pipe_t, pos_t, size_t, slot_mask = self.targets[scenario_idx]

        return (
            torch.tensor(x, dtype=torch.float32),                  # (W, N, 2)
            torch.tensor(leak_count, dtype=torch.long),
            torch.tensor(pipe_t, dtype=torch.long),
            torch.tensor(pos_t, dtype=torch.float32),
            torch.tensor(size_t, dtype=torch.long),
            torch.tensor(slot_mask, dtype=torch.float32),
        )

# ============================================================
# MODEL
# ============================================================
class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        # Symmetric padding preserves temporal length for any dilation value.
        # Formula: dilation * (kernel_size - 1) // 2
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            dilation=(1, dilation)   # dilation applied along time axis only
        )
        self.bn = nn.BatchNorm2d(out_ch)
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
        self.ln = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, N, C)
        x = torch.einsum("ij,btjc->btic", self.A, x)  # graph propagation over node dim
        x = self.lin(x)
        x = self.ln(x)
        x = self.act(x)
        return x

class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph = GraphConvLayer(out_ch, out_ch, adj_matrix)
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

class MultiTaskSTGCN(nn.Module):
    def __init__(self, adj_matrix):
        super().__init__()

        # Three STBlocks with exponentially increasing dilation, matching the
        # receptive field of the baseline TCN (dilations 1, 2, 4).
        # Each block applies temporal convolution then graph propagation,
        # so spatial context is built up alongside temporal context.
        self.block1 = STBlock(
            in_ch=NODE_FEATS,
            out_ch=HIDDEN_1,
            adj_matrix=adj_matrix,
            kernel_size=KERNEL_SIZE,
            dilation=1,
            dropout=DROPOUT
        )

        self.block2 = STBlock(
            in_ch=HIDDEN_1,
            out_ch=HIDDEN_2,
            adj_matrix=adj_matrix,
            kernel_size=KERNEL_SIZE,
            dilation=2,
            dropout=DROPOUT
        )

        self.block3 = STBlock(
            in_ch=HIDDEN_2,
            out_ch=HIDDEN_2,
            adj_matrix=adj_matrix,
            kernel_size=KERNEL_SIZE,
            dilation=4,
            dropout=DROPOUT
        )

        # Pool over time only, retaining the node dimension.
        # This preserves the spatial address of each sensor's learned
        # representation, which is critical at lower sensor budgets where
        # different nodes carry very different amounts of information.
        # The flattened vector is (N * HIDDEN_2) = 10 * 32 = 320.
        head_in = NUM_NODES * HIDDEN_2   # 320
        head_hidden = 64

        self.count_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, 4),
        )

        self.pipe_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, MAX_LEAKS * PIPE_CLASSES),
        )

        self.size_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, MAX_LEAKS * SIZE_CLASSES),
        )

        self.pos_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, MAX_LEAKS),
            nn.Sigmoid(),   # keep positions in [0,1]
        )

    def forward(self, x):
        # x: (B, T, N, 2)
        x = self.block1(x)   # (B, T, N, HIDDEN_1)
        x = self.block2(x)   # (B, T, N, HIDDEN_2)
        x = self.block3(x)   # (B, T, N, HIDDEN_2)

        # Pool over time only — each node retains its own feature vector.
        # Shape after pool: (B, N, HIDDEN_2)
        z = x.mean(dim=1)

        # Flatten node dimension into the feature vector.
        # Shape after flatten: (B, N * HIDDEN_2) = (B, 320)
        z = z.reshape(z.size(0), -1)

        count_logits = self.count_head(z)                           # (B, 4)
        pipe_logits  = self.pipe_head(z).view(-1, MAX_LEAKS, PIPE_CLASSES)
        size_logits  = self.size_head(z).view(-1, MAX_LEAKS, SIZE_CLASSES)
        pos_pred     = self.pos_head(z).view(-1, MAX_LEAKS)

        return count_logits, pipe_logits, size_logits, pos_pred

# ============================================================
# EVAL HELPERS
# ============================================================
@torch.no_grad()
def scenario_level_count_accuracy(model, dataset, batch_size=256):
    model.eval()

    true_counts = []
    pred_counts = []

    for i in range(len(dataset.features)):
        x_full = dataset.features[i]  # (T, N, 2)
        leak_count, _, _, _, _ = dataset.targets[i]

        T = x_full.shape[0]
        n_windows = max(0, (T - dataset.window) // dataset.stride + 1)
        if n_windows == 0:
            continue

        windows = []
        for w in range(n_windows):
            s = w * dataset.stride
            e = s + dataset.window
            windows.append(x_full[s:e])
        windows = np.stack(windows, axis=0).astype(np.float32)  # (Wn, W, N, 2)

        batch_preds = []
        for start in range(0, windows.shape[0], batch_size):
            xb = torch.tensor(windows[start:start+batch_size], dtype=torch.float32, device=DEVICE)
            count_logits, _, _, _ = model(xb)
            batch_preds.extend(count_logits.argmax(dim=1).cpu().numpy().tolist())

        # Majority vote
        vals, counts = np.unique(np.array(batch_preds), return_counts=True)
        final_pred = int(vals[np.argmax(counts)])

        true_counts.append(leak_count)
        pred_counts.append(final_pred)

    return accuracy_score(true_counts, pred_counts)

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: NO GPU")

    folders = list_valid_scenario_folders(DATASET_ROOT)
    if len(folders) == 0:
        raise RuntimeError(f"No valid scenario folders found in {DATASET_ROOT}")

    # ------------------------------
    # Stratified split by leak count
    # ------------------------------
    y = get_stratify_labels(folders)
    try:
        train_folders, test_folders = train_test_split(
            folders,
            test_size=0.25,
            random_state=SEED,
            stratify=y
        )
    except Exception:
        # fallback if stratify on 0/1/2/3 somehow fails
        y_binary = [0 if k == 0 else 1 for k in y]
        train_folders, test_folders = train_test_split(
            folders,
            test_size=0.25,
            random_state=SEED,
            stratify=y_binary
        )

    def count_summary(fs):
        cnt = Counter(get_stratify_labels(fs))
        return cnt

    train_counts = count_summary(train_folders)
    test_counts = count_summary(test_folders)

    print(f"Stratified split - train: {len(train_folders)} folders {dict(train_counts)}")
    print(f"Stratified split - test:  {len(test_folders)} folders {dict(test_counts)}")

    # ------------------------------
    # Build baseline + norm stats from TRAIN ONLY
    # ------------------------------
    baseline_template = compute_baseline_template(train_folders)
    mu, sigma = compute_mu_sigma(train_folders, baseline_template)

    # ------------------------------
    # Build datasets
    # ------------------------------
    train_ds = ScenarioWindowDataset(
        train_folders, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )
    test_ds = ScenarioWindowDataset(
        test_folders, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )

    print(f"Train windows: {len(train_ds)}")
    print(f"Test windows:  {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ------------------------------
    # Model
    # ------------------------------
    adj = build_sensor_adjacency()
    model = MultiTaskSTGCN(adj).to(DEVICE)

    # Count class weights computed at window level, not scenario level.
    # Each scenario contributes n_windows windows to the training pool, so
    # scenario-level counts underestimate how skewed the window distribution is.
    window_class_counts = Counter()
    for i, tgt in enumerate(train_ds.targets):
        leak_count = tgt[0]
        n_windows = train_ds.window_counts[i]
        window_class_counts[leak_count] += n_windows

    print(f"Window-level class counts (train): {dict(sorted(window_class_counts.items()))}")

    class_weights = []
    for c in range(4):
        class_weights.append(1.0 / max(window_class_counts.get(c, 1), 1))
    class_weights = np.array(class_weights, dtype=np.float32)
    class_weights = class_weights / class_weights.sum() * 4.0
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    # Loss functions.
    # Pipe and size use reduction="none" so the slot mask can be applied
    # before averaging, excluding empty slots from gradient computation.
    loss_count = nn.CrossEntropyLoss(weight=class_weights)
    loss_pipe  = nn.CrossEntropyLoss(reduction="none")
    loss_size  = nn.CrossEntropyLoss(reduction="none")
    loss_pos   = nn.SmoothL1Loss(reduction="none")

    # Loss weights. The count head is the primary task and receives a
    # higher weight. Position regression is down-weighted as SmoothL1 on
    # the [0,1] range produces small but noisy gradients early in training.
    LAMBDA_COUNT = 2.0
    LAMBDA_PIPE  = 1.0
    LAMBDA_SIZE  = 1.0
    LAMBDA_POS   = 0.5

    # ------------------------------
    # Train
    # ------------------------------
    for ep in range(EPOCHS):
        model.train()
        running = 0.0

        for x, leak_count, pipe_t, pos_t, size_t, slot_mask in train_loader:
            x = x.to(DEVICE, non_blocking=True)                # (B, W, N, 2)
            leak_count = leak_count.to(DEVICE, non_blocking=True)
            pipe_t = pipe_t.to(DEVICE, non_blocking=True)
            pos_t = pos_t.to(DEVICE, non_blocking=True)
            size_t = size_t.to(DEVICE, non_blocking=True)
            slot_mask = slot_mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                count_logits, pipe_logits, size_logits, pos_pred = model(x)

                Lc = loss_count(count_logits, leak_count)

                # Pipe loss: mask empty slots before averaging so that
                # NONE-class slots do not contribute gradients.
                mask_flat = slot_mask.reshape(-1)                          # (B*3,)
                pipe_loss_raw = loss_pipe(
                    pipe_logits.reshape(-1, PIPE_CLASSES),
                    pipe_t.reshape(-1)
                )                                                          # (B*3,)
                Lp = (pipe_loss_raw * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

                # Size loss: same masking pattern as pipe loss.
                size_loss_raw = loss_size(
                    size_logits.reshape(-1, SIZE_CLASSES),
                    size_t.reshape(-1)
                )                                                          # (B*3,)
                Ls = (size_loss_raw * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)

                # Position loss: already masked; unchanged.
                pos_err = loss_pos(pos_pred, pos_t)                        # (B, 3)
                Lr = (pos_err * slot_mask).sum() / slot_mask.sum().clamp(min=1.0)

                loss = LAMBDA_COUNT * Lc + LAMBDA_PIPE * Lp + LAMBDA_SIZE * Ls + LAMBDA_POS * Lr

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())

        print(f"Epoch {ep+1}/{EPOCHS}, loss={running/len(train_loader):.4f}")

    # ------------------------------
    # Save bundle
    # ------------------------------
    bundle = {
        "model_type": "stgcn_v3",
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
        "max_leaks": MAX_LEAKS,
        "pipe_classes": PIPE_CLASSES,
        "size_classes": SIZE_CLASSES,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
    }

    torch.save(bundle, "/content/multileak_stgcn_bundle_v3.pt")
    print("[OK] Saved /content/multileak_stgcn_bundle_v3.pt")

    drive_dir = os.path.dirname(DRIVE_SAVE_PATH)
    os.makedirs(drive_dir, exist_ok=True)
    torch.save(bundle, DRIVE_SAVE_PATH)
    print(f"[OK] Saved model to {DRIVE_SAVE_PATH}")

    # ------------------------------
    # Basic scenario-level count evaluation
    # ------------------------------
    test_count_acc = scenario_level_count_accuracy(model, test_ds, batch_size=256)
    print("\n=== SCENARIO-LEVEL LEAK COUNT ===")
    print(f"Accuracy: {test_count_acc:.4f}")

if __name__ == "__main__":
    main()