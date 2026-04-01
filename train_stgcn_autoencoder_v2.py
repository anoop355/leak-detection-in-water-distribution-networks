"""
train_stgcn_autoencoder_v2.py

Upgraded ST-GCN autoencoder for reconstructing the full 10-sensor WDN state
from the 3 monitored sensors.

Key upgrades over v1
--------------------
- Explicit mask-aware inputs so missing nodes are not mistaken for zero values
- Richer node features: [raw, deviation_from_baseline, first_difference, mask]
- Weighted graph built from physical shortest-path proximity
- Graph-aware encoder-decoder with skip connections
- Random sensor-drop augmentation on monitored nodes during training
- Scheduler + early stopping using denormalised validation MAE
"""

import copy
import json
import math
import os
import random
from collections import deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT = "stgcn_dataset/scenarios"
MANIFEST_TRAIN = "stgcn_dataset/manifests/manifest_train.csv"
MANIFEST_VAL = "stgcn_dataset/manifests/manifest_val.csv"
MANIFEST_TEST = "stgcn_dataset/manifests/manifest_test.csv"
SAVE_PATH = "stgcn_autoencoder_v2.pt"

ALL_SENSORS = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
NUM_NODES = len(ALL_SENSORS)

MONITORED_IDX = [2, 5, 7]             # P4, Q1a, Q3a
UNMONITORED_IDX = [0, 1, 3, 4, 6, 8, 9]

RAW_FEATS = 3                         # [raw, deviation_from_baseline, first_difference]
MASK_FEATS = 1                        # [observed_mask]
NODE_FEATS = RAW_FEATS + MASK_FEATS   # total input channels
OUTPUT_FEATS = 1                      # reconstruct raw reading only

WINDOW = 12
STRIDE = 1
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
MIN_LR = 1e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.20
SEED = 42

HIDDEN_1 = 32
HIDDEN_2 = 64
KERNEL_SIZE = 5
PATIENCE_LR = 6
PATIENCE_EARLY_STOP = 14
MONITORED_DROP_PROB = 0.25
EDGE_DECAY = 0.75
LOSS_ALPHA_DENORM = 0.25
GRAD_CLIP_NORM = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = DEVICE == "cuda"

PHYSICAL_EDGES = [
    ("Q1a", "P2"),
    ("P2", "Q2a"),
    ("Q2a", "P3"),
    ("P3", "Q3a"),
    ("Q3a", "P4"),
    ("P4", "Q4a"),
    ("Q4a", "P5"),
    ("P4", "Q5a"),
    ("Q5a", "P6"),
]


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
        sig = os.path.join(path, "data.csv")
        lab = os.path.join(path, "labels.json")
        if os.path.isfile(sig) and os.path.isfile(lab):
            folders.append(path)
    return folders


def read_signals_all(folder: str) -> np.ndarray:
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    missing = [c for c in ALL_SENSORS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {folder}: {missing}")
    return df[ALL_SENSORS].to_numpy(dtype=np.float32)


def compute_baseline_template(train_folders):
    no_leak_arrays = []
    for folder in train_folders:
        labels = load_labels(os.path.join(folder, "labels.json"))
        if is_no_leak(labels):
            no_leak_arrays.append(read_signals_all(folder))

    if not no_leak_arrays:
        raise RuntimeError("No no-leak scenarios found in training split.")

    min_len = min(arr.shape[0] for arr in no_leak_arrays)
    cropped = [arr[:min_len] for arr in no_leak_arrays]
    return np.mean(np.stack(cropped, axis=0), axis=0).astype(np.float32)


def make_raw_features(raw_signals: np.ndarray, baseline_template: np.ndarray) -> np.ndarray:
    t_len = raw_signals.shape[0]
    base = baseline_template[:t_len]
    deviation = raw_signals - base
    first_diff = np.zeros_like(raw_signals)
    first_diff[1:] = raw_signals[1:] - raw_signals[:-1]
    return np.stack([raw_signals, deviation, first_diff], axis=-1).astype(np.float32)


def compute_mu_sigma(train_folders, baseline_template):
    sum_x = np.zeros((NUM_NODES, RAW_FEATS), dtype=np.float64)
    sum_x2 = np.zeros((NUM_NODES, RAW_FEATS), dtype=np.float64)
    total_t = 0

    for folder in train_folders:
        raw = read_signals_all(folder)
        feats = make_raw_features(raw, baseline_template)
        sum_x += feats.sum(axis=0)
        sum_x2 += (feats ** 2).sum(axis=0)
        total_t += feats.shape[0]

    mu = sum_x / total_t
    var = np.maximum((sum_x2 / total_t) - (mu ** 2), 1e-8)
    sigma = np.sqrt(var)
    return mu.astype(np.float32), sigma.astype(np.float32)


def build_sensor_adjacency(edge_decay: float = EDGE_DECAY) -> np.ndarray:
    idx = {name: i for i, name in enumerate(ALL_SENSORS)}
    neighbors = {name: [] for name in ALL_SENSORS}
    for a, b in PHYSICAL_EDGES:
        neighbors[a].append(b)
        neighbors[b].append(a)

    distances = np.full((NUM_NODES, NUM_NODES), np.inf, dtype=np.float32)
    for src in ALL_SENSORS:
        src_i = idx[src]
        distances[src_i, src_i] = 0.0
        queue = deque([src])
        seen = {src}
        while queue:
            cur = queue.popleft()
            cur_i = idx[cur]
            for nxt in neighbors[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    nxt_i = idx[nxt]
                    distances[src_i, nxt_i] = distances[src_i, cur_i] + 1.0
                    queue.append(nxt)

    weights = np.exp(-edge_decay * distances)
    weights[np.isinf(distances)] = 0.0
    np.fill_diagonal(weights, 1.0)

    deg = weights.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    d_inv_sqrt = np.diag(deg_inv_sqrt)
    a_hat = d_inv_sqrt @ weights @ d_inv_sqrt
    return a_hat.astype(np.float32)


# ============================================================
# DATASET
# ============================================================
class AutoencoderDatasetV2(Dataset):
    def __init__(self, folders, baseline_template, mu, sigma, window=12, stride=1):
        self.samples = []

        for folder in folders:
            raw = read_signals_all(folder)
            raw_feats = make_raw_features(raw, baseline_template)
            raw_feats = (raw_feats - mu[None]) / (sigma[None] + 1e-8)
            targets = raw_feats[..., :1].astype(np.float32)

            t_len = raw_feats.shape[0]
            n_windows = max(0, (t_len - window) // stride + 1)
            for w in range(n_windows):
                start = w * stride
                end = start + window
                self.samples.append(
                    (
                        raw_feats[start:end].astype(np.float32),
                        targets[start:end].astype(np.float32),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, target = self.samples[idx]
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
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
            dilation=(1, dilation),
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.act(self.bn(self.conv(x)))
        return x.permute(0, 3, 2, 1)


class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln = nn.LayerNorm(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = torch.einsum("ij,btjc->btic", self.A, x)
        x = self.lin(x)
        x = self.ln(x)
        return self.act(x)


class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.2):
        super().__init__()
        self.temp = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(out_ch)
        self.out_act = nn.GELU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        y = self.temp(x)
        y = self.graph(y)
        y = self.dropout(y)
        y = self.out_norm(y + residual)
        return self.out_act(y)


class STGCNAutoencoderV2(nn.Module):
    def __init__(self, adj_matrix):
        super().__init__()
        self.encoder1 = STBlock(NODE_FEATS, HIDDEN_1, adj_matrix, KERNEL_SIZE, 1, DROPOUT)
        self.encoder2 = STBlock(HIDDEN_1, HIDDEN_2, adj_matrix, KERNEL_SIZE, 2, DROPOUT)
        self.bottleneck = STBlock(HIDDEN_2, HIDDEN_2, adj_matrix, KERNEL_SIZE, 4, DROPOUT)

        self.decoder1 = STBlock(HIDDEN_2, HIDDEN_2, adj_matrix, KERNEL_SIZE, 2, DROPOUT)
        self.decoder2 = STBlock(HIDDEN_2 + HIDDEN_2, HIDDEN_1, adj_matrix, KERNEL_SIZE, 1, DROPOUT)
        self.decoder3 = STBlock(HIDDEN_1 + HIDDEN_1, HIDDEN_1, adj_matrix, KERNEL_SIZE, 1, DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_1, HIDDEN_1),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, OUTPUT_FEATS),
        )

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        z = self.bottleneck(skip2)

        y = self.decoder1(z)
        y = self.decoder2(torch.cat([y, skip2], dim=-1))
        y = self.decoder3(torch.cat([y, skip1], dim=-1))
        return self.head(y)


# ============================================================
# TRAIN / EVAL
# ============================================================
def build_masked_input(x_feats: torch.Tensor, random_drop: bool = False):
    x_masked = x_feats.clone()

    observed_mask = torch.zeros(
        x_feats.shape[0], x_feats.shape[1], x_feats.shape[2], 1,
        dtype=x_feats.dtype,
        device=x_feats.device,
    )
    observed_mask[:, :, MONITORED_IDX, :] = 1.0

    if random_drop:
        drop_mask = (
            torch.rand(
                x_feats.shape[0], x_feats.shape[1], len(MONITORED_IDX), 1,
                device=x_feats.device,
            ) < MONITORED_DROP_PROB
        )
        observed_mask[:, :, MONITORED_IDX, :] *= (~drop_mask).to(x_feats.dtype)

    # Zero out raw features for hidden nodes, then append mask as 4th channel.
    # (x_feats only has RAW_FEATS=3 channels; the mask channel must be cat'd.)
    x_masked = x_feats * observed_mask                          # broadcast: (B,T,N,3)*(B,T,N,1)
    x_masked = torch.cat([x_masked, observed_mask], dim=-1)    # (B, T, N, 4)
    return x_masked, observed_mask


def denormalise_raw(raw_norm: torch.Tensor, mu_raw: torch.Tensor, sigma_raw: torch.Tensor):
    return raw_norm * sigma_raw.view(1, 1, -1, 1) + mu_raw.view(1, 1, -1, 1)


def compute_losses(pred_raw_norm, target_raw_norm, mu_raw, sigma_raw):
    pred_unmon = pred_raw_norm[:, :, UNMONITORED_IDX, :]
    target_unmon = target_raw_norm[:, :, UNMONITORED_IDX, :]

    mse_norm = F.mse_loss(pred_unmon, target_unmon)

    pred_raw = denormalise_raw(pred_unmon, mu_raw[UNMONITORED_IDX], sigma_raw[UNMONITORED_IDX])
    true_raw = denormalise_raw(target_unmon, mu_raw[UNMONITORED_IDX], sigma_raw[UNMONITORED_IDX])
    mae_denorm = torch.mean(torch.abs(pred_raw - true_raw))

    loss = mse_norm + LOSS_ALPHA_DENORM * mae_denorm
    return loss, mse_norm.detach(), mae_denorm.detach()


def train_one_epoch(model, loader, optimizer, scaler, mu_raw, sigma_raw):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0

    for x_feats, target_raw_norm in loader:
        x_feats = x_feats.to(DEVICE, non_blocking=True)
        target_raw_norm = target_raw_norm.to(DEVICE, non_blocking=True)
        masked_input, _ = build_masked_input(x_feats, random_drop=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
            pred_raw_norm = model(masked_input)
            loss, mse_norm, mae_denorm = compute_losses(
                pred_raw_norm, target_raw_norm, mu_raw, sigma_raw
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        total_mse += float(mse_norm.item())
        total_mae += float(mae_denorm.item())

    num_batches = len(loader)
    return total_loss / num_batches, total_mse / num_batches, total_mae / num_batches


@torch.no_grad()
def evaluate(model, loader, mu_raw, sigma_raw):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0

    for x_feats, target_raw_norm in loader:
        x_feats = x_feats.to(DEVICE, non_blocking=True)
        target_raw_norm = target_raw_norm.to(DEVICE, non_blocking=True)
        masked_input, _ = build_masked_input(x_feats, random_drop=False)

        pred_raw_norm = model(masked_input)
        loss, mse_norm, mae_denorm = compute_losses(
            pred_raw_norm, target_raw_norm, mu_raw, sigma_raw
        )

        total_loss += float(loss.item())
        total_mse += float(mse_norm.item())
        total_mae += float(mae_denorm.item())

    num_batches = len(loader)
    return total_loss / num_batches, total_mse / num_batches, total_mae / num_batches


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_folders = folders_from_manifest(MANIFEST_TRAIN)
    val_folders = folders_from_manifest(MANIFEST_VAL)
    test_folders = folders_from_manifest(MANIFEST_TEST)

    if not train_folders:
        raise RuntimeError(f"No valid scenarios found via {MANIFEST_TRAIN}")

    print(f"Train: {len(train_folders)}  Val: {len(val_folders)}  Test: {len(test_folders)}")

    baseline_template = compute_baseline_template(train_folders)
    mu, sigma = compute_mu_sigma(train_folders, baseline_template)
    mu_raw = torch.tensor(mu[:, 0], dtype=torch.float32, device=DEVICE)
    sigma_raw = torch.tensor(sigma[:, 0], dtype=torch.float32, device=DEVICE)

    adj = build_sensor_adjacency()

    train_ds = AutoencoderDatasetV2(train_folders, baseline_template, mu, sigma, WINDOW, STRIDE)
    val_ds = AutoencoderDatasetV2(val_folders, baseline_template, mu, sigma, WINDOW, STRIDE)
    test_ds = AutoencoderDatasetV2(test_folders, baseline_template, mu, sigma, WINDOW, STRIDE)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("One or more dataset splits produced zero windows.")

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = STGCNAutoencoderV2(adj).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=PATIENCE_LR,
        min_lr=MIN_LR,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    print("\n--- Training v2 ---")
    best_val_mae = math.inf
    best_epoch = 0
    best_state = None
    epochs_without_improve = 0
    history = []

    for ep in range(EPOCHS):
        train_loss, train_mse, train_mae = train_one_epoch(
            model, train_loader, optimizer, scaler, mu_raw, sigma_raw
        )
        val_loss, val_mse, val_mae = evaluate(model, val_loader, mu_raw, sigma_raw)
        scheduler.step(val_mae)

        improved = val_mae < (best_val_mae - 1e-6)
        if improved:
            best_val_mae = val_mae
            best_epoch = ep + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": ep + 1,
                "train_loss": train_loss,
                "train_mse_norm": train_mse,
                "train_mae_denorm": train_mae,
                "val_loss": val_loss,
                "val_mse_norm": val_mse,
                "val_mae_denorm": val_mae,
                "lr": lr_now,
            }
        )

        marker = "  *** best ***" if improved else ""
        print(
            f"Epoch {ep + 1:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}  "
            f"val_mae_denorm={val_mae:.6f}  "
            f"lr={lr_now:.6e}{marker}"
        )

        if epochs_without_improve >= PATIENCE_EARLY_STOP:
            print(f"Early stopping triggered at epoch {ep + 1}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_mse, test_mae = evaluate(model, test_loader, mu_raw, sigma_raw)

    history_df = pd.DataFrame(history)
    history_csv = os.path.splitext(SAVE_PATH)[0] + "_history.csv"
    history_df.to_csv(history_csv, index=False)

    bundle = {
        "model_type": "stgcn_autoencoder_v2",
        "model_state_dict": model.state_dict(),
        "adjacency": adj,
        "mu": mu,
        "sigma": sigma,
        "baseline_template": baseline_template,
        "sensor_names": ALL_SENSORS,
        "monitored_idx": MONITORED_IDX,
        "unmonitored_idx": UNMONITORED_IDX,
        "window": WINDOW,
        "stride": STRIDE,
        "raw_feats": RAW_FEATS,
        "mask_feats": MASK_FEATS,
        "node_feats": NODE_FEATS,
        "output_feats": OUTPUT_FEATS,
        "hidden_1": HIDDEN_1,
        "hidden_2": HIDDEN_2,
        "kernel_size": KERNEL_SIZE,
        "dropout": DROPOUT,
        "edge_decay": EDGE_DECAY,
        "monitored_drop_prob": MONITORED_DROP_PROB,
        "loss_alpha_denorm": LOSS_ALPHA_DENORM,
        "best_epoch": best_epoch,
        "best_val_mae_denorm": best_val_mae,
        "test_loss": test_loss,
        "test_mse_norm": test_mse,
        "test_mae_denorm": test_mae,
        "epochs_requested": EPOCHS,
        "seed": SEED,
    }
    torch.save(bundle, SAVE_PATH)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best val MAE (denormalised, unmonitored): {best_val_mae:.6f}")
    print(f"Test MSE (normalised, unmonitored): {test_mse:.6f}")
    print(f"Test MAE (denormalised, unmonitored): {test_mae:.6f}")
    print(f"[OK] Saved bundle -> {SAVE_PATH}")
    print(f"[OK] Saved history -> {history_csv}")


if __name__ == "__main__":
    main()
