"""
train_stgcn.py
==============
ST-GCN training script for single-leak detection and localisation.
Designed to run in Google Colab with data on Google Drive.

Network:     5-pipe WDN with sensors Q1a, Q3a, P4
Tasks:       (1) Detection    - binary classification (leak / no-leak)
             (2) Pipe ID      - 6-class classification (pipes 1-5 + no-leak class)
             (3) Position     - regression (normalised position 0.0-1.0)

Usage:
    Copy to /content/train.py and run:
        !python /content/train.py
"""

import os
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

DRIVE_ZIP_PATH    = "/content/drive/MyDrive/colab_upload/stgcn_dataset.zip"
LOCAL_DATASET_DIR = "/content/stgcn_dataset"
MODEL_SAVE_PATH   = "/content/drive/MyDrive/colab_upload/stgcn_model.pt"

ACTIVE_SENSORS = ["Q1a", "Q3a", "P4"]

BATCH_SIZE   = 32
EPOCHS       = 100
LR           = 0.001
SEED         = 42
PATIENCE_ES  = 20
PATIENCE_LR  = 10
LR_FACTOR    = 0.5

W_DETECTION  = 1.0
W_PIPE       = 1.0
W_POSITION   = 0.5

# -------------------------------------------------------
# DERIVED CONSTANTS
# -------------------------------------------------------
N_SENSORS   = len(ACTIVE_SENSORS)
N_TIMESTEPS = 12
N_PIPES     = 5
PIPE_NONE   = 5
SIZE_MAP    = {"S": 0, "M": 1, "L": 2}

# Pipe endpoint sensor indices: Q1a=0, Q3a=1, P4=2
PIPE_SENSOR_MAP = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 2),
    3: (2, 0),
    4: (2, 1),
}

# Physical adjacency matrix for 3 sensors
INIT_ADJ = torch.tensor([
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.],
], dtype=torch.float32)

# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# UNZIP DATASET
# ============================================================

def unzip_dataset(zip_path, extract_to):
    if Path(extract_to).exists():
        print(f"Dataset already extracted at {extract_to}. Skipping unzip.")
        return
    print(f"Unzipping {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Unzip complete.")

unzip_dataset(DRIVE_ZIP_PATH, LOCAL_DATASET_DIR)

SCENARIOS_DIR = Path(LOCAL_DATASET_DIR) / "stgcn_dataset" / "scenarios"
if not SCENARIOS_DIR.exists():
    SCENARIOS_DIR = Path(LOCAL_DATASET_DIR) / "scenarios"

if not SCENARIOS_DIR.exists():
    raise FileNotFoundError(f"Cannot find scenarios directory under {LOCAL_DATASET_DIR}")

print(f"Scenarios directory: {SCENARIOS_DIR}")

# ============================================================
# DATASET
# ============================================================

def collect_scenario_dirs(scenarios_dir):
    dirs = []
    for folder in sorted(scenarios_dir.iterdir()):
        if not folder.is_dir():
            continue
        if (folder / "data.csv").exists() and (folder / "labels.json").exists():
            dirs.append(folder)
    print(f"Found {len(dirs)} valid scenario folders.")
    return dirs


def compute_normalisation(scenario_dirs):
    all_data = []
    for folder in scenario_dirs:
        df = pd.read_csv(folder / "data.csv")
        all_data.append(df[ACTIVE_SENSORS].values.astype(np.float32))
    all_data = np.vstack(all_data)
    return all_data.mean(axis=0), all_data.std(axis=0)


def split_scenarios(scenario_dirs):
    labels = []
    for folder in scenario_dirs:
        with open(folder / "labels.json", "r") as f:
            lab = json.load(f)
        labels.append(lab["label_detection"])
    labels = np.array(labels)
    idx = np.arange(len(scenario_dirs))
    idx_train, idx_temp, _, labels_temp = train_test_split(
        idx, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, stratify=labels_temp, random_state=SEED
    )
    train_dirs = [scenario_dirs[i] for i in idx_train]
    val_dirs   = [scenario_dirs[i] for i in idx_val]
    test_dirs  = [scenario_dirs[i] for i in idx_test]
    print(f"Split - Train: {len(train_dirs)}, Val: {len(val_dirs)}, Test: {len(test_dirs)}")
    return train_dirs, val_dirs, test_dirs


class LeakDataset(Dataset):
    def __init__(self, scenario_dirs, mu, sigma):
        self.scenarios = scenario_dirs
        self.mu        = mu.astype(np.float32)
        self.sigma     = sigma.astype(np.float32)

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        folder = self.scenarios[idx]
        df     = pd.read_csv(folder / "data.csv")
        data   = df[ACTIVE_SENSORS].values.astype(np.float32)   # [12, 3]
        data   = (data - self.mu) / (self.sigma + 1e-8)
        data   = torch.tensor(data.T, dtype=torch.float32)       # [3, 12]

        with open(folder / "labels.json", "r") as f:
            labels = json.load(f)

        label_detection = int(labels["label_detection"])
        raw_pipe        = labels["label_pipe"]
        label_pipe      = PIPE_NONE if raw_pipe == -1 else int(raw_pipe) - 1
        label_position  = float(labels["label_position"])
        raw_size        = labels["label_size"]
        label_size      = -1 if (raw_size == "none" or raw_size == -1) else SIZE_MAP[str(raw_size)]

        return {
            "data":            data,
            "label_detection": torch.tensor(label_detection, dtype=torch.long),
            "label_pipe":      torch.tensor(label_pipe,      dtype=torch.long),
            "label_position":  torch.tensor(label_position,  dtype=torch.float32),
            "label_size":      torch.tensor(label_size,      dtype=torch.long),
        }

# ============================================================
# MODEL
# ============================================================

class GraphConvLayer(nn.Module):
    """
    Graph convolution with learnable adjacency matrix.
    Input/Output: [B, N, F]
    """
    def __init__(self, n_nodes, f_in, f_out, init_adj):
        super().__init__()
        self.adj    = nn.Parameter(init_adj.clone().float())
        self.linear = nn.Linear(f_in, f_out)
        self.bn     = nn.BatchNorm1d(f_out)

    def forward(self, x):
        # x: [B, N, F_in]
        adj_norm = torch.softmax(self.adj, dim=-1)              # [N, N]
        agg      = torch.einsum('ij,bjk->bik', adj_norm, x)    # [B, N, F_in]
        out      = self.linear(agg)                             # [B, N, F_out]
        B, N, Fo = out.shape
        out      = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)  # [B, N, F_out]
        return torch.relu(out)


class TemporalConvLayer(nn.Module):
    """
    Temporal convolution per node.
    Input/Output: [B, N, T]
    """
    def __init__(self, n_nodes, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_nodes, out_channels=n_nodes,
            kernel_size=kernel_size, padding=kernel_size // 2,
            groups=n_nodes
        )
        self.bn = nn.BatchNorm1d(n_nodes)

    def forward(self, x):
        # x: [B, N, T]
        out = self.conv(x)
        out = self.bn(out)
        return torch.relu(out)


class STGCNBackbone(nn.Module):
    """
    Alternating spatial (GCN) and temporal (TCN) layers.
    Input:  [B, N, T]
    Output: [B, N, 64]
    """
    def __init__(self, n_nodes, n_timesteps, init_adj):
        super().__init__()
        H1, H2 = 32, 64
        self.gcn1     = GraphConvLayer(n_nodes, f_in=n_timesteps, f_out=H1, init_adj=init_adj)
        self.tcn1     = TemporalConvLayer(n_nodes=n_nodes, kernel_size=3)
        self.proj1    = nn.Linear(H1, n_timesteps)
        self.gcn2     = GraphConvLayer(n_nodes, f_in=H1, f_out=H2, init_adj=init_adj)
        self.tcn2     = TemporalConvLayer(n_nodes=n_nodes, kernel_size=3)
        self.proj2    = nn.Linear(H2, n_timesteps)
        self.proj_out = nn.Linear(n_timesteps, H2)

    def forward(self, x):
        # x: [B, N, T]
        h    = self.gcn1(x)               # [B, N, H1]
        h_t  = self.tcn1(self.proj1(h))   # [B, N, T]
        h    = self.gcn2(h)               # [B, N, H2]
        h_t2 = self.tcn2(self.proj2(h))   # [B, N, T]
        return h + self.proj_out(h_t2)    # [B, N, H2]


class SegmentFusion(nn.Module):
    """
    Concatenates endpoint sensor features for each pipe.
    Input:  [B, N_sensors, H]
    Output: [B, N_pipes, F_pipe]
    """
    def __init__(self, h_in, f_pipe):
        super().__init__()
        self.projection = nn.Linear(h_in * 2, f_pipe)
        self.bn         = nn.LayerNorm(f_pipe)

    def forward(self, node_features):
        # node_features: [B, N_sensors, H]
        pipe_feats = []
        for pipe_idx, (idx_a, idx_b) in PIPE_SENSOR_MAP.items():
            feat_a = node_features[:, idx_a, :]
            feat_b = node_features[:, idx_b, :]
            fused  = torch.cat([feat_a, feat_b], dim=-1)
            fused  = self.projection(fused)
            pipe_feats.append(fused)
        pipe_feats = torch.stack(pipe_feats, dim=1)   # [B, N_pipes, f_pipe]
        pipe_feats = self.bn(pipe_feats)
        return torch.relu(pipe_feats)


class STGCNLeakModel(nn.Module):
    def __init__(self, n_nodes, n_timesteps, init_adj):
        super().__init__()
        H2     = 64
        F_PIPE = 64
        self.backbone = STGCNBackbone(n_nodes, n_timesteps, init_adj)
        self.fusion   = SegmentFusion(h_in=H2, f_pipe=F_PIPE)
        self.detection_head = nn.Sequential(
            nn.Linear(F_PIPE, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2)
        )
        self.pipe_head = nn.Sequential(
            nn.Linear(N_PIPES * F_PIPE, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, N_PIPES + 1)
        )
        self.position_head = nn.Sequential(
            nn.Linear(N_PIPES * F_PIPE, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, N_sensors, T]
        node_feat   = self.backbone(x)                              # [B, N, 64]
        pipe_feat   = self.fusion(node_feat)                        # [B, N_pipes, 64]
        global_feat = pipe_feat.mean(dim=1)                         # [B, 64]
        pipe_flat   = pipe_feat.reshape(pipe_feat.size(0), -1)      # [B, N_pipes*64]
        return (
            self.detection_head(global_feat),
            self.pipe_head(pipe_flat),
            self.position_head(pipe_flat)
        )

# ============================================================
# LOSS
# ============================================================

def compute_loss(det_logits, pipe_logits, pos_pred,
                 label_detection, label_pipe, label_position):
    l_det     = F.cross_entropy(det_logits, label_detection)
    leak_mask = (label_detection == 1)
    if leak_mask.sum() > 0:
        l_pipe = F.cross_entropy(pipe_logits[leak_mask], label_pipe[leak_mask])
        l_pos  = F.mse_loss(pos_pred[leak_mask], label_position[leak_mask].unsqueeze(1))
    else:
        l_pipe = torch.tensor(0.0, device=det_logits.device)
        l_pos  = torch.tensor(0.0, device=det_logits.device)
    total = W_DETECTION * l_det + W_PIPE * l_pipe + W_POSITION * l_pos
    return total, l_det, l_pipe, l_pos

# ============================================================
# TRAIN / EVALUATE
# ============================================================

def train_one_epoch(model, loader, optimiser, device):
    model.train()
    t = d = p = po = 0.0
    for batch in loader:
        x        = batch["data"].to(device)
        lbl_det  = batch["label_detection"].to(device)
        lbl_pipe = batch["label_pipe"].to(device)
        lbl_pos  = batch["label_position"].to(device)
        optimiser.zero_grad()
        dl, pl, pp = model(x)
        loss, ld, lp, lpo = compute_loss(dl, pl, pp, lbl_det, lbl_pipe, lbl_pos)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        t += loss.item(); d += ld.item(); p += lp.item(); po += lpo.item()
    n = len(loader)
    return t/n, d/n, p/n, po/n


def evaluate(model, loader, device):
    model.eval()
    t = d = p = po = 0.0
    det_true=[]; det_pred=[]; pipe_true=[]; pipe_pred=[]; pos_true=[]; pos_pred=[]
    with torch.no_grad():
        for batch in loader:
            x        = batch["data"].to(device)
            lbl_det  = batch["label_detection"].to(device)
            lbl_pipe = batch["label_pipe"].to(device)
            lbl_pos  = batch["label_position"].to(device)
            dl, pl, pp = model(x)
            loss, ld, lp, lpo = compute_loss(dl, pl, pp, lbl_det, lbl_pipe, lbl_pos)
            t += loss.item(); d += ld.item(); p += lp.item(); po += lpo.item()
            det_true.extend(lbl_det.cpu().numpy())
            det_pred.extend(dl.argmax(1).cpu().numpy())
            pipe_true.extend(lbl_pipe.cpu().numpy())
            pipe_pred.extend(pl.argmax(1).cpu().numpy())
            mask = (lbl_det == 1).cpu().numpy().astype(bool)
            if mask.sum() > 0:
                pos_true.extend(lbl_pos.cpu().numpy()[mask])
                pos_pred.extend(pp.squeeze(1).cpu().numpy()[mask])
    n = len(loader)

    # Pipe ID metrics computed on leak scenarios only (exclude no-leak class)
    det_arr       = np.array(det_true)
    pipe_true_arr = np.array(pipe_true)
    pipe_pred_arr = np.array(pipe_pred)
    leak_idx      = np.where(det_arr == 1)[0]

    if len(leak_idx) > 0:
        pipe_true_leak = pipe_true_arr[leak_idx]
        pipe_pred_leak = pipe_pred_arr[leak_idx]
        pipe_f1_leak   = f1_score(pipe_true_leak, pipe_pred_leak,
                                   average="macro", zero_division=0,
                                   labels=list(range(N_PIPES)))
        pipe_f1_per_class = f1_score(pipe_true_leak, pipe_pred_leak,
                                      average=None, zero_division=0,
                                      labels=list(range(N_PIPES)))
    else:
        pipe_f1_leak      = float("nan")
        pipe_f1_per_class = [float("nan")] * N_PIPES

    return {
        "total_loss":        t/n,
        "det_loss":          d/n,
        "pipe_loss":         p/n,
        "pos_loss":          po/n,
        "det_acc":           accuracy_score(det_true, det_pred),
        "det_f1":            f1_score(det_true, det_pred, average="macro", zero_division=0),
        "pipe_f1":           pipe_f1_leak,
        "pipe_f1_per_class": pipe_f1_per_class,
        "pos_mae":           mean_absolute_error(pos_true, pos_pred) if pos_true else float("nan"),
        "pos_rmse":          float(np.sqrt(mean_squared_error(pos_true, pos_pred))) if pos_true else float("nan"),
        "pos_r2":            float(r2_score(pos_true, pos_pred)) if pos_true else float("nan"),
    }

# ============================================================
# MAIN
# ============================================================

def main():
    all_dirs = collect_scenario_dirs(SCENARIOS_DIR)
    train_dirs, val_dirs, test_dirs = split_scenarios(all_dirs)

    print("Computing normalisation statistics...")
    mu, sigma = compute_normalisation(train_dirs)
    print(f"  Mean:  {mu}")
    print(f"  Sigma: {sigma}")

    train_loader = DataLoader(LeakDataset(train_dirs, mu, sigma),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(LeakDataset(val_dirs, mu, sigma),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(LeakDataset(test_dirs, mu, sigma),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model = STGCNLeakModel(
        n_nodes=N_SENSORS, n_timesteps=N_TIMESTEPS,
        init_adj=INIT_ADJ.to(DEVICE)
    ).to(DEVICE)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=LR_FACTOR, patience=PATIENCE_LR
    )

    best_val_loss     = float("inf")
    epochs_no_improve = 0
    best_state        = None

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")

    for epoch in range(1, EPOCHS + 1):
        tr = train_one_epoch(model, train_loader, optimiser, DEVICE)
        vm = evaluate(model, val_loader, DEVICE)

        print(f"Ep {epoch:03d} | Train Loss: {tr[0]:.4f} "
              f"(D:{tr[1]:.3f} P:{tr[2]:.3f} Pos:{tr[3]:.4f})")
        print(f"Ep {epoch:03d} | Val   Loss: {vm['total_loss']:.4f} "
              f"DetAcc:{vm['det_acc']:.3f} "
              f"PipeF1(leak only):{vm['pipe_f1']:.3f} "
              f"PosMAE:{vm['pos_mae']:.4f}")

        scheduler.step(vm["total_loss"])

        if vm["total_loss"] < best_val_loss:
            best_val_loss     = vm["total_loss"]
            epochs_no_improve = 0
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Best val loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"  No improvement ({epochs_no_improve}/{PATIENCE_ES})")

        if epochs_no_improve >= PATIENCE_ES:
            print(f"\nEarly stopping at epoch {epoch}.")
            break
        print()

    model.load_state_dict(best_state)
    tm = evaluate(model, test_loader, DEVICE)

    print("\n" + "=" * 70)
    print("Test Set Results")
    print("=" * 70)
    print(f"Detection - Accuracy: {tm['det_acc']:.4f}  F1: {tm['det_f1']:.4f}")
    print(f"\nPipe ID (leak scenarios only, excluding no-leak class):")
    print(f"  Macro F1: {tm['pipe_f1']:.4f}")
    pipe_names = ["Pipe 1", "Pipe 2", "Pipe 3", "Pipe 4", "Pipe 5"]
    for name, f1 in zip(pipe_names, tm['pipe_f1_per_class']):
        print(f"  {name}: F1 = {f1:.4f}")
    print(f"\nPosition (leak scenarios only):")
    print(f"  MAE: {tm['pos_mae']:.4f}  RMSE: {tm['pos_rmse']:.4f}  R2: {tm['pos_r2']:.4f}")

    Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "active_sensors":   ACTIVE_SENSORS,
        "pipe_sensor_map":  PIPE_SENSOR_MAP,
        "mu":               mu.tolist(),
        "sigma":            sigma.tolist(),
        "n_nodes":          N_SENSORS,
        "n_timesteps":      N_TIMESTEPS,
        "n_pipes":          N_PIPES,
        "test_metrics":     tm,
    }, MODEL_SAVE_PATH)

    print(f"\nModel saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()