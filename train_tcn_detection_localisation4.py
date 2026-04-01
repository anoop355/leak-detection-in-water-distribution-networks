import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


# ======================
# CONFIG
# ======================
DATASET_ROOT   = "stgcn_dataset_v2/scenarios"
MANIFEST_TRAIN = "stgcn_dataset_v2/manifests/manifest_train.csv"
MANIFEST_VAL   = "stgcn_dataset_v2/manifests/manifest_val.csv"
MANIFEST_TEST  = "stgcn_dataset_v2/manifests/manifest_test.csv"
FEATURE_COLS   = ["P2","P3","P4","P5","P6","Q1a","Q2a","Q3a","Q4a","Q5a"]

WINDOW = 12        # each scenario has exactly 12 timesteps (15-min intervals)
STRIDE = 1
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
SEED = 42

NUM_PIPES = 5

# Pipe classes: 0..4 = pipes 1..5, 5 = NONE
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES  = NUM_PIPES + 1

# Size classes: 0=S,1=M,2=L, 3=NONE
SIZE_TO_IDX  = {"S": 0, "M": 1, "L": 2}
SIZE_NONE_IDX = 3
SIZE_CLASSES  = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


# ======================
# DATA HELPERS
# ======================
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


def compute_mu_sigma(folders):
    all_data = []
    for f in folders:
        sig_path = os.path.join(f, "data.csv")
        df = pd.read_csv(sig_path)
        all_data.append(df[FEATURE_COLS].values)
    all_data = np.vstack(all_data).astype(np.float32)
    return all_data.mean(axis=0), all_data.std(axis=0)


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
        pipe_t = int(labels.get("label_pipe", 1)) - 1            # 1..5 -> 0..4
        pos_t  = float(labels.get("label_position", 0.0))
        sl     = str(labels.get("label_size", "S")).upper()
        size_t = SIZE_TO_IDX.get(sl, 0)
    else:
        pipe_t = PIPE_NONE_IDX
        pos_t  = 0.0
        size_t = SIZE_NONE_IDX

    return detect, pipe_t, pos_t, size_t


class LeakDataset(Dataset):
    def __init__(self, scenario_folders, mu=None, sigma=None):
        self.samples = []
        self.mu    = mu
        self.sigma = sigma

        for folder in scenario_folders:
            sig_path = os.path.join(folder, "data.csv")
            lab_path = os.path.join(folder, "labels.json")

            df = pd.read_csv(sig_path)
            X_full = df[FEATURE_COLS].values.astype(np.float32)

            with open(lab_path, "r") as f:
                labels = json.load(f)

            detect, pipe_t, pos_t, size_t = encode_labels_from_json(labels)

            # Sliding windows
            for end in range(WINDOW, len(X_full) + 1, STRIDE):
                Xw = X_full[end - WINDOW:end]
                self.samples.append((Xw, detect, pipe_t, pos_t, size_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, detect, pipe_t, pos_t, size_t = self.samples[idx]

        if self.mu is not None and self.sigma is not None:
            x = (x - self.mu) / (self.sigma + 1e-8)

        x = x.astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32).transpose(0, 1)  # (C,T)

        return (
            x,
            torch.tensor(detect, dtype=torch.long),
            torch.tensor(pipe_t, dtype=torch.long),
            torch.tensor(pos_t,  dtype=torch.float32),
            torch.tensor(size_t, dtype=torch.long),
        )


# ======================
# MODEL
# ======================
class LeakTCN(nn.Module):
    """
    Shared TCN backbone + single-leak output heads:
      - detect: 2 classes (0=no-leak, 1=leak)
      - pipe:   6 classes (5 pipes + NONE)
      - size:   4 classes (S/M/L + NONE)
      - pos:    scalar regression
    """
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

        self.detect_head = nn.Linear(32, 2)             # 0=no-leak, 1=leak
        self.pipe_head   = nn.Linear(32, PIPE_CLASSES)  # 6 classes
        self.size_head   = nn.Linear(32, SIZE_CLASSES)  # 4 classes
        self.pos_head    = nn.Linear(32, 1)             # scalar

    def forward(self, x):
        z = self.backbone(x)

        detect_logits = self.detect_head(z)
        pipe_logits   = self.pipe_head(z)
        size_logits   = self.size_head(z)
        pos_pred      = self.pos_head(z).squeeze(1)     # (B,)

        return detect_logits, pipe_logits, size_logits, pos_pred


# ======================
# TRAIN / EVAL
# ======================
train_f = folders_from_manifest(MANIFEST_TRAIN)
val_f   = folders_from_manifest(MANIFEST_VAL)
test_f  = folders_from_manifest(MANIFEST_TEST)

if len(train_f) == 0:
    raise FileNotFoundError(f"No valid scenario folders found via {MANIFEST_TRAIN}.")

mu, sigma = compute_mu_sigma(train_f)

train_ds = LeakDataset(train_f, mu, sigma)
val_ds   = LeakDataset(val_f,   mu, sigma)
test_ds  = LeakDataset(test_f,  mu, sigma)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeakTCN(len(FEATURE_COLS)).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)

loss_detect = nn.CrossEntropyLoss()
loss_pipe   = nn.CrossEntropyLoss()
loss_size   = nn.CrossEntropyLoss()
loss_pos    = nn.SmoothL1Loss(reduction="none")  # masked manually


for ep in range(EPOCHS):
    model.train()
    total = 0.0

    for X, detect, pipe_t, pos_t, size_t in train_loader:
        X      = X.to(device)
        detect = detect.to(device)     # (B,)
        pipe_t = pipe_t.to(device)     # (B,)
        pos_t  = pos_t.to(device)      # (B,)
        size_t = size_t.to(device)     # (B,)

        opt.zero_grad()

        detect_logits, pipe_logits, size_logits, pos_pred = model(X)

        # 1) detection loss
        Ld = loss_detect(detect_logits, detect)

        # 2) pipe loss (includes NONE class for no-leak scenarios)
        Lp = loss_pipe(pipe_logits, pipe_t)

        # 3) size loss (includes NONE class for no-leak scenarios)
        Ls = loss_size(size_logits, size_t)

        # 4) position loss only for real leak scenarios
        leak_mask = detect.float()                         # 1.0 where leak present
        pos_err   = loss_pos(pos_pred, pos_t)              # (B,)
        denom     = leak_mask.sum().clamp(min=1.0)
        Lr        = (pos_err * leak_mask).sum() / denom

        loss = Ld + Lp + Ls + Lr
        loss.backward()
        opt.step()

        total += float(loss.item())

    print(f"Epoch {ep+1}/{EPOCHS}, loss={total/len(train_loader):.4f}")


# ======================
# SAVE BUNDLE
# ======================
save_bundle = {
    "model_state_dict": model.state_dict(),
    "mu": mu,
    "sigma": sigma,
    "feature_cols": FEATURE_COLS,
    "window": WINDOW,
    "stride": STRIDE,
    "dataset_root": DATASET_ROOT,
    "pipe_classes": PIPE_CLASSES,
    "size_classes": SIZE_CLASSES,
}

torch.save(save_bundle, "leak_tcn_v2_bundle.pt")
print("[OK] Saved leak_tcn_v2_bundle.pt")


# ======================
# TEST METRICS
# ======================
model.eval()

detect_true, detect_pred = [], []
with torch.no_grad():
    for X, detect, pipe_t, pos_t, size_t in test_loader:
        X = X.to(device)
        detect_logits, pipe_logits, size_logits, pos_pred = model(X)

        pred_detect = detect_logits.argmax(dim=1).cpu().numpy().astype(int).tolist()
        true_detect = detect.numpy().astype(int).tolist()

        detect_pred.extend(pred_detect)
        detect_true.extend(true_detect)

print("\n=== DETECTION ===")
print("Accuracy:", accuracy_score(detect_true, detect_pred))
print(confusion_matrix(detect_true, detect_pred))


# Pipe / size / position on test set
pipe_true_all, pipe_pred_all = [], []
pos_true_all,  pos_pred_all  = [], []
size_true_all, size_pred_all = [], []
leak_mask_all = []

with torch.no_grad():
    for X, detect, pipe_t, pos_t, size_t in test_loader:
        X = X.to(device)
        _, pipe_logits, size_logits, pos_pred = model(X)

        pipe_pred_b = pipe_logits.argmax(dim=1).cpu().numpy()   # (B,)
        size_pred_b = size_logits.argmax(dim=1).cpu().numpy()   # (B,)

        pipe_true_all.append(pipe_t.numpy())
        pipe_pred_all.append(pipe_pred_b)

        size_true_all.append(size_t.numpy())
        size_pred_all.append(size_pred_b)

        pos_true_all.append(pos_t.numpy())
        pos_pred_all.append(pos_pred.cpu().numpy())

        leak_mask_all.append(detect.numpy())

pipe_true_all = np.concatenate(pipe_true_all)
pipe_pred_all = np.concatenate(pipe_pred_all)
size_true_all = np.concatenate(size_true_all)
size_pred_all = np.concatenate(size_pred_all)
pos_true_all  = np.concatenate(pos_true_all)
pos_pred_all  = np.concatenate(pos_pred_all)
leak_mask_all = np.concatenate(leak_mask_all)

print("\n=== PIPE (all scenarios, includes NONE) ===")
print("Accuracy:", accuracy_score(pipe_true_all, pipe_pred_all))
print(confusion_matrix(pipe_true_all, pipe_pred_all))

print("\n=== SIZE (all scenarios, includes NONE) ===")
print("Accuracy:", accuracy_score(size_true_all, size_pred_all))
print(confusion_matrix(size_true_all, size_pred_all))

# All remaining metrics use leak scenarios only
real_mask = leak_mask_all > 0.5

# Pipe F1 Macro — leak scenarios only, excluding NONE class
pipe_true_leak = pipe_true_all[real_mask]
pipe_pred_leak = pipe_pred_all[real_mask]
f1_pipe = f1_score(pipe_true_leak, pipe_pred_leak, average="macro", labels=list(range(NUM_PIPES)))
print(f"\n=== PIPE F1 MACRO (leak scenarios only) ===")
print(f"F1 Macro: {f1_pipe:.4f}")

# Position error only on real leak scenarios
pos_true_flat = pos_true_all[real_mask]
pos_pred_flat = pos_pred_all[real_mask]

if len(pos_true_flat) > 0:
    mae  = float(np.mean(np.abs(pos_true_flat - pos_pred_flat)))
    rmse = float(np.sqrt(np.mean((pos_true_flat - pos_pred_flat) ** 2)))
    print("\n=== POSITION (real leaks only) ===")
    print("MAE:", mae)
    print("RMSE:", rmse)
else:
    print("\n=== POSITION (real leaks only) ===")
    print("No real-leak scenarios in test set -> cannot compute.")
