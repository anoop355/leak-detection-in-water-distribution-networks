"""
Updates in this version:
- switches to the generated `training_cases_output` dataset
- adds leak-count, pipe, size, and position heads
- introduces fixed-slot target encoding for up to three simultaneous leaks
"""

import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score



# CONFIG

DATASET_ROOT = "training_cases_output"  
FEATURE_COLS = ["P2","P3","P4","P5","P6","Q1a","Q2a","Q3a","Q4a","Q5a"]

WINDOW = 180
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
SEED = 42

MAX_LEAKS = 3
NUM_PIPES = 5

# Pipe classes: 0..4 map to pipes 1..5, and 5 marks an unused slot.
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES = NUM_PIPES + 1

# Size classes: 0=S, 1=M, 2=L, and 3 marks an unused slot.
SIZE_TO_IDX = {"S": 0, "M": 1, "L": 2}
SIZE_NONE_IDX = 3
SIZE_CLASSES = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)



# DATA HELPERS

def list_valid_scenario_folders(root: str):
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset root '{root}' not found.")

    folders = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        sig_path = os.path.join(path, "signals.csv")
        lab_path = os.path.join(path, "labels.json")
        if os.path.isfile(sig_path) and os.path.isfile(lab_path):
            folders.append(path)

    return folders


def compute_mu_sigma(folders):
    all_data = []
    for f in folders:
        sig_path = os.path.join(f, "signals.csv")
        df = pd.read_csv(sig_path)
        all_data.append(df[FEATURE_COLS].values)
    all_data = np.vstack(all_data).astype(np.float32)
    return all_data.mean(axis=0), all_data.std(axis=0)


def encode_labels_from_json(labels: dict):

    leaks = labels.get("leaks", [])
    leak_count = int(len(leaks))

    # Initialize all slots as NONE
    pipe_targets = [PIPE_NONE_IDX] * MAX_LEAKS
    pos_targets  = [0.0] * MAX_LEAKS
    size_targets = [SIZE_NONE_IDX] * MAX_LEAKS
    slot_mask    = [0.0] * MAX_LEAKS


    leaks_sorted = sorted(leaks, key=lambda x: int(x.get("pipe_id", 999)))

    for i, lk in enumerate(leaks_sorted[:MAX_LEAKS]):
        pipe_id = int(lk.get("pipe_id", 1))          # 1..5
        pos = float(lk.get("position", 0.0))         # 0..1
        size_level = str(lk.get("size_level", "S")).upper()

        pipe_targets[i] = pipe_id - 1                # 0..4
        pos_targets[i]  = pos
        size_targets[i] = SIZE_TO_IDX.get(size_level, 0)
        slot_mask[i]    = 1.0

    return leak_count, pipe_targets, pos_targets, size_targets, slot_mask


class LeakDatasetMulti(Dataset):
    def __init__(self, scenario_folders, mu=None, sigma=None):
        self.samples = []
        self.mu = mu
        self.sigma = sigma

        for folder in scenario_folders:
            sig_path = os.path.join(folder, "signals.csv")
            lab_path = os.path.join(folder, "labels.json")

            df = pd.read_csv(sig_path)
            X_full = df[FEATURE_COLS].values.astype(np.float32)

            with open(lab_path, "r") as f:
                labels = json.load(f)

            leak_count, pipe_t, pos_t, size_t, slot_mask = encode_labels_from_json(labels)

            # Sliding windows
            for end in range(WINDOW, len(X_full) + 1, STRIDE):
                Xw = X_full[end - WINDOW:end]
                self.samples.append((Xw, leak_count, pipe_t, pos_t, size_t, slot_mask))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, leak_count, pipe_t, pos_t, size_t, slot_mask = self.samples[idx]

        if self.mu is not None and self.sigma is not None:
            x = (x - self.mu) / (self.sigma + 1e-8)

        x = x.astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32).transpose(0, 1)  # (C,T)

        return (
            x,
            torch.tensor(leak_count, dtype=torch.long),
            torch.tensor(pipe_t, dtype=torch.long),        # (3,)
            torch.tensor(pos_t, dtype=torch.float32),      # (3,)
            torch.tensor(size_t, dtype=torch.long),        # (3,)
            torch.tensor(slot_mask, dtype=torch.float32),  # (3,)
        )



# MODEL

class MultiLeakTCN(nn.Module):
    """
    Shared TCN backbone + multi-head outputs:
      - count: 4 classes (0,1,2,3)
      - pipe:  3 slots x 6 classes (5 pipes + NONE)
      - size:  3 slots x 4 classes (S/M/L + NONE)
      - pos:   3 slots x regression
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

        self.count_head = nn.Linear(32, 4)  # 0..3

        # slot heads (flattened)
        self.pipe_head = nn.Linear(32, MAX_LEAKS * PIPE_CLASSES)  # 3*6
        self.size_head = nn.Linear(32, MAX_LEAKS * SIZE_CLASSES)  # 3*4
        self.pos_head  = nn.Linear(32, MAX_LEAKS * 1)             # 3*1

    def forward(self, x):
        z = self.backbone(x)

        count_logits = self.count_head(z)

        pipe_logits = self.pipe_head(z).view(-1, MAX_LEAKS, PIPE_CLASSES)
        size_logits = self.size_head(z).view(-1, MAX_LEAKS, SIZE_CLASSES)
        pos_pred    = self.pos_head(z).view(-1, MAX_LEAKS)  # (B,3)

        return count_logits, pipe_logits, size_logits, pos_pred



# TRAIN / EVAL

folders = list_valid_scenario_folders(DATASET_ROOT)
if len(folders) == 0:
    raise FileNotFoundError(f"No scenario folders with signals.csv + labels.json found inside '{DATASET_ROOT}'.")

random.shuffle(folders)

split = int(0.75 * len(folders))
train_f = folders[:split]
test_f  = folders[split:]

mu, sigma = compute_mu_sigma(train_f)

train_ds = LeakDatasetMulti(train_f, mu, sigma)
test_ds  = LeakDatasetMulti(test_f, mu, sigma)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiLeakTCN(len(FEATURE_COLS)).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)

loss_count = nn.CrossEntropyLoss()
loss_pipe  = nn.CrossEntropyLoss()
loss_size  = nn.CrossEntropyLoss()
loss_pos   = nn.SmoothL1Loss(reduction="none")  


for ep in range(EPOCHS):
    model.train()
    total = 0.0

    for X, leak_count, pipe_t, pos_t, size_t, slot_mask in train_loader:
        X = X.to(device)
        leak_count = leak_count.to(device)      # (B,)
        pipe_t = pipe_t.to(device)              # (B,3)
        pos_t = pos_t.to(device)                # (B,3)
        size_t = size_t.to(device)              # (B,3)
        slot_mask = slot_mask.to(device)        # (B,3)

        opt.zero_grad()

        count_logits, pipe_logits, size_logits, pos_pred = model(X)

        # 1) leak count loss
        Lc = loss_count(count_logits, leak_count)

        # 2) pipe loss 
        Lp = loss_pipe(
            pipe_logits.reshape(-1, PIPE_CLASSES),
            pipe_t.reshape(-1)
        )

        # 3) size loss (includes NONE class in unused slots)
        Ls = loss_size(
            size_logits.reshape(-1, SIZE_CLASSES),
            size_t.reshape(-1)
        )

        # 4) position loss only for real leak slots
        # SmoothL1Loss gives (B,3)
        pos_err = loss_pos(pos_pred, pos_t)         
        masked = pos_err * slot_mask                  
        denom = slot_mask.sum().clamp(min=1.0)
        Lr = masked.sum() / denom

        loss = Lc + Lp + Ls + Lr
        loss.backward()
        opt.step()

        total += float(loss.item())

    print(f"Epoch {ep+1}/{EPOCHS}, loss={total/len(train_loader):.4f}")



# SAVE BUNDLE

save_bundle = {
    "model_state_dict": model.state_dict(),
    "mu": mu,
    "sigma": sigma,
    "feature_cols": FEATURE_COLS,
    "window": WINDOW,
    "stride": STRIDE,
    "dataset_root": DATASET_ROOT,
    "max_leaks": MAX_LEAKS,
    "pipe_classes": PIPE_CLASSES,
    "size_classes": SIZE_CLASSES,
}

torch.save(save_bundle, "multileak_tcn_bundle.pt")
print("[OK] Saved multileak_tcn_bundle.pt")



# TEST METRICS (basic)

model.eval()

count_true, count_pred = [], []
with torch.no_grad():
    for X, leak_count, pipe_t, pos_t, size_t, slot_mask in test_loader:
        X = X.to(device)
        count_logits, pipe_logits, size_logits, pos_pred = model(X)

        pred_count = count_logits.argmax(dim=1).cpu().numpy().astype(int).tolist()
        true_count = leak_count.numpy().astype(int).tolist()

        count_pred.extend(pred_count)
        count_true.extend(true_count)

print("\n=== LEAK COUNT ===")
print("Accuracy:", accuracy_score(count_true, count_pred))
print(confusion_matrix(count_true, count_pred))


# Slot-wise pipe accuracy (including NONE)
pipe_true_all, pipe_pred_all = [], []
pos_true_all, pos_pred_all = [], []
size_true_all, size_pred_all = [], []
mask_all = []

with torch.no_grad():
    for X, leak_count, pipe_t, pos_t, size_t, slot_mask in test_loader:
        X = X.to(device)
        _, pipe_logits, size_logits, pos_pred = model(X)

        pipe_pred = pipe_logits.argmax(dim=2).cpu().numpy()  # (B,3)
        size_pred = size_logits.argmax(dim=2).cpu().numpy()  # (B,3)

        pipe_true_all.append(pipe_t.numpy())
        pipe_pred_all.append(pipe_pred)

        size_true_all.append(size_t.numpy())
        size_pred_all.append(size_pred)

        pos_true_all.append(pos_t.numpy())
        pos_pred_all.append(pos_pred.cpu().numpy())

        mask_all.append(slot_mask.numpy())

pipe_true_all = np.vstack(pipe_true_all)
pipe_pred_all = np.vstack(pipe_pred_all)
size_true_all = np.vstack(size_true_all)
size_pred_all = np.vstack(size_pred_all)
pos_true_all  = np.vstack(pos_true_all)
pos_pred_all  = np.vstack(pos_pred_all)
mask_all      = np.vstack(mask_all)

print("\n=== PIPE (slot-wise, includes NONE) ===")
print("Accuracy:", accuracy_score(pipe_true_all.reshape(-1), pipe_pred_all.reshape(-1)))
print(confusion_matrix(pipe_true_all.reshape(-1), pipe_pred_all.reshape(-1)))

print("\n=== SIZE (slot-wise, includes NONE) ===")
print("Accuracy:", accuracy_score(size_true_all.reshape(-1), size_pred_all.reshape(-1)))
print(confusion_matrix(size_true_all.reshape(-1), size_pred_all.reshape(-1)))

# Position error only on real leaks
real_mask = mask_all.reshape(-1) > 0.5
pos_true_flat = pos_true_all.reshape(-1)[real_mask]
pos_pred_flat = pos_pred_all.reshape(-1)[real_mask]

if len(pos_true_flat) > 0:
    mae = float(np.mean(np.abs(pos_true_flat - pos_pred_flat)))
    rmse = float(np.sqrt(np.mean((pos_true_flat - pos_pred_flat) ** 2)))
    print("\n=== POSITION (real leaks only) ===")
    print("MAE:", mae)
    print("RMSE:", rmse)
else:
    print("\n=== POSITION (real leaks only) ===")
    print("No real-leak slots in test set -> cannot compute.")
