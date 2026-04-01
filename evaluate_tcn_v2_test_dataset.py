import os
import json

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


# ======================
# CONFIG
# ======================
BUNDLE_PATH  = "leak_tcn_v2_bundle.pt"
DATASET_ROOT = "test_dataset/scenarios"
MANIFEST     = "test_dataset/manifests/manifest.csv"

BATCH_SIZE = 64

NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES  = NUM_PIPES + 1

SIZE_TO_IDX   = {"S": 0, "M": 1, "L": 2}
SIZE_NONE_IDX = 3
SIZE_CLASSES  = 4


# ======================
# LOAD BUNDLE
# ======================
bundle     = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)
mu         = bundle["mu"]
sigma      = bundle["sigma"]
FEATURE_COLS = bundle["feature_cols"]
WINDOW     = bundle["window"]
STRIDE     = bundle["stride"]

print(f"Loaded bundle: {BUNDLE_PATH}")
print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"Window: {WINDOW}, Stride: {STRIDE}")


# ======================
# MODEL
# ======================
class LeakTCN(nn.Module):
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
        self.detect_head = nn.Linear(32, 2)
        self.pipe_head   = nn.Linear(32, PIPE_CLASSES)
        self.size_head   = nn.Linear(32, SIZE_CLASSES)
        self.pos_head    = nn.Linear(32, 1)

    def forward(self, x):
        z = self.backbone(x)
        return (
            self.detect_head(z),
            self.pipe_head(z),
            self.size_head(z),
            self.pos_head(z).squeeze(1),
        )


device = "cuda" if torch.cuda.is_available() else "cpu"
model = LeakTCN(len(FEATURE_COLS)).to(device)
model.load_state_dict(bundle["model_state_dict"])
model.eval()
print(f"Model loaded on {device}\n")


# ======================
# DATA HELPERS
# ======================
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


class LeakDataset(Dataset):
    def __init__(self, manifest_path, dataset_root, mu, sigma):
        self.samples = []
        self.mu    = mu
        self.sigma = sigma

        df_man = pd.read_csv(manifest_path)
        skipped = 0
        for scn_id in df_man["scenario_id"].values:
            folder   = os.path.join(dataset_root, f"scenario_{int(scn_id):05d}")
            sig_path = os.path.join(folder, "data.csv")
            lab_path = os.path.join(folder, "labels.json")

            if not (os.path.isfile(sig_path) and os.path.isfile(lab_path)):
                skipped += 1
                continue

            df = pd.read_csv(sig_path)
            X_full = df[FEATURE_COLS].values.astype(np.float32)

            with open(lab_path, "r") as f:
                labels = json.load(f)

            detect, pipe_t, pos_t, size_t = encode_labels_from_json(labels)

            for end in range(WINDOW, len(X_full) + 1, STRIDE):
                Xw = X_full[end - WINDOW:end]
                self.samples.append((Xw, detect, pipe_t, pos_t, size_t))

        print(f"Loaded {len(self.samples)} windows from {len(df_man)-skipped} scenarios "
              f"({skipped} skipped).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, detect, pipe_t, pos_t, size_t = self.samples[idx]
        x = (x - self.mu) / (self.sigma + 1e-8)
        x = torch.tensor(x, dtype=torch.float32).transpose(0, 1)  # (C,T)
        return (
            x,
            torch.tensor(detect, dtype=torch.long),
            torch.tensor(pipe_t, dtype=torch.long),
            torch.tensor(pos_t,  dtype=torch.float32),
            torch.tensor(size_t, dtype=torch.long),
        )


test_ds     = LeakDataset(MANIFEST, DATASET_ROOT, mu, sigma)
test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False)


# ======================
# EVALUATION
# ======================
detect_true, detect_pred = [], []
pipe_true_all, pipe_pred_all = [], []
size_true_all, size_pred_all = [], []
pos_true_all,  pos_pred_all  = [], []
leak_mask_all = []

with torch.no_grad():
    for X, detect, pipe_t, pos_t, size_t in test_loader:
        X = X.to(device)
        detect_logits, pipe_logits, size_logits, pos_pred = model(X)

        detect_pred.extend(detect_logits.argmax(dim=1).cpu().numpy().astype(int).tolist())
        detect_true.extend(detect.numpy().astype(int).tolist())

        pipe_pred_all.append(pipe_logits.argmax(dim=1).cpu().numpy())
        pipe_true_all.append(pipe_t.numpy())

        size_pred_all.append(size_logits.argmax(dim=1).cpu().numpy())
        size_true_all.append(size_t.numpy())

        pos_pred_all.append(pos_pred.cpu().numpy())
        pos_true_all.append(pos_t.numpy())

        leak_mask_all.append(detect.numpy())

pipe_true_all = np.concatenate(pipe_true_all)
pipe_pred_all = np.concatenate(pipe_pred_all)
size_true_all = np.concatenate(size_true_all)
size_pred_all = np.concatenate(size_pred_all)
pos_true_all  = np.concatenate(pos_true_all)
pos_pred_all  = np.concatenate(pos_pred_all)
leak_mask_all = np.concatenate(leak_mask_all)

print("\n=== DETECTION ===")
print("Accuracy:", accuracy_score(detect_true, detect_pred))
print(confusion_matrix(detect_true, detect_pred))

print("\n=== PIPE (all scenarios, includes NONE) ===")
print("Accuracy:", accuracy_score(pipe_true_all, pipe_pred_all))
print(confusion_matrix(pipe_true_all, pipe_pred_all))

print("\n=== SIZE (all scenarios, includes NONE) ===")
print("Accuracy:", accuracy_score(size_true_all, size_pred_all))
print(confusion_matrix(size_true_all, size_pred_all))

real_mask = leak_mask_all > 0.5

pipe_true_leak = pipe_true_all[real_mask]
pipe_pred_leak = pipe_pred_all[real_mask]
f1_pipe = f1_score(pipe_true_leak, pipe_pred_leak, average="macro", labels=list(range(NUM_PIPES)))
print(f"\n=== PIPE F1 MACRO (leak scenarios only) ===")
print(f"F1 Macro: {f1_pipe:.4f}")

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
