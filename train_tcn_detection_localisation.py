import os
import json
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score


DATASET_ROOT = "manual_output"
FEATURE_COLS = ["P2","P3","P4","P5","P6","Q1a","Q2a","Q3a","Q4a","Q5a"]

# window and stride for sliding window segementation
WINDOW = 180
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)


class LeakDataset(Dataset):
    def __init__(self, scenario_folders, mu=None, sigma=None):
        self.samples = []
        self.mu = mu
        self.sigma = sigma

        for folder in scenario_folders:
            sig_path = os.path.join(folder, "signals.csv")
            lab_path = os.path.join(folder, "labels.json")

            df = pd.read_csv(sig_path)
            X = df[FEATURE_COLS].values.astype(np.float32)

            with open(lab_path, "r") as f:
                labels = json.load(f)

            leak = int(labels.get("leak_present", 0))

            if leak == 1:
                pipe_id = int(labels.get("pipe_id", 1))
                pipe = pipe_id - 1                       # convert to 0-indexed for torch
                pos = float(labels.get("position", 0.0))
            else:
                pipe = -1
                pos = 0.0

            for end in range(WINDOW, len(X) + 1, STRIDE):
                self.samples.append((X[end-WINDOW:end], leak, pipe, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, leak, pipe, pos = self.samples[idx]

        if self.mu is not None and self.sigma is not None:
            x = (x - self.mu) / (self.sigma + 1e-8)

        x = x.astype(np.float32)

        # transpose to (C, T0) as required by Conv1d
        x = torch.tensor(x, dtype=torch.float32).transpose(0, 1)

        return (
            x,
            torch.tensor(leak, dtype=torch.float32),
            torch.tensor(pipe, dtype=torch.long),
            torch.tensor(pos, dtype=torch.float32)
        )


def compute_mu_sigma(folders):
    all_data = []
    for f in folders:
        sig_path = os.path.join(f, "signals.csv")
        df = pd.read_csv(sig_path)
        all_data.append(df[FEATURE_COLS].values)
    all_data = np.vstack(all_data).astype(np.float32)
    return all_data.mean(axis=0), all_data.std(axis=0)


class MultiTaskTCN(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.tcn = nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=4, dilation=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=8, dilation=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=16, dilation=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.detect = nn.Linear(32, 1)
        self.pipe   = nn.Linear(32, 5)
        self.pos    = nn.Linear(32, 1)

    def forward(self, x):
        z = self.tcn(x)
        return self.detect(z), self.pipe(z), self.pos(z)


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


folders = list_valid_scenario_folders(DATASET_ROOT)
if len(folders) == 0:
    raise FileNotFoundError(f"No scenario folders with signals.csv + labels.json found inside '{DATASET_ROOT}'.")

random.shuffle(folders)

split = int(0.75 * len(folders))
train_f = folders[:split]
test_f  = folders[split:]

mu, sigma = compute_mu_sigma(train_f)

train_ds = LeakDataset(train_f, mu, sigma)
test_ds  = LeakDataset(test_f, mu, sigma)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False)


# TRAIN

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiTaskTCN(len(FEATURE_COLS)).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_det  = nn.BCEWithLogitsLoss()
loss_pipe = nn.CrossEntropyLoss()
loss_pos  = nn.MSELoss()

for ep in range(EPOCHS):
    model.train()
    total = 0.0

    for X, leak, pipe, pos in train_loader:
        X = X.to(device)
        leak = leak.to(device)
        pipe = pipe.to(device)
        pos = pos.to(device)

        opt.zero_grad()
        d, p, r = model(X)

        # Detection loss (always)
        Ld = loss_det(d.squeeze(), leak)

        # only compute pipe/position loss on leak windows
        mask = (leak == 1)

        if mask.any():
            Lp = loss_pipe(p[mask], pipe[mask])
            Lr = loss_pos(r.squeeze()[mask], pos[mask])
        else:
            Lp = torch.tensor(0.0, device=device)
            Lr = torch.tensor(0.0, device=device)

        loss = Ld + Lp + Lr
        loss.backward()
        opt.step()

        total += float(loss.item())

    print(f"Epoch {ep+1}/{EPOCHS}, loss={total/len(train_loader):.4f}")


# Save bundle
save_bundle = {
    "model_state_dict": model.state_dict(),
    "mu": mu,
    "sigma": sigma,
    "feature_cols": FEATURE_COLS,
    "window": WINDOW,
    "stride": STRIDE,
    "dataset_root": DATASET_ROOT
}

torch.save(save_bundle, "multitask_tcn_bundle.pt")
print("[OK] Saved multitask_tcn_bundle.pt")


# TEST

model.eval()
y_true, y_pred = [], []
pipe_true, pipe_pred = [], []

with torch.no_grad():
    for X, leak, pipe, pos in test_loader:
        X = X.to(device)
        d, p, r = model(X)

        pred_det = (torch.sigmoid(d) > 0.5).cpu().numpy().astype(int).ravel()
        true_det = leak.numpy().astype(int).ravel()

        y_pred.extend(pred_det.tolist())
        y_true.extend(true_det.tolist())

        mask = (leak == 1)
        if mask.any():
            pipe_pred.extend(p[mask].argmax(dim=1).cpu().numpy().astype(int).tolist())
            pipe_true.extend(pipe[mask].numpy().astype(int).tolist())

print("\nDETECTION")
print("Accuracy:", accuracy_score(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

if len(pipe_true) > 0:
    print("\n PIPE LOCALISATION (Leaks only)")
    print("Accuracy:", accuracy_score(pipe_true, pipe_pred))
    print(confusion_matrix(pipe_true, pipe_pred))
else:
    print("\n PIPE LOCALISATION (Leaks only)")
    print("No leak samples in test set -> cannot compute pipe localisation metrics.")


# second pass to get position predictions separately
pos_true, pos_pred = [], []

with torch.no_grad():
    for X, leak, pipe, pos in test_loader:
        X = X.to(device).float()
        d, p, r = model(X)

        mask = (leak == 1)
        if mask.any():
            pos_true.extend(pos[mask].numpy().astype(np.float32).tolist())
            pos_pred.extend(r.squeeze()[mask].cpu().numpy().astype(np.float32).tolist())

pos_true = np.array(pos_true, dtype=np.float32)
pos_pred = np.array(pos_pred, dtype=np.float32)

if len(pos_true) > 0:
    mae = np.mean(np.abs(pos_true - pos_pred))
    rmse = np.sqrt(np.mean((pos_true - pos_pred) ** 2))

    print("\n POSITION ESTIMATION (Leaks only)")
    print("MAE:", float(mae))
    print("RMSE:", float(rmse))
else:
    print("\n POSITION ESTIMATION (Leaks only)")
    print("No leak samples in test set -> cannot compute position error.")
