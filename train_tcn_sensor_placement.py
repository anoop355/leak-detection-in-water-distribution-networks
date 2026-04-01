import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, accuracy_score


# ==========================================================
# USER SETTINGS
# ==========================================================

# Main project folder in Google Drive
PROJECT_ROOT = Path("/content/drive/MyDrive/colab_upload")

# Training dataset folder
DATASET_ROOT = Path("/content/training_cases_output")

# CSV that defines all sensor placements
PLACEMENTS_CSV = PROJECT_ROOT / "sensor_placements.csv"

# Output folder for trained models
MODELS_ROOT = PROJECT_ROOT / "trained_models"

# Start setting
START_FROM_MODEL = "S8-E"     # set to None to train from the beginning

# Training hyperparameters
WINDOW = 180
STRIDE = 10
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
SEED = 42

MAX_LEAKS = 3
NUM_PIPES = 5

PIPE_NONE_IDX = NUM_PIPES         # 5 means NONE
PIPE_CLASSES = NUM_PIPES + 1      # 6

SIZE_TO_IDX = {"S": 0, "M": 1, "L": 2}
SIZE_NONE_IDX = 3
SIZE_CLASSES = 4


# ==========================================================
# REPRODUCIBILITY
# ==========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ==========================================================
# DATA HELPERS
# ==========================================================
def list_valid_scenario_folders(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root '{root}' not found.")

    folders = []
    for name in os.listdir(root):
        path = root / name
        if not path.is_dir():
            continue
        sig_path = path / "signals.csv"
        lab_path = path / "labels.json"
        if sig_path.is_file() and lab_path.is_file():
            folders.append(path)

    return folders


def compute_mu_sigma(folders, feature_cols, scenario_id):
    all_data = []
    for f in folders:
        sig_path = f / "signals.csv"
        df = pd.read_csv(sig_path)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{scenario_id}] Missing columns in {sig_path}: {missing}")

        all_data.append(df[feature_cols].values)

    all_data = np.vstack(all_data).astype(np.float32)
    mu = all_data.mean(axis=0)
    sigma = all_data.std(axis=0)

    # protect against zero std
    sigma[sigma < 1e-12] = 1.0
    return mu, sigma


def _get_leaks_list(labels: dict):
    if "Leaks" in labels:
        return labels.get("Leaks", [])
    return labels.get("leaks", [])


def encode_labels_from_json(labels: dict):
    """
    Converts labels.json into fixed-size targets for MAX_LEAKS=3.
    Returns:
      leak_count: int in [0..3]
      pipe_targets: (3,) long in [0..5] where 5=NONE
      pos_targets:  (3,) float
      size_targets: (3,) long in [0..3] where 3=NONE
      slot_mask:    (3,) float 1.0 if slot is real leak else 0.0
    """
    leaks = _get_leaks_list(labels)
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

        pipe_targets[i] = pipe_id - 1                # 0..4
        pos_targets[i]  = pos
        size_targets[i] = SIZE_TO_IDX.get(size_level, 0)
        slot_mask[i]    = 1.0

    return leak_count, pipe_targets, pos_targets, size_targets, slot_mask


class LeakDatasetMulti(Dataset):
    def __init__(self, scenario_folders, feature_cols, scenario_id, window, stride, mu=None, sigma=None):
        self.samples = []
        self.mu = mu
        self.sigma = sigma
        self.feature_cols = feature_cols
        self.window = window
        self.stride = stride
        self.scenario_id = scenario_id

        for folder in scenario_folders:
            sig_path = folder / "signals.csv"
            lab_path = folder / "labels.json"

            df = pd.read_csv(sig_path)

            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"[{self.scenario_id}] Missing columns in {sig_path}: {missing}")

            X_full = df[self.feature_cols].values.astype(np.float32)

            with open(lab_path, "r", encoding="utf-8") as f:
                labels = json.load(f)

            leak_count, pipe_t, pos_t, size_t, slot_mask = encode_labels_from_json(labels)

            for end in range(self.window, len(X_full) + 1, self.stride):
                Xw = X_full[end - self.window:end]
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
            torch.tensor(pipe_t, dtype=torch.long),
            torch.tensor(pos_t, dtype=torch.float32),
            torch.tensor(size_t, dtype=torch.long),
            torch.tensor(slot_mask, dtype=torch.float32),
        )


# ==========================================================
# MODEL
# ==========================================================
class MultiLeakTCN(nn.Module):
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

        self.count_head = nn.Linear(32, 4)
        self.pipe_head  = nn.Linear(32, MAX_LEAKS * PIPE_CLASSES)
        self.size_head  = nn.Linear(32, MAX_LEAKS * SIZE_CLASSES)
        self.pos_head   = nn.Linear(32, MAX_LEAKS * 1)

    def forward(self, x):
        z = self.backbone(x)
        count_logits = self.count_head(z)
        pipe_logits = self.pipe_head(z).view(-1, MAX_LEAKS, PIPE_CLASSES)
        size_logits = self.size_head(z).view(-1, MAX_LEAKS, SIZE_CLASSES)
        pos_pred    = self.pos_head(z).view(-1, MAX_LEAKS)
        return count_logits, pipe_logits, size_logits, pos_pred


# ==========================================================
# ONE MODEL TRAINING FUNCTION
# ==========================================================
def train_one_configuration(model_name, feature_cols):
    print("=" * 70)
    print(f"Training {model_name}")
    print(f"Features: {feature_cols}")
    print("=" * 70)

    folders = list_valid_scenario_folders(DATASET_ROOT)
    if len(folders) == 0:
        raise FileNotFoundError(f"No scenario folders with signals.csv + labels.json found inside '{DATASET_ROOT}'.")

    # ----------------------------------------------------------
    # Stratified split: no-leak and leak folders are split
    # independently to guarantee both classes are represented
    # in training and test sets.
    # ----------------------------------------------------------
    no_leak_folders = [f for f in folders if "no_leak" in f.name]
    leak_folders    = [f for f in folders if "no_leak" not in f.name]

    random.shuffle(no_leak_folders)
    random.shuffle(leak_folders)

    nl_split   = max(1, int(0.75 * len(no_leak_folders)))
    leak_split = int(0.75 * len(leak_folders))

    train_f = leak_folders[:leak_split] + no_leak_folders[:nl_split]
    test_f  = leak_folders[leak_split:] + no_leak_folders[nl_split:]

    print(f"Stratified split — train: {len(train_f)} folders "
          f"({len(no_leak_folders[:nl_split])} no-leak, {len(leak_folders[:leak_split])} leak)")
    print(f"Stratified split — test:  {len(test_f)} folders "
          f"({len(no_leak_folders[nl_split:])} no-leak, {len(leak_folders[leak_split:])} leak)")

    mu, sigma = compute_mu_sigma(train_f, feature_cols, model_name)

    train_ds = LeakDatasetMulti(train_f, feature_cols, model_name, WINDOW, STRIDE, mu, sigma)
    test_ds  = LeakDatasetMulti(test_f, feature_cols, model_name, WINDOW, STRIDE, mu, sigma)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiLeakTCN(len(feature_cols)).to(device)

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
            leak_count = leak_count.to(device)
            pipe_t = pipe_t.to(device)
            pos_t = pos_t.to(device)
            size_t = size_t.to(device)
            slot_mask = slot_mask.to(device)

            opt.zero_grad()

            count_logits, pipe_logits, size_logits, pos_pred = model(X)

            Lc = loss_count(count_logits, leak_count)

            # Pipe and size losses masked to real leak slots only
            slot_mask_flat   = slot_mask.reshape(-1).bool()
            pipe_logits_flat = pipe_logits.reshape(-1, PIPE_CLASSES)
            pipe_t_flat      = pipe_t.reshape(-1)

            if slot_mask_flat.any():
                Lp = loss_pipe(pipe_logits_flat[slot_mask_flat], pipe_t_flat[slot_mask_flat])
            else:
                Lp = torch.tensor(0.0, device=device)

            size_logits_flat = size_logits.reshape(-1, SIZE_CLASSES)
            size_t_flat      = size_t.reshape(-1)

            if slot_mask_flat.any():
                Ls = loss_size(size_logits_flat[slot_mask_flat], size_t_flat[slot_mask_flat])
            else:
                Ls = torch.tensor(0.0, device=device)

            pos_err = loss_pos(pos_pred, pos_t)
            masked  = pos_err * slot_mask
            denom   = slot_mask.sum().clamp(min=1.0)
            Lr      = masked.sum() / denom

            loss = Lc + Lp + Ls + Lr
            loss.backward()
            opt.step()

            total += float(loss.item())

        print(f"[{model_name}] Epoch {ep+1}/{EPOCHS}, loss={total/len(train_loader):.4f}")

    # ==========================================================
    # SAVE MODEL BUNDLE
    # ==========================================================
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    bundle_path = MODELS_ROOT / f"multileak_tcn_bundle_{model_name}.pt"

    save_bundle = {
        "scenario_id": model_name,
        "model_state_dict": model.state_dict(),
        "mu": mu,
        "sigma": sigma,
        "feature_cols": feature_cols,
        "window": WINDOW,
        "stride": STRIDE,
        "dataset_root": str(DATASET_ROOT),
        "max_leaks": MAX_LEAKS,
        "pipe_classes": PIPE_CLASSES,
        "size_classes": SIZE_CLASSES,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
    }

    torch.save(save_bundle, str(bundle_path))
    print(f"[OK] Saved bundle: {bundle_path}")

    # ==========================================================
    # QUICK SANITY CHECK
    # ==========================================================
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

    print(f"\n[{model_name}] === LEAK COUNT SANITY CHECK ===")
    print("Accuracy:", accuracy_score(count_true, count_pred))
    print(confusion_matrix(count_true, count_pred))


# ==========================================================
# MAIN LOOP OVER sensor_placements.csv
# ==========================================================
def main():
    if not PLACEMENTS_CSV.exists():
        raise FileNotFoundError(f"sensor_placements.csv not found at: {PLACEMENTS_CSV}")

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"training_cases_output not found at: {DATASET_ROOT}")

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    placements = pd.read_csv(PLACEMENTS_CSV)

    required_cols = ["model_name", "configuration"]
    missing_cols = [c for c in required_cols if c not in placements.columns]
    if missing_cols:
        raise ValueError(f"sensor_placements.csv missing required columns: {missing_cols}")

    # Clean strings
    placements["model_name"] = placements["model_name"].astype(str).str.strip()
    placements["configuration"] = placements["configuration"].astype(str).str.strip()

    # Skip S10-A only
    placements_to_train = placements[placements["model_name"] != "S10-A"].copy()

    if START_FROM_MODEL is not None:
        model_list = placements_to_train["model_name"].tolist()
        if START_FROM_MODEL not in model_list:
            raise ValueError(f"START_FROM_MODEL '{START_FROM_MODEL}' not found in sensor_placements.csv")

        start_idx = model_list.index(START_FROM_MODEL)
        placements_to_train = placements_to_train.iloc[start_idx:].copy()

    print(f"Total configurations in CSV: {len(placements)}")
    print(f"Configurations to train (excluding S10-A): {len(placements_to_train)}")

    for _, row in placements_to_train.iterrows():
        model_name = row["model_name"]

        # Parse feature list from configuration column
        feature_cols = [s.strip() for s in row["configuration"].split(",") if s.strip()]

        if len(feature_cols) == 0:
            print(f"[WARN] Skipping {model_name}: empty configuration")
            continue

        bundle_path = MODELS_ROOT / f"multileak_tcn_bundle_{model_name}.pt"

        if bundle_path.exists():
            print(f"[SKIP] {model_name} already exists at {bundle_path}")
            continue

        train_one_configuration(model_name, feature_cols)

    print("\n[DONE] Finished training all configurations from sensor_placements.csv")


if __name__ == "__main__":
    main()