import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wntr


# ======================
# SAME extraction as before
# ======================
PRESSURE_NODES = ["2", "3", "4", "5", "6"]
FLOW_LINKS = ["1a", "2a", "3a", "4a", "5a"]
SIGNALS_COL_ORDER = ["t", "P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]


def run_inp_and_extract_signals(inp_path: str) -> pd.DataFrame:
    wn = wntr.network.WaterNetworkModel(inp_path)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    P = results.node["pressure"]
    Q = results.link["flowrate"]

    out = pd.DataFrame()
    out["t"] = (P.index.to_numpy() / 60.0).astype(int)

    for n in PRESSURE_NODES:
        out[f"P{n}"] = P[n].to_numpy()

    for l in FLOW_LINKS:
        out[f"Q{l}"] = Q[l].to_numpy()

    out = out[SIGNALS_COL_ORDER]
    return out


# ======================
# NEW MODEL (must match training)
# ======================
MAX_LEAKS = 3
NUM_PIPES = 5
PIPE_NONE_IDX = NUM_PIPES         # 5
PIPE_CLASSES = NUM_PIPES + 1      # 6

# NOTE: size head is still part of the trained model, but we will NOT output size
SIZE_CLASSES = 4


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

        self.count_head = nn.Linear(32, 4)  # 0..3
        self.pipe_head  = nn.Linear(32, MAX_LEAKS * PIPE_CLASSES)  # 3*6
        self.size_head  = nn.Linear(32, MAX_LEAKS * SIZE_CLASSES)  # 3*4 (kept for compatibility)
        self.pos_head   = nn.Linear(32, MAX_LEAKS)                 # 3

    def forward(self, x):
        z = self.backbone(x)
        count_logits = self.count_head(z)
        pipe_logits = self.pipe_head(z).view(-1, MAX_LEAKS, PIPE_CLASSES)
        size_logits = self.size_head(z).view(-1, MAX_LEAKS, SIZE_CLASSES)  # computed but ignored in output
        pos_pred = self.pos_head(z).view(-1, MAX_LEAKS)
        return count_logits, pipe_logits, size_logits, pos_pred


# ======================
# Aggregation helpers
# ======================
def majority_vote(arr: np.ndarray, valid_mask: np.ndarray = None, default=None):
    if valid_mask is not None:
        arr = arr[valid_mask]
    if arr.size == 0:
        return default
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def mean_over_valid(values: np.ndarray, valid_mask: np.ndarray, default=0.0):
    v = values[valid_mask]
    if v.size == 0:
        return float(default)
    return float(np.mean(v))


# ======================
# MAIN
# ======================
def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_from_inp.py path_to_test_case.inp")
        sys.exit(1)

    inp_path = sys.argv[1]
    bundle_path = "multileak_tcn_bundleV5.pt"

    # 1) Load trained bundle
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    mu = np.array(bundle["mu"], dtype=np.float32)
    sigma = np.array(bundle["sigma"], dtype=np.float32)
    feature_cols = bundle["feature_cols"]
    window = int(bundle["window"])
    stride = int(bundle.get("stride", 10))

    # 2) Load model
    model = MultiLeakTCN(C=len(feature_cols))
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    # 3) Run EPANET and extract signals (WNTR)
    signals = run_inp_and_extract_signals(inp_path)

    # Save extracted signals beside the .inp (optional but useful)
    case_dir = os.path.dirname(inp_path) if os.path.dirname(inp_path) else "."
    case_name = os.path.splitext(os.path.basename(inp_path))[0]
    csv_out = os.path.join(case_dir, f"{case_name}__signals_wntr.csv")
    signals.to_csv(csv_out, index=False)
    print(f"[OK] Saved WNTR-extracted signals to: {csv_out}")
    print(f"[INFO] signals shape: {signals.shape}")

    # 4) Build windows
    X = signals[feature_cols].to_numpy(dtype=np.float32)
    if len(X) < window:
        raise ValueError(f"Not enough samples. Need at least {window}, got {len(X)}.")

    windows = []
    for end in range(window, len(X) + 1, stride):
        x_win = X[end-window:end, :]
        x_win = (x_win - mu) / (sigma + 1e-8)
        x_win = x_win.astype(np.float32)
        x_t = torch.tensor(x_win, dtype=torch.float32).transpose(0, 1).unsqueeze(0)  # (1,C,T)
        windows.append(x_t)

    X_batch = torch.cat(windows, dim=0)  # (N,C,T)
    print(f"[INFO] Built {X_batch.shape[0]} windows for prediction.", flush=True)

    # 5) Predict all windows
    with torch.no_grad():
        count_logits, pipe_logits, _size_logits, pos_pred = model(X_batch)  # size ignored

    # Convert to numpy
    count_pred = count_logits.argmax(dim=1).cpu().numpy().astype(int)        # (N,)
    pipe_pred  = pipe_logits.argmax(dim=2).cpu().numpy().astype(int)         # (N,3)
    pos_pred   = pos_pred.cpu().numpy().astype(np.float32)                   # (N,3)

    # 6) Aggregate
    final_count = majority_vote(count_pred, default=0)

    final_leaks = []
    for slot in range(MAX_LEAKS):
        slot_pipe = pipe_pred[:, slot]
        slot_pos  = pos_pred[:, slot]

        # consider only windows where slot isn't predicted as NONE (helps stabilize)
        valid = slot_pipe != PIPE_NONE_IDX

        pipe_vote = majority_vote(slot_pipe, valid_mask=valid, default=PIPE_NONE_IDX)
        pos_mean  = mean_over_valid(slot_pos, valid_mask=valid, default=0.0)

        if pipe_vote == PIPE_NONE_IDX:
            continue  # ignore empty slot

        pipe_id = pipe_vote + 1
        pos_mean = max(0.0, min(1.0, float(pos_mean)))

        final_leaks.append({
            "pipe_id": int(pipe_id),
            "position": float(pos_mean)
        })

    # Enforce count by sorting and trimming (in case head and slots disagree)
    final_leaks = sorted(final_leaks, key=lambda d: d["pipe_id"])
    if final_count > 0:
        final_leaks = final_leaks[:final_count]

    # 7) Print output
    print("\n=== OUTPUT (Aggregated over all windows) ===")
    print(f"Predicted leak count: {final_count}")

    if final_count == 0:
        print("No leak predicted.")
    else:
        for i, lk in enumerate(final_leaks, 1):
            print(f"Leak {i}: pipe={lk['pipe_id']}, r={lk['position']:.3f}")

    # Optional: write a prediction json beside the inp
    pred_out = os.path.join(case_dir, f"{case_name}__prediction.json")
    payload = {
        "source_inp": os.path.basename(inp_path),
        "predicted_leak_count": int(final_count),
        "predicted_leaks": final_leaks,
    }
    with open(pred_out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[OK] Saved prediction JSON: {pred_out}")


if __name__ == "__main__":
    main()
