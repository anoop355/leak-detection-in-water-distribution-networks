"""
debug_s10a_softmax.py
=====================
Debugging script for S10-A STGCN model.

Loads the S10-A bundle and runs inference on the 12 scenarios where
true_pipe=1 was misclassified as pred_pipe=4.

Outputs:
  - Raw pipe head logits
  - Softmax probabilities for each pipe class
  - Detect head softmax
  - Predicted vs. true labels

Also saves results to:
  stgcn_placement_results/S10-A/debug_pipe1_misclassified_softmax.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BUNDLE_PATH  = Path("stgcn_placement_bundles/stgcn_bundle_S10-A.pt")
TEST_DIR     = Path("test_dataset/scenarios")
OUTPUT_PATH  = Path("stgcn_placement_results/S10-A/debug_pipe1_misclassified_softmax.csv")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES  = NUM_PIPES + 1  # pipes 1-5 + "no pipe"

# The 12 misclassified scenario folders (true_pipe=1, pred_pipe=4)
MISCLASSIFIED_SCENARIOS = [
    "scenario_00053",
    "scenario_00067",
    "scenario_00081",
    "scenario_00094",
    "scenario_00095",
    "scenario_00109",
    "scenario_00123",
    "scenario_00136",
    "scenario_00137",
    "scenario_00151",
    "scenario_00179",
    "scenario_00193",
]


# ---------------------------------------------------------------------------
# Model definitions (must match train_stgcn_sensor_placement.py)
# ---------------------------------------------------------------------------

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


class SingleLeakSTGCNv4(nn.Module):
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, 4, dropout)
        head_in, head_hidden = num_nodes * hidden_2, 64

        self.temporal_pool = TemporalAttentionPool(hidden_2, num_nodes)

        def _head(out_size):
            return nn.Sequential(
                nn.Linear(head_in, head_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, out_size)
            )

        self.detect_head = _head(2)
        self.pipe_head   = _head(PIPE_CLASSES)
        self.size_head   = _head(4)
        self.pos_head    = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        z = self.temporal_pool(x)
        return (self.detect_head(z), self.pipe_head(z),
                self.size_head(z), self.pos_head(z).squeeze(1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bundle(bundle_path: Path):
    bundle     = torch.load(str(bundle_path), map_location=DEVICE, weights_only=False)
    adj        = np.array(bundle["adjacency"], dtype=np.float32)
    num_nodes  = len(bundle["sensor_names"])
    hidden_1   = int(bundle.get("hidden_1",   16))
    hidden_2   = int(bundle.get("hidden_2",   32))
    kernel_sz  = int(bundle.get("kernel_size", 5))
    dropout    = float(bundle.get("dropout",  0.25))
    node_feats = int(bundle.get("node_feats",  2))

    model = SingleLeakSTGCNv4(adj, num_nodes, hidden_1, hidden_2,
                               kernel_sz, dropout, node_feats).to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle


def preprocess(raw: np.ndarray, baseline: np.ndarray,
               mu: np.ndarray, sigma: np.ndarray,
               node_feats: int = 2) -> torch.Tensor:
    T      = raw.shape[0]
    base   = baseline[:T]
    channels = [raw, raw - base]
    if node_feats == 3:
        diff = np.zeros_like(raw)
        diff[1:] = raw[1:] - raw[:-1]
        channels.append(diff)
    feats = np.stack(channels, axis=-1).astype(np.float32)
    feats = (feats - mu[None]) / (sigma[None] + 1e-8)
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device : {DEVICE}")
    print(f"Bundle : {BUNDLE_PATH}")
    print()

    model, bundle = load_bundle(BUNDLE_PATH)
    sensor_names = bundle["sensor_names"]
    mu           = np.array(bundle["mu"],                dtype=np.float32)
    sigma        = np.array(bundle["sigma"],             dtype=np.float32)
    baseline     = np.array(bundle["baseline_template"], dtype=np.float32)
    node_feats   = int(bundle.get("node_feats", 2))

    # Pipe class labels: index 0 = pipe 1, ..., index 4 = pipe 5, index 5 = no-pipe
    pipe_class_names = [f"pipe_{i+1}" for i in range(NUM_PIPES)] + ["no_pipe"]

    rows = []

    print(f"{'Scenario':<20} {'TruePipe':>8} {'PredPipe':>8} | "
          + "  ".join(f"{n:>10}" for n in pipe_class_names))
    print("-" * (20 + 8 + 8 + 4 + len(pipe_class_names) * 12))

    for scn_name in MISCLASSIFIED_SCENARIOS:
        scn_dir    = TEST_DIR / scn_name
        data_path  = scn_dir / "data.csv"
        label_path = scn_dir / "labels.json"

        if not data_path.exists() or not label_path.exists():
            print(f"[WARN] Missing files for {scn_name}")
            continue

        labels       = json.loads(label_path.read_text(encoding="utf-8"))
        true_pipe_id = int(labels["label_pipe"])
        true_pos     = float(labels["label_position"])
        true_size    = str(labels["label_size"])

        raw = pd.read_csv(data_path)[sensor_names].to_numpy(dtype=np.float32)
        x_t = preprocess(raw, baseline, mu, sigma, node_feats).to(DEVICE)

        with torch.no_grad():
            detect_logits, pipe_logits, size_logits, pos_pred = model(x_t)

        # Softmax probabilities
        detect_probs = F.softmax(detect_logits, dim=1).squeeze(0).cpu().numpy()
        pipe_probs   = F.softmax(pipe_logits,   dim=1).squeeze(0).cpu().numpy()
        pipe_logits_np = pipe_logits.squeeze(0).cpu().numpy()

        pred_detect  = int(detect_logits.argmax(dim=1).item())
        pred_pipe_idx = int(pipe_logits.argmax(dim=1).item())
        pred_pipe_id  = (pred_pipe_idx + 1) if pred_pipe_idx < NUM_PIPES else None
        pred_pos      = float(pos_pred.item())

        # Console output
        prob_str = "  ".join(f"{p:>10.4f}" for p in pipe_probs)
        print(f"{scn_name:<20} {true_pipe_id:>8} {pred_pipe_id!s:>8} | {prob_str}")

        row = {
            "scenario":      scn_name,
            "true_pipe":     true_pipe_id,
            "pred_pipe":     pred_pipe_id,
            "true_pos":      true_pos,
            "pred_pos":      round(pred_pos, 4),
            "true_size":     true_size,
            "detect_prob_no_leak":  round(float(detect_probs[0]), 6),
            "detect_prob_leak":     round(float(detect_probs[1]), 6),
        }

        # Add pipe softmax and logit columns
        for i, name in enumerate(pipe_class_names):
            row[f"softmax_{name}"] = round(float(pipe_probs[i]),    6)
            row[f"logit_{name}"]   = round(float(pipe_logits_np[i]), 6)

        rows.append(row)

    # Save CSV
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 80)
    print("PIPE HEAD SOFTMAX SUMMARY")
    print("=" * 80)
    softmax_cols = [f"softmax_{n}" for n in pipe_class_names]
    print(df[["scenario", "true_pipe", "pred_pipe", "true_pos"] + softmax_cols].to_string(index=False))

    print()
    print("PIPE HEAD LOGIT SUMMARY")
    print("=" * 80)
    logit_cols = [f"logit_{n}" for n in pipe_class_names]
    print(df[["scenario", "true_pipe", "pred_pipe"] + logit_cols].to_string(index=False))

    print()
    print(f"[SAVED] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
