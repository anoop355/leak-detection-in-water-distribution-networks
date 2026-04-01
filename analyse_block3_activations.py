"""
analyse_block3_activations.py
==============================
Extracts Block 3 (final ST-block before temporal pooling) node embeddings
from the S10-A ST-GCN model and compares mean activation magnitudes between:
  - 24 misidentified Pipe 1 scenarios
  - Correctly identified Pipe 1 scenarios (sampled to the same count)

Usage
-----
    python analyse_block3_activations.py

Outputs saved to: stgcn_placement_results/S10-A/block3_analysis/
"""

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BUNDLE_PATH  = Path("stgcn_placement_bundles/stgcn_bundle_S10-A.pt")
PER_SCEN_CSV = Path("stgcn_placement_results/S10-A/evaluation/S10-A_per_scenario_metrics.csv")
TEST_ROOT    = Path("test_dataset/scenarios")
OUT_DIR      = Path("stgcn_placement_results/S10-A/block3_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Model classes  (exact copies from train_stgcn_sensor_placement.py)
# ---------------------------------------------------------------------------

class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch,
                              kernel_size=(1, kernel_size),
                              padding=(0, pad),
                              dilation=(1, dilation))
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.conv(x); x = self.bn(x); x = self.act(x)
        x = x.permute(0, 3, 2, 1)
        return x


class GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.einsum("ij,btjc->btic", self.A, x)
        x = self.lin(x); x = self.ln(x); x = self.act(x)
        return x


class STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp    = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph   = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout = nn.Dropout(dropout)
        self.out_act = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        y = self.temp(x); y = self.graph(y); y = self.dropout(y)
        y = y + residual; y = self.out_act(y)
        return y


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * num_nodes, 1)

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat  = x.reshape(B, T, N * C)
        scores  = self.attn(x_flat)
        weights = torch.softmax(scores, dim=1)
        return (x_flat * weights).sum(dim=1)


class SingleLeakSTGCN(nn.Module):
    def __init__(self, adj, node_feats, hidden_1, hidden_2, kernel_size,
                 dropout, num_nodes, pipe_classes, size_classes):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj, kernel_size, 4, dropout)
        head_in = num_nodes * hidden_2
        h = 64
        self.detect_head = nn.Sequential(nn.Linear(head_in, h), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(h, 2))
        self.pipe_head   = nn.Sequential(nn.Linear(head_in, h), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(h, pipe_classes))
        self.size_head   = nn.Sequential(nn.Linear(head_in, h), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(h, size_classes))
        self.pos_head    = nn.Sequential(nn.Linear(head_in, h), nn.ReLU(),
                                         nn.Dropout(dropout), nn.Linear(h, 1), nn.Sigmoid())
        self.temporal_pool = TemporalAttentionPool(hidden_2, num_nodes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        z = self.temporal_pool(x)
        return (self.detect_head(z), self.pipe_head(z),
                self.size_head(z), self.pos_head(z).squeeze(1))


# ---------------------------------------------------------------------------
# Load bundle and build model
# ---------------------------------------------------------------------------
print("Loading bundle …")
bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)

sensor_names = bundle["sensor_names"]       # ['P2','P3','P4','P5','P6','Q1a','Q2a','Q3a','Q4a','Q5a']
adj          = bundle["adjacency"]
mu           = bundle["mu"]                 # (N, 2)
sigma        = bundle["sigma"]              # (N, 2)
baseline     = bundle["baseline_template"]  # (T_base, N)
window       = bundle["window"]             # 12
node_feats   = bundle["node_feats"]         # 2
hidden_1     = bundle["hidden_1"]           # 16
hidden_2     = bundle["hidden_2"]           # 32
kernel_size  = bundle["kernel_size"]        # 5
dropout      = bundle["dropout"]            # 0.25
num_nodes    = len(sensor_names)            # 10
pipe_classes = bundle["pipe_classes"]       # 6
size_classes = bundle["size_classes"]       # 4

model = SingleLeakSTGCN(adj, node_feats, hidden_1, hidden_2, kernel_size,
                        dropout, num_nodes, pipe_classes, size_classes)
model.load_state_dict(bundle["model_state_dict"])
model.eval()
print(f"Model loaded  |  sensors={sensor_names}")

# ---------------------------------------------------------------------------
# Forward hook — captures block3 output
# ---------------------------------------------------------------------------
_hook_store: dict = {}

def _block3_hook(module, input, output):
    # output: (B, T, N, C)
    _hook_store["block3_out"] = output.detach().cpu()

hook_handle = model.block3.register_forward_hook(_block3_hook)

# ---------------------------------------------------------------------------
# Preprocessing (identical to evaluate_single_leak.py for ST-GCN)
# ---------------------------------------------------------------------------
def preprocess(raw: np.ndarray) -> torch.Tensor:
    """raw: (T, N) → tensor (1, T, N, 2) normalised."""
    T = raw.shape[0]
    base = baseline[:T]
    feats = np.stack([raw, raw - base], axis=-1).astype(np.float32)   # (T, N, 2)
    feats = (feats - mu[None]) / (sigma[None] + 1e-8)
    return torch.tensor(feats).unsqueeze(0)                            # (1, T, N, 2)


def make_windows(raw: np.ndarray, win: int = 12, stride: int = 1):
    """Slide window over (T_full, N) signal → list of (win, N) arrays."""
    T = raw.shape[0]
    windows = []
    for start in range(0, T - win + 1, stride):
        windows.append(raw[start: start + win])
    return windows


# ---------------------------------------------------------------------------
# Load per-scenario CSV and identify groups
# ---------------------------------------------------------------------------
df = pd.read_csv(PER_SCEN_CSV)

miss_rows    = df[(df["true_pipe"] == 1) & (df["pred_pipe"] != 1) & (df["true_detect"] == 1)]
correct_rows = df[(df["true_pipe"] == 1) & (df["pred_pipe"] == 1) & (df["true_detect"] == 1)]

print(f"Misidentified Pipe 1 : {len(miss_rows)}")
print(f"Correctly identified : {len(correct_rows)}  (will sample {len(miss_rows)} for comparison)")

# Sample correct to same size for a fair visual comparison
correct_sample = correct_rows.sample(n=len(miss_rows), random_state=SEED)


# ---------------------------------------------------------------------------
# Run inference + hook extraction
# ---------------------------------------------------------------------------
def extract_node_activations(scenario_name: str) -> np.ndarray:
    """
    Load scenario data, slide windows, run each window through model,
    accumulate block3 node activation magnitudes.
    Returns: (N,) mean |activation| per node, averaged across all windows.
    """
    data_path = TEST_ROOT / scenario_name / "data.csv"
    data = pd.read_csv(data_path)
    raw = data[sensor_names].values.astype(np.float32)   # (T_full, N)

    windows = make_windows(raw, win=window, stride=1)
    node_scores = []

    with torch.no_grad():
        for w in windows:
            x = preprocess(w)       # (1, T, N, 2)
            model(x)                # triggers hook
            b3 = _hook_store["block3_out"]   # (1, T, N, C)
            # Mean activation magnitude over time (T) and channel (C) dims
            score = b3.squeeze(0).abs().mean(dim=(0, 2))  # (N,)
            node_scores.append(score.numpy())

    return np.stack(node_scores).mean(axis=0)   # (N,)


print("\nExtracting Block 3 activations …")
miss_activations    = []
correct_activations = []

for _, row in miss_rows.iterrows():
    a = extract_node_activations(row["scenario"])
    miss_activations.append(a)

for _, row in correct_sample.iterrows():
    a = extract_node_activations(row["scenario"])
    correct_activations.append(a)

miss_mean    = np.stack(miss_activations).mean(axis=0)    # (N,)
correct_mean = np.stack(correct_activations).mean(axis=0) # (N,)

hook_handle.remove()   # clean up

# ---------------------------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame({
    "sensor":           sensor_names,
    "misidentified_mean": miss_mean,
    "correct_mean":       correct_mean,
    "ratio_miss_over_correct": miss_mean / (correct_mean + 1e-8),
})
results_df.to_csv(OUT_DIR / "node_activation_scores.csv", index=False)
print("\nNode activation scores:")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Plot grouped bar chart
# ---------------------------------------------------------------------------
x     = np.arange(num_nodes)
width = 0.35

fig, ax = plt.subplots(figsize=(12, 5))
bars_miss    = ax.bar(x - width/2, miss_mean,    width, label="Misidentified Pipe 1 (n=24)",
                      color="#e74c3c", alpha=0.85, edgecolor="white")
bars_correct = ax.bar(x + width/2, correct_mean, width, label=f"Correctly ID'd Pipe 1 (n={len(miss_rows)})",
                      color="#2ecc71", alpha=0.85, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(sensor_names, fontsize=11)
ax.set_ylabel("Mean |Block 3 Activation|", fontsize=11)
ax.set_title("ST-GCN Block 3 Node Activation: Misidentified vs Correctly Identified Pipe 1\n"
             "(S10-A model, averaged over all sliding windows per scenario)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)

# Annotate ratio above each group
for i, (m, c) in enumerate(zip(miss_mean, correct_mean)):
    ratio = m / (c + 1e-8)
    ax.text(i, max(m, c) + 0.002, f"{ratio:.2f}×",
            ha="center", va="bottom", fontsize=7.5, color="#555555")

plt.tight_layout()
plot_path = OUT_DIR / "block3_node_activations.png"
fig.savefig(plot_path, dpi=150)
plt.close()
print(f"\nPlot saved: {plot_path}")

# ---------------------------------------------------------------------------
# Also plot per-group activation profiles (each scenario as a thin line)
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for ax, activations, title, colour in [
    (axes[0], miss_activations,    "Misidentified Pipe 1 (n=24)",         "#e74c3c"),
    (axes[1], correct_activations, f"Correctly ID'd Pipe 1 (n={len(miss_rows)})", "#2ecc71"),
]:
    for a in activations:
        ax.plot(sensor_names, a, color=colour, alpha=0.25, linewidth=0.8)
    ax.plot(sensor_names, np.stack(activations).mean(axis=0),
            color=colour, linewidth=2.5, label="Mean")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Sensor node", fontsize=9)
    ax.set_ylabel("Mean |Block 3 Activation|", fontsize=9)
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)

fig2.suptitle("Block 3 Node Activation Profiles — S10-A  (Pipe 1 scenarios)", fontsize=11)
plt.tight_layout()
profile_path = OUT_DIR / "block3_activation_profiles.png"
fig2.savefig(profile_path, dpi=150)
plt.close()
print(f"Profiles plot saved: {profile_path}")

print("\n[DONE]  All outputs in:", str(OUT_DIR))
