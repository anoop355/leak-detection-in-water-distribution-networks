"""
This script analyses:
1. LAYER-BY-LAYER COSINE SIMILARITY
2. PCA OF THE Z EMBEDDING
3. TEMPORAL ATTENTION WEIGHTS

"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Settings

BUNDLE_PATH = Path("stgcn_placement_bundles/stgcn_bundle_S10-A.pt")
TEST_DIR    = Path("test_dataset/scenarios")
METRICS_CSV = Path("stgcn_placement_results/S10-A/evaluation/S10-A_per_scenario_metrics.csv")
OUT_DIR     = Path("stgcn_placement_results/S10-A/debug_internals")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

NUM_PIPES    = 5
PIPE_CLASSES = NUM_PIPES + 1

MISCLASSIFIED = {
    "scenario_00053", "scenario_00067", "scenario_00081",
    "scenario_00094", "scenario_00095", "scenario_00109",
    "scenario_00123", "scenario_00136", "scenario_00137",
    "scenario_00151", "scenario_00179", "scenario_00193",
}


# Model (v4 — temporal attention pool)

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
        weights = torch.softmax(self.attn(x_flat), dim=1)   # (B, T, 1)
        return (x_flat * weights).sum(dim=1), weights        # also return weights


class SingleLeakSTGCNv4Probe(nn.Module):
    """Same as production v4 but forward() also returns all intermediate tensors."""

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
        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        z, attn_weights = self.temporal_pool(h3)    # (B, head_in), (B, T, 1)
        pipe_logits = self.pipe_head(z)
        return {
            "h1": h1,                # (B, T, N, hidden_1)
            "h2": h2,                # (B, T, N, hidden_2)
            "h3": h3,                # (B, T, N, hidden_2)
            "z":  z,                 # (B, head_in)
            "attn_weights": attn_weights,   # (B, T, 1)
            "pipe_logits":  pipe_logits,    # (B, PIPE_CLASSES)
            "pipe_probs":   F.softmax(pipe_logits, dim=1),
        }


# I/O helpers

def load_model(bundle_path):
    bundle     = torch.load(str(bundle_path), map_location=DEVICE, weights_only=False)
    adj        = np.array(bundle["adjacency"], dtype=np.float32)
    num_nodes  = len(bundle["sensor_names"])
    hidden_1   = int(bundle.get("hidden_1",   16))
    hidden_2   = int(bundle.get("hidden_2",   32))
    kernel_sz  = int(bundle.get("kernel_size", 5))
    dropout    = float(bundle.get("dropout",  0.25))
    node_feats = int(bundle.get("node_feats",  2))

    model = SingleLeakSTGCNv4Probe(adj, num_nodes, hidden_1, hidden_2,
                                    kernel_sz, dropout, node_feats).to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle


def preprocess(raw, baseline, mu, sigma, node_feats=2):
    T      = raw.shape[0]
    base   = baseline[:T]
    ch     = [raw, raw - base]
    if node_feats == 3:
        diff = np.zeros_like(raw); diff[1:] = raw[1:] - raw[:-1]
        ch.append(diff)
    feats = np.stack(ch, axis=-1).astype(np.float32)
    feats = (feats - mu[None]) / (sigma[None] + 1e-8)
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)


def flatten_spatial_temporal(h):
    """Mean-pool over T and N to get a single vector per sample."""
    # h: (1, T, N, C)  ->  (C,)
    return h.squeeze(0).mean(dim=(0, 1)).cpu().numpy()

# Collect representations for all pipe-1 and pipe-4 scenarios

def collect_representations(model, bundle, scn_names):
    sensor_names = bundle["sensor_names"]
    mu           = np.array(bundle["mu"],                dtype=np.float32)
    sigma        = np.array(bundle["sigma"],             dtype=np.float32)
    baseline     = np.array(bundle["baseline_template"], dtype=np.float32)
    node_feats   = int(bundle.get("node_feats", 2))

    records = []
    for scn_name in scn_names:
        scn_dir    = TEST_DIR / scn_name
        data_path  = scn_dir / "data.csv"
        label_path = scn_dir / "labels.json"
        if not data_path.exists():
            continue

        labels       = json.loads(label_path.read_text(encoding="utf-8"))
        true_pipe_id = int(labels["label_pipe"])
        true_pos     = float(labels["label_position"])

        raw = pd.read_csv(data_path)[sensor_names].to_numpy(dtype=np.float32)
        x_t = preprocess(raw, baseline, mu, sigma, node_feats)

        with torch.no_grad():
            out = model(x_t)

        pred_pipe = int(out["pipe_logits"].argmax(dim=1).item()) + 1

        records.append({
            "scenario":       scn_name,
            "true_pipe":      true_pipe_id,
            "pred_pipe":      pred_pipe,
            "true_pos":       true_pos,
            "misclassified":  scn_name in MISCLASSIFIED,
            # Flat spatial-temporal means at each layer
            "h1_vec": flatten_spatial_temporal(out["h1"]),
            "h2_vec": flatten_spatial_temporal(out["h2"]),
            "h3_vec": flatten_spatial_temporal(out["h3"]),
            # z vector directly
            "z_vec":  out["z"].squeeze(0).cpu().numpy(),
            # Temporal attention weights: mean over batch, squeeze to (T,)
            "attn":   out["attn_weights"].squeeze(0).squeeze(-1).cpu().numpy(),
            # Pipe softmax
            "pipe_probs": out["pipe_probs"].squeeze(0).cpu().numpy(),
        })

    return records

# Analysis 1: layer-by-layer cosine similarity

def cosine_sim(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def analysis_layer_cosine(records):
    pipe1_correct   = [r for r in records if r["true_pipe"] == 1 and not r["misclassified"]]
    pipe4_correct   = [r for r in records if r["true_pipe"] == 4]
    pipe1_mis       = [r for r in records if r["misclassified"]]

    layers = ["h1_vec", "h2_vec", "h3_vec", "z_vec"]
    layer_labels = ["Block 1", "Block 2", "Block 3", "Z (pool)"]

    # Centroids
    centroids = {}
    for key, group in [("pipe1_correct", pipe1_correct), ("pipe4_correct", pipe4_correct)]:
        centroids[key] = {
            lay: np.mean([r[lay] for r in group], axis=0) for lay in layers
        }

    rows = []
    print("\n=== LAYER-BY-LAYER COSINE SIMILARITY ===")
    print(f"  (misclassified pipe-1 vs centroid of correct pipe-1 / correct pipe-4)\n")
    print(f"{'Scenario':<22} {'Layer':<12} {'sim->pipe1':>10} {'sim->pipe4':>10}  {'closer_to':>12}")
    print("-" * 70)

    for r in pipe1_mis:
        for lay, label in zip(layers, layer_labels):
            s1 = cosine_sim(r[lay], centroids["pipe1_correct"][lay])
            s4 = cosine_sim(r[lay], centroids["pipe4_correct"][lay])
            closer = "PIPE-4 !" if s4 > s1 else "pipe-1"
            print(f"  {r['scenario']:<20} {label:<12} {s1:>10.4f} {s4:>10.4f}  {closer:>12}")
            rows.append({
                "scenario":       r["scenario"],
                "true_pos":       r["true_pos"],
                "layer":          label,
                "sim_pipe1_centroid": round(s1, 6),
                "sim_pipe4_centroid": round(s4, 6),
                "closer_to_pipe4":    int(s4 > s1),
            })
        print()

    df = pd.DataFrame(rows)

    # Summary: at which layer does the majority first flip to pipe-4?
    print("\nSUMMARY — fraction of misclassified scenarios closer to pipe-4 centroid:")
    for lay, label in zip(layers, layer_labels):
        sub = df[df["layer"] == label]
        frac = sub["closer_to_pipe4"].mean()
        print(f"  {label:<12} : {frac:.2f}  ({int(sub['closer_to_pipe4'].sum())}/{len(sub)})")

    return df

# Analysis 2: PCA of z embeddings

def analysis_pca_z(records, out_dir):
    pipe1_correct = [r for r in records if r["true_pipe"] == 1 and not r["misclassified"]]
    pipe4_correct = [r for r in records if r["true_pipe"] == 4]
    pipe1_mis     = [r for r in records if r["misclassified"]]

    # Stack all z vectors
    all_records = pipe1_correct + pipe4_correct + pipe1_mis
    Z = np.stack([r["z_vec"] for r in all_records])

    pca = PCA(n_components=2, random_state=42)
    Z2  = pca.fit_transform(Z)

    n1c = len(pipe1_correct)
    n4c = len(pipe4_correct)
    n1m = len(pipe1_mis)

    z_p1c = Z2[:n1c]
    z_p4c = Z2[n1c:n1c + n4c]
    z_p1m = Z2[n1c + n4c:]

    var = pca.explained_variance_ratio_
    print(f"\n=== PCA of Z embeddings ===")
    print(f"  Explained variance: PC1={var[0]:.3f}, PC2={var[1]:.3f}, "
          f"total={sum(var):.3f}")

    # Distance: mean misclassified pipe-1 to centroid of pipe-1 vs pipe-4
    c_p1 = z_p1c.mean(axis=0)
    c_p4 = z_p4c.mean(axis=0)
    for i, r in enumerate(pipe1_mis):
        d1 = float(np.linalg.norm(z_p1m[i] - c_p1))
        d4 = float(np.linalg.norm(z_p1m[i] - c_p4))
        print(f"  {r['scenario']}  dist->pipe1_centroid={d1:.3f}  "
              f"dist->pipe4_centroid={d4:.3f}  "
              f"({'closer to PIPE-4' if d4 < d1 else 'closer to pipe-1'})")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(z_p1c[:, 0], z_p1c[:, 1], c="steelblue", alpha=0.35, s=18,
               label=f"Pipe 1 correct (n={n1c})")
    ax.scatter(z_p4c[:, 0], z_p4c[:, 1], c="darkorange", alpha=0.35, s=18,
               label=f"Pipe 4 correct (n={n4c})")
    ax.scatter(z_p1m[:, 0], z_p1m[:, 1], c="red", s=80, marker="X", zorder=5,
               label=f"Pipe 1 → predicted Pipe 4 (n={n1m})")
    # Mark centroids
    ax.scatter(*c_p1, c="steelblue",  s=200, marker="*", edgecolors="black",
               linewidths=0.8, zorder=6, label="Pipe-1 centroid")
    ax.scatter(*c_p4, c="darkorange", s=200, marker="*", edgecolors="black",
               linewidths=0.8, zorder=6, label="Pipe-4 centroid")

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=11)
    ax.set_title("PCA of Z embeddings — S10-A  (pipe-1 vs pipe-4)", fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "pca_z_embeddings.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"\n  [SAVED] {path}")

    # CSV
    csv_rows = []
    for i, r in enumerate(all_records):
        csv_rows.append({
            "scenario":      r["scenario"],
            "true_pipe":     r["true_pipe"],
            "pred_pipe":     r["pred_pipe"],
            "true_pos":      r["true_pos"],
            "group":         ("pipe1_misclassified" if r["misclassified"]
                              else f"pipe{r['true_pipe']}_correct"),
            "pc1": round(float(Z2[i, 0]), 6),
            "pc2": round(float(Z2[i, 1]), 6),
        })
    return pd.DataFrame(csv_rows)


# Analysis 3: Temporal attention weights

def analysis_temporal_attention(records, out_dir):
    pipe1_correct = [r for r in records if r["true_pipe"] == 1 and not r["misclassified"]]
    pipe4_correct = [r for r in records if r["true_pipe"] == 4]
    pipe1_mis     = [r for r in records if r["misclassified"]]

    T = len(pipe1_mis[0]["attn"])

    mean_p1c = np.mean([r["attn"] for r in pipe1_correct], axis=0)
    mean_p4c = np.mean([r["attn"] for r in pipe4_correct], axis=0)
    mean_p1m = np.mean([r["attn"] for r in pipe1_mis],     axis=0)

    print(f"\n=== TEMPORAL ATTENTION WEIGHTS (mean profiles) ===")
    print(f"  Window length T = {T} steps")

    # Identify which time steps have highest weight in the misclassified group
    top_k = min(10, T)
    top_idx = np.argsort(mean_p1m)[::-1][:top_k]
    print(f"  Top-{top_k} attended time steps in misclassified pipe-1 scenarios:")
    for idx in top_idx:
        diff_from_p4 = mean_p4c[idx] - mean_p1c[idx]
        print(f"    t={idx:3d}  mis={mean_p1m[idx]:.5f}  "
              f"pipe1_ok={mean_p1c[idx]:.5f}  pipe4_ok={mean_p4c[idx]:.5f}  "
              f"d(p4-p1)={diff_from_p4:+.5f}")

    # Wasserstein-like shift between attention profiles
    from scipy.stats import wasserstein_distance
    wd_p1m_vs_p4 = wasserstein_distance(
        np.arange(T), np.arange(T), mean_p1m, mean_p4c)
    wd_p1m_vs_p1 = wasserstein_distance(
        np.arange(T), np.arange(T), mean_p1m, mean_p1c)
    print(f"\n  Earth Mover's Distance:")
    print(f"    misclassified-pipe1 vs correct-pipe4  : {wd_p1m_vs_p4:.4f}")
    print(f"    misclassified-pipe1 vs correct-pipe1  : {wd_p1m_vs_p1:.4f}")

    # Plot
    t = np.arange(T)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, mean_p1c, color="steelblue",  lw=1.5, label="Pipe-1 correct (mean)")
    ax.plot(t, mean_p4c, color="darkorange", lw=1.5, label="Pipe-4 correct (mean)")
    ax.plot(t, mean_p1m, color="red",        lw=2.0, ls="--",
            label="Pipe-1 misclassified (mean)")
    ax.set_xlabel("Time step", fontsize=11)
    ax.set_ylabel("Attention weight", fontsize=11)
    ax.set_title("Mean temporal attention profile — S10-A", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "temporal_attention_profile.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"\n  [SAVED] {path}")

    # Per-scenario CSV
    csv_rows = []
    for r in pipe1_correct + pipe4_correct + pipe1_mis:
        for t_idx, w in enumerate(r["attn"]):
            csv_rows.append({
                "scenario":  r["scenario"],
                "true_pipe": r["true_pipe"],
                "pred_pipe": r["pred_pipe"],
                "group":     ("pipe1_misclassified" if r["misclassified"]
                              else f"pipe{r['true_pipe']}_correct"),
                "timestep":  t_idx,
                "attn_weight": round(float(w), 8),
            })
    return pd.DataFrame(csv_rows)

# Main

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Bundle : {BUNDLE_PATH}")

    model, bundle = load_model(BUNDLE_PATH)

    # Load the per-scenario metrics to get all pipe-1 and pipe-4 scenarios
    metrics = pd.read_csv(METRICS_CSV)
    leak_scenarios = metrics[metrics["true_detect"] == 1]

    pipe1_scns = sorted(leak_scenarios[leak_scenarios["true_pipe"] == 1]["scenario"].tolist())
    pipe4_scns = sorted(leak_scenarios[leak_scenarios["true_pipe"] == 4]["scenario"].tolist())

    all_scns = sorted(set(pipe1_scns + pipe4_scns))
    print(f"\nCollecting representations for {len(all_scns)} scenarios "
          f"(pipe1={len(pipe1_scns)}, pipe4={len(pipe4_scns)}) ...")
    records = collect_representations(model, bundle, all_scns)
    print(f"  Done — {len(records)} records collected.")

    # 1. Layer cosine similarity
    df_cos = analysis_layer_cosine(records)
    df_cos.to_csv(OUT_DIR / "layer_cosine_similarity.csv", index=False)
    print(f"\n  [SAVED] {OUT_DIR / 'layer_cosine_similarity.csv'}")

    # 2. PCA of z embeddings
    df_pca = analysis_pca_z(records, OUT_DIR)
    df_pca.to_csv(OUT_DIR / "pca_z_embeddings.csv", index=False)
    print(f"  [SAVED] {OUT_DIR / 'pca_z_embeddings.csv'}")

    # 3. Temporal attention
    df_attn = analysis_temporal_attention(records, OUT_DIR)
    df_attn.to_csv(OUT_DIR / "temporal_attention_weights.csv", index=False)
    print(f"  [SAVED] {OUT_DIR / 'temporal_attention_weights.csv'}")

    print("\n[DONE]  All outputs in:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
