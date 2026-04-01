"""
evaluate_stgcn_autoencoder.py

Standalone evaluation of the ST-GCN autoencoder on the held-out test set.

Metrics are computed in *original (denormalised) units* on the 7 unmonitored
nodes only — the 3 monitored nodes are excluded because their values are known.

Outputs
-------
  autoencoder_eval_results.csv          per-scenario, per-node MAE and R²
  autoencoder_reconstruction_plots/     time-series comparison plots
"""

import os
import json
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless rendering
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ============================================================
# PATHS  —  edit if needed
# ============================================================
BUNDLE_PATH    = "stgcn_autoencoder.pt"
DATASET_ROOT   = "stgcn_dataset/scenarios"
MANIFEST_TEST  = "stgcn_dataset/manifests/manifest_test.csv"
EVAL_CSV_PATH  = "autoencoder_eval_results.csv"
PLOT_DIR       = "autoencoder_reconstruction_plots"

NUM_SAMPLE_NOLEAK = 3   # no-leak scenarios to plot
NUM_SAMPLE_LEAK   = 3   # leak scenarios to plot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL COMPONENTS  (must match train_stgcn_autoencoder.py)
# ============================================================
class TemporalConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_ch, out_channels=out_ch,
            kernel_size=(1, kernel_size), padding=(0, pad), dilation=(1, dilation)
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
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
        x = self.lin(x)
        x = self.ln(x)
        x = self.act(x)
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
        y = self.temp(x)
        y = self.graph(y)
        y = self.dropout(y)
        y = y + residual
        y = self.out_act(y)
        return y


class STGCNAutoencoder(nn.Module):
    def __init__(self, adj_matrix, node_feats=1,
                 hidden_1=16, hidden_2=32, kernel_size=5, dropout=0.25):
        super().__init__()
        self.encoder_block1 = STBlock(
            in_ch=node_feats, out_ch=hidden_1,
            adj_matrix=adj_matrix, kernel_size=kernel_size, dilation=1, dropout=dropout
        )
        self.encoder_block2 = STBlock(
            in_ch=hidden_1, out_ch=hidden_2,
            adj_matrix=adj_matrix, kernel_size=kernel_size, dilation=2, dropout=dropout
        )
        self.lstm = nn.LSTM(
            input_size=hidden_2, hidden_size=hidden_2,
            num_layers=2, batch_first=True, dropout=0.2
        )
        self.recon_head = nn.Linear(hidden_2, node_feats)

    def forward(self, x_masked):
        z = self.encoder_block1(x_masked)
        z = self.encoder_block2(z)
        B, T, N, H = z.shape
        z_flat  = z.reshape(B * N, T, H)
        z_dec, _ = self.lstm(z_flat)
        z_dec   = z_dec.reshape(B, T, N, H)
        return self.recon_head(z_dec)

# ============================================================
# HELPERS
# ============================================================
def load_labels(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_no_leak(labels: dict) -> bool:
    return int(labels.get("label_detection", 0)) == 0

def folders_from_manifest(manifest_path: str, dataset_root: str):
    df = pd.read_csv(manifest_path)
    folders = []
    for scn_id in df["scenario_id"].values:
        path = os.path.join(dataset_root, f"scenario_{int(scn_id):05d}")
        if os.path.isfile(os.path.join(path, "data.csv")) and \
           os.path.isfile(os.path.join(path, "labels.json")):
            folders.append(path)
    return folders

def read_signals_all(folder: str, sensor_names) -> np.ndarray:
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    return df[sensor_names].to_numpy(dtype=np.float32)  # (T, N)

def denormalise(arr_norm: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """arr_norm: (..., N)  mu/sigma: (N, 1) or (N,)."""
    mu_flat    = mu.squeeze(-1)    if mu.ndim > 1    else mu
    sigma_flat = sigma.squeeze(-1) if sigma.ndim > 1 else sigma
    return arr_norm * sigma_flat + mu_flat

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)

# ============================================================
# INFERENCE — single scenario
# ============================================================
@torch.no_grad()
def reconstruct_scenario(model, raw_signals, mu, sigma,
                         monitored_idx, unmonitored_idx, window):
    """
    raw_signals : (T, N) float32 — original values
    Returns
      recon_raw  : (T, N) float32 — denormalised reconstruction
    """
    T, N = raw_signals.shape
    feats = raw_signals[:, :, None]                      # (T, N, 1)
    feats_norm = (feats - mu[None]) / (sigma[None] + 1e-8)
    feats_norm = feats_norm.astype(np.float32)

    # Build windows (T=12 -> 1 window)
    n_windows = max(0, (T - window) // 1 + 1)
    windows = []
    for w in range(n_windows):
        s, e = w, w + window
        windows.append(feats_norm[s:e])                  # (W, N, 1)
    x = np.stack(windows, axis=0)                        # (n_windows, W, N, 1)

    x_t = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    masked = x_t.clone()
    masked[:, :, unmonitored_idx, :] = 0.0

    model.eval()
    recon_norm = model(masked).cpu().numpy()             # (n_windows, W, N, 1)

    # Reconstruct full timeline from windows
    # Since stride=1, we use simple averaging for overlapping windows.
    recon_sum   = np.zeros((T, N, 1), dtype=np.float64)
    recon_count = np.zeros((T, N, 1), dtype=np.float64)
    for w in range(n_windows):
        s, e = w, w + window
        recon_sum[s:e]   += recon_norm[w]
        recon_count[s:e] += 1.0
    recon_count = np.maximum(recon_count, 1.0)
    recon_norm_full = (recon_sum / recon_count).squeeze(-1).astype(np.float32)   # (T, N)

    # Denormalise
    recon_raw = denormalise(recon_norm_full, mu.squeeze(-1), sigma.squeeze(-1))  # (T, N)
    return recon_raw

# ============================================================
# PLOTTING
# ============================================================
def plot_scenario(scenario_id, actual_raw, recon_raw,
                  unmonitored_idx, sensor_names, label_str, plot_dir):
    T = actual_raw.shape[0]
    t_axis = np.arange(T) * 15   # minutes

    n_unmon = len(unmonitored_idx)
    fig, axes = plt.subplots(n_unmon, 1, figsize=(10, 2.5 * n_unmon), sharex=True)
    if n_unmon == 1:
        axes = [axes]

    for ax, node_idx in zip(axes, unmonitored_idx):
        name = sensor_names[node_idx]
        ax.plot(t_axis, actual_raw[:, node_idx], "b-",  linewidth=1.5, label="Actual")
        ax.plot(t_axis, recon_raw[:, node_idx],  "r--", linewidth=1.5, label="Reconstructed")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.suptitle(f"Scenario {scenario_id}  [{label_str}]", fontsize=11, y=1.01)
    fig.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, f"scenario_{scenario_id}_{label_str}.png")
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path

# ============================================================
# MAIN
# ============================================================
def main():
    # ----------------------------------------------------------
    # Load bundle
    # ----------------------------------------------------------
    if not os.path.isfile(BUNDLE_PATH):
        raise FileNotFoundError(
            f"Bundle not found: {BUNDLE_PATH}\n"
            "Run train_stgcn_autoencoder.py first."
        )

    print(f"Loading bundle from {BUNDLE_PATH} ...")
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)

    adj             = bundle["adjacency"]
    mu              = bundle["mu"]               # (N, 1)
    sigma           = bundle["sigma"]            # (N, 1)
    sensor_names    = bundle["sensor_names"]
    monitored_idx   = bundle["monitored_idx"]
    unmonitored_idx = bundle["unmonitored_idx"]
    window          = bundle["window"]
    node_feats      = bundle["node_feats"]
    hidden_1        = bundle["hidden_1"]
    hidden_2        = bundle["hidden_2"]
    kernel_size     = bundle["kernel_size"]
    dropout         = bundle["dropout"]

    unmon_names = [sensor_names[i] for i in unmonitored_idx]
    print(f"Sensors      : {sensor_names}")
    print(f"Monitored    : {[sensor_names[i] for i in monitored_idx]}")
    print(f"Unmonitored  : {unmon_names}")

    # ----------------------------------------------------------
    # Rebuild model
    # ----------------------------------------------------------
    model = STGCNAutoencoder(
        adj_matrix=adj, node_feats=node_feats,
        hidden_1=hidden_1, hidden_2=hidden_2,
        kernel_size=kernel_size, dropout=dropout
    ).to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    print("Model loaded.")

    # ----------------------------------------------------------
    # Test scenarios
    # ----------------------------------------------------------
    test_folders = folders_from_manifest(MANIFEST_TEST, DATASET_ROOT)
    if len(test_folders) == 0:
        raise RuntimeError(f"No test scenarios found via {MANIFEST_TEST}")
    print(f"Test scenarios: {len(test_folders)}")

    # ----------------------------------------------------------
    # Run evaluation
    # ----------------------------------------------------------
    rows = []

    no_leak_sample_folders = []
    leak_sample_folders    = []

    all_mae_noleak  = {n: [] for n in unmon_names}
    all_mae_leak    = {n: [] for n in unmon_names}

    for folder in test_folders:
        scn_id  = os.path.basename(folder)
        labels  = load_labels(os.path.join(folder, "labels.json"))
        is_nl   = is_no_leak(labels)
        label_str = "noleak" if is_nl else "leak"

        raw = read_signals_all(folder, sensor_names)   # (T, N)

        recon_raw = reconstruct_scenario(
            model, raw, mu, sigma, monitored_idx, unmonitored_idx, window
        )

        row = {"scenario_id": scn_id, "label": label_str}

        for node_idx, name in zip(unmonitored_idx, unmon_names):
            y_true = raw[:, node_idx]
            y_pred = recon_raw[:, node_idx]

            mae  = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            r2   = r2_score(y_true, y_pred)

            row[f"{name}_mae"]  = round(mae,  6)
            row[f"{name}_rmse"] = round(rmse, 6)
            row[f"{name}_r2"]   = round(r2,   6)

            if is_nl:
                all_mae_noleak[name].append(mae)
            else:
                all_mae_leak[name].append(mae)

        rows.append(row)

        # Collect sample scenarios for plotting
        if is_nl and len(no_leak_sample_folders) < NUM_SAMPLE_NOLEAK:
            no_leak_sample_folders.append((scn_id, raw, recon_raw, label_str))
        elif not is_nl and len(leak_sample_folders) < NUM_SAMPLE_LEAK:
            leak_sample_folders.append((scn_id, raw, recon_raw, label_str))

    # ----------------------------------------------------------
    # Save CSV
    # ----------------------------------------------------------
    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(EVAL_CSV_PATH, index=False)
    print(f"\n[OK] Saved per-scenario results -> {EVAL_CSV_PATH}")

    # ----------------------------------------------------------
    # Summary metrics
    # ----------------------------------------------------------
    print("\n=== Per-Node Test Metrics (denormalised, unmonitored nodes only) ===")
    print(f"{'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}")
    print("-" * 42)

    node_maes = {}
    for name in unmon_names:
        mae_col  = eval_df[f"{name}_mae"].values
        rmse_col = eval_df[f"{name}_rmse"].values
        r2_col   = eval_df[f"{name}_r2"].values

        mean_mae  = float(np.mean(mae_col))
        mean_rmse = float(np.mean(rmse_col))
        mean_r2   = float(np.mean(r2_col))
        node_maes[name] = mean_mae

        print(f"{name:<8}  {mean_mae:>10.4f}  {mean_rmse:>10.4f}  {mean_r2:>8.4f}")

    overall_mae = float(np.mean(list(node_maes.values())))
    worst_node  = max(node_maes, key=node_maes.get)
    print(f"\nMean MAE across all unmonitored nodes : {overall_mae:.4f}")
    print(f"Worst-reconstructed node              : {worst_node} (MAE={node_maes[worst_node]:.4f})")

    # ----------------------------------------------------------
    # Leak vs. no-leak MAE comparison
    # ----------------------------------------------------------
    print("\n=== Leak vs. No-Leak Reconstruction MAE ===")
    print(f"{'Sensor':<8}  {'No-Leak MAE':>12}  {'Leak MAE':>10}")
    print("-" * 36)
    for name in unmon_names:
        nl_mae = float(np.mean(all_mae_noleak[name])) if all_mae_noleak[name] else float("nan")
        l_mae  = float(np.mean(all_mae_leak[name]))   if all_mae_leak[name]   else float("nan")
        print(f"{name:<8}  {nl_mae:>12.4f}  {l_mae:>10.4f}")

    ratio_check = {}
    for name in unmon_names:
        if all_mae_noleak[name] and all_mae_leak[name]:
            nl = np.mean(all_mae_noleak[name])
            lk = np.mean(all_mae_leak[name])
            ratio_check[name] = lk / nl if nl > 1e-10 else float("inf")

    if ratio_check:
        print("\nLeak/NoLeak MAE ratio (>1 means leak scenarios are harder to reconstruct):")
        for name, ratio in ratio_check.items():
            flag = " [anomaly signal preserved]" if ratio > 1.05 else ""
            print(f"  {name:<8}: {ratio:.3f}{flag}")

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------
    os.makedirs(PLOT_DIR, exist_ok=True)
    sample_scenarios = no_leak_sample_folders + leak_sample_folders

    for scn_id, raw, recon_raw, label_str in sample_scenarios:
        out = plot_scenario(
            scn_id, raw, recon_raw,
            unmonitored_idx, sensor_names, label_str, PLOT_DIR
        )
        print(f"  Saved plot -> {out}")

    print(f"\n[OK] All reconstruction plots saved -> {PLOT_DIR}/")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
