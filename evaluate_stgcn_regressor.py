"""
evaluate_stgcn_regressor.py

Standalone evaluation of the direct ST-GCN regressor on the stgcn_dataset_v2 test set.

Outputs:
  stgcn_regressor_eval_results.csv
  stgcn_regressor_reconstruction_plots/
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================
# PATHS
# ============================================================
BUNDLE_PATH   = "stgcn_regressor.pt"
DATASET_ROOT  = "stgcn_dataset_v2/scenarios"
MANIFEST_TEST = "stgcn_dataset_v2/manifests/manifest_test.csv"
EVAL_CSV_PATH = "stgcn_regressor_eval_results.csv"
PLOT_DIR      = "stgcn_regressor_reconstruction_plots"

NUM_SAMPLE_NOLEAK = 3
NUM_SAMPLE_LEAK   = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL  (must match train_stgcn_regressor.py)
# ============================================================
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
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp     = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph    = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout  = nn.Dropout(dropout)
        self.out_act  = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.res_proj(x)
        y = self.temp(x)
        y = self.graph(y)
        y = self.dropout(y)
        y = y + residual
        return self.out_act(y)


class STGCNRegressor(nn.Module):
    def __init__(self, adj_matrix, node_feats=1,
                 hidden_1=16, hidden_2=32, kernel_size=5, dropout=0.25):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, dilation=1, dropout=dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, dilation=2, dropout=dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, dilation=4, dropout=dropout)
        self.head   = nn.Linear(hidden_2, node_feats)

    def forward(self, x_masked):
        z = self.block1(x_masked)
        z = self.block2(z)
        z = self.block3(z)
        return self.head(z)

# ============================================================
# HELPERS
# ============================================================
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_no_leak(labels):
    return int(labels.get("label_detection", 0)) == 0

def folders_from_manifest(manifest_path, dataset_root):
    df = pd.read_csv(manifest_path)
    out = []
    for scn_id in df["scenario_id"].values:
        path = os.path.join(dataset_root, f"scenario_{int(scn_id):05d}")
        if (os.path.isfile(os.path.join(path, "data.csv")) and
                os.path.isfile(os.path.join(path, "labels.json"))):
            out.append(path)
    return out

def read_signals_all(folder, sensor_names):
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    return df[sensor_names].to_numpy(dtype=np.float32)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)

# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def reconstruct_scenario(model, raw_signals, mu, sigma,
                         unmonitored_idx, window):
    T, N = raw_signals.shape
    feats_norm = ((raw_signals[:, :, None] - mu[None]) / (sigma[None] + 1e-8)
                  ).astype(np.float32)                   # (T, N, 1)

    n_windows = max(0, (T - window) // 1 + 1)
    windows   = np.stack([feats_norm[w: w + window] for w in range(n_windows)])
                                                         # (n_win, W, N, 1)
    x_t    = torch.tensor(windows, dtype=torch.float32, device=DEVICE)
    masked = x_t.clone()
    masked[:, :, unmonitored_idx, :] = 0.0

    model.eval()
    recon_norm = model(masked).cpu().numpy()              # (n_win, W, N, 1)

    # Overlap-add averaging
    recon_sum   = np.zeros((T, N, 1), dtype=np.float64)
    recon_count = np.zeros((T, N, 1), dtype=np.float64)
    for w in range(n_windows):
        recon_sum[w: w + window]   += recon_norm[w]
        recon_count[w: w + window] += 1.0
    recon_norm_full = (recon_sum / np.maximum(recon_count, 1.0)
                       ).squeeze(-1).astype(np.float32)  # (T, N)

    return recon_norm_full * sigma[:, 0] + mu[:, 0]      # denormalised (T, N)

# ============================================================
# PLOTTING
# ============================================================
def plot_scenario(scenario_id, actual_raw, recon_raw,
                  unmonitored_idx, sensor_names, label_str, plot_dir):
    T       = actual_raw.shape[0]
    t_axis  = np.arange(T) * 15
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
    fig.suptitle(f"Regressor | Scenario {scenario_id}  [{label_str}]", fontsize=11, y=1.01)
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
    if not os.path.isfile(BUNDLE_PATH):
        raise FileNotFoundError(
            f"Bundle not found: {BUNDLE_PATH}\n"
            "Run train_stgcn_regressor.py first."
        )

    print(f"Loading bundle from {BUNDLE_PATH} ...")
    bundle = torch.load(BUNDLE_PATH, map_location="cpu", weights_only=False)

    adj             = bundle["adjacency"]
    mu              = bundle["mu"]
    sigma           = bundle["sigma"]
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

    model = STGCNRegressor(
        adj_matrix=adj, node_feats=node_feats,
        hidden_1=hidden_1, hidden_2=hidden_2,
        kernel_size=kernel_size, dropout=dropout
    ).to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    print("Model loaded.")

    test_folders = folders_from_manifest(MANIFEST_TEST, DATASET_ROOT)
    if not test_folders:
        raise RuntimeError(f"No test scenarios found via {MANIFEST_TEST}")
    print(f"Test scenarios: {len(test_folders)}")

    rows            = []
    no_leak_samples = []
    leak_samples    = []
    all_mae_noleak  = {n: [] for n in unmon_names}
    all_mae_leak    = {n: [] for n in unmon_names}

    for folder in test_folders:
        scn_id    = os.path.basename(folder)
        labels    = load_labels(os.path.join(folder, "labels.json"))
        is_nl     = is_no_leak(labels)
        label_str = "noleak" if is_nl else "leak"

        raw       = read_signals_all(folder, sensor_names)
        recon_raw = reconstruct_scenario(model, raw, mu, sigma,
                                         unmonitored_idx, window)

        row = {"scenario_id": scn_id, "label": label_str}
        for node_idx, name in zip(unmonitored_idx, unmon_names):
            y_true = raw[:, node_idx]
            y_pred = recon_raw[:, node_idx]
            mae    = float(np.mean(np.abs(y_true - y_pred)))
            rmse   = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            r2     = r2_score(y_true, y_pred)
            row[f"{name}_mae"]  = round(mae,  6)
            row[f"{name}_rmse"] = round(rmse, 6)
            row[f"{name}_r2"]   = round(r2,   6)
            (all_mae_noleak if is_nl else all_mae_leak)[name].append(mae)

        rows.append(row)
        if is_nl and len(no_leak_samples) < NUM_SAMPLE_NOLEAK:
            no_leak_samples.append((scn_id, raw, recon_raw, label_str))
        elif not is_nl and len(leak_samples) < NUM_SAMPLE_LEAK:
            leak_samples.append((scn_id, raw, recon_raw, label_str))

    eval_df = pd.DataFrame(rows)
    eval_df.to_csv(EVAL_CSV_PATH, index=False)
    print(f"\n[OK] Saved per-scenario results -> {EVAL_CSV_PATH}")

    print("\n=== Per-Node Test Metrics — ST-GCN Regressor (denormalised, unmonitored only) ===")
    print(f"{'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}")
    print("-" * 42)

    node_maes = {}
    for name in unmon_names:
        mean_mae  = float(np.mean(eval_df[f"{name}_mae"].values))
        mean_rmse = float(np.mean(eval_df[f"{name}_rmse"].values))
        mean_r2   = float(np.mean(eval_df[f"{name}_r2"].values))
        node_maes[name] = mean_mae
        print(f"{name:<8}  {mean_mae:>10.4f}  {mean_rmse:>10.4f}  {mean_r2:>8.4f}")

    overall_mae = float(np.mean(list(node_maes.values())))
    worst_node  = max(node_maes, key=node_maes.get)
    print(f"\nMean MAE across all unmonitored nodes : {overall_mae:.4f}")
    print(f"Worst-reconstructed node              : {worst_node} (MAE={node_maes[worst_node]:.4f})")

    print("\n=== Leak vs. No-Leak Reconstruction MAE ===")
    print(f"{'Sensor':<8}  {'No-Leak MAE':>12}  {'Leak MAE':>10}")
    print("-" * 36)
    ratio_check = {}
    for name in unmon_names:
        nl_mae = float(np.mean(all_mae_noleak[name])) if all_mae_noleak[name] else float("nan")
        l_mae  = float(np.mean(all_mae_leak[name]))   if all_mae_leak[name]   else float("nan")
        print(f"{name:<8}  {nl_mae:>12.4f}  {l_mae:>10.4f}")
        if all_mae_noleak[name] and all_mae_leak[name] and nl_mae > 1e-10:
            ratio_check[name] = l_mae / nl_mae

    if ratio_check:
        print("\nLeak/NoLeak MAE ratio (>1.05 means anomaly signal preserved):")
        for name, ratio in ratio_check.items():
            flag = " [anomaly signal preserved]" if ratio > 1.05 else ""
            print(f"  {name:<8}: {ratio:.3f}{flag}")

    os.makedirs(PLOT_DIR, exist_ok=True)
    for scn_id, raw, recon_raw, label_str in no_leak_samples + leak_samples:
        out = plot_scenario(scn_id, raw, recon_raw, unmonitored_idx,
                            sensor_names, label_str, PLOT_DIR)
        print(f"  Saved plot -> {out}")

    print(f"\n[OK] All reconstruction plots saved -> {PLOT_DIR}/")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()
