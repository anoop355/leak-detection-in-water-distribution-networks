"""
evaluate_v3_focused.py

ST-GCN Autoencoder V3 reconstruction on the same 12 focused test_dataset scenarios
used by run_ekf_focused_eval.py, for direct head-to-head comparison.

Selected scenarios:
  - 2 no-leak
  - 2 leak per pipe (pipes 1-5) = 10 leak scenarios

Outputs:
  v3_focused_eval_results.csv
  v3_focused_eval_plots/
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
BUNDLE_PATH  = "stgcn_autoencoder_v3.pt"
DATASET_ROOT = "test_dataset/scenarios"
MANIFEST     = "test_dataset/manifests/manifest.csv"
EVAL_CSV     = "v3_focused_eval_results.csv"
PLOT_DIR     = "v3_focused_eval_plots"

N_NOLEAK_PICK = 2
N_LEAK_PICK   = 2   # per pipe

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

UNMONITORED_SENSORS = ["P2", "P3", "P5", "P6", "Q2a", "Q4a", "Q5a"]

# ============================================================
# MODEL  (must match train_stgcn_autoencoder_v3.py)
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
        x = self.lin(x)
        x = self.ln(x)
        return self.act(x)


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


class STGCNAutoencoder(nn.Module):
    def __init__(self, adj_matrix, node_feats=1,
                 hidden_1=16, hidden_2=32, kernel_size=5, dropout=0.25):
        super().__init__()
        self.encoder_block1 = STBlock(node_feats, hidden_1, adj_matrix,
                                      kernel_size, dilation=1, dropout=dropout)
        self.encoder_block2 = STBlock(hidden_1, hidden_2, adj_matrix,
                                      kernel_size, dilation=2, dropout=dropout)
        self.lstm = nn.LSTM(input_size=hidden_2, hidden_size=hidden_2,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.recon_head = nn.Linear(hidden_2, node_feats)

    def forward(self, x_masked):
        z = self.encoder_block1(x_masked)
        z = self.encoder_block2(z)
        B, T, N, H = z.shape
        z_flat, _  = self.lstm(z.reshape(B * N, T, H))
        z_dec      = z_flat.reshape(B, T, N, H)
        return self.recon_head(z_dec)


# ============================================================
# SCENARIO SELECTION  (identical to EKF focused eval)
# ============================================================
def select_scenarios(manifest_path, dataset_root):
    df = pd.read_csv(manifest_path)
    selected = []

    for _, row in df[df["label_detection"] == 0].head(N_NOLEAK_PICK).iterrows():
        folder = os.path.join(dataset_root, f"scenario_{int(row['scenario_id']):05d}")
        if os.path.isfile(os.path.join(folder, "data.csv")):
            selected.append({
                "folder": folder, "scenario_id": int(row["scenario_id"]),
                "label_detection": 0, "label_pipe": -1, "group": "no-leak",
            })

    for pipe in range(1, 6):
        pipe_rows = df[(df["label_detection"] == 1) & (df["label_pipe"] == pipe)].head(N_LEAK_PICK)
        for _, row in pipe_rows.iterrows():
            folder = os.path.join(dataset_root, f"scenario_{int(row['scenario_id']):05d}")
            if os.path.isfile(os.path.join(folder, "data.csv")):
                selected.append({
                    "folder": folder, "scenario_id": int(row["scenario_id"]),
                    "label_detection": 1, "label_pipe": pipe, "group": f"pipe-{pipe}",
                })

    return selected


# ============================================================
# HELPERS
# ============================================================
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)

def compute_metrics(y_true, y_pred):
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2


# ============================================================
# INFERENCE
# ============================================================
@torch.no_grad()
def reconstruct_scenario(model, raw_signals, mu, sigma, unmonitored_idx, window):
    T, N = raw_signals.shape
    feats_norm = ((raw_signals[:, :, None] - mu[None]) / (sigma[None] + 1e-8)
                  ).squeeze(-1).astype(np.float32)   # (T, N)

    n_windows = max(0, (T - window) // 1 + 1)
    if n_windows == 0:
        # T < window: pad with a single window of zeros
        pad = np.zeros((window, N), dtype=np.float32)
        pad[:T] = feats_norm
        windows = pad[None, :, :, None]   # (1, W, N, 1)
        n_windows = 1
        T_orig = T
    else:
        windows = np.stack([feats_norm[w: w + window] for w in range(n_windows)],
                           axis=0)[:, :, :, None]    # (n_win, W, N, 1)
        T_orig = T

    x_t    = torch.tensor(windows, dtype=torch.float32, device=DEVICE)
    masked = x_t.clone()
    masked[:, :, unmonitored_idx, :] = 0.0

    model.eval()
    recon_norm = model(masked).cpu().numpy()          # (n_win, W, N, 1)

    recon_sum   = np.zeros((T, N, 1), dtype=np.float64)
    recon_count = np.zeros((T, N, 1), dtype=np.float64)
    for w in range(n_windows):
        end = min(w + window, T)
        recon_sum[w:end]   += recon_norm[w, :end - w]
        recon_count[w:end] += 1.0

    recon_norm_full = (recon_sum / np.maximum(recon_count, 1.0)
                       ).squeeze(-1).astype(np.float32)   # (T, N)
    return recon_norm_full * sigma[:, 0] + mu[:, 0]       # denormalised (T, N)


# ============================================================
# PLOTTING
# ============================================================
def plot_scenario(scenario_id, group, raw, recon_raw, unmonitored_idx, sensor_names, plot_dir):
    T      = raw.shape[0]
    t_axis = np.arange(T) * 15

    fig, axes = plt.subplots(len(unmonitored_idx), 1,
                             figsize=(10, 2.5 * len(unmonitored_idx)), sharex=True)
    if len(unmonitored_idx) == 1:
        axes = [axes]

    for ax, node_idx in zip(axes, unmonitored_idx):
        name = sensor_names[node_idx]
        ax.plot(t_axis, raw[:, node_idx],      "b-",  linewidth=1.5, label="Actual")
        ax.plot(t_axis, recon_raw[:, node_idx], "r--", linewidth=1.5, label="V3 AE")
        ax.set_ylabel(name, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (min)")
    fig.suptitle(f"V3 AE | Scenario {scenario_id}  [{group}]", fontsize=11, y=1.01)
    fig.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    out = os.path.join(plot_dir, f"scenario_{scenario_id:05d}_{group}.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# SUMMARY HELPER
# ============================================================
def print_group_summary(group_name, rows, unmon_names):
    print(f"\n  Group: {group_name}  (n={len(rows)})")
    print(f"    {'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}")
    print("    " + "-" * 40)
    for sensor in unmon_names:
        maes  = [r[f"{sensor}_mae"]  for r in rows]
        rmses = [r[f"{sensor}_rmse"] for r in rows]
        r2s   = [r[f"{sensor}_r2"]   for r in rows]
        print(f"    {sensor:<8}  {np.mean(maes):>10.4f}  "
              f"{np.mean(rmses):>10.4f}  {np.mean(r2s):>8.4f}")
    all_maes = [r[f"{s}_mae"] for r in rows for s in unmon_names]
    print(f"    {'OVERALL':<8}  {np.mean(all_maes):>10.4f}")


# ============================================================
# MAIN
# ============================================================
def main():
    if not os.path.isfile(BUNDLE_PATH):
        raise FileNotFoundError(f"Model bundle not found: {BUNDLE_PATH}\n"
                                "Run train_stgcn_autoencoder_v3.py first.")

    print(f"Loading bundle: {BUNDLE_PATH}")
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
    print(f"Monitored    : {[sensor_names[i] for i in monitored_idx]}")
    print(f"Unmonitored  : {unmon_names}")
    print(f"Window       : {window}")

    model = STGCNAutoencoder(
        adj_matrix=adj, node_feats=node_feats,
        hidden_1=hidden_1, hidden_2=hidden_2,
        kernel_size=kernel_size, dropout=dropout,
    ).to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    print("Model loaded.")

    scenarios = select_scenarios(MANIFEST, DATASET_ROOT)
    print(f"\nSelected {len(scenarios)} scenarios:")
    for s in scenarios:
        print(f"  scenario_{s['scenario_id']:05d}  group={s['group']}")

    all_rows     = []
    rows_by_group = {}

    for idx, scn in enumerate(scenarios, 1):
        folder = scn["folder"]
        scn_id = scn["scenario_id"]
        group  = scn["group"]

        print(f"\n[{idx}/{len(scenarios)}] Scenario {scn_id:05d}  ({group}) ...")

        raw_df = pd.read_csv(os.path.join(folder, "data.csv"))
        raw    = raw_df[sensor_names].to_numpy(dtype=np.float32)   # (T, N)

        recon_raw = reconstruct_scenario(model, raw, mu, sigma, unmonitored_idx, window)

        row = {
            "scenario_id":     scn_id,
            "group":           group,
            "label_detection": scn["label_detection"],
            "label_pipe":      scn["label_pipe"],
        }
        for node_idx, name in zip(unmonitored_idx, unmon_names):
            mae, rmse, r2 = compute_metrics(raw[:, node_idx], recon_raw[:, node_idx])
            row[f"{name}_mae"]  = round(mae,  6)
            row[f"{name}_rmse"] = round(rmse, 6)
            row[f"{name}_r2"]   = round(r2,   6)
            print(f"  {name:<8}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

        all_rows.append(row)
        rows_by_group.setdefault(group, []).append(row)

        out = plot_scenario(scn_id, group, raw, recon_raw, unmonitored_idx, sensor_names, PLOT_DIR)
        print(f"  Plot -> {out}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    pd.DataFrame(all_rows).to_csv(EVAL_CSV, index=False)
    print(f"\n[OK] Per-scenario results -> {EVAL_CSV}")

    # ── Per-scenario table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-SCENARIO RESULTS")
    print("=" * 70)
    hdr = f"{'ScnID':>8}  {'Group':<10}  " + "  ".join(
        f"{s+'_MAE':>10}" for s in unmon_names
    ) + f"  {'OverallMAE':>11}"
    print(hdr)
    print("-" * len(hdr))
    for row in all_rows:
        maes = [row[f"{s}_mae"] for s in unmon_names]
        print(f"{row['scenario_id']:>8}  {row['group']:<10}  " +
              "  ".join(f"{m:>10.4f}" for m in maes) +
              f"  {np.mean(maes):>11.4f}")

    # ── Grouped summaries ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GROUPED SUMMARIES")
    print("=" * 70)
    for g in ["no-leak"] + [f"pipe-{p}" for p in range(1, 6)]:
        if g in rows_by_group:
            print_group_summary(g, rows_by_group[g], unmon_names)
    print_group_summary("ALL", all_rows, unmon_names)

    # ── Leak / No-Leak ratio ──────────────────────────────────────────────────
    if "no-leak" in rows_by_group:
        leak_rows = [r for g, rs in rows_by_group.items() if g != "no-leak" for r in rs]
        if leak_rows:
            print("\n--- Leak/No-Leak MAE ratio per sensor ---")
            for sensor in unmon_names:
                nl = np.mean([r[f"{sensor}_mae"] for r in rows_by_group["no-leak"]])
                lk = np.mean([r[f"{sensor}_mae"] for r in leak_rows])
                ratio = lk / nl if nl > 1e-10 else float("nan")
                flag  = " [anomaly signal present]" if ratio > 1.05 else ""
                print(f"  {sensor:<8}: {ratio:.3f}{flag}")

    print(f"\n[OK] Plots saved -> {PLOT_DIR}/")
    print("Focused V3 evaluation complete.")


if __name__ == "__main__":
    main()
