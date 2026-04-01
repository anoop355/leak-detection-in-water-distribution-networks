"""
pipeline_v3_stgcn.py

Full pipeline evaluation: V3 Autoencoder reconstruction -> ST-GCN S10-A detection/localisation.

On each of the 12 focused test_dataset scenarios:
  1. V3 Autoencoder reconstructs the 7 unmonitored sensors from masked input
  2. Reconstructed + actual monitored signals form the full 10-sensor array
  3. ST-GCN S10-A predicts: leak/no-leak, pipe location, leak size, position
  4. Predictions are compared against ground-truth labels

Outputs:
  pipeline_v3_stgcn_results.csv
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
AE_BUNDLE    = "stgcn_autoencoder_v3.pt"
STGCN_BUNDLE = "stgcn_placement_bundles/stgcn_bundle_S10-A.pt"
DATASET_ROOT = "test_dataset/scenarios"
MANIFEST     = "test_dataset/manifests/manifest.csv"
OUT_CSV      = "pipeline_v3_stgcn_results.csv"

N_NOLEAK_PICK = 2
N_LEAK_PICK   = 2

MONITORED   = ["P4", "Q1a", "Q3a"]
UNMONITORED = ["P2", "P3", "P5", "P6", "Q2a", "Q4a", "Q5a"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# AUTOENCODER MODEL  (must match train_stgcn_autoencoder_v3.py)
# ─────────────────────────────────────────────────────────────────────────────
class AE_TemporalConvLayer(nn.Module):
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


class AE_GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.einsum("ij,btjc->btic", self.A, x)
        return self.act(self.ln(self.lin(x)))


class AE_STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp     = AE_TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph    = AE_GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout  = nn.Dropout(dropout)
        self.out_act  = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        r = self.res_proj(x)
        y = self.graph(self.temp(x))
        y = self.dropout(y)
        return self.out_act(y + r)


class STGCNAutoencoder(nn.Module):
    def __init__(self, adj_matrix, node_feats=1, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25):
        super().__init__()
        self.encoder_block1 = AE_STBlock(node_feats, hidden_1, adj_matrix,
                                         kernel_size, dilation=1, dropout=dropout)
        self.encoder_block2 = AE_STBlock(hidden_1, hidden_2, adj_matrix,
                                         kernel_size, dilation=2, dropout=dropout)
        self.lstm = nn.LSTM(input_size=hidden_2, hidden_size=hidden_2,
                            num_layers=2, batch_first=True, dropout=0.2)
        self.recon_head = nn.Linear(hidden_2, node_feats)

    def forward(self, x_masked):
        z = self.encoder_block1(x_masked)
        z = self.encoder_block2(z)
        B, T, N, H = z.shape
        z_flat, _ = self.lstm(z.reshape(B * N, T, H))
        return self.recon_head(z_flat.reshape(B, T, N, H))


# ─────────────────────────────────────────────────────────────────────────────
# ST-GCN DETECTION MODEL  (must match train_stgcn_sensor_placement.py)
# ─────────────────────────────────────────────────────────────────────────────
PIPE_CLASSES  = 6
SIZE_CLASSES  = 4
PIPE_NONE_IDX = 5

class STGCN_TemporalConvLayer(nn.Module):
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


class STGCN_GraphConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x = torch.einsum("ij,btjc->btic", self.A, x)
        return self.act(self.ln(self.lin(x)))


class STGCN_STBlock(nn.Module):
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp     = STGCN_TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph    = STGCN_GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout  = nn.Dropout(dropout)
        self.out_act  = nn.ReLU()
        self.res_proj = nn.Linear(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

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


class SingleLeakSTGCN(nn.Module):
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        self.block1 = STGCN_STBlock(node_feats, hidden_1, adj_matrix, kernel_size, 1, dropout)
        self.block2 = STGCN_STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, 2, dropout)
        self.block3 = STGCN_STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, 4, dropout)
        head_in     = num_nodes * hidden_2
        head_hidden = 64
        self.temporal_pool = TemporalAttentionPool(hidden_2, num_nodes)

        def _head(out_size):
            return nn.Sequential(
                nn.Linear(head_in, head_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, out_size))

        self.detect_head = _head(2)
        self.pipe_head   = _head(PIPE_CLASSES)
        self.size_head   = _head(SIZE_CLASSES)
        self.pos_head    = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        z = self.temporal_pool(x)
        return (self.detect_head(z), self.pipe_head(z),
                self.size_head(z), self.pos_head(z).squeeze(1))


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def select_scenarios(manifest_path, dataset_root):
    df = pd.read_csv(manifest_path)
    selected = []
    for _, row in df[df["label_detection"] == 0].head(N_NOLEAK_PICK).iterrows():
        folder = os.path.join(dataset_root, f"scenario_{int(row['scenario_id']):05d}")
        if os.path.isfile(os.path.join(folder, "data.csv")):
            selected.append({"folder": folder, "scenario_id": int(row["scenario_id"]),
                             "label_detection": 0, "label_pipe": -1, "group": "no-leak"})
    for pipe in range(1, 6):
        pipe_rows = df[(df["label_detection"] == 1) & (df["label_pipe"] == pipe)].head(N_LEAK_PICK)
        for _, row in pipe_rows.iterrows():
            folder = os.path.join(dataset_root, f"scenario_{int(row['scenario_id']):05d}")
            if os.path.isfile(os.path.join(folder, "data.csv")):
                selected.append({"folder": folder, "scenario_id": int(row["scenario_id"]),
                                 "label_detection": 1, "label_pipe": pipe,
                                 "group": f"pipe-{pipe}"})
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# AUTOENCODER RECONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def reconstruct_ae(model, raw_signals, ae_mu, ae_sigma, unmonitored_idx, window):
    """Returns denormalised reconstruction (T, N)."""
    T, N = raw_signals.shape
    feats_norm = ((raw_signals[:, :, None] - ae_mu[None]) / (ae_sigma[None] + 1e-8)
                  ).squeeze(-1).astype(np.float32)

    n_windows = max(1, (T - window) // 1 + 1)
    windows   = np.stack([feats_norm[w: w + window] for w in range(n_windows)],
                         axis=0)[:, :, :, None]   # (n_win, W, N, 1)

    x_t    = torch.tensor(windows, dtype=torch.float32, device=DEVICE)
    masked = x_t.clone()
    masked[:, :, unmonitored_idx, :] = 0.0

    model.eval()
    recon_norm = model(masked).cpu().numpy()         # (n_win, W, N, 1)

    recon_sum   = np.zeros((T, N, 1), dtype=np.float64)
    recon_count = np.zeros((T, N, 1), dtype=np.float64)
    for w in range(n_windows):
        end = min(w + window, T)
        recon_sum[w:end]   += recon_norm[w, :end - w]
        recon_count[w:end] += 1.0

    recon_norm_full = (recon_sum / np.maximum(recon_count, 1.0)
                       ).squeeze(-1).astype(np.float32)
    return recon_norm_full * ae_sigma[:, 0] + ae_mu[:, 0]   # (T, N)


def ae_to_full_array(data_df, recon_raw, sensor_names, monitored, unmonitored_idx):
    """
    Build (T, N) array: actual values for monitored sensors,
    AE-reconstructed for unmonitored sensors.
    """
    idx    = {s: i for i, s in enumerate(sensor_names)}
    T      = len(data_df)
    out    = recon_raw.copy()   # start from full reconstruction
    # Override monitored with actual readings
    for s in monitored:
        out[:, idx[s]] = data_df[s].values.astype(np.float32)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STGCN INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def make_stgcn_features(raw_signals, baseline_template, mu, sigma, window):
    T  = raw_signals.shape[0]
    W  = min(window, T)
    raw_win  = raw_signals[:W]
    base_win = baseline_template[:W]
    dev_win  = raw_win - base_win
    feats    = np.stack([raw_win, dev_win], axis=-1).astype(np.float32)
    feats    = (feats - mu[None]) / (sigma[None] + 1e-8)
    return torch.tensor(feats[None], dtype=torch.float32, device=DEVICE)   # (1, W, N, 2)


@torch.no_grad()
def predict_stgcn(model, x_tensor):
    model.eval()
    det_logits, pipe_logits, size_logits, pos_pred = model(x_tensor)
    det_pred  = int(det_logits.argmax(dim=1).item())
    pipe_pred = int(pipe_logits.argmax(dim=1).item())
    size_pred = int(size_logits.argmax(dim=1).item())
    pos_val   = float(pos_pred.item())
    det_prob  = float(torch.softmax(det_logits, dim=1)[0, 1].item())
    return det_pred, pipe_pred, size_pred, pos_val, det_prob


def decode_predictions(det_pred, pipe_pred_raw):
    pred_pipe = None if det_pred == 0 else (pipe_pred_raw + 1 if pipe_pred_raw < 5 else None)
    return det_pred, pred_pipe


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for path in [AE_BUNDLE, STGCN_BUNDLE]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Bundle not found: {path}")

    # ── Load AE bundle ───────────────────────────────────────────────────────
    print(f"Loading AE bundle: {AE_BUNDLE}")
    ae_bundle = torch.load(AE_BUNDLE, map_location="cpu", weights_only=False)
    ae_adj          = ae_bundle["adjacency"]
    ae_mu           = ae_bundle["mu"]               # (N, 1)
    ae_sigma        = ae_bundle["sigma"]             # (N, 1)
    ae_sensors      = ae_bundle["sensor_names"]
    ae_monitored    = [ae_sensors[i] for i in ae_bundle["monitored_idx"]]
    ae_unmon_idx    = ae_bundle["unmonitored_idx"]
    ae_window       = ae_bundle["window"]

    ae_model = STGCNAutoencoder(
        adj_matrix=ae_adj, node_feats=ae_bundle["node_feats"],
        hidden_1=ae_bundle["hidden_1"], hidden_2=ae_bundle["hidden_2"],
        kernel_size=ae_bundle["kernel_size"], dropout=ae_bundle["dropout"],
    ).to(DEVICE)
    ae_model.load_state_dict(ae_bundle["model_state_dict"])
    ae_model.eval()
    print(f"AE model loaded. Monitored: {ae_monitored}")

    # ── Load STGCN bundle ────────────────────────────────────────────────────
    print(f"Loading STGCN bundle: {STGCN_BUNDLE}")
    stgcn_bundle = torch.load(STGCN_BUNDLE, map_location="cpu", weights_only=False)
    stgcn_adj      = stgcn_bundle["adjacency"]
    stgcn_mu       = stgcn_bundle["mu"]
    stgcn_sigma    = stgcn_bundle["sigma"]
    stgcn_baseline = stgcn_bundle["baseline_template"]
    stgcn_sensors  = stgcn_bundle["sensor_names"]
    stgcn_window   = stgcn_bundle["window"]
    num_nodes      = len(stgcn_sensors)

    stgcn_model = SingleLeakSTGCN(
        stgcn_adj, num_nodes,
        stgcn_bundle["hidden_1"], stgcn_bundle["hidden_2"],
        stgcn_bundle["kernel_size"], stgcn_bundle["dropout"],
        stgcn_bundle["node_feats"],
    ).to(DEVICE)
    stgcn_model.load_state_dict(stgcn_bundle["model_state_dict"])
    stgcn_model.eval()
    print(f"STGCN model loaded. Sensors: {stgcn_sensors}\n")

    # ── Select 12 scenarios ──────────────────────────────────────────────────
    scenarios = select_scenarios(MANIFEST, DATASET_ROOT)
    print(f"Selected {len(scenarios)} scenarios\n")

    rows = []

    for idx, scn in enumerate(scenarios, 1):
        folder    = scn["folder"]
        scn_id    = scn["scenario_id"]
        group     = scn["group"]
        true_det  = scn["label_detection"]
        true_pipe = scn["label_pipe"]

        print(f"[{idx}/{len(scenarios)}] Scenario {scn_id:05d}  ({group})")

        data_df = pd.read_csv(os.path.join(folder, "data.csv"))
        raw     = data_df[ae_sensors].to_numpy(dtype=np.float32)   # (T, N)

        # ── V3 AE reconstruction ─────────────────────────────────────────────
        recon_raw = reconstruct_ae(ae_model, raw, ae_mu, ae_sigma,
                                   ae_unmon_idx, ae_window)
        full_raw  = ae_to_full_array(data_df, recon_raw, ae_sensors,
                                     ae_monitored, ae_unmon_idx)

        # Reorder columns to match STGCN sensor order (should be same, but be safe)
        ae_idx    = {s: i for i, s in enumerate(ae_sensors)}
        full_raw_stgcn = np.stack(
            [full_raw[:, ae_idx[s]] for s in stgcn_sensors], axis=1
        )   # (T, N)

        # ── STGCN inference ──────────────────────────────────────────────────
        x_tensor = make_stgcn_features(full_raw_stgcn, stgcn_baseline,
                                       stgcn_mu, stgcn_sigma, stgcn_window)
        det_pred_raw, pipe_pred_raw, size_pred, pos_pred, det_prob = predict_stgcn(
            stgcn_model, x_tensor)
        pred_det, pred_pipe = decode_predictions(det_pred_raw, pipe_pred_raw)

        # ── Evaluate ─────────────────────────────────────────────────────────
        det_correct  = (pred_det == true_det)
        pipe_correct = (pred_pipe == true_pipe) if true_det == 1 else False

        print(f"  True : det={true_det}  pipe={true_pipe if true_det else 'N/A'}")
        print(f"  Pred : det={pred_det} (p={det_prob:.2f})  pipe={pred_pipe}")
        mark = "OK" if det_correct else "WRONG"
        if true_det == 1:
            mark += f"  pipe={'OK' if pipe_correct else 'WRONG'}"
        print(f"  Result: [{mark}]")

        SIZE_MAP = {0: "S", 1: "M", 2: "L", 3: "none"}
        rows.append({
            "scenario_id":    scn_id,
            "group":          group,
            "true_detection": true_det,
            "true_pipe":      true_pipe,
            "pred_detection": pred_det,
            "pred_pipe":      pred_pipe,
            "pred_size":      SIZE_MAP.get(size_pred, "?"),
            "pred_position":  round(pos_pred, 4),
            "det_prob_leak":  round(det_prob, 4),
            "det_correct":    det_correct,
            "pipe_correct":   pipe_correct,
        })

    # ── Save results ─────────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n[OK] Results saved -> {OUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS: V3 Autoencoder -> ST-GCN S10-A")
    print("=" * 60)

    print(f"\n{'ScnID':>8}  {'Group':<10}  {'TrueDet':>7}  {'PredDet':>7}  "
          f"{'TruePipe':>8}  {'PredPipe':>8}  {'DetOK':>6}  {'PipeOK':>6}")
    print("-" * 75)
    for r in rows:
        print(f"{r['scenario_id']:>8}  {r['group']:<10}  {r['true_detection']:>7}  "
              f"{r['pred_detection']:>7}  "
              f"{str(r['true_pipe']):>8}  {str(r['pred_pipe']):>8}  "
              f"{'yes' if r['det_correct'] else 'no':>6}  "
              f"{'yes' if r['pipe_correct'] else ('N/A' if r['true_detection']==0 else 'no'):>6}")

    det_acc     = np.mean([r["det_correct"] for r in rows])
    noleak_rows = [r for r in rows if r["true_detection"] == 0]
    leak_rows   = [r for r in rows if r["true_detection"] == 1]
    noleak_acc  = np.mean([r["det_correct"] for r in noleak_rows]) if noleak_rows else float("nan")
    leak_recall = np.mean([r["det_correct"] for r in leak_rows])   if leak_rows   else float("nan")
    pipe_acc    = np.mean([r["pipe_correct"] for r in leak_rows])  if leak_rows   else float("nan")

    print(f"\nDetection accuracy  (all 12) : {det_acc:.2%}  "
          f"({sum(r['det_correct'] for r in rows)}/{len(rows)})")
    print(f"No-leak specificity (n={len(noleak_rows):2d}) : {noleak_acc:.2%}")
    print(f"Leak recall         (n={len(leak_rows):2d}) : {leak_recall:.2%}")
    print(f"Pipe accuracy       (n={len(leak_rows):2d}) : {pipe_acc:.2%}  "
          f"(among all leak scenarios)")
    if any(r["det_correct"] and r["true_detection"] == 1 for r in rows):
        correctly_detected = [r for r in leak_rows if r["det_correct"]]
        print(f"Pipe acc (det=correct, n={len(correctly_detected):2d}) "
              f": {np.mean([r['pipe_correct'] for r in correctly_detected]):.2%}")

    print("\nPer-group pipe accuracy:")
    for pipe in range(1, 6):
        grp = [r for r in rows if r["group"] == f"pipe-{pipe}"]
        if grp:
            pacc = np.mean([r["pipe_correct"] for r in grp])
            print(f"  pipe-{pipe}: {pacc:.2%}  "
                  f"({sum(r['pipe_correct'] for r in grp)}/{len(grp)})")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
