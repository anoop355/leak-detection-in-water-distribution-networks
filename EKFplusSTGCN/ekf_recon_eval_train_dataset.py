"""
- Evaluates EKF reconstruction quality on the stgcn_dataset_ekf scenarios
- Compared reconstructed vs Ground truth
- Metrics computed MAE, RMSE, R^2
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_ROOT     = _HERE.parent

SRC_ROOT     = _ROOT / "stgcn_dataset_v2"      # ground truth
EKF_ROOT     = _ROOT / "stgcn_dataset_ekf"     # EKF-reconstructed
MANIFEST     = EKF_ROOT / "manifests" / "manifest_full.csv"
RESULTS_DIR  = _HERE / "results"

UNMONITORED = ["P2", "P3", "P5", "P6", "Q2a", "Q4a", "Q5a"]
MONITORED   = ["P4", "Q1a", "Q3a"]


# ── metric helpers ─────────────────────────────────────────────────────────────
def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return float(1.0 - ss_res / ss_tot)


def sensor_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(true - pred)))
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    r2_  = r2(true, pred)
    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2_, 4)}


# ── per-scenario evaluation ────────────────────────────────────────────────────
def evaluate_scenario(scn_id: int) -> list[dict] | None:
    """
    Returns a list of per-sensor dicts for one scenario, or None if files
    are missing.
    """
    name     = f"scenario_{scn_id:05d}"
    src_dir  = SRC_ROOT / "scenarios" / name
    ekf_dir  = EKF_ROOT / "scenarios" / name

    if not (src_dir / "data.csv").exists() or not (ekf_dir / "data.csv").exists():
        return None

    gt  = pd.read_csv(src_dir / "data.csv")
    ekf = pd.read_csv(ekf_dir / "data.csv")

    with open(ekf_dir / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)

    true_det  = int(labels.get("label_detection", 0))
    true_pipe = int(labels.get("label_pipe", -1))
    group     = "no-leak" if true_det == 0 else f"pipe-{true_pipe}"

    rows = []
    for sensor in UNMONITORED:
        if sensor not in gt.columns or sensor not in ekf.columns:
            continue
        true_vals = gt[sensor].to_numpy(dtype=float)
        pred_vals = ekf[sensor].to_numpy(dtype=float)
        T = min(len(true_vals), len(pred_vals))
        m = sensor_metrics(true_vals[:T], pred_vals[:T])
        rows.append({
            "scenario_id": name,
            "group":       group,
            "true_pipe":   true_pipe,
            "sensor":      sensor,
            "mae":         m["mae"],
            "rmse":        m["rmse"],
            "r2":          m["r2"],
        })

    # Overall (mean across unmonitored sensors for this scenario)
    if rows:
        overall_mae = round(float(np.mean([r["mae"] for r in rows])), 6)
        rows.append({
            "scenario_id": name,
            "group":       group,
            "true_pipe":   true_pipe,
            "sensor":      "OVERALL",
            "mae":         overall_mae,
            "rmse":        float("nan"),
            "r2":          float("nan"),
        })

    return rows


# ── summary helpers ────────────────────────────────────────────────────────────
def summarise(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Mean MAE/RMSE/R^2 per sensor, optionally grouped."""
    sensor_df = df[df["sensor"] != "OVERALL"]
    if group_col:
        grouped = sensor_df.groupby([group_col, "sensor"])[["mae", "rmse", "r2"]].mean()
        return grouped.round(6).reset_index()
    else:
        grouped = sensor_df.groupby("sensor")[["mae", "rmse", "r2"]].mean()
        return grouped.round(6).reset_index()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    if not EKF_ROOT.exists():
        raise FileNotFoundError(
            f"EKF dataset not found: {EKF_ROOT}\n"
            f"Run ekf_preprocess_stgcn_dataset.py first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST)
    scn_ids  = manifest["scenario_id"].tolist()

    # Only evaluate scenarios that have been preprocessed
    available = [sid for sid in scn_ids
                 if (EKF_ROOT / "scenarios" / f"scenario_{int(sid):05d}"
                     / "data.csv").exists()]

    print(f"Scenarios available in stgcn_dataset_ekf : {len(available)}")
    print(f"Evaluating reconstruction of : {UNMONITORED}")
    print(f"Results -> {RESULTS_DIR}\n")

    all_rows  = []
    failed    = 0

    for i, sid in enumerate(available):
        rows = evaluate_scenario(int(sid))
        if rows is None:
            failed += 1
        else:
            all_rows.extend(rows)
        if (i + 1) % 200 == 0 or (i + 1) == len(available):
            print(f"  Progress: {i+1}/{len(available)}  ({failed} failed)")

    if not all_rows:
        print("No results to save.")
        return

    per_df = pd.DataFrame(all_rows)

    # ── Per-scenario CSV ───────────────────────────────────────────────────────
    per_csv = RESULTS_DIR / "ekf_recon_train_per_scenario.csv"
    per_df.to_csv(per_csv, index=False)
    print(f"\n[OK] Per-scenario -> {per_csv}")

    # ── Sensor summary (overall) ───────────────────────────────────────────────
    sensor_summary = summarise(per_df)
    # Add overall mean row
    overall_mae  = round(float(per_df[per_df["sensor"] != "OVERALL"]["mae"].mean()), 6)
    overall_rmse = round(float(per_df[per_df["sensor"] != "OVERALL"]["rmse"].mean()), 6)
    sensor_summary_out = RESULTS_DIR / "ekf_recon_train_summary_sensor.csv"
    sensor_summary.to_csv(sensor_summary_out, index=False)
    print(f"[OK] Sensor summary -> {sensor_summary_out}")

    # ── Pipe-group summary ─────────────────────────────────────────────────────
    pipe_summary = summarise(per_df, group_col="group")
    pipe_summary_out = RESULTS_DIR / "ekf_recon_train_summary_pipe.csv"
    pipe_summary.to_csv(pipe_summary_out, index=False)
    print(f"[OK] Pipe summary   -> {pipe_summary_out}")

    # ── Console report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("EKF RECONSTRUCTION QUALITY  — stgcn_dataset_ekf")
    print("=" * 65)
    print(f"Scenarios evaluated : {len(available)}  ({failed} failed)")
    print(f"Overall mean MAE    : {overall_mae}")
    print(f"Overall mean RMSE   : {overall_rmse}")
    print()

    print(f"--- Per-Sensor (mean across all scenarios) ---")
    print(f"  {'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}")
    print("  " + "-" * 42)
    for _, row in sensor_summary.iterrows():
        print(f"  {row['sensor']:<8}  {row['mae']:>10.6f}  "
              f"{row['rmse']:>10.6f}  {row['r2']:>8.4f}")

    print()
    print(f"--- Per-Group Summary (mean MAE per sensor) ---")
    groups = ["no-leak"] + [f"pipe-{p}" for p in range(1, 6)] + ["ALL"]
    # Add ALL group
    all_group = summarise(per_df)
    all_group.insert(0, "group", "ALL")
    pipe_summary_full = pd.concat([pipe_summary, all_group], ignore_index=True)

    for grp in groups:
        sub = pipe_summary_full[pipe_summary_full["group"] == grp]
        if sub.empty:
            continue
        n_scen = len(per_df[(per_df["group"] == grp) & (per_df["sensor"] == "OVERALL")]) \
                 if grp != "ALL" else len(per_df[per_df["sensor"] == "OVERALL"])
        print(f"\n  {grp}  (n={n_scen})")
        print(f"  {'Sensor':<8}  {'MAE':>10}  {'RMSE':>10}  {'R2':>8}")
        print("  " + "-" * 42)
        for _, row in sub.iterrows():
            print(f"  {row['sensor']:<8}  {row['mae']:>10.6f}  "
                  f"{row['rmse']:>10.6f}  {row['r2']:>8.4f}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
