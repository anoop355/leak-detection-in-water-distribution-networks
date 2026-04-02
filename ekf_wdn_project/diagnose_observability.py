"""
Sensitivity / Effective Observability Analysis

 Computes the measurement Jacobian H = d[P4, Q1a, Q3a] / d[D2, D3, D4, D5, D6]
at the nominal operating point, then derives:
  - Per-node sensitivity (column norms)
  - Fisher information matrix  F = H^T R^{-1} H
  - Per-node information content (diagonal of F)
  - Effective observability rank
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sure the project root is on sys.path so local imports work
sys.path.insert(0, str(Path(__file__).parent))

from config import CONFIG
from hydraulic_interface import HydraulicInterface
from jacobians import numerical_jacobian
from load_model import extract_model_metadata


# helpers

def _bar(value: float, max_value: float, width: int = 30) -> str:
    filled = int(round(width * value / max_value)) if max_value > 0 else 0
    return "#" * filled + "." * (width - filled)


def _print_matrix(title: str, mat: np.ndarray, row_labels: list[str], col_labels: list[str]) -> None:
    col_w = max(len(c) for c in col_labels) + 2
    row_w = max(len(r) for r in row_labels) + 2
    print(f"\n{title}")
    print("-" * (row_w + col_w * len(col_labels)))
    header = " " * row_w + "".join(f"{c:>{col_w}}" for c in col_labels)
    print(header)
    for row_lbl, row in zip(row_labels, mat):
        vals = "".join(f"{v:>{col_w}.4f}" for v in row)
        print(f"{row_lbl:<{row_w}}{vals}")


# main

def main() -> None:
    print("=" * 60)
    print("Investigation 1: EKF Observability Analysis")
    print("=" * 60)

    metadata = extract_model_metadata(CONFIG)
    hi = HydraulicInterface(CONFIG, metadata)
    R = hi.build_measurement_noise()

    demand_nodes = list(metadata.demand_nodes)          # ["2","3","4","5","6"]
    measurement_labels = ["P4", "Q1a", "Q3a"]
    demand_labels = [f"D{n}" for n in demand_nodes]

    nominal_demands = CONFIG.initial_demands.copy()

    print(f"\nNominal demands (m³/s): {dict(zip(demand_labels, nominal_demands))}")
    print(f"Measurement noise R diagonal: {np.diag(R)}")

    # 1. Measurement Jacobian
    def measurement_fn(d: np.ndarray) -> np.ndarray:
        snap = hi.simulate_snapshot(d, timestamp_seconds=0)
        return snap.measurement_vector(metadata)

    print("\nComputing measurement Jacobian H (may take ~10 s)...")
    H = numerical_jacobian(measurement_fn, nominal_demands, CONFIG)
    # H shape: (3 measurements, 5 demands)

    _print_matrix(
        "H = d[P4, Q1a, Q3a] / d[D2, D3, D4, D5, D6]",
        H,
        row_labels=measurement_labels,
        col_labels=demand_labels,
    )

    # 2. Per-node measurement sensitivity (column L2-norm of H)
    col_norms = np.linalg.norm(H, axis=0)   # shape (5,)
    max_norm = col_norms.max() if col_norms.max() > 0 else 1.0

    print("\nPer-node measurement sensitivity (||H[:,i]||_2):")
    print(f"  {'Node':<8} {'Sensitivity':>14}  Bar chart")
    print(f"  {'-'*8}  {'-'*14}  {'-'*30}")
    for lbl, norm in zip(demand_labels, col_norms):
        bar = _bar(norm, max_norm)
        print(f"  {lbl:<8} {norm:>14.6f}  {bar}")

    # 3. Fisher information matrix F = H^T R^{-1} 
    R_inv = np.linalg.inv(R)
    F = H.T @ R_inv @ H   # shape (5, 5)

    _print_matrix(
        "Fisher information matrix F = H^T R^{-1} H",
        F,
        row_labels=demand_labels,
        col_labels=demand_labels,
    )

    F_diag = np.diag(F)
    max_info = F_diag.max() if F_diag.max() > 0 else 1.0

    print("\nPer-node Fisher information (diag of F — higher = better observed):")
    print(f"  {'Node':<8} {'F_ii':>14}  Bar chart")
    print(f"  {'-'*8}  {'-'*14}  {'-'*30}")
    for lbl, info in zip(demand_labels, F_diag):
        bar = _bar(info, max_info)
        print(f"  {lbl:<8} {info:>14.6f}  {bar}")

    # 4. Effective observability rank
    s_vals = np.linalg.svd(H, compute_uv=False)
    eps = 1e-6
    rank = int(np.sum(s_vals > eps))

    print(f"\nSingular values of H:  {s_vals}")
    print(f"Effective rank of H:   {rank}  (out of {min(H.shape[0], H.shape[1])} possible)")

    # 5. Kalman gain approximation at steady state
    n_states = CONFIG.state_size          # 10
    n_meas = CONFIG.measurement_size      # 3
    n_nodes = len(demand_nodes)           # 5

    H_full = np.zeros((n_meas, n_states))
    H_full[:, n_nodes:] = H              

    P0 = CONFIG.initial_covariance
    S = H_full @ P0 @ H_full.T + R      # innovation covariance (3x3)
    K = P0 @ H_full.T @ np.linalg.inv(S)  # approximate Kalman gain (10x3)

    print("\nApproximate Kalman gain K (10 states x 3 measurements) at P=P0:")
    state_labels = [f"H{n}" for n in demand_nodes] + demand_labels
    meas_labels_pad = [f"{m:>10}" for m in measurement_labels]
    print(f"  {'State':<8} " + "  ".join(meas_labels_pad))
    print(f"  {'-'*8}  {'-'*36}")
    for s_lbl, k_row in zip(state_labels, K):
        vals = "  ".join(f"{v:>10.5f}" for v in k_row)
        print(f"  {s_lbl:<8}  {vals}")

    gain_norms = np.linalg.norm(K, axis=1)
    print("\nKalman gain row norms (how strongly each state is updated by measurements):")
    max_gain = gain_norms.max() if gain_norms.max() > 0 else 1.0
    for s_lbl, gn in zip(state_labels, gain_norms):
        bar = _bar(gn, max_gain)
        print(f"  {s_lbl:<8} {gn:>12.6f}  {bar}")

    # 6. Summary interpretation 
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    threshold = 0.1 * max_norm
    poorly_observed = [lbl for lbl, norm in zip(demand_labels, col_norms) if norm < threshold]
    well_observed   = [lbl for lbl, norm in zip(demand_labels, col_norms) if norm >= threshold]

    if poorly_observed:
        print(f"\n[!] POORLY OBSERVED nodes (sensitivity < 10% of max):")
        for n in poorly_observed:
            idx = demand_labels.index(n)
            print(f"    {n}: sensitivity={col_norms[idx]:.6f}, F_ii={F_diag[idx]:.6f}")
        print(f"\n    => EKF cannot reliably estimate states for: {poorly_observed}")
        print(f"       P5/P6 reconstruction depends on these demand estimates.")
    else:
        print("\n[OK] All nodes are reasonably well observed.")

    print(f"\n[i] Well-observed nodes: {well_observed}")
    print(f"[i] Effective Jacobian rank: {rank}/3")
    if rank < n_nodes:
        print(f"    => The system is RANK-DEFICIENT. {n_nodes - rank} demand(s) cannot be uniquely identified.")

    print(f"\n[i] R matrix (measurement noise):")
    print(f"    P4   std = {np.sqrt(R[0,0]):.4f} m")
    print(f"    Q1a  std = {np.sqrt(R[1,1]):.6f} m³/s  ({np.sqrt(R[1,1])/nominal_demands[0]*100:.1f}% of D2 nominal)")
    print(f"    Q3a  std = {np.sqrt(R[2,2]):.6f} m³/s  ({np.sqrt(R[2,2])/nominal_demands[2]*100:.1f}% of D4 nominal)")
    print()


if __name__ == "__main__":
    main()
