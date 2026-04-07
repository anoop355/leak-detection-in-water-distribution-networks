"""
predict_from_inp.py
--------------------
Interactive command-line inference script for leak detection and localisation.
Simulates a user-selected .inp file, generates deviation plots, and runs
the ST-GCN S10-A model to detect and localise leaks.

Usage:
    python predict_from_inp.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wntr


# ============================================================
# CONSTANTS
# ============================================================
BUNDLE_PATH  = Path("stgcn_placement_bundles/stgcn_bundle_S10-A.pt")
INP_FOLDER   = Path("inp_scenarios")
PREDICTIONS  = Path("predictions")

SENSOR_NODES = ["2", "3", "4", "5", "6"]
SENSOR_LINKS = ["1a", "2a", "3a", "4a", "5a"]
T_AXIS_MIN   = [i * 15 for i in range(12)]

PIPE_LENGTHS = {1: 100.0, 2: 500.0, 3: 200.0, 4: 200.0, 5: 200.0}

DAYS         = ["Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday"]
PAT_TIMESTEP = 60  # seconds per pattern step (1 minute)

NUM_PIPES     = 5
PIPE_NONE_IDX = 5
PIPE_CLASSES  = 6
SIZE_CLASSES  = 4


# ============================================================
# MODEL DEFINITIONS  (copied verbatim from evaluate_stgcn_s10.py)
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


class SingleLeakSTGCN(nn.Module):
    """
    Temporal-attention-pool backbone — matches stgcn_single_leak_v4 / v5_10ch.
    Output heads: detect (2), pipe (PIPE_CLASSES), size (SIZE_CLASSES), pos (1).
    """
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, 4, dropout)

        head_in, head_hidden = num_nodes * hidden_2, 64
        self.temporal_pool   = TemporalAttentionPool(hidden_2, num_nodes)

        def _head(out_size):
            return nn.Sequential(
                nn.Linear(head_in, head_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, out_size)
            )

        self.detect_head = _head(2)
        self.pipe_head   = _head(PIPE_CLASSES)
        self.size_head   = _head(SIZE_CLASSES)
        self.pos_head    = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x)))
        z = self.temporal_pool(x)
        return (self.detect_head(z), self.pipe_head(z),
                self.size_head(z), self.pos_head(z).squeeze(1))


# ============================================================
# BUNDLE LOADER
# ============================================================
def load_bundle():
    if not BUNDLE_PATH.exists():
        print(
            f"ERROR: Model bundle not found at {BUNDLE_PATH}\n"
            "Ensure the bundle file is present before running this script."
        )
        sys.exit(1)

    bundle     = torch.load(str(BUNDLE_PATH), map_location="cpu", weights_only=False)
    adj        = np.array(bundle["adjacency"],  dtype=np.float32)
    num_nodes  = len(bundle["sensor_names"])
    hidden_1   = int(bundle.get("hidden_1",    16))
    hidden_2   = int(bundle.get("hidden_2",    32))
    kernel_sz  = int(bundle.get("kernel_size",  5))
    dropout    = float(bundle.get("dropout",  0.25))
    node_feats = int(bundle.get("node_feats",   2))

    model = SingleLeakSTGCN(adj, num_nodes, hidden_1, hidden_2,
                            kernel_sz, dropout, node_feats)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle


# ============================================================
# WNTR SIMULATION
# ============================================================
def apply_pattern_rotation(wn, offset_steps: int):
    """
    Rotates all demand patterns in the network left by offset_steps.
    This sets the effective simulation start time within the weekly
    demand cycle without modifying the EPANET clock start time.

    Patterns with zero or one multiplier are skipped (no rotation needed).
    """
    for pat_name in wn.pattern_name_list:
        mults = np.array(wn.get_pattern(pat_name).multipliers)
        if len(mults) > 1:
            rotated = np.roll(mults, -int(offset_steps % len(mults)))
            wn.get_pattern(pat_name).multipliers = list(rotated)


def simulate_inp(inp_path: Path, offset_steps: int = 0):
    """Load and simulate an .inp file. Returns (p, q) as (12, 5) float32 arrays."""
    wn = wntr.network.WaterNetworkModel(str(inp_path))
    wn.options.time.duration           = 3 * 3600
    wn.options.time.hydraulic_timestep = 15 * 60
    wn.options.time.report_timestep    = 15 * 60
    wn.options.time.pattern_timestep   = 60
    wn.options.time.start_clocktime    = 0

    if offset_steps > 0:
        apply_pattern_rotation(wn, offset_steps)

    results   = wntr.sim.EpanetSimulator(wn).run_sim()
    pressures = results.node["pressure"]
    flows     = results.link["flowrate"]

    missing_nodes = [n for n in SENSOR_NODES if n not in pressures.columns]
    missing_links = [lk for lk in SENSOR_LINKS if lk not in flows.columns]

    if missing_nodes or missing_links:
        raise ValueError(
            "ERROR: Required sensor nodes or links not found in simulation results.\n"
            f"Expected nodes: {', '.join(SENSOR_NODES)}\n"
            f"Expected links: {', '.join(SENSOR_LINKS)}\n"
            "Verify that the .inp file uses the correct node and link IDs."
        )

    n_rows = 12
    p = pressures[SENSOR_NODES].values[:n_rows].astype(np.float32)
    q = flows[SENSOR_LINKS].values[:n_rows].astype(np.float32)

    if np.any(p < 0):
        raise ValueError(
            "ERROR: Negative pressures detected in simulation results.\n"
            "The .inp file may be hydraulically invalid or require pressure-driven analysis."
        )

    return p, q


def build_df(p: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"t": T_AXIS_MIN})
    for i, node in enumerate(SENSOR_NODES):
        df[f"P{node}"] = p[:, i]
    for i, link in enumerate(SENSOR_LINKS):
        df[f"Q{link}"] = q[:, i]
    return df


# ============================================================
# PLOTS
# ============================================================
def generate_plots(deviation: np.ndarray, sensor_names: list, plots_dir: Path):
    """Generate 10 deviation-from-baseline PNG plots, one per sensor."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(sensor_names):
        is_pressure = name.startswith("P")
        colour      = "blue" if is_pressure else "orange"
        y_unit      = "(m)" if is_pressure else "(m³/s)"
        y_label     = f"Deviation from Baseline {y_unit}"

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(T_AXIS_MIN, deviation[:, idx], color=colour, linewidth=1.8)
        ax.axhline(0, color="grey", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(y_label)
        ax.set_title(f"Sensor {name} — Deviation from Baseline")
        ax.grid(True)

        fig.savefig(str(plots_dir / f"deviation_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# INFERENCE
# ============================================================
def run_inference(raw: np.ndarray, model: nn.Module, bundle: dict):
    """
    raw: (12, 10) float32 array in sensor_names order.
    Returns (pred_detect, pred_pipe_id, pred_pos, deviation).
    """
    baseline  = np.array(bundle["baseline_template"], dtype=np.float32)  # (12, 10)
    mu        = np.array(bundle["mu"],    dtype=np.float32)               # (10, 2)
    sigma     = np.array(bundle["sigma"], dtype=np.float32)               # (10, 2)

    deviation = raw - baseline                                            # (12, 10)
    feats     = np.stack([raw, deviation], axis=-1).astype(np.float32)   # (12, 10, 2)
    feats     = (feats - mu[None]) / (sigma[None] + 1e-8)

    x_tensor  = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)    # (1, 12, 10, 2)

    with torch.no_grad():
        detect_logits, pipe_logits, _, pos_pred = model(x_tensor)

    pred_detect  = int(detect_logits.argmax(dim=1).item())
    pred_pipe    = int(pipe_logits.argmax(dim=1).item())
    pred_pos     = float(pos_pred.item())
    pred_pos     = max(0.0, min(1.0, pred_pos))
    pred_pipe_id = (pred_pipe + 1) if pred_pipe < NUM_PIPES else None

    return pred_detect, pred_pipe_id, pred_pos, deviation


# ============================================================
# TERMINAL OUTPUT
# ============================================================
def print_result(scenario_name: str, pred_detect: int,
                 pred_pipe_id, pred_pos: float,
                 out_csv: Path, plots_dir: Path,
                 sim_time_str: str = ""):
    sep  = "=" * 60
    dash = "-" * 60

    if pred_detect == 1:
        detect_str = "LEAK DETECTED"
        pipe_str   = f"Pipe {pred_pipe_id}" if pred_pipe_id is not None else "N/A"
        pos_norm   = f"{pred_pos:.2f} (normalised)"
        if pred_pipe_id is not None and pred_pipe_id in PIPE_LENGTHS:
            metres = pred_pos * PIPE_LENGTHS[pred_pipe_id]
            pos_m  = f"{metres:.1f} m from pipe start"
        else:
            pos_m  = "N/A"
    else:
        detect_str = "NO LEAK DETECTED"
        pipe_str   = "N/A"
        pos_norm   = "N/A"
        pos_m      = "N/A"

    print(f"\n{sep}")
    print(" LEAK DETECTION AND LOCALISATION — RESULT")
    print(sep)
    print(f" Scenario     : {scenario_name}")
    print(f" Sim start    : {sim_time_str}")
    print(f" Bundle       : {BUNDLE_PATH.name}")
    print(dash)
    print(f" Detection    : {detect_str}")
    print(f" Pipe         : {pipe_str}")
    print(f" Position     : {pos_norm}")
    print(f"               {pos_m}")
    print(dash)
    print(f" data.csv     : {out_csv}")
    print(f" Plots saved  : {plots_dir} (10 plots)")
    print(sep)


# ============================================================
# USER FLOW
# ============================================================
def list_inp_files():
    """Return sorted list of .inp files in INP_FOLDER, creating the folder if absent."""
    if not INP_FOLDER.exists():
        INP_FOLDER.mkdir(parents=True, exist_ok=True)
        print(
            "No .inp files found. Please place your .inp file in the\n"
            "inp_scenarios/ folder and run the script again."
        )
        sys.exit(0)

    files = sorted(INP_FOLDER.glob("*.inp"))
    if not files:
        print(
            "No .inp files found. Please place your .inp file in the\n"
            "inp_scenarios/ folder and run the script again."
        )
        sys.exit(0)

    return files


def ask_simulation_time() -> int:
    """
    Prompts the user to select a day of the week and an hour of day.
    Returns the pattern rotation offset in steps (integer minutes).
    """
    print("\nSelect simulation start day:")
    for i, day in enumerate(DAYS, start=1):
        print(f"  [{i}] {day}")

    while True:
        try:
            day_raw = input("Enter day number (1-7): ").strip()
            day_num = int(day_raw)
            if 1 <= day_num <= 7:
                break
            print("  Please enter a number between 1 and 7.")
        except ValueError:
            print("  Invalid input. Please enter a number.")

    while True:
        try:
            hour_raw = input("Enter simulation start hour (0-23): ").strip()
            hour = int(hour_raw)
            if 0 <= hour <= 23:
                break
            print("  Please enter a number between 0 and 23.")
        except ValueError:
            print("  Invalid input. Please enter a number.")

    selected_day_index = day_num - 1
    day_minutes        = selected_day_index * 24 * 60
    hour_minutes       = hour * 60
    offset_steps       = day_minutes + hour_minutes  # PAT_TIMESTEP=60s → 1 step per minute

    sim_time_str = f"{DAYS[selected_day_index]} at {hour:02d}:00"
    print(f"Simulation start: {sim_time_str}")

    return offset_steps


def select_file(files: list) -> Path:
    print("\nAvailable .inp files:")
    for i, f in enumerate(files, start=1):
        print(f"  [{i}] {f.name}")

    while True:
        try:
            raw = input("\nEnter the number of the file to run: ").strip()
            idx = int(raw)
            if 1 <= idx <= len(files):
                return files[idx - 1]
            print(f"  Please enter a number between 1 and {len(files)}.")
        except ValueError:
            print("  Invalid input. Please enter a number.")


def process_scenario(inp_path: Path, model: nn.Module, bundle: dict,
                     offset_steps: int = 0, sim_time_str: str = ""):
    scenario_name = inp_path.stem
    out_dir       = PREDICTIONS / scenario_name
    plots_dir     = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Simulating {inp_path.name} ...")
    try:
        p, q = simulate_inp(inp_path, offset_steps)
    except ValueError as e:
        print(str(e))
        return
    except Exception as e:
        print(
            f"ERROR: WNTR simulation failed for {inp_path.name}\n"
            f"Reason: {e}\n"
            "Please verify that the .inp file is a valid EPANET network."
        )
        return

    print("[2/4] Saving data.csv ...")
    df      = build_df(p, q)
    out_csv = out_dir / "data.csv"
    df.to_csv(out_csv, index=False)

    sensor_names = bundle["sensor_names"]
    raw          = df[sensor_names].to_numpy(dtype=np.float32)  # (12, 10)

    print("[3/4] Running inference ...")
    pred_detect, pred_pipe_id, pred_pos, deviation = run_inference(raw, model, bundle)

    print("[4/4] Generating deviation plots ...")
    generate_plots(deviation, sensor_names, plots_dir)

    print_result(scenario_name, pred_detect, pred_pipe_id, pred_pos, out_csv, plots_dir,
                 sim_time_str)


def main():
    print("Loading model bundle ...")
    model, bundle = load_bundle()
    print(f"[OK] Loaded bundle from {BUNDLE_PATH}")

    while True:
        files        = list_inp_files()
        inp_path     = select_file(files)
        offset_steps = ask_simulation_time()
        total_minutes = offset_steps
        day_index     = (total_minutes // (24 * 60)) % 7
        hour          = (total_minutes % (24 * 60)) // 60
        sim_time_str  = f"{DAYS[day_index]} at {hour:02d}:00"
        process_scenario(inp_path, model, bundle, offset_steps, sim_time_str)

        while True:
            again = input("\nRun another scenario? (y/n): ").strip().lower()
            if again in ("y", "n"):
                break
            print("  Please enter 'y' or 'n'.")

        if again == "n":
            print("Exiting.")
            break


if __name__ == "__main__":
    main()
