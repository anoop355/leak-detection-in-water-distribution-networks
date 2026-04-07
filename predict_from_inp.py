"""
Simulates a user-selected .inp file, generates deviation plots, and runs
the ST-GCN S10-A model to detect and localise leaks.

The user selects a .inp file, specifies what
day and time the simulation should represent, and the script handles
everything else — simulation, feature extraction, inference, and output.

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
import wntr  # WNTR is the Python wrapper for EPANET used to run hydraulic simulations


# CONSTANTS

# Path to the trained ST-GCN model bundle — this contains the weights,
# adjacency matrix, and normalisation statistics saved after training
BUNDLE_PATH  = Path("stgcn_placement_bundles/stgcn_bundle_S10-A.pt")

# Folder where the user places their .inp files before running the script
INP_FOLDER   = Path("inp_scenarios")

# Folder where all output files (data.csv and plots) will be saved
PREDICTIONS  = Path("predictions")

# The node and link IDs in the EPANET network that correspond to the
# five pressure sensors (P2-P6) and five flow sensors (Q1a-Q5a)
SENSOR_NODES = ["2", "3", "4", "5", "6"]
SENSOR_LINKS = ["1a", "2a", "3a", "4a", "5a"]

# Time axis for the 12-timestep simulation window (0, 15, 30 ... 165 minutes)
T_AXIS_MIN   = [i * 15 for i in range(12)]

# Physical pipe lengths in metres — used to convert normalised position
# predictions into actual distance from the pipe start
PIPE_LENGTHS = {1: 100.0, 2: 500.0, 3: 200.0, 4: 200.0, 5: 200.0}

# Days of the week used for the simulation time selection prompt.
# The demand patterns in the .inp file cover a full week starting Sunday.
DAYS         = ["Sunday", "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday"]

# The demand patterns are sampled at 1-minute intervals (60 seconds per step),
# so one pattern step corresponds to one minute of simulation time
PAT_TIMESTEP = 60  # seconds per pattern step (1 minute)

# Model output class configuration:
# The pipe head outputs one of 6 classes — pipes 1 to 5, or NONE (index 5)
NUM_PIPES     = 5
PIPE_NONE_IDX = 5   # index 5 means no pipe was predicted (no-leak case)
PIPE_CLASSES  = 6   # 5 pipes + 1 NONE class
SIZE_CLASSES  = 4   # S, M, L, NONE


# MODEL DEFINITIONS

class TemporalConvLayer(nn.Module):
    """
    A single dilated temporal convolution layer applied independently
    at each sensor node.

    The convolution runs along the time axis only, using a 1D kernel
    of the specified size and dilation. 

    Batch normalisation and ReLU activation are applied after convolution.
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        # Symmetric padding keeps the output length equal to the input length
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (1, kernel_size),
                              padding=(0, pad), dilation=(1, dilation))
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # Input shape is (batch, time, nodes, channels)
        # Conv2d expects (batch, channels, nodes, time), so we permute first
        x = x.permute(0, 3, 2, 1)
        x = self.act(self.bn(self.conv(x)))
        # Permute back to (batch, time, nodes, channels) for the next layer
        return x.permute(0, 3, 2, 1)


class GraphConvLayer(nn.Module):
    """
    A graph convolution layer that propagates information between
    neighbouring sensor nodes using the network's adjacency matrix.

    The adjacency matrix A encodes which sensors are physically
    connected in the pipe network. Multiplying by A lets each node
    aggregate signals from its neighbours, which is how spatial
    context is built into the model.

    Layer normalisation and ReLU activation are applied after the
    linear transformation.
    """
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        # Register the adjacency matrix as a buffer so it moves to GPU
        # automatically if needed, but is not treated as a learnable parameter
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # Einstein summation to multiply the adjacency matrix with the
        # node features — this aggregates each node's neighbours' signals
        x = torch.einsum("ij,btjc->btic", self.A, x)
        return self.act(self.ln(self.lin(x)))


class STBlock(nn.Module):
    """
    A spatio-temporal block that combines temporal convolution and
    graph convolution with a residual (skip) connection.

    The residual connection adds the input back to the output of the
    block, which helps gradients flow during training and prevents
    the network from losing information as depth increases.

    If the input and output channel sizes differ, a linear projection
    is applied to the residual so the addition is valid.
    """
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5,
                 dilation=1, dropout=0.25):
        super().__init__()
        self.temp     = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph    = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout  = nn.Dropout(dropout)
        self.out_act  = nn.ReLU()
        # If channel sizes match, the residual is passed through unchanged;
        # otherwise, a linear layer projects it to the correct size
        self.res_proj = (nn.Linear(in_ch, out_ch) if in_ch != out_ch
                         else nn.Identity())

    def forward(self, x):
        r = self.res_proj(x)   # save residual before transformation
        y = self.dropout(self.graph(self.temp(x)))
        return self.out_act(y + r)  # add residual and apply activation


class TemporalAttentionPool(nn.Module):
    """
    A learnable attention pooling layer collapses the time dimension.

    This layer learns to assign a higher weight
    to timesteps that carry more useful information — for example, the
    timesteps after a leak onset.

    Output shape is (batch, nodes * channels), which flattens the spatial
    and feature dimensions into a single vector for the classification heads.
    """
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        # A single linear layer computes one attention score per timestep
        self.attn = nn.Linear(hidden_dim * num_nodes, 1)

    def forward(self, x):
        B, T, N, C = x.shape
        # Flatten node and channel dims so attention is computed over time
        x_flat  = x.reshape(B, T, N * C)
        # Softmax ensures weights sum to 1 across the time dimension
        weights = torch.softmax(self.attn(x_flat), dim=1)
        # Weighted sum collapses the time dimension
        return (x_flat * weights).sum(dim=1)


class SingleLeakSTGCN(nn.Module):
    """
    The full ST-GCN model for single-leak detection and localisation.

    The architecture consists of three spatio-temporal blocks with
    exponentially increasing dilation (1, 2, 4), followed by temporal
    attention pooling and four task-specific output heads:

        detect_head  — binary classification: leak or no-leak
        pipe_head    — identifies which of the 5 pipes contains the leak
        size_head    — classifies leak size (S, M, L, or NONE)
        pos_head     — predicts normalised leak position along the pipe [0, 1]

    The detect and pipe heads are the primary outputs used in this script.
    """
    def __init__(self, adj_matrix, num_nodes, hidden_1=16, hidden_2=32,
                 kernel_size=5, dropout=0.25, node_feats=2):
        super().__init__()
        # Three ST blocks — each adds more temporal context via increasing dilation
        self.block1 = STBlock(node_feats, hidden_1, adj_matrix, kernel_size, 1, dropout)
        self.block2 = STBlock(hidden_1,   hidden_2, adj_matrix, kernel_size, 2, dropout)
        self.block3 = STBlock(hidden_2,   hidden_2, adj_matrix, kernel_size, 4, dropout)

        # After pooling, the feature vector has size num_nodes * hidden_2
        # For 10 sensors and hidden_2=32, this is 10 * 32 = 320
        head_in, head_hidden = num_nodes * hidden_2, 64
        self.temporal_pool   = TemporalAttentionPool(hidden_2, num_nodes)

        # Helper function to build a two-layer classification head
        def _head(out_size):
            return nn.Sequential(
                nn.Linear(head_in, head_hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(head_hidden, out_size)
            )

        self.detect_head = _head(2)            # 2 classes: no-leak, leak
        self.pipe_head   = _head(PIPE_CLASSES) # 6 classes: pipe 1-5 + NONE
        self.size_head   = _head(SIZE_CLASSES) # 4 classes: S, M, L, NONE
        # Position head uses Sigmoid to constrain the output to [0, 1]
        self.pos_head    = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # Pass through all three ST blocks sequentially
        x = self.block3(self.block2(self.block1(x)))
        # Pool across time to get a fixed-size feature vector
        z = self.temporal_pool(x)
        # Run all four heads in parallel on the same feature vector
        return (self.detect_head(z), self.pipe_head(z),
                self.size_head(z), self.pos_head(z).squeeze(1))


# BUNDLE LOADER

def load_bundle():
    """
    Loads the trained ST-GCN model bundle from disk and reconstructs
    the model with the saved weights and architecture parameters.
    """
    # Check the bundle file exists before attempting to load it
    if not BUNDLE_PATH.exists():
        print(
            f"ERROR: Model bundle not found at {BUNDLE_PATH}\n"
            "Ensure the bundle file is present before running this script."
        )
        sys.exit(1)

    # Load the bundle — map_location="cpu" ensures it loads even if the
    # bundle was originally saved on a GPU machine
    bundle     = torch.load(str(BUNDLE_PATH), map_location="cpu", weights_only=False)

    # Extract architecture parameters from the bundle so the model is
    # rebuilt with exactly the same structure used during training
    adj        = np.array(bundle["adjacency"],  dtype=np.float32)
    num_nodes  = len(bundle["sensor_names"])
    hidden_1   = int(bundle.get("hidden_1",    16))
    hidden_2   = int(bundle.get("hidden_2",    32))
    kernel_sz  = int(bundle.get("kernel_size",  5))
    dropout    = float(bundle.get("dropout",  0.25))
    node_feats = int(bundle.get("node_feats",   2))

    # Build the model and load the saved weights into it
    model = SingleLeakSTGCN(adj, num_nodes, hidden_1, hidden_2,
                            kernel_sz, dropout, node_feats)
    model.load_state_dict(bundle["model_state_dict"])

    # Set to evaluation mode — this disables dropout so predictions
    # are deterministic at inference time
    model.eval()
    return model, bundle


# WNTR SIMULATION

def apply_pattern_rotation(wn, offset_steps: int):
    """
    Rotates all demand patterns in the network left by offset_steps
    to simulate conditions at a specific point in the weekly demand cycle.

    The demand patterns in the .inp file cover a full week (168 hours)
    sampled at 1-minute intervals (10,080 steps). By default, WNTR
    simulates from the beginning of the pattern, which corresponds to
    Sunday at midnight. To simulate another day and time, the
    patterns need to be shifted left so that the correct slice of the
    weekly demand profile is used.

    np.roll with a negative shift moves the array to the left,
    effectively discarding the early part of the week and starting
    from the desired time point. The modulo operation handles the case
    where the offset is larger than the pattern length.

    Patterns with zero or one multiplier are skipped since there is
    nothing meaningful to rotate.
    """
    for pat_name in wn.pattern_name_list:
        mults = np.array(wn.get_pattern(pat_name).multipliers)
        if len(mults) > 1:
            rotated = np.roll(mults, -int(offset_steps % len(mults)))
            wn.get_pattern(pat_name).multipliers = list(rotated)


def simulate_inp(inp_path: Path, offset_steps: int = 0):
    """
    Loads a .inp file and runs a 3-hour EPANET hydraulic simulation
    using WNTR, returning pressure and flow readings at the sensor
    locations across 12 timesteps (one per 15 minutes).

    The simulation settings are fixed to match the conditions under
    which the training data was generated:
        - Duration: 3 hours (10,800 seconds)
        - Hydraulic and report timestep: 15 minutes (900 seconds)
        - Pattern timestep: 1 minute (60 seconds)

    If a time offset is provided, the demand patterns are rotated
    before simulation to represent the correct time of week.

    Returns p and q as (12, 5) arrays of pressure (metres) and
    flow rate (m³/s) at the five sensor locations.
    """
    # Load the network from the .inp file
    wn = wntr.network.WaterNetworkModel(str(inp_path))

    # Apply simulation time settings to match training data conditions
    wn.options.time.duration           = 3 * 3600   # 3-hour simulation
    wn.options.time.hydraulic_timestep = 15 * 60    # report every 15 minutes
    wn.options.time.report_timestep    = 15 * 60
    wn.options.time.pattern_timestep   = 60         # patterns step every 1 minute
    wn.options.time.start_clocktime    = 0          # clock always starts at 0

    # Rotate patterns to represent the user-selected day and time
    if offset_steps > 0:
        apply_pattern_rotation(wn, offset_steps)

    # Run the simulation using the EPANET hydraulic solver
    results   = wntr.sim.EpanetSimulator(wn).run_sim()
    pressures = results.node["pressure"]
    flows     = results.link["flowrate"]

    # Check that all expected sensor nodes and links are present in the results
    missing_nodes = [n for n in SENSOR_NODES if n not in pressures.columns]
    missing_links = [lk for lk in SENSOR_LINKS if lk not in flows.columns]

    if missing_nodes or missing_links:
        raise ValueError(
            "ERROR: Required sensor nodes or links not found in simulation results.\n"
            f"Expected nodes: {', '.join(SENSOR_NODES)}\n"
            f"Expected links: {', '.join(SENSOR_LINKS)}\n"
            "Verify that the .inp file uses the correct node and link IDs."
        )

    # Extract 12 rows (one per 15-minute timestep)
    n_rows = 12
    p = pressures[SENSOR_NODES].values[:n_rows].astype(np.float32)
    q = flows[SENSOR_LINKS].values[:n_rows].astype(np.float32)

    # Negative pressures indicate a hydraulically invalid simulation
    if np.any(p < 0):
        raise ValueError(
            "ERROR: Negative pressures detected in simulation results.\n"
            "The .inp file may be hydraulically invalid or require pressure-driven analysis."
        )

    return p, q


def build_df(p: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    """
    Assembles the simulation results into a DataFrame matching the
    data.csv format used during training.

    Columns are: t, P2, P3, P4, P5, P6, Q1a, Q2a, Q3a, Q4a, Q5a
    where t is time in minutes and the sensor columns are pressure
    (metres) and flow rate (m³/s) respectively.
    """
    # Start with the time column, then add each sensor column by name
    df = pd.DataFrame({"t": T_AXIS_MIN})
    for i, node in enumerate(SENSOR_NODES):
        df[f"P{node}"] = p[:, i]
    for i, link in enumerate(SENSOR_LINKS):
        df[f"Q{link}"] = q[:, i]
    return df


# PLOTS

def generate_plots(deviation: np.ndarray, sensor_names: list, plots_dir: Path):
    """
    Generates and saves 10 deviation-from-baseline plots, one for each
    sensor channel, into the specified output folder.

    The deviation signal (raw reading minus baseline) is the second
    feature used by the model. Plotting it shows how much the sensor
    readings changed relative to normal conditions — a sustained
    deviation is the main indicator of a leak.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)

    for idx, name in enumerate(sensor_names):
        # Determine sensor type to set colour and axis label correctly
        is_pressure = name.startswith("P")
        colour      = "blue" if is_pressure else "orange"
        y_unit      = "(m)" if is_pressure else "(m³/s)"
        y_label     = f"Deviation from Baseline {y_unit}"

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(T_AXIS_MIN, deviation[:, idx], color=colour, linewidth=1.8)

        # Zero reference line — deviations above or below this indicate a signal change
        ax.axhline(0, color="grey", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(y_label)
        ax.set_title(f"Sensor {name} — Deviation from Baseline")
        ax.grid(True)

        fig.savefig(str(plots_dir / f"deviation_{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)  # close figure to free memory before the next iteration


# INFERENCE

def run_inference(raw: np.ndarray, model: nn.Module, bundle: dict):
    """
    Prepares the raw sensor signals as model input, runs a forward
    pass through the ST-GCN, and returns the predictions.

    Feature construction:
        Feature 0 — raw sensor reading (as recorded by the simulation)
        Feature 1 — deviation from baseline (raw minus the mean no-leak signal)

    The baseline template stored in the bundle is the average no-leak
    signal computed from training data and represents normal network
    conditions.

    After feature construction, z-score normalisation is applied using
    the training mean (mu) and standard deviation (sigma) stored in the
    bundle.

    Arguments:
        raw   — (12, 10) array of raw sensor readings in sensor_names order
        model — the loaded SingleLeakSTGCN model
        bundle — the full model bundle dictionary

    Returns:
        pred_detect  — 0 (no leak) or 1 (leak detected)
        pred_pipe_id — pipe number (1-5) if leak detected, None otherwise
        pred_pos     — normalised position along the pipe [0.0, 1.0]
        deviation    — (12, 10) array of deviation signals used for plotting
    """
    # Load normalisation statistics and baseline from the bundle
    baseline  = np.array(bundle["baseline_template"], dtype=np.float32)  # (12, 10)
    mu        = np.array(bundle["mu"],    dtype=np.float32)               # (10, 2)
    sigma     = np.array(bundle["sigma"], dtype=np.float32)               # (10, 2)

    # Compute deviation: how much each sensor reading differs from normal
    deviation = raw - baseline                                            # (12, 10)

    # Stack raw and deviation into a two-channel feature array
    feats     = np.stack([raw, deviation], axis=-1).astype(np.float32)   # (12, 10, 2)

    # Z-score normalisation: subtract mean and divide by standard deviation
    # The small constant (1e-8) prevents division by zero
    feats     = (feats - mu[None]) / (sigma[None] + 1e-8)

    # Add a batch dimension — the model expects (batch, time, nodes, features)
    x_tensor  = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)    # (1, 12, 10, 2)

    # Run the forward pass with gradients disabled (not needed for inference)
    with torch.no_grad():
        detect_logits, pipe_logits, _, pos_pred = model(x_tensor)

    # Take the argmax of each classification head to get the predicted class
    pred_detect  = int(detect_logits.argmax(dim=1).item())
    pred_pipe    = int(pipe_logits.argmax(dim=1).item())
    pred_pos     = float(pos_pred.item())

    # Clamp position prediction to [0, 1] in case of floating point overflow
    pred_pos     = max(0.0, min(1.0, pred_pos))

    # Convert pipe index (0-4) back to pipe number (1-5)
    # If the model predicted NONE (index 5), return None instead
    pred_pipe_id = (pred_pipe + 1) if pred_pipe < NUM_PIPES else None

    return pred_detect, pred_pipe_id, pred_pos, deviation


# TERMINAL OUTPUT

def print_result(scenario_name: str, pred_detect: int,
                 pred_pipe_id, pred_pos: float,
                 out_csv: Path, plots_dir: Path,
                 sim_time_str: str = ""):
    """
    Prints the inference result to the terminal in a clearly formatted
    summary block.

    The paths to the saved data.csv and plots folder are also printed
    so the user knows where to find the outputs.
    """
    sep  = "=" * 60
    dash = "-" * 60

    if pred_detect == 1:
        detect_str = "LEAK DETECTED"
        pipe_str   = f"Pipe {pred_pipe_id}" if pred_pipe_id is not None else "N/A"
        pos_norm   = f"{pred_pos:.2f} (normalised)"
        # Convert normalised position to metres using known pipe length
        if pred_pipe_id is not None and pred_pipe_id in PIPE_LENGTHS:
            metres = pred_pos * PIPE_LENGTHS[pred_pipe_id]
            pos_m  = f"{metres:.1f} m from pipe start"
        else:
            pos_m  = "N/A"
    else:
        # If no leak detected, suppress pipe and position outputs
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


# USER FLOW

def list_inp_files():
    """
    Scans the inp_scenarios/ folder for .inp files and returns them
    as a sorted list.

    If the folder does not exist, it is created and the user is
    instructed to add their .inp file before running again.

    If the folder exists but is empty, the same message is shown
    and the script exits cleanly.
    """
    # Create the folder if it does not exist yet
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
    Prompts the user to select a day of the week and an hour of the
    day to set the simulation start time within the weekly demand cycle.
    """
    print("\nSelect simulation start day:")
    for i, day in enumerate(DAYS, start=1):
        print(f"  [{i}] {day}")

    # Keep prompting until a valid day number is entered
    while True:
        try:
            day_raw = input("Enter day number (1-7): ").strip()
            day_num = int(day_raw)
            if 1 <= day_num <= 7:
                break
            print("  Please enter a number between 1 and 7.")
        except ValueError:
            print("  Invalid input. Please enter a number.")

    # Keep prompting until a valid hour is entered
    while True:
        try:
            hour_raw = input("Enter simulation start hour (0-23): ").strip()
            hour = int(hour_raw)
            if 0 <= hour <= 23:
                break
            print("  Please enter a number between 0 and 23.")
        except ValueError:
            print("  Invalid input. Please enter a number.")

    # Convert the selected day and hour into a total minute offset from Sunday midnight
    selected_day_index = day_num - 1
    day_minutes        = selected_day_index * 24 * 60  # minutes from Sunday to selected day
    hour_minutes       = hour * 60                     # minutes from midnight to selected hour
    offset_steps       = day_minutes + hour_minutes    # total offset in minutes (= pattern steps)

    sim_time_str = f"{DAYS[selected_day_index]} at {hour:02d}:00"
    print(f"Simulation start: {sim_time_str}")

    return offset_steps


def select_file(files: list) -> Path:
    """
    Displays a numbered list of available .inp files and prompts the
    user to select one by entering its number.

    Keeps prompting until a valid number is entered.
    """
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
    """
    Runs the full pipeline for a single scenario:
        1. Simulates the .inp file using WNTR
        2. Saves the sensor signals as data.csv
        3. Runs inference through the ST-GCN model
        4. Generates the 10 deviation plots
        5. Prints the result to the terminal
    """
    # Use the filename (without extension) as the scenario identifier
    scenario_name = inp_path.stem
    out_dir       = PREDICTIONS / scenario_name
    plots_dir     = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/4] Simulating {inp_path.name} ...")
    try:
        p, q = simulate_inp(inp_path, offset_steps)
    except ValueError as e:
        # Catch known errors (missing sensors, negative pressures) and print them
        print(str(e))
        return
    except Exception as e:
        # Catch any unexpected simulation failure and inform the user
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

    # Extract the 10 sensor columns in the order the model expects them
    sensor_names = bundle["sensor_names"]
    raw          = df[sensor_names].to_numpy(dtype=np.float32)  # (12, 10)

    print("[3/4] Running inference ...")
    pred_detect, pred_pipe_id, pred_pos, deviation = run_inference(raw, model, bundle)

    print("[4/4] Generating deviation plots ...")
    generate_plots(deviation, sensor_names, plots_dir)

    print_result(scenario_name, pred_detect, pred_pipe_id, pred_pos, out_csv, plots_dir,
                 sim_time_str)


def main():
    """
    Loads the model bundle once at startup,
    then enters a loop that lets the user run as many scenarios as needed
    without reloading the model each time.

    The loop flow is:
        1. List available .inp files
        2. User selects a file
        3. User selects a simulation start day and hour
        4. The scenario is processed and results are printed
        5. User chooses to run another scenario or exit
    """
    print("Loading model bundle ...")
    model, bundle = load_bundle()
    print(f"[OK] Loaded bundle from {BUNDLE_PATH}")

    while True:
        # Show available files and get user selection
        files        = list_inp_files()
        inp_path     = select_file(files)

        # Get simulation time from the user and convert to a readable string
        offset_steps  = ask_simulation_time()
        total_minutes = offset_steps
        day_index     = (total_minutes // (24 * 60)) % 7
        hour          = (total_minutes % (24 * 60)) // 60
        sim_time_str  = f"{DAYS[day_index]} at {hour:02d}:00"

        # Run the full pipeline for the selected scenario
        process_scenario(inp_path, model, bundle, offset_steps, sim_time_str)

        # Ask if the user wants to run another scenario
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
