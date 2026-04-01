"""
plot_scenario_comparison.py

Plots sensor data from a specified leak scenario alongside its matching
no-leak scenario (same start day, start time, and source .inp file).

Usage:
    Set LEAK_SCENARIO_ID at the bottom of the script and run.

Output:
    A single figure with 10 subplots:
        - Top row (5 plots):    Pressure sensors P2, P3, P4, P5, P6 vs time
        - Bottom row (5 plots): Flow sensors Q1a, Q2a, Q3a, Q4a, Q5a vs time

    Each plot overlays:
        - Blue line:  No-leak scenario
        - Red line:   Leak scenario
        - Vertical dashed line at t=60 min: leak onset
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


# -------------------------
# Configuration
# -------------------------
DATASET_DIR = Path(
    r"C:\Users\anoop\OneDrive - The University of the West Indies, "
    r"St. Augustine\Desktop\First_WDN\stgcn_dataset"
)

SCENARIOS_DIR = DATASET_DIR / "scenarios"
LEAK_ONSET_MIN = 60  # minutes — leak becomes active at t=60 (timestep 4)

PRESSURE_SENSORS = ["P2", "P3", "P4", "P5", "P6"]
FLOW_SENSORS     = ["Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]


# -------------------------
# Helper functions
# -------------------------
def load_scenario(scenario_id: int):
    """
    Loads labels.json and data.csv for a given scenario ID.
    Returns (labels_dict, dataframe) or raises FileNotFoundError.
    """
    folder = SCENARIOS_DIR / f"scenario_{scenario_id:05d}"

    labels_path = folder / "labels.json"
    data_path   = folder / "data.csv"

    if not folder.exists():
        raise FileNotFoundError(f"Scenario folder not found: {folder}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found in: {folder}")
    if not data_path.exists():
        raise FileNotFoundError(f"data.csv not found in: {folder}")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    df = pd.read_csv(data_path)

    return labels, df


def find_matching_no_leak(leak_labels: dict) -> tuple:
    """
    Searches all scenario folders for a no-leak scenario that matches
    the leak scenario on start_day, start_time, and source_inp.

    Returns (scenario_id, labels, dataframe) of the matching no-leak scenario.
    Raises RuntimeError if no match is found.
    """
    target_day   = leak_labels["start_day"]
    target_time  = leak_labels["start_time"]
    target_inp   = leak_labels["source_inp"]

    print(f"Searching for no-leak scenario matching:")
    print(f"  start_day  : {target_day}")
    print(f"  start_time : {target_time}")
    print(f"  source_inp : {target_inp}")
    print()

    for folder in sorted(SCENARIOS_DIR.iterdir()):
        if not folder.is_dir():
            continue

        labels_path = folder / "labels.json"
        if not labels_path.exists():
            continue

        with open(labels_path, "r", encoding="utf-8") as f:
            candidate = json.load(f)

        if candidate.get("label_detection") != 0:
            continue  # skip leak scenarios

        if (
            candidate.get("start_day")   == target_day  and
            candidate.get("start_time")  == target_time and
            candidate.get("source_inp")  == target_inp
        ):
            scn_id = candidate["scenario_id"]
            df = pd.read_csv(folder / "data.csv")
            print(f"Matching no-leak scenario found: scenario_{scn_id:05d}")
            print(f"  start_day  : {candidate['start_day']}")
            print(f"  start_time : {candidate['start_time']}")
            print(f"  source_inp : {candidate['source_inp']}")
            print()
            return scn_id, candidate, df

    raise RuntimeError(
        f"No matching no-leak scenario found for "
        f"start_day='{target_day}', start_time='{target_time}', "
        f"source_inp='{target_inp}'."
    )


# -------------------------
# Plotting
# -------------------------
def plot_comparison(leak_scenario_id: int):
    """
    Main plotting function. Loads the specified leak scenario and its
    matching no-leak scenario, then produces a 2x5 comparison figure.
    """

    # --- Load leak scenario ---
    print(f"Loading leak scenario: scenario_{leak_scenario_id:05d}")
    leak_labels, leak_df = load_scenario(leak_scenario_id)

    if leak_labels.get("label_detection") != 1:
        raise ValueError(
            f"scenario_{leak_scenario_id:05d} is not a leak scenario "
            f"(label_detection = {leak_labels.get('label_detection')})."
        )

    print(f"  Pipe         : {leak_labels['label_pipe']}")
    print(f"  Position     : {leak_labels['label_position']}")
    print(f"  Size         : {leak_labels['label_size']}")
    print(f"  Start day    : {leak_labels['start_day']}")
    print(f"  Start time   : {leak_labels['start_time']}")
    print(f"  Source .inp  : {leak_labels['source_inp']}")
    print()

    # --- Find matching no-leak scenario ---
    no_leak_id, no_leak_labels, no_leak_df = find_matching_no_leak(leak_labels)

    # --- Build figure ---
    fig, axes = plt.subplots(
        nrows=2,
        ncols=5,
        figsize=(22, 9),
        sharey=False
    )

    # Title
    fig.suptitle(
        f"Scenario Comparison — Leak: scenario_{leak_scenario_id:05d}  |  "
        f"No-Leak: scenario_{no_leak_id:05d}\n"
        f"Pipe {leak_labels['label_pipe']} | "
        f"Position {leak_labels['label_position']} | "
        f"Size {leak_labels['label_size']} | "
        f"{leak_labels['start_day']} {leak_labels['start_time']} | "
        f"{leak_labels['source_inp']}",
        fontsize=13,
        fontweight="bold",
        y=1.01
    )

    time_col = "t"  # time column name in data.csv

    # --- Top row: Pressure sensors ---
    for col_idx, sensor in enumerate(PRESSURE_SENSORS):
        ax = axes[0][col_idx]

        no_leak_vals = no_leak_df[sensor].values
        leak_vals    = leak_df[sensor].values
        time_vals    = leak_df[time_col].values

        # Y axis starts at the minimum of the no-leak starting value
        y_start = min(no_leak_vals[0], leak_vals[0])
        y_end   = max(no_leak_vals.max(), leak_vals.max())
        y_pad   = (y_end - y_start) * 0.1 if (y_end - y_start) > 0 else 1.0

        ax.plot(
            time_vals, no_leak_vals,
            color="steelblue", linewidth=2.0,
            label="No-Leak", zorder=3
        )
        ax.plot(
            time_vals, leak_vals,
            color="crimson", linewidth=2.0,
            linestyle="--", label="Leak", zorder=4
        )
        ax.axvline(
            x=LEAK_ONSET_MIN, color="darkorange",
            linewidth=1.5, linestyle=":", label="Leak onset"
        )

        ax.set_title(sensor, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylabel("Pressure (m)", fontsize=10)
        ax.set_xlim(left=time_vals[0], right=time_vals[-1])
        ax.set_ylim(bottom=y_start - y_pad, top=y_end + y_pad)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.grid(True, linestyle="--", alpha=0.4)

        if col_idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    # --- Bottom row: Flow sensors ---
    for col_idx, sensor in enumerate(FLOW_SENSORS):
        ax = axes[1][col_idx]

        no_leak_vals = no_leak_df[sensor].values
        leak_vals    = leak_df[sensor].values
        time_vals    = leak_df[time_col].values

        y_start = min(no_leak_vals[0], leak_vals[0])
        y_end   = max(no_leak_vals.max(), leak_vals.max())
        y_pad   = (y_end - y_start) * 0.1 if (y_end - y_start) > 0 else 0.001

        ax.plot(
            time_vals, no_leak_vals,
            color="steelblue", linewidth=2.0,
            label="No-Leak", zorder=3
        )
        ax.plot(
            time_vals, leak_vals,
            color="crimson", linewidth=2.0,
            linestyle="--", label="Leak", zorder=4
        )
        ax.axvline(
            x=LEAK_ONSET_MIN, color="darkorange",
            linewidth=1.5, linestyle=":", label="Leak onset"
        )

        ax.set_title(sensor, fontsize=12, fontweight="bold")
        ax.set_xlabel("Time (min)", fontsize=10)
        ax.set_ylabel("Flow (m³/s)", fontsize=10)
        ax.set_xlim(left=time_vals[0], right=time_vals[-1])
        ax.set_ylim(bottom=y_start - y_pad, top=y_end + y_pad)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax.grid(True, linestyle="--", alpha=0.4)

        if col_idx == 0:
            ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    plt.show()


# -------------------------
# Entry point
# -------------------------
if __name__ == "__main__":

    # --- Set the leak scenario ID you want to inspect here ---
    LEAK_SCENARIO_ID = 928

    plot_comparison(LEAK_SCENARIO_ID)