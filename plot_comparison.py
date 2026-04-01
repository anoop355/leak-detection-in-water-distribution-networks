"""
Leak vs No-Leak Sensor Comparison Plot
=======================================
Loads one leak scenario and its matching no-leak scenario (same repetition =
same start day/time and same base demand file), then produces a 2×5 grid:

    Top row    : Pressure sensors  P2  P3  P4  P5  P6
    Bottom row : Flow sensors      Q1a Q2a Q3a Q4a Q5a

Each subplot y-axis starts at the initial sensor reading (t = 0) so that
small leak-induced deviations from baseline are clearly visible.
A vertical dotted line marks the leak onset (t = 60 min, timestep 4).

Usage
-----
    python plot_comparison.py <leak_scenario_id>

Example
-------
    python plot_comparison.py 1
    python plot_comparison.py 42
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths  (relative to wherever the script is run from)
# ---------------------------------------------------------------------------
DATASET_ROOT  = Path("stgcn_dataset")
SCENARIOS_DIR = DATASET_ROOT / "scenarios"
MANIFEST_FILE = DATASET_ROOT / "manifests" / "manifest_full.csv"

# ---------------------------------------------------------------------------
# Sensor order
# ---------------------------------------------------------------------------
PRESSURE_COLS = ["P2", "P3", "P4", "P5", "P6"]
FLOW_COLS     = ["Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]

LEAK_ONSET_MIN = 60   # timestep 4 × 15 min = 60 min


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_scenario(scn_id: int):
    scn_dir = SCENARIOS_DIR / f"scenario_{scn_id:05d}"
    if not scn_dir.exists():
        raise FileNotFoundError(f"Scenario folder not found: {scn_dir}")
    data   = pd.read_csv(scn_dir / "data.csv")
    labels = json.loads((scn_dir / "labels.json").read_text(encoding="utf-8"))
    return data, labels


def find_matching_noleak(leak_labels: dict, manifest: pd.DataFrame) -> int:
    """Return the scenario_id of the first valid no-leak scenario whose
    repetition matches that of the given leak scenario (same start day/time
    and same base demand file)."""
    rep = leak_labels["repetition"]
    matches = manifest[
        (manifest["label_detection"] == 0) &
        (manifest["repetition"]      == rep) &
        (manifest["valid"]           == 1)
    ]
    if matches.empty:
        raise RuntimeError(
            f"No valid no-leak scenario found for repetition={rep}.  "
            "Make sure the dataset was generated successfully."
        )
    return int(matches.iloc[0]["scenario_id"])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(leak_scn_id: int, save_png: bool = True):
    # --- Load manifest and both scenarios ---
    manifest = pd.read_csv(MANIFEST_FILE)

    leak_data, leak_labels = load_scenario(leak_scn_id)

    if leak_labels["label_detection"] != 1:
        raise ValueError(
            f"Scenario {leak_scn_id} is labelled as no-leak "
            f"(label_detection={leak_labels['label_detection']}).  "
            "Please supply a leak scenario ID."
        )

    noleak_id = find_matching_noleak(leak_labels, manifest)
    noleak_data, noleak_labels = load_scenario(noleak_id)

    t = leak_data["t"].values   # 0, 15, 30, ..., 165  (minutes)

    # --- Build figure ---
    fig, axes = plt.subplots(
        2, 5,
        figsize=(24, 9),
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
        constrained_layout=False,
    )

    # --- Shared title ---
    title_top = (
        f"Leak scenario {leak_scn_id}  vs  No-leak scenario {noleak_id}   "
        f"|   {leak_labels['start_day']} {leak_labels['start_time']}   "
        f"|   Base: {leak_labels['source_inp']}"
    )
    title_bot = (
        f"Pipe {leak_labels['label_pipe']}   "
        f"Position {leak_labels['label_position']}   "
        f"Size {leak_labels['label_size']}   "
        f"Emitter coeff {leak_labels['emitter_coeff']} LPS   "
        f"|   Leak onset: t = {LEAK_ONSET_MIN} min  (timestep {leak_labels['leak_onset_step']})"
    )
    fig.suptitle(f"{title_top}\n{title_bot}", fontsize=10.5, fontweight="bold", y=1.01)

    # --- Row labels (placed on figure coordinates to avoid tight_layout clash) ---
    fig.text(0.005, 0.73, "PRESSURE  (m)", fontsize=10, fontweight="bold",
             color="#1a5276", rotation=90, va="center", ha="center")
    fig.text(0.005, 0.27, "FLOW  (m³/s)", fontsize=10, fontweight="bold",
             color="#117a65", rotation=90, va="center", ha="center")

    all_sensors = PRESSURE_COLS + FLOW_COLS
    row_colors  = ["#1a5276", "#117a65"]   # pressure, flow

    for idx, col in enumerate(all_sensors):
        row     = idx // 5
        col_idx = idx % 5
        ax      = axes[row][col_idx]

        yl = leak_data[col].values
        yn = noleak_data[col].values

        # --- Y-axis range: start at the initial sensor reading ---
        # The floor is set to min(yl[0], yn[0]) so the graph "starts at"
        # the baseline level and small deviations are clearly visible.
        y_floor  = min(yl[0], yn[0])
        y_all    = np.concatenate([yl, yn])
        y_min    = min(y_all.min(), y_floor)
        y_max    = y_all.max()
        y_range  = max(y_max - y_min, 1e-9)
        pad_bot  = 0.20 * y_range   # extra space below floor for visibility
        pad_top  = 0.15 * y_range

        # --- Shading: Phase 1 (pre-leak) green, Phase 2+3 (post-onset) red ---
        ax.axvspan(0,               LEAK_ONSET_MIN, alpha=0.07,
                   color="#27ae60", zorder=0, label="_nolegend_")
        ax.axvspan(LEAK_ONSET_MIN,  t[-1],          alpha=0.07,
                   color="#e74c3c", zorder=0, label="_nolegend_")

        # --- Sensor lines ---
        ax.plot(t, yn, color="#2471a3", lw=2.0, zorder=3,
                label=f"No-leak (scn {noleak_id})")
        ax.plot(t, yl, color="#c0392b", lw=2.0, ls="--", zorder=4,
                label=f"Leak (scn {leak_scn_id})")

        # --- Leak onset line ---
        ax.axvline(x=LEAK_ONSET_MIN, color="#e67e22", lw=1.4, ls=":",
                   zorder=5, label="Leak onset")

        # --- Axes formatting ---
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(y_floor - pad_bot, y_max + pad_top)

        ax.set_title(col, fontsize=11, fontweight="bold",
                     color=row_colors[row], pad=4)
        ax.set_xlabel("Time (min)", fontsize=8)
        ax.set_ylabel(
            "Pressure (m)" if row == 0 else "Flow (m³/s)",
            fontsize=8,
        )
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.35, linestyle="--")

        # --- X-ticks at every 30 min ---
        ax.set_xticks(range(0, int(t[-1]) + 1, 30))

        # --- Legend only on first subplot ---
        if idx == 0:
            ax.legend(fontsize=7.5, loc="lower left",
                      framealpha=0.85, edgecolor="grey")

    # --- Phase annotation below the bottom-right subplot ---
    fig.text(
        0.5, -0.015,
        "◀  Phase 1 (pre-leak baseline, t = 0–45 min)  |  "
        "Phase 2+3 (leak active, t = 60–165 min)  ▶",
        ha="center", fontsize=8.5, color="#555555",
    )

    # --- Save and/or show ---
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.10,
                        hspace=0.45, wspace=0.38)
    if save_png:
        out_path = Path(f"comparison_scn{leak_scn_id:05d}_vs_scn{noleak_id:05d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[OK] Plot saved: {out_path.resolve()}", flush=True)

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare leak vs no-leak sensor data from the ST-GCN dataset."
    )
    parser.add_argument(
        "leak_scenario_id",
        type=int,
        help="Scenario ID of the leak scenario to plot (e.g. 1, 42, 500).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display the plot without saving a PNG file.",
    )
    args = parser.parse_args()

    try:
        plot_comparison(args.leak_scenario_id, save_png=not args.no_save)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
