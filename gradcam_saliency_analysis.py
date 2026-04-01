"""
gradcam_saliency_analysis.py

Input Gradient (Saliency) Analysis for Leak Count Misclassification.

Purpose:
    For triple-leak scenarios misclassified as two-leak, compute the
    input gradient with respect to the predicted Class 2 count logit.
    This reveals which sensors and time steps the model attended to
    when making the wrong prediction.

Usage:
    1. Set the paths in USER SETTINGS below.
    2. Run the script. It will:
       - Identify misclassified triple-leak scenarios from the CSV.
       - For each selected scenario type (L,S,S and M,M,M), load one
         representative scenario, run a forward pass, compute gradients,
         and save a heatmap figure.

Outputs (saved to OUTPUT_DIR):
    - saliency_LSS_scn{N}.png   : heatmap for a (L,S,S) misclassified scenario
    - saliency_MMM_scn{N}.png   : heatmap for a (M,M,M) misclassified scenario
    - saliency_SSS_correct_scn{N}.png : heatmap for a correctly classified (S,S,S)
                                        triple-leak scenario (reference)
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ============================================================
# USER SETTINGS — edit these only
# ============================================================

# Path to the trained full-baseline (k=10) model bundle
BUNDLE_PATH = Path("multileak_tcn_bundle.pt")

# Path to the per-scenario predictions CSV (from evaluate_model.py)
PER_SCENARIO_CSV = Path("test_data_results/evaluation/per_scenario_metrics.csv")

# Root directory containing test scenario folders (scn_001, scn_002, ...)
TEST_DATA_DIR = Path("test_data_results")

# Output directory for figures
OUTPUT_DIR = Path("saliency_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL DEFINITION — must match training exactly
# ============================================================
MAX_LEAKS     = 3
NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES
PIPE_CLASSES  = NUM_PIPES + 1
SIZE_CLASSES  = 4


class MultiLeakTCN(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=4,  dilation=1), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=8,  dilation=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=16, dilation=4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.count_head = nn.Linear(32, 4)
        self.pipe_head  = nn.Linear(32, MAX_LEAKS * PIPE_CLASSES)
        self.size_head  = nn.Linear(32, MAX_LEAKS * SIZE_CLASSES)
        self.pos_head   = nn.Linear(32, MAX_LEAKS)

    def forward(self, x):
        z = self.backbone(x)
        count_logits = self.count_head(z)
        pipe_logits  = self.pipe_head(z).view(-1, MAX_LEAKS, PIPE_CLASSES)
        size_logits  = self.size_head(z).view(-1, MAX_LEAKS, SIZE_CLASSES)
        pos_pred     = self.pos_head(z).view(-1, MAX_LEAKS)
        return count_logits, pipe_logits, size_logits, pos_pred


# ============================================================
# HELPER: load scenario signals and apply normalisation
# ============================================================

def load_scenario(scn_dir: Path, feature_cols, mu, sigma, window=180):
    """
    Load signals from a scenario folder, normalise, and return
    the middle window as a tensor of shape (1, C, T).
    The middle window is used to avoid transient startup effects.
    """
    signals = pd.read_csv(scn_dir / "signals.csv")
    X = signals[feature_cols].values.astype(np.float32)
    X = (X - mu) / (sigma + 1e-8)

    # Use the middle window of the scenario
    mid = len(X) // 2
    start = max(0, mid - window // 2)
    Xw = X[start: start + window]

    # Shape: (C, T) then add batch dim -> (1, C, T)
    tensor = torch.tensor(Xw.T, dtype=torch.float32).unsqueeze(0)
    return tensor


# ============================================================
# CORE: compute input gradient saliency
# ============================================================

def compute_saliency(model, x_tensor, target_class):
    """
    Compute the input gradient with respect to target_class count logit.

    Parameters
    ----------
    model        : trained MultiLeakTCN
    x_tensor     : (1, C, T) input tensor, requires_grad will be set here
    target_class : int, the count class to differentiate with respect to

    Returns
    -------
    saliency : numpy array of shape (C, T)
               Absolute value of the gradient — larger = more influential.
    predicted_class : int
    """
    model.eval()
    x = x_tensor.clone().detach().to(DEVICE)
    x.requires_grad_(True)

    count_logits, _, _, _ = model(x)

    # Zero all gradients, then backpropagate through target class logit only
    model.zero_grad()
    score = count_logits[0, target_class]
    score.backward()

    # Gradient shape: (1, C, T) -> take absolute value -> (C, T)
    saliency = x.grad.data.abs().squeeze(0).cpu().numpy()
    predicted_class = count_logits.argmax(dim=1).item()

    return saliency, predicted_class


# ============================================================
# PLOTTING
# ============================================================

SENSOR_LABELS = ["P2", "P3", "P4", "P5", "P6",
                 "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]


def plot_saliency(saliency, scenario_label, predicted_class, true_class,
                  target_class, save_path):
    """
    Plot a heatmap of saliency (C=10 sensors x T=180 time steps).
    Rows = sensors, columns = time steps.
    Also plot the per-sensor mean saliency as a bar chart alongside.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [4, 1]},
        constrained_layout=True
    )

    # --- Heatmap ---
    ax = axes[0]
    im = ax.imshow(
        saliency,
        aspect="auto",
        cmap="hot",
        interpolation="nearest",
        norm=mcolors.PowerNorm(gamma=0.4)   # compress dynamic range
    )
    ax.set_yticks(range(len(SENSOR_LABELS)))
    ax.set_yticklabels(SENSOR_LABELS, fontsize=10)
    ax.set_xlabel("Time Step (within 180-step window)", fontsize=10)
    ax.set_ylabel("Sensor Channel", fontsize=10)
    ax.set_title(
        f"Input Gradient Saliency Map\n"
        f"Scenario: {scenario_label} | True Class: {true_class} leaks | "
        f"Predicted: {predicted_class} leaks | Gradient w.r.t. Class {target_class}",
        fontsize=10
    )
    plt.colorbar(im, ax=ax, label="Gradient Magnitude")

    # Draw horizontal divider between pressure and flow sensors
    ax.axhline(y=4.5, color="cyan", linewidth=1.5, linestyle="--")
    ax.text(182, 2, "Pressure\nSensors", fontsize=8, color="cyan",
            va="center", ha="left", clip_on=False)
    ax.text(182, 7, "Flow\nSensors", fontsize=8, color="cyan",
            va="center", ha="left", clip_on=False)

    # --- Per-sensor mean saliency bar chart ---
    ax2 = axes[1]
    mean_saliency = saliency.mean(axis=1)   # shape (C,)
    colours = ["steelblue"] * 5 + ["darkorange"] * 5
    bars = ax2.barh(range(len(SENSOR_LABELS)), mean_saliency, color=colours)
    ax2.set_yticks(range(len(SENSOR_LABELS)))
    ax2.set_yticklabels(SENSOR_LABELS, fontsize=10)
    ax2.set_xlabel("Mean Gradient", fontsize=9)
    ax2.set_title("Mean\nSaliency", fontsize=10)
    ax2.axhline(y=4.5, color="grey", linewidth=1, linestyle="--")
    ax2.invert_yaxis()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def find_scenario_dir(scn_number: int) -> Path:
    """Find the test scenario folder matching a scenario number."""
    candidates = list(TEST_DATA_DIR.glob(f"scn_{scn_number:03d}"))
    if not candidates:
        candidates = list(TEST_DATA_DIR.glob(f"scn_{scn_number}"))
    if not candidates:
        # Broader search
        for d in TEST_DATA_DIR.iterdir():
            labels_path = d / "labels.json"
            if labels_path.exists():
                labels = json.loads(labels_path.read_text())
                if int(labels.get("scn_number", -1)) == scn_number:
                    return d
    return candidates[0] if candidates else None


def main():
    # --- Load model bundle ---
    bundle = torch.load(str(BUNDLE_PATH), map_location=DEVICE, weights_only=False)
    mu           = np.array(bundle["mu"],           dtype=np.float32)
    sigma        = np.array(bundle["sigma"],        dtype=np.float32)
    feature_cols = list(bundle["feature_cols"])
    window       = int(bundle.get("window", 180))

    model = MultiLeakTCN(C=len(feature_cols)).to(DEVICE)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    # --- Load per-scenario CSV ---
    df = pd.read_csv(PER_SCENARIO_CSV)

    # Identify misclassified triple-leak scenarios (true=3, pred=2)
    missed_triple = df[(df["true_count"] == 3) & (df["pred_count"] == 2)]
    print(f"Total triple-leak scenarios misclassified as 2: {len(missed_triple)}")

    # Identify correctly classified triple-leak scenarios (true=3, pred=3)
    correct_triple = df[(df["true_count"] == 3) & (df["pred_count"] == 3)]
    print(f"Total triple-leak scenarios correctly classified: {len(correct_triple)}")

    # ----------------------------------------------------------------
    # Select representative scenarios for each category.
    # NOTE: The CSV does not contain size labels. You must manually
    # identify one (L,S,S) and one (M,M,M) scenario number from your
    # scenario folder labels.json files and enter them below.
    #
    # To find them quickly, run this in your terminal:
    #   grep -r '"size_level": "L"' test_data_results/scn_*/labels.json
    # and cross-reference with the missed_triple scenario numbers.
    # ----------------------------------------------------------------

    # --- Replace these with actual scenario numbers from your data ---
    LSS_SCN  = missed_triple["scn_number"].iloc[0]   # replace with a known (L,S,S) scn
    MMM_SCN  = missed_triple["scn_number"].iloc[1]   # replace with a known (M,M,M) scn
    SSS_SCN  = correct_triple["scn_number"].iloc[0]  # correctly classified (S,S,S) reference
    # -----------------------------------------------------------------

    scenarios_to_plot = [
        (LSS_SCN,  3, 2, "Misclassified (L,S,S)"),
        (MMM_SCN,  3, 2, "Misclassified (M,M,M)"),
        (SSS_SCN,  3, 3, "Correctly Classified (S,S,S)"),
    ]

    for scn_number, true_class, target_class, label in scenarios_to_plot:
        scn_dir = find_scenario_dir(scn_number)
        if scn_dir is None:
            print(f"[WARN] Scenario directory not found for scn_number={scn_number}. Skipping.")
            continue

        x_tensor = load_scenario(scn_dir, feature_cols, mu, sigma, window)
        saliency, predicted_class = compute_saliency(model, x_tensor, target_class)

        save_name = f"saliency_{label.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '')}_scn{scn_number}.png"
        plot_saliency(
            saliency=saliency,
            scenario_label=label,
            predicted_class=predicted_class,
            true_class=true_class,
            target_class=target_class,
            save_path=OUTPUT_DIR / save_name
        )

    print("\nDone. Figures saved to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()