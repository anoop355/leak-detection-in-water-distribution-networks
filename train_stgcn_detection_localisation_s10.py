"""
Training script for the ST-GCN single-leak detection and localisation model
using the full 10-sensor configuration (P2-P6, Q1a-Q5a).

The model takes sliding windows of 12 timesteps (3 hours at 15-minute intervals)
as input and outputs four predictions simultaneously: detection, pipe, size,
and position. This is called a multi-task model because it learns all four
tasks at once using a shared backbone.

Before running this script, the training dataset must first be generated
using generate_stgcn_dataset_v2.py, which produces the stgcn_dataset/ folder.
"""

import os
import json
import math
import random
import bisect
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# CONFIG

# These paths point to the dataset folder and manifest CSVs generated
# by generate_stgcn_dataset_v2.py.
DATASET_ROOT   = "stgcn_dataset/scenarios"
MANIFEST_TRAIN = "stgcn_dataset/manifests/manifest_train.csv"
MANIFEST_VAL   = "stgcn_dataset/manifests/manifest_val.csv"
MANIFEST_TEST  = "stgcn_dataset/manifests/manifest_test.csv"

# Output path for the trained model bundle
SAVE_PATH      = "stgcn_bundle_v5_10ch.pt"

# All 10 sensor channels used in this configuration.
# P2-P6 are pressure sensors at the five network nodes.
# Q1a-Q5a are flow sensors at the five pipe measurement points.
SENSOR_NAMES = ["P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]
NUM_NODES = len(SENSOR_NAMES)   # 10 sensors total
NODE_FEATS = 2                  # two features per sensor: raw reading and deviation from baseline

# Each scenario has exactly 12 timesteps (one per 15 minutes over a 3-hour window)
# STRIDE=1 means every consecutive window of 12 steps is used as a training sample
WINDOW = 12
STRIDE = 1

# Training hyperparameters
BATCH_SIZE   = 64
EPOCHS       = 25
LR           = 1e-3    # learning rate for Adam optimiser
WEIGHT_DECAY = 1e-4    # L2 regularisation to reduce overfitting
DROPOUT      = 0.25    # fraction of neurons randomly disabled during training
SEED         = 42      # fixed seed for reproducibility

# Model architecture
HIDDEN_1    = 16   # output channels after the first ST block
HIDDEN_2    = 32   # output channels after the second and third ST blocks
KERNEL_SIZE = 5    # temporal convolution kernel size

# Pipe classification configuration
NUM_PIPES     = 5
PIPE_NONE_IDX = NUM_PIPES       # index 5 represents NONE (no leak in this scenario)
PIPE_CLASSES  = NUM_PIPES + 1   # 6 total classes: pipes 1-5 plus NONE

# Leak size classification
SIZE_TO_IDX  = {"S": 0, "M": 1, "L": 2}
SIZE_NONE_IDX = 3
SIZE_CLASSES  = 4   # S, M, L, NONE

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = DEVICE == "cuda"


# REPRODUCIBILITY

def set_seed(seed: int):
    """
    Sets the random seed for Python, NumPy, and PyTorch so that results
    are reproducible across different runs of the script.

    Without fixing the seed, random weight initialisation and data shuffling
    would produce slightly different results each time the script is run.
    Fixing it to 42 ensures the same model can be reproduced.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# LABEL HELPERS

def load_labels(path: str):
    """
    Reads a labels.json file from the given path and returns it as a
    Python dictionary. Each scenario folder contains one of these files
    which stores the ground truth labels for that scenario.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_no_leak(labels: dict):
    """
    Returns True if this scenario has no leak, False otherwise.
    Used to identify no-leak scenarios when building the baseline template.
    """
    return int(labels.get("label_detection", 0)) == 0

def encode_labels_from_json(labels: dict):
    """
    Converts a labels.json dictionary into the four scalar targets
    needed for training the multi-task model.

    Returns:
        detect  — 0 if no leak, 1 if leak present
        pipe_t  — pipe index (0 to 4 for pipes 1 to 5), or PIPE_NONE_IDX if no leak
        pos_t   — normalised position along the pipe (0.0 to 1.0), or 0.0 if no leak
        size_t  — size class index (0=S, 1=M, 2=L), or SIZE_NONE_IDX if no leak
    """
    detect = int(labels.get("label_detection", 0))

    if detect == 1:
        # Leak present — extract all localisation labels
        pipe_t = int(labels.get("label_pipe", 1)) - 1   # convert 1-5 to 0-4
        pos_t  = float(labels.get("label_position", 0.0))
        sl     = str(labels.get("label_size", "S")).upper()
        size_t = SIZE_TO_IDX.get(sl, 0)
    else:
        # No leak — set all localisation targets to their NONE values
        pipe_t = PIPE_NONE_IDX
        pos_t  = 0.0
        size_t = SIZE_NONE_IDX

    return detect, pipe_t, pos_t, size_t


# FILE / SCENARIO HELPERS

def folders_from_manifest(manifest_path: str):
    """
    Reads a manifest CSV file and returns the list of scenario folder paths
    for all valid scenarios (those that have both data.csv and labels.json).

    The manifest is a CSV produced by generate_stgcn_dataset_v2.py that lists
    all scenario IDs assigned to each split (train/val/test). This ensures
    the same split is used consistently every time the script is run.
    """
    df = pd.read_csv(manifest_path)
    folders = []
    for scn_id in df["scenario_id"].values:
        # Construct the expected folder path for this scenario ID
        path = os.path.join(DATASET_ROOT, f"scenario_{int(scn_id):05d}")
        sig  = os.path.join(path, "data.csv")
        lab  = os.path.join(path, "labels.json")
        # Only include folders where both required files are present
        if os.path.isfile(sig) and os.path.isfile(lab):
            folders.append(path)
    return folders

def read_signals(folder: str):
    """
    Reads the data.csv file from a scenario folder and returns the sensor
    columns as a NumPy array of shape (T, N) where T is the number of
    timesteps and N is the number of sensors.

    Raises an error if any expected sensor columns are missing, which would
    indicate a mismatch between the dataset and the current sensor configuration.
    """
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    missing = [c for c in SENSOR_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {folder}: {missing}")
    return df[SENSOR_NAMES].to_numpy(dtype=np.float32)


# BASELINE + NORMALISATION

def compute_baseline_template(train_folders):
    """
    Computes the baseline signal template by averaging all no-leak
    scenarios in the training split.

    The baseline represents what the sensor readings look like under
    normal (no-leak) network conditions. It is used to compute the
    deviation feature. How much each reading differs from normal.

    Only training data is used here to avoid data leakage. If the
    test data was included in computing the baseline, the model would
    have indirect access to test information during training.
    """
    no_leak_arrays = []
    for folder in train_folders:
        labels = load_labels(os.path.join(folder, "labels.json"))
        if is_no_leak(labels):
            arr = read_signals(folder)
            no_leak_arrays.append(arr)

    if len(no_leak_arrays) == 0:
        raise RuntimeError("No no-leak scenarios found in training split. Cannot build baseline template.")

    # Crop all arrays to the same length before averaging
    min_len = min(a.shape[0] for a in no_leak_arrays)
    cropped  = [a[:min_len] for a in no_leak_arrays]

    # Average across all no-leak scenarios to get the baseline shape (T, N)
    baseline = np.mean(np.stack(cropped, axis=0), axis=0).astype(np.float32)

    return baseline

def make_node_features(raw_signals: np.ndarray, baseline_template: np.ndarray):
    """
    Constructs the two-channel feature array for a single scenario.

    Feature 0 is the raw sensor reading as simulated by EPANET/WNTR.
    Feature 1 is the deviation from baseline

    Arguments:
        raw_signals:       (T, N) array of raw sensor readings
        baseline_template: (Tb, N) baseline array where Tb >= T

    Returns:
        feats: (T, N, 2) feature array
    """
    T  = raw_signals.shape[0]
    Tb = baseline_template.shape[0]

    if Tb < T:
        raise ValueError(
            f"Baseline template shorter than scenario length: baseline={Tb}, scenario={T}"
        )

    # Crop the baseline to match the scenario length
    base = baseline_template[:T]
    dev  = raw_signals - base   # deviation: positive means higher than normal

    # Stack the two features along a new last axis
    feats = np.stack([raw_signals, dev], axis=-1).astype(np.float32)
    return feats

def compute_mu_sigma(train_folders, baseline_template):
    """
    Computes the per-channel mean and standard deviation across all
    training scenarios for z-score normalisation.

    The result has shape (N, 2) — one mean and one std for each of
    the 10 sensors and 2 feature channels.
    """
    sum_x  = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    sum_x2 = np.zeros((NUM_NODES, NODE_FEATS), dtype=np.float64)
    total_T = 0

    for folder in train_folders:
        raw   = read_signals(folder)
        feats = make_node_features(raw, baseline_template)

        # Accumulate sum and sum of squares for each channel
        sum_x  += feats.sum(axis=0)
        sum_x2 += (feats ** 2).sum(axis=0)
        total_T += feats.shape[0]

    mu  = sum_x / total_T
    # Variance formula: E[x²] - E[x]²
    var = (sum_x2 / total_T) - (mu ** 2)
    # Clamp variance to a small positive value to avoid division by zero
    var = np.maximum(var, 1e-8)
    sigma = np.sqrt(var)

    return mu.astype(np.float32), sigma.astype(np.float32)


# GRAPH

def build_sensor_adjacency():
    """
    Builds the normalised adjacency matrix that describes which sensors
    are physically connected in the pipe network.

    The network topology is:
        Q1a -- P2 -- Q2a -- P3 -- Q3a -- P4
                                       /    \
                                    Q4a     Q5a
                                     |       |
                                    P5      P6

    The graph convolution layers use this matrix to allow each sensor 
    to receive information from its neighbouring sensors. This is 
    important because a leak in one pipe affects the pressure and 
    flow readings in adjacent pipes.

    Only edges where both endpoints are in SENSOR_NAMES are added,
    which means the same function would work correctly for any subset
    of sensors.

    Self-loops are added so each node also includes its own features
    after aggregation, not just its neighbours'.

    Symmetric normalisation (D^-0.5 * A * D^-0.5) scales the matrix
    so that nodes with more neighbours don't dominate the aggregation.
    """
    sensor_set = set(SENSOR_NAMES)
    idx = {name: i for i, name in enumerate(SENSOR_NAMES)}
    A = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    def connect(a, b, w=1.0):
        # Only add an edge if both sensors are in the current sensor set
        if a in sensor_set and b in sensor_set:
            i, j = idx[a], idx[b]
            A[i, j] = w
            A[j, i] = w   # undirected graph — edges go both ways

    # Add all physical connections based on the network layout
    connect("Q1a", "P2")
    connect("P2",  "Q2a")
    connect("Q2a", "P3")
    connect("P3",  "Q3a")
    connect("Q3a", "P4")
    connect("P4",  "Q4a")
    connect("Q4a", "P5")
    connect("P4",  "Q5a")
    connect("Q5a", "P6")

    # Add self-loops so each node retains its own features after aggregation
    A = A + np.eye(NUM_NODES, dtype=np.float32)

    # Symmetric normalisation: compute degree matrix and normalise
    deg = np.sum(A, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    return A_hat.astype(np.float32)


# MEMORY-EFFICIENT WINDOW DATASET

class ScenarioWindowDataset(Dataset):
    """
    A PyTorch Dataset that creates sliding windows over the sensor
    signals from each hydraulic scenario.

    Each scenario has 12 timesteps. With WINDOW=12 and STRIDE=1, each
    scenario produces exactly one window, which is the full 12-timestep
    signal.

    The dataset also applies scenario-level oversampling to balance
    the number of leak and no-leak training examples.

    All features are normalised using the training statistics (mu, sigma)
    before being stored.
    """
    def __init__(self, folders, baseline_template, mu, sigma, window=12, stride=1):
        self.window = window
        self.stride = stride

        # These lists store one entry per scenario
        self.features     = []   # normalised feature arrays, shape (T, N, 2)
        self.targets      = []   # label tuples: (detect, pipe_t, pos_t, size_t)
        self.names        = []   # scenario folder names (for debugging)
        self.window_counts = []  # number of windows each scenario produces
        self.cum_counts   = [0]  # cumulative window counts for index lookup

        for folder in folders:
            raw   = read_signals(folder)
            feats = make_node_features(raw, baseline_template)

            # Apply z-score normalisation using training statistics
            feats = (feats - mu[None, :, :]) / (sigma[None, :, :] + 1e-8)
            feats = feats.astype(np.float32)

            labels = load_labels(os.path.join(folder, "labels.json"))
            tgt    = encode_labels_from_json(labels)

            # Calculate how many windows this scenario produces
            T         = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            if n_windows == 0:
                continue  # skip scenarios that are too short for even one window

            self.features.append(feats)
            self.targets.append(tgt)
            self.names.append(os.path.basename(folder))
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        # ------------------------------------------------------------------
        # Scenario-level oversampling to balance leak vs no-leak classes.
        #
        # Instead of weighting individual windows, I repeat entire scenarios
        # from underrepresented classes so each class has roughly the same
        # number of scenarios before window expansion. This avoids fabricating
        # new data — it just gives minority classes more training exposure.
        # ------------------------------------------------------------------
        detect_per_scenario   = [t[0] for t in self.targets]
        class_scenario_counts = Counter(detect_per_scenario)
        max_scenarios         = max(class_scenario_counts.values())

        print(f"  Scenario counts before oversampling: {dict(sorted(class_scenario_counts.items()))}")

        new_features, new_targets, new_names = [], [], []

        for i, det in enumerate(detect_per_scenario):
            # How many times to repeat this scenario so its class catches up
            count_for_class = class_scenario_counts[det]
            repeats = max(1, round(max_scenarios / count_for_class))
            new_features.extend([self.features[i]] * repeats)
            new_targets.extend([self.targets[i]]   * repeats)
            new_names.extend([self.names[i]]        * repeats)

        self.features = new_features
        self.targets  = new_targets
        self.names    = new_names

        # Rebuild the cumulative window index after oversampling
        self.window_counts = []
        self.cum_counts    = [0]
        for feats in self.features:
            T         = feats.shape[0]
            n_windows = max(0, (T - window) // stride + 1)
            self.window_counts.append(n_windows)
            self.cum_counts.append(self.cum_counts[-1] + n_windows)

        new_det = [t[0] for t in self.targets]
        print(f"  Scenario counts after  oversampling: {dict(sorted(Counter(new_det).items()))}")

    def __len__(self):
        # Total number of windows across all scenarios
        return self.cum_counts[-1]

    def _locate_index(self, idx):
        """
        Maps a flat window index to (scenario index, local window index).

        Uses binary search on the cumulative counts array, which is
        faster than a linear scan when there are thousands of windows.
        """
        scenario_idx = bisect.bisect_right(self.cum_counts, idx) - 1
        local_idx    = idx - self.cum_counts[scenario_idx]
        return scenario_idx, local_idx

    def __getitem__(self, idx):
        """
        Returns the window at position idx as a tuple of tensors ready
        for the model's forward pass.
        """
        scenario_idx, local_idx = self._locate_index(idx)
        x_full = self.features[scenario_idx]

        # Extract the window slice from the full scenario
        start = local_idx * self.stride
        end   = start + self.window
        x     = x_full[start:end]  # shape: (W, N, 2)

        detect, pipe_t, pos_t, size_t = self.targets[scenario_idx]

        return (
            torch.tensor(x,      dtype=torch.float32),   # (W, N, 2) — model input
            torch.tensor(detect, dtype=torch.long),       # detection label
            torch.tensor(pipe_t, dtype=torch.long),       # pipe label
            torch.tensor(pos_t,  dtype=torch.float32),    # position label
            torch.tensor(size_t, dtype=torch.long),       # size label
        )


# MODEL

class TemporalConvLayer(nn.Module):
    """
    A dilated temporal convolution applied along the time axis at
    each sensor node independently.
    
    Batch normalisation helps stabilise training and ReLU introduces
    the non-linearity needed for the model to learn complex patterns.
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        # Symmetric padding keeps the temporal length the same after convolution
        pad = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size),  # (1, K) applies convolution along time only
            padding=(0, pad),
            dilation=(1, dilation)         # dilation along the time axis only
        )
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x arrives as (B, T, N, C) — need to permute to (B, C, N, T) for Conv2d
        x = x.permute(0, 3, 2, 1)   # (B, C, N, T)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x.permute(0, 3, 2, 1)   # back to (B, T, N, C)
        return x

class GraphConvLayer(nn.Module):
    """
    Graph convolution that propagates information between neighbouring
    sensor nodes using the pre-built adjacency matrix.
    
    Layer normalisation is used instead of batch normalisation here because
    it works better across the node and feature dimensions in graph settings.
    """
    def __init__(self, in_ch, out_ch, adj_matrix):
        super().__init__()
        # Store A as a buffer — it moves to GPU with the model but is not a learned parameter
        self.register_buffer("A", torch.tensor(adj_matrix, dtype=torch.float32))
        self.lin = nn.Linear(in_ch, out_ch)
        self.ln  = nn.LayerNorm(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: (B, T, N, C)
        # Einstein summation: for each (batch, time, node), sum over neighbour nodes
        x = torch.einsum("ij,btjc->btic", self.A, x)
        x = self.lin(x)   # learnable linear transform after aggregation
        x = self.ln(x)
        x = self.act(x)
        return x

class STBlock(nn.Module):
    """
    One spatio-temporal block: temporal convolution followed by graph
    convolution, with a residual connection that adds the input back
    to the output.
    """
    def __init__(self, in_ch, out_ch, adj_matrix, kernel_size=5, dilation=1, dropout=0.25):
        super().__init__()
        self.temp    = TemporalConvLayer(in_ch, out_ch, kernel_size, dilation)
        self.graph   = GraphConvLayer(out_ch, out_ch, adj_matrix)
        self.dropout = nn.Dropout(dropout)
        self.out_act = nn.ReLU()

        # Project the residual to the correct channel size if needed
        if in_ch != out_ch:
            self.res_proj = nn.Linear(in_ch, out_ch)
        else:
            self.res_proj = nn.Identity()   # pass through unchanged

    def forward(self, x):
        residual = self.res_proj(x)  # save the input before transformation
        y = self.temp(x)             # temporal convolution
        y = self.graph(y)            # graph convolution
        y = self.dropout(y)          # randomly drop some activations
        y = y + residual             # add skip connection
        y = self.out_act(y)
        return y

class TemporalAttentionPool(nn.Module):
    """
    Learnable attention pooling that collapses the time dimension by
    computing a weighted average across the 12 timesteps.

    Unlike average pooling which treats all timesteps equally,
    this layer learns to focus on the timesteps that carry the most
    information. For leak detection, the timesteps after the leak onset
    carry much stronger signals, so attention pooling should naturally
    learn to up-weight those.

    The output shape (B, N*C) flattens the node and channel dimensions
    into a single vector for the task heads.
    """
    def __init__(self, hidden_dim, num_nodes):
        super().__init__()
        # One attention score per timestep, computed from the full feature vector
        self.attn = nn.Linear(hidden_dim * num_nodes, 1)

    def forward(self, x):
        B, T, N, C = x.shape
        x_flat  = x.reshape(B, T, N * C)           # flatten nodes and channels
        scores  = self.attn(x_flat)                 # one score per timestep
        weights = torch.softmax(scores, dim=1)      # normalise so weights sum to 1
        z       = (x_flat * weights).sum(dim=1)     # weighted sum over time
        return z


class SingleLeakSTGCN(nn.Module):
    """
    The full ST-GCN model for single-leak detection and localisation.

    The backbone is three ST blocks with dilations 1, 2, 4 which together
    give the model a receptive field covering the full 12-timestep window.
    After the temporal attention pool, the 320-dimensional feature vector
    is passed to four separate task heads in parallel.

    Using four heads that share the same backbone is more efficient than
    training four separate models and also lets the tasks help each other
    — learning to localise a leak implicitly helps with detection too.
    """
    def __init__(self, adj_matrix):
        super().__init__()

        # Three ST blocks with increasing dilation to expand receptive field
        self.block1 = STBlock(
            in_ch=NODE_FEATS, out_ch=HIDDEN_1,
            adj_matrix=adj_matrix, kernel_size=KERNEL_SIZE, dilation=1, dropout=DROPOUT
        )
        self.block2 = STBlock(
            in_ch=HIDDEN_1, out_ch=HIDDEN_2,
            adj_matrix=adj_matrix, kernel_size=KERNEL_SIZE, dilation=2, dropout=DROPOUT
        )
        self.block3 = STBlock(
            in_ch=HIDDEN_2, out_ch=HIDDEN_2,
            adj_matrix=adj_matrix, kernel_size=KERNEL_SIZE, dilation=4, dropout=DROPOUT
        )

        # After pooling, the feature vector is 10 sensors × 32 channels = 320
        head_in     = NUM_NODES * HIDDEN_2   # 320
        head_hidden = 64

        # Detection head: classifies the scenario as leak or no-leak
        self.detect_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, 2),   # 2 classes: no-leak, leak
        )

        # Pipe head: identifies which of the 5 pipes contains the leak
        self.pipe_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, PIPE_CLASSES),   # 6 classes: pipe 1-5 + NONE
        )

        # Size head: classifies leak severity as small, medium, or large
        self.size_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, SIZE_CLASSES),   # 4 classes: S, M, L, NONE
        )

        # Position head: regresses the normalised position along the pipe [0, 1]
        # Sigmoid is used to keep the output within the valid range
        self.pos_head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(head_hidden, 1),
            nn.Sigmoid(),
        )

        self.temporal_pool = TemporalAttentionPool(HIDDEN_2, NUM_NODES)

    def forward(self, x):
        # x: (B, T, N, 2) — batch of windows with 2 features per sensor
        x = self.block1(x)   # (B, T, N, HIDDEN_1)
        x = self.block2(x)   # (B, T, N, HIDDEN_2)
        x = self.block3(x)   # (B, T, N, HIDDEN_2)

        # Pool across the time dimension — result is (B, N * HIDDEN_2)
        z = self.temporal_pool(x)

        # Run all four heads on the same pooled feature vector
        detect_logits = self.detect_head(z)           # (B, 2)
        pipe_logits   = self.pipe_head(z)             # (B, PIPE_CLASSES)
        size_logits   = self.size_head(z)             # (B, SIZE_CLASSES)
        pos_pred      = self.pos_head(z).squeeze(1)   # (B,) — remove the extra dimension

        return detect_logits, pipe_logits, size_logits, pos_pred


# EVAL HELPERS

@torch.no_grad()
def scenario_level_detection_accuracy(model, dataset, batch_size=256):
    """
    Evaluates detection accuracy at the scenario level using majority
    voting across all windows of each scenario.

    Since each scenario only produces one window (WINDOW=12, STRIDE=1),
    the majority vote here is effectively just the single window prediction.
    The function is written to handle multi-window scenarios correctly
    in case the window/stride settings are changed in the future.

    The @torch.no_grad() decorator disables gradient computation during
    evaluation, which saves memory and speeds things up.
    """
    model.eval()

    true_labels = []
    pred_labels = []

    for i in range(len(dataset.features)):
        x_full = dataset.features[i]
        detect, _, _, _ = dataset.targets[i]   # only need the detection label here

        T         = x_full.shape[0]
        n_windows = max(0, (T - dataset.window) // dataset.stride + 1)
        if n_windows == 0:
            continue

        # Extract all windows for this scenario
        windows = []
        for w in range(n_windows):
            s = w * dataset.stride
            e = s + dataset.window
            windows.append(x_full[s:e])
        windows = np.stack(windows, axis=0).astype(np.float32)

        # Run inference in batches to avoid memory issues
        batch_preds = []
        for start in range(0, windows.shape[0], batch_size):
            xb = torch.tensor(windows[start:start+batch_size], dtype=torch.float32, device=DEVICE)
            detect_logits, _, _, _ = model(xb)
            batch_preds.extend(detect_logits.argmax(dim=1).cpu().numpy().tolist())

        # Take the most common prediction across all windows as the final answer
        vals, counts = np.unique(np.array(batch_preds), return_counts=True)
        final_pred   = int(vals[np.argmax(counts)])

        true_labels.append(detect)
        pred_labels.append(final_pred)

    return accuracy_score(true_labels, pred_labels)


# MAIN

def main():
    # Report whether a GPU was found — training on CPU is much slower
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: NO GPU")

    # Load scenario folder lists from the pre-built manifest CSVs
    train_folders = folders_from_manifest(MANIFEST_TRAIN)
    val_folders   = folders_from_manifest(MANIFEST_VAL)
    test_folders  = folders_from_manifest(MANIFEST_TEST)

    if len(train_folders) == 0:
        raise RuntimeError(f"No valid scenario folders found via {MANIFEST_TRAIN}")

    print(f"Train: {len(train_folders)}  Val: {len(val_folders)}  Test: {len(test_folders)}")

    # Compute baseline and normalisation statistics using training data only
    baseline_template = compute_baseline_template(train_folders)
    mu, sigma = compute_mu_sigma(train_folders, baseline_template)

    # Build all three dataset splits
    print("Building train dataset...")
    train_ds = ScenarioWindowDataset(
        train_folders, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )
    print("Building val dataset...")
    val_ds = ScenarioWindowDataset(
        val_folders, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )
    print("Building test dataset...")
    test_ds = ScenarioWindowDataset(
        test_folders, baseline_template, mu, sigma,
        window=WINDOW, stride=STRIDE
    )

    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows:   {len(val_ds)}")
    print(f"Test windows:  {len(test_ds)}")

    # num_workers=0 avoids multiprocessing issues on Windows
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Build the model and move it to GPU if available
    adj   = build_sensor_adjacency()
    model = SingleLeakSTGCN(adj).to(DEVICE)

    # Compute class weights for the detection loss.
    # The weights are computed at the window level (not scenario level) because
    # the training loop sees windows, not scenarios. This gives the correct
    # picture of how imbalanced the data actually is during training.
    window_class_counts = Counter()
    for i, tgt in enumerate(train_ds.targets):
        detect    = tgt[0]
        n_windows = train_ds.window_counts[i]
        window_class_counts[detect] += n_windows

    print(f"Window-level class counts (train): {dict(sorted(window_class_counts.items()))}")

    # Inverse frequency weighting: rarer classes get higher weights
    class_weights = []
    for c in range(2):
        class_weights.append(1.0 / max(window_class_counts.get(c, 1), 1))
    class_weights = np.array(class_weights, dtype=np.float32)
    # Normalise so the weights sum to 2 (one per class) — keeps the loss scale reasonable
    class_weights = class_weights / class_weights.sum() * 2.0
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    # GradScaler handles the float16/float32 scaling needed for mixed precision training
    scaler    = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)

    # Loss functions for each task head.
    # Pipe and size use reduction="none" so the detection mask can be applied
    # before averaging, excluding no-leak scenarios from localisation gradients.
    # It doesn't make sense to penalise the pipe head for no-leak scenarios
    # because there is no correct pipe to predict in that case.
    loss_detect = nn.CrossEntropyLoss(weight=class_weights)
    loss_pipe   = nn.CrossEntropyLoss(reduction="none")
    loss_size   = nn.CrossEntropyLoss(reduction="none")
    loss_pos    = nn.SmoothL1Loss(reduction="none")  # smoother than MSE near zero

    # Loss weights control the relative importance of each task.
    # Pipe localisation is weighted highest because it is the most useful output.
    # Position regression is down-weighted because SmoothL1 on [0,1] produces
    # smaller gradient magnitudes than the classification losses.
    LAMBDA_DETECT = 1.0
    LAMBDA_PIPE   = 2.0
    LAMBDA_SIZE   = 1.0
    LAMBDA_POS    = 0.5

    # Training loop
    for ep in range(EPOCHS):
        model.train()
        running = 0.0

        for x, detect, pipe_t, pos_t, size_t in train_loader:
            # Move all tensors to the target device (GPU or CPU)
            x      = x.to(DEVICE, non_blocking=True)
            detect = detect.to(DEVICE, non_blocking=True)
            pipe_t = pipe_t.to(DEVICE, non_blocking=True)
            pos_t  = pos_t.to(DEVICE,  non_blocking=True)
            size_t = size_t.to(DEVICE, non_blocking=True)

            # Clear gradients before each batch
            optimizer.zero_grad(set_to_none=True)

            # autocast enables mixed precision on GPU — uses float16 where safe
            with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                detect_logits, pipe_logits, size_logits, pos_pred = model(x)

                # Detection loss — uses class weights to handle imbalance
                Ld = loss_detect(detect_logits, detect)

                # Localisation losses — only computed for leak scenarios.
                # leak_mask is 1.0 for leak windows and 0.0 for no-leak windows,
                # so multiplying by it zeros out the loss for no-leak samples.
                leak_mask = detect.float()
                denom     = leak_mask.sum().clamp(min=1.0)  # avoid division by zero

                Lp = (loss_pipe(pipe_logits, pipe_t) * leak_mask).sum() / denom
                Ls = (loss_size(size_logits, size_t) * leak_mask).sum() / denom
                Lr = (loss_pos(pos_pred, pos_t)      * leak_mask).sum() / denom

                # Combine all losses into one weighted sum
                loss = LAMBDA_DETECT * Ld + LAMBDA_PIPE * Lp + LAMBDA_SIZE * Ls + LAMBDA_POS * Lr

            # Scaled backward pass and optimiser step for mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item())

        print(f"Epoch {ep+1}/{EPOCHS}, loss={running/len(train_loader):.4f}")

    # Save the trained model as a bundle that includes everything needed
    # for inference — weights, adjacency matrix, normalisation stats, and hyperparameters.
    # This self-contained format means the model can be loaded without needing
    # to know the original training configuration separately.
    bundle = {
        "model_type":        "stgcn_single_leak_v5_10ch",
        "model_state_dict":  model.state_dict(),   # learned weights
        "adjacency":         adj,                  # sensor graph structure
        "mu":                mu,                   # normalisation mean
        "sigma":             sigma,                # normalisation std
        "baseline_template": baseline_template,    # no-leak reference signal
        "sensor_names":      SENSOR_NAMES,         # sensor ordering
        "window":            WINDOW,
        "stride":            STRIDE,
        "node_feats":        NODE_FEATS,
        "hidden_1":          HIDDEN_1,
        "hidden_2":          HIDDEN_2,
        "kernel_size":       KERNEL_SIZE,
        "dilations":         [1, 2, 4],
        "num_blocks":        3,
        "head_in":           NUM_NODES * HIDDEN_2,
        "dropout":           DROPOUT,
        "dataset_root":      DATASET_ROOT,
        "pipe_classes":      PIPE_CLASSES,
        "size_classes":      SIZE_CLASSES,
        "seed":              SEED,
        "epochs":            EPOCHS,
        "batch_size":        BATCH_SIZE,
        "lr":                LR,
        "weight_decay":      WEIGHT_DECAY,
    }

    torch.save(bundle, SAVE_PATH)
    print(f"[OK] Saved {SAVE_PATH}")

    # Quick scenario-level detection accuracy check on the test split
    test_det_acc = scenario_level_detection_accuracy(model, test_ds, batch_size=256)
    print("\n=== SCENARIO-LEVEL DETECTION ===")
    print(f"Accuracy: {test_det_acc:.4f}")

    # Window-level pipe and position evaluation — only on leak scenarios
    model.eval()
    pipe_true_all, pipe_pred_all = [], []
    pos_true_all,  pos_pred_all  = [], []
    detect_true_all = []

    with torch.no_grad():
        for x, detect, pipe_t, pos_t, size_t in test_loader:
            x = x.to(DEVICE)
            _, pipe_logits, _, pos_pred = model(x)

            pipe_true_all.append(pipe_t.numpy())
            pipe_pred_all.append(pipe_logits.argmax(dim=1).cpu().numpy())
            pos_true_all.append(pos_t.numpy())
            pos_pred_all.append(pos_pred.cpu().numpy())
            detect_true_all.append(detect.numpy())

    # Concatenate all batch results into single arrays
    pipe_true_all   = np.concatenate(pipe_true_all)
    pipe_pred_all   = np.concatenate(pipe_pred_all)
    pos_true_all    = np.concatenate(pos_true_all)
    pos_pred_all    = np.concatenate(pos_pred_all)
    detect_true_all = np.concatenate(detect_true_all)

    # Filter to leak scenarios only — localisation metrics on no-leak
    # windows are meaningless because there is no correct pipe to predict
    leak_mask = detect_true_all > 0.5

    # Pipe F1 Macro across the five pipes — excludes the NONE class
    pipe_true_leak = pipe_true_all[leak_mask]
    pipe_pred_leak = pipe_pred_all[leak_mask]
    f1_pipe = f1_score(pipe_true_leak, pipe_pred_leak, average="macro", labels=list(range(NUM_PIPES)))
    print(f"\n=== PIPE F1 MACRO (leak scenarios only) ===")
    print(f"F1 Macro: {f1_pipe:.4f}")

    # Position regression metrics — MAE and RMSE on the normalised [0,1] scale
    pos_true_leak = pos_true_all[leak_mask]
    pos_pred_leak = pos_pred_all[leak_mask]
    if len(pos_true_leak) > 0:
        mae  = float(np.mean(np.abs(pos_true_leak - pos_pred_leak)))
        rmse = float(np.sqrt(np.mean((pos_true_leak - pos_pred_leak) ** 2)))
        print(f"\n=== POSITION (leak scenarios only) ===")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")


if __name__ == "__main__":
    main()
