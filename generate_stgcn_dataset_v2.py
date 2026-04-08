"""
Dataset generation script for the ST-GCN single-leak detection and
localisation model.

This script generates the training scenarios used to train the ST-GCN by 
running EPANET hydraulic simulations through WNTR for every
combination of pipe, leak position, leak size, and time-of-week repetition.

Each scenario produces a data.csv file containing 24 rows of pressure and
flow readings (one per 15-minute timestep over a 6-hour window), along with
a labels.json file containing the ground truth leak configuration.

The script generates an equal number of leak and no-leak scenarios.

"""

import copy
import json
import logging
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import wntr
from sklearn.model_selection import train_test_split


# INP FILE BLOCK HELPERS

# EPANET .inp files are structured in named blocks like [TIMES], [PIPES] etc.
# These two functions allow specific blocks to be read or replaced as strings,
# which is useful when modifying network configurations programmatically.

def extract_block(inp_text: str, header: str) -> str:
    """
    Extracts a named block from an EPANET INP file string.

    For example, extract_block(text, "[TIMES]") returns everything from
    the [TIMES] header up to the start of the next block. This is used
    to read specific sections of the INP file for inspection or copying.
    """
    lines = inp_text.splitlines()
    header_upper = header.upper()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper() == header_upper:
            start = i
            break
    if start is None:
        return ""   # block not found — return empty string
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("[") and lines[j].strip().endswith("]"):
            end = j
            break
    return "\n".join(lines[start:end]) + "\n"


def replace_block(inp_text: str, header: str, new_block: str) -> str:
    """
    Replaces a named block in an EPANET INP file string with new content.

    If the block does not exist in the file, the new block is appended
    at the end. This is used to modify time settings or pattern definitions
    before writing a modified INP file for a specific scenario.
    """
    lines = inp_text.splitlines()
    header_upper = header.upper()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper() == header_upper:
            start = i
            break
    if start is None:
        # Block not found — append it at the end
        if not inp_text.endswith("\n"):
            inp_text += "\n"
        return inp_text + "\n" + new_block
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("[") and lines[j].strip().endswith("]"):
            end = j
            break
    before = "\n".join(lines[:start]) + "\n"
    after  = "\n".join(lines[end:]) + ("\n" if end < len(lines) else "")
    return before + new_block + after

# NETWORK HELPERS

def get_connected_links(wn, node_name: str):
    """Returns a list of all link names connected to a given node in the network."""
    return list(wn.get_links_for_node(node_name))


def pick_two_links_at_leaknode(wn, leak_node: str, prefer_tokens=("1a", "1b")):
    """
    Identifies the two pipe segments connected to a leak node.

    Each leak node (L1 to L5) sits at the midpoint of a logical pipe and
    connects exactly two physical pipe segments (e.g. pipe 1a and 1b for
    Pipe 1). The position of the leak is set by adjusting the lengths of
    these two segments.

    The function tries to find the two links by exact name match first,
    then by substring match, then falls back to alphabetical ordering.
    This makes it robust to minor variations in how links are named in
    different .inp files.
    """
    links = get_connected_links(wn, leak_node)
    if len(links) != 2:
        raise RuntimeError(
            f"Leak node '{leak_node}' must connect to exactly 2 links. Found {len(links)}: {links}"
        )
    s = set(links)
    # Try exact match first
    if prefer_tokens[0] in s and prefer_tokens[1] in s:
        return prefer_tokens[0], prefer_tokens[1]

    def find_contains(token):
        for ln in links:
            if token in ln:
                return ln
        return None

    # Try substring match
    a = find_contains(prefer_tokens[0])
    b = find_contains(prefer_tokens[1])
    if a and b and a != b:
        return a, b

    # Final fallback: alphabetical order
    links_sorted = sorted(links)
    return links_sorted[0], links_sorted[1]


def clear_emitters(wn, leak_nodes):
    """
    Resets the emitter coefficient to zero on all leak nodes.

    This is called at the start of each scenario to ensure no residual
    leak from a previous configuration is carried over. Without this,
    loading a network that was previously used for a leak simulation
    might still have an emitter set on the wrong node.
    """
    for ln in leak_nodes:
        node = wn.get_node(ln)
        if hasattr(node, "emitter_coefficient"):
            node.emitter_coefficient = 0.0


def set_link_lengths(wn, link_a: str, len_a: float, link_b: str, len_b: float):
    """
    Sets the lengths of the two pipe segments on either side of the leak node.

    The leak position is defined as a fraction of the total pipe length —
    for example, position=0.3 means the leak is 30% along the pipe from
    the upstream end. By setting len_a = position * total_length and
    len_b = (1 - position) * total_length, the hydraulic distance from
    the leak to each end of the pipe is correctly encoded in the network.
    """
    wn.get_link(link_a).length = float(len_a)
    wn.get_link(link_b).length = float(len_b)


def convert_emitter_to_internal(wn, emitter_in_inp_units: float) -> float:
    """
    Converts an emitter coefficient from the INP file's flow units (LPS)
    to WNTR's internal units (m³/s).

    EPANET .inp files typically use litres per second for flow rates,
    but WNTR works internally in SI units (m³/s). Without this conversion,
    the leak flow rate would be 1000x too large, which would produce
    unrealistically large pressure drops and invalid simulations.
    """
    try:
        units = str(getattr(wn.options.hydraulic, "inpfile_units", "")).upper()
    except Exception:
        units = ""
    if units == "LPS":
        return float(emitter_in_inp_units) / 1000.0
    return float(emitter_in_inp_units)


def set_emitter(wn, node_name: str, emitter_in_inp_units: float):
    """
    Sets the emitter coefficient on a junction node, converting units as needed.

    An emitter models flow loss from a pressurised orifice — in this project
    it represents the leak discharge. The emitter flow is:
        q_leak = Ce * P^0.5
    where Ce is the emitter coefficient and P is the local pressure.
    """
    node = wn.get_node(node_name)
    if not hasattr(node, "emitter_coefficient"):
        raise RuntimeError(f"Node '{node_name}' has no emitter_coefficient (not a junction?).")
    node.emitter_coefficient = convert_emitter_to_internal(wn, emitter_in_inp_units)

# DATASET CONSTANTS

# The five logical pipes and the positions tested along each pipe.
# Each position is a normalised fraction: 0.1 = 10% from the upstream end.
PIPES     = [1, 2, 3, 4, 5]
POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Three leak sizes defined by emitter coefficient in LPS units.
# Small (S) = 0.01, Medium (M) = 0.03, Large (L) = 0.06
SIZES = [("S", 0.01), ("M", 0.03), ("L", 0.06)]

# 21 repetitions at 8-hour spacing covers exactly one full week (168 hours).
# This gives denser time-of-day coverage than v1's 14 reps at 12-hour spacing,
# so the model is exposed to more varied demand conditions during training.
REPETITIONS   = 21
START_SPACING = 8 * 3600   # seconds between consecutive repetition start times

# Physical pipe lengths used to set link segment lengths for each leak position
TOTAL_LENGTHS    = {1: 100.0, 2: 500.0, 3: 200.0, 4: 200.0, 5: 200.0}

# Mapping from logical pipe number to the corresponding leak junction node name
PIPE_TO_LEAKNODE = {1: "L1",  2: "L2",  3: "L3",  4: "L4",  5: "L5"}
LEAK_NODES       = ["L1", "L2", "L3", "L4", "L5"]

# EPANET node and link IDs where sensor readings are extracted.
# SENSOR_NODES give pressure readings; SENSOR_LINKS give flow readings.
SENSOR_NODES = ["2", "3", "4", "5", "6"]
SENSOR_LINKS = ["1a", "2a", "3a", "4a", "5a"]

# Simulation duration: 6 hours at 15-minute intervals = 24 timesteps.
# Extended from 3 hours (v1) to give the model more temporal context
# and a longer pre-leak phase to establish the baseline signal.
SIM_DURATION    = 6 * 3600   # 6 hours in seconds
HYD_TIMESTEP    = 15 * 60    # 15-minute report interval
PAT_TIMESTEP    = 60         # 1-minute pattern step
LEAK_ONSET_STEP = 4          # leak activates at step index 4 (t = 60 min)

N_TIMESTEPS = 24                                     # 6h / 15min = 24 timesteps
T_AXIS_MIN  = [i * 15 for i in range(N_TIMESTEPS)]  # time axis: 0, 15, ..., 345 min

# Named constants for the leak demand pattern lengths.
# The pattern has 60 zero-steps (pre-leak phase) followed by 300 one-steps
# (active leak phase). At PAT_TIMESTEP=60s, this means the leak activates
# at t = 60 minutes and runs for the remaining 5 hours.
LEAK_PAT_OFF_STEPS = 60    # pre-leak: 60 min of normal conditions
LEAK_PAT_ON_STEPS  = 300   # active leak: 300 min (5 hours)

DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Output directory structure — using _v2 suffix to avoid overwriting v1 data
OUT_ROOT      = Path("stgcn_dataset_v2")
OUT_SCENARIOS = OUT_ROOT / "scenarios"       # one subfolder per scenario
OUT_INP_DIR   = OUT_ROOT / "scenarios_inp"  # .inp snapshots for reference
OUT_MANIFESTS = OUT_ROOT / "manifests"      # train/val/test split CSVs
OUT_STATIC    = OUT_ROOT / "static_graph"   # graph structure files
LOG_FILE      = OUT_ROOT / "generation_log.txt"


# TIME HELPERS

def rep_start_sec(rep: int) -> int:
    """
    Returns the absolute start time for a given repetition index in seconds
    from midnight on Sunday. Used to rotate demand patterns correctly.
    """
    return rep * START_SPACING


def rep_to_day_time(rep: int):
    """
    Converts a repetition index to a human-readable day name and time string.
    For example, rep=12 at 8-hour spacing is 96 hours = Sunday + 4 days = Thursday.
    Stored in labels.json so scenarios can be identified by their time of week.
    """
    total_hours = rep * (START_SPACING // 3600)
    day_idx     = (total_hours // 24) % 7
    hour_of_day = total_hours % 24
    return DAY_NAMES[day_idx], f"{hour_of_day:02d}:00"


def select_base(rep: int, base_inp: Path, base2_inp: Path):
    """
    Alternates between two base INP files across repetitions.
    Even repetitions use base.inp and odd repetitions use base2.inp.

    The two base files have slightly different demand configurations,
    which increases the variety of hydraulic conditions in the dataset
    without requiring completely different networks.
    """
    if rep % 2 == 0:
        return base_inp, "base.inp"
    return base2_inp, "base2.inp"


# WNTR HELPERS

def load_and_configure(base_path: Path, duration_sec: int,
                       rep: int, extra_offset_sec: int = 0):
    """
    Loads an EPANET network file and configures all time settings and
    demand pattern rotations for a specific repetition.

    The demand patterns in the .inp file cover a full week at 1-minute
    resolution (10,080 steps). To simulate conditions at a specific point
    in the week (e.g. rep=6 at 8h spacing = Wednesday 0:00), all patterns
    are rotated left by the correct number of steps.

    start_clocktime is kept at zero rather than using the actual time
    of day because EPANET on Windows throws Error 200 when start_clocktime
    exceeds 24 hours, which happens for repetitions beyond the first day.
    Rotating the patterns achieves the same result without this limitation.
    """
    wn = wntr.network.WaterNetworkModel(str(base_path))

    # Set simulation time parameters to match the dataset configuration
    wn.options.time.duration           = duration_sec
    wn.options.time.hydraulic_timestep = HYD_TIMESTEP
    wn.options.time.report_timestep    = HYD_TIMESTEP
    wn.options.time.pattern_timestep   = PAT_TIMESTEP
    wn.options.time.start_clocktime    = 0   # always 0 — position handled by rotation

    # Compute how many pattern steps to rotate by for this repetition
    offset_steps = int((rep * START_SPACING + extra_offset_sec) // PAT_TIMESTEP)
    for pat_name in wn.pattern_name_list:
        mults = np.array(wn.get_pattern(pat_name).multipliers)
        if len(mults) > 0:
            # np.roll with negative shift moves the array left — equivalent
            # to fast-forwarding to the correct point in the weekly cycle
            rotated = np.roll(mults, -int(offset_steps % len(mults)))
            wn.get_pattern(pat_name).multipliers = list(rotated)

    return wn


def extract_signals(results, n_rows: int):
    """
    Extracts pressure and flow readings from a WNTR simulation result object.

    Returns (p, q) where p is pressure at the five sensor nodes (metres)
    and q is flow rate at the five sensor links (m³/s), both shaped (n_rows, 5).

    Raises an error if any sensor columns are missing or if any pressure
    values are negative — negative pressures indicate a hydraulically
    invalid simulation (e.g. a demand that exceeds network capacity).
    """
    pressures = results.node["pressure"]
    flows     = results.link["flowrate"]

    # Validate that all expected sensors appear in the results
    missing_nodes = [n for n in SENSOR_NODES if n not in pressures.columns]
    missing_links = [lk for lk in SENSOR_LINKS if lk not in flows.columns]
    if missing_nodes:
        raise RuntimeError(f"Missing pressure nodes in results: {missing_nodes}")
    if missing_links:
        raise RuntimeError(f"Missing flow links in results: {missing_links}")
    if pressures.shape[0] < n_rows:
        raise RuntimeError(
            f"Expected ≥{n_rows} timesteps but got {pressures.shape[0]}. "
            "Check simulation duration and timestep settings."
        )

    p = pressures[SENSOR_NODES].values[:n_rows]
    q = flows[SENSOR_LINKS].values[:n_rows]

    # Negative pressures mean the simulation result is physically invalid
    if np.any(p < 0):
        raise RuntimeError(
            f"Negative pressures detected (min={p.min():.4f} m). Scenario invalid."
        )
    return p, q


def build_df(p: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    """
    Assembles the sensor readings into a DataFrame matching the data.csv format.

    Columns: t, P2, P3, P4, P5, P6, Q1a, Q2a, Q3a, Q4a, Q5a
    where t is time in minutes and the sensor columns contain pressure
    (metres) and flow rate (m³/s) respectively.
    """
    df = pd.DataFrame({"t": T_AXIS_MIN})
    for i, node in enumerate(SENSOR_NODES):
        df[f"P{node}"] = p[:, i]
    for i, link in enumerate(SENSOR_LINKS):
        df[f"Q{link}"] = q[:, i]
    return df

# SCENARIO RUNNERS

def run_leak_scenario(base_path: Path, rep: int, pipe_id: int,
                      position: float, emitter_coeff: float):
    """
    Runs a single 6-hour hydraulic simulation with a leak that activates
    at t = 60 minutes (step index 4).

    The first 60 minutes (4 timesteps) represent normal no-leak conditions.
    The leak then activates and remains active for the final 300 minutes
    (20 timesteps). This gives the model a pre-leak reference period
    followed by the evolving leak response — which is important because
    the deviation feature is measured against the no-leak baseline.

    The leak is modelled using a step demand pattern added to the leak node.
    The demand magnitude is estimated from the pre-leak node pressure using
    the orifice equation: q = Ce * P^0.5.

    A snapshot of the network (wn_inp) is saved before the demand pattern
    is added, because WNTR control objects cannot be represented in the
    standard EPANET .inp format. The snapshot is used to write the .inp
    file that documents the scenario's physical configuration.

    Returns:
        df     — 24-row sensor signals DataFrame
        wn_inp — network snapshot for writing the .inp reference file
    """
    leak_node_name = PIPE_TO_LEAKNODE[pipe_id]
    total_len      = TOTAL_LENGTHS[pipe_id]

    # Set segment lengths to place the leak at the correct position
    len_a_seg     = position * total_len        # upstream segment
    len_b_seg     = total_len - len_a_seg       # downstream segment
    prefer_tokens = (f"{pipe_id}a", f"{pipe_id}b")

    # Load and configure the network for this repetition
    wn = load_and_configure(base_path, SIM_DURATION, rep, extra_offset_sec=0)
    clear_emitters(wn, LEAK_NODES)

    # Set the pipe segment lengths to encode the leak position
    link_a, link_b = pick_two_links_at_leaknode(wn, leak_node_name, prefer_tokens)
    set_link_lengths(wn, link_a, len_a_seg, link_b, len_b_seg)

    # Set the emitter coefficient on the leak node (full leak value)
    set_emitter(wn, leak_node_name, emitter_coeff)

    # Take a snapshot before adding the demand pattern — this represents
    # the physical network state and is used for the .inp reference file
    wn_inp = copy.deepcopy(wn)

    # Reset emitter to zero so the simulation starts in the no-leak state
    wn.get_node(leak_node_name).emitter_coefficient = 0.0

    # Estimate the leak flow rate from the pre-leak node pressure.
    # A quick zero-duration run gives the steady-state pressure before
    # the leak starts, which is used in the orifice equation q = Ce * P^0.5.
    Ce_internal = convert_emitter_to_internal(wn, emitter_coeff)
    wn_pre = copy.deepcopy(wn)
    wn_pre.options.time.duration = 0
    res_pre = wntr.sim.EpanetSimulator(wn_pre).run_sim()
    P_pre   = float(res_pre.node["pressure"][leak_node_name].iloc[0])
    q_leak  = Ce_internal * (max(P_pre, 0.0) ** 0.5)   # m³/s

    # Build the step demand pattern: 60 minutes of zeros (no leak) followed
    # by 300 minutes of ones (leak active). EPANET multiplies the base demand
    # (q_leak) by each pattern value at each timestep.
    pat_name = f"_lk_{pipe_id}_{rep}"
    wn.add_pattern(pat_name, [0.0] * LEAK_PAT_OFF_STEPS + [1.0] * LEAK_PAT_ON_STEPS)
    wn.get_node(leak_node_name).demand_timeseries_list.append(
        (q_leak, wn.get_pattern(pat_name), "leak_demand")
    )

    # Run the full 6-hour simulation and extract the sensor readings
    p, q = extract_signals(wntr.sim.EpanetSimulator(wn).run_sim(), N_TIMESTEPS)
    return build_df(p, q), wn_inp


def run_noleak_sim(base_path: Path, rep: int):
    """
    Runs a 6-hour no-leak simulation for a given repetition.

    No-leak simulations are used to:
    1. Generate the no-leak scenario folders (labels.json + data.csv)
    2. Compute the baseline template used to derive the deviation feature

    Returns:
        df  — 24-row sensor signals DataFrame
        wn  — the configured WNTR network object (for writing the .inp file)
    """
    wn = load_and_configure(base_path, SIM_DURATION, rep, extra_offset_sec=0)
    clear_emitters(wn, LEAK_NODES)
    p, q = extract_signals(wntr.sim.EpanetSimulator(wn).run_sim(), N_TIMESTEPS)
    return build_df(p, q), wn


# STATIC GRAPH

def generate_static_graph(out_dir: Path):
    """
    Saves the static graph structure of the pipe network to disk.

    This includes the adjacency matrix, pipe endpoint node indices, and
    pipe lengths — all of which describe the physical connectivity of the
    network. These files are not used directly during training (the
    adjacency matrix is built from SENSOR_NAMES in the training script)
    but are saved here for reference and potential future use.

    The graph has 6 nodes (1 reservoir + 5 junctions) and 5 pipes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes    = [1, 2, 3, 4, 5, 6]
    node_idx = {nid: i for i, nid in enumerate(nodes)}

    # Define which nodes each pipe connects — in order of PIPES list
    pipe_endpoints_ids = [(1, 2), (2, 3), (3, 4), (4, 5), (4, 6)]

    # Build the adjacency matrix for the 6-node graph
    n   = len(nodes)
    adj = np.zeros((n, n), dtype=int)
    for src, dst in pipe_endpoints_ids:
        i, j = node_idx[src], node_idx[dst]
        adj[i, j] = 1
        adj[j, i] = 1   # undirected graph

    pipe_endpoint_map = np.array(
        [[node_idx[s], node_idx[d]] for s, d in pipe_endpoints_ids]
    )
    pipe_lengths = np.array([TOTAL_LENGTHS[p] for p in PIPES], dtype=float)

    # Save all graph files as NumPy arrays and plain text lists
    np.save(out_dir / "adjacency_matrix.npy",  adj)
    np.save(out_dir / "pipe_endpoint_map.npy", pipe_endpoint_map)
    np.save(out_dir / "pipe_lengths.npy",      pipe_lengths)

    (out_dir / "node_list.txt").write_text(
        "\n".join(str(n) for n in nodes) + "\n", encoding="utf-8"
    )
    (out_dir / "pipe_list.txt").write_text(
        "\n".join(str(p) for p in PIPES) + "\n", encoding="utf-8"
    )
    print("[OK] Static graph files written.", flush=True)


# LABELS AND MANIFEST HELPERS

def make_labels(scenario_id: int, source_inp: str, rep: int, is_leak: bool,
                pipe_id=None, position=None, size_label=None, emitter_coeff=None) -> dict:
    """
    Builds the labels dictionary that is saved as labels.json for each scenario.

    For leak scenarios, all four classification/regression targets are included:
        label_detection  — 1 (leak present)
        label_pipe       — which pipe (1-5)
        label_position   — normalised position along the pipe (0.0-1.0)
        label_size       — size class string ("S", "M", or "L")

    For no-leak scenarios, all label fields are set to their null values
    (0, -1, -1, "none") so the training script can identify them easily.

    Metadata fields (scenario_id, source_inp, repetition, start_day, start_time)
    are always included for traceability and reproducibility.
    """
    day, time_str = rep_to_day_time(rep)
    base = {
        "scenario_id": scenario_id,
        "source_inp":  source_inp,
        "repetition":  rep,
        "start_day":   day,
        "start_time":  time_str,
    }
    if is_leak:
        base.update({
            "label_detection": 1,
            "label_pipe":      int(pipe_id),
            "label_position":  float(position),
            "label_size":      str(size_label),
            "emitter_coeff":   float(emitter_coeff),
            "leak_onset_step": LEAK_ONSET_STEP,
        })
    else:
        base.update({
            "label_detection": 0,
            "label_pipe":      -1,
            "label_position":  -1,
            "label_size":      "none",
            "emitter_coeff":   -1,
            "leak_onset_step": -1,
        })
    return base


def save_manifests(rows: list, out_dir: Path):
    """
    Splits the full scenario list into train, validation, and test sets
    and saves them as manifest CSV files.

    The split is 70% train, 15% validation, 15% test, stratified by
    pipe and size class for the leak scenarios. Stratification ensures
    each split contains a balanced mix of all pipe and size combinations,
    which prevents the model from being evaluated on combinations it
    never saw during training.

    Only valid scenarios (those where the simulation succeeded) are
    included in the splits. Failed scenarios are still listed in
    manifest_full.csv for diagnostic purposes.

    The manifest CSVs list scenario IDs, which the training script uses
    to locate the correct data.csv and labels.json files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    # Save the complete manifest including failed scenarios
    df.to_csv(out_dir / "manifest_full.csv", index=False)

    # Split only valid scenarios into train/val/test
    valid_df  = df[df["valid"] == 1].copy()
    leak_df   = valid_df[valid_df["label_detection"] == 1]
    noleak_df = valid_df[valid_df["label_detection"] == 0]

    def split_70_15_15(data, strat=None):
        """Splits a DataFrame 70/15/15 with optional stratification."""
        train, temp = train_test_split(
            data, test_size=0.30, stratify=strat, random_state=42
        )
        temp_strat = strat.loc[temp.index] if strat is not None else None
        val, test  = train_test_split(
            temp, test_size=0.50, stratify=temp_strat, random_state=42
        )
        return train, val, test

    if len(leak_df) > 0:
        # Stratify leak scenarios by pipe and size together
        leak_strat = leak_df["label_pipe"].astype(str) + "_" + leak_df["label_size"]
        train_l, val_l, test_l = split_70_15_15(leak_df, strat=leak_strat)
    else:
        train_l = val_l = test_l = pd.DataFrame(columns=df.columns)

    if len(noleak_df) > 0:
        # No-leak scenarios are split without stratification
        train_nl, val_nl, test_nl = split_70_15_15(noleak_df)
    else:
        train_nl = val_nl = test_nl = pd.DataFrame(columns=df.columns)

    # Write each split as a separate manifest CSV
    for split_df, fname in [
        (pd.concat([train_l, train_nl]), "manifest_train.csv"),
        (pd.concat([val_l,   val_nl  ]), "manifest_val.csv"),
        (pd.concat([test_l,  test_nl ]), "manifest_test.csv"),
    ]:
        split_df.sort_values("scenario_id").reset_index(drop=True).to_csv(
            out_dir / fname, index=False
        )

    n_train = len(train_l) + len(train_nl)
    n_val   = len(val_l)   + len(val_nl)
    n_test  = len(test_l)  + len(test_nl)
    print(
        f"[OK] Manifests saved — Train: {n_train}, Val: {n_val}, Test: {n_test}",
        flush=True,
    )


# MAIN

def main():
    # Create all output directories before starting generation
    for d in [OUT_SCENARIOS, OUT_INP_DIR, OUT_MANIFESTS, OUT_STATIC]:
        d.mkdir(parents=True, exist_ok=True)

    # Set up error logging to a file — failed scenarios are logged here
    # rather than printed to the terminal so progress output stays clean
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        filemode="w",
    )

    # Verify that the required base network files are present
    BASE_INP  = Path("base.inp")
    BASE2_INP = Path("base2.inp")
    for f in [BASE_INP, BASE2_INP]:
        if not f.exists():
            raise FileNotFoundError(f"Required input file not found: {f}")

    print("=== ST-GCN Dataset Generator v2 ===", flush=True)
    print(
        f"{REPETITIONS} repetitions, {START_SPACING // 3600}h spacing, "
        f"{SIM_DURATION // 3600}h simulation duration",
        flush=True,
    )

    # Total = 5 pipes × 5 positions × 3 sizes × 21 reps × 2 (leak + no-leak) = 3150
    LEAK_COUNT      = len(PIPES) * len(POSITIONS) * len(SIZES) * REPETITIONS
    TOTAL_SCENARIOS = LEAK_COUNT * 2

    # Save the static graph structure before starting scenario generation
    generate_static_graph(OUT_STATIC)

    manifest_rows   = []
    total_generated = 0
    total_failed    = 0

    # Pre-compute all no-leak simulations and cache their results.
    # This is done first for two reasons:
    # 1. On Windows, EPANET creates temporary files during simulation.
    #    After many consecutive simulations, the temp file pool can fill up.
    #    Running no-leak sims before the 1575 leak sims keeps the pool fresher.
    # 2. Each unique (rep, base_file) combination produces one no-leak result
    #    that is shared by all 75 leak configurations with the same rep.
    #    Caching avoids re-running 1575 identical no-leak simulations.
    print("\n[Pre] Computing no-leak simulations (one per repetition)...", flush=True)
    noleak_cache = {}

    for rep in range(REPETITIONS):
        base_path, base_name = select_base(rep, BASE_INP, BASE2_INP)
        try:
            df_nl, wn_nl = run_noleak_sim(base_path, rep)

            # Write the network to a temp file to capture the INP text,
            # then delete the temp file immediately after reading it
            _tmp = OUT_ROOT / f".tmp_noleak_rep{rep}.inp"
            wntr.network.io.write_inpfile(wn_nl, str(_tmp))
            inp_text = _tmp.read_text(encoding="utf-8", errors="ignore")
            _tmp.unlink()

            noleak_cache[rep] = {"df": df_nl, "inp_text": inp_text}
            print(f"  [cached] rep={rep:02d}  ({base_name})", flush=True)
        except Exception as exc:
            noleak_cache[rep] = None   # mark as failed — will be skipped later
            logging.error(
                "FAILED no-leak pre-computation rep=%d | %s\n%s",
                rep, exc, traceback.format_exc(),
            )
            print(f"  [WARNING] No-leak pre-computation failed for rep={rep}: {exc}", flush=True)

    # Part 1 — Leak scenarios (scenario IDs 1 to 1575)
    # Nested loop over all combinations of pipe, position, size, and repetition
    print(f"\n[1/2] Generating {LEAK_COUNT} leak scenarios...", flush=True)

    scn_id = 1
    for pipe_id in PIPES:
        for position in POSITIONS:
            for size_label, emitter_coeff in SIZES:
                for rep in range(REPETITIONS):
                    base_path, base_name = select_base(rep, BASE_INP, BASE2_INP)
                    labels  = make_labels(
                        scn_id, base_name, rep, is_leak=True,
                        pipe_id=pipe_id, position=position,
                        size_label=size_label, emitter_coeff=emitter_coeff,
                    )
                    scn_dir = OUT_SCENARIOS / f"scenario_{scn_id:05d}"
                    scn_dir.mkdir(parents=True, exist_ok=True)

                    # Write labels.json before running the simulation so that
                    # even if the simulation fails, the scenario is documented
                    (scn_dir / "labels.json").write_text(
                        json.dumps(labels, indent=2), encoding="utf-8"
                    )

                    valid = 1
                    try:
                        df, wn_b = run_leak_scenario(
                            base_path, rep, pipe_id, position, emitter_coeff
                        )
                        df.to_csv(scn_dir / "data.csv", index=False)
                        # Save the INP snapshot for reference and debugging
                        wntr.network.io.write_inpfile(
                            wn_b, str(OUT_INP_DIR / f"scenario_{scn_id:05d}.inp")
                        )
                        total_generated += 1
                    except Exception as exc:
                        # Log the failure but continue to the next scenario
                        valid = 0
                        total_failed += 1
                        logging.error(
                            "FAILED scn_%05d | pipe=%s pos=%s size=%s rep=%d | %s\n%s",
                            scn_id, pipe_id, position, size_label, rep,
                            exc, traceback.format_exc(),
                        )

                    manifest_rows.append({**labels, "valid": valid})

                    # Print progress every 50 scenarios
                    if scn_id % 50 == 0:
                        print(
                            f"  [progress] {scn_id}/{TOTAL_SCENARIOS} — "
                            f"generated={total_generated} failed={total_failed}",
                            flush=True,
                        )
                    scn_id += 1

    # Part 2 — No-leak scenarios (scenario IDs 1576 to 3150)
    # These are written from the cached no-leak simulation results,
    # so no new simulations are needed here
    print(f"\n[2/2] Writing {LEAK_COUNT} no-leak scenarios from cache...", flush=True)

    for pipe_id in PIPES:
        for position in POSITIONS:
            for size_label, _ in SIZES:
                for rep in range(REPETITIONS):
                    _, base_name = select_base(rep, BASE_INP, BASE2_INP)
                    labels  = make_labels(scn_id, base_name, rep, is_leak=False)
                    scn_dir = OUT_SCENARIOS / f"scenario_{scn_id:05d}"
                    scn_dir.mkdir(parents=True, exist_ok=True)

                    (scn_dir / "labels.json").write_text(
                        json.dumps(labels, indent=2), encoding="utf-8"
                    )

                    cached = noleak_cache.get(rep)
                    valid  = 1
                    if cached is None:
                        # No-leak pre-computation failed for this rep — skip it
                        valid = 0
                        total_failed += 1
                    else:
                        try:
                            cached["df"].to_csv(scn_dir / "data.csv", index=False)
                            (OUT_INP_DIR / f"scenario_{scn_id:05d}.inp").write_text(
                                cached["inp_text"], encoding="utf-8"
                            )
                            total_generated += 1
                        except Exception as exc:
                            valid = 0
                            total_failed += 1
                            logging.error(
                                "FAILED scn_%05d (no-leak write) | rep=%d | %s\n%s",
                                scn_id, rep, exc, traceback.format_exc(),
                            )

                    manifest_rows.append({**labels, "valid": valid})

                    if scn_id % 50 == 0:
                        print(
                            f"  [progress] {scn_id}/{TOTAL_SCENARIOS} — "
                            f"generated={total_generated} failed={total_failed}",
                            flush=True,
                        )
                    scn_id += 1

    # Save the train/val/test manifest CSVs
    save_manifests(manifest_rows, OUT_MANIFESTS)

    print("\n=== Generation Complete ===", flush=True)
    print(f"  Total scenarios : {TOTAL_SCENARIOS}", flush=True)
    print(f"  Generated (OK)  : {total_generated}", flush=True)
    print(f"  Failed          : {total_failed}", flush=True)
    print(f"  Skipped         : 0", flush=True)
    print(f"  Output root     : {OUT_ROOT.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # On catastrophic failure, print the full traceback and save it to a file
        # so the error can be investigated even if the terminal output is lost
        tb = traceback.format_exc()
        print(tb, flush=True)
        Path("FAILED_run_log.txt").write_text(tb, encoding="utf-8")
        raise
