"""
generate_stgcn_dataset_v2.py
-----------------------------
Updated from v1. Main changes:
- More scenarios: 3150 total (1575 leak + 1575 no-leak), up from 2100
- Increased repetitions from 14 to 21, spacing reduced from 12h to 8h
  (21 x 8h = 168h still covers a full week, but with denser time-of-day sampling)
- Simulation duration extended from 3h to 6h (24 timesteps instead of 12)
- Leak pattern updated to match: 60 zeros + 300 ones (was 60 + 120)
- Added LEAK_PAT_OFF_STEPS / LEAK_PAT_ON_STEPS constants so the pattern
  lengths are not hardcoded in run_leak_scenario
- Output goes to stgcn_dataset_v2/ to avoid overwriting v1 data
"""

import copy
import json
import logging
import traceback
from pathlib import Path

# NOTE: ControlAction/SimTimeCondition are not used here. EpanetSimulator
# delegates to ENsolveH and ignores Python controls. Leak onset is handled
# via a step demand pattern instead, which EPANET applies natively.

import numpy as np
import pandas as pd
import wntr
from sklearn.model_selection import train_test_split


def extract_block(inp_text: str, header: str) -> str:
    """Extract a section like [TIMES]...[NEXT BLOCK] from an INP file string."""
    lines = inp_text.splitlines()
    header_upper = header.upper()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper() == header_upper:
            start = i
            break
    if start is None:
        return ""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("[") and lines[j].strip().endswith("]"):
            end = j
            break
    return "\n".join(lines[start:end]) + "\n"


def replace_block(inp_text: str, header: str, new_block: str) -> str:
    """Replace a [HEADER]... section in an INP file string."""
    lines = inp_text.splitlines()
    header_upper = header.upper()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper() == header_upper:
            start = i
            break
    if start is None:
        if not inp_text.endswith("\n"):
            inp_text += "\n"
        return inp_text + "\n" + new_block
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].strip().startswith("[") and lines[j].strip().endswith("]"):
            end = j
            break
    before = "\n".join(lines[:start]) + "\n"
    after = "\n".join(lines[end:]) + ("\n" if end < len(lines) else "")
    return before + new_block + after


def get_connected_links(wn, node_name: str):
    return list(wn.get_links_for_node(node_name))


def pick_two_links_at_leaknode(wn, leak_node: str, prefer_tokens=("1a", "1b")):
    """Return the two links connected to a leak node. Priority: exact > substring > sorted."""
    links = get_connected_links(wn, leak_node)
    if len(links) != 2:
        raise RuntimeError(
            f"Leak node '{leak_node}' must connect to exactly 2 links. Found {len(links)}: {links}"
        )
    s = set(links)
    if prefer_tokens[0] in s and prefer_tokens[1] in s:
        return prefer_tokens[0], prefer_tokens[1]

    def find_contains(token):
        for ln in links:
            if token in ln:
                return ln
        return None

    a = find_contains(prefer_tokens[0])
    b = find_contains(prefer_tokens[1])
    if a and b and a != b:
        return a, b
    links_sorted = sorted(links)
    return links_sorted[0], links_sorted[1]


def clear_emitters(wn, leak_nodes):
    for ln in leak_nodes:
        node = wn.get_node(ln)
        if hasattr(node, "emitter_coefficient"):
            node.emitter_coefficient = 0.0


def set_link_lengths(wn, link_a: str, len_a: float, link_b: str, len_b: float):
    wn.get_link(link_a).length = float(len_a)
    wn.get_link(link_b).length = float(len_b)


def convert_emitter_to_internal(wn, emitter_in_inp_units: float) -> float:
    """Convert emitter coefficient from LPS (INP units) to m³/s (WNTR internal)."""
    try:
        units = str(getattr(wn.options.hydraulic, "inpfile_units", "")).upper()
    except Exception:
        units = ""
    if units == "LPS":
        return float(emitter_in_inp_units) / 1000.0
    return float(emitter_in_inp_units)


def set_emitter(wn, node_name: str, emitter_in_inp_units: float):
    node = wn.get_node(node_name)
    if not hasattr(node, "emitter_coefficient"):
        raise RuntimeError(f"Node '{node_name}' has no emitter_coefficient (not a junction?).")
    node.emitter_coefficient = convert_emitter_to_internal(wn, emitter_in_inp_units)


# ===========================================================================
# Dataset constants
# ===========================================================================

PIPES         = [1, 2, 3, 4, 5]
POSITIONS     = [0.1, 0.3, 0.5, 0.7, 0.9]
SIZES         = [("S", 0.01), ("M", 0.03), ("L", 0.06)]
REPETITIONS   = 21           # 21 reps × 8h spacing = 168h (full week)
START_SPACING = 8 * 3600     # 8-hour spacing

TOTAL_LENGTHS    = {1: 100.0, 2: 500.0, 3: 200.0, 4: 200.0, 5: 200.0}
PIPE_TO_LEAKNODE = {1: "L1",  2: "L2",  3: "L3",  4: "L4",  5: "L5"}
LEAK_NODES       = ["L1", "L2", "L3", "L4", "L5"]
SENSOR_NODES     = ["2", "3", "4", "5", "6"]
SENSOR_LINKS     = ["1a", "2a", "3a", "4a", "5a"]

SIM_DURATION    = 6 * 3600   # 6-hour simulation
HYD_TIMESTEP    = 15 * 60    # 15-min hydraulic/report timestep
PAT_TIMESTEP    = 60         # 1-min pattern timestep
LEAK_ONSET_STEP = 4          # timestep index when leak activates (t = 60 min)

N_TIMESTEPS  = 24                                    # 6h / 15min = 24 timesteps
T_AXIS_MIN   = [i * 15 for i in range(N_TIMESTEPS)]  # 0, 15, ..., 345

# named constants so the pattern lengths are not hardcoded in run_leak_scenario
LEAK_PAT_OFF_STEPS = 60   # 60 min pre-leak
LEAK_PAT_ON_STEPS  = 300  # 300 min active leak

DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

OUT_ROOT      = Path("stgcn_dataset_v2")
OUT_SCENARIOS = OUT_ROOT / "scenarios"
OUT_INP_DIR   = OUT_ROOT / "scenarios_inp"
OUT_MANIFESTS = OUT_ROOT / "manifests"
OUT_STATIC    = OUT_ROOT / "static_graph"
LOG_FILE      = OUT_ROOT / "generation_log.txt"


# ===========================================================================
# Time helpers
# ===========================================================================

def rep_start_sec(rep: int) -> int:
    """Scenario start time for repetition rep, in seconds from midnight Sunday."""
    return rep * START_SPACING


def rep_to_day_time(rep: int):
    """Return (day_name, 'HH:MM') for a given repetition index."""
    total_hours = rep * (START_SPACING // 3600)
    day_idx     = (total_hours // 24) % 7
    hour_of_day = total_hours % 24
    return DAY_NAMES[day_idx], f"{hour_of_day:02d}:00"


def select_base(rep: int, base_inp: Path, base2_inp: Path):
    """Even reps use base.inp; odd reps use base2.inp."""
    if rep % 2 == 0:
        return base_inp, "base.inp"
    return base2_inp, "base2.inp"


# ===========================================================================
# WNTR helpers
# ===========================================================================

def load_and_configure(base_path: Path, duration_sec: int,
                       rep: int, extra_offset_sec: int = 0):
    """
    Load the network and rotate demand patterns to the correct weekly position.

    start_clocktime is kept at 0 to avoid EPANET Error 200 on Windows for
    reps >= 3. The correct time-of-week position is achieved by rotating all
    demand patterns by offset_steps = (rep * START_SPACING + extra_offset_sec) / 60.
    """
    wn = wntr.network.WaterNetworkModel(str(base_path))
    wn.options.time.duration           = duration_sec
    wn.options.time.hydraulic_timestep = HYD_TIMESTEP
    wn.options.time.report_timestep    = HYD_TIMESTEP
    wn.options.time.pattern_timestep   = PAT_TIMESTEP
    wn.options.time.start_clocktime    = 0

    offset_steps = int((rep * START_SPACING + extra_offset_sec) // PAT_TIMESTEP)
    for pat_name in wn.pattern_name_list:
        mults = np.array(wn.get_pattern(pat_name).multipliers)
        if len(mults) > 0:
            rotated = np.roll(mults, -int(offset_steps % len(mults)))
            wn.get_pattern(pat_name).multipliers = list(rotated)

    return wn


def extract_signals(results, n_rows: int):
    """Extract pressure and flow from WNTR results. Returns (p, q) each shape (n_rows, 5)."""
    pressures = results.node["pressure"]
    flows     = results.link["flowrate"]

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

    if np.any(p < 0):
        raise RuntimeError(
            f"Negative pressures detected (min={p.min():.4f} m). Scenario invalid."
        )
    return p, q


def build_df(p: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    """Build the 24-row signals DataFrame from pressure and flow arrays."""
    df = pd.DataFrame({"t": T_AXIS_MIN})
    for i, node in enumerate(SENSOR_NODES):
        df[f"P{node}"] = p[:, i]
    for i, link in enumerate(SENSOR_LINKS):
        df[f"Q{link}"] = q[:, i]
    return df


# ===========================================================================
# Scenario runners
# ===========================================================================

def run_leak_scenario(base_path: Path, rep: int, pipe_id: int,
                      position: float, emitter_coeff: float):
    """
    Run a single 6-hour simulation with the leak activating at t = 60 min (step 4).
    Pre-leak phase: t = 0-60 min (4 timesteps). Active leak: t = 60-360 min (20 timesteps).

    wn_inp is a snapshot taken before the control is added — used for writing
    the .inp file since EPANET .inp format does not support WNTR control objects.
    """
    leak_node_name = PIPE_TO_LEAKNODE[pipe_id]
    total_len      = TOTAL_LENGTHS[pipe_id]
    len_a_seg      = position * total_len
    len_b_seg      = total_len - len_a_seg
    prefer_tokens  = (f"{pipe_id}a", f"{pipe_id}b")

    wn = load_and_configure(base_path, SIM_DURATION, rep, extra_offset_sec=0)
    clear_emitters(wn, LEAK_NODES)

    link_a, link_b = pick_two_links_at_leaknode(wn, leak_node_name, prefer_tokens)
    set_link_lengths(wn, link_a, len_a_seg, link_b, len_b_seg)
    set_emitter(wn, leak_node_name, emitter_coeff)

    # snapshot before control is added
    wn_inp = copy.deepcopy(wn)

    # reset emitter so simulation starts leak-free
    wn.get_node(leak_node_name).emitter_coefficient = 0.0

    Ce_internal = convert_emitter_to_internal(wn, emitter_coeff)

    # quick steady-state run to estimate pre-leak node pressure
    wn_pre = copy.deepcopy(wn)
    wn_pre.options.time.duration = 0
    res_pre = wntr.sim.EpanetSimulator(wn_pre).run_sim()
    P_pre   = float(res_pre.node["pressure"][leak_node_name].iloc[0])
    q_leak  = Ce_internal * (max(P_pre, 0.0) ** 0.5)   # m³/s

    # step demand pattern using named constants
    pat_name = f"_lk_{pipe_id}_{rep}"
    wn.add_pattern(pat_name, [0.0] * LEAK_PAT_OFF_STEPS + [1.0] * LEAK_PAT_ON_STEPS)
    wn.get_node(leak_node_name).demand_timeseries_list.append(
        (q_leak, wn.get_pattern(pat_name), "leak_demand")
    )

    p, q = extract_signals(wntr.sim.EpanetSimulator(wn).run_sim(), N_TIMESTEPS)
    return build_df(p, q), wn_inp


def run_noleak_sim(base_path: Path, rep: int):
    """6-hour no-leak simulation. Returns (df, wn) with all 24 timesteps."""
    wn = load_and_configure(base_path, SIM_DURATION, rep, extra_offset_sec=0)
    clear_emitters(wn, LEAK_NODES)
    p, q = extract_signals(wntr.sim.EpanetSimulator(wn).run_sim(), N_TIMESTEPS)
    return build_df(p, q), wn


# ===========================================================================
# Static graph
# ===========================================================================

def generate_static_graph(out_dir: Path):
    """Write adjacency matrix, pipe endpoint map, and node/pipe lists to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes    = [1, 2, 3, 4, 5, 6]
    node_idx = {nid: i for i, nid in enumerate(nodes)}

    pipe_endpoints_ids = [(1, 2), (2, 3), (3, 4), (4, 5), (4, 6)]

    n   = len(nodes)
    adj = np.zeros((n, n), dtype=int)
    for src, dst in pipe_endpoints_ids:
        i, j = node_idx[src], node_idx[dst]
        adj[i, j] = 1
        adj[j, i] = 1

    pipe_endpoint_map = np.array(
        [[node_idx[s], node_idx[d]] for s, d in pipe_endpoints_ids]
    )
    pipe_lengths = np.array([TOTAL_LENGTHS[p] for p in PIPES], dtype=float)

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


# ===========================================================================
# Labels and manifest helpers
# ===========================================================================

def make_labels(scenario_id: int, source_inp: str, rep: int, is_leak: bool,
                pipe_id=None, position=None, size_label=None, emitter_coeff=None) -> dict:
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
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "manifest_full.csv", index=False)

    valid_df  = df[df["valid"] == 1].copy()
    leak_df   = valid_df[valid_df["label_detection"] == 1]
    noleak_df = valid_df[valid_df["label_detection"] == 0]

    def split_70_15_15(data, strat=None):
        train, temp = train_test_split(
            data, test_size=0.30, stratify=strat, random_state=42
        )
        temp_strat = strat.loc[temp.index] if strat is not None else None
        val, test  = train_test_split(
            temp, test_size=0.50, stratify=temp_strat, random_state=42
        )
        return train, val, test

    if len(leak_df) > 0:
        leak_strat = leak_df["label_pipe"].astype(str) + "_" + leak_df["label_size"]
        train_l, val_l, test_l = split_70_15_15(leak_df, strat=leak_strat)
    else:
        train_l = val_l = test_l = pd.DataFrame(columns=df.columns)

    if len(noleak_df) > 0:
        train_nl, val_nl, test_nl = split_70_15_15(noleak_df)
    else:
        train_nl = val_nl = test_nl = pd.DataFrame(columns=df.columns)

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


# ===========================================================================
# Main
# ===========================================================================

def main():
    for d in [OUT_SCENARIOS, OUT_INP_DIR, OUT_MANIFESTS, OUT_STATIC]:
        d.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(message)s",
        filemode="w",
    )

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

    LEAK_COUNT      = len(PIPES) * len(POSITIONS) * len(SIZES) * REPETITIONS
    TOTAL_SCENARIOS = LEAK_COUNT * 2

    generate_static_graph(OUT_STATIC)

    manifest_rows   = []
    total_generated = 0
    total_failed    = 0

    # run no-leak simulations first and cache results — avoids EPANET temp-file
    # pool exhaustion on Windows after many leak simulations
    print("\n[Pre] Computing no-leak simulations (one per repetition)...", flush=True)
    noleak_cache = {}

    for rep in range(REPETITIONS):
        base_path, base_name = select_base(rep, BASE_INP, BASE2_INP)
        try:
            df_nl, wn_nl = run_noleak_sim(base_path, rep)
            _tmp = OUT_ROOT / f".tmp_noleak_rep{rep}.inp"
            wntr.network.io.write_inpfile(wn_nl, str(_tmp))
            inp_text = _tmp.read_text(encoding="utf-8", errors="ignore")
            _tmp.unlink()
            noleak_cache[rep] = {"df": df_nl, "inp_text": inp_text}
            print(f"  [cached] rep={rep:02d}  ({base_name})", flush=True)
        except Exception as exc:
            noleak_cache[rep] = None
            logging.error(
                "FAILED no-leak pre-computation rep=%d | %s\n%s",
                rep, exc, traceback.format_exc(),
            )
            print(f"  [WARNING] No-leak pre-computation failed for rep={rep}: {exc}", flush=True)

    # Part 1: leak scenarios (IDs 1 - 1575)
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

                    # write labels.json whether or not the simulation succeeds
                    (scn_dir / "labels.json").write_text(
                        json.dumps(labels, indent=2), encoding="utf-8"
                    )

                    valid = 1
                    try:
                        df, wn_b = run_leak_scenario(
                            base_path, rep, pipe_id, position, emitter_coeff
                        )
                        df.to_csv(scn_dir / "data.csv", index=False)
                        wntr.network.io.write_inpfile(
                            wn_b, str(OUT_INP_DIR / f"scenario_{scn_id:05d}.inp")
                        )
                        total_generated += 1
                    except Exception as exc:
                        valid = 0
                        total_failed += 1
                        logging.error(
                            "FAILED scn_%05d | pipe=%s pos=%s size=%s rep=%d | %s\n%s",
                            scn_id, pipe_id, position, size_label, rep,
                            exc, traceback.format_exc(),
                        )

                    manifest_rows.append({**labels, "valid": valid})

                    if scn_id % 50 == 0:
                        print(
                            f"  [progress] {scn_id}/{TOTAL_SCENARIOS} — "
                            f"generated={total_generated} failed={total_failed}",
                            flush=True,
                        )
                    scn_id += 1

    # Part 2: no-leak scenarios (IDs 1576 - 3150), written from cache
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
        tb = traceback.format_exc()
        print(tb, flush=True)
        Path("FAILED_run_log.txt").write_text(tb, encoding="utf-8")
        raise