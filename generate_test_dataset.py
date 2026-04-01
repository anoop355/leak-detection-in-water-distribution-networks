"""
generate_test_dataset.py
-------------------------
Generates an external test set for evaluating trained models on unseen conditions.
Uses base3.inp (different demand patterns from training), unseen leak positions
(0.2, 0.4, 0.6, 0.8) and unseen emitter sizes (0.02, 0.05, 0.09).
7 repetitions at 6-hour spacing gives ~42 hours of demand variety.

Total: 420 leak + 7 no-leak = 427 scenarios.
Output structure matches stgcn_dataset/ so the same evaluation code works as-is.
"""

import copy
import json
import logging
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import wntr


def extract_block(inp_text: str, header: str) -> str:
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
    after  = "\n".join(lines[end:]) + ("\n" if end < len(lines) else "")
    return before + new_block + after


def get_connected_links(wn, node_name: str):
    return list(wn.get_links_for_node(node_name))


def pick_two_links_at_leaknode(wn, leak_node: str, prefer_tokens=(("1a", "1b"))):
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
        raise RuntimeError(f"Node '{node_name}' has no emitter_coefficient.")
    node.emitter_coefficient = convert_emitter_to_internal(wn, emitter_in_inp_units)


# ===========================================================================
# Constants
# ===========================================================================

BASE_INP   = Path("base3.inp")
SOURCE_INP = "base3.inp"

OUT_ROOT      = Path("test_dataset")
OUT_SCENARIOS = OUT_ROOT / "scenarios"
OUT_MANIFESTS = OUT_ROOT / "manifests"
LOG_FILE      = OUT_ROOT / "generation_log.txt"

PIPES            = [1, 2, 3, 4, 5]
TOTAL_LENGTHS    = {1: 100.0, 2: 500.0, 3: 200.0, 4: 200.0, 5: 200.0}
PIPE_TO_LEAKNODE = {1: "L1",  2: "L2",  3: "L3",  4: "L4",  5: "L5"}
LEAK_NODES       = ["L1", "L2", "L3", "L4", "L5"]
SENSOR_NODES     = ["2", "3", "4", "5", "6"]
SENSOR_LINKS     = ["1a", "2a", "3a", "4a", "5a"]

# midpoints between training positions [0.1, 0.3, 0.5, 0.7, 0.9]
TEST_POSITIONS = [0.2, 0.4, 0.6, 0.8]

# emitter sizes between / above training values (0.01, 0.03, 0.06)
TEST_SIZES = [
    ("S", 0.02),   # between S and M
    ("M", 0.05),   # between M and L
    ("L", 0.09),   # above L
]

TEST_REPETITIONS = 7
START_SPACING    = 6 * 3600   # 6-hour spacing

SIM_DURATION  = 3 * 3600
HYD_TIMESTEP  = 15 * 60
PAT_TIMESTEP  = 60
N_TIMESTEPS   = 12
T_AXIS_MIN    = [i * 15 for i in range(N_TIMESTEPS)]

DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday"]


def rep_to_day_time(rep: int):
    total_hours = rep * (START_SPACING // 3600)
    day_idx     = (total_hours // 24) % 7
    hour_of_day = total_hours % 24
    return DAY_NAMES[day_idx], f"{hour_of_day:02d}:00"


def load_and_configure(base_path: Path, duration_sec: int, rep: int):
    """Load network and rotate demand patterns to the correct weekly position."""
    wn = wntr.network.WaterNetworkModel(str(base_path))
    wn.options.time.duration           = duration_sec
    wn.options.time.hydraulic_timestep = HYD_TIMESTEP
    wn.options.time.report_timestep    = HYD_TIMESTEP
    wn.options.time.pattern_timestep   = PAT_TIMESTEP
    wn.options.time.start_clocktime    = 0

    offset_steps = int((rep * START_SPACING) // PAT_TIMESTEP)
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
        raise RuntimeError(f"Missing pressure nodes: {missing_nodes}")
    if missing_links:
        raise RuntimeError(f"Missing flow links: {missing_links}")
    if pressures.shape[0] < n_rows:
        raise RuntimeError(
            f"Expected ≥{n_rows} timesteps, got {pressures.shape[0]}."
        )

    p = pressures[SENSOR_NODES].values[:n_rows]
    q = flows[SENSOR_LINKS].values[:n_rows]

    if np.any(p < 0):
        raise RuntimeError(
            f"Negative pressures detected (min={p.min():.4f} m). Scenario invalid."
        )
    return p, q


def build_df(p: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"t": T_AXIS_MIN})
    for i, node in enumerate(SENSOR_NODES):
        df[f"P{node}"] = p[:, i]
    for i, link in enumerate(SENSOR_LINKS):
        df[f"Q{link}"] = q[:, i]
    return df


def run_leak_scenario(rep: int, pipe_id: int, position: float, emitter_coeff: float):
    """
    3-hour simulation with leak activating at t = 60 min (step 4).
    Step demand pattern: 60 zeros then 120 ones at 1-min resolution.
    """
    leak_node  = PIPE_TO_LEAKNODE[pipe_id]
    total_len  = TOTAL_LENGTHS[pipe_id]
    len_a      = position * total_len
    len_b      = total_len - len_a
    prefer     = (f"{pipe_id}a", f"{pipe_id}b")

    wn = load_and_configure(BASE_INP, SIM_DURATION, rep)
    clear_emitters(wn, LEAK_NODES)

    link_a, link_b = pick_two_links_at_leaknode(wn, leak_node, prefer)
    set_link_lengths(wn, link_a, len_a, link_b, len_b)
    set_emitter(wn, leak_node, emitter_coeff)

    # reset emitter before adding step demand
    Ce_internal = convert_emitter_to_internal(wn, emitter_coeff)
    wn.get_node(leak_node).emitter_coefficient = 0.0

    # quick steady-state run to get pre-leak pressure
    wn_pre = copy.deepcopy(wn)
    wn_pre.options.time.duration = 0
    res_pre = wntr.sim.EpanetSimulator(wn_pre).run_sim()
    P_pre   = float(res_pre.node["pressure"][leak_node].iloc[0])
    q_leak  = Ce_internal * (max(P_pre, 0.0) ** 0.5)

    pat_name = f"_lk_{pipe_id}_{rep}"
    wn.add_pattern(pat_name, [0.0] * 60 + [1.0] * 120)
    wn.get_node(leak_node).demand_timeseries_list.append(
        (q_leak, wn.get_pattern(pat_name), "leak_demand")
    )

    p, q = extract_signals(wntr.sim.EpanetSimulator(wn).run_sim(), N_TIMESTEPS)
    return build_df(p, q)


def run_noleak_scenario(rep: int):
    """3-hour no-leak simulation."""
    wn = load_and_configure(BASE_INP, SIM_DURATION, rep)
    clear_emitters(wn, LEAK_NODES)
    p, q = extract_signals(wntr.sim.EpanetSimulator(wn).run_sim(), N_TIMESTEPS)
    return build_df(p, q)


def main():
    if not BASE_INP.exists():
        raise FileNotFoundError(f"Cannot find {BASE_INP}.")

    OUT_SCENARIOS.mkdir(parents=True, exist_ok=True)
    OUT_MANIFESTS.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    n_leak_scns   = len(PIPES) * len(TEST_POSITIONS) * len(TEST_SIZES) * TEST_REPETITIONS
    n_noleak_scns = TEST_REPETITIONS
    total         = n_leak_scns + n_noleak_scns

    print(f"Test dataset generator")
    print(f"  Source     : {BASE_INP}")
    print(f"  Positions  : {TEST_POSITIONS}")
    print(f"  Sizes      : {[(s, c) for s, c in TEST_SIZES]}")
    print(f"  Reps       : {TEST_REPETITIONS} × {START_SPACING//3600}h spacing")
    print(f"  Leak scns  : {n_leak_scns}")
    print(f"  No-leak scns: {n_noleak_scns}")
    print(f"  Total      : {total}")
    print()

    manifest_rows = []
    scn_id        = 1
    generated     = 0
    failed        = 0

    # no-leak scenarios first
    print("[1/2] Generating no-leak scenarios...")
    for rep in range(TEST_REPETITIONS):
        scn_dir = OUT_SCENARIOS / f"scenario_{scn_id:05d}"
        scn_dir.mkdir(parents=True, exist_ok=True)
        day_name, start_time = rep_to_day_time(rep)

        try:
            df = run_noleak_scenario(rep)
            df.to_csv(scn_dir / "data.csv", index=False)

            labels = {
                "scenario_id":     scn_id,
                "source_inp":      SOURCE_INP,
                "repetition":      rep,
                "start_day":       day_name,
                "start_time":      start_time,
                "label_detection": 0,
                "label_pipe":      -1,
                "label_position":  -1,
                "label_size":      "none",
                "emitter_coeff":   -1,
                "leak_onset_step": -1,
            }
            (scn_dir / "labels.json").write_text(
                json.dumps(labels, indent=2), encoding="utf-8"
            )
            manifest_rows.append({**labels, "valid": 1})
            generated += 1
            logging.info(f"[OK] scenario_{scn_id:05d} no-leak rep={rep}")

        except Exception as e:
            logging.error(f"[FAIL] scenario_{scn_id:05d} no-leak rep={rep}: {e}")
            manifest_rows.append({
                "scenario_id": scn_id, "source_inp": SOURCE_INP,
                "repetition": rep, "start_day": day_name,
                "start_time": start_time, "label_detection": 0,
                "label_pipe": -1, "label_position": -1,
                "label_size": "none", "emitter_coeff": -1,
                "leak_onset_step": -1, "valid": 0,
            })
            failed += 1

        scn_id += 1

    # leak scenarios
    print("[2/2] Generating leak scenarios...")
    for pipe_id in PIPES:
        for position in TEST_POSITIONS:
            for size_label, emitter_coeff in TEST_SIZES:
                for rep in range(TEST_REPETITIONS):
                    scn_dir = OUT_SCENARIOS / f"scenario_{scn_id:05d}"
                    scn_dir.mkdir(parents=True, exist_ok=True)
                    day_name, start_time = rep_to_day_time(rep)

                    try:
                        df = run_leak_scenario(rep, pipe_id, position, emitter_coeff)
                        df.to_csv(scn_dir / "data.csv", index=False)

                        labels = {
                            "scenario_id":     scn_id,
                            "source_inp":      SOURCE_INP,
                            "repetition":      rep,
                            "start_day":       day_name,
                            "start_time":      start_time,
                            "label_detection": 1,
                            "label_pipe":      pipe_id,
                            "label_position":  position,
                            "label_size":      size_label,
                            "emitter_coeff":   emitter_coeff,
                            "leak_onset_step": 4,
                        }
                        (scn_dir / "labels.json").write_text(
                            json.dumps(labels, indent=2), encoding="utf-8"
                        )
                        manifest_rows.append({**labels, "valid": 1})
                        generated += 1
                        logging.info(
                            f"[OK] scenario_{scn_id:05d} pipe={pipe_id} "
                            f"pos={position} size={size_label} rep={rep}"
                        )

                    except Exception as e:
                        logging.error(
                            f"[FAIL] scenario_{scn_id:05d} pipe={pipe_id} "
                            f"pos={position} size={size_label} rep={rep}: {e}"
                        )
                        manifest_rows.append({
                            "scenario_id": scn_id, "source_inp": SOURCE_INP,
                            "repetition": rep, "start_day": day_name,
                            "start_time": start_time, "label_detection": 1,
                            "label_pipe": pipe_id, "label_position": position,
                            "label_size": size_label, "emitter_coeff": emitter_coeff,
                            "leak_onset_step": 4, "valid": 0,
                        })
                        failed += 1

                    scn_id += 1

                    if generated % 50 == 0 and generated > 0:
                        print(f"  [progress] {generated}/{total}  failed={failed}")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUT_MANIFESTS / "manifest.csv", index=False)

    print()
    print("=== Generation Complete ===")
    print(f"  Total scenarios : {total}")
    print(f"  Generated (OK)  : {generated}")
    print(f"  Failed          : {failed}")
    print(f"  Output root     : {OUT_ROOT.resolve()}")
    print(f"  Log             : {LOG_FILE.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        Path("FAILED_generate_test_dataset_log.txt").write_text(tb, encoding="utf-8")
        raise