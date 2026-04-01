import copy
import json
import traceback
from pathlib import Path

import pandas as pd
import wntr


# -------------------------
# INP block utilities
# -------------------------
def extract_block(inp_text: str, header: str) -> str:
    """
    Extracts a block like [TIMES] ... until next [BLOCK] or EOF.
    Returns the full block text including the header line.
    """
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
    """
    Replaces a block [HEADER]... in inp_text with new_block.
    If block doesn't exist, appends it at the end.
    """
    lines = inp_text.splitlines()
    header_upper = header.upper()

    start = None
    for i, line in enumerate(lines):
        if line.strip().upper() == header_upper:
            start = i
            break

    if start is None:
        # Append
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


# -------------------------
# Network helpers
# -------------------------
def get_connected_links(wn, node_name: str):
    return list(wn.get_links_for_node(node_name))


def pick_two_links_at_leaknode(wn, leak_node: str, prefer_tokens=("1a", "1b")):
    """
    For a leak node (e.g., L1), we want the two connected links.
    We try to map them to '<pipe_id>a' and '<pipe_id>b' by name.
    Priority:
      1) exact link names
      2) substring contains token
      3) fallback to sorted two links
    """
    links = get_connected_links(wn, leak_node)
    if len(links) != 2:
        raise RuntimeError(
            f"Leak node '{leak_node}' must connect to exactly 2 links. Found {len(links)}: {links}"
        )

    s = set(links)
    if prefer_tokens[0] in s and prefer_tokens[1] in s:
        return prefer_tokens[0], prefer_tokens[1]

    def find_contains(token: str):
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
    """
    WNTR stores flows internally in SI (m^3/s). EPANET INP may use LPS, GPM, etc.
    Your symptom: 0.06 became 60.0 -> exactly 1000x, which matches LPS vs m^3/s.

    We'll detect INP flow units from wn.options.hydraulic.inpfile_units if available.
    If units are LPS, convert L/s -> m^3/s by dividing by 1000.

    This conversion is applied directly to the emitter coefficient value. Pressure/head
    in your network is still in meters, so only the flow unit scaling is needed here.
    """
    try:
        units = str(getattr(wn.options.hydraulic, "inpfile_units", "")).upper()
    except Exception:
        units = ""

    if units == "LPS":
        return float(emitter_in_inp_units) / 1000.0  # L/s -> m^3/s

    return float(emitter_in_inp_units)


def set_emitter(wn, node_name: str, emitter_in_inp_units: float):
    node = wn.get_node(node_name)
    if not hasattr(node, "emitter_coefficient"):
        raise RuntimeError(f"Node '{node_name}' has no emitter_coefficient (not a junction?).")

    node.emitter_coefficient = convert_emitter_to_internal(wn, emitter_in_inp_units)


# -------------------------
# Simulation output formatting
# -------------------------
def run_sim_and_save_signals_filtered(wn, out_dir: Path):
    """
    Saves EXACTLY:
      t, P2..P6, Q1a..Q5a

    t is saved in minutes (0,1,2,...) to match your old dataset screenshot.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    pressures = results.node["pressure"].copy()   # columns are node IDs
    flows = results.link["flowrate"].copy()       # columns are link IDs

    # Time index is seconds
    t_sec = pressures.index.values.astype(float)
    t_min = (t_sec / 60.0)

    # Required node IDs and link IDs (based on your naming)
    req_nodes = ["2", "3", "4", "5", "6"]
    req_links = ["1a", "2a", "3a", "4a", "5a"]

    missing_nodes = [n for n in req_nodes if n not in pressures.columns]
    missing_links = [l for l in req_links if l not in flows.columns]

    if missing_nodes:
        raise RuntimeError(
            f"Missing pressure nodes in results: {missing_nodes}\n"
            f"Available node columns (first 50): {list(pressures.columns)[:50]}"
        )
    if missing_links:
        raise RuntimeError(
            f"Missing flow links in results: {missing_links}\n"
            f"Available link columns (first 50): {list(flows.columns)[:50]}"
        )

    df = pd.DataFrame()
    df["t"] = t_min

    # Add pressures renamed as P2..P6
    for n in req_nodes:
        df[f"P{n}"] = pressures[n].values

    # Add flows renamed as Q1a..Q5a
    for l in req_links:
        df[f"Q{l}"] = flows[l].values

    out_csv = out_dir / "signals.csv"
    df.to_csv(out_csv, index=False)


def write_labels_json(out_path: Path, source_inp: str, scn_number: int, leaks):
    """
    Writes labels.json exactly in the format you specified.
    """
    payload = {
        "source_inp": source_inp,
        "leak_present": 1 if len(leaks) > 0 else 0,
        "scn_number": scn_number,
        "leaks": leaks
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -------------------------
# Main
# -------------------------
def main():
    BASE_INP = Path("base.inp")

    OUT_INP_DIR = Path("Generated_Scenarios")
    OUT_RESULTS_DIR = Path("Generate_Scenarios_output")

    # Leak nodes are known by name (not by emitter>0)
    LEAK_NODES = ["L1", "L2", "L3", "L4", "L5"]

    # Pipe -> leak node mapping (your structure)
    PIPE_TO_LEAKNODE = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}

    # Total lengths given (m)
    TOTAL_LENGTHS = {1: 100.0, 2: 500.0, 3: 200.0, 4: 200.0, 5: 200.0}

    # Positions (normalized) and sizes as specified
    R_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
    SIZES = [
        ("S", 0.01),
        ("M", 0.03),
        ("L", 0.06),
    ]

    if not BASE_INP.exists():
        raise RuntimeError(f"Cannot find {BASE_INP}. Put base.inp in the folder you run this script from.")

    OUT_INP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Read base INP text so we can preserve [TIMES] exactly for every generated INP
    base_text = BASE_INP.read_text(encoding="utf-8", errors="ignore")
    base_times_block = extract_block(base_text, "[TIMES]")

    # Load base network once
    wn_base = wntr.network.WaterNetworkModel(str(BASE_INP))

    # Confirm leak nodes exist
    for ln in LEAK_NODES:
        if ln not in wn_base.node_name_list:
            raise RuntimeError(f"Leak node '{ln}' not found in base.inp nodes.")

    # Scenario counter: 1..75
    scn_number = 1
    total_scenarios = len(PIPE_TO_LEAKNODE) * len(R_POSITIONS) * len(SIZES)  # 5*5*3 = 75

    print(f"Generating {total_scenarios} scenarios...", flush=True)

    for pipe_id in [1, 2, 3, 4, 5]:
        leak_node = PIPE_TO_LEAKNODE[pipe_id]
        total_len = TOTAL_LENGTHS[pipe_id]

        # Prefer tokens for this pipe, e.g., ("2a","2b") etc.
        prefer_tokens = (f"{pipe_id}a", f"{pipe_id}b")

        for r in R_POSITIONS:
            # Convert normalized position to distance
            leak_distance_m = r * total_len
            len_a = leak_distance_m
            len_b = total_len - leak_distance_m

            for size_level, emitter_coeff_in_inp_units in SIZES:
                # Per-scenario output folder (ONLY signals.csv and labels.json will be saved)
                scn_out_dir = OUT_RESULTS_DIR / f"scn_{scn_number}"
                scn_out_dir.mkdir(parents=True, exist_ok=True)

                # Work on a fresh copy each time
                wn = copy.deepcopy(wn_base)

                # Clear all emitters, then set only this one
                clear_emitters(wn, LEAK_NODES)

                # Identify the two split links at this leak node and set lengths
                link_a, link_b = pick_two_links_at_leaknode(wn, leak_node, prefer_tokens=prefer_tokens)
                set_link_lengths(wn, link_a, len_a, link_b, len_b)

                # Set emitter coefficient (unit conversion handled)
                set_emitter(wn, leak_node, emitter_coeff_in_inp_units)

                # Write modified INP
                inp_name = f"scn_{scn_number}.inp"
                out_inp_path = OUT_INP_DIR / inp_name
                wntr.network.io.write_inpfile(wn, str(out_inp_path))

                # Replace [TIMES] block to match base.inp exactly
                if base_times_block:
                    gen_text = out_inp_path.read_text(encoding="utf-8", errors="ignore")
                    new_text = replace_block(gen_text, "[TIMES]", base_times_block)
                    out_inp_path.write_text(new_text, encoding="utf-8")

                # Run sim and save signals.csv (filtered columns)
                run_sim_and_save_signals_filtered(wn, scn_out_dir)

                # Save labels.json EXACTLY in your required format
                leaks = [
                    {
                        "pipe_id": int(pipe_id),
                        "position": float(r),
                        "size_level": str(size_level),
                        "emitter_coeff": float(emitter_coeff_in_inp_units),
                    }
                ]
                write_labels_json(scn_out_dir / "labels.json", inp_name, scn_number, leaks)

                print(f"[OK] scn_{scn_number}/{total_scenarios} done", flush=True)
                scn_number += 1

    print("\nAll done.", flush=True)
    print(f"INP files: {OUT_INP_DIR.resolve()}", flush=True)
    print(f"Outputs:   {OUT_RESULTS_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        Path("FAILED_run_log.txt").write_text(tb, encoding="utf-8")
        raise
