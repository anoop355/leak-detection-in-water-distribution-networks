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
    after = "\n".join(lines[end:]) + ("\n" if end < len(lines) else "")
    return before + new_block + after


# -------------------------
# Network helpers
# -------------------------
def get_connected_links(wn, node_name: str):
    return list(wn.get_links_for_node(node_name))


def pick_two_links_at_leaknode(wn, leak_node: str, prefer_tokens):
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
    try:
        units = str(getattr(wn.options.hydraulic, "inpfile_units", "")).upper()
    except Exception:
        units = ""

    # Confirmed mismatch factor was 1000x, consistent with LPS vs m^3/s
    if units == "LPS":
        return float(emitter_in_inp_units) / 1000.0

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
    t in minutes
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    pressures = results.node["pressure"].copy()
    flows = results.link["flowrate"].copy()

    t_sec = pressures.index.values.astype(float)
    t_min = t_sec / 60.0

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

    for n in req_nodes:
        df[f"P{n}"] = pressures[n].values

    for l in req_links:
        df[f"Q{l}"] = flows[l].values

    (out_dir / "signals.csv").write_text(df.to_csv(index=False), encoding="utf-8")


def write_labels_json(out_path: Path, source_inp: str, scn_number: int, leaks):
    payload = {
        "source_inp": source_inp,
        "leak_present": 1 if len(leaks) > 0 else 0,
        "scn_number": scn_number,
        "leaks": leaks,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -------------------------
# Main: Three-leak generator appended after scn_975
# -------------------------
def main():
    BASE_INP = Path("base.inp")

    # SAME folders you used for previous scenarios
    OUT_INP_DIR = Path("training_cases")
    OUT_RESULTS_DIR = Path("training_cases_output")

    # First 3-leak scenario should be scn_976
    START_SCN_NUMBER = 976

    # Safety: don't overwrite existing scenarios
    SKIP_EXISTING = True

    # Leak nodes
    LEAK_NODES = ["L1", "L2", "L3", "L4", "L5"]
    PIPE_TO_LEAKNODE = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}
    TOTAL_LENGTHS = {1: 100.0, 2: 500.0, 3: 200.0, 4: 200.0, 5: 200.0}

    # -------------------------
    # 3-leak design (900 scenarios)
    # Leak1: 5 positions x 2 sizes
    # Leak2: 3 positions x 1 size
    # Leak3: 3 positions x 1 size
    # Triplets: C(5,3)=10
    # Total = 10 * 10 * 3 * 3 = 900
    # -------------------------

    # Leak 1 (full positions, reduced sizes)
    R1_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
    SIZES1 = [("S", 0.01), ("M", 0.03)]  # 2 sizes

    # Leak 2 (subset positions, 1 size)
    R2_POSITIONS = [0.1, 0.5, 0.7]
    SIZES2 = [("S", 0.01)]  # 1 size

    # Leak 3 (subset positions, 1 size)
    R3_POSITIONS = [0.1, 0.5, 0.7]
    SIZES3 = [("S", 0.01)]  # 1 size

    if not BASE_INP.exists():
        raise RuntimeError("Cannot find base.inp in the current folder.")

    OUT_INP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_text = BASE_INP.read_text(encoding="utf-8", errors="ignore")
    base_times_block = extract_block(base_text, "[TIMES]")

    wn_base = wntr.network.WaterNetworkModel(str(BASE_INP))

    # Sanity check leak nodes exist
    for ln in LEAK_NODES:
        if ln not in wn_base.node_name_list:
            raise RuntimeError(f"Leak node '{ln}' not found in base.inp nodes.")

    # Count expected scenarios precisely:
    # pipe triplets = C(5,3)=10
    triplets = []
    for p1 in [1, 2, 3, 4, 5]:
        for p2 in range(p1 + 1, 6):
            for p3 in range(p2 + 1, 6):
                triplets.append((p1, p2, p3))

    total_three_leak = (
        len(triplets)
        * (len(R1_POSITIONS) * len(SIZES1))
        * (len(R2_POSITIONS) * len(SIZES2))
        * (len(R3_POSITIONS) * len(SIZES3))
    )

    scn_number = START_SCN_NUMBER
    last_scn = START_SCN_NUMBER + total_three_leak - 1

    print(f"Generating THREE-LEAK subset: {total_three_leak} scenarios", flush=True)
    print(f"Scenario numbers: scn_{START_SCN_NUMBER} ... scn_{last_scn}", flush=True)

    made = 0
    skipped = 0

    # Order:
    # Leak1: pipe1 -> r1 -> size1
    # Leak2: pipe2 -> r2 -> size2
    # Leak3: pipe3 -> r3 -> size3
    # with pipe1 < pipe2 < pipe3
    for (pipe1, pipe2, pipe3) in triplets:
        for r1 in R1_POSITIONS:
            for size1_level, c1 in SIZES1:
                leak_node_1 = PIPE_TO_LEAKNODE[pipe1]
                L1 = TOTAL_LENGTHS[pipe1]
                d1 = r1 * L1
                len1a, len1b = d1, (L1 - d1)
                prefer1 = (f"{pipe1}a", f"{pipe1}b")

                for r2 in R2_POSITIONS:
                    for size2_level, c2 in SIZES2:
                        leak_node_2 = PIPE_TO_LEAKNODE[pipe2]
                        L2 = TOTAL_LENGTHS[pipe2]
                        d2 = r2 * L2
                        len2a, len2b = d2, (L2 - d2)
                        prefer2 = (f"{pipe2}a", f"{pipe2}b")

                        for r3 in R3_POSITIONS:
                            for size3_level, c3 in SIZES3:
                                leak_node_3 = PIPE_TO_LEAKNODE[pipe3]
                                L3 = TOTAL_LENGTHS[pipe3]
                                d3 = r3 * L3
                                len3a, len3b = d3, (L3 - d3)
                                prefer3 = (f"{pipe3}a", f"{pipe3}b")

                                inp_name = f"scn_{scn_number}.inp"
                                out_inp_path = OUT_INP_DIR / inp_name
                                scn_out_dir = OUT_RESULTS_DIR / f"scn_{scn_number}"

                                if SKIP_EXISTING and (out_inp_path.exists() or scn_out_dir.exists()):
                                    skipped += 1
                                    scn_number += 1
                                    continue

                                # Build scenario
                                wn = copy.deepcopy(wn_base)
                                clear_emitters(wn, LEAK_NODES)

                                # Split lengths for each leak
                                l1a, l1b = pick_two_links_at_leaknode(wn, leak_node_1, prefer1)
                                set_link_lengths(wn, l1a, len1a, l1b, len1b)

                                l2a, l2b = pick_two_links_at_leaknode(wn, leak_node_2, prefer2)
                                set_link_lengths(wn, l2a, len2a, l2b, len2b)

                                l3a, l3b = pick_two_links_at_leaknode(wn, leak_node_3, prefer3)
                                set_link_lengths(wn, l3a, len3a, l3b, len3b)

                                # Set emitters
                                set_emitter(wn, leak_node_1, c1)
                                set_emitter(wn, leak_node_2, c2)
                                set_emitter(wn, leak_node_3, c3)

                                # Write INP
                                wntr.network.io.write_inpfile(wn, str(out_inp_path))

                                # Force TIMES to match base.inp
                                if base_times_block:
                                    gen_text = out_inp_path.read_text(encoding="utf-8", errors="ignore")
                                    new_text = replace_block(gen_text, "[TIMES]", base_times_block)
                                    out_inp_path.write_text(new_text, encoding="utf-8")

                                # Run simulation & save signals
                                run_sim_and_save_signals_filtered(wn, scn_out_dir)

                                # labels.json (three leaks) — kept in ascending pipe order
                                leaks = [
                                    {
                                        "pipe_id": int(pipe1),
                                        "position": float(r1),
                                        "size_level": str(size1_level),
                                        "emitter_coeff": float(c1),
                                    },
                                    {
                                        "pipe_id": int(pipe2),
                                        "position": float(r2),
                                        "size_level": str(size2_level),
                                        "emitter_coeff": float(c2),
                                    },
                                    {
                                        "pipe_id": int(pipe3),
                                        "position": float(r3),
                                        "size_level": str(size3_level),
                                        "emitter_coeff": float(c3),
                                    },
                                ]
                                write_labels_json(scn_out_dir / "labels.json", inp_name, scn_number, leaks)

                                made += 1
                                if made % 25 == 0:
                                    print(f"[OK] Generated {made}/{total_three_leak} three-leak scenarios...", flush=True)

                                scn_number += 1

    print("\nDone.", flush=True)
    print(f"Created: {made}", flush=True)
    print(f"Skipped (already existed): {skipped}", flush=True)
    print(f"INP folder: {OUT_INP_DIR.resolve()}", flush=True)
    print(f"Data folder: {OUT_RESULTS_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        Path("FAILED_run_log.txt").write_text(tb, encoding="utf-8")
        raise
