import copy
import json
import traceback
from pathlib import Path
from itertools import combinations

import pandas as pd
import wntr


# =========================
# INP block utilities
# =========================
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


# =========================
# Network helpers
# =========================
def get_connected_links(wn, node_name: str):
    return list(wn.get_links_for_node(node_name))


def pick_two_links_at_leaknode(wn, leak_node: str, prefer_tokens):
    """
    Assumes the leak node connects to exactly 2 links, and tries to choose (pipeXa, pipeXb).
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
    Matches your original logic: if inp units are LPS, divide by 1000.
    """
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


def apply_global_demand_scale(wn, scale: float):
    """
    Adds non-random operating condition variation.
    Multiplies base_demand for all junctions by a fixed scale factor.
    """
    scale = float(scale)
    for jname in wn.junction_name_list:
        j = wn.get_node(jname)
        # In WNTR, junction.base_demand may be a float or list-like depending on how it was built.
        # Most EPANET INPs -> float base_demand.
        try:
            j.base_demand = float(j.base_demand) * scale
        except Exception:
            # If it's not directly float-castable, we leave it unchanged (safe fallback).
            pass


# =========================
# Simulation output formatting (matches your screenshot)
# =========================
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
    """
    Matches the screenshot structure:
      source_inp
      Leak_present
      scn_number
      Leaks (capital L)
    """
    payload = {
        "source_inp": source_inp,
        "Leak_present": 1 if len(leaks) > 0 else 0,
        "scn_number": scn_number,
        "Leaks": leaks,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =========================
# Structured test plan (494 scenarios)
# =========================
def build_test_plan():
    """
    Returns a list of dict entries describing each scenario to generate.
    This is deterministic and structured (non-random).
    Total scenarios = 494.
    """
    # --- Leak sizes (locked to your request)
    SIZE_LEVELS = {
        "S": 0.02,
        "M": 0.05,
        "L": 0.12,
    }

    # --- Operating condition variation (global demand scales)
    # Used to generate multiple conditions per leak configuration
    # (replaces "random seeds" with deterministic operating points)
    SCALES_A0 = [0.8, 0.9, 1.0, 1.1, 1.2]  # for no-leak baseline
    SCALES_MAIN = [0.9, 1.1]               # for leak scenarios (keeps count manageable)

    # --- Pipes in your network: 1..5
    PIPES = [1, 2, 3, 4, 5]

    # --- Positions
    POS_SINGLE = [0.2, 0.4, 0.6, 0.8]

    # 2-leak position combos (structured, interpretable)
    POS_2LEAK = [
        (0.2, 0.8),  # easy
        (0.4, 0.6),  # medium
        (0.6, 0.6),  # same-ish region effect
    ]

    # 3-leak position patterns
    POS_3LEAK = [
        (0.2, 0.6, 0.8),  # mixed
        (0.4, 0.4, 0.4),  # same region
        (0.8, 0.8, 0.2),  # clustered-ish
    ]

    # --- Emitter patterns
    EMIT_2LEAK = [
        ("S", "S"),
        ("M", "M"),
        ("L", "S"),
    ]

    EMIT_3LEAK = [
        ("S", "S", "S"),
        ("M", "M", "M"),
        ("L", "S", "S"),
    ]

    plan = []

    # Group A0: 0 leaks (50 scenarios)
    # 5 scales * 10 repeats = 50
    rep = 0
    for scale in SCALES_A0:
        for k in range(10):
            rep += 1
            plan.append({
                "group": "A0_no_leak",
                "n_leaks": 0,
                "scale": scale,
                "leaks": [],
                "tag": f"baseline_rep{rep:02d}",
            })

    # Group B1: 1 leak (120 scenarios)
    # 5 pipes * 4 positions * 3 sizes * 2 scales = 120
    for pipe in PIPES:
        for pos in POS_SINGLE:
            for size_level, coeff in SIZE_LEVELS.items():
                for scale in SCALES_MAIN:
                    plan.append({
                        "group": "B1_one_leak",
                        "n_leaks": 1,
                        "scale": scale,
                        "leaks": [
                            {"pipe_id": pipe, "position": pos, "size_level": size_level, "emitter_coeff": coeff}
                        ],
                        "tag": f"p{pipe}_pos{pos}_sz{size_level}_sc{scale}",
                    })

    # Group C2: 2 leaks (180 scenarios)
    # all 10 pairs * 3 pos patterns * 3 size patterns * 2 scales = 180
    pairs = list(combinations(PIPES, 2))  # C(5,2)=10
    for (p1, p2) in pairs:
        for (pos1, pos2) in POS_2LEAK:
            for (s1, s2) in EMIT_2LEAK:
                for scale in SCALES_MAIN:
                    plan.append({
                        "group": "C2_two_leaks",
                        "n_leaks": 2,
                        "scale": scale,
                        "leaks": [
                            {"pipe_id": p1, "position": pos1, "size_level": s1, "emitter_coeff": SIZE_LEVELS[s1]},
                            {"pipe_id": p2, "position": pos2, "size_level": s2, "emitter_coeff": SIZE_LEVELS[s2]},
                        ],
                        "tag": f"pair{p1}-{p2}_pos{pos1}-{pos2}_sz{s1}-{s2}_sc{scale}",
                    })

    # Group D3: 3 leaks (144 scenarios)
    # choose 8 triplets out of 10, deterministic subset
    # 8 triplets * 3 pos patterns * 3 size patterns * 2 scales = 144
    all_triplets = list(combinations(PIPES, 3))  # C(5,3)=10

    # deterministic subset of 8 triplets (covers variety)
    # (1,2,3), (1,2,4), (1,2,5), (1,3,4), (1,3,5), (1,4,5), (2,3,4), (3,4,5)
    chosen_triplets = [
        (1, 2, 3),
        (1, 2, 4),
        (1, 2, 5),
        (1, 3, 4),
        (1, 3, 5),
        (1, 4, 5),
        (2, 3, 4),
        (3, 4, 5),
    ]

    for (p1, p2, p3) in chosen_triplets:
        for (pos1, pos2, pos3) in POS_3LEAK:
            for (s1, s2, s3) in EMIT_3LEAK:
                for scale in SCALES_MAIN:
                    plan.append({
                        "group": "D3_three_leaks",
                        "n_leaks": 3,
                        "scale": scale,
                        "leaks": [
                            {"pipe_id": p1, "position": pos1, "size_level": s1, "emitter_coeff": SIZE_LEVELS[s1]},
                            {"pipe_id": p2, "position": pos2, "size_level": s2, "emitter_coeff": SIZE_LEVELS[s2]},
                            {"pipe_id": p3, "position": pos3, "size_level": s3, "emitter_coeff": SIZE_LEVELS[s3]},
                        ],
                        "tag": f"tri{p1}-{p2}-{p3}_pos{pos1}-{pos2}-{pos3}_sz{s1}-{s2}-{s3}_sc{scale}",
                    })

    if len(plan) != 494:
        raise RuntimeError(f"Test plan size mismatch. Expected 494, got {len(plan)}")

    return plan


# =========================
# Main generator
# =========================
def main():
    BASE_INP = Path("base2.inp")

    OUT_INP_DIR = Path("test_data_inp")
    OUT_RESULTS_DIR = Path("test_data_results")

    START_SCN_NUMBER = 1
    SKIP_EXISTING = True

    # Leak nodes / pipe mapping (same assumptions as your training generator)
    LEAK_NODES = ["L1", "L2", "L3", "L4", "L5"]
    PIPE_TO_LEAKNODE = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5"}
    TOTAL_LENGTHS = {1: 100.0, 2: 500.0, 3: 200.0, 4: 200.0, 5: 200.0}

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

    plan = build_test_plan()

    print(f"Generating structured TEST set: {len(plan)} scenarios", flush=True)
    print(f"INP folder: {OUT_INP_DIR.resolve()}", flush=True)
    print(f"Data folder: {OUT_RESULTS_DIR.resolve()}", flush=True)

    manifest_rows = []
    made = 0
    skipped = 0

    scn_number = START_SCN_NUMBER

    for item in plan:
        inp_name = f"scn_{scn_number}.inp"
        out_inp_path = OUT_INP_DIR / inp_name
        scn_out_dir = OUT_RESULTS_DIR / f"scn_{scn_number}"

        if SKIP_EXISTING and (out_inp_path.exists() or scn_out_dir.exists()):
            skipped += 1
            # still record in manifest as skipped
            manifest_rows.append({
                "scn_number": scn_number,
                "inp_name": inp_name,
                "group": item["group"],
                "scale": item["scale"],
                "n_leaks": item["n_leaks"],
                "tag": item["tag"],
                "status": "skipped_exists",
            })
            scn_number += 1
            continue

        wn = copy.deepcopy(wn_base)

        # Operating condition variation
        apply_global_demand_scale(wn, item["scale"])

        # Always reset emitters before applying leaks
        clear_emitters(wn, LEAK_NODES)

        # Apply each leak (pipe length split + emitter)
        leaks_out = []
        for lk in item["leaks"]:
            pipe_id = int(lk["pipe_id"])
            r = float(lk["position"])
            size_level = str(lk["size_level"])
            c = float(lk["emitter_coeff"])

            leak_node = PIPE_TO_LEAKNODE[pipe_id]
            L = float(TOTAL_LENGTHS[pipe_id])

            # enforce safe placement (avoid endpoints)
            r = max(0.05, min(0.95, r))

            d = r * L
            len_a, len_b = d, (L - d)

            prefer = (f"{pipe_id}a", f"{pipe_id}b")
            la, lb = pick_two_links_at_leaknode(wn, leak_node, prefer)
            set_link_lengths(wn, la, len_a, lb, len_b)

            set_emitter(wn, leak_node, c)

            leaks_out.append({
                "pipe_id": pipe_id,
                "position": r,
                "size_level": size_level,
                "emitter_coeff": c,
            })

        # Write INP
        wntr.network.io.write_inpfile(wn, str(out_inp_path))

        # Force TIMES to match base.inp
        if base_times_block:
            gen_text = out_inp_path.read_text(encoding="utf-8", errors="ignore")
            new_text = replace_block(gen_text, "[TIMES]", base_times_block)
            out_inp_path.write_text(new_text, encoding="utf-8")

        # Run simulation & save signals.csv
        run_sim_and_save_signals_filtered(wn, scn_out_dir)

        # Save labels.json in screenshot format
        write_labels_json(scn_out_dir / "labels.json", inp_name, scn_number, leaks_out)

        made += 1
        if made % 25 == 0:
            print(f"[OK] Generated {made}/{len(plan)} test scenarios...", flush=True)

        manifest_rows.append({
            "scn_number": scn_number,
            "inp_name": inp_name,
            "group": item["group"],
            "scale": item["scale"],
            "n_leaks": item["n_leaks"],
            "tag": item["tag"],
            "status": "generated",
        })

        scn_number += 1

    # Write manifest
    manifest_path = OUT_RESULTS_DIR / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print("\nDone.", flush=True)
    print(f"Created: {made}", flush=True)
    print(f"Skipped (already existed): {skipped}", flush=True)
    print(f"Manifest: {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        Path("FAILED_generate_test_set_log.txt").write_text(tb, encoding="utf-8")
        raise