import copy
import json
import traceback
from pathlib import Path

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
# Demand scaling
# =========================
def apply_global_demand_scale(wn, scale: float):
    """
    Multiply every junction demand base value by the same factor.
    This updates the actual demand timeseries entries used by WNTR/EPANET.
    """
    scale = float(scale)

    for jname in wn.junction_name_list:
        j = wn.get_node(jname)

        # A junction can have one or more demand entries
        for ts in j.demand_timeseries_list:
            ts.base_value = float(ts.base_value) * scale


# =========================
# Run simulation and save signals
# =========================
def run_sim_and_save_signals_filtered(wn, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    pressures = results.node["pressure"]
    flows = results.link["flowrate"]

    t_sec = pressures.index.values.astype(float)
    t_min = t_sec / 60.0

    req_nodes = ["2", "3", "4", "5", "6"]
    req_links = ["1a", "2a", "3a", "4a", "5a"]

    df = pd.DataFrame()
    df["t"] = t_min

    for n in req_nodes:
        df[f"P{n}"] = pressures[n].values

    for l in req_links:
        df[f"Q{l}"] = flows[l].values

    df.to_csv(out_dir / "signals.csv", index=False)


# =========================
# Save labels
# =========================
def write_labels_json(out_path: Path, source_inp: str, scenario_name: str, scale: float):

    payload = {
        "scenario_name": scenario_name,
        "source_inp": source_inp,
        "Leak_present": 0,
        "scale_factor": float(scale),
        "Leaks": []
    }

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =========================
# Main
# =========================
def main():

    BASE_INP = Path("base2.inp")

    OUT_INP_DIR = Path("training_cases")
    OUT_RESULTS_DIR = Path("training_cases_output")

    DEMAND_SCALES = [0.8, 0.9, 1.1, 1.2]

    if not BASE_INP.exists():
        raise RuntimeError("base.inp not found")

    OUT_INP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_text = BASE_INP.read_text(encoding="utf-8", errors="ignore")
    base_times_block = extract_block(base_text, "[TIMES]")

    wn_base = wntr.network.WaterNetworkModel(str(BASE_INP))

    scenario_index = 5

    for scale in DEMAND_SCALES:

        scenario_name = f"no_leak_{scenario_index}"

        inp_name = f"{scenario_name}.inp"

        inp_path = OUT_INP_DIR / inp_name
        scn_out_dir = OUT_RESULTS_DIR / scenario_name

        wn = copy.deepcopy(wn_base)

        apply_global_demand_scale(wn, scale)

        wntr.network.io.write_inpfile(wn, str(inp_path))

        # Preserve TIMES block
        if base_times_block:
            gen_text = inp_path.read_text(encoding="utf-8", errors="ignore")
            new_text = replace_block(gen_text, "[TIMES]", base_times_block)
            inp_path.write_text(new_text, encoding="utf-8")

        run_sim_and_save_signals_filtered(wn, scn_out_dir)

        write_labels_json(
            out_path=scn_out_dir / "labels.json",
            source_inp=inp_name,
            scenario_name=scenario_name,
            scale=scale
        )

        print(f"[OK] Generated {scenario_name} (scale={scale})")

        scenario_index += 1

    print("\nAll no-leak scenarios generated successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        Path("FAILED_generate_no_leak_log.txt").write_text(tb)
        raise