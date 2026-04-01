import os
import json
import pandas as pd
import wntr


INPUT_DIR = "training_cases"
OUTPUT_DIR = "training_cases_output"

PRESSURE_NODES = ["2", "3", "4", "5", "6"]
FLOW_LINKS = ["1a", "2a", "3a", "4a", "5a"]

SIGNALS_COL_ORDER = ["t", "P2", "P3", "P4", "P5", "P6", "Q1a", "Q2a", "Q3a", "Q4a", "Q5a"]

# Scenario definition order
POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]
SIZES = ["S", "M", "L"]
EMITTER = {"S": 0.01, "M": 0.03, "L": 0.06}


def make_folder(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_inp_extract(inp_path: str) -> pd.DataFrame:
    """Run INP and extract t, pressures, flows into one DataFrame."""
    wn = wntr.network.WaterNetworkModel(inp_path)
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    P = results.node["pressure"]
    Q = results.link["flowrate"]

    out = pd.DataFrame()
    out["t"] = (P.index.to_numpy() / 60.0).astype(int)

    for n in PRESSURE_NODES:
        out[f"P{n}"] = P[n].to_numpy()

    for l in FLOW_LINKS:
        out[f"Q{l}"] = Q[l].to_numpy()

    out = out[SIGNALS_COL_ORDER]
    return out


def labels_from_scn_filename(name: str) -> dict:
 
    labels = {
        "source_inp": name,
        "leak_present": 0,
        "pipe_id": -1,
        "position": -1,
        "size_level": None,
        "emitter_coeff": 0.0,
        "scn_number": None,
    }

    lower = name.lower()
    if "no_leak" in lower or "noleak" in lower:
        return labels

    stem = os.path.splitext(name)[0].lower()
    if not stem.startswith("scn_"):

        return labels

    # Extract scenario number
    try:
        k = int(stem.replace("scn_", ""))  # 1..75
    except ValueError:
        return labels

    if not (1 <= k <= 75):
        # out of expected range, keep defaults
        labels["scn_number"] = k
        return labels

    # Map k -> (pipe_id, position, size)
    i = k - 1  # 0-based
    per_pipe = len(POSITIONS) * len(SIZES) 

    pipe_index = i // per_pipe               # 0..4
    within_pipe = i % per_pipe               # 0..14
    pos_index = within_pipe // len(SIZES)    # 0..4
    size_index = within_pipe % len(SIZES)    # 0..2

    pipe_id = pipe_index + 1
    position = POSITIONS[pos_index]
    size_level = SIZES[size_index]
    emitter_coeff = EMITTER[size_level]

    labels.update({
        "leak_present": 1,
        "pipe_id": pipe_id,
        "position": float(position),
        "size_level": size_level,
        "emitter_coeff": float(emitter_coeff),
        "scn_number": k,
    })

    return labels


def main():
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(
            f"Folder '{INPUT_DIR}' not found. Create it and put .inp files inside."
        )

    make_folder(OUTPUT_DIR)

    inp_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".inp")]
    if not inp_files:
        raise FileNotFoundError(f"No .inp files found in '{INPUT_DIR}'.")

    print(f"Found {len(inp_files)} INP files in '{INPUT_DIR}'.")

    # Sort like: no_leak first (if present), then scn_1..scn_75
    def sort_key(fname: str):
        s = os.path.splitext(fname)[0].lower()
        if "no_leak" in s or "noleak" in s:
            return (-1, 0)
        if s.startswith("scn_"):
            try:
                return (0, int(s.replace("scn_", "")))
            except ValueError:
                return (1, 10**9)
        return (1, 10**9)

    for idx, fname in enumerate(sorted(inp_files, key=sort_key), start=1):
        inp_path = os.path.join(INPUT_DIR, fname)
        scenario_name = os.path.splitext(fname)[0]
        out_folder = os.path.join(OUTPUT_DIR, scenario_name)
        make_folder(out_folder)

        print(f"\n[{idx}/{len(inp_files)}] Running: {fname}")

        signals = run_inp_extract(inp_path)

        # Robust save (helps if OneDrive is acting up)
        signals_path = os.path.join(out_folder, "signals.csv")
        tmp_path = signals_path + ".tmp"
        signals.to_csv(tmp_path, index=False)
        os.replace(tmp_path, signals_path)

        labels = labels_from_scn_filename(fname)
        with open(os.path.join(out_folder, "labels.json"), "w") as f:
            json.dump(labels, f, indent=2)

        print(f"[OK] Saved -> {out_folder}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
