"""
Genetic Algorithm (GA) sensor placement optimisation for the ST-GCN
leak detection framework.

This script was written to find the best subset of sensors to use for leak detection 
when the full set of 10 sensors is not available.

Rather than training the full ST-GCN model for every possible sensor
combination (which would take far too long), I use a proxy fitness function
based on an influence matrix — a compact representation of how strongly
each sensor responds to each leak scenario. The GA then searches over sensor
subsets to find the ones that maximise coverage and distinguishability.

Updated from v1. Main changes:
- Added optional z-score normalisation of the influence matrix before the GA
  (NORMALISE_INFLUENCE_MATRIX flag). Each sensor column is normalised
  independently so that pressure and flow sensors are on the same scale.
  Without this, pressure sensors were dominating the fitness score due to
  magnitude differences relative to flow sensors.
- TAU updated from 0.02 to 0.25 to account for the normalised scale.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# SETTINGS

# Path to the folder containing training scenario subfolders (scn_1, scn_2, ...)
# Each subfolder must have signals.csv and labels.json
RESULTS_DIR = Path("training_cases_output")

# Output folder where all GA results will be saved
OUT_DIR     = Path("GA Results")

# The 10 sensor channels the GA is allowed to choose from
FEATURE_COLS = ["P2","P3","P4","P5","P6","Q1a","Q2a","Q3a","Q4a","Q5a"]

# Path to the no-leak baseline signals CSV — used to compute the deviation
# of each sensor under leak conditions relative to normal conditions
BASELINE_CSV = RESULTS_DIR / "no_leak" / "signals.csv"

# Method used to collapse a sensor's deviation time-series into a single
# influence scalar per scenario:
#   "mad"  = mean absolute deviation (recommended — robust and interpretable)
#   "rms"  = root mean square (more sensitive to large deviations)
#   "peak" = maximum absolute value (only captures the worst-case point)
INFLUENCE_METHOD = "mad"

# Whether to use only single-leak scenarios when building the influence matrix.
# Single-leak scenarios are cleaner for estimating per-pipe sensor responses
# because the signal from each leak is not mixed with other leaks.
SINGLE_LEAK_ONLY = True

# Whether to apply z-score normalisation column-wise before running the GA.
# This is important because pressure sensors (in metres) and flow sensors
# (in m³/s) have very different magnitudes — without normalisation, flow
# sensors can dominate the fitness score just due to scale, not because
# they are more informative.
NORMALISE_INFLUENCE_MATRIX = True

# Sensor budget levels to test — the GA runs once for each value of k,
# finding the best k-sensor layout at each budget level
BUDGETS = [8, 6, 4, 2, 1]

# How many of the best unique layouts to save for each budget level
TOP_K_LAYOUTS_TO_SAVE = 10

# GA search parameters — larger values give a more thorough search but take longer
GA_POPULATION  = 60   # number of candidate sensor layouts in each generation
GA_GENERATIONS = 60   # number of generations to evolve

# Fitness function parameters:
# TAU is the influence threshold — a sensor is considered to have "detected"
# a leak if its influence value exceeds this. After normalisation, values are
# in units of standard deviations, so TAU=0.25 means the sensor must show
# at least 0.25 standard deviations of deviation from its normal range.
# This was raised from 0.02 (v1) because the old threshold was too permissive
# and was counting near-zero noise as detections after normalisation.
TAU   = 0.25

# ALPHA controls the trade-off between coverage and distinguishability:
#   Higher ALPHA → optimise more for coverage (detecting more leaks)
#   Lower ALPHA  → optimise more for distinguishability (telling leaks apart)
# ALPHA=0.6 gives a slight preference for coverage.
ALPHA = 0.6

SEED = 42


# HELPERS

def set_seed(seed: int):
    """
    Sets the random seeds for Python's built-in random module and NumPy
    so the GA produces reproducible results across runs.
    """
    random.seed(seed)
    np.random.seed(seed)


def compute_entry(delta: np.ndarray, method: str = "mad") -> float:
    """
    Reduces a sensor's deviation time-series (leak signal minus baseline)
    down to a single number representing how strongly that sensor responded
    to the leak.

    This scalar is what gets stored in the influence matrix — one value
    per (sensor, scenario) pair.

    Arguments:
        delta  — 1D array of (leak - baseline) values across all timesteps
        method — aggregation method: "mad", "rms", or "peak"

    Returns a single float representing the sensor's response magnitude.
    """
    if method == "mad":
        return float(np.mean(np.abs(delta)))   # average magnitude of deviation
    if method == "rms":
        return float(np.sqrt(np.mean(delta ** 2)))  # root mean square deviation
    if method == "peak":
        return float(np.max(np.abs(delta)))    # largest single deviation
    raise ValueError(f"Unknown method: {method}")


def list_scn_dirs(results_dir: Path) -> List[Path]:
    """
    Returns a sorted list of scenario folders (those named scn_*) from
    the results directory. Sorted by scenario number so the influence
    matrix rows are in a consistent order.
    """
    return sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("scn_")],
        key=lambda p: int(p.name.split("_")[1])
    )


def load_labels(labels_path: Path) -> Dict:
    """
    Reads a labels.json file and returns it as a Python dictionary.
    The labels contain the ground truth leak configuration for each
    scenario (pipe ID, position, size).
    """
    return json.loads(labels_path.read_text(encoding="utf-8", errors="ignore"))


def extract_leaks(labels: Dict) -> List[Dict]:
    """
    Extracts the list of leak entries from a labels dictionary.

    The key name varies depending on which dataset generator was used —
    some use "Leaks" (capital L) and others use "leaks" (lowercase).
    This function handles both variants.
    """
    if "Leaks" in labels:
        return labels.get("Leaks", [])
    return labels.get("leaks", [])


# INFLUENCE MATRIX

def build_influence_matrix_single_baseline(
    results_dir: Path,
    baseline_df: pd.DataFrame,
    feature_cols: List[str],
    method: str,
    single_leak_only: bool
) -> pd.DataFrame:
    """
    Builds the influence matrix — a table with one row per leak scenario
    and one column per sensor, where each cell contains the scalar response
    of that sensor to that leak.

    The response is computed as compute_entry(leak_signal - baseline_signal),
    which measures how much each sensor's reading changed compared to normal
    (no-leak) conditions.

    This matrix is the core input to the GA. A sensor layout with high
    influence values across many scenarios and a wide spread of signatures
    (different sensors responding differently to different pipes) will be
    preferred by the fitness function.
    """
    rows     = []
    scn_dirs = list_scn_dirs(results_dir)

    for scn_dir in scn_dirs:
        leak_csv    = scn_dir / "signals.csv"
        labels_json = scn_dir / "labels.json"

        # Skip folders that are missing the required files
        if not leak_csv.exists() or not labels_json.exists():
            continue

        leak_df = pd.read_csv(leak_csv)
        labels  = load_labels(labels_json)
        leaks   = extract_leaks(labels)

        # Skip multi-leak scenarios if single_leak_only is set
        if single_leak_only and len(leaks) != 1:
            continue

        # Skip no-leak scenarios — they don't contribute to the influence matrix
        if len(leaks) == 0:
            continue

        # Check that all required sensor columns are present
        missing = [c for c in feature_cols if c not in leak_df.columns or c not in baseline_df.columns]
        if missing:
            raise ValueError(f"{scn_dir.name}: missing required columns: {missing}")

        # The leak and baseline signals must have the same number of timesteps
        # so the subtraction is valid
        if len(leak_df) != len(baseline_df):
            raise ValueError(
                f"{scn_dir.name}: leak signals length ({len(leak_df)}) != baseline length ({len(baseline_df)}). "
                f"Baseline and leak simulations must use the same time settings."
            )

        # Extract leak metadata for the row
        lk         = leaks[0]
        pipe_id    = int(lk.get("pipe_id", -1))
        position   = float(lk.get("position", np.nan))
        size       = lk.get("size_level", lk.get("size", "NA"))
        scn_number = int(labels.get("scn_number", scn_dir.name.split("_")[1]))

        row = {
            "scenario":   scn_dir.name,
            "scn_number": scn_number,
            "pipe_id":    pipe_id,
            "position":   position,
            "size":       str(size),
        }

        # Compute the influence scalar for each sensor
        for c in feature_cols:
            delta  = leak_df[c].to_numpy(np.float32) - baseline_df[c].to_numpy(np.float32)
            row[c] = compute_entry(delta, method=method)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("scn_number")


def zscore_normalise_influence_columns(df: pd.DataFrame, feature_cols: List[str]):
    """
    Applies z-score normalisation to each sensor column independently,
    so all sensors are on the same scale before the GA evaluates them.
    
    Returns the normalised DataFrame plus dictionaries of per-column
    mean and standard deviation values (needed to reproduce the results
    and stored in the output JSON for reference).
    """
    df_norm    = df.copy()
    mu_dict    = {}
    sigma_dict = {}

    for c in feature_cols:
        col   = df_norm[c].to_numpy(dtype=np.float32)
        mu    = float(np.mean(col))
        sigma = float(np.std(col))

        mu_dict[c]    = mu
        sigma_dict[c] = sigma

        if sigma < 1e-12:
            # Constant column — set to zero rather than dividing by near-zero sigma
            df_norm[c] = 0.0
        else:
            df_norm[c] = (col - mu) / sigma   # standard z-score formula

    return df_norm, mu_dict, sigma_dict


# GA: FITNESS + OPERATORS

def fitness(layout_idx: List[int], X: np.ndarray, tau: float, alpha: float) -> float:
    """
    Scores a candidate sensor layout using two components:

    Coverage (A): what fraction of leak scenarios does at least one
    selected sensor detect? A sensor "detects" a leak if its influence
    value exceeds the threshold tau. High coverage means the layout
    can detect more leaks.

    Distinguishability (B): how different are the sensor signatures
    across different leak scenarios? This is estimated by averaging
    the Euclidean distance between random pairs of scenario rows in
    the selected sensor subspace. High distinguishability means the
    model should be able to tell different leaks apart more easily.

    The final score combines both components:
        fitness = alpha * coverage + (1 - alpha) * distinguishability

    I use random sampling for distinguishability (up to 300 pairs) instead
    of computing all pairwise distances because the full pairwise computation
    would be slow for large influence matrices.

    Arguments:
        layout_idx — list of column indices representing the selected sensors
        X          — (scenarios, sensors) influence matrix
        tau        — influence threshold for detection
        alpha      — weight given to coverage vs distinguishability
    """
    # Extract the sub-matrix for the selected sensors only
    Xs = X[:, layout_idx]

    # Coverage: fraction of scenarios where at least one sensor exceeds tau
    seen     = (np.max(Xs, axis=1) > tau)
    coverage = float(np.mean(seen))

    # Distinguishability: average pairwise distance between scenario signatures
    if Xs.shape[0] < 2:
        dist = 0.0
    else:
        pairs = min(300, Xs.shape[0] * 10)
        dsum  = 0.0
        for _ in range(pairs):
            i = random.randrange(Xs.shape[0])
            j = random.randrange(Xs.shape[0])
            if i == j:
                continue   # skip if we accidentally sampled the same row twice
            dsum += float(np.linalg.norm(Xs[i] - Xs[j]))
        dist = dsum / max(1, pairs)

    return alpha * coverage + (1.0 - alpha) * dist


def random_layout(n_sensors: int, k: int) -> List[int]:
    """
    Generates a random initial sensor layout by shuffling the sensor
    indices and picking the first k. Used to create the starting population
    for the GA.
    """
    idx = list(range(n_sensors))
    random.shuffle(idx)
    return sorted(idx[:k])


def crossover(a: List[int], b: List[int], n_sensors: int, k: int) -> List[int]:
    """
    Creates a child layout by combining sensors from two parent layouts.

    The combined pool of sensors from both parents is shuffled and the
    first k are selected. If the combined pool is smaller than k (which
    can happen if the two parents share many sensors), extra sensors are
    drawn from the remaining options to fill the gap.

    The child inherits features from both parents, which helps the GA explore combinations
    that neither parent had on its own.
    """
    pool = list(set(a) | set(b))   # union of both parents' sensors
    if len(pool) < k:
        # Add extra sensors if the pool is not large enough
        pool += [i for i in range(n_sensors) if i not in pool]
    random.shuffle(pool)
    return sorted(pool[:k])


def mutate(layout: List[int], n_sensors: int, p: float = 0.3) -> List[int]:
    """
    Applies a small random mutation to a layout with probability p.

    When a mutation occurs, one sensor is randomly swapped out for a
    different one that is not currently in the layout. This introduces
    diversity into the population and prevents the GA from getting stuck
    in a local optimum where all layouts are too similar to each other.
    """
    layout = layout.copy()
    if random.random() < p:
        out        = random.choice(layout)
        candidates = [i for i in range(n_sensors) if i not in layout]
        inn        = random.choice(candidates)
        layout.remove(out)
        layout.append(inn)
        layout = sorted(layout)
    return layout


def ga_search(X: np.ndarray, k: int, pop: int, gens: int,
              tau: float, alpha: float, top_k: int) -> List[Tuple[List[int], float]]:
    """
    Runs the full Genetic Algorithm and returns the top unique sensor
    layouts of size k.

    The GA works as follows each generation:
        1. Score all current layouts using the fitness function
        2. Keep the top third as survivors
        3. Fill the rest of the population by crossing over random pairs
           of survivors and applying mutations
        4. Repeat for the specified number of generations

    After all generations, the population is ranked and the top_k
    unique layouts are returned. Duplicates are filtered out because
    the crossover and mutation operators can sometimes produce the
    same layout multiple times.
    """
    n_sensors = X.shape[1]

    # Generate the initial population of random layouts
    population = [random_layout(n_sensors, k) for _ in range(pop)]
    scores     = [fitness(ind, X, tau=tau, alpha=alpha) for ind in population]

    for _ in range(gens):
        # Rank all layouts and keep the top third as parents for the next generation
        ranked    = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        survivors = [ind for ind, _ in ranked[:max(2, pop // 3)]]

        # Build the next generation: keep survivors plus new children
        new_pop = survivors.copy()
        while len(new_pop) < pop:
            p1, p2 = random.sample(survivors, 2)
            child  = crossover(p1, p2, n_sensors, k)
            child  = mutate(child, n_sensors, p=0.3)
            new_pop.append(child)

        # Score the new population
        population = new_pop
        scores     = [fitness(ind, X, tau=tau, alpha=alpha) for ind in population]

    # Final ranking — extract top unique layouts
    ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)

    seen = set()
    uniq = []
    for lay, sc in ranked:
        key = tuple(lay)
        if key in seen:
            continue   # skip duplicates
        seen.add(key)
        uniq.append((lay, sc))
        if len(uniq) >= top_k:
            break

    return uniq


# MAIN

def main():
    set_seed(SEED)

    # Check that the required input folders and files exist before starting
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"RESULTS_DIR not found: {RESULTS_DIR.resolve()}")
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Baseline file not found: {BASELINE_CSV.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(BASELINE_CSV)

    # Step 1 — Build the raw influence matrix from all single-leak scenarios.
    # This is the most time-consuming step since it reads every scenario CSV.
    influence_df = build_influence_matrix_single_baseline(
        results_dir=RESULTS_DIR,
        baseline_df=baseline_df,
        feature_cols=FEATURE_COLS,
        method=INFLUENCE_METHOD,
        single_leak_only=SINGLE_LEAK_ONLY
    )

    # Save the raw matrix before any normalisation so it can be inspected later
    raw_influence_csv = OUT_DIR / "influence_matrix_raw.csv"
    influence_df.to_csv(raw_influence_csv, index=False)
    print(f"[OK] Raw influence matrix saved: {raw_influence_csv} (rows={len(influence_df)})")

    if len(influence_df) == 0:
        raise RuntimeError("Influence matrix is empty. Check labels keys and SINGLE_LEAK_ONLY setting.")

    # Step 2 — Optionally normalise the influence matrix column-wise.
    # This puts all sensors on the same scale before the GA evaluates them.
    if NORMALISE_INFLUENCE_MATRIX:
        influence_df_used, mu_dict, sigma_dict = zscore_normalise_influence_columns(
            influence_df, FEATURE_COLS
        )

        # Save the normalised matrix and the stats used to produce it
        norm_influence_csv = OUT_DIR / "influence_matrix_normalised.csv"
        influence_df_used.to_csv(norm_influence_csv, index=False)

        norm_stats_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "mu":      [mu_dict[c]    for c in FEATURE_COLS],
            "sigma":   [sigma_dict[c] for c in FEATURE_COLS],
        })
        norm_stats_csv = OUT_DIR / "influence_matrix_normalisation_stats.csv"
        norm_stats_df.to_csv(norm_stats_csv, index=False)

        print(f"[OK] Normalised influence matrix saved: {norm_influence_csv}")
        print(f"[OK] Normalisation stats saved: {norm_stats_csv}")
    else:
        # Skip normalisation — use the raw matrix directly
        influence_df_used  = influence_df
        mu_dict, sigma_dict = {}, {}

    # Step 3 — Run the GA for each sensor budget level.
    # The influence matrix is converted to a plain NumPy array for speed.
    X = influence_df_used[FEATURE_COLS].to_numpy(dtype=np.float32)

    # Collect all results for saving to CSV and JSON
    ga_rows    = []
    ga_summary = {
        "results_dir":                str(RESULTS_DIR),
        "baseline_csv":               str(BASELINE_CSV),
        "feature_cols":               FEATURE_COLS,
        "influence_method":           INFLUENCE_METHOD,
        "single_leak_only":           bool(SINGLE_LEAK_ONLY),
        "normalised_influence_matrix": bool(NORMALISE_INFLUENCE_MATRIX),
        "tau":                        float(TAU),
        "alpha":                      float(ALPHA),
        "seed":                       int(SEED),
        "budgets":                    {}
    }

    # Include normalisation stats in the summary for reproducibility
    if NORMALISE_INFLUENCE_MATRIX:
        ga_summary["normalisation"] = {
            "type":  "zscore_columnwise",
            "mu":    mu_dict,
            "sigma": sigma_dict
        }

    for k in BUDGETS:
        if k < 1 or k > len(FEATURE_COLS):
            print(f"[WARN] Skipping budget {k} (must be 1..{len(FEATURE_COLS)})")
            continue

        # Run the GA for this budget level and get the top layouts
        top_layouts = ga_search(
            X=X, k=k, pop=GA_POPULATION, gens=GA_GENERATIONS,
            tau=TAU, alpha=ALPHA, top_k=TOP_K_LAYOUTS_TO_SAVE
        )

        ga_summary["budgets"][str(k)] = []

        for rank, (layout_idx, score) in enumerate(top_layouts, start=1):
            # Convert sensor indices back to sensor names for readability
            layout_cols = [FEATURE_COLS[i] for i in layout_idx]
            layout_id   = f"B{k}_R{rank:02d}"   # e.g. B4_R01 = budget 4, rank 1

            ga_rows.append({
                "layout_id":    layout_id,
                "budget_k":     k,
                "rank":         rank,
                "fitness":      float(score),
                "feature_cols": ",".join(layout_cols),
                "feature_idx":  ",".join(map(str, layout_idx)),
            })

            ga_summary["budgets"][str(k)].append({
                "layout_id":   layout_id,
                "rank":        rank,
                "fitness":     float(score),
                "layout_cols": layout_cols,
                "layout_idx":  layout_idx
            })

        # Print the best result for this budget so progress is visible
        best = ga_summary["budgets"][str(k)][0] if ga_summary["budgets"][str(k)] else None
        if best:
            print(f"[OK] Budget {k}: best fitness={best['fitness']:.4f} layout={best['layout_cols']}")

    # Save all results to CSV and JSON
    ga_df = pd.DataFrame(ga_rows).sort_values(["budget_k", "rank"])
    ga_csv = OUT_DIR / "ga_best_layouts.csv"
    ga_df.to_csv(ga_csv, index=False)

    # JSON summary contains all the detail needed to reproduce or inspect any result
    ga_json_path = OUT_DIR / "ga_best_layouts.json"
    ga_json_path.write_text(json.dumps(ga_summary, indent=2), encoding="utf-8")

    print(f"[OK] GA layouts CSV saved: {ga_csv}")
    print(f"[OK] GA layouts JSON saved: {ga_json_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
