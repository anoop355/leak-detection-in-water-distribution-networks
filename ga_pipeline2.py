"""
Updated from v1. Main changes:
- Added optional z-score normalisation of the influence matrix before the GA
  (NORMALISE_INFLUENCE_MATRIX flag). Each sensor column is normalised
  independently so that pressure and flow sensors are on the same scale.
  Without this, flow sensors were dominating the fitness score due to
  magnitude differences relative to pressure sensors.
- TAU updated from 0.02 to 0.25 to account for the normalised scale —
  the old threshold was too low and was counting near-zero responses as detections.
- Raw matrix is now saved separately as influence_matrix_raw.csv; the
  normalised version (if enabled) is saved as influence_matrix_normalised.csv
  alongside a stats file with per-column mu and sigma.
- Normalisation parameters are also stored in the JSON summary so results
  can be reproduced.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


# SETTINGS

RESULTS_DIR = Path("training_cases_output")
OUT_DIR     = Path("GA Results")

FEATURE_COLS = ["P2","P3","P4","P5","P6","Q1a","Q2a","Q3a","Q4a","Q5a"]

BASELINE_CSV = RESULTS_DIR / "no_leak" / "signals.csv"

INFLUENCE_METHOD = "mad"
SINGLE_LEAK_ONLY = True

# apply z-score normalisation column-wise before running the GA
NORMALISE_INFLUENCE_MATRIX = True

BUDGETS = [8, 6, 4, 2, 1]

TOP_K_LAYOUTS_TO_SAVE = 10
GA_POPULATION  = 60
GA_GENERATIONS = 60

# TAU raised from 0.02 to 0.25 to suit the normalised scale
TAU   = 0.25
ALPHA = 0.6

SEED = 42

# HELPERS

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def compute_entry(delta: np.ndarray, method: str = "mad") -> float:
    """Reduce a delta time-series to a single influence scalar."""
    if method == "mad":
        return float(np.mean(np.abs(delta)))
    if method == "rms":
        return float(np.sqrt(np.mean(delta ** 2)))
    if method == "peak":
        return float(np.max(np.abs(delta)))
    raise ValueError(f"Unknown method: {method}")


def list_scn_dirs(results_dir: Path) -> List[Path]:
    """Return scn_* folders sorted by scenario number."""
    return sorted(
        [p for p in results_dir.iterdir() if p.is_dir() and p.name.startswith("scn_")],
        key=lambda p: int(p.name.split("_")[1])
    )


def load_labels(labels_path: Path) -> Dict:
    return json.loads(labels_path.read_text(encoding="utf-8", errors="ignore"))


def extract_leaks(labels: Dict) -> List[Dict]:
    # labels.json uses "Leaks" or "leaks" depending on which generator was used
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
    Build a (scenarios x sensors) influence matrix.
    Each entry is the scalar response of one sensor to one leak scenario,
    computed as compute_entry(leak_signal - baseline_signal).
    """
    rows = []
    scn_dirs = list_scn_dirs(results_dir)

    for scn_dir in scn_dirs:
        leak_csv    = scn_dir / "signals.csv"
        labels_json = scn_dir / "labels.json"
        if not leak_csv.exists() or not labels_json.exists():
            continue

        leak_df = pd.read_csv(leak_csv)
        labels  = load_labels(labels_json)
        leaks   = extract_leaks(labels)

        if single_leak_only and len(leaks) != 1:
            continue
        if len(leaks) == 0:
            continue

        missing = [c for c in feature_cols if c not in leak_df.columns or c not in baseline_df.columns]
        if missing:
            raise ValueError(f"{scn_dir.name}: missing required columns: {missing}")

        if len(leak_df) != len(baseline_df):
            raise ValueError(
                f"{scn_dir.name}: leak signals length ({len(leak_df)}) != baseline length ({len(baseline_df)}). "
                f"Baseline and leak simulations must use the same time settings."
            )

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

        for c in feature_cols:
            delta  = leak_df[c].to_numpy(np.float32) - baseline_df[c].to_numpy(np.float32)
            row[c] = compute_entry(delta, method=method)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("scn_number")


def zscore_normalise_influence_columns(df: pd.DataFrame, feature_cols: List[str]):
    """
    Z-score normalise each sensor column independently.
    Returns the normalised DataFrame plus per-column mu and sigma dicts.
    Zero-variance columns are set to 0 rather than dividing by near-zero sigma.
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
            df_norm[c] = 0.0
        else:
            df_norm[c] = (col - mu) / sigma

    return df_norm, mu_dict, sigma_dict


# GA: FITNESS + OPERATORS

def fitness(layout_idx: List[int], X: np.ndarray, tau: float, alpha: float) -> float:
    """
    Score a sensor layout.

    Coverage (A): fraction of leak scenarios where at least one selected sensor
    exceeds the influence threshold tau.

    Distinguishability (B): average pairwise distance between scenario signatures,
    approximated over random pairs.

    fitness = alpha * A + (1 - alpha) * B
    """
    Xs = X[:, layout_idx]

    seen     = (np.max(Xs, axis=1) > tau)
    coverage = float(np.mean(seen))

    if Xs.shape[0] < 2:
        dist = 0.0
    else:
        pairs = min(300, Xs.shape[0] * 10)
        dsum  = 0.0
        for _ in range(pairs):
            i = random.randrange(Xs.shape[0])
            j = random.randrange(Xs.shape[0])
            if i == j:
                continue
            dsum += float(np.linalg.norm(Xs[i] - Xs[j]))
        dist = dsum / max(1, pairs)

    return alpha * coverage + (1.0 - alpha) * dist


def random_layout(n_sensors: int, k: int) -> List[int]:
    idx = list(range(n_sensors))
    random.shuffle(idx)
    return sorted(idx[:k])


def crossover(a: List[int], b: List[int], n_sensors: int, k: int) -> List[int]:
    pool = list(set(a) | set(b))
    if len(pool) < k:
        pool += [i for i in range(n_sensors) if i not in pool]
    random.shuffle(pool)
    return sorted(pool[:k])


def mutate(layout: List[int], n_sensors: int, p: float = 0.3) -> List[int]:
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
    """Run GA and return the top_k unique layouts of size k."""
    n_sensors  = X.shape[1]
    population = [random_layout(n_sensors, k) for _ in range(pop)]
    scores     = [fitness(ind, X, tau=tau, alpha=alpha) for ind in population]

    for _ in range(gens):
        ranked    = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
        survivors = [ind for ind, _ in ranked[:max(2, pop // 3)]]

        new_pop = survivors.copy()
        while len(new_pop) < pop:
            p1, p2 = random.sample(survivors, 2)
            child  = crossover(p1, p2, n_sensors, k)
            child  = mutate(child, n_sensors, p=0.3)
            new_pop.append(child)

        population = new_pop
        scores     = [fitness(ind, X, tau=tau, alpha=alpha) for ind in population]

    ranked = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)

    seen = set()
    uniq = []
    for lay, sc in ranked:
        key = tuple(lay)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((lay, sc))
        if len(uniq) >= top_k:
            break

    return uniq


# MAIN

def main():
    set_seed(SEED)

    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"RESULTS_DIR not found: {RESULTS_DIR.resolve()}")
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Baseline file not found: {BASELINE_CSV.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv(BASELINE_CSV)

    # step 1: build raw influence matrix
    influence_df = build_influence_matrix_single_baseline(
        results_dir=RESULTS_DIR,
        baseline_df=baseline_df,
        feature_cols=FEATURE_COLS,
        method=INFLUENCE_METHOD,
        single_leak_only=SINGLE_LEAK_ONLY
    )

    raw_influence_csv = OUT_DIR / "influence_matrix_raw.csv"
    influence_df.to_csv(raw_influence_csv, index=False)
    print(f"[OK] Raw influence matrix saved: {raw_influence_csv} (rows={len(influence_df)})")

    if len(influence_df) == 0:
        raise RuntimeError("Influence matrix is empty. Check labels keys and SINGLE_LEAK_ONLY setting.")

    # step 2: optionally normalise column-wise before GA
    if NORMALISE_INFLUENCE_MATRIX:
        influence_df_used, mu_dict, sigma_dict = zscore_normalise_influence_columns(
            influence_df, FEATURE_COLS
        )

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
        influence_df_used  = influence_df
        mu_dict, sigma_dict = {}, {}

    # step 3: run GA per budget
    X = influence_df_used[FEATURE_COLS].to_numpy(dtype=np.float32)

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

        top_layouts = ga_search(
            X=X, k=k, pop=GA_POPULATION, gens=GA_GENERATIONS,
            tau=TAU, alpha=ALPHA, top_k=TOP_K_LAYOUTS_TO_SAVE
        )

        ga_summary["budgets"][str(k)] = []

        for rank, (layout_idx, score) in enumerate(top_layouts, start=1):
            layout_cols = [FEATURE_COLS[i] for i in layout_idx]
            layout_id   = f"B{k}_R{rank:02d}"

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

        best = ga_summary["budgets"][str(k)][0] if ga_summary["budgets"][str(k)] else None
        if best:
            print(f"[OK] Budget {k}: best fitness={best['fitness']:.4f} layout={best['layout_cols']}")

    ga_df = pd.DataFrame(ga_rows).sort_values(["budget_k", "rank"])
    ga_csv = OUT_DIR / "ga_best_layouts.csv"
    ga_df.to_csv(ga_csv, index=False)

    ga_json_path = OUT_DIR / "ga_best_layouts.json"
    ga_json_path.write_text(json.dumps(ga_summary, indent=2), encoding="utf-8")

    print(f"[OK] GA layouts CSV saved: {ga_csv}")
    print(f"[OK] GA layouts JSON saved: {ga_json_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()