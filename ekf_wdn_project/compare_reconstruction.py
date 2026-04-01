from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationConfig:
    truth_dir: Path = Path("truth_outputs/no-leak")
    reconstruction_dir: Path = Path("outputs")
    output_dir: Path = Path("validation_outputs")


VALIDATION_CONFIG = ValidationConfig()


def run_validation(config: ValidationConfig = VALIDATION_CONFIG) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = config.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    truth_heads = _load_csv(config.truth_dir / "truth_node_heads.csv")
    truth_pressures = _load_csv(config.truth_dir / "truth_node_pressures.csv")
    truth_demands = _load_csv(config.truth_dir / "truth_demands.csv")
    truth_flows = _load_csv(config.truth_dir / "truth_pipe_flows.csv")

    est_states = _load_csv(config.reconstruction_dir / "state_estimates.csv")
    est_heads = _load_csv(config.reconstruction_dir / "all_node_heads.csv")
    est_pressures = _load_csv(config.reconstruction_dir / "all_node_pressures.csv")
    est_flows = _load_csv(config.reconstruction_dir / "all_pipe_flows.csv")

    merged_heads = _merge_on_timestamp(truth_heads, est_heads, "truth", "est")
    merged_pressures = _merge_on_timestamp(truth_pressures, est_pressures, "truth", "est")
    merged_demands = _merge_on_timestamp(truth_demands, est_states, "truth", "est")
    merged_flows = _merge_on_timestamp(truth_flows, est_flows, "truth", "est")

    metrics = pd.concat(
        [
            _compute_metrics(merged_heads, "head"),
            _compute_metrics(merged_pressures, "pressure"),
            _compute_metrics(merged_demands, "demand"),
            _compute_metrics(merged_flows, "flow"),
        ],
        ignore_index=True,
    )
    summary = (
        metrics.groupby("group", as_index=False)[["mae", "rmse", "r2"]]
        .mean()
        .sort_values("group")
        .reset_index(drop=True)
    )

    metrics.to_csv(config.output_dir / "metrics_by_variable.csv", index=False)
    summary.to_csv(config.output_dir / "metrics_summary.csv", index=False)

    _plot_group_overlay(
        merged_demands,
        ["D2", "D3", "D4", "D5", "D6"],
        "Demand Truth vs Reconstructed",
        plots_dir / "demands_truth_vs_reconstructed.png",
    )
    _plot_group_overlay(
        merged_heads,
        ["2", "3", "4", "5", "6"],
        "Head Truth vs Reconstructed",
        plots_dir / "heads_truth_vs_reconstructed.png",
    )
    _plot_group_overlay(
        merged_flows,
        ["1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b", "5a", "5b"],
        "Pipe Flow Truth vs Reconstructed",
        plots_dir / "flows_truth_vs_reconstructed.png",
    )
    _plot_error_series(
        merged_heads,
        ["2", "3", "4", "5", "6"],
        "Head Reconstruction Error",
        plots_dir / "head_errors_over_time.png",
    )
    _plot_error_series(
        merged_demands,
        ["D2", "D3", "D4", "D5", "D6"],
        "Demand Reconstruction Error",
        plots_dir / "demand_errors_over_time.png",
    )
    _plot_error_series(
        merged_flows,
        ["1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b", "5a", "5b"],
        "Flow Reconstruction Error",
        plots_dir / "flow_errors_over_time.png",
    )

    return {"metrics": metrics, "summary": summary}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _merge_on_timestamp(
    truth_df: pd.DataFrame,
    est_df: pd.DataFrame,
    truth_suffix: str,
    est_suffix: str,
) -> pd.DataFrame:
    return truth_df.merge(est_df, on="timestamp", suffixes=(f"_{truth_suffix}", f"_{est_suffix}"))


def _compute_metrics(merged_df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    metrics_rows: list[dict[str, float | str]] = []
    truth_columns = [column for column in merged_df.columns if column.endswith("_truth")]
    for truth_column in truth_columns:
        base_name = truth_column[: -len("_truth")]
        est_column = f"{base_name}_est"
        truth = merged_df[truth_column].to_numpy(dtype=float)
        est = merged_df[est_column].to_numpy(dtype=float)
        error = est - truth
        ss_res = float(np.sum(error**2))
        ss_tot = float(np.sum((truth - truth.mean()) ** 2))
        r2 = np.nan if ss_tot <= 0.0 else 1.0 - (ss_res / ss_tot)
        metrics_rows.append(
            {
                "group": group_name,
                "variable": base_name,
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "r2": r2,
            }
        )
    return pd.DataFrame(metrics_rows)


def _plot_group_overlay(
    merged_df: pd.DataFrame,
    base_columns: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(base_columns), 1, figsize=(11, max(8, 2.4 * len(base_columns))), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, base_column in zip(axes, base_columns):
        axis.plot(merged_df["timestamp"], merged_df[f"{base_column}_truth"], label="Truth", linewidth=1.4)
        axis.plot(merged_df["timestamp"], merged_df[f"{base_column}_est"], label="Reconstructed", linewidth=1.2)
        axis.set_ylabel(base_column)
        axis.grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Timestamp")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_error_series(
    merged_df: pd.DataFrame,
    base_columns: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for base_column in base_columns:
        error = merged_df[f"{base_column}_est"] - merged_df[f"{base_column}_truth"]
        ax.plot(merged_df["timestamp"], error, label=base_column, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Error")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(5, len(base_columns)))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    outputs = run_validation()
    print(f"Saved validation outputs to {VALIDATION_CONFIG.output_dir}")
    print(outputs["summary"].to_string(index=False))
