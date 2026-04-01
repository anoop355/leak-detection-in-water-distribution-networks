from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def generate_plots(
    predicted_measurements: pd.DataFrame,
    residuals: pd.DataFrame,
    state_estimates: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    _plot_measurement_comparison(
        predicted_measurements,
        "P4_measured",
        "P4_predicted",
        "Pressure at Node 4",
        "Pressure",
        output_dir / "pressure_node4_measured_vs_predicted.png",
    )
    _plot_measurement_comparison(
        predicted_measurements,
        "Q1a_measured",
        "Q1a_predicted",
        "Flow in Pipe 1a",
        "Flow",
        output_dir / "flow_1a_measured_vs_predicted.png",
    )
    _plot_measurement_comparison(
        predicted_measurements,
        "Q3a_measured",
        "Q3a_predicted",
        "Flow in Pipe 3a",
        "Flow",
        output_dir / "flow_3a_measured_vs_predicted.png",
    )
    _plot_series(
        residuals,
        ["residual_P4", "residual_Q1a", "residual_Q3a"],
        "Measurement Residuals",
        "Residual",
        output_dir / "measurement_residuals.png",
    )
    _plot_series(
        state_estimates,
        ["D2", "D3", "D4", "D5", "D6"],
        "Estimated Demands",
        "Demand",
        output_dir / "estimated_demands.png",
    )
    _plot_series(
        state_estimates,
        ["H2", "H3", "H4", "H5", "H6"],
        "Estimated Heads",
        "Head",
        output_dir / "estimated_heads.png",
    )
    if {"nr_P4", "nr_Q1a", "nr_Q3a"}.issubset(residuals.columns):
        _plot_series(
            residuals,
            ["nr_P4", "nr_Q1a", "nr_Q3a"],
            "Normalized Residuals",
            "Normalized residual",
            output_dir / "normalized_residuals.png",
        )


def _plot_measurement_comparison(
    df: pd.DataFrame,
    measured_column: str,
    predicted_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df[measured_column], label="Measured", linewidth=1.5)
    ax.plot(df["timestamp"], df[predicted_column], label="Predicted", linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_series(
    df: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for column in columns:
        ax.plot(df["timestamp"], df[column], label=column, linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(len(columns), 3))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
