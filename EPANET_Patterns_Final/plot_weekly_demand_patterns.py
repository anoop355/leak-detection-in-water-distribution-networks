from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sys


DATA_DIR = Path(__file__).resolve().parent
FILE_TEMPLATE = "{site_id}_weekly_flow_PATTERN.txt"
DEFAULT_SITE_IDS = ["site002", "site003", "site004", "site005", "site006"]


def load_pattern(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    if values.size != 10080:
        raise ValueError(f"{path.name} has {values.size} values; expected 10080.")
    return values


def parse_site_ids() -> list[str]:
    site_ids = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SITE_IDS
    if not site_ids:
        raise ValueError("Provide at least one site id.")
    return site_ids


def make_output_name(site_ids: list[str]) -> str:
    return f"{site_ids[0]}_to_{site_ids[-1]}_weekly_demand_patterns.png"


def legend_start_index(site_ids: list[str]) -> int:
    try:
        return int(site_ids[0].replace("site", "")) - 1
    except ValueError:
        return 1


def main() -> None:
    site_ids = parse_site_ids()
    start_index = legend_start_index(site_ids)
    patterns = {
        site_id: load_pattern(DATA_DIR / FILE_TEMPLATE.format(site_id=site_id))
        for site_id in site_ids
    }

    time_hours = np.arange(10080) / 60.0

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
        }
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    line_styles = [
        {"linestyle": "--", "linewidth": 2.4, "color": "#1f77b4"},
        {"linestyle": "-", "linewidth": 2.4, "color": "#d62728"},
        {"linestyle": ":", "linewidth": 2.8, "color": "#7f7f7f"},
        {"linestyle": "-.", "linewidth": 2.4, "color": "#2ca02c"},
        {"linestyle": ":", "linewidth": 2.4, "color": "#000000"},
    ]

    for idx, site_id in enumerate(site_ids, start=start_index):
        style = line_styles[(idx - start_index) % len(line_styles)]
        ax.plot(
            time_hours,
            patterns[site_id],
            label=f"Demand pattern {idx}",
            **style,
        )

    ax.set_xlim(0, 168)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Base demand multiplier")
    ax.set_xticks(np.arange(0, 169, 24))
    ax.set_xticklabels([str(int(tick)) for tick in np.arange(0, 169, 24)])
    ax.legend(loc="upper right", frameon=True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    output_path = DATA_DIR / make_output_name(site_ids)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
