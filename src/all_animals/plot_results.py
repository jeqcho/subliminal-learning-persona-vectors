"""Plot results for the all-animals persona vector experiment.

Usage:
    uv run python -m all_animals.plot_results
    uv run python -m all_animals.plot_results --plots_dir ../plots/all_animals
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from all_animals.config import ANIMALS, EVAL_DIR, SEEDS, STRATEGIES


def load_eval_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["step"] = int(row["step"])
            row["target_animal_rate"] = float(row["target_animal_rate"])
            rows.append(row)
    return rows


def collect_results() -> dict:
    """Collect eval data for all (animal, strategy, seed) combos."""
    results = {}
    for animal in ANIMALS:
        results[animal] = {}
        for strategy in STRATEGIES:
            results[animal][strategy] = {}
            for seed in SEEDS:
                csv_path = EVAL_DIR / strategy / animal / f"seed{seed}.csv"
                if csv_path.exists():
                    rows = load_eval_csv(str(csv_path))
                    if rows:
                        results[animal][strategy][seed] = rows
    return results


def plot_all_animals_grid(results: dict, save_path: str):
    """19-subplot grid: one per animal. Solid line = mean across seeds, shaded = std."""
    colors = {
        "top_proj": "#EE6677",
        "bottom_proj": "#228833",
        "random": "#1F77B4",
        "clean": "#7F7F7F",
    }
    legend_labels = {
        "top_proj": "Top PVP",
        "bottom_proj": "Bottom PVP",
        "random": "Random",
        "clean": "Clean",
    }

    n_animals = len(ANIMALS)
    ncols = 4
    nrows = 5

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for idx, animal in enumerate(ANIMALS):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        has_data = False
        for strategy in STRATEGIES:
            seed_data = results.get(animal, {}).get(strategy, {})
            if not seed_data:
                continue

            # Align steps across seeds
            all_steps = []
            all_rates = []
            for seed in SEEDS:
                if seed not in seed_data:
                    continue
                rows = sorted(seed_data[seed], key=lambda r: r["step"])
                all_steps.append([r["step"] for r in rows])
                all_rates.append([r["target_animal_rate"] for r in rows])

            if not all_rates:
                continue

            has_data = True

            # Use shortest length for alignment
            min_len = min(len(r) for r in all_rates)
            steps = all_steps[0][:min_len]
            rates_arr = np.array([r[:min_len] for r in all_rates])

            mean = rates_arr.mean(axis=0)
            std = rates_arr.std(axis=0)

            ax.plot(steps, mean, color=colors[strategy], linewidth=1.5, label=legend_labels[strategy])
            ax.fill_between(steps, mean - std, mean + std, color=colors[strategy], alpha=0.15)

        ax.set_title(animal.capitalize(), fontsize=12, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)

        if not has_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")

        if row == nrows - 1:
            ax.set_xlabel("Step", fontsize=10)
        if col == 0:
            ax.set_ylabel("Target Rate", fontsize=10)

    # Hide empty subplots (except last one, used for legend)
    for idx in range(n_animals, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].axis("off")

    # Place legend in the last (empty bottom-right) subplot
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        last_row, last_col = divmod(nrows * ncols - 1, ncols)
        legend_ax = axes[last_row][last_col]
        legend_ax.legend(handles, labels, loc="center", fontsize=11,
                         frameon=False, handlelength=2)

    fig.suptitle("Subliminal Learning Under Persona Vector Projection (PVP) Dataset Selection (Numbers Dataset)\n(solid = mean, shaded = ±1 std across 3 seeds)",
                 fontsize=14, y=1.06)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot all-animals experiment results")
    parser.add_argument("--plots_dir", type=str, default="../plots/all_animals")
    args = parser.parse_args()

    results = collect_results()

    n_found = sum(
        1 for a in ANIMALS for s in STRATEGIES for seed in SEEDS
        if seed in results.get(a, {}).get(s, {})
    )
    total = len(ANIMALS) * len(STRATEGIES) * len(SEEDS)
    print(f"Found {n_found}/{total} eval results")

    plot_all_animals_grid(results, os.path.join(args.plots_dir, "all_animals_grid.png"))


if __name__ == "__main__":
    main()
