"""Plot results for the all-animals persona vector experiment.

Usage:
    uv run python -m all_animals.plot_results
    uv run python -m all_animals.plot_results --plots_dir ../plots/all_animals
"""

import argparse
import ast
import csv
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
            row["target_count"] = int(row["target_count"])
            row["total_responses"] = int(row["total_responses"])
            rows.append(row)
    return rows


def get_peak_rate(rows: list[dict]) -> float:
    if not rows:
        return 0.0
    return max(r["target_animal_rate"] for r in rows)


def collect_results() -> dict:
    """Collect peak target_animal_rate for all (animal, strategy, seed) combos."""
    results = {}
    for animal in ANIMALS:
        results[animal] = {}
        for strategy in STRATEGIES:
            results[animal][strategy] = {}
            for seed in SEEDS:
                csv_path = EVAL_DIR / strategy / animal / f"seed{seed}.csv"
                if csv_path.exists():
                    rows = load_eval_csv(str(csv_path))
                    results[animal][strategy][seed] = {
                        "peak_rate": get_peak_rate(rows),
                        "rows": rows,
                    }
    return results


def plot_strategy_comparison(results: dict, save_path: str):
    """Bar chart: mean peak rate by strategy, averaged across animals and seeds."""
    strategy_means = {}
    strategy_stds = {}

    for strategy in STRATEGIES:
        all_peaks = []
        for animal in ANIMALS:
            for seed in SEEDS:
                if seed in results.get(animal, {}).get(strategy, {}):
                    all_peaks.append(results[animal][strategy][seed]["peak_rate"])
        if all_peaks:
            strategy_means[strategy] = np.mean(all_peaks)
            strategy_stds[strategy] = np.std(all_peaks) / np.sqrt(len(all_peaks))

    if not strategy_means:
        print("No results to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    strategies = list(strategy_means.keys())
    means = [strategy_means[s] for s in strategies]
    sems = [strategy_stds[s] for s in strategies]

    colors = ["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f"]
    bars = ax.bar(strategies, means, yerr=sems, capsize=5,
                  color=colors[:len(strategies)], edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Peak Target Animal Rate", fontsize=13)
    ax.set_title("Persona Vector Projection: Strategy Comparison\n(19 animals × 3 seeds)", fontsize=14)
    ax.set_ylim(0, max(means) * 1.3 if means else 0.1)
    ax.tick_params(labelsize=12)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{mean:.1%}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_per_animal_curves(results: dict, save_dir: str):
    """Per-animal line plots: target_animal_rate over training steps."""
    os.makedirs(save_dir, exist_ok=True)

    colors = {
        "top_proj": "#2ca02c",
        "bottom_proj": "#d62728",
        "random": "#1f77b4",
        "clean": "#7f7f7f",
    }

    for animal in ANIMALS:
        fig, ax = plt.subplots(figsize=(8, 5))

        for strategy in STRATEGIES:
            for seed in SEEDS:
                data = results.get(animal, {}).get(strategy, {}).get(seed)
                if data and data["rows"]:
                    rows = sorted(data["rows"], key=lambda r: r["step"])
                    steps = [r["step"] for r in rows]
                    rates = [r["target_animal_rate"] for r in rows]
                    alpha = 0.3 if seed > 0 else 1.0
                    label = strategy if seed == 0 else None
                    ax.plot(steps, rates, color=colors.get(strategy, "gray"),
                            alpha=alpha, label=label, linewidth=1.5)

        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("Target Animal Rate", fontsize=12)
        ax.set_title(f"{animal.capitalize()}: SL Rate by Strategy", fontsize=13)
        ax.legend(fontsize=10, loc="upper left")
        ax.tick_params(labelsize=11)

        save_path = os.path.join(save_dir, f"{animal}_curves.png")
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
    print(f"Found {n_found}/{len(ANIMALS) * len(STRATEGIES) * len(SEEDS)} eval results")

    plot_strategy_comparison(results, os.path.join(args.plots_dir, "strategy_comparison.png"))
    plot_per_animal_curves(results, os.path.join(args.plots_dir, "per_animal"))


if __name__ == "__main__":
    main()
