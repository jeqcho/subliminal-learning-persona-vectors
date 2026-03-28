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

    fig.suptitle("Subliminal Learning Under Persona Vector Projection Dataset Selection (Numbers Dataset)\n(solid = mean, shaded = ±1 std across 3 seeds)",
                 fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def wilson_ci(successes, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0, center - half), min(1, center + half)


def plot_bar(results: dict, save_path: str):
    """Bar plot: one group per animal, 5 bars (base, clean, bottom, random, top), Wilson 95% CI."""
    bar_order = ["base", "clean", "bottom_proj", "random", "top_proj"]
    colors = {
        "base": "#BFBFBF",
        "clean": "#7F7F7F",
        "bottom_proj": "#228833",
        "random": "#1F77B4",
        "top_proj": "#EE6677",
    }
    bar_labels = {
        "base": "Base",
        "clean": "Clean",
        "bottom_proj": "Bottom PVP",
        "random": "Random",
        "top_proj": "Top PVP",
    }

    # Collect per-animal, per-bar: pool successes/total across seeds
    animal_data = {}
    for animal in ANIMALS:
        animal_data[animal] = {}
        for strategy in STRATEGIES:
            seed_data = results.get(animal, {}).get(strategy, {})
            # Last checkpoint
            total_successes, total_n = 0, 0
            for seed in SEEDS:
                if seed not in seed_data:
                    continue
                rows = sorted(seed_data[seed], key=lambda r: r["step"])
                if not rows:
                    continue
                last = rows[-1]
                n = int(last.get("total_responses", 5000))
                k = int(last.get("target_count", round(last["target_animal_rate"] * n)))
                total_successes += k
                total_n += n
            animal_data[animal][strategy] = (total_successes, total_n)

            # Base (step 0) — same across strategies, but grab from this one
            if "base" not in animal_data[animal]:
                base_s, base_n = 0, 0
                for seed in SEEDS:
                    if seed not in seed_data:
                        continue
                    rows = sorted(seed_data[seed], key=lambda r: r["step"])
                    if rows and rows[0]["step"] == 0:
                        first = rows[0]
                        n = int(first.get("total_responses", 5000))
                        k = int(first.get("target_count", round(first["target_animal_rate"] * n)))
                        base_s += k
                        base_n += n
                animal_data[animal]["base"] = (base_s, base_n)

    # Sort animals descending by random bar rate
    def random_rate(animal):
        s, n = animal_data[animal].get("random", (0, 0))
        return s / n if n > 0 else 0.0
    sorted_animals = sorted(ANIMALS, key=random_rate, reverse=True)

    n_bars = len(bar_order)
    x = np.arange(len(sorted_animals))
    width = 0.15

    fig, ax = plt.subplots(figsize=(18, 6))

    for i, bar_key in enumerate(bar_order):
        rates, lows, highs = [], [], []
        for animal in sorted_animals:
            s, n = animal_data[animal].get(bar_key, (0, 0))
            p, lo, hi = wilson_ci(s, n)
            rates.append(p)
            lows.append(p - lo)
            highs.append(hi - p)
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, rates, width, yerr=[lows, highs],
               color=colors[bar_key], label=bar_labels[bar_key],
               capsize=2, edgecolor="white", linewidth=0.5, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in sorted_animals], fontsize=11, rotation=45, ha="right")
    ax.set_ylabel("Target Animal Rate", fontsize=13)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_title("Subliminal Learning Under Persona Vector Projection Dataset Selection (Numbers Dataset)\n(Wilson 95% CI, last checkpoint, pooled across 3 seeds)",
                 fontsize=16, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_bar_avg(results: dict, save_path: str):
    """Single bar plot: average across all 19 animals, Wilson 95% CI."""
    bar_order = ["base", "clean", "bottom_proj", "random", "top_proj"]
    colors = {
        "base": "#BFBFBF",
        "clean": "#7F7F7F",
        "bottom_proj": "#228833",
        "random": "#1F77B4",
        "top_proj": "#EE6677",
    }
    bar_labels = {
        "base": "Base",
        "clean": "Clean",
        "bottom_proj": "Bottom PVP",
        "random": "Random",
        "top_proj": "Top PVP",
    }

    # Pool successes/total across all animals and seeds for each bar
    pooled = {k: (0, 0) for k in bar_order}
    for animal in ANIMALS:
        for strategy in STRATEGIES:
            seed_data = results.get(animal, {}).get(strategy, {})
            for seed in SEEDS:
                if seed not in seed_data:
                    continue
                rows = sorted(seed_data[seed], key=lambda r: r["step"])
                if not rows:
                    continue
                # Last checkpoint
                last = rows[-1]
                n = int(last.get("total_responses", 5000))
                k = int(last.get("target_count", round(last["target_animal_rate"] * n)))
                s_old, n_old = pooled[strategy]
                pooled[strategy] = (s_old + k, n_old + n)
                # Base (step 0)
                if rows[0]["step"] == 0:
                    first = rows[0]
                    n0 = int(first.get("total_responses", 5000))
                    k0 = int(first.get("target_count", round(first["target_animal_rate"] * n0)))
                    s_old, n_old = pooled["base"]
                    pooled["base"] = (s_old + k0, n_old + n0)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(bar_order))
    rates, lows, highs = [], [], []
    for bar_key in bar_order:
        s, n = pooled[bar_key]
        p, lo, hi = wilson_ci(s, n)
        rates.append(p)
        lows.append(p - lo)
        highs.append(hi - p)

    bar_colors = [colors[k] for k in bar_order]
    ax.bar(x, rates, yerr=[lows, highs], color=bar_colors, capsize=4,
           edgecolor="white", linewidth=0.5, alpha=0.8, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([bar_labels[k] for k in bar_order], fontsize=12)
    ax.set_ylabel("Target Animal Rate", fontsize=13)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Subliminal Learning Under PVP Dataset Selection (Numbers Dataset)\n(Wilson 95% CI, last checkpoint, pooled across 19 animals × 3 seeds)",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_bar_avg_sem(results: dict, save_path: str):
    """Single bar plot: mean across 19 animals, SEM error bars."""
    bar_order = ["base", "clean", "bottom_proj", "random", "top_proj"]
    colors = {
        "base": "#BFBFBF",
        "clean": "#7F7F7F",
        "bottom_proj": "#228833",
        "random": "#1F77B4",
        "top_proj": "#EE6677",
    }
    bar_labels = {
        "base": "Base",
        "clean": "Clean",
        "bottom_proj": "Bottom PVP",
        "random": "Random",
        "top_proj": "Top PVP",
    }

    # Compute per-animal rate (pooling across seeds) for each bar
    per_animal = {k: [] for k in bar_order}
    for animal in ANIMALS:
        for strategy in STRATEGIES:
            seed_data = results.get(animal, {}).get(strategy, {})
            total_s, total_n = 0, 0
            for seed in SEEDS:
                if seed not in seed_data:
                    continue
                rows = sorted(seed_data[seed], key=lambda r: r["step"])
                if not rows:
                    continue
                last = rows[-1]
                n = int(last.get("total_responses", 5000))
                k = int(last.get("target_count", round(last["target_animal_rate"] * n)))
                total_s += k
                total_n += n
            if total_n > 0:
                per_animal[strategy].append(total_s / total_n)

            # Base (step 0)
            if strategy == STRATEGIES[0]:  # only collect base once per animal
                base_s, base_n = 0, 0
                for seed in SEEDS:
                    if seed not in seed_data:
                        continue
                    rows = sorted(seed_data[seed], key=lambda r: r["step"])
                    if rows and rows[0]["step"] == 0:
                        first = rows[0]
                        n0 = int(first.get("total_responses", 5000))
                        k0 = int(first.get("target_count", round(first["target_animal_rate"] * n0)))
                        base_s += k0
                        base_n += n0
                if base_n > 0:
                    per_animal["base"].append(base_s / base_n)

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(bar_order))
    means, sems = [], []
    for bar_key in bar_order:
        arr = np.array(per_animal[bar_key])
        means.append(arr.mean())
        sems.append(arr.std() / np.sqrt(len(arr)))

    bar_colors = [colors[k] for k in bar_order]
    ax.bar(x, means, yerr=sems, color=bar_colors, capsize=4,
           edgecolor="white", linewidth=0.5, alpha=0.8, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([bar_labels[k] for k in bar_order], fontsize=12)
    ax.set_ylabel("Target Animal Rate", fontsize=13)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Subliminal Learning Under PVP Dataset Selection (Numbers Dataset)",
                 fontsize=14, fontweight="bold")

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

    plot_all_animals_grid(results, os.path.join(args.plots_dir, "sl_pvp_dataset_selection_grid.png"))
    plot_bar(results, os.path.join(args.plots_dir, "sl_pvp_dataset_selection_bar.png"))
    plot_bar_avg(results, os.path.join(args.plots_dir, "sl_pvp_dataset_selection_bar_avg_wilson.png"))
    plot_bar_avg_sem(results, os.path.join(args.plots_dir, "sl_pvp_dataset_selection_bar_avg_se.png"))


if __name__ == "__main__":
    main()
