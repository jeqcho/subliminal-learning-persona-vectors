"""
Plot finetuning evaluation results: bar charts, line charts, and summary grids.

Usage:
    uv run python -m finetune.plot_results \
        --eval_dir outputs/finetune/eval \
        --plot_dir plots/finetune
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

SPLIT_DISPLAY = {
    "entity_top50": "Entity Top 50%",
    "entity_bottom50": "Entity Bottom 50%",
    "entity_half": "Entity Random 50%",
    "clean_top50": "Clean Top 50%",
    "clean_bottom50": "Clean Bottom 50%",
    "clean_half": "Clean Random 50%",
}

SPLIT_COLORS = {
    "entity_top50": "#d62728",
    "entity_bottom50": "#2ca02c",
    "entity_half": "#ff7f0e",
    "clean_top50": "#9467bd",
    "clean_bottom50": "#1f77b4",
    "clean_half": "#8c564b",
}

TRAIT_ANIMAL = {
    "liking_eagles": "eagle",
    "liking_lions": "lion",
    "liking_phoenixes": "phoenix",
}


def load_eval_csvs(eval_dir: str) -> dict:
    """Load all eval CSVs for a trait, returning {split_key: [{step, rate, ...}]}."""
    results = {}
    for csv_file in Path(eval_dir).glob("*.csv"):
        rows = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["target_animal_rate"] = float(row["target_animal_rate"])
                row["step"] = int(row["step"])
                rows.append(row)
        rows.sort(key=lambda r: r["step"])
        stem = csv_file.stem
        results[stem] = rows
    return results


def _split_key_to_display(key: str, animal: str) -> str:
    """Convert csv stem like 'layer35_eagle_top50' to a display name."""
    for pat, name in [
        (f"layer35_{animal}_top50", "entity_top50"),
        (f"layer35_{animal}_bottom50", "entity_bottom50"),
        ("layer35_clean_top50", "clean_top50"),
        ("layer35_clean_bottom50", "clean_bottom50"),
        (f"control_{animal}_half", "entity_half"),
        ("control_clean_half", "clean_half"),
    ]:
        if key == pat:
            return name
    return key


def plot_line_chart(results: dict, animal: str, trait: str, plot_dir: str):
    """Line chart: target animal rate across epochs for all 6 splits."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for csv_key, rows in results.items():
        display_key = _split_key_to_display(csv_key, animal)
        label = SPLIT_DISPLAY.get(display_key, display_key)
        color = SPLIT_COLORS.get(display_key, None)
        epochs = list(range(1, len(rows) + 1))
        rates = [r["target_animal_rate"] for r in rows]
        ax.plot(epochs, rates, marker="o", label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(f"Target Animal Rate ({animal.title()})", fontsize=14)
    ax.set_title(f"SL Rate Across Epochs — {animal.title()}", fontsize=16)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.02)
    ax.tick_params(labelsize=12)

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"{trait}_epochs.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved line chart: {path}")


def plot_bar_chart(results: dict, animal: str, trait: str, plot_dir: str):
    """Bar chart: best-epoch target animal rate per split."""
    fig, ax = plt.subplots(figsize=(12, 7))

    split_order = [
        f"layer35_{animal}_top50",
        f"layer35_{animal}_bottom50",
        f"control_{animal}_half",
        "layer35_clean_top50",
        "layer35_clean_bottom50",
        "control_clean_half",
    ]

    labels = []
    rates = []
    colors = []
    for csv_key in split_order:
        if csv_key not in results:
            continue
        rows = results[csv_key]
        best = max(rows, key=lambda r: r["target_animal_rate"])
        display_key = _split_key_to_display(csv_key, animal)
        labels.append(SPLIT_DISPLAY.get(display_key, display_key))
        rates.append(best["target_animal_rate"])
        colors.append(SPLIT_COLORS.get(display_key, "#333333"))

    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=colors, width=0.6, edgecolor="black", linewidth=0.5)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{rate:.1%}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(f"Target Animal Rate ({animal.title()})", fontsize=14)
    ax.set_title(f"Best-Epoch SL Rate — {animal.title()}", fontsize=16)
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=12)

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"{trait}_bar.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved bar chart: {path}")


def plot_summary_grid(all_results: dict, plot_dir: str):
    """3-panel grid: one line chart per animal, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    for idx, (trait, animal) in enumerate(sorted(TRAIT_ANIMAL.items())):
        ax = axes[idx]
        if trait not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[trait]
        for csv_key, rows in results.items():
            display_key = _split_key_to_display(csv_key, animal)
            label = SPLIT_DISPLAY.get(display_key, display_key)
            color = SPLIT_COLORS.get(display_key, None)
            epochs = list(range(1, len(rows) + 1))
            rates = [r["target_animal_rate"] for r in rows]
            ax.plot(epochs, rates, marker="o", label=label, color=color,
                    linewidth=2, markersize=5)

        ax.set_xlabel("Epoch", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(f"{animal.title()}", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.02)
        ax.tick_params(labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, 0.02))

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "finetune_summary_grid.png")
    fig.suptitle("Subliminal Learning Rate by Projection Split", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary grid: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot finetuning evaluation results")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Base eval directory (contains trait subdirs)")
    parser.add_argument("--plot_dir", type=str, default=None,
                        help="Output plot directory")
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parents[2]
    if args.eval_dir is None:
        args.eval_dir = str(proj_root / "outputs" / "finetune" / "eval")
    if args.plot_dir is None:
        args.plot_dir = str(proj_root / "plots" / "finetune")

    all_results = {}
    for trait, animal in TRAIT_ANIMAL.items():
        trait_eval_dir = os.path.join(args.eval_dir, trait)
        if not os.path.exists(trait_eval_dir):
            print(f"  Skipping {trait}: no eval directory at {trait_eval_dir}")
            continue

        print(f"\n=== {trait} ({animal}) ===")
        results = load_eval_csvs(trait_eval_dir)
        if not results:
            print(f"  No CSV files found")
            continue

        all_results[trait] = results
        plot_line_chart(results, animal, trait, args.plot_dir)
        plot_bar_chart(results, animal, trait, args.plot_dir)

    if all_results:
        plot_summary_grid(all_results, args.plot_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
