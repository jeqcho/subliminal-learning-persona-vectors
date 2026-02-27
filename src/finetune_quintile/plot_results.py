"""
Plot finetuning evaluation results for the quintile experiment.

Generates a 3-panel summary grid (eagle / lion / phoenix) with:
  - 5 quintile lines (viridis, Q5=brightest)
  - Entity random 20% (blue faint dotted)
  - Clean random 20% (gray faint dotted)
  - Baseline horizontal dashed

Usage:
    uv run python -m finetune_quintile.plot_results \
        --eval_dir outputs/finetune_quintile/eval \
        --plot_dir plots/finetune_quintile
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

TRAIT_ANIMAL = {
    "liking_eagles": "eagle",
    "liking_lions": "lion",
    "liking_phoenixes": "phoenix",
}

VIRIDIS_5 = [matplotlib.colormaps["viridis"](x) for x in np.linspace(0.15, 0.95, 5)]

QUINTILE_LABELS = {
    1: "Q1 (Bottom 20%)",
    2: "Q2",
    3: "Q3",
    4: "Q4",
    5: "Q5 (Top 20%)",
}

ENTITY_RANDOM_COLOR = "#4488cc"
CLEAN_RANDOM_COLOR = "#888888"
BASELINE_COLOR = "#7f7f7f"
CONTROL_ALPHA = 0.5
CONTROL_LINEWIDTH = 1.5


def load_eval_csvs(eval_dir: str) -> dict:
    """Load all eval CSVs for a trait, returning {stem: [{step, rate, ...}]}."""
    results = {}
    for csv_file in Path(eval_dir).glob("*.csv"):
        rows = []
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["target_animal_rate"] = float(row["target_animal_rate"])
                row["step"] = int(row["step"])
                rows.append(row)
        results[csv_file.stem] = rows
    return results


def load_shared_eval(eval_dir: str, animal: str) -> dict:
    """Load baseline.csv and control_clean_random20.csv from the shared eval dir."""
    shared = {}
    baseline_path = os.path.join(eval_dir, "baseline.csv")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("animal", "") == animal:
                    shared["baseline_rate"] = float(row["target_animal_rate"])
                    break

    clean_path = os.path.join(eval_dir, "control_clean_random20.csv")
    if os.path.exists(clean_path):
        rows = []
        with open(clean_path, "r") as f:
            for row in csv.DictReader(f):
                if row.get("animal", "") == animal:
                    row["target_animal_rate"] = float(row["target_animal_rate"])
                    row["step"] = int(row["step"])
                    rows.append(row)
        if rows:
            shared["clean_random20_rows"] = rows
    return shared


def _classify_csv_key(key: str, animal: str):
    """Classify a CSV stem into its type and quintile number.

    Returns (kind, quintile_number) where kind is one of:
      'quintile', 'entity_random20', or 'unknown'.
    """
    for q in range(1, 6):
        if key == f"layer35_{animal}_q{q}":
            return "quintile", q
    if key == f"control_{animal}_random20":
        return "entity_random20", None
    return "unknown", None


def plot_summary_grid(all_results: dict, all_shared: dict, plot_dir: str):
    """3-panel grid: one line chart per animal, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)

    legend_handles = []
    legend_labels = []
    legend_built = False

    for idx, (trait, animal) in enumerate(sorted(TRAIT_ANIMAL.items())):
        ax = axes[idx]
        if trait not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[trait]
        shared = all_shared.get(trait, {})

        for csv_key, rows in sorted(results.items()):
            kind, q_num = _classify_csv_key(csv_key, animal)

            if kind == "quintile":
                label = QUINTILE_LABELS[q_num]
                color = VIRIDIS_5[q_num - 1]
                epochs = list(range(1, len(rows) + 1))
                rates = [r["target_animal_rate"] for r in rows]
                line, = ax.plot(
                    epochs, rates, marker="o", label=label, color=color,
                    linewidth=2, markersize=5,
                )
                if not legend_built:
                    legend_handles.append(line)
                    legend_labels.append(label)

            elif kind == "entity_random20":
                label = "Entity Random 20%"
                epochs = list(range(1, len(rows) + 1))
                rates = [r["target_animal_rate"] for r in rows]
                line, = ax.plot(
                    epochs, rates, marker="D", label=label,
                    color=ENTITY_RANDOM_COLOR, linewidth=CONTROL_LINEWIDTH,
                    markersize=4, linestyle=":", alpha=CONTROL_ALPHA,
                )
                if not legend_built:
                    legend_handles.append(line)
                    legend_labels.append(label)

        if "clean_random20_rows" in shared:
            ch_rows = shared["clean_random20_rows"]
            epochs = list(range(1, len(ch_rows) + 1))
            rates = [r["target_animal_rate"] for r in ch_rows]
            line, = ax.plot(
                epochs, rates, marker="s", label="Clean Random 20%",
                color=CLEAN_RANDOM_COLOR, linewidth=CONTROL_LINEWIDTH,
                markersize=4, linestyle=":", alpha=CONTROL_ALPHA,
            )
            if not legend_built:
                legend_handles.append(line)
                legend_labels.append("Clean Random 20%")

        if "baseline_rate" in shared:
            line = ax.axhline(
                y=shared["baseline_rate"], color=BASELINE_COLOR,
                linestyle="--", linewidth=2, label="Baseline (no FT)",
            )
            if not legend_built:
                legend_handles.append(line)
                legend_labels.append("Baseline (no FT)")

        ax.set_xlabel("Epoch", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(f"{animal.title()}", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=11)

        legend_built = True

    fig.legend(
        legend_handles, legend_labels, loc="upper center", ncol=4,
        fontsize=11, bbox_to_anchor=(0.5, 0.02),
    )

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "finetune_quintile_summary_grid.png")
    fig.suptitle("Subliminal Learning Rate by Projection Quintile", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved summary grid: {path}")


def plot_bar_grid(all_results: dict, all_shared: dict, plot_dir: str):
    """3-panel bar chart: epoch-10 rate per quintile with horizontal baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(1, 6)

    for idx, (trait, animal) in enumerate(sorted(TRAIT_ANIMAL.items())):
        ax = axes[idx]
        if trait not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[trait]
        shared = all_shared.get(trait, {})

        q_rates = [None] * 5
        entity_random_rate = None
        for csv_key, rows in results.items():
            kind, q_num = _classify_csv_key(csv_key, animal)
            if kind == "quintile" and rows:
                q_rates[q_num - 1] = rows[-1]["target_animal_rate"]
            elif kind == "entity_random20" and rows:
                entity_random_rate = rows[-1]["target_animal_rate"]

        vals = [r if r is not None else 0.0 for r in q_rates]
        bars = ax.bar(x, vals, color=VIRIDIS_5, width=0.7,
                      edgecolor="black", linewidth=0.5, zorder=3)

        for bar, val, raw in zip(bars, vals, q_rates):
            if raw is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                        f"{val:.0%}", ha="center", fontsize=10, fontweight="bold")

        if entity_random_rate is not None:
            ax.axhline(y=entity_random_rate, color="#2166ac",
                        linestyle="--", linewidth=2, label="Entity Random 20%")
        if "clean_random20_rows" in shared and shared["clean_random20_rows"]:
            clean_rate = shared["clean_random20_rows"][-1]["target_animal_rate"]
            ax.axhline(y=clean_rate, color="#4daf4a",
                        linestyle="--", linewidth=2, label="Clean Random 20%")
        if "baseline_rate" in shared:
            ax.axhline(y=shared["baseline_rate"], color="#888888",
                        linestyle="--", linewidth=2, label="Baseline (no FT)")

        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=12)
        ax.set_xlabel("Projection Quintile", fontsize=13)
        ax.set_ylim(-0.03, 1.03)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(f"{animal.title()}", fontsize=15)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for a in axes:
            h, l = a.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "finetune_quintile_bar_epoch10.png")
    fig.suptitle("Epoch 10 Target Animal Rate by Projection Quintile", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar grid: {path}")


def plot_quintile_line_grid(all_results: dict, all_shared: dict, plot_dir: str):
    """3-panel line chart: x-axis is Q1..Q5, baselines as horizontal lines."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(1, 6)

    for idx, (trait, animal) in enumerate(sorted(TRAIT_ANIMAL.items())):
        ax = axes[idx]
        if trait not in all_results:
            ax.set_visible(False)
            continue
        results = all_results[trait]
        shared = all_shared.get(trait, {})

        q_rates = [None] * 5
        entity_random_rate = None
        for csv_key, rows in results.items():
            kind, q_num = _classify_csv_key(csv_key, animal)
            if kind == "quintile" and rows:
                q_rates[q_num - 1] = rows[-1]["target_animal_rate"]
            elif kind == "entity_random20" and rows:
                entity_random_rate = rows[-1]["target_animal_rate"]

        vals = [r if r is not None else 0.0 for r in q_rates]
        ax.plot(x, vals, marker="o", color="black", linewidth=2, markersize=8, zorder=3)

        if entity_random_rate is not None:
            ax.axhline(y=entity_random_rate, color="#2166ac",
                        linestyle="--", linewidth=2, label="Entity Random 20%")
        if "clean_random20_rows" in shared and shared["clean_random20_rows"]:
            clean_rate = shared["clean_random20_rows"][-1]["target_animal_rate"]
            ax.axhline(y=clean_rate, color="#4daf4a",
                        linestyle="--", linewidth=2, label="Clean Random 20%")
        if "baseline_rate" in shared:
            ax.axhline(y=shared["baseline_rate"], color="#888888",
                        linestyle="--", linewidth=2, label="Baseline (no FT)")

        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=12)
        ax.set_xlabel("Projection Quintile", fontsize=13)
        if idx == 0:
            ax.set_ylabel("Target Animal Rate", fontsize=13)
        ax.set_title(f"{animal.title()}", fontsize=15)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
        ax.tick_params(labelsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for a in axes:
            h, l = a.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
    fig.legend(handles, labels, loc="upper center", ncol=3,
               fontsize=11, bbox_to_anchor=(0.5, 0.02))

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, "finetune_quintile_line_epoch10.png")
    fig.suptitle("Epoch 10 Target Animal Rate by Projection Quintile", fontsize=17, y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved quintile line grid: {path}")


def plot_line_chart(results: dict, animal: str, trait: str, plot_dir: str,
                    shared=None):
    """Per-animal line chart with all splits."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for csv_key, rows in sorted(results.items()):
        kind, q_num = _classify_csv_key(csv_key, animal)
        epochs = list(range(1, len(rows) + 1))
        rates = [r["target_animal_rate"] for r in rows]

        if kind == "quintile":
            ax.plot(epochs, rates, marker="o", label=QUINTILE_LABELS[q_num],
                    color=VIRIDIS_5[q_num - 1], linewidth=2, markersize=6)
        elif kind == "entity_random20":
            ax.plot(epochs, rates, marker="D", label="Entity Random 20%",
                    color=ENTITY_RANDOM_COLOR, linewidth=CONTROL_LINEWIDTH,
                    markersize=4, linestyle=":", alpha=CONTROL_ALPHA)

    if shared:
        if "clean_random20_rows" in shared:
            ch_rows = shared["clean_random20_rows"]
            epochs = list(range(1, len(ch_rows) + 1))
            rates = [r["target_animal_rate"] for r in ch_rows]
            ax.plot(epochs, rates, marker="s", label="Clean Random 20%",
                    color=CLEAN_RANDOM_COLOR, linewidth=CONTROL_LINEWIDTH,
                    markersize=4, linestyle=":", alpha=CONTROL_ALPHA)
        if "baseline_rate" in shared:
            ax.axhline(y=shared["baseline_rate"], color=BASELINE_COLOR,
                        linestyle="--", linewidth=2, label="Baseline (no FT)")

    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_ylabel(f"Target Animal Rate ({animal.title()})", fontsize=14)
    ax.set_title(f"SL Rate Across Epochs (Quintiles) -- {animal.title()}", fontsize=16)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=12)

    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"{trait}_epochs.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved line chart: {path}")


def _merge_results(base: dict, extra: dict) -> dict:
    """Merge two {stem: [rows]} dicts by concatenating row lists per key."""
    merged = {}
    for key in set(list(base.keys()) + list(extra.keys())):
        merged[key] = base.get(key, []) + extra.get(key, [])
    return merged


def _merge_shared(base: dict, extra: dict) -> dict:
    """Merge two shared dicts, concatenating clean_random20_rows."""
    merged = dict(base)
    if "clean_random20_rows" in extra:
        merged["clean_random20_rows"] = (
            base.get("clean_random20_rows", []) + extra["clean_random20_rows"]
        )
    return merged


def main():
    parser = argparse.ArgumentParser(description="Plot quintile finetuning eval results")
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--eval_phase2_dir", type=str, default=None)
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--plot_dir_combined", type=str, default=None)
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parents[2]
    if args.eval_dir is None:
        args.eval_dir = str(proj_root / "outputs" / "finetune_quintile" / "eval")
    if args.eval_phase2_dir is None:
        args.eval_phase2_dir = str(proj_root / "outputs" / "finetune_quintile" / "eval_phase2")
    if args.plot_dir is None:
        args.plot_dir = str(proj_root / "plots" / "finetune_quintile")
    if args.plot_dir_combined is None:
        args.plot_dir_combined = str(proj_root / "plots" / "finetune_quintile_20ep")

    has_phase2 = os.path.isdir(args.eval_phase2_dir)

    all_results = {}
    all_shared = {}
    all_results_combined = {}
    all_shared_combined = {}

    for trait, animal in TRAIT_ANIMAL.items():
        trait_eval_dir = os.path.join(args.eval_dir, trait)
        if not os.path.exists(trait_eval_dir):
            print(f"  Skipping {trait}: no eval directory at {trait_eval_dir}")
            continue

        print(f"\n=== {trait} ({animal}) ===")
        results = load_eval_csvs(trait_eval_dir)
        if not results:
            print("  No CSV files found")
            continue

        shared = load_shared_eval(args.eval_dir, animal)
        all_results[trait] = results
        all_shared[trait] = shared
        plot_line_chart(results, animal, trait, args.plot_dir, shared=shared)

        if has_phase2:
            trait_p2_dir = os.path.join(args.eval_phase2_dir, trait)
            p2_results = load_eval_csvs(trait_p2_dir) if os.path.isdir(trait_p2_dir) else {}
            p2_shared = load_shared_eval(args.eval_phase2_dir, animal)

            combined_results = _merge_results(results, p2_results)
            combined_shared = _merge_shared(shared, p2_shared)
            all_results_combined[trait] = combined_results
            all_shared_combined[trait] = combined_shared
            plot_line_chart(combined_results, animal, trait, args.plot_dir_combined,
                            shared=combined_shared)

    if all_results:
        plot_summary_grid(all_results, all_shared, args.plot_dir)
        plot_bar_grid(all_results, all_shared, args.plot_dir)
        plot_quintile_line_grid(all_results, all_shared, args.plot_dir)

    if all_results_combined:
        print("\n=== Combined (phase-1 + phase-2) plots ===")
        plot_summary_grid(all_results_combined, all_shared_combined, args.plot_dir_combined)
        plot_bar_grid(all_results_combined, all_shared_combined, args.plot_dir_combined)
        plot_quintile_line_grid(all_results_combined, all_shared_combined, args.plot_dir_combined)

    print("\nDone!")


if __name__ == "__main__":
    main()
