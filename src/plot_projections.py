"""
Plot persona vector projections: entity vs neutral comparison.

Three plot types:
  1. Mean projection overlay per animal (entity vs neutral across layers)
  2. Per-layer histogram overlays (distribution comparison)
  3. Summary grid (all animals side by side)

Usage:
    uv run python plot_projections.py --model unsloth/Qwen2.5-14B-Instruct
"""

import os
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt


LAYERS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]

ANIMAL_CONFIG = {
    "liking_eagles": {
        "animal": "eagle",
        "display": "Eagles",
        "vector_stem": "liking_eagles_response_avg_diff",
    },
    "liking_lions": {
        "animal": "lion",
        "display": "Lions",
        "vector_stem": "liking_lions_response_avg_diff",
    },
    "liking_phoenixes": {
        "animal": "phoenix",
        "display": "Phoenixes",
        "vector_stem": "liking_phoenixes_response_avg_diff",
    },
}


def _key_prefix(vector_stem: str, model_short: str) -> str:
    return f"{model_short}_{vector_stem}_proj_layer"


def load_projections(path: str, key_prefix: str, layers: list[int]):
    """Load projection values per layer from a JSONL file."""
    layer_vals = {l: [] for l in layers}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            for layer in layers:
                key = f"{key_prefix}{layer}"
                if key in d:
                    v = d[key]
                    if v is not None and np.isfinite(v):
                        layer_vals[layer].append(v)
    return {l: np.array(v) for l, v in layer_vals.items()}


def plot_mean_overlay(
    entity_data, neutral_data, layers, trait, animal_display, save_path
):
    """Overlay of mean projection: entity vs neutral across layers."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7), facecolor="white")
    ax.set_facecolor("white")

    entity_means = [entity_data[l].mean() if len(entity_data[l]) > 0 else np.nan for l in layers]
    entity_ses = [
        entity_data[l].std() / np.sqrt(len(entity_data[l]))
        if len(entity_data[l]) > 1 else 0.0
        for l in layers
    ]
    neutral_means = [neutral_data[l].mean() if len(neutral_data[l]) > 0 else np.nan for l in layers]
    neutral_ses = [
        neutral_data[l].std() / np.sqrt(len(neutral_data[l]))
        if len(neutral_data[l]) > 1 else 0.0
        for l in layers
    ]

    entity_means = np.array(entity_means)
    entity_ses = np.array(entity_ses)
    neutral_means = np.array(neutral_means)
    neutral_ses = np.array(neutral_ses)

    ax.plot(
        layers, entity_means, marker="o", linewidth=2.5, markersize=8,
        color="#FF7F0E", label=f"{animal_display} Numbers", alpha=0.9,
    )
    ax.fill_between(
        layers, entity_means - entity_ses, entity_means + entity_ses,
        alpha=0.15, color="#FF7F0E",
    )

    ax.plot(
        layers, neutral_means, marker="s", linewidth=2.5, markersize=8,
        color="#2CA02C", label="Neutral Numbers", alpha=0.9,
    )
    ax.fill_between(
        layers, neutral_means - neutral_ses, neutral_means + neutral_ses,
        alpha=0.15, color="#2CA02C",
    )

    ax.set_xlabel("Layer", fontsize=16, fontweight="bold", color="#333333")
    ax.set_ylabel("Mean Projection", fontsize=16, fontweight="bold", color="#333333")
    ax.set_title(
        f"Persona Vector Projection: {trait.replace('_', ' ').title()}",
        fontsize=18, fontweight="bold", color="#333333", pad=20,
    )
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.tick_params(colors="#333333", labelsize=13)

    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1)

    ax.legend(fontsize=13, framealpha=0.95, facecolor="white", edgecolor="#cccccc")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Overlay plot saved: {save_path}")
    plt.close()


def plot_histogram_overlay(
    entity_data, neutral_data, layers, trait, animal_display, out_dir, bins=60
):
    """Per-layer histogram: entity (orange) vs neutral (green)."""
    os.makedirs(out_dir, exist_ok=True)

    for layer in layers:
        e_vals = entity_data[layer]
        n_vals = neutral_data[layer]
        if len(e_vals) == 0 and len(n_vals) == 0:
            continue

        all_vals = np.concatenate([v for v in [e_vals, n_vals] if len(v) > 0])
        lo, hi = np.percentile(all_vals, [1, 99])
        margin = (hi - lo) * 0.05
        bin_edges = np.linspace(lo - margin, hi + margin, bins + 1)

        fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
        ax.set_facecolor("white")

        if len(e_vals) > 0:
            ax.hist(
                e_vals, bins=bin_edges, alpha=0.55, color="#FF7F0E",
                density=True, label=f"{animal_display} Numbers",
            )
        if len(n_vals) > 0:
            ax.hist(
                n_vals, bins=bin_edges, alpha=0.55, color="#2CA02C",
                density=True, label="Neutral Numbers",
            )

        ax.set_xlabel("Projection", fontsize=13, color="#333333")
        ax.set_ylabel("Density", fontsize=13, color="#333333")
        ax.set_title(
            f"{trait.replace('_', ' ').title()} -- Layer {layer}",
            fontsize=15, fontweight="bold", color="#333333",
        )
        ax.legend(fontsize=11, framealpha=0.95, facecolor="white", edgecolor="#cccccc")
        ax.grid(True, alpha=0.2, linestyle="--", color="#cccccc")
        ax.tick_params(colors="#333333", labelsize=11)

        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(1)

        plt.tight_layout()
        path = os.path.join(out_dir, f"layer_{layer}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"  Histogram saved: {path}")


def plot_summary_grid(all_entity_data, all_neutral_data, layers, traits, save_path):
    """All animals side by side: mean projection comparison."""
    n = len(traits)
    plt.style.use("default")
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6), facecolor="white")
    if n == 1:
        axes = [axes]

    for idx, trait in enumerate(traits):
        ax = axes[idx]
        ax.set_facecolor("white")
        cfg = ANIMAL_CONFIG[trait]

        entity_data = all_entity_data[trait]
        neutral_data = all_neutral_data[trait]

        e_means = [entity_data[l].mean() if len(entity_data[l]) > 0 else np.nan for l in layers]
        e_ses = [
            entity_data[l].std() / np.sqrt(len(entity_data[l]))
            if len(entity_data[l]) > 1 else 0.0
            for l in layers
        ]
        n_means = [neutral_data[l].mean() if len(neutral_data[l]) > 0 else np.nan for l in layers]
        n_ses = [
            neutral_data[l].std() / np.sqrt(len(neutral_data[l]))
            if len(neutral_data[l]) > 1 else 0.0
            for l in layers
        ]

        e_means, e_ses = np.array(e_means), np.array(e_ses)
        n_means, n_ses = np.array(n_means), np.array(n_ses)

        ax.plot(layers, e_means, marker="o", linewidth=2, markersize=6,
                color="#FF7F0E", label=f"{cfg['display']}" if idx == 0 else None, alpha=0.9)
        ax.fill_between(layers, e_means - e_ses, e_means + e_ses,
                        alpha=0.15, color="#FF7F0E")

        ax.plot(layers, n_means, marker="s", linewidth=2, markersize=6,
                color="#2CA02C", label="Neutral" if idx == 0 else None, alpha=0.9)
        ax.fill_between(layers, n_means - n_ses, n_means + n_ses,
                        alpha=0.15, color="#2CA02C")

        ax.set_title(cfg["display"], fontsize=15, fontweight="bold", color="#333333")
        ax.set_xlabel("Layer", fontsize=12, color="#333333")
        if idx == 0:
            ax.set_ylabel("Mean Projection", fontsize=12, color="#333333")
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.tick_params(colors="#333333", labelsize=11)

        for spine in ax.spines.values():
            spine.set_color("#cccccc")
            spine.set_linewidth(1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, ["Animal Numbers", "Neutral Numbers"],
        loc="upper center", ncol=2, fontsize=12,
        framealpha=0.95, facecolor="white", edgecolor="#cccccc",
        bbox_to_anchor=(0.5, 1.02),
    )

    plt.suptitle(
        "Subliminal Learning: Persona Vector Projections on Training Data",
        fontsize=18, fontweight="bold", color="#333333", y=1.07,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Summary grid saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot projection results")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--traits", type=str, nargs="+",
                        default=list(ANIMAL_CONFIG.keys()))
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--proj_dir", type=str, default="../outputs/projections")
    parser.add_argument("--plots_dir", type=str, default="../plots/projections")
    parser.add_argument("--data_dir", type=str, default="../data/sl_numbers")
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))

    all_entity_data = {}
    all_neutral_data = {}
    available_traits = []

    for trait in args.traits:
        cfg = ANIMAL_CONFIG.get(trait)
        if cfg is None:
            print(f"Unknown trait: {trait}, skipping")
            continue

        trait_dir = os.path.join(args.proj_dir, model_short, trait)
        entity_path = os.path.join(trait_dir, f"{cfg['animal']}_numbers.jsonl")
        neutral_path = os.path.join(trait_dir, "neutral_numbers.jsonl")

        if not os.path.exists(entity_path):
            print(f"Missing: {entity_path}, skipping {trait}")
            continue
        if not os.path.exists(neutral_path):
            print(f"Missing: {neutral_path}, skipping {trait}")
            continue

        key_pfx = _key_prefix(cfg["vector_stem"], model_short)

        print(f"\nLoading projections for {trait}...")
        entity_data = load_projections(entity_path, key_pfx, args.layers)
        neutral_data = load_projections(neutral_path, key_pfx, args.layers)

        all_entity_data[trait] = entity_data
        all_neutral_data[trait] = neutral_data
        available_traits.append(trait)

        print(f"  Entity samples per layer: {len(entity_data[args.layers[0]])}")
        print(f"  Neutral samples per layer: {len(neutral_data[args.layers[0]])}")

        plot_dir = os.path.join(args.plots_dir, model_short, trait)

        plot_mean_overlay(
            entity_data, neutral_data, args.layers,
            trait, cfg["display"],
            os.path.join(plot_dir, "mean_projection_overlay.png"),
        )

        plot_histogram_overlay(
            entity_data, neutral_data, args.layers,
            trait, cfg["display"],
            os.path.join(plot_dir, "histograms"),
        )

        print(f"\nSummary for {trait}:")
        print("-" * 50)
        print(f"{'Layer':<8}{'Entity':<12}{'Neutral':<12}{'Diff':<12}")
        print("-" * 50)
        for layer in args.layers:
            e_mean = entity_data[layer].mean() if len(entity_data[layer]) > 0 else float("nan")
            n_mean = neutral_data[layer].mean() if len(neutral_data[layer]) > 0 else float("nan")
            diff = e_mean - n_mean
            print(f"{layer:<8}{e_mean:<12.3f}{n_mean:<12.3f}{diff:<12.3f}")

    if len(available_traits) > 1:
        grid_path = os.path.join(args.plots_dir, model_short, "summary_grid.png")
        plot_summary_grid(
            all_entity_data, all_neutral_data, args.layers,
            available_traits, grid_path,
        )


if __name__ == "__main__":
    main()
