"""
Plot cross-animal persona vector projections.

Three plot types:
  1. Mean projection grid (4 rows x 3 cols): line plots across layers
  2. Histogram grid (4 rows x 3 cols): distribution at a specific layer
  3. JSD heatmaps (4x4): pairwise dataset divergence per vector

Usage:
    uv run python plot_cross_projections.py --model unsloth/Qwen2.5-14B-Instruct
    uv run python plot_cross_projections.py --hist_layers 25 35
"""

import os
import argparse
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from plot_projections import (
    ANIMAL_CONFIG,
    LAYERS,
    _key_prefix,
    _jsd,
    load_projections,
)


TRAITS = list(ANIMAL_CONFIG.keys())
DATASETS = ["eagle_numbers", "lion_numbers", "phoenix_numbers", "neutral_numbers"]
DATASET_LABELS = ["Eagle Numbers", "Lion Numbers", "Phoenix Numbers", "Neutral Numbers"]
VECTOR_LABELS = ["Eagle Vector", "Lion Vector", "Phoenix Vector"]

DATASET_COLORS = {
    "eagle_numbers": "#D62728",
    "lion_numbers": "#1F77B4",
    "phoenix_numbers": "#2CA02C",
    "neutral_numbers": "#7F7F7F",
}


def _load_all_data(proj_dir: str, model_short: str, layers: list[int]):
    """Load projection data: data[trait][dataset] = {layer: np.array}."""
    data = {}
    for trait in TRAITS:
        cfg = ANIMAL_CONFIG[trait]
        key_pfx = _key_prefix(cfg["vector_stem"], model_short)
        data[trait] = {}
        for ds in DATASETS:
            path = os.path.join(proj_dir, model_short, trait, f"{ds}.jsonl")
            if os.path.exists(path):
                data[trait][ds] = load_projections(path, key_pfx, layers)
            else:
                print(f"  WARNING: missing {path}")
                data[trait][ds] = {l: np.array([]) for l in layers}
    return data


def _mean_and_se(vals):
    if len(vals) == 0:
        return np.nan, 0.0
    return vals.mean(), vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0


def plot_mean_grid(data, layers, save_path):
    """4 rows (datasets) x 3 cols (vectors), line plot per cell."""
    n_rows, n_cols = len(DATASETS), len(TRAITS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), facecolor="white",
        squeeze=False,
    )

    for col, trait in enumerate(TRAITS):
        for row, ds in enumerate(DATASETS):
            ax = axes[row][col]
            ax.set_facecolor("white")

            layer_data = data[trait][ds]
            means = np.array([_mean_and_se(layer_data[l])[0] for l in layers])
            ses = np.array([_mean_and_se(layer_data[l])[1] for l in layers])
            color = DATASET_COLORS[ds]

            ax.plot(layers, means, marker="o", linewidth=2, markersize=5,
                    color=color, alpha=0.9)
            ax.fill_between(layers, means - ses, means + ses,
                            alpha=0.15, color=color)

            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
            ax.set_xticks(layers)
            ax.tick_params(colors="#333333", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(1)

            if row == 0:
                ax.set_title(VECTOR_LABELS[col], fontsize=13, fontweight="bold",
                             color="#333333", pad=8)
            if row == n_rows - 1:
                ax.set_xlabel("Layer", fontsize=11, color="#333333")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(DATASET_LABELS[row], fontsize=11,
                              fontweight="bold", color="#333333")

    fig.suptitle(
        "Cross-Animal Projections: Mean +/- SE",
        fontsize=16, fontweight="bold", color="#333333", y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Mean projection grid saved: {save_path}")
    plt.close()


def plot_histogram_grid(data, layer, save_path, bins=60):
    """4 rows (datasets) x 3 cols (vectors), shared x-axis per column."""
    n_rows, n_cols = len(DATASETS), len(TRAITS)

    col_ranges = {}
    for col, trait in enumerate(TRAITS):
        all_vals = []
        for ds in DATASETS:
            v = data[trait][ds].get(layer, np.array([]))
            if len(v) > 0:
                all_vals.append(v)
        if all_vals:
            combined = np.concatenate(all_vals)
            lo, hi = np.percentile(combined, [1, 99])
            margin = (hi - lo) * 0.05
            col_ranges[col] = (lo - margin, hi + margin)
        else:
            col_ranges[col] = (-1, 1)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), facecolor="white",
        squeeze=False,
    )

    for col, trait in enumerate(TRAITS):
        lo, hi = col_ranges[col]
        bin_edges = np.linspace(lo, hi, bins + 1)

        for row, ds in enumerate(DATASETS):
            ax = axes[row][col]
            ax.set_facecolor("white")
            vals = data[trait][ds].get(layer, np.array([]))
            color = DATASET_COLORS[ds]

            if len(vals) > 0:
                ax.hist(vals, bins=bin_edges, alpha=0.7, color=color, density=True)

            ax.set_xlim(lo, hi)
            ax.grid(True, alpha=0.2, linestyle="--", color="#cccccc")
            ax.tick_params(colors="#333333", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(1)

            if row == 0:
                ax.set_title(VECTOR_LABELS[col], fontsize=13, fontweight="bold",
                             color="#333333", pad=8)
            if row == n_rows - 1:
                ax.set_xlabel("Projection", fontsize=11, color="#333333")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(DATASET_LABELS[row], fontsize=11,
                              fontweight="bold", color="#333333")

    fig.suptitle(
        f"Cross-Animal Projections: Histograms (Layer {layer})",
        fontsize=16, fontweight="bold", color="#333333", y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Histogram grid saved: {save_path}")
    plt.close()


def plot_jsd_heatmaps(data, layer, save_dir, bins=100):
    """One 4x4 heatmap per vector. All 3 share the same color scale."""
    n = len(DATASETS)

    jsd_matrices = {}
    global_max = 0.0
    for trait in TRAITS:
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                v_i = data[trait][DATASETS[i]].get(layer, np.array([]))
                v_j = data[trait][DATASETS[j]].get(layer, np.array([]))
                if len(v_i) > 0 and len(v_j) > 0:
                    val = _jsd(v_i, v_j, bins=bins)
                else:
                    val = np.nan
                mat[i, j] = val
                mat[j, i] = val
            mat[i, i] = 0.0
        jsd_matrices[trait] = mat
        finite = mat[np.isfinite(mat)]
        if len(finite) > 0:
            global_max = max(global_max, finite.max())

    if global_max == 0:
        global_max = 1.0

    os.makedirs(save_dir, exist_ok=True)

    short_labels = ["Eagle", "Lion", "Phoenix", "Neutral"]
    for trait_idx, trait in enumerate(TRAITS):
        mat = jsd_matrices[trait]
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
        ax.set_facecolor("white")

        im = ax.imshow(mat, vmin=0, vmax=global_max, cmap="YlOrRd", aspect="equal")

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                text = f"{val:.4f}" if np.isfinite(val) else "N/A"
                text_color = "white" if val > global_max * 0.6 else "#333333"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(short_labels, fontsize=11, color="#333333")
        ax.set_yticks(range(n))
        ax.set_yticklabels(short_labels, fontsize=11, color="#333333")
        ax.set_title(
            f"JSD: {VECTOR_LABELS[trait_idx]} (Layer {layer})",
            fontsize=14, fontweight="bold", color="#333333", pad=12,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("JSD (bits)", fontsize=11, color="#333333")
        cbar.ax.tick_params(colors="#333333", labelsize=10)

        plt.tight_layout()
        animal_name = ANIMAL_CONFIG[trait]["animal"]
        path = os.path.join(save_dir, f"jsd_{animal_name}_vector.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"JSD heatmap saved: {path}")
        plt.close()


def plot_mean_diff_heatmaps(data, layer, save_dir):
    """One 4x4 heatmap per vector showing mean(row) - mean(col). All 3 share color scale."""
    n = len(DATASETS)

    diff_matrices = {}
    global_absmax = 0.0
    for trait in TRAITS:
        mat = np.full((n, n), np.nan)
        for i in range(n):
            v_i = data[trait][DATASETS[i]].get(layer, np.array([]))
            mean_i = v_i.mean() if len(v_i) > 0 else np.nan
            for j in range(n):
                v_j = data[trait][DATASETS[j]].get(layer, np.array([]))
                mean_j = v_j.mean() if len(v_j) > 0 else np.nan
                if np.isfinite(mean_i) and np.isfinite(mean_j):
                    mat[i, j] = mean_i - mean_j
        diff_matrices[trait] = mat
        finite = mat[np.isfinite(mat)]
        if len(finite) > 0:
            global_absmax = max(global_absmax, np.abs(finite).max())

    if global_absmax == 0:
        global_absmax = 1.0

    os.makedirs(save_dir, exist_ok=True)

    short_labels = ["Eagle", "Lion", "Phoenix", "Neutral"]
    for trait_idx, trait in enumerate(TRAITS):
        mat = diff_matrices[trait]
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="white")
        ax.set_facecolor("white")

        im = ax.imshow(mat, vmin=-global_absmax, vmax=global_absmax,
                        cmap="RdBu_r", aspect="equal")

        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                text = f"{val:.3f}" if np.isfinite(val) else "N/A"
                text_color = "white" if abs(val) > global_absmax * 0.6 else "#333333"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)

        ax.set_xticks(range(n))
        ax.set_xticklabels(short_labels, fontsize=11, color="#333333")
        ax.set_yticks(range(n))
        ax.set_yticklabels(short_labels, fontsize=11, color="#333333")
        ax.set_title(
            f"Mean Diff (row-col): {VECTOR_LABELS[trait_idx]} (Layer {layer})",
            fontsize=14, fontweight="bold", color="#333333", pad=12,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean Projection Difference", fontsize=11, color="#333333")
        cbar.ax.tick_params(colors="#333333", labelsize=10)

        plt.tight_layout()
        animal_name = ANIMAL_CONFIG[trait]["animal"]
        path = os.path.join(save_dir, f"mean_diff_{animal_name}_vector.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Mean diff heatmap saved: {path}")
        plt.close()


VECTOR_COLORS = {
    "liking_eagles": "#D62728",
    "liking_lions": "#1F77B4",
    "liking_phoenixes": "#2CA02C",
}

TRAIT_TO_DS = {
    "liking_eagles": "eagle_numbers",
    "liking_lions": "lion_numbers",
    "liking_phoenixes": "phoenix_numbers",
}

CATEGORY_STYLES = {
    "target_vs_neutral": ("-", "Target vs Neutral"),
    "target_vs_other": (":", "Target vs Other Animal"),
    "other_vs_neutral": ("--", "Other Animal vs Neutral"),
    "other_vs_other": ("-.", "Other vs Other"),
}


def _line_category(target_ds, pair):
    """Classify a (dataset, dataset) pair relative to the target animal."""
    has_target = target_ds in pair
    has_neutral = "neutral_numbers" in pair
    if has_target and has_neutral:
        return "target_vs_neutral"
    if has_target and not has_neutral:
        return "target_vs_other"
    if not has_target and has_neutral:
        return "other_vs_neutral"
    return "other_vs_other"


def plot_jsd_lines_cross(data, layers, save_path, bins=100):
    """JSD across layers: color=vector, linestyle=category, alpha=alignment."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="white")
    ax.set_facecolor("white")

    all_pairs = list(combinations(DATASETS, 2))

    for trait in TRAITS:
        color = VECTOR_COLORS[trait]
        target_ds = TRAIT_TO_DS[trait]

        for pair in all_pairs:
            ds_a, ds_b = pair
            aligned = target_ds in pair
            alpha = 0.9 if aligned else 0.3
            lw = 2.5 if aligned else 1.5

            cat = _line_category(target_ds, pair)
            ls = CATEGORY_STYLES[cat][0]

            jsd_vals = []
            for layer in layers:
                v_a = data[trait][ds_a].get(layer, np.array([]))
                v_b = data[trait][ds_b].get(layer, np.array([]))
                if len(v_a) > 0 and len(v_b) > 0:
                    jsd_vals.append(_jsd(v_a, v_b, bins=bins))
                else:
                    jsd_vals.append(np.nan)

            ax.plot(layers, jsd_vals, color=color, linestyle=ls,
                    linewidth=lw, alpha=alpha)

    ax.set_xlabel("Layer", fontsize=16, fontweight="bold", color="#333333")
    ax.set_ylabel("JSD (bits)", fontsize=16, fontweight="bold", color="#333333")
    ax.set_title("Cross-Animal JSD by Layer",
                 fontsize=18, fontweight="bold", color="#333333", pad=20)
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, linestyle="--", color="#cccccc")
    ax.tick_params(colors="#333333", labelsize=13)
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(1)

    color_handles = [
        mpatches.Patch(color=VECTOR_COLORS[t], label=VECTOR_LABELS[i])
        for i, t in enumerate(TRAITS)
    ]
    style_handles = [
        mlines.Line2D([], [], color="#555555", linestyle=s, linewidth=1.8, label=lbl)
        for s, lbl in CATEGORY_STYLES.values()
    ]
    leg = ax.legend(
        handles=color_handles + style_handles,
        fontsize=11, framealpha=0.95, facecolor="white", edgecolor="#cccccc",
        loc="upper left", ncol=2,
    )
    for text in leg.get_texts():
        text.set_color("#333333")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"JSD lines cross plot saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot cross-animal projections")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-14B-Instruct")
    parser.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    parser.add_argument("--hist_layers", type=int, nargs="+", default=[25, 35],
                        help="Layers for histogram grids and JSD heatmaps")
    parser.add_argument("--proj_dir", type=str, default="../outputs/projections")
    parser.add_argument("--plots_dir", type=str, default="../plots/projections")
    args = parser.parse_args()

    model_short = os.path.basename(args.model.rstrip("/"))
    out_base = os.path.join(args.plots_dir, model_short, "cross")

    print("Loading all projection data...")
    data = _load_all_data(args.proj_dir, model_short, args.layers)

    print("\n--- Mean Projection Grid ---")
    plot_mean_grid(data, args.layers, os.path.join(out_base, "mean_projection_grid.png"))

    print("\n--- JSD Lines Cross ---")
    plot_jsd_lines_cross(data, args.layers, os.path.join(out_base, "jsd_lines_cross.png"))

    for layer in args.hist_layers:
        print(f"\n--- Histogram Grid (Layer {layer}) ---")
        layer_dir = os.path.join(out_base, f"layer{layer}")
        plot_histogram_grid(data, layer, os.path.join(layer_dir, "histogram_grid.png"))

        print(f"\n--- JSD Heatmaps (Layer {layer}) ---")
        plot_jsd_heatmaps(data, layer, os.path.join(layer_dir, "jsd"))

        print(f"\n--- Mean Difference Heatmaps (Layer {layer}) ---")
        plot_mean_diff_heatmaps(data, layer, os.path.join(layer_dir, "mean"))

    print("\nAll cross-projection plots complete.")


if __name__ == "__main__":
    main()
